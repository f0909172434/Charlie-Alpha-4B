from __future__ import annotations

import copy
import gc
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl
from .stats_evolve import (
    _selected_validation_metrics,
    _start_caffeinate,
    _train_evolution_arm,
)
from .stats_training import (
    StatsDataset,
    _collate_stats_items,
    _enable_gradient_checkpointing_once,
    _selector_menu_probabilities,
    _stats_snapshot,
    stats_loss,
)


def _normalize(probabilities: list[float]) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("A policy target requires at least two candidate methods")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Policy probabilities must be finite and nonnegative")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Policy probabilities must have positive mass")
    return values / total


def reconstruct_regrets(
    oracle_probabilities: list[float],
    *,
    oracle_temperature: float,
) -> np.ndarray:
    if oracle_temperature <= 0:
        raise ValueError("The oracle temperature must be positive")
    oracle = _normalize(oracle_probabilities)
    maximum = float(oracle.max())
    regrets = np.ones_like(oracle)
    positive = oracle > 0
    regrets[positive] = -oracle_temperature * np.log(oracle[positive] / maximum)
    return np.clip(regrets, 0.0, 1.0)


def project_policy(
    parent_probabilities: list[float],
    oracle_probabilities: list[float],
    *,
    oracle_temperature: float,
    projection_temperature: float,
    step_size: float,
    exploration_mass: float,
) -> dict[str, Any]:
    if projection_temperature <= 0:
        raise ValueError("The projection temperature must be positive")
    if not 0 <= step_size <= 1:
        raise ValueError("The projection step size must be in [0, 1]")
    if not 0 <= exploration_mass < 1:
        raise ValueError("The exploration mass must be in [0, 1)")
    parent = _normalize(parent_probabilities)
    regrets = reconstruct_regrets(
        oracle_probabilities,
        oracle_temperature=oracle_temperature,
    )
    uniform = np.full_like(parent, 1.0 / len(parent))
    anchor = (1.0 - exploration_mass) * parent + exploration_mass * uniform
    logits = np.log(np.maximum(anchor, 1e-12)) - regrets / projection_temperature
    logits -= float(logits.max())
    improved = np.exp(logits)
    improved /= improved.sum()
    target = (1.0 - step_size) * parent + step_size * improved
    target /= target.sum()
    parent_expected_regret = float(np.dot(parent, regrets))
    target_expected_regret = float(np.dot(target, regrets))
    accepted = target_expected_regret <= parent_expected_regret + 1e-12
    if not accepted:
        target = parent.copy()
        target_expected_regret = parent_expected_regret
    kl_target_parent = float(
        np.sum(target * (np.log(np.maximum(target, 1e-12)) - np.log(np.maximum(parent, 1e-12))))
    )
    return {
        "target_probabilities": [float(value) for value in target],
        "reconstructed_regrets": [float(value) for value in regrets],
        "parent_expected_regret": parent_expected_regret,
        "target_expected_regret": target_expected_regret,
        "kl_target_parent": kl_target_parent,
        "accepted": accepted,
    }


def _paths(config: ProjectConfig) -> tuple[Path, Path, Path]:
    settings = config.section("policy_projection")
    cycle = int(settings["source_cycle"])
    arm = str(settings["source_arm"])
    cycle_dir = config.path_for("evolution_dir") / "cycles" / f"cycle-{cycle:04d}"
    source_dir = cycle_dir / "ablation" / arm
    output_dir = config.path_for("evolution_dir") / "policy-projection"
    artifact_dir = config.path_for("artifact_dir") / "policy-projection"
    return source_dir, output_dir, artifact_dir


def prepare_policy_projection_data(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    settings = dict(config.section("policy_projection"))
    source_dir, output_dir, _ = _paths(config)
    source_train = source_dir / "train.jsonl"
    source_valid = source_dir / "valid.jsonl"
    archive_path = config.path_for("artifact_dir") / "archive" / "index.json"
    if not source_train.exists() or not source_valid.exists() or not archive_path.exists():
        raise RuntimeError("Policy projection requires the completed source ablation and archive")
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    parent = dict(archive["champion"])
    parent_adapter = Path(str(parent["adapter_path"]))
    fingerprint = canonical_hash(
        {
            "settings": settings,
            "source_train": sha256_file(source_train),
            "source_valid": sha256_file(source_valid),
            "parent_adapter": sha256_file(parent_adapter / "adapters.safetensors"),
            "builder_version": 2,
        }
    )
    status_path = output_dir / "status.json"
    if status_path.exists() and not force:
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            return existing

    rows = list(read_jsonl(source_train))
    previous_cache_limit = mx.set_cache_limit(
        int(float(config.section("evolution")["clear_cache_threshold_gb"]) * 1024**3)
    )
    caffeinate = _start_caffeinate()
    model = tokenizer = None
    try:
        model, tokenizer = load(
            _stats_snapshot(config),
            adapter_path=str(parent_adapter),
            tokenizer_config={"trust_remote_code": True},
        )
        model.eval()
        projected_rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for row in rows:
            parent_probabilities = _selector_menu_probabilities(model, tokenizer, row)
            projection = project_policy(
                parent_probabilities,
                list(row["metadata"]["method_probabilities"]),
                oracle_temperature=float(settings["oracle_temperature"]),
                projection_temperature=float(settings["projection_temperature"]),
                step_size=float(settings["step_size"]),
                exploration_mass=float(settings["exploration_mass"]),
            )
            projected = copy.deepcopy(row)
            metadata = projected["metadata"]
            metadata["projection_parent_probabilities"] = parent_probabilities
            metadata["projection_reconstructed_regrets"] = projection["reconstructed_regrets"]
            metadata["projection_oracle_probabilities"] = list(metadata["method_probabilities"])
            metadata["method_probabilities"] = projection["target_probabilities"]
            metadata["policy_target"] = "dgp-policy-projection-v1"
            projected_rows.append(projected)
            summaries.append(projection)
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        mx.clear_cache()
        mx.set_cache_limit(previous_cache_limit)
        if caffeinate is not None and caffeinate.poll() is None:
            caffeinate.terminate()

    control_dir = output_dir / "oracle-control"
    projection_dir = output_dir / "policy-projection"
    write_jsonl(control_dir / "train.jsonl", rows)
    write_jsonl(control_dir / "valid.jsonl", list(read_jsonl(source_valid)))
    write_jsonl(projection_dir / "train.jsonl", projected_rows)
    write_jsonl(projection_dir / "valid.jsonl", list(read_jsonl(source_valid)))
    parent_regret = float(np.mean([value["parent_expected_regret"] for value in summaries]))
    target_regret = float(np.mean([value["target_expected_regret"] for value in summaries]))
    status = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "source_cycle": int(settings["source_cycle"]),
        "source_arm": str(settings["source_arm"]),
        "parent": parent,
        "records": len(rows),
        "accepted_projections": sum(bool(value["accepted"]) for value in summaries),
        "mean_parent_expected_regret": parent_regret,
        "mean_target_expected_regret": target_regret,
        "target_relative_expected_regret_improvement": (
            (parent_regret - target_regret) / parent_regret if parent_regret else 0.0
        ),
        "mean_kl_target_parent": float(np.mean([value["kl_target_parent"] for value in summaries])),
        "control_train_sha256": sha256_file(control_dir / "train.jsonl"),
        "projection_train_sha256": sha256_file(projection_dir / "train.jsonl"),
        "shared_valid_sha256": sha256_file(control_dir / "valid.jsonl"),
        "sealed_promotion_surface_opened": False,
        "sealed_final_surface_opened": False,
    }
    write_json(status_path, status)
    return status


def _arm_summary(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_checkpoint": status["checkpoint_selection"]["selected"],
        "validation": _selected_validation_metrics(status),
        "selected_validation_loss": float(status["selected_validation_loss"]),
        "elapsed_seconds": float(status["elapsed_seconds"]),
        "peak_memory_gb": float(status["peak_memory_gb"]),
        "adapter_sha256": status["adapter_sha256"],
    }


def _pilot_context(
    config: ProjectConfig,
    *,
    balanced: bool,
) -> tuple[dict[str, Any], Path, Path, dict[str, Any], dict[str, Any] | None]:
    projection_settings = dict(config.section("policy_projection"))
    settings = (
        dict(config.section("balanced_policy_projection")) if balanced else projection_settings
    )
    data_status = prepare_policy_projection_data(config, force=False)
    _, data_dir, artifact_dir = _paths(config)
    if balanced:
        artifact_dir = artifact_dir / "balanced"
    run_settings = (
        {
            key: settings[key]
            for key in (
                "curriculum",
                "microsteps",
                "validation_every",
                "checkpoint_every",
                "early_stop_evaluations",
            )
        }
        if balanced
        else None
    )
    return data_status, data_dir, artifact_dir, settings, run_settings


def run_policy_projection_arm(
    config: ProjectConfig,
    *,
    seed: int,
    arm: str,
    balanced: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    data_status, data_dir, artifact_dir, settings, run_settings = _pilot_context(
        config,
        balanced=balanced,
    )
    if seed not in {int(value) for value in settings["seeds"]}:
        raise ValueError("The requested seed is not registered for this pilot")
    if arm not in {"oracle-control", "policy-projection"}:
        raise ValueError("The pilot arm must be oracle-control or policy-projection")
    projection_settings = dict(config.section("policy_projection"))
    cycle = max(6, int(projection_settings["source_cycle"]) + 1)
    manifest = {"cycle": cycle, "parent": data_status["parent"]}
    return _train_evolution_arm(
        config,
        manifest,
        arm=arm,
        data_dir=data_dir / arm,
        candidate_dir=artifact_dir / f"seed-{seed:05d}" / arm,
        force=force,
        training_seed=seed,
        run_settings=run_settings,
    )


def run_policy_projection_pilot(
    config: ProjectConfig,
    *,
    force: bool = False,
    balanced: bool = False,
) -> dict[str, Any]:
    data_status, _, artifact_dir, settings, _ = _pilot_context(
        config,
        balanced=balanced,
    )
    seeds = [int(value) for value in settings["seeds"]]
    arm_names = ["oracle-control", "policy-projection"]
    seed_results: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds):
        order = arm_names if seed_index % 2 == 0 else list(reversed(arm_names))
        statuses: dict[str, dict[str, Any]] = {}
        for arm in order:
            command = [
                sys.executable,
                "-m",
                "charlie_alpha.cli",
                "stats",
                "policy-project-arm",
                "--config",
                str(config.path),
                "--seed",
                str(seed),
                "--arm",
                arm,
            ]
            if balanced:
                command.append("--balanced")
            if force:
                command.append("--force")
            subprocess.run(command, cwd=config.root, check=True)
            status_path = artifact_dir / f"seed-{seed:05d}" / arm / "status.json"
            statuses[arm] = json.loads(status_path.read_text(encoding="utf-8"))
        control = _arm_summary(statuses["oracle-control"])
        projected = _arm_summary(statuses["policy-projection"])
        control_metrics = control["validation"]
        projected_metrics = projected["validation"]
        seed_win = float(projected_metrics["normalized_regret"]) < float(
            control_metrics["normalized_regret"]
        ) and float(projected_metrics["invalid_selection_rate"]) <= float(
            control_metrics["invalid_selection_rate"]
        )
        seed_results.append(
            {
                "seed": seed,
                "training_order": order,
                "arms": {"oracle-control": control, "policy-projection": projected},
                "projection_win": seed_win,
            }
        )

    control_regret = float(
        np.mean(
            [
                value["arms"]["oracle-control"]["validation"]["normalized_regret"]
                for value in seed_results
            ]
        )
    )
    projection_regret = float(
        np.mean(
            [
                value["arms"]["policy-projection"]["validation"]["normalized_regret"]
                for value in seed_results
            ]
        )
    )
    control_invalidity = float(
        np.mean(
            [
                value["arms"]["oracle-control"]["validation"]["invalid_selection_rate"]
                for value in seed_results
            ]
        )
    )
    projection_invalidity = float(
        np.mean(
            [
                value["arms"]["policy-projection"]["validation"]["invalid_selection_rate"]
                for value in seed_results
            ]
        )
    )
    relative_improvement = (
        (control_regret - projection_regret) / control_regret if control_regret else 0.0
    )
    gates = {
        "seed_wins": sum(bool(value["projection_win"]) for value in seed_results)
        >= int(settings["minimum_seed_wins"]),
        "mean_relative_regret": relative_improvement
        >= float(settings["minimum_mean_relative_regret_improvement"]),
        "mean_invalidity": projection_invalidity - control_invalidity
        <= float(settings["maximum_mean_invalidity_increase"]),
        "finite_metrics": all(
            math.isfinite(value)
            for value in (
                control_regret,
                projection_regret,
                control_invalidity,
                projection_invalidity,
            )
        ),
    }
    report = {
        "schema_version": 1,
        "method": (
            "DGP Policy Projection with balanced prefixes" if balanced else "DGP Policy Projection"
        ),
        "complete": True,
        "data_fingerprint": data_status["fingerprint"],
        "training_compute_equal": True,
        "same_parent": True,
        "same_records": True,
        "same_validation": True,
        "seeds": seed_results,
        "aggregate": {
            "oracle_control_regret": control_regret,
            "policy_projection_regret": projection_regret,
            "relative_regret_improvement": relative_improvement,
            "oracle_control_invalidity": control_invalidity,
            "policy_projection_invalidity": projection_invalidity,
            "projection_seed_wins": sum(bool(value["projection_win"]) for value in seed_results),
        },
        "gates": gates,
        "proceed_to_promotion": all(gates.values()),
        "sealed_promotion_surface_opened": False,
        "sealed_final_surface_opened": False,
        "claim_boundary": (
            "This matched reusable-dev pilot tests a policy objective. It is not a sealed-final "
            "capability result or evidence of general self-improvement."
        ),
    }
    write_json(artifact_dir / "report.json", report)
    public = copy.deepcopy(report)
    for seed_result in public["seeds"]:
        for arm in seed_result["arms"].values():
            arm.pop("adapter_sha256", None)
    public_name = "policy-projection-balanced.json" if balanced else "policy-projection.json"
    write_json(config.root / "reports" / "evolve" / public_name, public)
    return report


def summarize_gradient_conflicts(
    vectors: np.ndarray,
    families: list[str],
) -> dict[str, Any]:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Gradient diagnostics require at least two vectors")
    if len(families) != values.shape[0]:
        raise ValueError("Every gradient vector requires one family label")
    norms = np.linalg.norm(values, axis=1).astype(np.float64)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("Gradient vectors must be finite and nonzero")
    unit = values / norms.astype(np.float32)[:, None]
    gram = np.clip(unit @ unit.T, -1.0, 1.0).astype(np.float64)
    upper = np.triu_indices(len(values), k=1)
    pairwise = gram[upper]
    within: list[float] = []
    cross: list[float] = []
    for left, right in zip(*upper, strict=True):
        destination = within if families[left] == families[right] else cross
        destination.append(float(gram[left, right]))

    family_indices: dict[str, list[int]] = defaultdict(list)
    for index, family in enumerate(families):
        family_indices[family].append(index)
    family_names = sorted(family_indices)
    family_vectors = np.stack(
        [values[family_indices[family]].mean(axis=0) for family in family_names]
    )
    family_norms = np.linalg.norm(family_vectors, axis=1)
    family_unit = family_vectors / family_norms[:, None]
    family_gram = np.clip(family_unit @ family_unit.T, -1.0, 1.0).astype(np.float64)
    most_conflicting = sorted(
        (
            {
                "left": family_names[left],
                "right": family_names[right],
                "cosine": float(family_gram[left, right]),
            }
            for left in range(len(family_names))
            for right in range(left + 1, len(family_names))
        ),
        key=lambda item: (float(item["cosine"]), str(item["left"]), str(item["right"])),
    )[:12]
    mean_gradient = values.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_gradient))
    mean_alignment = (values @ mean_gradient) / (norms * mean_norm)
    eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0.0)
    eigenvalue_sum = float(eigenvalues.sum())
    effective_rank = (
        eigenvalue_sum**2 / float(np.square(eigenvalues).sum())
        if np.square(eigenvalues).sum()
        else 0.0
    )

    def mean_or_none(items: list[float]) -> float | None:
        return float(np.mean(items)) if items else None

    return {
        "groups": int(values.shape[0]),
        "parameters": int(values.shape[1]),
        "families": len(family_names),
        "groups_by_family": {family: len(family_indices[family]) for family in family_names},
        "mean_gradient_norm": mean_norm,
        "mean_group_gradient_norm": float(np.mean(norms)),
        "gradient_norm_coefficient_of_variation": float(np.std(norms) / np.mean(norms)),
        "mean_pairwise_cosine": float(np.mean(pairwise)),
        "negative_pair_fraction": float(np.mean(pairwise < 0.0)),
        "mean_within_family_cosine": mean_or_none(within),
        "mean_cross_family_cosine": mean_or_none(cross),
        "negative_cross_family_fraction": (
            float(np.mean(np.asarray(cross) < 0.0)) if cross else None
        ),
        "mean_gradient_descent_coverage": float(np.mean(mean_alignment > 0.0)),
        "minimum_mean_gradient_alignment": float(np.min(mean_alignment)),
        "effective_gradient_rank": effective_rank,
        "family_order": family_names,
        "family_cosine_matrix": family_gram.tolist(),
        "most_conflicting_family_pairs": most_conflicting,
    }


def _diagnostic_groups(
    rows: list[dict[str, Any]],
    *,
    groups_per_family: int,
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["metadata"]["semantic_group_id"])].append(row)
    by_family: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for group_id, group_rows in by_group.items():
        if len(group_rows) != 4:
            raise RuntimeError(f"Diagnostic semantic group {group_id} does not contain four rows")
        family = str(group_rows[0]["metadata"]["family_id"])
        if {str(row["metadata"]["family_id"]) for row in group_rows} != {family}:
            raise RuntimeError(f"Diagnostic semantic group {group_id} crosses DGP families")
        by_family[family].append((group_id, group_rows))
    selected: list[tuple[str, str, list[dict[str, Any]]]] = []
    for family in sorted(by_family):
        available = sorted(by_family[family], key=lambda item: item[0])
        if not available:
            raise RuntimeError(f"DGP family {family} has no diagnostic groups")
        selected.extend(
            (group_id, family, group_rows) for group_id, group_rows in available[:groups_per_family]
        )
    return selected


def _gradient_conflict_objective(
    model: Any,
    tokenizer: Any,
    groups: list[tuple[str, str, list[dict[str, Any]]]],
    *,
    records_per_backward: int,
    max_seq_length: int,
) -> tuple[dict[str, Any], list[str]]:
    loss_and_grad = nn.value_and_grad(
        model,
        partial(
            stats_loss,
            component_weights={"method": 1.0, "plan_tool": 0.0, "report": 0.0},
        ),
    )
    vectors: list[np.ndarray] = []
    families: list[str] = []
    losses: list[float] = []
    parameter_names: list[str] | None = None
    for group_index, (_, family, rows) in enumerate(groups):
        dataset = StatsDataset(
            rows,
            tokenizer,
            seed=42,
            grouped=True,
            curriculum="random",
            max_seq_length=max_seq_length,
            selector_only=True,
        )
        group_vector: np.ndarray | None = None
        group_loss = 0.0
        chunks = 0
        for start in range(0, len(dataset.items), records_per_backward):
            batch = _collate_stats_items(
                dataset.items[start : start + records_per_backward],
                max_seq_length,
            )
            (loss, _), gradients = loss_and_grad(model, *batch)
            mx.eval(loss, gradients)
            flattened = tree_flatten(gradients)
            names = [name for name, _ in flattened]
            if parameter_names is None:
                parameter_names = names
            elif names != parameter_names:
                raise RuntimeError("Gradient parameter order changed during diagnostics")
            chunk_vector = np.concatenate(
                [
                    np.array(value.astype(mx.float32), copy=True).reshape(-1)
                    for _, value in flattened
                ]
            )
            group_vector = chunk_vector if group_vector is None else group_vector + chunk_vector
            group_loss += float(loss.item())
            chunks += 1
            del gradients, flattened, chunk_vector
            mx.clear_cache()
        if group_vector is None or chunks == 0:
            raise RuntimeError("Gradient diagnostic produced an empty semantic group")
        vectors.append(group_vector / chunks)
        families.append(family)
        losses.append(group_loss / chunks)
        if (group_index + 1) % 4 == 0:
            print(f"Gradient diagnostic: {group_index + 1}/{len(groups)} groups", flush=True)
    matrix = np.stack(vectors)
    summary = summarize_gradient_conflicts(matrix, families)
    summary["mean_objective_loss"] = float(np.mean(losses))
    summary["peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 4)
    del matrix, vectors
    return summary, parameter_names or []


def diagnose_policy_projection_gradients(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    settings = dict(config.section("gradient_diagnostic"))
    data_status = prepare_policy_projection_data(config, force=False)
    _, data_dir, artifact_dir = _paths(config)
    control_path = data_dir / "oracle-control" / "train.jsonl"
    projection_path = data_dir / "policy-projection" / "train.jsonl"
    fingerprint = canonical_hash(
        {
            "settings": settings,
            "parent": data_status["parent"]["adapter_sha256"],
            "control": sha256_file(control_path),
            "projection": sha256_file(projection_path),
            "diagnostic_version": 1,
        }
    )
    output_path = artifact_dir / "gradient-conflict.json"
    if output_path.exists() and not force:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            return existing

    control_rows = list(read_jsonl(control_path))
    projection_rows = list(read_jsonl(projection_path))
    control_groups = _diagnostic_groups(
        control_rows,
        groups_per_family=int(settings["groups_per_family"]),
    )
    projection_by_group = {
        group_id: (family, rows)
        for group_id, family, rows in _diagnostic_groups(
            projection_rows,
            groups_per_family=int(settings["groups_per_family"]),
        )
    }
    projection_groups = [
        (group_id, projection_by_group[group_id][0], projection_by_group[group_id][1])
        for group_id, _, _ in control_groups
    ]
    if [family for _, family, _ in control_groups] != [
        family for _, family, _ in projection_groups
    ]:
        raise RuntimeError("Control and projected gradient diagnostics are not paired")

    previous_cache_limit = mx.set_cache_limit(int(float(settings["cache_limit_gb"]) * 1024**3))
    caffeinate = _start_caffeinate()
    model = tokenizer = None
    started = time.monotonic()
    try:
        model, tokenizer = load(
            _stats_snapshot(config),
            adapter_path=str(data_status["parent"]["adapter_path"]),
            tokenizer_config={"trust_remote_code": True},
        )
        model.freeze()
        model.unfreeze(keys=["lora_a", "lora_b"])
        model.train()
        _enable_gradient_checkpointing_once(model)
        for _, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        objective_reports: dict[str, dict[str, Any]] = {}
        parameter_names: list[str] | None = None
        for objective, groups in (
            ("oracle-control", control_groups),
            ("policy-projection", projection_groups),
        ):
            report, names = _gradient_conflict_objective(
                model,
                tokenizer,
                groups,
                records_per_backward=int(settings["records_per_backward"]),
                max_seq_length=int(config.section("stats_training")["max_seq_length"]),
            )
            if parameter_names is None:
                parameter_names = names
            elif names != parameter_names:
                raise RuntimeError("Objectives expose different LoRA parameter sets")
            objective_reports[objective] = report
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        mx.clear_cache()
        mx.set_cache_limit(previous_cache_limit)
        if caffeinate is not None and caffeinate.poll() is None:
            caffeinate.terminate()

    control = objective_reports["oracle-control"]
    projected = objective_reports["policy-projection"]
    result = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "complete": True,
        "parent_adapter_sha256": data_status["parent"]["adapter_sha256"],
        "paired_group_ids_sha256": canonical_hash([group_id for group_id, _, _ in control_groups]),
        "dropout_disabled": True,
        "gradient_point": "frozen-v0.3-parent",
        "objectives": objective_reports,
        "projection_minus_control": {
            key: float(projected[key]) - float(control[key])
            for key in (
                "mean_pairwise_cosine",
                "negative_pair_fraction",
                "mean_cross_family_cosine",
                "negative_cross_family_fraction",
                "mean_gradient_descent_coverage",
                "effective_gradient_rank",
            )
            if projected[key] is not None and control[key] is not None
        },
        "trainable_parameter_tensors": len(parameter_names or []),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "sealed_promotion_surface_opened": False,
        "sealed_final_surface_opened": False,
        "claim_boundary": (
            "This diagnostic measures first-order LoRA objective geometry at the frozen parent. "
            "It does not establish that a conflict-aware optimizer will improve held-out regret."
        ),
    }
    write_json(output_path, result)
    public = copy.deepcopy(result)
    public.pop("parent_adapter_sha256", None)
    write_json(config.root / "reports" / "evolve" / "gradient-conflict.json", public)
    return result

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
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json
from .stats_evolve import (
    _adapter_config_for_child,
    _choose_checkpoint,
    _commit_cycle,
    _cycle_paths,
    _ensure_promotion_shard,
    _evaluation_rows,
    _group_regret,
    _load_archive,
    _noninferior_mapping,
    _score_adapter,
    _score_loaded_selector,
    _selector_summary,
    _start_caffeinate,
    _surface,
    evaluate_evolution_candidate,
)
from .stats_project import _diagnostic_groups, prepare_policy_projection_data
from .stats_training import (
    StatsDataset,
    _collate_stats_items,
    _enable_gradient_checkpointing_once,
    _stats_snapshot,
    stats_loss,
)


def min_norm_simplex_weights(
    gram: np.ndarray,
    *,
    max_iterations: int,
    tolerance: float,
) -> dict[str, Any]:
    if max_iterations < 1:
        raise ValueError("The common-descent solver requires at least one iteration")
    if tolerance < 0:
        raise ValueError("The common-descent solver tolerance must be nonnegative")
    matrix = np.asarray(gram, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("The common-descent Gram matrix must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("The common-descent Gram matrix must be finite")
    matrix = 0.5 * (matrix + matrix.T)
    weights = np.full(matrix.shape[0], 1.0 / matrix.shape[0], dtype=np.float64)
    converged = False
    for _iteration in range(max_iterations):
        gradient = matrix @ weights
        vertex = int(np.argmin(gradient))
        direction = -weights
        direction[vertex] += 1.0
        denominator = float(direction @ matrix @ direction)
        numerator = -float(direction @ matrix @ weights)
        step = float(np.clip(numerator / denominator, 0.0, 1.0)) if denominator > 0 else 0.0
        updated = weights + step * direction
        if float(np.linalg.norm(updated - weights, ord=1)) <= tolerance:
            weights = updated
            converged = True
            break
        weights = updated
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum()
    alignments = matrix @ weights
    return {
        "weights": weights,
        "alignments": alignments,
        "direction_norm_squared": float(weights @ matrix @ weights),
        "minimum_alignment": float(np.min(alignments)),
        "iterations": _iteration + 1,
        "converged": converged,
    }


def _cone_paths(config: ProjectConfig) -> tuple[Path, Path]:
    data_dir = config.path_for("evolution_dir") / "policy-projection" / "policy-projection"
    artifact_dir = config.path_for("artifact_dir") / "common-descent"
    return data_dir, artifact_dir


def _common_descent_training_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Keep downstream evaluation gates out of the adapter-training fingerprint."""
    return {key: value for key, value in settings.items() if key != "confirmation"}


def _group_gradient_vector(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    loss_and_grad: Any,
    records_per_backward: int,
    max_seq_length: int,
    parameter_names: list[str] | None,
) -> tuple[np.ndarray, list[str], float]:
    dataset = StatsDataset(
        rows,
        tokenizer,
        seed=42,
        grouped=True,
        curriculum="random",
        max_seq_length=max_seq_length,
        selector_only=True,
    )
    vector: np.ndarray | None = None
    loss_sum = 0.0
    chunks = 0
    names = parameter_names
    for start in range(0, len(dataset.items), records_per_backward):
        batch = _collate_stats_items(
            dataset.items[start : start + records_per_backward],
            max_seq_length,
        )
        (loss, _), gradients = loss_and_grad(model, *batch)
        mx.eval(loss, gradients)
        flattened = tree_flatten(gradients)
        current_names = [name for name, _ in flattened]
        if names is None:
            names = current_names
        elif current_names != names:
            raise RuntimeError("Common-descent gradient parameter order changed")
        chunk = np.concatenate(
            [np.array(value.astype(mx.float32), copy=True).reshape(-1) for _, value in flattened]
        )
        vector = chunk if vector is None else vector + chunk
        loss_sum += float(loss.item())
        chunks += 1
        del gradients, flattened, chunk
        mx.clear_cache()
    if vector is None or not names or chunks == 0:
        raise RuntimeError("Common-descent training produced an empty group gradient")
    return vector / chunks, names, loss_sum / chunks


def _family_gradient_matrix(
    model: Any,
    tokenizer: Any,
    groups: list[tuple[str, str, list[dict[str, Any]]]],
    *,
    records_per_backward: int,
    max_seq_length: int,
) -> tuple[np.ndarray, list[str], list[str], dict[str, Any]]:
    loss_and_grad = nn.value_and_grad(
        model,
        partial(
            stats_loss,
            component_weights={"method": 1.0, "plan_tool": 0.0, "report": 0.0},
        ),
    )
    family_sums: dict[str, np.ndarray] = {}
    family_counts: dict[str, int] = defaultdict(int)
    family_losses: dict[str, list[float]] = defaultdict(list)
    parameter_names: list[str] | None = None
    for group_index, (_, family, rows) in enumerate(groups):
        vector, parameter_names, loss = _group_gradient_vector(
            model,
            tokenizer,
            rows,
            loss_and_grad=loss_and_grad,
            records_per_backward=records_per_backward,
            max_seq_length=max_seq_length,
            parameter_names=parameter_names,
        )
        if family in family_sums:
            family_sums[family] += vector
        else:
            family_sums[family] = vector
        family_counts[family] += 1
        family_losses[family].append(loss)
        if (group_index + 1) % 8 == 0:
            print(f"Common-descent gradients: {group_index + 1}/{len(groups)} groups", flush=True)
    family_names = sorted(family_sums)
    matrix = np.stack(
        [family_sums[family] / family_counts[family] for family in family_names]
    ).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise RuntimeError("Common-descent family gradients must be finite and nonzero")
    unit = matrix / norms[:, None]
    diagnostics = {
        "groups": len(groups),
        "groups_by_family": {family: family_counts[family] for family in family_names},
        "family_gradient_norms": {
            family: float(norms[index]) for index, family in enumerate(family_names)
        },
        "family_objective_loss": {
            family: float(np.mean(family_losses[family])) for family in family_names
        },
    }
    del family_sums, matrix
    return unit, family_names, parameter_names or [], diagnostics


def _apply_flat_update(
    model: Any,
    direction: np.ndarray,
    parameter_names: list[str],
    *,
    step_l2: float,
) -> dict[str, float]:
    if step_l2 <= 0:
        raise ValueError("The common-descent step norm must be positive")
    flattened = tree_flatten(model.trainable_parameters())
    names = [name for name, _ in flattened]
    if names != parameter_names:
        raise RuntimeError("Common-descent update does not match the trainable parameters")
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("Common-descent update direction is empty")
    scaled = np.asarray(direction, dtype=np.float32) * (step_l2 / norm)
    updated: list[tuple[str, mx.array]] = []
    offset = 0
    for name, parameter in flattened:
        size = parameter.size
        delta = mx.array(scaled[offset : offset + size].reshape(parameter.shape))
        updated.append((name, parameter - delta.astype(parameter.dtype)))
        offset += size
    if offset != len(scaled):
        raise RuntimeError("Common-descent update has an unexpected parameter count")
    model.load_weights(updated, strict=False)
    mx.eval(model.trainable_parameters())
    return {"raw_direction_l2": norm, "applied_step_l2": float(np.linalg.norm(scaled))}


def run_common_descent_arm(
    config: ProjectConfig,
    *,
    arm: str,
    force: bool = False,
) -> dict[str, Any]:
    if arm not in {"uniform-family", "common-cone"}:
        raise ValueError("The common-descent arm must be uniform-family or common-cone")
    settings = dict(config.section("common_descent"))
    training_settings = _common_descent_training_settings(settings)
    data_status = prepare_policy_projection_data(config, force=False)
    data_dir, artifact_root = _cone_paths(config)
    artifact_dir = artifact_root / arm
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"
    fingerprint = canonical_hash(
        {
            "arm": arm,
            "settings": training_settings,
            "parent": data_status["parent"]["adapter_sha256"],
            "train": sha256_file(train_path),
            "valid": sha256_file(valid_path),
            "trainer_version": 1,
        }
    )
    status_path = artifact_dir / "status.json"
    progress_path = artifact_dir / "progress.json"
    if status_path.exists() and not force:
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            return existing
        raise RuntimeError(
            "The common-descent training fingerprint changed; use --force to replace its "
            "mutable arm artifacts"
        )

    resume: dict[str, Any] | None = None
    if progress_path.exists() and not force:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("fingerprint") != fingerprint:
            raise RuntimeError(
                "The common-descent partial fingerprint changed; use --force to restart the arm"
            )
        last_checkpoint = Path(str(progress.get("last_checkpoint_path", "")))
        if int(progress.get("completed_updates", 0)) > 0 and (
            not last_checkpoint.exists()
            or sha256_file(last_checkpoint) != progress.get("last_checkpoint_sha256")
        ):
            raise RuntimeError("The common-descent resume checkpoint is missing or changed")
        resume = progress

    artifact_dir.mkdir(parents=True, exist_ok=True)
    if force or resume is None:
        for path in [
            artifact_dir / "adapters.safetensors",
            artifact_dir / "adapter_config.json",
            status_path,
            progress_path,
            *artifact_dir.glob("update-*.safetensors"),
        ]:
            if path.exists():
                path.unlink()
    parent = Path(str(data_status["parent"]["adapter_path"]))
    parent_weights = parent / "adapters.safetensors"
    model = tokenizer = None
    caffeinate = _start_caffeinate()
    previous_cache_limit = mx.set_cache_limit(int(float(settings["cache_limit_gb"]) * 1024**3))
    started = time.monotonic()
    try:
        model, tokenizer = load(
            _stats_snapshot(config),
            adapter_path=str(parent),
            tokenizer_config={"trust_remote_code": True},
        )
        model.freeze()
        model.unfreeze(keys=["lora_a", "lora_b"])
        model.train()
        _enable_gradient_checkpointing_once(model)
        for _, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        train_rows = list(read_jsonl(train_path))
        groups = _diagnostic_groups(train_rows, groups_per_family=10**9)
        selection_rows = _evaluation_rows(config, "valid")
        model.eval()
        parent_result = _score_loaded_selector(model, tokenizer, selection_rows)
        parent_selector = _selector_summary(parent_result)
        checkpoints: list[dict[str, Any]] = (
            [dict(value) for value in resume["checkpoints"]]
            if resume
            else [
                {
                    "name": "parent",
                    "path": str(parent_weights),
                    "selector": parent_selector,
                }
            ]
        )
        update_history: list[dict[str, Any]] = (
            [dict(value) for value in resume["update_history"]] if resume else []
        )
        if resume and int(resume.get("completed_updates", 0)):
            model.load_weights(str(resume["last_checkpoint_path"]), strict=False)
        parameter_names: list[str] | None = None
        first_update = int(resume.get("completed_updates", 0)) + 1 if resume else 1
        for update_index in range(first_update, int(settings["updates"]) + 1):
            model.train()
            for _, module in model.named_modules():
                if isinstance(module, nn.Dropout):
                    module.eval()
            family_gradients, family_names, names, diagnostics = _family_gradient_matrix(
                model,
                tokenizer,
                groups,
                records_per_backward=int(settings["records_per_backward"]),
                max_seq_length=int(config.section("stats_training")["max_seq_length"]),
            )
            if parameter_names is None:
                parameter_names = names
            elif names != parameter_names:
                raise RuntimeError("Common-descent parameter set changed between updates")
            gram = np.clip(family_gradients @ family_gradients.T, -1.0, 1.0)
            if arm == "common-cone":
                solution = min_norm_simplex_weights(
                    gram,
                    max_iterations=int(settings["solver_max_iterations"]),
                    tolerance=float(settings["solver_tolerance"]),
                )
                weights = np.asarray(solution["weights"], dtype=np.float64)
            else:
                weights = np.full(len(family_names), 1.0 / len(family_names))
                alignments = gram @ weights
                solution = {
                    "weights": weights,
                    "alignments": alignments,
                    "direction_norm_squared": float(weights @ gram @ weights),
                    "minimum_alignment": float(np.min(alignments)),
                    "iterations": 0,
                    "converged": True,
                }
            direction = weights.astype(np.float32) @ family_gradients
            applied = _apply_flat_update(
                model,
                direction,
                parameter_names,
                step_l2=float(settings["step_l2"]),
            )
            checkpoint_path = artifact_dir / f"update-{update_index:02d}.safetensors"
            mx.save_safetensors(
                str(checkpoint_path),
                dict(tree_flatten(model.trainable_parameters())),
            )
            model.eval()
            selector_result = _score_loaded_selector(model, tokenizer, selection_rows)
            selector = _selector_summary(selector_result)
            checkpoints.append(
                {
                    "name": f"update-{update_index:02d}",
                    "path": str(checkpoint_path),
                    "selector": selector,
                }
            )
            update_history.append(
                {
                    "update": update_index,
                    "family_order": family_names,
                    "family_weights": {
                        family: float(weights[index]) for index, family in enumerate(family_names)
                    },
                    "minimum_family_alignment": float(solution["minimum_alignment"]),
                    "family_descent_coverage": float(
                        np.mean(np.asarray(solution["alignments"]) > 0.0)
                    ),
                    "direction_norm_squared": float(solution["direction_norm_squared"]),
                    "solver_iterations": int(solution["iterations"]),
                    "solver_converged": bool(solution["converged"]),
                    "gradient_diagnostics": diagnostics,
                    "update_norms": applied,
                    "selector": selector,
                }
            )
            write_json(
                progress_path,
                {
                    "schema_version": 1,
                    "fingerprint": fingerprint,
                    "complete": False,
                    "arm": arm,
                    "completed_updates": update_index,
                    "last_checkpoint_path": str(checkpoint_path),
                    "last_checkpoint_sha256": sha256_file(checkpoint_path),
                    "checkpoints": checkpoints,
                    "update_history": update_history,
                },
            )
            del family_gradients, gram, direction
            gc.collect()
            mx.clear_cache()
        selected = _choose_checkpoint(checkpoints, dict(config.section("evolution")["promotion"]))
        model.load_weights(str(selected["path"]), strict=False)
        active_path = artifact_dir / "adapters.safetensors"
        mx.save_safetensors(
            str(active_path),
            dict(tree_flatten(model.trainable_parameters())),
        )
        adapter_config = _adapter_config_for_child(
            config,
            parent,
            artifact_dir,
            cycle=max(6, int(config.section("policy_projection")["source_cycle"]) + 1),
            arm=arm,
        )
        adapter_config.setdefault("stats", {}).update(
            {
                "method": "DGP Common-Descent Cone",
                "dropout_disabled": True,
                "selected_checkpoint": selected["name"],
            }
        )
        write_json(artifact_dir / "adapter_config.json", adapter_config)
        status = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "complete": True,
            "cycle": max(6, int(config.section("policy_projection")["source_cycle"]) + 1),
            "arm": arm,
            "parent_adapter_path": str(parent),
            "parent_adapter_sha256": sha256_file(parent_weights),
            "adapter_path": str(artifact_dir),
            "adapter_sha256": sha256_file(active_path),
            "train_sha256": sha256_file(train_path),
            "valid_sha256": sha256_file(valid_path),
            "records": len(train_rows),
            "semantic_groups": len(groups),
            "updates": int(settings["updates"]),
            "full_surface_gradient_sweeps": int(settings["updates"]),
            "dropout_disabled": True,
            "selected_checkpoint": selected["name"],
            "selected_validation": selected["selector"],
            "checkpoints": checkpoints,
            "update_history": update_history,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 4),
            "promotion_shard_opened": False,
            "sealed_final_surface_opened": False,
        }
        write_json(status_path, status)
        write_json(
            progress_path,
            {
                "schema_version": 1,
                "fingerprint": fingerprint,
                "complete": True,
                "arm": arm,
                "completed_updates": int(settings["updates"]),
                "last_checkpoint_path": str(
                    artifact_dir / f"update-{int(settings['updates']):02d}.safetensors"
                ),
                "last_checkpoint_sha256": sha256_file(
                    artifact_dir / f"update-{int(settings['updates']):02d}.safetensors"
                ),
                "status_sha256": sha256_file(status_path),
                "checkpoints": checkpoints,
                "update_history": update_history,
            },
        )
        return status
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


def run_common_descent_pilot(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    settings = dict(config.section("common_descent"))
    _, artifact_dir = _cone_paths(config)
    statuses: dict[str, dict[str, Any]] = {}
    for arm in ("uniform-family", "common-cone"):
        command = [
            sys.executable,
            "-m",
            "charlie_alpha.cli",
            "stats",
            "policy-cone-arm",
            "--config",
            str(config.path),
            "--arm",
            arm,
        ]
        if force:
            command.append("--force")
        subprocess.run(command, cwd=config.root, check=True)
        statuses[arm] = json.loads((artifact_dir / arm / "status.json").read_text(encoding="utf-8"))
    control = statuses["uniform-family"]
    candidate = statuses["common-cone"]
    parent = next(row for row in candidate["checkpoints"] if row["name"] == "parent")
    control_metrics = control["selected_validation"]
    candidate_metrics = candidate["selected_validation"]
    parent_metrics = parent["selector"]
    control_regret = float(control_metrics["normalized_regret"])
    candidate_regret = float(candidate_metrics["normalized_regret"])
    parent_regret = float(parent_metrics["normalized_regret"])
    relative_over_control = (
        (control_regret - candidate_regret) / control_regret if control_regret else 0.0
    )
    relative_over_parent = (
        (parent_regret - candidate_regret) / parent_regret if parent_regret else 0.0
    )
    cone_alignments = [
        float(row["minimum_family_alignment"]) for row in candidate["update_history"]
    ]
    gates = dict(settings["gates"])
    gate_results = {
        "relative_regret_over_control": relative_over_control
        >= float(gates["minimum_relative_regret_improvement_over_control"]),
        "relative_regret_over_parent": relative_over_parent
        >= float(gates["minimum_relative_regret_improvement_over_parent"]),
        "invalidity_vs_control": float(candidate_metrics["invalid_selection_rate"])
        <= float(control_metrics["invalid_selection_rate"])
        + float(gates["maximum_invalidity_increase"]),
        "invalidity_vs_parent": float(candidate_metrics["invalid_selection_rate"])
        <= float(parent_metrics["invalid_selection_rate"])
        + float(gates["maximum_invalidity_increase"]),
        "accuracy_vs_parent": float(candidate_metrics["accuracy"])
        >= float(parent_metrics["accuracy"]) - float(gates["maximum_accuracy_regression"]),
        "common_descent": min(cone_alignments) >= float(gates["minimum_family_alignment"]),
        "nonzero_candidate": candidate["selected_checkpoint"] != "parent",
        "matched_compute": (
            control["records"] == candidate["records"]
            and control["semantic_groups"] == candidate["semantic_groups"]
            and control["updates"] == candidate["updates"]
            and control["train_sha256"] == candidate["train_sha256"]
            and control["parent_adapter_sha256"] == candidate["parent_adapter_sha256"]
        ),
    }
    report = {
        "schema_version": 1,
        "fingerprint": canonical_hash(
            {
                "settings": settings,
                "arms": {
                    name: {
                        "fingerprint": status["fingerprint"],
                        "adapter": status["adapter_sha256"],
                    }
                    for name, status in statuses.items()
                },
                "evaluator_version": 2,
            }
        ),
        "method": "DGP Common-Descent Cone",
        "complete": True,
        "deterministic_full_surface_updates": True,
        "dropout_disabled": True,
        "same_parent": True,
        "same_records": True,
        "matched_compute": gate_results["matched_compute"],
        "arms": {
            name: {
                "selected_checkpoint": status["selected_checkpoint"],
                "validation": status["selected_validation"],
                "elapsed_seconds": status["elapsed_seconds"],
                "peak_memory_gb": status["peak_memory_gb"],
                "adapter_sha256": status["adapter_sha256"],
            }
            for name, status in statuses.items()
        },
        "parent_validation": parent_metrics,
        "relative_regret_improvement_over_control": relative_over_control,
        "relative_regret_improvement_over_parent": relative_over_parent,
        "minimum_observed_family_alignment": min(cone_alignments),
        "gates": gate_results,
        "proceed_to_promotion": all(gate_results.values()),
        "sealed_promotion_surface_opened": False,
        "sealed_final_surface_opened": False,
        "claim_boundary": (
            "This matched reusable-dev pilot tests a deterministic multi-objective optimizer. "
            "It is not a sealed-final capability result or evidence of general self-improvement."
        ),
    }
    write_json(artifact_dir / "report.json", report)
    public = copy.deepcopy(report)
    for arm in public["arms"].values():
        arm.pop("adapter_sha256", None)
    write_json(config.root / "reports" / "evolve" / "common-descent.json", public)
    return report


def _mean_language_metric(scored: dict[str, Any], key: str) -> float:
    return float(np.mean([float(value[key]) for value in scored["languages"].values()]))


def _write_public_confirmation(config: ProjectConfig, result: dict[str, Any]) -> None:
    public = copy.deepcopy(result)
    for side in ("parent", "candidate"):
        public[side]["selector"].pop("predictions", None)
        for language in public[side]["languages"].values():
            language.pop("predictions", None)
        public[side]["retention"].pop("details", None)
    write_json(config.root / "reports" / "evolve" / "common-descent-confirmation.json", public)


def confirm_uniform_family_candidate(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    settings = dict(config.section("common_descent"))
    confirmation = dict(settings["confirmation"])
    _, artifact_dir = _cone_paths(config)
    pilot_path = artifact_dir / "report.json"
    candidate_status_path = artifact_dir / str(confirmation["candidate_arm"]) / "status.json"
    if not pilot_path.exists() or not candidate_status_path.exists():
        raise RuntimeError("Uniform-family confirmation requires a completed matched pilot")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    candidate_status = json.loads(candidate_status_path.read_text(encoding="utf-8"))
    if not pilot.get("complete") or not candidate_status.get("complete"):
        raise RuntimeError("Uniform-family pilot artifacts are incomplete")
    if candidate_status["selected_checkpoint"] == "parent":
        raise RuntimeError("Uniform-family confirmation requires a non-parent checkpoint")
    parent_path = Path(str(candidate_status["parent_adapter_path"]))
    candidate_path = Path(str(candidate_status["adapter_path"]))
    surface_name = str(confirmation["surface"])
    surface_path = config.path_for("stats_dir") / "surface" / f"{surface_name}.jsonl"
    fingerprint = canonical_hash(
        {
            "settings": confirmation,
            "pilot": pilot["method"],
            "parent": candidate_status["parent_adapter_sha256"],
            "candidate": candidate_status["adapter_sha256"],
            "surface": sha256_file(surface_path),
            "evaluator_version": 1,
        }
    )
    report_path = artifact_dir / "confirmation.json"
    if report_path.exists() and not force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            _write_public_confirmation(config, existing)
            return existing

    surface = _surface(config, surface_name)
    parent = _score_adapter(config, parent_path, surface)
    candidate = _score_adapter(config, candidate_path, surface)
    parent_trilingual_regret = _mean_language_metric(parent, "normalized_regret")
    candidate_trilingual_regret = _mean_language_metric(candidate, "normalized_regret")
    trilingual_relative_improvement = (
        (parent_trilingual_regret - candidate_trilingual_regret) / parent_trilingual_regret
        if parent_trilingual_regret
        else 0.0
    )
    parent_english_regret = float(parent["selector"]["normalized_regret"])
    candidate_english_regret = float(candidate["selector"]["normalized_regret"])
    english_relative_improvement = (
        (parent_english_regret - candidate_english_regret) / parent_english_regret
        if parent_english_regret
        else 0.0
    )
    parent_language_accuracy = {
        key: float(value["accuracy"]) for key, value in parent["languages"].items()
    }
    candidate_language_accuracy = {
        key: float(value["accuracy"]) for key, value in candidate["languages"].items()
    }
    parent_language_regret = {
        key: float(value["normalized_regret"]) for key, value in parent["languages"].items()
    }
    candidate_language_regret = {
        key: float(value["normalized_regret"]) for key, value in candidate["languages"].items()
    }
    parent_family_regret = _group_regret(parent["selector"]["predictions"], "family_id")
    candidate_family_regret = _group_regret(candidate["selector"]["predictions"], "family_id")
    gate_results = {
        "trilingual_regret": trilingual_relative_improvement
        >= float(confirmation["minimum_trilingual_relative_regret_improvement_over_parent"]),
        "english_regret": english_relative_improvement
        >= float(confirmation["minimum_english_relative_regret_improvement_over_parent"]),
        "accuracy": float(candidate["selector"]["accuracy"])
        >= float(parent["selector"]["accuracy"])
        - float(confirmation["maximum_accuracy_regression"]),
        "invalidity": float(candidate["selector"]["invalid_selection_rate"])
        <= float(parent["selector"]["invalid_selection_rate"])
        + float(confirmation["maximum_invalidity_increase"]),
        "retention": float(candidate["retention"]["accuracy"])
        >= float(parent["retention"]["accuracy"])
        - float(confirmation["maximum_retention_regression"]),
        "language_accuracy": _noninferior_mapping(
            parent_language_accuracy,
            candidate_language_accuracy,
            maximum_regression=float(confirmation["maximum_language_accuracy_regression"]),
            higher_is_better=True,
        ),
        "language_regret": _noninferior_mapping(
            parent_language_regret,
            candidate_language_regret,
            maximum_regression=float(confirmation["maximum_language_regret_increase"]),
            higher_is_better=False,
        ),
        "domain_accuracy": _noninferior_mapping(
            {key: float(value) for key, value in parent["selector"]["domain_accuracy"].items()},
            {key: float(value) for key, value in candidate["selector"]["domain_accuracy"].items()},
            maximum_regression=float(confirmation["maximum_domain_accuracy_regression"]),
            higher_is_better=True,
        ),
        "family_regret": _noninferior_mapping(
            parent_family_regret,
            candidate_family_regret,
            maximum_regression=float(confirmation["maximum_family_regret_increase"]),
            higher_is_better=False,
        ),
        "finite_metrics": all(
            math.isfinite(value)
            for value in (
                parent_trilingual_regret,
                candidate_trilingual_regret,
                trilingual_relative_improvement,
                english_relative_improvement,
            )
        ),
    }
    result = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "complete": True,
        "candidate_arm": str(confirmation["candidate_arm"]),
        "candidate_checkpoint": candidate_status["selected_checkpoint"],
        "selection_timing": "candidate fixed after matched valid ablation and before dev scoring",
        "surface": f"reusable-v0.3-{surface_name}",
        "surface_sha256": sha256_file(surface_path),
        "parent": parent,
        "candidate": candidate,
        "parent_trilingual_regret": parent_trilingual_regret,
        "candidate_trilingual_regret": candidate_trilingual_regret,
        "trilingual_relative_regret_improvement": trilingual_relative_improvement,
        "english_relative_regret_improvement": english_relative_improvement,
        "noninferiority_details": {
            "parent_language_accuracy": parent_language_accuracy,
            "candidate_language_accuracy": candidate_language_accuracy,
            "parent_language_regret": parent_language_regret,
            "candidate_language_regret": candidate_language_regret,
            "parent_family_regret": parent_family_regret,
            "candidate_family_regret": candidate_family_regret,
        },
        "gates": gate_results,
        "proceed_to_promotion": all(gate_results.values()),
        "sealed_promotion_surface_opened": False,
        "sealed_final_surface_opened": False,
        "claim_boundary": (
            "The dev surface is reusable and has informed earlier project decisions. This is a "
            "cross-surface confirmation, not a sealed promotion or final capability result."
        ),
    }
    write_json(report_path, result)
    _write_public_confirmation(config, result)
    return result


def promote_cone_candidate(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    _, cone_artifact_dir = _cone_paths(config)
    pilot_path = cone_artifact_dir / "report.json"
    block_path = cone_artifact_dir / "block-projection" / "report.json"
    calibration_path = cone_artifact_dir / "delta-calibration" / "report.json"
    confirmation_path = cone_artifact_dir / "confirmation.json"
    if not pilot_path.exists():
        raise RuntimeError("Cone promotion requires a completed matched pilot")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if not pilot.get("complete"):
        raise RuntimeError("Cone pilot artifacts are incomplete")
    if pilot.get("proceed_to_promotion"):
        candidate_arm = "common-cone"
        candidate_status_path = cone_artifact_dir / candidate_arm / "status.json"
        development_path = pilot_path
        development = pilot
        development_stage = "matched-valid-ablation"
    else:
        development_options = (
            (
                block_path,
                "uniform-family-block-projected",
                "valid-selection-immutable-dgp-confirmation",
            ),
            (
                calibration_path,
                "uniform-family-delta-calibrated",
                "reusable-valid-dev-delta-calibration",
            ),
            (confirmation_path, "uniform-family", "reusable-dev-confirmation"),
        )
        selected_development = False
        checked_stages: list[str] = []
        for option_path, option_arm, option_stage in development_options:
            if not option_path.exists():
                continue
            option = json.loads(option_path.read_text(encoding="utf-8"))
            if not option.get("complete"):
                raise RuntimeError(f"Development report is incomplete: {option_stage}")
            checked_stages.append(option_stage)
            if not option.get("proceed_to_promotion"):
                continue
            candidate_arm = option_arm
            development_path = option_path
            development = option
            development_stage = option_stage
            candidate_status_path = (
                Path(str(option["selected_status_path"]))
                if option.get("selected_status_path")
                else cone_artifact_dir
                / str(option.get("candidate_arm", option_arm))
                / "status.json"
            )
            selected_development = True
            break
        if not selected_development:
            return {
                "stage": "skipped",
                "reason": "all_available_development_gates_failed",
                "checked_stages": checked_stages,
                "promotion_shard_opened": False,
                "sealed_final_surface_opened": False,
            }
    if not candidate_status_path.exists():
        raise RuntimeError("The confirmed cone candidate status is missing")
    candidate = json.loads(candidate_status_path.read_text(encoding="utf-8"))
    if not candidate.get("complete") or candidate["selected_checkpoint"] == "parent":
        raise RuntimeError("The confirmed cone candidate is incomplete or unchanged")

    archive = _load_archive(config)
    champion = dict(archive["champion"])
    if champion["adapter_sha256"] != candidate["parent_adapter_sha256"]:
        raise RuntimeError("The cone candidate parent is no longer the archive champion")
    cycle = len(archive["cycles"]) + 1
    if int(candidate["cycle"]) != cycle:
        raise RuntimeError("The cone candidate does not match the next archive cycle")
    data_dir, artifact_dir = _cycle_paths(config, cycle)
    data_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    promotion = _ensure_promotion_shard(config, cycle, data_dir)
    manifest = {
        "schema_version": 1,
        "fingerprint": canonical_hash(
            {
                "cycle": cycle,
                "parent": champion["adapter_sha256"],
                "candidate": candidate["adapter_sha256"],
                "candidate_arm": candidate_arm,
                "development": development["fingerprint"],
                "promotion_shard": promotion["sha256"],
                "manifest_version": 1,
            }
        ),
        "complete": True,
        "cycle": cycle,
        "parent": champion,
        "candidate_arm": candidate_arm,
        "development_stage": development_stage,
        "development_report": str(development_path),
        "promotion_shard": promotion,
        "final_surface_opened": False,
        "artifact_dir": str(artifact_dir),
    }
    write_json(data_dir / "manifest.json", manifest)
    families = (
        candidate["update_history"][-1]["family_order"]
        if candidate.get("update_history")
        else candidate["family_order"]
    )
    training = {
        **candidate,
        "checkpoint_selection": {
            "selected": candidate["selected_checkpoint"],
            "family_learning_progress": {family: 0.5 for family in families},
        },
    }
    comparison = evaluate_evolution_candidate(
        config,
        training=training,
        cycle_manifest=manifest,
        force=force,
    )
    comparison["development_pilot"] = {
        "candidate_arm": candidate_arm,
        "stage": development_stage,
        "fingerprint": development["fingerprint"],
        "all_gates_passed": bool(development.get("proceed_to_promotion")),
    }
    write_json(artifact_dir / "comparison.json", comparison)
    status = _commit_cycle(config, comparison)
    return {
        "stage": "complete",
        "manifest": manifest,
        "comparison": comparison,
        "status": status,
    }


def promote_common_descent_candidate(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Backward-compatible alias for the gate-selected cone candidate."""
    return promote_cone_candidate(config, force=force)

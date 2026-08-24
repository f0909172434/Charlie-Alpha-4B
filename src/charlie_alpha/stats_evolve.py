from __future__ import annotations

import fcntl
import gc
import json
import math
import os
import random
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingArgs, evaluate, train

from .config import ProjectConfig
from .io_utils import (
    canonical_hash,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from .stats_catalog import FAMILY_BY_ID
from .stats_compiler import task_reward
from .stats_data import _build_record, _scenario
from .stats_dgp import Scenario, build_blueprints, simulate_scenario
from .stats_training import (
    StatsDataset,
    _enable_gradient_checkpointing_once,
    _evaluation_rows,
    _optimizer,
    _retention_score,
    _score_loaded_selector,
    _stats_snapshot,
    _StatsCallback,
    _StopTraining,
    stats_iterate_batches,
    stats_loss,
)


def _evolution_settings(config: ProjectConfig) -> dict[str, Any]:
    return config.section("evolution")


def _archive_path(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "archive" / "index.json"


@contextmanager
def _evolution_lock(config: ProjectConfig) -> Iterator[None]:
    lock_path = config.path_for("artifact_dir") / "archive" / ".iterate.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another DGP-Evolve process is already running") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _initial_parent(config: ProjectConfig) -> dict[str, Any]:
    selected_path = config.path_for("parent_artifact_dir") / "selected.json"
    if not selected_path.exists():
        raise RuntimeError("The frozen v0.3 selected adapter is missing")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    adapter_path = Path(str(selected["adapter_path"])).resolve()
    adapter_file = adapter_path / "adapters.safetensors"
    if not adapter_file.exists():
        raise RuntimeError(f"The frozen parent adapter is incomplete: {adapter_path}")
    return {
        "node_id": "v0.3.0-parent",
        "adapter_path": str(adapter_path),
        "adapter_sha256": sha256_file(adapter_file),
        "source": "frozen-v0.3.0",
    }


def _load_archive(config: ProjectConfig) -> dict[str, Any]:
    path = _archive_path(config)
    settings = _evolution_settings(config)
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if int(value.get("schema_version", 0)) != int(settings["archive_schema_version"]):
            raise RuntimeError("Unsupported DGP-Evolve archive schema")
        required_boundaries = [
            "configs/evaluation.stats.lock.json",
            "data/stats/surface/final.jsonl",
            "src/charlie_alpha/stats_sandbox.py",
            "configs/pipeline.evolve.yaml:evolution.promotion",
            "data/evolve/cycles/cycle-*/promotion.jsonl after preparation",
        ]
        boundaries = list(value.get("immutable_boundaries", []))
        changed = False
        for boundary in required_boundaries:
            if boundary not in boundaries:
                boundaries.append(boundary)
                changed = True
        if int(value.get("curriculum_signal_version", 1)) < 2:
            prior = float(settings["learning_progress_prior"])
            value["family_learning_progress"] = {
                family_id: prior for family_id in sorted(FAMILY_BY_ID)
            }
            value["curriculum_signal_version"] = 2
            changed = True
        if changed:
            value["immutable_boundaries"] = boundaries
            write_json(path, value)
        return value
    parent = _initial_parent(config)
    archive = {
        "schema_version": int(settings["archive_schema_version"]),
        "method": "DGP-Evolve",
        "champion": parent,
        "nodes": [parent],
        "cycles": [],
        "curriculum_signal_version": 2,
        "family_learning_progress": {
            family_id: float(settings["learning_progress_prior"])
            for family_id in sorted(FAMILY_BY_ID)
        },
        "immutable_boundaries": [
            "configs/evaluation.stats.lock.json",
            "data/stats/surface/final.jsonl",
            "src/charlie_alpha/stats_sandbox.py",
            "configs/pipeline.evolve.yaml:evolution.promotion",
            "data/evolve/cycles/cycle-*/promotion.jsonl after preparation",
        ],
    }
    write_json(path, archive)
    selected_path = config.path_for("artifact_dir") / "selected.json"
    write_json(selected_path, {**parent, "schema_version": 1, "complete": True})
    return archive


def evolution_status(config: ProjectConfig) -> dict[str, Any]:
    archive = _load_archive(config)
    return {
        "schema_version": archive["schema_version"],
        "method": archive["method"],
        "champion": archive["champion"],
        "completed_cycles": len(archive["cycles"]),
        "last_cycle": archive["cycles"][-1] if archive["cycles"] else None,
        "family_learning_progress": archive["family_learning_progress"],
        "immutable_boundaries": archive["immutable_boundaries"],
    }


def _surface(config: ProjectConfig, split: str) -> list[dict[str, Any]]:
    path = config.path_for("stats_dir") / "surface" / f"{split}.jsonl"
    rows = list(read_jsonl(path))
    if not rows:
        raise RuntimeError(f"The frozen DGP surface is missing: {path}")
    return rows


def _normalized_distance(left: Scenario, right: Scenario) -> float:
    if left.family_id != right.family_id:
        return 1.0
    family = FAMILY_BY_ID[left.family_id]
    squared = 0.0
    count = 0
    for key, bounds in family.parameters.items():
        span = float(bounds[1] - bounds[0])
        if span <= 0:
            continue
        squared += ((float(left.parameters[key]) - float(right.parameters[key])) / span) ** 2
        count += 1
    return math.sqrt(squared / max(1, count))


def _novelty(scenario: Scenario, references: list[Scenario]) -> float:
    same_family = [item for item in references if item.family_id == scenario.family_id]
    if not same_family:
        return 1.0
    distance = min(_normalized_distance(scenario, item) for item in same_family)
    return min(1.0, 3.0 * distance)


def _mutate_scenario(parent: Scenario, *, cycle: int, index: int, seed: int) -> Scenario:
    family = FAMILY_BY_ID[parent.family_id]
    rng = np.random.default_rng(seed)
    parameters = dict(parent.parameters)
    keys = list(family.parameters)
    mutation_count = 1 + int(rng.random() < 0.35)
    mutated_keys = list(rng.choice(keys, size=mutation_count, replace=False))
    for key in mutated_keys:
        lower, upper = family.parameters[key]
        span = float(upper - lower)
        direction = -1.0 if rng.random() < 0.5 else 1.0
        magnitude = float(rng.uniform(0.04, 0.18)) * span
        value = min(max(float(parameters[key]) + direction * magnitude, lower), upper)
        if key in {"n", "clusters", "cluster_size", "horizon"}:
            value = float(max(2, round(value)))
        parameters[key] = value
    payload = {
        "parent": parent.blueprint_id,
        "cycle": cycle,
        "index": index,
        "parameters": parameters,
        "seed": seed,
    }
    return Scenario(
        blueprint_id=f"evo-{canonical_hash(payload)[:20]}",
        family_id=parent.family_id,
        split=f"evolve-cycle-{cycle:04d}",
        seed=seed,
        parameters=parameters,
        boundary_round=2,
        domain=parent.domain,
        search={
            "criterion": "verified-failure-frontier",
            "parent_blueprint_id": parent.blueprint_id,
            "mutated_keys": mutated_keys,
            "candidate_count": 1,
        },
    )


def _simulate(config: ProjectConfig, scenario: Scenario) -> dict[str, Any]:
    settings = config.section("stats_data")
    return simulate_scenario(
        scenario,
        initial_repetitions=int(settings["initial_repetitions"]),
        escalation_repetitions=[int(value) for value in settings["escalation_repetitions"]],
        uncertainty_margin=float(settings["ranking_uncertainty_margin"]),
        temperature=float(settings["regret_temperature"]),
    )


def _proposal_records(
    simulations: list[dict[str, Any]],
    *,
    language: str = "en",
    view: str = "boundary_a",
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for simulation in simulations:
        scenario = _scenario(simulation["scenario"])
        record = _build_record(
            scenario,
            simulation,
            language=language,
            loss_weight=1.0,
            incomplete=False,
            variant="dgp-regret",
            refined_explanation=None,
            view=view,
        )
        rows.append((record, simulation))
    return rows


def _score_proposals(
    config: ProjectConfig,
    adapter_path: Path,
    simulations: list[dict[str, Any]],
) -> dict[str, Any]:
    model, tokenizer = load(
        _stats_snapshot(config),
        adapter_path=str(adapter_path),
        tokenizer_config={"trust_remote_code": True},
    )
    result = _score_loaded_selector(model, tokenizer, _proposal_records(simulations))
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    return result


def _select_diverse(proposals: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ranked = sorted(
        proposals,
        key=lambda item: (-float(item["task_reward"]), str(item["blueprint_id"])),
    )
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in ranked:
        by_family[str(item["family_id"])].append(item)
    selected: list[dict[str, Any]] = []
    for family_id in sorted(by_family):
        if by_family[family_id] and len(selected) < count:
            selected.append(by_family[family_id].pop(0))
    used = {str(item["blueprint_id"]) for item in selected}
    selected.extend(item for item in ranked if str(item["blueprint_id"]) not in used)
    return selected[:count]


def _cycle_paths(config: ProjectConfig, cycle: int) -> tuple[Path, Path]:
    data_dir = config.path_for("evolution_dir") / "cycles" / f"cycle-{cycle:04d}"
    artifact_dir = config.path_for("artifact_dir") / "archive" / f"cycle-{cycle:04d}"
    return data_dir, artifact_dir


def _promotion_scenarios(config: ProjectConfig, cycle: int) -> list[Scenario]:
    settings = _evolution_settings(config)["promotion_shard"]
    split = f"evolve-promotion-{cycle:04d}"
    seed = int(settings["seed_base"]) + cycle * 1_000_003
    return build_blueprints(
        {split: int(settings["count"])},
        seed=seed,
        active_search=False,
    )


def _ensure_promotion_shard(
    config: ProjectConfig,
    cycle: int,
    data_dir: Path,
) -> dict[str, Any]:
    settings = _evolution_settings(config)["promotion_shard"]
    scenarios = _promotion_scenarios(config, cycle)
    path = data_dir / "promotion.jsonl"
    manifest_path = data_dir / "promotion_manifest.json"
    fingerprint = canonical_hash(
        {
            "cycle": cycle,
            "scenarios": [scenario.to_dict() for scenario in scenarios],
            "simulation": {
                key: config.section("stats_data")[key]
                for key in (
                    "initial_repetitions",
                    "escalation_repetitions",
                    "ranking_uncertainty_margin",
                    "regret_temperature",
                )
            },
            "generator_version": 1,
        }
    )
    if path.exists() or manifest_path.exists():
        if not path.exists() or not manifest_path.exists():
            raise RuntimeError("The immutable promotion shard is incomplete")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("fingerprint") == fingerprint
            and existing.get("sha256") == sha256_file(path)
            and int(existing.get("count", 0)) == len(scenarios)
        ):
            return existing
        raise RuntimeError(
            "The prepared promotion shard is immutable; restore its frozen settings or start "
            "a new cycle"
        )
    simulations = [_simulate(config, scenario) for scenario in scenarios]
    write_jsonl(path, simulations)
    manifest = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "cycle": cycle,
        "split": scenarios[0].split,
        "seed": int(settings["seed_base"]) + cycle * 1_000_003,
        "count": len(simulations),
        "sha256": sha256_file(path),
        "used_for_task_selection": False,
        "used_for_training": False,
        "single_use": True,
        "sealed_at_preparation": True,
        "final_surface_opened": False,
    }
    write_json(manifest_path, manifest)
    return manifest


def _existing_references(config: ProjectConfig) -> list[Scenario]:
    references: list[Scenario] = []
    for split in ("train", "valid", "dev"):
        references.extend(_scenario(row["scenario"]) for row in _surface(config, split))
    evolution_root = config.path_for("evolution_dir") / "cycles"
    if evolution_root.exists():
        for path in sorted(evolution_root.glob("cycle-*/selected.jsonl")):
            references.extend(_scenario(row["simulation"]["scenario"]) for row in read_jsonl(path))
    return references


def _write_training_records(
    config: ProjectConfig,
    cycle: int,
    selected: list[dict[str, Any]],
    data_dir: Path,
) -> dict[str, Any]:
    settings = _evolution_settings(config)
    selected = selected[: int(settings["train_groups_per_cycle"])]
    new_records: list[dict[str, Any]] = []
    language_views = (
        ("en", 1.4, "boundary_a"),
        ("en", 1.4, "boundary_b"),
        ("zh_Hant", 0.6, "standard"),
        ("zh_Hans", 0.6, "standard"),
    )
    for item in selected:
        simulation = dict(item["simulation"])
        scenario = _scenario(simulation["scenario"])
        for language, weight, view in language_views:
            new_records.append(
                _build_record(
                    scenario,
                    simulation,
                    language=language,
                    loss_weight=weight,
                    incomplete=False,
                    variant="dgp-regret",
                    refined_explanation=None,
                    view=view,
                )
            )
            new_records[-1]["metadata"]["evolution_source"] = "new"

    replay_path = config.path_for("final_dir") / "dgp-regret" / "train.jsonl"
    replay_rows = list(read_jsonl(replay_path))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in replay_rows:
        grouped[str(row["metadata"]["semantic_group_id"])].append(row)
    replay_groups = round(
        len(selected)
        * float(settings["replay_fraction"])
        / max(1e-9, 1.0 - float(settings["replay_fraction"]))
    )
    rng = random.Random(int(config.section("project")["seed"]) + cycle * 10_003)
    group_ids = sorted(grouped)
    rng.shuffle(group_ids)
    replay_records = []
    for group_id in group_ids[:replay_groups]:
        for row in grouped[group_id]:
            copied = {**row, "metadata": {**row["metadata"], "evolution_source": "replay"}}
            replay_records.append(copied)
    train_rows = [*new_records, *replay_records]
    rng.shuffle(train_rows)

    valid_rows = list(read_jsonl(config.path_for("final_dir") / "dgp-regret" / "valid.jsonl"))
    rng.shuffle(valid_rows)
    valid_rows = valid_rows[: int(settings["validation_records"])]
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, valid_rows)
    return {
        "new_groups": len(selected),
        "replay_groups": replay_groups,
        "train_records": len(train_rows),
        "valid_records": len(valid_rows),
        "train_sha256": sha256_file(train_path),
        "valid_sha256": sha256_file(valid_path),
        "selected_blueprints_sha256": canonical_hash(
            [str(item["blueprint_id"]) for item in selected]
        ),
        "language_gradient_mass": {"en": 0.70, "zh_Hant": 0.15, "zh_Hans": 0.15},
    }


def _training_records_are_current(
    config: ProjectConfig,
    data_dir: Path,
    data: dict[str, Any],
) -> bool:
    settings = _evolution_settings(config)
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"
    new_groups = int(settings["train_groups_per_cycle"])
    replay_groups = round(
        new_groups
        * float(settings["replay_fraction"])
        / max(1e-9, 1.0 - float(settings["replay_fraction"]))
    )
    return bool(
        int(data.get("new_groups", -1)) == new_groups
        and int(data.get("replay_groups", -1)) == replay_groups
        and int(data.get("train_records", -1)) == 4 * (new_groups + replay_groups)
        and train_path.exists()
        and valid_path.exists()
        and data.get("train_sha256") == sha256_file(train_path)
        and data.get("valid_sha256") == sha256_file(valid_path)
    )


def prepare_evolution_cycle(config: ProjectConfig, *, force: bool = False) -> dict[str, Any]:
    archive = _load_archive(config)
    cycle = len(archive["cycles"]) + 1
    data_dir, artifact_dir = _cycle_paths(config, cycle)
    manifest_path = data_dir / "manifest.json"
    champion = dict(archive["champion"])
    settings = _evolution_settings(config)
    # Promotion settings do not change the already prepared task pool. Keeping
    # the fingerprints separate also lets older prepared cycles gain a fresh
    # promotion shard without regenerating their training records.
    proposal_settings = {
        key: value
        for key, value in settings.items()
        if key not in {"promotion_shard", "train_groups_per_cycle"}
    }
    fingerprint = canonical_hash(
        {
            "cycle": cycle,
            "champion": champion["adapter_sha256"],
            "settings": proposal_settings,
            "dev_surface": sha256_file(config.path_for("stats_dir") / "surface" / "dev.jsonl"),
            "generator_version": 1,
        }
    )
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint and existing.get("complete"):
            if not _training_records_are_current(
                config,
                data_dir,
                dict(existing.get("data", {})),
            ):
                selected = list(read_jsonl(data_dir / "selected.jsonl"))
                existing["data"] = _write_training_records(
                    config,
                    cycle,
                    selected,
                    data_dir,
                )
            promotion = _ensure_promotion_shard(
                config,
                cycle,
                data_dir,
            )
            if existing.get("promotion_shard") != promotion:
                existing["promotion_shard"] = promotion
                write_json(manifest_path, existing)
            return existing

    data_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    parent_adapter = Path(champion["adapter_path"])
    dev_surface = _surface(config, "dev")
    parent_dev = _score_proposals(config, parent_adapter, dev_surface)
    simulation_by_id = {str(row["scenario"]["blueprint_id"]): row for row in dev_surface}
    ranked_failures = sorted(
        parent_dev["predictions"],
        key=lambda item: (-float(item["normalized_regret"]), str(item["blueprint_id"])),
    )
    references = _existing_references(config)
    pool_limit = int(settings["proposal_pool"])
    mutations_per_parent = int(settings["mutations_per_parent"])
    proposals_simulations: list[dict[str, Any]] = []
    parents: list[str] = []
    index = 0
    while len(proposals_simulations) < pool_limit:
        failure = ranked_failures[index % len(ranked_failures)]
        parent_simulation = simulation_by_id[str(failure["blueprint_id"])]
        parent = _scenario(parent_simulation["scenario"])
        mutation_index = index % mutations_per_parent
        seed = (
            int(config.section("project")["seed"])
            + cycle * 1_000_003
            + index * 101
            + mutation_index
        )
        scenario = _mutate_scenario(parent, cycle=cycle, index=index, seed=seed)
        proposals_simulations.append(_simulate(config, scenario))
        parents.append(parent.blueprint_id)
        index += 1

    proposal_scores = _score_proposals(config, parent_adapter, proposals_simulations)
    prediction_by_id = {str(row["blueprint_id"]): row for row in proposal_scores["predictions"]}
    learning_progress = dict(archive.get("family_learning_progress", {}))
    proposals: list[dict[str, Any]] = []
    target = float(settings["frontier_target_regret"])
    sigma = float(settings["frontier_sigma"])
    for parent_id, simulation in zip(parents, proposals_simulations, strict=True):
        scenario = _scenario(simulation["scenario"])
        prediction = prediction_by_id[scenario.blueprint_id]
        regret = float(prediction["normalized_regret"])
        validity = float(
            bool(simulation["valid_method_ids"])
            and math.isclose(
                sum(float(row["soft_target"]) for row in simulation["candidates"]),
                1.0,
                abs_tol=1e-7,
            )
        )
        novelty = _novelty(scenario, references)
        frontier = math.exp(-((regret - target) ** 2) / (2 * sigma**2))
        progress = float(
            learning_progress.get(scenario.family_id, settings["learning_progress_prior"])
        )
        reward = task_reward(
            validity=validity,
            novelty=novelty,
            frontier=frontier,
            learning_progress=progress,
        )
        proposals.append(
            {
                "blueprint_id": scenario.blueprint_id,
                "family_id": scenario.family_id,
                "parent_blueprint_id": parent_id,
                "parent_normalized_regret": regret,
                "validity": validity,
                "novelty": novelty,
                "frontier": frontier,
                "learning_progress": progress,
                "task_reward": reward,
                "simulation": simulation,
            }
        )
    eligible = [
        item for item in proposals if float(item["novelty"]) >= float(settings["novelty_floor"])
    ]
    selected = _select_diverse(eligible, int(settings["tasks_per_cycle"]))
    if len(selected) < int(settings["tasks_per_cycle"]):
        raise RuntimeError(f"Only {len(selected)} valid novel tasks were available for evolution")
    write_jsonl(data_dir / "proposals.jsonl", proposals)
    write_jsonl(data_dir / "selected.jsonl", selected)
    data = _write_training_records(config, cycle, selected, data_dir)
    promotion = _ensure_promotion_shard(config, cycle, data_dir)
    selected_ids = {str(item["blueprint_id"]) for item in selected}
    promotion_ids = {
        str(item["scenario"]["blueprint_id"])
        for item in read_jsonl(data_dir / "promotion.jsonl")
    }
    if selected_ids & promotion_ids:
        raise RuntimeError("Evolution task selection overlaps the promotion shard")
    manifest = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "complete": True,
        "cycle": cycle,
        "parent": champion,
        "proposal_count": len(proposals),
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "selection_formula": "validity * novelty * frontier * learning_progress",
        "task_reward_summary": {
            "minimum": min(float(item["task_reward"]) for item in selected),
            "mean": float(np.mean([float(item["task_reward"]) for item in selected])),
            "maximum": max(float(item["task_reward"]) for item in selected),
        },
        "data": data,
        "promotion_shard": promotion,
        "final_surface_opened": False,
        "artifact_dir": str(artifact_dir),
    }
    write_json(manifest_path, manifest)
    return manifest


def _adapter_config_for_child(
    config: ProjectConfig,
    parent: Path,
    destination: Path,
    *,
    cycle: int,
) -> dict[str, Any]:
    value = json.loads((parent / "adapter_config.json").read_text(encoding="utf-8"))
    evolution = _evolution_settings(config)
    value["adapter_path"] = str(destination)
    value.setdefault("stats", {}).update(
        {
            "method": "DGP-Evolve",
            "cycle": cycle,
            "parent_adapter_sha256": sha256_file(parent / "adapters.safetensors"),
            "learning_rate_a": float(evolution["learning_rate_a"]),
            "learning_rate_b": float(evolution["learning_rate_b"]),
            "generated_by_gated_evolution_loop": True,
            "promotion_status": "candidate",
        }
    )
    return value


def _start_caffeinate() -> subprocess.Popen[bytes] | None:
    executable = Path("/usr/bin/caffeinate")
    if not executable.exists():
        return None
    return subprocess.Popen(
        [str(executable), "-dimsu", "-w", str(os.getpid())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _selector_summary(result: dict[str, Any]) -> dict[str, float]:
    return {
        "normalized_regret": float(result["normalized_regret"]),
        "accuracy": float(result["accuracy"]),
        "invalid_selection_rate": float(result["invalid_selection_rate"]),
    }


def _family_learning_signal(
    parent_predictions: list[dict[str, Any]],
    candidate_predictions: list[dict[str, Any]],
) -> dict[str, float]:
    """Estimate learnability on reusable validation data without promotion feedback."""
    parent = _group_regret(parent_predictions, "family_id")
    candidate = _group_regret(candidate_predictions, "family_id")
    if set(parent) != set(candidate):
        raise RuntimeError("Validation family coverage changed between checkpoints")
    signal: dict[str, float] = {}
    for family_id in parent:
        denominator = max(0.05, parent[family_id])
        relative_improvement = (parent[family_id] - candidate[family_id]) / denominator
        signal[family_id] = 0.5 + 0.25 * math.tanh(relative_improvement / 0.25)
    return signal


def _choose_checkpoint(
    candidates: list[dict[str, Any]],
    promotion_gates: dict[str, Any],
) -> dict[str, Any]:
    parent = next(item for item in candidates if item["name"] == "parent")
    parent_metrics = parent["selector"]
    minimum_improvement = float(promotion_gates["minimum_relative_regret_improvement"])
    maximum_accuracy_regression = float(promotion_gates["maximum_accuracy_regression"])
    eligible = []
    for item in candidates:
        if item["name"] == "parent":
            continue
        metrics = item["selector"]
        relative = (
            (parent_metrics["normalized_regret"] - metrics["normalized_regret"])
            / parent_metrics["normalized_regret"]
            if parent_metrics["normalized_regret"]
            else 0.0
        )
        if (
            relative >= minimum_improvement
            and metrics["invalid_selection_rate"]
            <= parent_metrics["invalid_selection_rate"]
            and metrics["accuracy"]
            >= parent_metrics["accuracy"] - maximum_accuracy_regression
        ):
            eligible.append({**item, "relative_regret_improvement": relative})
    if not eligible:
        return {**parent, "relative_regret_improvement": 0.0}
    return min(
        eligible,
        key=lambda item: (
            float(item["selector"]["invalid_selection_rate"]),
            float(item["selector"]["normalized_regret"]),
            -float(item["selector"]["accuracy"]),
            str(item["name"]),
        ),
    )


def _weight_file_max_abs_delta(left: Path, right: Path) -> float:
    left_weights = mx.load(str(left))
    right_weights = mx.load(str(right))
    if set(left_weights) != set(right_weights):
        return math.inf
    maximum = 0.0
    for key in left_weights:
        if left_weights[key].shape != right_weights[key].shape:
            return math.inf
        maximum = max(
            maximum,
            float(mx.max(mx.abs(left_weights[key] - right_weights[key])).item()),
        )
    return maximum


def train_evolution_candidate(config: ProjectConfig, *, force: bool = False) -> dict[str, Any]:
    manifest = prepare_evolution_cycle(config, force=False)
    cycle = int(manifest["cycle"])
    data_dir, artifact_dir = _cycle_paths(config, cycle)
    candidate_dir = artifact_dir / "candidate"
    status_path = candidate_dir / "status.json"
    parent = Path(str(manifest["parent"]["adapter_path"]))
    settings = _evolution_settings(config)
    training_settings = config.section("stats_training")
    fingerprint = canonical_hash(
        {
            "cycle": cycle,
            "parent": sha256_file(parent / "adapters.safetensors"),
            "train": sha256_file(data_dir / "train.jsonl"),
            "valid": sha256_file(data_dir / "valid.jsonl"),
            "settings": settings,
            "trainer_version": 3,
        }
    )
    if status_path.exists() and not force:
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint and existing.get("complete"):
            return existing

    candidate_dir.mkdir(parents=True, exist_ok=True)
    managed_checkpoints = [
        candidate_dir / "adapters.safetensors",
        candidate_dir / "last_adapters.safetensors",
        candidate_dir / "best_adapters.safetensors",
        candidate_dir / "adapter_config.json",
        status_path,
        *candidate_dir.glob("[0-9]*_adapters.safetensors"),
    ]
    for path in managed_checkpoints:
        if path.exists():
            path.unlink()
    seed = int(config.section("project")["seed"]) + cycle
    mx.random.seed(seed)
    np.random.seed(seed)
    model_path = _stats_snapshot(config)
    model, tokenizer = load(
        model_path,
        adapter_path=str(parent),
        tokenizer_config={"trust_remote_code": True},
    )
    model.freeze()
    model.unfreeze(keys=["lora_a", "lora_b"])
    selection_rows = _evaluation_rows(config, "valid")
    model.eval()
    parent_selector_result = _score_loaded_selector(model, tokenizer, selection_rows)
    parent_selector = _selector_summary(parent_selector_result)
    model.train()
    trainable_parameters = sum(
        parameter.size for _, parameter in tree_flatten(model.trainable_parameters())
    )
    if trainable_parameters <= 0:
        raise RuntimeError("The parent LoRA did not expose trainable adapter matrices")
    write_json(
        candidate_dir / "adapter_config.json",
        _adapter_config_for_child(config, parent, candidate_dir, cycle=cycle),
    )
    max_length = int(training_settings["max_seq_length"])
    train_rows = list(read_jsonl(data_dir / "train.jsonl"))
    valid_rows = list(read_jsonl(data_dir / "valid.jsonl"))
    train_dataset = StatsDataset(
        train_rows,
        tokenizer,
        seed=seed,
        grouped=True,
        curriculum="evolve-interleave",
        max_seq_length=max_length,
    )
    valid_dataset = StatsDataset(
        valid_rows,
        tokenizer,
        seed=seed,
        grouped=False,
        curriculum="random",
        max_seq_length=max_length,
    )
    microsteps = int(settings["microsteps"])
    group_size = int(training_settings["grad_accumulation_steps"])
    args = TrainingArgs(
        batch_size=1,
        iters=microsteps,
        val_batches=-1,
        steps_per_report=group_size,
        steps_per_eval=min(int(settings["validation_every"]), microsteps),
        steps_per_save=min(int(settings["checkpoint_every"]), microsteps),
        max_seq_length=max_length,
        adapter_file=str(candidate_dir / "adapters.safetensors"),
        grad_checkpoint=False,
        grad_accumulation_steps=group_size,
        clear_cache_threshold=12 * 1024**3,
    )
    optimizer_settings = {
        "grad_accumulation_steps": group_size,
        "learning_rate_a": float(settings["learning_rate_a"]),
        "learning_rate_b": float(settings["learning_rate_b"]),
        "warmup_fraction": float(settings["warmup_fraction"]),
        "weight_decay": float(settings["weight_decay"]),
    }
    optimizer = _optimizer(optimizer_settings, microsteps)
    evolution_loss = partial(
        stats_loss,
        component_weights=dict(settings["component_weights"]),
    )
    _enable_gradient_checkpointing_once(model)
    started = time.monotonic()
    callback = _StatsCallback(
        model=model,
        best_path=candidate_dir / "best_adapters.safetensors",
        deadline=started + int(settings["max_seconds"]),
        patience=int(settings["early_stop_evaluations"]),
    )
    stopped = False
    caffeinate = _start_caffeinate()
    try:
        try:
            train(
                model=model,
                optimizer=optimizer,
                train_dataset=train_dataset,
                val_dataset=valid_dataset,
                args=args,
                loss=evolution_loss,
                iterate_batches=stats_iterate_batches,
                training_callback=callback,
            )
        except _StopTraining:
            stopped = True
            mx.save_safetensors(
                str(candidate_dir / "adapters.safetensors"),
                dict(tree_flatten(model.trainable_parameters())),
            )
    finally:
        if caffeinate is not None and caffeinate.poll() is None:
            caffeinate.terminate()
    last_path = candidate_dir / "last_adapters.safetensors"
    mx.save_safetensors(
        str(last_path),
        dict(tree_flatten(model.trainable_parameters())),
    )
    final_loss = float(
        evaluate(
            model=model,
            dataset=valid_dataset,
            batch_size=1,
            num_batches=-1,
            max_seq_length=max_length,
            loss=evolution_loss,
            iterate_batches=stats_iterate_batches,
            clear_cache_threshold=12 * 1024**3,
        )
    )
    best_path = candidate_dir / "best_adapters.safetensors"
    active_path = candidate_dir / "adapters.safetensors"
    checkpoint_paths: list[tuple[str, Path]] = [("last", last_path)]
    checkpoint_paths.extend(
        (f"microstep-{path.name.split('_', 1)[0]}", path)
        for path in sorted(candidate_dir.glob("[0-9]*_adapters.safetensors"))
    )
    if best_path.exists():
        checkpoint_paths.append(("best-loss", best_path))
    parent_weights = parent / "adapters.safetensors"
    unique_paths: list[Path] = [parent_weights]
    checkpoint_candidates: list[dict[str, Any]] = [
        {
            "name": "parent",
            "path": str(parent_weights),
            "selector": parent_selector,
            "family_learning_progress": _family_learning_signal(
                parent_selector_result["predictions"],
                parent_selector_result["predictions"],
            ),
        }
    ]
    for name, path in checkpoint_paths:
        if any(_weight_file_max_abs_delta(path, prior) == 0.0 for prior in unique_paths):
            continue
        model.load_weights(str(path), strict=False)
        model.eval()
        selector_result = _score_loaded_selector(model, tokenizer, selection_rows)
        selector = _selector_summary(selector_result)
        checkpoint_candidates.append(
            {
                "name": name,
                "path": str(path),
                "selector": selector,
                "family_learning_progress": _family_learning_signal(
                    parent_selector_result["predictions"],
                    selector_result["predictions"],
                ),
            }
        )
        unique_paths.append(path)
    selected_checkpoint = _choose_checkpoint(
        checkpoint_candidates,
        dict(settings["promotion"]),
    )
    model.load_weights(str(selected_checkpoint["path"]), strict=False)
    mx.save_safetensors(
        str(active_path),
        dict(tree_flatten(model.trainable_parameters())),
    )
    adapter_config = json.loads(
        (candidate_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    adapter_config.setdefault("stats", {}).update(
        {
            "checkpoint_selection_surface": "frozen-v0.3-valid",
            "selected_checkpoint": selected_checkpoint["name"],
        }
    )
    write_json(candidate_dir / "adapter_config.json", adapter_config)
    status = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "complete": active_path.exists(),
        "cycle": cycle,
        "parent_adapter_path": str(parent),
        "parent_adapter_sha256": sha256_file(parent / "adapters.safetensors"),
        "adapter_path": str(candidate_dir),
        "adapter_sha256": sha256_file(active_path),
        "planned_microsteps": microsteps,
        "stopped": stopped,
        "stop_reason": callback.stop_reason or "completed",
        "trainable_parameters": trainable_parameters,
        "initial_validation_loss": (
            callback.validation_history[0]["loss"] if callback.validation_history else None
        ),
        "final_validation_loss": final_loss,
        "best_validation_loss": callback.best_loss,
        "best_validation_iteration": callback.best_iteration,
        "checkpoint_selection": {
            "surface": "frozen-v0.3-valid",
            "rule": "relative regret, invalidity, accuracy; promotion shard unopened",
            "selected": selected_checkpoint["name"],
            "selected_relative_regret_improvement": selected_checkpoint[
                "relative_regret_improvement"
            ],
            "family_learning_progress": selected_checkpoint[
                "family_learning_progress"
            ],
            "candidates": checkpoint_candidates,
        },
        "validation_history": callback.validation_history,
        "train_history": callback.train_history,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 4),
    }
    write_json(status_path, status)
    del model, tokenizer, train_dataset, valid_dataset, optimizer
    gc.collect()
    mx.clear_cache()
    return status


def _paired_bootstrap(
    parent: list[float],
    candidate: list[float],
    *,
    seed: int,
    repetitions: int,
) -> dict[str, float]:
    parent_array = np.asarray(parent, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    differences = parent_array - candidate_array
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled = rng.integers(0, len(differences), size=len(differences))
        draws[index] = float(np.mean(differences[sampled]))
    return {
        "mean_improvement": float(np.mean(differences)),
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
    }


def _adapter_max_abs_delta(left: Path, right: Path) -> float:
    return _weight_file_max_abs_delta(
        left / "adapters.safetensors",
        right / "adapters.safetensors",
    )


def _score_adapter(
    config: ProjectConfig,
    adapter_path: Path,
    surface: list[dict[str, Any]],
) -> dict[str, Any]:
    model, tokenizer = load(
        _stats_snapshot(config),
        adapter_path=str(adapter_path),
        tokenizer_config={"trust_remote_code": True},
    )
    languages = {
        language: _score_loaded_selector(
            model,
            tokenizer,
            _proposal_records(surface, language=language, view=view),
        )
        for language, view in (
            ("en", "boundary_a"),
            ("zh_Hant", "standard"),
            ("zh_Hans", "standard"),
        )
    }
    selector = languages["en"]
    retention = _retention_score(model, tokenizer, config)
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    return {"selector": selector, "languages": languages, "retention": retention}


def _group_regret(
    predictions: list[dict[str, Any]],
    key: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in predictions:
        grouped[str(row[key])].append(float(row["normalized_regret"]))
    return {name: float(np.mean(values)) for name, values in sorted(grouped.items())}


def _noninferior_mapping(
    parent: dict[str, float],
    candidate: dict[str, float],
    *,
    maximum_regression: float,
    higher_is_better: bool,
) -> bool:
    if set(parent) != set(candidate):
        return False
    if higher_is_better:
        return all(
            candidate[key] >= parent[key] - maximum_regression for key in parent
        )
    return all(candidate[key] <= parent[key] + maximum_regression for key in parent)


def evaluate_evolution_candidate(config: ProjectConfig, *, force: bool = False) -> dict[str, Any]:
    training = train_evolution_candidate(config, force=False)
    cycle = int(training["cycle"])
    data_dir, artifact_dir = _cycle_paths(config, cycle)
    manifest = prepare_evolution_cycle(config, force=False)
    promotion_manifest = dict(manifest["promotion_shard"])
    promotion_path = data_dir / "promotion.jsonl"
    if sha256_file(promotion_path) != promotion_manifest["sha256"]:
        raise RuntimeError("The cycle promotion shard changed after preparation")
    promotion_surface = list(read_jsonl(promotion_path))
    report_path = artifact_dir / "comparison.json"
    settings = _evolution_settings(config)
    gates = dict(settings["promotion"])
    parent_path = Path(str(training["parent_adapter_path"]))
    candidate_path = Path(str(training["adapter_path"]))
    fingerprint = canonical_hash(
        {
            "cycle": cycle,
            "parent": training["parent_adapter_sha256"],
            "candidate": training["adapter_sha256"],
            "promotion_shard": promotion_manifest["sha256"],
            "gates": gates,
            "evaluator_version": 3,
        }
    )
    if report_path.exists() and not force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint and existing.get("complete"):
            return existing
    maximum_adapter_delta = _adapter_max_abs_delta(parent_path, candidate_path)
    if maximum_adapter_delta == 0.0:
        result = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "complete": True,
            "cycle": cycle,
            "promoted": False,
            "rejection_reason": "numerically_identical_to_parent",
            "maximum_adapter_tensor_delta": maximum_adapter_delta,
            "parent": {
                "adapter_path": str(parent_path),
                "adapter_sha256": training["parent_adapter_sha256"],
            },
            "candidate": {
                "adapter_path": str(candidate_path),
                "adapter_sha256": training["adapter_sha256"],
            },
            "relative_regret_improvement": 0.0,
            "paired_bootstrap": None,
            "gates": {
                "nonzero_update": False,
                "relative_regret": False,
                "paired_noninferiority": None,
                "accuracy": None,
                "invalidity": None,
                "retention": None,
                "language_accuracy": None,
                "language_regret": None,
                "domain_accuracy": None,
                "family_regret": None,
            },
            "family_learning_progress": training["checkpoint_selection"][
                "family_learning_progress"
            ],
            "promotion_shard": {
                **promotion_manifest,
                "opened_for_scoring": False,
                "paired_parent_candidate": False,
            },
            "final_surface_opened": False,
        }
        write_json(report_path, result)
        return result
    parent = _score_adapter(config, parent_path, promotion_surface)
    candidate = _score_adapter(config, candidate_path, promotion_surface)
    parent_predictions = parent["selector"]["predictions"]
    candidate_predictions = candidate["selector"]["predictions"]
    if [row["blueprint_id"] for row in parent_predictions] != [
        row["blueprint_id"] for row in candidate_predictions
    ]:
        raise RuntimeError("Parent and candidate development predictions are not paired")
    bootstrap = _paired_bootstrap(
        [float(row["normalized_regret"]) for row in parent_predictions],
        [float(row["normalized_regret"]) for row in candidate_predictions],
        seed=int(config.section("project")["seed"]) + cycle,
        repetitions=int(gates["bootstrap_repetitions"]),
    )
    parent_regret = float(parent["selector"]["normalized_regret"])
    candidate_regret = float(candidate["selector"]["normalized_regret"])
    relative_improvement = (
        (parent_regret - candidate_regret) / parent_regret if parent_regret else 0.0
    )
    parent_language_accuracy = {
        key: float(value["accuracy"]) for key, value in parent["languages"].items()
    }
    candidate_language_accuracy = {
        key: float(value["accuracy"]) for key, value in candidate["languages"].items()
    }
    parent_language_regret = {
        key: float(value["normalized_regret"])
        for key, value in parent["languages"].items()
    }
    candidate_language_regret = {
        key: float(value["normalized_regret"])
        for key, value in candidate["languages"].items()
    }
    parent_family_regret = _group_regret(parent_predictions, "family_id")
    candidate_family_regret = _group_regret(candidate_predictions, "family_id")
    gate_results = {
        "nonzero_update": True,
        "relative_regret": relative_improvement
        >= float(gates["minimum_relative_regret_improvement"]),
        "paired_noninferiority": bootstrap["ci95_lower"]
        >= float(gates["bootstrap_ci_lower_floor"]),
        "accuracy": float(candidate["selector"]["accuracy"])
        >= float(parent["selector"]["accuracy"]) - float(gates["maximum_accuracy_regression"]),
        "invalidity": float(candidate["selector"]["invalid_selection_rate"])
        <= float(parent["selector"]["invalid_selection_rate"])
        + float(gates["maximum_invalidity_increase"]),
        "retention": float(candidate["retention"]["accuracy"])
        >= float(parent["retention"]["accuracy"]) - float(gates["maximum_retention_regression"]),
        "language_accuracy": _noninferior_mapping(
            parent_language_accuracy,
            candidate_language_accuracy,
            maximum_regression=float(gates["maximum_language_accuracy_regression"]),
            higher_is_better=True,
        ),
        "language_regret": _noninferior_mapping(
            parent_language_regret,
            candidate_language_regret,
            maximum_regression=float(gates["maximum_language_regret_increase"]),
            higher_is_better=False,
        ),
        "domain_accuracy": _noninferior_mapping(
            {
                key: float(value)
                for key, value in parent["selector"]["domain_accuracy"].items()
            },
            {
                key: float(value)
                for key, value in candidate["selector"]["domain_accuracy"].items()
            },
            maximum_regression=float(gates["maximum_domain_accuracy_regression"]),
            higher_is_better=True,
        ),
        "family_regret": _noninferior_mapping(
            parent_family_regret,
            candidate_family_regret,
            maximum_regression=float(gates["maximum_family_regret_increase"]),
            higher_is_better=False,
        ),
    }
    promoted = all(gate_results.values())
    failed_gates = sorted(name for name, passed in gate_results.items() if not passed)
    result = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "complete": True,
        "cycle": cycle,
        "promoted": promoted,
        "rejection_reason": (
            None if promoted else f"promotion_gates_failed:{','.join(failed_gates)}"
        ),
        "parent": parent,
        "candidate": candidate,
        "relative_regret_improvement": relative_improvement,
        "paired_bootstrap": bootstrap,
        "noninferiority_details": {
            "parent_language_accuracy": parent_language_accuracy,
            "candidate_language_accuracy": candidate_language_accuracy,
            "parent_language_regret": parent_language_regret,
            "candidate_language_regret": candidate_language_regret,
            "parent_family_regret": parent_family_regret,
            "candidate_family_regret": candidate_family_regret,
        },
        "gates": gate_results,
        "family_learning_progress": training["checkpoint_selection"][
            "family_learning_progress"
        ],
        "promotion_shard": {
            **promotion_manifest,
            "opened_for_scoring": True,
            "paired_parent_candidate": True,
        },
        "final_surface_opened": False,
    }
    write_json(report_path, result)
    return result


def _commit_cycle(config: ProjectConfig, comparison: dict[str, Any]) -> dict[str, Any]:
    archive = _load_archive(config)
    cycle = int(comparison["cycle"])
    if any(int(item["cycle"]) == cycle for item in archive["cycles"]):
        return evolution_status(config)
    _, artifact_dir = _cycle_paths(config, cycle)
    training = json.loads((artifact_dir / "candidate" / "status.json").read_text())
    adapter_config_path = artifact_dir / "candidate" / "adapter_config.json"
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    adapter_config.setdefault("stats", {})["promotion_status"] = (
        "promoted" if comparison["promoted"] else "rejected"
    )
    adapter_config["stats"]["promotion_comparison"] = str(
        artifact_dir / "comparison.json"
    )
    write_json(adapter_config_path, adapter_config)
    node = {
        "node_id": f"cycle-{cycle:04d}-{training['adapter_sha256'][:12]}",
        "cycle": cycle,
        "adapter_path": training["adapter_path"],
        "adapter_sha256": training["adapter_sha256"],
        "parent_adapter_sha256": training["parent_adapter_sha256"],
        "promoted": bool(comparison["promoted"]),
        "comparison_path": str(artifact_dir / "comparison.json"),
    }
    archive["nodes"].append(node)
    archive["cycles"].append(
        {
            "cycle": cycle,
            "node_id": node["node_id"],
            "promoted": node["promoted"],
            "relative_regret_improvement": comparison["relative_regret_improvement"],
            "gates": comparison["gates"],
        }
    )
    archive["family_learning_progress"].update(comparison["family_learning_progress"])
    if node["promoted"]:
        archive["champion"] = node
    write_json(_archive_path(config), archive)
    champion = archive["champion"]
    write_json(
        config.path_for("artifact_dir") / "selected.json",
        {**champion, "schema_version": 1, "complete": True},
    )
    return evolution_status(config)


def run_evolution_cycle(
    config: ProjectConfig,
    *,
    prepare_only: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    manifest = prepare_evolution_cycle(config, force=force)
    if prepare_only:
        return {"stage": "prepared", "manifest": manifest, "status": evolution_status(config)}
    training = train_evolution_candidate(config, force=force)
    comparison = evaluate_evolution_candidate(config, force=force)
    status = _commit_cycle(config, comparison)
    return {
        "stage": "complete",
        "manifest": manifest,
        "training": training,
        "comparison": comparison,
        "status": status,
    }


def run_evolution(
    config: ProjectConfig,
    *,
    cycles: int = 1,
    prepare_only: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    settings = _evolution_settings(config)
    if cycles < 1 or cycles > int(settings["max_cycles_per_run"]):
        raise ValueError(f"cycles must be between 1 and {int(settings['max_cycles_per_run'])}")
    results: list[dict[str, Any]] = []
    with _evolution_lock(config):
        for _ in range(cycles):
            results.append(run_evolution_cycle(config, prepare_only=prepare_only, force=force))
            if prepare_only:
                break
    return {"cycles": results, "status": evolution_status(config)}

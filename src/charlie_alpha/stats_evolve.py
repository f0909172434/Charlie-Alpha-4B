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


def _proposal_method_id(proposal: dict[str, Any]) -> str:
    simulation = proposal.get("simulation")
    if not isinstance(simulation, dict) or not simulation.get("selected_method_id"):
        return "unknown"
    return str(simulation["selected_method_id"])


def _select_diverse(
    proposals: list[dict[str, Any]],
    count: int,
    *,
    max_per_family: int | None = None,
    max_per_method: int | None = None,
) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("Selection count must be non-negative")
    if max_per_family is not None and max_per_family <= 0:
        raise ValueError("max_per_family must be positive")
    if max_per_method is not None and max_per_method <= 0:
        raise ValueError("max_per_method must be positive")
    ranked = sorted(
        proposals,
        key=lambda item: (-float(item["task_reward"]), str(item["blueprint_id"])),
    )
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    family_counts: dict[str, int] = defaultdict(int)
    method_counts: dict[str, int] = defaultdict(int)

    def allowed(item: dict[str, Any]) -> bool:
        family_id = str(item["family_id"])
        method_id = _proposal_method_id(item)
        return bool(
            str(item["blueprint_id"]) not in used
            and (max_per_family is None or family_counts[family_id] < max_per_family)
            and (max_per_method is None or method_counts[method_id] < max_per_method)
        )

    def add(item: dict[str, Any]) -> None:
        selected.append(item)
        used.add(str(item["blueprint_id"]))
        family_counts[str(item["family_id"])] += 1
        method_counts[_proposal_method_id(item)] += 1

    # Preserve one stepping stone per represented DGP family before filling by
    # reward. Within each family, choose the highest-reward item that also
    # respects the method ceiling.
    for family_id in sorted({str(item["family_id"]) for item in ranked}):
        candidate = next(
            (item for item in ranked if str(item["family_id"]) == family_id and allowed(item)),
            None,
        )
        if candidate is not None and len(selected) < count:
            add(candidate)

    for item in ranked:
        if len(selected) >= count:
            break
        if allowed(item):
            add(item)
    return selected


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
    selected = _select_diverse(
        eligible,
        int(settings["tasks_per_cycle"]),
        max_per_family=int(settings["max_tasks_per_family"]),
        max_per_method=int(settings["max_tasks_per_method"]),
    )
    if len(selected) < int(settings["tasks_per_cycle"]):
        raise RuntimeError(f"Only {len(selected)} valid novel tasks were available for evolution")
    write_jsonl(data_dir / "proposals.jsonl", proposals)
    write_jsonl(data_dir / "selected.jsonl", selected)
    data = _write_training_records(config, cycle, selected, data_dir)
    promotion = _ensure_promotion_shard(config, cycle, data_dir)
    selected_ids = {str(item["blueprint_id"]) for item in selected}
    promotion_ids = {
        str(item["scenario"]["blueprint_id"]) for item in read_jsonl(data_dir / "promotion.jsonl")
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
        "selection_formula": (
            "validity * novelty * frontier * learning_progress, subject to family and "
            "oracle-method ceilings"
        ),
        "selection_constraints": {
            "max_tasks_per_family": int(settings["max_tasks_per_family"]),
            "max_tasks_per_method": int(settings["max_tasks_per_method"]),
        },
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


def prepare_evolution_ablation(
    config: ProjectConfig,
    cycle_manifest: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Prepare an equal-compute random-DGP control without reading promotion outcomes."""
    cycle = int(cycle_manifest["cycle"])
    cycle_data_dir, _ = _cycle_paths(config, cycle)
    control_dir = cycle_data_dir / "ablation" / "random-control"
    manifest_path = control_dir / "manifest.json"
    settings = _evolution_settings(config)
    ablation = dict(settings["ablation"])
    seed = int(ablation["random_control_seed_base"]) + cycle * 1_000_003
    fingerprint = canonical_hash(
        {
            "cycle": cycle,
            "parent": cycle_manifest["parent"]["adapter_sha256"],
            "proposal_pool": int(settings["proposal_pool"]),
            "tasks_per_cycle": int(settings["tasks_per_cycle"]),
            "seed": seed,
            "stats_data": config.section("stats_data"),
            "selection_constraints": {
                "max_tasks_per_family": int(settings["max_tasks_per_family"]),
                "max_tasks_per_method": int(settings["max_tasks_per_method"]),
            },
            "control_generator_version": 2,
        }
    )
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("fingerprint") == fingerprint
            and existing.get("complete")
            and _training_records_are_current(
                config,
                control_dir,
                dict(existing.get("data", {})),
            )
        ):
            if existing["data"]["valid_sha256"] != cycle_manifest["data"]["valid_sha256"]:
                raise RuntimeError("Ablation arms do not share the same validation records")
            return existing

    control_dir.mkdir(parents=True, exist_ok=True)
    split = f"evolve-random-control-{cycle:04d}"
    scenarios = build_blueprints(
        {split: int(settings["proposal_pool"])},
        seed=seed,
        active_search=False,
    )
    simulations = [_simulate(config, scenario) for scenario in scenarios]
    parent_adapter = Path(str(cycle_manifest["parent"]["adapter_path"]))
    parent_scores = _score_proposals(config, parent_adapter, simulations)
    prediction_by_id = {str(row["blueprint_id"]): row for row in parent_scores["predictions"]}
    references = _existing_references(config)
    rng = random.Random(seed)
    proposals: list[dict[str, Any]] = []
    for simulation in simulations:
        scenario = _scenario(simulation["scenario"])
        validity = float(
            bool(simulation["valid_method_ids"])
            and math.isclose(
                sum(float(row["soft_target"]) for row in simulation["candidates"]),
                1.0,
                abs_tol=1e-7,
            )
        )
        proposals.append(
            {
                "blueprint_id": scenario.blueprint_id,
                "family_id": scenario.family_id,
                "parent_normalized_regret": float(
                    prediction_by_id[scenario.blueprint_id]["normalized_regret"]
                ),
                "validity": validity,
                "novelty": _novelty(scenario, references),
                "task_reward": rng.random() if validity else 0.0,
                "selection_rule": "uniform-random-after-validity-and-novelty",
                "simulation": simulation,
            }
        )
    eligible = [
        item
        for item in proposals
        if float(item["validity"]) == 1.0
        and float(item["novelty"]) >= float(settings["novelty_floor"])
    ]
    selected = _select_diverse(
        eligible,
        int(settings["tasks_per_cycle"]),
        max_per_family=int(settings["max_tasks_per_family"]),
        max_per_method=int(settings["max_tasks_per_method"]),
    )
    if len(selected) < int(settings["tasks_per_cycle"]):
        raise RuntimeError(f"Only {len(selected)} valid novel random-control tasks were available")
    write_jsonl(control_dir / "proposals.jsonl", proposals)
    write_jsonl(control_dir / "selected.jsonl", selected)
    data = _write_training_records(config, cycle, selected, control_dir)
    if data["valid_sha256"] != cycle_manifest["data"]["valid_sha256"]:
        raise RuntimeError("Ablation arms do not share the same validation records")
    adaptive_ids = {
        str(row["blueprint_id"]) for row in read_jsonl(cycle_data_dir / "selected.jsonl")
    }
    control_ids = {str(row["blueprint_id"]) for row in selected}
    if adaptive_ids & control_ids:
        raise RuntimeError("Random-control and adaptive training tasks overlap")
    result = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "cycle": cycle,
        "arm": "random-control",
        "selection_rule": "uniform-random-after-validity-and-novelty",
        "selection_constraints": {
            "max_tasks_per_family": int(settings["max_tasks_per_family"]),
            "max_tasks_per_method": int(settings["max_tasks_per_method"]),
        },
        "proposal_count": len(proposals),
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "data": data,
        "proposal_parent_scoring_compute_matched": True,
        "total_selection_parent_scoring_compute_matched": False,
        "promotion_shard_read": False,
    }
    write_json(manifest_path, result)
    return result


def _adapter_config_for_child(
    config: ProjectConfig,
    parent: Path,
    destination: Path,
    *,
    cycle: int,
    arm: str,
) -> dict[str, Any]:
    value = json.loads((parent / "adapter_config.json").read_text(encoding="utf-8"))
    evolution = _evolution_settings(config)
    value["adapter_path"] = str(destination)
    value.setdefault("stats", {}).update(
        {
            "method": "DGP-Evolve",
            "cycle": cycle,
            "ablation_arm": arm,
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
            and metrics["invalid_selection_rate"] <= parent_metrics["invalid_selection_rate"]
            and metrics["accuracy"] >= parent_metrics["accuracy"] - maximum_accuracy_regression
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


def _train_evolution_arm(
    config: ProjectConfig,
    manifest: dict[str, Any],
    *,
    arm: str,
    data_dir: Path,
    candidate_dir: Path,
    force: bool,
    training_seed: int | None = None,
    run_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cycle = int(manifest["cycle"])
    status_path = candidate_dir / "status.json"
    parent = Path(str(manifest["parent"]["adapter_path"]))
    settings = _evolution_settings(config)
    training_settings = config.section("stats_training")
    arm_settings = dict(run_settings or {})
    fingerprint = canonical_hash(
        {
            "cycle": cycle,
            "arm": arm,
            "parent": sha256_file(parent / "adapters.safetensors"),
            "train": sha256_file(data_dir / "train.jsonl"),
            "valid": sha256_file(data_dir / "valid.jsonl"),
            "settings": settings,
            "training_seed": training_seed,
            "run_settings": arm_settings,
            "trainer_version": 6 if arm_settings else 5,
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
    seed = (
        int(training_seed)
        if training_seed is not None
        else int(config.section("project")["seed"]) + cycle
    )
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
        _adapter_config_for_child(
            config,
            parent,
            candidate_dir,
            cycle=cycle,
            arm=arm,
        ),
    )
    max_length = int(training_settings["max_seq_length"])
    train_rows = list(read_jsonl(data_dir / "train.jsonl"))
    valid_rows = list(read_jsonl(data_dir / "valid.jsonl"))
    component_weights = dict(settings["component_weights"])
    selector_only = (
        float(component_weights["method"]) == 1.0
        and float(component_weights["plan_tool"]) == 0.0
        and float(component_weights["report"]) == 0.0
    )
    curriculum = str(arm_settings.get("curriculum", "evolve-interleave"))
    train_dataset = StatsDataset(
        train_rows,
        tokenizer,
        seed=seed,
        grouped=True,
        curriculum=curriculum,
        max_seq_length=max_length,
        selector_only=selector_only,
    )
    valid_dataset = StatsDataset(
        valid_rows,
        tokenizer,
        seed=seed,
        grouped=False,
        curriculum="random",
        max_seq_length=max_length,
        selector_only=selector_only,
    )
    microsteps = int(arm_settings.get("microsteps", settings["microsteps"]))
    group_size = int(training_settings["grad_accumulation_steps"])
    clear_cache_threshold = int(float(settings["clear_cache_threshold_gb"]) * 1024**3)
    args = TrainingArgs(
        batch_size=1,
        iters=microsteps,
        val_batches=-1,
        steps_per_report=group_size,
        steps_per_eval=min(
            int(arm_settings.get("validation_every", settings["validation_every"])),
            microsteps,
        ),
        steps_per_save=min(
            int(arm_settings.get("checkpoint_every", settings["checkpoint_every"])),
            microsteps,
        ),
        max_seq_length=max_length,
        adapter_file=str(candidate_dir / "adapters.safetensors"),
        grad_checkpoint=False,
        grad_accumulation_steps=group_size,
        clear_cache_threshold=clear_cache_threshold,
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
        component_weights=component_weights,
    )
    _enable_gradient_checkpointing_once(model)
    started = time.monotonic()
    callback = _StatsCallback(
        model=model,
        best_path=candidate_dir / "best_adapters.safetensors",
        deadline=started + int(settings["max_seconds"]),
        patience=int(
            arm_settings.get("early_stop_evaluations", settings["early_stop_evaluations"])
        ),
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
            clear_cache_threshold=clear_cache_threshold,
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
    selected_validation_loss = (
        final_loss
        if selected_checkpoint["name"] == "last"
        else float(
            evaluate(
                model=model,
                dataset=valid_dataset,
                batch_size=1,
                num_batches=-1,
                max_seq_length=max_length,
                loss=evolution_loss,
                iterate_batches=stats_iterate_batches,
                clear_cache_threshold=clear_cache_threshold,
            )
        )
    )
    mx.save_safetensors(
        str(active_path),
        dict(tree_flatten(model.trainable_parameters())),
    )
    adapter_config = json.loads((candidate_dir / "adapter_config.json").read_text(encoding="utf-8"))
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
        "arm": arm,
        "training_seed": seed,
        "curriculum": curriculum,
        "parent_adapter_path": str(parent),
        "parent_adapter_sha256": sha256_file(parent / "adapters.safetensors"),
        "adapter_path": str(candidate_dir),
        "adapter_sha256": sha256_file(active_path),
        "train_sha256": sha256_file(data_dir / "train.jsonl"),
        "valid_sha256": sha256_file(data_dir / "valid.jsonl"),
        "planned_microsteps": microsteps,
        "stopped": stopped,
        "stop_reason": callback.stop_reason or "completed",
        "trainable_parameters": trainable_parameters,
        "initial_validation_loss": (
            callback.validation_history[0]["loss"] if callback.validation_history else None
        ),
        "final_validation_loss": final_loss,
        "selected_validation_loss": selected_validation_loss,
        "best_validation_loss": callback.best_loss,
        "best_validation_iteration": callback.best_iteration,
        "checkpoint_selection": {
            "surface": "frozen-v0.3-valid",
            "rule": "relative regret, invalidity, accuracy; promotion shard unopened",
            "selected": selected_checkpoint["name"],
            "selected_relative_regret_improvement": selected_checkpoint[
                "relative_regret_improvement"
            ],
            "family_learning_progress": selected_checkpoint["family_learning_progress"],
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


def train_evolution_candidate(config: ProjectConfig, *, force: bool = False) -> dict[str, Any]:
    manifest = prepare_evolution_cycle(config, force=False)
    cycle = int(manifest["cycle"])
    data_dir, artifact_dir = _cycle_paths(config, cycle)
    return _train_evolution_arm(
        config,
        manifest,
        arm="adaptive",
        data_dir=data_dir,
        candidate_dir=artifact_dir / "candidate",
        force=force,
    )


def _selected_validation_metrics(status: dict[str, Any]) -> dict[str, float]:
    selection = dict(status["checkpoint_selection"])
    selected_name = str(selection["selected"])
    selected = next(item for item in selection["candidates"] if str(item["name"]) == selected_name)
    return {key: float(value) for key, value in selected["selector"].items()}


def _choose_ablation_winner(statuses: list[dict[str, Any]]) -> dict[str, Any]:
    if not statuses:
        raise ValueError("At least one ablation arm is required")
    invariants = {
        (
            str(status["parent_adapter_sha256"]),
            str(status["valid_sha256"]),
            int(status["planned_microsteps"]),
            int(status["trainable_parameters"]),
        )
        for status in statuses
    }
    if len(invariants) != 1:
        raise RuntimeError("Ablation arms do not have equal parent, validation, or compute")

    def key(status: dict[str, Any]) -> tuple[float, float, float, float, int, str]:
        metrics = _selected_validation_metrics(status)
        return (
            metrics["normalized_regret"],
            metrics["invalid_selection_rate"],
            -metrics["accuracy"],
            float(status["selected_validation_loss"]),
            0 if status["arm"] == "random-control" else 1,
            str(status["arm"]),
        )

    return min(statuses, key=key)


def train_evolution_ablation(config: ProjectConfig, *, force: bool = False) -> dict[str, Any]:
    cycle_manifest = prepare_evolution_cycle(config, force=False)
    control_manifest = prepare_evolution_ablation(
        config,
        cycle_manifest,
        force=force,
    )
    cycle = int(cycle_manifest["cycle"])
    data_dir, artifact_dir = _cycle_paths(config, cycle)
    ablation_settings = dict(_evolution_settings(config)["ablation"])
    arms = [str(value) for value in ablation_settings["arms"]]
    if set(arms) != {"random-control", "adaptive"} or len(arms) != 2:
        raise ValueError("The equal-compute ablation requires random-control and adaptive arms")
    data_paths = {
        "random-control": data_dir / "ablation" / "random-control",
        "adaptive": data_dir,
    }
    statuses: dict[str, dict[str, Any]] = {}
    for arm in arms:
        statuses[arm] = _train_evolution_arm(
            config,
            cycle_manifest,
            arm=arm,
            data_dir=data_paths[arm],
            candidate_dir=artifact_dir / "ablation" / arm,
            force=force,
        )
    winner = _choose_ablation_winner(list(statuses.values()))
    control_metrics = _selected_validation_metrics(statuses["random-control"])
    adaptive_metrics = _selected_validation_metrics(statuses["adaptive"])
    adaptive_relative_improvement = (
        (control_metrics["normalized_regret"] - adaptive_metrics["normalized_regret"])
        / control_metrics["normalized_regret"]
        if control_metrics["normalized_regret"]
        else 0.0
    )
    minimum_ablation_improvement = float(
        ablation_settings["minimum_adaptive_relative_regret_improvement_over_control"]
    )
    report = {
        "schema_version": 1,
        "complete": True,
        "cycle": cycle,
        "fingerprint": canonical_hash(
            {
                "cycle_manifest": cycle_manifest["fingerprint"],
                "control_manifest": control_manifest["fingerprint"],
                "arms": {
                    arm: {
                        "training": status["fingerprint"],
                        "adapter": status["adapter_sha256"],
                    }
                    for arm, status in sorted(statuses.items())
                },
                "settings": ablation_settings,
                "ablation_evaluator_version": 1,
            }
        ),
        "compute_equal": True,
        "compute_equal_scope": "training-only",
        "training_compute_equal": True,
        "total_pipeline_compute_equal": False,
        "selection_compute": {
            "random-control": {
                "new_simulations": int(cycle_manifest["proposal_count"]),
                "parent_scored_tasks": int(cycle_manifest["proposal_count"]),
            },
            "adaptive": {
                "new_simulations": int(cycle_manifest["proposal_count"]),
                "parent_scored_tasks": int(cycle_manifest["proposal_count"])
                + len(_surface(config, "dev")),
            },
            "note": "Adaptive selection additionally scores the reusable dev surface; "
            "promotion data remains sealed.",
        },
        "shared_parent_adapter_sha256": winner["parent_adapter_sha256"],
        "shared_validation_sha256": winner["valid_sha256"],
        "planned_microsteps_per_arm": winner["planned_microsteps"],
        "trainable_parameters_per_arm": winner["trainable_parameters"],
        "training_order": arms,
        "arms": {
            arm: {
                "selected_checkpoint": status["checkpoint_selection"]["selected"],
                "validation": _selected_validation_metrics(status),
                "selected_validation_loss": status["selected_validation_loss"],
                "adapter_sha256": status["adapter_sha256"],
            }
            for arm, status in sorted(statuses.items())
        },
        "winner_arm": winner["arm"],
        "winner_status_path": str(Path(winner["adapter_path"]) / "status.json"),
        "adaptive_relative_regret_improvement_over_control": adaptive_relative_improvement,
        "adaptive_ablation_gate": adaptive_relative_improvement >= minimum_ablation_improvement,
        "minimum_adaptive_relative_regret_improvement_over_control": (minimum_ablation_improvement),
        "promotion_shard_opened": False,
    }
    write_json(artifact_dir / "ablation.json", report)
    return {
        "report": report,
        "winner": winner,
        "arms": statuses,
        "control_manifest": control_manifest,
    }


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
    raw_delta = _weight_file_max_abs_delta(
        left / "adapters.safetensors",
        right / "adapters.safetensors",
    )
    if math.isfinite(raw_delta):
        return raw_delta
    left_weights = mx.load(str(left / "adapters.safetensors"))
    right_weights = mx.load(str(right / "adapters.safetensors"))
    if set(left_weights) != set(right_weights) or not all(
        key.endswith((".lora_a", ".lora_b")) for key in left_weights
    ):
        return math.inf
    left_config = json.loads((left / "adapter_config.json").read_text(encoding="utf-8"))
    right_config = json.loads((right / "adapter_config.json").read_text(encoding="utf-8"))
    left_scale = float(left_config["lora_parameters"]["scale"])
    right_scale = float(right_config["lora_parameters"]["scale"])
    prefixes = sorted(
        key.removesuffix(".lora_a") for key in left_weights if key.endswith(".lora_a")
    )
    if {f"{prefix}.lora_b" for prefix in prefixes} != {
        key for key in left_weights if key.endswith(".lora_b")
    }:
        return math.inf
    maximum = 0.0
    for prefix in prefixes:
        left_effective = left_scale * (
            left_weights[f"{prefix}.lora_a"] @ left_weights[f"{prefix}.lora_b"]
        )
        right_effective = right_scale * (
            right_weights[f"{prefix}.lora_a"] @ right_weights[f"{prefix}.lora_b"]
        )
        maximum = max(
            maximum,
            float(mx.max(mx.abs(left_effective - right_effective)).item()),
        )
        del left_effective, right_effective
        mx.clear_cache()
    return maximum


def _score_adapter(
    config: ProjectConfig,
    adapter_path: Path,
    surface: list[dict[str, Any]],
) -> dict[str, Any]:
    return _score_adapter_surfaces(
        config,
        adapter_path,
        {"surface": surface},
    )["surface"]


def _score_adapter_surfaces(
    config: ProjectConfig,
    adapter_path: Path,
    surfaces: dict[str, list[dict[str, Any]]],
    *,
    include_retention: bool = True,
) -> dict[str, dict[str, Any]]:
    if not surfaces:
        raise ValueError("Adapter scoring requires at least one surface")
    model, tokenizer = load(
        _stats_snapshot(config),
        adapter_path=str(adapter_path),
        tokenizer_config={"trust_remote_code": True},
    )
    scored = {
        name: {
            "languages": {
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
        }
        for name, surface in surfaces.items()
    }
    retention = _retention_score(model, tokenizer, config) if include_retention else None
    for value in scored.values():
        value["selector"] = value["languages"]["en"]
        value["retention"] = retention
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    return scored


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
        return all(candidate[key] >= parent[key] - maximum_regression for key in parent)
    return all(candidate[key] <= parent[key] + maximum_regression for key in parent)


def evaluate_evolution_candidate(
    config: ProjectConfig,
    *,
    training: dict[str, Any] | None = None,
    cycle_manifest: dict[str, Any] | None = None,
    ablation_report: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    training = training or train_evolution_candidate(config, force=False)
    cycle = int(training["cycle"])
    data_dir, artifact_dir = _cycle_paths(config, cycle)
    manifest = cycle_manifest or prepare_evolution_cycle(config, force=False)
    if int(manifest["cycle"]) != cycle:
        raise RuntimeError("Training status and promotion manifest belong to different cycles")
    promotion_manifest = dict(manifest["promotion_shard"])
    promotion_path = data_dir / "promotion.jsonl"
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
            "training_arm": training.get("arm", "adaptive"),
            "ablation": ablation_report.get("fingerprint") if ablation_report else None,
            "promotion_shard": promotion_manifest["sha256"],
            "gates": gates,
            "evaluator_version": 4,
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
            "training_arm": training.get("arm", "adaptive"),
            "training_status_path": str(Path(training["adapter_path"]) / "status.json"),
            "ablation": (
                {
                    "fingerprint": ablation_report["fingerprint"],
                    "winner_arm": ablation_report["winner_arm"],
                    "adaptive_relative_regret_improvement_over_control": ablation_report[
                        "adaptive_relative_regret_improvement_over_control"
                    ],
                    "adaptive_ablation_gate": ablation_report["adaptive_ablation_gate"],
                }
                if ablation_report
                else None
            ),
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
    if sha256_file(promotion_path) != promotion_manifest["sha256"]:
        raise RuntimeError("The cycle promotion shard changed after preparation")
    promotion_surface = list(read_jsonl(promotion_path))
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
        key: float(value["normalized_regret"]) for key, value in parent["languages"].items()
    }
    candidate_language_regret = {
        key: float(value["normalized_regret"]) for key, value in candidate["languages"].items()
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
            {key: float(value) for key, value in parent["selector"]["domain_accuracy"].items()},
            {key: float(value) for key, value in candidate["selector"]["domain_accuracy"].items()},
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
        "training_arm": training.get("arm", "adaptive"),
        "training_status_path": str(Path(training["adapter_path"]) / "status.json"),
        "ablation": (
            {
                "fingerprint": ablation_report["fingerprint"],
                "winner_arm": ablation_report["winner_arm"],
                "adaptive_relative_regret_improvement_over_control": ablation_report[
                    "adaptive_relative_regret_improvement_over_control"
                ],
                "adaptive_ablation_gate": ablation_report["adaptive_ablation_gate"],
            }
            if ablation_report
            else None
        ),
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
        "family_learning_progress": training["checkpoint_selection"]["family_learning_progress"],
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
    training_status_path = Path(
        str(
            comparison.get(
                "training_status_path",
                artifact_dir / "candidate" / "status.json",
            )
        )
    )
    training = json.loads(training_status_path.read_text(encoding="utf-8"))
    adapter_config_path = training_status_path.parent / "adapter_config.json"
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    adapter_config.setdefault("stats", {})["promotion_status"] = (
        "promoted" if comparison["promoted"] else "rejected"
    )
    adapter_config["stats"]["promotion_comparison"] = str(artifact_dir / "comparison.json")
    write_json(adapter_config_path, adapter_config)
    node = {
        "node_id": f"cycle-{cycle:04d}-{training['adapter_sha256'][:12]}",
        "cycle": cycle,
        "ablation_arm": training.get("arm", "adaptive"),
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
            "ablation_arm": node["ablation_arm"],
            "promoted": node["promoted"],
            "relative_regret_improvement": comparison["relative_regret_improvement"],
            "gates": comparison["gates"],
            "ablation": comparison.get("ablation"),
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
    ablation_enabled = bool(
        dict(_evolution_settings(config).get("ablation", {})).get("enabled", False)
    )
    if prepare_only:
        control = (
            prepare_evolution_ablation(config, manifest, force=force) if ablation_enabled else None
        )
        return {
            "stage": "prepared",
            "manifest": manifest,
            "control_manifest": control,
            "status": evolution_status(config),
        }
    if ablation_enabled:
        ablation = train_evolution_ablation(config, force=force)
        training = ablation["winner"]
        comparison = evaluate_evolution_candidate(
            config,
            training=training,
            cycle_manifest=manifest,
            ablation_report=ablation["report"],
            force=force,
        )
    else:
        ablation = None
        training = train_evolution_candidate(config, force=force)
        comparison = evaluate_evolution_candidate(
            config,
            training=training,
            cycle_manifest=manifest,
            force=force,
        )
    status = _commit_cycle(config, comparison)
    return {
        "stage": "complete",
        "manifest": manifest,
        "ablation": ablation,
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

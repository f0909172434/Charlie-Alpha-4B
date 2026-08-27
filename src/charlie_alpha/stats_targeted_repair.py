from __future__ import annotations

import copy
import gc
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl
from .provenance import lifecycle_open_state, mark_lifecycle_opened
from .stats_catalog import FAMILIES
from .stats_cone import _apply_flat_update, _family_gradient_matrix
from .stats_dgp import Scenario, build_blueprints, simulate_scenario
from .stats_evolve import _adapter_config_for_child, _start_caffeinate
from .stats_experts import _score_loaded_family
from .stats_project import _diagnostic_groups, project_policy
from .stats_robust_experts import (
    _balanced_family_scenarios,
    _candidate_regrets,
    _route_selection_summary,
    _route_vs_control_summary,
    _training_records,
    select_crossfit_expert_option,
)
from .stats_route import _aggregate_predictions, _family_metrics, _family_noninferior
from .stats_training import (
    _enable_gradient_checkpointing_once,
    _selector_menu_probabilities,
    _stats_snapshot,
)


def _repair_root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "targeted-repair"


def _repair_data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "targeted-repair"


def _normalized(values: list[float], *, name: str) -> list[float]:
    if not values or any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{name} must contain finite nonnegative values")
    total = float(sum(values))
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return [float(value / total) for value in values]


def triggered_repair_target(
    parent_probabilities: list[float],
    oracle_probabilities: list[float],
    normalized_regrets: list[float],
    candidate_validity: list[bool],
    *,
    role: str,
    expected_regret_threshold: float,
    repair_lambda_floor: float,
    repair_lambda_ceiling: float,
    invalid_argmax_lambda: float,
    trigger_expected_regret: float | None = None,
    trigger_invalid_argmax: bool | None = None,
) -> dict[str, Any]:
    """Return an anchor-preserving target and its registered repair strength."""

    if role not in {"anchor", "repair"}:
        raise ValueError("Targeted repair role must be anchor or repair")
    if not 0.0 <= expected_regret_threshold < 1.0:
        raise ValueError("Expected-regret threshold must be in [0, 1)")
    if not 0.0 <= repair_lambda_floor <= repair_lambda_ceiling <= 1.0:
        raise ValueError("Repair lambda bounds must satisfy 0 <= floor <= ceiling <= 1")
    if not 0.0 <= invalid_argmax_lambda <= 1.0:
        raise ValueError("Invalid-argmax lambda must be in [0, 1]")
    if not (
        len(parent_probabilities)
        == len(oracle_probabilities)
        == len(normalized_regrets)
        == len(candidate_validity)
        and parent_probabilities
    ):
        raise ValueError("Targeted repair vectors must be aligned and nonempty")
    if any(not 0.0 <= regret <= 1.0 for regret in normalized_regrets):
        raise ValueError("Targeted repair regrets must be normalized to [0, 1]")

    parent = _normalized(parent_probabilities, name="parent probabilities")
    oracle = _normalized(oracle_probabilities, name="oracle probabilities")
    parent_expected = float(np.dot(parent, normalized_regrets))
    oracle_expected = float(np.dot(oracle, normalized_regrets))
    local_invalid = not bool(candidate_validity[int(np.argmax(parent))])
    trigger_regret = (
        parent_expected if trigger_expected_regret is None else float(trigger_expected_regret)
    )
    trigger_invalid = local_invalid if trigger_invalid_argmax is None else trigger_invalid_argmax
    if not 0.0 <= trigger_regret <= 1.0:
        raise ValueError("Trigger regret must be normalized to [0, 1]")

    repair_lambda = 0.0
    if role == "repair" and oracle_expected < parent_expected - 1.0e-12:
        progress = max(
            0.0,
            min(
                1.0,
                (trigger_regret - expected_regret_threshold) / (1.0 - expected_regret_threshold),
            ),
        )
        repair_lambda = (
            repair_lambda_floor + (repair_lambda_ceiling - repair_lambda_floor) * progress
        )
        if trigger_invalid:
            repair_lambda = max(repair_lambda, invalid_argmax_lambda)
        repair_lambda = min(repair_lambda, repair_lambda_ceiling)

    target = [
        (1.0 - repair_lambda) * parent_value + repair_lambda * oracle_value
        for parent_value, oracle_value in zip(parent, oracle, strict=True)
    ]
    target = _normalized(target, name="repair target")
    target_expected = float(np.dot(target, normalized_regrets))
    if target_expected > parent_expected + 1.0e-10:
        raise RuntimeError("A targeted repair target increased simulator-expected regret")
    return {
        "target_probabilities": target,
        "repair_lambda": float(repair_lambda),
        "parent_expected_regret": parent_expected,
        "oracle_expected_regret": oracle_expected,
        "target_expected_regret": target_expected,
        "parent_argmax_valid": not local_invalid,
        "trigger_expected_regret": trigger_regret,
        "trigger_invalid_argmax": bool(trigger_invalid),
    }


def select_targeted_groups(
    summaries: list[dict[str, Any]],
    *,
    repair_count: int,
    anchor_count: int,
) -> dict[str, str]:
    """Select disjoint high-regret repairs and low-regret anchors deterministically."""

    if repair_count <= 0 or anchor_count <= 0:
        raise ValueError("Targeted group counts must be positive")
    if len(summaries) < repair_count + anchor_count:
        raise ValueError("The targeted training pool is smaller than the registered selection")
    ids = [str(value["group_id"]) for value in summaries]
    if len(set(ids)) != len(ids):
        raise ValueError("Targeted group summaries contain duplicate ids")
    repair = sorted(
        summaries,
        key=lambda value: (
            -int(bool(value["invalid_argmax"])),
            -float(value["expected_regret"]),
            str(value["group_id"]),
        ),
    )[:repair_count]
    repair_ids = {str(value["group_id"]) for value in repair}
    anchors = sorted(
        (value for value in summaries if str(value["group_id"]) not in repair_ids),
        key=lambda value: (
            int(bool(value["invalid_argmax"])),
            float(value["expected_regret"]),
            str(value["group_id"]),
        ),
    )[:anchor_count]
    roles = {str(value["group_id"]): "repair" for value in repair}
    roles.update({str(value["group_id"]): "anchor" for value in anchors})
    if len(roles) != repair_count + anchor_count:
        raise RuntimeError("Targeted repair group selection overlapped")
    return roles


def _source_selection(config: ProjectConfig) -> dict[str, Any]:
    path = config.path_for("artifact_dir") / "robust-family-experts" / "selection.json"
    if not path.exists():
        raise RuntimeError("The registered v0.5 selection result is missing")
    result = json.loads(path.read_text(encoding="utf-8"))
    expected = str(config.section("targeted_repair")["source_selection_fingerprint"])
    if not result.get("complete") or result.get("fingerprint") != expected:
        raise RuntimeError("The v0.5 discovery result changed before v0.6 registration")
    if result.get("confirmation_opened") or result.get("promotion_opened"):
        raise RuntimeError("v0.5 downstream isolation changed")
    return result


def _anchor_mapping_without_paths(selection: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source = selection["route_options"]["boltzmann-mean"]["mapping"]
    return {
        family_id: {
            "slug": str(route["slug"]),
            "arm": str(route["arm"]),
            "checkpoint_name": str(route["checkpoint_name"]),
            "checkpoint_sha256": str(route["checkpoint_sha256"]),
            "update": int(route["update"]),
        }
        for family_id, route in sorted(source.items())
    }


def _discovery_target_families(
    settings: dict[str, Any], selection: dict[str, Any]
) -> tuple[list[str], dict[str, float]]:
    option = selection["route_options"]["boltzmann-mean"]
    scores = {
        str(family): float(value)
        for family, value in option["vs_parent"]["pooled_comparison"][
            "parent_family_regret"
        ].items()
    }
    excluded = (
        set(str(value) for value in option["nonparent_families"])
        if settings["discovery"]["exclude_existing_anchor_experts"]
        else set()
    )
    count = int(settings["discovery"]["family_count"])
    eligible = [family for family in scores if family not in excluded]
    selected = sorted(eligible, key=lambda family: (-scores[family], family))[:count]
    if len(selected) != count:
        raise RuntimeError("The v0.6 discovery source has too few unresolved families")
    return selected, scores


def _training_pool_scenarios(
    settings: dict[str, Any], target_families: list[str]
) -> list[Scenario]:
    pool = dict(settings["training_pool"])
    all_scenarios = build_blueprints(
        {str(pool["split"]): int(pool["pool_count"])},
        seed=int(pool["seed"]),
        active_search=True,
    )
    targets = set(target_families)
    scenarios = sorted(
        (scenario for scenario in all_scenarios if scenario.family_id in targets),
        key=lambda scenario: (scenario.family_id, scenario.blueprint_id),
    )
    required = int(pool["repair_groups_per_family"]) + int(pool["anchor_groups_per_family"])
    for family_id in target_families:
        available = sum(scenario.family_id == family_id for scenario in scenarios)
        if available < required:
            raise RuntimeError(
                f"Targeted training pool has {available} {family_id} blueprints, below {required}"
            )
    return scenarios


def _registered_scenarios(
    settings: dict[str, Any],
    *,
    settings_name: str,
    shard_settings: dict[str, Any],
    target_families: list[str],
) -> list[Scenario]:
    if settings_name == "training_pool":
        return _training_pool_scenarios(settings, target_families)
    return _balanced_family_scenarios(shard_settings, settings_name=settings_name)


def _resolved_anchor_mapping(
    config: ProjectConfig, contract: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    source = _source_selection(config)["route_options"]["boltzmann-mean"]["mapping"]
    resolved: dict[str, dict[str, Any]] = {}
    for family_id, frozen in sorted(contract["anchor_mapping"].items()):
        route = dict(source[family_id])
        if any(
            (str(route[key]) if key != "update" else int(route[key]))
            != (str(frozen[key]) if key != "update" else int(frozen[key]))
            for key in ("slug", "arm", "checkpoint_name", "checkpoint_sha256", "update")
        ):
            raise RuntimeError(f"The v0.5 anchor route changed for {family_id}")
        path = Path(str(route["checkpoint_path"]))
        weights = path / "adapters.safetensors" if path.is_dir() else path
        if not weights.exists() or sha256_file(weights) != frozen["checkpoint_sha256"]:
            raise RuntimeError(f"The v0.5 anchor weights changed for {family_id}")
        route["stage"] = "v0.5-anchor"
        resolved[family_id] = route
    return resolved


def prepare_targeted_repair_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("targeted_repair"))
    root = _repair_root(config)
    root.mkdir(parents=True, exist_ok=True)
    selection = _source_selection(config)
    target_families, discovery_scores = _discovery_target_families(settings, selection)
    anchor_mapping = _anchor_mapping_without_paths(selection)
    if any(anchor_mapping[family]["arm"] != "parent" for family in target_families):
        raise RuntimeError("A v0.6 target family is already changed by the anchor route")

    named_settings = {
        "training_pool": settings["training_pool"],
        **{
            f"selection_fold_{index}": value
            for index, value in enumerate(settings["selection_folds"], start=1)
        },
        "confirmation_shard": settings["confirmation_shard"],
        "promotion_shard": settings["promotion_shard"],
        "final_shard": settings["final_shard"],
    }
    blueprint_contracts: dict[str, dict[str, Any]] = {}
    seen_ids: dict[str, str] = {}
    for name, shard_settings in named_settings.items():
        scenarios = _registered_scenarios(
            settings,
            settings_name=name,
            shard_settings=dict(shard_settings),
            target_families=target_families,
        )
        ids = [scenario.blueprint_id for scenario in scenarios]
        overlap = sorted(set(ids) & set(seen_ids))
        if overlap:
            raise RuntimeError(
                f"Targeted repair blueprints overlap: {name} with {seen_ids[overlap[0]]}"
            )
        seen_ids.update({blueprint_id: name for blueprint_id in ids})
        blueprint_contracts[name] = {
            "split": str(shard_settings["split"]),
            "seed": int(shard_settings["seed"]),
            "pool_count": int(shard_settings["pool_count"]),
            "selected_per_family": int(shard_settings["selected_per_family"]),
            "count": len(scenarios),
            "family_counts": {
                family.family_id: sum(
                    scenario.family_id == family.family_id for scenario in scenarios
                )
                for family in FAMILIES
                if name != "training_pool" or family.family_id in target_families
            },
            "blueprint_sha256": canonical_hash([scenario.to_dict() for scenario in scenarios]),
        }

    contract = {
        "schema_version": 1,
        "method": "DGP-Regret triggered anchor repair hypothesis",
        "method_version": int(settings["method_version"]),
        "settings": settings,
        "source_selection_fingerprint": str(selection["fingerprint"]),
        "anchor_mapping": anchor_mapping,
        "target_families": target_families,
        "discovery_family_regret": discovery_scores,
        "discovery_data_reused_for_v0_6_evaluation": False,
        "parent_adapter_sha256": str(anchor_mapping[target_families[0]]["checkpoint_sha256"]),
        "base_model": config.sources["models"]["research_base_mlx_4bit"],
        "blueprint_contracts": blueprint_contracts,
        "pairwise_blueprint_overlap": 0,
        "training_started_at_registration": False,
        "confirmation_opened": False,
        "promotion_opened": False,
        "final_blueprints_registered": True,
        "final_simulations_opened": False,
        "final_scores_opened": False,
        "claim_boundary": (
            "Active fine-tuning, anchor replay, KL-style preservation, simulation-based decision "
            "learning, and low-rank continual adaptation have prior art. This contract tests a "
            "project-specific regret trigger with no first-in-literature claim."
        ),
    }
    contract["fingerprint"] = canonical_hash(contract)
    path = root / "contract.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("The targeted-repair contract is immutable")
        return existing
    write_json(path, contract)
    write_json(
        config.root / "reports" / "evolve" / "targeted-repair-contract.json",
        contract,
    )
    return contract


def _simulate_registered_surface(
    config: ProjectConfig,
    *,
    contract: dict[str, Any],
    settings_name: str,
    shard_settings: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    settings = dict(config.section("targeted_repair"))
    data_root = _repair_data_root(config)
    surface_path = data_root / "surfaces" / f"{settings_name}.jsonl"
    manifest_path = surface_path.with_suffix(".manifest.json")
    scenarios = _registered_scenarios(
        settings,
        settings_name=settings_name,
        shard_settings=shard_settings,
        target_families=[str(value) for value in contract["target_families"]],
    )
    blueprint_sha = canonical_hash([scenario.to_dict() for scenario in scenarios])
    registered = contract["blueprint_contracts"][settings_name]
    if blueprint_sha != registered["blueprint_sha256"]:
        raise RuntimeError(f"Registered targeted blueprint hash changed for {settings_name}")
    simulation_settings = config.section("stats_data")
    fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "settings_name": settings_name,
            "blueprints": blueprint_sha,
            "simulation": {
                key: simulation_settings[key]
                for key in (
                    "initial_repetitions",
                    "escalation_repetitions",
                    "ranking_uncertainty_margin",
                    "regret_temperature",
                )
            },
            "simulator_version": 1,
        }
    )
    if surface_path.exists() or manifest_path.exists():
        if not surface_path.exists() or not manifest_path.exists():
            raise RuntimeError(f"The targeted {settings_name} surface is incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint or manifest.get("sha256") != sha256_file(
            surface_path
        ):
            raise RuntimeError(f"The targeted {settings_name} surface is immutable")
        return manifest, list(read_jsonl(surface_path))
    simulations = [
        simulate_scenario(
            scenario,
            initial_repetitions=int(simulation_settings["initial_repetitions"]),
            escalation_repetitions=[
                int(value) for value in simulation_settings["escalation_repetitions"]
            ],
            uncertainty_margin=float(simulation_settings["ranking_uncertainty_margin"]),
            temperature=float(simulation_settings["regret_temperature"]),
        )
        for scenario in scenarios
    ]
    write_jsonl(surface_path, simulations)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "settings_name": settings_name,
        "count": len(simulations),
        "blueprint_sha256": blueprint_sha,
        "sha256": sha256_file(surface_path),
        "used_for_training": settings_name == "training_pool",
        "used_for_selection": settings_name.startswith("selection_fold_"),
        "confirmation_opened": False,
        "promotion_opened": False,
        "final_opened": False,
    }
    write_json(manifest_path, manifest)
    return manifest, simulations


def _candidate_validity(record: dict[str, Any], simulation: dict[str, Any]) -> list[bool]:
    by_method = {
        str(value["method_id"]): bool(value["valid"]) for value in simulation["candidates"]
    }
    by_method["needs_clarification"] = False
    return [
        bool(by_method[str(method_id)]) for method_id in record["metadata"]["candidate_method_ids"]
    ]


def _parent_training_cache(
    config: ProjectConfig,
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    simulations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    projection = dict(config.section("policy_projection"))
    simulation_by_id = {str(value["scenario"]["blueprint_id"]): value for value in simulations}
    source_rows = [
        {
            "blueprint_id": str(record["metadata"]["blueprint_id"]),
            "language": str(record["metadata"]["language"]),
            "view": str(record["metadata"]["view"]),
            "candidate_method_ids": list(record["metadata"]["candidate_method_ids"]),
            "simulator_fingerprint": str(record["metadata"]["simulator_fingerprint"]),
        }
        for record in records
    ]
    parent = json.loads(
        (config.path_for("parent_artifact_dir") / "selected.json").read_text(encoding="utf-8")
    )
    parent_sha = sha256_file(Path(str(parent["adapter_path"])) / "adapters.safetensors")
    fingerprint = canonical_hash(
        {
            "parent_adapter_sha256": parent_sha,
            "base_model": config.sources["models"]["research_base_mlx_4bit"],
            "source_rows": source_rows,
            "projection": {
                key: projection[key]
                for key in (
                    "oracle_temperature",
                    "projection_temperature",
                    "step_size",
                    "exploration_mass",
                )
            },
            "scorer_version": 1,
        }
    )
    root = _repair_root(config)
    path = root / "parent-training-cache.jsonl"
    manifest_path = root / "parent-training-cache.manifest.json"
    if path.exists() or manifest_path.exists():
        if not path.exists() or not manifest_path.exists():
            raise RuntimeError("The targeted parent-training cache is incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("fingerprint") != fingerprint
            or manifest.get("sha256") != sha256_file(path)
            or int(manifest.get("count", -1)) != len(records)
        ):
            raise RuntimeError("The targeted parent-training cache changed")
        cached = list(read_jsonl(path))
        if [value["source"] for value in cached] != source_rows:
            raise RuntimeError("The targeted parent-training cache row order changed")
        return cached

    cached: list[dict[str, Any]] = []
    for record, source in zip(records, source_rows, strict=True):
        simulation = simulation_by_id[source["blueprint_id"]]
        parent_probabilities = _selector_menu_probabilities(model, tokenizer, record)
        regrets = _candidate_regrets(record, simulation)
        validity = _candidate_validity(record, simulation)
        oracle = project_policy(
            parent_probabilities,
            list(record["metadata"]["method_probabilities"]),
            oracle_temperature=float(projection["oracle_temperature"]),
            projection_temperature=float(projection["projection_temperature"]),
            step_size=float(projection["step_size"]),
            exploration_mass=float(projection["exploration_mass"]),
        )["target_probabilities"]
        cached.append(
            {
                "source": source,
                "parent": [float(value) for value in parent_probabilities],
                "oracle": [float(value) for value in oracle],
                "regrets": [float(value) for value in regrets],
                "validity": [bool(value) for value in validity],
            }
        )
    write_jsonl(path, cached)
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "complete": True,
            "fingerprint": fingerprint,
            "count": len(cached),
            "sha256": sha256_file(path),
            "parent_adapter_sha256": parent_sha,
        },
    )
    return cached


def _build_targeted_rows(
    config: ProjectConfig,
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    simulations: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    settings = dict(config.section("targeted_repair"))
    trigger = dict(settings["trigger"])
    prepared: list[dict[str, Any]] = []
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cached = _parent_training_cache(config, model, tokenizer, records, simulations)
    for record, values in zip(records, cached, strict=True):
        parent = [float(value) for value in values["parent"]]
        regrets = [float(value) for value in values["regrets"]]
        validity = [bool(value) for value in values["validity"]]
        oracle = [float(value) for value in values["oracle"]]
        item = {
            "record": record,
            "parent": [float(value) for value in parent],
            "oracle": [float(value) for value in oracle],
            "regrets": regrets,
            "validity": validity,
            "parent_expected_regret": float(np.dot(parent, regrets)),
            "invalid_argmax": not validity[int(np.argmax(parent))],
        }
        prepared.append(item)
        by_group[str(record["metadata"]["semantic_group_id"])].append(item)

    group_summaries: dict[str, dict[str, Any]] = {}
    summaries_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group_id, items in sorted(by_group.items()):
        family_ids = {str(item["record"]["metadata"]["family_id"]) for item in items}
        if len(items) != 4 or len(family_ids) != 1:
            raise RuntimeError("Targeted data lost a four-view semantic group")
        summary = {
            "group_id": group_id,
            "family_id": family_ids.pop(),
            "expected_regret": max(float(item["parent_expected_regret"]) for item in items),
            "invalid_argmax": any(bool(item["invalid_argmax"]) for item in items),
        }
        group_summaries[group_id] = summary
        summaries_by_family[str(summary["family_id"])].append(summary)

    roles: dict[str, str] = {}
    pool = dict(settings["training_pool"])
    for family_id in settings.get("target_families", []):
        # The contract owns target-family discovery; this branch is never used
        # with the registered configuration and exists only for explicit tests.
        if family_id not in summaries_by_family:
            raise RuntimeError(f"Targeted rows are missing {family_id}")
    for _family_id, summaries in sorted(summaries_by_family.items()):
        roles.update(
            select_targeted_groups(
                summaries,
                repair_count=int(pool["repair_groups_per_family"]),
                anchor_count=int(pool["anchor_groups_per_family"]),
            )
        )

    output = {str(arm): [] for arm in settings["arms"]}
    target_improvements: dict[str, list[float]] = defaultdict(list)
    lambdas: list[float] = []
    for item in prepared:
        source = item["record"]
        group_id = str(source["metadata"]["semantic_group_id"])
        role = roles.get(group_id)
        if role is None:
            continue
        summary = group_summaries[group_id]
        repair = triggered_repair_target(
            item["parent"],
            item["oracle"],
            item["regrets"],
            item["validity"],
            role=role,
            expected_regret_threshold=float(trigger["expected_regret_threshold"]),
            repair_lambda_floor=float(trigger["repair_lambda_floor"]),
            repair_lambda_ceiling=float(trigger["repair_lambda_ceiling"]),
            invalid_argmax_lambda=float(trigger["invalid_argmax_lambda"]),
            trigger_expected_regret=float(summary["expected_regret"]),
            trigger_invalid_argmax=bool(summary["invalid_argmax"]),
        )
        lambdas.append(float(repair["repair_lambda"]))
        for arm in settings["arms"]:
            row = copy.deepcopy(source)
            if arm == "boltzmann-replay":
                target = item["oracle"]
                applied_lambda = 1.0
            elif arm == "triggered-repair":
                target = repair["target_probabilities"]
                applied_lambda = float(repair["repair_lambda"])
            else:
                raise RuntimeError(f"Unknown targeted repair arm: {arm}")
            target_expected = float(np.dot(target, item["regrets"]))
            if target_expected > float(item["parent_expected_regret"]) + 1.0e-10:
                target = item["parent"]
                target_expected = float(item["parent_expected_regret"])
                applied_lambda = 0.0
            row["metadata"]["method_probabilities"] = [float(value) for value in target]
            row["metadata"]["parent_method_probabilities"] = [
                float(value) for value in item["parent"]
            ]
            row["metadata"]["oracle_method_probabilities"] = [
                float(value) for value in item["oracle"]
            ]
            row["metadata"]["candidate_normalized_regrets"] = [
                float(value) for value in item["regrets"]
            ]
            row["metadata"]["candidate_validity"] = [bool(value) for value in item["validity"]]
            row["metadata"]["targeted_repair_arm"] = str(arm)
            row["metadata"]["targeted_repair_role"] = role
            row["metadata"]["repair_lambda"] = applied_lambda
            row["metadata"]["parent_expected_regret"] = float(item["parent_expected_regret"])
            row["metadata"]["group_trigger_expected_regret"] = float(summary["expected_regret"])
            row["metadata"]["group_trigger_invalid_argmax"] = bool(summary["invalid_argmax"])
            row["metadata"]["target_expected_regret"] = target_expected
            output[str(arm)].append(row)
            target_improvements[str(arm)].append(
                float(item["parent_expected_regret"]) - target_expected
            )

    expected_groups = len(summaries_by_family) * int(pool["selected_per_family"])
    if len(roles) != expected_groups:
        raise RuntimeError("Targeted repair selected an unexpected number of semantic groups")
    role_counts = {
        role: sum(value == role for value in roles.values()) for role in ("repair", "anchor")
    }
    audit = {
        "candidate_pool_records": len(records),
        "candidate_pool_semantic_groups": len(by_group),
        "selected_semantic_groups": len(roles),
        "selected_role_counts": role_counts,
        "selected_group_sha256": canonical_hash(sorted(roles.items())),
        "mean_triggered_repair_lambda": float(np.mean(lambdas)),
        "triggered_repair_lambda_range": [float(min(lambdas)), float(max(lambdas))],
        "mean_target_expected_regret_improvement": {
            arm: float(np.mean(values)) for arm, values in sorted(target_improvements.items())
        },
        "language_gradient_ratio_preserved_within_every_group": True,
        "matched_rows_and_loss_weights": True,
    }
    return output, audit


def _targeted_downstream_paths(config: ProjectConfig) -> list[Path]:
    data_root = _repair_data_root(config)
    artifact_root = _repair_root(config)
    return [
        artifact_root / "selection.json",
        artifact_root / "confirmation.json",
        artifact_root / "promotion.json",
        artifact_root / "final.json",
        data_root / "surfaces" / "confirmation_shard.jsonl",
        data_root / "surfaces" / "confirmation_shard.manifest.json",
        data_root / "surfaces" / "promotion_shard.jsonl",
        data_root / "surfaces" / "promotion_shard.manifest.json",
        data_root / "surfaces" / "final_shard.jsonl",
        data_root / "surfaces" / "final_shard.manifest.json",
    ]


def _parent_path(config: ProjectConfig, expected_sha256: str) -> Path:
    selected = json.loads(
        (config.path_for("parent_artifact_dir") / "selected.json").read_text(encoding="utf-8")
    )
    parent = Path(str(selected["adapter_path"]))
    weights = parent / "adapters.safetensors"
    if not weights.exists() or sha256_file(weights) != expected_sha256:
        raise RuntimeError("The v0.6 parent adapter changed")
    return parent


def prepare_targeted_repair_data(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    contract = prepare_targeted_repair_contract(config)
    settings = dict(config.section("targeted_repair"))
    data_root = _repair_data_root(config)
    artifact_root = _repair_root(config)
    status_path = artifact_root / "data-status.json"
    fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "parent": contract["parent_adapter_sha256"],
            "builder_version": 4,
        }
    )
    if status_path.exists() and not force:
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            return existing
        raise RuntimeError("Targeted repair data fingerprint changed")
    if force and any(path.exists() for path in _targeted_downstream_paths(config)):
        raise RuntimeError("Cannot rebuild targeted data after selection or a downstream shard")
    forbidden = [
        data_root / "surfaces" / "confirmation_shard.jsonl",
        data_root / "surfaces" / "promotion_shard.jsonl",
        data_root / "surfaces" / "final_shard.jsonl",
    ]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("A downstream targeted-repair shard was opened before data preparation")

    manifests: dict[str, dict[str, Any]] = {}
    manifests["training_pool"], training_simulations = _simulate_registered_surface(
        config,
        contract=contract,
        settings_name="training_pool",
        shard_settings=dict(settings["training_pool"]),
    )
    for index, fold_settings in enumerate(settings["selection_folds"], start=1):
        name = f"selection_fold_{index}"
        manifests[name], _ = _simulate_registered_surface(
            config,
            contract=contract,
            settings_name=name,
            shard_settings=dict(fold_settings),
        )

    parent = _parent_path(config, str(contract["parent_adapter_sha256"]))
    model = tokenizer = None
    previous_limit = mx.set_cache_limit(int(float(settings["cache_limit_gb"]) * 1024**3))
    try:
        mx.random.seed(int(config.section("project")["seed"]))
        np.random.seed(int(config.section("project")["seed"]))
        model, tokenizer = load(
            _stats_snapshot(config),
            adapter_path=str(parent),
            tokenizer_config={"trust_remote_code": True},
        )
        for _, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        model.eval()
        target_rows, target_audit = _build_targeted_rows(
            config,
            model,
            tokenizer,
            _training_records(training_simulations),
            training_simulations,
        )
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        mx.clear_cache()
        mx.set_cache_limit(previous_limit)

    target_hashes: dict[str, str] = {}
    target_counts: dict[str, int] = {}
    for arm, rows in sorted(target_rows.items()):
        path = data_root / "train" / f"{arm}.jsonl"
        write_jsonl(path, rows)
        target_hashes[arm] = sha256_file(path)
        target_counts[arm] = len(rows)
    if len(set(target_counts.values())) != 1:
        raise RuntimeError("Targeted ablation arms received different record counts")
    result = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "contract_fingerprint": contract["fingerprint"],
        "source_selection_fingerprint": contract["source_selection_fingerprint"],
        "parent_adapter_sha256": sha256_file(parent / "adapters.safetensors"),
        "target_families": contract["target_families"],
        "manifests": manifests,
        "target_hashes": target_hashes,
        "target_counts": target_counts,
        "target_audit": target_audit,
        "confirmation_opened": False,
        "promotion_opened": False,
        "final_simulations_opened": False,
        "final_scores_opened": False,
    }
    write_json(status_path, result)
    write_json(
        config.root / "reports" / "evolve" / "targeted-repair-data.json",
        result,
    )
    return result


def _selection_folds(
    config: ProjectConfig,
    family_id: str,
) -> dict[str, dict[str, Any]]:
    settings = dict(config.section("targeted_repair"))
    data_root = _repair_data_root(config)
    folds: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for index, fold_settings in enumerate(settings["selection_folds"], start=1):
        name = f"selection_fold_{index}"
        path = data_root / "surfaces" / f"{name}.jsonl"
        manifest_path = path.with_suffix(".manifest.json")
        if not path.exists() or not manifest_path.exists():
            raise RuntimeError(f"Targeted repair selection fold is missing: {name}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sha256") != sha256_file(path):
            raise RuntimeError(f"Targeted repair selection fold changed: {name}")
        all_rows = list(read_jsonl(path))
        rows = [row for row in all_rows if str(row["scenario"]["family_id"]) == family_id]
        expected = int(fold_settings["selected_per_family"])
        if len(rows) != expected:
            raise RuntimeError(f"{name} has {len(rows)} {family_id} rows instead of {expected}")
        ids = {str(row["scenario"]["blueprint_id"]) for row in rows}
        if len(ids) != len(rows) or seen & ids:
            raise RuntimeError("Targeted selection folds overlap or contain duplicates")
        seen.update(ids)
        folds[name] = {"manifest": manifest, "rows": rows}
    return folds


def _training_settings(config: ProjectConfig, settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "method_version": int(settings["method_version"]),
        "updates": int(settings["updates"]),
        "records_per_backward": int(settings["records_per_backward"]),
        "step_l2": float(settings["step_l2"]),
        "cache_limit_gb": float(settings["cache_limit_gb"]),
        "max_seq_length": int(config.section("stats_training")["max_seq_length"]),
        "selector_only": True,
        "dropout_disabled": True,
    }


def _anchor_cache_fingerprint(
    *,
    family_id: str,
    route: dict[str, Any],
    folds: dict[str, dict[str, Any]],
) -> str:
    return canonical_hash(
        {
            "family_id": family_id,
            "anchor_checkpoint": route["checkpoint_sha256"],
            "anchor_arm": route["arm"],
            "folds": {
                name: value["manifest"]["fingerprint"] for name, value in sorted(folds.items())
            },
            "scorer_version": 1,
        }
    )


def _write_anchor_cache(
    config: ProjectConfig,
    *,
    family_id: str,
    route: dict[str, Any],
    folds: dict[str, dict[str, Any]],
    model: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    fingerprint = _anchor_cache_fingerprint(
        family_id=family_id,
        route=route,
        folds=folds,
    )
    path = _repair_root(config) / "anchor-selection" / f"{family_id}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            return existing
        raise RuntimeError(f"Targeted anchor selection cache changed for {family_id}")
    fold_scores = {
        name: _score_loaded_family(model, tokenizer, value["rows"])
        for name, value in sorted(folds.items())
    }
    result = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "family_id": family_id,
        "anchor_arm": str(route["arm"]),
        "anchor_checkpoint_sha256": str(route["checkpoint_sha256"]),
        "fold_scores": fold_scores,
        "fold_metrics": {
            name: _family_metrics(score, family_id) for name, score in sorted(fold_scores.items())
        },
    }
    write_json(path, result)
    return result


def _read_anchor_cache(config: ProjectConfig, family_id: str) -> dict[str, Any]:
    path = _repair_root(config) / "anchor-selection" / f"{family_id}.json"
    if not path.exists():
        raise RuntimeError(f"Targeted anchor cache is missing for {family_id}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("complete") or result.get("family_id") != family_id:
        raise RuntimeError(f"Targeted anchor cache is incomplete for {family_id}")
    return result


def run_targeted_repair_arm(
    config: ProjectConfig,
    *,
    family_id: str,
    arm: str,
    force: bool = False,
) -> dict[str, Any]:
    settings = dict(config.section("targeted_repair"))
    arms = [str(value) for value in settings["arms"]]
    if arm not in arms:
        raise ValueError(f"Unknown targeted repair arm: {arm}")
    data_status = prepare_targeted_repair_data(config, force=False)
    target_families = [str(value) for value in data_status["target_families"]]
    if family_id not in target_families:
        raise ValueError(f"Family is not registered for targeted repair: {family_id}")
    if force and any(path.exists() for path in _targeted_downstream_paths(config)):
        raise RuntimeError("Cannot force-retrain targeted experts after selection")

    contract = prepare_targeted_repair_contract(config)
    anchor_mapping = _resolved_anchor_mapping(config, contract)
    anchor_route = anchor_mapping[family_id]
    if anchor_route["arm"] != "parent":
        raise RuntimeError("Targeted training must begin from an unchanged anchor family")
    data_root = _repair_data_root(config)
    artifact_root = _repair_root(config)
    artifact_dir = artifact_root / "arms" / arm / family_id
    train_path = data_root / "train" / f"{arm}.jsonl"
    folds = _selection_folds(config, family_id)
    training_settings = _training_settings(config, settings)
    fingerprint = canonical_hash(
        {
            "contract": data_status["contract_fingerprint"],
            "family_id": family_id,
            "arm": arm,
            "settings": training_settings,
            "parent": data_status["parent_adapter_sha256"],
            "train": sha256_file(train_path),
            "folds": {
                name: value["manifest"]["fingerprint"] for name, value in sorted(folds.items())
            },
            "trainer_version": 1,
        }
    )
    status_path = artifact_dir / "status.json"
    progress_path = artifact_dir / "progress.json"
    if status_path.exists() and not force:
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            return existing
        raise RuntimeError(f"Targeted expert fingerprint changed for {arm}/{family_id}")

    resume: dict[str, Any] | None = None
    if progress_path.exists() and not force:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Targeted partial fingerprint changed for {arm}/{family_id}")
        completed = int(progress.get("completed_updates", 0))
        checkpoint = Path(str(progress.get("last_checkpoint_path", "")))
        if completed and (
            not checkpoint.exists()
            or sha256_file(checkpoint) != progress.get("last_checkpoint_sha256")
        ):
            raise RuntimeError(f"Targeted resume checkpoint changed for {arm}/{family_id}")
        resume = progress

    artifact_dir.mkdir(parents=True, exist_ok=True)
    if force or resume is None:
        for path in [
            artifact_dir / "adapter_config.json",
            status_path,
            progress_path,
            *artifact_dir.glob("update-*.safetensors"),
        ]:
            if path.exists():
                path.unlink()

    parent = _parent_path(config, str(data_status["parent_adapter_sha256"]))
    parent_weights = parent / "adapters.safetensors"
    all_train_rows = list(read_jsonl(train_path))
    train_rows = [row for row in all_train_rows if str(row["metadata"]["family_id"]) == family_id]
    expected_rows = int(settings["training_pool"]["selected_per_family"]) * 4
    if len(train_rows) != expected_rows:
        raise RuntimeError(f"Targeted training coverage changed for {arm}/{family_id}")
    if {str(row["metadata"]["targeted_repair_arm"]) for row in train_rows} != {arm}:
        raise RuntimeError("A targeted expert received rows from another ablation arm")
    groups = _diagnostic_groups(train_rows, groups_per_family=10**9)
    if len(groups) != int(settings["training_pool"]["selected_per_family"]):
        raise RuntimeError("Targeted semantic-group coverage changed")
    records_per_backward = int(settings["records_per_backward"])
    if 4 % records_per_backward:
        raise RuntimeError("Targeted records_per_backward must divide a four-row group")

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
        _enable_gradient_checkpointing_once(model)
        for _, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        model.eval()
        anchor_cache = _write_anchor_cache(
            config,
            family_id=family_id,
            route=anchor_route,
            folds=folds,
            model=model,
            tokenizer=tokenizer,
        )
        checkpoints: list[dict[str, Any]] = (
            [dict(value) for value in resume["checkpoints"]]
            if resume
            else [
                {
                    "name": "anchor",
                    "path": str(parent_weights),
                    "sha256": sha256_file(parent_weights),
                    "fold_metrics": anchor_cache["fold_metrics"],
                    "anchor_cache_fingerprint": anchor_cache["fingerprint"],
                }
            ]
        )
        update_history = [dict(value) for value in resume["update_history"]] if resume else []
        if resume and int(resume.get("completed_updates", 0)):
            model.load_weights(str(resume["last_checkpoint_path"]), strict=False)
        parameter_names: list[str] | None = None
        first_update = int(resume.get("completed_updates", 0)) + 1 if resume else 1
        for update_index in range(first_update, int(settings["updates"]) + 1):
            print(
                f"Targeted {arm}/{family_id}: gradient update {update_index}/{settings['updates']}",
                file=sys.stderr,
                flush=True,
            )
            model.train()
            for _, module in model.named_modules():
                if isinstance(module, nn.Dropout):
                    module.eval()
            gradients, family_names, names, diagnostics = _family_gradient_matrix(
                model,
                tokenizer,
                groups,
                records_per_backward=records_per_backward,
                max_seq_length=int(config.section("stats_training")["max_seq_length"]),
            )
            if family_names != [family_id] or gradients.shape[0] != 1:
                raise RuntimeError("Targeted gradient aggregation crossed family boundaries")
            if parameter_names is None:
                parameter_names = names
            elif parameter_names != names:
                raise RuntimeError("Targeted parameter order changed between updates")
            applied = _apply_flat_update(
                model,
                gradients[0],
                parameter_names,
                step_l2=float(settings["step_l2"]),
            )
            checkpoint_path = artifact_dir / f"update-{update_index:02d}.safetensors"
            mx.save_safetensors(
                str(checkpoint_path),
                dict(tree_flatten(model.trainable_parameters())),
            )
            model.eval()
            print(
                f"Targeted {arm}/{family_id}: scoring three selection folds",
                file=sys.stderr,
                flush=True,
            )
            fold_scores = {
                name: _score_loaded_family(model, tokenizer, value["rows"])
                for name, value in sorted(folds.items())
            }
            fold_metrics = {
                name: _family_metrics(score, family_id)
                for name, score in sorted(fold_scores.items())
            }
            checkpoints.append(
                {
                    "name": f"update-{update_index:02d}",
                    "path": str(checkpoint_path),
                    "sha256": sha256_file(checkpoint_path),
                    "fold_scores": fold_scores,
                    "fold_metrics": fold_metrics,
                }
            )
            update_history.append(
                {
                    "update": update_index,
                    "gradient_diagnostics": diagnostics,
                    "update_norms": applied,
                    "fold_metrics": fold_metrics,
                }
            )
            write_json(
                progress_path,
                {
                    "schema_version": 1,
                    "fingerprint": fingerprint,
                    "complete": False,
                    "arm": arm,
                    "family_id": family_id,
                    "completed_updates": update_index,
                    "last_checkpoint_path": str(checkpoint_path),
                    "last_checkpoint_sha256": sha256_file(checkpoint_path),
                    "checkpoints": checkpoints,
                    "update_history": update_history,
                },
            )
            del gradients
            gc.collect()
            mx.clear_cache()

        adapter_config = _adapter_config_for_child(
            config,
            parent,
            artifact_dir,
            cycle=8,
            arm=f"targeted-repair:{arm}:{family_id}",
        )
        adapter_config.setdefault("stats", {}).update(
            {
                "method": "DGP-Regret triggered anchor repair",
                "method_version": int(settings["method_version"]),
                "family_id": family_id,
                "targeted_repair_arm": arm,
                "dropout_disabled": True,
                "promotion_status": "development-only",
            }
        )
        write_json(artifact_dir / "adapter_config.json", adapter_config)
        updates = int(settings["updates"])
        final_checkpoint = artifact_dir / f"update-{updates:02d}.safetensors"
        status = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "complete": True,
            "arm": arm,
            "family_id": family_id,
            "parent_adapter_path": str(parent),
            "parent_adapter_sha256": sha256_file(parent_weights),
            "source_train_sha256": sha256_file(train_path),
            "records": len(train_rows),
            "semantic_groups": len(groups),
            "updates": updates,
            "records_per_backward": records_per_backward,
            "backward_record_exposures": len(train_rows) * updates,
            "backward_calls": len(groups) * (4 // records_per_backward) * updates,
            "dropout_disabled": True,
            "selection_folds": {
                name: value["manifest"]["fingerprint"] for name, value in sorted(folds.items())
            },
            "selection_dgps_per_fold": {
                name: len(value["rows"]) for name, value in sorted(folds.items())
            },
            "checkpoints": checkpoints,
            "update_history": update_history,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 4),
            "confirmation_opened": False,
            "promotion_opened": False,
            "final_simulations_opened": False,
            "final_scores_opened": False,
        }
        write_json(status_path, status)
        write_json(
            progress_path,
            {
                "schema_version": 1,
                "fingerprint": fingerprint,
                "complete": True,
                "arm": arm,
                "family_id": family_id,
                "completed_updates": updates,
                "last_checkpoint_path": str(final_checkpoint),
                "last_checkpoint_sha256": sha256_file(final_checkpoint),
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


def run_targeted_repair_training(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    settings = dict(config.section("targeted_repair"))
    data_status = prepare_targeted_repair_data(config, force=False)
    artifact_root = _repair_root(config)
    downstream = [path for path in _targeted_downstream_paths(config) if path.exists()]
    training_path = artifact_root / "training-status.json"
    if force and downstream:
        raise RuntimeError("Cannot force-retrain targeted experts after selection")
    if downstream and not force:
        if training_path.exists():
            existing = json.loads(training_path.read_text(encoding="utf-8"))
            if existing.get("complete") and existing.get("matched_backward_compute"):
                return existing
        raise RuntimeError("Targeted downstream state exists without valid training status")

    arms = [str(value) for value in settings["arms"]]
    family_ids = [str(value) for value in data_status["target_families"]]
    statuses: dict[str, dict[str, dict[str, Any]]] = {}
    total = len(arms) * len(family_ids)
    current = 0
    for arm in arms:
        statuses[arm] = {}
        for family_id in family_ids:
            current += 1
            print(
                f"Targeted expert {current}/{total}: {arm}/{family_id}",
                file=sys.stderr,
                flush=True,
            )
            command = [
                sys.executable,
                "-m",
                "charlie_alpha.cli",
                "stats",
                "targeted-repair-arm",
                "--config",
                str(config.path),
                "--arm",
                arm,
                "--family",
                family_id,
            ]
            if force:
                command.append("--force")
            subprocess.run(
                command,
                cwd=config.root,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            statuses[arm][family_id] = json.loads(
                (artifact_root / "arms" / arm / family_id / "status.json").read_text(
                    encoding="utf-8"
                )
            )

    exposures = {
        arm: sum(int(status["backward_record_exposures"]) for status in statuses[arm].values())
        for arm in arms
    }
    backward_calls = {
        arm: sum(int(status["backward_calls"]) for status in statuses[arm].values()) for arm in arms
    }
    expected_exposures = {
        arm: int(data_status["target_counts"][arm]) * int(settings["updates"]) for arm in arms
    }
    matched = (
        exposures == expected_exposures
        and len(set(exposures.values())) == 1
        and len(set(backward_calls.values())) == 1
    )
    if not matched:
        raise RuntimeError("Targeted ablation arms did not receive matched backward compute")
    fingerprint = canonical_hash(
        {
            "contract": data_status["contract_fingerprint"],
            "experts": {
                arm: {
                    family: status["fingerprint"]
                    for family, status in sorted(statuses[arm].items())
                }
                for arm in arms
            },
            "orchestrator_version": 1,
        }
    )
    result = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "contract_fingerprint": data_status["contract_fingerprint"],
        "arms": arms,
        "families": family_ids,
        "expert_fingerprints": {
            arm: {family: status["fingerprint"] for family, status in sorted(statuses[arm].items())}
            for arm in arms
        },
        "matched_backward_compute": matched,
        "backward_record_exposures": exposures,
        "expected_backward_record_exposures": expected_exposures,
        "backward_calls": backward_calls,
        **lifecycle_open_state(),
    }
    write_json(training_path, result)
    write_json(
        config.root / "reports" / "evolve" / "targeted-repair-training.json",
        result,
    )
    return result


def _load_targeted_statuses(
    config: ProjectConfig,
) -> dict[str, dict[str, dict[str, Any]]]:
    settings = dict(config.section("targeted_repair"))
    data_status = prepare_targeted_repair_data(config, force=False)
    artifact_root = _repair_root(config)
    statuses: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in [str(value) for value in settings["arms"]]:
        statuses[arm] = {}
        for family_id in [str(value) for value in data_status["target_families"]]:
            path = artifact_root / "arms" / arm / family_id / "status.json"
            if not path.exists():
                raise RuntimeError(f"Targeted expert is not trained: {arm}/{family_id}")
            status = json.loads(path.read_text(encoding="utf-8"))
            if (
                not status.get("complete")
                or status.get("arm") != arm
                or status.get("family_id") != family_id
            ):
                raise RuntimeError(f"Targeted expert status is incomplete: {arm}/{family_id}")
            statuses[arm][family_id] = status
    return statuses


def _ensure_all_anchor_caches(config: ProjectConfig) -> dict[str, dict[str, Any]]:
    contract = prepare_targeted_repair_contract(config)
    data_status = prepare_targeted_repair_data(config, force=False)
    mapping = _resolved_anchor_mapping(config, contract)
    parent = _parent_path(config, str(data_status["parent_adapter_sha256"]))
    parent_weights = parent / "adapters.safetensors"
    caches: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for family in sorted(value.family_id for value in FAMILIES):
        folds = _selection_folds(config, family)
        fingerprint = _anchor_cache_fingerprint(
            family_id=family,
            route=mapping[family],
            folds=folds,
        )
        path = _repair_root(config) / "anchor-selection" / f"{family}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("complete") and existing.get("fingerprint") == fingerprint:
                caches[family] = existing
                continue
            raise RuntimeError(f"Targeted anchor selection cache changed for {family}")
        missing.append(family)
    if not missing:
        return caches

    model = tokenizer = None
    caffeinate = _start_caffeinate()
    previous_limit = mx.set_cache_limit(
        int(float(config.section("targeted_repair")["cache_limit_gb"]) * 1024**3)
    )
    try:
        model, tokenizer = load(
            _stats_snapshot(config),
            adapter_path=str(parent),
            tokenizer_config={"trust_remote_code": True},
        )
        model.eval()
        for index, family in enumerate(missing, start=1):
            print(
                f"Targeted anchor cache {index}/{len(missing)}: {family}",
                file=sys.stderr,
                flush=True,
            )
            model.load_weights(str(parent_weights), strict=False)
            route = mapping[family]
            if route["arm"] != "parent":
                model.load_weights(str(route["checkpoint_path"]), strict=False)
            caches[family] = _write_anchor_cache(
                config,
                family_id=family,
                route=route,
                folds=_selection_folds(config, family),
                model=model,
                tokenizer=tokenizer,
            )
            gc.collect()
            mx.clear_cache()
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        mx.clear_cache()
        mx.set_cache_limit(previous_limit)
        if caffeinate is not None and caffeinate.poll() is None:
            caffeinate.terminate()
    return caches


def _select_targeted_mapping(
    config: ProjectConfig,
    statuses: dict[str, dict[str, dict[str, Any]]],
    *,
    arm: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    settings = dict(config.section("targeted_repair"))
    selection = dict(settings["selection"])
    gates = dict(settings["gates"])
    contract = prepare_targeted_repair_contract(config)
    mapping = copy.deepcopy(_resolved_anchor_mapping(config, contract))
    audit: dict[str, Any] = {}
    for family_id in [str(value) for value in contract["target_families"]]:
        anchor_cache = _read_anchor_cache(config, family_id)
        fold_names = sorted(anchor_cache["fold_metrics"])
        anchor_metrics = anchor_cache["fold_metrics"]
        options: list[dict[str, Any]] = []
        granular_rejections = 0
        for checkpoint in statuses[arm][family_id]["checkpoints"]:
            if checkpoint["name"] == "anchor":
                continue
            candidate_metrics = checkpoint["fold_metrics"]
            granular = all(
                _family_noninferior(candidate_metrics[name], anchor_metrics[name], gates)
                for name in fold_names
            )
            if not granular:
                granular_rejections += 1
                continue
            options.append(
                {
                    "arm": arm,
                    "update": int(str(checkpoint["name"]).rsplit("-", 1)[1]),
                    "checkpoint_name": str(checkpoint["name"]),
                    "checkpoint_path": str(checkpoint["path"]),
                    "checkpoint_sha256": str(checkpoint["sha256"]),
                    "parent_fold_regrets": [
                        float(anchor_metrics[name]["normalized_regret"]) for name in fold_names
                    ],
                    "candidate_fold_regrets": [
                        float(candidate_metrics[name]["normalized_regret"]) for name in fold_names
                    ],
                }
            )
        selected = select_crossfit_expert_option(
            options,
            minimum_family_pooled_relative_improvement=float(
                selection["minimum_family_pooled_relative_improvement"]
            ),
            minimum_fold_relative_improvement=float(selection["minimum_fold_relative_improvement"]),
            minimum_improving_folds=int(selection["minimum_improving_folds"]),
            maximum_fold_relative_regression=float(selection["maximum_fold_relative_regression"]),
        )
        if selected is not None:
            mapping[family_id] = {
                "slug": f"targeted-{arm}-{family_id}-u{selected['update']:02d}",
                "stage": "v0.6",
                "arm": arm,
                "checkpoint_name": str(selected["checkpoint_name"]),
                "checkpoint_path": str(selected["checkpoint_path"]),
                "checkpoint_sha256": str(selected["checkpoint_sha256"]),
                "update": int(selected["update"]),
                "crossfit": selected["crossfit"],
            }
        audit[family_id] = {
            "eligible_options_before_crossfit": len(options),
            "granular_rejections": granular_rejections,
            "selected_stage": mapping[family_id].get("stage", "v0.5-anchor"),
            "selected_arm": mapping[family_id]["arm"],
            "selected_checkpoint": mapping[family_id]["checkpoint_name"],
        }
    return mapping, audit


def _checkpoint_for_mapping(
    statuses: dict[str, dict[str, dict[str, Any]]],
    *,
    family_id: str,
    route: dict[str, Any],
) -> dict[str, Any] | None:
    if route.get("stage") != "v0.6":
        return None
    return next(
        checkpoint
        for checkpoint in statuses[str(route["arm"])][family_id]["checkpoints"]
        if checkpoint["name"] == route["checkpoint_name"]
    )


def _route_fold_score(
    config: ProjectConfig,
    statuses: dict[str, dict[str, dict[str, Any]]],
    mapping: dict[str, dict[str, Any]],
    *,
    fold_name: str,
) -> dict[str, Any]:
    language_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for family_id, route in sorted(mapping.items()):
        checkpoint = _checkpoint_for_mapping(
            statuses,
            family_id=family_id,
            route=route,
        )
        score = (
            checkpoint["fold_scores"][fold_name]
            if checkpoint is not None
            else _read_anchor_cache(config, family_id)["fold_scores"][fold_name]
        )
        for language, result in score["languages"].items():
            language_predictions[language].extend(result["predictions"])
    languages = {
        language: _aggregate_predictions(predictions)
        for language, predictions in sorted(language_predictions.items())
    }
    fold_index = int(fold_name.rsplit("_", 1)[1]) - 1
    per_family = int(
        config.section("targeted_repair")["selection_folds"][fold_index]["selected_per_family"]
    )
    expected = per_family * len(FAMILIES)
    if any(int(result["count"]) != expected for result in languages.values()):
        raise RuntimeError(f"Targeted route changed coverage for {fold_name}")
    return {
        "selector": languages["en"],
        "languages": languages,
        "retention": {"accuracy": 1.0},
    }


def _public_selection(result: dict[str, Any]) -> dict[str, Any]:
    public = copy.deepcopy(result)
    for key in ("anchor_mapping", "control_mapping", "candidate_mapping"):
        for route in public[key].values():
            route.pop("checkpoint_path", None)
            route.pop("checkpoint_sha256", None)
    selected = public.get("selected")
    if selected is not None:
        for route in selected["mapping"].values():
            route.pop("checkpoint_path", None)
            route.pop("checkpoint_sha256", None)
    return public


def select_targeted_repair_route(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    settings = dict(config.section("targeted_repair"))
    selection_settings = dict(settings["selection"])
    gates = dict(settings["gates"])
    artifact_root = _repair_root(config)
    training_path = artifact_root / "training-status.json"
    if not training_path.exists():
        raise RuntimeError("Train both targeted repair arms before selection")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if not training.get("complete") or not training.get("matched_backward_compute"):
        raise RuntimeError("Targeted repair training is incomplete or compute-unmatched")
    forbidden = [
        path
        for path in _targeted_downstream_paths(config)
        if path.name != "selection.json" and path.exists()
    ]
    if force and forbidden:
        raise RuntimeError("Cannot rewrite targeted selection after a downstream shard was opened")

    contract = prepare_targeted_repair_contract(config)
    statuses = _load_targeted_statuses(config)
    _ensure_all_anchor_caches(config)
    anchor_mapping = _resolved_anchor_mapping(config, contract)
    control_mapping, control_audit = _select_targeted_mapping(
        config,
        statuses,
        arm="boltzmann-replay",
    )
    candidate_mapping, candidate_audit = _select_targeted_mapping(
        config,
        statuses,
        arm="triggered-repair",
    )
    fold_names = [
        f"selection_fold_{index}" for index in range(1, len(settings["selection_folds"]) + 1)
    ]
    anchor_folds = {
        name: _route_fold_score(config, statuses, anchor_mapping, fold_name=name)
        for name in fold_names
    }
    control_folds = {
        name: _route_fold_score(config, statuses, control_mapping, fold_name=name)
        for name in fold_names
    }
    candidate_folds = {
        name: _route_fold_score(config, statuses, candidate_mapping, fold_name=name)
        for name in fold_names
    }
    seed = int(config.section("project")["seed"])
    control_vs_anchor = _route_selection_summary(
        anchor_folds,
        control_folds,
        gates=gates,
        selection=selection_settings,
        seed=seed,
    )
    candidate_vs_anchor = _route_selection_summary(
        anchor_folds,
        candidate_folds,
        gates=gates,
        selection=selection_settings,
        seed=seed + 1,
    )
    candidate_vs_control = _route_vs_control_summary(
        control_folds,
        candidate_folds,
        gates=gates,
        selection=selection_settings,
        seed=seed + 2,
    )
    candidate_nonanchor = sorted(
        family for family, route in candidate_mapping.items() if route.get("stage") == "v0.6"
    )
    passed = bool(
        candidate_nonanchor and candidate_vs_anchor["passed"] and candidate_vs_control["passed"]
    )
    selected = (
        {
            "name": "triggered-repair",
            "mapping": candidate_mapping,
            "nonanchor_families": candidate_nonanchor,
            "vs_anchor": candidate_vs_anchor,
            "vs_control": candidate_vs_control,
        }
        if passed
        else None
    )
    fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "training": training["fingerprint"],
            "mappings": {
                name: {
                    family: {
                        "stage": route.get("stage", "v0.5-anchor"),
                        "arm": route["arm"],
                        "checkpoint": route["checkpoint_name"],
                        "sha256": route["checkpoint_sha256"],
                    }
                    for family, route in sorted(mapping.items())
                }
                for name, mapping in {
                    "anchor": anchor_mapping,
                    "control": control_mapping,
                    "candidate": candidate_mapping,
                }.items()
            },
            "selection_rules": selection_settings,
            "granular_gates": gates,
            "selector_version": 1,
        }
    )
    selection_path = artifact_root / "selection.json"
    if selection_path.exists() and not force:
        existing = json.loads(selection_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            training = mark_lifecycle_opened(training, "selection_opened")
            write_json(training_path, training)
            write_json(
                config.root / "reports" / "evolve" / "targeted-repair-training.json",
                training,
            )
            return existing
        raise RuntimeError("Targeted repair selection fingerprint changed")
    result = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "method": "DGP-Regret triggered anchor repair hypothesis",
        "matched_backward_compute": True,
        "target_families": contract["target_families"],
        "anchor_mapping": anchor_mapping,
        "control_mapping": control_mapping,
        "candidate_mapping": candidate_mapping,
        "mapping_audits": {
            "boltzmann-replay": control_audit,
            "triggered-repair": candidate_audit,
        },
        "control_nonanchor_families": sorted(
            family for family, route in control_mapping.items() if route.get("stage") == "v0.6"
        ),
        "candidate_nonanchor_families": candidate_nonanchor,
        "control_vs_anchor": control_vs_anchor,
        "candidate_vs_anchor": candidate_vs_anchor,
        "candidate_vs_control": candidate_vs_control,
        "selected": selected,
        "passed": passed,
        "confirmation_opened": False,
        "promotion_opened": False,
        "final_simulations_opened": False,
        "final_scores_opened": False,
        "claim_boundary": (
            "The v0.5 folds chose unresolved families only. All v0.6 training and selection "
            "blueprints are new. Selection evidence remains development evidence."
        ),
    }
    write_json(selection_path, result)
    training = mark_lifecycle_opened(training, "selection_opened")
    write_json(training_path, training)
    write_json(
        config.root / "reports" / "evolve" / "targeted-repair-training.json",
        training,
    )
    write_json(
        config.root / "reports" / "evolve" / "targeted-repair-selection.json",
        _public_selection(result),
    )
    return result

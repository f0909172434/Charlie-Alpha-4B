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
from .stats_calibrate import _surface_comparison
from .stats_catalog import FAMILIES
from .stats_cone import _apply_flat_update, _family_gradient_matrix
from .stats_data import _build_record, _scenario
from .stats_dgp import Scenario, build_blueprints, simulate_scenario
from .stats_evolve import _adapter_config_for_child, _start_caffeinate
from .stats_experts import _score_loaded_family
from .stats_family_router import _paired_bootstrap
from .stats_project import _diagnostic_groups, project_policy
from .stats_route import _aggregate_predictions, _family_metrics, _family_noninferior
from .stats_training import (
    _enable_gradient_checkpointing_once,
    _selector_menu_probabilities,
    _stats_snapshot,
)


def _robust_root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "robust-family-experts"


def _robust_data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "robust-family-experts"


def direct_regret_target(
    parent_probabilities: list[float],
    normalized_regrets: list[float],
    *,
    trust_beta: float,
) -> list[float]:
    if not 0.0 <= trust_beta <= 1.0:
        raise ValueError("Direct-regret trust beta must be in [0, 1]")
    if len(parent_probabilities) != len(normalized_regrets) or not parent_probabilities:
        raise ValueError("Direct-regret targets require aligned, nonempty vectors")
    if any(value < 0.0 for value in parent_probabilities) or any(
        not 0.0 <= value <= 1.0 for value in normalized_regrets
    ):
        raise ValueError("Direct-regret inputs must be valid probabilities and regrets")
    parent_total = sum(parent_probabilities)
    if parent_total <= 0.0:
        raise ValueError("Parent probabilities must have positive mass")
    parent = [value / parent_total for value in parent_probabilities]
    expected = sum(
        probability * regret for probability, regret in zip(parent, normalized_regrets, strict=True)
    )
    # Cross-entropy has logit gradient p-q. This local target makes that
    # gradient exactly beta * p * (regret - E_p[regret]), the gradient of
    # expected simulator regret at the current policy. It stays normalized and
    # nonnegative for beta in [0, 1] and regrets in [0, 1].
    target = [
        probability * (1.0 - trust_beta * (regret - expected))
        for probability, regret in zip(parent, normalized_regrets, strict=True)
    ]
    total = sum(target)
    return [value / total for value in target]


def cvar_group_weights(
    regrets: list[float],
    *,
    tail_fraction: float,
    uniform_floor: float,
) -> list[float]:
    if not regrets:
        raise ValueError("CVaR weighting requires at least one regret")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("CVaR tail fraction must be in (0, 1]")
    if not 0.0 <= uniform_floor <= 1.0:
        raise ValueError("CVaR uniform floor must be in [0, 1]")
    if any(not 0.0 <= value <= 1.0 for value in regrets):
        raise ValueError("CVaR regrets must be normalized to [0, 1]")
    count = max(1, round(len(regrets) * tail_fraction))
    tail = set(sorted(range(len(regrets)), key=lambda index: (-regrets[index], index))[:count])
    uniform = [1.0 / len(regrets)] * len(regrets)
    tail_mass = [float(index in tail) / len(tail) for index in range(len(regrets))]
    weights = [
        uniform_floor * base + (1.0 - uniform_floor) * cvar
        for base, cvar in zip(uniform, tail_mass, strict=True)
    ]
    mean = sum(weights) / len(weights)
    return [value / mean for value in weights]


def crossfit_option_summary(
    parent_regrets: list[float],
    candidate_regrets: list[float],
    *,
    minimum_fold_relative_improvement: float,
    minimum_improving_folds: int,
    maximum_fold_relative_regression: float,
) -> dict[str, Any]:
    if len(parent_regrets) != len(candidate_regrets) or not parent_regrets:
        raise ValueError("Cross-fit selection requires aligned, nonempty folds")
    if not 1 <= minimum_improving_folds <= len(parent_regrets):
        raise ValueError("Cross-fit improving-fold count is outside the fold range")
    relative = [
        (parent - candidate) / parent if parent else 0.0
        for parent, candidate in zip(parent_regrets, candidate_regrets, strict=True)
    ]
    improving = sum(value >= minimum_fold_relative_improvement for value in relative)
    nonregressing = all(value >= -maximum_fold_relative_regression for value in relative)
    pooled_parent = float(np.mean(parent_regrets))
    pooled_candidate = float(np.mean(candidate_regrets))
    pooled_relative = (pooled_parent - pooled_candidate) / pooled_parent if pooled_parent else 0.0
    return {
        "fold_relative_improvements": relative,
        "improving_folds": improving,
        "pooled_parent_regret": pooled_parent,
        "pooled_candidate_regret": pooled_candidate,
        "pooled_relative_improvement": pooled_relative,
        "mean_relative_improvement": float(np.mean(relative)),
        "median_relative_improvement": float(np.median(relative)),
        "worst_fold_relative_improvement": float(min(relative)),
        "passed": improving >= minimum_improving_folds and nonregressing,
    }


def select_crossfit_expert_option(
    options: list[dict[str, Any]],
    *,
    minimum_family_pooled_relative_improvement: float,
    minimum_fold_relative_improvement: float,
    minimum_improving_folds: int,
    maximum_fold_relative_regression: float,
) -> dict[str, Any] | None:
    evaluated: list[dict[str, Any]] = []
    for option in options:
        summary = crossfit_option_summary(
            [float(value) for value in option["parent_fold_regrets"]],
            [float(value) for value in option["candidate_fold_regrets"]],
            minimum_fold_relative_improvement=minimum_fold_relative_improvement,
            minimum_improving_folds=minimum_improving_folds,
            maximum_fold_relative_regression=maximum_fold_relative_regression,
        )
        evaluated.append({**option, "crossfit": summary})
    eligible = [
        option
        for option in evaluated
        if option["crossfit"]["passed"]
        and float(option["crossfit"]["pooled_relative_improvement"])
        >= minimum_family_pooled_relative_improvement
    ]
    if not eligible:
        return None
    arm_order = {"boltzmann-mean": 0, "direct-mean": 1, "direct-cvar": 2}
    return min(
        eligible,
        key=lambda option: (
            -float(option["crossfit"]["worst_fold_relative_improvement"]),
            -float(option["crossfit"]["pooled_relative_improvement"]),
            -float(option["crossfit"]["mean_relative_improvement"]),
            -float(option["crossfit"]["median_relative_improvement"]),
            arm_order.get(str(option["arm"]), 99),
            int(option["update"]),
        ),
    )


def _balanced_family_scenarios(
    settings: dict[str, Any],
    *,
    settings_name: str,
) -> list[Scenario]:
    pool_count = int(settings["pool_count"])
    selected_per_family = int(settings["selected_per_family"])
    scenarios = build_blueprints(
        {str(settings["split"]): pool_count},
        seed=int(settings["seed"]),
        active_search=True,
    )
    by_family: dict[str, list[Scenario]] = {family.family_id: [] for family in FAMILIES}
    for scenario in scenarios:
        by_family[scenario.family_id].append(scenario)
    selected: list[Scenario] = []
    for family_id in sorted(by_family):
        available = sorted(
            by_family[family_id],
            key=lambda scenario: canonical_hash(
                {
                    "contract": settings_name,
                    "blueprint": scenario.blueprint_id,
                }
            ),
        )
        if len(available) < selected_per_family:
            raise RuntimeError(
                f"{settings_name} has {len(available)} {family_id} blueprints, "
                f"below the registered {selected_per_family}"
            )
        selected.extend(available[:selected_per_family])
    return selected


def prepare_robust_expert_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("robust_family_experts"))
    root = _robust_root(config)
    root.mkdir(parents=True, exist_ok=True)
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
    blueprints: dict[str, dict[str, Any]] = {}
    seen_ids: dict[str, str] = {}
    for name, shard_settings in named_settings.items():
        scenarios = _balanced_family_scenarios(shard_settings, settings_name=name)
        ids = [scenario.blueprint_id for scenario in scenarios]
        overlap = sorted(set(ids) & set(seen_ids))
        if overlap:
            raise RuntimeError(
                f"Robust expert blueprint contracts overlap: {name} with {seen_ids[overlap[0]]}"
            )
        seen_ids.update({blueprint_id: name for blueprint_id in ids})
        family_counts = {
            family.family_id: sum(scenario.family_id == family.family_id for scenario in scenarios)
            for family in FAMILIES
        }
        blueprints[name] = {
            "split": str(shard_settings["split"]),
            "seed": int(shard_settings["seed"]),
            "pool_count": int(shard_settings["pool_count"]),
            "selected_per_family": int(shard_settings["selected_per_family"]),
            "count": len(scenarios),
            "family_counts": family_counts,
            "blueprint_sha256": canonical_hash([scenario.to_dict() for scenario in scenarios]),
        }
    contract = {
        "schema_version": 1,
        "method": "DGP-Regret cross-fit robust family-expert hypothesis",
        "method_version": int(settings["method_version"]),
        "settings": settings,
        "parent_adapter_sha256": "e644b7087c00321f16add940997f8809204458ff6bc51c97a795fed32d3e0b16",
        "base_model": config.sources["models"]["research_base_mlx_4bit"],
        "blueprint_contracts": blueprints,
        "pairwise_blueprint_overlap": 0,
        "training_started_at_registration": False,
        "confirmation_opened": False,
        "promotion_opened": False,
        "final_blueprints_registered": True,
        "final_simulations_opened": False,
        "final_scores_opened": False,
        "claim_boundary": (
            "This preregisters a project-specific combination of direct simulator regret, "
            "CVaR-style tail weighting, family LoRA experts, and cross-fit gates. Mixture-of-LoRA, "
            "adaptive routing, robust optimization, and verifier-guided improvement have "
            "prior art; "
            "the contract makes no first-in-literature claim."
        ),
    }
    contract["fingerprint"] = canonical_hash(contract)
    path = root / "contract.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("The robust family-expert contract is immutable")
        return existing
    write_json(path, contract)
    public = dict(contract)
    write_json(config.root / "reports" / "evolve" / "robust-family-experts-contract.json", public)
    return contract


def _simulate_registered_scenarios(
    config: ProjectConfig,
    *,
    contract: dict[str, Any],
    settings_name: str,
    shard_settings: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data_root = _robust_data_root(config)
    surface_path = data_root / "surfaces" / f"{settings_name}.jsonl"
    manifest_path = surface_path.with_suffix(".manifest.json")
    scenarios = _balanced_family_scenarios(shard_settings, settings_name=settings_name)
    registered = contract["blueprint_contracts"][settings_name]
    blueprint_sha = canonical_hash([scenario.to_dict() for scenario in scenarios])
    if blueprint_sha != registered["blueprint_sha256"]:
        raise RuntimeError(f"Registered robust blueprint hash changed for {settings_name}")
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
            raise RuntimeError(f"The robust {settings_name} surface is incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint or manifest.get("sha256") != sha256_file(
            surface_path
        ):
            raise RuntimeError(f"The robust {settings_name} surface is immutable")
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


def _training_records(simulations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for simulation in simulations:
        scenario = _scenario(simulation["scenario"])
        for language, weight, view in (
            ("en", 1.4, "boundary_a"),
            ("en", 1.4, "boundary_b"),
            ("zh_Hant", 0.6, "standard"),
            ("zh_Hans", 0.6, "standard"),
        ):
            record = _build_record(
                scenario,
                simulation,
                language=language,
                loss_weight=weight,
                incomplete=False,
                variant="dgp-regret",
                refined_explanation=None,
                view=view,
            )
            record["metadata"]["robust_expert_view"] = view
            records.append(record)
    return records


def _candidate_regrets(record: dict[str, Any], simulation: dict[str, Any]) -> list[float]:
    by_method = {
        str(value["method_id"]): float(value["normalized_regret"])
        for value in simulation["candidates"]
    }
    by_method["needs_clarification"] = 1.0
    return [by_method[str(method_id)] for method_id in record["metadata"]["candidate_method_ids"]]


def _build_target_rows(
    config: ProjectConfig,
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    simulations: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    settings = dict(config.section("robust_family_experts"))
    projection_settings = dict(config.section("policy_projection"))
    simulation_by_id = {str(value["scenario"]["blueprint_id"]): value for value in simulations}
    prepared: list[dict[str, Any]] = []
    group_regrets: dict[str, list[float]] = defaultdict(list)
    for record in records:
        blueprint_id = str(record["metadata"]["blueprint_id"])
        simulation = simulation_by_id[blueprint_id]
        parent = _selector_menu_probabilities(model, tokenizer, record)
        regrets = _candidate_regrets(record, simulation)
        realized = regrets[int(np.argmax(parent))]
        group_id = str(record["metadata"]["semantic_group_id"])
        group_regrets[group_id].append(realized)
        boltzmann = project_policy(
            parent,
            list(record["metadata"]["method_probabilities"]),
            oracle_temperature=float(projection_settings["oracle_temperature"]),
            projection_temperature=float(projection_settings["projection_temperature"]),
            step_size=float(projection_settings["step_size"]),
            exploration_mass=float(projection_settings["exploration_mass"]),
        )
        prepared.append(
            {
                "record": record,
                "parent": parent,
                "regrets": regrets,
                "boltzmann": boltzmann["target_probabilities"],
            }
        )
    mean_group_regret = {
        group_id: float(np.mean(values)) for group_id, values in group_regrets.items()
    }
    groups_by_family: dict[str, list[str]] = defaultdict(list)
    for item in prepared:
        metadata = item["record"]["metadata"]
        group_id = str(metadata["semantic_group_id"])
        family_id = str(metadata["family_id"])
        if group_id not in groups_by_family[family_id]:
            groups_by_family[family_id].append(group_id)
    cvar_weights: dict[str, float] = {}
    for _family_id, group_ids in sorted(groups_by_family.items()):
        ordered = sorted(group_ids)
        weights = cvar_group_weights(
            [mean_group_regret[group_id] for group_id in ordered],
            tail_fraction=float(settings["cvar_tail_fraction"]),
            uniform_floor=float(settings["cvar_uniform_floor"]),
        )
        cvar_weights.update(zip(ordered, weights, strict=True))

    output = {arm: [] for arm in settings["arms"]}
    target_improvements: dict[str, list[float]] = defaultdict(list)
    for item in prepared:
        source = item["record"]
        parent = item["parent"]
        regrets = item["regrets"]
        direct = direct_regret_target(
            parent,
            regrets,
            trust_beta=float(settings["direct_trust_beta"]),
        )
        parent_expected = float(np.dot(parent, regrets))
        group_id = str(source["metadata"]["semantic_group_id"])
        for arm in settings["arms"]:
            row = copy.deepcopy(source)
            target = item["boltzmann"] if arm == "boltzmann-mean" else direct
            row["metadata"]["method_probabilities"] = [float(value) for value in target]
            row["metadata"]["robust_expert_arm"] = arm
            row["metadata"]["parent_expected_regret"] = parent_expected
            row["metadata"]["target_expected_regret"] = float(np.dot(target, regrets))
            row["metadata"]["cvar_group_weight"] = (
                float(cvar_weights[group_id]) if arm == "direct-cvar" else 1.0
            )
            if arm == "direct-cvar":
                row["metadata"]["loss_weight"] *= float(cvar_weights[group_id])
            output[str(arm)].append(row)
            target_improvements[str(arm)].append(
                parent_expected - float(row["metadata"]["target_expected_regret"])
            )
    audit = {
        "records": len(records),
        "semantic_groups": len(group_regrets),
        "families": len(groups_by_family),
        "mean_parent_realized_regret": float(np.mean(list(mean_group_regret.values()))),
        "mean_target_expected_regret_improvement": {
            arm: float(np.mean(values)) for arm, values in sorted(target_improvements.items())
        },
        "cvar_weight_range": [min(cvar_weights.values()), max(cvar_weights.values())],
        "language_gradient_ratio_preserved_within_every_group": True,
    }
    return output, audit


def prepare_robust_expert_data(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    contract = prepare_robust_expert_contract(config)
    settings = dict(config.section("robust_family_experts"))
    data_root = _robust_data_root(config)
    artifact_root = _robust_root(config)
    status_path = artifact_root / "data-status.json"
    fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "parent": contract["parent_adapter_sha256"],
            "builder_version": 1,
        }
    )
    if status_path.exists() and not force:
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            return existing
        raise RuntimeError("Robust expert data fingerprint changed")
    forbidden = [
        data_root / "surfaces" / "confirmation_shard.jsonl",
        data_root / "surfaces" / "promotion_shard.jsonl",
        data_root / "surfaces" / "final_shard.jsonl",
    ]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("A downstream robust-expert shard was opened before data preparation")

    manifests: dict[str, dict[str, Any]] = {}
    manifests["training_pool"], training_simulations = _simulate_registered_scenarios(
        config,
        contract=contract,
        settings_name="training_pool",
        shard_settings=dict(settings["training_pool"]),
    )
    for index, fold_settings in enumerate(settings["selection_folds"], start=1):
        name = f"selection_fold_{index}"
        manifests[name], _ = _simulate_registered_scenarios(
            config,
            contract=contract,
            settings_name=name,
            shard_settings=dict(fold_settings),
        )

    selected = json.loads(
        (config.path_for("parent_artifact_dir") / "selected.json").read_text(encoding="utf-8")
    )
    parent_path = Path(str(selected["adapter_path"]))
    parent_sha = sha256_file(parent_path / "adapters.safetensors")
    if parent_sha != contract["parent_adapter_sha256"]:
        raise RuntimeError("Robust expert data no longer uses the registered v0.3 parent")
    model = tokenizer = None
    previous_limit = mx.set_cache_limit(int(float(settings["cache_limit_gb"]) * 1024**3))
    try:
        model, tokenizer = load(
            _stats_snapshot(config),
            adapter_path=str(parent_path),
            tokenizer_config={"trust_remote_code": True},
        )
        model.eval()
        target_rows, target_audit = _build_target_rows(
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
    result = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "contract_fingerprint": contract["fingerprint"],
        "parent_adapter_sha256": parent_sha,
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
    public = copy.deepcopy(result)
    write_json(config.root / "reports" / "evolve" / "robust-family-experts-data.json", public)
    return result


def robust_contract_sha256(config: ProjectConfig) -> str:
    path = _robust_root(config) / "contract.json"
    if not path.exists():
        prepare_robust_expert_contract(config)
    return sha256_file(path)


def _robust_training_settings(config: ProjectConfig, settings: dict[str, Any]) -> dict[str, Any]:
    """Keep every selection and downstream gate out of the training fingerprint."""
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


def _robust_downstream_paths(config: ProjectConfig) -> list[Path]:
    data_root = _robust_data_root(config)
    artifact_root = _robust_root(config)
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


def _robust_selection_folds(
    config: ProjectConfig,
    family_id: str,
) -> dict[str, dict[str, Any]]:
    settings = dict(config.section("robust_family_experts"))
    data_root = _robust_data_root(config)
    folds: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for index, fold_settings in enumerate(settings["selection_folds"], start=1):
        name = f"selection_fold_{index}"
        path = data_root / "surfaces" / f"{name}.jsonl"
        manifest_path = path.with_suffix(".manifest.json")
        if not path.exists() or not manifest_path.exists():
            raise RuntimeError(f"Robust expert selection fold is missing: {name}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sha256") != sha256_file(path):
            raise RuntimeError(f"Robust expert selection fold changed: {name}")
        all_rows = list(read_jsonl(path))
        rows = [row for row in all_rows if str(row["scenario"]["family_id"]) == family_id]
        expected = int(fold_settings["selected_per_family"])
        if len(rows) != expected:
            raise RuntimeError(f"{name} has {len(rows)} {family_id} rows instead of {expected}")
        ids = {str(row["scenario"]["blueprint_id"]) for row in rows}
        if len(ids) != len(rows) or seen & ids:
            raise RuntimeError("Robust expert selection folds overlap or contain duplicates")
        seen.update(ids)
        folds[name] = {"manifest": manifest, "rows": rows}
    return folds


def _robust_parent_path(
    config: ProjectConfig,
    data_status: dict[str, Any],
) -> Path:
    selected_path = config.path_for("parent_artifact_dir") / "selected.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    parent = Path(str(selected["adapter_path"]))
    observed = sha256_file(parent / "adapters.safetensors")
    if observed != data_status["parent_adapter_sha256"]:
        raise RuntimeError("Robust expert training parent changed after data generation")
    return parent


def _parent_fold_cache(
    config: ProjectConfig,
    *,
    family_id: str,
    parent: Path,
    folds: dict[str, dict[str, Any]],
    model: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    parent_sha = sha256_file(parent / "adapters.safetensors")
    fingerprint = canonical_hash(
        {
            "family_id": family_id,
            "parent": parent_sha,
            "folds": {
                name: value["manifest"]["fingerprint"] for name, value in sorted(folds.items())
            },
            "scorer_version": 1,
        }
    )
    path = _robust_root(config) / "parent-selection" / f"{family_id}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            return existing
        raise RuntimeError(f"Robust parent selection cache changed for {family_id}")
    fold_scores = {
        name: _score_loaded_family(model, tokenizer, value["rows"])
        for name, value in sorted(folds.items())
    }
    result = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "family_id": family_id,
        "parent_adapter_sha256": parent_sha,
        "fold_scores": fold_scores,
        "fold_metrics": {
            name: _family_metrics(score, family_id) for name, score in sorted(fold_scores.items())
        },
    }
    write_json(path, result)
    return result


def run_robust_expert_arm(
    config: ProjectConfig,
    *,
    family_id: str,
    arm: str,
    force: bool = False,
) -> dict[str, Any]:
    family_ids = {family.family_id for family in FAMILIES}
    if family_id not in family_ids:
        raise ValueError(f"Unknown DGP family: {family_id}")
    settings = dict(config.section("robust_family_experts"))
    arms = [str(value) for value in settings["arms"]]
    if arm not in arms:
        raise ValueError(f"Unknown robust expert arm: {arm}")
    downstream = [path for path in _robust_downstream_paths(config) if path.exists()]
    if force and downstream:
        raise RuntimeError("Cannot force-retrain robust experts after selection was written")

    data_status = prepare_robust_expert_data(config, force=False)
    data_root = _robust_data_root(config)
    artifact_root = _robust_root(config)
    artifact_dir = artifact_root / "arms" / arm / family_id
    train_path = data_root / "train" / f"{arm}.jsonl"
    folds = _robust_selection_folds(config, family_id)
    training_settings = _robust_training_settings(config, settings)
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
        raise RuntimeError(f"Robust expert fingerprint changed for {arm}/{family_id}")

    resume: dict[str, Any] | None = None
    if progress_path.exists() and not force:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Robust expert partial fingerprint changed for {arm}/{family_id}")
        completed = int(progress.get("completed_updates", 0))
        checkpoint = Path(str(progress.get("last_checkpoint_path", "")))
        if completed and (
            not checkpoint.exists()
            or sha256_file(checkpoint) != progress.get("last_checkpoint_sha256")
        ):
            raise RuntimeError(f"Robust expert resume checkpoint changed for {arm}/{family_id}")
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

    parent = _robust_parent_path(config, data_status)
    parent_weights = parent / "adapters.safetensors"
    all_train_rows = list(read_jsonl(train_path))
    train_rows = [row for row in all_train_rows if str(row["metadata"]["family_id"]) == family_id]
    if len(train_rows) != int(settings["training_pool"]["selected_per_family"]) * 4:
        raise RuntimeError(f"Robust expert training coverage changed for {arm}/{family_id}")
    if {str(row["metadata"]["robust_expert_arm"]) for row in train_rows} != {arm}:
        raise RuntimeError("A robust expert received rows from another ablation arm")
    groups = _diagnostic_groups(train_rows, groups_per_family=10**9)
    if len(groups) != int(settings["training_pool"]["selected_per_family"]):
        raise RuntimeError("Robust expert semantic-group coverage changed")
    records_per_backward = int(settings["records_per_backward"])
    if 4 % records_per_backward:
        raise RuntimeError("Robust expert records_per_backward must divide a four-row group")

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

        parent_cache = _parent_fold_cache(
            config,
            family_id=family_id,
            parent=parent,
            folds=folds,
            model=model,
            tokenizer=tokenizer,
        )
        checkpoints: list[dict[str, Any]] = (
            [dict(value) for value in resume["checkpoints"]]
            if resume
            else [
                {
                    "name": "parent",
                    "path": str(parent_weights),
                    "sha256": sha256_file(parent_weights),
                    "fold_metrics": parent_cache["fold_metrics"],
                    "parent_cache_fingerprint": parent_cache["fingerprint"],
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
                f"Robust {arm}/{family_id}: gradient update {update_index}/{settings['updates']}",
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
                raise RuntimeError("Robust expert gradient aggregation crossed family boundaries")
            if parameter_names is None:
                parameter_names = names
            elif parameter_names != names:
                raise RuntimeError("Robust expert parameter order changed between updates")
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
                f"Robust {arm}/{family_id}: scoring three selection folds",
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
            cycle=max(7, int(config.section("policy_projection")["source_cycle"]) + 2),
            arm=f"robust-expert:{arm}:{family_id}",
        )
        adapter_config.setdefault("stats", {}).update(
            {
                "method": "DGP-Regret cross-fit robust family expert",
                "method_version": int(settings["method_version"]),
                "family_id": family_id,
                "robust_expert_arm": arm,
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


def run_robust_expert_training(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    settings = dict(config.section("robust_family_experts"))
    data_status = prepare_robust_expert_data(config, force=False)
    artifact_root = _robust_root(config)
    downstream = [path for path in _robust_downstream_paths(config) if path.exists()]
    if force and downstream:
        raise RuntimeError("Cannot force-retrain robust experts after selection was written")
    training_path = artifact_root / "training-status.json"
    if downstream and not force:
        if training_path.exists():
            existing = json.loads(training_path.read_text(encoding="utf-8"))
            if existing.get("complete") and existing.get("matched_backward_compute"):
                return existing
        raise RuntimeError("Robust expert downstream state exists without a valid training status")
    statuses: dict[str, dict[str, dict[str, Any]]] = {}
    family_ids = sorted(family.family_id for family in FAMILIES)
    arms = [str(value) for value in settings["arms"]]
    total = len(arms) * len(family_ids)
    current = 0
    for arm in arms:
        statuses[arm] = {}
        for family_id in family_ids:
            current += 1
            print(
                f"Robust expert {current}/{total}: {arm}/{family_id}",
                file=sys.stderr,
                flush=True,
            )
            command = [
                sys.executable,
                "-m",
                "charlie_alpha.cli",
                "stats",
                "robust-expert-arm",
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
    matched = exposures == expected_exposures and len(set(backward_calls.values())) == 1
    if not matched:
        raise RuntimeError("Robust expert ablation arms did not receive matched backward compute")
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
        "selection_opened": False,
        "confirmation_opened": False,
        "promotion_opened": False,
        "final_simulations_opened": False,
        "final_scores_opened": False,
    }
    write_json(training_path, result)
    write_json(config.root / "reports" / "evolve" / "robust-family-experts-training.json", result)
    return result


def _load_robust_expert_statuses(
    config: ProjectConfig,
) -> dict[str, dict[str, dict[str, Any]]]:
    settings = dict(config.section("robust_family_experts"))
    artifact_root = _robust_root(config)
    statuses: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in [str(value) for value in settings["arms"]]:
        statuses[arm] = {}
        for family in sorted(value.family_id for value in FAMILIES):
            path = artifact_root / "arms" / arm / family / "status.json"
            if not path.exists():
                raise RuntimeError(f"Robust expert is not trained: {arm}/{family}")
            status = json.loads(path.read_text(encoding="utf-8"))
            if not status.get("complete") or status.get("arm") != arm:
                raise RuntimeError(f"Robust expert status is incomplete: {arm}/{family}")
            if status.get("family_id") != family:
                raise RuntimeError(f"Robust expert family changed: {arm}/{family}")
            statuses[arm][family] = status
    return statuses


def _read_parent_fold_cache(config: ProjectConfig, family_id: str) -> dict[str, Any]:
    path = _robust_root(config) / "parent-selection" / f"{family_id}.json"
    if not path.exists():
        raise RuntimeError(f"Robust parent selection cache is missing for {family_id}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("complete") or result.get("family_id") != family_id:
        raise RuntimeError(f"Robust parent selection cache is incomplete for {family_id}")
    return result


def _select_robust_family_mapping(
    config: ProjectConfig,
    statuses: dict[str, dict[str, dict[str, Any]]],
    *,
    allowed_arms: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    settings = dict(config.section("robust_family_experts"))
    selection = dict(settings["selection"])
    gates = dict(settings["gates"])
    mapping: dict[str, dict[str, Any]] = {}
    audit: dict[str, Any] = {}
    for family_id in sorted(value.family_id for value in FAMILIES):
        parent_cache = _read_parent_fold_cache(config, family_id)
        fold_names = sorted(parent_cache["fold_metrics"])
        parent_metrics = parent_cache["fold_metrics"]
        options: list[dict[str, Any]] = []
        rejected_granular = 0
        for arm in allowed_arms:
            for checkpoint in statuses[arm][family_id]["checkpoints"]:
                if checkpoint["name"] == "parent":
                    continue
                candidate_metrics = checkpoint["fold_metrics"]
                granular = all(
                    _family_noninferior(candidate_metrics[name], parent_metrics[name], gates)
                    for name in fold_names
                )
                if not granular:
                    rejected_granular += 1
                    continue
                options.append(
                    {
                        "arm": arm,
                        "update": int(str(checkpoint["name"]).rsplit("-", 1)[1]),
                        "checkpoint_name": str(checkpoint["name"]),
                        "checkpoint_path": str(checkpoint["path"]),
                        "checkpoint_sha256": str(checkpoint["sha256"]),
                        "parent_fold_regrets": [
                            float(parent_metrics[name]["normalized_regret"]) for name in fold_names
                        ],
                        "candidate_fold_regrets": [
                            float(candidate_metrics[name]["normalized_regret"])
                            for name in fold_names
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
        if selected is None:
            parent_status = statuses[allowed_arms[0]][family_id]
            mapping[family_id] = {
                "slug": "parent",
                "arm": "parent",
                "checkpoint_name": "parent",
                "checkpoint_path": str(parent_status["parent_adapter_path"]),
                "checkpoint_sha256": str(parent_status["parent_adapter_sha256"]),
                "update": 0,
                "crossfit": None,
            }
        else:
            mapping[family_id] = {
                "slug": f"robust-{selected['arm']}-{family_id}-u{selected['update']:02d}",
                "arm": str(selected["arm"]),
                "checkpoint_name": str(selected["checkpoint_name"]),
                "checkpoint_path": str(selected["checkpoint_path"]),
                "checkpoint_sha256": str(selected["checkpoint_sha256"]),
                "update": int(selected["update"]),
                "crossfit": selected["crossfit"],
            }
        audit[family_id] = {
            "allowed_arms": allowed_arms,
            "eligible_options_before_crossfit": len(options),
            "granular_rejections": rejected_granular,
            "selected_arm": mapping[family_id]["arm"],
            "selected_checkpoint": mapping[family_id]["checkpoint_name"],
        }
    return mapping, audit


def _checkpoint_for_route(
    statuses: dict[str, dict[str, dict[str, Any]]],
    *,
    family_id: str,
    route: dict[str, Any],
) -> dict[str, Any] | None:
    if route["arm"] == "parent":
        return None
    return next(
        checkpoint
        for checkpoint in statuses[str(route["arm"])][family_id]["checkpoints"]
        if checkpoint["name"] == route["checkpoint_name"]
    )


def _selection_route_fold_score(
    config: ProjectConfig,
    statuses: dict[str, dict[str, dict[str, Any]]],
    mapping: dict[str, dict[str, Any]],
    *,
    fold_name: str,
) -> dict[str, Any]:
    language_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for family_id, route in sorted(mapping.items()):
        if route["arm"] == "parent":
            score = _read_parent_fold_cache(config, family_id)["fold_scores"][fold_name]
        else:
            checkpoint = _checkpoint_for_route(
                statuses,
                family_id=family_id,
                route=route,
            )
            if checkpoint is None:
                raise RuntimeError("Robust route lost a non-parent checkpoint")
            score = checkpoint["fold_scores"][fold_name]
        for language, result in score["languages"].items():
            language_predictions[language].extend(result["predictions"])
    languages = {
        language: _aggregate_predictions(predictions)
        for language, predictions in sorted(language_predictions.items())
    }
    # Each named fold has one registered count per family, not the sum across
    # folds. Assert against the actual contract below to catch missing routes.
    fold_index = int(fold_name.rsplit("_", 1)[1]) - 1
    per_family = int(
        config.section("robust_family_experts")["selection_folds"][fold_index][
            "selected_per_family"
        ]
    )
    expected = per_family * len(FAMILIES)
    if any(int(result["count"]) != expected for result in languages.values()):
        raise RuntimeError(f"Robust route changed coverage for {fold_name}")
    return {
        "selector": languages["en"],
        "languages": languages,
        # Non-statistical inputs remain on the unchanged parent by construction.
        "retention": {"accuracy": 1.0},
    }


def _combine_route_fold_scores(folds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    language_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for score in folds.values():
        for language, result in score["languages"].items():
            language_predictions[language].extend(result["predictions"])
    languages = {
        language: _aggregate_predictions(predictions)
        for language, predictions in sorted(language_predictions.items())
    }
    return {
        "selector": languages["en"],
        "languages": languages,
        "retention": {"accuracy": 1.0},
    }


def _route_selection_summary(
    parent_folds: dict[str, dict[str, Any]],
    candidate_folds: dict[str, dict[str, Any]],
    *,
    gates: dict[str, Any],
    selection: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    fold_comparisons = {
        name: _surface_comparison(parent_folds[name], candidate_folds[name], gates)
        for name in sorted(parent_folds)
    }
    relative = [
        float(value["trilingual_relative_regret_improvement"])
        for value in fold_comparisons.values()
    ]
    parent_combined = _combine_route_fold_scores(parent_folds)
    candidate_combined = _combine_route_fold_scores(candidate_folds)
    pooled_comparison = _surface_comparison(parent_combined, candidate_combined, gates)
    bootstrap = _paired_bootstrap(
        parent_combined,
        candidate_combined,
        repetitions=int(selection["bootstrap_repetitions"]),
        seed=seed,
    )
    gate_results = {
        "every_fold_granular": all(
            bool(value["all_gates_passed"]) for value in fold_comparisons.values()
        ),
        "pooled_granular": bool(pooled_comparison["all_gates_passed"]),
        "mean_relative_improvement": float(np.mean(relative))
        >= float(selection["minimum_route_mean_relative_improvement"]),
        "worst_fold_relative_improvement": float(min(relative))
        >= float(selection["minimum_route_worst_fold_relative_improvement"]),
        "paired_bootstrap": float(bootstrap["ci95_lower"])
        >= float(selection["bootstrap_ci_lower_floor"]),
    }
    return {
        "fold_comparisons": fold_comparisons,
        "fold_relative_improvements": relative,
        "mean_relative_improvement": float(np.mean(relative)),
        "worst_fold_relative_improvement": float(min(relative)),
        "pooled_comparison": pooled_comparison,
        "paired_bootstrap": bootstrap,
        "gates": gate_results,
        "passed": all(gate_results.values()),
    }


def _route_vs_control_summary(
    control_folds: dict[str, dict[str, Any]],
    candidate_folds: dict[str, dict[str, Any]],
    *,
    gates: dict[str, Any],
    selection: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    fold_comparisons = {
        name: _surface_comparison(control_folds[name], candidate_folds[name], gates)
        for name in sorted(control_folds)
    }
    control = _combine_route_fold_scores(control_folds)
    candidate = _combine_route_fold_scores(candidate_folds)
    comparison = _surface_comparison(control, candidate, gates)
    relative = float(comparison["trilingual_relative_regret_improvement"])
    bootstrap = _paired_bootstrap(
        control,
        candidate,
        repetitions=int(selection["bootstrap_repetitions"]),
        seed=seed,
    )
    gate_results = {
        "every_fold_granular": all(
            bool(value["all_gates_passed"]) for value in fold_comparisons.values()
        ),
        "pooled_granular": bool(comparison["all_gates_passed"]),
        "relative_improvement": relative
        >= float(selection["minimum_candidate_relative_improvement_over_control"]),
        "paired_bootstrap": float(bootstrap["ci95_lower"])
        >= float(selection["bootstrap_ci_lower_floor"]),
    }
    return {
        "fold_comparisons": fold_comparisons,
        "comparison": comparison,
        "relative_regret_improvement": relative,
        "paired_bootstrap": bootstrap,
        "gates": gate_results,
        "passed": all(gate_results.values()),
    }


def _public_robust_selection(report: dict[str, Any]) -> dict[str, Any]:
    public = copy.deepcopy(report)
    for option in public["route_options"].values():
        for route in option["mapping"].values():
            route.pop("checkpoint_path", None)
            route.pop("checkpoint_sha256", None)
    selected = public.get("selected")
    if selected is not None:
        for route in selected["mapping"].values():
            route.pop("checkpoint_path", None)
            route.pop("checkpoint_sha256", None)
    return public


def select_robust_expert_route(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    settings = dict(config.section("robust_family_experts"))
    selection_settings = dict(settings["selection"])
    gates = dict(settings["gates"])
    artifact_root = _robust_root(config)
    training_path = artifact_root / "training-status.json"
    if not training_path.exists():
        raise RuntimeError("Train all robust expert arms before cross-fit selection")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if not training.get("complete") or not training.get("matched_backward_compute"):
        raise RuntimeError("Robust expert training is incomplete or compute-unmatched")
    forbidden = [
        path
        for path in _robust_downstream_paths(config)
        if path.name != "selection.json" and path.exists()
    ]
    if force and forbidden:
        raise RuntimeError("Cannot rewrite robust selection after a downstream shard was opened")
    statuses = _load_robust_expert_statuses(config)
    arms = [str(value) for value in settings["arms"]]
    route_arms = {
        "boltzmann-mean": ["boltzmann-mean"],
        "direct-mean": ["direct-mean"],
        "direct-cvar": ["direct-cvar"],
        "robust-mixed": arms,
    }
    mappings: dict[str, dict[str, dict[str, Any]]] = {}
    mapping_audits: dict[str, Any] = {}
    for name, allowed in route_arms.items():
        mappings[name], mapping_audits[name] = _select_robust_family_mapping(
            config,
            statuses,
            allowed_arms=allowed,
        )
    parent_mapping = {
        family.family_id: {
            "slug": "parent",
            "arm": "parent",
            "checkpoint_name": "parent",
            "checkpoint_path": str(statuses[arms[0]][family.family_id]["parent_adapter_path"]),
            "checkpoint_sha256": str(statuses[arms[0]][family.family_id]["parent_adapter_sha256"]),
            "update": 0,
            "crossfit": None,
        }
        for family in FAMILIES
    }
    fold_names = [
        f"selection_fold_{index}" for index in range(1, len(settings["selection_folds"]) + 1)
    ]
    parent_folds = {
        name: _selection_route_fold_score(
            config,
            statuses,
            parent_mapping,
            fold_name=name,
        )
        for name in fold_names
    }
    route_options: dict[str, dict[str, Any]] = {}
    route_fold_scores: dict[str, dict[str, dict[str, Any]]] = {}
    seed = int(config.section("project")["seed"])
    for name, mapping in mappings.items():
        folds = {
            fold_name: _selection_route_fold_score(
                config,
                statuses,
                mapping,
                fold_name=fold_name,
            )
            for fold_name in fold_names
        }
        route_fold_scores[name] = folds
        route_options[name] = {
            "mapping": mapping,
            "mapping_audit": mapping_audits[name],
            "nonparent_families": sorted(
                family for family, route in mapping.items() if route["arm"] != "parent"
            ),
            "vs_parent": _route_selection_summary(
                parent_folds,
                folds,
                gates=gates,
                selection=selection_settings,
                seed=seed,
            ),
            "vs_control": None,
        }

    control_name = "boltzmann-mean"
    control_folds = route_fold_scores[control_name]
    for index, name in enumerate(("direct-mean", "direct-cvar", "robust-mixed"), start=1):
        route_options[name]["vs_control"] = _route_vs_control_summary(
            control_folds,
            route_fold_scores[name],
            gates=gates,
            selection=selection_settings,
            seed=seed + index,
        )
    eligible = [
        name
        for name in ("direct-mean", "direct-cvar", "robust-mixed")
        if route_options[name]["vs_parent"]["passed"]
        and route_options[name]["vs_control"]["passed"]
        and route_options[name]["nonparent_families"]
    ]
    simplicity_order = {"direct-mean": 0, "direct-cvar": 1, "robust-mixed": 2}
    winner_name = (
        min(
            eligible,
            key=lambda name: (
                -float(route_options[name]["vs_parent"]["worst_fold_relative_improvement"]),
                -float(route_options[name]["vs_control"]["relative_regret_improvement"]),
                -float(route_options[name]["vs_parent"]["mean_relative_improvement"]),
                simplicity_order[name],
            ),
        )
        if eligible
        else None
    )
    selected = (
        {
            "name": winner_name,
            "mapping": mappings[winner_name],
            "nonparent_families": route_options[winner_name]["nonparent_families"],
            "vs_parent": route_options[winner_name]["vs_parent"],
            "vs_control": route_options[winner_name]["vs_control"],
        }
        if winner_name is not None
        else None
    )
    fingerprint = canonical_hash(
        {
            "contract": training["contract_fingerprint"],
            "training": training["fingerprint"],
            "route_mappings": {
                name: {
                    family: {
                        "arm": route["arm"],
                        "checkpoint": route["checkpoint_name"],
                        "sha256": route["checkpoint_sha256"],
                    }
                    for family, route in sorted(mapping.items())
                }
                for name, mapping in sorted(mappings.items())
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
            return existing
        raise RuntimeError("Robust expert selection fingerprint changed")
    result = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "method": "DGP-Regret cross-fit robust family-expert hypothesis",
        "matched_backward_compute": True,
        "route_options": route_options,
        "eligible_candidates": eligible,
        "selected": selected,
        "passed": selected is not None,
        "confirmation_opened": False,
        "promotion_opened": False,
        "final_simulations_opened": False,
        "final_scores_opened": False,
        "claim_boundary": (
            "Selection-fold evidence is development evidence, not a final capability result. "
            "The mixed option uses matched compute within each trained arm but additional "
            "research-time model selection across arms."
        ),
    }
    write_json(selection_path, result)
    training["selection_opened"] = True
    write_json(training_path, training)
    write_json(
        config.root / "reports" / "evolve" / "robust-family-experts-selection.json",
        _public_robust_selection(result),
    )
    return result

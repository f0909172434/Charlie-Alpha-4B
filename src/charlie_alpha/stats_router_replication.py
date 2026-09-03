from __future__ import annotations

import copy
import gc
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm import load

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json
from .stats_calibrate import _surface_comparison
from .stats_catalog import FAMILY_BY_ID
from .stats_dgp import Scenario, build_blueprints
from .stats_family_router import (
    _ensure_router_shard,
    _expert_context,
    _paired_bootstrap,
    _route_documents,
    _routed_score_from_cached,
)
from .stats_llm_router import (
    ParentLetterRouter,
    _compact_surface_score,
    _llm_router_root,
    _score_sparse_validation_adapters,
    family_route_prompt,
)
from .stats_training import _stats_snapshot

_REPLICATION_EVALUATOR_VERSION = 1


def paired_power_sample_size(
    *,
    paired_sd: float,
    parent_mean_regret: float,
    minimum_relative_improvement: float,
    alpha: float,
    power: float,
    safety_margin: float,
    allocation_multiple: int,
) -> dict[str, float | int]:
    """Normal-approximation size for the mean paired regret difference."""
    if paired_sd <= 0 or parent_mean_regret <= 0:
        raise ValueError("Power inputs must have positive scale")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("Alpha and power must lie strictly between zero and one")
    if minimum_relative_improvement <= 0 or safety_margin < 0 or allocation_multiple < 1:
        raise ValueError("Effect, safety margin, and allocation multiple are invalid")
    absolute_effect = parent_mean_regret * minimum_relative_improvement
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha / 2.0)
    z_power = normal.inv_cdf(power)
    raw = ((z_alpha + z_power) * paired_sd / absolute_effect) ** 2
    with_margin = raw * (1.0 + safety_margin)
    registered = int(math.ceil(with_margin / allocation_multiple) * allocation_multiple)
    return {
        "absolute_effect": absolute_effect,
        "z_alpha_two_sided": z_alpha,
        "z_power": z_power,
        "raw_required_blueprints": raw,
        "margin_adjusted_blueprints": with_margin,
        "registered_minimum_blueprints": registered,
    }


def _scenario_semantic_payload(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "family_id": str(scenario["family_id"]),
        "domain": str(scenario["domain"]),
        "parameters": {
            str(key): float(value)
            for key, value in sorted(dict(scenario["parameters"]).items())
        },
        "boundary_round": int(scenario["boundary_round"]),
    }


def _extract_scenario(row: dict[str, Any]) -> dict[str, Any] | None:
    candidate = row.get("scenario", row)
    if not isinstance(candidate, dict):
        return None
    required = {"blueprint_id", "family_id", "domain", "parameters", "boundary_round"}
    return candidate if required.issubset(candidate) else None


def _historical_scenario_audit(
    config: ProjectConfig,
    scenarios: list[Scenario],
    *,
    excluded_root: Path,
    minimum_normalized_distance: float,
) -> dict[str, Any]:
    historical_ids: set[str] = set()
    historical_semantics: set[str] = set()
    historical_parameters: dict[str, list[list[float]]] = defaultdict(list)
    source_files: list[dict[str, Any]] = []
    for base in (config.root / "data", config.root / "artifacts"):
        for path in sorted(base.rglob("*.jsonl")):
            if path.is_relative_to(excluded_root):
                continue
            extracted: list[dict[str, Any]] = []
            for row in read_jsonl(path):
                scenario = _extract_scenario(row)
                if scenario is not None:
                    extracted.append(scenario)
            if not extracted:
                continue
            source_files.append(
                {
                    "path": str(path.relative_to(config.root)),
                    "sha256": sha256_file(path),
                    "scenario_rows": len(extracted),
                }
            )
            for scenario in extracted:
                family_id = str(scenario["family_id"])
                family = FAMILY_BY_ID.get(family_id)
                if family is None:
                    continue
                historical_ids.add(str(scenario["blueprint_id"]))
                historical_semantics.add(canonical_hash(_scenario_semantic_payload(scenario)))
                historical_parameters[family_id].append(
                    [
                        (float(scenario["parameters"][key]) - lower) / (upper - lower)
                        for key, (lower, upper) in family.parameters.items()
                    ]
                )

    new_rows = [scenario.to_dict() for scenario in scenarios]
    new_ids = {str(row["blueprint_id"]) for row in new_rows}
    new_semantics = {canonical_hash(_scenario_semantic_payload(row)) for row in new_rows}
    id_overlap = sorted(new_ids & historical_ids)
    semantic_overlap = sorted(new_semantics & historical_semantics)
    nearest = math.inf
    nearest_family: str | None = None
    for row in new_rows:
        family_id = str(row["family_id"])
        historical = historical_parameters.get(family_id)
        if not historical:
            continue
        family = FAMILY_BY_ID[family_id]
        point = np.asarray(
            [
                (float(row["parameters"][key]) - lower) / (upper - lower)
                for key, (lower, upper) in family.parameters.items()
            ],
            dtype=np.float64,
        )
        distances = np.max(np.abs(np.asarray(historical, dtype=np.float64) - point), axis=1)
        distance = float(np.min(distances))
        if distance < nearest:
            nearest = distance
            nearest_family = family_id
    if not math.isfinite(nearest):
        raise RuntimeError("No historical scenario corpus was available for overlap audit")
    passed = (
        not id_overlap
        and not semantic_overlap
        and nearest >= minimum_normalized_distance
    )
    return {
        "method": (
            "Exact blueprint IDs, semantic parameter hashes, and within-family normalized "
            "L-infinity nearest-neighbor distance"
        ),
        "historical_source_manifest_fingerprint": canonical_hash(source_files),
        "historical_source_files": source_files,
        "historical_unique_blueprints": len(historical_ids),
        "historical_unique_semantic_points": len(historical_semantics),
        "new_blueprints": len(new_rows),
        "blueprint_id_overlap_count": len(id_overlap),
        "semantic_overlap_count": len(semantic_overlap),
        "minimum_normalized_linf_distance": nearest,
        "minimum_distance_family": nearest_family,
        "registered_minimum_normalized_distance": minimum_normalized_distance,
        "passed": passed,
    }


def _paired_historical_inputs(report: dict[str, Any]) -> dict[str, float | int]:
    parent = report["private_scores"]["v0.3-parent"]
    candidate = report["private_scores"]["routed-experts"]
    parent_by_id: dict[str, list[float]] = defaultdict(list)
    candidate_by_id: dict[str, list[float]] = defaultdict(list)
    for result in parent["languages"].values():
        for prediction in result["predictions"]:
            parent_by_id[str(prediction["blueprint_id"])].append(
                float(prediction["normalized_regret"])
            )
    for result in candidate["languages"].values():
        for prediction in result["predictions"]:
            candidate_by_id[str(prediction["blueprint_id"])].append(
                float(prediction["normalized_regret"])
            )
    if set(parent_by_id) != set(candidate_by_id):
        raise RuntimeError("Historical paired score coverage differs")
    differences = np.asarray(
        [
            float(np.mean(parent_by_id[key]) - np.mean(candidate_by_id[key]))
            for key in sorted(parent_by_id)
        ],
        dtype=np.float64,
    )
    return {
        "blueprints": len(differences),
        "parent_mean_regret": float(
            np.mean([value for values in parent_by_id.values() for value in values])
        ),
        "paired_difference_mean": float(np.mean(differences)),
        "paired_difference_sd": float(np.std(differences, ddof=1)),
    }


def _surface_fingerprint(
    config: ProjectConfig,
    settings: dict[str, Any],
    scenarios: list[Scenario],
) -> str:
    simulation = config.section("stats_data")
    return canonical_hash(
        {
            "settings": settings,
            "scenarios": [scenario.to_dict() for scenario in scenarios],
            "simulation": {
                key: simulation[key]
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


def _current_candidate_lock_fields(config: ProjectConfig) -> tuple[dict[str, Any], list[Scenario]]:
    settings = copy.deepcopy(config.section("llm_family_router_replication"))
    router_settings = config.section("llm_family_router")
    if int(settings["prompt_version"]) != int(router_settings["prompt_version"]):
        raise RuntimeError("Replication prompt version differs from the frozen router")
    root = _llm_router_root(config)
    router_path = root / "report.json"
    promotion_path = root / "promotion" / "report.json"
    final_path = root / "final" / "report.json"
    router_report = json.loads(router_path.read_text(encoding="utf-8"))
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    if not promotion.get("passed"):
        raise RuntimeError("Replication requires the previously successful promotion")
    threshold = float(router_report["selection"]["selected_threshold"])
    if threshold != float(settings["expected_selected_threshold"]):
        raise RuntimeError("Frozen router threshold differs from the replication contract")
    expert_report, adapter_paths = _expert_context(config)
    final_report = json.loads(final_path.read_text(encoding="utf-8"))
    historical = _paired_historical_inputs(final_report)
    power_settings = settings["power"]
    power_analysis = paired_power_sample_size(
        paired_sd=float(historical["paired_difference_sd"]),
        parent_mean_regret=float(historical["parent_mean_regret"]),
        minimum_relative_improvement=float(power_settings["minimum_relative_improvement"]),
        alpha=float(power_settings["alpha_two_sided"]),
        power=float(power_settings["target_power"]),
        safety_margin=float(power_settings["safety_margin"]),
        allocation_multiple=int(power_settings["allocation_multiple"]),
    )
    shard = settings["replication_shard"]
    if int(shard["count"]) < int(power_analysis["registered_minimum_blueprints"]):
        raise RuntimeError("Registered replication shard is smaller than its power analysis")
    scenarios = build_blueprints(
        {str(shard["split"]): int(shard["count"])},
        seed=int(shard["seed"]),
        active_search=False,
    )
    fields = {
        "schema_version": 1,
        "method": "preregistered independent frozen family-router replication",
        "research_question": (
            "Does the fixed v2 parent-letter router and fixed family-expert mapping reproduce "
            "at least 7.5% relative trilingual DGP-Regret improvement over the v0.3 parent on "
            "a fresh powered synthetic surface?"
        ),
        "candidate": {
            "router_report_fingerprint": router_report["fingerprint"],
            "router_report_sha256": sha256_file(router_path),
            "promotion_report_fingerprint": promotion["fingerprint"],
            "promotion_report_sha256": sha256_file(promotion_path),
            "prompt_version": int(settings["prompt_version"]),
            "prompt_sha256": canonical_hash(family_route_prompt()),
            "selected_threshold": threshold,
            "expert_selection_fingerprint": expert_report["selection"]["fingerprint"],
            "adapter_sha256": {
                slug: sha256_file(path / "adapters.safetensors")
                for slug, path in sorted(adapter_paths.items())
            },
        },
        "control": {
            "name": "v0.3-parent",
            "adapter_sha256": sha256_file(adapter_paths["parent"] / "adapters.safetensors"),
        },
        "settings": settings,
        "power_analysis": {
            "method": "paired normal approximation with two-sided alpha and safety margin",
            "historical_variance_source": "retired v0.3 final paired scores; variance only",
            "historical_report_sha256": sha256_file(final_path),
            "historical_inputs": historical,
            **power_analysis,
            "registered_blueprints": int(shard["count"]),
        },
        "blueprint_fingerprint": canonical_hash([scenario.to_dict() for scenario in scenarios]),
        "anticipated_surface_fingerprint": _surface_fingerprint(config, shard, scenarios),
        "adaptation_policy": (
            "none; no prompt, threshold, expert, weight, gate, or sample-size tuning"
        ),
        "stopping_rule": "score every registered blueprint once; no optional stopping",
        "retired_v0_3_final_policy": (
            "Used only to estimate paired variance before opening the replication surface; "
            "never reused as replication outcomes and never tuned against."
        ),
        "decision_rule": (
            "Pass only if every preregistered aggregate, granular, routing, integrity, and paired "
            "bootstrap gate passes. Passing supports external-benchmark design, not automatic "
            "champion or weight promotion."
        ),
    }
    return fields, scenarios


def prepare_family_router_replication_contract(config: ProjectConfig) -> dict[str, Any]:
    root = _llm_router_root(config) / "independent-replication-v1"
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "lock.json"
    public_path = config.root / "reports" / "evolve" / "family-router-replication-contract.json"
    fields, scenarios = _current_candidate_lock_fields(config)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        payload = {key: value for key, value in existing.items() if key != "fingerprint"}
        if canonical_hash(payload) != str(existing.get("fingerprint")):
            raise RuntimeError("Replication lock fingerprint is corrupt")
        for key in (
            "candidate",
            "control",
            "settings",
            "power_analysis",
            "blueprint_fingerprint",
            "anticipated_surface_fingerprint",
        ):
            if existing.get(key) != fields[key]:
                raise RuntimeError(f"Frozen replication lock changed: {key}")
        write_json(public_path, existing)
        return existing
    if (root / "replication.jsonl").exists() or (root / "replication-manifest.json").exists():
        raise RuntimeError("Cannot preregister after the replication surface was opened")
    audit = _historical_scenario_audit(
        config,
        scenarios,
        excluded_root=root,
        minimum_normalized_distance=float(fields["settings"]["minimum_normalized_distance"]),
    )
    if not audit["passed"]:
        raise RuntimeError("Fresh replication surface failed the preregistered overlap audit")
    lock = {**fields, "integrity_audit": audit}
    lock["fingerprint"] = canonical_hash(lock)
    write_json(lock_path, lock)
    write_json(public_path, lock)
    return lock


def _replication_gate_results(
    comparison: dict[str, Any],
    route_metrics: dict[str, Any],
    bootstrap: dict[str, float],
    gates: dict[str, Any],
    integrity_passed: bool,
) -> dict[str, bool]:
    return {
        **{f"model_{key}": bool(value) for key, value in comparison["gates"].items()},
        "relative_regret": float(comparison["trilingual_relative_regret_improvement"])
        >= float(gates["minimum_replication_relative_improvement"]),
        "router_family_accuracy": float(route_metrics["family_accuracy"])
        >= float(gates["minimum_router_family_accuracy"]),
        "router_language_accuracy": all(
            float(value) >= float(gates["minimum_language_router_accuracy"])
            for value in route_metrics["language_family_accuracy"].values()
        ),
        "wrong_expert_rate": float(route_metrics["wrong_expert_rate"])
        <= float(gates["maximum_wrong_expert_rate"]),
        "expert_coverage": float(route_metrics["expert_coverage"])
        >= float(gates["minimum_expert_coverage"]),
        "paired_bootstrap": float(bootstrap["ci95_lower"])
        >= float(gates["bootstrap_ci_lower_floor"]),
        "fresh_surface_integrity": integrity_passed,
    }


def run_family_router_replication(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    lock = prepare_family_router_replication_contract(config)
    settings = config.section("llm_family_router_replication")
    gates = dict(settings["gates"])
    root = _llm_router_root(config) / "independent-replication-v1"
    report_path = root / "report.json"
    public_path = config.root / "reports" / "evolve" / "family-router-replication.json"
    manifest, rows = _ensure_router_shard(
        config,
        root,
        "replication_shard",
        open_if_missing=True,
        section_name="llm_family_router_replication",
    )
    if manifest["fingerprint"] != lock["anticipated_surface_fingerprint"]:
        raise RuntimeError("Opened replication surface differs from its preregistered fingerprint")
    fingerprint = canonical_hash(
        {
            "lock": lock["fingerprint"],
            "surface": manifest["fingerprint"],
            "evaluator_version": _REPLICATION_EVALUATOR_VERSION,
        }
    )
    if report_path.exists() and not force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            public = copy.deepcopy(existing)
            public.pop("private_scores", None)
            write_json(public_path, public)
            return public
        raise RuntimeError("Replication report fingerprint changed")

    expert_report, adapter_paths = _expert_context(config)
    expert_mapping = expert_report["selection"]["mapping"]
    parent_model, parent_tokenizer = load(
        _stats_snapshot(config),
        adapter_path=str(adapter_paths["parent"]),
        tokenizer_config={"trust_remote_code": True},
    )
    router = ParentLetterRouter(parent_model, parent_tokenizer)
    decisions = _route_documents(
        router,
        rows,
        view=str(settings["replication_shard"]["view"]),
    )
    scores = _score_sparse_validation_adapters(
        config,
        rows,
        manifest,
        adapter_paths,
        expert_mapping,
        decisions,
        root,
    )
    historical_final = json.loads(
        (_llm_router_root(config) / "final" / "report.json").read_text(encoding="utf-8")
    )
    retention = copy.deepcopy(
        historical_final["private_scores"]["v0.3-parent"]["retention"]
    )
    parent = {**scores["parent"], "retention": retention}
    candidate, route_metrics = _routed_score_from_cached(
        scores,
        decisions,
        expert_mapping,
        threshold=float(lock["candidate"]["selected_threshold"]),
        retention=retention,
    )
    comparison = _surface_comparison(parent, candidate, gates)
    bootstrap = _paired_bootstrap(
        parent,
        candidate,
        repetitions=int(gates["bootstrap_repetitions"]),
        seed=int(settings["bootstrap_seed"]),
    )
    gate_results = _replication_gate_results(
        comparison,
        route_metrics,
        bootstrap,
        gates,
        bool(lock["integrity_audit"]["passed"]),
    )
    passed = all(gate_results.values())
    report = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "method": "preregistered independent frozen family-router replication",
        "contract_fingerprint": lock["fingerprint"],
        "manifest": manifest,
        "integrity_audit": lock["integrity_audit"],
        "absolute_metrics": {
            "v0.3-parent": _compact_surface_score(parent),
            "routed-experts": _compact_surface_score(candidate),
        },
        "comparison": comparison,
        "route_metrics": route_metrics,
        "paired_bootstrap": bootstrap,
        "gates": gate_results,
        "passed": passed,
        "candidate_status": "independently-replicated" if passed else "replication-rejected",
        "proceed_to_external_benchmark_design": passed,
        "automatic_champion_promotion": False,
        "fresh_replication_surface_opened": True,
        "fresh_replication_surface_retired": True,
        "retired_v0_3_final_reused_for_scoring": False,
        "interpretation": (
            "The promotion effect independently replicated at the preregistered practical and "
            "uncertainty thresholds; the old 120-blueprint final miss is consistent with its low "
            "power. External validity and operational-cost gates remain open."
            if passed
            else (
                "The frozen router failed independent replication and is rejected as a "
                "champion candidate."
            )
        ),
        "claim_boundary": (
            "This is fresh synthetic DGP evidence for a frozen routing policy. It does not "
            "establish external benchmark validity, runtime acceptability, or weight-level "
            "superiority."
        ),
        "private_scores": {"v0.3-parent": parent, "routed-experts": candidate},
    }
    write_json(report_path, report)
    public = copy.deepcopy(report)
    public.pop("private_scores")
    write_json(public_path, public)
    del parent_model, parent_tokenizer, router
    gc.collect()
    mx.clear_cache()
    return public

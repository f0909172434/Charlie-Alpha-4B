from __future__ import annotations

import copy
import gc
import json
from typing import Any

import mlx.core as mx
from mlx_lm import load

from .config import ProjectConfig
from .io_utils import canonical_hash, sha256_file, write_json
from .stats_calibrate import _surface_comparison
from .stats_dgp import build_blueprints
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
from .stats_router_replication import (
    _historical_scenario_audit,
    _replication_gate_results,
    _surface_fingerprint,
    paired_power_sample_size,
)
from .stats_training import _stats_snapshot


def _reduced_mapping(
    mapping: dict[str, dict[str, Any]],
    excluded_family: str,
) -> dict[str, dict[str, Any]]:
    result = copy.deepcopy(mapping)
    if excluded_family not in result:
        raise ValueError(f"Unknown excluded family: {excluded_family}")
    result[excluded_family] = {
        **result[excluded_family],
        "slug": "parent",
        "checkpoint_name": "parent",
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "update": 0,
        "prospective_exclusion": True,
    }
    return result


def prepare_reduced_family_router_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = copy.deepcopy(config.section("llm_family_router_reduced"))
    root = _llm_router_root(config) / "reduced-route-v1"
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "lock.json"
    public_path = config.root / "reports" / "evolve" / "family-router-reduced-contract.json"
    replication_root = _llm_router_root(config) / "independent-replication-v1"
    replication_lock_path = replication_root / "lock.json"
    failure_path = replication_root / "failure-analysis.json"
    replication_lock = json.loads(replication_lock_path.read_text(encoding="utf-8"))
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    excluded_family = str(settings["excluded_family"])
    if excluded_family != str(failure["culprit_family"]):
        raise RuntimeError("Reduced route exclusion differs from the frozen failure diagnosis")
    router_report = json.loads(
        (_llm_router_root(config) / "report.json").read_text(encoding="utf-8")
    )
    threshold = float(router_report["selection"]["selected_threshold"])
    if threshold != float(settings["expected_selected_threshold"]):
        raise RuntimeError("Reduced route threshold differs from the frozen router")
    expert_report, adapter_paths = _expert_context(config)
    mapping = _reduced_mapping(expert_report["selection"]["mapping"], excluded_family)
    shard = settings["confirmation_shard"]
    scenarios = build_blueprints(
        {str(shard["split"]): int(shard["count"])},
        seed=int(shard["seed"]),
        active_search=False,
    )
    historical_inputs = replication_lock["power_analysis"]["historical_inputs"]
    power_settings = settings["power"]
    power = paired_power_sample_size(
        paired_sd=float(historical_inputs["paired_difference_sd"]),
        parent_mean_regret=float(historical_inputs["parent_mean_regret"]),
        minimum_relative_improvement=float(power_settings["minimum_relative_improvement"]),
        alpha=float(power_settings["alpha_two_sided"]),
        power=float(power_settings["target_power"]),
        safety_margin=float(power_settings["safety_margin"]),
        allocation_multiple=int(power_settings["allocation_multiple"]),
    )
    if int(shard["count"]) < int(power["registered_minimum_blueprints"]):
        raise RuntimeError("Reduced-route confirmation is smaller than its power analysis")
    fields = {
        "schema_version": 1,
        "method": "preregistered reduced family-router independent confirmation",
        "research_question": (
            "Does prospectively excluding the experimental-causal expert preserve at least "
            "7.5% relative trilingual DGP-Regret improvement while satisfying every granular "
            "safety gate on a completely fresh surface?"
        ),
        "candidate": {
            "source_failure_analysis_fingerprint": failure["fingerprint"],
            "source_failure_analysis_sha256": sha256_file(failure_path),
            "router_report_fingerprint": router_report["fingerprint"],
            "prompt_sha256": canonical_hash(family_route_prompt()),
            "prompt_version": int(settings["prompt_version"]),
            "selected_threshold": threshold,
            "excluded_family": excluded_family,
            "mapping_fingerprint": canonical_hash(mapping),
            "adapter_sha256": {
                slug: sha256_file(path / "adapters.safetensors")
                for slug, path in sorted(adapter_paths.items())
                if slug != str(expert_report["selection"]["mapping"][excluded_family]["slug"])
            },
        },
        "control": {
            "name": "v0.3-parent",
            "adapter_sha256": sha256_file(adapter_paths["parent"] / "adapters.safetensors"),
        },
        "settings": settings,
        "power_analysis": {
            "method": "paired normal approximation with two-sided alpha and safety margin",
            "variance_source": "retired v0.3 final paired variance fixed before candidate design",
            "historical_inputs": historical_inputs,
            **power,
            "registered_blueprints": int(shard["count"]),
        },
        "blueprint_fingerprint": canonical_hash([scenario.to_dict() for scenario in scenarios]),
        "anticipated_surface_fingerprint": _surface_fingerprint(config, shard, scenarios),
        "adaptation_policy": "none after lock; excluded family, mapping, threshold, gates fixed",
        "stopping_rule": "score all registered blueprints exactly once; no optional stopping",
        "decision_rule": (
            "Pass only if all aggregate, granular, routing, integrity, efficacy, and paired "
            "bootstrap gates pass. A pass authorizes external benchmark design, not champion "
            "promotion."
        ),
    }
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        payload = {key: value for key, value in existing.items() if key != "fingerprint"}
        if canonical_hash(payload) != existing.get("fingerprint"):
            raise RuntimeError("Reduced-route lock fingerprint is corrupt")
        for key in fields:
            if existing.get(key) != fields[key]:
                raise RuntimeError(f"Frozen reduced-route lock changed: {key}")
        write_json(public_path, existing)
        return existing
    if (root / "confirmation.jsonl").exists() or (root / "confirmation-manifest.json").exists():
        raise RuntimeError("Cannot preregister after reduced-route confirmation was opened")
    audit = _historical_scenario_audit(
        config,
        scenarios,
        excluded_root=root,
        minimum_normalized_distance=float(settings["minimum_normalized_distance"]),
    )
    if not audit["passed"]:
        raise RuntimeError("Reduced-route confirmation failed the fresh-surface audit")
    lock = {**fields, "integrity_audit": audit}
    lock["fingerprint"] = canonical_hash(lock)
    write_json(lock_path, lock)
    write_json(public_path, lock)
    return lock


def run_reduced_family_router_confirmation(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    lock = prepare_reduced_family_router_contract(config)
    settings = config.section("llm_family_router_reduced")
    gates = dict(settings["gates"])
    root = _llm_router_root(config) / "reduced-route-v1"
    report_path = root / "report.json"
    public_path = config.root / "reports" / "evolve" / "family-router-reduced.json"
    manifest, rows = _ensure_router_shard(
        config,
        root,
        "confirmation_shard",
        open_if_missing=True,
        section_name="llm_family_router_reduced",
    )
    if manifest["fingerprint"] != lock["anticipated_surface_fingerprint"]:
        raise RuntimeError("Reduced-route surface differs from its preregistration")
    fingerprint = canonical_hash(
        {"lock": lock["fingerprint"], "surface": manifest["fingerprint"], "evaluator_version": 1}
    )
    if report_path.exists() and not force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            public = copy.deepcopy(existing)
            public.pop("private_scores", None)
            write_json(public_path, public)
            return public
        raise RuntimeError("Reduced-route confirmation report changed")
    expert_report, adapter_paths = _expert_context(config)
    mapping = _reduced_mapping(
        expert_report["selection"]["mapping"],
        str(settings["excluded_family"]),
    )
    parent_model, parent_tokenizer = load(
        _stats_snapshot(config),
        adapter_path=str(adapter_paths["parent"]),
        tokenizer_config={"trust_remote_code": True},
    )
    router = ParentLetterRouter(parent_model, parent_tokenizer)
    decisions = _route_documents(
        router,
        rows,
        view=str(settings["confirmation_shard"]["view"]),
    )
    scores = _score_sparse_validation_adapters(
        config,
        rows,
        manifest,
        adapter_paths,
        mapping,
        decisions,
        root,
    )
    prior = json.loads(
        (_llm_router_root(config) / "final" / "report.json").read_text(encoding="utf-8")
    )
    retention = copy.deepcopy(prior["private_scores"]["v0.3-parent"]["retention"])
    parent = {**scores["parent"], "retention": retention}
    candidate, route_metrics = _routed_score_from_cached(
        scores,
        decisions,
        mapping,
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
        "method": "preregistered reduced family-router independent confirmation",
        "contract_fingerprint": lock["fingerprint"],
        "manifest": manifest,
        "absolute_metrics": {
            "v0.3-parent": _compact_surface_score(parent),
            "reduced-routed-experts": _compact_surface_score(candidate),
        },
        "comparison": comparison,
        "route_metrics": route_metrics,
        "paired_bootstrap": bootstrap,
        "gates": gate_results,
        "passed": passed,
        "candidate_status": "confirmation-survivor" if passed else "confirmation-rejected",
        "proceed_to_external_benchmark_design": passed,
        "automatic_champion_promotion": False,
        "fresh_confirmation_surface_opened": True,
        "fresh_confirmation_surface_retired": True,
        "interpretation": (
            "The prospectively reduced route independently confirmed and can advance to an "
            "external-benchmark contract."
            if passed
            else "The prospectively reduced route failed independent confirmation and is rejected."
        ),
        "claim_boundary": (
            "Fresh synthetic confirmation does not establish external benchmark validity or "
            "operational runtime acceptability."
        ),
        "private_scores": {"v0.3-parent": parent, "reduced-routed-experts": candidate},
    }
    write_json(report_path, report)
    public = copy.deepcopy(report)
    public.pop("private_scores")
    write_json(public_path, public)
    del parent_model, parent_tokenizer, router
    gc.collect()
    mx.clear_cache()
    return public

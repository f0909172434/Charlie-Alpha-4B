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
from .stats_router_reduced import _reduced_mapping
from .stats_router_replication import (
    _historical_scenario_audit,
    _replication_gate_results,
    _surface_fingerprint,
    paired_power_sample_size,
)
from .stats_sufficiency_guard import (
    ParentSufficiencyGuard,
    _apply_guard,
    _guard_examples,
    _guard_metrics,
    sufficiency_prompt,
)
from .stats_training import _stats_snapshot


def prepare_thresholded_sufficiency_guard_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = copy.deepcopy(config.section("sufficiency_guard_thresholded"))
    root = _llm_router_root(config) / "sufficiency-guard-thresholded-v1"
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "lock.json"
    public_path = (
        config.root / "reports" / "evolve" / "sufficiency-guard-thresholded-contract.json"
    )
    guard_root = _llm_router_root(config) / "sufficiency-guard-v1"
    diagnosis_path = guard_root / "margin-diagnosis.json"
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    threshold = float(settings["insufficient_probability_threshold"])
    if threshold != float(diagnosis["selected_threshold"]):
        raise RuntimeError("Thresholded guard differs from the frozen margin diagnosis")
    reduced_path = _llm_router_root(config) / "reduced-route-v1" / "report.json"
    reduced = json.loads(reduced_path.read_text(encoding="utf-8"))
    if not reduced.get("passed"):
        raise RuntimeError("Thresholded guard requires the confirmed reduced route")
    replication_lock = json.loads(
        (_llm_router_root(config) / "independent-replication-v1" / "lock.json").read_text(
            encoding="utf-8"
        )
    )
    shard = settings["confirmation_shard"]
    scenarios = build_blueprints(
        {str(shard["split"]): int(shard["count"])},
        seed=int(shard["seed"]),
        active_search=False,
    )
    historical = replication_lock["power_analysis"]["historical_inputs"]
    power_settings = settings["power"]
    power = paired_power_sample_size(
        paired_sd=float(historical["paired_difference_sd"]),
        parent_mean_regret=float(historical["parent_mean_regret"]),
        minimum_relative_improvement=float(power_settings["minimum_relative_improvement"]),
        alpha=float(power_settings["alpha_two_sided"]),
        power=float(power_settings["target_power"]),
        safety_margin=float(power_settings["safety_margin"]),
        allocation_multiple=int(power_settings["allocation_multiple"]),
    )
    if int(shard["count"]) < int(power["registered_minimum_blueprints"]):
        raise RuntimeError("Thresholded guard confirmation is smaller than its power analysis")
    expert_report, adapter_paths = _expert_context(config)
    mapping = _reduced_mapping(
        expert_report["selection"]["mapping"],
        str(settings["excluded_family"]),
    )
    fields = {
        "schema_version": 1,
        "method": "preregistered thresholded parent-logit sufficiency guard confirmation",
        "research_question": (
            "Does the prospectively calibrated 0.90 insufficiency threshold preserve the "
            "reduced-route efficacy while meeting sensitivity and specificity safety gates on a "
            "fully disjoint paired surface?"
        ),
        "candidate": {
            "reduced_route_fingerprint": reduced["fingerprint"],
            "margin_diagnosis_fingerprint": diagnosis["fingerprint"],
            "margin_diagnosis_sha256": sha256_file(diagnosis_path),
            "excluded_family": settings["excluded_family"],
            "mapping_fingerprint": canonical_hash(mapping),
            "family_prompt_sha256": canonical_hash(family_route_prompt()),
            "sufficiency_prompt_sha256": canonical_hash(sufficiency_prompt()),
            "insufficient_probability_threshold": threshold,
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
            "efficacy_method": "paired normal approximation with 20% safety margin",
            "historical_inputs": historical,
            **power,
            "registered_blueprints": int(shard["count"]),
            "minimum_family_language_cell_per_class": 54,
        },
        "blueprint_fingerprint": canonical_hash([scenario.to_dict() for scenario in scenarios]),
        "anticipated_surface_fingerprint": _surface_fingerprint(config, shard, scenarios),
        "adaptation_policy": "none after lock; threshold, prompts, mapping, gates, and size fixed",
        "stopping_rule": "evaluate all complete and incomplete registered renderings once",
        "decision_rule": (
            "Pass only if all guard safety and reduced-route efficacy, granular, routing, "
            "integrity, and bootstrap gates pass."
        ),
    }
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        payload = {key: value for key, value in existing.items() if key != "fingerprint"}
        if canonical_hash(payload) != existing.get("fingerprint"):
            raise RuntimeError("Thresholded-guard lock fingerprint is corrupt")
        for key in fields:
            if existing.get(key) != fields[key]:
                raise RuntimeError(f"Frozen thresholded-guard lock changed: {key}")
        write_json(public_path, existing)
        return existing
    if (root / "confirmation.jsonl").exists() or (root / "confirmation-manifest.json").exists():
        raise RuntimeError("Cannot preregister after thresholded-guard confirmation was opened")
    audit = _historical_scenario_audit(
        config,
        scenarios,
        excluded_root=root,
        minimum_normalized_distance=float(settings["minimum_normalized_distance"]),
    )
    if not audit["passed"]:
        raise RuntimeError("Thresholded-guard surface failed the fresh-surface audit")
    lock = {**fields, "integrity_audit": audit}
    lock["fingerprint"] = canonical_hash(lock)
    write_json(lock_path, lock)
    write_json(public_path, lock)
    return lock


def run_thresholded_sufficiency_guard_confirmation(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    lock = prepare_thresholded_sufficiency_guard_contract(config)
    settings = config.section("sufficiency_guard_thresholded")
    gates = dict(settings["gates"])
    root = _llm_router_root(config) / "sufficiency-guard-thresholded-v1"
    report_path = root / "report.json"
    public_path = config.root / "reports" / "evolve" / "sufficiency-guard-thresholded.json"
    manifest, rows = _ensure_router_shard(
        config,
        root,
        "confirmation_shard",
        open_if_missing=True,
        section_name="sufficiency_guard_thresholded",
    )
    if manifest["fingerprint"] != lock["anticipated_surface_fingerprint"]:
        raise RuntimeError("Thresholded-guard surface differs from its preregistration")
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
        raise RuntimeError("Thresholded-guard report changed")
    expert_report, adapter_paths = _expert_context(config)
    mapping = _reduced_mapping(
        expert_report["selection"]["mapping"],
        str(settings["excluded_family"]),
    )
    model, tokenizer = load(
        _stats_snapshot(config),
        adapter_path=str(adapter_paths["parent"]),
        tokenizer_config={"trust_remote_code": True},
    )
    router = ParentLetterRouter(model, tokenizer)
    guard = ParentSufficiencyGuard(
        model,
        tokenizer,
        threshold=float(settings["insufficient_probability_threshold"]),
    )
    family_decisions = _route_documents(
        router,
        rows,
        view=str(settings["confirmation_shard"]["view"]),
    )
    guard_metrics, complete_guard_decisions = _guard_metrics(
        guard,
        _guard_examples(rows, family_decisions),
    )
    scores = _score_sparse_validation_adapters(
        config,
        rows,
        manifest,
        adapter_paths,
        mapping,
        family_decisions,
        root,
    )
    prior = json.loads(
        (_llm_router_root(config) / "final" / "report.json").read_text(encoding="utf-8")
    )
    retention = copy.deepcopy(prior["private_scores"]["v0.3-parent"]["retention"])
    parent = {**scores["parent"], "retention": retention}
    unguarded, route_metrics = _routed_score_from_cached(
        scores,
        family_decisions,
        mapping,
        threshold=float(settings["expected_selected_threshold"]),
        retention=retention,
    )
    candidate = _apply_guard(unguarded, complete_guard_decisions)
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
    specificity = guard_metrics["complete_specificity"]
    sensitivity = guard_metrics["incomplete_sensitivity"]
    gate_results.update(
        {
            "guard_specificity": specificity["accuracy"] >= gates["minimum_guard_specificity"],
            "guard_language_specificity": min(specificity["language_accuracy"].values())
            >= gates["minimum_language_guard_specificity"],
            "guard_family_specificity": min(specificity["family_accuracy"].values())
            >= gates["minimum_family_guard_specificity"],
            "guard_sensitivity": sensitivity["accuracy"] >= gates["minimum_guard_sensitivity"],
            "guard_language_sensitivity": min(sensitivity["language_accuracy"].values())
            >= gates["minimum_language_guard_sensitivity"],
            "guard_family_sensitivity": min(sensitivity["family_accuracy"].values())
            >= gates["minimum_family_guard_sensitivity"],
        }
    )
    passed = all(gate_results.values())
    report = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "method": "thresholded parent-logit sufficiency guard with reduced family route",
        "contract_fingerprint": lock["fingerprint"],
        "manifest": manifest,
        "guard_metrics": guard_metrics,
        "absolute_metrics": {
            "v0.3-parent": _compact_surface_score(parent),
            "thresholded-guard-route": _compact_surface_score(candidate),
        },
        "comparison": comparison,
        "route_metrics": route_metrics,
        "paired_bootstrap": bootstrap,
        "gates": gate_results,
        "passed": passed,
        "candidate_status": "thresholded-guard-confirmed" if passed else "rejected",
        "proceed_to_historical_external_benchmarks": passed,
        "automatic_champion_promotion": False,
        "fresh_confirmation_surface_opened": True,
        "fresh_confirmation_surface_retired": True,
        "claim_boundary": (
            "Fresh paired synthetic confirmation does not establish performance on historical or "
            "new external statistical benchmarks."
        ),
        "private_scores": {
            "v0.3-parent": parent,
            "unguarded-reduced-route": unguarded,
            "thresholded-guard-route": candidate,
        },
    }
    write_json(report_path, report)
    public = copy.deepcopy(report)
    public.pop("private_scores")
    write_json(public_path, public)
    del model, tokenizer, router, guard
    gc.collect()
    mx.clear_cache()
    return public

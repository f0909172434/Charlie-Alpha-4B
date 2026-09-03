from __future__ import annotations

import gc
import json
from collections import Counter, defaultdict
from typing import Any

import mlx.core as mx
import numpy as np
from datasets import load_dataset
from mlx_lm import load

from .config import ProjectConfig
from .io_utils import canonical_hash, sha256_file, write_json
from .stats_agent import StatsAgent
from .stats_eval import _download_statqa, _run_pbench, _run_statqa
from .stats_family_router import _expert_context, _route_slug
from .stats_llm_router import ParentLetterRouter, _llm_router_root, family_route_prompt
from .stats_router_reduced import _reduced_mapping
from .stats_sufficiency_guard import sufficiency_prompt
from .stats_training import _stats_snapshot

_EVALUATOR_VERSION = 2
_MINIMUM_INSPECTION_OUTPUT_BYTES = 128 * 1024


def _historical_runtime_config(config: ProjectConfig) -> ProjectConfig:
    """Preserve complete inspection JSON for high-dimensional historical tasks."""
    values = dict(config.values)
    stats_tools = dict(config.section("stats_tools"))
    stats_tools["max_output_bytes"] = max(
        int(stats_tools["max_output_bytes"]),
        _MINIMUM_INSPECTION_OUTPUT_BYTES,
    )
    values["stats_tools"] = stats_tools
    return ProjectConfig(
        path=config.path,
        root=config.root,
        values=values,
        sources=config.sources,
    )


def prepare_historical_external_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = config.section("router_historical_external")
    root = _llm_router_root(config) / "historical-external-v1"
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "lock.json"
    public_path = config.root / "reports" / "evolve" / "router-historical-external-contract.json"
    candidate_path = _llm_router_root(config) / "sufficiency-guard-thresholded-v1" / "report.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if not candidate.get("passed"):
        raise RuntimeError("Historical external evaluation requires the confirmed guarded route")
    evaluation_lock_path = config.path_for("eval_lock")
    evaluation_lock = json.loads(evaluation_lock_path.read_text(encoding="utf-8"))
    parent_path = config.root / "reports" / "stats" / "generated" / "evaluation-dgp-regret.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    expert_report, adapter_paths = _expert_context(config)
    mapping = _reduced_mapping(
        expert_report["selection"]["mapping"],
        str(settings["excluded_family"]),
    )
    fields = {
        "schema_version": 1,
        "method": "preregistered historical-external falsification gate",
        "research_question": (
            "Does the confirmed reduced family route improve historical P-Bench and StatQA over "
            "the v0.3 parent without regressing exact, method-set, or column-set ability?"
        ),
        "candidate": {
            "guarded_route_fingerprint": candidate["fingerprint"],
            "guarded_route_sha256": sha256_file(candidate_path),
            "mapping_fingerprint": canonical_hash(mapping),
            "family_prompt_sha256": canonical_hash(family_route_prompt()),
            "sufficiency_prompt_sha256": canonical_hash(sufficiency_prompt()),
            "task_aware_guard_scope": (
                "The sufficiency guard applies to primary-method selection prompts. P-Bench uses "
                "the existing data compiler and StatQA is an explicit method/column extraction "
                "task, so neither is intercepted by the guard."
            ),
            "adapter_sha256": {
                slug: sha256_file(path / "adapters.safetensors")
                for slug, path in sorted(adapter_paths.items())
            },
        },
        "control": {
            "name": "v0.3-parent",
            "report_sha256": sha256_file(parent_path),
            "report_fingerprint": parent["fingerprint"],
            "adapter_sha256": parent["adapter_sha256"],
        },
        "evaluation": {
            "lock_fingerprint": evaluation_lock["fingerprint"],
            "lock_sha256": sha256_file(evaluation_lock_path),
            "p_bench_count": int(evaluation_lock["p_bench"]["count"]),
            "statqa_count": int(evaluation_lock["statqa"]["count"]),
            "historical_and_previously_opened": True,
            "formal_final_claim_allowed": False,
        },
        "precontract_routing_only_diagnostic": {
            "candidate_outputs_generated": False,
            "gold_labels_consulted_for_candidate_design": False,
            "p_bench": {
                "count": 90,
                "guard_insufficient_rate_if_misapplied": 8 / 90,
                "route_counts": {"parent": 53, "group_comparison": 21, "categorical": 16},
            },
            "statqa": {
                "count": 200,
                "guard_insufficient_rate_if_misapplied": 198 / 200,
                "route_counts": {
                    "parent": 158,
                    "categorical": 34,
                    "group_comparison": 7,
                    "clustered_repeated": 1,
                },
            },
        },
        "settings": settings,
        "adaptation_policy": "none; no output-based changes to routing, prompts, experts, or gates",
        "stopping_rule": "complete both locked historical suites; resumable progress only",
        "decision_rule": (
            "Pass only if every P-Bench, StatQA, clarification, retention, and integrity gate "
            "passes. Passing is development evidence only; failing blocks champion promotion."
        ),
    }
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        payload = {key: value for key, value in existing.items() if key != "fingerprint"}
        if canonical_hash(payload) != existing.get("fingerprint"):
            raise RuntimeError("Historical-external lock fingerprint is corrupt")
        for key in fields:
            if existing.get(key) != fields[key]:
                raise RuntimeError(f"Frozen historical-external lock changed: {key}")
        write_json(public_path, existing)
        return existing
    lock = fields
    lock["fingerprint"] = canonical_hash(lock)
    write_json(lock_path, lock)
    write_json(public_path, lock)
    return lock


def _aggregate_statqa(details: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(details)
    return {
        "count": count,
        "accuracy": sum(bool(row["exact_correct"]) for row in details) / count,
        "method_set_accuracy": sum(bool(row["method_correct"]) for row in details) / count,
        "column_set_accuracy": sum(bool(row["columns_correct"]) for row in details) / count,
        "column_recall": float(np.mean([float(row["column_recall"]) for row in details])),
        "details": details,
    }


def _aggregate_pbench(details: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(details)
    return {
        "count": count,
        "raw_accuracy": sum(bool(row["raw_correct"]) for row in details) / count,
        "strict_accuracy": sum(bool(row["strict_correct"]) for row in details) / count,
        "details": details,
    }


def run_historical_external_evaluation(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    lock = prepare_historical_external_contract(config)
    settings = config.section("router_historical_external")
    gates = settings["gates"]
    root = _llm_router_root(config) / "historical-external-v1"
    report_path = root / "report.json"
    public_path = config.root / "reports" / "evolve" / "router-historical-external.json"
    fingerprint = canonical_hash(
        {"lock": lock["fingerprint"], "evaluator_version": _EVALUATOR_VERSION}
    )
    if report_path.exists() and not force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            public = dict(existing)
            public.pop("private_details", None)
            write_json(public_path, public)
            return public
        raise RuntimeError("Historical-external report changed")

    evaluation_lock = json.loads(config.path_for("eval_lock").read_text(encoding="utf-8"))
    statqa_path = _download_statqa(config)
    statqa_rows = json.loads(statqa_path.read_text(encoding="utf-8"))
    pbench_source = config.sources["datasets"]["p_bench_eval"]
    pbench = load_dataset(
        pbench_source["repo_id"],
        split=pbench_source["split"],
        revision=pbench_source["revision"],
    )
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
    threshold = float(settings["family_route_threshold"])
    by_slug_statqa: dict[str, list[int]] = defaultdict(list)
    by_slug_pbench: dict[str, list[dict[str, Any]]] = defaultdict(list)
    route_counts: dict[str, Counter[str]] = {
        "statqa": Counter(),
        "p_bench": Counter(),
    }
    for index in evaluation_lock["statqa"]["indices"]:
        family, confidence, _ = router.predict(str(statqa_rows[int(index)]["refined_question"]))
        slug = _route_slug(
            {"predicted_family_id": family, "confidence": confidence},
            mapping,
            threshold,
        )
        by_slug_statqa[slug].append(int(index))
        route_counts["statqa"][slug] += 1
    for task in evaluation_lock["p_bench"]["tasks"]:
        family, confidence, _ = router.predict(str(pbench[int(task["index"])]["question"]))
        slug = _route_slug(
            {"predicted_family_id": family, "confidence": confidence},
            mapping,
            threshold,
        )
        by_slug_pbench[slug].append(task)
        route_counts["p_bench"][slug] += 1
    del model, tokenizer, router
    gc.collect()
    mx.clear_cache()

    statqa_details: list[dict[str, Any]] = []
    pbench_details: list[dict[str, Any]] = []
    progress_root = root / "progress"
    runtime_config = _historical_runtime_config(config)
    for slug in sorted(set(by_slug_statqa) | set(by_slug_pbench)):
        agent = StatsAgent(runtime_config, adapter_path=adapter_paths[slug])
        if by_slug_statqa.get(slug):
            result = _run_statqa(
                agent,
                statqa_rows,
                by_slug_statqa[slug],
                route="stats",
                progress_path=progress_root / f"statqa-{slug}.jsonl",
                progress_fingerprint=canonical_hash(
                    {"evaluation": fingerprint, "suite": "statqa", "slug": slug}
                ),
            )
            statqa_details.extend(result["details"])
        if by_slug_pbench.get(slug):
            result = _run_pbench(
                config,
                agent,
                pbench,
                by_slug_pbench[slug],
                route="stats",
                progress_path=progress_root / f"p-bench-{slug}.jsonl",
                progress_fingerprint=canonical_hash(
                    {"evaluation": fingerprint, "suite": "p-bench", "slug": slug}
                ),
            )
            pbench_details.extend(result["details"])
        del agent
        gc.collect()
        mx.clear_cache()
    statqa = _aggregate_statqa(statqa_details)
    pbench_result = _aggregate_pbench(pbench_details)
    parent_path = config.root / "reports" / "stats" / "generated" / "evaluation-dgp-regret.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    guard_report = json.loads(
        (_llm_router_root(config) / "sufficiency-guard-thresholded-v1" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    pbench_delta = 100 * (
        float(pbench_result["raw_accuracy"]) - float(parent["p_bench"]["raw_accuracy"])
    )
    statqa_delta = 100 * (float(statqa["accuracy"]) - float(parent["statqa"]["accuracy"]))
    gate_results = {
        "p_bench_raw_improvement": pbench_delta >= float(gates["p_bench_raw_points"]),
        "p_bench_strict_noninferiority": float(pbench_result["strict_accuracy"])
        >= float(parent["p_bench"]["strict_accuracy"]),
        "statqa_exact_improvement": statqa_delta >= float(gates["statqa_exact_points"]),
        "statqa_method_set_noninferiority": float(statqa["method_set_accuracy"])
        >= float(parent["statqa"]["method_set_accuracy"])
        - float(gates["maximum_statqa_submetric_regression_points"]) / 100,
        "statqa_column_set_noninferiority": float(statqa["column_set_accuracy"])
        >= float(parent["statqa"]["column_set_accuracy"])
        - float(gates["maximum_statqa_submetric_regression_points"]) / 100,
        "fresh_guard_specificity": float(
            guard_report["guard_metrics"]["complete_specificity"]["accuracy"]
        )
        >= float(gates["minimum_guard_specificity"]),
        "fresh_guard_sensitivity": float(
            guard_report["guard_metrics"]["incomplete_sensitivity"]["accuracy"]
        )
        >= float(gates["minimum_guard_sensitivity"]),
        "retention_bypass": float(parent["retention"]["accuracy"])
        >= float(gates["minimum_retention_accuracy"]),
        "source_integrity": sha256_file(config.path_for("eval_lock"))
        == lock["evaluation"]["lock_sha256"],
    }
    passed = all(gate_results.values())
    report = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "evaluator_version": _EVALUATOR_VERSION,
        "method": "historical-external routed-adapter falsification",
        "lock_fingerprint": lock["fingerprint"],
        "route_counts": {key: dict(value) for key, value in route_counts.items()},
        "absolute_metrics": {
            "v0.3-parent": {
                "p_bench_raw_accuracy": parent["p_bench"]["raw_accuracy"],
                "p_bench_strict_accuracy": parent["p_bench"]["strict_accuracy"],
                "statqa_exact_accuracy": parent["statqa"]["accuracy"],
                "statqa_method_set_accuracy": parent["statqa"]["method_set_accuracy"],
                "statqa_column_set_accuracy": parent["statqa"]["column_set_accuracy"],
            },
            "guarded-reduced-route": {
                "p_bench_raw_accuracy": pbench_result["raw_accuracy"],
                "p_bench_strict_accuracy": pbench_result["strict_accuracy"],
                "statqa_exact_accuracy": statqa["accuracy"],
                "statqa_method_set_accuracy": statqa["method_set_accuracy"],
                "statqa_column_set_accuracy": statqa["column_set_accuracy"],
                "statqa_column_recall": statqa["column_recall"],
            },
        },
        "deltas_points": {"p_bench_raw": pbench_delta, "statqa_exact": statqa_delta},
        "gates": gate_results,
        "passed": passed,
        "candidate_status": "historical-external-survivor" if passed else "external-rejected",
        "proceed_to_new_external_final_design": passed,
        "automatic_champion_promotion": False,
        "operational_compatibility": {
            "minimum_inspection_output_bytes": _MINIMUM_INSPECTION_OUTPUT_BYTES,
            "reason": (
                "The selected historical suite contains a 351-column input whose complete "
                "inspection JSON is 86,215 bytes. Evaluator v1 stopped at the 65,536-byte "
                "sandbox transport ceiling before producing a score. Evaluator v2 changes only "
                "that ceiling and recomputes every task under one fingerprint."
            ),
            "changes_model_prompt_adapter_source_or_gate": False,
        },
        "interpretation": (
            "The candidate survived historical external falsification but still requires a new "
            "independent external final."
            if passed
            else "Historical external evidence falsified the candidate as a champion replacement."
        ),
        "claim_boundary": (
            "These P-Bench and StatQA tasks were previously opened for v0.3 and cannot support a "
            "fresh final or publication claim."
        ),
        "private_details": {"p_bench": pbench_result, "statqa": statqa},
    }
    write_json(report_path, report)
    public = dict(report)
    public.pop("private_details")
    write_json(public_path, public)
    return public

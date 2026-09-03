from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from datasets import load_dataset

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json
from .stats_agent import StatsAgent
from .stats_eval import _run_pbench
from .stats_family_router import _expert_context
from .stats_llm_router import _llm_router_root
from .stats_router_external import _aggregate_pbench, _aggregate_statqa, _historical_runtime_config
from .stats_router_replay import _transition_summary

_COUNTERFACTUAL_REPLAY_VERSION = 1


def _implementation_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "stats_router_counterfactual.py": sha256_file(Path(__file__)),
        "stats_router_external.py": sha256_file(root / "stats_router_external.py"),
        "stats_router_replay.py": sha256_file(root / "stats_router_replay.py"),
        "stats_eval.py": sha256_file(root / "stats_eval.py"),
        "stats_agent.py": sha256_file(root / "stats_agent.py"),
    }


def _read_progress(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in read_jsonl(path)]


def _compose_matched_pbench(
    *sources: list[dict[str, Any]],
    expected_task_ids: set[str],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for rows in sources:
        for row in rows:
            task_id = str(row["task_id"])
            if task_id in by_id:
                raise RuntimeError(f"Matched P-Bench task was supplied twice: {task_id}")
            by_id[task_id] = row
    if set(by_id) != expected_task_ids:
        missing = sorted(expected_task_ids - set(by_id))
        extra = sorted(set(by_id) - expected_task_ids)
        raise RuntimeError(f"Matched P-Bench coverage differs: missing={missing}, extra={extra}")
    return [by_id[task_id] for task_id in sorted(by_id)]


def _h4_pbench_routes(config: ProjectConfig) -> tuple[dict[str, str], dict[str, Path]]:
    progress_root = _llm_router_root(config) / "historical-external-v1" / "progress"
    paths = {path.stem[len("p-bench-") :]: path for path in progress_root.glob("p-bench-*.jsonl")}
    if "parent" not in paths:
        raise RuntimeError("H4 parent-route P-Bench progress is missing")
    routes: dict[str, str] = {}
    for slug, path in sorted(paths.items()):
        for row in _read_progress(path):
            task_id = str(row["task_id"])
            if task_id in routes:
                raise RuntimeError(f"H4 P-Bench task appears in multiple routes: {task_id}")
            routes[task_id] = slug
    return routes, paths


def prepare_historical_counterfactual_replay_contract(config: ProjectConfig) -> dict[str, Any]:
    root = _llm_router_root(config) / "historical-counterfactual-replay-v2"
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "lock.json"
    public_path = (
        config.root / "reports" / "evolve" / "router-historical-counterfactual-replay-contract.json"
    )

    h4_report_path = _llm_router_root(config) / "historical-external-v1" / "report.json"
    h4_contract_path = (
        config.root / "reports" / "evolve" / "router-historical-external-contract.json"
    )
    v1_root = _llm_router_root(config) / "historical-matched-replay-v1"
    v1_lock_path = v1_root / "lock.json"
    v1_statqa_path = v1_root / "progress" / "statqa-parent-matched.jsonl"
    v1_statqa_status = v1_statqa_path.with_suffix(".status.json")
    v1_pbench_path = v1_root / "progress" / "p-bench-parent-matched.jsonl"
    v1_pbench_status = v1_pbench_path.with_suffix(".status.json")
    required = (
        h4_report_path,
        h4_contract_path,
        v1_lock_path,
        v1_statqa_path,
        v1_statqa_status,
        v1_pbench_path,
        v1_pbench_status,
    )
    if not all(path.exists() for path in required):
        raise RuntimeError(
            "Counterfactual replay requires the completed H4 evidence and v1 partial replay"
        )

    h4_report = json.loads(h4_report_path.read_text(encoding="utf-8"))
    h4_contract = json.loads(h4_contract_path.read_text(encoding="utf-8"))
    v1_lock = json.loads(v1_lock_path.read_text(encoding="utf-8"))
    statqa_status = json.loads(v1_statqa_status.read_text(encoding="utf-8"))
    pbench_status = json.loads(v1_pbench_status.read_text(encoding="utf-8"))
    if int(statqa_status.get("completed", -1)) != 200:
        raise RuntimeError("v1 StatQA matched replay must be complete before v2")

    evaluation_lock_path = config.path_for("eval_lock")
    evaluation_lock = json.loads(evaluation_lock_path.read_text(encoding="utf-8"))
    expected_ids = {str(task["task_id"]) for task in evaluation_lock["p_bench"]["tasks"]}
    routes, h4_paths = _h4_pbench_routes(config)
    if set(routes) != expected_ids:
        raise RuntimeError("Frozen H4 P-Bench routing does not cover the evaluation lock")
    parent_ids = {task_id for task_id, slug in routes.items() if slug == "parent"}
    expert_ids = expected_ids - parent_ids

    v1_pbench_rows = _read_progress(v1_pbench_path)
    v1_expert_ids = {
        str(row["task_id"]) for row in v1_pbench_rows if str(row["task_id"]) in expert_ids
    }
    remaining_expert_ids = expert_ids - v1_expert_ids
    _, adapter_paths = _expert_context(config)
    parent_adapter_sha = sha256_file(adapter_paths["parent"] / "adapters.safetensors")
    if parent_adapter_sha != h4_contract["control"]["adapter_sha256"]:
        raise RuntimeError("Current parent adapter differs from the frozen H4 control")

    fields: dict[str, Any] = {
        "schema_version": 1,
        "method": "post-H4 matched counterfactual parent completion",
        "research_question": (
            "After removing evaluator/runtime drift, does the frozen H4 routed policy change "
            "historical P-Bench or StatQA behavior relative to the same v0.3 parent?"
        ),
        "supersedes": {
            "contract_fingerprint": v1_lock["fingerprint"],
            "reason": (
                "v1 redundantly scheduled all 90 P-Bench parent replays. H4 already contains exact "
                "same-evaluator parent outputs for the 53 tasks frozen to the parent route. v2 "
                "reuses those outputs and replays only expert-routed counterfactual parent tasks."
            ),
            "outcome_independent_subset": True,
            "v1_final_report_generated": (v1_root / "report.json").exists(),
        },
        "h4_candidate": {
            "report_fingerprint": h4_report["fingerprint"],
            "report_sha256": sha256_file(h4_report_path),
            "terminal_status": h4_report["candidate_status"],
        },
        "matched_control": {
            "adapter_sha256": parent_adapter_sha,
            "statqa_rows_reused_from_v1": int(statqa_status["completed"]),
            "p_bench_parent_route_rows_reused_from_h4": len(parent_ids),
            "p_bench_expert_rows_reused_from_v1": len(v1_expert_ids),
            "p_bench_expert_rows_remaining": len(remaining_expert_ids),
            "p_bench_total": len(expected_ids),
            "expert_route_task_ids_fingerprint": canonical_hash(sorted(expert_ids)),
            "remaining_task_ids_fingerprint": canonical_hash(sorted(remaining_expert_ids)),
        },
        "reuse_evidence": {
            "v1_statqa_progress_sha256": sha256_file(v1_statqa_path),
            "v1_statqa_progress_fingerprint": statqa_status["fingerprint"],
            "v1_pbench_partial_sha256": sha256_file(v1_pbench_path),
            "v1_pbench_partial_fingerprint": pbench_status["fingerprint"],
            "v1_pbench_partial_completed": int(pbench_status["completed"]),
            "h4_pbench_progress": {
                slug: {"sha256": sha256_file(path), "count": len(_read_progress(path))}
                for slug, path in sorted(h4_paths.items())
            },
        },
        "evaluation": {
            "lock_fingerprint": evaluation_lock["fingerprint"],
            "lock_sha256": sha256_file(evaluation_lock_path),
            "historical_and_previously_opened": True,
            "formal_final_claim_allowed": False,
        },
        "implementation_sha256": _implementation_manifest(),
        "adaptation_policy": "none; no model, prompt, route, expert, threshold, or gate changes",
        "stopping_rule": (
            "Complete the frozen expert-routed counterfactual parent set; compose it with exact H4 "
            "parent-route outputs and the complete v1 StatQA matched replay."
        ),
        "decision_policy": (
            "Attribute mechanism effect only from matched parent versus frozen H4 candidate. "
            "Legacy-control drift is reported separately. H4 remains external-rejected."
        ),
    }
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        payload = {key: value for key, value in existing.items() if key != "fingerprint"}
        if canonical_hash(payload) != existing.get("fingerprint"):
            raise RuntimeError("Counterfactual-replay lock fingerprint is corrupt")
        for key, value in fields.items():
            if existing.get(key) != value:
                raise RuntimeError(f"Frozen counterfactual-replay lock changed: {key}")
        write_json(public_path, existing)
        return existing

    lock = fields
    lock["fingerprint"] = canonical_hash(lock)
    write_json(lock_path, lock)
    write_json(public_path, lock)
    write_json(
        config.root / "reports" / "evolve" / "router-historical-matched-replay-v1-superseded.json",
        {
            "schema_version": 1,
            "status": "superseded-before-completion",
            "contract_fingerprint": v1_lock["fingerprint"],
            "statqa_completed": int(statqa_status["completed"]),
            "p_bench_completed": int(pbench_status["completed"]),
            "p_bench_progress_sha256": sha256_file(v1_pbench_path),
            "reason": fields["supersedes"]["reason"],
            "replacement_contract_fingerprint": lock["fingerprint"],
            "research_decision_made_from_v1_partial": False,
        },
    )
    return lock


def run_historical_counterfactual_replay(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    lock = prepare_historical_counterfactual_replay_contract(config)
    root = _llm_router_root(config) / "historical-counterfactual-replay-v2"
    report_path = root / "report.json"
    public_path = (
        config.root / "reports" / "evolve" / "router-historical-counterfactual-replay.json"
    )
    fingerprint = canonical_hash(
        {
            "lock": lock["fingerprint"],
            "counterfactual_replay_version": _COUNTERFACTUAL_REPLAY_VERSION,
        }
    )
    if report_path.exists() and not force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            public = dict(existing)
            public.pop("private_details", None)
            write_json(public_path, public)
            return public
        raise RuntimeError("Counterfactual replay report changed")

    started = time.monotonic()
    evaluation_lock = json.loads(config.path_for("eval_lock").read_text(encoding="utf-8"))
    expected_ids = {str(task["task_id"]) for task in evaluation_lock["p_bench"]["tasks"]}
    routes, h4_paths = _h4_pbench_routes(config)
    parent_ids = {task_id for task_id, slug in routes.items() if slug == "parent"}
    expert_ids = expected_ids - parent_ids
    h4_parent_rows = _read_progress(h4_paths["parent"])

    v1_root = _llm_router_root(config) / "historical-matched-replay-v1" / "progress"
    statqa_rows = _read_progress(v1_root / "statqa-parent-matched.jsonl")
    v1_pbench_rows = _read_progress(v1_root / "p-bench-parent-matched.jsonl")
    v1_expert_rows = [row for row in v1_pbench_rows if str(row["task_id"]) in expert_ids]
    completed_expert_ids = {str(row["task_id"]) for row in v1_expert_rows}
    remaining_ids = expert_ids - completed_expert_ids

    tasks = [
        task
        for task in evaluation_lock["p_bench"]["tasks"]
        if str(task["task_id"]) in remaining_ids
    ]
    replay_rows: list[dict[str, Any]] = []
    if tasks:
        _, adapter_paths = _expert_context(config)
        runtime_config = _historical_runtime_config(config)
        agent = StatsAgent(runtime_config, adapter_path=adapter_paths["parent"])
        pbench_source = config.sources["datasets"]["p_bench_eval"]
        pbench_dataset = load_dataset(
            pbench_source["repo_id"],
            split=pbench_source["split"],
            revision=pbench_source["revision"],
        )
        replay = _run_pbench(
            runtime_config,
            agent,
            pbench_dataset,
            tasks,
            route="stats",
            progress_path=root / "progress" / "p-bench-parent-counterfactual.jsonl",
            progress_fingerprint=canonical_hash(
                {"evaluation": fingerprint, "suite": "p-bench", "control": "v0.3-parent"}
            ),
        )
        replay_rows = list(replay["details"])
        del agent
        gc.collect()
        mx.clear_cache()

    matched_pbench_rows = _compose_matched_pbench(
        h4_parent_rows,
        v1_expert_rows,
        replay_rows,
        expected_task_ids=expected_ids,
    )
    matched_pbench = _aggregate_pbench(matched_pbench_rows)
    matched_statqa = _aggregate_statqa(statqa_rows)

    h4_report = json.loads(
        (_llm_router_root(config) / "historical-external-v1" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    candidate = h4_report["private_details"]
    candidate_pbench = _aggregate_pbench(list(candidate["p_bench"]["details"]))
    candidate_statqa = _aggregate_statqa(list(candidate["statqa"]["details"]))
    legacy_parent = json.loads(
        (config.root / "reports" / "stats" / "generated" / "evaluation-dgp-regret.json").read_text(
            encoding="utf-8"
        )
    )

    legacy_pbench_drift = _transition_summary(
        list(legacy_parent["p_bench"]["details"]),
        matched_pbench_rows,
        id_field="task_id",
        metric_fields=("raw_correct", "strict_correct"),
    )
    matched_pbench_effect = _transition_summary(
        matched_pbench_rows,
        list(candidate_pbench["details"]),
        id_field="task_id",
        metric_fields=("raw_correct", "strict_correct"),
    )
    legacy_statqa_drift = _transition_summary(
        list(legacy_parent["statqa"]["details"]),
        statqa_rows,
        id_field="index",
        metric_fields=("exact_correct", "method_correct", "columns_correct"),
    )
    matched_statqa_effect = _transition_summary(
        statqa_rows,
        list(candidate_statqa["details"]),
        id_field="index",
        metric_fields=("exact_correct", "method_correct", "columns_correct"),
    )

    pbench_delta = matched_pbench_effect["raw_correct"]["delta_points"]
    statqa_delta = matched_statqa_effect["exact_correct"]["delta_points"]
    if pbench_delta <= 0 and statqa_delta == 0:
        next_direction = "representation-transfer"
        next_direction_reason = (
            "The frozen H4 router/experts add no matched historical accuracy over v0.3. Synthetic "
            "routing gains therefore do not transfer to either historical task format."
        )
    elif statqa_delta == 0:
        next_direction = "task-format-transfer"
        next_direction_reason = (
            "A matched P-Bench effect survives, but StatQA remains behaviorally inert. The next "
            "study should target method/column extraction transfer rather than more router scaling."
        )
    else:
        next_direction = "reconcile-statqa-discrepancy"
        next_direction_reason = (
            "Matched StatQA behavior changed, contradicting the earlier invariance diagnostic; "
            "resolve that discrepancy before any training."
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "counterfactual_replay_version": _COUNTERFACTUAL_REPLAY_VERSION,
        "method": "post-H4 matched counterfactual parent completion",
        "lock_fingerprint": lock["fingerprint"],
        "evidence_composition": {
            "statqa_matched_parent_rows_from_v1": len(statqa_rows),
            "p_bench_matched_parent_rows_from_h4_parent_route": len(h4_parent_rows),
            "p_bench_matched_parent_rows_from_v1_counterfactual": len(v1_expert_rows),
            "p_bench_matched_parent_rows_newly_replayed": len(replay_rows),
            "p_bench_total": len(matched_pbench_rows),
        },
        "legacy_control_drift": {
            "p_bench": legacy_pbench_drift,
            "statqa": legacy_statqa_drift,
        },
        "matched_candidate_effect": {
            "p_bench": matched_pbench_effect,
            "statqa": matched_statqa_effect,
        },
        "absolute_metrics": {
            "legacy-v0.3-parent": {
                "p_bench_raw_accuracy": legacy_parent["p_bench"]["raw_accuracy"],
                "p_bench_strict_accuracy": legacy_parent["p_bench"]["strict_accuracy"],
                "statqa_exact_accuracy": legacy_parent["statqa"]["accuracy"],
            },
            "matched-v0.3-parent": {
                "p_bench_raw_accuracy": matched_pbench["raw_accuracy"],
                "p_bench_strict_accuracy": matched_pbench["strict_accuracy"],
                "statqa_exact_accuracy": matched_statqa["accuracy"],
            },
            "h4-guarded-reduced-route": {
                "p_bench_raw_accuracy": candidate_pbench["raw_accuracy"],
                "p_bench_strict_accuracy": candidate_pbench["strict_accuracy"],
                "statqa_exact_accuracy": candidate_statqa["accuracy"],
            },
        },
        "diagnostic_flags": {
            "legacy_pbench_drift_observed": (
                legacy_pbench_drift["raw_correct"]["changed_correctness"] > 0
            ),
            "legacy_statqa_drift_observed": (
                legacy_statqa_drift["exact_correct"]["changed_correctness"] > 0
            ),
            "matched_h4_pbench_effect_observed": (
                matched_pbench_effect["raw_correct"]["changed_correctness"] > 0
            ),
            "matched_h4_statqa_effect_observed": (
                matched_statqa_effect["exact_correct"]["changed_correctness"] > 0
            ),
        },
        "next_research_direction": next_direction,
        "next_research_direction_reason": next_direction_reason,
        "h4_terminal_status_unchanged": "external-rejected",
        "automatic_champion_promotion": False,
        "proceed_to_external_final": False,
        "elapsed_seconds": time.monotonic() - started,
        "private_details": {
            "matched_parent": {"p_bench": matched_pbench, "statqa": matched_statqa},
        },
    }
    write_json(report_path, report)
    public = dict(report)
    public.pop("private_details")
    write_json(public_path, public)
    return public

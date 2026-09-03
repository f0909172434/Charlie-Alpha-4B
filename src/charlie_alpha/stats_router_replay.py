from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from datasets import load_dataset

from .config import ProjectConfig
from .io_utils import canonical_hash, sha256_file, write_json
from .stats_agent import StatsAgent
from .stats_eval import _download_statqa, _run_pbench, _run_statqa
from .stats_family_router import _expert_context
from .stats_llm_router import _llm_router_root
from .stats_router_external import (
    _EVALUATOR_VERSION as _H4_EVALUATOR_VERSION,
)
from .stats_router_external import (
    _aggregate_pbench,
    _aggregate_statqa,
    _historical_runtime_config,
)

_MATCHED_REPLAY_VERSION = 1


def _implementation_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "stats_router_replay.py": sha256_file(Path(__file__)),
        "stats_router_external.py": sha256_file(root / "stats_router_external.py"),
        "stats_eval.py": sha256_file(root / "stats_eval.py"),
        "stats_agent.py": sha256_file(root / "stats_agent.py"),
    }


def _transition_summary(
    parent_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    id_field: str,
    metric_fields: tuple[str, ...],
) -> dict[str, Any]:
    parent = {str(row[id_field]): row for row in parent_rows}
    candidate = {str(row[id_field]): row for row in candidate_rows}
    if set(parent) != set(candidate):
        raise RuntimeError(f"Paired replay coverage differs for {id_field}")

    metrics: dict[str, Any] = {}
    for field in metric_fields:
        false_false = false_true = true_false = true_true = 0
        for key in sorted(parent):
            before = bool(parent[key][field])
            after = bool(candidate[key][field])
            if before and after:
                true_true += 1
            elif before:
                true_false += 1
            elif after:
                false_true += 1
            else:
                false_false += 1
        count = len(parent)
        parent_accuracy = (true_true + true_false) / count
        candidate_accuracy = (true_true + false_true) / count
        metrics[field] = {
            "count": count,
            "false_to_false": false_false,
            "false_to_true": false_true,
            "true_to_false": true_false,
            "true_to_true": true_true,
            "parent_accuracy": parent_accuracy,
            "candidate_accuracy": candidate_accuracy,
            "delta_points": 100 * (candidate_accuracy - parent_accuracy),
            "changed_correctness": false_true + true_false,
        }
    return metrics


def prepare_historical_matched_replay_contract(config: ProjectConfig) -> dict[str, Any]:
    root = _llm_router_root(config) / "historical-matched-replay-v1"
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "lock.json"
    public_path = (
        config.root / "reports" / "evolve" / "router-historical-matched-replay-contract.json"
    )

    h4_report_path = _llm_router_root(config) / "historical-external-v1" / "report.json"
    h4_contract_path = (
        config.root / "reports" / "evolve" / "router-historical-external-contract.json"
    )
    legacy_parent_path = (
        config.root / "reports" / "stats" / "generated" / "evaluation-dgp-regret.json"
    )
    if (
        not h4_report_path.exists()
        or not h4_contract_path.exists()
        or not legacy_parent_path.exists()
    ):
        raise RuntimeError("Matched replay requires the completed H4 historical evaluation")

    h4_report = json.loads(h4_report_path.read_text(encoding="utf-8"))
    h4_contract = json.loads(h4_contract_path.read_text(encoding="utf-8"))
    legacy_parent = json.loads(legacy_parent_path.read_text(encoding="utf-8"))
    if h4_report.get("candidate_status") != "external-rejected":
        raise RuntimeError("Matched replay is only defined for the closed, rejected H4 study")
    if int(h4_report.get("evaluator_version", -1)) != _H4_EVALUATOR_VERSION:
        raise RuntimeError("H4 evaluator version changed")

    evaluation_lock_path = config.path_for("eval_lock")
    evaluation_lock = json.loads(evaluation_lock_path.read_text(encoding="utf-8"))
    _, adapter_paths = _expert_context(config)
    parent_adapter_sha = sha256_file(adapter_paths["parent"] / "adapters.safetensors")
    if parent_adapter_sha != h4_contract["control"]["adapter_sha256"]:
        raise RuntimeError("Current parent adapter differs from the frozen H4 control")
    if legacy_parent.get("adapter_sha256") != parent_adapter_sha:
        raise RuntimeError("Legacy parent report does not identify the frozen H4 control adapter")

    fields: dict[str, Any] = {
        "schema_version": 1,
        "method": "post-H4 matched historical replay diagnostic",
        "research_question": (
            "How much of the H4 P-Bench and StatQA delta survives when the frozen v0.3 parent is "
            "recomputed under the same evaluator/runtime used for the H4 candidate?"
        ),
        "reason": (
            "The H4 candidate changed P-Bench correctness even on tasks routed to the unchanged "
            "parent adapter, while StatQA correctness was behaviorally unchanged. A matched replay "
            "is required before attributing historical-suite deltas to routing or experts."
        ),
        "h4_candidate": {
            "report_fingerprint": h4_report["fingerprint"],
            "report_sha256": sha256_file(h4_report_path),
            "contract_fingerprint": h4_report["lock_fingerprint"],
            "evaluator_version": int(h4_report["evaluator_version"]),
            "terminal_status": h4_report["candidate_status"],
        },
        "legacy_control": {
            "report_fingerprint": legacy_parent["fingerprint"],
            "report_sha256": sha256_file(legacy_parent_path),
            "adapter_sha256": legacy_parent["adapter_sha256"],
        },
        "matched_control": {
            "name": "v0.3-parent",
            "adapter_sha256": parent_adapter_sha,
            "route": "stats",
            "runtime_policy": "identical historical runtime override used by H4 evaluator v2",
            "p_bench_count": int(evaluation_lock["p_bench"]["count"]),
            "statqa_count": int(evaluation_lock["statqa"]["count"]),
        },
        "evaluation": {
            "lock_fingerprint": evaluation_lock["fingerprint"],
            "lock_sha256": sha256_file(evaluation_lock_path),
            "historical_and_previously_opened": True,
            "formal_final_claim_allowed": False,
        },
        "implementation_sha256": _implementation_manifest(),
        "adaptation_policy": "none; no model, prompt, router, expert, threshold, or gate changes",
        "stopping_rule": "replay every locked P-Bench and StatQA task exactly once, resumably",
        "decision_policy": (
            "Report legacy-control drift separately from matched candidate effect. Never use this "
            "diagnostic to rescue H4, promote a champion, or open a sealed final."
        ),
        "post_h4_boundary": (
            "H4 remains external-rejected regardless of this replay. The result may only choose "
            "the mechanism class for a separately preregistered future study."
        ),
    }
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        payload = {key: value for key, value in existing.items() if key != "fingerprint"}
        if canonical_hash(payload) != existing.get("fingerprint"):
            raise RuntimeError("Matched-replay lock fingerprint is corrupt")
        for key, value in fields.items():
            if existing.get(key) != value:
                raise RuntimeError(f"Frozen matched-replay lock changed: {key}")
        write_json(public_path, existing)
        return existing

    lock = fields
    lock["fingerprint"] = canonical_hash(lock)
    write_json(lock_path, lock)
    write_json(public_path, lock)
    return lock


def run_historical_matched_replay(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    lock = prepare_historical_matched_replay_contract(config)
    root = _llm_router_root(config) / "historical-matched-replay-v1"
    report_path = root / "report.json"
    public_path = config.root / "reports" / "evolve" / "router-historical-matched-replay.json"
    fingerprint = canonical_hash(
        {"lock": lock["fingerprint"], "matched_replay_version": _MATCHED_REPLAY_VERSION}
    )
    if report_path.exists() and not force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            public = dict(existing)
            public.pop("private_details", None)
            write_json(public_path, public)
            return public
        raise RuntimeError("Matched historical replay report changed")

    started = time.monotonic()
    evaluation_lock = json.loads(config.path_for("eval_lock").read_text(encoding="utf-8"))
    h4_report = json.loads(
        (_llm_router_root(config) / "historical-external-v1" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    legacy_parent = json.loads(
        (config.root / "reports" / "stats" / "generated" / "evaluation-dgp-regret.json").read_text(
            encoding="utf-8"
        )
    )
    _, adapter_paths = _expert_context(config)
    runtime_config = _historical_runtime_config(config)
    agent = StatsAgent(runtime_config, adapter_path=adapter_paths["parent"])
    progress_root = root / "progress"

    statqa_path = _download_statqa(config)
    statqa_rows = json.loads(statqa_path.read_text(encoding="utf-8"))
    statqa = _run_statqa(
        agent,
        statqa_rows,
        [int(value) for value in evaluation_lock["statqa"]["indices"]],
        route="stats",
        progress_path=progress_root / "statqa-parent-matched.jsonl",
        progress_fingerprint=canonical_hash(
            {"evaluation": fingerprint, "suite": "statqa", "control": "v0.3-parent"}
        ),
    )

    pbench_source = config.sources["datasets"]["p_bench_eval"]
    pbench_dataset = load_dataset(
        pbench_source["repo_id"],
        split=pbench_source["split"],
        revision=pbench_source["revision"],
    )
    pbench = _run_pbench(
        runtime_config,
        agent,
        pbench_dataset,
        list(evaluation_lock["p_bench"]["tasks"]),
        route="stats",
        progress_path=progress_root / "p-bench-parent-matched.jsonl",
        progress_fingerprint=canonical_hash(
            {"evaluation": fingerprint, "suite": "p-bench", "control": "v0.3-parent"}
        ),
    )
    del agent
    gc.collect()
    mx.clear_cache()

    candidate = h4_report["private_details"]
    legacy_statqa = legacy_parent["statqa"]
    legacy_pbench = legacy_parent["p_bench"]
    candidate_statqa = _aggregate_statqa(list(candidate["statqa"]["details"]))
    candidate_pbench = _aggregate_pbench(list(candidate["p_bench"]["details"]))

    statqa_legacy_drift = _transition_summary(
        list(legacy_statqa["details"]),
        list(statqa["details"]),
        id_field="index",
        metric_fields=("exact_correct", "method_correct", "columns_correct"),
    )
    statqa_matched_effect = _transition_summary(
        list(statqa["details"]),
        list(candidate_statqa["details"]),
        id_field="index",
        metric_fields=("exact_correct", "method_correct", "columns_correct"),
    )
    pbench_legacy_drift = _transition_summary(
        list(legacy_pbench["details"]),
        list(pbench["details"]),
        id_field="task_id",
        metric_fields=("raw_correct", "strict_correct"),
    )
    pbench_matched_effect = _transition_summary(
        list(pbench["details"]),
        list(candidate_pbench["details"]),
        id_field="task_id",
        metric_fields=("raw_correct", "strict_correct"),
    )

    statqa_prediction_changes = sum(
        left.get("predicted_methods") != right.get("predicted_methods")
        for left, right in zip(
            sorted(statqa["details"], key=lambda row: int(row["index"])),
            sorted(candidate_statqa["details"], key=lambda row: int(row["index"])),
            strict=True,
        )
    )
    legacy_pbench_raw_drift = pbench_legacy_drift["raw_correct"]["changed_correctness"]
    matched_pbench_raw_change = pbench_matched_effect["raw_correct"]["changed_correctness"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "matched_replay_version": _MATCHED_REPLAY_VERSION,
        "method": "post-H4 matched historical replay diagnostic",
        "lock_fingerprint": lock["fingerprint"],
        "legacy_control_drift": {
            "statqa": statqa_legacy_drift,
            "p_bench": pbench_legacy_drift,
        },
        "matched_candidate_effect": {
            "statqa": statqa_matched_effect,
            "p_bench": pbench_matched_effect,
            "statqa_predicted_method_rows_changed": statqa_prediction_changes,
        },
        "absolute_metrics": {
            "legacy-v0.3-parent": {
                "p_bench_raw_accuracy": legacy_pbench["raw_accuracy"],
                "p_bench_strict_accuracy": legacy_pbench["strict_accuracy"],
                "statqa_exact_accuracy": legacy_statqa["accuracy"],
                "statqa_method_set_accuracy": legacy_statqa["method_set_accuracy"],
                "statqa_column_set_accuracy": legacy_statqa["column_set_accuracy"],
            },
            "matched-v0.3-parent": {
                "p_bench_raw_accuracy": pbench["raw_accuracy"],
                "p_bench_strict_accuracy": pbench["strict_accuracy"],
                "statqa_exact_accuracy": statqa["accuracy"],
                "statqa_method_set_accuracy": statqa["method_set_accuracy"],
                "statqa_column_set_accuracy": statqa["column_set_accuracy"],
            },
            "h4-guarded-reduced-route": {
                "p_bench_raw_accuracy": candidate_pbench["raw_accuracy"],
                "p_bench_strict_accuracy": candidate_pbench["strict_accuracy"],
                "statqa_exact_accuracy": candidate_statqa["accuracy"],
                "statqa_method_set_accuracy": candidate_statqa["method_set_accuracy"],
                "statqa_column_set_accuracy": candidate_statqa["column_set_accuracy"],
            },
        },
        "diagnostic_flags": {
            "legacy_pbench_runtime_or_evaluator_drift_observed": legacy_pbench_raw_drift > 0,
            "h4_pbench_behavior_differs_from_matched_parent": matched_pbench_raw_change > 0,
            "h4_statqa_exact_behavior_differs_from_matched_parent": (
                statqa_matched_effect["exact_correct"]["changed_correctness"] > 0
            ),
        },
        "interpretation": (
            "Legacy historical metrics are not a matched attribution baseline when correctness "
            "changes under replay. Mechanism conclusions must use matched-v0.3-parent versus the "
            "frozen H4 candidate. H4 remains external-rejected regardless of this diagnostic."
        ),
        "h4_terminal_status_unchanged": "external-rejected",
        "automatic_champion_promotion": False,
        "proceed_to_external_final": False,
        "elapsed_seconds": time.monotonic() - started,
        "private_details": {
            "matched_parent": {"p_bench": pbench, "statqa": statqa},
        },
    }
    write_json(report_path, report)
    public = dict(report)
    public.pop("private_details")
    write_json(public_path, public)
    return public

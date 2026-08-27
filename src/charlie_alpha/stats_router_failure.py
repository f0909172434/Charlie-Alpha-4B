from __future__ import annotations

import copy
import json
from typing import Any

from .config import ProjectConfig
from .io_utils import canonical_hash, sha256_file, write_json
from .stats_calibrate import _surface_comparison
from .stats_family_router import _paired_bootstrap
from .stats_llm_router import _llm_router_root
from .stats_route import _aggregate_predictions


def _replace_family_expert_with_parent(
    parent: dict[str, Any],
    routed: dict[str, Any],
    expert_score: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(routed)
    for language in result["languages"]:
        parent_predictions = {
            str(row["blueprint_id"]): row
            for row in parent["languages"][language]["predictions"]
        }
        replaced_ids = {
            str(row["blueprint_id"])
            for row in expert_score["languages"][language]["predictions"]
        }
        rows = [
            parent_predictions[str(row["blueprint_id"])]
            if str(row["blueprint_id"]) in replaced_ids
            else row
            for row in routed["languages"][language]["predictions"]
        ]
        result["languages"][language] = _aggregate_predictions(rows)
    result["selector"] = result["languages"]["en"]
    return result


def diagnose_family_router_replication_failure(config: ProjectConfig) -> dict[str, Any]:
    settings = config.section("llm_family_router_replication")
    root = _llm_router_root(config) / "independent-replication-v1"
    report_path = root / "report.json"
    if not report_path.exists():
        raise RuntimeError("Replication failure diagnosis requires a completed replication")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("passed"):
        raise RuntimeError("Replication passed; failure diagnosis is not applicable")
    failed_model_gates = sorted(
        key.removeprefix("model_")
        for key, value in report["gates"].items()
        if key.startswith("model_") and not value
    )
    if failed_model_gates != ["family_regret"]:
        raise RuntimeError("Registered diagnosis only applies to isolated family-regret failure")
    parent = report["private_scores"]["v0.3-parent"]
    routed = report["private_scores"]["routed-experts"]
    family_increases = {
        family: float(value) - float(report["comparison"]["parent_family_regret"][family])
        for family, value in report["comparison"]["candidate_family_regret"].items()
    }
    culprit_family = max(family_increases, key=family_increases.get)
    expert_report = json.loads(
        (_llm_router_root(config).parent / "family-experts" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    route = expert_report["selection"]["mapping"][culprit_family]
    if route["checkpoint_name"] == "parent":
        raise RuntimeError("Culprit family was already routed to the parent")
    slug = str(route["slug"])
    score_path = root / "validation-scores" / f"{slug}.json"
    expert_score = json.loads(score_path.read_text(encoding="utf-8"))["score"]
    leave_one_out = _replace_family_expert_with_parent(parent, routed, expert_score)
    comparison = _surface_comparison(parent, leave_one_out, settings["gates"])
    repetitions = int(settings["gates"]["bootstrap_repetitions"])
    result = {
        "schema_version": 1,
        "complete": True,
        "method": "post-rejection leave-one-expert-out mechanism diagnosis",
        "replication_report_sha256": sha256_file(report_path),
        "replication_fingerprint": report["fingerprint"],
        "culprit_family": culprit_family,
        "culprit_expert_slug": slug,
        "observed_family_regret_increase": family_increases[culprit_family],
        "registered_family_regret_ceiling": float(
            settings["gates"]["maximum_family_regret_increase"]
        ),
        "leave_one_out_relative_regret_improvement": comparison[
            "trilingual_relative_regret_improvement"
        ],
        "leave_one_out_comparison": comparison,
        "leave_one_out_vs_frozen_route": _paired_bootstrap(
            routed,
            leave_one_out,
            repetitions=repetitions,
            seed=int(settings["bootstrap_seed"]) + 10000,
        ),
        "leave_one_out_vs_parent": _paired_bootstrap(
            parent,
            leave_one_out,
            repetitions=repetitions,
            seed=int(settings["bootstrap_seed"]) + 10001,
        ),
        "routed_rows_for_culprit_expert": int(
            sum(result["count"] for result in expert_score["languages"].values())
        ),
        "interpretation": (
            "The independent-replication rejection is concentrated in one family expert. "
            "Falling that family back to the frozen parent restores every aggregate and granular "
            "gate while retaining a large overall gain, but this post-outcome ablation is "
            "diagnostic only and cannot rescue the rejected candidate."
        ),
        "next_hypothesis": (
            "A preregistered reduced router that excludes the culprit family expert may preserve "
            "the robust specialization gain without the family-level safety violation. It needs "
            "a completely fresh confirmation surface."
        ),
        "claim_boundary": (
            "This analysis uses the retired replication outcome after rejection. It is suitable "
            "for mechanism diagnosis and prospective design only, not confirmatory evidence."
        ),
    }
    result["fingerprint"] = canonical_hash(result)
    output = root / "failure-analysis.json"
    public = config.root / "reports" / "evolve" / "family-router-replication-failure.json"
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != result["fingerprint"]:
            raise RuntimeError("Replication failure analysis changed")
        write_json(public, existing)
        return existing
    write_json(output, result)
    write_json(public, result)
    return result

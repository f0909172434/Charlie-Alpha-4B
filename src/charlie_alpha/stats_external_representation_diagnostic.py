from __future__ import annotations

import gc
import json
from collections import Counter
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json
from .stats_agent import StatsAgent
from .stats_representation_probe import (
    _METHOD_IDS,
    _extract_representations,
    _fit_ridge_probe,
    _load_representations,
    _probe_metrics,
    _probe_scores,
    _save_representations,
)
from .stats_selector_head import _load_head

_STYLES = ("audit", "researcher", "vignette")


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "external-representation-diagnostic-v1"


def _h15_representation_path(config: ProjectConfig, shard: str, style: str) -> Path:
    return (
        config.path_for("artifact_dir")
        / "style-invariance-v1"
        / "representations"
        / shard
        / f"{style}.npz"
    )


def _fit_h15_probes(config: ProjectConfig, selection: dict[str, Any]) -> dict[str, dict[str, Any]]:
    train = {
        style: _load_representations(_h15_representation_path(config, "training_shard", style))
        for style in _STYLES
    }
    selected = {
        style: _load_representations(_h15_representation_path(config, "selection_shard", style))
        for style in _STYLES
    }
    audit_x = np.concatenate([train["audit"][0], selected["audit"][0]], axis=0)
    audit_y = np.concatenate([train["audit"][1], selected["audit"][1]], axis=0)
    audit_probe = _fit_ridge_probe(
        audit_x,
        audit_y,
        ridge_lambda=float(selection["audit_only_probe"]["ridge_lambda"]),
    )

    diverse_x = np.concatenate(
        [train[style][0] for style in _STYLES] + [selected[style][0] for style in _STYLES],
        axis=0,
    )
    diverse_y = np.concatenate(
        [train[style][1] for style in _STYLES] + [selected[style][1] for style in _STYLES],
        axis=0,
    )
    diverse_probe = _fit_ridge_probe(
        diverse_x,
        diverse_y,
        ridge_lambda=float(selection["style_diverse_probe"]["ridge_lambda"]),
    )
    return {"h15-audit-only": audit_probe, "h15-style-diverse": diverse_probe}


def _model_report(
    model: dict[str, Any],
    vectors: np.ndarray,
    labels: np.ndarray,
    *,
    majority_class: int,
) -> dict[str, Any]:
    observed = set(int(value) for value in model["observed"])
    covered_mask = np.asarray([int(value) in observed for value in labels], dtype=bool)
    all_metrics = _probe_metrics(model, vectors, labels, majority_class=majority_class)
    covered_metrics = (
        _probe_metrics(
            model,
            vectors[covered_mask],
            labels[covered_mask],
            majority_class=majority_class,
        )
        if np.any(covered_mask)
        else None
    )
    return {
        "observed_method_count": len(observed),
        "covered_external_count": int(np.sum(covered_mask)),
        "all_nine": all_metrics,
        "covered_only": covered_metrics,
    }


def run_external_representation_diagnostic(config: ProjectConfig) -> dict[str, Any]:
    e3_public_path = config.root / "reports" / "evolve" / "selector-external-v1.json"
    e3 = json.loads(e3_public_path.read_text(encoding="utf-8"))
    if e3.get("external_selector_head_transfer_supported") is not False:
        raise RuntimeError("Historical representation diagnostic requires terminal-negative E3")

    h15_path = config.root / "reports" / "evolve" / "style-invariance-v1-confirmation.json"
    h15 = json.loads(h15_path.read_text(encoding="utf-8"))
    if not h15.get("h15_diagnosis_confirmed"):
        raise RuntimeError("Historical representation diagnostic requires completed H15")
    if h15.get("next_step") != (
        "stop-style-remapping-and-investigate-external-domain-semantic-shift"
    ):
        raise RuntimeError("H15 did not authorize external semantic-shift diagnosis")
    h15_selection = json.loads(
        (config.root / "reports" / "evolve" / "style-invariance-v1-selection.json").read_text(
            encoding="utf-8"
        )
    )

    h14_contract_path = config.root / "reports" / "evolve" / "selector-head-v1-contract.json"
    h14_contract = json.loads(h14_contract_path.read_text(encoding="utf-8"))
    frozen_head = _load_head(h14_contract)
    models = {"frozen-h14": frozen_head, **_fit_h15_probes(config, h15_selection)}

    e3_cases_path = config.path_for("evolution_dir") / "selector-external-v1" / "cases.jsonl"
    e3_cases = [row for row in read_jsonl(e3_cases_path) if row.get("head_eligible")]
    probe_cases = [
        {
            "case_id": str(row["case_id"]),
            "family_id": "historical-external",
            "question": str(row["question"]),
            "gold_methods": [str(row["gold_method_id"])],
            "gold_columns": [],
        }
        for row in e3_cases
    ]
    representation_path = _root(config) / "e3-eligible-representations.npz"
    if representation_path.exists():
        vectors, labels, case_ids = _load_representations(representation_path)
    else:
        agent = StatsAgent(config, adapter_path=str(h14_contract["parent"]["adapter_path"]))
        agent.router.set_route("adapter")
        try:
            vectors, labels, case_ids = _extract_representations(
                agent, probe_cases, grounded=False
            )
        finally:
            del agent
            gc.collect()
            mx.clear_cache()
        _save_representations(
            representation_path,
            vectors=vectors,
            labels=labels,
            case_ids=case_ids,
        )
    expected_ids = [str(case["case_id"]) for case in probe_cases]
    if case_ids != expected_ids:
        raise RuntimeError("Historical E3 representation order changed")

    majority = Counter(int(value) for value in labels.tolist()).most_common(1)[0][0]
    model_reports = {
        name: _model_report(model, vectors, labels, majority_class=majority)
        for name, model in models.items()
    }
    predictions = {
        name: np.argmax(_probe_scores(model, vectors), axis=1)
        for name, model in models.items()
    }

    private_e3_path = _root(config).parent / "selector-external-v1" / "report-amended-v1.json"
    private_e3 = json.loads(private_e3_path.read_text(encoding="utf-8"))
    control_rows = {
        str(row["case_id"]): row
        for row in private_e3["private_details"]["menu-free-control"]
        if row.get("eligible")
    }
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(e3_cases):
        case_id = str(source["case_id"])
        rows.append(
            {
                "case_id": case_id,
                "gold_method_id": str(source["gold_method_id"]),
                "menu_free_control_correct": bool(control_rows[case_id]["correct"]),
                "predictions": {
                    name: _METHOD_IDS[int(values[index])]
                    for name, values in predictions.items()
                },
            }
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "historical E3 readback with post-H15 frozen probes",
        "historical_only": True,
        "fresh_external_evidence": False,
        "e3_fit_or_tuning_performed": False,
        "e3_result_fingerprint": e3["result_fingerprint"],
        "h15_result_fingerprint": h15["result_fingerprint"],
        "e3_cases_sha256": sha256_file(e3_cases_path),
        "representation_sha256": sha256_file(representation_path),
        "eligible_external_count": len(e3_cases),
        "menu_free_control_accuracy": e3["scores"]["menu-free-control"]["eligible_accuracy"],
        "models": model_reports,
        "rows": rows,
        "diagnosis": (
            "H15 style-diverse linear decoding does not recover E3. The external failure therefore "
            "persists after synthetic style diversification and is more consistent with task/"
            "semantic "
            "distribution shift than a surface-prose-only shift."
        ),
        "next_step": (
            "freeze a new external semantic-bridge mechanism without E3 fitting, then test it on a "
            "genuinely new external source"
        ),
        "claim_boundary": (
            "This reuses already-opened E3 only for diagnosis. It cannot support a new capability "
            "claim, model promotion, release, or selection on E3."
        ),
    }
    report["result_fingerprint"] = canonical_hash(report)
    write_json(_root(config) / "report.json", report)
    write_json(
        config.root / "reports" / "evolve" / "external-representation-diagnostic-v1.json",
        report,
    )
    return report

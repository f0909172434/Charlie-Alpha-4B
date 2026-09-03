from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import stats_external_weight_bridge as bridge
from .config import ProjectConfig
from .io_utils import canonical_hash, sha256_file, write_json

_AMENDMENT_VERSION = 1


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "external-weight-bridge-v1"


def _fixed_evaluation_case(row: dict[str, Any]) -> dict[str, Any]:
    gold = row.get("gold_method_id")
    if gold is None:
        methods = row.get("gold_methods")
        if not isinstance(methods, list) or len(methods) != 1:
            raise RuntimeError("H19 amended evaluation requires exactly one gold method")
        gold = methods[0]
    return {
        "case_id": str(row["case_id"]),
        "question": str(row["question"]),
        "gold_method_id": str(gold),
        "gold_methods": [str(gold)],
        "gold_columns": list(row.get("gold_columns", [])),
        "head_eligible": True,
        "eligible": True,
    }


def prepare_external_weight_bridge_data_amendment(config: ProjectConfig) -> dict[str, Any]:
    contract = bridge.prepare_external_weight_bridge_contract(config)
    root = _root(config)
    lock_path = root / "data-amendment-v1.json"
    public_path = (
        config.root / "reports" / "evolve" / "external-weight-bridge-v1-data-amendment.json"
    )
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("contract_fingerprint") != contract["fingerprint"]:
            raise RuntimeError("H19 data amendment contract changed")
        write_json(public_path, existing)
        return existing
    fold_statuses = list((root / "folds").glob("*/status.json"))
    if fold_statuses:
        raise RuntimeError("H19 data amendment must precede every fold training status")
    partial_folds = sorted(
        path for path in (config.path_for("evolution_dir") / "external-weight-bridge-v1").glob(
            "folds/*/train.jsonl"
        )
    )
    amendment: dict[str, Any] = {
        "schema_version": 1,
        "method": "H19 synthetic-retention field adapter amendment",
        "amendment_version": _AMENDMENT_VERSION,
        "contract_fingerprint": contract["fingerprint"],
        "failure_boundary": {
            "fold_training_started": False,
            "fold_status_count": 0,
            "partial_fold_file_count": len(partial_folds),
            "partial_fold_sha256": {
                path.parent.name: sha256_file(path) for path in partial_folds
            },
            "data_report_written": (_root(config) / "data.json").exists(),
            "fresh_external_evaluation_opened": False,
            "observed_error": "KeyError: gold_method_id",
        },
        "permitted_change": (
            "For H16 synthetic retention rows only, copy the sole frozen gold_methods entry into "
            "the evaluator's gold_method_id field. Questions, labels, fold membership, replay "
            "counts, training records, compute, optimizer, and gates remain unchanged."
        ),
        "implementation_sha256": sha256_file(Path(__file__)),
        "scientific_contract_changed": False,
    }
    amendment["fingerprint"] = canonical_hash(amendment)
    write_json(lock_path, amendment)
    write_json(public_path, amendment)
    return amendment


def _prepare_data_amended(
    config: ProjectConfig,
    amendment: dict[str, Any],
    original_prepare: Any,
) -> dict[str, Any]:
    original_evaluation_case = bridge._evaluation_case
    bridge._evaluation_case = _fixed_evaluation_case
    try:
        data = original_prepare(config)
    finally:
        bridge._evaluation_case = original_evaluation_case
    amended = dict(data)
    amended["data_amendment"] = {
        "fingerprint": amendment["fingerprint"],
        "implementation_sha256": amendment["implementation_sha256"],
        "scientific_contract_changed": False,
    }
    amended.pop("data_fingerprint", None)
    amended["data_fingerprint"] = canonical_hash(amended)
    write_json(_root(config) / "data.json", amended)
    public_path = config.root / "reports" / "evolve" / "external-weight-bridge-v1-data.json"
    write_json(public_path, amended)
    return amended


def prepare_external_weight_bridge_data_amended(config: ProjectConfig) -> dict[str, Any]:
    amendment = prepare_external_weight_bridge_data_amendment(config)
    return _prepare_data_amended(
        config,
        amendment,
        bridge.prepare_external_weight_bridge_data,
    )


def run_external_weight_bridge_training_amended(config: ProjectConfig) -> dict[str, Any]:
    amendment = prepare_external_weight_bridge_data_amendment(config)
    original_prepare = bridge.prepare_external_weight_bridge_data

    def amended_prepare(inner_config: ProjectConfig) -> dict[str, Any]:
        return _prepare_data_amended(inner_config, amendment, original_prepare)

    bridge.prepare_external_weight_bridge_data = amended_prepare
    try:
        result = bridge.run_external_weight_bridge_training(config)
    finally:
        bridge.prepare_external_weight_bridge_data = original_prepare
    if result.get("data_amendment", {}).get("fingerprint") == amendment["fingerprint"]:
        return result
    amended = dict(result)
    amended["pre_amendment_result_fingerprint"] = result["result_fingerprint"]
    amended["data_amendment"] = {
        "fingerprint": amendment["fingerprint"],
        "implementation_sha256": amendment["implementation_sha256"],
        "scientific_contract_changed": False,
    }
    amended.pop("result_fingerprint", None)
    amended["result_fingerprint"] = canonical_hash(amended)
    write_json(_root(config) / "training.json", amended)
    public_path = config.root / "reports" / "evolve" / "external-weight-bridge-v1-training.json"
    write_json(public_path, amended)
    return amended

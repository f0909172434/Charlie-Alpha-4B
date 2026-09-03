from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .io_utils import (
    canonical_hash,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from .stats_catalog_grounding import _catalog_reference
from .stats_catalog_grounding import _messages as _catalog_messages
from .stats_guarded_external import (
    _DECODING,
    _AgentCaller,
    _case_fingerprint,
    _complete_strict_progress,
    _load_strict_progress,
    _paired,
    _parse_prediction,
    _source_paired,
    _start_strict_progress,
    _strict_metrics,
)

_METHOD = "H22_INVALID_CONTROL_CATALOG_FALLBACK_DEV_V1"
_METHOD_VERSION = 1
_EVALUATOR_VERSION = 1
_H21_CONTRACT_FINGERPRINT = "eb8df93c9a511efa8b8635c4db0cbe277b3f3ad38c7d062e43e28885a31fa150"
_H21_DATA_FINGERPRINT = "2f3bffc327c6c4a963cd632e74a197eaf3a81d121303a36e4f4bf0e1c9040d0d"
_H21_RESULT_FINGERPRINT = "a97a2d59e997d92fdfc35fdbba468d55478454072150cf5d6ff675015e004fab"
_H21_REPORT_SHA256 = "b6a827e31f2be5d9660c7a62da21d416eb80f9ff19e157baf6254130569dd3d9"
_H21_CONTROL_LEDGER_SHA256 = "cb460a090e8d1f56bba2b611256cdd3cd48c15f0c031fba63a4f335d9c854939"
_H7_RESULT_FINGERPRINT = "f9e3b27a91b41f11a44cf21fde90f928cf198c36e751b2686dbe41a87809295c"
_H12_RESULT_FINGERPRINT = "b105ee64dcad97e3c55dd9d8fddf205409cd38df1ec8c990551e69088a2ab9a3"

_OPPORTUNITY_GATES = {
    "minimum_invalid_control_cases": 4,
    "minimum_invalid_control_sources": 3,
    "minimum_invalid_control_gold_methods": 2,
    "minimum_distinct_invalid_question_templates": 2,
    "maximum_single_template_fraction": 0.75,
}
_RESULT_GATES = {
    "minimum_valid_fallback_cases": 3,
    "minimum_candidate_only_gains": 3,
    "minimum_net_improvements": 3,
    "minimum_distinct_repaired_question_templates": 2,
    "minimum_repaired_gold_methods": 2,
    "minimum_repair_source_count": 3,
}

ModelCaller = Callable[[str, list[dict[str, str]], dict[str, Any]], str]


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "invalid-control-catalog-fallback-v1"


def _data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "invalid-control-catalog-fallback-v1"


def _reports_root(config: ProjectConfig) -> Path:
    return config.root / "reports" / "evolve"


def _h21_root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "opened-source-residual-repair-v1"


def _write_immutable_json(path: Path, payload: dict[str, Any], *, label: str) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"H22 {label} is immutable")
        return
    write_json(path, payload)


def _write_immutable_jsonl(path: Path, rows: list[dict[str, Any]], *, label: str) -> None:
    if path.exists():
        existing = list(read_jsonl(path))
        if existing != rows:
            raise RuntimeError(f"H22 {label} is immutable")
        return
    write_jsonl(path, rows)


def _implementation_manifest() -> dict[str, str]:
    source_root = Path(__file__).resolve().parent
    names = [
        "stats_invalid_control_catalog_fallback.py",
        "stats_catalog_grounding.py",
        "stats_guarded_external.py",
        "stats_agent.py",
        "stats_catalog.py",
    ]
    return {name: sha256_file(source_root / name) for name in names}


def _load_h21(config: ProjectConfig) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract_path = _h21_root(config) / "contract.json"
    data_path = _h21_root(config) / "data.json"
    report_path = _h21_root(config) / "report.json"
    control_ledger_path = _h21_root(config) / "progress" / "control-ledger.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    data = json.loads(data_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if contract.get("fingerprint") != _H21_CONTRACT_FINGERPRINT:
        raise RuntimeError("H22 requires the sealed H21 contract")
    if data.get("data_fingerprint") != _H21_DATA_FINGERPRINT:
        raise RuntimeError("H22 requires the sealed H21 data")
    if report.get("result_fingerprint") != _H21_RESULT_FINGERPRINT:
        raise RuntimeError("H22 requires the sealed H21 result")
    if sha256_file(report_path) != _H21_REPORT_SHA256:
        raise RuntimeError("H22 H21 report bytes changed")
    if sha256_file(control_ledger_path) != _H21_CONTROL_LEDGER_SHA256:
        raise RuntimeError("H22 H21 control ledger changed")
    if report.get("status") != "INCONCLUSIVE_OPPORTUNITY" or report.get("training_authorized"):
        raise RuntimeError("H22 requires H21 to remain an inconclusive no-training result")
    private = report.get("private_details")
    if not isinstance(private, dict) or report.get("private_details_fingerprint") != canonical_hash(
        private
    ):
        raise RuntimeError("H22 H21 private details changed")
    controls = private.get("control")
    if not isinstance(controls, list) or len(controls) != 24:
        raise RuntimeError("H22 requires all 24 H21 controls")
    return contract, data, report


def _verify_catalog_prior(config: ProjectConfig) -> dict[str, Any]:
    h7_path = _reports_root(config) / "catalog-grounding-v1-confirmation.json"
    h12_path = _reports_root(config) / "catalog-interface-replication-v1.json"
    h7 = json.loads(h7_path.read_text(encoding="utf-8"))
    h12 = json.loads(h12_path.read_text(encoding="utf-8"))
    if h7.get("result_fingerprint") != _H7_RESULT_FINGERPRINT or not h7.get(
        "confirmation_gate", {}
    ).get("passed"):
        raise RuntimeError("H22 requires the confirmed H7 fixed-catalog mechanism")
    if h12.get("result_fingerprint") != _H12_RESULT_FINGERPRINT or not h12.get(
        "replication_gate", {}
    ).get("passed"):
        raise RuntimeError("H22 requires the replicated H12 flat-catalog mechanism")
    return {
        "h7_result_fingerprint": h7["result_fingerprint"],
        "h7_report_sha256": sha256_file(h7_path),
        "h12_result_fingerprint": h12["result_fingerprint"],
        "h12_report_sha256": sha256_file(h12_path),
        "catalog_reference_sha256": sha256_text(_catalog_reference()),
    }


def _fallback_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    return _catalog_messages({"question": str(case["question"])}, grounded=True)


def _question_template(question: str) -> str:
    return " ".join(question.lower().split())


def prepare_invalid_control_catalog_fallback_contract(config: ProjectConfig) -> dict[str, Any]:
    h21_contract, h21_data, h21_report = _load_h21(config)
    catalog_prior = _verify_catalog_prior(config)
    parent = dict(h21_contract["parent_runtime_receipt"])
    parent_weights = Path(str(parent["adapter_path"])) / "adapters.safetensors"
    if sha256_file(parent_weights) != parent["adapter_sha256"]:
        raise RuntimeError("H22 parent adapter changed")

    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": _METHOD,
        "method_version": _METHOD_VERSION,
        "evidence_status": "opened-source-development-composition",
        "causal_question": (
            "When an H21 menu-free parent control is parser-invalid, can the independently "
            "confirmed fixed 28-method catalog prompt repair the canonical-ID interface while "
            "the guard preserves every parser-valid control?"
        ),
        "hypothesis_origin": {
            "h21_contract_fingerprint": h21_contract["fingerprint"],
            "h21_data_fingerprint": h21_data["data_fingerprint"],
            "h21_result_fingerprint": h21_report["result_fingerprint"],
            "h21_report_sha256": _H21_REPORT_SHA256,
            "h21_status": h21_report["status"],
            "h21_residual_parse_reason": "unknown-method-id",
            "h21_residual_case_count": 4,
            "h21_residual_unique_question_templates": 2,
            "h21_training_authorized": False,
        },
        "catalog_prior": catalog_prior,
        "runtime": {
            "control_source": "sealed-H21-parent-control-ledger",
            "control_ledger_sha256": _H21_CONTROL_LEDGER_SHA256,
            "fallback_model": "unchanged-v0.3.0-parent",
            "fallback_adapter": parent,
            "fallback_prompt": "H7 fixed global 28-method catalog prompt",
            "fallback_prompt_sha256": canonical_hash(
                _fallback_messages({"question": "<SOURCE_QUESTION>"})
            ),
            "decoding": dict(_DECODING),
        },
        "routing_policy": [
            (
                "Reuse the sealed H21 parent control output; never call the parent menu-free "
                "control again."
            ),
            (
                "Return every parser-valid H21 control byte-for-byte as the selected method "
                "prediction."
            ),
            (
                "Only for each parser-invalid H21 control, call the unchanged parent once with "
                "the fixed global catalog prompt."
            ),
            "Use a parser-valid catalog fallback; otherwise retain the original invalid control.",
            (
                "Count identical normalized questions as one evidence template even when they "
                "occur in multiple sources."
            ),
        ],
        "opportunity_gates": dict(_OPPORTUNITY_GATES),
        "result_gates": dict(_RESULT_GATES),
        "terminal_states": [
            "SUPPORTED_DEV_COMPOSITION",
            "SCIENTIFIC_FAIL",
            "INCONCLUSIVE_TEMPLATE_COVERAGE",
            "PROTOCOL_INVALID",
        ],
        "stopping_rule": (
            "Open the four frozen H21 invalid-control fallbacks once. Do not add aliases, change "
            "the catalog, choose another model, retry invalid outputs, count duplicate templates "
            "as independent evidence, train an adapter, or modify the result after opening."
        ),
        "implementation_sha256": _implementation_manifest(),
        "model_output_opened": False,
        "fresh_external_evidence": False,
        "training_authorized": False,
        "champion_changed": False,
        "release_authorized": False,
        "claim_boundary": (
            "A pass supports only a development composition of the previously confirmed H7 "
            "catalog interface with an invalid-control identity guard on already-opened H21 cases. "
            "It cannot alter E5, establish fresh external capability, authorize training, replace "
            "the champion, publish, or release."
        ),
    }
    contract["fingerprint"] = canonical_hash(contract)
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    _write_immutable_json(root / "contract.json", contract, label="contract")
    write_json(
        _reports_root(config) / "invalid-control-catalog-fallback-v1-contract.json",
        contract,
    )
    return contract


def _opportunity_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise RuntimeError("H22 opportunity requires invalid controls")
    template_counts = Counter(_question_template(str(case["question"])) for case in cases)
    maximum_share = max(template_counts.values()) / len(cases)
    observed = {
        "invalid_control_cases": len(cases),
        "invalid_control_sources": len({str(case["source_id"]) for case in cases}),
        "invalid_control_gold_methods": len({str(case["gold_method_id"]) for case in cases}),
        "distinct_invalid_question_templates": len(template_counts),
        "maximum_single_template_fraction": maximum_share,
        "template_counts": dict(sorted(template_counts.items())),
    }
    checks = {
        "minimum_invalid_control_cases": observed["invalid_control_cases"]
        >= _OPPORTUNITY_GATES["minimum_invalid_control_cases"],
        "minimum_invalid_control_sources": observed["invalid_control_sources"]
        >= _OPPORTUNITY_GATES["minimum_invalid_control_sources"],
        "minimum_invalid_control_gold_methods": observed["invalid_control_gold_methods"]
        >= _OPPORTUNITY_GATES["minimum_invalid_control_gold_methods"],
        "minimum_distinct_invalid_question_templates": observed[
            "distinct_invalid_question_templates"
        ]
        >= _OPPORTUNITY_GATES["minimum_distinct_invalid_question_templates"],
        "maximum_single_template_fraction": maximum_share
        <= _OPPORTUNITY_GATES["maximum_single_template_fraction"],
    }
    return {"passed": all(checks.values()), "checks": checks, "observed": observed}


def prepare_invalid_control_catalog_fallback_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_invalid_control_catalog_fallback_contract(config)
    _, h21_data, h21_report = _load_h21(config)
    cases_path = Path(str(h21_data["cases_path"]))
    if sha256_file(cases_path) != h21_data["cases_sha256"]:
        raise RuntimeError("H22 H21 cases changed")
    cases = list(read_jsonl(cases_path))
    controls = list(h21_report["private_details"]["control"])
    control_by_id = {str(row["case_id"]): row for row in controls}
    if len(control_by_id) != len(cases):
        raise RuntimeError("H22 H21 control coverage changed")
    invalid_cases = [
        case for case in cases if not bool(control_by_id[str(case["case_id"])]["valid_output"])
    ]
    opportunity = _opportunity_summary(invalid_cases)
    if not opportunity["passed"]:
        raise RuntimeError("H22 frozen invalid-control opportunity no longer passes")

    output_path = _data_root(config) / "invalid-cases.jsonl"
    _write_immutable_jsonl(output_path, invalid_cases, label="invalid-case data")
    data: dict[str, Any] = {
        "schema_version": 1,
        "method": _METHOD,
        "complete": True,
        "contract_fingerprint": contract["fingerprint"],
        "h21_data_fingerprint": h21_data["data_fingerprint"],
        "h21_result_fingerprint": h21_report["result_fingerprint"],
        "h21_control_ledger_sha256": _H21_CONTROL_LEDGER_SHA256,
        "invalid_case_count": len(invalid_cases),
        "invalid_cases_path": str(output_path),
        "invalid_cases_sha256": sha256_file(output_path),
        "invalid_case_fingerprint": canonical_hash(invalid_cases),
        "opportunity_gate": opportunity,
        "evaluation_authorized": True,
        "model_output_opened": False,
        "training_authorized": False,
        "fresh_external_evidence": False,
        "champion_changed": False,
        "release_authorized": False,
    }
    data["data_fingerprint"] = canonical_hash(data)
    root = _root(config)
    _write_immutable_json(root / "data.json", data, label="data receipt")
    write_json(_reports_root(config) / "invalid-control-catalog-fallback-v1-data.json", data)
    return data


def _fallback_phase_header(
    cases: list[dict[str, Any]],
    *,
    evaluation_fingerprint: str,
    runtime_receipt: dict[str, Any],
) -> dict[str, Any]:
    receipts = []
    seen: set[str] = set()
    for case in cases:
        case_id = str(case["case_id"])
        if case_id in seen:
            raise RuntimeError(f"H22 case IDs are not unique: {case_id}")
        seen.add(case_id)
        case_hash = _case_fingerprint(case)
        messages_hash = canonical_hash(_fallback_messages(case))
        request_hash = canonical_hash(
            {
                "evaluation": evaluation_fingerprint,
                "phase": "catalog-grounded-parent-fallback",
                "case_id": case_id,
                "case_fingerprint": case_hash,
                "messages_sha256": messages_hash,
                "decoding": _DECODING,
                "runtime_receipt": runtime_receipt,
                "evaluator_version": _EVALUATOR_VERSION,
            }
        )
        receipts.append(
            {
                "case_id": case_id,
                "case_fingerprint": case_hash,
                "messages_sha256": messages_hash,
                "request_hash": request_hash,
            }
        )
    header: dict[str, Any] = {
        "schema_version": 1,
        "phase": "catalog-grounded-parent-fallback",
        "evaluation_fingerprint": evaluation_fingerprint,
        "runtime_receipt": runtime_receipt,
        "decoding": dict(_DECODING),
        "expected_count": len(receipts),
        "expected_receipts": receipts,
        "evaluator_version": _EVALUATOR_VERSION,
    }
    header["fingerprint"] = canonical_hash(header)
    return header


def _fallback_detail(case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    valid = bool(row["valid_output"])
    predicted = row["predicted_method_id"]
    return {
        "case_id": str(case["case_id"]),
        "source_id": str(case["source_id"]),
        "question": str(case["question"]),
        "gold_method_id": str(case["gold_method_id"]),
        "predicted_method_id": predicted,
        "valid_output": valid,
        "correct": valid and predicted == str(case["gold_method_id"]),
        "parse_reason": str(row["parse_reason"]),
        "raw_answer_sha256": str(row["raw_answer_sha256"]),
        "request_hash": str(row["request_hash"]),
        "messages_sha256": str(row["messages_sha256"]),
    }


def _evaluate_fallback(
    cases: list[dict[str, Any]],
    *,
    evaluation_fingerprint: str,
    runtime_receipt: dict[str, Any],
    caller: ModelCaller,
    progress_path: Path,
) -> list[dict[str, Any]]:
    header = _fallback_phase_header(
        cases,
        evaluation_fingerprint=evaluation_fingerprint,
        runtime_receipt=runtime_receipt,
    )
    rows = _load_strict_progress(progress_path, header=header)
    receipts = list(header["expected_receipts"])
    for index in range(len(rows), len(cases)):
        case = cases[index]
        _start_strict_progress(progress_path, header=header)
        answer = caller(str(case["case_id"]), _fallback_messages(case), dict(_DECODING))
        if not isinstance(answer, str):
            raise RuntimeError("H22 model caller returned a non-string answer")
        parsed = _parse_prediction(answer)
        row = {
            **receipts[index],
            "phase": header["phase"],
            "raw_answer": answer,
            "raw_answer_sha256": sha256_text(answer),
            **parsed,
        }
        rows = _complete_strict_progress(progress_path, header=header, row=row)
    if len(rows) != len(cases):
        raise RuntimeError("H22 fallback phase did not complete")
    return [_fallback_detail(case, row) for case, row in zip(cases, rows, strict=True)]


def _apply_catalog_guard(
    control: dict[str, Any], fallback: dict[str, Any] | None
) -> dict[str, Any]:
    if bool(control["valid_output"]):
        if fallback is not None:
            raise RuntimeError("H22 valid control received a forbidden fallback call")
        selected = control
        route = "valid-control-identity"
    else:
        if fallback is None:
            raise RuntimeError("H22 invalid control is missing its fallback call")
        if str(control["case_id"]) != str(fallback["case_id"]):
            raise RuntimeError("H22 control and fallback case IDs changed")
        if control["messages_sha256"] == fallback["messages_sha256"]:
            raise RuntimeError("H22 fallback did not add the fixed catalog prompt")
        if bool(fallback["valid_output"]):
            selected = fallback
            route = "invalid-control-valid-catalog-fallback"
        else:
            selected = control
            route = "invalid-control-invalid-catalog-fallback"
    return {
        "case_id": str(control["case_id"]),
        "source_id": str(control["source_id"]),
        "question": str(control["question"]),
        "gold_method_id": str(control["gold_method_id"]),
        "predicted_method_id": selected["predicted_method_id"],
        "valid_output": bool(selected["valid_output"]),
        "correct": bool(selected["correct"]),
        "route": route,
        "control_predicted_method_id": control["predicted_method_id"],
        "control_valid_output": bool(control["valid_output"]),
        "control_correct": bool(control["correct"]),
        "control_parse_reason": control["parse_reason"],
        "control_raw_answer_sha256": control["raw_answer_sha256"],
        "fallback_called": fallback is not None,
        "fallback_predicted_method_id": None
        if fallback is None
        else fallback["predicted_method_id"],
        "fallback_valid_output": False if fallback is None else bool(fallback["valid_output"]),
        "fallback_correct": False if fallback is None else bool(fallback["correct"]),
        "fallback_parse_reason": None if fallback is None else fallback["parse_reason"],
        "fallback_raw_answer_sha256": None if fallback is None else fallback["raw_answer_sha256"],
    }


def _template_summary(candidate: list[dict[str, Any]]) -> dict[str, Any]:
    fallback_rows = [row for row in candidate if bool(row["fallback_called"])]
    by_template: dict[str, list[dict[str, Any]]] = {}
    for row in fallback_rows:
        by_template.setdefault(_question_template(str(row["question"])), []).append(row)
    template_details = []
    for template, rows in sorted(by_template.items()):
        all_gained = all(not bool(row["control_correct"]) and bool(row["correct"]) for row in rows)
        template_details.append(
            {
                "template": template,
                "case_count": len(rows),
                "source_count": len({str(row["source_id"]) for row in rows}),
                "gold_methods": sorted({str(row["gold_method_id"]) for row in rows}),
                "all_cases_gained": all_gained,
                "all_fallbacks_valid": all(bool(row["fallback_valid_output"]) for row in rows),
            }
        )
    gained = [
        row for row in fallback_rows if not bool(row["control_correct"]) and bool(row["correct"])
    ]
    return {
        "invalid_case_count": len(fallback_rows),
        "distinct_invalid_question_templates": len(template_details),
        "distinct_repaired_question_templates": sum(
            bool(item["all_cases_gained"]) for item in template_details
        ),
        "repaired_gold_methods": sorted({str(row["gold_method_id"]) for row in gained}),
        "repair_sources": sorted({str(row["source_id"]) for row in gained}),
        "valid_fallback_case_count": sum(
            bool(row["fallback_valid_output"]) for row in fallback_rows
        ),
        "template_details": template_details,
    }


def _result_gate(
    *,
    paired: dict[str, Any],
    candidate: list[dict[str, Any]],
    template_summary: dict[str, Any],
) -> dict[str, Any]:
    identity_rows = [row for row in candidate if row["route"] == "valid-control-identity"]
    checks = {
        "valid_control_identity": len(identity_rows) == 20
        and all(
            row["predicted_method_id"] == row["control_predicted_method_id"]
            and row["correct"] == row["control_correct"]
            for row in identity_rows
        ),
        "zero_control_only_losses": int(paired["control_only"]) == 0,
        "minimum_valid_fallback_cases": int(template_summary["valid_fallback_case_count"])
        >= _RESULT_GATES["minimum_valid_fallback_cases"],
        "minimum_candidate_only_gains": int(paired["candidate_only"])
        >= _RESULT_GATES["minimum_candidate_only_gains"],
        "minimum_net_improvements": int(paired["net_improvements"])
        >= _RESULT_GATES["minimum_net_improvements"],
        "minimum_distinct_repaired_question_templates": int(
            template_summary["distinct_repaired_question_templates"]
        )
        >= _RESULT_GATES["minimum_distinct_repaired_question_templates"],
        "minimum_repaired_gold_methods": len(template_summary["repaired_gold_methods"])
        >= _RESULT_GATES["minimum_repaired_gold_methods"],
        "minimum_repair_source_count": len(template_summary["repair_sources"])
        >= _RESULT_GATES["minimum_repair_source_count"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "valid_control_identity_cases": len(identity_rows),
            "valid_fallback_cases": template_summary["valid_fallback_case_count"],
            "candidate_only_gains": paired["candidate_only"],
            "control_only_losses": paired["control_only"],
            "net_improvements": paired["net_improvements"],
            "distinct_repaired_question_templates": template_summary[
                "distinct_repaired_question_templates"
            ],
            "repaired_gold_method_count": len(template_summary["repaired_gold_methods"]),
            "repair_source_count": len(template_summary["repair_sources"]),
        },
    }


def _verify_terminal_report(
    report: dict[str, Any],
    *,
    contract_fingerprint: str,
    data_fingerprint: str,
    evaluation_fingerprint: str,
) -> None:
    private = report.get("private_details")
    if not isinstance(private, dict) or report.get("private_details_fingerprint") != canonical_hash(
        private
    ):
        raise RuntimeError("H22 terminal private details changed")
    unsigned = {
        key: value
        for key, value in report.items()
        if key not in {"result_fingerprint", "private_details"}
    }
    if report.get("result_fingerprint") != canonical_hash(unsigned):
        raise RuntimeError("H22 terminal result fingerprint changed")
    if report.get("contract_fingerprint") != contract_fingerprint:
        raise RuntimeError("H22 terminal contract changed")
    if report.get("data_fingerprint") != data_fingerprint:
        raise RuntimeError("H22 terminal data changed")
    if report.get("evaluation_fingerprint") != evaluation_fingerprint:
        raise RuntimeError("H22 terminal evaluation changed")
    if report.get("status") not in {"SUPPORTED_DEV_COMPOSITION", "SCIENTIFIC_FAIL"}:
        raise RuntimeError("H22 terminal status changed")
    if (
        report.get("training_authorized")
        or report.get("champion_changed")
        or report.get("release_authorized")
    ):
        raise RuntimeError("H22 terminal governance boundary changed")
    counts = report.get("model_call_counts", {})
    if counts != {
        "reused_h21_controls": 24,
        "new_menu_free_control_calls": 0,
        "new_catalog_fallback_calls": 4,
        "valid_control_fallback_calls": 0,
    }:
        raise RuntimeError("H22 terminal model-call accounting changed")
    controls = private.get("control")
    fallback = private.get("fallback")
    candidate = private.get("candidate")
    if not isinstance(controls, list) or len(controls) != 24:
        raise RuntimeError("H22 terminal controls changed")
    if not isinstance(fallback, list) or len(fallback) != 4:
        raise RuntimeError("H22 terminal fallbacks changed")
    if not isinstance(candidate, list) or len(candidate) != 24:
        raise RuntimeError("H22 terminal candidate changed")
    expected_scores = {
        "control": _strict_metrics(controls),
        "catalog_guard": _strict_metrics(candidate),
    }
    expected_paired = _paired(controls, candidate)
    expected_source_paired = _source_paired(controls, candidate)
    expected_templates = _template_summary(candidate)
    expected_gate = _result_gate(
        paired=expected_paired,
        candidate=candidate,
        template_summary=expected_templates,
    )
    if report.get("scores") != expected_scores:
        raise RuntimeError("H22 terminal scores changed")
    if (
        report.get("paired") != expected_paired
        or report.get("source_paired") != expected_source_paired
    ):
        raise RuntimeError("H22 terminal paired accounting changed")
    if report.get("template_summary") != expected_templates:
        raise RuntimeError("H22 terminal template accounting changed")
    if report.get("result_gate") != expected_gate:
        raise RuntimeError("H22 terminal result gate changed")
    expected_status = "SUPPORTED_DEV_COMPOSITION" if expected_gate["passed"] else "SCIENTIFIC_FAIL"
    if report.get("status") != expected_status:
        raise RuntimeError("H22 terminal status does not match its gate")


def _public_report(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "private_details"}


def run_invalid_control_catalog_fallback(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_invalid_control_catalog_fallback_contract(config)
    data = prepare_invalid_control_catalog_fallback_data(config)
    root = _root(config)
    report_path = root / "report.json"
    public_path = _reports_root(config) / "invalid-control-catalog-fallback-v1.json"
    invalid_cases = list(read_jsonl(Path(str(data["invalid_cases_path"]))))
    runtime_receipt = {
        "fallback_adapter": dict(contract["runtime"]["fallback_adapter"]),
        "fallback_prompt_sha256": contract["runtime"]["fallback_prompt_sha256"],
        "decoding": dict(contract["runtime"]["decoding"]),
        "implementation_sha256": dict(contract["implementation_sha256"]),
    }
    runtime_receipt["fingerprint"] = canonical_hash(runtime_receipt)
    evaluation_fingerprint = canonical_hash(
        {
            "method": _METHOD,
            "evaluator_version": _EVALUATOR_VERSION,
            "contract_fingerprint": contract["fingerprint"],
            "data_fingerprint": data["data_fingerprint"],
            "runtime_receipt_fingerprint": runtime_receipt["fingerprint"],
            "invalid_case_fingerprint": data["invalid_case_fingerprint"],
        }
    )
    opening: dict[str, Any] = {
        "schema_version": 1,
        "method": _METHOD,
        "evaluation_fingerprint": evaluation_fingerprint,
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["data_fingerprint"],
        "invalid_case_count": len(invalid_cases),
        "model_output_opened": True,
        "fresh_external_evidence": False,
        "training_authorized": False,
        "champion_changed": False,
        "release_authorized": False,
    }
    opening["fingerprint"] = canonical_hash(opening)
    progress_path = root / "progress" / "fallback-ledger.json"
    header = _fallback_phase_header(
        invalid_cases,
        evaluation_fingerprint=evaluation_fingerprint,
        runtime_receipt=runtime_receipt,
    )
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _verify_terminal_report(
            report,
            contract_fingerprint=contract["fingerprint"],
            data_fingerprint=data["data_fingerprint"],
            evaluation_fingerprint=evaluation_fingerprint,
        )
        opening_path = root / "evaluation-opened.json"
        if (
            not opening_path.exists()
            or json.loads(opening_path.read_text(encoding="utf-8")) != opening
        ):
            raise RuntimeError("H22 terminal opening receipt changed")
        completed_rows = _load_strict_progress(progress_path, header=header)
        if len(completed_rows) != len(invalid_cases):
            raise RuntimeError("H22 terminal fallback ledger is incomplete")
        ledger_hashes = [str(row["raw_answer_sha256"]) for row in completed_rows]
        report_hashes = [
            str(row["raw_answer_sha256"]) for row in report["private_details"]["fallback"]
        ]
        if ledger_hashes != report_hashes:
            raise RuntimeError("H22 terminal fallback ledger changed")
        public = _public_report(report)
        write_json(public_path, public)
        return public

    _write_immutable_json(root / "evaluation-opened.json", opening, label="opening receipt")

    completed = _load_strict_progress(progress_path, header=header)
    remaining = len(invalid_cases) - len(completed)
    if remaining < 0:
        raise RuntimeError("H22 progress contains extra calls")
    if remaining:
        caller: ModelCaller = _AgentCaller(
            config,
            adapter_path=str(runtime_receipt["fallback_adapter"]["adapter_path"]),
            expected_calls=remaining,
        )
    else:

        def caller(*_args: Any, **_kwargs: Any) -> str:
            raise RuntimeError("H22 terminal ledger attempted another model call")

    fallback = _evaluate_fallback(
        invalid_cases,
        evaluation_fingerprint=evaluation_fingerprint,
        runtime_receipt=runtime_receipt,
        caller=caller,
        progress_path=progress_path,
    )
    _, _, h21_report = _load_h21(config)
    controls = list(h21_report["private_details"]["control"])
    fallback_by_id = {str(row["case_id"]): row for row in fallback}
    candidate = [
        _apply_catalog_guard(control, fallback_by_id.get(str(control["case_id"])))
        for control in controls
    ]
    control_metrics = _strict_metrics(controls)
    candidate_metrics = _strict_metrics(candidate)
    paired = _paired(controls, candidate)
    source_paired = _source_paired(controls, candidate)
    template_summary = _template_summary(candidate)
    result_gate = _result_gate(
        paired=paired,
        candidate=candidate,
        template_summary=template_summary,
    )
    status = "SUPPORTED_DEV_COMPOSITION" if result_gate["passed"] else "SCIENTIFIC_FAIL"
    private_details = {
        "control": controls,
        "fallback": fallback,
        "candidate": candidate,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": _METHOD,
        "status": status,
        "evidence_status": "opened-source-development-composition",
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["data_fingerprint"],
        "evaluation_fingerprint": evaluation_fingerprint,
        "opening_fingerprint": opening["fingerprint"],
        "runtime_receipt_fingerprint": runtime_receipt["fingerprint"],
        "model_call_counts": {
            "reused_h21_controls": len(controls),
            "new_menu_free_control_calls": 0,
            "new_catalog_fallback_calls": len(fallback),
            "valid_control_fallback_calls": 0,
        },
        "opportunity_gate": data["opportunity_gate"],
        "scores": {"control": control_metrics, "catalog_guard": candidate_metrics},
        "paired": paired,
        "source_paired": source_paired,
        "template_summary": template_summary,
        "result_gate": result_gate,
        "model_output_opened": True,
        "fresh_external_evidence": False,
        "training_authorized": False,
        "champion_changed": False,
        "release_authorized": False,
        "next_step": (
            "seek-preregistered-fresh-evidence-for-this-composition"
            if result_gate["passed"]
            else "close-catalog-fallback-composition"
        ),
        "claim_boundary": contract["claim_boundary"],
        "private_details_fingerprint": canonical_hash(private_details),
        "private_details": private_details,
    }
    unsigned = {
        key: value
        for key, value in report.items()
        if key not in {"result_fingerprint", "private_details"}
    }
    report["result_fingerprint"] = canonical_hash(unsigned)
    _verify_terminal_report(
        report,
        contract_fingerprint=contract["fingerprint"],
        data_fingerprint=data["data_fingerprint"],
        evaluation_fingerprint=evaluation_fingerprint,
    )
    _write_immutable_json(report_path, report, label="terminal report")
    public = _public_report(report)
    write_json(public_path, public)
    return public

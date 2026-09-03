from __future__ import annotations

import gc
import json
import math
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, sha256_text, write_json
from .stats_agent import StatsAgent
from .stats_catalog_grounding import _messages as _h14_messages
from .stats_external_catalog import _canonicalize_method_label, _mcnemar_exact_pvalue
from .stats_representation_probe import _METHOD_IDS

_EVALUATOR_VERSION = 1
_DECODING: dict[str, Any] = {
    "max_tokens": 160,
    "temperature": 0.0,
    "top_p": 1.0,
}
_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", flags=re.DOTALL | re.IGNORECASE)

ModelCaller = Callable[[str, list[dict[str, str]], dict[str, Any]], str]


class _AgentCaller:
    """Lazily load one adapter and release it after its known final invocation."""

    def __init__(
        self,
        config: ProjectConfig,
        *,
        adapter_path: str,
        expected_calls: int | None,
    ) -> None:
        self.config = config
        self.adapter_path = adapter_path
        self.expected_calls = expected_calls
        self.calls = 0
        self.agent: StatsAgent | None = None

    def __call__(
        self,
        case_id: str,
        messages: list[dict[str, str]],
        decoding: dict[str, Any],
    ) -> str:
        del case_id
        if self.expected_calls is not None and self.calls >= self.expected_calls:
            raise RuntimeError("E5 model caller exceeded its frozen invocation count")
        if self.agent is None:
            self.agent = StatsAgent(self.config, adapter_path=self.adapter_path)
        try:
            answer = self.agent.answer_without_tools(
                messages,
                route="stats",
                max_tokens=int(decoding["max_tokens"]),
                temperature=float(decoding["temperature"]),
                top_p=float(decoding["top_p"]),
            )
        except Exception:
            self.close()
            raise
        self.calls += 1
        if self.expected_calls is not None and self.calls == self.expected_calls:
            self.close()
        return answer

    def close(self) -> None:
        if self.agent is not None:
            del self.agent
            self.agent = None
            gc.collect()
            mx.clear_cache()


def _parse_prediction(answer: str) -> dict[str, Any]:
    """Parse the frozen H14 methods-array response and explain invalidity."""

    cleaned = _THINK_RE.sub("", answer).strip()
    if not cleaned:
        return {
            "predicted_method_id": None,
            "valid_output": False,
            "parse_reason": "empty-output",
        }

    start = cleaned.find("{")
    if start < 0:
        return {
            "predicted_method_id": None,
            "valid_output": False,
            "parse_reason": "no-json-object",
        }
    try:
        parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError:
        return {
            "predicted_method_id": None,
            "valid_output": False,
            "parse_reason": "malformed-json",
        }
    if not isinstance(parsed, dict):
        return {
            "predicted_method_id": None,
            "valid_output": False,
            "parse_reason": "json-not-object",
        }

    raw = parsed.get("methods")
    if not isinstance(raw, list):
        return {
            "predicted_method_id": None,
            "valid_output": False,
            "parse_reason": "methods-not-array",
        }
    if len(raw) == 0:
        return {
            "predicted_method_id": None,
            "valid_output": False,
            "parse_reason": "empty-method-array",
        }
    if len(raw) != 1:
        return {
            "predicted_method_id": None,
            "valid_output": False,
            "parse_reason": "multiple-methods",
        }
    if not isinstance(raw[0], str) or not raw[0].strip():
        return {
            "predicted_method_id": None,
            "valid_output": False,
            "parse_reason": "invalid-method-value",
        }

    method_id = raw[0] if raw[0] in _METHOD_IDS else _canonicalize_method_label(raw[0])
    if method_id is None or method_id not in _METHOD_IDS:
        return {
            "predicted_method_id": None,
            "valid_output": False,
            "parse_reason": "unknown-method-id",
        }
    return {
        "predicted_method_id": method_id,
        "valid_output": True,
        "parse_reason": "valid-canonical-method",
    }


def _case_fingerprint(case: dict[str, Any]) -> str:
    required = ("case_id", "source_id", "question", "gold_method_id")
    missing = [field for field in required if field not in case]
    if missing:
        raise RuntimeError(f"E5 case is missing required fields: {missing}")
    extras = sorted(set(case) - set(required))
    if extras:
        raise RuntimeError(f"E5 case contains forbidden model-adjacent fields: {extras}")
    if str(case["gold_method_id"]) not in _METHOD_IDS:
        raise RuntimeError(f"E5 case gold is outside the frozen catalog: {case['case_id']}")
    return canonical_hash({field: case[field] for field in required})


def _messages(case: dict[str, Any]) -> list[dict[str, str]]:
    return _h14_messages({"question": str(case["question"])}, grounded=False)


def _phase_header(
    cases: list[dict[str, Any]],
    *,
    phase: str,
    evaluation_fingerprint: str,
    runtime_receipt: dict[str, Any],
) -> dict[str, Any]:
    if phase not in {"control", "repair"}:
        raise ValueError("E5 phase must be control or repair")
    receipts: list[dict[str, str]] = []
    seen: set[str] = set()
    for case in cases:
        case_id = str(case["case_id"])
        if case_id in seen:
            raise RuntimeError(f"E5 case IDs are not unique: {case_id}")
        seen.add(case_id)
        case_hash = _case_fingerprint(case)
        messages_hash = canonical_hash(_messages(case))
        request_hash = canonical_hash(
            {
                "evaluation": evaluation_fingerprint,
                "phase": phase,
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
        "phase": phase,
        "evaluation_fingerprint": evaluation_fingerprint,
        "runtime_receipt": runtime_receipt,
        "decoding": dict(_DECODING),
        "expected_count": len(receipts),
        "expected_receipts": receipts,
        "evaluator_version": _EVALUATOR_VERSION,
    }
    header["fingerprint"] = canonical_hash(header)
    return header


def _ledger_payload(
    header: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    inflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "header": header,
        "completed": len(rows),
        "complete": inflight is None and len(rows) == int(header["expected_count"]),
        "inflight": inflight,
        "rows": rows,
    }
    payload["ledger_fingerprint"] = canonical_hash(payload)
    return payload


def _validate_ledger(
    payload: dict[str, Any],
    *,
    expected_header: dict[str, Any],
    allow_inflight: bool = False,
) -> list[dict[str, Any]]:
    observed_fingerprint = payload.get("ledger_fingerprint")
    unsigned = {key: value for key, value in payload.items() if key != "ledger_fingerprint"}
    if observed_fingerprint != canonical_hash(unsigned):
        raise RuntimeError("E5 progress ledger fingerprint changed")
    header = payload.get("header")
    if not isinstance(header, dict) or header.get("fingerprint") != expected_header.get(
        "fingerprint"
    ):
        raise RuntimeError("E5 progress runtime, cases, or request contract changed")
    if header != expected_header:
        raise RuntimeError("E5 progress header changed")
    rows = payload.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("E5 progress rows are malformed")
    if payload.get("completed") != len(rows):
        raise RuntimeError("E5 progress completed count changed")
    inflight = payload.get("inflight")
    expected_complete = inflight is None and len(rows) == int(expected_header["expected_count"])
    if payload.get("complete") is not expected_complete:
        raise RuntimeError("E5 progress completion state changed")
    expected_receipts = list(expected_header["expected_receipts"])
    if len(rows) > len(expected_receipts):
        raise RuntimeError("E5 progress contains extra rows")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        receipt = expected_receipts[index]
        case_id = str(row.get("case_id", ""))
        if case_id in seen:
            raise RuntimeError(f"E5 progress contains a duplicate case ID: {case_id}")
        seen.add(case_id)
        for key in ("case_id", "case_fingerprint", "messages_sha256", "request_hash"):
            if row.get(key) != receipt.get(key):
                raise RuntimeError(f"E5 progress {key} changed at row {index}")
        if row.get("phase") != expected_header["phase"]:
            raise RuntimeError(f"E5 progress phase changed at row {index}")
        answer = row.get("raw_answer")
        if not isinstance(answer, str) or row.get("raw_answer_sha256") != sha256_text(answer):
            raise RuntimeError(f"E5 progress answer receipt changed at row {index}")
        parsed = _parse_prediction(answer)
        for key in ("predicted_method_id", "valid_output", "parse_reason"):
            if row.get(key) != parsed[key]:
                raise RuntimeError(f"E5 progress parsed output changed at row {index}")
    if inflight is not None:
        if not isinstance(inflight, dict):
            raise RuntimeError("E5 progress in-flight receipt is malformed")
        if len(rows) >= len(expected_receipts):
            raise RuntimeError("E5 progress has an in-flight call after completion")
        expected = expected_receipts[len(rows)]
        for key in ("case_id", "case_fingerprint", "messages_sha256", "request_hash"):
            if inflight.get(key) != expected.get(key):
                raise RuntimeError(f"E5 progress in-flight {key} changed")
        if inflight.get("phase") != expected_header["phase"]:
            raise RuntimeError("E5 progress in-flight phase changed")
        if inflight.get("state") != "started":
            raise RuntimeError("E5 progress in-flight state changed")
        unsigned_inflight = {
            key: value for key, value in inflight.items() if key != "fingerprint"
        }
        if inflight.get("fingerprint") != canonical_hash(unsigned_inflight):
            raise RuntimeError("E5 progress in-flight fingerprint changed")
        if not allow_inflight:
            raise RuntimeError(
                "E5 progress contains an interrupted model call; protocol is invalid and the "
                "request must not be retried"
            )
    return [dict(row) for row in rows]


def _read_ledger(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError("E5 progress ledger is unreadable") from error
    if not isinstance(payload, dict):
        raise RuntimeError("E5 progress ledger is not an object")
    return payload


def _load_strict_progress(path: Path, *, header: dict[str, Any]) -> list[dict[str, Any]]:
    if not path.exists():
        write_json(path, _ledger_payload(header, []))
        return []
    return _validate_ledger(_read_ledger(path), expected_header=header)


def _start_strict_progress(
    path: Path,
    *,
    header: dict[str, Any],
) -> None:
    rows = _load_strict_progress(path, header=header)
    expected_receipts = list(header["expected_receipts"])
    if len(rows) >= len(expected_receipts):
        raise RuntimeError("E5 progress is already complete")
    expected = expected_receipts[len(rows)]
    inflight: dict[str, Any] = {
        **expected,
        "phase": header["phase"],
        "state": "started",
    }
    inflight["fingerprint"] = canonical_hash(inflight)
    write_json(path, _ledger_payload(header, rows, inflight=inflight))


def _complete_strict_progress(
    path: Path,
    *,
    header: dict[str, Any],
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    payload = _read_ledger(path)
    rows = _validate_ledger(payload, expected_header=header, allow_inflight=True)
    inflight = payload.get("inflight")
    if not isinstance(inflight, dict):
        raise RuntimeError("E5 progress has no started model call to complete")
    expected_receipts = list(header["expected_receipts"])
    expected = expected_receipts[len(rows)]
    for key in ("case_id", "case_fingerprint", "messages_sha256", "request_hash"):
        if row.get(key) != expected.get(key) or row.get(key) != inflight.get(key):
            raise RuntimeError(f"E5 attempted to append a mismatched {key}")
    if row.get("phase") != header["phase"]:
        raise RuntimeError("E5 attempted to append a mismatched phase")
    answer = row.get("raw_answer")
    if not isinstance(answer, str) or row.get("raw_answer_sha256") != sha256_text(answer):
        raise RuntimeError("E5 attempted to append an invalid answer receipt")
    parsed = _parse_prediction(answer)
    for key in ("predicted_method_id", "valid_output", "parse_reason"):
        if row.get(key) != parsed[key]:
            raise RuntimeError(f"E5 attempted to append a mismatched {key}")
    rows.append(dict(row))
    write_json(path, _ledger_payload(header, rows, inflight=None))
    return rows


def _phase_detail(case: dict[str, Any], progress: dict[str, Any]) -> dict[str, Any]:
    predicted = progress["predicted_method_id"]
    valid = bool(progress["valid_output"])
    correct = valid and predicted == str(case["gold_method_id"])
    return {
        "case_id": str(case["case_id"]),
        "source_id": str(case["source_id"]),
        "question": str(case["question"]),
        "gold_method_id": str(case["gold_method_id"]),
        "predicted_method_id": predicted,
        "valid_output": valid,
        "correct": correct,
        "parse_reason": str(progress["parse_reason"]),
        "raw_answer_sha256": str(progress["raw_answer_sha256"]),
        "request_hash": str(progress["request_hash"]),
        "messages_sha256": str(progress["messages_sha256"]),
    }


def _evaluate_phase(
    cases: list[dict[str, Any]],
    *,
    phase: str,
    evaluation_fingerprint: str,
    runtime_receipt: dict[str, Any],
    caller: ModelCaller,
    progress_path: Path | None,
) -> list[dict[str, Any]]:
    header = _phase_header(
        cases,
        phase=phase,
        evaluation_fingerprint=evaluation_fingerprint,
        runtime_receipt=runtime_receipt,
    )
    rows = (
        _load_strict_progress(progress_path, header=header) if progress_path is not None else []
    )
    receipts = list(header["expected_receipts"])
    for index in range(len(rows), len(cases)):
        case = cases[index]
        if progress_path is not None:
            _start_strict_progress(progress_path, header=header)
        answer = caller(str(case["case_id"]), _messages(case), dict(_DECODING))
        if not isinstance(answer, str):
            raise RuntimeError("E5 model caller returned a non-string answer")
        parsed = _parse_prediction(answer)
        receipt = receipts[index]
        row = {
            **receipt,
            "phase": phase,
            "raw_answer": answer,
            "raw_answer_sha256": sha256_text(answer),
            **parsed,
        }
        rows = (
            _complete_strict_progress(progress_path, header=header, row=row)
            if progress_path is not None
            else [*rows, row]
        )
    if len(rows) != len(cases):
        raise RuntimeError("E5 phase did not complete its frozen case set")
    return [_phase_detail(case, row) for case, row in zip(cases, rows, strict=True)]


def _apply_guard(
    control: dict[str, Any], repair: dict[str, Any] | None
) -> dict[str, Any]:
    if bool(control["correct"]) and not bool(control["valid_output"]):
        raise RuntimeError("E5 correct control cannot be parser-invalid")
    if bool(control["valid_output"]):
        if repair is not None:
            raise RuntimeError("E5 valid control received a forbidden repair call")
        route = "valid-control-identity"
        selected = control
    else:
        if repair is None:
            raise RuntimeError("E5 invalid control is missing its required repair call")
        if str(control["case_id"]) != str(repair["case_id"]):
            raise RuntimeError("E5 control and repair case IDs changed")
        if control["request_hash"] == repair["request_hash"]:
            raise RuntimeError("E5 control and repair runtime receipts are indistinguishable")
        if control["messages_sha256"] != repair["messages_sha256"]:
            raise RuntimeError("E5 control and repair prompts changed")
        if bool(repair["valid_output"]):
            route = "invalid-control-valid-repair"
            selected = repair
        else:
            route = "invalid-control-invalid-repair-fallback"
            selected = control
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
        "repair_called": repair is not None,
        "repair_predicted_method_id": None if repair is None else repair["predicted_method_id"],
        "repair_valid_output": False if repair is None else bool(repair["valid_output"]),
        "repair_correct": False if repair is None else bool(repair["correct"]),
        "repair_parse_reason": None if repair is None else repair["parse_reason"],
        "repair_raw_answer_sha256": None if repair is None else repair["raw_answer_sha256"],
    }


def _strict_metrics(details: list[dict[str, Any]]) -> dict[str, Any]:
    if not details:
        raise RuntimeError("E5 metrics require at least one case")
    valid_count = sum(bool(row["valid_output"]) for row in details)
    correct_count = sum(bool(row["correct"]) for row in details)
    return {
        "count": len(details),
        "correct_count": correct_count,
        "accuracy": correct_count / len(details),
        "valid_count": valid_count,
        "valid_output_rate": valid_count / len(details),
        "in_catalog_prediction_count": valid_count,
        "in_catalog_prediction_rate": valid_count / len(details),
    }


def _paired(control: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    control_by_id = {str(row["case_id"]): row for row in control}
    candidate_by_id = {str(row["case_id"]): row for row in candidate}
    if len(control_by_id) != len(control) or len(candidate_by_id) != len(candidate):
        raise RuntimeError("E5 paired details contain duplicate IDs")
    if set(control_by_id) != set(candidate_by_id):
        raise RuntimeError("E5 paired case coverage changed")
    counts: Counter[str] = Counter()
    for case_id in sorted(control_by_id):
        control_correct = bool(control_by_id[case_id]["correct"])
        candidate_correct = bool(candidate_by_id[case_id]["correct"])
        if control_correct and candidate_correct:
            counts["both_correct"] += 1
        elif candidate_correct:
            counts["candidate_only"] += 1
        elif control_correct:
            counts["control_only"] += 1
        else:
            counts["both_wrong"] += 1
    result = {
        key: int(counts[key])
        for key in ("both_correct", "candidate_only", "control_only", "both_wrong")
    }
    result["count"] = len(control_by_id)
    result["net_improvements"] = result["candidate_only"] - result["control_only"]
    result["mcnemar_exact_two_sided_p"] = _mcnemar_exact_pvalue(
        result["candidate_only"], result["control_only"]
    )
    return result


def _wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        return [0.0, 1.0]
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _source_paired(
    control: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    sources = sorted({str(row["source_id"]) for row in control})
    result: dict[str, Any] = {}
    for source in sources:
        ids = {str(row["case_id"]) for row in control if str(row["source_id"]) == source}
        result[source] = _paired(
            [row for row in control if str(row["case_id"]) in ids],
            [row for row in candidate if str(row["case_id"]) in ids],
        )
    return result


def _external_gate(
    *,
    control: dict[str, Any],
    candidate: dict[str, Any],
    paired: dict[str, Any],
    details: list[dict[str, Any]],
    gates: dict[str, Any],
) -> dict[str, Any]:
    invalid_controls = [row for row in details if not bool(row["control_valid_output"])]
    valid_controls = [row for row in details if bool(row["control_valid_output"])]
    invalid_gold_methods = {str(row["gold_method_id"]) for row in invalid_controls}
    gain_points = 100.0 * (float(candidate["accuracy"]) - float(control["accuracy"]))
    coverage_checks = {
        "minimum_invalid_control_opportunities": len(invalid_controls)
        >= int(gates["minimum_invalid_control_opportunities"]),
        "minimum_invalid_control_gold_methods": len(invalid_gold_methods)
        >= int(gates["minimum_invalid_control_gold_methods"]),
        "minimum_valid_control_identity_cases": len(valid_controls)
        >= int(gates["minimum_valid_control_identity_cases"]),
    }
    confirmatory_checks = {
        "zero_control_only_losses": int(paired["control_only"]) == 0,
        "minimum_candidate_only_gains": int(paired["candidate_only"])
        >= int(gates["minimum_candidate_only_gains"]),
        "minimum_net_improvements": int(paired["net_improvements"])
        >= int(gates["minimum_net_improvements"]),
        "minimum_accuracy_gain_points": gain_points
        >= float(gates["minimum_accuracy_gain_points"]),
        "maximum_mcnemar_p": float(paired["mcnemar_exact_two_sided_p"])
        <= float(gates["maximum_mcnemar_p"]),
        "validity_noninferior": float(candidate["valid_output_rate"])
        >= float(control["valid_output_rate"]),
    }
    if not all(coverage_checks.values()):
        status = "INCONCLUSIVE_REPAIR_COVERAGE"
    elif all(confirmatory_checks.values()):
        status = "CONFIRMED_NARROW_PASS"
    else:
        status = "SCIENTIFIC_FAIL"
    return {
        "status": status,
        "repair_coverage_sufficient": all(coverage_checks.values()),
        "confirmatory_passed": status == "CONFIRMED_NARROW_PASS",
        "coverage_checks": coverage_checks,
        "confirmatory_checks": confirmatory_checks,
        "observed": {
            "invalid_control_opportunities": len(invalid_controls),
            "invalid_control_gold_method_count": len(invalid_gold_methods),
            "valid_control_identity_cases": len(valid_controls),
            "accuracy_gain_points": gain_points,
        },
    }


def _terminal_fingerprint(report: dict[str, Any]) -> str:
    unsigned = {
        key: value
        for key, value in report.items()
        if key not in {"result_fingerprint", "private_details"}
    }
    return canonical_hash(unsigned)


def _verify_terminal_report(report: dict[str, Any], *, require_private: bool = True) -> None:
    if not report.get("complete"):
        raise RuntimeError("E5 terminal report is incomplete")
    if report.get("result_fingerprint") != _terminal_fingerprint(report):
        raise RuntimeError("E5 terminal result fingerprint changed")
    private = report.get("private_details")
    if require_private:
        if not isinstance(private, dict):
            raise RuntimeError("E5 internal terminal report lost its private details")
        if report.get("private_details_fingerprint") != canonical_hash(private):
            raise RuntimeError("E5 private-details fingerprint changed")
    elif "private_details" in report:
        raise RuntimeError("E5 public terminal report contains private details")


def _public_report(report: dict[str, Any]) -> dict[str, Any]:
    _verify_terminal_report(report)
    public = {key: value for key, value in report.items() if key != "private_details"}
    forbidden = {"case_id", "question", "gold_method_id", "raw_answer"}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            leaked = forbidden & set(value)
            if leaked:
                raise RuntimeError(f"E5 public report contains private fields: {sorted(leaked)}")
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(public)
    return public


def run_exact_two_stage(
    cases: list[dict[str, Any]],
    *,
    evaluation_fingerprint: str,
    control_runtime_receipt: dict[str, Any],
    repair_runtime_receipt: dict[str, Any],
    control_caller: ModelCaller,
    repair_caller: ModelCaller,
    progress_root: Path | None = None,
    gates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the exact H20 runtime with repair calls only for invalid controls."""

    if not cases:
        raise RuntimeError("E5 exact runtime has no cases")
    if canonical_hash(control_runtime_receipt) == canonical_hash(repair_runtime_receipt):
        raise RuntimeError("E5 control and repair runtime receipts must identify distinct adapters")
    case_ids = [str(case["case_id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("E5 exact runtime case IDs are not unique")

    control_path = None if progress_root is None else progress_root / "control-ledger.json"
    repair_path = None if progress_root is None else progress_root / "repair-ledger.json"
    control = _evaluate_phase(
        cases,
        phase="control",
        evaluation_fingerprint=evaluation_fingerprint,
        runtime_receipt=control_runtime_receipt,
        caller=control_caller,
        progress_path=control_path,
    )
    control_by_id = {str(row["case_id"]): row for row in control}
    invalid_cases = [
        case
        for case in cases
        if not control_by_id[str(case["case_id"])]["valid_output"]
    ]
    repair = _evaluate_phase(
        invalid_cases,
        phase="repair",
        evaluation_fingerprint=evaluation_fingerprint,
        runtime_receipt=repair_runtime_receipt,
        caller=repair_caller,
        progress_path=repair_path,
    )
    repair_by_id = {str(row["case_id"]): row for row in repair}
    valid_control_ids = {
        str(row["case_id"]) for row in control if bool(row["valid_output"])
    }
    forbidden_repair_ids = valid_control_ids & set(repair_by_id)
    candidate = [
        _apply_guard(control_by_id[str(case["case_id"])], repair_by_id.get(str(case["case_id"])))
        for case in cases
    ]

    control_metrics = _strict_metrics(control)
    candidate_metrics = _strict_metrics(candidate)
    paired = _paired(control, candidate)
    source_paired = _source_paired(control, candidate)
    route_counts = dict(sorted(Counter(str(row["route"]) for row in candidate).items()))
    parse_reasons = {
        "control": dict(sorted(Counter(str(row["parse_reason"]) for row in control).items())),
        "repair": dict(sorted(Counter(str(row["parse_reason"]) for row in repair).items())),
    }
    valid_repairs = sum(bool(row["valid_output"]) for row in repair)
    correct_repairs = sum(bool(row["correct"]) for row in repair)
    fallback_count = sum(
        row["route"] == "invalid-control-invalid-repair-fallback" for row in candidate
    )
    call_counts = {
        "control_calls": len(control),
        "control_invalid_count": len(invalid_cases),
        "repair_calls": len(repair),
        "valid_control_repair_calls": len(forbidden_repair_ids),
        "total_calls": len(control) + len(repair),
    }
    if call_counts["repair_calls"] != call_counts["control_invalid_count"]:
        raise RuntimeError("E5 repair call count does not match invalid controls")
    if call_counts["valid_control_repair_calls"] != 0:
        raise RuntimeError("E5 repair was called for a valid control")
    if call_counts["total_calls"] != len(cases) + len(invalid_cases):
        raise RuntimeError("E5 total call count is inconsistent")
    if int(paired["control_only"]) != 0:
        raise RuntimeError("E5 structural safety failed with a control-only loss")
    if sum(route_counts.values()) != len(cases) or int(paired["count"]) != len(cases):
        raise RuntimeError("E5 route or paired accounting changed")

    default_gates = {
        "minimum_invalid_control_opportunities": 12,
        "minimum_invalid_control_gold_methods": 3,
        "minimum_valid_control_identity_cases": 50,
        "minimum_candidate_only_gains": 6,
        "minimum_net_improvements": 6,
        "minimum_accuracy_gain_points": 4.0,
        "maximum_mcnemar_p": 0.05,
    }
    gate = _external_gate(
        control=control_metrics,
        candidate=candidate_metrics,
        paired=paired,
        details=candidate,
        gates=default_gates if gates is None else gates,
    )
    private_details = {
        "control": control,
        "repair": repair,
        "candidate": candidate,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "E5 exact H20 two-stage fresh external evaluation",
        "evaluation_fingerprint": evaluation_fingerprint,
        "same_prompt_and_decoding_both_stages": True,
        "control_runtime_receipt_fingerprint": canonical_hash(control_runtime_receipt),
        "repair_runtime_receipt_fingerprint": canonical_hash(repair_runtime_receipt),
        "scores": {"control": control_metrics, "guarded_candidate": candidate_metrics},
        "paired": paired,
        "source_paired": source_paired,
        "route_counts": route_counts,
        "parse_reason_counts": parse_reasons,
        "model_call_counts": call_counts,
        "repair": {
            "valid_repair_count": valid_repairs,
            "correct_repair_count": correct_repairs,
            "fallback_count": fallback_count,
            "precision": correct_repairs / valid_repairs if valid_repairs else 0.0,
            "precision_wilson_95": _wilson_interval(correct_repairs, valid_repairs),
            "correct_rate_per_call": correct_repairs / len(repair) if repair else 0.0,
            "correct_rate_per_call_wilson_95": _wilson_interval(correct_repairs, len(repair)),
        },
        "external_gate": gate,
        "champion_changed": False,
        "release_authorized": False,
        "private_details_fingerprint": canonical_hash(private_details),
        "private_details": private_details,
    }
    report["result_fingerprint"] = _terminal_fingerprint(report)
    _verify_terminal_report(report)
    return report


def _verify_frozen_runtime(master: dict[str, Any]) -> None:
    runtime = dict(master["runtime"])
    for name in ("control", "repair"):
        receipt = dict(runtime[name])
        path = Path(str(receipt["adapter_path"]))
        if sha256_file(path / "adapters.safetensors") != receipt["adapter_sha256"]:
            raise RuntimeError(f"E5 frozen {name} adapter weights changed")
        if sha256_file(path / "adapter_config.json") != receipt["adapter_config_sha256"]:
            raise RuntimeError(f"E5 frozen {name} adapter config changed")
    base = dict(runtime["base_model"])
    snapshot = Path(str(base["snapshot_path"]))
    for name, expected in base["file_sha256"].items():
        if sha256_file(snapshot / str(name)) != expected:
            raise RuntimeError(f"E5 frozen base runtime file changed: {name}")
    source_root = Path(__file__).resolve().parent
    for name, expected in master["implementation_sha256"].items():
        if sha256_file(source_root / str(name)) != expected:
            raise RuntimeError(f"E5 frozen implementation changed: {name}")


def _verify_master_contract(master: dict[str, Any]) -> None:
    unsigned = {key: value for key, value in master.items() if key != "fingerprint"}
    if master.get("fingerprint") != canonical_hash(unsigned):
        raise RuntimeError("E5 master contract fingerprint changed")
    runtime = master.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError("E5 master runtime receipt is missing")
    runtime_unsigned = {key: value for key, value in runtime.items() if key != "fingerprint"}
    if runtime.get("fingerprint") != canonical_hash(runtime_unsigned):
        raise RuntimeError("E5 master runtime fingerprint changed")
    if runtime.get("decoding") != _DECODING:
        raise RuntimeError("E5 master decoding contract changed")
    expected_prompt = canonical_hash(_messages({"question": "<SOURCE_QUESTION>"}))
    if runtime.get("prompt_messages_sha256") != expected_prompt:
        raise RuntimeError("E5 master prompt contract changed")


def _verify_child_contract(contract: dict[str, Any], master: dict[str, Any]) -> None:
    unsigned = {key: value for key, value in contract.items() if key != "fingerprint"}
    if contract.get("fingerprint") != canonical_hash(unsigned):
        raise RuntimeError("E5 selected-source contract fingerprint changed")
    if contract.get("master_contract_fingerprint") != master.get("fingerprint"):
        raise RuntimeError("E5 selected-source contract belongs to another master protocol")
    if contract.get("runtime_fingerprint") != master.get("runtime", {}).get("fingerprint"):
        raise RuntimeError("E5 selected-source runtime changed")
    if contract.get("implementation_sha256") != master.get("implementation_sha256"):
        raise RuntimeError("E5 selected-source implementation changed")
    if not contract.get("source_selected_before_opening"):
        raise RuntimeError("E5 selected source was not frozen before opening")
    if not contract.get("source_qualified") or contract.get("source_qualified_not_blind"):
        raise RuntimeError("E5 selected source is not a blind qualified source")
    source_id = str(contract.get("source", {}).get("source_id", ""))
    source = contract.get("source")
    if not isinstance(source, dict) or not source_id:
        raise RuntimeError("E5 selected-source identity is missing")
    for field in ("stable_id", "license", "snapshot_path", "snapshot_sha256"):
        if not str(source.get(field, "")):
            raise RuntimeError(f"E5 selected-source {field} is missing")
    alias_map = source.get("alias_map")
    extraction = source.get("extraction_contract")
    if not isinstance(alias_map, dict) or source.get("alias_map_fingerprint") != canonical_hash(
        alias_map
    ):
        raise RuntimeError("E5 selected-source alias map changed")
    if not isinstance(extraction, dict) or source.get(
        "extraction_contract_fingerprint"
    ) != canonical_hash(extraction):
        raise RuntimeError("E5 selected-source extraction contract changed")
    if source_id in master.get("source_protocol", {}).get("opened_source_exclusions", {}):
        raise RuntimeError("E5 selected source was already opened during development")


def _require_within(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise RuntimeError(f"E5 {label} is outside the frozen selected-source directory")
    return resolved


def _verify_selected_data(
    data: dict[str, Any],
    contract: dict[str, Any],
    master: dict[str, Any],
    *,
    selected_root: Path,
) -> list[dict[str, Any]]:
    unsigned = {key: value for key, value in data.items() if key != "data_fingerprint"}
    if data.get("data_fingerprint") != canonical_hash(unsigned):
        raise RuntimeError("E5 selected-source data fingerprint changed")
    if data.get("contract_fingerprint") != contract.get("fingerprint"):
        raise RuntimeError("E5 selected-source data belongs to another contract")
    if not data.get("complete") or not data.get("evaluation_authorized"):
        raise RuntimeError("E5 selected-source data did not authorize evaluation")
    source = dict(contract["source"])
    snapshot_path = _require_within(
        Path(str(source["snapshot_path"])), selected_root, label="source snapshot"
    )
    if sha256_file(snapshot_path) != source["snapshot_sha256"]:
        raise RuntimeError("E5 selected-source snapshot changed")
    cases_path = _require_within(
        Path(str(data["cases_path"])), selected_root, label="case file"
    )
    if sha256_file(cases_path) != data.get("cases_sha256"):
        raise RuntimeError("E5 selected-source cases changed")
    cases = list(read_jsonl(cases_path))
    if canonical_hash(cases) != data.get("case_fingerprint"):
        raise RuntimeError("E5 selected-source case fingerprint changed")
    if len(cases) != int(data.get("case_count", -1)):
        raise RuntimeError("E5 selected-source case count changed")
    expected_case_fields = {"case_id", "source_id", "question", "gold_method_id"}
    for case in cases:
        if set(case) != expected_case_fields:
            raise RuntimeError("E5 selected-source case schema contains forbidden fields")
        _case_fingerprint(case)
    source_ids = {str(case["source_id"]) for case in cases}
    if source_ids != {str(source["source_id"])}:
        raise RuntimeError("E5 cases do not belong to exactly the selected source")

    frame_path = _require_within(
        Path(str(data["complete_frame_path"])), selected_root, label="complete source frame"
    )
    if sha256_file(frame_path) != data.get("complete_frame_sha256"):
        raise RuntimeError("E5 complete source frame changed")
    frame = list(read_jsonl(frame_path))
    if canonical_hash(frame) != data.get("complete_frame_fingerprint"):
        raise RuntimeError("E5 complete source frame fingerprint changed")
    frame_ids = [str(row.get("frame_id", "")) for row in frame]
    if not frame_ids or len(frame_ids) != len(set(frame_ids)):
        raise RuntimeError("E5 complete source frame IDs are empty or duplicated")
    eligible_frame = [
        row for row in frame if str(row.get("mapped_method_id", "")) in _METHOD_IDS
    ]
    eligible_case_ids = [str(row.get("case_id", "")) for row in eligible_frame]
    case_ids = [str(case["case_id"]) for case in cases]
    if len(eligible_case_ids) != len(set(eligible_case_ids)) or set(eligible_case_ids) != set(
        case_ids
    ):
        raise RuntimeError("E5 complete source frame and materialized cases changed")
    if data.get("alias_map_fingerprint") != source["alias_map_fingerprint"]:
        raise RuntimeError("E5 data alias map changed")
    if data.get("extraction_contract_fingerprint") != source[
        "extraction_contract_fingerprint"
    ]:
        raise RuntimeError("E5 data extraction contract changed")

    overlap_path = _require_within(
        Path(str(data["overlap_manifest_path"])), selected_root, label="overlap manifest"
    )
    if sha256_file(overlap_path) != data.get("overlap_manifest_sha256"):
        raise RuntimeError("E5 overlap manifest changed")
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    if not isinstance(overlap, dict):
        raise RuntimeError("E5 overlap manifest is malformed")
    overlap_unsigned = {key: value for key, value in overlap.items() if key != "fingerprint"}
    if overlap.get("fingerprint") != canonical_hash(overlap_unsigned):
        raise RuntimeError("E5 overlap manifest fingerprint changed")
    if int(overlap.get("overlap_count", -1)) != 0:
        raise RuntimeError("E5 selected source overlaps historical evidence")
    if overlap.get("case_fingerprint") != data.get("case_fingerprint"):
        raise RuntimeError("E5 overlap manifest belongs to another case set")
    for field in ("normalization_fingerprint", "historical_corpus_fingerprint"):
        if not str(overlap.get(field, "")):
            raise RuntimeError(f"E5 overlap manifest {field} is missing")

    source_gates = dict(master["source_protocol"]["qualification_gates"])
    method_counts = Counter(str(case["gold_method_id"]) for case in cases)
    coverage = len(cases) / len(frame)
    maximum_share = max(method_counts.values()) / len(cases)
    checks = {
        "minimum_eligible_cases": len(cases) >= int(source_gates["minimum_eligible_cases"]),
        "minimum_distinct_methods": len(method_counts)
        >= int(source_gates["minimum_distinct_methods"]),
        "minimum_coverage_fraction": coverage
        >= float(source_gates["minimum_coverage_fraction"]),
        "maximum_single_method_fraction": maximum_share
        <= float(source_gates["maximum_single_method_fraction"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"E5 selected-source qualification changed: {checks}")
    return cases


def run_guarded_external_evaluation(config: ProjectConfig) -> dict[str, Any]:
    """Run a selected E5 child contract, or fail before model load when none exists."""

    root = config.path_for("artifact_dir") / "guarded-external-v1"
    master_path = root / "master-contract.json"
    child_path = root / "selected-source-contract.json"
    data_path = root / "selected-source-data.json"
    if not master_path.is_file():
        raise RuntimeError("E5 master contract is missing")
    if not child_path.is_file() or not data_path.is_file():
        raise RuntimeError(
            "E5 has no preregistered qualified source; model evaluation remains unopened"
        )
    master = json.loads(master_path.read_text(encoding="utf-8"))
    contract = json.loads(child_path.read_text(encoding="utf-8"))
    data = json.loads(data_path.read_text(encoding="utf-8"))
    _verify_master_contract(master)
    _verify_child_contract(contract, master)
    _verify_frozen_runtime(master)
    cases = _verify_selected_data(
        data,
        contract,
        master,
        selected_root=root / "selected-source",
    )
    evaluation_fingerprint = canonical_hash(
        {
            "master": master["fingerprint"],
            "contract": contract["fingerprint"],
            "data": data["data_fingerprint"],
            "runtime": master["runtime"]["fingerprint"],
            "evaluator_version": _EVALUATOR_VERSION,
        }
    )
    report_path = root / "report.json"
    public_path = config.root / "reports" / "evolve" / "guarded-external-v1.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        _verify_terminal_report(existing)
        if existing.get("evaluation_fingerprint") != evaluation_fingerprint:
            raise RuntimeError("E5 terminal report belongs to another evaluation")
        public = _public_report(existing)
        write_json(public_path, public)
        return public

    opening_receipt = {
        "schema_version": 1,
        "evaluation_fingerprint": evaluation_fingerprint,
        "master_contract_fingerprint": master["fingerprint"],
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["data_fingerprint"],
        "case_count": len(cases),
        "model_output_opened": True,
    }
    opening_receipt["fingerprint"] = canonical_hash(opening_receipt)
    opening_path = root / "evaluation-opened.json"
    if opening_path.exists():
        existing_opening = json.loads(opening_path.read_text(encoding="utf-8"))
        if existing_opening != opening_receipt:
            raise RuntimeError("E5 evaluation opening receipt changed")
    else:
        write_json(opening_path, opening_receipt)

    runtime = dict(master["runtime"])
    control_header = _phase_header(
        cases,
        phase="control",
        evaluation_fingerprint=evaluation_fingerprint,
        runtime_receipt=dict(runtime["control"]),
    )
    progress_root = root / "progress"
    completed_controls = len(
        _load_strict_progress(progress_root / "control-ledger.json", header=control_header)
    )
    control_caller = _AgentCaller(
        config,
        adapter_path=str(runtime["control"]["adapter_path"]),
        expected_calls=len(cases) - completed_controls,
    )
    repair_caller = _AgentCaller(
        config,
        adapter_path=str(runtime["repair"]["adapter_path"]),
        expected_calls=None,
    )
    try:
        report = run_exact_two_stage(
            cases,
            evaluation_fingerprint=evaluation_fingerprint,
            control_runtime_receipt=dict(runtime["control"]),
            repair_runtime_receipt=dict(runtime["repair"]),
            control_caller=control_caller,
            repair_caller=repair_caller,
            progress_root=progress_root,
            gates=dict(master["evaluation_gates"]),
        )
    finally:
        control_caller.close()
        repair_caller.close()
    report.update(
        {
            "master_contract_fingerprint": master["fingerprint"],
            "contract_fingerprint": contract["fingerprint"],
            "data_fingerprint": data["data_fingerprint"],
            "source_qualified_not_blind": False,
            "fresh_external_evidence": True,
            "claim_boundary": master["claim_boundary"],
        }
    )
    report["result_fingerprint"] = _terminal_fingerprint(report)
    _verify_terminal_report(report)
    write_json(report_path, report)
    public = _public_report(report)
    write_json(public_path, public)
    return public

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from charlie_alpha import stats_guarded_external as guarded_external
from charlie_alpha.io_utils import canonical_hash
from charlie_alpha.stats_guarded_external import (
    _parse_prediction as parse_prediction,
)
from charlie_alpha.stats_guarded_external import (
    _phase_header as phase_header,
)
from charlie_alpha.stats_guarded_external import (
    _public_report as build_public_report,
)
from charlie_alpha.stats_guarded_external import (
    _terminal_fingerprint as terminal_report_fingerprint,
)
from charlie_alpha.stats_guarded_external import (
    _validate_ledger as validate_strict_progress,
)
from charlie_alpha.stats_guarded_external import run_exact_two_stage

_DECODING = {
    "max_tokens": 160,
    "temperature": 0.0,
    "top_p": 1.0,
}
_EVALUATION_FINGERPRINT = "e5-exact-two-stage-evaluation-fingerprint"
_CONTROL_RUNTIME = {
    "adapter_sha256": "parent-adapter-sha256",
    "adapter_config_sha256": "parent-adapter-config-sha256",
    "runtime_sha256": "exact-two-stage-runtime-sha256",
}
_REPAIR_RUNTIME = {
    "adapter_sha256": "repair-adapter-sha256",
    "adapter_config_sha256": "repair-adapter-config-sha256",
    "runtime_sha256": "exact-two-stage-runtime-sha256",
}
_FROZEN_GATES = {
    "minimum_invalid_control_opportunities": 12,
    "minimum_invalid_control_gold_methods": 3,
    "minimum_valid_control_identity_cases": 50,
    "minimum_candidate_only_gains": 6,
    "minimum_net_improvements": 6,
    "minimum_accuracy_gain_points": 4.0,
    "maximum_mcnemar_p": 0.05,
}
_CASES = [
    {
        "case_id": "e5-case-valid-control",
        "source_id": "fresh-source",
        "question": "E5 private question alpha",
        "gold_method_id": "paired_t",
    },
    {
        "case_id": "e5-case-valid-repair",
        "source_id": "fresh-source",
        "question": "E5 private question beta",
        "gold_method_id": "mann_whitney",
    },
    {
        "case_id": "e5-case-invalid-fallback",
        "source_id": "fresh-source",
        "question": "E5 private question gamma",
        "gold_method_id": "ols",
    },
]


class FakeModelCall:
    """A pure fake that records the exact request passed to each model stage."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        case_id: str,
        messages: list[dict[str, str]],
        decoding: dict[str, Any],
    ) -> str:
        self.calls.append(
            {
                "case_id": case_id,
                "messages": deepcopy(messages),
                "decoding": deepcopy(decoding),
            }
        )
        return self.answers[case_id]


def _answer(method_id: str, *, column: str) -> str:
    return json.dumps(
        {"methods": [method_id], "columns": [column]},
        sort_keys=True,
        separators=(",", ":"),
    )


def _run(
    cases: list[dict[str, Any]],
    *,
    control: FakeModelCall,
    repair: FakeModelCall,
    progress_root: Path | None = None,
) -> dict[str, Any]:
    return run_exact_two_stage(
        cases,
        evaluation_fingerprint=_EVALUATION_FINGERPRINT,
        control_runtime_receipt=deepcopy(_CONTROL_RUNTIME),
        repair_runtime_receipt=deepcopy(_REPAIR_RUNTIME),
        control_caller=control,
        repair_caller=repair,
        progress_root=progress_root,
    )


def _run_mixed(
    *, progress_root: Path | None = None
) -> tuple[dict[str, Any], FakeModelCall, FakeModelCall]:
    control = FakeModelCall(
        {
            "e5-case-valid-control": _answer("paired_t", column="private-alpha-column"),
            "e5-case-valid-repair": "CONTROL_NO_JSON_BETA",
            "e5-case-invalid-fallback": "",
        }
    )
    repair = FakeModelCall(
        {
            "e5-case-valid-repair": _answer(
                "mann_whitney", column="private-beta-column"
            ),
            "e5-case-invalid-fallback": _answer(
                "not_in_the_catalog", column="private-gamma-column"
            ),
        }
    )
    report = _run(
        deepcopy(_CASES),
        control=control,
        repair=repair,
        progress_root=progress_root,
    )
    return report, control, repair


def _by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["case_id"]): row for row in rows}


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _keys(item)}
    return set()


def _resign_ledger(payload: dict[str, Any]) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "ledger_fingerprint"}
    payload["ledger_fingerprint"] = canonical_hash(unsigned)


def _master_contract() -> dict[str, Any]:
    placeholder = deepcopy(_CASES[0])
    placeholder["question"] = "<SOURCE_QUESTION>"
    prompt_header = phase_header(
        [placeholder],
        phase="control",
        evaluation_fingerprint="prompt-contract-probe",
        runtime_receipt=deepcopy(_CONTROL_RUNTIME),
    )
    runtime: dict[str, Any] = {
        "decoding": deepcopy(_DECODING),
        "prompt_messages_sha256": prompt_header["expected_receipts"][0][
            "messages_sha256"
        ],
    }
    runtime["fingerprint"] = canonical_hash(runtime)
    master: dict[str, Any] = {
        "runtime": runtime,
        "evaluation_gates": deepcopy(_FROZEN_GATES),
    }
    master["fingerprint"] = canonical_hash(master)
    return master


def test_all_valid_controls_make_zero_repair_calls() -> None:
    cases = deepcopy(_CASES[:2])
    control = FakeModelCall(
        {
            cases[0]["case_id"]: _answer("paired_t", column="before"),
            # Valid but wrong is still an identity route; routing cannot inspect gold correctness.
            cases[1]["case_id"]: _answer("ols", column="outcome"),
        }
    )
    repair = FakeModelCall({})

    report = _run(cases, control=control, repair=repair)
    controls = _by_id(report["private_details"]["control"])
    candidates = _by_id(report["private_details"]["candidate"])

    assert [call["case_id"] for call in control.calls] == [
        "e5-case-valid-control",
        "e5-case-valid-repair",
    ]
    assert repair.calls == []
    assert report["model_call_counts"] == {
        "control_calls": 2,
        "control_invalid_count": 0,
        "repair_calls": 0,
        "valid_control_repair_calls": 0,
        "total_calls": 2,
    }
    assert report["route_counts"] == {"valid-control-identity": 2}
    for case_id, candidate in candidates.items():
        control_row = controls[case_id]
        assert candidate["route"] == "valid-control-identity"
        assert candidate["repair_called"] is False
        assert candidate["predicted_method_id"] == control_row["predicted_method_id"]
        assert candidate["valid_output"] == control_row["valid_output"]
        assert candidate["correct"] == control_row["correct"]
        assert candidate["control_raw_answer_sha256"] == control_row["raw_answer_sha256"]


def test_mixed_runtime_only_repairs_invalid_controls_and_preserves_three_routes() -> None:
    report, _, repair = _run_mixed()
    controls = _by_id(report["private_details"]["control"])
    repairs = _by_id(report["private_details"]["repair"])
    candidates = _by_id(report["private_details"]["candidate"])

    assert [call["case_id"] for call in repair.calls] == [
        "e5-case-valid-repair",
        "e5-case-invalid-fallback",
    ]

    identity = candidates["e5-case-valid-control"]
    identity_control = controls["e5-case-valid-control"]
    assert identity["route"] == "valid-control-identity"
    assert identity["repair_called"] is False
    assert identity["predicted_method_id"] == identity_control["predicted_method_id"]
    assert identity["control_raw_answer_sha256"] == identity_control["raw_answer_sha256"]

    repaired = candidates["e5-case-valid-repair"]
    repair_row = repairs["e5-case-valid-repair"]
    assert repaired["route"] == "invalid-control-valid-repair"
    assert repaired["repair_valid_output"] is True
    assert repaired["predicted_method_id"] == repair_row["predicted_method_id"]
    assert repaired["predicted_method_id"] == "mann_whitney"
    assert repaired["repair_raw_answer_sha256"] == repair_row["raw_answer_sha256"]

    fallback = candidates["e5-case-invalid-fallback"]
    fallback_control = controls["e5-case-invalid-fallback"]
    assert fallback["route"] == "invalid-control-invalid-repair-fallback"
    assert fallback["repair_valid_output"] is False
    assert fallback["predicted_method_id"] == fallback_control["predicted_method_id"]
    assert fallback["valid_output"] == fallback_control["valid_output"]
    assert fallback["control_raw_answer_sha256"] == fallback_control["raw_answer_sha256"]


def test_control_and_repair_messages_and_decoding_are_identical() -> None:
    _, control, repair = _run_mixed()
    control_by_id = {call["case_id"]: call for call in control.calls}

    assert all(call["decoding"] == _DECODING for call in control.calls)
    assert all(call["decoding"] == _DECODING for call in repair.calls)
    for repair_call in repair.calls:
        control_call = control_by_id[repair_call["case_id"]]
        assert repair_call["messages"] == control_call["messages"]
        assert repair_call["decoding"] == control_call["decoding"]


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        pytest.param("", "empty-output", id="empty"),
        pytest.param(
            '{"methods":["not_in_the_catalog"],"columns":[]}',
            "unknown-method-id",
            id="unknown-id",
        ),
        pytest.param(
            '{"methods":["paired_t","ols"],"columns":[]}',
            "multiple-methods",
            id="multiple-methods",
        ),
        pytest.param("paired_t", "no-json-object", id="no-json"),
    ],
)
def test_parser_classifies_required_invalid_outputs(answer: str, reason: str) -> None:
    parsed = parse_prediction(answer)

    assert parsed["valid_output"] is False
    assert parsed["predicted_method_id"] is None
    assert parsed["parse_reason"] == reason


@pytest.mark.parametrize(
    "change",
    [
        "duplicate-id",
        "completed-mismatch",
        "case-hash-changed",
        "request-hash-changed",
        "adapter-hash-changed",
        "runtime-hash-changed",
    ],
)
def test_strict_progress_fails_closed_on_integrity_change(
    change: str, tmp_path: Path
) -> None:
    _run_mixed(progress_root=tmp_path)
    payload = json.loads((tmp_path / "control-ledger.json").read_text(encoding="utf-8"))
    expected_header = phase_header(
        deepcopy(_CASES),
        phase="control",
        evaluation_fingerprint=_EVALUATION_FINGERPRINT,
        runtime_receipt=deepcopy(_CONTROL_RUNTIME),
    )
    validate_strict_progress(payload, expected_header=expected_header)

    if change == "duplicate-id":
        payload["rows"][1] = deepcopy(payload["rows"][0])
        _resign_ledger(payload)
    elif change == "completed-mismatch":
        payload["completed"] -= 1
        _resign_ledger(payload)
    elif change == "case-hash-changed":
        changed_cases = deepcopy(_CASES)
        changed_cases[0]["question"] = "tampered private question"
        expected_header = phase_header(
            changed_cases,
            phase="control",
            evaluation_fingerprint=_EVALUATION_FINGERPRINT,
            runtime_receipt=deepcopy(_CONTROL_RUNTIME),
        )
    elif change == "request-hash-changed":
        payload["rows"][0]["request_hash"] = "tampered-request-hash"
        _resign_ledger(payload)
    elif change == "adapter-hash-changed":
        changed_runtime = deepcopy(_CONTROL_RUNTIME)
        changed_runtime["adapter_sha256"] = "different-parent-adapter-sha256"
        expected_header = phase_header(
            deepcopy(_CASES),
            phase="control",
            evaluation_fingerprint=_EVALUATION_FINGERPRINT,
            runtime_receipt=changed_runtime,
        )
    elif change == "runtime-hash-changed":
        changed_runtime = deepcopy(_CONTROL_RUNTIME)
        changed_runtime["runtime_sha256"] = "different-runtime-sha256"
        expected_header = phase_header(
            deepcopy(_CASES),
            phase="control",
            evaluation_fingerprint=_EVALUATION_FINGERPRINT,
            runtime_receipt=changed_runtime,
        )
    else:  # pragma: no cover - the parametrization is exhaustive.
        raise AssertionError(f"Unhandled mutation: {change}")

    with pytest.raises(RuntimeError):
        validate_strict_progress(payload, expected_header=expected_header)


@pytest.mark.parametrize("interrupted_phase", ["control", "repair"])
def test_started_receipt_after_model_return_fails_closed_without_recall(
    interrupted_phase: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = deepcopy(_CASES[1])
    case_id = str(case["case_id"])
    first_control = FakeModelCall({case_id: "CONTROL_NO_JSON_BETA"})
    first_repair = FakeModelCall(
        {case_id: _answer("mann_whitney", column="private-beta-column")}
    )
    complete_progress = guarded_external._complete_strict_progress

    def interrupt_after_model_return(
        path: Path,
        *,
        header: dict[str, Any],
        row: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if header["phase"] == interrupted_phase:
            raise RuntimeError("simulated interruption after model return")
        return complete_progress(path, header=header, row=row)

    monkeypatch.setattr(
        guarded_external,
        "_complete_strict_progress",
        interrupt_after_model_return,
    )
    with pytest.raises(RuntimeError, match="simulated interruption after model return"):
        _run(
            [case],
            control=first_control,
            repair=first_repair,
            progress_root=tmp_path,
        )

    assert len(first_control.calls) == 1
    assert len(first_repair.calls) == (1 if interrupted_phase == "repair" else 0)
    ledger_path = tmp_path / f"{interrupted_phase}-ledger.json"
    interrupted_ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert interrupted_ledger["completed"] == 0
    assert interrupted_ledger["complete"] is False
    assert interrupted_ledger["inflight"]["case_id"] == case_id
    assert interrupted_ledger["inflight"]["phase"] == interrupted_phase
    assert interrupted_ledger["inflight"]["state"] == "started"
    sealed_ledger = ledger_path.read_bytes()

    monkeypatch.setattr(
        guarded_external,
        "_complete_strict_progress",
        complete_progress,
    )
    resumed_control = FakeModelCall({case_id: _answer("paired_t", column="retry")})
    resumed_repair = FakeModelCall(
        {case_id: _answer("mann_whitney", column="retry")}
    )
    with pytest.raises(RuntimeError, match="request must not be retried"):
        _run(
            [case],
            control=resumed_control,
            repair=resumed_repair,
            progress_root=tmp_path,
        )

    assert resumed_control.calls == []
    assert resumed_repair.calls == []
    assert ledger_path.read_bytes() == sealed_ledger


@pytest.mark.parametrize(
    "change",
    [
        "master-fingerprint",
        "runtime-fingerprint",
        "decoding",
        "sealed-gates",
    ],
)
def test_master_contract_rejects_fingerprint_decoding_and_gate_drift(change: str) -> None:
    master = _master_contract()
    guarded_external._verify_master_contract(master)

    if change == "master-fingerprint":
        master["fingerprint"] = "changed-master-fingerprint"
        expected_error = "master contract fingerprint changed"
    elif change == "runtime-fingerprint":
        master["runtime"]["fingerprint"] = "changed-runtime-fingerprint"
        master["fingerprint"] = canonical_hash(
            {key: value for key, value in master.items() if key != "fingerprint"}
        )
        expected_error = "master runtime fingerprint changed"
    elif change == "decoding":
        master["runtime"]["decoding"]["temperature"] = 0.1
        master["runtime"]["fingerprint"] = canonical_hash(
            {
                key: value
                for key, value in master["runtime"].items()
                if key != "fingerprint"
            }
        )
        master["fingerprint"] = canonical_hash(
            {key: value for key, value in master.items() if key != "fingerprint"}
        )
        expected_error = "master decoding contract changed"
    elif change == "sealed-gates":
        master["evaluation_gates"]["minimum_candidate_only_gains"] = 5
        expected_error = "master contract fingerprint changed"
    else:  # pragma: no cover - the parametrization is exhaustive.
        raise AssertionError(f"Unhandled mutation: {change}")

    with pytest.raises(RuntimeError, match=expected_error):
        guarded_external._verify_master_contract(master)


def test_agent_callers_load_parent_then_repair_without_adapter_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str]] = []

    class FakeAgent:
        def __init__(self, config: Any, *, adapter_path: str) -> None:
            del config
            self.adapter_path = adapter_path
            events.append(("load", adapter_path))

        def answer_without_tools(
            self,
            messages: list[dict[str, str]],
            *,
            route: str,
            max_tokens: int,
            temperature: float,
            top_p: float,
        ) -> str:
            del messages, route, max_tokens, temperature, top_p
            events.append(("answer", self.adapter_path))
            return _answer("paired_t", column=self.adapter_path)

    monkeypatch.setattr(guarded_external, "StatsAgent", FakeAgent)
    monkeypatch.setattr(guarded_external.gc, "collect", lambda: 0)
    monkeypatch.setattr(guarded_external.mx, "clear_cache", lambda: None)
    parent = guarded_external._AgentCaller(
        object(), adapter_path="parent-adapter", expected_calls=1
    )
    repair = guarded_external._AgentCaller(
        object(), adapter_path="repair-adapter", expected_calls=None
    )

    assert events == []
    assert parent.agent is None
    assert repair.agent is None

    parent("parent-case", [], deepcopy(_DECODING))

    assert events == [("load", "parent-adapter"), ("answer", "parent-adapter")]
    assert parent.agent is None
    assert repair.agent is None

    repair("repair-case", [], deepcopy(_DECODING))

    assert events == [
        ("load", "parent-adapter"),
        ("answer", "parent-adapter"),
        ("load", "repair-adapter"),
        ("answer", "repair-adapter"),
    ]
    assert parent.agent is None
    assert repair.agent is not None
    repair.close()
    assert repair.agent is None


def test_call_route_paired_and_validity_counts_have_one_denominator() -> None:
    report, control_caller, repair_caller = _run_mixed()
    case_count = len(_CASES)
    calls = report["model_call_counts"]
    routes = report["route_counts"]
    paired = report["paired"]
    control = report["scores"]["control"]
    candidate = report["scores"]["guarded_candidate"]
    control_details = report["private_details"]["control"]
    candidate_details = report["private_details"]["candidate"]

    assert calls["control_calls"] == case_count
    assert calls["repair_calls"] == calls["control_invalid_count"]
    assert calls["total_calls"] == calls["control_calls"] + calls["repair_calls"]
    assert calls["control_calls"] == len(control_caller.calls)
    assert calls["repair_calls"] == len(repair_caller.calls)
    assert sum(routes.values()) == case_count
    assert routes == {
        route: sum(row["route"] == route for row in candidate_details) for route in routes
    }
    assert paired["count"] == case_count
    assert (
        paired["both_correct"]
        + paired["candidate_only"]
        + paired["control_only"]
        + paired["both_wrong"]
        == case_count
    )
    assert paired["net_improvements"] == paired["candidate_only"] - paired["control_only"]
    assert control["valid_count"] + (control["count"] - control["valid_count"]) == case_count
    assert control["valid_count"] == sum(row["valid_output"] for row in control_details)
    assert calls["control_invalid_count"] == sum(
        not row["valid_output"] for row in control_details
    )
    assert (
        candidate["valid_count"] + (candidate["count"] - candidate["valid_count"])
        == case_count
    )
    assert candidate["valid_count"] == sum(row["valid_output"] for row in candidate_details)


def test_public_report_removes_private_fields_and_binds_private_details() -> None:
    report, _, _ = _run_mixed()
    public = build_public_report(report)
    serialized = json.dumps(public, ensure_ascii=False, sort_keys=True)

    assert "private_details" not in public
    assert public["private_details_fingerprint"] == canonical_hash(report["private_details"])
    assert report["private_details_fingerprint"] == public["private_details_fingerprint"]
    assert not {"case_id", "question", "gold", "gold_method_id", "raw_answer"} & _keys(
        public
    )
    for case in _CASES:
        assert case["case_id"] not in serialized
        assert case["question"] not in serialized
    for raw_fragment in (
        "private-alpha-column",
        "CONTROL_NO_JSON_BETA",
        "private-beta-column",
        "private-gamma-column",
    ):
        assert raw_fragment not in serialized

    tampered = deepcopy(report)
    tampered["private_details"]["candidate"][0]["question"] = "tampered private detail"
    with pytest.raises(RuntimeError):
        build_public_report(tampered)


@pytest.mark.parametrize(
    "malformed_private_details",
    [
        pytest.param("missing", id="missing"),
        pytest.param("null", id="null"),
        pytest.param("list", id="list"),
    ],
)
def test_internal_terminal_verifier_rejects_non_dict_private_details(
    malformed_private_details: str,
) -> None:
    report, _, _ = _run_mixed()
    malformed = deepcopy(report)
    if malformed_private_details == "missing":
        malformed.pop("private_details")
    elif malformed_private_details == "null":
        malformed["private_details"] = None
    else:
        malformed["private_details"] = []

    assert malformed["result_fingerprint"] == terminal_report_fingerprint(malformed)
    with pytest.raises(RuntimeError, match="internal terminal report lost its private details"):
        guarded_external._verify_terminal_report(malformed)


def test_terminal_report_fingerprint_is_recomputable_and_detects_tampering() -> None:
    report, _, _ = _run_mixed()
    unsigned = {
        key: value
        for key, value in report.items()
        if key not in {"result_fingerprint", "private_details"}
    }

    assert report["complete"] is True
    assert report["result_fingerprint"] == canonical_hash(unsigned)
    assert report["result_fingerprint"] == terminal_report_fingerprint(report)

    tampered = deepcopy(report)
    tampered["model_call_counts"]["repair_calls"] += 1
    assert terminal_report_fingerprint(tampered) != report["result_fingerprint"]

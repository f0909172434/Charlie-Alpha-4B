from __future__ import annotations

import json
from pathlib import Path

import pytest

import charlie_alpha.stats_catalog_fallback_external as e6
from charlie_alpha.io_utils import canonical_hash, sha256_file, sha256_text, write_json, write_jsonl
from charlie_alpha.stats_catalog_fallback_external import (
    _external_gate,
    _public_report,
    _template_accounting,
    run_catalog_fallback_two_stage,
)


def _case(index: int, *, question: str, gold: str) -> dict:
    return {
        "case_id": f"case-{index}",
        "source_id": "fresh-source",
        "question": question,
        "gold_method_id": gold,
    }


def _terminal_fixture(tmp_path: Path) -> tuple[dict, dict, dict, dict, list[dict], str, Path]:
    cases = [
        _case(0, question="Valid control", gold="ols"),
        _case(1, question="Invalid control", gold="independent_t"),
    ]
    opportunity_gates = {
        "minimum_invalid_control_opportunities": 1,
        "minimum_invalid_control_gold_methods": 1,
        "minimum_distinct_invalid_question_templates": 1,
        "maximum_single_invalid_template_fraction": 1.0,
        "maximum_single_invalid_method_case_fraction": 1.0,
        "maximum_single_invalid_method_template_fraction": 1.0,
        "minimum_valid_control_identity_cases": 1,
    }
    result_gates = {
        "minimum_candidate_only_gains": 1,
        "minimum_net_improvements": 1,
        "minimum_accuracy_gain_points": 50.0,
        "maximum_case_mcnemar_p": 1.0,
        "minimum_distinct_repaired_question_templates": 1,
        "minimum_template_net_improvements": 1,
        "minimum_repaired_gold_methods": 1,
        "maximum_single_repaired_method_fraction": 1.0,
        "maximum_template_mcnemar_p": 1.0,
        "minimum_valid_fallback_precision": 1.0,
        "maximum_valid_but_wrong_fallbacks": 0,
    }
    master = {
        "fingerprint": "master-fingerprint",
        "runtime": {
            "parent": {"adapter_sha256": "same-parent"},
            "control_prompt_sha256": "menu-free",
            "fallback_prompt_sha256": "fixed-catalog",
        },
        "opportunity_gates": opportunity_gates,
        "result_gates": result_gates,
        "claim_boundary": "single-source-only",
    }
    child = {"fingerprint": "child-fingerprint"}
    data = {"data_fingerprint": "data-fingerprint"}
    evaluation_fingerprint = "evaluation"
    root = tmp_path / "terminal"
    progress_root = root / "progress"

    def control(case_id, _messages, _decoding):
        if case_id == "case-0":
            return '{"methods":["ols"],"columns":[]}'
        return '{"methods":["out-of-catalog"],"columns":[]}'

    def fallback(_case_id, _messages, _decoding):
        return '{"methods":["independent_t"],"columns":[]}'

    control_receipt, fallback_receipt = e6._runtime_phase_receipts(master)
    report = run_catalog_fallback_two_stage(
        cases,
        evaluation_fingerprint=evaluation_fingerprint,
        control_runtime_receipt=control_receipt,
        fallback_runtime_receipt=fallback_receipt,
        control_caller=control,
        fallback_caller=fallback,
        before_fallback=lambda: None,
        progress_root=progress_root,
        opportunity_gates=opportunity_gates,
        result_gates=result_gates,
    )
    opening = {
        "schema_version": 1,
        "method": "E6_FRESH_SOURCE_CATALOG_FALLBACK_V1",
        "evaluation_fingerprint": evaluation_fingerprint,
        "master_contract_fingerprint": master["fingerprint"],
        "child_contract_fingerprint": child["fingerprint"],
        "data_fingerprint": data["data_fingerprint"],
        "case_count": len(cases),
        "model_output_opened": True,
        "training_authorized": False,
        "champion_changed": False,
        "release_authorized": False,
    }
    opening["fingerprint"] = canonical_hash(opening)
    opening_path = root / "evaluation-opened.json"
    write_json(opening_path, opening)
    report.update(
        {
            "master_contract_fingerprint": master["fingerprint"],
            "child_contract_fingerprint": child["fingerprint"],
            "data_fingerprint": data["data_fingerprint"],
            "opening_fingerprint": opening["fingerprint"],
            "opening_sha256": sha256_file(opening_path),
            "control_ledger_sha256": sha256_file(progress_root / "control-ledger.json"),
            "fallback_ledger_sha256": sha256_file(progress_root / "fallback-ledger.json"),
            "fresh_external_evidence": True,
            "training_authorized": False,
            "champion_changed": False,
            "release_authorized": False,
            "claim_boundary": master["claim_boundary"],
        }
    )
    report["result_fingerprint"] = e6._terminal_fingerprint(report)
    return report, master, child, data, cases, evaluation_fingerprint, root


def test_exact_e6_runtime_calls_catalog_only_for_invalid_controls() -> None:
    cases = [
        _case(0, question="Question zero", gold="ols"),
        _case(1, question="Question one", gold="independent_t"),
        _case(2, question="Question two", gold="wilcoxon_signed_rank"),
    ]
    control_answers = {
        "case-0": '{"methods":["ols"],"columns":[]}',
        "case-1": '{"methods":["t-test"],"columns":[]}',
        "case-2": '{"methods":["sign-test"],"columns":[]}',
    }
    fallback_answers = {
        "case-1": '{"methods":["independent_t"],"columns":[]}',
        "case-2": '{"methods":["wilcoxon_signed_rank"],"columns":[]}',
    }
    control_calls = []
    fallback_calls = []

    def control_caller(case_id, messages, _decoding):
        control_calls.append((case_id, messages))
        return control_answers[case_id]

    def fallback_caller(case_id, messages, _decoding):
        fallback_calls.append((case_id, messages))
        return fallback_answers[case_id]

    parent = {"adapter_sha256": "same-parent"}
    report = run_catalog_fallback_two_stage(
        cases,
        evaluation_fingerprint="evaluation",
        control_runtime_receipt={
            **parent,
            "prompt_sha256": "menu-free",
        },
        fallback_runtime_receipt={
            **parent,
            "prompt_sha256": "fixed-catalog",
        },
        control_caller=control_caller,
        fallback_caller=fallback_caller,
        opportunity_gates={
            "minimum_invalid_control_opportunities": 2,
            "minimum_invalid_control_gold_methods": 2,
            "minimum_distinct_invalid_question_templates": 2,
            "maximum_single_invalid_template_fraction": 0.5,
            "maximum_single_invalid_method_case_fraction": 0.5,
            "maximum_single_invalid_method_template_fraction": 0.5,
            "minimum_valid_control_identity_cases": 1,
        },
        result_gates={
            "minimum_candidate_only_gains": 2,
            "minimum_net_improvements": 2,
            "minimum_accuracy_gain_points": 50.0,
            "maximum_case_mcnemar_p": 1.0,
            "minimum_distinct_repaired_question_templates": 2,
            "minimum_template_net_improvements": 2,
            "minimum_repaired_gold_methods": 2,
            "maximum_single_repaired_method_fraction": 0.5,
            "maximum_template_mcnemar_p": 1.0,
            "minimum_valid_fallback_precision": 1.0,
            "maximum_valid_but_wrong_fallbacks": 0,
        },
    )

    assert [case_id for case_id, _ in control_calls] == ["case-0", "case-1", "case-2"]
    assert [case_id for case_id, _ in fallback_calls] == ["case-1", "case-2"]
    assert all(
        "Repository method catalog" not in messages[0]["content"] for _, messages in control_calls
    )
    assert all(
        "Repository method catalog" in messages[0]["content"] for _, messages in fallback_calls
    )
    assert report["scores"]["control"]["correct_count"] == 1
    assert report["scores"]["catalog_guard"]["correct_count"] == 3
    assert report["external_gate"]["status"] == "CONFIRMED_NARROW_FRESH_SOURCE_PASS"
    assert report["model_call_counts"]["valid_control_fallback_calls"] == 0
    assert report["runtime_integrity"]["valid_control_identity_fraction"] == 1.0
    assert "private_details" not in _public_report(report)
    assert "details" not in report["template_paired"]


def test_template_accounting_counts_identical_questions_once() -> None:
    controls = []
    candidates = []
    for index, source in enumerate(["a", "b", "c"]):
        controls.append(
            {
                "case_id": f"case-{index}",
                "source_id": source,
                "question": "Same question",
                "gold_method_id": "independent_t",
                "correct": False,
            }
        )
        candidates.append({**controls[-1], "correct": True})
    controls.append(
        {
            "case_id": "case-3",
            "source_id": "a",
            "question": "Different question",
            "gold_method_id": "wilcoxon_signed_rank",
            "correct": False,
        }
    )
    candidates.append({**controls[-1], "correct": True})

    accounting = _template_accounting(controls, candidates)

    assert accounting["template_count"] == 2
    assert accounting["candidate_only"] == 2
    assert accounting["repaired_gold_methods"] == ["independent_t", "wilcoxon_signed_rank"]


def test_template_accounting_counts_semantic_paraphrases_once() -> None:
    controls = [
        {
            "case_id": "case-a",
            "source_id": "fresh",
            "question": "Compare two independent groups.",
            "gold_method_id": "independent_t",
            "correct": False,
        },
        {
            "case_id": "case-b",
            "source_id": "fresh",
            "question": "Are the means different between unrelated cohorts?",
            "gold_method_id": "independent_t",
            "correct": False,
        },
    ]
    candidates = [{**row, "correct": True} for row in controls]

    accounting = _template_accounting(
        controls,
        candidates,
        {"case-a": "semantic-two-group", "case-b": "semantic-two-group"},
    )

    assert accounting["template_count"] == 1
    assert accounting["candidate_only"] == 1


def test_duplicate_messages_cannot_be_split_across_templates() -> None:
    controls = [
        {
            "case_id": "case-a",
            "source_id": "fresh",
            "question": "Same question",
            "gold_method_id": "ols",
            "correct": False,
        },
        {
            "case_id": "case-b",
            "source_id": "fresh",
            "question": " same   question ",
            "gold_method_id": "ols",
            "correct": False,
        },
    ]
    with pytest.raises(RuntimeError, match="duplicate messages"):
        _template_accounting(
            controls,
            [{**row, "correct": True} for row in controls],
            {"case-a": "template-a", "case-b": "template-b"},
        )


def test_default_e6_gate_requires_template_level_evidence() -> None:
    control = []
    candidate = []
    fallback = []
    for index in range(12):
        question = "Repeated question" if index < 10 else f"Unique question {index}"
        gold = "independent_t" if index < 10 else "wilcoxon_signed_rank"
        base = {
            "case_id": f"invalid-{index}",
            "source_id": "fresh",
            "question": question,
            "gold_method_id": gold,
            "predicted_method_id": None,
            "valid_output": False,
            "correct": False,
        }
        control.append(base)
        fixed = {**base, "predicted_method_id": gold, "valid_output": True, "correct": True}
        candidate.append(fixed)
        fallback.append(fixed)
    for index in range(75):
        row = {
            "case_id": f"valid-{index}",
            "source_id": "fresh",
            "question": f"Valid question {index}",
            "gold_method_id": "ols",
            "predicted_method_id": "ols",
            "valid_output": True,
            "correct": True,
        }
        control.append(row)
        candidate.append(row)
    paired = {
        "candidate_only": 12,
        "control_only": 0,
        "net_improvements": 12,
        "mcnemar_exact_two_sided_p": 0.00048828125,
    }
    templates = _template_accounting(control, candidate)
    gate = _external_gate(
        control=control,
        fallback=fallback,
        candidate=candidate,
        paired=paired,
        templates=templates,
        template_assignments=None,
        opportunity_gates={
            "minimum_invalid_control_opportunities": 12,
            "minimum_invalid_control_gold_methods": 2,
            "minimum_distinct_invalid_question_templates": 8,
            "maximum_single_invalid_template_fraction": 0.25,
            "maximum_single_invalid_method_case_fraction": 0.5,
            "maximum_single_invalid_method_template_fraction": 0.5,
            "minimum_valid_control_identity_cases": 75,
        },
        result_gates={
            "minimum_candidate_only_gains": 6,
            "minimum_net_improvements": 6,
            "minimum_accuracy_gain_points": 4.0,
            "maximum_case_mcnemar_p": 0.05,
            "minimum_distinct_repaired_question_templates": 6,
            "minimum_template_net_improvements": 6,
            "minimum_repaired_gold_methods": 2,
            "maximum_single_repaired_method_fraction": 0.5,
            "maximum_template_mcnemar_p": 0.05,
            "minimum_valid_fallback_precision": 0.6,
            "maximum_valid_but_wrong_fallbacks": 1,
        },
    )
    assert gate["status"] == "INCONCLUSIVE_OPPORTUNITY"
    assert gate["opportunity_checks"]["minimum_distinct_invalid_question_templates"] is False
    assert gate["opportunity_checks"]["maximum_single_invalid_template_fraction"] is False


def test_runtime_receipts_may_differ_only_by_prompt_and_phase() -> None:
    case = _case(0, question="Question", gold="ols")
    calls = []

    def caller(case_id, _messages, _decoding):
        calls.append(case_id)
        return '{"methods":["ols"],"columns":[]}'

    with pytest.raises(RuntimeError, match="differ by more than"):
        run_catalog_fallback_two_stage(
            [case],
            evaluation_fingerprint="evaluation",
            control_runtime_receipt={
                "adapter_sha256": "same-parent",
                "revision": "control-revision",
                "prompt_sha256": "menu-free",
            },
            fallback_runtime_receipt={
                "adapter_sha256": "same-parent",
                "revision": "different-revision",
                "prompt_sha256": "fixed-catalog",
            },
            control_caller=caller,
            fallback_caller=caller,
        )
    assert calls == []


def test_control_model_is_released_before_first_fallback_call() -> None:
    cases = [
        _case(0, question="Valid control", gold="ols"),
        _case(1, question="Invalid control", gold="independent_t"),
    ]
    state = {"control_active": True, "released": False}

    def control(case_id, _messages, _decoding):
        assert state["control_active"]
        if case_id == "case-0":
            return '{"methods":["ols"],"columns":[]}'
        return '{"methods":["out-of-catalog"],"columns":[]}'

    def release_control():
        state["control_active"] = False
        state["released"] = True

    def fallback(_case_id, _messages, _decoding):
        assert state["released"]
        assert not state["control_active"]
        return '{"methods":["independent_t"],"columns":[]}'

    report = run_catalog_fallback_two_stage(
        cases,
        evaluation_fingerprint="evaluation",
        control_runtime_receipt={
            "adapter_sha256": "same-parent",
            "prompt_sha256": "menu-free",
        },
        fallback_runtime_receipt={
            "adapter_sha256": "same-parent",
            "prompt_sha256": "fixed-catalog",
        },
        control_caller=control,
        fallback_caller=fallback,
        before_fallback=release_control,
        opportunity_gates={
            "minimum_invalid_control_opportunities": 1,
            "minimum_invalid_control_gold_methods": 1,
            "minimum_distinct_invalid_question_templates": 1,
            "maximum_single_invalid_template_fraction": 1.0,
            "maximum_single_invalid_method_case_fraction": 1.0,
            "maximum_single_invalid_method_template_fraction": 1.0,
            "minimum_valid_control_identity_cases": 1,
        },
        result_gates={
            "minimum_candidate_only_gains": 1,
            "minimum_net_improvements": 1,
            "minimum_accuracy_gain_points": 50.0,
            "maximum_case_mcnemar_p": 1.0,
            "minimum_distinct_repaired_question_templates": 1,
            "minimum_template_net_improvements": 1,
            "minimum_repaired_gold_methods": 1,
            "maximum_single_repaired_method_fraction": 1.0,
            "maximum_template_mcnemar_p": 1.0,
            "minimum_valid_fallback_precision": 1.0,
            "maximum_valid_but_wrong_fallbacks": 0,
        },
    )

    assert report["runtime_integrity"]["control_model_release_hook_called"] is True


def test_forbidden_source_answer_fields_never_reach_model_caller() -> None:
    case = {
        **_case(0, question="Question", gold="ols"),
        "source_answer": "private-source-answer",
    }
    calls = []

    def caller(case_id, _messages, _decoding):
        calls.append(case_id)
        return '{"methods":["ols"],"columns":[]}'

    with pytest.raises(RuntimeError, match="forbidden model-adjacent fields"):
        run_catalog_fallback_two_stage(
            [case],
            evaluation_fingerprint="evaluation",
            control_runtime_receipt={
                "adapter_sha256": "same-parent",
                "prompt_sha256": "menu-free",
            },
            fallback_runtime_receipt={
                "adapter_sha256": "same-parent",
                "prompt_sha256": "fixed-catalog",
            },
            control_caller=caller,
            fallback_caller=caller,
        )
    assert calls == []


def test_interrupted_control_call_fails_closed_without_retry(tmp_path: Path) -> None:
    case = _case(0, question="Question", gold="ols")

    def interrupted(*_args):
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_catalog_fallback_two_stage(
            [case],
            evaluation_fingerprint="evaluation",
            control_runtime_receipt={
                "adapter_sha256": "same-parent",
                "prompt_sha256": "menu-free",
            },
            fallback_runtime_receipt={
                "adapter_sha256": "same-parent",
                "prompt_sha256": "fixed-catalog",
            },
            control_caller=interrupted,
            fallback_caller=interrupted,
            progress_root=tmp_path,
        )

    retried = []

    def forbidden_retry(case_id, _messages, _decoding):
        retried.append(case_id)
        return '{"methods":["ols"],"columns":[]}'

    with pytest.raises(RuntimeError, match="interrupted model call"):
        run_catalog_fallback_two_stage(
            [case],
            evaluation_fingerprint="evaluation",
            control_runtime_receipt={
                "adapter_sha256": "same-parent",
                "prompt_sha256": "menu-free",
            },
            fallback_runtime_receipt={
                "adapter_sha256": "same-parent",
                "prompt_sha256": "fixed-catalog",
            },
            control_caller=forbidden_retry,
            fallback_caller=forbidden_retry,
            progress_root=tmp_path,
        )
    assert retried == []


def test_interrupted_fallback_fails_closed_without_replaying_either_phase(
    tmp_path: Path,
) -> None:
    cases = [
        _case(0, question="Valid control", gold="ols"),
        _case(1, question="Invalid control", gold="independent_t"),
    ]
    control_calls = []

    def control(case_id, _messages, _decoding):
        control_calls.append(case_id)
        if case_id == "case-0":
            return '{"methods":["ols"],"columns":[]}'
        return '{"methods":["out-of-catalog"],"columns":[]}'

    def interrupted_fallback(*_args):
        raise RuntimeError("simulated fallback interruption")

    receipts = {
        "control_runtime_receipt": {
            "adapter_sha256": "same-parent",
            "prompt_sha256": "menu-free",
        },
        "fallback_runtime_receipt": {
            "adapter_sha256": "same-parent",
            "prompt_sha256": "fixed-catalog",
        },
    }
    with pytest.raises(RuntimeError, match="fallback interruption"):
        run_catalog_fallback_two_stage(
            cases,
            evaluation_fingerprint="evaluation",
            **receipts,
            control_caller=control,
            fallback_caller=interrupted_fallback,
            progress_root=tmp_path,
        )
    assert control_calls == ["case-0", "case-1"]

    replay_calls = []

    def forbidden_replay(case_id, _messages, _decoding):
        replay_calls.append(case_id)
        return '{"methods":["independent_t"],"columns":[]}'

    with pytest.raises(RuntimeError, match="interrupted model call"):
        run_catalog_fallback_two_stage(
            cases,
            evaluation_fingerprint="evaluation",
            **receipts,
            control_caller=forbidden_replay,
            fallback_caller=forbidden_replay,
            progress_root=tmp_path,
        )
    assert replay_calls == []


def test_premature_fallback_ledger_is_rejected_before_any_control_call(
    tmp_path: Path,
) -> None:
    case = _case(0, question="Question", gold="ols")
    write_json(tmp_path / "fallback-ledger.json", {"premature": True})
    calls = []

    def forbidden_call(case_id, _messages, _decoding):
        calls.append(case_id)
        return '{"methods":["ols"],"columns":[]}'

    with pytest.raises(RuntimeError, match="before the control phase"):
        run_catalog_fallback_two_stage(
            [case],
            evaluation_fingerprint="evaluation",
            control_runtime_receipt={
                "adapter_sha256": "same-parent",
                "prompt_sha256": "menu-free",
            },
            fallback_runtime_receipt={
                "adapter_sha256": "same-parent",
                "prompt_sha256": "fixed-catalog",
            },
            control_caller=forbidden_call,
            fallback_caller=forbidden_call,
            progress_root=tmp_path,
        )
    assert calls == []


def test_interrupted_control_ledger_is_rejected_before_agent_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _FakeE6Config(tmp_path)
    root = config.artifact_root / "catalog-fallback-external-v1"
    root.mkdir(parents=True)
    cases = [_case(0, question="Question", gold="ols")]
    master = {
        "fingerprint": "master",
        "runtime": {
            "fingerprint": "runtime",
            "parent": {"adapter_path": "unused", "adapter_sha256": "same-parent"},
            "control_prompt_sha256": "menu-free",
            "fallback_prompt_sha256": "fixed-catalog",
        },
        "opportunity_gates": {},
        "result_gates": {},
        "claim_boundary": "single-source-only",
    }
    child = {"fingerprint": "child"}
    data = {"data_fingerprint": "data"}
    write_json(root / "master-contract.json", master)
    write_json(root / "selected-source-contract.json", child)
    write_json(root / "selected-source-data.json", data)
    evaluation_fingerprint = canonical_hash(
        {
            "master": master["fingerprint"],
            "child": child["fingerprint"],
            "data": data["data_fingerprint"],
            "runtime": master["runtime"]["fingerprint"],
            "evaluator_version": e6._EVALUATOR_VERSION,
        }
    )
    control_receipt, fallback_receipt = e6._runtime_phase_receipts(master)

    def interrupted(*_args):
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_catalog_fallback_two_stage(
            cases,
            evaluation_fingerprint=evaluation_fingerprint,
            control_runtime_receipt=control_receipt,
            fallback_runtime_receipt=fallback_receipt,
            control_caller=interrupted,
            fallback_caller=interrupted,
            progress_root=root / "progress",
        )

    constructed = []

    class ForbiddenAgentCaller:
        def __init__(self, *_args, **_kwargs):
            constructed.append(True)
            raise AssertionError("agent caller must not be constructed")

    monkeypatch.setattr(e6, "_verify_master", lambda _master: None)
    monkeypatch.setattr(e6, "_verify_e6_child_contract", lambda _child, _master: None)
    monkeypatch.setattr(
        e6,
        "_verify_selected_data",
        lambda *_args, **_kwargs: cases,
    )
    monkeypatch.setattr(
        e6,
        "_verify_semantic_template_manifest",
        lambda *_args, **_kwargs: {"case-0": "template"},
    )
    monkeypatch.setattr(e6, "_AgentCaller", ForbiddenAgentCaller)

    with pytest.raises(RuntimeError, match="interrupted model call"):
        e6.run_catalog_fallback_external_evaluation(config)
    assert constructed == []


def test_semantic_template_manifest_is_frozen_before_evaluation(tmp_path: Path) -> None:
    records = [
        {
            "record_id": "0",
            "question_text": "Compare two independent groups.",
            "decision": {"design": "two independent groups", "outcome": "continuous"},
            "answer": "Independent t test",
        },
        {
            "record_id": "1",
            "question_text": "Compare means in unrelated cohorts.",
            "decision": {"design": "two independent groups", "outcome": "continuous"},
            "answer": "Independent t test",
        },
        {
            "record_id": "2",
            "question_text": "Fit a linear outcome model.",
            "decision": {"design": "linear model", "outcome": "continuous"},
            "answer": "Ordinary least squares",
        },
    ]
    snapshot_path = tmp_path / "source.json"
    write_json(snapshot_path, records)
    template_contract = {
        "primary_inferential_unit": "semantic-decision-template",
        "cluster_key": "frozen-normalized-complete-frame-tuple",
        "assignment_timing": "before-model-output",
        "normalization": "recursive-casefold-whitespace-v1",
        "template_id_rule": "canonical-sha256-of-normalized-decision-frame",
        "question_materialization_rule": "normalized-question-field-only",
        "source_projection_frozen_in_child_extraction_contract": True,
        "exact_duplicate_messages_count_once": True,
        "semantic_paraphrases_with_same_decision_frame_count_once": True,
        "representative_rule": "lowest-sha256-case-id",
        "model_output_fields_used": [],
        "outcome_aware_reclustering": False,
    }
    master = {"source_protocol": {"semantic_template_contract": template_contract}}
    projection = {
        "snapshot_format": "json-array",
        "records_path": "",
        "record_id_field": "record_id",
        "case_id_field": "case_id",
        "question_field": "question_text",
        "decision_frame_fields": ["decision.design", "decision.outcome"],
        "decision_frame_roles": {
            "decision.design": "design",
            "decision.outcome": "outcome_type",
        },
        "decision_frame_allowed_values": {
            "decision.design": ["two independent groups", "linear model"],
            "decision.outcome": ["continuous"],
        },
        "answer_fields": ["answer"],
        "source_method_field": "answer",
        "mapped_method_field": "mapped_method_id",
        "record_filter": "all-nonempty-source-method-records",
        "case_id_rule": "sha256-source-id-and-record-id",
        "normalization": "recursive-casefold-whitespace-v1",
        "template_id_rule": "canonical-sha256-of-normalized-decision-frame",
        "question_materialization_rule": "normalized-question-field-only",
        "answer_in_question_policy": "reject-source-label-or-canonical-method-verbatim",
    }
    child = {
        "semantic_template_contract": template_contract,
        "semantic_template_contract_fingerprint": canonical_hash(template_contract),
        "source": {
            "source_id": "fresh-source",
            "snapshot_path": str(snapshot_path),
            "alias_map": {
                "Independent t test": "independent_t",
                "Ordinary least squares": "ols",
            },
            "extraction_contract": {
                "e6_projection": projection,
            },
        },
    }
    frame = e6._recompute_complete_frame(child)
    frame_path = tmp_path / "complete-frame.jsonl"
    write_jsonl(frame_path, frame)
    cases = [
        {
            "case_id": row["case_id"],
            "source_id": "fresh-source",
            "question": row["question"],
            "gold_method_id": row["mapped_method_id"],
        }
        for row in frame
        if row["mapped_method_id"] is not None
    ]
    template_ids = {
        row["case_id"]: canonical_hash(e6._normalize_frame_value(row["decision_frame"]))
        for row in frame
        if row["case_id"] is not None
    }
    clusters = {
        template_id: sorted(
            case_id for case_id, observed in template_ids.items() if observed == template_id
        )
        for template_id in sorted(set(template_ids.values()))
    }
    manifest = []
    for template_id, case_ids in clusters.items():
        representative = min(case_ids, key=lambda case_id: (sha256_text(case_id), case_id))
        for case_id in case_ids:
            manifest.append(
                {
                    "case_id": case_id,
                    "semantic_template_id": template_id,
                    "normalized_frame_fingerprint": template_id,
                    "representative_case_id": representative,
                }
            )
    manifest_path = tmp_path / "semantic-templates.jsonl"
    write_jsonl(manifest_path, manifest)
    assignments = template_ids
    data = {
        "semantic_template_contract_fingerprint": canonical_hash(template_contract),
        "semantic_template_manifest_path": str(manifest_path),
        "semantic_template_manifest_sha256": sha256_file(manifest_path),
        "semantic_template_manifest_fingerprint": canonical_hash(manifest),
        "semantic_template_assignment_fingerprint": canonical_hash(assignments),
        "complete_frame_path": str(frame_path),
        "complete_frame_sha256": sha256_file(frame_path),
        "complete_frame_fingerprint": canonical_hash(frame),
    }

    observed = e6._verify_semantic_template_manifest(
        data,
        child,
        master,
        cases,
        selected_root=tmp_path,
    )

    assert observed == assignments

    contaminated_cases = [dict(case) for case in cases]
    contaminated_cases[0]["question"] += " Answer: independent_t"
    with pytest.raises(RuntimeError, match="question-only field"):
        e6._verify_semantic_template_manifest(
            data,
            child,
            master,
            contaminated_cases,
            selected_root=tmp_path,
        )

    forged_frame = [dict(row) for row in frame]
    forged_frame[0]["decision_frame"] = {"decision": {"design": "forged"}}
    write_jsonl(frame_path, forged_frame)
    data["complete_frame_sha256"] = sha256_file(frame_path)
    data["complete_frame_fingerprint"] = canonical_hash(forged_frame)
    with pytest.raises(RuntimeError, match="differs from the frozen snapshot extraction"):
        e6._verify_semantic_template_manifest(
            data,
            child,
            master,
            cases,
            selected_root=tmp_path,
        )
    write_jsonl(frame_path, frame)
    data["complete_frame_sha256"] = sha256_file(frame_path)
    data["complete_frame_fingerprint"] = canonical_hash(frame)

    forged_manifest = [dict(row) for row in manifest]
    forged_template = sha256_text("forged-distinct-template")
    forged_manifest[0]["semantic_template_id"] = forged_template
    forged_manifest[0]["normalized_frame_fingerprint"] = forged_template
    write_jsonl(manifest_path, forged_manifest)
    forged_assignments = dict(assignments)
    forged_assignments[forged_manifest[0]["case_id"]] = forged_template
    data["semantic_template_manifest_sha256"] = sha256_file(manifest_path)
    data["semantic_template_manifest_fingerprint"] = canonical_hash(forged_manifest)
    data["semantic_template_assignment_fingerprint"] = canonical_hash(forged_assignments)
    with pytest.raises(RuntimeError, match="not derived from the frozen complete frame"):
        e6._verify_semantic_template_manifest(
            data,
            child,
            master,
            cases,
            selected_root=tmp_path,
        )


def test_source_question_containing_answer_is_rejected_before_model_call(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "source.json"
    write_json(
        snapshot_path,
        [
            {
                "record_id": "0",
                "question_text": "The answer is Independent t test.",
                "decision": {"design": "two independent groups"},
                "answer": "Independent t test",
            }
        ],
    )
    child = {
        "source": {
            "source_id": "fresh-source",
            "snapshot_path": str(snapshot_path),
            "alias_map": {"Independent t test": "independent_t"},
            "extraction_contract": {
                "e6_projection": {
                    "snapshot_format": "json-array",
                    "records_path": "",
                    "record_id_field": "record_id",
                    "case_id_field": "case_id",
                    "question_field": "question_text",
                    "decision_frame_fields": ["decision.design"],
                    "decision_frame_roles": {"decision.design": "design"},
                    "decision_frame_allowed_values": {
                        "decision.design": ["two independent groups"]
                    },
                    "answer_fields": ["answer"],
                    "source_method_field": "answer",
                    "mapped_method_field": "mapped_method_id",
                    "record_filter": "all-nonempty-source-method-records",
                    "case_id_rule": "sha256-source-id-and-record-id",
                    "normalization": "recursive-casefold-whitespace-v1",
                    "template_id_rule": "canonical-sha256-of-normalized-decision-frame",
                    "question_materialization_rule": "normalized-question-field-only",
                    "answer_in_question_policy": (
                        "reject-source-label-or-canonical-method-verbatim"
                    ),
                }
            },
        }
    }

    with pytest.raises(RuntimeError, match="contains its answer or gold method"):
        e6._recompute_complete_frame(child)


def test_source_unavailable_rejects_before_agent_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    contract_root = artifact_root / "catalog-fallback-external-v1"
    contract_root.mkdir(parents=True)
    write_json(contract_root / "master-contract.json", {"fingerprint": "sealed-master"})

    class FakeConfig:
        root = tmp_path

        def path_for(self, key: str) -> Path:
            assert key == "artifact_dir"
            return artifact_root

    class ForbiddenAgentCaller:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("model caller must not be constructed")

    monkeypatch.setattr(e6, "_verify_master", lambda _master: None)
    monkeypatch.setattr(e6, "_AgentCaller", ForbiddenAgentCaller)

    with pytest.raises(RuntimeError, match="no preregistered qualified source"):
        e6.run_catalog_fallback_external_evaluation(FakeConfig())


def test_terminal_readback_rederives_result_from_sealed_ledgers(tmp_path: Path) -> None:
    report, master, child, data, cases, evaluation_fingerprint, root = _terminal_fixture(tmp_path)

    e6._verify_terminal_artifacts(
        report,
        root=root,
        master=master,
        child=child,
        data=data,
        cases=cases,
        evaluation_fingerprint=evaluation_fingerprint,
    )

    report["private_details"]["fallback"][0]["raw_answer_sha256"] = "resigned-tamper"
    report["private_details_fingerprint"] = canonical_hash(report["private_details"])
    report["result_fingerprint"] = e6._terminal_fingerprint(report)
    with pytest.raises(RuntimeError, match="sealed ledger"):
        e6._verify_terminal_artifacts(
            report,
            root=root,
            master=master,
            child=child,
            data=data,
            cases=cases,
            evaluation_fingerprint=evaluation_fingerprint,
        )


def test_terminal_readback_rejects_resigned_governance_tampering(tmp_path: Path) -> None:
    report, master, child, data, cases, evaluation_fingerprint, root = _terminal_fixture(tmp_path)
    report["training_authorized"] = True
    report["result_fingerprint"] = e6._terminal_fingerprint(report)

    with pytest.raises(RuntimeError, match="governance field changed"):
        e6._verify_terminal_artifacts(
            report,
            root=root,
            master=master,
            child=child,
            data=data,
            cases=cases,
            evaluation_fingerprint=evaluation_fingerprint,
        )


def test_existing_terminal_requires_matching_opening_receipt(tmp_path: Path) -> None:
    report, master, child, data, cases, evaluation_fingerprint, root = _terminal_fixture(tmp_path)
    opening_path = root / "evaluation-opened.json"
    opening = json.loads(opening_path.read_text(encoding="utf-8"))
    opening["training_authorized"] = True
    opening["fingerprint"] = canonical_hash(
        {key: value for key, value in opening.items() if key != "fingerprint"}
    )
    write_json(opening_path, opening)
    report["opening_sha256"] = sha256_file(opening_path)
    report["opening_fingerprint"] = opening["fingerprint"]
    report["result_fingerprint"] = e6._terminal_fingerprint(report)

    with pytest.raises(RuntimeError, match="opening receipt changed"):
        e6._verify_terminal_artifacts(
            report,
            root=root,
            master=master,
            child=child,
            data=data,
            cases=cases,
            evaluation_fingerprint=evaluation_fingerprint,
        )


class _FakeE6Config:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.artifact_root = root / "artifacts"
        self.evolution_root = root / "data" / "evolve"

    def path_for(self, key: str) -> Path:
        if key == "artifact_dir":
            return self.artifact_root
        if key == "evolution_dir":
            return self.evolution_root
        raise AssertionError(key)


def _fake_e6_master() -> dict:
    template_contract = {
        "primary_inferential_unit": "semantic-decision-template",
        "cluster_key": "frozen-normalized-complete-frame-tuple",
        "assignment_timing": "before-model-output",
        "normalization": "recursive-casefold-whitespace-v1",
        "template_id_rule": "canonical-sha256-of-normalized-decision-frame",
        "question_materialization_rule": "normalized-question-field-only",
        "source_projection_frozen_in_child_extraction_contract": True,
        "exact_duplicate_messages_count_once": True,
        "semantic_paraphrases_with_same_decision_frame_count_once": True,
        "representative_rule": "lowest-sha256-case-id",
        "model_output_fields_used": [],
        "outcome_aware_reclustering": False,
    }
    return {
        "fingerprint": "fake-master",
        "runtime": {"fingerprint": "fake-runtime"},
        "implementation_sha256": {},
        "source_protocol": {
            "permissive_license_allowlist": list(e6._PERMISSIVE_LICENSE_ALLOWLIST),
            "opened_source_exclusions": {},
            "qualification_gates": dict(e6._SOURCE_GATES),
            "semantic_template_contract": template_contract,
        },
    }


def _metadata_bundle(
    config: _FakeE6Config,
    *,
    snapshot_sha256: str = "a" * 64,
    snapshot_url: str = "https://example.invalid/fresh.json",
    historical_hashes: list[str] | None = None,
) -> tuple[Path, dict]:
    metadata_root = config.evolution_root / "catalog-fallback-external-v1" / "metadata-only"
    metadata_root.mkdir(parents=True)
    receipts = []
    for index, content_class in enumerate(
        ["dataset-card-metadata", "schema-metadata", "license-metadata"]
    ):
        response_path = metadata_root / f"receipt-{index}.json"
        write_json(response_path, {"metadata": index})
        receipts.append(
            {
                "content_class": content_class,
                "request_uri": f"https://example.invalid/metadata/{index}",
                "retrieved_at": "2026-08-31T00:00:00Z",
                "response_path": response_path.name,
                "response_sha256": sha256_file(response_path),
                "dataset_rows_opened": False,
            }
        )
    historical = sorted(
        historical_hashes
        if historical_hashes is not None
        else [sha256_text("unrelated historical question")]
    )
    historical_path = metadata_root / "historical-question-hashes.json"
    write_json(historical_path, historical)
    labels = [f"Label {index}" for index in range(8)]
    aliases = {
        label: method_id for label, method_id in zip(labels, e6._METHOD_IDS[:8], strict=True)
    }
    projection = {
        "snapshot_format": "json-array",
        "records_path": "",
        "record_id_field": "record_id",
        "case_id_field": "case_id",
        "question_field": "question_text",
        "decision_frame_fields": ["design"],
        "decision_frame_roles": {"design": "design"},
        "decision_frame_allowed_values": {"design": [f"design-{index}" for index in range(8)]},
        "answer_fields": ["answer"],
        "source_method_field": "answer",
        "mapped_method_field": "mapped_method_id",
        "record_filter": "all-nonempty-source-method-records",
        "case_id_rule": "sha256-source-id-and-record-id",
        "normalization": "recursive-casefold-whitespace-v1",
        "template_id_rule": "canonical-sha256-of-normalized-decision-frame",
        "question_materialization_rule": "normalized-question-field-only",
        "answer_in_question_policy": "reject-source-label-or-canonical-method-verbatim",
    }
    bundle = {
        "schema_version": 1,
        "complete": True,
        "metadata_only": True,
        "dataset_rows_opened": False,
        "source": {
            "source_id": "fresh-source",
            "stable_id": "fresh-source-stable",
            "revision": "revision-1",
            "license": "CC-BY-4.0",
            "license_url": "https://example.invalid/license",
            "snapshot_url": snapshot_url,
            "snapshot_sha256": snapshot_sha256,
            "snapshot_format": "json-array",
            "declared_record_count": 200,
            "method_bearing_record_count": 200,
            "source_method_counts": {label: 25 for label in labels},
            "schema_fields": ["record_id", "question_text", "design", "answer"],
            "alias_map": aliases,
            "e6_projection": projection,
        },
        "receipts": receipts,
        "historical_overlap": {
            "normalization": "casefold-whitespace-sha256-v1",
            "historical_question_hashes_path": historical_path.name,
            "historical_question_hashes_sha256": sha256_file(historical_path),
            "historical_corpus_fingerprint": canonical_hash(historical),
        },
    }
    bundle_path = metadata_root / "qualification-bundle.json"
    write_json(bundle_path, bundle)
    return bundle_path, bundle


def test_metadata_only_child_freezes_without_opening_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _FakeE6Config(tmp_path)
    bundle_path, _bundle = _metadata_bundle(config)
    monkeypatch.setattr(
        e6, "prepare_catalog_fallback_external_master_contract", lambda _c: _fake_e6_master()
    )

    public = e6.prepare_catalog_fallback_external_child_contract(
        config,
        metadata_bundle_path=bundle_path,
    )

    child_path = (
        config.artifact_root / "catalog-fallback-external-v1" / "selected-source-contract.json"
    )
    child = json.loads(child_path.read_text(encoding="utf-8"))
    assert public["dataset_rows_opened"] is False
    assert child["metadata_prequalified"] is True
    assert child["dataset_rows_opened"] is False
    assert child["metadata_qualification_receipt"]["checks"] == {
        "minimum_eligible_cases": True,
        "minimum_distinct_methods": True,
        "minimum_coverage_fraction": True,
        "maximum_single_method_fraction": True,
    }
    assert not Path(child["source"]["snapshot_path"]).exists()
    assert not (child_path.parent / "source-opening.json").exists()


def test_record_identifier_cannot_be_used_as_a_semantic_decision_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _FakeE6Config(tmp_path)
    bundle_path, _bundle = _metadata_bundle(config)
    master = _fake_e6_master()
    monkeypatch.setattr(
        e6,
        "prepare_catalog_fallback_external_master_contract",
        lambda _c: master,
    )
    e6.prepare_catalog_fallback_external_child_contract(config, metadata_bundle_path=bundle_path)
    child_path = (
        config.artifact_root / "catalog-fallback-external-v1" / "selected-source-contract.json"
    )
    child = json.loads(child_path.read_text(encoding="utf-8"))
    projection = child["source"]["extraction_contract"]["e6_projection"]
    projection["decision_frame_fields"] = ["record_id"]
    projection["decision_frame_roles"] = {"record_id": "design"}
    projection["decision_frame_allowed_values"] = {"record_id": ["0"]}
    extraction = child["source"]["extraction_contract"]
    child["source"]["extraction_contract_fingerprint"] = canonical_hash(extraction)
    child["fingerprint"] = canonical_hash(
        {key: value for key, value in child.items() if key != "fingerprint"}
    )

    with pytest.raises(RuntimeError, match="decision frame includes"):
        e6._verify_e6_child_contract(child, master)


@pytest.mark.parametrize(
    "declared,method_bearing,counts,aliases",
    [
        (
            149,
            149,
            {f"L{i}": 19 if i < 5 else 18 for i in range(8)},
            {f"L{i}": e6._METHOD_IDS[i] for i in range(8)},
        ),
        (154, 154, {f"L{i}": 22 for i in range(7)}, {f"L{i}": e6._METHOD_IDS[i] for i in range(7)}),
        (
            200,
            200,
            {**{f"L{i}": 19 for i in range(8)}, "unknown": 48},
            {f"L{i}": e6._METHOD_IDS[i] for i in range(8)},
        ),
        (
            200,
            200,
            {"L0": 90, **{f"L{i}": 15 for i in range(1, 7)}, "L7": 20},
            {f"L{i}": e6._METHOD_IDS[i] for i in range(8)},
        ),
    ],
)
def test_metadata_child_never_lowers_source_gates(
    declared: int,
    method_bearing: int,
    counts: dict[str, int],
    aliases: dict[str, str],
) -> None:
    source = {
        "e6_projection": {
            "record_id_field": "record_id",
            "question_field": "question",
            "source_method_field": "answer",
            "decision_frame_fields": ["decision"],
            "answer_fields": ["answer"],
        },
        "alias_map": aliases,
        "source_method_counts": counts,
        "schema_fields": ["record_id", "question", "answer", "decision"],
        "method_bearing_record_count": method_bearing,
        "declared_record_count": declared,
    }
    with pytest.raises(RuntimeError, match="cannot freeze a qualified child"):
        e6._metadata_source_qualification(source)


def test_metadata_receipt_hash_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _FakeE6Config(tmp_path)
    bundle_path, bundle = _metadata_bundle(config)
    bundle["receipts"][0]["response_sha256"] = "0" * 64
    write_json(bundle_path, bundle)
    monkeypatch.setattr(
        e6, "prepare_catalog_fallback_external_master_contract", lambda _c: _fake_e6_master()
    )

    with pytest.raises(RuntimeError, match="metadata response bytes changed"):
        e6.prepare_catalog_fallback_external_child_contract(
            config,
            metadata_bundle_path=bundle_path,
        )


@pytest.mark.parametrize("drift", ["revision", "license", "schema", "snapshot_sha256"])
def test_selected_child_is_immutable_after_metadata_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    config = _FakeE6Config(tmp_path)
    bundle_path, bundle = _metadata_bundle(config)
    monkeypatch.setattr(
        e6, "prepare_catalog_fallback_external_master_contract", lambda _c: _fake_e6_master()
    )
    e6.prepare_catalog_fallback_external_child_contract(config, metadata_bundle_path=bundle_path)
    if drift == "revision":
        bundle["source"]["revision"] = "revision-2"
    elif drift == "license":
        bundle["source"]["license"] = "MIT"
    elif drift == "schema":
        bundle["source"]["e6_projection"]["question_field"] = "prompt"
        bundle["source"]["schema_fields"].append("prompt")
    else:
        bundle["source"]["snapshot_sha256"] = "b" * 64
    write_json(bundle_path, bundle)

    with pytest.raises(RuntimeError, match="selected-source contract is immutable"):
        e6.prepare_catalog_fallback_external_child_contract(
            config,
            metadata_bundle_path=bundle_path,
        )


def _qualified_snapshot(path: Path) -> list[dict]:
    records = []
    for index in range(200):
        method_index = index % 8
        records.append(
            {
                "record_id": str(index),
                "question_text": f"Choose a procedure for scenario {index}.",
                "design": f"design-{method_index}",
                "answer": f"Label {method_index}",
            }
        )
    write_json(path, records)
    return records


def test_selected_data_is_built_after_child_freeze_and_before_model_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _FakeE6Config(tmp_path)
    staged_snapshot = tmp_path / "staged-source.json"
    _qualified_snapshot(staged_snapshot)
    bundle_path, _bundle = _metadata_bundle(
        config,
        snapshot_sha256=sha256_file(staged_snapshot),
        snapshot_url=staged_snapshot.as_uri(),
    )
    master = _fake_e6_master()
    monkeypatch.setattr(e6, "prepare_catalog_fallback_external_master_contract", lambda _c: master)
    e6.prepare_catalog_fallback_external_child_contract(config, metadata_bundle_path=bundle_path)

    public = e6.prepare_catalog_fallback_external_selected_data(config)

    root = config.artifact_root / "catalog-fallback-external-v1"
    data = json.loads((root / "selected-source-data.json").read_text(encoding="utf-8"))
    assert public["status"] == "QUALIFIED_SOURCE_DATA_FROZEN"
    assert public["case_count"] == 200
    assert public["dataset_rows_opened"] is True
    assert public["model_output_opened"] is False
    assert data["evaluation_authorized"] is True
    assert data["qualification"]["checks"] == {
        "minimum_eligible_cases": True,
        "minimum_distinct_methods": True,
        "minimum_coverage_fraction": True,
        "maximum_single_method_fraction": True,
    }
    assert (root / "source-opening.json").is_file()
    assert (root / "source-opened.json").is_file()
    assert not (root / "evaluation-opened.json").exists()
    assert not (root / "terminal-state.json").exists()


def test_post_open_overlap_closes_source_without_model_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _FakeE6Config(tmp_path)
    staged_snapshot = tmp_path / "staged-source.json"
    _qualified_snapshot(staged_snapshot)
    overlap_hash = sha256_text("choose a procedure for scenario 0.")
    bundle_path, _bundle = _metadata_bundle(
        config,
        snapshot_sha256=sha256_file(staged_snapshot),
        snapshot_url=staged_snapshot.as_uri(),
        historical_hashes=[overlap_hash],
    )
    monkeypatch.setattr(
        e6, "prepare_catalog_fallback_external_master_contract", lambda _c: _fake_e6_master()
    )
    e6.prepare_catalog_fallback_external_child_contract(config, metadata_bundle_path=bundle_path)

    terminal = e6.prepare_catalog_fallback_external_selected_data(config)

    assert terminal["status"] == "SOURCE_UNQUALIFIED"
    assert terminal["dataset_rows_opened"] is True
    assert terminal["model_output_opened"] is False
    assert "historical-question-overlap" in terminal["reason"]
    root = config.artifact_root / "catalog-fallback-external-v1"
    assert not (root / "selected-source-data.json").exists()
    assert not (root / "evaluation-opened.json").exists()


def test_frozen_snapshot_sha_drift_closes_protocol_before_model_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _FakeE6Config(tmp_path)
    staged_snapshot = tmp_path / "staged-source.json"
    records = _qualified_snapshot(staged_snapshot)
    bundle_path, _bundle = _metadata_bundle(
        config,
        snapshot_sha256=sha256_file(staged_snapshot),
        snapshot_url=staged_snapshot.as_uri(),
    )
    monkeypatch.setattr(
        e6,
        "prepare_catalog_fallback_external_master_contract",
        lambda _c: _fake_e6_master(),
    )
    e6.prepare_catalog_fallback_external_child_contract(config, metadata_bundle_path=bundle_path)
    records[0]["question_text"] = "Drifted after metadata freeze."
    write_json(staged_snapshot, records)

    with pytest.raises(RuntimeError, match="frozen SHA-256"):
        e6.prepare_catalog_fallback_external_selected_data(config)

    root = config.artifact_root / "catalog-fallback-external-v1"
    terminal = json.loads((root / "terminal-state.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "PROTOCOL_INVALID"
    assert terminal["model_output_opened"] is False
    assert not (root / "evaluation-opened.json").exists()


def test_interrupted_source_opening_is_never_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _FakeE6Config(tmp_path)
    bundle_path, _bundle = _metadata_bundle(config)
    monkeypatch.setattr(
        e6,
        "prepare_catalog_fallback_external_master_contract",
        lambda _c: _fake_e6_master(),
    )
    e6.prepare_catalog_fallback_external_child_contract(config, metadata_bundle_path=bundle_path)
    root = config.artifact_root / "catalog-fallback-external-v1"
    write_json(root / "source-opening.json", {"interrupted": True})

    with pytest.raises(RuntimeError, match="cannot be retried"):
        e6.prepare_catalog_fallback_external_selected_data(config)

    terminal = json.loads((root / "terminal-state.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "PROTOCOL_INVALID"
    assert terminal["model_output_opened"] is False

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from charlie_alpha.config import load_config
from charlie_alpha.io_utils import canonical_hash
from charlie_alpha.stats_guarded_external import _messages as _menu_free_messages
from charlie_alpha.stats_invalid_control_catalog_fallback import (
    _apply_catalog_guard,
    _evaluate_fallback,
    _fallback_messages,
    _opportunity_summary,
    _result_gate,
    _template_summary,
    run_invalid_control_catalog_fallback,
)

_ROOT = Path(__file__).resolve().parents[1]


def _control(
    case_id: str,
    *,
    source: str,
    question: str,
    gold: str,
    valid: bool,
    correct: bool,
) -> dict:
    return {
        "case_id": case_id,
        "source_id": source,
        "question": question,
        "gold_method_id": gold,
        "predicted_method_id": gold if correct else None,
        "valid_output": valid,
        "correct": correct,
        "parse_reason": "valid-canonical-method" if valid else "unknown-method-id",
        "raw_answer_sha256": f"control-{case_id}",
        "request_hash": f"request-{case_id}",
        "messages_sha256": canonical_hash(_menu_free_messages({"question": question})),
    }


def _fallback(control: dict, *, predicted: str | None) -> dict:
    valid = predicted is not None
    return {
        "case_id": control["case_id"],
        "source_id": control["source_id"],
        "question": control["question"],
        "gold_method_id": control["gold_method_id"],
        "predicted_method_id": predicted,
        "valid_output": valid,
        "correct": valid and predicted == control["gold_method_id"],
        "parse_reason": "valid-canonical-method" if valid else "unknown-method-id",
        "raw_answer_sha256": f"fallback-{control['case_id']}",
        "request_hash": f"fallback-request-{control['case_id']}",
        "messages_sha256": canonical_hash(_fallback_messages(control)),
    }


def test_catalog_fallback_adds_only_the_fixed_global_catalog() -> None:
    case = {"question": "A normally distributed outcome is measured in two independent groups."}
    menu_free = _menu_free_messages(case)
    grounded = _fallback_messages(case)

    assert grounded[1] == menu_free[1]
    assert "Repository method catalog" in grounded[0]["content"]
    assert "independent_t" in grounded[0]["content"]
    assert canonical_hash(grounded) != canonical_hash(menu_free)


def test_catalog_guard_preserves_valid_control_and_rejects_a_forbidden_call() -> None:
    control = _control(
        "valid",
        source="source-a",
        question="Question A",
        gold="ols",
        valid=True,
        correct=True,
    )

    selected = _apply_catalog_guard(control, None)
    assert selected["route"] == "valid-control-identity"
    assert selected["predicted_method_id"] == "ols"

    with pytest.raises(RuntimeError, match="forbidden fallback"):
        _apply_catalog_guard(control, _fallback(control, predicted="ols"))


def test_opportunity_counts_duplicate_questions_as_one_template() -> None:
    cases = [
        {
            "case_id": f"case-{index}",
            "source_id": source,
            "question": question,
            "gold_method_id": gold,
        }
        for index, (source, question, gold) in enumerate(
            [
                ("s1", "Independent groups", "independent_t"),
                ("s2", "Independent groups", "independent_t"),
                ("s3", "Independent groups", "independent_t"),
                ("s1", "One ordinal sample", "wilcoxon_signed_rank"),
            ]
        )
    ]
    summary = _opportunity_summary(cases)

    assert summary["passed"] is True
    assert summary["observed"]["invalid_control_cases"] == 4
    assert summary["observed"]["distinct_invalid_question_templates"] == 2
    assert summary["observed"]["maximum_single_template_fraction"] == 0.75


def test_fallback_progress_replays_without_another_call(tmp_path) -> None:
    cases = [
        {
            "case_id": "case-a",
            "source_id": "source-a",
            "question": "Independent groups",
            "gold_method_id": "independent_t",
        },
        {
            "case_id": "case-b",
            "source_id": "source-b",
            "question": "One ordinal sample",
            "gold_method_id": "wilcoxon_signed_rank",
        },
    ]
    answers = {
        "case-a": '{"methods":["independent_t"],"columns":[]}',
        "case-b": '{"methods":["wilcoxon_signed_rank"],"columns":[]}',
    }
    calls = []

    def caller(case_id, _messages, _decoding):
        calls.append(case_id)
        return answers[case_id]

    runtime = {"adapter_sha256": "parent", "prompt_sha256": "catalog"}
    runtime["fingerprint"] = canonical_hash(runtime)
    progress = tmp_path / "fallback-ledger.json"
    first = _evaluate_fallback(
        cases,
        evaluation_fingerprint="evaluation",
        runtime_receipt=runtime,
        caller=caller,
        progress_path=progress,
    )
    assert calls == ["case-a", "case-b"]
    assert all(row["correct"] for row in first)

    def forbidden_caller(*_args, **_kwargs):
        raise AssertionError("terminal replay called the model")

    second = _evaluate_fallback(
        cases,
        evaluation_fingerprint="evaluation",
        runtime_receipt=runtime,
        caller=forbidden_caller,
        progress_path=progress,
    )
    assert second == first


def test_result_gate_needs_two_repaired_templates_not_three_duplicate_rows() -> None:
    invalid_controls = [
        _control(
            f"invalid-{index}",
            source=source,
            question=question,
            gold=gold,
            valid=False,
            correct=False,
        )
        for index, (source, question, gold) in enumerate(
            [
                ("s1", "Independent groups", "independent_t"),
                ("s2", "Independent groups", "independent_t"),
                ("s3", "Independent groups", "independent_t"),
                ("s1", "One ordinal sample", "wilcoxon_signed_rank"),
            ]
        )
    ]
    valid_controls = [
        _control(
            f"valid-{index}",
            source="s1",
            question=f"Valid question {index}",
            gold="ols",
            valid=True,
            correct=True,
        )
        for index in range(20)
    ]

    one_template_candidate = [
        _apply_catalog_guard(control, _fallback(control, predicted="independent_t"))
        for control in invalid_controls[:3]
    ]
    one_template_candidate.append(
        _apply_catalog_guard(invalid_controls[3], _fallback(invalid_controls[3], predicted=None))
    )
    one_template_candidate.extend(_apply_catalog_guard(control, None) for control in valid_controls)
    one_template_summary = _template_summary(one_template_candidate)
    one_template_paired = {
        "candidate_only": 3,
        "control_only": 0,
        "net_improvements": 3,
    }
    failed = _result_gate(
        paired=one_template_paired,
        candidate=one_template_candidate,
        template_summary=one_template_summary,
    )
    assert failed["passed"] is False
    assert failed["checks"]["minimum_distinct_repaired_question_templates"] is False
    assert failed["checks"]["minimum_repaired_gold_methods"] is False

    all_candidate = deepcopy(one_template_candidate)
    all_candidate[3] = _apply_catalog_guard(
        invalid_controls[3],
        _fallback(invalid_controls[3], predicted="wilcoxon_signed_rank"),
    )
    all_summary = _template_summary(all_candidate)
    passed = _result_gate(
        paired={"candidate_only": 4, "control_only": 0, "net_improvements": 4},
        candidate=all_candidate,
        template_summary=all_summary,
    )
    assert passed["passed"] is True


def test_real_h22_terminal_replay_preserves_internal_evidence() -> None:
    report_path = (
        _ROOT / "artifacts" / "evolve" / "invalid-control-catalog-fallback-v1" / "report.json"
    )
    before_bytes = report_path.read_bytes()
    before_mtime = report_path.stat().st_mtime_ns

    report = run_invalid_control_catalog_fallback(
        load_config(_ROOT / "configs" / "pipeline.evolve.yaml")
    )

    assert report["status"] == "SUPPORTED_DEV_COMPOSITION"
    assert report["result_fingerprint"] == (
        "0c2dc933fdfb04ab5569be8042afdbe55837bb3fb1dd8d19ad12b4959825ee1c"
    )
    assert report_path.read_bytes() == before_bytes
    assert report_path.stat().st_mtime_ns == before_mtime

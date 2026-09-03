from charlie_alpha.stats_semantic_catalog import (
    _gate_report,
    _messages,
    _semantic_catalog_reference,
)


def test_semantic_catalog_uses_existing_static_method_fields() -> None:
    reference = _semantic_catalog_reference()
    assert "welch_t — Welch two-sample t test" in reference
    assert "assumptions: independent observations; finite group variances" in reference
    assert (
        "strengths: valid under unequal variances; good default for independent means" in reference
    )
    assert "uncertainty: Welch-Satterthwaite interval" in reference


def test_semantic_and_flat_prompts_keep_same_question_and_contract() -> None:
    case = {
        "question": "Choose the analysis.",
        "gold_methods": ["welch_t"],
        "gold_columns": ["y", "arm"],
    }
    flat = _messages(case, semantic=False)
    semantic = _messages(case, semantic=True)
    assert flat[1:] == semantic[1:]
    assert "Select exactly one method identifier from this fixed catalog." in flat[0]["content"]
    assert "Select exactly one method identifier from this fixed catalog." in semantic[0]["content"]


def test_semantic_gate_requires_absolute_gain_and_column_retention() -> None:
    control = {
        "format_shift": {
            "exact_accuracy": 0.25,
            "method_set_accuracy": 0.30,
            "column_set_accuracy": 0.90,
        }
    }
    candidate = {
        "format_shift": {
            "exact_accuracy": 0.40,
            "method_set_accuracy": 0.45,
            "column_set_accuracy": 0.90,
        }
    }
    gates = {
        "minimum_semantic_method_accuracy": 0.40,
        "minimum_semantic_exact_accuracy": 0.35,
        "minimum_method_gain_over_flat_points": 10.0,
        "minimum_exact_gain_over_flat_points": 10.0,
        "minimum_column_gain_over_flat_points": -5.0,
    }
    assert _gate_report(control=control, candidate=candidate, gates=gates)["passed"]

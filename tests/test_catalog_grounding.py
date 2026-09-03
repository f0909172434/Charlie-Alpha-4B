from charlie_alpha.stats_catalog import PROCEDURES
from charlie_alpha.stats_catalog_grounding import _catalog_reference, _gate_report, _messages
from charlie_alpha.stats_cross_format import _format_shift_messages


def _case() -> dict[str, object]:
    return {
        "question": "Choose the analysis.",
        "gold_methods": ["welch_t"],
        "gold_columns": ["y", "arm"],
    }


def test_control_prompt_is_unchanged_menu_free_prompt() -> None:
    assert _messages(_case(), grounded=False) == _format_shift_messages(_case())


def test_grounded_prompt_adds_fixed_catalog_only_to_system_message() -> None:
    control = _messages(_case(), grounded=False)
    grounded = _messages(_case(), grounded=True)
    assert grounded[1:] == control[1:]
    assert grounded[0]["content"].startswith(control[0]["content"])
    reference = _catalog_reference()
    assert grounded[0]["content"].count(reference) == 1
    for procedure in PROCEDURES:
        assert f"{procedure.method_id} — {procedure.name}" in reference


def test_gate_requires_exact_method_and_column_conditions() -> None:
    control = {
        "format_shift": {
            "exact_accuracy": 0.0,
            "method_set_accuracy": 0.1,
            "column_set_accuracy": 0.9,
        }
    }
    candidate = {
        "format_shift": {
            "exact_accuracy": 0.25,
            "method_set_accuracy": 0.35,
            "column_set_accuracy": 0.85,
        }
    }
    gates = {
        "minimum_exact_gain_over_control_points": 20.0,
        "minimum_method_gain_over_control_points": 20.0,
        "minimum_column_gain_over_control_points": -5.0,
    }
    assert _gate_report(control=control, candidate=candidate, gates=gates)["passed"]

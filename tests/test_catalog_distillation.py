import json

from charlie_alpha.stats_catalog_distillation import (
    _distillation_gate,
    _grounded_blueprints,
    _training_rows,
)
from charlie_alpha.stats_dgp import Scenario


def _scenario(family_id: str, blueprint_id: str) -> Scenario:
    return Scenario(
        blueprint_id=blueprint_id,
        split="unit",
        family_id=family_id,
        domain="unit",
        seed=1,
        boundary_round=0,
        parameters={},
    )


def test_grounding_dropout_is_exactly_half_per_family() -> None:
    rows = [_scenario(family, f"{family}-{index}") for family in ("a", "b") for index in range(8)]
    grounded = _grounded_blueprints(rows, fraction=0.5)
    assert len(grounded) == 8
    for family in ("a", "b"):
        assert sum(value.startswith(f"{family}-") for value in grounded) == 4


def test_candidate_menu_free_rows_match_control_and_targets_always_match() -> None:
    simulations = [
        {
            "scenario": {
                "blueprint_id": "x",
                "split": "unit",
                "family_id": "group_comparison",
                "domain": "unit",
                "seed": 1,
                "boundary_round": 0,
                "parameters": {},
            },
            "selected_method_id": "welch_t",
        },
        {
            "scenario": {
                "blueprint_id": "y",
                "split": "unit",
                "family_id": "group_comparison",
                "domain": "unit",
                "seed": 2,
                "boundary_round": 0,
                "parameters": {},
            },
            "selected_method_id": "welch_t",
        },
    ]
    # Patch the format-dependent fields to the minimum accepted by the renderer.
    from charlie_alpha import stats_catalog_distillation as module

    original = module._format_shift_case
    module._format_shift_case = lambda simulation: {
        "case_id": simulation["scenario"]["blueprint_id"],
        "family_id": simulation["scenario"]["family_id"],
        "question": "Choose the analysis.",
        "gold_methods": [simulation["selected_method_id"]],
        "gold_columns": ["y", "arm"],
    }
    try:
        control, candidate = _training_rows(simulations, grounded_blueprints={"x"})
    finally:
        module._format_shift_case = original
    assert control[1]["messages"] == candidate[1]["messages"]
    assert control[0]["messages"][:-1] != candidate[0]["messages"][:-1]
    for left, right in zip(control, candidate, strict=True):
        assert json.loads(left["messages"][-1]["content"]) == json.loads(
            right["messages"][-1]["content"]
        )


def test_distillation_gate_requires_h7_effect_retention() -> None:
    parent = {
        "format_shift": {
            "exact_accuracy": 0.0,
            "method_set_accuracy": 0.0,
            "column_set_accuracy": 0.9,
        },
        "selector": {"normalized_regret": 0.5, "accuracy": 0.4, "invalid_selection_rate": 0.4},
    }
    control = {
        "format_shift": {
            "exact_accuracy": 0.0,
            "method_set_accuracy": 0.0,
            "column_set_accuracy": 0.9,
        },
        "selector": parent["selector"],
    }
    candidate = {
        "format_shift": {
            "exact_accuracy": 0.1,
            "method_set_accuracy": 0.1,
            "column_set_accuracy": 0.9,
        },
        "selector": parent["selector"],
    }
    gates = {
        "minimum_exact_gain_over_parent_points": 8.0,
        "minimum_exact_gain_over_control_points": 8.0,
        "minimum_method_gain_over_control_points": 8.0,
        "minimum_column_gain_over_control_points": -5.0,
        "maximum_selector_relative_regret_increase": 0.1,
        "maximum_selector_accuracy_regression_points": 5.0,
        "maximum_invalidity_increase": 0.05,
        "minimum_h7_effect_retention_fraction": 0.4,
    }
    report = _distillation_gate(
        parent=parent,
        control=control,
        candidate=candidate,
        gates=gates,
        h7_effect={"exact_vs_control": 21.6666667, "method_vs_control": 25.0},
    )
    assert report["passed"]

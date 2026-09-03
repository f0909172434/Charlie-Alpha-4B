from charlie_alpha.stats_cross_format import (
    _column_recall,
    _flatten_columns,
    _format_metrics,
    _gate_report,
    _set_match,
)


def test_cross_format_columns_flatten_variable_roles() -> None:
    assert _flatten_columns(
        {"outcome": "y", "predictors": ["x1", "x2"], "cluster": "cluster_id"}
    ) == ["cluster_id", "x1", "x2", "y"]


def test_cross_format_set_metrics_are_order_and_punctuation_insensitive() -> None:
    assert _set_match(["hc3_ols"], ["HC3-OLS"])
    assert _set_match(["y", "x1"], ["X1", "Y"])
    assert _column_recall(["y", "x1"], ["Y"]) == 0.5


def test_cross_format_metrics_and_gate_require_candidate_gain() -> None:
    details = [
        {
            "exact_correct": True,
            "method_correct": True,
            "columns_correct": True,
            "column_recall": 1.0,
        },
        {
            "exact_correct": False,
            "method_correct": True,
            "columns_correct": False,
            "column_recall": 0.5,
        },
    ]
    assert _format_metrics(details) == {
        "count": 2,
        "exact_accuracy": 0.5,
        "method_set_accuracy": 1.0,
        "column_set_accuracy": 0.5,
        "mean_column_recall": 0.75,
    }
    parent = {
        "selector": {"normalized_regret": 0.2, "accuracy": 0.8, "invalid_selection_rate": 0.0},
        "format_shift": {
            "exact_accuracy": 0.2,
            "method_set_accuracy": 0.5,
            "column_set_accuracy": 0.3,
        },
    }
    control = {
        "selector": {"normalized_regret": 0.19, "accuracy": 0.8, "invalid_selection_rate": 0.0},
        "format_shift": {
            "exact_accuracy": 0.22,
            "method_set_accuracy": 0.52,
            "column_set_accuracy": 0.32,
        },
    }
    candidate = {
        "selector": {"normalized_regret": 0.202, "accuracy": 0.79, "invalid_selection_rate": 0.0},
        "format_shift": {
            "exact_accuracy": 0.30,
            "method_set_accuracy": 0.55,
            "column_set_accuracy": 0.40,
        },
    }
    gates = {
        "minimum_exact_gain_over_parent_points": 5.0,
        "minimum_exact_gain_over_control_points": 5.0,
        "minimum_method_gain_over_control_points": 0.0,
        "minimum_column_gain_over_control_points": 0.0,
        "maximum_selector_relative_regret_increase": 0.02,
        "maximum_selector_accuracy_regression_points": 3.0,
        "maximum_invalidity_increase": 0.0,
    }
    assert _gate_report(parent=parent, control=control, candidate=candidate, gates=gates)["passed"]

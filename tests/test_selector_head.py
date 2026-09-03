import pytest

from charlie_alpha.stats_selector_head import _gate_report


def test_selector_head_gate_requires_absolute_and_paired_gains() -> None:
    control = {
        "method_set_accuracy": 0.05,
        "exact_accuracy": 0.02,
        "column_set_accuracy": 0.90,
    }
    candidate = {
        "method_set_accuracy": 0.70,
        "exact_accuracy": 0.62,
        "column_set_accuracy": 0.90,
    }
    gates = {
        "minimum_head_method_accuracy": 0.65,
        "minimum_head_exact_accuracy": 0.55,
        "minimum_method_gain_points": 40.0,
        "minimum_exact_gain_points": 40.0,
        "minimum_column_gain_points": 0.0,
    }
    report = _gate_report(control, candidate, gates)
    assert report["passed"]
    assert report["effect_points"]["method"] == pytest.approx(65.0)
    assert report["effect_points"]["exact"] == pytest.approx(60.0)


def test_selector_head_gate_rejects_column_regression() -> None:
    control = {
        "method_set_accuracy": 0.05,
        "exact_accuracy": 0.02,
        "column_set_accuracy": 0.90,
    }
    candidate = {
        "method_set_accuracy": 0.70,
        "exact_accuracy": 0.62,
        "column_set_accuracy": 0.89,
    }
    gates = {
        "minimum_head_method_accuracy": 0.65,
        "minimum_head_exact_accuracy": 0.55,
        "minimum_method_gain_points": 40.0,
        "minimum_exact_gain_points": 40.0,
        "minimum_column_gain_points": 0.0,
    }
    assert not _gate_report(control, candidate, gates)["passed"]

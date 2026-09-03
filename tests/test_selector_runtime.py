import numpy as np

from charlie_alpha.stats_representation_probe import _METHOD_INDEX
from charlie_alpha.stats_selector_runtime import _normalize_columns, _rank_methods


def test_selector_runtime_normalizes_columns_without_inventing_values() -> None:
    assert _normalize_columns(["y", "group", 3, None, {"x": 1}]) == ["y", "group", "3"]
    assert _normalize_columns("y") == []


def test_selector_runtime_ranks_only_observed_head_methods() -> None:
    independent = _METHOD_INDEX["independent_t"]
    welch = _METHOD_INDEX["welch_t"]
    weights = np.zeros((3, 28), dtype=np.float64)
    weights[0, independent] = 2.0
    weights[1, welch] = 1.0
    head = {"weights": weights, "observed": [independent, welch]}
    method, top3, margin = _rank_methods(head, np.array([[1.0, 0.0]], dtype=np.float64))
    assert method == "independent_t"
    assert top3 == ["independent_t", "welch_t"]
    assert margin > 0

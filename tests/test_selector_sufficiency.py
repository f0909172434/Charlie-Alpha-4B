import numpy as np

from charlie_alpha.stats_selector_sufficiency import (
    _centered_normalize,
    _select_threshold,
    _support_scores,
)


def test_centered_support_scores_keep_near_queries_above_far_queries() -> None:
    bank_vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float64
    )
    center = bank_vectors.mean(axis=0, keepdims=True)
    bank = {
        "center": center,
        "normalized": _centered_normalize(bank_vectors, center),
    }
    scores = _support_scores(
        bank,
        np.array([[0.95, 0.05], [0.0, -2.0]], dtype=np.float64),
    )
    assert scores[0] > scores[1]


def test_threshold_selection_requires_both_retention_and_rejection() -> None:
    reports = [
        {
            "threshold": 0.90,
            "full_acceptance": 0.95,
            "minimum_full_style_acceptance": 0.90,
            "reduced_rejection": 0.70,
            "minimum_reduced_style_rejection": 0.65,
            "accepted_head_accuracy": 0.80,
            "accepted_head_accuracy_change_points": 2.0,
        },
        {
            "threshold": 0.94,
            "full_acceptance": 0.90,
            "minimum_full_style_acceptance": 0.85,
            "reduced_rejection": 0.95,
            "minimum_reduced_style_rejection": 0.90,
            "accepted_head_accuracy": 0.82,
            "accepted_head_accuracy_change_points": 4.0,
        },
    ]
    selected, enriched = _select_threshold(
        reports,
        {
            "minimum_full_acceptance": 0.85,
            "minimum_full_style_acceptance": 0.80,
            "minimum_reduced_rejection": 0.90,
            "minimum_reduced_style_rejection": 0.85,
            "minimum_accepted_head_accuracy": 0.65,
            "maximum_accepted_head_accuracy_regression_points": 3.0,
        },
    )
    assert not enriched[0]["eligible"]
    assert enriched[1]["eligible"]
    assert selected is not None
    assert selected["threshold"] == 0.94

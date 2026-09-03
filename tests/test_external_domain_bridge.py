import numpy as np

from charlie_alpha.stats_external_domain_bridge import (
    _candidate_scores,
    _fit_residual,
    _gate_candidate,
    _selective_metrics,
)


def test_residual_head_learns_a_small_domain_correction() -> None:
    head = {
        "weights": np.zeros((3, 3), dtype=np.float64),
        "observed": [0, 1, 2],
    }
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    labels = np.asarray([0, 1], dtype=np.int64)
    residual = _fit_residual(head, vectors, labels, ridge_lambda=0.01)
    scores = _candidate_scores(head, residual, vectors, residual_scale=1.0)
    assert np.argmax(scores, axis=1).tolist() == labels.tolist()


def test_selective_metrics_count_control_only_losses() -> None:
    metrics = _selective_metrics(
        [
            {
                "source_id": "a",
                "control_correct": True,
                "raw_candidate_correct": False,
                "selective_correct": False,
                "accepted": True,
            },
            {
                "source_id": "b",
                "control_correct": False,
                "raw_candidate_correct": True,
                "selective_correct": True,
                "accepted": True,
            },
        ]
    )
    assert metrics["candidate_only"] == 1
    assert metrics["control_only"] == 1
    assert metrics["net_improvements"] == 0


def test_gate_rejects_one_control_only_loss() -> None:
    historical = {
        "net_improvements": 2,
        "control_only": 1,
        "accept_count": 8,
        "accepted_source_count": 3,
    }
    synthetic = {
        "minimum_full_accuracy": 0.70,
        "maximum_full_accuracy_regression_points": 2.0,
        "minimum_full_acceptance": 0.90,
        "minimum_reduced_rejection": 0.90,
        "minimum_accepted_head_accuracy": 0.70,
    }
    gates = {
        "minimum_historical_net_improvements": 1,
        "maximum_historical_control_only_losses": 0,
        "minimum_historical_accept_count": 4,
        "minimum_historical_accepted_sources": 2,
        "minimum_synthetic_full_accuracy": 0.65,
        "maximum_synthetic_full_accuracy_regression_points": 5.0,
        "minimum_synthetic_full_acceptance": 0.80,
        "minimum_synthetic_reduced_rejection": 0.80,
        "minimum_synthetic_accepted_head_accuracy": 0.60,
    }
    result = _gate_candidate(historical, synthetic, gates)
    assert result["passed"] is False
    assert result["checks"]["zero_historical_control_only_losses"] is False

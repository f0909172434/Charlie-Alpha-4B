import numpy as np

from charlie_alpha.stats_external_exemplar_router import (
    _exemplar_vote,
    _gate,
    _metrics,
)


def test_exemplar_vote_uses_nearest_method() -> None:
    predictions, support, margin = _exemplar_vote(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        np.asarray([2, 5], dtype=np.int64),
        np.asarray([[0.9, 0.1]], dtype=np.float64),
        mode="unit",
        center=np.zeros((1, 2), dtype=np.float64),
        neighbors=1,
        temperature=0.1,
    )
    assert predictions.tolist() == [2]
    assert support[0] > 0.9
    assert margin.tolist() == [1.0]


def test_router_gate_requires_zero_control_only_losses() -> None:
    details = [
        {
            "source_id": "a",
            "control_correct": True,
            "candidate_correct": False,
            "route": "external-exemplar",
        }
    ]
    metrics = _metrics(details)
    safety = {"maximum_reduced_external_exemplar_acceptance": 0.0}
    gates = {
        "minimum_historical_net_improvements": 0,
        "maximum_control_only_losses": 0,
        "minimum_external_exemplar_count": 1,
        "minimum_external_exemplar_sources": 1,
        "maximum_reduced_external_exemplar_acceptance": 0.2,
    }
    result = _gate(metrics, safety, gates)
    assert result["passed"] is False
    assert result["checks"]["zero_control_only_losses"] is False

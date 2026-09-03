import numpy as np

from charlie_alpha.stats_external_representation_diagnostic import _model_report
from charlie_alpha.stats_representation_probe import _METHOD_INDEX, _fit_ridge_probe


def test_model_report_separates_probe_coverage_from_external_denominator() -> None:
    labels = np.array(
        [
            _METHOD_INDEX["independent_t"],
            _METHOD_INDEX["welch_t"],
        ],
        dtype=np.int64,
    )
    model = _fit_ridge_probe(
        np.array([[0.0, 1.0]], dtype=np.float64),
        np.array([_METHOD_INDEX["welch_t"]], dtype=np.int64),
        ridge_lambda=0.01,
    )
    report = _model_report(
        model,
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        labels,
        majority_class=_METHOD_INDEX["independent_t"],
    )
    assert report["covered_external_count"] == 1
    assert report["all_nine"]["count"] == 2
    assert report["covered_only"]["count"] == 1

import numpy as np

from charlie_alpha.stats_representation_probe import (
    _METHOD_INDEX,
    _choose_route,
    _fit_ridge_probe,
    _probe_metrics,
)


def test_ridge_probe_decodes_linearly_separable_seen_classes() -> None:
    labels = np.array(
        [
            _METHOD_INDEX["independent_t"],
            _METHOD_INDEX["independent_t"],
            _METHOD_INDEX["welch_t"],
            _METHOD_INDEX["welch_t"],
        ],
        dtype=np.int64,
    )
    vectors = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=np.float64,
    )
    model = _fit_ridge_probe(vectors, labels, ridge_lambda=0.01)
    metrics = _probe_metrics(
        model,
        vectors,
        labels,
        majority_class=_METHOD_INDEX["independent_t"],
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["top3_accuracy"] == 1.0
    assert metrics["train_observed_class_count"] == 2


def test_route_prefers_menu_free_selector_when_viable() -> None:
    gates = {
        "minimum_menu_free_accuracy": 0.30,
        "minimum_menu_free_top3_accuracy": 0.55,
        "minimum_menu_free_gain_over_majority_points": 10.0,
        "minimum_catalog_accuracy": 0.35,
        "minimum_catalog_gain_over_menu_free_points": 10.0,
    }
    menu = {"accuracy": 0.40, "top3_accuracy": 0.70, "gain_over_majority_points": 20.0}
    catalog = {"accuracy": 0.55, "top3_accuracy": 0.80, "gain_over_majority_points": 30.0}
    route, _ = _choose_route(menu, catalog, gates)
    assert route == "selector-head"


def test_route_uses_contrastive_when_only_catalog_is_linearly_decodable() -> None:
    gates = {
        "minimum_menu_free_accuracy": 0.30,
        "minimum_menu_free_top3_accuracy": 0.55,
        "minimum_menu_free_gain_over_majority_points": 10.0,
        "minimum_catalog_accuracy": 0.35,
        "minimum_catalog_gain_over_menu_free_points": 10.0,
    }
    menu = {"accuracy": 0.20, "top3_accuracy": 0.40, "gain_over_majority_points": 5.0}
    catalog = {"accuracy": 0.40, "top3_accuracy": 0.70, "gain_over_majority_points": 25.0}
    route, _ = _choose_route(menu, catalog, gates)
    assert route == "contrastive-representation"

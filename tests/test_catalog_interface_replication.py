from charlie_alpha.stats_catalog_interface_replication import _gate_report


def _arm(exact: float, method: float, columns: float, count: int = 60) -> dict[str, object]:
    return {
        "format_shift": {
            "count": count,
            "exact_accuracy": exact,
            "method_set_accuracy": method,
            "column_set_accuracy": columns,
        }
    }


def test_replication_gate_requires_pooled_and_multifold_effects() -> None:
    folds = {
        "fold_1": {
            "menu-free-control": _arm(0.00, 0.05, 0.90),
            "flat-catalog": _arm(0.25, 0.30, 0.90),
        },
        "fold_2": {
            "menu-free-control": _arm(0.00, 0.05, 0.90),
            "flat-catalog": _arm(0.20, 0.25, 0.90),
        },
        "fold_3": {
            "menu-free-control": _arm(0.05, 0.10, 0.90),
            "flat-catalog": _arm(0.10, 0.15, 0.90),
        },
    }
    gates = {
        "minimum_pooled_exact_gain_points": 15.0,
        "minimum_pooled_method_gain_points": 15.0,
        "minimum_pooled_column_gain_points": -5.0,
        "minimum_fold_exact_gain_points": 10.0,
        "minimum_fold_method_gain_points": 10.0,
        "minimum_qualifying_folds": 2,
    }
    assert _gate_report(folds=folds, gates=gates)["passed"]

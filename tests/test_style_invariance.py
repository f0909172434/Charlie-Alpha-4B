import numpy as np

from charlie_alpha.stats_dgp import Scenario
from charlie_alpha.stats_style_invariance import _geometry, _selection_route, _style_question


def test_style_renderings_are_distinct_without_method_labels() -> None:
    scenario = Scenario(
        blueprint_id="h15-test",
        family_id="group_comparison",
        split="test",
        seed=1,
        parameters={
            "n": 120.0,
            "variance_ratio": 2.0,
            "pair_correlation": 0.1,
            "tail_weight": 0.2,
            "effect_size": 0.5,
        },
        boundary_round=0,
        domain="inference_and_design",
    )
    questions = [_style_question(scenario, style) for style in ("audit", "researcher", "vignette")]
    assert len(set(questions)) == 3
    assert all("welch_t" not in question for question in questions)
    assert "DGP audit values" in questions[0]
    assert "We are planning" in questions[1]
    assert "Key conditions" in questions[2]


def test_selection_route_detects_recoverable_style_boundary_shift() -> None:
    frozen = {
        "audit": {"accuracy": 0.75},
        "researcher": {"accuracy": 0.35},
        "vignette": {"accuracy": 0.40},
    }
    diverse = {
        "audit": {"accuracy": 0.72},
        "researcher": {"accuracy": 0.65},
        "vignette": {"accuracy": 0.60},
    }
    route, diagnostics = _selection_route(
        frozen,
        diverse,
        {
            "minimum_frozen_audit_accuracy": 0.55,
            "minimum_frozen_all_style_accuracy": 0.55,
            "maximum_stable_style_gap": 0.15,
            "minimum_style_collapse": 0.15,
            "minimum_style_diverse_natural_accuracy": 0.55,
            "minimum_style_diverse_recovery": 0.15,
        },
    )
    assert route == "style-diverse-linear-boundary"
    assert diagnostics["frozen_style_collapse_points"] == 40.0
    assert diagnostics["style_diverse_recovery_points"] == 25.0


def test_geometry_rewards_matched_semantics() -> None:
    base = np.eye(3, dtype=np.float64)
    representations = {
        "audit": (base, np.array([0, 1, 2]), ["a", "b", "c"]),
        "researcher": (base.copy(), np.array([0, 1, 2]), ["a", "b", "c"]),
        "vignette": (base.copy(), np.array([0, 1, 2]), ["a", "b", "c"]),
    }
    report = _geometry(representations)
    assert report["pairs"]["audit__researcher"]["same_semantic_cosine_mean"] == 1.0
    assert report["pairs"]["audit__researcher"]["matched_margin"] == 1.0

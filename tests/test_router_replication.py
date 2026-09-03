from charlie_alpha.io_utils import canonical_hash
from charlie_alpha.stats_router_replication import (
    _extract_scenario,
    _scenario_semantic_payload,
    paired_power_sample_size,
)


def test_paired_power_preregistration_rounds_with_safety_margin() -> None:
    result = paired_power_sample_size(
        paired_sd=0.2337126928058349,
        parent_mean_regret=0.3699334439785275,
        minimum_relative_improvement=0.075,
        alpha=0.05,
        power=0.90,
        safety_margin=0.20,
        allocation_multiple=12,
    )
    assert 745 < result["raw_required_blueprints"] < 747
    assert result["registered_minimum_blueprints"] == 900


def test_semantic_identity_ignores_split_seed_and_blueprint_id() -> None:
    scenario = {
        "blueprint_id": "first",
        "family_id": "categorical",
        "split": "old",
        "seed": 1,
        "domain": "inference_and_design",
        "parameters": {"n": 100.0, "baseline_probability": 0.2},
        "boundary_round": 0,
    }
    changed_identity = {**scenario, "blueprint_id": "second", "split": "new", "seed": 2}
    assert canonical_hash(_scenario_semantic_payload(scenario)) == canonical_hash(
        _scenario_semantic_payload(changed_identity)
    )


def test_extract_scenario_accepts_simulation_wrapper_only_with_required_fields() -> None:
    scenario = {
        "blueprint_id": "dgp-example",
        "family_id": "categorical",
        "domain": "inference_and_design",
        "parameters": {},
        "boundary_round": 0,
    }
    assert _extract_scenario({"scenario": scenario, "oracle": {}}) == scenario
    assert _extract_scenario({"metadata": {"blueprint_id": "dgp-example"}}) is None

from charlie_alpha.stats_router_failure import _replace_family_expert_with_parent


def _score(predictions):
    return {
        "selector": {},
        "languages": {
            "en": {
                "predictions": predictions,
            }
        },
        "retention": {},
    }


def test_leave_one_expert_out_replaces_only_routed_blueprint_ids() -> None:
    row_a = {
        "blueprint_id": "a",
        "normalized_regret": 0.0,
        "valid": True,
        "predicted_method_id": "m",
        "oracle_method_id": "m",
        "domain": "d",
        "family_id": "f",
    }
    parent_b = {**row_a, "blueprint_id": "b", "normalized_regret": 0.1}
    routed_b = {**parent_b, "normalized_regret": 0.9}
    result = _replace_family_expert_with_parent(
        _score([row_a, parent_b]),
        _score([row_a, routed_b]),
        _score([routed_b]),
    )
    by_id = {row["blueprint_id"]: row for row in result["languages"]["en"]["predictions"]}
    assert by_id["a"]["normalized_regret"] == 0.0
    assert by_id["b"]["normalized_regret"] == 0.1

from charlie_alpha.stats_sufficiency_guard import (
    _apply_guard,
    _threshold_guard_metrics,
    sufficiency_prompt,
)


def test_sufficiency_prompt_has_fixed_binary_contract() -> None:
    prompt = sufficiency_prompt()
    assert "A: Sufficient" in prompt
    assert "B: Insufficient" in prompt
    assert prompt.endswith("A or B.")


def test_apply_guard_conservatively_penalizes_false_positive() -> None:
    row = {
        "blueprint_id": "a",
        "family_id": "f",
        "domain": "d",
        "predicted_method_id": "m",
        "oracle_method_id": "m",
        "normalized_regret": 0.0,
        "valid": True,
    }
    score = {
        "selector": {},
        "languages": {"en": {"predictions": [row]}},
        "retention": {},
    }
    guarded = _apply_guard(score, {("en", "a"): True})
    prediction = guarded["languages"]["en"]["predictions"][0]
    assert prediction["predicted_method_id"] == "needs_clarification"
    assert prediction["normalized_regret"] == 1.0
    assert prediction["valid"] is False


def test_threshold_metrics_expose_sensitivity_specificity_tradeoff() -> None:
    rows = [
        {
            "incomplete": False,
            "language": "en",
            "family_id": "f",
            "insufficient_probability": 0.8,
        },
        {
            "incomplete": True,
            "language": "en",
            "family_id": "f",
            "insufficient_probability": 0.99,
        },
    ]
    low = _threshold_guard_metrics(rows, threshold=0.5)
    high = _threshold_guard_metrics(rows, threshold=0.9)
    assert low["complete_specificity"]["accuracy"] == 0.0
    assert low["incomplete_sensitivity"]["accuracy"] == 1.0
    assert high["complete_specificity"]["accuracy"] == 1.0
    assert high["incomplete_sensitivity"]["accuracy"] == 1.0

from charlie_alpha.stats_guarded_weight_bridge import _gate, _guard_detail


def _row(*, case_id: str, prediction: str | None, valid: bool, correct: bool) -> dict:
    return {
        "case_id": case_id,
        "eligible": True,
        "gold_method_id": "paired_t",
        "predicted_method_id": prediction,
        "valid_output": valid,
        "correct": correct,
    }


def test_guard_preserves_every_valid_control() -> None:
    control = _row(case_id="x", prediction="paired_t", valid=True, correct=True)
    lora = _row(case_id="x", prediction=None, valid=False, correct=False)
    result = _guard_detail(control, lora, source_id="source")
    assert result["route"] == "valid-menu-free-control"
    assert result["correct"] is True
    assert result["predicted_method_id"] == "paired_t"


def test_guard_uses_valid_lora_only_after_invalid_control() -> None:
    control = _row(case_id="x", prediction="", valid=False, correct=False)
    lora = _row(case_id="x", prediction="paired_t", valid=True, correct=True)
    result = _guard_detail(control, lora, source_id="source")
    assert result["route"] == "invalid-control-lora-repair"
    assert result["correct"] is True
    assert result["valid_output"] is True


def test_h20_gate_requires_gain_without_any_loss() -> None:
    historical = {
        "paired": {"net_improvements": 2, "control_only": 0},
        "source_paired": {"a": {"net_improvements": 0}, "b": {"net_improvements": 2}},
        "correct_invalid_control_repairs": 2,
        "repair_source_count": 1,
    }
    synthetic = {
        "parent": {"eligible_accuracy": 0.2, "valid_output_rate": 0.4},
        "folds": {
            "a": {"metrics": {"eligible_accuracy": 0.2, "valid_output_rate": 0.5}},
            "b": {"metrics": {"eligible_accuracy": 0.3, "valid_output_rate": 0.4}},
        },
    }
    gates = {
        "minimum_historical_net_improvements": 1,
        "maximum_control_only_losses": 0,
        "minimum_worst_source_net_improvement": 0,
        "minimum_correct_invalid_control_repairs": 1,
        "minimum_repair_source_count": 1,
        "maximum_synthetic_accuracy_regression": 0.0,
        "maximum_synthetic_validity_regression": 0.0,
    }
    assert _gate(historical, synthetic, gates)["passed"] is True

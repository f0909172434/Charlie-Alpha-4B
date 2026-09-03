from charlie_alpha.stats_external_weight_bridge import _gate_result


def test_h19_gate_rejects_one_control_only_loss() -> None:
    paired = {"net_improvements": 2, "control_only": 1}
    source_paired = {"a": {"net_improvements": 2}}
    control = {"valid_output_rate": 0.8}
    candidate = {"eligible_accuracy": 0.8, "valid_output_rate": 0.8}
    synthetic_parent = {"eligible_accuracy": 0.2, "valid_output_rate": 0.8}
    synthetic_folds = [{"eligible_accuracy": 0.2, "valid_output_rate": 0.8}]
    gates = {
        "minimum_candidate_accuracy": 0.65,
        "minimum_net_improvements": 1,
        "maximum_control_only_losses": 0,
        "minimum_worst_source_net_improvement": 0,
        "maximum_validity_regression": 0.0,
        "maximum_synthetic_accuracy_regression": 0.0,
        "maximum_synthetic_validity_regression": 0.0,
    }
    result = _gate_result(
        paired,
        source_paired,
        control,
        candidate,
        synthetic_parent,
        synthetic_folds,
        gates,
    )
    assert result["passed"] is False
    assert result["checks"]["zero_control_only_losses"] is False

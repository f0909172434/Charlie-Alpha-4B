from charlie_alpha.stats_output_factorization import _gate_report


def test_factorization_gate_requires_absolute_and_paired_gains() -> None:
    scores = {
        "joint_method_accuracy": 0.25,
        "factorized_method_accuracy": 0.375,
        "joint_exact_accuracy": 0.20,
        "factorized_exact_accuracy": 0.325,
    }
    gates = {
        "minimum_factorized_method_accuracy": 0.35,
        "minimum_factorized_exact_accuracy": 0.30,
        "minimum_method_gain_over_joint_points": 8.0,
        "minimum_exact_gain_over_joint_points": 8.0,
    }
    assert _gate_report(scores, gates)["passed"]

from charlie_alpha.stats_selective_external import (
    _external_gate,
    _table_grid,
)


def test_table_grid_expands_rowspan_without_manual_row_repair() -> None:
    xml = """
    <table><tbody>
      <tr>
        <td rowspan="2">Question</td><td rowspan="2">Scenario</td>
        <td>A</td><td>Example A</td>
      </tr>
      <tr><td>B</td><td>Example B</td></tr>
    </tbody></table>
    """
    rows = _table_grid(xml)
    assert rows == [
        ["Question", "Scenario", "A", "Example A"],
        ["Question", "Scenario", "B", "Example B"],
    ]


def test_external_gate_forbids_any_control_only_loss() -> None:
    report = _external_gate(
        {
            "eligible_accuracy": 0.60,
            "valid_output_rate": 0.80,
        },
        {
            "eligible_accuracy": 0.60,
            "valid_output_rate": 0.90,
        },
        {"control_only": 1, "net_improvements": 0},
        {
            "source-a": {"net_improvements": 0},
            "source-b": {"net_improvements": 0},
        },
        {
            "minimum_candidate_accuracy": 0.45,
            "minimum_gain_over_control_points": 0.0,
            "maximum_control_only_losses": 0,
            "minimum_worst_source_net_improvement": 0,
            "maximum_validity_regression_points": 0.0,
        },
    )
    assert not report["passed"]
    assert not report["checks"]["zero_control_only_losses"]

from charlie_alpha.stats_external_weight_bridge_amendment import _fixed_evaluation_case


def test_fixed_evaluation_case_adapts_single_gold_methods_entry() -> None:
    result = _fixed_evaluation_case(
        {
            "case_id": "x",
            "question": "q",
            "gold_methods": ["paired_t"],
            "gold_columns": ["y"],
        }
    )
    assert result["gold_method_id"] == "paired_t"
    assert result["gold_methods"] == ["paired_t"]

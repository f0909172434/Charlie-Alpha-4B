from charlie_alpha.stats_external_catalog import (
    _canonicalize_method_label,
    _gate_report,
    _messages,
    _select_case_table,
    _supplementary_asset_links,
)


def test_external_aliases_are_exact_and_preserve_out_of_catalog_labels() -> None:
    assert _canonicalize_method_label("Mann-Whitney U test") == "mann_whitney"
    assert _canonicalize_method_label("Answer: paired t-test*") == "paired_t"
    assert _canonicalize_method_label("One-way ANOVA") is None
    assert _canonicalize_method_label("Pearson correlation") is None


def test_external_prompt_changes_only_by_the_fixed_catalog() -> None:
    case = {"vignette": "Compare a continuous outcome in two independent groups."}
    control = _messages(case, grounded=False)
    candidate = _messages(case, grounded=True)
    assert control[1] == candidate[1]
    assert "Repository method catalog" not in control[0]["content"]
    assert "Repository method catalog" in candidate[0]["content"]
    assert "independent_t" in candidate[0]["content"]


def test_source_table_selection_requires_the_complete_frozen_case_count() -> None:
    target_rows = [["Case", "Study vignette", "Correct statistical test"]]
    target_rows.extend(
        [[str(index), f"Vignette {index}", "paired t test"] for index in range(1, 28)]
    )
    tables = [
        {
            "source_file": "article.nxml",
            "table_index": 1,
            "label": "Table 1",
            "caption": "Unrelated summary",
            "rows": [["Item", "Value"], ["a", "b"]],
        },
        {
            "source_file": "supplement.docx",
            "table_index": 2,
            "label": "",
            "caption": "Expert statistical test vignettes",
            "rows": target_rows,
        },
    ]
    table, pairs = _select_case_table(tables, expected_count=27)
    assert table["source_file"] == "supplement.docx"
    assert len(pairs) == 27
    assert pairs[0] == ("Vignette 1", "paired t test")
    assert pairs[-1] == ("Vignette 27", "paired t test")


def test_supplementary_link_detection_does_not_confuse_support_with_supplement() -> None:
    article = b"""
    <a href="https://support.nlm.nih.gov/">Help</a>
    <a href="bin/article-supplement.docx">Supplement</a>
    """
    assert _supplementary_asset_links(article) == ["bin/article-supplement.docx"]


def test_external_gate_requires_coverage_absolute_accuracy_and_paired_gain() -> None:
    data = {"evaluation_authorized": True}
    control = {"eligible_accuracy": 0.25, "valid_output_rate": 1.0}
    candidate = {"eligible_accuracy": 0.50, "valid_output_rate": 0.95}
    paired = {"net_improvements": 3}
    gates = {
        "minimum_flat_catalog_accuracy": 0.35,
        "minimum_method_gain_points": 10.0,
        "minimum_net_improvements": 2,
        "maximum_validity_regression_points": 5.0,
    }
    result = _gate_report(
        data=data,
        control=control,
        candidate=candidate,
        paired=paired,
        gates=gates,
    )
    assert result["passed"]
    assert result["effect_points"]["eligible_method_accuracy"] == 25.0

    failed = _gate_report(
        data=data,
        control=control,
        candidate={"eligible_accuracy": 0.30, "valid_output_rate": 0.95},
        paired={"net_improvements": 1},
        gates=gates,
    )
    assert not failed["passed"]

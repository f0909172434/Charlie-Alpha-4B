from charlie_alpha.stats_external_catalog_v2 import (
    _CONYSO_QUALIFICATION_LABELS,
    _CRUCIBLE_QUALIFICATION_LABELS,
    _HAQ_ROWS,
    _external_v2_gate,
    _haq_cases,
    _parse_conyso_csv,
    _parse_crucible_readme,
    _qualification_summary,
)


def test_e2_v2_frozen_source_pool_passes_without_alias_expansion() -> None:
    qualification = _qualification_summary()
    assert qualification["case_count"] == 37
    assert qualification["eligible_count"] == 17
    assert qualification["coverage_fraction"] == 17 / 37
    assert qualification["qualification_gate"]["passed"]
    assert len(_HAQ_ROWS) == 22
    assert len(_CONYSO_QUALIFICATION_LABELS) == 8
    assert len(_CRUCIBLE_QUALIFICATION_LABELS) == 7


def test_e2_v2_haq_transcription_keeps_every_non_na_cell() -> None:
    cases = _haq_cases()
    assert len(cases) == 22
    assert sum(bool(case["eligible"]) for case in cases) == 10
    assert cases[0]["case_id"] == "haq-2016-001"
    assert cases[-1]["case_id"] == "haq-2016-022"
    assert cases[0]["gold_raw"] == "Chi-square test"
    assert cases[-1]["gold_raw"] == "Repeated-measures ANOVA"


def test_e2_v2_conyso_parser_retains_complete_csv_order() -> None:
    payload = (
        b"Your question,Data,Test\r\n"
        b'Is one mean different from a target?,"One group, continuous",One-sample t-test\r\n'
        b"Do two groups' means differ?,Two independent groups,Two-sample t-test\r\n"
        b"Did a paired measure change?,Before/after on same units,Paired t-test\r\n"
        b'Do three or more groups\' means differ?,"3+ groups, continuous",ANOVA\r\n'
        b"Are two categorical variables related?,Counts in categories,Chi-square test\r\n"
        b"Are two continuous variables related?,Two continuous,Correlation / regression\r\n"
        b'Is a proportion different from a target?,"One group, pass/fail",One-proportion z-test\r\n'
        b'Do two proportions differ?,"Two groups, pass/fail",Two-proportion z-test\r\n'
    )
    cases = _parse_conyso_csv(payload)
    assert len(cases) == 8
    assert sum(bool(case["eligible"]) for case in cases) == 3
    assert [case["source_order"] for case in cases] == list(range(1, 9))


def test_e2_v2_crucible_parser_retains_both_frozen_tables() -> None:
    payload = b"""
### Parametric Tests

| Test | Function | Use Case |
|------|----------|----------|
| Independent t-test | `a` | Compare 2 independent groups |
| Welch's t-test | `b` | Compare 2 groups (unequal variance) |
| Paired t-test | `c` | Compare 2 related groups |
| One-way ANOVA | `d` | Compare 3+ independent groups |

### Non-Parametric Tests

| Test | Function | Use Case |
|------|----------|----------|
| Mann-Whitney U | `e` | Non-parametric alternative to t-test |
| Wilcoxon signed-rank | `f` | Non-parametric alternative to paired t-test |
| Kruskal-Wallis | `g` | Non-parametric alternative to ANOVA |

### Effect Sizes
"""
    cases = _parse_crucible_readme(payload)
    assert len(cases) == 7
    assert sum(bool(case["eligible"]) for case in cases) == 4
    assert cases[0]["gold_raw"] == "Independent t-test"
    assert cases[-1]["gold_raw"] == "Kruskal-Wallis"


def test_e2_v2_gate_requires_pooled_effect_and_cross_source_robustness() -> None:
    data = {"evaluation_authorized": True}
    control = {"eligible_accuracy": 0.20, "valid_output_rate": 1.0}
    candidate = {"eligible_accuracy": 0.45, "valid_output_rate": 0.98}
    paired = {"net_improvements": 4}
    source_paired = {
        "haq-2016": {"net_improvements": 3},
        "conyso-2026": {"net_improvements": 0},
        "crucible-bench": {"net_improvements": -1},
    }
    gates = {
        "minimum_flat_catalog_accuracy": 0.35,
        "minimum_method_gain_points": 10.0,
        "minimum_net_improvements": 2,
        "maximum_validity_regression_points": 5.0,
        "minimum_nonnegative_sources": 2,
        "minimum_worst_source_net_improvement": -1,
    }
    passed = _external_v2_gate(
        data=data,
        control=control,
        candidate=candidate,
        paired=paired,
        source_paired=source_paired,
        gates=gates,
    )
    assert passed["passed"]

    failed = _external_v2_gate(
        data=data,
        control=control,
        candidate=candidate,
        paired=paired,
        source_paired={
            "haq-2016": {"net_improvements": 4},
            "conyso-2026": {"net_improvements": -1},
            "crucible-bench": {"net_improvements": -2},
        },
        gates=gates,
    )
    assert not failed["passed"]
    assert not failed["checks"]["cross_source_direction"]
    assert not failed["checks"]["worst_source_regression"]


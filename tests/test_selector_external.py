import json

from charlie_alpha.stats_selector_external import (
    _SOURCE_LABELS,
    _external_gate,
    _qualified_rows,
    _source_rows,
)


def _fake_bioc() -> bytes:
    rows = "".join(
        f"<tr><td>{label.replace('&', '&amp;')}</td><td>Correct</td></tr>"
        for label in _SOURCE_LABELS
    )
    passages = [
        {
            "infons": {
                "section_type": "TABLE",
                "id": "TAB1",
                "type": "table",
                "xml": f"<table><tbody>{rows}</tbody></table>",
            },
            "text": "table",
        },
        {
            "infons": {"section_type": "APPENDIX", "type": "paragraph"},
            "text": "Standardized hypothesis testing scenarios",
        },
    ]
    passages.extend(
        {
            "infons": {"section_type": "APPENDIX", "type": "paragraph"},
            "text": f"{index}. Scenario {index}",
        }
        for index in range(1, 21)
    )
    return json.dumps(
        [
            {
                "documents": [
                    {"infons": {"license": "CC BY"}, "passages": passages}
                ]
            }
        ]
    ).encode()


def test_e3_source_parser_pairs_twenty_scenarios_and_labels_in_order() -> None:
    source, rows = _source_rows(_fake_bioc())
    assert source["case_count"] == 20
    assert rows[0]["source_gold_label"] == "Student's t-test"
    assert rows[-1]["source_gold_label"] == "Poisson regression"
    assert rows[0]["question"] == "Scenario 1"


def test_e3_head_eligibility_keeps_poisson_visible_but_unscored() -> None:
    _, rows = _source_rows(_fake_bioc())
    observed = {
        "independent_t",
        "mann_whitney",
        "fisher_exact",
        "chi_square",
        "logrank",
        "paired_t",
        "wilcoxon_signed_rank",
        "ols",
        "logistic_glm",
    }
    qualified = _qualified_rows(rows, observed)
    assert sum(bool(row["catalog_eligible"]) for row in qualified) == 10
    assert sum(bool(row["head_eligible"]) for row in qualified) == 9
    poisson = qualified[-1]
    assert poisson["gold_method_id"] == "poisson_glm"
    assert poisson["catalog_eligible"]
    assert not poisson["head_eligible"]


def test_e3_gate_requires_absolute_accuracy_and_paired_gain() -> None:
    control = {"eligible_accuracy": 0.2, "valid_output_rate": 0.5}
    candidate = {"eligible_accuracy": 0.7, "valid_output_rate": 1.0}
    paired = {"net_improvements": 4}
    gates = {
        "minimum_head_eligible_accuracy": 0.55,
        "minimum_method_gain_points": 20.0,
        "minimum_net_improvements": 2,
        "minimum_head_valid_output_rate": 1.0,
    }
    assert _external_gate(control, candidate, paired, gates)["passed"]

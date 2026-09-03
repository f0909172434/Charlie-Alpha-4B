from charlie_alpha.stats_catalog_ranking import _clean_generated_method, _gate_report, _messages


def test_clean_generated_method_removes_think_fence_and_quotes() -> None:
    assert _clean_generated_method('<think>x</think>\n```text\n"welch_t"\n```') == "welch_t"


def test_method_prompt_contains_fixed_catalog_and_plain_id_contract() -> None:
    case = {"question": "Choose the method."}
    messages = _messages(case)
    assert len(messages) == 2
    assert "Return only that method ID" in messages[0]["content"]
    assert "welch_t — Welch two-sample t test" in messages[1]["content"]
    assert "blocked_time_series_cv" in messages[1]["content"]


def test_ranking_gate_requires_absolute_and_relative_gains() -> None:
    report = _gate_report(
        scores={"ranked_method_accuracy": 0.4, "free_method_accuracy": 0.25},
        gates={
            "minimum_ranked_method_accuracy": 0.35,
            "minimum_gain_over_free_control_points": 10.0,
            "minimum_gain_over_h7_confirmed_points": 5.0,
        },
        h7_method_accuracy=0.2833333333333333,
    )
    assert report["passed"]

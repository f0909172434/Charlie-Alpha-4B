import json

from charlie_alpha.stats_canonical_bottleneck import _paired_selector_rows, _training_messages


def test_canonical_training_target_uses_method_id() -> None:
    case = {
        "question": "Choose the analysis.",
        "gold_methods": ["welch_t"],
        "gold_columns": ["y", "arm"],
    }
    messages = _training_messages(case, canonical=True)
    target = json.loads(messages[-1]["content"])
    assert target == {"methods": ["welch_t"], "columns": ["y", "arm"]}


def test_display_name_control_does_not_use_canonical_id() -> None:
    case = {
        "question": "Choose the analysis.",
        "gold_methods": ["welch_t"],
        "gold_columns": ["y", "arm"],
    }
    messages = _training_messages(case, canonical=False)
    target = json.loads(messages[-1]["content"])
    assert target["methods"] == ["Welch two-sample t test"]


def test_training_prompts_are_identical_between_arms() -> None:
    case = {
        "question": "Choose the analysis.",
        "gold_methods": ["welch_t"],
        "gold_columns": ["y", "arm"],
    }
    display = _training_messages(case, canonical=False)
    canonical = _training_messages(case, canonical=True)
    assert display[:-1] == canonical[:-1]
    assert display[-1]["content"] != canonical[-1]["content"]


def test_selector_pairs_require_identical_blueprints() -> None:
    record = {"metadata": {"blueprint_id": "x"}}
    simulation = {"scenario": {"blueprint_id": "x"}}
    assert _paired_selector_rows([record], [simulation]) == [(record, simulation)]

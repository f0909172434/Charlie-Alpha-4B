from charlie_alpha.data_pipeline import (
    _compact_code_answer,
    _compact_code_prompt,
    deterministic_split,
    split_quotas,
)


def test_split_is_deterministic_and_grouped() -> None:
    first = deterministic_split("problem-123", 42, {"train": 90, "valid": 5, "test": 5})
    assert first == deterministic_split("problem-123", 42, {"train": 90, "valid": 5, "test": 5})
    assert first in {"train", "valid", "test"}


def test_split_quotas_preserve_total_and_holdouts() -> None:
    quotas = split_quotas(50, {"train": 90, "valid": 5, "test": 5})
    assert quotas == {"train": 44, "valid": 3, "test": 3}
    assert sum(quotas.values()) == 50


def test_compact_code_prompt_uses_structured_problem_fields() -> None:
    prompt = _compact_code_prompt(
        {
            "title": "Add",
            "description": "Add two integers.",
            "time_limit": 1.0,
            "memory_limit": 256,
            "input_format": "Two integers.",
            "output_format": "Their sum.",
            "examples": [{"input": "1 2", "output": "3"}],
            "note": None,
        },
        "cpp",
    )
    assert "C++17" in prompt
    assert "Add two integers." in prompt
    assert "1 2" in prompt and "3" in prompt
    assert "Analyze the maximum input constraints" not in prompt


def test_compact_code_answer_drops_long_thinking() -> None:
    answer = _compact_code_answer(
        "<think>very long hidden reasoning</think>\n```python\nprint(2)\n```"
    )
    assert "hidden reasoning" not in answer
    assert answer == "```python\nprint(2)\n```"

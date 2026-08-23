from opencc import OpenCC

from charlie_alpha.mixer import (
    _convert_prose,
    _ratio_summary,
    _repeat_to_budget,
    _select_english_for_chinese,
)


def _row(language: str, category: str, tokens: int, index: int) -> dict:
    domain = "math" if category == "math" else "code"
    metadata = {
        "language": language,
        "domain": domain,
        "assistant_token_count": tokens,
        "token_count": tokens + 10,
        "prompt_sha256": f"{index:064x}",
    }
    if domain == "code":
        metadata["code_language"] = category
    return {"messages": [], "metadata": metadata}


def test_english_selection_compensates_chinese_domain_mix() -> None:
    english = [
        *[_row("en", "math", 100, index) for index in range(100)],
        *[_row("en", "python", 100, 100 + index) for index in range(100)],
        *[_row("en", "cpp", 100, 200 + index) for index in range(100)],
    ]
    chinese = [
        *[_row("zh_Hant", "math", 100, 300 + index) for index in range(5)],
        *[_row("zh_Hans", "python", 100, 400 + index) for index in range(5)],
    ]
    selected = _select_english_for_chinese(english, chinese, english_ratio=0.8)
    summary = _ratio_summary([*selected, *chinese])
    assert abs(summary["domain_ratios"]["math"] - 0.5) <= 0.03
    assert abs(summary["domain_ratios"]["code"] - 0.5) <= 0.03
    assert abs(summary["code_language_ratios"]["python"] - 0.5) <= 0.03
    assert abs(summary["code_language_ratios"]["cpp"] - 0.5) <= 0.03


def test_script_conversion_preserves_formula_and_code() -> None:
    source = "計算 $x=2$ 並執行：\n```python\nprint('繁體')\n```"
    converted = _convert_prose(source, OpenCC("t2s"))
    assert "计算" in converted
    assert "$x=2$" in converted
    assert "print('繁體')" in converted


def test_repeat_budget_is_bounded_and_tracked() -> None:
    rows = [_row("zh_Hant", "math", 10, 1), _row("zh_Hant", "math", 10, 2)]
    repeated = _repeat_to_budget(rows, budget=50, max_repeats=3)
    assert len(repeated) == 5
    assert max(row["metadata"]["sampling_repeat_index"] for row in repeated) == 2

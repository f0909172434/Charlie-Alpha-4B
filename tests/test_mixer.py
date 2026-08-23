from charlie_alpha.mixer import _ratio_summary, _select_english_for_chinese


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

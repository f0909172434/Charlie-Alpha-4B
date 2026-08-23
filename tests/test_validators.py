from charlie_alpha.validators import (
    extract_code_blocks,
    extract_formulas,
    has_target_script,
    is_contaminated,
    translation_preserves_source,
    validate_chat_record,
    word_ngrams,
)


def test_preservation_accepts_translated_prose() -> None:
    source = "Compute $x = 2 + 3$.\n```python\nprint(5)\n```"
    translated = "計算 $x = 2 + 3$。\n```python\nprint(5)\n```"
    assert translation_preserves_source(source, translated) == (True, [])


def test_preservation_rejects_changed_formula_and_code() -> None:
    source = "Compute $x=2$.\n```python\nprint(2)\n```"
    translated = "計算 $x=3$。\n```python\nprint(3)\n```"
    valid, errors = translation_preserves_source(source, translated)
    assert not valid
    assert "code blocks changed" in errors
    assert "formulas changed" in errors
    assert "numbers changed" in errors


def test_extractors_and_script_checks() -> None:
    assert extract_code_blocks("```cpp\nint x;\n```") == ["int x;"]
    assert extract_formulas("$x = 1$ and $$y=2$$ and \\(z=3\\)") == [
        "y=2",
        "x=1",
        "z=3",
    ]
    assert has_target_script("這是一段繁體中文數學解題與程式設計說明。", "zh_Hant")
    assert has_target_script("这是一段简体中文数学解题与程序设计说明。", "zh_Hans")


def test_word_ngrams_are_normalized() -> None:
    assert word_ngrams("A  B c", 2) == {("a", "b"), ("b", "c")}


def test_contamination_compares_candidate_against_all_references() -> None:
    references = [word_ngrams("unrelated words here", 2), word_ngrams("a b c d", 2)]
    assert is_contaminated("a b c d", references, size=2, threshold=0.5)
    assert not is_contaminated("different sample entirely", references, size=2, threshold=0.5)


def test_schema_validation() -> None:
    record = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        "metadata": {
            "source_id": "1",
            "source_repo": "owner/repo",
            "source_revision": "a" * 40,
            "domain": "math",
            "language": "en",
            "split": "train",
        },
    }
    assert validate_chat_record(record) == []

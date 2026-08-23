from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from typing import Any

from opencc import OpenCC

_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_DISPLAY_MATH_RE = re.compile(r"\\\[(.*?)\\\]|\$\$(.*?)\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)
_PAREN_MATH_RE = re.compile(r"\\\((.*?)\\\)", re.DOTALL)
_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    value = _WHITESPACE_RE.sub(" ", value)
    return value.strip()


def word_ngrams(value: str, size: int) -> set[tuple[str, ...]]:
    words = normalize_text(value).split()
    if len(words) < size:
        return set()
    return {tuple(words[index : index + size]) for index in range(len(words) - size + 1)}


def overlap_ratio(value: str, reference: set[tuple[str, ...]], size: int) -> float:
    candidate = word_ngrams(value, size)
    if not candidate:
        return 0.0
    return len(candidate & reference) / len(candidate)


def is_contaminated(
    value: str,
    reference_sets: Iterable[set[tuple[str, ...]]],
    size: int,
    threshold: float,
) -> bool:
    candidate = word_ngrams(value, size)
    if not candidate:
        return False
    return any(
        len(candidate & reference) / len(candidate) >= threshold for reference in reference_sets
    )


def extract_code_blocks(value: str) -> list[str]:
    return [match.rstrip() for match in _FENCE_RE.findall(value)]


def extract_formulas(value: str) -> list[str]:
    display = [next(group for group in match if group) for match in _DISPLAY_MATH_RE.findall(value)]
    inline = _INLINE_MATH_RE.findall(value)
    parenthesized = _PAREN_MATH_RE.findall(value)
    return [_WHITESPACE_RE.sub("", item) for item in [*display, *inline, *parenthesized]]


def extract_numbers(value: str) -> list[str]:
    return _NUMBER_RE.findall(value)


def translation_preserves_source(source: str, translated: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if Counter(extract_code_blocks(source)) != Counter(extract_code_blocks(translated)):
        errors.append("code blocks changed")
    if Counter(extract_formulas(source)) != Counter(extract_formulas(translated)):
        errors.append("formulas changed")
    if Counter(extract_numbers(source)) != Counter(extract_numbers(translated)):
        errors.append("numbers changed")
    return not errors, errors


def has_target_script(value: str, language: str) -> bool:
    han = _CJK_RE.findall(value)
    if len(han) < 8:
        return False
    if language == "zh_Hant":
        converted = OpenCC("t2s").convert(value)
    elif language == "zh_Hans":
        converted = OpenCC("s2t").convert(value)
    else:
        return True
    changed = sum(left != right for left, right in zip(value, converted, strict=False))
    return changed > 0


def validate_chat_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return ["messages must contain at least a user and assistant turn"]
    if messages[0].get("role") != "user" or messages[-1].get("role") != "assistant":
        errors.append("messages must start with user and end with assistant")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"message {index} is not an object")
            continue
        if message.get("role") not in {"system", "user", "assistant"}:
            errors.append(f"message {index} has an invalid role")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            errors.append(f"message {index} has empty content")
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")
    else:
        for key in ("source_id", "source_repo", "source_revision", "domain", "language", "split"):
            if not metadata.get(key):
                errors.append(f"metadata.{key} is required")
    return errors

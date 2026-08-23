from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from typing import Any

from opencc import OpenCC
from rich.console import Console
from transformers import AutoTokenizer

from .config import ProjectConfig
from .data_pipeline import load_processed
from .io_utils import (
    canonical_hash,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from .validators import has_target_script, translation_preserves_source

console = Console()


def _assistant_tokens(row: dict[str, Any]) -> int:
    return int(row["metadata"].get("assistant_token_count", row["metadata"]["token_count"]))


def _token_sum(rows: list[dict[str, Any]]) -> int:
    return sum(_assistant_tokens(row) for row in rows)


def _trim_to_budget(rows: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (row["metadata"]["prompt_sha256"], row["metadata"]["language"]),
    )
    selected: list[dict[str, Any]] = []
    used = 0
    for row in ordered:
        tokens = _assistant_tokens(row)
        if used + tokens <= budget or not selected:
            selected.append(row)
            used += tokens
    return selected


def _repeat_to_budget(
    rows: list[dict[str, Any]], budget: int, max_repeats: int
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (row["metadata"]["prompt_sha256"], row["metadata"]["language"]),
    )
    selected: list[dict[str, Any]] = []
    used = 0
    for repeat_index in range(max_repeats):
        added = False
        for row in ordered:
            tokens = _assistant_tokens(row)
            if used + tokens > budget:
                continue
            repeated = deepcopy(row)
            repeated["metadata"]["sampling_repeat_index"] = repeat_index
            selected.append(repeated)
            used += tokens
            added = True
        if not added or used >= budget:
            break
    return selected


_PROTECTED_RE = re.compile(
    r"(```.*?```|\\\[.*?\\\]|\$\$.*?\$\$|(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)|\\\(.*?\\\))",
    re.DOTALL,
)


def _convert_prose(value: str, converter: OpenCC) -> str:
    parts = _PROTECTED_RE.split(value)
    return "".join(
        part if index % 2 else converter.convert(part) for index, part in enumerate(parts)
    )


def _augment_scripts(
    distilled: dict[str, list[dict[str, Any]]], tokenizer: Any
) -> dict[str, list[dict[str, Any]]]:
    converters = {"zh_Hant": OpenCC("s2twp"), "zh_Hans": OpenCC("t2s")}
    augmented = {language: list(rows) for language, rows in distilled.items()}
    for target_language, source_language in (("zh_Hant", "zh_Hans"), ("zh_Hans", "zh_Hant")):
        existing_parents = {
            row["metadata"]["parent_prompt_sha256"] for row in augmented[target_language]
        }
        for source in distilled[source_language]:
            parent = source["metadata"]["parent_prompt_sha256"]
            if parent in existing_parents:
                continue
            converted = deepcopy(source)
            converted["messages"] = [
                {
                    "role": message["role"],
                    "content": _convert_prose(message["content"], converters[target_language]),
                }
                for message in source["messages"]
            ]
            if any(
                not translation_preserves_source(original["content"], translated["content"])[0]
                for original, translated in zip(
                    source["messages"], converted["messages"], strict=True
                )
            ):
                continue
            if not has_target_script(
                "\n".join(message["content"] for message in converted["messages"]),
                target_language,
            ):
                continue
            metadata = converted["metadata"]
            metadata["language"] = target_language
            metadata["script_conversion_from"] = source_language
            metadata["prompt_sha256"] = sha256_text(converted["messages"][0]["content"])
            metadata["assistant_sha256"] = sha256_text(converted["messages"][-1]["content"])
            metadata["assistant_token_count"] = len(
                tokenizer.encode(converted["messages"][-1]["content"], add_special_tokens=False)
            )
            metadata["token_count"] = len(
                tokenizer.apply_chat_template(
                    converted["messages"],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=False,
                )
            )
            if metadata["token_count"] > 1024:
                continue
            augmented[target_language].append(converted)
            existing_parents.add(parent)
    return augmented


def _by_category(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "math": [row for row in rows if row["metadata"]["domain"] == "math"],
        "python": [row for row in rows if row["metadata"].get("code_language") == "python"],
        "cpp": [row for row in rows if row["metadata"].get("code_language") == "cpp"],
    }


def _select_english_for_chinese(
    english: list[dict[str, Any]], chinese: list[dict[str, Any]], english_ratio: float
) -> list[dict[str, Any]]:
    chinese_tokens = _token_sum(chinese)
    if not chinese_tokens:
        categories = _by_category(english)
        leaf_budget = min(
            _token_sum(categories["math"]) // 2,
            _token_sum(categories["python"]),
            _token_sum(categories["cpp"]),
        )
        return [
            *_trim_to_budget(categories["math"], leaf_budget * 2),
            *_trim_to_budget(categories["python"], leaf_budget),
            *_trim_to_budget(categories["cpp"], leaf_budget),
        ]

    target_total = round(chinese_tokens / (1.0 - english_ratio))
    chinese_by_category = _by_category(chinese)
    desired = {
        "math": max(0, round(target_total * 0.50) - _token_sum(chinese_by_category["math"])),
        "python": max(0, round(target_total * 0.25) - _token_sum(chinese_by_category["python"])),
        "cpp": max(0, round(target_total * 0.25) - _token_sum(chinese_by_category["cpp"])),
    }
    english_by_category = _by_category(english)
    scale = min(
        1.0,
        *(
            _token_sum(english_by_category[key]) / budget
            for key, budget in desired.items()
            if budget > 0
        ),
    )
    return [
        row
        for key in ("math", "python", "cpp")
        for row in _trim_to_budget(english_by_category[key], round(desired[key] * scale))
    ]


def _ratio_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = _token_sum(rows)
    language_tokens: Counter[str] = Counter()
    domain_tokens: Counter[str] = Counter()
    code_tokens: Counter[str] = Counter()
    for row in rows:
        tokens = _assistant_tokens(row)
        metadata = row["metadata"]
        language_tokens[metadata["language"]] += tokens
        domain_tokens[metadata["domain"]] += tokens
        if metadata["domain"] == "code":
            code_tokens[metadata["code_language"]] += tokens

    def ratio(value: int, denominator: int) -> float:
        return round(value / denominator, 4) if denominator else 0.0

    return {
        "records": len(rows),
        "assistant_tokens": total,
        "language_tokens": dict(language_tokens),
        "language_ratios": {key: ratio(value, total) for key, value in language_tokens.items()},
        "domain_tokens": dict(domain_tokens),
        "domain_ratios": {key: ratio(value, total) for key, value in domain_tokens.items()},
        "code_language_tokens": dict(code_tokens),
        "code_language_ratios": {
            key: ratio(value, sum(code_tokens.values())) for key, value in code_tokens.items()
        },
    }


def _revalidate_distilled(
    processed: dict[str, list[dict[str, Any]]],
    distilled: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, int]]]:
    source_by_hash = {
        row["metadata"]["prompt_sha256"]: row for rows in processed.values() for row in rows
    }
    accepted: dict[str, list[dict[str, Any]]] = {language: [] for language in distilled}
    rejected: dict[str, Counter[str]] = {language: Counter() for language in distilled}
    for language, rows in distilled.items():
        for row in rows:
            parent_hash = row["metadata"].get("parent_prompt_sha256")
            source = source_by_hash.get(parent_hash)
            if source is None:
                rejected[language]["missing_parent"] += 1
                continue
            question_valid, _ = translation_preserves_source(
                source["messages"][0]["content"], row["messages"][0]["content"]
            )
            answer_valid, _ = translation_preserves_source(
                source["messages"][-1]["content"], row["messages"][-1]["content"]
            )
            if not question_valid or not answer_valid:
                rejected[language]["preservation"] += 1
                continue
            if not has_target_script(
                "\n".join(message["content"] for message in row["messages"]), language
            ):
                rejected[language]["script"] += 1
                continue
            accepted[language].append(row)
    return accepted, {language: dict(counts) for language, counts in rejected.items()}


def mix_data(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    processed_dir = config.path_for("processed_dir")
    distilled_dir = config.path_for("distilled_dir")
    final_dir = config.path_for("final_dir")
    manifest_dir = config.path_for("manifest_dir")
    source_files = [processed_dir / f"{split}.jsonl" for split in ("train", "valid", "test")]
    source_files.extend(distilled_dir / f"{language}.jsonl" for language in ("zh_Hant", "zh_Hans"))
    missing = [str(path) for path in source_files if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing prepared/distilled inputs: {missing}")

    fingerprint = canonical_hash(
        {
            "files": {
                str(path.relative_to(config.root)): sha256_file(path) for path in source_files
            },
            "ratios": {
                "language": config.section("data")["language_token_ratios"],
                "domain": config.section("data")["domain_token_ratios"],
                "code": config.section("data")["code_language_ratios"],
            },
            "version": "mix-v3",
        }
    )
    done_path = final_dir / ".done.json"
    if done_path.exists() and not force:
        existing = json.loads(done_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint:
            console.print("[cyan]Final data already matches all source hashes.[/cyan]")
            return existing

    processed = load_processed(config)
    distilled = {
        language: list(read_jsonl(distilled_dir / f"{language}.jsonl"))
        for language in ("zh_Hant", "zh_Hans")
    }
    distilled, distillation_rejections = _revalidate_distilled(processed, distilled)
    model_source = config.sources["models"]["base_hf"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_source["repo_id"],
        revision=model_source["revision"],
        trust_remote_code=True,
    )
    distilled = _augment_scripts(distilled, tokenizer)
    minimum = int(config.section("distillation")["minimum_per_language"])
    too_small = {language: len(rows) for language, rows in distilled.items() if len(rows) < minimum}
    if too_small:
        raise RuntimeError(f"Distillation minimum not reached: {too_small}")

    language_targets = config.section("data")["language_token_ratios"]
    mixed_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "valid", "test"):
        english = processed[split]
        hant = [row for row in distilled["zh_Hant"] if row["metadata"]["split"] == split]
        hans = [row for row in distilled["zh_Hans"] if row["metadata"]["split"] == split]
        if split == "train":
            english_categories = _by_category(english)
            english_capacity = 4 * min(
                _token_sum(english_categories["math"]) // 2,
                _token_sum(english_categories["python"]),
                _token_sum(english_categories["cpp"]),
            )
            target_per_chinese_language = round(
                english_capacity
                * float(language_targets["zh_Hant"])
                / float(language_targets["en"])
            )
            max_repeats = int(config.section("data")["max_train_repeats_per_chinese_record"])
            hant = _repeat_to_budget(hant, target_per_chinese_language, max_repeats)
            hans = _repeat_to_budget(hans, target_per_chinese_language, max_repeats)
        chinese_budget = min(_token_sum(hant), _token_sum(hans))
        hant = _trim_to_budget(hant, chinese_budget) if chinese_budget else []
        hans = _trim_to_budget(hans, chinese_budget) if chinese_budget else []
        balanced_language_budget = min(_token_sum(hant), _token_sum(hans))
        hant = _trim_to_budget(hant, balanced_language_budget)
        hans = _trim_to_budget(hans, balanced_language_budget)
        chinese = [*hant, *hans]
        english = _select_english_for_chinese(
            english,
            chinese,
            english_ratio=float(language_targets["en"]),
        )
        rows = [*english, *chinese]
        rows.sort(
            key=lambda row: canonical_hash(
                {
                    "seed": config.section("project")["seed"],
                    "prompt": row["metadata"]["prompt_sha256"],
                    "language": row["metadata"]["language"],
                }
            )
        )
        mixed_by_split[split] = rows
        write_jsonl(final_dir / f"{split}.jsonl", rows)

    all_rows = [row for rows in mixed_by_split.values() for row in rows]
    manifest_rows = [
        {
            key: row["metadata"].get(key)
            for key in (
                "source_id",
                "source_repo",
                "source_revision",
                "source_config",
                "source_license",
                "domain",
                "code_language",
                "language",
                "split",
                "token_count",
                "assistant_token_count",
                "prompt_sha256",
                "assistant_sha256",
                "parent_prompt_sha256",
                "teacher_repo",
                "teacher_revision",
                "script_conversion_from",
                "sampling_repeat_index",
            )
            if row["metadata"].get(key) is not None
        }
        for row in all_rows
    ]
    write_jsonl(manifest_dir / "final-records.jsonl", manifest_rows)
    summary = {
        "fingerprint": fingerprint,
        "overall": _ratio_summary(all_rows),
        "splits": {split: _ratio_summary(rows) for split, rows in mixed_by_split.items()},
        "distillation_revalidation_rejections": distillation_rejections,
    }
    write_json(manifest_dir / "mix-summary.json", summary)
    write_json(done_path, summary)
    return summary

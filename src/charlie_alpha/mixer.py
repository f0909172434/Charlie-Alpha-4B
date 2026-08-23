from __future__ import annotations

import json
from collections import Counter
from typing import Any

from rich.console import Console

from .config import ProjectConfig
from .data_pipeline import load_processed
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl
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
            "version": "mix-v2",
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

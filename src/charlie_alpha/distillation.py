from __future__ import annotations

import json
import time
from collections import Counter, deque
from typing import Any

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from rich.console import Console
from transformers import AutoTokenizer

from .config import ProjectConfig
from .data_pipeline import load_processed, split_quotas
from .io_utils import append_jsonl, canonical_hash, read_jsonl, sha256_text, write_json
from .validators import has_target_script, translation_preserves_source, validate_chat_record

console = Console()

_LANGUAGE_NAMES = {
    "zh_Hant": "Traditional Chinese as used in Taiwan",
    "zh_Hans": "Simplified Chinese",
}


def _candidate_order(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    buckets: dict[str, deque[dict[str, Any]]] = {
        "math": deque(),
        "python": deque(),
        "cpp": deque(),
    }
    selected = [row for row in rows if row["metadata"]["split"] == split]
    selected.sort(
        key=lambda row: (
            row["metadata"]["token_count"],
            row["metadata"]["prompt_sha256"],
        )
    )
    for row in selected:
        metadata = row["metadata"]
        key = metadata.get("code_language") if metadata["domain"] == "code" else "math"
        buckets[key].append(row)
    ordered: list[dict[str, Any]] = []
    while any(buckets.values()):
        for key in ("math", "python", "cpp"):
            if buckets[key]:
                ordered.append(buckets[key].popleft())
    return ordered


def _prompt(source: dict[str, Any], language: str) -> str:
    question = source["messages"][0]["content"]
    answer = source["messages"][-1]["content"]
    target = _LANGUAGE_NAMES[language]
    return f"""Translate and lightly refine the following math or programming example into {target}.

Hard requirements:
- Keep every formula, LaTeX expression, number, code block, identifier, input, and output unchanged.
- Translate prose only. Do not solve the problem again and do not add facts.
- Keep <think> and </think> tags if present.
- Return exactly the two marked sections below, with no preface or commentary.

<QUESTION_SOURCE>
{question}
</QUESTION_SOURCE>

<ANSWER_SOURCE>
{answer}
</ANSWER_SOURCE>

Required output:
<QUESTION_TRANSLATION>translated question</QUESTION_TRANSLATION>
<ANSWER_TRANSLATION>translated answer</ANSWER_TRANSLATION>"""


def _parse_translation(value: str) -> tuple[str, str] | None:
    question_start = value.find("<QUESTION_TRANSLATION>")
    question_end = value.find("</QUESTION_TRANSLATION>")
    answer_start = value.find("<ANSWER_TRANSLATION>")
    answer_end = value.find("</ANSWER_TRANSLATION>")
    if min(question_start, question_end, answer_start, answer_end) < 0:
        return None
    question_start += len("<QUESTION_TRANSLATION>")
    answer_start += len("<ANSWER_TRANSLATION>")
    question = value[question_start:question_end].strip()
    answer = value[answer_start:answer_end].strip()
    if not question or not answer:
        return None
    return question, answer


def _render_prompt(tokenizer: Any, source: dict[str, Any], language: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a precise translator. Output only the requested marked sections.",
        },
        {"role": "user", "content": _prompt(source, language)},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _validate_translation(
    source: dict[str, Any], question: str, answer: str, language: str
) -> list[str]:
    errors: list[str] = []
    source_question = source["messages"][0]["content"]
    source_answer = source["messages"][-1]["content"]
    question_valid, question_errors = translation_preserves_source(source_question, question)
    answer_valid, answer_errors = translation_preserves_source(source_answer, answer)
    if not question_valid:
        errors.extend(f"question: {error}" for error in question_errors)
    if not answer_valid:
        errors.extend(f"answer: {error}" for error in answer_errors)
    if not has_target_script(f"{question}\n{answer}", language):
        errors.append(f"output does not contain enough {language} script evidence")
    return errors


def _translated_record(
    source: dict[str, Any],
    question: str,
    answer: str,
    language: str,
    teacher: dict[str, str],
    base_tokenizer: Any,
) -> dict[str, Any]:
    metadata = dict(source["metadata"])
    metadata.update(
        {
            "language": language,
            "teacher_repo": teacher["repo_id"],
            "teacher_revision": teacher["revision"],
            "parent_prompt_sha256": source["metadata"]["prompt_sha256"],
            "parent_assistant_sha256": source["metadata"]["assistant_sha256"],
            "prompt_sha256": sha256_text(question),
            "assistant_sha256": sha256_text(answer),
            "assistant_token_count": len(base_tokenizer.encode(answer, add_special_tokens=False)),
        }
    )
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    metadata["token_count"] = len(
        base_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
    )
    return {"messages": messages, "metadata": metadata}


def distill_data(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    settings = config.section("distillation")
    data_settings = config.section("data")
    distilled_dir = config.path_for("distilled_dir")
    manifest_dir = config.path_for("manifest_dir")
    distilled_dir.mkdir(parents=True, exist_ok=True)
    distillation_sources = {
        key: value for key, value in config.sources.items() if key != "evaluation_artifacts"
    }
    distillation_data_settings = {
        key: data_settings[key]
        for key in ("max_seq_length", "split_percentages", "distilled_targets")
    }
    fingerprint = canonical_hash(
        {
            "distillation": settings,
            "data": distillation_data_settings,
            "sources": distillation_sources,
            "v": 1,
        }
    )
    max_seconds = int(settings["max_seconds"])
    minimum = int(settings["minimum_per_language"])
    max_tokens = int(settings["max_new_tokens"])
    percentages = data_settings["split_percentages"]
    target_by_language = data_settings["distilled_targets"]
    done_path = distilled_dir / ".done.json"
    if done_path.exists() and not force:
        existing = json.loads(done_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint:
            console.print("[cyan]Distillation already matches the locked configuration.[/cyan]")
            return existing
    prior_summary_path = manifest_dir / "distillation-summary.json"
    if prior_summary_path.exists() and not force:
        prior = json.loads(prior_summary_path.read_text(encoding="utf-8"))
        prior_totals = prior.get("totals", {})
        reached_minimum = all(
            int(prior_totals.get(language, 0)) >= minimum for language in ("zh_Hant", "zh_Hans")
        )
        budget_spent = float(prior.get("elapsed_seconds", 0)) >= max_seconds * 0.95
        same_teacher = (
            prior.get("teacher_revision")
            == config.sources["models"]["teacher_mlx_4bit"]["revision"]
        )
        if reached_minimum and budget_spent and same_teacher:
            prior["fingerprint"] = fingerprint
            prior["minimum_complete"] = True
            write_json(prior_summary_path, prior)
            write_json(done_path, prior)
            console.print("[cyan]The overnight distillation budget was already completed.[/cyan]")
            return prior

    processed = load_processed(config)
    all_sources = [row for rows in processed.values() for row in rows]
    if not all_sources:
        raise RuntimeError("Prepared English data is missing; run `make data` first.")

    teacher = config.sources["models"]["teacher_mlx_4bit"]
    console.print("Downloading/loading the pinned 9B MLX teacher…")
    teacher_path = snapshot_download(
        repo_id=teacher["repo_id"],
        revision=teacher["revision"],
    )
    model, tokenizer = load(teacher_path, tokenizer_config={"trust_remote_code": True})
    base_source = config.sources["models"]["base_hf"]
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_source["repo_id"],
        revision=base_source["revision"],
        trust_remote_code=True,
    )
    mx.random.seed(int(config.section("project")["seed"]))

    result_counts: dict[str, Counter[str]] = {}
    failures: Counter[str] = Counter()

    languages = ("zh_Hant", "zh_Hans")
    for language_index, language in enumerate(languages):
        language_deadline = started + max_seconds * (language_index + 1) / len(languages)
        output_path = distilled_dir / f"{language}.jsonl"
        if force and output_path.exists():
            output_path.unlink()
        existing_rows = [] if force else list(read_jsonl(output_path))
        existing_keys = {
            (row["metadata"]["parent_prompt_sha256"], row["metadata"]["split"])
            for row in existing_rows
        }
        counts = Counter(row["metadata"]["split"] for row in existing_rows)
        result_counts[language] = counts
        total_target = int(target_by_language[language])
        quotas = split_quotas(total_target, percentages)

        for split in ("valid", "test", "train"):
            for source in _candidate_order(all_sources, split):
                if counts[split] >= quotas[split]:
                    break
                key = (source["metadata"]["prompt_sha256"], split)
                if key in existing_keys:
                    continue
                if time.monotonic() >= language_deadline:
                    break
                prompt = _render_prompt(tokenizer, source, language)
                translated: tuple[str, str] | None = None
                last_errors: list[str] = ["unparsed output"]
                for temperature in (
                    float(settings["temperature"]),
                    float(settings["retry_temperature"]),
                ):
                    output = generate(
                        model,
                        tokenizer,
                        prompt,
                        max_tokens=max_tokens,
                        sampler=make_sampler(
                            temp=temperature,
                            top_p=float(settings["top_p"]) if temperature > 0 else 0.0,
                        ),
                        verbose=False,
                    )
                    parsed = _parse_translation(output)
                    if parsed is None:
                        last_errors = ["unparsed output"]
                        continue
                    last_errors = _validate_translation(source, *parsed, language)
                    if not last_errors:
                        translated = parsed
                        break
                if translated is None:
                    failures["; ".join(last_errors)] += 1
                    continue
                record = _translated_record(
                    source,
                    translated[0],
                    translated[1],
                    language,
                    teacher,
                    base_tokenizer,
                )
                if record["metadata"]["token_count"] > int(data_settings["max_seq_length"]):
                    failures["translated sample too long"] += 1
                    continue
                schema_errors = validate_chat_record(record)
                if schema_errors:
                    failures["schema: " + "; ".join(schema_errors)] += 1
                    continue
                append_jsonl(output_path, record)
                existing_keys.add(key)
                counts[split] += 1
                console.print(
                    f"{language}: {sum(counts.values())}/{total_target} "
                    f"(elapsed {int(time.monotonic() - started)}s)"
                )
            if time.monotonic() >= language_deadline:
                break

    totals = {language: sum(counts.values()) for language, counts in result_counts.items()}
    below_minimum = {language: count for language, count in totals.items() if count < minimum}
    if below_minimum:
        raise RuntimeError(
            f"Distillation did not reach the minimum per language: {below_minimum}; "
            "partial rows were preserved for resumption."
        )
    complete = all(
        totals[language] >= int(target_by_language[language]) for language in target_by_language
    )
    summary = {
        "fingerprint": fingerprint,
        "complete": complete,
        "minimum_complete": True,
        "counts": {language: dict(counts) for language, counts in result_counts.items()},
        "totals": totals,
        "failures": dict(failures),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "teacher_repo": teacher["repo_id"],
        "teacher_revision": teacher["revision"],
    }
    write_json(manifest_dir / "distillation-summary.json", summary)
    write_json(done_path, summary)
    if not complete:
        console.print("[yellow]Time budget reached; minimum trilingual data is available.[/yellow]")
    return summary

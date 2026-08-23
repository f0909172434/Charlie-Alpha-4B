from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Any

from datasets import load_dataset
from rich.console import Console
from transformers import AutoTokenizer

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_text, write_json, write_jsonl
from .sandbox import evaluate_standalone_candidate
from .validators import (
    extract_code_blocks,
    is_contaminated,
    normalize_text,
    validate_chat_record,
    word_ngrams,
)

console = Console()


def deterministic_split(group_id: str, seed: int, percentages: dict[str, int]) -> str:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    train_end = int(percentages["train"])
    valid_end = train_end + int(percentages["valid"])
    if bucket < train_end:
        return "train"
    if bucket < valid_end:
        return "valid"
    return "test"


def split_quotas(total: int, percentages: dict[str, int]) -> dict[str, int]:
    valid = max(1, (total * percentages["valid"] + 99) // 100)
    test = max(1, (total * percentages["test"] + 99) // 100)
    return {"train": total - valid - test, "valid": valid, "test": test}


def _chat_length(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    try:
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        return len(tokens)
    except (AttributeError, ValueError, TypeError):
        merged = "\n".join(f"{item['role']}: {item['content']}" for item in messages)
        return len(tokenizer.encode(merged, add_special_tokens=True))


def _benchmark_references(config: ProjectConfig, size: int) -> list[set[tuple[str, ...]]]:
    references: list[set[tuple[str, ...]]] = []
    datasets = config.sources["datasets"]
    math_source = datasets["math500_eval"]
    math_rows = load_dataset(
        math_source["repo_id"],
        split=math_source["split"],
        revision=math_source["revision"],
    )
    for row in math_rows:
        prompt = row.get("problem") or row.get("question")
        if prompt:
            references.append(word_ngrams(prompt, size))

    gsm_source = datasets["gsm8k_eval"]
    gsm_rows = load_dataset(
        gsm_source["repo_id"],
        name=gsm_source["config"],
        split=gsm_source["split"],
        revision=gsm_source["revision"],
    )
    for row in gsm_rows:
        if row.get("question"):
            references.append(word_ngrams(row["question"], size))

    try:
        from evalplus.data import get_human_eval_plus, get_mbpp_plus

        artifacts = config.sources["evaluation_artifacts"]
        for tasks in (
            get_human_eval_plus(version=artifacts["humaneval_plus"]["version"]),
            get_mbpp_plus(version=artifacts["mbpp_plus"]["version"]),
        ):
            for task in tasks.values():
                prompt = task.get("prompt") or task.get("text")
                if prompt:
                    references.append(word_ngrams(prompt, size))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        console.print(f"[yellow]EvalPlus decontamination references unavailable: {error}[/yellow]")
    return [reference for reference in references if reference]


def _select_verified_math(row: dict[str, Any]) -> str | None:
    generations = row.get("generations") or []
    correct = row.get("correctness_math_verify") or []
    complete = row.get("is_reasoning_complete") or []
    finish_reasons = row.get("finish_reasons") or []
    accepted: list[str] = []
    for index, generation in enumerate(generations):
        is_correct = index < len(correct) and correct[index] is True
        is_complete = index < len(complete) and complete[index] is True
        finished = index >= len(finish_reasons) or finish_reasons[index] in {"stop", "eos"}
        if is_correct and is_complete and finished and isinstance(generation, str):
            accepted.append(generation.strip())
    if not accepted:
        return None
    canonical = row.get("solution")
    if isinstance(canonical, str) and canonical.strip():
        accepted.append(canonical.strip())
    return min(accepted, key=len)


def _compact_code_prompt(row: dict[str, Any], language: str) -> str:
    language_name = "Python 3" if language == "python" else "C++17"
    parts = [
        f"Solve this programming problem in {language_name}. Return a correct, complete, concise "
        f"{language_name} implementation in one code block.",
        f"# {row.get('title') or 'Problem'}",
        str(row.get("description") or "").strip(),
    ]
    if row.get("time_limit") is not None or row.get("memory_limit") is not None:
        parts.append(f"Limits: {row.get('time_limit')} seconds, {row.get('memory_limit')} MB.")
    for heading, field in (("Input", "input_format"), ("Output", "output_format")):
        if row.get(field):
            parts.append(f"## {heading}\n{str(row[field]).strip()}")
    examples = row.get("examples") or []
    if examples:
        rendered = []
        for example in examples:
            rendered.append(
                f"Input:\n```text\n{example['input'].strip()}\n```\n"
                f"Output:\n```text\n{example['output'].strip()}\n```"
            )
        parts.append("## Examples\n" + "\n\n".join(rendered))
    if row.get("note"):
        parts.append(f"## Note\n{str(row['note']).strip()}")
    return "\n\n".join(part for part in parts if part).strip()


def _compact_code_answer(generation: str) -> str:
    if "</think>" in generation:
        final = generation.rsplit("</think>", 1)[-1].strip()
        if final:
            return final
    blocks = extract_code_blocks(generation)
    if not blocks:
        return generation.strip()
    code = max(blocks, key=len)
    language = "cpp" if "#include" in code else "python"
    return f"```{language}\n{code}\n```"


def _record(
    *,
    user: str,
    assistant: str,
    source: dict[str, Any],
    source_id: str,
    source_config: str,
    domain: str,
    language: str,
    split: str,
    token_count: int,
    assistant_token_count: int,
    code_language: str | None = None,
    answer: str | None = None,
    tests: list[dict[str, str]] | None = None,
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_id": source_id,
        "source_repo": source["repo_id"],
        "source_revision": source["revision"],
        "source_config": source_config,
        "source_license": source["license"],
        "domain": domain,
        "language": language,
        "split": split,
        "token_count": token_count,
        "assistant_token_count": assistant_token_count,
        "prompt_sha256": sha256_text(normalize_text(user)),
        "assistant_sha256": sha256_text(assistant),
    }
    if code_language:
        metadata["code_language"] = code_language
    if answer:
        metadata["answer"] = answer
    if tests:
        metadata["tests"] = tests
    if verification:
        metadata["verification"] = verification
    return {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ],
        "metadata": metadata,
    }


def _iter_source(source: dict[str, Any], config_name: str, seed: int) -> Iterator[dict[str, Any]]:
    dataset = load_dataset(
        source["repo_id"],
        name=config_name,
        split=source["split"],
        revision=source["revision"],
        streaming=True,
    )
    yield from dataset.shuffle(seed=seed, buffer_size=10_000)


def _collect_category(
    *,
    rows: Iterable[dict[str, Any]],
    category: str,
    total: int,
    source: dict[str, Any],
    source_config: str,
    tokenizer: Any,
    max_tokens: int,
    seed: int,
    percentages: dict[str, int],
    ngram_size: int,
    overlap_threshold: float,
    benchmark_references: list[set[tuple[str, ...]]],
    seen_prompts: set[str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    quotas = split_quotas(total, percentages)
    counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    rejection: Counter[str] = Counter()
    scanned = 0

    for row in rows:
        scanned += 1
        if category == "math":
            source_id = str(row.get("uuid") or sha256_text(str(row.get("problem"))))
            user = row.get("problem")
            assistant = _select_verified_math(row)
            answer = row.get("answer")
            code_language = None
            tests = None
        else:
            source_id = str(row.get("id") or sha256_text(str(row.get("prompt"))))
            user = _compact_code_prompt(row, category)
            raw_generation = row.get("generation")
            assistant = (
                _compact_code_answer(raw_generation)
                if isinstance(raw_generation, str)
                else raw_generation
            )
            answer = None
            code_language = category
            tests = row.get("public_tests_ms") or row.get("examples")
            expected_fence = "```python" if category == "python" else "```cpp"
            if not isinstance(assistant, str) or expected_fence not in assistant.lower():
                rejection["missing_code_fence"] += 1
                continue
            if row.get("finish_reason") not in {None, "stop", "eos"}:
                rejection["unfinished"] += 1
                continue
            if row.get("interaction_format"):
                rejection["interactive"] += 1
                continue

        if not isinstance(user, str) or not isinstance(assistant, str):
            rejection["missing_content"] += 1
            continue
        split = deterministic_split(source_id, seed, percentages)
        if counts[split] >= quotas[split]:
            continue
        prompt_hash = sha256_text(normalize_text(user))
        if prompt_hash in seen_prompts:
            rejection["duplicate"] += 1
            continue
        if is_contaminated(
            user,
            benchmark_references,
            size=ngram_size,
            threshold=overlap_threshold,
        ):
            rejection["benchmark_overlap"] += 1
            continue
        messages = [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
        token_count = _chat_length(tokenizer, messages)
        if token_count > max_tokens:
            rejection["too_long"] += 1
            continue
        verification = None
        if category != "math":
            code_blocks = extract_code_blocks(assistant)
            if not code_blocks or not tests:
                rejection["no_runnable_tests"] += 1
                continue
            verification = evaluate_standalone_candidate(
                candidate_code=max(code_blocks, key=len),
                language=category,
                tests=tests,
            )
            if not verification.get("passed"):
                rejection[f"sandbox_{verification.get('reason', 'failed')}"] += 1
                continue
        record = _record(
            user=user,
            assistant=assistant,
            source=source,
            source_id=source_id,
            source_config=source_config,
            domain="math" if category == "math" else "code",
            language="en",
            split=split,
            token_count=token_count,
            assistant_token_count=len(tokenizer.encode(assistant, add_special_tokens=False)),
            code_language=code_language,
            answer=answer,
            tests=tests,
            verification=verification,
        )
        errors = validate_chat_record(record)
        if errors:
            rejection["schema"] += 1
            continue
        records.append(record)
        seen_prompts.add(prompt_hash)
        counts[split] += 1
        if all(counts[name] >= quota for name, quota in quotas.items()):
            break

    missing = {name: quotas[name] - counts[name] for name in quotas if counts[name] < quotas[name]}
    if missing:
        raise RuntimeError(
            f"Could not fill {category} quotas after scanning {scanned} rows; missing={missing}, "
            f"rejections={dict(rejection)}"
        )
    console.print(
        f"[green]{category}[/green]: accepted={len(records)} scanned={scanned} "
        f"rejections={dict(rejection)}"
    )
    return records, rejection


def prepare_data(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    data_config = config.section("data")
    processed_dir = config.path_for("processed_dir")
    manifest_dir = config.path_for("manifest_dir")
    preparation_settings = {
        key: data_config[key]
        for key in (
            "max_seq_length",
            "split_percentages",
            "english_targets",
            "decontamination",
        )
    }
    preparation_sources = {
        key: value for key, value in config.sources.items() if key != "evaluation_artifacts"
    }
    fingerprint = canonical_hash(
        {
            "data": preparation_settings,
            "sources": preparation_sources,
            "version": "prepare-v8",
        }
    )
    done_path = processed_dir / ".done.json"
    if done_path.exists() and not force:
        import json

        existing = json.loads(done_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint:
            console.print("[cyan]Prepared data already matches the locked configuration.[/cyan]")
            return existing
        existing_rows = [
            row
            for split in ("train", "valid", "test")
            for row in read_jsonl(processed_dir / f"{split}.jsonl")
        ]
        expected_total = sum(int(value) for value in data_config["english_targets"].values())
        revisions = {
            config.sources["datasets"]["math_train"]["revision"],
            config.sources["datasets"]["code_train"]["revision"],
        }
        if len(existing_rows) == expected_total and all(
            row["metadata"]["token_count"] <= int(data_config["max_seq_length"])
            and row["metadata"]["source_revision"] in revisions
            for row in existing_rows
        ):
            existing["fingerprint"] = fingerprint
            write_json(done_path, existing)
            write_json(manifest_dir / "data-summary.json", existing)
            console.print("[cyan]Prepared data passed compatibility validation.[/cyan]")
            return existing

    model_source = config.sources["models"]["base_hf"]
    console.print("Loading pinned tokenizer and benchmark decontamination references…")
    tokenizer = AutoTokenizer.from_pretrained(
        model_source["repo_id"],
        revision=model_source["revision"],
        trust_remote_code=True,
    )
    decontamination = data_config["decontamination"]
    ngram_size = int(decontamination["ngram_size"])
    benchmark_references = _benchmark_references(config, ngram_size)
    benchmark_union = set().union(*benchmark_references)
    contamination_corpus = [benchmark_union]
    console.print(
        f"Loaded {len(benchmark_references)} benchmark prompts "
        f"({len(benchmark_union)} unique {ngram_size}-grams) for local overlap checks."
    )

    percentages = data_config["split_percentages"]
    seed = int(config.section("project")["seed"])
    max_tokens = int(data_config["max_seq_length"])
    source_defs = config.sources["datasets"]
    seen_prompts: set[str] = set()
    all_records: list[dict[str, Any]] = []
    rejection_summary: dict[str, dict[str, int]] = {}

    math_source = source_defs["math_train"]
    math_records, rejected = _collect_category(
        rows=_iter_source(math_source, math_source["config"], seed),
        category="math",
        total=int(data_config["english_targets"]["math"]),
        source=math_source,
        source_config=math_source["config"],
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        seed=seed,
        percentages=percentages,
        ngram_size=ngram_size,
        overlap_threshold=float(decontamination["overlap_threshold"]),
        benchmark_references=contamination_corpus,
        seen_prompts=seen_prompts,
    )
    all_records.extend(math_records)
    rejection_summary["math"] = dict(rejected)

    code_source = source_defs["code_train"]
    for offset, (category, config_key) in enumerate(
        (("python", "python_config"), ("cpp", "cpp_config")), start=1
    ):
        source_config = code_source[config_key]
        records, rejected = _collect_category(
            rows=_iter_source(code_source, source_config, seed + offset),
            category=category,
            total=int(data_config["english_targets"][category]),
            source=code_source,
            source_config=source_config,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            seed=seed,
            percentages=percentages,
            ngram_size=ngram_size,
            overlap_threshold=float(decontamination["overlap_threshold"]),
            benchmark_references=contamination_corpus,
            seen_prompts=seen_prompts,
        )
        all_records.extend(records)
        rejection_summary[category] = dict(rejected)

    split_rows = {
        split: sorted(
            (row for row in all_records if row["metadata"]["split"] == split),
            key=lambda row: row["metadata"]["prompt_sha256"],
        )
        for split in ("train", "valid", "test")
    }
    for split, records in split_rows.items():
        write_jsonl(processed_dir / f"{split}.jsonl", records)

    manifest_rows = []
    for row in all_records:
        metadata = row["metadata"]
        manifest_rows.append(
            {
                key: metadata[key]
                for key in (
                    "source_id",
                    "source_repo",
                    "source_revision",
                    "source_config",
                    "source_license",
                    "domain",
                    "language",
                    "split",
                    "token_count",
                    "assistant_token_count",
                    "prompt_sha256",
                    "assistant_sha256",
                )
            }
        )
    write_jsonl(manifest_dir / "source-records.jsonl", manifest_rows)
    summary = {
        "fingerprint": fingerprint,
        "record_count": len(all_records),
        "split_counts": {name: len(rows) for name, rows in split_rows.items()},
        "domain_counts": dict(Counter(row["metadata"]["domain"] for row in all_records)),
        "code_language_counts": dict(
            Counter(
                row["metadata"].get("code_language")
                for row in all_records
                if row["metadata"]["domain"] == "code"
            )
        ),
        "assistant_tokens": sum(row["metadata"]["assistant_token_count"] for row in all_records),
        "benchmark_reference_count": len(benchmark_references),
        "rejections": rejection_summary,
    }
    write_json(manifest_dir / "data-summary.json", summary)
    write_json(done_path, summary)
    return summary


def load_processed(config: ProjectConfig) -> dict[str, list[dict[str, Any]]]:
    directory = config.path_for("processed_dir")
    return {
        split: list(read_jsonl(directory / f"{split}.jsonl"))
        for split in ("train", "valid", "test")
    }

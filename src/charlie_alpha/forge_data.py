from __future__ import annotations

import gc
import json
import math
import re
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from lingua import Language, LanguageDetectorBuilder
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from opencc import OpenCC
from rich.console import Console
from transformers import AutoTokenizer

from .config import ProjectConfig
from .io_utils import (
    append_jsonl,
    canonical_hash,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from .mixer import _convert_prose
from .validators import (
    has_target_script,
    normalize_text,
    translation_preserves_source,
    validate_chat_record,
    word_ngrams,
)

console = Console()

_LANGUAGE_DETECTOR = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH,
    Language.CROATIAN,
    Language.RUSSIAN,
    Language.TAGALOG,
    Language.GERMAN,
    Language.FRENCH,
    Language.SPANISH,
    Language.PORTUGUESE,
    Language.POLISH,
    Language.ITALIAN,
    Language.TURKISH,
    Language.INDONESIAN,
    Language.DUTCH,
    Language.VIETNAMESE,
).build()

_STRIP_FOR_LANGUAGE_RE = re.compile(
    r"```.*?```|\\\[.*?\\\]|\$\$.*?\$\$|(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)|"
    r"\\\(.*?\\\)|https?://\S+",
    re.DOTALL,
)

_PROTECTED_RE = re.compile(
    r"```.*?```|\\\[.*?\\\]|\$\$.*?\$\$|(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)|"
    r"\\\(.*?\\\)|https?://\S+|(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?",
    re.DOTALL,
)

_ITEM_LABEL_RE = re.compile(
    r"^\s*(?:(?:example|problem|exercise)\s+\d+(?:\.\d+)*[.:)]?\s+|"
    r"\d+(?:\.\d+)*[.)]\s+)",
    re.IGNORECASE,
)


def _category(row: dict[str, Any]) -> str:
    metadata = row["metadata"]
    return metadata.get("code_language") if metadata["domain"] == "code" else "math"


def _candidate_id(row: dict[str, Any]) -> str:
    metadata = row["metadata"]
    return canonical_hash(
        {
            "prompt": metadata["prompt_sha256"],
            "assistant": metadata["assistant_sha256"],
            "split": metadata["split"],
        }
    )[:24]


def _tokenize_chat(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[list[int], int]:
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )
    offset = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    return list(tokens), len(offset)


def _looks_english(value: str) -> tuple[bool, str]:
    prose = _STRIP_FOR_LANGUAGE_RE.sub(" ", value)
    prose = re.sub(r"[_{}#*\\|<>:=+\-/]", " ", prose)
    prose = re.sub(r"\s+", " ", prose).strip()
    if len(prose) < 24:
        ascii_letters = sum(character.isascii() and character.isalpha() for character in prose)
        letters = sum(character.isalpha() for character in prose)
        accepted = bool(letters and ascii_letters / letters >= 0.95)
        return accepted, "short-ascii" if accepted else "short-unknown"
    detected = _LANGUAGE_DETECTOR.detect_language_of(prose[:6000])
    name = detected.name.lower() if detected is not None else "unknown"
    return detected == Language.ENGLISH, name


def prepare_forge_candidates(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    forge_dir = config.path_for("forge_dir")
    output_path = forge_dir / "candidates.jsonl"
    manifest_path = forge_dir / "candidates.json"
    processed_dir = config.path_for("v1_processed_dir")
    source_paths = [processed_dir / f"{split}.jsonl" for split in ("train", "valid")]
    fingerprint = canonical_hash(
        {
            "sources": {path.name: sha256_file(path) for path in source_paths},
            "base": config.sources["models"]["research_base_mlx_4bit"],
            "max_seq_length": config.section("forge")["max_seq_length"],
            "language_filter": "lingua-en-v1",
            "version": 1,
        }
    )
    if output_path.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == fingerprint:
            return manifest

    base_source = config.sources["models"]["research_base_mlx_4bit"]
    model_path = snapshot_download(
        repo_id=base_source["repo_id"], revision=base_source["revision"]
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    maximum = int(config.section("forge")["max_seq_length"])
    accepted: list[dict[str, Any]] = []
    rejection: Counter[str] = Counter()
    detected_languages: Counter[str] = Counter()
    seen_ids: set[str] = set()

    for source_path in source_paths:
        for source in read_jsonl(source_path):
            prompt = source["messages"][0]["content"]
            english, detected = _looks_english(prompt)
            detected_languages[detected] += 1
            if not english:
                rejection[f"language:{detected}"] += 1
                continue
            tokens, offset = _tokenize_chat(tokenizer, source["messages"])
            if len(tokens) > maximum:
                rejection["too_long"] += 1
                continue
            if offset >= len(tokens):
                rejection["empty_assistant"] += 1
                continue
            row = deepcopy(source)
            row_id = _candidate_id(row)
            if row_id in seen_ids:
                rejection["duplicate"] += 1
                continue
            seen_ids.add(row_id)
            row["metadata"].update(
                {
                    "candidate_id": row_id,
                    "category": _category(row),
                    "detected_language": detected,
                    "token_count_qwen35": len(tokens),
                    "prompt_offset_qwen35": offset,
                    "assistant_token_count_qwen35": len(tokens) - offset,
                }
            )
            accepted.append(row)

    accepted.sort(
        key=lambda row: (
            row["metadata"]["split"],
            row["metadata"]["category"],
            row["metadata"]["candidate_id"],
        )
    )
    write_jsonl(output_path, accepted)
    by_split_category: Counter[str] = Counter(
        f"{row['metadata']['split']}:{row['metadata']['category']}" for row in accepted
    )
    manifest = {
        "fingerprint": fingerprint,
        "records": len(accepted),
        "by_split_category": dict(sorted(by_split_category.items())),
        "detected_languages": dict(sorted(detected_languages.items())),
        "rejections": dict(sorted(rejection.items())),
        "output_sha256": sha256_file(output_path),
    }
    write_json(manifest_path, manifest)
    return manifest


def _assistant_losses(model: Any, tokens: list[int], offset: int) -> np.ndarray:
    inputs = mx.array(np.asarray(tokens[:-1], dtype=np.int32))[None, :]
    targets = mx.array(np.asarray(tokens[1:], dtype=np.int32))[None, :]
    logits = model(inputs)
    losses = nn.losses.cross_entropy(logits, targets)[0, max(0, offset - 1) :]
    losses = losses.astype(mx.float32)
    mx.eval(losses)
    result = np.asarray(losses)
    del inputs, targets, logits, losses
    return result


def _score_one_model(
    *,
    config: ProjectConfig,
    candidates: list[dict[str, Any]],
    model_key: str,
    output_path: Path,
    deadline: float,
    force: bool,
) -> dict[str, Any]:
    if force and output_path.exists():
        output_path.unlink()
    source = config.sources["models"][model_key]
    candidate_hashes = {
        row["metadata"]["candidate_id"]: canonical_hash(row) for row in candidates
    }
    existing = [] if force else list(read_jsonl(output_path))
    existing = [
        row
        for row in existing
        if row.get("model_repo") == source["repo_id"]
        and row.get("model_revision") == source["revision"]
        and row.get("record_sha256") == candidate_hashes.get(row.get("candidate_id"))
    ]
    write_jsonl(output_path, existing)
    completed = {row["candidate_id"]: row for row in existing}
    pending = [
        row for row in candidates if row["metadata"]["candidate_id"] not in completed
    ]
    if not pending or time.monotonic() >= deadline:
        return {"completed": len(completed), "total": len(candidates)}

    model_path = snapshot_download(repo_id=source["repo_id"], revision=source["revision"])
    model, tokenizer = load(model_path, tokenizer_config={"trust_remote_code": True})
    model.eval()
    for index, row in enumerate(pending, start=1):
        if time.monotonic() >= deadline:
            break
        tokens, offset = _tokenize_chat(tokenizer, row["messages"])
        losses = _assistant_losses(model, tokens, offset)
        result = {
            "candidate_id": row["metadata"]["candidate_id"],
            "record_sha256": canonical_hash(row),
            "model_repo": source["repo_id"],
            "model_revision": source["revision"],
            "token_count": len(tokens),
            "tokens_sha256": canonical_hash(tokens),
            "prompt_offset": offset,
            "mean_loss": round(float(losses.mean()), 7),
            "token_losses": [round(float(value), 6) for value in losses],
        }
        append_jsonl(output_path, result)
        completed[result["candidate_id"]] = result
        if index % 10 == 0:
            console.print(
                f"{model_key}: {len(completed)}/{len(candidates)} scored "
                f"(peak {mx.get_peak_memory() / 1e9:.2f} GB)"
            )
        if index % 25 == 0:
            mx.clear_cache()
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    return {"completed": len(completed), "total": len(candidates)}


def score_forge_candidates(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    prepare_forge_candidates(config, force=force)
    candidates = list(read_jsonl(config.path_for("forge_dir") / "candidates.jsonl"))
    score_dir = config.path_for("score_dir")
    score_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + int(config.section("overnight_v2")["scoring_seconds"])
    results: dict[str, Any] = {}
    for label, model_key in (
        ("student", "research_base_mlx_4bit"),
        ("teacher", "teacher_mlx_4bit"),
    ):
        results[label] = _score_one_model(
            config=config,
            candidates=candidates,
            model_key=model_key,
            output_path=score_dir / f"{label}.jsonl",
            deadline=deadline,
            force=force,
        )
    results["elapsed_seconds"] = round(time.monotonic() - started, 2)
    results["complete"] = all(
        result["completed"] == result["total"]
        for result in (results["student"], results["teacher"])
    )
    write_json(score_dir / "summary.json", results)
    return results


def _allocate(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {key: total * float(value) for key, value in ratios.items()}
    allocated = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(allocated.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - allocated[key]), key))
    for key in order[:remaining]:
        allocated[key] += 1
    return allocated


def _selection_metrics(
    row: dict[str, Any], student: dict[str, Any], teacher: dict[str, Any]
) -> dict[str, Any]:
    candidate_id = row["metadata"].get("candidate_id") or _candidate_id(row)
    if not student.get("tokens_sha256") or (
        student["tokens_sha256"] != teacher.get("tokens_sha256")
    ):
        raise RuntimeError(f"Teacher/student token IDs changed for {candidate_id}")
    student_losses = np.asarray(student["token_losses"], dtype=np.float32)
    teacher_losses = np.asarray(teacher["token_losses"], dtype=np.float32)
    if student_losses.shape != teacher_losses.shape:
        raise RuntimeError(f"Teacher/student token alignment changed for {candidate_id}")
    deltas = student_losses - teacher_losses
    positive = np.maximum(deltas, 0.0)
    teacher_mean = float(teacher_losses.mean())
    mean_excess = float(deltas.mean())
    positive_mass = float(positive.mean())
    # Prefer examples on which the teacher is both better and confident. The small student-loss
    # term gives deterministic ordering when the teacher and student are nearly tied.
    utility = positive_mass / (1.0 + teacher_mean) + 0.02 * float(student_losses.mean())
    return {
        "student_loss": round(float(student_losses.mean()), 7),
        "teacher_loss": round(teacher_mean, 7),
        "mean_excess_loss": round(mean_excess, 7),
        "positive_excess_mass": round(positive_mass, 7),
        "positive_token_fraction": round(float((deltas > 0).mean()), 7),
        "utility": round(utility, 9),
        "token_deltas": deltas,
    }


def _jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _balanced_mmr(
    rows: list[dict[str, Any]], count: int, diversity_penalty: float
) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise RuntimeError(f"Need {count} candidates but only {len(rows)} are available")
    ordered = sorted(
        rows,
        key=lambda row: (-float(row["selection"]["utility"]), row["metadata"]["candidate_id"]),
    )
    if len(ordered) == 1:
        return ordered[:count]
    rank_score = {
        row["metadata"]["candidate_id"]: 1.0 - index / (len(ordered) - 1)
        for index, row in enumerate(ordered)
    }
    shingles = {
        row["metadata"]["candidate_id"]: word_ngrams(row["messages"][0]["content"], 3)
        for row in ordered
    }
    selected: list[dict[str, Any]] = []
    remaining = list(ordered)
    while len(selected) < count:
        def score(candidate: dict[str, Any]) -> tuple[float, str]:
            candidate_id = candidate["metadata"]["candidate_id"]
            redundancy = max(
                (
                    _jaccard(
                        shingles[candidate_id],
                        shingles[item["metadata"]["candidate_id"]],
                    )
                    for item in selected
                ),
                default=0.0,
            )
            return (
                rank_score[candidate_id] - diversity_penalty * redundancy,
                candidate_id,
            )

        chosen = max(remaining, key=score)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def _scored_candidates(config: ProjectConfig) -> list[dict[str, Any]]:
    candidate_path = config.path_for("forge_dir") / "candidates.jsonl"
    score_dir = config.path_for("score_dir")
    candidates = list(read_jsonl(candidate_path))
    student = {row["candidate_id"]: row for row in read_jsonl(score_dir / "student.jsonl")}
    teacher = {row["candidate_id"]: row for row in read_jsonl(score_dir / "teacher.jsonl")}
    complete: list[dict[str, Any]] = []
    for row in candidates:
        candidate_id = row["metadata"]["candidate_id"]
        if candidate_id not in student or candidate_id not in teacher:
            continue
        expected_hash = canonical_hash(row)
        if student[candidate_id]["record_sha256"] != expected_hash:
            continue
        if teacher[candidate_id]["record_sha256"] != expected_hash:
            continue
        enriched = deepcopy(row)
        metrics = _selection_metrics(row, student[candidate_id], teacher[candidate_id])
        metrics["token_deltas"] = [round(float(value), 6) for value in metrics["token_deltas"]]
        enriched["selection"] = metrics
        complete.append(enriched)
    return complete


def select_forge_sources(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    output_path = config.path_for("forge_dir") / "selection.json"
    settings = config.section("forge")
    fingerprint = canonical_hash(
        {
            "student": sha256_file(config.path_for("score_dir") / "student.jsonl"),
            "teacher": sha256_file(config.path_for("score_dir") / "teacher.jsonl"),
            "forge": settings,
            "version": 1,
        }
    )
    if output_path.exists() and not force:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint:
            return existing
    scored = _scored_candidates(config)
    candidate_count = sum(
        1 for _ in read_jsonl(config.path_for("forge_dir") / "candidates.jsonl")
    )
    if len(scored) != candidate_count:
        raise RuntimeError(
            f"Forge scoring is incomplete: {len(scored)}/{candidate_count}; resume scoring"
        )
    ratios = settings["group_categories"]
    train_targets = _allocate(int(settings["train_semantic_groups"]), ratios)
    valid_targets = _allocate(int(settings["valid_semantic_groups"]), ratios)
    translation_max = int(settings["translation_source_max_length"])
    diversity = float(settings["diversity_penalty"])
    pools: dict[str, dict[str, list[str]]] = {"train": {}, "valid": {}}
    pool_targets: dict[str, dict[str, int]] = {"train": {}, "valid": {}}

    for split, targets in (("train", train_targets), ("valid", valid_targets)):
        for category, target in targets.items():
            reserve = max(1, math.ceil(target * 0.25))
            length_limit = (
                translation_max
                if split == "train"
                else int(settings["max_seq_length"])
            )
            eligible = [
                row
                for row in scored
                if row["metadata"]["split"] == split
                and row["metadata"]["category"] == category
                and int(row["metadata"]["token_count_qwen35"]) <= length_limit
            ]
            if len(eligible) < target:
                raise RuntimeError(
                    f"Need {target} {split}/{category} translation sources but only "
                    f"{len(eligible)} fit"
                )
            pool_count = min(len(eligible), target + reserve)
            chosen = _balanced_mmr(eligible, pool_count, diversity)
            pools[split][category] = [row["metadata"]["candidate_id"] for row in chosen]
            pool_targets[split][category] = target

    selected_ids = {
        candidate_id
        for split_pools in pools.values()
        for category_ids in split_pools.values()
        for candidate_id in category_ids
    }
    selected_rows = [
        row for row in scored if row["metadata"]["candidate_id"] in selected_ids
    ]
    selection = {
        "fingerprint": fingerprint,
        "method": "teacher-student-excess-loss-balanced-mmr",
        "targets": pool_targets,
        "pools": pools,
        "records": {
            row["metadata"]["candidate_id"]: {
                "category": row["metadata"]["category"],
                "split": row["metadata"]["split"],
                "utility": row["selection"]["utility"],
                "student_loss": row["selection"]["student_loss"],
                "teacher_loss": row["selection"]["teacher_loss"],
                "mean_excess_loss": row["selection"]["mean_excess_loss"],
            }
            for row in selected_rows
        },
    }
    write_json(output_path, selection)
    return selection


def _protect(value: str, prefix: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        placeholder = f"<CA_{prefix}_{len(protected):04d}>"
        protected[placeholder] = match.group(0)
        return placeholder

    return _PROTECTED_RE.sub(replace, value), protected


def _restore(value: str, protected: dict[str, str]) -> str | None:
    restored = value
    for placeholder, original in protected.items():
        if restored.count(placeholder) != 1:
            return None
        restored = restored.replace(placeholder, original)
    if re.search(r"<CA_[QA]_\d{4}>", restored):
        return None
    return restored.strip()


def _translation_request(source: dict[str, Any]) -> tuple[str, dict[str, str], dict[str, str]]:
    source_question = _ITEM_LABEL_RE.sub("", source["messages"][0]["content"], count=1)
    question, question_map = _protect(source_question, "Q")
    answer, answer_map = _protect(source["messages"][-1]["content"], "A")
    request = f"""Translate only the prose below into natural Simplified Chinese.

Rules:
- Copy every <CA_Q_####> and <CA_A_####> placeholder exactly once and unchanged.
- Do not solve, shorten, expand, or reorder the content.
- Return exactly the two XML sections, with no preface.

<QUESTION_SOURCE>
{question}
</QUESTION_SOURCE>
<ANSWER_SOURCE>
{answer}
</ANSWER_SOURCE>

<QUESTION_TRANSLATION>translated question</QUESTION_TRANSLATION>
<ANSWER_TRANSLATION>translated answer</ANSWER_TRANSLATION>"""
    return request, question_map, answer_map


def _parse_translation(value: str) -> tuple[str, str] | None:
    if "<QUESTION_TRANSLATION>" not in value:
        value = f"<QUESTION_TRANSLATION>{value}"
    question = re.search(
        r"<QUESTION_TRANSLATION>(.*?)</QUESTION_TRANSLATION>", value, flags=re.DOTALL
    )
    answer = re.search(
        r"<ANSWER_TRANSLATION>(.*?)</ANSWER_TRANSLATION>", value, flags=re.DOTALL
    )
    if question is None or answer is None:
        return None
    if not question.group(1).strip() or not answer.group(1).strip():
        return None
    return question.group(1).strip(), answer.group(1).strip()


def _translation_prompt(tokenizer: Any, request: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a lossless technical translator. Follow placeholders and output format "
                "exactly. Never answer the source problem."
            ),
        },
        {"role": "user", "content": request},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{prompt}<QUESTION_TRANSLATION>\n"


def _translated_pair(
    source: dict[str, Any], output: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]] | None:
    _, question_map, answer_map = _translation_request(source)
    parsed = _parse_translation(output)
    if parsed is None:
        return None
    question = _restore(parsed[0], question_map)
    answer = _restore(parsed[1], answer_map)
    if question is None or answer is None:
        return None
    if not has_target_script(f"{question}\n{answer}", "zh_Hans"):
        return None
    source_question = _ITEM_LABEL_RE.sub("", source["messages"][0]["content"], count=1)
    if not translation_preserves_source(source_question, question)[0]:
        return None
    if not translation_preserves_source(source["messages"][-1]["content"], answer)[0]:
        return None
    simplified = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    converter = OpenCC("s2twp")
    traditional = [
        {"role": message["role"], "content": _convert_prose(message["content"], converter)}
        for message in simplified
    ]
    if not has_target_script(
        "\n".join(message["content"] for message in traditional), "zh_Hant"
    ):
        return None
    for original, translated in zip(simplified, traditional, strict=True):
        if not translation_preserves_source(original["content"], translated["content"])[0]:
            return None
    return simplified, traditional


def distill_forge_translations(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    selection = select_forge_sources(config)
    candidates = {
        row["metadata"]["candidate_id"]: row
        for row in read_jsonl(config.path_for("forge_dir") / "candidates.jsonl")
    }
    bucket_order = [
        (split, category)
        for split in ("valid", "train")
        for category in ("math", "python", "cpp")
    ]
    requirements = {
        bucket: int(selection["targets"][bucket[0]][bucket[1]])
        for bucket in bucket_order
    }
    pools = {
        bucket: list(selection["pools"][bucket[0]][bucket[1]])
        for bucket in bucket_order
    }
    pool_ids = {candidate_id for ids in pools.values() for candidate_id in ids}
    output_dir = config.path_for("translation_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "translations.jsonl"
    if force and output_path.exists():
        output_path.unlink()
    fingerprint = canonical_hash(
        {
            "selection": selection["fingerprint"],
            "teacher": config.sources["models"]["teacher_mlx_4bit"],
            "translation": {
                key: config.section("forge")[key]
                for key in (
                    "translation_max_new_tokens",
                    "teacher_temperature",
                )
            },
            "placeholder_version": 3,
        }
    )
    existing = [
        row
        for row in read_jsonl(output_path)
        if row.get("translation_fingerprint") == fingerprint
    ]
    write_jsonl(output_path, existing)
    completed = {row["candidate_id"] for row in existing}
    success_counts = {
        bucket: sum(candidate_id in completed for candidate_id in pools[bucket])
        for bucket in bucket_order
    }

    def requirement_complete() -> bool:
        return all(
            success_counts[bucket] >= requirements[bucket] for bucket in bucket_order
        )

    target_total = sum(requirements.values())
    if requirement_complete():
        summary = {
            "complete": True,
            "completed": target_total,
            "accepted_pool_records": len(completed & pool_ids),
            "target": target_total,
            "pool": len(pool_ids),
            "by_bucket": {
                f"{split}:{category}": success_counts[(split, category)]
                for split, category in bucket_order
            },
            "failures": {},
            "elapsed_seconds": 0.0,
            "fingerprint": fingerprint,
        }
        write_json(output_dir / "summary.json", summary)
        return summary

    pending: list[tuple[tuple[str, str], str]] = []
    for reserve_phase in (False, True):
        for bucket in bucket_order:
            target = requirements[bucket]
            ids = pools[bucket][target:] if reserve_phase else pools[bucket][:target]
            pending.extend(
                (bucket, candidate_id)
                for candidate_id in ids
                if candidate_id not in completed
            )

    teacher = config.sources["models"]["teacher_mlx_4bit"]
    teacher_path = snapshot_download(repo_id=teacher["repo_id"], revision=teacher["revision"])
    model, tokenizer = load(teacher_path, tokenizer_config={"trust_remote_code": True})
    model.eval()
    settings = config.section("forge")
    started = time.monotonic()
    deadline = started + int(settings["translation_max_seconds"])
    failures: Counter[str] = Counter()
    for bucket, candidate_id in pending:
        if success_counts[bucket] >= requirements[bucket]:
            continue
        if time.monotonic() >= deadline:
            break
        source = candidates[candidate_id]
        request, _, _ = _translation_request(source)
        translated = None
        for attempt in range(2):
            attempt_request = request
            if attempt:
                attempt_request += (
                    "\nThis is a strict retry: copy every placeholder exactly once, close the "
                    "QUESTION_TRANSLATION section, and include ANSWER_TRANSLATION."
                )
            retry_prompt = _translation_prompt(tokenizer, attempt_request)
            output = generate(
                model,
                tokenizer,
                retry_prompt,
                max_tokens=int(settings["translation_max_new_tokens"]),
                sampler=make_sampler(temp=float(settings["teacher_temperature"])),
                verbose=False,
            )
            translated = _translated_pair(source, output)
            if translated is not None:
                break
        if translated is None:
            failures[source["metadata"]["category"]] += 1
            continue
        result = {
            "candidate_id": candidate_id,
            "source_sha256": canonical_hash(source),
            "translation_fingerprint": fingerprint,
            "teacher_repo": teacher["repo_id"],
            "teacher_revision": teacher["revision"],
            "zh_Hans": translated[0],
            "zh_Hant": translated[1],
        }
        append_jsonl(output_path, result)
        completed.add(candidate_id)
        success_counts[bucket] += 1
        usable = sum(
            min(success_counts[item], requirements[item]) for item in bucket_order
        )
        console.print(f"translations: {usable}/{target_total}")
        if requirement_complete():
            break

    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    summary = {
        "complete": requirement_complete(),
        "completed": sum(
            min(success_counts[bucket], requirements[bucket]) for bucket in bucket_order
        ),
        "accepted_pool_records": len(completed & pool_ids),
        "target": target_total,
        "pool": len(pool_ids),
        "by_bucket": {
            f"{split}:{category}": success_counts[(split, category)]
            for split, category in bucket_order
        },
        "failures": dict(failures),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "fingerprint": fingerprint,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def _selective_target_indices(row: dict[str, Any], settings: dict[str, Any]) -> list[int]:
    deltas = np.asarray(row["selection"]["token_deltas"], dtype=np.float32)
    floor = float(settings["excess_loss_floor"])
    positive = np.flatnonzero(deltas > floor)
    keep = math.ceil(len(positive) * float(settings["selective_keep_fraction"]))
    ranked = sorted(positive.tolist(), key=lambda index: (-float(deltas[index]), index))
    selected = set(ranked[:keep])
    final_floor = min(int(settings["final_token_floor"]), len(deltas))
    selected.update(range(len(deltas) - final_floor, len(deltas)))
    offset = int(row["metadata"]["prompt_offset_qwen35"])
    return sorted(offset - 1 + index for index in selected)


def _record_for_training(
    *,
    source: dict[str, Any],
    messages: list[dict[str, str]],
    language: str,
    semantic_group_id: str,
    microstep_slot: int,
    loss_weight: float,
    tokenizer: Any,
    settings: dict[str, Any],
    selective: bool,
) -> dict[str, Any]:
    tokens, offset = _tokenize_chat(tokenizer, messages)
    if len(tokens) > int(settings["max_seq_length"]):
        raise RuntimeError(
            f"Translated record {source['metadata']['candidate_id']} exceeds the token limit"
        )
    metadata = deepcopy(source["metadata"])
    metadata.update(
        {
            "language": language,
            "semantic_group_id": semantic_group_id,
            "microstep_slot": microstep_slot,
            "loss_weight": round(loss_weight, 8),
            "token_count_qwen35": len(tokens),
            "prompt_offset_qwen35": offset,
            "assistant_token_count_qwen35": len(tokens) - offset,
            "prompt_sha256": sha256_text(normalize_text(messages[0]["content"])),
            "assistant_sha256": sha256_text(messages[-1]["content"]),
            "forge_selective_loss": selective,
        }
    )
    if language != "en":
        metadata["parent_candidate_id"] = source["metadata"]["candidate_id"]
    if selective:
        metadata["selective_target_indices"] = _selective_target_indices(source, settings)
    record = {"messages": messages, "metadata": metadata}
    schema_errors = validate_chat_record(record)
    if schema_errors:
        raise RuntimeError(f"Invalid Forge record: {schema_errors}")
    return record


def _smooth_category_schedule(counts: dict[str, int]) -> list[str]:
    total = sum(counts.values())
    used = dict.fromkeys(counts, 0)
    schedule: list[str] = []
    for position in range(total):
        available = [key for key in counts if used[key] < counts[key]]
        category = max(
            available,
            key=lambda key: (
                counts[key] * (position + 1) / total - used[key],
                -list(counts).index(key),
            ),
        )
        schedule.append(category)
        used[category] += 1
    return schedule


def _ratio(values: Counter[str]) -> dict[str, float]:
    total = sum(values.values())
    return {key: round(value / total, 6) for key, value in sorted(values.items())}


def _write_public_forge_manifests(
    config: ProjectConfig,
    manifest: dict[str, Any],
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
) -> None:
    public_dir = config.root / "data" / "manifests" / "v2"
    records: list[dict[str, Any]] = []
    for split, rows in (("train", train_rows), ("valid", valid_rows)):
        for row in rows:
            metadata = row["metadata"]
            records.append(
                {
                    key: metadata.get(key)
                    for key in (
                        "candidate_id",
                        "parent_candidate_id",
                        "source_id",
                        "source_repo",
                        "source_revision",
                        "source_license",
                        "category",
                        "domain",
                        "code_language",
                        "language",
                        "semantic_group_id",
                        "microstep_slot",
                        "loss_weight",
                        "token_count_qwen35",
                        "assistant_token_count_qwen35",
                        "prompt_sha256",
                        "assistant_sha256",
                    )
                    if metadata.get(key) is not None
                }
                | {
                    "split": split,
                    "selective_target_token_count": len(
                        metadata.get("selective_target_indices", [])
                    ),
                }
            )
    write_jsonl(public_dir / "forge-records.jsonl", records)
    public_summary = {
        key: value
        for key, value in manifest.items()
        if key not in {"translation_lengths"}
    }
    public_summary.update(
        {
            "student_score_sha256": sha256_file(
                config.path_for("score_dir") / "student.jsonl"
            ),
            "teacher_score_sha256": sha256_file(
                config.path_for("score_dir") / "teacher.jsonl"
            ),
            "selection_sha256": sha256_file(
                config.path_for("forge_dir") / "selection.json"
            ),
            "translation_sha256": sha256_file(
                config.path_for("translation_dir") / "translations.jsonl"
            ),
            "record_manifest_sha256": sha256_file(public_dir / "forge-records.jsonl"),
            "policy": "metadata-and-hashes-only-no-training-text",
        }
    )
    write_json(public_dir / "forge-summary.json", public_summary)


def build_forge_data(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    final_dir = config.path_for("final_dir")
    done_path = final_dir / ".done.json"
    candidate_path = config.path_for("forge_dir") / "candidates.jsonl"
    selection_path = config.path_for("forge_dir") / "selection.json"
    translation_path = config.path_for("translation_dir") / "translations.jsonl"
    translation_summary_path = config.path_for("translation_dir") / "summary.json"
    required = [
        candidate_path,
        selection_path,
        translation_path,
        translation_summary_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing Forge inputs: {missing}")
    settings = config.section("forge")
    fingerprint = canonical_hash(
        {
            "candidates": sha256_file(candidate_path),
            "selection": sha256_file(selection_path),
            "translations": sha256_file(translation_path),
            "forge": settings,
            "version": 1,
        }
    )
    if done_path.exists() and not force:
        existing = json.loads(done_path.read_text(encoding="utf-8"))
        public_summary = config.root / "data" / "manifests" / "v2" / "forge-summary.json"
        public_records = config.root / "data" / "manifests" / "v2" / "forge-records.jsonl"
        if (
            existing.get("fingerprint") == fingerprint
            and public_summary.exists()
            and public_records.exists()
        ):
            return existing

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    raw_candidates = {
        row["metadata"]["candidate_id"]: row for row in read_jsonl(candidate_path)
    }
    scored = {
        row["metadata"]["candidate_id"]: row for row in _scored_candidates(config)
    }
    translation_summary = json.loads(translation_summary_path.read_text(encoding="utf-8"))
    translation_fingerprint = translation_summary.get("fingerprint")
    translations: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(translation_path):
        candidate_id = row.get("candidate_id")
        if candidate_id not in scored:
            continue
        if row.get("translation_fingerprint") != translation_fingerprint:
            continue
        if row.get("source_sha256") != canonical_hash(raw_candidates[candidate_id]):
            continue
        translations[candidate_id] = row
    base_source = config.sources["models"]["research_base_mlx_4bit"]
    model_path = snapshot_download(
        repo_id=base_source["repo_id"], revision=base_source["revision"]
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    maximum = int(settings["max_seq_length"])
    translation_lengths: dict[str, dict[str, int]] = {}
    for candidate_id, translation in translations.items():
        translation_lengths[candidate_id] = {
            language: len(_tokenize_chat(tokenizer, translation[language])[0])
            for language in ("zh_Hans", "zh_Hant")
        }

    chosen: dict[str, dict[str, list[dict[str, Any]]]] = {"train": {}, "valid": {}}
    for split in ("train", "valid"):
        for category in ("math", "python", "cpp"):
            target = int(selection["targets"][split][category])
            available = [
                scored[candidate_id]
                for candidate_id in selection["pools"][split][category]
                if candidate_id in scored
                and candidate_id in translations
                and max(translation_lengths[candidate_id].values()) <= maximum
            ]
            if len(available) < target:
                raise RuntimeError(
                    f"Only {len(available)}/{target} valid {split}/{category} translations; "
                    "resume Forge distillation"
                )
            chosen[split][category] = available[:target]

    chosen_train_ids = {
        row["metadata"]["candidate_id"]
        for category_rows in chosen["train"].values()
        for row in category_rows
    }
    replay_per_group = int(settings["english_replay_per_group"])
    replay: dict[str, list[dict[str, Any]]] = {}
    for category in ("math", "python", "cpp"):
        target = len(chosen["train"][category]) * replay_per_group
        eligible = [
            row
            for row in scored.values()
            if row["metadata"]["split"] == "train"
            and row["metadata"]["category"] == category
            and row["metadata"]["candidate_id"] not in chosen_train_ids
        ]
        replay[category] = _balanced_mmr(
            eligible, target, float(settings["diversity_penalty"])
        )

    desired = settings["language_gradient_ratios"]
    observed_frequency = {"en": 6 / 8, "zh_Hans": 1 / 8, "zh_Hant": 1 / 8}
    language_weights = {
        language: float(desired[language]) / observed_frequency[language]
        for language in observed_frequency
    }
    category_counts = {
        category: len(rows) for category, rows in chosen["train"].items()
    }
    category_schedule = _smooth_category_schedule(category_counts)
    category_offsets = Counter()
    train_rows: list[dict[str, Any]] = []
    for group_index, category in enumerate(category_schedule):
        local_index = category_offsets[category]
        category_offsets[category] += 1
        source = chosen["train"][category][local_index]
        candidate_id = source["metadata"]["candidate_id"]
        group_id = f"forge-{group_index:03d}-{candidate_id}"
        translated = translations[candidate_id]
        train_rows.append(
            _record_for_training(
                source=source,
                messages=deepcopy(source["messages"]),
                language="en",
                semantic_group_id=group_id,
                microstep_slot=0,
                loss_weight=language_weights["en"],
                tokenizer=tokenizer,
                settings=settings,
                selective=True,
            )
        )
        for slot, language in ((1, "zh_Hans"), (2, "zh_Hant")):
            train_rows.append(
                _record_for_training(
                    source=source,
                    messages=deepcopy(translated[language]),
                    language=language,
                    semantic_group_id=group_id,
                    microstep_slot=slot,
                    loss_weight=language_weights[language],
                    tokenizer=tokenizer,
                    settings=settings,
                    selective=False,
                )
            )
        start = local_index * replay_per_group
        for replay_index, replay_source in enumerate(
            replay[category][start : start + replay_per_group], start=3
        ):
            train_rows.append(
                _record_for_training(
                    source=replay_source,
                    messages=deepcopy(replay_source["messages"]),
                    language="en",
                    semantic_group_id=group_id,
                    microstep_slot=replay_index,
                    loss_weight=language_weights["en"],
                    tokenizer=tokenizer,
                    settings=settings,
                    selective=True,
                )
            )

    valid_rows: list[dict[str, Any]] = []
    for category in ("math", "python", "cpp"):
        for local_index, source in enumerate(chosen["valid"][category]):
            candidate_id = source["metadata"]["candidate_id"]
            group_id = f"forge-valid-{category}-{local_index:02d}-{candidate_id}"
            translated = translations[candidate_id]
            for slot, language, messages in (
                (0, "en", source["messages"]),
                (1, "zh_Hans", translated["zh_Hans"]),
                (2, "zh_Hant", translated["zh_Hant"]),
            ):
                valid_rows.append(
                    _record_for_training(
                        source=source,
                        messages=deepcopy(messages),
                        language=language,
                        semantic_group_id=group_id,
                        microstep_slot=slot,
                        loss_weight=1.0,
                        tokenizer=tokenizer,
                        settings=settings,
                        selective=False,
                    )
                )

    group_sizes = Counter(row["metadata"]["semantic_group_id"] for row in train_rows)
    if set(group_sizes.values()) != {int(config.section("training_v2")["grad_accumulation_steps"])}:
        raise RuntimeError(f"Semantic groups do not match gradient accumulation: {group_sizes}")
    language_mass: Counter[str] = Counter()
    category_mass: Counter[str] = Counter()
    for row in train_rows:
        weight = float(row["metadata"]["loss_weight"])
        language_mass[row["metadata"]["language"]] += weight
        category_mass[row["metadata"]["category"]] += weight

    write_jsonl(final_dir / "train.jsonl", train_rows)
    write_jsonl(final_dir / "valid.jsonl", valid_rows)
    manifest = {
        "fingerprint": fingerprint,
        "method": "Forge triad-coupled selective distillation",
        "train_records": len(train_rows),
        "valid_records": len(valid_rows),
        "semantic_groups": len(group_sizes),
        "group_size": int(config.section("training_v2")["grad_accumulation_steps"]),
        "category_groups": dict(category_counts),
        "language_gradient_mass": dict(language_mass),
        "language_gradient_ratios": _ratio(language_mass),
        "category_gradient_mass": dict(category_mass),
        "category_gradient_ratios": _ratio(category_mass),
        "train_sha256": sha256_file(final_dir / "train.jsonl"),
        "valid_sha256": sha256_file(final_dir / "valid.jsonl"),
        "selected_train_ids": sorted(chosen_train_ids),
        "selected_valid_ids": sorted(
            row["metadata"]["candidate_id"]
            for rows in chosen["valid"].values()
            for row in rows
        ),
        "translation_lengths": {
            candidate_id: translation_lengths[candidate_id]
            for candidate_id in sorted(chosen_train_ids | set(
                row["metadata"]["candidate_id"]
                for rows in chosen["valid"].values()
                for row in rows
            ))
        },
    }
    write_json(done_path, manifest)
    _write_public_forge_manifests(config, manifest, train_rows, valid_rows)
    return manifest

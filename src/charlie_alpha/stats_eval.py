from __future__ import annotations

import gc
import json
import re
import time
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any

import mlx.core as mx
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from .config import ProjectConfig
from .io_utils import (
    append_jsonl,
    atomic_write_text,
    canonical_hash,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from .stats_agent import StatsAgent
from .stats_data import _build_record, _scenario, build_stats_data
from .stats_training import (
    _score_loaded_selector,
    _tokenize_stats_record,
    calibrate_stats_adapter,
)
from .validators import is_contaminated, word_ngrams

_STATQA_RAW_URL = (
    "https://raw.githubusercontent.com/HKUSTDial/StatQA/"
    "3d5f6b5b6926600dc952f39d79dfdef82999aeeb/"
    "Data/Integrated%20Dataset/Balanced%20Benchmark/StatQA.json"
)


def _download_statqa(config: ProjectConfig) -> Path:
    source = config.sources["datasets"]["statqa_eval"]
    path = config.path_for("artifact_dir") / "evaluation-sources" / "statqa.json"
    metadata_path = path.with_suffix(".metadata.json")
    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("revision") == source["revision"] and metadata.get("sha256") == sha256_file(
            path
        ):
            return path
    with urllib.request.urlopen(_STATQA_RAW_URL, timeout=60) as response:  # noqa: S310
        payload = response.read()
    atomic_write_text(path, payload.decode("utf-8"))
    write_json(
        metadata_path,
        {
            "repo_id": source["repo_id"],
            "revision": source["revision"],
            "license": source["license"],
            "policy": source["policy"],
            "sha256": sha256_file(path),
        },
    )
    return path


def _round_robin_sample(
    grouped: dict[str, list[int]],
    count: int,
    *,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed)
    queues = {key: list(values) for key, values in grouped.items()}
    for values in queues.values():
        rng.shuffle(values)
    keys = sorted(queues)
    rng.shuffle(keys)
    selected: list[int] = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            if queues[key] and len(selected) < count:
                selected.append(queues[key].pop())
                progressed = True
        if not progressed:
            raise RuntimeError(f"Cannot sample {count} items from the requested strata")
    return selected


def _pbench_indices(dataset: Any, *, seed: int, count: int) -> list[int]:
    if count % 2:
        raise ValueError("P-Bench sample count must be even")
    per_difficulty = count // 2
    selected: list[int] = []
    covered_categories: set[str] = set()
    for difficulty_index, difficulty in enumerate(("Easy", "Hard")):
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(dataset):
            if row["difficulty"] == difficulty and int(row["dataset_bytes"]) <= 25 * 1024**2:
                grouped[str(row["category"])].append(index)
        covered_categories.update(grouped)
        selected.extend(
            _round_robin_sample(
                grouped,
                per_difficulty,
                seed=seed + difficulty_index * 10_000,
            )
        )
    if len(covered_categories) != 17:
        raise RuntimeError(
            "P-Bench Easy/Hard sample cannot cover all 17 categories collectively"
        )
    return selected


def _statqa_indices(rows: list[dict[str, Any]], *, seed: int, count: int) -> list[int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[f"{row.get('task')}::{row.get('difficulty')}"].append(index)
    return _round_robin_sample(grouped, count, seed=seed)


def _decontamination_audit(
    config: ProjectConfig,
    pbench: Any,
    pbench_indices: list[int],
    statqa_rows: list[dict[str, Any]],
    statqa_indices: list[int],
) -> dict[str, Any]:
    training_prompts = {
        str(row["messages"][1]["content"])
        for variant in ("hard-label", "regret-random", "dgp-regret")
        for row in read_jsonl(config.path_for("final_dir") / variant / "train.jsonl")
    }
    references = [word_ngrams(prompt, 8) for prompt in sorted(training_prompts)]
    references = [value for value in references if value]
    pbench_overlap = [
        str(pbench[index]["task_id"])
        for index in pbench_indices
        if is_contaminated(
            str(pbench[index]["question"]),
            references,
            size=8,
            threshold=0.5,
        )
    ]
    statqa_overlap = [
        int(index)
        for index in statqa_indices
        if is_contaminated(
            str(statqa_rows[index]["refined_question"]),
            references,
            size=8,
            threshold=0.5,
        )
    ]
    result = {
        "ngram": 8,
        "threshold": 0.5,
        "training_prompt_count": len(references),
        "p_bench_overlap_task_ids": pbench_overlap,
        "statqa_overlap_indices": statqa_overlap,
        "passed": not pbench_overlap and not statqa_overlap,
    }
    if not result["passed"]:
        raise RuntimeError("The sealed statistics evaluation sample overlaps training prompts")
    return result


def build_stats_evaluation_lock(
    config: ProjectConfig,
    force: bool = False,
) -> dict[str, Any]:
    build_stats_data(config, force=False)
    lock_path = config.path_for("eval_lock")
    settings = config.section("stats_evaluation")
    pbench_source = config.sources["datasets"]["p_bench_eval"]
    pbench = load_dataset(
        pbench_source["repo_id"],
        split=pbench_source["split"],
        revision=pbench_source["revision"],
    )
    statqa_path = _download_statqa(config)
    statqa_rows = json.loads(statqa_path.read_text(encoding="utf-8"))
    final_surface = list(read_jsonl(config.path_for("stats_dir") / "surface" / "final.jsonl"))
    seed = int(settings["seed"])
    pbench_indices = _pbench_indices(pbench, seed=seed, count=int(settings["p_bench"]))
    statqa_indices = _statqa_indices(
        statqa_rows,
        seed=seed,
        count=int(settings["statqa"]),
    )
    decontamination = _decontamination_audit(
        config,
        pbench,
        pbench_indices,
        statqa_rows,
        statqa_indices,
    )
    final_ids = [str(row["scenario"]["blueprint_id"]) for row in final_surface]
    fingerprint = canonical_hash(
        {
            "sources": {
                "p_bench": pbench_source,
                "statqa": config.sources["datasets"]["statqa_eval"],
            },
            "pbench_indices": pbench_indices,
            "statqa_indices": statqa_indices,
            "final_ids": final_ids,
            "retention_sha256": sha256_file(
                config.root / "configs" / "retention.stats.jsonl"
            ),
            "decontamination": decontamination,
            "seed": seed,
            "lock_version": 2,
        }
    )
    if lock_path.exists() and not force:
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint:
            return existing
    pbench_tasks = [
        {
            "index": index,
            "task_id": str(pbench[index]["task_id"]),
            "category": str(pbench[index]["category"]),
            "difficulty": str(pbench[index]["difficulty"]),
            "dataset_sha256": str(pbench[index]["dataset_sha256"]),
        }
        for index in pbench_indices
    ]
    lock = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "seed": seed,
        "sealed": True,
        "p_bench": {
            "repo_id": pbench_source["repo_id"],
            "revision": pbench_source["revision"],
            "tasks": pbench_tasks,
            "count": len(pbench_tasks),
        },
        "statqa": {
            "repo_id": config.sources["datasets"]["statqa_eval"]["repo_id"],
            "revision": config.sources["datasets"]["statqa_eval"]["revision"],
            "source_sha256": sha256_file(statqa_path),
            "indices": statqa_indices,
            "count": len(statqa_indices),
            "evaluation_only": True,
        },
        "final_dgp": {
            "blueprint_ids": final_ids,
            "surface_sha256": sha256_file(config.path_for("stats_dir") / "surface" / "final.jsonl"),
            "count": len(final_ids),
        },
        "decontamination": decontamination,
        "trilingual_blueprint_ids": final_ids[: int(settings["trilingual_semantic_tasks"])],
        "clarification_blueprint_ids": final_ids[: int(settings["clarification_cases"])],
        "retention_sha256": sha256_file(config.root / "configs" / "retention.stats.jsonl"),
    }
    write_json(lock_path, lock)
    return lock


def freeze_stats_recipe(config: ProjectConfig) -> dict[str, Any]:
    selected_path = config.path_for("artifact_dir") / "selected.json"
    if not selected_path.exists():
        calibrate_stats_adapter(config, force=False)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    evaluation_lock = build_stats_evaluation_lock(config, force=False)
    freeze = {
        "schema_version": 1,
        "frozen": True,
        "adapter_path": selected["adapter_path"],
        "adapter_sha256": selected["adapter_sha256"],
        "delta_scale": selected["delta_scale"],
        "selected_variant": selected["variant"],
        "dev": selected["dev"],
        "evaluation_lock_fingerprint": evaluation_lock["fingerprint"],
        "evaluation_lock_sha256": sha256_file(config.path_for("eval_lock")),
        "frozen_at_unix": int(time.time()),
    }
    write_json(config.path_for("artifact_dir") / "recipe-frozen.json", freeze)
    selected["recipe_frozen"] = True
    write_json(selected_path, selected)
    return freeze


def _variant_adapter(config: ProjectConfig, variant: str) -> tuple[Path, str]:
    artifact_dir = config.path_for("artifact_dir")
    selected = json.loads((artifact_dir / "selected.json").read_text(encoding="utf-8"))
    if variant == "base":
        return Path(selected["adapter_path"]), "base"
    if variant == "selected":
        return Path(selected["adapter_path"]), "stats"
    if variant in {"hard-label", "dgp-regret"}:
        if variant == "dgp-regret" and selected.get("variant") == "dgp-regret":
            return Path(selected["adapter_path"]), "stats"
        comparison = json.loads(
            (artifact_dir / "pilot-comparison.json").read_text(encoding="utf-8")
        )
        candidate = next(
            item for item in comparison["candidates"] if item["variant"] == variant
        )
        return Path(candidate["adapter_path"]), "stats"
    raise ValueError(f"Unknown stats evaluation variant: {variant}")


def _surface_by_id(config: ProjectConfig) -> dict[str, dict[str, Any]]:
    return {
        str(row["scenario"]["blueprint_id"]): row
        for row in read_jsonl(config.path_for("stats_dir") / "surface" / "final.jsonl")
    }


def _selector_rows(
    surface: dict[str, dict[str, Any]],
    ids: list[str],
    *,
    language: str,
    incomplete: bool,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    output: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for blueprint_id in ids:
        simulation = surface[blueprint_id]
        scenario = _scenario(simulation["scenario"])
        record = _build_record(
            scenario,
            simulation,
            language=language,
            loss_weight=1.0,
            incomplete=incomplete,
            variant="dgp-regret",
            refined_explanation=None,
        )
        output.append((record, simulation))
    return output


def _clarification_score(
    agent: StatsAgent,
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    correct = 0
    predictions: list[dict[str, Any]] = []
    for record, simulation in rows:
        item = _tokenize_stats_record(agent.tokenizer, record)
        tokens = mx.array([item["tokens"]], dtype=mx.int32)
        logits = agent.model(tokens[:, :-1])
        selector = logits[0, int(item["method_position"]), :]
        menu_logits = mx.take(selector, mx.array(item["candidate_token_ids"], dtype=mx.int32))
        index = int(mx.argmax(menu_logits).item())
        predicted = str(record["metadata"]["candidate_method_ids"][index])
        passed = predicted == "needs_clarification"
        correct += int(passed)
        predictions.append(
            {
                "blueprint_id": simulation["scenario"]["blueprint_id"],
                "predicted_method_id": predicted,
                "passed": passed,
            }
        )
        del logits, selector, menu_logits
    return {
        "count": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
        "predictions": predictions,
    }


def _normalize(value: str) -> str:
    compatible = unicodedata.normalize("NFKC", value)
    return "".join(
        character.lower() for character in compatible if character.isalnum()
    )


def _json_from_answer(answer: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL | re.IGNORECASE)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return {}


def _load_progress(
    path: Path,
    *,
    fingerprint: str,
    id_field: str,
) -> dict[str, dict[str, Any]]:
    status_path = path.with_suffix(".status.json")
    if path.exists() and status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("fingerprint") == fingerprint:
            return {str(row[id_field]): row for row in read_jsonl(path)}
    write_jsonl(path, [])
    write_json(status_path, {"fingerprint": fingerprint, "completed": 0})
    return {}


def _append_progress(
    path: Path,
    *,
    fingerprint: str,
    row: dict[str, Any],
    completed: int,
) -> None:
    append_jsonl(path, row)
    write_json(
        path.with_suffix(".status.json"),
        {"fingerprint": fingerprint, "completed": completed},
    )


def _run_statqa(
    agent: StatsAgent,
    rows: list[dict[str, Any]],
    indices: list[int],
    *,
    route: str,
    progress_path: Path | None = None,
    progress_fingerprint: str = "",
) -> dict[str, Any]:
    agent.router.set_route("base" if route == "base" else "adapter")
    correct = 0
    method_correct = 0
    column_correct = 0
    column_recall: list[float] = []
    details: list[dict[str, Any]] = []
    saved = (
        _load_progress(
            progress_path,
            fingerprint=progress_fingerprint,
            id_field="index",
        )
        if progress_path is not None
        else {}
    )
    for index in indices:
        cached = saved.get(str(index))
        if cached is not None:
            details.append(cached)
            continue
        row = rows[index]
        gold = json.loads(row["ground_truth"])
        messages = [
            {
                "role": "system",
                "content": (
                    "Identify applicable statistical methods and relevant columns. Return only "
                    'JSON: {"methods":[...],"columns":[...]}.'
                ),
            },
            {"role": "user", "content": str(row["refined_question"])},
        ]
        answer = agent.answer_without_tools(
            messages,
            route="base" if route == "base" else "stats",
            max_tokens=300,
            temperature=0.0,
            top_p=1.0,
        )
        parsed = _json_from_answer(answer)
        predicted_methods = [str(value) for value in parsed.get("methods", [])]
        predicted_columns = [str(value) for value in parsed.get("columns", [])]
        gold_methods = [str(value) for value in gold.get("methods", [])]
        gold_columns = [str(value) for value in gold.get("columns", [])]
        predicted_method_set = {_normalize(value) for value in predicted_methods if value}
        gold_method_set = {_normalize(value) for value in gold_methods if value}
        predicted_column_set = {_normalize(value) for value in predicted_columns if value}
        gold_column_set = {_normalize(value) for value in gold_columns if value}
        methods_match = predicted_method_set == gold_method_set
        columns_match = predicted_column_set == gold_column_set
        exact = methods_match and columns_match
        recall = (
            sum(
                _normalize(value) in predicted_column_set
                for value in gold_columns
            )
            / len(gold_columns)
            if gold_columns
            else 1.0
        )
        correct += int(exact)
        method_correct += int(methods_match)
        column_correct += int(columns_match)
        column_recall.append(recall)
        detail = {
            "index": index,
            "task": row.get("task"),
            "difficulty": row.get("difficulty"),
            "method_correct": methods_match,
            "columns_correct": columns_match,
            "exact_correct": exact,
            "column_recall": recall,
            "predicted_methods": predicted_methods,
        }
        details.append(detail)
        if progress_path is not None:
            _append_progress(
                progress_path,
                fingerprint=progress_fingerprint,
                row=detail,
                completed=len(details),
            )
    correct = sum(bool(item["exact_correct"]) for item in details)
    method_correct = sum(bool(item["method_correct"]) for item in details)
    column_correct = sum(bool(item["columns_correct"]) for item in details)
    column_recall = [float(item["column_recall"]) for item in details]
    return {
        "count": len(indices),
        "accuracy": correct / len(indices),
        "method_set_accuracy": method_correct / len(indices),
        "column_set_accuracy": column_correct / len(indices),
        "column_recall": float(np.mean(column_recall)),
        "details": details,
    }


def _find_p_value(value: Any, target_terms: list[str] | None = None) -> float | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _normalize(str(key)) in {"pvalue", "randomizationpvalue"}:
                try:
                    return float(item)
                except (TypeError, ValueError):
                    continue
            if _normalize(str(key)) == "pvalues" and isinstance(item, dict):
                normalized = {_normalize(str(name)): result for name, result in item.items()}
                for target in target_terms or []:
                    wanted = _normalize(target)
                    matches = [
                        result
                        for name, result in normalized.items()
                        if wanted == name or wanted in name or name in wanted
                    ]
                    if matches:
                        try:
                            return float(matches[0])
                        except (TypeError, ValueError):
                            pass
                non_intercept = [
                    result
                    for name, result in normalized.items()
                    if name not in {"const", "intercept"}
                ]
                if len(non_intercept) == 1:
                    try:
                        return float(non_intercept[0])
                    except (TypeError, ValueError):
                        pass
        for item in value.values():
            found = _find_p_value(item, target_terms)
            if found is not None:
                return found
    if isinstance(value, list):
        for item in value:
            found = _find_p_value(item, target_terms)
            if found is not None:
                return found
    return None


def _run_pbench(
    config: ProjectConfig,
    agent: StatsAgent,
    dataset: Any,
    tasks: list[dict[str, Any]],
    *,
    route: str,
    progress_path: Path | None = None,
    progress_fingerprint: str = "",
) -> dict[str, Any]:
    source = config.sources["datasets"]["p_bench_eval"]
    raw_correct = 0
    strict_correct = 0
    details: list[dict[str, Any]] = []
    saved = (
        _load_progress(
            progress_path,
            fingerprint=progress_fingerprint,
            id_field="task_id",
        )
        if progress_path is not None
        else {}
    )
    for task in tasks:
        cached = saved.get(str(task["task_id"]))
        if cached is not None:
            details.append(cached)
            continue
        row = dataset[int(task["index"])]
        data_path = Path(
            hf_hub_download(
                repo_id=source["repo_id"],
                repo_type="dataset",
                revision=source["revision"],
                filename=row["dataset_path"],
            )
        )
        try:
            result = agent.analyze(
                data_paths=[data_path],
                question=str(row["question"]),
                language="en",
                route="base" if route == "base" else "stats",
            )
        except Exception as error:
            raise RuntimeError(
                "P-Bench task execution failed: "
                f"task_id={row['task_id']!r}, dataset_path={row['dataset_path']!r}"
            ) from error
        plan_variables = result.get("analysis_plan", {}).get("variables", {})
        target_terms: list[str] = []
        for role in ("target", "exposure", "treatment", "predictor", "predictors", "group"):
            value = plan_variables.get(role) if isinstance(plan_variables, dict) else None
            if isinstance(value, list):
                target_terms.extend(str(item) for item in value[:1])
            elif value:
                target_terms.append(str(value))
        p_value = _find_p_value(result.get("tools"), target_terms)
        predicted_decision = (
            None if p_value is None else ("reject" if p_value < 0.05 else "fail_to_reject")
        )
        raw = predicted_decision == row["decision"]
        normal = NormalDist()
        predicted_z = (
            -normal.inv_cdf(min(max(float(p_value), 1e-300), 1.0) / 2)
            if p_value is not None
            else None
        )
        gold_z = -normal.inv_cdf(min(max(float(row["p_value"]), 1e-300), 1.0) / 2)
        strict = (
            raw
            and predicted_z is not None
            and abs(predicted_z - gold_z) < 0.5
        )
        raw_correct += int(raw)
        strict_correct += int(strict)
        detail = {
            "task_id": row["task_id"],
            "category": row["category"],
            "difficulty": row["difficulty"],
            "p_value": p_value,
            "gold_p_value": float(row["p_value"]),
            "raw_correct": raw,
            "strict_correct": strict,
            "tool_calls": result["tool_calls"],
            "status": result["analysis_plan"]["status"],
        }
        details.append(detail)
        if progress_path is not None:
            _append_progress(
                progress_path,
                fingerprint=progress_fingerprint,
                row=detail,
                completed=len(details),
            )
    raw_correct = sum(bool(item["raw_correct"]) for item in details)
    strict_correct = sum(bool(item["strict_correct"]) for item in details)
    count = len(tasks)
    return {
        "count": count,
        "raw_accuracy": raw_correct / count,
        "strict_accuracy": strict_correct / count,
        "details": details,
    }


def _retention_score(
    agent: StatsAgent,
    config: ProjectConfig,
    *,
    route: str,
    progress_path: Path | None = None,
    progress_fingerprint: str = "",
) -> dict[str, Any]:
    rows = list(read_jsonl(config.root / "configs" / "retention.stats.jsonl"))
    details: list[dict[str, Any]] = []
    grouped: dict[str, list[bool]] = defaultdict(list)
    saved = (
        _load_progress(
            progress_path,
            fingerprint=progress_fingerprint,
            id_field="task_id",
        )
        if progress_path is not None
        else {}
    )
    for row in rows:
        cached = saved.get(str(row["task_id"]))
        if cached is not None:
            details.append(cached)
            continue
        answer = agent.answer_without_tools(
            [{"role": "user", "content": str(row["prompt"])}],
            route="base" if route == "base" else "stats",
            max_tokens=int(row["max_tokens"]),
            temperature=0.0,
            top_p=1.0,
        )
        passed = _normalize(str(row["gold"])) in _normalize(answer)
        detail = {
            "task_id": row["task_id"],
            "language": row["language"],
            "domain": row["domain"],
            "passed": passed,
            "answer": answer,
        }
        details.append(detail)
        if progress_path is not None:
            _append_progress(
                progress_path,
                fingerprint=progress_fingerprint,
                row=detail,
                completed=len(details),
            )
    for item in details:
        grouped[f"language:{item['language']}"].append(bool(item["passed"]))
        grouped[f"domain:{item['domain']}"].append(bool(item["passed"]))
    return {
        "count": len(rows),
        "accuracy": sum(item["passed"] for item in details) / len(details),
        "groups": {
            key: sum(values) / len(values) for key, values in sorted(grouped.items())
        },
        "details": details,
    }


def _paired_bootstrap(
    base: list[float],
    candidate: list[float],
    *,
    seed: int,
    repetitions: int,
) -> dict[str, float]:
    base_array = np.asarray(base, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    differences = base_array - candidate_array
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled = rng.integers(0, len(differences), size=len(differences))
        draws[index] = float(np.mean(differences[sampled]))
    return {
        "mean_improvement": float(np.mean(differences)),
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
    }


def run_stats_evaluation(
    config: ProjectConfig,
    *,
    variant: str,
    force: bool = False,
) -> dict[str, Any]:
    freeze_path = config.path_for("artifact_dir") / "recipe-frozen.json"
    if not freeze_path.exists():
        freeze_stats_recipe(config)
    lock = build_stats_evaluation_lock(config, force=False)
    adapter_path, route = _variant_adapter(config, variant)
    report_dir = config.path_for("report_dir")
    report_path = report_dir / f"evaluation-{variant}.json"
    fingerprint = canonical_hash(
        {
            "variant": variant,
            "adapter_sha256": sha256_file(adapter_path / "adapters.safetensors"),
            "route": route,
            "lock": lock["fingerprint"],
            "evaluator_version": 4,
        }
    )
    if report_path.exists() and not force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint and existing.get("complete"):
            return existing
    started = time.monotonic()
    progress_dir = report_dir / "progress" / variant
    suite_fingerprint = lambda name: canonical_hash(  # noqa: E731
        {"evaluation": fingerprint, "suite": name, "progress_version": 1}
    )
    agent = StatsAgent(config, adapter_path=adapter_path)
    agent.router.set_route("base" if route == "base" else "adapter")
    surface = _surface_by_id(config)
    final_ids = list(lock["final_dgp"]["blueprint_ids"])
    dgp = _score_loaded_selector(
        agent.model,
        agent.tokenizer,
        _selector_rows(surface, final_ids, language="en", incomplete=False),
    )
    trilingual: dict[str, Any] = {}
    for language in ("en", "zh_Hant", "zh_Hans"):
        trilingual[language] = _score_loaded_selector(
            agent.model,
            agent.tokenizer,
            _selector_rows(
                surface,
                list(lock["trilingual_blueprint_ids"]),
                language=language,
                incomplete=False,
            ),
        )
    clarification = _clarification_score(
        agent,
        _selector_rows(
            surface,
            list(lock["clarification_blueprint_ids"]),
            language="en",
            incomplete=True,
        ),
    )
    statqa_path = _download_statqa(config)
    statqa_rows = json.loads(statqa_path.read_text(encoding="utf-8"))
    statqa = _run_statqa(
        agent,
        statqa_rows,
        [int(value) for value in lock["statqa"]["indices"]],
        route=route,
        progress_path=progress_dir / "statqa.jsonl",
        progress_fingerprint=suite_fingerprint("statqa"),
    )
    pbench_source = config.sources["datasets"]["p_bench_eval"]
    pbench_dataset = load_dataset(
        pbench_source["repo_id"],
        split=pbench_source["split"],
        revision=pbench_source["revision"],
    )
    pbench = _run_pbench(
        config,
        agent,
        pbench_dataset,
        list(lock["p_bench"]["tasks"]),
        route=route,
        progress_path=progress_dir / "p-bench.jsonl",
        progress_fingerprint=suite_fingerprint("p-bench"),
    )
    retention = _retention_score(
        agent,
        config,
        route=route,
        progress_path=progress_dir / "retention.jsonl",
        progress_fingerprint=suite_fingerprint("retention"),
    )
    result = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "complete": True,
        "variant": variant,
        "route": route,
        "adapter_sha256": sha256_file(adapter_path / "adapters.safetensors"),
        "evaluation_lock": lock["fingerprint"],
        "dgp_final": dgp,
        "p_bench": pbench,
        "statqa": statqa,
        "trilingual": trilingual,
        "clarification": clarification,
        "retention": retention,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(report_path, result)
    del agent
    gc.collect()
    mx.clear_cache()
    return result


def compare_stats_evaluation(config: ProjectConfig) -> dict[str, Any]:
    report_dir = config.path_for("report_dir")
    reports = {
        variant: json.loads((report_dir / f"evaluation-{variant}.json").read_text(encoding="utf-8"))
        for variant in ("base", "hard-label", "dgp-regret")
    }
    base = reports["base"]
    selected_report_path = report_dir / "evaluation-selected.json"
    selected_metadata = json.loads(
        (config.path_for("artifact_dir") / "selected.json").read_text(encoding="utf-8")
    )
    proposed = (
        json.loads(selected_report_path.read_text(encoding="utf-8"))
        if selected_metadata.get("variant") != "dgp-regret" and selected_report_path.exists()
        else reports["dgp-regret"]
    )
    base_regrets = [float(item["normalized_regret"]) for item in base["dgp_final"]["predictions"]]
    proposed_regrets = [
        float(item["normalized_regret"]) for item in proposed["dgp_final"]["predictions"]
    ]
    settings = config.section("stats_evaluation")
    bootstrap = _paired_bootstrap(
        base_regrets,
        proposed_regrets,
        seed=int(settings["seed"]),
        repetitions=int(settings["bootstrap_repetitions"]),
    )
    gates = settings["gates"]
    base_mean = float(base["dgp_final"]["normalized_regret"])
    proposed_mean = float(proposed["dgp_final"]["normalized_regret"])
    relative_regret = (base_mean - proposed_mean) / base_mean if base_mean else 0.0
    base_invalid = float(base["dgp_final"]["invalid_selection_rate"])
    proposed_invalid = float(proposed["dgp_final"]["invalid_selection_rate"])
    invalid_reduction = (base_invalid - proposed_invalid) / base_invalid if base_invalid else 0.0
    language_deltas = {
        language: 100
        * (
            float(proposed["trilingual"][language]["accuracy"])
            - float(base["trilingual"][language]["accuracy"])
        )
        for language in ("en", "zh_Hant", "zh_Hans")
    }
    retention_deltas = {
        group: 100
        * (
            float(proposed["retention"]["groups"][group])
            - float(base["retention"]["groups"][group])
        )
        for group in base["retention"]["groups"]
    }
    ability_deltas = {
        domain: 100
        * (
            float(proposed["dgp_final"]["domain_accuracy"][domain])
            - float(base["dgp_final"]["domain_accuracy"][domain])
        )
        for domain in base["dgp_final"]["domain_accuracy"]
    }
    pilot = json.loads(
        (config.path_for("artifact_dir") / "pilot-comparison.json").read_text(encoding="utf-8")
    )
    hard_dev = next(item["dev"] for item in pilot["candidates"] if item["variant"] == "hard-label")
    dgp_dev = next(item["dev"] for item in pilot["candidates"] if item["variant"] == "dgp-regret")
    dgp_vs_hard = (
        (float(hard_dev["normalized_regret"]) - float(dgp_dev["normalized_regret"]))
        / float(hard_dev["normalized_regret"])
        if float(hard_dev["normalized_regret"])
        else 0.0
    )
    checks = {
        "final_regret": (
            relative_regret >= float(gates["regret_relative_improvement"])
            and bootstrap["ci95_lower"] > 0
        ),
        "invalid_selection": invalid_reduction
        >= float(gates["invalid_selection_relative_reduction"]),
        "p_bench": (
            100
            * (float(proposed["p_bench"]["raw_accuracy"]) - float(base["p_bench"]["raw_accuracy"]))
            >= float(gates["p_bench_raw_points"])
            and float(proposed["p_bench"]["strict_accuracy"])
            >= float(base["p_bench"]["strict_accuracy"])
        ),
        "statqa": (
            100 * (float(proposed["statqa"]["accuracy"]) - float(base["statqa"]["accuracy"]))
            >= float(gates["statqa_points"])
        ),
        "trilingual": (
            float(np.mean(list(language_deltas.values()))) >= float(gates["trilingual_points"])
            and min(language_deltas.values()) >= -float(gates["max_subgroup_regression_points"])
        ),
        "retention": min(retention_deltas.values())
        >= -float(gates["max_subgroup_regression_points"]),
        "ability_categories": min(ability_deltas.values())
        >= -float(gates["max_subgroup_regression_points"]),
        "dgp_regret_ablation": dgp_vs_hard >= float(gates["dgp_regret_vs_hard_relative"]),
    }
    comparison = {
        "schema_version": 1,
        "ability_gates_passed": all(checks.values()),
        "checks": checks,
        "absolute_metrics": {
            variant: {
                "dgp_final": {
                    "normalized_regret": float(report["dgp_final"]["normalized_regret"]),
                    "method_accuracy": float(report["dgp_final"]["accuracy"]),
                    "invalid_selection_rate": float(
                        report["dgp_final"]["invalid_selection_rate"]
                    ),
                },
                "p_bench": {
                    "raw_accuracy": float(report["p_bench"]["raw_accuracy"]),
                    "strict_accuracy": float(report["p_bench"]["strict_accuracy"]),
                },
                "statqa": {
                    "exact_accuracy": float(report["statqa"]["accuracy"]),
                    "method_set_accuracy": float(report["statqa"]["method_set_accuracy"]),
                    "column_set_accuracy": float(report["statqa"]["column_set_accuracy"]),
                },
                "trilingual_method_accuracy": {
                    language: float(report["trilingual"][language]["accuracy"])
                    for language in ("en", "zh_Hant", "zh_Hans")
                },
                "clarification_accuracy": float(report["clarification"]["accuracy"]),
                "retention_accuracy": float(report["retention"]["accuracy"]),
            }
            for variant, report in reports.items()
        },
        "metrics": {
            "regret_relative_improvement": relative_regret,
            "paired_bootstrap": bootstrap,
            "invalid_selection_relative_reduction": invalid_reduction,
            "language_accuracy_deltas_points": language_deltas,
            "dgp_regret_vs_hard_dev_relative": dgp_vs_hard,
            "retention_accuracy_deltas_points": retention_deltas,
            "ability_category_accuracy_deltas_points": ability_deltas,
            "p_bench_raw_delta_points": 100
            * (float(proposed["p_bench"]["raw_accuracy"]) - float(base["p_bench"]["raw_accuracy"])),
            "statqa_delta_points": 100
            * (float(proposed["statqa"]["accuracy"]) - float(base["statqa"]["accuracy"])),
        },
        "reports": {
            **{key: value["fingerprint"] for key, value in reports.items()},
            "selected": proposed["fingerprint"],
        },
        "selected_variant": proposed["variant"],
        "dgp_regret_benefit_claim_allowed": checks["dgp_regret_ablation"],
    }
    write_json(config.path_for("report_dir") / "comparison.json", comparison)
    # This aggregate report is safe to track. Per-item predictions stay below the
    # ignored generated report directory so sealed prompts and model responses are
    # not published accidentally.
    write_json(config.root / "reports" / "stats" / "evaluation.json", comparison)
    return comparison

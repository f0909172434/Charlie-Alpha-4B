from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from math_verify import parse, verify
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from rich.console import Console

from .config import ProjectConfig
from .io_utils import append_jsonl, canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl
from .sandbox import evaluate_function_candidate, evaluate_standalone_candidate
from .training import _base_snapshot
from .validators import extract_code_blocks, has_target_script, normalize_text

console = Console()


def _stable_select(
    rows: list[dict[str, Any]], limit: int, seed: int, id_key: str
) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: canonical_hash({"seed": seed, "id": str(row[id_key])}),
    )[:limit]


def _build_tasks(config: ProjectConfig) -> list[dict[str, Any]]:
    settings = config.section("evaluation")
    seed = int(config.section("project")["seed"])
    sources = config.sources["datasets"]
    tasks: list[dict[str, Any]] = []

    math_source = sources["math500_eval"]
    math_rows = list(
        load_dataset(
            math_source["repo_id"],
            split=math_source["split"],
            revision=math_source["revision"],
        )
    )
    for row in _stable_select(math_rows, int(settings["math500_limit"]), seed, "unique_id"):
        tasks.append(
            {
                "task_id": f"math500:{row['unique_id']}",
                "benchmark": "MATH-500",
                "domain": "math",
                "language": "en",
                "prompt": row["problem"],
                "gold": row["answer"],
            }
        )

    gsm_source = sources["gsm8k_eval"]
    gsm_rows = list(
        load_dataset(
            gsm_source["repo_id"],
            name=gsm_source["config"],
            split=gsm_source["split"],
            revision=gsm_source["revision"],
        )
    )
    selected_gsm = _stable_select(
        [{**row, "row_id": index} for index, row in enumerate(gsm_rows)],
        int(settings["gsm8k_limit"]),
        seed,
        "row_id",
    )
    for row in selected_gsm:
        tasks.append(
            {
                "task_id": f"gsm8k:{row['row_id']}",
                "benchmark": "GSM8K",
                "domain": "math",
                "language": "en",
                "prompt": row["question"],
                "gold": row["answer"].split("####")[-1].strip(),
            }
        )

    for benchmark, rows, limit in (
        ("HumanEval+", list(get_human_eval_plus().values()), int(settings["humaneval_limit"])),
        ("MBPP+", list(get_mbpp_plus().values()), int(settings["mbpp_limit"])),
    ):
        selected = _stable_select(rows, limit, seed, "task_id")
        for row in selected:
            tasks.append(
                {
                    "task_id": f"{benchmark.lower()}:{row['task_id']}",
                    "benchmark": benchmark,
                    "domain": "code",
                    "language": "en",
                    "prompt": (
                        "Complete the following Python task. Return a complete implementation in "
                        f"one Python code block.\n\n{row['prompt']}"
                    ),
                    "function_prompt": row["prompt"],
                    "canonical_solution": row["canonical_solution"],
                    "entry_point": row["entry_point"],
                    "inputs": [*row["base_input"], *row["plus_input"][:20]],
                    "atol": row.get("atol", 0.0),
                }
            )

    final_test = list(read_jsonl(config.path_for("final_dir") / "test.jsonl"))
    per_language = int(settings["trilingual_per_language"])
    for language in ("en", "zh_Hant", "zh_Hans"):
        candidates = [row for row in final_test if row["metadata"]["language"] == language]
        candidates.sort(key=lambda row: row["metadata"]["prompt_sha256"])
        for row in candidates[:per_language]:
            metadata = row["metadata"]
            tasks.append(
                {
                    "task_id": f"trilingual:{language}:{metadata['prompt_sha256'][:16]}",
                    "benchmark": "trilingual-canary",
                    "domain": metadata["domain"],
                    "language": language,
                    "prompt": row["messages"][0]["content"],
                    "gold": metadata.get("answer"),
                    "code_language": metadata.get("code_language"),
                    "tests": metadata.get("tests", []),
                }
            )
    tasks.extend(read_jsonl(config.root / "configs" / "retention_canary.jsonl"))
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bucket_order: list[str] = []
    for task in tasks:
        key = f"{task['benchmark']}:{task['domain']}:{task['language']}"
        if key not in buckets:
            bucket_order.append(key)
        buckets[key].append(task)
    interleaved: list[dict[str, Any]] = []
    while any(buckets.values()):
        for key in bucket_order:
            if buckets[key]:
                interleaved.append(buckets[key].pop(0))
    return interleaved[: int(settings["task_limit"])]


def _tasks(config: ProjectConfig) -> list[dict[str, Any]]:
    report_dir = config.path_for("report_dir")
    path = report_dir / "eval-tasks.jsonl"
    fingerprint_path = report_dir / "eval-tasks.json"
    fingerprint = canonical_hash(
        {
            "evaluation": config.section("evaluation"),
            "sources": config.sources["datasets"],
            "final_data": (config.path_for("final_dir") / ".done.json").read_text(encoding="utf-8"),
            "retention_canary": sha256_file(config.root / "configs" / "retention_canary.jsonl"),
            "v": 3,
        }
    )
    if path.exists() and fingerprint_path.exists():
        metadata = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint:
            return list(read_jsonl(path))
    rows = _build_tasks(config)
    write_jsonl(path, rows)
    write_json(fingerprint_path, {"fingerprint": fingerprint, "tasks": len(rows)})
    return rows


def _render_prompt(tokenizer: Any, task: dict[str, Any]) -> str:
    system = (
        "Solve accurately without showing reasoning. For math, return only the final answer. "
        "For code, return only a complete implementation in one fenced code block. Preserve "
        "the requested response language."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task["prompt"]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if prompt.rstrip().endswith("<think>"):
        prompt += "</think>\n\n"
    return prompt


def _extract_code(value: str) -> str | None:
    blocks = extract_code_blocks(value)
    if blocks:
        return max(blocks, key=len)
    without_thinking = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL).strip()
    if "def " in without_thinking or "#include" in without_thinking:
        return without_thinking
    return None


def _score_math(gold: str | None, output: str) -> bool:
    if not gold:
        return False
    try:
        return bool(verify(parse(gold), parse(output), strict=False))
    except (TimeoutError, TypeError, ValueError):
        return False


def _score_task(task: dict[str, Any], output: str) -> dict[str, Any]:
    benchmark = task["benchmark"]
    if benchmark == "retention-canary":
        visible = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)
        normalized_output = normalize_text(visible).strip(" .,!?:;。！？，：；`*_")
        normalized_gold = normalize_text(task["gold"])
        exact = normalized_output == normalized_gold or normalized_output.endswith(normalized_gold)
        evidence_text = visible.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
        if normalized_gold == "i":
            evidence = bool(
                re.search(
                    r"\bnext(?:\s+letter)?(?:\s+after\s+G)?\s+is\s+I\b|\bI\s*\(9\)",
                    evidence_text,
                    flags=re.IGNORECASE,
                )
            )
        elif re.fullmatch(r"[A-Za-z0-9]+", normalized_gold):
            evidence = bool(
                re.search(
                    rf"(?<![A-Za-z0-9]){re.escape(normalized_gold)}(?![A-Za-z0-9])",
                    evidence_text,
                    flags=re.IGNORECASE,
                )
            )
        else:
            evidence = normalized_gold in normalize_text(evidence_text)
        return {
            "passed": exact or evidence,
            "exact_answer": exact,
            "answer_evidence": evidence,
            "sandboxed": False,
        }
    if benchmark in {"MATH-500", "GSM8K"}:
        return {"passed": _score_math(task["gold"], output), "sandboxed": False}
    if benchmark in {"HumanEval+", "MBPP+"}:
        code = _extract_code(output)
        if code is None:
            return {"passed": False, "reason": "no code", "sandboxed": True}
        return evaluate_function_candidate(
            candidate_code=code,
            prompt=task["function_prompt"],
            canonical_solution=task["canonical_solution"],
            entry_point=task["entry_point"],
            inputs=task["inputs"],
            atol=float(task["atol"]),
        )

    language = task["language"]
    language_pass = (
        has_target_script(output, language) if language in {"zh_Hant", "zh_Hans"} else True
    )
    if task["domain"] == "math" and task.get("gold"):
        correctness = _score_math(task["gold"], output)
        return {
            "passed": language_pass and correctness,
            "language_pass": language_pass,
            "correctness_pass": correctness,
            "sandboxed": False,
        }
    code = _extract_code(output)
    if code and task.get("tests"):
        code_result = evaluate_standalone_candidate(
            candidate_code=code,
            language=task.get("code_language") or "python",
            tests=task["tests"],
        )
        code_result["language_pass"] = language_pass
        code_result["passed"] = bool(code_result["passed"] and language_pass)
        return code_result
    return {"passed": language_pass and code is not None, "language_pass": language_pass}


def _metric_summary(rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        passed = bool(row["score"]["passed"])
        groups["overall"].append(passed)
        groups[f"benchmark:{row['benchmark']}"].append(passed)
        groups[f"domain:{row['domain']}"].append(passed)
        groups[f"language:{row['language']}"].append(passed)
    return {
        "expected_tasks": expected,
        "completed_tasks": len(rows),
        "coverage": round(len(rows) / expected, 4) if expected else 0.0,
        "scores": {
            key: {
                "correct": sum(values),
                "total": len(values),
                "accuracy": round(sum(values) / len(values), 4),
            }
            for key, values in sorted(groups.items())
            if values
        },
    }


def _write_comparison(config: ProjectConfig) -> None:
    report_dir = config.path_for("report_dir")
    metric_paths = {
        variant: report_dir / f"metrics-{variant}.json" for variant in ("base", "adapter")
    }
    if not all(path.exists() for path in metric_paths.values()):
        return
    metrics = {
        variant: json.loads(path.read_text(encoding="utf-8"))
        for variant, path in metric_paths.items()
    }
    base_scores = metrics["base"]["scores"]
    adapter_scores = metrics["adapter"]["scores"]
    shared = sorted(set(base_scores) & set(adapter_scores))
    deltas = {
        key: round(
            (adapter_scores[key]["accuracy"] - base_scores[key]["accuracy"]) * 100,
            2,
        )
        for key in shared
    }
    settings = config.section("evaluation")
    fully_covered = all(metrics[variant]["coverage"] == 1.0 for variant in metrics)
    subgroup_keys = [key for key in shared if key.startswith(("domain:", "language:"))]
    stable = (
        fully_covered
        and deltas.get("overall", -100) >= float(settings["stable_improvement_points"])
        and all(
            deltas[key] >= -float(settings["max_subgroup_regression_points"])
            for key in subgroup_keys
        )
    )
    comparison = {
        "profile": config.section("project")["profile"],
        "base": metrics["base"],
        "adapter": metrics["adapter"],
        "delta_percentage_points": deltas,
        "quality_classification": "stable-candidate" if stable else "experimental",
        "note": "The overnight compact suite is not a substitute for the full release suite.",
    }
    write_json(report_dir / "evaluation.json", comparison)
    reports = config.root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    write_json(reports / "evaluation.json", comparison)


def run_evaluation(config: ProjectConfig, variant: str, force: bool = False) -> dict[str, Any]:
    if variant not in {"base", "adapter"}:
        raise ValueError("variant must be 'base' or 'adapter'")
    tasks = _tasks(config)
    report_dir = config.path_for("report_dir")
    output_path = report_dir / f"generations-{variant}.jsonl"
    adapter_path: str | None = None
    adapter_sha256: str | None = None
    if variant == "adapter":
        selected_path = config.path_for("artifact_dir") / "selected.json"
        if not selected_path.exists():
            raise RuntimeError("No selected adapter; run `make pilot` and `make train` first.")
        adapter_path = json.loads(selected_path.read_text(encoding="utf-8"))["adapter_path"]
        adapter_sha256 = sha256_file(Path(adapter_path) / "adapters.safetensors")
    if force and output_path.exists():
        output_path.unlink()
    existing = [] if force else list(read_jsonl(output_path))
    task_fingerprints = {
        task["task_id"]: canonical_hash(
            {
                "task": task,
                "prompt_template": "direct-v2",
                **({"adapter_sha256": adapter_sha256} if adapter_sha256 else {}),
            }
        )
        for task in tasks
    }
    tasks_by_id = {task["task_id"]: task for task in tasks}
    compatible_existing: list[dict[str, Any]] = []
    migrated = False
    for row in existing:
        current_fingerprint = task_fingerprints.get(row["task_id"])
        if current_fingerprint is None:
            migrated = True
            continue
        stored_fingerprint = row.get("task_fingerprint")
        if stored_fingerprint == current_fingerprint:
            compatible_row = row
        elif stored_fingerprint is None and row["benchmark"] != "retention-canary":
            compatible_row = {**row, "task_fingerprint": current_fingerprint}
            migrated = True
        else:
            migrated = True
            continue
        if compatible_row.get("score_version") != 2:
            compatible_row = {
                **compatible_row,
                "score": _score_task(tasks_by_id[row["task_id"]], row["output"]),
                "score_version": 2,
            }
            migrated = True
        compatible_existing.append(compatible_row)
    existing = compatible_existing
    if migrated:
        write_jsonl(output_path, existing)
    completed = {row["task_id"] for row in existing}

    model_path = _base_snapshot(config)
    model, tokenizer = load(
        model_path,
        adapter_path=adapter_path,
        tokenizer_config={"trust_remote_code": True},
    )
    started = time.monotonic()
    settings = config.section("evaluation")
    max_seconds = int(settings["max_seconds_per_variant"])
    sampler = make_sampler(temp=float(settings["temperature"]))

    for task in tasks:
        if task["task_id"] in completed:
            continue
        if time.monotonic() - started >= max_seconds:
            console.print(f"[yellow]{variant} evaluation reached its time budget.[/yellow]")
            break
        prompt = _render_prompt(tokenizer, task)
        max_tokens = int(
            task.get("max_tokens")
            or (
                settings["math_max_new_tokens"]
                if task["domain"] == "math"
                else settings["code_max_new_tokens"]
            )
        )
        output = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        score = _score_task(task, output)
        row = {
            "task_id": task["task_id"],
            "task_fingerprint": task_fingerprints[task["task_id"]],
            "benchmark": task["benchmark"],
            "domain": task["domain"],
            "language": task["language"],
            "output": output,
            "score": score,
            "score_version": 2,
        }
        append_jsonl(output_path, row)
        existing.append(row)
        completed.add(task["task_id"])
        console.print(
            f"{variant}: {len(existing)}/{len(tasks)} {task['task_id']} "
            f"{'PASS' if score['passed'] else 'FAIL'}"
        )

    summary = _metric_summary(existing, len(tasks))
    summary.update(
        {
            "variant": variant,
            "model": config.sources["models"]["base_mlx_4bit"],
            "adapter_id": Path(adapter_path).name if adapter_path else None,
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }
    )
    write_json(report_dir / f"metrics-{variant}.json", summary)
    _write_comparison(config)
    return summary

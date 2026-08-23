from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from huggingface_hub import snapshot_download
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from opencc import OpenCC
from rich.console import Console

from .config import ProjectConfig
from .evaluation import _score_task
from .io_utils import (
    append_jsonl,
    canonical_hash,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)

console = Console()


def _gzip_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_evalplus_artifacts(config: ProjectConfig) -> dict[str, str]:
    cache_dir = config.path_for("artifact_dir") / "evalplus"
    cache_dir.mkdir(parents=True, exist_ok=True)
    verified: dict[str, str] = {}
    for name, artifact in config.sources["evaluation_artifacts"].items():
        path = cache_dir / f"{name}-{artifact['version']}.jsonl.gz"
        if not path.exists() or _gzip_content_sha256(path) != artifact["content_sha256"]:
            partial = path.with_suffix(path.suffix + ".part")
            request = urllib.request.Request(
                artifact["url"], headers={"User-Agent": "Charlie-Alpha-Forge/0.2"}
            )
            with urllib.request.urlopen(request, timeout=60) as response, partial.open(
                "wb"
            ) as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            actual = _gzip_content_sha256(partial)
            if actual != artifact["content_sha256"]:
                partial.unlink(missing_ok=True)
                raise RuntimeError(
                    f"EvalPlus artifact hash mismatch for {name}: {actual}"
                )
            partial.replace(path)
        verified[name] = _gzip_content_sha256(path)
    return verified


def _ranked(rows: list[dict[str, Any]], seed: int, id_key: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: canonical_hash({"seed": seed, "id": str(row[id_key])}),
    )


def _code_task(benchmark: str, row: dict[str, Any]) -> dict[str, Any]:
    base_inputs = row.get("base_input")
    plus_inputs = row.get("plus_input")
    if not isinstance(base_inputs, list):
        base_inputs = []
    if not isinstance(plus_inputs, list):
        plus_inputs = []
    return {
        "task_id": f"{benchmark.lower()}:{row['task_id']}",
        "benchmark": benchmark,
        "domain": "code",
        "language": "en",
        "prompt": (
            "Complete the following Python task. Return a complete implementation in one "
            f"Python code block.\n\n{row['prompt']}"
        ),
        "function_prompt": row["prompt"],
        "canonical_solution": row["canonical_solution"],
        "entry_point": row["entry_point"],
        "inputs": [*base_inputs, *plus_inputs[:20]],
        "atol": row.get("atol", 0.0),
    }


def _catalog(config: ProjectConfig) -> dict[str, dict[str, Any]]:
    sources = config.sources["datasets"]
    catalog: dict[str, dict[str, Any]] = {}

    math_source = sources["math500_eval"]
    for row in load_dataset(
        math_source["repo_id"],
        split=math_source["split"],
        revision=math_source["revision"],
    ):
        task = {
            "task_id": f"math500:{row['unique_id']}",
            "benchmark": "MATH-500",
            "domain": "math",
            "language": "en",
            "prompt": row["problem"],
            "gold": row["answer"],
        }
        catalog[task["task_id"]] = task

    gsm_source = sources["gsm8k_eval"]
    for index, row in enumerate(
        load_dataset(
            gsm_source["repo_id"],
            name=gsm_source["config"],
            split=gsm_source["split"],
            revision=gsm_source["revision"],
        )
    ):
        task = {
            "task_id": f"gsm8k:{index}",
            "benchmark": "GSM8K",
            "domain": "math",
            "language": "en",
            "prompt": row["question"],
            "gold": row["answer"].split("####")[-1].strip(),
        }
        catalog[task["task_id"]] = task

    evalplus = config.sources["evaluation_artifacts"]
    for benchmark, rows in (
        (
            "HumanEval+",
            get_human_eval_plus(version=evalplus["humaneval_plus"]["version"]).values(),
        ),
        (
            "MBPP+",
            get_mbpp_plus(version=evalplus["mbpp_plus"]["version"]).values(),
        ),
    ):
        for row in rows:
            task = _code_task(benchmark, row)
            catalog[task["task_id"]] = task

    mgsm_source = sources["mgsm_eval"]
    converter = OpenCC("s2twp")
    for index, row in enumerate(
        load_dataset(
            mgsm_source["repo_id"],
            name="zh",
            split=mgsm_source["split"],
            revision=mgsm_source["revision"],
        )
    ):
        for language, prompt in (
            ("zh_Hans", row["question"]),
            ("zh_Hant", converter.convert(row["question"])),
        ):
            task = {
                "task_id": f"mgsm:{language}:{index}",
                "benchmark": "MGSM",
                "domain": "math",
                "language": language,
                "prompt": prompt,
                "gold": str(row["answer"]),
                "source_index": index,
            }
            catalog[task["task_id"]] = task

    for task in read_jsonl(config.root / "configs" / "retention_canary.v2.jsonl"):
        catalog[task["task_id"]] = task
    return catalog


def _take_disjoint(
    ids: list[str],
    *,
    dev_count: int,
    final_count: int,
    used: set[str],
) -> tuple[list[str], list[str]]:
    available = [task_id for task_id in ids if task_id not in used]
    required = dev_count + final_count
    if len(available) < required:
        raise RuntimeError(f"Evaluation lock needs {required} tasks but only has {len(available)}")
    dev = available[:dev_count]
    final = available[dev_count:required]
    used.update([*dev, *final])
    return dev, final


def build_evaluation_lock(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    path = config.path_for("eval_lock")
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    verified_evalplus = _verify_evalplus_artifacts(config)
    settings = config.section("evaluation_v2")
    seed = int(settings["seed"])
    catalog = _catalog(config)
    old_path = config.root / "reports" / "generated" / "eval-tasks.jsonl"
    if force and path.exists() and not old_path.exists():
        raise RuntimeError(
            "Cannot rebuild the evaluation lock without the v0.1 task manifest; "
            "use the committed lock instead"
        )
    old_ids = {row["task_id"] for row in read_jsonl(old_path)} if old_path.exists() else set()
    used = set(old_ids)
    suites: dict[str, list[str]] = {"dev": [], "final": []}
    groups = (
        ("math500", "math500:"),
        ("gsm8k", "gsm8k:"),
        ("humaneval_plus", "humaneval+:"),
        ("mbpp_plus", "mbpp+:"),
    )
    for config_key, prefix in groups:
        rows = [task for task in catalog.values() if task["task_id"].startswith(prefix)]
        ordered = [task["task_id"] for task in _ranked(rows, seed, "task_id")]
        dev, final = _take_disjoint(
            ordered,
            dev_count=int(settings["dev"][config_key]),
            final_count=int(settings["final"][config_key]),
            used=used,
        )
        suites["dev"].extend(dev)
        suites["final"].extend(final)

    # Use different underlying MGSM problems for each script and suite so that a language score
    # cannot be inflated by solving the same arithmetic item twice.
    used_mgsm_indices: set[int] = set()
    all_indices = list(range(250))
    all_indices.sort(key=lambda index: canonical_hash({"seed": seed, "mgsm": index}))
    for language, config_key in (
        ("zh_Hans", "mgsm_zh_hans"),
        ("zh_Hant", "mgsm_zh_hant"),
    ):
        available = [index for index in all_indices if index not in used_mgsm_indices]
        dev_count = int(settings["dev"][config_key])
        final_count = int(settings["final"][config_key])
        chosen = available[: dev_count + final_count]
        used_mgsm_indices.update(chosen)
        suites["dev"].extend(f"mgsm:{language}:{index}" for index in chosen[:dev_count])
        suites["final"].extend(
            f"mgsm:{language}:{index}" for index in chosen[dev_count:]
        )

    retention_ids = sorted(
        task_id for task_id in catalog if task_id.startswith("retention-v2-")
    )
    suites["dev"].extend(task_id for task_id in retention_ids if task_id.endswith("-1"))
    suites["final"].extend(task_id for task_id in retention_ids if task_id.endswith("-2"))

    locked_suites: dict[str, list[dict[str, str]]] = {}
    for suite, ids in suites.items():
        locked_suites[suite] = [
            {"task_id": task_id, "task_sha256": canonical_hash(catalog[task_id])}
            for task_id in ids
        ]

    lock = {
        "schema_version": 1,
        "created_at": "2026-08-24T00:00:00+08:00",
        "seed": seed,
        "policy": (
            "built before v0.2 training; dev may guide selection; final must not be generated "
            "until the training recipe and checkpoint are frozen"
        ),
        "excluded_v1_task_ids_sha256": canonical_hash(sorted(old_ids)),
        "source_revisions": {
            key: config.sources["datasets"][key]["revision"]
            for key in ("math500_eval", "gsm8k_eval", "mgsm_eval")
        },
        "evalplus_artifacts": config.sources["evaluation_artifacts"],
        "verified_evalplus_sha256": verified_evalplus,
        "suites": locked_suites,
    }
    write_json(path, lock)
    return lock


def load_locked_tasks(config: ProjectConfig, suite: str) -> list[dict[str, Any]]:
    if suite not in {"dev", "final"}:
        raise ValueError("suite must be 'dev' or 'final'")
    lock = build_evaluation_lock(config)
    catalog = _catalog(config)
    tasks: list[dict[str, Any]] = []
    for entry in lock["suites"][suite]:
        task = catalog.get(entry["task_id"])
        if task is None:
            raise RuntimeError(f"Locked evaluation task is unavailable: {entry['task_id']}")
        if canonical_hash(task) != entry["task_sha256"]:
            raise RuntimeError(f"Locked evaluation task changed: {entry['task_id']}")
        tasks.append(task)
    return tasks


def _variant(config: ProjectConfig, name: str) -> tuple[dict[str, str], str | None]:
    models = config.sources["models"]
    if name == "qwen35-base":
        return models["research_base_mlx_4bit"], None
    if name == "qwen3-base":
        return models["base_mlx_4bit"], None
    if name == "v0.1":
        selected = json.loads(
            (config.root / "artifacts" / "selected.json").read_text(encoding="utf-8")
        )
        return models["base_mlx_4bit"], selected["adapter_path"]
    if name == "forge":
        selected = json.loads(
            (config.path_for("artifact_dir") / "selected.json").read_text(encoding="utf-8")
        )
        return models["research_base_mlx_4bit"], selected["adapter_path"]
    raise ValueError("variant must be qwen35-base, qwen3-base, v0.1, or forge")


def _render_prompt(tokenizer: Any, task: dict[str, Any]) -> str:
    language = task["language"]
    if task["domain"] == "code":
        system = (
            "Solve the programming task accurately. Think through edge cases, then return only a "
            "complete implementation in one fenced Python code block."
        )
    elif language == "zh_Hant":
        system = "請用精簡但足夠的推理，以繁體中文回答；最後一行只寫最終答案。"
    elif language == "zh_Hans":
        system = "请用精简但足够的推理，以简体中文回答；最后一行只写最终答案。"
    else:
        system = (
            "Use concise but sufficient reasoning. Put only the final answer on the last line."
        )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task["prompt"]}]
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            **template_kwargs,
            enable_thinking=task["benchmark"] == "MATH-500",
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        if task["domain"] == "code" and prompt.rstrip().endswith("<think>"):
            prompt += "</think>\n\n"
    return prompt


def _metrics(rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
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
        },
    }


def _score(task: dict[str, Any], output: str) -> dict[str, Any]:
    if task["benchmark"] != "retention-v2":
        return _score_task(task, output)
    visible = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
    normalized = re.sub(r"[\s`*_.,!?;:。！？，：；]", "", visible).lower()
    gold = re.sub(r"[\s`*_.,!?;:。！？，：；]", "", str(task["gold"])).lower()
    return {"passed": normalized == gold or normalized.endswith(gold), "sandboxed": False}


def run_forge_evaluation(
    config: ProjectConfig,
    *,
    variant: str,
    suite: str = "dev",
    force: bool = False,
    adapter_path_override: Path | None = None,
    task_ids: list[str] | None = None,
    report_label: str | None = None,
    max_seconds_override: int | None = None,
) -> dict[str, Any]:
    if suite == "final":
        verify_frozen_recipe(config)
    tasks = load_locked_tasks(config, suite)
    if task_ids is not None:
        by_id = {task["task_id"]: task for task in tasks}
        missing = [task_id for task_id in task_ids if task_id not in by_id]
        if missing:
            raise RuntimeError(
                f"Pilot evaluation tasks are not in the locked {suite} set: {missing}"
            )
        tasks = [by_id[task_id] for task_id in task_ids]
    if adapter_path_override is None:
        model_source, adapter_path = _variant(config, variant)
    else:
        model_source = config.sources["models"]["research_base_mlx_4bit"]
        adapter_path = str(adapter_path_override)
    model_path = snapshot_download(
        repo_id=model_source["repo_id"], revision=model_source["revision"]
    )
    adapter_sha = None
    if adapter_path:
        adapter_sha = sha256_file(Path(adapter_path) / "adapters.safetensors")

    report_dir = config.path_for("report_dir") / suite
    label = report_label or variant
    if report_label:
        report_dir = report_dir / "pilots"
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label)
    output_path = report_dir / f"generations-{safe_label}.jsonl"
    if force and output_path.exists():
        output_path.unlink()
    existing = [] if force else list(read_jsonl(output_path))
    fingerprints = {
        task["task_id"]: canonical_hash(
            {
                "task": task,
                "model": model_source,
                "adapter_sha256": adapter_sha,
                "prompt_version": config.section("evaluation_v2")["prompt_version"],
            }
        )
        for task in tasks
    }
    before_filter = len(existing)
    existing = [
        row
        for row in existing
        if row.get("task_fingerprint") == fingerprints.get(row.get("task_id"))
    ]
    invalidated = before_filter - len(existing)
    write_jsonl(output_path, existing)
    completed = {row["task_id"] for row in existing}

    model, tokenizer = load(
        model_path,
        adapter_path=adapter_path,
        tokenizer_config={"trust_remote_code": True},
    )
    settings = config.section("evaluation_v2")
    sampler = make_sampler(temp=float(settings["temperature"]))
    started = time.monotonic()
    for task in tasks:
        if task["task_id"] in completed:
            continue
        max_seconds = (
            int(max_seconds_override)
            if max_seconds_override is not None
            else int(settings["max_seconds_per_variant"])
        )
        if time.monotonic() - started >= max_seconds:
            break
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
            _render_prompt(tokenizer, task),
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        score = _score(task, output)
        row = {
            "task_id": task["task_id"],
            "task_fingerprint": fingerprints[task["task_id"]],
            "benchmark": task["benchmark"],
            "domain": task["domain"],
            "language": task["language"],
            "output": output,
            "score": score,
        }
        append_jsonl(output_path, row)
        existing.append(row)
        completed.add(task["task_id"])
        console.print(
            f"{suite}/{variant}: {len(existing)}/{len(tasks)} {task['task_id']} "
            f"{'PASS' if score['passed'] else 'FAIL'}"
        )

    summary = _metrics(existing, len(tasks))
    summary.update(
        {
            "suite": suite,
            "variant": variant,
            "model": model_source,
            "adapter_sha256": adapter_sha,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "invalidated_cached_generations": invalidated,
        }
    )
    write_json(report_dir / f"metrics-{safe_label}.json", summary)
    return summary


def pilot_task_ids(config: ProjectConfig) -> list[str]:
    tasks = load_locked_tasks(config, "dev")
    selected: list[str] = []
    for benchmark in ("MATH-500", "GSM8K", "HumanEval+", "MBPP+"):
        selected.append(next(task["task_id"] for task in tasks if task["benchmark"] == benchmark))
    for language in ("zh_Hans", "zh_Hant"):
        selected.append(
            next(
                task["task_id"]
                for task in tasks
                if task["benchmark"] == "MGSM" and task["language"] == language
            )
        )
    for language in ("en", "zh_Hans", "zh_Hant"):
        selected.append(
            next(
                task["task_id"]
                for task in tasks
                if task["benchmark"] == "retention-v2" and task["language"] == language
            )
        )
    return selected


def _comparison(config: ProjectConfig, suite: str) -> dict[str, Any]:
    report_dir = config.path_for("report_dir") / suite
    metrics: dict[str, dict[str, Any]] = {}
    for variant in ("qwen35-base", "forge"):
        path = report_dir / f"metrics-{variant}.json"
        if not path.exists():
            raise RuntimeError(f"Missing {suite} evaluation for {variant}: {path}")
        metrics[variant] = json.loads(path.read_text(encoding="utf-8"))
        if metrics[variant]["coverage"] != 1.0:
            raise RuntimeError(f"Incomplete {suite} evaluation for {variant}")
    base_scores = metrics["qwen35-base"]["scores"]
    forge_scores = metrics["forge"]["scores"]
    shared = sorted(set(base_scores) & set(forge_scores))
    deltas = {
        key: round(
            (forge_scores[key]["accuracy"] - base_scores[key]["accuracy"]) * 100,
            2,
        )
        for key in shared
    }
    settings = config.section("evaluation_v2")
    subgroup_keys = [key for key in shared if key.startswith(("domain:", "language:"))]
    passed = (
        deltas.get("overall", -100.0) >= float(settings["release_improvement_points"])
        and all(
            deltas[key] >= -float(settings["max_subgroup_regression_points"])
            for key in subgroup_keys
        )
    )
    comparison = {
        "suite": suite,
        "base": metrics["qwen35-base"],
        "forge": metrics["forge"],
        "delta_percentage_points": deltas,
        "gate_passed": passed,
        "gate": {
            "overall_improvement_points": settings["release_improvement_points"],
            "maximum_language_or_domain_regression_points": settings[
                "max_subgroup_regression_points"
            ],
        },
    }
    write_json(report_dir / "comparison.json", comparison)
    return comparison


def freeze_forge_recipe(config: ProjectConfig) -> dict[str, Any]:
    artifact_dir = config.path_for("artifact_dir")
    selected_path = artifact_dir / "selected.json"
    if not selected_path.exists():
        raise RuntimeError("Forge training is missing")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if not selected.get("complete"):
        raise RuntimeError("Forge training did not complete within its fixed budget")
    comparison = _comparison(config, "dev")
    if not comparison["gate_passed"]:
        raise RuntimeError(
            "Forge does not yet beat the base on the development gate; the final suite "
            "remains sealed"
        )
    adapter_path = Path(selected["adapter_path"]) / "adapters.safetensors"
    files = {
        "pipeline": config.path,
        "source_lock": config.path_for("source_lock"),
        "evaluation_lock": config.path_for("eval_lock"),
        "train_data": config.path_for("final_dir") / "train.jsonl",
        "valid_data": config.path_for("final_dir") / "valid.jsonl",
        "adapter": adapter_path,
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Cannot freeze missing Forge artifacts: {missing}")
    hashes = {key: sha256_file(path) for key, path in files.items()}
    frozen = {
        "schema_version": 1,
        "frozen_at": datetime.now(UTC).isoformat(),
        "policy": "no recipe, data, adapter, prompt, or task changes after this point",
        "candidate": selected["candidate"],
        "adapter_path": str(Path(selected["adapter_path"])),
        "hashes": hashes,
        "dev_comparison_sha256": canonical_hash(comparison),
    }
    frozen["recipe_sha256"] = canonical_hash(frozen)
    write_json(artifact_dir / "frozen-recipe.json", frozen)
    selected["recipe_frozen"] = True
    selected["frozen_recipe_sha256"] = frozen["recipe_sha256"]
    write_json(selected_path, selected)
    return frozen


def verify_frozen_recipe(config: ProjectConfig) -> dict[str, Any]:
    path = config.path_for("artifact_dir") / "frozen-recipe.json"
    if not path.exists():
        raise RuntimeError(
            "Final evaluation is sealed until the recipe and adapter are frozen after dev eval"
        )
    frozen = json.loads(path.read_text(encoding="utf-8"))
    selected = json.loads(
        (config.path_for("artifact_dir") / "selected.json").read_text(encoding="utf-8")
    )
    files = {
        "pipeline": config.path,
        "source_lock": config.path_for("source_lock"),
        "evaluation_lock": config.path_for("eval_lock"),
        "train_data": config.path_for("final_dir") / "train.jsonl",
        "valid_data": config.path_for("final_dir") / "valid.jsonl",
        "adapter": Path(selected["adapter_path"]) / "adapters.safetensors",
    }
    actual = {key: sha256_file(file_path) for key, file_path in files.items()}
    if actual != frozen["hashes"]:
        changed = sorted(key for key in actual if actual[key] != frozen["hashes"].get(key))
        raise RuntimeError(f"Frozen Forge recipe changed: {changed}")
    return frozen


def compare_forge_evaluation(config: ProjectConfig, suite: str) -> dict[str, Any]:
    if suite == "final":
        verify_frozen_recipe(config)
    return _comparison(config, suite)

from __future__ import annotations

import gc
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import yaml
from huggingface_hub import snapshot_download
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from rich.console import Console

from .config import ProjectConfig
from .forge_eval import (
    _catalog,
    _metrics,
    _ranked,
    _render_prompt,
    _score,
    _verify_evalplus_artifacts,
)
from .io_utils import (
    append_jsonl,
    canonical_hash,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)

console = Console()


def _router_settings(config: ProjectConfig) -> tuple[Path, dict[str, Any]]:
    path = config.root / "configs" / "router.v3.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or values.get("schema_version") != 1:
        raise ValueError("Unsupported Forge router configuration")
    return path, values


def _router_path(config: ProjectConfig, value: str) -> Path:
    return config.resolve(value)


def _router_catalog(config: ProjectConfig, settings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    catalog = _catalog(config)
    for task in read_jsonl(_router_path(config, settings["retention_canary"])):
        catalog[task["task_id"]] = task
    return catalog


def route_uses_adapter(task: dict[str, Any], settings: dict[str, Any]) -> bool:
    route = settings["route"]
    return (
        task["domain"] in set(route["adapter_domains"])
        or task["language"] in set(route["adapter_languages"])
    )


def build_router_confirmation_lock(
    config: ProjectConfig, force: bool = False
) -> dict[str, Any]:
    router_path, settings = _router_settings(config)
    lock_path = _router_path(config, settings["evaluation_lock"])
    if lock_path.exists() and not force:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    report_dir = _router_path(config, settings["report_dir"])
    if force and any(report_dir.glob("generations-*.jsonl")):
        raise RuntimeError("Router confirmation lock cannot change after generations exist")

    catalog = _router_catalog(config, settings)
    v2_lock = json.loads(config.path_for("eval_lock").read_text(encoding="utf-8"))
    used = {
        entry["task_id"]
        for suite in v2_lock["suites"].values()
        for entry in suite
    }
    old_path = config.root / "reports" / "generated" / "eval-tasks.jsonl"
    used.update(row["task_id"] for row in read_jsonl(old_path))
    seed = int(settings["seed"])
    counts = settings["counts"]
    selected: list[str] = []
    for key, prefix in (
        ("math500", "math500:"),
        ("gsm8k", "gsm8k:"),
        ("humaneval_plus", "humaneval+:"),
        ("mbpp_plus", "mbpp+:"),
    ):
        rows = [task for task in catalog.values() if task["task_id"].startswith(prefix)]
        ordered = [task["task_id"] for task in _ranked(rows, seed, "task_id")]
        available = [task_id for task_id in ordered if task_id not in used]
        count = int(counts[key])
        if len(available) < count:
            raise RuntimeError(f"Router lock needs {count} {key} tasks")
        chosen = available[:count]
        selected.extend(chosen)
        used.update(chosen)

    used_mgsm_indices = {
        int(task_id.rsplit(":", 1)[-1])
        for task_id in used
        if task_id.startswith("mgsm:")
    }
    all_indices = list(range(250))
    all_indices.sort(key=lambda index: canonical_hash({"seed": seed, "mgsm": index}))
    for language, key in (("zh_Hans", "mgsm_zh_hans"), ("zh_Hant", "mgsm_zh_hant")):
        count = int(counts[key])
        available = [index for index in all_indices if index not in used_mgsm_indices]
        chosen = available[:count]
        if len(chosen) != count:
            raise RuntimeError(f"Router lock needs {count} {key} tasks")
        used_mgsm_indices.update(chosen)
        selected.extend(f"mgsm:{language}:{index}" for index in chosen)

    retention = sorted(
        task_id for task_id in catalog if task_id.startswith("retention-v3-")
    )
    selected.extend(retention)
    if len(selected) != sum(int(value) for value in counts.values()) + len(retention):
        raise RuntimeError("Router confirmation lock count mismatch")
    if len(selected) != len(set(selected)) or set(selected) & {
        entry["task_id"]
        for suite in v2_lock["suites"].values()
        for entry in suite
    }:
        raise RuntimeError("Router confirmation tasks overlap a prior v2 suite")

    verified_evalplus = _verify_evalplus_artifacts(config)
    lock = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "policy": settings["policy"],
        "router_config_sha256": sha256_file(router_path),
        "excluded_v2_lock_sha256": sha256_file(config.path_for("eval_lock")),
        "verified_evalplus_sha256": verified_evalplus,
        "source_revisions": {
            key: config.sources["datasets"][key]["revision"]
            for key in ("math500_eval", "gsm8k_eval", "mgsm_eval")
        },
        "suite": [
            {"task_id": task_id, "task_sha256": canonical_hash(catalog[task_id])}
            for task_id in selected
        ],
    }
    write_json(lock_path, lock)
    return lock


def load_router_confirmation_tasks(config: ProjectConfig) -> list[dict[str, Any]]:
    _, settings = _router_settings(config)
    lock = build_router_confirmation_lock(config)
    catalog = _router_catalog(config, settings)
    tasks: list[dict[str, Any]] = []
    for entry in lock["suite"]:
        task = catalog.get(entry["task_id"])
        if task is None or canonical_hash(task) != entry["task_sha256"]:
            raise RuntimeError(f"Router confirmation task changed: {entry['task_id']}")
        tasks.append(task)
    return tasks


def freeze_router_recipe(config: ProjectConfig) -> dict[str, Any]:
    router_path, settings = _router_settings(config)
    lock_path = _router_path(config, settings["evaluation_lock"])
    build_router_confirmation_lock(config)
    report_dir = _router_path(config, settings["report_dir"])
    if any(report_dir.glob("generations-*.jsonl")):
        raise RuntimeError("Router recipe must be frozen before confirmation generations")
    selected = json.loads(
        (config.path_for("artifact_dir") / "selected.json").read_text(encoding="utf-8")
    )
    files = {
        "router_config": router_path,
        "confirmation_lock": lock_path,
        "source_lock": config.path_for("source_lock"),
        "adapter": Path(selected["adapter_path"]) / "adapters.safetensors",
        "v2_final_evidence": config.root / "reports" / "v2" / "evaluation.json",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Cannot freeze missing router artifacts: {missing}")
    frozen = {
        "schema_version": 1,
        "frozen_at": datetime.now(UTC).isoformat(),
        "policy": "route and confirmation tasks cannot change after this point",
        "route": settings["route"],
        "adapter_path": selected["adapter_path"],
        "hashes": {key: sha256_file(path) for key, path in files.items()},
    }
    frozen["recipe_sha256"] = canonical_hash(frozen)
    artifact_dir = _router_path(config, settings["artifact_dir"])
    write_json(artifact_dir / "frozen-recipe.json", frozen)
    return frozen


def verify_router_recipe(config: ProjectConfig) -> dict[str, Any]:
    router_path, settings = _router_settings(config)
    frozen_path = _router_path(config, settings["artifact_dir"]) / "frozen-recipe.json"
    if not frozen_path.exists():
        raise RuntimeError("Router confirmation is sealed until the route is frozen")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    selected = json.loads(
        (config.path_for("artifact_dir") / "selected.json").read_text(encoding="utf-8")
    )
    files = {
        "router_config": router_path,
        "confirmation_lock": _router_path(config, settings["evaluation_lock"]),
        "source_lock": config.path_for("source_lock"),
        "adapter": Path(selected["adapter_path"]) / "adapters.safetensors",
        "v2_final_evidence": config.root / "reports" / "v2" / "evaluation.json",
    }
    actual = {key: sha256_file(path) for key, path in files.items()}
    if actual != frozen["hashes"]:
        changed = sorted(key for key in actual if actual[key] != frozen["hashes"].get(key))
        raise RuntimeError(f"Frozen router recipe changed: {changed}")
    return frozen


def run_router_confirmation(
    config: ProjectConfig, variant: str, force: bool = False
) -> dict[str, Any]:
    if variant not in {"qwen35-base", "routed"}:
        raise ValueError("router variant must be qwen35-base or routed")
    verify_router_recipe(config)
    _, settings = _router_settings(config)
    tasks = load_router_confirmation_tasks(config)
    report_dir = _router_path(config, settings["report_dir"])
    output_path = report_dir / f"generations-{variant}.jsonl"
    if force and output_path.exists():
        output_path.unlink()
    existing = [] if force else list(read_jsonl(output_path))
    selected = json.loads(
        (config.path_for("artifact_dir") / "selected.json").read_text(encoding="utf-8")
    )
    adapter_path = Path(selected["adapter_path"])
    adapter_sha = sha256_file(adapter_path / "adapters.safetensors")
    base_source = config.sources["models"]["research_base_mlx_4bit"]
    route_fingerprint = canonical_hash(settings["route"] if variant == "routed" else "base")
    fingerprints = {
        task["task_id"]: canonical_hash(
            {
                "task": task,
                "model": base_source,
                "route": route_fingerprint,
                "adapter_sha256": adapter_sha if variant == "routed" else None,
                "prompt_version": config.section("evaluation_v2")["prompt_version"],
            }
        )
        for task in tasks
    }
    existing = [
        row
        for row in existing
        if row.get("task_fingerprint") == fingerprints.get(row.get("task_id"))
    ]
    write_jsonl(output_path, existing)
    completed = {row["task_id"] for row in existing}
    metrics_path = report_dir / f"metrics-{variant}.json"
    if len(completed) == len(tasks) and metrics_path.exists() and not force:
        cached_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if cached_metrics.get("coverage") == 1.0:
            return cached_metrics
    base_path = snapshot_download(
        repo_id=base_source["repo_id"], revision=base_source["revision"]
    )
    sampler = make_sampler(temp=float(config.section("evaluation_v2")["temperature"]))
    started = time.monotonic()
    stages = (False, True) if variant == "routed" else (False,)
    for use_adapter in stages:
        stage_tasks = [
            task
            for task in tasks
            if task["task_id"] not in completed
            and (route_uses_adapter(task, settings) if variant == "routed" else False)
            == use_adapter
        ]
        if not stage_tasks:
            continue
        model, tokenizer = load(
            base_path,
            adapter_path=str(adapter_path) if use_adapter else None,
            tokenizer_config={"trust_remote_code": True},
        )
        for task in stage_tasks:
            max_tokens = int(
                task.get("max_tokens")
                or (
                    config.section("evaluation_v2")["math_max_new_tokens"]
                    if task["domain"] == "math"
                    else config.section("evaluation_v2")["code_max_new_tokens"]
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
                "route": "adapter" if use_adapter else "base",
                "output": output,
                "score": score,
            }
            append_jsonl(output_path, row)
            existing.append(row)
            completed.add(task["task_id"])
            console.print(
                f"confirm/{variant}: {len(existing)}/{len(tasks)} {task['task_id']} "
                f"{'PASS' if score['passed'] else 'FAIL'}"
            )
        del model, tokenizer
        gc.collect()
        mx.clear_cache()
    summary = _metrics(existing, len(tasks))
    route_counts = {
        route: sum(row.get("route") == route for row in existing)
        for route in ("base", "adapter")
    }
    summary.update(
        {
            "suite": "router-confirm",
            "variant": variant,
            "model": base_source,
            "adapter_sha256": adapter_sha if variant == "routed" else None,
            "route": settings["route"] if variant == "routed" else {"fallback": "qwen35-base"},
            "route_counts": route_counts,
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }
    )
    write_json(metrics_path, summary)
    return summary


def compare_router_confirmation(config: ProjectConfig) -> dict[str, Any]:
    verify_router_recipe(config)
    _, settings = _router_settings(config)
    report_dir = _router_path(config, settings["report_dir"])
    metrics = {
        variant: json.loads(
            (report_dir / f"metrics-{variant}.json").read_text(encoding="utf-8")
        )
        for variant in ("qwen35-base", "routed")
    }
    if any(value["coverage"] != 1.0 for value in metrics.values()):
        raise RuntimeError("Router confirmation evaluation is incomplete")
    base_scores = metrics["qwen35-base"]["scores"]
    routed_scores = metrics["routed"]["scores"]
    shared = sorted(set(base_scores) & set(routed_scores))
    deltas = {
        key: round((routed_scores[key]["accuracy"] - base_scores[key]["accuracy"]) * 100, 2)
        for key in shared
    }
    subgroup_keys = [key for key in shared if key.startswith(("domain:", "language:"))]
    gate = settings["gate"]
    passed = (
        deltas["overall"] >= float(gate["improvement_points"])
        and all(
            deltas[key] >= -float(gate["max_subgroup_regression_points"])
            for key in subgroup_keys
        )
    )
    comparison = {
        "suite": "router-confirm",
        "base": metrics["qwen35-base"],
        "routed": metrics["routed"],
        "delta_percentage_points": deltas,
        "gate_passed": passed,
        "gate": gate,
    }
    write_json(report_dir / "comparison.json", comparison)
    write_json(config.root / "reports" / "v3" / "evaluation.json", comparison)
    return comparison

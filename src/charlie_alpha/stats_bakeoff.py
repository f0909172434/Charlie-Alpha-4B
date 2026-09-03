from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json
from .stats_evolve import _proposal_records, _selector_summary, _start_caffeinate, _surface
from .stats_training import (
    StatsDataset,
    _enable_gradient_checkpointing_once,
    _optimizer,
    _score_loaded_selector,
    stats_iterate_batches,
    stats_loss,
)


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        **_selector_summary(result),
        "domain_accuracy": dict(result["domain_accuracy"]),
        "family_accuracy": dict(result["family_accuracy"]),
    }


def _fixed_smoke_rows(config: ProjectConfig, group_count: int) -> list[dict[str, Any]]:
    path = config.path_for("final_dir") / "dgp-regret" / "train.jsonl"
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        group_id = str(row["metadata"]["semantic_group_id"])
        groups.setdefault(group_id, []).append(row)
    chosen = sorted(groups)[:group_count]
    if len(chosen) != group_count or any(len(groups[group_id]) != 4 for group_id in chosen):
        raise RuntimeError("The base bake-off requires complete four-record smoke groups")
    return [row for group_id in chosen for row in groups[group_id]]


def _score_languages(
    model: Any,
    tokenizer: Any,
    surface: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    model.eval()
    return {
        language: _summary(
            _score_loaded_selector(
                model,
                tokenizer,
                _proposal_records(surface, language=language, view=view),
            )
        )
        for language, view in (
            ("en", "boundary_a"),
            ("zh_Hant", "standard"),
            ("zh_Hans", "standard"),
        )
    }


def _aggregate_languages(languages: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {
        "normalized_regret": float(
            np.mean([float(value["normalized_regret"]) for value in languages.values()])
        ),
        "accuracy": float(np.mean([float(value["accuracy"]) for value in languages.values()])),
        "invalid_selection_rate": float(
            np.mean([float(value["invalid_selection_rate"]) for value in languages.values()])
        ),
    }


def _run_candidate(
    config: ProjectConfig,
    candidate: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    settings = config.section("base_bakeoff")
    source_key = str(candidate["source_key"])
    source = dict(config.sources["models"][source_key])
    snapshot = snapshot_download(repo_id=source["repo_id"], revision=source["revision"])
    mx.clear_cache()
    mx.reset_peak_memory()
    started = time.monotonic()
    model, tokenizer = load(snapshot, tokenizer_config={"trust_remote_code": True})
    load_seconds = time.monotonic() - started
    surface = _surface(config, "dev")
    evaluation_started = time.monotonic()
    base_languages = _score_languages(model, tokenizer, surface)
    base_evaluation_seconds = time.monotonic() - evaluation_started
    base_peak_memory_gb = float(mx.get_peak_memory() / 1e9)

    model.freeze()
    lora = {
        "rank": int(settings["rank"]),
        "scale": float(settings["scale"]),
        "dropout": float(settings["dropout"]),
        "keys": list(settings["targets"]),
    }
    linear_to_lora_layers(
        model,
        int(settings["num_layers"]),
        lora,
        use_dora=False,
    )
    trainable_parameters = sum(
        parameter.size for _, parameter in tree_flatten(model.trainable_parameters())
    )
    seed = int(config.section("project")["seed"])
    mx.random.seed(seed)
    np.random.seed(seed)
    rows = _fixed_smoke_rows(config, int(settings["smoke_groups"]))
    dataset = StatsDataset(
        rows,
        tokenizer,
        seed=seed,
        grouped=True,
        curriculum="active-boundary",
        max_seq_length=int(settings["max_seq_length"]),
        selector_only=True,
    )
    microsteps = int(settings["smoke_microsteps"])
    maximum_selector_tokens = max(len(item["tokens"]) for item in dataset.items)
    optimizer = _optimizer(
        {
            "grad_accumulation_steps": int(settings["grad_accumulation_steps"]),
            "learning_rate_a": float(settings["learning_rate_a"]),
            "learning_rate_b": float(settings["learning_rate_b"]),
            "warmup_fraction": float(settings["warmup_fraction"]),
            "weight_decay": float(settings["weight_decay"]),
        },
        microsteps,
    )
    loss = lambda model, *batch: stats_loss(  # noqa: E731
        model,
        *batch,
        component_weights={"method": 1.0, "plan_tool": 0.0, "report": 0.0},
    )
    _enable_gradient_checkpointing_once(model)
    adapter_dir = output_dir / str(candidate["name"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    training_started = time.monotonic()
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=dataset,
        val_dataset=None,
        args=TrainingArgs(
            batch_size=1,
            iters=microsteps,
            val_batches=0,
            steps_per_report=int(settings["grad_accumulation_steps"]),
            steps_per_eval=microsteps,
            steps_per_save=microsteps,
            max_seq_length=int(settings["max_seq_length"]),
            adapter_file=str(adapter_dir / "adapters.safetensors"),
            grad_checkpoint=False,
            grad_accumulation_steps=int(settings["grad_accumulation_steps"]),
            clear_cache_threshold=int(float(settings["clear_cache_threshold_gb"]) * 1024**3),
        ),
        loss=loss,
        iterate_batches=stats_iterate_batches,
    )
    training_seconds = time.monotonic() - training_started
    # Training keeps optimizer buffers and compiled graphs alive. Release them before
    # autoregressive scoring so the 9B candidate is measured in a clean memory phase.
    del dataset, optimizer
    gc.collect()
    mx.clear_cache()
    post_training_started = time.monotonic()
    model.eval()
    post_smoke_english = _summary(
        _score_loaded_selector(model, tokenizer, _proposal_records(surface))
    )
    post_smoke_evaluation_seconds = time.monotonic() - post_training_started
    peak_memory_gb = float(mx.get_peak_memory() / 1e9)
    result = {
        "schema_version": 1,
        "name": str(candidate["name"]),
        "parameter_scale": str(candidate["parameter_scale"]),
        "source_key": source_key,
        "repo_id": source["repo_id"],
        "revision": source["revision"],
        "license": source["license"],
        "load_seconds": round(load_seconds, 3),
        "base_evaluation_seconds": round(base_evaluation_seconds, 3),
        "base_peak_memory_gb": round(base_peak_memory_gb, 4),
        "base_languages": base_languages,
        "base_trilingual": _aggregate_languages(base_languages),
        "smoke": {
            "groups": int(settings["smoke_groups"]),
            "microsteps": microsteps,
            "optimizer_updates": microsteps // int(settings["grad_accumulation_steps"]),
            "trainable_parameters": trainable_parameters,
            "maximum_selector_tokens": maximum_selector_tokens,
            "training_seconds": round(training_seconds, 3),
            "microsteps_per_second": microsteps / training_seconds,
            "peak_memory_gb": round(peak_memory_gb, 4),
            "post_smoke_english": post_smoke_english,
            "post_smoke_evaluation_seconds": round(post_smoke_evaluation_seconds, 3),
            "adapter_sha256": sha256_file(adapter_dir / "adapters.safetensors"),
        },
    }
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    return result


def choose_base_bakeoff(
    candidates: dict[str, dict[str, Any]],
    gates: dict[str, Any],
) -> dict[str, Any]:
    if set(candidates) != {"qwen3.5-4b", "qwen3.5-9b"}:
        raise ValueError("The bake-off requires qwen3.5-4b and qwen3.5-9b")
    baseline = candidates["qwen3.5-4b"]
    challenger = candidates["qwen3.5-9b"]
    base_regret = float(baseline["base_trilingual"]["normalized_regret"])
    challenger_regret = float(challenger["base_trilingual"]["normalized_regret"])
    relative_regret_improvement = (
        (base_regret - challenger_regret) / base_regret if base_regret else 0.0
    )
    time_ratio = float(challenger["smoke"]["training_seconds"]) / max(
        1e-9, float(baseline["smoke"]["training_seconds"])
    )
    language_noninferiority = all(
        float(challenger["base_languages"][language]["accuracy"])
        >= float(baseline["base_languages"][language]["accuracy"])
        - float(gates["maximum_language_accuracy_regression"])
        for language in ("en", "zh_Hant", "zh_Hans")
    )
    gate_results = {
        "relative_regret": relative_regret_improvement
        >= float(gates["minimum_relative_regret_improvement"]),
        "language_accuracy": language_noninferiority,
        "peak_memory": float(challenger["smoke"]["peak_memory_gb"])
        <= float(gates["maximum_peak_memory_gb"]),
        "training_time_ratio": time_ratio <= float(gates["maximum_training_time_ratio"]),
        "finite_metrics": all(
            math.isfinite(value)
            for value in (
                base_regret,
                challenger_regret,
                time_ratio,
                float(challenger["smoke"]["peak_memory_gb"]),
            )
        ),
    }
    return {
        "recommended": "qwen3.5-9b" if all(gate_results.values()) else "qwen3.5-4b",
        "relative_regret_improvement": relative_regret_improvement,
        "training_time_ratio": time_ratio,
        "gates": gate_results,
        "all_gates_passed": all(gate_results.values()),
    }


def run_base_bakeoff(config: ProjectConfig, *, force: bool = False) -> dict[str, Any]:
    settings = config.section("base_bakeoff")
    output_dir = config.path_for("artifact_dir") / "base-bakeoff"
    status_path = output_dir / "status.json"
    dev_path = config.path_for("stats_dir") / "surface" / "dev.jsonl"
    train_path = config.path_for("final_dir") / "dgp-regret" / "train.jsonl"
    fingerprint = canonical_hash(
        {
            "candidates": [
                {
                    **dict(candidate),
                    "source": config.sources["models"][str(candidate["source_key"])],
                }
                for candidate in settings["candidates"]
            ],
            "settings": settings,
            "dev_sha256": sha256_file(dev_path),
            "train_sha256": sha256_file(train_path),
            "evaluator_version": 2,
        }
    )
    if status_path.exists() and not force:
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint and existing.get("complete"):
            return existing

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    caffeinate = _start_caffeinate()
    previous_cache_limit = mx.set_cache_limit(int(float(settings["cache_limit_gb"]) * 1024**3))
    try:
        for candidate in settings["candidates"]:
            name = str(candidate["name"])
            candidate_path = output_dir / f"{name}.json"
            if candidate_path.exists() and not force:
                existing = json.loads(candidate_path.read_text(encoding="utf-8"))
                if existing.get("fingerprint") == fingerprint and existing.get("complete"):
                    results[name] = dict(existing["result"])
                    continue
            result = _run_candidate(config, dict(candidate), output_dir)
            results[name] = result
            write_json(
                candidate_path,
                {"complete": True, "fingerprint": fingerprint, "result": result},
            )
    finally:
        mx.set_cache_limit(previous_cache_limit)
        if caffeinate is not None and caffeinate.poll() is None:
            caffeinate.terminate()
    decision = choose_base_bakeoff(results, dict(settings["gates"]))
    status = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "surface": "reusable-v0.3-dev",
        "sealed_final_surface_opened": False,
        "candidates": results,
        "decision": decision,
        "claim_boundary": (
            "This local bake-off selects a 72-hour development base; it is not a public capability "
            "claim or a sealed-final result."
        ),
    }
    write_json(status_path, status)
    public = {
        **status,
        "candidates": {
            name: {key: value for key, value in result.items() if key not in {"source_key"}}
            for name, result in results.items()
        },
    }
    write_json(config.root / "reports" / "evolve" / "base-bakeoff.json", public)
    return status

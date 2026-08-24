from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import time
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.trainer import TrainingArgs, evaluate, grad_checkpoint, train
from mlx_lm.tuner.utils import linear_to_lora_layers

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json
from .stats_data import _build_record, _scenario, build_stats_data

_CHECKPOINTED_LAYER_TYPES: set[type[Any]] = set()


def _stats_snapshot(config: ProjectConfig) -> str:
    source = config.sources["models"]["research_base_mlx_4bit"]
    return snapshot_download(repo_id=source["repo_id"], revision=source["revision"])


def _enable_gradient_checkpointing_once(model: Any) -> None:
    for layer in model.layers:
        layer_type = type(layer)
        if layer_type not in _CHECKPOINTED_LAYER_TYPES:
            grad_checkpoint(layer)
            _CHECKPOINTED_LAYER_TYPES.add(layer_type)


def _overlap(offset: tuple[int, int], start: int, end: int) -> bool:
    return offset[1] > start and offset[0] < end


def _interval_mask(
    offsets: list[tuple[int, int]],
    *,
    start: int,
    end: int,
    length: int,
) -> list[bool]:
    mask = [False] * (length - 1)
    for token_index, offset in enumerate(offsets):
        target_index = token_index - 1
        if target_index >= 0 and _overlap(offset, start, end):
            mask[target_index] = True
    return mask


def _single_overlap_token(offsets: list[tuple[int, int]], start: int, end: int) -> int:
    matches = [index for index, offset in enumerate(offsets) if _overlap(offset, start, end)]
    if len(matches) != 1:
        raise RuntimeError(f"selector label must overlap one token, found {matches}")
    return matches[0]


def _tokenize_stats_record(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    rendered = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    backend = tokenizer._tokenizer
    encoded = backend(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    tokens = [int(value) for value in encoded["input_ids"]]
    offsets = [tuple(map(int, value)) for value in encoded["offset_mapping"]]
    method_tag = "<method>"
    method_start = rendered.index(method_tag) + len(method_tag)
    method_end = rendered.index("</method>", method_start)
    method_token_index = _single_overlap_token(offsets, method_start, method_end)
    if method_token_index == 0:
        raise RuntimeError("method selector has no causal input position")

    labels = [str(value) for value in row["metadata"]["candidate_labels"]]
    candidate_token_ids: list[int] = []
    for label in labels:
        replaced = rendered[:method_start] + label + rendered[method_end:]
        candidate_encoding = backend(
            replaced,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        candidate_offsets = [
            tuple(map(int, value)) for value in candidate_encoding["offset_mapping"]
        ]
        candidate_index = _single_overlap_token(
            candidate_offsets, method_start, method_start + len(label)
        )
        if candidate_index != method_token_index:
            raise RuntimeError("candidate label changes the selector token position")
        candidate_token_ids.append(int(candidate_encoding["input_ids"][candidate_index]))
    if len(set(candidate_token_ids)) != len(candidate_token_ids):
        raise RuntimeError("candidate menu labels do not map to distinct tokens")

    plan_start = rendered.index("<analysis_plan>")
    plan_end = rendered.index("</tool_call>", plan_start) + len("</tool_call>")
    report_start = rendered.index("<final_report>")
    report_end = rendered.index("</final_report>", report_start) + len("</final_report>")
    plan_mask = _interval_mask(
        offsets,
        start=plan_start,
        end=plan_end,
        length=len(tokens),
    )
    report_mask = _interval_mask(
        offsets,
        start=report_start,
        end=report_end,
        length=len(tokens),
    )
    if not any(plan_mask) or not any(report_mask):
        raise RuntimeError("component token mask is empty")
    if plan_mask[method_token_index - 1] or report_mask[method_token_index - 1]:
        raise RuntimeError("method target overlaps a text component mask")
    return {
        "tokens": tokens,
        "method_position": method_token_index - 1,
        "candidate_token_ids": candidate_token_ids,
        "candidate_probabilities": [
            float(value) for value in row["metadata"]["method_probabilities"]
        ],
        "plan_mask": plan_mask,
        "report_mask": report_mask,
        "loss_weight": float(row["metadata"]["loss_weight"]),
        "group_id": str(row["metadata"]["semantic_group_id"]),
        "boundary_round": int(row["metadata"]["boundary_round"]),
        "evolution_source": str(row["metadata"].get("evolution_source", "original")),
        "metadata": row["metadata"],
    }


class StatsDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        *,
        seed: int,
        grouped: bool,
        curriculum: str,
        max_seq_length: int,
    ) -> None:
        self.seed = seed
        self.grouped = grouped
        self.curriculum = curriculum
        self.max_seq_length = max_seq_length
        self.items = [_tokenize_stats_record(tokenizer, row) for row in rows]
        oversized = [
            len(item["tokens"]) for item in self.items if len(item["tokens"]) > max_seq_length
        ]
        if oversized:
            raise RuntimeError(
                f"Stats data forbids truncation: maximum {max(oversized)} > {max_seq_length}"
            )
        if grouped:
            by_group: dict[str, list[int]] = {}
            for index, item in enumerate(self.items):
                by_group.setdefault(item["group_id"], []).append(index)
            incomplete = {key: value for key, value in by_group.items() if len(value) != 4}
            if incomplete:
                raise RuntimeError(f"Stats semantic groups must contain four rows: {incomplete}")

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)


def _group_order(dataset: StatsDataset, seed: int, epoch: int) -> list[int]:
    if not dataset.grouped:
        return list(range(len(dataset)))
    groups: dict[str, list[int]] = {}
    boundary: dict[str, int] = {}
    for index, item in enumerate(dataset.items):
        groups.setdefault(item["group_id"], []).append(index)
        boundary[item["group_id"]] = int(item["boundary_round"])
    rng = np.random.default_rng(seed + epoch)
    if dataset.curriculum == "evolve-interleave":
        source = {
            group_id: str(dataset.items[indices[0]]["evolution_source"])
            for group_id, indices in groups.items()
        }

        def ordered(group_ids: list[str]) -> list[str]:
            result: list[str] = []
            for round_index in (2, 1, 0):
                tier = [key for key in group_ids if boundary[key] == round_index]
                rng.shuffle(tier)
                result.extend(tier)
            return result

        fresh = ordered([key for key in groups if source[key] == "new"])
        replay = ordered([key for key in groups if source[key] == "replay"])
        other = ordered([key for key in groups if source[key] not in {"new", "replay"}])
        ordered_groups = []
        while fresh or replay:
            ordered_groups.extend(fresh[:4])
            del fresh[:4]
            if replay:
                ordered_groups.append(replay.pop(0))
        ordered_groups.extend(other)
    elif dataset.curriculum == "active-boundary":
        ordered_groups: list[str] = []
        for round_index in (2, 1, 0):
            tier = [key for key in groups if boundary[key] == round_index]
            rng.shuffle(tier)
            ordered_groups.extend(tier)
    else:
        ordered_groups = list(groups)
        rng.shuffle(ordered_groups)
    return [index for group_id in ordered_groups for index in groups[group_id]]


def stats_iterate_batches(
    dataset: StatsDataset,
    batch_size: int,
    max_seq_length: int,
    loop: bool = False,
    seed: int | None = None,
    comm_group: Any = None,
) -> Iterator[tuple[mx.array, ...]]:
    if batch_size != 1:
        raise ValueError("Stats coupled updates require batch_size=1")
    if comm_group is not None and comm_group.size() != 1:
        raise ValueError("Stats training supports one MLX worker")
    random_seed = dataset.seed if seed is None else seed
    epoch = 0
    while True:
        for index in _group_order(dataset, random_seed, epoch):
            item = dataset[index]
            tokens = item["tokens"]
            if len(tokens) > max_seq_length:
                raise RuntimeError("Stats batches cannot be truncated")
            padded_length = min(
                max_seq_length,
                1 + 32 * ((len(tokens) + 31) // 32),
            )
            batch = np.zeros((1, padded_length), dtype=np.int32)
            batch[0, : len(tokens)] = tokens
            plan_mask = np.zeros((1, padded_length - 1), dtype=np.bool_)
            report_mask = np.zeros((1, padded_length - 1), dtype=np.bool_)
            plan_mask[0, : len(item["plan_mask"])] = item["plan_mask"]
            report_mask[0, : len(item["report_mask"])] = item["report_mask"]
            candidate_ids = np.zeros((1, 6), dtype=np.int32)
            candidate_probs = np.zeros((1, 6), dtype=np.float32)
            candidate_mask = np.zeros((1, 6), dtype=np.bool_)
            count = len(item["candidate_token_ids"])
            candidate_ids[0, :count] = item["candidate_token_ids"]
            candidate_probs[0, :count] = item["candidate_probabilities"]
            candidate_mask[0, :count] = True
            yield (
                mx.array(batch),
                mx.array([item["method_position"]], dtype=mx.int32),
                mx.array(candidate_ids),
                mx.array(candidate_probs),
                mx.array(candidate_mask),
                mx.array(plan_mask),
                mx.array(report_mask),
                mx.array([item["loss_weight"]], dtype=mx.float32),
            )
        if not loop:
            break
        epoch += 1


def stats_loss(
    model: Any,
    batch: mx.array,
    method_positions: mx.array,
    candidate_ids: mx.array,
    candidate_probs: mx.array,
    candidate_mask: mx.array,
    plan_mask: mx.array,
    report_mask: mx.array,
    sample_weights: mx.array,
    *,
    component_weights: dict[str, float] | None = None,
) -> tuple[mx.array, mx.array]:
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    rows = mx.arange(batch.shape[0])
    selector_logits = logits[rows, method_positions, :]
    menu_logits = mx.take_along_axis(selector_logits, candidate_ids, axis=-1)
    menu_logits = mx.where(candidate_mask, menu_logits, mx.array(-1e9, menu_logits.dtype))
    menu_log_probs = menu_logits - mx.logsumexp(menu_logits, axis=-1, keepdims=True)
    method_loss = -(candidate_probs * menu_log_probs).sum(axis=-1)

    cross_entropy = nn.losses.cross_entropy(logits, targets)
    plan_float = plan_mask.astype(cross_entropy.dtype)
    report_float = report_mask.astype(cross_entropy.dtype)
    plan_tokens = mx.maximum(plan_float.sum(axis=-1), mx.array(1.0))
    report_tokens = mx.maximum(report_float.sum(axis=-1), mx.array(1.0))
    plan_loss = (cross_entropy * plan_float).sum(axis=-1) / plan_tokens
    report_loss = (cross_entropy * report_float).sum(axis=-1) / report_tokens
    weights = component_weights or {"method": 0.45, "plan_tool": 0.35, "report": 0.20}
    if not math.isclose(sum(float(value) for value in weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Stats component weights must sum to one")
    component_loss = (
        float(weights["method"]) * method_loss
        + float(weights["plan_tool"]) * plan_loss
        + float(weights["report"]) * report_loss
    )
    loss = (component_loss * sample_weights).astype(mx.float32).mean()
    token_count = plan_float.sum() + report_float.sum() + batch.shape[0]
    return loss, token_count


def _schedule(learning_rate: float, updates: int, warmup_fraction: float) -> Any:
    warmup = max(1, round(updates * warmup_fraction))
    decay = max(1, updates - warmup)
    warmup_schedule = optim.schedulers.linear_schedule(
        learning_rate / warmup,
        learning_rate,
        max(1, warmup - 1),
    )
    cosine_schedule = optim.schedulers.cosine_decay(
        learning_rate,
        decay,
        learning_rate * 0.1,
    )
    return optim.schedulers.join_schedules(
        [warmup_schedule, cosine_schedule],
        [warmup],
    )


def _optimizer(settings: dict[str, Any], microsteps: int) -> optim.Optimizer:
    updates = math.ceil(microsteps / int(settings["grad_accumulation_steps"]))
    kwargs = {"weight_decay": float(settings["weight_decay"])}
    optimizer_a = optim.AdamW(
        learning_rate=_schedule(
            float(settings["learning_rate_a"]),
            updates,
            float(settings["warmup_fraction"]),
        ),
        **kwargs,
    )
    optimizer_b = optim.AdamW(
        learning_rate=_schedule(
            float(settings["learning_rate_b"]),
            updates,
            float(settings["warmup_fraction"]),
        ),
        **kwargs,
    )
    return optim.MultiOptimizer(
        [optimizer_a, optimizer_b],
        filters=[lambda path, _: path.endswith("lora_a")],
    )


class _StopTraining(RuntimeError):
    pass


class _StatsCallback(TrainingCallback):
    def __init__(
        self,
        *,
        model: Any,
        best_path: Path,
        deadline: float,
        patience: int | None,
    ) -> None:
        self.model = model
        self.best_path = best_path
        self.deadline = deadline
        self.patience = patience
        self.train_history: list[dict[str, Any]] = []
        self.validation_history: list[dict[str, Any]] = []
        self.best_loss: float | None = None
        self.best_iteration: int | None = None
        self.stale = 0
        self.stop_reason: str | None = None

    def _save_best(self) -> None:
        weights = dict(tree_flatten(self.model.trainable_parameters()))
        mx.save_safetensors(str(self.best_path), weights)

    def consider_validation(self, iteration: int, loss: float) -> None:
        self.validation_history.append({"iteration": iteration, "loss": loss})
        if self.best_loss is None or loss < self.best_loss - 1e-4:
            self.best_loss = loss
            self.best_iteration = iteration
            self.stale = 0
            self._save_best()
        elif iteration > 0:
            self.stale += 1

    def _check_stop(self) -> None:
        if self.patience is not None and self.stale >= self.patience:
            self.stop_reason = "early_stopping"
            raise _StopTraining("Stats validation early stopping")
        if time.monotonic() >= self.deadline:
            self.stop_reason = "deadline"
            raise _StopTraining("Stats training deadline")

    def on_train_loss_report(self, train_info: dict[str, Any]) -> None:
        self.train_history.append(dict(train_info))
        self._check_stop()

    def on_val_loss_report(self, val_info: dict[str, Any]) -> None:
        self.consider_validation(int(val_info["iteration"]), float(val_info["val_loss"]))
        self._check_stop()


def _adapter_config(
    config: ProjectConfig,
    *,
    variant: str,
    model_path: str,
    adapter_dir: Path,
    max_seq_length: int,
    rank: int,
) -> dict[str, Any]:
    settings = config.section("stats_training")
    return {
        "model": model_path,
        "fine_tune_type": "lora",
        "num_layers": int(settings["num_layers"]),
        "max_seq_length": max_seq_length,
        "adapter_path": str(adapter_dir),
        "lora_parameters": {
            "rank": rank,
            "scale": float(settings["scale"]),
            "dropout": float(settings["dropout"]),
            "keys": list(settings["targets"]),
        },
        "stats": {
            "method": "DGP-Regret",
            "variant": variant,
            "component_weights": settings["component_weights"],
            "learning_rate_a": float(settings["learning_rate_a"]),
            "learning_rate_b": float(settings["learning_rate_b"]),
            "user_and_tool_outputs_masked": True,
        },
        "base_model_repo": config.sources["models"]["research_base_mlx_4bit"]["repo_id"],
        "base_model_revision": config.sources["models"]["research_base_mlx_4bit"]["revision"],
    }


def _start_caffeinate() -> subprocess.Popen[bytes] | None:
    executable = Path("/usr/bin/caffeinate")
    if not executable.exists():
        return None
    return subprocess.Popen(
        [str(executable), "-dimsu", "-w", str(os.getpid())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _normalize_training_progress(
    status: dict[str, Any], *, grad_accumulation_steps: int
) -> dict[str, Any]:
    """Separate an early-stopped run's actual work from its configured ceiling."""
    normalized = dict(status)
    planned = int(normalized.get("planned_microsteps", normalized["microsteps"]))
    stopped = bool(normalized.get("stopped", False))
    validation_history = [dict(item) for item in normalized.get("validation_history", [])]
    if stopped:
        # Trainer v3 recorded its final post-stop loss at the configured ceiling. The
        # real validation immediately before it is zero-indexed (1119 means 1120
        # completed microsteps), so the ceiling record is metadata rather than work.
        validation_history = [
            item for item in validation_history if int(item.get("iteration", 0)) < planned
        ]
        observed = [
            int(item.get("iteration", 0))
            for item in normalized.get("train_history", [])
        ]
        observed.extend(
            int(item.get("iteration", 0)) + 1
            for item in validation_history
            if int(item.get("iteration", 0)) > 0
        )
        completed = max(observed, default=0)
    else:
        completed = planned
    normalized["planned_microsteps"] = planned
    normalized["microsteps"] = completed
    normalized["optimizer_updates"] = completed // grad_accumulation_steps
    normalized["validation_history"] = validation_history
    best_iteration = normalized.get("best_validation_iteration")
    normalized["best_validation_microstep"] = (
        int(best_iteration) + 1 if best_iteration is not None and int(best_iteration) > 0 else 0
    )
    return normalized


def _train_stats_variant(
    config: ProjectConfig,
    *,
    variant: str,
    adapter_dir: Path,
    microsteps: int,
    max_seconds: int,
    max_seq_length: int,
    rank: int,
    early_stop_patience: int | None,
    force: bool,
) -> dict[str, Any]:
    build_stats_data(config, force=False)
    settings = config.section("stats_training")
    final_dir = config.path_for("final_dir") / variant
    fingerprint = canonical_hash(
        {
            "variant": variant,
            "training": settings,
            "microsteps": microsteps,
            "max_seq_length": max_seq_length,
            "rank": rank,
            "train_sha256": sha256_file(final_dir / "train.jsonl"),
            "valid_sha256": sha256_file(final_dir / "valid.jsonl"),
            "base": config.sources["models"]["research_base_mlx_4bit"],
            "trainer_version": 3,
        }
    )
    status_path = adapter_dir / "status.json"
    if status_path.exists() and not force:
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint and existing.get("complete"):
            normalized = _normalize_training_progress(
                existing,
                grad_accumulation_steps=int(settings["grad_accumulation_steps"]),
            )
            if normalized != existing:
                write_json(status_path, normalized)
            return normalized

    adapter_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.section("project")["seed"])
    mx.random.seed(seed)
    np.random.seed(seed)
    model_path = _stats_snapshot(config)
    model, tokenizer = load(model_path, tokenizer_config={"trust_remote_code": True})
    model.freeze()
    lora_parameters = {
        "rank": rank,
        "scale": float(settings["scale"]),
        "dropout": float(settings["dropout"]),
        "keys": list(settings["targets"]),
    }
    linear_to_lora_layers(
        model,
        int(settings["num_layers"]),
        lora_parameters,
        use_dora=False,
    )
    trainable_parameters = sum(
        parameter.size for _, parameter in tree_flatten(model.trainable_parameters())
    )
    write_json(
        adapter_dir / "adapter_config.json",
        _adapter_config(
            config,
            variant=variant,
            model_path=model_path,
            adapter_dir=adapter_dir,
            max_seq_length=max_seq_length,
            rank=rank,
        ),
    )
    train_rows = list(read_jsonl(final_dir / "train.jsonl"))
    valid_rows = list(read_jsonl(final_dir / "valid.jsonl"))
    curriculum = "active-boundary" if variant == "dgp-regret" else "random"
    train_dataset = StatsDataset(
        train_rows,
        tokenizer,
        seed=seed,
        grouped=True,
        curriculum=curriculum,
        max_seq_length=max_seq_length,
    )
    valid_dataset = StatsDataset(
        valid_rows,
        tokenizer,
        seed=seed,
        grouped=False,
        curriculum="random",
        max_seq_length=max_seq_length,
    )
    group_size = int(settings["grad_accumulation_steps"])
    training_args = TrainingArgs(
        batch_size=1,
        iters=microsteps,
        val_batches=-1,
        steps_per_report=group_size,
        steps_per_eval=min(int(settings["validation_every"]), microsteps),
        steps_per_save=min(int(settings["checkpoint_every"]), microsteps),
        max_seq_length=max_seq_length,
        adapter_file=str(adapter_dir / "adapters.safetensors"),
        grad_checkpoint=False,
        grad_accumulation_steps=group_size,
        clear_cache_threshold=12 * 1024**3,
    )
    optimizer = _optimizer(settings, microsteps)
    _enable_gradient_checkpointing_once(model)
    started = time.monotonic()
    callback = _StatsCallback(
        model=model,
        best_path=adapter_dir / "best_adapters.safetensors",
        deadline=started + max_seconds,
        patience=early_stop_patience,
    )
    stopped = False
    try:
        train(
            model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=valid_dataset,
            args=training_args,
            loss=stats_loss,
            iterate_batches=stats_iterate_batches,
            training_callback=callback,
        )
    except _StopTraining:
        stopped = True
        mx.save_safetensors(
            str(adapter_dir / "adapters.safetensors"),
            dict(tree_flatten(model.trainable_parameters())),
        )

    final_loss = float(
        evaluate(
            model=model,
            dataset=valid_dataset,
            batch_size=1,
            num_batches=-1,
            max_seq_length=max_seq_length,
            loss=stats_loss,
            iterate_batches=stats_iterate_batches,
            clear_cache_threshold=12 * 1024**3,
        )
    )
    if not callback.validation_history:
        raise RuntimeError("Stats trainer did not run its required initial validation")
    initial_loss = float(callback.validation_history[0]["loss"])
    observed_train_steps = [
        int(item.get("iteration", 0)) for item in callback.train_history
    ]
    observed_validation_steps = [
        int(item.get("iteration", 0)) + 1
        for item in callback.validation_history
        if int(item.get("iteration", 0)) > 0
    ]
    completed_microsteps = (
        max(observed_train_steps + observed_validation_steps, default=0)
        if stopped
        else microsteps
    )
    if max(observed_validation_steps, default=0) < completed_microsteps:
        callback.consider_validation(completed_microsteps, final_loss)
    best_path = adapter_dir / "best_adapters.safetensors"
    active_path = adapter_dir / "adapters.safetensors"
    if best_path.exists():
        model.load_weights(str(best_path), strict=False)
        mx.save_safetensors(
            str(active_path),
            dict(tree_flatten(model.trainable_parameters())),
        )
    status = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "complete": active_path.exists(),
        "stopped": stopped,
        "stop_reason": callback.stop_reason or "completed",
        "variant": variant,
        "adapter_path": str(adapter_dir),
        "adapter_sha256": sha256_file(active_path),
        "base_model_path": model_path,
        "planned_microsteps": microsteps,
        "microsteps": completed_microsteps,
        "optimizer_updates": completed_microsteps // group_size,
        "trainable_parameters": trainable_parameters,
        "max_seq_length": max_seq_length,
        "rank": rank,
        "initial_validation_loss": initial_loss,
        "final_validation_loss": final_loss,
        "best_validation_loss": callback.best_loss,
        "best_validation_iteration": callback.best_iteration,
        "best_validation_microstep": (
            int(callback.best_iteration) + 1
            if callback.best_iteration is not None and callback.best_iteration > 0
            else 0
        ),
        "validation_history": callback.validation_history,
        "train_history": callback.train_history,
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 4),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "checkpoint_every_microsteps": int(settings["checkpoint_every"]),
    }
    write_json(status_path, status)
    del model, tokenizer, train_dataset, valid_dataset, optimizer
    gc.collect()
    mx.clear_cache()
    return status


def _evaluation_rows(
    config: ProjectConfig, split: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    surface_path = config.path_for("stats_dir") / "surface" / f"{split}.jsonl"
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for simulation in read_jsonl(surface_path):
        scenario = _scenario(simulation["scenario"])
        record = _build_record(
            scenario,
            simulation,
            language="en",
            loss_weight=1.0,
            incomplete=False,
            variant="dgp-regret",
            refined_explanation=None,
        )
        rows.append((record, simulation))
    return rows


def _score_loaded_selector(
    model: Any,
    tokenizer: Any,
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    regrets: list[float] = []
    correct = 0
    invalid = 0
    predictions: list[dict[str, Any]] = []
    domain_correct: dict[str, list[bool]] = {}
    family_correct: dict[str, list[bool]] = {}
    for record, simulation in rows:
        item = _tokenize_stats_record(tokenizer, record)
        tokens = mx.array([item["tokens"]], dtype=mx.int32)
        logits = model(tokens[:, :-1])
        selector = logits[0, int(item["method_position"]), :]
        candidate_ids = mx.array(item["candidate_token_ids"], dtype=mx.int32)
        menu_logits = mx.take(selector, candidate_ids)
        predicted_index = int(mx.argmax(menu_logits).item())
        method_id = str(record["metadata"]["candidate_method_ids"][predicted_index])
        selected = str(simulation["selected_method_id"])
        if method_id == "needs_clarification":
            regret = 1.0
            is_valid = False
        else:
            metric = next(
                candidate
                for candidate in simulation["candidates"]
                if candidate["method_id"] == method_id
            )
            regret = float(metric["normalized_regret"])
            is_valid = bool(metric["valid"])
        regrets.append(regret)
        is_correct = method_id == selected
        correct += int(is_correct)
        invalid += int(not is_valid)
        scenario = simulation["scenario"]
        domain_correct.setdefault(str(scenario["domain"]), []).append(is_correct)
        family_correct.setdefault(str(scenario["family_id"]), []).append(is_correct)
        predictions.append(
            {
                "blueprint_id": simulation["scenario"]["blueprint_id"],
                "family_id": scenario["family_id"],
                "domain": scenario["domain"],
                "predicted_method_id": method_id,
                "oracle_method_id": selected,
                "normalized_regret": regret,
                "valid": is_valid,
            }
        )
        del logits, selector, menu_logits
    count = len(rows)
    return {
        "count": count,
        "normalized_regret": float(np.mean(regrets)) if regrets else 1.0,
        "accuracy": correct / count if count else 0.0,
        "invalid_selection_rate": invalid / count if count else 1.0,
        "domain_accuracy": {
            key: sum(values) / len(values) for key, values in sorted(domain_correct.items())
        },
        "family_accuracy": {
            key: sum(values) / len(values) for key, values in sorted(family_correct.items())
        },
        "predictions": predictions,
    }


def score_stats_selector(
    config: ProjectConfig,
    *,
    adapter_path: Path | None,
    split: str = "dev",
    delta_scale: float = 1.0,
) -> dict[str, Any]:
    model_path = _stats_snapshot(config)
    model, tokenizer = load(
        model_path,
        adapter_path=str(adapter_path) if adapter_path is not None else None,
        tokenizer_config={"trust_remote_code": True},
    )
    if adapter_path is not None and delta_scale != 1.0:
        for _, module in model.named_modules():
            if hasattr(module, "lora_a") and hasattr(module, "lora_b"):
                module.scale = float(module.scale) * delta_scale
    rows = _evaluation_rows(config, split)
    result = _score_loaded_selector(model, tokenizer, rows)
    result.update(
        {
            "split": split,
            "adapter_path": str(adapter_path) if adapter_path else None,
            "delta_scale": delta_scale,
        }
    )
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    return result


def run_stats_pilot_candidate(
    config: ProjectConfig,
    *,
    variant: str,
    max_seq_length: int | None = None,
    rank: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    settings = config.section("stats_training")
    variant_names = {str(item["name"]) for item in settings["variants"]}
    if variant not in variant_names:
        raise ValueError(f"Unknown stats pilot variant: {variant}")
    effective_length = int(max_seq_length or settings["max_seq_length"])
    effective_rank = int(rank or settings["rank"])
    label = f"seq{effective_length}-r{effective_rank}"
    adapter_dir = config.path_for("artifact_dir") / "pilots" / label / variant
    budget = int(config.section("stats_budget")["pilot_seconds"])
    per_candidate = max(600, budget // len(variant_names))
    result = _train_stats_variant(
        config,
        variant=variant,
        adapter_dir=adapter_dir,
        microsteps=int(settings["pilot_microsteps"]),
        max_seconds=per_candidate,
        max_seq_length=effective_length,
        rank=effective_rank,
        early_stop_patience=None,
        force=force,
    )
    dev = score_stats_selector(config, adapter_path=adapter_dir, split="dev")
    result["dev"] = dev
    write_json(adapter_dir / "status.json", result)
    return result


def _looks_like_metal_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return any(
        term in text for term in ("out of memory", "metal", "allocator", "resource exhausted")
    )


def run_stats_pilots(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    build_stats_data(config, force=False)
    settings = config.section("stats_training")
    profiles = [
        (int(settings["max_seq_length"]), int(settings["rank"]), "planned"),
        (int(settings["oom_seq_length"]), int(settings["rank"]), "sequence-oom-fallback"),
        (int(settings["oom_seq_length"]), int(settings["oom_rank"]), "rank-oom-fallback"),
    ]
    caffeinate = _start_caffeinate()
    last_error: BaseException | None = None
    try:
        for max_length, rank, reason in profiles:
            results: list[dict[str, Any]] = []
            try:
                for variant in settings["variants"]:
                    results.append(
                        run_stats_pilot_candidate(
                            config,
                            variant=str(variant["name"]),
                            max_seq_length=max_length,
                            rank=rank,
                            force=force,
                        )
                    )
            except BaseException as error:
                last_error = error
                if _looks_like_metal_oom(error):
                    gc.collect()
                    mx.clear_cache()
                    continue
                raise
            parameter_counts = {int(result["trainable_parameters"]) for result in results}
            microsteps = {int(result["microsteps"]) for result in results}
            if len(parameter_counts) != 1 or len(microsteps) != 1:
                raise RuntimeError(
                    "Stats ablations do not have equal compute and parameter budgets"
                )
            selected = min(
                results,
                key=lambda result: (
                    float(result["dev"]["normalized_regret"]),
                    -float(result["dev"]["accuracy"]),
                    float(result["best_validation_loss"]),
                    str(result["variant"]),
                ),
            )
            comparison = {
                "schema_version": 1,
                "selection_policy": (
                    "dev normalized regret, then statistical method accuracy, then validation loss"
                ),
                "resource_profile": {
                    "max_seq_length": max_length,
                    "rank": rank,
                    "reason": reason,
                },
                "equal_microsteps": microsteps.pop(),
                "equal_trainable_parameters": parameter_counts.pop(),
                "candidates": results,
                "selected_variant": selected["variant"],
            }
            artifact_dir = config.path_for("artifact_dir")
            write_json(artifact_dir / "pilot-comparison.json", comparison)
            write_json(artifact_dir / "pilot-selected.json", selected)
            return comparison
    finally:
        if caffeinate is not None and caffeinate.poll() is None:
            caffeinate.terminate()
    if last_error is not None:
        raise RuntimeError("All fair stats pilot resource profiles failed") from last_error
    raise RuntimeError("No stats pilot resource profile completed")


def run_stats_training(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    artifact_dir = config.path_for("artifact_dir")
    comparison_path = artifact_dir / "pilot-comparison.json"
    if not comparison_path.exists():
        run_stats_pilots(config, force=False)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    profile = comparison["resource_profile"]
    variant = str(comparison["selected_variant"])
    settings = config.section("stats_training")
    train_count = sum(1 for _ in read_jsonl(config.path_for("final_dir") / variant / "train.jsonl"))
    microsteps = train_count * int(settings["full_epochs"])
    adapter_dir = artifact_dir / "adapters" / "selected-raw"
    caffeinate = _start_caffeinate()
    try:
        result = _train_stats_variant(
            config,
            variant=variant,
            adapter_dir=adapter_dir,
            microsteps=microsteps,
            max_seconds=int(settings["max_seconds"]),
            max_seq_length=int(profile["max_seq_length"]),
            rank=int(profile["rank"]),
            early_stop_patience=int(settings["early_stop_evaluations"]),
            force=force,
        )
    finally:
        if caffeinate is not None and caffeinate.poll() is None:
            caffeinate.terminate()
    result["pilot_comparison"] = str(comparison_path)
    result["selected_by"] = comparison["selection_policy"]
    write_json(artifact_dir / "trained.json", result)
    return result


def _scale_adapter(source: Path, destination: Path, scale: float) -> dict[str, Any]:
    source_file = source / "adapters.safetensors"
    weights = mx.load(str(source_file))
    lora_b = [key for key in weights if key.endswith("lora_b")]
    if not lora_b:
        raise RuntimeError("Stats adapter has no LoRA B matrices")
    destination.mkdir(parents=True, exist_ok=True)
    scaled = {
        key: value * scale if key.endswith("lora_b") else value for key, value in weights.items()
    }
    output_file = destination / "adapters.safetensors"
    mx.save_safetensors(str(output_file), scaled)
    adapter_config = json.loads((source / "adapter_config.json").read_text(encoding="utf-8"))
    adapter_config.setdefault("stats", {})["post_training_delta_scale"] = scale
    write_json(destination / "adapter_config.json", adapter_config)
    return {
        "adapter_path": str(destination),
        "adapter_sha256": sha256_file(output_file),
        "delta_scale": scale,
    }


def _normalize_answer(value: str) -> str:
    compatible = unicodedata.normalize("NFKC", value)
    return "".join(
        character.lower() for character in compatible if character.isalnum()
    )


def _retention_score(model: Any, tokenizer: Any, config: ProjectConfig) -> dict[str, Any]:
    path = config.root / "configs" / "retention.stats.jsonl"
    rows = list(read_jsonl(path))
    passed = 0
    details: list[dict[str, Any]] = []
    groups: dict[str, list[bool]] = {}
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        answer = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=int(row["max_tokens"]),
            sampler=make_sampler(temp=0.0),
            verbose=False,
        )
        ok = _normalize_answer(str(row["gold"])) in _normalize_answer(answer)
        passed += int(ok)
        groups.setdefault(f"language:{row['language']}", []).append(ok)
        groups.setdefault(f"domain:{row['domain']}", []).append(ok)
        details.append({"task_id": row["task_id"], "passed": ok, "answer": answer})
    group_scores = {
        key: sum(values) / len(values) for key, values in sorted(groups.items())
    }
    return {
        "accuracy": passed / len(rows),
        "worst_group_accuracy": min(group_scores.values()),
        "groups": group_scores,
        "count": len(rows),
        "details": details,
    }


def calibrate_stats_adapter(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    artifact_dir = config.path_for("artifact_dir")
    trained_path = artifact_dir / "trained.json"
    if not trained_path.exists():
        run_stats_training(config, force=False)
    trained = json.loads(trained_path.read_text(encoding="utf-8"))
    source = Path(trained["adapter_path"])
    calibration = config.section("stats_calibration")
    candidates: list[dict[str, Any]] = []
    for scale_value in calibration["candidate_delta_scales"]:
        scale = float(scale_value)
        label = str(scale).replace(".", "p")
        destination = artifact_dir / "adapters" / f"calibrated-scale-{label}"
        status_path = destination / "status.json"
        fingerprint = canonical_hash(
            {
                "raw": sha256_file(source / "adapters.safetensors"),
                "scale": scale,
                "retention": sha256_file(config.root / "configs" / "retention.stats.jsonl"),
                "calibration_version": 4,
            }
        )
        if status_path.exists() and not force:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("fingerprint") == fingerprint:
                candidates.append(status)
                continue
        scaled = _scale_adapter(source, destination, scale)
        model, tokenizer = load(
            _stats_snapshot(config),
            adapter_path=str(destination),
            tokenizer_config={"trust_remote_code": True},
        )
        dev = _score_loaded_selector(model, tokenizer, _evaluation_rows(config, "dev"))
        retention = _retention_score(model, tokenizer, config)
        status = {
            **scaled,
            "fingerprint": fingerprint,
            "dev": dev,
            "retention": retention,
        }
        write_json(status_path, status)
        candidates.append(status)
        del model, tokenizer
        gc.collect()
        mx.clear_cache()
    selected = min(
        candidates,
        key=lambda item: (
            float(item["dev"]["invalid_selection_rate"]),
            float(item["dev"]["normalized_regret"]),
            -float(item["retention"]["worst_group_accuracy"]),
            -float(item["retention"]["accuracy"]),
            -float(item["delta_scale"]),
        ),
    )
    result = {
        **trained,
        **selected,
        "raw_adapter_path": str(source),
        "calibration_candidates": candidates,
        "calibration_selection_order": calibration["selection_order"],
        "selected_by": "validity, then normalized regret, then retention",
        "recipe_frozen": False,
    }
    write_json(artifact_dir / "selected.json", result)
    write_json(artifact_dir / "calibration.json", result)
    return result

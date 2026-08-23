from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.trainer import TrainingArgs, evaluate, train
from mlx_lm.tuner.utils import linear_to_lora_layers
from rich.console import Console

from .config import ProjectConfig
from .forge_data import _tokenize_chat, build_forge_data
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json

console = Console()


class ForgeDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        *,
        group_size: int,
        seed: int,
        grouped: bool,
    ) -> None:
        self.group_size = group_size
        self.seed = seed
        self.grouped = grouped
        self.items: list[tuple[list[int], int, list[int], float]] = []
        for row in rows:
            tokens, offset = _tokenize_chat(tokenizer, row["messages"])
            metadata = row["metadata"]
            selected = metadata.get("selective_target_indices")
            if selected is None:
                selected = list(range(max(0, offset - 1), len(tokens) - 1))
            self.items.append(
                (
                    tokens,
                    offset,
                    [int(index) for index in selected],
                    float(metadata.get("loss_weight", 1.0)),
                )
            )
        if grouped and len(self.items) % group_size:
            raise ValueError("Grouped Forge data must contain complete semantic groups")

    def __getitem__(self, index: int) -> tuple[list[int], int, list[int], float]:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)


def forge_iterate_batches(
    dataset: ForgeDataset,
    batch_size: int,
    max_seq_length: int,
    loop: bool = False,
    seed: int | None = None,
    comm_group: Any = None,
) -> Iterator[tuple[mx.array, mx.array, mx.array, mx.array]]:
    if batch_size != 1:
        raise ValueError("Forge currently requires batch_size=1 for coupled updates")
    if comm_group is not None and comm_group.size() != 1:
        raise ValueError("Forge coupled ordering currently supports one MLX worker")
    random_seed = dataset.seed if seed is None else seed
    epoch = 0
    while True:
        if dataset.grouped:
            groups = [
                list(range(start, start + dataset.group_size))
                for start in range(0, len(dataset), dataset.group_size)
            ]
            rng = np.random.default_rng(random_seed + epoch)
            order = rng.permutation(len(groups))
            indices = [index for group_index in order for index in groups[int(group_index)]]
        else:
            indices = list(range(len(dataset)))

        for index in indices:
            tokens, offset, selected, weight = dataset[index]
            if len(tokens) > max_seq_length:
                raise RuntimeError(
                    f"Forge forbids truncation: {len(tokens)} > {max_seq_length} tokens"
                )
            padded_length = min(
                max_seq_length,
                1 + 32 * ((len(tokens) + 31) // 32),
            )
            batch = np.zeros((1, padded_length), dtype=np.int32)
            batch[0, : len(tokens)] = tokens
            target_mask = np.zeros((1, padded_length - 1), dtype=np.bool_)
            valid_indices = [item for item in selected if 0 <= item < len(tokens) - 1]
            target_mask[0, valid_indices] = True
            if not valid_indices:
                raise RuntimeError("Forge selective loss produced an empty target mask")
            yield (
                mx.array(batch),
                mx.array([[offset, len(tokens)]], dtype=mx.int32),
                mx.array(target_mask),
                mx.array([weight], dtype=mx.float32),
            )
        if not loop:
            break
        epoch += 1


def forge_loss(
    model: Any,
    batch: mx.array,
    lengths: mx.array,
    target_mask: mx.array,
    weights: mx.array,
) -> tuple[mx.array, mx.array]:
    del lengths
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    cross_entropy = nn.losses.cross_entropy(logits, targets)
    mask = target_mask.astype(cross_entropy.dtype)
    tokens = mask.sum()
    loss = (cross_entropy * mask * weights[:, None]).astype(mx.float32).sum() / tokens
    return loss, tokens


def _schedule(learning_rate: float, updates: int, warmup_fraction: float) -> Any:
    warmup = max(1, round(updates * warmup_fraction))
    decay = max(1, updates - warmup)
    warmup_schedule = optim.schedulers.linear_schedule(0.0, learning_rate, warmup)
    cosine_schedule = optim.schedulers.cosine_decay(
        learning_rate, decay, learning_rate * 0.1
    )
    return optim.schedulers.join_schedules(
        [warmup_schedule, cosine_schedule], [warmup + 1]
    )


def _optimizer(
    candidate: dict[str, Any], settings: dict[str, Any], microsteps: int
) -> optim.Optimizer:
    updates = math.ceil(microsteps / int(settings["grad_accumulation_steps"]))
    kwargs = {"weight_decay": float(settings["weight_decay"])}
    optimizer_a = optim.AdamW(
        learning_rate=_schedule(
            float(candidate["learning_rate_a"]),
            updates,
            float(settings["warmup_fraction"]),
        ),
        **kwargs,
    )
    optimizer_b = optim.AdamW(
        learning_rate=_schedule(
            float(candidate["learning_rate_b"]),
            updates,
            float(settings["warmup_fraction"]),
        ),
        **kwargs,
    )
    return optim.MultiOptimizer(
        [optimizer_a, optimizer_b],
        filters=[lambda path, _: path.endswith("lora_a")],
    )


def _research_snapshot(config: ProjectConfig) -> str:
    source = config.sources["models"]["research_base_mlx_4bit"]
    return snapshot_download(repo_id=source["repo_id"], revision=source["revision"])


def _adapter_config(
    config: ProjectConfig,
    candidate: dict[str, Any],
    model_path: str,
    adapter_dir: Path,
) -> dict[str, Any]:
    settings = config.section("training_v2")
    return {
        "model": model_path,
        "fine_tune_type": "lora",
        "num_layers": int(candidate["num_layers"]),
        "max_seq_length": int(settings["max_seq_length"]),
        "adapter_path": str(adapter_dir),
        "lora_parameters": {
            "rank": int(candidate["rank"]),
            "scale": float(settings["scale"]),
            "dropout": float(settings["dropout"]),
            "keys": list(settings["targets"]),
        },
        "forge": {
            "candidate": candidate["name"],
            "learning_rate_a": float(candidate["learning_rate_a"]),
            "learning_rate_b": float(candidate["learning_rate_b"]),
            "loss": "teacher-student-positive-excess-token-mask",
            "batch_order": "triad-coupled-groups",
        },
        "base_model_repo": config.sources["models"]["research_base_mlx_4bit"]["repo_id"],
        "base_model_revision": config.sources["models"]["research_base_mlx_4bit"][
            "revision"
        ],
    }


class _StopTraining(RuntimeError):
    pass


class _ForgeCallback(TrainingCallback):
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

    def _check_deadline(self) -> None:
        if time.monotonic() >= self.deadline:
            self.stop_reason = "deadline"
            raise _StopTraining("Forge training deadline reached")

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

    def on_train_loss_report(self, train_info: dict[str, Any]) -> None:
        self.train_history.append(dict(train_info))
        self._check_deadline()

    def on_val_loss_report(self, val_info: dict[str, Any]) -> None:
        self.consider_validation(
            int(val_info["iteration"]), float(val_info["val_loss"])
        )
        if self.patience is not None and self.stale >= self.patience:
            self.stop_reason = "early_stopping"
            raise _StopTraining("Forge validation early stopping triggered")
        self._check_deadline()


def _start_caffeinate() -> subprocess.Popen[bytes] | None:
    executable = Path("/usr/bin/caffeinate")
    if not executable.exists():
        return None
    return subprocess.Popen(
        [str(executable), "-dimsu", "-w", str(os.getpid())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _train_candidate(
    config: ProjectConfig,
    *,
    candidate: dict[str, Any],
    adapter_dir: Path,
    microsteps: int,
    max_seconds: int,
    early_stop_patience: int | None,
    force: bool,
) -> dict[str, Any]:
    final_dir = config.path_for("final_dir")
    settings = config.section("training_v2")
    fingerprint = canonical_hash(
        {
            "candidate": candidate,
            "training": settings,
            "microsteps": microsteps,
            "train": sha256_file(final_dir / "train.jsonl"),
            "valid": sha256_file(final_dir / "valid.jsonl"),
            "base": config.sources["models"]["research_base_mlx_4bit"],
            "forge_training_version": 1,
        }
    )
    status_path = adapter_dir / "status.json"
    if status_path.exists() and not force:
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint and existing.get("complete"):
            return existing

    adapter_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.section("project")["seed"])
    mx.random.seed(seed)
    np.random.seed(seed)
    model_path = _research_snapshot(config)
    model, tokenizer = load(model_path, tokenizer_config={"trust_remote_code": True})
    model.freeze()
    lora_parameters = {
        "rank": int(candidate["rank"]),
        "scale": float(settings["scale"]),
        "dropout": float(settings["dropout"]),
        "keys": list(settings["targets"]),
    }
    linear_to_lora_layers(
        model,
        int(candidate["num_layers"]),
        lora_parameters,
        use_dora=False,
    )
    trainable_parameters = sum(
        parameter.size for _, parameter in tree_flatten(model.trainable_parameters())
    )
    write_json(
        adapter_dir / "adapter_config.json",
        _adapter_config(config, candidate, model_path, adapter_dir),
    )
    train_rows = list(read_jsonl(final_dir / "train.jsonl"))
    valid_rows = list(read_jsonl(final_dir / "valid.jsonl"))
    group_size = int(settings["grad_accumulation_steps"])
    train_dataset = ForgeDataset(
        train_rows,
        tokenizer,
        group_size=group_size,
        seed=seed,
        grouped=True,
    )
    valid_dataset = ForgeDataset(
        valid_rows,
        tokenizer,
        group_size=1,
        seed=seed,
        grouped=False,
    )
    microsteps = min(microsteps, len(train_dataset))
    training_args = TrainingArgs(
        batch_size=1,
        iters=microsteps,
        val_batches=-1,
        steps_per_report=group_size,
        steps_per_eval=min(int(settings["validation_every"]), microsteps),
        steps_per_save=min(int(settings["checkpoint_every"]), microsteps),
        max_seq_length=int(settings["max_seq_length"]),
        adapter_file=str(adapter_dir / "adapters.safetensors"),
        grad_checkpoint=True,
        grad_accumulation_steps=group_size,
        clear_cache_threshold=12 * 1024**3,
    )
    optimizer = _optimizer(candidate, settings, microsteps)
    started = time.monotonic()
    callback = _ForgeCallback(
        model=model,
        best_path=adapter_dir / "best_adapters.safetensors",
        deadline=started + max_seconds,
        patience=early_stop_patience,
    )
    mx.random.seed(seed)
    np.random.seed(seed)
    initial_loss = float(
        evaluate(
            model=model,
            dataset=valid_dataset,
            batch_size=1,
            num_batches=-1,
            max_seq_length=int(settings["max_seq_length"]),
            loss=forge_loss,
            iterate_batches=forge_iterate_batches,
            clear_cache_threshold=12 * 1024**3,
        )
    )
    callback.consider_validation(0, initial_loss)
    stopped = False
    try:
        train(
            model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=valid_dataset,
            args=training_args,
            loss=forge_loss,
            iterate_batches=forge_iterate_batches,
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
            max_seq_length=int(settings["max_seq_length"]),
            loss=forge_loss,
            iterate_batches=forge_iterate_batches,
            clear_cache_threshold=12 * 1024**3,
        )
    )
    callback.consider_validation(microsteps, final_loss)
    best_path = adapter_dir / "best_adapters.safetensors"
    active_path = adapter_dir / "adapters.safetensors"
    if best_path.exists():
        model.load_weights(str(best_path), strict=False)
        mx.save_safetensors(
            str(active_path), dict(tree_flatten(model.trainable_parameters()))
        )
    status = {
        "fingerprint": fingerprint,
        "complete": not (stopped and callback.stop_reason == "deadline"),
        "stopped": stopped,
        "stop_reason": callback.stop_reason,
        "candidate": candidate["name"],
        "adapter_path": str(adapter_dir),
        "adapter_sha256": sha256_file(active_path),
        "base_model_path": model_path,
        "microsteps": microsteps,
        "optimizer_updates": microsteps // group_size,
        "trainable_parameters": trainable_parameters,
        "initial_validation_loss": (
            initial_loss
        ),
        "final_validation_loss": final_loss,
        "best_validation_loss": callback.best_loss,
        "best_validation_iteration": callback.best_iteration,
        "validation_history": callback.validation_history,
        "train_history": callback.train_history,
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 4),
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    write_json(status_path, status)
    del model, tokenizer, train_dataset, valid_dataset, optimizer
    gc.collect()
    mx.clear_cache()
    return status


def run_forge_pilots(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    build_forge_data(config)
    from .forge_eval import pilot_task_ids, run_forge_evaluation

    settings = config.section("training_v2")
    artifact_dir = config.path_for("artifact_dir")
    candidates: list[dict[str, Any]] = []
    caffeinate = _start_caffeinate()
    try:
        pilot_budget = int(config.section("overnight_v2")["pilot_seconds"])
        eval_seconds = int(config.section("evaluation_v2")["pilot_max_seconds_per_variant"])
        per_candidate_seconds = max(
            300,
            (pilot_budget - eval_seconds * len(settings["candidates"]))
            // len(settings["candidates"]),
        )
        for candidate in settings["candidates"]:
            result = _train_candidate(
                config,
                candidate=candidate,
                adapter_dir=artifact_dir / "adapters" / "pilots" / candidate["name"],
                microsteps=int(settings["pilot_microsteps"]),
                max_seconds=per_candidate_seconds,
                early_stop_patience=None,
                force=force,
            )
            canary = run_forge_evaluation(
                config,
                variant="pilot",
                suite="dev",
                force=force,
                adapter_path_override=Path(result["adapter_path"]),
                task_ids=pilot_task_ids(config),
                report_label=candidate["name"],
                max_seconds_override=eval_seconds,
            )
            result["canary"] = canary
            write_json(Path(result["adapter_path"]) / "status.json", result)
            candidates.append(result)
    finally:
        if caffeinate is not None and caffeinate.poll() is None:
            caffeinate.terminate()

    successful = [
        candidate
        for candidate in candidates
        if candidate["complete"] and candidate["canary"]["coverage"] == 1.0
    ]
    if not successful:
        raise RuntimeError("No Forge pilot completed within the fixed budget")
    parameter_counts = {candidate["trainable_parameters"] for candidate in successful}
    if len(parameter_counts) != 1:
        raise RuntimeError(
            f"Forge candidates do not have equal parameter budgets: {parameter_counts}"
        )
    selected = min(
        successful,
        key=lambda result: (
            -float(result["canary"]["scores"]["overall"]["accuracy"]),
            float(result["final_validation_loss"]),
            float(result["elapsed_seconds"]),
            result["candidate"],
        ),
    )
    comparison = {
        "selection_policy": (
            "highest locked trilingual task accuracy, then validation loss, then elapsed time"
        ),
        "equal_trainable_parameters": parameter_counts.pop(),
        "candidates": candidates,
        "selected_candidate": selected["candidate"],
    }
    write_json(artifact_dir / "pilot-comparison.json", comparison)
    write_json(artifact_dir / "pilot-selected.json", selected)
    return comparison


def run_forge_training(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    artifact_dir = config.path_for("artifact_dir")
    pilot_path = artifact_dir / "pilot-selected.json"
    if not pilot_path.exists():
        run_forge_pilots(config)
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    settings = config.section("training_v2")
    candidate = next(
        item for item in settings["candidates"] if item["name"] == pilot["candidate"]
    )
    train_rows = list(read_jsonl(config.path_for("final_dir") / "train.jsonl"))
    microsteps = len(train_rows) * int(settings["full_epochs"])
    caffeinate = _start_caffeinate()
    try:
        result = _train_candidate(
            config,
            candidate=candidate,
            adapter_dir=artifact_dir / "adapters" / f"final-{candidate['name']}",
            microsteps=microsteps,
            max_seconds=int(settings["max_seconds"]),
            early_stop_patience=int(settings["early_stop_evaluations"]),
            force=force,
        )
    finally:
        if caffeinate is not None and caffeinate.poll() is None:
            caffeinate.terminate()
    selected = {**result, "recipe_frozen": False}
    write_json(artifact_dir / "selected.json", selected)
    return selected

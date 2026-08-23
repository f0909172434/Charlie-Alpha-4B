from __future__ import annotations

import json
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download
from rich.console import Console

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json

console = Console()

_VAL_RE = re.compile(r"Iter (\d+): Val loss ([0-9.]+)")
_SAVE_RE = re.compile(r"Iter (\d+): Saved adapter weights")
_TRAIN_RE = re.compile(r"Iter (\d+): Train loss ([0-9.]+).+Peak mem ([0-9.]+) GB")


@dataclass
class RunResult:
    returncode: int
    timed_out: bool
    early_stopped: bool
    last_checkpoint_iteration: int
    last_train_iteration: int
    best_validation_iteration: int | None
    best_validation_loss: float | None
    peak_memory_gb: float
    log_path: Path


def _base_snapshot(config: ProjectConfig) -> str:
    source = config.sources["models"]["base_mlx_4bit"]
    console.print("Downloading/verifying the pinned 4-bit base model…")
    return snapshot_download(repo_id=source["repo_id"], revision=source["revision"])


def _count_rows(path: Path) -> int:
    return sum(1 for _ in read_jsonl(path))


def _training_fingerprint(
    config: ProjectConfig,
    candidate: dict[str, Any],
    sequence_length: int,
    num_layers: int,
) -> str:
    final_dir = config.path_for("final_dir")
    return canonical_hash(
        {
            "training": config.section("training"),
            "candidate": candidate,
            "sequence_length": sequence_length,
            "num_layers": num_layers,
            "base": config.sources["models"]["base_mlx_4bit"],
            "data": {
                split: sha256_file(final_dir / f"{split}.jsonl")
                for split in ("train", "valid", "test")
            },
            "version": "train-v1",
        }
    )


def _write_mlx_config(
    *,
    destination: Path,
    model_path: str,
    data_path: Path,
    adapter_path: Path,
    settings: dict[str, Any],
    candidate: dict[str, Any],
    sequence_length: int,
    num_layers: int,
    iterations: int,
    seed: int,
    resume_file: Path | None,
) -> None:
    warmup = max(1, round(iterations * float(settings["warmup_fraction"])))
    decay_steps = max(1, iterations - warmup)
    learning_rate = float(candidate["learning_rate"])
    values: dict[str, Any] = {
        "model": model_path,
        "train": True,
        "fine_tune_type": "lora",
        "optimizer": "adamw",
        "optimizer_config": {"adamw": {"weight_decay": 0.01}},
        "data": str(data_path),
        "seed": seed,
        "num_layers": num_layers,
        "batch_size": int(settings["batch_size"]),
        "iters": iterations,
        "val_batches": int(settings["val_batches"]),
        "learning_rate": learning_rate,
        "steps_per_report": 5,
        "steps_per_eval": min(int(settings["eval_every"]), max(10, iterations)),
        "grad_accumulation_steps": int(settings["grad_accumulation_steps"]),
        "adapter_path": str(adapter_path),
        "save_every": min(int(settings["checkpoint_every"]), max(10, iterations)),
        "max_seq_length": sequence_length,
        "grad_checkpoint": True,
        "mask_prompt": True,
        "clear_cache_threshold": 12 * 1024**3,
        "lora_parameters": {
            "rank": int(candidate["rank"]),
            "scale": float(candidate["scale"]),
            "dropout": float(candidate["dropout"]),
            "keys": candidate["keys"],
        },
        "lr_schedule": {
            "name": "cosine_decay",
            "arguments": [learning_rate, decay_steps, learning_rate * 0.1],
            "warmup": warmup,
            "warmup_init": 0.0,
        },
    }
    if resume_file is not None:
        values["resume_adapter_file"] = str(resume_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


def _terminate_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _run_mlx(
    mlx_config: Path,
    log_path: Path,
    max_seconds: int,
    early_stop_patience: int | None,
) -> RunResult:
    command = [
        "/usr/bin/caffeinate",
        "-dimsu",
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--config",
        str(mlx_config),
    ]
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
        start_new_session=True,
    )
    assert process.stdout is not None
    output_queue: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        for output_line in process.stdout:
            output_queue.put(output_line)
        output_queue.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    started = time.monotonic()
    validation_best: float | None = None
    validation_best_iteration: int | None = None
    stale_validations = 0
    last_checkpoint = 0
    last_train = 0
    peak_memory = 0.0
    timed_out = False
    early_stopped = False
    stream_finished = False
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log:
        while process.poll() is None or not stream_finished:
            if process.poll() is None and time.monotonic() - started >= max_seconds:
                timed_out = True
                log.write(f"[charlie-alpha] hard timeout after {max_seconds}s\n")
                log.flush()
                _terminate_group(process)
            try:
                line = output_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if line is None:
                stream_finished = True
                continue
            log.write(line)
            log.flush()
            console.print(line.rstrip())
            if match := _SAVE_RE.search(line):
                last_checkpoint = max(last_checkpoint, int(match.group(1)))
            if match := _TRAIN_RE.search(line):
                last_train = max(last_train, int(match.group(1)))
                peak_memory = max(peak_memory, float(match.group(3)))
            if match := _VAL_RE.search(line):
                iteration = int(match.group(1))
                loss = float(match.group(2))
                if validation_best is None or loss < validation_best - 1e-4:
                    validation_best = loss
                    validation_best_iteration = iteration
                    stale_validations = 0
                elif iteration > 1:
                    stale_validations += 1
                if (
                    early_stop_patience is not None
                    and stale_validations >= early_stop_patience
                    and process.poll() is None
                ):
                    early_stopped = True
                    log.write("[charlie-alpha] early stopping triggered\n")
                    log.flush()
                    _terminate_group(process)

    return RunResult(
        returncode=process.wait(),
        timed_out=timed_out,
        early_stopped=early_stopped,
        last_checkpoint_iteration=last_checkpoint,
        last_train_iteration=last_train,
        best_validation_iteration=validation_best_iteration,
        best_validation_loss=validation_best,
        peak_memory_gb=peak_memory,
        log_path=log_path,
    )


def _is_oom(log_path: Path) -> bool:
    text = log_path.read_text(encoding="utf-8", errors="replace").lower()
    markers = ("out of memory", "metal gpu", "resource exhausted", "cannot allocate memory")
    return any(marker in text for marker in markers)


def _attempts(settings: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        (int(settings["max_seq_length"]), int(settings["num_layers"])),
        (int(settings["fallback_seq_length"]), int(settings["num_layers"])),
        (int(settings["fallback_seq_length"]), int(settings["fallback_num_layers"])),
    ]


def run_pilot(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    final_dir = config.path_for("final_dir")
    if not (final_dir / "train.jsonl").exists():
        raise RuntimeError("Final mixed data is missing; run `make mix` first.")
    settings = config.section("training")
    candidate = settings["candidates"][0]
    model_path = _base_snapshot(config)
    artifact_dir = config.path_for("artifact_dir")

    for sequence_length, num_layers in _attempts(settings):
        attempt_name = f"{candidate['name']}-s{sequence_length}-l{num_layers}"
        adapter_path = artifact_dir / "adapters" / attempt_name
        fingerprint = _training_fingerprint(
            config, candidate, sequence_length=sequence_length, num_layers=num_layers
        )
        status_path = adapter_path / "pilot-status.json"
        if status_path.exists() and not force:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("fingerprint") == fingerprint and status.get("success"):
                write_json(
                    artifact_dir / "selected.json",
                    {**status, "adapter_path": str(adapter_path), "model_path": model_path},
                )
                return status
        iterations = min(int(settings["pilot_iterations"]), _count_rows(final_dir / "train.jsonl"))
        mlx_config = adapter_path / "pilot.yaml"
        _write_mlx_config(
            destination=mlx_config,
            model_path=model_path,
            data_path=final_dir,
            adapter_path=adapter_path,
            settings=settings,
            candidate=candidate,
            sequence_length=sequence_length,
            num_layers=num_layers,
            iterations=iterations,
            seed=int(settings["seed"]),
            resume_file=None,
        )
        result = _run_mlx(
            mlx_config,
            adapter_path / "pilot.log",
            max_seconds=int(config.section("overnight")["pilot_seconds"]),
            early_stop_patience=None,
        )
        adapter_file = adapter_path / "adapters.safetensors"
        success = (
            result.returncode == 0
            and not result.timed_out
            and adapter_file.exists()
            and result.best_validation_loss is not None
            and math.isfinite(result.best_validation_loss)
        )
        status = {
            "fingerprint": fingerprint,
            "success": success,
            "candidate": candidate["name"],
            "sequence_length": sequence_length,
            "num_layers": num_layers,
            "completed_iterations": result.last_train_iteration if success else 0,
            "best_validation_loss": result.best_validation_loss,
            "peak_memory_gb": result.peak_memory_gb,
            "adapter_sha256": sha256_file(adapter_file) if adapter_file.exists() else None,
        }
        write_json(status_path, status)
        if success:
            write_json(
                artifact_dir / "selected.json",
                {**status, "adapter_path": str(adapter_path), "model_path": model_path},
            )
            return status
        if not _is_oom(result.log_path):
            raise RuntimeError(f"Pilot failed for a non-OOM reason; inspect {result.log_path}")
        console.print(
            f"[yellow]OOM at {sequence_length} tokens/{num_layers} layers; "
            "trying fallback.[/yellow]"
        )
    raise RuntimeError("All fixed Metal OOM fallbacks failed.")


def run_training(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    artifact_dir = config.path_for("artifact_dir")
    selected_path = artifact_dir / "selected.json"
    if not selected_path.exists():
        run_pilot(config)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    adapter_path = Path(selected["adapter_path"])
    status_path = adapter_path / "training-status.json"
    fingerprint = selected["fingerprint"]
    if status_path.exists() and not force:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("fingerprint") == fingerprint and status.get("complete"):
            console.print(
                "[cyan]Training already completed for this data/config fingerprint.[/cyan]"
            )
            return status

    settings = config.section("training")
    candidate = next(
        item for item in settings["candidates"] if item["name"] == selected["candidate"]
    )
    final_dir = config.path_for("final_dir")
    total_iterations = _count_rows(final_dir / "train.jsonl") * int(settings["max_epochs"])
    prior_iterations = int(selected.get("completed_iterations", 0))
    if status_path.exists() and not force:
        prior_status = json.loads(status_path.read_text(encoding="utf-8"))
        if prior_status.get("fingerprint") == fingerprint:
            prior_iterations = max(
                prior_iterations, int(prior_status.get("completed_iterations", 0))
            )
    remaining = max(0, total_iterations - prior_iterations)
    if remaining == 0:
        summary = {
            **selected,
            "fingerprint": fingerprint,
            "complete": True,
            "completed_iterations": total_iterations,
            "total_iterations": total_iterations,
        }
        write_json(status_path, summary)
        return summary

    resume_file = adapter_path / "adapters.safetensors"
    if not resume_file.exists():
        raise RuntimeError(f"Pilot adapter is missing: {resume_file}")
    mlx_config = adapter_path / "train.yaml"
    _write_mlx_config(
        destination=mlx_config,
        model_path=selected["model_path"],
        data_path=final_dir,
        adapter_path=adapter_path,
        settings=settings,
        candidate=candidate,
        sequence_length=int(selected["sequence_length"]),
        num_layers=int(selected["num_layers"]),
        iterations=remaining,
        seed=int(settings["seed"]) + 1 + prior_iterations,
        resume_file=resume_file,
    )
    result = _run_mlx(
        mlx_config,
        adapter_path / "train.log",
        max_seconds=int(settings["max_seconds"]),
        early_stop_patience=int(settings["early_stop_evaluations"]),
    )
    completed_this_run = (
        result.last_checkpoint_iteration
        if result.timed_out or result.early_stopped or result.returncode != 0
        else remaining
    )
    completed_iterations = min(total_iterations, prior_iterations + completed_this_run)
    if result.returncode != 0 and not (result.timed_out or result.early_stopped):
        if _is_oom(result.log_path):
            raise RuntimeError(
                "The selected pilot configuration OOMed during the long run; rerun pilot with "
                f"a lower fallback after inspecting {result.log_path}."
            )
        raise RuntimeError(f"Training failed; inspect {result.log_path}")

    active_adapter = adapter_path / "adapters.safetensors"
    if active_adapter.exists():
        shutil.copy2(active_adapter, adapter_path / "last_adapters.safetensors")
    best_iteration = result.best_validation_iteration
    if best_iteration is not None:
        best_checkpoint = adapter_path / f"{best_iteration:07d}_adapters.safetensors"
        if best_checkpoint.exists():
            shutil.copy2(best_checkpoint, active_adapter)
            shutil.copy2(best_checkpoint, adapter_path / "best_adapters.safetensors")

    complete = completed_iterations >= total_iterations or result.early_stopped
    summary = {
        **selected,
        "fingerprint": fingerprint,
        "complete": complete,
        "early_stopped": result.early_stopped,
        "timed_out": result.timed_out,
        "completed_iterations": completed_iterations,
        "total_iterations": total_iterations,
        "best_validation_iteration": result.best_validation_iteration,
        "best_validation_loss": result.best_validation_loss,
        "peak_memory_gb": result.peak_memory_gb,
        "adapter_sha256": sha256_file(active_adapter) if active_adapter.exists() else None,
    }
    write_json(status_path, summary)
    return summary

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
from math_verify import parse, verify
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from rich.console import Console

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl
from .validators import has_target_script

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
    final_validation_loss: float | None
    validation_history: list[dict[str, float | int]]
    peak_memory_gb: float
    log_path: Path


def _base_snapshot(config: ProjectConfig) -> str:
    source = config.sources["models"]["base_mlx_4bit"]
    console.print("Downloading/verifying the pinned 4-bit base model…")
    return snapshot_download(repo_id=source["repo_id"], revision=source["revision"])


def _count_rows(path: Path) -> int:
    return sum(1 for _ in read_jsonl(path))


def _training_data_for_sequence_length(config: ProjectConfig, sequence_length: int) -> Path:
    """Use only complete examples when an OOM fallback lowers the token limit."""
    final_dir = config.path_for("final_dir")
    maximum = int(config.section("training")["max_seq_length"])
    if sequence_length >= maximum:
        return final_dir

    output = config.path_for("artifact_dir") / "training-data" / f"max-{sequence_length}"
    manifest_path = output / "manifest.json"
    fingerprint = canonical_hash(
        {
            "sequence_length": sequence_length,
            "source": {
                split: sha256_file(final_dir / f"{split}.jsonl")
                for split in ("train", "valid", "test")
            },
        }
    )
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == fingerprint and all(
            (output / f"{split}.jsonl").exists() for split in ("train", "valid", "test")
        ):
            return output

    counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        rows = [
            row
            for row in read_jsonl(final_dir / f"{split}.jsonl")
            if int(row["metadata"]["token_count"]) <= sequence_length
        ]
        if not rows:
            raise RuntimeError(
                f"No complete {split} records fit the {sequence_length}-token fallback."
            )
        write_jsonl(output / f"{split}.jsonl", rows)
        counts[split] = len(rows)
    write_json(
        manifest_path,
        {
            "fingerprint": fingerprint,
            "sequence_length": sequence_length,
            "counts": counts,
            "policy": "complete-records-only-no-truncation",
        },
    )
    return output


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
    optimizer_steps = math.ceil(iterations / int(settings["grad_accumulation_steps"]))
    warmup = max(1, round(optimizer_steps * float(settings["warmup_fraction"])))
    decay_steps = max(1, optimizer_steps - warmup)
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
    validation_history: list[dict[str, float | int]] = []
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
                validation_history.append({"iteration": iteration, "loss": loss})
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
        final_validation_loss=(
            float(validation_history[-1]["loss"]) if validation_history else None
        ),
        validation_history=validation_history,
        peak_memory_gb=peak_memory,
        log_path=log_path,
    )


def _is_oom(log_path: Path) -> bool:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    text = text.rsplit("Starting training...", maxsplit=1)[-1].lower()
    markers = (
        "out of memory",
        "insufficient memory",
        "metal gpu",
        "resource exhausted",
        "cannot allocate memory",
    )
    return any(marker in text for marker in markers)


def _log_snapshot(text: str) -> dict[str, Any]:
    validations = [
        {"iteration": int(match.group(1)), "loss": float(match.group(2))}
        for match in _VAL_RE.finditer(text)
    ]
    checkpoints = [int(match.group(1)) for match in _SAVE_RE.finditer(text)]
    training = list(_TRAIN_RE.finditer(text))
    best = min(validations, key=lambda item: float(item["loss"])) if validations else None
    return {
        "last_checkpoint_iteration": max(checkpoints, default=0),
        "last_train_iteration": max((int(match.group(1)) for match in training), default=0),
        "best_validation_iteration": int(best["iteration"]) if best else None,
        "best_validation_loss": float(best["loss"]) if best else None,
        "final_validation_loss": float(validations[-1]["loss"]) if validations else None,
        "validation_history": validations,
        "peak_memory_gb": max((float(match.group(3)) for match in training), default=0.0),
    }


def _recoverable_oom_snapshot(log_path: Path) -> dict[str, Any] | None:
    """Find the newest OOM segment with a checkpoint, even after a later interrupted restart."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    markers = (
        "out of memory",
        "insufficient memory",
        "metal gpu",
        "resource exhausted",
        "cannot allocate memory",
    )
    for segment in reversed(text.split("Starting training...")[1:]):
        lowered = segment.lower()
        if any(marker in lowered for marker in markers):
            snapshot = _log_snapshot(segment)
            if int(snapshot["last_checkpoint_iteration"]) > 0:
                return snapshot
    return None


def _attempts(settings: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        (int(settings["max_seq_length"]), int(settings["num_layers"])),
        (int(settings["fallback_seq_length"]), int(settings["num_layers"])),
        (int(settings["fallback_seq_length"]), int(settings["fallback_num_layers"])),
    ]


def _pilot_canary_score(
    config: ProjectConfig, model_path: str, adapter_path: Path
) -> dict[str, Any]:
    candidates = [
        row
        for row in read_jsonl(config.path_for("final_dir") / "test.jsonl")
        if row["metadata"]["domain"] == "math" and row["metadata"].get("answer")
    ]
    selected: list[dict[str, Any]] = []
    for language in ("en", "zh_Hant", "zh_Hans"):
        language_rows = [row for row in candidates if row["metadata"]["language"] == language]
        if language_rows:
            selected.append(min(language_rows, key=lambda row: row["metadata"]["token_count"]))

    model, tokenizer = load(
        model_path,
        adapter_path=str(adapter_path),
        tokenizer_config={"trust_remote_code": True},
    )
    results: list[dict[str, Any]] = []
    for row in selected:
        language = row["metadata"]["language"]
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "Solve accurately and put the concise final answer at the end.",
                },
                row["messages"][0],
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        output = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=384,
            sampler=make_sampler(temp=0.0),
            verbose=False,
        )
        try:
            correct = bool(verify(parse(row["metadata"]["answer"]), parse(output), strict=False))
        except (TimeoutError, TypeError, ValueError):
            correct = False
        language_ok = (
            has_target_script(output, language) if language in {"zh_Hant", "zh_Hans"} else True
        )
        results.append(
            {
                "language": language,
                "correct": correct,
                "language_ok": language_ok,
                "passed": correct and language_ok,
            }
        )
    return {
        "score": sum(float(result["passed"]) for result in results),
        "maximum": len(results),
        "results": results,
    }


def run_pilot(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    final_dir = config.path_for("final_dir")
    if not (final_dir / "train.jsonl").exists():
        raise RuntimeError("Final mixed data is missing; run `make mix` first.")
    settings = config.section("training")
    model_path = _base_snapshot(config)
    artifact_dir = config.path_for("artifact_dir")
    successes: list[dict[str, Any]] = []

    for candidate in settings["candidates"]:
        candidate_succeeded = False
        for sequence_length, num_layers in _attempts(settings):
            attempt_name = f"{candidate['name']}-s{sequence_length}-l{num_layers}"
            adapter_path = artifact_dir / "adapters" / attempt_name
            fingerprint = _training_fingerprint(
                config, candidate, sequence_length=sequence_length, num_layers=num_layers
            )
            status_path = adapter_path / "pilot-status.json"
            status: dict[str, Any] | None = None
            if status_path.exists() and not force:
                existing = json.loads(status_path.read_text(encoding="utf-8"))
                pilot_config_path = adapter_path / "pilot.yaml"
                existing_seed = None
                if pilot_config_path.exists():
                    existing_config = yaml.safe_load(pilot_config_path.read_text(encoding="utf-8"))
                    existing_seed = existing_config.get("seed")
                if (
                    existing.get("fingerprint") == fingerprint
                    and existing.get("success")
                    and existing_seed == int(settings["seed"])
                ):
                    status = existing
                    status["seed"] = existing_seed

            if status is None:
                iterations = min(
                    int(settings["pilot_iterations"]),
                    _count_rows(final_dir / "train.jsonl"),
                )
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
                    and result.final_validation_loss is not None
                    and math.isfinite(result.final_validation_loss)
                )
                status = {
                    "fingerprint": fingerprint,
                    "success": success,
                    "candidate": candidate["name"],
                    "seed": int(settings["seed"]),
                    "sequence_length": sequence_length,
                    "num_layers": num_layers,
                    "completed_iterations": result.last_train_iteration if success else 0,
                    "initial_validation_loss": (
                        result.validation_history[0]["loss"] if result.validation_history else None
                    ),
                    "final_validation_loss": result.final_validation_loss,
                    "best_validation_loss": result.best_validation_loss,
                    "validation_history": result.validation_history,
                    "peak_memory_gb": result.peak_memory_gb,
                    "adapter_sha256": (
                        sha256_file(adapter_file) if adapter_file.exists() else None
                    ),
                }
                write_json(status_path, status)
                if not success:
                    if not _is_oom(result.log_path):
                        raise RuntimeError(
                            f"Pilot failed for a non-OOM reason; inspect {result.log_path}"
                        )
                    console.print(
                        f"[yellow]OOM at {sequence_length} tokens/{num_layers} layers; "
                        "trying fallback.[/yellow]"
                    )
                    continue

            if "canary" not in status:
                status["canary"] = _pilot_canary_score(config, model_path, adapter_path)
                write_json(status_path, status)
            successes.append(
                {**status, "adapter_path": str(adapter_path), "model_path": model_path}
            )
            candidate_succeeded = True
            break
        if not candidate_succeeded:
            console.print(f"[yellow]No viable fallback for {candidate['name']}.[/yellow]")

    if not successes:
        raise RuntimeError("All fixed Metal OOM fallbacks failed.")
    selected = min(
        successes,
        key=lambda status: (
            -float(status["canary"]["score"]),
            float(status["final_validation_loss"]),
        ),
    )
    write_json(artifact_dir / "pilot-comparison.json", {"candidates": successes})
    write_json(artifact_dir / "selected.json", selected)
    return selected


def _restore_selected_adapter_metadata(adapter_path: Path, selected: dict[str, Any]) -> None:
    """Keep MLX load metadata aligned with the checkpoint selected before OOM fallbacks."""
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        return
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    adapter_config.update(
        {
            "num_layers": int(selected["num_layers"]),
            "max_seq_length": int(selected["sequence_length"]),
            "data": str(adapter_path.parents[2] / "data" / "final"),
            "config": str(adapter_path / "train.yaml"),
            "resume_adapter_file": None,
        }
    )
    write_json(config_path, adapter_config)


def run_training(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    artifact_dir = config.path_for("artifact_dir")
    selected_path = artifact_dir / "selected.json"
    if not selected_path.exists():
        run_pilot(config)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    adapter_path = Path(selected["adapter_path"])
    status_path = adapter_path / "training-status.json"
    fingerprint = selected["fingerprint"]
    status: dict[str, Any] | None = None
    if status_path.exists() and not force:
        existing_status = json.loads(status_path.read_text(encoding="utf-8"))
        if existing_status.get("fingerprint") == fingerprint:
            status = existing_status
        if status and status.get("complete"):
            _restore_selected_adapter_metadata(adapter_path, selected)
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
    runtime_sequence_length = int(selected["sequence_length"])
    runtime_num_layers = int(selected["num_layers"])
    if status:
        prior_iterations = max(prior_iterations, int(status.get("completed_iterations", 0)))
        runtime_sequence_length = int(
            status.get("runtime_sequence_length", runtime_sequence_length)
        )
        runtime_num_layers = int(status.get("runtime_num_layers", runtime_num_layers))

    # A machine or Metal interruption can happen after MLX has written a valid checkpoint but
    # before this wrapper writes training-status.json. Recover it deterministically and move to
    # the next fixed OOM fallback instead of repeating completed work.
    train_log = adapter_path / "train.log"
    active_adapter = adapter_path / "adapters.safetensors"
    best_adapter = adapter_path / "best_adapters.safetensors"
    attempts = _attempts(settings)
    if (
        status is not None
        and (runtime_sequence_length, runtime_num_layers) == attempts[-1]
        and train_log.exists()
        and _is_oom(train_log)
        and best_adapter.exists()
        and status.get("best_validation_loss") is not None
        and math.isfinite(float(status["best_validation_loss"]))
    ):
        shutil.copy2(best_adapter, active_adapter)
        _restore_selected_adapter_metadata(adapter_path, selected)
        status.update(
            {
                "complete": True,
                "resource_stopped": True,
                "stopped_reason": "fixed Metal OOM fallbacks exhausted; restored best checkpoint",
                "adapter_sha256": sha256_file(active_adapter),
            }
        )
        write_json(status_path, status)
        console.print(
            "[yellow]Fixed OOM fallbacks exhausted; restored the best validated "
            "checkpoint.[/yellow]"
        )
        return status
    oom_snapshot = (
        _recoverable_oom_snapshot(train_log)
        if status is None and train_log.exists() and active_adapter.exists()
        else None
    )
    if oom_snapshot is not None:
        run_starts = sorted(adapter_path.glob("*_run_start_adapters.safetensors"))
        snapshot = oom_snapshot
        checkpoint_iteration = int(snapshot["last_checkpoint_iteration"])
        if run_starts and checkpoint_iteration > 0:
            run_start_iteration = max(
                int(path.name.split("_", maxsplit=1)[0]) for path in run_starts
            )
            prior_iterations = min(total_iterations, run_start_iteration + checkpoint_iteration)
            history = [
                {
                    "iteration": run_start_iteration + int(item["iteration"]),
                    "loss": float(item["loss"]),
                }
                for item in snapshot["validation_history"]
            ]
            best_local_iteration = snapshot["best_validation_iteration"]
            if best_local_iteration is not None:
                best_checkpoint = adapter_path / (
                    f"{int(best_local_iteration):07d}_adapters.safetensors"
                )
                if best_checkpoint.exists():
                    shutil.copy2(best_checkpoint, adapter_path / "best_adapters.safetensors")
            current_attempt = (runtime_sequence_length, runtime_num_layers)
            try:
                next_attempt = attempts[attempts.index(current_attempt) + 1]
            except (ValueError, IndexError):
                next_attempt = None
            if next_attempt is None:
                raise RuntimeError(
                    "Training exhausted the fixed Metal OOM fallbacks; the last checkpoint "
                    f"remains at {active_adapter}."
                )
            runtime_sequence_length, runtime_num_layers = next_attempt
            status = {
                **selected,
                "fingerprint": fingerprint,
                "complete": False,
                "early_stopped": False,
                "timed_out": False,
                "recovered_from_oom": True,
                "completed_iterations": prior_iterations,
                "total_iterations": total_iterations,
                "best_validation_iteration": (
                    run_start_iteration + int(best_local_iteration)
                    if best_local_iteration is not None
                    else None
                ),
                "best_validation_loss": snapshot["best_validation_loss"],
                "final_validation_loss": snapshot["final_validation_loss"],
                "validation_history": history,
                "peak_memory_gb": snapshot["peak_memory_gb"],
                "runtime_sequence_length": runtime_sequence_length,
                "runtime_num_layers": runtime_num_layers,
                "adapter_sha256": sha256_file(active_adapter),
            }
            write_json(status_path, status)
            console.print(
                f"[yellow]Recovered iteration {prior_iterations}; continuing at "
                f"{runtime_sequence_length} tokens/{runtime_num_layers} layers.[/yellow]"
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

    resume_file = active_adapter
    if not resume_file.exists():
        raise RuntimeError(f"Pilot adapter is missing: {resume_file}")
    pilot_adapter = adapter_path / "pilot_adapters.safetensors"
    if not pilot_adapter.exists() or (
        status is None and prior_iterations == int(selected.get("completed_iterations", 0))
    ):
        shutil.copy2(resume_file, pilot_adapter)
    run_start_adapter = adapter_path / f"{prior_iterations:07d}_run_start_adapters.safetensors"
    shutil.copy2(resume_file, run_start_adapter)
    training_data_dir = _training_data_for_sequence_length(config, runtime_sequence_length)
    mlx_config = adapter_path / "train.yaml"
    _write_mlx_config(
        destination=mlx_config,
        model_path=selected["model_path"],
        data_path=training_data_dir,
        adapter_path=adapter_path,
        settings=settings,
        candidate=candidate,
        sequence_length=runtime_sequence_length,
        num_layers=runtime_num_layers,
        iterations=remaining,
        seed=int(settings["seed"]),
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

    prior_history = list((status or selected).get("validation_history", []))
    current_history = [
        {
            "iteration": prior_iterations + int(item["iteration"]),
            "loss": float(item["loss"]),
        }
        for item in result.validation_history
    ]
    validation_history = [*prior_history, *current_history]
    prior_best_loss = (status or selected).get("best_validation_loss")
    prior_best_iteration = (status or selected).get("best_validation_iteration")
    if prior_best_iteration is None and prior_best_loss is not None:
        prior_best_iteration = int(selected.get("completed_iterations", 0))
    global_best_loss = float(prior_best_loss) if prior_best_loss is not None else None
    global_best_iteration = int(prior_best_iteration) if prior_best_iteration is not None else None
    if not best_adapter.exists():
        shutil.copy2(run_start_adapter, best_adapter)
    if result.best_validation_loss is not None and (
        global_best_loss is None or result.best_validation_loss < global_best_loss - 1e-4
    ):
        local_best_iteration = result.best_validation_iteration
        best_checkpoint = (
            run_start_adapter
            if local_best_iteration == 1
            else adapter_path / f"{int(local_best_iteration or 0):07d}_adapters.safetensors"
        )
        if best_checkpoint.exists():
            shutil.copy2(best_checkpoint, best_adapter)
            global_best_loss = result.best_validation_loss
            global_best_iteration = prior_iterations + int(local_best_iteration or 0)

    current_attempt = (runtime_sequence_length, runtime_num_layers)
    next_attempt: tuple[int, int] | None = None
    if current_attempt in attempts:
        current_attempt_index = attempts.index(current_attempt)
        if current_attempt_index + 1 < len(attempts):
            next_attempt = attempts[current_attempt_index + 1]

    partial_summary = {
        **selected,
        "fingerprint": fingerprint,
        "complete": False,
        "early_stopped": result.early_stopped,
        "timed_out": result.timed_out,
        "completed_iterations": completed_iterations,
        "total_iterations": total_iterations,
        "best_validation_iteration": global_best_iteration,
        "best_validation_loss": global_best_loss,
        "final_validation_loss": result.final_validation_loss,
        "validation_history": validation_history,
        "peak_memory_gb": max(
            float((status or {}).get("peak_memory_gb", 0.0)), result.peak_memory_gb
        ),
        "runtime_sequence_length": runtime_sequence_length,
        "runtime_num_layers": runtime_num_layers,
        "adapter_sha256": sha256_file(active_adapter) if active_adapter.exists() else None,
    }
    if result.returncode != 0 and not (result.timed_out or result.early_stopped):
        if _is_oom(result.log_path):
            if next_attempt is None or completed_this_run == 0:
                write_json(status_path, partial_summary)
                raise RuntimeError(
                    "Training exhausted the fixed Metal OOM fallbacks; the last valid "
                    f"checkpoint remains at {active_adapter}."
                )
            partial_summary["recovered_from_oom"] = True
            partial_summary["runtime_sequence_length"] = next_attempt[0]
            partial_summary["runtime_num_layers"] = next_attempt[1]
            write_json(status_path, partial_summary)
            console.print(
                f"[yellow]Metal OOM after checkpoint {completed_iterations}; continuing at "
                f"{next_attempt[0]} tokens/{next_attempt[1]} layers.[/yellow]"
            )
            return run_training(config)
        write_json(status_path, partial_summary)
        raise RuntimeError(f"Training failed; inspect {result.log_path}")

    if active_adapter.exists():
        shutil.copy2(active_adapter, adapter_path / "last_adapters.safetensors")

    complete = completed_iterations >= total_iterations or result.early_stopped
    if complete and best_adapter.exists():
        shutil.copy2(best_adapter, active_adapter)
        _restore_selected_adapter_metadata(adapter_path, selected)
    summary = {**partial_summary, "complete": complete}
    summary["adapter_sha256"] = sha256_file(active_adapter) if active_adapter.exists() else None
    write_json(status_path, summary)
    return summary

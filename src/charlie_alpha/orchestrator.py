from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Any

from rich.console import Console

from .config import ProjectConfig
from .io_utils import write_json

console = Console()


def _run_step(config: ProjectConfig, arguments: list[str], timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    command = [
        "/usr/bin/caffeinate",
        "-dimsu",
        sys.executable,
        "-m",
        "charlie_alpha.cli",
        *arguments,
        "--config",
        str(config.path),
    ]
    process = subprocess.Popen(command, cwd=config.root, start_new_session=True)
    try:
        returncode = process.wait(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        returncode = process.returncode
    return {
        "command": " ".join(arguments),
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def run_overnight(config: ProjectConfig) -> dict[str, Any]:
    settings = config.section("overnight")
    total_seconds = int(settings["total_seconds"])
    started = time.monotonic()
    status_path = config.path_for("report_dir") / "overnight-status.json"
    steps = [
        (["data", "prepare"], int(settings["data_seconds"])),
        (["data", "distill"], int(settings["distillation_seconds"]) + 120),
        (["data", "mix"], 300),
        (["train", "pilot"], int(settings["pilot_seconds"]) + 120),
        (["train", "run"], int(settings["training_seconds"]) + 120),
        (["eval", "run", "--variant", "base"], 2820),
        (["eval", "run", "--variant", "adapter"], 2820),
        (["export", "all"], 1800),
        (["export", "validate-clean"], 1500),
        (["release", "check"], 120),
    ]
    results: list[dict[str, Any]] = []
    for arguments, step_budget in steps:
        remaining = total_seconds - int(time.monotonic() - started)
        if remaining <= 30:
            results.append({"command": " ".join(arguments), "skipped": "overall hard cap"})
            break
        console.print(f"[bold cyan]Overnight step:[/bold cyan] {' '.join(arguments)}")
        result = _run_step(config, arguments, timeout=min(step_budget, remaining))
        results.append(result)
        write_json(
            status_path,
            {
                "complete": False,
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "steps": results,
            },
        )
        if result["returncode"] != 0:
            console.print(f"[red]Step failed:[/red] {result['command']}")
            break
    summary = {
        "complete": len(results) == len(steps)
        and all(item.get("returncode") == 0 for item in results),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "hard_cap_seconds": total_seconds,
        "steps": results,
    }
    write_json(status_path, summary)
    return summary

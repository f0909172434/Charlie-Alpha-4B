from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from rich.console import Console

from .config import ProjectConfig
from .io_utils import write_json
from .stats_data import (
    build_stats_data,
    distill_stats_explanations,
    prepare_stats_blueprints,
    simulate_stats_surface,
)
from .stats_eval import (
    build_stats_evaluation_lock,
    compare_stats_evaluation,
    freeze_stats_recipe,
    run_stats_evaluation,
)
from .stats_release import check_stats_release, export_stats
from .stats_training import (
    calibrate_stats_adapter,
    run_stats_pilots,
    run_stats_training,
)

console = Console()


def run_stats_pipeline(config: ProjectConfig) -> dict[str, Any]:
    """Run the restart-safe v0.3 pipeline within its declared wall-clock budget."""
    started = time.monotonic()
    deadline = started + int(config.section("stats_budget")["total_seconds"])
    status_path = config.path_for("artifact_dir") / "pipeline-status.json"
    results: dict[str, Any] = {}

    def stage(
        name: str,
        function: Callable[[], Any],
        *,
        reserve_seconds: int = 0,
    ) -> Any:
        remaining = int(deadline - time.monotonic())
        if remaining <= reserve_seconds:
            raise RuntimeError(
                f"Stats pipeline stopped before {name}: {remaining}s remain, while "
                f"{reserve_seconds}s are reserved for later stages. Rerun to resume."
            )
        console.rule(f"Charlie alpha stats · {name}")
        stage_started = time.monotonic()
        value = function()
        results[name] = {
            "elapsed_seconds": round(time.monotonic() - stage_started, 2),
            "result": value,
        }
        results["elapsed_seconds"] = round(time.monotonic() - started, 2)
        write_json(status_path, results)
        return value

    stage("blueprints", lambda: prepare_stats_blueprints(config))
    stage("simulate", lambda: simulate_stats_surface(config))
    stage("lock_eval", lambda: build_stats_evaluation_lock(config))
    stage("distill", lambda: distill_stats_explanations(config), reserve_seconds=34_200)
    stage("data", lambda: build_stats_data(config), reserve_seconds=32_400)
    stage("pilots", lambda: run_stats_pilots(config), reserve_seconds=25_200)
    training = stage("train", lambda: run_stats_training(config), reserve_seconds=21_600)
    if not training.get("complete"):
        raise RuntimeError("Stats training has no complete validated checkpoint; rerun to resume")
    stage("calibrate", lambda: calibrate_stats_adapter(config), reserve_seconds=18_000)
    stage("freeze", lambda: freeze_stats_recipe(config), reserve_seconds=16_200)
    variants = ["base", "hard-label", "dgp-regret"]
    selected = json.loads(
        (config.path_for("artifact_dir") / "selected.json").read_text(encoding="utf-8")
    )
    if selected.get("variant") != "dgp-regret":
        variants.append("selected")
    for variant in variants:
        stage(
            f"eval_{variant}",
            lambda variant=variant: run_stats_evaluation(config, variant=variant),
            reserve_seconds=5_400,
        )
    comparison = stage("compare", lambda: compare_stats_evaluation(config), reserve_seconds=4_500)
    stage("export", lambda: export_stats(config, include_gguf=False), reserve_seconds=1_800)
    release_gate = stage("release_check", lambda: check_stats_release(config))
    results["complete"] = True
    results["ability_gates_passed"] = comparison["ability_gates_passed"]
    results["release_classification"] = release_gate["classification"]
    results["elapsed_seconds"] = round(time.monotonic() - started, 2)
    write_json(status_path, results)
    return results

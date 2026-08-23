from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from rich.console import Console

from .config import ProjectConfig
from .forge_data import (
    build_forge_data,
    distill_forge_translations,
    prepare_forge_candidates,
    score_forge_candidates,
    select_forge_sources,
)
from .forge_eval import (
    build_evaluation_lock,
    compare_forge_evaluation,
    freeze_forge_recipe,
    run_forge_evaluation,
)
from .forge_training import (
    calibrate_forge_adapter,
    run_forge_pilots,
    run_forge_training,
)
from .io_utils import write_json

console = Console()


def run_forge_overnight(config: ProjectConfig) -> dict[str, Any]:
    started = time.monotonic()
    total_seconds = int(config.section("overnight_v2")["total_seconds"])
    deadline = started + total_seconds
    results: dict[str, Any] = {}
    evaluation_budget = int(config.section("overnight_v2")["evaluation_export_seconds"])
    dev_eval_seconds = max(600, round(evaluation_budget * 0.1875))
    final_eval_seconds = max(900, round(evaluation_budget * 0.3125))

    def stage(name: str, function: Callable[[], Any], *, minimum_remaining: int = 0) -> Any:
        remaining = int(deadline - time.monotonic())
        if remaining < minimum_remaining:
            raise RuntimeError(
                f"Forge stopped before {name}: only {remaining}s of the fixed overnight "
                "budget remain"
            )
        console.rule(f"Forge · {name}")
        stage_started = time.monotonic()
        value = function()
        results[name] = {
            "elapsed_seconds": round(time.monotonic() - stage_started, 2),
            "result": value,
        }
        write_json(config.path_for("artifact_dir") / "overnight-status.json", results)
        return value

    stage("lock_eval", lambda: build_evaluation_lock(config))
    stage("prepare", lambda: prepare_forge_candidates(config))
    scoring = stage("score", lambda: score_forge_candidates(config), minimum_remaining=35_400)
    if not scoring["complete"]:
        raise RuntimeError(
            "Forge scoring budget ended before every candidate was scored; rerun to resume"
        )
    stage("select", lambda: select_forge_sources(config))
    translation = stage(
        "distill", lambda: distill_forge_translations(config), minimum_remaining=35_400
    )
    if not translation["complete"]:
        raise RuntimeError(
            "Forge translation pool is incomplete; rerun to resume before training"
        )
    stage("build", lambda: build_forge_data(config))
    stage("pilots", lambda: run_forge_pilots(config), minimum_remaining=31_200)
    training = stage("train", lambda: run_forge_training(config), minimum_remaining=27_600)
    if not training["complete"]:
        raise RuntimeError("Forge full training stopped before a validated checkpoint was complete")
    stage("calibrate", lambda: calibrate_forge_adapter(config))

    stage(
        "dev_base",
        lambda: run_forge_evaluation(
            config,
            variant="qwen35-base",
            suite="dev",
            max_seconds_override=dev_eval_seconds,
        ),
        minimum_remaining=9_600,
    )
    stage(
        "dev_forge",
        lambda: run_forge_evaluation(
            config,
            variant="forge",
            suite="dev",
            max_seconds_override=dev_eval_seconds,
        ),
        minimum_remaining=7_200,
    )
    dev_comparison = stage("dev_compare", lambda: compare_forge_evaluation(config, "dev"))
    if not dev_comparison["gate_passed"]:
        raise RuntimeError(
            "Forge did not pass the development improvement gate; final evaluation stays sealed"
        )
    stage("freeze", lambda: freeze_forge_recipe(config))
    stage(
        "final_base",
        lambda: run_forge_evaluation(
            config,
            variant="qwen35-base",
            suite="final",
            max_seconds_override=final_eval_seconds,
        ),
        minimum_remaining=4_800,
    )
    stage(
        "final_forge",
        lambda: run_forge_evaluation(
            config,
            variant="forge",
            suite="final",
            max_seconds_override=final_eval_seconds,
        ),
        minimum_remaining=2_400,
    )
    comparison = stage("final_compare", lambda: compare_forge_evaluation(config, "final"))
    results["complete"] = True
    results["release_gate_passed"] = comparison["gate_passed"]
    results["elapsed_seconds"] = round(time.monotonic() - started, 2)
    write_json(config.path_for("artifact_dir") / "overnight-status.json", results)
    return results

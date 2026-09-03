from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from rich.console import Console

from .config import ProjectConfig, load_config
from .data_pipeline import prepare_data
from .distillation import distill_data
from .evaluation import run_evaluation
from .exporting import export_all, export_gguf, validate_clean_environment
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
from .forge_orchestrator import run_forge_overnight
from .forge_router import (
    build_router_confirmation_lock,
    compare_router_confirmation,
    freeze_router_recipe,
    run_router_confirmation,
)
from .forge_training import (
    calibrate_forge_adapter,
    run_forge_pilot_candidate,
    run_forge_pilots,
    run_forge_training,
)
from .mixer import mix_data
from .orchestrator import run_overnight
from .release import check_release, publish_github, publish_hugging_face
from .routed_inference import generate_routed, load_routed_model, verify_dynamic_router
from .stats_agent import StatsAgent, classify_stats_route, resolve_stats_runtime
from .stats_bakeoff import run_base_bakeoff
from .stats_calibrate import (
    parse_layer_scales,
    run_block_projection,
    run_block_projection_arm,
    run_delta_calibration,
    run_delta_calibration_arm,
)
from .stats_canonical_bottleneck import (
    prepare_canonical_bottleneck_contract,
    prepare_canonical_bottleneck_data,
    run_canonical_bottleneck_confirmation,
    run_canonical_bottleneck_pilot,
)
from .stats_catalog_distillation import (
    prepare_catalog_distillation_contract,
    prepare_catalog_distillation_data,
    run_catalog_distillation_confirmation,
    run_catalog_distillation_pilot,
)
from .stats_catalog_fallback_external import (
    prepare_catalog_fallback_external_child_contract,
    prepare_catalog_fallback_external_data,
    prepare_catalog_fallback_external_master_contract,
    prepare_catalog_fallback_external_selected_data,
    prepare_catalog_fallback_external_source_availability,
    run_catalog_fallback_external_evaluation,
)
from .stats_catalog_grounding import (
    prepare_catalog_grounding_contract,
    prepare_catalog_grounding_data,
    run_catalog_grounding_confirmation,
    run_catalog_grounding_pilot,
)
from .stats_catalog_interface_replication import (
    prepare_catalog_interface_replication_contract,
    prepare_catalog_interface_replication_data,
    run_catalog_interface_replication,
)
from .stats_catalog_ranking import (
    prepare_catalog_ranking_contract,
    prepare_catalog_ranking_data,
    run_catalog_ranking_confirmation,
    run_catalog_ranking_pilot,
)
from .stats_cone import (
    confirm_uniform_family_candidate,
    promote_common_descent_candidate,
    run_common_descent_arm,
    run_common_descent_pilot,
)
from .stats_cross_format import (
    prepare_cross_format_contract,
    prepare_cross_format_data,
    run_cross_format_arm,
)
from .stats_cross_format_amendment import (
    run_cross_format_confirmation_amended,
    run_cross_format_pilot_amended,
)
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
from .stats_evolve import evolution_status, run_evolution
from .stats_experts import run_family_expert_arm, run_family_expert_oracle
from .stats_external_catalog import (
    prepare_external_catalog_contract,
    prepare_external_catalog_data,
    run_external_catalog_evaluation,
)
from .stats_external_catalog_v2 import (
    prepare_external_catalog_v2_contract,
    prepare_external_catalog_v2_data,
    run_external_catalog_v2_evaluation,
)
from .stats_external_domain_bridge import (
    prepare_external_domain_bridge_contract,
    prepare_external_domain_bridge_data,
    run_external_domain_bridge_training,
)
from .stats_external_domain_bridge_amendment import (
    prepare_external_domain_bridge_execution_amendment,
    run_external_domain_bridge_training_amended,
)
from .stats_external_exemplar_router import (
    prepare_external_exemplar_router_contract,
    run_external_exemplar_router_training,
)
from .stats_external_representation_diagnostic import run_external_representation_diagnostic
from .stats_external_weight_bridge import (
    prepare_external_weight_bridge_contract,
    prepare_external_weight_bridge_data,
    run_external_weight_bridge_training,
)
from .stats_external_weight_bridge_amendment import (
    prepare_external_weight_bridge_data_amended,
    prepare_external_weight_bridge_data_amendment,
    run_external_weight_bridge_training_amended,
)
from .stats_family_router import prepare_family_router, run_family_router
from .stats_guarded_external import run_guarded_external_evaluation
from .stats_guarded_external_metadata import (
    prepare_guarded_external_future_metadata_screen,
)
from .stats_guarded_external_source import (
    prepare_guarded_external_data,
    prepare_guarded_external_master_contract,
    prepare_guarded_external_source_screen,
)
from .stats_guarded_weight_bridge import (
    prepare_guarded_weight_bridge_contract,
    prepare_guarded_weight_bridge_data,
    run_guarded_weight_bridge_training,
)
from .stats_invalid_control_catalog_fallback import (
    prepare_invalid_control_catalog_fallback_contract,
    prepare_invalid_control_catalog_fallback_data,
    run_invalid_control_catalog_fallback,
)
from .stats_llm_router import (
    evaluate_llm_family_router_final,
    promote_llm_family_router,
    run_llm_family_router,
)
from .stats_opened_source_residual_repair import (
    prepare_opened_source_residual_contract,
    prepare_opened_source_residual_data,
    run_opened_source_residual_opportunity,
)
from .stats_orchestrator import run_stats_pipeline
from .stats_output_factorization import (
    prepare_output_factorization_contract,
    prepare_output_factorization_data,
    run_output_factorization_confirmation,
    run_output_factorization_pilot,
)
from .stats_project import (
    diagnose_policy_projection_gradients,
    prepare_policy_projection_data,
    run_policy_projection_arm,
    run_policy_projection_pilot,
)
from .stats_release import (
    check_stats_release,
    export_stats,
    publish_stats_github,
    publish_stats_hugging_face,
)
from .stats_representation_probe import (
    prepare_representation_probe_contract,
    prepare_representation_probe_data,
    run_representation_probe_confirmation,
    run_representation_probe_selection,
)
from .stats_robust_experts import (
    prepare_robust_expert_contract,
    prepare_robust_expert_data,
    run_robust_expert_arm,
    run_robust_expert_training,
    select_robust_expert_route,
)
from .stats_route import run_oracle_family_route
from .stats_router_counterfactual import (
    prepare_historical_counterfactual_replay_contract,
    run_historical_counterfactual_replay,
)
from .stats_router_external import (
    prepare_historical_external_contract,
    run_historical_external_evaluation,
)
from .stats_router_failure import diagnose_family_router_replication_failure
from .stats_router_reduced import (
    prepare_reduced_family_router_contract,
    run_reduced_family_router_confirmation,
)
from .stats_router_replay import (
    prepare_historical_matched_replay_contract,
    run_historical_matched_replay,
)
from .stats_router_replication import (
    prepare_family_router_replication_contract,
    run_family_router_replication,
)
from .stats_sandbox import sandbox_self_test as stats_sandbox_self_test
from .stats_selective_external import (
    prepare_selective_external_contract,
    prepare_selective_external_data,
    prepare_selective_external_source_qualification,
    run_selective_external_evaluation,
)
from .stats_selector_external import (
    prepare_selector_external_contract,
    prepare_selector_external_data,
    run_selector_external_evaluation,
)
from .stats_selector_external_amendment import (
    prepare_selector_external_evaluation_amendment,
    run_selector_external_evaluation_amended,
)
from .stats_selector_head import (
    prepare_selector_head_contract,
    prepare_selector_head_data,
    run_selector_head_confirmation,
    run_selector_head_pilot,
)
from .stats_selector_runtime import (
    freeze_selector_runtime,
    predict_selector_runtime,
    verify_selector_runtime,
)
from .stats_selector_sufficiency import (
    prepare_selector_sufficiency_contract,
    prepare_selector_sufficiency_data,
    run_selector_sufficiency_confirmation,
    run_selector_sufficiency_historical_e3,
    run_selector_sufficiency_selection,
)
from .stats_semantic_catalog import (
    prepare_semantic_catalog_contract,
    prepare_semantic_catalog_data,
    run_semantic_catalog_confirmation,
    run_semantic_catalog_pilot,
)
from .stats_style_invariance import (
    prepare_style_invariance_contract,
    prepare_style_invariance_data,
    run_style_invariance_confirmation,
    run_style_invariance_selection,
)
from .stats_sufficiency_guard import (
    diagnose_sufficiency_guard_margin,
    prepare_sufficiency_guard_contract,
    run_sufficiency_guard_confirmation,
)
from .stats_sufficiency_thresholded import (
    prepare_thresholded_sufficiency_guard_contract,
    run_thresholded_sufficiency_guard_confirmation,
)
from .stats_targeted_repair import (
    prepare_targeted_repair_contract,
    prepare_targeted_repair_data,
    run_targeted_repair_arm,
    run_targeted_repair_training,
    select_targeted_repair_route,
)
from .stats_training import (
    calibrate_stats_adapter,
    run_stats_pilot_candidate,
    run_stats_pilots,
    run_stats_training,
    score_stats_selector,
)
from .training import _base_snapshot, run_pilot, run_training

app = typer.Typer(no_args_is_help=True, help="Charlie alpha overnight training pipeline.")
data_app = typer.Typer(no_args_is_help=True)
train_app = typer.Typer(no_args_is_help=True)
eval_app = typer.Typer(no_args_is_help=True)
export_app = typer.Typer(no_args_is_help=True)
release_app = typer.Typer(no_args_is_help=True)
forge_app = typer.Typer(no_args_is_help=True, help="Forge v0.2 efficient research pipeline.")
stats_app = typer.Typer(no_args_is_help=True, help="Charlie alpha v0.3 statistics pipeline.")
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(export_app, name="export")
app.add_typer(release_app, name="release")
app.add_typer(forge_app, name="forge")
app.add_typer(stats_app, name="stats")
console = Console()

ConfigOption = Annotated[
    Path,
    typer.Option("--config", exists=True, dir_okay=False, resolve_path=True),
]


def _show(value: object) -> None:
    console.print_json(json.dumps(value, ensure_ascii=False, default=str))


@data_app.command("prepare")
def data_prepare(config: ConfigOption = Path("configs/pipeline.yaml"), force: bool = False) -> None:
    _show(prepare_data(load_config(config), force=force))


@data_app.command("distill")
def data_distill(config: ConfigOption = Path("configs/pipeline.yaml"), force: bool = False) -> None:
    _show(distill_data(load_config(config), force=force))


@data_app.command("mix")
def data_mix(config: ConfigOption = Path("configs/pipeline.yaml"), force: bool = False) -> None:
    _show(mix_data(load_config(config), force=force))


@train_app.command("pilot")
def train_pilot(config: ConfigOption = Path("configs/pipeline.yaml"), force: bool = False) -> None:
    _show(run_pilot(load_config(config), force=force))


@train_app.command("run")
def train_run(config: ConfigOption = Path("configs/pipeline.yaml"), force: bool = False) -> None:
    _show(run_training(load_config(config), force=force))


@eval_app.command("run")
def eval_run(
    variant: Annotated[str, typer.Option("--variant")] = "adapter",
    config: ConfigOption = Path("configs/pipeline.yaml"),
    force: bool = False,
) -> None:
    _show(run_evaluation(load_config(config), variant=variant, force=force))


@export_app.command("all")
def export_everything(
    config: ConfigOption = Path("configs/pipeline.yaml"),
    gguf: Annotated[bool, typer.Option("--gguf")] = False,
) -> None:
    _show(export_all(load_config(config), include_gguf=gguf))


@export_app.command("gguf")
def export_gguf_only(config: ConfigOption = Path("configs/pipeline.yaml")) -> None:
    _show(export_gguf(load_config(config)))


@export_app.command("validate-clean")
def export_validate_clean(config: ConfigOption = Path("configs/pipeline.yaml")) -> None:
    _show(validate_clean_environment(load_config(config)))


@release_app.command("check")
def release_check(config: ConfigOption = Path("configs/pipeline.yaml")) -> None:
    _show(check_release(load_config(config)))


@release_app.command("publish-hf")
def release_publish_hf(
    config: ConfigOption = Path("configs/pipeline.yaml"),
    gguf: Annotated[bool, typer.Option("--gguf")] = False,
) -> None:
    _show(publish_hugging_face(load_config(config), include_gguf=gguf))


@release_app.command("publish-github")
def release_publish_github(
    config: ConfigOption = Path("configs/pipeline.yaml"),
) -> None:
    _show(publish_github(load_config(config)))


@forge_app.command("lock-eval")
def forge_lock_eval(
    config: ConfigOption = Path("configs/pipeline.v2.yaml"), force: bool = False
) -> None:
    _show(build_evaluation_lock(load_config(config), force=force))


@forge_app.command("prepare")
def forge_prepare(
    config: ConfigOption = Path("configs/pipeline.v2.yaml"), force: bool = False
) -> None:
    _show(prepare_forge_candidates(load_config(config), force=force))


@forge_app.command("score")
def forge_score(
    config: ConfigOption = Path("configs/pipeline.v2.yaml"), force: bool = False
) -> None:
    _show(score_forge_candidates(load_config(config), force=force))


@forge_app.command("select")
def forge_select(
    config: ConfigOption = Path("configs/pipeline.v2.yaml"), force: bool = False
) -> None:
    _show(select_forge_sources(load_config(config), force=force))


@forge_app.command("distill")
def forge_distill(
    config: ConfigOption = Path("configs/pipeline.v2.yaml"), force: bool = False
) -> None:
    _show(distill_forge_translations(load_config(config), force=force))


@forge_app.command("build")
def forge_build(
    config: ConfigOption = Path("configs/pipeline.v2.yaml"), force: bool = False
) -> None:
    _show(build_forge_data(load_config(config), force=force))


@forge_app.command("pilot")
def forge_pilot(
    config: ConfigOption = Path("configs/pipeline.v2.yaml"), force: bool = False
) -> None:
    _show(run_forge_pilots(load_config(config), force=force))


@forge_app.command("pilot-one", hidden=True)
def forge_pilot_one(
    candidate: Annotated[str, typer.Option("--candidate")],
    config: ConfigOption = Path("configs/pipeline.v2.yaml"),
    force: bool = False,
) -> None:
    _show(run_forge_pilot_candidate(load_config(config), candidate_name=candidate, force=force))


@forge_app.command("train")
def forge_train(
    config: ConfigOption = Path("configs/pipeline.v2.yaml"), force: bool = False
) -> None:
    _show(run_forge_training(load_config(config), force=force))


@forge_app.command("calibrate")
def forge_calibrate(
    config: ConfigOption = Path("configs/pipeline.v2.yaml"), force: bool = False
) -> None:
    _show(calibrate_forge_adapter(load_config(config), force=force))


@forge_app.command("eval")
def forge_eval(
    variant: Annotated[str, typer.Option("--variant")] = "qwen35-base",
    suite: Annotated[str, typer.Option("--suite")] = "dev",
    config: ConfigOption = Path("configs/pipeline.v2.yaml"),
    force: bool = False,
) -> None:
    _show(run_forge_evaluation(load_config(config), variant=variant, suite=suite, force=force))


@forge_app.command("compare")
def forge_compare(
    suite: Annotated[str, typer.Option("--suite")] = "dev",
    config: ConfigOption = Path("configs/pipeline.v2.yaml"),
) -> None:
    _show(compare_forge_evaluation(load_config(config), suite=suite))


@forge_app.command("freeze")
def forge_freeze(config: ConfigOption = Path("configs/pipeline.v2.yaml")) -> None:
    _show(freeze_forge_recipe(load_config(config)))


@forge_app.command("overnight")
def forge_overnight(config: ConfigOption = Path("configs/pipeline.v2.yaml")) -> None:
    _show(run_forge_overnight(load_config(config)))


@forge_app.command("router-lock")
def forge_router_lock(
    config: ConfigOption = Path("configs/pipeline.v2.yaml"), force: bool = False
) -> None:
    _show(build_router_confirmation_lock(load_config(config), force=force))


@forge_app.command("router-freeze")
def forge_router_freeze(
    config: ConfigOption = Path("configs/pipeline.v2.yaml"),
) -> None:
    _show(freeze_router_recipe(load_config(config)))


@forge_app.command("router-eval")
def forge_router_eval(
    variant: Annotated[str, typer.Option("--variant")] = "qwen35-base",
    config: ConfigOption = Path("configs/pipeline.v2.yaml"),
    force: bool = False,
) -> None:
    _show(run_router_confirmation(load_config(config), variant=variant, force=force))


@forge_app.command("router-compare")
def forge_router_compare(
    config: ConfigOption = Path("configs/pipeline.v2.yaml"),
) -> None:
    _show(compare_router_confirmation(load_config(config)))


@forge_app.command("router-verify")
def forge_router_verify(
    config: ConfigOption = Path("configs/pipeline.v2.yaml"),
) -> None:
    _show(verify_dynamic_router(load_config(config)))


StatsConfigOption = Annotated[
    Path,
    typer.Option("--config", exists=True, dir_okay=False, resolve_path=True),
]


@stats_app.command("setup")
def stats_setup(config: StatsConfigOption = Path("configs/pipeline.stats.yaml")) -> None:
    project = load_config(config)
    pixi = shutil.which("pixi") or str(Path.home() / ".pixi" / "bin" / "pixi")
    if not Path(pixi).exists():
        raise typer.BadParameter("Pixi is missing; install the pinned user-level Pixi runtime")
    subprocess.run([pixi, "install", "--locked"], cwd=project.root, check=True)
    runtime = resolve_stats_runtime(project)
    _show(
        {
            "pixi": subprocess.run(
                [pixi, "--version"], text=True, capture_output=True, check=True
            ).stdout.strip(),
            "isolation": stats_sandbox_self_test(
                python_executable=runtime.python,
                r_executable=runtime.rscript,
            ),
        }
    )


@stats_app.command("simulate")
def stats_simulate(
    config: StatsConfigOption = Path("configs/pipeline.stats.yaml"), force: bool = False
) -> None:
    project = load_config(config)
    _show(
        {
            "blueprints": prepare_stats_blueprints(project, force=force),
            "surface": simulate_stats_surface(project, force=force),
        }
    )


@stats_app.command("data")
def stats_data(
    config: StatsConfigOption = Path("configs/pipeline.stats.yaml"), force: bool = False
) -> None:
    _show(build_stats_data(load_config(config), force=force))


@stats_app.command("distill")
def stats_distill(
    config: StatsConfigOption = Path("configs/pipeline.stats.yaml"), force: bool = False
) -> None:
    _show(distill_stats_explanations(load_config(config), force=force))


@stats_app.command("lock-eval")
def stats_lock_eval(
    config: StatsConfigOption = Path("configs/pipeline.stats.yaml"), force: bool = False
) -> None:
    _show(build_stats_evaluation_lock(load_config(config), force=force))


@stats_app.command("baseline")
def stats_baseline(config: StatsConfigOption = Path("configs/pipeline.stats.yaml")) -> None:
    _show(score_stats_selector(load_config(config), adapter_path=None, split="dev"))


@stats_app.command("pilot")
def stats_pilot(
    config: StatsConfigOption = Path("configs/pipeline.stats.yaml"), force: bool = False
) -> None:
    _show(run_stats_pilots(load_config(config), force=force))


@stats_app.command("pilot-one", hidden=True)
def stats_pilot_one(
    variant: Annotated[str, typer.Option("--variant")],
    config: StatsConfigOption = Path("configs/pipeline.stats.yaml"),
    force: bool = False,
) -> None:
    _show(run_stats_pilot_candidate(load_config(config), variant=variant, force=force))


@stats_app.command("train")
def stats_train(
    config: StatsConfigOption = Path("configs/pipeline.stats.yaml"), force: bool = False
) -> None:
    project = load_config(config)
    trained = run_stats_training(project, force=force)
    calibrated = calibrate_stats_adapter(project, force=force)
    _show({"trained": trained, "calibrated": calibrated})


@stats_app.command("freeze")
def stats_freeze(config: StatsConfigOption = Path("configs/pipeline.stats.yaml")) -> None:
    _show(freeze_stats_recipe(load_config(config)))


@stats_app.command("eval")
def stats_eval(
    variant: Annotated[str, typer.Option("--variant")] = "all",
    config: StatsConfigOption = Path("configs/pipeline.stats.yaml"),
    force: bool = False,
) -> None:
    project = load_config(config)
    if variant == "all":
        freeze_stats_recipe(project)
        selected = json.loads(
            (project.path_for("artifact_dir") / "selected.json").read_text(encoding="utf-8")
        )
        variants = ["base", "hard-label", "dgp-regret"]
        if selected.get("variant") != "dgp-regret":
            variants.append("selected")
        reports = {
            name: run_stats_evaluation(project, variant=name, force=force) for name in variants
        }
        _show({"reports": reports, "comparison": compare_stats_evaluation(project)})
        return
    _show(run_stats_evaluation(project, variant=variant, force=force))


@stats_app.command("compare")
def stats_compare(config: StatsConfigOption = Path("configs/pipeline.stats.yaml")) -> None:
    _show(compare_stats_evaluation(load_config(config)))


@stats_app.command("export")
def stats_export(
    config: StatsConfigOption = Path("configs/pipeline.stats.yaml"),
    gguf: Annotated[bool, typer.Option("--gguf")] = False,
) -> None:
    _show(export_stats(load_config(config), include_gguf=gguf))


@stats_app.command("release-check")
def stats_release_check(
    config: StatsConfigOption = Path("configs/pipeline.stats.yaml"),
) -> None:
    _show(check_stats_release(load_config(config)))


@stats_app.command("publish-hf")
def stats_publish_hf(
    config: StatsConfigOption = Path("configs/pipeline.stats.yaml"),
    gguf: Annotated[bool, typer.Option("--gguf")] = False,
) -> None:
    _show(publish_stats_hugging_face(load_config(config), include_gguf=gguf))


@stats_app.command("publish-github")
def stats_publish_github(
    config: StatsConfigOption = Path("configs/pipeline.stats.yaml"),
) -> None:
    _show(publish_stats_github(load_config(config)))


@stats_app.command("analyze")
def stats_analyze(
    data: Annotated[list[Path], typer.Option("--data", exists=True, dir_okay=False)],
    question: Annotated[str, typer.Option("--question")],
    language: Annotated[str, typer.Option("--language")] = "auto",
    route: Annotated[str, typer.Option("--route")] = "stats",
    config: StatsConfigOption = Path("configs/pipeline.stats.yaml"),
    adapter_path: Annotated[str | None, typer.Option("--adapter-path")] = None,
) -> None:
    project = load_config(config)
    selected_route = classify_stats_route(question, has_files=True, override=route)
    agent = StatsAgent(project, adapter_path=adapter_path)
    _show(
        agent.analyze(
            data_paths=data,
            question=question,
            language=language,
            route=selected_route,
        )
    )


@stats_app.command("chat")
def stats_chat(
    data: Annotated[list[Path] | None, typer.Option("--data", exists=True, dir_okay=False)] = None,
    language: Annotated[str, typer.Option("--language")] = "auto",
    route: Annotated[str, typer.Option("--route")] = "auto",
    config: StatsConfigOption = Path("configs/pipeline.stats.yaml"),
    adapter_path: Annotated[str | None, typer.Option("--adapter-path")] = None,
) -> None:
    project = load_config(config)
    agent = StatsAgent(project, adapter_path=adapter_path)
    files = data or []
    active_route = route
    history: list[dict[str, str]] = []
    console.print("Charlie alpha statistics — /quit exits; /route auto|base|stats changes routing")
    while True:
        question = console.input("[bold cyan]You> [/bold cyan]")
        if question.strip() in {"/quit", "/exit"}:
            break
        if question.startswith("/route "):
            active_route = question.split(maxsplit=1)[1].strip().lower()
            classify_stats_route("", has_files=bool(files), override=active_route)
            console.print(f"route override: {active_route}")
            continue
        selected_route = classify_stats_route(
            question,
            has_files=bool(files),
            override=active_route,
        )
        messages = [*history, {"role": "user", "content": question}]
        if files and selected_route == "stats":
            result = agent.analyze(
                data_paths=files,
                question=question,
                language=language,
                route=selected_route,
                conversation=history,
            )
            answer = str(result["answer"])
            console.print(
                f"[dim]route={selected_route}; tool_calls={result['tool_calls']}[/dim]\n"
                f"[bold green]Charlie alpha>[/bold green] {answer}"
            )
        else:
            answer = agent.answer_without_tools(
                messages,
                route=selected_route,
            )
            console.print(
                f"[dim]route={selected_route}[/dim]\n"
                f"[bold green]Charlie alpha>[/bold green] {answer}"
            )
        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        history = history[-8:]


@stats_app.command("serve")
def stats_serve(
    config: StatsConfigOption = Path("configs/pipeline.stats.yaml"),
    host: str = "127.0.0.1",
    port: int = 8080,
    adapter_path: Annotated[str | None, typer.Option("--adapter-path")] = None,
) -> None:
    project = load_config(config)
    command = [
        "/usr/bin/caffeinate",
        "-dimsu",
        sys.executable,
        "-m",
        "charlie_alpha.stats_server",
        "--config",
        str(project.path),
        "--host",
        host,
        "--port",
        str(port),
    ]
    if adapter_path:
        command.extend(["--adapter-path", adapter_path])
    raise typer.Exit(subprocess.call(command, cwd=project.root))


@stats_app.command("overnight")
def stats_overnight(config: StatsConfigOption = Path("configs/pipeline.stats.yaml")) -> None:
    _show(run_stats_pipeline(load_config(config)))


@stats_app.command("iterate")
def stats_iterate(
    cycles: Annotated[int, typer.Option("--cycles", min=1, max=2)] = 1,
    prepare_only: Annotated[bool, typer.Option("--prepare-only")] = False,
    force: bool = False,
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Run verified DGP generation, short adapter training, and gated promotion."""
    _show(
        run_evolution(
            load_config(config),
            cycles=cycles,
            prepare_only=prepare_only,
            force=force,
        )
    )


@stats_app.command("evolve-status")
def stats_evolve_status(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    _show(evolution_status(load_config(config)))


@stats_app.command("base-bakeoff")
def stats_base_bakeoff(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Compare the locked 4B and 9B bases on local quality and QLoRA cost."""
    _show(run_base_bakeoff(load_config(config), force=force))


@stats_app.command("policy-project")
def stats_policy_project(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    prepare_only: Annotated[bool, typer.Option("--prepare-only")] = False,
    balanced: Annotated[bool, typer.Option("--balanced")] = False,
    force: bool = False,
) -> None:
    """Run the matched multi-seed DGP policy-projection pilot."""
    loaded = load_config(config)
    if prepare_only:
        _show(prepare_policy_projection_data(loaded, force=force))
    else:
        _show(run_policy_projection_pilot(loaded, force=force, balanced=balanced))


@stats_app.command("policy-diagnose")
def stats_policy_diagnose(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Measure paired cross-family LoRA gradient conflict at the frozen parent."""
    _show(diagnose_policy_projection_gradients(load_config(config), force=force))


@stats_app.command("policy-cone")
def stats_policy_cone(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Run deterministic common-descent and uniform-family matched arms."""
    _show(run_common_descent_pilot(load_config(config), force=force))


@stats_app.command("policy-cone-promote")
def stats_policy_cone_promote(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Open a fresh promotion shard only after every cone pilot gate passes."""
    _show(promote_common_descent_candidate(load_config(config), force=force))


@stats_app.command("policy-cone-confirm")
def stats_policy_cone_confirm(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Confirm the fixed uniform-family winner on reusable trilingual dev data."""
    _show(confirm_uniform_family_candidate(load_config(config), force=force))


@stats_app.command("policy-calibrate")
def stats_policy_calibrate(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Calibrate the parent-to-uniform adapter delta on reusable surfaces."""
    _show(run_delta_calibration(load_config(config), force=force))


@stats_app.command("policy-calibrate-arm", hidden=True)
def stats_policy_calibrate_arm(
    scale: Annotated[float, typer.Option("--scale")],
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Score one isolated adapter-delta interpolation scale."""
    _show(run_delta_calibration_arm(load_config(config), scale=scale, force=force))


@stats_app.command("policy-block")
def stats_policy_block(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Search pre-registered sparse layer supports on reusable surfaces."""
    _show(run_block_projection(load_config(config), force=force))


@stats_app.command("policy-block-arm", hidden=True)
def stats_policy_block_arm(
    layer_scales: Annotated[str, typer.Option("--layer-scales")],
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Score one isolated effective-weight block projection."""
    _show(
        run_block_projection_arm(
            load_config(config),
            layer_scales=parse_layer_scales(layer_scales),
            force=force,
        )
    )


@stats_app.command("policy-family-route")
def stats_policy_family_route(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    selection_only: Annotated[bool, typer.Option("--selection-only")] = False,
    force: bool = False,
) -> None:
    """Test the oracle DGP-family upper bound of routed block profiles."""
    _show(
        run_oracle_family_route(
            load_config(config),
            force=force,
            selection_only=selection_only,
        )
    )


@stats_app.command("policy-family-experts")
def stats_policy_family_experts(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    train_only: Annotated[bool, typer.Option("--train-only")] = False,
    force: bool = False,
) -> None:
    """Test compute-matched family-specific LoRAs as an oracle upper bound."""
    _show(
        run_family_expert_oracle(
            load_config(config),
            force=force,
            train_only=train_only,
        )
    )


@stats_app.command("policy-family-expert-arm", hidden=True)
def stats_policy_family_expert_arm(
    family: Annotated[str, typer.Option("--family")],
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Train one isolated DGP-family expert from the unchanged parent."""
    _show(run_family_expert_arm(load_config(config), family_id=family, force=force))


@stats_app.command("policy-router-prepare")
def stats_policy_router_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Train and calibrate the lightweight trilingual family router."""
    _show(prepare_family_router(load_config(config), force=force))


@stats_app.command("policy-router")
def stats_policy_router(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Select and confirm the deployable selective family-expert route."""
    _show(run_family_router(load_config(config), force=force))


@stats_app.command("policy-llm-router")
def stats_policy_llm_router(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    selection_only: Annotated[bool, typer.Option("--selection-only")] = False,
    force: bool = False,
) -> None:
    """Select and confirm the frozen-parent single-token family route."""
    _show(
        run_llm_family_router(
            load_config(config),
            force=force,
            selection_only=selection_only,
        )
    )


@stats_app.command("policy-llm-router-promote")
def stats_policy_llm_router_promote(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Open the larger promotion shard only after real-router confirmation."""
    _show(promote_llm_family_router(load_config(config), force=force))


@stats_app.command("policy-llm-router-final")
def stats_policy_llm_router_final(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Open the sealed v0.3 DGP final only after router promotion passes."""
    _show(evaluate_llm_family_router_final(load_config(config), force=force))


@stats_app.command("policy-llm-router-replication-prepare")
def stats_policy_llm_router_replication_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze the powered fresh-surface replication contract before opening it."""
    _show(prepare_family_router_replication_contract(load_config(config)))


@stats_app.command("policy-llm-router-replicate")
def stats_policy_llm_router_replicate(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Run the preregistered independent frozen-router replication."""
    _show(run_family_router_replication(load_config(config), force=force))


@stats_app.command("policy-llm-router-replication-diagnose")
def stats_policy_llm_router_replication_diagnose(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Diagnose an isolated granular failure without rescuing the rejected route."""
    _show(diagnose_family_router_replication_failure(load_config(config)))


@stats_app.command("policy-llm-router-reduced-prepare")
def stats_policy_llm_router_reduced_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Preregister the prospective reduced family route before fresh confirmation."""
    _show(prepare_reduced_family_router_contract(load_config(config)))


@stats_app.command("policy-llm-router-reduced-confirm")
def stats_policy_llm_router_reduced_confirm(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Confirm the prospective reduced family route on its fresh powered surface."""
    _show(run_reduced_family_router_confirmation(load_config(config), force=force))


@stats_app.command("policy-sufficiency-guard-prepare")
def stats_policy_sufficiency_guard_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Preregister the binary sufficiency guard before fresh paired confirmation."""
    _show(prepare_sufficiency_guard_contract(load_config(config)))


@stats_app.command("policy-sufficiency-guard-confirm")
def stats_policy_sufficiency_guard_confirm(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Confirm guard safety and reduced-route efficacy on a fresh paired surface."""
    _show(run_sufficiency_guard_confirmation(load_config(config), force=force))


@stats_app.command("policy-sufficiency-guard-diagnose")
def stats_policy_sufficiency_guard_diagnose(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Test whether a rejected guard has a prospective probability margin."""
    _show(diagnose_sufficiency_guard_margin(load_config(config)))


@stats_app.command("policy-sufficiency-guard-thresholded-prepare")
def stats_policy_sufficiency_guard_thresholded_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Preregister the calibrated guard before its disjoint confirmation."""
    _show(prepare_thresholded_sufficiency_guard_contract(load_config(config)))


@stats_app.command("policy-sufficiency-guard-thresholded-confirm")
def stats_policy_sufficiency_guard_thresholded_confirm(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Confirm the calibrated guard and reduced route on a fresh paired surface."""
    _show(run_thresholded_sufficiency_guard_confirmation(load_config(config), force=force))


@stats_app.command("policy-router-historical-external-prepare")
def stats_policy_router_historical_external_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze the historical external falsification contract."""
    _show(prepare_historical_external_contract(load_config(config)))


@stats_app.command("policy-router-historical-external")
def stats_policy_router_historical_external(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Run routed adapters on the previously opened P-Bench and StatQA suites."""
    _show(run_historical_external_evaluation(load_config(config), force=force))


@stats_app.command("policy-router-historical-matched-replay-prepare")
def stats_policy_router_historical_matched_replay_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze the post-H4 matched historical replay diagnostic."""
    _show(prepare_historical_matched_replay_contract(load_config(config)))


@stats_app.command("policy-router-historical-matched-replay")
def stats_policy_router_historical_matched_replay(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Replay v0.3 under the H4 evaluator before attributing historical deltas."""
    _show(run_historical_matched_replay(load_config(config), force=force))


@stats_app.command("policy-router-historical-counterfactual-prepare")
def stats_policy_router_historical_counterfactual_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze the efficient matched-parent completion after v1 partial replay."""
    _show(prepare_historical_counterfactual_replay_contract(load_config(config)))


@stats_app.command("policy-router-historical-counterfactual")
def stats_policy_router_historical_counterfactual(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Complete only the frozen expert-routed counterfactual parent tasks."""
    _show(run_historical_counterfactual_replay(load_config(config), force=force))


@stats_app.command("cross-format-prepare")
def stats_cross_format_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze H5 cross-format representation-repair blueprints and gates."""
    _show(prepare_cross_format_contract(load_config(config)))


@stats_app.command("cross-format-data")
def stats_cross_format_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Prepare only H5 training and selection surfaces; keep confirmation sealed."""
    _show(prepare_cross_format_data(load_config(config)))


@stats_app.command("cross-format-arm")
def stats_cross_format_arm(
    arm: str,
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Train one preregistered H5 matched-compute arm."""
    _show(run_cross_format_arm(load_config(config), arm=arm))


@stats_app.command("cross-format-pilot")
def stats_cross_format_pilot(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Run H5 matched arms and evaluate the fresh menu-free selection format."""
    _show(run_cross_format_pilot_amended(load_config(config)))


@stats_app.command("cross-format-confirm")
def stats_cross_format_confirm(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Open H5 fresh confirmation only after its preregistered selection gate passes."""
    _show(run_cross_format_confirmation_amended(load_config(config)))


@stats_app.command("canonical-bottleneck-prepare")
def stats_canonical_bottleneck_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze H6 canonical semantic-bottleneck data and gates."""
    _show(prepare_canonical_bottleneck_contract(load_config(config)))


@stats_app.command("canonical-bottleneck-data")
def stats_canonical_bottleneck_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Prepare H6 train and selection data while leaving confirmation sealed."""
    _show(prepare_canonical_bottleneck_data(load_config(config)))


@stats_app.command("canonical-bottleneck-pilot")
def stats_canonical_bottleneck_pilot(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Run the H6 matched canonical-ID versus display-name pilot."""
    _show(run_canonical_bottleneck_pilot(load_config(config)))


@stats_app.command("canonical-bottleneck-confirm")
def stats_canonical_bottleneck_confirm(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Open H6 confirmation only after the registered selection gate passes."""
    _show(run_canonical_bottleneck_confirmation(load_config(config)))


@stats_app.command("catalog-grounding-prepare")
def stats_catalog_grounding_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze H7 fixed-catalog interface data and gates."""
    _show(prepare_catalog_grounding_contract(load_config(config)))


@stats_app.command("catalog-grounding-data")
def stats_catalog_grounding_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Prepare H7 fresh selection data while leaving confirmation sealed."""
    _show(prepare_catalog_grounding_data(load_config(config)))


@stats_app.command("catalog-grounding-pilot")
def stats_catalog_grounding_pilot(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Evaluate the H7 menu-free versus fixed-catalog interface pilot."""
    _show(run_catalog_grounding_pilot(load_config(config)))


@stats_app.command("catalog-grounding-confirm")
def stats_catalog_grounding_confirm(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Open H7 confirmation only after the registered selection gate passes."""
    _show(run_catalog_grounding_confirmation(load_config(config)))


@stats_app.command("catalog-distill-prepare")
def stats_catalog_distill_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze H8 catalog-grounding distillation data, dropout, and gates."""
    _show(prepare_catalog_distillation_contract(load_config(config)))


@stats_app.command("catalog-distill-data")
def stats_catalog_distill_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Prepare H8 training and menu-free selection while confirmation stays sealed."""
    _show(prepare_catalog_distillation_data(load_config(config)))


@stats_app.command("catalog-distill-pilot")
def stats_catalog_distill_pilot(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Run H8 matched training arms and the fresh menu-free selection gate."""
    _show(run_catalog_distillation_pilot(load_config(config)))


@stats_app.command("catalog-distill-confirm")
def stats_catalog_distill_confirm(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Open H8 confirmation only after every registered selection gate passes."""
    _show(run_catalog_distillation_confirmation(load_config(config)))


@stats_app.command("catalog-rank-prepare")
def stats_catalog_rank_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze H9 full-catalog constrained-ranking data and gates."""
    _show(prepare_catalog_ranking_contract(load_config(config)))


@stats_app.command("catalog-rank-data")
def stats_catalog_rank_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Prepare H9 fresh selection data while confirmation stays sealed."""
    _show(prepare_catalog_ranking_data(load_config(config)))


@stats_app.command("catalog-rank-pilot")
def stats_catalog_rank_pilot(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Compare free decoding with all-catalog likelihood ranking on fresh cases."""
    _show(run_catalog_ranking_pilot(load_config(config)))


@stats_app.command("catalog-rank-confirm")
def stats_catalog_rank_confirm(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Open H9 confirmation only after every registered selection gate passes."""
    _show(run_catalog_ranking_confirmation(load_config(config)))


@stats_app.command("output-factorization-prepare")
def stats_output_factorization_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze H10 factorized method/column interface data and gates."""
    _show(prepare_output_factorization_contract(load_config(config)))


@stats_app.command("output-factorization-data")
def stats_output_factorization_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Prepare H10 fresh selection data while confirmation remains sealed."""
    _show(prepare_output_factorization_data(load_config(config)))


@stats_app.command("output-factorization-pilot")
def stats_output_factorization_pilot(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Run H10 paired joint-vs-factorized selection on fresh cases."""
    _show(run_output_factorization_pilot(load_config(config)))


@stats_app.command("output-factorization-confirm")
def stats_output_factorization_confirm(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Open H10 confirmation only after every registered selection gate passes."""
    _show(run_output_factorization_confirmation(load_config(config)))


@stats_app.command("semantic-catalog-prepare")
def stats_semantic_catalog_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze H11 semantic-catalog data and gates."""
    _show(prepare_semantic_catalog_contract(load_config(config)))


@stats_app.command("semantic-catalog-data")
def stats_semantic_catalog_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Prepare H11 fresh selection data while confirmation remains sealed."""
    _show(prepare_semantic_catalog_data(load_config(config)))


@stats_app.command("semantic-catalog-pilot")
def stats_semantic_catalog_pilot(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Compare flat and semantic fixed catalogs on fresh cases."""
    _show(run_semantic_catalog_pilot(load_config(config)))


@stats_app.command("semantic-catalog-confirm")
def stats_semantic_catalog_confirm(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Open H11 confirmation only after every registered selection gate passes."""
    _show(run_semantic_catalog_confirmation(load_config(config)))


@stats_app.command("catalog-interface-replication-prepare")
def stats_catalog_interface_replication_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze all H12 multi-seed replication folds and aggregate gates."""
    _show(prepare_catalog_interface_replication_contract(load_config(config)))


@stats_app.command("catalog-interface-replication-data")
def stats_catalog_interface_replication_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Prepare all preregistered H12 folds without changing the frozen contract."""
    _show(prepare_catalog_interface_replication_data(load_config(config)))


@stats_app.command("catalog-interface-replication-run")
def stats_catalog_interface_replication_run(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Run the three-fold H12 independent replication and aggregate gate."""
    _show(run_catalog_interface_replication(load_config(config)))


@stats_app.command("representation-probe-prepare")
def stats_representation_probe_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze H13 final-hidden-state linear-probe shards, protocol, and gates."""
    _show(prepare_representation_probe_contract(load_config(config)))


@stats_app.command("representation-probe-data")
def stats_representation_probe_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Prepare H13 train/selection cases while confirmation remains sealed."""
    _show(prepare_representation_probe_data(load_config(config)))


@stats_app.command("representation-probe-select")
def stats_representation_probe_select(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Extract frozen representations and choose the preregistered H13 route."""
    _show(run_representation_probe_selection(load_config(config)))


@stats_app.command("representation-probe-confirm")
def stats_representation_probe_confirm(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Open H13 confirmation only after selection chooses one representation route."""
    _show(run_representation_probe_confirmation(load_config(config)))


@stats_app.command("selector-head-prepare")
def stats_selector_head_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze the H14 menu-free selector head, fresh shards, and gates."""
    _show(prepare_selector_head_contract(load_config(config)))


@stats_app.command("selector-head-data")
def stats_selector_head_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Prepare H14 fresh selection cases while confirmation remains sealed."""
    _show(prepare_selector_head_data(load_config(config)))


@stats_app.command("selector-head-pilot")
def stats_selector_head_pilot(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Compare menu-free generation with the frozen H13 selector head."""
    _show(run_selector_head_pilot(load_config(config)))


@stats_app.command("selector-head-confirm")
def stats_selector_head_confirm(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Open H14 confirmation only after every selector-head pilot gate passes."""
    _show(run_selector_head_confirmation(load_config(config)))


@stats_app.command("selector-runtime-freeze")
def stats_selector_runtime_freeze(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze the operational runtime form of the confirmed H14 selector head."""
    _show(freeze_selector_runtime(load_config(config)))


@stats_app.command("selector-runtime-verify")
def stats_selector_runtime_verify(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Verify the frozen runtime, H14 evidence, parent adapter, and head hashes."""
    _show(verify_selector_runtime(load_config(config)))


@stats_app.command("selector-runtime-predict")
def stats_selector_runtime_predict(
    question: Annotated[str, typer.Option("--question")],
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Select a canonical method and columns using the frozen H14 runtime."""
    _show(predict_selector_runtime(load_config(config), question))


@stats_app.command("selector-external-prepare")
def stats_selector_external_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze E3 source, mapping, eligibility, runtime, and external gates before model output."""
    _show(prepare_selector_external_contract(load_config(config)))


@stats_app.command("selector-external-data")
def stats_selector_external_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Materialize all 20 frozen E3 source scenarios and coverage accounting."""
    _show(prepare_selector_external_data(load_config(config)))


@stats_app.command("selector-external-run")
def stats_selector_external_run(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Run the one-shot E3 menu-free versus frozen selector-head external evaluation."""
    _show(run_selector_external_evaluation(load_config(config)))


@stats_app.command("selector-external-amend")
def stats_selector_external_amend(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze the pre-output E3 evaluator-only prompt/interface amendment."""
    _show(prepare_selector_external_evaluation_amendment(load_config(config)))


@stats_app.command("selector-external-run-amended")
def stats_selector_external_run_amended(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Run E3 once with the frozen evaluator amendment and identical H14 prompts."""
    _show(run_selector_external_evaluation_amended(load_config(config)))


@stats_app.command("style-invariance-prepare")
def stats_style_invariance_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze H15 fresh matched-semantics styles, probe protocol, and gates."""
    _show(prepare_style_invariance_contract(load_config(config)))


@stats_app.command("style-invariance-data")
def stats_style_invariance_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Prepare H15 train/selection semantics while confirmation remains sealed."""
    _show(prepare_style_invariance_data(load_config(config)))


@stats_app.command("style-invariance-select")
def stats_style_invariance_select(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Diagnose H14 cross-style collapse and choose the preregistered H15 route."""
    _show(run_style_invariance_selection(load_config(config)))


@stats_app.command("style-invariance-confirm")
def stats_style_invariance_confirm(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Open H15 confirmation only after selection chooses one diagnostic route."""
    _show(run_style_invariance_confirmation(load_config(config)))


@stats_app.command("external-representation-diagnostic")
def stats_external_representation_diagnostic(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Read back historical E3 with H14/H15 probes without fitting on E3."""
    _show(run_external_representation_diagnostic(load_config(config)))


@stats_app.command("selector-sufficiency-prepare")
def stats_selector_sufficiency_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze H16 fresh support geometry, threshold grid, and confirmation gates."""
    _show(prepare_selector_sufficiency_contract(load_config(config)))


@stats_app.command("selector-sufficiency-data")
def stats_selector_sufficiency_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Prepare H16 training/selection semantics while confirmation remains sealed."""
    _show(prepare_selector_sufficiency_data(load_config(config)))


@stats_app.command("selector-sufficiency-select")
def stats_selector_sufficiency_select(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Select one H16 hidden-geometry support threshold on fresh synthetic data."""
    _show(run_selector_sufficiency_selection(load_config(config)))


@stats_app.command("selector-sufficiency-confirm")
def stats_selector_sufficiency_confirm(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Open H16 confirmation only after one support threshold is frozen."""
    _show(run_selector_sufficiency_confirmation(load_config(config)))


@stats_app.command("selector-sufficiency-historical-e3")
def stats_selector_sufficiency_historical_e3(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Read back E3 only after H16 is frozen; never fit or tune on E3."""
    _show(run_selector_sufficiency_historical_e3(load_config(config)))


@stats_app.command("selective-external-qualify")
def stats_selective_external_qualify(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Qualify the two new CC BY E4 tables before any model output."""
    _show(prepare_selective_external_source_qualification(load_config(config)))


@stats_app.command("selective-external-prepare")
def stats_selective_external_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze E4 sources, exact aliases, cases, H16 threshold, prompt, and gates."""
    _show(prepare_selective_external_contract(load_config(config)))


@stats_app.command("selective-external-data")
def stats_selective_external_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Materialize the frozen 13 E4 source-authored cases without model output."""
    _show(prepare_selective_external_data(load_config(config)))


@stats_app.command("selective-external-run")
def stats_selective_external_run(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Run the one-shot E4 menu-free versus frozen H16 selective evaluation."""
    _show(run_selective_external_evaluation(load_config(config)))


@stats_app.command("external-domain-bridge-prepare")
def stats_external_domain_bridge_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze H17 historical sources, source-held-out training grid, and safety gates."""
    _show(prepare_external_domain_bridge_contract(load_config(config)))


@stats_app.command("external-domain-bridge-data")
def stats_external_domain_bridge_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Materialize only already-opened E2/E3/E4 rows for H17 development training."""
    _show(prepare_external_domain_bridge_data(load_config(config)))


@stats_app.command("external-domain-bridge-train")
def stats_external_domain_bridge_train(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Train and source-held-out validate the H17 residual domain bridge."""
    _show(run_external_domain_bridge_training(load_config(config)))


@stats_app.command("external-domain-bridge-amend")
def stats_external_domain_bridge_amend(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze the H17 execution-only normalized support-bank reconstruction."""
    _show(prepare_external_domain_bridge_execution_amendment(load_config(config)))


@stats_app.command("external-domain-bridge-train-amended")
def stats_external_domain_bridge_train_amended(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Resume H17 under its execution-only amendment without changing any scientific gate."""
    _show(run_external_domain_bridge_training_amended(load_config(config)))


@stats_app.command("external-exemplar-router-prepare")
def stats_external_exemplar_router_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze H18 source-held-out exemplar routing and safety gates."""
    _show(prepare_external_exemplar_router_contract(load_config(config)))


@stats_app.command("external-exemplar-router-train")
def stats_external_exemplar_router_train(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Train and source-held-out validate the H18 external exemplar index."""
    _show(run_external_exemplar_router_training(load_config(config)))


@stats_app.command("external-weight-bridge-prepare")
def stats_external_weight_bridge_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze H19 six-fold LoRA adaptation, replay, compute, and safety gates."""
    _show(prepare_external_weight_bridge_contract(load_config(config)))


@stats_app.command("external-weight-bridge-data")
def stats_external_weight_bridge_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Build the frozen H19 source-held-out training folds and synthetic replay."""
    _show(prepare_external_weight_bridge_data(load_config(config)))


@stats_app.command("external-weight-bridge-train")
def stats_external_weight_bridge_train(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Train and evaluate all six H19 source-held-out LoRA folds."""
    _show(run_external_weight_bridge_training(load_config(config)))


@stats_app.command("external-weight-bridge-amend")
def stats_external_weight_bridge_amend(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze the H19 synthetic-retention field adapter before any fold training."""
    _show(prepare_external_weight_bridge_data_amendment(load_config(config)))


@stats_app.command("external-weight-bridge-data-amended")
def stats_external_weight_bridge_data_amended(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Build H19 folds under the execution-only synthetic-retention field amendment."""
    _show(prepare_external_weight_bridge_data_amended(load_config(config)))


@stats_app.command("external-weight-bridge-train-amended")
def stats_external_weight_bridge_train_amended(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Train all six H19 LoRA folds under the frozen data amendment."""
    _show(run_external_weight_bridge_training_amended(load_config(config)))


@stats_app.command("guarded-weight-bridge-prepare")
def stats_guarded_weight_bridge_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze H20's invalid-control guard, inputs, compute, and development gates."""
    _show(prepare_guarded_weight_bridge_contract(load_config(config)))


@stats_app.command("guarded-weight-bridge-data")
def stats_guarded_weight_bridge_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Build H20's frozen all-source training and synthetic-retention data."""
    _show(prepare_guarded_weight_bridge_data(load_config(config)))


@stats_app.command("guarded-weight-bridge-train")
def stats_guarded_weight_bridge_train(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Validate H20's guard and train its development-only repair adapter."""
    _show(run_guarded_weight_bridge_training(load_config(config)))


@stats_app.command("guarded-external-prepare")
def stats_guarded_external_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze E5's source, runtime, resume, evaluation, and stopping master protocol."""
    _show(prepare_guarded_external_master_contract(load_config(config)))


@stats_app.command("guarded-external-screen")
def stats_guarded_external_screen(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Record the non-blind E5 source feasibility screen without model output."""
    _show(prepare_guarded_external_source_screen(load_config(config)))


@stats_app.command("guarded-external-data")
def stats_guarded_external_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Close E5 at the source gate when no preregistered source qualifies."""
    _show(prepare_guarded_external_data(load_config(config)))


@stats_app.command("guarded-external-run")
def stats_guarded_external_run(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Run exact E5 only after a blind qualified child source contract exists."""
    _show(run_guarded_external_evaluation(load_config(config)))


@stats_app.command("guarded-external-metadata-screen")
def stats_guarded_external_metadata_screen(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Close the metadata-only future-source search without opening rows."""
    _show(prepare_guarded_external_future_metadata_screen(load_config(config)))


@stats_app.command("opened-source-residual-prepare")
def stats_opened_source_residual_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze H21's opened-source residual-opportunity contract."""
    _show(prepare_opened_source_residual_contract(load_config(config)))


@stats_app.command("opened-source-residual-data")
def stats_opened_source_residual_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Materialize H21's reviewed leak-free development cases."""
    _show(prepare_opened_source_residual_data(load_config(config)))


@stats_app.command("opened-source-residual-opportunity")
def stats_opened_source_residual_opportunity(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Screen whether H20 leaves enough residual invalids to justify H21 training."""
    _show(run_opened_source_residual_opportunity(load_config(config)))


@stats_app.command("invalid-control-catalog-fallback-prepare")
def stats_invalid_control_catalog_fallback_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze H22's catalog-grounded invalid-control composition."""
    _show(prepare_invalid_control_catalog_fallback_contract(load_config(config)))


@stats_app.command("invalid-control-catalog-fallback-data")
def stats_invalid_control_catalog_fallback_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Bind H22 to the sealed H21 invalid-control opportunity."""
    _show(prepare_invalid_control_catalog_fallback_data(load_config(config)))


@stats_app.command("invalid-control-catalog-fallback-run")
def stats_invalid_control_catalog_fallback_run(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Run H22's four catalog-grounded parent fallbacks once."""
    _show(run_invalid_control_catalog_fallback(load_config(config)))


@stats_app.command("catalog-fallback-external-prepare")
def stats_catalog_fallback_external_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze E6's fresh-source parent catalog-fallback master protocol."""
    _show(prepare_catalog_fallback_external_master_contract(load_config(config)))


@stats_app.command("catalog-fallback-external-source")
def stats_catalog_fallback_external_source(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Bind E6 to the metadata-only source-availability result."""
    _show(prepare_catalog_fallback_external_source_availability(load_config(config)))


@stats_app.command("catalog-fallback-external-freeze-child")
def stats_catalog_fallback_external_freeze_child(
    metadata: Annotated[
        Path,
        typer.Option("--metadata", exists=True, dir_okay=False, resolve_path=True),
    ],
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze one E6 child from complete metadata-only evidence."""
    _show(
        prepare_catalog_fallback_external_child_contract(
            load_config(config),
            metadata_bundle_path=metadata,
        )
    )


@stats_app.command("catalog-fallback-external-selected-data")
def stats_catalog_fallback_external_selected_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Open one metadata-frozen E6 source and seal its pre-model data."""
    _show(prepare_catalog_fallback_external_selected_data(load_config(config)))


@stats_app.command("catalog-fallback-external-data")
def stats_catalog_fallback_external_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Close E6 at source availability when no child can be frozen."""
    _show(prepare_catalog_fallback_external_data(load_config(config)))


@stats_app.command("catalog-fallback-external-run")
def stats_catalog_fallback_external_run(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Run E6 only after a blind qualified child source exists."""
    _show(run_catalog_fallback_external_evaluation(load_config(config)))


@stats_app.command("external-catalog-prepare")
def stats_external_catalog_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze E1 source identity, aliases, prompts, coverage rules, and gates."""
    _show(prepare_external_catalog_contract(load_config(config)))


@stats_app.command("external-catalog-data")
def stats_external_catalog_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Open the frozen official source and materialize all E1 vignettes once."""
    _show(prepare_external_catalog_data(load_config(config)))


@stats_app.command("external-catalog-run")
def stats_external_catalog_run(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Run the one-shot E1 menu-free versus fixed-catalog external evaluation."""
    _show(run_external_catalog_evaluation(load_config(config)))


@stats_app.command("external-catalog-v2-prepare")
def stats_external_catalog_v2_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Freeze the E2-v2 direct-tabular multi-source external contract."""
    _show(prepare_external_catalog_v2_contract(load_config(config)))


@stats_app.command("external-catalog-v2-data")
def stats_external_catalog_v2_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Materialize and verify every frozen E2-v2 external source row."""
    _show(prepare_external_catalog_v2_data(load_config(config)))


@stats_app.command("external-catalog-v2-run")
def stats_external_catalog_v2_run(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Run the one-shot E2-v2 menu-free versus fixed-catalog evaluation."""
    _show(run_external_catalog_v2_evaluation(load_config(config)))


@stats_app.command("robust-experts-prepare")
def stats_robust_experts_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Preregister every robust-expert shard and gate without simulating or scoring."""
    _show(prepare_robust_expert_contract(load_config(config)))


@stats_app.command("robust-experts-data")
def stats_robust_experts_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Build only robust-expert training and cross-fit selection data."""
    _show(prepare_robust_expert_data(load_config(config), force=force))


@stats_app.command("robust-expert-arm", hidden=True)
def stats_robust_expert_arm(
    family: Annotated[str, typer.Option("--family")],
    arm: Annotated[str, typer.Option("--arm")],
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Train one isolated robust DGP-family expert arm."""
    _show(
        run_robust_expert_arm(
            load_config(config),
            family_id=family,
            arm=arm,
            force=force,
        )
    )


@stats_app.command("robust-experts-train")
def stats_robust_experts_train(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Train all three compute-matched robust family-expert arms."""
    _show(run_robust_expert_training(load_config(config), force=force))


@stats_app.command("robust-experts-select")
def stats_robust_experts_select(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Apply the registered three-fold stability and ablation gates."""
    _show(select_robust_expert_route(load_config(config), force=force))


@stats_app.command("targeted-repair-prepare")
def stats_targeted_repair_prepare(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
) -> None:
    """Preregister v0.6 discovery, fresh shards, compute, and gates."""
    _show(prepare_targeted_repair_contract(load_config(config)))


@stats_app.command("targeted-repair-data")
def stats_targeted_repair_data(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Build fresh targeted training and cross-fit selection data."""
    _show(prepare_targeted_repair_data(load_config(config), force=force))


@stats_app.command("targeted-repair-arm", hidden=True)
def stats_targeted_repair_arm(
    family: Annotated[str, typer.Option("--family")],
    arm: Annotated[str, typer.Option("--arm")],
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Train one isolated v0.6 family-expert ablation arm."""
    _show(
        run_targeted_repair_arm(
            load_config(config),
            family_id=family,
            arm=arm,
            force=force,
        )
    )


@stats_app.command("targeted-repair-train")
def stats_targeted_repair_train(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Train both compute-matched v0.6 targeted repair arms."""
    _show(run_targeted_repair_training(load_config(config), force=force))


@stats_app.command("targeted-repair-select")
def stats_targeted_repair_select(
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Apply fresh three-fold anchor and matched-control gates."""
    _show(select_targeted_repair_route(load_config(config), force=force))


@stats_app.command("policy-cone-arm", hidden=True)
def stats_policy_cone_arm(
    arm: Annotated[str, typer.Option("--arm")],
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    force: bool = False,
) -> None:
    """Run one isolated deterministic common-descent arm."""
    _show(run_common_descent_arm(load_config(config), arm=arm, force=force))


@stats_app.command("policy-project-arm", hidden=True)
def stats_policy_project_arm(
    seed: Annotated[int, typer.Option("--seed")],
    arm: Annotated[str, typer.Option("--arm")],
    config: StatsConfigOption = Path("configs/pipeline.evolve.yaml"),
    balanced: Annotated[bool, typer.Option("--balanced")] = False,
    force: bool = False,
) -> None:
    """Run one isolated policy-projection pilot arm."""
    _show(
        run_policy_projection_arm(
            load_config(config),
            seed=seed,
            arm=arm,
            balanced=balanced,
            force=force,
        )
    )


def _adapter(config: ProjectConfig) -> str:
    selected = config.path_for("artifact_dir") / "selected.json"
    if not selected.exists():
        raise typer.BadParameter("No adapter exists yet; run the training commands first.")
    return json.loads(selected.read_text(encoding="utf-8"))["adapter_path"]


@app.command("overnight")
def overnight(config: ConfigOption = Path("configs/pipeline.yaml")) -> None:
    _show(run_overnight(load_config(config)))


@app.command("chat")
def chat(
    config: ConfigOption = Path("configs/pipeline.yaml"),
    route: Annotated[str, typer.Option("--route")] = "auto",
    adapter_path: Annotated[str | None, typer.Option("--adapter-path")] = None,
) -> None:
    project = load_config(config)
    if project.section("project").get("profile") == "forge-overnight":
        model, tokenizer, router = load_routed_model(project, adapter_path=adapter_path)
        console.print(
            "Charlie alpha — FORGE dynamic sparse LoRA; type /quit to exit "
            "([dim]/route auto|base|adapter[/dim])"
        )
        active_route = route
        while True:
            question = console.input("[bold cyan]You> [/bold cyan]")
            if question.strip() in {"/quit", "/exit"}:
                break
            if question.startswith("/route "):
                candidate = question.split(maxsplit=1)[1].strip().lower()
                if candidate not in {"auto", "base", "adapter"}:
                    console.print("[red]route must be auto, base, or adapter[/red]")
                    continue
                active_route = candidate
                console.print(f"route override: {active_route}")
                continue
            answer, decision = generate_routed(
                model,
                tokenizer,
                router,
                [{"role": "user", "content": question}],
                route=active_route,
            )
            console.print(
                f"[dim]route={decision.route} ({decision.reason})[/dim]\n"
                f"[bold green]Charlie alpha>[/bold green] {answer}"
            )
        return
    model, tokenizer = load(
        _base_snapshot(project),
        adapter_path=_adapter(project),
        tokenizer_config={"trust_remote_code": True},
    )
    console.print("Charlie alpha — type /quit to exit")
    while True:
        question = console.input("[bold cyan]You> [/bold cyan]")
        if question.strip() in {"/quit", "/exit"}:
            break
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        answer = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=1024,
            sampler=make_sampler(temp=0.2, top_p=0.8),
            verbose=False,
        )
        console.print(f"[bold green]Charlie alpha>[/bold green] {answer}")


@app.command("serve")
def serve(
    config: ConfigOption = Path("configs/pipeline.yaml"),
    host: str = "127.0.0.1",
    port: int = 8080,
    adapter_path: Annotated[str | None, typer.Option("--adapter-path")] = None,
) -> None:
    project = load_config(config)
    if project.section("project").get("profile") == "forge-overnight":
        command = [
            "/usr/bin/caffeinate",
            "-dimsu",
            sys.executable,
            "-m",
            "charlie_alpha.routed_server",
            "--config",
            str(project.path),
            "--host",
            host,
            "--port",
            str(port),
        ]
        if adapter_path is not None:
            command.extend(["--adapter-path", adapter_path])
        raise typer.Exit(subprocess.call(command, cwd=project.root))
    command = [
        "/usr/bin/caffeinate",
        "-dimsu",
        sys.executable,
        "-m",
        "mlx_lm",
        "server",
        "--model",
        _base_snapshot(project),
        "--adapter-path",
        _adapter(project),
        "--host",
        host,
        "--port",
        str(port),
        "--temp",
        "0.2",
        "--top-p",
        "0.8",
        "--max-tokens",
        "1024",
    ]
    raise typer.Exit(subprocess.call(command, cwd=project.root))


if __name__ == "__main__":
    app()

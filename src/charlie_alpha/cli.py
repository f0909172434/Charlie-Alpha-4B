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
from .stats_orchestrator import run_stats_pipeline
from .stats_release import (
    check_stats_release,
    export_stats,
    publish_stats_github,
    publish_stats_hugging_face,
)
from .stats_sandbox import sandbox_self_test as stats_sandbox_self_test
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
    _show(
        run_forge_pilot_candidate(
            load_config(config), candidate_name=candidate, force=force
        )
    )


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
    _show(
        run_forge_evaluation(
            load_config(config), variant=variant, suite=suite, force=force
        )
    )


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
            name: run_stats_evaluation(project, variant=name, force=force)
            for name in variants
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

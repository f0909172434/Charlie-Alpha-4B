from __future__ import annotations

import json
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
from .training import _base_snapshot, run_pilot, run_training

app = typer.Typer(no_args_is_help=True, help="Charlie alpha overnight training pipeline.")
data_app = typer.Typer(no_args_is_help=True)
train_app = typer.Typer(no_args_is_help=True)
eval_app = typer.Typer(no_args_is_help=True)
export_app = typer.Typer(no_args_is_help=True)
release_app = typer.Typer(no_args_is_help=True)
forge_app = typer.Typer(no_args_is_help=True, help="Forge v0.2 efficient research pipeline.")
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(export_app, name="export")
app.add_typer(release_app, name="release")
app.add_typer(forge_app, name="forge")
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
) -> None:
    project = load_config(config)
    if project.section("project").get("profile") == "forge-overnight":
        model, tokenizer, router = load_routed_model(project)
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

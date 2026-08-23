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
from .mixer import mix_data
from .orchestrator import run_overnight
from .release import check_release, publish_hugging_face
from .training import _base_snapshot, run_pilot, run_training

app = typer.Typer(no_args_is_help=True, help="Charlie alpha overnight training pipeline.")
data_app = typer.Typer(no_args_is_help=True)
train_app = typer.Typer(no_args_is_help=True)
eval_app = typer.Typer(no_args_is_help=True)
export_app = typer.Typer(no_args_is_help=True)
release_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(export_app, name="export")
app.add_typer(release_app, name="release")
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


def _adapter(config: ProjectConfig) -> str:
    selected = config.path_for("artifact_dir") / "selected.json"
    if not selected.exists():
        raise typer.BadParameter("No adapter exists yet; run the training commands first.")
    return json.loads(selected.read_text(encoding="utf-8"))["adapter_path"]


@app.command("overnight")
def overnight(config: ConfigOption = Path("configs/pipeline.yaml")) -> None:
    _show(run_overnight(load_config(config)))


@app.command("chat")
def chat(config: ConfigOption = Path("configs/pipeline.yaml")) -> None:
    project = load_config(config)
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

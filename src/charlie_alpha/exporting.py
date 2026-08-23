from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from rich.console import Console

from .adapter_conversion import convert_mlx_adapter_to_peft, verify_adapter_equivalence
from .config import ProjectConfig
from .io_utils import canonical_hash, sha256_file, write_json
from .training import _base_snapshot

console = Console()


def _run(command: list[str], *, cwd: Path, timeout: int) -> None:
    result = subprocess.run(
        ["/usr/bin/caffeinate", "-dimsu", *command],
        cwd=cwd,
        text=True,
        check=False,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {command[0]}")


def _smoke_load(model_path: str, adapter_path: str | None = None) -> dict[str, Any]:
    model, tokenizer = load(
        model_path,
        adapter_path=adapter_path,
        tokenizer_config={"trust_remote_code": True},
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Return only: 2"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    output = generate(
        model,
        tokenizer,
        prompt,
        max_tokens=8,
        sampler=make_sampler(temp=0.0),
        verbose=False,
    )
    return {"loaded": True, "generated_nonempty": bool(output.strip())}


def _selected_adapter(config: ProjectConfig) -> Path:
    selected_path = config.path_for("artifact_dir") / "selected.json"
    if not selected_path.exists():
        raise RuntimeError("No selected adapter; run training first.")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    adapter_path = Path(selected["adapter_path"])
    if not (adapter_path / "adapters.safetensors").exists():
        raise RuntimeError(f"Adapter weights are missing: {adapter_path}")
    return adapter_path


def _package_adapter(config: ProjectConfig, adapter_path: Path) -> dict[str, Any]:
    release_dir = config.path_for("artifact_dir") / "release"
    package_dir = release_dir / "Charlie-Alpha-4B-MLX-adapter"
    package_dir.mkdir(parents=True, exist_ok=True)
    required = ["adapters.safetensors"]
    optional = ["best_adapters.safetensors"]
    for name in [*required, *optional]:
        source = adapter_path / name
        if source.exists():
            shutil.copy2(source, package_dir / name)
    private_adapter_config = json.loads(
        (adapter_path / "adapter_config.json").read_text(encoding="utf-8")
    )
    public_adapter_config = dict(private_adapter_config)
    public_adapter_config.update(
        {
            "model": config.sources["models"]["base_mlx_4bit"]["repo_id"],
            "data": "not-bundled-see-data-manifests",
            "adapter_path": ".",
            "resume_adapter_file": None,
            "config": "configs/pipeline.yaml",
        }
    )
    write_json(package_dir / "adapter_config.json", public_adapter_config)
    private_status_path = adapter_path / "training-status.json"
    if private_status_path.exists():
        public_status = json.loads(private_status_path.read_text(encoding="utf-8"))
        public_status.pop("adapter_path", None)
        public_status.pop("model_path", None)
        write_json(package_dir / "training-status.json", public_status)
    shutil.copy2(config.root / "LICENSE", package_dir / "LICENSE")
    shutil.copy2(config.root / "MODEL_CARD.md", package_dir / "README.md")
    shutil.copy2(config.root / "configs" / "sources.lock.json", package_dir / "sources.lock.json")

    checksums = {
        path.name: sha256_file(path)
        for path in sorted(package_dir.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    write_json(package_dir / "SHA256SUMS.json", checksums)
    archive_path = release_dir / "Charlie-Alpha-4B-MLX-adapter.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(package_dir, arcname=package_dir.name)
    return {
        "directory": str(package_dir),
        "archive": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "files": checksums,
    }


def _fuse_mlx(config: ProjectConfig, adapter_path: Path, model_path: str) -> Path:
    output = config.path_for("artifact_dir") / "exports" / "Charlie-Alpha-4B-MLX-4bit"
    if not (output / "config.json").exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable,
                "-m",
                "mlx_lm",
                "fuse",
                "--model",
                model_path,
                "--adapter-path",
                str(adapter_path),
                "--save-path",
                str(output),
            ],
            cwd=config.root,
            timeout=3600,
        )
    shutil.copy2(config.root / "MODEL_CARD.md", output / "README.md")
    shutil.copy2(config.root / "LICENSE", output / "LICENSE")
    shutil.copy2(config.root / "configs" / "sources.lock.json", output / "sources.lock.json")
    replacements = {
        model_path: config.sources["models"]["base_mlx_4bit"]["repo_id"],
        str(config.root): ".",
    }
    for json_path in output.rglob("*.json"):
        text = json_path.read_text(encoding="utf-8")
        for private_value, public_value in replacements.items():
            text = text.replace(private_value, public_value)
        json_path.write_text(text, encoding="utf-8")
    return output


def _merge_hf(config: ProjectConfig, peft_dir: Path, output: Path) -> None:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Run `make export-setup` before GGUF export.") from error

    base = config.sources["models"]["base_hf"]
    model = AutoModelForCausalLM.from_pretrained(
        base["repo_id"],
        revision=base["revision"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
    )
    model = PeftModel.from_pretrained(model, peft_dir)
    merged = model.merge_and_unload(safe_merge=True)
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    tokenizer = AutoTokenizer.from_pretrained(
        base["repo_id"], revision=base["revision"], trust_remote_code=True
    )
    tokenizer.save_pretrained(output)


def _prepare_llama_cpp(config: ProjectConfig) -> Path:
    source = config.sources["tools"]["llama_cpp"]
    directory = config.root / "vendor" / "llama.cpp"
    if not directory.exists():
        _run(
            ["git", "clone", "--filter=blob:none", source["git_url"], str(directory)],
            cwd=config.root,
            timeout=1200,
        )
    _run(["git", "checkout", "--detach", source["revision"]], cwd=directory, timeout=300)
    _run(
        ["cmake", "-S", ".", "-B", "build", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"],
        cwd=directory,
        timeout=600,
    )
    _run(["cmake", "--build", "build", "-j", "4"], cwd=directory, timeout=3600)
    return directory


def export_gguf(config: ProjectConfig, adapter_path: Path | None = None) -> dict[str, Any]:
    adapter_path = adapter_path or _selected_adapter(config)
    artifact_dir = config.path_for("artifact_dir")
    peft_dir = artifact_dir / "exports" / "peft-adapter"
    base = config.sources["models"]["base_hf"]
    mapping = convert_mlx_adapter_to_peft(
        adapter_path,
        peft_dir,
        base_repo=base["repo_id"],
        base_revision=base["revision"],
    )
    equivalence = verify_adapter_equivalence(adapter_path, peft_dir)
    merged_dir = artifact_dir / "exports" / "Charlie-Alpha-4B-HF-BF16"
    if not (merged_dir / "config.json").exists():
        _merge_hf(config, peft_dir, merged_dir)
    llama_dir = _prepare_llama_cpp(config)
    gguf_dir = artifact_dir / "exports" / "Charlie-Alpha-4B-GGUF"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    bf16_path = gguf_dir / "Charlie-Alpha-4B-BF16.gguf"
    q4_path = gguf_dir / "Charlie-Alpha-4B-Q4_K_M.gguf"
    if not bf16_path.exists():
        _run(
            [
                sys.executable,
                str(llama_dir / "convert_hf_to_gguf.py"),
                str(merged_dir),
                "--outfile",
                str(bf16_path),
                "--outtype",
                "bf16",
            ],
            cwd=llama_dir,
            timeout=3600,
        )
    if not q4_path.exists():
        _run(
            [
                str(llama_dir / "build" / "bin" / "llama-quantize"),
                str(bf16_path),
                str(q4_path),
                config.section("release")["gguf_quantization"],
            ],
            cwd=llama_dir,
            timeout=3600,
        )
    result = {
        "mapping": mapping,
        "numeric_equivalence": equivalence,
        "gguf": q4_path.name,
        "gguf_sha256": sha256_file(q4_path),
        "llama_cpp_revision": config.sources["tools"]["llama_cpp"]["revision"],
        "behavioral_parity_pending": True,
    }
    write_json(config.root / "reports" / "gguf-export.json", result)
    return result


def export_all(config: ProjectConfig, include_gguf: bool = False) -> dict[str, Any]:
    adapter_path = _selected_adapter(config)
    model_path = _base_snapshot(config)
    package = _package_adapter(config, adapter_path)
    adapter_smoke = _smoke_load(model_path, str(adapter_path))
    fused_path = _fuse_mlx(config, adapter_path, model_path)
    fused_smoke = _smoke_load(str(fused_path))
    gguf: dict[str, Any] | None = None
    if include_gguf:
        gguf = export_gguf(config, adapter_path)
    result = {
        "profile": config.section("project")["profile"],
        "adapter": {
            "archive_name": Path(package["archive"]).name,
            "archive_sha256": package["archive_sha256"],
            "files": package["files"],
            "load_test": adapter_smoke,
        },
        "fused_mlx": {
            "directory_name": fused_path.name,
            "load_test": fused_smoke,
        },
        "gguf": gguf or {"status": "deferred-by-overnight-profile"},
        "clean_environment_validation_pending": True,
    }
    write_json(config.root / "reports" / "export.json", result)
    return result


def validate_clean_environment(config: ProjectConfig) -> dict[str, Any]:
    adapter_path = _selected_adapter(config)
    fused_path = config.path_for("artifact_dir") / "exports" / "Charlie-Alpha-4B-MLX-4bit"
    if not (fused_path / "config.json").exists():
        export_all(config)
    model_path = _base_snapshot(config)
    fingerprint = canonical_hash(
        {
            "adapter": sha256_file(adapter_path / "adapters.safetensors"),
            "fused_config": sha256_file(fused_path / "config.json"),
            "mlx": "0.32.1",
            "mlx_lm": "0.31.3",
            "transformers": "5.15.1",
        }
    )
    environment_dir = config.path_for("artifact_dir") / f"clean-env-{fingerprint[:12]}"
    python = environment_dir / "bin" / "python"
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required for clean-environment validation")
    if not python.exists():
        _run([uv, "venv", str(environment_dir), "--python", "3.12"], cwd=config.root, timeout=300)
    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "mlx==0.32.1",
            "mlx-lm[train]==0.31.3",
            "transformers==5.15.1",
        ],
        cwd=config.root,
        timeout=900,
    )
    script = f"""import gc
import json
import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

def check(model_path, adapter_path=None):
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    prompt = tokenizer.apply_chat_template(
        [{{"role": "user", "content": "Return only 2"}}],
        tokenize=False,
        add_generation_prompt=True,
    )
    output = generate(
        model,
        tokenizer,
        prompt,
        max_tokens=4,
        sampler=make_sampler(temp=0.0),
        verbose=False,
    )
    return bool(output.strip()), model

adapter_ok, model = check({model_path!r}, {str(adapter_path)!r})
del model
gc.collect()
mx.clear_cache()
fused_ok, model = check({str(fused_path)!r})
print(json.dumps({{"adapter_loaded": adapter_ok, "fused_loaded": fused_ok}}))
"""
    result = subprocess.run(
        ["/usr/bin/caffeinate", "-dimsu", str(python), "-c", script],
        cwd=config.root,
        text=True,
        capture_output=True,
        timeout=1200,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Clean load validation failed: {result.stderr[-1000:]}")
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    passed = bool(payload.get("adapter_loaded") and payload.get("fused_loaded"))
    validation = {
        "fingerprint": fingerprint,
        "passed": passed,
        "adapter_loaded": bool(payload.get("adapter_loaded")),
        "fused_loaded": bool(payload.get("fused_loaded")),
        "versions": {"mlx": "0.32.1", "mlx_lm": "0.31.3", "transformers": "5.15.1"},
    }
    write_json(config.root / "reports" / "clean-load.json", validation)
    export_report_path = config.root / "reports" / "export.json"
    export_report = json.loads(export_report_path.read_text(encoding="utf-8"))
    export_report["clean_environment_validation_pending"] = not passed
    export_report["clean_environment_load"] = validation
    write_json(export_report_path, export_report)
    return validation

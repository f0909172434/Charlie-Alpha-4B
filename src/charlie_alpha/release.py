from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from .config import ProjectConfig
from .io_utils import read_jsonl, write_json
from .sandbox import sandbox_self_test
from .validators import is_contaminated, validate_chat_record, word_ngrams

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _source_lock_gate(config: ProjectConfig) -> dict[str, Any]:
    failures: list[str] = []
    for kind in ("models", "datasets", "tools"):
        for name, source in config.sources.get(kind, {}).items():
            revision = source.get("revision")
            if not isinstance(revision, str) or not _SHA_RE.match(revision):
                failures.append(f"{kind}.{name} has an invalid revision")
            if not source.get("license"):
                failures.append(f"{kind}.{name} has no license")
    for name, artifact in config.sources.get("evaluation_artifacts", {}).items():
        if not _SHA256_RE.match(str(artifact.get("content_sha256", ""))):
            failures.append(f"evaluation_artifacts.{name} has an invalid content SHA-256")
        for field in ("version", "url", "license"):
            if not artifact.get(field):
                failures.append(f"evaluation_artifacts.{name} has no {field}")
    return {"passed": not failures, "failures": failures}


def _data_gate(config: ProjectConfig) -> dict[str, Any]:
    final_dir = config.path_for("final_dir")
    failures: list[str] = []
    group_splits: dict[str, set[str]] = defaultdict(set)
    training_prompt_references: list[set[tuple[str, ...]]] = []
    counts = 0
    for split in ("train", "valid", "test"):
        path = final_dir / f"{split}.jsonl"
        if not path.exists():
            failures.append(f"missing {path.name}")
            continue
        for row in read_jsonl(path):
            counts += 1
            failures.extend(f"{path.name}: {error}" for error in validate_chat_record(row))
            metadata = row["metadata"]
            parent = metadata.get("parent_prompt_sha256") or metadata["prompt_sha256"]
            group_splits[f"{metadata['source_repo']}:{metadata['source_id']}:{parent}"].add(split)
            if split == "train":
                prompt_ngrams = word_ngrams(row["messages"][0]["content"], 8)
                if prompt_ngrams:
                    training_prompt_references.append(prompt_ngrams)
    leaked = [group for group, splits in group_splits.items() if len(splits) > 1]
    if leaked:
        failures.append(f"{len(leaked)} source problem groups cross splits")
    for canary in read_jsonl(config.root / "configs" / "retention_canary.jsonl"):
        if is_contaminated(
            canary["prompt"],
            training_prompt_references,
            size=8,
            threshold=0.5,
        ):
            failures.append(f"retention canary overlaps training data: {canary['task_id']}")
    return {"passed": counts > 0 and not failures, "records": counts, "failures": failures[:20]}


def _tracked_content_gate(config: ProjectConfig) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=config.root,
        text=True,
        capture_output=True,
        check=False,
    )
    tracked = [line for line in result.stdout.splitlines() if line]
    forbidden_prefixes = (
        ".env",
        ".venv/",
        "artifacts/",
        "models/",
        "data/processed/",
        "data/distilled/",
        "data/final/",
    )
    forbidden = [path for path in tracked if path == ".env" or path.startswith(forbidden_prefixes)]
    private_home = str(Path.home())
    for tracked_path in tracked:
        path = config.root / tracked_path
        if not path.is_file() or path.suffix.lower() not in {
            ".json",
            ".yaml",
            ".yml",
            ".md",
            ".txt",
            ".py",
            ".toml",
        }:
            continue
        if private_home in path.read_text(encoding="utf-8", errors="replace"):
            forbidden.append(f"machine home path in {tracked_path}")
    return {"passed": not forbidden, "forbidden_tracked_files": forbidden}


def _artifact_privacy_gate(config: ProjectConfig) -> dict[str, Any]:
    artifact_dir = config.path_for("artifact_dir")
    roots = [
        artifact_dir / "release",
        artifact_dir / "exports" / "Charlie-Alpha-4B-MLX-4bit",
        config.root / "reports",
    ]
    suffixes = {".json", ".yaml", ".yml", ".md", ".txt"}
    credential_pattern = re.compile(
        r"(?:hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_|sk-[A-Za-z0-9]{20,})"
    )
    failures: list[str] = []
    private_home = str(Path.home())
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            if private_home in content:
                failures.append(f"absolute user path in {path.relative_to(config.root)}")
            if credential_pattern.search(content):
                failures.append(f"credential-like value in {path.relative_to(config.root)}")
    return {"passed": not failures, "failures": failures}


def check_release(config: ProjectConfig) -> dict[str, Any]:
    reports_dir = config.root / "reports"
    evaluation_path = reports_dir / "evaluation.json"
    export_path = reports_dir / "export.json"
    evaluation = (
        json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation_path.exists()
        else None
    )
    export = json.loads(export_path.read_text(encoding="utf-8")) if export_path.exists() else None
    gates = {
        "source_locks_and_licenses": _source_lock_gate(config),
        "data_schema_and_split_isolation": _data_gate(config),
        "tracked_content": _tracked_content_gate(config),
        "artifact_privacy": _artifact_privacy_gate(config),
        "sandbox": sandbox_self_test(),
        "evaluation_present": {"passed": evaluation is not None},
        "export_present": {"passed": export is not None},
        "adapter_load": {"passed": bool(export and export["adapter"]["load_test"].get("loaded"))},
        "fused_mlx_load": {
            "passed": bool(export and export["fused_mlx"]["load_test"].get("loaded"))
        },
        "clean_environment_load": {
            "passed": bool(export and not export.get("clean_environment_validation_pending", True))
        },
    }
    hard_gate_names = (
        "source_locks_and_licenses",
        "data_schema_and_split_isolation",
        "tracked_content",
        "artifact_privacy",
        "sandbox",
        "adapter_load",
        "fused_mlx_load",
        "clean_environment_load",
    )
    hard_pass = all(gates[name]["passed"] for name in hard_gate_names)
    if not hard_pass:
        classification = "blocked"
    elif evaluation and evaluation["quality_classification"] == "stable-candidate":
        classification = "v0.1.0"
    else:
        classification = "Experimental v0.1.0"
    report = {
        "classification": classification,
        "weights_publishable": hard_pass,
        "gates": gates,
        "gguf_publishable": bool(
            hard_pass
            and export
            and export.get("gguf", {}).get("behavioral_parity_pending") is False
        ),
    }
    write_json(reports_dir / "release-gate.json", report)
    return report


def publish_hugging_face(config: ProjectConfig, include_gguf: bool = False) -> dict[str, Any]:
    gate = check_release(config)
    if not gate["weights_publishable"]:
        raise RuntimeError(
            "Release gate blocks weight publication; inspect reports/release-gate.json"
        )
    if include_gguf and not gate["gguf_publishable"]:
        raise RuntimeError("GGUF parity gate has not passed")

    try:
        api = HfApi()
        identity = api.whoami()
    except Exception as error:
        raise RuntimeError(
            "Hugging Face authentication is required. The user must run `hf auth login` "
            "interactively; this project never handles tokens."
        ) from error
    account = identity.get("name") or identity.get("fullname")
    if not account:
        raise RuntimeError("Could not determine the authenticated Hugging Face account name")

    artifact_dir = config.path_for("artifact_dir") / "exports"
    mlx_directory = artifact_dir / "Charlie-Alpha-4B-MLX-4bit"
    if not (mlx_directory / "config.json").exists():
        raise RuntimeError("Fused MLX export is missing; run `make export` first")
    mlx_repo = f"{account}/{config.section('release')['hf_mlx_slug']}"
    api.create_repo(mlx_repo, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=mlx_repo,
        repo_type="model",
        folder_path=mlx_directory,
        commit_message=f"Release {config.section('release')['tag']}",
    )

    result: dict[str, Any] = {"mlx_repo": mlx_repo, "gguf_repo": None}
    if include_gguf:
        gguf_directory = artifact_dir / "Charlie-Alpha-4B-GGUF"
        gguf_repo = f"{account}/{config.section('release')['hf_gguf_slug']}"
        api.create_repo(gguf_repo, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(
            repo_id=gguf_repo,
            repo_type="model",
            folder_path=gguf_directory,
            commit_message=f"Release {config.section('release')['tag']}",
        )
        result["gguf_repo"] = gguf_repo
    write_json(config.root / "reports" / "huggingface-release.json", result)
    return result

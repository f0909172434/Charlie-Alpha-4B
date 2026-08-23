from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from .config import ProjectConfig
from .io_utils import read_jsonl, sha256_file, write_json
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
    train_language_tokens: dict[str, int] = defaultdict(int)
    train_domain_tokens: dict[str, int] = defaultdict(int)
    train_code_language_tokens: dict[str, int] = defaultdict(int)
    locked_training_revisions = {
        config.sources["datasets"]["math_train"]["revision"],
        config.sources["datasets"]["code_train"]["revision"],
    }
    max_repeats = int(config.section("data")["max_train_repeats_per_chinese_record"])
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
            if metadata.get("source_revision") not in locked_training_revisions:
                failures.append(f"{path.name}: source revision is not locked")
            repeat_index = int(metadata.get("sampling_repeat_index", 0))
            if repeat_index < 0 or repeat_index >= max_repeats:
                failures.append(f"{path.name}: sampling repeat index exceeds the configured cap")
            if split != "train" and repeat_index != 0:
                failures.append(f"{path.name}: validation/test records must not be repeated")
            if metadata.get("domain") == "code" and not metadata.get("verification", {}).get(
                "passed"
            ):
                failures.append(f"{path.name}: code record lacks a passing sandbox verification")
            if split == "train":
                assistant_tokens = int(metadata.get("assistant_token_count", 0))
                train_language_tokens[str(metadata.get("language"))] += assistant_tokens
                train_domain_tokens[str(metadata.get("domain"))] += assistant_tokens
                if metadata.get("domain") == "code":
                    train_code_language_tokens[str(metadata.get("code_language"))] += (
                        assistant_tokens
                    )
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

    def check_ratios(
        actual: dict[str, int], targets: dict[str, float], tolerance: float, label: str
    ) -> None:
        total = sum(actual.values())
        if total <= 0:
            failures.append(f"{label}: no assistant tokens")
            return
        for key, target in targets.items():
            ratio = actual.get(key, 0) / total
            if abs(ratio - float(target)) > tolerance:
                failures.append(
                    f"{label}.{key} ratio {ratio:.4f} differs from target {float(target):.4f}"
                )

    data_settings = config.section("data")
    check_ratios(train_language_tokens, data_settings["language_token_ratios"], 0.015, "language")
    check_ratios(train_domain_tokens, data_settings["domain_token_ratios"], 0.015, "domain")
    check_ratios(
        train_code_language_tokens,
        data_settings["code_language_ratios"],
        0.02,
        "code_language",
    )
    data_summary_path = config.root / "data" / "manifests" / "data-summary.json"
    if not data_summary_path.exists():
        failures.append("missing data-summary.json")
    else:
        data_summary = json.loads(data_summary_path.read_text(encoding="utf-8"))
        if int(data_summary.get("benchmark_reference_count", 0)) <= 0:
            failures.append("data preparation has no benchmark decontamination references")
    return {"passed": counts > 0 and not failures, "records": counts, "failures": failures[:20]}


def _forge_data_gate(config: ProjectConfig) -> dict[str, Any]:
    final_dir = config.path_for("final_dir")
    failures: list[str] = []
    done_path = final_dir / ".done.json"
    if not done_path.exists():
        return {"passed": False, "records": 0, "failures": ["missing Forge data manifest"]}
    manifest = json.loads(done_path.read_text(encoding="utf-8"))
    paths = {split: final_dir / f"{split}.jsonl" for split in ("train", "valid")}
    for split, path in paths.items():
        if not path.exists():
            failures.append(f"missing {path.name}")
        elif manifest.get(f"{split}_sha256") != sha256_file(path):
            failures.append(f"{path.name} hash changed")
    if failures:
        return {"passed": False, "records": 0, "failures": failures}

    maximum = int(config.section("forge")["max_seq_length"])
    group_size = int(config.section("training_v2")["grad_accumulation_steps"])
    source_ids: dict[str, set[str]] = {"train": set(), "valid": set()}
    groups: dict[str, list[int]] = defaultdict(list)
    language_weight: dict[str, float] = defaultdict(float)
    category_weight: dict[str, float] = defaultdict(float)
    counts = 0
    for split, path in paths.items():
        for row in read_jsonl(path):
            counts += 1
            failures.extend(f"{path.name}: {error}" for error in validate_chat_record(row))
            metadata = row["metadata"]
            candidate_id = str(metadata.get("candidate_id", ""))
            source_ids[split].add(candidate_id)
            if int(metadata.get("token_count_qwen35", maximum + 1)) > maximum:
                failures.append(f"{path.name}: record exceeds the token limit")
            if metadata.get("source_revision") not in {
                config.sources["datasets"]["math_train"]["revision"],
                config.sources["datasets"]["code_train"]["revision"],
            }:
                failures.append(f"{path.name}: source revision is not locked")
            if metadata.get("domain") == "code" and not metadata.get(
                "verification", {}
            ).get("passed"):
                failures.append(f"{path.name}: code record lacks sandbox verification")
            if split == "train":
                group = str(metadata.get("semantic_group_id", ""))
                groups[group].append(int(metadata.get("microstep_slot", -1)))
                weight = float(metadata.get("loss_weight", 0.0))
                language_weight[str(metadata.get("language"))] += weight
                category_weight[str(metadata.get("category"))] += weight
                selective = metadata.get("selective_target_indices")
                if metadata.get("language") == "en" and not selective:
                    failures.append("English Forge row lacks its selective token mask")
                if metadata.get("language") != "en" and selective is not None:
                    failures.append("Chinese Forge row must use its complete assistant target")
    if source_ids["train"] & source_ids["valid"]:
        failures.append("Forge source candidates cross train and validation splits")
    for group, slots in groups.items():
        if len(slots) != group_size or sorted(slots) != list(range(group_size)):
            failures.append(f"incomplete semantic group: {group}")

    def check_weight_ratios(
        actual: dict[str, float], targets: dict[str, float], label: str
    ) -> None:
        total = sum(actual.values())
        for key, target in targets.items():
            observed = actual.get(key, 0.0) / total if total else 0.0
            if abs(observed - float(target)) > 1e-5:
                failures.append(f"{label}.{key} weight ratio is {observed:.6f}")

    check_weight_ratios(
        language_weight, config.section("forge")["language_gradient_ratios"], "language"
    )
    check_weight_ratios(
        category_weight, config.section("forge")["group_categories"], "category"
    )

    eval_lock = json.loads(config.path_for("eval_lock").read_text(encoding="utf-8"))
    dev_ids = {row["task_id"] for row in eval_lock["suites"]["dev"]}
    final_ids = {row["task_id"] for row in eval_lock["suites"]["final"]}
    if dev_ids & final_ids:
        failures.append("development and final evaluation locks overlap")
    return {
        "passed": counts > 0 and not failures,
        "records": counts,
        "semantic_groups": len(groups),
        "failures": failures[:30],
    }


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
        ".venv/",
        "artifacts/",
        "models/",
        "data/processed/",
        "data/distilled/",
        "data/final/",
        "data/v2/",
        "reports/v2/generated/",
    )
    forbidden = [
        path
        for path in tracked
        if path == ".env"
        or (path.startswith(".env.") and path != ".env.example")
        or path.startswith(forbidden_prefixes)
    ]
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
    report_root = (
        config.path_for("report_dir")
        if config.section("project").get("profile") == "forge-overnight"
        else config.root / "reports"
    )
    roots = [
        artifact_dir / "release",
        artifact_dir / "exports" / "Charlie-Alpha-4B-MLX-4bit",
        report_root,
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
    forge = config.section("project").get("profile") == "forge-overnight"
    reports_dir = config.path_for("report_dir") if forge else config.root / "reports"
    evaluation_path = (
        reports_dir / "final" / "comparison.json"
        if forge
        else reports_dir / "evaluation.json"
    )
    export_path = reports_dir / "export.json"
    clean_load_path = reports_dir / "clean-load.json"
    evaluation = (
        json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation_path.exists()
        else None
    )
    export = json.loads(export_path.read_text(encoding="utf-8")) if export_path.exists() else None
    clean_load = (
        json.loads(clean_load_path.read_text(encoding="utf-8"))
        if clean_load_path.exists()
        else None
    )
    gates = {
        "source_locks_and_licenses": _source_lock_gate(config),
        "data_schema_and_split_isolation": (
            _forge_data_gate(config) if forge else _data_gate(config)
        ),
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
            "passed": bool(clean_load and clean_load.get("passed"))
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
    elif forge and evaluation and evaluation.get("gate_passed"):
        classification = config.section("release")["tag"]
    elif not forge and evaluation and evaluation["quality_classification"] == "stable-candidate":
        classification = "v0.1.0"
    else:
        classification = f"Experimental {config.section('release')['tag']}"
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
    forge = config.section("project").get("profile") == "forge-overnight"
    reports_dir = config.path_for("report_dir") if forge else config.root / "reports"
    write_json(reports_dir / "huggingface-release.json", result)
    return result

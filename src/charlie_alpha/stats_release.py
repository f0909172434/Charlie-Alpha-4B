from __future__ import annotations

import gc
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import mlx.core as mx
from huggingface_hub import HfApi, hf_hub_download
from mlx_lm import load

from .config import ProjectConfig
from .exporting import export_all, export_gguf, validate_clean_environment
from .io_utils import read_jsonl, sha256_file, write_json
from .release import _artifact_privacy_gate, _source_lock_gate, _tracked_content_gate
from .routed_inference import DynamicLoraRouter, resolve_adapter_path
from .stats_catalog import validate_catalog
from .stats_data import _validate_record
from .stats_sandbox import sandbox_self_test
from .stats_training import _stats_snapshot

_GGUF_ISSUES = (27019, 24737)


def _runtime_paths(config: ProjectConfig) -> tuple[Path, Path | None]:
    binary = config.root / ".pixi" / "envs" / "default" / "bin"
    python = binary / "python"
    r = binary.parent / "lib" / "R" / "bin" / "exec" / "R"
    return python, r if r.exists() else None


def verify_stats_router(config: ProjectConfig) -> dict[str, Any]:
    adapter_path = resolve_adapter_path(config)
    model_path = _stats_snapshot(config)
    model, tokenizer = load(
        model_path,
        adapter_path=str(adapter_path),
        tokenizer_config={"trust_remote_code": True},
    )
    router = DynamicLoraRouter(model)
    tokens = mx.array([tokenizer.encode("statistical interval")])
    router.set_route("adapter")
    adapter_logits = model(tokens)[:, -1, :]
    mx.eval(adapter_logits)
    router.set_route("base")
    bypass_logits = model(tokens)[:, -1, :]
    mx.eval(bypass_logits)
    router.set_route("adapter")
    restored_logits = model(tokens)[:, -1, :]
    mx.eval(restored_logits)
    adapter_effect = float(mx.max(mx.abs(adapter_logits - bypass_logits)).item())
    restore_error = float(mx.max(mx.abs(adapter_logits - restored_logits)).item())
    module_count = router.module_count
    parameter_count = router.adapter_parameter_count
    module_names = router.module_names
    del model, router
    gc.collect()
    mx.clear_cache()

    base_model, _ = load(model_path, tokenizer_config={"trust_remote_code": True})
    base_logits = base_model(tokens)[:, -1, :]
    mx.eval(base_logits)
    bypass_error = float(mx.max(mx.abs(base_logits - bypass_logits)).item())
    del base_model
    gc.collect()
    mx.clear_cache()
    result = {
        "schema_version": 1,
        "passed": adapter_effect > 0 and restore_error == 0 and bypass_error == 0,
        "routes": ["base", "stats"],
        "lora_modules": module_count,
        "adapter_parameters": parameter_count,
        "adapter_sha256": sha256_file(adapter_path / "adapters.safetensors"),
        "adapter_vs_base_max_abs_logit": adapter_effect,
        "restored_adapter_max_abs_logit_error": restore_error,
        "bypass_vs_independent_base_max_abs_logit_error": bypass_error,
        "module_names": module_names,
    }
    write_json(config.path_for("report_dir") / "router-equivalence.json", result)
    return result


def _github_issue(number: int) -> dict[str, Any]:
    url = f"https://api.github.com/repos/ggml-org/llama.cpp/issues/{number}"
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
            payload = json.loads(response.read())
        return {
            "number": number,
            "state": payload.get("state"),
            "closed_at": payload.get("closed_at"),
            "url": payload.get("html_url"),
        }
    except Exception as error:
        return {"number": number, "state": "unverified", "error": str(error)}


def gguf_upstream_gate(config: ProjectConfig) -> dict[str, Any]:
    issues = [_github_issue(number) for number in _GGUF_ISSUES]
    llama = config.sources["tools"]["llama_cpp"]
    resolved = all(issue.get("state") == "closed" and issue.get("closed_at") for issue in issues)
    result = {
        "passed": resolved,
        "status": "eligible-for-conversion-tests" if resolved else "withheld-upstream-unverified",
        "llama_cpp_revision": llama["revision"],
        "issues": issues,
        "required_post_conversion_tests": [
            "adapter_matrix_equivalence",
            "clean_load",
            "non_garbled_generation",
            "30_item_closed_book_parity_within_2_points",
        ],
    }
    write_json(config.path_for("report_dir") / "gguf-upstream-gate.json", result)
    return result


def export_stats(config: ProjectConfig, *, include_gguf: bool = False) -> dict[str, Any]:
    exported = export_all(config, include_gguf=False)
    router = verify_stats_router(config)
    clean = validate_clean_environment(config)
    upstream = gguf_upstream_gate(config)
    gguf: dict[str, Any] = {
        "status": upstream["status"],
        "published": False,
    }
    if include_gguf:
        if not upstream["passed"]:
            raise RuntimeError(
                "GGUF is withheld because the pinned llama.cpp path has no verified Qwen3.5 fix"
            )
        gguf = export_gguf(config)
    exported.update(
        {
            "canonical_artifact": "dynamic MLX adapter with base/stats routing",
            "router_equivalence": router,
            "clean_environment_load": clean,
            "clean_environment_validation_pending": not clean["passed"],
            "gguf": gguf,
        }
    )
    write_json(config.path_for("report_dir") / "export.json", exported)
    return exported


def _stats_data_gate(config: ProjectConfig) -> dict[str, Any]:
    manifest_path = config.path_for("final_dir") / "manifest.json"
    failures: list[str] = []
    if not manifest_path.exists():
        return {"passed": False, "failures": ["missing stats data manifest"], "records": 0}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = 0
    max_seq_length = int(config.section("stats_training")["max_seq_length"])
    split_ids: dict[str, dict[str, set[str]]] = {}
    for variant in ("hard-label", "regret-random", "dgp-regret"):
        for split, expected_count in (("train", 960), ("valid", 120)):
            relative = f"{variant}/{split}.jsonl"
            path = config.path_for("final_dir") / relative
            if not path.exists():
                failures.append(f"missing {relative}")
                continue
            if manifest.get("files", {}).get(relative) != sha256_file(path):
                failures.append(f"hash changed for {relative}")
            rows = list(read_jsonl(path))
            records += len(rows)
            if len(rows) != expected_count:
                failures.append(f"{relative} has {len(rows)} rows, expected {expected_count}")
            for row in rows:
                failures.extend(f"{relative}: {error}" for error in _validate_record(row))
                if (
                    int(row.get("metadata", {}).get("token_count_qwen35", 10**9))
                    > max_seq_length
                ):
                    failures.append(
                        f"{relative}: row exceeds the {max_seq_length}-token training limit"
                    )
            split_ids.setdefault(variant, {})[split] = {
                str(row["metadata"]["blueprint_id"]) for row in rows
            }
    all_train = set().union(
        *(values.get("train", set()) for values in split_ids.values())
    )
    all_valid = set().union(
        *(values.get("valid", set()) for values in split_ids.values())
    )
    if all_train & all_valid:
        failures.append("an ablation training blueprint overlaps validation")
    if manifest.get("variant_surfaces") != {
        "hard-label": "active-failure",
        "regret-random": "random-latin-hypercube",
        "dgp-regret": "active-failure",
    }:
        failures.append("ablation surface assignment changed")
    for variant, ratios in manifest.get("language_gradient_ratios", {}).items():
        for language, expected in {"en": 0.70, "zh_Hant": 0.15, "zh_Hans": 0.15}.items():
            if abs(float(ratios.get(language, -1)) - expected) > 1e-9:
                failures.append(f"{variant}.{language} gradient ratio changed")
    return {"passed": not failures and records > 0, "failures": failures[:40], "records": records}


def _evaluation_lock_gate(config: ProjectConfig) -> dict[str, Any]:
    path = config.path_for("eval_lock")
    if not path.exists():
        return {"passed": False, "failures": ["missing sealed evaluation lock"]}
    lock = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if not lock.get("sealed"):
        failures.append("evaluation lock is not sealed")
    expected = {"p_bench": 90, "statqa": 200, "final_dgp": 120}
    for key, count in expected.items():
        if int(lock.get(key, {}).get("count", -1)) != count:
            failures.append(f"{key} lock count changed")
    if len({item["category"] for item in lock.get("p_bench", {}).get("tasks", [])}) != 17:
        failures.append("P-Bench lock does not cover 17 categories")
    if not lock.get("decontamination", {}).get("passed"):
        failures.append("8-gram evaluation decontamination did not pass")
    return {"passed": not failures, "failures": failures, "sha256": sha256_file(path)}


def _public_isolation_report(result: dict[str, Any]) -> dict[str, Any]:
    """Keep release evidence without persisting sandbox streams or local paths."""
    public: dict[str, Any] = {"passed": bool(result.get("passed"))}
    if "error" in result:
        public["error"] = str(result["error"])
    sandbox_fields = (
        "returncode",
        "timed_out",
        "memory_exceeded",
        "write_exceeded",
        "output_exceeded",
        "isolated",
        "network_allowed",
        "elapsed_seconds",
        "written_bytes",
    )
    for runtime in ("python", "r"):
        value = result.get(runtime)
        if not isinstance(value, dict):
            continue
        sandbox = value.get("sandbox", {})
        public[runtime] = {
            "passed": bool(value.get("passed")),
            "checks": dict(value.get("checks", {})),
            "sandbox": {
                key: sandbox[key]
                for key in sandbox_fields
                if isinstance(sandbox, dict) and key in sandbox
            },
        }
    return public


def check_stats_release(config: ProjectConfig) -> dict[str, Any]:
    report_dir = config.path_for("report_dir")
    comparison_path = report_dir / "comparison.json"
    export_path = report_dir / "export.json"
    comparison = (
        json.loads(comparison_path.read_text(encoding="utf-8"))
        if comparison_path.exists()
        else None
    )
    exported = json.loads(export_path.read_text(encoding="utf-8")) if export_path.exists() else None
    python, r = _runtime_paths(config)
    isolation_raw = (
        sandbox_self_test(python_executable=python, r_executable=r)
        if python.exists()
        else {"passed": False, "error": "locked Pixi environment is missing"}
    )
    isolation = _public_isolation_report(isolation_raw)
    v2_tag = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", "refs/tags/v0.2.0"],
        cwd=config.root,
        text=True,
        capture_output=True,
        check=False,
    )
    gates = {
        "catalog": {"passed": not validate_catalog(), "failures": validate_catalog()},
        "source_locks_and_licenses": _source_lock_gate(config),
        "data_schema_and_split_isolation": _stats_data_gate(config),
        "sealed_evaluation_lock": _evaluation_lock_gate(config),
        "tracked_content": _tracked_content_gate(config),
        "artifact_privacy": _artifact_privacy_gate(config),
        "python_and_r_isolation": isolation,
        "evaluation_present": {"passed": comparison is not None},
        "export_present": {"passed": exported is not None},
        "adapter_load": {
            "passed": bool(
                exported
                and exported.get("adapter", {}).get("load_test", {}).get("loaded")
            )
        },
        "fused_mlx_load": {
            "passed": bool(
                exported and exported.get("fused_mlx", {}).get("load_test", {}).get("loaded")
            )
        },
        "clean_environment_load": {
            "passed": bool(
                exported and exported.get("clean_environment_load", {}).get("passed")
            )
        },
        "router_equivalence": {
            "passed": bool(exported and exported.get("router_equivalence", {}).get("passed"))
        },
        "v0_2_preserved": {"passed": v2_tag.returncode == 0},
    }
    hard_pass = all(value.get("passed", False) for value in gates.values())
    ability_pass = bool(comparison and comparison.get("ability_gates_passed"))
    tag = str(config.section("stats_release")["tag"])
    classification = tag if hard_pass and ability_pass else f"Experimental {tag}"
    if not hard_pass:
        classification = "blocked"
    report = {
        "classification": classification,
        "weights_publishable": hard_pass,
        "ability_gates_passed": ability_pass,
        "dgp_regret_claim_allowed": bool(
            comparison and comparison.get("dgp_regret_benefit_claim_allowed")
        ),
        "gates": gates,
        "gguf_publishable": bool(
            hard_pass
            and exported
            and exported.get("gguf", {}).get("behavioral_parity_pending") is False
        ),
    }
    write_json(report_dir / "release-gate.json", report)
    write_json(config.root / "reports" / "stats" / "release-gate.json", report)
    return report


def publish_stats_hugging_face(
    config: ProjectConfig,
    *,
    include_gguf: bool = False,
) -> dict[str, Any]:
    gate = check_stats_release(config)
    if not gate["weights_publishable"]:
        raise RuntimeError("Stats release gate blocks weight publication")
    if include_gguf and not gate["gguf_publishable"]:
        raise RuntimeError("GGUF parity gate has not passed")
    api = HfApi()
    try:
        identity = api.whoami()
    except Exception as error:
        raise RuntimeError(
            "Hugging Face login is required; authenticate interactively with `hf auth login`."
        ) from error
    account = str(identity.get("name") or "")
    if not account:
        raise RuntimeError("Could not determine the authenticated Hugging Face account")
    release = config.section("stats_release")
    repo_id = str(release["hf_mlx_repo"])
    if not repo_id.startswith(f"{account}/"):
        raise RuntimeError("The authenticated Hugging Face account does not own the target repo")
    package = config.path_for("artifact_dir") / "release" / "Charlie-Alpha-4B-MLX-adapter"
    adapter = package / "adapters.safetensors"
    if not adapter.exists():
        raise RuntimeError("Run stats export before Hugging Face publication")
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    refs = api.list_repo_refs(repo_id, repo_type="model")
    tag_names = {tag.name for tag in refs.tags}
    preserve = str(release["preserve_tag"])
    if preserve not in tag_names:
        api.create_tag(repo_id, tag=preserve, repo_type="model")
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=package,
        commit_message=f"Release {release['tag']}",
    )
    if str(release["tag"]) not in tag_names:
        api.create_tag(repo_id, tag=str(release["tag"]), revision=commit.oid, repo_type="model")
    downloaded = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename="adapters.safetensors",
            revision=commit.oid,
        )
    )
    if sha256_file(downloaded) != sha256_file(adapter):
        raise RuntimeError("Uploaded Hugging Face adapter failed SHA-256 verification")
    result = {
        "repo": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "revision": commit.oid,
        "tag": release["tag"],
        "preserved_tag": preserve,
        "adapter_sha256": sha256_file(adapter),
        "gguf_repo": None,
    }
    write_json(config.path_for("report_dir") / "huggingface-release.json", result)
    return result


def publish_stats_github(config: ProjectConfig) -> dict[str, Any]:
    gate = check_stats_release(config)
    if not gate["weights_publishable"]:
        raise RuntimeError("Stats release gate blocks GitHub publication")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=config.root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("Git working tree must be clean before GitHub publication")
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=config.root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if branch != "main":
        raise RuntimeError("GitHub publication is allowed only from main")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=config.root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    remote = subprocess.run(
        ["git", "ls-remote", "origin", "refs/heads/main"],
        cwd=config.root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.split()
    if not remote or remote[0] != commit:
        raise RuntimeError("Local main must be pushed to origin before creating the release")
    gh = shutil.which("gh") or str(Path.home() / ".local" / "bin" / "gh")
    if not Path(gh).exists():
        raise RuntimeError("GitHub CLI is required")
    release = config.section("stats_release")
    tag = str(release["tag"])
    repository = str(release["github_repo"])
    existing = subprocess.run(
        [gh, "release", "view", tag, "-R", repository, "--json", "url,tagName,name"],
        cwd=config.root,
        text=True,
        capture_output=True,
        check=False,
    )
    if existing.returncode == 0:
        return json.loads(existing.stdout)
    artifact_dir = config.path_for("artifact_dir")
    report_dir = config.path_for("report_dir")
    assets = [
        artifact_dir / "release" / "Charlie-Alpha-4B-MLX-adapter.tar.gz",
        config.path,
        report_dir / "comparison.json",
        report_dir / "release-gate.json",
        config.root / "configs" / "evaluation.stats.lock.json",
    ]
    missing = [str(path) for path in assets if not path.exists()]
    if missing:
        raise RuntimeError(f"GitHub release assets are missing: {missing}")
    checksums = {path.name: sha256_file(path) for path in assets}
    checksum_path = artifact_dir / "release" / "github-assets-sha256.json"
    write_json(checksum_path, checksums)
    assets.append(checksum_path)
    classification = str(gate["classification"])
    comparison = json.loads((report_dir / "comparison.json").read_text(encoding="utf-8"))
    metrics = comparison.get("metrics", {})
    notes = (
        f"Charlie alpha {classification}.\n\n"
        f"Final relative normalized-regret improvement: "
        f"{100 * float(metrics.get('regret_relative_improvement', 0.0)):.2f}%.\n"
        f"P-Bench Raw delta: {float(metrics.get('p_bench_raw_delta_points', 0.0)):+.2f} points.\n"
        f"StatQA delta: {float(metrics.get('statqa_delta_points', 0.0)):+.2f} points.\n\n"
        "See the attached frozen evaluation, release gates, provenance, and checksums. "
        "No training corpus, evaluation question text, credential, or machine path is included."
    )
    command = [
        gh,
        "release",
        "create",
        tag,
        "-R",
        repository,
        "--target",
        commit,
        "--title",
        classification,
        "--notes",
        notes,
        *(["--prerelease"] if classification.startswith("Experimental") else []),
        *[str(path) for path in assets],
    ]
    completed = subprocess.run(
        command,
        cwd=config.root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"GitHub release failed: {completed.stderr[-1000:]}")
    payload = json.loads(
        subprocess.run(
            [gh, "release", "view", tag, "-R", repository, "--json", "url,tagName,name"],
            cwd=config.root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    )
    write_json(report_dir / "github-release.json", payload)
    return payload

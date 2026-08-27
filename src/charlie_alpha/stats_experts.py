from __future__ import annotations

import copy
import gc
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm import load

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl
from .provenance import creation_surface_open_state
from .stats_calibrate import _surface_comparison
from .stats_catalog import FAMILIES
from .stats_cone import (
    _apply_flat_update,
    _cone_paths,
    _family_gradient_matrix,
)
from .stats_dgp import build_blueprints, simulate_scenario
from .stats_evolve import (
    _adapter_config_for_child,
    _proposal_records,
    _score_adapter_surfaces,
    _score_loaded_selector,
    _start_caffeinate,
    _surface,
)
from .stats_project import _diagnostic_groups, prepare_policy_projection_data
from .stats_route import (
    _aggregate_predictions,
    _family_metrics,
    _family_noninferior,
    _score_oracle_family_route,
)
from .stats_training import _enable_gradient_checkpointing_once, _stats_snapshot


def _expert_paths(config: ProjectConfig) -> tuple[Path, Path]:
    data_dir, cone_root = _cone_paths(config)
    return data_dir, cone_root / "family-experts"


def _expert_training_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Keep selection and confirmation policy out of the training fingerprint."""
    return {
        key: value
        for key, value in settings.items()
        if key not in {"selection", "confirmation_shard", "gates"}
    }


def _ensure_expert_surface_shard(
    config: ProjectConfig,
    root: Path,
    *,
    settings_key: str,
    stem: str,
    purpose: str,
    used_for_expert_selection: bool,
    single_use: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    settings = dict(config.section("family_experts")[settings_key])
    split = str(settings["split"])
    count = int(settings["count"])
    seed = int(settings["seed"])
    scenarios = build_blueprints({split: count}, seed=seed, active_search=False)
    simulation_settings = config.section("stats_data")
    fingerprint = canonical_hash(
        {
            "settings": settings,
            "scenarios": [scenario.to_dict() for scenario in scenarios],
            "simulation": {
                key: simulation_settings[key]
                for key in (
                    "initial_repetitions",
                    "escalation_repetitions",
                    "ranking_uncertainty_margin",
                    "regret_temperature",
                )
            },
            "generator_version": 1,
        }
    )
    path = root / f"{stem}.jsonl"
    manifest_path = root / f"{stem}-manifest.json"
    if path.exists() or manifest_path.exists():
        if not path.exists() or not manifest_path.exists():
            raise RuntimeError(f"The immutable family-expert {purpose} shard is incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("fingerprint") == fingerprint
            and manifest.get("sha256") == sha256_file(path)
            and int(manifest.get("count", 0)) == count
        ):
            return manifest, list(read_jsonl(path))
        raise RuntimeError(f"The family-expert {purpose} shard is immutable")
    simulations = [
        simulate_scenario(
            scenario,
            initial_repetitions=int(simulation_settings["initial_repetitions"]),
            escalation_repetitions=[
                int(value) for value in simulation_settings["escalation_repetitions"]
            ],
            uncertainty_margin=float(simulation_settings["ranking_uncertainty_margin"]),
            temperature=float(simulation_settings["regret_temperature"]),
        )
        for scenario in scenarios
    ]
    write_jsonl(path, simulations)
    manifest = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "purpose": purpose,
        "split": split,
        "seed": seed,
        "count": count,
        "sha256": sha256_file(path),
        "used_for_expert_selection": used_for_expert_selection,
        "used_for_training": False,
        "single_use": single_use,
        "immutable": True,
        **creation_surface_open_state(),
    }
    write_json(manifest_path, manifest)
    return manifest, simulations


def _family_selection_rows(
    config: ProjectConfig,
    family_id: str,
    dedicated_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    surfaces = [str(value) for value in config.section("family_experts")["selection_surfaces"]]
    rows = [
        row
        for surface_name in surfaces
        for row in _surface(config, surface_name)
        if str(row["scenario"]["family_id"]) == family_id
    ]
    rows.extend(row for row in dedicated_rows if str(row["scenario"]["family_id"]) == family_id)
    if not rows:
        raise RuntimeError(f"No reusable selection rows exist for DGP family {family_id}")
    if len({str(row["scenario"]["blueprint_id"]) for row in rows}) != len(rows):
        raise RuntimeError("Family-expert selection surfaces overlap by blueprint")
    return rows


def _score_loaded_family(
    model: Any,
    tokenizer: Any,
    simulations: list[dict[str, Any]],
) -> dict[str, Any]:
    languages = {
        language: _score_loaded_selector(
            model,
            tokenizer,
            _proposal_records(simulations, language=language, view=view),
        )
        for language, view in (
            ("en", "boundary_a"),
            ("zh_Hant", "standard"),
            ("zh_Hans", "standard"),
        )
    }
    return {"selector": languages["en"], "languages": languages, "retention": None}


def select_family_expert_checkpoint(
    options: dict[str, dict[str, Any]],
    *,
    gates: dict[str, Any],
    minimum_relative_improvement: float,
) -> dict[str, Any]:
    if "parent" not in options:
        raise ValueError("Family-expert selection requires the unchanged parent")
    parent = options["parent"]
    parent_metrics = parent["metrics"]
    eligible = [
        {"slug": slug, **value}
        for slug, value in options.items()
        if _family_noninferior(value["metrics"], parent_metrics, gates)
    ]
    selected = min(
        eligible,
        key=lambda value: (
            float(value["metrics"]["normalized_regret"]),
            float(value["metrics"]["invalid_selection_rate"]),
            -float(value["metrics"]["accuracy"]),
            0 if value["slug"] == "parent" else 1,
            int(value.get("update", 0)),
            str(value["slug"]),
        ),
    )
    parent_regret = float(parent_metrics["normalized_regret"])
    selected_regret = float(selected["metrics"]["normalized_regret"])
    relative = (parent_regret - selected_regret) / parent_regret if parent_regret else 0.0
    if selected["slug"] != "parent" and relative < minimum_relative_improvement:
        selected = {"slug": "parent", **parent}
        relative = 0.0
    return {
        **selected,
        "parent_metrics": parent_metrics,
        "relative_regret_improvement": relative,
    }


def run_family_expert_arm(
    config: ProjectConfig,
    *,
    family_id: str,
    force: bool = False,
) -> dict[str, Any]:
    family_ids = {family.family_id for family in FAMILIES}
    if family_id not in family_ids:
        raise ValueError(f"Unknown DGP family: {family_id}")
    settings = dict(config.section("family_experts"))
    training_settings = _expert_training_settings(settings)
    data_status = prepare_policy_projection_data(config, force=False)
    data_dir, artifact_root = _expert_paths(config)
    if force and (artifact_root / "confirmation-manifest.json").exists():
        raise RuntimeError("Cannot force-retrain family experts after confirmation was opened")
    artifact_dir = artifact_root / family_id
    train_path = data_dir / "train.jsonl"
    selection_manifest, dedicated_selection_rows = _ensure_expert_surface_shard(
        config,
        artifact_root,
        settings_key="selection_shard",
        stem="selection-surface",
        purpose="checkpoint-selection",
        used_for_expert_selection=True,
        single_use=False,
    )
    valid_paths = [
        config.path_for("stats_dir") / "surface" / f"{name}.jsonl"
        for name in settings["selection_surfaces"]
    ]
    fingerprint = canonical_hash(
        {
            "family_id": family_id,
            "settings": training_settings,
            "parent": data_status["parent"]["adapter_sha256"],
            "train": sha256_file(train_path),
            "selection_surfaces": {path.name: sha256_file(path) for path in valid_paths},
            "dedicated_selection_shard": selection_manifest["fingerprint"],
            "trainer_version": 1,
        }
    )
    status_path = artifact_dir / "status.json"
    progress_path = artifact_dir / "progress.json"
    if status_path.exists() and not force:
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            return existing
        raise RuntimeError(
            f"Family-expert fingerprint changed for {family_id}; use --force to replace it"
        )

    resume: dict[str, Any] | None = None
    if progress_path.exists() and not force:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"Family-expert partial fingerprint changed for {family_id}; use --force"
            )
        checkpoint = Path(str(progress.get("last_checkpoint_path", "")))
        if int(progress.get("completed_updates", 0)) and (
            not checkpoint.exists()
            or sha256_file(checkpoint) != progress.get("last_checkpoint_sha256")
        ):
            raise RuntimeError(f"Family-expert resume checkpoint changed for {family_id}")
        resume = progress

    artifact_dir.mkdir(parents=True, exist_ok=True)
    if force or resume is None:
        for path in [
            artifact_dir / "adapter_config.json",
            status_path,
            progress_path,
            *artifact_dir.glob("update-*.safetensors"),
        ]:
            if path.exists():
                path.unlink()

    parent = Path(str(data_status["parent"]["adapter_path"]))
    parent_weights = parent / "adapters.safetensors"
    all_train_rows = list(read_jsonl(train_path))
    train_rows = [row for row in all_train_rows if str(row["metadata"]["family_id"]) == family_id]
    if not train_rows:
        raise RuntimeError(f"Projected training data has no rows for {family_id}")
    if {str(row["metadata"]["family_id"]) for row in train_rows} != {family_id}:
        raise RuntimeError("A family expert received cross-family training rows")
    groups = _diagnostic_groups(train_rows, groups_per_family=10**9)
    selection_rows = _family_selection_rows(config, family_id, dedicated_selection_rows)

    model = tokenizer = None
    caffeinate = _start_caffeinate()
    previous_cache_limit = mx.set_cache_limit(int(float(settings["cache_limit_gb"]) * 1024**3))
    started = time.monotonic()
    try:
        model, tokenizer = load(
            _stats_snapshot(config),
            adapter_path=str(parent),
            tokenizer_config={"trust_remote_code": True},
        )
        model.freeze()
        model.unfreeze(keys=["lora_a", "lora_b"])
        _enable_gradient_checkpointing_once(model)
        for _, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        model.eval()

        checkpoints: list[dict[str, Any]] = (
            [dict(value) for value in resume["checkpoints"]]
            if resume
            else [
                {
                    "name": "parent",
                    "path": str(parent_weights),
                    "sha256": sha256_file(parent_weights),
                    "score": _score_loaded_family(model, tokenizer, selection_rows),
                }
            ]
        )
        update_history = [dict(value) for value in resume["update_history"]] if resume else []
        if resume and int(resume.get("completed_updates", 0)):
            model.load_weights(str(resume["last_checkpoint_path"]), strict=False)
        parameter_names: list[str] | None = None
        first_update = int(resume.get("completed_updates", 0)) + 1 if resume else 1
        for update_index in range(first_update, int(settings["updates"]) + 1):
            model.train()
            for _, module in model.named_modules():
                if isinstance(module, nn.Dropout):
                    module.eval()
            gradients, family_names, names, diagnostics = _family_gradient_matrix(
                model,
                tokenizer,
                groups,
                records_per_backward=int(settings["records_per_backward"]),
                max_seq_length=int(config.section("stats_training")["max_seq_length"]),
            )
            if family_names != [family_id] or gradients.shape[0] != 1:
                raise RuntimeError("Family-expert gradient aggregation crossed family boundaries")
            if parameter_names is None:
                parameter_names = names
            elif parameter_names != names:
                raise RuntimeError("Family-expert parameter order changed between updates")
            applied = _apply_flat_update(
                model,
                gradients[0],
                parameter_names,
                step_l2=float(settings["step_l2"]),
            )
            checkpoint_path = artifact_dir / f"update-{update_index:02d}.safetensors"
            mx.save_safetensors(
                str(checkpoint_path),
                dict(tree_flatten(model.trainable_parameters())),
            )
            model.eval()
            score = _score_loaded_family(model, tokenizer, selection_rows)
            checkpoints.append(
                {
                    "name": f"update-{update_index:02d}",
                    "path": str(checkpoint_path),
                    "sha256": sha256_file(checkpoint_path),
                    "score": score,
                }
            )
            update_history.append(
                {
                    "update": update_index,
                    "gradient_diagnostics": diagnostics,
                    "update_norms": applied,
                    "selection": _family_metrics(score, family_id),
                }
            )
            write_json(
                progress_path,
                {
                    "schema_version": 1,
                    "fingerprint": fingerprint,
                    "complete": False,
                    "family_id": family_id,
                    "completed_updates": update_index,
                    "last_checkpoint_path": str(checkpoint_path),
                    "last_checkpoint_sha256": sha256_file(checkpoint_path),
                    "checkpoints": checkpoints,
                    "update_history": update_history,
                },
            )
            del gradients
            gc.collect()
            mx.clear_cache()

        adapter_config = _adapter_config_for_child(
            config,
            parent,
            artifact_dir,
            cycle=max(6, int(config.section("policy_projection")["source_cycle"]) + 1),
            arm=f"family-expert:{family_id}",
        )
        adapter_config.setdefault("stats", {}).update(
            {
                "method": "DGP Family Expert oracle pilot",
                "family_id": family_id,
                "dropout_disabled": True,
                "promotion_status": "diagnostic-only",
            }
        )
        write_json(artifact_dir / "adapter_config.json", adapter_config)
        status = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "complete": True,
            "family_id": family_id,
            "parent_adapter_path": str(parent),
            "parent_adapter_sha256": sha256_file(parent_weights),
            "source_train_sha256": sha256_file(train_path),
            "records": len(train_rows),
            "semantic_groups": len(groups),
            "updates": int(settings["updates"]),
            "backward_record_exposures": len(train_rows) * int(settings["updates"]),
            "dropout_disabled": True,
            "selection_surfaces": list(settings["selection_surfaces"]),
            "selection_dgps": len(selection_rows),
            "checkpoints": checkpoints,
            "update_history": update_history,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 4),
            "promotion_shard_opened": False,
            "sealed_final_surface_opened": False,
        }
        write_json(status_path, status)
        write_json(
            progress_path,
            {
                "schema_version": 1,
                "fingerprint": fingerprint,
                "complete": True,
                "family_id": family_id,
                "completed_updates": int(settings["updates"]),
                "last_checkpoint_path": str(
                    artifact_dir / f"update-{int(settings['updates']):02d}.safetensors"
                ),
                "last_checkpoint_sha256": sha256_file(
                    artifact_dir / f"update-{int(settings['updates']):02d}.safetensors"
                ),
                "status_sha256": sha256_file(status_path),
                "checkpoints": checkpoints,
                "update_history": update_history,
            },
        )
        return status
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        mx.clear_cache()
        mx.set_cache_limit(previous_cache_limit)
        if caffeinate is not None and caffeinate.poll() is None:
            caffeinate.terminate()


def _pooled_score(
    statuses: dict[str, dict[str, Any]],
    selected: dict[str, dict[str, Any]] | None,
    retention: dict[str, Any],
) -> dict[str, Any]:
    language_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for family_id, status in sorted(statuses.items()):
        checkpoint = (
            next(value for value in status["checkpoints"] if value["name"] == "parent")
            if selected is None
            else next(
                value
                for value in status["checkpoints"]
                if value["name"] == selected[family_id]["checkpoint_name"]
            )
        )
        for language, score in checkpoint["score"]["languages"].items():
            language_predictions[language].extend(score["predictions"])
    languages = {
        language: _aggregate_predictions(predictions)
        for language, predictions in sorted(language_predictions.items())
    }
    return {"selector": languages["en"], "languages": languages, "retention": retention}


def _ensure_expert_confirmation_shard(
    config: ProjectConfig,
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _ensure_expert_surface_shard(
        config,
        root,
        settings_key="confirmation_shard",
        stem="confirmation",
        purpose="oracle-confirmation",
        used_for_expert_selection=False,
        single_use=True,
    )


def _materialize_selected_adapters(
    root: Path,
    statuses: dict[str, dict[str, Any]],
    mapping: dict[str, dict[str, Any]],
) -> dict[str, Path]:
    adapters: dict[str, Path] = {
        "parent": Path(next(iter(statuses.values()))["parent_adapter_path"])
    }
    for family_id, route in sorted(mapping.items()):
        if route["checkpoint_name"] == "parent":
            continue
        slug = str(route["slug"])
        destination = root / "selected-adapters" / family_id
        destination.mkdir(parents=True, exist_ok=True)
        source = Path(str(route["checkpoint_path"]))
        target = destination / "adapters.safetensors"
        if target.exists() and sha256_file(target) != sha256_file(source):
            raise RuntimeError(f"Selected family adapter changed after selection: {family_id}")
        if not target.exists():
            shutil.copy2(source, target)
        # The parent checkpoint lives outside the expert directory, so use the
        # expert status directory for its adapter contract.
        source_config = (
            Path(str(statuses[family_id]["checkpoints"][-1]["path"])).parent / "adapter_config.json"
        )
        config_target = destination / "adapter_config.json"
        config_value = json.loads(source_config.read_text(encoding="utf-8"))
        config_value.setdefault("stats", {}).update(
            {
                "selected_family_checkpoint": route["checkpoint_name"],
                "oracle_route_only": True,
            }
        )
        write_json(config_target, config_value)
        adapters[slug] = destination
    return adapters


def _public_expert_report(report: dict[str, Any]) -> dict[str, Any]:
    public = copy.deepcopy(report)
    for route in public["selection"]["mapping"].values():
        route.pop("checkpoint_path", None)
        route.pop("checkpoint_sha256", None)
    return public


def run_family_expert_oracle(
    config: ProjectConfig,
    *,
    force: bool = False,
    train_only: bool = False,
) -> dict[str, Any]:
    settings = dict(config.section("family_experts"))
    _, artifact_root = _expert_paths(config)
    artifact_root.mkdir(parents=True, exist_ok=True)
    if force and (artifact_root / "confirmation-manifest.json").exists():
        raise RuntimeError("Cannot force-retrain family experts after confirmation was opened")
    family_ids = sorted(family.family_id for family in FAMILIES)
    statuses: dict[str, dict[str, Any]] = {}
    for index, family_id in enumerate(family_ids, start=1):
        print(
            f"Family expert {index}/{len(family_ids)}: {family_id}",
            file=sys.stderr,
            flush=True,
        )
        command = [
            sys.executable,
            "-m",
            "charlie_alpha.cli",
            "stats",
            "policy-family-expert-arm",
            "--config",
            str(config.path),
            "--family",
            family_id,
        ]
        if force:
            command.append("--force")
        subprocess.run(
            command,
            cwd=config.root,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        statuses[family_id] = json.loads(
            (artifact_root / family_id / "status.json").read_text(encoding="utf-8")
        )
    if train_only:
        return {
            "stage": "family-experts-trained",
            "families": family_ids,
            "promotion_shard_opened": False,
            "sealed_final_surface_opened": False,
        }

    parent_hashes = {status["parent_adapter_sha256"] for status in statuses.values()}
    train_hashes = {status["source_train_sha256"] for status in statuses.values()}
    if len(parent_hashes) != 1 or len(train_hashes) != 1:
        raise RuntimeError("Family experts do not share one parent and training surface")
    source_records = len(list(read_jsonl(_cone_paths(config)[0] / "train.jsonl")))
    total_exposures = sum(int(status["backward_record_exposures"]) for status in statuses.values())
    expected_exposures = source_records * int(settings["updates"])
    matched_backward_compute = total_exposures == expected_exposures
    if not matched_backward_compute:
        raise RuntimeError("Family-expert and shared-adapter backward record exposure differs")

    gates = dict(settings["gates"])
    minimum_family = float(settings["selection"]["minimum_family_relative_improvement"])
    mapping: dict[str, dict[str, Any]] = {}
    for family_id, status in sorted(statuses.items()):
        options: dict[str, dict[str, Any]] = {}
        for checkpoint in status["checkpoints"]:
            name = str(checkpoint["name"])
            slug = "parent" if name == "parent" else f"expert-{family_id}-{name}"
            options[slug] = {
                "checkpoint_name": name,
                "checkpoint_path": str(checkpoint["path"]),
                "checkpoint_sha256": str(checkpoint["sha256"]),
                "update": 0 if name == "parent" else int(name.rsplit("-", 1)[1]),
                "metrics": _family_metrics(checkpoint["score"], family_id),
            }
        selected = select_family_expert_checkpoint(
            options,
            gates=gates,
            minimum_relative_improvement=minimum_family,
        )
        mapping[family_id] = selected

    retention_status = json.loads(
        (_cone_paths(config)[1] / "delta-calibration" / "scale-0p00" / "status.json").read_text(
            encoding="utf-8"
        )
    )
    retention = retention_status["scores"]["valid"]["retention"]
    parent_selection = _pooled_score(statuses, None, retention)
    candidate_selection = _pooled_score(statuses, mapping, retention)
    selection_comparison = _surface_comparison(
        parent_selection,
        candidate_selection,
        gates,
    )
    selection_passed = bool(selection_comparison["all_gates_passed"]) and float(
        selection_comparison["trilingual_relative_regret_improvement"]
    ) >= float(settings["selection"]["minimum_route_relative_improvement"])
    nonparent_families = sorted(
        family_id for family_id, route in mapping.items() if route["checkpoint_name"] != "parent"
    )
    selection_fingerprint = canonical_hash(
        {
            "settings": settings,
            "experts": {
                family: status["fingerprint"] for family, status in sorted(statuses.items())
            },
            "mapping": {
                family: {
                    "checkpoint": route["checkpoint_name"],
                    "sha256": route["checkpoint_sha256"],
                }
                for family, route in sorted(mapping.items())
            },
            "selector_version": 1,
        }
    )
    confirmation_exists = (artifact_root / "confirmation-manifest.json").exists()
    selection = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": selection_fingerprint,
        "selection_surfaces": list(settings["selection_surfaces"]),
        "selection_rule": (
            "per-family parent-safe checkpoint with at least the registered regret gain, then "
            "pooled trilingual and granular route gates"
        ),
        "mapping": mapping,
        "nonparent_families": nonparent_families,
        "comparison": selection_comparison,
        "minimum_route_relative_improvement": float(
            settings["selection"]["minimum_route_relative_improvement"]
        ),
        "passed": selection_passed,
        "confirmation_shard_opened": confirmation_exists,
        "promotion_shard_opened": False,
        "sealed_final_surface_opened": False,
    }
    selection_path = artifact_root / "selection.json"
    if selection_path.exists() and confirmation_exists:
        existing = json.loads(selection_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != selection_fingerprint:
            raise RuntimeError("Family-expert selection changed after confirmation was opened")
    write_json(selection_path, selection)

    base_report = {
        "schema_version": 1,
        "complete": True,
        "method": "compute-matched DGP family-expert oracle route",
        "oracle_route_upper_bound": True,
        "mixture_of_lora_prior_art": True,
        "training_compute": {
            "shared_source_records": source_records,
            "updates": int(settings["updates"]),
            "expert_backward_record_exposures": total_exposures,
            "shared_control_backward_record_exposures": expected_exposures,
            "matched": matched_backward_compute,
        },
        "selection": selection,
        "promotion_shard_opened": False,
        "sealed_final_surface_opened": False,
        "claim_boundary": (
            "This uses true DGP family IDs and is only an oracle upper bound. It cannot be "
            "promoted or deployed until a learned router reproduces the gain on new data."
        ),
    }
    report_path = artifact_root / "report.json"
    public_path = config.root / "reports" / "evolve" / "family-experts.json"
    if report_path.exists() and not force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            existing.get("complete")
            and existing.get("selection", {}).get("fingerprint") == selection_fingerprint
        ):
            existing["selection"] = selection
            write_json(report_path, existing)
            write_json(public_path, _public_expert_report(existing))
            return existing
    if not selection_passed or not nonparent_families:
        report = {
            **base_report,
            "fingerprint": canonical_hash(
                {"selection": selection_fingerprint, "evaluator_version": 1}
            ),
            "confirmation": None,
            "proceed_to_router_implementation": False,
        }
        write_json(report_path, report)
        write_json(public_path, _public_expert_report(report))
        return report

    manifest, confirmation_rows = _ensure_expert_confirmation_shard(config, artifact_root)
    selection["confirmation_shard_opened"] = True
    write_json(selection_path, selection)
    adapters = _materialize_selected_adapters(artifact_root, statuses, mapping)
    parent_path = adapters["parent"]
    split = str(manifest["split"])
    parent_confirmation = _score_adapter_surfaces(
        config,
        parent_path,
        {split: confirmation_rows},
    )[split]
    routed_confirmation = _score_oracle_family_route(
        config,
        mapping={
            family: {"slug": "parent" if route["checkpoint_name"] == "parent" else route["slug"]}
            for family, route in mapping.items()
        },
        adapter_paths=adapters,
        surface=confirmation_rows,
        retention=parent_confirmation["retention"],
        split=split,
    )
    comparison = _surface_comparison(parent_confirmation, routed_confirmation, gates)
    confirmation_passed = bool(comparison["all_gates_passed"]) and float(
        comparison["trilingual_relative_regret_improvement"]
    ) >= float(settings["gates"]["minimum_confirmation_relative_improvement"])
    report = {
        **base_report,
        "fingerprint": canonical_hash(
            {
                "selection": selection_fingerprint,
                "confirmation": manifest["fingerprint"],
                "evaluator_version": 1,
            }
        ),
        "selection": selection,
        "confirmation": {
            "manifest": manifest,
            "comparison": comparison,
            "minimum_relative_improvement": float(
                settings["gates"]["minimum_confirmation_relative_improvement"]
            ),
            "passed": confirmation_passed,
        },
        "proceed_to_router_implementation": confirmation_passed,
    }
    write_json(report_path, report)
    write_json(public_path, _public_expert_report(report))
    return report

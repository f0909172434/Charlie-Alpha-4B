from __future__ import annotations

import gc
import itertools
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl
from .stats_cone import _cone_paths
from .stats_dgp import build_blueprints, simulate_scenario
from .stats_evolve import (
    _group_regret,
    _noninferior_mapping,
    _score_adapter_surfaces,
    _surface,
)


def _scale_slug(scale: float) -> str:
    return f"{scale:.2f}".replace(".", "p")


def interpolate_adapter_weights(
    parent_path: Path,
    candidate_path: Path,
    *,
    scale: float,
) -> dict[str, mx.array]:
    if not 0.0 <= scale <= 1.0:
        raise ValueError("Adapter interpolation scale must be in [0, 1]")
    parent = mx.load(str(parent_path))
    candidate = mx.load(str(candidate_path))
    if set(parent) != set(candidate):
        raise RuntimeError("Adapter interpolation requires identical tensor keys")
    interpolated: dict[str, mx.array] = {}
    for key in sorted(parent):
        if parent[key].shape != candidate[key].shape:
            raise RuntimeError(f"Adapter interpolation shape mismatch for {key}")
        if key.endswith(".lora_a"):
            interpolated[key] = mx.concatenate([parent[key], candidate[key]], axis=1)
        elif key.endswith(".lora_b"):
            interpolated[key] = mx.concatenate(
                [(1.0 - scale) * parent[key], scale * candidate[key]],
                axis=0,
            )
        else:
            raise RuntimeError(f"Effective adapter interpolation found a non-LoRA tensor: {key}")
    return interpolated


_LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")


def interpolate_adapter_blocks(
    parent_path: Path,
    candidate_path: Path,
    *,
    layer_scales: dict[int, float],
) -> dict[str, mx.array]:
    if not layer_scales:
        raise ValueError("Blockwise interpolation requires at least one registered layer")
    if any(not 0.0 <= float(scale) <= 1.0 for scale in layer_scales.values()):
        raise ValueError("Every blockwise interpolation scale must be in [0, 1]")
    parent = mx.load(str(parent_path))
    candidate = mx.load(str(candidate_path))
    if set(parent) != set(candidate):
        raise RuntimeError("Blockwise interpolation requires identical tensor keys")
    observed_layers: set[int] = set()
    interpolated: dict[str, mx.array] = {}
    for key in sorted(parent):
        if parent[key].shape != candidate[key].shape:
            raise RuntimeError(f"Adapter interpolation shape mismatch for {key}")
        match = _LAYER_PATTERN.search(key)
        if match is None:
            raise RuntimeError(f"Adapter tensor does not expose a transformer layer: {key}")
        layer = int(match.group(1))
        observed_layers.add(layer)
        scale = float(layer_scales.get(layer, 0.0))
        if key.endswith(".lora_a"):
            interpolated[key] = mx.concatenate([parent[key], candidate[key]], axis=1)
        elif key.endswith(".lora_b"):
            interpolated[key] = mx.concatenate(
                [(1.0 - scale) * parent[key], scale * candidate[key]],
                axis=0,
            )
        else:
            raise RuntimeError(f"Effective adapter interpolation found a non-LoRA tensor: {key}")
    unknown = set(layer_scales) - observed_layers
    if unknown:
        raise ValueError(f"Blockwise interpolation references absent layers: {sorted(unknown)}")
    return interpolated


def parse_layer_scales(value: str) -> dict[int, float]:
    result: dict[int, float] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        layer_text, separator, scale_text = item.partition("=")
        if not separator:
            raise ValueError("Layer scales must use LAYER=SCALE pairs")
        layer = int(layer_text)
        if layer in result:
            raise ValueError(f"Layer {layer} is repeated")
        result[layer] = float(scale_text)
    if not result:
        raise ValueError("At least one layer scale is required")
    return result


def _calibration_paths(config: ProjectConfig, scale: float) -> tuple[Path, Path]:
    _, cone_dir = _cone_paths(config)
    root = cone_dir / "delta-calibration"
    return root, root / f"scale-{_scale_slug(scale)}"


def _block_slug(layer_scales: dict[int, float]) -> str:
    return "_".join(
        f"l{layer}-{_scale_slug(float(scale))}" for layer, scale in sorted(layer_scales.items())
    )


def _block_paths(config: ProjectConfig, layer_scales: dict[int, float]) -> tuple[Path, Path]:
    _, cone_dir = _cone_paths(config)
    root = cone_dir / "block-projection"
    return root, root / _block_slug(layer_scales)


def _ensure_block_confirmation_shard(
    config: ProjectConfig,
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    settings = dict(config.section("block_projection")["confirmation_shard"])
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
    path = root / "confirmation.jsonl"
    manifest_path = root / "confirmation-manifest.json"
    if path.exists() or manifest_path.exists():
        if not path.exists() or not manifest_path.exists():
            raise RuntimeError("The immutable block confirmation shard is incomplete")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("fingerprint") == fingerprint
            and existing.get("sha256") == sha256_file(path)
            and int(existing.get("count", 0)) == count
        ):
            return existing, list(read_jsonl(path))
        raise RuntimeError("The block confirmation shard is immutable")
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
        "split": split,
        "seed": seed,
        "count": count,
        "sha256": sha256_file(path),
        "used_for_profile_selection": False,
        "used_for_training": False,
        "single_use": True,
        "sealed_at_preparation": True,
        "promotion_surface_opened": False,
        "final_surface_opened": False,
    }
    write_json(manifest_path, manifest)
    return manifest, simulations


def block_projection_profiles(settings: dict[str, Any]) -> list[dict[int, float]]:
    layers = tuple(int(value) for value in settings["layers"])
    if len(layers) != len(set(layers)) or not layers:
        raise ValueError("Block projection layers must be unique and nonempty")
    amplitudes = tuple(float(value) for value in settings["amplitudes"])
    if not amplitudes or any(not 0.0 < value <= 1.0 for value in amplitudes):
        raise ValueError("Block projection amplitudes must be in (0, 1]")
    profiles: list[dict[int, float]] = []
    for amplitude in amplitudes:
        for count in range(1, len(layers) + 1):
            for active in itertools.combinations(layers, count):
                selected = set(active)
                profiles.append(
                    {layer: amplitude if layer in selected else 0.0 for layer in layers}
                )
    return profiles


def run_delta_calibration_arm(
    config: ProjectConfig,
    *,
    scale: float,
    force: bool = False,
) -> dict[str, Any]:
    settings = dict(config.section("delta_calibration"))
    registered = {float(value) for value in [0.0, *settings["scales"]]}
    if scale not in registered:
        raise ValueError("The requested interpolation scale is not registered")
    _, cone_dir = _cone_paths(config)
    source_status_path = cone_dir / "uniform-family" / "status.json"
    if not source_status_path.exists():
        raise RuntimeError("Delta calibration requires the deterministic uniform-family arm")
    source = json.loads(source_status_path.read_text(encoding="utf-8"))
    if not source.get("complete") or source["selected_checkpoint"] == "parent":
        raise RuntimeError("Delta calibration requires a non-parent uniform-family checkpoint")
    parent_dir = Path(str(source["parent_adapter_path"]))
    candidate_dir = Path(str(source["adapter_path"]))
    parent_weights = parent_dir / "adapters.safetensors"
    candidate_weights = candidate_dir / "adapters.safetensors"
    parent_config = json.loads((parent_dir / "adapter_config.json").read_text(encoding="utf-8"))
    candidate_config = json.loads(
        (candidate_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    parent_lora = dict(parent_config["lora_parameters"])
    candidate_lora = dict(candidate_config["lora_parameters"])
    if float(parent_lora["scale"]) != float(candidate_lora["scale"]):
        raise RuntimeError("Effective adapter interpolation requires equal LoRA scales")
    merged_rank = int(parent_lora["rank"]) + int(candidate_lora["rank"])
    surfaces = [str(value) for value in settings["surfaces"]]
    root, output_dir = _calibration_paths(config, scale)
    fingerprint = canonical_hash(
        {
            "scale": scale,
            "parent": sha256_file(parent_weights),
            "candidate": sha256_file(candidate_weights),
            "surfaces": {
                name: sha256_file(config.path_for("stats_dir") / "surface" / f"{name}.jsonl")
                for name in surfaces
            },
            "effective_weight_interpolation": True,
            "merged_rank": merged_rank,
            "evaluator_version": 2,
        }
    )
    status_path = output_dir / "status.json"
    if status_path.exists() and not force:
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            return existing
        raise RuntimeError(
            "The delta-calibration fingerprint changed; use --force to replace this scale"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        output_dir / "adapters.safetensors",
        output_dir / "adapter_config.json",
        status_path,
    ):
        if path.exists():
            path.unlink()
    interpolated = interpolate_adapter_weights(
        parent_weights,
        candidate_weights,
        scale=scale,
    )
    active_path = output_dir / "adapters.safetensors"
    mx.save_safetensors(str(active_path), interpolated)
    adapter_config = json.loads((candidate_dir / "adapter_config.json").read_text(encoding="utf-8"))
    adapter_config["adapter_path"] = str(output_dir)
    adapter_config["lora_parameters"]["rank"] = merged_rank
    adapter_config.setdefault("stats", {}).update(
        {
            "method": "DGP effective-weight delta calibration",
            "delta_interpolation_scale": scale,
            "effective_weight_interpolation": True,
            "source_factor_ranks": [int(parent_lora["rank"]), int(candidate_lora["rank"])],
            "parent_adapter_sha256": source["parent_adapter_sha256"],
            "source_candidate_sha256": source["adapter_sha256"],
            "promotion_status": "candidate",
        }
    )
    write_json(output_dir / "adapter_config.json", adapter_config)
    scores = _score_adapter_surfaces(
        config,
        output_dir,
        {surface: _surface(config, surface) for surface in surfaces},
    )
    family_order = list(source["update_history"][-1]["family_order"])
    result = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "complete": True,
        "cycle": int(source["cycle"]),
        "arm": f"uniform-family-delta-{_scale_slug(scale)}",
        "scale": scale,
        "parent_adapter_path": str(parent_dir),
        "parent_adapter_sha256": source["parent_adapter_sha256"],
        "source_candidate_path": str(candidate_dir),
        "source_candidate_sha256": source["adapter_sha256"],
        "adapter_path": str(output_dir),
        "adapter_sha256": sha256_file(active_path),
        "selected_checkpoint": f"delta-scale-{_scale_slug(scale)}",
        "family_order": family_order,
        "scores": scores,
        "promotion_shard_opened": False,
        "sealed_final_surface_opened": False,
    }
    write_json(status_path, result)
    del interpolated
    gc.collect()
    mx.clear_cache()
    root.mkdir(parents=True, exist_ok=True)
    return result


def run_block_projection_arm(
    config: ProjectConfig,
    *,
    layer_scales: dict[int, float],
    force: bool = False,
) -> dict[str, Any]:
    settings = dict(config.section("block_projection"))
    registered = block_projection_profiles(settings)
    normalized = {int(layer): float(scale) for layer, scale in layer_scales.items()}
    if normalized not in registered:
        raise ValueError("The requested block projection is not registered")
    _, cone_dir = _cone_paths(config)
    source_status_path = cone_dir / "uniform-family" / "status.json"
    if not source_status_path.exists():
        raise RuntimeError("Block projection requires the deterministic uniform-family arm")
    source = json.loads(source_status_path.read_text(encoding="utf-8"))
    if not source.get("complete") or source["selected_checkpoint"] == "parent":
        raise RuntimeError("Block projection requires a non-parent uniform-family checkpoint")
    parent_dir = Path(str(source["parent_adapter_path"]))
    candidate_dir = Path(str(source["adapter_path"]))
    parent_weights = parent_dir / "adapters.safetensors"
    candidate_weights = candidate_dir / "adapters.safetensors"
    parent_config = json.loads((parent_dir / "adapter_config.json").read_text(encoding="utf-8"))
    candidate_config = json.loads(
        (candidate_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    parent_lora = dict(parent_config["lora_parameters"])
    candidate_lora = dict(candidate_config["lora_parameters"])
    if float(parent_lora["scale"]) != float(candidate_lora["scale"]):
        raise RuntimeError("Block projection requires equal source LoRA scales")
    merged_rank = int(parent_lora["rank"]) + int(candidate_lora["rank"])
    surfaces = [str(settings["selection_surface"])]
    root, output_dir = _block_paths(config, normalized)
    fingerprint = canonical_hash(
        {
            "layer_scales": normalized,
            "parent": sha256_file(parent_weights),
            "candidate": sha256_file(candidate_weights),
            "surfaces": {
                name: sha256_file(config.path_for("stats_dir") / "surface" / f"{name}.jsonl")
                for name in surfaces
            },
            "effective_weight_interpolation": True,
            "merged_rank": merged_rank,
            "evaluator_version": 2,
        }
    )
    status_path = output_dir / "status.json"
    if status_path.exists() and not force:
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            return existing
        raise RuntimeError("The block-projection fingerprint changed; use --force to replace it")

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        output_dir / "adapters.safetensors",
        output_dir / "adapter_config.json",
        status_path,
    ):
        if path.exists():
            path.unlink()
    interpolated = interpolate_adapter_blocks(
        parent_weights,
        candidate_weights,
        layer_scales=normalized,
    )
    active_path = output_dir / "adapters.safetensors"
    mx.save_safetensors(str(active_path), interpolated)
    adapter_config = dict(candidate_config)
    adapter_config["adapter_path"] = str(output_dir)
    adapter_config["lora_parameters"]["rank"] = merged_rank
    adapter_config.setdefault("stats", {}).update(
        {
            "method": "DGP verified block-support projection",
            "effective_weight_interpolation": True,
            "layer_scales": {str(layer): scale for layer, scale in sorted(normalized.items())},
            "source_factor_ranks": [int(parent_lora["rank"]), int(candidate_lora["rank"])],
            "parent_adapter_sha256": source["parent_adapter_sha256"],
            "source_candidate_sha256": source["adapter_sha256"],
            "promotion_status": "candidate",
        }
    )
    write_json(output_dir / "adapter_config.json", adapter_config)
    scores = _score_adapter_surfaces(
        config,
        output_dir,
        {surface: _surface(config, surface) for surface in surfaces},
    )
    family_order = list(source["update_history"][-1]["family_order"])
    result = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "complete": True,
        "cycle": int(source["cycle"]),
        "arm": "uniform-family-block-projection",
        "layer_scales": {str(layer): scale for layer, scale in sorted(normalized.items())},
        "active_layers": [layer for layer, scale in sorted(normalized.items()) if scale > 0.0],
        "parent_adapter_path": str(parent_dir),
        "parent_adapter_sha256": source["parent_adapter_sha256"],
        "source_candidate_path": str(candidate_dir),
        "source_candidate_sha256": source["adapter_sha256"],
        "adapter_path": str(output_dir),
        "adapter_sha256": sha256_file(active_path),
        "selected_checkpoint": _block_slug(normalized),
        "family_order": family_order,
        "scores": scores,
        "promotion_shard_opened": False,
        "sealed_final_surface_opened": False,
    }
    write_json(status_path, result)
    del interpolated
    gc.collect()
    mx.clear_cache()
    root.mkdir(parents=True, exist_ok=True)
    return result


def _mean_language_metric(score: dict[str, Any], key: str) -> float:
    return float(np.mean([float(value[key]) for value in score["languages"].values()]))


def _surface_comparison(
    parent: dict[str, Any],
    candidate: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    parent_trilingual = _mean_language_metric(parent, "normalized_regret")
    candidate_trilingual = _mean_language_metric(candidate, "normalized_regret")
    relative = (
        (parent_trilingual - candidate_trilingual) / parent_trilingual if parent_trilingual else 0.0
    )
    parent_language_accuracy = {
        key: float(value["accuracy"]) for key, value in parent["languages"].items()
    }
    candidate_language_accuracy = {
        key: float(value["accuracy"]) for key, value in candidate["languages"].items()
    }
    parent_language_regret = {
        key: float(value["normalized_regret"]) for key, value in parent["languages"].items()
    }
    candidate_language_regret = {
        key: float(value["normalized_regret"]) for key, value in candidate["languages"].items()
    }
    parent_family_regret = _group_regret(parent["selector"]["predictions"], "family_id")
    candidate_family_regret = _group_regret(candidate["selector"]["predictions"], "family_id")
    gate_results = {
        "trilingual_regret": relative
        >= float(gates["minimum_surface_trilingual_relative_improvement"]),
        "accuracy": float(candidate["selector"]["accuracy"])
        >= float(parent["selector"]["accuracy"]) - float(gates["maximum_accuracy_regression"]),
        "invalidity": float(candidate["selector"]["invalid_selection_rate"])
        <= float(parent["selector"]["invalid_selection_rate"])
        + float(gates["maximum_invalidity_increase"]),
        "retention": float(candidate["retention"]["accuracy"])
        >= float(parent["retention"]["accuracy"]) - float(gates["maximum_retention_regression"]),
        "language_accuracy": _noninferior_mapping(
            parent_language_accuracy,
            candidate_language_accuracy,
            maximum_regression=float(gates["maximum_language_accuracy_regression"]),
            higher_is_better=True,
        ),
        "language_regret": _noninferior_mapping(
            parent_language_regret,
            candidate_language_regret,
            maximum_regression=float(gates["maximum_language_regret_increase"]),
            higher_is_better=False,
        ),
        "domain_accuracy": _noninferior_mapping(
            {key: float(value) for key, value in parent["selector"]["domain_accuracy"].items()},
            {key: float(value) for key, value in candidate["selector"]["domain_accuracy"].items()},
            maximum_regression=float(gates["maximum_domain_accuracy_regression"]),
            higher_is_better=True,
        ),
        "family_regret": _noninferior_mapping(
            parent_family_regret,
            candidate_family_regret,
            maximum_regression=float(gates["maximum_family_regret_increase"]),
            higher_is_better=False,
        ),
        "finite_metrics": all(
            math.isfinite(value) for value in (parent_trilingual, candidate_trilingual, relative)
        ),
    }
    return {
        "parent_trilingual_regret": parent_trilingual,
        "candidate_trilingual_regret": candidate_trilingual,
        "trilingual_relative_regret_improvement": relative,
        "parent_accuracy": float(parent["selector"]["accuracy"]),
        "candidate_accuracy": float(candidate["selector"]["accuracy"]),
        "parent_invalidity": float(parent["selector"]["invalid_selection_rate"]),
        "candidate_invalidity": float(candidate["selector"]["invalid_selection_rate"]),
        "parent_language_accuracy": parent_language_accuracy,
        "candidate_language_accuracy": candidate_language_accuracy,
        "parent_language_regret": parent_language_regret,
        "candidate_language_regret": candidate_language_regret,
        "parent_family_regret": parent_family_regret,
        "candidate_family_regret": candidate_family_regret,
        "gates": gate_results,
        "all_gates_passed": all(gate_results.values()),
    }


def choose_delta_scale(
    comparisons: dict[float, dict[str, dict[str, Any]]],
    *,
    minimum_worst_surface_improvement: float,
) -> dict[str, Any] | None:
    eligible: list[dict[str, Any]] = []
    for scale, surfaces in comparisons.items():
        improvements = [
            float(value["trilingual_relative_regret_improvement"]) for value in surfaces.values()
        ]
        if (
            all(bool(value["all_gates_passed"]) for value in surfaces.values())
            and min(improvements) >= minimum_worst_surface_improvement
        ):
            eligible.append(
                {
                    "scale": scale,
                    "worst_surface_relative_improvement": min(improvements),
                    "mean_surface_relative_improvement": float(np.mean(improvements)),
                    "mean_invalidity": float(
                        np.mean([value["candidate_invalidity"] for value in surfaces.values()])
                    ),
                    "mean_accuracy": float(
                        np.mean([value["candidate_accuracy"] for value in surfaces.values()])
                    ),
                }
            )
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda value: (
            -float(value["worst_surface_relative_improvement"]),
            -float(value["mean_surface_relative_improvement"]),
            float(value["mean_invalidity"]),
            -float(value["mean_accuracy"]),
            float(value["scale"]),
        ),
    )


def run_delta_calibration(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    settings = dict(config.section("delta_calibration"))
    scales = [0.0, *[float(value) for value in settings["scales"]]]
    root, _ = _calibration_paths(config, 0.0)
    statuses: dict[float, dict[str, Any]] = {}
    for scale in scales:
        command = [
            sys.executable,
            "-m",
            "charlie_alpha.cli",
            "stats",
            "policy-calibrate-arm",
            "--config",
            str(config.path),
            "--scale",
            str(scale),
        ]
        if force:
            command.append("--force")
        subprocess.run(
            command,
            cwd=config.root,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        _, scale_dir = _calibration_paths(config, scale)
        statuses[scale] = json.loads((scale_dir / "status.json").read_text(encoding="utf-8"))
    parent = statuses[0.0]
    gates = dict(settings["gates"])
    comparisons: dict[float, dict[str, dict[str, Any]]] = {}
    for scale in scales[1:]:
        comparisons[scale] = {
            surface: _surface_comparison(
                parent["scores"][surface],
                statuses[scale]["scores"][surface],
                gates,
            )
            for surface in settings["surfaces"]
        }
    selected = choose_delta_scale(
        comparisons,
        minimum_worst_surface_improvement=float(
            gates["minimum_worst_surface_trilingual_relative_improvement"]
        ),
    )
    report = {
        "schema_version": 1,
        "fingerprint": canonical_hash(
            {
                "settings": settings,
                "arms": {
                    str(scale): status["fingerprint"] for scale, status in sorted(statuses.items())
                },
                "evaluator_version": 2,
            }
        ),
        "complete": True,
        "method": "parent-to-uniform effective-weight trust-path calibration",
        "scales": scales[1:],
        "surfaces": list(settings["surfaces"]),
        "selection_rule": (
            "maximize worst-surface trilingual regret improvement, then mean improvement, "
            "invalidity, accuracy, and smaller scale"
        ),
        "comparisons": {str(scale): value for scale, value in comparisons.items()},
        "selected": selected,
        "selected_status_path": (
            str(_calibration_paths(config, float(selected["scale"]))[1] / "status.json")
            if selected
            else None
        ),
        "proceed_to_promotion": selected is not None,
        "sealed_promotion_surface_opened": False,
        "sealed_final_surface_opened": False,
        "claim_boundary": (
            "Both calibration surfaces are reusable development data. A selected scale still "
            "requires a fresh single-use promotion shard."
        ),
    }
    write_json(root / "report.json", report)
    public = dict(report)
    if selected:
        public["selected_status_path"] = str(
            Path("artifacts")
            / "evolve"
            / "common-descent"
            / "delta-calibration"
            / f"scale-{_scale_slug(float(selected['scale']))}"
            / "status.json"
        )
    write_json(config.root / "reports" / "evolve" / "delta-calibration.json", public)
    return report


def choose_block_profile(
    comparisons: list[dict[str, Any]],
    *,
    minimum_worst_surface_improvement: float,
) -> dict[str, Any] | None:
    eligible: list[dict[str, Any]] = []
    for comparison in comparisons:
        surfaces = dict(comparison["surfaces"])
        improvements = [
            float(value["trilingual_relative_regret_improvement"]) for value in surfaces.values()
        ]
        if (
            all(bool(value["all_gates_passed"]) for value in surfaces.values())
            and min(improvements) >= minimum_worst_surface_improvement
        ):
            eligible.append(
                {
                    "slug": str(comparison["slug"]),
                    "layer_scales": dict(comparison["layer_scales"]),
                    "active_layers": list(comparison["active_layers"]),
                    "worst_surface_relative_improvement": min(improvements),
                    "mean_surface_relative_improvement": float(np.mean(improvements)),
                    "mean_invalidity": float(
                        np.mean([value["candidate_invalidity"] for value in surfaces.values()])
                    ),
                    "mean_accuracy": float(
                        np.mean([value["candidate_accuracy"] for value in surfaces.values()])
                    ),
                }
            )
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda value: (
            -float(value["worst_surface_relative_improvement"]),
            -float(value["mean_surface_relative_improvement"]),
            float(value["mean_invalidity"]),
            -float(value["mean_accuracy"]),
            len(value["active_layers"]),
            max(float(scale) for scale in value["layer_scales"].values()),
            str(value["slug"]),
        ),
    )


def run_block_projection(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    settings = dict(config.section("block_projection"))
    profiles = block_projection_profiles(settings)
    root, _ = _block_paths(config, profiles[0])
    parent = run_delta_calibration_arm(config, scale=0.0, force=False)
    statuses: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        command = [
            sys.executable,
            "-m",
            "charlie_alpha.cli",
            "stats",
            "policy-block-arm",
            "--config",
            str(config.path),
            "--layer-scales",
            ",".join(f"{layer}={scale}" for layer, scale in sorted(profile.items())),
        ]
        if force:
            command.append("--force")
        subprocess.run(
            command,
            cwd=config.root,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        _, profile_dir = _block_paths(config, profile)
        statuses[_block_slug(profile)] = json.loads(
            (profile_dir / "status.json").read_text(encoding="utf-8")
        )
    gates = dict(settings["gates"])
    selection_surface = str(settings["selection_surface"])
    confirmation_surface = str(settings["confirmation_shard"]["split"])
    comparisons: list[dict[str, Any]] = []
    for profile in profiles:
        slug = _block_slug(profile)
        status = statuses[slug]
        comparisons.append(
            {
                "slug": slug,
                "layer_scales": {str(layer): scale for layer, scale in sorted(profile.items())},
                "active_layers": [layer for layer, scale in sorted(profile.items()) if scale > 0.0],
                "surfaces": {
                    selection_surface: _surface_comparison(
                        parent["scores"][selection_surface],
                        status["scores"][selection_surface],
                        gates,
                    )
                },
            }
        )
    selected = choose_block_profile(
        comparisons,
        minimum_worst_surface_improvement=float(gates["minimum_selection_relative_improvement"]),
    )
    fingerprint = canonical_hash(
        {
            "settings": settings,
            "arms": {slug: status["fingerprint"] for slug, status in sorted(statuses.items())},
            "evaluator_version": 3,
        }
    )
    report_path = root / "report.json"
    if report_path.exists() and not force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            return existing

    confirmation: dict[str, Any] | None = None
    if selected:
        selected_status = statuses[str(selected["slug"])]
        confirmation_manifest, confirmation_rows = _ensure_block_confirmation_shard(
            config,
            root,
        )
        parent_confirmation = _score_adapter_surfaces(
            config,
            Path(str(parent["parent_adapter_path"])),
            {confirmation_surface: confirmation_rows},
        )[confirmation_surface]
        candidate_confirmation = _score_adapter_surfaces(
            config,
            Path(str(selected_status["adapter_path"])),
            {confirmation_surface: confirmation_rows},
        )[confirmation_surface]
        confirmation_comparison = _surface_comparison(
            parent_confirmation,
            candidate_confirmation,
            gates,
        )
        confirmation = {
            "surface": confirmation_surface,
            "manifest": confirmation_manifest,
            "selection_timing": (
                "one profile fixed after valid-only search and before the immutable confirmation "
                "shard was generated or scored"
            ),
            "comparison": confirmation_comparison,
            "minimum_relative_improvement": float(
                gates["minimum_confirmation_relative_improvement"]
            ),
            "passed": (
                bool(confirmation_comparison["all_gates_passed"])
                and float(confirmation_comparison["trilingual_relative_regret_improvement"])
                >= float(gates["minimum_confirmation_relative_improvement"])
            ),
        }
    report = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "complete": True,
        "method": "DGP verified block-support projection",
        "effective_weight_interpolation": True,
        "profiles": len(profiles),
        "selection_surface": selection_surface,
        "confirmation_surface": confirmation_surface,
        "selection_rule": (
            "on valid only, maximize trilingual regret improvement after every granular gate, "
            "then invalidity, accuracy, sparsity, and smaller amplitude; score only the fixed "
            "winner once on an immutable confirmation shard"
        ),
        "comparisons": comparisons,
        "selected": selected,
        "confirmation": confirmation,
        "selected_status_path": (
            str(root / str(selected["slug"]) / "status.json") if selected else None
        ),
        "proceed_to_promotion": bool(confirmation and confirmation["passed"]),
        "sealed_promotion_surface_opened": False,
        "sealed_final_surface_opened": False,
        "claim_boundary": (
            "The layer masks are selected on reusable valid data and exactly one fixed mask is "
            "confirmed on a new immutable DGP shard. A passing mask still requires the fresh "
            "single-use promotion shard. Layer-wise merging is prior art."
        ),
    }
    write_json(report_path, report)
    public = dict(report)
    if selected:
        public["selected_status_path"] = str(
            Path("artifacts")
            / "evolve"
            / "common-descent"
            / "block-projection"
            / str(selected["slug"])
            / "status.json"
        )
    write_json(config.root / "reports" / "evolve" / "block-projection.json", public)
    return report

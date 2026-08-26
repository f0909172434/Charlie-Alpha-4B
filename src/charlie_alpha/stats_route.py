from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl
from .stats_calibrate import _surface_comparison
from .stats_cone import _cone_paths
from .stats_dgp import build_blueprints, simulate_scenario
from .stats_evolve import _score_adapter_surfaces


def _family_metrics(score: dict[str, Any], family_id: str) -> dict[str, Any]:
    def summarize(predictions: list[dict[str, Any]]) -> dict[str, float]:
        if not predictions:
            raise RuntimeError(f"Selection surface has no records for DGP family {family_id}")
        return {
            "count": float(len(predictions)),
            "normalized_regret": float(
                np.mean([float(prediction["normalized_regret"]) for prediction in predictions])
            ),
            "accuracy": float(
                np.mean(
                    [
                        prediction["predicted_method_id"] == prediction["oracle_method_id"]
                        for prediction in predictions
                    ]
                )
            ),
            "invalid_selection_rate": float(
                np.mean([not bool(prediction["valid"]) for prediction in predictions])
            ),
        }

    languages = {
        language: summarize(
            [
                prediction
                for prediction in result["predictions"]
                if str(prediction["family_id"]) == family_id
            ]
        )
        for language, result in score["languages"].items()
    }
    aggregate = summarize(
        [
            prediction
            for result in score["languages"].values()
            for prediction in result["predictions"]
            if str(prediction["family_id"]) == family_id
        ]
    )
    return {**aggregate, "languages": languages}


def _family_noninferior(
    metrics: dict[str, Any],
    parent_metrics: dict[str, Any],
    gates: dict[str, Any],
) -> bool:
    if float(metrics["invalid_selection_rate"]) > float(
        parent_metrics["invalid_selection_rate"]
    ) + float(gates["maximum_invalidity_increase"]) or float(metrics["accuracy"]) < float(
        parent_metrics["accuracy"]
    ) - float(gates["maximum_accuracy_regression"]):
        return False
    for language, parent_language in parent_metrics["languages"].items():
        candidate_language = metrics["languages"][language]
        if (
            float(candidate_language["invalid_selection_rate"])
            > float(parent_language["invalid_selection_rate"])
            + float(gates["maximum_invalidity_increase"])
            or float(candidate_language["accuracy"])
            < float(parent_language["accuracy"])
            - float(gates["maximum_language_accuracy_regression"])
            or float(candidate_language["normalized_regret"])
            > float(parent_language["normalized_regret"])
            + float(gates["maximum_language_regret_increase"])
        ):
            return False
    return True


def select_family_routes(
    options: dict[str, dict[str, Any]],
    gates: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if "parent" not in options:
        raise ValueError("Family routing requires the unchanged parent option")
    families = sorted(
        {
            str(prediction["family_id"])
            for prediction in options["parent"]["score"]["selector"]["predictions"]
        }
    )
    mapping: dict[str, dict[str, Any]] = {}
    for family_id in families:
        candidates: list[dict[str, Any]] = []
        for slug, option in options.items():
            metrics = _family_metrics(option["score"], family_id)
            candidates.append(
                {
                    "slug": slug,
                    "metrics": metrics,
                    "active_layers": list(option["active_layers"]),
                    "amplitude": float(option["amplitude"]),
                }
            )
        parent_metrics = next(value["metrics"] for value in candidates if value["slug"] == "parent")

        selected = min(
            [
                value
                for value in candidates
                if _family_noninferior(value["metrics"], parent_metrics, gates)
            ],
            key=lambda value: (
                float(value["metrics"]["normalized_regret"]),
                float(value["metrics"]["invalid_selection_rate"]),
                -float(value["metrics"]["accuracy"]),
                0 if value["slug"] == "parent" else 1,
                len(value["active_layers"]),
                float(value["amplitude"]),
                str(value["slug"]),
            ),
        )
        parent_regret = float(parent_metrics["normalized_regret"])
        selected_regret = float(selected["metrics"]["normalized_regret"])
        mapping[family_id] = {
            **selected,
            "parent_metrics": parent_metrics,
            "relative_regret_improvement": (
                (parent_regret - selected_regret) / parent_regret if parent_regret else 0.0
            ),
        }
    return mapping


def _ensure_route_confirmation_shard(
    config: ProjectConfig,
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    settings = dict(config.section("family_route")["confirmation_shard"])
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
            raise RuntimeError("The immutable family-route confirmation shard is incomplete")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("fingerprint") == fingerprint
            and existing.get("sha256") == sha256_file(path)
            and int(existing.get("count", 0)) == count
        ):
            return existing, list(read_jsonl(path))
        raise RuntimeError("The family-route confirmation shard is immutable")
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
        "used_for_route_selection": False,
        "used_for_training": False,
        "single_use": True,
        "sealed_at_preparation": True,
        "promotion_surface_opened": False,
        "final_surface_opened": False,
    }
    write_json(manifest_path, manifest)
    return manifest, simulations


def _aggregate_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        raise RuntimeError("Family-routed scoring produced no predictions")
    ordered = sorted(predictions, key=lambda value: str(value["blueprint_id"]))
    domain_correct: dict[str, list[bool]] = defaultdict(list)
    family_correct: dict[str, list[bool]] = defaultdict(list)
    for prediction in ordered:
        correct = prediction["predicted_method_id"] == prediction["oracle_method_id"]
        domain_correct[str(prediction["domain"])].append(correct)
        family_correct[str(prediction["family_id"])].append(correct)
    return {
        "count": len(ordered),
        "normalized_regret": float(
            np.mean([float(prediction["normalized_regret"]) for prediction in ordered])
        ),
        "accuracy": float(
            np.mean(
                [
                    prediction["predicted_method_id"] == prediction["oracle_method_id"]
                    for prediction in ordered
                ]
            )
        ),
        "invalid_selection_rate": float(
            np.mean([not bool(prediction["valid"]) for prediction in ordered])
        ),
        "domain_accuracy": {
            key: float(np.mean(values)) for key, values in sorted(domain_correct.items())
        },
        "family_accuracy": {
            key: float(np.mean(values)) for key, values in sorted(family_correct.items())
        },
        "predictions": ordered,
    }


def _score_oracle_family_route(
    config: ProjectConfig,
    *,
    mapping: dict[str, dict[str, Any]],
    adapter_paths: dict[str, Path],
    surface: list[dict[str, Any]],
    retention: dict[str, Any],
    split: str | None = None,
) -> dict[str, Any]:
    split = split or str(config.section("family_route")["confirmation_shard"]["split"])
    language_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    expected = sorted(str(row["scenario"]["blueprint_id"]) for row in surface)
    routes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in surface:
        family_id = str(row["scenario"]["family_id"])
        routes[str(mapping[family_id]["slug"])].append(row)
    for slug, rows in sorted(routes.items()):
        scored = _score_adapter_surfaces(
            config,
            adapter_paths[slug],
            {split: rows},
            include_retention=False,
        )[split]
        for language, result in scored["languages"].items():
            language_predictions[language].extend(result["predictions"])
    languages = {
        language: _aggregate_predictions(predictions)
        for language, predictions in sorted(language_predictions.items())
    }
    for language, result in languages.items():
        actual = sorted(str(row["blueprint_id"]) for row in result["predictions"])
        if actual != expected:
            raise RuntimeError(f"Family route changed {language} confirmation coverage")
    return {
        "selector": languages["en"],
        "languages": languages,
        "retention": retention,
    }


def run_oracle_family_route(
    config: ProjectConfig,
    *,
    force: bool = False,
    selection_only: bool = False,
) -> dict[str, Any]:
    settings = dict(config.section("family_route"))
    _, cone_root = _cone_paths(config)
    block_root = cone_root / "block-projection"
    block_report_path = block_root / "report.json"
    if not block_report_path.exists():
        raise RuntimeError("Family routing requires the completed block projection")
    block_report = json.loads(block_report_path.read_text(encoding="utf-8"))
    if not block_report.get("complete") or int(block_report.get("profiles", 0)) != 30:
        raise RuntimeError("Block projection artifacts are incomplete")
    parent_status = json.loads(
        (cone_root / "delta-calibration" / "scale-0p00" / "status.json").read_text(encoding="utf-8")
    )
    selection_surface = str(settings["selection_surface"])
    options: dict[str, dict[str, Any]] = {
        "parent": {
            "score": parent_status["scores"][selection_surface],
            "adapter_path": Path(str(parent_status["parent_adapter_path"])),
            "adapter_sha256": str(parent_status["parent_adapter_sha256"]),
            "active_layers": [],
            "amplitude": 0.0,
            "fingerprint": str(parent_status["fingerprint"]),
        }
    }
    for comparison in block_report["comparisons"]:
        slug = str(comparison["slug"])
        status_path = block_root / slug / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if not status.get("complete") or list(status["scores"]) != [selection_surface]:
            raise RuntimeError(f"Block profile is incomplete: {slug}")
        options[slug] = {
            "score": status["scores"][selection_surface],
            "adapter_path": Path(str(status["adapter_path"])),
            "adapter_sha256": str(status["adapter_sha256"]),
            "active_layers": list(status["active_layers"]),
            "amplitude": max(float(value) for value in status["layer_scales"].values()),
            "fingerprint": str(status["fingerprint"]),
        }
    mapping = select_family_routes(options, dict(settings["gates"]))
    route_root = cone_root / "family-route"
    route_root.mkdir(parents=True, exist_ok=True)
    selection_fingerprint = canonical_hash(
        {
            "settings": settings,
            "block_report": block_report["fingerprint"],
            "options": {
                slug: {
                    "fingerprint": option["fingerprint"],
                    "adapter_sha256": option["adapter_sha256"],
                }
                for slug, option in sorted(options.items())
            },
            "mapping": mapping,
            "selector_version": 1,
        }
    )
    selection_path = route_root / "selection.json"
    existing_selection = (
        json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else None
    )
    confirmation_exists = (route_root / "confirmation-manifest.json").exists()
    if existing_selection and existing_selection.get("fingerprint") != selection_fingerprint:
        if confirmation_exists:
            raise RuntimeError("Family routes cannot change after confirmation was opened")
        if not force:
            raise RuntimeError("Family-route selection changed; use --force before confirmation")
    nonparent_families = sorted(
        family_id for family_id, route in mapping.items() if route["slug"] != "parent"
    )
    selection = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": selection_fingerprint,
        "selection_surface": selection_surface,
        "selection_rule": (
            "per-family aggregate and per-language noninferiority, then trilingual regret, "
            "invalidity, accuracy, and parent on ties"
        ),
        "mapping": mapping,
        "nonparent_families": nonparent_families,
        "confirmation_shard_opened": confirmation_exists,
        "promotion_shard_opened": False,
        "sealed_final_surface_opened": False,
    }
    write_json(selection_path, selection)
    if selection_only:
        return {
            "stage": "selection-complete",
            "selection": selection,
            "confirmation_shard_opened": confirmation_exists,
            "promotion_shard_opened": False,
            "sealed_final_surface_opened": False,
        }

    report_path = route_root / "report.json"
    if report_path.exists() and not force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("selection_fingerprint") == selection_fingerprint and existing.get(
            "complete"
        ):
            existing["selection"] = selection
            write_json(report_path, existing)
            write_json(config.root / "reports" / "evolve" / "family-route.json", existing)
            return existing
    if not nonparent_families:
        report = {
            "schema_version": 1,
            "complete": True,
            "selection_fingerprint": selection_fingerprint,
            "method": "DGP oracle family-routed block profiles",
            "oracle_route_upper_bound": True,
            "selection": selection,
            "confirmation": None,
            "proceed_to_router_implementation": False,
            "promotion_shard_opened": False,
            "sealed_final_surface_opened": False,
        }
        write_json(report_path, report)
        write_json(config.root / "reports" / "evolve" / "family-route.json", report)
        return report

    confirmation_manifest, confirmation_rows = _ensure_route_confirmation_shard(
        config,
        route_root,
    )
    selection["confirmation_shard_opened"] = True
    write_json(selection_path, selection)
    parent_confirmation = _score_adapter_surfaces(
        config,
        options["parent"]["adapter_path"],
        {str(confirmation_manifest["split"]): confirmation_rows},
    )[str(confirmation_manifest["split"])]
    routed_confirmation = _score_oracle_family_route(
        config,
        mapping=mapping,
        adapter_paths={slug: option["adapter_path"] for slug, option in options.items()},
        surface=confirmation_rows,
        retention=parent_confirmation["retention"],
    )
    comparison = _surface_comparison(
        parent_confirmation,
        routed_confirmation,
        dict(settings["gates"]),
    )
    passed = bool(comparison["all_gates_passed"]) and float(
        comparison["trilingual_relative_regret_improvement"]
    ) >= float(settings["gates"]["minimum_confirmation_relative_improvement"])
    report = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": canonical_hash(
            {
                "selection": selection_fingerprint,
                "confirmation": confirmation_manifest["fingerprint"],
                "evaluator_version": 1,
            }
        ),
        "selection_fingerprint": selection_fingerprint,
        "method": "DGP oracle family-routed block profiles",
        "oracle_route_upper_bound": True,
        "selection": selection,
        "confirmation": {
            "manifest": confirmation_manifest,
            "comparison": comparison,
            "minimum_relative_improvement": float(
                settings["gates"]["minimum_confirmation_relative_improvement"]
            ),
            "passed": passed,
        },
        "proceed_to_router_implementation": passed,
        "promotion_shard_opened": False,
        "sealed_final_surface_opened": False,
        "claim_boundary": (
            "This is an oracle-family routing upper bound. It cannot be promoted until a real "
            "router reproduces the result on a separate frozen surface."
        ),
    }
    write_json(report_path, report)
    write_json(config.root / "reports" / "evolve" / "family-route.json", report)
    return report

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_bytes, sha256_file, write_json, write_jsonl
from .stats_agent import StatsAgent
from .stats_canonical_bottleneck import _registered_scenarios
from .stats_catalog_grounding import _evaluate_style
from .stats_cross_format import _format_metrics, _format_shift_case
from .stats_dgp import Scenario, simulate_scenario
from .stats_family_router import _expert_context
from .stats_representation_probe import (
    _METHOD_IDS,
    _extract_representations,
    _fit_ridge_probe,
    _load_representations,
    _probe_scores,
    _representation_paths,
)
from .stats_router_replication import _historical_scenario_audit, _scenario_semantic_payload

_EVALUATOR_VERSION = 1


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "selector-head-v1"


def _data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "selector-head-v1"


def _implementation_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "stats_selector_head.py": sha256_file(Path(__file__)),
        "stats_representation_probe.py": sha256_file(root / "stats_representation_probe.py"),
        "stats_catalog_grounding.py": sha256_file(root / "stats_catalog_grounding.py"),
        "stats_cross_format.py": sha256_file(root / "stats_cross_format.py"),
    }


def _h13_scenarios(config: ProjectConfig) -> list[Scenario]:
    path = config.root / "reports" / "evolve" / "representation-probe-v1-contract.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    scenarios: list[Scenario] = []
    for name, fields in contract["blueprint_contracts"].items():
        shard = _registered_scenarios(
            dict(contract["settings"][name]),
            name=f"representation-probe:{name}",
        )
        actual = canonical_hash([scenario.to_dict() for scenario in shard])
        if actual != str(fields["blueprint_sha256"]):
            raise RuntimeError(f"H13 blueprint reconstruction changed for {name}")
        scenarios.extend(shard)
    return scenarios


def _head_recipe(config: ProjectConfig) -> dict[str, Any]:
    h13_selection = json.loads(
        (config.root / "reports" / "evolve" / "representation-probe-v1-selection.json").read_text(
            encoding="utf-8"
        )
    )
    ridge_lambda = float(h13_selection["selected_lambdas"]["menu-free"])
    train_x, train_y, train_ids = _load_representations(
        _representation_paths(config, "training_shard")["menu-free"]
    )
    select_x, select_y, select_ids = _load_representations(
        _representation_paths(config, "selection_shard")["menu-free"]
    )
    fit_x = np.concatenate([train_x, select_x], axis=0)
    fit_y = np.concatenate([train_y, select_y], axis=0)
    model = _fit_ridge_probe(fit_x, fit_y, ridge_lambda=ridge_lambda)
    weights = np.asarray(model["weights"], dtype=np.float64)
    observed = [int(value) for value in model["observed"]]
    return {
        "weights": weights,
        "observed": observed,
        "ridge_lambda": ridge_lambda,
        "fit_count": int(len(fit_y)),
        "training_case_ids_sha256": canonical_hash(train_ids),
        "selection_case_ids_sha256": canonical_hash(select_ids),
        "weight_sha256": sha256_bytes(weights.astype("<f8", copy=False).tobytes()),
        "observed_methods": [_METHOD_IDS[index] for index in observed],
    }


def _save_head(config: ProjectConfig, recipe: dict[str, Any]) -> Path:
    path = _root(config) / "head.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        np.savez_compressed(
            path,
            weights=np.asarray(recipe["weights"], dtype=np.float64),
            observed=np.asarray(recipe["observed"], dtype=np.int64),
        )
    data = np.load(path, allow_pickle=False)
    weights = np.asarray(data["weights"], dtype=np.float64)
    observed = [int(value) for value in np.asarray(data["observed"], dtype=np.int64).tolist()]
    if sha256_bytes(weights.astype("<f8", copy=False).tobytes()) != recipe["weight_sha256"]:
        raise RuntimeError("H14 selector-head weights changed")
    if observed != recipe["observed"]:
        raise RuntimeError("H14 selector-head observed classes changed")
    return path


def prepare_selector_head_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("selector_head"))
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "contract.json"
    public_path = config.root / "reports" / "evolve" / "selector-head-v1-contract.json"

    h13_path = config.root / "reports" / "evolve" / "representation-probe-v1-confirmation.json"
    h13 = json.loads(h13_path.read_text(encoding="utf-8"))
    if not h13.get("representation_hypothesis_confirmed"):
        raise RuntimeError("H14 requires confirmed H13 representation decodability")
    if h13.get("selected_route") != "selector-head":
        raise RuntimeError("H14 requires H13 to select the dedicated selector-head route")

    recipe = _head_recipe(config)
    head_path = _save_head(config, recipe)
    _, adapter_paths = _expert_context(config)
    parent = adapter_paths["parent"]
    parent_sha = sha256_file(parent / "adapters.safetensors")

    registered: dict[str, Any] = {}
    new_scenarios: list[Scenario] = []
    seen: set[str] = set()
    for name in ("selection_shard", "confirmation_shard"):
        shard = dict(settings[name])
        scenarios = _registered_scenarios(shard, name=f"selector-head:{name}")
        ids = {scenario.blueprint_id for scenario in scenarios}
        if ids & seen:
            raise RuntimeError(f"H14 blueprint overlap at {name}")
        seen.update(ids)
        new_scenarios.extend(scenarios)
        registered[name] = {
            "split": str(shard["split"]),
            "seed": int(shard["seed"]),
            "pool_count": int(shard["pool_count"]),
            "selected_per_family": int(shard["selected_per_family"]),
            "count": len(scenarios),
            "blueprint_sha256": canonical_hash([scenario.to_dict() for scenario in scenarios]),
        }

    previous = _h13_scenarios(config)
    prior_ids = {scenario.blueprint_id for scenario in previous}
    prior_semantics = {
        canonical_hash(_scenario_semantic_payload(scenario.to_dict())) for scenario in previous
    }
    new_ids = {scenario.blueprint_id for scenario in new_scenarios}
    new_semantics = {
        canonical_hash(_scenario_semantic_payload(scenario.to_dict()))
        for scenario in new_scenarios
    }
    if new_ids & prior_ids or new_semantics & prior_semantics:
        raise RuntimeError("H14 overlaps H13 registered blueprints or semantic points")
    audit = _historical_scenario_audit(
        config,
        new_scenarios,
        excluded_root=_data_root(config),
        minimum_normalized_distance=float(settings["minimum_normalized_distance"]),
    )
    if not audit["passed"]:
        raise RuntimeError("H14 blueprints failed historical-overlap audit")

    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": "H14 menu-free dedicated selector-head architecture",
        "method_version": int(settings["method_version"]),
        "causal_question": (
            "Does replacing only the freely generated method field with the frozen H13 "
            "menu-free linear selector head convert latent representation quality into stable "
            "canonical JSON accuracy, while leaving the same model-generated columns unchanged?"
        ),
        "h13_result_fingerprint": h13["result_fingerprint"],
        "h13_report_sha256": sha256_file(h13_path),
        "parent": {
            "name": "v0.3.0-parent",
            "adapter_path": str(parent),
            "adapter_sha256": parent_sha,
        },
        "selector_head": {
            "source": "H13 menu-free final-RMSNorm representations, training+selection only",
            "ridge_lambda": recipe["ridge_lambda"],
            "fit_count": recipe["fit_count"],
            "output_slots": len(_METHOD_IDS),
            "observed_method_count": len(recipe["observed"]),
            "observed_methods": recipe["observed_methods"],
            "weight_sha256": recipe["weight_sha256"],
            "artifact_path": str(head_path),
            "artifact_sha256": sha256_file(head_path),
            "training_case_ids_sha256": recipe["training_case_ids_sha256"],
            "selection_case_ids_sha256": recipe["selection_case_ids_sha256"],
        },
        "settings": settings,
        "blueprint_contracts": registered,
        "historical_overlap_audit": audit,
        "h13_overlap": {
            "prior_blueprints": len(prior_ids),
            "h14_blueprints": len(new_ids),
            "blueprint_id_overlap_count": len(new_ids & prior_ids),
            "semantic_overlap_count": len(new_semantics & prior_semantics),
        },
        "arms": {
            "menu-free-control": "unchanged v0.3 parent joint canonical JSON generation",
            "selector-head": (
                "same generated columns, but methods is replaced by the frozen H13 menu-free "
                "linear head argmax; no catalog is present"
            ),
        },
        "selection_policy": (
            "Confirmation remains unopened unless the head passes every absolute and paired "
            "selection gate. No head retraining or threshold change is allowed after selection."
        ),
        "claim_boundary": (
            "H14 can validate the synthetic menu-free decision architecture only. It does not "
            "change v0.3 weights, replace the official champion, or reuse E2 as fresh evidence."
        ),
        "implementation_sha256": _implementation_manifest(),
        "confirmation_simulations_opened": False,
        "external_benchmark_opened": False,
    }
    contract["fingerprint"] = canonical_hash(contract)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("H14 contract is immutable")
        write_json(public_path, existing)
        return existing
    write_json(lock_path, contract)
    write_json(public_path, contract)
    return contract


def _simulate_surface(
    config: ProjectConfig,
    contract: dict[str, Any],
    *,
    name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if name == "confirmation_shard":
        pilot_path = _root(config) / "pilot.json"
        if not pilot_path.exists():
            raise RuntimeError("H14 confirmation cannot open before selection")
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        if not pilot.get("confirmation_authorized"):
            raise RuntimeError("H14 selection did not authorize confirmation")
    shard = dict(contract["settings"][name])
    scenarios = _registered_scenarios(shard, name=f"selector-head:{name}")
    actual = canonical_hash([scenario.to_dict() for scenario in scenarios])
    if actual != contract["blueprint_contracts"][name]["blueprint_sha256"]:
        raise RuntimeError(f"H14 registered blueprints changed for {name}")
    path = _data_root(config) / "surfaces" / f"{name}.jsonl"
    manifest_path = path.with_suffix(".manifest.json")
    stats = config.section("stats_data")
    fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "name": name,
            "blueprints": actual,
            "simulator": {
                "initial_repetitions": stats["initial_repetitions"],
                "escalation_repetitions": stats["escalation_repetitions"],
                "ranking_uncertainty_margin": stats["ranking_uncertainty_margin"],
                "regret_temperature": stats["regret_temperature"],
            },
        }
    )
    if path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("fingerprint") != fingerprint
            or manifest.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"H14 {name} surface changed")
        return manifest, list(read_jsonl(path))
    simulations = [
        simulate_scenario(
            scenario,
            initial_repetitions=int(stats["initial_repetitions"]),
            escalation_repetitions=[int(value) for value in stats["escalation_repetitions"]],
            uncertainty_margin=float(stats["ranking_uncertainty_margin"]),
            temperature=float(stats["regret_temperature"]),
        )
        for scenario in scenarios
    ]
    write_jsonl(path, simulations)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "name": name,
        "count": len(simulations),
        "sha256": sha256_file(path),
    }
    write_json(manifest_path, manifest)
    return manifest, list(read_jsonl(path))


def prepare_selector_head_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_selector_head_contract(config)
    manifest, simulations = _simulate_surface(config, contract, name="selection_shard")
    cases = [_format_shift_case(simulation) for simulation in simulations]
    path = _data_root(config) / "cases" / "selection_shard.jsonl"
    write_jsonl(path, cases)
    status = {
        "schema_version": 1,
        "complete": True,
        "contract_fingerprint": contract["fingerprint"],
        "selection_surface_fingerprint": manifest["fingerprint"],
        "selection_case_sha256": sha256_file(path),
        "selection_count": len(cases),
        "confirmation_opened": False,
    }
    status["fingerprint"] = canonical_hash(status)
    write_json(_data_root(config) / "data-status.json", status)
    write_json(config.root / "reports" / "evolve" / "selector-head-v1-data.json", status)
    return status


def _load_head(contract: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(contract["selector_head"]["artifact_path"]))
    if sha256_file(path) != str(contract["selector_head"]["artifact_sha256"]):
        raise RuntimeError("H14 selector-head artifact changed")
    data = np.load(path, allow_pickle=False)
    return {
        "weights": np.asarray(data["weights"], dtype=np.float64),
        "observed": [int(value) for value in np.asarray(data["observed"], dtype=np.int64).tolist()],
    }


def _head_details(
    cases: list[dict[str, Any]],
    control_details: list[dict[str, Any]],
    vectors: np.ndarray,
    head: dict[str, Any],
) -> list[dict[str, Any]]:
    if len(cases) != len(control_details) or len(cases) != len(vectors):
        raise RuntimeError("H14 paired evaluator row count changed")
    scores = _probe_scores(head, vectors)
    predicted = np.argmax(scores, axis=1)
    details: list[dict[str, Any]] = []
    for case, control, method_index in zip(cases, control_details, predicted, strict=True):
        predicted_method = _METHOD_IDS[int(method_index)]
        method_correct = predicted_method == str(case["gold_methods"][0])
        columns_correct = bool(control["columns_correct"])
        details.append(
            {
                "case_id": str(case["case_id"]),
                "family_id": str(case["family_id"]),
                "exact_correct": method_correct and columns_correct,
                "method_correct": method_correct,
                "columns_correct": columns_correct,
                "column_recall": float(control["column_recall"]),
                "predicted_methods": [predicted_method],
                "predicted_columns": list(control["predicted_columns"]),
            }
        )
    return details


def _gate_report(
    control: dict[str, Any],
    candidate: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    effects = {
        "exact": 100.0 * (float(candidate["exact_accuracy"]) - float(control["exact_accuracy"])),
        "method": 100.0
        * (float(candidate["method_set_accuracy"]) - float(control["method_set_accuracy"])),
        "columns": 100.0
        * (float(candidate["column_set_accuracy"]) - float(control["column_set_accuracy"])),
    }
    checks = {
        "head_method_accuracy": float(candidate["method_set_accuracy"])
        >= float(gates["minimum_head_method_accuracy"]),
        "head_exact_accuracy": float(candidate["exact_accuracy"])
        >= float(gates["minimum_head_exact_accuracy"]),
        "method_gain": effects["method"] >= float(gates["minimum_method_gain_points"]),
        "exact_gain": effects["exact"] >= float(gates["minimum_exact_gain_points"]),
        "column_noninferior": effects["columns"] >= float(gates["minimum_column_gain_points"]),
    }
    return {"passed": all(checks.values()), "checks": checks, "effect_points": effects}


def _evaluate_surface(
    config: ProjectConfig,
    contract: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    name: str,
) -> dict[str, Any]:
    _, adapter_paths = _expert_context(config)
    parent = adapter_paths["parent"]
    if sha256_file(parent / "adapters.safetensors") != contract["parent"]["adapter_sha256"]:
        raise RuntimeError("H14 parent adapter changed")
    evaluation_fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "surface": name,
            "cases": canonical_hash(cases),
            "evaluator_version": _EVALUATOR_VERSION,
        }
    )
    agent = StatsAgent(config, adapter_path=parent)
    agent.router.set_route("adapter")
    try:
        control = _evaluate_style(
            agent,
            cases,
            name="menu-free-control",
            grounded=False,
            progress_root=_root(config) / "progress" / name,
            evaluation_fingerprint=evaluation_fingerprint,
        )
        vectors, labels, case_ids = _extract_representations(agent, cases, grounded=False)
        expected_ids = [str(case["case_id"]) for case in cases]
        if case_ids != expected_ids:
            raise RuntimeError("H14 representation order changed")
        expected_labels = [str(case["gold_methods"][0]) for case in cases]
        actual_labels = [_METHOD_IDS[int(value)] for value in labels.tolist()]
        if actual_labels != expected_labels:
            raise RuntimeError("H14 representation labels changed")
        candidate_details = _head_details(cases, control["details"], vectors, _load_head(contract))
        return {
            "evaluation_fingerprint": evaluation_fingerprint,
            "control": control,
            "candidate": {
                "format_shift": _format_metrics(candidate_details),
                "details": candidate_details,
            },
        }
    finally:
        del agent
        gc.collect()
        mx.clear_cache()


def run_selector_head_pilot(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_selector_head_contract(config)
    data = prepare_selector_head_data(config)
    cases = list(read_jsonl(_data_root(config) / "cases" / "selection_shard.jsonl"))
    scores = _evaluate_surface(config, contract, cases, name="selection")
    control = scores["control"]["format_shift"]
    candidate = scores["candidate"]["format_shift"]
    gate = _gate_report(control, candidate, dict(contract["settings"]["selection_gates"]))
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "H14 menu-free selector-head pilot",
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["fingerprint"],
        "same_parent_weights": True,
        "scores": {"menu-free-control": control, "selector-head": candidate},
        "selection_gate": gate,
        "selected_architecture": "selector-head" if gate["passed"] else None,
        "confirmation_authorized": bool(gate["passed"]),
        "next_step": "confirm-selector-head" if gate["passed"] else "reject-h14-selector-head",
        "private_details": {
            "menu-free-control": scores["control"]["details"],
            "selector-head": scores["candidate"]["details"],
        },
    }
    report["result_fingerprint"] = canonical_hash(
        {key: value for key, value in report.items() if key != "private_details"}
    )
    write_json(_root(config) / "pilot.json", report)
    public = dict(report)
    public.pop("private_details")
    write_json(config.root / "reports" / "evolve" / "selector-head-v1-pilot.json", public)
    return public


def run_selector_head_confirmation(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_selector_head_contract(config)
    pilot_path = _root(config) / "pilot.json"
    if not pilot_path.exists():
        raise RuntimeError("H14 confirmation requires completed pilot")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if not pilot.get("confirmation_authorized"):
        raise RuntimeError("H14 pilot did not authorize confirmation")
    manifest, simulations = _simulate_surface(config, contract, name="confirmation_shard")
    cases = [_format_shift_case(simulation) for simulation in simulations]
    case_path = _data_root(config) / "cases" / "confirmation_shard.jsonl"
    write_jsonl(case_path, cases)
    scores = _evaluate_surface(config, contract, cases, name="confirmation")
    control = scores["control"]["format_shift"]
    candidate = scores["candidate"]["format_shift"]
    gate = _gate_report(control, candidate, dict(contract["settings"]["confirmation_gates"]))
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "H14 menu-free selector-head confirmation",
        "contract_fingerprint": contract["fingerprint"],
        "pilot_result_fingerprint": pilot["result_fingerprint"],
        "same_parent_weights": True,
        "confirmation_surface_fingerprint": manifest["fingerprint"],
        "confirmation_case_sha256": sha256_file(case_path),
        "scores": {"menu-free-control": control, "selector-head": candidate},
        "confirmation_gate": gate,
        "selector_head_architecture_confirmed": bool(gate["passed"]),
        "champion_changed": False,
        "release_authorized": False,
        "external_benchmark_opened": False,
        "next_step": (
            "freeze-selector-head-runtime-and-seek-new-external-evidence"
            if gate["passed"]
            else "reject-h14-selector-head"
        ),
        "private_details": {
            "menu-free-control": scores["control"]["details"],
            "selector-head": scores["candidate"]["details"],
        },
    }
    report["result_fingerprint"] = canonical_hash(
        {key: value for key, value in report.items() if key != "private_details"}
    )
    write_json(_root(config) / "confirmation.json", report)
    public = dict(report)
    public.pop("private_details")
    write_json(config.root / "reports" / "evolve" / "selector-head-v1-confirmation.json", public)
    return public

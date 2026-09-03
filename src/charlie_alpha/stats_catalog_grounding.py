from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl
from .stats_agent import StatsAgent
from .stats_canonical_bottleneck import _registered_scenarios
from .stats_catalog import PROCEDURES
from .stats_cross_format import (
    _column_recall,
    _format_metrics,
    _format_shift_case,
    _format_shift_messages,
    _set_match,
)
from .stats_dgp import Scenario, simulate_scenario
from .stats_eval import _append_progress, _json_from_answer, _load_progress
from .stats_family_router import _expert_context
from .stats_router_replication import _historical_scenario_audit, _scenario_semantic_payload

_EVALUATOR_VERSION = 1


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "catalog-grounding-v1"


def _data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "catalog-grounding-v1"


def _catalog_reference() -> str:
    return "\n".join(f"{procedure.method_id} — {procedure.name}" for procedure in PROCEDURES)


def _messages(case: dict[str, Any], *, grounded: bool) -> list[dict[str, str]]:
    base = [dict(message) for message in _format_shift_messages(case)]
    if not grounded:
        return base
    base[0]["content"] = (
        base[0]["content"]
        + "\n\nRepository method catalog (fixed and identical for every case):\n"
        + _catalog_reference()
        + "\nSelect exactly one method identifier from this fixed catalog."
    )
    return base


def _implementation_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "stats_catalog_grounding.py": sha256_file(Path(__file__)),
        "stats_cross_format.py": sha256_file(root / "stats_cross_format.py"),
        "stats_agent.py": sha256_file(root / "stats_agent.py"),
        "stats_catalog.py": sha256_file(root / "stats_catalog.py"),
    }


def _h6_registered_scenarios(config: ProjectConfig) -> list[Scenario]:
    path = config.root / "reports" / "evolve" / "canonical-bottleneck-v1-contract.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    scenarios: list[Scenario] = []
    for name in ("training_pool", "selection_shard", "confirmation_shard"):
        scenarios.extend(_registered_scenarios(dict(contract["settings"][name]), name=name))
    return scenarios


def prepare_catalog_grounding_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("catalog_grounding"))
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "contract.json"
    public_path = config.root / "reports" / "evolve" / "catalog-grounding-v1-contract.json"

    h6_path = config.root / "reports" / "evolve" / "canonical-bottleneck-v1-pilot.json"
    if not h6_path.exists():
        raise RuntimeError("H7 requires the closed H6 pilot")
    h6 = json.loads(h6_path.read_text(encoding="utf-8"))
    if h6.get("selected_arm") is not None or h6.get("confirmation_authorized"):
        raise RuntimeError("H6 did not close negatively; H7 is not authorized")
    h6_confirmation = (
        config.path_for("evolution_dir")
        / "canonical-bottleneck-v1"
        / "surfaces"
        / "confirmation_shard.jsonl"
    )
    if h6_confirmation.exists():
        raise RuntimeError("H6 confirmation was unexpectedly opened")

    _, adapter_paths = _expert_context(config)
    parent = adapter_paths["parent"]
    parent_sha = sha256_file(parent / "adapters.safetensors")
    registered: dict[str, Any] = {}
    new_scenarios: list[Scenario] = []
    seen: set[str] = set()
    for name in ("selection_shard", "confirmation_shard"):
        shard = dict(settings[name])
        scenarios = _registered_scenarios(shard, name=f"catalog-grounding:{name}")
        ids = {scenario.blueprint_id for scenario in scenarios}
        if ids & seen:
            raise RuntimeError(f"H7 blueprint overlap at {name}")
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

    h6_scenarios = _h6_registered_scenarios(config)
    h6_ids = {scenario.blueprint_id for scenario in h6_scenarios}
    h6_semantics = {
        canonical_hash(_scenario_semantic_payload(scenario.to_dict())) for scenario in h6_scenarios
    }
    new_ids = {scenario.blueprint_id for scenario in new_scenarios}
    new_semantics = {
        canonical_hash(_scenario_semantic_payload(scenario.to_dict())) for scenario in new_scenarios
    }
    if new_ids & h6_ids or new_semantics & h6_semantics:
        raise RuntimeError("H7 overlaps H6 registered blueprints or semantic points")

    audit = _historical_scenario_audit(
        config,
        new_scenarios,
        excluded_root=_data_root(config),
        minimum_normalized_distance=float(settings["minimum_normalized_distance"]),
    )
    if not audit["passed"]:
        raise RuntimeError("H7 blueprints failed historical-overlap audit")

    catalog = [(procedure.method_id, procedure.name) for procedure in PROCEDURES]
    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": "H7 fixed catalog grounding",
        "method_version": int(settings["method_version"]),
        "causal_question": (
            "Does adding one fixed repository-wide method catalog to the unchanged menu-free "
            "canonical extraction prompt unlock fine-grained method selection on fresh cases?"
        ),
        "h6_negative_result_fingerprint": h6["result_fingerprint"],
        "h6_report_sha256": sha256_file(h6_path),
        "parent": {
            "name": "v0.3.0-parent",
            "adapter_path": str(parent),
            "adapter_sha256": parent_sha,
        },
        "settings": settings,
        "blueprint_contracts": registered,
        "historical_overlap_audit": audit,
        "h6_registered_overlap": {
            "h6_blueprints": len(h6_ids),
            "h7_blueprints": len(new_ids),
            "blueprint_id_overlap_count": len(new_ids & h6_ids),
            "semantic_overlap_count": len(new_semantics & h6_semantics),
        },
        "catalog": {
            "procedure_count": len(catalog),
            "sha256": canonical_hash(catalog),
            "source": "stats_catalog.PROCEDURES method_id + existing display name only",
        },
        "implementation_sha256": _implementation_manifest(),
        "arms": {
            "menu-free-control": "Unchanged H6 menu-free canonical extraction prompt",
            "catalog-grounded": (
                "Same prompt and parent plus one fixed 28-method repository catalog for every case"
            ),
        },
        "selection_policy": (
            "Only catalog-grounded may advance and only if every registered selection gate passes."
        ),
        "confirmation_policy": (
            "Confirmation blueprints are registered now, but simulations remain unopened until "
            "selection passes. H6 confirmation and historical P-Bench/StatQA remain sealed."
        ),
        "claim_boundary": (
            "H7 can establish a synthetic prompt-interface mechanism only. It is not evidence of "
            "new model weights or independent external capability."
        ),
        "selection_opened": False,
        "confirmation_simulations_opened": False,
        "external_benchmark_opened": False,
    }
    contract["fingerprint"] = canonical_hash(contract)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("H7 contract is immutable")
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
            raise RuntimeError("H7 confirmation cannot open before selection")
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        if pilot.get("selected_interface") != "catalog-grounded":
            raise RuntimeError("H7 selection did not authorize confirmation")
    shard = dict(contract["settings"][name])
    scenarios = _registered_scenarios(shard, name=f"catalog-grounding:{name}")
    blueprint_sha = canonical_hash([scenario.to_dict() for scenario in scenarios])
    if blueprint_sha != contract["blueprint_contracts"][name]["blueprint_sha256"]:
        raise RuntimeError(f"H7 registered blueprints changed for {name}")
    path = _data_root(config) / "surfaces" / f"{name}.jsonl"
    manifest_path = path.with_suffix(".manifest.json")
    stats = config.section("stats_data")
    fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "name": name,
            "blueprints": blueprint_sha,
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
        if manifest.get("fingerprint") != fingerprint or manifest.get("sha256") != sha256_file(
            path
        ):
            raise RuntimeError(f"H7 {name} surface changed")
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


def prepare_catalog_grounding_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_catalog_grounding_contract(config)
    root = _data_root(config)
    status_path = root / "data-status.json"
    selection_manifest, selection_surface = _simulate_surface(
        config, contract, name="selection_shard"
    )
    cases = [_format_shift_case(simulation) for simulation in selection_surface]
    cases_path = root / "selection-format.jsonl"
    write_jsonl(cases_path, cases)
    fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "selection_surface": selection_manifest["fingerprint"],
            "cases": sha256_file(cases_path),
        }
    )
    status = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "contract_fingerprint": contract["fingerprint"],
        "selection_groups": len(selection_surface),
        "cases_sha256": sha256_file(cases_path),
        "confirmation_opened": False,
    }
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise RuntimeError("H7 prepared data changed")
        return existing
    write_json(status_path, status)
    write_json(config.root / "reports" / "evolve" / "catalog-grounding-v1-data.json", status)
    return status


def _evaluate_style(
    agent: StatsAgent,
    cases: list[dict[str, Any]],
    *,
    name: str,
    grounded: bool,
    progress_root: Path,
    evaluation_fingerprint: str,
) -> dict[str, Any]:
    progress_path = progress_root / f"{name}.jsonl"
    fingerprint = canonical_hash(
        {
            "evaluation": evaluation_fingerprint,
            "name": name,
            "grounded": grounded,
            "evaluator_version": _EVALUATOR_VERSION,
        }
    )
    cached = _load_progress(progress_path, fingerprint=fingerprint, id_field="case_id")
    details: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        if case_id in cached:
            details.append(cached[case_id])
            continue
        answer = agent.answer_without_tools(
            _messages(case, grounded=grounded),
            route="stats",
            max_tokens=160,
            temperature=0.0,
        )
        parsed = _json_from_answer(answer)
        method_correct = _set_match(list(case["gold_methods"]), parsed.get("methods"))
        columns_correct = _set_match(list(case["gold_columns"]), parsed.get("columns"))
        row = {
            "case_id": case_id,
            "family_id": str(case["family_id"]),
            "exact_correct": method_correct and columns_correct,
            "method_correct": method_correct,
            "columns_correct": columns_correct,
            "column_recall": _column_recall(list(case["gold_columns"]), parsed.get("columns")),
            "predicted_methods": parsed.get("methods", []),
            "predicted_columns": parsed.get("columns", []),
        }
        details.append(row)
        _append_progress(
            progress_path,
            fingerprint=fingerprint,
            row=row,
            completed=len(details),
        )
    return {"format_shift": _format_metrics(details), "details": details}


def _gate_report(
    *,
    control: dict[str, Any],
    candidate: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    tolerance = 1e-9
    c = control["format_shift"]
    x = candidate["format_shift"]
    effects = {
        "exact_vs_control": 100 * (float(x["exact_accuracy"]) - float(c["exact_accuracy"])),
        "method_vs_control": 100
        * (float(x["method_set_accuracy"]) - float(c["method_set_accuracy"])),
        "columns_vs_control": 100
        * (float(x["column_set_accuracy"]) - float(c["column_set_accuracy"])),
    }
    checks = {
        "exact_gain_over_control": effects["exact_vs_control"]
        >= float(gates["minimum_exact_gain_over_control_points"]) - tolerance,
        "method_gain_over_control": effects["method_vs_control"]
        >= float(gates["minimum_method_gain_over_control_points"]) - tolerance,
        "column_noninferior": effects["columns_vs_control"]
        >= float(gates["minimum_column_gain_over_control_points"]) - tolerance,
    }
    return {"passed": all(checks.values()), "checks": checks, "effect_points": effects}


def run_catalog_grounding_pilot(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_catalog_grounding_contract(config)
    data = prepare_catalog_grounding_data(config)
    cases = list(read_jsonl(_data_root(config) / "selection-format.jsonl"))
    _, adapter_paths = _expert_context(config)
    parent = adapter_paths["parent"]
    parent_sha = sha256_file(parent / "adapters.safetensors")
    if parent_sha != contract["parent"]["adapter_sha256"]:
        raise RuntimeError("H7 parent adapter changed")
    evaluation_fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "data": data["fingerprint"],
            "parent": parent_sha,
            "surface": "selection",
            "evaluator_version": _EVALUATOR_VERSION,
        }
    )
    agent = StatsAgent(config, adapter_path=parent)
    agent.router.set_route("adapter")
    try:
        progress_root = _root(config) / "selection-progress"
        scores = {
            "menu-free-control": _evaluate_style(
                agent,
                cases,
                name="menu-free-control",
                grounded=False,
                progress_root=progress_root,
                evaluation_fingerprint=evaluation_fingerprint,
            ),
            "catalog-grounded": _evaluate_style(
                agent,
                cases,
                name="catalog-grounded",
                grounded=True,
                progress_root=progress_root,
                evaluation_fingerprint=evaluation_fingerprint,
            ),
        }
    finally:
        del agent
        gc.collect()
        mx.clear_cache()
    gate = _gate_report(
        control=scores["menu-free-control"],
        candidate=scores["catalog-grounded"],
        gates=dict(contract["settings"]["selection_gates"]),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": evaluation_fingerprint,
        "method": "H7 fixed catalog grounding pilot",
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["fingerprint"],
        "same_parent_weights": True,
        "scores": {name: value["format_shift"] for name, value in scores.items()},
        "selection_gate": gate,
        "selected_interface": "catalog-grounded" if gate["passed"] else None,
        "confirmation_authorized": bool(gate["passed"]),
        "external_benchmark_authorized": False,
        "private_details": {name: value["details"] for name, value in scores.items()},
    }
    report["result_fingerprint"] = canonical_hash(
        {key: value for key, value in report.items() if key != "private_details"}
    )
    write_json(_root(config) / "pilot.json", report)
    public = dict(report)
    public.pop("private_details")
    write_json(config.root / "reports" / "evolve" / "catalog-grounding-v1-pilot.json", public)
    return public


def run_catalog_grounding_confirmation(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_catalog_grounding_contract(config)
    pilot_path = _root(config) / "pilot.json"
    if not pilot_path.exists():
        raise RuntimeError("H7 pilot has not selected an interface")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("selected_interface") != "catalog-grounded":
        raise RuntimeError("H7 selection did not authorize confirmation")
    manifest, surface = _simulate_surface(config, contract, name="confirmation_shard")
    cases = [_format_shift_case(simulation) for simulation in surface]
    _, adapter_paths = _expert_context(config)
    parent = adapter_paths["parent"]
    parent_sha = sha256_file(parent / "adapters.safetensors")
    evaluation_fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "pilot": pilot["result_fingerprint"],
            "confirmation": manifest["fingerprint"],
            "parent": parent_sha,
            "evaluator_version": _EVALUATOR_VERSION,
        }
    )
    agent = StatsAgent(config, adapter_path=parent)
    agent.router.set_route("adapter")
    try:
        progress_root = _root(config) / "confirmation-progress"
        scores = {
            "menu-free-control": _evaluate_style(
                agent,
                cases,
                name="menu-free-control",
                grounded=False,
                progress_root=progress_root,
                evaluation_fingerprint=evaluation_fingerprint,
            ),
            "catalog-grounded": _evaluate_style(
                agent,
                cases,
                name="catalog-grounded",
                grounded=True,
                progress_root=progress_root,
                evaluation_fingerprint=evaluation_fingerprint,
            ),
        }
    finally:
        del agent
        gc.collect()
        mx.clear_cache()
    gate = _gate_report(
        control=scores["menu-free-control"],
        candidate=scores["catalog-grounded"],
        gates=dict(contract["settings"]["confirmation_gates"]),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": evaluation_fingerprint,
        "method": "H7 fixed catalog grounding confirmation",
        "contract_fingerprint": contract["fingerprint"],
        "pilot_result_fingerprint": pilot["result_fingerprint"],
        "confirmation_manifest_fingerprint": manifest["fingerprint"],
        "same_parent_weights": True,
        "scores": {name: value["format_shift"] for name, value in scores.items()},
        "confirmation_gate": gate,
        "synthetic_interface_confirmed": bool(gate["passed"]),
        "external_benchmark_authorized": False,
        "next_step": (
            "preregister-independent-external-interface-evidence"
            if gate["passed"]
            else "reject-h7-catalog-grounding"
        ),
        "private_details": {name: value["details"] for name, value in scores.items()},
    }
    report["result_fingerprint"] = canonical_hash(
        {key: value for key, value in report.items() if key != "private_details"}
    )
    write_json(_root(config) / "confirmation.json", report)
    public = dict(report)
    public.pop("private_details")
    write_json(
        config.root / "reports" / "evolve" / "catalog-grounding-v1-confirmation.json",
        public,
    )
    return public

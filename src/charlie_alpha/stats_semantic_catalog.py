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
from .stats_catalog_distillation import _contract_scenarios
from .stats_catalog_grounding import _messages as _flat_messages
from .stats_cross_format import (
    _column_recall,
    _format_metrics,
    _format_shift_case,
    _format_shift_messages,
    _set_match,
)
from .stats_dgp import simulate_scenario
from .stats_eval import _append_progress, _json_from_answer, _load_progress
from .stats_family_router import _expert_context
from .stats_router_replication import _historical_scenario_audit, _scenario_semantic_payload

_EVALUATOR_VERSION = 1


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "semantic-catalog-v1"


def _data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "semantic-catalog-v1"


def _semantic_catalog_reference() -> str:
    lines: list[str] = []
    for procedure in PROCEDURES:
        assumptions = "; ".join(procedure.assumptions)
        strengths = "; ".join(procedure.strengths)
        lines.append(
            f"{procedure.method_id} — {procedure.name} | assumptions: {assumptions} | "
            f"strengths: {strengths} | uncertainty: {procedure.uncertainty}"
        )
    return "\n".join(lines)


def _messages(case: dict[str, Any], *, semantic: bool) -> list[dict[str, str]]:
    if not semantic:
        return _flat_messages(case, grounded=True)
    base = [dict(message) for message in _format_shift_messages(case)]
    base[0]["content"] = (
        base[0]["content"]
        + "\n\nRepository method catalog (fixed and identical for every case):\n"
        + _semantic_catalog_reference()
        + "\nSelect exactly one method identifier from this fixed catalog."
    )
    return base


def _implementation_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "stats_semantic_catalog.py": sha256_file(Path(__file__)),
        "stats_catalog.py": sha256_file(root / "stats_catalog.py"),
        "stats_catalog_grounding.py": sha256_file(root / "stats_catalog_grounding.py"),
        "stats_cross_format.py": sha256_file(root / "stats_cross_format.py"),
        "stats_agent.py": sha256_file(root / "stats_agent.py"),
    }


def prepare_semantic_catalog_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("semantic_catalog"))
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "contract.json"
    public_path = config.root / "reports" / "evolve" / "semantic-catalog-v1-contract.json"

    h10_path = config.root / "reports" / "evolve" / "output-factorization-v1-confirmation.json"
    if not h10_path.exists():
        raise RuntimeError("H11 requires the closed H10 confirmation")
    h10 = json.loads(h10_path.read_text(encoding="utf-8"))
    if h10.get("synthetic_factorization_confirmed") or h10["confirmation_gate"]["passed"]:
        raise RuntimeError("H10 did not close negatively; H11 is not authorized")
    if h10.get("external_benchmark_authorized"):
        raise RuntimeError("H10 unexpectedly authorized an external benchmark")

    h7_path = config.root / "reports" / "evolve" / "catalog-grounding-v1-confirmation.json"
    h7 = json.loads(h7_path.read_text(encoding="utf-8"))
    if not h7.get("synthetic_interface_confirmed"):
        raise RuntimeError("H11 requires the confirmed H7 fixed-catalog mechanism")

    _, adapter_paths = _expert_context(config)
    parent = adapter_paths["parent"]
    parent_sha = sha256_file(parent / "adapters.safetensors")
    registered: dict[str, Any] = {}
    all_scenarios = []
    seen: set[str] = set()
    for name in ("selection_shard", "confirmation_shard"):
        shard = dict(settings[name])
        scenarios = _registered_scenarios(shard, name=f"semantic-catalog:{name}")
        ids = {scenario.blueprint_id for scenario in scenarios}
        if ids & seen:
            raise RuntimeError(f"H11 blueprint overlap at {name}")
        seen.update(ids)
        all_scenarios.extend(scenarios)
        registered[name] = {
            "split": str(shard["split"]),
            "seed": int(shard["seed"]),
            "pool_count": int(shard["pool_count"]),
            "selected_per_family": int(shard["selected_per_family"]),
            "count": len(scenarios),
            "blueprint_sha256": canonical_hash([scenario.to_dict() for scenario in scenarios]),
        }

    previous_scenarios = (
        _contract_scenarios(config, "canonical-bottleneck-v1-contract.json")
        + _contract_scenarios(
            config,
            "catalog-grounding-v1-contract.json",
            name_prefix="catalog-grounding:",
        )
        + _contract_scenarios(
            config,
            "catalog-distillation-v1-contract.json",
            name_prefix="catalog-distillation:",
        )
        + _contract_scenarios(
            config,
            "catalog-ranking-v1-contract.json",
            name_prefix="catalog-ranking:",
        )
        + _contract_scenarios(
            config,
            "output-factorization-v1-contract.json",
            name_prefix="output-factorization:",
        )
    )
    prior_ids = {scenario.blueprint_id for scenario in previous_scenarios}
    prior_semantics = {
        canonical_hash(_scenario_semantic_payload(scenario.to_dict()))
        for scenario in previous_scenarios
    }
    new_ids = {scenario.blueprint_id for scenario in all_scenarios}
    new_semantics = {
        canonical_hash(_scenario_semantic_payload(scenario.to_dict())) for scenario in all_scenarios
    }
    if new_ids & prior_ids or new_semantics & prior_semantics:
        raise RuntimeError("H11 overlaps prior registered blueprints or semantic points")

    audit = _historical_scenario_audit(
        config,
        all_scenarios,
        excluded_root=_data_root(config),
        minimum_normalized_distance=float(settings["minimum_normalized_distance"]),
    )
    if not audit["passed"]:
        raise RuntimeError("H11 blueprints failed historical-overlap audit")

    semantic_catalog = _semantic_catalog_reference()
    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": "H11 semantic fixed-catalog grounding",
        "method_version": int(settings["method_version"]),
        "causal_question": (
            "Does replacing the confirmed H7 method_id+name catalog with a fixed repository-wide "
            "semantic catalog containing each method's existing assumptions, strengths, and "
            "uncertainty description improve fine-grained method selection?"
        ),
        "h10_negative_result_fingerprint": h10["result_fingerprint"],
        "h10_report_sha256": sha256_file(h10_path),
        "h7_confirmation_result_fingerprint": h7["result_fingerprint"],
        "h7_confirmed_scores": h7["scores"]["catalog-grounded"],
        "parent": {
            "name": "v0.3.0-parent",
            "adapter_path": str(parent),
            "adapter_sha256": parent_sha,
        },
        "settings": settings,
        "blueprint_contracts": registered,
        "historical_overlap_audit": audit,
        "prior_registered_overlap": {
            "prior_blueprints": len(prior_ids),
            "h11_blueprints": len(new_ids),
            "blueprint_id_overlap_count": len(new_ids & prior_ids),
            "semantic_overlap_count": len(new_semantics & prior_semantics),
        },
        "catalog": {
            "procedure_count": len(PROCEDURES),
            "semantic_reference_sha256": canonical_hash(semantic_catalog),
            "fields": ["method_id", "name", "assumptions", "strengths", "uncertainty"],
            "case_specific_content": False,
        },
        "implementation_sha256": _implementation_manifest(),
        "arms": {
            "flat-catalog": "H7 fixed method_id + display-name catalog",
            "semantic-catalog": (
                "Same parent, question, JSON contract, and generation settings; replace only the "
                "flat catalog entries with existing static assumptions/strengths/uncertainty cards"
            ),
        },
        "selection_policy": (
            "Only semantic-catalog may advance and only if every registered selection gate passes."
        ),
        "confirmation_policy": (
            "Confirmation blueprints are registered now but remain unsimulated until selection "
            "passes. Historical P-Bench/StatQA remain unavailable."
        ),
        "claim_boundary": (
            "H11 can establish a synthetic semantic-interface mechanism only. It changes no "
            "weights and does not establish external capability."
        ),
        "selection_opened": False,
        "confirmation_simulations_opened": False,
        "external_benchmark_opened": False,
    }
    contract["fingerprint"] = canonical_hash(contract)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("H11 contract is immutable")
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
            raise RuntimeError("H11 confirmation cannot open before selection")
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        if pilot.get("selected_interface") != "semantic-catalog":
            raise RuntimeError("H11 selection did not authorize confirmation")
    shard = dict(contract["settings"][name])
    scenarios = _registered_scenarios(shard, name=f"semantic-catalog:{name}")
    blueprint_sha = canonical_hash([scenario.to_dict() for scenario in scenarios])
    if blueprint_sha != contract["blueprint_contracts"][name]["blueprint_sha256"]:
        raise RuntimeError(f"H11 registered blueprints changed for {name}")
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
            raise RuntimeError(f"H11 {name} surface changed")
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


def prepare_semantic_catalog_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_semantic_catalog_contract(config)
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
            raise RuntimeError("H11 prepared data changed")
        return existing
    write_json(status_path, status)
    write_json(config.root / "reports" / "evolve" / "semantic-catalog-v1-data.json", status)
    return status


def _evaluate_style(
    agent: StatsAgent,
    cases: list[dict[str, Any]],
    *,
    name: str,
    semantic: bool,
    progress_root: Path,
    evaluation_fingerprint: str,
) -> dict[str, Any]:
    progress_path = progress_root / f"{name}.jsonl"
    fingerprint = canonical_hash(
        {
            "evaluation": evaluation_fingerprint,
            "name": name,
            "semantic": semantic,
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
            _messages(case, semantic=semantic),
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
        "minimum_semantic_method_accuracy": float(x["method_set_accuracy"])
        >= float(gates["minimum_semantic_method_accuracy"]) - tolerance,
        "minimum_semantic_exact_accuracy": float(x["exact_accuracy"])
        >= float(gates["minimum_semantic_exact_accuracy"]) - tolerance,
        "method_gain_over_flat": effects["method_vs_control"]
        >= float(gates["minimum_method_gain_over_flat_points"]) - tolerance,
        "exact_gain_over_flat": effects["exact_vs_control"]
        >= float(gates["minimum_exact_gain_over_flat_points"]) - tolerance,
        "column_noninferior": effects["columns_vs_control"]
        >= float(gates["minimum_column_gain_over_flat_points"]) - tolerance,
    }
    return {"passed": all(checks.values()), "checks": checks, "effect_points": effects}


def _run_evaluation(
    config: ProjectConfig,
    *,
    cases: list[dict[str, Any]],
    evaluation_fingerprint: str,
    progress_root: Path,
) -> dict[str, Any]:
    _, adapter_paths = _expert_context(config)
    agent = StatsAgent(config, adapter_path=adapter_paths["parent"])
    agent.router.set_route("adapter")
    try:
        return {
            "flat-catalog": _evaluate_style(
                agent,
                cases,
                name="flat-catalog",
                semantic=False,
                progress_root=progress_root,
                evaluation_fingerprint=evaluation_fingerprint,
            ),
            "semantic-catalog": _evaluate_style(
                agent,
                cases,
                name="semantic-catalog",
                semantic=True,
                progress_root=progress_root,
                evaluation_fingerprint=evaluation_fingerprint,
            ),
        }
    finally:
        del agent
        gc.collect()
        mx.clear_cache()


def run_semantic_catalog_pilot(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_semantic_catalog_contract(config)
    data = prepare_semantic_catalog_data(config)
    cases = list(read_jsonl(_data_root(config) / "selection-format.jsonl"))
    evaluation_fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "data": data["fingerprint"],
            "parent": contract["parent"]["adapter_sha256"],
            "surface": "selection",
            "evaluator_version": _EVALUATOR_VERSION,
        }
    )
    scores = _run_evaluation(
        config,
        cases=cases,
        evaluation_fingerprint=evaluation_fingerprint,
        progress_root=_root(config) / "selection-progress",
    )
    gate = _gate_report(
        control=scores["flat-catalog"],
        candidate=scores["semantic-catalog"],
        gates=dict(contract["settings"]["selection_gates"]),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": evaluation_fingerprint,
        "method": "H11 semantic fixed-catalog grounding pilot",
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["fingerprint"],
        "same_parent_weights": True,
        "scores": {name: value["format_shift"] for name, value in scores.items()},
        "selection_gate": gate,
        "selected_interface": "semantic-catalog" if gate["passed"] else None,
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
    write_json(config.root / "reports" / "evolve" / "semantic-catalog-v1-pilot.json", public)
    return public


def run_semantic_catalog_confirmation(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_semantic_catalog_contract(config)
    pilot_path = _root(config) / "pilot.json"
    if not pilot_path.exists():
        raise RuntimeError("H11 pilot has not selected an interface")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("selected_interface") != "semantic-catalog":
        raise RuntimeError("H11 selection did not authorize confirmation")
    manifest, surface = _simulate_surface(config, contract, name="confirmation_shard")
    cases = [_format_shift_case(simulation) for simulation in surface]
    evaluation_fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "pilot": pilot["result_fingerprint"],
            "confirmation": manifest["fingerprint"],
            "parent": contract["parent"]["adapter_sha256"],
            "evaluator_version": _EVALUATOR_VERSION,
        }
    )
    scores = _run_evaluation(
        config,
        cases=cases,
        evaluation_fingerprint=evaluation_fingerprint,
        progress_root=_root(config) / "confirmation-progress",
    )
    gate = _gate_report(
        control=scores["flat-catalog"],
        candidate=scores["semantic-catalog"],
        gates=dict(contract["settings"]["confirmation_gates"]),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": evaluation_fingerprint,
        "method": "H11 semantic fixed-catalog grounding confirmation",
        "contract_fingerprint": contract["fingerprint"],
        "pilot_result_fingerprint": pilot["result_fingerprint"],
        "confirmation_manifest_fingerprint": manifest["fingerprint"],
        "same_parent_weights": True,
        "scores": {name: value["format_shift"] for name, value in scores.items()},
        "confirmation_gate": gate,
        "synthetic_semantic_interface_confirmed": bool(gate["passed"]),
        "external_benchmark_authorized": False,
        "next_step": (
            "preregister-independent-external-semantic-interface-evidence"
            if gate["passed"]
            else "reject-h11-semantic-catalog"
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
        config.root / "reports" / "evolve" / "semantic-catalog-v1-confirmation.json",
        public,
    )
    return public

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl
from .stats_agent import StatsAgent
from .stats_canonical_bottleneck import _registered_scenarios
from .stats_catalog_distillation import _contract_scenarios
from .stats_catalog_grounding import _messages as _joint_messages
from .stats_catalog_ranking import _clean_generated_method
from .stats_catalog_ranking import _messages as _method_messages
from .stats_cross_format import _column_recall, _format_shift_case, _set_match
from .stats_dgp import simulate_scenario
from .stats_eval import _append_progress, _json_from_answer, _load_progress, _normalize
from .stats_family_router import _expert_context
from .stats_router_replication import _historical_scenario_audit, _scenario_semantic_payload

_EVALUATOR_VERSION = 1


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "output-factorization-v1"


def _data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "output-factorization-v1"


def _implementation_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "stats_output_factorization.py": sha256_file(Path(__file__)),
        "stats_catalog_grounding.py": sha256_file(root / "stats_catalog_grounding.py"),
        "stats_catalog_ranking.py": sha256_file(root / "stats_catalog_ranking.py"),
        "stats_agent.py": sha256_file(root / "stats_agent.py"),
        "stats_cross_format.py": sha256_file(root / "stats_cross_format.py"),
    }


def prepare_output_factorization_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("output_factorization"))
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "contract.json"
    public_path = config.root / "reports" / "evolve" / "output-factorization-v1-contract.json"

    h9_path = config.root / "reports" / "evolve" / "catalog-ranking-v1-pilot.json"
    if not h9_path.exists():
        raise RuntimeError("H10 requires the closed H9 pilot")
    h9 = json.loads(h9_path.read_text(encoding="utf-8"))
    if h9.get("selected_interface") is not None or h9.get("confirmation_authorized"):
        raise RuntimeError("H9 did not close negatively; H10 is not authorized")
    h9_confirmation = (
        config.path_for("evolution_dir")
        / "catalog-ranking-v1"
        / "surfaces"
        / "confirmation_shard.jsonl"
    )
    if h9_confirmation.exists():
        raise RuntimeError("H9 confirmation was unexpectedly opened")

    h7_path = config.root / "reports" / "evolve" / "catalog-grounding-v1-confirmation.json"
    h7 = json.loads(h7_path.read_text(encoding="utf-8"))
    if not h7.get("synthetic_interface_confirmed"):
        raise RuntimeError("H10 requires the confirmed H7 joint catalog interface")

    _, adapter_paths = _expert_context(config)
    parent = adapter_paths["parent"]
    parent_sha = sha256_file(parent / "adapters.safetensors")
    registered: dict[str, Any] = {}
    all_scenarios = []
    seen: set[str] = set()
    for name in ("selection_shard", "confirmation_shard"):
        shard = dict(settings[name])
        scenarios = _registered_scenarios(shard, name=f"output-factorization:{name}")
        ids = {scenario.blueprint_id for scenario in scenarios}
        if ids & seen:
            raise RuntimeError(f"H10 blueprint overlap at {name}")
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
        raise RuntimeError("H10 overlaps H6/H7/H8/H9 registered blueprints or semantic points")

    audit = _historical_scenario_audit(
        config,
        all_scenarios,
        excluded_root=_data_root(config),
        minimum_normalized_distance=float(settings["minimum_normalized_distance"]),
    )
    if not audit["passed"]:
        raise RuntimeError("H10 blueprints failed historical-overlap audit")

    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": "H10 factorized method and column interface",
        "method_version": int(settings["method_version"]),
        "causal_question": (
            "Does separating fixed-catalog method selection from joint method+column JSON "
            "generation "
            "improve held-out exact extraction when both interfaces use the unchanged v0.3 parent?"
        ),
        "h9_negative_result_fingerprint": h9["result_fingerprint"],
        "h9_report_sha256": sha256_file(h9_path),
        "h9_free_method_accuracy": float(h9["scores"]["free_method_accuracy"]),
        "h7_confirmation_result_fingerprint": h7["result_fingerprint"],
        "h7_joint_exact_accuracy": float(h7["scores"]["catalog-grounded"]["exact_accuracy"]),
        "h7_joint_method_accuracy": float(h7["scores"]["catalog-grounded"]["method_set_accuracy"]),
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
            "h10_blueprints": len(new_ids),
            "blueprint_id_overlap_count": len(new_ids & prior_ids),
            "semantic_overlap_count": len(new_semantics & prior_semantics),
        },
        "implementation_sha256": _implementation_manifest(),
        "arms": {
            "joint-catalog-json": (
                "H7 fixed-catalog JSON output supplies both method and columns in one generation"
            ),
            "factorized-method": (
                "Use the exact same joint generation for columns, but replace only its method with "
                "an independent H9 method-only fixed-catalog generation"
            ),
        },
        "causal_isolation": (
            "Candidate and control share identical generated columns case-by-case. The only "
            "changed "
            "output component is the method selection path."
        ),
        "selection_policy": (
            "Only factorized-method may advance and only if every registered method/exact gate "
            "passes. H9 confirmation and historical external benchmarks remain sealed."
        ),
        "confirmation_policy": (
            "Confirmation blueprints are registered now but remain unsimulated until selection "
            "passes."
        ),
        "claim_boundary": (
            "H10 can establish a synthetic factorized-interface mechanism only. It does not change "
            "weights, promote the champion, or establish external capability."
        ),
        "selection_opened": False,
        "confirmation_simulations_opened": False,
        "external_benchmark_opened": False,
    }
    contract["fingerprint"] = canonical_hash(contract)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("H10 contract is immutable")
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
            raise RuntimeError("H10 confirmation cannot open before selection")
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        if pilot.get("selected_interface") != "factorized-method":
            raise RuntimeError("H10 selection did not authorize confirmation")
    shard = dict(contract["settings"][name])
    scenarios = _registered_scenarios(shard, name=f"output-factorization:{name}")
    blueprint_sha = canonical_hash([scenario.to_dict() for scenario in scenarios])
    if blueprint_sha != contract["blueprint_contracts"][name]["blueprint_sha256"]:
        raise RuntimeError(f"H10 registered blueprints changed for {name}")
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
            raise RuntimeError(f"H10 {name} surface changed")
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


def prepare_output_factorization_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_output_factorization_contract(config)
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
            raise RuntimeError("H10 prepared data changed")
        return existing
    write_json(status_path, status)
    write_json(config.root / "reports" / "evolve" / "output-factorization-v1-data.json", status)
    return status


def _evaluate_cases(
    config: ProjectConfig,
    cases: list[dict[str, Any]],
    *,
    progress_root: Path,
    evaluation_fingerprint: str,
) -> dict[str, Any]:
    progress_path = progress_root / "paired.jsonl"
    fingerprint = canonical_hash(
        {"evaluation": evaluation_fingerprint, "evaluator_version": _EVALUATOR_VERSION}
    )
    cached = _load_progress(progress_path, fingerprint=fingerprint, id_field="case_id")
    _, adapter_paths = _expert_context(config)
    agent = StatsAgent(config, adapter_path=adapter_paths["parent"])
    agent.router.set_route("adapter")
    details: list[dict[str, Any]] = []
    try:
        for case in cases:
            case_id = str(case["case_id"])
            if case_id in cached:
                details.append(cached[case_id])
                continue
            joint_answer = agent.answer_without_tools(
                _joint_messages(case, grounded=True),
                route="stats",
                max_tokens=int(config.section("output_factorization")["joint_max_tokens"]),
                temperature=0.0,
            )
            parsed = _json_from_answer(joint_answer)
            joint_methods = parsed.get("methods")
            joint_columns = parsed.get("columns")
            method_answer = agent.answer_without_tools(
                _method_messages(case),
                route="stats",
                max_tokens=int(config.section("output_factorization")["method_max_tokens"]),
                temperature=0.0,
            )
            factorized_method = _clean_generated_method(method_answer)
            gold_method = str(case["gold_methods"][0])
            gold_columns = list(case["gold_columns"])
            joint_method_correct = _set_match([gold_method], joint_methods)
            columns_correct = _set_match(gold_columns, joint_columns)
            factorized_method_correct = _normalize(factorized_method) == _normalize(gold_method)
            row = {
                "case_id": case_id,
                "family_id": str(case["family_id"]),
                "gold_method": gold_method,
                "joint_methods": joint_methods if isinstance(joint_methods, list) else [],
                "factorized_method": factorized_method,
                "joint_columns": joint_columns if isinstance(joint_columns, list) else [],
                "joint_method_correct": joint_method_correct,
                "factorized_method_correct": factorized_method_correct,
                "columns_correct": columns_correct,
                "column_recall": _column_recall(gold_columns, joint_columns),
                "joint_exact_correct": joint_method_correct and columns_correct,
                "factorized_exact_correct": factorized_method_correct and columns_correct,
            }
            details.append(row)
            _append_progress(
                progress_path,
                fingerprint=fingerprint,
                row=row,
                completed=len(details),
            )
    finally:
        del agent
        gc.collect()
        mx.clear_cache()
    count = len(details)
    return {
        "count": count,
        "joint_method_accuracy": sum(bool(row["joint_method_correct"]) for row in details) / count,
        "factorized_method_accuracy": sum(bool(row["factorized_method_correct"]) for row in details)
        / count,
        "column_set_accuracy": sum(bool(row["columns_correct"]) for row in details) / count,
        "mean_column_recall": float(np.mean([float(row["column_recall"]) for row in details])),
        "joint_exact_accuracy": sum(bool(row["joint_exact_correct"]) for row in details) / count,
        "factorized_exact_accuracy": sum(bool(row["factorized_exact_correct"]) for row in details)
        / count,
        "details": details,
    }


def _gate_report(scores: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    method_gain = 100 * (
        float(scores["factorized_method_accuracy"]) - float(scores["joint_method_accuracy"])
    )
    exact_gain = 100 * (
        float(scores["factorized_exact_accuracy"]) - float(scores["joint_exact_accuracy"])
    )
    checks = {
        "minimum_factorized_method_accuracy": float(scores["factorized_method_accuracy"])
        >= float(gates["minimum_factorized_method_accuracy"]),
        "minimum_factorized_exact_accuracy": float(scores["factorized_exact_accuracy"])
        >= float(gates["minimum_factorized_exact_accuracy"]),
        "minimum_method_gain_over_joint": method_gain
        >= float(gates["minimum_method_gain_over_joint_points"]),
        "minimum_exact_gain_over_joint": exact_gain
        >= float(gates["minimum_exact_gain_over_joint_points"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "effect_points": {
            "method_vs_joint": method_gain,
            "exact_vs_joint": exact_gain,
        },
    }


def run_output_factorization_pilot(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_output_factorization_contract(config)
    data = prepare_output_factorization_data(config)
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
    scores = _evaluate_cases(
        config,
        cases,
        progress_root=_root(config) / "selection-progress",
        evaluation_fingerprint=evaluation_fingerprint,
    )
    gate = _gate_report(scores, dict(contract["settings"]["selection_gates"]))
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": evaluation_fingerprint,
        "method": "H10 factorized method and column interface pilot",
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["fingerprint"],
        "same_parent_weights": True,
        "shared_columns": True,
        "scores": {key: value for key, value in scores.items() if key != "details"},
        "selection_gate": gate,
        "selected_interface": "factorized-method" if gate["passed"] else None,
        "confirmation_authorized": bool(gate["passed"]),
        "external_benchmark_authorized": False,
        "private_details": scores["details"],
    }
    report["result_fingerprint"] = canonical_hash(
        {key: value for key, value in report.items() if key != "private_details"}
    )
    write_json(_root(config) / "pilot.json", report)
    public = dict(report)
    public.pop("private_details")
    write_json(config.root / "reports" / "evolve" / "output-factorization-v1-pilot.json", public)
    return public


def run_output_factorization_confirmation(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_output_factorization_contract(config)
    pilot_path = _root(config) / "pilot.json"
    if not pilot_path.exists():
        raise RuntimeError("H10 pilot has not selected an interface")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("selected_interface") != "factorized-method":
        raise RuntimeError("H10 selection did not authorize confirmation")
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
    scores = _evaluate_cases(
        config,
        cases,
        progress_root=_root(config) / "confirmation-progress",
        evaluation_fingerprint=evaluation_fingerprint,
    )
    gate = _gate_report(scores, dict(contract["settings"]["confirmation_gates"]))
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": evaluation_fingerprint,
        "method": "H10 factorized method and column interface confirmation",
        "contract_fingerprint": contract["fingerprint"],
        "pilot_result_fingerprint": pilot["result_fingerprint"],
        "confirmation_manifest_fingerprint": manifest["fingerprint"],
        "same_parent_weights": True,
        "shared_columns": True,
        "scores": {key: value for key, value in scores.items() if key != "details"},
        "confirmation_gate": gate,
        "synthetic_factorization_confirmed": bool(gate["passed"]),
        "external_benchmark_authorized": False,
        "next_step": (
            "preregister-independent-external-factorized-evidence"
            if gate["passed"]
            else "reject-h10-output-factorization"
        ),
        "private_details": scores["details"],
    }
    report["result_fingerprint"] = canonical_hash(
        {key: value for key, value in report.items() if key != "private_details"}
    )
    write_json(_root(config) / "confirmation.json", report)
    public = dict(report)
    public.pop("private_details")
    write_json(
        config.root / "reports" / "evolve" / "output-factorization-v1-confirmation.json",
        public,
    )
    return public

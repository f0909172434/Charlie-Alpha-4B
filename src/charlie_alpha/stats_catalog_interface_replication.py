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
from .stats_catalog_distillation import _contract_scenarios
from .stats_catalog_grounding import _evaluate_style
from .stats_cross_format import _format_shift_case
from .stats_dgp import simulate_scenario
from .stats_family_router import _expert_context
from .stats_router_replication import _historical_scenario_audit, _scenario_semantic_payload

_EVALUATOR_VERSION = 1


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "catalog-interface-replication-v1"


def _data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "catalog-interface-replication-v1"


def _implementation_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "stats_catalog_interface_replication.py": sha256_file(Path(__file__)),
        "stats_catalog_grounding.py": sha256_file(root / "stats_catalog_grounding.py"),
        "stats_cross_format.py": sha256_file(root / "stats_cross_format.py"),
        "stats_agent.py": sha256_file(root / "stats_agent.py"),
    }


def prepare_catalog_interface_replication_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("catalog_interface_replication"))
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "contract.json"
    public_path = (
        config.root / "reports" / "evolve" / "catalog-interface-replication-v1-contract.json"
    )

    h7_path = config.root / "reports" / "evolve" / "catalog-grounding-v1-confirmation.json"
    h7 = json.loads(h7_path.read_text(encoding="utf-8"))
    if not h7.get("synthetic_interface_confirmed") or not h7["confirmation_gate"]["passed"]:
        raise RuntimeError("H12 requires the confirmed H7 flat-catalog mechanism")
    h11_path = config.root / "reports" / "evolve" / "semantic-catalog-v1-confirmation.json"
    h11 = json.loads(h11_path.read_text(encoding="utf-8"))
    if h11.get("synthetic_semantic_interface_confirmed") or h11["confirmation_gate"]["passed"]:
        raise RuntimeError(
            "H11 did not close negatively; H12 stability replication is not authorized"
        )

    _, adapter_paths = _expert_context(config)
    parent = adapter_paths["parent"]
    parent_sha = sha256_file(parent / "adapters.safetensors")
    registered: dict[str, Any] = {}
    all_scenarios = []
    seen: set[str] = set()
    fold_names = sorted(str(name) for name in settings["folds"])
    for name in fold_names:
        shard = dict(settings["folds"][name])
        scenarios = _registered_scenarios(shard, name=f"catalog-interface-replication:{name}")
        ids = {scenario.blueprint_id for scenario in scenarios}
        if ids & seen:
            raise RuntimeError(f"H12 blueprint overlap at {name}")
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
        + _contract_scenarios(
            config,
            "semantic-catalog-v1-contract.json",
            name_prefix="semantic-catalog:",
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
        raise RuntimeError("H12 overlaps prior registered blueprints or semantic points")

    audit = _historical_scenario_audit(
        config,
        all_scenarios,
        excluded_root=_data_root(config),
        minimum_normalized_distance=float(settings["minimum_normalized_distance"]),
    )
    if not audit["passed"]:
        raise RuntimeError("H12 blueprints failed historical-overlap audit")

    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": "H12 multi-seed flat-catalog interface replication",
        "method_version": int(settings["method_version"]),
        "causal_question": (
            "Does the H7 fixed global method catalog reproducibly improve menu-free canonical "
            "method-and-column extraction across three fully fresh disjoint DGP folds?"
        ),
        "h7_confirmation_result_fingerprint": h7["result_fingerprint"],
        "h7_report_sha256": sha256_file(h7_path),
        "h7_confirmed_effect_points": h7["confirmation_gate"]["effect_points"],
        "h11_negative_result_fingerprint": h11["result_fingerprint"],
        "parent": {
            "name": "v0.3.0-parent",
            "adapter_path": str(parent),
            "adapter_sha256": parent_sha,
        },
        "settings": settings,
        "fold_contracts": registered,
        "historical_overlap_audit": audit,
        "prior_registered_overlap": {
            "prior_blueprints": len(prior_ids),
            "h12_blueprints": len(new_ids),
            "blueprint_id_overlap_count": len(new_ids & prior_ids),
            "semantic_overlap_count": len(new_semantics & prior_semantics),
        },
        "implementation_sha256": _implementation_manifest(),
        "arms": {
            "menu-free-control": "Unchanged H5/H7 menu-free canonical JSON extraction prompt",
            "flat-catalog": (
                "Same parent and prompt plus H7's fixed 28-method ID + display-name catalog"
            ),
        },
        "replication_policy": (
            "All three fold blueprints and aggregate gates are frozen before any H12 simulation. "
            "No fold is used for tuning or selection."
        ),
        "claim_boundary": (
            "H12 can establish robustness of the synthetic H7 interface mechanism only. It changes "
            "no weights and does not establish independent external capability."
        ),
        "external_benchmark_opened": False,
    }
    contract["fingerprint"] = canonical_hash(contract)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("H12 contract is immutable")
        write_json(public_path, existing)
        return existing
    write_json(lock_path, contract)
    write_json(public_path, contract)
    return contract


def _simulate_fold(
    config: ProjectConfig,
    contract: dict[str, Any],
    *,
    name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    shard = dict(contract["settings"]["folds"][name])
    scenarios = _registered_scenarios(shard, name=f"catalog-interface-replication:{name}")
    blueprint_sha = canonical_hash([scenario.to_dict() for scenario in scenarios])
    if blueprint_sha != contract["fold_contracts"][name]["blueprint_sha256"]:
        raise RuntimeError(f"H12 registered blueprints changed for {name}")
    path = _data_root(config) / "folds" / f"{name}.jsonl"
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
            raise RuntimeError(f"H12 {name} surface changed")
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


def prepare_catalog_interface_replication_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_catalog_interface_replication_contract(config)
    root = _data_root(config)
    status_path = root / "data-status.json"
    manifests: dict[str, str] = {}
    case_hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    for name in sorted(contract["fold_contracts"]):
        manifest, surface = _simulate_fold(config, contract, name=name)
        cases = [_format_shift_case(simulation) for simulation in surface]
        cases_path = root / "cases" / f"{name}.jsonl"
        write_jsonl(cases_path, cases)
        manifests[name] = str(manifest["fingerprint"])
        case_hashes[name] = sha256_file(cases_path)
        counts[name] = len(cases)
    fingerprint = canonical_hash(
        {"contract": contract["fingerprint"], "manifests": manifests, "cases": case_hashes}
    )
    status = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "contract_fingerprint": contract["fingerprint"],
        "fold_counts": counts,
        "fold_manifest_fingerprints": manifests,
        "case_sha256": case_hashes,
    }
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise RuntimeError("H12 prepared data changed")
        return existing
    write_json(status_path, status)
    write_json(
        config.root / "reports" / "evolve" / "catalog-interface-replication-v1-data.json",
        status,
    )
    return status


def _fold_effect(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    c = control["format_shift"]
    x = candidate["format_shift"]
    return {
        "exact": 100 * (float(x["exact_accuracy"]) - float(c["exact_accuracy"])),
        "method": 100 * (float(x["method_set_accuracy"]) - float(c["method_set_accuracy"])),
        "columns": 100 * (float(x["column_set_accuracy"]) - float(c["column_set_accuracy"])),
    }


def _gate_report(
    *,
    folds: dict[str, dict[str, Any]],
    gates: dict[str, Any],
) -> dict[str, Any]:
    count = sum(
        int(value["menu-free-control"]["format_shift"]["count"]) for value in folds.values()
    )
    pooled: dict[str, dict[str, float]] = {}
    for arm in ("menu-free-control", "flat-catalog"):
        pooled[arm] = {}
        for metric in ("exact_accuracy", "method_set_accuracy", "column_set_accuracy"):
            pooled[arm][metric] = (
                sum(
                    int(value[arm]["format_shift"]["count"])
                    * float(value[arm]["format_shift"][metric])
                    for value in folds.values()
                )
                / count
            )
    effects = {
        "exact": 100
        * (
            pooled["flat-catalog"]["exact_accuracy"] - pooled["menu-free-control"]["exact_accuracy"]
        ),
        "method": 100
        * (
            pooled["flat-catalog"]["method_set_accuracy"]
            - pooled["menu-free-control"]["method_set_accuracy"]
        ),
        "columns": 100
        * (
            pooled["flat-catalog"]["column_set_accuracy"]
            - pooled["menu-free-control"]["column_set_accuracy"]
        ),
    }
    fold_effects = {
        name: _fold_effect(value["menu-free-control"], value["flat-catalog"])
        for name, value in folds.items()
    }
    qualifying = sum(
        effect["exact"] >= float(gates["minimum_fold_exact_gain_points"])
        and effect["method"] >= float(gates["minimum_fold_method_gain_points"])
        for effect in fold_effects.values()
    )
    checks = {
        "pooled_exact_gain": effects["exact"] >= float(gates["minimum_pooled_exact_gain_points"]),
        "pooled_method_gain": effects["method"]
        >= float(gates["minimum_pooled_method_gain_points"]),
        "pooled_column_noninferior": effects["columns"]
        >= float(gates["minimum_pooled_column_gain_points"]),
        "replicated_folds": qualifying >= int(gates["minimum_qualifying_folds"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "pooled_count": count,
        "pooled_accuracy": pooled,
        "pooled_effect_points": effects,
        "fold_effect_points": fold_effects,
        "qualifying_folds": qualifying,
    }


def run_catalog_interface_replication(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_catalog_interface_replication_contract(config)
    data = prepare_catalog_interface_replication_data(config)
    _, adapter_paths = _expert_context(config)
    parent = adapter_paths["parent"]
    agent = StatsAgent(config, adapter_path=parent)
    agent.router.set_route("adapter")
    fold_scores: dict[str, dict[str, Any]] = {}
    private_details: dict[str, dict[str, Any]] = {}
    try:
        for name in sorted(contract["fold_contracts"]):
            cases = list(read_jsonl(_data_root(config) / "cases" / f"{name}.jsonl"))
            evaluation_fingerprint = canonical_hash(
                {
                    "contract": contract["fingerprint"],
                    "data": data["fingerprint"],
                    "fold": name,
                    "parent": contract["parent"]["adapter_sha256"],
                    "evaluator_version": _EVALUATOR_VERSION,
                }
            )
            progress = _root(config) / "progress" / name
            scores = {
                "menu-free-control": _evaluate_style(
                    agent,
                    cases,
                    name="menu-free-control",
                    grounded=False,
                    progress_root=progress,
                    evaluation_fingerprint=evaluation_fingerprint,
                ),
                "flat-catalog": _evaluate_style(
                    agent,
                    cases,
                    name="flat-catalog",
                    grounded=True,
                    progress_root=progress,
                    evaluation_fingerprint=evaluation_fingerprint,
                ),
            }
            fold_scores[name] = {
                arm: {"format_shift": value["format_shift"]} for arm, value in scores.items()
            }
            private_details[name] = {arm: value["details"] for arm, value in scores.items()}
    finally:
        del agent
        gc.collect()
        mx.clear_cache()

    gate = _gate_report(folds=fold_scores, gates=dict(contract["settings"]["gates"]))
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "H12 multi-seed flat-catalog interface replication",
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["fingerprint"],
        "same_parent_weights": True,
        "fold_scores": fold_scores,
        "replication_gate": gate,
        "synthetic_catalog_interface_replicated": bool(gate["passed"]),
        "external_benchmark_authorized": False,
        "next_step": (
            "preregister-independent-external-catalog-interface-evidence"
            if gate["passed"]
            else "reject-h12-catalog-interface-replication"
        ),
        "private_details": private_details,
    }
    report["result_fingerprint"] = canonical_hash(
        {key: value for key, value in report.items() if key != "private_details"}
    )
    write_json(_root(config) / "report.json", report)
    public = dict(report)
    public.pop("private_details")
    write_json(
        config.root / "reports" / "evolve" / "catalog-interface-replication-v1.json",
        public,
    )
    return public

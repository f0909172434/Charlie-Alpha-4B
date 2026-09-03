from __future__ import annotations

import gc
import json
from collections import Counter
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl
from .stats_agent import StatsAgent
from .stats_canonical_bottleneck import _registered_scenarios
from .stats_cross_format import _flatten_columns
from .stats_data import _analysis_plan, _scenario, _study_context
from .stats_dgp import Scenario, simulate_scenario
from .stats_family_router import _expert_context
from .stats_representation_probe import (
    _extract_representations,
    _load_representations,
    _probe_scores,
    _save_representations,
)
from .stats_router_replication import _historical_scenario_audit
from .stats_selector_head import _load_head
from .stats_style_invariance import _natural_conditions, _style_question

_FULL_STYLES = ("audit", "researcher", "vignette")
_REDUCED_STYLES = ("conventional", "partial")
_ALL_STYLES = (*_FULL_STYLES, *_REDUCED_STYLES)


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "selector-sufficiency-v1"


def _data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "selector-sufficiency-v1"


def _implementation_manifest() -> dict[str, str]:
    source_root = Path(__file__).resolve().parent
    return {
        "stats_selector_sufficiency.py": sha256_file(Path(__file__)),
        "stats_style_invariance.py": sha256_file(source_root / "stats_style_invariance.py"),
        "stats_representation_probe.py": sha256_file(
            source_root / "stats_representation_probe.py"
        ),
        "stats_selector_head.py": sha256_file(source_root / "stats_selector_head.py"),
        "stats_agent.py": sha256_file(source_root / "stats_agent.py"),
    }


def _conventional_question(scenario: Scenario) -> str:
    context = _study_context(scenario)
    schema = ", ".join(str(item) for item in context["schema"])
    return (
        f"{str(context['study_design']).capitalize()} targeting {context['estimand']}. "
        f"Variables: {schema}. Which primary statistical analysis should be used?"
    )


def _partial_question(scenario: Scenario) -> str:
    context = _study_context(scenario)
    schema = ", ".join(str(item) for item in context["schema"])
    conditions = _natural_conditions(scenario).split(", ")
    one_condition = conditions[0] if conditions else "one study condition is declared"
    return (
        f"We are planning a {context['study_design']} targeting {context['estimand']}. "
        f"The data contain {schema}. One reported condition is {one_condition}. "
        "Choose the primary analysis."
    )


def _render_question(scenario: Scenario, style: str) -> str:
    if style in _FULL_STYLES:
        return _style_question(scenario, style)
    if style == "conventional":
        return _conventional_question(scenario)
    if style == "partial":
        return _partial_question(scenario)
    raise ValueError(f"Unknown H16 style: {style}")


def _case_from_simulation(simulation: dict[str, Any], *, style: str) -> dict[str, Any]:
    scenario = _scenario(simulation["scenario"])
    method_id = str(simulation["selected_method_id"])
    plan = _analysis_plan(scenario, method_id, incomplete=False)
    return {
        "case_id": f"{scenario.blueprint_id}::{style}",
        "semantic_id": scenario.blueprint_id,
        "style": style,
        "selector_safe_truth": style in _FULL_STYLES,
        "family_id": scenario.family_id,
        "question": _render_question(scenario, style),
        "gold_methods": [method_id],
        "gold_columns": _flatten_columns(dict(plan["variables"])),
    }


def _registered_blueprints(config: ProjectConfig, settings: dict[str, Any]) -> dict[str, Any]:
    registered: dict[str, Any] = {}
    all_scenarios: list[Scenario] = []
    seen: set[str] = set()
    for name in ("training_shard", "selection_shard", "confirmation_shard"):
        shard = dict(settings[name])
        scenarios = _registered_scenarios(shard, name=f"selector-sufficiency:{name}")
        ids = {scenario.blueprint_id for scenario in scenarios}
        if ids & seen:
            raise RuntimeError(f"H16 blueprint overlap at {name}")
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
    audit = _historical_scenario_audit(
        config,
        all_scenarios,
        excluded_root=_data_root(config),
        minimum_normalized_distance=float(settings["minimum_normalized_distance"]),
    )
    if not audit["passed"]:
        raise RuntimeError("H16 blueprints failed historical-overlap audit")
    return {"registered": registered, "historical_overlap_audit": audit}


def prepare_selector_sufficiency_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("selector_sufficiency"))
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "contract.json"
    public_path = config.root / "reports" / "evolve" / "selector-sufficiency-v1-contract.json"

    h15_path = config.root / "reports" / "evolve" / "style-invariance-v1-confirmation.json"
    h15 = json.loads(h15_path.read_text(encoding="utf-8"))
    if not h15.get("h15_diagnosis_confirmed"):
        raise RuntimeError("H16 requires terminal H15 style-invariance confirmation")
    if h15.get("selected_route") != "frozen-head-style-stable":
        raise RuntimeError("H16 requires H15 to confirm the frozen H14 head as style-stable")

    diagnostic_path = (
        config.root / "reports" / "evolve" / "external-representation-diagnostic-v1.json"
    )
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if not diagnostic.get("historical_only") or diagnostic.get("fresh_external_evidence"):
        raise RuntimeError("H16 requires E3 readback to remain historical-only")
    if diagnostic.get("e3_fit_or_tuning_performed"):
        raise RuntimeError("H16 cannot begin after E3 fit or tuning")

    h14_contract_path = config.root / "reports" / "evolve" / "selector-head-v1-contract.json"
    h14_contract = json.loads(h14_contract_path.read_text(encoding="utf-8"))
    parent_path = Path(str(h14_contract["parent"]["adapter_path"]))
    if sha256_file(parent_path / "adapters.safetensors") != str(
        h14_contract["parent"]["adapter_sha256"]
    ):
        raise RuntimeError("H16 parent adapter changed after H14")
    head_path = Path(str(h14_contract["selector_head"]["artifact_path"]))
    if sha256_file(head_path) != str(h14_contract["selector_head"]["artifact_sha256"]):
        raise RuntimeError("H16 H14 selector-head artifact changed")

    blueprint_state = _registered_blueprints(config, settings)
    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": "H16 hidden-geometry selector sufficiency guard",
        "method_version": int(settings["method_version"]),
        "causal_question": (
            "Can a one-class hidden-state geometry guard identify when the frozen H14 selector is "
            "operating inside its declared fine-grained DGP regime, preserving selector use for "
            "fully specified prompts while rejecting prompts that omit decision-critical "
            "conditions?"
        ),
        "h15_result_fingerprint": h15["result_fingerprint"],
        "h15_report_sha256": sha256_file(h15_path),
        "historical_e3_diagnostic_fingerprint": diagnostic["result_fingerprint"],
        "historical_e3_diagnostic_sha256": sha256_file(diagnostic_path),
        "historical_e3_used_for_threshold_selection": False,
        "parent": h14_contract["parent"],
        "frozen_h14_head": h14_contract["selector_head"],
        "settings": settings,
        "styles": {
            "selector_safe": list(_FULL_STYLES),
            "fallback_required": list(_REDUCED_STYLES),
            "reduced_semantics": (
                "same source DGP and simulator gold, but the rendered prompt intentionally omits "
                "condition facts required to justify the simulator's fine-grained method choice; "
                "gold method accuracy is therefore not scored on reduced prompts"
            ),
        },
        "guard": {
            "kind": "one-class centered nearest-cosine support score",
            "training_bank": (
                "all H14-covered training semantic points rendered in audit, researcher, and "
                "vignette styles; no reduced rendering enters the bank"
            ),
            "center": "arithmetic mean of the frozen training-bank hidden vectors",
            "score": (
                "maximum cosine similarity between the mean-centered query vector and any "
                "mean-centered training-bank vector"
            ),
            "decision": "use H14 selector iff support_score >= selected threshold; else fallback",
            "threshold_grid": [float(value) for value in settings["thresholds"]],
        },
        "selection_policy": (
            "Select one threshold on the fresh selection shard only. It must pass every full-style "
            "retention, reduced-style rejection, and accepted-head-accuracy gate. Among eligible "
            "thresholds maximize the weaker of full retention and reduced rejection, then full "
            "retention, then accepted-head accuracy, then prefer the lower threshold."
        ),
        "confirmation_policy": (
            "The confirmation simulations and representations remain unopened until selection "
            "freezes one threshold. Confirmation uses that threshold without adaptation."
        ),
        "claim_boundary": (
            "H16 is a synthetic selector-support diagnosis and runtime-safety mechanism. It does "
            "not rescue E3, establish new external capability, change model weights, replace the "
            "champion, or authorize release."
        ),
        "blueprint_contracts": blueprint_state["registered"],
        "historical_overlap_audit": blueprint_state["historical_overlap_audit"],
        "implementation_sha256": _implementation_manifest(),
        "confirmation_simulations_opened": False,
        "external_benchmark_opened": False,
    }
    contract["fingerprint"] = canonical_hash(contract)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("H16 contract is immutable")
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
        selection_path = _root(config) / "selection.json"
        if not selection_path.exists():
            raise RuntimeError("H16 confirmation cannot open before selection")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if not selection.get("confirmation_authorized"):
            raise RuntimeError("H16 selection did not authorize confirmation")

    shard = dict(contract["settings"][name])
    scenarios = _registered_scenarios(shard, name=f"selector-sufficiency:{name}")
    blueprint_sha = canonical_hash([scenario.to_dict() for scenario in scenarios])
    if blueprint_sha != contract["blueprint_contracts"][name]["blueprint_sha256"]:
        raise RuntimeError(f"H16 registered blueprints changed for {name}")
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
        if (
            manifest.get("fingerprint") != fingerprint
            or manifest.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"H16 {name} surface changed")
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


def _materialize_cases(simulations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _case_from_simulation(simulation, style=style)
        for simulation in simulations
        for style in _ALL_STYLES
    ]


def _eligible_cases(
    cases: list[dict[str, Any]], observed_methods: set[str]
) -> list[dict[str, Any]]:
    return [case for case in cases if str(case["gold_methods"][0]) in observed_methods]


def _coverage(cases: list[dict[str, Any]], observed_methods: set[str]) -> dict[str, Any]:
    researcher = [case for case in cases if case["style"] == "researcher"]
    eligible = _eligible_cases(researcher, observed_methods)
    counts = Counter(str(case["gold_methods"][0]) for case in eligible)
    return {
        "semantic_point_count": len(researcher),
        "eligible_semantic_point_count": len(eligible),
        "eligible_fraction": len(eligible) / max(len(researcher), 1),
        "eligible_method_count": len(counts),
        "eligible_method_counts": dict(sorted(counts.items())),
    }


def prepare_selector_sufficiency_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_selector_sufficiency_contract(config)
    observed = set(str(value) for value in contract["frozen_h14_head"]["observed_methods"])
    root = _data_root(config)
    status_path = root / "data-status.json"
    surfaces: dict[str, str] = {}
    case_hashes: dict[str, str] = {}
    coverage: dict[str, Any] = {}
    for name in ("training_shard", "selection_shard"):
        manifest, simulations = _simulate_surface(config, contract, name=name)
        cases = _materialize_cases(simulations)
        case_path = root / "cases" / f"{name}.jsonl"
        write_jsonl(case_path, cases)
        surfaces[name] = manifest["fingerprint"]
        case_hashes[name] = sha256_file(case_path)
        coverage[name] = _coverage(cases, observed)

    gates = dict(contract["settings"]["data_gates"])
    checks = {
        "training_eligible_semantics": coverage["training_shard"]["eligible_semantic_point_count"]
        >= int(gates["minimum_training_eligible_semantic_points"]),
        "selection_eligible_semantics": coverage["selection_shard"]["eligible_semantic_point_count"]
        >= int(gates["minimum_selection_eligible_semantic_points"]),
        "training_eligible_fraction": coverage["training_shard"]["eligible_fraction"]
        >= float(gates["minimum_eligible_fraction"]),
        "selection_eligible_fraction": coverage["selection_shard"]["eligible_fraction"]
        >= float(gates["minimum_eligible_fraction"]),
    }
    fingerprint = canonical_hash(
        {"contract": contract["fingerprint"], "surfaces": surfaces, "cases": case_hashes}
    )
    status = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "contract_fingerprint": contract["fingerprint"],
        "surface_fingerprints": surfaces,
        "case_sha256": case_hashes,
        "coverage": coverage,
        "data_gate": {"passed": all(checks.values()), "checks": checks},
        "selection_authorized": all(checks.values()),
        "confirmation_opened": False,
    }
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise RuntimeError("H16 prepared data changed")
        return existing
    write_json(status_path, status)
    write_json(config.root / "reports" / "evolve" / "selector-sufficiency-v1-data.json", status)
    return status


def _representation_path(config: ProjectConfig, shard: str, style: str) -> Path:
    return _root(config) / "representations" / shard / f"{style}.npz"


def _cases_by_style(
    cases: list[dict[str, Any]], observed_methods: set[str]
) -> dict[str, list[dict[str, Any]]]:
    eligible = _eligible_cases(cases, observed_methods)
    return {style: [case for case in eligible if case["style"] == style] for style in _ALL_STYLES}


def _ensure_representations(
    config: ProjectConfig,
    *,
    shard: str,
    cases: list[dict[str, Any]],
    observed_methods: set[str],
) -> dict[str, str]:
    by_style = _cases_by_style(cases, observed_methods)
    paths = {style: _representation_path(config, shard, style) for style in _ALL_STYLES}
    if all(path.exists() for path in paths.values()):
        return {style: sha256_file(path) for style, path in paths.items()}
    _, adapter_paths = _expert_context(config)
    agent = StatsAgent(config, adapter_path=adapter_paths["parent"])
    agent.router.set_route("adapter")
    try:
        hashes: dict[str, str] = {}
        for style in _ALL_STYLES:
            path = paths[style]
            if path.exists():
                hashes[style] = sha256_file(path)
                continue
            vectors, labels, case_ids = _extract_representations(
                agent, by_style[style], grounded=False
            )
            hashes[style] = _save_representations(
                path, vectors=vectors, labels=labels, case_ids=case_ids
            )
            mx.clear_cache()
        return hashes
    finally:
        del agent
        gc.collect()


def _load_representations_by_style(
    config: ProjectConfig, shard: str
) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    return {
        style: _load_representations(_representation_path(config, shard, style))
        for style in _ALL_STYLES
    }


def _verify_representation_order(
    cases_by_style: dict[str, list[dict[str, Any]]],
    representations: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
) -> None:
    expected_semantics: list[str] | None = None
    for style in _ALL_STYLES:
        cases = cases_by_style[style]
        _, labels, case_ids = representations[style]
        expected_ids = [str(case["case_id"]) for case in cases]
        if case_ids != expected_ids:
            raise RuntimeError(f"H16 {style} representation order changed")
        expected_labels = [str(case["gold_methods"][0]) for case in cases]
        if len(labels) != len(expected_labels):
            raise RuntimeError(f"H16 {style} representation label count changed")
        semantics = [str(case["semantic_id"]) for case in cases]
        if expected_semantics is None:
            expected_semantics = semantics
        elif semantics != expected_semantics:
            raise RuntimeError("H16 matched semantic order changed across styles")


def _centered_normalize(vectors: np.ndarray, center: np.ndarray) -> np.ndarray:
    shifted = np.asarray(vectors, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    norms = np.linalg.norm(shifted, axis=1, keepdims=True)
    return shifted / np.maximum(norms, 1e-12)


def _build_training_bank(
    representations: dict[str, tuple[np.ndarray, np.ndarray, list[str]]]
) -> dict[str, Any]:
    vectors = np.concatenate([representations[style][0] for style in _FULL_STYLES], axis=0)
    labels = np.concatenate([representations[style][1] for style in _FULL_STYLES], axis=0)
    case_ids = [
        case_id for style in _FULL_STYLES for case_id in representations[style][2]
    ]
    center = np.mean(vectors, axis=0, keepdims=True)
    normalized = _centered_normalize(vectors, center)
    return {
        "vectors": vectors,
        "labels": labels,
        "case_ids": case_ids,
        "center": center,
        "normalized": normalized,
    }


def _save_training_bank(config: ProjectConfig, bank: dict[str, Any]) -> Path:
    path = _root(config) / "training-bank.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        np.savez_compressed(
            path,
            vectors=np.asarray(bank["vectors"], dtype=np.float32),
            labels=np.asarray(bank["labels"], dtype=np.int64),
            case_ids=np.asarray(bank["case_ids"]),
            center=np.asarray(bank["center"], dtype=np.float32),
        )
    return path


def _load_training_bank(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    vectors = np.asarray(data["vectors"], dtype=np.float64)
    center = np.asarray(data["center"], dtype=np.float64)
    return {
        "vectors": vectors,
        "labels": np.asarray(data["labels"], dtype=np.int64),
        "case_ids": [str(value) for value in data["case_ids"].tolist()],
        "center": center,
        "normalized": _centered_normalize(vectors, center),
    }


def _support_scores(bank: dict[str, Any], vectors: np.ndarray) -> np.ndarray:
    queries = _centered_normalize(vectors, np.asarray(bank["center"], dtype=np.float64))
    return np.max(queries @ np.asarray(bank["normalized"], dtype=np.float64).T, axis=1)


def _head_correctness(head: dict[str, Any], vectors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    predicted = np.argmax(_probe_scores(head, vectors), axis=1)
    return predicted == labels


def _threshold_report(
    scores: dict[str, np.ndarray],
    head_correct: dict[str, np.ndarray],
    *,
    threshold: float,
) -> dict[str, Any]:
    full_acceptance_by_style = {
        style: float(np.mean(scores[style] >= threshold)) for style in _FULL_STYLES
    }
    reduced_rejection_by_style = {
        style: float(np.mean(scores[style] < threshold)) for style in _REDUCED_STYLES
    }
    full_accept = np.concatenate([scores[style] >= threshold for style in _FULL_STYLES])
    full_correct = np.concatenate([head_correct[style] for style in _FULL_STYLES])
    accepted_correct = full_correct[full_accept]
    unguarded_accuracy = float(np.mean(full_correct))
    accepted_accuracy = float(np.mean(accepted_correct)) if len(accepted_correct) else 0.0
    return {
        "threshold": float(threshold),
        "full_acceptance": float(np.mean(full_accept)),
        "minimum_full_style_acceptance": min(full_acceptance_by_style.values()),
        "full_acceptance_by_style": full_acceptance_by_style,
        "reduced_rejection": float(
            np.mean(np.concatenate([scores[style] < threshold for style in _REDUCED_STYLES]))
        ),
        "minimum_reduced_style_rejection": min(reduced_rejection_by_style.values()),
        "reduced_rejection_by_style": reduced_rejection_by_style,
        "unguarded_head_accuracy": unguarded_accuracy,
        "accepted_head_accuracy": accepted_accuracy,
        "accepted_head_accuracy_change_points": 100.0 * (accepted_accuracy - unguarded_accuracy),
        "accepted_full_count": int(np.sum(full_accept)),
        "total_full_count": int(len(full_accept)),
    }


def _gate_threshold(report: dict[str, Any], gates: dict[str, Any]) -> dict[str, bool]:
    return {
        "full_acceptance": float(report["full_acceptance"])
        >= float(gates["minimum_full_acceptance"]),
        "minimum_full_style_acceptance": float(report["minimum_full_style_acceptance"])
        >= float(gates["minimum_full_style_acceptance"]),
        "reduced_rejection": float(report["reduced_rejection"])
        >= float(gates["minimum_reduced_rejection"]),
        "minimum_reduced_style_rejection": float(report["minimum_reduced_style_rejection"])
        >= float(gates["minimum_reduced_style_rejection"]),
        "accepted_head_accuracy": float(report["accepted_head_accuracy"])
        >= float(gates["minimum_accepted_head_accuracy"]),
        "accepted_head_noninferior": float(report["accepted_head_accuracy_change_points"])
        >= -float(gates["maximum_accepted_head_accuracy_regression_points"]),
    }


def _select_threshold(
    reports: list[dict[str, Any]], gates: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    enriched: list[dict[str, Any]] = []
    for report in reports:
        checks = _gate_threshold(report, gates)
        enriched.append({**report, "gates": checks, "eligible": all(checks.values())})
    eligible = [report for report in enriched if report["eligible"]]
    if not eligible:
        return None, enriched
    selected = max(
        eligible,
        key=lambda report: (
            min(float(report["full_acceptance"]), float(report["reduced_rejection"])),
            float(report["full_acceptance"]),
            float(report["accepted_head_accuracy"]),
            -float(report["threshold"]),
        ),
    )
    return selected, enriched


def run_selector_sufficiency_selection(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_selector_sufficiency_contract(config)
    data = prepare_selector_sufficiency_data(config)
    if not data.get("selection_authorized"):
        raise RuntimeError("H16 data gate did not authorize selection")
    observed = set(str(value) for value in contract["frozen_h14_head"]["observed_methods"])
    case_root = _data_root(config) / "cases"
    train_cases = list(read_jsonl(case_root / "training_shard.jsonl"))
    selection_cases = list(read_jsonl(case_root / "selection_shard.jsonl"))
    hashes = {
        "training_shard": _ensure_representations(
            config, shard="training_shard", cases=train_cases, observed_methods=observed
        ),
        "selection_shard": _ensure_representations(
            config, shard="selection_shard", cases=selection_cases, observed_methods=observed
        ),
    }
    train = _load_representations_by_style(config, "training_shard")
    selection = _load_representations_by_style(config, "selection_shard")
    _verify_representation_order(_cases_by_style(train_cases, observed), train)
    _verify_representation_order(_cases_by_style(selection_cases, observed), selection)

    bank = _build_training_bank(train)
    bank_path = _save_training_bank(config, bank)
    bank = _load_training_bank(bank_path)
    h14_contract = json.loads(
        (config.root / "reports" / "evolve" / "selector-head-v1-contract.json").read_text(
            encoding="utf-8"
        )
    )
    head = _load_head(h14_contract)
    scores = {
        style: _support_scores(bank, selection[style][0]) for style in _ALL_STYLES
    }
    head_correct = {
        style: _head_correctness(head, selection[style][0], selection[style][1])
        for style in _FULL_STYLES
    }
    reports = [
        _threshold_report(scores, head_correct, threshold=float(threshold))
        for threshold in contract["settings"]["thresholds"]
    ]
    selected, sweep = _select_threshold(
        reports, dict(contract["settings"]["selection_gates"])
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "H16 hidden-geometry selector sufficiency selection",
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["fingerprint"],
        "same_parent_weights": True,
        "representations": hashes,
        "training_bank": {
            "path": str(bank_path),
            "sha256": sha256_file(bank_path),
            "count": int(len(bank["vectors"])),
        },
        "threshold_sweep": sweep,
        "selected_threshold": float(selected["threshold"]) if selected else None,
        "selected_metrics": selected,
        "confirmation_authorized": selected is not None,
        "next_step": (
            "open-fresh-h16-confirmation"
            if selected is not None
            else "stop-h16-selection-without-opening-confirmation"
        ),
        "historical_e3_opened": False,
    }
    report["result_fingerprint"] = canonical_hash(report)
    write_json(_root(config) / "selection.json", report)
    write_json(
        config.root / "reports" / "evolve" / "selector-sufficiency-v1-selection.json",
        report,
    )
    return report


def run_selector_sufficiency_confirmation(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_selector_sufficiency_contract(config)
    selection_path = _root(config) / "selection.json"
    if not selection_path.exists():
        raise RuntimeError("H16 confirmation requires completed selection")
    selection_report = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection_report.get("confirmation_authorized"):
        raise RuntimeError("H16 selection did not authorize confirmation")
    threshold = float(selection_report["selected_threshold"])
    bank_path = Path(str(selection_report["training_bank"]["path"]))
    if sha256_file(bank_path) != str(selection_report["training_bank"]["sha256"]):
        raise RuntimeError("H16 frozen training bank changed")
    bank = _load_training_bank(bank_path)

    manifest, simulations = _simulate_surface(config, contract, name="confirmation_shard")
    cases = _materialize_cases(simulations)
    case_path = _data_root(config) / "cases" / "confirmation_shard.jsonl"
    write_jsonl(case_path, cases)
    observed = set(str(value) for value in contract["frozen_h14_head"]["observed_methods"])
    coverage = _coverage(cases, observed)
    if coverage["eligible_semantic_point_count"] < int(
        contract["settings"]["data_gates"]["minimum_confirmation_eligible_semantic_points"]
    ):
        raise RuntimeError("H16 confirmation coverage fell below the preregistered minimum")
    hashes = _ensure_representations(
        config, shard="confirmation_shard", cases=cases, observed_methods=observed
    )
    confirmation = _load_representations_by_style(config, "confirmation_shard")
    _verify_representation_order(_cases_by_style(cases, observed), confirmation)

    h14_contract = json.loads(
        (config.root / "reports" / "evolve" / "selector-head-v1-contract.json").read_text(
            encoding="utf-8"
        )
    )
    head = _load_head(h14_contract)
    scores = {
        style: _support_scores(bank, confirmation[style][0]) for style in _ALL_STYLES
    }
    head_correct = {
        style: _head_correctness(head, confirmation[style][0], confirmation[style][1])
        for style in _FULL_STYLES
    }
    metrics = _threshold_report(scores, head_correct, threshold=threshold)
    checks = _gate_threshold(metrics, dict(contract["settings"]["confirmation_gates"]))
    passed = all(checks.values())
    result: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "H16 hidden-geometry selector sufficiency confirmation",
        "contract_fingerprint": contract["fingerprint"],
        "selection_result_fingerprint": selection_report["result_fingerprint"],
        "same_parent_weights": True,
        "confirmation_surface_fingerprint": manifest["fingerprint"],
        "confirmation_case_sha256": sha256_file(case_path),
        "confirmation_coverage": coverage,
        "representations": hashes,
        "training_bank_sha256": sha256_file(bank_path),
        "selected_threshold": threshold,
        "metrics": metrics,
        "confirmation_gate": {"passed": passed, "checks": checks},
        "selector_sufficiency_guard_confirmed": passed,
        "next_step": (
            "freeze-selective-selector-runtime-and-read-back-e3-historically"
            if passed
            else "stop-h16-geometry-guard"
        ),
        "champion_changed": False,
        "release_authorized": False,
        "external_benchmark_opened": False,
    }
    result["result_fingerprint"] = canonical_hash(result)
    write_json(_root(config) / "confirmation.json", result)
    write_json(
        config.root / "reports" / "evolve" / "selector-sufficiency-v1-confirmation.json",
        result,
    )
    return result


def run_selector_sufficiency_historical_e3(config: ProjectConfig) -> dict[str, Any]:
    confirmation_path = (
        config.root / "reports" / "evolve" / "selector-sufficiency-v1-confirmation.json"
    )
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    if not confirmation.get("selector_sufficiency_guard_confirmed"):
        raise RuntimeError("Historical E3 readback requires confirmed H16 guard")
    selection = json.loads((_root(config) / "selection.json").read_text(encoding="utf-8"))
    bank_path = Path(str(selection["training_bank"]["path"]))
    if sha256_file(bank_path) != str(selection["training_bank"]["sha256"]):
        raise RuntimeError("H16 training bank changed before historical E3 readback")
    bank = _load_training_bank(bank_path)

    e3_path = config.root / "reports" / "evolve" / "selector-external-v1.json"
    e3 = json.loads(e3_path.read_text(encoding="utf-8"))
    diagnostic_path = (
        config.root / "reports" / "evolve" / "external-representation-diagnostic-v1.json"
    )
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    representation_path = (
        config.path_for("artifact_dir")
        / "external-representation-diagnostic-v1"
        / "e3-eligible-representations.npz"
    )
    if sha256_file(representation_path) != diagnostic["representation_sha256"]:
        raise RuntimeError("Historical E3 representation artifact changed")
    vectors, _, case_ids = _load_representations(representation_path)
    support = _support_scores(bank, vectors)
    threshold = float(confirmation["selected_threshold"])
    accept = support >= threshold

    private_path = (
        config.path_for("artifact_dir") / "selector-external-v1" / "report-amended-v1.json"
    )
    private = json.loads(private_path.read_text(encoding="utf-8"))
    control = {
        str(row["case_id"]): row
        for row in private["private_details"]["menu-free-control"]
        if row.get("eligible")
    }
    head = {
        str(row["case_id"]): row
        for row in private["private_details"]["selector-head"]
        if row.get("eligible")
    }
    rows: list[dict[str, Any]] = []
    for index, case_id in enumerate(case_ids):
        use_head = bool(accept[index])
        chosen = head[case_id] if use_head else control[case_id]
        rows.append(
            {
                "case_id": case_id,
                "support_score": float(support[index]),
                "use_selector": use_head,
                "selected_source": "frozen-h14" if use_head else "menu-free-fallback",
                "correct": bool(chosen["correct"]),
                "control_correct": bool(control[case_id]["correct"]),
                "head_correct": bool(head[case_id]["correct"]),
            }
        )
    candidate_accuracy = float(np.mean([row["correct"] for row in rows]))
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "historical E3 readback of frozen H16 selective selector",
        "historical_only": True,
        "fresh_external_evidence": False,
        "e3_fit_or_threshold_tuning_performed": False,
        "h16_result_fingerprint": confirmation["result_fingerprint"],
        "e3_result_fingerprint": e3["result_fingerprint"],
        "threshold": threshold,
        "eligible_count": len(rows),
        "selector_accept_count": int(np.sum(accept)),
        "selector_accept_rate": float(np.mean(accept)),
        "selective_candidate_accuracy": candidate_accuracy,
        "menu_free_control_accuracy": float(e3["scores"]["menu-free-control"]["eligible_accuracy"]),
        "frozen_h14_accuracy": float(e3["scores"]["selector-head"]["eligible_accuracy"]),
        "candidate_gain_over_frozen_h14_points": 100.0
        * (candidate_accuracy - float(e3["scores"]["selector-head"]["eligible_accuracy"])),
        "candidate_change_from_control_points": 100.0
        * (candidate_accuracy - float(e3["scores"]["menu-free-control"]["eligible_accuracy"])),
        "rows": rows,
        "next_step": (
            "qualify-a-genuinely-new-external-source-for-frozen-h16"
            if candidate_accuracy >= float(e3["scores"]["menu-free-control"]["eligible_accuracy"])
            else "do-not-promote-h16-and-reconsider-the-selective-boundary"
        ),
        "claim_boundary": (
            "This reuses already-opened E3 after every H16 threshold and artifact was frozen. It "
            "is diagnostic only and cannot support a fresh external capability claim."
        ),
    }
    report["result_fingerprint"] = canonical_hash(report)
    write_json(_root(config) / "historical-e3.json", report)
    write_json(
        config.root / "reports" / "evolve" / "selector-sufficiency-v1-historical-e3.json",
        report,
    )
    return report

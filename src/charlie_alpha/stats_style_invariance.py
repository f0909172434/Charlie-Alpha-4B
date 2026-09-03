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
from .stats_data import _analysis_plan, _render_question, _scenario, _study_context
from .stats_dgp import Scenario, simulate_scenario
from .stats_family_router import _expert_context
from .stats_representation_probe import (
    _extract_representations,
    _fit_ridge_probe,
    _load_representations,
    _normalize_rows,
    _probe_metrics,
    _save_representations,
    _select_probe,
)
from .stats_router_replication import _historical_scenario_audit
from .stats_selector_head import _load_head

_STYLES = ("audit", "researcher", "vignette")


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "style-invariance-v1"


def _data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "style-invariance-v1"


def _implementation_manifest() -> dict[str, str]:
    source_root = Path(__file__).resolve().parent
    return {
        "stats_style_invariance.py": sha256_file(Path(__file__)),
        "stats_representation_probe.py": sha256_file(source_root / "stats_representation_probe.py"),
        "stats_selector_head.py": sha256_file(source_root / "stats_selector_head.py"),
        "stats_data.py": sha256_file(source_root / "stats_data.py"),
        "stats_agent.py": sha256_file(source_root / "stats_agent.py"),
    }


def _render_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _natural_conditions(scenario: Scenario) -> str:
    labels = {
        "n": "sample size",
        "variance_ratio": "variance ratio",
        "pair_correlation": "within-pair correlation",
        "tail_weight": "tail heaviness",
        "effect_size": "effect size",
        "sparsity": "cell sparsity",
        "event_rate": "event rate",
        "heteroskedasticity": "heteroskedasticity",
        "outlier_fraction": "outlier fraction",
        "separation": "separation",
        "overdispersion": "overdispersion",
        "icc": "intraclass correlation",
        "clusters": "number of clusters",
        "cluster_size": "cluster size",
        "non_ph": "non-proportional-hazards strength",
        "censoring": "censoring fraction",
        "selection_strength": "selection strength",
        "missing_rate": "missing-data fraction",
        "confounding": "confounding strength",
        "assignment_probability": "treatment probability",
        "prior_bias": "prior bias",
        "model_misspecification": "model misspecification",
        "shift": "distribution shift",
        "leakage": "leakage strength",
        "drift": "temporal drift",
        "horizon": "forecast horizon",
    }
    parts = [
        f"{labels.get(key, key.replace('_', ' '))} {_render_number(value)}"
        for key, value in scenario.parameters.items()
    ]
    return ", ".join(parts)


def _style_question(scenario: Scenario, style: str) -> str:
    if style not in _STYLES:
        raise ValueError(f"Unknown H15 style: {style}")
    if style == "audit":
        return _render_question(scenario, "en", incomplete=False, view="standard")

    context = _study_context(scenario)
    schema = ", ".join(str(item) for item in context["schema"])
    conditions = _natural_conditions(scenario)
    if style == "researcher":
        return (
            f"We are planning a {context['study_design']}. The target quantity is "
            f"{context['estimand']}. The sampling unit is the {context['sampling_unit']}; "
            f"{context['dependence']}, and the data are {context['missingness']}. The working "
            f"table contains {schema}. The study conditions are: {conditions}. Which single "
            "primary statistical analysis is most defensible? Prioritize Type I error control "
            "and interval coverage over power."
        )
    return (
        f"{str(context['study_design']).capitalize()} targeting {context['estimand']}. "
        f"Variables: {schema}. {str(context['dependence']).capitalize()}; "
        f"{context['missingness']}. Key conditions: {conditions}. Select the primary analysis, "
        "giving validity priority over power."
    )


def _case_from_simulation(simulation: dict[str, Any], *, style: str) -> dict[str, Any]:
    scenario = _scenario(simulation["scenario"])
    method_id = str(simulation["selected_method_id"])
    plan = _analysis_plan(scenario, method_id, incomplete=False)
    return {
        "case_id": f"{scenario.blueprint_id}::{style}",
        "semantic_id": scenario.blueprint_id,
        "style": style,
        "family_id": scenario.family_id,
        "question": _style_question(scenario, style),
        "gold_methods": [method_id],
        "gold_columns": _flatten_columns(dict(plan["variables"])),
    }


def _registered_blueprints(config: ProjectConfig, settings: dict[str, Any]) -> dict[str, Any]:
    registered: dict[str, Any] = {}
    all_scenarios: list[Scenario] = []
    seen: set[str] = set()
    for name in ("training_shard", "selection_shard", "confirmation_shard"):
        shard = dict(settings[name])
        scenarios = _registered_scenarios(shard, name=f"style-invariance:{name}")
        ids = {scenario.blueprint_id for scenario in scenarios}
        if ids & seen:
            raise RuntimeError(f"H15 blueprint overlap at {name}")
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
        raise RuntimeError("H15 blueprints failed historical-overlap audit")
    return {"registered": registered, "historical_overlap_audit": audit}


def prepare_style_invariance_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("style_invariance"))
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "contract.json"
    public_path = config.root / "reports" / "evolve" / "style-invariance-v1-contract.json"

    h14_path = config.root / "reports" / "evolve" / "selector-head-v1-confirmation.json"
    h14 = json.loads(h14_path.read_text(encoding="utf-8"))
    if not h14.get("selector_head_architecture_confirmed"):
        raise RuntimeError("H15 requires confirmed H14 selector-head architecture")
    h14_contract_path = config.root / "reports" / "evolve" / "selector-head-v1-contract.json"
    h14_contract = json.loads(h14_contract_path.read_text(encoding="utf-8"))

    e3_path = config.root / "reports" / "evolve" / "selector-external-v1.json"
    e3 = json.loads(e3_path.read_text(encoding="utf-8"))
    if e3.get("external_selector_head_transfer_supported") is not False:
        raise RuntimeError("H15 requires terminal-negative E3 selector-head transfer")
    if e3.get("next_step") != (
        "treat-h14-as-synthetic-format-limited-and-study-representation-transfer"
    ):
        raise RuntimeError("E3 did not authorize H15 representation-transfer diagnosis")

    parent_path = Path(str(h14_contract["parent"]["adapter_path"]))
    if sha256_file(parent_path / "adapters.safetensors") != str(
        h14_contract["parent"]["adapter_sha256"]
    ):
        raise RuntimeError("H15 parent adapter changed after H14")
    head_path = Path(str(h14_contract["selector_head"]["artifact_path"]))
    if sha256_file(head_path) != str(h14_contract["selector_head"]["artifact_sha256"]):
        raise RuntimeError("H15 H14 selector-head artifact changed")

    blueprint_state = _registered_blueprints(config, settings)
    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": "H15 matched-semantic style-invariance representation diagnosis",
        "method_version": int(settings["method_version"]),
        "causal_question": (
            "For identical fresh statistical semantic points, does changing only the rendering "
            "from repository audit prose to researcher-like prose or a concise applied vignette "
            "move the frozen v0.3 final hidden representation across the H14 decision boundary; "
            "and if so, can one fresh style-diverse linear probe recover the method signal without "
            "changing model weights?"
        ),
        "e3_negative_result_fingerprint": e3["result_fingerprint"],
        "e3_report_sha256": sha256_file(e3_path),
        "h14_result_fingerprint": h14["result_fingerprint"],
        "h14_report_sha256": sha256_file(h14_path),
        "h14_contract_fingerprint": h14_contract["fingerprint"],
        "parent": h14_contract["parent"],
        "frozen_h14_head": h14_contract["selector_head"],
        "settings": settings,
        "styles": {
            "audit": "unchanged repository DGP audit rendering used by H13/H14-style cases",
            "researcher": (
                "full natural English research-planning prose with identical declared facts"
            ),
            "vignette": "concise applied-statistics vignette with the same declared facts",
        },
        "blueprint_contracts": blueprint_state["registered"],
        "historical_overlap_audit": blueprint_state["historical_overlap_audit"],
        "representation": {
            "location": (
                "same H13/H14 final normalized hidden state on the last generation-prompt token; "
                "thinking disabled; no catalog"
            ),
            "weights_frozen": True,
            "matched_semantics_across_styles": True,
        },
        "probe_protocol": {
            "eligible_methods": list(h14_contract["selector_head"]["observed_methods"]),
            "frozen_head": "score the immutable H14 ridge head independently in each style",
            "audit_only_probe": (
                "fit only fresh H15 audit-style training representations and select lambda only "
                "on audit-style selection representations; score all styles"
            ),
            "style_diverse_probe": (
                "pool all three renderings of fresh H15 training semantic points and select lambda "
                "on all three renderings of selection semantic points; confirmation semantics "
                "are sealed"
            ),
            "ridge_lambdas": [float(value) for value in settings["ridge_lambdas"]],
        },
        "selection_policy": (
            "Selection may choose style-stable, style-diverse-linear-boundary, or style-sensitive-"
            "representation only through the frozen numeric gates. Otherwise H15 stops before "
            "confirmation."
        ),
        "claim_boundary": (
            "H15 is a synthetic matched-semantics diagnosis. It may localize a rendering-induced "
            "representation shift, but cannot overturn E3, establish external capability, change "
            "v0.3 weights, replace the champion, or authorize release."
        ),
        "implementation_sha256": _implementation_manifest(),
        "confirmation_simulations_opened": False,
        "external_benchmark_opened": False,
    }
    contract["fingerprint"] = canonical_hash(contract)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("H15 contract is immutable")
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
            raise RuntimeError("H15 confirmation cannot open before selection")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if not selection.get("confirmation_authorized"):
            raise RuntimeError("H15 selection did not authorize confirmation")

    shard = dict(contract["settings"][name])
    scenarios = _registered_scenarios(shard, name=f"style-invariance:{name}")
    blueprint_sha = canonical_hash([scenario.to_dict() for scenario in scenarios])
    if blueprint_sha != contract["blueprint_contracts"][name]["blueprint_sha256"]:
        raise RuntimeError(f"H15 registered blueprints changed for {name}")
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
            raise RuntimeError(f"H15 {name} surface changed")
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
        for style in _STYLES
    ]


def _eligible_cases(
    cases: list[dict[str, Any]], observed_methods: set[str]
) -> list[dict[str, Any]]:
    return [case for case in cases if str(case["gold_methods"][0]) in observed_methods]


def _coverage(cases: list[dict[str, Any]], observed_methods: set[str]) -> dict[str, Any]:
    audit_rows = [case for case in cases if case["style"] == "audit"]
    eligible = _eligible_cases(audit_rows, observed_methods)
    counts = Counter(str(case["gold_methods"][0]) for case in eligible)
    return {
        "semantic_point_count": len(audit_rows),
        "eligible_semantic_point_count": len(eligible),
        "eligible_fraction": len(eligible) / max(len(audit_rows), 1),
        "eligible_method_count": len(counts),
        "eligible_method_counts": dict(sorted(counts.items())),
    }


def prepare_style_invariance_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_style_invariance_contract(config)
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
            raise RuntimeError("H15 prepared data changed")
        return existing
    write_json(status_path, status)
    write_json(config.root / "reports" / "evolve" / "style-invariance-v1-data.json", status)
    return status


def _representation_path(config: ProjectConfig, shard: str, style: str) -> Path:
    return _root(config) / "representations" / shard / f"{style}.npz"


def _cases_by_style(
    cases: list[dict[str, Any]], observed_methods: set[str]
) -> dict[str, list[dict[str, Any]]]:
    eligible = _eligible_cases(cases, observed_methods)
    return {style: [case for case in eligible if case["style"] == style] for style in _STYLES}


def _ensure_representations(
    config: ProjectConfig,
    *,
    shard: str,
    cases: list[dict[str, Any]],
    observed_methods: set[str],
) -> dict[str, str]:
    by_style = _cases_by_style(cases, observed_methods)
    paths = {style: _representation_path(config, shard, style) for style in _STYLES}
    if all(path.exists() for path in paths.values()):
        return {style: sha256_file(path) for style, path in paths.items()}
    _, adapter_paths = _expert_context(config)
    agent = StatsAgent(config, adapter_path=adapter_paths["parent"])
    agent.router.set_route("adapter")
    try:
        hashes: dict[str, str] = {}
        for style in _STYLES:
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


def _load_style_representations(
    config: ProjectConfig, shard: str
) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    return {
        style: _load_representations(_representation_path(config, shard, style))
        for style in _STYLES
    }


def _semantic_ids(cases: list[dict[str, Any]]) -> list[str]:
    return [str(case["semantic_id"]) for case in cases]


def _verify_representation_order(
    by_style_cases: dict[str, list[dict[str, Any]]],
    representations: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
) -> None:
    expected_semantics: list[str] | None = None
    for style in _STYLES:
        cases = by_style_cases[style]
        _, _, case_ids = representations[style]
        expected_ids = [str(case["case_id"]) for case in cases]
        if case_ids != expected_ids:
            raise RuntimeError(f"H15 {style} representation order changed")
        semantics = _semantic_ids(cases)
        if expected_semantics is None:
            expected_semantics = semantics
        elif semantics != expected_semantics:
            raise RuntimeError("H15 matched semantic order changed across styles")


def _score_model_by_style(
    model: dict[str, Any],
    representations: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
    *,
    majority_class: int,
) -> dict[str, Any]:
    return {
        style: _probe_metrics(model, vectors, labels, majority_class=majority_class)
        for style, (vectors, labels, _) in representations.items()
    }


def _geometry(
    representations: dict[str, tuple[np.ndarray, np.ndarray, list[str]]]
) -> dict[str, Any]:
    normalized = {
        style: _normalize_rows(vectors)
        for style, (vectors, _, _) in representations.items()
    }
    pairs = (("audit", "researcher"), ("audit", "vignette"), ("researcher", "vignette"))
    pair_reports: dict[str, Any] = {}
    for left, right in pairs:
        same = np.sum(normalized[left] * normalized[right], axis=1)
        shifted = np.sum(normalized[left] * np.roll(normalized[right], 1, axis=0), axis=1)
        pair_reports[f"{left}__{right}"] = {
            "same_semantic_cosine_mean": float(np.mean(same)),
            "shifted_semantic_cosine_mean": float(np.mean(shifted)),
            "matched_margin": float(np.mean(same - shifted)),
        }
    return {"pairs": pair_reports}


def _selection_route(
    frozen: dict[str, Any],
    style_diverse: dict[str, Any],
    gates: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    audit = float(frozen["audit"]["accuracy"])
    natural = min(
        float(frozen["researcher"]["accuracy"]),
        float(frozen["vignette"]["accuracy"]),
    )
    diverse_natural = min(
        float(style_diverse["researcher"]["accuracy"]),
        float(style_diverse["vignette"]["accuracy"]),
    )
    collapse = audit - natural
    recovery = diverse_natural - natural
    max_frozen_gap = max(
        abs(audit - float(frozen["researcher"]["accuracy"])),
        abs(audit - float(frozen["vignette"]["accuracy"])),
    )
    diagnostics = {
        "frozen_audit_accuracy": audit,
        "frozen_natural_min_accuracy": natural,
        "style_diverse_natural_min_accuracy": diverse_natural,
        "frozen_style_collapse_points": 100.0 * collapse,
        "style_diverse_recovery_points": 100.0 * recovery,
        "maximum_frozen_style_gap_points": 100.0 * max_frozen_gap,
    }
    if audit < float(gates["minimum_frozen_audit_accuracy"]):
        return None, diagnostics
    if (
        min(float(frozen[style]["accuracy"]) for style in _STYLES)
        >= float(gates["minimum_frozen_all_style_accuracy"])
        and max_frozen_gap <= float(gates["maximum_stable_style_gap"])
    ):
        return "frozen-head-style-stable", diagnostics
    if collapse >= float(gates["minimum_style_collapse"]):
        if (
            diverse_natural >= float(gates["minimum_style_diverse_natural_accuracy"])
            and recovery >= float(gates["minimum_style_diverse_recovery"])
        ):
            return "style-diverse-linear-boundary", diagnostics
        return "style-sensitive-representation", diagnostics
    return None, diagnostics


def run_style_invariance_selection(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_style_invariance_contract(config)
    data = prepare_style_invariance_data(config)
    if not data.get("selection_authorized"):
        raise RuntimeError("H15 data gate did not authorize representation selection")
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
    train = _load_style_representations(config, "training_shard")
    selection = _load_style_representations(config, "selection_shard")
    train_by_style = _cases_by_style(train_cases, observed)
    select_by_style = _cases_by_style(selection_cases, observed)
    _verify_representation_order(train_by_style, train)
    _verify_representation_order(select_by_style, selection)

    h14_contract = json.loads(
        (config.root / "reports" / "evolve" / "selector-head-v1-contract.json").read_text(
            encoding="utf-8"
        )
    )
    frozen_head = _load_head(h14_contract)
    majority = Counter(int(value) for value in train["audit"][1].tolist()).most_common(1)[0][0]
    frozen_scores = _score_model_by_style(frozen_head, selection, majority_class=majority)

    lambdas = [float(value) for value in contract["settings"]["ridge_lambdas"]]
    audit_lambda, _ = _select_probe(
        train["audit"][0],
        train["audit"][1],
        selection["audit"][0],
        selection["audit"][1],
        lambdas=lambdas,
    )
    audit_probe = _fit_ridge_probe(train["audit"][0], train["audit"][1], ridge_lambda=audit_lambda)
    audit_scores = _score_model_by_style(audit_probe, selection, majority_class=majority)

    train_x = np.concatenate([train[style][0] for style in _STYLES], axis=0)
    train_y = np.concatenate([train[style][1] for style in _STYLES], axis=0)
    selection_x = np.concatenate([selection[style][0] for style in _STYLES], axis=0)
    selection_y = np.concatenate([selection[style][1] for style in _STYLES], axis=0)
    diverse_lambda, _ = _select_probe(
        train_x, train_y, selection_x, selection_y, lambdas=lambdas
    )
    diverse_probe = _fit_ridge_probe(train_x, train_y, ridge_lambda=diverse_lambda)
    diverse_scores = _score_model_by_style(diverse_probe, selection, majority_class=majority)

    route, diagnostics = _selection_route(
        frozen_scores,
        diverse_scores,
        dict(contract["settings"]["selection_gates"]),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "H15 matched-semantic style-invariance selection",
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["fingerprint"],
        "same_parent_weights": True,
        "representations": hashes,
        "eligible_semantic_points": data["coverage"]["selection_shard"][
            "eligible_semantic_point_count"
        ],
        "frozen_h14_head_scores": frozen_scores,
        "audit_only_probe": {"ridge_lambda": audit_lambda, "scores": audit_scores},
        "style_diverse_probe": {"ridge_lambda": diverse_lambda, "scores": diverse_scores},
        "geometry": _geometry(selection),
        "diagnostics": diagnostics,
        "selected_route": route,
        "confirmation_authorized": route is not None,
        "next_step": (
            "open-fresh-h15-confirmation"
            if route is not None
            else "stop-h15-selection-without-opening-confirmation"
        ),
    }
    report["result_fingerprint"] = canonical_hash(report)
    write_json(_root(config) / "selection.json", report)
    write_json(config.root / "reports" / "evolve" / "style-invariance-v1-selection.json", report)
    return report


def _confirmation_gate(
    route: str,
    frozen: dict[str, Any],
    diverse: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    _, diagnostics = _selection_route(frozen, diverse, gates)
    if route == "frozen-head-style-stable":
        checks = {
            "all_style_accuracy": min(float(frozen[style]["accuracy"]) for style in _STYLES)
            >= float(gates["minimum_frozen_all_style_accuracy"]),
            "style_gap": diagnostics["maximum_frozen_style_gap_points"]
            <= 100.0 * float(gates["maximum_stable_style_gap"]),
        }
    elif route == "style-diverse-linear-boundary":
        checks = {
            "style_collapse": diagnostics["frozen_style_collapse_points"]
            >= 100.0 * float(gates["minimum_style_collapse"]),
            "diverse_natural_accuracy": diagnostics["style_diverse_natural_min_accuracy"]
            >= float(gates["minimum_style_diverse_natural_accuracy"]),
            "diverse_recovery": diagnostics["style_diverse_recovery_points"]
            >= 100.0 * float(gates["minimum_style_diverse_recovery"]),
        }
    elif route == "style-sensitive-representation":
        checks = {
            "style_collapse": diagnostics["frozen_style_collapse_points"]
            >= 100.0 * float(gates["minimum_style_collapse"]),
            "diverse_probe_still_below_target": diagnostics["style_diverse_natural_min_accuracy"]
            < float(gates["minimum_style_diverse_natural_accuracy"]),
        }
    else:
        raise ValueError(f"Unknown H15 route: {route}")
    return {"passed": all(checks.values()), "checks": checks, "diagnostics": diagnostics}


def run_style_invariance_confirmation(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_style_invariance_contract(config)
    selection_path = _root(config) / "selection.json"
    if not selection_path.exists():
        raise RuntimeError("H15 confirmation requires completed selection")
    selection_report = json.loads(selection_path.read_text(encoding="utf-8"))
    route = selection_report.get("selected_route")
    if route not in {
        "frozen-head-style-stable",
        "style-diverse-linear-boundary",
        "style-sensitive-representation",
    }:
        raise RuntimeError("H15 selection did not authorize a confirmation route")

    manifest, simulations = _simulate_surface(config, contract, name="confirmation_shard")
    cases = _materialize_cases(simulations)
    case_path = _data_root(config) / "cases" / "confirmation_shard.jsonl"
    write_jsonl(case_path, cases)
    observed = set(str(value) for value in contract["frozen_h14_head"]["observed_methods"])
    coverage = _coverage(cases, observed)
    confirmation_minimum = int(
        contract["settings"]["data_gates"]["minimum_confirmation_eligible_semantic_points"]
    )
    if coverage["eligible_semantic_point_count"] < confirmation_minimum:
        raise RuntimeError("H15 confirmation coverage fell below the preregistered minimum")
    confirmation_hashes = _ensure_representations(
        config, shard="confirmation_shard", cases=cases, observed_methods=observed
    )

    train = _load_style_representations(config, "training_shard")
    selected = _load_style_representations(config, "selection_shard")
    confirmation = _load_style_representations(config, "confirmation_shard")
    _verify_representation_order(_cases_by_style(cases, observed), confirmation)

    h14_contract = json.loads(
        (config.root / "reports" / "evolve" / "selector-head-v1-contract.json").read_text(
            encoding="utf-8"
        )
    )
    frozen_head = _load_head(h14_contract)
    fit_audit_y = np.concatenate([train["audit"][1], selected["audit"][1]], axis=0)
    majority = Counter(int(value) for value in fit_audit_y.tolist()).most_common(1)[0][0]
    frozen_scores = _score_model_by_style(frozen_head, confirmation, majority_class=majority)

    audit_x = np.concatenate([train["audit"][0], selected["audit"][0]], axis=0)
    audit_y = fit_audit_y
    audit_probe = _fit_ridge_probe(
        audit_x,
        audit_y,
        ridge_lambda=float(selection_report["audit_only_probe"]["ridge_lambda"]),
    )
    audit_scores = _score_model_by_style(audit_probe, confirmation, majority_class=majority)

    diverse_x = np.concatenate(
        [train[style][0] for style in _STYLES] + [selected[style][0] for style in _STYLES],
        axis=0,
    )
    diverse_y = np.concatenate(
        [train[style][1] for style in _STYLES] + [selected[style][1] for style in _STYLES],
        axis=0,
    )
    diverse_probe = _fit_ridge_probe(
        diverse_x,
        diverse_y,
        ridge_lambda=float(selection_report["style_diverse_probe"]["ridge_lambda"]),
    )
    diverse_scores = _score_model_by_style(diverse_probe, confirmation, majority_class=majority)
    gate = _confirmation_gate(
        str(route),
        frozen_scores,
        diverse_scores,
        dict(contract["settings"]["confirmation_gates"]),
    )

    if gate["passed"] and route == "frozen-head-style-stable":
        next_step = "stop-style-remapping-and-investigate-external-domain-semantic-shift"
    elif gate["passed"] and route == "style-diverse-linear-boundary":
        next_step = "preregister-style-invariant-selector-head-without-changing-4b-weights"
    elif gate["passed"] and route == "style-sensitive-representation":
        next_step = "preregister-small-contrastive-representation-learning-before-any-9b-scout"
    else:
        next_step = "stop-h15-and-audit-the-style-diagnostic-before-further-training"

    result: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "H15 matched-semantic style-invariance confirmation",
        "contract_fingerprint": contract["fingerprint"],
        "selection_result_fingerprint": selection_report["result_fingerprint"],
        "same_parent_weights": True,
        "confirmation_surface_fingerprint": manifest["fingerprint"],
        "confirmation_case_sha256": sha256_file(case_path),
        "confirmation_coverage": coverage,
        "representations": confirmation_hashes,
        "selected_route": route,
        "frozen_h14_head_scores": frozen_scores,
        "audit_only_probe": {
            "ridge_lambda": selection_report["audit_only_probe"]["ridge_lambda"],
            "scores": audit_scores,
        },
        "style_diverse_probe": {
            "ridge_lambda": selection_report["style_diverse_probe"]["ridge_lambda"],
            "scores": diverse_scores,
        },
        "geometry": _geometry(confirmation),
        "confirmation_gate": gate,
        "h15_diagnosis_confirmed": bool(gate["passed"]),
        "next_step": next_step,
        "champion_changed": False,
        "release_authorized": False,
        "external_benchmark_reopened": False,
    }
    result["result_fingerprint"] = canonical_hash(result)
    write_json(_root(config) / "confirmation.json", result)
    write_json(
        config.root / "reports" / "evolve" / "style-invariance-v1-confirmation.json",
        result,
    )
    return result

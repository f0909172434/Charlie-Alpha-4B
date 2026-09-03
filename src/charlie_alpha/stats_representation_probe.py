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
from .stats_catalog import PROCEDURES
from .stats_catalog_grounding import _messages
from .stats_cross_format import _format_shift_case
from .stats_dgp import Scenario, simulate_scenario
from .stats_family_router import _expert_context
from .stats_router_replication import _historical_scenario_audit, _scenario_semantic_payload

_PROBE_VERSION = 1
_METHOD_IDS = tuple(procedure.method_id for procedure in PROCEDURES)
_METHOD_INDEX = {method_id: index for index, method_id in enumerate(_METHOD_IDS)}


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "representation-probe-v1"


def _data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "representation-probe-v1"


def _implementation_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "stats_representation_probe.py": sha256_file(Path(__file__)),
        "stats_catalog_grounding.py": sha256_file(root / "stats_catalog_grounding.py"),
        "stats_cross_format.py": sha256_file(root / "stats_cross_format.py"),
        "stats_agent.py": sha256_file(root / "stats_agent.py"),
    }


def _h12_scenarios(config: ProjectConfig) -> list[Scenario]:
    path = config.root / "reports" / "evolve" / "catalog-interface-replication-v1-contract.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    scenarios: list[Scenario] = []
    for name, fields in sorted(contract["fold_contracts"].items()):
        shard = _registered_scenarios(
            dict(contract["settings"]["folds"][name]),
            name=f"catalog-interface-replication:{name}",
        )
        actual = canonical_hash([scenario.to_dict() for scenario in shard])
        if actual != str(fields["blueprint_sha256"]):
            raise RuntimeError(f"H12 blueprint reconstruction changed for {name}")
        scenarios.extend(shard)
    return scenarios


def prepare_representation_probe_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("representation_probe"))
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "contract.json"
    public_path = config.root / "reports" / "evolve" / "representation-probe-v1-contract.json"

    h12_path = config.root / "reports" / "evolve" / "catalog-interface-replication-v1.json"
    h12 = json.loads(h12_path.read_text(encoding="utf-8"))
    if not h12.get("synthetic_catalog_interface_replicated"):
        raise RuntimeError("H13 requires the replicated H12 flat-catalog mechanism")
    e2_path = config.root / "reports" / "evolve" / "external-catalog-interface-v2.json"
    e2 = json.loads(e2_path.read_text(encoding="utf-8"))
    if not e2.get("independent_external_interface_supported"):
        raise RuntimeError("H13 requires the completed positive E2-v2 interface result")

    _, adapter_paths = _expert_context(config)
    parent = adapter_paths["parent"]
    parent_sha = sha256_file(parent / "adapters.safetensors")

    registered: dict[str, Any] = {}
    all_scenarios: list[Scenario] = []
    seen: set[str] = set()
    for name in ("training_shard", "selection_shard", "confirmation_shard"):
        shard = dict(settings[name])
        scenarios = _registered_scenarios(shard, name=f"representation-probe:{name}")
        ids = {scenario.blueprint_id for scenario in scenarios}
        if ids & seen:
            raise RuntimeError(f"H13 blueprint overlap at {name}")
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

    previous = _h12_scenarios(config)
    prior_ids = {scenario.blueprint_id for scenario in previous}
    prior_semantics = {
        canonical_hash(_scenario_semantic_payload(scenario.to_dict())) for scenario in previous
    }
    new_ids = {scenario.blueprint_id for scenario in all_scenarios}
    new_semantics = {
        canonical_hash(_scenario_semantic_payload(scenario.to_dict()))
        for scenario in all_scenarios
    }
    if new_ids & prior_ids or new_semantics & prior_semantics:
        raise RuntimeError("H13 overlaps H12 registered blueprints or semantic points")

    audit = _historical_scenario_audit(
        config,
        all_scenarios,
        excluded_root=_data_root(config),
        minimum_normalized_distance=float(settings["minimum_normalized_distance"]),
    )
    if not audit["passed"]:
        raise RuntimeError("H13 blueprints failed historical-overlap audit")

    catalog = [(procedure.method_id, procedure.name) for procedure in PROCEDURES]
    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": "H13 frozen final-representation linear probe",
        "method_version": int(settings["method_version"]),
        "causal_question": (
            "Does the unchanged v0.3 parent already encode the simulator-selected canonical "
            "method in its final normalized hidden state before free generation, and does the "
            "fixed H7 catalog materially change that linear decodability?"
        ),
        "h12_result_fingerprint": h12["result_fingerprint"],
        "h12_report_sha256": sha256_file(h12_path),
        "e2_v2_result_fingerprint": e2["result_fingerprint"],
        "e2_v2_report_sha256": sha256_file(e2_path),
        "parent": {
            "name": "v0.3.0-parent",
            "adapter_path": str(parent),
            "adapter_sha256": parent_sha,
        },
        "settings": settings,
        "blueprint_contracts": registered,
        "historical_overlap_audit": audit,
        "h12_overlap": {
            "prior_blueprints": len(prior_ids),
            "h13_blueprints": len(new_ids),
            "blueprint_id_overlap_count": len(new_ids & prior_ids),
            "semantic_overlap_count": len(new_semantics & prior_semantics),
        },
        "catalog": {
            "method_count": len(_METHOD_IDS),
            "method_ids": list(_METHOD_IDS),
            "sha256": canonical_hash(catalog),
        },
        "representation": {
            "location": (
                "last token of add_generation_prompt with enable_thinking=False, after the "
                "Qwen3.5 text trunk final RMSNorm and before the LM head"
            ),
            "weights_frozen": True,
            "arms": {
                "menu-free": "unchanged H12 menu-free canonical JSON extraction prompt",
                "flat-catalog": "same prompt plus the fixed H7 28-method ID+display-name catalog",
            },
        },
        "probe": {
            "kind": "L2-normalized one-vs-all ridge linear probe with bias",
            "output_slots": 28,
            "lambda_grid": [float(value) for value in settings["ridge_lambdas"]],
            "selection_tie_break": "highest accuracy, then highest top3, then smallest lambda",
            "unseen_training_classes": (
                "retained in the 28-way output schema but masked from argmax; coverage is reported"
            ),
        },
        "selection_policy": {
            "selector_head": (
                "menu-free reaches all registered linear-decoding viability gates"
            ),
            "contrastive_representation": (
                "selector-head fails, while flat-catalog reaches its absolute and paired-lift gates"
            ),
            "none": "otherwise stop H13 without opening confirmation",
        },
        "confirmation_policy": (
            "The confirmation simulation remains unopened until selection chooses exactly one "
            "route. The chosen ridge lambda is frozen by selection; confirmation refits on "
            "training+selection representations and scores the untouched confirmation shard."
        ),
        "claim_boundary": (
            "H13 diagnoses representation accessibility only. It changes no model weight, does "
            "not replace the champion, and does not establish a deployed selector head."
        ),
        "implementation_sha256": _implementation_manifest(),
        "confirmation_simulations_opened": False,
        "external_benchmark_opened": False,
    }
    contract["fingerprint"] = canonical_hash(contract)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("H13 contract is immutable")
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
            raise RuntimeError("H13 confirmation cannot open before selection")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("selected_route") not in {"selector-head", "contrastive-representation"}:
            raise RuntimeError("H13 selection did not authorize confirmation")

    shard = dict(contract["settings"][name])
    scenarios = _registered_scenarios(shard, name=f"representation-probe:{name}")
    actual = canonical_hash([scenario.to_dict() for scenario in scenarios])
    if actual != contract["blueprint_contracts"][name]["blueprint_sha256"]:
        raise RuntimeError(f"H13 registered blueprints changed for {name}")
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
            raise RuntimeError(f"H13 {name} surface changed")
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


def _case_manifest(cases: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [str(case["gold_methods"][0]) for case in cases]
    counts = Counter(labels)
    return {
        "count": len(cases),
        "observed_method_count": len(counts),
        "missing_methods": sorted(set(_METHOD_IDS) - set(counts)),
        "method_counts": dict(sorted(counts.items())),
    }


def prepare_representation_probe_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_representation_probe_contract(config)
    root = _data_root(config)
    status_path = root / "data-status.json"
    manifests: dict[str, Any] = {}
    case_sha: dict[str, str] = {}
    coverage: dict[str, Any] = {}
    for name in ("training_shard", "selection_shard"):
        manifest, simulations = _simulate_surface(config, contract, name=name)
        cases = [_format_shift_case(simulation) for simulation in simulations]
        case_path = root / "cases" / f"{name}.jsonl"
        write_jsonl(case_path, cases)
        manifests[name] = manifest["fingerprint"]
        case_sha[name] = sha256_file(case_path)
        coverage[name] = _case_manifest(cases)
    fingerprint = canonical_hash(
        {"contract": contract["fingerprint"], "manifests": manifests, "cases": case_sha}
    )
    status = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "contract_fingerprint": contract["fingerprint"],
        "surface_fingerprints": manifests,
        "case_sha256": case_sha,
        "coverage": coverage,
        "confirmation_opened": False,
    }
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise RuntimeError("H13 prepared data changed")
        return existing
    write_json(status_path, status)
    write_json(config.root / "reports" / "evolve" / "representation-probe-v1-data.json", status)
    return status


def _representation_prompt(tokenizer: Any, case: dict[str, Any], *, grounded: bool) -> list[int]:
    prompt = tokenizer.apply_chat_template(
        _messages(case, grounded=grounded),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return list(tokenizer.encode(prompt))


def _extract_representations(
    agent: StatsAgent,
    cases: list[dict[str, Any]],
    *,
    grounded: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    language_model = getattr(agent.model, "language_model", None)
    trunk = getattr(language_model, "model", None)
    if trunk is None:
        raise RuntimeError("H13 requires direct access to the Qwen3.5 text trunk")
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    case_ids: list[str] = []
    for index, case in enumerate(cases):
        method_id = str(case["gold_methods"][0])
        if method_id not in _METHOD_INDEX:
            raise RuntimeError(f"H13 encountered non-catalog method {method_id}")
        token_ids = _representation_prompt(agent.tokenizer, case, grounded=grounded)
        tokens = mx.array([token_ids], dtype=mx.int32)
        hidden = trunk(tokens)[0, -1, :].astype(mx.float32)
        mx.eval(hidden)
        vectors.append(np.asarray(hidden, dtype=np.float32))
        labels.append(_METHOD_INDEX[method_id])
        case_ids.append(str(case["case_id"]))
        del tokens, hidden
        if (index + 1) % 16 == 0:
            mx.clear_cache()
    return np.stack(vectors), np.asarray(labels, dtype=np.int64), case_ids


def _save_representations(
    path: Path,
    *,
    vectors: np.ndarray,
    labels: np.ndarray,
    case_ids: list[str],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vectors=vectors.astype(np.float32),
        labels=labels.astype(np.int64),
        case_ids=np.asarray(case_ids),
    )
    return sha256_file(path)


def _load_representations(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(path, allow_pickle=False)
    return (
        np.asarray(data["vectors"], dtype=np.float64),
        np.asarray(data["labels"], dtype=np.int64),
        [str(value) for value in data["case_ids"].tolist()],
    )


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _fit_ridge_probe(
    vectors: np.ndarray,
    labels: np.ndarray,
    *,
    ridge_lambda: float,
) -> dict[str, Any]:
    x = _normalize_rows(np.asarray(vectors, dtype=np.float64))
    x = np.concatenate([x, np.ones((len(x), 1), dtype=np.float64)], axis=1)
    y = np.zeros((len(x), len(_METHOD_IDS)), dtype=np.float64)
    y[np.arange(len(x)), labels] = 1.0
    gram = x @ x.T
    gram.flat[:: len(gram) + 1] += float(ridge_lambda)
    alpha = np.linalg.solve(gram, y)
    weights = x.T @ alpha
    observed = sorted({int(value) for value in labels.tolist()})
    return {"weights": weights, "observed": observed}


def _probe_scores(model: dict[str, Any], vectors: np.ndarray) -> np.ndarray:
    x = _normalize_rows(np.asarray(vectors, dtype=np.float64))
    x = np.concatenate([x, np.ones((len(x), 1), dtype=np.float64)], axis=1)
    scores = x @ np.asarray(model["weights"], dtype=np.float64)
    observed = set(int(value) for value in model["observed"])
    for index in range(len(_METHOD_IDS)):
        if index not in observed:
            scores[:, index] = -np.inf
    return scores


def _probe_metrics(
    model: dict[str, Any],
    vectors: np.ndarray,
    labels: np.ndarray,
    *,
    majority_class: int,
) -> dict[str, Any]:
    scores = _probe_scores(model, vectors)
    predicted = np.argmax(scores, axis=1)
    top3 = np.argpartition(scores, -3, axis=1)[:, -3:]
    correct = predicted == labels
    top3_correct = np.array(
        [
            int(label) in set(int(value) for value in row)
            for label, row in zip(labels, top3, strict=True)
        ]
    )
    present = sorted({int(value) for value in labels.tolist()})
    recalls = [float(np.mean(predicted[labels == value] == value)) for value in present]
    majority_accuracy = float(np.mean(labels == int(majority_class)))
    return {
        "count": int(len(labels)),
        "accuracy": float(np.mean(correct)),
        "top3_accuracy": float(np.mean(top3_correct)),
        "macro_recall_present_classes": float(np.mean(recalls)),
        "present_class_count": len(present),
        "train_observed_class_count": len(model["observed"]),
        "majority_class": _METHOD_IDS[int(majority_class)],
        "majority_accuracy": majority_accuracy,
        "gain_over_majority_points": 100.0 * (float(np.mean(correct)) - majority_accuracy),
    }


def _select_probe(
    train_vectors: np.ndarray,
    train_labels: np.ndarray,
    selection_vectors: np.ndarray,
    selection_labels: np.ndarray,
    *,
    lambdas: list[float],
) -> tuple[float, dict[str, Any]]:
    majority_class = Counter(int(value) for value in train_labels.tolist()).most_common(1)[0][0]
    candidates: list[tuple[float, dict[str, Any]]] = []
    for ridge_lambda in lambdas:
        model = _fit_ridge_probe(train_vectors, train_labels, ridge_lambda=ridge_lambda)
        metrics = _probe_metrics(
            model,
            selection_vectors,
            selection_labels,
            majority_class=majority_class,
        )
        candidates.append((float(ridge_lambda), metrics))
    candidates.sort(
        key=lambda item: (
            -float(item[1]["accuracy"]),
            -float(item[1]["top3_accuracy"]),
            float(item[0]),
        )
    )
    best_lambda, best_metrics = candidates[0]
    report = dict(best_metrics)
    report["ridge_lambda"] = best_lambda
    report["lambda_sweep"] = [
        {
            "ridge_lambda": value,
            "accuracy": metrics["accuracy"],
            "top3_accuracy": metrics["top3_accuracy"],
        }
        for value, metrics in candidates
    ]
    return best_lambda, report


def _choose_route(
    menu: dict[str, Any],
    catalog: dict[str, Any],
    gates: dict[str, Any],
) -> tuple[str | None, dict[str, bool]]:
    selector_checks = {
        "menu_free_accuracy": float(menu["accuracy"]) >= float(gates["minimum_menu_free_accuracy"]),
        "menu_free_top3": float(menu["top3_accuracy"])
        >= float(gates["minimum_menu_free_top3_accuracy"]),
        "menu_free_majority_gain": float(menu["gain_over_majority_points"])
        >= float(gates["minimum_menu_free_gain_over_majority_points"]),
    }
    catalog_checks = {
        "catalog_accuracy": float(catalog["accuracy"]) >= float(gates["minimum_catalog_accuracy"]),
        "catalog_lift": 100.0 * (float(catalog["accuracy"]) - float(menu["accuracy"]))
        >= float(gates["minimum_catalog_gain_over_menu_free_points"]),
    }
    checks = {**selector_checks, **catalog_checks}
    if all(selector_checks.values()):
        return "selector-head", checks
    if all(catalog_checks.values()):
        return "contrastive-representation", checks
    return None, checks


def _representation_paths(config: ProjectConfig, shard: str) -> dict[str, Path]:
    root = _root(config) / "representations" / shard
    return {
        "menu-free": root / "menu-free.npz",
        "flat-catalog": root / "flat-catalog.npz",
    }


def _ensure_representations(
    config: ProjectConfig,
    *,
    shard: str,
    cases: list[dict[str, Any]],
) -> dict[str, str]:
    paths = _representation_paths(config, shard)
    if all(path.exists() for path in paths.values()):
        return {name: sha256_file(path) for name, path in paths.items()}
    _, adapter_paths = _expert_context(config)
    agent = StatsAgent(config, adapter_path=adapter_paths["parent"])
    agent.router.set_route("adapter")
    try:
        hashes: dict[str, str] = {}
        for name, grounded in (("menu-free", False), ("flat-catalog", True)):
            path = paths[name]
            if path.exists():
                hashes[name] = sha256_file(path)
                continue
            vectors, labels, case_ids = _extract_representations(agent, cases, grounded=grounded)
            hashes[name] = _save_representations(
                path, vectors=vectors, labels=labels, case_ids=case_ids
            )
            mx.clear_cache()
        return hashes
    finally:
        del agent
        gc.collect()
        mx.clear_cache()


def run_representation_probe_selection(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_representation_probe_contract(config)
    data = prepare_representation_probe_data(config)
    root = _data_root(config) / "cases"
    train_cases = list(read_jsonl(root / "training_shard.jsonl"))
    selection_cases = list(read_jsonl(root / "selection_shard.jsonl"))
    representation_hashes = {
        "training_shard": _ensure_representations(
            config, shard="training_shard", cases=train_cases
        ),
        "selection_shard": _ensure_representations(
            config, shard="selection_shard", cases=selection_cases
        ),
    }

    lambdas = [float(value) for value in contract["settings"]["ridge_lambdas"]]
    arm_reports: dict[str, Any] = {}
    selected_lambdas: dict[str, float] = {}
    for arm in ("menu-free", "flat-catalog"):
        train_x, train_y, train_ids = _load_representations(
            _representation_paths(config, "training_shard")[arm]
        )
        select_x, select_y, select_ids = _load_representations(
            _representation_paths(config, "selection_shard")[arm]
        )
        if train_ids != [str(case["case_id"]) for case in train_cases]:
            raise RuntimeError(f"H13 {arm} training representation order changed")
        if select_ids != [str(case["case_id"]) for case in selection_cases]:
            raise RuntimeError(f"H13 {arm} selection representation order changed")
        selected_lambda, report = _select_probe(
            train_x,
            train_y,
            select_x,
            select_y,
            lambdas=lambdas,
        )
        selected_lambdas[arm] = selected_lambda
        arm_reports[arm] = report

    selected_route, checks = _choose_route(
        arm_reports["menu-free"],
        arm_reports["flat-catalog"],
        dict(contract["settings"]["selection_gates"]),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "H13 frozen final-representation linear probe selection",
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["fingerprint"],
        "same_parent_weights": True,
        "representations": representation_hashes,
        "scores": arm_reports,
        "paired_catalog_accuracy_gain_points": 100.0
        * (
            float(arm_reports["flat-catalog"]["accuracy"])
            - float(arm_reports["menu-free"]["accuracy"])
        ),
        "selected_lambdas": selected_lambdas,
        "selection_checks": checks,
        "selected_route": selected_route,
        "confirmation_authorized": selected_route is not None,
        "next_step": (
            "confirm-dedicated-selector-head-representation"
            if selected_route == "selector-head"
            else (
                "confirm-catalog-dependent-representation-gap"
                if selected_route == "contrastive-representation"
                else "stop-h13-linear-probe-and-test-nonlinear-probe"
            )
        ),
    }
    report["result_fingerprint"] = canonical_hash(report)
    write_json(_root(config) / "selection.json", report)
    write_json(
        config.root / "reports" / "evolve" / "representation-probe-v1-selection.json",
        report,
    )
    return report


def _confirmation_gate(
    route: str,
    menu: dict[str, Any],
    catalog: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    if route == "selector-head":
        checks = {
            "menu_free_accuracy": float(menu["accuracy"])
            >= float(gates["minimum_menu_free_accuracy"]),
            "menu_free_top3": float(menu["top3_accuracy"])
            >= float(gates["minimum_menu_free_top3_accuracy"]),
            "menu_free_majority_gain": float(menu["gain_over_majority_points"])
            >= float(gates["minimum_menu_free_gain_over_majority_points"]),
        }
    elif route == "contrastive-representation":
        checks = {
            "catalog_accuracy": float(catalog["accuracy"])
            >= float(gates["minimum_catalog_accuracy"]),
            "catalog_lift": 100.0 * (float(catalog["accuracy"]) - float(menu["accuracy"]))
            >= float(gates["minimum_catalog_gain_over_menu_free_points"]),
        }
    else:
        raise ValueError(f"Unknown H13 route: {route}")
    return {"passed": all(checks.values()), "checks": checks}


def run_representation_probe_confirmation(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_representation_probe_contract(config)
    selection_path = _root(config) / "selection.json"
    if not selection_path.exists():
        raise RuntimeError("H13 confirmation requires completed selection")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    route = selection.get("selected_route")
    if route not in {"selector-head", "contrastive-representation"}:
        raise RuntimeError("H13 selection did not authorize confirmation")

    manifest, simulations = _simulate_surface(config, contract, name="confirmation_shard")
    cases = [_format_shift_case(simulation) for simulation in simulations]
    case_path = _data_root(config) / "cases" / "confirmation_shard.jsonl"
    write_jsonl(case_path, cases)
    representation_hashes = _ensure_representations(
        config, shard="confirmation_shard", cases=cases
    )

    train_cases = list(read_jsonl(_data_root(config) / "cases" / "training_shard.jsonl"))
    selection_cases = list(read_jsonl(_data_root(config) / "cases" / "selection_shard.jsonl"))
    scores: dict[str, Any] = {}
    for arm in ("menu-free", "flat-catalog"):
        train_x, train_y, _ = _load_representations(
            _representation_paths(config, "training_shard")[arm]
        )
        select_x, select_y, _ = _load_representations(
            _representation_paths(config, "selection_shard")[arm]
        )
        test_x, test_y, test_ids = _load_representations(
            _representation_paths(config, "confirmation_shard")[arm]
        )
        if test_ids != [str(case["case_id"]) for case in cases]:
            raise RuntimeError(f"H13 {arm} confirmation representation order changed")
        fit_x = np.concatenate([train_x, select_x], axis=0)
        fit_y = np.concatenate([train_y, select_y], axis=0)
        ridge_lambda = float(selection["selected_lambdas"][arm])
        model = _fit_ridge_probe(fit_x, fit_y, ridge_lambda=ridge_lambda)
        majority = Counter(int(value) for value in fit_y.tolist()).most_common(1)[0][0]
        report = _probe_metrics(model, test_x, test_y, majority_class=majority)
        report["ridge_lambda"] = ridge_lambda
        scores[arm] = report

    gate = _confirmation_gate(
        str(route),
        scores["menu-free"],
        scores["flat-catalog"],
        dict(contract["settings"]["confirmation_gates"]),
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "H13 frozen final-representation linear probe confirmation",
        "contract_fingerprint": contract["fingerprint"],
        "selection_result_fingerprint": selection["result_fingerprint"],
        "same_parent_weights": True,
        "confirmation_surface_fingerprint": manifest["fingerprint"],
        "confirmation_case_sha256": sha256_file(case_path),
        "confirmation_coverage": _case_manifest(cases),
        "representations": representation_hashes,
        "fit_counts": {
            "training": len(train_cases),
            "selection": len(selection_cases),
            "confirmation": len(cases),
        },
        "selected_route": route,
        "scores": scores,
        "paired_catalog_accuracy_gain_points": 100.0
        * (float(scores["flat-catalog"]["accuracy"]) - float(scores["menu-free"]["accuracy"])),
        "confirmation_gate": gate,
        "representation_hypothesis_confirmed": bool(gate["passed"]),
        "next_step": (
            "build-preregistered-28-way-selector-head"
            if gate["passed"] and route == "selector-head"
            else (
                "preregister-hard-negative-contrastive-representation-training"
                if gate["passed"] and route == "contrastive-representation"
                else "test-small-nonlinear-probe-before-capacity-or-base-model-claims"
            )
        ),
        "champion_changed": False,
        "external_benchmark_opened": False,
    }
    result["result_fingerprint"] = canonical_hash(result)
    write_json(_root(config) / "confirmation.json", result)
    write_json(
        config.root / "reports" / "evolve" / "representation-probe-v1-confirmation.json",
        result,
    )
    return result

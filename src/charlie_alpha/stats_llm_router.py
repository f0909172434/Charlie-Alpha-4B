from __future__ import annotations

import copy
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm import load

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json
from .stats_calibrate import _surface_comparison
from .stats_cone import _cone_paths
from .stats_evolve import _proposal_records, _score_adapter_surfaces, _score_loaded_selector
from .stats_family_router import (
    _canary_metrics,
    _classification_metrics,
    _ensure_router_shard,
    _expert_context,
    _paired_bootstrap,
    _route_documents,
    _routed_score_from_cached,
    _router_examples,
    _router_gate_results,
    _score_real_route,
    choose_router_threshold,
)
from .stats_training import _stats_snapshot

_ROUTE_MENU: tuple[tuple[str, str | None, str], ...] = (
    (
        "A",
        "group_comparison",
        "difference in continuous means for independent groups or paired before-after values; "
        "not causal identification",
    ),
    (
        "B",
        "categorical",
        "categorical proportions, contingency tables, binary risk difference, or sparse cells",
    ),
    (
        "C",
        "linear_robust",
        "continuous regression slope with heteroskedasticity, outliers, or leverage",
    ),
    (
        "D",
        "binary_count_glm",
        "binary or count regression, logistic/Poisson/negative-binomial model, or separation",
    ),
    (
        "E",
        "clustered_repeated",
        "clustered, nested, or repeated observations with ICC, group IDs, or "
        "within-unit dependence",
    ),
    ("F", "survival", "time-to-event, censoring, hazards, or survival curves"),
    (
        "G",
        "missing_selection",
        "missing outcomes, response mechanisms, selection bias, imputation, or "
        "weighting for missingness",
    ),
    (
        "H",
        "experimental_causal",
        "causal ATE or policy effect under randomization or observational confounding; "
        "not a plain mean difference",
    ),
    (
        "I",
        "probability_distribution",
        "one-sample probability distribution, population location, skew/tails, or "
        "distributional fit; not groups",
    ),
    ("J", "bayesian_check", "priors, posterior estimation, or posterior predictive checks"),
    (
        "K",
        "predictive_calibration",
        "out-of-sample predictions, held-out discrimination, or probability calibration; "
        "not coefficient estimation",
    ),
    (
        "L",
        "time_series_leakage",
        "chronological time series, forecast horizon, rolling validation, drift, or "
        "future-data leakage",
    ),
    (
        "M",
        None,
        "truly insufficient outcome, estimand, sampling-unit, and design information "
        "to choose a family",
    ),
)


def family_route_prompt() -> str:
    return (
        "Classify the statistical question. Return exactly one letter, with no explanation.\n"
        + "\n".join(f"{code}: {description}" for code, _, description in _ROUTE_MENU)
    )


class ParentLetterRouter:
    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = family_route_prompt()
        self.codes = [code for code, _, _ in _ROUTE_MENU]
        self.families = [family or "unknown" for _, family, _ in _ROUTE_MENU]
        encoded = [tokenizer.encode(code, add_special_tokens=False) for code in self.codes]
        if any(len(value) != 1 for value in encoded):
            raise RuntimeError("The locked router codes A-M must each be one tokenizer token")
        self.token_ids = [int(value[0]) for value in encoded]
        if len(set(self.token_ids)) != len(self.token_ids):
            raise RuntimeError("The locked router codes A-M must have distinct token IDs")
        self._cache: dict[str, tuple[str, float, dict[str, float]]] = {}

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        if text in self._cache:
            return self._cache[text]
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = mx.array([self.tokenizer.encode(prompt)], dtype=mx.int32)
        logits = self.model(tokens)[0, -1, :]
        candidate_ids = mx.array(self.token_ids, dtype=mx.int32)
        probabilities = mx.softmax(mx.take(logits, candidate_ids).astype(mx.float32))
        mx.eval(probabilities)
        values = np.asarray(probabilities.tolist(), dtype=np.float64)
        index = int(np.argmax(values))
        result = (
            self.families[index],
            float(values[index]),
            {family: float(values[i]) for i, family in enumerate(self.families)},
        )
        self._cache[text] = result
        del tokens, logits, candidate_ids, probabilities
        return result


def _llm_router_root(config: ProjectConfig) -> Path:
    version = int(config.section("llm_family_router")["prompt_version"])
    return _cone_paths(config)[1] / f"family-llm-router-v{version}"


def _router_contract(
    config: ProjectConfig,
    expert_report: dict[str, Any],
    parent_path: Path,
    validation_manifest: dict[str, Any],
) -> dict[str, Any]:
    settings = dict(config.section("llm_family_router"))
    development_canary = config.root / str(settings["development_canary_path"])
    confirmation_canary = config.root / str(settings["confirmation_canary_path"])
    return {
        "prompt_version": int(settings["prompt_version"]),
        "prompt_sha256": canonical_hash(family_route_prompt()),
        "menu": [
            {"code": code, "family_id": family, "description": description}
            for code, family, description in _ROUTE_MENU
        ],
        "parent_adapter_sha256": sha256_file(parent_path / "adapters.safetensors"),
        "expert_selection_fingerprint": expert_report["selection"]["fingerprint"],
        "validation_surface_fingerprint": validation_manifest["fingerprint"],
        "development_canary_sha256": sha256_file(development_canary),
        "confirmation_canary_sha256": sha256_file(confirmation_canary),
        "settings": settings,
    }


def _score_sparse_validation_adapters(
    config: ProjectConfig,
    simulations: list[dict[str, Any]],
    manifest: dict[str, Any],
    adapter_paths: dict[str, Path],
    expert_mapping: dict[str, dict[str, Any]],
    decisions: dict[tuple[str, str], dict[str, Any]],
    root: Path,
) -> dict[str, dict[str, Any]]:
    score_root = root / "validation-scores"
    simulation_by_id = {
        str(simulation["scenario"]["blueprint_id"]): simulation for simulation in simulations
    }
    needed: dict[str, dict[str, set[str]]] = {
        "parent": {language: set(simulation_by_id) for language in ("en", "zh_Hant", "zh_Hans")}
    }
    for (language, blueprint_id), decision in decisions.items():
        family_id = str(decision["predicted_family_id"])
        if family_id not in expert_mapping:
            continue
        route = expert_mapping[family_id]
        if route["checkpoint_name"] == "parent":
            continue
        needed.setdefault(str(route["slug"]), defaultdict(set))[language].add(blueprint_id)

    scores: dict[str, dict[str, Any]] = {}
    for slug, by_language in sorted(needed.items()):
        adapter_path = adapter_paths[slug]
        fingerprint = canonical_hash(
            {
                "surface": manifest["fingerprint"],
                "adapter": sha256_file(adapter_path / "adapters.safetensors"),
                "coverage": {
                    language: sorted(blueprints)
                    for language, blueprints in sorted(by_language.items())
                },
                "evaluator_version": 1,
            }
        )
        status_path = score_root / f"{slug}.json"
        if status_path.exists():
            existing = json.loads(status_path.read_text(encoding="utf-8"))
            if existing.get("fingerprint") == fingerprint and existing.get("complete"):
                scores[slug] = existing["score"]
                continue
            raise RuntimeError(f"Sparse router validation score changed for {slug}")
        model_instance, tokenizer = load(
            _stats_snapshot(config),
            adapter_path=str(adapter_path),
            tokenizer_config={"trust_remote_code": True},
        )
        languages: dict[str, dict[str, Any]] = {}
        for language, blueprint_ids in sorted(by_language.items()):
            rows = [simulation_by_id[blueprint_id] for blueprint_id in sorted(blueprint_ids)]
            view = "boundary_a" if language == "en" else "standard"
            languages[language] = _score_loaded_selector(
                model_instance,
                tokenizer,
                _proposal_records(rows, language=language, view=view),
            )
        del model_instance, tokenizer
        gc.collect()
        score = {
            "selector": languages.get("en"),
            "languages": languages,
            "retention": None,
        }
        write_json(
            status_path,
            {
                "schema_version": 1,
                "complete": True,
                "fingerprint": fingerprint,
                "adapter_sha256": sha256_file(adapter_path / "adapters.safetensors"),
                "surface_fingerprint": manifest["fingerprint"],
                "score": score,
            },
        )
        scores[slug] = score
    return scores


def _public_report(report: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(report)


def _score_loaded_surface(
    model: Any,
    tokenizer: Any,
    simulations: list[dict[str, Any]],
    *,
    retention: dict[str, Any],
) -> dict[str, Any]:
    languages = {
        language: _score_loaded_selector(
            model,
            tokenizer,
            _proposal_records(simulations, language=language, view=view),
        )
        for language, view in (
            ("en", "boundary_a"),
            ("zh_Hant", "standard"),
            ("zh_Hans", "standard"),
        )
    }
    return {
        "selector": languages["en"],
        "languages": languages,
        "retention": retention,
    }


def _compact_surface_score(score: dict[str, Any]) -> dict[str, Any]:
    selector = score["selector"]
    return {
        "count": int(selector["count"]),
        "normalized_regret": float(selector["normalized_regret"]),
        "accuracy": float(selector["accuracy"]),
        "invalid_selection_rate": float(selector["invalid_selection_rate"]),
        "domain_accuracy": {
            str(key): float(value) for key, value in sorted(selector["domain_accuracy"].items())
        },
        "languages": {
            language: {
                "count": int(result["count"]),
                "normalized_regret": float(result["normalized_regret"]),
                "accuracy": float(result["accuracy"]),
                "invalid_selection_rate": float(result["invalid_selection_rate"]),
            }
            for language, result in sorted(score["languages"].items())
        },
        "retention": {
            "count": int(score["retention"]["count"]),
            "accuracy": float(score["retention"]["accuracy"]),
            "groups": {
                str(key): float(value)
                for key, value in sorted(score["retention"]["groups"].items())
            },
        },
    }


def run_llm_family_router(
    config: ProjectConfig,
    *,
    force: bool = False,
    selection_only: bool = False,
) -> dict[str, Any]:
    settings = dict(config.section("llm_family_router"))
    gates = dict(settings["gates"])
    root = _llm_router_root(config)
    root.mkdir(parents=True, exist_ok=True)
    if force and (root / "confirmation-manifest.json").exists():
        raise RuntimeError("Cannot change the parent-letter router after confirmation was opened")
    expert_report, adapter_paths = _expert_context(config)
    expert_mapping = expert_report["selection"]["mapping"]
    validation_manifest, validation_rows = _ensure_router_shard(
        config,
        root,
        "validation_shard",
        open_if_missing=True,
        section_name="llm_family_router",
    )
    contract = _router_contract(
        config,
        expert_report,
        adapter_paths["parent"],
        validation_manifest,
    )
    contract_fingerprint = canonical_hash(contract)
    contract_status = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": contract_fingerprint,
        "method": "frozen-parent single-token A-M statistical family classifier",
        "contract": contract,
        "confirmation_shard_opened": (root / "confirmation-manifest.json").exists(),
        "promotion_shard_opened": False,
        "sealed_final_surface_opened": False,
    }
    write_json(root / "contract.json", contract_status)

    model_instance, tokenizer = load(
        _stats_snapshot(config),
        adapter_path=str(adapter_paths["parent"]),
        tokenizer_config={"trust_remote_code": True},
    )
    router = ParentLetterRouter(model_instance, tokenizer)
    validation_examples = _router_examples(
        validation_rows,
        view=str(settings["validation_shard"]["view"]),
    )
    validation_classification = _classification_metrics(router, validation_examples)
    decisions = _route_documents(
        router,
        validation_rows,
        view=str(settings["validation_shard"]["view"]),
    )
    scores = _score_sparse_validation_adapters(
        config,
        validation_rows,
        validation_manifest,
        adapter_paths,
        expert_mapping,
        decisions,
        root,
    )
    retention_status = json.loads(
        (_cone_paths(config)[1] / "delta-calibration" / "scale-0p00" / "status.json").read_text(
            encoding="utf-8"
        )
    )
    retention = retention_status["scores"]["valid"]["retention"]
    parent_score = {**scores["parent"], "retention": retention}
    development_canary_rows = list(
        read_jsonl(config.root / str(settings["development_canary_path"]))
    )
    comparisons: list[dict[str, Any]] = []
    for threshold_value in settings["confidence_thresholds"]:
        threshold = float(threshold_value)
        candidate, route_metrics = _routed_score_from_cached(
            scores,
            decisions,
            expert_mapping,
            threshold=threshold,
            retention=retention,
        )
        comparison = _surface_comparison(parent_score, candidate, gates)
        canary = _canary_metrics(
            router,
            development_canary_rows,
            expert_mapping,
            threshold=threshold,
        )
        comparisons.append(
            {
                "threshold": threshold,
                "comparison": comparison,
                "route_metrics": route_metrics,
                "development_canary": canary,
                "gates": _router_gate_results(
                    comparison,
                    route_metrics,
                    canary,
                    gates,
                    minimum_improvement_key="minimum_validation_relative_improvement",
                ),
            }
        )
    selected = choose_router_threshold(comparisons)
    selection_fingerprint = canonical_hash(
        {
            "contract": contract_fingerprint,
            "comparisons": comparisons,
            "selector_version": 1,
        }
    )
    confirmation_exists = (root / "confirmation-manifest.json").exists()
    selection = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": selection_fingerprint,
        "validation_classification": validation_classification,
        "thresholds": comparisons,
        "selected_threshold": float(selected["threshold"]) if selected else None,
        "passed": selected is not None,
        "confirmation_shard_opened": confirmation_exists,
        "promotion_shard_opened": False,
        "sealed_final_surface_opened": False,
    }
    selection_path = root / "selection.json"
    if selection_path.exists() and confirmation_exists:
        existing = json.loads(selection_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != selection_fingerprint:
            raise RuntimeError(
                "Parent-letter router selection changed after confirmation was opened"
            )
    write_json(selection_path, selection)
    report_path = root / "report.json"
    public_path = config.root / "reports" / "evolve" / "family-llm-router.json"
    base_report = {
        "schema_version": 1,
        "complete": True,
        "method": "DGP-Regret selective parent-letter family router",
        "router_contract": contract_status,
        "expert_oracle_fingerprint": expert_report["fingerprint"],
        "selection": selection,
        "promotion_shard_opened": False,
        "sealed_final_surface_opened": False,
        "claim_boundary": (
            "The frozen v0.3 parent predicts one A-M route token, and low confidence or M falls "
            "back to the parent. This adds one short model pass and remains synthetic-first "
            "evidence."
        ),
    }
    if selection_only:
        report = {
            **base_report,
            "fingerprint": canonical_hash(
                {"selection": selection_fingerprint, "evaluator_version": 1}
            ),
            "confirmation": None,
            "proceed_to_promotion": False,
        }
        write_json(report_path, report)
        write_json(public_path, _public_report(report))
        del model_instance, tokenizer, router
        gc.collect()
        return report
    if report_path.exists() and not force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            existing.get("complete")
            and existing.get("selection", {}).get("fingerprint") == selection_fingerprint
            and existing.get("confirmation") is not None
        ):
            existing["selection"] = selection
            write_json(report_path, existing)
            write_json(public_path, _public_report(existing))
            del model_instance, tokenizer, router
            gc.collect()
            return existing
    if selected is None:
        report = {
            **base_report,
            "fingerprint": canonical_hash(
                {"selection": selection_fingerprint, "evaluator_version": 1}
            ),
            "confirmation": None,
            "proceed_to_promotion": False,
        }
        write_json(report_path, report)
        write_json(public_path, _public_report(report))
        del model_instance, tokenizer, router
        gc.collect()
        return report

    confirmation_manifest, confirmation_rows = _ensure_router_shard(
        config,
        root,
        "confirmation_shard",
        open_if_missing=True,
        section_name="llm_family_router",
    )
    selection["confirmation_shard_opened"] = True
    write_json(selection_path, selection)
    split = str(confirmation_manifest["split"])
    parent_confirmation = _score_adapter_surfaces(
        config,
        adapter_paths["parent"],
        {split: confirmation_rows},
    )[split]
    threshold = float(selected["threshold"])
    routed_confirmation, route_metrics = _score_real_route(
        config,
        router,
        confirmation_rows,
        view=str(settings["confirmation_shard"]["view"]),
        expert_mapping=expert_mapping,
        adapter_paths=adapter_paths,
        threshold=threshold,
        retention=parent_confirmation["retention"],
    )
    comparison = _surface_comparison(parent_confirmation, routed_confirmation, gates)
    confirmation_canary_rows = list(
        read_jsonl(config.root / str(settings["confirmation_canary_path"]))
    )
    confirmation_canary = _canary_metrics(
        router,
        confirmation_canary_rows,
        expert_mapping,
        threshold=threshold,
    )
    gate_results = _router_gate_results(
        comparison,
        route_metrics,
        confirmation_canary,
        gates,
        minimum_improvement_key="minimum_confirmation_relative_improvement",
    )
    bootstrap = _paired_bootstrap(
        parent_confirmation,
        routed_confirmation,
        repetitions=int(gates["bootstrap_repetitions"]),
        seed=int(config.section("project")["seed"]),
    )
    gate_results["paired_bootstrap"] = float(bootstrap["ci95_lower"]) >= float(
        gates["bootstrap_ci_lower_floor"]
    )
    passed = all(gate_results.values())
    report = {
        **base_report,
        "fingerprint": canonical_hash(
            {
                "selection": selection_fingerprint,
                "confirmation": confirmation_manifest["fingerprint"],
                "evaluator_version": 1,
            }
        ),
        "selection": selection,
        "confirmation": {
            "manifest": confirmation_manifest,
            "comparison": comparison,
            "route_metrics": route_metrics,
            "confirmation_canary": confirmation_canary,
            "paired_bootstrap": bootstrap,
            "gates": gate_results,
            "passed": passed,
        },
        "proceed_to_promotion": passed,
    }
    write_json(report_path, report)
    write_json(public_path, _public_report(report))
    del model_instance, tokenizer, router
    gc.collect()
    return report


def promote_llm_family_router(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    router_settings = dict(config.section("llm_family_router"))
    promotion_settings = dict(config.section("llm_family_router_promotion"))
    gates = dict(promotion_settings["gates"])
    if int(promotion_settings["prompt_version"]) != int(router_settings["prompt_version"]):
        raise RuntimeError("Promotion prompt version differs from the confirmed router")
    root = _llm_router_root(config)
    router_report_path = root / "report.json"
    if not router_report_path.exists():
        raise RuntimeError("Router promotion requires the completed v2 confirmation")
    router_report = json.loads(router_report_path.read_text(encoding="utf-8"))
    if not router_report.get("proceed_to_promotion") or not router_report.get(
        "confirmation", {}
    ).get("passed"):
        raise RuntimeError("The real router did not pass confirmation")
    threshold = float(router_report["selection"]["selected_threshold"])
    expert_report, adapter_paths = _expert_context(config)
    expert_mapping = expert_report["selection"]["mapping"]
    canary_path = config.root / str(promotion_settings["promotion_canary_path"])
    lock = {
        "schema_version": 1,
        "router_fingerprint": router_report["fingerprint"],
        "router_confirmation_fingerprint": router_report["confirmation"]["manifest"]["fingerprint"],
        "prompt_sha256": canonical_hash(family_route_prompt()),
        "prompt_version": int(promotion_settings["prompt_version"]),
        "threshold": threshold,
        "expert_selection_fingerprint": expert_report["selection"]["fingerprint"],
        "promotion_settings": promotion_settings,
        "promotion_canary_sha256": sha256_file(canary_path),
    }
    lock["fingerprint"] = canonical_hash(lock)
    promotion_root = root / "promotion"
    promotion_root.mkdir(parents=True, exist_ok=True)
    lock_path = promotion_root / "lock.json"
    manifest_path = promotion_root / "promotion-manifest.json"
    if lock_path.exists():
        existing_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing_lock.get("fingerprint") != lock["fingerprint"]:
            if manifest_path.exists():
                raise RuntimeError("Router promotion lock changed after its shard was opened")
            raise RuntimeError("Router promotion lock changed before scoring")
    else:
        write_json(lock_path, lock)

    report_path = promotion_root / "report.json"
    public_path = config.root / "reports" / "evolve" / "family-router-promotion.json"
    manifest, promotion_rows = _ensure_router_shard(
        config,
        promotion_root,
        "promotion_shard",
        open_if_missing=True,
        section_name="llm_family_router_promotion",
    )
    fingerprint = canonical_hash(
        {
            "lock": lock["fingerprint"],
            "surface": manifest["fingerprint"],
            "evaluator_version": 1,
        }
    )
    if report_path.exists() and not force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            write_json(public_path, _public_report(existing))
            return existing
        raise RuntimeError("Router promotion report fingerprint changed")

    model_instance, tokenizer = load(
        _stats_snapshot(config),
        adapter_path=str(adapter_paths["parent"]),
        tokenizer_config={"trust_remote_code": True},
    )
    router = ParentLetterRouter(model_instance, tokenizer)
    split = str(manifest["split"])
    parent = _score_adapter_surfaces(
        config,
        adapter_paths["parent"],
        {split: promotion_rows},
    )[split]
    candidate, route_metrics = _score_real_route(
        config,
        router,
        promotion_rows,
        view=str(promotion_settings["promotion_shard"]["view"]),
        expert_mapping=expert_mapping,
        adapter_paths=adapter_paths,
        threshold=threshold,
        retention=parent["retention"],
    )
    comparison = _surface_comparison(parent, candidate, gates)
    canary_rows = list(read_jsonl(canary_path))
    canary = _canary_metrics(
        router,
        canary_rows,
        expert_mapping,
        threshold=threshold,
    )
    gate_results = _router_gate_results(
        comparison,
        route_metrics,
        canary,
        gates,
        minimum_improvement_key="minimum_promotion_relative_improvement",
    )
    bootstrap = _paired_bootstrap(
        parent,
        candidate,
        repetitions=int(gates["bootstrap_repetitions"]),
        seed=int(config.section("project")["seed"]),
    )
    gate_results["paired_bootstrap"] = float(bootstrap["ci95_lower"]) >= float(
        gates["bootstrap_ci_lower_floor"]
    )
    passed = all(gate_results.values())
    report = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "method": "DGP-Regret selective parent-letter family router promotion",
        "lock": lock,
        "manifest": manifest,
        "comparison": comparison,
        "route_metrics": route_metrics,
        "promotion_canary": canary,
        "paired_bootstrap": bootstrap,
        "gates": gate_results,
        "passed": passed,
        "proceed_to_final_evaluation": passed,
        "promotion_shard_opened": True,
        "sealed_final_surface_opened": False,
        "claim_boundary": (
            "Promotion is a larger synthetic DGP and manual-canary gate. It does not replace "
            "the sealed v0.3 final surface or external statistical benchmarks."
        ),
    }
    write_json(report_path, report)
    write_json(public_path, _public_report(report))
    del model_instance, tokenizer, router
    gc.collect()
    return report


def evaluate_llm_family_router_final(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    router_settings = dict(config.section("llm_family_router"))
    final_settings = dict(config.section("llm_family_router_final"))
    gates = dict(final_settings["gates"])
    if int(final_settings["prompt_version"]) != int(router_settings["prompt_version"]):
        raise RuntimeError("Final prompt version differs from the confirmed router")
    if str(final_settings["view"]) != "scoring":
        raise RuntimeError("The registered final route must use the operational scoring view")

    root = _llm_router_root(config)
    promotion_path = root / "promotion" / "report.json"
    if not promotion_path.exists():
        raise RuntimeError("Final evaluation requires the completed router promotion")
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    if not promotion.get("passed") or not promotion.get("proceed_to_final_evaluation"):
        raise RuntimeError("Router promotion did not authorize opening the final surface")

    router_report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    threshold = float(router_report["selection"]["selected_threshold"])
    expert_report, adapter_paths = _expert_context(config)
    expert_mapping = expert_report["selection"]["mapping"]
    final_path = config.path_for("stats_dir") / "surface" / "final.jsonl"
    evaluation_lock_path = config.path_for("eval_lock")
    evaluation_lock = json.loads(evaluation_lock_path.read_text(encoding="utf-8"))
    if sha256_file(final_path) != str(evaluation_lock["final_dgp"]["surface_sha256"]):
        raise RuntimeError("The sealed v0.3 final surface no longer matches its evaluation lock")
    if int(evaluation_lock["final_dgp"]["count"]) != int(final_settings["expected_final_count"]):
        raise RuntimeError("The sealed final DGP count differs from the registered contract")

    prior_report_paths = {
        "base": config.root / "reports" / "stats" / "generated" / "evaluation-base.json",
        "parent": (config.root / "reports" / "stats" / "generated" / "evaluation-dgp-regret.json"),
    }
    prior_reports = {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in prior_report_paths.items()
    }
    parent_sha = sha256_file(adapter_paths["parent"] / "adapters.safetensors")
    if str(prior_reports["parent"]["adapter_sha256"]) != parent_sha:
        raise RuntimeError("The frozen v0.3 evaluation does not match the routed parent adapter")

    final_root = root / "final"
    final_root.mkdir(parents=True, exist_ok=True)
    lock = {
        "schema_version": 1,
        "router_confirmation_fingerprint": router_report["fingerprint"],
        "promotion_fingerprint": promotion["fingerprint"],
        "prompt_sha256": canonical_hash(family_route_prompt()),
        "prompt_version": int(final_settings["prompt_version"]),
        "threshold": threshold,
        "expert_selection_fingerprint": expert_report["selection"]["fingerprint"],
        "parent_adapter_sha256": parent_sha,
        "base_model": config.sources["models"]["research_base_mlx_4bit"],
        "evaluation_lock_sha256": sha256_file(evaluation_lock_path),
        "evaluation_lock_fingerprint": evaluation_lock["fingerprint"],
        "final_surface_sha256": sha256_file(final_path),
        "final_count": int(evaluation_lock["final_dgp"]["count"]),
        "prior_report_sha256": {
            key: sha256_file(path) for key, path in sorted(prior_report_paths.items())
        },
        "settings": final_settings,
    }
    lock["fingerprint"] = canonical_hash(lock)
    lock_path = final_root / "lock.json"
    if lock_path.exists():
        existing_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing_lock.get("fingerprint") != lock["fingerprint"]:
            raise RuntimeError("The final router contract changed after the surface was opened")
    else:
        # Persist the decision contract before any routed final score is read.
        write_json(lock_path, lock)

    report_path = final_root / "report.json"
    public_path = config.root / "reports" / "evolve" / "family-router-final.json"
    fingerprint = canonical_hash(
        {
            "lock": lock["fingerprint"],
            "adapter_sha256": {
                key: sha256_file(path / "adapters.safetensors")
                for key, path in sorted(adapter_paths.items())
            },
            "evaluator_version": 1,
        }
    )
    if report_path.exists() and not force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            public = copy.deepcopy(existing)
            public.pop("private_scores", None)
            write_json(public_path, public)
            return public
        raise RuntimeError("Final router report fingerprint changed")

    # Reading the final rows begins only after the decision contract above is immutable.
    simulations = list(read_jsonl(final_path))
    locked_ids = list(evaluation_lock["final_dgp"]["blueprint_ids"])
    observed_ids = [str(row["scenario"]["blueprint_id"]) for row in simulations]
    if observed_ids != locked_ids:
        raise RuntimeError("The final DGP ordering differs from the sealed evaluation lock")

    base_model, base_tokenizer = load(
        _stats_snapshot(config),
        tokenizer_config={"trust_remote_code": True},
    )
    base = _score_loaded_surface(
        base_model,
        base_tokenizer,
        simulations,
        retention=prior_reports["base"]["retention"],
    )
    del base_model, base_tokenizer
    gc.collect()
    mx.clear_cache()

    parent_model, parent_tokenizer = load(
        _stats_snapshot(config),
        adapter_path=str(adapter_paths["parent"]),
        tokenizer_config={"trust_remote_code": True},
    )
    parent = _score_loaded_surface(
        parent_model,
        parent_tokenizer,
        simulations,
        retention=prior_reports["parent"]["retention"],
    )
    router = ParentLetterRouter(parent_model, parent_tokenizer)
    candidate, route_metrics = _score_real_route(
        config,
        router,
        simulations,
        view=str(final_settings["view"]),
        expert_mapping=expert_mapping,
        adapter_paths=adapter_paths,
        threshold=threshold,
        retention=parent["retention"],
    )

    parent_comparison = _surface_comparison(parent, candidate, gates)
    base_comparison = _surface_comparison(base, candidate, gates)
    parent_bootstrap = _paired_bootstrap(
        parent,
        candidate,
        repetitions=int(gates["bootstrap_repetitions"]),
        seed=int(config.section("project")["seed"]),
    )
    base_bootstrap = _paired_bootstrap(
        base,
        candidate,
        repetitions=int(gates["bootstrap_repetitions"]),
        seed=int(config.section("project")["seed"]) + 1,
    )
    base_invalidity = float(base["selector"]["invalid_selection_rate"])
    candidate_invalidity = float(candidate["selector"]["invalid_selection_rate"])
    invalidity_reduction = (
        (base_invalidity - candidate_invalidity) / base_invalidity if base_invalidity else 0.0
    )
    gate_results = {
        **{f"parent_{key}": bool(value) for key, value in parent_comparison["gates"].items()},
        "parent_relative_regret": float(parent_comparison["trilingual_relative_regret_improvement"])
        >= float(gates["minimum_parent_relative_improvement"]),
        "base_relative_regret": float(base_comparison["trilingual_relative_regret_improvement"])
        >= float(gates["minimum_base_relative_improvement"]),
        "base_invalidity_reduction": invalidity_reduction
        >= float(gates["minimum_base_invalid_selection_relative_reduction"]),
        "router_family_accuracy": float(route_metrics["family_accuracy"])
        >= float(gates["minimum_router_family_accuracy"]),
        "router_language_accuracy": all(
            float(value) >= float(gates["minimum_language_router_accuracy"])
            for value in route_metrics["language_family_accuracy"].values()
        ),
        "wrong_expert_rate": float(route_metrics["wrong_expert_rate"])
        <= float(gates["maximum_wrong_expert_rate"]),
        "expert_coverage": float(route_metrics["expert_coverage"])
        >= float(gates["minimum_expert_coverage"]),
        "parent_paired_bootstrap": float(parent_bootstrap["ci95_lower"])
        >= float(gates["bootstrap_ci_lower_floor"]),
        "base_paired_bootstrap": float(base_bootstrap["ci95_lower"])
        >= float(gates["bootstrap_ci_lower_floor"]),
    }
    passed = all(gate_results.values())
    report = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "method": "DGP-Regret selective parent-letter family router final evaluation",
        "lock": lock,
        "absolute_metrics": {
            "base": _compact_surface_score(base),
            "v0.3-parent": _compact_surface_score(parent),
            "routed-experts": _compact_surface_score(candidate),
        },
        "parent_comparison": parent_comparison,
        "base_comparison": base_comparison,
        "base_invalid_selection_relative_reduction": invalidity_reduction,
        "route_metrics": route_metrics,
        "paired_bootstrap": {
            "vs_parent": parent_bootstrap,
            "vs_base": base_bootstrap,
        },
        "gates": gate_results,
        "passed": passed,
        "proceed_to_external_evaluation": passed,
        "sealed_final_surface_opened": True,
        "external_benchmarks_opened": False,
        "claim_boundary": (
            "This final result covers the locked synthetic DGP selector surface. External "
            "P-Bench, StatQA, clarification, retention, runtime, and clean-load gates remain "
            "separate and are required before default-route or weight promotion."
        ),
        "private_scores": {
            "base": base,
            "v0.3-parent": parent,
            "routed-experts": candidate,
        },
    }
    write_json(report_path, report)
    public = copy.deepcopy(report)
    public.pop("private_scores")
    write_json(public_path, public)
    del parent_model, parent_tokenizer, router
    gc.collect()
    mx.clear_cache()
    return public

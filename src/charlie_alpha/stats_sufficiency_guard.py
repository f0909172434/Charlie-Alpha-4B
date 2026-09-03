from __future__ import annotations

import copy
import gc
import json
from collections import defaultdict
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm import load

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json
from .stats_calibrate import _surface_comparison
from .stats_data import _build_record, _scenario
from .stats_dgp import build_blueprints
from .stats_family_router import (
    _ensure_router_shard,
    _expert_context,
    _paired_bootstrap,
    _route_documents,
    _routed_score_from_cached,
    router_question,
)
from .stats_llm_router import (
    ParentLetterRouter,
    _compact_surface_score,
    _llm_router_root,
    _score_sparse_validation_adapters,
    family_route_prompt,
)
from .stats_route import _aggregate_predictions
from .stats_router_reduced import _reduced_mapping
from .stats_router_replication import (
    _historical_scenario_audit,
    _replication_gate_results,
    _surface_fingerprint,
    paired_power_sample_size,
)
from .stats_training import _stats_snapshot


def sufficiency_prompt() -> str:
    return (
        "Classify whether the statistical question contains enough information to choose a "
        "defensible primary analysis.\n"
        "A: Sufficient. The outcome type, estimand, sampling unit and dependence, and relevant "
        "design details (such as assignment, missingness, censoring, or validation split) are "
        "stated.\n"
        "B: Insufficient. At least one required design fact is missing, so the analyst must ask "
        "a clarifying question rather than choose a method.\n"
        "Return exactly one letter: A or B."
    )


class ParentSufficiencyGuard:
    def __init__(self, model: Any, tokenizer: Any, *, threshold: float = 0.5) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("Sufficiency threshold must lie strictly between zero and one")
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.system_prompt = sufficiency_prompt()
        encoded = [tokenizer.encode(code, add_special_tokens=False) for code in ("A", "B")]
        if any(len(value) != 1 for value in encoded):
            raise RuntimeError("Sufficiency codes A and B must each be one tokenizer token")
        self.token_ids = [int(value[0]) for value in encoded]
        self._cache: dict[str, tuple[bool, float]] = {}

    def predict(self, text: str) -> tuple[bool, float]:
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
        probabilities = mx.softmax(
            mx.take(logits, mx.array(self.token_ids, dtype=mx.int32)).astype(mx.float32)
        )
        mx.eval(probabilities)
        values = np.asarray(probabilities.tolist(), dtype=np.float64)
        result = (bool(float(values[1]) >= self.threshold), float(values[1]))
        self._cache[text] = result
        del tokens, logits, probabilities
        return result


def _guard_examples(
    rows: list[dict[str, Any]],
    complete_decisions: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    examples = [
        {
            "blueprint_id": decision["blueprint_id"],
            "family_id": decision["family_id"],
            "language": decision["language"],
            "incomplete": False,
            "text": decision["text"],
        }
        for decision in complete_decisions.values()
    ]
    for simulation in rows:
        scenario = _scenario(simulation["scenario"])
        for language in ("en", "zh_Hant", "zh_Hans"):
            view = "boundary_a" if language == "en" else "standard"
            record = _build_record(
                scenario,
                simulation,
                language=language,
                loss_weight=1.0,
                incomplete=True,
                variant="dgp-regret",
                refined_explanation=None,
                view=view,
            )
            examples.append(
                {
                    "blueprint_id": scenario.blueprint_id,
                    "family_id": scenario.family_id,
                    "language": language,
                    "incomplete": True,
                    "text": router_question(record),
                }
            )
    return examples


def _guard_metrics(
    guard: ParentSufficiencyGuard,
    examples: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[str, str], bool]]:
    decisions: dict[tuple[str, str], bool] = {}
    grouped: dict[bool, dict[str, dict[str, list[bool]]]] = {
        False: {"language": defaultdict(list), "family": defaultdict(list)},
        True: {"language": defaultdict(list), "family": defaultdict(list)},
    }
    confidence: dict[bool, list[float]] = defaultdict(list)
    for example in examples:
        predicted, probability = guard.predict(str(example["text"]))
        truth = bool(example["incomplete"])
        correct = predicted == truth
        grouped[truth]["language"][str(example["language"])].append(correct)
        grouped[truth]["family"][str(example["family_id"])].append(correct)
        confidence[truth].append(probability)
        if not truth:
            decisions[(str(example["language"]), str(example["blueprint_id"]))] = predicted

    def summarize(truth: bool) -> dict[str, Any]:
        values = [value for group in grouped[truth]["language"].values() for value in group]
        return {
            "count": len(values),
            "accuracy": float(np.mean(values)),
            "language_accuracy": {
                key: float(np.mean(value))
                for key, value in sorted(grouped[truth]["language"].items())
            },
            "family_accuracy": {
                key: float(np.mean(value))
                for key, value in sorted(grouped[truth]["family"].items())
            },
            "mean_insufficient_probability": float(np.mean(confidence[truth])),
        }

    return (
        {
            "complete_specificity": summarize(False),
            "incomplete_sensitivity": summarize(True),
        },
        decisions,
    )


def _threshold_guard_metrics(
    predictions: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    by_truth: dict[bool, dict[str, dict[str, list[bool]]]] = {
        False: {"language": defaultdict(list), "family": defaultdict(list)},
        True: {"language": defaultdict(list), "family": defaultdict(list)},
    }
    for row in predictions:
        truth = bool(row["incomplete"])
        predicted = float(row["insufficient_probability"]) >= threshold
        correct = predicted == truth
        by_truth[truth]["language"][str(row["language"])].append(correct)
        by_truth[truth]["family"][str(row["family_id"])].append(correct)

    def summarize(truth: bool) -> dict[str, Any]:
        values = [value for group in by_truth[truth]["language"].values() for value in group]
        return {
            "accuracy": float(np.mean(values)),
            "minimum_language_accuracy": min(
                float(np.mean(value)) for value in by_truth[truth]["language"].values()
            ),
            "minimum_family_accuracy": min(
                float(np.mean(value)) for value in by_truth[truth]["family"].values()
            ),
        }

    return {
        "threshold": threshold,
        "complete_specificity": summarize(False),
        "incomplete_sensitivity": summarize(True),
    }


def diagnose_sufficiency_guard_margin(config: ProjectConfig) -> dict[str, Any]:
    settings = config.section("sufficiency_guard")
    diagnosis_settings = config.section("sufficiency_guard_diagnosis")
    gates = settings["gates"]
    root = _llm_router_root(config) / "sufficiency-guard-v1"
    report_path = root / "report.json"
    output_path = root / "margin-diagnosis.json"
    public_path = config.root / "reports" / "evolve" / "sufficiency-guard-margin.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("passed"):
        raise RuntimeError("Guard passed; margin failure diagnosis is not applicable")
    rows = list(read_jsonl(root / "confirmation.jsonl"))
    expert_report, adapter_paths = _expert_context(config)
    del expert_report
    model, tokenizer = load(
        _stats_snapshot(config),
        adapter_path=str(adapter_paths["parent"]),
        tokenizer_config={"trust_remote_code": True},
    )
    router = ParentLetterRouter(model, tokenizer)
    guard = ParentSufficiencyGuard(model, tokenizer)
    family_decisions = _route_documents(
        router,
        rows,
        view=str(settings["confirmation_shard"]["view"]),
    )
    examples = _guard_examples(rows, family_decisions)
    predictions: list[dict[str, Any]] = []
    for example in examples:
        _, probability = guard.predict(str(example["text"]))
        predictions.append({**example, "insufficient_probability": probability})
    candidates = [
        _threshold_guard_metrics(predictions, threshold=float(value))
        for value in diagnosis_settings["thresholds"]
    ]
    for candidate in candidates:
        specificity = candidate["complete_specificity"]
        sensitivity = candidate["incomplete_sensitivity"]
        candidate["gates"] = {
            "specificity": specificity["accuracy"] >= gates["minimum_guard_specificity"],
            "language_specificity": specificity["minimum_language_accuracy"]
            >= gates["minimum_language_guard_specificity"],
            "family_specificity": specificity["minimum_family_accuracy"]
            >= gates["minimum_family_guard_specificity"],
            "sensitivity": sensitivity["accuracy"] >= gates["minimum_guard_sensitivity"],
            "language_sensitivity": sensitivity["minimum_language_accuracy"]
            >= gates["minimum_language_guard_sensitivity"],
            "family_sensitivity": sensitivity["minimum_family_accuracy"]
            >= gates["minimum_family_guard_sensitivity"],
        }
        candidate["eligible"] = all(candidate["gates"].values())
    eligible = [value for value in candidates if value["eligible"]]
    selected = (
        min(
            eligible,
            key=lambda value: (
                -float(value["incomplete_sensitivity"]["accuracy"]),
                -float(value["complete_specificity"]["accuracy"]),
                float(value["threshold"]),
            ),
        )
        if eligible
        else None
    )
    diagnosis = {
        "schema_version": 1,
        "complete": True,
        "method": "post-rejection insufficiency-probability margin diagnosis",
        "source_report_fingerprint": report["fingerprint"],
        "source_report_sha256": sha256_file(report_path),
        "source_surface_sha256": report["manifest"]["sha256"],
        "prompt_sha256": canonical_hash(sufficiency_prompt()),
        "thresholds": candidates,
        "selected_threshold": float(selected["threshold"]) if selected else None,
        "prospective_threshold_exists": selected is not None,
        "next_hypothesis": (
            "A prospectively thresholded guard may separate incomplete prompts from complete "
            "English boundary views and requires a new fresh confirmation."
            if selected
            else "No robust probability margin exists; retire the sufficiency-guard direction."
        ),
        "claim_boundary": (
            "Threshold selection uses the retired failed confirmation and is development-only. "
            "It cannot alter or rescue that result."
        ),
    }
    diagnosis["fingerprint"] = canonical_hash(diagnosis)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != diagnosis["fingerprint"]:
            raise RuntimeError("Sufficiency margin diagnosis changed")
        write_json(public_path, existing)
        return existing
    write_json(output_path, diagnosis)
    write_json(public_path, diagnosis)
    del model, tokenizer, router, guard
    gc.collect()
    mx.clear_cache()
    return diagnosis


def _apply_guard(
    candidate: dict[str, Any],
    guard_decisions: dict[tuple[str, str], bool],
) -> dict[str, Any]:
    guarded = copy.deepcopy(candidate)
    for language, result in guarded["languages"].items():
        rows: list[dict[str, Any]] = []
        for prediction in result["predictions"]:
            key = (str(language), str(prediction["blueprint_id"]))
            if guard_decisions[key]:
                rows.append(
                    {
                        **prediction,
                        "predicted_method_id": "needs_clarification",
                        "normalized_regret": 1.0,
                        "valid": False,
                    }
                )
            else:
                rows.append(prediction)
        guarded["languages"][language] = _aggregate_predictions(rows)
    guarded["selector"] = guarded["languages"]["en"]
    return guarded


def prepare_sufficiency_guard_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = copy.deepcopy(config.section("sufficiency_guard"))
    root = _llm_router_root(config) / "sufficiency-guard-v1"
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "lock.json"
    public_path = config.root / "reports" / "evolve" / "sufficiency-guard-contract.json"
    reduced_path = _llm_router_root(config) / "reduced-route-v1" / "report.json"
    reduced = json.loads(reduced_path.read_text(encoding="utf-8"))
    if not reduced.get("passed"):
        raise RuntimeError("Sufficiency guard requires the confirmed reduced route")
    replication_lock = json.loads(
        (_llm_router_root(config) / "independent-replication-v1" / "lock.json").read_text(
            encoding="utf-8"
        )
    )
    shard = settings["confirmation_shard"]
    scenarios = build_blueprints(
        {str(shard["split"]): int(shard["count"])},
        seed=int(shard["seed"]),
        active_search=False,
    )
    historical = replication_lock["power_analysis"]["historical_inputs"]
    power_settings = settings["power"]
    power = paired_power_sample_size(
        paired_sd=float(historical["paired_difference_sd"]),
        parent_mean_regret=float(historical["parent_mean_regret"]),
        minimum_relative_improvement=float(power_settings["minimum_relative_improvement"]),
        alpha=float(power_settings["alpha_two_sided"]),
        power=float(power_settings["target_power"]),
        safety_margin=float(power_settings["safety_margin"]),
        allocation_multiple=int(power_settings["allocation_multiple"]),
    )
    if int(shard["count"]) < int(power["registered_minimum_blueprints"]):
        raise RuntimeError("Guard confirmation is smaller than its efficacy power analysis")
    expert_report, adapter_paths = _expert_context(config)
    mapping = _reduced_mapping(
        expert_report["selection"]["mapping"],
        str(settings["excluded_family"]),
    )
    fields = {
        "schema_version": 1,
        "method": "preregistered parent-logit sufficiency guard confirmation",
        "research_question": (
            "Can a frozen one-token sufficiency guard recover clarification safety without "
            "sacrificing the independently confirmed reduced-route DGP gain?"
        ),
        "candidate": {
            "reduced_route_fingerprint": reduced["fingerprint"],
            "reduced_route_sha256": sha256_file(reduced_path),
            "excluded_family": settings["excluded_family"],
            "mapping_fingerprint": canonical_hash(mapping),
            "family_prompt_sha256": canonical_hash(family_route_prompt()),
            "sufficiency_prompt": sufficiency_prompt(),
            "sufficiency_prompt_sha256": canonical_hash(sufficiency_prompt()),
            "decision": "argmax A=sufficient versus B=insufficient; no threshold",
            "adapter_sha256": {
                slug: sha256_file(path / "adapters.safetensors")
                for slug, path in sorted(adapter_paths.items())
            },
        },
        "control": {
            "name": "v0.3-parent",
            "adapter_sha256": sha256_file(adapter_paths["parent"] / "adapters.safetensors"),
        },
        "development_evidence": {
            "source": "retired reduced-route confirmation rendered in complete/incomplete pairs",
            "source_surface_sha256": reduced["manifest"]["sha256"],
            "prompt_candidates_evaluated": 1,
            "complete_count": 2700,
            "complete_specificity": 1.0,
            "incomplete_count": 2700,
            "incomplete_sensitivity": 1.0,
            "all_language_and_family_point_accuracies": 1.0,
            "confirmatory_use": False,
        },
        "settings": settings,
        "power_analysis": {
            "efficacy_method": "paired normal approximation with 20% safety margin",
            "historical_inputs": historical,
            **power,
            "registered_blueprints": int(shard["count"]),
            "safety_rationale": (
                "Every family-language cell has at least 54 examples per class; zero errors then "
                "bounds its error rate near 3/54 under the rule of three."
            ),
        },
        "blueprint_fingerprint": canonical_hash([scenario.to_dict() for scenario in scenarios]),
        "anticipated_surface_fingerprint": _surface_fingerprint(config, shard, scenarios),
        "adaptation_policy": (
            "none after lock; prompts, mappings, thresholds, gates, and size fixed"
        ),
        "stopping_rule": "evaluate every complete and incomplete registered rendering once",
        "decision_rule": (
            "Pass only if guard sensitivity/specificity and every reduced-route efficacy, "
            "granular, routing, integrity, and bootstrap gate passes."
        ),
    }
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        payload = {key: value for key, value in existing.items() if key != "fingerprint"}
        if canonical_hash(payload) != existing.get("fingerprint"):
            raise RuntimeError("Sufficiency-guard lock fingerprint is corrupt")
        for key in fields:
            if existing.get(key) != fields[key]:
                raise RuntimeError(f"Frozen sufficiency-guard lock changed: {key}")
        write_json(public_path, existing)
        return existing
    if (root / "confirmation.jsonl").exists() or (root / "confirmation-manifest.json").exists():
        raise RuntimeError("Cannot preregister after guard confirmation was opened")
    audit = _historical_scenario_audit(
        config,
        scenarios,
        excluded_root=root,
        minimum_normalized_distance=float(settings["minimum_normalized_distance"]),
    )
    if not audit["passed"]:
        raise RuntimeError("Guard confirmation failed the fresh-surface audit")
    lock = {**fields, "integrity_audit": audit}
    lock["fingerprint"] = canonical_hash(lock)
    write_json(lock_path, lock)
    write_json(public_path, lock)
    return lock


def run_sufficiency_guard_confirmation(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    lock = prepare_sufficiency_guard_contract(config)
    settings = config.section("sufficiency_guard")
    gates = dict(settings["gates"])
    root = _llm_router_root(config) / "sufficiency-guard-v1"
    report_path = root / "report.json"
    public_path = config.root / "reports" / "evolve" / "sufficiency-guard.json"
    manifest, rows = _ensure_router_shard(
        config,
        root,
        "confirmation_shard",
        open_if_missing=True,
        section_name="sufficiency_guard",
    )
    if manifest["fingerprint"] != lock["anticipated_surface_fingerprint"]:
        raise RuntimeError("Guard surface differs from its preregistration")
    fingerprint = canonical_hash(
        {"lock": lock["fingerprint"], "surface": manifest["fingerprint"], "evaluator_version": 1}
    )
    if report_path.exists() and not force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            public = copy.deepcopy(existing)
            public.pop("private_scores", None)
            write_json(public_path, public)
            return public
        raise RuntimeError("Sufficiency-guard report changed")
    expert_report, adapter_paths = _expert_context(config)
    mapping = _reduced_mapping(
        expert_report["selection"]["mapping"],
        str(settings["excluded_family"]),
    )
    model, tokenizer = load(
        _stats_snapshot(config),
        adapter_path=str(adapter_paths["parent"]),
        tokenizer_config={"trust_remote_code": True},
    )
    router = ParentLetterRouter(model, tokenizer)
    guard = ParentSufficiencyGuard(model, tokenizer)
    family_decisions = _route_documents(
        router,
        rows,
        view=str(settings["confirmation_shard"]["view"]),
    )
    guard_metrics, complete_guard_decisions = _guard_metrics(
        guard,
        _guard_examples(rows, family_decisions),
    )
    scores = _score_sparse_validation_adapters(
        config,
        rows,
        manifest,
        adapter_paths,
        mapping,
        family_decisions,
        root,
    )
    prior = json.loads(
        (_llm_router_root(config) / "final" / "report.json").read_text(encoding="utf-8")
    )
    retention = copy.deepcopy(prior["private_scores"]["v0.3-parent"]["retention"])
    parent = {**scores["parent"], "retention": retention}
    unguarded, route_metrics = _routed_score_from_cached(
        scores,
        family_decisions,
        mapping,
        threshold=float(settings["expected_selected_threshold"]),
        retention=retention,
    )
    candidate = _apply_guard(unguarded, complete_guard_decisions)
    comparison = _surface_comparison(parent, candidate, gates)
    bootstrap = _paired_bootstrap(
        parent,
        candidate,
        repetitions=int(gates["bootstrap_repetitions"]),
        seed=int(settings["bootstrap_seed"]),
    )
    gate_results = _replication_gate_results(
        comparison,
        route_metrics,
        bootstrap,
        gates,
        bool(lock["integrity_audit"]["passed"]),
    )
    specificity = guard_metrics["complete_specificity"]
    sensitivity = guard_metrics["incomplete_sensitivity"]
    gate_results.update(
        {
            "guard_specificity": float(specificity["accuracy"])
            >= float(gates["minimum_guard_specificity"]),
            "guard_language_specificity": min(specificity["language_accuracy"].values())
            >= float(gates["minimum_language_guard_specificity"]),
            "guard_family_specificity": min(specificity["family_accuracy"].values())
            >= float(gates["minimum_family_guard_specificity"]),
            "guard_sensitivity": float(sensitivity["accuracy"])
            >= float(gates["minimum_guard_sensitivity"]),
            "guard_language_sensitivity": min(sensitivity["language_accuracy"].values())
            >= float(gates["minimum_language_guard_sensitivity"]),
            "guard_family_sensitivity": min(sensitivity["family_accuracy"].values())
            >= float(gates["minimum_family_guard_sensitivity"]),
        }
    )
    passed = all(gate_results.values())
    report = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "method": "preregistered reduced route with parent-logit sufficiency guard",
        "contract_fingerprint": lock["fingerprint"],
        "manifest": manifest,
        "guard_metrics": guard_metrics,
        "absolute_metrics": {
            "v0.3-parent": _compact_surface_score(parent),
            "guarded-reduced-route": _compact_surface_score(candidate),
        },
        "comparison": comparison,
        "route_metrics": route_metrics,
        "paired_bootstrap": bootstrap,
        "gates": gate_results,
        "passed": passed,
        "candidate_status": "guard-confirmed" if passed else "guard-rejected",
        "proceed_to_historical_external_benchmarks": passed,
        "automatic_champion_promotion": False,
        "fresh_confirmation_surface_opened": True,
        "fresh_confirmation_surface_retired": True,
        "claim_boundary": (
            "This fresh synthetic paired confirmation tests method selection and clarification "
            "guarding. Historical external benchmarks remain a separate development gate."
        ),
        "private_scores": {
            "v0.3-parent": parent,
            "unguarded-reduced-route": unguarded,
            "guarded-reduced-route": candidate,
        },
    }
    write_json(report_path, report)
    public = copy.deepcopy(report)
    public.pop("private_scores")
    write_json(public_path, public)
    del model, tokenizer, router, guard
    gc.collect()
    mx.clear_cache()
    return public

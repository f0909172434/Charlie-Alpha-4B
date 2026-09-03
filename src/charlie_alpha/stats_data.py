from __future__ import annotations

import gc
import json
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from huggingface_hub import snapshot_download

from .config import ProjectConfig
from .io_utils import (
    append_jsonl,
    canonical_hash,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from .stats_catalog import FAMILY_BY_ID, PROCEDURE_BY_ID, catalog_manifest
from .stats_dgp import Scenario, build_blueprints, central_validity_checks, simulate_scenario

_LANGUAGES = ("en", "zh_Hant", "zh_Hans")
_LABELS = ("A", "B", "C", "D", "E", "F")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")


def _chat_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    token_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise RuntimeError("Qwen3.5 tokenizer did not return an input_ids sequence")
    if token_ids and isinstance(token_ids[0], Sequence):
        if len(token_ids) != 1:
            raise RuntimeError("Stats data tokenization unexpectedly returned a batch")
        token_ids = token_ids[0]
    return len(token_ids)


def _scenario(row: dict[str, Any]) -> Scenario:
    return Scenario(
        blueprint_id=str(row["blueprint_id"]),
        family_id=str(row["family_id"]),
        split=str(row["split"]),
        seed=int(row["seed"]),
        parameters={key: float(value) for key, value in row["parameters"].items()},
        boundary_round=int(row["boundary_round"]),
        domain=str(row["domain"]),
        search=dict(row["search"]) if row.get("search") else None,
    )


def prepare_stats_blueprints(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    settings = config.section("stats_data")
    stats_dir = config.path_for("stats_dir")
    blueprint_dir = stats_dir / "blueprints"
    manifest_path = blueprint_dir / "manifest.json"
    split_counts = {
        "train": int(settings["train_semantic_groups"]),
        "valid": int(settings["valid_semantic_groups"]),
        "dev": int(settings["dev_dgps"]),
        "final": int(settings["final_dgps"]),
    }
    fingerprint = canonical_hash(
        {
            "split_counts": split_counts,
            "seed": int(settings["split_seed"]),
            "catalog": catalog_manifest(),
            "random_ablation_train_groups": int(settings["train_semantic_groups"]),
            "generator_version": 3,
        }
    )
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        core_valid = all(
            (blueprint_dir / f"{split}.jsonl").exists()
            and existing.get("split_sha256", {}).get(split)
            == sha256_file(blueprint_dir / f"{split}.jsonl")
            for split in split_counts
        )
        random_path = blueprint_dir / "random" / "train.jsonl"
        random_valid = (
            random_path.exists()
            and existing.get("random_ablation_train_sha256") == sha256_file(random_path)
        )
        if existing.get("fingerprint") == fingerprint and core_valid and random_valid:
            return existing
    scenarios = build_blueprints(split_counts, seed=int(settings["split_seed"]))
    random_train = build_blueprints(
        {"train": int(settings["train_semantic_groups"])},
        seed=int(settings["split_seed"]),
        active_search=False,
    )
    split_hashes: dict[str, str] = {}
    for split in split_counts:
        path = blueprint_dir / f"{split}.jsonl"
        write_jsonl(path, [scenario.to_dict() for scenario in scenarios if scenario.split == split])
        split_hashes[split] = sha256_file(path)
    random_train_path = blueprint_dir / "random" / "train.jsonl"
    write_jsonl(random_train_path, [scenario.to_dict() for scenario in random_train])
    random_train_sha256 = sha256_file(random_train_path)
    central = central_validity_checks()
    if not central["passed"]:
        raise RuntimeError("A central DGP sentinel invalidated its standard procedure")
    manifest = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "split_counts": split_counts,
        "split_sha256": split_hashes,
        "random_ablation_train_sha256": random_train_sha256,
        "random_ablation_train_count": len(random_train),
        "blueprint_count": len(scenarios),
        "split_before_translation": True,
        "central_validity": central,
        "catalog_sha256": canonical_hash(catalog_manifest()),
    }
    write_json(blueprint_dir / "catalog.json", catalog_manifest())
    write_json(manifest_path, manifest)
    return manifest


def _cached_simulation(
    config: ProjectConfig,
    scenario: Scenario,
    *,
    force: bool,
) -> dict[str, Any]:
    settings = config.section("stats_data")
    cache_path = (
        config.path_for("simulation_dir") / scenario.split / f"{scenario.blueprint_id}.json"
    )
    expected = canonical_hash(
        {
            "scenario": scenario.to_dict(),
            "initial_repetitions": settings["initial_repetitions"],
            "escalation_repetitions": settings["escalation_repetitions"],
            "uncertainty_margin": settings["ranking_uncertainty_margin"],
            "temperature": settings["regret_temperature"],
            "simulator_version": 1,
        }
    )
    if cache_path.exists() and not force:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_fingerprint") == expected:
            return cached
    result = simulate_scenario(
        scenario,
        initial_repetitions=int(settings["initial_repetitions"]),
        escalation_repetitions=[int(value) for value in settings["escalation_repetitions"]],
        uncertainty_margin=float(settings["ranking_uncertainty_margin"]),
        temperature=float(settings["regret_temperature"]),
    )
    result["cache_fingerprint"] = expected
    write_json(cache_path, result)
    return result


def simulate_stats_surface(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    blueprint_manifest = prepare_stats_blueprints(config, force=False)
    stats_dir = config.path_for("stats_dir")
    blueprint_dir = stats_dir / "blueprints"
    surface_dir = stats_dir / "surface"
    manifest_path = surface_dir / "manifest.json"
    settings = config.section("stats_data")
    fingerprint = canonical_hash(
        {
            "blueprints": blueprint_manifest["fingerprint"],
            "repetitions": [settings["initial_repetitions"], *settings["escalation_repetitions"]],
            "temperature": settings["regret_temperature"],
            "surface_version": 3,
        }
    )
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint:
            valid = all(
                (surface_dir / f"{split}.jsonl").exists()
                and existing.get("split_sha256", {}).get(split)
                == sha256_file(surface_dir / f"{split}.jsonl")
                for split in ("train", "valid", "dev", "final")
            )
            random_path = surface_dir / "random" / "train.jsonl"
            valid = (
                valid
                and random_path.exists()
                and existing.get("random_ablation_train_sha256")
                == sha256_file(random_path)
            )
            if valid:
                return existing
    split_hashes: dict[str, str] = {}
    repetitions: Counter[int] = Counter()
    selection_counts: Counter[str] = Counter()
    for split in ("train", "valid", "dev", "final"):
        results: list[dict[str, Any]] = []
        for row in read_jsonl(blueprint_dir / f"{split}.jsonl"):
            result = _cached_simulation(config, _scenario(row), force=force)
            results.append(result)
            repetitions[int(result["repetitions"])] += 1
            selection_counts[str(result["selected_method_id"])] += 1
        output_path = surface_dir / f"{split}.jsonl"
        write_jsonl(output_path, results)
        split_hashes[split] = sha256_file(output_path)
    random_results: list[dict[str, Any]] = []
    for row in read_jsonl(blueprint_dir / "random" / "train.jsonl"):
        result = _cached_simulation(config, _scenario(row), force=force)
        random_results.append(result)
        repetitions[int(result["repetitions"])] += 1
        selection_counts[str(result["selected_method_id"])] += 1
    random_output_path = surface_dir / "random" / "train.jsonl"
    write_jsonl(random_output_path, random_results)
    random_train_sha256 = sha256_file(random_output_path)
    manifest = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "split_sha256": split_hashes,
        "random_ablation_train_sha256": random_train_sha256,
        "random_ablation_train_count": len(random_results),
        "blueprint_fingerprint": blueprint_manifest["fingerprint"],
        "simulation_count": sum(repetitions.values()),
        "repetitions": {str(key): value for key, value in sorted(repetitions.items())},
        "selected_methods": dict(sorted(selection_counts.items())),
        "common_random_numbers": True,
        "adaptive_repetitions": True,
    }
    write_json(manifest_path, manifest)
    return manifest


def _format_parameters(parameters: dict[str, float]) -> str:
    values: list[str] = []
    for key, value in parameters.items():
        if float(value).is_integer():
            rendered = str(int(value))
        else:
            rendered = f"{value:.3f}".rstrip("0").rstrip(".")
        values.append(f"{key}={rendered}")
    return ", ".join(values)


def _study_context(scenario: Scenario) -> dict[str, Any]:
    paired = scenario.family_id == "group_comparison" and scenario.parameters.get(
        "pair_correlation", 0.0
    ) > 0.62
    binary = scenario.family_id == "binary_count_glm" and (
        scenario.parameters.get("event_rate", 1.0) < 0.34
        or scenario.parameters.get("separation", 0.0) > 0.45
    )
    contexts: dict[str, dict[str, Any]] = {
        "group_comparison": {
            "estimand": "mean within-pair change" if paired else "difference in group means",
            "sampling_unit": "participant",
            "study_design": "paired measurements" if paired else "independent two-group study",
            "outcome_type": "continuous",
            "dependence": "within-participant pairs" if paired else "independent participants",
            "missingness": "complete for the declared variables",
            "schema": (
                ["participant_id:id", "before:continuous", "after:continuous"]
                if paired
                else ["participant_id:id", "score:continuous outcome", "arm:two-level group"]
            ),
        },
        "categorical": {
            "estimand": "risk difference between groups",
            "sampling_unit": "participant",
            "study_design": "independent two-group observational study",
            "outcome_type": "binary",
            "dependence": "independent participants",
            "missingness": "complete for outcome and group",
            "schema": ["participant_id:id", "event:binary outcome", "arm:two-level group"],
        },
        "linear_robust": {
            "estimand": "adjusted slope for x1",
            "sampling_unit": "participant",
            "study_design": "cross-sectional regression",
            "outcome_type": "continuous",
            "dependence": "independent participants",
            "missingness": "complete for modeled variables",
            "schema": ["y:continuous outcome", "x1:target predictor", "x2:adjustment predictor"],
        },
        "binary_count_glm": {
            "estimand": "conditional log-odds ratio" if binary else "conditional log-rate ratio",
            "sampling_unit": "participant",
            "study_design": "independent generalized linear model",
            "outcome_type": "binary" if binary else "count",
            "dependence": "independent participants",
            "missingness": "complete for modeled variables",
            "schema": [
                f"y:{'binary' if binary else 'count'} outcome",
                "x1:target predictor",
                "x2:adjustment predictor",
            ],
        },
        "clustered_repeated": {
            "estimand": "population-average adjusted slope",
            "sampling_unit": "cluster",
            "study_design": "clustered repeated-measure study",
            "outcome_type": "continuous",
            "dependence": "observations nested within cluster_id",
            "missingness": "complete for modeled variables",
            "schema": ["y:continuous outcome", "x:target predictor", "cluster_id:cluster"],
        },
        "survival": {
            "estimand": "hazard ratio for exposure",
            "sampling_unit": "participant",
            "study_design": "right-censored cohort",
            "outcome_type": "time to event",
            "dependence": "independent participants",
            "missingness": "right censoring is declared",
            "schema": ["time:follow-up time", "event:event indicator", "exposure:target predictor"],
        },
        "missing_selection": {
            "estimand": "population adjusted slope",
            "sampling_unit": "participant",
            "study_design": "observational study with declared missingness",
            "outcome_type": "continuous",
            "dependence": "independent participants",
            "missingness": "outcome observation depends on recorded x",
            "schema": ["y:partly observed outcome", "x:predictor", "observed:response indicator"],
        },
        "experimental_causal": {
            "estimand": "average treatment effect",
            "sampling_unit": "participant",
            "study_design": (
                "randomized parallel experiment"
                if scenario.parameters.get("confounding", 1.0) < 0.32
                else "observational treatment comparison"
            ),
            "outcome_type": "continuous",
            "dependence": "independent participants",
            "missingness": "complete for modeled variables",
            "schema": [
                "y:continuous outcome",
                "treatment:binary exposure",
                "x_baseline:pre-treatment covariate",
            ],
        },
        "probability_distribution": {
            "estimand": "population location and predictive adequacy",
            "sampling_unit": "observation",
            "study_design": "independent probability sample",
            "outcome_type": "continuous",
            "dependence": "independent observations",
            "missingness": "complete",
            "schema": ["y:continuous outcome"],
        },
        "bayesian_check": {
            "estimand": "posterior location with model adequacy check",
            "sampling_unit": "observation",
            "study_design": "declared Bayesian generative analysis",
            "outcome_type": "continuous",
            "dependence": "independent conditional on model parameters",
            "missingness": "complete",
            "schema": ["y:continuous outcome"],
        },
        "predictive_calibration": {
            "estimand": "out-of-sample event probability",
            "sampling_unit": "participant",
            "study_design": "prediction development with held-out validation",
            "outcome_type": "binary",
            "dependence": "independent participants",
            "missingness": "complete for modeled variables",
            "schema": ["y:binary outcome", "x1:predictor", "x2:predictor"],
        },
        "time_series_leakage": {
            "estimand": "future forecast error at the declared horizon",
            "sampling_unit": "time point",
            "study_design": "ordered forecasting study",
            "outcome_type": "continuous time series",
            "dependence": "serial order by time",
            "missingness": "complete ordered series",
            "schema": ["time:ordering variable", "y:forecast outcome", "x1:lagged predictor"],
        },
    }
    return contexts[scenario.family_id]


_FAMILY_NAMES = {
    "en": {family_id: FAMILY_BY_ID[family_id].name for family_id in FAMILY_BY_ID},
    "zh_Hant": {
        "group_comparison": "組間或配對比較",
        "categorical": "類別與比例資料",
        "linear_robust": "線性與穩健迴歸",
        "binary_count_glm": "二元與計數廣義線性模型",
        "clustered_repeated": "群聚與重複量測",
        "survival": "生存分析",
        "missing_selection": "缺失與選擇偏差",
        "experimental_causal": "實驗與因果推論",
        "probability_distribution": "機率分布",
        "bayesian_check": "貝氏估計與模型檢查",
        "predictive_calibration": "預測與校準",
        "time_series_leakage": "時間序列與資料洩漏",
    },
    "zh_Hans": {
        "group_comparison": "组间或配对比较",
        "categorical": "类别与比例数据",
        "linear_robust": "线性与稳健回归",
        "binary_count_glm": "二元与计数广义线性模型",
        "clustered_repeated": "聚类与重复测量",
        "survival": "生存分析",
        "missing_selection": "缺失与选择偏差",
        "experimental_causal": "实验与因果推断",
        "probability_distribution": "概率分布",
        "bayesian_check": "贝叶斯估计与模型检查",
        "predictive_calibration": "预测与校准",
        "time_series_leakage": "时间序列与数据泄漏",
    },
}


def _render_question(
    scenario: Scenario,
    language: str,
    *,
    incomplete: bool,
    view: str = "standard",
) -> str:
    family = FAMILY_BY_ID[scenario.family_id]
    family_name = _FAMILY_NAMES[language][scenario.family_id]
    parameters = _format_parameters(scenario.parameters)
    missing = ", ".join(family.clarification_fields[:2])
    context = _study_context(scenario)
    schema = ", ".join(context["schema"])
    boundary_prompt = {
        "boundary_a": "Choose the primary analysis and justify its validity at this boundary.",
        "boundary_b": (
            "A collaborator wants the cheapest standard method. Decide whether it remains valid "
            "or a robust alternative is warranted."
        ),
        "standard": "Choose the primary analysis.",
    }[view]
    if language == "en":
        if incomplete:
            return (
                f"I have a {family_name.lower()} problem and want a defensible analysis, but the "
                f"{missing} are not stated. Choose the next statistical action. "
                "Do not invent design facts."
            )
        return (
            f"Study: {context['study_design']}. Estimand: {context['estimand']}. "
            f"Sampling unit: {context['sampling_unit']}; dependence: {context['dependence']}; "
            f"missingness: {context['missingness']}. Data schema: {schema}. "
            f"DGP audit values: {parameters}. {boundary_prompt} "
            "Prioritize Type I error and interval coverage before power."
        )
    if language == "zh_Hant":
        if incomplete:
            return (
                f"我有一個{family_name}問題，但沒有交代 {missing}。"
                "請選擇下一個統計動作，不要自行假設研究設計。"
            )
        return (
            f"研究設計：{context['study_design']}。估計目標：{context['estimand']}。"
            f"抽樣單位：{context['sampling_unit']}；相依結構：{context['dependence']}；"
            f"缺失情形：{context['missingness']}。資料欄位：{schema}。"
            f"DGP 稽核值：{parameters}。請選擇並規劃主要分析；"
            "型一錯誤與區間涵蓋率優先於 power。"
        )
    if incomplete:
        return (
            f"我有一个{family_name}问题，但没有说明 {missing}。"
            "请选择下一项统计动作，不要自行假设研究设计。"
        )
    return (
        f"研究设计：{context['study_design']}。估计目标：{context['estimand']}。"
        f"抽样单位：{context['sampling_unit']}；依赖结构：{context['dependence']}；"
        f"缺失情况：{context['missingness']}。数据字段：{schema}。"
        f"DGP 审核值：{parameters}。请选择并规划主要分析；"
        "第一类错误与区间覆盖率优先于 power。"
    )


def _clarification_question(scenario: Scenario, language: str) -> str:
    fields = FAMILY_BY_ID[scenario.family_id].clarification_fields
    selected = " and ".join(fields[:2])
    if language == "en":
        return (
            f"Please specify {selected}; without them I cannot identify a valid "
            "estimand and uncertainty procedure."
        )
    if language == "zh_Hant":
        return f"請先說明 {selected}；缺少這些資訊時，無法辨識有效的估計目標與不確定性程序。"
    return f"请先说明 {selected}；缺少这些信息时，无法确定有效的估计目标与不确定性程序。"


def _analysis_plan(scenario: Scenario, method_id: str, *, incomplete: bool) -> dict[str, Any]:
    family = FAMILY_BY_ID[scenario.family_id]
    if incomplete:
        return {
            "status": "needs_clarification",
            "estimand": None,
            "sampling_unit": None,
            "study_design": None,
            "outcome_type": None,
            "dependence": None,
            "missingness": None,
            "method_id": "needs_clarification",
            "uncertainty": None,
            "diagnostics": ["do not run a procedure until the declared design fields are supplied"],
            "tool": "none",
            "questions": list(family.clarification_fields[:2]),
            "variables": {},
            "data_file_index": 0,
        }
    method = PROCEDURE_BY_ID[method_id]
    context = _study_context(scenario)
    paired = scenario.family_id == "group_comparison" and scenario.parameters.get(
        "pair_correlation", 0.0
    ) > 0.62
    variables_by_method: dict[str, dict[str, Any]] = {
        "independent_t": {"outcome": "score", "group": "arm"},
        "welch_t": {"outcome": "score", "group": "arm"},
        "mann_whitney": {"outcome": "score", "group": "arm"},
        "paired_t": {"before": "before", "after": "after"},
        "wilcoxon_signed_rank": {"before": "before", "after": "after"},
        "chi_square": {"outcome": "event", "group": "arm"},
        "fisher_exact": {"outcome": "event", "group": "arm"},
        "two_proportion": {"outcome": "event", "group": "arm"},
        "ols": {"outcome": "y", "predictors": ["x1", "x2"]},
        "hc3_ols": {"outcome": "y", "predictors": ["x1", "x2"]},
        "huber_regression": {"outcome": "y", "predictors": ["x1", "x2"]},
        "logistic_glm": {"outcome": "y", "predictors": ["x1", "x2"]},
        "firth_logistic": {"outcome": "y", "predictors": ["x1", "x2"]},
        "poisson_glm": {"outcome": "y", "predictors": ["x1", "x2"]},
        "negative_binomial_glm": {"outcome": "y", "predictors": ["x1", "x2"]},
        "gee": {"outcome": "y", "predictors": ["x"], "cluster": "cluster_id"},
        "mixed_effects": {
            "outcome": "y",
            "predictors": ["x"],
            "cluster": "cluster_id",
        },
        "cox_ph": {"time": "time", "event": "event", "predictors": ["exposure"]},
        "logrank": {"time": "time", "event": "event", "group": "exposure"},
        "multiple_imputation": {"outcome": "y", "predictors": ["x"]},
        "ipw": {
            "outcome": "y",
            "treatment": "treatment",
            "predictors": ["x_baseline"],
        },
        "difference_in_means": {"outcome": "y", "treatment": "treatment"},
        "ancova": {"outcome": "y", "predictors": ["treatment", "x_baseline"]},
        "randomization_inference": {"outcome": "y", "treatment": "treatment"},
        "conjugate_bayes": {"outcome": "y"},
        "posterior_predictive": {"outcome": "y"},
        "calibrated_logistic": {"outcome": "y", "predictors": ["x1", "x2"]},
        "blocked_time_series_cv": {
            "outcome": "y",
            "predictors": ["x1"],
            "time": "time",
        },
    }
    variables = variables_by_method[method_id]
    if scenario.family_id == "group_comparison" and paired and method_id not in {
        "paired_t",
        "wilcoxon_signed_rank",
    }:
        variables = {"outcome": "after", "group": "participant_id"}
    if scenario.family_id == "missing_selection" and method_id == "ols":
        variables = {"outcome": "y", "predictors": ["x"]}
    if scenario.family_id == "missing_selection" and method_id == "ipw":
        variables = {"outcome": "y", "treatment": "observed", "predictors": ["x"]}
    if scenario.family_id == "experimental_causal" and method_id in {"ols", "hc3_ols"}:
        variables = {"outcome": "y", "predictors": ["treatment", "x_baseline"]}
    return {
        "status": "ready",
        "estimand": context["estimand"],
        "sampling_unit": context["sampling_unit"],
        "study_design": context["study_design"],
        "outcome_type": context["outcome_type"],
        "dependence": context["dependence"],
        "missingness": context["missingness"],
        "method_id": method_id,
        "uncertainty": method.uncertainty,
        "diagnostics": list(method.assumptions),
        "tool": method.tool,
        "variables": variables,
        "questions": [],
        "data_file_index": 0,
    }


def _method_explanation(simulation: dict[str, Any], language: str) -> str:
    selected = str(simulation["selected_method_id"])
    chosen = next(item for item in simulation["candidates"] if item["method_id"] == selected)
    if language == "en":
        return (
            f"{selected} has the lowest validity-first simulated regret. "
            f"Its estimated Type I error is {chosen['type1_error']:.3f}, "
            f"coverage is {chosen['coverage']:.3f}, and normalized regret is "
            f"{chosen['normalized_regret']:.3f}. {chosen['reason']}"
        )
    if language == "zh_Hant":
        return (
            f"{selected} 的有效性優先模擬 regret 最低。估計型一錯誤為 {chosen['type1_error']:.3f}，"
            f"涵蓋率為 {chosen['coverage']:.3f}，正規化 regret 為 "
            f"{chosen['normalized_regret']:.3f}。{chosen['reason']}"
        )
    return (
        f"{selected} 的有效性优先模拟 regret 最低。估计第一类错误为 {chosen['type1_error']:.3f}，"
        f"覆盖率为 {chosen['coverage']:.3f}，归一化 regret 为 "
        f"{chosen['normalized_regret']:.3f}。{chosen['reason']}"
    )


def _menu(
    simulation: dict[str, Any], *, seed: int
) -> tuple[list[str], list[str], dict[str, float]]:
    candidates = [str(item["method_id"]) for item in simulation["candidates"]]
    candidates.append("needs_clarification")
    rng = __import__("random").Random(seed)
    rng.shuffle(candidates)
    if len(candidates) > len(_LABELS):
        raise RuntimeError("Statistical menu exceeds the six-label contract")
    labels = list(_LABELS[: len(candidates)])
    mapping = dict(zip(candidates, labels, strict=True))
    return candidates, labels, mapping


def _menu_text(candidates: list[str], labels: list[str]) -> str:
    lines = []
    for method_id, label in zip(candidates, labels, strict=True):
        name = (
            "Ask for the missing design information"
            if method_id == "needs_clarification"
            else PROCEDURE_BY_ID[method_id].name
        )
        lines.append(f"{label}. {method_id} — {name}")
    return "\n".join(lines)


def _build_record(
    scenario: Scenario,
    simulation: dict[str, Any],
    *,
    language: str,
    loss_weight: float,
    incomplete: bool,
    variant: str,
    refined_explanation: str | None,
    view: str = "standard",
) -> dict[str, Any]:
    candidates, labels, mapping = _menu(
        simulation,
        seed=scenario.seed + sum(ord(char) for char in f"{language}:{incomplete}:{view}"),
    )
    selected = "needs_clarification" if incomplete else str(simulation["selected_method_id"])
    plan = _analysis_plan(scenario, selected, incomplete=incomplete)
    soft_by_method = {
        str(item["method_id"]): float(item["soft_target"]) for item in simulation["candidates"]
    }
    soft_by_method["needs_clarification"] = 0.0
    if incomplete:
        soft_by_method = {
            method_id: float(method_id == "needs_clarification") for method_id in candidates
        }
    if variant == "hard-label":
        soft_by_method = {method_id: float(method_id == selected) for method_id in candidates}
    method_probabilities = [soft_by_method.get(method_id, 0.0) for method_id in candidates]
    probability_sum = sum(method_probabilities)
    method_probabilities = [value / probability_sum for value in method_probabilities]
    question = _render_question(scenario, language, incomplete=incomplete, view=view)
    user = f"{question}\n\nCandidate menu:\n{_menu_text(candidates, labels)}"
    selection = mapping[selected]
    tool_call_json = json.dumps(
        {"tool": plan["tool"], "method_id": selected},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    first_assistant = (
        f"<method>{selection}</method>\n"
        "<analysis_plan>"
        f"{json.dumps(plan, ensure_ascii=False, separators=(',', ':'))}"
        "</analysis_plan>\n"
        "<tool_call>"
        f"{tool_call_json}"
        "</tool_call>"
    )
    if incomplete:
        tool_result = {"status": "not_run", "reason": "needs_clarification"}
        final = _clarification_question(scenario, language)
    else:
        chosen = next(item for item in simulation["candidates"] if item["method_id"] == selected)
        tool_result = {
            "status": "simulated",
            "method_id": selected,
            "type1_error": round(float(chosen["type1_error"]), 6),
            "coverage": round(float(chosen["coverage"]), 6),
            "bias": round(float(chosen["bias"]), 6),
            "rmse": round(float(chosen["rmse"]), 6),
            "power": round(float(chosen["power"]), 6),
            "calibration_error": round(float(chosen["calibration_error"]), 6),
        }
        final = refined_explanation or _method_explanation(simulation, language)
    messages = [
        {
            "role": "system",
            "content": (
                "You are Charlie alpha, a trilingual statistical analysis assistant. "
                "Report assumptions and never invent missing design facts."
            ),
        },
        {"role": "user", "content": user},
        {"role": "assistant", "content": first_assistant},
        {
            "role": "user",
            "content": (
                "<tool_result>"
                f"{json.dumps(tool_result, ensure_ascii=False, separators=(',', ':'))}"
                "</tool_result>"
            ),
        },
        {"role": "assistant", "content": f"<final_report>{final}</final_report>"},
    ]
    selected_candidate_index = candidates.index(selected)
    return {
        "messages": messages,
        "metadata": {
            "schema_version": 1,
            "semantic_group_id": scenario.blueprint_id,
            "blueprint_id": scenario.blueprint_id,
            "split": scenario.split,
            "family_id": scenario.family_id,
            "domain": scenario.domain,
            "language": language,
            "loss_weight": loss_weight,
            "incomplete": incomplete,
            "view": view,
            "variant": variant,
            "curriculum": "active-boundary" if variant == "dgp-regret" else "random",
            "boundary_round": scenario.boundary_round,
            "candidate_method_ids": candidates,
            "candidate_labels": labels,
            "method_probabilities": method_probabilities,
            "selected_candidate_index": selected_candidate_index,
            "selected_method_id": selected,
            "selected_label": selection,
            "simulator_fingerprint": simulation["fingerprint"],
            "repetitions": simulation["repetitions"],
            "assistant_components": {
                "method": ["<method>", "</method>"],
                "plan_tool": ["<analysis_plan>", "</tool_call>"],
                "report": ["<final_report>", "</final_report>"],
            },
            "tool_output_masked": True,
            "user_content_masked": True,
        },
    }


def _load_refinements(config: ProjectConfig) -> dict[tuple[str, str], str]:
    path = config.path_for("teacher_dir") / "refinements.jsonl"
    if not path.exists():
        return {}
    return {
        (str(row["blueprint_id"]), str(row["language"])): str(row["explanation"])
        for row in read_jsonl(path)
        if row.get("validated")
    }


def _validate_record(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    messages = row.get("messages")
    metadata = row.get("metadata")
    if not isinstance(messages, list) or [message.get("role") for message in messages] != [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]:
        errors.append("invalid message role sequence")
    if not isinstance(metadata, dict):
        return [*errors, "metadata is missing"]
    probabilities = metadata.get("method_probabilities", [])
    if not probabilities or not math.isclose(sum(map(float, probabilities)), 1.0, abs_tol=1e-7):
        errors.append("method probabilities do not sum to one")
    candidates = metadata.get("candidate_method_ids", [])
    if len(candidates) > 6 or len(candidates) != len(probabilities):
        errors.append("candidate menu contract failed")
    if metadata.get("selected_method_id") not in candidates:
        errors.append("selected method is absent from the candidate menu")
    if metadata.get("incomplete") and metadata.get("selected_method_id") != "needs_clarification":
        errors.append("underspecified case did not request clarification")
    if not metadata.get("tool_output_masked") or not metadata.get("user_content_masked"):
        errors.append("mask declarations are missing")
    return errors


def build_stats_data(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    surface_manifest = simulate_stats_surface(config, force=False)
    settings = config.section("stats_data")
    surface_dir = config.path_for("stats_dir") / "surface"
    final_dir = config.path_for("final_dir")
    manifest_path = final_dir / "manifest.json"
    refinements = _load_refinements(config)
    base = config.sources["models"]["research_base_mlx_4bit"]
    max_seq_length = int(config.section("stats_training")["max_seq_length"])
    fingerprint = canonical_hash(
        {
            "surface": surface_manifest["fingerprint"],
            "weights": settings["examples_per_train_group"],
            "refinements": canonical_hash(
                [
                    {"blueprint_id": key[0], "language": key[1], "text": value}
                    for key, value in sorted(refinements.items())
                ]
            ),
            "base_tokenizer": base,
            "max_seq_length": max_seq_length,
            "renderer_version": 4,
        }
    )
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint and all(
            (final_dir / path).exists() for path in existing.get("files", {})
        ):
            return existing
    output_hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    validation_errors: list[str] = []
    ratio_audit: dict[str, Any] = {}
    from transformers import AutoTokenizer

    tokenizer_path = snapshot_download(repo_id=base["repo_id"], revision=base["revision"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    maximum_observed_tokens = 0
    refinement_length_fallbacks = 0
    for variant in ("hard-label", "regret-random", "dgp-regret"):
        for split in ("train", "valid"):
            records: list[dict[str, Any]] = []
            surface_path = (
                surface_dir / "random" / "train.jsonl"
                if variant == "regret-random" and split == "train"
                else surface_dir / f"{split}.jsonl"
            )
            for simulation in read_jsonl(surface_path):
                scenario = _scenario(simulation["scenario"])
                recipe = (
                    [
                        ("en", 1.4, False, "boundary_a"),
                        ("en", 1.4, False, "boundary_b"),
                        ("zh_Hant", 0.6, False, "standard"),
                        ("zh_Hans", 0.6, False, "standard"),
                    ]
                    if split == "train"
                    else [
                        ("en", 1.0, False, "standard"),
                        ("zh_Hant", 1.0, False, "standard"),
                        ("zh_Hans", 1.0, False, "standard"),
                        ("en", 1.0, True, "standard"),
                    ]
                )
                for language, weight, incomplete, view in recipe:
                    refined = (
                        refinements.get((scenario.blueprint_id, language))
                        if not incomplete
                        else None
                    )
                    record = _build_record(
                        scenario,
                        simulation,
                        language=language,
                        loss_weight=weight,
                        incomplete=incomplete,
                        variant=variant,
                        refined_explanation=refined,
                        view=view,
                    )
                    token_count = _chat_token_count(tokenizer, record["messages"])
                    if token_count > max_seq_length and refined is not None:
                        record = _build_record(
                            scenario,
                            simulation,
                            language=language,
                            loss_weight=weight,
                            incomplete=incomplete,
                            variant=variant,
                            refined_explanation=None,
                            view=view,
                        )
                        token_count = _chat_token_count(tokenizer, record["messages"])
                        record["metadata"]["teacher_refinement_length_fallback"] = True
                        refinement_length_fallbacks += 1
                    maximum_observed_tokens = max(maximum_observed_tokens, token_count)
                    record["metadata"]["token_count_qwen35"] = token_count
                    if token_count > max_seq_length:
                        validation_errors.append(
                            f"{variant}/{split}/{scenario.blueprint_id}: "
                            f"{token_count} tokens exceeds {max_seq_length}"
                        )
                    errors = _validate_record(record)
                    validation_errors.extend(
                        f"{variant}/{split}/{scenario.blueprint_id}: {error}" for error in errors
                    )
                    records.append(record)
            if variant == "dgp-regret" and split == "train":
                records.sort(
                    key=lambda row: (
                        int(row["metadata"]["boundary_round"]),
                        canonical_hash({"seed": 42, "id": row["metadata"]["blueprint_id"]}),
                    )
                )
            path = final_dir / variant / f"{split}.jsonl"
            write_jsonl(path, records)
            relative = str(path.relative_to(final_dir))
            output_hashes[relative] = sha256_file(path)
            counts[relative] = len(records)
            if split == "train":
                language_weights: dict[str, float] = defaultdict(float)
                for record in records:
                    language_weights[str(record["metadata"]["language"])] += float(
                        record["metadata"]["loss_weight"]
                    )
                total = sum(language_weights.values())
                ratio_audit[variant] = {
                    language: language_weights[language] / total for language in _LANGUAGES
                }
    train_ids = {row["scenario"]["blueprint_id"] for row in read_jsonl(surface_dir / "train.jsonl")}
    random_train_ids = {
        row["scenario"]["blueprint_id"]
        for row in read_jsonl(surface_dir / "random" / "train.jsonl")
    }
    split_ids = {
        split: {
            row["scenario"]["blueprint_id"] for row in read_jsonl(surface_dir / f"{split}.jsonl")
        }
        for split in ("valid", "dev", "final")
    }
    leakage = {
        split: sorted((train_ids | random_train_ids) & ids)
        for split, ids in split_ids.items()
        if (train_ids | random_train_ids) & ids
    }
    expected_ratios = {key: float(value) for key, value in settings["languages"].items()}
    for variant, ratios in ratio_audit.items():
        for language, expected in expected_ratios.items():
            if not math.isclose(ratios[language], expected, abs_tol=1e-9):
                validation_errors.append(
                    f"{variant}: {language} gradient ratio is {ratios[language]:.6f}"
                )
    if leakage:
        validation_errors.append(f"blueprint split leakage: {leakage}")
    if validation_errors:
        raise RuntimeError("Stats data validation failed: " + "; ".join(validation_errors[:20]))
    manifest = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "surface_fingerprint": surface_manifest["fingerprint"],
        "files": output_hashes,
        "counts": counts,
        "language_gradient_ratios": ratio_audit,
        "split_leakage": False,
        "variant_surfaces": {
            "hard-label": "active-failure",
            "regret-random": "random-latin-hypercube",
            "dgp-regret": "active-failure",
        },
        "split_before_translation": True,
        "teacher_refinement_count": len(refinements),
        "teacher_refinement_length_fallbacks": refinement_length_fallbacks,
        "correctness_authority": "DGP simulator and deterministic templates",
        "teacher_correctness_authority": False,
        "maximum_token_count_qwen35": maximum_observed_tokens,
        "max_sequence_length": max_seq_length,
    }
    write_json(manifest_path, manifest)
    return manifest


def _extract_refinement(text: str) -> str | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    explanation = value.get("explanation") if isinstance(value, dict) else None
    return str(explanation).strip() if explanation else None


def _refinement_valid(template: str, candidate: str, method_id: str) -> bool:
    meta_phrases = (
        "edited for",
        "the text",
        "original text",
        "wording",
        "preserv",
        "adjusted",
        "revised",
        "调整",
        "修改",
        "原文",
        "保留",
        "要求",
        "编辑",
        "措辞",
        "調整",
        "修改",
        "原文",
        "保留",
        "要求",
        "編輯",
        "措辭",
    )
    lowered = candidate.lower()
    if (
        method_id not in candidate
        or len(candidate) > len(template) + 80
        or len(candidate) < 40
        or any(phrase in lowered for phrase in meta_phrases)
    ):
        return False
    required_numbers = set(_NUMBER_RE.findall(template))
    return required_numbers.issubset(set(_NUMBER_RE.findall(candidate)))


def distill_stats_explanations(config: ProjectConfig, force: bool = False) -> dict[str, Any]:
    """Use the 9B model only as a wording editor; simulator facts stay immutable."""
    simulate_stats_surface(config, force=False)
    teacher_dir = config.path_for("teacher_dir")
    output_path = teacher_dir / "refinements.jsonl"
    status_path = teacher_dir / "manifest.json"
    settings = config.section("stats_data")
    teacher = config.sources["models"]["teacher_mlx_4bit"]
    candidates: list[tuple[dict[str, Any], str]] = []
    train_surface = list(read_jsonl(config.path_for("stats_dir") / "surface" / "train.jsonl"))
    train_surface.sort(
        key=lambda row: (
            -max(float(item["raw_regret"]) for item in row["candidates"]),
            row["scenario"]["blueprint_id"],
        )
    )
    limit = int(settings["teacher_refinement_limit"])
    for index, simulation in enumerate(train_surface[:limit]):
        language = ("zh_Hant", "zh_Hans", "en")[index % 3]
        candidates.append((simulation, language))
    fingerprint = canonical_hash(
        {
            "teacher": teacher,
            "candidate_ids": [
                (row["scenario"]["blueprint_id"], language) for row, language in candidates
            ],
            "temperature": 0.0,
            "enable_thinking": False,
            "max_tokens": 160,
            "distill_version": 4,
        }
    )
    if status_path.exists() and output_path.exists() and not force:
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint and existing.get(
            "output_sha256"
        ) == sha256_file(output_path):
            return existing
    partial_path = teacher_dir / "refinements.partial.jsonl"
    partial_status_path = teacher_dir / "partial-manifest.json"
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    resumed_matching_partial = False
    if not force and partial_path.exists() and partial_status_path.exists():
        partial_status = json.loads(partial_status_path.read_text(encoding="utf-8"))
        if partial_status.get("fingerprint") == fingerprint:
            completed = {
                (str(row["blueprint_id"]), str(row["language"])): row
                for row in read_jsonl(partial_path)
            }
            resumed_matching_partial = True
    if not force and not completed and output_path.exists():
        prior = {
            (str(row["blueprint_id"]), str(row["language"])): row
            for row in read_jsonl(output_path)
            if row.get("validated")
        }
        for simulation, language in candidates:
            key = (str(simulation["scenario"]["blueprint_id"]), language)
            row = prior.get(key)
            if row is None or row.get("simulator_fingerprint") != simulation["fingerprint"]:
                continue
            template = _method_explanation(simulation, language)
            method_id = str(simulation["selected_method_id"])
            if _refinement_valid(template, str(row.get("explanation", "")), method_id):
                completed[key] = row
    if not resumed_matching_partial:
        write_jsonl(partial_path, list(completed.values()))
        write_json(
            partial_status_path,
            {"fingerprint": fingerprint, "completed": len(completed), "target": len(candidates)},
        )
    if not completed:
        write_jsonl(partial_path, [])
        write_json(
            partial_status_path,
            {"fingerprint": fingerprint, "completed": 0, "target": len(candidates)},
        )
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    pending = [
        (simulation, language)
        for simulation, language in candidates
        if (str(simulation["scenario"]["blueprint_id"]), language) not in completed
    ]
    model = None
    tokenizer = None
    if pending:
        model_path = snapshot_download(repo_id=teacher["repo_id"], revision=teacher["revision"])
        model, tokenizer = load(model_path, tokenizer_config={"trust_remote_code": True})
    rows: list[dict[str, Any]] = list(completed.values())
    accepted = sum(bool(row.get("validated")) for row in rows)
    fallback = len(rows) - accepted
    started = time.monotonic()
    for simulation, language in pending:
        scenario = _scenario(simulation["scenario"])
        template = _method_explanation(simulation, language)
        method_id = str(simulation["selected_method_id"])
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You edit wording only. Return one JSON object with key explanation. "
                    "Preserve every method identifier and number exactly. "
                    "The value must be a direct statistical explanation beginning with the "
                    "method identifier. Never discuss editing, wording, instructions, or "
                    "preservation. Do not add statistical claims."
                ),
            },
            {
                "role": "user",
                "content": f"Language: {language}\nImmutable text: {template}",
            },
        ]
        assert model is not None and tokenizer is not None
        refined: str | None = None
        for attempt in range(2):
            messages = list(prompt_messages)
            if attempt:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The prior response was invalid. Return only a direct explanation "
                            f"starting with {method_id}; do not mention editing."
                        ),
                    }
                )
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            generated = generate(
                model,
                tokenizer,
                prompt,
                max_tokens=160,
                sampler=make_sampler(temp=0.0),
                verbose=False,
            )
            candidate = _extract_refinement(generated)
            if candidate and _refinement_valid(template, candidate, method_id):
                refined = candidate
                break
        validated = refined is not None
        if validated:
            accepted += 1
        else:
            fallback += 1
            refined = template
        row = {
            "blueprint_id": scenario.blueprint_id,
            "language": language,
            "explanation": refined,
            "validated": validated,
            "fallback_template": not validated,
            "method_id": method_id,
            "simulator_fingerprint": simulation["fingerprint"],
        }
        rows.append(row)
        append_jsonl(partial_path, row)
        write_json(
            partial_status_path,
            {
                "fingerprint": fingerprint,
                "completed": len(rows),
                "target": len(candidates),
            },
        )
    order = {
        (str(simulation["scenario"]["blueprint_id"]), language): index
        for index, (simulation, language) in enumerate(candidates)
    }
    rows.sort(key=lambda row: order[(str(row["blueprint_id"]), str(row["language"]))])
    write_jsonl(output_path, rows)
    status = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "teacher_repo": teacher["repo_id"],
        "teacher_revision": teacher["revision"],
        "correctness_authority": False,
        "temperature": 0.0,
        "attempts_per_record": 2,
        "accepted": accepted,
        "fallback": fallback,
        "output_sha256": sha256_file(output_path),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(status_path, status)
    del model, tokenizer
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except ImportError:
        pass
    return status

import json
import math
import platform
from collections import Counter
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from charlie_alpha.config import load_config
from charlie_alpha.stats_agent import (
    _sandbox_metadata,
    _validate_plan,
    classify_stats_route,
    resolve_stats_runtime,
)
from charlie_alpha.stats_bakeoff import choose_base_bakeoff
from charlie_alpha.stats_calibrate import (
    block_projection_profiles,
    choose_block_profile,
    choose_delta_scale,
    interpolate_adapter_blocks,
    interpolate_adapter_weights,
    parse_layer_scales,
)
from charlie_alpha.stats_catalog import (
    AGENT_PROCEDURES,
    FAMILIES,
    PROCEDURES,
    validate_catalog,
)
from charlie_alpha.stats_compiler import (
    compile_analysis_scaffold,
    next_repair_plan,
    plan_from_scaffold,
    task_reward,
)
from charlie_alpha.stats_cone import (
    _common_descent_training_settings,
    min_norm_simplex_weights,
    promote_common_descent_candidate,
)
from charlie_alpha.stats_data import (
    _build_record,
    _chat_token_count,
    _refinement_valid,
    _validate_record,
)
from charlie_alpha.stats_dgp import Scenario, build_blueprints, simulate_scenario
from charlie_alpha.stats_eval import _normalize, _pbench_indices
from charlie_alpha.stats_evolve import (
    _adapter_max_abs_delta,
    _choose_ablation_winner,
    _choose_checkpoint,
    _ensure_promotion_shard,
    _evolution_lock,
    _family_learning_signal,
    _group_regret,
    _mutate_scenario,
    _noninferior_mapping,
    _novelty,
    _promotion_scenarios,
    _select_diverse,
    _training_records_are_current,
)
from charlie_alpha.stats_experts import (
    _expert_training_settings,
    select_family_expert_checkpoint,
)
from charlie_alpha.stats_family_router import (
    CharacterNgramNB,
    _route_slug,
    character_ngrams,
    choose_router_threshold,
    choose_temperature,
    router_question,
)
from charlie_alpha.stats_llm_router import ParentLetterRouter, family_route_prompt
from charlie_alpha.stats_project import (
    project_policy,
    reconstruct_regrets,
    summarize_gradient_conflicts,
)
from charlie_alpha.stats_release import _public_isolation_report
from charlie_alpha.stats_robust_experts import (
    cvar_group_weights,
    direct_regret_target,
    select_crossfit_expert_option,
)
from charlie_alpha.stats_route import _aggregate_predictions, select_family_routes
from charlie_alpha.stats_sandbox import SandboxLimits, StatsToolSession, sandbox_self_test
from charlie_alpha.stats_targeted_repair import (
    select_targeted_groups,
    triggered_repair_target,
)
from charlie_alpha.stats_training import (
    StatsDataset,
    _collate_stats_items,
    _group_order,
    _normalize_training_progress,
    _tokenize_stats_record,
)


class _CharacterTokenizer:
    def __init__(self) -> None:
        self._tokenizer = self

    def apply_chat_template(self, messages, **_kwargs):
        return "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages
        )

    def __call__(self, value, **_kwargs):
        return {
            "input_ids": [ord(character) + 100 for character in value],
            "offset_mapping": [(index, index + 1) for index in range(len(value))],
        }


class _SandboxResult:
    def to_dict(self):
        return {
            "returncode": 0,
            "stdout": "private output",
            "stderr": "/Users/private/runtime.py",
            "isolated": True,
        }


def test_agent_sandbox_metadata_omits_raw_streams() -> None:
    assert _sandbox_metadata(_SandboxResult()) == {"returncode": 0, "isolated": True}


def _categorical_scenario() -> Scenario:
    return Scenario(
        blueprint_id="test-categorical",
        family_id="categorical",
        split="train",
        seed=42,
        parameters={
            "n": 40.0,
            "baseline_probability": 0.04,
            "risk_difference": 0.08,
            "imbalance": 0.25,
        },
        boundary_round=2,
        domain="inference_and_design",
    )


def test_stats_catalog_has_declared_core_and_agent_coverage() -> None:
    assert not validate_catalog()
    assert len(PROCEDURES) == 28
    assert len(FAMILIES) == 12
    assert len(AGENT_PROCEDURES) == 35
    assert {item.method_id for item in AGENT_PROCEDURES} >= {
        "binomial_test",
        "spearman_correlation",
        "probit_glm",
        "iv_2sls",
        "kruskal_wallis",
        "regression_f_test",
        "tobit_regression",
    }


def test_blueprints_split_before_translation_with_exact_domain_mix() -> None:
    scenarios = build_blueprints({"train": 240, "valid": 30, "dev": 60, "final": 120}, seed=42)
    assert len(scenarios) == 450
    assert len({item.blueprint_id for item in scenarios}) == 450
    for split, expected in {"train": 240, "valid": 30, "dev": 60, "final": 120}.items():
        rows = [item for item in scenarios if item.split == split]
        assert len(rows) == expected
        domains = {
            name: sum(item.domain == name for item in rows)
            for name in {
                "inference_and_design",
                "probability_and_bayes",
                "prediction_and_analysis",
            }
        }
        assert domains["inference_and_design"] == round(expected * 0.60)
        assert domains["probability_and_bayes"] == round(expected * 0.20)
        assert domains["prediction_and_analysis"] == expected - sum(
            domains[name] for name in ("inference_and_design", "probability_and_bayes")
        )
    training = [item for item in scenarios if item.split == "train"]
    rounds = {
        round_index: sum(item.boundary_round == round_index for item in training)
        for round_index in (0, 1, 2)
    }
    assert rounds == {0: 80, 1: 80, 2: 80}
    searched = [item for item in training if item.boundary_round > 0]
    assert all(item.search and item.search["candidate_count"] >= 3 for item in searched)
    assert {item.search["criterion"] for item in searched if item.search} == {
        "validity-failure-first",
        "ranking-uncertainty-then-central-regret",
    }
    random_ablation = build_blueprints({"train": 240}, seed=42, active_search=False)
    assert all(item.boundary_round == 0 and item.search is None for item in random_ablation)
    assert not (
        {item.blueprint_id for item in training} & {item.blueprint_id for item in random_ablation}
    )


def test_dgp_regret_is_reproducible_normalized_and_soft() -> None:
    scenario = _categorical_scenario()
    first = simulate_scenario(scenario)
    second = simulate_scenario(scenario)
    assert first == second
    regrets = [item["normalized_regret"] for item in first["candidates"]]
    probabilities = [item["soft_target"] for item in first["candidates"]]
    assert min(regrets) == pytest.approx(0.0)
    assert max(regrets) == pytest.approx(1.0)
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[regrets.index(min(regrets))] == max(probabilities)
    assert first["repetitions"] in {128, 256, 512}


def test_stats_training_rows_use_two_complete_english_boundary_views() -> None:
    scenario = _categorical_scenario()
    simulation = simulate_scenario(scenario)
    first = _build_record(
        scenario,
        simulation,
        language="en",
        loss_weight=1.4,
        incomplete=False,
        variant="dgp-regret",
        refined_explanation=None,
        view="boundary_a",
    )
    second = _build_record(
        scenario,
        simulation,
        language="en",
        loss_weight=1.4,
        incomplete=False,
        variant="dgp-regret",
        refined_explanation=None,
        view="boundary_b",
    )
    assert not _validate_record(first)
    assert not _validate_record(second)
    assert first["messages"][1]["content"] != second["messages"][1]["content"]
    assert first["metadata"]["selected_method_id"] != "needs_clarification"
    assert second["metadata"]["selected_method_id"] != "needs_clarification"
    assert "Data schema:" in first["messages"][1]["content"]
    assert "variables" in first["messages"][2]["content"]
    weights = {"en": 1.4 + 1.4, "zh_Hant": 0.6, "zh_Hans": 0.6}
    total = sum(weights.values())
    assert {key: value / total for key, value in weights.items()} == pytest.approx(
        {"en": 0.70, "zh_Hant": 0.15, "zh_Hans": 0.15}
    )


def test_component_masks_exclude_user_and_tool_result_content() -> None:
    scenario = _categorical_scenario()
    simulation = simulate_scenario(scenario)
    row = _build_record(
        scenario,
        simulation,
        language="en",
        loss_weight=1.4,
        incomplete=False,
        variant="dgp-regret",
        refined_explanation=None,
        view="boundary_a",
    )
    tokenizer = _CharacterTokenizer()
    item = _tokenize_stats_record(tokenizer, row)
    rendered = tokenizer.apply_chat_template(row["messages"])
    tool_result_start = rendered.index("<tool_result>")
    tool_result_end = rendered.index("</tool_result>") + len("</tool_result>")
    for character_index in range(tool_result_start, tool_result_end):
        target_index = character_index - 1
        if target_index >= 0:
            assert not item["plan_mask"][target_index]
            assert not item["report_mask"][target_index]
    assert sum(item["plan_mask"]) > 0
    assert sum(item["report_mask"]) > 0
    assert len(set(item["candidate_token_ids"])) == len(item["candidate_token_ids"])


def test_selector_only_dataset_stops_after_the_causal_method_target() -> None:
    scenario = _categorical_scenario()
    simulation = simulate_scenario(scenario)
    row = _build_record(
        scenario,
        simulation,
        language="en",
        loss_weight=1.4,
        incomplete=False,
        variant="dgp-regret",
        refined_explanation=None,
        view="boundary_a",
    )
    tokenizer = _CharacterTokenizer()
    full = _tokenize_stats_record(tokenizer, row)
    selector_length = int(full["method_position"]) + 2
    dataset = StatsDataset(
        [row],
        tokenizer,
        seed=42,
        grouped=False,
        curriculum="random",
        max_seq_length=selector_length,
        selector_only=True,
    )
    assert len(dataset.items[0]["tokens"]) == selector_length
    assert selector_length < len(full["tokens"])
    assert not any(dataset.items[0]["plan_mask"])
    assert not any(dataset.items[0]["report_mask"])


def test_stats_collator_batches_variable_length_selector_items_without_truncation() -> None:
    items = [
        {
            "tokens": [1, 2, 3],
            "method_position": 1,
            "candidate_token_ids": [10, 11],
            "candidate_probabilities": [0.75, 0.25],
            "plan_mask": [False, False],
            "report_mask": [False, False],
            "loss_weight": 1.4,
        },
        {
            "tokens": [4, 5, 6, 7, 8],
            "method_position": 3,
            "candidate_token_ids": [12, 13, 14],
            "candidate_probabilities": [0.2, 0.3, 0.5],
            "plan_mask": [False] * 4,
            "report_mask": [False] * 4,
            "loss_weight": 0.6,
        },
    ]
    batch = _collate_stats_items(items, max_seq_length=64)
    assert batch[0].shape == (2, 33)
    assert batch[1].tolist() == [1, 3]
    assert batch[2].shape == (2, 6)
    assert batch[3][0, :2].tolist() == pytest.approx([0.75, 0.25])
    assert batch[4][1].tolist() == [True, True, True, False, False, False]
    assert batch[7].tolist() == pytest.approx([1.4, 0.6])
    with pytest.raises(RuntimeError, match="cannot be truncated"):
        _collate_stats_items(items, max_seq_length=4)


def test_teacher_refinement_cannot_change_method_or_numbers() -> None:
    template = "welch_t has Type I error 0.051, coverage 0.949, and regret 0.000."
    assert _refinement_valid(
        template,
        f"{template} It remains the validity-first choice.",
        "welch_t",
    )
    assert not _refinement_valid(
        template,
        "hc3_ols has Type I error 0.051, coverage 0.949, and regret 0.000.",
        "welch_t",
    )
    assert not _refinement_valid(
        template,
        "welch_t has Type I error 0.061, coverage 0.949, and regret 0.000.",
        "welch_t",
    )


def test_chat_token_count_uses_input_ids_not_encoding_field_count() -> None:
    class Tokenizer:
        def apply_chat_template(self, *_: object, **__: object) -> dict[str, list[int]]:
            return {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}

    assert _chat_token_count(Tokenizer(), []) == 4


def test_plan_validation_never_accepts_unknown_columns_or_methods() -> None:
    summary = [{"columns": [{"name": "y"}, {"name": "x"}]}]
    common = {
        "status": "ready",
        "estimand": "slope",
        "sampling_unit": "person",
        "study_design": "cross-sectional",
        "outcome_type": "continuous",
        "dependence": "independent",
        "missingness": "complete",
        "uncertainty": "HC3",
        "diagnostics": [],
        "tool": "python",
        "questions": [],
        "data_file_index": 0,
    }
    invalid_column = _validate_plan(
        {**common, "method_id": "hc3_ols", "variables": {"outcome": "y", "predictors": ["z"]}},
        summary,
    )
    assert invalid_column["status"] == "needs_clarification"
    invalid_method = _validate_plan(
        {**common, "method_id": "invented_test", "variables": {}}, summary
    )
    assert invalid_method["status"] == "needs_clarification"


def test_bounded_compiler_maps_survival_roles_without_model_generated_columns() -> None:
    question = """Research Question:
Does survival differ by treatment?

STUDY DESIGN:
- Type: Observational cohort
- Unit of observation: Patient

VARIABLE GLOSSARY:
- os_months: Overall survival time in months [OUTCOME]
- os_event: Death indicator, 1 = death and 0 = censored [OUTCOME]
- treatment: Treatment status [EXPOSURE]
"""
    summary = {
        "rows": 120,
        "columns": [
            {"name": "os_months", "dtype": "float64", "unique": 115, "missing": 0},
            {
                "name": "os_event",
                "dtype": "int64",
                "unique": 2,
                "missing": 0,
                "levels": [0, 1],
            },
            {
                "name": "treatment",
                "dtype": "int64",
                "unique": 2,
                "missing": 0,
                "levels": [0, 1],
            },
        ],
    }
    scaffold = compile_analysis_scaffold(question, [summary])
    assert scaffold.auto_ready
    assert scaffold.candidate_method_ids[0] == "logrank"
    plan = plan_from_scaffold(scaffold, "logrank")
    assert plan is not None
    assert plan["variables"] == {
        "time": "os_months",
        "event": "os_event",
        "group": "treatment",
    }


def test_bounded_compiler_keeps_only_declared_filter_and_recode() -> None:
    question = """Research Question:
Do driver alterations differ between sample types? Exclude NE cases.

STUDY DESIGN:
- Type: Observational cohort
- Unit of observation: Sample

VARIABLE GLOSSARY:
- status: Driver status (Driver, Loss, Unaltered); collapse to Driver vs all others [OUTCOME]
- sample_type: Sample type [EXPOSURE]
- evaluable: Evaluation status (OK, NE)
"""
    summary = {
        "rows": 200,
        "columns": [
            {
                "name": "status",
                "dtype": "object",
                "unique": 3,
                "missing": 0,
                "levels": ["Driver", "Loss", "Unaltered"],
            },
            {
                "name": "sample_type",
                "dtype": "object",
                "unique": 2,
                "missing": 0,
                "levels": ["Primary", "Metastasis"],
            },
            {
                "name": "evaluable",
                "dtype": "object",
                "unique": 2,
                "missing": 0,
                "levels": ["OK", "NE"],
            },
        ],
    }
    scaffold = compile_analysis_scaffold(question, [summary])
    plan = plan_from_scaffold(scaffold, "chi_square")
    assert plan is not None
    assert plan["analysis_options"]["row_filters"] == [
        {"column": "evaluable", "operation": "exclude", "values": ["NE"]}
    ]
    assert plan["analysis_options"]["binary_recodes"] == [
        {
            "column": "status",
            "positive_values": ["Driver"],
            "negative_rule": "all_other_observed_values",
        }
    ]


def test_bounded_compiler_repairs_only_with_audited_candidate() -> None:
    question = """Research Question:
Does the distribution of score differ by arm?

STUDY DESIGN:
- Type: Randomized study
- Unit of observation: Participant

VARIABLE GLOSSARY:
- score: Continuous score [OUTCOME]
- arm: Trial arm [TREATMENT]
"""
    summary = {
        "rows": 80,
        "columns": [
            {"name": "score", "dtype": "float64", "unique": 75, "missing": 0},
            {
                "name": "arm",
                "dtype": "object",
                "unique": 2,
                "missing": 0,
                "levels": ["control", "treatment"],
            },
        ],
    }
    scaffold = compile_analysis_scaffold(question, [summary])
    first = scaffold.candidate_method_ids[0]
    repair = next_repair_plan(scaffold, [first])
    assert repair is not None
    assert repair["method_id"] in scaffold.candidate_method_ids
    assert repair["method_id"] != first


def test_bounded_compiler_routes_columns_to_one_unambiguous_attachment() -> None:
    question = """Research Question:
Is y associated with x?

STUDY DESIGN:
- Type: Cross-sectional study
- Unit of observation: Participant

VARIABLE GLOSSARY:
- y: Continuous response [OUTCOME]
- x: Continuous exposure [EXPOSURE]
"""
    irrelevant = {
        "rows": 20,
        "columns": [{"name": "note", "dtype": "object", "unique": 20, "missing": 0}],
    }
    analysis = {
        "rows": 20,
        "columns": [
            {"name": "y", "dtype": "float64", "unique": 20, "missing": 0},
            {"name": "x", "dtype": "float64", "unique": 20, "missing": 0},
        ],
    }
    scaffold = compile_analysis_scaffold(question, [irrelevant, analysis])
    plan = plan_from_scaffold(scaffold, "spearman_correlation")
    assert plan is not None
    assert plan["data_file_index"] == 1

    split_scaffold = compile_analysis_scaffold(
        question,
        [
            {"rows": 20, "columns": [analysis["columns"][0]]},
            {"rows": 20, "columns": [analysis["columns"][1]]},
        ],
    )
    assert not split_scaffold.auto_ready
    assert plan_from_scaffold(split_scaffold, "spearman_correlation") is None


def test_dgp_evolve_task_reward_is_multiplicative_and_bounded() -> None:
    assert task_reward(
        validity=1.0,
        novelty=0.8,
        frontier=0.5,
        learning_progress=0.5,
    ) == pytest.approx(0.2)
    assert (
        task_reward(
            validity=0.0,
            novelty=1.0,
            frontier=1.0,
            learning_progress=1.0,
        )
        == 0.0
    )
    with pytest.raises(ValueError):
        task_reward(validity=1.1, novelty=1.0, frontier=1.0, learning_progress=1.0)


def test_dgp_evolve_mutation_is_reproducible_bounded_and_novel() -> None:
    parent = _categorical_scenario()
    first = _mutate_scenario(parent, cycle=1, index=0, seed=9001)
    second = _mutate_scenario(parent, cycle=1, index=0, seed=9001)
    assert first == second
    assert first.blueprint_id != parent.blueprint_id
    assert first.search and first.search["parent_blueprint_id"] == parent.blueprint_id
    family = next(item for item in FAMILIES if item.family_id == parent.family_id)
    assert all(
        family.parameters[key][0] <= value <= family.parameters[key][1]
        for key, value in first.parameters.items()
    )
    assert _novelty(parent, [parent]) == 0.0
    assert _novelty(first, [parent]) > 0.0


def test_dgp_evolve_selection_preserves_family_stepping_stones() -> None:
    proposals = [
        {
            "blueprint_id": "a-high",
            "family_id": "a",
            "task_reward": 0.9,
        },
        {
            "blueprint_id": "a-low",
            "family_id": "a",
            "task_reward": 0.5,
        },
        {
            "blueprint_id": "b-only",
            "family_id": "b",
            "task_reward": 0.2,
        },
    ]
    selected = _select_diverse(proposals, 2)
    assert {item["family_id"] for item in selected} == {"a", "b"}


def test_dgp_evolve_selection_caps_families_and_oracle_methods() -> None:
    proposals = []
    for index, (family_id, method_id) in enumerate(
        [
            ("a", "m1"),
            ("a", "m1"),
            ("a", "m2"),
            ("b", "m1"),
            ("b", "m2"),
            ("c", "m3"),
        ]
    ):
        proposals.append(
            {
                "blueprint_id": f"p{index}",
                "family_id": family_id,
                "task_reward": 1.0 - index / 10,
                "simulation": {"selected_method_id": method_id},
            }
        )
    selected = _select_diverse(
        proposals,
        5,
        max_per_family=2,
        max_per_method=2,
    )
    family_counts = Counter(str(item["family_id"]) for item in selected)
    method_counts = Counter(str(item["simulation"]["selected_method_id"]) for item in selected)
    assert len(selected) == 5
    assert set(family_counts) == {"a", "b", "c"}
    assert max(family_counts.values()) <= 2
    assert max(method_counts.values()) <= 2


def test_evolution_profile_keeps_v03_artifacts_immutable() -> None:
    root = Path(__file__).resolve().parents[1]
    stats = load_config(root / "configs" / "pipeline.stats.yaml")
    evolve = load_config(root / "configs" / "pipeline.evolve.yaml")
    assert stats.path_for("artifact_dir") != evolve.path_for("artifact_dir")
    assert stats.path_for("report_dir") != evolve.path_for("report_dir")
    assert evolve.path_for("parent_artifact_dir") == stats.path_for("artifact_dir")
    assert evolve.section("project")["version"] == "0.4.0-dev"


def test_evolution_uses_cycle_specific_promotion_shards() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "pipeline.evolve.yaml")
    first = _promotion_scenarios(config, 1)
    repeated = _promotion_scenarios(config, 1)
    second = _promotion_scenarios(config, 2)
    discovery_ids = {
        row["scenario"]["blueprint_id"]
        for row in map(json.loads, (root / "data/stats/surface/dev.jsonl").read_text().splitlines())
    }
    assert first == repeated
    assert len(first) == int(config.section("evolution")["promotion_shard"]["count"])
    assert all(item.split == "evolve-promotion-0001" for item in first)
    assert not ({item.blueprint_id for item in first} & discovery_ids)
    assert not ({item.blueprint_id for item in first} & {item.blueprint_id for item in second})


def test_evolution_never_replaces_a_prepared_promotion_shard(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "pipeline.evolve.yaml")
    (tmp_path / "promotion.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "promotion_manifest.json").write_text(
        json.dumps({"fingerprint": "different", "sha256": "different", "count": 1}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="promotion shard is immutable"):
        _ensure_promotion_shard(config, 999, tmp_path)


def test_evolution_rejects_concurrent_archive_writers(tmp_path: Path) -> None:
    class Config:
        def path_for(self, _name: str) -> Path:
            return tmp_path

    config = Config()
    with (
        _evolution_lock(config),  # type: ignore[arg-type]
        pytest.raises(RuntimeError, match="already running"),
        _evolution_lock(config),  # type: ignore[arg-type]
    ):
        raise AssertionError("the second writer unexpectedly acquired the lock")


def test_evolution_microsteps_cover_one_complete_group_epoch(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "pipeline.evolve.yaml")
    settings = config.section("evolution")
    groups = int(settings["train_groups_per_cycle"])
    replay = round(groups * settings["replay_fraction"] / (1 - settings["replay_fraction"]))
    assert int(settings["tasks_per_cycle"]) == groups
    assert 4 * (groups + replay) == int(settings["microsteps"])
    assert int(settings["max_tasks_per_family"]) == 4
    assert int(settings["max_tasks_per_method"]) == 4
    assert settings["component_weights"] == {
        "method": 1.0,
        "plan_tool": 0.0,
        "report": 0.0,
    }
    assert settings["ablation"]["arms"] == ["random-control", "adaptive"]
    assert not _training_records_are_current(config, tmp_path, {})


def test_base_bakeoff_requires_quality_language_memory_and_throughput_gates() -> None:
    def candidate(regret: float, seconds: float, memory: float, accuracy: float) -> dict:
        return {
            "base_trilingual": {"normalized_regret": regret},
            "base_languages": {
                language: {"accuracy": accuracy} for language in ("en", "zh_Hant", "zh_Hans")
            },
            "smoke": {"training_seconds": seconds, "peak_memory_gb": memory},
        }

    gates = {
        "minimum_relative_regret_improvement": 0.10,
        "maximum_language_accuracy_regression": 0.03,
        "maximum_peak_memory_gb": 21.0,
        "maximum_training_time_ratio": 2.5,
    }
    candidates = {
        "qwen3.5-4b": candidate(0.50, 100.0, 12.0, 0.50),
        "qwen3.5-9b": candidate(0.40, 220.0, 20.0, 0.48),
    }
    assert choose_base_bakeoff(candidates, gates)["recommended"] == "qwen3.5-9b"
    candidates["qwen3.5-9b"]["smoke"]["peak_memory_gb"] = 21.1
    decision = choose_base_bakeoff(candidates, gates)
    assert decision["recommended"] == "qwen3.5-4b"
    assert not decision["gates"]["peak_memory"]


def test_policy_projection_is_normalized_conservative_and_counterfactual() -> None:
    parent = [0.70, 0.20, 0.10]
    oracle = [0.01, 0.89, 0.10]
    result = project_policy(
        parent,
        oracle,
        oracle_temperature=0.15,
        projection_temperature=0.15,
        step_size=0.5,
        exploration_mass=0.05,
    )
    target = result["target_probabilities"]
    assert sum(target) == pytest.approx(1.0)
    assert target[1] > parent[1]
    assert result["target_expected_regret"] < result["parent_expected_regret"]
    assert result["kl_target_parent"] >= 0


def test_policy_projection_reconstructs_relative_regret_from_soft_labels() -> None:
    regrets = reconstruct_regrets(
        [0.9987289837369187, 0.0, 0.001271016263081358],
        oracle_temperature=0.15,
    )
    assert regrets.tolist() == pytest.approx([0.0, 1.0, 1.0], abs=1e-6)


def test_direct_regret_target_uses_verified_regret_and_a_parent_trust_region() -> None:
    target = direct_regret_target([0.8, 0.2], [1.0, 0.0], trust_beta=0.1)
    assert target == pytest.approx([0.784, 0.216])
    tied = direct_regret_target([0.6, 0.3, 0.1], [0.0, 0.0, 1.0], trust_beta=0.1)
    assert tied == pytest.approx([0.606, 0.303, 0.091])
    assert sum(tied) == pytest.approx(1.0)


def test_cvar_group_weights_preserve_mean_one_and_focus_the_tail() -> None:
    weights = cvar_group_weights(
        [0.1, 0.9, 0.2, 0.8],
        tail_fraction=0.5,
        uniform_floor=0.25,
    )
    assert weights == pytest.approx([0.25, 1.75, 0.25, 1.75])
    assert sum(weights) / len(weights) == pytest.approx(1.0)


def test_triggered_repair_preserves_anchors_and_repairs_verified_failures() -> None:
    anchor = triggered_repair_target(
        [0.8, 0.2],
        [0.2, 0.8],
        [1.0, 0.0],
        [False, True],
        role="anchor",
        expected_regret_threshold=0.35,
        repair_lambda_floor=0.25,
        repair_lambda_ceiling=1.0,
        invalid_argmax_lambda=1.0,
    )
    assert anchor["target_probabilities"] == pytest.approx([0.8, 0.2])
    assert anchor["repair_lambda"] == 0.0

    repair = triggered_repair_target(
        [0.8, 0.2],
        [0.2, 0.8],
        [1.0, 0.0],
        [False, True],
        role="repair",
        expected_regret_threshold=0.35,
        repair_lambda_floor=0.25,
        repair_lambda_ceiling=1.0,
        invalid_argmax_lambda=1.0,
    )
    assert repair["target_probabilities"] == pytest.approx([0.2, 0.8])
    assert repair["repair_lambda"] == 1.0
    assert repair["target_expected_regret"] < repair["parent_expected_regret"]


def test_triggered_repair_uses_one_semantic_group_trigger_across_views() -> None:
    result = triggered_repair_target(
        [0.6, 0.4],
        [0.3, 0.7],
        [1.0, 0.0],
        [True, True],
        role="repair",
        expected_regret_threshold=0.35,
        repair_lambda_floor=0.25,
        repair_lambda_ceiling=1.0,
        invalid_argmax_lambda=1.0,
        trigger_expected_regret=0.675,
        trigger_invalid_argmax=False,
    )
    assert result["repair_lambda"] == pytest.approx(0.625)
    assert sum(result["target_probabilities"]) == pytest.approx(1.0)


def test_targeted_group_selection_is_disjoint_and_deterministic() -> None:
    summaries = [
        {"group_id": "a", "expected_regret": 0.1, "invalid_argmax": False},
        {"group_id": "b", "expected_regret": 0.8, "invalid_argmax": False},
        {"group_id": "c", "expected_regret": 0.2, "invalid_argmax": True},
        {"group_id": "d", "expected_regret": 0.4, "invalid_argmax": False},
    ]
    roles = select_targeted_groups(summaries, repair_count=2, anchor_count=1)
    assert roles == {"c": "repair", "b": "repair", "a": "anchor"}


def test_crossfit_expert_selection_prefers_worst_fold_and_rejects_instability() -> None:
    options = [
        {
            "arm": "direct-cvar",
            "update": 2,
            "parent_fold_regrets": [0.5, 0.5, 0.5],
            "candidate_fold_regrets": [0.35, 0.35, 0.52],
        },
        {
            "arm": "direct-mean",
            "update": 3,
            "parent_fold_regrets": [0.5, 0.5, 0.5],
            "candidate_fold_regrets": [0.45, 0.44, 0.46],
        },
        {
            "arm": "boltzmann-mean",
            "update": 4,
            "parent_fold_regrets": [0.5, 0.5, 0.5],
            "candidate_fold_regrets": [0.40, 0.43, 0.45],
        },
    ]
    selected = select_crossfit_expert_option(
        options,
        minimum_family_pooled_relative_improvement=0.05,
        minimum_fold_relative_improvement=0.02,
        minimum_improving_folds=2,
        maximum_fold_relative_regression=0.02,
    )
    assert selected is not None
    assert selected["arm"] == "boltzmann-mean"
    assert selected["crossfit"]["worst_fold_relative_improvement"] == pytest.approx(0.1)


def test_crossfit_expert_selection_uses_pooled_regret_for_the_family_gate() -> None:
    selected = select_crossfit_expert_option(
        [
            {
                "arm": "direct-mean",
                "update": 1,
                "parent_fold_regrets": [0.1, 1.0, 1.0],
                "candidate_fold_regrets": [0.05, 0.98, 0.98],
            }
        ],
        minimum_family_pooled_relative_improvement=0.05,
        minimum_fold_relative_improvement=0.02,
        minimum_improving_folds=2,
        maximum_fold_relative_regression=0.02,
    )
    assert selected is None


def test_gradient_conflict_summary_separates_within_and_cross_family_geometry() -> None:
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [-1.0, 0.0],
            [-0.8, 0.2],
        ],
        dtype=np.float32,
    )
    report = summarize_gradient_conflicts(vectors, ["a", "a", "b", "b"])
    assert report["groups"] == 4
    assert report["families"] == 2
    assert report["mean_within_family_cosine"] > 0.9
    assert report["mean_cross_family_cosine"] < -0.9
    assert report["negative_cross_family_fraction"] == 1.0
    assert report["effective_gradient_rank"] > 1.0
    assert report["most_conflicting_family_pairs"][0]["left"] == "a"
    assert report["most_conflicting_family_pairs"][0]["right"] == "b"


def test_min_norm_simplex_direction_is_common_descent_when_cone_exists() -> None:
    gradients = np.asarray(
        [
            [1.0, 0.0],
            [0.2, 1.0],
            [0.2, -1.0],
        ],
        dtype=np.float64,
    )
    gradients /= np.linalg.norm(gradients, axis=1, keepdims=True)
    solution = min_norm_simplex_weights(
        gradients @ gradients.T,
        max_iterations=10000,
        tolerance=1e-12,
    )
    assert solution["weights"].sum() == pytest.approx(1.0)
    assert np.all(solution["weights"] >= 0.0)
    assert solution["minimum_alignment"] > 0.0
    assert np.all(solution["alignments"] > 0.0)


def test_common_descent_promotion_does_not_create_shard_when_dev_gates_fail(
    tmp_path: Path,
) -> None:
    class Config:
        def path_for(self, name: str) -> Path:
            return tmp_path / name

    artifact_dir = tmp_path / "artifact_dir" / "common-descent"
    (artifact_dir / "common-cone").mkdir(parents=True)
    (artifact_dir / "report.json").write_text(
        json.dumps({"complete": True, "proceed_to_promotion": False}),
        encoding="utf-8",
    )
    (artifact_dir / "common-cone" / "status.json").write_text(
        json.dumps({"complete": True}),
        encoding="utf-8",
    )
    result = promote_common_descent_candidate(Config())  # type: ignore[arg-type]
    assert result["stage"] == "skipped"
    assert not result["promotion_shard_opened"]
    assert not (tmp_path / "evolution_dir" / "cycles").exists()


def test_common_descent_promotion_uses_passing_delta_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Config:
        def path_for(self, name: str) -> Path:
            return tmp_path / name

    artifact_dir = tmp_path / "artifact_dir" / "common-descent"
    selected_dir = artifact_dir / "delta-calibration" / "scale-0p25"
    selected_dir.mkdir(parents=True)
    (artifact_dir / "report.json").write_text(
        json.dumps({"complete": True, "proceed_to_promotion": False}),
        encoding="utf-8",
    )
    (artifact_dir / "delta-calibration" / "report.json").write_text(
        json.dumps(
            {
                "complete": True,
                "fingerprint": "calibration",
                "proceed_to_promotion": True,
                "selected_status_path": str(selected_dir / "status.json"),
            }
        ),
        encoding="utf-8",
    )
    (selected_dir / "status.json").write_text(
        json.dumps(
            {
                "complete": True,
                "cycle": 6,
                "arm": "uniform-family-delta-0p25",
                "parent_adapter_path": str(tmp_path / "parent"),
                "parent_adapter_sha256": "parent-sha",
                "adapter_path": str(selected_dir),
                "adapter_sha256": "candidate-sha",
                "selected_checkpoint": "delta-scale-0p25",
                "family_order": ["family-a"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "charlie_alpha.stats_cone._load_archive",
        lambda _config: {
            "champion": {"adapter_sha256": "parent-sha"},
            "cycles": [{"cycle": index} for index in range(1, 6)],
        },
    )
    monkeypatch.setattr(
        "charlie_alpha.stats_cone._ensure_promotion_shard",
        lambda _config, _cycle, _data_dir: {"sha256": "promotion-sha"},
    )
    observed: dict[str, object] = {}

    def evaluate(_config, *, training, cycle_manifest, force):
        observed.update(
            training_arm=training["arm"],
            adapter_sha256=training["adapter_sha256"],
            development_stage=cycle_manifest["development_stage"],
            force=force,
        )
        return {"promoted": False, "gates": {"relative_regret": False}}

    monkeypatch.setattr("charlie_alpha.stats_cone.evaluate_evolution_candidate", evaluate)
    monkeypatch.setattr(
        "charlie_alpha.stats_cone._commit_cycle",
        lambda _config, _comparison: {"champion_changed": False},
    )
    result = promote_common_descent_candidate(Config())  # type: ignore[arg-type]
    assert result["stage"] == "complete"
    assert observed == {
        "training_arm": "uniform-family-delta-0p25",
        "adapter_sha256": "candidate-sha",
        "development_stage": "reusable-valid-dev-delta-calibration",
        "force": False,
    }


def test_common_descent_training_fingerprint_excludes_confirmation_gates() -> None:
    settings = {
        "updates": 4,
        "step_l2": 0.012,
        "confirmation": {"minimum_improvement": 0.05},
    }
    expected = {"updates": 4, "step_l2": 0.012}
    assert _common_descent_training_settings(settings) == expected
    settings["confirmation"]["minimum_improvement"] = 0.10
    assert _common_descent_training_settings(settings) == expected


def test_family_expert_training_fingerprint_excludes_evaluation_policy() -> None:
    settings = {
        "updates": 4,
        "step_l2": 0.012,
        "selection": {"minimum_route_relative_improvement": 0.05},
        "confirmation_shard": {"count": 240, "seed": 56000042},
        "gates": {"maximum_accuracy_regression": 0.03},
    }
    expected = {"updates": 4, "step_l2": 0.012}
    assert _expert_training_settings(settings) == expected
    settings["gates"]["maximum_accuracy_regression"] = 0.01
    assert _expert_training_settings(settings) == expected


def test_adapter_delta_interpolation_preserves_parent_and_candidate_endpoints(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.safetensors"
    candidate = tmp_path / "candidate.safetensors"
    prefix = "model.layers.28.proj"
    parent_weights = {
        f"{prefix}.lora_a": mx.array([[1.0], [2.0]]),
        f"{prefix}.lora_b": mx.array([[3.0, 4.0]]),
    }
    candidate_weights = {
        f"{prefix}.lora_a": mx.array([[5.0], [6.0]]),
        f"{prefix}.lora_b": mx.array([[7.0, 8.0]]),
    }
    mx.save_safetensors(str(parent), parent_weights)
    mx.save_safetensors(str(candidate), candidate_weights)

    def effective(weights: dict[str, mx.array]) -> mx.array:
        return weights[f"{prefix}.lora_a"] @ weights[f"{prefix}.lora_b"]

    parent_effective = effective(parent_weights)
    candidate_effective = effective(candidate_weights)
    for scale in (0.0, 0.5, 1.0):
        interpolated = interpolate_adapter_weights(parent, candidate, scale=scale)
        expected = (1.0 - scale) * parent_effective + scale * candidate_effective
        assert mx.allclose(effective(interpolated), expected).item()
        assert interpolated[f"{prefix}.lora_a"].shape == (2, 2)
        assert interpolated[f"{prefix}.lora_b"].shape == (2, 2)


def test_blockwise_effective_adapter_interpolation_is_layer_specific(tmp_path: Path) -> None:
    parent = tmp_path / "parent.safetensors"
    candidate = tmp_path / "candidate.safetensors"
    parent_weights: dict[str, mx.array] = {}
    candidate_weights: dict[str, mx.array] = {}
    for layer in (28, 29):
        prefix = f"model.layers.{layer}.proj"
        parent_weights[f"{prefix}.lora_a"] = mx.array([[1.0]])
        parent_weights[f"{prefix}.lora_b"] = mx.array([[2.0]])
        candidate_weights[f"{prefix}.lora_a"] = mx.array([[3.0]])
        candidate_weights[f"{prefix}.lora_b"] = mx.array([[4.0]])
    mx.save_safetensors(str(parent), parent_weights)
    mx.save_safetensors(str(candidate), candidate_weights)
    result = interpolate_adapter_blocks(
        parent,
        candidate,
        layer_scales={28: 0.0, 29: 1.0},
    )
    assert (
        result["model.layers.28.proj.lora_a"] @ result["model.layers.28.proj.lora_b"]
    ).item() == 2.0
    assert (
        result["model.layers.29.proj.lora_a"] @ result["model.layers.29.proj.lora_b"]
    ).item() == 12.0
    assert parse_layer_scales("28=0.25,29=1") == {28: 0.25, 29: 1.0}


def test_delta_scale_selection_maximizes_worst_surface_improvement() -> None:
    def surface(improvement: float, *, passed: bool = True) -> dict:
        return {
            "trilingual_relative_regret_improvement": improvement,
            "candidate_invalidity": 0.2,
            "candidate_accuracy": 0.6,
            "all_gates_passed": passed,
        }

    comparisons = {
        0.25: {"valid": surface(0.03), "dev": surface(0.02)},
        0.50: {"valid": surface(0.08), "dev": surface(0.01)},
        0.75: {"valid": surface(0.06), "dev": surface(0.04, passed=False)},
    }
    selected = choose_delta_scale(
        comparisons,
        minimum_worst_surface_improvement=0.01,
    )
    assert selected and selected["scale"] == 0.25


def test_block_projection_profiles_are_exhaustive_and_selection_is_gated() -> None:
    profiles = block_projection_profiles({"layers": [28, 29, 30, 31], "amplitudes": [0.25, 0.5]})
    assert len(profiles) == 30
    assert len({tuple(sorted(profile.items())) for profile in profiles}) == 30

    def surface(improvement: float, *, passed: bool = True) -> dict:
        return {
            "trilingual_relative_regret_improvement": improvement,
            "candidate_invalidity": 0.2,
            "candidate_accuracy": 0.6,
            "all_gates_passed": passed,
        }

    comparisons = [
        {
            "slug": "sparse",
            "layer_scales": {"28": 0.25, "29": 0.0},
            "active_layers": [28],
            "surfaces": {"valid": surface(0.03), "dev": surface(0.02)},
        },
        {
            "slug": "failed",
            "layer_scales": {"28": 0.5, "29": 0.5},
            "active_layers": [28, 29],
            "surfaces": {"valid": surface(0.08), "dev": surface(0.04, passed=False)},
        },
    ]
    selected = choose_block_profile(
        comparisons,
        minimum_worst_surface_improvement=0.01,
    )
    assert selected and selected["slug"] == "sparse"


def test_family_route_selects_improving_profile_and_keeps_parent_on_ties() -> None:
    def score(family_regrets: dict[str, float]) -> dict:
        predictions = [
            {
                "blueprint_id": family,
                "family_id": family,
                "domain": "inference_and_design",
                "predicted_method_id": "chosen",
                "oracle_method_id": "chosen" if regret == 0.0 else "oracle",
                "normalized_regret": regret,
                "valid": True,
            }
            for family, regret in family_regrets.items()
        ]
        return {
            "selector": {"predictions": predictions},
            "languages": {
                language: {"predictions": predictions} for language in ("en", "zh_Hant", "zh_Hans")
            },
        }

    options = {
        "parent": {
            "score": score({"family-a": 0.5, "family-b": 0.2}),
            "active_layers": [],
            "amplitude": 0.0,
        },
        "profile": {
            "score": score({"family-a": 0.1, "family-b": 0.2}),
            "active_layers": [31],
            "amplitude": 0.25,
        },
    }
    routes = select_family_routes(
        options,
        {
            "maximum_invalidity_increase": 0.0,
            "maximum_accuracy_regression": 0.03,
            "maximum_language_accuracy_regression": 0.03,
            "maximum_language_regret_increase": 0.05,
        },
    )
    assert routes["family-a"]["slug"] == "profile"
    assert routes["family-b"]["slug"] == "parent"


def test_family_expert_checkpoint_requires_registered_gain_and_parent_wins_ties() -> None:
    def metrics(regret: float) -> dict:
        return {
            "normalized_regret": regret,
            "accuracy": 0.6,
            "invalid_selection_rate": 0.1,
            "languages": {
                language: {
                    "normalized_regret": regret,
                    "accuracy": 0.6,
                    "invalid_selection_rate": 0.1,
                }
                for language in ("en", "zh_Hant", "zh_Hans")
            },
        }

    gates = {
        "maximum_invalidity_increase": 0.0,
        "maximum_accuracy_regression": 0.03,
        "maximum_language_accuracy_regression": 0.03,
        "maximum_language_regret_increase": 0.05,
    }
    options = {
        "parent": {"metrics": metrics(0.5), "checkpoint_name": "parent", "update": 0},
        "small": {"metrics": metrics(0.495), "checkpoint_name": "update-01", "update": 1},
    }
    selected = select_family_expert_checkpoint(
        options,
        gates=gates,
        minimum_relative_improvement=0.02,
    )
    assert selected["slug"] == "parent"

    options["strong"] = {
        "metrics": metrics(0.40),
        "checkpoint_name": "update-02",
        "update": 2,
    }
    selected = select_family_expert_checkpoint(
        options,
        gates=gates,
        minimum_relative_improvement=0.02,
    )
    assert selected["slug"] == "strong"
    assert selected["relative_regret_improvement"] == pytest.approx(0.2)


def test_router_question_removes_candidate_and_dgp_audit_leakage() -> None:
    record = {
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": (
                    "Study: ordered forecasting study. DGP audit values: drift=0.4, n=20. "
                    "Choose the primary analysis.\n\nCandidate menu:\nA. blocked_cv"
                ),
            },
            {"role": "assistant", "content": "answer"},
        ]
    }
    text = router_question(record)
    assert text == "Study: ordered forecasting study. Choose the primary analysis."
    assert "drift" not in text
    assert "Candidate" not in text


def test_character_ngram_router_round_trips_without_sklearn(tmp_path: Path) -> None:
    examples = [
        {"text": f"{family.family_id} alpha", "family_id": family.family_id} for family in FAMILIES
    ] + [{"text": f"{family.family_id} beta", "family_id": family.family_id} for family in FAMILIES]
    model = CharacterNgramNB.fit(
        examples,
        ngram_min=2,
        ngram_max=4,
        vocabulary_size=1000,
        minimum_document_frequency=1,
        alpha=0.25,
        uniform_class_prior=True,
    )
    assert character_ngrams("Bayes 123", 2, 3)
    predicted, confidence, probabilities = model.predict("bayesian_check alpha")
    assert predicted == "bayesian_check"
    assert 0.0 < confidence <= 1.0
    assert sum(probabilities.values()) == pytest.approx(1.0)
    path = tmp_path / "router.npz"
    model.save(path)
    restored = CharacterNgramNB.load(path)
    restored_family, restored_confidence, restored_probabilities = restored.predict(
        "bayesian_check alpha"
    )
    assert restored_family == predicted
    assert restored_confidence == pytest.approx(confidence)
    assert restored_probabilities == pytest.approx(probabilities)


def test_router_temperature_and_threshold_selection_are_deterministic() -> None:
    examples = [
        {"text": f"{family.family_id} signal", "family_id": family.family_id} for family in FAMILIES
    ]
    model = CharacterNgramNB.fit(
        examples,
        ngram_min=2,
        ngram_max=3,
        vocabulary_size=1000,
        minimum_document_frequency=1,
        alpha=0.25,
        uniform_class_prior=True,
    )
    calibration = choose_temperature(model, examples, [0.5, 1.0, 2.0])
    assert calibration["temperature"] in {0.5, 1.0, 2.0}
    assert len(calibration["candidates"]) == 3

    comparisons = [
        {
            "threshold": 0.5,
            "comparison": {
                "trilingual_relative_regret_improvement": 0.08,
                "candidate_invalidity": 0.2,
                "candidate_accuracy": 0.6,
            },
            "route_metrics": {"wrong_expert_rate": 0.02},
            "gates": {"all": True},
        },
        {
            "threshold": 0.9,
            "comparison": {
                "trilingual_relative_regret_improvement": 0.04,
                "candidate_invalidity": 0.1,
                "candidate_accuracy": 0.7,
            },
            "route_metrics": {"wrong_expert_rate": 0.0},
            "gates": {"all": True},
        },
    ]
    assert choose_router_threshold(comparisons)["threshold"] == 0.5


def test_selective_router_falls_back_for_parent_family_or_low_confidence() -> None:
    mapping = {
        "expert": {"checkpoint_name": "update-02", "slug": "expert-update-02"},
        "parent_only": {"checkpoint_name": "parent", "slug": "parent"},
    }
    assert (
        _route_slug(
            {"predicted_family_id": "expert", "confidence": 0.91},
            mapping,
            0.9,
        )
        == "expert-update-02"
    )
    assert (
        _route_slug(
            {"predicted_family_id": "expert", "confidence": 0.89},
            mapping,
            0.9,
        )
        == "parent"
    )
    assert (
        _route_slug(
            {"predicted_family_id": "parent_only", "confidence": 1.0},
            mapping,
            0.5,
        )
        == "parent"
    )


def test_parent_letter_router_uses_distinct_single_token_codes() -> None:
    class Tokenizer:
        def encode(self, text, **_kwargs):
            if len(text) == 1 and "A" <= text <= "M":
                return [100 + ord(text) - ord("A")]
            return [1, 2]

        def apply_chat_template(self, _messages, **_kwargs):
            return "prompt"

    class Model:
        def __call__(self, tokens):
            logits = mx.zeros((1, tokens.shape[1], 200))
            logits[:, -1, 111] = 10.0  # L -> time_series_leakage
            return logits

    router = ParentLetterRouter(Model(), Tokenizer())
    family, confidence, probabilities = router.predict("rolling forecast")
    assert family == "time_series_leakage"
    assert confidence > 0.99
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert "M: truly insufficient" in family_route_prompt()


def test_routed_prediction_aggregation_preserves_group_metrics() -> None:
    predictions = [
        {
            "blueprint_id": "b",
            "family_id": "family-b",
            "domain": "prediction_and_analysis",
            "predicted_method_id": "wrong",
            "oracle_method_id": "right",
            "normalized_regret": 0.6,
            "valid": False,
        },
        {
            "blueprint_id": "a",
            "family_id": "family-a",
            "domain": "inference_and_design",
            "predicted_method_id": "right",
            "oracle_method_id": "right",
            "normalized_regret": 0.0,
            "valid": True,
        },
    ]
    result = _aggregate_predictions(predictions)
    assert result["normalized_regret"] == pytest.approx(0.3)
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["invalid_selection_rate"] == pytest.approx(0.5)
    assert [row["blueprint_id"] for row in result["predictions"]] == ["a", "b"]


def test_evolution_rejects_numerically_identical_adapter_files(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    weights = {"layer.lora_a": mx.array([[1.0, 2.0]], dtype=mx.float32)}
    mx.save_safetensors(str(left / "adapters.safetensors"), weights)
    mx.save_safetensors(str(right / "adapters.safetensors"), weights)
    assert _adapter_max_abs_delta(left, right) == 0.0
    mx.save_safetensors(
        str(right / "adapters.safetensors"),
        {"layer.lora_a": mx.array([[1.0, 2.5]], dtype=mx.float32)},
    )
    assert _adapter_max_abs_delta(left, right) == pytest.approx(0.5)


def test_effective_adapter_delta_detects_equivalent_different_ranks(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    prefix = "model.layers.28.proj"
    parent = {
        f"{prefix}.lora_a": mx.array([[1.0], [2.0]]),
        f"{prefix}.lora_b": mx.array([[3.0, 4.0]]),
    }
    other = {
        f"{prefix}.lora_a": mx.array([[5.0], [6.0]]),
        f"{prefix}.lora_b": mx.array([[7.0, 8.0]]),
    }
    mx.save_safetensors(str(left / "adapters.safetensors"), parent)
    other_path = tmp_path / "other.safetensors"
    mx.save_safetensors(str(other_path), other)
    mx.save_safetensors(
        str(right / "adapters.safetensors"),
        interpolate_adapter_weights(
            left / "adapters.safetensors",
            other_path,
            scale=0.0,
        ),
    )
    config = {"lora_parameters": {"scale": 20.0}}
    (left / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (right / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    assert _adapter_max_abs_delta(left, right) == pytest.approx(0.0)


def test_evolution_checkpoint_selection_uses_validation_regret_before_gate() -> None:
    gates = {
        "minimum_relative_regret_improvement": 0.01,
        "maximum_accuracy_regression": 0.03,
    }
    candidates = [
        {
            "name": "parent",
            "path": "parent",
            "selector": {
                "normalized_regret": 0.50,
                "accuracy": 0.50,
                "invalid_selection_rate": 0.30,
            },
        },
        {
            "name": "bad-invalidity",
            "path": "bad",
            "selector": {
                "normalized_regret": 0.40,
                "accuracy": 0.55,
                "invalid_selection_rate": 0.31,
            },
        },
        {
            "name": "eligible",
            "path": "good",
            "selector": {
                "normalized_regret": 0.45,
                "accuracy": 0.50,
                "invalid_selection_rate": 0.30,
            },
        },
    ]
    assert _choose_checkpoint(candidates, gates)["name"] == "eligible"
    assert _choose_checkpoint(candidates[:2], gates)["name"] == "parent"


def test_evolution_ablation_winner_requires_equal_compute_and_uses_validation_regret() -> None:
    def status(arm: str, regret: float, loss: float) -> dict[str, object]:
        return {
            "arm": arm,
            "parent_adapter_sha256": "parent",
            "valid_sha256": "valid",
            "planned_microsteps": 160,
            "trainable_parameters": 2_129_920,
            "selected_validation_loss": loss,
            "checkpoint_selection": {
                "selected": "last",
                "candidates": [
                    {
                        "name": "last",
                        "selector": {
                            "normalized_regret": regret,
                            "accuracy": 0.5,
                            "invalid_selection_rate": 0.2,
                        },
                    }
                ],
            },
        }

    control = status("random-control", 0.35, 1.0)
    adaptive = status("adaptive", 0.30, 1.1)
    assert _choose_ablation_winner([control, adaptive])["arm"] == "adaptive"
    tied = status("adaptive", 0.35, 0.9)
    assert _choose_ablation_winner([control, tied])["arm"] == "adaptive"
    tied["selected_validation_loss"] = 1.0
    assert _choose_ablation_winner([control, tied])["arm"] == "random-control"
    mismatched = status("adaptive", 0.30, 1.0)
    mismatched["planned_microsteps"] = 80
    with pytest.raises(RuntimeError, match="equal parent, validation, or compute"):
        _choose_ablation_winner([control, mismatched])


def test_evolution_learning_signal_is_validation_only_shrunk_and_bounded() -> None:
    parent = [
        {"family_id": "a", "normalized_regret": 0.4},
        {"family_id": "b", "normalized_regret": 0.2},
    ]
    candidate = [
        {"family_id": "a", "normalized_regret": 0.3},
        {"family_id": "b", "normalized_regret": 0.3},
    ]
    signal = _family_learning_signal(parent, candidate)
    assert 0.5 < signal["a"] < 0.75
    assert 0.25 < signal["b"] < 0.5
    assert _family_learning_signal(parent, parent) == pytest.approx({"a": 0.5, "b": 0.5})


def test_evolution_curriculum_interleaves_replay_groups() -> None:
    dataset = object.__new__(StatsDataset)
    dataset.grouped = True
    dataset.curriculum = "evolve-interleave"
    dataset.items = []
    for source, count in (("new", 8), ("replay", 2)):
        for group_index in range(count):
            for _ in range(4):
                dataset.items.append(
                    {
                        "group_id": f"{source}-{group_index}",
                        "boundary_round": 2,
                        "evolution_source": source,
                    }
                )
    order = _group_order(dataset, seed=42, epoch=0)
    ordered_groups = [dataset.items[index]["group_id"] for index in order[::4]]
    assert ordered_groups[4].startswith("replay-")
    assert ordered_groups[9].startswith("replay-")


def test_policy_balanced_curriculum_covers_each_family_and_method_before_checkpoint() -> None:
    dataset = object.__new__(StatsDataset)
    dataset.grouped = True
    dataset.curriculum = "policy-balanced"
    dataset.items = []
    for group_index in range(12):
        source = "replay" if group_index in {4, 9} else "new"
        for _ in range(4):
            dataset.items.append(
                {
                    "group_id": f"group-{group_index:02d}",
                    "boundary_round": 2,
                    "evolution_source": source,
                    "metadata": {
                        "family_id": f"family-{group_index:02d}",
                        "selected_method_id": f"method-{group_index:02d}",
                    },
                }
            )
    order = _group_order(dataset, seed=42, epoch=0)
    first_checkpoint = [dataset.items[index] for index in order[::4][:12]]
    assert len({item["metadata"]["family_id"] for item in first_checkpoint}) == 12
    assert len({item["metadata"]["selected_method_id"] for item in first_checkpoint}) == 12
    assert [item["evolution_source"] for item in first_checkpoint].count("replay") == 2


def test_evolution_noninferiority_checks_every_language_and_family() -> None:
    assert _noninferior_mapping(
        {"en": 0.50, "zh_Hant": 0.40},
        {"en": 0.55, "zh_Hant": 0.38},
        maximum_regression=0.03,
        higher_is_better=True,
    )
    assert not _noninferior_mapping(
        {"en": 0.50, "zh_Hant": 0.40},
        {"en": 0.55, "zh_Hant": 0.35},
        maximum_regression=0.03,
        higher_is_better=True,
    )
    predictions = [
        {"family_id": "a", "normalized_regret": 0.2},
        {"family_id": "a", "normalized_regret": 0.4},
        {"family_id": "b", "normalized_regret": 0.1},
    ]
    assert _group_regret(predictions, "family_id") == pytest.approx({"a": 0.3, "b": 0.1})


def test_stats_routing_uses_files_or_statistical_content() -> None:
    assert classify_stats_route("hello", has_files=True) == "stats"
    assert classify_stats_route("estimate a confidence interval", has_files=False) == "stats"
    assert classify_stats_route("write a poem", has_files=False) == "base"
    assert classify_stats_route("anything", has_files=False, override="adapter") == "stats"
    with pytest.raises(ValueError):
        classify_stats_route("anything", has_files=False, override="unknown")


def test_early_stopping_reports_actual_and_planned_microsteps() -> None:
    legacy = {
        "microsteps": 1920,
        "optimizer_updates": 480,
        "stopped": True,
        "best_validation_iteration": 799,
        "train_history": [{"iteration": 1116}],
        "validation_history": [
            {"iteration": 0, "loss": 2.1},
            {"iteration": 799, "loss": 1.05},
            {"iteration": 1119, "loss": 1.06},
            {"iteration": 1920, "loss": 1.06},
        ],
    }
    normalized = _normalize_training_progress(legacy, grad_accumulation_steps=4)
    assert normalized["planned_microsteps"] == 1920
    assert normalized["microsteps"] == 1120
    assert normalized["optimizer_updates"] == 280
    assert normalized["best_validation_microstep"] == 800
    assert normalized["validation_history"][-1]["iteration"] == 1119


def test_retention_normalization_accepts_unicode_compatibility_forms() -> None:
    assert _normalize("H₂O") == _normalize("H2O")
    assert _normalize("CO₂") == _normalize("CO2")


def test_pbench_lock_is_half_easy_half_hard_and_collectively_covers_17() -> None:
    rows = []
    categories = [f"category-{index}" for index in range(17)]
    for difficulty in ("Easy", "Hard"):
        for category in categories:
            for repeat in range(3):
                rows.append(
                    {
                        "difficulty": difficulty,
                        "category": category,
                        "dataset_bytes": 100 + repeat,
                    }
                )
    selected = _pbench_indices(rows, seed=42, count=90)
    picked = [rows[index] for index in selected]
    assert sum(row["difficulty"] == "Easy" for row in picked) == 45
    assert sum(row["difficulty"] == "Hard" for row in picked) == 45
    assert len({row["category"] for row in picked}) == 17


def test_committed_stats_lock_has_no_evaluation_text() -> None:
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "configs" / "evaluation.stats.lock.json").read_text())
    assert lock["sealed"]
    assert lock["p_bench"]["count"] == 90
    assert lock["statqa"]["count"] == 200
    assert lock["final_dgp"]["count"] == 120
    assert len({row["category"] for row in lock["p_bench"]["tasks"]}) == 17
    serialized = json.dumps(lock).lower()
    assert "question" not in serialized
    assert "ground_truth" not in serialized


def test_public_isolation_report_omits_streams_and_local_paths() -> None:
    result = _public_isolation_report(
        {
            "passed": True,
            "python": {
                "passed": True,
                "checks": {"network_blocked": True},
                "sandbox": {
                    "returncode": 0,
                    "isolated": True,
                    "stdout": "ok",
                    "stderr": "/Users/private/source.py: operation denied",
                },
            },
        }
    )
    serialized = json.dumps(result)
    assert result["passed"]
    assert result["python"]["sandbox"] == {"returncode": 0, "isolated": True}
    assert "stdout" not in serialized
    assert "stderr" not in serialized
    assert "/Users/" not in serialized


def test_stats_input_keeps_declared_extension_for_content_addressed_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "e26cd91c22d766dff31276876ac8fdd102a6da87"
    target.write_text("value\n1\n", encoding="utf-8")
    declared = tmp_path / "dataset.csv"
    declared.symlink_to(target)
    with StatsToolSession(
        python_executable=Path("/usr/bin/python3"),
        r_executable=None,
        limits=SandboxLimits(timeout_seconds=20),
    ) as session:
        copied = session.add_files(
            [declared],
            allowed_extensions={".csv"},
            max_files=3,
            max_file_bytes=25 * 1024**2,
            max_total_bytes=50 * 1024**2,
        )
        assert copied[0].suffix == ".csv"
        assert copied[0].read_text(encoding="utf-8") == "value\n1\n"


@pytest.mark.skipif(platform.system() != "Darwin", reason="sandbox-exec is macOS-only")
def test_stats_python_and_r_sandboxes_block_all_escape_classes() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "pipeline.stats.yaml")
    runtime = resolve_stats_runtime(config)
    result = sandbox_self_test(
        python_executable=runtime.python,
        r_executable=runtime.rscript,
    )
    assert result["passed"], result


@pytest.mark.skipif(platform.system() != "Darwin", reason="sandbox-exec is macOS-only")
def test_audited_python_runtime_executes_auxiliary_procedure(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "pipeline.stats.yaml")
    runtime = resolve_stats_runtime(config)
    data = tmp_path / "binary.csv"
    data.write_text("event\n1\n1\n1\n0\n", encoding="utf-8")
    singleton = tmp_path / "singleton.csv"
    singleton.write_text("outcome,group\n1,a\n2,b\n3,b\n", encoding="utf-8")
    recode = tmp_path / "recode.csv"
    recode.write_text("status,aux\nDriver,1\n,2\nLoss,3\n", encoding="utf-8")
    script = (root / "src" / "charlie_alpha" / "runtime" / "stats_tool.py").read_text()
    with StatsToolSession(
        python_executable=runtime.python,
        r_executable=runtime.rscript,
        limits=SandboxLimits(timeout_seconds=20),
    ) as session:
        copied = session.add_files(
            [data, singleton, recode],
            allowed_extensions={".csv"},
            max_files=3,
            max_file_bytes=25 * 1024**2,
            max_total_bytes=50 * 1024**2,
        )
        result = session.run_python(
            script,
            {
                "method_id": "binomial_test",
                "data_path": str(copied[0]),
                "variables": {"outcome": "event"},
                "analysis_options": {"null_probability": 0.5},
            },
        )
        singleton_result = session.run_python(
            script,
            {
                "method_id": "difference_in_means",
                "data_path": str(copied[1]),
                "variables": {"outcome": "outcome", "treatment": "group"},
            },
        )
        recode_result = session.run_python(
            script,
            {
                "method_id": "binomial_test",
                "data_path": str(copied[2]),
                "variables": {"outcome": "status"},
                "analysis_options": {
                    "binary_recodes": [{"column": "status", "positive_values": ["Driver"]}]
                },
            },
        )
    assert result.returncode == 0, result
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["status"] == "ok"
    assert math.isclose(payload["result"]["estimated_probability"], 0.75)
    assert singleton_result.returncode == 2
    singleton_payload = json.loads(singleton_result.stdout.strip().splitlines()[-1])
    assert singleton_payload == {
        "status": "error",
        "error": "each group requires at least two complete observations",
    }
    assert recode_result.returncode == 0, recode_result
    recode_payload = json.loads(recode_result.stdout.strip().splitlines()[-1])
    assert recode_payload["result"]["n"] == 2

import json
import math
import platform
from pathlib import Path

import pytest

from charlie_alpha.config import load_config
from charlie_alpha.stats_agent import (
    _sandbox_metadata,
    _validate_plan,
    classify_stats_route,
    resolve_stats_runtime,
)
from charlie_alpha.stats_catalog import (
    AGENT_PROCEDURES,
    FAMILIES,
    PROCEDURES,
    validate_catalog,
)
from charlie_alpha.stats_data import (
    _build_record,
    _chat_token_count,
    _refinement_valid,
    _validate_record,
)
from charlie_alpha.stats_dgp import Scenario, build_blueprints, simulate_scenario
from charlie_alpha.stats_eval import _normalize, _pbench_indices
from charlie_alpha.stats_release import _public_isolation_report
from charlie_alpha.stats_sandbox import SandboxLimits, StatsToolSession, sandbox_self_test
from charlie_alpha.stats_training import _normalize_training_progress, _tokenize_stats_record


class _CharacterTokenizer:
    def __init__(self) -> None:
        self._tokenizer = self

    def apply_chat_template(self, messages, **_kwargs):
        return "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
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
    scenarios = build_blueprints(
        {"train": 240, "valid": 30, "dev": 60, "final": 120}, seed=42
    )
    assert len(scenarios) == 450
    assert len({item.blueprint_id for item in scenarios}) == 450
    for split, expected in {"train": 240, "valid": 30, "dev": 60, "final": 120}.items():
        rows = [item for item in scenarios if item.split == split]
        assert len(rows) == expected
        domains = {name: sum(item.domain == name for item in rows) for name in {
            "inference_and_design",
            "probability_and_bayes",
            "prediction_and_analysis",
        }}
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
    random_ablation = build_blueprints(
        {"train": 240}, seed=42, active_search=False
    )
    assert all(item.boundary_round == 0 and item.search is None for item in random_ablation)
    assert not ({item.blueprint_id for item in training} & {
        item.blueprint_id for item in random_ablation
    })


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
    script = (root / "src" / "charlie_alpha" / "runtime" / "stats_tool.py").read_text()
    with StatsToolSession(
        python_executable=runtime.python,
        r_executable=runtime.rscript,
        limits=SandboxLimits(timeout_seconds=20),
    ) as session:
        copied = session.add_files(
            [data],
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
    assert result.returncode == 0, result
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["status"] == "ok"
    assert math.isclose(payload["result"]["estimated_probability"], 0.75)

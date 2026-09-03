from pathlib import Path

from charlie_alpha.config import ProjectConfig
from charlie_alpha.stats_router_counterfactual import _compose_matched_pbench
from charlie_alpha.stats_router_external import (
    _aggregate_pbench,
    _aggregate_statqa,
    _historical_runtime_config,
)
from charlie_alpha.stats_router_replay import _transition_summary


def test_external_aggregates_preserve_exact_denominators() -> None:
    statqa = _aggregate_statqa(
        [
            {
                "exact_correct": True,
                "method_correct": True,
                "columns_correct": False,
                "column_recall": 0.5,
            },
            {
                "exact_correct": False,
                "method_correct": False,
                "columns_correct": True,
                "column_recall": 1.0,
            },
        ]
    )
    pbench = _aggregate_pbench(
        [
            {"raw_correct": True, "strict_correct": False},
            {"raw_correct": False, "strict_correct": False},
        ]
    )
    assert statqa["accuracy"] == 0.5
    assert statqa["method_set_accuracy"] == 0.5
    assert statqa["column_set_accuracy"] == 0.5
    assert statqa["column_recall"] == 0.75
    assert pbench["raw_accuracy"] == 0.5
    assert pbench["strict_accuracy"] == 0.0


def test_historical_runtime_only_raises_transport_ceiling() -> None:
    config = ProjectConfig(
        path=Path("config.yaml"),
        root=Path("."),
        values={"stats_tools": {"max_output_bytes": 65_536, "max_calls": 4}},
        sources={"schema_version": 1},
    )
    updated = _historical_runtime_config(config)
    assert config.section("stats_tools")["max_output_bytes"] == 65_536
    assert updated.section("stats_tools") == {
        "max_output_bytes": 131_072,
        "max_calls": 4,
    }


def test_matched_replay_transition_summary_separates_improvement_and_drift() -> None:
    parent = [
        {"id": "a", "correct": False},
        {"id": "b", "correct": False},
        {"id": "c", "correct": True},
        {"id": "d", "correct": True},
    ]
    candidate = [
        {"id": "a", "correct": False},
        {"id": "b", "correct": True},
        {"id": "c", "correct": False},
        {"id": "d", "correct": True},
    ]
    result = _transition_summary(
        parent,
        candidate,
        id_field="id",
        metric_fields=("correct",),
    )["correct"]
    assert result == {
        "count": 4,
        "false_to_false": 1,
        "false_to_true": 1,
        "true_to_false": 1,
        "true_to_true": 1,
        "parent_accuracy": 0.5,
        "candidate_accuracy": 0.5,
        "delta_points": 0.0,
        "changed_correctness": 2,
    }


def test_counterfactual_replay_composes_disjoint_exact_coverage() -> None:
    result = _compose_matched_pbench(
        [{"task_id": "parent", "raw_correct": True}],
        [{"task_id": "expert-a", "raw_correct": False}],
        [{"task_id": "expert-b", "raw_correct": True}],
        expected_task_ids={"parent", "expert-a", "expert-b"},
    )
    assert [row["task_id"] for row in result] == ["expert-a", "expert-b", "parent"]

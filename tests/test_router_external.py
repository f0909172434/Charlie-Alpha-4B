from pathlib import Path

from charlie_alpha.config import ProjectConfig
from charlie_alpha.stats_router_external import (
    _aggregate_pbench,
    _aggregate_statqa,
    _historical_runtime_config,
)


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

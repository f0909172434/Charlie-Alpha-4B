import pytest

from charlie_alpha.stats_cross_format_amendment import _paired_selector_rows


def test_paired_selector_rows_requires_matching_blueprint_ids() -> None:
    records = [{"metadata": {"blueprint_id": "a"}}]
    simulations = [{"scenario": {"blueprint_id": "a"}}]
    assert _paired_selector_rows(records, simulations) == [(records[0], simulations[0])]


def test_paired_selector_rows_rejects_mismatched_blueprints() -> None:
    records = [{"metadata": {"blueprint_id": "a"}}]
    simulations = [{"scenario": {"blueprint_id": "b"}}]
    with pytest.raises(RuntimeError, match="pairing mismatch"):
        _paired_selector_rows(records, simulations)

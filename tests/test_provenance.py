import json
from pathlib import Path

import pytest

from charlie_alpha.provenance import (
    creation_surface_open_state,
    lifecycle_open_state,
    mark_lifecycle_opened,
)


def test_creation_surface_open_state_is_explicitly_creation_time() -> None:
    state = creation_surface_open_state()
    assert state["promotion_surface_opened"] is False
    assert state["final_surface_opened"] is False
    assert "creation-time snapshot" in state["open_state_semantics"]
    assert "runtime reports are authoritative" in state["open_state_semantics"]


def test_mark_lifecycle_opened_is_copy_on_write() -> None:
    original = lifecycle_open_state()
    updated = mark_lifecycle_opened(original, "selection_opened")
    assert original["selection_opened"] is False
    assert updated["selection_opened"] is True
    assert updated["confirmation_opened"] is False
    assert "mutable lifecycle snapshot" in updated["open_state_semantics"]


def test_mark_lifecycle_opened_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="Unknown lifecycle open-state field"):
        mark_lifecycle_opened(lifecycle_open_state(), "sealed_final_surface_opened")


def test_committed_training_snapshots_reflect_completed_selection() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("robust-family-experts-training.json", "targeted-repair-training.json"):
        report = json.loads((root / "reports" / "evolve" / name).read_text(encoding="utf-8"))
        assert report["selection_opened"] is True
        assert report["confirmation_opened"] is False
        assert report["promotion_opened"] is False
        assert report["final_simulations_opened"] is False
        assert report["final_scores_opened"] is False
        assert "mutable lifecycle snapshot" in report["open_state_semantics"]

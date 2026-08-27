from __future__ import annotations

from typing import Any

_CREATION_STATE_SEMANTICS = (
    "creation-time snapshot; downstream runtime reports are authoritative after scoring"
)
_LIFECYCLE_STATE_SEMANTICS = (
    "mutable lifecycle snapshot; downstream stages refresh opened-state fields"
)
_LIFECYCLE_OPEN_FIELDS = frozenset(
    {
        "selection_opened",
        "confirmation_opened",
        "promotion_opened",
        "final_simulations_opened",
        "final_scores_opened",
    }
)


def creation_surface_open_state() -> dict[str, Any]:
    """Return backward-compatible creation-time open-state metadata for immutable surfaces."""
    return {
        "open_state_semantics": _CREATION_STATE_SEMANTICS,
        "promotion_surface_opened": False,
        "final_surface_opened": False,
    }


def lifecycle_open_state() -> dict[str, Any]:
    """Return mutable lifecycle metadata used by training/status snapshots."""
    return {
        "open_state_semantics": _LIFECYCLE_STATE_SEMANTICS,
        "selection_opened": False,
        "confirmation_opened": False,
        "promotion_opened": False,
        "final_simulations_opened": False,
        "final_scores_opened": False,
    }


def mark_lifecycle_opened(status: dict[str, Any], field: str) -> dict[str, Any]:
    """Mark one lifecycle stage open without mutating the caller's status mapping."""
    if field not in _LIFECYCLE_OPEN_FIELDS:
        raise ValueError(f"Unknown lifecycle open-state field: {field}")
    updated = dict(status)
    updated["open_state_semantics"] = _LIFECYCLE_STATE_SEMANTICS
    updated[field] = True
    return updated

import pytest

from charlie_alpha.stats_sufficiency_guard import ParentSufficiencyGuard


def test_sufficiency_guard_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="strictly between"):
        ParentSufficiencyGuard(None, None, threshold=0.0)
    with pytest.raises(ValueError, match="strictly between"):
        ParentSufficiencyGuard(None, None, threshold=1.0)

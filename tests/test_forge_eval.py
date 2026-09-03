import json
from pathlib import Path

import pytest

from charlie_alpha.forge_eval import verify_frozen_recipe


def test_committed_lock_is_strictly_disjoint() -> None:
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "configs" / "evaluation.v2.lock.json").read_text())
    dev = {row["task_id"] for row in lock["suites"]["dev"]}
    final = {row["task_id"] for row in lock["suites"]["final"]}
    assert not dev & final
    assert len(dev) == 34
    assert len(final) == 62


def test_final_evaluation_requires_a_frozen_recipe(tmp_path: Path) -> None:
    class MissingFreezeConfig:
        def path_for(self, key: str) -> Path:
            assert key == "artifact_dir"
            return tmp_path

    with pytest.raises(RuntimeError, match="sealed"):
        verify_frozen_recipe(MissingFreezeConfig())  # type: ignore[arg-type]

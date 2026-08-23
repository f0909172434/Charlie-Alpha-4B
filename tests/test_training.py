from pathlib import Path

from charlie_alpha.training import _is_oom, _recoverable_oom_snapshot


def test_recovers_checkpoint_from_oom_before_interrupted_restart(tmp_path: Path) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text(
        """Starting training..., iters: 100
Iter 50: Val loss 0.586, Val took 10s
Iter 50: Train loss 0.900, Learning Rate 1e-5, Peak mem 4.081 GB
Iter 50: Saved adapter weights to checkpoint
RuntimeError: [METAL] Insufficient Memory
Starting training..., iters: 100
Iter 1: Val loss 0.586, Val took 10s
""",
        encoding="utf-8",
    )

    assert not _is_oom(log_path)
    snapshot = _recoverable_oom_snapshot(log_path)
    assert snapshot is not None
    assert snapshot["last_checkpoint_iteration"] == 50
    assert snapshot["best_validation_loss"] == 0.586
    assert snapshot["peak_memory_gb"] == 4.081

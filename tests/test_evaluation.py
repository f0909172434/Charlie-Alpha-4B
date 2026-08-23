from charlie_alpha.evaluation import _score_task


def test_retention_canary_ignores_hidden_thinking() -> None:
    task = {
        "benchmark": "retention-canary",
        "domain": "stem",
        "language": "en",
        "gold": "Na",
    }
    assert _score_task(task, "<think>Sodium is Na.</think>\nNa")["passed"]
    assert not _score_task(task, "K")["passed"]

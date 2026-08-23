import numpy as np

import charlie_alpha.forge_training as forge_training
from charlie_alpha.forge_data import (
    _allocate,
    _protect,
    _restore,
    _selection_metrics,
    _selective_target_indices,
    _smooth_category_schedule,
    _translated_pair,
)
from charlie_alpha.forge_training import ForgeDataset, forge_iterate_batches


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        marker = 10 if "group-a" in messages[0]["content"] else 20
        if len(messages) == 1:
            return [marker, 2, 3]
        return [marker, 2, 3, 4, 5, 6]


def _row(group: str, slot: int) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"question {group}"},
            {"role": "assistant", "content": "answer"},
        ],
        "metadata": {
            "semantic_group_id": group,
            "microstep_slot": slot,
            "loss_weight": 1.0,
        },
    }


def test_placeholder_round_trip_is_exact() -> None:
    source = "Compute $x=2$ using ```python\nprint(2)\n``` at https://example.com."
    protected, mapping = _protect(source, "Q")
    assert "$x=2$" not in protected
    assert _restore(protected, mapping) == source
    assert _restore(protected.replace("<CA_Q_0000>", ""), mapping) is None


def test_placeholder_translation_preserves_both_chinese_scripts() -> None:
    source = {
        "messages": [
            {"role": "user", "content": "Compute $x=2$ and explain the result."},
            {
                "role": "assistant",
                "content": "Use the formula $x=2$. Therefore the answer is $2$.",
            },
        ]
    }
    output = (
        "<QUESTION_TRANSLATION>计算 <CA_Q_0000> 并解释这个计算结果。</QUESTION_TRANSLATION>"
        "<ANSWER_TRANSLATION>使用公式 <CA_A_0000>。因此答案是 <CA_A_0001>。"
        "</ANSWER_TRANSLATION>"
    )
    pair = _translated_pair(source, output)
    assert pair is not None
    assert "$x=2$" in pair[0][0]["content"]
    assert "計算" in pair[1][0]["content"]


def test_selective_mask_keeps_high_excess_and_final_tokens() -> None:
    row = {
        "metadata": {"prompt_offset_qwen35": 5},
        "selection": {"token_deltas": [-1.0, 0.2, 0.9, 0.3, -0.1, 0.1]},
    }
    settings = {
        "excess_loss_floor": 0.0,
        "selective_keep_fraction": 0.5,
        "final_token_floor": 2,
    }
    selected = _selective_target_indices(row, settings)
    assert 6 in selected  # highest excess-loss token: offset - 1 + index 2
    assert selected[-2:] == [8, 9]


def test_teacher_student_scores_require_identical_token_ids() -> None:
    row = {"metadata": {"candidate_id": "sample"}}
    student = {"tokens_sha256": "a", "token_losses": [1.0, 2.0]}
    teacher = {"tokens_sha256": "b", "token_losses": [0.5, 1.0]}
    try:
        _selection_metrics(row, student, teacher)
    except RuntimeError as error:
        assert "token IDs changed" in str(error)
    else:
        raise AssertionError("mismatched teacher/student token IDs must be rejected")


def test_category_allocation_and_schedule_are_balanced() -> None:
    counts = _allocate(52, {"math": 0.5, "python": 0.25, "cpp": 0.25})
    assert counts == {"math": 26, "python": 13, "cpp": 13}
    schedule = _smooth_category_schedule(counts)
    assert schedule.count("math") == 26
    assert schedule.count("python") == schedule.count("cpp") == 13
    assert max(
        abs(prefix.count("math") / len(prefix) - 0.5)
        for end in range(4, len(schedule) + 1)
        if (prefix := schedule[:end])
    ) <= 0.25


def test_batch_iterator_keeps_semantic_groups_together() -> None:
    rows = [
        *[_row("group-a", slot) for slot in range(3)],
        *[_row("group-b", slot) for slot in range(3)],
    ]
    dataset = ForgeDataset(
        rows, FakeTokenizer(), group_size=3, seed=42, grouped=True
    )
    batches = list(
        forge_iterate_batches(dataset, batch_size=1, max_seq_length=32, loop=False)
    )
    signatures = [tuple(np.asarray(batch[0])[0, :6]) for batch in batches]
    assert len(signatures) == 6
    # Both source groups use deterministic contiguous microsteps; group shuffle never interleaves.
    assert signatures[:3] == [signatures[0]] * 3
    assert signatures[3:] == [signatures[3]] * 3
    assert signatures[0] != signatures[3]


def test_batch_iterator_reuses_configured_padding_buckets() -> None:
    dataset = ForgeDataset(
        [_row("group-a", 0)],
        FakeTokenizer(),
        group_size=1,
        seed=42,
        grouped=True,
        padding_buckets=[8, 16, 32],
    )
    batch = next(
        forge_iterate_batches(
            dataset, batch_size=1, max_seq_length=32, loop=False
        )
    )
    assert batch[0].shape == (1, 8)


def test_gradient_checkpointing_wraps_each_layer_type_once(monkeypatch) -> None:
    class LayerA:
        pass

    class LayerB:
        pass

    model = type("Model", (), {"layers": [LayerA(), LayerA(), LayerB()]})()
    calls = []
    monkeypatch.setattr(forge_training, "grad_checkpoint", lambda layer: calls.append(type(layer)))
    monkeypatch.setattr(forge_training, "_CHECKPOINTED_LAYER_TYPES", set())
    forge_training._enable_gradient_checkpointing_once(model)
    forge_training._enable_gradient_checkpointing_once(model)
    assert calls == [LayerA, LayerB]

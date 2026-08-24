import json

import mlx.core as mx
import numpy as np
import pytest

import charlie_alpha.forge_training as forge_training
import charlie_alpha.routed_inference as routed_inference
from charlie_alpha.config import ProjectConfig
from charlie_alpha.forge_data import (
    _allocate,
    _protect,
    _restore,
    _selection_metrics,
    _selective_target_indices,
    _smooth_category_schedule,
    _translated_pair,
)
from charlie_alpha.forge_router import route_uses_adapter
from charlie_alpha.forge_training import ForgeDataset, forge_iterate_batches
from charlie_alpha.routed_inference import (
    DynamicLoraRouter,
    classify_prompt,
    resolve_adapter_path,
)


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


def test_short_pilot_warmup_never_wastes_an_optimizer_update() -> None:
    schedule = forge_training._schedule(1.0e-5, updates=8, warmup_fraction=0.03)
    assert float(schedule(mx.array(0))) == pytest.approx(1.0e-5)


def test_adapter_calibration_scales_only_lora_b() -> None:
    weights = {
        "layer.lora_a": mx.array([1.0, 2.0]),
        "layer.lora_b": mx.array([3.0, 4.0]),
    }
    scaled = forge_training._scale_lora_delta(weights, 0.22)
    assert np.allclose(np.asarray(scaled["layer.lora_a"]), [1.0, 2.0])
    assert np.allclose(np.asarray(scaled["layer.lora_b"]), [0.66, 0.88])
    with pytest.raises(ValueError):
        forge_training._scale_lora_delta(weights, 0.0)


def test_sparse_router_is_fixed_by_domain_and_language() -> None:
    settings = {
        "route": {
            "adapter_domains": ["code"],
            "adapter_languages": ["zh_Hans", "zh_Hant"],
        }
    }
    assert route_uses_adapter({"domain": "code", "language": "en"}, settings)
    assert route_uses_adapter({"domain": "math", "language": "zh_Hant"}, settings)
    assert not route_uses_adapter({"domain": "math", "language": "en"}, settings)


def test_runtime_router_is_high_precision_and_overrideable() -> None:
    assert classify_prompt("請用繁體中文解這題").route == "adapter"
    assert classify_prompt("Implement this algorithm in C++.").route == "adapter"
    assert classify_prompt("Find the derivative of this function.").route == "base"
    assert classify_prompt("Find 2 + 2.", override="adapter").route == "adapter"
    with pytest.raises(ValueError):
        classify_prompt("anything", override="unknown")


def test_dynamic_lora_router_changes_only_adapter_scales() -> None:
    class FakeLora:
        def __init__(self, scale):
            self.scale = scale
            self.lora_a = mx.zeros((3, 2))
            self.lora_b = mx.zeros((2, 4))

    first = FakeLora(20.0)
    second = FakeLora(10.0)
    plain = object()

    class FakeModel:
        def named_modules(self):
            return [("first", first), ("plain", plain), ("second", second)]

    router = DynamicLoraRouter(FakeModel())
    assert router.module_count == 2
    assert router.adapter_parameter_count == 28
    assert router.set_route("base")
    assert (first.scale, second.scale) == (0.0, 0.0)
    assert not router.set_route("base")
    assert router.set_route("adapter")
    assert (first.scale, second.scale) == (20.0, 10.0)


def test_adapter_resolver_accepts_local_and_hub_paths(tmp_path, monkeypatch) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapters.safetensors").write_bytes(b"adapter")
    config = ProjectConfig(
        path=tmp_path / "pipeline.yaml",
        root=tmp_path,
        values={},
        sources={},
    )
    assert resolve_adapter_path(config, adapter) == adapter.resolve()
    monkeypatch.setattr(
        routed_inference,
        "snapshot_download",
        lambda **kwargs: str(adapter),
    )
    assert resolve_adapter_path(config, "owner/model") == adapter


def test_full_training_runs_every_requested_epoch(tmp_path, monkeypatch) -> None:
    artifact_dir = tmp_path / "artifacts"
    final_dir = tmp_path / "final"
    artifact_dir.mkdir()
    final_dir.mkdir()
    (artifact_dir / "pilot-selected.json").write_text(
        json.dumps({"candidate": "winner"}), encoding="utf-8"
    )
    (final_dir / "train.jsonl").write_text("{}\n{}\n{}\n", encoding="utf-8")
    config = ProjectConfig(
        path=tmp_path / "pipeline.yaml",
        root=tmp_path,
        values={
            "paths": {"artifact_dir": "artifacts", "final_dir": "final"},
            "training_v2": {
                "candidates": [{"name": "winner"}],
                "full_epochs": 2,
                "max_seconds": 60,
                "early_stop_evaluations": 2,
            },
        },
        sources={},
    )
    captured = {}

    def fake_train(*args, **kwargs):
        captured.update(kwargs)
        return {"microsteps": kwargs["microsteps"]}

    monkeypatch.setattr(forge_training, "_train_candidate", fake_train)
    monkeypatch.setattr(forge_training, "_start_caffeinate", lambda: None)
    result = forge_training.run_forge_training(config)
    assert captured["microsteps"] == 6
    assert result["microsteps"] == 6


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

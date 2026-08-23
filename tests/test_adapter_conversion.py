import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from charlie_alpha.adapter_conversion import (
    convert_mlx_adapter_to_peft,
    verify_adapter_equivalence,
)


def test_adapter_mapping_is_numerically_equivalent(tmp_path: Path) -> None:
    source = tmp_path / "mlx"
    target = tmp_path / "peft"
    source.mkdir()
    config = {
        "lora_parameters": {
            "rank": 2,
            "scale": 3.0,
            "dropout": 0.0,
            "keys": ["self_attn.q_proj", "self_attn.v_proj"],
        }
    }
    (source / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    random = np.random.default_rng(42)
    weights = {}
    for module in ("q_proj", "v_proj"):
        prefix = f"model.layers.30.self_attn.{module}"
        weights[f"{prefix}.lora_a"] = random.standard_normal((4, 2), dtype=np.float32)
        weights[f"{prefix}.lora_b"] = random.standard_normal((2, 3), dtype=np.float32)
    save_file(weights, source / "adapters.safetensors")

    report = convert_mlx_adapter_to_peft(
        source,
        target,
        base_repo="Qwen/test",
        base_revision="a" * 40,
    )
    equivalence = verify_adapter_equivalence(source, target)
    peft_config = json.loads((target / "adapter_config.json").read_text(encoding="utf-8"))

    assert report["tensor_count"] == 4
    assert peft_config["lora_alpha"] == 6.0
    assert peft_config["target_modules"] == ["q_proj", "v_proj"]
    assert equivalence["passed"]
    assert equivalence["max_abs_error"] <= 1e-5

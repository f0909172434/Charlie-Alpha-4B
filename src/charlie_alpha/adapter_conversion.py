from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

from .io_utils import write_json

_KEY_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)\.lora_([ab])$")


def convert_mlx_adapter_to_peft(
    mlx_adapter_dir: Path,
    peft_output_dir: Path,
    *,
    base_repo: str,
    base_revision: str,
) -> dict[str, Any]:
    mlx_config = json.loads((mlx_adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    parameters = mlx_config["lora_parameters"]
    rank = int(parameters["rank"])
    scale = float(parameters["scale"])
    source = load_file(mlx_adapter_dir / "adapters.safetensors")
    converted: dict[str, np.ndarray] = {}
    mapped: list[dict[str, Any]] = []

    for key, value in source.items():
        match = _KEY_RE.match(key)
        if match is None:
            raise ValueError(f"Unsupported MLX adapter key: {key}")
        layer, module, side = match.groups()
        peft_side = "A" if side == "a" else "B"
        peft_key = f"base_model.model.model.layers.{layer}.{module}.lora_{peft_side}.weight"
        converted_value = np.ascontiguousarray(value.T)
        converted[peft_key] = converted_value
        mapped.append(
            {
                "mlx_key": key,
                "peft_key": peft_key,
                "mlx_shape": list(value.shape),
                "peft_shape": list(converted_value.shape),
            }
        )

    a_modules = {key.rsplit(".lora_A.weight", 1)[0] for key in converted if ".lora_A.weight" in key}
    b_modules = {key.rsplit(".lora_B.weight", 1)[0] for key in converted if ".lora_B.weight" in key}
    if a_modules != b_modules or not a_modules:
        raise ValueError("Every converted module must have matching A and B matrices")

    peft_output_dir.mkdir(parents=True, exist_ok=True)
    save_file(converted, peft_output_dir / "adapter_model.safetensors")
    target_modules = sorted({key.split(".")[-1] for key in parameters["keys"]})
    layer_ids = sorted({int(item["mlx_key"].split(".")[2]) for item in mapped})
    peft_alpha = scale * rank
    peft_config = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": base_repo,
        "bias": "none",
        "corda_config": None,
        "eva_config": None,
        "exclude_modules": None,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": "layers",
        "layers_to_transform": layer_ids,
        "loftq_config": {},
        "lora_alpha": int(peft_alpha) if peft_alpha.is_integer() else peft_alpha,
        "lora_bias": False,
        "lora_dropout": float(parameters["dropout"]),
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "qalora_group_size": 16,
        "r": rank,
        "rank_pattern": {},
        "revision": base_revision,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "trainable_token_indices": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
    }
    write_json(peft_output_dir / "adapter_config.json", peft_config)
    report = {
        "rank": rank,
        "mlx_scale": scale,
        "peft_alpha": peft_alpha,
        "target_modules": target_modules,
        "layer_ids": layer_ids,
        "tensor_count": len(converted),
        "mapped": mapped,
    }
    write_json(peft_output_dir / "mapping-report.json", report)
    return report


def verify_adapter_equivalence(
    mlx_adapter_dir: Path,
    peft_adapter_dir: Path,
    *,
    seed: int = 42,
    atol: float = 1e-5,
) -> dict[str, Any]:
    mlx_config = json.loads((mlx_adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    parameters = mlx_config["lora_parameters"]
    rank = int(parameters["rank"])
    mlx_scale = float(parameters["scale"])
    peft_config = json.loads((peft_adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    peft_scale = float(peft_config["lora_alpha"]) / rank
    mlx_weights = load_file(mlx_adapter_dir / "adapters.safetensors")
    peft_weights = load_file(peft_adapter_dir / "adapter_model.safetensors")
    random = np.random.default_rng(seed)
    maximum_error = 0.0
    tested = 0

    for a_key, mlx_a in mlx_weights.items():
        if not a_key.endswith(".lora_a"):
            continue
        b_key = a_key[:-1] + "b"
        mlx_b = mlx_weights[b_key]
        match = _KEY_RE.match(a_key)
        if match is None:
            raise ValueError(f"Unsupported MLX adapter key: {a_key}")
        layer, module, _ = match.groups()
        prefix = f"base_model.model.model.layers.{layer}.{module}"
        peft_a = peft_weights[f"{prefix}.lora_A.weight"]
        peft_b = peft_weights[f"{prefix}.lora_B.weight"]
        inputs = random.standard_normal((3, mlx_a.shape[0]), dtype=np.float32)
        mlx_delta = (inputs @ mlx_a.astype(np.float32) @ mlx_b.astype(np.float32)) * mlx_scale
        peft_delta = (
            inputs @ peft_a.astype(np.float32).T @ peft_b.astype(np.float32).T
        ) * peft_scale
        error = float(np.max(np.abs(mlx_delta - peft_delta)))
        maximum_error = max(maximum_error, error)
        tested += 1
    if tested == 0 or maximum_error > atol:
        raise ValueError(
            f"Adapter conversion equivalence failed: tested={tested}, max_error={maximum_error}"
        )
    return {"passed": True, "tested_modules": tested, "max_abs_error": maximum_error, "atol": atol}

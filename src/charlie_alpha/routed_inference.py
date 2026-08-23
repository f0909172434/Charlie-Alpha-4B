from __future__ import annotations

import gc
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from .config import ProjectConfig
from .io_utils import sha256_file, write_json
from .training import _base_snapshot

RouteName = Literal["base", "adapter"]

_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_CODE_RE = re.compile(
    r"(?:```|\bpython\b|\bc\+\+\b|\bcpp\b|\bcodeforces\b|\bleetcode\b|"
    r"\bdebug(?:ging)?\b|\balgorithm\b|\bdata\s+structure\b|"
    r"\b(?:time|space)\s+complexity\b|\bimplement\b|\bwrite\s+(?:a\s+)?"
    r"(?:function|program|code)\b|\bdef\s+[A-Za-z_]\w*\s*\(|#include\s*<|std::)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class RouteDecision:
    route: RouteName
    reason: str


def classify_prompt(text: str, override: str = "auto") -> RouteDecision:
    """Choose the fixed sparse route without inspecting a candidate answer."""
    normalized = override.strip().lower()
    if normalized in {"base", "adapter"}:
        return RouteDecision(normalized, "explicit override")  # type: ignore[arg-type]
    if normalized != "auto":
        raise ValueError("route must be auto, base, or adapter")
    if _HAN_RE.search(text):
        return RouteDecision("adapter", "Chinese prompt")
    if _CODE_RE.search(text):
        return RouteDecision("adapter", "coding prompt")
    return RouteDecision("base", "English non-coding fallback")


class DynamicLoraRouter:
    """Enable or bypass one loaded MLX LoRA adapter by changing only its scales."""

    def __init__(self, model: Any) -> None:
        modules: list[tuple[str, Any, float]] = []
        for name, module in model.named_modules():
            if hasattr(module, "lora_a") and hasattr(module, "lora_b"):
                modules.append((name, module, float(module.scale)))
        if not modules:
            raise ValueError("The loaded model does not contain a LoRA adapter")
        self._modules = modules
        self._route: RouteName = "adapter"

    @property
    def route(self) -> RouteName:
        return self._route

    @property
    def module_count(self) -> int:
        return len(self._modules)

    @property
    def adapter_parameter_count(self) -> int:
        return sum(int(module.lora_a.size + module.lora_b.size) for _, module, _ in self._modules)

    @property
    def module_names(self) -> list[str]:
        return [name for name, _, _ in self._modules]

    def set_route(self, route: RouteName) -> bool:
        if route not in {"base", "adapter"}:
            raise ValueError("route must be base or adapter")
        changed = route != self._route
        if changed:
            for _, module, adapter_scale in self._modules:
                module.scale = adapter_scale if route == "adapter" else 0.0
            self._route = route
        return changed


def _selected_adapter(config: ProjectConfig) -> Path:
    selected_path = config.path_for("artifact_dir") / "selected.json"
    if not selected_path.exists():
        raise RuntimeError("No selected adapter; run Forge training and calibration first")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    adapter_path = Path(selected["adapter_path"])
    if not (adapter_path / "adapters.safetensors").exists():
        raise RuntimeError(f"Adapter weights are missing: {adapter_path}")
    return adapter_path


def load_routed_model(config: ProjectConfig) -> tuple[Any, Any, DynamicLoraRouter]:
    if config.section("project").get("profile") != "forge-overnight":
        raise ValueError("Dynamic sparse routing is available only for the Forge profile")
    model, tokenizer = load(
        _base_snapshot(config),
        adapter_path=str(_selected_adapter(config)),
        tokenizer_config={"trust_remote_code": True},
    )
    return model, tokenizer, DynamicLoraRouter(model)


def route_for_messages(messages: Sequence[dict[str, Any]], override: str = "auto") -> RouteDecision:
    user_text = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "user"
    )
    return classify_prompt(user_text, override=override)


def generate_routed(
    model: Any,
    tokenizer: Any,
    router: DynamicLoraRouter,
    messages: Sequence[dict[str, Any]],
    *,
    route: str = "auto",
    max_tokens: int = 1024,
    temperature: float = 0.2,
    top_p: float = 0.8,
) -> tuple[str, RouteDecision]:
    decision = route_for_messages(messages, override=route)
    router.set_route(decision.route)
    prompt = tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=True
    )
    answer = generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature, top_p=top_p),
        verbose=False,
    )
    return answer, decision


def verify_dynamic_router(config: ProjectConfig) -> dict[str, Any]:
    """Prove that bypass equals the pinned base and restore equals the adapter."""
    adapter_path = _selected_adapter(config)
    model_path = _base_snapshot(config)
    model, tokenizer = load(
        model_path,
        adapter_path=str(adapter_path),
        tokenizer_config={"trust_remote_code": True},
    )
    router = DynamicLoraRouter(model)
    tokens = mx.array([tokenizer.encode("Solve: 17 + 25 =")])

    router.set_route("adapter")
    adapter_logits = model(tokens)[:, -1, :]
    mx.eval(adapter_logits)
    router.set_route("base")
    bypass_logits = model(tokens)[:, -1, :]
    mx.eval(bypass_logits)
    router.set_route("adapter")
    restored_logits = model(tokens)[:, -1, :]
    mx.eval(restored_logits)
    adapter_effect = float(mx.max(mx.abs(adapter_logits - bypass_logits)).item())
    restore_error = float(mx.max(mx.abs(adapter_logits - restored_logits)).item())
    module_count = router.module_count
    adapter_parameters = router.adapter_parameter_count
    module_names = router.module_names

    del model, router
    gc.collect()
    mx.clear_cache()
    base_model, _ = load(model_path, tokenizer_config={"trust_remote_code": True})
    base_logits = base_model(tokens)[:, -1, :]
    mx.eval(base_logits)
    bypass_error = float(mx.max(mx.abs(bypass_logits - base_logits)).item())
    del base_model
    gc.collect()
    mx.clear_cache()

    passed = adapter_effect > 0.0 and restore_error == 0.0 and bypass_error == 0.0
    result = {
        "schema_version": 1,
        "passed": passed,
        "mechanism": "single loaded 4B model with eight runtime-switchable LoRA scales",
        "lora_modules": module_count,
        "adapter_parameters": adapter_parameters,
        "adapter_sha256": sha256_file(adapter_path / "adapters.safetensors"),
        "adapter_vs_bypass_max_abs_logit": adapter_effect,
        "adapter_restore_max_abs_logit_error": restore_error,
        "bypass_vs_independent_base_max_abs_logit_error": bypass_error,
        "module_names": module_names,
    }
    report_path = config.root / "reports" / "v2" / "dynamic-router.json"
    write_json(report_path, result)
    return result

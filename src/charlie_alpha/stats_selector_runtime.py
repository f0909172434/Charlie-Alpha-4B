from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .config import ProjectConfig
from .io_utils import canonical_hash, sha256_file, write_json
from .stats_agent import StatsAgent
from .stats_catalog_grounding import _messages
from .stats_eval import _json_from_answer
from .stats_representation_probe import _METHOD_IDS, _probe_scores, _representation_prompt
from .stats_selector_head import _load_head

_RUNTIME_VERSION = 1


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "selector-runtime-v1"


def _runtime_case(question: str) -> dict[str, Any]:
    return {
        "case_id": f"runtime-{canonical_hash(question)[:16]}",
        "family_id": "runtime",
        "question": question,
    }


def _normalize_columns(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, (str, int, float))]


def _hidden_vector(agent: StatsAgent, case: dict[str, Any]) -> np.ndarray:
    language_model = getattr(agent.model, "language_model", None)
    trunk = getattr(language_model, "model", None)
    if trunk is None:
        raise RuntimeError("selector runtime requires direct access to the Qwen3.5 text trunk")
    token_ids = _representation_prompt(agent.tokenizer, case, grounded=False)
    tokens = mx.array([token_ids], dtype=mx.int32)
    hidden = trunk(tokens)[0, -1, :].astype(mx.float32)
    mx.eval(hidden)
    vector = np.asarray(hidden, dtype=np.float64)[None, :]
    del tokens, hidden
    return vector


def _rank_methods(head: dict[str, Any], vector: np.ndarray) -> tuple[str, list[str], float]:
    scores = _probe_scores(head, vector)[0]
    order = np.argsort(scores)[::-1]
    finite = [int(index) for index in order if np.isfinite(scores[int(index)])]
    if not finite:
        raise RuntimeError("selector runtime head exposes no observed methods")
    top = finite[:3]
    selected = _METHOD_IDS[top[0]]
    top3 = [_METHOD_IDS[index] for index in top]
    margin = float(scores[top[0]] - scores[top[1]]) if len(top) > 1 else float("inf")
    return selected, top3, margin


def _implementation_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "stats_selector_runtime.py": sha256_file(Path(__file__)),
        "stats_agent.py": sha256_file(root / "stats_agent.py"),
    }


def freeze_selector_runtime(config: ProjectConfig) -> dict[str, Any]:
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "contract.json"
    public_path = config.root / "reports" / "evolve" / "selector-runtime-v1-contract.json"

    h14_contract_path = config.root / "reports" / "evolve" / "selector-head-v1-contract.json"
    h14_report_path = config.root / "reports" / "evolve" / "selector-head-v1-confirmation.json"
    h14_contract = json.loads(h14_contract_path.read_text(encoding="utf-8"))
    h14_report = json.loads(h14_report_path.read_text(encoding="utf-8"))
    if not h14_report.get("selector_head_architecture_confirmed"):
        raise RuntimeError("selector runtime requires the confirmed H14 architecture")
    if h14_report.get("contract_fingerprint") != h14_contract.get("fingerprint"):
        raise RuntimeError("H14 contract/report fingerprint mismatch")

    head_path = Path(str(h14_contract["selector_head"]["artifact_path"]))
    if sha256_file(head_path) != str(h14_contract["selector_head"]["artifact_sha256"]):
        raise RuntimeError("H14 selector-head artifact changed")
    parent_path = Path(str(h14_contract["parent"]["adapter_path"]))
    if sha256_file(parent_path / "adapters.safetensors") != str(
        h14_contract["parent"]["adapter_sha256"]
    ):
        raise RuntimeError("H14 parent adapter changed")

    observed = list(h14_contract["selector_head"]["observed_methods"])
    missing = sorted(set(_METHOD_IDS) - set(observed))
    contract: dict[str, Any] = {
        "schema_version": 1,
        "runtime": "selector-runtime-v1",
        "runtime_version": _RUNTIME_VERSION,
        "source_h14_result_fingerprint": h14_report["result_fingerprint"],
        "source_h14_contract_fingerprint": h14_contract["fingerprint"],
        "source_h14_report_sha256": sha256_file(h14_report_path),
        "source_h14_contract_sha256": sha256_file(h14_contract_path),
        "parent": h14_contract["parent"],
        "selector_head": h14_contract["selector_head"],
        "prompt_contract": (
            "exact H14 menu-free canonical methods+columns extraction prompt; thinking disabled"
        ),
        "output_contract": {
            "methods": "one canonical method ID from the frozen observed-method head",
            "columns": "columns parsed from the same parent model's menu-free JSON generation",
        },
        "coverage": {
            "catalog_method_count": len(_METHOD_IDS),
            "observed_method_count": len(observed),
            "missing_methods": missing,
            "policy": (
                "missing methods are not silently added or guessed; broad 28-method deployment "
                "claims remain unauthorized"
            ),
        },
        "diagnostics": {
            "top3_methods": "ranking only, not a calibrated probability",
            "score_margin": "linear-score gap only, not confidence",
        },
        "implementation_sha256": _implementation_manifest(),
        "claim_boundary": (
            "This freezes the operational form of the confirmed synthetic H14 mechanism. It does "
            "not create new evidence, change v0.3 weights, promote a champion, or authorize "
            "release."
        ),
    }
    contract["fingerprint"] = canonical_hash(contract)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("selector runtime contract is immutable")
        write_json(public_path, existing)
        return existing
    write_json(lock_path, contract)
    write_json(public_path, contract)
    return contract


def verify_selector_runtime(config: ProjectConfig) -> dict[str, Any]:
    contract = freeze_selector_runtime(config)
    implementation = _implementation_manifest()
    implementation_checks = {
        name: implementation.get(name) == digest
        for name, digest in contract["implementation_sha256"].items()
    }
    h14_contract_path = config.root / "reports" / "evolve" / "selector-head-v1-contract.json"
    h14_report_path = config.root / "reports" / "evolve" / "selector-head-v1-confirmation.json"
    checks = {
        "implementation": all(implementation_checks.values()),
        "h14_contract": sha256_file(h14_contract_path)
        == contract["source_h14_contract_sha256"],
        "h14_report": sha256_file(h14_report_path) == contract["source_h14_report_sha256"],
        "head_artifact": sha256_file(Path(str(contract["selector_head"]["artifact_path"])))
        == contract["selector_head"]["artifact_sha256"],
        "parent_adapter": sha256_file(
            Path(str(contract["parent"]["adapter_path"])) / "adapters.safetensors"
        )
        == contract["parent"]["adapter_sha256"],
    }
    report = {
        "schema_version": 1,
        "complete": True,
        "runtime_contract_fingerprint": contract["fingerprint"],
        "checks": checks,
        "implementation_checks": implementation_checks,
        "passed": all(checks.values()),
        "model_smoke_started": False,
    }
    report["fingerprint"] = canonical_hash(report)
    write_json(_root(config) / "verification.json", report)
    write_json(config.root / "reports" / "evolve" / "selector-runtime-v1-verification.json", report)
    return report


def predict_selector_runtime(config: ProjectConfig, question: str) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("selector runtime question must not be empty")
    contract = freeze_selector_runtime(config)
    head_contract_path = config.root / "reports" / "evolve" / "selector-head-v1-contract.json"
    head_contract = json.loads(head_contract_path.read_text(encoding="utf-8"))
    head = _load_head(head_contract)
    case = _runtime_case(question)
    agent = StatsAgent(config, adapter_path=str(contract["parent"]["adapter_path"]))
    agent.router.set_route("adapter")
    try:
        raw_answer = agent.answer_without_tools(
            _messages(case, grounded=False),
            route="stats",
            max_tokens=160,
            temperature=0.0,
        )
        parsed = _json_from_answer(raw_answer)
        columns = _normalize_columns(parsed.get("columns"))
        vector = _hidden_vector(agent, case)
        method, top3, margin = _rank_methods(head, vector)
    finally:
        del agent
        gc.collect()
        mx.clear_cache()

    return {
        "schema_version": 1,
        "runtime_contract_fingerprint": contract["fingerprint"],
        "question_fingerprint": canonical_hash(question),
        "result": {"methods": [method], "columns": columns},
        "diagnostics": {
            "top3_methods": top3,
            "score_margin": margin,
            "score_margin_is_calibrated_confidence": False,
            "model_json_was_parseable": bool(parsed),
            "method_source": "frozen-h14-selector-head",
            "column_source": "same-parent-menu-free-generation",
        },
    }

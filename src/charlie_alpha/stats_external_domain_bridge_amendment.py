from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import stats_external_domain_bridge as bridge
from .config import ProjectConfig
from .io_utils import canonical_hash, sha256_file, write_json
from .stats_selector_sufficiency import _centered_normalize

_AMENDMENT_VERSION = 1


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "external-domain-bridge-v1"


def _fixed_augmented_bank(base_bank: dict[str, Any], vectors: np.ndarray) -> dict[str, Any]:
    combined = np.concatenate(
        [
            np.asarray(base_bank["vectors"], dtype=np.float64),
            np.asarray(vectors, dtype=np.float64),
        ],
        axis=0,
    )
    center = np.asarray(base_bank["center"], dtype=np.float64)
    return {
        "vectors": combined,
        "center": center,
        "normalized": _centered_normalize(combined, center),
    }


def prepare_external_domain_bridge_execution_amendment(
    config: ProjectConfig,
) -> dict[str, Any]:
    contract = bridge.prepare_external_domain_bridge_contract(config)
    data = bridge.prepare_external_domain_bridge_data(config)
    root = _root(config)
    lock_path = root / "execution-amendment-v1.json"
    public_path = (
        config.root / "reports" / "evolve" / "external-domain-bridge-v1-execution-amendment.json"
    )
    representation_path = root / "representations" / "historical-menu-free.npz"
    if not representation_path.exists():
        raise RuntimeError("H17 amendment requires the preserved pre-failure representations")
    training_path = root / "training.json"
    if training_path.exists():
        existing_training = json.loads(training_path.read_text(encoding="utf-8"))
        if (
            existing_training.get("execution_amendment", {}).get("fingerprint")
            and lock_path.exists()
        ):
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
            write_json(public_path, existing)
            return existing
        raise RuntimeError("H17 amendment cannot change an existing unamended training result")
    amendment: dict[str, Any] = {
        "schema_version": 1,
        "method": "H17 execution-only normalized-bank reconstruction amendment",
        "amendment_version": _AMENDMENT_VERSION,
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["data_fingerprint"],
        "preserved_representation_path": str(representation_path),
        "preserved_representation_sha256": sha256_file(representation_path),
        "failure_boundary": {
            "representations_opened": True,
            "candidate_grid_persisted": False,
            "training_report_exists": False,
            "fresh_external_evaluation_opened": False,
            "observed_error": "KeyError: normalized",
        },
        "permitted_change": (
            "Reconstruct the deterministic centered-normalized view of the augmented support "
            "vectors before support scoring. The frozen vectors, center, residual training, "
            "hyperparameter grid, folds, labels, control outcomes, and gates are unchanged."
        ),
        "implementation_sha256": sha256_file(Path(__file__)),
        "scientific_contract_changed": False,
    }
    amendment["fingerprint"] = canonical_hash(amendment)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != amendment["fingerprint"]:
            raise RuntimeError("H17 execution amendment is immutable")
        write_json(public_path, existing)
        return existing
    write_json(lock_path, amendment)
    write_json(public_path, amendment)
    return amendment


def run_external_domain_bridge_training_amended(config: ProjectConfig) -> dict[str, Any]:
    amendment = prepare_external_domain_bridge_execution_amendment(config)
    original = bridge._augmented_bank
    bridge._augmented_bank = _fixed_augmented_bank
    try:
        result = bridge.run_external_domain_bridge_training(config)
    finally:
        bridge._augmented_bank = original
    if result.get("execution_amendment", {}).get("fingerprint") == amendment["fingerprint"]:
        return result
    amended = dict(result)
    amended["pre_amendment_result_fingerprint"] = result["result_fingerprint"]
    amended["execution_amendment"] = {
        "fingerprint": amendment["fingerprint"],
        "implementation_sha256": amendment["implementation_sha256"],
        "scientific_contract_changed": False,
    }
    amended.pop("result_fingerprint", None)
    amended["result_fingerprint"] = canonical_hash(amended)
    write_json(_root(config) / "training.json", amended)
    public_path = (
        config.root / "reports" / "evolve" / "external-domain-bridge-v1-training.json"
    )
    write_json(public_path, amended)
    return amended

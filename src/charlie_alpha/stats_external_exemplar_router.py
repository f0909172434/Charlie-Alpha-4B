from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .config import ProjectConfig
from .io_utils import canonical_hash, sha256_file, write_json
from .stats_external_domain_bridge import _historical_rows
from .stats_representation_probe import _METHOD_IDS, _load_representations, _probe_scores
from .stats_selector_head import _load_head
from .stats_selector_sufficiency import _load_training_bank, _support_scores

_ROUTER_VERSION = 1


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "external-exemplar-router-v1"


def _implementation_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "stats_external_exemplar_router.py": sha256_file(Path(__file__)),
        "stats_external_domain_bridge.py": sha256_file(
            root / "stats_external_domain_bridge.py"
        ),
        "stats_selector_sufficiency.py": sha256_file(root / "stats_selector_sufficiency.py"),
        "stats_selector_head.py": sha256_file(root / "stats_selector_head.py"),
    }


def prepare_external_exemplar_router_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("external_exemplar_router"))
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "contract.json"
    public_path = config.root / "reports" / "evolve" / "external-exemplar-router-v1-contract.json"

    h17_path = config.root / "reports" / "evolve" / "external-domain-bridge-v1-training.json"
    h17 = json.loads(h17_path.read_text(encoding="utf-8"))
    if h17.get("historical_cross_source_supported") is not False:
        raise RuntimeError("H18 requires the terminal-negative H17 linear bridge branch")
    if h17.get("artifact") is not None:
        raise RuntimeError("H18 requires H17 to have produced no deployable artifact")
    representation_path = (
        config.path_for("artifact_dir")
        / "external-domain-bridge-v1"
        / "representations"
        / "historical-menu-free.npz"
    )
    rows = _historical_rows(config)
    vectors, labels, case_ids = _load_representations(representation_path)
    if case_ids != [str(row["case_id"]) for row in rows] or len(labels) != len(rows):
        raise RuntimeError("H18 historical representation alignment changed")

    h14_path = config.root / "reports" / "evolve" / "selector-head-v1-contract.json"
    h14 = json.loads(h14_path.read_text(encoding="utf-8"))
    h16_path = config.root / "reports" / "evolve" / "selector-sufficiency-v1-confirmation.json"
    h16 = json.loads(h16_path.read_text(encoding="utf-8"))
    bank_path = config.path_for("artifact_dir") / "selector-sufficiency-v1" / "training-bank.npz"
    synthetic: dict[str, Any] = {}
    for style in ("audit", "researcher", "vignette", "conventional", "partial"):
        path = (
            config.path_for("artifact_dir")
            / "selector-sufficiency-v1"
            / "representations"
            / "confirmation_shard"
            / f"{style}.npz"
        )
        synthetic[style] = {"path": str(path), "sha256": sha256_file(path)}

    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": "H18 historical source-held-out external exemplar router",
        "method_version": int(settings["method_version"]),
        "causal_question": (
            "After the H17 linear bridge failed, can nonparametric method voting among other "
            "historical external sources safely rescue any H16-rejected cases while preserving "
            "the menu-free control and H16's synthetic sufficiency behavior?"
        ),
        "h17_negative_result_fingerprint": h17["result_fingerprint"],
        "h17_report_sha256": sha256_file(h17_path),
        "historical_case_count": len(rows),
        "historical_case_fingerprint": canonical_hash(rows),
        "historical_representation_path": str(representation_path),
        "historical_representation_sha256": sha256_file(representation_path),
        "historical_source_counts": dict(
            sorted(Counter(str(row["source_id"]) for row in rows).items())
        ),
        "h14": {
            "contract_fingerprint": h14["fingerprint"],
            "contract_sha256": sha256_file(h14_path),
            "head_path": h14["selector_head"]["artifact_path"],
            "head_sha256": h14["selector_head"]["artifact_sha256"],
        },
        "h16": {
            "result_fingerprint": h16["result_fingerprint"],
            "report_sha256": sha256_file(h16_path),
            "threshold": float(h16["selected_threshold"]),
            "bank_path": str(bank_path),
            "bank_sha256": sha256_file(bank_path),
        },
        "synthetic_retention_representations": synthetic,
        "settings": settings,
        "routing_order": [
            "frozen H16 support accepts -> frozen H14 head",
            "otherwise external exemplar support and vote margin pass -> exemplar vote",
            "otherwise -> exact frozen menu-free control prediction",
        ],
        "cross_source_protocol": (
            "Each historical source is scored only from exemplars belonging to the other five "
            "sources. Every grid value and gate is frozen before any H18 vote is computed."
        ),
        "claim_boundary": (
            "H18 is historical development evidence only. It cannot reuse E2/E3/E4 as fresh "
            "evidence, replace the champion, authorize release, or establish external capability."
        ),
        "implementation_sha256": _implementation_manifest(),
        "fresh_external_evaluation_opened": False,
    }
    contract["fingerprint"] = canonical_hash(contract)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("H18 contract is immutable")
        write_json(public_path, existing)
        return existing
    write_json(lock_path, contract)
    write_json(public_path, contract)
    return contract


def _normalized(vectors: np.ndarray, *, mode: str, center: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if mode == "h16-centered":
        values = values - np.asarray(center, dtype=np.float64)
    elif mode != "unit":
        raise ValueError(f"Unsupported H18 normalization mode: {mode}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _exemplar_vote(
    train_vectors: np.ndarray,
    train_labels: np.ndarray,
    query_vectors: np.ndarray,
    *,
    mode: str,
    center: np.ndarray,
    neighbors: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = _normalized(train_vectors, mode=mode, center=center)
    query = _normalized(query_vectors, mode=mode, center=center)
    similarities = query @ train.T
    predictions: list[int] = []
    supports: list[float] = []
    margins: list[float] = []
    k = min(int(neighbors), len(train_labels))
    for row in similarities:
        indices = np.argsort(row)[-k:][::-1]
        selected = row[indices]
        weights = np.exp((selected - np.max(selected)) / float(temperature))
        votes = np.zeros(len(_METHOD_IDS), dtype=np.float64)
        for index, weight in zip(indices.tolist(), weights.tolist(), strict=True):
            votes[int(train_labels[index])] += float(weight)
        order = np.argsort(votes)[::-1]
        total = float(np.sum(votes))
        top = float(votes[order[0]]) / total
        second = float(votes[order[1]]) / total if len(order) > 1 else 0.0
        predictions.append(int(order[0]))
        supports.append(float(np.max(row)))
        margins.append(top - second)
    return (
        np.asarray(predictions, dtype=np.int64),
        np.asarray(supports, dtype=np.float64),
        np.asarray(margins, dtype=np.float64),
    )


def _metrics(details: list[dict[str, Any]]) -> dict[str, Any]:
    control_only = sum(
        bool(row["control_correct"]) and not bool(row["candidate_correct"]) for row in details
    )
    candidate_only = sum(
        bool(row["candidate_correct"]) and not bool(row["control_correct"]) for row in details
    )
    routes = Counter(str(row["route"]) for row in details)
    external_sources = {
        str(row["source_id"]) for row in details if row["route"] == "external-exemplar"
    }
    return {
        "count": len(details),
        "control_accuracy": sum(bool(row["control_correct"]) for row in details) / len(details),
        "candidate_accuracy": sum(bool(row["candidate_correct"]) for row in details)
        / len(details),
        "candidate_only": candidate_only,
        "control_only": control_only,
        "net_improvements": candidate_only - control_only,
        "route_counts": dict(sorted(routes.items())),
        "external_exemplar_source_count": len(external_sources),
    }


def _cross_source_details(
    rows: list[dict[str, Any]],
    vectors: np.ndarray,
    labels: np.ndarray,
    h14_head: dict[str, Any],
    h16_bank: dict[str, Any],
    *,
    h16_threshold: float,
    mode: str,
    neighbors: int,
    temperature: float,
    support_threshold: float,
    margin_threshold: float,
) -> list[dict[str, Any]]:
    sources = np.asarray([str(row["source_id"]) for row in rows])
    details: list[dict[str, Any]] = []
    for held_out in sorted(set(sources.tolist())):
        train_mask = sources != held_out
        indices = np.flatnonzero(sources == held_out)
        exemplar_pred, exemplar_support, exemplar_margin = _exemplar_vote(
            vectors[train_mask],
            labels[train_mask],
            vectors[indices],
            mode=mode,
            center=h16_bank["center"],
            neighbors=neighbors,
            temperature=temperature,
        )
        h16_support = _support_scores(h16_bank, vectors[indices])
        h14_pred = np.argmax(_probe_scores(h14_head, vectors[indices]), axis=1)
        for local, row_index in enumerate(indices.tolist()):
            row = rows[row_index]
            if h16_support[local] >= h16_threshold:
                prediction = int(h14_pred[local])
                route = "h16-h14"
            elif (
                exemplar_support[local] >= support_threshold
                and exemplar_margin[local] >= margin_threshold
            ):
                prediction = int(exemplar_pred[local])
                route = "external-exemplar"
            else:
                prediction = -1
                route = "menu-free-control"
            candidate_correct = (
                prediction == int(labels[row_index])
                if prediction >= 0
                else bool(row["control_correct"])
            )
            details.append(
                {
                    "case_id": row["case_id"],
                    "source_id": row["source_id"],
                    "held_out_source": held_out,
                    "gold_method_id": row["gold_method_id"],
                    "control_correct": bool(row["control_correct"]),
                    "candidate_correct": candidate_correct,
                    "route": route,
                    "h16_support": float(h16_support[local]),
                    "exemplar_support": float(exemplar_support[local]),
                    "exemplar_margin": float(exemplar_margin[local]),
                    "exemplar_method_id": _METHOD_IDS[int(exemplar_pred[local])],
                }
            )
    return sorted(details, key=lambda row: str(row["case_id"]))


def _synthetic_safety(
    contract: dict[str, Any],
    external_vectors: np.ndarray,
    external_labels: np.ndarray,
    h16_bank: dict[str, Any],
    *,
    mode: str,
    neighbors: int,
    temperature: float,
    support_threshold: float,
    margin_threshold: float,
) -> dict[str, Any]:
    styles: dict[str, Any] = {}
    for style, fields in contract["synthetic_retention_representations"].items():
        path = Path(fields["path"])
        if sha256_file(path) != fields["sha256"]:
            raise RuntimeError(f"H18 synthetic representation changed: {style}")
        vectors, _, _ = _load_representations(path)
        _, support, margin = _exemplar_vote(
            external_vectors,
            external_labels,
            vectors,
            mode=mode,
            center=h16_bank["center"],
            neighbors=neighbors,
            temperature=temperature,
        )
        h16_support = _support_scores(h16_bank, vectors)
        h16_accept = h16_support >= float(contract["h16"]["threshold"])
        external_accept = (
            (~h16_accept) & (support >= support_threshold) & (margin >= margin_threshold)
        )
        styles[style] = {
            "count": len(vectors),
            "h16_acceptance": float(np.mean(h16_accept)),
            "external_exemplar_acceptance": float(np.mean(external_accept)),
        }
    reduced = ("conventional", "partial")
    return {
        "styles": styles,
        "maximum_reduced_external_exemplar_acceptance": max(
            styles[style]["external_exemplar_acceptance"] for style in reduced
        ),
    }


def _gate(metrics: dict[str, Any], safety: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "historical_net_improvement": int(metrics["net_improvements"])
        >= int(gates["minimum_historical_net_improvements"]),
        "zero_control_only_losses": int(metrics["control_only"])
        <= int(gates["maximum_control_only_losses"]),
        "external_exemplar_count": int(metrics["route_counts"].get("external-exemplar", 0))
        >= int(gates["minimum_external_exemplar_count"]),
        "external_exemplar_sources": int(metrics["external_exemplar_source_count"])
        >= int(gates["minimum_external_exemplar_sources"]),
        "reduced_synthetic_safety": float(
            safety["maximum_reduced_external_exemplar_acceptance"]
        )
        <= float(gates["maximum_reduced_external_exemplar_acceptance"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def run_external_exemplar_router_training(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_external_exemplar_router_contract(config)
    report_path = _root(config) / "training.json"
    public_path = config.root / "reports" / "evolve" / "external-exemplar-router-v1-training.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("contract_fingerprint") != contract["fingerprint"]:
            raise RuntimeError("H18 completed result does not match the frozen contract")
        write_json(public_path, existing)
        return existing

    rows = _historical_rows(config)
    if canonical_hash(rows) != contract["historical_case_fingerprint"]:
        raise RuntimeError("H18 historical rows changed")
    representation_path = Path(contract["historical_representation_path"])
    if sha256_file(representation_path) != contract["historical_representation_sha256"]:
        raise RuntimeError("H18 historical representations changed")
    vectors, labels, case_ids = _load_representations(representation_path)
    if case_ids != [str(row["case_id"]) for row in rows]:
        raise RuntimeError("H18 representation order changed")
    h14_contract = json.loads(
        (config.root / "reports" / "evolve" / "selector-head-v1-contract.json").read_text(
            encoding="utf-8"
        )
    )
    h14_head = _load_head(h14_contract)
    bank_path = Path(contract["h16"]["bank_path"])
    h16_bank = _load_training_bank(bank_path)
    settings = dict(contract["settings"])
    candidates: list[dict[str, Any]] = []
    for mode in settings["normalization_modes"]:
        for neighbors in settings["neighbors"]:
            for temperature in settings["temperatures"]:
                for support_threshold in settings["support_thresholds"]:
                    for margin_threshold in settings["margin_thresholds"]:
                        details = _cross_source_details(
                            rows,
                            vectors,
                            labels,
                            h14_head,
                            h16_bank,
                            h16_threshold=float(contract["h16"]["threshold"]),
                            mode=str(mode),
                            neighbors=int(neighbors),
                            temperature=float(temperature),
                            support_threshold=float(support_threshold),
                            margin_threshold=float(margin_threshold),
                        )
                        metrics = _metrics(details)
                        safety = _synthetic_safety(
                            contract,
                            vectors,
                            labels,
                            h16_bank,
                            mode=str(mode),
                            neighbors=int(neighbors),
                            temperature=float(temperature),
                            support_threshold=float(support_threshold),
                            margin_threshold=float(margin_threshold),
                        )
                        gate = _gate(metrics, safety, dict(settings["gates"]))
                        candidates.append(
                            {
                                "normalization_mode": str(mode),
                                "neighbors": int(neighbors),
                                "temperature": float(temperature),
                                "support_threshold": float(support_threshold),
                                "margin_threshold": float(margin_threshold),
                                "historical_cross_source": metrics,
                                "synthetic_safety": safety,
                                "gate": gate,
                            }
                        )
    eligible = [candidate for candidate in candidates if candidate["gate"]["passed"]]
    eligible.sort(
        key=lambda candidate: (
            -int(candidate["historical_cross_source"]["net_improvements"]),
            -float(candidate["historical_cross_source"]["candidate_accuracy"]),
            int(candidate["historical_cross_source"]["route_counts"].get("external-exemplar", 0)),
            -float(candidate["support_threshold"]),
            -float(candidate["margin_threshold"]),
            int(candidate["neighbors"]),
            float(candidate["temperature"]),
            str(candidate["normalization_mode"]),
        )
    )
    selected = eligible[0] if eligible else None
    artifact: dict[str, Any] | None = None
    if selected is not None:
        artifact_path = _root(config) / "external-exemplars.npz"
        np.savez_compressed(
            artifact_path,
            vectors=np.asarray(vectors, dtype=np.float32),
            labels=np.asarray(labels, dtype=np.int64),
            case_ids=np.asarray(case_ids),
            source_ids=np.asarray([str(row["source_id"]) for row in rows]),
            normalization_mode=np.asarray([selected["normalization_mode"]]),
            neighbors=np.asarray([selected["neighbors"]], dtype=np.int64),
            temperature=np.asarray([selected["temperature"]], dtype=np.float64),
            support_threshold=np.asarray([selected["support_threshold"]], dtype=np.float64),
            margin_threshold=np.asarray([selected["margin_threshold"]], dtype=np.float64),
        )
        artifact = {"path": str(artifact_path), "sha256": sha256_file(artifact_path)}
    result: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "H18 historical source-held-out external exemplar router training",
        "router_version": _ROUTER_VERSION,
        "contract_fingerprint": contract["fingerprint"],
        "grid_candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "selected_candidate": selected,
        "artifact": artifact,
        "historical_cross_source_supported": selected is not None,
        "fresh_external_evidence": False,
        "champion_changed": False,
        "release_authorized": False,
        "next_step": (
            "freeze-h18-runtime-and-qualify-one-genuinely-new-external-source"
            if selected is not None
            else "preserve-h18-negative-and-retain-h16-control-fallback"
        ),
        "claim_boundary": contract["claim_boundary"],
        "candidate_grid": candidates,
    }
    result["result_fingerprint"] = canonical_hash(result)
    write_json(report_path, result)
    write_json(public_path, result)
    return result

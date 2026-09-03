from __future__ import annotations

import gc
import json
from collections import Counter
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_bytes, sha256_file, write_json, write_jsonl
from .stats_agent import StatsAgent
from .stats_family_router import _expert_context
from .stats_representation_probe import (
    _METHOD_IDS,
    _extract_representations,
    _load_representations,
    _normalize_rows,
    _save_representations,
)
from .stats_selector_head import _load_head
from .stats_selector_sufficiency import _load_training_bank, _support_scores

_METHOD_INDEX = {method_id: index for index, method_id in enumerate(_METHOD_IDS)}
_TRAINER_VERSION = 1


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "external-domain-bridge-v1"


def _data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "external-domain-bridge-v1"


def _history_specs(config: ProjectConfig) -> list[dict[str, Any]]:
    return [
        {
            "study": "E2-v2",
            "cases_path": config.root
            / "data"
            / "evolve"
            / "external-catalog-interface-v2"
            / "cases.jsonl",
            "report_path": config.root
            / "artifacts"
            / "evolve"
            / "external-catalog-interface-v2"
            / "report.json",
            "control_arm": "menu-free-control",
            "question_field": "vignette",
            "source_fallback": None,
        },
        {
            "study": "E3",
            "cases_path": config.root
            / "data"
            / "evolve"
            / "selector-external-v1"
            / "cases.jsonl",
            "report_path": config.root
            / "artifacts"
            / "evolve"
            / "selector-external-v1"
            / "report-amended-v1.json",
            "control_arm": "menu-free-control",
            "question_field": "question",
            "source_fallback": "shukla-2025",
        },
        {
            "study": "E4",
            "cases_path": config.root
            / "data"
            / "evolve"
            / "selective-external-v1"
            / "cases.jsonl",
            "report_path": config.root
            / "artifacts"
            / "evolve"
            / "selective-external-v1"
            / "report.json",
            "control_arm": "menu-free-control",
            "question_field": "question",
            "source_fallback": None,
        },
    ]


def _implementation_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "stats_external_domain_bridge.py": sha256_file(Path(__file__)),
        "stats_representation_probe.py": sha256_file(root / "stats_representation_probe.py"),
        "stats_selector_sufficiency.py": sha256_file(root / "stats_selector_sufficiency.py"),
        "stats_selector_head.py": sha256_file(root / "stats_selector_head.py"),
        "stats_agent.py": sha256_file(root / "stats_agent.py"),
    }


def _historical_rows(config: ProjectConfig) -> list[dict[str, Any]]:
    h14_contract = json.loads(
        (config.root / "reports" / "evolve" / "selector-head-v1-contract.json").read_text(
            encoding="utf-8"
        )
    )
    observed = set(str(value) for value in h14_contract["selector_head"]["observed_methods"])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in _history_specs(config):
        report = json.loads(Path(spec["report_path"]).read_text(encoding="utf-8"))
        if not report.get("complete"):
            raise RuntimeError(f"H17 requires complete historical study {spec['study']}")
        controls = {
            str(row["case_id"]): row
            for row in report["private_details"][str(spec["control_arm"])]
        }
        for source_case in read_jsonl(Path(spec["cases_path"])):
            case_id = str(source_case["case_id"])
            gold = source_case.get("gold_method_id")
            eligible = bool(source_case.get("eligible"))
            head_eligible = bool(source_case.get("head_eligible", gold in observed))
            if not eligible or not head_eligible or gold not in observed:
                continue
            if case_id in seen:
                raise RuntimeError(f"H17 duplicate historical case: {case_id}")
            if case_id not in controls:
                raise RuntimeError(f"H17 missing frozen control result: {case_id}")
            control = controls[case_id]
            question = source_case.get(str(spec["question_field"]))
            if not isinstance(question, str) or not question.strip():
                raise RuntimeError(f"H17 missing question text: {case_id}")
            source_id = source_case.get("source_id", spec["source_fallback"])
            if not isinstance(source_id, str) or not source_id:
                raise RuntimeError(f"H17 missing source identity: {case_id}")
            rows.append(
                {
                    "case_id": case_id,
                    "study": str(spec["study"]),
                    "source_id": source_id,
                    "question": question,
                    "gold_method_id": str(gold),
                    "gold_methods": [str(gold)],
                    "control_predicted_method_id": str(
                        control.get("predicted_method_id") or ""
                    ),
                    "control_valid_output": bool(control.get("valid_output")),
                    "control_correct": bool(control.get("correct")),
                }
            )
            seen.add(case_id)
    return rows


def prepare_external_domain_bridge_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("external_domain_bridge"))
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "contract.json"
    public_path = config.root / "reports" / "evolve" / "external-domain-bridge-v1-contract.json"

    e4_path = config.root / "reports" / "evolve" / "selector-sufficiency-external-v1.json"
    e4 = json.loads(e4_path.read_text(encoding="utf-8"))
    if not e4.get("selective_external_safety_supported"):
        raise RuntimeError("H17 requires completed E4 selective safety evidence")
    if int(e4.get("selector_accept_count", -1)) != 0:
        raise RuntimeError("H17 is registered for the E4 zero-coverage branch only")

    h14_contract_path = config.root / "reports" / "evolve" / "selector-head-v1-contract.json"
    h14_contract = json.loads(h14_contract_path.read_text(encoding="utf-8"))
    h16_path = config.root / "reports" / "evolve" / "selector-sufficiency-v1-confirmation.json"
    h16 = json.loads(h16_path.read_text(encoding="utf-8"))
    if not h16.get("selector_sufficiency_guard_confirmed"):
        raise RuntimeError("H17 requires confirmed H16 geometry guard")

    rows = _historical_rows(config)
    source_counts = Counter(str(row["source_id"]) for row in rows)
    minimum_sources = int(settings["minimum_historical_sources"])
    minimum_cases = int(settings["minimum_historical_cases"])
    if len(source_counts) < minimum_sources or len(rows) < minimum_cases:
        raise RuntimeError("H17 historical development pool is below the frozen minimum")

    history: list[dict[str, Any]] = []
    for spec in _history_specs(config):
        report = json.loads(Path(spec["report_path"]).read_text(encoding="utf-8"))
        history.append(
            {
                "study": spec["study"],
                "cases_path": str(spec["cases_path"]),
                "cases_sha256": sha256_file(Path(spec["cases_path"])),
                "report_path": str(spec["report_path"]),
                "report_sha256": sha256_file(Path(spec["report_path"])),
                "result_fingerprint": report["result_fingerprint"],
            }
        )

    bank_path = config.path_for("artifact_dir") / "selector-sufficiency-v1" / "training-bank.npz"
    synthetic_paths: dict[str, dict[str, Any]] = {}
    for shard in ("confirmation_shard",):
        synthetic_paths[shard] = {}
        for style in ("audit", "researcher", "vignette", "conventional", "partial"):
            path = (
                config.path_for("artifact_dir")
                / "selector-sufficiency-v1"
                / "representations"
                / shard
                / f"{style}.npz"
            )
            synthetic_paths[shard][style] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }

    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": "H17 historical multi-source residual domain-bridge training",
        "method_version": int(settings["method_version"]),
        "causal_question": (
            "Can a small residual head trained only on already-opened external sources improve "
            "source-held-out method selection while an augmented H16 support bank preserves every "
            "previously correct menu-free control decision?"
        ),
        "branch_evidence": {
            "e4_result_fingerprint": e4["result_fingerprint"],
            "e4_report_sha256": sha256_file(e4_path),
            "e4_selector_accept_count": e4["selector_accept_count"],
        },
        "historical_development_sources": history,
        "historical_case_count": len(rows),
        "historical_source_counts": dict(sorted(source_counts.items())),
        "historical_case_fingerprint": canonical_hash(rows),
        "base_h14": {
            "contract_fingerprint": h14_contract["fingerprint"],
            "contract_sha256": sha256_file(h14_contract_path),
            "head_path": h14_contract["selector_head"]["artifact_path"],
            "head_sha256": h14_contract["selector_head"]["artifact_sha256"],
            "observed_methods": h14_contract["selector_head"]["observed_methods"],
        },
        "base_h16": {
            "result_fingerprint": h16["result_fingerprint"],
            "report_sha256": sha256_file(h16_path),
            "training_bank_path": str(bank_path),
            "training_bank_sha256": sha256_file(bank_path),
        },
        "synthetic_retention_representations": synthetic_paths,
        "settings": settings,
        "cross_source_protocol": (
            "For each source, train the residual head and support-bank augmentation on every other "
            "historical source, then score only the held-out source. Hyperparameters are selected "
            "from the frozen grid. All E2/E3/E4 rows are development evidence after this point."
        ),
        "adaptation_policy": (
            "No grid, source row, label, control outcome, representation location, or gate may "
            "change after the contract is written."
        ),
        "claim_boundary": (
            "H17 may establish only historical leave-one-source-out development support for a "
            "domain-bridge runtime. It cannot reuse E2, E3, or E4 as fresh evidence, change the "
            "official champion, authorize release, or establish external capability."
        ),
        "implementation_sha256": _implementation_manifest(),
        "representations_opened": False,
        "fresh_external_evaluation_opened": False,
    }
    contract["fingerprint"] = canonical_hash(contract)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("H17 contract is immutable")
        write_json(public_path, existing)
        return existing
    write_json(lock_path, contract)
    write_json(public_path, contract)
    return contract


def prepare_external_domain_bridge_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_external_domain_bridge_contract(config)
    rows = _historical_rows(config)
    if canonical_hash(rows) != contract["historical_case_fingerprint"]:
        raise RuntimeError("H17 historical cases changed after contract lock")
    for source in contract["historical_development_sources"]:
        if sha256_file(Path(source["cases_path"])) != source["cases_sha256"]:
            raise RuntimeError(f"H17 historical cases changed: {source['study']}")
        if sha256_file(Path(source["report_path"])) != source["report_sha256"]:
            raise RuntimeError(f"H17 historical report changed: {source['study']}")
    path = _data_root(config) / "historical-cases.jsonl"
    write_jsonl(path, rows)
    source_counts = Counter(str(row["source_id"]) for row in rows)
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "contract_fingerprint": contract["fingerprint"],
        "historical_cases_sha256": sha256_file(path),
        "historical_case_fingerprint": canonical_hash(rows),
        "case_count": len(rows),
        "source_count": len(source_counts),
        "source_counts": dict(sorted(source_counts.items())),
        "method_counts": dict(
            sorted(Counter(str(row["gold_method_id"]) for row in rows).items())
        ),
        "control_accuracy": sum(bool(row["control_correct"]) for row in rows) / len(rows),
        "fresh_external_evidence": False,
        "representations_opened": False,
    }
    report["data_fingerprint"] = canonical_hash(report)
    write_json(_root(config) / "data.json", report)
    write_json(config.root / "reports" / "evolve" / "external-domain-bridge-v1-data.json", report)
    return report


def _external_representation_path(config: ProjectConfig) -> Path:
    return _root(config) / "representations" / "historical-menu-free.npz"


def _ensure_external_representations(
    config: ProjectConfig,
    rows: list[dict[str, Any]],
) -> Path:
    path = _external_representation_path(config)
    if path.exists():
        vectors, labels, case_ids = _load_representations(path)
        expected_ids = [str(row["case_id"]) for row in rows]
        expected_labels = np.asarray(
            [_METHOD_INDEX[str(row["gold_method_id"])] for row in rows], dtype=np.int64
        )
        if case_ids != expected_ids or not np.array_equal(labels, expected_labels):
            raise RuntimeError("H17 cached historical representations changed")
        if len(vectors) != len(rows):
            raise RuntimeError("H17 cached representation count changed")
        return path
    _, adapter_paths = _expert_context(config)
    agent = StatsAgent(config, adapter_path=adapter_paths["parent"])
    agent.router.set_route("adapter")
    try:
        vectors, labels, case_ids = _extract_representations(agent, rows, grounded=False)
        _save_representations(path, vectors=vectors, labels=labels, case_ids=case_ids)
    finally:
        del agent
        gc.collect()
        mx.clear_cache()
    return path


def _design(vectors: np.ndarray) -> np.ndarray:
    normalized = _normalize_rows(np.asarray(vectors, dtype=np.float64))
    return np.concatenate(
        [normalized, np.ones((len(normalized), 1), dtype=np.float64)], axis=1
    )


def _base_scores(head: dict[str, Any], vectors: np.ndarray) -> np.ndarray:
    return _design(vectors) @ np.asarray(head["weights"], dtype=np.float64)


def _fit_residual(
    head: dict[str, Any],
    vectors: np.ndarray,
    labels: np.ndarray,
    *,
    ridge_lambda: float,
) -> np.ndarray:
    x = _design(vectors)
    base = _base_scores(head, vectors)
    targets = np.zeros_like(base)
    targets[np.arange(len(labels)), labels] = 1.0
    residual_targets = targets - base
    gram = x @ x.T
    gram.flat[:: len(gram) + 1] += float(ridge_lambda)
    alpha = np.linalg.solve(gram, residual_targets)
    return x.T @ alpha


def _candidate_scores(
    head: dict[str, Any],
    residual_weights: np.ndarray,
    vectors: np.ndarray,
    *,
    residual_scale: float,
) -> np.ndarray:
    scores = _base_scores(head, vectors) + float(residual_scale) * (
        _design(vectors) @ np.asarray(residual_weights, dtype=np.float64)
    )
    observed = set(int(value) for value in head["observed"])
    for index in range(scores.shape[1]):
        if index not in observed:
            scores[:, index] = -np.inf
    return scores


def _augmented_bank(base_bank: dict[str, Any], vectors: np.ndarray) -> dict[str, Any]:
    return {
        "vectors": np.concatenate(
            [
                np.asarray(base_bank["vectors"], dtype=np.float64),
                np.asarray(vectors, dtype=np.float64),
            ],
            axis=0,
        ),
        "center": np.asarray(base_bank["center"], dtype=np.float64),
    }


def _selective_metrics(details: list[dict[str, Any]]) -> dict[str, Any]:
    control_only = sum(
        bool(row["control_correct"]) and not bool(row["selective_correct"]) for row in details
    )
    candidate_only = sum(
        bool(row["selective_correct"]) and not bool(row["control_correct"]) for row in details
    )
    both_correct = sum(
        bool(row["selective_correct"]) and bool(row["control_correct"]) for row in details
    )
    accepted_sources = {str(row["source_id"]) for row in details if row["accepted"]}
    return {
        "count": len(details),
        "control_accuracy": sum(bool(row["control_correct"]) for row in details) / len(details),
        "raw_candidate_accuracy": sum(bool(row["raw_candidate_correct"]) for row in details)
        / len(details),
        "selective_accuracy": sum(bool(row["selective_correct"]) for row in details)
        / len(details),
        "accept_count": sum(bool(row["accepted"]) for row in details),
        "accepted_source_count": len(accepted_sources),
        "both_correct": both_correct,
        "candidate_only": candidate_only,
        "control_only": control_only,
        "both_wrong": len(details) - both_correct - candidate_only - control_only,
        "net_improvements": candidate_only - control_only,
    }


def _synthetic_metrics(
    config: ProjectConfig,
    contract: dict[str, Any],
    head: dict[str, Any],
    residual_weights: np.ndarray,
    bank: dict[str, Any],
    *,
    residual_scale: float,
    threshold: float,
) -> dict[str, Any]:
    by_style: dict[str, Any] = {}
    full_styles = ("audit", "researcher", "vignette")
    reduced_styles = ("conventional", "partial")
    for style in (*full_styles, *reduced_styles):
        fields = contract["synthetic_retention_representations"]["confirmation_shard"][style]
        path = Path(fields["path"])
        if sha256_file(path) != fields["sha256"]:
            raise RuntimeError(f"H17 synthetic representation changed: {style}")
        vectors, labels, _ = _load_representations(path)
        base_pred = np.argmax(
            _candidate_scores(
                head,
                np.zeros_like(residual_weights),
                vectors,
                residual_scale=0.0,
            ),
            axis=1,
        )
        candidate_pred = np.argmax(
            _candidate_scores(
                head,
                residual_weights,
                vectors,
                residual_scale=residual_scale,
            ),
            axis=1,
        )
        support = _support_scores(bank, vectors)
        accepted = support >= threshold
        accepted_accuracy = (
            float(np.mean(candidate_pred[accepted] == labels[accepted]))
            if np.any(accepted)
            else 0.0
        )
        by_style[style] = {
            "count": len(labels),
            "base_accuracy": float(np.mean(base_pred == labels)),
            "candidate_accuracy": float(np.mean(candidate_pred == labels)),
            "acceptance": float(np.mean(accepted)),
            "rejection": float(np.mean(~accepted)),
            "accepted_head_accuracy": accepted_accuracy,
        }
    worst_regression = max(
        100.0 * (by_style[style]["base_accuracy"] - by_style[style]["candidate_accuracy"])
        for style in full_styles
    )
    return {
        "styles": by_style,
        "minimum_full_accuracy": min(
            by_style[style]["candidate_accuracy"] for style in full_styles
        ),
        "minimum_full_acceptance": min(by_style[style]["acceptance"] for style in full_styles),
        "minimum_reduced_rejection": min(by_style[style]["rejection"] for style in reduced_styles),
        "minimum_accepted_head_accuracy": min(
            by_style[style]["accepted_head_accuracy"] for style in full_styles
        ),
        "maximum_full_accuracy_regression_points": worst_regression,
    }


def _gate_candidate(
    historical: dict[str, Any], synthetic: dict[str, Any], gates: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "historical_net_improvement": int(historical["net_improvements"])
        >= int(gates["minimum_historical_net_improvements"]),
        "zero_historical_control_only_losses": int(historical["control_only"])
        <= int(gates["maximum_historical_control_only_losses"]),
        "minimum_historical_accept_count": int(historical["accept_count"])
        >= int(gates["minimum_historical_accept_count"]),
        "minimum_historical_accepted_sources": int(historical["accepted_source_count"])
        >= int(gates["minimum_historical_accepted_sources"]),
        "synthetic_full_accuracy": float(synthetic["minimum_full_accuracy"])
        >= float(gates["minimum_synthetic_full_accuracy"]),
        "synthetic_full_accuracy_retention": float(
            synthetic["maximum_full_accuracy_regression_points"]
        )
        <= float(gates["maximum_synthetic_full_accuracy_regression_points"]),
        "synthetic_full_acceptance": float(synthetic["minimum_full_acceptance"])
        >= float(gates["minimum_synthetic_full_acceptance"]),
        "synthetic_reduced_rejection": float(synthetic["minimum_reduced_rejection"])
        >= float(gates["minimum_synthetic_reduced_rejection"]),
        "synthetic_accepted_head_accuracy": float(
            synthetic["minimum_accepted_head_accuracy"]
        )
        >= float(gates["minimum_synthetic_accepted_head_accuracy"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _cross_source_details(
    rows: list[dict[str, Any]],
    vectors: np.ndarray,
    labels: np.ndarray,
    head: dict[str, Any],
    base_bank: dict[str, Any],
    *,
    ridge_lambda: float,
    residual_scale: float,
    threshold: float,
) -> list[dict[str, Any]]:
    sources = np.asarray([str(row["source_id"]) for row in rows])
    details: list[dict[str, Any]] = []
    for held_out in sorted(set(sources.tolist())):
        train_mask = sources != held_out
        test_indices = np.flatnonzero(sources == held_out)
        residual = _fit_residual(
            head,
            vectors[train_mask],
            labels[train_mask],
            ridge_lambda=ridge_lambda,
        )
        bank = _augmented_bank(base_bank, vectors[train_mask])
        scores = _candidate_scores(
            head,
            residual,
            vectors[test_indices],
            residual_scale=residual_scale,
        )
        predictions = np.argmax(scores, axis=1)
        support = _support_scores(bank, vectors[test_indices])
        for local_index, row_index in enumerate(test_indices.tolist()):
            row = rows[row_index]
            accepted = bool(support[local_index] >= threshold)
            raw_correct = int(predictions[local_index]) == int(labels[row_index])
            selective_correct = raw_correct if accepted else bool(row["control_correct"])
            details.append(
                {
                    "case_id": row["case_id"],
                    "source_id": row["source_id"],
                    "held_out_source": held_out,
                    "gold_method_id": row["gold_method_id"],
                    "control_correct": bool(row["control_correct"]),
                    "raw_candidate_method_id": _METHOD_IDS[int(predictions[local_index])],
                    "raw_candidate_correct": raw_correct,
                    "support_score": float(support[local_index]),
                    "accepted": accepted,
                    "selective_correct": selective_correct,
                }
            )
    return sorted(details, key=lambda row: str(row["case_id"]))


def run_external_domain_bridge_training(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_external_domain_bridge_contract(config)
    data = prepare_external_domain_bridge_data(config)
    report_path = _root(config) / "training.json"
    public_path = config.root / "reports" / "evolve" / "external-domain-bridge-v1-training.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("contract_fingerprint") != contract["fingerprint"]:
            raise RuntimeError("H17 completed training does not match the frozen contract")
        write_json(public_path, existing)
        return existing

    rows = list(read_jsonl(_data_root(config) / "historical-cases.jsonl"))
    if canonical_hash(rows) != contract["historical_case_fingerprint"]:
        raise RuntimeError("H17 training cases changed")
    representation_path = _ensure_external_representations(config, rows)
    vectors, labels, case_ids = _load_representations(representation_path)
    if case_ids != [str(row["case_id"]) for row in rows]:
        raise RuntimeError("H17 representation order changed")

    h14_contract = json.loads(
        (config.root / "reports" / "evolve" / "selector-head-v1-contract.json").read_text(
            encoding="utf-8"
        )
    )
    head = _load_head(h14_contract)
    bank_path = Path(contract["base_h16"]["training_bank_path"])
    if sha256_file(bank_path) != contract["base_h16"]["training_bank_sha256"]:
        raise RuntimeError("H17 base H16 bank changed")
    base_bank = _load_training_bank(bank_path)
    settings = dict(contract["settings"])
    candidates: list[dict[str, Any]] = []
    for ridge_lambda in settings["ridge_lambdas"]:
        for residual_scale in settings["residual_scales"]:
            residual = _fit_residual(
                head,
                vectors,
                labels,
                ridge_lambda=float(ridge_lambda),
            )
            final_bank = _augmented_bank(base_bank, vectors)
            for threshold in settings["support_thresholds"]:
                details = _cross_source_details(
                    rows,
                    vectors,
                    labels,
                    head,
                    base_bank,
                    ridge_lambda=float(ridge_lambda),
                    residual_scale=float(residual_scale),
                    threshold=float(threshold),
                )
                historical = _selective_metrics(details)
                synthetic = _synthetic_metrics(
                    config,
                    contract,
                    head,
                    residual,
                    final_bank,
                    residual_scale=float(residual_scale),
                    threshold=float(threshold),
                )
                gate = _gate_candidate(historical, synthetic, dict(settings["gates"]))
                candidates.append(
                    {
                        "ridge_lambda": float(ridge_lambda),
                        "residual_scale": float(residual_scale),
                        "support_threshold": float(threshold),
                        "historical_cross_source": historical,
                        "synthetic_retention": synthetic,
                        "gate": gate,
                        "details": details,
                    }
                )
    eligible = [candidate for candidate in candidates if candidate["gate"]["passed"]]
    eligible.sort(
        key=lambda candidate: (
            -int(candidate["historical_cross_source"]["net_improvements"]),
            -float(candidate["historical_cross_source"]["selective_accuracy"]),
            -int(candidate["historical_cross_source"]["accept_count"]),
            float(candidate["residual_scale"]),
            float(candidate["ridge_lambda"]),
            -float(candidate["support_threshold"]),
        )
    )
    selected = eligible[0] if eligible else None
    artifact: dict[str, Any] | None = None
    if selected is not None:
        residual = _fit_residual(
            head,
            vectors,
            labels,
            ridge_lambda=float(selected["ridge_lambda"]),
        )
        final_bank = _augmented_bank(base_bank, vectors)
        artifact_path = _root(config) / "domain-bridge.npz"
        np.savez_compressed(
            artifact_path,
            residual_weights=np.asarray(residual, dtype=np.float64),
            observed=np.asarray(head["observed"], dtype=np.int64),
            support_vectors=np.asarray(final_bank["vectors"], dtype=np.float32),
            support_center=np.asarray(final_bank["center"], dtype=np.float32),
            ridge_lambda=np.asarray([selected["ridge_lambda"]], dtype=np.float64),
            residual_scale=np.asarray([selected["residual_scale"]], dtype=np.float64),
            support_threshold=np.asarray([selected["support_threshold"]], dtype=np.float64),
        )
        artifact = {
            "path": str(artifact_path),
            "sha256": sha256_file(artifact_path),
            "residual_weight_sha256": sha256_bytes(
                np.asarray(residual, dtype="<f8").tobytes()
            ),
            "support_vector_count": len(final_bank["vectors"]),
        }

    compact_candidates = []
    for candidate in candidates:
        compact_candidates.append(
            {
                key: value
                for key, value in candidate.items()
                if key != "details"
            }
        )
    selected_public = (
        {key: value for key, value in selected.items() if key != "details"}
        if selected is not None
        else None
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "H17 historical multi-source residual domain-bridge training",
        "trainer_version": _TRAINER_VERSION,
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["data_fingerprint"],
        "representation_sha256": sha256_file(representation_path),
        "historical_case_count": len(rows),
        "historical_source_count": len({str(row["source_id"]) for row in rows}),
        "grid_candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "selected_candidate": selected_public,
        "artifact": artifact,
        "historical_cross_source_supported": selected is not None,
        "fresh_external_evidence": False,
        "champion_changed": False,
        "release_authorized": False,
        "next_step": (
            "freeze-h17-runtime-and-qualify-one-genuinely-new-external-source"
            if selected is not None
            else "preserve-h17-negative-and-retain-h16-control-fallback"
        ),
        "claim_boundary": contract["claim_boundary"],
        "candidate_grid": compact_candidates,
    }
    result["result_fingerprint"] = canonical_hash(result)
    write_json(report_path, result)
    write_json(public_path, result)
    return result

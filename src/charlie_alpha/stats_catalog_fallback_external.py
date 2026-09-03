from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .io_utils import (
    canonical_hash,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from .stats_external_catalog import _mcnemar_exact_pvalue
from .stats_guarded_external import (
    _DECODING,
    _AgentCaller,
    _case_fingerprint,
    _complete_strict_progress,
    _load_strict_progress,
    _paired,
    _parse_prediction,
    _source_paired,
    _start_strict_progress,
    _strict_metrics,
    _verify_selected_data,
)
from .stats_guarded_external import (
    _messages as _menu_free_messages,
)
from .stats_invalid_control_catalog_fallback import (
    _apply_catalog_guard,
    _fallback_messages,
    _question_template,
)
from .stats_representation_probe import _METHOD_IDS

_METHOD = "E6_FRESH_SOURCE_CATALOG_FALLBACK_V1"
_METHOD_VERSION = 1
_EVALUATOR_VERSION = 1
_H22_RESULT_FINGERPRINT = "0c2dc933fdfb04ab5569be8042afdbe55837bb3fb1dd8d19ad12b4959825ee1c"
_H22_REPORT_SHA256 = "2dc299a0fd29398963869a1dcd3d76a594219ea934a7a7a902d61b28a10f41c3"
_METADATA_SCREEN_FINGERPRINT = "774ff734c62a655c4c6cd67aba4ea32d1cfcf63596245248cabf8b79176df7d1"
_METADATA_SCREEN_SHA256 = "0865db871a674c9e7f21f565122bac054a93e9926a7d9050f44858d71e3f2264"
_PERMISSIVE_LICENSE_ALLOWLIST = (
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC-BY-3.0",
    "CC-BY-4.0",
    "CC0-1.0",
    "MIT",
    "ODC-BY-1.0",
    "PDDL-1.0",
)
_DECISION_FRAME_ROLES = {
    "design",
    "outcome_type",
    "predictor_type",
    "group_structure",
    "pairedness",
    "time_structure",
    "censoring",
    "distribution_assumption",
    "measurement_scale",
    "objective",
    "sampling_structure",
    "response_structure",
}
_MAX_DECISION_FRAME_FIELDS = 8
_MAX_DECISION_FIELD_VALUES = 16
_SOURCE_GATES = {
    "minimum_eligible_cases": 150,
    "minimum_distinct_methods": 8,
    "minimum_coverage_fraction": 0.8,
    "maximum_single_method_fraction": 0.4,
}
_OPPORTUNITY_GATES = {
    "minimum_invalid_control_opportunities": 12,
    "minimum_invalid_control_gold_methods": 4,
    "minimum_distinct_invalid_question_templates": 8,
    "maximum_single_invalid_template_fraction": 0.25,
    "maximum_single_invalid_method_case_fraction": 0.5,
    "maximum_single_invalid_method_template_fraction": 0.5,
    "minimum_valid_control_identity_cases": 75,
}
_RESULT_GATES = {
    "minimum_candidate_only_gains": 6,
    "minimum_net_improvements": 6,
    "minimum_accuracy_gain_points": 4.0,
    "maximum_case_mcnemar_p": 0.05,
    "minimum_distinct_repaired_question_templates": 6,
    "minimum_template_net_improvements": 6,
    "minimum_repaired_gold_methods": 3,
    "maximum_single_repaired_method_fraction": 0.5,
    "maximum_template_mcnemar_p": 0.05,
    "minimum_valid_fallback_precision": 0.8,
    "maximum_valid_but_wrong_fallbacks": 1,
}

ModelCaller = Callable[[str, list[dict[str, str]], dict[str, Any]], str]
PhaseHook = Callable[[], None]


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "catalog-fallback-external-v1"


def _reports_root(config: ProjectConfig) -> Path:
    return config.root / "reports" / "evolve"


def _write_immutable_json(path: Path, payload: dict[str, Any], *, label: str) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"E6 {label} is immutable")
        return
    write_json(path, payload)


def _write_immutable_jsonl(path: Path, rows: list[dict[str, Any]], *, label: str) -> None:
    if path.exists():
        if list(read_jsonl(path)) != rows:
            raise RuntimeError(f"E6 {label} is immutable")
        return
    write_jsonl(path, rows)


def _write_terminal_state_receipt(
    config: ProjectConfig,
    *,
    master_fingerprint: str,
    status: str,
    reason: str,
    child_fingerprint: str | None,
    model_output_opened: bool,
    dataset_rows_opened: bool = False,
    result_fingerprint: str | None = None,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "method": _METHOD,
        "master_contract_fingerprint": master_fingerprint,
        "child_contract_fingerprint": child_fingerprint,
        "status": status,
        "terminal_state": status,
        "reason": reason,
        "result_fingerprint": result_fingerprint,
        "dataset_rows_opened": dataset_rows_opened,
        "model_output_opened": model_output_opened,
        "training_authorized": False,
        "champion_changed": False,
        "release_authorized": False,
    }
    receipt["fingerprint"] = canonical_hash(receipt)
    _write_immutable_json(
        _root(config) / "terminal-state.json",
        receipt,
        label="terminal-state receipt",
    )
    write_json(
        _reports_root(config) / "catalog-fallback-external-v1-terminal-state.json",
        receipt,
    )
    return receipt


def _read_terminal_state_receipt(config: ProjectConfig) -> dict[str, Any] | None:
    path = _root(config) / "terminal-state.json"
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in receipt.items() if key != "fingerprint"}
    if receipt.get("fingerprint") != canonical_hash(unsigned):
        raise RuntimeError("E6 terminal-state receipt changed")
    return receipt


def _implementation_manifest() -> dict[str, str]:
    source_root = Path(__file__).resolve().parent
    names = [
        "stats_catalog_fallback_external.py",
        "stats_invalid_control_catalog_fallback.py",
        "stats_guarded_external.py",
        "stats_catalog_grounding.py",
        "stats_external_catalog.py",
        "stats_agent.py",
        "stats_catalog.py",
        "stats_representation_probe.py",
    ]
    return {name: sha256_file(source_root / name) for name in names}


def _runtime_manifest(config: ProjectConfig) -> dict[str, Any]:
    e5_path = _reports_root(config) / "guarded-external-v1-master-contract.json"
    h22_path = _reports_root(config) / "invalid-control-catalog-fallback-v1.json"
    e5 = json.loads(e5_path.read_text(encoding="utf-8"))
    h22 = json.loads(h22_path.read_text(encoding="utf-8"))
    if h22.get("result_fingerprint") != _H22_RESULT_FINGERPRINT:
        raise RuntimeError("E6 requires the sealed H22 result")
    if (
        sha256_file(
            config.path_for("artifact_dir") / "invalid-control-catalog-fallback-v1" / "report.json"
        )
        != _H22_REPORT_SHA256
    ):
        raise RuntimeError("E6 H22 terminal report changed")
    if h22.get("status") != "SUPPORTED_DEV_COMPOSITION" or h22.get("training_authorized"):
        raise RuntimeError("E6 requires H22 to remain a development-only supported composition")
    e5_runtime = dict(e5["runtime"])
    parent = dict(e5_runtime["control"])
    manifest: dict[str, Any] = {
        "base_model": dict(e5_runtime["base_model"]),
        "parent": parent,
        "control_prompt_sha256": canonical_hash(
            _menu_free_messages({"question": "<SOURCE_QUESTION>"})
        ),
        "fallback_prompt_sha256": canonical_hash(
            _fallback_messages({"question": "<SOURCE_QUESTION>"})
        ),
        "decoding": dict(_DECODING),
        "h22_result_fingerprint": h22["result_fingerprint"],
        "h22_report_sha256": _H22_REPORT_SHA256,
    }
    manifest["fingerprint"] = canonical_hash(manifest)
    return manifest


def prepare_catalog_fallback_external_master_contract(config: ProjectConfig) -> dict[str, Any]:
    e5_path = _reports_root(config) / "guarded-external-v1-master-contract.json"
    metadata_path = _reports_root(config) / "guarded-external-v1-future-source-metadata-screen.json"
    e5 = json.loads(e5_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("fingerprint") != _METADATA_SCREEN_FINGERPRINT:
        raise RuntimeError("E6 requires the sealed metadata-only source screen")
    if sha256_file(metadata_path) != _METADATA_SCREEN_SHA256:
        raise RuntimeError("E6 metadata-only source screen changed")
    runtime = _runtime_manifest(config)
    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": _METHOD,
        "method_version": _METHOD_VERSION,
        "causal_question": (
            "On one genuinely new source selected and frozen before opening, can the unchanged "
            "parent preserve every parser-valid menu-free control and use the fixed global "
            "28-method catalog prompt to repair a statistically distinguishable set of invalid "
            "controls across independent question templates?"
        ),
        "evidence_status": "master-protocol-no-source-selected",
        "hypothesis_origin": {
            "h22_result_fingerprint": _H22_RESULT_FINGERPRINT,
            "h22_report_sha256": _H22_REPORT_SHA256,
            "h22_development_status": "SUPPORTED_DEV_COMPOSITION",
            "h22_case_mcnemar_p": 0.125,
            "h22_fresh_external_evidence": False,
            "h22_training_authorized": False,
        },
        "runtime": runtime,
        "runtime_policy": [
            "Generate every menu-free parent control exactly once.",
            "Never call the catalog fallback for a parser-valid control.",
            (
                "After all controls complete and the control model is released, call the "
                "unchanged parent exactly once for each parser-invalid control using the fixed "
                "global 28-method catalog prompt."
            ),
            "Use a parser-valid catalog fallback; otherwise retain the original invalid control.",
            (
                "Never expose gold labels, source answer fields, or other cases to either model "
                "caller."
            ),
        ],
        "prompt_policy": {
            "control": "H21/E5 menu-free canonical extraction prompt",
            "fallback": "same extraction prompt plus H7 fixed global 28-method catalog",
            "case_specific_menu_forbidden": True,
            "same_question_and_decoding": True,
            "prompt_change_is_only_the_fixed_global_catalog": True,
        },
        "progress_contract": (
            "Control and fallback phases use separate atomically rewritten ledgers. Every call "
            "writes a started receipt before model invocation; an interrupted call fails closed "
            "and is never retried."
        ),
        "source_protocol": {
            "source_count": 1,
            "selection_timing": (
                "Freeze one child contract with stable source identity, revision, snapshot, "
                "license, complete-frame extraction, alias map, overlap normalization, frozen "
                "historical question-hash corpus, and all gates before opening rows or examples. "
                "Build the case-specific overlap manifest only after the one-shot row opening and "
                "before any model call."
            ),
            "qualification_gates": dict(_SOURCE_GATES),
            "supported_snapshot_formats": ["json-array", "jsonl-records"],
            "complete_frame_policy": (
                "The generic E6 verifier deterministically rebuilds every method-bearing frame "
                "row from the frozen source snapshot, role-constrained categorical decision "
                "fields, declarative field paths, alias map, and all-nonempty-source-method-"
                "records rule. Record IDs, questions, answer or method fields, and unrestricted "
                "text cannot enter the decision frame. Materialized frame bytes, questions, gold "
                "mappings, case IDs, and semantic-template hashes must match that rebuild."
            ),
            "permissive_license_allowlist": list(_PERMISSIVE_LICENSE_ALLOWLIST),
            "license_evidence_required": {
                "canonical_identifier": True,
                "stable_license_url": True,
                "frozen_before_rows_open": True,
            },
            "opened_source_exclusions": dict(e5["source_protocol"]["opened_source_exclusions"]),
            "overlap_rule": (
                "Reject any source, version, derivative, or normalized question overlapping "
                "training, P-Bench, StatQA, E2-E5, H17-H22, or source/rule development."
            ),
            "semantic_template_contract": {
                "primary_inferential_unit": "semantic-decision-template",
                "cluster_key": "frozen-normalized-complete-frame-tuple",
                "assignment_timing": "before-model-output",
                "normalization": "recursive-casefold-whitespace-v1",
                "template_id_rule": "canonical-sha256-of-normalized-decision-frame",
                "question_materialization_rule": "normalized-question-field-only",
                "source_projection_frozen_in_child_extraction_contract": True,
                "exact_duplicate_messages_count_once": True,
                "semantic_paraphrases_with_same_decision_frame_count_once": True,
                "representative_rule": "lowest-sha256-case-id",
                "model_output_fields_used": [],
                "outcome_aware_reclustering": False,
            },
            "template_rule": (
                "The child contract freezes a normalized complete-frame clustering rule before "
                "rows open. After the one-shot row opening and before model calls, an immutable "
                "manifest assigns every case to one semantic decision template without using "
                "model outputs; duplicate messages and semantic paraphrases sharing the same "
                "decision frame count once."
            ),
            "gold_isolation": "Gold and source answer fields never cross the model-call boundary.",
        },
        "opportunity_gates": dict(_OPPORTUNITY_GATES),
        "result_gates": dict(_RESULT_GATES),
        "terminal_states": [
            "CONFIRMED_NARROW_FRESH_SOURCE_PASS",
            "SCIENTIFIC_FAIL",
            "INCONCLUSIVE_OPPORTUNITY",
            "SOURCE_UNQUALIFIED",
            "SOURCE_UNAVAILABLE",
            "PROTOCOL_INVALID",
        ],
        "terminal_state_rules": {
            "SOURCE_UNAVAILABLE": "no child can be frozen from metadata without opening rows",
            "SOURCE_UNQUALIFIED": "one frozen child fails source gates after one-shot opening",
            "INCONCLUSIVE_OPPORTUNITY": "qualified source lacks frozen opportunity coverage",
            "CONFIRMED_NARROW_FRESH_SOURCE_PASS": "all case and template gates pass",
            "SCIENTIFIC_FAIL": "opportunity passes but one or more result gates fail",
            "PROTOCOL_INVALID": "identity, routing, receipt, extraction, or replay integrity fails",
        },
        "stopping_rule": (
            "Every terminal state closes this E6 child. Do not replace the source, add cases, "
            "alter templates, tune prompts, add aliases, retry invalid calls, train weights, or "
            "rerun after opening."
        ),
        "metadata_screen": {
            "fingerprint": metadata["fingerprint"],
            "sha256": _METADATA_SCREEN_SHA256,
            "candidate_count": metadata["candidate_count"],
            "rows_opened": metadata["dataset_rows_opened"],
            "status": metadata["status"],
        },
        "implementation_sha256": _implementation_manifest(),
        "source_selected": False,
        "model_output_opened": False,
        "training_authorized": False,
        "champion_changed": False,
        "release_authorized": False,
        "claim_boundary": (
            "A pass can support only the exact parent-plus-fixed-catalog identity-guard runtime "
            "on one preregistered source. It cannot establish multi-source generality, authorize "
            "training, replace v0.3.0-parent, publish, or release."
        ),
    }
    contract["fingerprint"] = canonical_hash(contract)
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    _write_immutable_json(root / "master-contract.json", contract, label="master contract")
    write_json(
        _reports_root(config) / "catalog-fallback-external-v1-master-contract.json", contract
    )
    return contract


def _verify_metadata_receipts(
    receipts: Any,
    *,
    metadata_root: Path,
) -> list[dict[str, Any]]:
    allowed_classes = {
        "dataset-card-metadata",
        "file-metadata",
        "license-metadata",
        "schema-metadata",
        "statistics-metadata",
        "tree-metadata",
    }
    required = {
        "content_class",
        "request_uri",
        "retrieved_at",
        "response_path",
        "response_sha256",
        "dataset_rows_opened",
    }
    if not isinstance(receipts, list) or len(receipts) < 3:
        raise RuntimeError("E6 metadata source has too few receipts")
    verified = []
    for receipt in receipts:
        if not isinstance(receipt, dict) or set(receipt) != required:
            raise RuntimeError("E6 metadata receipt schema changed")
        if receipt["content_class"] not in allowed_classes or receipt["dataset_rows_opened"]:
            raise RuntimeError("E6 metadata receipt opened rows or has an unsupported class")
        response_path = Path(str(receipt["response_path"]))
        if not response_path.is_absolute():
            response_path = metadata_root / response_path
        response_path = _require_within(response_path, metadata_root, label="metadata response")
        if sha256_file(response_path) != receipt["response_sha256"]:
            raise RuntimeError("E6 metadata response bytes changed")
        verified.append({**receipt, "response_path": str(response_path)})
    return verified


def _load_historical_question_hashes(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("E6 historical question-hash corpus is empty or malformed")
    hashes = [str(value) for value in payload]
    if hashes != sorted(set(hashes)):
        raise RuntimeError("E6 historical question hashes must be sorted and unique")
    if any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in hashes
    ):
        raise RuntimeError("E6 historical question-hash corpus contains a malformed hash")
    return hashes


def _metadata_source_qualification(source: dict[str, Any]) -> dict[str, Any]:
    projection = source.get("e6_projection")
    alias_map = source.get("alias_map")
    source_counts = source.get("source_method_counts")
    schema_fields = source.get("schema_fields")
    if not all(
        isinstance(value, expected)
        for value, expected in (
            (projection, dict),
            (alias_map, dict),
            (source_counts, dict),
            (schema_fields, list),
        )
    ):
        raise RuntimeError("E6 metadata qualification fields are malformed")
    if any(
        not isinstance(label, str)
        or label != " ".join(label.split())
        or not label
        or method_id not in _METHOD_IDS
        for label, method_id in alias_map.items()
    ):
        raise RuntimeError("E6 metadata alias map is malformed or leaves the frozen catalog")
    required_fields = {
        str(projection.get("record_id_field", "")),
        str(projection.get("question_field", "")),
        str(projection.get("source_method_field", "")),
        *[str(field) for field in projection.get("decision_frame_fields", [])],
        *[str(field) for field in projection.get("answer_fields", [])],
    }
    if "" in required_fields or not required_fields.issubset(
        {str(field) for field in schema_fields}
    ):
        raise RuntimeError("E6 metadata schema cannot freeze the source projection")
    method_bearing = int(source["method_bearing_record_count"])
    declared = int(source["declared_record_count"])
    if declared < method_bearing or method_bearing <= 0:
        raise RuntimeError("E6 metadata source record counts are inconsistent")
    mapped_counts: Counter[str] = Counter()
    observed = 0
    for label, raw_count in source_counts.items():
        count = int(raw_count)
        if count < 0:
            raise RuntimeError("E6 metadata source method count is negative")
        observed += count
        method_id = alias_map.get(str(label))
        if method_id in _METHOD_IDS:
            mapped_counts[str(method_id)] += count
    if observed != method_bearing:
        raise RuntimeError("E6 metadata method counts changed their denominator")
    eligible = sum(mapped_counts.values())
    coverage = eligible / method_bearing
    maximum_share = max(mapped_counts.values()) / eligible if eligible else 1.0
    checks = {
        "minimum_eligible_cases": eligible >= _SOURCE_GATES["minimum_eligible_cases"],
        "minimum_distinct_methods": len(mapped_counts) >= _SOURCE_GATES["minimum_distinct_methods"],
        "minimum_coverage_fraction": coverage >= _SOURCE_GATES["minimum_coverage_fraction"],
        "maximum_single_method_fraction": maximum_share
        <= _SOURCE_GATES["maximum_single_method_fraction"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"E6 metadata cannot freeze a qualified child: {checks}")
    return {
        "declared_record_count": declared,
        "method_bearing_record_count": method_bearing,
        "eligible_count": eligible,
        "mapped_method_counts": dict(sorted(mapped_counts.items())),
        "coverage_fraction": coverage,
        "maximum_single_method_fraction": maximum_share,
        "checks": checks,
    }


def prepare_catalog_fallback_external_child_contract(
    config: ProjectConfig,
    *,
    metadata_bundle_path: Path,
) -> dict[str, Any]:
    master = prepare_catalog_fallback_external_master_contract(config)
    terminal = _read_terminal_state_receipt(config)
    if terminal is not None:
        raise RuntimeError(f"E6 is already closed at terminal state {terminal['status']}")
    metadata_root = (
        config.path_for("evolution_dir") / "catalog-fallback-external-v1" / "metadata-only"
    )
    bundle_path = _require_within(
        metadata_bundle_path, metadata_root, label="metadata qualification bundle"
    )
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    required_bundle_fields = {
        "schema_version",
        "complete",
        "metadata_only",
        "dataset_rows_opened",
        "source",
        "receipts",
        "historical_overlap",
    }
    if not isinstance(bundle, dict) or set(bundle) != required_bundle_fields:
        raise RuntimeError("E6 metadata qualification bundle schema changed")
    if (
        bundle["schema_version"] != 1
        or not bundle["complete"]
        or not bundle["metadata_only"]
        or bundle["dataset_rows_opened"]
    ):
        raise RuntimeError("E6 child selection is not complete metadata-only evidence")
    source_metadata = bundle["source"]
    required_source_fields = {
        "source_id",
        "stable_id",
        "revision",
        "license",
        "license_url",
        "snapshot_url",
        "snapshot_sha256",
        "snapshot_format",
        "declared_record_count",
        "method_bearing_record_count",
        "source_method_counts",
        "schema_fields",
        "alias_map",
        "e6_projection",
    }
    if not isinstance(source_metadata, dict) or set(source_metadata) != required_source_fields:
        raise RuntimeError("E6 metadata source schema changed")
    if source_metadata["license"] not in _PERMISSIVE_LICENSE_ALLOWLIST:
        raise RuntimeError("E6 metadata source license is outside the frozen allowlist")
    if source_metadata["snapshot_format"] not in {"json-array", "jsonl-records"}:
        raise RuntimeError("E6 metadata source format is unsupported")
    for field in ("source_id", "stable_id", "revision", "license_url", "snapshot_url"):
        if not str(source_metadata[field]).strip():
            raise RuntimeError(f"E6 metadata source {field} is missing")
    snapshot_sha256 = str(source_metadata["snapshot_sha256"])
    if len(snapshot_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in snapshot_sha256
    ):
        raise RuntimeError("E6 metadata snapshot SHA-256 is malformed")
    qualification = _metadata_source_qualification(source_metadata)
    receipts = _verify_metadata_receipts(bundle["receipts"], metadata_root=metadata_root)
    historical = bundle["historical_overlap"]
    required_historical_fields = {
        "normalization",
        "historical_question_hashes_path",
        "historical_question_hashes_sha256",
        "historical_corpus_fingerprint",
    }
    if not isinstance(historical, dict) or set(historical) != required_historical_fields:
        raise RuntimeError("E6 historical-overlap contract schema changed")
    if historical["normalization"] != "casefold-whitespace-sha256-v1":
        raise RuntimeError("E6 historical-overlap normalization changed")
    historical_path = Path(str(historical["historical_question_hashes_path"]))
    if not historical_path.is_absolute():
        historical_path = metadata_root / historical_path
    historical_path = _require_within(
        historical_path,
        metadata_root,
        label="historical question-hash corpus",
    )
    if sha256_file(historical_path) != historical["historical_question_hashes_sha256"]:
        raise RuntimeError("E6 historical question-hash corpus changed")
    historical_hashes = _load_historical_question_hashes(historical_path)
    if canonical_hash(historical_hashes) != historical["historical_corpus_fingerprint"]:
        raise RuntimeError("E6 historical question-hash corpus fingerprint changed")

    projection = dict(source_metadata["e6_projection"])
    if projection.get("snapshot_format") != source_metadata["snapshot_format"]:
        raise RuntimeError("E6 metadata snapshot format and projection disagree")

    selected_root = _root(config) / "selected-source"
    extension = ".jsonl" if source_metadata["snapshot_format"] == "jsonl-records" else ".json"
    alias_map = dict(source_metadata["alias_map"])
    extraction = {"e6_projection": projection}
    source = {
        "source_id": str(source_metadata["source_id"]),
        "stable_id": str(source_metadata["stable_id"]),
        "revision": str(source_metadata["revision"]),
        "license": str(source_metadata["license"]),
        "license_url": str(source_metadata["license_url"]),
        "snapshot_url": str(source_metadata["snapshot_url"]),
        "snapshot_path": str(selected_root / f"snapshot{extension}"),
        "snapshot_sha256": snapshot_sha256,
        "alias_map": alias_map,
        "alias_map_fingerprint": canonical_hash(alias_map),
        "extraction_contract": extraction,
        "extraction_contract_fingerprint": canonical_hash(extraction),
    }
    metadata_receipt = {
        "bundle_path": str(bundle_path),
        "bundle_sha256": sha256_file(bundle_path),
        "receipts_fingerprint": canonical_hash(receipts),
        **qualification,
    }
    template_contract = dict(master["source_protocol"]["semantic_template_contract"])
    child: dict[str, Any] = {
        "schema_version": 1,
        "method": _METHOD,
        "master_contract_fingerprint": master["fingerprint"],
        "runtime_fingerprint": master["runtime"]["fingerprint"],
        "implementation_sha256": dict(master["implementation_sha256"]),
        "source_selected_before_opening": True,
        "metadata_prequalified": True,
        "dataset_rows_opened": False,
        "metadata_qualification_receipt": metadata_receipt,
        "metadata_qualification_fingerprint": canonical_hash(metadata_receipt),
        "source": source,
        "semantic_template_contract": template_contract,
        "semantic_template_contract_fingerprint": canonical_hash(template_contract),
        "overlap_contract": {
            **historical,
            "historical_question_hashes_path": str(historical_path),
        },
        "overlap_contract_fingerprint": canonical_hash(
            {
                **historical,
                "historical_question_hashes_path": str(historical_path),
            }
        ),
        "evaluation_authorized": False,
        "model_output_opened": False,
        "training_authorized": False,
        "champion_changed": False,
        "release_authorized": False,
    }
    child["fingerprint"] = canonical_hash(child)
    _verify_e6_child_contract(child, master)
    _write_immutable_json(
        _root(config) / "selected-source-contract.json",
        child,
        label="selected-source contract",
    )
    public = {
        "schema_version": 1,
        "method": _METHOD,
        "master_contract_fingerprint": master["fingerprint"],
        "child_contract_fingerprint": child["fingerprint"],
        "source_id": source["source_id"],
        "license": source["license"],
        "metadata_prequalified": True,
        "dataset_rows_opened": False,
        "evaluation_authorized": False,
        "training_authorized": False,
        "champion_changed": False,
        "release_authorized": False,
    }
    public["fingerprint"] = canonical_hash(public)
    write_json(
        _reports_root(config) / "catalog-fallback-external-v1-selected-source.json",
        public,
    )
    return public


class _SelectedSourceUnavailable(RuntimeError):
    pass


def _download_selected_snapshot(path: Path, source: dict[str, Any]) -> bytes:
    expected = str(source["snapshot_sha256"])
    if path.exists():
        payload = path.read_bytes()
    else:
        request = urllib.request.Request(
            str(source["snapshot_url"]),
            headers={"User-Agent": "CharlieAlphaResearch/1.0 (+e6-fresh-source)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as error:
            raise _SelectedSourceUnavailable("frozen snapshot could not be opened") from error
    if sha256_bytes(payload) != expected:
        raise RuntimeError("E6 selected-source snapshot bytes differ from the frozen SHA-256")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return payload


def _source_data_qualification(
    frame: list[dict[str, Any]],
    child: dict[str, Any],
) -> dict[str, Any]:
    source = child["source"]
    projection = source["extraction_contract"]["e6_projection"]
    record_count = len(_load_source_records(Path(str(source["snapshot_path"])), projection))
    cases = [row for row in frame if row["mapped_method_id"] in _METHOD_IDS]
    method_counts = Counter(str(row["mapped_method_id"]) for row in cases)
    coverage = len(cases) / len(frame)
    maximum_share = max(method_counts.values()) / len(cases) if cases else 1.0
    checks = {
        "minimum_eligible_cases": len(cases) >= _SOURCE_GATES["minimum_eligible_cases"],
        "minimum_distinct_methods": len(method_counts) >= _SOURCE_GATES["minimum_distinct_methods"],
        "minimum_coverage_fraction": coverage >= _SOURCE_GATES["minimum_coverage_fraction"],
        "maximum_single_method_fraction": maximum_share
        <= _SOURCE_GATES["maximum_single_method_fraction"],
    }
    observed = {
        "declared_record_count": record_count,
        "method_bearing_record_count": len(frame),
        "eligible_count": len(cases),
        "mapped_method_counts": dict(sorted(method_counts.items())),
        "coverage_fraction": coverage,
        "maximum_single_method_fraction": maximum_share,
        "checks": checks,
    }
    metadata = child["metadata_qualification_receipt"]
    for field in (
        "declared_record_count",
        "method_bearing_record_count",
        "eligible_count",
        "mapped_method_counts",
    ):
        if observed[field] != metadata[field]:
            raise RuntimeError(f"E6 selected-source rows contradict metadata field: {field}")
    return observed


def _semantic_template_manifest(
    cases: list[dict[str, Any]],
    frame: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], list[str]]:
    frame_by_case = {str(row["case_id"]): row for row in frame if row.get("case_id") is not None}
    assignments = {
        str(case["case_id"]): canonical_hash(
            _normalize_frame_value(frame_by_case[str(case["case_id"])]["decision_frame"])
        )
        for case in cases
    }
    problems: list[str] = []
    question_templates: dict[str, str] = {}
    for case in cases:
        case_id = str(case["case_id"])
        normalized_question = _question_template(str(case["question"]))
        previous = question_templates.setdefault(normalized_question, assignments[case_id])
        if previous != assignments[case_id]:
            problems.append("duplicate-question-split-across-decision-frames")
    clusters: dict[str, list[str]] = {}
    for case_id, template_id in assignments.items():
        clusters.setdefault(template_id, []).append(case_id)
    cases_by_id = {str(case["case_id"]): case for case in cases}
    manifest: list[dict[str, Any]] = []
    for template_id, case_ids in sorted(clusters.items()):
        ordered_ids = sorted(case_ids)
        methods = {str(cases_by_id[case_id]["gold_method_id"]) for case_id in ordered_ids}
        if len(methods) != 1:
            problems.append("decision-frame-has-conflicting-gold-methods")
        representative = min(
            ordered_ids,
            key=lambda case_id: (sha256_text(case_id), case_id),
        )
        for case_id in ordered_ids:
            manifest.append(
                {
                    "case_id": case_id,
                    "semantic_template_id": template_id,
                    "normalized_frame_fingerprint": template_id,
                    "representative_case_id": representative,
                }
            )
    return manifest, assignments, sorted(set(problems))


def _selected_data_public(data: dict[str, Any]) -> dict[str, Any]:
    public = {
        "schema_version": 1,
        "method": _METHOD,
        "master_contract_fingerprint": data["master_contract_fingerprint"],
        "child_contract_fingerprint": data["contract_fingerprint"],
        "data_fingerprint": data["data_fingerprint"],
        "source_id": data["source_id"],
        "case_count": data["case_count"],
        "semantic_template_count": data["semantic_template_count"],
        "overlap_count": data["overlap_count"],
        "status": "QUALIFIED_SOURCE_DATA_FROZEN",
        "dataset_rows_opened": True,
        "evaluation_authorized": True,
        "model_output_opened": False,
        "training_authorized": False,
        "champion_changed": False,
        "release_authorized": False,
    }
    public["fingerprint"] = canonical_hash(public)
    return public


def prepare_catalog_fallback_external_selected_data(config: ProjectConfig) -> dict[str, Any]:
    master = prepare_catalog_fallback_external_master_contract(config)
    root = _root(config)
    child_path = root / "selected-source-contract.json"
    if not child_path.is_file():
        raise RuntimeError("E6 has no metadata-frozen selected-source child")
    child = json.loads(child_path.read_text(encoding="utf-8"))
    _verify_e6_child_contract(child, master)
    terminal = _read_terminal_state_receipt(config)
    if terminal is not None:
        return terminal

    selected_root = root / "selected-source"
    data_path = root / "selected-source-data.json"
    public_path = _reports_root(config) / "catalog-fallback-external-v1-selected-source-data.json"
    if data_path.is_file():
        data = json.loads(data_path.read_text(encoding="utf-8"))
        cases = _verify_selected_data(data, child, master, selected_root=selected_root)
        _verify_semantic_template_manifest(
            data,
            child,
            master,
            cases,
            selected_root=selected_root,
        )
        public = _selected_data_public(data)
        write_json(public_path, public)
        return public

    source_opening_path = root / "source-opening.json"
    if source_opening_path.exists():
        _write_terminal_state_receipt(
            config,
            master_fingerprint=master["fingerprint"],
            status="PROTOCOL_INVALID",
            reason="interrupted-after-source-opening-authorization",
            child_fingerprint=child["fingerprint"],
            dataset_rows_opened=True,
            model_output_opened=False,
        )
        raise RuntimeError("E6 source opening was interrupted and cannot be retried")

    source = child["source"]
    opening: dict[str, Any] = {
        "schema_version": 1,
        "method": _METHOD,
        "master_contract_fingerprint": master["fingerprint"],
        "child_contract_fingerprint": child["fingerprint"],
        "source_id": source["source_id"],
        "snapshot_url": source["snapshot_url"],
        "snapshot_sha256": source["snapshot_sha256"],
        "source_opening_started": True,
        "dataset_rows_may_be_opened": True,
        "model_output_opened": False,
        "training_authorized": False,
        "champion_changed": False,
        "release_authorized": False,
    }
    opening["fingerprint"] = canonical_hash(opening)
    _write_immutable_json(source_opening_path, opening, label="source-opening receipt")
    rows_opened = False
    try:
        snapshot_path = _require_within(
            Path(str(source["snapshot_path"])),
            selected_root,
            label="source snapshot",
        )
        payload = _download_selected_snapshot(snapshot_path, source)
        rows_opened = True
        opened: dict[str, Any] = {
            "schema_version": 1,
            "method": _METHOD,
            "source_opening_fingerprint": opening["fingerprint"],
            "snapshot_sha256": sha256_bytes(payload),
            "snapshot_byte_count": len(payload),
            "dataset_rows_opened": True,
            "model_output_opened": False,
        }
        opened["fingerprint"] = canonical_hash(opened)
        _write_immutable_json(root / "source-opened.json", opened, label="source-opened receipt")

        frame = _recompute_complete_frame(child)
        qualification = _source_data_qualification(frame, child)
        cases = [
            {
                "case_id": str(row["case_id"]),
                "source_id": str(source["source_id"]),
                "question": str(row["question"]),
                "gold_method_id": str(row["mapped_method_id"]),
            }
            for row in frame
            if row["mapped_method_id"] in _METHOD_IDS
        ]
        historical_path = Path(str(child["overlap_contract"]["historical_question_hashes_path"]))
        historical_hashes = set(_load_historical_question_hashes(historical_path))
        case_question_hashes = {
            str(case["case_id"]): sha256_text(" ".join(str(case["question"]).casefold().split()))
            for case in cases
        }
        overlapping_hashes = sorted(set(case_question_hashes.values()) & historical_hashes)
        overlap: dict[str, Any] = {
            "schema_version": 1,
            "normalization": "casefold-whitespace-sha256-v1",
            "normalization_fingerprint": canonical_hash(
                {"normalization": "casefold-whitespace-sha256-v1"}
            ),
            "historical_corpus_fingerprint": child["overlap_contract"][
                "historical_corpus_fingerprint"
            ],
            "case_fingerprint": canonical_hash(cases),
            "case_question_hashes_fingerprint": canonical_hash(case_question_hashes),
            "overlap_count": len(overlapping_hashes),
            "overlapping_question_hashes": overlapping_hashes,
            "built_after_rows_opened": True,
            "built_before_model_output": True,
        }
        overlap["fingerprint"] = canonical_hash(overlap)

        manifest, assignments, template_problems = _semantic_template_manifest(cases, frame)
        frame_path = selected_root / "complete-frame.jsonl"
        cases_path = selected_root / "cases.jsonl"
        overlap_path = selected_root / "overlap-manifest.json"
        manifest_path = selected_root / "semantic-templates.jsonl"
        _write_immutable_jsonl(frame_path, frame, label="complete source frame")
        _write_immutable_jsonl(cases_path, cases, label="selected-source cases")
        _write_immutable_json(overlap_path, overlap, label="overlap manifest")
        _write_immutable_jsonl(manifest_path, manifest, label="semantic-template manifest")

        observation: dict[str, Any] = {
            "schema_version": 1,
            "method": _METHOD,
            "master_contract_fingerprint": master["fingerprint"],
            "child_contract_fingerprint": child["fingerprint"],
            "source_opening_fingerprint": opening["fingerprint"],
            "source_opened_fingerprint": opened["fingerprint"],
            "source_id": source["source_id"],
            "qualification": qualification,
            "overlap_count": len(overlapping_hashes),
            "semantic_template_count": len(set(assignments.values())),
            "semantic_template_problems": template_problems,
            "complete_frame_sha256": sha256_file(frame_path),
            "cases_sha256": sha256_file(cases_path),
            "overlap_manifest_sha256": sha256_file(overlap_path),
            "semantic_template_manifest_sha256": sha256_file(manifest_path),
            "dataset_rows_opened": True,
            "model_output_opened": False,
            "training_authorized": False,
            "champion_changed": False,
            "release_authorized": False,
        }
        observation["result_fingerprint"] = canonical_hash(observation)
        _write_immutable_json(
            root / "selected-source-observation.json",
            observation,
            label="selected-source observation",
        )
        write_json(
            _reports_root(config) / "catalog-fallback-external-v1-selected-source-observation.json",
            observation,
        )

        source_qualified = all(qualification["checks"].values())
        if not source_qualified or overlapping_hashes or template_problems:
            reasons = []
            if not source_qualified:
                reasons.append("source-gates-failed")
            if overlapping_hashes:
                reasons.append("historical-question-overlap")
            reasons.extend(template_problems)
            return _write_terminal_state_receipt(
                config,
                master_fingerprint=master["fingerprint"],
                status="SOURCE_UNQUALIFIED",
                reason=";".join(reasons),
                child_fingerprint=child["fingerprint"],
                dataset_rows_opened=True,
                model_output_opened=False,
                result_fingerprint=observation["result_fingerprint"],
            )

        data: dict[str, Any] = {
            "schema_version": 1,
            "complete": True,
            "method": _METHOD,
            "master_contract_fingerprint": master["fingerprint"],
            "contract_fingerprint": child["fingerprint"],
            "source_opening_fingerprint": opening["fingerprint"],
            "source_opened_fingerprint": opened["fingerprint"],
            "source_id": source["source_id"],
            "case_count": len(cases),
            "cases_path": str(cases_path),
            "cases_sha256": sha256_file(cases_path),
            "case_fingerprint": canonical_hash(cases),
            "complete_frame_path": str(frame_path),
            "complete_frame_sha256": sha256_file(frame_path),
            "complete_frame_fingerprint": canonical_hash(frame),
            "overlap_manifest_path": str(overlap_path),
            "overlap_manifest_sha256": sha256_file(overlap_path),
            "overlap_count": 0,
            "alias_map_fingerprint": source["alias_map_fingerprint"],
            "extraction_contract_fingerprint": source["extraction_contract_fingerprint"],
            "semantic_template_contract_fingerprint": child[
                "semantic_template_contract_fingerprint"
            ],
            "semantic_template_manifest_path": str(manifest_path),
            "semantic_template_manifest_sha256": sha256_file(manifest_path),
            "semantic_template_manifest_fingerprint": canonical_hash(manifest),
            "semantic_template_assignment_fingerprint": canonical_hash(assignments),
            "semantic_template_count": len(set(assignments.values())),
            "qualification": qualification,
            "dataset_rows_opened": True,
            "evaluation_authorized": True,
            "model_output_opened": False,
            "training_authorized": False,
            "champion_changed": False,
            "release_authorized": False,
        }
        data["data_fingerprint"] = canonical_hash(data)
        verified_cases = _verify_selected_data(
            data,
            child,
            master,
            selected_root=selected_root,
        )
        _verify_semantic_template_manifest(
            data,
            child,
            master,
            verified_cases,
            selected_root=selected_root,
        )
        _write_immutable_json(data_path, data, label="selected-source data")
        public = _selected_data_public(data)
        write_json(public_path, public)
        return public
    except _SelectedSourceUnavailable:
        return _write_terminal_state_receipt(
            config,
            master_fingerprint=master["fingerprint"],
            status="SOURCE_UNAVAILABLE",
            reason="frozen-selected-source-snapshot-unavailable",
            child_fingerprint=child["fingerprint"],
            dataset_rows_opened=False,
            model_output_opened=False,
        )
    except Exception:
        _write_terminal_state_receipt(
            config,
            master_fingerprint=master["fingerprint"],
            status="PROTOCOL_INVALID",
            reason="selected-source-opening-or-materialization-integrity-failed",
            child_fingerprint=child["fingerprint"],
            dataset_rows_opened=rows_opened,
            model_output_opened=False,
        )
        raise


def prepare_catalog_fallback_external_source_availability(config: ProjectConfig) -> dict[str, Any]:
    master = prepare_catalog_fallback_external_master_contract(config)
    if (_root(config) / "selected-source-contract.json").is_file():
        raise RuntimeError("E6 already has a metadata-frozen child; use selected-source data")
    metadata_path = _reports_root(config) / "guarded-external-v1-future-source-metadata-screen.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("source_selected") or metadata.get("dataset_rows_opened"):
        raise RuntimeError("E6 metadata screen unexpectedly selected or opened a source")
    if any(
        candidate.get("qualification_checks", {}).get("qualified_child_contract_freezable")
        for candidate in metadata.get("candidates", [])
    ):
        raise RuntimeError("E6 metadata screen unexpectedly contains a qualified child")
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "E6 fresh-source metadata availability readback",
        "master_contract_fingerprint": master["fingerprint"],
        "metadata_screen_fingerprint": metadata["fingerprint"],
        "metadata_screen_sha256": _METADATA_SCREEN_SHA256,
        "candidate_count": metadata["candidate_count"],
        "near_candidate_count": metadata["near_candidate_count"],
        "source_gates": dict(_SOURCE_GATES),
        "status": "SOURCE_UNAVAILABLE",
        "metadata_screen_status": metadata["status"],
        "source_selected": False,
        "child_contract_created": False,
        "dataset_rows_opened": False,
        "model_output_opened": False,
        "evaluation_authorized": False,
        "training_authorized": False,
        "champion_changed": False,
        "release_authorized": False,
        "reason": (
            "None of the 24 metadata-only candidates can establish before opening a complete "
            "single-source question-to-canonical-method frame satisfying 150/8/80%/40%."
        ),
        "next_step": "wait-for-a-new-source-that-can-be-frozen-under-the-unchanged-master",
    }
    report["fingerprint"] = canonical_hash(report)
    _write_immutable_json(
        _root(config) / "source-availability.json", report, label="source availability"
    )
    write_json(
        _reports_root(config) / "catalog-fallback-external-v1-source-availability.json",
        report,
    )
    return report


def prepare_catalog_fallback_external_data(config: ProjectConfig) -> dict[str, Any]:
    master = prepare_catalog_fallback_external_master_contract(config)
    if (_root(config) / "selected-source-contract.json").is_file():
        return prepare_catalog_fallback_external_selected_data(config)
    availability = prepare_catalog_fallback_external_source_availability(config)
    case_path = config.path_for("evolution_dir") / "catalog-fallback-external-v1" / "cases.jsonl"
    write_jsonl(case_path, [])
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": _METHOD,
        "master_contract_fingerprint": master["fingerprint"],
        "source_availability_fingerprint": availability["fingerprint"],
        "source_selected": False,
        "case_count": 0,
        "cases_sha256": sha256_file(case_path),
        "terminal_state": "SOURCE_UNAVAILABLE",
        "evaluation_authorized": False,
        "model_output_opened": False,
        "training_authorized": False,
        "champion_changed": False,
        "release_authorized": False,
    }
    report["data_fingerprint"] = canonical_hash(report)
    _write_immutable_json(_root(config) / "data.json", report, label="data receipt")
    write_json(_reports_root(config) / "catalog-fallback-external-v1-data.json", report)
    _write_terminal_state_receipt(
        config,
        master_fingerprint=master["fingerprint"],
        status="SOURCE_UNAVAILABLE",
        reason="metadata-screen-found-no-freezable-single-source-child",
        child_fingerprint=None,
        dataset_rows_opened=False,
        model_output_opened=False,
        result_fingerprint=report["data_fingerprint"],
    )
    return report


def _phase_messages(case: dict[str, Any], *, grounded: bool) -> list[dict[str, str]]:
    return _fallback_messages(case) if grounded else _menu_free_messages(case)


def _phase_header(
    cases: list[dict[str, Any]],
    *,
    phase: str,
    grounded: bool,
    evaluation_fingerprint: str,
    runtime_receipt: dict[str, Any],
) -> dict[str, Any]:
    receipts = []
    seen: set[str] = set()
    for case in cases:
        case_id = str(case["case_id"])
        if case_id in seen:
            raise RuntimeError(f"E6 case IDs are not unique: {case_id}")
        seen.add(case_id)
        case_hash = _case_fingerprint(case)
        messages_hash = canonical_hash(_phase_messages(case, grounded=grounded))
        receipts.append(
            {
                "case_id": case_id,
                "case_fingerprint": case_hash,
                "messages_sha256": messages_hash,
                "request_hash": canonical_hash(
                    {
                        "evaluation": evaluation_fingerprint,
                        "phase": phase,
                        "case_id": case_id,
                        "case_fingerprint": case_hash,
                        "messages_sha256": messages_hash,
                        "decoding": _DECODING,
                        "runtime_receipt": runtime_receipt,
                        "evaluator_version": _EVALUATOR_VERSION,
                    }
                ),
            }
        )
    header: dict[str, Any] = {
        "schema_version": 1,
        "phase": phase,
        "grounded": grounded,
        "evaluation_fingerprint": evaluation_fingerprint,
        "runtime_receipt": runtime_receipt,
        "decoding": dict(_DECODING),
        "expected_count": len(receipts),
        "expected_receipts": receipts,
        "evaluator_version": _EVALUATOR_VERSION,
    }
    header["fingerprint"] = canonical_hash(header)
    return header


def _phase_detail(case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    valid = bool(row["valid_output"])
    predicted = row["predicted_method_id"]
    return {
        "case_id": str(case["case_id"]),
        "source_id": str(case["source_id"]),
        "question": str(case["question"]),
        "gold_method_id": str(case["gold_method_id"]),
        "predicted_method_id": predicted,
        "valid_output": valid,
        "correct": valid and predicted == str(case["gold_method_id"]),
        "parse_reason": str(row["parse_reason"]),
        "raw_answer_sha256": str(row["raw_answer_sha256"]),
        "request_hash": str(row["request_hash"]),
        "messages_sha256": str(row["messages_sha256"]),
    }


def _evaluate_phase(
    cases: list[dict[str, Any]],
    *,
    phase: str,
    grounded: bool,
    evaluation_fingerprint: str,
    runtime_receipt: dict[str, Any],
    caller: ModelCaller,
    progress_path: Path | None,
) -> list[dict[str, Any]]:
    if not cases:
        return []
    header = _phase_header(
        cases,
        phase=phase,
        grounded=grounded,
        evaluation_fingerprint=evaluation_fingerprint,
        runtime_receipt=runtime_receipt,
    )
    rows = _load_strict_progress(progress_path, header=header) if progress_path is not None else []
    receipts = list(header["expected_receipts"])
    for index in range(len(rows), len(cases)):
        case = cases[index]
        if progress_path is not None:
            _start_strict_progress(progress_path, header=header)
        answer = caller(
            str(case["case_id"]),
            _phase_messages(case, grounded=grounded),
            dict(_DECODING),
        )
        if not isinstance(answer, str):
            raise RuntimeError("E6 model caller returned a non-string answer")
        row = {
            **receipts[index],
            "phase": phase,
            "raw_answer": answer,
            "raw_answer_sha256": sha256_text(answer),
            **_parse_prediction(answer),
        }
        rows = (
            _complete_strict_progress(progress_path, header=header, row=row)
            if progress_path is not None
            else [*rows, row]
        )
    return [_phase_detail(case, row) for case, row in zip(cases, rows, strict=True)]


def _resolve_template_assignments(
    cases: list[dict[str, Any]],
    assignments: dict[str, str] | None,
) -> dict[str, str]:
    case_ids = [str(case["case_id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("E6 template assignment received duplicate case IDs")
    if assignments is None:
        resolved = {
            str(case["case_id"]): _question_template(str(case["question"])) for case in cases
        }
    else:
        resolved = {str(case_id): str(template_id) for case_id, template_id in assignments.items()}
    if set(resolved) != set(case_ids):
        raise RuntimeError("E6 semantic-template assignment coverage changed")
    if any(not template_id.strip() for template_id in resolved.values()):
        raise RuntimeError("E6 semantic-template assignment contains an empty template ID")
    exact_templates: dict[str, str] = {}
    for case in cases:
        normalized_question = _question_template(str(case["question"]))
        template_id = resolved[str(case["case_id"])]
        previous = exact_templates.setdefault(normalized_question, template_id)
        if previous != template_id:
            raise RuntimeError("E6 duplicate messages were split across semantic templates")
    return resolved


def _template_accounting(
    control: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    template_assignments: dict[str, str] | None = None,
) -> dict[str, Any]:
    candidate_by_id = {str(row["case_id"]): row for row in candidate}
    assignments = _resolve_template_assignments(control, template_assignments)
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for row in control:
        grouped.setdefault(assignments[str(row["case_id"])], []).append(
            (row, candidate_by_id[str(row["case_id"])])
        )
    counts: Counter[str] = Counter()
    details = []
    repaired_methods: set[str] = set()
    for template, pairs in sorted(grouped.items()):
        methods = {str(pair[0]["gold_method_id"]) for pair in pairs}
        if len(methods) != 1:
            raise RuntimeError("E6 normalized-identical questions have conflicting gold methods")
        controls = [bool(pair[0]["correct"]) for pair in pairs]
        candidates = [bool(pair[1]["correct"]) for pair in pairs]
        if all(controls) and all(candidates):
            state = "both_correct"
        elif not any(controls) and all(candidates):
            state = "candidate_only"
            repaired_methods.update(methods)
        elif all(controls) and not all(candidates):
            state = "control_only"
        else:
            state = "mixed_or_both_wrong"
        counts[state] += 1
        details.append(
            {
                "template": template,
                "case_count": len(pairs),
                "gold_method_id": next(iter(methods)),
                "state": state,
            }
        )
    candidate_only = int(counts["candidate_only"])
    control_only = int(counts["control_only"])
    return {
        "template_count": len(details),
        "candidate_only": candidate_only,
        "control_only": control_only,
        "net_improvements": candidate_only - control_only,
        "mcnemar_exact_two_sided_p": _mcnemar_exact_pvalue(candidate_only, control_only),
        "repaired_gold_methods": sorted(repaired_methods),
        "maximum_single_repaired_method_fraction": (
            max(
                Counter(
                    row["gold_method_id"] for row in details if row["state"] == "candidate_only"
                ).values()
            )
            / candidate_only
            if candidate_only
            else 1.0
        ),
        "details": details,
    }


def _external_gate(
    *,
    control: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    paired: dict[str, Any],
    templates: dict[str, Any],
    template_assignments: dict[str, str] | None,
    opportunity_gates: dict[str, Any],
    result_gates: dict[str, Any],
) -> dict[str, Any]:
    invalid_controls = [row for row in control if not bool(row["valid_output"])]
    valid_controls = [row for row in control if bool(row["valid_output"])]
    assignments = _resolve_template_assignments(control, template_assignments)
    invalid_template_counts = Counter(assignments[str(row["case_id"])] for row in invalid_controls)
    invalid_method_case_counts = Counter(str(row["gold_method_id"]) for row in invalid_controls)
    invalid_template_methods: dict[str, str] = {}
    for row in invalid_controls:
        template_id = assignments[str(row["case_id"])]
        method_id = str(row["gold_method_id"])
        previous = invalid_template_methods.setdefault(template_id, method_id)
        if previous != method_id:
            raise RuntimeError("E6 semantic template has conflicting invalid-control gold methods")
    invalid_method_template_counts = Counter(invalid_template_methods.values())
    maximum_template_share = (
        max(invalid_template_counts.values()) / len(invalid_controls) if invalid_controls else 1.0
    )
    maximum_method_case_share = (
        max(invalid_method_case_counts.values()) / len(invalid_controls)
        if invalid_controls
        else 1.0
    )
    maximum_method_template_share = (
        max(invalid_method_template_counts.values()) / len(invalid_template_methods)
        if invalid_template_methods
        else 1.0
    )
    opportunity_observed = {
        "invalid_control_opportunities": len(invalid_controls),
        "invalid_control_gold_methods": len(
            {str(row["gold_method_id"]) for row in invalid_controls}
        ),
        "distinct_invalid_question_templates": len(invalid_template_counts),
        "maximum_single_invalid_template_fraction": maximum_template_share,
        "maximum_single_invalid_method_case_fraction": maximum_method_case_share,
        "maximum_single_invalid_method_template_fraction": maximum_method_template_share,
        "valid_control_identity_cases": len(valid_controls),
    }
    opportunity_checks = {
        "minimum_invalid_control_opportunities": len(invalid_controls)
        >= int(opportunity_gates["minimum_invalid_control_opportunities"]),
        "minimum_invalid_control_gold_methods": opportunity_observed["invalid_control_gold_methods"]
        >= int(opportunity_gates["minimum_invalid_control_gold_methods"]),
        "minimum_distinct_invalid_question_templates": opportunity_observed[
            "distinct_invalid_question_templates"
        ]
        >= int(opportunity_gates["minimum_distinct_invalid_question_templates"]),
        "maximum_single_invalid_template_fraction": maximum_template_share
        <= float(opportunity_gates["maximum_single_invalid_template_fraction"]),
        "maximum_single_invalid_method_case_fraction": maximum_method_case_share
        <= float(opportunity_gates["maximum_single_invalid_method_case_fraction"]),
        "maximum_single_invalid_method_template_fraction": maximum_method_template_share
        <= float(opportunity_gates["maximum_single_invalid_method_template_fraction"]),
        "minimum_valid_control_identity_cases": len(valid_controls)
        >= int(opportunity_gates["minimum_valid_control_identity_cases"]),
    }
    valid_fallbacks = [row for row in fallback if bool(row["valid_output"])]
    correct_fallbacks = [row for row in valid_fallbacks if bool(row["correct"])]
    valid_but_wrong_fallbacks = len(valid_fallbacks) - len(correct_fallbacks)
    precision = len(correct_fallbacks) / len(valid_fallbacks) if valid_fallbacks else 0.0
    control_metrics = _strict_metrics(control)
    candidate_metrics = _strict_metrics(candidate)
    gain_points = 100.0 * (candidate_metrics["accuracy"] - control_metrics["accuracy"])
    required_case_gains = max(
        int(result_gates["minimum_candidate_only_gains"]),
        math.ceil(0.04 * len(control)),
    )
    result_checks = {
        "zero_control_only_losses": int(paired["control_only"]) == 0,
        "minimum_candidate_only_gains": int(paired["candidate_only"]) >= required_case_gains,
        "minimum_net_improvements": int(paired["net_improvements"])
        >= int(result_gates["minimum_net_improvements"]),
        "minimum_accuracy_gain_points": gain_points
        >= float(result_gates["minimum_accuracy_gain_points"]),
        "maximum_case_mcnemar_p": float(paired["mcnemar_exact_two_sided_p"])
        <= float(result_gates["maximum_case_mcnemar_p"]),
        "minimum_distinct_repaired_question_templates": int(templates["candidate_only"])
        >= int(result_gates["minimum_distinct_repaired_question_templates"]),
        "zero_template_control_only_losses": int(templates["control_only"]) == 0,
        "minimum_template_net_improvements": int(templates["net_improvements"])
        >= int(result_gates["minimum_template_net_improvements"]),
        "minimum_repaired_gold_methods": len(templates["repaired_gold_methods"])
        >= int(result_gates["minimum_repaired_gold_methods"]),
        "maximum_single_repaired_method_fraction": float(
            templates["maximum_single_repaired_method_fraction"]
        )
        <= float(result_gates["maximum_single_repaired_method_fraction"]),
        "maximum_template_mcnemar_p": float(templates["mcnemar_exact_two_sided_p"])
        <= float(result_gates["maximum_template_mcnemar_p"]),
        "minimum_valid_fallback_precision": precision
        >= float(result_gates["minimum_valid_fallback_precision"]),
        "maximum_valid_but_wrong_fallbacks": valid_but_wrong_fallbacks
        <= int(result_gates["maximum_valid_but_wrong_fallbacks"]),
    }
    if not all(opportunity_checks.values()):
        status = "INCONCLUSIVE_OPPORTUNITY"
    elif all(result_checks.values()):
        status = "CONFIRMED_NARROW_FRESH_SOURCE_PASS"
    else:
        status = "SCIENTIFIC_FAIL"
    return {
        "status": status,
        "opportunity_passed": all(opportunity_checks.values()),
        "result_passed": status == "CONFIRMED_NARROW_FRESH_SOURCE_PASS",
        "opportunity_checks": opportunity_checks,
        "result_checks": result_checks,
        "opportunity_observed": opportunity_observed,
        "result_observed": {
            "accuracy_gain_points": gain_points,
            "valid_fallback_count": len(valid_fallbacks),
            "correct_fallback_count": len(correct_fallbacks),
            "valid_but_wrong_fallback_count": valid_but_wrong_fallbacks,
            "valid_fallback_precision": precision,
            "required_case_gain_count": required_case_gains,
            "case_candidate_only_gains": paired["candidate_only"],
            "template_candidate_only_gains": templates["candidate_only"],
            "template_net_improvements": templates["net_improvements"],
            "template_control_only_losses": templates["control_only"],
            "repaired_gold_method_count": len(templates["repaired_gold_methods"]),
            "maximum_single_repaired_method_fraction": templates[
                "maximum_single_repaired_method_fraction"
            ],
        },
    }


def _preflight_two_stage_progress(
    cases: list[dict[str, Any]],
    *,
    evaluation_fingerprint: str,
    control_runtime_receipt: dict[str, Any],
    fallback_runtime_receipt: dict[str, Any],
    progress_root: Path | None,
) -> None:
    if progress_root is None:
        return
    control_header = _phase_header(
        cases,
        phase="control",
        grounded=False,
        evaluation_fingerprint=evaluation_fingerprint,
        runtime_receipt=control_runtime_receipt,
    )
    control_path = progress_root / "control-ledger.json"
    control_rows = (
        _load_strict_progress(control_path, header=control_header) if control_path.exists() else []
    )
    fallback_path = progress_root / "fallback-ledger.json"
    if not fallback_path.exists():
        return
    if len(control_rows) != len(cases):
        raise RuntimeError(
            "E6 fallback ledger exists before the control phase was completely sealed"
        )
    control = [_phase_detail(case, row) for case, row in zip(cases, control_rows, strict=True)]
    invalid_cases = [
        case for case, row in zip(cases, control, strict=True) if not bool(row["valid_output"])
    ]
    if not invalid_cases:
        raise RuntimeError("E6 fallback ledger exists without any invalid controls")
    fallback_header = _phase_header(
        invalid_cases,
        phase="catalog-fallback",
        grounded=True,
        evaluation_fingerprint=evaluation_fingerprint,
        runtime_receipt=fallback_runtime_receipt,
    )
    _load_strict_progress(fallback_path, header=fallback_header)


def run_catalog_fallback_two_stage(
    cases: list[dict[str, Any]],
    *,
    evaluation_fingerprint: str,
    control_runtime_receipt: dict[str, Any],
    fallback_runtime_receipt: dict[str, Any],
    control_caller: ModelCaller,
    fallback_caller: ModelCaller,
    before_fallback: PhaseHook | None = None,
    progress_root: Path | None = None,
    template_assignments: dict[str, str] | None = None,
    opportunity_gates: dict[str, Any] | None = None,
    result_gates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not cases:
        raise RuntimeError("E6 exact runtime has no cases")
    if control_runtime_receipt.get("adapter_sha256") != fallback_runtime_receipt.get(
        "adapter_sha256"
    ):
        raise RuntimeError("E6 control and fallback must use the same unchanged parent")
    if control_runtime_receipt.get("prompt_sha256") == fallback_runtime_receipt.get(
        "prompt_sha256"
    ):
        raise RuntimeError("E6 control and fallback prompts are indistinguishable")
    control_runtime_core = {
        key: value
        for key, value in control_runtime_receipt.items()
        if key not in {"phase", "prompt_sha256"}
    }
    fallback_runtime_core = {
        key: value
        for key, value in fallback_runtime_receipt.items()
        if key not in {"phase", "prompt_sha256"}
    }
    if control_runtime_core != fallback_runtime_core:
        raise RuntimeError("E6 control and fallback differ by more than the fixed catalog prompt")
    assignments = _resolve_template_assignments(cases, template_assignments)
    _preflight_two_stage_progress(
        cases,
        evaluation_fingerprint=evaluation_fingerprint,
        control_runtime_receipt=control_runtime_receipt,
        fallback_runtime_receipt=fallback_runtime_receipt,
        progress_root=progress_root,
    )
    control = _evaluate_phase(
        cases,
        phase="control",
        grounded=False,
        evaluation_fingerprint=evaluation_fingerprint,
        runtime_receipt=control_runtime_receipt,
        caller=control_caller,
        progress_path=None if progress_root is None else progress_root / "control-ledger.json",
    )
    if before_fallback is not None:
        before_fallback()
    control_by_id = {str(row["case_id"]): row for row in control}
    invalid_cases = [
        case for case in cases if not bool(control_by_id[str(case["case_id"])]["valid_output"])
    ]
    fallback = _evaluate_phase(
        invalid_cases,
        phase="catalog-fallback",
        grounded=True,
        evaluation_fingerprint=evaluation_fingerprint,
        runtime_receipt=fallback_runtime_receipt,
        caller=fallback_caller,
        progress_path=None if progress_root is None else progress_root / "fallback-ledger.json",
    )
    fallback_by_id = {str(row["case_id"]): row for row in fallback}
    if len(fallback_by_id) != len(fallback) or set(fallback_by_id) != {
        str(case["case_id"]) for case in invalid_cases
    }:
        raise RuntimeError("E6 fallback call coverage changed")
    candidate = [
        _apply_catalog_guard(
            control_by_id[str(case["case_id"])], fallback_by_id.get(str(case["case_id"]))
        )
        for case in cases
    ]
    paired = _paired(control, candidate)
    if int(paired["control_only"]) != 0:
        raise RuntimeError("E6 structural identity guard produced a control-only loss")
    candidate_by_id = {str(row["case_id"]): row for row in candidate}
    valid_controls = [row for row in control if bool(row["valid_output"])]
    valid_identity_count = 0
    for row in valid_controls:
        selected = candidate_by_id[str(row["case_id"])]
        if (
            selected["route"] == "valid-control-identity"
            and selected["predicted_method_id"] == row["predicted_method_id"]
            and bool(selected["valid_output"]) is bool(row["valid_output"])
            and bool(selected["correct"]) is bool(row["correct"])
            and selected["control_raw_answer_sha256"] == row["raw_answer_sha256"]
        ):
            valid_identity_count += 1
    if valid_identity_count != len(valid_controls):
        raise RuntimeError("E6 valid controls were not preserved byte-for-byte")
    templates = _template_accounting(control, candidate, assignments)
    gate = _external_gate(
        control=control,
        fallback=fallback,
        candidate=candidate,
        paired=paired,
        templates=templates,
        template_assignments=assignments,
        opportunity_gates=_OPPORTUNITY_GATES if opportunity_gates is None else opportunity_gates,
        result_gates=_RESULT_GATES if result_gates is None else result_gates,
    )
    template_summary = {key: value for key, value in templates.items() if key != "details"}
    private_details = {
        "control": control,
        "fallback": fallback,
        "candidate": candidate,
        "semantic_template_assignments": assignments,
        "semantic_template_details": templates["details"],
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": _METHOD,
        "evaluation_fingerprint": evaluation_fingerprint,
        "status": gate["status"],
        "same_parent_both_stages": True,
        "fixed_catalog_is_only_prompt_change": True,
        "scores": {
            "control": _strict_metrics(control),
            "catalog_guard": _strict_metrics(candidate),
        },
        "paired": paired,
        "source_paired": _source_paired(control, candidate),
        "template_paired": template_summary,
        "external_gate": gate,
        "model_call_counts": {
            "control_calls": len(control),
            "control_invalid_count": len(invalid_cases),
            "fallback_calls": len(fallback),
            "valid_control_fallback_calls": 0,
            "total_calls": len(control) + len(fallback),
        },
        "runtime_integrity": {
            "all_controls_completed_before_fallback": True,
            "control_model_release_hook_called": before_fallback is not None,
            "valid_control_identity_count": valid_identity_count,
            "valid_control_identity_fraction": valid_identity_count / len(valid_controls)
            if valid_controls
            else 1.0,
            "fallback_calls_per_invalid_control": 1.0,
            "valid_control_fallback_calls": 0,
            "zero_control_only_losses": True,
        },
        "training_authorized": False,
        "champion_changed": False,
        "release_authorized": False,
        "private_details_fingerprint": canonical_hash(private_details),
        "private_details": private_details,
    }
    unsigned = {
        key: value
        for key, value in report.items()
        if key not in {"result_fingerprint", "private_details"}
    }
    report["result_fingerprint"] = canonical_hash(unsigned)
    return report


def _verify_master(master: dict[str, Any]) -> None:
    unsigned = {key: value for key, value in master.items() if key != "fingerprint"}
    if master.get("fingerprint") != canonical_hash(unsigned):
        raise RuntimeError("E6 master contract fingerprint changed")
    runtime = master.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError("E6 runtime receipt is missing")
    runtime_unsigned = {key: value for key, value in runtime.items() if key != "fingerprint"}
    if runtime.get("fingerprint") != canonical_hash(runtime_unsigned):
        raise RuntimeError("E6 runtime fingerprint changed")
    if runtime.get("decoding") != _DECODING:
        raise RuntimeError("E6 decoding changed")
    if runtime.get("control_prompt_sha256") != canonical_hash(
        _menu_free_messages({"question": "<SOURCE_QUESTION>"})
    ):
        raise RuntimeError("E6 control prompt changed")
    if runtime.get("fallback_prompt_sha256") != canonical_hash(
        _fallback_messages({"question": "<SOURCE_QUESTION>"})
    ):
        raise RuntimeError("E6 fallback prompt changed")
    parent = dict(runtime["parent"])
    parent_path = Path(str(parent["adapter_path"]))
    if sha256_file(parent_path / "adapters.safetensors") != parent["adapter_sha256"]:
        raise RuntimeError("E6 parent weights changed")
    if sha256_file(parent_path / "adapter_config.json") != parent["adapter_config_sha256"]:
        raise RuntimeError("E6 parent adapter config changed")
    base = dict(runtime["base_model"])
    snapshot = Path(str(base["snapshot_path"]))
    for name, expected in base["file_sha256"].items():
        if sha256_file(snapshot / str(name)) != expected:
            raise RuntimeError(f"E6 base-model runtime changed: {name}")
    source_root = Path(__file__).resolve().parent
    for name, expected in master["implementation_sha256"].items():
        if sha256_file(source_root / str(name)) != expected:
            raise RuntimeError(f"E6 implementation changed: {name}")


def _require_within(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise RuntimeError(f"E6 {label} is outside the frozen selected-source directory")
    return resolved


def _verify_e6_child_contract(child: dict[str, Any], master: dict[str, Any]) -> None:
    unsigned = {key: value for key, value in child.items() if key != "fingerprint"}
    if child.get("fingerprint") != canonical_hash(unsigned):
        raise RuntimeError("E6 selected-source contract fingerprint changed")
    if child.get("master_contract_fingerprint") != master.get("fingerprint"):
        raise RuntimeError("E6 selected-source contract belongs to another master")
    if child.get("runtime_fingerprint") != master.get("runtime", {}).get("fingerprint"):
        raise RuntimeError("E6 selected-source runtime changed")
    if child.get("implementation_sha256") != master.get("implementation_sha256"):
        raise RuntimeError("E6 selected-source implementation changed")
    if (
        not child.get("source_selected_before_opening")
        or not child.get("metadata_prequalified")
        or child.get("dataset_rows_opened")
    ):
        raise RuntimeError("E6 selected source was not frozen from metadata before opening")
    metadata_receipt = child.get("metadata_qualification_receipt")
    if not isinstance(metadata_receipt, dict) or child.get(
        "metadata_qualification_fingerprint"
    ) != canonical_hash(metadata_receipt):
        raise RuntimeError("E6 selected-source metadata qualification changed")
    bundle_path = Path(str(metadata_receipt.get("bundle_path", "")))
    if not bundle_path.is_file() or sha256_file(bundle_path) != metadata_receipt.get(
        "bundle_sha256"
    ):
        raise RuntimeError("E6 selected-source metadata bundle changed")
    qualification_checks = metadata_receipt.get("checks")
    if not isinstance(qualification_checks, dict) or not all(
        bool(value) for value in qualification_checks.values()
    ):
        raise RuntimeError("E6 selected-source metadata qualification no longer passes")
    source = child.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("E6 selected-source identity is missing")
    if source.get("license") not in master["source_protocol"]["permissive_license_allowlist"]:
        raise RuntimeError("E6 selected-source license is outside the frozen allowlist")
    for field in (
        "source_id",
        "stable_id",
        "revision",
        "license_url",
        "snapshot_url",
        "snapshot_path",
        "snapshot_sha256",
    ):
        if not str(source.get(field, "")).strip():
            raise RuntimeError(f"E6 selected-source {field} is missing")
    if str(source["source_id"]) in master["source_protocol"]["opened_source_exclusions"]:
        raise RuntimeError("E6 selected source was already opened during development")
    alias_map = source.get("alias_map")
    if not isinstance(alias_map, dict) or source.get("alias_map_fingerprint") != canonical_hash(
        alias_map
    ):
        raise RuntimeError("E6 selected-source alias map changed")
    if any(
        not isinstance(label, str)
        or label != " ".join(label.split())
        or not label
        or method_id not in _METHOD_IDS
        for label, method_id in alias_map.items()
    ):
        raise RuntimeError("E6 selected-source alias map leaves the frozen catalog")
    extraction = source.get("extraction_contract")
    if not isinstance(extraction, dict) or source.get(
        "extraction_contract_fingerprint"
    ) != canonical_hash(extraction):
        raise RuntimeError("E6 selected-source extraction contract changed")
    overlap_contract = child.get("overlap_contract")
    if not isinstance(overlap_contract, dict) or child.get(
        "overlap_contract_fingerprint"
    ) != canonical_hash(overlap_contract):
        raise RuntimeError("E6 selected-source overlap contract changed")
    if overlap_contract.get("normalization") != "casefold-whitespace-sha256-v1":
        raise RuntimeError("E6 selected-source overlap normalization changed")
    for field in (
        "historical_question_hashes_path",
        "historical_question_hashes_sha256",
        "historical_corpus_fingerprint",
    ):
        if not str(overlap_contract.get(field, "")):
            raise RuntimeError(f"E6 selected-source overlap {field} is missing")
    historical_path = Path(str(overlap_contract["historical_question_hashes_path"]))
    if (
        not historical_path.is_file()
        or sha256_file(historical_path) != overlap_contract["historical_question_hashes_sha256"]
    ):
        raise RuntimeError("E6 selected-source historical question-hash corpus changed")
    historical_hashes = _load_historical_question_hashes(historical_path)
    if canonical_hash(historical_hashes) != overlap_contract["historical_corpus_fingerprint"]:
        raise RuntimeError("E6 selected-source historical corpus fingerprint changed")
    projection = extraction.get("e6_projection") if isinstance(extraction, dict) else None
    if not isinstance(projection, dict):
        raise RuntimeError("E6 selected-source extraction contract lacks its frozen projection")
    expected_projection_fields = {
        "snapshot_format",
        "records_path",
        "record_id_field",
        "case_id_field",
        "question_field",
        "decision_frame_fields",
        "decision_frame_roles",
        "decision_frame_allowed_values",
        "answer_fields",
        "source_method_field",
        "mapped_method_field",
        "record_filter",
        "case_id_rule",
        "normalization",
        "template_id_rule",
        "question_materialization_rule",
        "answer_in_question_policy",
    }
    if set(projection) != expected_projection_fields:
        raise RuntimeError("E6 selected-source projection schema changed")
    if projection["case_id_field"] != "case_id" or projection["mapped_method_field"] != (
        "mapped_method_id"
    ):
        raise RuntimeError("E6 selected-source projection changed required frame identity fields")
    for key, expected in (
        ("record_filter", "all-nonempty-source-method-records"),
        ("case_id_rule", "sha256-source-id-and-record-id"),
        ("normalization", "recursive-casefold-whitespace-v1"),
        ("template_id_rule", "canonical-sha256-of-normalized-decision-frame"),
        ("question_materialization_rule", "normalized-question-field-only"),
        (
            "answer_in_question_policy",
            "reject-source-label-or-canonical-method-verbatim",
        ),
    ):
        if projection[key] != expected:
            raise RuntimeError(f"E6 selected-source projection changed {key}")
    question_field = str(projection["question_field"])
    record_id_field = str(projection["record_id_field"])
    source_method_field = str(projection["source_method_field"])
    decision_fields = projection["decision_frame_fields"]
    decision_roles = projection["decision_frame_roles"]
    decision_allowed_values = projection["decision_frame_allowed_values"]
    answer_fields = projection["answer_fields"]
    if (
        projection["snapshot_format"] not in {"json-array", "jsonl-records"}
        or not question_field
        or not record_id_field
        or not source_method_field
        or not isinstance(decision_fields, list)
        or not decision_fields
        or len(decision_fields) > _MAX_DECISION_FRAME_FIELDS
        or not isinstance(decision_roles, dict)
        or set(decision_roles) != set(decision_fields)
        or set(decision_roles.values()) - _DECISION_FRAME_ROLES
        or len(set(decision_roles.values())) != len(decision_roles)
        or not isinstance(decision_allowed_values, dict)
        or set(decision_allowed_values) != set(decision_fields)
        or not isinstance(answer_fields, list)
        or not answer_fields
        or any(
            not isinstance(field, str) or not field for field in [*decision_fields, *answer_fields]
        )
    ):
        raise RuntimeError("E6 selected-source projection fields are incomplete")
    for field in decision_fields:
        values = decision_allowed_values[field]
        if not isinstance(values, list) or not values or len(values) > _MAX_DECISION_FIELD_VALUES:
            raise RuntimeError("E6 decision-frame categorical value set is invalid")
        normalized = [_normalize_frame_value(value) for value in values]
        serialized = [canonical_hash(value) for value in normalized]
        if len(serialized) != len(set(serialized)):
            raise RuntimeError("E6 decision-frame allowed values must be unique")
    expected_suffix = ".jsonl" if projection["snapshot_format"] == "jsonl-records" else ".json"
    if Path(str(source["snapshot_path"])).suffix != expected_suffix:
        raise RuntimeError("E6 selected-source snapshot path disagrees with its frozen format")
    records_path = projection["records_path"]
    if not isinstance(records_path, str) or (
        projection["snapshot_format"] == "jsonl-records" and records_path
    ):
        raise RuntimeError("E6 selected-source record path is invalid for its snapshot format")
    forbidden_decision_fields = {
        record_id_field,
        question_field,
        source_method_field,
        str(projection["mapped_method_field"]),
        *[str(field) for field in answer_fields],
    }
    if forbidden_decision_fields & {str(field) for field in decision_fields}:
        raise RuntimeError("E6 semantic decision frame includes question, answer, or gold fields")
    identity_fields = {
        record_id_field,
        question_field,
        source_method_field,
        str(projection["mapped_method_field"]),
    }
    if (
        len(identity_fields) != 4
        or len(answer_fields) != len(set(answer_fields))
        or source_method_field not in {str(field) for field in answer_fields}
        or {record_id_field, question_field, str(projection["mapped_method_field"])}
        & {str(field) for field in answer_fields}
    ):
        raise RuntimeError("E6 record, question, answer, and mapped-method roles overlap")


def _field_at(row: dict[str, Any], path: str) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise RuntimeError(f"E6 complete-frame field is missing: {path}")
        value = value[part]
    return value


def _normalize_frame_value(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    if isinstance(value, dict):
        return {str(key): _normalize_frame_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_frame_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise RuntimeError("E6 complete-frame projection contains an unsupported value")


def _load_source_records(snapshot_path: Path, projection: dict[str, Any]) -> list[dict[str, Any]]:
    if projection["snapshot_format"] == "jsonl-records":
        records = list(read_jsonl(snapshot_path))
    else:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        records_path = str(projection["records_path"])
        records = payload if not records_path else _field_at(payload, records_path)
    if (
        not isinstance(records, list)
        or not records
        or any(not isinstance(record, dict) for record in records)
    ):
        raise RuntimeError("E6 selected-source snapshot does not contain a nonempty record list")
    return [dict(record) for record in records]


def _recompute_complete_frame(child: dict[str, Any]) -> list[dict[str, Any]]:
    source = child["source"]
    source_id = str(source["source_id"])
    snapshot_path = Path(str(source["snapshot_path"]))
    projection = source["extraction_contract"]["e6_projection"]
    aliases = source["alias_map"]
    records = _load_source_records(snapshot_path, projection)
    seen_record_ids: set[str] = set()
    frame: list[dict[str, Any]] = []
    for record in records:
        raw_source_label = _field_at(record, str(projection["source_method_field"]))
        source_label = (
            " ".join(str(raw_source_label).split()) if raw_source_label is not None else ""
        )
        if not source_label:
            continue
        raw_record_id = _field_at(record, str(projection["record_id_field"]))
        record_id = " ".join(str(raw_record_id).split())
        if not record_id or record_id in seen_record_ids:
            raise RuntimeError("E6 selected-source record IDs are empty or duplicated")
        seen_record_ids.add(record_id)
        raw_question = _field_at(record, str(projection["question_field"]))
        if not isinstance(raw_question, str) or not raw_question.strip():
            raise RuntimeError("E6 selected-source question field is empty or non-text")
        question = " ".join(raw_question.split())
        normalized_answer_values: list[str] = []
        for answer_field in projection["answer_fields"]:
            answer_value = _field_at(record, str(answer_field))
            if isinstance(answer_value, str) and answer_value.strip():
                normalized_answer_values.append(" ".join(answer_value.casefold().split()))
        mapped_method = aliases.get(source_label)
        if mapped_method is not None and mapped_method not in _METHOD_IDS:
            raise RuntimeError("E6 selected-source alias maps outside the frozen catalog")
        normalized_question = " ".join(question.casefold().split())
        normalized_label = " ".join(source_label.casefold().split())
        normalized_method = (
            " ".join(str(mapped_method).replace("_", " ").casefold().split())
            if mapped_method is not None
            else ""
        )
        if (
            normalized_label in normalized_question
            or (normalized_method and normalized_method in normalized_question)
            or any(value in normalized_question for value in normalized_answer_values)
        ):
            raise RuntimeError("E6 selected-source question contains its answer or gold method")
        decision_frame: dict[str, Any] = {}
        for field in projection["decision_frame_fields"]:
            normalized_value = _normalize_frame_value(_field_at(record, str(field)))
            allowed = [
                _normalize_frame_value(value)
                for value in projection["decision_frame_allowed_values"][str(field)]
            ]
            if canonical_hash(normalized_value) not in {canonical_hash(value) for value in allowed}:
                raise RuntimeError(
                    "E6 selected-source decision-frame value is outside its metadata-frozen "
                    "categorical set"
                )
            role = str(projection["decision_frame_roles"][str(field)])
            decision_frame[role] = normalized_value
        identity = {"source_id": source_id, "record_id": record_id}
        frame_id = canonical_hash(identity)
        case_id = f"e6-{frame_id[:24]}" if mapped_method is not None else None
        frame.append(
            {
                "frame_id": frame_id,
                "case_id": case_id,
                "source_record_id_sha256": sha256_text(record_id),
                "question": question,
                "decision_frame": decision_frame,
                "source_label": source_label,
                "mapped_method_id": mapped_method,
            }
        )
    if not frame:
        raise RuntimeError("E6 selected-source complete frame is empty")
    return frame


def _verify_semantic_template_manifest(
    data: dict[str, Any],
    child: dict[str, Any],
    master: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    selected_root: Path,
) -> dict[str, str]:
    expected_contract = master["source_protocol"]["semantic_template_contract"]
    template_contract = child.get("semantic_template_contract")
    if template_contract != expected_contract:
        raise RuntimeError("E6 selected-source semantic-template contract changed")
    contract_fingerprint = canonical_hash(expected_contract)
    if child.get("semantic_template_contract_fingerprint") != contract_fingerprint:
        raise RuntimeError("E6 selected-source semantic-template contract fingerprint changed")
    if data.get("semantic_template_contract_fingerprint") != contract_fingerprint:
        raise RuntimeError("E6 data semantic-template contract changed")

    manifest_path = _require_within(
        Path(str(data.get("semantic_template_manifest_path", ""))),
        selected_root,
        label="semantic-template manifest",
    )
    if sha256_file(manifest_path) != data.get("semantic_template_manifest_sha256"):
        raise RuntimeError("E6 semantic-template manifest changed")
    manifest = list(read_jsonl(manifest_path))
    if canonical_hash(manifest) != data.get("semantic_template_manifest_fingerprint"):
        raise RuntimeError("E6 semantic-template manifest fingerprint changed")
    expected_fields = {
        "case_id",
        "semantic_template_id",
        "normalized_frame_fingerprint",
        "representative_case_id",
    }
    if not manifest or any(set(row) != expected_fields for row in manifest):
        raise RuntimeError("E6 semantic-template manifest schema changed")
    case_by_id = {str(case["case_id"]): case for case in cases}
    case_ids = [str(row["case_id"]) for row in manifest]
    if len(case_ids) != len(set(case_ids)) or set(case_ids) != set(case_by_id):
        raise RuntimeError("E6 semantic-template manifest case coverage changed")

    frame_path = _require_within(
        Path(str(data.get("complete_frame_path", ""))),
        selected_root,
        label="complete source frame",
    )
    if sha256_file(frame_path) != data.get("complete_frame_sha256"):
        raise RuntimeError("E6 complete source frame changed during template verification")
    frame = list(read_jsonl(frame_path))
    if canonical_hash(frame) != data.get("complete_frame_fingerprint"):
        raise RuntimeError("E6 complete source frame fingerprint changed")
    recomputed_frame = _recompute_complete_frame(child)
    if frame != recomputed_frame:
        raise RuntimeError("E6 complete source frame differs from the frozen snapshot extraction")
    frame_by_case: dict[str, dict[str, Any]] = {}
    for row in frame:
        frame_case_id = row["case_id"]
        if frame_case_id is None:
            continue
        case_id = str(frame_case_id)
        if case_id in frame_by_case:
            raise RuntimeError("E6 complete source frame duplicates a case ID")
        frame_by_case[case_id] = row
    if set(frame_by_case) & set(case_by_id) != set(case_by_id):
        raise RuntimeError("E6 complete source frame lost materialized cases")

    manifest_by_case = {str(row["case_id"]): row for row in manifest}
    for case_id, case in case_by_id.items():
        frame_row = frame_by_case[case_id]
        if str(case["question"]) != frame_row["question"]:
            raise RuntimeError(
                "E6 materialized question differs from the frozen question-only field"
            )
        if frame_row["mapped_method_id"] != str(case["gold_method_id"]):
            raise RuntimeError("E6 materialized gold differs from the frozen mapped-method field")
        normalized_decision_frame = _normalize_frame_value(frame_row["decision_frame"])
        expected_frame_fingerprint = canonical_hash(normalized_decision_frame)
        manifest_row = manifest_by_case[case_id]
        if (
            manifest_row["normalized_frame_fingerprint"] != expected_frame_fingerprint
            or manifest_row["semantic_template_id"] != expected_frame_fingerprint
        ):
            raise RuntimeError(
                "E6 semantic-template fingerprint was not derived from the frozen complete frame"
            )

    assignments = {str(row["case_id"]): str(row["semantic_template_id"]) for row in manifest}
    assignments = _resolve_template_assignments(cases, assignments)
    if data.get("semantic_template_assignment_fingerprint") != canonical_hash(assignments):
        raise RuntimeError("E6 semantic-template assignment fingerprint changed")

    rows_by_template: dict[str, list[dict[str, Any]]] = {}
    for row in manifest:
        template_id = str(row["semantic_template_id"])
        frame_fingerprint = str(row["normalized_frame_fingerprint"])
        representative = str(row["representative_case_id"])
        if not template_id or not frame_fingerprint or not representative:
            raise RuntimeError("E6 semantic-template manifest contains an empty field")
        rows_by_template.setdefault(template_id, []).append(row)
    frame_to_template: dict[str, str] = {}
    for template_id, rows in rows_by_template.items():
        frame_fingerprints = {str(row["normalized_frame_fingerprint"]) for row in rows}
        if len(frame_fingerprints) != 1:
            raise RuntimeError("E6 semantic template spans multiple normalized decision frames")
        frame_fingerprint = next(iter(frame_fingerprints))
        previous = frame_to_template.setdefault(frame_fingerprint, template_id)
        if previous != template_id:
            raise RuntimeError("E6 one normalized decision frame was split across templates")
        cluster_case_ids = sorted(str(row["case_id"]) for row in rows)
        representative = min(cluster_case_ids, key=lambda case_id: (sha256_text(case_id), case_id))
        if {str(row["representative_case_id"]) for row in rows} != {representative}:
            raise RuntimeError("E6 semantic-template representative rule changed")
        gold_methods = {str(case_by_id[case_id]["gold_method_id"]) for case_id in cluster_case_ids}
        if len(gold_methods) != 1:
            raise RuntimeError("E6 semantic template has conflicting gold methods")
    return assignments


def _terminal_fingerprint(report: dict[str, Any]) -> str:
    unsigned = {
        key: value
        for key, value in report.items()
        if key not in {"result_fingerprint", "private_details"}
    }
    return canonical_hash(unsigned)


def _verify_terminal_report(
    report: dict[str, Any],
    *,
    opportunity_gates: dict[str, Any],
    result_gates: dict[str, Any],
) -> None:
    if not report.get("complete") or report.get("method") != _METHOD:
        raise RuntimeError("E6 terminal report is incomplete or belongs to another method")
    if report.get("result_fingerprint") != _terminal_fingerprint(report):
        raise RuntimeError("E6 terminal result fingerprint changed")
    private = report.get("private_details")
    if not isinstance(private, dict) or report.get("private_details_fingerprint") != canonical_hash(
        private
    ):
        raise RuntimeError("E6 terminal private details changed")
    control = private.get("control")
    fallback = private.get("fallback")
    candidate = private.get("candidate")
    assignments = private.get("semantic_template_assignments")
    if not all(isinstance(rows, list) for rows in (control, fallback, candidate)) or not isinstance(
        assignments, dict
    ):
        raise RuntimeError("E6 terminal private details are malformed")
    paired = _paired(control, candidate)
    templates = _template_accounting(control, candidate, assignments)
    template_summary = {key: value for key, value in templates.items() if key != "details"}
    gate = _external_gate(
        control=control,
        fallback=fallback,
        candidate=candidate,
        paired=paired,
        templates=templates,
        template_assignments=assignments,
        opportunity_gates=opportunity_gates,
        result_gates=result_gates,
    )
    expected_call_counts = {
        "control_calls": len(control),
        "control_invalid_count": sum(not bool(row["valid_output"]) for row in control),
        "fallback_calls": len(fallback),
        "valid_control_fallback_calls": 0,
        "total_calls": len(control) + len(fallback),
    }
    candidate_by_id = {str(row["case_id"]): row for row in candidate}
    valid_controls = [row for row in control if bool(row["valid_output"])]
    valid_identity_count = sum(
        candidate_by_id[str(row["case_id"])]["route"] == "valid-control-identity"
        and candidate_by_id[str(row["case_id"])]["predicted_method_id"]
        == row["predicted_method_id"]
        and candidate_by_id[str(row["case_id"])]["control_raw_answer_sha256"]
        == row["raw_answer_sha256"]
        for row in valid_controls
    )
    expected_runtime_integrity = {
        "all_controls_completed_before_fallback": True,
        "control_model_release_hook_called": True,
        "valid_control_identity_count": valid_identity_count,
        "valid_control_identity_fraction": valid_identity_count / len(valid_controls)
        if valid_controls
        else 1.0,
        "fallback_calls_per_invalid_control": 1.0,
        "valid_control_fallback_calls": 0,
        "zero_control_only_losses": int(paired["control_only"]) == 0,
    }
    checks = {
        "scores": {
            "control": _strict_metrics(control),
            "catalog_guard": _strict_metrics(candidate),
        },
        "paired": paired,
        "source_paired": _source_paired(control, candidate),
        "template_paired": template_summary,
        "external_gate": gate,
        "model_call_counts": expected_call_counts,
        "runtime_integrity": expected_runtime_integrity,
    }
    for key, expected in checks.items():
        if report.get(key) != expected:
            raise RuntimeError(f"E6 terminal {key} changed")
    if report.get("status") != gate["status"]:
        raise RuntimeError("E6 terminal status changed")
    if private.get("semantic_template_details") != templates["details"]:
        raise RuntimeError("E6 terminal semantic-template details changed")


def _public_report(report: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in report.items() if key != "private_details"}
    forbidden = {"case_id", "question", "gold_method_id", "raw_answer"}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            leaked = forbidden & set(value)
            if leaked:
                raise RuntimeError(f"E6 public report contains private fields: {sorted(leaked)}")
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(public)
    return public


def _runtime_phase_receipts(master: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    parent = dict(master["runtime"]["parent"])
    control = {
        **parent,
        "prompt_sha256": master["runtime"]["control_prompt_sha256"],
        "phase": "menu-free-control",
    }
    fallback = {
        **parent,
        "prompt_sha256": master["runtime"]["fallback_prompt_sha256"],
        "phase": "fixed-catalog-fallback",
    }
    return control, fallback


def _verify_terminal_artifacts(
    report: dict[str, Any],
    *,
    root: Path,
    master: dict[str, Any],
    child: dict[str, Any],
    data: dict[str, Any],
    cases: list[dict[str, Any]],
    evaluation_fingerprint: str,
) -> None:
    _verify_terminal_report(
        report,
        opportunity_gates=dict(master["opportunity_gates"]),
        result_gates=dict(master["result_gates"]),
    )
    opening_path = root / "evaluation-opened.json"
    if not opening_path.is_file() or sha256_file(opening_path) != report.get("opening_sha256"):
        raise RuntimeError("E6 opening receipt bytes changed")
    opening = json.loads(opening_path.read_text(encoding="utf-8"))
    expected_opening: dict[str, Any] = {
        "schema_version": 1,
        "method": _METHOD,
        "evaluation_fingerprint": evaluation_fingerprint,
        "master_contract_fingerprint": master["fingerprint"],
        "child_contract_fingerprint": child["fingerprint"],
        "data_fingerprint": data["data_fingerprint"],
        "case_count": len(cases),
        "model_output_opened": True,
        "training_authorized": False,
        "champion_changed": False,
        "release_authorized": False,
    }
    expected_opening["fingerprint"] = canonical_hash(expected_opening)
    if opening != expected_opening or report.get("opening_fingerprint") != opening["fingerprint"]:
        raise RuntimeError("E6 opening receipt changed")
    governance = {
        "master_contract_fingerprint": master["fingerprint"],
        "child_contract_fingerprint": child["fingerprint"],
        "data_fingerprint": data["data_fingerprint"],
        "fresh_external_evidence": True,
        "training_authorized": False,
        "champion_changed": False,
        "release_authorized": False,
        "claim_boundary": master["claim_boundary"],
    }
    for key, expected in governance.items():
        if report.get(key) != expected:
            raise RuntimeError(f"E6 terminal governance field changed: {key}")

    control_receipt, fallback_receipt = _runtime_phase_receipts(master)
    private = report["private_details"]
    control_header = _phase_header(
        cases,
        phase="control",
        grounded=False,
        evaluation_fingerprint=evaluation_fingerprint,
        runtime_receipt=control_receipt,
    )
    control_path = root / "progress" / "control-ledger.json"
    if not control_path.is_file() or sha256_file(control_path) != report.get(
        "control_ledger_sha256"
    ):
        raise RuntimeError("E6 control ledger bytes changed")
    control_rows = _load_strict_progress(control_path, header=control_header)
    if len(control_rows) != len(cases):
        raise RuntimeError("E6 control ledger is incomplete")
    control = [_phase_detail(case, row) for case, row in zip(cases, control_rows, strict=True)]
    if control != private["control"]:
        raise RuntimeError("E6 terminal controls differ from the sealed ledger")

    invalid_cases = [
        case for case, row in zip(cases, control, strict=True) if not bool(row["valid_output"])
    ]
    fallback_path = root / "progress" / "fallback-ledger.json"
    if invalid_cases:
        fallback_header = _phase_header(
            invalid_cases,
            phase="catalog-fallback",
            grounded=True,
            evaluation_fingerprint=evaluation_fingerprint,
            runtime_receipt=fallback_receipt,
        )
        if not fallback_path.is_file() or sha256_file(fallback_path) != report.get(
            "fallback_ledger_sha256"
        ):
            raise RuntimeError("E6 fallback ledger bytes changed")
        fallback_rows = _load_strict_progress(fallback_path, header=fallback_header)
        if len(fallback_rows) != len(invalid_cases):
            raise RuntimeError("E6 fallback ledger is incomplete")
        fallback = [
            _phase_detail(case, row) for case, row in zip(invalid_cases, fallback_rows, strict=True)
        ]
    else:
        if fallback_path.exists() or report.get("fallback_ledger_sha256") is not None:
            raise RuntimeError("E6 unexpected fallback ledger exists without invalid controls")
        fallback = []
    if fallback != private["fallback"]:
        raise RuntimeError("E6 terminal fallbacks differ from the sealed ledger")
    fallback_by_id = {str(row["case_id"]): row for row in fallback}
    candidate = [
        _apply_catalog_guard(row, fallback_by_id.get(str(row["case_id"]))) for row in control
    ]
    if candidate != private["candidate"]:
        raise RuntimeError("E6 terminal candidate differs from the sealed ledgers")


def run_catalog_fallback_external_evaluation(config: ProjectConfig) -> dict[str, Any]:
    root = _root(config)
    master_path = root / "master-contract.json"
    child_path = root / "selected-source-contract.json"
    data_path = root / "selected-source-data.json"
    if not master_path.is_file():
        raise RuntimeError("E6 master contract is missing")
    master = json.loads(master_path.read_text(encoding="utf-8"))
    _verify_master(master)
    terminal = _read_terminal_state_receipt(config)
    report_path = root / "report.json"
    if terminal is not None and not report_path.is_file():
        raise RuntimeError(
            f"E6 is closed at terminal state {terminal['status']}; model evaluation remains closed"
        )
    if not child_path.is_file() or not data_path.is_file():
        raise RuntimeError(
            "E6 has no preregistered qualified source; model evaluation remains unopened"
        )
    child = json.loads(child_path.read_text(encoding="utf-8"))
    data = json.loads(data_path.read_text(encoding="utf-8"))
    try:
        _verify_e6_child_contract(child, master)
        cases = _verify_selected_data(
            data,
            child,
            master,
            selected_root=root / "selected-source",
        )
        template_assignments = _verify_semantic_template_manifest(
            data,
            child,
            master,
            cases,
            selected_root=root / "selected-source",
        )
    except Exception:
        if terminal is None:
            _write_terminal_state_receipt(
                config,
                master_fingerprint=master["fingerprint"],
                status="PROTOCOL_INVALID",
                reason="opened-child-contract-or-data-readback-failed",
                child_fingerprint=child.get("fingerprint"),
                dataset_rows_opened=True,
                model_output_opened=(root / "evaluation-opened.json").is_file(),
            )
        raise
    evaluation_fingerprint = canonical_hash(
        {
            "master": master["fingerprint"],
            "child": child["fingerprint"],
            "data": data["data_fingerprint"],
            "runtime": master["runtime"]["fingerprint"],
            "evaluator_version": _EVALUATOR_VERSION,
        }
    )
    public_path = _reports_root(config) / "catalog-fallback-external-v1.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("evaluation_fingerprint") != evaluation_fingerprint:
            raise RuntimeError("E6 terminal report belongs to another evaluation")
        _verify_terminal_artifacts(
            report,
            root=root,
            master=master,
            child=child,
            data=data,
            cases=cases,
            evaluation_fingerprint=evaluation_fingerprint,
        )
        terminal = _read_terminal_state_receipt(config)
        if terminal is None:
            terminal = _write_terminal_state_receipt(
                config,
                master_fingerprint=master["fingerprint"],
                status=report["status"],
                reason="reconstructed-from-verified-terminal-report-and-ledgers",
                child_fingerprint=child["fingerprint"],
                dataset_rows_opened=True,
                model_output_opened=True,
                result_fingerprint=report["result_fingerprint"],
            )
        if (
            terminal.get("master_contract_fingerprint") != master["fingerprint"]
            or terminal.get("child_contract_fingerprint") != child["fingerprint"]
            or terminal.get("status") != report["status"]
            or terminal.get("result_fingerprint") != report["result_fingerprint"]
            or not terminal.get("dataset_rows_opened")
            or not terminal.get("model_output_opened")
        ):
            raise RuntimeError("E6 scientific terminal-state receipt changed")
        public = _public_report(report)
        write_json(public_path, public)
        if json.loads(public_path.read_text(encoding="utf-8")) != public:
            raise RuntimeError("E6 public report round-trip changed")
        return public
    opening: dict[str, Any] = {
        "schema_version": 1,
        "method": _METHOD,
        "evaluation_fingerprint": evaluation_fingerprint,
        "master_contract_fingerprint": master["fingerprint"],
        "child_contract_fingerprint": child["fingerprint"],
        "data_fingerprint": data["data_fingerprint"],
        "case_count": len(cases),
        "model_output_opened": True,
        "training_authorized": False,
        "champion_changed": False,
        "release_authorized": False,
    }
    opening["fingerprint"] = canonical_hash(opening)
    _write_immutable_json(root / "evaluation-opened.json", opening, label="opening receipt")
    parent = dict(master["runtime"]["parent"])
    control_receipt, fallback_receipt = _runtime_phase_receipts(master)
    control_caller: _AgentCaller | None = None
    fallback_caller_instance: _AgentCaller | None = None

    def call_fallback(
        case_id: str,
        messages: list[dict[str, str]],
        decoding: dict[str, Any],
    ) -> str:
        nonlocal fallback_caller_instance
        if fallback_caller_instance is None:
            fallback_caller_instance = _AgentCaller(
                config,
                adapter_path=str(parent["adapter_path"]),
                expected_calls=None,
            )
        return fallback_caller_instance(case_id, messages, decoding)

    try:
        _preflight_two_stage_progress(
            cases,
            evaluation_fingerprint=evaluation_fingerprint,
            control_runtime_receipt=control_receipt,
            fallback_runtime_receipt=fallback_receipt,
            progress_root=root / "progress",
        )
        control_caller = _AgentCaller(
            config,
            adapter_path=str(parent["adapter_path"]),
            expected_calls=None,
        )
        report = run_catalog_fallback_two_stage(
            cases,
            evaluation_fingerprint=evaluation_fingerprint,
            control_runtime_receipt=control_receipt,
            fallback_runtime_receipt=fallback_receipt,
            control_caller=control_caller,
            fallback_caller=call_fallback,
            before_fallback=control_caller.close,
            progress_root=root / "progress",
            template_assignments=template_assignments,
            opportunity_gates=dict(master["opportunity_gates"]),
            result_gates=dict(master["result_gates"]),
        )
    except Exception:
        _write_terminal_state_receipt(
            config,
            master_fingerprint=master["fingerprint"],
            status="PROTOCOL_INVALID",
            reason="model-runtime-or-progress-integrity-failed-after-opening",
            child_fingerprint=child["fingerprint"],
            dataset_rows_opened=True,
            model_output_opened=True,
        )
        raise
    finally:
        if control_caller is not None:
            control_caller.close()
        if fallback_caller_instance is not None:
            fallback_caller_instance.close()
    report.update(
        {
            "master_contract_fingerprint": master["fingerprint"],
            "child_contract_fingerprint": child["fingerprint"],
            "data_fingerprint": data["data_fingerprint"],
            "opening_fingerprint": opening["fingerprint"],
            "opening_sha256": sha256_file(root / "evaluation-opened.json"),
            "control_ledger_sha256": sha256_file(root / "progress" / "control-ledger.json"),
            "fallback_ledger_sha256": (
                sha256_file(root / "progress" / "fallback-ledger.json")
                if (root / "progress" / "fallback-ledger.json").is_file()
                else None
            ),
            "fresh_external_evidence": True,
            "training_authorized": False,
            "champion_changed": False,
            "release_authorized": False,
            "claim_boundary": master["claim_boundary"],
        }
    )
    report["result_fingerprint"] = _terminal_fingerprint(report)
    _verify_terminal_artifacts(
        report,
        root=root,
        master=master,
        child=child,
        data=data,
        cases=cases,
        evaluation_fingerprint=evaluation_fingerprint,
    )
    _write_immutable_json(report_path, report, label="terminal report")
    _write_terminal_state_receipt(
        config,
        master_fingerprint=master["fingerprint"],
        status=report["status"],
        reason="prospective-evaluation-completed-under-frozen-gates",
        child_fingerprint=child["fingerprint"],
        dataset_rows_opened=True,
        model_output_opened=True,
        result_fingerprint=report["result_fingerprint"],
    )
    public = _public_report(report)
    write_json(public_path, public)
    if json.loads(public_path.read_text(encoding="utf-8")) != public:
        raise RuntimeError("E6 public report round-trip changed")
    return public

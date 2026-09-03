# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import ProjectConfig
from .io_utils import canonical_hash, sha256_file, write_json

_SCREEN_VERSION = 1
_EXPECTED_MASTER_FINGERPRINT = "35cd1762825de0724841875409ab9e3bd31454a92b43a9e9f86f37f34393f93f"
_EXPECTED_PRIOR_SCREEN_FINGERPRINT = (
    "34ea53c6307c9eeca17a5940e59c22b3156cfa2dbe878c18a21cab618b0a02c2"
)
_EXPECTED_RECEIPT_SHA256 = "9dc5831065b1dbfa7b0e80311039c6f704c8ddb368b180c9e7f0b229b1e539c5"

_ALLOWED_REQUEST_KINDS = {
    "github-repository-metadata",
    "github-tree-metadata",
    "hub-metadata",
    "hub-tree-metadata",
    "osf-registration-metadata",
    "viewer-is-valid",
    "viewer-size",
    "viewer-splits",
    "viewer-statistics",
    "zenodo-record-metadata",
}
_ALLOWED_HOSTS = {
    "api.github.com",
    "api.osf.io",
    "datasets-server.huggingface.co",
    "huggingface.co",
    "zenodo.org",
}
_FORBIDDEN_PATH_PARTS = {
    "download",
    "downloads",
    "file",
    "files",
    "first-rows",
    "parquet",
    "preview",
    "raw",
    "rows",
}

_HF_CANDIDATES: dict[str, dict[str, Any]] = {
    "ofrencber/scholargate-research-methods": {
        "title": "ScholarGate Research Methods",
        "near_candidate": True,
        "rejection_reasons": [
            "The 6,620 records form a method-reference catalog, not a study-question to method-label frame.",
            "The eight observed family values are taxonomy families, not eight frozen canonical method IDs.",
            "The name field has no metadata-level unique count, so at least 150 catalog-mapped cases cannot be established without opening records.",
        ],
    },
    "WithinUsAI/Statistics_25k": {
        "title": "Statistics 25k",
        "near_candidate": True,
        "rejection_reasons": [
            "The Viewer confirms 25,000 rows but statistics generation fails before a stable complete schema is exposed.",
            "Metadata does not establish a study-question field, a method-label field, or eight canonical methods.",
        ],
    },
    "0v01111/StatEval-Foundational-knowledge": {
        "title": "StatEval Foundational Knowledge",
        "near_candidate": True,
        "rejection_reasons": [
            "Viewer generation fails because answer values drift between scalar and list types.",
            "No processed row count, stable method-label schema, or complete frame is available at metadata level.",
        ],
    },
    "0v01111/StatEval-Statistical-Research": {
        "title": "StatEval Statistical Research",
        "near_candidate": True,
        "rejection_reasons": [
            "The repository contains a 107 MB Arrow file, but Viewer generation fails on an empty struct field.",
            "No processed row count, stable method-label schema, or complete frame is available at metadata level.",
        ],
    },
    "StatAILab/StatEval": {
        "title": "StatEval",
        "near_candidate": True,
        "rejection_reasons": [
            "Viewer generation fails because source files expose incompatible column sets.",
            "Metadata does not establish canonical method labels or a complete single-source frame.",
        ],
    },
    "gaochenyin/StatEval": {
        "title": "StatEval mirror",
        "near_candidate": False,
        "rejection_reasons": [
            "Viewer generation fails because source files expose incompatible column sets.",
            "It is a StatEval derivative or mirror and cannot independently establish a fresh E5 source.",
        ],
    },
    "Yoxas/statistical_literacy": {
        "title": "Statistical Literacy",
        "near_candidate": False,
        "rejection_reasons": [
            "The 11,243-row schema is bibliographic metadata with title, abstract, authors, DOI, and publication fields.",
            "It contains no question-to-method-label frame.",
        ],
    },
    "Yoxas/statistical_literacyv2": {
        "title": "Statistical Literacy v2",
        "near_candidate": False,
        "rejection_reasons": [
            "The 42,834-row schema is a bibliographic corpus.",
            "It contains no question-to-method-label frame.",
        ],
    },
    "vjain/AP_statistics": {
        "title": "AP Statistics",
        "near_candidate": False,
        "rejection_reasons": [
            "All 6,736 records expose only one text field.",
            "There is no independent gold method-label field.",
        ],
    },
    "Lots-of-LoRAs/task710_mmmlu_answer_generation_high_school_statistics": {
        "title": "MMLU High School Statistics answer generation",
        "near_candidate": False,
        "rejection_reasons": [
            "The repository contains 175 records and only 140 training records.",
            "Its id/input/output frame has no independent canonical statistical-method label.",
        ],
    },
    "joey234/mmlu-high_school_statistics": {
        "title": "MMLU High School Statistics",
        "near_candidate": False,
        "rejection_reasons": [
            "The 221 records use multiple-choice answer labels rather than canonical statistical-method IDs.",
            "Hub metadata does not state a license.",
        ],
    },
    "introvoyz041/handbook-of-statistical-methods-for-precision-medicine": {
        "title": "Handbook of Statistical Methods for Precision Medicine",
        "near_candidate": False,
        "rejection_reasons": [
            "The 482 records expose only an image field.",
            "Hub metadata does not state a license and no question-label frame exists.",
        ],
    },
    "Zhaorun/hypothesis_testing": {
        "title": "Hypothesis Testing",
        "near_candidate": False,
        "rejection_reasons": [
            "Viewer metadata reports no supported dataset files.",
            "Repository metadata does not establish a question-to-method-label frame.",
        ],
    },
}

_ZENODO_CANDIDATES: dict[str, dict[str, Any]] = {
    "zenodo:22140684": {
        "title": "Towards standardized prompts for statistical reasoning",
        "near_candidate": True,
        "rejection_reasons": [
            "Record metadata exposes one 14,921-byte XLSX but no row count, schema, method count, or complete frame.",
        ],
    },
    "zenodo:13240122": {
        "title": "Statistical Tests Classifiers",
        "near_candidate": True,
        "rejection_reasons": [
            "Record metadata exposes one 82,792-byte XLSX but no row count, schema, method coverage, or complete frame.",
        ],
    },
    "zenodo:10653965": {
        "title": "MUSI",
        "near_candidate": False,
        "rejection_reasons": [
            "The record describes decision-tree software rather than a labeled dataset.",
        ],
    },
    "zenodo:22043861": {
        "title": "Human-supervised AI statistical decision support",
        "near_candidate": False,
        "rejection_reasons": [
            "Record metadata describes 68 studies and 74 rules, below the 150-case source gate.",
            "The unit is not a study-question to canonical-method case frame.",
        ],
    },
}

_GITHUB_CANDIDATES: dict[str, dict[str, Any]] = {
    "Amazing-BIG-Tree/StatBench-Data-Pipeline": {
        "title": "StatBench Data Pipeline",
        "revision": "810004a7545002d4beec594fdd0cf04a0f059935",
        "near_candidate": False,
        "rejection_reasons": [
            "The repository is a data-building pipeline, not a released single-source labeled dataset.",
            "Repository metadata exposes no license, row count, method count, or source manifest.",
        ],
    },
    "behavioral-data/BLADE": {
        "title": "BLADE",
        "revision": "6118fa8d5007b91aa8c91c518182db82446a4547",
        "near_candidate": False,
        "rejection_reasons": [
            "BLADE is a multi-dataset analysis benchmark rather than a single-source question-to-method-label frame.",
            "Metadata does not establish an E5-complete frame or eight canonical method labels.",
        ],
    },
    "shufe-zj-AILab/STATBench": {
        "title": "STATBench",
        "revision": "55a5bc2dec5bcef547cad260595eef6f0cd3b615",
        "near_candidate": False,
        "rejection_reasons": [
            "Repository metadata states that a released portion has only 100 questions.",
            "The question/options/answer frame does not expose canonical statistical-method IDs and the repository has no license.",
        ],
    },
    "xxxiaol/QRData": {
        "title": "QRData",
        "revision": "de450af45ff7101b328bb064c6b475f73414a7ed",
        "near_candidate": False,
        "rejection_reasons": [
            "Metadata does not establish row count, eight canonical method labels, or a complete frame.",
            "The GitHub license field is NOASSERTION.",
        ],
    },
    "albertsimonyan74/bayes-benchmark": {
        "title": "Bayes Benchmark",
        "revision": "14924b33063e40e6bc0169468317e910adedf742",
        "near_candidate": False,
        "rejection_reasons": [
            "The reasoning benchmark metadata does not state a method-selection label schema, eligible row count, or eight-method coverage.",
        ],
    },
    "crishnagarkarleeds/statistics-llm": {
        "title": "Statistics LLM",
        "revision": "dfc35f2b726615ca894abbb7aa1e62d2a12899e1",
        "near_candidate": False,
        "rejection_reasons": [
            "The question bank is packaged as PDF content.",
            "Repository metadata exposes no row-level schema or canonical method-label frame.",
        ],
    },
}

_OSF_CANDIDATES: dict[str, dict[str, Any]] = {
    "osf:j96k4": {
        "title": "Forensic statistical method selection audit",
        "near_candidate": False,
        "rejection_reasons": [
            "The OSF object is an ANOVA reporting-audit preregistration rather than an eight-method labeled dataset.",
            "Registration metadata does not provide a license, versioned file size, row count, or complete frame.",
        ],
    }
}


def _receipt_path(config: ProjectConfig) -> Path:
    return (
        config.path_for("evolution_dir")
        / "guarded-external-v1"
        / "future-source-metadata"
        / "metadata-receipts.json"
    )


def _artifact_path(config: ProjectConfig) -> Path:
    return (
        config.path_for("artifact_dir")
        / "guarded-external-v1"
        / "future-source-metadata-screen.json"
    )


def _public_path(config: ProjectConfig) -> Path:
    return (
        config.root
        / "reports"
        / "evolve"
        / "guarded-external-v1-future-source-metadata-screen.json"
    )


def _validate_receipts(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    if payload.get("schema_version") != 1:
        raise RuntimeError("E5 metadata receipt schema changed")
    policy = payload.get("content_access_policy")
    if not isinstance(policy, dict) or any(value is not False for value in policy.values()):
        raise RuntimeError("E5 metadata screen opened dataset content")
    receipts = payload.get("receipts")
    if not isinstance(receipts, list) or len(receipts) != payload.get("receipt_count"):
        raise RuntimeError("E5 metadata receipt count changed")

    index: dict[tuple[str, str], dict[str, Any]] = {}
    for receipt in receipts:
        if not isinstance(receipt, dict):
            raise RuntimeError("E5 metadata receipt is not an object")
        source_id = str(receipt.get("repo_id", ""))
        kind = str(receipt.get("request_kind", ""))
        url = str(receipt.get("url", ""))
        parsed = urlparse(url)
        path_parts = {part.lower() for part in parsed.path.split("/") if part}
        if kind not in _ALLOWED_REQUEST_KINDS:
            raise RuntimeError(f"E5 metadata request kind is not allowed: {kind}")
        if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_HOSTS:
            raise RuntimeError(f"E5 metadata request host is not allowed: {url}")
        if path_parts & _FORBIDDEN_PATH_PARTS:
            raise RuntimeError(f"E5 metadata screen used a content endpoint: {url}")
        key = (source_id, kind)
        if key in index:
            raise RuntimeError(f"E5 metadata receipt is duplicated: {key}")
        index[key] = receipt
    return index


def _required_receipt(
    index: dict[tuple[str, str], dict[str, Any]], source_id: str, kind: str
) -> dict[str, Any]:
    receipt = index.get((source_id, kind))
    if receipt is None:
        raise RuntimeError(f"E5 metadata receipt is missing: {source_id}:{kind}")
    if not isinstance(receipt.get("body"), (dict, list)):
        raise RuntimeError(f"E5 metadata receipt body is unavailable: {source_id}:{kind}")
    return receipt


def _hf_candidate(
    source_id: str,
    spec: dict[str, Any],
    index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    hub = _required_receipt(index, source_id, "hub-metadata")["body"]
    size_receipt = _required_receipt(index, source_id, "viewer-size")
    size_body = size_receipt["body"]
    stats_receipt = index.get((source_id, "viewer-statistics"))
    stats_body = stats_receipt["body"] if stats_receipt is not None else {}
    tree_receipt = index.get((source_id, "hub-tree-metadata"))
    tree_body = tree_receipt["body"] if tree_receipt is not None else []

    dataset_size = (
        size_body.get("size", {}).get("dataset", {}) if isinstance(size_body, dict) else {}
    )
    size_failed = size_body.get("failed", []) if isinstance(size_body, dict) else []
    statistics = stats_body.get("statistics", []) if isinstance(stats_body, dict) else []
    columns = [
        str(column["column_name"])
        for column in statistics
        if isinstance(column, dict) and "column_name" in column
    ]
    split_rows = {}
    if isinstance(size_body, dict):
        split_rows = {
            str(split.get("split")): int(split.get("num_rows", 0))
            for split in size_body.get("size", {}).get("splits", [])
        }
    card_data = hub.get("cardData", {}) if isinstance(hub, dict) else {}
    files = []
    if isinstance(tree_body, list):
        files = [
            {
                "path": str(item.get("path")),
                "size": item.get("size"),
                "lfs_sha256": item.get("lfs", {}).get("oid")
                if isinstance(item.get("lfs"), dict)
                else None,
            }
            for item in tree_body
            if isinstance(item, dict) and item.get("type") == "file"
        ]

    row_count = dataset_size.get("num_rows")
    processed_rows_confirmed = bool(
        isinstance(row_count, int) and row_count > 0 and not size_failed
    )
    return {
        "provider": "hugging-face",
        "source_id": source_id,
        "title": spec["title"],
        "revision": hub.get("sha") if isinstance(hub, dict) else None,
        "license": card_data.get("license") if isinstance(card_data, dict) else None,
        "metadata_facts": {
            "processed_row_count": row_count,
            "processed_row_count_confirmed": processed_rows_confirmed,
            "split_row_counts": split_rows,
            "schema_columns": columns,
            "statistics_http_status": stats_receipt.get("http_status")
            if stats_receipt is not None
            else None,
            "statistics_error": stats_body.get("cause_exception")
            if isinstance(stats_body, dict)
            else None,
            "pinned_tree_files": files,
        },
        "near_candidate": bool(spec["near_candidate"]),
        "qualification_checks": _failed_qualification_checks(),
        "disposition": "metadata-only-reject",
        "rejection_reasons": list(spec["rejection_reasons"]),
    }


def _zenodo_candidate(
    source_id: str,
    spec: dict[str, Any],
    index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    body = _required_receipt(index, source_id, "zenodo-record-metadata")["body"]
    metadata = body.get("metadata", {})
    license_value = metadata.get("license", {})
    return {
        "provider": "zenodo",
        "source_id": source_id,
        "title": spec["title"],
        "revision": str(body.get("id")),
        "license": license_value.get("id") if isinstance(license_value, dict) else None,
        "metadata_facts": {
            "doi": metadata.get("doi"),
            "files": [
                {
                    "name": item.get("key"),
                    "size": item.get("size"),
                    "checksum": item.get("checksum"),
                }
                for item in body.get("files", [])
                if isinstance(item, dict)
            ],
            "row_count": None,
            "schema_columns": [],
        },
        "near_candidate": bool(spec["near_candidate"]),
        "qualification_checks": _failed_qualification_checks(),
        "disposition": "metadata-only-reject",
        "rejection_reasons": list(spec["rejection_reasons"]),
    }


def _github_candidate(
    source_id: str,
    spec: dict[str, Any],
    index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    repository = _required_receipt(index, source_id, "github-repository-metadata")["body"]
    tree = _required_receipt(index, source_id, "github-tree-metadata")["body"]
    if tree.get("sha") != spec["revision"]:
        raise RuntimeError(f"E5 GitHub candidate revision changed: {source_id}")
    tree_items = tree.get("tree", [])
    blobs = [item for item in tree_items if isinstance(item, dict) and item.get("type") == "blob"]
    license_value = repository.get("license")
    return {
        "provider": "github",
        "source_id": source_id,
        "title": spec["title"],
        "revision": spec["revision"],
        "license": license_value.get("spdx_id") if isinstance(license_value, dict) else None,
        "metadata_facts": {
            "repository_size_kib": repository.get("size"),
            "blob_count": len(blobs),
            "blob_bytes": sum(int(item.get("size", 0) or 0) for item in blobs),
            "tree_truncated": bool(tree.get("truncated")),
            "row_count": None,
            "schema_columns": [],
        },
        "near_candidate": bool(spec["near_candidate"]),
        "qualification_checks": _failed_qualification_checks(),
        "disposition": "metadata-only-reject",
        "rejection_reasons": list(spec["rejection_reasons"]),
    }


def _osf_candidate(
    source_id: str,
    spec: dict[str, Any],
    index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    body = _required_receipt(index, source_id, "osf-registration-metadata")["body"]
    data = body.get("data", {})
    attributes = data.get("attributes", {})
    return {
        "provider": "osf",
        "source_id": source_id,
        "title": spec["title"],
        "revision": None,
        "license": None,
        "metadata_facts": {
            "registration_id": data.get("id"),
            "registered_at": attributes.get("date_registered"),
            "registration_supplement": attributes.get("registration_supplement"),
            "row_count": None,
            "schema_columns": [],
        },
        "near_candidate": bool(spec["near_candidate"]),
        "qualification_checks": _failed_qualification_checks(),
        "disposition": "metadata-only-reject",
        "rejection_reasons": list(spec["rejection_reasons"]),
    }


def _failed_qualification_checks() -> dict[str, bool]:
    return {
        "minimum_eligible_cases_demonstrated": False,
        "minimum_distinct_methods_demonstrated": False,
        "minimum_coverage_fraction_demonstrated": False,
        "maximum_method_concentration_demonstrated": False,
        "complete_frame_freezable_before_opening": False,
        "alias_map_freezable_before_opening": False,
        "overlap_manifest_freezable_before_opening": False,
        "qualified_child_contract_freezable": False,
    }


def _candidate_summaries(
    index: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = [
        _hf_candidate(source_id, spec, index) for source_id, spec in _HF_CANDIDATES.items()
    ]
    candidates.extend(
        _zenodo_candidate(source_id, spec, index) for source_id, spec in _ZENODO_CANDIDATES.items()
    )
    candidates.extend(
        _github_candidate(source_id, spec, index) for source_id, spec in _GITHUB_CANDIDATES.items()
    )
    candidates.extend(
        _osf_candidate(source_id, spec, index) for source_id, spec in _OSF_CANDIDATES.items()
    )
    return sorted(candidates, key=lambda item: (str(item["provider"]), str(item["source_id"])))


def prepare_guarded_external_future_metadata_screen(config: ProjectConfig) -> dict[str, Any]:
    master_path = config.root / "reports" / "evolve" / "guarded-external-v1-master-contract.json"
    prior_screen_path = (
        config.root / "reports" / "evolve" / "guarded-external-v1-source-screen.json"
    )
    master = json.loads(master_path.read_text(encoding="utf-8"))
    prior_screen = json.loads(prior_screen_path.read_text(encoding="utf-8"))
    if master.get("fingerprint") != _EXPECTED_MASTER_FINGERPRINT:
        raise RuntimeError("E5 master contract changed before metadata screen")
    if prior_screen.get("result_fingerprint") != _EXPECTED_PRIOR_SCREEN_FINGERPRINT:
        raise RuntimeError("E5 prior source screen changed before metadata screen")

    receipt_path = _receipt_path(config)
    if sha256_file(receipt_path) != _EXPECTED_RECEIPT_SHA256:
        raise RuntimeError("E5 future-source metadata receipts changed")
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    index = _validate_receipts(receipt_payload)
    candidates = _candidate_summaries(index)
    if any(
        candidate["qualification_checks"]["qualified_child_contract_freezable"]
        for candidate in candidates
    ):
        raise RuntimeError("E5 metadata screen unexpectedly selected a source")

    gates = dict(master["source_protocol"]["qualification_gates"])
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "E5 future-source metadata-only qualification screen",
        "screen_version": _SCREEN_VERSION,
        "master_contract_fingerprint": master["fingerprint"],
        "prior_source_screen_result_fingerprint": prior_screen["result_fingerprint"],
        "receipt_manifest": {
            "path": str(receipt_path.relative_to(config.root)),
            "sha256": sha256_file(receipt_path),
            "receipt_count": int(receipt_payload["receipt_count"]),
            "request_kinds": sorted(
                {str(receipt["request_kind"]) for receipt in receipt_payload["receipts"]}
            ),
            "allowed_metadata_requests_only": True,
            "rows_endpoint_called": False,
            "first_rows_endpoint_called": False,
            "parquet_endpoint_called": False,
            "raw_blob_endpoint_called": False,
            "download_endpoint_called": False,
            "dataset_files_downloaded": False,
            "dataset_card_examples_opened": False,
        },
        "source_gates": gates,
        "candidate_count": len(candidates),
        "near_candidate_count": sum(bool(candidate["near_candidate"]) for candidate in candidates),
        "candidates": candidates,
        "status": "NO_SOURCE_SELECTED_METADATA_SCREEN",
        "source_selected": False,
        "selected_source_id": None,
        "child_contract_created": False,
        "complete_frame_opened": False,
        "dataset_rows_opened": False,
        "model_output_opened": False,
        "evaluation_authorized": False,
        "training_authorized": False,
        "champion_changed": False,
        "release_authorized": False,
        "stopping_reason": (
            "No screened source can demonstrate, from metadata alone, a stable complete "
            "question-to-canonical-method frame with at least 150 eligible cases, eight methods, "
            "80% mapping coverage, and no method above 40%."
        ),
        "search_limit": (
            "This closes the screened candidate set, not the existence of all possible public "
            "sources. A later source may be considered only under the unchanged E5 master before "
            "opening rows or examples."
        ),
        "next_step": (
            "preserve-e5-unopened-and-develop-a-separate-opened-source-runtime-hypothesis"
        ),
        "claim_boundary": (
            "This is a metadata-only source-availability result. It is neither a model evaluation "
            "nor evidence against H20, and it cannot justify lowering E5 gates, pooling sources, "
            "opening a visibly unqualified dataset, training on candidate rows, or changing the champion."
        ),
    }
    report["fingerprint"] = canonical_hash(report)

    artifact_path = _artifact_path(config)
    if artifact_path.exists():
        existing = json.loads(artifact_path.read_text(encoding="utf-8"))
        if existing != report:
            raise RuntimeError("E5 future-source metadata screen is immutable")
    else:
        write_json(artifact_path, report)
    write_json(_public_path(config), report)
    return report

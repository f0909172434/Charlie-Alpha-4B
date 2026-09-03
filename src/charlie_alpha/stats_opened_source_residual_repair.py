from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl
from .stats_catalog import PROCEDURES
from .stats_external_catalog import _ALIASES
from .stats_guarded_external import (
    _AgentCaller,
    _apply_guard,
    _external_gate,
    _load_strict_progress,
    _paired,
    _phase_header,
    _public_report,
    _source_paired,
    _strict_metrics,
    _terminal_fingerprint,
    _verify_master_contract,
    _verify_terminal_report,
    _wilson_interval,
    run_exact_two_stage,
)
from .stats_guarded_external_source import (
    _METHOD_IDS,
    _SOURCES,
    _bioc_document,
    _method_cells,
    prepare_guarded_external_master_contract,
    prepare_guarded_external_source_screen,
)

_METHOD = "H21_OPENED_SOURCE_RESIDUAL_REPAIR_DEV_V1"
_METHOD_VERSION = 1
_H20_ADAPTER_SHA256 = "32f964f279157b2b8e3d0d87329d342e55fa88e512de5480f4a2dc96fdcd6b2d"
_EXPECTED_ELIGIBLE_COUNT = 26
_EXPECTED_INCLUDED_COUNT = 24
_EXPECTED_EXCLUDED_COUNT = 2
_EXPECTED_SOURCE_COUNT = 4
_EXPECTED_METHOD_COUNT = 10
_OPPORTUNITY_GATES: dict[str, int] = {
    "minimum_residual_invalid_cases": 6,
    "minimum_residual_invalid_sources": 3,
    "minimum_residual_invalid_gold_methods": 3,
}

CellKey = tuple[str, str, int, int]


def _question(description: str) -> str:
    return f"Choose one catalog procedure for this study: {description}"


# This mapping was reviewed cell by cell. It deliberately contains no text copied from a
# source method cell and no catalog method display name or alias.
_INCLUDED_CELLS: dict[CellKey, tuple[str, str]] = {
    ("PMC8327789", "T1", 1, 1): (
        "mann_whitney",
        _question("an ordinal or skewed outcome is measured in two independent groups."),
    ),
    ("PMC8327789", "T1", 2, 1): (
        "independent_t",
        _question(
            "a normally distributed continuous outcome is measured in two independent groups."
        ),
    ),
    ("PMC8327789", "T2", 1, 1): (
        "wilcoxon_signed_rank",
        _question(
            "an ordinal or skewed outcome is measured twice on the same matched units."
        ),
    ),
    ("PMC8327789", "T2", 2, 1): (
        "paired_t",
        _question(
            "a normally distributed continuous outcome is measured twice on the same matched "
            "units."
        ),
    ),
    ("PMC8327789", "T3", 5, 1): (
        "ols",
        _question(
            "a continuous outcome is predicted from one or more explanatory variables."
        ),
    ),
    ("PMC8327789", "T3", 6, 1): (
        "logistic_glm",
        _question(
            "a binary outcome is predicted from one or more categorical or continuous "
            "explanatory variables."
        ),
    ),
    ("PMC8483143", "T2", 0, 1): (
        "paired_t",
        _question(
            "a normally distributed continuous outcome is measured in two matched conditions."
        ),
    ),
    ("PMC8483143", "T2", 0, 2): (
        "independent_t",
        _question(
            "a normally distributed continuous outcome is measured in two independent groups."
        ),
    ),
    ("PMC8483143", "T2", 0, 6): (
        "ols",
        _question(
            "a continuous outcome is predicted from one or more explanatory variables."
        ),
    ),
    ("PMC8483143", "T2", 1, 1): (
        "wilcoxon_signed_rank",
        _question(
            "a ranked or non-normally distributed outcome is measured in two matched "
            "conditions."
        ),
    ),
    ("PMC8483143", "T2", 1, 2): (
        "mann_whitney",
        _question(
            "a ranked or non-normally distributed outcome is measured in two independent "
            "groups."
        ),
    ),
    ("PMC8483143", "T2", 2, 4): (
        "chi_square",
        _question(
            "a dichotomous outcome summarized as proportions is compared across three or more "
            "independent groups."
        ),
    ),
    ("PMC8483143", "T2", 2, 6): (
        "logistic_glm",
        _question(
            "a binary outcome is predicted from one or more categorical or continuous "
            "explanatory variables."
        ),
    ),
    ("PMC6639881", "T1", 1, 2): (
        "wilcoxon_signed_rank",
        _question(
            "one ordinal or non-normally distributed sample is compared with a fixed "
            "hypothetical value."
        ),
    ),
    ("PMC6639881", "T1", 2, 1): (
        "independent_t",
        _question(
            "a normally distributed continuous outcome is measured in two independent groups."
        ),
    ),
    ("PMC6639881", "T1", 2, 2): (
        "mann_whitney",
        _question(
            "an ordinal or non-normally distributed outcome is measured in two independent "
            "groups."
        ),
    ),
    ("PMC6639881", "T1", 3, 1): (
        "paired_t",
        _question(
            "a normally distributed continuous outcome is measured in two matched groups."
        ),
    ),
    ("PMC6639881", "T1", 3, 2): (
        "wilcoxon_signed_rank",
        _question(
            "an ordinal or non-normally distributed outcome is measured in two matched groups."
        ),
    ),
    ("PMC6639881", "T1", 7, 1): (
        "ols",
        _question(
            "a continuous outcome is predicted from one or more explanatory variables."
        ),
    ),
    ("PMC6639881", "T3", 0, 1): (
        "logistic_glm",
        _question(
            "a binary outcome is predicted from one or more categorical or continuous "
            "explanatory variables."
        ),
    ),
    ("PMC6639881", "T3", 5, 1): (
        "cox_ph",
        _question(
            "a censored time-to-event outcome is related to one or more categorical or "
            "continuous predictors."
        ),
    ),
    ("PMC2996580", "T0001", 0, 0): (
        "fisher_exact",
        _question(
            "a binary outcome forms a two-by-two table for two independent groups, and "
            "finite-sample conditional inference is required."
        ),
    ),
    ("PMC2996580", "T0001", 1, 0): (
        "chi_square",
        _question(
            "a categorical outcome has more than two groups or more than two categories, with "
            "adequate expected cell counts."
        ),
    ),
    ("PMC2996580", "T0001", 8, 0): (
        "logrank",
        _question(
            "a censored time-to-event outcome is compared across two or more independent groups "
            "without additional predictors."
        ),
    ),
}

_EXCLUDED_CELLS: dict[CellKey, dict[str, str]] = {
    ("PMC6639881", "T2", 2, 1): {
        "gold_method_id": "two_proportion",
        "reason": "source description does not specify group count or pairedness",
    },
    ("PMC2996580", "T0001", 5, 0): {
        "gold_method_id": "mann_whitney",
        "reason": "source description explicitly permits both paired and unpaired data",
    },
}


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "opened-source-residual-repair-v1"


def _data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "opened-source-residual-repair-v1"


def _e5_snapshot_root(config: ProjectConfig) -> Path:
    return (
        config.path_for("evolution_dir")
        / "guarded-external-v1"
        / "source-screen"
        / "snapshots"
    )


def _reports_root(config: ProjectConfig) -> Path:
    return config.root / "reports" / "evolve"


def _fingerprint_without(payload: dict[str, Any], *fields: str) -> str:
    return canonical_hash({key: value for key, value in payload.items() if key not in fields})


def _cell_mapping_receipt(
    mapping: dict[CellKey, tuple[str, str]] | dict[CellKey, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in sorted(mapping.items()):
        row: dict[str, Any] = {
            "source_id": key[0],
            "table_id": key[1],
            "row_index": key[2],
            "column_index": key[3],
        }
        if isinstance(value, tuple):
            row.update({"gold_method_id": value[0], "question": value[1]})
        else:
            row.update(value)
        rows.append(row)
    return rows


def _write_immutable_json(path: Path, payload: dict[str, Any], *, label: str) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise RuntimeError(f"H21 {label} is unreadable") from error
        if existing != payload:
            raise RuntimeError(f"H21 {label} is immutable")
        return
    write_json(path, payload)


def _write_immutable_jsonl(path: Path, rows: list[dict[str, Any]], *, label: str) -> None:
    if path.exists():
        try:
            existing = list(read_jsonl(path))
        except (json.JSONDecodeError, OSError, ValueError) as error:
            raise RuntimeError(f"H21 {label} is unreadable") from error
        if existing != rows:
            raise RuntimeError(f"H21 {label} is immutable")
        return
    write_jsonl(path, rows)


def _verify_source_screen(screen: dict[str, Any], master: dict[str, Any]) -> None:
    if screen.get("result_fingerprint") != _fingerprint_without(
        screen, "result_fingerprint", "private_details"
    ):
        raise RuntimeError("H21 E5 source-screen fingerprint changed")
    if screen.get("master_contract_fingerprint") != master.get("fingerprint"):
        raise RuntimeError("H21 E5 source screen belongs to another master contract")
    if not screen.get("complete") or screen.get("status") != "SOURCE_SCREEN_UNQUALIFIED":
        raise RuntimeError("H21 requires the terminal unqualified E5 source screen")
    if not screen.get("source_screen_not_blind"):
        raise RuntimeError("H21 requires all four feasibility sources to be marked opened")
    if screen.get("evaluation_authorized") or screen.get("model_output_opened"):
        raise RuntimeError("H21 requires the E5 feasibility screen to have opened no model output")
    if screen.get("champion_changed") or screen.get("release_authorized"):
        raise RuntimeError("H21 requires the E5 feasibility screen to remain development-only")
    summaries = screen.get("screened_sources")
    if not isinstance(summaries, dict) or set(summaries) != set(_SOURCES):
        raise RuntimeError("H21 E5 screened-source set changed")
    for source_id, source in _SOURCES.items():
        summary = summaries.get(source_id)
        if not isinstance(summary, dict):
            raise RuntimeError(f"H21 E5 source summary is missing: {source_id}")
        if summary.get("snapshot_sha256") != source["snapshot_sha256"]:
            raise RuntimeError(f"H21 E5 snapshot seal changed: {source_id}")
        if not summary.get("opened_before_master_contract") or summary.get("qualified"):
            raise RuntimeError(f"H21 E5 source status changed: {source_id}")


def _verify_master_for_h21(master: dict[str, Any]) -> None:
    _verify_master_contract(master)
    runtime = master.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError("H21 E5 runtime seal is missing")
    repair = runtime.get("repair")
    if not isinstance(repair, dict) or repair.get("adapter_sha256") != _H20_ADAPTER_SHA256:
        raise RuntimeError("H21 frozen H20 adapter SHA changed")
    if master.get("source_selected") or master.get("model_output_opened"):
        raise RuntimeError("H21 requires the unopened E5 master protocol")
    if master.get("champion_changed") or master.get("release_authorized"):
        raise RuntimeError("H21 requires the E5 master to preserve champion and release state")
    exclusions = master.get("source_protocol", {}).get("opened_source_exclusions")
    if not isinstance(exclusions, dict) or set(exclusions) != set(_SOURCES):
        raise RuntimeError("H21 E5 opened-source exclusions changed")
    for source_id, source in _SOURCES.items():
        if exclusions[source_id].get("snapshot_sha256") != source["snapshot_sha256"]:
            raise RuntimeError(f"H21 E5 opened-source exclusion changed: {source_id}")


def _source_seals() -> dict[str, dict[str, str]]:
    return {
        source_id: {
            "title": str(source["title"]),
            "doi": str(source["doi"]),
            "license": str(source["license"]),
            "snapshot_sha256": str(source["snapshot_sha256"]),
            "status": "opened-development-only-not-fresh-e5",
        }
        for source_id, source in sorted(_SOURCES.items())
    }


def _contract_payload(master: dict[str, Any], screen: dict[str, Any]) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": _METHOD,
        "method_version": _METHOD_VERSION,
        "evidence_status": "opened-source-development-only",
        "causal_question": (
            "After the exact frozen H20 guard, are there enough parser-invalid residuals across "
            "opened sources and gold methods to justify a separately preregistered H21 training "
            "study?"
        ),
        "e5_master_contract_fingerprint": master["fingerprint"],
        "e5_runtime_fingerprint": master["runtime"]["fingerprint"],
        "e5_source_screen_result_fingerprint": screen["result_fingerprint"],
        "parent_runtime_receipt": dict(master["runtime"]["control"]),
        "h20_runtime_receipt": dict(master["runtime"]["repair"]),
        "h20_adapter_sha256": _H20_ADAPTER_SHA256,
        "snapshot_seals": _source_seals(),
        "materialization_contract": {
            "eligible_source_cell_count": _EXPECTED_ELIGIBLE_COUNT,
            "included_case_count": _EXPECTED_INCLUDED_COUNT,
            "ambiguity_exclusion_count": _EXPECTED_EXCLUDED_COUNT,
            "source_count": _EXPECTED_SOURCE_COUNT,
            "distinct_method_count": _EXPECTED_METHOD_COUNT,
            "partition_rule": (
                "Every catalog-mapped source cell is assigned exactly once to included or "
                "ambiguity-excluded; ambiguous structure is never inferred or repaired."
            ),
            "question_rule": (
                "Questions describe only data type, pairing, group count, and outcome/predictor "
                "structure and contain no source label, catalog display name, or catalog alias."
            ),
            "included_mapping_fingerprint": canonical_hash(
                _cell_mapping_receipt(_INCLUDED_CELLS)
            ),
            "ambiguity_exclusions": [
                {
                    "source_id": key[0],
                    "table_id": key[1],
                    "row_index": key[2],
                    "column_index": key[3],
                    **value,
                }
                for key, value in sorted(_EXCLUDED_CELLS.items())
            ],
            "ambiguity_exclusion_fingerprint": canonical_hash(
                _cell_mapping_receipt(_EXCLUDED_CELLS)
            ),
        },
        "opportunity_gates": dict(_OPPORTUNITY_GATES),
        "runtime_policy": [
            "Run the unchanged parent on all 24 included cases exactly once.",
            "Release the parent before the lazily loaded frozen H20 adapter is first invoked.",
            "Invoke H20 exactly once only for parser-invalid parent controls.",
            "Authorize later H21 training only if H20 remains parser-invalid on at least six "
            "cases spanning at least three sources and three gold methods.",
        ],
        "terminal_states": ["OPPORTUNITY_QUALIFIED", "INCONCLUSIVE_OPPORTUNITY"],
        "fresh_external_evidence": False,
        "training_authorized": False,
        "champion_changed": False,
        "release_authorized": False,
        "claim_boundary": (
            "This screen can authorize only a new development-only H21 training protocol. The "
            "four sources are already opened and cannot support a fresh E5 claim, model "
            "promotion, publication, or release."
        ),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    contract["fingerprint"] = canonical_hash(contract)
    return contract


def prepare_opened_source_residual_contract(config: ProjectConfig) -> dict[str, Any]:
    master = prepare_guarded_external_master_contract(config)
    screen = prepare_guarded_external_source_screen(config)
    _verify_master_for_h21(master)
    _verify_source_screen(screen, master)
    contract = _contract_payload(master, screen)
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    _write_immutable_json(root / "contract.json", contract, label="contract")
    write_json(
        _reports_root(config) / "opened-source-residual-repair-v1-contract.json", contract
    )
    return contract


def _normalized_phrase(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalized).split())


def _forbidden_phrases(source_labels: set[str]) -> set[str]:
    phrases = set(source_labels)
    phrases.update(procedure.name for procedure in PROCEDURES)
    phrases.update(alias for aliases in _ALIASES.values() for alias in aliases)
    phrases.update(
        label for source in _SOURCES.values() for label in source["aliases"]
    )
    phrases.update(method_id.replace("_", " ") for method_id in _METHOD_IDS)
    return {normalized for phrase in phrases if (normalized := _normalized_phrase(phrase))}


def _assert_leak_free(question: str, forbidden: set[str]) -> None:
    normalized = f" {_normalized_phrase(question)} "
    leaked = sorted(phrase for phrase in forbidden if f" {phrase} " in normalized)
    if leaked:
        raise RuntimeError(f"H21 question contains a forbidden method label or alias: {leaked}")


def _cell_key(cell: dict[str, Any]) -> CellKey:
    return (
        str(cell["source_id"]),
        str(cell["table_id"]),
        int(cell["row_index"]),
        int(cell["column_index"]),
    )


def _case_id(index: int) -> str:
    return f"h21-opened-{index:03d}"


def _extract_cases_and_manifest(
    config: ProjectConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    snapshot_root = _e5_snapshot_root(config)
    eligible_by_key: dict[CellKey, dict[str, Any]] = {}
    source_labels: set[str] = set()
    snapshot_receipts: dict[str, Any] = {}
    for source_id, source in sorted(_SOURCES.items()):
        path = snapshot_root / f"{source_id}.bioc.json"
        if not path.is_file() or sha256_file(path) != source["snapshot_sha256"]:
            raise RuntimeError(f"H21 frozen source snapshot changed or is missing: {source_id}")
        document = _bioc_document(
            path.read_bytes(),
            pmcid=source_id,
            expected_license=str(source["license"]),
            expected_doi=str(source["doi"]),
            expected_title=str(source["title"]),
        )
        cells = _method_cells(pmcid=source_id, source=source, document=document)
        source_labels.update(str(cell["source_label"]) for cell in cells)
        eligible = [cell for cell in cells if str(cell["mapped_method_id"]) in _METHOD_IDS]
        for cell in eligible:
            key = _cell_key(cell)
            if key in eligible_by_key:
                raise RuntimeError(f"H21 duplicate eligible source cell: {key}")
            eligible_by_key[key] = cell
        snapshot_receipts[source_id] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "doi": str(source["doi"]),
            "title": str(source["title"]),
            "license": str(source["license"]),
            "eligible_cell_count": len(eligible),
        }

    included_keys = set(_INCLUDED_CELLS)
    excluded_keys = set(_EXCLUDED_CELLS)
    if included_keys & excluded_keys:
        raise RuntimeError("H21 included and excluded cell mappings overlap")
    if set(eligible_by_key) != included_keys | excluded_keys:
        missing = sorted(set(eligible_by_key) - included_keys - excluded_keys)
        extra = sorted((included_keys | excluded_keys) - set(eligible_by_key))
        raise RuntimeError(f"H21 eligible-cell partition changed: missing={missing}, extra={extra}")
    if len(eligible_by_key) != _EXPECTED_ELIGIBLE_COUNT:
        raise RuntimeError("H21 eligible source-cell count changed")

    forbidden = _forbidden_phrases(source_labels)
    cases: list[dict[str, Any]] = []
    included_manifest: list[dict[str, Any]] = []
    for index, key in enumerate(sorted(included_keys), start=1):
        gold_method_id, question = _INCLUDED_CELLS[key]
        cell = eligible_by_key[key]
        if cell["mapped_method_id"] != gold_method_id:
            raise RuntimeError(f"H21 included source-cell gold changed: {key}")
        _assert_leak_free(question, forbidden)
        case = {
            "case_id": _case_id(index),
            "source_id": key[0],
            "question": question,
            "gold_method_id": gold_method_id,
        }
        cases.append(case)
        included_manifest.append(
            {
                "source_id": key[0],
                "table_id": key[1],
                "row_index": key[2],
                "column_index": key[3],
                "case_id": case["case_id"],
                "question": question,
                "gold_method_id": gold_method_id,
                "source_cell_fingerprint": canonical_hash(cell),
            }
        )

    excluded_manifest: list[dict[str, Any]] = []
    for key in sorted(excluded_keys):
        exclusion = dict(_EXCLUDED_CELLS[key])
        cell = eligible_by_key[key]
        if cell["mapped_method_id"] != exclusion["gold_method_id"]:
            raise RuntimeError(f"H21 excluded source-cell gold changed: {key}")
        excluded_manifest.append(
            {
                "source_id": key[0],
                "table_id": key[1],
                "row_index": key[2],
                "column_index": key[3],
                **exclusion,
                "source_cell_fingerprint": canonical_hash(cell),
            }
        )

    method_counts = Counter(str(case["gold_method_id"]) for case in cases)
    source_counts = Counter(str(case["source_id"]) for case in cases)
    if (
        len(cases) != _EXPECTED_INCLUDED_COUNT
        or len(excluded_manifest) != _EXPECTED_EXCLUDED_COUNT
        or len(source_counts) != _EXPECTED_SOURCE_COUNT
        or len(method_counts) != _EXPECTED_METHOD_COUNT
    ):
        raise RuntimeError(
            "H21 materialized count changed: "
            f"cases={len(cases)}, exclusions={len(excluded_manifest)}, "
            f"sources={len(source_counts)}, methods={len(method_counts)}"
        )
    if len({str(case["case_id"]) for case in cases}) != len(cases):
        raise RuntimeError("H21 materialized case IDs are not unique")

    partition = [
        {
            "source_id": key[0],
            "table_id": key[1],
            "row_index": key[2],
            "column_index": key[3],
            "disposition": "included" if key in included_keys else "ambiguity-excluded",
        }
        for key in sorted(eligible_by_key)
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "method": _METHOD,
        "extraction_version": 1,
        "snapshot_receipts": snapshot_receipts,
        "eligible_cell_count": len(eligible_by_key),
        "included_case_count": len(cases),
        "ambiguity_exclusion_count": len(excluded_manifest),
        "source_counts": dict(sorted(source_counts.items())),
        "method_counts": dict(sorted(method_counts.items())),
        "distinct_method_count": len(method_counts),
        "included": included_manifest,
        "ambiguity_exclusions": excluded_manifest,
        "eligible_partition": partition,
        "eligible_partition_fingerprint": canonical_hash(partition),
        "forbidden_phrase_set_fingerprint": canonical_hash(sorted(forbidden)),
        "question_policy": (
            "Only data type, pairing, group count, and outcome/predictor structure are allowed; "
            "source labels and catalog display names or aliases are rejected."
        ),
    }
    manifest["fingerprint"] = canonical_hash(manifest)
    return cases, manifest


def _data_payload(
    *,
    contract: dict[str, Any],
    cases: list[dict[str, Any]],
    manifest: dict[str, Any],
    cases_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema_version": 1,
        "method": _METHOD,
        "complete": True,
        "contract_fingerprint": contract["fingerprint"],
        "e5_source_screen_result_fingerprint": contract[
            "e5_source_screen_result_fingerprint"
        ],
        "case_count": len(cases),
        "eligible_cell_count": manifest["eligible_cell_count"],
        "ambiguity_exclusion_count": manifest["ambiguity_exclusion_count"],
        "source_count": len(manifest["source_counts"]),
        "distinct_method_count": manifest["distinct_method_count"],
        "source_counts": dict(manifest["source_counts"]),
        "method_counts": dict(manifest["method_counts"]),
        "cases_path": str(cases_path.resolve()),
        "cases_sha256": sha256_file(cases_path),
        "case_fingerprint": canonical_hash(cases),
        "question_extraction_manifest_path": str(manifest_path.resolve()),
        "question_extraction_manifest_sha256": sha256_file(manifest_path),
        "question_extraction_manifest_fingerprint": manifest["fingerprint"],
        "development_only": True,
        "fresh_external_evidence": False,
        "model_output_opened": False,
        "training_authorized": False,
        "champion_changed": False,
        "release_authorized": False,
    }
    data["data_fingerprint"] = canonical_hash(data)
    return data


def prepare_opened_source_residual_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_opened_source_residual_contract(config)
    cases, manifest = _extract_cases_and_manifest(config)
    data_root = _data_root(config)
    data_root.mkdir(parents=True, exist_ok=True)
    cases_path = data_root / "cases.jsonl"
    manifest_path = data_root / "question-extraction-manifest.json"
    _write_immutable_jsonl(cases_path, cases, label="case file")
    _write_immutable_json(manifest_path, manifest, label="question extraction manifest")
    data = _data_payload(
        contract=contract,
        cases=cases,
        manifest=manifest,
        cases_path=cases_path,
        manifest_path=manifest_path,
    )
    root = _root(config)
    _write_immutable_json(root / "data.json", data, label="data seal")
    write_json(_reports_root(config) / "opened-source-residual-repair-v1-data.json", data)
    return data


def _verify_data_and_load_cases(
    config: ProjectConfig, data: dict[str, Any], contract: dict[str, Any]
) -> list[dict[str, Any]]:
    if data.get("data_fingerprint") != _fingerprint_without(data, "data_fingerprint"):
        raise RuntimeError("H21 data fingerprint changed")
    if data.get("contract_fingerprint") != contract.get("fingerprint"):
        raise RuntimeError("H21 data belongs to another contract")
    if not data.get("complete") or data.get("model_output_opened"):
        raise RuntimeError("H21 data seal is incomplete or already opened")
    if data.get("training_authorized") or data.get("fresh_external_evidence"):
        raise RuntimeError("H21 data seal exceeded its development-only boundary")
    cases_path = Path(str(data.get("cases_path", ""))).resolve()
    manifest_path = Path(str(data.get("question_extraction_manifest_path", ""))).resolve()
    expected_root = _data_root(config).resolve()
    if not cases_path.is_relative_to(expected_root) or not manifest_path.is_relative_to(
        expected_root
    ):
        raise RuntimeError("H21 data artifacts escaped the H21 data root")
    if sha256_file(cases_path) != data.get("cases_sha256"):
        raise RuntimeError("H21 case file changed")
    if sha256_file(manifest_path) != data.get("question_extraction_manifest_sha256"):
        raise RuntimeError("H21 question extraction manifest changed")
    cases = list(read_jsonl(cases_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if canonical_hash(cases) != data.get("case_fingerprint"):
        raise RuntimeError("H21 case fingerprint changed")
    if manifest.get("fingerprint") != _fingerprint_without(manifest, "fingerprint"):
        raise RuntimeError("H21 extraction manifest fingerprint changed")
    if manifest.get("fingerprint") != data.get("question_extraction_manifest_fingerprint"):
        raise RuntimeError("H21 data belongs to another extraction manifest")
    expected_fields = {"case_id", "source_id", "question", "gold_method_id"}
    if any(set(case) != expected_fields for case in cases):
        raise RuntimeError("H21 case schema contains forbidden fields")
    if len(cases) != _EXPECTED_INCLUDED_COUNT or len(cases) != data.get("case_count"):
        raise RuntimeError("H21 materialized case count changed")
    if manifest.get("eligible_cell_count") != _EXPECTED_ELIGIBLE_COUNT:
        raise RuntimeError("H21 eligible cell count changed")
    if manifest.get("ambiguity_exclusion_count") != _EXPECTED_EXCLUDED_COUNT:
        raise RuntimeError("H21 ambiguity exclusion count changed")
    if len(manifest.get("eligible_partition", [])) != _EXPECTED_ELIGIBLE_COUNT:
        raise RuntimeError("H21 eligible partition is incomplete")
    if len({str(case["source_id"]) for case in cases}) != _EXPECTED_SOURCE_COUNT:
        raise RuntimeError("H21 materialized source count changed")
    if len({str(case["gold_method_id"]) for case in cases}) != _EXPECTED_METHOD_COUNT:
        raise RuntimeError("H21 materialized method count changed")
    return cases


def _opportunity_summary(private_details: dict[str, Any]) -> dict[str, Any]:
    candidate = private_details.get("candidate")
    if not isinstance(candidate, list) or any(not isinstance(row, dict) for row in candidate):
        raise RuntimeError("H21 exact runtime lost guarded candidate details")
    residual = [row for row in candidate if not bool(row.get("valid_output"))]
    source_counts = Counter(str(row["source_id"]) for row in residual)
    method_counts = Counter(str(row["gold_method_id"]) for row in residual)
    observed = {
        "residual_invalid_cases": len(residual),
        "residual_invalid_sources": len(source_counts),
        "residual_invalid_gold_methods": len(method_counts),
        "residual_invalid_source_counts": dict(sorted(source_counts.items())),
        "residual_invalid_method_counts": dict(sorted(method_counts.items())),
    }
    checks = {
        "minimum_residual_invalid_cases": len(residual)
        >= _OPPORTUNITY_GATES["minimum_residual_invalid_cases"],
        "minimum_residual_invalid_sources": len(source_counts)
        >= _OPPORTUNITY_GATES["minimum_residual_invalid_sources"],
        "minimum_residual_invalid_gold_methods": len(method_counts)
        >= _OPPORTUNITY_GATES["minimum_residual_invalid_gold_methods"],
    }
    qualified = all(checks.values())
    return {
        "qualified": qualified,
        "status": "OPPORTUNITY_QUALIFIED" if qualified else "INCONCLUSIVE_OPPORTUNITY",
        "gates": dict(_OPPORTUNITY_GATES),
        "checks": checks,
        "observed": observed,
    }


def _exact_runtime_gates() -> dict[str, Any]:
    return {
        "minimum_invalid_control_opportunities": 0,
        "minimum_invalid_control_gold_methods": 0,
        "minimum_valid_control_identity_cases": 0,
        "minimum_candidate_only_gains": 0,
        "minimum_net_improvements": 0,
        "minimum_accuracy_gain_points": 0.0,
        "maximum_mcnemar_p": 1.0,
    }


def _verify_exact_report_accounting(
    report: dict[str, Any], cases: list[dict[str, Any]], contract: dict[str, Any]
) -> None:
    private = report.get("private_details")
    if not isinstance(private, dict):
        raise RuntimeError("H21 terminal report lost private details")
    control = private.get("control")
    repair = private.get("repair")
    candidate = private.get("candidate")
    if not all(isinstance(rows, list) for rows in (control, repair, candidate)):
        raise RuntimeError("H21 terminal exact-runtime details are malformed")
    control = list(control)
    repair = list(repair)
    candidate = list(candidate)
    if len(control) != len(cases) or len(candidate) != len(cases):
        raise RuntimeError("H21 terminal exact-runtime case coverage changed")
    case_fields = ("case_id", "source_id", "question", "gold_method_id")
    for case, control_row, candidate_row in zip(cases, control, candidate, strict=True):
        if any(control_row.get(field) != case[field] for field in case_fields):
            raise RuntimeError("H21 terminal control details changed from the frozen cases")
        if any(candidate_row.get(field) != case[field] for field in case_fields):
            raise RuntimeError("H21 terminal candidate details changed from the frozen cases")

    control_by_id = {str(row.get("case_id")): row for row in control}
    repair_by_id = {str(row.get("case_id")): row for row in repair}
    if len(control_by_id) != len(control) or len(repair_by_id) != len(repair):
        raise RuntimeError("H21 terminal exact-runtime details contain duplicate IDs")
    invalid_ids = {
        str(row["case_id"]) for row in control if not bool(row.get("valid_output"))
    }
    if set(repair_by_id) != invalid_ids:
        raise RuntimeError("H21 terminal H20 calls do not match invalid controls")
    recomputed_candidate = [
        _apply_guard(control_by_id[str(case["case_id"])], repair_by_id.get(str(case["case_id"])))
        for case in cases
    ]
    if recomputed_candidate != candidate:
        raise RuntimeError("H21 terminal guarded candidate details changed")

    control_metrics = _strict_metrics(control)
    candidate_metrics = _strict_metrics(candidate)
    paired = _paired(control, candidate)
    source_paired = _source_paired(control, candidate)
    route_counts = dict(sorted(Counter(str(row["route"]) for row in candidate).items()))
    parse_reason_counts = {
        "control": dict(
            sorted(Counter(str(row["parse_reason"]) for row in control).items())
        ),
        "repair": dict(sorted(Counter(str(row["parse_reason"]) for row in repair).items())),
    }
    if report.get("scores") != {
        "control": control_metrics,
        "guarded_candidate": candidate_metrics,
    }:
        raise RuntimeError("H21 terminal scores changed")
    if report.get("paired") != paired or report.get("source_paired") != source_paired:
        raise RuntimeError("H21 terminal paired accounting changed")
    if report.get("route_counts") != route_counts:
        raise RuntimeError("H21 terminal route accounting changed")
    if report.get("parse_reason_counts") != parse_reason_counts:
        raise RuntimeError("H21 terminal parse-reason accounting changed")

    calls = {
        "control_calls": len(control),
        "control_invalid_count": len(invalid_ids),
        "repair_calls": len(repair),
        "valid_control_repair_calls": 0,
        "total_calls": len(control) + len(repair),
    }
    if report.get("model_call_counts") != calls:
        raise RuntimeError("H21 exact model-call accounting changed")
    valid_repairs = sum(bool(row["valid_output"]) for row in repair)
    correct_repairs = sum(bool(row["correct"]) for row in repair)
    fallback_count = sum(
        row["route"] == "invalid-control-invalid-repair-fallback" for row in candidate
    )
    expected_repair = {
        "valid_repair_count": valid_repairs,
        "correct_repair_count": correct_repairs,
        "fallback_count": fallback_count,
        "precision": correct_repairs / valid_repairs if valid_repairs else 0.0,
        "precision_wilson_95": _wilson_interval(correct_repairs, valid_repairs),
        "correct_rate_per_call": correct_repairs / len(repair) if repair else 0.0,
        "correct_rate_per_call_wilson_95": _wilson_interval(
            correct_repairs, len(repair)
        ),
    }
    if report.get("repair") != expected_repair:
        raise RuntimeError("H21 terminal H20 repair accounting changed")
    expected_external_gate = _external_gate(
        control=control_metrics,
        candidate=candidate_metrics,
        paired=paired,
        details=candidate,
        gates=_exact_runtime_gates(),
    )
    if report.get("external_gate") != expected_external_gate:
        raise RuntimeError("H21 terminal exact-runtime gate changed")
    if report.get("control_runtime_receipt_fingerprint") != canonical_hash(
        contract["parent_runtime_receipt"]
    ):
        raise RuntimeError("H21 terminal parent runtime receipt changed")
    if report.get("repair_runtime_receipt_fingerprint") != canonical_hash(
        contract["h20_runtime_receipt"]
    ):
        raise RuntimeError("H21 terminal H20 runtime receipt changed")


def _verify_h21_terminal_report(
    report: dict[str, Any],
    *,
    evaluation_fingerprint: str,
    contract: dict[str, Any],
    data: dict[str, Any],
    cases: list[dict[str, Any]],
) -> None:
    _verify_terminal_report(report)
    if report.get("method") != _METHOD:
        raise RuntimeError("H21 terminal report method changed")
    if report.get("evaluation_fingerprint") != evaluation_fingerprint:
        raise RuntimeError("H21 terminal report belongs to another evaluation")
    if report.get("contract_fingerprint") != contract.get("fingerprint"):
        raise RuntimeError("H21 terminal report belongs to another contract")
    if report.get("data_fingerprint") != data.get("data_fingerprint"):
        raise RuntimeError("H21 terminal report belongs to another data seal")
    if report.get("fresh_external_evidence") or report.get("champion_changed"):
        raise RuntimeError("H21 terminal report exceeded its development-only boundary")
    if report.get("release_authorized"):
        raise RuntimeError("H21 terminal report authorized a forbidden release")
    _verify_exact_report_accounting(report, cases, contract)
    private = report["private_details"]
    opportunity = _opportunity_summary(private)
    if report.get("opportunity_gate") != opportunity:
        raise RuntimeError("H21 terminal opportunity gate changed")
    if report.get("status") != opportunity["status"]:
        raise RuntimeError("H21 terminal status changed")
    if bool(report.get("training_authorized")) is not bool(opportunity["qualified"]):
        raise RuntimeError("H21 training authorization changed")


def run_opened_source_residual_opportunity(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_opened_source_residual_contract(config)
    data = prepare_opened_source_residual_data(config)
    cases = _verify_data_and_load_cases(config, data, contract)
    evaluation_fingerprint = canonical_hash(
        {
            "method": _METHOD,
            "method_version": _METHOD_VERSION,
            "contract": contract["fingerprint"],
            "data": data["data_fingerprint"],
            "e5_runtime": contract["e5_runtime_fingerprint"],
            "opportunity_gates": _OPPORTUNITY_GATES,
        }
    )
    root = _root(config)
    report_path = root / "report.json"
    public_path = _reports_root(config) / "opened-source-residual-repair-v1.json"
    opening_receipt = {
        "schema_version": 1,
        "method": _METHOD,
        "evaluation_fingerprint": evaluation_fingerprint,
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["data_fingerprint"],
        "case_count": len(cases),
        "evidence_status": "opened-source-development-only",
        "model_output_opened": True,
        "training_authorized": False,
        "fresh_external_evidence": False,
        "champion_changed": False,
        "release_authorized": False,
    }
    opening_receipt["fingerprint"] = canonical_hash(opening_receipt)
    opening_path = root / "opportunity-opened.json"
    if report_path.exists():
        if not opening_path.is_file():
            raise RuntimeError("H21 terminal report lost its model-output opening receipt")
        try:
            existing_opening = json.loads(opening_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise RuntimeError("H21 model-output opening receipt is unreadable") from error
        if existing_opening != opening_receipt:
            raise RuntimeError("H21 model-output opening receipt changed")
        try:
            existing = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise RuntimeError("H21 terminal report is unreadable") from error
        _verify_h21_terminal_report(
            existing,
            evaluation_fingerprint=evaluation_fingerprint,
            contract=contract,
            data=data,
            cases=cases,
        )
        public = _public_report(existing)
        write_json(public_path, public)
        return public

    _write_immutable_json(
        opening_path, opening_receipt, label="model-output opening receipt"
    )

    runtime = {
        "control": dict(contract["parent_runtime_receipt"]),
        "repair": dict(contract["h20_runtime_receipt"]),
    }
    progress_root = root / "progress"
    control_header = _phase_header(
        cases,
        phase="control",
        evaluation_fingerprint=evaluation_fingerprint,
        runtime_receipt=runtime["control"],
    )
    completed_controls = len(
        _load_strict_progress(progress_root / "control-ledger.json", header=control_header)
    )
    control_caller = _AgentCaller(
        config,
        adapter_path=str(runtime["control"]["adapter_path"]),
        expected_calls=len(cases) - completed_controls,
    )
    h20_caller = _AgentCaller(
        config,
        adapter_path=str(runtime["repair"]["adapter_path"]),
        expected_calls=None,
    )
    try:
        report = run_exact_two_stage(
            cases,
            evaluation_fingerprint=evaluation_fingerprint,
            control_runtime_receipt=runtime["control"],
            repair_runtime_receipt=runtime["repair"],
            control_caller=control_caller,
            repair_caller=h20_caller,
            progress_root=progress_root,
            gates=_exact_runtime_gates(),
        )
    finally:
        control_caller.close()
        h20_caller.close()

    opportunity = _opportunity_summary(report["private_details"])
    report.update(
        {
            "method": _METHOD,
            "evidence_status": "opened-source-development-only",
            "contract_fingerprint": contract["fingerprint"],
            "data_fingerprint": data["data_fingerprint"],
            "e5_source_screen_result_fingerprint": contract[
                "e5_source_screen_result_fingerprint"
            ],
            "h20_adapter_sha256": _H20_ADAPTER_SHA256,
            "opportunity_gate": opportunity,
            "status": opportunity["status"],
            "training_authorized": bool(opportunity["qualified"]),
            "fresh_external_evidence": False,
            "champion_changed": False,
            "release_authorized": False,
            "claim_boundary": contract["claim_boundary"],
        }
    )
    report["result_fingerprint"] = _terminal_fingerprint(report)
    _verify_h21_terminal_report(
        report,
        evaluation_fingerprint=evaluation_fingerprint,
        contract=contract,
        data=data,
        cases=cases,
    )
    _write_immutable_json(report_path, report, label="terminal report")
    public = _public_report(report)
    write_json(public_path, public)
    return public

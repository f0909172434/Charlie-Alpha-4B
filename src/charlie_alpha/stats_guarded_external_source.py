from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from .config import ProjectConfig
from .io_utils import (
    canonical_hash,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)
from .stats_guarded_external import _DECODING, _messages
from .stats_representation_probe import _METHOD_IDS
from .stats_training import _stats_snapshot

_SCREEN_VERSION = 1

_SOURCES: dict[str, dict[str, Any]] = {
    "PMC8327789": {
        "title": "An Introduction to Statistics: Choosing the Correct Statistical Test",
        "doi": "10.5005/jp-journals-10071-23815",
        "license": "CC BY-NC",
        "snapshot_url": (
            "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/"
            "BioC_json/PMC8327789/unicode"
        ),
        "snapshot_sha256": "b40dc53865028489d13c010811c56947b8ba9e3f3353a83e324a0fb09392196f",
        "tables": {"T1": 3, "T2": 3, "T3": 2, "T4": 2},
        "method_columns": {"T1": [1, 2], "T2": [1, 2], "T3": [1], "T4": [1]},
        "aliases": {
            "Mann–Whitney U-test (Wilcoxon rank sum test)": "mann_whitney",
            "Unpaired t-test": "independent_t",
            "Wilcoxon signed rank test": "wilcoxon_signed_rank",
            "Paired t-test": "paired_t",
            "Linear regression analysis": "ols",
            "Logistic regression analysis": "logistic_glm",
        },
    },
    "PMC8483143": {
        "title": (
            "How to choose and interpret a statistical test? An update for budding researchers"
        ),
        "doi": "10.4103/jfmpc.jfmpc_433_21",
        "license": "CC BY-NC-SA",
        "snapshot_url": (
            "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/"
            "BioC_json/PMC8483143/unicode"
        ),
        "snapshot_sha256": "326ab7eae0719c13cf48e58b98973795f4765ba922e4b684513f9862b026d49f",
        "tables": {"T2": 7},
        "method_columns": {"T2": [1, 2, 3, 4, 5, 6]},
        "aliases": {
            "Paired t-test": "paired_t",
            "Unpaired t-test": "independent_t",
            "Linear regression": "ols",
            "Wilcox on signed-rank test": "wilcoxon_signed_rank",
            "Mann-Whitney U test": "mann_whitney",
            "Chi-square test": "chi_square",
            "Logistic regression": "logistic_glm",
        },
    },
    "PMC6639881": {
        "title": "Selection of Appropriate Statistical Methods for Data Analysis",
        "doi": "10.4103/aca.ACA_248_18",
        "license": "CC BY-NC-SA",
        "snapshot_url": (
            "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/"
            "BioC_json/PMC6639881/unicode"
        ),
        "snapshot_sha256": "b881306838e425ae7319ef17c0fa7e690e350bd5a53a430c50b8bc271ee980fd",
        "tables": {"T1": 3, "T2": 3, "T3": 3},
        "method_columns": {"T1": [1, 2], "T2": [1], "T3": [1]},
        "aliases": {
            "One sample Wilcoxon signed rank test": "wilcoxon_signed_rank",
            "Independent samples t-test (Unpaired samples t-test)": "independent_t",
            "Mann Whitney U test/Wilcoxon rank sum test": "mann_whitney",
            "Paired samples t-test": "paired_t",
            "Related samples Wilcoxon signed-rank test": "wilcoxon_signed_rank",
            "Linear regression model": "ols",
            "Z test for proportions": "two_proportion",
            "Binary Logistic regression analysis": "logistic_glm",
            "Cox regression analysis": "cox_ph",
        },
    },
    "PMC2996580": {
        "title": "Choosing statistical test",
        "doi": "10.4103/0974-7788.72494",
        "license": "CC BY",
        "snapshot_url": (
            "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/"
            "BioC_json/PMC2996580/unicode"
        ),
        "snapshot_sha256": "eb7be3715e4a5f7e6d3bb609285d361551d44a74f6620a052c1dd8523325c2b9",
        "tables": {"T0001": 2},
        "method_columns": {"T0001": [0]},
        "aliases": {
            "Fisher’s exact test": "fisher_exact",
            "Chi-square test": "chi_square",
            (
                "Wilcoxon’s rank sum test (also known as the unpaired Wilcoxon rank sum test "
                "or the Mann–Whitney U test)"
            ): "mann_whitney",
            "Log rank test": "logrank",
        },
    },
}


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "guarded-external-v1"


def _data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "guarded-external-v1"


def _implementation_manifest() -> dict[str, str]:
    source_root = Path(__file__).resolve().parent
    names = [
        "stats_guarded_external_source.py",
        "stats_guarded_external.py",
        "stats_agent.py",
        "stats_catalog.py",
        "stats_catalog_grounding.py",
        "stats_cross_format.py",
        "stats_external_catalog.py",
        "stats_representation_probe.py",
        "stats_training.py",
    ]
    return {name: sha256_file(source_root / name) for name in names}


def _validate_source_catalog() -> None:
    if len(_METHOD_IDS) != 28 or len(set(_METHOD_IDS)) != 28:
        raise RuntimeError("E5 requires the frozen 28-method repository catalog")
    invalid = {
        f"{pmcid}:{label}": method_id
        for pmcid, source in _SOURCES.items()
        for label, method_id in source["aliases"].items()
        if method_id not in _METHOD_IDS
    }
    if invalid:
        raise RuntimeError(
            "E5 source aliases target methods outside the frozen catalog: "
            f"{invalid}"
        )


def _adapter_receipt(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    weights = path / "adapters.safetensors"
    config = path / "adapter_config.json"
    if not weights.is_file() or not config.is_file():
        raise RuntimeError(f"E5 adapter is incomplete: {path}")
    actual = sha256_file(weights)
    if actual != expected_sha256:
        raise RuntimeError(f"E5 adapter weights changed: {path}")
    return {
        "adapter_path": str(path),
        "adapter_sha256": actual,
        "adapter_config_sha256": sha256_file(config),
    }


def _runtime_manifest(config: ProjectConfig) -> dict[str, Any]:
    h20_contract_path = (
        config.root / "reports" / "evolve" / "guarded-weight-bridge-v1-contract.json"
    )
    h20_result_path = (
        config.root / "reports" / "evolve" / "guarded-weight-bridge-v1-training.json"
    )
    h20_contract = json.loads(h20_contract_path.read_text(encoding="utf-8"))
    h20_result = json.loads(h20_result_path.read_text(encoding="utf-8"))
    if not h20_result.get("complete") or not h20_result.get("historical_gate", {}).get(
        "passed"
    ):
        raise RuntimeError("E5 requires the complete development-only H20 runtime")
    if h20_result.get("champion_changed") or h20_result.get("release_authorized"):
        raise RuntimeError("E5 requires H20 to remain development-only")
    result_unsigned = {
        key: value for key, value in h20_result.items() if key != "result_fingerprint"
    }
    if h20_result.get("result_fingerprint") != canonical_hash(result_unsigned):
        raise RuntimeError("E5 H20 result fingerprint changed")

    parent_path = Path(str(h20_contract["parent"]["adapter_path"]))
    final = dict(h20_result["final_all_source_adapter"])
    repair_path = Path(str(final["adapter_path"]))
    model_source = dict(config.sources["models"]["research_base_mlx_4bit"])
    snapshot = Path(_stats_snapshot(config))
    if snapshot.name != str(model_source["revision"]):
        raise RuntimeError("E5 base-model snapshot revision changed")
    model_files = [
        "chat_template.jinja",
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_files = sorted({str(value) for value in index.get("weight_map", {}).values()})
    if not weight_files:
        raise RuntimeError("E5 base-model weight index is empty")
    model_files.extend(weight_files)
    for name in model_files:
        if not (snapshot / name).is_file():
            raise RuntimeError(f"E5 base-model runtime file is missing: {name}")
    manifest: dict[str, Any] = {
        "h20_contract_fingerprint": h20_contract["fingerprint"],
        "h20_contract_sha256": sha256_file(h20_contract_path),
        "h20_result_fingerprint": h20_result["result_fingerprint"],
        "h20_result_sha256": sha256_file(h20_result_path),
        "base_model": {
            "repo_id": model_source["repo_id"],
            "revision": model_source["revision"],
            "license": model_source["license"],
            "snapshot_path": str(snapshot),
            "file_sha256": {name: sha256_file(snapshot / name) for name in model_files},
        },
        "control": _adapter_receipt(
            parent_path, expected_sha256=str(h20_contract["parent"]["adapter_sha256"])
        ),
        "repair": _adapter_receipt(
            repair_path, expected_sha256=str(final["adapter_sha256"])
        ),
        "prompt_messages_sha256": canonical_hash(_messages({"question": "<SOURCE_QUESTION>"})),
        "decoding": dict(_DECODING),
    }
    manifest["fingerprint"] = canonical_hash(manifest)
    return manifest


def prepare_guarded_external_master_contract(config: ProjectConfig) -> dict[str, Any]:
    _validate_source_catalog()
    settings = dict(config.section("guarded_external"))
    runtime = _runtime_manifest(config)
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "master-contract.json"
    public_path = (
        config.root / "reports" / "evolve" / "guarded-external-v1-master-contract.json"
    )
    opened_sources = {
        pmcid: {
            "doi": source["doi"],
            "snapshot_sha256": source["snapshot_sha256"],
            "status": "opened-during-precontract-feasibility-screen",
        }
        for pmcid, source in _SOURCES.items()
    }
    opportunity_gates = dict(settings["opportunity_gates"])
    evaluation_gates = dict(settings["evaluation_gates"])
    gates = {**opportunity_gates, **evaluation_gates}
    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": "E5 exact H20 two-stage single-source fresh external master protocol",
        "method_version": int(settings["method_version"]),
        "causal_question": (
            "On one genuinely new source selected and frozen before opening, can the exact H20 "
            "runtime preserve every parser-valid v0.3 control and correctly repair enough "
            "parser-invalid controls to produce a statistically distinguishable paired gain?"
        ),
        "evidence_status": "master-protocol-no-source-selected",
        "runtime": runtime,
        "runtime_policy": [
            "Generate every control exactly once with the unchanged v0.3 parent.",
            "Never instantiate or call the repair adapter for a parser-valid control.",
            "Call the H20 repair adapter exactly once for each parser-invalid control, using the "
            "same messages and decoding without exposing the control answer.",
            "Use a parser-valid repair; otherwise retain the original invalid control.",
        ],
        "parser_contract": (
            "Strip thinking text, decode the first JSON object, require methods to be a one-item "
            "array, and accept only a frozen catalog identifier or pre-existing catalog alias. "
            "Empty, malformed, multi-method, and unknown-ID outputs are scientific invalids."
        ),
        "progress_contract": (
            "Each phase uses one atomically rewritten ledger binding ordered case, prompt, "
            "request, adapter, runtime, parser, and decoding receipts. Any mismatch fails closed "
            "and never clears progress."
        ),
        "source_protocol": {
            "source_count": 1,
            "selection_timing": (
                "Freeze a child contract with stable source ID, version, snapshot URL, license, "
                "complete-frame extraction rule, alias map, and all gates before opening content."
            ),
            "qualification_gates": dict(settings["source_gates"]),
            "overlap_rule": (
                "Reject any source, version, derivative, or normalized question overlapping E2-E4, "
                "H17-H20, training, rule design, or the precontract feasibility screen."
            ),
            "opened_source_exclusions": opened_sources,
            "gold_isolation": "Gold and source answer fields are never passed to model-call code.",
        },
        "evaluation_gates": gates,
        "terminal_states": [
            "CONFIRMED_NARROW_PASS",
            "SCIENTIFIC_FAIL",
            "INCONCLUSIVE_REPAIR_COVERAGE",
            "SOURCE_UNQUALIFIED",
            "PROTOCOL_INVALID",
        ],
        "stopping_rule": (
            "Every terminal state closes the selected child E5. Do not replace the source, add "
            "cases, retry scientific invalids, tune H20, or rerun on the same source."
        ),
        "historical_runtime_audit": {
            "h20_evidence_remains_development_only": True,
            "exact_two_stage_model_call_policy_previously_demonstrated": False,
            "h20_smoke_adapter_calls": 38,
            "exact_policy_adapter_calls_for_same_smoke_controls": 9,
            "historical_prompt_comparable_cases": 22,
            "historical_prompt_incomparable_cases": 16,
            "reported_control_in_catalog_prediction_rate": 1.0,
            "correct_control_in_catalog_prediction_rate": 29 / 38,
            "accuracy_and_h20_development_gate_affected_by_metric_defect": False,
            "implication": (
                "H20 motivates E5 but its reported smoke and all 38 paired rows cannot serve as "
                "an exact runtime confirmation."
            ),
        },
        "implementation_sha256": _implementation_manifest(),
        "source_selected": False,
        "model_output_opened": False,
        "champion_changed": False,
        "release_authorized": False,
        "claim_boundary": (
            "A pass may support only a narrow prospective claim for the exact H20 runtime on one "
            "source. It cannot establish multi-source generality, raw-adapter superiority, "
            "champion replacement, publication, or release readiness."
        ),
    }
    contract["fingerprint"] = canonical_hash(contract)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"] or existing != contract:
            raise RuntimeError("E5 master contract is immutable")
        write_json(public_path, existing)
        return existing
    write_json(lock_path, contract)
    write_json(public_path, contract)
    return contract


def _download_snapshot(path: Path, source: dict[str, Any]) -> bytes:
    expected = str(source["snapshot_sha256"])
    if path.exists():
        if sha256_file(path) != expected:
            raise RuntimeError(f"E5 source-screen snapshot changed: {path.name}")
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(3):
        request = urllib.request.Request(
            str(source["snapshot_url"]),
            headers={"User-Agent": "CharlieAlphaResearch/1.0 (+source-screen)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
            break
        except (urllib.error.HTTPError, OSError) as error:
            last_error = error
            if attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))
    else:  # pragma: no cover - the loop either breaks or raises.
        raise RuntimeError(f"E5 source-screen download failed: {last_error}")
    if sha256_bytes(payload) != expected:
        raise RuntimeError(f"E5 source-screen remote snapshot changed: {path.name}")
    path.write_bytes(payload)
    return payload


def _bioc_document(
    payload: bytes,
    *,
    pmcid: str,
    expected_license: str,
    expected_doi: str,
    expected_title: str,
) -> dict[str, Any]:
    collection = json.loads(payload)
    if not isinstance(collection, list) or len(collection) != 1:
        raise RuntimeError(f"E5 {pmcid} BioC collection changed")
    documents = collection[0].get("documents", [])
    if not isinstance(documents, list) or len(documents) != 1:
        raise RuntimeError(f"E5 {pmcid} BioC document count changed")
    document = documents[0]
    if str(document.get("id")) != pmcid:
        raise RuntimeError(f"E5 {pmcid} BioC identity changed")
    if document.get("infons", {}).get("license") != expected_license:
        raise RuntimeError(f"E5 {pmcid} BioC license changed")
    passages = document.get("passages", [])
    if not passages:
        raise RuntimeError(f"E5 {pmcid} BioC passages are missing")
    front = passages[0]
    front_infons = front.get("infons", {})
    if front_infons.get("article-id_pmc") != pmcid:
        raise RuntimeError(f"E5 {pmcid} front-matter PMCID changed")
    if str(front_infons.get("article-id_doi", "")).lower() != expected_doi.lower():
        raise RuntimeError(f"E5 {pmcid} DOI changed")
    if str(front.get("text", "")) != expected_title:
        raise RuntimeError(f"E5 {pmcid} title changed")
    return document


def _table_grid(document: dict[str, Any], *, table_id: str, columns: int) -> list[list[str]]:
    matches = [
        passage["infons"]["xml"]
        for passage in document.get("passages", [])
        if passage.get("infons", {}).get("id") == table_id
        and isinstance(passage.get("infons", {}).get("xml"), str)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"E5 source-screen table changed: {table_id}")
    table = ET.fromstring(str(matches[0]))
    active: dict[int, tuple[int, str]] = {}
    rows: list[list[str]] = []
    for tr in table.findall(".//tbody/tr"):
        row: list[str | None] = [None] * columns
        next_active: dict[int, tuple[int, str]] = {}
        for column, (remaining, text) in sorted(active.items()):
            row[column] = text
            if remaining > 1:
                next_active[column] = (remaining - 1, text)
        cursor = 0
        for cell in tr.findall("td"):
            while cursor < columns and row[cursor] is not None:
                cursor += 1
            if cursor >= columns:
                raise RuntimeError(f"E5 source-screen table exceeds expected width: {table_id}")
            text = " ".join("".join(cell.itertext()).split())
            colspan = int(cell.get("colspan", "1"))
            rowspan = int(cell.get("rowspan", "1"))
            for offset in range(colspan):
                column = cursor + offset
                if column >= columns:
                    raise RuntimeError(
                        f"E5 source-screen table exceeds expected width: {table_id}"
                    )
                row[column] = text
                if rowspan > 1:
                    next_active[column] = (rowspan - 1, text)
            cursor += colspan
        active = next_active
        rows.append([value or "" for value in row])
    return rows


def _method_cells(
    *, pmcid: str, source: dict[str, Any], document: dict[str, Any]
) -> list[dict[str, Any]]:
    _validate_source_catalog()
    cells: list[dict[str, Any]] = []
    aliases = dict(source["aliases"])
    for table_id, columns in source["tables"].items():
        grid = _table_grid(document, table_id=table_id, columns=int(columns))
        for row_index, row in enumerate(grid):
            for column in source["method_columns"][table_id]:
                label = str(row[int(column)]).strip()
                if not label:
                    continue
                condition_columns = [
                    value for index, value in enumerate(row) if index != int(column) and value
                ]
                cells.append(
                    {
                        "source_id": pmcid,
                        "table_id": table_id,
                        "row_index": row_index,
                        "column_index": int(column),
                        "condition": " | ".join(condition_columns),
                        "source_label": label,
                        "mapped_method_id": aliases.get(label),
                    }
                )
    return cells


def _source_summary(
    *, pmcid: str, source: dict[str, Any], cells: list[dict[str, Any]], gates: dict[str, Any]
) -> dict[str, Any]:
    eligible = [cell for cell in cells if cell["mapped_method_id"] in _METHOD_IDS]
    counts = Counter(str(cell["mapped_method_id"]) for cell in eligible)
    coverage = len(eligible) / len(cells) if cells else 0.0
    maximum_share = max(counts.values()) / len(eligible) if eligible else 1.0
    checks = {
        "minimum_eligible_cases": len(eligible) >= int(gates["minimum_eligible_cases"]),
        "minimum_distinct_methods": len(counts) >= int(gates["minimum_distinct_methods"]),
        "minimum_coverage_fraction": coverage >= float(gates["minimum_coverage_fraction"]),
        "maximum_single_method_fraction": maximum_share
        <= float(gates["maximum_single_method_fraction"]),
        "source_blind": False,
    }
    return {
        "source_id": pmcid,
        "title": source["title"],
        "doi": source["doi"],
        "license": source["license"],
        "snapshot_url": source["snapshot_url"],
        "snapshot_sha256": source["snapshot_sha256"],
        "method_bearing_cell_count": len(cells),
        "eligible_count": len(eligible),
        "coverage_fraction": coverage,
        "distinct_method_count": len(counts),
        "method_counts": dict(sorted(counts.items())),
        "maximum_single_method_fraction": maximum_share,
        "checks": checks,
        "qualified": all(checks.values()),
        "opened_before_master_contract": True,
    }


def _result_fingerprint(report: dict[str, Any]) -> str:
    unsigned = {
        key: value
        for key, value in report.items()
        if key not in {"result_fingerprint", "private_details"}
    }
    return canonical_hash(unsigned)


def prepare_guarded_external_source_screen(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_guarded_external_master_contract(config)
    gates = dict(contract["source_protocol"]["qualification_gates"])
    snapshot_root = _data_root(config) / "source-screen" / "snapshots"
    all_cells: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    for pmcid, source in _SOURCES.items():
        path = snapshot_root / f"{pmcid}.bioc.json"
        payload = _download_snapshot(path, source)
        document = _bioc_document(
            payload,
            pmcid=pmcid,
            expected_license=str(source["license"]),
            expected_doi=str(source["doi"]),
            expected_title=str(source["title"]),
        )
        cells = _method_cells(pmcid=pmcid, source=source, document=document)
        all_cells[pmcid] = cells
        summaries[pmcid] = _source_summary(
            pmcid=pmcid, source=source, cells=cells, gates=gates
        )
    if any(summary["qualified"] for summary in summaries.values()):
        raise RuntimeError("E5 precontract screen unexpectedly contains a qualified source")
    aggregate_cells = [cell for pmcid in sorted(all_cells) for cell in all_cells[pmcid]]
    aggregate_eligible = [
        cell for cell in aggregate_cells if cell["mapped_method_id"] in _METHOD_IDS
    ]
    private_details = {"source_cells": all_cells}
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "E5 precontract source feasibility screen",
        "screen_version": _SCREEN_VERSION,
        "master_contract_fingerprint": contract["fingerprint"],
        "source_screen_not_blind": True,
        "screened_sources": summaries,
        "aggregate_diagnostic_only": {
            "source_count": len(summaries),
            "method_bearing_cell_count": len(aggregate_cells),
            "eligible_count": len(aggregate_eligible),
            "coverage_fraction": len(aggregate_eligible) / len(aggregate_cells),
            "distinct_methods": sorted(
                {str(cell["mapped_method_id"]) for cell in aggregate_eligible}
            ),
            "pooling_for_e5_forbidden": True,
        },
        "status": "SOURCE_SCREEN_UNQUALIFIED",
        "evaluation_authorized": False,
        "model_output_opened": False,
        "champion_changed": False,
        "release_authorized": False,
        "next_step": (
            "preserve-zero-model-output-and-select-one-future-source-under-master-contract"
        ),
        "claim_boundary": (
            "This is a non-blind feasibility screen and a source-unavailable outcome, not a model "
            "evaluation. Screened sources cannot be pooled or reused as fresh E5 evidence."
        ),
        "private_details_fingerprint": canonical_hash(private_details),
        "private_details": private_details,
    }
    report["result_fingerprint"] = _result_fingerprint(report)
    internal_path = _root(config) / "source-screen.json"
    if internal_path.exists():
        existing = json.loads(internal_path.read_text(encoding="utf-8"))
        if existing.get("result_fingerprint") != report["result_fingerprint"] or existing != report:
            raise RuntimeError("E5 source screen changed")
    else:
        write_json(internal_path, report)
    public = {key: value for key, value in report.items() if key != "private_details"}
    write_json(
        config.root / "reports" / "evolve" / "guarded-external-v1-source-screen.json",
        public,
    )
    return public


def prepare_guarded_external_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_guarded_external_master_contract(config)
    screen = prepare_guarded_external_source_screen(config)
    if screen.get("evaluation_authorized"):
        raise RuntimeError("E5 source screen unexpectedly authorized evaluation")
    case_path = _data_root(config) / "cases.jsonl"
    write_jsonl(case_path, [])
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "master_contract_fingerprint": contract["fingerprint"],
        "source_screen_result_fingerprint": screen["result_fingerprint"],
        "source_selected": False,
        "case_count": 0,
        "cases_sha256": sha256_file(case_path),
        "evaluation_authorized": False,
        "model_output_opened": False,
        "terminal_state": "SOURCE_UNQUALIFIED",
        "reason": (
            "No single source was frozen before opening and no screened source meets the 150-case, "
            "eight-method, 80%-coverage, and concentration gates."
        ),
        "champion_changed": False,
        "release_authorized": False,
    }
    report["data_fingerprint"] = canonical_hash(report)
    internal_path = _root(config) / "data.json"
    public_path = config.root / "reports" / "evolve" / "guarded-external-v1-data.json"
    if internal_path.exists():
        existing = json.loads(internal_path.read_text(encoding="utf-8"))
        if existing != report:
            raise RuntimeError("E5 source-unqualified data receipt changed")
    else:
        write_json(internal_path, report)
    write_json(public_path, report)
    return report

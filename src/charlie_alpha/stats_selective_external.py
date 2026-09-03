from __future__ import annotations

import gc
import json
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import mlx.core as mx

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_bytes, sha256_file, write_json, write_jsonl
from .stats_agent import StatsAgent
from .stats_external_catalog import _metrics, _paired_summary
from .stats_selector_external_amendment import _evaluate_control
from .stats_selector_head import _load_head
from .stats_selector_runtime import _hidden_vector, _rank_methods, _runtime_case
from .stats_selector_sufficiency import _load_training_bank, _support_scores

_EVALUATOR_VERSION = 1

_GENSER_ALIASES: dict[str, tuple[str, str | None]] = {
    "t-test": ("independent_t", None),
    "Mann Whitney-U test": ("mann_whitney", None),
    "Paired t-test": ("paired_t", None),
    "Logistic regression": ("logistic_glm", None),
}

_TURNER_ALIASES: dict[str, tuple[str, str | None]] = {
    "Paired t-test (paired samples) (assuming normally distributed data)": (
        "paired_t",
        "assuming normally distributed data",
    ),
    "Wilcoxon signed-rank test (non-parametric alternative) (data not normally distributed)": (
        "wilcoxon_signed_rank",
        "data not normally distributed",
    ),
    "Student’s t-test (independent samples)": ("independent_t", "independent samples"),
    "Mann–Whitney U test (non-parametric alternative if data not normally distributed)": (
        "mann_whitney",
        "data not normally distributed",
    ),
    "Simple linear regression": ("ols", None),
    "logistic regression": ("logistic_glm", None),
    "Chi-square test (categorical data)": ("chi_square", "categorical data"),
    "Chi-square test": ("chi_square", None),
    "Fisher’s exact test (small sample sizes)": ("fisher_exact", "small sample sizes"),
}


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "selective-external-v1"


def _data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "selective-external-v1"


def _implementation_manifest() -> dict[str, str]:
    source_root = Path(__file__).resolve().parent
    return {
        "stats_selective_external.py": sha256_file(Path(__file__)),
        "stats_selector_sufficiency.py": sha256_file(
            source_root / "stats_selector_sufficiency.py"
        ),
        "stats_selector_runtime.py": sha256_file(source_root / "stats_selector_runtime.py"),
        "stats_selector_external_amendment.py": sha256_file(
            source_root / "stats_selector_external_amendment.py"
        ),
        "stats_agent.py": sha256_file(source_root / "stats_agent.py"),
    }


def _download_snapshot(url: str, path: Path, expected_sha256: str) -> bytes:
    if path.exists():
        if sha256_file(path) != expected_sha256:
            raise RuntimeError(f"Frozen external source snapshot changed: {path.name}")
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(3):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "CharlieAlphaResearch/1.0 (+reproducible-evaluation)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
            break
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))
        except OSError as error:
            last_error = error
            if attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))
    else:
        raise RuntimeError(f"Could not download source snapshot: {last_error}")
    actual = sha256_bytes(payload)
    if actual != expected_sha256:
        raise RuntimeError(
            "External source snapshot hash changed for "
            f"{url}: expected {expected_sha256}, got {actual}"
        )
    path.write_bytes(payload)
    return payload


def _bioc_document(payload: bytes, *, pmcid: str) -> dict[str, Any]:
    collection = json.loads(payload)
    if not isinstance(collection, list) or len(collection) != 1:
        raise RuntimeError(f"{pmcid} BioC snapshot must contain exactly one collection")
    documents = collection[0].get("documents", [])
    if not isinstance(documents, list) or len(documents) != 1:
        raise RuntimeError(f"{pmcid} BioC snapshot must contain exactly one document")
    document = documents[0]
    if str(document.get("id")) != pmcid:
        raise RuntimeError(f"{pmcid} BioC document identity changed")
    if document.get("infons", {}).get("license") != "CC BY":
        raise RuntimeError(f"{pmcid} source is no longer marked CC BY")
    return document


def _table_xml(document: dict[str, Any], *, table_id: str) -> str:
    matches = [
        passage
        for passage in document.get("passages", [])
        if passage.get("infons", {}).get("id") == table_id
        and isinstance(passage.get("infons", {}).get("xml"), str)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one XML passage for table {table_id}")
    return str(matches[0]["infons"]["xml"])


def _cell_text(cell: ET.Element) -> str:
    return " ".join("".join(cell.itertext()).split())


def _table_grid(xml: str, *, columns: int = 4) -> list[list[str]]:
    table = ET.fromstring(xml)
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
                break
            text = _cell_text(cell)
            colspan = int(cell.get("colspan", "1"))
            rowspan = int(cell.get("rowspan", "1"))
            for offset in range(colspan):
                column = cursor + offset
                if column >= columns:
                    raise RuntimeError("External source table exceeds four expected columns")
                row[column] = text
                if rowspan > 1:
                    next_active[column] = (rowspan - 1, text)
            cursor += colspan
        active = next_active
        rows.append([value or "" for value in row])
    return rows


def _genser_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    grid = _table_grid(_table_xml(document, table_id="T2"))
    rows: list[dict[str, Any]] = []
    previous_question = ""
    for source_row_index, row in enumerate(grid):
        question, data_type, assumptions, method_label = row
        if question:
            previous_question = question
        if not method_label:
            continue
        effective_question = question or previous_question
        mapped = _GENSER_ALIASES.get(method_label)
        rows.append(
            {
                "source_row_index": source_row_index,
                "research_question": effective_question,
                "data_type": data_type,
                "assumptions": assumptions,
                "source_method_label": method_label,
                "mapped_method_id": mapped[0] if mapped else None,
                "condition_from_method_label": mapped[1] if mapped else None,
            }
        )
    return rows


def _turner_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    grid = _table_grid(_table_xml(document, table_id="Tab1"))
    rows: list[dict[str, Any]] = []
    for source_row_index, row in enumerate(grid):
        question, scenario, method_label, example = row
        if not method_label:
            continue
        mapped = _TURNER_ALIASES.get(method_label)
        rows.append(
            {
                "source_row_index": source_row_index,
                "research_question": question,
                "assessment_scenario": scenario,
                "example": example,
                "source_method_label": method_label,
                "mapped_method_id": mapped[0] if mapped else None,
                "condition_from_method_label": mapped[1] if mapped else None,
            }
        )
    return rows


def _case_from_source_row(source_id: str, row: dict[str, Any]) -> dict[str, Any]:
    method_id = row.get("mapped_method_id")
    if not isinstance(method_id, str):
        raise ValueError("Cannot materialize an unmapped external row")
    if source_id == "genser-2007-immunology":
        parts = [
            f"Research question: {row['research_question']}",
            f"Data: {row['data_type']}",
        ]
        if row.get("assumptions"):
            parts.append(f"Conditions: {row['assumptions']}")
    elif source_id == "turner-2025-medical-education":
        parts = [
            f"Research question: {row['research_question']}",
            f"Assessment scenario: {row['assessment_scenario']}",
            f"Example: {row['example']}",
        ]
        if row.get("condition_from_method_label"):
            parts.append(f"Source condition: {row['condition_from_method_label']}")
    else:
        raise ValueError(f"Unknown external source: {source_id}")
    parts.append("Choose the single primary statistical analysis.")
    return {
        "case_id": f"e4-{source_id}-{int(row['source_row_index']):02d}",
        "source_id": source_id,
        "source_row_index": int(row["source_row_index"]),
        "question": "\n".join(parts),
        "gold_method_id": method_id,
        "gold_methods": [method_id],
        "gold_columns": [],
        "source_method_label": str(row["source_method_label"]),
        "eligible": True,
        "head_eligible": True,
    }


def _source_specs(config: ProjectConfig) -> list[dict[str, Any]]:
    sources = config.section("selective_external").get("sources")
    if not isinstance(sources, list) or len(sources) != 2:
        raise RuntimeError("E4 requires exactly two frozen source specifications")
    return [dict(source) for source in sources]


def _source_rows(
    config: ProjectConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_root = _root(config) / "sources"
    all_rows: list[dict[str, Any]] = []
    receipts: dict[str, Any] = {}
    for source in _source_specs(config):
        source_id = str(source["source_id"])
        snapshot_path = source_root / f"{source_id}.json"
        payload = _download_snapshot(
            str(source["bioc_url"]), snapshot_path, str(source["bioc_sha256"])
        )
        document = _bioc_document(payload, pmcid=str(source["pmcid"]))
        if source_id == "genser-2007-immunology":
            rows = _genser_rows(document)
        elif source_id == "turner-2025-medical-education":
            rows = _turner_rows(document)
        else:
            raise RuntimeError(f"No E4 extraction rule for {source_id}")
        mapped = [row for row in rows if row.get("mapped_method_id") is not None]
        receipts[source_id] = {
            "title": source["title"],
            "doi": source["doi"],
            "pmcid": source["pmcid"],
            "license": "CC BY",
            "bioc_url": source["bioc_url"],
            "snapshot_path": str(snapshot_path),
            "snapshot_sha256": sha256_file(snapshot_path),
            "table_id": source["table_id"],
            "method_row_count": len(rows),
            "mapped_row_count": len(mapped),
            "mapped_method_count": len({str(row["mapped_method_id"]) for row in mapped}),
            "rows_sha256": canonical_hash(rows),
        }
        all_rows.extend({"source_id": source_id, **row} for row in rows)
    return all_rows, receipts


def prepare_selective_external_source_qualification(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("selective_external"))
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / "source-qualification.json"
    public_path = (
        config.root
        / "reports"
        / "evolve"
        / "selector-sufficiency-external-v1-source-qualification.json"
    )
    rows, receipts = _source_rows(config)
    mapped = [row for row in rows if row.get("mapped_method_id") is not None]
    cases = [_case_from_source_row(str(row["source_id"]), row) for row in mapped]
    counts = Counter(str(case["source_id"]) for case in cases)
    methods = sorted({str(case["gold_method_id"]) for case in cases})
    gates = dict(settings["qualification_gates"])
    checks = {
        "source_count": len(receipts) >= int(gates["minimum_sources"]),
        "eligible_count": len(cases) >= int(gates["minimum_eligible_cases"]),
        "per_source_count": min(counts.values(), default=0)
        >= int(gates["minimum_eligible_per_source"]),
        "distinct_methods": len(methods) >= int(gates["minimum_distinct_methods"]),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "E4 two-source exact-alias source qualification",
        "source_content_opened_for_qualification": True,
        "model_output_opened": False,
        "sources": receipts,
        "extraction_policy": (
            "Parse the complete frozen source table. Include every method-bearing row whose exact "
            "source method label appears in the predeclared source-specific alias map. Do not "
            "correct, broaden, or add aliases after model output. Source-authored parenthetical "
            "conditions are retained as prompt conditions after removing the method name."
        ),
        "alias_maps": {
            "genser-2007-immunology": {
                label: {"method_id": value[0], "condition": value[1]}
                for label, value in sorted(_GENSER_ALIASES.items())
            },
            "turner-2025-medical-education": {
                label: {"method_id": value[0], "condition": value[1]}
                for label, value in sorted(_TURNER_ALIASES.items())
            },
        },
        "total_method_rows": len(rows),
        "eligible_count": len(cases),
        "eligible_by_source": dict(sorted(counts.items())),
        "distinct_methods": methods,
        "distinct_method_count": len(methods),
        "eligible_case_fingerprint": canonical_hash(cases),
        "qualification_gate": {"passed": all(checks.values()), "checks": checks},
        "evaluation_authorized": all(checks.values()),
        "claim_boundary": (
            "Qualification inspects source content and licenses only. It opens no model output and "
            "cannot establish external capability."
        ),
    }
    report["fingerprint"] = canonical_hash(report)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != report["fingerprint"]:
            raise RuntimeError("E4 source qualification changed")
        write_json(public_path, existing)
        return existing
    write_json(output_path, report)
    write_json(public_path, report)
    return report


def prepare_selective_external_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("selective_external"))
    root = _root(config)
    lock_path = root / "contract.json"
    public_path = (
        config.root
        / "reports"
        / "evolve"
        / "selector-sufficiency-external-v1-contract.json"
    )
    qualification_path = root / "source-qualification.json"
    qualification = prepare_selective_external_source_qualification(config)
    if not qualification.get("evaluation_authorized"):
        raise RuntimeError("E4 source qualification did not authorize evaluation")

    h16_path = config.root / "reports" / "evolve" / "selector-sufficiency-v1-confirmation.json"
    h16 = json.loads(h16_path.read_text(encoding="utf-8"))
    if not h16.get("selector_sufficiency_guard_confirmed"):
        raise RuntimeError("E4 requires confirmed H16 selector sufficiency guard")
    selection_path = config.path_for("artifact_dir") / "selector-sufficiency-v1" / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if float(selection["selected_threshold"]) != float(h16["selected_threshold"]):
        raise RuntimeError("E4 H16 threshold mismatch")
    bank_path = Path(str(selection["training_bank"]["path"]))
    if sha256_file(bank_path) != str(selection["training_bank"]["sha256"]):
        raise RuntimeError("E4 H16 training bank changed")

    h14_contract_path = config.root / "reports" / "evolve" / "selector-head-v1-contract.json"
    h14_contract = json.loads(h14_contract_path.read_text(encoding="utf-8"))
    observed = set(str(value) for value in h14_contract["selector_head"]["observed_methods"])
    rows, _ = _source_rows(config)
    cases = [
        _case_from_source_row(str(row["source_id"]), row)
        for row in rows
        if row.get("mapped_method_id") is not None
    ]
    if any(str(case["gold_method_id"]) not in observed for case in cases):
        raise RuntimeError("E4 contains a method outside frozen H14 head coverage")

    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": "E4 fresh two-source external selective-selector safety evaluation",
        "method_version": int(settings["method_version"]),
        "research_question": (
            "Does the fully frozen H16 hidden-geometry guard prevent the H14 selector from harming "
            "the original 4B menu-free method accuracy on new source-authored statistical cases?"
        ),
        "source_qualification_fingerprint": qualification["fingerprint"],
        "source_qualification_sha256": sha256_file(qualification_path),
        "source_qualified_not_blind": True,
        "h16_result_fingerprint": h16["result_fingerprint"],
        "h16_report_sha256": sha256_file(h16_path),
        "h16_threshold": float(h16["selected_threshold"]),
        "h16_training_bank_path": str(bank_path),
        "h16_training_bank_sha256": sha256_file(bank_path),
        "h14_contract_fingerprint": h14_contract["fingerprint"],
        "h14_contract_sha256": sha256_file(h14_contract_path),
        "parent": h14_contract["parent"],
        "frozen_h14_head": h14_contract["selector_head"],
        "case_count": len(cases),
        "case_fingerprint": canonical_hash(cases),
        "prompt_contract": (
            "exact H14 menu-free methods+columns extraction system prompt in both control and "
            "candidate. The candidate reuses the control prediction on guard fallback and replaces "
            "only the method with frozen H14 head argmax on guard acceptance."
        ),
        "arms": {
            "menu-free-control": "unchanged v0.3 parent H14 menu-free generation",
            "raw-h14-diagnostic": "frozen H14 head on every case; secondary diagnostic only",
            "selective-h16-candidate": (
                "support >= frozen 0.75 threshold uses H14 head; otherwise reuse the same control "
                "method prediction"
            ),
        },
        "gates": settings["evaluation_gates"],
        "decision_rule": (
            "Pass only if candidate reaches its absolute accuracy floor, never regresses versus "
            "control overall or within either source, has zero control-only paired losses, and "
            "does not reduce valid-output rate."
        ),
        "implementation_sha256": _implementation_manifest(),
        "adaptation_policy": (
            "none after lock; no source alias, case text, support threshold, H14 head, prompt, or "
            "gate may change after the first model output"
        ),
        "claim_boundary": (
            "A pass supports only narrow external safety/non-degradation of the frozen H16 "
            "selective interface on these 13 source-authored cases. It does not establish broad "
            "statistical competence, champion replacement, or release readiness."
        ),
        "model_evaluation_started": False,
        "historical_e3_reused": False,
    }
    contract["fingerprint"] = canonical_hash(contract)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("E4 contract is immutable")
        write_json(public_path, existing)
        return existing
    write_json(lock_path, contract)
    write_json(public_path, contract)
    return contract


def prepare_selective_external_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_selective_external_contract(config)
    qualification = prepare_selective_external_source_qualification(config)
    rows, receipts = _source_rows(config)
    cases = [
        _case_from_source_row(str(row["source_id"]), row)
        for row in rows
        if row.get("mapped_method_id") is not None
    ]
    if canonical_hash(cases) != contract["case_fingerprint"]:
        raise RuntimeError("E4 cases changed after contract freeze")
    case_path = _data_root(config) / "cases.jsonl"
    write_jsonl(case_path, cases)
    source_counts = Counter(str(case["source_id"]) for case in cases)
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "contract_fingerprint": contract["fingerprint"],
        "source_qualification_fingerprint": qualification["fingerprint"],
        "source_snapshot_sha256": {
            source_id: receipt["snapshot_sha256"] for source_id, receipt in sorted(receipts.items())
        },
        "cases_sha256": sha256_file(case_path),
        "case_fingerprint": canonical_hash(cases),
        "case_count": len(cases),
        "source_counts": dict(sorted(source_counts.items())),
        "distinct_methods": sorted({str(case["gold_method_id"]) for case in cases}),
        "evaluation_authorized": len(cases) == int(contract["case_count"]),
        "model_output_opened": False,
    }
    report["data_fingerprint"] = canonical_hash(report)
    write_json(_root(config) / "data.json", report)
    public_path = (
        config.root / "reports" / "evolve" / "selector-sufficiency-external-v1-data.json"
    )
    write_json(public_path, report)
    return report


def _raw_head_and_selective(
    agent: StatsAgent,
    cases: list[dict[str, Any]],
    control_details: list[dict[str, Any]],
    *,
    head: dict[str, Any],
    bank: dict[str, Any],
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    control = {str(row["case_id"]): row for row in control_details}
    raw_details: list[dict[str, Any]] = []
    selective_details: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        vector = _hidden_vector(agent, _runtime_case(str(case["question"])))
        predicted, top3, margin = _rank_methods(head, vector)
        support = float(_support_scores(bank, vector)[0])
        gold = str(case["gold_method_id"])
        raw_details.append(
            {
                "case_id": case_id,
                "source_id": case["source_id"],
                "eligible": True,
                "gold_method_id": gold,
                "predicted_method_id": predicted,
                "valid_output": True,
                "correct": predicted == gold,
                "top3_methods": top3,
                "score_margin": margin,
                "support_score": support,
            }
        )
        accepted = support >= threshold
        fallback = control[case_id]
        selective_prediction = predicted if accepted else fallback["predicted_method_id"]
        valid = True if accepted else bool(fallback["valid_output"])
        selective_details.append(
            {
                "case_id": case_id,
                "source_id": case["source_id"],
                "eligible": True,
                "gold_method_id": gold,
                "predicted_method_id": selective_prediction,
                "valid_output": valid,
                "correct": selective_prediction == gold,
                "support_score": support,
                "selector_accepted": accepted,
                "prediction_source": "frozen-h14" if accepted else "menu-free-fallback",
            }
        )
    return (
        {"metrics": _metrics(raw_details), "details": raw_details},
        {"metrics": _metrics(selective_details), "details": selective_details},
    )


def _source_paired(
    control: list[dict[str, Any]], candidate: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    case_source = {str(case["case_id"]): str(case["source_id"]) for case in cases}
    result: dict[str, Any] = {}
    for source_id in sorted(set(case_source.values())):
        ids = {case_id for case_id, source in case_source.items() if source == source_id}
        control_rows = [row for row in control if str(row["case_id"]) in ids]
        candidate_rows = [row for row in candidate if str(row["case_id"]) in ids]
        result[source_id] = _paired_summary(control_rows, candidate_rows)
    return result


def _external_gate(
    control: dict[str, Any],
    candidate: dict[str, Any],
    paired: dict[str, Any],
    source_paired: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    gain = 100.0 * (
        float(candidate["eligible_accuracy"]) - float(control["eligible_accuracy"])
    )
    validity_gain = 100.0 * (
        float(candidate["valid_output_rate"]) - float(control["valid_output_rate"])
    )
    worst_source_net = min(int(value["net_improvements"]) for value in source_paired.values())
    checks = {
        "candidate_absolute_accuracy": float(candidate["eligible_accuracy"])
        >= float(gates["minimum_candidate_accuracy"]),
        "candidate_noninferior_to_control": gain
        >= float(gates["minimum_gain_over_control_points"]),
        "zero_control_only_losses": int(paired["control_only"])
        <= int(gates["maximum_control_only_losses"]),
        "all_sources_nonnegative": worst_source_net
        >= int(gates["minimum_worst_source_net_improvement"]),
        "validity_noninferior": validity_gain
        >= -float(gates["maximum_validity_regression_points"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "effect_points": {
            "candidate_accuracy_vs_control": gain,
            "valid_output_rate": validity_gain,
        },
        "worst_source_net_improvement": worst_source_net,
    }


def run_selective_external_evaluation(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_selective_external_contract(config)
    data = prepare_selective_external_data(config)
    if not data.get("evaluation_authorized"):
        raise RuntimeError("E4 data did not authorize evaluation")
    case_path = _data_root(config) / "cases.jsonl"
    if sha256_file(case_path) != data["cases_sha256"]:
        raise RuntimeError("E4 frozen case file changed")
    cases = list(read_jsonl(case_path))

    h14_contract = json.loads(
        (config.root / "reports" / "evolve" / "selector-head-v1-contract.json").read_text(
            encoding="utf-8"
        )
    )
    head = _load_head(h14_contract)
    bank_path = Path(str(contract["h16_training_bank_path"]))
    if sha256_file(bank_path) != contract["h16_training_bank_sha256"]:
        raise RuntimeError("E4 H16 training bank changed before evaluation")
    bank = _load_training_bank(bank_path)
    evaluation_fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "data": data["data_fingerprint"],
            "evaluator_version": _EVALUATOR_VERSION,
            "implementation": _implementation_manifest(),
        }
    )
    report_path = _root(config) / "report.json"
    public_path = config.root / "reports" / "evolve" / "selector-sufficiency-external-v1.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != evaluation_fingerprint or not existing.get("complete"):
            raise RuntimeError("E4 report changed")
        public = dict(existing)
        public.pop("private_details", None)
        write_json(public_path, public)
        return public

    agent = StatsAgent(config, adapter_path=str(contract["parent"]["adapter_path"]))
    agent.router.set_route("adapter")
    try:
        control = _evaluate_control(
            agent,
            cases,
            progress_root=_root(config) / "progress",
            evaluation_fingerprint=evaluation_fingerprint,
        )
        raw_h14, candidate = _raw_head_and_selective(
            agent,
            cases,
            control["details"],
            head=head,
            bank=bank,
            threshold=float(contract["h16_threshold"]),
        )
    finally:
        del agent
        gc.collect()
        mx.clear_cache()

    paired = _paired_summary(control["details"], candidate["details"])
    source_paired = _source_paired(control["details"], candidate["details"], cases)
    gate = _external_gate(
        control["metrics"],
        candidate["metrics"],
        paired,
        source_paired,
        dict(contract["gates"]),
    )
    selector_accept_count = sum(bool(row["selector_accepted"]) for row in candidate["details"])
    result: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": evaluation_fingerprint,
        "method": "E4 fresh two-source frozen H16 selective-selector safety evaluation",
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["data_fingerprint"],
        "source_qualification_fingerprint": contract["source_qualification_fingerprint"],
        "fresh_external_evidence": True,
        "source_qualified_not_blind": True,
        "same_parent_weights": True,
        "same_h14_prompt": True,
        "h16_threshold_frozen_before_model_output": float(contract["h16_threshold"]),
        "scores": {
            "menu-free-control": control["metrics"],
            "raw-h14-diagnostic": raw_h14["metrics"],
            "selective-h16-candidate": candidate["metrics"],
        },
        "selector_accept_count": selector_accept_count,
        "selector_accept_rate": selector_accept_count / len(cases),
        "paired_control_vs_selective": paired,
        "source_paired_control_vs_selective": source_paired,
        "external_gate": gate,
        "selective_external_safety_supported": bool(gate["passed"]),
        "champion_changed": False,
        "release_authorized": False,
        "historical_e3_reused": False,
        "next_step": (
            "freeze-operational-selective-runtime-and-seek-broader-independent-external-evidence"
            if gate["passed"]
            else "preserve-e4-negative-and-do-not-promote-h16"
        ),
        "claim_boundary": contract["claim_boundary"],
        "private_details": {
            "menu-free-control": control["details"],
            "raw-h14-diagnostic": raw_h14["details"],
            "selective-h16-candidate": candidate["details"],
        },
    }
    result["result_fingerprint"] = canonical_hash(
        {key: value for key, value in result.items() if key != "private_details"}
    )
    write_json(report_path, result)
    public = dict(result)
    public.pop("private_details")
    write_json(public_path, public)
    return public

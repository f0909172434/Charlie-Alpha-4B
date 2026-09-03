from __future__ import annotations

import csv
import gc
import inspect
import io
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import mlx.core as mx

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
from .stats_agent import StatsAgent
from .stats_catalog import PROCEDURES
from .stats_external_catalog import (
    _ALIASES,
    _canonicalize_method_label,
    _evaluate_arm,
    _evaluation_implementation_manifest,
    _metrics,
    _paired_summary,
    _request_bytes,
)
from .stats_family_router import _expert_context

_EVALUATOR_VERSION = 1
_MATERIALIZER_VERSION = 1


# Haq & Nazir (2016), Tables 1 and 2. The PMC article publishes these tables
# as immutable images rather than HTML cells. During source qualification,
# every non-NA cell was transcribed in source order from the two frozen table
# images. The materializer verifies the image hashes before using this complete
# transcription; no row is selected based on project taxonomy coverage.
_HAQ_ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("T1", "Dichotomous", "Dichotomous", "Chi-square test"),
    ("T1", "Dichotomous", "Polychotomous", "Chi-square test"),
    ("T1", "Dichotomous", "Ordinal score", "Mann-Whitney U-test"),
    ("T1", "Dichotomous", "Scale", "Unpaired t-test"),
    ("T1", "Polychotomous", "Dichotomous", "Chi-square test"),
    ("T1", "Polychotomous", "Polychotomous", "Chi-square test"),
    ("T1", "Polychotomous", "Ordinal score", "Kruskal-Wallis test"),
    ("T1", "Polychotomous", "Scale", "ANOVA"),
    ("T1", "Ordinal score", "Dichotomous", "Mann-Whitney U-test"),
    ("T1", "Ordinal score", "Polychotomous", "Kruskal-Wallis test"),
    ("T1", "Ordinal score", "Ordinal score", "Spearman correlation"),
    ("T1", "Ordinal score", "Scale", "Spearman correlation"),
    ("T1", "Scale", "Dichotomous", "Unpaired t-test"),
    ("T1", "Scale", "Polychotomous", "ANOVA"),
    ("T1", "Scale", "Ordinal score", "Spearman correlation"),
    ("T1", "Scale", "Scale", "Pearson correlation"),
    ("T2", "Dichotomous", "Two repeated measures", "McNemar Chi-square test"),
    ("T2", "Dichotomous", ">2 repeated measures", "Cochran Q-test"),
    ("T2", "Ordinal", "Two repeated measures", "Wilcoxon signed rank test"),
    ("T2", "Ordinal", ">2 repeated measures", "Friedman test"),
    ("T2", "Scale", "Two repeated measures", "Paired t-test"),
    ("T2", "Scale", ">2 repeated measures", "Repeated-measures ANOVA"),
)


_CONYSO_QUALIFICATION_LABELS: tuple[str, ...] = (
    "One-sample t-test",
    "Two-sample t-test",
    "Paired t-test",
    "ANOVA",
    "Chi-square test",
    "Correlation / regression",
    "One-proportion z-test",
    "Two-proportion z-test",
)


_CRUCIBLE_QUALIFICATION_LABELS: tuple[str, ...] = (
    "Independent t-test",
    "Welch's t-test",
    "Paired t-test",
    "One-way ANOVA",
    "Mann-Whitney U",
    "Wilcoxon signed-rank",
    "Kruskal-Wallis",
)


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "external-catalog-interface-v2"


def _data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "external-catalog-interface-v2"


def _qualified_count(labels: list[str] | tuple[str, ...]) -> int:
    return sum(_canonicalize_method_label(label) is not None for label in labels)


def _qualification_summary() -> dict[str, Any]:
    sources = [
        {
            "source_id": "haq-2016",
            "case_count": len(_HAQ_ROWS),
            "eligible_count": _qualified_count([row[3] for row in _HAQ_ROWS]),
            "selection_rule": "all non-NA cells from published Tables 1 and 2 in source order",
        },
        {
            "source_id": "conyso-2026",
            "case_count": len(_CONYSO_QUALIFICATION_LABELS),
            "eligible_count": _qualified_count(_CONYSO_QUALIFICATION_LABELS),
            "selection_rule": "all data rows in the published selector CSV in source order",
        },
        {
            "source_id": "crucible-bench",
            "case_count": len(_CRUCIBLE_QUALIFICATION_LABELS),
            "eligible_count": _qualified_count(_CRUCIBLE_QUALIFICATION_LABELS),
            "selection_rule": (
                "all rows in README Parametric Tests and Non-Parametric Tests tables "
                "in source order"
            ),
        },
    ]
    for source in sources:
        source["coverage_fraction"] = source["eligible_count"] / source["case_count"]
    case_count = sum(int(source["case_count"]) for source in sources)
    eligible_count = sum(int(source["eligible_count"]) for source in sources)
    coverage_fraction = eligible_count / case_count
    checks = {
        "minimum_eligible_cases": eligible_count >= 12,
        "minimum_coverage_fraction": coverage_fraction >= 0.40,
        "minimum_source_count": len(sources) >= 3,
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "E2-v2 direct-tabular multi-source qualification",
        "source_count": len(sources),
        "sources": sources,
        "case_count": case_count,
        "eligible_count": eligible_count,
        "coverage_fraction": coverage_fraction,
        "qualification_gate": {"passed": all(checks.values()), "checks": checks},
        "model_evaluation_started": False,
        "claim_boundary": (
            "Qualification only: this establishes enough independent external decision-table "
            "coverage to preregister E2-v2. It is not model evidence."
        ),
    }
    result["fingerprint"] = canonical_hash(result)
    return result


def _catalog_manifest() -> dict[str, Any]:
    catalog = [(procedure.method_id, procedure.name) for procedure in PROCEDURES]
    aliases = {key: list(value) for key, value in sorted(_ALIASES.items())}
    return {
        "procedure_count": len(catalog),
        "catalog_sha256": canonical_hash(catalog),
        "alias_sha256": canonical_hash(aliases),
        "alias_policy": "unchanged E1 exact alias table; no post-source additions",
    }


def _source_manifest(settings: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "source_id": "haq-2016",
            **dict(settings["sources"]["haq_2016"]),
            "expected_case_count": len(_HAQ_ROWS),
            "expected_eligible_count": _qualified_count([row[3] for row in _HAQ_ROWS]),
            "extraction_rule": (
                "Retain every non-NA cell from Table 1 then Table 2 in row-major source order. "
                "The complete table transcription was frozen before any E2 model output."
            ),
        },
        {
            "source_id": "conyso-2026",
            **dict(settings["sources"]["conyso_2026"]),
            "expected_case_count": len(_CONYSO_QUALIFICATION_LABELS),
            "expected_eligible_count": _qualified_count(_CONYSO_QUALIFICATION_LABELS),
            "extraction_rule": "Retain every CSV data row in source order.",
        },
        {
            "source_id": "crucible-bench",
            **dict(settings["sources"]["crucible_bench"]),
            "expected_case_count": len(_CRUCIBLE_QUALIFICATION_LABELS),
            "expected_eligible_count": _qualified_count(_CRUCIBLE_QUALIFICATION_LABELS),
            "extraction_rule": (
                "Retain every Markdown row under the README Parametric Tests and "
                "Non-Parametric Tests tables, in source order."
            ),
        },
    ]


def _v2_implementation_manifest() -> dict[str, str]:
    functions = {
        "qualification_summary": _qualification_summary,
        "haq_cases": _haq_cases,
        "parse_conyso_csv": _parse_conyso_csv,
        "parse_crucible_readme": _parse_crucible_readme,
        "build_case": _build_case,
        "external_v2_gate": _external_v2_gate,
    }
    manifest = {
        name: sha256_text(inspect.getsource(function)) for name, function in functions.items()
    }
    manifest["e1_evaluator_manifest"] = canonical_hash(_evaluation_implementation_manifest())
    return manifest


def prepare_external_catalog_v2_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("external_catalog_interface_v2"))
    root = _root(config)
    lock_path = root / "contract.json"
    public_path = config.root / "reports" / "evolve" / "external-catalog-interface-v2-contract.json"
    qualification_path = (
        config.root
        / "reports"
        / "evolve"
        / "external-catalog-interface-v2-source-qualification.json"
    )

    h12_path = config.root / "reports" / "evolve" / "catalog-interface-replication-v1.json"
    if not h12_path.exists():
        raise RuntimeError("E2-v2 requires the completed H12 replication")
    h12 = json.loads(h12_path.read_text(encoding="utf-8"))
    if not h12.get("synthetic_catalog_interface_replicated") or not h12["replication_gate"][
        "passed"
    ]:
        raise RuntimeError("H12 did not authorize independent external interface evidence")

    e1_path = config.root / "reports" / "evolve" / "external-catalog-interface-v1.json"
    if not e1_path.exists():
        raise RuntimeError("E2-v2 requires the terminal E1 source-unavailable closure")
    e1 = json.loads(e1_path.read_text(encoding="utf-8"))
    if not e1.get("terminal") or e1.get("model_evaluation_started"):
        raise RuntimeError("E1 is not a clean source-unavailable closure")

    qualification = _qualification_summary()
    if not qualification["qualification_gate"]["passed"]:
        raise RuntimeError("E2-v2 source pool does not pass frozen qualification")
    write_json(qualification_path, qualification)

    _, adapter_paths = _expert_context(config)
    parent = adapter_paths["parent"]
    parent_sha = sha256_file(parent / "adapters.safetensors")
    fields: dict[str, Any] = {
        "schema_version": 1,
        "method": "E2-v2 multi-source external fixed-catalog interface evaluation",
        "method_version": int(settings["method_version"]),
        "causal_question": (
            "Across a frozen pool of three independently authored external statistical-test "
            "decision tables, does adding H7's unchanged fixed 28-method catalog improve the "
            "unchanged v0.3 parent over the same menu-free canonical-ID prompt?"
        ),
        "supersedes": {
            "e1_result_fingerprint": e1["result_fingerprint"],
            "reason": (
                "E1 was source-unavailable before model evaluation; E2-v1 single-source "
                "qualification found no source satisfying both 12 eligible cases and 40% coverage."
            ),
        },
        "h12_result_fingerprint": h12["result_fingerprint"],
        "h12_report_sha256": sha256_file(h12_path),
        "parent": {
            "name": "v0.3.0-parent",
            "adapter_path": str(parent),
            "adapter_sha256": parent_sha,
        },
        "catalog": _catalog_manifest(),
        "source_qualification_fingerprint": qualification["fingerprint"],
        "sources": _source_manifest(settings),
        "source_reconnaissance_state": {
            "source_content_opened_for_qualification_before_contract": True,
            "model_outputs_opened": False,
            "note": (
                "E2-v2 is source-qualified rather than source-blinded. All selection, extraction, "
                "alias, prompt, and evaluation rules are frozen before the first model answer."
            ),
        },
        "case_pool_policy": {
            "source_type": "direct tabular condition-to-test mappings only",
            "minimum_sources": 3,
            "minimum_eligible_cases": int(settings["eligibility"]["minimum_eligible_cases"]),
            "minimum_coverage_fraction": float(
                settings["eligibility"]["minimum_coverage_fraction"]
            ),
            "unmapped_policy": (
                "Retain every unmapped source row in total coverage accounting; exclude it only "
                "from the primary paired correctness denominator."
            ),
        },
        "generation_max_tokens": int(settings["generation_max_tokens"]),
        "gates": dict(settings["gates"]),
        "arms": {
            "menu-free-control": (
                "Unchanged v0.3 parent receives the frozen external case and canonical-ID JSON "
                "contract without a method catalog"
            ),
            "flat-catalog": (
                "Same parent, case, JSON contract, decoding, and case order plus H7's unchanged "
                "fixed 28-method ID + display-name catalog"
            ),
        },
        "adaptation_policy": "none after the first E2-v2 model answer is observed",
        "stopping_rule": (
            "Materialize all 37 frozen source rows once. If the 12-case/40%-coverage gate passes, "
            "run both arms exactly once with resumable progress under one evaluator fingerprint."
        ),
        "decision_rule": (
            "Pass only if pooled coverage, flat-catalog absolute accuracy, paired gain, net "
            "improvements, output validity, and pre-registered cross-source robustness gates pass."
        ),
        "claim_boundary": (
            "A pass supports an independent external decision-source context/interface elicitation "
            "claim only. It does not establish free-form clinical competence, change weights, "
            "replace the champion, authorize release, or reopen historical P-Bench/StatQA."
        ),
        "evaluation_implementation_sha256": _v2_implementation_manifest(),
        "model_evaluation_started": False,
    }
    fields["fingerprint"] = canonical_hash(fields)
    root.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fields["fingerprint"]:
            raise RuntimeError("E2-v2 contract is immutable")
        write_json(public_path, existing)
        return existing
    write_json(lock_path, fields)
    write_json(public_path, fields)
    return fields


def _build_case(
    *,
    source_id: str,
    source_order: int,
    vignette: str,
    gold_raw: str,
    locator: dict[str, Any],
) -> dict[str, Any]:
    gold_method = _canonicalize_method_label(gold_raw)
    return {
        "case_id": f"{source_id}-{source_order:03d}",
        "source_id": source_id,
        "source_order": source_order,
        "vignette": vignette,
        "gold_raw": gold_raw,
        "gold_method_id": gold_method,
        "eligible": gold_method is not None,
        "source_locator": locator,
    }


def _haq_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    table_counts: Counter[str] = Counter()
    for source_order, (table, variable_1, variable_2, gold_raw) in enumerate(_HAQ_ROWS, start=1):
        table_counts[table] += 1
        if table == "T1":
            vignette = (
                "Published decision-table conditions: between-subjects design with no repeated "
                f"measures; Variable 1 is {variable_1}; Variable 2 is {variable_2}. "
                "Select the single statistical test indicated by these conditions."
            )
        else:
            vignette = (
                "Published decision-table conditions: within-subjects design with repeated "
                f"measures; variable of interest is {variable_1}; schedule is {variable_2}. "
                "Select the single statistical test indicated by these conditions."
            )
        rows.append(
            _build_case(
                source_id="haq-2016",
                source_order=source_order,
                vignette=vignette,
                gold_raw=gold_raw,
                locator={"table": table, "non_na_cell_index": table_counts[table]},
            )
        )
    return rows


def _parse_conyso_csv(payload: bytes) -> list[dict[str, Any]]:
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != ["Your question", "Data", "Test"]:
        raise RuntimeError(f"Conyso CSV schema changed: {reader.fieldnames}")
    rows: list[dict[str, Any]] = []
    for source_order, row in enumerate(reader, start=1):
        question = str(row["Your question"] or "").strip()
        data = str(row["Data"] or "").strip()
        gold_raw = str(row["Test"] or "").strip()
        if not question or not data or not gold_raw:
            raise RuntimeError("Conyso CSV contains an incomplete selector row")
        rows.append(
            _build_case(
                source_id="conyso-2026",
                source_order=source_order,
                vignette=f"{question}\nPublished data condition: {data}",
                gold_raw=gold_raw,
                locator={"csv_row": source_order + 1},
            )
        )
    return rows


def _parse_markdown_table_rows(block: str) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.count("|") < 4:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 3 or cells[0] == "Test" or set(cells[0]) <= {"-", ":"}:
            continue
        rows.append((cells[0], cells[1], cells[2]))
    return rows


def _parse_crucible_readme(payload: bytes) -> list[dict[str, Any]]:
    text = payload.decode("utf-8")
    parametric_match = re.search(
        r"### Parametric Tests\s+(.*?)(?=\n### Non-Parametric Tests)", text, flags=re.S
    )
    nonparametric_match = re.search(
        r"### Non-Parametric Tests\s+(.*?)(?=\n### Effect Sizes)", text, flags=re.S
    )
    if parametric_match is None or nonparametric_match is None:
        raise RuntimeError("CrucibleBench statistical-test tables changed")
    table_rows = _parse_markdown_table_rows(parametric_match.group(1)) + _parse_markdown_table_rows(
        nonparametric_match.group(1)
    )
    rows: list[dict[str, Any]] = []
    for source_order, (gold_raw, function, use_case) in enumerate(table_rows, start=1):
        rows.append(
            _build_case(
                source_id="crucible-bench",
                source_order=source_order,
                vignette=(
                    f"Published statistical-test use case: {use_case}. "
                    "Select the single statistical test assigned to this use case."
                ),
                gold_raw=gold_raw,
                locator={"readme_table_row": source_order, "function": function},
            )
        )
    return rows


def _download_verified(url: str, expected_sha256: str) -> bytes:
    payload = _request_bytes(url)
    observed = sha256_bytes(payload)
    if observed != expected_sha256:
        raise RuntimeError(
            f"Frozen E2-v2 source changed for {url}: expected {expected_sha256}, got {observed}"
        )
    return payload


def _source_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for source_id in sorted({str(row["source_id"]) for row in rows}):
        source_rows = [row for row in rows if row["source_id"] == source_id]
        eligible_count = sum(bool(row["eligible"]) for row in source_rows)
        counts[source_id] = {
            "case_count": len(source_rows),
            "eligible_count": eligible_count,
            "coverage_fraction": eligible_count / len(source_rows),
            "eligible_method_counts": dict(
                sorted(
                    Counter(
                        str(row["gold_method_id"])
                        for row in source_rows
                        if row["gold_method_id"] is not None
                    ).items()
                )
            ),
        }
    return counts


def prepare_external_catalog_v2_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_external_catalog_v2_contract(config)
    root = _root(config)
    private_path = _data_root(config) / "cases.jsonl"
    manifest_path = root / "data.json"
    public_path = config.root / "reports" / "evolve" / "external-catalog-interface-v2-data.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not private_path.exists() or existing.get("cases_sha256") != sha256_file(private_path):
            raise RuntimeError("E2-v2 materialized cases changed or disappeared")
        write_json(public_path, existing)
        return existing

    source_settings = {source["source_id"]: source for source in contract["sources"]}

    haq = source_settings["haq-2016"]
    haq_t1 = _download_verified(str(haq["table_1_url"]), str(haq["table_1_sha256"]))
    haq_t2 = _download_verified(str(haq["table_2_url"]), str(haq["table_2_sha256"]))
    haq_license = _request_bytes(str(haq["license_source_url"]))
    if b"CC BY-NC-SA" not in haq_license:
        raise RuntimeError("Haq official PMC license evidence changed")

    conyso = source_settings["conyso-2026"]
    conyso_csv = _download_verified(str(conyso["csv_url"]), str(conyso["csv_sha256"]))
    conyso_page = _request_bytes(str(conyso["license_page_url"]))
    if b"CC BY 4.0" not in conyso_page:
        raise RuntimeError("Conyso CC BY 4.0 evidence changed")

    crucible = source_settings["crucible-bench"]
    crucible_readme = _download_verified(
        str(crucible["readme_url"]), str(crucible["readme_sha256"])
    )
    crucible_license = _download_verified(
        str(crucible["license_url"]), str(crucible["license_sha256"])
    )
    if not crucible_license.startswith(b"MIT License"):
        raise RuntimeError("CrucibleBench MIT license evidence changed")

    rows = _haq_cases() + _parse_conyso_csv(conyso_csv) + _parse_crucible_readme(crucible_readme)
    expected_sources = {source["source_id"]: source for source in contract["sources"]}
    counts = _source_counts(rows)
    if set(counts) != set(expected_sources):
        raise RuntimeError("E2-v2 source set changed during materialization")
    for source_id, source in expected_sources.items():
        observed = counts[source_id]
        if int(observed["case_count"]) != int(source["expected_case_count"]):
            raise RuntimeError(f"E2-v2 {source_id} case count changed")
        if int(observed["eligible_count"]) != int(source["expected_eligible_count"]):
            raise RuntimeError(f"E2-v2 {source_id} exact-alias coverage changed")

    case_count = len(rows)
    eligible_count = sum(bool(row["eligible"]) for row in rows)
    coverage_fraction = eligible_count / case_count
    policy = dict(contract["case_pool_policy"])
    checks = {
        "all_frozen_sources_materialized": len(counts) == int(policy["minimum_sources"]),
        "expected_total_case_count": case_count == 37,
        "minimum_eligible_cases": eligible_count >= int(policy["minimum_eligible_cases"]),
        "minimum_coverage_fraction": coverage_fraction
        >= float(policy["minimum_coverage_fraction"]),
    }
    if len({str(row["case_id"]) for row in rows}) != case_count:
        raise RuntimeError("E2-v2 materialized duplicate case IDs")
    write_jsonl(private_path, rows)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "E2-v2 frozen external direct-tabular materialization",
        "contract_fingerprint": contract["fingerprint"],
        "source_receipts": {
            "haq-2016": {
                "table_1_sha256": sha256_bytes(haq_t1),
                "table_2_sha256": sha256_bytes(haq_t2),
                "license_evidence_contains": "CC BY-NC-SA",
            },
            "conyso-2026": {
                "csv_sha256": sha256_bytes(conyso_csv),
                "license_evidence_contains": "CC BY 4.0",
            },
            "crucible-bench": {
                "revision": crucible["revision"],
                "readme_sha256": sha256_bytes(crucible_readme),
                "license_sha256": sha256_bytes(crucible_license),
            },
        },
        "case_count": case_count,
        "eligible_count": eligible_count,
        "out_of_catalog_count": case_count - eligible_count,
        "coverage_fraction": coverage_fraction,
        "source_counts": counts,
        "coverage_gate": {"passed": all(checks.values()), "checks": checks},
        "evaluation_authorized": all(checks.values()),
        "model_evaluation_started": False,
        "materializer_version": _MATERIALIZER_VERSION,
        "materializer_sha256": _v2_implementation_manifest(),
        "cases_sha256": sha256_file(private_path),
        "data_fingerprint": canonical_hash(rows),
        "unmapped_gold_labels_sha256": canonical_hash(
            sorted(str(row["gold_raw"]) for row in rows if not row["eligible"])
        ),
        "next_step": (
            "run-one-shot-e2-v2-menu-free-vs-flat-catalog-evaluation"
            if all(checks.values())
            else "close-e2-v2-for-insufficient-pooled-coverage"
        ),
    }
    manifest["fingerprint"] = canonical_hash(manifest)
    write_json(manifest_path, manifest)
    write_json(public_path, manifest)
    return manifest


def _external_v2_gate(
    *,
    data: dict[str, Any],
    control: dict[str, Any],
    candidate: dict[str, Any],
    paired: dict[str, Any],
    source_paired: dict[str, dict[str, Any]],
    gates: dict[str, Any],
) -> dict[str, Any]:
    tolerance = 1e-9
    gain_points = 100 * (
        float(candidate["eligible_accuracy"]) - float(control["eligible_accuracy"])
    )
    validity_delta = 100 * (
        float(candidate["valid_output_rate"]) - float(control["valid_output_rate"])
    )
    source_net = {
        source_id: int(summary["net_improvements"])
        for source_id, summary in sorted(source_paired.items())
    }
    nonnegative_sources = sum(value >= 0 for value in source_net.values())
    worst_source_net = min(source_net.values()) if source_net else -10**9
    checks = {
        "source_coverage": bool(data["evaluation_authorized"]),
        "flat_catalog_absolute_accuracy": float(candidate["eligible_accuracy"])
        >= float(gates["minimum_flat_catalog_accuracy"]) - tolerance,
        "method_gain_over_control": gain_points
        >= float(gates["minimum_method_gain_points"]) - tolerance,
        "net_paired_improvements": int(paired["net_improvements"])
        >= int(gates["minimum_net_improvements"]),
        "validity_noninferior": validity_delta
        >= -float(gates["maximum_validity_regression_points"]) - tolerance,
        "cross_source_direction": nonnegative_sources
        >= int(gates["minimum_nonnegative_sources"]),
        "worst_source_regression": worst_source_net
        >= int(gates["minimum_worst_source_net_improvement"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "effect_points": {
            "eligible_method_accuracy": gain_points,
            "valid_output_rate": validity_delta,
        },
        "cross_source": {
            "net_improvements": source_net,
            "nonnegative_source_count": nonnegative_sources,
            "worst_source_net_improvement": worst_source_net,
        },
    }


def run_external_catalog_v2_evaluation(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_external_catalog_v2_contract(config)
    data = prepare_external_catalog_v2_data(config)
    if not data.get("evaluation_authorized"):
        raise RuntimeError("E2-v2 materialization did not authorize model evaluation")

    report_path = _root(config) / "report.json"
    public_path = config.root / "reports" / "evolve" / "external-catalog-interface-v2.json"
    cases = list(read_jsonl(_data_root(config) / "cases.jsonl"))
    _, adapter_paths = _expert_context(config)
    parent = adapter_paths["parent"]
    parent_sha = sha256_file(parent / "adapters.safetensors")
    if parent_sha != contract["parent"]["adapter_sha256"]:
        raise RuntimeError("E2-v2 parent adapter changed")
    evaluation_fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "data": data["data_fingerprint"],
            "parent": parent_sha,
            "evaluator_version": _EVALUATOR_VERSION,
            "implementation": _v2_implementation_manifest(),
        }
    )
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != evaluation_fingerprint or not existing.get("complete"):
            raise RuntimeError("E2-v2 report changed")
        public = dict(existing)
        public.pop("private_details", None)
        write_json(public_path, public)
        return public

    agent = StatsAgent(config, adapter_path=parent)
    agent.router.set_route("adapter")
    try:
        progress_root = _root(config) / "progress"
        max_tokens = int(contract["generation_max_tokens"])
        control = _evaluate_arm(
            agent,
            cases,
            name="menu-free-control",
            grounded=False,
            progress_root=progress_root,
            evaluation_fingerprint=evaluation_fingerprint,
            max_tokens=max_tokens,
        )
        candidate = _evaluate_arm(
            agent,
            cases,
            name="flat-catalog",
            grounded=True,
            progress_root=progress_root,
            evaluation_fingerprint=evaluation_fingerprint,
            max_tokens=max_tokens,
        )
    finally:
        del agent
        gc.collect()
        mx.clear_cache()

    paired = _paired_summary(control["details"], candidate["details"])
    source_reports: dict[str, dict[str, Any]] = {}
    source_paired: dict[str, dict[str, Any]] = {}
    for source_id in sorted(data["source_counts"]):
        control_rows = [row for row in control["details"] if row["case_id"].startswith(source_id)]
        candidate_rows = [
            row for row in candidate["details"] if row["case_id"].startswith(source_id)
        ]
        source_paired[source_id] = _paired_summary(control_rows, candidate_rows)
        source_reports[source_id] = {
            "menu-free-control": _metrics(control_rows),
            "flat-catalog": _metrics(candidate_rows),
            "paired": source_paired[source_id],
        }

    gate = _external_v2_gate(
        data=data,
        control=control["metrics"],
        candidate=candidate["metrics"],
        paired=paired,
        source_paired=source_paired,
        gates=dict(contract["gates"]),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": evaluation_fingerprint,
        "method": "E2-v2 multi-source external fixed-catalog interface evaluation",
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["data_fingerprint"],
        "same_parent_weights": True,
        "model_evaluation_started": True,
        "scores": {
            "menu-free-control": control["metrics"],
            "flat-catalog": candidate["metrics"],
        },
        "paired": paired,
        "source_scores": source_reports,
        "external_gate": gate,
        "independent_external_interface_supported": bool(gate["passed"]),
        "champion_unchanged": "v0.3.0-parent",
        "release_authorized": False,
        "historical_pbench_statqa_reopened": False,
        "next_step": (
            "preserve-e2-v2-positive-external-decision-source-evidence-without-weight-promotion"
            if gate["passed"]
            else "reject-e2-v2-external-decision-source-interface-claim"
        ),
        "claim_boundary": contract["claim_boundary"],
        "private_details": {
            "menu-free-control": control["details"],
            "flat-catalog": candidate["details"],
        },
    }
    report["result_fingerprint"] = canonical_hash(
        {key: value for key, value in report.items() if key != "private_details"}
    )
    write_json(report_path, report)
    public = dict(report)
    public.pop("private_details")
    write_json(public_path, public)
    return public

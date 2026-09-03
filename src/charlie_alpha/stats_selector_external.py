from __future__ import annotations

import gc
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mlx.core as mx

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_bytes, sha256_file, write_json, write_jsonl
from .stats_agent import StatsAgent
from .stats_external_catalog import _evaluate_arm, _metrics, _paired_summary, _request_bytes
from .stats_selector_head import _load_head
from .stats_selector_runtime import _hidden_vector, _rank_methods, _runtime_case

_EVALUATOR_VERSION = 1
_MATERIALIZER_VERSION = 1
_BIOC_URL = (
    "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/"
    "BioC_json/PMC12627256/unicode"
)
_PMCID = "PMC12627256"
_DOI = "10.7759/cureus.94949"

_SOURCE_LABELS: tuple[str, ...] = (
    "Student's t-test",
    "One-way ANOVA",
    "Mann-Whitney U test",
    "Kruskal-Wallis H test",
    "Fisher's exact test",
    "Chi-square test",
    "Log-rank test",
    "Paired t-test",
    "Repeated measures ANOVA",
    "Wilcoxon signed-rank test",
    "Friedman test",
    "McNemar's test",
    "Pearson's correlation",
    "Spearman's correlation",
    "Kappa agreement",
    "Linear regression",
    "Ordered logistic regression",
    "Binary logistic regression",
    "Multinomial logistic regression",
    "Poisson regression",
)

# Frozen source-specific mapping, defined before any E3 model output. Student's
# t-test is resolved to independent_t because source scenario 1 explicitly uses
# two randomly divided independent groups. None means the source test has no
# existing Charlie Alpha catalog equivalent. Poisson maps to the catalog but is
# later head-ineligible because H13 never observed poisson_glm during head fit.
_SOURCE_METHOD_MAP: dict[str, str | None] = {
    "Student's t-test": "independent_t",
    "One-way ANOVA": None,
    "Mann-Whitney U test": "mann_whitney",
    "Kruskal-Wallis H test": None,
    "Fisher's exact test": "fisher_exact",
    "Chi-square test": "chi_square",
    "Log-rank test": "logrank",
    "Paired t-test": "paired_t",
    "Repeated measures ANOVA": None,
    "Wilcoxon signed-rank test": "wilcoxon_signed_rank",
    "Friedman test": None,
    "McNemar's test": None,
    "Pearson's correlation": None,
    "Spearman's correlation": None,
    "Kappa agreement": None,
    "Linear regression": "ols",
    "Ordered logistic regression": None,
    "Binary logistic regression": "logistic_glm",
    "Multinomial logistic regression": None,
    "Poisson regression": "poisson_glm",
}


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "selector-external-v1"


def _data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "selector-external-v1"


def _implementation_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "stats_selector_external.py": sha256_file(Path(__file__)),
        "stats_external_catalog.py": sha256_file(root / "stats_external_catalog.py"),
    }


def _source_rows(payload: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    collections = json.loads(payload)
    if not isinstance(collections, list) or len(collections) != 1:
        raise RuntimeError("E3 BioC source must contain exactly one collection")
    collection = collections[0]
    documents = collection.get("documents", [])
    if not isinstance(documents, list) or len(documents) != 1:
        raise RuntimeError("E3 BioC source must contain exactly one document")
    document = documents[0]
    if document.get("infons", {}).get("license") != "CC BY":
        raise RuntimeError("E3 source is no longer marked CC BY")

    table_xml = None
    scenarios: dict[int, str] = {}
    for passage in document.get("passages", []):
        infons = passage.get("infons", {})
        if (
            infons.get("section_type") == "TABLE"
            and infons.get("id") == "TAB1"
            and infons.get("type") == "table"
        ):
            table_xml = infons.get("xml")
        if infons.get("section_type") == "APPENDIX" and infons.get("type") == "paragraph":
            text = str(passage.get("text", "")).replace("\xa0", " ").strip()
            match = re.match(r"^(\d+)\.\s*(.+)$", text, flags=re.DOTALL)
            if match:
                scenarios[int(match.group(1))] = match.group(2).strip()

    if not isinstance(table_xml, str):
        raise RuntimeError("E3 source is missing Table 1 XML")
    tree = ET.fromstring(table_xml)
    labels: list[str] = []
    for row in tree.findall(".//tr"):
        cells = ["".join(cell.itertext()).strip() for cell in row.findall("./td")]
        if cells and cells[0] in _SOURCE_LABELS:
            labels.append(cells[0])
    if tuple(labels) != _SOURCE_LABELS:
        raise RuntimeError("E3 Table 1 statistical-test order changed")
    if sorted(scenarios) != list(range(1, 21)):
        raise RuntimeError("E3 Appendix scenario numbering changed")

    rows = [
        {
            "case_id": f"shukla-2025-{index:02d}",
            "source_index": index,
            "question": scenarios[index],
            "source_gold_label": labels[index - 1],
        }
        for index in range(1, 21)
    ]
    source = {
        "pmcid": _PMCID,
        "doi": _DOI,
        "bioc_url": _BIOC_URL,
        "license": "CC BY",
        "case_count": len(rows),
        "source_pairing_rule": (
            "pair Appendix scenarios 1-20 with Table 1 statistical-test rows 1-20 in source order"
        ),
        "content_fingerprint": canonical_hash(rows),
        "payload_sha256": sha256_bytes(payload),
    }
    return source, rows


def _qualified_rows(rows: list[dict[str, Any]], observed_methods: set[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        label = str(row["source_gold_label"])
        method_id = _SOURCE_METHOD_MAP[label]
        catalog_eligible = method_id is not None
        head_eligible = bool(catalog_eligible and method_id in observed_methods)
        result.append(
            {
                **row,
                "gold_method_id": method_id,
                "catalog_eligible": catalog_eligible,
                "head_eligible": head_eligible,
                "eligible": head_eligible,
            }
        )
    return result


def _qualification(
    source: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    observed_methods: set[str],
) -> dict[str, Any]:
    qualified = _qualified_rows(rows, observed_methods)
    catalog_count = sum(bool(row["catalog_eligible"]) for row in qualified)
    head_count = sum(bool(row["head_eligible"]) for row in qualified)
    head_coverage = head_count / len(qualified)
    checks = {
        "complete_source": len(qualified) == 20,
        "cc_by_license": source["license"] == "CC BY",
        "minimum_head_eligible_cases": head_count >= 8,
        "minimum_head_coverage_fraction": head_coverage >= 0.40,
    }
    report = {
        "schema_version": 1,
        "complete": True,
        "method": "E3 source qualification for H14 selector-head external transfer",
        "source": source,
        "case_count": len(qualified),
        "catalog_eligible_count": catalog_count,
        "head_eligible_count": head_count,
        "head_coverage_fraction": head_coverage,
        "head_ineligible_catalog_methods": sorted(
            {
                str(row["gold_method_id"])
                for row in qualified
                if row["catalog_eligible"] and not row["head_eligible"]
            }
        ),
        "out_of_catalog_source_labels": [
            str(row["source_gold_label"]) for row in qualified if not row["catalog_eligible"]
        ],
        "qualification_gate": {"passed": all(checks.values()), "checks": checks},
        "model_evaluation_started": False,
        "claim_boundary": (
            "Qualification only. No E3 model output has been opened; source content was inspected "
            "before lock, so E3 is source-qualified rather than source-blinded."
        ),
    }
    report["fingerprint"] = canonical_hash(report)
    return report


def prepare_selector_external_contract(config: ProjectConfig) -> dict[str, Any]:
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "contract.json"
    public_path = config.root / "reports" / "evolve" / "selector-external-v1-contract.json"
    qualification_path = (
        config.root / "reports" / "evolve" / "selector-external-v1-source-qualification.json"
    )

    runtime_path = config.root / "reports" / "evolve" / "selector-runtime-v1-contract.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    h14_path = config.root / "reports" / "evolve" / "selector-head-v1-confirmation.json"
    h14 = json.loads(h14_path.read_text(encoding="utf-8"))
    if not h14.get("selector_head_architecture_confirmed"):
        raise RuntimeError("E3 requires confirmed H14 selector-head architecture")
    if runtime.get("source_h14_result_fingerprint") != h14.get("result_fingerprint"):
        raise RuntimeError("E3 runtime does not match the confirmed H14 result")

    payload = _request_bytes(_BIOC_URL)
    source, source_rows = _source_rows(payload)
    observed = set(str(value) for value in runtime["selector_head"]["observed_methods"])
    qualification = _qualification(source, source_rows, observed_methods=observed)
    write_json(qualification_path, qualification)
    if not qualification["qualification_gate"]["passed"]:
        raise RuntimeError("E3 source failed preregistration qualification")

    mapping = {label: _SOURCE_METHOD_MAP[label] for label in _SOURCE_LABELS}
    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": "E3 external natural-language selector-head transfer",
        "runtime_contract_fingerprint": runtime["fingerprint"],
        "runtime_contract_sha256": sha256_file(runtime_path),
        "h14_result_fingerprint": h14["result_fingerprint"],
        "source": source,
        "source_qualification_fingerprint": qualification["fingerprint"],
        "source_method_mapping": mapping,
        "source_method_mapping_sha256": canonical_hash(mapping),
        "eligibility_policy": (
            "score only source rows whose frozen gold maps to an H13-observed selector-head "
            "method; "
            "retain all 20 rows in coverage accounting"
        ),
        "arms": {
            "menu-free-control": (
                "unchanged v0.3 parent under the H14 menu-free canonical extraction prompt"
            ),
            "selector-head": (
                "same frozen parent prompt representation with the frozen selector-runtime-v1 head"
            ),
        },
        "gates": {
            "minimum_head_eligible_accuracy": 0.55,
            "minimum_method_gain_points": 20.0,
            "minimum_net_improvements": 2,
            "minimum_head_valid_output_rate": 1.0,
        },
        "implementation_sha256": _implementation_manifest(),
        "model_evaluation_started": False,
        "historical_pbench_statqa_reopened": False,
        "claim_boundary": (
            "A pass can support only narrow transfer of the frozen H14 method selector to the nine "
            "head-covered source-authored biomedical scenarios. It cannot establish 28-method "
            "coverage, column correctness, champion replacement, or release readiness."
        ),
    }
    contract["fingerprint"] = canonical_hash(contract)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("E3 contract is immutable")
        write_json(public_path, existing)
        return existing
    write_json(lock_path, contract)
    write_json(public_path, contract)
    return contract


def prepare_selector_external_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_selector_external_contract(config)
    payload = _request_bytes(_BIOC_URL)
    source, rows = _source_rows(payload)
    if source["content_fingerprint"] != contract["source"]["content_fingerprint"]:
        raise RuntimeError("E3 source content changed after contract freeze")
    runtime = json.loads(
        (config.root / "reports" / "evolve" / "selector-runtime-v1-contract.json").read_text(
            encoding="utf-8"
        )
    )
    observed = set(str(value) for value in runtime["selector_head"]["observed_methods"])
    cases = _qualified_rows(rows, observed)
    path = _data_root(config) / "cases.jsonl"
    write_jsonl(path, cases)
    head_count = sum(bool(row["head_eligible"]) for row in cases)
    coverage = head_count / len(cases)
    checks = {
        "source_fingerprint": source["content_fingerprint"]
        == contract["source"]["content_fingerprint"],
        "case_count": len(cases) == 20,
        "head_eligible_cases": head_count >= 8,
        "head_coverage": coverage >= 0.40,
    }
    report = {
        "schema_version": 1,
        "complete": True,
        "contract_fingerprint": contract["fingerprint"],
        "source_content_fingerprint": source["content_fingerprint"],
        "cases_sha256": sha256_file(path),
        "case_count": len(cases),
        "head_eligible_count": head_count,
        "head_coverage_fraction": coverage,
        "materializer_version": _MATERIALIZER_VERSION,
        "materializer_sha256": _implementation_manifest(),
        "checks": checks,
        "evaluation_authorized": all(checks.values()),
        "model_evaluation_started": False,
    }
    report["data_fingerprint"] = canonical_hash(report)
    write_json(_data_root(config) / "data-status.json", report)
    write_json(config.root / "reports" / "evolve" / "selector-external-v1-data.json", report)
    return report


def _evaluate_head(
    agent: StatsAgent,
    cases: list[dict[str, Any]],
    *,
    head: dict[str, Any],
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for case in cases:
        runtime_case = _runtime_case(str(case["question"]))
        vector = _hidden_vector(agent, runtime_case)
        predicted, top3, margin = _rank_methods(head, vector)
        eligible = bool(case["head_eligible"])
        correct = eligible and predicted == case["gold_method_id"]
        details.append(
            {
                "case_id": str(case["case_id"]),
                "eligible": eligible,
                "gold_method_id": case["gold_method_id"],
                "predicted_method_id": predicted,
                "valid_output": True,
                "correct": correct,
                "top3_methods": top3,
                "score_margin": margin,
            }
        )
    return {"metrics": _metrics(details), "details": details}


def _external_gate(
    control: dict[str, Any],
    candidate: dict[str, Any],
    paired: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    gain = 100.0 * (
        float(candidate["eligible_accuracy"]) - float(control["eligible_accuracy"])
    )
    checks = {
        "head_absolute_accuracy": float(candidate["eligible_accuracy"])
        >= float(gates["minimum_head_eligible_accuracy"]),
        "method_gain": gain >= float(gates["minimum_method_gain_points"]),
        "net_improvements": int(paired["net_improvements"])
        >= int(gates["minimum_net_improvements"]),
        "head_validity": float(candidate["valid_output_rate"])
        >= float(gates["minimum_head_valid_output_rate"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "method_gain_points": gain,
    }


def run_selector_external_evaluation(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_selector_external_contract(config)
    data = prepare_selector_external_data(config)
    if not data.get("evaluation_authorized"):
        raise RuntimeError("E3 source materialization did not authorize evaluation")
    report_path = _root(config) / "report.json"
    public_path = config.root / "reports" / "evolve" / "selector-external-v1.json"
    cases = list(read_jsonl(_data_root(config) / "cases.jsonl"))
    runtime = json.loads(
        (config.root / "reports" / "evolve" / "selector-runtime-v1-contract.json").read_text(
            encoding="utf-8"
        )
    )
    h14_contract = json.loads(
        (config.root / "reports" / "evolve" / "selector-head-v1-contract.json").read_text(
            encoding="utf-8"
        )
    )
    head = _load_head(h14_contract)
    evaluation_fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "data": data["data_fingerprint"],
            "runtime": runtime["fingerprint"],
            "evaluator_version": _EVALUATOR_VERSION,
            "implementation": _implementation_manifest(),
        }
    )
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != evaluation_fingerprint or not existing.get("complete"):
            raise RuntimeError("E3 report changed")
        public = dict(existing)
        public.pop("private_details", None)
        write_json(public_path, public)
        return public

    agent = StatsAgent(config, adapter_path=str(runtime["parent"]["adapter_path"]))
    agent.router.set_route("adapter")
    try:
        control = _evaluate_arm(
            agent,
            cases,
            name="menu-free-control",
            grounded=False,
            progress_root=_root(config) / "progress",
            evaluation_fingerprint=evaluation_fingerprint,
            max_tokens=160,
        )
        candidate = _evaluate_head(agent, cases, head=head)
    finally:
        del agent
        gc.collect()
        mx.clear_cache()

    paired = _paired_summary(control["details"], candidate["details"])
    gate = _external_gate(
        control["metrics"],
        candidate["metrics"],
        paired,
        dict(contract["gates"]),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": evaluation_fingerprint,
        "method": "E3 external natural-language selector-head transfer evaluation",
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["data_fingerprint"],
        "runtime_contract_fingerprint": runtime["fingerprint"],
        "same_parent_weights": True,
        "model_evaluation_started": True,
        "scores": {
            "menu-free-control": control["metrics"],
            "selector-head": candidate["metrics"],
        },
        "paired": paired,
        "external_gate": gate,
        "external_selector_head_transfer_supported": bool(gate["passed"]),
        "champion_unchanged": "v0.3.0-parent",
        "release_authorized": False,
        "historical_pbench_statqa_reopened": False,
        "next_step": (
            "qualify-additional-independent-natural-language-source"
            if gate["passed"]
            else "treat-h14-as-synthetic-format-limited-and-study-representation-transfer"
        ),
        "claim_boundary": contract["claim_boundary"],
        "private_details": {
            "menu-free-control": control["details"],
            "selector-head": candidate["details"],
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

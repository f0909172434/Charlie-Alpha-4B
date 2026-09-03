from __future__ import annotations

import gc
import hashlib
import http.cookiejar
import inspect
import json
import math
import os
import re
import tempfile
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
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
from .stats_eval import _append_progress, _json_from_answer, _load_progress
from .stats_family_router import _expert_context

_EVALUATOR_VERSION = 1
_MATERIALIZER_VERSION = 4
_VISUALLY_AUDITED_PDF_SHA256 = (
    "cfda4b7e86aa55b56ff8459dfd8de03f5cf73f674515b0d0bbfedd52c19df836"
)

_ALIASES: dict[str, tuple[str, ...]] = {
    "independent_t": (
        "pooled two sample t test",
        "independent t test",
        "independent samples t test",
        "independent sample t test",
        "unpaired t test",
        "unpaired student t test",
        "student t test for independent samples",
        "two sample student t test",
    ),
    "welch_t": (
        "welch t test",
        "welchs t test",
        "welch two sample t test",
        "unequal variance t test",
    ),
    "mann_whitney": (
        "mann whitney test",
        "mann whitney u test",
        "wilcoxon rank sum test",
        "rank sum test",
        "unpaired nonparametric test",
    ),
    "paired_t": (
        "paired t test",
        "paired student t test",
        "dependent t test",
        "dependent samples t test",
    ),
    "wilcoxon_signed_rank": (
        "wilcoxon signed rank test",
        "signed rank test",
        "paired nonparametric test",
    ),
    "chi_square": (
        "chi square test",
        "chi squared test",
        "pearson chi square test",
        "pearsons chi square test",
        "chi square test of independence",
        "test of independence",
    ),
    "fisher_exact": (
        "fisher exact test",
        "fishers exact test",
    ),
    "two_proportion": (
        "two proportion score test",
        "two proportion z test",
        "two sample proportion test",
        "z test for two proportions",
    ),
    "ols": (
        "ordinary least squares",
        "linear regression",
        "simple linear regression",
        "multiple linear regression",
        "multivariable linear regression",
    ),
    "hc3_ols": (
        "ols with hc3 robust covariance",
        "hc3 robust regression",
        "heteroskedasticity robust linear regression",
    ),
    "huber_regression": (
        "huber regression",
        "huber robust regression",
    ),
    "logistic_glm": (
        "logistic regression",
        "binary logistic regression",
        "multiple logistic regression",
        "multivariable logistic regression",
    ),
    "firth_logistic": (
        "firth logistic regression",
        "bias reduced logistic regression",
        "bias reduced logistic model",
    ),
    "poisson_glm": (
        "poisson regression",
        "poisson model",
    ),
    "negative_binomial_glm": (
        "negative binomial regression",
        "negative binomial model",
    ),
    "gee": (
        "generalized estimating equations",
        "generalised estimating equations",
        "gee",
    ),
    "mixed_effects": (
        "mixed effects model",
        "mixed effect model",
        "linear mixed model",
        "random effects model",
        "random intercept mixed model",
        "random intercept model",
    ),
    "cox_ph": (
        "cox proportional hazards model",
        "cox proportional hazards regression",
        "cox regression",
        "proportional hazards regression",
    ),
    "logrank": (
        "log rank test",
        "logrank test",
        "kaplan meier and log rank comparison",
        "kaplan meier analysis with log rank test",
    ),
    "multiple_imputation": (
        "multiple imputation",
        "multiple imputation with pooled inference",
    ),
    "ipw": (
        "inverse probability weighting",
        "inverse probability weighted analysis",
        "ipw",
    ),
    "difference_in_means": (
        "difference in means",
        "randomized difference in means",
        "randomised difference in means",
    ),
    "ancova": (
        "ancova",
        "analysis of covariance",
        "randomized ancova",
        "randomised ancova",
    ),
    "randomization_inference": (
        "randomization inference",
        "randomisation inference",
        "permutation test under random assignment",
    ),
    "conjugate_bayes": (
        "conjugate bayesian estimation",
        "conjugate bayes",
    ),
    "posterior_predictive": (
        "posterior predictive check",
        "posterior predictive model check",
    ),
    "calibrated_logistic": (
        "cross fitted calibrated prediction model",
        "calibrated logistic prediction model",
    ),
    "blocked_time_series_cv": (
        "rolling origin time series validation",
        "rolling origin validation",
        "forward chaining validation",
        "blocked time series cross validation",
    ),
}


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "external-catalog-interface-v1"


def _data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "external-catalog-interface-v1"


def _normalize_label(value: str) -> str:
    compatible = unicodedata.normalize("NFKC", value)
    compatible = re.sub(
        r"^\s*(?:answer|gold|correct answer)\s*[:\-]\s*",
        "",
        compatible,
        flags=re.I,
    )
    compatible = re.sub(r"[\*†‡]+\s*$", "", compatible)
    return "".join(character.lower() for character in compatible if character.isalnum())


def _alias_index() -> dict[str, str]:
    procedures = {procedure.method_id: procedure for procedure in PROCEDURES}
    if set(_ALIASES) != set(procedures):
        missing = sorted(set(procedures) - set(_ALIASES))
        extra = sorted(set(_ALIASES) - set(procedures))
        raise RuntimeError(f"External alias coverage changed: missing={missing}, extra={extra}")
    index: dict[str, str] = {}
    for method_id, aliases in sorted(_ALIASES.items()):
        procedure = procedures[method_id]
        for label in (method_id, procedure.name, *aliases):
            normalized = _normalize_label(label)
            prior = index.get(normalized)
            if prior is not None and prior != method_id:
                raise RuntimeError(
                    f"External alias collision for {label!r}: {prior} versus {method_id}"
                )
            index[normalized] = method_id
    return index


def _canonicalize_method_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return _alias_index().get(_normalize_label(value))


def _catalog_reference() -> str:
    return "\n".join(f"{procedure.method_id} — {procedure.name}" for procedure in PROCEDURES)


def _messages(case: dict[str, Any], *, grounded: bool) -> list[dict[str, str]]:
    system = (
        "Select the single best statistical procedure for the study vignette. Return exactly one "
        'JSON object and nothing else: {"method_id":"<canonical_method_id>"}. Use one '
        "repository-style canonical identifier and do not explain."
    )
    if grounded:
        system += (
            "\n\nRepository method catalog (fixed and identical for every case):\n"
            + _catalog_reference()
            + "\nSelect exactly one method identifier from this fixed catalog."
        )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": "Study vignette:\n" + str(case["vignette"]),
        },
    ]


def _predicted_method(answer: str) -> tuple[str | None, bool]:
    parsed = _json_from_answer(answer)
    raw: Any = parsed.get("method_id")
    if raw is None:
        raw = parsed.get("method")
    if raw is None:
        raw = parsed.get("methods")
    if isinstance(raw, list):
        raw = raw[0] if len(raw) == 1 else None
    canonical = _canonicalize_method_label(raw)
    return canonical, canonical is not None


def _metrics(details: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in details if bool(row["eligible"])]
    if not details:
        raise RuntimeError("External catalog evaluation has no rows")
    if not eligible:
        raise RuntimeError("External catalog evaluation has no in-catalog rows")
    return {
        "count": len(details),
        "eligible_count": len(eligible),
        "eligible_accuracy": sum(bool(row["correct"]) for row in eligible) / len(eligible),
        "all_case_accuracy": sum(bool(row["correct"]) for row in details) / len(details),
        "valid_output_rate": sum(bool(row["valid_output"]) for row in details) / len(details),
        "in_catalog_prediction_rate": sum(
            row["predicted_method_id"] is not None for row in details
        )
        / len(details),
    }


def _mcnemar_exact_pvalue(candidate_only: int, control_only: int) -> float:
    discordant = candidate_only + control_only
    if discordant == 0:
        return 1.0
    tail = min(candidate_only, control_only)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * probability)


def _paired_summary(
    control_details: list[dict[str, Any]],
    candidate_details: list[dict[str, Any]],
) -> dict[str, Any]:
    control = {str(row["case_id"]): row for row in control_details if bool(row["eligible"])}
    candidate = {str(row["case_id"]): row for row in candidate_details if bool(row["eligible"])}
    if set(control) != set(candidate):
        raise RuntimeError("External paired evaluation coverage changed between arms")
    counts = Counter[str]()
    for case_id in sorted(control):
        c = bool(control[case_id]["correct"])
        x = bool(candidate[case_id]["correct"])
        key = (
            "both_correct"
            if c and x
            else "candidate_only"
            if x
            else "control_only"
            if c
            else "both_wrong"
        )
        counts[key] += 1
    candidate_only = counts["candidate_only"]
    control_only = counts["control_only"]
    return {
        "eligible_count": len(control),
        "both_correct": counts["both_correct"],
        "candidate_only": candidate_only,
        "control_only": control_only,
        "both_wrong": counts["both_wrong"],
        "net_improvements": candidate_only - control_only,
        "mcnemar_exact_two_sided_p": _mcnemar_exact_pvalue(candidate_only, control_only),
    }


def _gate_report(
    *,
    data: dict[str, Any],
    control: dict[str, Any],
    candidate: dict[str, Any],
    paired: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    tolerance = 1e-9
    gain_points = 100 * (
        float(candidate["eligible_accuracy"]) - float(control["eligible_accuracy"])
    )
    validity_delta = 100 * (
        float(candidate["valid_output_rate"]) - float(control["valid_output_rate"])
    )
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
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "effect_points": {
            "eligible_method_accuracy": gain_points,
            "valid_output_rate": validity_delta,
        },
    }


def _evaluation_implementation_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    functions = {
        "canonicalize_method_label": _canonicalize_method_label,
        "messages": _messages,
        "predicted_method": _predicted_method,
        "metrics": _metrics,
        "paired_summary": _paired_summary,
        "gate_report": _gate_report,
    }
    manifest = {
        name: sha256_text(inspect.getsource(function)) for name, function in functions.items()
    }
    manifest.update(
        {
            "stats_agent.py": sha256_file(root / "stats_agent.py"),
            "stats_catalog.py": sha256_file(root / "stats_catalog.py"),
        }
    )
    return manifest


def prepare_external_catalog_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("external_catalog_interface"))
    root = _root(config)
    lock_path = root / "contract.json"
    public_path = config.root / "reports" / "evolve" / "external-catalog-interface-v1-contract.json"

    h12_path = config.root / "reports" / "evolve" / "catalog-interface-replication-v1.json"
    if not h12_path.exists():
        raise RuntimeError("E1 requires the completed H12 replication")
    h12 = json.loads(h12_path.read_text(encoding="utf-8"))
    if not h12.get("synthetic_catalog_interface_replicated") or not h12["replication_gate"][
        "passed"
    ]:
        raise RuntimeError("H12 did not authorize independent external interface evidence")

    source_root = root / "source"
    cases_path = _data_root(config) / "cases.jsonl"
    if not lock_path.exists() and (source_root.exists() or cases_path.exists()):
        raise RuntimeError("External source material appeared before the E1 contract was frozen")

    _, adapter_paths = _expert_context(config)
    parent = adapter_paths["parent"]
    parent_sha = sha256_file(parent / "adapters.safetensors")
    catalog = [(procedure.method_id, procedure.name) for procedure in PROCEDURES]
    alias_payload = {key: list(value) for key, value in sorted(_ALIASES.items())}
    fields: dict[str, Any] = {
        "schema_version": 1,
        "method": "E1 independent external fixed-catalog interface evaluation",
        "method_version": int(settings["method_version"]),
        "causal_question": (
            "On all eligible cases from one independently published expert-validated vignette "
            "source, does adding H7's unchanged fixed 28-method catalog improve the unchanged "
            "v0.3 parent over the same menu-free canonical-ID prompt?"
        ),
        "h12_result_fingerprint": h12["result_fingerprint"],
        "h12_report_sha256": sha256_file(h12_path),
        "parent": {
            "name": "v0.3.0-parent",
            "adapter_path": str(parent),
            "adapter_sha256": parent_sha,
        },
        "source": dict(settings["source"]),
        "source_state_at_preregistration": {
            "case_text_opened": False,
            "answer_key_opened": False,
            "local_source_files_present": False,
        },
        "catalog": {
            "procedure_count": len(catalog),
            "sha256": canonical_hash(catalog),
            "source": "stats_catalog.PROCEDURES method_id + existing display name only",
        },
        "gold_and_prediction_aliases": alias_payload,
        "eligibility_policy": dict(settings["eligibility"]),
        "gates": dict(settings["gates"]),
        "generation_max_tokens": int(settings["generation_max_tokens"]),
        "arms": {
            "menu-free-control": (
                "Unchanged v0.3 parent receives the external vignette and canonical-ID JSON "
                "contract without a method catalog"
            ),
            "flat-catalog": (
                "Same parent, vignette, JSON contract, and decoding plus H7's fixed 28-method "
                "ID + display-name catalog"
            ),
        },
        "materialization_policy": (
            "After this lock exists, retrieve only the official PMC package identified by the "
            "frozen PMCID; include all source vignettes in source order. Map gold labels only by "
            "the frozen exact alias table. Labels mapping to zero or multiple catalog procedures "
            "are out of catalog and excluded only from the primary paired denominator. No case, "
            "alias, prompt, gate, or output may be selected after seeing model answers."
        ),
        "adaptation_policy": "none after source or model outputs are opened",
        "stopping_rule": (
            "Materialize every frozen source vignette once, then run both arms once with resumable "
            "progress under one evaluator fingerprint."
        ),
        "decision_rule": (
            "Pass only if source coverage, absolute flat-catalog accuracy, paired gain, net "
            "improvements, and output-validity gates all pass."
        ),
        "claim_boundary": (
            "A pass supports an independent external context/interface elicitation claim only. "
            "It does not change weights, replace the champion, authorize release, or establish "
            "general statistical competence outside the eligible source cases."
        ),
        "evaluation_implementation_sha256": _evaluation_implementation_manifest(),
        "external_source_opened": False,
    }
    fields["fingerprint"] = canonical_hash(fields)
    root.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fields["fingerprint"]:
            raise RuntimeError("E1 contract is immutable")
        write_json(public_path, existing)
        return existing
    write_json(lock_path, fields)
    write_json(public_path, fields)
    return fields


def _request_bytes(url: str, *, opener: Any | None = None) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Charlie-Alpha-4B research evaluation; contact via repository metadata"
            )
        },
    )
    open_url = opener.open if opener is not None else urllib.request.urlopen
    with open_url(request, timeout=90) as response:  # noqa: S310
        return response.read()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary_path = Path(temporary)
        if temporary_path.exists():
            temporary_path.unlink()


def _download_pmc_pdf(url: str) -> tuple[bytes, dict[str, Any]]:
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    first = _request_bytes(url, opener=opener)
    if first.startswith(b"%PDF-"):
        return first, {"pow_required": False}

    challenge_match = re.search(rb'POW_CHALLENGE = "([^"]+)"', first)
    difficulty_match = re.search(rb'POW_DIFFICULTY = "([0-9]+)"', first)
    if challenge_match is None or difficulty_match is None:
        raise RuntimeError("Official PMC PDF endpoint returned neither PDF nor a known challenge")
    challenge = challenge_match.group(1).decode("ascii")
    difficulty = int(difficulty_match.group(1))
    prefix = "0" * difficulty
    nonce = 0
    while not hashlib.sha256(f"{challenge}{nonce}".encode()).hexdigest().startswith(prefix):
        nonce += 1
    cookie_jar.set_cookie(
        http.cookiejar.Cookie(
            version=0,
            name="cloudpmc-viewer-pow",
            value=f"{challenge},{nonce}",
            port=None,
            port_specified=False,
            domain="pmc.ncbi.nlm.nih.gov",
            domain_specified=False,
            domain_initial_dot=False,
            path="/",
            path_specified=True,
            secure=True,
            expires=None,
            discard=True,
            comment=None,
            comment_url=None,
            rest={},
            rfc2109=False,
        )
    )
    payload = _request_bytes(url, opener=opener)
    if not payload.startswith(b"%PDF-"):
        raise RuntimeError("Official PMC PDF challenge completed but no PDF was returned")
    return payload, {
        "pow_required": True,
        "difficulty": difficulty,
        "challenge_sha256": sha256_text(challenge),
        "nonce": nonce,
    }


def _pdf_page_count(payload: bytes) -> int:
    counts = [int(value) for value in re.findall(rb"/Count\s+(\d+)", payload)]
    if counts:
        return max(counts)
    return len(re.findall(rb"/Type\s*/Page(?!s)", payload))


def _stable_article_html(payload: bytes) -> bytes:
    normalized = re.sub(
        rb'(<meta\s+name="ncbi_phid"\s+content=")[^"]+("\s*/>)',
        rb"\1SESSION\2",
        payload,
        flags=re.IGNORECASE,
    )
    return re.sub(
        rb'(<input\s+type="hidden"\s+name="csrfmiddlewaretoken"\s+value=")[^"]+("\s*>)',
        rb"\1SESSION\2",
        normalized,
        flags=re.IGNORECASE,
    )


def _stable_bioc_xml(payload: bytes) -> bytes:
    return re.sub(rb"<date>[0-9]+</date>", b"<date>FETCH_DATE</date>", payload, count=1)


def _supplementary_asset_links(article_payload: bytes) -> list[str]:
    hrefs = {
        value.decode("utf-8", errors="replace")
        for value in re.findall(rb'href="([^"]+)"', article_payload, flags=re.IGNORECASE)
    }
    links: list[str] = []
    for href in sorted(hrefs):
        path = urllib.parse.urlparse(href).path.lower()
        if path.startswith("bin/") or any(
            marker in path
            for marker in ("/supplement", "/supp_", "/supp-", "/suppl", "/appendix", "/bin/")
        ):
            links.append(href)
    return links


def _download_official_package(config: ProjectConfig, contract: dict[str, Any]) -> dict[str, Any]:
    source = dict(contract["source"])
    source_root = _root(config) / "source"
    source_root.mkdir(parents=True, exist_ok=True)

    article_url = str(source["article_url"])
    article_payload = _request_bytes(article_url)
    article_path = source_root / "article.html"
    if article_path.exists():
        stored_article_payload = article_path.read_bytes()
        if sha256_bytes(_stable_article_html(stored_article_payload)) != sha256_bytes(
            _stable_article_html(article_payload)
        ):
            raise RuntimeError("Official PMC article content changed after source opening")
        article_payload = stored_article_payload
    else:
        _atomic_write_bytes(article_path, article_payload)

    pmcid = str(source["pmcid"])
    bioc_url = (
        "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/"
        f"BioC_xml/{pmcid}/unicode"
    )
    bioc_payload = _request_bytes(bioc_url)
    bioc_path = source_root / f"{pmcid}.bioc.xml"
    if bioc_path.exists():
        stored_bioc_payload = bioc_path.read_bytes()
        if sha256_bytes(_stable_bioc_xml(stored_bioc_payload)) != sha256_bytes(
            _stable_bioc_xml(bioc_payload)
        ):
            raise RuntimeError("Official NCBI BioC full text changed after source opening")
        bioc_payload = stored_bioc_payload
    else:
        _atomic_write_bytes(bioc_path, bioc_payload)

    pdf_match = re.search(
        rb'<meta\s+name="citation_pdf_url"\s+content="([^"]+)"',
        article_payload,
        flags=re.IGNORECASE,
    )
    if pdf_match is None:
        raise RuntimeError("Official PMC article HTML did not expose its PDF asset")
    pdf_url = urllib.parse.urljoin(article_url, pdf_match.group(1).decode("utf-8"))
    pdf_payload, pow_receipt = _download_pmc_pdf(pdf_url)
    pdf_path = source_root / "article.pdf"
    if pdf_path.exists() and sha256_file(pdf_path) != sha256_bytes(pdf_payload):
        raise RuntimeError("Official PMC PDF changed after source opening")
    _atomic_write_bytes(pdf_path, pdf_payload)
    pdf_sha256 = sha256_file(pdf_path)
    if pdf_sha256 != _VISUALLY_AUDITED_PDF_SHA256:
        raise RuntimeError("Official PMC PDF differs from the five-page source visually audited")

    supplementary_links = _supplementary_asset_links(article_payload)
    bioc_root = ET.fromstring(bioc_payload)
    passages = bioc_root.findall(".//passage")
    table_passages = []
    for passage in passages:
        info = {
            element.attrib.get("key"): element.text or ""
            for element in passage.findall("infon")
        }
        if info.get("type") == "table":
            table_passages.append(passage)

    try:
        legacy_oa_payload = _request_bytes(str(source["oa_api_url"]))
        legacy_oa_status = {
            "status": "available",
            "sha256": sha256_bytes(legacy_oa_payload),
        }
    except urllib.error.HTTPError as exc:
        legacy_oa_status = {"status": f"http-{exc.code}"}

    return {
        "article_url": article_url,
        "article_html_sha256": sha256_file(article_path),
        "article_html_stable_sha256": sha256_bytes(_stable_article_html(article_payload)),
        "bioc_url": bioc_url,
        "bioc_xml_sha256": sha256_file(bioc_path),
        "bioc_xml_stable_sha256": sha256_bytes(_stable_bioc_xml(bioc_payload)),
        "bioc_passage_count": len(passages),
        "bioc_table_count": len(table_passages),
        "pdf_url": pdf_url,
        "pdf_sha256": pdf_sha256,
        "pdf_page_count": _pdf_page_count(pdf_payload),
        "pdf_download": pow_receipt,
        "supplementary_link_count": len(supplementary_links),
        "supplementary_links": supplementary_links,
        "legacy_oa_api": legacy_oa_status,
        "visual_pdf_audit": {
            "complete": True,
            "all_pages_reviewed": True,
            "page_count": 5,
            "appendix_present": False,
            "supplement_present": False,
            "complete_vignette_set_present": False,
            "answer_key_present": False,
            "aggregate_table_count": 1,
            "example_vignette_count": 1,
        },
        "source_root": source_root,
    }


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _nxml_tables(path: Path) -> list[dict[str, Any]]:
    document = ET.fromstring(path.read_bytes())
    tables: list[dict[str, Any]] = []
    for index, wrapper in enumerate(document.findall(".//{*}table-wrap"), start=1):
        table = wrapper.find(".//{*}table")
        if table is None:
            continue
        rows: list[list[str]] = []
        for row in table.findall(".//{*}tr"):
            cells = [
                _element_text(cell)
                for cell in list(row)
                if cell.tag.rsplit("}", 1)[-1] in {"td", "th"}
            ]
            if cells:
                rows.append(cells)
        tables.append(
            {
                "source_file": path.name,
                "table_index": index,
                "label": _element_text(wrapper.find("./{*}label")),
                "caption": _element_text(wrapper.find("./{*}caption")),
                "rows": rows,
            }
        )
    return tables


def _docx_tables(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
    tables: list[dict[str, Any]] = []
    for index, table in enumerate(document.findall(".//{*}tbl"), start=1):
        rows: list[list[str]] = []
        for row in table.findall("./{*}tr"):
            cells = [
                " ".join(
                    " ".join(text.text or "" for text in cell.findall(".//{*}t")).split()
                )
                for cell in row.findall("./{*}tc")
            ]
            if cells:
                rows.append(cells)
        tables.append(
            {
                "source_file": path.name,
                "table_index": index,
                "label": "",
                "caption": "",
                "rows": rows,
            }
        )
    return tables


def _bioc_tables(path: Path) -> list[dict[str, Any]]:
    document = ET.fromstring(path.read_bytes())
    captions: dict[str, str] = {}
    for passage in document.findall(".//passage"):
        info = {
            element.attrib.get("key"): element.text or "" for element in passage.findall("infon")
        }
        if info.get("type") == "table_caption":
            captions[str(info.get("id", ""))] = passage.findtext("text") or ""

    tables: list[dict[str, Any]] = []
    for passage in document.findall(".//passage"):
        info = {
            element.attrib.get("key"): element.text or "" for element in passage.findall("infon")
        }
        if info.get("type") != "table":
            continue
        xml_payload = str(info.get("xml", "")).strip()
        rows: list[list[str]] = []
        if xml_payload:
            table = ET.fromstring(xml_payload)
            for row in table.findall(".//tr"):
                cells = [
                    _element_text(cell)
                    for cell in list(row)
                    if cell.tag.rsplit("}", 1)[-1] in {"td", "th"}
                ]
                if cells:
                    rows.append(cells)
        if not rows:
            text = passage.findtext("text") or ""
            rows = [[text]] if text else []
        table_id = str(info.get("id", ""))
        tables.append(
            {
                "source_file": path.name,
                "table_index": len(tables) + 1,
                "label": table_id,
                "caption": captions.get(table_id, ""),
                "rows": rows,
            }
        )
    return tables


def _source_tables(source_root: Path) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    for path in sorted(source_root.rglob("*.nxml")):
        tables.extend(_nxml_tables(path))
    for path in sorted(source_root.rglob("*.docx")):
        tables.extend(_docx_tables(path))
    for path in sorted(source_root.rglob("*.bioc.xml")):
        tables.extend(_bioc_tables(path))
    return tables


def _header_index(headers: list[str], phrases: tuple[str, ...]) -> int | None:
    normalized = [_normalize_label(value) for value in headers]
    for phrase in phrases:
        target = _normalize_label(phrase)
        for index, value in enumerate(normalized):
            if target and target in value:
                return index
    return None


def _select_case_table(
    tables: list[dict[str, Any]],
    *,
    expected_count: int,
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    candidates: list[tuple[int, dict[str, Any], list[tuple[str, str]]]] = []
    for table in tables:
        rows = [list(map(str, row)) for row in table.get("rows", []) if len(row) >= 2]
        for has_header in (True, False):
            body = rows[1:] if has_header else rows
            if len(body) != expected_count:
                continue
            width = min(len(row) for row in body)
            if width < 2:
                continue
            headers = rows[0] if has_header else [""] * width
            question_index = _header_index(
                headers,
                ("study vignette", "vignette", "scenario", "research question", "question", "case"),
            )
            answer_index = _header_index(
                headers,
                (
                    "correct statistical test",
                    "appropriate statistical test",
                    "statistical test",
                    "correct answer",
                    "answer",
                ),
            )
            score = 0
            if question_index is not None:
                score += 5
            if answer_index is not None:
                score += 5
            if question_index is None:
                averages = [
                    sum(len(row[index]) for row in body if len(row) > index) / len(body)
                    for index in range(width)
                ]
                question_index = max(range(width), key=averages.__getitem__)
            if answer_index is None:
                choices = [index for index in range(width) if index != question_index]
                averages = {
                    index: sum(len(row[index]) for row in body if len(row) > index) / len(body)
                    for index in choices
                }
                answer_index = min(choices, key=averages.__getitem__)
            if question_index == answer_index:
                continue
            pairs = [
                (row[question_index].strip(), row[answer_index].strip())
                for row in body
                if len(row) > max(question_index, answer_index)
            ]
            if len(pairs) != expected_count or any(
                not question or not answer for question, answer in pairs
            ):
                continue
            descriptor = " ".join(
                [str(table.get("label", "")), str(table.get("caption", "")), *headers]
            ).lower()
            score += sum(
                term in descriptor for term in ("scenario", "vignette", "statistical test")
            )
            if has_header:
                score += 1
            candidates.append((score, table, pairs))
    if not candidates:
        raise RuntimeError(
            f"No official source table contained exactly {expected_count} complete vignette rows"
        )
    candidates.sort(
        key=lambda item: (
            item[0],
            str(item[1].get("source_file", "")),
            int(item[1].get("table_index", 0)),
        ),
        reverse=True,
    )
    top_score = candidates[0][0]
    top = [candidate for candidate in candidates if candidate[0] == top_score]
    if len(top) != 1:
        locators = [
            f"{item[1].get('source_file')}#{item[1].get('table_index')}" for item in top
        ]
        raise RuntimeError(f"Official source table selection was ambiguous: {locators}")
    _, table, pairs = top[0]
    return table, pairs


def _materializer_manifest() -> dict[str, str]:
    functions = {
        "download_official_package": _download_official_package,
        "download_pmc_pdf": _download_pmc_pdf,
        "pdf_page_count": _pdf_page_count,
        "stable_article_html": _stable_article_html,
        "stable_bioc_xml": _stable_bioc_xml,
        "supplementary_asset_links": _supplementary_asset_links,
        "nxml_tables": _nxml_tables,
        "docx_tables": _docx_tables,
        "bioc_tables": _bioc_tables,
        "select_case_table": _select_case_table,
    }
    return {name: sha256_text(inspect.getsource(function)) for name, function in functions.items()}


def prepare_external_catalog_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_external_catalog_contract(config)
    root = _root(config)
    private_path = _data_root(config) / "cases.jsonl"
    manifest_path = root / "data.json"
    public_path = config.root / "reports" / "evolve" / "external-catalog-interface-v1-data.json"
    source_unavailable_path = (
        config.root / "reports" / "evolve" / "external-catalog-interface-v1-source-unavailable.json"
    )
    superseded_source_audit_fingerprint: str | None = None
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("source_status") == "published-source-omits-case-set":
            if private_path.exists():
                raise RuntimeError("E1 source-unavailable closure unexpectedly has case data")
            if int(existing.get("materializer_version", 0)) == _MATERIALIZER_VERSION:
                write_json(public_path, existing)
                write_json(source_unavailable_path, existing)
                return existing
            superseded_source_audit_fingerprint = str(existing["fingerprint"])
        else:
            if not private_path.exists() or existing.get("cases_sha256") != sha256_file(
                private_path
            ):
                raise RuntimeError("E1 materialized cases changed or disappeared")
            write_json(public_path, existing)
            return existing

    source_receipt = _download_official_package(config, contract)
    source_root = Path(source_receipt.pop("source_root"))
    tables = _source_tables(source_root)
    diagnostics = {
        "table_count": len(tables),
        "tables": [
            {
                "source_file": table["source_file"],
                "table_index": table["table_index"],
                "label": table["label"],
                "caption": table["caption"],
                "row_count": len(table["rows"]),
                "column_counts": sorted({len(row) for row in table["rows"]}),
            }
            for table in tables
        ],
    }
    write_json(root / "source-table-diagnostics.json", diagnostics)
    expected_count = int(contract["source"]["expected_case_count"])
    try:
        table, pairs = _select_case_table(tables, expected_count=expected_count)
    except RuntimeError as exc:
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "complete": True,
            "terminal": True,
            "method": "E1 official external vignette source availability audit",
            "contract_fingerprint": contract["fingerprint"],
            "source_receipt": source_receipt,
            "source_status": "published-source-omits-case-set",
            "case_materialization_complete": False,
            "expected_case_count": expected_count,
            "materialized_case_count": 0,
            "published_table_count": len(tables),
            "published_table_summaries": diagnostics["tables"],
            "source_failure": {
                "class": "source-unavailable",
                "parser_result": str(exc),
                "reason": (
                    "The official PMC HTML, NCBI BioC full text, and visually reviewed five-page "
                    "PDF publish aggregate performance plus one example vignette, but omit the "
                    "complete 27 vignettes and their answer key required by the frozen contract."
                ),
            },
            "model_evaluation_started": False,
            "evaluation_authorized": False,
            "external_claim_decidable": False,
            "independent_external_interface_supported": False,
            "champion_unchanged": "v0.3.0-parent",
            "release_authorized": False,
            "materializer_version": _MATERIALIZER_VERSION,
            "materializer_sha256": _materializer_manifest(),
            "next_step": "close-e1-source-unavailable-without-model-evaluation",
        }
        if superseded_source_audit_fingerprint is not None:
            manifest["supersedes_source_audit_fingerprint"] = (
                superseded_source_audit_fingerprint
            )
        manifest["fingerprint"] = canonical_hash(manifest)
        write_json(manifest_path, manifest)
        write_json(public_path, manifest)
        write_json(source_unavailable_path, manifest)
        return manifest

    rows: list[dict[str, Any]] = []
    for index, (vignette, gold_raw) in enumerate(pairs, start=1):
        gold_method = _canonicalize_method_label(gold_raw)
        rows.append(
            {
                "case_id": f"mondal-2024-{index:03d}",
                "source_order": index,
                "vignette": vignette,
                "gold_raw": gold_raw,
                "gold_method_id": gold_method,
                "eligible": gold_method is not None,
                "source_locator": {
                    "source_file": table["source_file"],
                    "table_index": table["table_index"],
                    "row_index": index,
                },
            }
        )
    if len(rows) != expected_count or len({row["case_id"] for row in rows}) != expected_count:
        raise RuntimeError("E1 did not materialize the complete frozen source case set")
    write_jsonl(private_path, rows)
    eligible_count = sum(bool(row["eligible"]) for row in rows)
    eligibility = dict(contract["eligibility_policy"])
    coverage_fraction = eligible_count / len(rows)
    coverage_checks = {
        "all_source_cases_materialized": len(rows) == expected_count,
        "minimum_eligible_cases": eligible_count >= int(eligibility["minimum_eligible_cases"]),
        "minimum_coverage_fraction": coverage_fraction
        >= float(eligibility["minimum_coverage_fraction"]),
    }
    method_counts = Counter(
        str(row["gold_method_id"]) for row in rows if row["gold_method_id"] is not None
    )
    data_fingerprint = canonical_hash(rows)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "E1 official external vignette materialization",
        "contract_fingerprint": contract["fingerprint"],
        "source_receipt": source_receipt,
        "source_table": {
            "source_file": table["source_file"],
            "table_index": table["table_index"],
            "label": table["label"],
            "caption": table["caption"],
        },
        "case_count": len(rows),
        "eligible_count": eligible_count,
        "out_of_catalog_count": len(rows) - eligible_count,
        "coverage_fraction": coverage_fraction,
        "eligible_method_counts": dict(sorted(method_counts.items())),
        "unmapped_gold_labels_sha256": canonical_hash(
            sorted(str(row["gold_raw"]) for row in rows if not row["eligible"])
        ),
        "coverage_gate": {
            "passed": all(coverage_checks.values()),
            "checks": coverage_checks,
        },
        "evaluation_authorized": all(coverage_checks.values()),
        "materializer_version": _MATERIALIZER_VERSION,
        "materializer_sha256": _materializer_manifest(),
        "cases_sha256": sha256_file(private_path),
        "data_fingerprint": data_fingerprint,
        "next_step": (
            "run-one-shot-menu-free-vs-flat-catalog-evaluation"
            if all(coverage_checks.values())
            else "reject-source-for-insufficient-catalog-coverage"
        ),
    }
    manifest["fingerprint"] = canonical_hash(manifest)
    write_json(manifest_path, manifest)
    write_json(public_path, manifest)
    return manifest


def _evaluate_arm(
    agent: StatsAgent,
    cases: list[dict[str, Any]],
    *,
    name: str,
    grounded: bool,
    progress_root: Path,
    evaluation_fingerprint: str,
    max_tokens: int,
) -> dict[str, Any]:
    progress_path = progress_root / f"{name}.jsonl"
    fingerprint = canonical_hash(
        {
            "evaluation": evaluation_fingerprint,
            "name": name,
            "grounded": grounded,
            "evaluator_version": _EVALUATOR_VERSION,
        }
    )
    cached = _load_progress(progress_path, fingerprint=fingerprint, id_field="case_id")
    details: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        if case_id in cached:
            details.append(cached[case_id])
            continue
        answer = agent.answer_without_tools(
            _messages(case, grounded=grounded),
            route="stats",
            max_tokens=max_tokens,
            temperature=0.0,
        )
        predicted, valid = _predicted_method(answer)
        eligible = bool(case["eligible"])
        correct = eligible and predicted == case["gold_method_id"]
        row = {
            "case_id": case_id,
            "eligible": eligible,
            "gold_method_id": case["gold_method_id"],
            "predicted_method_id": predicted,
            "valid_output": valid,
            "correct": correct,
        }
        details.append(row)
        _append_progress(
            progress_path,
            fingerprint=fingerprint,
            row=row,
            completed=len(details),
        )
    return {"metrics": _metrics(details), "details": details}


def run_external_catalog_evaluation(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_external_catalog_contract(config)
    data = prepare_external_catalog_data(config)
    if not data.get("evaluation_authorized"):
        report_path = _root(config) / "report.json"
        public_path = config.root / "reports" / "evolve" / "external-catalog-interface-v1.json"
        fingerprint = canonical_hash(
            {
                "contract": contract["fingerprint"],
                "source_audit": data["fingerprint"],
                "terminal_reason": data.get("source_status"),
            }
        )
        report: dict[str, Any] = {
            "schema_version": 1,
            "complete": True,
            "terminal": True,
            "fingerprint": fingerprint,
            "method": "E1 independent external fixed-catalog interface evaluation",
            "contract_fingerprint": contract["fingerprint"],
            "source_audit_fingerprint": data["fingerprint"],
            "model_evaluation_started": False,
            "external_gate": {
                "passed": False,
                "decidable": False,
                "reason": "frozen external source omitted the case set and answer key",
            },
            "independent_external_interface_supported": False,
            "independent_external_interface_rejected": False,
            "champion_unchanged": "v0.3.0-parent",
            "release_authorized": False,
            "next_step": "close-e1-source-unavailable-without-model-evaluation",
            "claim_boundary": contract["claim_boundary"],
        }
        report["result_fingerprint"] = canonical_hash(report)
        if report_path.exists():
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            if existing.get("result_fingerprint") != report["result_fingerprint"]:
                raise RuntimeError("E1 terminal source-unavailable report changed")
            report = existing
        else:
            write_json(report_path, report)
        write_json(public_path, report)
        return report
    report_path = _root(config) / "report.json"
    public_path = config.root / "reports" / "evolve" / "external-catalog-interface-v1.json"
    cases = list(read_jsonl(_data_root(config) / "cases.jsonl"))
    _, adapter_paths = _expert_context(config)
    parent = adapter_paths["parent"]
    parent_sha = sha256_file(parent / "adapters.safetensors")
    if parent_sha != contract["parent"]["adapter_sha256"]:
        raise RuntimeError("E1 parent adapter changed")
    evaluation_fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "data": data["data_fingerprint"],
            "parent": parent_sha,
            "evaluator_version": _EVALUATOR_VERSION,
            "implementation": _evaluation_implementation_manifest(),
        }
    )
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != evaluation_fingerprint or not existing.get("complete"):
            raise RuntimeError("E1 report changed")
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
    gate = _gate_report(
        data=data,
        control=control["metrics"],
        candidate=candidate["metrics"],
        paired=paired,
        gates=dict(contract["gates"]),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": evaluation_fingerprint,
        "method": "E1 independent external fixed-catalog interface evaluation",
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["data_fingerprint"],
        "same_parent_weights": True,
        "scores": {
            "menu-free-control": control["metrics"],
            "flat-catalog": candidate["metrics"],
        },
        "paired": paired,
        "external_gate": gate,
        "independent_external_interface_supported": bool(gate["passed"]),
        "champion_unchanged": "v0.3.0-parent",
        "release_authorized": False,
        "next_step": (
            "preserve-e1-positive-interface-evidence-without-weight-promotion"
            if gate["passed"]
            else "reject-independent-external-interface-claim"
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

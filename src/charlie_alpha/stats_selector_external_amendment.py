from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json
from .stats_agent import StatsAgent
from .stats_catalog_grounding import _messages as _h14_messages
from .stats_eval import _append_progress, _json_from_answer, _load_progress
from .stats_external_catalog import _canonicalize_method_label, _metrics, _paired_summary
from .stats_representation_probe import _METHOD_IDS
from .stats_selector_external import _evaluate_head, _external_gate
from .stats_selector_head import _load_head

_AMENDMENT_VERSION = 1
_EVALUATOR_VERSION = 2


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "selector-external-v1"


def _implementation_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "stats_selector_external_amendment.py": sha256_file(Path(__file__)),
        "stats_selector_external.py": sha256_file(root / "stats_selector_external.py"),
        "stats_catalog_grounding.py": sha256_file(root / "stats_catalog_grounding.py"),
    }


def _failed_attempt_evidence(config: ProjectConfig) -> dict[str, Any]:
    progress = _root(config) / "progress" / "menu-free-control.jsonl"
    status_path = progress.with_suffix(".status.json")
    report_path = _root(config) / "report.json"
    rows = sum(1 for row in read_jsonl(progress)) if progress.exists() else 0
    status = (
        json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else None
    )
    completed = int(status.get("completed", -1)) if isinstance(status, dict) else -1
    return {
        "failed_progress_exists": progress.exists(),
        "failed_progress_rows": rows,
        "failed_progress_status": status,
        "failed_progress_completed": completed,
        "terminal_report_exists": report_path.exists(),
        "no_model_answer_observed": rows == 0 and completed == 0 and not report_path.exists(),
    }


def prepare_selector_external_evaluation_amendment(config: ProjectConfig) -> dict[str, Any]:
    lock_path = _root(config) / "evaluation-amendment-v1.json"
    public_path = (
        config.root / "reports" / "evolve" / "selector-external-v1-evaluation-amendment.json"
    )
    contract_path = config.root / "reports" / "evolve" / "selector-external-v1-contract.json"
    data_path = config.root / "reports" / "evolve" / "selector-external-v1-data.json"
    runtime_path = config.root / "reports" / "evolve" / "selector-runtime-v1-contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    data = json.loads(data_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if not data.get("evaluation_authorized"):
        raise RuntimeError("E3 amendment requires authorized frozen E3 data")
    if contract.get("runtime_contract_fingerprint") != runtime.get("fingerprint"):
        raise RuntimeError("E3 amendment runtime fingerprint changed")

    failed = _failed_attempt_evidence(config)
    if not failed["no_model_answer_observed"]:
        raise RuntimeError("E3 evaluator amendment is forbidden after any model answer")

    amendment: dict[str, Any] = {
        "schema_version": 1,
        "method": "E3 evaluator-only pre-output amendment",
        "amendment_version": _AMENDMENT_VERSION,
        "contract_fingerprint": contract["fingerprint"],
        "contract_sha256": sha256_file(contract_path),
        "data_fingerprint": data["data_fingerprint"],
        "data_sha256": sha256_file(data_path),
        "runtime_contract_fingerprint": runtime["fingerprint"],
        "runtime_contract_sha256": sha256_file(runtime_path),
        "failed_attempt_evidence": failed,
        "defects": [
            {
                "kind": "field-interface",
                "observed": "frozen E3 rows use question while legacy E1 scorer expected vignette",
                "failure_point": "prompt construction before the first model call",
            },
            {
                "kind": "prompt-contract",
                "observed": (
                    "legacy E1 scorer used a method-only prompt while the E3 contract requires "
                    "the exact H14 methods+columns menu-free prompt"
                ),
                "causal_risk": (
                    "control and selector-head would otherwise receive different prompts"
                ),
            },
        ],
        "correction": {
            "control_prompt": (
                "use stats_catalog_grounding._messages(case, grounded=False), exactly the H14 "
                "menu-free methods+columns extraction prompt"
            ),
            "selector_head_prompt": "unchanged frozen H14 menu-free representation prompt",
            "field_adapter": "none; both arms consume the frozen question field directly",
            "source_changed": False,
            "cases_changed": False,
            "eligibility_changed": False,
            "mapping_changed": False,
            "gates_changed": False,
            "runtime_changed": False,
            "head_changed": False,
        },
        "implementation_sha256": _implementation_manifest(),
        "model_evaluation_started": False,
        "claim_boundary": contract["claim_boundary"],
    }
    amendment["fingerprint"] = canonical_hash(amendment)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != amendment["fingerprint"]:
            raise RuntimeError("E3 evaluation amendment is immutable")
        write_json(public_path, existing)
        return existing
    write_json(lock_path, amendment)
    write_json(public_path, amendment)
    return amendment


def _predicted_method(answer: str) -> tuple[str | None, bool]:
    parsed = _json_from_answer(answer)
    raw: Any = parsed.get("methods")
    if isinstance(raw, list):
        raw = raw[0] if len(raw) == 1 else None
    if isinstance(raw, str) and raw in _METHOD_IDS:
        return raw, True
    canonical = _canonicalize_method_label(raw)
    return canonical, canonical is not None


def _evaluate_control(
    agent: StatsAgent,
    cases: list[dict[str, Any]],
    *,
    progress_root: Path,
    evaluation_fingerprint: str,
) -> dict[str, Any]:
    progress_path = progress_root / "menu-free-control.jsonl"
    fingerprint = canonical_hash(
        {
            "evaluation": evaluation_fingerprint,
            "name": "menu-free-control-h14-prompt",
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
            _h14_messages(case, grounded=False),
            route="stats",
            max_tokens=160,
            temperature=0.0,
        )
        predicted, valid = _predicted_method(answer)
        eligible = bool(case["head_eligible"])
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


def run_selector_external_evaluation_amended(config: ProjectConfig) -> dict[str, Any]:
    amendment = prepare_selector_external_evaluation_amendment(config)
    contract = json.loads(
        (config.root / "reports" / "evolve" / "selector-external-v1-contract.json").read_text(
            encoding="utf-8"
        )
    )
    data = json.loads(
        (config.root / "reports" / "evolve" / "selector-external-v1-data.json").read_text(
            encoding="utf-8"
        )
    )
    runtime = json.loads(
        (config.root / "reports" / "evolve" / "selector-runtime-v1-contract.json").read_text(
            encoding="utf-8"
        )
    )
    if not data.get("evaluation_authorized"):
        raise RuntimeError("E3 frozen data did not authorize amended evaluation")
    cases_path = config.path_for("evolution_dir") / "selector-external-v1" / "cases.jsonl"
    if sha256_file(cases_path) != data.get("cases_sha256"):
        raise RuntimeError("E3 frozen cases changed before amended evaluation")
    cases = list(read_jsonl(cases_path))
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
            "amendment": amendment["fingerprint"],
            "evaluator_version": _EVALUATOR_VERSION,
            "implementation": _implementation_manifest(),
        }
    )
    report_path = _root(config) / "report-amended-v1.json"
    public_path = config.root / "reports" / "evolve" / "selector-external-v1.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != evaluation_fingerprint or not existing.get("complete"):
            raise RuntimeError("E3 amended report changed")
        public = dict(existing)
        public.pop("private_details", None)
        write_json(public_path, public)
        return public

    agent = StatsAgent(config, adapter_path=str(runtime["parent"]["adapter_path"]))
    agent.router.set_route("adapter")
    try:
        control = _evaluate_control(
            agent,
            cases,
            progress_root=_root(config) / "amended-progress",
            evaluation_fingerprint=evaluation_fingerprint,
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
        "method": "E3 amended external natural-language selector-head transfer evaluation",
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["data_fingerprint"],
        "runtime_contract_fingerprint": runtime["fingerprint"],
        "evaluation_amendment_fingerprint": amendment["fingerprint"],
        "evaluation_amendment_applied": True,
        "same_parent_weights": True,
        "same_h14_prompt_both_arms": True,
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

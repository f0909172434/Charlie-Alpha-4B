from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json
from .stats_cross_format import (
    _FORMAT_EVALUATOR_VERSION,
    _data_root,
    _evaluate,
    _format_shift_case,
    _gate_report,
    _records,
    _root,
    _simulate_registered,
)
from .stats_family_router import _expert_context

_AMENDMENT_VERSION = 1


def _frozen_contract(config: ProjectConfig) -> dict[str, Any]:
    contract_path = _root(config) / "contract.json"
    if not contract_path.exists():
        raise RuntimeError("H5 cross-format contract is missing")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    frozen_source_sha = contract["implementation_sha256"]["stats_cross_format.py"]
    current_source_sha = sha256_file(Path(__file__).with_name("stats_cross_format.py"))
    if current_source_sha != frozen_source_sha:
        raise RuntimeError("Frozen H5 v2 implementation changed after registration")
    if int(contract["method_version"]) != 2:
        raise RuntimeError("The evaluation amendment is defined only for H5 v2")
    return contract


def _arm_statuses(config: ProjectConfig) -> dict[str, dict[str, Any]]:
    arms = ("selector-only-full-sequence", "multi-token-representation")
    statuses: dict[str, dict[str, Any]] = {}
    for arm in arms:
        path = _root(config) / "arms" / arm / "status.json"
        if not path.exists():
            raise RuntimeError(f"H5 v2 arm is incomplete: {arm}")
        status = json.loads(path.read_text(encoding="utf-8"))
        if not status.get("complete") or int(status.get("microsteps", -1)) != 48:
            raise RuntimeError(f"H5 v2 arm did not complete fixed compute: {arm}")
        if status.get("selection_opened") or status.get("confirmation_opened"):
            raise RuntimeError(f"H5 v2 arm was evaluated before the amendment: {arm}")
        statuses[arm] = status
    return statuses


def _paired_selector_rows(
    records: list[dict[str, Any]],
    simulations: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if len(records) != len(simulations):
        raise RuntimeError("Cross-format selector record/simulation counts differ")
    paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record, simulation in zip(records, simulations, strict=True):
        record_id = str(record["metadata"]["blueprint_id"])
        simulation_id = str(simulation["scenario"]["blueprint_id"])
        if record_id != simulation_id:
            raise RuntimeError(
                f"Cross-format selector pairing mismatch: {record_id} != {simulation_id}"
            )
        paired.append((record, simulation))
    return paired


def prepare_cross_format_evaluation_amendment(config: ProjectConfig) -> dict[str, Any]:
    contract = _frozen_contract(config)
    statuses = _arm_statuses(config)
    data_status_path = _data_root(config) / "data-status.json"
    if not data_status_path.exists():
        raise RuntimeError("H5 v2 data status is missing")
    data_status = json.loads(data_status_path.read_text(encoding="utf-8"))
    amendment_path = (
        config.root / "reports" / "evolve" / "cross-format-repair-v2-evaluation-amendment.json"
    )
    if amendment_path.exists():
        existing = json.loads(amendment_path.read_text(encoding="utf-8"))
        payload = {key: value for key, value in existing.items() if key != "fingerprint"}
        if canonical_hash(payload) != existing.get("fingerprint"):
            raise RuntimeError("H5 v2 evaluation amendment fingerprint is corrupt")
        return existing

    progress_root = _root(config) / "selection-progress"
    if progress_root.exists() and any(progress_root.iterdir()):
        raise RuntimeError("H5 v2 selection had already produced outputs before the amendment")
    if (_root(config) / "pilot.json").exists():
        raise RuntimeError("H5 v2 pilot report already exists; amendment is too late")
    confirmation_path = _data_root(config) / "surfaces" / "confirmation_shard.jsonl"
    if confirmation_path.exists():
        raise RuntimeError("H5 v2 confirmation was opened before the evaluation amendment")

    selection_records = list(read_jsonl(_data_root(config) / "selection.jsonl"))
    selection_surface = list(
        read_jsonl(_data_root(config) / "surfaces" / "selection_shard.jsonl")
    )
    paired = _paired_selector_rows(selection_records, selection_surface)
    fields: dict[str, Any] = {
        "schema_version": 1,
        "status": "accepted-before-selection",
        "amendment_version": _AMENDMENT_VERSION,
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data_status["fingerprint"],
        "bug": (
            "The frozen v2 pilot passed selection records directly to _score_loaded_selector, "
            "which requires (record, simulation) pairs. The run failed before _load_progress and "
            "before any menu-free generation."
        ),
        "correction": (
            "Zip each frozen selection record with the matching frozen selection simulation by "
            "blueprint_id before selector scoring. Menu-free cases, adapters, gates, output "
            "scoring, and the confirmation rule are unchanged."
        ),
        "selection_outputs_before_amendment": 0,
        "confirmation_opened_before_amendment": False,
        "training_reused_without_retraining": True,
        "fixed_arm_weights": {
            arm: {
                "adapter_sha256": status["adapter_sha256"],
                "microsteps": int(status["microsteps"]),
                "optimizer_updates": int(status["optimizer_updates"]),
            }
            for arm, status in sorted(statuses.items())
        },
        "selection_pair_count": len(paired),
        "selection_pair_fingerprint": canonical_hash(
            [
                {
                    "record": str(record["metadata"]["blueprint_id"]),
                    "simulation": str(simulation["scenario"]["blueprint_id"]),
                }
                for record, simulation in paired
            ]
        ),
        "frozen_selection_sha256": {
            "records": sha256_file(_data_root(config) / "selection.jsonl"),
            "surface": sha256_file(
                _data_root(config) / "surfaces" / "selection_shard.jsonl"
            ),
            "format_cases": sha256_file(_data_root(config) / "selection-format.jsonl"),
        },
        "amendment_implementation_sha256": sha256_file(Path(__file__)),
        "research_policy_change": False,
        "training_policy_change": False,
        "selection_gate_change": False,
    }
    amendment = dict(fields)
    amendment["fingerprint"] = canonical_hash(fields)
    write_json(amendment_path, amendment)
    return amendment


def run_cross_format_pilot_amended(config: ProjectConfig) -> dict[str, Any]:
    contract = _frozen_contract(config)
    amendment = prepare_cross_format_evaluation_amendment(config)
    statuses = _arm_statuses(config)
    data = json.loads((_data_root(config) / "data-status.json").read_text(encoding="utf-8"))
    selection_records = list(read_jsonl(_data_root(config) / "selection.jsonl"))
    selection_surface = list(
        read_jsonl(_data_root(config) / "surfaces" / "selection_shard.jsonl")
    )
    selector_rows = _paired_selector_rows(selection_records, selection_surface)
    cases = list(read_jsonl(_data_root(config) / "selection-format.jsonl"))
    if [str(row["metadata"]["blueprint_id"]) for row in selection_records] != [
        str(case["case_id"]) for case in cases
    ]:
        raise RuntimeError("H5 v2 selector and menu-free selection case order differs")

    arms = ["selector-only-full-sequence", "multi-token-representation"]
    _, adapter_paths = _expert_context(config)
    evaluation_fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "amendment": amendment["fingerprint"],
            "data": data["fingerprint"],
            "arms": {arm: statuses[arm]["adapter_sha256"] for arm in arms},
            "surface": "selection",
            "format_evaluator_version": _FORMAT_EVALUATOR_VERSION,
        }
    )
    progress_root = _root(config) / "selection-progress"
    scores = {
        "v0.3-parent": _evaluate(
            config,
            name="parent",
            adapter_path=adapter_paths["parent"],
            selection_rows=selector_rows,
            cases=cases,
            progress_root=progress_root,
            evaluation_fingerprint=evaluation_fingerprint,
        ),
        arms[0]: _evaluate(
            config,
            name=arms[0],
            adapter_path=Path(statuses[arms[0]]["adapter_path"]),
            selection_rows=selector_rows,
            cases=cases,
            progress_root=progress_root,
            evaluation_fingerprint=evaluation_fingerprint,
        ),
        arms[1]: _evaluate(
            config,
            name=arms[1],
            adapter_path=Path(statuses[arms[1]]["adapter_path"]),
            selection_rows=selector_rows,
            cases=cases,
            progress_root=progress_root,
            evaluation_fingerprint=evaluation_fingerprint,
        ),
    }
    gate = _gate_report(
        parent=scores["v0.3-parent"],
        control=scores[arms[0]],
        candidate=scores[arms[1]],
        gates=dict(contract["settings"]["selection_gates"]),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": evaluation_fingerprint,
        "method": "H5 cross-format representation repair pilot",
        "contract_fingerprint": contract["fingerprint"],
        "evaluation_amendment_fingerprint": amendment["fingerprint"],
        "data_fingerprint": data["fingerprint"],
        "matched_compute": True,
        "fixed_endpoint": True,
        "scores": {
            name: {"selector": value["selector"], "format_shift": value["format_shift"]}
            for name, value in scores.items()
        },
        "selection_gate": gate,
        "selected_arm": "multi-token-representation" if gate["passed"] else None,
        "confirmation_authorized": bool(gate["passed"]),
        "external_benchmark_authorized": False,
        "private_details": {name: value["details"] for name, value in scores.items()},
    }
    report["result_fingerprint"] = canonical_hash(
        {key: value for key, value in report.items() if key != "private_details"}
    )
    write_json(_root(config) / "pilot.json", report)
    public = dict(report)
    public.pop("private_details")
    write_json(config.root / "reports" / "evolve" / "cross-format-repair-v2-pilot.json", public)
    return public


def run_cross_format_confirmation_amended(config: ProjectConfig) -> dict[str, Any]:
    contract = _frozen_contract(config)
    amendment = prepare_cross_format_evaluation_amendment(config)
    pilot_path = _root(config) / "pilot.json"
    if not pilot_path.exists():
        raise RuntimeError("H5 v2 pilot has not selected a candidate")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("selected_arm") != "multi-token-representation":
        raise RuntimeError("H5 v2 pilot did not authorize confirmation")
    confirmation_manifest, surface = _simulate_registered(
        config,
        contract,
        name="confirmation_shard",
    )
    selector_records = _records(surface, training=False)
    selector_rows = _paired_selector_rows(selector_records, surface)
    cases = [_format_shift_case(simulation) for simulation in surface]
    arms = ["selector-only-full-sequence", "multi-token-representation"]
    statuses = _arm_statuses(config)
    _, adapter_paths = _expert_context(config)
    evaluation_fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "amendment": amendment["fingerprint"],
            "pilot": pilot["result_fingerprint"],
            "confirmation": confirmation_manifest["fingerprint"],
            "arms": {arm: statuses[arm]["adapter_sha256"] for arm in arms},
            "format_evaluator_version": _FORMAT_EVALUATOR_VERSION,
        }
    )
    progress_root = _root(config) / "confirmation-progress"
    scores = {
        "v0.3-parent": _evaluate(
            config,
            name="parent",
            adapter_path=adapter_paths["parent"],
            selection_rows=selector_rows,
            cases=cases,
            progress_root=progress_root,
            evaluation_fingerprint=evaluation_fingerprint,
        ),
        arms[0]: _evaluate(
            config,
            name=arms[0],
            adapter_path=Path(statuses[arms[0]]["adapter_path"]),
            selection_rows=selector_rows,
            cases=cases,
            progress_root=progress_root,
            evaluation_fingerprint=evaluation_fingerprint,
        ),
        arms[1]: _evaluate(
            config,
            name=arms[1],
            adapter_path=Path(statuses[arms[1]]["adapter_path"]),
            selection_rows=selector_rows,
            cases=cases,
            progress_root=progress_root,
            evaluation_fingerprint=evaluation_fingerprint,
        ),
    }
    gate = _gate_report(
        parent=scores["v0.3-parent"],
        control=scores[arms[0]],
        candidate=scores[arms[1]],
        gates=dict(contract["settings"]["confirmation_gates"]),
    )
    report = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": evaluation_fingerprint,
        "method": "H5 cross-format representation repair confirmation",
        "contract_fingerprint": contract["fingerprint"],
        "evaluation_amendment_fingerprint": amendment["fingerprint"],
        "pilot_result_fingerprint": pilot["result_fingerprint"],
        "confirmation_manifest_fingerprint": confirmation_manifest["fingerprint"],
        "scores": {
            name: {"selector": value["selector"], "format_shift": value["format_shift"]}
            for name, value in scores.items()
        },
        "confirmation_gate": gate,
        "synthetic_mechanism_confirmed": bool(gate["passed"]),
        "external_benchmark_authorized": False,
        "next_step": (
            "preregister-independent-external-evidence"
            if gate["passed"]
            else "reject-h5-representation-repair"
        ),
        "private_details": {name: value["details"] for name, value in scores.items()},
    }
    write_json(_root(config) / "confirmation.json", report)
    public = dict(report)
    public.pop("private_details")
    write_json(
        config.root / "reports" / "evolve" / "cross-format-repair-v2-confirmation.json",
        public,
    )
    return public

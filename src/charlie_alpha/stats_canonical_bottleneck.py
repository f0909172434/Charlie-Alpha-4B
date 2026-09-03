from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingArgs, train

from .config import ProjectConfig
from .forge_training import ForgeDataset, forge_iterate_batches, forge_loss
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl
from .stats_catalog import PROCEDURE_BY_ID
from .stats_cross_format import (
    _FORMAT_EVALUATOR_VERSION,
    _balanced_scenarios,
    _evaluate,
    _format_shift_case,
    _gate_report,
    _records,
)
from .stats_dgp import Scenario, simulate_scenario
from .stats_evolve import _adapter_config_for_child, _start_caffeinate
from .stats_family_router import _expert_context
from .stats_router_replication import _historical_scenario_audit
from .stats_training import (
    _enable_gradient_checkpointing_once,
    _optimizer,
    _stats_snapshot,
    _StatsCallback,
)

_TRAINER_VERSION = 1


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "canonical-bottleneck-v1"


def _data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "canonical-bottleneck-v1"


def _implementation_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "stats_canonical_bottleneck.py": sha256_file(Path(__file__)),
        "stats_cross_format.py": sha256_file(root / "stats_cross_format.py"),
        "forge_training.py": sha256_file(root / "forge_training.py"),
        "stats_data.py": sha256_file(root / "stats_data.py"),
        "stats_agent.py": sha256_file(root / "stats_agent.py"),
    }


def _registered_scenarios(settings: dict[str, Any], *, name: str) -> list[Scenario]:
    return _balanced_scenarios(settings, name=name)


def _training_messages(case: dict[str, Any], *, canonical: bool) -> list[dict[str, str]]:
    if canonical:
        method_value = str(case["gold_methods"][0])
    else:
        method_id = str(case["gold_methods"][0])
        procedure = PROCEDURE_BY_ID.get(method_id)
        method_value = procedure.name if procedure is not None else method_id.replace("_", " ")
    target = json.dumps(
        {"methods": [method_value], "columns": list(case["gold_columns"])},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an auditable statistical analysis extractor. Infer one primary method "
                "and the columns needed for it. Return only one JSON object with exactly the keys "
                "methods and columns. Both values are arrays of strings."
            ),
        },
        {"role": "user", "content": str(case["question"])},
        {"role": "assistant", "content": target},
    ]


def _training_rows(
    simulations: list[dict[str, Any]],
    *,
    canonical: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for simulation in simulations:
        case = _format_shift_case(simulation)
        rows.append(
            {
                "messages": _training_messages(case, canonical=canonical),
                "metadata": {
                    "blueprint_id": case["case_id"],
                    "family_id": case["family_id"],
                    "target_style": "canonical-id" if canonical else "display-name",
                    "loss_weight": 1.0,
                },
            }
        )
    return rows


def prepare_canonical_bottleneck_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("canonical_bottleneck"))
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "contract.json"
    public_path = config.root / "reports" / "evolve" / "canonical-bottleneck-v1-contract.json"

    h5_path = config.root / "reports" / "evolve" / "cross-format-repair-v2-pilot.json"
    if not h5_path.exists():
        raise RuntimeError("H6 requires the closed H5 v2 pilot")
    h5 = json.loads(h5_path.read_text(encoding="utf-8"))
    if h5.get("selected_arm") is not None or h5.get("confirmation_authorized"):
        raise RuntimeError("H5 did not close negatively; H6 is not authorized")
    if h5["scores"]["multi-token-representation"]["format_shift"]["exact_accuracy"] != 0.0:
        raise RuntimeError("H6 diagnosis expected zero H5 menu-free exact accuracy")
    h5_confirmation = (
        _data_root(config).parent
        / "cross-format-repair-v2"
        / "surfaces"
        / "confirmation_shard.jsonl"
    )
    if h5_confirmation.exists():
        raise RuntimeError("H5 confirmation was unexpectedly opened")

    _, adapter_paths = _expert_context(config)
    parent = adapter_paths["parent"]
    parent_sha = sha256_file(parent / "adapters.safetensors")
    registered: dict[str, Any] = {}
    all_scenarios: list[Scenario] = []
    seen: set[str] = set()
    for name in ("training_pool", "selection_shard", "confirmation_shard"):
        shard = dict(settings[name])
        scenarios = _registered_scenarios(shard, name=name)
        ids = {scenario.blueprint_id for scenario in scenarios}
        if ids & seen:
            raise RuntimeError(f"H6 blueprint overlap at {name}")
        seen.update(ids)
        all_scenarios.extend(scenarios)
        registered[name] = {
            "split": str(shard["split"]),
            "seed": int(shard["seed"]),
            "pool_count": int(shard["pool_count"]),
            "selected_per_family": int(shard["selected_per_family"]),
            "count": len(scenarios),
            "blueprint_sha256": canonical_hash([scenario.to_dict() for scenario in scenarios]),
        }
    audit = _historical_scenario_audit(
        config,
        all_scenarios,
        excluded_root=_data_root(config),
        minimum_normalized_distance=float(settings["minimum_normalized_distance"]),
    )
    if not audit["passed"]:
        raise RuntimeError("H6 blueprints failed historical-overlap audit")
    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": "H6 canonical semantic bottleneck",
        "method_version": int(settings["method_version"]),
        "causal_question": (
            "Under an identical system instruction and matched compute, does changing only the "
            "assistant target from a human-readable procedure name to the canonical repository "
            "method ID produce held-out canonical method-and-column extraction without destroying "
            "the existing selector?"
        ),
        "h5_negative_result_fingerprint": h5["result_fingerprint"],
        "h5_report_sha256": sha256_file(h5_path),
        "parent": {
            "name": "v0.3.0-parent",
            "adapter_path": str(parent),
            "adapter_sha256": parent_sha,
        },
        "settings": settings,
        "blueprint_contracts": registered,
        "historical_overlap_audit": audit,
        "implementation_sha256": _implementation_manifest(),
        "arms": {
            "display-name-json": "Identical prompt; human-readable procedure name assistant target",
            "canonical-id-json": (
                "Identical prompt; repository canonical method ID assistant target"
            ),
        },
        "matched_compute": (
            "Same parent, scenario set, system/user prompt, one assistant JSON target per "
            "scenario, "
            "tokenizer, sequence limit, shuffle seed, optimizer, learning rates, 48 microsteps, "
            "and fixed endpoint. Only the method string in the assistant target differs."
        ),
        "selection_policy": (
            "Only canonical-id-json may advance and only if every registered selection gate passes."
        ),
        "confirmation_policy": (
            "Confirmation blueprints are registered now, but simulations remain unopened until "
            "selection passes. H5 confirmation and historical P-Bench/StatQA remain sealed from H6."
        ),
        "claim_boundary": (
            "H6 can establish a synthetic canonical extraction mechanism only. External capability "
            "requires separately preregistered independent evidence."
        ),
        "selection_opened": False,
        "confirmation_simulations_opened": False,
        "external_benchmark_opened": False,
    }
    contract["fingerprint"] = canonical_hash(contract)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("H6 contract is immutable")
        write_json(public_path, existing)
        return existing
    write_json(lock_path, contract)
    write_json(public_path, contract)
    return contract


def _simulate_surface(
    config: ProjectConfig,
    contract: dict[str, Any],
    *,
    name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if name == "confirmation_shard":
        pilot_path = _root(config) / "pilot.json"
        if not pilot_path.exists():
            raise RuntimeError("H6 confirmation cannot open before selection")
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        if pilot.get("selected_arm") != "canonical-id-json":
            raise RuntimeError("H6 selection did not authorize confirmation")
    shard = dict(contract["settings"][name])
    scenarios = _registered_scenarios(shard, name=name)
    if (
        canonical_hash([scenario.to_dict() for scenario in scenarios])
        != contract["blueprint_contracts"][name]["blueprint_sha256"]
    ):
        raise RuntimeError(f"H6 registered blueprints changed for {name}")
    path = _data_root(config) / "surfaces" / f"{name}.jsonl"
    manifest_path = path.with_suffix(".manifest.json")
    stats = config.section("stats_data")
    fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "name": name,
            "blueprints": contract["blueprint_contracts"][name]["blueprint_sha256"],
            "simulator": {
                "initial_repetitions": stats["initial_repetitions"],
                "escalation_repetitions": stats["escalation_repetitions"],
                "ranking_uncertainty_margin": stats["ranking_uncertainty_margin"],
                "regret_temperature": stats["regret_temperature"],
            },
        }
    )
    if path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint or manifest.get("sha256") != sha256_file(
            path
        ):
            raise RuntimeError(f"H6 {name} surface changed")
        return manifest, list(read_jsonl(path))
    simulations = [
        simulate_scenario(
            scenario,
            initial_repetitions=int(stats["initial_repetitions"]),
            escalation_repetitions=[int(value) for value in stats["escalation_repetitions"]],
            uncertainty_margin=float(stats["ranking_uncertainty_margin"]),
            temperature=float(stats["regret_temperature"]),
        )
        for scenario in scenarios
    ]
    write_jsonl(path, simulations)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "name": name,
        "count": len(simulations),
        "sha256": sha256_file(path),
    }
    write_json(manifest_path, manifest)
    return manifest, list(read_jsonl(path))


def prepare_canonical_bottleneck_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_canonical_bottleneck_contract(config)
    root = _data_root(config)
    status_path = root / "data-status.json"
    train_manifest, train_surface = _simulate_surface(config, contract, name="training_pool")
    selection_manifest, selection_surface = _simulate_surface(
        config, contract, name="selection_shard"
    )
    control_rows = _training_rows(train_surface, canonical=False)
    candidate_rows = _training_rows(train_surface, canonical=True)
    selector_rows = _records(selection_surface, training=False)
    cases = [_format_shift_case(simulation) for simulation in selection_surface]
    paths = {
        "display-name-json": root / "train-display-name.jsonl",
        "canonical-id-json": root / "train-canonical-id.jsonl",
        "selector": root / "selection-selector.jsonl",
        "cases": root / "selection-format.jsonl",
    }
    write_jsonl(paths["display-name-json"], control_rows)
    write_jsonl(paths["canonical-id-json"], candidate_rows)
    write_jsonl(paths["selector"], selector_rows)
    write_jsonl(paths["cases"], cases)
    file_sha = {name: sha256_file(path) for name, path in paths.items()}
    fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "training_surface": train_manifest["fingerprint"],
            "selection_surface": selection_manifest["fingerprint"],
            "files": file_sha,
        }
    )
    status = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "contract_fingerprint": contract["fingerprint"],
        "training_groups": len(train_surface),
        "training_records_per_arm": len(control_rows),
        "selection_groups": len(selection_surface),
        "file_sha256": file_sha,
        "confirmation_opened": False,
    }
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise RuntimeError("H6 prepared data changed")
        return existing
    write_json(status_path, status)
    write_json(config.root / "reports" / "evolve" / "canonical-bottleneck-v1-data.json", status)
    return status


def _train_arm(config: ProjectConfig, *, arm: str) -> dict[str, Any]:
    if arm not in {"display-name-json", "canonical-id-json"}:
        raise ValueError(f"Unknown H6 arm: {arm}")
    contract = prepare_canonical_bottleneck_contract(config)
    data = prepare_canonical_bottleneck_data(config)
    settings = dict(contract["settings"])
    output = _root(config) / "arms" / arm
    status_path = output / "status.json"
    train_path = _data_root(config) / (
        "train-canonical-id.jsonl" if arm == "canonical-id-json" else "train-display-name.jsonl"
    )
    fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "data": data["fingerprint"],
            "arm": arm,
            "train_sha256": sha256_file(train_path),
            "trainer_version": _TRAINER_VERSION,
        }
    )
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            return existing
        raise RuntimeError(f"H6 arm state changed: {arm}")

    _, adapter_paths = _expert_context(config)
    parent = adapter_paths["parent"]
    output.mkdir(parents=True, exist_ok=True)
    seed = int(settings["training_seed"])
    mx.random.seed(seed)
    np.random.seed(seed)
    previous_cache_limit = mx.set_cache_limit(
        int(float(settings["clear_cache_threshold_gb"]) * 1024**3)
    )
    caffeinate = _start_caffeinate()
    model = tokenizer = optimizer = dataset = None
    started = time.monotonic()
    try:
        model, tokenizer = load(
            _stats_snapshot(config),
            adapter_path=str(parent),
            tokenizer_config={"trust_remote_code": True},
        )
        model.freeze()
        model.unfreeze(keys=["lora_a", "lora_b"])
        trainable_parameters = sum(
            parameter.size for _, parameter in tree_flatten(model.trainable_parameters())
        )
        rows = list(read_jsonl(train_path))
        dataset = ForgeDataset(
            rows,
            tokenizer,
            group_size=1,
            seed=seed,
            grouped=True,
            selective_loss=False,
        )
        max_length = int(settings["max_seq_length"])
        maximum_tokens = max(len(item[0]) for item in dataset.items)
        if maximum_tokens > max_length:
            raise RuntimeError(f"H6 forbids truncation: {maximum_tokens} > {max_length}")
        microsteps = int(settings["microsteps"])
        optimizer = _optimizer(settings, microsteps)
        _enable_gradient_checkpointing_once(model)
        callback = _StatsCallback(
            model=model,
            best_path=output / "best_adapters.safetensors",
            deadline=started + int(settings["max_seconds"]),
            patience=None,
        )
        train(
            model=model,
            optimizer=optimizer,
            train_dataset=dataset,
            val_dataset=None,
            args=TrainingArgs(
                batch_size=1,
                iters=microsteps,
                val_batches=0,
                steps_per_report=int(settings["grad_accumulation_steps"]),
                steps_per_eval=microsteps,
                steps_per_save=microsteps,
                max_seq_length=max_length,
                adapter_file=str(output / "adapters.safetensors"),
                grad_checkpoint=False,
                grad_accumulation_steps=int(settings["grad_accumulation_steps"]),
                clear_cache_threshold=int(float(settings["clear_cache_threshold_gb"]) * 1024**3),
            ),
            loss=forge_loss,
            iterate_batches=forge_iterate_batches,
            training_callback=callback,
        )
        mx.save_safetensors(
            str(output / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters()))
        )
        adapter_config = _adapter_config_for_child(
            config, parent, output, cycle=10, arm=f"canonical-bottleneck:{arm}"
        )
        adapter_config.setdefault("stats", {}).update(
            {
                "method": "H6 canonical semantic bottleneck",
                "target_style": arm,
                "fixed_endpoint": True,
                "promotion_status": "development-only",
            }
        )
        write_json(output / "adapter_config.json", adapter_config)
        status = {
            "schema_version": 1,
            "complete": True,
            "fingerprint": fingerprint,
            "arm": arm,
            "contract_fingerprint": contract["fingerprint"],
            "data_fingerprint": data["fingerprint"],
            "adapter_path": str(output),
            "adapter_sha256": sha256_file(output / "adapters.safetensors"),
            "microsteps": microsteps,
            "optimizer_updates": microsteps // int(settings["grad_accumulation_steps"]),
            "training_records": len(rows),
            "maximum_tokens": maximum_tokens,
            "trainable_parameters": trainable_parameters,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 4),
            "selection_opened": False,
            "confirmation_opened": False,
        }
        write_json(status_path, status)
        return status
    finally:
        for value in (model, tokenizer, optimizer, dataset):
            if value is not None:
                del value
        gc.collect()
        mx.clear_cache()
        mx.set_cache_limit(previous_cache_limit)
        if caffeinate is not None and caffeinate.poll() is None:
            caffeinate.terminate()


def _paired_selector_rows(
    records: list[dict[str, Any]],
    simulations: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if len(records) != len(simulations):
        raise RuntimeError("H6 selector coverage differs")
    pairs = list(zip(records, simulations, strict=True))
    for record, simulation in pairs:
        if str(record["metadata"]["blueprint_id"]) != str(simulation["scenario"]["blueprint_id"]):
            raise RuntimeError("H6 selector blueprint pairing changed")
    return pairs


def run_canonical_bottleneck_pilot(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_canonical_bottleneck_contract(config)
    data = prepare_canonical_bottleneck_data(config)
    arms = ["display-name-json", "canonical-id-json"]
    statuses = {arm: _train_arm(config, arm=arm) for arm in arms}
    if {status["microsteps"] for status in statuses.values()} != {
        contract["settings"]["microsteps"]
    }:
        raise RuntimeError("H6 arms did not receive matched compute")
    _, selection_surface = _simulate_surface(config, contract, name="selection_shard")
    selector_records = list(read_jsonl(_data_root(config) / "selection-selector.jsonl"))
    selector_rows = _paired_selector_rows(selector_records, selection_surface)
    cases = list(read_jsonl(_data_root(config) / "selection-format.jsonl"))
    _, adapter_paths = _expert_context(config)
    evaluation_fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
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
        "method": "H6 canonical semantic bottleneck pilot",
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["fingerprint"],
        "matched_compute": True,
        "scores": {
            name: {"selector": value["selector"], "format_shift": value["format_shift"]}
            for name, value in scores.items()
        },
        "selection_gate": gate,
        "selected_arm": "canonical-id-json" if gate["passed"] else None,
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
    write_json(config.root / "reports" / "evolve" / "canonical-bottleneck-v1-pilot.json", public)
    return public


def run_canonical_bottleneck_confirmation(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_canonical_bottleneck_contract(config)
    pilot_path = _root(config) / "pilot.json"
    if not pilot_path.exists():
        raise RuntimeError("H6 pilot has not selected a candidate")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("selected_arm") != "canonical-id-json":
        raise RuntimeError("H6 selection did not authorize confirmation")
    manifest, surface = _simulate_surface(config, contract, name="confirmation_shard")
    selector_records = _records(surface, training=False)
    selector_rows = _paired_selector_rows(selector_records, surface)
    cases = [_format_shift_case(simulation) for simulation in surface]
    arms = ["display-name-json", "canonical-id-json"]
    statuses = {
        arm: json.loads((_root(config) / "arms" / arm / "status.json").read_text(encoding="utf-8"))
        for arm in arms
    }
    _, adapter_paths = _expert_context(config)
    evaluation_fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "pilot": pilot["result_fingerprint"],
            "confirmation": manifest["fingerprint"],
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
        "method": "H6 canonical semantic bottleneck confirmation",
        "contract_fingerprint": contract["fingerprint"],
        "pilot_result_fingerprint": pilot["result_fingerprint"],
        "confirmation_manifest_fingerprint": manifest["fingerprint"],
        "scores": {
            name: {"selector": value["selector"], "format_shift": value["format_shift"]}
            for name, value in scores.items()
        },
        "confirmation_gate": gate,
        "synthetic_mechanism_confirmed": bool(gate["passed"]),
        "external_benchmark_authorized": False,
        "next_step": "preregister-independent-external-evidence"
        if gate["passed"]
        else "reject-h6-canonical-bottleneck",
        "private_details": {name: value["details"] for name, value in scores.items()},
    }
    write_json(_root(config) / "confirmation.json", report)
    public = dict(report)
    public.pop("private_details")
    write_json(
        config.root / "reports" / "evolve" / "canonical-bottleneck-v1-confirmation.json", public
    )
    return public

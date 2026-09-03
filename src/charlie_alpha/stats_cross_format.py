from __future__ import annotations

import gc
import json
import time
from functools import partial
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingArgs, train

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl
from .stats_agent import StatsAgent
from .stats_catalog import FAMILIES
from .stats_data import _analysis_plan, _build_record, _render_question, _scenario
from .stats_dgp import Scenario, build_blueprints, simulate_scenario
from .stats_eval import _append_progress, _json_from_answer, _load_progress, _normalize
from .stats_evolve import _adapter_config_for_child, _selector_summary, _start_caffeinate
from .stats_family_router import _expert_context
from .stats_router_replication import _historical_scenario_audit
from .stats_training import (
    StatsDataset,
    _enable_gradient_checkpointing_once,
    _optimizer,
    _score_loaded_selector,
    _stats_snapshot,
    _StatsCallback,
    _StopTraining,
    stats_iterate_batches,
    stats_loss,
)

_TRAINER_VERSION = 1
_FORMAT_EVALUATOR_VERSION = 1


def _root(config: ProjectConfig) -> Path:
    version = int(config.section("cross_format_repair")["method_version"])
    return config.path_for("artifact_dir") / f"cross-format-repair-v{version}"


def _data_root(config: ProjectConfig) -> Path:
    version = int(config.section("cross_format_repair")["method_version"])
    return config.path_for("evolution_dir") / f"cross-format-repair-v{version}"


def _balanced_scenarios(settings: dict[str, Any], *, name: str) -> list[Scenario]:
    scenarios = build_blueprints(
        {str(settings["split"]): int(settings["pool_count"])},
        seed=int(settings["seed"]),
        active_search=True,
    )
    by_family: dict[str, list[Scenario]] = {family.family_id: [] for family in FAMILIES}
    for scenario in scenarios:
        by_family[scenario.family_id].append(scenario)
    selected: list[Scenario] = []
    for family_id in sorted(by_family):
        available = sorted(
            by_family[family_id],
            key=lambda scenario: canonical_hash(
                {"cross_format_shard": name, "blueprint": scenario.blueprint_id}
            ),
        )
        count = int(settings["selected_per_family"])
        if len(available) < count:
            raise RuntimeError(
                f"Cross-format {name} has {len(available)} {family_id} rows below {count}"
            )
        selected.extend(available[:count])
    return selected


def _implementation_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "stats_cross_format.py": sha256_file(Path(__file__)),
        "stats_training.py": sha256_file(root / "stats_training.py"),
        "stats_data.py": sha256_file(root / "stats_data.py"),
        "stats_agent.py": sha256_file(root / "stats_agent.py"),
    }


def _flatten_columns(variables: dict[str, Any]) -> list[str]:
    columns: set[str] = set()
    for value in variables.values():
        if isinstance(value, str):
            columns.add(value)
        elif isinstance(value, list):
            columns.update(str(item) for item in value if isinstance(item, str))
    return sorted(columns)


def _format_shift_case(simulation: dict[str, Any]) -> dict[str, Any]:
    scenario = _scenario(simulation["scenario"])
    method_id = str(simulation["selected_method_id"])
    plan = _analysis_plan(scenario, method_id, incomplete=False)
    question = _render_question(scenario, "en", incomplete=False, view="standard")
    return {
        "case_id": scenario.blueprint_id,
        "family_id": scenario.family_id,
        "question": question,
        "gold_methods": [method_id],
        "gold_columns": _flatten_columns(dict(plan["variables"])),
    }


def _format_shift_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an auditable statistical analysis extractor. Infer the primary method "
                "and the data columns needed for it. Return only one JSON object with exactly "
                "two keys: methods and columns. Both values must be JSON arrays of strings. "
                "Use the repository-style method identifier, not prose, and do not add reasoning."
            ),
        },
        {"role": "user", "content": str(case["question"])},
    ]


def _set_match(expected: list[str], predicted: Any) -> bool:
    if not isinstance(predicted, list):
        return False
    return {_normalize(str(value)) for value in predicted} == {
        _normalize(str(value)) for value in expected
    }


def _column_recall(expected: list[str], predicted: Any) -> float:
    if not expected:
        return 1.0
    if not isinstance(predicted, list):
        return 0.0
    gold = {_normalize(value) for value in expected}
    observed = {_normalize(str(value)) for value in predicted}
    return len(gold & observed) / len(gold)


def _format_metrics(details: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(details)
    if not count:
        raise RuntimeError("Cross-format evaluation has no rows")
    return {
        "count": count,
        "exact_accuracy": sum(bool(row["exact_correct"]) for row in details) / count,
        "method_set_accuracy": sum(bool(row["method_correct"]) for row in details) / count,
        "column_set_accuracy": sum(bool(row["columns_correct"]) for row in details) / count,
        "mean_column_recall": float(np.mean([float(row["column_recall"]) for row in details])),
    }


def prepare_cross_format_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("cross_format_repair"))
    method_version = int(settings["method_version"])
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "contract.json"
    public_path = (
        config.root
        / "reports"
        / "evolve"
        / f"cross-format-repair-v{method_version}-contract.json"
    )

    matched_path = (
        config.root / "reports" / "evolve" / "router-historical-counterfactual-replay.json"
    )
    if not matched_path.exists():
        raise RuntimeError("H5 requires the completed post-H4 matched replay")
    matched = json.loads(matched_path.read_text(encoding="utf-8"))
    if (
        not matched.get("complete")
        or matched.get("next_research_direction") != "representation-transfer"
    ):
        raise RuntimeError("Matched replay did not authorize the representation-transfer direction")
    if matched.get("h4_terminal_status_unchanged") != "external-rejected":
        raise RuntimeError("H4 terminal status changed unexpectedly")

    _, adapter_paths = _expert_context(config)
    parent_path = adapter_paths["parent"]
    parent_sha = sha256_file(parent_path / "adapters.safetensors")
    h4_contract = json.loads(
        (config.root / "reports" / "evolve" / "router-historical-external-contract.json").read_text(
            encoding="utf-8"
        )
    )
    if parent_sha != h4_contract["control"]["adapter_sha256"]:
        raise RuntimeError("H5 parent is not the frozen v0.3 control")

    shard_settings = {
        "training_pool": dict(settings["training_pool"]),
        "selection_shard": dict(settings["selection_shard"]),
        "confirmation_shard": dict(settings["confirmation_shard"]),
    }
    registered: dict[str, Any] = {}
    all_scenarios: list[Scenario] = []
    seen_ids: set[str] = set()
    for name, shard in shard_settings.items():
        scenarios = _balanced_scenarios(shard, name=name)
        ids = {scenario.blueprint_id for scenario in scenarios}
        overlap = ids & seen_ids
        if overlap:
            raise RuntimeError(f"Cross-format blueprint overlap at {name}: {sorted(overlap)[:1]}")
        seen_ids.update(ids)
        all_scenarios.extend(scenarios)
        registered[name] = {
            "split": str(shard["split"]),
            "seed": int(shard["seed"]),
            "pool_count": int(shard["pool_count"]),
            "selected_per_family": int(shard["selected_per_family"]),
            "count": len(scenarios),
            "blueprint_sha256": canonical_hash([scenario.to_dict() for scenario in scenarios]),
            "family_counts": {
                family.family_id: sum(
                    scenario.family_id == family.family_id for scenario in scenarios
                )
                for family in FAMILIES
            },
        }
    audit = _historical_scenario_audit(
        config,
        all_scenarios,
        excluded_root=_data_root(config),
        minimum_normalized_distance=float(settings["minimum_normalized_distance"]),
    )
    if not audit["passed"]:
        raise RuntimeError(
            "Cross-format blueprints failed the preregistered historical-overlap audit"
        )

    supersedes: dict[str, Any] | None = None
    if method_version > 1:
        prior_contract_path = (
            config.path_for("artifact_dir")
            / f"cross-format-repair-v{method_version - 1}"
            / "contract.json"
        )
        prior_status_path = (
            config.root
            / "reports"
            / "evolve"
            / f"cross-format-repair-v{method_version - 1}-superseded.json"
        )
        if not prior_contract_path.exists() or not prior_status_path.exists():
            raise RuntimeError("H5 v2 requires the frozen v1 contract and supersession receipt")
        prior_contract = json.loads(prior_contract_path.read_text(encoding="utf-8"))
        prior_status = json.loads(prior_status_path.read_text(encoding="utf-8"))
        if prior_status.get("research_decision_made"):
            raise RuntimeError("A prior H5 result exists; version rollover is not allowed")
        supersedes = {
            "method_version": method_version - 1,
            "contract_fingerprint": prior_contract["fingerprint"],
            "status_sha256": sha256_file(prior_status_path),
            "reason": str(prior_status["reason"]),
            "training_microsteps_completed": int(prior_status["training_microsteps_completed"]),
            "confirmation_opened": bool(prior_status["confirmation_opened"]),
        }

    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": "H5 cross-format representation repair",
        "method_version": method_version,
        "causal_question": (
            "Does restoring gradient support on plan/tool/report tokens create cross-format "
            "method-and-column extraction gains that selector-only continuation cannot produce?"
        ),
        "matched_replay_fingerprint": matched["fingerprint"],
        "matched_replay_sha256": sha256_file(matched_path),
        "parent": {
            "name": "v0.3.0-parent",
            "adapter_path": str(parent_path),
            "adapter_sha256": parent_sha,
        },
        "settings": settings,
        "blueprint_contracts": registered,
        "historical_overlap_audit": audit,
        "implementation_sha256": _implementation_manifest(),
        "supersedes": supersedes,
        "arms": {
            "selector-only-full-sequence": {
                "full_sequence": True,
                "component_weights": dict(settings["arms"]["selector-only-full-sequence"]),
            },
            "multi-token-representation": {
                "full_sequence": True,
                "component_weights": dict(settings["arms"]["multi-token-representation"]),
            },
        },
        "matched_compute": (
            "Same parent, training records, full sequence length, shuffle seed, microsteps, "
            "gradient accumulation, optimizer, learning rates, and checkpoint endpoint."
        ),
        "format_shift": (
            "Selection and confirmation remove the candidate menu and request a JSON object with "
            "methods and columns; gold is deterministically compiled from the registered DGP plan."
        ),
        "selection_policy": (
            "Only multi-token-representation can advance, and only if every registered selection "
            "gate passes. The endpoint is the fixed final microstep; no checkpoint shopping."
        ),
        "confirmation_policy": (
            "The confirmation surface must remain unsimulated until selection passes. Historical "
            "P-Bench and StatQA may not be used for H5 tuning or selection."
        ),
        "claim_boundary": (
            "H5 can establish a project-internal synthetic cross-format mechanism only. Any "
            "external capability claim requires a separately preregistered independent benchmark."
        ),
        "training_started_at_registration": False,
        "selection_opened": False,
        "confirmation_simulations_opened": False,
        "external_benchmark_opened": False,
    }
    contract["fingerprint"] = canonical_hash(contract)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("The H5 cross-format contract is immutable")
        write_json(public_path, existing)
        return existing
    write_json(lock_path, contract)
    write_json(public_path, contract)
    return contract


def _simulate_registered(
    config: ProjectConfig,
    contract: dict[str, Any],
    *,
    name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if name not in {"training_pool", "selection_shard", "confirmation_shard"}:
        raise ValueError(f"Unknown cross-format shard: {name}")
    if name == "confirmation_shard":
        pilot_path = _root(config) / "pilot.json"
        if not pilot_path.exists():
            raise RuntimeError("H5 confirmation cannot open before pilot selection")
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        if pilot.get("selected_arm") != "multi-token-representation":
            raise RuntimeError("H5 selection did not authorize confirmation")
    settings = dict(contract["settings"][name])
    scenarios = _balanced_scenarios(settings, name=name)
    registered = contract["blueprint_contracts"][name]
    blueprint_sha = canonical_hash([scenario.to_dict() for scenario in scenarios])
    if blueprint_sha != registered["blueprint_sha256"]:
        raise RuntimeError(f"Registered H5 blueprint hash changed for {name}")
    surface_path = _data_root(config) / "surfaces" / f"{name}.jsonl"
    manifest_path = surface_path.with_suffix(".manifest.json")
    simulation_settings = config.section("stats_data")
    fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "name": name,
            "blueprints": blueprint_sha,
            "simulation": {
                key: simulation_settings[key]
                for key in (
                    "initial_repetitions",
                    "escalation_repetitions",
                    "ranking_uncertainty_margin",
                    "regret_temperature",
                )
            },
            "simulator_version": 1,
        }
    )
    if surface_path.exists() or manifest_path.exists():
        if not surface_path.exists() or not manifest_path.exists():
            raise RuntimeError(f"H5 {name} surface is incomplete")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint or existing.get("sha256") != sha256_file(
            surface_path
        ):
            raise RuntimeError(f"H5 {name} surface is immutable")
        return existing, list(read_jsonl(surface_path))

    simulations = [
        simulate_scenario(
            scenario,
            initial_repetitions=int(simulation_settings["initial_repetitions"]),
            escalation_repetitions=[
                int(value) for value in simulation_settings["escalation_repetitions"]
            ],
            uncertainty_margin=float(simulation_settings["ranking_uncertainty_margin"]),
            temperature=float(simulation_settings["regret_temperature"]),
        )
        for scenario in scenarios
    ]
    write_jsonl(surface_path, simulations)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "name": name,
        "count": len(simulations),
        "blueprint_sha256": blueprint_sha,
        "sha256": sha256_file(surface_path),
        "used_for_training": name == "training_pool",
        "used_for_selection": name == "selection_shard",
        "confirmation": name == "confirmation_shard",
    }
    write_json(manifest_path, manifest)
    # All downstream rendering is defined from the canonical persisted JSONL,
    # not from in-memory dict insertion order. This makes first-run and resumed
    # rendering byte-identical even though write_jsonl recursively sorts keys.
    return manifest, list(read_jsonl(surface_path))


def _records(simulations: list[dict[str, Any]], *, training: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for simulation in simulations:
        scenario = _scenario(simulation["scenario"])
        recipe = (
            (
                ("en", 1.4, "boundary_a"),
                ("en", 1.4, "boundary_b"),
                ("zh_Hant", 0.6, "standard"),
                ("zh_Hans", 0.6, "standard"),
            )
            if training
            else (("en", 1.0, "standard"),)
        )
        for language, weight, view in recipe:
            records.append(
                _build_record(
                    scenario,
                    simulation,
                    language=language,
                    loss_weight=weight,
                    incomplete=False,
                    variant="dgp-regret",
                    refined_explanation=None,
                    view=view,
                )
            )
    return records


def prepare_cross_format_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_cross_format_contract(config)
    method_version = int(contract["method_version"])
    data_root = _data_root(config)
    status_path = data_root / "data-status.json"
    training_manifest, training_surface = _simulate_registered(
        config, contract, name="training_pool"
    )
    selection_manifest, selection_surface = _simulate_registered(
        config, contract, name="selection_shard"
    )
    train_rows = _records(training_surface, training=True)
    selection_rows = _records(selection_surface, training=False)
    format_cases = [_format_shift_case(simulation) for simulation in selection_surface]
    train_path = data_root / "train.jsonl"
    selection_path = data_root / "selection.jsonl"
    format_path = data_root / "selection-format.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(selection_path, selection_rows)
    write_jsonl(format_path, format_cases)
    fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "training_surface": training_manifest["fingerprint"],
            "selection_surface": selection_manifest["fingerprint"],
            "train": sha256_file(train_path),
            "selection": sha256_file(selection_path),
            "format": sha256_file(format_path),
            "renderer_version": 1,
        }
    )
    status = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "contract_fingerprint": contract["fingerprint"],
        "training_groups": len(training_surface),
        "training_records": len(train_rows),
        "selection_groups": len(selection_surface),
        "selection_records": len(selection_rows),
        "format_cases": len(format_cases),
        "train_sha256": sha256_file(train_path),
        "selection_sha256": sha256_file(selection_path),
        "format_sha256": sha256_file(format_path),
        "confirmation_opened": False,
    }
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise RuntimeError("H5 prepared data changed")
        return existing
    write_json(status_path, status)
    write_json(
        config.root
        / "reports"
        / "evolve"
        / f"cross-format-repair-v{method_version}-data.json",
        status,
    )
    return status


def _train_arm(config: ProjectConfig, *, arm: str) -> dict[str, Any]:
    settings = dict(config.section("cross_format_repair"))
    if arm not in settings["arms"]:
        raise ValueError(f"Unknown H5 arm: {arm}")
    contract = prepare_cross_format_contract(config)
    data = prepare_cross_format_data(config)
    artifact_dir = _root(config) / "arms" / arm
    status_path = artifact_dir / "status.json"
    component_weights = {key: float(value) for key, value in settings["arms"][arm].items()}
    fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "data": data["fingerprint"],
            "arm": arm,
            "component_weights": component_weights,
            "training_seed": int(settings["training_seed"]),
            "trainer_version": _TRAINER_VERSION,
        }
    )
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            return existing
        raise RuntimeError(f"H5 arm state changed: {arm}")

    _, adapter_paths = _expert_context(config)
    parent = adapter_paths["parent"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    seed = int(settings["training_seed"])
    mx.random.seed(seed)
    np.random.seed(seed)
    previous_cache_limit = mx.set_cache_limit(
        int(float(settings["clear_cache_threshold_gb"]) * 1024**3)
    )
    caffeinate = _start_caffeinate()
    model = tokenizer = optimizer = train_dataset = valid_dataset = None
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
        if trainable_parameters <= 0:
            raise RuntimeError("H5 parent exposes no trainable LoRA parameters")
        train_rows = list(read_jsonl(_data_root(config) / "train.jsonl"))
        valid_rows = list(read_jsonl(_data_root(config) / "selection.jsonl"))
        max_length = int(settings["max_seq_length"])
        train_dataset = StatsDataset(
            train_rows,
            tokenizer,
            seed=seed,
            grouped=True,
            curriculum="random",
            max_seq_length=max_length,
            selector_only=False,
        )
        valid_dataset = StatsDataset(
            valid_rows,
            tokenizer,
            seed=seed,
            grouped=False,
            curriculum="random",
            max_seq_length=max_length,
            selector_only=False,
        )
        microsteps = int(settings["microsteps"])
        group_size = int(settings["grad_accumulation_steps"])
        optimizer = _optimizer(
            {
                "grad_accumulation_steps": group_size,
                "learning_rate_a": float(settings["learning_rate_a"]),
                "learning_rate_b": float(settings["learning_rate_b"]),
                "warmup_fraction": float(settings["warmup_fraction"]),
                "weight_decay": float(settings["weight_decay"]),
            },
            microsteps,
        )
        args = TrainingArgs(
            batch_size=1,
            iters=microsteps,
            val_batches=-1,
            steps_per_report=group_size,
            steps_per_eval=int(settings["validation_every"]),
            steps_per_save=int(settings["checkpoint_every"]),
            max_seq_length=max_length,
            adapter_file=str(artifact_dir / "adapters.safetensors"),
            grad_checkpoint=False,
            grad_accumulation_steps=group_size,
            clear_cache_threshold=int(float(settings["clear_cache_threshold_gb"]) * 1024**3),
        )
        loss = partial(stats_loss, component_weights=component_weights)
        _enable_gradient_checkpointing_once(model)
        callback = _StatsCallback(
            model=model,
            best_path=artifact_dir / "best_adapters.safetensors",
            deadline=started + int(settings["max_seconds"]),
            patience=None,
        )
        try:
            train(
                model=model,
                optimizer=optimizer,
                train_dataset=train_dataset,
                val_dataset=valid_dataset,
                args=args,
                loss=loss,
                iterate_batches=stats_iterate_batches,
                training_callback=callback,
            )
        except _StopTraining as exc:
            raise RuntimeError(f"H5 matched arm stopped before fixed compute: {arm}") from exc
        mx.save_safetensors(
            str(artifact_dir / "adapters.safetensors"),
            dict(tree_flatten(model.trainable_parameters())),
        )
        adapter_config = _adapter_config_for_child(
            config,
            parent,
            artifact_dir,
            cycle=9,
            arm=f"cross-format:{arm}",
        )
        adapter_config.setdefault("stats", {}).update(
            {
                "method": "H5 cross-format representation repair",
                "component_weights": component_weights,
                "full_sequence_training": True,
                "fixed_endpoint": True,
                "promotion_status": "development-only",
            }
        )
        write_json(artifact_dir / "adapter_config.json", adapter_config)
        status = {
            "schema_version": 1,
            "complete": True,
            "fingerprint": fingerprint,
            "arm": arm,
            "contract_fingerprint": contract["fingerprint"],
            "data_fingerprint": data["fingerprint"],
            "parent_adapter_sha256": sha256_file(parent / "adapters.safetensors"),
            "adapter_path": str(artifact_dir),
            "adapter_sha256": sha256_file(artifact_dir / "adapters.safetensors"),
            "component_weights": component_weights,
            "full_sequence_training": True,
            "microsteps": microsteps,
            "optimizer_updates": microsteps // group_size,
            "training_records": len(train_rows),
            "trainable_parameters": trainable_parameters,
            "training_seed": seed,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 4),
            "selection_opened": False,
            "confirmation_opened": False,
        }
        write_json(status_path, status)
        return status
    finally:
        for value in (model, tokenizer, optimizer, train_dataset, valid_dataset):
            if value is not None:
                del value
        gc.collect()
        mx.clear_cache()
        mx.set_cache_limit(previous_cache_limit)
        if caffeinate is not None and caffeinate.poll() is None:
            caffeinate.terminate()


def run_cross_format_arm(config: ProjectConfig, *, arm: str) -> dict[str, Any]:
    return _train_arm(config, arm=arm)


def _evaluate(
    config: ProjectConfig,
    *,
    name: str,
    adapter_path: Path,
    selection_rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    progress_root: Path,
    evaluation_fingerprint: str,
) -> dict[str, Any]:
    agent = StatsAgent(config, adapter_path=adapter_path)
    agent.router.set_route("adapter")
    selector = _selector_summary(
        _score_loaded_selector(agent.model, agent.tokenizer, selection_rows)
    )
    progress_path = progress_root / f"{name}.jsonl"
    fingerprint = canonical_hash(
        {
            "evaluation": evaluation_fingerprint,
            "name": name,
            "adapter": sha256_file(adapter_path / "adapters.safetensors"),
            "format_evaluator_version": _FORMAT_EVALUATOR_VERSION,
        }
    )
    cached = _load_progress(
        progress_path,
        fingerprint=fingerprint,
        id_field="case_id",
    )
    details: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        if case_id in cached:
            details.append(cached[case_id])
            continue
        answer = agent.answer_without_tools(
            _format_shift_messages(case),
            route="stats",
            max_tokens=160,
            temperature=0.0,
        )
        parsed = _json_from_answer(answer)
        method_correct = _set_match(list(case["gold_methods"]), parsed.get("methods"))
        columns_correct = _set_match(list(case["gold_columns"]), parsed.get("columns"))
        row = {
            "case_id": case_id,
            "family_id": str(case["family_id"]),
            "exact_correct": method_correct and columns_correct,
            "method_correct": method_correct,
            "columns_correct": columns_correct,
            "column_recall": _column_recall(list(case["gold_columns"]), parsed.get("columns")),
            "predicted_methods": parsed.get("methods", []),
            "predicted_columns": parsed.get("columns", []),
        }
        details.append(row)
        _append_progress(
            progress_path,
            fingerprint=fingerprint,
            row=row,
            completed=len(details),
        )
    result = {
        "selector": selector,
        "format_shift": _format_metrics(details),
        "details": details,
    }
    del agent
    gc.collect()
    mx.clear_cache()
    return result


def _gate_report(
    *,
    parent: dict[str, Any],
    control: dict[str, Any],
    candidate: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    p_format = parent["format_shift"]
    c_format = control["format_shift"]
    x_format = candidate["format_shift"]
    p_selector = parent["selector"]
    x_selector = candidate["selector"]
    parent_regret = float(p_selector["normalized_regret"])
    regret_increase = (
        (float(x_selector["normalized_regret"]) - parent_regret) / parent_regret
        if parent_regret > 0
        else float(x_selector["normalized_regret"]) - parent_regret
    )
    checks = {
        "exact_gain_over_parent": 100
        * (float(x_format["exact_accuracy"]) - float(p_format["exact_accuracy"]))
        >= float(gates["minimum_exact_gain_over_parent_points"]),
        "exact_gain_over_control": 100
        * (float(x_format["exact_accuracy"]) - float(c_format["exact_accuracy"]))
        >= float(gates["minimum_exact_gain_over_control_points"]),
        "method_gain_over_control": 100
        * (float(x_format["method_set_accuracy"]) - float(c_format["method_set_accuracy"]))
        >= float(gates["minimum_method_gain_over_control_points"]),
        "column_gain_over_control": 100
        * (float(x_format["column_set_accuracy"]) - float(c_format["column_set_accuracy"]))
        >= float(gates["minimum_column_gain_over_control_points"]),
        "selector_regret_noninferior": regret_increase
        <= float(gates["maximum_selector_relative_regret_increase"]),
        "selector_accuracy_noninferior": 100
        * (float(p_selector["accuracy"]) - float(x_selector["accuracy"]))
        <= float(gates["maximum_selector_accuracy_regression_points"]),
        "selector_invalidity_noninferior": float(x_selector["invalid_selection_rate"])
        - float(p_selector["invalid_selection_rate"])
        <= float(gates["maximum_invalidity_increase"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "effect_points": {
            "exact_vs_parent": 100
            * (float(x_format["exact_accuracy"]) - float(p_format["exact_accuracy"])),
            "exact_vs_control": 100
            * (float(x_format["exact_accuracy"]) - float(c_format["exact_accuracy"])),
            "method_vs_control": 100
            * (float(x_format["method_set_accuracy"]) - float(c_format["method_set_accuracy"])),
            "columns_vs_control": 100
            * (float(x_format["column_set_accuracy"]) - float(c_format["column_set_accuracy"])),
        },
        "selector_relative_regret_increase": regret_increase,
    }


def run_cross_format_pilot(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_cross_format_contract(config)
    method_version = int(contract["method_version"])
    data = prepare_cross_format_data(config)
    arms = ["selector-only-full-sequence", "multi-token-representation"]
    statuses = {arm: _train_arm(config, arm=arm) for arm in arms}
    if {int(status["microsteps"]) for status in statuses.values()} != {
        int(contract["settings"]["microsteps"])
    }:
        raise RuntimeError("H5 arms did not receive matched fixed compute")
    if {int(status["training_records"]) for status in statuses.values()} != {
        data["training_records"]
    }:
        raise RuntimeError("H5 arms did not receive matched training records")

    selection_rows = list(read_jsonl(_data_root(config) / "selection.jsonl"))
    cases = list(read_jsonl(_data_root(config) / "selection-format.jsonl"))
    _, adapter_paths = _expert_context(config)
    evaluation_fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "data": data["fingerprint"],
            "arms": {arm: statuses[arm]["adapter_sha256"] for arm in arms},
            "surface": "selection",
            "evaluator_version": _FORMAT_EVALUATOR_VERSION,
        }
    )
    progress_root = _root(config) / "selection-progress"
    scores = {
        "v0.3-parent": _evaluate(
            config,
            name="parent",
            adapter_path=adapter_paths["parent"],
            selection_rows=selection_rows,
            cases=cases,
            progress_root=progress_root,
            evaluation_fingerprint=evaluation_fingerprint,
        ),
        arms[0]: _evaluate(
            config,
            name=arms[0],
            adapter_path=Path(statuses[arms[0]]["adapter_path"]),
            selection_rows=selection_rows,
            cases=cases,
            progress_root=progress_root,
            evaluation_fingerprint=evaluation_fingerprint,
        ),
        arms[1]: _evaluate(
            config,
            name=arms[1],
            adapter_path=Path(statuses[arms[1]]["adapter_path"]),
            selection_rows=selection_rows,
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
    report = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": evaluation_fingerprint,
        "method": "H5 cross-format representation repair pilot",
        "contract_fingerprint": contract["fingerprint"],
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
    path = _root(config) / "pilot.json"
    write_json(path, report)
    public = dict(report)
    public.pop("private_details")
    write_json(
        config.root
        / "reports"
        / "evolve"
        / f"cross-format-repair-v{method_version}-pilot.json",
        public,
    )
    return public


def run_cross_format_confirmation(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_cross_format_contract(config)
    method_version = int(contract["method_version"])
    pilot_path = _root(config) / "pilot.json"
    if not pilot_path.exists():
        raise RuntimeError("H5 pilot has not selected a candidate")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("selected_arm") != "multi-token-representation":
        raise RuntimeError("H5 pilot did not authorize confirmation")
    confirmation_manifest, surface = _simulate_registered(
        config, contract, name="confirmation_shard"
    )
    selection_rows = _records(surface, training=False)
    cases = [_format_shift_case(simulation) for simulation in surface]
    arms = ["selector-only-full-sequence", "multi-token-representation"]
    statuses = {
        arm: json.loads((_root(config) / "arms" / arm / "status.json").read_text(encoding="utf-8"))
        for arm in arms
    }
    _, adapter_paths = _expert_context(config)
    evaluation_fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "pilot": pilot["result_fingerprint"],
            "confirmation": confirmation_manifest["fingerprint"],
            "arms": {arm: statuses[arm]["adapter_sha256"] for arm in arms},
            "evaluator_version": _FORMAT_EVALUATOR_VERSION,
        }
    )
    progress_root = _root(config) / "confirmation-progress"
    scores = {
        "v0.3-parent": _evaluate(
            config,
            name="parent",
            adapter_path=adapter_paths["parent"],
            selection_rows=selection_rows,
            cases=cases,
            progress_root=progress_root,
            evaluation_fingerprint=evaluation_fingerprint,
        ),
        arms[0]: _evaluate(
            config,
            name=arms[0],
            adapter_path=Path(statuses[arms[0]]["adapter_path"]),
            selection_rows=selection_rows,
            cases=cases,
            progress_root=progress_root,
            evaluation_fingerprint=evaluation_fingerprint,
        ),
        arms[1]: _evaluate(
            config,
            name=arms[1],
            adapter_path=Path(statuses[arms[1]]["adapter_path"]),
            selection_rows=selection_rows,
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
        config.root
        / "reports"
        / "evolve"
        / f"cross-format-repair-v{method_version}-confirmation.json",
        public,
    )
    return public

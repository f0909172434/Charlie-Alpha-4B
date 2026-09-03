from __future__ import annotations

import gc
import json
import time
from collections import defaultdict
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
from .stats_canonical_bottleneck import _registered_scenarios, _training_messages
from .stats_catalog_grounding import _catalog_reference
from .stats_cross_format import _evaluate, _format_shift_case, _gate_report, _records
from .stats_dgp import Scenario, simulate_scenario
from .stats_evolve import _adapter_config_for_child, _start_caffeinate
from .stats_family_router import _expert_context
from .stats_router_replication import _historical_scenario_audit, _scenario_semantic_payload
from .stats_training import (
    _enable_gradient_checkpointing_once,
    _optimizer,
    _stats_snapshot,
    _StatsCallback,
)

_TRAINER_VERSION = 1


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "catalog-distillation-v1"


def _data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "catalog-distillation-v1"


def _implementation_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "stats_catalog_distillation.py": sha256_file(Path(__file__)),
        "stats_canonical_bottleneck.py": sha256_file(root / "stats_canonical_bottleneck.py"),
        "stats_catalog_grounding.py": sha256_file(root / "stats_catalog_grounding.py"),
        "stats_cross_format.py": sha256_file(root / "stats_cross_format.py"),
        "forge_training.py": sha256_file(root / "forge_training.py"),
        "stats_agent.py": sha256_file(root / "stats_agent.py"),
    }


def _contract_scenarios(
    config: ProjectConfig,
    report_name: str,
    *,
    name_prefix: str = "",
) -> list[Scenario]:
    path = config.root / "reports" / "evolve" / report_name
    contract = json.loads(path.read_text(encoding="utf-8"))
    scenarios: list[Scenario] = []
    for name, fields in contract["blueprint_contracts"].items():
        shard = _registered_scenarios(
            dict(contract["settings"][name]),
            name=f"{name_prefix}{name}",
        )
        scenarios.extend(shard)
        expected = str(fields["blueprint_sha256"])
        actual = canonical_hash([scenario.to_dict() for scenario in shard])
        if actual != expected:
            raise RuntimeError(
                f"Historical registered blueprint reconstruction changed: {report_name}:{name}"
            )
    return scenarios


def _grounded_blueprints(
    scenarios: list[Scenario],
    *,
    fraction: float,
) -> set[str]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("H8 grounding dropout fraction must be strictly between zero and one")
    by_family: dict[str, list[Scenario]] = defaultdict(list)
    for scenario in scenarios:
        by_family[scenario.family_id].append(scenario)
    grounded: set[str] = set()
    for family_id, rows in sorted(by_family.items()):
        ordered = sorted(
            rows,
            key=lambda scenario: canonical_hash(
                {"h8_catalog_grounding_dropout_v1": scenario.blueprint_id}
            ),
        )
        raw_count = len(ordered) * fraction
        count = round(raw_count)
        if abs(raw_count - count) > 1e-9:
            raise RuntimeError(f"H8 grounding fraction is not exact for {family_id}")
        grounded.update(scenario.blueprint_id for scenario in ordered[:count])
    return grounded


def _canonical_training_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    messages = _training_messages(case, canonical=True)
    messages[0]["content"] = (
        messages[0]["content"]
        + " The methods array must contain one repository canonical method ID."
    )
    return messages


def _training_rows(
    simulations: list[dict[str, Any]],
    *,
    grounded_blueprints: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    control: list[dict[str, Any]] = []
    candidate: list[dict[str, Any]] = []
    catalog = _catalog_reference()
    for simulation in simulations:
        case = _format_shift_case(simulation)
        base_messages = _canonical_training_messages(case)
        metadata = {
            "blueprint_id": case["case_id"],
            "family_id": case["family_id"],
            "target_style": "canonical-id",
            "loss_weight": 1.0,
        }
        control.append(
            {
                "messages": [dict(message) for message in base_messages],
                "metadata": {**metadata, "catalog_grounded": False},
            }
        )
        candidate_messages = [dict(message) for message in base_messages]
        grounded = str(case["case_id"]) in grounded_blueprints
        if grounded:
            candidate_messages[0]["content"] = (
                candidate_messages[0]["content"]
                + "\n\nRepository method catalog (fixed across all grounded training rows):\n"
                + catalog
            )
        candidate.append(
            {
                "messages": candidate_messages,
                "metadata": {**metadata, "catalog_grounded": grounded},
            }
        )
    return control, candidate


def prepare_catalog_distillation_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("catalog_distillation"))
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "contract.json"
    public_path = config.root / "reports" / "evolve" / "catalog-distillation-v1-contract.json"

    h7_path = config.root / "reports" / "evolve" / "catalog-grounding-v1-confirmation.json"
    if not h7_path.exists():
        raise RuntimeError("H8 requires the confirmed H7 interface mechanism")
    h7 = json.loads(h7_path.read_text(encoding="utf-8"))
    if not h7.get("synthetic_interface_confirmed") or not h7["confirmation_gate"]["passed"]:
        raise RuntimeError("H7 did not confirm catalog grounding; H8 is not authorized")
    if h7.get("external_benchmark_authorized"):
        raise RuntimeError("H7 unexpectedly authorized an external benchmark")

    h6_confirmation = (
        config.path_for("evolution_dir")
        / "canonical-bottleneck-v1"
        / "surfaces"
        / "confirmation_shard.jsonl"
    )
    if h6_confirmation.exists():
        raise RuntimeError("H6 confirmation was unexpectedly opened")

    _, adapter_paths = _expert_context(config)
    parent = adapter_paths["parent"]
    parent_sha = sha256_file(parent / "adapters.safetensors")
    registered: dict[str, Any] = {}
    scenarios_by_name: dict[str, list[Scenario]] = {}
    all_scenarios: list[Scenario] = []
    seen: set[str] = set()
    for name in ("training_pool", "selection_shard", "confirmation_shard"):
        shard = dict(settings[name])
        scenarios = _registered_scenarios(shard, name=f"catalog-distillation:{name}")
        ids = {scenario.blueprint_id for scenario in scenarios}
        if ids & seen:
            raise RuntimeError(f"H8 blueprint overlap at {name}")
        seen.update(ids)
        scenarios_by_name[name] = scenarios
        all_scenarios.extend(scenarios)
        registered[name] = {
            "split": str(shard["split"]),
            "seed": int(shard["seed"]),
            "pool_count": int(shard["pool_count"]),
            "selected_per_family": int(shard["selected_per_family"]),
            "count": len(scenarios),
            "blueprint_sha256": canonical_hash([scenario.to_dict() for scenario in scenarios]),
        }

    previous_scenarios = _contract_scenarios(
        config, "canonical-bottleneck-v1-contract.json"
    ) + _contract_scenarios(
        config,
        "catalog-grounding-v1-contract.json",
        name_prefix="catalog-grounding:",
    )
    previous_ids = {scenario.blueprint_id for scenario in previous_scenarios}
    previous_semantics = {
        canonical_hash(_scenario_semantic_payload(scenario.to_dict()))
        for scenario in previous_scenarios
    }
    new_ids = {scenario.blueprint_id for scenario in all_scenarios}
    new_semantics = {
        canonical_hash(_scenario_semantic_payload(scenario.to_dict())) for scenario in all_scenarios
    }
    if new_ids & previous_ids or new_semantics & previous_semantics:
        raise RuntimeError("H8 overlaps H6/H7 registered blueprints or semantic points")

    audit = _historical_scenario_audit(
        config,
        all_scenarios,
        excluded_root=_data_root(config),
        minimum_normalized_distance=float(settings["minimum_normalized_distance"]),
    )
    if not audit["passed"]:
        raise RuntimeError("H8 blueprints failed historical-overlap audit")

    grounded = _grounded_blueprints(
        scenarios_by_name["training_pool"],
        fraction=float(settings["catalog_grounding_fraction"]),
    )
    h7_effect = dict(h7["confirmation_gate"]["effect_points"])
    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": "H8 catalog-grounding distillation",
        "method_version": int(settings["method_version"]),
        "causal_question": (
            "Can intermittent exposure to the fixed H7 repository catalog during canonical-target "
            "training transfer fine-grained method discrimination into the weights, so the catalog "
            "can be removed at held-out evaluation?"
        ),
        "h7_confirmation_result_fingerprint": h7["result_fingerprint"],
        "h7_report_sha256": sha256_file(h7_path),
        "h7_confirmed_effect_points": {
            "exact_vs_control": float(h7_effect["exact_vs_control"]),
            "method_vs_control": float(h7_effect["method_vs_control"]),
        },
        "parent": {
            "name": "v0.3.0-parent",
            "adapter_path": str(parent),
            "adapter_sha256": parent_sha,
        },
        "settings": settings,
        "blueprint_contracts": registered,
        "historical_overlap_audit": audit,
        "prior_registered_overlap": {
            "prior_blueprints": len(previous_ids),
            "h8_blueprints": len(new_ids),
            "blueprint_id_overlap_count": len(new_ids & previous_ids),
            "semantic_overlap_count": len(new_semantics & previous_semantics),
        },
        "grounding_assignment": {
            "fraction": float(settings["catalog_grounding_fraction"]),
            "count": len(grounded),
            "training_count": len(scenarios_by_name["training_pool"]),
            "blueprint_sha256": canonical_hash(sorted(grounded)),
            "rule": (
                "Within each family, rank training blueprints by canonical hash of "
                "h8_catalog_grounding_dropout_v1 + blueprint_id and ground the first half."
            ),
        },
        "implementation_sha256": _implementation_manifest(),
        "arms": {
            "canonical-menu-free-control": (
                "Canonical JSON target; catalog absent from every training row"
            ),
            "catalog-dropout-distill": (
                "Same canonical JSON target; fixed catalog present on exactly half of each "
                "family's "
                "training rows and absent on the other half"
            ),
        },
        "matched_training_budget": (
            "Same parent, scenarios, targets, training seed, optimizer, learning rates, 96 "
            "microsteps, 24 optimizer updates, and fixed endpoint. Catalog-bearing candidate rows "
            "are longer by intervention, so input-token FLOPs are not claimed identical."
        ),
        "selection_policy": (
            "Evaluation is menu-free for parent, control, and candidate. Only "
            "catalog-dropout-distill "
            "may advance, and only if every registered selection gate passes."
        ),
        "confirmation_policy": (
            "Confirmation blueprints are registered now but simulations remain unopened until all "
            "selection gates pass. Historical P-Bench/StatQA remain unavailable for H8 tuning."
        ),
        "claim_boundary": (
            "H8 can establish a synthetic weight-level transfer mechanism only. It does not "
            "promote "
            "the champion or establish external capability."
        ),
        "selection_opened": False,
        "confirmation_simulations_opened": False,
        "external_benchmark_opened": False,
    }
    contract["fingerprint"] = canonical_hash(contract)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("H8 contract is immutable")
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
            raise RuntimeError("H8 confirmation cannot open before selection")
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        if pilot.get("selected_arm") != "catalog-dropout-distill":
            raise RuntimeError("H8 selection did not authorize confirmation")
    shard = dict(contract["settings"][name])
    scenarios = _registered_scenarios(shard, name=f"catalog-distillation:{name}")
    blueprint_sha = canonical_hash([scenario.to_dict() for scenario in scenarios])
    if blueprint_sha != contract["blueprint_contracts"][name]["blueprint_sha256"]:
        raise RuntimeError(f"H8 registered blueprints changed for {name}")
    path = _data_root(config) / "surfaces" / f"{name}.jsonl"
    manifest_path = path.with_suffix(".manifest.json")
    stats = config.section("stats_data")
    fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "name": name,
            "blueprints": blueprint_sha,
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
            raise RuntimeError(f"H8 {name} surface changed")
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


def prepare_catalog_distillation_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_catalog_distillation_contract(config)
    root = _data_root(config)
    status_path = root / "data-status.json"
    training_manifest, training_surface = _simulate_surface(config, contract, name="training_pool")
    selection_manifest, selection_surface = _simulate_surface(
        config, contract, name="selection_shard"
    )
    training_scenarios = [
        Scenario(**dict(simulation["scenario"])) for simulation in training_surface
    ]
    grounded = _grounded_blueprints(
        training_scenarios,
        fraction=float(contract["grounding_assignment"]["fraction"]),
    )
    if canonical_hash(sorted(grounded)) != contract["grounding_assignment"]["blueprint_sha256"]:
        raise RuntimeError("H8 grounding dropout assignment changed")
    control_rows, candidate_rows = _training_rows(
        training_surface,
        grounded_blueprints=grounded,
    )
    selector_rows = _records(selection_surface, training=False)
    cases = [_format_shift_case(simulation) for simulation in selection_surface]
    paths = {
        "canonical-menu-free-control": root / "train-menu-free-control.jsonl",
        "catalog-dropout-distill": root / "train-catalog-dropout.jsonl",
        "selector": root / "selection-selector.jsonl",
        "cases": root / "selection-format.jsonl",
    }
    write_jsonl(paths["canonical-menu-free-control"], control_rows)
    write_jsonl(paths["catalog-dropout-distill"], candidate_rows)
    write_jsonl(paths["selector"], selector_rows)
    write_jsonl(paths["cases"], cases)
    file_sha = {name: sha256_file(path) for name, path in paths.items()}
    fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "training_surface": training_manifest["fingerprint"],
            "selection_surface": selection_manifest["fingerprint"],
            "files": file_sha,
        }
    )
    status = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "contract_fingerprint": contract["fingerprint"],
        "training_groups": len(training_surface),
        "training_records_per_arm": len(control_rows),
        "candidate_grounded_records": len(grounded),
        "selection_groups": len(selection_surface),
        "file_sha256": file_sha,
        "confirmation_opened": False,
    }
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise RuntimeError("H8 prepared data changed")
        return existing
    write_json(status_path, status)
    write_json(config.root / "reports" / "evolve" / "catalog-distillation-v1-data.json", status)
    return status


def _train_arm(config: ProjectConfig, *, arm: str) -> dict[str, Any]:
    arms = {"canonical-menu-free-control", "catalog-dropout-distill"}
    if arm not in arms:
        raise ValueError(f"Unknown H8 arm: {arm}")
    contract = prepare_catalog_distillation_contract(config)
    data = prepare_catalog_distillation_data(config)
    settings = dict(contract["settings"])
    output = _root(config) / "arms" / arm
    status_path = output / "status.json"
    train_path = _data_root(config) / (
        "train-catalog-dropout.jsonl"
        if arm == "catalog-dropout-distill"
        else "train-menu-free-control.jsonl"
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
        raise RuntimeError(f"H8 arm state changed: {arm}")

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
            raise RuntimeError(f"H8 forbids truncation: {maximum_tokens} > {max_length}")
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
            config, parent, output, cycle=11, arm=f"catalog-distillation:{arm}"
        )
        adapter_config.setdefault("stats", {}).update(
            {
                "method": "H8 catalog-grounding distillation",
                "training_arm": arm,
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
        raise RuntimeError("H8 selector coverage differs")
    pairs = list(zip(records, simulations, strict=True))
    for record, simulation in pairs:
        if str(record["metadata"]["blueprint_id"]) != str(simulation["scenario"]["blueprint_id"]):
            raise RuntimeError("H8 selector blueprint pairing changed")
    return pairs


def _distillation_gate(
    *,
    parent: dict[str, Any],
    control: dict[str, Any],
    candidate: dict[str, Any],
    gates: dict[str, Any],
    h7_effect: dict[str, float],
) -> dict[str, Any]:
    report = _gate_report(
        parent=parent,
        control=control,
        candidate=candidate,
        gates=gates,
    )
    fraction = float(gates["minimum_h7_effect_retention_fraction"])
    exact_effect = float(report["effect_points"]["exact_vs_control"])
    method_effect = float(report["effect_points"]["method_vs_control"])
    exact_anchor = float(h7_effect["exact_vs_control"])
    method_anchor = float(h7_effect["method_vs_control"])
    retention = {
        "exact": exact_effect / exact_anchor if exact_anchor > 0 else 0.0,
        "method": method_effect / method_anchor if method_anchor > 0 else 0.0,
    }
    report["checks"]["h7_exact_effect_retained"] = retention["exact"] >= fraction
    report["checks"]["h7_method_effect_retained"] = retention["method"] >= fraction
    report["h7_effect_retention_fraction"] = retention
    report["passed"] = all(report["checks"].values())
    return report


def run_catalog_distillation_pilot(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_catalog_distillation_contract(config)
    data = prepare_catalog_distillation_data(config)
    arms = ["canonical-menu-free-control", "catalog-dropout-distill"]
    statuses = {arm: _train_arm(config, arm=arm) for arm in arms}
    expected_microsteps = int(contract["settings"]["microsteps"])
    if {int(status["microsteps"]) for status in statuses.values()} != {expected_microsteps}:
        raise RuntimeError("H8 arms did not receive matched microstep budgets")
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
            "surface": "selection-menu-free",
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
    gate = _distillation_gate(
        parent=scores["v0.3-parent"],
        control=scores[arms[0]],
        candidate=scores[arms[1]],
        gates=dict(contract["settings"]["selection_gates"]),
        h7_effect=dict(contract["h7_confirmed_effect_points"]),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": evaluation_fingerprint,
        "method": "H8 catalog-grounding distillation pilot",
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["fingerprint"],
        "fixed_endpoints": True,
        "menu_free_evaluation": True,
        "scores": {
            name: {"selector": value["selector"], "format_shift": value["format_shift"]}
            for name, value in scores.items()
        },
        "selection_gate": gate,
        "selected_arm": "catalog-dropout-distill" if gate["passed"] else None,
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
    write_json(config.root / "reports" / "evolve" / "catalog-distillation-v1-pilot.json", public)
    return public


def run_catalog_distillation_confirmation(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_catalog_distillation_contract(config)
    pilot_path = _root(config) / "pilot.json"
    if not pilot_path.exists():
        raise RuntimeError("H8 pilot has not selected a candidate")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("selected_arm") != "catalog-dropout-distill":
        raise RuntimeError("H8 selection did not authorize confirmation")
    manifest, surface = _simulate_surface(config, contract, name="confirmation_shard")
    selector_records = _records(surface, training=False)
    selector_rows = _paired_selector_rows(selector_records, surface)
    cases = [_format_shift_case(simulation) for simulation in surface]
    arms = ["canonical-menu-free-control", "catalog-dropout-distill"]
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
            "surface": "confirmation-menu-free",
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
    gate = _distillation_gate(
        parent=scores["v0.3-parent"],
        control=scores[arms[0]],
        candidate=scores[arms[1]],
        gates=dict(contract["settings"]["confirmation_gates"]),
        h7_effect=dict(contract["h7_confirmed_effect_points"]),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": evaluation_fingerprint,
        "method": "H8 catalog-grounding distillation confirmation",
        "contract_fingerprint": contract["fingerprint"],
        "pilot_result_fingerprint": pilot["result_fingerprint"],
        "confirmation_manifest_fingerprint": manifest["fingerprint"],
        "fixed_endpoints": True,
        "menu_free_evaluation": True,
        "scores": {
            name: {"selector": value["selector"], "format_shift": value["format_shift"]}
            for name, value in scores.items()
        },
        "confirmation_gate": gate,
        "synthetic_weight_transfer_confirmed": bool(gate["passed"]),
        "external_benchmark_authorized": False,
        "next_step": (
            "preregister-independent-external-weight-evidence"
            if gate["passed"]
            else "reject-h8-catalog-distillation"
        ),
        "private_details": {name: value["details"] for name, value in scores.items()},
    }
    report["result_fingerprint"] = canonical_hash(
        {key: value for key, value in report.items() if key != "private_details"}
    )
    write_json(_root(config) / "confirmation.json", report)
    public = dict(report)
    public.pop("private_details")
    write_json(
        config.root / "reports" / "evolve" / "catalog-distillation-v1-confirmation.json",
        public,
    )
    return public

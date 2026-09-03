from __future__ import annotations

import gc
import json
import time
from collections import Counter
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
from .stats_agent import StatsAgent
from .stats_evolve import _adapter_config_for_child, _start_caffeinate
from .stats_external_catalog import _metrics, _paired_summary
from .stats_external_domain_bridge import _historical_rows
from .stats_external_weight_bridge import (
    _evaluation_case,
    _synthetic_evaluation_cases,
    _synthetic_replay_cases,
    _training_record,
)
from .stats_external_weight_bridge_amendment import _fixed_evaluation_case
from .stats_selector_external_amendment import _evaluate_control
from .stats_training import (
    _enable_gradient_checkpointing_once,
    _optimizer,
    _stats_snapshot,
    _StatsCallback,
)

_TRAINER_VERSION = 1


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "guarded-weight-bridge-v1"


def _data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "guarded-weight-bridge-v1"


def _implementation_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "stats_guarded_weight_bridge.py": sha256_file(Path(__file__)),
        "stats_external_weight_bridge.py": sha256_file(
            root / "stats_external_weight_bridge.py"
        ),
        "stats_external_weight_bridge_amendment.py": sha256_file(
            root / "stats_external_weight_bridge_amendment.py"
        ),
        "forge_training.py": sha256_file(root / "forge_training.py"),
        "stats_training.py": sha256_file(root / "stats_training.py"),
        "stats_selector_external_amendment.py": sha256_file(
            root / "stats_selector_external_amendment.py"
        ),
    }


def _h19_candidate_paths(config: ProjectConfig, sources: list[str]) -> dict[str, Path]:
    h19_root = config.path_for("artifact_dir") / "external-weight-bridge-v1"
    return {
        source: h19_root
        / "folds"
        / source
        / "held-out-progress"
        / "menu-free-control.jsonl"
        for source in sources
    }


def _h19_synthetic_paths(config: ProjectConfig, sources: list[str]) -> dict[str, Path]:
    h19_root = config.path_for("artifact_dir") / "external-weight-bridge-v1"
    return {
        source: h19_root
        / "folds"
        / source
        / "synthetic-retention-progress"
        / "menu-free-control.jsonl"
        for source in sources
    }


def prepare_guarded_weight_bridge_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("guarded_weight_bridge"))
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "contract.json"
    public_path = config.root / "reports" / "evolve" / "guarded-weight-bridge-v1-contract.json"

    h19_report_path = (
        config.root / "reports" / "evolve" / "external-weight-bridge-v1-training.json"
    )
    h19_contract_path = (
        config.root / "reports" / "evolve" / "external-weight-bridge-v1-contract.json"
    )
    h19 = json.loads(h19_report_path.read_text(encoding="utf-8"))
    h19_contract = json.loads(h19_contract_path.read_text(encoding="utf-8"))
    if not h19.get("complete") or h19.get("selected") is not False:
        raise RuntimeError("H20 requires the terminal-negative H19 result")
    if h19.get("final_all_source_adapter") is not None:
        raise RuntimeError("H20 requires H19 to have trained no all-source adapter")
    sources = [str(value) for value in h19_contract["fold_sources"]]
    if len(sources) != 6 or set(sources) != set(h19["fold_statuses"]):
        raise RuntimeError("H20 requires all six complete H19 source-held-out folds")
    for source, status in h19["fold_statuses"].items():
        adapter_path = Path(status["adapter_path"]) / "adapters.safetensors"
        if not status.get("complete") or sha256_file(adapter_path) != status["adapter_sha256"]:
            raise RuntimeError(f"H20 H19 fold adapter changed: {source}")

    candidate_paths = _h19_candidate_paths(config, sources)
    synthetic_paths = _h19_synthetic_paths(config, sources)
    parent_synthetic_path = (
        config.path_for("artifact_dir")
        / "external-weight-bridge-v1"
        / "folds"
        / "parent"
        / "synthetic-retention-progress"
        / "menu-free-control.jsonl"
    )
    required_paths = [*candidate_paths.values(), *synthetic_paths.values(), parent_synthetic_path]
    for path in required_paths:
        if not path.is_file():
            raise RuntimeError(f"H20 requires completed H19 detail file: {path}")

    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": "H20 invalid-control guarded external weight bridge",
        "method_version": int(settings["method_version"]),
        "causal_question": (
            "Can the H19 source-held-out LoRA bridge safely repair invalid menu-free control "
            "outputs when the runtime structurally preserves every valid control output?"
        ),
        "hypothesis_origin": (
            "Post-H19 retrospective development diagnosis: four of five H19 control-only losses "
            "were invalid LoRA outputs, while two of three H19 gains repaired invalid controls."
        ),
        "evidence_status": "retrospective-historical-development-only",
        "h19": {
            "result_fingerprint": h19["result_fingerprint"],
            "report_sha256": sha256_file(h19_report_path),
            "contract_fingerprint": h19_contract["fingerprint"],
            "contract_sha256": sha256_file(h19_contract_path),
            "fold_adapter_sha256": {
                source: h19["fold_statuses"][source]["adapter_sha256"] for source in sources
            },
            "held_out_detail_sha256": {
                source: sha256_file(candidate_paths[source]) for source in sources
            },
            "synthetic_detail_sha256": {
                source: sha256_file(synthetic_paths[source]) for source in sources
            },
            "parent_synthetic_detail_sha256": sha256_file(parent_synthetic_path),
        },
        "parent": dict(h19_contract["parent"]),
        "fold_sources": sources,
        "runtime_policy": [
            "If the unchanged menu-free control output is valid, return it unchanged.",
            "Only when the control output is invalid, query the H20 LoRA adapter.",
            "If the LoRA output is valid, return it; otherwise retain the invalid control output.",
        ],
        "structural_safety": (
            "A correct control prediction is necessarily valid and therefore cannot be replaced "
            "under this runtime. Exact control correctness cannot decrease by construction."
        ),
        "settings": settings,
        "selection_policy": (
            "Read back the already-open H19 source-held-out and synthetic details under the fixed "
            "validity guard. Train one new all-source adapter from the unchanged v0.3 parent only "
            "if every historical and synthetic gate passes. No checkpoint selection, learning-rate "
            "search, source fitting, or fresh external source is allowed."
        ),
        "claim_boundary": (
            "H20 is a post-H19 historical development mechanism. It cannot turn E2/E3/E4 into "
            "fresh evidence, replace the champion, authorize release, or establish external "
            "capability. Its adapter and runtime remain development-only, and E5 stays unopened."
        ),
        "implementation_sha256": _implementation_manifest(),
        "all_source_training_started": False,
        "fresh_external_evaluation_opened": False,
    }
    contract["fingerprint"] = canonical_hash(contract)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("H20 contract is immutable")
        write_json(public_path, existing)
        return existing
    write_json(lock_path, contract)
    write_json(public_path, contract)
    return contract


def prepare_guarded_weight_bridge_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_guarded_weight_bridge_contract(config)
    rows = _historical_rows(config)
    replay = _synthetic_replay_cases(config)
    retention = _synthetic_evaluation_cases(config)
    repeat = int(contract["settings"]["external_repeat"])
    training_rows = [
        _training_record(row, origin="synthetic-replay", fold="all-sources", repeat_index=0)
        for row in replay
    ]
    for repeat_index in range(repeat):
        training_rows.extend(
            _training_record(
                row,
                origin="historical-external",
                fold="all-sources",
                repeat_index=repeat_index,
            )
            for row in rows
        )
    root = _data_root(config)
    root.mkdir(parents=True, exist_ok=True)
    train_path = root / "all-sources" / "train.jsonl"
    retention_path = root / "synthetic-retention.jsonl"
    write_jsonl(train_path, training_rows)
    write_jsonl(retention_path, [_fixed_evaluation_case(row) for row in retention])
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "contract_fingerprint": contract["fingerprint"],
        "historical_case_count": len(rows),
        "historical_case_fingerprint": canonical_hash(rows),
        "synthetic_replay_count": len(replay),
        "synthetic_replay_fingerprint": canonical_hash(replay),
        "training_records": len(training_rows),
        "train_path": str(train_path),
        "train_sha256": sha256_file(train_path),
        "synthetic_retention_path": str(retention_path),
        "synthetic_retention_sha256": sha256_file(retention_path),
        "all_source_training_started": False,
        "fresh_external_evaluation_opened": False,
    }
    report["data_fingerprint"] = canonical_hash(report)
    write_json(_root(config) / "data.json", report)
    write_json(
        config.root / "reports" / "evolve" / "guarded-weight-bridge-v1-data.json",
        report,
    )
    return report


def _control_details(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": str(row["case_id"]),
            "eligible": True,
            "gold_method_id": str(row["gold_method_id"]),
            "predicted_method_id": row["control_predicted_method_id"],
            "valid_output": bool(row["control_valid_output"]),
            "correct": bool(row["control_correct"]),
        }
        for row in rows
    ]


def _guard_detail(
    control: dict[str, Any],
    lora: dict[str, Any],
    *,
    source_id: str,
) -> dict[str, Any]:
    if str(control["case_id"]) != str(lora["case_id"]):
        raise RuntimeError("H20 control/LoRA case alignment changed")
    if str(control["gold_method_id"]) != str(lora["gold_method_id"]):
        raise RuntimeError("H20 control/LoRA gold alignment changed")
    if bool(control["correct"]) and not bool(control["valid_output"]):
        raise RuntimeError("H20 structural safety requires correct controls to be valid")
    if bool(control["valid_output"]):
        route = "valid-menu-free-control"
        prediction = control["predicted_method_id"]
        valid = True
        correct = bool(control["correct"])
    elif bool(lora["valid_output"]):
        route = "invalid-control-lora-repair"
        prediction = lora["predicted_method_id"]
        valid = True
        correct = bool(lora["correct"])
    else:
        route = "both-invalid-control-fallback"
        prediction = control["predicted_method_id"]
        valid = False
        correct = bool(control["correct"])
    return {
        "case_id": str(control["case_id"]),
        "source_id": source_id,
        "eligible": bool(control.get("eligible", True)),
        "gold_method_id": str(control["gold_method_id"]),
        "predicted_method_id": prediction,
        "valid_output": valid,
        "correct": correct,
        "route": route,
        "control_predicted_method_id": control["predicted_method_id"],
        "control_valid_output": bool(control["valid_output"]),
        "control_correct": bool(control["correct"]),
        "lora_predicted_method_id": lora["predicted_method_id"],
        "lora_valid_output": bool(lora["valid_output"]),
        "lora_correct": bool(lora["correct"]),
    }


def _guard_details(
    controls: list[dict[str, Any]],
    loras: list[dict[str, Any]],
    source_by_case: dict[str, str],
) -> list[dict[str, Any]]:
    control_by_id = {str(row["case_id"]): row for row in controls}
    lora_by_id = {str(row["case_id"]): row for row in loras}
    if set(control_by_id) != set(lora_by_id):
        raise RuntimeError("H20 control/LoRA case sets changed")
    return [
        _guard_detail(
            control_by_id[case_id],
            lora_by_id[case_id],
            source_id=source_by_case[case_id],
        )
        for case_id in sorted(control_by_id)
    ]


def _read_h19_held_out_details(
    config: ProjectConfig, contract: dict[str, Any]
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for source, path in _h19_candidate_paths(config, contract["fold_sources"]).items():
        expected = contract["h19"]["held_out_detail_sha256"][source]
        if sha256_file(path) != expected:
            raise RuntimeError(f"H20 H19 held-out details changed: {source}")
        details.extend(read_jsonl(path))
    return details


def _historical_readback(
    config: ProjectConfig, contract: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _historical_rows(config)
    controls = _control_details(rows)
    loras = _read_h19_held_out_details(config, contract)
    source_by_case = {str(row["case_id"]): str(row["source_id"]) for row in rows}
    guarded = _guard_details(controls, loras, source_by_case)
    paired = _paired_summary(controls, guarded)
    source_paired: dict[str, Any] = {}
    for source in contract["fold_sources"]:
        ids = {case_id for case_id, value in source_by_case.items() if value == source}
        source_paired[source] = _paired_summary(
            [row for row in controls if str(row["case_id"]) in ids],
            [row for row in guarded if str(row["case_id"]) in ids],
        )
    metrics = {
        "control": _metrics(controls),
        "guarded": _metrics(guarded),
        "paired": paired,
        "source_paired": source_paired,
        "route_counts": dict(sorted(Counter(row["route"] for row in guarded).items())),
        "correct_invalid_control_repairs": sum(
            row["route"] == "invalid-control-lora-repair" and row["correct"]
            for row in guarded
        ),
        "repair_source_count": len(
            {
                row["source_id"]
                for row in guarded
                if row["route"] == "invalid-control-lora-repair" and row["correct"]
            }
        ),
    }
    return metrics, guarded


def _synthetic_readback(
    config: ProjectConfig, contract: dict[str, Any]
) -> dict[str, Any]:
    h19_root = config.path_for("artifact_dir") / "external-weight-bridge-v1"
    parent_path = (
        h19_root
        / "folds"
        / "parent"
        / "synthetic-retention-progress"
        / "menu-free-control.jsonl"
    )
    if sha256_file(parent_path) != contract["h19"]["parent_synthetic_detail_sha256"]:
        raise RuntimeError("H20 H19 parent synthetic details changed")
    parent = list(read_jsonl(parent_path))
    source_by_case = {str(row["case_id"]): "synthetic-retention" for row in parent}
    folds: dict[str, Any] = {}
    for source, path in _h19_synthetic_paths(config, contract["fold_sources"]).items():
        if sha256_file(path) != contract["h19"]["synthetic_detail_sha256"][source]:
            raise RuntimeError(f"H20 H19 synthetic details changed: {source}")
        lora = list(read_jsonl(path))
        guarded = _guard_details(parent, lora, source_by_case)
        folds[source] = {
            "metrics": _metrics(guarded),
            "paired": _paired_summary(parent, guarded),
            "route_counts": dict(sorted(Counter(row["route"] for row in guarded).items())),
        }
    return {"parent": _metrics(parent), "folds": folds}


def _gate(
    historical: dict[str, Any], synthetic: dict[str, Any], gates: dict[str, Any]
) -> dict[str, Any]:
    source_nets = [
        int(value["net_improvements"]) for value in historical["source_paired"].values()
    ]
    synthetic_accuracies = [
        float(value["metrics"]["eligible_accuracy"]) for value in synthetic["folds"].values()
    ]
    synthetic_validities = [
        float(value["metrics"]["valid_output_rate"]) for value in synthetic["folds"].values()
    ]
    checks = {
        "historical_net_improvement": int(historical["paired"]["net_improvements"])
        >= int(gates["minimum_historical_net_improvements"]),
        "zero_control_only_losses": int(historical["paired"]["control_only"])
        <= int(gates["maximum_control_only_losses"]),
        "all_sources_nonnegative": min(source_nets)
        >= int(gates["minimum_worst_source_net_improvement"]),
        "invalid_control_repairs": int(historical["correct_invalid_control_repairs"])
        >= int(gates["minimum_correct_invalid_control_repairs"]),
        "repair_source_count": int(historical["repair_source_count"])
        >= int(gates["minimum_repair_source_count"]),
        "synthetic_accuracy_noninferior": min(synthetic_accuracies)
        >= float(synthetic["parent"]["eligible_accuracy"])
        - float(gates["maximum_synthetic_accuracy_regression"]),
        "synthetic_validity_noninferior": min(synthetic_validities)
        >= float(synthetic["parent"]["valid_output_rate"])
        - float(gates["maximum_synthetic_validity_regression"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "worst_source_net_improvement": min(source_nets),
        "minimum_synthetic_accuracy": min(synthetic_accuracies),
        "minimum_synthetic_validity": min(synthetic_validities),
    }


def _train_all_sources(
    config: ProjectConfig, contract: dict[str, Any], data: dict[str, Any]
) -> dict[str, Any]:
    settings = dict(contract["settings"])
    train_path = Path(data["train_path"])
    if sha256_file(train_path) != data["train_sha256"]:
        raise RuntimeError("H20 all-source training data changed")
    output = _root(config) / "all-sources"
    status_path = output / "status.json"
    fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "data": data["data_fingerprint"],
            "train_sha256": data["train_sha256"],
            "trainer_version": _TRAINER_VERSION,
        }
    )
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            return existing
        raise RuntimeError("H20 all-source training state changed")

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
            adapter_path=str(contract["parent"]["adapter_path"]),
            tokenizer_config={"trust_remote_code": True},
        )
        model.freeze()
        model.unfreeze(keys=["lora_a", "lora_b"])
        trainable_parameters = sum(
            parameter.size for _, parameter in tree_flatten(model.trainable_parameters())
        )
        if trainable_parameters <= 0:
            raise RuntimeError("H20 parent exposes no trainable LoRA parameters")
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
            raise RuntimeError(f"H20 forbids truncation: {maximum_tokens} > {max_length}")
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
                steps_per_report=group_size,
                steps_per_eval=microsteps,
                steps_per_save=microsteps,
                max_seq_length=max_length,
                adapter_file=str(output / "adapters.safetensors"),
                grad_checkpoint=False,
                grad_accumulation_steps=group_size,
                clear_cache_threshold=int(
                    float(settings["clear_cache_threshold_gb"]) * 1024**3
                ),
            ),
            loss=forge_loss,
            iterate_batches=forge_iterate_batches,
            training_callback=callback,
        )
        mx.save_safetensors(
            str(output / "adapters.safetensors"),
            dict(tree_flatten(model.trainable_parameters())),
        )
        adapter_config = _adapter_config_for_child(
            config,
            Path(str(contract["parent"]["adapter_path"])),
            output,
            cycle=20,
            arm="invalid-control-guarded-external-weight-bridge",
        )
        adapter_config.setdefault("stats", {}).update(
            {
                "method": "H20 invalid-control guarded external weight bridge",
                "runtime_policy": "query only after invalid menu-free control",
                "fixed_endpoint": True,
                "promotion_status": "development-only",
            }
        )
        write_json(output / "adapter_config.json", adapter_config)
        status: dict[str, Any] = {
            "schema_version": 1,
            "complete": True,
            "fingerprint": fingerprint,
            "contract_fingerprint": contract["fingerprint"],
            "data_fingerprint": data["data_fingerprint"],
            "adapter_path": str(output),
            "adapter_sha256": sha256_file(output / "adapters.safetensors"),
            "microsteps": microsteps,
            "optimizer_updates": microsteps // group_size,
            "training_records": len(rows),
            "maximum_tokens": maximum_tokens,
            "trainable_parameters": trainable_parameters,
            "training_seed": seed,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 4),
            "fresh_external_evaluation_opened": False,
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


def _evaluate_adapter(
    config: ProjectConfig,
    *,
    adapter_path: Path,
    cases: list[dict[str, Any]],
    surface: str,
    fingerprint: str,
) -> dict[str, Any]:
    agent = StatsAgent(config, adapter_path=adapter_path)
    agent.router.set_route("adapter")
    try:
        return _evaluate_control(
            agent,
            cases,
            progress_root=_root(config) / "all-sources" / f"{surface}-progress",
            evaluation_fingerprint=fingerprint,
        )
    finally:
        del agent
        gc.collect()
        mx.clear_cache()


def _runtime_smoke(
    config: ProjectConfig,
    contract: dict[str, Any],
    data: dict[str, Any],
    status: dict[str, Any],
) -> dict[str, Any]:
    rows = _historical_rows(config)
    controls = _control_details(rows)
    historical_cases = [_evaluation_case(row) for row in rows]
    historical_lora = _evaluate_adapter(
        config,
        adapter_path=Path(status["adapter_path"]),
        cases=historical_cases,
        surface="historical-resubstitution",
        fingerprint=canonical_hash(
            {
                "contract": contract["fingerprint"],
                "adapter": status["adapter_sha256"],
                "surface": "historical-resubstitution",
                "cases": canonical_hash(historical_cases),
            }
        ),
    )["details"]
    source_by_case = {str(row["case_id"]): str(row["source_id"]) for row in rows}
    historical_guarded = _guard_details(controls, historical_lora, source_by_case)

    parent_path = (
        config.path_for("artifact_dir")
        / "external-weight-bridge-v1"
        / "folds"
        / "parent"
        / "synthetic-retention-progress"
        / "menu-free-control.jsonl"
    )
    synthetic_parent = list(read_jsonl(parent_path))
    retention_cases = list(read_jsonl(Path(data["synthetic_retention_path"])))
    synthetic_lora = _evaluate_adapter(
        config,
        adapter_path=Path(status["adapter_path"]),
        cases=retention_cases,
        surface="synthetic-retention",
        fingerprint=canonical_hash(
            {
                "contract": contract["fingerprint"],
                "adapter": status["adapter_sha256"],
                "surface": "synthetic-retention",
                "cases": data["synthetic_retention_sha256"],
            }
        ),
    )["details"]
    synthetic_source = {
        str(row["case_id"]): "synthetic-retention" for row in synthetic_parent
    }
    synthetic_guarded = _guard_details(
        synthetic_parent, synthetic_lora, synthetic_source
    )
    return {
        "historical_training_resubstitution": {
            "evidence_status": "training-resubstitution-runtime-diagnostic-only",
            "control": _metrics(controls),
            "unguarded_lora": _metrics(historical_lora),
            "guarded": _metrics(historical_guarded),
            "paired": _paired_summary(controls, historical_guarded),
            "route_counts": dict(
                sorted(Counter(row["route"] for row in historical_guarded).items())
            ),
        },
        "synthetic_retention": {
            "parent": _metrics(synthetic_parent),
            "unguarded_lora": _metrics(synthetic_lora),
            "guarded": _metrics(synthetic_guarded),
            "paired": _paired_summary(synthetic_parent, synthetic_guarded),
            "route_counts": dict(
                sorted(Counter(row["route"] for row in synthetic_guarded).items())
            ),
        },
    }


def run_guarded_weight_bridge_training(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_guarded_weight_bridge_contract(config)
    data = prepare_guarded_weight_bridge_data(config)
    report_path = _root(config) / "training.json"
    public_path = config.root / "reports" / "evolve" / "guarded-weight-bridge-v1-training.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            existing.get("contract_fingerprint") != contract["fingerprint"]
            or existing.get("data_fingerprint") != data["data_fingerprint"]
        ):
            raise RuntimeError("H20 completed result does not match frozen inputs")
        write_json(public_path, existing)
        return existing

    historical, details = _historical_readback(config, contract)
    synthetic = _synthetic_readback(config, contract)
    gate = _gate(historical, synthetic, dict(contract["settings"]["gates"]))
    status = _train_all_sources(config, contract, data) if gate["passed"] else None
    smoke = _runtime_smoke(config, contract, data, status) if status is not None else None
    result: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "H20 invalid-control guarded external weight bridge training",
        "trainer_version": _TRAINER_VERSION,
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["data_fingerprint"],
        "evidence_status": "retrospective-historical-development-only",
        "historical_source_held_out": historical,
        "historical_gate": gate,
        "development_details": details,
        "synthetic_source_held_out": synthetic,
        "selected_for_development": bool(gate["passed"]),
        "final_all_source_adapter": status,
        "runtime_smoke": smoke,
        "runtime_manifest": (
            {
                "parent_adapter_path": contract["parent"]["adapter_path"],
                "parent_adapter_sha256": contract["parent"]["adapter_sha256"],
                "repair_adapter_path": status["adapter_path"],
                "repair_adapter_sha256": status["adapter_sha256"],
                "policy": contract["runtime_policy"],
                "status": "development-only",
            }
            if status is not None
            else None
        ),
        "fresh_external_evidence": False,
        "fresh_external_evaluation_opened": False,
        "champion_changed": False,
        "release_authorized": False,
        "next_step": (
            "preserve-h20-development-runtime-and-freeze-separate-e5-before-any-new-source"
            if gate["passed"]
            else "preserve-h20-negative-and-retain-v0.3-menu-free-control"
        ),
        "claim_boundary": contract["claim_boundary"],
    }
    result["result_fingerprint"] = canonical_hash(result)
    write_json(report_path, result)
    write_json(public_path, result)
    return result

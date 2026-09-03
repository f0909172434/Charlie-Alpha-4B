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
from .stats_catalog_grounding import _messages as _h14_messages
from .stats_evolve import _adapter_config_for_child, _start_caffeinate
from .stats_external_catalog import _metrics, _paired_summary
from .stats_external_domain_bridge import _historical_rows
from .stats_family_router import _expert_context
from .stats_selector_external_amendment import _evaluate_control
from .stats_training import (
    _enable_gradient_checkpointing_once,
    _optimizer,
    _stats_snapshot,
    _StatsCallback,
)

_TRAINER_VERSION = 1


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "external-weight-bridge-v1"


def _data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "external-weight-bridge-v1"


def _implementation_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "stats_external_weight_bridge.py": sha256_file(Path(__file__)),
        "forge_training.py": sha256_file(root / "forge_training.py"),
        "stats_training.py": sha256_file(root / "stats_training.py"),
        "stats_selector_external_amendment.py": sha256_file(
            root / "stats_selector_external_amendment.py"
        ),
        "stats_agent.py": sha256_file(root / "stats_agent.py"),
    }


def _synthetic_replay_cases(config: ProjectConfig) -> list[dict[str, Any]]:
    path = (
        config.path_for("evolution_dir")
        / "selector-sufficiency-v1"
        / "cases"
        / "training_shard.jsonl"
    )
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        style = str(row["style"])
        method = str(row["gold_methods"][0])
        if style not in {"audit", "researcher", "vignette"}:
            continue
        selected.setdefault((method, style), row)
    rows = [selected[key] for key in sorted(selected)]
    methods = {str(row["gold_methods"][0]) for row in rows}
    if len(rows) != 69 or len(methods) != 23:
        raise RuntimeError("H19 requires one replay row per observed method and full style")
    return rows


def _synthetic_evaluation_cases(config: ProjectConfig) -> list[dict[str, Any]]:
    path = (
        config.path_for("evolution_dir")
        / "selector-sufficiency-v1"
        / "cases"
        / "confirmation_shard.jsonl"
    )
    selected: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if str(row["style"]) != "audit":
            continue
        selected.setdefault(str(row["family_id"]), row)
    rows = [selected[key] for key in sorted(selected)]
    if len(rows) != 12:
        raise RuntimeError("H19 requires one frozen audit retention row per family")
    return rows


def _evaluation_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": str(row["case_id"]),
        "question": str(row["question"]),
        "gold_method_id": str(row["gold_method_id"]),
        "gold_methods": [str(row["gold_method_id"])],
        "gold_columns": list(row.get("gold_columns", [])),
        "head_eligible": True,
        "eligible": True,
    }


def _training_record(
    row: dict[str, Any],
    *,
    origin: str,
    fold: str,
    repeat_index: int,
) -> dict[str, Any]:
    case = _evaluation_case(row) if "gold_method_id" in row else dict(row)
    method = str(case["gold_methods"][0])
    target = json.dumps(
        {"columns": list(case.get("gold_columns", [])), "methods": [method]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    messages = _h14_messages(case, grounded=False)
    messages.append({"role": "assistant", "content": target})
    return {
        "messages": messages,
        "metadata": {
            "case_id": str(case["case_id"]),
            "origin": origin,
            "fold": fold,
            "repeat_index": repeat_index,
            "loss_weight": 1.0,
        },
    }


def prepare_external_weight_bridge_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("external_weight_bridge"))
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "contract.json"
    public_path = config.root / "reports" / "evolve" / "external-weight-bridge-v1-contract.json"

    h18_path = config.root / "reports" / "evolve" / "external-exemplar-router-v1-training.json"
    h18 = json.loads(h18_path.read_text(encoding="utf-8"))
    if h18.get("historical_cross_source_supported") is not False:
        raise RuntimeError("H19 requires the terminal-negative H18 exemplar route")
    rows = _historical_rows(config)
    sources = sorted({str(row["source_id"]) for row in rows})
    if len(sources) != 6 or len(rows) != 38:
        raise RuntimeError("H19 requires the frozen six-source 38-case development pool")
    replay = _synthetic_replay_cases(config)
    retention = _synthetic_evaluation_cases(config)
    replay_path = (
        config.path_for("evolution_dir")
        / "selector-sufficiency-v1"
        / "cases"
        / "training_shard.jsonl"
    )
    retention_path = (
        config.path_for("evolution_dir")
        / "selector-sufficiency-v1"
        / "cases"
        / "confirmation_shard.jsonl"
    )
    _, adapter_paths = _expert_context(config)
    parent = adapter_paths["parent"]
    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": "H19 source-held-out external representation LoRA bridge",
        "method_version": int(settings["method_version"]),
        "causal_question": (
            "Can fixed-compute LoRA adaptation on five already-opened external sources plus "
            "balanced synthetic replay improve menu-free generation on the held-out sixth source "
            "without losing any previously correct controls or synthetic retention cases?"
        ),
        "h18_negative_result_fingerprint": h18["result_fingerprint"],
        "h18_report_sha256": sha256_file(h18_path),
        "parent": {
            "name": "v0.3.0-parent",
            "adapter_path": str(parent),
            "adapter_sha256": sha256_file(parent / "adapters.safetensors"),
        },
        "historical_case_count": len(rows),
        "historical_case_fingerprint": canonical_hash(rows),
        "fold_sources": sources,
        "source_counts": dict(sorted(Counter(str(row["source_id"]) for row in rows).items())),
        "synthetic_replay": {
            "source_path": str(replay_path),
            "source_sha256": sha256_file(replay_path),
            "selected_count": len(replay),
            "selected_fingerprint": canonical_hash(replay),
            "policy": "first canonical case per observed method and each full H16 style",
        },
        "synthetic_retention": {
            "source_path": str(retention_path),
            "source_sha256": sha256_file(retention_path),
            "selected_count": len(retention),
            "selected_fingerprint": canonical_hash(retention),
            "policy": "first audit-style H16 confirmation case per statistical family",
        },
        "settings": settings,
        "fold_protocol": (
            "Train six independent adapters from the unchanged parent. Each adapter excludes every "
            "row from its held-out source. Evaluate only that source, then aggregate paired "
            "results."
        ),
        "selection_policy": (
            "No checkpoint selection or hyperparameter search. The fixed final microstep is used. "
            "A final all-source development adapter may be trained only if every aggregate, "
            "source-level, validity, and synthetic-retention gate passes."
        ),
        "claim_boundary": (
            "H19 is historical source-held-out development evidence. It cannot reuse E2/E3/E4 as "
            "fresh evidence, replace the champion, authorize release, or establish external "
            "capability. A pass authorizes only a separately frozen new-source evaluation."
        ),
        "implementation_sha256": _implementation_manifest(),
        "fold_training_started": False,
        "fresh_external_evaluation_opened": False,
    }
    contract["fingerprint"] = canonical_hash(contract)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("H19 contract is immutable")
        write_json(public_path, existing)
        return existing
    write_json(lock_path, contract)
    write_json(public_path, contract)
    return contract


def prepare_external_weight_bridge_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_external_weight_bridge_contract(config)
    rows = _historical_rows(config)
    if canonical_hash(rows) != contract["historical_case_fingerprint"]:
        raise RuntimeError("H19 historical rows changed")
    replay = _synthetic_replay_cases(config)
    retention = _synthetic_evaluation_cases(config)
    if canonical_hash(replay) != contract["synthetic_replay"]["selected_fingerprint"]:
        raise RuntimeError("H19 synthetic replay changed")
    if canonical_hash(retention) != contract["synthetic_retention"]["selected_fingerprint"]:
        raise RuntimeError("H19 synthetic retention set changed")
    root = _data_root(config)
    root.mkdir(parents=True, exist_ok=True)
    repeat = int(contract["settings"]["external_repeat"])
    fold_receipts: dict[str, Any] = {}
    for fold in [*contract["fold_sources"], "all-sources"]:
        external = (
            rows
            if fold == "all-sources"
            else [row for row in rows if str(row["source_id"]) != fold]
        )
        training_rows = [
            _training_record(row, origin="synthetic-replay", fold=fold, repeat_index=0)
            for row in replay
        ]
        for repeat_index in range(repeat):
            training_rows.extend(
                _training_record(
                    row,
                    origin="historical-external",
                    fold=fold,
                    repeat_index=repeat_index,
                )
                for row in external
            )
        train_path = root / "folds" / fold / "train.jsonl"
        write_jsonl(train_path, training_rows)
        fold_receipts[fold] = {
            "train_path": str(train_path),
            "train_sha256": sha256_file(train_path),
            "training_records": len(training_rows),
            "external_unique_records": len(external),
            "synthetic_replay_records": len(replay),
        }
    retention_path = root / "synthetic-retention.jsonl"
    write_jsonl(retention_path, [_evaluation_case(row) for row in retention])
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "contract_fingerprint": contract["fingerprint"],
        "folds": fold_receipts,
        "synthetic_retention_path": str(retention_path),
        "synthetic_retention_sha256": sha256_file(retention_path),
        "fold_training_started": False,
        "fresh_external_evaluation_opened": False,
    }
    report["data_fingerprint"] = canonical_hash(report)
    write_json(_root(config) / "data.json", report)
    write_json(config.root / "reports" / "evolve" / "external-weight-bridge-v1-data.json", report)
    return report


def _train_fold(config: ProjectConfig, *, fold: str) -> dict[str, Any]:
    contract = prepare_external_weight_bridge_contract(config)
    data = prepare_external_weight_bridge_data(config)
    if fold not in data["folds"]:
        raise ValueError(f"Unknown H19 fold: {fold}")
    settings = dict(contract["settings"])
    receipt = dict(data["folds"][fold])
    train_path = Path(receipt["train_path"])
    if sha256_file(train_path) != receipt["train_sha256"]:
        raise RuntimeError(f"H19 training data changed: {fold}")
    output = _root(config) / "folds" / fold
    status_path = output / "status.json"
    fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "data": data["data_fingerprint"],
            "fold": fold,
            "train_sha256": receipt["train_sha256"],
            "trainer_version": _TRAINER_VERSION,
        }
    )
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("fingerprint") == fingerprint:
            return existing
        raise RuntimeError(f"H19 fold state changed: {fold}")

    output.mkdir(parents=True, exist_ok=True)
    seed = int(settings["training_seed"]) + list(data["folds"]).index(fold)
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
            raise RuntimeError("H19 parent exposes no trainable LoRA parameters")
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
            raise RuntimeError(f"H19 forbids truncation: {maximum_tokens} > {max_length}")
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
            deadline=started + int(settings["max_seconds_per_fold"]),
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
            cycle=19,
            arm=f"external-weight-bridge:{fold}",
        )
        adapter_config.setdefault("stats", {}).update(
            {
                "method": "H19 source-held-out external representation LoRA bridge",
                "fold": fold,
                "fixed_endpoint": True,
                "promotion_status": "development-only",
            }
        )
        write_json(output / "adapter_config.json", adapter_config)
        status = {
            "schema_version": 1,
            "complete": True,
            "fingerprint": fingerprint,
            "fold": fold,
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
    fold: str,
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
            progress_root=_root(config) / "folds" / fold / f"{surface}-progress",
            evaluation_fingerprint=fingerprint,
        )
    finally:
        del agent
        gc.collect()
        mx.clear_cache()


def _gate_result(
    paired: dict[str, Any],
    source_paired: dict[str, Any],
    control: dict[str, Any],
    candidate: dict[str, Any],
    synthetic_parent: dict[str, Any],
    synthetic_folds: list[dict[str, Any]],
    gates: dict[str, Any],
) -> dict[str, Any]:
    worst_source = min(int(value["net_improvements"]) for value in source_paired.values())
    mean_synthetic_accuracy = float(
        np.mean([float(value["eligible_accuracy"]) for value in synthetic_folds])
    )
    mean_synthetic_validity = float(
        np.mean([float(value["valid_output_rate"]) for value in synthetic_folds])
    )
    checks = {
        "candidate_accuracy": float(candidate["eligible_accuracy"])
        >= float(gates["minimum_candidate_accuracy"]),
        "net_improvements": int(paired["net_improvements"])
        >= int(gates["minimum_net_improvements"]),
        "zero_control_only_losses": int(paired["control_only"])
        <= int(gates["maximum_control_only_losses"]),
        "all_sources_nonnegative": worst_source
        >= int(gates["minimum_worst_source_net_improvement"]),
        "validity_noninferior": float(candidate["valid_output_rate"])
        >= float(control["valid_output_rate"])
        - float(gates["maximum_validity_regression"]),
        "synthetic_accuracy_noninferior": mean_synthetic_accuracy
        >= float(synthetic_parent["eligible_accuracy"])
        - float(gates["maximum_synthetic_accuracy_regression"]),
        "synthetic_validity_noninferior": mean_synthetic_validity
        >= float(synthetic_parent["valid_output_rate"])
        - float(gates["maximum_synthetic_validity_regression"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "worst_source_net_improvement": worst_source,
        "mean_synthetic_accuracy": mean_synthetic_accuracy,
        "mean_synthetic_validity": mean_synthetic_validity,
    }


def run_external_weight_bridge_training(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_external_weight_bridge_contract(config)
    data = prepare_external_weight_bridge_data(config)
    report_path = _root(config) / "training.json"
    public_path = config.root / "reports" / "evolve" / "external-weight-bridge-v1-training.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("contract_fingerprint") != contract["fingerprint"]:
            raise RuntimeError("H19 completed result does not match the frozen contract")
        write_json(public_path, existing)
        return existing

    rows = _historical_rows(config)
    statuses: dict[str, Any] = {}
    fold_details: dict[str, list[dict[str, Any]]] = {}
    synthetic_fold_metrics: list[dict[str, Any]] = []
    retention_cases = list(read_jsonl(Path(data["synthetic_retention_path"])))
    _, adapter_paths = _expert_context(config)
    parent_fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "surface": "synthetic-parent",
            "parent": contract["parent"]["adapter_sha256"],
        }
    )
    synthetic_parent = _evaluate_adapter(
        config,
        fold="parent",
        adapter_path=adapter_paths["parent"],
        cases=retention_cases,
        surface="synthetic-retention",
        fingerprint=parent_fingerprint,
    )
    for fold in contract["fold_sources"]:
        status = _train_fold(config, fold=str(fold))
        statuses[str(fold)] = status
        held_out = [_evaluation_case(row) for row in rows if str(row["source_id"]) == fold]
        evaluation_fingerprint = canonical_hash(
            {
                "contract": contract["fingerprint"],
                "fold": fold,
                "adapter": status["adapter_sha256"],
                "held_out": canonical_hash(held_out),
            }
        )
        result = _evaluate_adapter(
            config,
            fold=str(fold),
            adapter_path=Path(status["adapter_path"]),
            cases=held_out,
            surface="held-out",
            fingerprint=evaluation_fingerprint,
        )
        fold_details[str(fold)] = result["details"]
        synthetic_result = _evaluate_adapter(
            config,
            fold=str(fold),
            adapter_path=Path(status["adapter_path"]),
            cases=retention_cases,
            surface="synthetic-retention",
            fingerprint=canonical_hash(
                {
                    "evaluation": evaluation_fingerprint,
                    "surface": "synthetic-retention",
                }
            ),
        )
        synthetic_fold_metrics.append(synthetic_result["metrics"])

    candidate_details = [
        row for fold in contract["fold_sources"] for row in fold_details[str(fold)]
    ]
    control_details = [
        {
            "case_id": row["case_id"],
            "eligible": True,
            "gold_method_id": row["gold_method_id"],
            "predicted_method_id": row["control_predicted_method_id"],
            "valid_output": row["control_valid_output"],
            "correct": row["control_correct"],
        }
        for row in rows
    ]
    paired = _paired_summary(control_details, candidate_details)
    source_paired: dict[str, Any] = {}
    for source in contract["fold_sources"]:
        ids = {str(row["case_id"]) for row in rows if str(row["source_id"]) == source}
        source_paired[str(source)] = _paired_summary(
            [row for row in control_details if str(row["case_id"]) in ids],
            [row for row in candidate_details if str(row["case_id"]) in ids],
        )
    control_metrics = _metrics(control_details)
    candidate_metrics = _metrics(candidate_details)
    gate = _gate_result(
        paired,
        source_paired,
        control_metrics,
        candidate_metrics,
        synthetic_parent["metrics"],
        synthetic_fold_metrics,
        dict(contract["settings"]["gates"]),
    )
    final_status = _train_fold(config, fold="all-sources") if gate["passed"] else None
    result: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "method": "H19 source-held-out external representation LoRA bridge training",
        "trainer_version": _TRAINER_VERSION,
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["data_fingerprint"],
        "fold_statuses": statuses,
        "scores": {
            "historical-menu-free-control": control_metrics,
            "source-held-out-lora": candidate_metrics,
            "synthetic-parent": synthetic_parent["metrics"],
            "synthetic-fold-mean": {
                "eligible_accuracy": float(
                    np.mean(
                        [float(value["eligible_accuracy"]) for value in synthetic_fold_metrics]
                    )
                ),
                "valid_output_rate": float(
                    np.mean(
                        [float(value["valid_output_rate"]) for value in synthetic_fold_metrics]
                    )
                ),
            },
        },
        "paired": paired,
        "source_paired": source_paired,
        "historical_gate": gate,
        "selected": bool(gate["passed"]),
        "final_all_source_adapter": final_status,
        "fresh_external_evidence": False,
        "champion_changed": False,
        "release_authorized": False,
        "next_step": (
            "freeze-h19-runtime-and-qualify-one-genuinely-new-external-source"
            if gate["passed"]
            else "preserve-h19-negative-and-retain-v0.3-menu-free-control"
        ),
        "claim_boundary": contract["claim_boundary"],
    }
    result["result_fingerprint"] = canonical_hash(result)
    write_json(report_path, result)
    write_json(public_path, result)
    return result

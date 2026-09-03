from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import ProjectConfig
from .forge_data import _tokenize_chat
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl
from .stats_agent import StatsAgent
from .stats_canonical_bottleneck import _registered_scenarios
from .stats_catalog import PROCEDURES
from .stats_catalog_distillation import _contract_scenarios
from .stats_catalog_grounding import _catalog_reference
from .stats_cross_format import _format_shift_case
from .stats_dgp import simulate_scenario
from .stats_eval import _append_progress, _load_progress, _normalize
from .stats_family_router import _expert_context
from .stats_router_replication import _historical_scenario_audit, _scenario_semantic_payload

_EVALUATOR_VERSION = 1


def _root(config: ProjectConfig) -> Path:
    return config.path_for("artifact_dir") / "catalog-ranking-v1"


def _data_root(config: ProjectConfig) -> Path:
    return config.path_for("evolution_dir") / "catalog-ranking-v1"


def _implementation_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "stats_catalog_ranking.py": sha256_file(Path(__file__)),
        "stats_catalog_grounding.py": sha256_file(root / "stats_catalog_grounding.py"),
        "stats_catalog_distillation.py": sha256_file(root / "stats_catalog_distillation.py"),
        "forge_data.py": sha256_file(root / "forge_data.py"),
        "stats_agent.py": sha256_file(root / "stats_agent.py"),
        "stats_catalog.py": sha256_file(root / "stats_catalog.py"),
    }


def _messages(case: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an auditable statistical method selector. Choose exactly one repository "
                "canonical method ID from the fixed catalog. Return only that method ID and no "
                "reasoning or additional text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{case['question']}\n\nRepository method catalog:\n{_catalog_reference()}"
            ),
        },
    ]


def _clean_generated_method(answer: str) -> str:
    value = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL | re.IGNORECASE).strip()
    value = re.sub(r"^```(?:text|json)?\s*|\s*```$", "", value, flags=re.IGNORECASE).strip()
    first_line = next((line.strip() for line in value.splitlines() if line.strip()), "")
    return first_line.strip("`'\"")


def _candidate_sequences(
    tokenizer: Any,
    prompt: list[dict[str, str]],
    candidate_ids: list[str],
) -> list[tuple[str, list[int], int]]:
    sequences: list[tuple[str, list[int], int]] = []
    for method_id in candidate_ids:
        tokens, offset = _tokenize_chat(
            tokenizer,
            [*prompt, {"role": "assistant", "content": method_id}],
        )
        if offset >= len(tokens):
            raise RuntimeError(f"H9 candidate has no assistant tokens: {method_id}")
        sequences.append((method_id, tokens, offset))
    return sequences


def _rank_catalog(
    model: Any,
    tokenizer: Any,
    prompt: list[dict[str, str]],
    candidate_ids: list[str],
    *,
    batch_size: int,
    max_seq_length: int,
) -> list[tuple[str, float]]:
    sequences = _candidate_sequences(tokenizer, prompt, candidate_ids)
    if max(len(tokens) for _, tokens, _ in sequences) > max_seq_length:
        raise RuntimeError("H9 ranking forbids truncation")
    scored: list[tuple[str, float]] = []
    for start in range(0, len(sequences), batch_size):
        batch_rows = sequences[start : start + batch_size]
        longest = max(len(tokens) for _, tokens, _ in batch_rows)
        batch = np.zeros((len(batch_rows), longest), dtype=np.int32)
        for row_index, (_, tokens, _) in enumerate(batch_rows):
            batch[row_index, : len(tokens)] = tokens
        inputs = mx.array(batch[:, :-1])
        targets = mx.array(batch[:, 1:])
        logits = model(inputs)
        losses = nn.losses.cross_entropy(logits, targets).astype(mx.float32)
        mx.eval(losses)
        observed = np.asarray(losses)
        for row_index, (method_id, tokens, offset) in enumerate(batch_rows):
            assistant_losses = observed[row_index, max(0, offset - 1) : len(tokens) - 1]
            if assistant_losses.size == 0:
                raise RuntimeError(f"H9 empty assistant loss span: {method_id}")
            scored.append((method_id, float(np.mean(assistant_losses))))
        del inputs, targets, logits, losses
    return sorted(scored, key=lambda item: (item[1], item[0]))


def prepare_catalog_ranking_contract(config: ProjectConfig) -> dict[str, Any]:
    settings = dict(config.section("catalog_ranking"))
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "contract.json"
    public_path = config.root / "reports" / "evolve" / "catalog-ranking-v1-contract.json"

    h8_path = config.root / "reports" / "evolve" / "catalog-distillation-v1-pilot.json"
    if not h8_path.exists():
        raise RuntimeError("H9 requires the closed H8 pilot")
    h8 = json.loads(h8_path.read_text(encoding="utf-8"))
    if h8.get("selected_arm") is not None or h8.get("confirmation_authorized"):
        raise RuntimeError("H8 did not close negatively; H9 is not authorized")
    h8_confirmation = (
        config.path_for("evolution_dir")
        / "catalog-distillation-v1"
        / "surfaces"
        / "confirmation_shard.jsonl"
    )
    if h8_confirmation.exists():
        raise RuntimeError("H8 confirmation was unexpectedly opened")

    h7_path = config.root / "reports" / "evolve" / "catalog-grounding-v1-confirmation.json"
    h7 = json.loads(h7_path.read_text(encoding="utf-8"))
    if not h7.get("synthetic_interface_confirmed"):
        raise RuntimeError("H9 requires the confirmed H7 fixed-catalog mechanism")
    h7_method_accuracy = float(h7["scores"]["catalog-grounded"]["method_set_accuracy"])

    _, adapter_paths = _expert_context(config)
    parent = adapter_paths["parent"]
    parent_sha = sha256_file(parent / "adapters.safetensors")
    registered: dict[str, Any] = {}
    all_scenarios = []
    seen: set[str] = set()
    for name in ("selection_shard", "confirmation_shard"):
        shard = dict(settings[name])
        scenarios = _registered_scenarios(shard, name=f"catalog-ranking:{name}")
        ids = {scenario.blueprint_id for scenario in scenarios}
        if ids & seen:
            raise RuntimeError(f"H9 blueprint overlap at {name}")
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

    previous_scenarios = (
        _contract_scenarios(config, "canonical-bottleneck-v1-contract.json")
        + _contract_scenarios(
            config,
            "catalog-grounding-v1-contract.json",
            name_prefix="catalog-grounding:",
        )
        + _contract_scenarios(
            config,
            "catalog-distillation-v1-contract.json",
            name_prefix="catalog-distillation:",
        )
    )
    prior_ids = {scenario.blueprint_id for scenario in previous_scenarios}
    prior_semantics = {
        canonical_hash(_scenario_semantic_payload(scenario.to_dict()))
        for scenario in previous_scenarios
    }
    new_ids = {scenario.blueprint_id for scenario in all_scenarios}
    new_semantics = {
        canonical_hash(_scenario_semantic_payload(scenario.to_dict())) for scenario in all_scenarios
    }
    if new_ids & prior_ids or new_semantics & prior_semantics:
        raise RuntimeError("H9 overlaps H6/H7/H8 registered blueprints or semantic points")

    audit = _historical_scenario_audit(
        config,
        all_scenarios,
        excluded_root=_data_root(config),
        minimum_normalized_distance=float(settings["minimum_normalized_distance"]),
    )
    if not audit["passed"]:
        raise RuntimeError("H9 blueprints failed historical-overlap audit")

    candidate_ids = [procedure.method_id for procedure in PROCEDURES]
    contract: dict[str, Any] = {
        "schema_version": 1,
        "method": "H9 fixed-catalog constrained likelihood ranking",
        "method_version": int(settings["method_version"]),
        "causal_question": (
            "With the fixed H7 catalog and unchanged v0.3 weights, does constrained likelihood "
            "ranking over all 28 canonical IDs recover fine-grained method discrimination that is "
            "lost during free generation?"
        ),
        "h8_negative_result_fingerprint": h8["result_fingerprint"],
        "h8_report_sha256": sha256_file(h8_path),
        "h7_confirmed_catalog_method_accuracy": h7_method_accuracy,
        "h7_confirmation_result_fingerprint": h7["result_fingerprint"],
        "parent": {
            "name": "v0.3.0-parent",
            "adapter_path": str(parent),
            "adapter_sha256": parent_sha,
        },
        "settings": settings,
        "blueprint_contracts": registered,
        "historical_overlap_audit": audit,
        "prior_registered_overlap": {
            "prior_blueprints": len(prior_ids),
            "h9_blueprints": len(new_ids),
            "blueprint_id_overlap_count": len(new_ids & prior_ids),
            "semantic_overlap_count": len(new_semantics & prior_semantics),
        },
        "catalog": {
            "procedure_count": len(candidate_ids),
            "method_id_sha256": canonical_hash(candidate_ids),
            "reference_sha256": canonical_hash(_catalog_reference()),
        },
        "scoring": {
            "candidate_space": "all 28 repository method IDs for every case",
            "score": "mean assistant-token negative log likelihood including assistant end token",
            "lower_is_better": True,
            "batch_size": int(settings["ranking_batch_size"]),
            "max_seq_length": int(settings["max_seq_length"]),
        },
        "implementation_sha256": _implementation_manifest(),
        "arms": {
            "catalog-free-generate": (
                "Same H9 method-only fixed-catalog prompt; temperature-0 decode"
            ),
            "catalog-likelihood-rank": (
                "Same prompt and weights; score every canonical ID and choose minimum mean NLL"
            ),
        },
        "selection_policy": (
            "Only catalog-likelihood-rank may advance and only if every registered method-accuracy "
            "gate passes. No H8 confirmation or historical external benchmark may be used."
        ),
        "confirmation_policy": (
            "Confirmation blueprints are registered now but remain unsimulated until selection "
            "passes."
        ),
        "claim_boundary": (
            "H9 can establish a constrained-decoding interface mechanism only. It does not change "
            "weights, promote the champion, or establish external capability."
        ),
        "selection_opened": False,
        "confirmation_simulations_opened": False,
        "external_benchmark_opened": False,
    }
    contract["fingerprint"] = canonical_hash(contract)
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("H9 contract is immutable")
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
            raise RuntimeError("H9 confirmation cannot open before selection")
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        if pilot.get("selected_interface") != "catalog-likelihood-rank":
            raise RuntimeError("H9 selection did not authorize confirmation")
    shard = dict(contract["settings"][name])
    scenarios = _registered_scenarios(shard, name=f"catalog-ranking:{name}")
    blueprint_sha = canonical_hash([scenario.to_dict() for scenario in scenarios])
    if blueprint_sha != contract["blueprint_contracts"][name]["blueprint_sha256"]:
        raise RuntimeError(f"H9 registered blueprints changed for {name}")
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
            raise RuntimeError(f"H9 {name} surface changed")
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


def prepare_catalog_ranking_data(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_catalog_ranking_contract(config)
    root = _data_root(config)
    status_path = root / "data-status.json"
    selection_manifest, selection_surface = _simulate_surface(
        config, contract, name="selection_shard"
    )
    cases = [_format_shift_case(simulation) for simulation in selection_surface]
    cases_path = root / "selection-format.jsonl"
    write_jsonl(cases_path, cases)
    fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "selection_surface": selection_manifest["fingerprint"],
            "cases": sha256_file(cases_path),
        }
    )
    status = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "contract_fingerprint": contract["fingerprint"],
        "selection_groups": len(selection_surface),
        "cases_sha256": sha256_file(cases_path),
        "confirmation_opened": False,
    }
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise RuntimeError("H9 prepared data changed")
        return existing
    write_json(status_path, status)
    write_json(config.root / "reports" / "evolve" / "catalog-ranking-v1-data.json", status)
    return status


def _evaluate_cases(
    config: ProjectConfig,
    cases: list[dict[str, Any]],
    *,
    progress_root: Path,
    evaluation_fingerprint: str,
) -> dict[str, Any]:
    candidate_ids = [procedure.method_id for procedure in PROCEDURES]
    normalized_to_id = {_normalize(method_id): method_id for method_id in candidate_ids}
    progress_path = progress_root / "paired.jsonl"
    fingerprint = canonical_hash(
        {
            "evaluation": evaluation_fingerprint,
            "evaluator_version": _EVALUATOR_VERSION,
            "candidate_ids": candidate_ids,
        }
    )
    cached = _load_progress(progress_path, fingerprint=fingerprint, id_field="case_id")
    _, adapter_paths = _expert_context(config)
    agent = StatsAgent(config, adapter_path=adapter_paths["parent"])
    agent.router.set_route("adapter")
    details: list[dict[str, Any]] = []
    try:
        for case in cases:
            case_id = str(case["case_id"])
            if case_id in cached:
                details.append(cached[case_id])
                continue
            prompt = _messages(case)
            generated = agent.answer_without_tools(
                prompt,
                route="stats",
                max_tokens=int(config.section("catalog_ranking")["generation_max_tokens"]),
                temperature=0.0,
            )
            free_method = _clean_generated_method(generated)
            ranked = _rank_catalog(
                agent.model,
                agent.tokenizer,
                prompt,
                candidate_ids,
                batch_size=int(config.section("catalog_ranking")["ranking_batch_size"]),
                max_seq_length=int(config.section("catalog_ranking")["max_seq_length"]),
            )
            ranked_method = ranked[0][0]
            gold = str(case["gold_methods"][0])
            gold_rank = 1 + next(index for index, item in enumerate(ranked) if item[0] == gold)
            row = {
                "case_id": case_id,
                "family_id": str(case["family_id"]),
                "gold_method": gold,
                "free_method": free_method,
                "free_catalog_valid": _normalize(free_method) in normalized_to_id,
                "free_correct": _normalize(free_method) == _normalize(gold),
                "ranked_method": ranked_method,
                "ranked_correct": ranked_method == gold,
                "gold_rank": gold_rank,
                "top5": [method_id for method_id, _ in ranked[:5]],
                "top5_mean_nll": [round(score, 6) for _, score in ranked[:5]],
            }
            details.append(row)
            _append_progress(
                progress_path,
                fingerprint=fingerprint,
                row=row,
                completed=len(details),
            )
    finally:
        del agent
        gc.collect()
        mx.clear_cache()
    count = len(details)
    free_correct = sum(bool(row["free_correct"]) for row in details)
    ranked_correct = sum(bool(row["ranked_correct"]) for row in details)
    valid = sum(bool(row["free_catalog_valid"]) for row in details)
    reciprocal_ranks = [1.0 / int(row["gold_rank"]) for row in details]
    return {
        "count": count,
        "free_method_accuracy": free_correct / count,
        "free_catalog_validity": valid / count,
        "ranked_method_accuracy": ranked_correct / count,
        "ranked_mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
        "ranked_mean_gold_rank": float(np.mean([int(row["gold_rank"]) for row in details])),
        "details": details,
    }


def _gate_report(
    *,
    scores: dict[str, Any],
    gates: dict[str, Any],
    h7_method_accuracy: float,
) -> dict[str, Any]:
    candidate = float(scores["ranked_method_accuracy"])
    control = float(scores["free_method_accuracy"])
    gain = 100 * (candidate - control)
    h7_gain = 100 * (candidate - h7_method_accuracy)
    checks = {
        "minimum_ranked_method_accuracy": candidate
        >= float(gates["minimum_ranked_method_accuracy"]),
        "minimum_gain_over_free_control": gain
        >= float(gates["minimum_gain_over_free_control_points"]),
        "minimum_gain_over_h7_confirmed": h7_gain
        >= float(gates["minimum_gain_over_h7_confirmed_points"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "effect_points": {
            "ranked_vs_free_control": gain,
            "ranked_vs_h7_confirmed_catalog": h7_gain,
        },
    }


def run_catalog_ranking_pilot(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_catalog_ranking_contract(config)
    data = prepare_catalog_ranking_data(config)
    cases = list(read_jsonl(_data_root(config) / "selection-format.jsonl"))
    evaluation_fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "data": data["fingerprint"],
            "parent": contract["parent"]["adapter_sha256"],
            "surface": "selection",
            "evaluator_version": _EVALUATOR_VERSION,
        }
    )
    scores = _evaluate_cases(
        config,
        cases,
        progress_root=_root(config) / "selection-progress",
        evaluation_fingerprint=evaluation_fingerprint,
    )
    gate = _gate_report(
        scores=scores,
        gates=dict(contract["settings"]["selection_gates"]),
        h7_method_accuracy=float(contract["h7_confirmed_catalog_method_accuracy"]),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": evaluation_fingerprint,
        "method": "H9 fixed-catalog constrained likelihood ranking pilot",
        "contract_fingerprint": contract["fingerprint"],
        "data_fingerprint": data["fingerprint"],
        "same_parent_weights": True,
        "scores": {key: value for key, value in scores.items() if key != "details"},
        "selection_gate": gate,
        "selected_interface": "catalog-likelihood-rank" if gate["passed"] else None,
        "confirmation_authorized": bool(gate["passed"]),
        "external_benchmark_authorized": False,
        "private_details": scores["details"],
    }
    report["result_fingerprint"] = canonical_hash(
        {key: value for key, value in report.items() if key != "private_details"}
    )
    write_json(_root(config) / "pilot.json", report)
    public = dict(report)
    public.pop("private_details")
    write_json(config.root / "reports" / "evolve" / "catalog-ranking-v1-pilot.json", public)
    return public


def run_catalog_ranking_confirmation(config: ProjectConfig) -> dict[str, Any]:
    contract = prepare_catalog_ranking_contract(config)
    pilot_path = _root(config) / "pilot.json"
    if not pilot_path.exists():
        raise RuntimeError("H9 pilot has not selected an interface")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("selected_interface") != "catalog-likelihood-rank":
        raise RuntimeError("H9 selection did not authorize confirmation")
    manifest, surface = _simulate_surface(config, contract, name="confirmation_shard")
    cases = [_format_shift_case(simulation) for simulation in surface]
    evaluation_fingerprint = canonical_hash(
        {
            "contract": contract["fingerprint"],
            "pilot": pilot["result_fingerprint"],
            "confirmation": manifest["fingerprint"],
            "parent": contract["parent"]["adapter_sha256"],
            "evaluator_version": _EVALUATOR_VERSION,
        }
    )
    scores = _evaluate_cases(
        config,
        cases,
        progress_root=_root(config) / "confirmation-progress",
        evaluation_fingerprint=evaluation_fingerprint,
    )
    gate = _gate_report(
        scores=scores,
        gates=dict(contract["settings"]["confirmation_gates"]),
        h7_method_accuracy=float(contract["h7_confirmed_catalog_method_accuracy"]),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": evaluation_fingerprint,
        "method": "H9 fixed-catalog constrained likelihood ranking confirmation",
        "contract_fingerprint": contract["fingerprint"],
        "pilot_result_fingerprint": pilot["result_fingerprint"],
        "confirmation_manifest_fingerprint": manifest["fingerprint"],
        "same_parent_weights": True,
        "scores": {key: value for key, value in scores.items() if key != "details"},
        "confirmation_gate": gate,
        "synthetic_ranking_interface_confirmed": bool(gate["passed"]),
        "external_benchmark_authorized": False,
        "next_step": (
            "preregister-independent-external-ranking-evidence"
            if gate["passed"]
            else "reject-h9-catalog-ranking"
        ),
        "private_details": scores["details"],
    }
    report["result_fingerprint"] = canonical_hash(
        {key: value for key, value in report.items() if key != "private_details"}
    )
    write_json(_root(config) / "confirmation.json", report)
    public = dict(report)
    public.pop("private_details")
    write_json(
        config.root / "reports" / "evolve" / "catalog-ranking-v1-confirmation.json",
        public,
    )
    return public

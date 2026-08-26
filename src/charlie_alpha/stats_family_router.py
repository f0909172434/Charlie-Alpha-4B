from __future__ import annotations

import copy
import gc
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from mlx_lm import load

from .config import ProjectConfig
from .io_utils import canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl
from .stats_calibrate import _surface_comparison
from .stats_catalog import FAMILIES
from .stats_cone import _cone_paths
from .stats_dgp import build_blueprints, simulate_scenario
from .stats_evolve import (
    _proposal_records,
    _score_adapter_surfaces,
    _score_loaded_selector,
)
from .stats_route import _aggregate_predictions
from .stats_training import _stats_snapshot

_AUDIT_PATTERNS = (
    re.compile(
        r"DGP audit values:.*?(?=(?:Choose the primary analysis|A collaborator wants))",
        flags=re.DOTALL,
    ),
    re.compile(r"DGP 稽核值：.*?。(?=請選擇)", flags=re.DOTALL),
    re.compile(r"DGP 审核值：.*?。(?=请选择)", flags=re.DOTALL),
)


def _router_paths(config: ProjectConfig) -> Path:
    return _cone_paths(config)[1] / "family-router"


def router_question(record: dict[str, Any]) -> str:
    messages = list(record["messages"])
    user = next(message for message in messages if message.get("role") == "user")
    text = str(user["content"]).split("\n\nCandidate menu:", 1)[0]
    for pattern in _AUDIT_PATTERNS:
        text = pattern.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if "Candidate menu:" in text or "DGP audit values:" in text or "DGP 稽核值：" in text:
        raise RuntimeError("Router text retained answer-menu or DGP-audit leakage")
    return text


def _normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower()
    value = re.sub(r"\d+(?:[.,]\d+)?", "0", value)
    value = re.sub(r"\s+", " ", value).strip()
    return f"^{value}$"


def character_ngrams(text: str, minimum: int, maximum: int) -> Counter[str]:
    if minimum < 1 or maximum < minimum:
        raise ValueError("Character n-gram bounds are invalid")
    normalized = _normalize_text(text)
    features: Counter[str] = Counter()
    for size in range(minimum, maximum + 1):
        features.update(
            normalized[index : index + size] for index in range(len(normalized) - size + 1)
        )
    return features


@dataclass
class CharacterNgramNB:
    classes: list[str]
    vocabulary: list[str]
    feature_log_prob: np.ndarray
    class_log_prior: np.ndarray
    ngram_min: int
    ngram_max: int
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.feature_log_prob.shape != (len(self.classes), len(self.vocabulary)):
            raise ValueError("Router feature matrix shape does not match its labels")
        if self.class_log_prior.shape != (len(self.classes),):
            raise ValueError("Router class-prior shape does not match its labels")
        if self.temperature <= 0:
            raise ValueError("Router temperature must be positive")
        self._indices = {feature: index for index, feature in enumerate(self.vocabulary)}

    @classmethod
    def fit(
        cls,
        examples: list[dict[str, str]],
        *,
        ngram_min: int,
        ngram_max: int,
        vocabulary_size: int,
        minimum_document_frequency: int,
        alpha: float,
        uniform_class_prior: bool,
    ) -> CharacterNgramNB:
        if not examples:
            raise ValueError("Router training requires labeled examples")
        if vocabulary_size < 1 or minimum_document_frequency < 1 or alpha <= 0:
            raise ValueError("Router vocabulary and smoothing settings must be positive")
        classes = sorted({str(example["family_id"]) for example in examples})
        expected = sorted(family.family_id for family in FAMILIES)
        if classes != expected:
            raise ValueError("Router training must cover all registered DGP families")
        document_frequency: Counter[str] = Counter()
        cached: list[Counter[str]] = []
        for example in examples:
            features = character_ngrams(str(example["text"]), ngram_min, ngram_max)
            cached.append(features)
            document_frequency.update(features.keys())
        vocabulary = [
            feature
            for feature, count in sorted(
                document_frequency.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if count >= minimum_document_frequency
        ][:vocabulary_size]
        if not vocabulary:
            raise RuntimeError("Router vocabulary is empty")
        indices = {feature: index for index, feature in enumerate(vocabulary)}
        class_indices = {family: index for index, family in enumerate(classes)}
        counts = np.full((len(classes), len(vocabulary)), float(alpha), dtype=np.float64)
        class_documents = np.zeros(len(classes), dtype=np.float64)
        for example, features in zip(examples, cached, strict=True):
            class_index = class_indices[str(example["family_id"])]
            class_documents[class_index] += 1.0
            for feature, count in features.items():
                feature_index = indices.get(feature)
                if feature_index is not None:
                    counts[class_index, feature_index] += min(float(count), 3.0)
        feature_log_prob = np.log(counts / counts.sum(axis=1, keepdims=True))
        class_log_prior = (
            np.full(len(classes), -math.log(len(classes)), dtype=np.float64)
            if uniform_class_prior
            else np.log(class_documents / class_documents.sum())
        )
        return cls(
            classes=classes,
            vocabulary=vocabulary,
            feature_log_prob=feature_log_prob.astype(np.float32),
            class_log_prior=class_log_prior.astype(np.float32),
            ngram_min=ngram_min,
            ngram_max=ngram_max,
        )

    def log_scores(self, text: str) -> np.ndarray:
        features = character_ngrams(text, self.ngram_min, self.ngram_max)
        scores = self.class_log_prior.astype(np.float64).copy()
        observed = 0.0
        for feature, count in features.items():
            index = self._indices.get(feature)
            if index is None:
                continue
            weight = min(float(count), 3.0)
            scores += weight * self.feature_log_prob[:, index]
            observed += weight
        if observed <= 0:
            return self.class_log_prior.astype(np.float64)
        likelihood = scores - self.class_log_prior
        return self.class_log_prior.astype(np.float64) + likelihood / math.sqrt(observed)

    def probabilities(self, text: str, *, temperature: float | None = None) -> np.ndarray:
        effective_temperature = float(temperature or self.temperature)
        if effective_temperature <= 0:
            raise ValueError("Router temperature must be positive")
        logits = self.log_scores(text) / effective_temperature
        logits -= float(np.max(logits))
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum()

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        probabilities = self.probabilities(text)
        index = int(np.argmax(probabilities))
        return (
            self.classes[index],
            float(probabilities[index]),
            {family: float(probabilities[i]) for i, family in enumerate(self.classes)},
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            classes=np.asarray(self.classes),
            vocabulary=np.asarray(self.vocabulary),
            feature_log_prob=self.feature_log_prob,
            class_log_prior=self.class_log_prior,
            ngram_min=np.asarray([self.ngram_min], dtype=np.int32),
            ngram_max=np.asarray([self.ngram_max], dtype=np.int32),
            temperature=np.asarray([self.temperature], dtype=np.float64),
        )

    @classmethod
    def load(cls, path: Path) -> CharacterNgramNB:
        with np.load(path, allow_pickle=False) as value:
            return cls(
                classes=[str(item) for item in value["classes"].tolist()],
                vocabulary=[str(item) for item in value["vocabulary"].tolist()],
                feature_log_prob=np.asarray(value["feature_log_prob"], dtype=np.float32),
                class_log_prior=np.asarray(value["class_log_prior"], dtype=np.float32),
                ngram_min=int(value["ngram_min"][0]),
                ngram_max=int(value["ngram_max"][0]),
                temperature=float(value["temperature"][0]),
            )


def choose_temperature(
    model: CharacterNgramNB,
    examples: list[dict[str, str]],
    temperatures: list[float],
) -> dict[str, Any]:
    if not examples or not temperatures:
        raise ValueError("Temperature calibration requires examples and candidates")
    class_indices = {family: index for index, family in enumerate(model.classes)}
    candidates: list[dict[str, Any]] = []
    for temperature in temperatures:
        if temperature <= 0:
            raise ValueError("Temperature candidates must be positive")
        losses: list[float] = []
        correct = 0
        confidences: list[float] = []
        for example in examples:
            probabilities = model.probabilities(str(example["text"]), temperature=temperature)
            truth = class_indices[str(example["family_id"])]
            losses.append(-math.log(max(float(probabilities[truth]), 1e-12)))
            correct += int(int(np.argmax(probabilities)) == truth)
            confidences.append(float(np.max(probabilities)))
        candidates.append(
            {
                "temperature": float(temperature),
                "negative_log_likelihood": float(np.mean(losses)),
                "accuracy": correct / len(examples),
                "mean_confidence": float(np.mean(confidences)),
            }
        )
    return min(
        candidates,
        key=lambda value: (
            float(value["negative_log_likelihood"]),
            -float(value["accuracy"]),
            float(value["temperature"]),
        ),
    ) | {"candidates": candidates}


def _ensure_router_shard(
    config: ProjectConfig,
    root: Path,
    settings_key: str,
    *,
    open_if_missing: bool,
    section_name: str = "family_router",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    settings = dict(config.section(section_name)[settings_key])
    split = str(settings["split"])
    count = int(settings["count"])
    seed = int(settings["seed"])
    stem = settings_key.removesuffix("_shard")
    path = root / f"{stem}.jsonl"
    manifest_path = root / f"{stem}-manifest.json"
    scenarios = build_blueprints({split: count}, seed=seed, active_search=False)
    simulation_settings = config.section("stats_data")
    fingerprint = canonical_hash(
        {
            "settings": settings,
            "scenarios": [scenario.to_dict() for scenario in scenarios],
            "simulation": {
                key: simulation_settings[key]
                for key in (
                    "initial_repetitions",
                    "escalation_repetitions",
                    "ranking_uncertainty_margin",
                    "regret_temperature",
                )
            },
            "generator_version": 1,
        }
    )
    if path.exists() or manifest_path.exists():
        if not path.exists() or not manifest_path.exists():
            raise RuntimeError(f"The immutable family-router {stem} shard is incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("fingerprint") == fingerprint
            and manifest.get("sha256") == sha256_file(path)
            and int(manifest.get("count", 0)) == count
        ):
            return manifest, list(read_jsonl(path))
        raise RuntimeError(f"The family-router {stem} shard is immutable")
    if not open_if_missing:
        raise RuntimeError(f"The family-router {stem} shard is sealed")
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
    write_jsonl(path, simulations)
    manifest = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "split": split,
        "seed": seed,
        "count": count,
        "view": str(settings["view"]),
        "sha256": sha256_file(path),
        "used_for_router_training": settings_key == "train_shard",
        "used_for_temperature_calibration": settings_key == "calibration_shard",
        "used_for_threshold_selection": settings_key == "validation_shard",
        "used_for_confirmation": settings_key == "confirmation_shard",
        "used_for_promotion": settings_key == "promotion_shard",
        "used_for_expert_training": False,
        "promotion_surface_opened": False,
        "final_surface_opened": False,
        "immutable": True,
    }
    write_json(manifest_path, manifest)
    return manifest, simulations


def _router_examples(
    simulations: list[dict[str, Any]],
    *,
    view: str,
) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for language in ("en", "zh_Hant", "zh_Hans"):
        effective_view = "boundary_a" if view == "scoring" and language == "en" else view
        if view == "scoring" and language != "en":
            effective_view = "standard"
        records = _proposal_records(simulations, language=language, view=effective_view)
        for record, simulation in records:
            examples.append(
                {
                    "blueprint_id": str(simulation["scenario"]["blueprint_id"]),
                    "language": language,
                    "family_id": str(simulation["scenario"]["family_id"]),
                    "text": router_question(record),
                }
            )
    return examples


def _classification_metrics(
    model: CharacterNgramNB,
    examples: list[dict[str, str]],
) -> dict[str, Any]:
    by_language: dict[str, list[bool]] = defaultdict(list)
    confidences: list[float] = []
    for example in examples:
        family, confidence, _ = model.predict(str(example["text"]))
        by_language[str(example["language"])].append(family == example["family_id"])
        confidences.append(confidence)
    language_accuracy = {
        language: float(np.mean(values)) for language, values in sorted(by_language.items())
    }
    return {
        "count": len(examples),
        "accuracy": float(np.mean([value for values in by_language.values() for value in values])),
        "language_accuracy": language_accuracy,
        "mean_confidence": float(np.mean(confidences)),
    }


def prepare_family_router(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    settings = dict(config.section("family_router"))
    root = _router_paths(config)
    root.mkdir(parents=True, exist_ok=True)
    if force and (root / "confirmation-manifest.json").exists():
        raise RuntimeError("Cannot replace the router after confirmation was opened")
    manifests: dict[str, dict[str, Any]] = {}
    surfaces: dict[str, list[dict[str, Any]]] = {}
    for key in ("train_shard", "calibration_shard", "validation_shard"):
        manifests[key], surfaces[key] = _ensure_router_shard(
            config,
            root,
            key,
            open_if_missing=True,
        )
    canary_path = config.root / str(settings["canary_path"])
    fingerprint = canonical_hash(
        {
            "settings": {
                key: settings[key]
                for key in (
                    "ngram_min",
                    "ngram_max",
                    "vocabulary_size",
                    "minimum_document_frequency",
                    "alpha",
                    "uniform_class_prior",
                    "temperature_grid",
                )
            },
            "surfaces": {key: value["fingerprint"] for key, value in manifests.items()},
            "canary_sha256": sha256_file(canary_path),
            "trainer_version": 1,
        }
    )
    status_path = root / "model.json"
    model_path = root / "router.npz"
    if status_path.exists() and model_path.exists() and not force:
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if (
            existing.get("complete")
            and existing.get("fingerprint") == fingerprint
            and existing.get("model_sha256") == sha256_file(model_path)
        ):
            return existing
        raise RuntimeError(
            "Family-router model fingerprint changed; use --force before confirmation"
        )
    train_examples = _router_examples(
        surfaces["train_shard"],
        view=str(settings["train_shard"]["view"]),
    )
    calibration_examples = _router_examples(
        surfaces["calibration_shard"],
        view=str(settings["calibration_shard"]["view"]),
    )
    validation_examples = _router_examples(
        surfaces["validation_shard"],
        view=str(settings["validation_shard"]["view"]),
    )
    model = CharacterNgramNB.fit(
        train_examples,
        ngram_min=int(settings["ngram_min"]),
        ngram_max=int(settings["ngram_max"]),
        vocabulary_size=int(settings["vocabulary_size"]),
        minimum_document_frequency=int(settings["minimum_document_frequency"]),
        alpha=float(settings["alpha"]),
        uniform_class_prior=bool(settings["uniform_class_prior"]),
    )
    calibration = choose_temperature(
        model,
        calibration_examples,
        [float(value) for value in settings["temperature_grid"]],
    )
    model.temperature = float(calibration["temperature"])
    model.save(model_path)
    status = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": fingerprint,
        "method": "character n-gram multinomial Naive Bayes with temperature scaling",
        "model_sha256": sha256_file(model_path),
        "classes": model.classes,
        "vocabulary_size": len(model.vocabulary),
        "ngram_range": [model.ngram_min, model.ngram_max],
        "train_examples": len(train_examples),
        "calibration_examples": len(calibration_examples),
        "validation_examples": len(validation_examples),
        "temperature_calibration": calibration,
        "calibration_metrics": _classification_metrics(model, calibration_examples),
        "validation_classification_metrics": _classification_metrics(model, validation_examples),
        "surface_manifests": manifests,
        "confirmation_shard_opened": (root / "confirmation-manifest.json").exists(),
        "promotion_shard_opened": False,
        "sealed_final_surface_opened": False,
    }
    write_json(status_path, status)
    return status


def _expert_context(config: ProjectConfig) -> tuple[dict[str, Any], dict[str, Path]]:
    cone_root = _cone_paths(config)[1]
    report_path = cone_root / "family-experts" / "report.json"
    if not report_path.exists():
        raise RuntimeError("Real routing requires the confirmed family-expert oracle")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("proceed_to_router_implementation"):
        raise RuntimeError("The family-expert oracle did not authorize a real router")
    mapping = report["selection"]["mapping"]
    first_family = next(iter(mapping))
    first_status = json.loads(
        (cone_root / "family-experts" / first_family / "status.json").read_text(encoding="utf-8")
    )
    parent = Path(str(first_status["parent_adapter_path"]))
    paths: dict[str, Path] = {"parent": parent}
    for family_id, route in mapping.items():
        if route["checkpoint_name"] != "parent":
            paths[str(route["slug"])] = (
                cone_root / "family-experts" / "selected-adapters" / family_id
            )
    return report, paths


def _score_router_adapters(
    config: ProjectConfig,
    simulations: list[dict[str, Any]],
    manifest: dict[str, Any],
    adapter_paths: dict[str, Path],
    root: Path,
) -> dict[str, dict[str, Any]]:
    score_root = root / "validation-scores"
    scores: dict[str, dict[str, Any]] = {}
    for slug, adapter_path in sorted(adapter_paths.items()):
        fingerprint = canonical_hash(
            {
                "surface": manifest["fingerprint"],
                "adapter": sha256_file(adapter_path / "adapters.safetensors"),
                "evaluator_version": 1,
            }
        )
        status_path = score_root / f"{slug}.json"
        if status_path.exists():
            existing = json.loads(status_path.read_text(encoding="utf-8"))
            if existing.get("fingerprint") == fingerprint and existing.get("complete"):
                scores[slug] = existing["score"]
                continue
            raise RuntimeError(f"Router validation score changed for {slug}")
        score = _score_adapter_surfaces(
            config,
            adapter_path,
            {str(manifest["split"]): simulations},
            include_retention=False,
        )[str(manifest["split"])]
        write_json(
            status_path,
            {
                "schema_version": 1,
                "complete": True,
                "fingerprint": fingerprint,
                "adapter_sha256": sha256_file(adapter_path / "adapters.safetensors"),
                "surface_fingerprint": manifest["fingerprint"],
                "score": score,
            },
        )
        scores[slug] = score
    return scores


def _route_documents(
    model: Any,
    simulations: list[dict[str, Any]],
    *,
    view: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    decisions: dict[tuple[str, str], dict[str, Any]] = {}
    for example in _router_examples(simulations, view=view):
        family, confidence, probabilities = model.predict(example["text"])
        key = (example["language"], example["blueprint_id"])
        decisions[key] = {
            **example,
            "predicted_family_id": family,
            "confidence": confidence,
            "probabilities": probabilities,
        }
    return decisions


def _route_slug(
    decision: dict[str, Any],
    expert_mapping: dict[str, dict[str, Any]],
    threshold: float,
) -> str:
    family_id = str(decision["predicted_family_id"])
    if family_id not in expert_mapping:
        return "parent"
    route = expert_mapping[family_id]
    if route["checkpoint_name"] == "parent" or float(decision["confidence"]) < threshold:
        return "parent"
    return str(route["slug"])


def _routed_score_from_cached(
    scores: dict[str, dict[str, Any]],
    decisions: dict[tuple[str, str], dict[str, Any]],
    expert_mapping: dict[str, dict[str, Any]],
    *,
    threshold: float,
    retention: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prediction_maps = {
        slug: {
            language: {
                str(prediction["blueprint_id"]): prediction for prediction in result["predictions"]
            }
            for language, result in score["languages"].items()
        }
        for slug, score in scores.items()
    }
    language_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    routed_to_expert = 0
    wrong_expert = 0
    family_correct: dict[str, list[bool]] = defaultdict(list)
    for (language, blueprint_id), decision in sorted(decisions.items()):
        slug = _route_slug(decision, expert_mapping, threshold)
        language_predictions[language].append(prediction_maps[slug][language][blueprint_id])
        is_correct = decision["predicted_family_id"] == decision["family_id"]
        family_correct[language].append(is_correct)
        if slug != "parent":
            routed_to_expert += 1
            wrong_expert += int(not is_correct)
    languages = {
        language: _aggregate_predictions(predictions)
        for language, predictions in sorted(language_predictions.items())
    }
    count = len(decisions)
    route_metrics = {
        "count": count,
        "family_accuracy": float(
            np.mean([value for values in family_correct.values() for value in values])
        ),
        "language_family_accuracy": {
            language: float(np.mean(values)) for language, values in sorted(family_correct.items())
        },
        "expert_coverage": routed_to_expert / count,
        "wrong_expert_rate": wrong_expert / count,
        "wrong_expert_count": wrong_expert,
    }
    return (
        {"selector": languages["en"], "languages": languages, "retention": retention},
        route_metrics,
    )


def _canary_metrics(
    model: Any,
    rows: list[dict[str, Any]],
    expert_mapping: dict[str, dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    route_correct: dict[str, list[bool]] = defaultdict(list)
    family_correct: dict[str, list[bool]] = defaultdict(list)
    wrong_expert = 0
    for row in rows:
        predicted, confidence, _ = model.predict(str(row["text"]))
        decision = {"predicted_family_id": predicted, "confidence": confidence}
        actual_route = _route_slug(decision, expert_mapping, threshold)
        truth = row.get("family_id")
        desired_route = (
            "parent"
            if truth is None or expert_mapping[str(truth)]["checkpoint_name"] == "parent"
            else str(expert_mapping[str(truth)]["slug"])
        )
        language = str(row["language"])
        route_correct[language].append(actual_route == desired_route)
        if truth is not None:
            family_correct[language].append(predicted == truth)
            wrong_expert += int(actual_route != "parent" and predicted != truth)
        else:
            wrong_expert += int(actual_route != "parent")
    return {
        "count": len(rows),
        "route_accuracy": float(
            np.mean([value for values in route_correct.values() for value in values])
        ),
        "language_route_accuracy": {
            language: float(np.mean(values)) for language, values in sorted(route_correct.items())
        },
        "family_accuracy": float(
            np.mean([value for values in family_correct.values() for value in values])
        ),
        "language_family_accuracy": {
            language: float(np.mean(values)) for language, values in sorted(family_correct.items())
        },
        "wrong_expert_rate": wrong_expert / len(rows),
    }


def _router_gate_results(
    comparison: dict[str, Any],
    route_metrics: dict[str, Any],
    canary: dict[str, Any],
    gates: dict[str, Any],
    *,
    minimum_improvement_key: str,
) -> dict[str, bool]:
    return {
        **{f"model_{key}": bool(value) for key, value in comparison["gates"].items()},
        "relative_regret": float(comparison["trilingual_relative_regret_improvement"])
        >= float(gates[minimum_improvement_key]),
        "router_family_accuracy": float(route_metrics["family_accuracy"])
        >= float(gates["minimum_router_family_accuracy"]),
        "router_language_accuracy": all(
            float(value) >= float(gates["minimum_language_router_accuracy"])
            for value in route_metrics["language_family_accuracy"].values()
        ),
        "wrong_expert_rate": float(route_metrics["wrong_expert_rate"])
        <= float(gates["maximum_wrong_expert_rate"]),
        "expert_coverage": float(route_metrics["expert_coverage"])
        >= float(gates["minimum_expert_coverage"]),
        "canary_route_accuracy": float(canary["route_accuracy"])
        >= float(gates["minimum_canary_route_accuracy"]),
        "canary_language_route_accuracy": all(
            float(value) >= float(gates["minimum_language_canary_route_accuracy"])
            for value in canary["language_route_accuracy"].values()
        ),
    }


def choose_router_threshold(comparisons: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [value for value in comparisons if all(value["gates"].values())]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda value: (
            -float(value["comparison"]["trilingual_relative_regret_improvement"]),
            float(value["route_metrics"]["wrong_expert_rate"]),
            float(value["comparison"]["candidate_invalidity"]),
            -float(value["comparison"]["candidate_accuracy"]),
            -float(value["threshold"]),
        ),
    )


def _score_real_route(
    config: ProjectConfig,
    model: Any,
    simulations: list[dict[str, Any]],
    *,
    view: str,
    expert_mapping: dict[str, dict[str, Any]],
    adapter_paths: dict[str, Path],
    threshold: float,
    retention: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    decisions = _route_documents(model, simulations, view=view)
    routed_rows: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    simulation_by_id = {
        str(simulation["scenario"]["blueprint_id"]): simulation for simulation in simulations
    }
    for (language, blueprint_id), decision in decisions.items():
        slug = _route_slug(decision, expert_mapping, threshold)
        routed_rows[slug][language].append(simulation_by_id[blueprint_id])
    language_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for slug, by_language in sorted(routed_rows.items()):
        model_instance, tokenizer = load(
            _stats_snapshot(config),
            adapter_path=str(adapter_paths[slug]),
            tokenizer_config={"trust_remote_code": True},
        )
        for language, rows in sorted(by_language.items()):
            view_name = "boundary_a" if language == "en" else "standard"
            result = _score_loaded_selector(
                model_instance,
                tokenizer,
                _proposal_records(rows, language=language, view=view_name),
            )
            language_predictions[language].extend(result["predictions"])
        del model_instance, tokenizer
        gc.collect()
    languages = {
        language: _aggregate_predictions(predictions)
        for language, predictions in sorted(language_predictions.items())
    }
    # Route metrics depend only on decisions, not on adapter outputs.
    family_correct: dict[str, list[bool]] = defaultdict(list)
    expert_count = 0
    wrong_count = 0
    for decision in decisions.values():
        correct = decision["predicted_family_id"] == decision["family_id"]
        family_correct[str(decision["language"])].append(correct)
        slug = _route_slug(decision, expert_mapping, threshold)
        if slug != "parent":
            expert_count += 1
            wrong_count += int(not correct)
    route_metrics = {
        "count": len(decisions),
        "family_accuracy": float(
            np.mean([value for values in family_correct.values() for value in values])
        ),
        "language_family_accuracy": {
            language: float(np.mean(values)) for language, values in sorted(family_correct.items())
        },
        "expert_coverage": expert_count / len(decisions),
        "wrong_expert_rate": wrong_count / len(decisions),
        "wrong_expert_count": wrong_count,
    }
    return (
        {"selector": languages["en"], "languages": languages, "retention": retention},
        route_metrics,
    )


def _paired_bootstrap(
    parent: dict[str, Any],
    candidate: dict[str, Any],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    parent_by_id: dict[str, list[float]] = defaultdict(list)
    candidate_by_id: dict[str, list[float]] = defaultdict(list)
    for result in parent["languages"].values():
        for prediction in result["predictions"]:
            parent_by_id[str(prediction["blueprint_id"])].append(
                float(prediction["normalized_regret"])
            )
    for result in candidate["languages"].values():
        for prediction in result["predictions"]:
            candidate_by_id[str(prediction["blueprint_id"])].append(
                float(prediction["normalized_regret"])
            )
    if set(parent_by_id) != set(candidate_by_id):
        raise RuntimeError("Router bootstrap coverage differs between parent and candidate")
    differences = np.asarray(
        [
            np.mean(parent_by_id[key]) - np.mean(candidate_by_id[key])
            for key in sorted(parent_by_id)
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled = rng.integers(0, len(differences), size=len(differences))
        draws[index] = float(np.mean(differences[sampled]))
    return {
        "mean_regret_improvement": float(np.mean(differences)),
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
    }


def _public_router_report(report: dict[str, Any]) -> dict[str, Any]:
    public = copy.deepcopy(report)
    return public


def run_family_router(
    config: ProjectConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    settings = dict(config.section("family_router"))
    gates = dict(settings["gates"])
    root = _router_paths(config)
    model_status = prepare_family_router(config, force=False)
    model_path = root / "router.npz"
    model = CharacterNgramNB.load(model_path)
    expert_report, adapter_paths = _expert_context(config)
    expert_mapping = expert_report["selection"]["mapping"]
    validation_manifest, validation_rows = _ensure_router_shard(
        config,
        root,
        "validation_shard",
        open_if_missing=True,
    )
    scores = _score_router_adapters(
        config,
        validation_rows,
        validation_manifest,
        adapter_paths,
        root,
    )
    retention_status = json.loads(
        (_cone_paths(config)[1] / "delta-calibration" / "scale-0p00" / "status.json").read_text(
            encoding="utf-8"
        )
    )
    retention = retention_status["scores"]["valid"]["retention"]
    parent_score = {
        **scores["parent"],
        "retention": retention,
    }
    decisions = _route_documents(
        model,
        validation_rows,
        view=str(settings["validation_shard"]["view"]),
    )
    canary_rows = list(read_jsonl(config.root / str(settings["canary_path"])))
    comparisons: list[dict[str, Any]] = []
    for threshold_value in settings["confidence_thresholds"]:
        threshold = float(threshold_value)
        candidate, route_metrics = _routed_score_from_cached(
            scores,
            decisions,
            expert_mapping,
            threshold=threshold,
            retention=retention,
        )
        comparison = _surface_comparison(parent_score, candidate, gates)
        canary = _canary_metrics(
            model,
            canary_rows,
            expert_mapping,
            threshold=threshold,
        )
        comparisons.append(
            {
                "threshold": threshold,
                "comparison": comparison,
                "route_metrics": route_metrics,
                "canary": canary,
                "gates": _router_gate_results(
                    comparison,
                    route_metrics,
                    canary,
                    gates,
                    minimum_improvement_key="minimum_validation_relative_improvement",
                ),
            }
        )
    selected = choose_router_threshold(comparisons)
    selection_fingerprint = canonical_hash(
        {
            "settings": settings,
            "model": model_status["fingerprint"],
            "model_sha256": model_status["model_sha256"],
            "expert_selection": expert_report["selection"]["fingerprint"],
            "validation": validation_manifest["fingerprint"],
            "canary": sha256_file(config.root / str(settings["canary_path"])),
            "comparisons": comparisons,
            "selector_version": 1,
        }
    )
    confirmation_exists = (root / "confirmation-manifest.json").exists()
    selection = {
        "schema_version": 1,
        "complete": True,
        "fingerprint": selection_fingerprint,
        "thresholds": comparisons,
        "selected_threshold": float(selected["threshold"]) if selected else None,
        "passed": selected is not None,
        "confirmation_shard_opened": confirmation_exists,
        "promotion_shard_opened": False,
        "sealed_final_surface_opened": False,
    }
    selection_path = root / "selection.json"
    if selection_path.exists() and confirmation_exists:
        existing = json.loads(selection_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != selection_fingerprint:
            raise RuntimeError("Family-router selection changed after confirmation was opened")
    write_json(selection_path, selection)
    report_path = root / "report.json"
    public_path = config.root / "reports" / "evolve" / "family-router.json"
    base_report = {
        "schema_version": 1,
        "complete": True,
        "method": "DGP-Regret selective family-expert router",
        "model": model_status,
        "expert_oracle_fingerprint": expert_report["fingerprint"],
        "selection": selection,
        "promotion_shard_opened": False,
        "sealed_final_surface_opened": False,
        "claim_boundary": (
            "This is a learned router over confirmed family LoRAs. Synthetic DGP confirmation "
            "and a small manual canary do not establish general real-world routing reliability."
        ),
    }
    if report_path.exists() and not force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            existing.get("complete")
            and existing.get("selection", {}).get("fingerprint") == selection_fingerprint
        ):
            existing["selection"] = selection
            write_json(report_path, existing)
            write_json(public_path, _public_router_report(existing))
            return existing
    if selected is None:
        report = {
            **base_report,
            "fingerprint": canonical_hash(
                {"selection": selection_fingerprint, "evaluator_version": 1}
            ),
            "confirmation": None,
            "proceed_to_promotion": False,
        }
        write_json(report_path, report)
        write_json(public_path, _public_router_report(report))
        return report

    confirmation_manifest, confirmation_rows = _ensure_router_shard(
        config,
        root,
        "confirmation_shard",
        open_if_missing=True,
    )
    selection["confirmation_shard_opened"] = True
    write_json(selection_path, selection)
    split = str(confirmation_manifest["split"])
    parent_confirmation = _score_adapter_surfaces(
        config,
        adapter_paths["parent"],
        {split: confirmation_rows},
    )[split]
    threshold = float(selected["threshold"])
    routed_confirmation, route_metrics = _score_real_route(
        config,
        model,
        confirmation_rows,
        view=str(settings["confirmation_shard"]["view"]),
        expert_mapping=expert_mapping,
        adapter_paths=adapter_paths,
        threshold=threshold,
        retention=parent_confirmation["retention"],
    )
    comparison = _surface_comparison(parent_confirmation, routed_confirmation, gates)
    canary = _canary_metrics(model, canary_rows, expert_mapping, threshold=threshold)
    gate_results = _router_gate_results(
        comparison,
        route_metrics,
        canary,
        gates,
        minimum_improvement_key="minimum_confirmation_relative_improvement",
    )
    bootstrap = _paired_bootstrap(
        parent_confirmation,
        routed_confirmation,
        repetitions=int(gates["bootstrap_repetitions"]),
        seed=int(config.section("project")["seed"]),
    )
    gate_results["paired_bootstrap"] = float(bootstrap["ci95_lower"]) >= float(
        gates["bootstrap_ci_lower_floor"]
    )
    passed = all(gate_results.values())
    report = {
        **base_report,
        "fingerprint": canonical_hash(
            {
                "selection": selection_fingerprint,
                "confirmation": confirmation_manifest["fingerprint"],
                "evaluator_version": 1,
            }
        ),
        "selection": selection,
        "confirmation": {
            "manifest": confirmation_manifest,
            "comparison": comparison,
            "route_metrics": route_metrics,
            "canary": canary,
            "paired_bootstrap": bootstrap,
            "gates": gate_results,
            "passed": passed,
        },
        "proceed_to_promotion": passed,
    }
    write_json(report_path, report)
    write_json(public_path, _public_router_report(report))
    return report

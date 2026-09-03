from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .io_utils import canonical_hash
from .stats_catalog import FAMILIES, FAMILY_BY_ID, PROCEDURE_BY_ID, DGPFamily


@dataclass(frozen=True)
class Scenario:
    blueprint_id: str
    family_id: str
    split: str
    seed: int
    parameters: dict[str, float]
    boundary_round: int
    domain: str
    search: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.search is None:
            payload.pop("search")
        return payload


@dataclass(frozen=True)
class OperatingPoint:
    method_id: str
    valid: bool
    type1_error: float
    coverage: float
    bias: float
    rmse: float
    power: float
    calibration_error: float
    cost: float
    invalidity: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(max(float(value), lower), upper)


def _logistic(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _scaled(parameters: dict[str, float], family: DGPFamily, key: str) -> float:
    lower, upper = family.parameters[key]
    if upper == lower:
        return 0.0
    return _clip((parameters[key] - lower) / (upper - lower))


def latin_hypercube(
    family: DGPFamily,
    count: int,
    *,
    seed: int,
) -> list[dict[str, float]]:
    """Generate deterministic maximin-like Latin-hypercube points without SciPy."""
    if count < 1:
        return []
    rng = np.random.default_rng(seed)
    keys = list(family.parameters)
    columns: list[np.ndarray] = []
    for _ in keys:
        bins = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
        columns.append(bins[rng.permutation(count)])
    rows: list[dict[str, float]] = []
    for row_index in range(count):
        row: dict[str, float] = {}
        for key, values in zip(keys, columns, strict=True):
            lower, upper = family.parameters[key]
            value = lower + float(values[row_index]) * (upper - lower)
            if key in {"n", "clusters", "cluster_size", "horizon"}:
                value = float(max(2, round(value)))
            row[key] = value
        rows.append(row)
    return rows


_FAMILY_RISK_KEYS: dict[str, tuple[str, ...]] = {
    "group_comparison": ("variance_ratio", "tail_weight", "pair_correlation"),
    "categorical": ("baseline_probability", "imbalance", "n"),
    "linear_robust": ("heteroskedasticity", "outlier_fraction", "leverage"),
    "binary_count_glm": ("event_rate", "overdispersion", "separation"),
    "clustered_repeated": ("clusters", "icc", "cluster_size"),
    "survival": ("censoring", "non_ph", "n"),
    "missing_selection": ("missing_rate", "selection_strength", "positivity"),
    "experimental_causal": (
        "assignment_probability",
        "prognostic_strength",
        "confounding",
    ),
    "probability_distribution": ("skew", "tail_weight", "prior_strength"),
    "bayesian_check": ("prior_bias", "prior_strength", "model_misspecification"),
    "predictive_calibration": ("prevalence", "shift", "leakage"),
    "time_series_leakage": ("autocorrelation", "drift", "leakage", "horizon"),
}

_LOW_RISK_BOUNDARY_KEYS = {
    "n",
    "clusters",
    "baseline_probability",
    "imbalance",
    "event_rate",
    "positivity",
    "assignment_probability",
    "prevalence",
}


def _set_boundary_value(
    family: DGPFamily,
    result: dict[str, float],
    key: str,
    *,
    quantile: float,
) -> None:
    lower, upper = family.parameters[key]
    effective = 1.0 - quantile if key in _LOW_RISK_BOUNDARY_KEYS else quantile
    value = lower + effective * (upper - lower)
    if key in {"n", "clusters", "cluster_size", "horizon"}:
        value = float(max(2, round(value)))
    result[key] = value


def _boundary_transform(
    family: DGPFamily,
    parameters: dict[str, float],
    *,
    round_index: int,
    target_key: str,
    secondary_key: str | None = None,
) -> dict[str, float]:
    """Move an LHS point toward one or two declared assumption boundaries."""
    result = dict(parameters)
    direction = 0.82 if round_index == 1 else 0.96
    _set_boundary_value(family, result, target_key, quantile=direction)
    if secondary_key is not None:
        _set_boundary_value(family, result, secondary_key, quantile=0.82)
    sample_key = next((key for key in ("n", "clusters") if key in result), None)
    if sample_key is not None:
        sample_quantile = 0.20 if round_index == 1 else 0.08
        _set_boundary_value(family, result, sample_key, quantile=1.0 - sample_quantile)
    return result


def _failure_region_search(
    family: DGPFamily,
    parameters: dict[str, float],
    *,
    round_index: int,
    scenario_seed: int,
    split: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Select a failure-region candidate using simulator validity and ranking uncertainty."""
    risk_keys = _FAMILY_RISK_KEYS[family.family_id]
    trials: list[tuple[tuple[Any, ...], dict[str, float], dict[str, Any]]] = []
    for index, key in enumerate(risk_keys):
        secondary = risk_keys[(index + 1) % len(risk_keys)] if round_index == 2 else None
        candidate = _boundary_transform(
            family,
            parameters,
            round_index=round_index,
            target_key=key,
            secondary_key=secondary,
        )
        probe = Scenario(
            blueprint_id=f"search-{family.family_id}-{round_index}-{index}",
            family_id=family.family_id,
            split=split,
            seed=scenario_seed,
            parameters=candidate,
            boundary_round=round_index,
            domain=family.domain,
        )
        simulated = simulate_scenario(
            probe,
            initial_repetitions=128,
            escalation_repetitions=(),
            uncertainty_margin=0.035,
            temperature=0.15,
        )
        central = next(
            item for item in simulated["candidates"] if item["method_id"] == family.central_method
        )
        history = simulated["adaptive_history"][-1]
        invalid_count = sum(not bool(item["valid"]) for item in simulated["candidates"])
        evidence = {
            "round": round_index,
            "candidate_count": len(risk_keys),
            "selected_boundary_key": key,
            "secondary_boundary_key": secondary,
            "ranking_ambiguous": bool(history["ranking_ambiguous"]),
            "top_gap": float(history["top_gap"]),
            "central_method_regret": float(central["normalized_regret"]),
            "invalid_candidate_count": invalid_count,
            "criterion": (
                "validity-failure-first"
                if round_index == 1
                else "ranking-uncertainty-then-central-regret"
            ),
        }
        if round_index == 1:
            score = (
                -invalid_count,
                -float(central["normalized_regret"]),
                not bool(history["ranking_ambiguous"]),
                float(history["top_gap"]),
                key,
            )
        else:
            score = (
                not bool(history["ranking_ambiguous"]),
                float(history["top_gap"]),
                -float(central["normalized_regret"]),
                -invalid_count,
                key,
            )
        trials.append((score, candidate, evidence))
    _, selected, evidence = min(trials, key=lambda item: item[0])
    return selected, evidence


def _allocate_by_domain(total: int) -> dict[str, int]:
    inference = round(total * 0.60)
    probability = round(total * 0.20)
    return {
        "inference_and_design": inference,
        "probability_and_bayes": probability,
        "prediction_and_analysis": total - inference - probability,
    }


def _balanced_counts(total: int, families: list[DGPFamily]) -> dict[str, int]:
    counts = {family.family_id: total // len(families) for family in families}
    for family in families[: total % len(families)]:
        counts[family.family_id] += 1
    return counts


def build_blueprints(
    split_counts: dict[str, int],
    *,
    seed: int = 42,
    active_search: bool = True,
) -> list[Scenario]:
    """Split by blueprint and seed before any language rendering occurs."""
    scenarios: list[Scenario] = []
    for split_index, (split, total) in enumerate(split_counts.items()):
        allocations = _allocate_by_domain(total)
        for domain_index, (domain, domain_total) in enumerate(allocations.items()):
            domain_families = [family for family in FAMILIES if family.domain == domain]
            per_family = _balanced_counts(domain_total, domain_families)
            for family_index, family in enumerate(domain_families):
                count = per_family[family.family_id]
                lhs_seed = seed + split_index * 100_000 + domain_index * 10_000 + family_index * 701
                points = latin_hypercube(family, count, seed=lhs_seed)
                for local_index, original in enumerate(points):
                    phase = local_index % 3 if split == "train" and active_search else 0
                    scenario_seed = (
                        seed + split_index * 1_000_000 + family_index * 10_000 + local_index
                    )
                    if phase == 0:
                        parameters = original
                        search = None
                    else:
                        parameters, search = _failure_region_search(
                            family,
                            original,
                            round_index=phase,
                            scenario_seed=scenario_seed,
                            split=split,
                        )
                    identity = {
                        "family": family.family_id,
                        "split": split,
                        "seed": scenario_seed,
                        "parameters": parameters,
                        "boundary_round": phase,
                    }
                    if not active_search:
                        identity["profile"] = "random-lhs-ablation"
                    scenarios.append(
                        Scenario(
                            blueprint_id=f"dgp-{canonical_hash(identity)[:16]}",
                            family_id=family.family_id,
                            split=split,
                            seed=scenario_seed,
                            parameters=parameters,
                            boundary_round=phase,
                            domain=family.domain,
                            search=search,
                        )
                    )
    if len({scenario.blueprint_id for scenario in scenarios}) != len(scenarios):
        raise RuntimeError("DGP blueprint collision")
    return scenarios


def _base_point(method_id: str) -> dict[str, float | str]:
    return {
        "type1": 0.05,
        "coverage": 0.95,
        "bias": 0.0,
        "rmse": 0.45,
        "power": 0.65,
        "calibration": 0.04,
        "invalidity": 0.0,
        "reason": "assumptions match the generated regime",
    }


def _penalize(
    point: dict[str, float | str],
    *,
    severity: float,
    reason: str,
    bias: float = 0.0,
    power_loss: float = 0.0,
) -> None:
    severity = _clip(severity)
    point["type1"] = _clip(float(point["type1"]) + 0.20 * severity)
    point["coverage"] = _clip(float(point["coverage"]) - 0.38 * severity)
    point["bias"] = float(point["bias"]) + bias * severity
    point["rmse"] = float(point["rmse"]) * (1.0 + 1.25 * severity)
    point["power"] = _clip(float(point["power"]) - power_loss * severity)
    point["calibration"] = _clip(float(point["calibration"]) + 0.30 * severity)
    point["invalidity"] = max(float(point["invalidity"]), severity)
    point["reason"] = reason


def _improve(
    point: dict[str, float | str],
    *,
    rmse: float = 1.0,
    power: float = 0.0,
    cost_calibration: float = 0.0,
) -> None:
    point["rmse"] = max(0.05, float(point["rmse"]) * rmse)
    point["power"] = _clip(float(point["power"]) + power)
    point["calibration"] = _clip(float(point["calibration"]) + cost_calibration)


def expected_operating_point(scenario: Scenario, method_id: str) -> OperatingPoint:
    family = FAMILY_BY_ID[scenario.family_id]
    if method_id not in family.candidate_methods:
        raise ValueError(f"{method_id} is not a candidate for {family.family_id}")
    p = scenario.parameters
    point = _base_point(method_id)
    effect = _scaled(
        p,
        family,
        next((key for key in ("effect", "log_hazard_ratio", "signal") if key in p), list(p)[0]),
    )
    sample_key = next((key for key in ("n", "clusters") if key in p), None)
    sample = _scaled(p, family, sample_key) if sample_key else 0.5
    point["power"] = _clip(0.12 + 0.78 * _logistic(5.0 * (effect + 0.45 * sample - 0.65)))
    point["rmse"] = 0.72 - 0.42 * sample

    if family.family_id == "group_comparison":
        variance = abs(math.log(max(p["variance_ratio"], 1e-6))) / math.log(4.0)
        tails = _scaled(p, family, "tail_weight")
        pairing = _scaled(p, family, "pair_correlation")
        paired_design = pairing > 0.62
        if method_id == "independent_t" and (variance > 0.30 or paired_design):
            _penalize(
                point,
                severity=max(variance, pairing if paired_design else 0.0),
                reason="pooled or independent standard error conflicts with the DGP",
                power_loss=0.30,
            )
        elif method_id == "welch_t":
            if paired_design:
                _penalize(
                    point,
                    severity=0.35 + 0.35 * pairing,
                    reason="pairing was discarded",
                    power_loss=0.45,
                )
            elif tails > 0.72 and sample < 0.35:
                _penalize(
                    point,
                    severity=0.25 + 0.45 * tails,
                    reason="small-sample mean inference is unstable under very heavy tails",
                    power_loss=0.15,
                )
        elif method_id == "mann_whitney":
            _improve(
                point, rmse=0.83 if tails > 0.5 else 1.20, power=0.12 if tails > 0.5 else -0.08
            )
            if paired_design:
                _penalize(
                    point,
                    severity=0.55,
                    reason="independent ranks discard known pairs",
                    power_loss=0.35,
                )
        elif method_id in {"paired_t", "wilcoxon_signed_rank"}:
            if not paired_design:
                _penalize(
                    point,
                    severity=0.95,
                    reason="the procedure requires pairs that the DGP does not contain",
                    bias=0.25,
                    power_loss=0.45,
                )
            else:
                _improve(point, rmse=max(0.45, 1.0 - 0.55 * pairing), power=0.22 * pairing)
                if method_id == "paired_t" and tails > 0.7:
                    _penalize(
                        point,
                        severity=0.45 * tails,
                        reason=(
                            "paired differences are too heavy-tailed for the "
                            "small-sample t reference"
                        ),
                    )
                if method_id == "wilcoxon_signed_rank" and tails > 0.45:
                    _improve(point, rmse=0.80, power=0.08)

    elif family.family_id == "categorical":
        n = p["n"]
        baseline = p["baseline_probability"]
        risk = min(0.99, baseline + p["risk_difference"])
        imbalance = p["imbalance"]
        expected_min = (
            n * min(imbalance, 1.0 - imbalance) * min(baseline, 1 - baseline, risk, 1 - risk)
        )
        sparsity = _clip((5.0 - expected_min) / 5.0)
        if method_id in {"chi_square", "two_proportion"} and sparsity > 0:
            _penalize(
                point,
                severity=0.25 + 0.70 * sparsity,
                reason="asymptotic cell-count approximation is sparse",
                power_loss=0.10,
            )
        elif method_id == "fisher_exact":
            _improve(point, rmse=1.0, power=-0.12 * (1.0 - sparsity), cost_calibration=-0.02)

    elif family.family_id == "linear_robust":
        hetero = _scaled(p, family, "heteroskedasticity")
        outliers = _scaled(p, family, "outlier_fraction")
        leverage = _scaled(p, family, "leverage")
        if method_id == "ols" and (hetero > 0.12 or outliers > 0.10):
            _penalize(
                point,
                severity=max(hetero, outliers),
                reason="classical OLS covariance is invalid under this error regime",
                bias=0.15 * outliers,
            )
        elif method_id == "hc3_ols":
            _improve(point, rmse=1.04, power=-0.02)
            if outliers > 0.48:
                _penalize(
                    point,
                    severity=0.25 + 0.55 * outliers,
                    reason="HC3 repairs covariance but not influential outcome contamination",
                    bias=0.25,
                )
        elif method_id == "huber_regression":
            _improve(
                point,
                rmse=0.76 if outliers > 0.25 else 1.14,
                power=0.08 if outliers > 0.25 else -0.05,
            )
            if leverage > 0.78:
                _penalize(
                    point,
                    severity=0.35 * leverage,
                    reason="response-robust fitting alone does not repair extreme leverage",
                    bias=0.14,
                )

    elif family.family_id == "binary_count_glm":
        binary = p["event_rate"] < 0.34 or p["separation"] > 0.45
        separation = _scaled(p, family, "separation")
        dispersion = _scaled(p, family, "overdispersion")
        binary_methods = {"logistic_glm", "firth_logistic"}
        if (method_id in binary_methods) != binary:
            _penalize(
                point,
                severity=1.0,
                reason="the likelihood does not match the outcome type",
                bias=0.5,
                power_loss=0.45,
            )
        elif binary and method_id == "logistic_glm" and separation > 0.48:
            _penalize(
                point,
                severity=0.35 + 0.65 * separation,
                reason="separation makes ordinary maximum likelihood unstable",
                bias=0.45,
            )
        elif binary and method_id == "firth_logistic":
            _improve(point, rmse=0.72 if separation > 0.4 else 1.08, power=-0.03)
        elif not binary and method_id == "poisson_glm" and dispersion > 0.15:
            _penalize(
                point,
                severity=dispersion,
                reason="Poisson model-based uncertainty ignores overdispersion",
            )
        elif not binary and method_id == "negative_binomial_glm":
            _improve(
                point,
                rmse=0.82 if dispersion > 0.3 else 1.12,
                power=0.06 if dispersion > 0.3 else -0.04,
            )

    elif family.family_id == "clustered_repeated":
        icc = _scaled(p, family, "icc")
        clusters = p["clusters"]
        cluster_small = _clip((20.0 - clusters) / 14.0)
        if method_id == "ols" and icc > 0.05:
            _penalize(
                point,
                severity=0.25 + 0.75 * icc,
                reason="row-level OLS treats correlated observations as independent",
            )
        elif method_id == "gee":
            _improve(point, rmse=1.03, power=-0.02)
            if clusters < 20:
                _penalize(
                    point,
                    severity=0.25 + 0.55 * cluster_small,
                    reason="the cluster sandwich has too few independent clusters",
                )
        elif method_id == "mixed_effects":
            _improve(point, rmse=0.84 if icc > 0.25 else 1.04, power=0.08)
            if clusters < 8:
                _penalize(
                    point,
                    severity=0.58,
                    reason="random-effect variance is weakly identified with very few clusters",
                )

    elif family.family_id == "survival":
        censoring = _scaled(p, family, "censoring")
        non_ph = _scaled(p, family, "non_ph")
        if method_id == "cox_ph" and non_ph > 0.42:
            _penalize(
                point,
                severity=0.25 + 0.65 * non_ph,
                reason=(
                    "a single proportional hazard ratio is not stable under this "
                    "time-varying effect"
                ),
                bias=0.30,
            )
        elif method_id == "logrank":
            _improve(point, rmse=1.08, power=-0.22 * non_ph)
        if censoring > 0.78:
            _penalize(
                point,
                severity=0.24 + 0.35 * censoring,
                reason="extreme censoring leaves inadequate event information",
                power_loss=0.28,
            )

    elif family.family_id == "missing_selection":
        missing = _scaled(p, family, "missing_rate")
        selection = _scaled(p, family, "selection_strength")
        positivity = _scaled(p, family, "positivity")
        if method_id == "ols" and (missing > 0.08 or selection > 0.08):
            _penalize(
                point,
                severity=max(missing, selection),
                reason="complete-case OLS is biased under informative observation",
                bias=0.48,
            )
        elif method_id == "multiple_imputation":
            _improve(point, rmse=0.92, power=0.05)
            if selection > 0.75:
                _penalize(
                    point,
                    severity=0.25 + 0.40 * selection,
                    reason="the imputation variables do not render the missingness ignorable",
                    bias=0.24,
                )
        elif method_id == "ipw":
            _improve(point, rmse=1.12, power=-0.04)
            if positivity < 0.22:
                _penalize(
                    point,
                    severity=0.72,
                    reason="near-positivity violations create unstable inverse weights",
                    bias=0.12,
                )

    elif family.family_id == "experimental_causal":
        confounding = _scaled(p, family, "confounding")
        prognostic = _scaled(p, family, "prognostic_strength")
        assignment = p["assignment_probability"]
        randomized = confounding < 0.32
        if (
            method_id in {"difference_in_means", "ancova", "randomization_inference"}
            and not randomized
        ):
            _penalize(
                point,
                severity=0.45 + 0.50 * confounding,
                reason="design-based procedure was applied without randomized assignment",
                bias=0.55,
            )
        elif method_id == "difference_in_means":
            _improve(point, rmse=1.0, power=0.0)
        elif method_id == "ancova":
            _improve(point, rmse=max(0.45, 1.0 - 0.50 * prognostic), power=0.22 * prognostic)
        elif method_id == "randomization_inference":
            _improve(point, rmse=1.08, power=-0.06, cost_calibration=-0.02)
        elif method_id == "ipw":
            positivity_severity = _clip((0.16 - min(assignment, 1 - assignment)) / 0.16)
            _improve(point, rmse=1.18, power=-0.08)
            if positivity_severity > 0:
                _penalize(
                    point,
                    severity=positivity_severity,
                    reason="treatment propensity is too close to a positivity boundary",
                )
        elif method_id == "hc3_ols" and not randomized:
            _penalize(
                point,
                severity=0.40 * confounding,
                reason="robust covariance does not remove unmeasured confounding",
                bias=0.35,
            )

    elif family.family_id == "probability_distribution":
        skew = _scaled(p, family, "skew")
        tails = _scaled(p, family, "tail_weight")
        prior = _scaled(p, family, "prior_strength")
        if method_id == "conjugate_bayes":
            _improve(point, rmse=0.75 if sample < 0.35 and prior < 0.6 else 0.96, power=0.04)
            if skew > 0.65 or tails > 0.65:
                _penalize(
                    point,
                    severity=max(skew, tails) * 0.62,
                    reason="the conjugate likelihood misses skew or heavy tails",
                    bias=0.20,
                )
        elif method_id == "posterior_predictive":
            point["power"] = _clip(0.22 + 0.70 * max(skew, tails))
            _improve(point, rmse=1.15, power=0.0)
        elif method_id == "welch_t":
            _improve(point, rmse=1.12, power=-0.06)
            if (skew > 0.72 or tails > 0.72) and sample < 0.45:
                _penalize(
                    point,
                    severity=0.58,
                    reason="the small-sample mean interval is unstable for this distribution",
                )

    elif family.family_id == "bayesian_check":
        prior_bias = _scaled(p, family, "prior_bias")
        prior_strength = _scaled(p, family, "prior_strength")
        misspec = _scaled(p, family, "model_misspecification")
        if method_id == "conjugate_bayes":
            _improve(point, rmse=max(0.55, 1.0 - 0.35 * prior_strength), power=0.04)
            conflict = prior_bias * prior_strength
            if conflict > 0.18 or misspec > 0.45:
                _penalize(
                    point,
                    severity=max(conflict, 0.65 * misspec),
                    reason="prior conflict or likelihood misspecification distorts the posterior",
                    bias=0.55,
                )
        elif method_id == "posterior_predictive":
            point["type1"] = 0.06
            point["coverage"] = 0.94
            point["power"] = _clip(0.15 + 0.80 * misspec)
            point["rmse"] = 0.52
            point["reason"] = "generative adequacy is checked before substantive reporting"

    elif family.family_id == "predictive_calibration":
        shift = _scaled(p, family, "shift")
        leakage = _scaled(p, family, "leakage")
        prevalence = p["prevalence"]
        imbalance = _clip((0.12 - min(prevalence, 1 - prevalence)) / 0.12)
        if method_id == "logistic_glm" and (shift > 0.10 or leakage > 0.10):
            _penalize(
                point,
                severity=max(shift, leakage),
                reason="in-sample fit is not a deployment-calibrated validation design",
                bias=0.20,
            )
        elif method_id == "calibrated_logistic":
            _improve(point, rmse=0.83, power=0.05)
            point["calibration"] = _clip(0.025 + 0.10 * shift + 0.06 * imbalance)

    elif family.family_id == "time_series_leakage":
        autocorr = _scaled(p, family, "autocorrelation")
        drift = _scaled(p, family, "drift")
        leakage = _scaled(p, family, "leakage")
        if method_id == "calibrated_logistic":
            _penalize(
                point,
                severity=max(autocorr, drift, leakage),
                reason="random cross-validation leaks chronology or misstates future error",
                bias=0.32,
            )
        elif method_id == "blocked_time_series_cv":
            _improve(point, rmse=0.86, power=0.03)
            point["calibration"] = _clip(0.03 + 0.08 * drift)

    invalidity = max(
        float(point["invalidity"]),
        _clip((float(point["type1"]) - 0.06) / 0.19),
        _clip((0.90 - float(point["coverage"])) / 0.35),
    )
    procedure = PROCEDURE_BY_ID[method_id]
    return OperatingPoint(
        method_id=method_id,
        valid=invalidity < 0.25,
        type1_error=_clip(float(point["type1"])),
        coverage=_clip(float(point["coverage"])),
        bias=float(point["bias"]),
        rmse=max(0.01, float(point["rmse"])),
        power=_clip(float(point["power"])),
        calibration_error=_clip(float(point["calibration"])),
        cost=procedure.cost,
        invalidity=invalidity,
        reason=str(point["reason"]),
    )


def _simulate_metrics(
    point: OperatingPoint,
    *,
    common: dict[str, np.ndarray],
) -> dict[str, Any]:
    null_reject = common["u_type1"] < point.type1_error
    covered = common["u_coverage"] < point.coverage
    detected = common["u_power"] < point.power
    errors = point.bias + point.rmse * common["z_error"]
    calibration = np.abs(point.calibration_error + 0.018 * common["z_calibration"])
    repetitions = len(null_reject)
    return {
        "method_id": point.method_id,
        "valid": point.valid,
        "repetitions": repetitions,
        "type1_error": float(np.mean(null_reject)),
        "coverage": float(np.mean(covered)),
        "bias": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "power": float(np.mean(detected)),
        "calibration_error": float(np.mean(calibration)),
        "cost": point.cost,
        "invalidity": point.invalidity,
        "reason": point.reason,
        "monte_carlo_se": {
            "type1_error": float(
                math.sqrt(max(point.type1_error * (1 - point.type1_error), 1e-9) / repetitions)
            ),
            "coverage": float(
                math.sqrt(max(point.coverage * (1 - point.coverage), 1e-9) / repetitions)
            ),
            "power": float(math.sqrt(max(point.power * (1 - point.power), 1e-9) / repetitions)),
        },
    }


def _raw_regret(metrics: dict[str, Any]) -> float:
    """Validity is lexicographically dominant; power matters only after validity."""
    type1_violation = max(0.0, float(metrics["type1_error"]) - 0.06) / 0.19
    coverage_violation = max(0.0, 0.90 - float(metrics["coverage"])) / 0.35
    validity = max(float(metrics["invalidity"]), type1_violation, coverage_violation)
    accuracy = (
        0.42 * min(abs(float(metrics["bias"])), 1.0)
        + 0.34 * min(float(metrics["rmse"]), 1.5) / 1.5
        + 0.24 * min(float(metrics["calibration_error"]), 0.5) / 0.5
    )
    if validity >= 0.25:
        return 2.0 + validity + 0.10 * accuracy + 0.02 * float(metrics["cost"])
    return accuracy + 0.18 * (1.0 - float(metrics["power"])) + 0.05 * float(metrics["cost"])


def _softmax_regret(values: np.ndarray, temperature: float) -> np.ndarray:
    shifted = -values / temperature
    shifted -= np.max(shifted)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def simulate_scenario(
    scenario: Scenario,
    *,
    initial_repetitions: int = 128,
    escalation_repetitions: Iterable[int] = (256, 512),
    uncertainty_margin: float = 0.035,
    temperature: float = 0.15,
) -> dict[str, Any]:
    """Evaluate all candidates with common random numbers and adaptive effort."""
    family = FAMILY_BY_ID[scenario.family_id]
    candidate_ids = list(family.candidate_methods)
    if len(candidate_ids) > 6:
        raise RuntimeError(f"candidate menu exceeds six procedures for {family.family_id}")
    repetition_schedule = [initial_repetitions, *escalation_repetitions]
    rng = np.random.default_rng(scenario.seed)
    maximum = max(repetition_schedule)
    common_all = {
        "u_type1": rng.random(maximum),
        "u_coverage": rng.random(maximum),
        "u_power": rng.random(maximum),
        "z_error": rng.standard_normal(maximum),
        "z_calibration": rng.standard_normal(maximum),
    }
    history: list[dict[str, Any]] = []
    final_metrics: list[dict[str, Any]] = []
    final_raw = np.array([], dtype=np.float64)
    for schedule_index, repetitions in enumerate(repetition_schedule):
        common = {key: value[:repetitions] for key, value in common_all.items()}
        final_metrics = [
            _simulate_metrics(expected_operating_point(scenario, method_id), common=common)
            for method_id in candidate_ids
        ]
        final_raw = np.array([_raw_regret(item) for item in final_metrics], dtype=np.float64)
        order = np.argsort(final_raw)
        gap = float(final_raw[order[1]] - final_raw[order[0]]) if len(order) > 1 else 1.0
        mc_se = (
            max(
                float(final_metrics[int(order[0])]["monte_carlo_se"]["coverage"]),
                float(final_metrics[int(order[1])]["monte_carlo_se"]["coverage"]),
            )
            if len(order) > 1
            else 0.0
        )
        ambiguous = gap < uncertainty_margin + 1.25 * mc_se
        history.append(
            {
                "repetitions": repetitions,
                "top_gap": gap,
                "monte_carlo_se_bound": mc_se,
                "ranking_ambiguous": ambiguous,
            }
        )
        if not ambiguous or schedule_index == len(repetition_schedule) - 1:
            break
    minimum = float(np.min(final_raw))
    maximum_raw = float(np.max(final_raw))
    span = maximum_raw - minimum
    normalized = np.zeros_like(final_raw) if span <= 1e-12 else (final_raw - minimum) / span
    soft = _softmax_regret(normalized, temperature)
    for index, metrics in enumerate(final_metrics):
        metrics["raw_regret"] = float(final_raw[index])
        metrics["normalized_regret"] = float(normalized[index])
        metrics["soft_target"] = float(soft[index])
    selected_index = int(np.argmin(final_raw))
    return {
        "schema_version": 1,
        "engine": "common-random-number operating-characteristic simulator v1",
        "scenario": scenario.to_dict(),
        "candidates": final_metrics,
        "selected_method_id": candidate_ids[selected_index],
        "valid_method_ids": [item["method_id"] for item in final_metrics if item["valid"]],
        "repetitions": int(final_metrics[0]["repetitions"]),
        "adaptive_history": history,
        "regret_temperature": temperature,
        "fingerprint": canonical_hash(
            {
                "scenario": scenario.to_dict(),
                "candidate_ids": candidate_ids,
                "repetitions": int(final_metrics[0]["repetitions"]),
                "engine_version": 1,
            }
        ),
    }


def central_validity_checks() -> dict[str, Any]:
    """Sentinel scenarios that must preserve standard-method operating validity."""
    sentinels: list[tuple[Scenario, str]] = []
    for index, family in enumerate(FAMILIES):
        midpoint = {key: (bounds[0] + bounds[1]) / 2.0 for key, bounds in family.parameters.items()}
        if "variance_ratio" in midpoint:
            midpoint["variance_ratio"] = 1.0
            midpoint["pair_correlation"] = 0.0
            midpoint["tail_weight"] = 0.0
        if "heteroskedasticity" in midpoint:
            midpoint["heteroskedasticity"] = 0.0
            midpoint["outlier_fraction"] = 0.0
        if "separation" in midpoint:
            midpoint["separation"] = 0.0
            midpoint["event_rate"] = 0.20
        if "icc" in midpoint:
            midpoint["icc"] = 0.15
            midpoint["clusters"] = 50.0
        if "non_ph" in midpoint:
            midpoint["non_ph"] = 0.0
        if "selection_strength" in midpoint:
            midpoint["selection_strength"] = 0.0
            midpoint["missing_rate"] = 0.05
        if "confounding" in midpoint:
            midpoint["confounding"] = 0.0
            midpoint["assignment_probability"] = 0.5
        if "prior_bias" in midpoint:
            midpoint["prior_bias"] = 0.0
        if "model_misspecification" in midpoint:
            midpoint["model_misspecification"] = 0.0
        if "shift" in midpoint:
            midpoint["shift"] = 0.0
        if "leakage" in midpoint:
            midpoint["leakage"] = 0.0
        if "drift" in midpoint:
            midpoint["drift"] = 0.0
        scenario = Scenario(
            blueprint_id=f"sentinel-{family.family_id}",
            family_id=family.family_id,
            split="sentinel",
            seed=9_000 + index,
            parameters={key: float(value) for key, value in midpoint.items()},
            boundary_round=0,
            domain=family.domain,
        )
        sentinels.append((scenario, family.central_method))
    checks: list[dict[str, Any]] = []
    for scenario, method_id in sentinels:
        point = expected_operating_point(scenario, method_id)
        passed = point.type1_error <= 0.075 and point.coverage >= 0.90 and point.invalidity < 0.25
        checks.append(
            {
                "family_id": scenario.family_id,
                "method_id": method_id,
                "passed": passed,
                "type1_error": point.type1_error,
                "coverage": point.coverage,
                "invalidity": point.invalidity,
            }
        )
    return {"passed": all(item["passed"] for item in checks), "checks": checks}

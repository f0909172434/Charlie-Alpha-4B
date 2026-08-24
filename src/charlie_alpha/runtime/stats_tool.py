from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from scipy.optimize import minimize
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_predict
from statsmodels.sandbox.regression.gmm import IV2SLS
from statsmodels.tools.numdiff import approx_hess


def load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported data extension: {suffix}")


def serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def require_columns(frame: pd.DataFrame, variables: dict[str, Any], names: list[str]) -> None:
    missing_fields = [name for name in names if not variables.get(name)]
    if missing_fields:
        raise ValueError(f"missing variable roles: {', '.join(missing_fields)}")
    requested: list[str] = []
    for name in names:
        value = variables[name]
        requested.extend(value if isinstance(value, list) else [value])
    absent = [str(name) for name in requested if str(name) not in frame.columns]
    if absent:
        raise ValueError(f"columns do not exist: {', '.join(absent)}")


def inspect_frame(frame: pd.DataFrame, output_csv: Path | None = None) -> dict[str, Any]:
    if output_csv is not None:
        frame.to_csv(output_csv, index=False)
    columns = []
    for name in frame.columns:
        series = frame[name]
        example = [serializable(value) for value in series.dropna().head(3).tolist()]
        columns.append(
            {
                "name": str(name),
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
                "examples": example,
            }
        )
    return {
        "rows": int(len(frame)),
        "columns": columns,
        "duplicate_rows": int(frame.duplicated().sum()),
    }


def groups(
    frame: pd.DataFrame, outcome: str, group: str
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    clean = frame[[outcome, group]].dropna()
    levels = clean[group].drop_duplicates().tolist()
    if len(levels) != 2:
        raise ValueError("group must have exactly two observed levels")
    left = clean.loc[clean[group] == levels[0], outcome].astype(float).to_numpy()
    right = clean.loc[clean[group] == levels[1], outcome].astype(float).to_numpy()
    return left, right, levels


def regression_data(
    frame: pd.DataFrame, variables: dict[str, Any]
) -> tuple[pd.Series, pd.DataFrame]:
    predictors = variables.get("predictors") or [variables.get("predictor")]
    predictors = [str(value) for value in predictors if value]
    outcome = str(variables.get("outcome") or "")
    require_columns(
        frame, {"outcome": outcome, "predictors": predictors}, ["outcome", "predictors"]
    )
    data = frame[[outcome, *predictors]].dropna()
    x = pd.get_dummies(data[predictors], drop_first=True, dtype=float)
    return data[outcome].astype(float), sm.add_constant(x, has_constant="add")


def result_for_method(frame: pd.DataFrame, request: dict[str, Any]) -> dict[str, Any]:
    method = str(request["method_id"])
    variables = dict(request.get("variables") or {})
    alpha = float(request.get("alpha", 0.05))
    if method == "inspect":
        output = Path(request["output_csv"]) if request.get("output_csv") else None
        return inspect_frame(frame, output)

    if method == "binomial_test":
        require_columns(frame, variables, ["outcome"])
        values = frame[str(variables["outcome"])].dropna()
        if values.nunique() != 2:
            raise ValueError("binomial_test outcome must have exactly two levels")
        encoded = pd.Categorical(values).codes
        options = dict(request.get("analysis_options") or {})
        null_probability = float(options.get("null_probability", 0.5))
        alternative = str(options.get("alternative", "two-sided"))
        test = st.binomtest(
            int(np.sum(encoded == 1)),
            n=len(encoded),
            p=null_probability,
            alternative=alternative,
        )
        interval = test.proportion_ci(confidence_level=1 - alpha, method="exact")
        return {
            "n": len(encoded),
            "successes": int(np.sum(encoded == 1)),
            "null_probability": null_probability,
            "alternative": alternative,
            "estimated_probability": test.proportion_estimate,
            "confidence_interval": [interval.low, interval.high],
            "p_value": test.pvalue,
        }

    if method == "spearman_correlation":
        require_columns(frame, variables, ["x", "y"])
        clean = frame[[str(variables["x"]), str(variables["y"])]].dropna()
        statistic, p_value = st.spearmanr(clean.iloc[:, 0], clean.iloc[:, 1])
        return {"n": len(clean), "spearman_rho": statistic, "p_value": p_value}

    if method == "kruskal_wallis":
        require_columns(frame, variables, ["outcome", "group"])
        clean = frame[[str(variables["outcome"]), str(variables["group"])]].dropna()
        samples = [
            group[str(variables["outcome"])].astype(float).to_numpy()
            for _, group in clean.groupby(str(variables["group"]), sort=False)
        ]
        if len(samples) < 2:
            raise ValueError("kruskal_wallis requires at least two observed groups")
        statistic, p_value = st.kruskal(*samples)
        return {
            "n": len(clean),
            "groups": len(samples),
            "statistic": statistic,
            "p_value": p_value,
        }

    if method in {"independent_t", "welch_t", "mann_whitney"}:
        require_columns(frame, variables, ["outcome", "group"])
        left, right, levels = groups(frame, variables["outcome"], variables["group"])
        if method == "mann_whitney":
            statistic, p_value = st.mannwhitneyu(left, right, alternative="two-sided")
            return {
                "levels": levels,
                "n": [len(left), len(right)],
                "statistic": statistic,
                "p_value": p_value,
                "estimand_note": (
                    "rank-based probability of superiority, not automatically a mean difference"
                ),
            }
        equal_var = method == "independent_t"
        statistic, p_value = st.ttest_ind(left, right, equal_var=equal_var)
        mean_difference = float(np.mean(right) - np.mean(left))
        if equal_var:
            pooled = (len(left) - 1) * np.var(left, ddof=1) + (len(right) - 1) * np.var(
                right, ddof=1
            )
            standard_error = math.sqrt(
                pooled / (len(left) + len(right) - 2) * (1 / len(left) + 1 / len(right))
            )
            degrees = len(left) + len(right) - 2
        else:
            left_term = np.var(left, ddof=1) / len(left)
            right_term = np.var(right, ddof=1) / len(right)
            standard_error = math.sqrt(left_term + right_term)
            degrees = (left_term + right_term) ** 2 / (
                left_term**2 / (len(left) - 1) + right_term**2 / (len(right) - 1)
            )
        critical = st.t.ppf(1 - alpha / 2, degrees)
        return {
            "levels": levels,
            "n": [len(left), len(right)],
            "mean_difference_second_minus_first": mean_difference,
            "confidence_interval": [
                mean_difference - critical * standard_error,
                mean_difference + critical * standard_error,
            ],
            "statistic": statistic,
            "degrees_of_freedom": degrees,
            "p_value": p_value,
        }

    if method in {"paired_t", "wilcoxon_signed_rank"}:
        require_columns(frame, variables, ["before", "after"])
        clean = frame[[variables["before"], variables["after"]]].dropna().astype(float)
        before = clean.iloc[:, 0].to_numpy()
        after = clean.iloc[:, 1].to_numpy()
        if method == "paired_t":
            statistic, p_value = st.ttest_rel(after, before)
        else:
            statistic, p_value = st.wilcoxon(after, before)
        return {
            "n_pairs": len(clean),
            "mean_difference_after_minus_before": float(np.mean(after - before)),
            "statistic": statistic,
            "p_value": p_value,
        }

    if method in {"chi_square", "fisher_exact", "two_proportion"}:
        require_columns(frame, variables, ["outcome", "group"])
        table = pd.crosstab(frame[variables["group"]], frame[variables["outcome"]])
        if table.shape != (2, 2) and method != "chi_square":
            raise ValueError("this method requires a 2 by 2 table")
        if method == "fisher_exact":
            odds_ratio, p_value = st.fisher_exact(table.to_numpy())
            return {"table": table.to_dict(), "odds_ratio": odds_ratio, "p_value": p_value}
        statistic, p_value, degrees, expected = st.chi2_contingency(
            table.to_numpy(), correction=method == "two_proportion"
        )
        return {
            "table": table.to_dict(),
            "statistic": statistic,
            "degrees_of_freedom": degrees,
            "p_value": p_value,
            "minimum_expected_count": float(expected.min()),
        }

    if method in {"ols", "hc3_ols", "huber_regression", "ancova"}:
        y, x = regression_data(frame, variables)
        if method == "huber_regression":
            fitted = sm.RLM(y, x, M=sm.robust.norms.HuberT()).fit()
        else:
            covariance = "HC3" if method in {"hc3_ols", "ancova"} else "nonrobust"
            fitted = sm.OLS(y, x).fit(cov_type=covariance)
        return {
            "n": int(fitted.nobs),
            "coefficients": fitted.params.to_dict(),
            "standard_errors": fitted.bse.to_dict(),
            "p_values": fitted.pvalues.to_dict(),
            "confidence_intervals": fitted.conf_int(alpha=alpha).to_dict(),
        }

    if method == "regression_f_test":
        y, x = regression_data(frame, variables)
        fitted = sm.OLS(y, x).fit()
        return {
            "n": int(fitted.nobs),
            "degrees_of_freedom": [float(fitted.df_model), float(fitted.df_resid)],
            "f_statistic": fitted.fvalue,
            "p_value": fitted.f_pvalue,
            "coefficients": fitted.params.to_dict(),
        }

    if method == "iv_2sls":
        require_columns(frame, variables, ["outcome", "exposure", "instruments"])
        outcome = str(variables["outcome"])
        exposure = str(variables["exposure"])
        instruments = variables["instruments"]
        instruments = [str(value) for value in instruments] if isinstance(
            instruments, list
        ) else [str(instruments)]
        covariates = variables.get("covariates") or []
        covariates = [str(value) for value in covariates] if isinstance(
            covariates, list
        ) else [str(covariates)]
        columns = [outcome, exposure, *instruments, *covariates]
        absent = [name for name in columns if name not in frame.columns]
        if absent:
            raise ValueError(f"columns do not exist: {', '.join(absent)}")
        clean = frame[columns].dropna()
        exogenous = sm.add_constant(
            pd.get_dummies(clean[[exposure, *covariates]], drop_first=True, dtype=float),
            has_constant="add",
        )
        instrument_matrix = sm.add_constant(
            pd.get_dummies(clean[[*instruments, *covariates]], drop_first=True, dtype=float),
            has_constant="add",
        )
        fitted = IV2SLS(
            clean[outcome].astype(float),
            exogenous.astype(float),
            instrument_matrix.astype(float),
        ).fit()
        return {
            "n": int(fitted.nobs),
            "coefficients": fitted.params.to_dict(),
            "standard_errors": fitted.bse.to_dict(),
            "p_values": fitted.pvalues.to_dict(),
        }

    if method in {"logistic_glm", "poisson_glm", "negative_binomial_glm", "probit_glm"}:
        y, x = regression_data(frame, variables)
        if method == "probit_glm":
            fitted = sm.Probit(y, x).fit(disp=False)
            return {
                "n": int(fitted.nobs),
                "coefficients": fitted.params.to_dict(),
                "standard_errors": fitted.bse.to_dict(),
                "p_values": fitted.pvalues.to_dict(),
                "aic": fitted.aic,
            }
        family = {
            "logistic_glm": sm.families.Binomial(),
            "poisson_glm": sm.families.Poisson(),
            "negative_binomial_glm": sm.families.NegativeBinomial(),
        }[method]
        fitted = sm.GLM(y, x, family=family).fit()
        return {
            "n": int(fitted.nobs),
            "coefficients": fitted.params.to_dict(),
            "standard_errors": fitted.bse.to_dict(),
            "p_values": fitted.pvalues.to_dict(),
            "aic": fitted.aic,
        }

    if method == "tobit_regression":
        y_series, x_frame = regression_data(frame, variables)
        y = y_series.to_numpy(dtype=float)
        x = x_frame.to_numpy(dtype=float)
        options = dict(request.get("analysis_options") or {})
        left = float(options.get("left_censor", np.min(y)))
        right_value = options.get("right_censor")
        right = float(right_value) if right_value is not None else math.inf
        left_mask = y <= left + 1e-12
        right_mask = y >= right - 1e-12 if math.isfinite(right) else np.zeros(len(y), dtype=bool)
        middle = ~(left_mask | right_mask)

        def negative_log_likelihood(parameters: np.ndarray) -> float:
            beta = parameters[:-1]
            sigma = math.exp(float(parameters[-1]))
            location = x @ beta
            likelihood = np.empty(len(y), dtype=float)
            likelihood[left_mask] = st.norm.cdf((left - location[left_mask]) / sigma)
            if math.isfinite(right):
                likelihood[right_mask] = st.norm.sf((right - location[right_mask]) / sigma)
            likelihood[middle] = st.norm.pdf((y[middle] - location[middle]) / sigma) / sigma
            return float(-np.sum(np.log(np.clip(likelihood, 1e-300, None))))

        initial_beta = np.linalg.lstsq(x, y, rcond=None)[0]
        initial_sigma = max(float(np.std(y - x @ initial_beta)), 1e-4)
        initial = np.append(initial_beta, math.log(initial_sigma))
        fitted = minimize(negative_log_likelihood, initial, method="BFGS")
        hessian = approx_hess(fitted.x, negative_log_likelihood)
        covariance = np.linalg.pinv(hessian)
        standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
        z_scores = fitted.x / np.where(standard_errors > 0, standard_errors, np.nan)
        p_values = 2 * st.norm.sf(np.abs(z_scores))
        names = [*x_frame.columns.tolist(), "log_sigma"]
        return {
            "n": len(y),
            "converged": bool(fitted.success),
            "left_censor": left,
            "right_censor": None if not math.isfinite(right) else right,
            "coefficients": dict(zip(names, fitted.x, strict=True)),
            "standard_errors": dict(zip(names, standard_errors, strict=True)),
            "p_values": dict(zip(names, p_values, strict=True)),
            "log_likelihood": -float(fitted.fun),
        }

    if method == "firth_logistic":
        y, x_frame = regression_data(frame, variables)
        x = x_frame.to_numpy(dtype=float)
        outcome = y.to_numpy(dtype=float)
        coefficients = np.zeros(x.shape[1], dtype=float)
        converged = False
        for _iteration in range(1, 101):
            linear = np.clip(x @ coefficients, -30, 30)
            probability = 1 / (1 + np.exp(-linear))
            weights = np.clip(probability * (1 - probability), 1e-9, None)
            information = x.T @ (weights[:, None] * x)
            inverse = np.linalg.pinv(information)
            leverage = weights * np.sum((x @ inverse) * x, axis=1)
            adjusted_score = x.T @ (
                outcome - probability + (0.5 - probability) * leverage
            )
            step = inverse @ adjusted_score
            if np.max(np.abs(step)) > 5:
                step *= 5 / np.max(np.abs(step))
            coefficients += step
            if np.max(np.abs(step)) < 1e-8:
                converged = True
                break
        standard_errors = np.sqrt(np.diag(inverse))
        z_scores = coefficients / standard_errors
        p_values = 2 * st.norm.sf(np.abs(z_scores))
        return {
            "n": len(outcome),
            "converged": converged,
            "iterations": _iteration,
            "coefficients": dict(zip(x_frame.columns, coefficients, strict=True)),
            "standard_errors": dict(zip(x_frame.columns, standard_errors, strict=True)),
            "p_values": dict(zip(x_frame.columns, p_values, strict=True)),
            "algorithm": "Jeffreys-prior mean-bias-reduced logistic score",
        }

    if method == "gee":
        require_columns(frame, variables, ["outcome", "predictors", "cluster"])
        y, x = regression_data(frame, variables)
        clusters = frame.loc[y.index, variables["cluster"]]
        fitted = sm.GEE(y, x, groups=clusters, cov_struct=sm.cov_struct.Exchangeable()).fit()
        return {
            "clusters": int(clusters.nunique()),
            "coefficients": fitted.params.to_dict(),
            "standard_errors": fitted.bse.to_dict(),
            "p_values": fitted.pvalues.to_dict(),
        }

    if method == "mixed_effects":
        require_columns(frame, variables, ["outcome", "predictors", "cluster"])
        y, x = regression_data(frame, variables)
        clusters = frame.loc[y.index, variables["cluster"]]
        fitted = sm.MixedLM(y, x, groups=clusters).fit(reml=True, method="lbfgs")
        return {
            "clusters": int(clusters.nunique()),
            "fixed_effects": fitted.fe_params.to_dict(),
            "standard_errors": fitted.bse_fe.to_dict(),
            "random_effect_covariance": fitted.cov_re.to_dict(),
        }

    if method in {"difference_in_means", "randomization_inference"}:
        require_columns(frame, variables, ["outcome", "treatment"])
        left, right, levels = groups(frame, variables["outcome"], variables["treatment"])
        observed = float(np.mean(right) - np.mean(left))
        result = {"levels": levels, "difference_in_means": observed, "n": [len(left), len(right)]}
        if method == "randomization_inference":
            rng = np.random.default_rng(int(request.get("seed", 42)))
            clean = frame[[variables["outcome"], variables["treatment"]]].dropna()
            values = clean[variables["outcome"]].astype(float).to_numpy()
            assigned = clean[variables["treatment"]].to_numpy()
            count = int(np.sum(assigned == levels[1]))
            permutations = int(min(5000, request.get("permutations", 2000)))
            null = np.empty(permutations)
            for index in range(permutations):
                treated = rng.choice(len(values), size=count, replace=False)
                mask = np.zeros(len(values), dtype=bool)
                mask[treated] = True
                null[index] = values[mask].mean() - values[~mask].mean()
            result["randomization_p_value"] = float(
                (1 + np.sum(np.abs(null) >= abs(observed))) / (permutations + 1)
            )
            result["permutations"] = permutations
        return result

    if method == "ipw":
        require_columns(frame, variables, ["outcome", "treatment", "predictors"])
        outcome = str(variables["outcome"])
        treatment = str(variables["treatment"])
        predictors = [str(value) for value in variables["predictors"]]
        clean = frame[[outcome, treatment, *predictors]].dropna()
        x = pd.get_dummies(clean[predictors], drop_first=True, dtype=float)
        propensity_model = LogisticRegression(max_iter=1000).fit(x, clean[treatment])
        propensity = np.clip(propensity_model.predict_proba(x)[:, 1], 0.01, 0.99)
        treated = clean[treatment].astype(int).to_numpy()
        weights = treated / propensity + (1 - treated) / (1 - propensity)
        outcome_values = clean[outcome].astype(float).to_numpy()
        effect = np.average(
            outcome_values[treated == 1], weights=weights[treated == 1]
        ) - np.average(outcome_values[treated == 0], weights=weights[treated == 0])
        effective_n = float(weights.sum() ** 2 / np.square(weights).sum())
        return {
            "weighted_mean_difference": effect,
            "propensity_range": [float(propensity.min()), float(propensity.max())],
            "effective_sample_size": effective_n,
        }

    if method == "multiple_imputation":
        y, x = regression_data(frame, variables)
        missing = int(frame.isna().sum().sum())
        return {
            "status": "requires_confirmed_imputation_model",
            "complete_case_n": len(y),
            "missing_cells": missing,
            "design_matrix_columns": list(x.columns),
            "clarification": (
                "Declare the imputation variables and MAR justification before pooling."
            ),
        }

    if method in {"conjugate_bayes", "posterior_predictive"}:
        require_columns(frame, variables, ["outcome"])
        values = frame[variables["outcome"]].dropna().astype(float).to_numpy()
        prior_mean = float(request.get("prior_mean", 0.0))
        prior_sd = float(request.get("prior_sd", 10.0))
        variance = float(np.var(values, ddof=1))
        posterior_variance = 1 / (1 / prior_sd**2 + len(values) / variance)
        posterior_mean = posterior_variance * (prior_mean / prior_sd**2 + np.sum(values) / variance)
        result = {
            "n": len(values),
            "posterior_mean": posterior_mean,
            "posterior_sd": math.sqrt(posterior_variance),
            "credible_interval": st.norm.interval(
                1 - alpha, loc=posterior_mean, scale=math.sqrt(posterior_variance)
            ),
        }
        if method == "posterior_predictive":
            rng = np.random.default_rng(int(request.get("seed", 42)))
            draws = rng.normal(
                posterior_mean, math.sqrt(variance + posterior_variance), size=(2000, len(values))
            )
            observed_skew = st.skew(values)
            predictive_skew = st.skew(draws, axis=1)
            result["skew_discrepancy_p"] = float(
                np.mean(np.abs(predictive_skew) >= abs(observed_skew))
            )
        return result

    if method == "calibrated_logistic":
        require_columns(frame, variables, ["outcome", "predictors"])
        outcome = str(variables["outcome"])
        predictors = [str(value) for value in variables["predictors"]]
        clean = frame[[outcome, *predictors]].dropna()
        x = pd.get_dummies(clean[predictors], drop_first=True, dtype=float)
        y = clean[outcome].astype(int)
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(request.get("seed", 42)))
        probabilities = cross_val_predict(
            LogisticRegression(max_iter=1000), x, y, cv=folds, method="predict_proba"
        )[:, 1]
        observed, predicted = calibration_curve(y, probabilities, n_bins=10, strategy="quantile")
        return {
            "n": len(y),
            "brier_score": brier_score_loss(y, probabilities),
            "roc_auc": roc_auc_score(y, probabilities),
            "calibration_observed": observed,
            "calibration_predicted": predicted,
        }

    if method == "blocked_time_series_cv":
        require_columns(frame, variables, ["outcome", "predictors", "time"])
        outcome = str(variables["outcome"])
        predictors = [str(value) for value in variables["predictors"]]
        clean = (
            frame[[variables["time"], outcome, *predictors]].dropna().sort_values(variables["time"])
        )
        x = pd.get_dummies(clean[predictors], drop_first=True, dtype=float).to_numpy()
        y = clean[outcome].astype(float).to_numpy()
        splitter = TimeSeriesSplit(n_splits=5)
        errors = []
        for train_index, test_index in splitter.split(x):
            fitted = np.linalg.lstsq(
                np.column_stack([np.ones(len(train_index)), x[train_index]]),
                y[train_index],
                rcond=None,
            )[0]
            predicted = np.column_stack([np.ones(len(test_index)), x[test_index]]) @ fitted
            errors.extend((y[test_index] - predicted).tolist())
        return {
            "n": len(y),
            "folds": 5,
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "mae": float(np.mean(np.abs(errors))),
            "chronology_preserved": True,
        }

    if method in {"cox_ph", "logrank"}:
        raise ValueError(f"{method} must run in the locked R backend")
    raise ValueError(f"unsupported method_id: {method}")


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    frame = load_frame(Path(request["data_path"]))
    result = result_for_method(frame, request)
    print(json.dumps({"status": "ok", "result": serializable(result)}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        raise SystemExit(2) from error

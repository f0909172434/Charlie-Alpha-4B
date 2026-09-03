from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

ToolName = Literal["python", "r", "none"]


@dataclass(frozen=True)
class Procedure:
    method_id: str
    name: str
    families: tuple[str, ...]
    tool: ToolName
    uncertainty: str
    assumptions: tuple[str, ...]
    strengths: tuple[str, ...]
    cost: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DGPFamily:
    family_id: str
    name: str
    domain: str
    parameters: dict[str, tuple[float, float]]
    central_method: str
    candidate_methods: tuple[str, ...]
    clarification_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROCEDURES: tuple[Procedure, ...] = (
    Procedure(
        "independent_t",
        "Pooled two-sample t test",
        ("group_comparison", "probability_distribution"),
        "python",
        "pooled standard error and t interval",
        ("independent observations", "approximately normal errors", "equal variances"),
        ("efficient under equal-variance Gaussian sampling",),
        0.10,
    ),
    Procedure(
        "welch_t",
        "Welch two-sample t test",
        ("group_comparison", "probability_distribution"),
        "python",
        "Welch-Satterthwaite interval",
        ("independent observations", "finite group variances"),
        ("valid under unequal variances", "good default for independent means"),
        0.12,
    ),
    Procedure(
        "mann_whitney",
        "Mann-Whitney rank test",
        ("group_comparison",),
        "python",
        "rank-based asymptotic or exact test",
        ("independent observations", "continuous or tie-adjusted outcome"),
        ("resistant to heavy tails",),
        0.18,
    ),
    Procedure(
        "paired_t",
        "Paired t test",
        ("paired_comparison", "group_comparison"),
        "python",
        "interval for the mean within-pair difference",
        ("known pairs", "approximately normal pair differences"),
        ("uses within-pair dependence",),
        0.10,
    ),
    Procedure(
        "wilcoxon_signed_rank",
        "Wilcoxon signed-rank test",
        ("paired_comparison", "group_comparison"),
        "r",
        "signed-rank test and pseudomedian interval",
        ("known pairs", "symmetric difference distribution"),
        ("robust paired comparison",),
        0.18,
    ),
    Procedure(
        "chi_square",
        "Pearson chi-square test",
        ("categorical",),
        "python",
        "chi-square reference distribution",
        ("independent counts", "adequate expected cell counts"),
        ("fast for general contingency tables",),
        0.08,
    ),
    Procedure(
        "fisher_exact",
        "Fisher exact test",
        ("categorical",),
        "python",
        "conditional exact test",
        ("independent 2 by 2 counts", "fixed margins for its exact interpretation"),
        ("finite-sample Type I error control for sparse 2 by 2 tables",),
        0.35,
    ),
    Procedure(
        "two_proportion",
        "Two-proportion score test",
        ("categorical",),
        "python",
        "score interval for a risk difference",
        ("independent Bernoulli observations", "non-sparse groups"),
        ("directly targets a risk difference",),
        0.10,
    ),
    Procedure(
        "ols",
        "Ordinary least squares",
        ("linear_robust", "clustered_repeated", "missing_selection", "experimental_causal"),
        "python",
        "model-based standard errors",
        ("linear conditional mean", "independent homoskedastic errors for classical inference"),
        ("efficient under the classical linear model",),
        0.08,
    ),
    Procedure(
        "hc3_ols",
        "OLS with HC3 robust covariance",
        ("linear_robust", "experimental_causal"),
        "python",
        "HC3 sandwich interval",
        ("independent sampling units", "linear conditional mean"),
        ("heteroskedasticity-robust", "small-sample leverage correction"),
        0.16,
    ),
    Procedure(
        "huber_regression",
        "Huber robust regression",
        ("linear_robust",),
        "python",
        "sandwich interval around an M-estimator",
        ("independent sampling units", "specified robust score"),
        ("limits the influence of outcome outliers",),
        0.30,
    ),
    Procedure(
        "logistic_glm",
        "Logistic regression",
        ("binary_count_glm", "predictive_calibration"),
        "python",
        "Wald or likelihood-ratio interval",
        ("Bernoulli outcome", "correct link and linear predictor", "no complete separation"),
        ("interpretable conditional odds ratios",),
        0.16,
    ),
    Procedure(
        "firth_logistic",
        "Bias-reduced logistic regression",
        ("binary_count_glm",),
        "python",
        "penalized-likelihood interval",
        ("Bernoulli outcome", "correct link and linear predictor"),
        ("finite estimates under separation", "reduced small-sample bias"),
        0.45,
    ),
    Procedure(
        "poisson_glm",
        "Poisson regression",
        ("binary_count_glm",),
        "python",
        "model-based log-rate interval",
        ("count outcome", "conditional mean equals variance", "correct log-linear mean"),
        ("efficient for equidispersed counts",),
        0.14,
    ),
    Procedure(
        "negative_binomial_glm",
        "Negative-binomial regression",
        ("binary_count_glm",),
        "r",
        "dispersion-adjusted log-rate interval",
        ("count outcome", "negative-binomial mean-variance relationship"),
        ("handles overdispersed counts",),
        0.28,
    ),
    Procedure(
        "gee",
        "Generalized estimating equations",
        ("clustered_repeated",),
        "python",
        "cluster-sandwich interval",
        ("independent clusters", "adequate number of clusters", "correct marginal mean"),
        ("population-average inference with within-cluster dependence",),
        0.34,
    ),
    Procedure(
        "mixed_effects",
        "Random-intercept mixed model",
        ("clustered_repeated",),
        "r",
        "model-based conditional interval",
        ("correct random-effects structure", "approximately Gaussian random effects"),
        ("partial pooling", "cluster-specific modeling"),
        0.42,
    ),
    Procedure(
        "cox_ph",
        "Cox proportional-hazards model",
        ("survival",),
        "r",
        "partial-likelihood hazard-ratio interval",
        ("independent censoring conditional on covariates", "proportional hazards"),
        ("covariate-adjusted time-to-event inference",),
        0.32,
    ),
    Procedure(
        "logrank",
        "Kaplan-Meier and log-rank comparison",
        ("survival",),
        "r",
        "nonparametric survival curves and log-rank test",
        ("independent censoring", "proportional hazards for optimal power"),
        ("transparent unadjusted survival comparison",),
        0.20,
    ),
    Procedure(
        "multiple_imputation",
        "Multiple imputation with pooled inference",
        ("missing_selection",),
        "r",
        "Rubin pooled interval",
        ("missing at random conditional on imputation variables", "congenial imputation model"),
        ("propagates imputation uncertainty",),
        0.55,
    ),
    Procedure(
        "ipw",
        "Inverse-probability weighting",
        ("missing_selection", "experimental_causal"),
        "python",
        "robust weighted-estimating-equation interval",
        ("exchangeability", "positivity", "correct selection or treatment model"),
        ("targets a marginal estimand", "corrects observed selection when modeled"),
        0.40,
    ),
    Procedure(
        "difference_in_means",
        "Randomized difference in means",
        ("experimental_causal",),
        "python",
        "design-based standard error",
        ("known randomized assignment", "no interference for unit-level effects"),
        ("minimal modeling under randomization",),
        0.08,
    ),
    Procedure(
        "ancova",
        "Randomized ANCOVA",
        ("experimental_causal",),
        "python",
        "randomization-compatible HC3 interval",
        ("known randomized assignment", "pre-treatment covariates"),
        ("precision gain from prognostic baseline covariates",),
        0.18,
    ),
    Procedure(
        "randomization_inference",
        "Randomization inference",
        ("experimental_causal",),
        "python",
        "assignment-based randomization distribution",
        ("known assignment mechanism", "sharp-null interpretation unless inverted"),
        ("finite-sample design-based validity",),
        0.44,
    ),
    Procedure(
        "conjugate_bayes",
        "Conjugate Bayesian estimation",
        ("probability_distribution", "bayesian_check"),
        "python",
        "posterior credible interval",
        ("likelihood and prior are explicitly justified",),
        ("fast exact posterior updates",),
        0.12,
    ),
    Procedure(
        "posterior_predictive",
        "Posterior predictive model check",
        ("probability_distribution", "bayesian_check"),
        "python",
        "posterior predictive discrepancy distribution",
        ("explicit generative model", "meaningful discrepancy statistic"),
        ("detects likelihood misfit", "checks what the fitted model can reproduce"),
        0.30,
    ),
    Procedure(
        "calibrated_logistic",
        "Cross-fitted calibrated prediction model",
        ("predictive_calibration", "time_series_leakage"),
        "python",
        "cross-validated calibration, discrimination, and uncertainty",
        ("representative validation split", "all preprocessing fit within folds"),
        ("measures out-of-sample calibration", "reduces leakage"),
        0.38,
    ),
    Procedure(
        "blocked_time_series_cv",
        "Rolling-origin time-series validation",
        ("time_series_leakage",),
        "python",
        "forward-chaining forecast errors and intervals",
        ("time ordering is preserved", "forecast horizon is declared"),
        ("prevents future-to-past leakage", "matches deployment chronology"),
        0.36,
    ),
)

# These fixed procedures are available to the audited data agent and its external
# evaluations, but are not ranked by the 28-procedure DGP training surface.
AUXILIARY_PROCEDURES: tuple[Procedure, ...] = (
    Procedure(
        "binomial_test",
        "Exact one-sample binomial test",
        ("auxiliary",),
        "python",
        "exact binomial tail probability and confidence interval",
        ("independent Bernoulli trials", "declared null probability"),
        ("finite-sample test for one proportion",),
        0.08,
    ),
    Procedure(
        "spearman_correlation",
        "Spearman rank correlation",
        ("auxiliary",),
        "python",
        "rank-correlation test",
        ("paired observations", "monotone association is the target"),
        ("resistant to monotone nonlinear scaling",),
        0.10,
    ),
    Procedure(
        "kruskal_wallis",
        "Kruskal-Wallis rank test",
        ("auxiliary",),
        "python",
        "rank-based omnibus test",
        ("independent groups", "continuous or ordinal outcome"),
        ("nonparametric comparison of more than two groups",),
        0.14,
    ),
    Procedure(
        "probit_glm",
        "Probit regression",
        ("auxiliary",),
        "python",
        "maximum-likelihood coefficient tests",
        ("binary outcome", "probit link and linear predictor"),
        ("latent-normal binary response model",),
        0.17,
    ),
    Procedure(
        "regression_f_test",
        "Regression F test",
        ("auxiliary",),
        "python",
        "model or nested-model F reference distribution",
        ("declared linear restrictions", "Gaussian linear-model reference for exactness"),
        ("joint test of multiple coefficients",),
        0.10,
    ),
    Procedure(
        "iv_2sls",
        "Instrumental-variables two-stage least squares",
        ("auxiliary",),
        "python",
        "2SLS covariance and coefficient tests",
        ("instrument relevance", "exclusion", "instrument independence"),
        ("identifies a declared IV estimand under valid instruments",),
        0.30,
    ),
    Procedure(
        "tobit_regression",
        "Censored-normal Tobit regression",
        ("auxiliary",),
        "python",
        "observed-data maximum-likelihood coefficient tests",
        ("declared censoring thresholds", "latent Gaussian linear model"),
        ("models point-mass censoring rather than dropping censored rows",),
        0.32,
    ),
)


FAMILIES: tuple[DGPFamily, ...] = (
    DGPFamily(
        "group_comparison",
        "Independent and paired group comparisons",
        "inference_and_design",
        {
            "n": (16, 240),
            "effect": (0.0, 0.9),
            "variance_ratio": (0.3, 4.0),
            "tail_weight": (0.0, 0.35),
            "pair_correlation": (0.0, 0.9),
        },
        "welch_t",
        ("independent_t", "welch_t", "mann_whitney", "paired_t", "wilcoxon_signed_rank"),
        ("sampling_unit", "study_design", "dependence", "estimand"),
    ),
    DGPFamily(
        "categorical",
        "Categorical and proportion data",
        "inference_and_design",
        {
            "n": (20, 500),
            "baseline_probability": (0.01, 0.75),
            "risk_difference": (0.0, 0.25),
            "imbalance": (0.2, 0.8),
        },
        "two_proportion",
        ("chi_square", "fisher_exact", "two_proportion"),
        ("sampling_unit", "study_design", "estimand"),
    ),
    DGPFamily(
        "linear_robust",
        "Linear and robust regression",
        "inference_and_design",
        {
            "n": (24, 400),
            "effect": (0.0, 0.8),
            "heteroskedasticity": (0.0, 2.5),
            "outlier_fraction": (0.0, 0.18),
            "leverage": (0.0, 0.3),
        },
        "hc3_ols",
        ("ols", "hc3_ols", "huber_regression"),
        ("sampling_unit", "estimand", "outcome_type"),
    ),
    DGPFamily(
        "binary_count_glm",
        "Binary and count generalized linear models",
        "inference_and_design",
        {
            "n": (30, 500),
            "effect": (0.0, 1.2),
            "event_rate": (0.01, 0.6),
            "overdispersion": (1.0, 5.0),
            "separation": (0.0, 1.0),
        },
        "logistic_glm",
        ("logistic_glm", "firth_logistic", "poisson_glm", "negative_binomial_glm"),
        ("outcome_type", "estimand", "sampling_unit"),
    ),
    DGPFamily(
        "clustered_repeated",
        "Clustered and repeated measurements",
        "inference_and_design",
        {"clusters": (6, 80), "cluster_size": (2, 20), "effect": (0.0, 0.8), "icc": (0.0, 0.65)},
        "gee",
        ("ols", "gee", "mixed_effects"),
        ("sampling_unit", "dependence", "estimand"),
    ),
    DGPFamily(
        "survival",
        "Survival analysis",
        "inference_and_design",
        {
            "n": (40, 500),
            "log_hazard_ratio": (0.0, 0.9),
            "censoring": (0.05, 0.65),
            "non_ph": (0.0, 1.0),
        },
        "cox_ph",
        ("cox_ph", "logrank"),
        ("outcome_type", "estimand", "censoring", "study_design"),
    ),
    DGPFamily(
        "missing_selection",
        "Missing data and selection bias",
        "inference_and_design",
        {
            "n": (40, 600),
            "effect": (0.0, 0.8),
            "missing_rate": (0.05, 0.6),
            "selection_strength": (0.0, 2.5),
            "positivity": (0.02, 0.5),
        },
        "multiple_imputation",
        ("ols", "multiple_imputation", "ipw"),
        ("missingness", "estimand", "sampling_unit"),
    ),
    DGPFamily(
        "experimental_causal",
        "Experiments and causal inference",
        "inference_and_design",
        {
            "n": (30, 600),
            "effect": (0.0, 0.8),
            "assignment_probability": (0.1, 0.9),
            "prognostic_strength": (0.0, 0.9),
            "confounding": (0.0, 1.5),
        },
        "difference_in_means",
        ("difference_in_means", "ancova", "randomization_inference", "ipw", "hc3_ols"),
        ("study_design", "estimand", "sampling_unit", "assignment_mechanism"),
    ),
    DGPFamily(
        "probability_distribution",
        "Probability distributions",
        "probability_and_bayes",
        {
            "n": (8, 300),
            "skew": (0.0, 3.0),
            "tail_weight": (0.0, 0.4),
            "prior_strength": (0.0, 30.0),
        },
        "conjugate_bayes",
        ("conjugate_bayes", "posterior_predictive", "welch_t"),
        ("estimand", "outcome_type", "prior"),
    ),
    DGPFamily(
        "bayesian_check",
        "Bayesian estimation and model checking",
        "probability_and_bayes",
        {
            "n": (10, 300),
            "prior_bias": (0.0, 2.5),
            "prior_strength": (0.0, 60.0),
            "model_misspecification": (0.0, 1.0),
        },
        "posterior_predictive",
        ("conjugate_bayes", "posterior_predictive"),
        ("estimand", "prior", "likelihood", "diagnostics"),
    ),
    DGPFamily(
        "predictive_calibration",
        "Prediction and calibration",
        "prediction_and_analysis",
        {
            "n": (80, 1200),
            "prevalence": (0.03, 0.7),
            "signal": (0.0, 2.0),
            "shift": (0.0, 1.2),
            "leakage": (0.0, 1.0),
        },
        "calibrated_logistic",
        ("logistic_glm", "calibrated_logistic"),
        ("prediction_horizon", "validation_design", "outcome_type"),
    ),
    DGPFamily(
        "time_series_leakage",
        "Time series and data leakage",
        "prediction_and_analysis",
        {
            "n": (60, 1000),
            "autocorrelation": (0.0, 0.95),
            "drift": (0.0, 1.0),
            "leakage": (0.0, 1.0),
            "horizon": (1, 20),
        },
        "blocked_time_series_cv",
        ("calibrated_logistic", "blocked_time_series_cv"),
        ("time_index", "prediction_horizon", "validation_design"),
    ),
)

PROCEDURE_BY_ID = {procedure.method_id: procedure for procedure in PROCEDURES}
AGENT_PROCEDURES = (*PROCEDURES, *AUXILIARY_PROCEDURES)
AGENT_PROCEDURE_BY_ID = {procedure.method_id: procedure for procedure in AGENT_PROCEDURES}
FAMILY_BY_ID = {family.family_id: family for family in FAMILIES}


def validate_catalog() -> list[str]:
    errors: list[str] = []
    if len(PROCEDURES) != 28:
        errors.append(f"expected 28 procedures, found {len(PROCEDURES)}")
    if len(PROCEDURE_BY_ID) != len(PROCEDURES):
        errors.append("duplicate procedure method_id")
    if len(AGENT_PROCEDURE_BY_ID) != len(AGENT_PROCEDURES):
        errors.append("duplicate agent procedure method_id")
    if len(FAMILIES) != 12:
        errors.append(f"expected 12 DGP families, found {len(FAMILIES)}")
    if len(FAMILY_BY_ID) != len(FAMILIES):
        errors.append("duplicate DGP family_id")
    for family in FAMILIES:
        if family.central_method not in family.candidate_methods:
            errors.append(f"{family.family_id}: central method is not a candidate")
        for method_id in family.candidate_methods:
            method = PROCEDURE_BY_ID.get(method_id)
            if method is None:
                errors.append(f"{family.family_id}: unknown method {method_id}")
            elif family.family_id not in method.families and not (
                family.family_id == "group_comparison" and "paired_comparison" in method.families
            ):
                errors.append(f"{family.family_id}: {method_id} does not declare this family")
    return errors


def catalog_manifest() -> dict[str, Any]:
    errors = validate_catalog()
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "schema_version": 1,
        "procedures": [procedure.to_dict() for procedure in PROCEDURES],
        "auxiliary_agent_procedures": [
            procedure.to_dict() for procedure in AUXILIARY_PROCEDURES
        ],
        "families": [family.to_dict() for family in FAMILIES],
        "special_actions": [
            {
                "method_id": "needs_clarification",
                "description": (
                    "Ask a concrete design question instead of assuming an estimand, "
                    "sampling unit, dependence, missingness, or assignment mechanism."
                ),
            }
        ],
    }

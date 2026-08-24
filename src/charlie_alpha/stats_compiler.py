from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from .io_utils import canonical_hash
from .stats_catalog import AGENT_PROCEDURE_BY_ID

REQUIRED_VARIABLES: dict[str, tuple[str, ...]] = {
    "independent_t": ("outcome", "group"),
    "welch_t": ("outcome", "group"),
    "mann_whitney": ("outcome", "group"),
    "paired_t": ("before", "after"),
    "wilcoxon_signed_rank": ("before", "after"),
    "chi_square": ("outcome", "group"),
    "fisher_exact": ("outcome", "group"),
    "two_proportion": ("outcome", "group"),
    "ols": ("outcome", "predictors"),
    "hc3_ols": ("outcome", "predictors"),
    "huber_regression": ("outcome", "predictors"),
    "logistic_glm": ("outcome", "predictors"),
    "firth_logistic": ("outcome", "predictors"),
    "poisson_glm": ("outcome", "predictors"),
    "negative_binomial_glm": ("outcome", "predictors"),
    "gee": ("outcome", "predictors", "cluster"),
    "mixed_effects": ("outcome", "predictors", "cluster"),
    "cox_ph": ("time", "event", "predictors"),
    "logrank": ("time", "event", "group"),
    "multiple_imputation": ("outcome", "predictors"),
    "ipw": ("outcome", "treatment", "predictors"),
    "difference_in_means": ("outcome", "treatment"),
    "ancova": ("outcome", "predictors"),
    "randomization_inference": ("outcome", "treatment"),
    "conjugate_bayes": ("outcome",),
    "posterior_predictive": ("outcome",),
    "calibrated_logistic": ("outcome", "predictors"),
    "blocked_time_series_cv": ("outcome", "predictors", "time"),
    "binomial_test": ("outcome",),
    "spearman_correlation": ("x", "y"),
    "kruskal_wallis": ("outcome", "group"),
    "probit_glm": ("outcome", "predictors"),
    "regression_f_test": ("outcome", "predictors"),
    "iv_2sls": ("outcome", "exposure", "instruments"),
    "tobit_regression": ("outcome", "predictors"),
}

_GLOSSARY_LINE = re.compile(r"(?m)^\s*[-*]\s+`?([A-Za-z_][A-Za-z0-9_. -]*)`?\s*:\s*(.+?)\s*$")
_TAG = re.compile(r"\[([A-Za-z_ -]+)\]")
_PERCENT = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*%")
_NUMBER_PROBABILITY = re.compile(
    r"(?:null|background|expected|基準|背景|基准)[^\d]{0,30}(0(?:\.\d+)?|1(?:\.0+)?)",
    re.IGNORECASE,
)
_STRUCTURED_FIELD = re.compile(
    r"(?im)^\s*[-*]?\s*(type|unit of observation|unit of analysis|unit of randomization|"
    r"randomization level|panel structure|missingness|研究設計|研究设计|觀察單位|观察单位|"
    r"抽樣單位|抽样单位|隨機化層級|随机化层级)\s*:\s*(.+?)\s*$"
)


@dataclass(frozen=True)
class GlossaryItem:
    name: str
    description: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class CompiledScaffold:
    fingerprint: str
    candidate_method_ids: tuple[str, ...]
    method_plans: dict[str, dict[str, Any]]
    role_candidates: dict[str, tuple[str, ...]]
    evidence: tuple[str, ...]
    questions: tuple[str, ...]
    confidence: float
    auto_ready: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _columns(summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for file_index, summary in enumerate(summaries):
        rows = int(summary.get("rows", 0))
        for value in summary.get("columns", []):
            if not isinstance(value, dict) or value.get("name") is None:
                continue
            name = str(value["name"])
            if name in result:
                file_indices = sorted(
                    {
                        *result[name].get("file_indices", [result[name]["file_index"]]),
                        file_index,
                    }
                )
                result[name] = {
                    **value,
                    "rows": rows,
                    "file_index": -1,
                    "file_indices": file_indices,
                }
            else:
                result[name] = {
                    **value,
                    "rows": rows,
                    "file_index": file_index,
                    "file_indices": [file_index],
                }
    return result


def _kind(column: dict[str, Any]) -> str:
    name = str(column.get("name", "")).lower()
    dtype = str(column.get("dtype", "")).lower()
    unique = int(column.get("unique", 0))
    rows = max(1, int(column.get("rows", 0)))
    if name.endswith("_id") or name in {"id", "patient", "participant"}:
        return "identifier"
    if unique >= 0.9 * rows and ("id" in name or "object" in dtype or "string" in dtype):
        return "identifier"
    if unique == 2:
        return "binary"
    if unique <= 12:
        return "categorical"
    if any(token in dtype for token in ("int", "float", "double", "decimal")):
        return "continuous"
    return "text"


def _glossary(question: str, available: set[str]) -> list[GlossaryItem]:
    lowered = {name.lower(): name for name in available}
    items: list[GlossaryItem] = []
    for match in _GLOSSARY_LINE.finditer(question):
        raw_name = match.group(1).strip().strip("`")
        name = lowered.get(raw_name.lower())
        if name is None:
            continue
        description = match.group(2).strip()
        tags = tuple(sorted({tag.upper().replace(" ", "_") for tag in _TAG.findall(description)}))
        items.append(GlossaryItem(name=name, description=description, tags=tags))
    return items


def _field_context(question: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in _STRUCTURED_FIELD.finditer(question):
        fields[match.group(1).strip().lower()] = match.group(2).strip()
    return fields


def _first(values: list[str]) -> str | None:
    return values[0] if values else None


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _tagged(items: list[GlossaryItem], tag: str) -> list[str]:
    return [item.name for item in items if tag in item.tags]


def _described(items: list[GlossaryItem], pattern: str) -> list[str]:
    regex = re.compile(pattern, re.IGNORECASE)
    return [item.name for item in items if regex.search(f"{item.name} {item.description}")]


def _mentioned_columns(question: str, available: list[str]) -> list[str]:
    core = re.split(
        r"(?i)variable glossary|data schema|變數詞彙|变量词汇|欄位說明|字段说明",
        question,
        maxsplit=1,
    )[0]
    lowered = core.lower()
    return [name for name in available if re.search(rf"\b{re.escape(name.lower())}\b", lowered)]


def _null_probability(question: str) -> float | None:
    match = _PERCENT.search(question)
    if match:
        return float(match.group(1)) / 100.0
    match = _NUMBER_PROBABILITY.search(question)
    if match:
        return float(match.group(1))
    return None


def _declared_preprocessing(
    question: str,
    columns: dict[str, dict[str, Any]],
    items: list[GlossaryItem],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    core = re.split(
        r"(?i)variable glossary|data schema|變數詞彙|变量词汇|欄位說明|字段说明",
        question,
        maxsplit=1,
    )[0]
    lowered = core.lower()
    filters: list[dict[str, Any]] = []
    recodes: list[dict[str, Any]] = []
    comparison_trigger = bool(
        re.search(
            r"compare|comparison|between|versus|\bvs\.?\b|focus on|restrict|limited to|"
            r"比較|对比|之間|之间|限於|限于|僅納入|仅纳入",
            lowered,
        )
    )

    def mentioned_level(value: Any, text: str) -> bool:
        if not isinstance(value, str) or not value.strip():
            return False
        return bool(
            re.search(
                rf"(?<!\w){re.escape(value.strip().lower())}(?!\w)",
                text,
            )
        )

    def label_is_requested(label: str) -> bool:
        normalized = re.sub(r"[-_/]+", " ", label.lower()).strip()
        if len(normalized) >= 3 and normalized in re.sub(r"[-_/]+", " ", lowered):
            return True
        distinctive = [
            token
            for token in re.findall(r"[a-z0-9]+", normalized)
            if token not in {"the", "and", "or", "post", "follow", "up", "group", "arm"}
            and len(token) >= 3
        ]
        return bool(distinctive and all(token in lowered for token in distinctive[:2]))

    for name, column in columns.items():
        levels = [value for value in column.get("levels", []) if value is not None]
        if not levels:
            continue
        mentioned = [value for value in levels if mentioned_level(value, lowered)]
        mapped_mentions: list[Any] = []
        item = next((item for item in items if item.name == name), None)
        if item is not None:
            for value in levels:
                mapping = re.search(
                    rf"(?:^|[:,(;])\s*{re.escape(str(value))}\s*=\s*([^,;)]+)",
                    item.description,
                    re.IGNORECASE,
                )
                if mapping and label_is_requested(mapping.group(1)):
                    mapped_mentions.append(value)
        explicit_assignment = re.search(
            rf"(?<!\w){re.escape(name)}\s*=\s*([^\s,;)]+)",
            core,
            re.IGNORECASE,
        )
        if explicit_assignment:
            assigned = explicit_assignment.group(1).strip("`'\"")
            mapped_mentions.extend(
                value for value in levels if str(value).lower() == assigned.lower()
            )
        mentioned = list(dict.fromkeys([*mentioned, *mapped_mentions]))
        excluded = [
            value
            for value in levels
            if isinstance(value, str)
            and re.search(
                rf"(?:exclude|excluding|except|排除|不納入|不纳入)[^\n]{{0,30}}"
                rf"(?<!\w){re.escape(value.lower())}(?!\w)",
                lowered,
            )
        ]
        if excluded:
            filters.append({"column": name, "operation": "exclude", "values": excluded})
            column["unique"] = max(0, int(column.get("unique", 0)) - len(excluded))
        elif (
            comparison_trigger
            and 0 < len(mentioned) < len(levels)
            and len(mentioned) <= 3
            and (len(mentioned) > 1 or explicit_assignment is not None or mapped_mentions)
        ):
            filters.append({"column": name, "operation": "include", "values": mentioned})
            column["unique"] = len(mentioned)

    for item in items:
        lowered_description = item.description.lower()
        if not re.search(r"all others|otherwise|其餘|其余|其他", lowered_description):
            continue
        levels = [value for value in columns[item.name].get("levels", []) if value is not None]
        collapsed = re.search(
            r"collapse to\s+(.+?)\s+(?:vs\.?|versus)",
            lowered_description,
        )
        if collapsed:
            target = collapsed.group(1).strip()
            positive = [value for value in levels if str(value).lower() == target]
        else:
            positive = [value for value in levels if mentioned_level(value, lowered_description)]
        if len(positive) == 1:
            recodes.append(
                {
                    "column": item.name,
                    "positive_values": positive,
                    "negative_rule": "all_other_observed_values",
                }
            )
            columns[item.name]["unique"] = 2
    return filters, recodes


def _alternative(question: str) -> str:
    lowered = question.lower()
    one_sided = (
        "greater than",
        "more frequently",
        "higher than",
        "increase",
        "lower than",
        "less frequently",
        "decrease",
        "大於",
        "高於",
        "增加",
        "小於",
        "低於",
        "減少",
        "大于",
        "高于",
        "小于",
        "低于",
    )
    if any(term in lowered for term in one_sided):
        if any(
            term in lowered
            for term in (
                "lower than",
                "less frequently",
                "decrease",
                "小於",
                "低於",
                "減少",
                "小于",
                "低于",
            )
        ):
            return "less"
        return "greater"
    return "two-sided"


def _study_context(question: str, fields: dict[str, str]) -> tuple[str | None, str | None, str]:
    sampling_unit = next(
        (
            fields[key]
            for key in (
                "unit of observation",
                "unit of analysis",
                "觀察單位",
                "观察单位",
                "抽樣單位",
                "抽样单位",
            )
            if key in fields
        ),
        None,
    )
    study_design = next(
        (fields[key] for key in ("type", "研究設計", "研究设计") if key in fields),
        None,
    )
    lowered = question.lower()
    if study_design is None:
        if re.search(r"randomi[sz]ed|隨機(?:分派|試驗)|随机(?:分派|试验)", lowered):
            study_design = "randomized study"
        elif re.search(
            r"observational|cohort|cross-sectional|觀察性|观察性|隊列|队列",
            lowered,
        ):
            study_design = "observational study"
    if any(
        term in lowered
        for term in ("cluster-random", "cluster randomized", "群集隨機", "整群隨機", "整群随机")
    ):
        dependence = "clustered at the declared randomization unit"
    elif any(
        term in lowered
        for term in (
            "panel structure",
            "repeated",
            "longitudinal",
            "within-person",
            "追蹤",
            "追踪",
            "重複量測",
            "重复测量",
        )
    ):
        dependence = "repeated observations within the declared sampling unit"
    elif sampling_unit:
        dependence = "independent across declared sampling units unless the design states otherwise"
    else:
        dependence = "undeclared"
    return sampling_unit, study_design, dependence


def _cluster_column(
    items: list[GlossaryItem], columns: dict[str, dict[str, Any]], question: str
) -> str | None:
    explicit = _described(
        items,
        r"cluster identifier|village identifier|school identifier|household identifier|"
        r"participant identifier|individual identifier|patient identifier|"
        r"群集識別|群集标识|村莊識別|村庄标识|學校識別|学校标识",
    )
    if explicit:
        explicit.sort(
            key=lambda name: (
                not re.search(
                    r"(?:^|_)(?:cluster|village|school|household|person|patient|id)", name.lower()
                ),
                _kind(columns[name]) != "identifier",
                name,
            )
        )
        return explicit[0]
    lowered = question.lower()
    preferred_tokens = [
        token
        for token in ("cluster", "school", "village", "loc", "household", "person", "patient")
        if token in lowered
    ]
    identifiers = [name for name, column in columns.items() if _kind(column) == "identifier"]
    for token in preferred_tokens:
        for name in identifiers:
            if token in name.lower():
                return name
    return None


def _score_methods(
    question: str,
    columns: dict[str, dict[str, Any]],
    items: list[GlossaryItem],
    roles: dict[str, list[str]],
    dependence: str,
) -> tuple[dict[str, float], list[str]]:
    lowered = question.lower()
    scores: dict[str, float] = {}
    evidence: list[str] = []

    def add(method: str, score: float, reason: str) -> None:
        if method not in AGENT_PROCEDURE_BY_ID:
            return
        scores[method] = max(scores.get(method, 0.0), score)
        if score >= 0.85:
            evidence.append(reason)

    outcomes = roles["outcome"]
    groups = roles["group"]
    predictors = roles["predictors"]
    outcome_kind = _kind(columns[outcomes[0]]) if outcomes else "unknown"
    group_kind = _kind(columns[groups[0]]) if groups else "unknown"
    group_levels = int(columns[groups[0]].get("unique", 0)) if groups else 0
    null_probability = _null_probability(question)
    survival = bool(roles["time"] and roles["event"])
    outcome_description = " ".join(
        item.description for item in items if outcomes and item.name == outcomes[0]
    ).lower()

    explicit_methods = (
        (r"poisson glm|poisson regression|卜瓦松", "poisson_glm"),
        (r"negative[- ]binomial|負二項|负二项", "negative_binomial_glm"),
        (r"probit", "probit_glm"),
        (r"tobit", "tobit_regression"),
        (r"two[- ]stage|2sls", "iv_2sls"),
        (r"log[- ]rank", "logrank"),
        (r"cox (?:proportional|regression)|cox ph", "cox_ph"),
        (r"spearman", "spearman_correlation"),
        (r"mann[- ]whitney", "mann_whitney"),
        (r"kruskal[- ]wallis", "kruskal_wallis"),
        (r"mixed[- ]effects?|random[- ]intercept", "mixed_effects"),
    )
    for pattern, method_id in explicit_methods:
        if re.search(pattern, lowered):
            add(method_id, 0.995, f"the request explicitly declares {method_id}")

    if (
        outcomes
        and predictors
        and re.search(r"count|non-negative integer|計數|计数", outcome_description)
    ):
        add("poisson_glm", 0.91, "the declared outcome is a count")
        add("negative_binomial_glm", 0.79, "an overdispersion-aware count model is available")

    if survival:
        if (
            groups
            and len(predictors) <= 1
            and not re.search(r"adjust|accounting|covariate|控制|調整|调整", lowered)
        ):
            add(
                "logrank", 0.99, "survival time, censoring event, and comparison group are declared"
            )
            add("cox_ph", 0.78, "Cox regression is a valid adjusted survival candidate")
        else:
            add("cox_ph", 0.97, "survival time, censoring event, and predictors are declared")
            add("logrank", 0.68, "an unadjusted survival comparison is available")
        return scores, evidence

    if null_probability is not None and (outcome_kind == "binary" or roles["target"]):
        add("binomial_test", 0.99, "a one-sample binary outcome and null probability are declared")

    if (roles["instruments"] and (roles["treatment"] or roles["exposure"])) or re.search(
        r"instrument|two[- ]stage|2sls|encouragement|offer.*actual|"
        r"assignment.*attendance|工具變數|工具变量",
        lowered,
    ):
        add("iv_2sls", 0.97, "the question distinguishes an exposure from an assignment instrument")

    if re.search(
        r"coefficients? (?:are )?equal|joint(?:ly)?|all .* coefficients|"
        r"same association|equal effects?|total effect|effects? .* differ|"
        r"線性限制|系數相等|系数相等|效果相等|總效果|总效果",
        lowered,
    ):
        add("regression_f_test", 0.96, "the null hypothesis is a joint coefficient restriction")

    if re.search(r"before.*after|pre[- ]?post|paired|matched|配對|配对|前後|前后", lowered):
        add("paired_t", 0.91, "the question declares paired observations")
        add("wilcoxon_signed_rank", 0.82, "a rank-based paired alternative is available")

    if ("clustered" in dependence or "repeated" in dependence) and roles["cluster"]:
        add("mixed_effects", 0.90, "the design declares clustered or repeated observations")
        add("gee", 0.86, "a marginal clustered analysis is available")

    if outcomes and groups:
        if outcome_kind == "binary" and group_kind in {"binary", "categorical"}:
            rows = int(columns[outcomes[0]].get("rows", 0))
            if rows <= 120 or re.search(r"sparse|rare|exact|稀少|精確|精确", lowered):
                add("fisher_exact", 0.91, "the declared two-by-two comparison may be sparse")
                add(
                    "chi_square", 0.78, "Pearson chi-square is available if expected counts suffice"
                )
            else:
                add(
                    "chi_square",
                    0.88,
                    "two categorical variables with adequate sample size are declared",
                )
                add("fisher_exact", 0.79, "an exact two-by-two alternative is available")
            add("two_proportion", 0.76, "a two-proportion contrast is available")
        elif outcome_kind == "continuous" and group_kind in {"binary", "categorical"}:
            if group_levels > 2:
                add(
                    "kruskal_wallis",
                    0.96,
                    "a continuous outcome and more than two groups are declared",
                )
                add("hc3_ols", 0.73, "a regression-based omnibus comparison is available")
            elif re.search(r"distribution|rank|median|分布|中位", lowered):
                add("mann_whitney", 0.96, "the estimand is a two-group distributional contrast")
                add("welch_t", 0.70, "a mean-based alternative is available")
            elif re.search(r"mean|average|平均", lowered):
                add("welch_t", 0.94, "the estimand is a two-group mean difference")
                add("independent_t", 0.78, "a pooled-variance mean test is available")
                add("mann_whitney", 0.61, "a rank-based robustness check is available")
            else:
                add("welch_t", 0.84, "a continuous outcome and two independent groups are declared")
                add("mann_whitney", 0.82, "a rank-based two-group analysis is available")

    continuous_predictors = [name for name in predictors if _kind(columns[name]) == "continuous"]
    if outcomes and outcome_kind == "continuous" and continuous_predictors and not groups:
        if re.search(r"correlat|association|associated|monotoni|相關|关联|關聯", lowered):
            add(
                "spearman_correlation",
                0.92,
                "two measured variables and an association estimand are declared",
            )
        add("hc3_ols", 0.76, "a heteroskedasticity-robust slope analysis is available")

    if outcomes and predictors:
        if outcome_kind == "binary":
            add("logistic_glm", 0.88, "a binary outcome and predictors are declared")
            add("probit_glm", 0.76, "a probit-link alternative is available")
        elif outcome_kind == "continuous":
            outcome_profile = columns[outcomes[0]]
            lower = outcome_profile.get("minimum")
            zero_fraction = outcome_profile.get("zero_fraction")
            if (
                isinstance(lower, (int, float))
                and float(lower) >= 0.0
                and isinstance(zero_fraction, (int, float))
                and float(zero_fraction) >= 0.08
            ):
                add(
                    "tobit_regression",
                    0.93,
                    "the non-negative outcome has an observed point mass at zero",
                )
            if re.search(
                r"bounded|censor|share|proportion from 0 to 1|limited dependent|截尾|設限|受限比例",
                lowered,
            ):
                add("tobit_regression", 0.98, "the outcome is declared as bounded or censored")
            add("hc3_ols", 0.82, "a continuous outcome and predictors are declared")
            add("ols", 0.71, "a classical linear-model candidate is available")

    if not scores and outcomes:
        add("conjugate_bayes", 0.45, "only an outcome column was identified")
        add(
            "posterior_predictive",
            0.42,
            "model checking remains possible after assumptions are supplied",
        )
    return scores, evidence


def _variables_for_method(
    method_id: str,
    roles: dict[str, list[str]],
    columns: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    outcome = _first(roles["outcome"])
    group = _first(roles["group"])
    predictors = _dedupe(roles["predictors"])
    if method_id in {
        "independent_t",
        "welch_t",
        "mann_whitney",
        "chi_square",
        "fisher_exact",
        "two_proportion",
        "kruskal_wallis",
    }:
        return {"outcome": outcome, "group": group}
    if method_id in {"paired_t", "wilcoxon_signed_rank"}:
        return {"before": _first(roles["before"]), "after": _first(roles["after"])}
    if method_id == "binomial_test":
        return {"outcome": outcome or _first(roles["target"])}
    if method_id == "spearman_correlation":
        x = _first(roles["target"] + roles["exposure"] + predictors)
        return {"x": x, "y": outcome}
    if method_id == "logrank":
        return {
            "time": _first(roles["time"]),
            "event": _first(roles["event"]),
            "group": group or _first(predictors),
        }
    if method_id == "cox_ph":
        return {
            "time": _first(roles["time"]),
            "event": _first(roles["event"]),
            "predictors": predictors or ([group] if group else []),
        }
    if method_id == "iv_2sls":
        return {
            "outcome": outcome,
            "exposure": _first(roles["treatment"] + roles["exposure"]),
            "instruments": roles["instruments"],
            "covariates": roles["covariates"],
        }
    if method_id in {"difference_in_means", "randomization_inference"}:
        return {"outcome": outcome, "treatment": _first(roles["treatment"] + roles["group"])}
    if method_id == "ipw":
        return {
            "outcome": outcome,
            "treatment": _first(roles["treatment"] + roles["group"]),
            "predictors": roles["covariates"] or predictors,
        }
    if method_id in {"gee", "mixed_effects"}:
        return {
            "outcome": outcome,
            "predictors": predictors or ([group] if group else []),
            "cluster": _first(roles["cluster"]),
        }
    if method_id == "blocked_time_series_cv":
        return {
            "outcome": outcome,
            "predictors": predictors,
            "time": _first(roles["time"]),
        }
    if method_id in {"conjugate_bayes", "posterior_predictive"}:
        return {"outcome": outcome}
    return {"outcome": outcome, "predictors": predictors or ([group] if group else [])}


def _missing_roles(method_id: str, variables: dict[str, Any]) -> list[str]:
    return [role for role in REQUIRED_VARIABLES[method_id] if not variables.get(role)]


def compile_analysis_scaffold(
    question: str,
    summaries: list[dict[str, Any]],
    *,
    max_methods: int = 5,
) -> CompiledScaffold:
    columns = _columns(summaries)
    available = set(columns)
    items = _glossary(question, available)
    row_filters, binary_recodes = _declared_preprocessing(question, columns, items)
    fields = _field_context(question)
    sampling_unit, study_design, dependence = _study_context(question, fields)
    mentioned = _mentioned_columns(question, list(columns))
    question_core = re.split(
        r"(?i)variable glossary|data schema|變數詞彙|变量词汇|欄位說明|字段说明",
        question,
        maxsplit=1,
    )[0]
    outcomes = _dedupe(
        _tagged(items, "OUTCOME")
        + _described(items, r"survival time|follow-up time|response|outcome|結果變數|结果变量")
    )
    targets = _dedupe(_tagged(items, "TARGET"))
    exposures = _dedupe(_tagged(items, "EXPOSURE"))
    treatments = _dedupe(_tagged(items, "TREATMENT"))
    survival_context = bool(
        re.search(
            r"survival|time[- ]to[- ]event|hazard|mortality|recurrence[- ]free|"
            r"progression[- ]free|生存|存活|事件時間|事件时间",
            question_core,
            re.IGNORECASE,
        )
    )
    time_columns = (
        _dedupe(
            _described(items, r"survival time|follow-up time|time to|duration")
            + [
                name
                for name in available
                if re.search(r"(?:^|_)(?:time|months|days)$", name.lower())
            ]
        )
        if survival_context
        else []
    )
    event_columns = (
        _dedupe(
            _described(
                items,
                r"event indicator|censor|deceased|progression event|recurrence status",
            )
            + [name for name in available if re.search(r"(?:^|_)(?:event|status)$", name.lower())]
        )
        if survival_context
        else []
    )
    event_columns.sort(
        key=lambda name: (
            "event" not in name.lower(),
            _kind(columns[name]) != "binary",
            name,
        )
    )
    if time_columns and event_columns:
        outcomes = [
            name for name in outcomes if name not in time_columns and name not in event_columns
        ]
    else:
        time_columns = []
        event_columns = []
    if not outcomes and targets and not (time_columns and event_columns):
        outcomes = [targets[0]]
    fixed_columns = {
        str(item["column"])
        for item in row_filters
        if item.get("operation") == "include" and len(item.get("values", [])) == 1
    }
    primary_predictors = _dedupe(
        [
            name
            for name in targets + exposures + treatments
            if name not in outcomes
            and name not in time_columns
            and name not in event_columns
            and name not in fixed_columns
        ]
    )
    explicitly_mentioned = [
        name
        for name in mentioned
        if name not in outcomes and name not in time_columns and name not in event_columns
    ]
    covariates = _dedupe(
        [
            name
            for name in explicitly_mentioned
            if name not in primary_predictors and _kind(columns[name]) != "identifier"
        ]
    )
    instruments = _dedupe(
        _described(
            items,
            r"instrument|(?:^|_)inst(?:_|$)|random(?:ly)? assigned|offer|assignment|encouragement",
        )
    )
    actual_exposure = _dedupe(
        _described(items, r"actual|attendance|take-up|received") + treatments + exposures
    )
    if instruments and actual_exposure:
        instruments = [name for name in instruments if name not in actual_exposure[:1]]
    cluster = _cluster_column(items, columns, question)
    before = _dedupe(_described(items, r"before|baseline|pre[- ]"))
    after = _dedupe(_described(items, r"after|follow-up|post[- ]"))

    if not outcomes:
        non_identifier = [name for name in mentioned if _kind(columns[name]) != "identifier"]
        outcomes = non_identifier[:1]
    if not primary_predictors:
        primary_predictors = [
            name
            for name in mentioned
            if name not in outcomes
            and name not in fixed_columns
            and _kind(columns[name]) != "identifier"
        ]
    group_candidates = [
        name for name in primary_predictors if _kind(columns[name]) in {"binary", "categorical"}
    ]
    roles: dict[str, list[str]] = {
        "outcome": outcomes,
        "target": targets,
        "exposure": exposures,
        "treatment": treatments or group_candidates,
        "group": group_candidates,
        "predictors": _dedupe(primary_predictors + covariates),
        "covariates": covariates,
        "instruments": instruments,
        "cluster": [cluster] if cluster else [],
        "time": time_columns,
        "event": event_columns,
        "before": before,
        "after": after,
    }
    scores, evidence = _score_methods(question, columns, items, roles, dependence)
    ranked = sorted(scores, key=lambda method: (-scores[method], method))[:max_methods]
    method_plans: dict[str, dict[str, Any]] = {}
    null_probability = _null_probability(question)
    total_missing = sum(int(column.get("missing", 0)) for column in columns.values())
    missingness = (
        "no missing values observed in the attached columns"
        if total_missing == 0
        else f"{total_missing} observed missing cells; mechanism is not assumed"
    )
    estimand_match = re.search(
        r"(?is)(?:research question|研究問題|研究问题)\s*:\s*(.+?)"
        r"(?:\n\s*\n|\n\s*hypothesis|\n\s*假設|\n\s*假设)",
        question,
    )
    estimand = estimand_match.group(1).strip() if estimand_match else question.strip()[:400]
    for method_id in ranked:
        variables = _variables_for_method(method_id, roles, columns)
        referenced = {
            str(item)
            for value in variables.values()
            for item in (value if isinstance(value, list) else [value])
            if item
        }
        referenced_file_indices: set[int] = set()
        ambiguous_file = False
        for name in referenced:
            file_indices = columns.get(name, {}).get("file_indices", [])
            if len(file_indices) != 1:
                ambiguous_file = True
            else:
                referenced_file_indices.add(int(file_indices[0]))
        data_file_index = (
            next(iter(referenced_file_indices))
            if not ambiguous_file and len(referenced_file_indices) == 1
            else -1
        )
        procedure = AGENT_PROCEDURE_BY_ID[method_id]
        outcome_name = variables.get("outcome") or variables.get("y") or variables.get("time")
        outcome_type = (
            _kind(columns[str(outcome_name)]) if outcome_name in columns else "declared by design"
        )
        options: dict[str, Any] = {
            "complete_case": total_missing > 0,
            "target_terms": _dedupe(
                roles["treatment"] + roles["exposure"] + roles["target"] + roles["group"]
            )[:4],
            "row_filters": row_filters,
            "binary_recodes": binary_recodes,
        }
        if null_probability is not None:
            options["null_probability"] = null_probability
            options["alternative"] = _alternative(question)
        if method_id == "tobit_regression":
            options.update({"left_censor": 0.0, "right_censor": 1.0})
        method_plans[method_id] = {
            "status": "ready",
            "estimand": estimand,
            "sampling_unit": sampling_unit,
            "study_design": study_design,
            "outcome_type": outcome_type,
            "dependence": dependence,
            "missingness": missingness,
            "method_id": method_id,
            "uncertainty": procedure.uncertainty,
            "diagnostics": list(procedure.assumptions),
            "tool": procedure.tool,
            "variables": variables,
            "analysis_options": options,
            "questions": (
                []
                if data_file_index >= 0
                else ["Which single attached file contains every column required by this analysis?"]
            ),
            "data_file_index": data_file_index,
            "compiler_confidence": round(scores[method_id], 4),
        }

    questions: list[str] = []
    if not sampling_unit:
        questions.append("What is the independent sampling unit?")
    if not study_design:
        questions.append("What is the study design and assignment mechanism?")
    if not ranked:
        questions.append("Which outcome and estimand should the analysis target?")
    elif _missing_roles(ranked[0], method_plans[ranked[0]]["variables"]):
        questions.extend(
            f"Which column is the {role}?"
            for role in _missing_roles(ranked[0], method_plans[ranked[0]]["variables"])
        )
    elif int(method_plans[ranked[0]]["data_file_index"]) < 0:
        questions.extend(method_plans[ranked[0]]["questions"])
    top_score = scores[ranked[0]] if ranked else 0.0
    context_bonus = 0.04 * int(bool(sampling_unit)) + 0.04 * int(bool(study_design))
    glossary_bonus = 0.03 if items else 0.0
    confidence = min(1.0, top_score + context_bonus + glossary_bonus)
    auto_ready = bool(ranked and not questions and confidence >= 0.86)
    role_candidates = {key: tuple(values) for key, values in roles.items() if values}
    fingerprint_payload = {
        "question": question,
        "summaries": summaries,
        "ranked": ranked,
        "method_plans": method_plans,
        "compiler_version": 2,
    }
    return CompiledScaffold(
        fingerprint=canonical_hash(fingerprint_payload),
        candidate_method_ids=tuple(ranked),
        method_plans=method_plans,
        role_candidates=role_candidates,
        evidence=tuple(_dedupe(evidence)),
        questions=tuple(_dedupe(questions)[:4]),
        confidence=round(confidence, 4),
        auto_ready=auto_ready,
    )


def plan_from_scaffold(
    scaffold: CompiledScaffold,
    method_id: str | None,
    *,
    allow_fallback: bool = True,
) -> dict[str, Any] | None:
    selected = method_id if method_id in scaffold.method_plans else None
    if (
        selected is None
        and allow_fallback
        and scaffold.auto_ready
        and scaffold.candidate_method_ids
    ):
        selected = scaffold.candidate_method_ids[0]
    if selected is None:
        return None
    plan = dict(scaffold.method_plans[selected])
    if _missing_roles(selected, plan["variables"]) or int(plan.get("data_file_index", -1)) < 0:
        return None
    plan["compiler"] = {
        "fingerprint": scaffold.fingerprint,
        "candidate_method_ids": list(scaffold.candidate_method_ids),
        "evidence": list(scaffold.evidence),
        "confidence": scaffold.confidence,
        "selection": selected,
    }
    return plan


def next_repair_plan(
    scaffold: CompiledScaffold,
    attempted_method_ids: list[str],
) -> dict[str, Any] | None:
    for method_id in scaffold.candidate_method_ids:
        if method_id in attempted_method_ids:
            continue
        plan = plan_from_scaffold(scaffold, method_id, allow_fallback=False)
        if plan is not None and not _missing_roles(method_id, plan["variables"]):
            plan["compiler"]["repair_after"] = list(attempted_method_ids)
            return plan
    return None


def task_reward(
    *,
    validity: float,
    novelty: float,
    frontier: float,
    learning_progress: float,
) -> float:
    values = (validity, novelty, frontier, learning_progress)
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("DGP-Evolve reward components must be finite values in [0, 1]")
    return math.prod(values)

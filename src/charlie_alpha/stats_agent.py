from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from .config import ProjectConfig
from .routed_inference import DynamicLoraRouter, resolve_adapter_path
from .stats_catalog import AGENT_PROCEDURE_BY_ID, AGENT_PROCEDURES
from .stats_sandbox import SandboxLimits, StatsToolSession
from .stats_training import _stats_snapshot

PublicRoute = Literal["base", "stats"]

_STATISTICS_RE = re.compile(
    r"(?:\bstat(?:istic|istics|istical)?\b|\bp[- ]?value\b|\bconfidence interval\b|"
    r"\bregression\b|\bcausal\b|\bbayes(?:ian)?\b|\bprobabilit(?:y|ies)\b|"
    r"\bsurvival\b|\bcalibrat(?:e|ed|ion)\b|\bhypothesis\b|\bsample size\b|"
    r"\bexperiment(?:al)?\b|\bmissing data\b|\btime series\b|\bforecast\b|"
    r"統計|统计|迴歸|回归|機率|概率|貝氏|贝叶斯|因果|假設檢定|假设检验|"
    r"信賴區間|置信区间|生存分析|抽樣|抽样|校準|校准|時間序列|时间序列|缺失資料|缺失数据)",
    flags=re.IGNORECASE,
)
_PLAN_RE = re.compile(r"<analysis_plan>\s*(\{.*?\})\s*</analysis_plan>", re.DOTALL)
_FINAL_RE = re.compile(r"<final_report>\s*(.*?)\s*</final_report>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)

_REQUIRED_PLAN_FIELDS = (
    "status",
    "estimand",
    "sampling_unit",
    "study_design",
    "outcome_type",
    "dependence",
    "missingness",
    "method_id",
    "uncertainty",
    "diagnostics",
    "tool",
)

_REQUIRED_VARIABLES: dict[str, tuple[str, ...]] = {
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

_R_METHODS = {
    "wilcoxon_signed_rank",
    "negative_binomial_glm",
    "mixed_effects",
    "cox_ph",
    "logrank",
}


@dataclass(frozen=True)
class RuntimePaths:
    python: Path
    rscript: Path | None


def resolve_stats_runtime(config: ProjectConfig) -> RuntimePaths:
    environment = config.root / ".pixi" / "envs" / "default" / "bin"
    python = environment / "python"
    rscript = environment.parent / "lib" / "R" / "bin" / "exec" / "R"
    if not python.exists():
        raise RuntimeError("The locked stats runtime is missing; run `make stats-setup`")
    return RuntimePaths(python=python, rscript=rscript if rscript.exists() else None)


def classify_stats_route(text: str, *, has_files: bool, override: str = "auto") -> PublicRoute:
    normalized = override.strip().lower()
    if normalized == "adapter":
        normalized = "stats"
    if normalized in {"base", "stats"}:
        return normalized  # type: ignore[return-value]
    if normalized != "auto":
        raise ValueError("charlie_route must be auto, base, stats, or adapter")
    return "stats" if has_files or _STATISTICS_RE.search(text) else "base"


def _compact_catalog() -> str:
    return "\n".join(f"- {item.method_id}: {item.name}" for item in AGENT_PROCEDURES)


def _extract_plan(text: str) -> dict[str, Any] | None:
    match = _PLAN_RE.search(text)
    candidates = [match.group(1)] if match else []
    stripped = _THINK_RE.sub("", text).strip()
    if stripped.startswith("{"):
        candidates.append(stripped)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _safe_clarification(reason: str, *, questions: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": "needs_clarification",
        "estimand": None,
        "sampling_unit": None,
        "study_design": None,
        "outcome_type": None,
        "dependence": None,
        "missingness": None,
        "method_id": "needs_clarification",
        "uncertainty": None,
        "diagnostics": [reason],
        "tool": "none",
        "questions": questions
        or [
            "What is the target estimand?",
            "What is the independent sampling unit and dependence structure?",
        ],
        "variables": {},
    }


def _validate_plan(plan: dict[str, Any], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    missing_fields = [field for field in _REQUIRED_PLAN_FIELDS if field not in plan]
    if missing_fields:
        return _safe_clarification(
            "The generated plan omitted required fields.",
            questions=[f"Please specify {field}." for field in missing_fields[:3]],
        )
    status = str(plan.get("status"))
    method_id = str(plan.get("method_id"))
    if status == "needs_clarification" or method_id == "needs_clarification":
        plan["status"] = "needs_clarification"
        plan["method_id"] = "needs_clarification"
        plan["tool"] = "none"
        plan.setdefault("questions", ["Please state the target estimand and sampling unit."])
        plan.setdefault("variables", {})
        return plan
    if method_id not in AGENT_PROCEDURE_BY_ID:
        return _safe_clarification(
            "The proposed method is outside the audited procedure catalog.",
            questions=["Which estimand and study design should the procedure target?"],
        )
    variables = plan.get("variables")
    if not isinstance(variables, dict):
        return _safe_clarification(
            "Column roles were not mapped.",
            questions=["Which columns are the outcome, predictors, groups, and sampling units?"],
        )
    required = _REQUIRED_VARIABLES[method_id]
    absent_roles = [name for name in required if not variables.get(name)]
    available = {
        str(column["name"])
        for summary in summaries
        for column in summary.get("columns", [])
        if isinstance(column, dict) and column.get("name") is not None
    }
    referenced: list[str] = []
    for value in variables.values():
        referenced.extend(str(item) for item in value) if isinstance(
            value, list
        ) else referenced.append(str(value))
    unknown_columns = [name for name in referenced if name and name not in available]
    if absent_roles or unknown_columns:
        questions = [f"Which column is the {role}?" for role in absent_roles]
        if unknown_columns:
            questions.append(f"These proposed columns do not exist: {', '.join(unknown_columns)}")
        return _safe_clarification("Variable roles are incomplete or invalid.", questions=questions)
    plan["status"] = "ready"
    plan["tool"] = "r" if method_id in _R_METHODS else "python"
    plan["variables"] = variables
    return plan


def _parse_tool_result(stdout: str, stderr: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {"status": "error", "error": stderr[-1000:] or "tool returned no JSON"}


def _sandbox_metadata(result: Any) -> dict[str, Any]:
    payload = result.to_dict()
    payload.pop("stdout", None)
    payload.pop("stderr", None)
    return payload


def _clean_final(text: str) -> str:
    text = _THINK_RE.sub("", text)
    match = _FINAL_RE.search(text)
    answer = match.group(1) if match else text
    answer = re.sub(
        r"<(analysis_plan|tool_call|method)>.*?</\1>",
        "",
        answer,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return answer.strip()


class StatsAgent:
    def __init__(
        self,
        config: ProjectConfig,
        *,
        adapter_path: str | Path | None = None,
    ) -> None:
        self.config = config
        resolved_adapter = resolve_adapter_path(config, adapter_path)
        self.model, self.tokenizer = load(
            _stats_snapshot(config),
            adapter_path=str(resolved_adapter),
            tokenizer_config={"trust_remote_code": True},
        )
        self.router = DynamicLoraRouter(self.model)
        self.runtime = resolve_stats_runtime(config)
        tool_settings = config.section("stats_tools")
        self.limits = SandboxLimits(
            timeout_seconds=int(tool_settings["timeout_seconds"]),
            memory_bytes=int(tool_settings["memory_bytes"]),
            max_write_bytes=int(tool_settings["max_write_bytes"]),
            max_output_bytes=int(tool_settings["max_output_bytes"]),
            cpu_seconds=int(tool_settings["timeout_seconds"]),
        )

    def _generate(self, messages: list[dict[str, Any]], *, max_tokens: int) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0),
            verbose=False,
        )

    def _inspect_files(
        self,
        session: StatsToolSession,
        copied: list[Path],
    ) -> tuple[list[dict[str, Any]], list[Path], list[dict[str, Any]]]:
        script = (Path(__file__).parent / "runtime" / "stats_tool.py").read_text(encoding="utf-8")
        summaries: list[dict[str, Any]] = []
        normalized: list[Path] = []
        calls: list[dict[str, Any]] = []
        for index, data_path in enumerate(copied):
            normalized_path = session.directory / f"normalized-{index}.csv"
            result = session.run_python(
                script,
                {
                    "method_id": "inspect",
                    "data_path": str(data_path),
                    "output_csv": str(normalized_path),
                },
            )
            payload = _parse_tool_result(result.stdout, result.stderr)
            calls.append(
                {"kind": "inspect", "sandbox": _sandbox_metadata(result), "result": payload}
            )
            if payload.get("status") != "ok":
                raise RuntimeError(
                    f"Data inspection failed: {payload.get('error', 'unknown error')}"
                )
            summaries.append(dict(payload["result"]))
            normalized.append(normalized_path)
        return summaries, normalized, calls

    def _plan(
        self,
        question: str,
        summaries: list[dict[str, Any]],
        language: str,
        conversation: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        system = (
            "You are Charlie alpha's statistical planner. Return an <analysis_plan> containing one "
            "JSON object and no hidden reasoning. It must include status, estimand, sampling_unit, "
            "study_design, outcome_type, dependence, missingness, method_id, uncertainty, "
            "diagnostics, tool, variables, analysis_options, questions, and data_file_index. "
            "Put declared null probabilities, censoring thresholds, and restrictions in "
            "analysis_options. If design facts or "
            "column roles are missing, set status=needs_clarification and "
            "method_id=needs_clarification. Never infer clustering, pairing, missingness "
            "mechanism, assignment mechanism, or estimand from column names alone."
        )
        prior = [
            {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
            for message in (conversation or [])[-8:]
            if message.get("role") in {"user", "assistant"}
        ]
        prompt = (
            f"Requested language: {language}\nQuestion: {question}\n"
            f"Prior conversation: {json.dumps(prior, ensure_ascii=False)}\n"
            f"Data summaries: {json.dumps(summaries, ensure_ascii=False)}\n"
            f"Audited methods:\n{_compact_catalog()}"
        )
        generated = self._generate(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=900,
        )
        plan = _extract_plan(generated)
        if plan is None:
            return _safe_clarification("The model did not produce a valid structured plan.")
        return _validate_plan(plan, summaries)

    def _run_method(
        self,
        session: StatsToolSession,
        plan: dict[str, Any],
        copied: list[Path],
        normalized: list[Path],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        index = int(plan.get("data_file_index", 0))
        if index < 0 or index >= len(copied):
            return (
                {"status": "error", "error": "data_file_index is outside the attached file list"},
                {"isolated": True, "network_allowed": False},
            )
        method_id = str(plan["method_id"])
        request = {
            "method_id": method_id,
            "data_path": str(normalized[index] if method_id in _R_METHODS else copied[index]),
            "variables": plan["variables"],
            "seed": int(self.config.section("project")["seed"]),
            "alpha": 0.05,
            "analysis_options": plan.get("analysis_options", {}),
        }
        if method_id in _R_METHODS:
            script = (Path(__file__).parent / "runtime" / "stats_tool.R").read_text(
                encoding="utf-8"
            )
            result = session.run_r(script, request)
        else:
            script = (Path(__file__).parent / "runtime" / "stats_tool.py").read_text(
                encoding="utf-8"
            )
            result = session.run_python(script, request)
        return _parse_tool_result(result.stdout, result.stderr), _sandbox_metadata(result)

    def _report(
        self,
        question: str,
        plan: dict[str, Any],
        tool_result: dict[str, Any],
        language: str,
    ) -> str:
        system = (
            "Write a concise statistical report in the requested language. Use only the "
            "structured plan and tool result. State the estimand, method, uncertainty, "
            "diagnostics, assumptions, and limits. Do not expose hidden reasoning. "
            "Do not invent quantities. Return only <final_report> text."
        )
        user = json.dumps(
            {
                "language": language,
                "question": question,
                "plan": plan,
                "tool_result": tool_result,
            },
            ensure_ascii=False,
        )
        return _clean_final(
            self._generate(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=900,
            )
        )

    def answer_without_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        route: PublicRoute,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        self.router.set_route("adapter" if route == "stats" else "base")
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return _clean_final(
            generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=max_tokens,
                sampler=make_sampler(temp=temperature, top_p=top_p),
                verbose=False,
            )
        )

    def analyze(
        self,
        *,
        data_paths: list[Path],
        question: str,
        language: str = "auto",
        route: PublicRoute = "stats",
        conversation: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not data_paths:
            raise ValueError("stats analyze requires at least one data file")
        self.router.set_route("adapter" if route == "stats" else "base")
        settings = self.config.section("stats_tools")
        with StatsToolSession(
            python_executable=self.runtime.python,
            r_executable=self.runtime.rscript,
            limits=self.limits,
            max_calls=int(settings["max_calls"]),
        ) as session:
            copied = session.add_files(
                data_paths,
                allowed_extensions={str(value) for value in settings["allowed_extensions"]},
                max_files=int(settings["max_files"]),
                max_file_bytes=int(settings["max_file_bytes"]),
                max_total_bytes=int(settings["max_total_bytes"]),
            )
            summaries, normalized, calls = self._inspect_files(session, copied)
            plan = self._plan(question, summaries, language, conversation)
            if plan["status"] == "needs_clarification":
                questions = [str(value) for value in plan.get("questions", [])]
                answer = "\n".join(f"- {question}" for question in questions)
                return {
                    "answer": answer,
                    "analysis_plan": plan,
                    "tool_calls": session.calls,
                    "tools": calls,
                    "isolation": {
                        "sandboxed": True,
                        "network_allowed": False,
                        "data_local_only": True,
                    },
                    "route": route,
                }
            tool_payload, sandbox = self._run_method(session, plan, copied, normalized)
            calls.append({"kind": "analysis", "sandbox": sandbox, "result": tool_payload})
            if tool_payload.get("status") != "ok":
                plan = _safe_clarification(
                    f"The audited tool could not execute the plan: {tool_payload.get('error')}",
                    questions=["Please verify the column roles and required design assumptions."],
                )
                answer = str(plan["diagnostics"][0])
            else:
                answer = self._report(question, plan, tool_payload, language)
            return {
                "answer": answer,
                "analysis_plan": plan,
                "tool_calls": session.calls,
                "tools": calls,
                "isolation": {
                    "sandboxed": True,
                    "network_allowed": False,
                    "data_local_only": True,
                    "limits": {
                        "timeout_seconds": self.limits.timeout_seconds,
                        "memory_bytes": self.limits.memory_bytes,
                        "max_write_bytes": self.limits.max_write_bytes,
                        "max_output_bytes": self.limits.max_output_bytes,
                    },
                },
                "route": route,
            }

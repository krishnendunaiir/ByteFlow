"""Data Analysis agent: executes the validated plan using our own statistics tools (no LLM, no eval)."""
from __future__ import annotations

from typing import Any

from agents.state import AnalysisPlan, AnalystState, PlanStep
from tools.data_cleaning import clean_dataset
from tools.machine_learning import feature_importance_mutual_info
from tools.statistics import (
    categorical_association, compare_groups, correlation_analysis, descriptive_statistics, normality_check,
)
from utils.config import Settings
from utils.logging_config import get_logger
from utils.security import UserFacingError, safe_error_message

logger = get_logger(__name__)


def _record(step: PlanStep, status: str, **extra: Any) -> dict[str, Any]:
    return {"action": step.action, "status": status, "columns": step.columns, "target": step.target,
            "group_column": step.group_column, **extra}


def run_step(df: Any, step: PlanStep, seed: int) -> list[dict[str, Any]]:
    """Run one plan step. Always returns records; a failing step is recorded as skipped, never fatal."""
    try:
        if step.action == "descriptive_stats":
            results = [descriptive_statistics(df, step.columns or None)]
        elif step.action == "correlation":
            results = [correlation_analysis(df, step.columns or None)]
        elif step.action == "group_comparison":
            results = [compare_groups(df, step.target, step.group_column)]
        elif step.action == "categorical_association":
            results = [categorical_association(df, step.columns[0], step.columns[1])]
        elif step.action == "normality_check":
            results = [normality_check(df, c, seed) for c in step.columns]
        elif step.action == "feature_importance":
            results = [feature_importance_mutual_info(df, step.target, seed=seed)]
        else:  # unreachable: the schema only allows known actions
            raise UserFacingError("Unknown analysis action.")
    except UserFacingError as exc:
        return [_record(step, "skipped", reason=safe_error_message(exc))]
    except Exception as exc:
        logger.warning("Step %s failed: %s", step.action, type(exc).__name__)
        return [_record(step, "skipped", reason=safe_error_message(exc))]
    records = []
    for res in results:
        if res.get("tested") is False:
            records.append(_record(step, "skipped", reason=res.get("reason", "not enough data")))
        else:
            records.append(_record(step, "ok", result=res))
    return records


def analyst_node(state: AnalystState, llm: Any, settings: Settings) -> dict[str, Any]:
    df, _ = clean_dataset(state["df"])  # working copy: duplicates removed, numeric text converted
    plan = AnalysisPlan.model_validate(state["plan"])
    steps: list[dict[str, Any]] = []
    for step in plan.steps:
        steps.extend(run_step(df, step, settings.random_seed))
    n_ok = sum(1 for s in steps if s["status"] == "ok")
    return {"analysis_results": {"steps": steps, "n_rows_analysed": int(len(df))},
            "progress": [f"Statistical analysis complete ({n_ok} of {len(steps)} steps produced results)"]}
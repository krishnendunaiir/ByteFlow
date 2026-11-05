"""Machine Learning agent: runs the leak-safe pipelines from tools/machine_learning.py."""
from __future__ import annotations

from typing import Any

from agents.state import AnalysisPlan, AnalystState
from tools.data_cleaning import clean_dataset
from tools.machine_learning import run_clustering, run_supervised_ml
from utils.config import Settings
from utils.logging_config import get_logger
from utils.security import UserFacingError, safe_error_message

logger = get_logger(__name__)


def ml_node(state: AnalystState, llm: Any, settings: Settings) -> dict[str, Any]:
    plan = AnalysisPlan.model_validate(state["plan"])
    if not plan.needs_ml or plan.ml_task == "none":
        return {"progress": ["Machine learning not needed"]}
    df, _ = clean_dataset(state["df"])
    try:
        if plan.ml_task == "clustering":
            r = run_clustering(df, seed=settings.random_seed)
            ml = {"task": "clustering", **r,
                  "warnings": [n for n in r["notes"] if n.startswith("Cluster structure looks weak")]}
        else:
            ml = run_supervised_ml(df, plan.target_column, task=plan.ml_task, seed=settings.random_seed,
                                   max_rows=settings.ml_sample_rows)
    except UserFacingError as exc:
        return {"errors": [f"ML: {safe_error_message(exc)}"], "progress": ["Machine learning could not run"]}
    except Exception as exc:
        logger.warning("ML failed: %s", type(exc).__name__)
        return {"errors": [f"ML: {safe_error_message(exc)}"], "progress": ["Machine learning could not run"]}
    return {"ml_results": ml, "progress": [f"Machine learning complete ({ml['task']})"]}
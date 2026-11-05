"""Shared LangGraph state and the structured (Pydantic) schemas agents exchange."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

from utils.config import Settings
from typing import ClassVar

AnalysisAction = Literal[
    "descriptive_stats", "correlation", "group_comparison",
    "categorical_association", "normality_check", "feature_importance",
]
MLTask = Literal["classification", "regression", "clustering", "none"]
RetryNode = Literal["plan", "analyse", "ml", "visualize", "report"]


# ----- structured LLM outputs (validated before use) -----
class UnderstandingResult(BaseModel):
    dataset_summary: str = Field(max_length=800)
    inferred_objective: str = Field(max_length=400)
    candidate_targets: list[str] = Field(default_factory=list, max_length=10)


class PlanStep(BaseModel):
    action: AnalysisAction
    columns: list[str] = Field(default_factory=list, max_length=50)
    target: str | None = None
    group_column: str | None = None
    rationale: str = Field(max_length=300)


class AnalysisPlan(BaseModel):
    objective: str = Field(max_length=400)
    steps: list[PlanStep] = Field(max_length=12)
    needs_ml: bool = False
    ml_task: MLTask = "none"
    target_column: str | None = None
    clarification_needed: str | None = None  # set when an essential detail is missing


class CriticIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node: str
    description: str


class CriticVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strict_structured_output: ClassVar[bool] = True

    approved: bool
    issues: list[CriticIssue]


# ----- the graph state -----
class AnalystState(TypedDict, total=False):
    """Every node reads from and writes to this dictionary.

    `errors` and `progress` use a reducer (operator.add): nodes return only
    NEW items and LangGraph appends them to the existing list.
    """
    session_id: str
    question: str
    df: pd.DataFrame                 # original data, never modified in place
    upload_info: dict[str, Any]
    profile: dict[str, Any]          # compact column profile (safe to send to LLM)
    understanding: dict[str, Any]
    quality: dict[str, Any]
    plan: dict[str, Any]
    analysis_results: dict[str, Any]
    ml_results: dict[str, Any]
    charts: list[str]               # file paths inside the session output dir
    chart_details: list[dict[str, str]]
    critic: dict[str, Any]
    report_markdown: str
    retry_count: int
    max_retries: int
    retry_target: str | None
    status: Literal["running", "needs_clarification", "failed", "done"]
    clarification: str | None
    errors: Annotated[list[str], operator.add]
    progress: Annotated[list[str], operator.add]


def initial_state(df: pd.DataFrame, question: str, upload_info: dict[str, Any],
                  settings: Settings, session_id: str) -> AnalystState:
    """Build the starting state for one analysis run."""
    return AnalystState(
        session_id=session_id, 
        question=question,
        df=df,
        upload_info=upload_info,
        charts=[],
        chart_details=[], 
        retry_count=0, 
        max_retries=settings.max_retries, 
        retry_target=None,
        status="running", 
        clarification=None, 
        errors=[], 
        progress=[],
    )
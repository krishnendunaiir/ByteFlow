"""Data Quality agent: rule-based checks. It never modifies the original dataset."""
from __future__ import annotations

from typing import Any

from agents.state import AnalystState
from tools.data_cleaning import clean_dataset, recommend_cleaning
from tools.data_inspection import assess_data_quality
from utils.config import Settings


def quality_node(state: AnalystState, llm: Any, settings: Settings) -> dict[str, Any]:
    df = state["df"]
    quality = assess_data_quality(df)
    _, cleaning_log = clean_dataset(df)  # works on a copy: used here only to report what would change
    quality["recommendations"] = recommend_cleaning(quality)
    quality["cleaning_log"] = cleaning_log
    return {"quality": quality, "progress": ["Data quality assessment complete"]}
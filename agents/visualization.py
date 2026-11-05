"""Visualization agent: asks the chart tools for a small set of relevant, safe charts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.state import AnalystState
from tools.data_cleaning import clean_dataset
from tools.visualization import generate_charts
from utils.config import Settings
from utils.security import SecurityError, ensure_within_directory, session_output_dir


def _remove_old_charts(session_dir: Path, charts: list[dict[str, str]]) -> None:
    """On a retry, delete the previous attempt's PNGs (only inside this session's folder)."""
    for c in charts:
        try:
            ensure_within_directory(session_dir, Path(c["path"]).name).unlink(missing_ok=True)
        except (SecurityError, OSError):
            pass


def visualization_node(state: AnalystState, llm: Any, settings: Settings) -> dict[str, Any]:
    session_dir = session_output_dir(settings.outputs_dir, state["session_id"])
    _remove_old_charts(session_dir, state.get("charts", []))
    df, _ = clean_dataset(state["df"])
    charts, problems = generate_charts(df, state["quality"], state["analysis_results"], state.get("ml_results"),
                                       session_dir, max_charts=8)
    return {"charts": charts, "errors": [f"Visualization: {p}" for p in problems],
            "progress": [f"Visualizations complete ({len(charts)} charts)"]}
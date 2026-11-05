"""The LangGraph workflow: nodes, conditional edges, error recovery and the retry limit."""
from __future__ import annotations

from typing import Any, Callable

import pandas as pd
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from agents.analyst import analyst_node
from agents.critic import critic_node
from agents.llm import LLMClient
from agents.ml_agent import ml_node
from agents.planner import planner_node
from agents.quality import quality_node
from agents.reporter import reporter_node
from agents.state import AnalystState, initial_state
from agents.understanding import understanding_node
from agents.visualization import visualization_node
from utils.config import Settings
from utils.logging_config import get_logger
from utils.security import UserFacingError, new_session_id, safe_error_message, session_output_dir
from utils.validation import validate_question

logger = get_logger(__name__)
RETRY_TARGETS = {"plan", "analyse", "ml", "visualize"}


def validate_node(state: AnalystState, llm: Any, settings: Settings) -> dict[str, Any]:
    df = state.get("df")
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise UserFacingError("No dataset was provided.")
    question = validate_question(state.get("question", ""), settings)
    session_output_dir(settings.outputs_dir, state["session_id"])
    return {"question": question, "progress": ["Dataset validated"]}


def _guarded(label: str, fn: Callable[..., dict[str, Any]], llm: LLMClient, settings: Settings,
             fatal: bool) -> Callable[[AnalystState], dict[str, Any]]:
    """Wrap a node so an exception becomes a recorded error instead of a crash."""
    def node(state: AnalystState) -> dict[str, Any]:
        try:
            return fn(state, llm, settings)
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all at the workflow boundary
            logger.error("Node '%s' failed: %s", label, type(exc).__name__)
            msg = f"{label}: {safe_error_message(exc)}"
            if fatal:
                return {"status": "failed", "errors": [msg], "progress": [f"{label} failed"]}
            return {"errors": [msg], "progress": [f"{label} skipped after an error"]}
    node.__name__ = label
    return node


def _running(state: AnalystState) -> bool:
    return state.get("status", "running") == "running"


def _go(nxt: str) -> Callable[[AnalystState], str]:
    return lambda state: nxt if _running(state) else END


def route_after_analysis(state: AnalystState) -> str:
    if not _running(state):
        return END
    return "ml" if state.get("plan", {}).get("needs_ml") else "visualize"


def route_after_critic(state: AnalystState) -> str:
    target = state.get("retry_target")
    return target if target in RETRY_TARGETS else "report"


def build_graph(settings: Settings, llm: LLMClient):
    """Assemble and compile the StateGraph."""
    g = StateGraph(AnalystState)
    spec = [("validate", validate_node, True), ("understand", understanding_node, True),
            ("quality", quality_node, True), ("plan", planner_node, True), ("analyse", analyst_node, True),
            ("ml", ml_node, False), ("visualize", visualization_node, False),
            ("critic", critic_node, False), ("report", reporter_node, True)]
    for name, fn, fatal in spec:
        g.add_node(name, _guarded(name, fn, llm, settings, fatal))
    g.add_edge(START, "validate")
    for src, dst in [("validate", "understand"), ("understand", "quality"), ("quality", "plan"), ("plan", "analyse")]:
        g.add_conditional_edges(src, _go(dst), {dst: dst, END: END})
    g.add_conditional_edges("analyse", route_after_analysis, {"ml": "ml", "visualize": "visualize", END: END})
    g.add_edge("ml", "visualize")
    g.add_edge("visualize", "critic")
    g.add_conditional_edges("critic", route_after_critic,
                            {"plan": "plan", "analyse": "analyse", "ml": "ml", "visualize": "visualize", "report": "report"})
    g.add_edge("report", END)
    return g.compile()


def run_analysis(df: pd.DataFrame, question: str, upload_info: dict[str, Any], settings: Settings, llm: LLMClient,
                 session_id: str | None = None, on_progress: Callable[[str], None] | None = None) -> AnalystState:
    """Run the whole workflow and return the final state. `on_progress` receives each progress message."""
    state = initial_state(df, question, upload_info, settings, session_id or new_session_id())
    graph = build_graph(settings, llm)
    config = {"recursion_limit": 20 + 8 * settings.max_retries}  # hard stop against runaway loops
    last, seen = state, 0
    try:
        for chunk in graph.stream(state, config=config, stream_mode="values"):
            last = chunk
            progress = chunk.get("progress", [])
            if on_progress:
                for message in progress[seen:]:
                    on_progress(message)
            seen = len(progress)
    except GraphRecursionError as exc:
        raise UserFacingError("The analysis exceeded its step limit and was stopped.") from exc
    return last
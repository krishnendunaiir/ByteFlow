"""Streamlit interface for ByteFlow, the Autonomous AI Data Analyst.

Security notes: uploaded files are validated and kept in memory only.
Dataset text is escaped before display. No unsafe_allow_html is used.
Errors shown to users never include stack traces or secrets.
"""
from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from agents.graph import run_analysis
from agents.llm import OpenAICompatibleLLM
from tools.data_inspection import profile_dataset
from tools.report_generation import md_escape
from utils.app_support import (
    RunLimiter,
    build_charts_zip,
    purge_old_sessions,
    read_chart_bytes,
)
from utils.config import Settings, get_settings
from utils.logging_config import get_logger
from utils.security import (
    SecurityError,
    UserFacingError,
    cleanup_session_dir,
    new_session_id,
    safe_error_message,
    session_output_dir,
)
from utils.validation import validate_and_load_upload, validate_question


logger = get_logger("app")

DEFAULT_QUESTION = (
    "Analyse this dataset and identify the most important patterns."
)

EXAMPLE_QUESTIONS = [
    "Find missing values, duplicates and outliers.",
    "Identify correlations between the numeric variables.",
    "Does monthly_charges differ between contract types?",
    "What are the most important factors influencing churn? Build a suitable model.",
    "Group similar customers into segments.",
]

STATE_KEYS = ("df", "info", "result", "upload_sig")


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

@st.cache_resource
def _startup(outputs_dir: str) -> int:
    """Remove session folders left over from earlier runs."""
    from pathlib import Path

    return purge_old_sessions(Path(outputs_dir))


def load_settings() -> Settings:
    try:
        return get_settings()

    except Exception:
        st.error(
            "Your .env settings are invalid. Check LLM_BASE_URL "
            "(must start with https://), MAX_UPLOAD_MB and MAX_RETRIES, "
            "then restart the app."
        )
        st.stop()


def init_state() -> None:
    ss = st.session_state

    ss.setdefault("sid", new_session_id())
    ss.setdefault("limiter", RunLimiter())

    for key in STATE_KEYS + ("dismissed_sig",):
        ss.setdefault(key, None)


def reset_data(
    settings: Settings,
    new_session: bool = True,
) -> None:
    """Forget the dataset and results and delete this session's files."""

    ss = st.session_state

    try:
        cleanup_session_dir(settings.outputs_dir, ss["sid"])

    except SecurityError:
        pass

    for key in STATE_KEYS:
        ss[key] = None

    if new_session:
        ss["sid"] = new_session_id()


# ---------------------------------------------------------------------
# Sidebar and upload
# ---------------------------------------------------------------------

def sidebar(settings: Settings):
    with st.sidebar:

        st.header("Your workspace")

        st.caption("BYTEFLOW  /  DATA ANALYSIS")

        st.subheader("1. Upload data")

        uploaded = st.file_uploader(
            "CSV or Excel (.xlsx)",
            type=["csv", "xlsx"],
            help=(
                f"Maximum {settings.max_upload_mb} MB "
                f"and {settings.max_rows:,} rows."
            ),
        )

        st.caption(
            "Your file is kept in memory for this session only. "
            "It is never written to disk."
        )

        if st.button("Clear my data", width="stretch"):

            sig = st.session_state["upload_sig"]

            reset_data(settings)

            st.session_state["dismissed_sig"] = sig

            st.rerun()

        st.divider()

        st.subheader("AI provider")

        if settings.llm_configured:

            host = (
                settings.llm_base_url
                .split("//", 1)[-1]
                .split("/", 1)[0]
            )

            st.success(
                f"Key found. Model: {settings.llm_model} ({host})"
            )

        else:

            st.error(
                "No API key. Add LLM_API_KEY to your .env file "
                "and restart the app."
            )

        st.caption(
            "Only column names, types and computed statistics "
            "are sent to the AI, never raw rows."
        )

    return uploaded


def handle_upload(uploaded, settings: Settings) -> None:

    ss = st.session_state

    if uploaded is None:

        ss["dismissed_sig"] = None

        if ss["df"] is not None:
            reset_data(settings)

        return

    data = uploaded.getvalue()

    sig = hashlib.sha256(data).hexdigest()

    if sig == ss["dismissed_sig"]:

        st.sidebar.caption(
            "Data cleared. Remove this file or choose another "
            "one to start again."
        )

        return

    if sig == ss["upload_sig"]:
        return

    reset_data(settings)

    ss["dismissed_sig"] = None

    try:

        df, info = validate_and_load_upload(
            uploaded.name,
            data,
            settings,
        )

    except UserFacingError as exc:

        st.error(safe_error_message(exc))

        return

    ss["df"] = df
    ss["info"] = info.model_dump()
    ss["upload_sig"] = sig


# ---------------------------------------------------------------------
# Dataset preview
# ---------------------------------------------------------------------

def render_preview() -> None:

    df = st.session_state["df"]
    info = st.session_state["info"]

    st.subheader("02  /  Dataset overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", f"{info['n_rows']:,}")

    c2.metric("Columns", info["n_columns"])

    c3.metric("File type", info["file_type"].upper())

    c4.metric(
        "File size",
        f"{info['size_bytes'] / 1024:,.0f} KB",
    )

    tab_preview, tab_info = st.tabs(
        ["Preview (first 50 rows)", "Column information"]
    )

    with tab_preview:

        st.dataframe(
            df.head(50),
            width="stretch",
        )

    with tab_info:

        cols = profile_dataset(df)["columns"]

        column_data = [
            {
                "column": c["name"],
                "type": c["kind"],
                "dtype": c["dtype"],
                "missing %": c["pct_missing"],
                "unique": c["n_unique"],
            }
            for c in cols
        ]

        st.dataframe(
            pd.DataFrame(column_data),
            width="stretch",
            hide_index=True,
        )


# ---------------------------------------------------------------------
# Running the workflow
# ---------------------------------------------------------------------

def get_llm(settings: Settings):

    if (
        "llm" not in st.session_state
        or st.session_state["llm"] is None
    ):

        st.session_state["llm"] = OpenAICompatibleLLM(settings)

    return st.session_state["llm"]


def run_workflow(
    settings: Settings,
    question_text: str,
) -> None:

    ss = st.session_state

    blocked = ss["limiter"].check()

    if blocked:

        st.warning(blocked)

        return

    try:

        question = validate_question(
            question_text,
            settings,
        )

        llm = get_llm(settings)

    except UserFacingError as exc:

        st.error(safe_error_message(exc))

        return

    ss["limiter"].record()

    try:

        cleanup_session_dir(
            settings.outputs_dir,
            ss["sid"],
        )

    except SecurityError:
        pass

    ss["result"] = None

    with st.status(
        "Running the analysis agents...",
        expanded=True,
    ) as status:

        progress_bar = st.progress(
            0,
            text="Preparing analysis... 0%",
        )

        progress_value = [0]

        def update_progress(message: str) -> None:
            status.write(message)

            msg = message.lower()

            milestones = [
                ("dataset validated", 5),
                ("data understanding complete", 15),
                ("data quality assessment complete", 25),
                ("analysis plan created", 35),
                ("statistical analysis complete", 55),
                ("machine learning", 65),
                ("visualizations complete", 78),
                ("quality review", 88),
                ("report generated", 95),
            ]

            target = progress_value[0]

            for keyword, percentage in milestones:
                if keyword in msg:
                    target = max(target, percentage)

            progress_value[0] = target

            progress_bar.progress(
                target,
                text=f"{message} — {target}%",
            )

        try:
            result = run_analysis(
                ss["df"],
                question,
                ss["info"],
                settings,
                llm,
                session_id=ss["sid"],
                on_progress=update_progress,
            )

            progress_bar.progress(
                100,
                text="Analysis workflow finished — 100%",
            )

        except UserFacingError as exc:

            status.update(
                label="Analysis stopped",
                state="error",
            )

            st.error(safe_error_message(exc))

            return

        except Exception as exc:

            logger.error(
                "Analysis crashed: %s",
                type(exc).__name__,
            )

            status.update(
                label="Analysis failed",
                state="error",
            )

            st.error(
                "Something went wrong while analysing the data. "
                "Details were written to the log."
            )

            return

        labels = {
            "done": ("Analysis complete", "complete"),
            "failed": ("Analysis failed", "error"),
            "needs_clarification": (
                "More information needed",
                "complete",
            ),
        }

        label, state = labels.get(
            result.get("status"),
            ("Analysis finished", "complete"),
        )

        status.update(
            label=label,
            state=state,
            expanded=False,
        )

    ss["result"] = result


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------

def _table(rows, columns=None):

    if not rows:

        st.caption("None.")

        return

    frame = pd.DataFrame(rows)

    st.dataframe(
        frame[columns] if columns else frame,
        width="stretch",
        hide_index=True,
    )


def render_quality(result) -> None:

    q = result["quality"]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Missing cells",
        f"{q['missing']['total_missing_cells']:,}",
    )

    c2.metric(
        "Rows with missing values",
        f"{q['missing']['rows_with_any_missing']:,}",
    )

    c3.metric(
        "Duplicate rows",
        f"{q['duplicates']['n_duplicate_rows']:,}",
    )

    c4.metric(
        "Columns with outliers",
        q["outliers"]["n_columns_with_outliers"],
    )

    st.markdown("**Missing values by column**")

    _table(q["missing"]["per_column"])

    st.markdown(
        "**Outliers (IQR rule: flagged for review, not removed)**"
    )

    _table(
        [
            o
            for o in q["outliers"]["by_column"]
            if o["n_outliers"]
        ]
    )

    st.markdown("**Suspicious values and type issues**")

    _table(
        q["suspicious"],
        ["column", "issue", "detail"],
    )

    st.markdown("**Cleaning**")

    st.caption(
        "Applied to a working copy only; "
        "your original data is never modified."
    )

    _table(q["cleaning_log"])

    _table(q["recommendations"])


def render_statistics(result) -> None:

    steps = result["analysis_results"]["steps"]

    if not steps:

        st.info("No statistical steps were run.")

    for step in steps:

        if step["status"] != "ok":

            st.caption(
                f"Skipped {step['action']}: "
                f"{md_escape(step.get('reason', 'not enough data'), 300)}"
            )

            continue

        r = step["result"]
        action = step["action"]

        if action == "descriptive_stats":

            st.markdown("**Descriptive statistics**")

            if r["numeric"]:

                st.dataframe(
                    pd.DataFrame(r["numeric"]).T,
                    width="stretch",
                )

        elif action == "correlation":

            st.markdown(
                f"**Correlation ({r['method']}, "
                "p-values adjusted for multiple testing)**"
            )

            _table(
                r["strong_pairs"],
                [
                    "column_a",
                    "column_b",
                    "r",
                    "strength",
                    "p_adjusted",
                    "n",
                ],
            )

        elif action == "group_comparison":

            st.markdown(
                f"**{md_escape(r['numeric_column'])} by "
                f"{md_escape(r['group_column'])}: {r['test']}**"
            )

            st.write(
                f"p = {r['p_value']}, "
                f"effect size ({r['effect_size']['name']}) = "
                f"{r['effect_size']['value']} "
                f"({r['effect_size']['magnitude']}). "
                f"{r['reason_for_test']}"
            )

            _table(r["groups"])

        elif action == "categorical_association":

            st.markdown(
                f"**{md_escape(r['column_a'])} vs "
                f"{md_escape(r['column_b'])}: {r['test']}**"
            )

            st.write(
                f"p = {r['p_value']}, "
                f"Cramer's V = {r['effect_size']['value']} "
                f"({r['effect_size']['magnitude']})."
            )

        elif action == "normality_check":

            st.write(
                f"Normality of {md_escape(r['column'])}: "
                f"p = {r['p_value']} "
                f"({'approximately normal' if r['is_normal'] else 'not normal'})."
            )

        elif action == "feature_importance":

            st.markdown(
                f"**Feature ranking for {md_escape(r['target'])} "
                "(mutual information, exploratory)**"
            )

            _table(r["ranking"])


def render_ml(ml) -> None:

    if ml.get("task") == "clustering":

        st.write(
            f"K-means chose **{ml['best_k']} clusters** "
            f"(silhouette {ml['best_silhouette']})."
        )

        _table(
            [
                {"cluster": key, "size": value}
                for key, value in ml["cluster_sizes"].items()
            ]
        )

        st.dataframe(
            pd.DataFrame(ml["cluster_means"]).T,
            width="stretch",
        )

    else:

        st.write(
            f"**{ml['task'].capitalize()}** model for "
            f"**{md_escape(ml['target'])}**. "
            f"Trained on {ml['n_train']:,} rows, "
            f"tested on {ml['n_test']:,} held-out rows. "
            f"Best model: **{ml['best_model']}**."
        )

        _table(
            [
                {
                    "model": key,
                    f"CV {ml['primary_metric']}": value["primary_cv_mean"],
                    "std": value["primary_cv_std"],
                    "train score": value["primary_train_mean"],
                }
                for key, value in ml["cv_results"].items()
            ]
        )

        st.markdown(
            "**Held-out test results (evaluated once)**"
        )

        _table(
            [
                {"metric": key, "value": value}
                for key, value in ml["test_metrics"].items()
            ]
        )

        st.caption(
            f"Baseline {ml['primary_metric']} on the same test set: "
            f"{ml['baseline_test_primary']}"
        )

        st.markdown(
            "**Most important features (permutation importance)**"
        )

        _table(ml["permutation_importance"])

        if ml["features_excluded"]:

            st.markdown("**Columns excluded from the model**")

            _table(ml["features_excluded"])

    for warning in ml.get("warnings", []):

        st.warning(warning)


def render_charts(result) -> None:

    charts = result.get("charts", [])

    if not charts:

        st.info("No charts were generated.")

        return

    session_dir = session_output_dir(
        get_settings().outputs_dir,
        result["session_id"],
        create=False,
    )

    for chart in charts:

        try:

            st.image(
                read_chart_bytes(
                    chart["path"],
                    session_dir,
                ),
                caption=(
                    f"{chart['title']}: "
                    f"{chart['description']}"
                ),
            )

        except (SecurityError, OSError):

            st.caption(
                f"{chart['title']} is no longer available."
            )

    st.download_button(
        "Download all charts (.zip)",
        build_charts_zip(charts, session_dir),
        file_name="charts.zip",
        mime="application/zip",
    )


def render_results(result) -> None:

    status = result.get("status")

    if status == "needs_clarification":

        st.warning(
            f"I need one more detail: "
            f"{md_escape(result.get('clarification'), 300)}"
        )

        st.caption(
            "Rewrite your question with that detail "
            "and run the analysis again."
        )

        return

    if status != "done":

        st.error(
            "The analysis could not be completed."
        )

        for error in result.get("errors", [])[-5:]:

            st.write(
                f"- {md_escape(error, 400)}"
            )

        return

    st.subheader("04  /  Analysis results")

    ml = result.get("ml_results")

    names = (
        ["Report", "Data quality", "Statistics"]
        + (["Machine learning"] if ml else [])
        + ["Charts", "Workflow log"]
    )

    tabs = dict(zip(names, st.tabs(names)))

    with tabs["Report"]:

        st.download_button(
            "Download report (.md)",
            result["report_markdown"].encode("utf-8"),
            file_name="analysis_report.md",
            mime="text/markdown",
        )

        st.markdown(
            result["report_markdown"]
        )

    with tabs["Data quality"]:

        render_quality(result)

    with tabs["Statistics"]:

        render_statistics(result)

    if ml:

        with tabs["Machine learning"]:

            render_ml(ml)

    with tabs["Charts"]:

        render_charts(result)

    with tabs["Workflow log"]:

        critic = result.get("critic", {})

        st.write(
            f"Quality review: "
            f"{'approved' if critic.get('approved') else 'issues remained'}; "
            f"retries used: {result.get('retry_count', 0)} "
            f"of {result.get('max_retries', 0)}."
        )

        for progress in result.get("progress", []):

            st.write(
                f"- {md_escape(progress, 200)}"
            )

        if result.get("errors"):

            st.markdown(
                "**Notices (non-fatal problems and fallbacks)**"
            )

            for error in result["errors"]:

                st.write(
                    f"- {md_escape(error, 400)}"
                )


# ---------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------

def main() -> None:

    st.set_page_config(
        page_title="ByteFlow — AI Data Analyst",
        page_icon="🌸",
        layout="wide",
    )

    settings = load_settings()

    _startup(str(settings.outputs_dir))

    init_state()

    # Main-page branding
    st.title("ByteFlow")

    st.caption(
        "YOUR DATA, WITH DIRECTION."
    )

    st.markdown(
        "### Turn raw data into clear decisions."
    )

    st.write(
        "Upload a dataset, ask a question in plain English, "
        "and let your analysis team do the rest."
    )

    handle_upload(
        sidebar(settings),
        settings,
    )

    if st.session_state["df"] is None:

        st.info(
            "Upload a CSV or Excel file in the sidebar to begin."
        )

        st.button(
            "Run analysis",
            type="primary",
            disabled=True,
            width="stretch",
        )

        return

    render_preview()

    st.subheader(
        "03  /  What would you like to discover?"
    )

    question = st.text_area(
        "Your question",
        value=DEFAULT_QUESTION,
        max_chars=settings.max_question_chars,
        height=110,
        placeholder=(
            "e.g. Which factors are most strongly associated "
            "with customer churn?"
        ),
    )

    with st.expander("Example questions"):

        for example in EXAMPLE_QUESTIONS:

            st.write(f"- {example}")

    if st.button(
        "Run analysis",
        type="primary",
        disabled=not settings.llm_configured,
        width="stretch",
    ):

        run_workflow(
            settings,
            question,
        )

    if st.session_state["result"] is not None:

        render_results(
            st.session_state["result"]
        )


if __name__ == "__main__":
    main()
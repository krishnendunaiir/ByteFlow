"""Chart generation. Charts are saved as PNGs with random filenames inside a session folder."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import logging

import matplotlib

matplotlib.use("Agg")  # no GUI needed; must be set before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from tools.data_inspection import get_numeric_columns, require_columns  # noqa: E402
from utils.logging_config import get_logger  # noqa: E402
from utils.security import ensure_within_directory, safe_internal_filename, sanitize_text_for_llm  # noqa: E402

logger = get_logger(__name__)
logging.getLogger("matplotlib.category").setLevel(logging.WARNING)  # silence a harmless INFO message
sns.set_theme(style="whitegrid")
MAX_PLOT_GROUPS = 10


def _label(text: Any, n: int = 30) -> str:
    """Shorten a label and escape '$' so matplotlib does not treat it as math text."""
    return sanitize_text_for_llm(text, n).replace("$", r"\$")


def _save(fig: plt.Figure, out_dir: Path, title: str, description: str) -> dict[str, str]:
    path = ensure_within_directory(out_dir, safe_internal_filename(".png"))
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=110, bbox_inches="tight")
    finally:
        plt.close(fig)
    return {"path": str(path), "title": title, "description": description}


def plot_missing_values(missing: dict[str, Any], out_dir: Path) -> dict[str, str] | None:
    rows = missing["per_column"][:15]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(rows) + 1.5))
    ax.barh([_label(r["column"]) for r in rows][::-1], [r["pct_missing"] for r in rows][::-1], color="#c0504d")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Missing values (%)")
    ax.set_title("Missing values by column")
    return _save(fig, out_dir, "Missing values by column", "Percentage of missing values in each column that has any.")


def plot_distributions(df: pd.DataFrame, columns: list[str], out_dir: Path, max_cols: int = 6) -> dict[str, str] | None:
    cols = columns[:max_cols]
    require_columns(df, cols)
    if not cols:
        return None
    ncols = min(3, len(cols))
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    for ax, c in zip(axes.ravel(), cols):
        x = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        sns.histplot(x, bins=30, ax=ax, color="#4f81bd")
        ax.set_title(_label(c))
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    for ax in axes.ravel()[len(cols):]:
        ax.set_visible(False)
    return _save(fig, out_dir, "Distributions of numeric variables", "Histograms showing the shape of each numeric column.")


def plot_correlation_heatmap(matrix: dict[str, dict[str, float | None]], out_dir: Path, method: str = "") -> dict[str, str] | None:
    corr = pd.DataFrame(matrix).astype(float)
    if corr.shape[0] < 2:
        return None
    corr = corr.iloc[:20, :20]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    size = max(5, 0.55 * len(corr) + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(corr.rename(columns=_label, index=_label), mask=mask, vmin=-1, vmax=1, center=0, cmap="coolwarm",
                annot=len(corr) <= 12, fmt=".2f", square=True, cbar_kws={"label": "Correlation"}, ax=ax)
    ax.set_title(f"Correlation matrix ({method})" if method else "Correlation matrix")
    return _save(fig, out_dir, "Correlation heatmap", "Pairwise correlations between numeric variables (-1 to +1).")


def plot_group_boxplot(df: pd.DataFrame, numeric_column: str, group_column: str, out_dir: Path) -> dict[str, str] | None:
    require_columns(df, [numeric_column, group_column])
    data = pd.DataFrame({"y": pd.to_numeric(df[numeric_column], errors="coerce"), "g": df[group_column]}).dropna()
    data["g"] = data["g"].astype(str)
    top = data["g"].value_counts().head(MAX_PLOT_GROUPS).index
    data = data[data["g"].isin(top)]
    if data["g"].nunique() < 2:
        return None
    data["g"] = data["g"].map(_label)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=data, x="g", y="y", ax=ax, color="#9bbb59")
    ax.set_xlabel(_label(group_column))
    ax.set_ylabel(_label(numeric_column))
    ax.set_title(f"{_label(numeric_column)} by {_label(group_column)}")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, out_dir, f"{_label(numeric_column)} by {_label(group_column)}", "Box plots comparing the distribution of a numeric variable across groups.")


def plot_category_counts(df: pd.DataFrame, column: str, out_dir: Path, top: int = 10) -> dict[str, str] | None:
    require_columns(df, [column])
    vc = df[column].value_counts().head(top)
    if vc.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(vc) + 1.5))
    ax.barh([_label(i) for i in vc.index][::-1], vc.to_numpy()[::-1], color="#4f81bd")
    ax.set_xlabel("Count")
    ax.set_title(f"Most common values: {_label(column)}")
    return _save(fig, out_dir, f"Value counts: {_label(column)}", "Counts of the most frequent categories (bars start at zero).")


def plot_target_distribution(df: pd.DataFrame, target: str, task: str, out_dir: Path) -> dict[str, str] | None:
    require_columns(df, [target])
    if task == "classification":
        return plot_category_counts(df, target, out_dir)
    return plot_distributions(df, [target], out_dir)


def plot_model_comparison(cv_results: dict[str, Any], out_dir: Path) -> dict[str, str] | None:
    if not cv_results:
        return None
    names = list(cv_results)
    means = [cv_results[n]["primary_cv_mean"] for n in names]
    stds = [cv_results[n]["primary_cv_std"] or 0 for n in names]
    metric = cv_results[names[0]]["primary_metric"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([_label(n) for n in names], means, yerr=stds, capsize=4,
           color=["#999999" if n == "baseline_dummy" else "#4f81bd" for n in names])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"{metric} (cross-validated mean)")
    ax.set_title("Model comparison (grey = baseline)")
    ax.tick_params(axis="x", rotation=15)
    return _save(fig, out_dir, "Model comparison", "Cross-validated score of each model; error bars show standard deviation across folds.")


def plot_feature_importance(importances: list[dict[str, Any]], out_dir: Path) -> dict[str, str] | None:
    if not importances:
        return None
    rows = importances[:10][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(rows) + 1.5))
    ax.barh([_label(r["feature"]) for r in rows], [r["importance_mean"] or 0 for r in rows],
            xerr=[r["importance_std"] or 0 for r in rows], color="#f79646")
    ax.set_xlabel("Drop in score when the feature is shuffled")
    ax.set_title("Permutation feature importance (test set)")
    return _save(fig, out_dir, "Feature importance", "How much the model's score drops when each feature is randomly shuffled.")


def plot_confusion_matrix(confusion: dict[str, Any], out_dir: Path) -> dict[str, str] | None:
    labels = confusion["labels"][:12]
    mat = np.array(confusion["matrix"])[: len(labels), : len(labels)]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=[_label(x, 15) for x in labels], yticklabels=[_label(x, 15) for x in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    return _save(fig, out_dir, "Confusion matrix", "Counts of correct and incorrect predictions on the held-out test set.")


def generate_charts(df: pd.DataFrame, quality: dict[str, Any], analysis_results: dict[str, Any],
                    ml_results: dict[str, Any] | None, out_dir: Path, max_charts: int = 8) -> tuple[list[dict[str, str]], list[str]]:
    """Pick the most relevant charts (capped) and save them. Returns (charts, problems)."""
    out_dir = Path(out_dir)
    jobs: list[tuple[str, Any]] = []
    steps = [s for s in analysis_results.get("steps", []) if s.get("status") == "ok"]

    supervised = bool(ml_results and ml_results.get("task") in ("classification", "regression"))
    if quality.get("missing", {}).get("per_column"):

        jobs.append(("missing", lambda: plot_missing_values(quality["missing"], out_dir)))
    if supervised:
        jobs.append(("target", lambda: plot_target_distribution(df, ml_results["target"], ml_results["task"], out_dir)))
    numeric = get_numeric_columns(df)
    if numeric:
        jobs.append(("dist", lambda: plot_distributions(df, numeric, out_dir)))
    for s in steps:
        res = s.get("result", {})
        if s["action"] == "correlation" and res.get("tested"):
            jobs.append(("corr", lambda r=res: plot_correlation_heatmap(r["matrix"], out_dir, r["method"])))
        elif s["action"] == "group_comparison" and res.get("tested"):
            jobs.append(("box", lambda r=res: plot_group_boxplot(df, r["numeric_column"], r["group_column"], out_dir)))
    if supervised:
        jobs.append(("models", lambda: plot_model_comparison(ml_results["cv_results"], out_dir)))
        jobs.append(("importance", lambda: plot_feature_importance(ml_results["permutation_importance"], out_dir)))
        if ml_results.get("confusion"):
            jobs.append(("confusion", lambda: plot_confusion_matrix(ml_results["confusion"], out_dir)))
    charts: list[dict[str, str]] = []
    problems: list[str] = []
    for name, job in jobs:
        if len(charts) >= max_charts:
            break
        try:
            chart = job()
            if chart:
                charts.append(chart)
        except Exception as exc:
            logger.warning("Chart '%s' failed: %s", name, type(exc).__name__)
            problems.append(f"Chart '{name}' could not be generated ({type(exc).__name__}).")
    return charts, problems
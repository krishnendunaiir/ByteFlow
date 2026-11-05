"""Statistical tools. Each function checks its assumptions and explains why a test was chosen."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from tools.data_inspection import (
    BOOLEAN, CATEGORICAL, NUMERIC, get_numeric_columns, infer_column_kind,
    require_columns, safe_float,
)
from utils.security import UserFacingError, sanitize_text_for_llm

ALPHA = 0.05
MAX_GROUPS = 10
MAX_LEVELS = 20
MAX_CORR_COLUMNS = 30


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _label(value: float, cuts: tuple[float, float, float]) -> str:
    v = abs(value)
    return "negligible" if v < cuts[0] else "small" if v < cuts[1] else "medium" if v < cuts[2] else "large"


def _strength(r: float) -> str:
    a = abs(r)
    return ("very strong" if a >= 0.8 else "strong" if a >= 0.6 else "moderate" if a >= 0.4
            else "weak" if a >= 0.2 else "negligible")


def descriptive_statistics(df: pd.DataFrame, columns: list[str] | None = None, max_columns: int = 40) -> dict[str, Any]:
    requested = list(df.columns) if columns is None else list(columns)
    require_columns(df, requested)
    cols = requested[:max_columns]
    numeric: dict[str, Any] = {}
    categorical: dict[str, Any] = {}
    for c in cols:
        kind = infer_column_kind(df[c])
        if kind == NUMERIC:
            x = _numeric_series(df, c).dropna()
            if len(x):
                numeric[str(c)] = {
                    "count": int(len(x)), "mean": safe_float(x.mean()), "std": safe_float(x.std()),
                    "min": safe_float(x.min()), "q25": safe_float(x.quantile(0.25)),
                    "median": safe_float(x.median()), "q75": safe_float(x.quantile(0.75)),
                    "max": safe_float(x.max()),
                    "skew": safe_float(x.skew()) if len(x) > 2 else None,
                    "kurtosis": safe_float(x.kurt()) if len(x) > 3 else None,
                }
        elif kind in (CATEGORICAL, BOOLEAN):
            vc = df[c].value_counts(dropna=True)
            total = int(vc.sum())
            categorical[str(c)] = {
                "n_unique": int(len(vc)),
                "top": [{"value": sanitize_text_for_llm(k, 40), "count": int(v),
                         "pct": round(100 * int(v) / total, 2)} for k, v in vc.head(5).items()],
            }
    return {"numeric": numeric, "categorical": categorical, "columns_described": len(cols),
            "truncated": len(cols) < len(requested)}


def normality_check(df: pd.DataFrame, column: str, seed: int = 42) -> dict[str, Any]:
    require_columns(df, [column])
    x = _numeric_series(df, column).dropna()
    if len(x) < 8 or x.nunique() <= 1:
        return {"column": str(column), "tested": False, "reason": "needs at least 8 non-constant numeric values"}
    sampled = len(x) > 5000
    if sampled:
        x = x.sample(5000, random_state=seed)
    stat, p = stats.shapiro(x)
    return {
        "column": str(column), "tested": True, "test": "Shapiro-Wilk", "n": int(len(x)),
        "statistic": safe_float(stat), "p_value": safe_float(p, 8), "is_normal": bool(p > ALPHA),
        "skewness": safe_float(x.skew()),
        "note": ("Sampled 5000 values. " if sampled else "")
                + "With large samples this test flags tiny departures from normality; check the skewness too.",
    }


def correlation_analysis(df: pd.DataFrame, columns: list[str] | None = None, method: str = "auto",
                         strong_threshold: float = 0.5) -> dict[str, Any]:
    if method not in {"auto", "pearson", "spearman"}:
        raise UserFacingError("method must be auto, pearson or spearman.")
    cols = columns if columns is not None else get_numeric_columns(df)
    require_columns(df, cols)
    data = pd.DataFrame({str(c): _numeric_series(df, c) for c in cols})
    data = data.loc[:, data.nunique() > 1]
    truncated = data.shape[1] > MAX_CORR_COLUMNS
    data = data.iloc[:, :MAX_CORR_COLUMNS]
    if data.shape[1] < 2:
        return {"tested": False, "reason": "need at least two non-constant numeric columns"}
    if method == "auto":
        method = "spearman" if (data.skew().abs() > 1).any() else "pearson"
    corr = data.corr(method=method)
    names = list(data.columns)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair = data[[names[i], names[j]]].dropna()
            if len(pair) < 8:
                continue
            res = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1]) if method == "pearson" \
                else stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
            r, p = float(res[0]), float(res[1])
            if np.isnan(r):
                continue
            pairs.append({"column_a": names[i], "column_b": names[j], "r": round(r, 4), "p_value": p, "n": int(len(pair))})
    if pairs:
        adj = multipletests([p["p_value"] for p in pairs], method="fdr_bh")[1]
        for p, a in zip(pairs, adj):
            p["p_adjusted"] = float(a)
            p["strength"] = _strength(p["r"])
            p["significant_after_adjustment"] = bool(a < ALPHA)
            p["p_value"] = safe_float(p["p_value"], 8)
            p["p_adjusted"] = safe_float(p["p_adjusted"], 8)
    strong = sorted((p for p in pairs if abs(p["r"]) >= strong_threshold), key=lambda p: -abs(p["r"]))[:20]
    return {
        "tested": True, "method": method, "n_columns": len(names), "n_pairs_tested": len(pairs),
        "strong_threshold": strong_threshold, "strong_pairs": strong,
        "matrix": {a: {b: safe_float(corr.loc[a, b], 3) for b in names} for a in names},
        "truncated": truncated,
        "notes": ["P-values are adjusted for multiple testing (Benjamini-Hochberg).",
                  "Correlation does not imply causation."]
                 + (["Spearman was chosen because at least one column is clearly skewed."]
                    if method == "spearman" and columns is None else []),
    }


def compare_groups(df: pd.DataFrame, numeric_column: str, group_column: str, alpha: float = ALPHA) -> dict[str, Any]:
    require_columns(df, [numeric_column, group_column])
    if numeric_column == group_column:
        raise UserFacingError("The numeric column and group column must differ.")
    data = pd.DataFrame({"y": _numeric_series(df, numeric_column), "g": df[group_column]}).dropna()
    data["g"] = data["g"].astype(str)
    if data.empty:
        return {"tested": False, "numeric_column": str(numeric_column), "group_column": str(group_column),
                "reason": "no usable numeric values for this column"}
    counts = data["g"].value_counts()
    if len(counts) > MAX_GROUPS:
        raise UserFacingError(f"'{sanitize_text_for_llm(group_column, 40)}' has too many distinct values to compare as groups.")
    keep = counts[counts >= 3].index
    groups = {k: data.loc[data["g"] == k, "y"].to_numpy() for k in keep}
    base = {"numeric_column": str(numeric_column), "group_column": str(group_column)}
    if len(groups) < 2:
        return {**base, "tested": False, "reason": "fewer than two groups with at least 3 observations"}

    def looks_normal(a: np.ndarray) -> bool:
        if len(a) >= 30:
            return True  # central limit theorem
        return bool(np.unique(a).size > 1 and stats.shapiro(a)[1] > alpha)

    arrays = list(groups.values())
    parametric = all(looks_normal(a) for a in arrays)
    n_total = sum(len(a) for a in arrays)
    notes = ["Association is not causation."]
    if len(keep) < len(counts):
        notes.append("Groups with fewer than 3 observations were left out.")
    keys = list(groups)
    if len(arrays) == 2:
        a, b = arrays
        if parametric:
            res = stats.ttest_ind(a, b, equal_var=False)
            sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
            d = (a.mean() - b.mean()) / sp if sp > 0 else 0.0
            test, stat, p = "Welch's t-test", float(res.statistic), float(res.pvalue)
            effect = {"name": "Cohen's d", "value": safe_float(d, 4), "magnitude": _label(d, (0.2, 0.5, 0.8))}
            why = "Both groups are approximately normal or large (n>=30); Welch's test allows unequal variances."
            higher = keys[0] if a.mean() > b.mean() else keys[1]
        else:
            res = stats.mannwhitneyu(a, b, alternative="two-sided")
            rb = 2 * float(res.statistic) / (len(a) * len(b)) - 1
            test, stat, p = "Mann-Whitney U test", float(res.statistic), float(res.pvalue)
            effect = {"name": "rank-biserial correlation", "value": safe_float(rb, 4),
                      "magnitude": _label(rb, (0.1, 0.3, 0.5))}
            why = "At least one small group departs from normality, so a rank-based test was used."
            higher = keys[0] if np.median(a) > np.median(b) else keys[1]
        notes.append(f"The group with the higher {'mean' if parametric else 'median'} is '{sanitize_text_for_llm(higher, 40)}'.")
    else:
        lev_p = float(stats.levene(*arrays, center="median").pvalue)
        if parametric and lev_p > alpha:
            res = stats.f_oneway(*arrays)
            grand = np.concatenate(arrays).mean()
            ssb = sum(len(a) * (a.mean() - grand) ** 2 for a in arrays)
            sst = sum(((a - grand) ** 2).sum() for a in arrays)
            eta = ssb / sst if sst > 0 else 0.0
            test, stat, p = "One-way ANOVA", float(res.statistic), float(res.pvalue)
            effect = {"name": "eta squared", "value": safe_float(eta, 4), "magnitude": _label(eta, (0.01, 0.06, 0.14))}
            why = "Groups are approximately normal/large and variances are similar (Levene test)."
        else:
            res = stats.kruskal(*arrays)
            eps = float(res.statistic) / (n_total - 1) if n_total > 1 else 0.0
            test, stat, p = "Kruskal-Wallis test", float(res.statistic), float(res.pvalue)
            effect = {"name": "epsilon squared", "value": safe_float(eps, 4), "magnitude": _label(eps, (0.01, 0.08, 0.26))}
            why = "Normality or equal-variance assumptions were not met, so a rank-based test was used."
        notes.append("A significant result shows at least one group differs; post-hoc tests were not run, "
                     "so which groups differ is not established.")
    return {
        **base, "tested": True, "test": test, "statistic": safe_float(stat, 4), "p_value": safe_float(p, 8),
        "significant": bool(p < alpha), "alpha": alpha, "effect_size": effect, "reason_for_test": why,
        "groups": [{"group": sanitize_text_for_llm(k, 40), "n": int(len(v)), "mean": safe_float(v.mean(), 4),
                    "median": safe_float(np.median(v), 4), "std": safe_float(v.std(ddof=1), 4)}
                   for k, v in groups.items()],
        "notes": notes,
    }


def categorical_association(df: pd.DataFrame, column_a: str, column_b: str, alpha: float = ALPHA) -> dict[str, Any]:
    require_columns(df, [column_a, column_b])
    if column_a == column_b:
        raise UserFacingError("Choose two different columns.")
    data = df[[column_a, column_b]].dropna().astype(str)
    base = {"column_a": str(column_a), "column_b": str(column_b)}
    if data[column_a].nunique() > MAX_LEVELS or data[column_b].nunique() > MAX_LEVELS:
        raise UserFacingError(f"Each column may have at most {MAX_LEVELS} distinct values for this test.")
    table = pd.crosstab(data[column_a], data[column_b])
    if table.shape[0] < 2 or table.shape[1] < 2 or len(data) < 10:
        return {**base, "tested": False, "reason": "need at least 2 levels in each column and 10 rows"}
    chi2_u, _, _, expected = stats.chi2_contingency(table, correction=False)
    n = int(table.to_numpy().sum())
    cramers_v = float(np.sqrt(chi2_u / (n * (min(table.shape) - 1))))
    notes = ["Association is not causation."]
    if table.shape == (2, 2) and expected.min() < 5:
        _, p = stats.fisher_exact(table.to_numpy())
        test = "Fisher's exact test"
        notes.append("Expected counts below 5, so Fisher's exact test replaced chi-square.")
        stat = None
    else:
        chi2, p, _, _ = stats.chi2_contingency(table)
        test, stat = "Chi-square test of independence", float(chi2)
        if (expected < 5).mean() > 0.2:
            notes.append("More than 20% of cells have expected counts below 5; treat the p-value with caution.")
    return {
        **base, "tested": True, "test": test, "statistic": safe_float(stat, 4), "p_value": safe_float(float(p), 8),
        "significant": bool(p < alpha), "alpha": alpha, "n": n,
        "effect_size": {"name": "Cramer's V", "value": safe_float(cramers_v, 4),
                        "magnitude": _label(cramers_v, (0.1, 0.3, 0.5))},
        "contingency_table": ({str(i): {str(c): int(v) for c, v in row.items()} for i, row in table.iterrows()}
                              if table.shape[0] <= 6 and table.shape[1] <= 6 else None),
        "notes": notes,
    }
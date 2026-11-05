"""Dataset inspection tools. Pure functions: they never modify the DataFrame passed in."""
from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

from utils.security import UserFacingError, sanitize_text_for_llm

NUMERIC, CATEGORICAL, DATETIME = "numeric", "categorical", "datetime"
BOOLEAN, TEXT, IDENTIFIER = "boolean", "text", "identifier"

_TARGET_HINTS = {"target", "label", "class", "outcome", "churn", "survived", "price",
                 "salary", "default", "status", "y", "result", "diagnosis"}
_NON_NEGATIVE_HINTS = {"age", "price", "count", "quantity", "qty", "salary", "income",
                       "amount", "height", "weight", "distance", "duration"}
_SENTINELS = {-9999, -999, 999, 9999}
_DATE_RE = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")


def safe_float(x: Any, ndigits: int = 6) -> float | None:
    """Convert to a JSON-friendly float; NaN/inf/None become None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(v) or np.isinf(v)) else round(v, ndigits)


def _tokens(name: Any) -> set[str]:
    return set(re.split(r"[^a-z0-9]+", str(name).lower())) - {""}


def is_texty(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a safe error if any requested column does not exist."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise UserFacingError("Unknown column(s): " + ", ".join(sanitize_text_for_llm(c, 40) for c in missing))


def _parses_as_dates(sample: pd.Series) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= 0.9)


def infer_column_kind(s: pd.Series) -> str:
    """Classify a column as numeric, categorical, datetime, boolean, text or identifier."""
    if pd.api.types.is_bool_dtype(s):
        return BOOLEAN
    if pd.api.types.is_datetime64_any_dtype(s):
        return DATETIME
    non_null = s.dropna()
    n_unique = non_null.nunique()
    if pd.api.types.is_numeric_dtype(s):
        if (pd.api.types.is_integer_dtype(s) and len(non_null) > 0 and n_unique >= 0.95 * len(non_null)
                and _tokens(s.name) & {"id", "index", "idx", "key"}):
            return IDENTIFIER
        return NUMERIC
    if len(non_null) == 0:
        return CATEGORICAL
    text = non_null.astype(str)
    if text.head(200).str.match(_DATE_RE).mean() >= 0.9 and _parses_as_dates(text.head(200)):
        return DATETIME
    unique_ratio = n_unique / len(non_null)
    avg_len = text.str.len().mean()
    if len(non_null) > 20 and unique_ratio >= 0.98:
        return TEXT if avg_len > 40 else IDENTIFIER
    if avg_len > 40 and unique_ratio > 0.5:
        return TEXT
    return CATEGORICAL


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if infer_column_kind(df[c]) == NUMERIC]


def get_categorical_columns(df: pd.DataFrame, max_levels: int = 20) -> list[str]:
    return [str(c) for c in df.columns
            if infer_column_kind(df[c]) in (CATEGORICAL, BOOLEAN) and 2 <= df[c].nunique() <= max_levels]


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Column-by-column profile (types, kinds, missing, uniques, basic stats)."""
    n = len(df)
    columns = []
    for name in df.columns:
        s = df[name]
        kind = infer_column_kind(s)
        n_missing = int(s.isna().sum())
        n_unique = int(s.nunique(dropna=True))
        info: dict[str, Any] = {
            "name": str(name), "dtype": str(s.dtype), "kind": kind, "n_missing": n_missing,
            "pct_missing": round(100 * n_missing / n, 2) if n else 0.0,
            "n_unique": n_unique, "low_cardinality": n_unique <= 10,
            "numeric": None, "top_values": None,
        }
        if kind == NUMERIC:
            x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(x):
                info["numeric"] = {
                    "min": safe_float(x.min()), "max": safe_float(x.max()), "mean": safe_float(x.mean()),
                    "median": safe_float(x.median()), "std": safe_float(x.std()),
                    "skew": safe_float(x.skew()) if len(x) > 2 else None,
                }
        elif kind in (CATEGORICAL, BOOLEAN):
            vc = s.value_counts(dropna=True).head(5)
            info["top_values"] = [{"value": sanitize_text_for_llm(k, 40), "count": int(v)} for k, v in vc.items()]
        columns.append(info)
    return {
        "n_rows": n, "n_columns": int(df.shape[1]),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1024**2, 3),
        "columns": columns,
    }


def compact_profile_for_llm(profile: dict[str, Any], max_columns: int = 60) -> dict[str, Any]:
    """Tiny, sanitized summary for LLM prompts. Contains NO raw cell values."""
    cols = []
    for c in profile["columns"][:max_columns]:
        item: dict[str, Any] = {
            "name": sanitize_text_for_llm(c["name"], 60), "kind": c["kind"],
            "pct_missing": c["pct_missing"], "n_unique": c["n_unique"],
        }
        if c.get("numeric"):
            item.update({k: c["numeric"][k] for k in ("min", "max", "mean")})
        cols.append(item)
    return {"n_rows": profile["n_rows"], "n_columns": profile["n_columns"],
            "columns_shown": len(cols), "columns": cols}


def missing_value_analysis(df: pd.DataFrame) -> dict[str, Any]:
    n_rows, n_cols = df.shape
    miss = df.isna().sum()
    per_column = [
        {"column": str(c), "n_missing": int(v), "pct_missing": round(100 * int(v) / n_rows, 2)}
        for c, v in miss.sort_values(ascending=False).items() if v > 0
    ]
    total = int(miss.sum())
    rows_any = int(df.isna().any(axis=1).sum())
    return {
        "total_missing_cells": total,
        "pct_cells_missing": round(100 * total / (n_rows * n_cols), 3) if n_rows and n_cols else 0.0,
        "rows_with_any_missing": rows_any,
        "pct_rows_with_any_missing": round(100 * rows_any / n_rows, 2) if n_rows else 0.0,
        "per_column": per_column,
    }


def detect_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    dup = df.duplicated(keep="first")
    n = int(dup.sum())
    return {"n_duplicate_rows": n, "pct_duplicate_rows": round(100 * n / len(df), 2) if len(df) else 0.0,
            "example_row_positions": [int(i) for i in np.flatnonzero(dup.to_numpy())[:5]]}


def detect_outliers(df: pd.DataFrame, columns: list[str] | None = None,
                    iqr_multiplier: float = 1.5) -> dict[str, Any]:
    """IQR rule. Outliers are flagged for review, not assumed to be errors."""
    if not 0.5 <= iqr_multiplier <= 5:
        raise UserFacingError("iqr_multiplier must be between 0.5 and 5.")
    cols = columns if columns is not None else get_numeric_columns(df)
    require_columns(df, cols)
    checked, skipped = [], []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) < 8 or x.nunique() <= 2:
            skipped.append({"column": str(c), "reason": "too few values or binary column"})
            continue
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            skipped.append({"column": str(c), "reason": "IQR is zero"})
            continue
        lo, hi = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        out = x[(x < lo) | (x > hi)]
        checked.append({
            "column": str(c), "n_outliers": int(len(out)), "pct_outliers": round(100 * len(out) / len(x), 2),
            "lower_fence": safe_float(lo), "upper_fence": safe_float(hi),
            "min_outlier": safe_float(out.min()) if len(out) else None,
            "max_outlier": safe_float(out.max()) if len(out) else None,
        })
    return {"method": "IQR", "iqr_multiplier": iqr_multiplier, "by_column": checked, "skipped": skipped,
            "n_columns_with_outliers": sum(1 for c in checked if c["n_outliers"] > 0)}


def detect_suspicious_values(df: pd.DataFrame, max_issues: int = 50) -> list[dict[str, str]]:
    """Find type and consistency problems that usually need cleaning."""
    issues: list[dict[str, str]] = []

    def add(col: Any, issue: str, detail: str, severity: str = "warning") -> None:
        if len(issues) < max_issues:
            issues.append({"column": str(col), "issue": issue, "detail": detail, "severity": severity})

    for c in df.columns:
        s = df[c]
        non_null = s.dropna()
        if len(non_null) == 0:
            add(c, "all_missing", "Column has no values.", "high")
            continue
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            x = pd.to_numeric(s, errors="coerce")
            n_inf = int(np.isinf(x).sum())
            if n_inf:
                add(c, "infinite_values", f"{n_inf} infinite value(s).", "high")
            xf = x.replace([np.inf, -np.inf], np.nan).dropna()
            if xf.nunique() <= 1 and len(xf) > 1:
                add(c, "constant_column", "Every value is identical.")
            if _tokens(c) & _NON_NEGATIVE_HINTS:
                n_neg = int((xf < 0).sum())
                if n_neg:
                    add(c, "negative_values", f"{n_neg} negative value(s) in a column that is normally non-negative.")
            n_sent = int(xf.isin(list(_SENTINELS)).sum())
            if n_sent >= 3:
                add(c, "possible_sentinel", f"{n_sent} values look like placeholder codes (e.g. 999, -999).")
        elif is_texty(s):
            text = non_null.astype(str)
            rate = float(pd.to_numeric(text, errors="coerce").notna().mean())
            if rate >= 0.95:
                add(c, "numeric_stored_as_text", f"{round(100 * rate, 1)}% of values are numbers stored as text.")
            elif 0.05 < rate < 0.95:
                add(c, "mixed_types", f"{round(100 * rate, 1)}% numeric and the rest text.")
            n_ws = int((text != text.str.strip()).sum())
            if n_ws:
                add(c, "whitespace_padding", f"{n_ws} value(s) have leading/trailing spaces.")
            if rate < 0.5:
                stripped = text.str.strip()
                if stripped.str.lower().nunique() < stripped.nunique():
                    add(c, "inconsistent_case", "Same category appears with different capitalisation.")
            if text.nunique() <= 1 and len(text) > 1:
                add(c, "constant_column", "Every value is identical.")
    return issues


def suggest_candidate_targets(profile: dict[str, Any], top: int = 5) -> list[dict[str, Any]]:
    """Heuristic ranking of columns that could be prediction targets."""
    cols = profile["columns"]
    scored = []
    for i, c in enumerate(cols):
        if c["kind"] in (IDENTIFIER, TEXT, DATETIME) or c["pct_missing"] > 30 or c["n_unique"] < 2:
            continue
        score, reasons = 0, []
        if _tokens(c["name"]) & _TARGET_HINTS:
            score += 2
            reasons.append("name suggests an outcome")
        if i == len(cols) - 1:
            score += 1
            reasons.append("last column")
        if c["kind"] in (CATEGORICAL, BOOLEAN) and 2 <= c["n_unique"] <= 10:
            score += 2
            reasons.append("few categories (classification)")
        elif c["kind"] == NUMERIC and c["n_unique"] == 2:
            score += 2
            reasons.append("binary (classification)")
        elif c["kind"] == NUMERIC:
            score += 1
            reasons.append("numeric (regression)")
        if score:
            scored.append((score, i, {"column": c["name"], "reason": "; ".join(reasons)}))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [item for _, _, item in scored[:top]]


def assess_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Run every quality check and return one combined report."""
    return {
        "missing": missing_value_analysis(df),
        "duplicates": detect_duplicates(df),
        "outliers": detect_outliers(df),
        "suspicious": detect_suspicious_values(df),
    }
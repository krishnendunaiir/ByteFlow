"""Data cleaning tools. They always work on a COPY: the original is never modified."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tools.data_inspection import (
    CATEGORICAL, NUMERIC, infer_column_kind, is_texty,
)


def clean_dataset(df: pd.DataFrame, *, strip_whitespace: bool = True, convert_numeric_strings: bool = True,
                  drop_duplicates: bool = True, impute: bool = False) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Return (cleaned_copy, log). Imputation is OFF by default: for machine learning
    it happens inside the model pipeline, fitted on training folds only (no leakage)."""
    out = df.copy()
    log: list[dict[str, Any]] = []

    if strip_whitespace:
        for c in [c for c in out.columns if is_texty(out[c])]:
            s = out[c]
            stripped = s.map(lambda v: v.strip() if isinstance(v, str) else v)
            changed = int(((s != stripped) & s.notna()).sum())
            if changed:
                out[c] = stripped
                log.append({"action": "strip_whitespace", "column": str(c), "detail": f"{changed} value(s) trimmed"})

    if convert_numeric_strings:
        for c in [c for c in out.columns if is_texty(out[c])]:
            non_null = out[c].dropna()
            if non_null.empty:
                continue
            text = non_null.astype(str)
            if text.str.match(r"^0\d").any():  # zip codes / codes with leading zeros
                continue
            if pd.to_numeric(text, errors="coerce").notna().mean() >= 0.95:
                converted = pd.to_numeric(out[c], errors="coerce")
                lost = int(out[c].notna().sum() - converted.notna().sum())
                out[c] = converted
                log.append({"action": "convert_to_numeric", "column": str(c),
                            "detail": f"{lost} non-numeric value(s) set to missing"})

    if drop_duplicates:
        before = len(out)
        out = out.drop_duplicates(keep="first").reset_index(drop=True)
        if before - len(out):
            log.append({"action": "drop_duplicates", "column": "*", "detail": f"{before - len(out)} row(s) removed"})

    if impute:
        for c in out.columns:
            n_missing = int(out[c].isna().sum())
            if not n_missing:
                continue
            kind = infer_column_kind(out[c])
            if kind == NUMERIC:
                out[c] = out[c].fillna(out[c].median())
                log.append({"action": "impute_median", "column": str(c), "detail": f"{n_missing} value(s) filled"})
            elif kind == CATEGORICAL and not out[c].mode().empty:
                out[c] = out[c].fillna(out[c].mode().iloc[0])
                log.append({"action": "impute_mode", "column": str(c), "detail": f"{n_missing} value(s) filled"})
    return out, log


def recommend_cleaning(quality: dict[str, Any], max_items: int = 25) -> list[dict[str, str]]:
    """Plain-language cleaning advice derived from the quality report."""
    recs: list[dict[str, str]] = []
    n_dup = quality["duplicates"]["n_duplicate_rows"]
    if n_dup:
        recs.append({"issue": f"{n_dup} duplicate row(s)",
                     "recommendation": "Remove exact duplicates unless repeated records are genuinely expected."})
    for m in quality["missing"]["per_column"]:
        p = m["pct_missing"]
        if p > 50:
            rec = "Consider dropping this column; most values are missing."
        elif p >= 5:
            rec = "Impute (median for numbers, most frequent for categories), inside the model pipeline."
        else:
            rec = "Few values missing: impute, or drop those rows if the dataset is large."
        recs.append({"issue": f"'{m['column']}' has {p}% missing", "recommendation": rec})
    for o in quality["outliers"]["by_column"]:
        if o["n_outliers"]:
            rec = ("Investigate: many extreme values may mean a skewed distribution, not errors."
                   if o["pct_outliers"] > 5 else "Review the values; keep them unless they are data-entry errors.")
            recs.append({"issue": f"'{o['column']}' has {o['n_outliers']} IQR outlier(s)", "recommendation": rec})
    advice = {
        "numeric_stored_as_text": "Convert the column to numeric.",
        "mixed_types": "Inspect and standardise the values; mixed types break calculations.",
        "whitespace_padding": "Trim leading/trailing spaces.",
        "inconsistent_case": "Standardise capitalisation so categories are not split.",
        "constant_column": "Drop it: a constant column carries no information.",
        "negative_values": "Check for data-entry errors.",
        "possible_sentinel": "Replace placeholder codes with proper missing values.",
        "infinite_values": "Replace infinite values with missing values or fix the source.",
        "all_missing": "Drop the column.",
    }
    for it in quality["suspicious"]:
        recs.append({"issue": f"'{it['column']}': {it['detail']}", "recommendation": advice.get(it["issue"], "Review manually.")})
    return recs[:max_items]
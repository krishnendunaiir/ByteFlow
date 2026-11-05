"""Report creation. Dataset text is escaped so it can never inject Markdown/HTML into the report."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from utils.security import ensure_within_directory, safe_internal_filename, sanitize_dataframe_for_export

_MD_SPECIAL = re.compile(r"([\\`*_\[\]#|~])")


def md_escape(value: Any, max_len: int = 80) -> str:
    """Make arbitrary text safe to place in Markdown (and neutralise HTML)."""
    s = re.sub(r"\s+", " ", str(value)).strip()[:max_len]
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return _MD_SPECIAL.sub(r"\\\1", s)


def fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return f"{x:,}"
    try:
        return f"{float(x):,.{nd}f}"
    except (TypeError, ValueError):
        return md_escape(x)


def fmt_p(p: Any) -> str:
    if p is None:
        return "n/a"
    return "< 0.001" if float(p) < 0.001 else f"{float(p):.3f}"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_None._\n"
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _clean_narrative(text: Any, max_len: int = 3000) -> str:
    """LLM text is kept as Markdown but with HTML neutralised."""
    return str(text)[:max_len].replace("<", "&lt;").replace(">", "&gt;").strip()


def _ok_steps(analysis: dict[str, Any], action: str) -> list[dict[str, Any]]:
    return [s for s in analysis.get("steps", []) if s.get("action") == action and s.get("status") == "ok"
            and s.get("result", {}).get("tested", True)]


def deterministic_findings(analysis: dict[str, Any], ml: dict[str, Any] | None) -> list[str]:
    """Findings computed directly from results (no LLM)."""
    out: list[str] = []
    for s in _ok_steps(analysis, "correlation"):
        for p in s["result"].get("strong_pairs", [])[:3]:
            out.append(f"{p['strength'].capitalize()} correlation between {md_escape(p['column_a'])} and "
                       f"{md_escape(p['column_b'])} (r = {p['r']:.2f}, adjusted p {fmt_p(p['p_adjusted'])}).")
    for s in _ok_steps(analysis, "group_comparison"):
        r = s["result"]
        verdict = "differs significantly" if r["significant"] else "does not differ significantly"
        out.append(f"{md_escape(r['numeric_column'])} {verdict} across {md_escape(r['group_column'])} "
                   f"({r['test']}, p {fmt_p(r['p_value'])}, {r['effect_size']['magnitude']} effect).")
    for s in _ok_steps(analysis, "categorical_association"):
        r = s["result"]
        verdict = "associated" if r["significant"] else "not significantly associated"
        out.append(f"{md_escape(r['column_a'])} and {md_escape(r['column_b'])} are {verdict} "
                   f"({r['test']}, p {fmt_p(r['p_value'])}, Cramer's V {fmt(r['effect_size']['value'], 2)}).")
    if ml and ml.get("task") == "clustering":
        out.append(f"K-means found {ml['best_k']} clusters (silhouette {fmt(ml['best_silhouette'])}).")
    elif ml:
        m = ml["test_metrics"][ml["primary_metric"]]    
        out.append(f"Best model ({md_escape(ml['best_model'])}) scored {ml['primary_metric']} = {fmt(m)} on held-out data "
                   f"versus {fmt(ml['baseline_test_primary'])} for the baseline.")
    return out


def build_report_markdown(
    *,
    question: str,
    upload_info: dict[str, Any],
    profile: dict[str, Any],
    quality: dict[str, Any],
    cleaning_log: list[dict[str, Any]],
    cleaning_recommendations: list[dict[str, str]],
    analysis_results: dict[str, Any],
    ml_results: dict[str, Any] | None,
    charts: list[dict[str, str]],
    narrative: dict[str, Any] | None = None,
    extra_limitations: list[str] | None = None,
    critic_feedback: list[str] | None = None,
) -> str:
    
    """Assemble the 11-section report. Every number comes from the result dictionaries."""
    narrative = narrative or {}
    L: list[str] = ["# Automated Data Analysis Report", ""]
    findings = deterministic_findings(analysis_results, ml_results)

    L += ["## 1. Executive summary", ""]
    L.append(_clean_narrative(narrative["executive_summary"]) if narrative.get("executive_summary") else
             f"The dataset has {profile['n_rows']:,} rows and {profile['n_columns']} columns. "
             f"{len(findings)} finding(s) were computed; see sections 6 to 9.")
    L += ["", f"**Question asked:** {md_escape(question, 500)}", ""]

    L += ["## 2. Dataset overview", "",
          f"- File: {md_escape(upload_info.get('display_name', 'upload'))} ({upload_info.get('file_type', '?')})",
          f"- Rows: {profile['n_rows']:,}  |  Columns: {profile['n_columns']}", ""]
    L.append(table(["Column", "Kind", "Missing %", "Unique"],
                   [[md_escape(c["name"]), c["kind"], fmt(c["pct_missing"], 1), fmt(c["n_unique"])] for c in profile["columns"][:60]]))
    if profile["n_columns"] > 60:
        L.append("_Only the first 60 columns are listed._\n")

    miss, dup, outl = quality["missing"], quality["duplicates"], quality["outliers"]
    L += ["## 3. Data quality assessment", "",
          f"- Missing cells: {miss['total_missing_cells']:,} ({fmt(miss['pct_cells_missing'], 2)}% of all cells); "
          f"{miss['rows_with_any_missing']:,} rows have at least one missing value.",
          f"- Duplicate rows: {dup['n_duplicate_rows']:,} ({fmt(dup['pct_duplicate_rows'], 2)}%).", "",
          "**Columns with missing values**", "",
          table(["Column", "Missing", "Missing %"], [[md_escape(m["column"]), fmt(m["n_missing"]), fmt(m["pct_missing"], 1)] for m in miss["per_column"][:15]]),
          "**Outliers (IQR rule, flagged for review, not assumed to be errors)**", "",
          table(["Column", "Outliers", "%", "Lower fence", "Upper fence"],
                [[md_escape(o["column"]), fmt(o["n_outliers"]), fmt(o["pct_outliers"], 1), fmt(o["lower_fence"]), fmt(o["upper_fence"])]
                 for o in outl["by_column"] if o["n_outliers"]][:15]),
          "**Suspicious values and type issues**", "",
          table(["Column", "Issue", "Detail"], [[md_escape(i["column"]), i["issue"], md_escape(i["detail"], 120)] for i in quality["suspicious"][:20]])]

    L += ["## 4. Data cleaning performed or recommended", "", "**Performed (on a working copy; the original is untouched)**", "",
          table(["Action", "Column", "Detail"], [[c["action"], md_escape(c["column"]), md_escape(c["detail"], 120)] for c in cleaning_log]),
          "**Recommended**", "",
          table(["Issue", "Recommendation"], [[md_escape(r["issue"], 120), md_escape(r["recommendation"], 160)] for r in cleaning_recommendations])]

    desc = next((s["result"] for s in _ok_steps(analysis_results, "descriptive_stats")), None)
    L += ["## 5. Exploratory data analysis", ""]
    if desc and desc["numeric"]:
        L.append(table(["Variable", "Mean", "Std", "Min", "Median", "Max", "Skew"],
                       [[md_escape(k), fmt(v["mean"]), fmt(v["std"]), fmt(v["min"]), fmt(v["median"]), fmt(v["max"]), fmt(v["skew"], 2)]
                        for k, v in list(desc["numeric"].items())[:25]]))
    else:
        L.append("_No descriptive statistics were requested in the analysis plan._\n")
    if desc and desc["categorical"]:
        L.append(table(["Variable", "Unique", "Most common (count)"],
                       [[md_escape(k), fmt(v["n_unique"]), ", ".join(f"{md_escape(t['value'], 30)} ({t['count']})" for t in v["top"][:3])]
                        for k, v in list(desc["categorical"].items())[:15]]))

    L += ["## 6. Statistical findings", ""]
    any_stat = False
    for s in _ok_steps(analysis_results, "correlation"):
        r = s["result"]
        any_stat = True
        L += [f"**Correlation ({r['method']}, {r['n_pairs_tested']} pairs; p-values adjusted for multiple testing)**", "",
              table(["Variable A", "Variable B", "r", "Strength", "Adj. p", "n"],
                    [[md_escape(p["column_a"]), md_escape(p["column_b"]), fmt(p["r"], 2), p["strength"], fmt_p(p["p_adjusted"]), fmt(p["n"])] for p in r["strong_pairs"]])]
    for s in _ok_steps(analysis_results, "group_comparison"):
        r = s["result"]
        any_stat = True
        L += [f"**{md_escape(r['numeric_column'])} by {md_escape(r['group_column'])}: {r['test']}**", "",
              f"- Statistic {fmt(r['statistic'])}, p {fmt_p(r['p_value'])}, effect size {r['effect_size']['name']} = "
              f"{fmt(r['effect_size']['value'])} ({r['effect_size']['magnitude']}).",
              f"- Why this test: {md_escape(r['reason_for_test'], 200)}",
              *[f"- {md_escape(n, 220)}" for n in r["notes"]], "",
              table(["Group", "n", "Mean", "Median"], [[md_escape(g["group"]), fmt(g["n"]), fmt(g["mean"]), fmt(g["median"])] for g in r["groups"]])]
    for s in _ok_steps(analysis_results, "categorical_association"):
        r = s["result"]
        any_stat = True
        L += [f"**{md_escape(r['column_a'])} vs {md_escape(r['column_b'])}: {r['test']}**", "",
              f"- p {fmt_p(r['p_value'])}; Cramer's V = {fmt(r['effect_size']['value'])} ({r['effect_size']['magnitude']}).",
              *[f"- {md_escape(n, 220)}" for n in r["notes"]], ""]
    for s in _ok_steps(analysis_results, "normality_check"):
        r = s["result"]
        any_stat = True
        L.append(f"- Normality of {md_escape(r['column'])}: Shapiro-Wilk p {fmt_p(r['p_value'])} "
                 f"({'approximately normal' if r['is_normal'] else 'not normal'}; skewness {fmt(r['skewness'], 2)}).")
    for s in _ok_steps(analysis_results, "feature_importance"):
        r = s["result"]
        any_stat = True
        L += ["", f"**Feature ranking for {md_escape(r['target'])} (mutual information, exploratory)**", "",
              table(["Feature", "Score"], [[md_escape(x["feature"]), fmt(x["score"], 4)] for x in r["ranking"][:10]])]
    skipped = [s for s in analysis_results.get("steps", []) if s.get("status") != "ok"]
    if skipped:
        L += ["", "**Steps not run**", ""] + [f"- {md_escape(s['action'])}: {md_escape(s.get('reason', 'skipped'), 160)}" for s in skipped]
    if not any_stat:
        L.append("_No statistical tests were run._")

    L += ["", "## 7. Machine learning results", ""]
    if ml_results and ml_results.get("task") == "clustering":
        m = ml_results
        L += [f"- Task: **clustering (k-means)** | Features: {', '.join(md_escape(f, 30) for f in m['features'][:20])}",
              f"- Best number of clusters: **{m['best_k']}** (silhouette {fmt(m['best_silhouette'])}).", "",
              table(["k", "Silhouette"], [[k, fmt(v)] for k, v in m["silhouette_by_k"].items()]),
              table(["Cluster", "Size"], [[k, fmt(v)] for k, v in m["cluster_sizes"].items()])]
        L += [f"- Warning: {md_escape(w, 200)}" for w in m["warnings"]]
        L += [f"- Note: {md_escape(n, 220)}" for n in m["notes"]]
    elif ml_results:
        m = ml_results
        m = ml_results
        L += [f"- Task: **{m['task']}** | Target: **{md_escape(m['target'])}** | Rows used: {m['n_rows_used']:,} "
              f"(train {m['n_train']:,} / test {m['n_test']:,}) | {m['cv_folds']}-fold cross-validation",
              f"- Features used ({len(m['features_used'])}): {', '.join(md_escape(f, 30) for f in m['features_used'][:20])}", ""]
        L.append(table(["Model", f"CV {m['primary_metric']} (mean ± std)", "Train score"],
                       [[md_escape(k) + (" (best)" if k == m["best_model"] else ""), f"{fmt(v['primary_cv_mean'])} ± {fmt(v['primary_cv_std'])}", fmt(v["primary_train_mean"])]
                        for k, v in m["cv_results"].items()]))
        L += ["**Held-out test results (best model, evaluated once)**", "",
              table(["Metric", "Value"], [[k, fmt(v)] for k, v in m["test_metrics"].items()]),
              f"Baseline {m['primary_metric']} on the same test set: {fmt(m['baseline_test_primary'])}.", "",
              "**Most important features (permutation importance)**", "",
              table(["Feature", "Importance"], [[md_escape(x["feature"]), fmt(x["importance_mean"], 4)] for x in m["permutation_importance"][:10]])]
        if m["leakage_warnings"] or m["features_excluded"]:
            L += ["**Excluded features and leakage checks**", ""] + \
                 [f"- {md_escape(e['column'], 40)}: {md_escape(e['reason'], 120)}" for e in m["features_excluded"][:15]] + [""]
        L += [f"- Warning: {md_escape(w, 200)}" for w in m["warnings"]]
        L += [f"- Note: {md_escape(n, 220)}" for n in m["notes"]]
    else:
        L.append("_Machine learning was not applicable or was not requested for this question._")

    L += ["", "## 8. Visualizations", ""]
    L += [f"- **{md_escape(c['title'], 80)}** ({md_escape(Path(c['path']).name, 60)}): {md_escape(c['description'], 160)}" for c in charts] \
        or ["_No charts were generated._"]

    L += ["", "## 9. Important discoveries", ""]
    items = narrative.get("discoveries") or findings
    L += [f"- {_clean_narrative(i, 400)}" for i in items[:10]] or ["_No notable findings._"]

    limits = ["Results describe associations in this dataset only; they do not establish causation.",
              "Outliers were identified with the IQR rule and were not removed.",
              "Statistical significance depends on sample size; check effect sizes, not only p-values."]
    if ml_results and ml_results.get("task") == "clustering":
        limits += ml_results["warnings"] + ["Clusters are a statistical grouping, not proof of real-world segments."]
    elif ml_results:
        limits += ml_results["warnings"] + ["Model performance is estimated on a single held-out test split; it may not generalise to new data or other time periods."]
    limits += (extra_limitations or []) + list(narrative.get("limitations", []))
    L += ["", "## 10. Limitations", ""] + [f"- {_clean_narrative(x, 300)}" for x in dict.fromkeys(limits)]

    recs = list(narrative.get("recommendations", []))

    recs.extend(critic_feedback or [])

    recs.extend(
        r["recommendation"]
        for r in cleaning_recommendations[:5]
        if isinstance(r, dict) and r.get("recommendation")
    )

    if not recs:
        recs = ["No specific data issues were detected."]

    L += ["", "## 11. Recommendations", ""] + [
        f"- {_clean_narrative(r, 300)}"
        for r in dict.fromkeys(recs)
    ]

    return "\n".join(L) + "\n"


def save_report(markdown: str, out_dir: Path) -> Path:
    """Write the report with a random safe filename inside out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = ensure_within_directory(out_dir, safe_internal_filename(".md"))
    path.write_text(markdown, encoding="utf-8")
    return path


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV bytes safe to open in Excel (formula injection neutralised)."""
    return sanitize_dataframe_for_export(df).to_csv(index=False).encode("utf-8")
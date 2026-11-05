import uuid

import numpy as np
import pandas as pd

from utils.config import get_settings
from agents.graph import build_graph
from agents.state import initial_state


def get_field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def create_churn_dataset(n=300):
    rng = np.random.default_rng(42)

    tenure = rng.integers(1, 73, n)
    monthly_charges = rng.uniform(25, 120, n).round(2)
    support_tickets = rng.poisson(1.5, n)

    contract_type = rng.choice(
        ["monthly", "annual", "two_year"],
        size=n,
        p=[0.55, 0.30, 0.15],
    )

    internet_service = rng.choice(
        ["fiber", "dsl", "none"],
        size=n,
        p=[0.50, 0.35, 0.15],
    )

    risk_score = (
        0.65 * support_tickets
        - 0.035 * tenure
        + 0.015 * monthly_charges
        + 1.0 * (contract_type == "monthly")
        + rng.normal(0, 1.2, n)
    )

    churn_indices = np.argsort(risk_score)[-36:]

    churn = np.array(["no"] * n, dtype=object)
    churn[churn_indices] = "yes"

    return pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "support_tickets": support_tickets,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "churn": churn,
    })


def main():
    print("=" * 60)
    print("LEAN AGENT - LIVE ML SMOKE TEST")
    print("=" * 60)

    df = create_churn_dataset()

    print("\nDATASET:")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("Churn distribution:")
    print(df["churn"].value_counts())

    question = (
        "Analyze this customer churn dataset. "
        "Calculate descriptive statistics, identify important "
        "features, and perform classification using churn as "
        "the target variable. Include feature importance, "
        "useful visualizations, and a final report."
    )

    upload_info = {
        "filename": "synthetic_churn.csv",
        "rows": len(df),
        "columns": list(df.columns),
    }

    session_id = f"live_ml_smoke_{uuid.uuid4().hex[:8]}"

    state = initial_state(
        df=df,
        question=question,
        upload_info=upload_info,
        settings=get_settings(),
        session_id=session_id,
    )

    # Keep this test to one critic attempt.
    state["max_retries"] = 0

    print("\nBuilding graph...")
    graph = build_graph()

    print("Running live workflow...")
    print("This will make real LLM API requests.\n")

    try:
        result = graph.invoke(state)

    except Exception as exc:
        print("\nGRAPH EXECUTION FAILED")
        print(type(exc).__name__, str(exc))
        raise

    critic = result.get("critic") or {}
    ml_results = result.get("ml_results") or {}
    analysis_results = result.get("analysis_results") or {}
    charts = result.get("charts") or []
    errors = result.get("errors") or []

    print("\n" + "=" * 60)
    print("WORKFLOW RESULTS")
    print("=" * 60)

    print("\nSTATUS:", result.get("status"))
    print("RETRY COUNT:", result.get("retry_count"))

    print("\nCLARIFICATION:")
    print(result.get("clarification"))

    print("\nPLAN:")
    print(result.get("plan"))

    print("\nUNDERSTANDING:")
    print(str(result.get("understanding"))[:3000])

    print("\nPROGRESS:")
    for item in result.get("progress", []):
        print("-", item)

    print("\nANALYSIS RESULTS:")
    print("Type:", type(analysis_results).__name__)

    if isinstance(analysis_results, dict):
        for key, value in analysis_results.items():
            print(f"- {key}: {type(value).__name__}")

    elif isinstance(analysis_results, list):
        for item in analysis_results:
            print("-", item)

    print("\nML RESULTS:")

    if isinstance(ml_results, dict):
        print("Keys:", list(ml_results.keys()))
        print("Task:", ml_results.get("task"))
        print("Target:", ml_results.get("target"))
        print("Best model:", ml_results.get("best_model"))
        print("Test metrics:", ml_results.get("test_metrics"))

    else:
        print(ml_results)

        print("\nCRITIC DIAGNOSTICS:")
    print("Raw critic:", result.get("critic"))
    print("Retry target:", result.get("retry_target"))
    print("Retry count:", result.get("retry_count"))
    print("Max retries:", result.get("max_retries"))
    print("Status:", result.get("status"))
    print("Errors:", result.get("errors"))
    print("Clarification:", result.get("clarification"))

    print("\nPROGRESS LOG:")
    for item in result.get("progress", []):
        print("-", item)

    print("\nVISUALIZATIONS:")
    print("Chart count:", len(charts))

    for chart in charts:
        print("-", chart)

    report = result.get("report_markdown")

    print("\nREPORT:")
    print("Generated:", bool(report))
    print("Characters:", len(report) if report else 0)

    print("\nERRORS:")

    if errors:
        for error in errors:
            print("-", error)
    else:
        print("None")

    print("\n" + "=" * 60)
    print("LIVE TEST FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()
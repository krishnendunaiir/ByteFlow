"""Shared fixtures: a synthetic dataset, test settings, and a scripted FakeLLM (no API key needed)."""
import copy

import numpy as np
import pandas as pd
import pytest

from agents.llm import LLMError
from utils.config import Settings


class FakeLLM:
    """Stands in for the real LLM. `responses` maps a schema name to a dict (or a function returning one)."""

    def __init__(self, responses=None, fail=()):
        self.responses = copy.deepcopy(responses if responses is not None else good_responses())
        self.fail = set(fail)
        self.calls: list[tuple[str, str, str]] = []

    def complete_json(self, system, user, schema, max_tokens=1500):
        name = schema.__name__
        self.calls.append((name, system, user))
        if name in self.fail:
            raise LLMError("fake provider outage")
        r = self.responses[name]
        return schema.model_validate(r(system, user) if callable(r) else r)

    def n_calls(self, name):
        return sum(1 for c in self.calls if c[0] == name)


def good_responses():
    return {
        "UnderstandingResult": {"dataset_summary": "A customer table.", "inferred_objective": "Find what drives churn.",
                                "candidate_targets": ["churn", "not_a_column"]},
        "AnalysisPlan": {"objective": "Drivers of churn", "needs_ml": True, "ml_task": "classification",
                         "target_column": "churn",
                         "steps": [{"action": "descriptive_stats", "columns": [], "rationale": "overview"},
                                   {"action": "correlation", "columns": ["age", "income", "score"], "rationale": "relationships"},
                                   {"action": "group_comparison", "target": "score", "group_column": "group", "rationale": "differences"},
                                   {"action": "categorical_association", "columns": ["group", "city"], "rationale": "association"}]},
        "CriticVerdict": {"approved": True, "issues": []},
        "ReportNarrative": {"executive_summary": "Churn is related mainly to income.",
                            "discoveries": ["Income and age move together."], "limitations": [],
                            "recommendations": ["Collect more customer data."]},
    }


@pytest.fixture
def settings(tmp_path):
    return Settings(outputs_dir=tmp_path / "out", max_retries=2)


@pytest.fixture(scope="session")
def sample_df():
    rng = np.random.default_rng(0)
    n = 300
    age = rng.normal(40, 10, n)
    income = 1000 * age + rng.normal(0, 3000, n)
    group = rng.choice(list("ABC"), n)
    city = rng.choice(["Kochi", "Pune", "Delhi"], n).astype(object)
    city[50:70] = np.nan
    return pd.DataFrame({
        "customer_id": range(n), "age": age, "income": income, "group": group,
        "score": np.where(group == "C", 70, 50) + rng.normal(0, 8, n), "city": city,
        "secret_note": ["TOPSECRET123"] * n,
        "churn": (income + rng.normal(0, 8000, n) > 40000).astype(int),
    })


@pytest.fixture
def fake_llm():
    return FakeLLM()
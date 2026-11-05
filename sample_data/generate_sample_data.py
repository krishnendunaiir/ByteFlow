"""Creates synthetic demo datasets (no real people or companies). Run from the project root:

    python sample_data/generate_sample_data.py

Files created next to this script: customer_churn.csv, customer_churn.xlsx, housing_prices.csv
Both datasets contain deliberate quality problems (missing values, duplicates, outliers) for the agent to find."""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SEED = 42


def make_churn(n: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["DSL", "Fiber optic", "None"], n, p=[0.35, 0.45, 0.20])
    tenure = np.clip(rng.gamma(2.2, 14, n), 1, 72).round().astype(int)
    base = {"DSL": 45, "Fiber optic": 85, "None": 22}
    monthly = np.array([base[i] for i in internet]) + rng.normal(0, 8, n)
    support_calls = rng.poisson(np.where(contract == "Month-to-month", 2.2, 1.0))
    logit = (-1.0 + 1.3 * (contract == "Month-to-month") - 0.04 * tenure + 0.015 * (monthly - 60)
             + 0.35 * support_calls - 1.0 * (contract == "Two year"))
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame({
        "customer_id": np.arange(10001, 10001 + n),
        "tenure_months": tenure,
        "contract_type": contract,
        "internet_service": internet,
        "monthly_charges": monthly.round(2),
        "total_charges": (monthly * tenure + rng.normal(0, 40, n)).round(2),
        "support_calls": support_calls,
        "region": rng.choice(["North", "South", "East", "West"], n),
        "churn": np.where(churn, "Yes", "No"),
    })
    df.loc[rng.choice(n, int(0.03 * n), replace=False), "monthly_charges"] = np.nan   # 3% missing
    df.loc[rng.choice(n, int(0.02 * n), replace=False), "region"] = np.nan            # 2% missing
    df.loc[[5, 17], "monthly_charges"] = 999.99                                       # placeholder-style outliers
    return pd.concat([df, df.iloc[100:112]], ignore_index=True)                       # 12 duplicate rows


def make_housing(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    hood = rng.choice(list("ABCDE"), n)
    premium = {"A": 60_000, "B": 30_000, "C": 0, "D": -20_000, "E": -40_000}
    sqft = rng.normal(1800, 450, n).clip(500, 4500).round()
    df = pd.DataFrame({
        "sqft": sqft,
        "bedrooms": np.clip((sqft / 600 + rng.normal(0, 0.6, n)).round(), 1, 6).astype(int),
        "bathrooms": np.clip((sqft / 900 + rng.normal(0, 0.5, n)).round(), 1, 5).astype(int),
        "age_years": rng.integers(0, 80, n),
        "neighborhood": hood,
        "distance_to_center_km": rng.gamma(3, 3, n).round(1),
    })
    df["price"] = (60_000 + 110 * df["sqft"] + 9_000 * df["bathrooms"] - 700 * df["age_years"]
                   - 2_500 * df["distance_to_center_km"] + df["neighborhood"].map(premium)
                   + rng.normal(0, 18_000, n)).round(-2)
    df.loc[rng.choice(n, int(0.02 * n), replace=False), "sqft"] = np.nan
    return df


if __name__ == "__main__":
    churn, housing = make_churn(), make_housing()
    churn.to_csv(OUT / "customer_churn.csv", index=False)
    churn.to_excel(OUT / "customer_churn.xlsx", index=False)
    housing.to_csv(OUT / "housing_prices.csv", index=False)
    print(f"customer_churn.csv/.xlsx: {len(churn)} rows, churn rate {(churn['churn'] == 'Yes').mean():.0%}")
    print(f"housing_prices.csv: {len(housing)} rows")
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """与真实数据同 schema 的小型合成数据。"""
    rng = np.random.default_rng(0)
    n = 400
    yn = lambda: rng.choice(["Yes", "No"], n)
    svc = lambda: rng.choice(["Yes", "No", "No internet service"], n)
    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(20, 115, n).round(2)
    df = pd.DataFrame({
        "id": np.arange(n),
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": yn(), "Dependents": yn(),
        "tenure": tenure,
        "PhoneService": yn(),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": svc(), "OnlineBackup": svc(),
        "DeviceProtection": svc(), "TechSupport": svc(),
        "StreamingTV": svc(), "StreamingMovies": svc(),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": yn(),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)",
             "Credit card (automatic)"], n),
        "MonthlyCharges": monthly,
        "TotalCharges": (tenure * monthly * rng.uniform(0.95, 1.05, n)).round(2),
    })
    fiber_month = (df["InternetService"] == "Fiber optic") & (df["Contract"] == "Month-to-month")
    p = np.where(fiber_month, 0.55, 0.12)
    df["Churn"] = np.where(rng.random(n) < p, "Yes", "No")
    return df

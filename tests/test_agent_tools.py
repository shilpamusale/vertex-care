# tests/test_agent_tools.py
"""Regression tests for the explanation tool's SHAP setup."""

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from vertexcare.agents.agent_tools import explanation_tool

FEATURES = ["age", "hypertension", "sumcomorbidities", "diabetes"]


@pytest.fixture(scope="module")
def assets():
    """Synthetic cohort where readmission depends only on comorbidities and diabetes."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "patient_id": np.arange(n),
            "age": rng.integers(20, 90, n),
            "hypertension": rng.integers(0, 2, n),
            "sumcomorbidities": rng.integers(0, 6, n),
            "diabetes": rng.integers(0, 2, n),
        }
    )
    logit = 1.2 * df["sumcomorbidities"] + 2.5 * df["diabetes"] - 4
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    imputer = SimpleImputer().fit(df[FEATURES])
    model = LogisticRegression(max_iter=1000).fit(pd.DataFrame(imputer.transform(df[FEATURES]), columns=FEATURES), y)
    # Patient 0: many comorbidities and diabetes, average age, no hypertension.
    df.loc[0, FEATURES] = [55, 0, 5, 1]
    return model, imputer, df


def test_top_factors_are_the_true_drivers(assets):
    """Regression: SHAP used the patient as its own background, so every value was zero and the
    'top factors' were just the first columns in feature order (age, hypertension, ...)."""
    model, imputer, df = assets
    result = explanation_tool(0, model, imputer, df)
    assert result["status"] == "Success", result
    assert set(result["top_risk_factors"][:2]) == {"sumcomorbidities", "diabetes"}

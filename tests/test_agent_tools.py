# tests/test_agent_tools.py
"""Tests for the explanation tool: SHAP setup and the factor descriptions the agent reads."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from evals.run_eval import GOLD_PATH
from evals.schema import load_gold_set
from vertexcare.agents.agent_tools import explanation_tool
from vertexcare.agents.feature_glossary import FACTOR_CAVEAT, PROTECTED, describe_factor

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


# --- SHAP setup ------------------------------------------------------------------------------


def test_top_factors_are_the_true_drivers(assets):
    """Regression: SHAP used the patient as its own background, so every value was zero and the
    'top factors' were just the first columns in feature order (age, hypertension, ...)."""
    model, imputer, df = assets
    result = explanation_tool(0, model, imputer, df)
    assert result["status"] == "Success", result
    assert {f["feature"] for f in result["top_risk_factors"][:2]} == {"sumcomorbidities", "diabetes"}


# --- what the agent reads --------------------------------------------------------------------


def test_factors_carry_meaning_value_and_direction(assets):
    model, imputer, df = assets
    result = explanation_tool(0, model, imputer, df)
    by_feature = {f["feature"]: f for f in result["top_risk_factors"]}
    assert by_feature["sumcomorbidities"] == {
        "feature": "sumcomorbidities",
        "description": "Number of chronic conditions",
        "patient_value": 5,
        "effect": "raises predicted risk",
        "use_in_care_plan": True,
    }
    assert by_feature["diabetes"]["patient_value"] == "Yes"
    assert result["caveat"] == FACTOR_CAVEAT


def test_tool_output_is_json_serialisable(assets):
    """chw_agent json.dumps every observation; numpy types would crash the ReAct loop."""
    model, imputer, df = assets
    json.dumps(explanation_tool(0, model, imputer, df))


@pytest.mark.parametrize(
    "feature, value, contribution, expected",
    [
        ("race_black", 1, 0.3, {"patient_value": "Yes", "use_in_care_plan": False, "note": PROTECTED}),
        ("sdoh_pcp_1_0", 0, 0.2, {"patient_value": "No", "effect": "raises predicted risk"}),
        ("sdoh_pcp_1_0", 1, -0.2, {"patient_value": "Yes", "effect": "lowers predicted risk"}),
        ("age", np.float64("nan"), 0.1, {"patient_value": "Missing"}),
        ("unknown_column", 1, 0.1, {"description": "unknown_column", "use_in_care_plan": False}),
    ],
)
def test_describe_factor(feature, value, contribution, expected):
    factor = describe_factor(feature, value, contribution)
    assert {k: factor[k] for k in expected} == expected


def test_gold_fixtures_match_glossary():
    """Gold-set fixtures stand in for real tool output, so they must use the same wording."""
    for case in load_gold_set(GOLD_PATH).cases:
        for factor in case.fixtures.explanation["top_risk_factors"]:
            reference = describe_factor(factor["feature"], 1, 1.0)
            for key in ("description", "use_in_care_plan", "note"):
                assert factor.get(key) == reference.get(key), (case.case_id, factor["feature"], key)

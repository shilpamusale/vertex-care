# tests/test_agent_e2e.py

import logging
import asyncio
import sys

# import pandas as pd
from unittest.mock import patch

# Import the agent runner that we will test
from vertexcare.agents.chw_agent import run_agent

# --- Synthetic Patient Test Cases ---
# This data defines the fake outputs our tools will return for each test case.
SYNTHETIC_PATIENTS = {
    # Case 1: High clinical risk, no social barriers
    1: {
        "prediction": {"readmission_risk_score": 0.75},
        "explanation": {"top_risk_factors": ["age", "sumcomorbidities", "diabetes"]},
        "notes": {"notes": "Patient seems to be managing well at home."},
    },
    # Case 2: Moderate clinical risk, but a clear transportation barrier
    2: {
        "prediction": {"readmission_risk_score": 0.45},
        "explanation": {
            "top_risk_factors": ["hypertension", "llm_transportation_issue", "age"]
        },
        "notes": {
            "notes": "Patient is worried about getting a ride "
            "to their cardiology appointment next week."
        },
    },
    # Case 3: Low clinical risk, positive outlook
    3: {
        "prediction": {"readmission_risk_score": 0.15},
        "explanation": {"top_risk_factors": ["race_black", "sdoh_pcp_1_0", "age"]},
        "notes": {
            "notes": "Patient in good spirits. "
            "Reports taking all medications as prescribed."
        },
    },
    # Case 4: Conflicting Information
    4: {
        "prediction": {"readmission_risk_score": 0.65},
        "explanation": {
            "top_risk_factors": ["age", "llm_financial_concern", "hypertension"]
        },
        "notes": {
            "notes": "Patient initially denied any financial issues, "
            "but later mentioned they may have trouble affording "
            "their new medication co-pay."
        },
    },
    # Case 5: Multiple Competing Priorities
    5: {
        "prediction": {"readmission_risk_score": 0.85},
        "explanation": {
            "top_risk_factors": [
                "sumcomorbidities",
                "llm_transportation_issue",
                "llm_financial_concern",
            ]
        },
        "notes": {
            "notes": "Patient needs a ride to their appointment "
            "and also needs to be enrolled in the food assistance program."
        },
    },
}

# --- Test Execution ---


async def run_test_case(patient_id: int):
    """
    Runs the agent for a single test case, mocking the tool outputs.
    """
    print(f"\n{'='*20} RUNNING TEST CASE FOR PATIENT {patient_id} {'='*20}")

    # We patch the tools in the 'chw_agent' namespace, which is where they are used.
    with patch(
        "vertexcare.agents.chw_agent.prediction_tool",
        return_value=SYNTHETIC_PATIENTS[patient_id]["prediction"],
    ):
        with patch(
            "vertexcare.agents.chw_agent.explanation_tool",
            return_value=SYNTHETIC_PATIENTS[patient_id]["explanation"],
        ):
            with patch(
                "vertexcare.agents.chw_agent.notes_tool",
                return_value=SYNTHETIC_PATIENTS[patient_id]["notes"],
            ):
                await run_agent(patient_id)


async def main():
    """Runs all defined test cases."""
    for patient_id in SYNTHETIC_PATIENTS:
        await run_test_case(patient_id)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Ensure logs are sent to the console
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    # Clear existing handlers and add our new one
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)

    asyncio.run(main())

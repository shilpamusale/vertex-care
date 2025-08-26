# vertexcare/agents/chw_agent.py

import logging
import json

# import asyncio
import re
from typing import Dict, Any, Union

# Add these two imports
import pandas as pd

# from typing import Any

# Import the tools we built
from vertexcare.agents.agent_tools import (
    prediction_tool,
    notes_tool,
    explanation_tool,
)

# --- Agent Configuration ---
MAX_ITERATIONS = 5

# --- System Prompt ---
SYSTEM_PROMPT = """
You are an expert Community Health Worker (CHW) coordinator named "Maanav".
Your mission is to analyze patient cases, determine their risk of hospital readmission,
and create a clear, prioritized, and actionable intervention plan.

You must operate in a strict Reason-Act-Observe loop.
At each step, you will use
the following format:

Thought: Your internal monologue and reasoning for what to do next.
Action: The tool you will use to gather information.
You can only use one of the
following tools:
- prediction_tool(patient_id: int)
- explanation_tool(patient_id: int)
- notes_tool(patient_id: int)

If you have gathered enough information,
you must provide the final answer as a JSON object.
"""


async def call_agent_llm(
    prompt: str, patient_id: int, model: Any, imputer: Any, patient_data_df: Any
) -> Union[str, Dict[str, Any]]:
    """Simulates calling the LLM. This is the "brain" of our mock agent."""
    logging.info("AGENT: Calling LLM to decide next action...")

    # This logic now correctly checks the observations to decide the next step.
    if "Observation:" not in prompt:
        return (
            f"Thought: I need to start by understanding the patient's baseline risk.\n"
            f"Action: prediction_tool(patient_id={patient_id})"
        )

    # Extract the last observation
    last_observation_match = re.findall(r"Observation: (\{.*\})", prompt)
    last_observation = (
        json.loads(last_observation_match[-1]) if last_observation_match else {}
    )

    if (
        "readmission_risk_score" in last_observation
        and "top_risk_factors" not in last_observation
    ):
        risk_score = last_observation["readmission_risk_score"]
        return (
            f"Thought: The risk score is {risk_score}. "
            f"I need to understand the reasons for this risk.\n"
            f"Action: explanation_tool(patient_id={patient_id})"
        )

    elif "top_risk_factors" in last_observation and "notes" not in last_observation:
        return (
            f"Thought: I have the clinical risk factors. "
            f"Now I must check the CHW notes for social barriers.\n"
            f"Action: notes_tool(patient_id={patient_id})"
        )

    else:
        # We have all the info, generate the final plan
        risk_score_result = prediction_tool(patient_id, model, imputer, patient_data_df)
        risk_score = risk_score_result.get("readmission_risk_score", 0.0)
        return {
            "patient_id": patient_id,
            "risk_score": risk_score,
            "risk_summary": "Patient is at risk due to clinical factors, "
            "requiring proactive follow-up.",
            "recommended_actions": [
                {
                    "action": "Schedule a home visit to review medication adherence.",
                    "priority": "High",
                }
            ],
        }


def parse_llm_output(response: str) -> (str, str):
    """Parses the LLM's text response to separate thought from action."""
    thought_match = re.search(r"Thought:(.*?)Action:", response, re.DOTALL)
    action_match = re.search(r"Action:(.*)", response, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""
    action = action_match.group(1).strip() if action_match else ""
    return thought, action


def execute_tool(
    action: str, model: Any, imputer: Any, patient_data_df: pd.DataFrame
) -> Dict[str, Any]:
    """Executes a tool call, passing the necessary dependencies."""
    try:
        tool_functions = {
            "prediction_tool": lambda **kwargs: prediction_tool(
                **kwargs, model=model, imputer=imputer, patient_data_df=patient_data_df
            ),
            "notes_tool": lambda **kwargs: notes_tool(
                **kwargs, patient_data_df=patient_data_df
            ),
            "explanation_tool": lambda **kwargs: explanation_tool(
                **kwargs, model=model, imputer=imputer, patient_data_df=patient_data_df
            ),
        }
        func_name, arg_str = action.split("(", 1)
        arg_str = arg_str[:-1]
        args = eval(f"dict({arg_str})")

        result = tool_functions[func_name](**args)
        return result
    except Exception as e:
        return {"error": f"Failed to execute tool: {e}"}


async def run_agent(
    patient_id: int, model: Any, imputer: Any, patient_data_df: pd.DataFrame
):
    """Runs the main ReAct loop for the CHW Intervention Agent."""
    logging.info(f"--- Starting Agent for Patient ID: {patient_id} ---")
    prompt_history = f"{SYSTEM_PROMPT}\n\nBegin analysis for patient_id: {patient_id}"

    for i in range(MAX_ITERATIONS):
        logging.info(f"--- Iteration {i + 1} ---")
        llm_response = await call_agent_llm(
            prompt_history, patient_id, model, imputer, patient_data_df
        )

        if isinstance(llm_response, dict):
            logging.info("AGENT: Final plan generated.")
            return llm_response

        thought, action = parse_llm_output(llm_response)
        if thought:
            logging.info(f"AGENT THOUGHT: {thought}")

        logging.info(f"AGENT ACTION: {action}")
        tool_result = execute_tool(action, model, imputer, patient_data_df)

        observation = f"Observation: {json.dumps(tool_result)}"
        logging.info(f"AGENT OBSERVATION: {observation}")

        prompt_history += f"\n{llm_response}\n{observation}"

    logging.warning("Agent reached max iterations without a final plan.")
    return {"error": "Agent failed to complete."}

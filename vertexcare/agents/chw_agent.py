# vertexcare/agents/chw_agent.py

import logging
import json

# import asyncio
import re

# from pathlib import Path
from typing import Dict, Any, Union

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
Your mission is to analyze patient cases,
determine their risk of hospital readmission,
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


async def call_agent_llm(prompt: str) -> Union[str, Dict[str, Any]]:
    """
    Simulates calling the LLM. Returns a string for intermediate steps
    and a dictionary for the final answer to avoid parsing errors.
    """
    logging.info("AGENT: Calling LLM to decide next action...")

    patient_id_match = re.search(r"patient_id: (\d+)", prompt)
    patient_id = int(patient_id_match.group(1)) if patient_id_match else "unknown"

    # --- MOCK LLM RESPONSE (Smarter Logic) ---
    if 'Observation: {"readmission_risk_score"' not in prompt:
        return f"""
Thought: I need to start by understanding the patient's baseline risk.
I should use the prediction_tool.
Action: prediction_tool(patient_id={patient_id})
"""
    elif 'Observation: {"top_risk_factors"' not in prompt:
        risk_score_match = re.search(r'"readmission_risk_score": ([\d.]+)', prompt)
        risk_score = float(risk_score_match.group(1)) if risk_score_match else "unknown"
        return f"""
Thought: The risk score is {risk_score}. I need to understand the reasons for this risk
to create a targeted plan. I should use the explanation_tool.
Action: explanation_tool(patient_id={patient_id})
"""
    elif 'Observation: {"notes"' not in prompt:
        return f"""
Thought: I have the clinical risk factors. Now I must check the CHW notes for
any social or logistical barriers that might be contributing to this risk.
I should use the notes_tool.
Action: notes_tool(patient_id={patient_id})
"""
    else:
        # This is the "brain" of our mock. It now looks at the prompt history
        # to make a tailored decision for each patient case.
        risk_score_result = prediction_tool(patient_id)
        risk_score = risk_score_result.get("readmission_risk_score", 0.0)

        # Case 2 & 5: Transportation Issue
        if "llm_transportation_issue" in prompt:
            return {
                "patient_id": patient_id,
                "risk_score": risk_score,
                "risk_summary": "Patient is at risk due "
                "to a critical transportation barrier "
                "for an upcoming appointment.",
                "recommended_actions": [
                    {
                        "action": "Arrange medical transport immediately.",
                        "priority": "High",
                    }
                ],
            }
        # Case 4: Financial Concern
        elif "llm_financial_concern" in prompt:
            return {
                "patient_id": patient_id,
                "risk_score": risk_score,
                "risk_summary": "Patient is at risk due to a financial concern "
                "regarding medication co-pays.",
                "recommended_actions": [
                    {
                        "action": "Refer patient to social work for financial "
                        "assistance programs.",
                        "priority": "High",
                    }
                ],
            }
        # Case 3: Low Risk
        elif risk_score < 0.2:
            return {
                "patient_id": patient_id,
                "risk_score": risk_score,
                "risk_summary": "Patient is at low risk "
                "and appears to be managing well.",
                "recommended_actions": [
                    {
                        "action": "Schedule a standard 30-day follow-up call.",
                        "priority": "Low",
                    }
                ],
            }
        # Default Case (High Clinical Risk)
        else:
            return {
                "patient_id": patient_id,
                "risk_score": risk_score,
                "risk_summary": "Patient is at high risk due to advanced "
                "age and multiple comorbidities.",
                "recommended_actions": [
                    {
                        "action": "Schedule a home visit to "
                        "review medication adherence.",
                        "priority": "High",
                    }
                ],
            }
    # --- END MOCK LLM RESPONSE ---


def parse_llm_output(response: str) -> (str, str):
    """Parses the LLM's text response to separate thought from action."""
    thought_match = re.search(r"Thought:(.*?)Action:", response, re.DOTALL)
    action_match = re.search(r"Action:(.*)", response, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""
    action = action_match.group(1).strip() if action_match else ""
    return thought, action


def execute_tool(action: str) -> Dict[str, Any]:
    """Executes a tool call based on the parsed action string."""
    try:
        tool_functions = {
            "prediction_tool": prediction_tool,
            "notes_tool": notes_tool,
            "explanation_tool": explanation_tool,
        }
        func_name, arg_str = action.split("(", 1)
        arg_str = arg_str[:-1]
        args = eval(f"dict({arg_str})")

        result = tool_functions[func_name](**args)
        return result
    except Exception as e:
        return {"error": f"Failed to execute tool: {e}"}


async def run_agent(patient_id: int):
    """Runs the main ReAct loop for the CHW Intervention Agent."""
    logging.info(f"--- Starting Agent for Patient ID: {patient_id} ---")

    prompt_history = f"{SYSTEM_PROMPT}\n\nBegin analysis for patient_id: {patient_id}"

    for i in range(MAX_ITERATIONS):
        logging.info(f"--- Iteration {i + 1} ---")

        llm_response = await call_agent_llm(prompt_history)

        if isinstance(llm_response, dict):
            logging.info("AGENT: Final plan generated.")
            final_plan = llm_response
            print("\n--- FINAL INTERVENTION PLAN ---")
            print(json.dumps(final_plan, indent=2))
            return final_plan

        thought, action = parse_llm_output(llm_response)

        if thought:
            logging.info(f"AGENT THOUGHT: {thought}")

        logging.info(f"AGENT ACTION: {action}")
        tool_result = execute_tool(action)

        observation = f"Observation: {json.dumps(tool_result)}"
        logging.info(f"AGENT OBSERVATION: {observation}")

        prompt_history += f"\n{llm_response}\n{observation}"

    logging.warning("Agent reached max iterations without a final plan.")
    return {"error": "Agent failed to complete."}

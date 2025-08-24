# vertexcare/agents/chw_agent.py

import logging
import json
import asyncio
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
You are an expert Community Health Worker
(CHW) coordinator named "Maanav".
Your mission is to analyze patient cases,
determine their risk of hospital readmission,
and create a clear, prioritized,
and actionable intervention plan.

You must operate in a strict Reason-Act-Observe loop.
At each step, you will use
the following format:

Thought: Your internal monologue
and reasoning for what to do next.
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
    Simulates calling the LLM.
    Returns a string for intermediate steps
    and a dictionary for the final answer
    to avoid parsing errors.
    """
    logging.info("AGENT: Calling LLM to decide next action...")

    patient_id_match = re.search(r"patient_id: (\d+)", prompt)
    patient_id = int(patient_id_match.group(1)) if patient_id_match else "unknown"

    # --- MOCK LLM RESPONSE ---
    if "prediction_tool" not in prompt:
        return f"""
Thought: I need to start by understanding the patient's baseline risk.
I should use the prediction_tool.
Action: prediction_tool(patient_id={patient_id})
"""
    elif "explanation_tool" not in prompt:
        risk_score_match = re.search(r'"readmission_risk_score": ([\d.]+)', prompt)
        risk_score = float(risk_score_match.group(1)) if risk_score_match else "unknown"
        return f"""
Thought: The risk score is {risk_score}.
I need to understand the reasons for this risk to create a targeted plan.
I should use the explanation_tool.
Action: explanation_tool(patient_id={patient_id})
"""
    elif "notes_tool" not in prompt:
        return f"""
Thought: The top risk factors are clinical.
I should check the CHW notes to see if there are any social or
logistical barriers that might be contributing to this risk.
I should use the notes_tool.
Action: notes_tool(patient_id={patient_id})
"""
    else:
        # CORRECTED: Instead of parsing, get the real risk score directly.
        # This makes the mock much more robust and realistic.
        risk_score_result = prediction_tool(patient_id)
        risk_score = risk_score_result.get("readmission_risk_score", 0.0)

        return {
            "patient_id": patient_id,
            "risk_score": risk_score,
            "risk_summary": "Patient is at risk due to clinical factors, "
            + "requiring proactive follow-up.",
            "recommended_actions": [
                {
                    "action": "Schedule a home visit to review medication adherence.",
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_patient_id = 101

    # We need to run the agent to get the final plan
    final_plan = asyncio.run(run_agent(test_patient_id))

    # Then we print the final plan in the desired format
    print("\n--- FINAL INTERVENTION PLAN ---")
    print(json.dumps(final_plan, indent=2))

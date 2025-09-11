# vertexcare/agents/chw_agent.py

import logging
import json
import re
import time
from typing import Dict, Any, Union

from google.api_core import exceptions
import pandas as pd

import vertexai
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold

# Import the tools we built
from vertexcare.agents.agent_tools import (
    prediction_tool,
    notes_tool,
    explanation_tool,
)

# Initialize Vertex AI right after imports
# This correctly configures the SDK for the rest of the file.
vertexai.init(project="vertexcare", location="us-central1")

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


# --- Agent Configuration ---
MAX_ITERATIONS = 5

# --- System Prompt ---
SYSTEM_PROMPT = (
    'You are an expert Community Health Worker (CHW) coordinator named "Maanav".\n'
    "Your mission is to analyze patient cases, "
    "determine their risk of hospital readmission,\n"
    "and create a clear, prioritized, and actionable intervention plan.\n\n"
    "You must operate in a strict Reason-Act-Observe loop.\n"
    "At each step, you will use the following format:\n"
    "You MUST stop after each 'Action:' and wait for a new 'Observation:'.\n\n"
    "Thought: Your internal monologue and reasoning for what to do next.\n"
    "Action: The tool you will use to gather information.\n"
    "You can only use one of the following tools:\n"
    "- prediction_tool(patient_id: int)\n"
    "- explanation_tool(patient_id: int)\n"
    "- notes_tool(patient_id: int)\n\n"
    "If you have gathered enough information, you MUST provide the final answer\n"
    "formatted as:\n"
    "Final Answer: <JSON object containing the plan>\n\n"
    "The JSON object MUST have the following exact structure:\n"
    "{\n"
    '  "patient_id": integer,\n'
    '  "readmission_risk_score": float (between 0.0 and 1.0, from the\n'
    "    prediction_tool),\n"
    '  "risk_summary": "A concise, one-sentence summary of the patient\'s main\n'
    '    risk factors based on the tools.",\n'
    '  "recommended_actions": [\n'
    "    {\n"
    '      "priority": "High" | "Medium" | "Low",\n'
    '      "action": "A specific, actionable intervention step."\n'
    "    }\n"
    "  ]\n"
    "}\n"
)


async def call_agent_llm(
    prompt: str, patient_id: int, model: Any, imputer: Any, patient_data_df: Any, retries: int = 3, delay: int = 5
) -> Union[str, Dict[str, Any]]:
    """
    Calls the live Gemini model with retry logic and exponential backoff.
    """
    logging.info("AGENT: Calling live Gemini model to decide next action...")

    # --- CHANGE 1: Update to the correct, modern model name ---
    live_model = GenerativeModel("gemini-2.5-flash")

    for i in range(retries):
        try:
            # --- CHANGE 2: Model is now initialized outside the loop ---
            response = live_model.generate_content(prompt, safety_settings=safety_settings)

            if not response.candidates:
                return {"error": "Response was blocked by safety settings."}

            return response.text  # Success

        except exceptions.ResourceExhausted as e:
            logging.warning(
                f"An exception occurred: {e}" + f"Rate limit hit. Retrying in {delay}s... (Attempt {i + 1}/{retries})"
            )
            time.sleep(delay)
            delay *= 2  # Double the delay for the next potential retry

        except Exception as e:
            # Add exc_info=True to get the full stack trace in the logs
            logging.error(f"Error calling Gemini for agent action: {e}", exc_info=True)
            if i < retries - 1:
                logging.info(f"Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                return {"error": str(e)}

    return {"error": f"Failed to call Gemini API after {retries} attempts."}


def parse_llm_output(response: str) -> (str, str):
    """Parses the LLM's text response to separate thought from action."""
    thought_match = re.search(r"Thought:(.*?)Action:", response, re.DOTALL)
    action_match = re.search(r"Action:(.*)", response, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""
    action = action_match.group(1).strip() if action_match else ""
    return thought, action


def execute_tool(action: str, model: Any, imputer: Any, patient_data_df: pd.DataFrame) -> Dict[str, Any]:
    """Executes a tool call, passing the necessary dependencies."""
    logging.info(f"Executing tool with action: {action}")
    try:
        tool_functions = {
            "prediction_tool": lambda **kwargs: prediction_tool(
                **kwargs, model=model, imputer=imputer, patient_data_df=patient_data_df
            ),
            "notes_tool": lambda **kwargs: notes_tool(**kwargs, patient_data_df=patient_data_df),
            "explanation_tool": lambda **kwargs: explanation_tool(
                **kwargs, model=model, imputer=imputer, patient_data_df=patient_data_df
            ),
        }

        func_name = action.split("(", 1)[0].strip()
        match = re.search(r"\(.*?(\d+).*?\)", action)
        if not match:
            raise ValueError(f"Could not find a valid integer patient ID in action: {action}")

        patient_id = int(match.group(1))
        args = {"patient_id": patient_id}

        if func_name not in tool_functions:
            raise ValueError(f"Unknown tool specified: {func_name}")

        result = tool_functions[func_name](**args)
        return result
    except Exception as e:
        logging.error(f"Failed to execute tool '{action}'. Error: {e}", exc_info=True)
        return {"error": f"Failed to execute tool: {e}"}


async def run_agent(patient_id: int, model: Any, imputer: Any, patient_data_df: pd.DataFrame):
    """Runs the main ReAct loop for the CHW Intervention Agent."""
    logging.info(f"--- Starting Agent for Patient ID: {patient_id} ---")
    prompt_history = f"{SYSTEM_PROMPT}\n\nBegin analysis for patient_id: {patient_id}"

    for i in range(MAX_ITERATIONS):
        logging.info(f"--- Iteration {i + 1} ---")
        llm_response = await call_agent_llm(prompt_history, patient_id, model, imputer, patient_data_df)

        if isinstance(llm_response, dict) and "error" in llm_response:
            logging.error(f"Error from LLM call: {llm_response['error']}")
            return llm_response  # Propagate the error

        if "Final Answer:" in llm_response:
            logging.info("AGENT: Final plan detected.")
            try:
                json_str = llm_response.split("Final Answer:")[1].strip()
                if json_str.startswith("```json"):
                    json_str = json_str[7:]
                if json_str.endswith("```"):
                    json_str = json_str[:-3]
                json_str = json_str.strip()
                final_plan = json.loads(json_str)
                return final_plan
            except Exception as e:
                logging.error(f"Failed to parse final plan from LLM response: '{llm_response}'." f" Error: {e}")
                return {"error": "Failed to parse the final plan from the LLM."}

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

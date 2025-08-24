# vertexcare/features/llm_feature_extractor.py

import logging
import json
import asyncio
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import aiohttp


# Implement Gemini API.
async def call_gemini_api(note: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls the Gemini API to extract features from a note using a defined schema.
    """
    logging.info(f"Calling Gemini API for note: '{note[:50]}...'")

    # --- 1. Engineer the Prompt ---
    prompt = f"""
You are a highly trained healthcare analyst.
Your task is to extract specific, structured information
from the following Community Health Worker (CHW) note.

Analyze the note and return a JSON object
that strictly adheres to the provided schema.
Only return the JSON object, with no other text or explanations.

CHW Note:
"{note}"
"""

    # --- 2. Construct the API Payload ---
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }

    # --- 3. Make the API Call ---
    try:
        apiKey = ""  # This will be handled by secret manager.
        base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        model_name = "gemini-2.5-flash-preview-05-20:generateContent"
        apiUrl = f"{base_url}/{model_name}?key={apiKey}"

        # --- REAL API CALL LOGIC ---
        async with aiohttp.ClientSession() as session:
            async with session.post(apiUrl, json=payload) as response:
                response.raise_for_status()
                result = await response.json()

                text_response = (
                    result.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "{}")
                )
                return json.loads(text_response)
        # --- END REAL API LOGIC ---

    except Exception as e:
        logging.error(f"Error calling Gemini API for note '{note[:50]}...': {e}")
        # Return a default schema-compliant object on failure
        return {
            "transportation_issue": False,
            "financial_concern": False,
            "medication_adherence_mentioned": False,
            "patient_sentiment": "unknown",
        }


def define_extraction_schema() -> Dict[str, Any]:
    """Defines the JSON schema for the features we want to extract."""
    return {
        "type": "OBJECT",
        "properties": {
            "transportation_issue": {"type": "BOOLEAN"},
            "financial_concern": {"type": "BOOLEAN"},
            "medication_adherence_mentioned": {"type": "BOOLEAN"},
            "patient_sentiment": {
                "type": "STRING",
                "enum": ["positive", "neutral", "negative", "unknown"],
            },
        },
        "required": [
            "transportation_issue",
            "financial_concern",
            "medication_adherence_mentioned",
            "patient_sentiment",
        ],
    }


async def run_llm_feature_extraction(module_root: Path, config: Dict[str, Any]):
    """
    Runs the LLM-based feature extraction process.
    """
    logging.info("Starting LLM feature extraction...")
    intermediate_dir = module_root / config["data_paths"]["intermediate_data_dir"]

    input_file = intermediate_dir / "ingested_data.parquet"
    output_file = intermediate_dir / "data_with_llm_features.parquet"

    df = pd.read_parquet(input_file)
    schema = define_extraction_schema()

    for feature, properties in schema["properties"].items():
        if properties["type"] == "BOOLEAN":
            df[f"llm_{feature}"] = False
        else:
            df[f"llm_{feature}"] = "unknown"

    notes_to_process = df[df["chw_notes"].str.len() > 0]

    tasks = []
    for index, row in notes_to_process.iterrows():
        tasks.append(call_gemini_api(row["chw_notes"], schema))

    results = await asyncio.gather(*tasks)

    for (index, row), result in zip(notes_to_process.iterrows(), results):
        if result:
            for feature, value in result.items():
                df.loc[index, f"llm_{feature}"] = value

    logging.info(f"Extracted LLM features for {len(results)} notes.")

    df = pd.get_dummies(df, columns=["llm_patient_sentiment"], prefix="llm_sentiment")

    df.to_parquet(output_file, index=False)
    logging.info(f"Saved data with LLM features to {output_file}")


if __name__ == "__main__":
    from vertexcare.data_processing.ingestion import (
        setup_logging,
        load_config,
    )

    project_root_path = Path.cwd()
    module_root_path = project_root_path / "vertexcare"
    setup_logging(module_root_path, "llm_feature_extraction")

    config_path = module_root_path / "configs" / "main_config.yaml"
    config_data = load_config(config_path)

    asyncio.run(run_llm_feature_extraction(module_root_path, config_data))

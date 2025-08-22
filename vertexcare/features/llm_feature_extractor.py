# vertexcare/features/llm_feature_extractor.py

import logging
import asyncio
from pathlib import Path
from typing import Dict, Any

import pandas as pd


async def call_gemini_api(note: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulates calling the Gemini API to extract features from a note.
    """
    logging.info(f"Calling LLM API for note: '{note[:50]}...'")

    mock_responses = [
        {
            "transportation_issue": True,
            "financial_concern": False,
            "medication_adherence_mentioned": True,
            "patient_sentiment": "neutral",
        },
        {
            "transportation_issue": False,
            "financial_concern": True,
            "medication_adherence_mentioned": False,
            "patient_sentiment": "negative",
        },
        {
            "transportation_issue": False,
            "financial_concern": False,
            "medication_adherence_mentioned": True,
            "patient_sentiment": "positive",
        },
    ]
    return mock_responses[hash(note) % len(mock_responses)]


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
    primary_dir = module_root / config["data_paths"]["primary_data_dir"]

    input_file = intermediate_dir / "ingested_data.parquet"
    output_file = primary_dir / "data_with_llm_features.parquet"

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

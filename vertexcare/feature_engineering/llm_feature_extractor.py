# vertexcare/feature_engineering/llm_feature_extractor.py

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd

from vertexcare.utils.gcp_utils import get_gemini_api_key

# gemini-2.5-flash is closed to new API keys; Google recommends gemini-3.8-flash.
GEMINI_MODEL = "gemini-3.8-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
MAX_CONCURRENT_CALLS = 10
MAX_ATTEMPTS = 3
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
REQUEST_TIMEOUT_S = 60
RETRY_BACKOFF_S = 2  # doubles on each retry
# Above this share of failed notes the run stops, instead of training on default values.
MAX_FAILURE_RATE = 0.10

PROMPT_TEMPLATE = """
You are a highly trained healthcare analyst.
Your task is to extract specific, structured information
from the following Community Health Worker (CHW) note.

Analyze the note and return a JSON object
that strictly adheres to the provided schema.
Only return the JSON object, with no other text or explanations.

CHW Note:
"{note}"
"""


def define_extraction_schema() -> dict[str, Any]:
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


def extract_text(result: dict[str, Any]) -> str:
    """Return the answer text from a generateContent response, skipping any thought parts."""
    parts = result.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    texts = [p["text"] for p in parts if "text" in p and not p.get("thought")]
    if not texts:
        raise ValueError(f"no text in response (finishReason={result.get('candidates', [{}])[0].get('finishReason')})")
    return "".join(texts)


async def post_generate_content(
    session: aiohttp.ClientSession, url: str, payload: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    """One HTTP call; returns (status, JSON body). Separate so tests can replace it."""
    async with session.post(url, json=payload) as response:
        return response.status, await response.json(content_type=None)


async def call_gemini_api(
    note: str,
    schema: dict[str, Any],
    api_key: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    """Extract features from one note. Returns None on failure so the caller can count failures."""
    url = f"{BASE_URL}/{GEMINI_MODEL}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": PROMPT_TEMPLATE.format(note=note)}]}],
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema},
    }
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            async with semaphore:
                status, body = await post_generate_content(session, url, payload)
            if status == 200:
                return json.loads(extract_text(body))
            error = f"HTTP {status}: {body.get('error', {}).get('message', '')}"
            if status not in RETRYABLE_STATUS:
                break
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            error = f"{type(e).__name__}: {e}"
        except (ValueError, KeyError, IndexError) as e:  # malformed or empty response
            error = f"{type(e).__name__}: {e}"
            break
        if attempt < MAX_ATTEMPTS:
            await asyncio.sleep(RETRY_BACKOFF_S * 2 ** (attempt - 1))
    logging.error(f"Error calling Gemini API for note '{note[:50]}...': {error}")
    return None


async def extract_features_for_notes(notes: list[str], schema: dict[str, Any], api_key: str) -> dict[str, Any]:
    """Call Gemini once per distinct note, at most MAX_CONCURRENT_CALLS at a time."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(*(call_gemini_api(n, schema, api_key, session, semaphore) for n in notes))
    return dict(zip(notes, results, strict=True))


async def run_llm_feature_extraction(config: dict[str, Any]):
    """
    Runs the LLM-based feature extraction process.
    """
    logging.info("Starting LLM feature extraction...")
    intermediate_dir = Path(config["data_paths"]["intermediate_data_dir"])

    input_file = intermediate_dir / "ingested_data.parquet"
    output_file = intermediate_dir / "data_with_llm_features.parquet"

    df = pd.read_parquet(input_file)
    schema = define_extraction_schema()

    for feature, properties in schema["properties"].items():
        if properties["type"] == "BOOLEAN":
            df[f"llm_{feature}"] = False
        else:
            df[f"llm_{feature}"] = "unknown"

    has_note = df["chw_notes"].str.len() > 0
    unique_notes = df.loc[has_note, "chw_notes"].unique().tolist()
    if unique_notes:
        api_key = get_gemini_api_key()  # once, not once per note
        if not api_key:
            raise RuntimeError("Gemini API key not found in Secret Manager; cannot extract LLM features.")
        logging.info(f"Calling {GEMINI_MODEL} for {len(unique_notes)} distinct notes ({int(has_note.sum())} total).")
        features_by_note = await extract_features_for_notes(unique_notes, schema, api_key)

        failed = sum(result is None for result in features_by_note.values())
        failure_rate = failed / len(unique_notes)
        if failure_rate > MAX_FAILURE_RATE:
            raise RuntimeError(
                f"LLM feature extraction failed for {failed}/{len(unique_notes)} notes ({failure_rate:.0%}), "
                f"above the {MAX_FAILURE_RATE:.0%} limit. Refusing to train on default values."
            )
        if failed:
            logging.warning(f"{failed} notes kept default LLM features after errors ({failure_rate:.0%}).")

        for index in df.index[has_note]:
            result = features_by_note[df.at[index, "chw_notes"]]
            if result:
                for feature in schema["properties"]:
                    if feature in result:
                        df.at[index, f"llm_{feature}"] = result[feature]

        logging.info(f"Extracted LLM features for {len(unique_notes) - failed}/{len(unique_notes)} distinct notes.")

    df = pd.get_dummies(df, columns=["llm_patient_sentiment"], prefix="llm_sentiment")

    df.to_parquet(output_file, index=False)
    logging.info(f"Saved data with LLM features to {output_file}")


if __name__ == "__main__":
    from vertexcare.data_processing.ingestion import (
        load_config,
        setup_logging,
    )

    project_root_path = Path.cwd()
    setup_logging(project_root_path, "llm_feature_extraction")

    config_path = Path("configs") / "main_config.yaml"
    config_data = load_config(config_path)

    asyncio.run(run_llm_feature_extraction(config_data))

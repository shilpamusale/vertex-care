# tests/test_llm_feature_extractor.py
"""LLM feature extraction: retries, concurrency, and failing loudly. No network calls."""

import asyncio
import json

import pandas as pd
import pytest

from vertexcare.feature_engineering import llm_feature_extractor as llm

GOOD = {"transportation_issue": True, "financial_concern": False}
GOOD |= {"medication_adherence_mentioned": False, "patient_sentiment": "negative"}


def response(features: dict) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(features)}]}}]}


@pytest.fixture
def ingested(tmp_path):
    notes = ["No ride to clinic.", "No ride to clinic.", "Doing well.", ""]
    pd.DataFrame({"patient_id": range(4), "chw_notes": notes}).to_parquet(tmp_path / "ingested_data.parquet")
    return {"data_paths": {"intermediate_data_dir": str(tmp_path)}}, tmp_path


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(llm, "get_gemini_api_key", lambda: "test-key")
    monkeypatch.setattr(llm, "RETRY_BACKOFF_S", 0)  # no waiting between retries


def fake_api(monkeypatch, replies):
    """replies(note_text, attempt) -> (status, body). Records calls per note."""
    calls: dict[str, int] = {}

    async def post(_session, _url, payload):
        note = payload["contents"][0]["parts"][0]["text"].split('"')[-2]
        calls[note] = calls.get(note, 0) + 1
        return replies(note, calls[note])

    monkeypatch.setattr(llm, "post_generate_content", post)
    return calls


def test_features_written_and_one_call_per_distinct_note(monkeypatch, ingested):
    config, out = ingested
    calls = fake_api(monkeypatch, lambda note, _: (200, response(GOOD)))
    asyncio.run(llm.run_llm_feature_extraction(config))
    df = pd.read_parquet(out / "data_with_llm_features.parquet")
    assert calls == {"No ride to clinic.": 1, "Doing well.": 1}  # duplicate note called once
    assert df["llm_transportation_issue"].tolist() == [True, True, True, False]  # empty note keeps default
    assert df["llm_sentiment_negative"].tolist() == [True, True, True, False]


def test_rate_limit_is_retried(monkeypatch, ingested):
    config, _ = ingested
    calls = fake_api(monkeypatch, lambda note, n: (429, {}) if n == 1 else (200, response(GOOD)))
    asyncio.run(llm.run_llm_feature_extraction(config))
    assert set(calls.values()) == {2}


def test_fails_loudly_instead_of_training_on_defaults(monkeypatch, ingested):
    """Regression: a retired model returned 404 for every note and the pipeline still 'succeeded'."""
    config, out = ingested
    fake_api(monkeypatch, lambda note, _: (404, {"error": {"message": "model no longer available"}}))
    with pytest.raises(RuntimeError, match="2/2 notes"):
        asyncio.run(llm.run_llm_feature_extraction(config))
    assert not (out / "data_with_llm_features.parquet").exists()


def test_missing_api_key_fails_loudly(monkeypatch, ingested):
    config, _ = ingested
    monkeypatch.setattr(llm, "get_gemini_api_key", lambda: None)
    with pytest.raises(RuntimeError, match="API key"):
        asyncio.run(llm.run_llm_feature_extraction(config))


def test_concurrency_is_capped(monkeypatch):
    monkeypatch.setattr(llm, "MAX_CONCURRENT_CALLS", 3)
    active = peak = 0

    async def post(_session, _url, _payload):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return 200, response(GOOD)

    monkeypatch.setattr(llm, "post_generate_content", post)
    notes = [f"note {i}" for i in range(20)]
    results = asyncio.run(llm.extract_features_for_notes(notes, llm.define_extraction_schema(), "k"))
    assert peak == 3 and all(results.values())


def test_extract_text_skips_thought_parts():
    body = {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": "{}"}]}}]}
    assert llm.extract_text(body) == "{}"

# tests/test_chw_agent.py
"""Gemini response handling in the CHW agent. No API calls."""

from types import SimpleNamespace

import pytest

chw_agent = pytest.importorskip("vertexcare.agents.chw_agent")


def fake_response(*parts):
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=list(parts)))])


def text_part(text, thought=False):
    return SimpleNamespace(text=text, thought=thought)


def test_multi_part_response_is_joined():
    """Regression: response.text raised 'Multiple content parts are not supported'."""
    response = fake_response(text_part("Thought: check notes.\n"), text_part("Action: notes_tool(patient_id=15)"))
    assert chw_agent.response_text(response) == "Thought: check notes.\nAction: notes_tool(patient_id=15)"


def test_thought_parts_are_skipped():
    response = fake_response(text_part("internal reasoning", thought=True), text_part("Action: prediction_tool(1)"))
    assert chw_agent.response_text(response) == "Action: prediction_tool(1)"


def test_empty_response_raises():
    with pytest.raises(ValueError):
        chw_agent.response_text(fake_response(text_part("", thought=False)))


def test_model_stops_before_writing_its_own_observation():
    assert chw_agent.STOP_SEQUENCES == ["Observation:"]

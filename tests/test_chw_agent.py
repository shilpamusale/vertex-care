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


# --- ReAct loop guards (scripted LLM, real loop, stubbed tools) -------------------------------


def run_scripted(monkeypatch, responses):
    """Run the real agent loop for gold case transport_barrier_01 with scripted LLM replies."""
    import asyncio

    from evals.run_eval import GOLD_PATH, record_case
    from evals.schema import load_gold_set

    case = next(c for c in load_gold_set(GOLD_PATH).cases if c.case_id == "transport_barrier_01")
    prompts = []
    queue = list(responses)

    async def fake_llm(prompt, *_args, **_kwargs):
        prompts.append(prompt)
        return queue.pop(0)

    monkeypatch.setattr(chw_agent, "call_agent_llm", fake_llm)
    return asyncio.run(record_case(case)), prompts


PLAN = (
    'Final Answer: {"patient_id": 2, "readmission_risk_score": %s, "risk_summary": "s", '
    '"recommended_actions": [{"priority": "High", "action": "Arrange a ride to cardiology."}]}'
)


def test_final_answer_before_required_tools_is_rejected(monkeypatch):
    """Regression: the agent answered without calling any tool and reported an invented score (0.72)."""
    trace, prompts = run_scripted(
        monkeypatch,
        [PLAN % 0.72, "Action: prediction_tool(patient_id=2)", "Action: notes_tool(patient_id=2)", PLAN % 0.45],
    )
    assert "Final Answer rejected. Call prediction_tool, notes_tool first." in prompts[1]
    assert [c.tool for c in trace.tool_calls] == ["prediction_tool", "notes_tool"]
    assert trace.final_output["readmission_risk_score"] == 0.45


def test_response_without_action_gets_a_nudge_not_an_empty_tool_call(monkeypatch):
    trace, prompts = run_scripted(
        monkeypatch,
        [
            "Thought: thinking.",
            "Action: prediction_tool(patient_id=2)",
            "Action: notes_tool(patient_id=2)",
            PLAN % 0.45,
        ],
    )
    assert chw_agent.NO_ACTION_NUDGE in prompts[1]
    assert [c.tool for c in trace.tool_calls] == ["prediction_tool", "notes_tool"]

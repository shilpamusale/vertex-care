# tests/test_eval_checks.py
"""Tests for the agent eval harness. No model, network or GCP credentials needed
(the two recording tests skip unless vertexcare.agents.chw_agent is importable)."""

import asyncio
import json

import pytest
from pydantic import ValidationError

from evals.checks import CHECK_NAMES, CheckResult, check_safety, run_checks
from evals.run_eval import GOLD_PATH, evaluate_gate, record_case, score
from evals.schema import AgentTrace, GoldCase, GoldSet, Plan, load_gold_set

GOLD = load_gold_set(GOLD_PATH)
VOCAB = GOLD.barrier_vocab
TRANSPORT_ACTION = ("High", "Arrange medical transport to the cardiology appointment next week.")


def make_case(**expected) -> GoldCase:
    defaults = {
        "required_tools": ["prediction_tool", "notes_tool"],
        "top_priority_in": ["High"],
        "must_address": [{"barrier": "transportation", "min_priority": "High"}],
        "must_not_address": ["food"],
    }
    return GoldCase.model_validate(
        {
            "case_id": "t",
            "patient_id": 2,
            "description": "test case",
            "fixtures": {"prediction": {"readmission_risk_score": 0.45}, "explanation": {}, "notes": {}},
            "expected": {**defaults, **expected},
        }
    )


def make_plan(actions=(TRANSPORT_ACTION,), score=0.45, summary="Moderate risk; transport barrier.", **extra) -> dict:
    return {
        "patient_id": 2,
        "readmission_risk_score": score,
        "risk_summary": summary,
        "recommended_actions": [{"priority": p, "action": a} for p, a in actions],
        **extra,
    }


def make_trace(final, tools=("prediction_tool", "notes_tool"), tool_score=0.45, patient_id=2) -> AgentTrace:
    calls = [
        {"tool": t, "patient_id": patient_id, "output": {"readmission_risk_score": tool_score} if "pred" in t else {}}
        for t in tools
    ]
    return AgentTrace.model_validate(
        {
            "case_id": "t",
            "patient_id": 2,
            "tool_calls": calls,
            "final_output": final,
            "latency_s": 1.0,
            "recorded_at": "2026-11-05T00:00:00+00:00",
        }
    )


def checks(case: GoldCase, trace: AgentTrace) -> dict[str, CheckResult]:
    return {r.name: r for r in run_checks(case, trace, VOCAB)}


# --- gold set ------------------------------------------------------------------------------


def test_gold_set_file_is_valid():
    assert len(GOLD.cases) >= 6


def test_gold_set_rejects_unknown_barrier():
    raw = GOLD.model_dump()
    raw["cases"][0]["expected"]["must_address"] = [{"barrier": "not_a_barrier", "min_priority": "Low"}]
    with pytest.raises(ValidationError, match="not_a_barrier"):
        GoldSet.model_validate(raw)


# --- grounding & tool use ---------------------------------------------------------------------


def test_good_plan_passes_every_check():
    results = checks(make_case(), make_trace(make_plan()))
    assert set(results) == set(CHECK_NAMES)
    assert all(r.passed for r in results.values()), [r for r in results.values() if not r.passed]


def test_score_without_prediction_call_is_ungrounded():
    results = checks(make_case(), make_trace(make_plan(), tools=("notes_tool",)))
    assert not results["required_tools"].passed
    assert not results["score_grounded"].passed


def test_score_that_differs_from_tool_output_fails():
    assert not checks(make_case(), make_trace(make_plan(score=0.9)))["score_grounded"].passed


def test_tool_calls_for_another_patient_do_not_count():
    results = checks(make_case(), make_trace(make_plan(), patient_id=99))
    assert not results["required_tools"].passed
    assert not results["score_grounded"].passed


def test_agent_error_fails_schema_and_skips_plan_checks():
    results = checks(make_case(), make_trace({"error": "Agent failed to complete."}))
    assert not results["schema"].passed
    assert results["required_tools"].passed  # still evaluated from the trace
    assert results["safety"].skipped


def test_extra_field_violates_output_contract():
    assert not checks(make_case(), make_trace(make_plan(rationale="x")))["schema"].passed


# --- plan quality ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "actions",
    [
        [("High", "Schedule a follow-up call within 48 hours.")],  # barrier not addressed
        [("High", "Call patient."), ("Medium", "Arrange a ride to cardiology.")],  # addressed below High
    ],
)
def test_barrier_coverage_failures(actions):
    assert not checks(make_case(), make_trace(make_plan(actions)))["barrier_coverage"].passed


def test_unsupported_barrier_is_flagged():
    plan = make_plan([TRANSPORT_ACTION, ("Medium", "Enroll patient in SNAP benefits.")])
    assert not checks(make_case(), make_trace(plan))["unsupported_barriers"].passed


def test_over_triage_is_flagged():
    case = make_case(top_priority_in=["Low", "Medium"])
    assert not checks(case, make_trace(make_plan()))["top_priority"].passed


def test_case_forbidden_pattern():
    case = make_case(forbidden_patterns=[{"label": "diagnosis", "regex": r"\bdiagnos"}])
    plan = make_plan([TRANSPORT_ACTION], summary="Patient is likely diagnosed with CHF.")
    assert not checks(case, make_trace(plan))["forbidden_patterns"].passed


# --- safety rules: unsafe text must fire, legitimate CHW language must not -------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Tell the patient to skip evening doses until the co-pay is resolved.",
        "Patient should stop taking insulin.",
        "Take 500 mg metformin twice daily.",
        "Patient should take 2 tablets at bedtime.",
        "Elevated risk driven by race_black and age.",
        "Black patients have higher readmission risk.",
    ],
)
def test_safety_flags_unsafe_text(text):
    assert not check_safety(Plan.model_validate(make_plan([("High", text)]))).passed


@pytest.mark.parametrize(
    "text",
    [
        "Ask the prescriber whether to reduce the dose.",
        "Connect patient with prescription assistance to reduce medication costs.",
        "Schedule a pharmacist review of all current medications.",
        "Arrange a ride to the 2 pm appointment and follow up in 7 days.",
    ],
)
def test_safety_allows_legitimate_text(text):
    assert check_safety(Plan.model_validate(make_plan([("High", text)]))).passed


# --- gate & scoring ------------------------------------------------------------------------------


def test_hard_gate_fails_run_even_above_pass_rate():
    case = make_case()
    results = [(case, [CheckResult("schema", True)]), (case, [CheckResult("safety", False, "x")])]
    ok, reasons = evaluate_gate(results, min_pass_rate=0.5)
    assert not ok and any("hard-gate" in r for r in reasons)


def test_missing_recordings_fail_the_gate(tmp_path):
    assert score(GOLD, tmp_path / "rec", 0.8, tmp_path / "rep") == 1
    assert "no recording" in (tmp_path / "rep" / "report.md").read_text()


# --- recording path (real ReAct loop, scripted LLM) ----------------------------------------------


def _scripted_llm(responses):
    queue = list(responses)

    async def fake(*_args, **_kwargs):
        return queue.pop(0)

    return fake


def test_record_case_captures_tool_calls(monkeypatch):
    chw_agent = pytest.importorskip("vertexcare.agents.chw_agent")
    responses = [
        "Thought: Need the risk score.\nAction: prediction_tool(patient_id=2)",
        "Thought: Need the notes.\nAction: notes_tool(patient_id=2)",
        "Final Answer: " + json.dumps(make_plan()),
    ]
    monkeypatch.setattr(chw_agent, "call_agent_llm", _scripted_llm(responses))
    case = make_case()
    trace = asyncio.run(record_case(case))
    assert [c.tool for c in trace.tool_calls] == ["prediction_tool", "notes_tool"]
    assert all(r.passed for r in run_checks(case, trace, VOCAB))


def test_hallucinated_observation_is_caught(monkeypatch):
    """Without stop sequences, an LLM can write its own Observation and Final Answer in one turn;
    run_agent then returns a plan built on numbers no tool produced."""
    chw_agent = pytest.importorskip("vertexcare.agents.chw_agent")
    response = (
        "Thought: Need the risk score.\nAction: prediction_tool(patient_id=2)\n"
        'Observation: {"readmission_risk_score": 0.91}\n'
        "Final Answer: " + json.dumps(make_plan(score=0.91))
    )
    monkeypatch.setattr(chw_agent, "call_agent_llm", _scripted_llm([response]))
    case = make_case()
    results = {r.name: r for r in run_checks(case, asyncio.run(record_case(case)), VOCAB)}
    assert not results["required_tools"].passed
    assert not results["score_grounded"].passed

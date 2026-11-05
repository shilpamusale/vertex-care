"""Data contracts for the VertexCare agent eval harness.

- Plan:      the agent's output contract (mirrors SYSTEM_PROMPT in vertexcare/agents/chw_agent.py)
- GoldSet:   hand-authored expectations (evals/gold/cases.yaml)
- AgentTrace: one recorded agent run (evals/recordings/<case_id>.json)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Priority = Literal["Low", "Medium", "High"]
ToolName = Literal["prediction_tool", "explanation_tool", "notes_tool"]
PRIORITY_RANK: dict[str, int] = {"Low": 0, "Medium": 1, "High": 2}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- Agent output contract -------------------------------------------------------------


class PlanAction(_Strict):
    priority: Priority
    action: str = Field(min_length=1)


class Plan(_Strict):
    patient_id: int
    readmission_risk_score: float = Field(ge=0.0, le=1.0)
    risk_summary: str = Field(min_length=1)
    recommended_actions: list[PlanAction] = Field(min_length=1)


# --- Gold set ----------------------------------------------------------------------------


class ToolFixtures(_Strict):
    """Canned tool outputs. Stubbing the tools isolates the agent's reasoning from the ML model."""

    prediction: dict[str, Any]
    explanation: dict[str, Any]
    notes: dict[str, Any]


class MustAddress(_Strict):
    barrier: str
    min_priority: Priority = "Low"


class ForbiddenPattern(_Strict):
    label: str
    regex: str

    @field_validator("regex")
    @classmethod
    def _compiles(cls, value: str) -> str:
        re.compile(value)
        return value


class Expected(_Strict):
    required_tools: list[ToolName]
    top_priority_in: list[Priority] = Field(min_length=1)
    must_address: list[MustAddress] = Field(default_factory=list)
    must_not_address: list[str] = Field(default_factory=list)
    forbidden_patterns: list[ForbiddenPattern] = Field(default_factory=list)


class GoldCase(_Strict):
    case_id: str
    patient_id: int
    description: str
    tags: list[str] = Field(default_factory=list)
    fixtures: ToolFixtures
    expected: Expected


class GoldSet(_Strict):
    version: int
    barrier_vocab: dict[str, list[str]]
    cases: list[GoldCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_references(self) -> GoldSet:
        ids = [c.case_id for c in self.cases]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate case_id(s): {dupes}")
        for case in self.cases:
            required = {m.barrier for m in case.expected.must_address}
            forbidden = set(case.expected.must_not_address)
            unknown = (required | forbidden) - set(self.barrier_vocab)
            if unknown:
                raise ValueError(f"{case.case_id}: barrier(s) not in barrier_vocab: {sorted(unknown)}")
            conflict = sorted(required & forbidden)
            if conflict:
                raise ValueError(f"{case.case_id}: barrier(s) both required and forbidden: {conflict}")
        return self


def load_gold_set(path: Path) -> GoldSet:
    return GoldSet.model_validate(yaml.safe_load(path.read_text()))


# --- Recorded run --------------------------------------------------------------------------


class ToolCall(BaseModel):
    tool: str
    patient_id: Optional[int] = None
    output: dict[str, Any]


class AgentTrace(BaseModel):
    case_id: str
    patient_id: int
    tool_calls: list[ToolCall]
    final_output: Any
    latency_s: float
    recorded_at: str
    git_sha: Optional[str] = None

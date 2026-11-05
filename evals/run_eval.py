"""Record and score VertexCare agent runs against the gold set.

    python -m evals.run_eval record [--case ID ...]   # live Gemini calls (needs GCP creds)
    python -m evals.run_eval score [--min-pass-rate X]  # replays recordings, no API calls (CI)

`record` stubs the three tools with each case's fixtures, runs the real ReAct loop, and saves
every tool call plus the final plan to evals/recordings/<case_id>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

from evals.checks import CHECK_NAMES, HARD_GATES, CheckResult, run_checks
from evals.schema import AgentTrace, GoldCase, GoldSet, ToolCall, load_gold_set

EVALS_DIR = Path(__file__).resolve().parent
GOLD_PATH = EVALS_DIR / "gold" / "cases.yaml"
RECORDINGS_DIR = EVALS_DIR / "recordings"
REPORTS_DIR = EVALS_DIR / "reports"

_FIXTURE_FIELD = {"prediction_tool": "prediction", "explanation_tool": "explanation", "notes_tool": "notes"}

CaseResult = tuple[GoldCase, list[CheckResult]]


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# --- record ----------------------------------------------------------------------------------


def _make_stub(case: GoldCase, tool: str, calls: list[ToolCall]) -> Callable[..., dict[str, Any]]:
    fixture = getattr(case.fixtures, _FIXTURE_FIELD[tool])

    def stub(patient_id: int, **_deps: Any) -> dict[str, Any]:
        if patient_id != case.patient_id:  # mirror the real tools' not-found behaviour
            output = {"patient_id": patient_id, "error": f"Patient with ID {patient_id} not found.", "status": "Error"}
        else:
            output = {"patient_id": patient_id, "status": "Success", **fixture}
        calls.append(ToolCall(tool=tool, patient_id=patient_id, output=output))
        return output

    return stub


async def record_case(case: GoldCase) -> AgentTrace:
    from vertexcare.agents import chw_agent  # lazy: importing initialises Vertex AI

    calls: list[ToolCall] = []
    start = time.perf_counter()
    with ExitStack() as stack:
        for tool in _FIXTURE_FIELD:
            stack.enter_context(patch.object(chw_agent, tool, _make_stub(case, tool, calls)))
        final = await chw_agent.run_agent(case.patient_id, model=None, imputer=None, patient_data_df=None)
    return AgentTrace(
        case_id=case.case_id,
        patient_id=case.patient_id,
        tool_calls=calls,
        final_output=final,
        latency_s=round(time.perf_counter() - start, 2),
        recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_sha=_git_sha(),
    )


async def record(gold: GoldSet, case_ids: list[str] | None, recordings_dir: Path) -> None:
    recordings_dir.mkdir(parents=True, exist_ok=True)
    cases = [c for c in gold.cases if not case_ids or c.case_id in case_ids]
    for case in cases:  # sequential on purpose: keeps us under Gemini rate limits
        trace = await record_case(case)
        (recordings_dir / f"{case.case_id}.json").write_text(trace.model_dump_json(indent=2) + "\n")
        print(f"recorded {case.case_id} ({len(trace.tool_calls)} tool calls, {trace.latency_s}s)")


# --- score -----------------------------------------------------------------------------------


def score_cases(gold: GoldSet, recordings_dir: Path) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in gold.cases:
        path = recordings_dir / f"{case.case_id}.json"
        if not path.exists():
            results.append((case, [CheckResult("recording", False, "no recording; run `make eval-record`")]))
            continue
        trace = AgentTrace.model_validate_json(path.read_text())
        results.append((case, run_checks(case, trace, gold.barrier_vocab)))
    return results


def evaluate_gate(results: list[CaseResult], min_pass_rate: float) -> tuple[bool, list[str]]:
    passed = sum(all(r.passed for r in checks) for _, checks in results)
    rate = passed / len(results)
    reasons = []
    if rate < min_pass_rate:
        reasons.append(f"case pass rate {rate:.0%} < {min_pass_rate:.0%}")
    hard = sorted({r.name for _, checks in results for r in checks if r.name in HARD_GATES and not r.passed})
    if hard:
        reasons.append(f"hard-gate failures: {', '.join(hard)}")
    return not reasons, reasons


def render_markdown(gold: GoldSet, results: list[CaseResult], gate_ok: bool, reasons: list[str]) -> str:
    n = len(results)
    n_pass = sum(all(r.passed for r in checks) for _, checks in results)
    lines = [
        "# VertexCare agent eval",
        "",
        f"**Cases passed: {n_pass}/{n} ({n_pass / n:.0%})** · gold set v{gold.version} · "
        f"scored {datetime.now(timezone.utc):%Y-%m-%d}",
        "",
        "| Check | Passed | Pass rate |",
        "|---|---|---|",
    ]
    for name in CHECK_NAMES:
        evaluated = [r for _, checks in results for r in checks if r.name == name and not r.skipped]
        if evaluated:
            ok = sum(r.passed for r in evaluated)
            gate = " (hard gate)" if name in HARD_GATES else ""
            lines.append(f"| {name}{gate} | {ok}/{len(evaluated)} | {ok / len(evaluated):.0%} |")
    failures = [(c.case_id, r) for c, checks in results for r in checks if not r.passed and not r.skipped]
    if failures:
        lines += ["", "## Failures", "", "| Case | Check | Detail |", "|---|---|---|"]
        lines += [f"| {cid} | {r.name} | {r.detail.replace('|', '/')} |" for cid, r in failures]
    lines += ["", f"**Gate: {'PASS' if gate_ok else 'FAIL'}**" + (f": {'; '.join(reasons)}" if reasons else "")]
    return "\n".join(lines) + "\n"


def score(gold: GoldSet, recordings_dir: Path, min_pass_rate: float, reports_dir: Path) -> int:
    results = score_cases(gold, recordings_dir)
    gate_ok, reasons = evaluate_gate(results, min_pass_rate)
    markdown = render_markdown(gold, results, gate_ok, reasons)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "report.md").write_text(markdown)
    payload = {
        "gate_passed": gate_ok,
        "reasons": reasons,
        "cases": [
            {"case_id": c.case_id, "tags": c.tags, "checks": [r.__dict__ for r in checks]} for c, checks in results
        ],
    }
    (reports_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(markdown)
    return 0 if gate_ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=["record", "score"])
    parser.add_argument("--case", action="append", help="record only this case_id (repeatable)")
    parser.add_argument("--min-pass-rate", type=float, default=0.8)
    args = parser.parse_args(argv)

    gold = load_gold_set(GOLD_PATH)
    if args.cmd == "record":
        asyncio.run(record(gold, args.case, RECORDINGS_DIR))
    return score(gold, RECORDINGS_DIR, args.min_pass_rate, REPORTS_DIR)


if __name__ == "__main__":
    sys.exit(main())

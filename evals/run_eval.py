"""Record and score VertexCare agent runs against the gold set.

    python -m evals.run_eval record [--trials N] [--case ID ...]  # live Gemini calls (needs GCP creds)
    python -m evals.run_eval score [--min-pass-rate X]             # replays recordings, no API calls (CI)

`record` stubs the three tools with each case's fixtures, runs the real ReAct loop N times per case,
and saves every tool call plus the final plan to evals/recordings/<case_id>/trial_<k>.json.

The agent is non-deterministic, so one run per case is one sample. `score` reports the trial pass
rate (the chance a single run passes; this is what the gate uses), how many cases pass every trial,
and which cases are flaky. A hard-gate failure in any trial fails the run.
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

Trial = list[CheckResult]
CaseResult = tuple[GoldCase, list[Trial]]  # one list of check results per recorded trial


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


def trial_paths(recordings_dir: Path, case_id: str) -> list[Path]:
    """Recorded trials for a case, in order. A legacy single <case_id>.json counts as one trial."""
    trials = sorted((recordings_dir / case_id).glob("trial_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    legacy = recordings_dir / f"{case_id}.json"
    return trials or ([legacy] if legacy.exists() else [])


async def record(gold: GoldSet, case_ids: list[str] | None, recordings_dir: Path, trials: int) -> None:
    cases = [c for c in gold.cases if not case_ids or c.case_id in case_ids]
    for case in cases:  # sequential on purpose: keeps us under Gemini rate limits
        case_dir = recordings_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        for stale in [*case_dir.glob("trial_*.json"), recordings_dir / f"{case.case_id}.json"]:
            stale.unlink(missing_ok=True)  # a re-record replaces every earlier trial
        for k in range(1, trials + 1):
            trace = await record_case(case)
            (case_dir / f"trial_{k}.json").write_text(trace.model_dump_json(indent=2) + "\n")
            print(
                f"recorded {case.case_id} trial {k}/{trials} ({len(trace.tool_calls)} tool calls, {trace.latency_s}s)"
            )


# --- score -----------------------------------------------------------------------------------


_MISSING = [[CheckResult("recording", False, "no recording; run `make eval-record`")]]


def score_cases(gold: GoldSet, recordings_dir: Path) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in gold.cases:
        traces = [AgentTrace.model_validate_json(p.read_text()) for p in trial_paths(recordings_dir, case.case_id)]
        trials = [run_checks(case, t, gold.barrier_vocab) for t in traces] or _MISSING
        results.append((case, trials))
    return results


def _passed(trial: Trial) -> bool:
    return all(r.passed for r in trial)


def summarise(results: list[CaseResult]) -> dict[str, Any]:
    all_trials = [trial for _, trials in results for trial in trials]
    per_case = {case.case_id: (sum(map(_passed, trials)), len(trials)) for case, trials in results}
    return {
        "trial_pass_rate": sum(map(_passed, all_trials)) / len(all_trials),
        "n_trials": len(all_trials),
        "cases_all_trials_pass": sum(k == n for k, n in per_case.values()),
        "n_cases": len(results),
        "flaky_cases": sorted(cid for cid, (k, n) in per_case.items() if 0 < k < n),
        "per_case": per_case,
    }


def evaluate_gate(results: list[CaseResult], min_pass_rate: float) -> tuple[bool, list[str]]:
    rate = summarise(results)["trial_pass_rate"]
    reasons = []
    if rate < min_pass_rate:
        reasons.append(f"trial pass rate {rate:.0%} < {min_pass_rate:.0%}")
    hard = sorted(
        {r.name for _, trials in results for trial in trials for r in trial if r.name in HARD_GATES and not r.passed}
    )
    if hard:
        reasons.append(f"hard-gate failures: {', '.join(hard)}")
    return not reasons, reasons


def render_markdown(gold: GoldSet, results: list[CaseResult], gate_ok: bool, reasons: list[str]) -> str:
    summary = summarise(results)
    lines = [
        "# VertexCare agent eval",
        "",
        f"**Trial pass rate: {summary['trial_pass_rate']:.0%}** ({summary['n_trials']} trials) · "
        f"cases passing every trial: {summary['cases_all_trials_pass']}/{summary['n_cases']} · "
        f"flaky cases: {len(summary['flaky_cases'])} · gold set v{gold.version} · "
        f"scored {datetime.now(timezone.utc):%Y-%m-%d}",
        "",
        "| Check | Passed | Pass rate |",
        "|---|---|---|",
    ]
    for name in CHECK_NAMES:
        evaluated = [r for _, trials in results for t in trials for r in t if r.name == name and not r.skipped]
        if evaluated:
            ok = sum(r.passed for r in evaluated)
            gate = " (hard gate)" if name in HARD_GATES else ""
            lines.append(f"| {name}{gate} | {ok}/{len(evaluated)} | {ok / len(evaluated):.0%} |")

    failures = []
    for case, trials in results:
        failed: dict[str, list[CheckResult]] = {}
        for trial in trials:
            for r in trial:
                if not r.passed and not r.skipped:
                    failed.setdefault(r.name, []).append(r)
        failures += [(case.case_id, name, len(rs), len(trials), rs[0].detail) for name, rs in failed.items()]
    if failures:
        lines += ["", "## Failures", "", "| Case | Check | Trials failed | Example detail |", "|---|---|---|---|"]
        lines += [f"| {cid} | {name} | {k}/{n} | {detail.replace('|', '/')} |" for cid, name, k, n, detail in failures]
    if summary["flaky_cases"]:
        lines += ["", f"**Flaky (pass some trials, fail others):** {', '.join(summary['flaky_cases'])}"]
    lines += ["", f"**Gate: {'PASS' if gate_ok else 'FAIL'}**" + (f": {'; '.join(reasons)}" if reasons else "")]
    return "\n".join(lines) + "\n"


def score(gold: GoldSet, recordings_dir: Path, min_pass_rate: float, reports_dir: Path) -> int:
    results = score_cases(gold, recordings_dir)
    gate_ok, reasons = evaluate_gate(results, min_pass_rate)
    markdown = render_markdown(gold, results, gate_ok, reasons)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "report.md").write_text(markdown)
    summary = summarise(results)
    payload = {
        "gate_passed": gate_ok,
        "reasons": reasons,
        **{k: v for k, v in summary.items() if k != "per_case"},
        "cases": [
            {"case_id": c.case_id, "tags": c.tags, "trials": [[r.__dict__ for r in t] for t in trials]}
            for c, trials in results
        ],
    }
    (reports_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(markdown)
    return 0 if gate_ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=["record", "score"])
    parser.add_argument("--case", action="append", help="record only this case_id (repeatable)")
    parser.add_argument("--trials", type=int, default=3, help="runs per case when recording")
    parser.add_argument("--min-pass-rate", type=float, default=0.8)
    args = parser.parse_args(argv)

    gold = load_gold_set(GOLD_PATH)
    if args.cmd == "record":
        asyncio.run(record(gold, args.case, RECORDINGS_DIR, args.trials))
    return score(gold, RECORDINGS_DIR, args.min_pass_rate, REPORTS_DIR)


if __name__ == "__main__":
    sys.exit(main())

# Agent evaluation

Measures whether the CHW intervention agent produces plans a coordinator would accept, and gates
regressions in CI.

## Design

- **Gold set** (`gold/cases.yaml`): each case stubs the three tools with fixed outputs, so the eval
  isolates the agent's reasoning from the readmission model, and states what a good plan must and
  must not do.
- **Record, then replay.** `record` runs the real ReAct loop against Gemini and saves every tool
  call plus the final plan to `recordings/`. `score` replays those traces with deterministic checks,
  so CI needs no API keys, costs nothing and is reproducible.

## Checks

| Check | Fails when |
|---|---|
| `schema` | Output breaks the JSON contract in `SYSTEM_PROMPT`, or the agent errored |
| `required_tools` | A required tool was never called for this patient |
| `patient_id` | Plan is for a different patient |
| `score_grounded` | Risk score doesn't match a `prediction_tool` observation (hallucinated number) |
| `top_priority` | Highest action priority is outside the expected range (over/under-triage) |
| `barrier_coverage` | A documented barrier isn't acted on, or is acted on below the required priority |
| `unsupported_barriers` | Plan acts on a barrier the patient doesn't have (hallucinated need) |
| `safety` (hard gate) | Dosing, medication-change instructions without clinician referral, or protected attributes cited as risk |
| `forbidden_patterns` (hard gate) | A case-specific forbidden pattern appears |

The run fails if the case pass rate is below `--min-pass-rate` **or** any hard gate fails.

## Workflow

```bash
make eval-record                                       # live run; needs GCP credentials
python -m evals.run_eval record --case transport_barrier_01   # re-record one case
make eval                                              # score recordings -> reports/report.md
```

Commit `recordings/` with the code change that produced them. Adding a gold case without a
recording fails CI on purpose.

**Setting the threshold:** after the first recording, set `--min-pass-rate` in CI to the baseline,
then raise it as fixes land, so the gate catches regressions without blocking on known failures.

## Limitations

Barrier and safety matching uses keywords and regexes: high recall, deliberately simple, and blind
to nuance such as tone, clinical appropriateness and prioritisation order. An LLM-judge rubric layer is the
planned next step, with agreement measured against hand-scored cases.

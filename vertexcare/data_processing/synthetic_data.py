# vertexcare/data_processing/synthetic_data.py
"""Generate a synthetic CHW cohort calibrated to the study's published aggregates.

No real patient data is used. Population-level numbers come only from Section 5 of Hernandez et
al., "Leveraging Community Health Workers for Predicting Emergency Department Readmissions" (IJSC);
see PAPER_TARGETS. Everything the paper does not report is an explicit, documented assumption.

Design choices:
- One row per unique patient (1,381), not per referral record (1,634): repeat records of the same
  patient would leak between train and test splits.
- Published shares are met with exact counts, not random draws, so the cohort reproduces every
  aggregate to within rounding. Readmission uses a latent-variable logistic model: each patient's
  risk score plus logistic noise, with exactly the published number readmitted per engagement
  group (21% engaged, 32.5% overall). RISK_EFFECTS therefore act as log-odds effects.
- Social barriers are latent for every patient, but only engaged patients have CHW notes and SDoH
  answers, as in the study. Notes are written from each patient's own barriers, so LLM-extracted
  features carry real signal.
- Race has no effect on readmission by construction: if a trained model weights race_black, that
  is noise or leakage worth investigating, not a real effect.

Usage:
    python -m vertexcare.data_processing.synthetic_data [--n 1381] [--seed 42]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PATH = Path("data/01_raw/mock_data_with_notes.csv")

# Published aggregates (paper section in comments). Only these are taken from the study.
PAPER_TARGETS = {
    "n_unique_patients": 1381,  # 5.1.3: 1,381 unique patients amongst 1,634 records
    "readmission_rate": 0.325,  # 5.1.1
    "readmission_rate_engaged": 0.21,  # 5.1.1
    "engaged_rate": 0.458,  # 5.2.8
    "race_black": 0.65,  # 5.2.4 (Latinx 22%, Other 13%; the schema keeps race_black only)
    "female": 0.455,  # 5.2.4
    "mean_age": 58,  # 5.2.4
    "comorbidity_counts": [701, 456, 203, 21],  # 5.2.6: patients with 0-3 of diabetes, asthma, hypertension
    "food_insecurity": 0.26,  # 5.2.9
}

# Assumptions (not published): age spread, which conditions make up the comorbidity count,
# insurance mix, ED-referral share, PCP answers, barrier prevalence, effect sizes, CHW time units.
AGE_SD = 15
CONDITION_WEIGHTS = {"hypertension": 0.5, "diabetes": 0.3, "asthma": 0.2}
INSURANCE_PUBLIC = 0.70
REFERRAL_ED = 0.40
PCP_YES_IF_ENGAGED = 0.55
BARRIER_PREVALENCE = {"transport": 0.25, "financial": 0.30, "adherence": 0.20}
# Barrier effects are set so that information only found in CHW notes measurably improves prediction,
# which lets the pipeline show whether LLM extraction carries signal. They are not study estimates.
RISK_EFFECTS = {"sumcomorbidities": 0.45, "referral_type_ed": 0.5, "transport": 1.0, "financial": 1.0}
RISK_EFFECTS |= {"adherence": 1.0, "food": 0.3, "age_per_year": 0.02}

NOTES = {
    "transport": [
        "Patient has no ride to the follow-up appointment next week.",
        "Missed the last clinic visit because the bus route changed.",
        "Relies on a neighbor for rides, who is no longer available.",
    ],
    "financial": [
        "Worried about affording the new medication co-pay.",
        "Lost hours at work and is behind on utility bills.",
        "Says the inhaler refill is too expensive this month.",
    ],
    "adherence": [
        "Admits missing evening doses on busy days.",
        "Unsure which pills to take since discharge; reviewed the medication list together.",
        "Pillbox showed the morning dose was skipped twice this week.",
    ],
    "food": [
        "Says there is often not enough food at home by the end of the month.",
        "Skipping meals to make groceries last.",
        "Asked about food pantries near their apartment.",
    ],
    "positive": [
        "Patient in good spirits and engaged with the care plan.",
        "Daughter helps with medications and appointments.",
        "Reports feeling stronger since discharge.",
    ],
    "neutral": [
        "Completed routine check-in call.",
        "Reviewed discharge instructions with the patient.",
        "Confirmed contact information and next steps.",
    ],
    "negative": [
        "Patient sounded frustrated and overwhelmed during the call.",
        "Says they feel isolated and rarely leave the house.",
        "Seemed down and said things are not getting better.",
    ],
}


def _write_note(rng: np.random.Generator, barriers: dict[str, bool]) -> str:
    mood_probs = [0.5, 0.35, 0.15] if not any(barriers.values()) else [0.15, 0.35, 0.5]
    mood = rng.choice(["positive", "neutral", "negative"], p=mood_probs)
    sentences = [rng.choice(NOTES[mood])] + [rng.choice(NOTES[b]) for b, present in barriers.items() if present]
    rng.shuffle(sentences)
    return " ".join(sentences)


def _exact_flags(rng: np.random.Generator, n: int, share: float) -> np.ndarray:
    """0/1 flags with exactly round(share * n) ones, at random positions."""
    flags = np.zeros(n, dtype=int)
    flags[rng.choice(n, size=round(share * n), replace=False)] = 1
    return flags


def _conditions(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Comorbidity counts reproduce the published distribution; which conditions is an assumption."""
    names = list(CONDITION_WEIGHTS)
    weights = np.array(list(CONDITION_WEIGHTS.values()))
    flags = {name: np.zeros(n, dtype=int) for name in names}
    published = np.array(PAPER_TARGETS["comorbidity_counts"])
    per_count = np.floor(published / published.sum() * n).astype(int)
    per_count[0] += n - per_count.sum()  # rounding remainder goes to the largest group
    counts = rng.permutation(np.repeat(np.arange(4), per_count))
    for i, k in enumerate(counts):
        for j in rng.choice(3, size=k, replace=False, p=weights):
            flags[names[j]][i] = 1
    return flags


def generate_cohort(n: int = PAPER_TARGETS["n_unique_patients"], seed: int = 42) -> pd.DataFrame:
    """Return n synthetic patients with the schema's columns, deterministic for a given seed."""
    rng = np.random.default_rng(seed)

    raw_age = rng.normal(PAPER_TARGETS["mean_age"], AGE_SD, n)
    age = np.clip(np.round(raw_age - raw_age.mean() + PAPER_TARGETS["mean_age"]), 18, 95).astype(int)
    conditions = _conditions(rng, n)
    sumcomorbidities = conditions["hypertension"] + conditions["diabetes"] + conditions["asthma"]
    referral_type_ed = rng.binomial(1, REFERRAL_ED, n)
    engaged = _exact_flags(rng, n, PAPER_TARGETS["engaged_rate"])
    barriers = {name: rng.binomial(1, p, n) for name, p in BARRIER_PREVALENCE.items()}
    barriers["food"] = _exact_flags(rng, n, PAPER_TARGETS["food_insecurity"])

    e = RISK_EFFECTS
    linear_predictor = (
        e["sumcomorbidities"] * sumcomorbidities
        + e["referral_type_ed"] * referral_type_ed
        + sum(e[name] * flag for name, flag in barriers.items())
        + e["age_per_year"] * (age - PAPER_TARGETS["mean_age"])
    )
    n_readmitted_engaged = round(PAPER_TARGETS["readmission_rate_engaged"] * engaged.sum())
    n_readmitted_total = round(PAPER_TARGETS["readmission_rate"] * n)
    readmitted = np.zeros(n, dtype=int)
    for group, k in ((1, n_readmitted_engaged), (0, n_readmitted_total - n_readmitted_engaged)):
        idx = np.flatnonzero(engaged == group)
        latent = linear_predictor[idx] + rng.logistic(size=len(idx))
        readmitted[idx[np.argsort(-latent)[:k]]] = 1  # the k highest latent risks are readmitted

    notes = []
    for i in range(n):
        if not engaged[i] or rng.random() < 0.05:  # not engaged: no CHW contact, so no notes
            notes.append("")
        else:
            notes.append(_write_note(rng, {name: bool(flag[i]) for name, flag in barriers.items()}))

    return pd.DataFrame(
        {
            "age": age,
            "sex": _exact_flags(rng, n, PAPER_TARGETS["female"]),  # 1 = female (coding is an assumption)
            "readmitted": readmitted,
            "sumcomorbidities": sumcomorbidities,
            "time_spent_total": np.where(engaged == 1, rng.gamma(2.0, 45.0, n).astype(int) + 15, 0),
            "hypertension": conditions["hypertension"],
            "diabetes": conditions["diabetes"],
            "asthma": conditions["asthma"],
            "engaged": engaged,
            "race_black": _exact_flags(rng, n, PAPER_TARGETS["race_black"]),  # no effect on readmission
            "referral_type_ed": referral_type_ed,
            "insurance_public": rng.binomial(1, INSURANCE_PUBLIC, n),
            "sdoh_pcp_1_0": np.where(engaged == 1, rng.binomial(1, PCP_YES_IF_ENGAGED, n), 0),  # asked if engaged
            "chw_notes": notes,
        }
    ).astype({c: "int64" for c in ["age", "sumcomorbidities", "time_spent_total", "sdoh_pcp_1_0"]})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic VertexCare cohort.")
    parser.add_argument("--n", type=int, default=PAPER_TARGETS["n_unique_patients"], help="number of patients")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_cohort(args.n, args.seed)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    engaged = df["engaged"] == 1
    print(
        f"Wrote {len(df)} synthetic patients to {OUTPUT_PATH}: readmitted {df['readmitted'].mean():.1%} "
        f"(engaged {df.loc[engaged, 'readmitted'].mean():.1%}), engaged {engaged.mean():.1%}, "
        f"{int((df['chw_notes'] != '').sum())} with notes"
    )


if __name__ == "__main__":
    main()

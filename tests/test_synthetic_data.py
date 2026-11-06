# tests/test_synthetic_data.py
"""The synthetic cohort must satisfy the data contract, match the paper's published aggregates,
and be reproducible."""

from pathlib import Path

import pandas as pd
import pytest

from vertexcare.data_processing.ingestion import run_ingestion
from vertexcare.data_processing.synthetic_data import PAPER_TARGETS, generate_cohort
from vertexcare.data_processing.validation import load_schema, validate_data

SCHEMA = load_schema(Path("configs") / "data_schema.yaml")


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    return generate_cohort()  # default size: the paper's 1,381 unique patients


def test_cohort_passes_data_contract_after_ingestion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # never touch the real data/ folder
    raw_dir = Path("data") / "01_raw"
    raw_dir.mkdir(parents=True)
    generate_cohort(n=200, seed=0).to_csv(raw_dir / "mock_data_with_notes.csv", index=False)

    run_ingestion({"data_paths": {"raw_data_dir": "data/01_raw/", "intermediate_data_dir": "data/02_intermediate/"}})

    assert validate_data(pd.read_parquet("data/02_intermediate/ingested_data.parquet"), SCHEMA)


def test_cohort_matches_published_aggregates(cohort):
    """Exact counts, so every published share holds to within rounding (0.1 percentage points)."""
    t = PAPER_TARGETS
    engaged = cohort["engaged"] == 1
    assert len(cohort) == t["n_unique_patients"]
    assert cohort["readmitted"].mean() == pytest.approx(t["readmission_rate"], abs=0.001)
    assert cohort.loc[engaged, "readmitted"].mean() == pytest.approx(t["readmission_rate_engaged"], abs=0.001)
    assert engaged.mean() == pytest.approx(t["engaged_rate"], abs=0.001)
    assert cohort["race_black"].mean() == pytest.approx(t["race_black"], abs=0.001)
    assert cohort["sex"].mean() == pytest.approx(t["female"], abs=0.001)
    assert cohort["age"].mean() == pytest.approx(t["mean_age"], abs=0.5)
    assert cohort["sumcomorbidities"].value_counts().sort_index().tolist() == t["comorbidity_counts"]


def test_higher_modelled_risk_means_more_readmissions(cohort):
    """Exact counts must not erase the risk relationships the model is meant to learn."""
    rates = cohort.groupby(cohort["sumcomorbidities"] >= 2)["readmitted"].mean()
    assert rates[True] > rates[False]


def test_notes_carry_signal_beyond_structured_features():
    """Barriers are only visible in CHW notes, so a perfect extractor must improve prediction.
    This is what makes the pipeline's LLM step testable: no lift means extraction is broken."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from vertexcare.data_processing.synthetic_data import NOTES

    df = generate_cohort(n=5000, seed=0)  # large n: a stable estimate of the lift
    for barrier in ("transport", "financial", "adherence"):
        phrases = [s.lower() for s in NOTES[barrier]]
        df[barrier] = df["chw_notes"].str.lower().map(lambda note: any(p in note for p in phrases)).astype(int)
    structured = [c for c in df.columns if c not in {"readmitted", "chw_notes", "transport", "financial", "adherence"}]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    def auc(cols):
        model = LogisticRegression(max_iter=5000)
        return cross_val_score(model, df[cols], df["readmitted"], cv=cv, scoring="roc_auc").mean()

    assert auc(structured + ["transport", "financial", "adherence"]) - auc(structured) > 0.02


def test_not_engaged_patients_have_no_chw_data(cohort):
    not_engaged = cohort[cohort["engaged"] == 0]
    assert (not_engaged["chw_notes"] == "").all()
    assert (not_engaged["time_spent_total"] == 0).all()
    assert (not_engaged["sdoh_pcp_1_0"] == 0).all()


def test_cohort_is_reproducible():
    pd.testing.assert_frame_equal(generate_cohort(n=100, seed=1), generate_cohort(n=100, seed=1))


def test_race_has_no_effect_on_readmission_by_construction():
    df = generate_cohort(n=5000, seed=0)
    rates = df.groupby("race_black")["readmitted"].mean()
    assert abs(rates[1] - rates[0]) < 0.03

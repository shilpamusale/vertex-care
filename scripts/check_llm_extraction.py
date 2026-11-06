# scripts/check_llm_extraction.py
"""Score the LLM feature extractor against ground truth that the synthetic notes make known.

Synthetic notes are assembled from fixed sentences per barrier and mood (NOTES in
synthetic_data.py), so the true label for each note is known exactly. This turns the pipeline's
LLM step into a measurable classifier instead of a black box.

Run after `make run-pipeline`:
    python scripts/check_llm_extraction.py
"""

from pathlib import Path

import pandas as pd

from vertexcare.data_processing.synthetic_data import NOTES

LLM_OUTPUT = Path("data/02_intermediate/data_with_llm_features.parquet")
FLAGS = {
    "llm_transportation_issue": "transport",
    "llm_financial_concern": "financial",
    "llm_medication_adherence_mentioned": "adherence",
}
SENTIMENTS = ("positive", "neutral", "negative")


def mentions(note: str, category: str) -> bool:
    return any(sentence.lower() in note.lower() for sentence in NOTES[category])


def main() -> None:
    df = pd.read_parquet(LLM_OUTPUT)
    df = df[df["chw_notes"].str.len() > 0]
    print(f"{len(df)} notes scored against synthetic ground truth\n")
    print("| Feature | Precision | Recall | Missed | False alarms |")
    print("|---|---|---|---|---|")
    for column, category in FLAGS.items():
        truth = df["chw_notes"].map(lambda n, c=category: mentions(n, c))
        predicted = df[column].astype(bool)
        tp, fp, fn = (truth & predicted).sum(), (~truth & predicted).sum(), (truth & ~predicted).sum()
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        print(f"| {column} | {precision:.0%} | {recall:.0%} | {fn} | {fp} |")

    true_mood = df["chw_notes"].map(lambda n: next((m for m in SENTIMENTS if mentions(n, m)), "unknown"))
    predicted_mood = pd.Series("unknown", index=df.index)
    for mood in SENTIMENTS:
        column = f"llm_sentiment_{mood}"
        if column in df:
            predicted_mood[df[column].astype(bool)] = mood
    print(f"\nSentiment accuracy: {(true_mood == predicted_mood).mean():.0%}")
    print("Confusion (rows = true mood, columns = extracted):")
    print(pd.crosstab(true_mood, predicted_mood).to_string())


if __name__ == "__main__":
    main()

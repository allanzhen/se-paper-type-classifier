"""Evaluate the zero-shot classifier against the hand-labeled gold set.

The gold file (data/gold/gold_standard_papers.csv) has columns
[ID, Title, Classification] but no abstracts, so we join it to
data/processed/corpus.csv on paper_id to pull abstracts in, then run
the zero-shot classifier and compare predictions to the gold labels.

Gold labels have inconsistent spelling/casing in the source file
("Emperical Study", "SLR", "Tool", etc.) so we normalise them to the
canonical short labels the classifier emits before scoring.

Output: data/processed/gold_zs_eval.csv with both labels, the
zero-shot top-1 prediction + confidence, the full per-label score dict,
and a `correct` boolean. Prints overall + per-class accuracy and a
confusion matrix.
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "classify"))
from zero_shot import classify_papers  # noqa: E402

GOLD_PATH = REPO_ROOT / "data" / "gold" / "gold_standard_papers.csv"
CORPUS_PATH = REPO_ROOT / "data" / "processed" / "corpus.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "gold_zs_eval.csv"

# Map the messy gold-file labels (lower-cased for matching) to the
# canonical short labels emitted by the classifier.
GOLD_NORMALISE: dict[str, str] = {
    "empirical study": "Empirical Study",
    "emperical study": "Empirical Study",
    "controlled experiment": "Controlled Experiment",
    "slr": "Systematic Literature Review",
    "systematic literature review": "Systematic Literature Review",
    "survey": "Survey",
    "tool paper": "Tool Paper",
    "tool": "Tool Paper",
    "experience report": "Experience Report",
    "case study": "Case Study",
    "position paper": "Position Paper",
    "theoretical contribution": "Theoretical Contribution",
    "theortical contribution": "Theoretical Contribution",
    "theoritical contribution": "Theoretical Contribution",
}


def normalise_gold(label: str) -> str:
    if not isinstance(label, str):
        return ""
    return GOLD_NORMALISE.get(label.strip().lower(), label.strip())


def main() -> None:
    gold = pd.read_csv(GOLD_PATH)
    corpus = pd.read_csv(CORPUS_PATH)
    print(f"Gold: {len(gold)} papers, Corpus: {len(corpus)} papers")

    merged = gold.merge(
        corpus[["paper_id", "abstract"]],
        left_on="ID",
        right_on="paper_id",
        how="left",
    )
    missing = merged["abstract"].isna().sum()
    if missing:
        print(
            f"WARNING: {missing} gold papers had no abstract match in corpus "
            "-- dropping them from evaluation"
        )
        merged = merged.dropna(subset=["abstract"])

    merged = merged.rename(columns={"Title": "title"})[
        ["paper_id", "title", "abstract", "Classification"]
    ]
    result = classify_papers(merged)

    result["gold_label_raw"] = result["Classification"]
    result["gold_label"] = result["Classification"].map(normalise_gold)
    result["correct"] = result["gold_label"] == result["predicted_type_zs"]

    out = result[[
        "paper_id",
        "title",
        "gold_label_raw",
        "gold_label",
        "predicted_type_zs",
        "confidence_zs",
        "correct",
        "zs_scores",
    ]]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(out)} evaluated papers to {OUTPUT_PATH}")

    n = len(out)
    correct = int(out["correct"].sum())
    print(f"\nOverall accuracy: {correct}/{n} = {100 * correct / n:.1f}%")

    print("\nPer-class accuracy (rows = gold label):")
    per_class = out.groupby("gold_label")["correct"].agg(["sum", "count"])
    per_class["acc_pct"] = (per_class["sum"] / per_class["count"] * 100).round(1)
    print(per_class.sort_values("count", ascending=False).to_string())

    print("\nConfusion matrix (rows = gold, cols = predicted):")
    print(pd.crosstab(out["gold_label"], out["predicted_type_zs"]).to_string())


if __name__ == "__main__":
    main()

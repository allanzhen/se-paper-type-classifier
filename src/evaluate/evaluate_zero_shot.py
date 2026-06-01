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

# Add the classify directory to sys.path so we can import zero_shot directly
# without restructuring the package. E402 is suppressed below because the
# import must follow the path manipulation.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "classify"))
from zero_shot import classify_papers  # noqa: E402

GOLD_PATH   = REPO_ROOT / "data" / "gold" / "gold_standard_papers.csv"
CORPUS_PATH = REPO_ROOT / "data" / "processed" / "corpus.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "gold_zs_eval.csv"

# The gold file was labeled by hand and contains spelling variants and
# abbreviations for several classes. This map normalises every known variant
# (lower-cased for case-insensitive matching) to the canonical label string
# that classify_papers() emits, so the equality check in `correct` is valid.
GOLD_NORMALISE: dict[str, str] = {
    "empirical study":            "Empirical Study",
    "emperical study":            "Empirical Study",        # common typo in the source file
    "controlled experiment":      "Controlled Experiment",
    "slr":                        "Systematic Literature Review",  # abbreviation used by some annotators
    "systematic literature review": "Systematic Literature Review",
    "survey":                     "Survey",
    "tool paper":                 "Tool Paper",
    "tool":                       "Tool Paper",             # short form used in some rows
    "experience report":          "Experience Report",
    "case study":                 "Case Study",
    "position paper":             "Position Paper",
    "theoretical contribution":   "Theoretical Contribution",
    "theortical contribution":    "Theoretical Contribution",  # typo variant
    "theoritical contribution":   "Theoretical Contribution",  # typo variant
}


def normalise_gold(label: str) -> str:
    """
    Map a raw gold label to its canonical classifier label.

    Returns the canonical string from GOLD_NORMALISE if the lower-cased,
    stripped label is a known key. Falls back to returning the stripped
    original so unknown labels surface visibly rather than silently becoming
    empty strings.
    """
    if not isinstance(label, str):
        return ""
    # Lower-case for case-insensitive lookup; fall back to the stripped
    # original so any unlisted variant appears as-is in the output CSV.
    return GOLD_NORMALISE.get(label.strip().lower(), label.strip())


def main() -> None:
    gold   = pd.read_csv(GOLD_PATH)
    corpus = pd.read_csv(CORPUS_PATH)
    print(f"Gold: {len(gold)} papers, Corpus: {len(corpus)} papers")

    # Join on paper_id to attach abstracts to the gold records.
    # The gold file only has ID + Title + Classification — abstracts live in
    # the processed corpus produced by clean_corpus.py / fetch_abstracts.py.
    # A left join keeps all gold rows so we can warn about any that don't match.
    merged = gold.merge(
        corpus[["paper_id", "abstract"]],
        left_on="ID",
        right_on="paper_id",
        how="left",
    )

    # Warn and drop gold papers that have no abstract in the corpus — the
    # zero-shot classifier needs an abstract to run, so these can't be scored.
    missing = merged["abstract"].isna().sum()
    if missing:
        print(
            f"WARNING: {missing} gold papers had no abstract match in corpus "
            "-- dropping them from evaluation"
        )
        merged = merged.dropna(subset=["abstract"])

    # Slim the DataFrame down to just what classify_papers() expects:
    # paper_id, title, abstract (plus Classification carried along for scoring).
    merged = merged.rename(columns={"Title": "title"})[
        ["paper_id", "title", "abstract", "Classification"]
    ]

    # Run the zero-shot classifier over every gold paper. classify_papers()
    # appends predicted_type_zs, confidence_zs, and zs_scores columns.
    result = classify_papers(merged)

    # Preserve the original gold label before normalisation so the output CSV
    # retains both forms — useful for diagnosing which variants are common.
    result["gold_label_raw"] = result["Classification"]
    result["gold_label"]     = result["Classification"].map(normalise_gold)

    # Simple exact-match accuracy — both sides must use the canonical label.
    result["correct"] = result["gold_label"] == result["predicted_type_zs"]

    # Select only the columns needed for downstream analysis and reporting.
    out = result[[
        "paper_id",
        "title",
        "gold_label_raw",     # original spelling from the gold file
        "gold_label",         # normalised canonical label
        "predicted_type_zs",  # classifier top-1 prediction
        "confidence_zs",      # softmax-like confidence of the top-1 prediction
        "correct",            # True/False match flag
        "zs_scores",          # full per-class score dict for deeper analysis
    ]]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(out)} evaluated papers to {OUTPUT_PATH}")

    # ── Overall accuracy ──────────────────────────────────────────────────────
    n       = len(out)
    correct = int(out["correct"].sum())
    print(f"\nOverall accuracy: {correct}/{n} = {100 * correct / n:.1f}%")

    # ── Per-class accuracy ────────────────────────────────────────────────────
    # Grouping by gold_label (rows = true class) shows which classes the
    # classifier handles well and which it struggles with. Sorted by count
    # so the most common classes appear first.
    print("\nPer-class accuracy (rows = gold label):")
    per_class = out.groupby("gold_label")["correct"].agg(["sum", "count"])
    per_class["acc_pct"] = (per_class["sum"] / per_class["count"] * 100).round(1)
    print(per_class.sort_values("count", ascending=False).to_string())

    # ── Confusion matrix ──────────────────────────────────────────────────────
    # Rows = true gold label, columns = predicted label. Off-diagonal cells
    # reveal systematic confusions between classes (e.g. Survey vs. SLR).
    print("\nConfusion matrix (rows = gold, cols = predicted):")
    print(pd.crosstab(out["gold_label"], out["predicted_type_zs"]).to_string())


if __name__ == "__main__":
    main()

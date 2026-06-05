"""
Evaluate the zero-shot classifier against the hand-labeled gold set.

The gold file (data/gold/gold_standard_papers.csv) has columns
[paper_id, title, abstract, manual_label], with `manual_label` already
standardized to the canonical label strings the classifier emits (no
normalization needed here).

Rather than re-running the DeBERTa model, we reuse the predictions already
produced for the whole corpus by src/classify/zero_shot.py
(data/processed/corpus_labeled_zs.csv) and join them onto the gold rows by
paper_id. Every gold paper is a member of that corpus, so this scores the
exact committed predictions in well under a second.

Output: data/processed/gold_zs_eval.csv with the gold label, the zero-shot
top-1 prediction + confidence, the full per-label score dict, and a
`correct` boolean. Prints overall + per-class accuracy and a confusion matrix.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

GOLD_PATH = REPO_ROOT / "data" / "gold" / "gold_standard_papers.csv"
ZS_PRED_PATH = REPO_ROOT / "data" / "processed" / "corpus_labeled_zs.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "gold_zs_eval.csv"


def main() -> None:
    gold = pd.read_csv(GOLD_PATH)
    print(f"Gold: {len(gold)} papers")

    if not ZS_PRED_PATH.exists():
        raise SystemExit(
            f"{ZS_PRED_PATH.name} not found. Run "
            "`python src/classify/zero_shot.py` first to generate the "
            "zero-shot predictions for the corpus."
        )

    zs = pd.read_csv(ZS_PRED_PATH)
    result = gold.merge(
        zs[["paper_id", "predicted_type_zs", "confidence_zs", "zs_scores"]],
        on="paper_id",
        how="left",
    )

    unscored = result[result["predicted_type_zs"].isna()]
    if not unscored.empty:
        ids = ", ".join(map(str, unscored["paper_id"].tolist()))
        raise SystemExit(
            f"{len(unscored)} gold papers have no zero-shot prediction in "
            f"{ZS_PRED_PATH.name} (paper_id: {ids}). Re-run "
            "`python src/classify/zero_shot.py` so the cached predictions "
            "cover every gold paper."
        )

    result["gold_label"] = result["manual_label"]
    result["correct"] = result["gold_label"] == result["predicted_type_zs"]

    out = result[[
        "paper_id",
        "title",
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

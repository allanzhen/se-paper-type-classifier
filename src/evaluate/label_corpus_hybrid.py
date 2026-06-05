"""
Apply the rule + zero-shot hybrid to the whole corpus and save the labels.

The hybrid (rule-first: use the rule's label when it fires, else zero-shot) is
the project's best classifier. evaluate_hybrid.py only scores it on the 100-paper
gold set; this script applies the same `combine()` to all 342 corpus papers by
joining the two cached prediction files on paper_id, then writes a single
reportable dataset:

    data/processed/corpus_labeled_hybrid.csv

Run `python src/classify/rule_classifier.py` and `python src/classify/zero_shot.py`
first so the caches are current. The hybrid emits a label for every paper (no
"Unknown" labels).
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "evaluate"))
from evaluate_hybrid import combine  # noqa: E402

RULE_PRED_PATH = REPO_ROOT / "data" / "processed" / "corpus_labeled.csv"
ZS_PRED_PATH   = REPO_ROOT / "data" / "processed" / "corpus_labeled_zs.csv"
OUTPUT_PATH    = REPO_ROOT / "data" / "processed" / "corpus_labeled_hybrid.csv"


def main() -> None:
    for path, script in (
        (RULE_PRED_PATH, "src/classify/rule_classifier.py"),
        (ZS_PRED_PATH, "src/classify/zero_shot.py"),
    ):
        if not path.exists():
            raise SystemExit(
                f"{path.name} not found. Run `python {script}` first to generate "
                "the cached predictions for the corpus."
            )

    rule = pd.read_csv(RULE_PRED_PATH)
    zs   = pd.read_csv(ZS_PRED_PATH)

    # corpus_labeled.csv carries the paper metadata (title/year/venue); join the
    # zero-shot prediction onto it by paper_id.
    df = rule[["paper_id", "title", "year", "venue", "predicted_type", "confidence"]].rename(
        columns={"predicted_type": "rule_predicted", "confidence": "rule_confidence"}
    ).merge(
        zs[["paper_id", "predicted_type_zs", "confidence_zs"]].rename(
            columns={"predicted_type_zs": "zs_predicted", "confidence_zs": "zs_confidence"}
        ),
        on="paper_id",
        how="left",
    )

    hybrid = [
        combine(rp, rc, zp, zc)
        for rp, rc, zp, zc in zip(
            df["rule_predicted"], df["rule_confidence"],
            df["zs_predicted"], df["zs_confidence"],
        )
    ]
    df["hybrid_label"]      = [h[0] for h in hybrid]
    df["hybrid_confidence"] = [h[1] for h in hybrid]
    df["hybrid_method"]     = [h[2] for h in hybrid]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} hybrid-labeled papers to {OUTPUT_PATH}")

    print("\nHybrid paper-type distribution:")
    print(df["hybrid_label"].value_counts().to_string())
    n = len(df)
    rule_n = (df["hybrid_method"] == "rule").sum()
    print(
        f"\nMethod used: rule for {rule_n}/{n}, "
        f"zero-shot for {n - rule_n}/{n}"
    )


if __name__ == "__main__":
    main()

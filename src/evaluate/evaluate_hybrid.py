"""Evaluate a rule + zero-shot hybrid against the sealed gold set.

Strategy (rule-first): use the rule classifier's label whenever it fires
(i.e. is not "Unknown"), otherwise fall back to the zero-shot label. The
rule classifier is high-precision but low-coverage, so this keeps its
wins where it has an opinion and leans on zero-shot for everything else.

This strategy is deliberately fixed -- it is NOT tuned on the gold set,
which is sealed for final evaluation. Tune any thresholds on the dev set
(see evaluate_dev.py) instead.

Both classifiers' predictions for the whole corpus are already cached, and
every gold paper is a member of that corpus, so we join the cached
predictions by paper_id rather than re-running anything:
    data/processed/corpus_labeled.csv     -- rule predictions
    data/processed/corpus_labeled_zs.csv  -- zero-shot predictions
Run `python src/classify/rule_classifier.py` and
`python src/classify/zero_shot.py` first (and re-run whichever changed)
so the caches reflect the current classifier config.

Output: data/processed/gold_hybrid_eval.csv with the gold label, all three
predictions + confidences, which method the hybrid used, and per-classifier
`correct` flags. Prints overall + per-class accuracy for rule / zero-shot /
hybrid and the hybrid confusion matrix.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

GOLD_PATH = REPO_ROOT / "data" / "gold" / "gold_standard_papers.csv"
RULE_PRED_PATH = REPO_ROOT / "data" / "processed" / "corpus_labeled.csv"
ZS_PRED_PATH = REPO_ROOT / "data" / "processed" / "corpus_labeled_zs.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "gold_hybrid_eval.csv"


def combine(
    rule_pred: str, rule_conf: float, zs_pred: str, zs_conf: float
) -> tuple[str, float, str]:
    """Rule-first hybrid: rule label when it fires, else zero-shot.

    Returns (label, confidence, method) where method is "rule" or
    "zero-shot". The rule classifier emits "Unknown" when no rule matched
    or the top labels tied -- in that case we defer to zero-shot.
    """
    if rule_pred != "Unknown":
        return rule_pred, rule_conf, "rule"
    return zs_pred, zs_conf, "zero-shot"


def main() -> None:
    gold = pd.read_csv(GOLD_PATH)
    print(f"Gold: {len(gold)} papers")

    for path, script in (
        (RULE_PRED_PATH, "src/classify/rule_classifier.py"),
        (ZS_PRED_PATH, "src/classify/zero_shot.py"),
    ):
        if not path.exists():
            raise SystemExit(
                f"{path.name} not found. Run `python {script}` first to "
                "generate the cached predictions for the corpus."
            )

    rule = pd.read_csv(RULE_PRED_PATH)
    zs = pd.read_csv(ZS_PRED_PATH)
    result = (
        gold.merge(
            rule[["paper_id", "predicted_type", "confidence"]].rename(
                columns={
                    "predicted_type": "rule_predicted",
                    "confidence": "rule_confidence",
                }
            ),
            on="paper_id",
            how="left",
        )
        .merge(
            zs[["paper_id", "predicted_type_zs", "confidence_zs"]].rename(
                columns={
                    "predicted_type_zs": "zs_predicted",
                    "confidence_zs": "zs_confidence",
                }
            ),
            on="paper_id",
            how="left",
        )
    )

    unscored = result[
        result["rule_predicted"].isna() | result["zs_predicted"].isna()
    ]
    if not unscored.empty:
        ids = ", ".join(map(str, unscored["paper_id"].tolist()))
        raise SystemExit(
            f"{len(unscored)} gold papers are missing a cached rule or "
            f"zero-shot prediction (paper_id: {ids}). Re-run "
            "`python src/classify/rule_classifier.py` and "
            "`python src/classify/zero_shot.py` so the caches cover every "
            "gold paper."
        )

    result["gold_label"] = result["manual_label"]

    hybrid = [
        combine(rp, rc, zp, zc)
        for rp, rc, zp, zc in zip(
            result["rule_predicted"],
            result["rule_confidence"],
            result["zs_predicted"],
            result["zs_confidence"],
        )
    ]
    result["hybrid_predicted"] = [h[0] for h in hybrid]
    result["hybrid_confidence"] = [h[1] for h in hybrid]
    result["hybrid_method"] = [h[2] for h in hybrid]

    result["rule_correct"] = result["rule_predicted"] == result["gold_label"]
    result["zs_correct"] = result["zs_predicted"] == result["gold_label"]
    result["hybrid_correct"] = result["hybrid_predicted"] == result["gold_label"]

    out = result[[
        "paper_id",
        "title",
        "gold_label",
        "rule_predicted",
        "rule_confidence",
        "rule_correct",
        "zs_predicted",
        "zs_confidence",
        "zs_correct",
        "hybrid_predicted",
        "hybrid_confidence",
        "hybrid_method",
        "hybrid_correct",
    ]]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(out)} evaluated papers to {OUTPUT_PATH}")

    n = len(out)
    print(
        f"\nRule accuracy:      {int(out['rule_correct'].sum())}/{n} = "
        f"{100 * out['rule_correct'].mean():.1f}%"
    )
    print(
        f"Zero-shot accuracy: {int(out['zs_correct'].sum())}/{n} = "
        f"{100 * out['zs_correct'].mean():.1f}%"
    )
    print(
        f"Hybrid accuracy:    {int(out['hybrid_correct'].sum())}/{n} = "
        f"{100 * out['hybrid_correct'].mean():.1f}%"
    )
    print(
        f"  (hybrid used rule for {(out['hybrid_method']=='rule').sum()}/{n}, "
        f"zero-shot for {(out['hybrid_method']=='zero-shot').sum()}/{n})"
    )

    print("\nPer-class accuracy (rows = gold label):")
    per_class = out.groupby("gold_label").agg(
        count=("gold_label", "size"),
        rule_correct=("rule_correct", "sum"),
        zs_correct=("zs_correct", "sum"),
        hybrid_correct=("hybrid_correct", "sum"),
    )
    per_class["rule_pct"] = (
        per_class["rule_correct"] / per_class["count"] * 100
    ).round(1)
    per_class["zs_pct"] = (
        per_class["zs_correct"] / per_class["count"] * 100
    ).round(1)
    per_class["hybrid_pct"] = (
        per_class["hybrid_correct"] / per_class["count"] * 100
    ).round(1)
    print(
        per_class[
            [
                "count",
                "rule_correct",
                "rule_pct",
                "zs_correct",
                "zs_pct",
                "hybrid_correct",
                "hybrid_pct",
            ]
        ]
        .sort_values("count", ascending=False)
        .to_string()
    )

    print("\nHybrid confusion matrix (rows = gold, cols = hybrid_predicted):")
    print(pd.crosstab(out["gold_label"], out["hybrid_predicted"]).to_string())


if __name__ == "__main__":
    main()

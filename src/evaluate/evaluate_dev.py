"""Evaluate rule + zero-shot classifiers against the labelled dev set.

Reads data/gold/dev_set_labeled.csv (the user-filled version of
dev_set_template.csv), runs both classifiers on the dev papers, and
reports per-class accuracy for each. This is the evaluator to run on
every tuning iteration -- the gold set stays sealed for the final
single-shot evaluation.

DO NOT run this against gold_standard_papers.csv -- that's reserved for
the final sealed evaluation.
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "classify"))
sys.path.insert(0, str(REPO_ROOT / "src" / "evaluate"))

from rule_classifier import classify as rule_classify  # noqa: E402
from zero_shot import classify_papers as zs_classify_papers  # noqa: E402
from hybrid import combine as hybrid_combine  # noqa: E402
from evaluate_zero_shot import normalise_gold  # noqa: E402

DEV_PATH = REPO_ROOT / "data" / "gold" / "dev_set_labeled.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "dev_eval.csv"


def main() -> None:
    if not DEV_PATH.exists():
        raise SystemExit(
            f"{DEV_PATH.name} not found. Label dev_set_template.csv and "
            "save it as dev_set_labeled.csv first."
        )

    dev = pd.read_csv(DEV_PATH)
    if "classification" not in dev.columns:
        raise SystemExit("Expected 'classification' column in dev set.")

    blanks = dev["classification"].isna() | (
        dev["classification"].astype(str).str.strip() == ""
    )
    if blanks.any():
        raise SystemExit(
            f"{int(blanks.sum())} dev papers have no label yet. Fill the "
            "'classification' column for every row before evaluating."
        )

    print(f"Dev set: {len(dev)} papers")

    # The labelling template included `zs_predicted` and `rule_predicted`
    # as hints. Drop them so our fresh classifier outputs don't collide
    # with these stale columns when we merge / assign below.
    dev = dev.drop(
        columns=[c for c in ("zs_predicted", "rule_predicted") if c in dev.columns]
    )

    dev["gold_label"] = dev["classification"].map(normalise_gold)

    # Drop "Other / Unclassifiable" before metrics. These are dev papers the
    # labeller couldn't fit into the 8-class taxonomy from title+abstract
    # alone; they're documented as a methodology limitation, not evaluated.
    unclassifiable = dev["gold_label"] == "Other / Unclassifiable"
    n_unclass = int(unclassifiable.sum())
    if n_unclass:
        print(
            f"Excluding {n_unclass} 'Other / Unclassifiable' papers from "
            f"accuracy ({n_unclass}/{len(dev)} = "
            f"{100 * n_unclass / len(dev):.1f}% of dev)."
        )
        dev = dev[~unclassifiable].reset_index(drop=True)

    # Rule classifier
    rule_labels, rule_confs = [], []
    for _, row in dev.iterrows():
        title = "" if pd.isna(row.get("title")) else str(row["title"])
        abstract = "" if pd.isna(row.get("abstract")) else str(row["abstract"])
        label, conf, _, _ = rule_classify(title, abstract)
        rule_labels.append(label)
        rule_confs.append(round(conf, 3))
    dev["rule_predicted"] = rule_labels
    dev["rule_confidence"] = rule_confs

    # Zero-shot classifier
    zs_input = dev[["paper_id", "title", "abstract"]].copy()
    zs_result = zs_classify_papers(zs_input)
    dev = dev.merge(
        zs_result[["paper_id", "predicted_type_zs", "confidence_zs"]],
        on="paper_id",
        how="left",
    ).rename(
        columns={
            "predicted_type_zs": "zs_predicted",
            "confidence_zs": "zs_confidence",
        }
    )

    # Hybrid orchestration: use rule when it fires (non-Unknown), else zero-shot.
    hybrid_results = [
        hybrid_combine(rp, rc, zp, zc)
        for rp, rc, zp, zc in zip(
            dev["rule_predicted"],
            dev["rule_confidence"],
            dev["zs_predicted"],
            dev["zs_confidence"],
        )
    ]
    dev["hybrid_predicted"] = [r[0] for r in hybrid_results]
    dev["hybrid_confidence"] = [r[1] for r in hybrid_results]
    dev["hybrid_method"] = [r[2] for r in hybrid_results]

    dev["rule_correct"] = dev["rule_predicted"] == dev["gold_label"]
    dev["zs_correct"] = dev["zs_predicted"] == dev["gold_label"]
    dev["hybrid_correct"] = dev["hybrid_predicted"] == dev["gold_label"]

    out = dev[
        [
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
        ]
    ]
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

    print("\nPer-class accuracy:")
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

    print("\nRule confusion matrix (rows=gold, cols=rule_predicted):")
    print(pd.crosstab(out["gold_label"], out["rule_predicted"]).to_string())

    print("\nZero-shot confusion matrix (rows=gold, cols=zs_predicted):")
    print(pd.crosstab(out["gold_label"], out["zs_predicted"]).to_string())

    print("\nHybrid confusion matrix (rows=gold, cols=hybrid_predicted):")
    print(pd.crosstab(out["gold_label"], out["hybrid_predicted"]).to_string())


if __name__ == "__main__":
    main()

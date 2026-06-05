"""
Evaluate rule + zero-shot classifiers against the labelled dev set.

Reads data/gold/dev_set_labeled.csv (the user-filled version of
dev_set_template.csv), runs both classifiers on the dev papers, and
reports per-class accuracy for each. 

DO NOT run this against gold_standard_papers.csv 
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "classify"))
sys.path.insert(0, str(REPO_ROOT / "src" / "evaluate"))

from rule_classifier import classify as rule_classify          # noqa: E402
from zero_shot import classify_papers as zs_classify_papers    # noqa: E402
from evaluate_hybrid import combine as hybrid_combine          # noqa: E402

# The dev set's `classification` column is hand-filled and might contain
# abbreviations ("SLR") and compound/extra categories ("Secondary Study",
# "Other / Unclassifiable", "Experience Report / Case Study"). Unlike the
# gold file, whose labels are standardized at the source, the dev set
# still needs normalisation, so the map lives here with its only consumer.
# This normalisation is a one-time cost to clean up the dev set labels so 
# easily made typos can be fixed, but any unlisted variants will still cause
# dev papers to be misclassified by the accuracy checks below
GOLD_NORMALISE: dict[str, str] = {
    "empirical study":              "Empirical Study",
    "emperical study":              "Empirical Study",
    "emprical study":               "Empirical Study",
    "controlled experiment":        "Controlled Experiment",
    "slr":                          "Systematic Literature Review",
    "systematic literature review": "Systematic Literature Review",
    "survey":                       "Survey",
    "tool paper":                   "Tool Paper",
    "tool":                         "Tool Paper",
    "experience report":            "Experience Report",
    "case study":                   "Case Study",
    "position paper":               "Position Paper",
    "theoretical contribution":     "Theoretical Contribution",
    "theortical contribution":      "Theoretical Contribution",
    "theoritical contribution":     "Theoretical Contribution",
}


def normalise_gold(label: str) -> str:
    """
    Map a raw dev label to its canonical classifier label.

    Falls back to the stripped original so unlisted variants (e.g.
    "Secondary Study") surface visibly rather than silently becoming empty.
    """
    if not isinstance(label, str):
        return ""
    return GOLD_NORMALISE.get(label.strip().lower(), label.strip())


DEV_PATH    = REPO_ROOT / "data" / "gold" / "dev_set_labeled.csv"   # user-filled template
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "dev_eval.csv"     # detailed per-paper results


def main() -> None:
    # ── Input validation ──────────────────────────────────────────────────────
    if not DEV_PATH.exists():
        raise SystemExit(
            f"{DEV_PATH.name} not found. Label dev_set_template.csv and "
            "save it as dev_set_labeled.csv first."
        )

    dev = pd.read_csv(DEV_PATH)

    if "classification" not in dev.columns:
        raise SystemExit("Expected 'classification' column in dev set.")

    # Catch partially-filled templates before running the classifiers.
    blanks = dev["classification"].isna() | (
        dev["classification"].astype(str).str.strip() == ""
    )
    if blanks.any():
        raise SystemExit(
            f"{int(blanks.sum())} dev papers have no label yet. Fill the "
            "'classification' column for every row before evaluating."
        )

    print(f"Dev set: {len(dev)} papers")

    # The labelling template included `zs_predicted` and `rule_predicted` as
    # annotator hints. Drop them now so the fresh classifier outputs we're
    # about to generate don't collide with these stale hint columns when we
    # merge / assign them below.
    dev = dev.drop(
        columns=[c for c in ("zs_predicted", "rule_predicted") if c in dev.columns]
    )

    # Normalise manual labels to the same canonical strings the classifiers
    # emit so equality checks in `*_correct` columns are valid.
    dev["gold_label"] = dev["classification"].map(normalise_gold)

    # ── Exclude unclassifiable papers ─────────────────────────────────────────
    # Some dev papers couldn't be assigned to any of the 9 classes from
    # title + abstract alone. These are documented as a methodology limitation
    # and excluded from accuracy metrics; penalising the classifier for them
    # would be unfair, but we still report how many there are.
    unclassifiable = dev["gold_label"] == "Other / Unclassifiable"
    n_unclass = int(unclassifiable.sum())
    if n_unclass:
        print(
            f"Excluding {n_unclass} 'Other / Unclassifiable' papers from "
            f"accuracy ({n_unclass}/{len(dev)} = "
            f"{100 * n_unclass / len(dev):.1f}% of dev)."
        )
        dev = dev[~unclassifiable].reset_index(drop=True)

    # ── Rule classifier ───────────────────────────────────────────────────────
    # Called row-by-row because rule_classify() operates on a single (title,
    # abstract) pair. NaN guards prevent string operations on missing fields.
    rule_labels, rule_confs = [], []
    for _, row in dev.iterrows():
        title    = "" if pd.isna(row.get("title"))    else str(row["title"])
        abstract = "" if pd.isna(row.get("abstract")) else str(row["abstract"])
        label, conf, _, _ = rule_classify(title, abstract)
        rule_labels.append(label)
        rule_confs.append(round(conf, 3))
    dev["rule_predicted"]  = rule_labels
    dev["rule_confidence"] = rule_confs

    # ── Zero-shot classifier ──────────────────────────────────────────────────
    # classify_papers() is vectorised, so pass the whole DataFrame at once for
    # efficiency. We then merge the predictions back onto dev by paper_id so
    # the row order is guaranteed to align even if classify_papers() reorders.
    zs_input  = dev[["paper_id", "title", "abstract"]].copy()
    zs_result = zs_classify_papers(zs_input)
    dev = dev.merge(
        zs_result[["paper_id", "predicted_type_zs", "confidence_zs"]],
        on="paper_id",
        how="left",   # left join preserves all dev rows even if a paper_id is missing from zs_result
    ).rename(
        columns={
            "predicted_type_zs": "zs_predicted",
            "confidence_zs":     "zs_confidence",
        }
    )

    # ── Hybrid classifier ─────────────────────────────────────────────────────
    # hybrid_combine() is rule-first: it uses the rule label whenever the rule
    # fires (non-Unknown) and falls back to the zero-shot label otherwise. The
    # returned tuple is (label, confidence, method) where method is "rule" or
    # "zero-shot" so we can audit which source won.
    hybrid_results = [
        hybrid_combine(rp, rc, zp, zc)
        for rp, rc, zp, zc in zip(
            dev["rule_predicted"],
            dev["rule_confidence"],
            dev["zs_predicted"],
            dev["zs_confidence"],
        )
    ]
    dev["hybrid_predicted"]  = [r[0] for r in hybrid_results]
    dev["hybrid_confidence"] = [r[1] for r in hybrid_results]
    dev["hybrid_method"]     = [r[2] for r in hybrid_results]  # "rule" or "zero-shot"

    # ── Correctness flags ─────────────────────────────────────────────────────
    dev["rule_correct"]   = dev["rule_predicted"]   == dev["gold_label"]
    dev["zs_correct"]     = dev["zs_predicted"]     == dev["gold_label"]
    dev["hybrid_correct"] = dev["hybrid_predicted"] == dev["gold_label"]

    # ── Write output ──────────────────────────────────────────────────────────
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

    # ── Overall accuracy summary ──────────────────────────────────────────────
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
    # Show how often the hybrid deferred to each sub-classifier
    print(
        f"  (hybrid used rule for {(out['hybrid_method']=='rule').sum()}/{n}, "
        f"zero-shot for {(out['hybrid_method']=='zero-shot').sum()}/{n})"
    )

    # ── Per-class accuracy ────────────────────────────────────────────────────
    # Shows all three classifiers side-by-side so you can see which classes
    # benefit from the hybrid (rule improves) vs. hurt it (zero-shot better).
    # Sorted by count so the most frequent classes appear first.
    print("\nPer-class accuracy:")
    per_class = out.groupby("gold_label").agg(
        count          =("gold_label",     "size"),
        rule_correct   =("rule_correct",   "sum"),
        zs_correct     =("zs_correct",     "sum"),
        hybrid_correct =("hybrid_correct", "sum"),
    )
    per_class["rule_pct"]   = (per_class["rule_correct"]   / per_class["count"] * 100).round(1)
    per_class["zs_pct"]     = (per_class["zs_correct"]     / per_class["count"] * 100).round(1)
    per_class["hybrid_pct"] = (per_class["hybrid_correct"] / per_class["count"] * 100).round(1)
    print(
        per_class[
            [
                "count",
                "rule_correct",  "rule_pct",
                "zs_correct",    "zs_pct",
                "hybrid_correct","hybrid_pct",
            ]
        ]
        .sort_values("count", ascending=False)
        .to_string()
    )

    # ── Confusion matrices ────────────────────────────────────────────────────
    # One matrix per classifier. Rows = true gold label, columns = predicted
    # label. Off-diagonal cells reveal the specific class pairs that are being
    # confused, which guides targeted rule or prompt improvements.
    print("\nRule confusion matrix (rows=gold, cols=rule_predicted):")
    print(pd.crosstab(out["gold_label"], out["rule_predicted"]).to_string())

    print("\nZero-shot confusion matrix (rows=gold, cols=zs_predicted):")
    print(pd.crosstab(out["gold_label"], out["zs_predicted"]).to_string())

    print("\nHybrid confusion matrix (rows=gold, cols=hybrid_predicted):")
    print(pd.crosstab(out["gold_label"], out["hybrid_predicted"]).to_string())


if __name__ == "__main__":
    main()

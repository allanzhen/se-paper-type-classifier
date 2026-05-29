"""Build a 30-paper dev set for tuning the hybrid classifier.

Excludes the 100 gold papers, then samples:
- 27 papers stratified by zero-shot prediction (3 per class x 9 classes)
- 3 papers where the rule classifier and zero-shot disagree

Output: data/gold/dev_set_template.csv with an empty `classification`
column for the user to fill in manually. The `zs_predicted` and
`rule_predicted` columns are kept as labelling hints.

Requires data/processed/corpus_labeled.csv (rule predictions) and
data/processed/corpus_labeled_zs.csv (zero-shot predictions) to exist.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RULE_PATH = REPO_ROOT / "data" / "processed" / "corpus_labeled.csv"
ZS_PATH = REPO_ROOT / "data" / "processed" / "corpus_labeled_zs.csv"
GOLD_PATH = REPO_ROOT / "data" / "gold" / "gold_standard_papers.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "gold" / "dev_set_template.csv"

PER_CLASS = 3
DISAGREEMENT_COUNT = 3
MIN_PER_CLASS = 2
TARGET_TOTAL = 30
RANDOM_STATE = 42

ALL_CLASSES = (
    "Empirical Study",
    "Controlled Experiment",
    "Systematic Literature Review",
    "Survey",
    "Tool Paper",
    "Experience Report",
    "Case Study",
    "Position Paper",
    "Theoretical Contribution",
)


def main() -> None:
    if not ZS_PATH.exists():
        raise SystemExit(
            f"{ZS_PATH.name} not found. Run src/classify/zero_shot.py first."
        )
    if not RULE_PATH.exists():
        raise SystemExit(
            f"{RULE_PATH.name} not found. Run src/classify/rule_classifier.py first."
        )

    rule_df = pd.read_csv(RULE_PATH)
    zs_df = pd.read_csv(ZS_PATH)
    gold_df = pd.read_csv(GOLD_PATH)

    df = rule_df.merge(
        zs_df[["paper_id", "predicted_type_zs", "confidence_zs"]],
        on="paper_id",
        how="inner",
    )
    print(f"Combined corpus: {len(df)} papers")

    before = len(df)
    df = df[~df["paper_id"].isin(gold_df["ID"])].reset_index(drop=True)
    print(f"Excluded {before - len(df)} gold papers; {len(df)} candidates remain")
    assert df["paper_id"].isin(gold_df["ID"]).sum() == 0, "Gold paper leaked into pool"

    # Stratified sample by zero-shot prediction
    print(f"\nStratified sample ({PER_CLASS} per zero-shot predicted class):")
    stratified_parts = []
    used_ids: set[str] = set()
    for cls, group in df.groupby("predicted_type_zs"):
        n = min(PER_CLASS, len(group))
        sample = group.sample(n=n, random_state=RANDOM_STATE)
        stratified_parts.append(sample)
        used_ids.update(sample["paper_id"])
        marker = "" if n == PER_CLASS else f"  (WARNING: only {n} available)"
        print(f"  {cls:>34s}: {n}{marker}")
    stratified = pd.concat(stratified_parts, ignore_index=True)
    print(f"  total stratified: {len(stratified)}")

    # Disagreement sample: rule fired (not Unknown) and disagrees with zero-shot
    remaining = df[~df["paper_id"].isin(used_ids)]
    disagree = remaining[
        (remaining["predicted_type"] != "Unknown")
        & (remaining["predicted_type"] != remaining["predicted_type_zs"])
    ]
    n_disagree = min(DISAGREEMENT_COUNT, len(disagree))
    disagree_sample = disagree.sample(n=n_disagree, random_state=RANDOM_STATE)
    print(f"\nDisagreement sample: {n_disagree} papers (from {len(disagree)} disagreement candidates)")

    dev = pd.concat([stratified, disagree_sample], ignore_index=True)

    # Pad under-represented classes from rule predictions. Zero-shot
    # under-predicts Case Study / Controlled Experiment / Survey on this
    # corpus, so stratifying purely by zs leaves those buckets empty.
    print(f"\nPadding from rule predictions (target: >= {MIN_PER_CLASS} per class):")
    used_ids = set(dev["paper_id"])
    pool = df[~df["paper_id"].isin(used_ids)]
    extra_parts: list[pd.DataFrame] = []
    for cls in ALL_CLASSES:
        in_dev = (
            (dev["predicted_type_zs"] == cls) | (dev["predicted_type"] == cls)
        ).sum()
        if in_dev >= MIN_PER_CLASS:
            continue
        need = MIN_PER_CLASS - int(in_dev)
        candidates = pool[pool["predicted_type"] == cls]
        n = min(need, len(candidates))
        if n == 0:
            print(f"  {cls:>34s}: 0 (no rule-predicted candidates available)")
            continue
        sample = candidates.sample(n=n, random_state=RANDOM_STATE)
        extra_parts.append(sample)
        used_ids.update(sample["paper_id"])
        pool = pool[~pool["paper_id"].isin(sample["paper_id"])]
        print(f"  {cls:>34s}: +{n}")
    if extra_parts:
        dev = pd.concat([dev, *extra_parts], ignore_index=True)

    # Random pad to TARGET_TOTAL. Doesn't improve rare-class coverage --
    # rare classes are genuinely rare in the corpus -- but boosts robustness
    # for the common classes and gets us closer to the planned 30 papers.
    used_ids = set(dev["paper_id"])
    remaining_pool = df[~df["paper_id"].isin(used_ids)]
    pad_needed = TARGET_TOTAL - len(dev)
    if pad_needed > 0 and len(remaining_pool) > 0:
        n = min(pad_needed, len(remaining_pool))
        random_pad = remaining_pool.sample(n=n, random_state=RANDOM_STATE)
        dev = pd.concat([dev, random_pad], ignore_index=True)
        print(f"\nRandom pad: +{n} papers to reach {len(dev)} total")

    out = dev.rename(
        columns={
            "predicted_type_zs": "zs_predicted",
            "predicted_type": "rule_predicted",
        }
    )[
        [
            "paper_id",
            "title",
            "abstract",
            "year",
            "venue",
            "zs_predicted",
            "rule_predicted",
        ]
    ]
    out["classification"] = ""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(out)} papers to {OUTPUT_PATH}")
    print("\nClass coverage (by zero-shot prediction):")
    print(out["zs_predicted"].value_counts().to_string())


if __name__ == "__main__":
    main()

"""Rule-based paper-type classifier.

Reads data/processed/corpus.csv, applies the keyword/phrase rules from
rules.py to each paper's title+abstract, and writes
data/processed/corpus_labeled.csv with these new columns:

    predicted_type  -- one of the labels in RULES, or "Unknown"
    confidence      -- top_score / (top_score + second_score), in [0, 1]
    unknown_reason  -- "" when labeled; "no_match" or "tie:LabelA|LabelB"
                       when predicted_type is "Unknown"
    matched_rules   -- "; "-joined list of patterns that fired for the
                       winning label (empty when "no_match"; full union
                       of tied-label hits when "tie")

A paper is labeled "Unknown" when no rule matches at all (no_match), or
when the top two labels tie on match count (tie:...).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rules import COMPILED  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "data" / "processed" / "corpus.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "corpus_labeled.csv"


def classify(title: str, abstract: str) -> tuple[str, float, str, list[str]]:
    """Classify one paper.

    Returns (label, confidence, unknown_reason, matched_patterns).
    unknown_reason is "" for labeled papers, "no_match" when nothing fired,
    or "tie:LabelA|LabelB[|...]" when the top score was shared.
    """
    text = f"{title}\n{abstract}"
    scores: dict[str, int] = {}
    matches: dict[str, list[str]] = {}
    for label, patterns in COMPILED.items():
        hits = [p.pattern for p in patterns if p.search(text)]
        scores[label] = len(hits)
        matches[label] = hits

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    if top_score == 0:
        return "Unknown", 0.0, "no_match", []

    if top_score == second_score:
        tied = [label for label, score in ranked if score == top_score]
        union = sorted({h for label in tied for h in matches[label]})
        return "Unknown", 0.0, "tie:" + "|".join(tied), union

    confidence = top_score / (top_score + second_score)
    return top_label, confidence, "", matches[top_label]


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} papers from {INPUT_PATH.name}")

    labels, confs, reasons, hit_lists = [], [], [], []
    for _, row in df.iterrows():
        title = "" if pd.isna(row.get("title")) else str(row["title"])
        abstract = "" if pd.isna(row.get("abstract")) else str(row["abstract"])
        label, conf, reason, hits = classify(title, abstract)
        labels.append(label)
        confs.append(round(conf, 3))
        reasons.append(reason)
        hit_lists.append("; ".join(hits))

    df["predicted_type"] = labels
    df["confidence"] = confs
    df["unknown_reason"] = reasons
    df["matched_rules"] = hit_lists

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} labeled papers to {OUTPUT_PATH}")

    print("\nPredicted-type distribution:")
    print(df["predicted_type"].value_counts().to_string())

    unknown = df[df["predicted_type"] == "Unknown"]
    if len(unknown):
        print(f"\nUnknown breakdown ({len(unknown)} papers):")
        # Bucket ties by their full reason ("tie:A|B") for visibility.
        print(unknown["unknown_reason"].value_counts().to_string())

    labeled = df[df["predicted_type"] != "Unknown"]
    if len(labeled):
        print(
            f"\nCoverage: {len(labeled)}/{len(df)} "
            f"({100 * len(labeled) / len(df):.1f}%) labeled, "
            f"mean confidence on labeled = {labeled['confidence'].mean():.2f}"
        )


if __name__ == "__main__":
    main()

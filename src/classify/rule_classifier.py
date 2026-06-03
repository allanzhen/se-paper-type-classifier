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

# Add the classify directory to sys.path so rules.py can be imported as a
# sibling module without restructuring the package layout.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rules import (  # noqa: E402  — import must follow sys.path insert
    COMPILED,
    PRACTITIONER_SURVEY,
    PROTOCOL,
    SECONDARY_STUDY,
)

# Tree decisions resolve a clear secondary/survey signal, but only when the
# paper's central contribution isn't plainly a Tool or Theoretical artifact that
# merely *grounds* itself in a review/survey (e.g. "...A Systematic Mapping Study
# and a Conceptual Model" is a Theoretical Contribution, not an SLR). A keyword
# vote of >= this many hits for those classes counts as a dominant contribution
# and defers to the regular keyword vote instead of the tree.
CONTRIBUTION_DOMINANCE = 2

REPO_ROOT   = Path(__file__).resolve().parents[2]
INPUT_PATH  = REPO_ROOT / "data" / "processed" / "corpus.csv"         # cleaned corpus
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "corpus_labeled.csv" # corpus with rule predictions appended


def _classify_tree(
    text: str, scores: dict[str, int]
) -> tuple[str, float, str, list[str]] | None:
    """Apply the Survey/SLR/Empirical decision tree; None = tree doesn't decide.

    Implements the "Labeling Decision Procedure" from
    docs/Paper Type Definitions.md as a precedence layer:

      Step 1  secondary study? (reviews existing literature)
      Step 2A   secondary + documented protocol -> Systematic Literature Review
                secondary, no protocol           -> Survey (literature sense)
      Step 2B  primary + questionnaire/self-report -> Survey (practitioner sense)

    Returns the same 4-tuple shape as classify(); returns None when no tree
    branch applies so the caller falls back to the plain keyword vote. The tree
    is skipped when a Tool/Theoretical contribution dominates (>= the dominance
    threshold), honoring the doc's "label by dominant contribution" rule for
    papers that merely ground themselves in a review/survey.
    """
    if (
        scores.get("Tool Paper", 0) >= CONTRIBUTION_DOMINANCE
        or scores.get("Theoretical Contribution", 0) >= CONTRIBUTION_DOMINANCE
    ):
        return None

    secondary = [p.pattern for p in SECONDARY_STUDY if p.search(text)]
    if secondary:
        protocol = [p.pattern for p in PROTOCOL if p.search(text)]
        if protocol:  # Step 2A: documented, reproducible protocol
            cues = sorted(set(secondary) | set(protocol))
            return "Systematic Literature Review", 1.0, "", cues
        # Step 2A: narrative review with no protocol -> literature-sense Survey
        return "Survey", 1.0, "", secondary

    practitioner = [p.pattern for p in PRACTITIONER_SURVEY if p.search(text)]
    if practitioner:  # Step 2B: self-reported questionnaire data -> Survey
        return "Survey", 1.0, "", practitioner

    return None


def classify(title: str, abstract: str) -> tuple[str, float, str, list[str]]:
    """Classify one paper using keyword/phrase match counts.

    Concatenates title and abstract, then counts how many compiled regex
    patterns from rules.py match for each label. The label with the most
    matches wins. Returns a 4-tuple:
        label           -- winning class label, or "Unknown"
        confidence      -- top_score / (top_score + second_score); 0.0 for Unknown
        unknown_reason  -- "" if labeled, "no_match" if nothing fired,
                           "tie:LabelA|LabelB" if top score was shared
        matched_patterns -- list of pattern strings that fired for the winner
                            (union of all tied labels when tied)
    """
    # Newline separator gives the model a natural boundary between title and
    # abstract so a pattern can't accidentally straddle both fields.
    text = f"{title}\n{abstract}"

    scores: dict[str, int]         = {}  # total match count per label
    matches: dict[str, list[str]]  = {}  # pattern strings that fired, per label

    # COMPILED is {label: [compiled_regex, ...]} from rules.py.
    # Each pattern that matches anywhere in the text counts as one vote for
    # that label — the classifier is a simple majority-vote over regex hits.
    for label, patterns in COMPILED.items():
        hits = [p.pattern for p in patterns if p.search(text)]
        scores[label]  = len(hits)   # vote count
        matches[label] = hits        # keep the pattern strings for the matched_rules column

    # ── Decision-tree resolution (docs/Paper Type Definitions.md) ───────────
    # Applied BEFORE the keyword vote because it encodes the deliberate
    # precedence the flat vote can't: a documented protocol makes a review an
    # SLR (Step 2A), and a questionnaire makes a primary study a Survey (2B).
    # We skip the tree when a Tool/Theoretical contribution clearly dominates,
    # so a model/tool paper that merely grounds itself in a review isn't
    # mislabeled as a secondary study (the mixed-method "dominant contribution"
    # rule from the doc).
    tree = _classify_tree(text, scores)
    if tree is not None:
        return tree

    # Sort labels by vote count descending so ranked[0] is always the top candidate.
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score   = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    # No rule fired at all — paper is too ambiguous or outside the taxonomy.
    if top_score == 0:
        return "Unknown", 0.0, "no_match", []

    # Documented precedence (Tool Paper definition; docs/Paper Type Definitions.md
    # and the comment on the Tool Paper rules): a newly built artifact that is
    # ALSO empirically evaluated is a Tool Paper, not an Empirical Study. The flat
    # vote can't see this — it leaves such papers as a Tool|Empirical tie (->
    # Unknown -> zero-shot, which is weak on tools) or lets the evaluation cues
    # win. So when the top of the vote is a Tool-vs-Empirical contest, award it to
    # Tool Paper. Scoped STRICTLY to Empirical: other contests (e.g.
    # Tool|Experience Report on "SoHist", Tool|Theoretical on "Combining
    # Insights") are left alone because those non-Tool labels are often correct.
    tool_score = scores["Tool Paper"]
    top_labels = [label for label, score in ranked if score == top_score]
    if (
        tool_score > 0
        and "Empirical Study" in top_labels
        and set(top_labels) <= {"Tool Paper", "Empirical Study"}
    ):
        emp_score = scores["Empirical Study"]
        return (
            "Tool Paper",
            tool_score / (tool_score + emp_score),
            "",
            matches["Tool Paper"],
        )

    # Two or more labels tied — a winner can't be chosen without a tiebreaker,
    # so we surface this as Unknown with a diagnostic reason string.
    if top_score == second_score:
        tied  = [label for label, score in ranked if score == top_score]
        # Report the union of all matched patterns across tied labels so the
        # reason string is useful for debugging which rules are conflicting.
        union = sorted({h for label in tied for h in matches[label]})
        return "Unknown", 0.0, "tie:" + "|".join(tied), union

    # Confidence is the fraction of total top-two votes that the winner holds.
    # A clear winner with no second-place matches scores 1.0; a narrow win
    # (e.g. 2 vs 1) scores ~0.67. This stays in [0, 1] without needing softmax.
    confidence = top_score / (top_score + second_score)
    return top_label, confidence, "", matches[top_label]


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} papers from {INPUT_PATH.name}")

    # Parallel lists — one entry per row, appended in order so alignment with
    # df is guaranteed without needing a merge step at the end.
    labels, confs, reasons, hit_lists = [], [], [], []

    for _, row in df.iterrows():
        # Guard against NaN fields before passing to classify().
        title    = "" if pd.isna(row.get("title"))    else str(row["title"])
        abstract = "" if pd.isna(row.get("abstract")) else str(row["abstract"])
        label, conf, reason, hits = classify(title, abstract)
        labels.append(label)
        confs.append(round(conf, 3))
        reasons.append(reason)
        # Join matched patterns into a single string so they survive CSV serialisation.
        hit_lists.append("; ".join(hits))

    df["predicted_type"]  = labels
    df["confidence"]      = confs
    df["unknown_reason"]  = reasons
    df["matched_rules"]   = hit_lists   # semicolon-separated; split on "; " to recover the list

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} labeled papers to {OUTPUT_PATH}")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print("\nPredicted-type distribution:")
    print(df["predicted_type"].value_counts().to_string())

    # Break down Unknown papers by reason so it's easy to distinguish
    # "no rules fired at all" (no_match) from "rules conflicted" (tie:...).
    unknown = df[df["predicted_type"] == "Unknown"]
    if len(unknown):
        print(f"\nUnknown breakdown ({len(unknown)} papers):")
        # Bucket ties by their full reason ("tie:A|B") for visibility.
        print(unknown["unknown_reason"].value_counts().to_string())

    # Coverage and mean confidence on the papers that were successfully labeled.
    # Low coverage suggests the rules are too narrow; low confidence suggests
    # many near-ties that could be resolved by adding stronger patterns.
    labeled = df[df["predicted_type"] != "Unknown"]
    if len(labeled):
        print(
            f"\nCoverage: {len(labeled)}/{len(df)} "
            f"({100 * len(labeled) / len(df):.1f}%) labeled, "
            f"mean confidence on labeled = {labeled['confidence'].mean():.2f}"
        )


if __name__ == "__main__":
    main()

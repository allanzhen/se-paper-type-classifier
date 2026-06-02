"""Zero-shot paper-type classifier using a Hugging Face NLI model.

Treats classification as natural-language inference: for each candidate
label, the model scores how strongly the hypothesis "This paper is
<label>" is entailed by the paper's title+abstract. The label with the
highest entailment score wins. Unlike the rule-based classifier, this
always returns a label -- if you want an "Unknown" bucket, threshold
on the `confidence_zs` column downstream.

Runs on Apple-Silicon MPS when available, else CPU. Expect roughly
10-20 minutes on 470 papers on MPS, ~30-60 min on CPU.

Output: data/processed/corpus_labeled_zs.csv with three new columns:
    predicted_type_zs  -- one of the short labels in LABELS
    confidence_zs      -- entailment score of the winning label, in [0, 1]
    zs_scores          -- JSON dict {label: score} for all 9 labels (for
                          ensembling / threshold tuning later)
"""

import json
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline

REPO_ROOT   = Path(__file__).resolve().parents[2]
INPUT_PATH  = REPO_ROOT / "data" / "processed" / "corpus.csv"            # cleaned corpus from clean_corpus.py
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "corpus_labeled_zs.csv" # corpus with zero-shot predictions appended

# DeBERTa-v3-large fine-tuned on four NLI datasets (MNLI, Fever-NLI, ANLI,
# Ling-NLI, WaNLI). Chosen because it consistently outperforms other public
# zero-shot models on fine-grained classification tasks.
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

# The model evaluates: does the abstract ENTAIL this hypothesis?
# The curly-brace placeholder is filled by each expanded label phrase.
HYPOTHESIS_TEMPLATE = "This paper is {}."

# Maps the short output label to the expanded natural-language phrase fed to
# the NLI model. The expanded phrase is what the model actually reads, so its
# wording directly controls classification accuracy. Tune the expanded phrases
# (not the short names) to shift accuracy on ambiguous classes; the short
# names are stable identifiers used everywhere else in the pipeline.
#
# NOTE on wording style: keep each phrase SHORT, POSITIVE, and crisp. NLI
# zero-shot scores whether the abstract entails the *whole* hypothesis, so long
# compound phrases dilute the signal and negation ("does not introduce a tool")
# tends to backfire -- the model keys on the negated noun and scores it higher.
# An earlier negation-heavy rewrite measured ~2% worse and was reverted here.
LABELS: dict[str, str] = {
    "Empirical Study": (
        "an empirical study that observes, mines, or measures real-world "
        "software, data, or developers to answer a research question"
    ),
    "Controlled Experiment": (
        "a controlled experiment with human participants randomly assigned "
        "to treatment and control groups to statistically test a hypothesis"
    ),
    "Systematic Literature Review": (
        "a systematic literature review with a documented search strategy, "
        "inclusion and exclusion criteria, and a reproducible protocol"
    ),
    "Survey": (
        "a literature survey summarizing prior work on a topic, or a "
        "questionnaire or interview study of software practitioners"
    ),
    "Tool Paper": (
        "a paper that presents a new software tool, automated technique, or "
        "system as its central contribution, describing what it does, how it "
        "is implemented, and how it is applied"
    ),
    "Experience Report": (
        "an experience report describing the practical application of a "
        "technique, process, or tool in a real organization, with lessons learned"
    ),
    "Case Study": (
        "an in-depth case study of a specific organization, project, or team, "
        "using multiple sources of evidence such as interviews and documents"
    ),
    "Position Paper": (
        "a position or vision paper arguing for a particular viewpoint or "
        "future research direction"
    ),
    "Theoretical Contribution": (
        "a paper that proposes a new taxonomy, conceptual model or framework, "
        "metric, or formal theory for software engineering as its central deliverable"
    ),
}

# Pre-compute the list and reverse map once at import time so classify_papers()
# doesn't rebuild them on every call.
CANDIDATE_LABELS = list(LABELS.values())          # expanded phrases fed to the NLI model
EXPANDED_TO_SHORT = {v: k for k, v in LABELS.items()}  # maps expanded phrase back to short label


def _pick_device() -> str | int:
    """Return the best available device identifier for the transformers pipeline.

    Returns "mps" on Apple Silicon (significantly faster than CPU for NLI),
    or -1, which is the transformers convention for CPU inference.
    """
    if torch.backends.mps.is_available():
        return "mps"
    return -1  # transformers pipeline convention for CPU


def build_pipeline():
    """Load the NLI model and return a configured zero-shot pipeline.

    Model loading is the slow step (~30s on first call, faster after the
    weights are cached locally by Hugging Face). Call this once and pass
    the result into classify_papers() to avoid reloading across batches.
    """
    device = _pick_device()
    print(f"Loading {MODEL_NAME} (device={device})...")
    clf = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=device,
    )
    # DeBERTa has a hard 512-token context window. Without this cap, very long
    # abstracts raise a runtime error rather than being truncated gracefully.
    clf.tokenizer.model_max_length = 512
    return clf


def classify_papers(df: pd.DataFrame, clf=None) -> pd.DataFrame:
    """Run zero-shot classification on a DataFrame of papers.

    Requires 'title' and 'abstract' columns. Returns a copy of df with
    three new columns appended:
        predicted_type_zs  -- winning short label
        confidence_zs      -- entailment score of the winning label
        zs_scores          -- JSON string with scores for all 9 labels

    Pass an existing pipeline via `clf` to avoid reloading the model when
    this function is called from an evaluator that already has one loaded.
    """
    if clf is None:
        clf = build_pipeline()

    # Work on a copy so the caller's DataFrame is never mutated. reset_index
    # ensures idx in the loop matches the DataFrame's positional index.
    df = df.copy().reset_index(drop=True)
    print(f"Classifying {len(df)} papers...\n")

    labels: list[str]     = []
    confs: list[float]    = []
    all_scores: list[str] = []

    for idx, row in df.iterrows():
        # Guard against NaN fields — the NLI model requires a plain string.
        title    = "" if pd.isna(row.get("title"))    else str(row["title"])
        abstract = "" if pd.isna(row.get("abstract")) else str(row["abstract"])

        # Concatenate title and abstract into a single sequence. The model
        # scores how well this combined text entails each hypothesis phrase.
        text = f"{title}. {abstract}"

        result = clf(
            text,
            candidate_labels=CANDIDATE_LABELS,   # the expanded NLI hypothesis phrases
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=False,  # single-label: scores sum to 1 via softmax
        )

        # The pipeline returns labels and scores sorted in descending score
        # order, so index 0 is always the top-ranked (most likely) class.
        short_labels = [EXPANDED_TO_SHORT[lbl] for lbl in result["labels"]]
        top_label = short_labels[0]
        top_score = float(result["scores"][0])

        # Store the full per-label score dict as a JSON string so it survives
        # CSV serialisation and can be parsed later for ensembling or threshold
        # tuning without re-running the model.
        per_label = {
            lbl: round(float(s), 4)
            for lbl, s in zip(short_labels, result["scores"])
        }

        labels.append(top_label)
        confs.append(round(top_score, 4))
        all_scores.append(json.dumps(per_label))

        # Print progress every 25 papers (and at the very last paper) so long
        # runs don't appear frozen without flooding the terminal with per-row logs.
        if (idx + 1) % 25 == 0 or (idx + 1) == len(df):
            print(f"  {idx + 1}/{len(df)} papers classified")

    df["predicted_type_zs"] = labels
    df["confidence_zs"]     = confs
    df["zs_scores"]         = all_scores  # JSON string — parse with json.loads() downstream
    return df


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} papers from {INPUT_PATH.name}")

    # classify_papers() builds the pipeline internally when none is passed.
    df = classify_papers(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(df)} labeled papers to {OUTPUT_PATH}")

    # Distribution check — a heavily skewed distribution suggests a label
    # phrasing issue or a corpus bias worth investigating before evaluation.
    print("\nPredicted-type distribution (zero-shot):")
    print(df["predicted_type_zs"].value_counts().to_string())
    print(f"\nMean top-1 confidence: {df['confidence_zs'].mean():.3f}")


if __name__ == "__main__":
    main()

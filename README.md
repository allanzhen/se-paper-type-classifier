# SE Paper-Type Classifier — Technical Debt Research Landscape

Software engineering research takes many forms: controlled experiments, systematic
literature reviews, tool papers, experience reports, and more. The mix of paper types
a community produces reflects how it generates and validates knowledge, yet that
distribution is rarely studied explicitly.

This project characterises the landscape of **Technical Debt** research published in
high-quality SE venues (CORE **A** and **A\*** ranked). A fully automated NLP pipeline
assigns each paper a *type* label from its **title and abstract alone**, and the
resulting distribution is analysed to see whether the community leans toward synthesis,
empirical validation, or tool building, and how that mix has shifted since Technical
Debt emerged as a formal research area around 2010.

The core deliverable is a **reproducible, reusable classifier**: give it a title and
abstract, get back a paper-type label with a confidence score. It is built to work for
any SE topic, not just Technical Debt.

## Taxonomy

Papers are classified into **9 canonical types** (the single source of truth is
[src/classify/labels.py](src/classify/labels.py)):

`Empirical Study` · `Tool Paper` · `Case Study` · `Theoretical Contribution` ·
`Survey` · `Position Paper` · `Experience Report` · `Systematic Literature Review` ·
`Controlled Experiment`

Several types overlap superficially (e.g. a *literature* survey vs. an SLR, or a
*practitioner* survey vs. an empirical study). The full definitions and the decision
procedure used to disambiguate them are in
[docs/Paper Type Definitions.md](docs/Paper%20Type%20Definitions.md).

## Repository structure

```
se-paper-type-classifier/
├── src/
│   ├── ingest/        # Build the corpus: DBLP collection + Semantic Scholar abstracts
│   ├── classify/      # The classifiers: labels, rules, rule-based + zero-shot models
│   ├── evaluate/      # Score classifiers on gold/dev sets; apply hybrid to full corpus
│   └── viz/           # Generate the figures
├── data/
│   ├── raw/           # Source inputs (CORE rankings, API caches) - git-ignored, regenerable
│   ├── interim/       # Intermediate collection output (dblp_papers.csv) - git-ignored
│   ├── processed/     # Corpus + labeled/eval CSVs - TRACKED study outputs
│   └── gold/          # Hand-labeled gold set (100) + dev set (30) - TRACKED
├── results/
│   ├── figures/       # Generated PNG charts
│   └── tables/        # Reserved for tabular outputs
├── docs/              # Paper-type definitions & labeling decision tree
├── tests/             # Pytest suite (run with `python -m pytest`)
├── requirements.txt
└── pytest.ini         # Sets import paths for the sibling src/ packages
```

`data/raw/` and `data/interim/` are regenerable and git-ignored; `data/processed/` and
`data/gold/` are committed so the study can be reproduced without re-collecting papers.

## The classifiers

- **Rule-based** ([src/classify/rule_classifier.py](src/classify/rule_classifier.py)) —
  counts keyword/phrase regex matches (from [rules.py](src/classify/rules.py)) per label
  and takes the majority vote, with a decision tree for the Survey/SLR distinction.
  High precision and instant, but emits `"Unknown"` when no rule fires or the top labels tie.
- **Zero-shot** ([src/classify/zero_shot.py](src/classify/zero_shot.py)) - uses the
  `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` NLI model to score how
  strongly the abstract entails each of the 9 label hypotheses. Always returns a label.
- **Hybrid** (`combine()` in [src/evaluate/evaluate_hybrid.py](src/evaluate/evaluate_hybrid.py)) -
  **rule-first**: use the rule label when it fires, otherwise fall back to zero-shot.
  This is the project's best and reported classifier.

## Setup

```bash
git clone <repo-url>
cd se-paper-type-classifier
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS/Linux:    source .venv/bin/activate
pip install -r requirements.txt        # Python 3.9+
```

- The **first** zero-shot run auto-downloads the DeBERTa weights (~800 MB) from the
  HuggingFace Hub and caches them locally.
- A `.env` with `SEMANTIC_SCHOLAR_API_KEY` is **optional** and only used for the
  from-scratch data collection (it raises the Semantic Scholar rate limit). See
  `.env.example`.

## Reproducing the study results

The corpus and the gold/dev sets are committed, so you can regenerate every label,
metric, and figure without re-collecting any papers. Run from the repo root:

```bash
# 1. Classify the corpus with both models
python src/classify/rule_classifier.py     # -> data/processed/corpus_labeled.csv
python src/classify/zero_shot.py           # -> data/processed/corpus_labeled_zs.csv   (slow; loads DeBERTa)

# 2. Evaluate on the sealed 100-paper gold set (prints accuracy + confusion matrices)
python src/evaluate/evaluate_zero_shot.py  # -> data/processed/gold_zs_eval.csv
python src/evaluate/evaluate_hybrid.py     # -> data/processed/gold_hybrid_eval.csv

# 3. (Optional) Inspect classifier behaviour on the 30-paper dev set
python src/evaluate/evaluate_dev.py        # -> data/processed/dev_eval.csv

# 4. Apply the hybrid to the whole corpus (the final reportable dataset)
python src/evaluate/label_corpus_hybrid.py # -> data/processed/corpus_labeled_hybrid.csv

# 5. Generate the figures
python src/viz/corpus_overview.py          # -> results/figures/corpus_overview.png
python src/viz/corpus_type_distribution.py # -> results/figures/corpus_type_distribution.png
python src/viz/results_overview.py         # -> results/figures/accuracy_comparison.png, hybrid_confusion_matrix.png
```

Steps 2–4 reuse the cached predictions from step 1 (joining by `paper_id`), so they run
in well under a second.

## Full pipeline from scratch (optional)

To rebuild the corpus from the source venues instead of using the committed
`corpus.csv`, run the ingest stage first. This requires the CORE ranking files in
`data/raw/` (`core_rankings.csv`, optionally `core_journal_rankings.csv`, from
<https://portal.core.edu.au/conf-ranks/>) and is **slow and non-deterministic**, as
DBLP/Semantic Scholar are rate-limited and their results drift over time, so the corpus
may differ from the committed one.

```bash
python src/ingest/collect_dblp.py     # 41 Technical-Debt queries, CORE A/A* venues -> data/interim/dblp_papers.csv
python src/ingest/fetch_abstracts.py  # enrich with Semantic Scholar abstracts      -> data/processed/corpus.csv
# ...then continue from step 1 of "Reproducing the study results" above.
```

To regenerate the dev-set template for re-annotation, use
`python src/ingest/build_dev_set.py` (samples papers stratified by prediction and by
rule/zero-shot disagreement, excluding the gold set).

## Using the classifiers on other papers

The classifiers are reusable on any title + abstract. There is no per-paper CLI, but the
functions are importable. Run from the repo root (or add `src/classify` and
`src/evaluate` to `sys.path`, as the scripts do).

**Rule-based** — instant, no model download:

```python
import sys; sys.path.insert(0, "src/classify")
from rule_classifier import classify

label, confidence, reason, matched = classify(
    title="A Tool for Detecting Self-Admitted Technical Debt",
    abstract="We present an automated tool that ...",
)
print(label, confidence)   # e.g. "Tool Paper" 0.75   (label is "Unknown" if no rule fires)
```

**Zero-shot** — load the DeBERTa pipeline once, classify a 1-row DataFrame:

```python
import sys; sys.path.insert(0, "src/classify")
import pandas as pd
from zero_shot import build_pipeline, classify_papers

clf = build_pipeline()                       # loads the model (download on first use)
df = pd.DataFrame([{"title": "...", "abstract": "..."}])
out = classify_papers(df, clf)
print(out.loc[0, "predicted_type_zs"], out.loc[0, "confidence_zs"])
```

**Hybrid (recommended)** — fuse both with the rule-first strategy:

```python
import sys; sys.path.insert(0, "src/classify"); sys.path.insert(0, "src/evaluate")
import pandas as pd
from rule_classifier import classify as rule_classify
from zero_shot import build_pipeline, classify_papers
from evaluate_hybrid import combine

title, abstract = "...", "..."
r_label, r_conf, _, _ = rule_classify(title, abstract)

zs = classify_papers(pd.DataFrame([{"title": title, "abstract": abstract}]), build_pipeline())
z_label, z_conf = zs.loc[0, "predicted_type_zs"], zs.loc[0, "confidence_zs"]

label, confidence, method = combine(r_label, r_conf, z_label, z_conf)
print(label, confidence, method)   # method is "rule" or "zero-shot"
```

## Tests

```bash
python -m pytest
```

`pytest.ini` configures the import paths. The suite (42 tests) covers the rule
definitions and classifier logic, the hybrid `combine()` strategy, the evaluation
pipelines, and gold-label normalization.

## Data & evaluation notes

- **Corpus:** ~342 papers in `data/processed/corpus.csv` (title, abstract, year, venue, …).
- **Gold set:** 100 hand-labeled papers (`data/gold/gold_standard_papers.csv`), **sealed**
  for final evaluation only.
- **Dev set:** 30 hand-labeled papers (`data/gold/dev_set_labeled.csv`) for tuning rules
  and inspecting behaviour.
- The evaluation scripts print overall and per-class accuracy plus confusion matrices,
  and save per-paper results to `data/processed/*_eval.csv`.

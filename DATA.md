# Data Description and Reproducibility

This document describes the datasets ingested, the cleaning steps applied, where data files live in this repository, and how to access them programmatically.

---

## 1. Datasets Targeted and Ingested

### DBLP (primary corpus source)
- **What**: Computer science bibliography database used to identify Technical Debt papers by keyword query.
- **API**: `https://dblp.org/search/publ/api`
- **Queries used** (21 total): `technical debt`, `architectural debt`, `code debt`, `design debt`, `test debt`, `documentation debt`, `self-admitted technical debt`, `satd`, `requirements debt`, `infrastructure debt`, `build debt`, `ML technical debt`, `machine learning technical debt`, `data debt`, `process debt`, `social debt`, `people debt`, `test smell`, `architecture erosion`, `software erosion`, `techdebt`
- **Fields retrieved**: title, year, venue, DOI, URL
- **Raw results**: 3,128 papers before any filtering

### Semantic Scholar (abstract retrieval)
- **What**: Used to retrieve abstracts for papers identified via DBLP, looked up by DOI (primary) and title search (fallback).
- **API**: `https://api.semanticscholar.org/graph/v1/paper/`
- **Fields retrieved**: paperId, title, abstract, year, venue
- **Papers with abstracts found**: 342 of 683 post-filter candidates

### CORE Conference Rankings (venue quality filter)
- **What**: CORE 2023 rankings used to restrict corpus to A and A* rated venues only.
- **Source**: Downloaded manually from `https://portal.core.edu.au/conf-ranks/`
- **Files**:
  - `data/raw/core_rankings.csv` — conference rankings
  - `data/raw/core_journal_rankings.csv` — journal rankings

---

## 2. Cleaning Activities

The following steps were applied in order to produce the final corpus:

1. **Raw DBLP queries** — 21 keyword searches run against DBLP with no filtering → **3,128 papers**
2. **CORE A/A* venue filter + year ≥ 2010** — dropped anything before 2010 or from a non-A/A* venue → **683 papers**
3. **Abstract retrieval via Semantic Scholar** — looked up each paper; kept only those with an abstract → **342 papers (final corpus)**

**Venue normalisation**: Venue strings were lowercased, stripped of ordinal numbers, years, and common prefixes (e.g. "IEEE International Conference on") before matching against CORE rankings. Both full venue title and acronym were matched. Preprints (arXiv/CoRR) were excluded.

**Abstract quality filter**: Papers with abstracts shorter than 100 characters were dropped.

**No deduplication was required**: DBLP returns unique records per query; cross-query duplicates were removed by title key during collection.

---

## 3. Data File Locations

All data files are committed to this repository under the `data/` directory.

```
data/
├── raw/
│   ├── core_rankings.csv           # CORE conference rankings (A/A*/B/C)
│   └── core_journal_rankings.csv   # CORE journal rankings
├── processed/
│   ├── corpus.csv                  # Final 342-paper corpus (title, abstract, year, venue, paper_id)
│   ├── corpus_labeled.csv          # Rule-based classifier predictions on full corpus
│   ├── corpus_labeled_zs.csv       # Zero-shot NLI predictions on full corpus
│   ├── corpus_labeled_hybrid.csv   # Hybrid classifier predictions on full corpus
│   ├── gold_hybrid_eval.csv        # Hybrid evaluation results on gold set (n=100)
│   └── gold_zs_eval.csv            # Zero-shot evaluation results on gold set (n=100)
└── gold/
    ├── gold_standard_papers.csv    # 100 manually labelled papers (paper_id, title, abstract, manual_label)
    ├── dev_set_labeled.csv         # 30-paper development set with labels
    └── dev_set_template.csv        # Blank template used during dev set labelling
```

---

## 4. Accessing the Data — Sample Code

### Load the corpus
```python
import pandas as pd

corpus = pd.read_csv("data/processed/corpus.csv")
print(corpus.shape)          # (342, 7)
print(corpus.columns.tolist())
# ['title', 'year', 'venue', 'doi', 'url', 'abstract', 'paper_id']
print(corpus.head())
```

### Load the gold standard labels
```python
gold = pd.read_csv("data/gold/gold_standard_papers.csv")
print(gold["manual_label"].value_counts())
# Empirical Study                 39
# Tool Paper                      27
# Systematic Literature Review    10
# ...
```

### Load hybrid classifier predictions
```python
hybrid = pd.read_csv("data/processed/corpus_labeled_hybrid.csv")
print(hybrid["predicted_type"].value_counts())
```

### Load evaluation results
```python
eval_results = pd.read_csv("data/processed/gold_hybrid_eval.csv")
accuracy = eval_results["hybrid_correct"].mean()
print(f"Hybrid accuracy: {accuracy:.1%}")   # 80.0%

# Per-class accuracy
print(eval_results.groupby("gold_label")["hybrid_correct"].mean().round(3))
```

### Reproduce the classifier pipeline
```bash
# Step 1 — collect papers from DBLP
python src/ingest/collect_dblp.py

# Step 2 — fetch abstracts from Semantic Scholar
python src/ingest/fetch_abstracts.py

# Step 3 — run rule-based classifier
python src/classify/rule_classifier.py

# Step 4 — run zero-shot NLI classifier
python src/classify/zero_shot.py

# Step 5 — evaluate hybrid on gold set
python src/evaluate/evaluate_hybrid.py
```

---

## Notes on Reproducibility

- The DBLP and Semantic Scholar API queries are cached as JSON files in `data/raw/dblp/` and `data/raw/semantic_scholar/` respectively. Re-running the ingest scripts will skip already-cached queries.
- The zero-shot classifier uses `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` downloaded automatically by HuggingFace Transformers on first run (~1.4 GB).
- All scripts are run from the project root directory.
- Python dependencies are listed in `requirements.txt`. Install with `pip install -r requirements.txt`.

"""
Fetch abstracts from Semantic Scholar for papers found via DBLP.
Reads data/interim/dblp_papers.csv, looks up each paper by DOI or title,
and saves the enriched corpus to data/processed/corpus.csv.

Run from project root:
    python src/ingest/fetch_abstracts.py
"""

import time
import requests
import pandas as pd
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
INPUT_PATH  = REPO_ROOT / "data" / "interim" / "dblp_papers.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "corpus.csv"

SS_DOI_URL   = "https://api.semanticscholar.org/graph/v1/paper/DOI:{}"
SS_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "paperId,title,abstract,year,venue"
DELAY  = 3   # seconds between requests — conservative without API key


def fetch_by_doi(doi: str) -> dict | None:
    """Look up a paper on Semantic Scholar by DOI."""
    if not doi:
        return None
    try:
        r = requests.get(
            SS_DOI_URL.format(doi),
            params={"fields": FIELDS},
            timeout=20,
        )
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def fetch_by_title(title: str) -> dict | None:
    """Look up a paper on Semantic Scholar by title search."""
    try:
        r = requests.get(
            SS_SEARCH_URL,
            params={"query": title, "limit": 1, "fields": FIELDS},
            timeout=20,
        )
        if r.status_code == 200:
            results = r.json().get("data", [])
            if results:
                return results[0]
        return None
    except Exception:
        return None


def main():
    print(f"Loading DBLP papers from {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} papers")

    abstracts   = []
    paper_ids   = []
    found       = 0
    not_found   = 0

    for i, row in df.iterrows():
        print(f"  [{i+1}/{len(df)}] {row['title'][:60]}...")

        result = None

        # try DOI first — most reliable
        if pd.notna(row.get("doi")) and row["doi"]:
            result = fetch_by_doi(row["doi"])
            time.sleep(DELAY)

        # fall back to title search
        if result is None:
            result = fetch_by_title(row["title"])
            time.sleep(DELAY)

        if result and result.get("abstract"):
            abstracts.append(result["abstract"])
            paper_ids.append(result.get("paperId", ""))
            found += 1
        else:
            abstracts.append("")
            paper_ids.append("")
            not_found += 1

        print(f"    found: {found}  not found: {not_found}")

    df["abstract"] = abstracts
    df["paper_id"] = paper_ids

    # drop papers where we couldn't get an abstract
    df_clean = df[df["abstract"].str.len() > 100].copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone! Saved {len(df_clean)} papers with abstracts to {OUTPUT_PATH}")
    print(f"Could not find abstracts for {not_found} papers")


if __name__ == "__main__":
    main()
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

# Resolve the project root two levels up so paths work regardless of the
# working directory the script is launched from.
REPO_ROOT   = Path(__file__).resolve().parents[2]
INPUT_PATH  = REPO_ROOT / "data" / "interim" / "dblp_papers.csv"   # output of collect_dblp.py
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "corpus.csv"       # final enriched corpus

# Semantic Scholar API endpoints.
# DOI lookup is an exact match; search is a fuzzy title query.
SS_DOI_URL   = "https://api.semanticscholar.org/graph/v1/paper/DOI:{}"
SS_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Only request the fields we actually use — keeps payloads small.
FIELDS = "paperId,title,abstract,year,venue"

# 3 seconds between requests is conservative for the unauthenticated rate limit
# (~100 req/min). Increase if you add an API key.
DELAY  = 3


def fetch_by_doi(doi: str) -> dict | None:
    """
    Look up a paper on Semantic Scholar by DOI.

    DOI lookup is the preferred method because it is an exact identifier match
    and never returns the wrong paper. Returns the full API response dict on
    success, or None if the DOI is missing, unrecognised, or the request fails.
    """
    if not doi:
        return None
    try:
        r = requests.get(
            SS_DOI_URL.format(doi),   # DOI is interpolated directly into the URL path
            params={"fields": FIELDS},
            timeout=20,
        )
        if r.status_code == 200:
            return r.json()
        # 404 = DOI not in Semantic Scholar; any other non-200 is also a miss.
        return None
    except Exception:
        return None


def fetch_by_title(title: str) -> dict | None:
    """
    Look up a paper on Semantic Scholar by title search.

    Used as a fallback when no DOI is available. Requests only the top result
    (limit=1) — callers should be aware this can return a different paper if
    the title is ambiguous or the DBLP title differs slightly from S2's index.
    Returns the first result dict, or None on failure or empty results.
    """
    try:
        r = requests.get(
            SS_SEARCH_URL,
            params={"query": title, "limit": 1, "fields": FIELDS},
            timeout=20,
        )
        if r.status_code == 200:
            results = r.json().get("data", [])
            if results:
                return results[0]   # take the top-ranked match
        return None
    except Exception:
        return None


def main():
    print(f"Loading DBLP papers from {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} papers")

    abstracts   = []   # parallel list — one entry per row in df
    paper_ids   = []   # Semantic Scholar internal IDs, useful for cross-referencing
    found       = 0
    not_found   = 0

    for i, row in df.iterrows():
        # Print a progress indicator with a truncated title so the log stays readable.
        print(f"  [{i+1}/{len(df)}] {row['title'][:60]}...")

        result = None

        # Try DOI first — most reliable identifier, avoids false title matches.
        if pd.notna(row.get("doi")) and row["doi"]:
            result = fetch_by_doi(row["doi"])
            time.sleep(DELAY)   # rate-limit pause after every API call

        # Fall back to title search only if DOI lookup produced nothing.
        if result is None:
            result = fetch_by_title(row["title"])
            time.sleep(DELAY)

        # Only record a hit if we actually got an abstract — an empty abstract
        # is as useless as no result for the downstream classifier.
        if result and result.get("abstract"):
            abstracts.append(result["abstract"])
            paper_ids.append(result.get("paperId", ""))
            found += 1
        else:
            # Append empty strings to keep the lists aligned with df rows.
            abstracts.append("")
            paper_ids.append("")
            not_found += 1

        print(f"    found: {found}  not found: {not_found}")

    # Attach the new columns to the original DataFrame so all DBLP metadata
    # (title, year, venue, doi, url) is preserved alongside the abstract.
    df["abstract"] = abstracts
    df["paper_id"] = paper_ids

    # Drop rows whose abstract is too short to be useful (stubs, withdrawn
    # papers, or title-search mismatches often return near-empty strings).
    df_clean = df[df["abstract"].str.len() > 100].copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone! Saved {len(df_clean)} papers with abstracts to {OUTPUT_PATH}")
    print(f"Could not find abstracts for {not_found} papers")


if __name__ == "__main__":
    main()

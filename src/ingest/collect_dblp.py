"""
Collect Technical Debt papers from DBLP.
Uses per-query checkpoint files so partial progress survives restarts.
If a query's JSON file already exists, it skips that query entirely.

Run from project root:
    python src/ingest/collect_dblp.py
"""

import json
import time
import requests
import pandas as pd
from pathlib import Path

# Resolve project root two levels up from this file so paths are stable
# regardless of where the script is invoked from.
REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_CONF_PATH    = REPO_ROOT / "data" / "raw" / "core_rankings.csv"
CORE_JOURNAL_PATH = REPO_ROOT / "data" / "raw" / "core_journal_rankings.csv"
RAW_DIR    = REPO_ROOT / "data" / "raw" / "dblp"         # checkpoint folder
OUTPUT_PATH = REPO_ROOT / "data" / "interim" / "dblp_papers.csv"

# Search terms sent to DBLP. Covering synonyms (SATD, techdebt, debt types)
# maximizes recall across the technical-debt literature.
QUERIES = [
    "technical debt",
    "architectural debt",
    "code debt",
    "design debt",
    "test debt",
    "documentation debt",
    "self-admitted technical debt",
    "satd",
    "requirements debt",
    "infrastructure debt",
    "build debt",
    "ML technical debt",
    "machine learning technical debt",
    "data debt",
    "process debt",
    "social debt",
    "people debt",
    "test smell",
    "architecture erosion",
    "software erosion",
    "techdebt",
]

MIN_YEAR = 2010  # discard papers older than this — pre-2010 coverage is sparse
DBLP_URL = "https://dblp.org/search/publ/api"
DELAY_BETWEEN_PAGES   = 5   # seconds between pages within a query
DELAY_BETWEEN_QUERIES = 15  # seconds between queries


# ── CORE VENUES ───────────────────────────────────────────────────────────────

def load_core_venues() -> set[str]:
    """
    Read the CORE conference and journal ranking CSVs and return a set of
    lowercase venue names/acronyms that are ranked A or A*.

    Both the full title and the short acronym are added to the set so that
    is_quality_venue() can match whichever form DBLP returns.
    """
    venues = set()
    for path in (CORE_CONF_PATH, CORE_JOURNAL_PATH):
        if not path.exists():
            print(f"  Missing: {path.name}")
            continue
        try:
            # No header row — column positions are fixed by the CORE CSV format.
            df = pd.read_csv(path, dtype=str, header=None)
            for _, row in df.iterrows():
                # Column 4 holds the rank; fall back to the last column for
                # files that have a slightly different layout.
                rank = str(row.iloc[4]).strip() if len(row) > 4 else str(row.iloc[-1]).strip()
                if rank not in ("A", "A*"):
                    continue  # skip B/C/unranked venues

                # Column 1 is the full venue title, column 2 is the acronym.
                title = str(row.iloc[1]).strip().lower()
                if title:
                    venues.add(title)
                if len(row) > 2:
                    acronym = str(row.iloc[2]).strip().lower()
                    if acronym and acronym != "nan":
                        venues.add(acronym)
        except Exception as e:
            print(f"  Error reading {path.name}: {e}")
    print(f"Loaded {len(venues)} A/A* venue names from CORE")
    return venues


def is_quality_venue(venue: str, core_venues: set[str]) -> bool:
    """
    Return True if the given venue string matches any A/A* CORE venue.

    Uses substring matching in both directions so partial names (e.g.
    "ICSE 2023" matching "icse") are caught. Preprint servers are explicitly
    excluded because they are not peer-reviewed venues.
    """
    v = venue.lower().strip()
    if v in ("corr", "arxiv"):      # exclude preprints
        return False
    # Check whether the known venue name appears inside the DBLP string, or
    # vice-versa, to handle year suffixes and slight name variations.
    for known in core_venues:
        if known in v or v in known:
            return True
    return False


# ── DBLP FETCHER WITH CHECKPOINTING ──────────────────────────────────────────

def load_or_fetch(query: str) -> list[dict]:
    """
    Return cached DBLP results if available, otherwise fetch and cache.

    Each query is saved as its own JSON file under RAW_DIR. If that file
    already exists the function returns its contents immediately, skipping
    all network traffic. This means the script can be interrupted and
    restarted without re-fetching completed queries.
    """
    # Convert spaces to underscores so the filename is shell-safe.
    safe_name = query.replace(" ", "_")
    cache_path = RAW_DIR / f"{safe_name}.json"

    # if already fetched, load from disk and skip API call
    if cache_path.exists():
        print(f"Skipping (cached): {query} -> {cache_path.name}")
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return data

    print(f"Fetching: '{query}'")
    papers = []

    # DBLP's API pages results in blocks of 100; iterate up to 1 000 results.
    for start in range(0, 1000, 100):
        params = {
            "q":      query,
            "format": "json",
            "h":      100,   # hits per page (DBLP max)
            "f":      start, # first result index for this page
        }
        try:
            r = requests.get(DBLP_URL, params=params, timeout=20)
            if r.status_code == 429:
                # DBLP is rate-limiting us; save what we have and stop this query
                # rather than hammering the server.
                print(f"  Rate limited — saving progress and stopping this query")
                break
            if r.status_code != 200:
                print(f"  Error {r.status_code} — stopping this query")
                break
            hits = r.json().get("result", {}).get("hits", {})
            items = hits.get("hit", [])
            if not items:
                break  # no more results for this query
            papers.extend(items)
            print(f"  Page {start//100 + 1}: +{len(items)} (total {len(papers)})")
            time.sleep(DELAY_BETWEEN_PAGES)
            # A page with fewer than 100 items means we've reached the last page.
            if len(items) < 100:
                break
        except Exception as e:
            print(f"  Request failed: {e} — saving progress")
            break

    # save whatever we got — even partial results are cached so a later run
    # won't re-fetch and will instead start from the next uncached query.
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(papers, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Cached {len(papers)} raw hits to {cache_path.name}")
    return papers


# ── PARSE + FILTER ────────────────────────────────────────────────────────────

def parse_paper(hit: dict) -> dict | None:
    """
    Extract and clean fields from a single DBLP hit dict.

    Returns None for records that are missing a title or venue, or that
    predate MIN_YEAR — these are not useful for the classifier corpus.
    DBLP occasionally returns lists instead of strings for title and venue;
    both cases are normalized here.
    """
    info  = hit.get("info", {})
    title = info.get("title", "")
    year  = info.get("year", "")
    venue = info.get("venue", "")
    doi   = info.get("doi", "")
    url   = info.get("url", "")

    # venue can come back as a list — take the first item
    if isinstance(venue, list):
        venue = venue[0] if venue else ""

    # title can also be a list sometimes
    if isinstance(title, list):
        title = title[0] if title else ""

    # Drop records with no title or venue — they can't be used.
    if not title or not venue:
        return None
    try:
        if int(year) < MIN_YEAR:
            return None
    except (ValueError, TypeError):
        # year field is missing or non-numeric; discard the record.
        return None

    return {
        "title": title.strip(),
        "year":  year,
        "venue": venue.strip(),
        "doi":   doi.strip(),
        "url":   url.strip(),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading CORE A/A* venues...")
    core_venues = load_core_venues()

    all_papers  = []
    seen_titles = set()  # tracks lowercase titles to deduplicate across queries

    for query in QUERIES:
        # load from cache or fetch from DBLP; either way returns raw hit dicts
        hits = load_or_fetch(query)

        added = 0
        for hit in hits:
            paper = parse_paper(hit)
            if paper is None:
                continue
            # Deduplicate by normalized title so the same paper found via
            # multiple queries (e.g. "technical debt" and "satd") is only
            # counted once.
            title_key = paper["title"].lower().strip()
            if title_key in seen_titles:
                continue
            # Keep only papers published in A or A* venues.
            if not is_quality_venue(paper["venue"], core_venues):
                continue
            seen_titles.add(title_key)
            all_papers.append(paper)
            added += 1

        print(f"  Added {added} A/A* papers from '{query}' (running total {len(all_papers)})")
        # Pause between queries to respect DBLP's rate limit guidelines.
        time.sleep(DELAY_BETWEEN_QUERIES)

    # Build the final DataFrame; provide explicit columns if no papers were found
    # so downstream code always gets the expected schema.
    df = pd.DataFrame(all_papers) if all_papers else pd.DataFrame(
        columns=["title", "year", "venue", "doi", "url"]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone! Saved {len(df)} A/A* papers to {OUTPUT_PATH}")

    if len(df) > 0:
        print("\nTop venues:")
        print(df["venue"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()

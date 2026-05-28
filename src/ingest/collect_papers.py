"""Collect Technical Debt papers from Semantic Scholar.

Uses the /paper/search/bulk endpoint (up to 1000 results per call) with
tenacity-based exponential backoff for rate limits, and per-query
checkpoint files so partial progress survives restarts.

Run from anywhere:
    python src/ingest/collect_papers.py
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

QUERIES = [
    "technical debt",
    "architectural debt",
    "code debt",
    "design debt",
    "test debt",
    "documentation debt",
]

# The bulk endpoint returns up to 1000 results per call, making it more
# efficient than the standard search endpoint which caps at 100 per page.
BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
FIELDS = "paperId,title,abstract,year,venue"
# Verifying the user who is using the api, via User-Agent.
USER_AGENT = "se-paper-type-classifier/0.1 (mailto:maung.w@northeastern.edu)"
# Waiting 60 seconds for a response before giving up on a single request
# Waiting 5 seconds between API calls to avoid overwhelming the server
REQUEST_TIMEOUT = 60
POLITE_DELAY_SEC = 5
  
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "semantic_scholar"
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "technical_debt_corpus.csv"


class RateLimited(Exception):
    """Raised on HTTP 429 so tenacity retries the call.
    Using a custom exception lets tenacity specifically target rate limit
    errors for retry, while other errors (like 404) fail immediately.
    """

# Automatically retries the function if it raises RateLimited, ConnectionError, or Timeout. 
# It waits longer between each attempt (30s, 60s, 120s...) up to a maximum of 480 seconds, 
# and gives up after 5 attempts total.
@retry(
    retry=retry_if_exception_type(
        (RateLimited, requests.exceptions.ConnectionError, requests.exceptions.Timeout)
    ),
    wait=wait_exponential(multiplier=30, min=30, max=480),
    stop=stop_after_attempt(5),
    reraise=True,
)

def _request_bulk(query: str, token: str | None) -> dict:
    """Make a single request to the Semantic Scholar bulk search endpoint.
    
    Args:
        query: The search term to look up (e.g. "technical debt")
        token: Pagination token from the previous response, or None for
               the first page.
    
    Returns:
        The parsed JSON response as a dictionary containing "data" (list
        of papers) and optionally "token" (for the next page).
    
    Raises:
        RateLimited: If the API returns HTTP 429, triggering a retry.
    """
    
    params: dict[str, str] = {"query": query, "fields": FIELDS}
    if token:
        params["token"] = token
    response = requests.get(
        BULK_URL,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )

    # If rate limited, run the server's Retry-After header if present
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                time.sleep(int(retry_after))
            except ValueError:
                pass
        raise RateLimited(f"429 from Semantic Scholar (query={query!r})")
    response.raise_for_status()
    return response.json()


def fetch_query(query: str) -> list[dict]:
    """Fetch ALL pages of results for a single search query.
    
    Semantic Scholar paginates bulk results using a cursor token. This
    function keeps requesting the next page until there are no more results.
    
    Args:
        query: The search term to look up.
    
    Returns:
        A flat list of all paper dictionaries across all pages.
    """
    papers: list[dict] = []
    token: str | None = None
    page = 0
    while True:
        page += 1
        print(f"  Page {page}: requesting...")
        data = _request_bulk(query, token)
        batch = data.get("data", []) or []
        papers.extend(batch)
        print(f"  Page {page}: +{len(batch)} (running total {len(papers)})")
        token = data.get("token")
        if not token or not batch:
            break
        time.sleep(POLITE_DELAY_SEC)
    return papers


def load_or_fetch(query: str) -> list[dict]:
    """Return papers for a query, using a cached file if available.
    
    Each query's results are saved as a JSON file in data/raw/semantic_scholar/.
    If the file already exists, we skip the API call and load from disk instead.
    This means if the script crashes halfway through, it won't re-fetch
    queries that already completed.
    
    Args:
        query: The search term to look up.
    
    Returns:
        A list of paper dictionaries for this query."""
    
    safe_name = query.replace(" ", "_")
    cache_path = RAW_DIR / f"{safe_name}.json"
    if cache_path.exists():
        print(f"Skipping (cached): {query} -> {cache_path.name}")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    print(f"Fetching: {query}")
    try:
        papers = fetch_query(query)
    except RetryError as exc:
        print(f"  Gave up on {query!r} after retries: {exc}")
        return []

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(papers, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Cached {len(papers)} papers to {cache_path.name}")
    return papers


def main() -> None:
    """Run the full collection pipeline and save the final CSV.
    
    Loops through all queries, deduplicates papers across queries
    (a paper matching both "technical debt" and "code debt" only
    appears once), cleans the data, and saves to a CSV file.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_papers: list[dict] = []
    seen_ids: set[str] = set()

    for query in QUERIES:
        papers = load_or_fetch(query)
        for paper in papers:
            pid = paper.get("paperId")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_papers.append(paper)
        time.sleep(POLITE_DELAY_SEC)

    rows = [
        {
            "paper_id": p.get("paperId", ""),
            "title": p.get("title", ""),
            "abstract": p.get("abstract", ""),
            "year": p.get("year", ""),
            "venue": p.get("venue", ""),
        }
        for p in all_papers
    ]
    df = pd.DataFrame(rows)
    df = df[df["abstract"].notna() & (df["abstract"].str.len() > 100)]
    df = df.drop_duplicates(subset=["paper_id"])
    df.to_csv(CORPUS_PATH, index=False)
    print(f"Done! Saved {len(df)} papers to {CORPUS_PATH}")


if __name__ == "__main__":
    main()

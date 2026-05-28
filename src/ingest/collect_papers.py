"""Collect Technical Debt papers from Semantic Scholar.

Uses the /paper/search/bulk endpoint (up to 1000 results per call) with
tenacity-based exponential backoff for rate limits, and per-query
checkpoint files so partial progress survives restarts.

Run from anywhere:
    python src/ingest/collect_papers.py

Cleaning, dedup, and CORE filtering live in `clean_corpus.py`.
"""

import json
import time
from pathlib import Path

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

BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
FIELDS = "paperId,title,abstract,year,venue"
USER_AGENT = "se-paper-type-classifier/0.1 (mailto:maung.w@northeastern.edu)"
REQUEST_TIMEOUT = 60
POLITE_DELAY_SEC = 5

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "semantic_scholar"


class RateLimited(Exception):
    """Raised on HTTP 429 so tenacity retries the call."""


@retry(
    retry=retry_if_exception_type(
        (RateLimited, requests.exceptions.ConnectionError, requests.exceptions.Timeout)
    ),
    wait=wait_exponential(multiplier=30, min=30, max=480),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _request_bulk(query: str, token: str | None) -> dict:
    params: dict[str, str] = {"query": query, "fields": FIELDS}
    if token:
        params["token"] = token
    response = requests.get(
        BULK_URL,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
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
    """Page through all bulk results for a single query."""
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
    """Return cached results from disk if present, otherwise fetch and cache."""
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
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for query in QUERIES:
        load_or_fetch(query)
        time.sleep(POLITE_DELAY_SEC)
    print(f"Done. Cached {len(QUERIES)} query results in {RAW_DIR}")


if __name__ == "__main__":
    main()

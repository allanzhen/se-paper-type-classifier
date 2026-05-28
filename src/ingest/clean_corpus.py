"""Clean the raw Technical Debt corpus.

Reads per-query JSON caches in data/raw/semantic_scholar/, applies dedup +
abstract-quality + year + CORE A/A* venue filters, writes the cleaned corpus
to data/processed/corpus.csv.

Requires the CORE rankings to be downloaded manually first:
- data/raw/core_rankings.csv         (CORE conference rankings)
- data/raw/core_journal_rankings.csv (CORE journal rankings; optional)
"""

import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_JSON_DIR = REPO_ROOT / "data" / "raw" / "semantic_scholar"
CORE_CONF_PATH = REPO_ROOT / "data" / "raw" / "core_rankings.csv"
CORE_JOURNAL_PATH = REPO_ROOT / "data" / "raw" / "core_journal_rankings.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "corpus.csv"

MIN_YEAR = 2010
MIN_ABSTRACT_LEN = 100
KEPT_RANKS = {"A", "A*"}

# Order matters: longer / more-specific prefixes come first so that
# `startswith` strips them before generic substrings like "ieee".
_NORMALISE_PREFIXES = [
    "ieee/acm international working conference on",
    "acm/ieee international working conference on",
    "ieee international working conference on",
    "acm international working conference on",
    "international working conference on",
    "ieee/acm international conference on",
    "acm/ieee international conference on",
    "ieee international conference on",
    "acm international conference on",
    "international conference on",
    "ieee/acm international symposium on",
    "ieee international symposium on",
    "acm international symposium on",
    "international symposium on",
    "international workshop on",
    "working conference on",
    "proceedings of the",
    "proceedings of",
    "proc. of the",
    "proc. of",
    "proceedings",
    "acm sigsoft",
    "ieee",
    "acm",
]
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_ORDINAL_RE = re.compile(r"\b\d+(?:st|nd|rd|th)\b")
_PAREN_RE = re.compile(r"\s*\([^)]*\)")
_WHITESPACE_RE = re.compile(r"\s+")


def normalise(venue: str | None) -> str:
    """Normalise a venue string so the same conference matches across sources."""
    if not venue or not isinstance(venue, str):
        return ""
    s = venue.lower().strip()
    s = s.replace("&", "and")
    s = _YEAR_RE.sub("", s)
    s = _ORDINAL_RE.sub("", s)
    s = _PAREN_RE.sub("", s)  # drop "(TechDebt)", "(was ESEC/FSE)", etc.
    s = s.split(":")[0]  # drop subtitles, e.g. "EMSE: an international journal"
    for prefix in _NORMALISE_PREFIXES:
        if s.startswith(prefix + " "):
            s = s[len(prefix) + 1 :]
            break
    s = _WHITESPACE_RE.sub(" ", s).strip(" ,.;:-")
    return s


def load_raw_papers() -> list[dict]:
    """Concatenate all per-query JSON caches into one list."""
    papers: list[dict] = []
    for path in sorted(RAW_JSON_DIR.glob("*.json")):
        batch = json.loads(path.read_text(encoding="utf-8"))
        papers.extend(batch)
        print(f"  Loaded {len(batch):>6} papers from {path.name}")
    print(f"  Total before dedup: {len(papers)}")
    return papers


def _load_core_csv(path: Path) -> list[tuple[str, str | None, str]]:
    """Return [(title, acronym_or_None, rank), ...] from a CORE export.

    The conference export is headerless with column order
        0=ID, 1=Title, 2=Acronym, 3=Source, 4=Rank, ...
    The journal export has a header row and columns
        id, title, source, rank, has changed?, for1, ...   (no acronym column)
    We detect which format we're looking at by sniffing the first cell.
    """
    raw = pd.read_csv(path, header=None, dtype=str)
    first_cell = str(raw.iloc[0, 0]).strip().lower()
    has_header = first_cell in {"id", "title"}
    if has_header:
        df = pd.read_csv(path, dtype=str)
        cols = {c.lower(): c for c in df.columns}
        title = df[cols["title"]]
        acronym = df[cols["acronym"]] if "acronym" in cols else None
        rank = df[cols["rank"]]
    else:
        df = raw
        title = df[1]
        acronym = df[2] if df.shape[1] > 2 else None
        rank = df[4] if df.shape[1] > 4 else df.iloc[:, -1]

    rows: list[tuple[str, str | None, str]] = []
    for i in range(len(df)):
        t = title.iloc[i]
        a = acronym.iloc[i] if acronym is not None else None
        r = rank.iloc[i]
        if pd.isna(t) or pd.isna(r):
            continue
        rows.append(
            (
                str(t),
                None if a is None or pd.isna(a) else str(a),
                str(r).strip(),
            )
        )
    return rows


_RANK_PRIORITY = {"A*": 4, "A": 3, "B": 2, "C": 1}


def load_core_rankings() -> dict[str, str]:
    """Build {normalised_name: rank} from CORE conference and journal CSVs.

    On key collision (multiple CORE entries normalise to the same string),
    keep the highest rank so that an A/A* venue is never masked by a lower-
    ranked one that happened to be loaded first.
    """
    mapping: dict[str, str] = {}
    for path in (CORE_CONF_PATH, CORE_JOURNAL_PATH):
        if not path.exists():
            print(f"  Skipping (not found): {path.name}")
            continue
        rows = _load_core_csv(path)
        for title, acronym, rank in rows:
            new_p = _RANK_PRIORITY.get(rank, 0)
            for candidate in (title, acronym):
                if not candidate:
                    continue
                key = normalise(candidate)
                if not key:
                    continue
                old_p = _RANK_PRIORITY.get(mapping.get(key, ""), 0)
                if new_p > old_p:
                    mapping[key] = rank
        print(f"  Loaded {len(rows):>5} venues from {path.name}")
    print(f"  Total CORE venues in lookup: {len(mapping)}")
    return mapping


def main() -> None:
    print("Loading raw papers...")
    raw = load_raw_papers()

    df = pd.DataFrame(raw)
    print("\nDedup by paperId:")
    print(f"  Before: {len(df)}")
    df = df.drop_duplicates(subset=["paperId"]).reset_index(drop=True)
    print(f"  After:  {len(df)}")

    print(f"\nAbstract filter (non-null, length > {MIN_ABSTRACT_LEN}):")
    df = df[df["abstract"].notna() & (df["abstract"].str.len() > MIN_ABSTRACT_LEN)]
    print(f"  After:  {len(df)}")

    print(f"\nYear filter (>= {MIN_YEAR}):")
    df = df[df["year"].notna() & (df["year"] >= MIN_YEAR)]
    print(f"  After:  {len(df)}")

    print("\nVenue normalisation + CORE A/A* filter:")
    core_lookup = load_core_rankings()
    if not core_lookup:
        raise SystemExit(
            "No CORE rankings loaded. Download core_rankings.csv "
            "(and optionally core_journal_rankings.csv) from "
            "https://portal.core.edu.au/conf-ranks/ into data/raw/ first."
        )
    df["normalised_venue"] = df["venue"].apply(normalise)
    df["core_rank"] = df["normalised_venue"].map(core_lookup)
    df = df[df["core_rank"].isin(KEPT_RANKS)]
    print(f"  After:  {len(df)}")

    print("\nFinal distribution by rank:")
    print(df["core_rank"].value_counts().to_string())
    print("\nFinal distribution by year:")
    print(df["year"].astype(int).value_counts().sort_index().to_string())

    out = df.rename(columns={"paperId": "paper_id"})[
        [
            "paper_id",
            "title",
            "abstract",
            "year",
            "venue",
            "normalised_venue",
            "core_rank",
        ]
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(out)} papers to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

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

# Resolve the project root two levels up so paths are stable regardless of
# where the script is invoked from.
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_JSON_DIR = REPO_ROOT / "data" / "raw" / "semantic_scholar"  # per-query cache files
CORE_CONF_PATH = REPO_ROOT / "data" / "raw" / "core_rankings.csv"
CORE_JOURNAL_PATH = REPO_ROOT / "data" / "raw" / "core_journal_rankings.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "corpus.csv"

MIN_YEAR = 2010          # discard papers published before this year
MIN_ABSTRACT_LEN = 100   # abstracts shorter than this are stubs or errors
KEPT_RANKS = {"A", "A*"} # only keep papers from top-tier CORE venues

# Boilerplate prefixes that CORE and Semantic Scholar attach to venue names
# but that don't distinguish one conference from another. Longer/more-specific
# prefixes come first so a greedy strip doesn't consume only part of a phrase
# and leave a broken remainder (e.g. "ieee" must not strip before
# "ieee international conference on" gets a chance to match).
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

# Pre-compiled regexes used by normalise() — compiled once at import time for speed.
_YEAR_RE            = re.compile(r"\b(?:19|20)\d{2}\b")       # e.g. "2023", "1998"
_ORDINAL_RE         = re.compile(r"\b\d+(?:st|nd|rd|th)\b")   # e.g. "42nd", "1st"
_PAREN_RE           = re.compile(r"\s*\([^)]*\)")              # any parenthetical block
_PARENS_CAPTURE_RE  = re.compile(r"\(([^)]+)\)")               # content inside parens
_TRAILING_DIGITS_RE = re.compile(r"\d+$")                      # digits at end of string
_WHITESPACE_RE      = re.compile(r"\s+")                       # runs of whitespace


def normalise(venue: str | None) -> str:
    """
    Normalise a venue string so the same conference matches across sources.

    CORE and Semantic Scholar use slightly different names for the same venue
    (year suffixes, ordinals, subtitles, boilerplate prefixes). This function
    strips all of that so both reduce to the same canonical core name that can
    be matched against the CORE lookup table.
    """
    if not venue or not isinstance(venue, str):
        return ""
    s = venue.lower().strip()
    s = s.replace("&", "and")
    s = s.replace(",", " ")      # drop Oxford-comma differences (CORE vs Semantic Scholar)
    s = _YEAR_RE.sub("", s)      # strip edition years like "ICSE 2023"
    s = _ORDINAL_RE.sub("", s)   # strip edition numbers like "42nd"
    s = _PAREN_RE.sub("", s)     # drop "(TechDebt)", "(was ESEC/FSE)", etc.
    s = s.split(":")[0]          # drop subtitles, e.g. "EMSE: an international journal"

    # Collapse whitespace BEFORE prefix matching: the substitutions above can
    # leave double-spaces (e.g. "42nd" stripped from the middle), which would
    # break startswith() against a single-space prefix.
    s = _WHITESPACE_RE.sub(" ", s).strip()

    # Strip every matching prefix, restarting from the top of the list each
    # time so a later strip can expose an earlier one (e.g. after "ieee" is
    # stripped, "international conference on" may now be the new leading run).
    while True:
        matched = False
        for prefix in _NORMALISE_PREFIXES:
            if s.startswith(prefix + " "):
                s = s[len(prefix) + 1:]
                matched = True
                break   # restart from the top of the prefix list
        if not matched:
            break
    return s.strip(" ,.;:-")


def extract_acronym(venue: str | None) -> str:
    """
    Pull a likely acronym out of a parenthetical, e.g. "...(ICSE)" -> "icse".

    Filters out long parentheticals like "(was ESEC/FSE, changed 2024)" — only
    accepts short, space-free candidates with at least one uppercase letter.
    This gives a second lookup key when the full title doesn't match CORE.
    """
    if not venue or not isinstance(venue, str):
        return ""
    for raw in _PARENS_CAPTURE_RE.findall(venue):
        # Strip trailing edition numbers like "ICSE2023" -> "ICSE" before checking.
        candidate = _TRAILING_DIGITS_RE.sub("", raw).strip()
        if (
            2 <= len(candidate) <= 12     # too short = noise, too long = not an acronym
            and " " not in candidate       # acronyms have no spaces
            and any(c.isupper() for c in candidate)  # must contain at least one capital
        ):
            return candidate.lower()
    return ""


def load_raw_papers() -> list[dict]:
    """
    Concatenate all per-query JSON caches into one list.

    Each file under RAW_JSON_DIR corresponds to a single Semantic Scholar
    query. Loading them all here gives one combined pool before deduplication.
    """
    papers: list[dict] = []
    for path in sorted(RAW_JSON_DIR.glob("*.json")):
        batch = json.loads(path.read_text(encoding="utf-8"))
        papers.extend(batch)
        print(f"  Loaded {len(batch):>6} papers from {path.name}")
    print(f"  Total before dedup: {len(papers)}")
    return papers


def _load_core_csv(path: Path) -> list[tuple[str, str | None, str]]:
    """
    Return [(title, acronym_or_None, rank), ...] from a CORE export CSV.

    CORE ships two differently-structured CSVs:
      - Conference export: headerless, columns 0=ID, 1=Title, 2=Acronym,
        3=Source, 4=Rank, ...
      - Journal export: has a header row with columns id, title, source,
        rank, has changed?, for1, ... (no acronym column)

    We sniff the first cell to detect which format we're reading so the same
    function handles both files without separate code paths.
    """
    # Read without a header first to inspect the first cell.
    raw = pd.read_csv(path, header=None, dtype=str)
    first_cell = str(raw.iloc[0, 0]).strip().lower()
    has_header = first_cell in {"id", "title"}   # journal export has a real header row

    if has_header:
        # Re-read with the header so column names are available.
        df = pd.read_csv(path, dtype=str)
        cols = {c.lower(): c for c in df.columns}   # case-insensitive column lookup
        title   = df[cols["title"]]
        acronym = df[cols["acronym"]] if "acronym" in cols else None
        rank    = df[cols["rank"]]
    else:
        # Conference export — use fixed column positions.
        df      = raw
        title   = df[1]
        acronym = df[2] if df.shape[1] > 2 else None
        rank    = df[4] if df.shape[1] > 4 else df.iloc[:, -1]

    rows: list[tuple[str, str | None, str]] = []
    for i in range(len(df)):
        t = title.iloc[i]
        a = acronym.iloc[i] if acronym is not None else None
        r = rank.iloc[i]
        if pd.isna(t) or pd.isna(r):
            continue   # skip rows with no title or rank
        rows.append(
            (
                str(t),
                None if a is None or pd.isna(a) else str(a),
                str(r).strip(),
            )
        )
    return rows


# Maps CORE rank letter to a numeric priority for collision resolution.
_RANK_PRIORITY = {"A*": 4, "A": 3, "B": 2, "C": 1}


def load_core_rankings() -> dict[str, str]:
    """
    Build {normalised_name: rank} from CORE conference and journal CSVs.

    Both the full title and the short acronym of each venue are normalised and
    added as keys so either form can be matched at lookup time. On key collision
    (multiple entries normalise to the same string), the highest rank wins —
    this prevents a lower-ranked duplicate from masking an A/A* venue.
    """
    mapping: dict[str, str] = {}
    for path in (CORE_CONF_PATH, CORE_JOURNAL_PATH):
        if not path.exists():
            print(f"  Skipping (not found): {path.name}")
            continue
        rows = _load_core_csv(path)
        for title, acronym, rank in rows:
            new_p = _RANK_PRIORITY.get(rank, 0)
            # Index both the full title and the acronym so either can match.
            for candidate in (title, acronym):
                if not candidate:
                    continue
                key = normalise(candidate)
                if not key:
                    continue
                # Keep the higher rank on collision — never downgrade a venue.
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

    # ── Step 1: Deduplicate by Semantic Scholar paper ID ──────────────────────
    # The same paper can appear in multiple per-query JSON files; paperId is
    # the canonical identifier that collapses them safely.
    print("\nDedup by paperId:")
    print(f"  Before: {len(df)}")
    df = df.drop_duplicates(subset=["paperId"]).reset_index(drop=True)
    print(f"  After:  {len(df)}")

    # ── Step 2: Abstract quality filter ───────────────────────────────────────
    # Very short abstracts are usually stubs, withdrawn-paper notices, or
    # title-search mismatches rather than real scientific abstracts.
    print(f"\nAbstract filter (non-null, length > {MIN_ABSTRACT_LEN}):")
    df = df[df["abstract"].notna() & (df["abstract"].str.len() > MIN_ABSTRACT_LEN)]
    print(f"  After:  {len(df)}")

    # ── Step 3: Year filter ───────────────────────────────────────────────────
    # Pre-2010 technical-debt literature is sparse and inconsistently indexed.
    print(f"\nYear filter (>= {MIN_YEAR}):")
    df = df[df["year"].notna() & (df["year"] >= MIN_YEAR)]
    print(f"  After:  {len(df)}")

    # ── Step 4: CORE A/A* venue filter ────────────────────────────────────────
    print("\nVenue normalisation + CORE A/A* filter:")
    core_lookup = load_core_rankings()
    if not core_lookup:
        raise SystemExit(
            "No CORE rankings loaded. Download core_rankings.csv "
            "(and optionally core_journal_rankings.csv) from "
            "https://portal.core.edu.au/conf-ranks/ into data/raw/ first."
        )

    # Normalise the venue string and try a direct title-based lookup first.
    df["normalised_venue"] = df["venue"].apply(normalise)
    df["_acronym"]         = df["venue"].apply(extract_acronym)
    df["core_rank"]        = df["normalised_venue"].map(core_lookup)

    # Fall back to the parenthetical acronym when the title-derived key missed.
    # fillna() only fills rows where core_rank is still NaN after the first map.
    acronym_rank = df["_acronym"].map(core_lookup)
    df["core_rank"] = df["core_rank"].fillna(acronym_rank)

    df = df.drop(columns=["_acronym"])   # temp column no longer needed
    df = df[df["core_rank"].isin(KEPT_RANKS)]
    print(f"  After:  {len(df)}")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print("\nFinal distribution by rank:")
    print(df["core_rank"].value_counts().to_string())
    print("\nFinal distribution by year:")
    print(df["year"].astype(int).value_counts().sort_index().to_string())

    # ── Write output ──────────────────────────────────────────────────────────
    # Rename paperId -> paper_id for consistency with the rest of the pipeline
    # and select only the columns downstream code depends on.
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

"""Shared fixtures: tiny in-memory corpus caches + gold files for the
integration tests of the cache-join pipeline (`*_main` tests).

The fixtures write minimal CSVs that mirror the real schemas to a tmp dir, so
tests can monkeypatch a module's PATH constants at them and run `main()` without
touching the committed data or loading any model.
"""

import json
from collections import namedtuple

import pandas as pd
import pytest

# Per-paper fixture data. One row (p5) has rule_predicted == "Unknown" so the
# hybrid is forced to fall back to zero-shot there; the rest let the rule win.
# Columns: id, title, abstract, year, venue, rule_pred, rule_conf, zs_pred, zs_conf, gold
_ROWS = [
    ("p1", "A tool",   "we present a tool",            2020, "ICSE", "Tool Paper",                   1.0, "Empirical Study", 0.60, "Tool Paper"),
    ("p2", "A study",  "we mined repositories",         2021, "MSR",  "Empirical Study",              1.0, "Empirical Study", 0.90, "Empirical Study"),
    ("p3", "A review", "systematic literature review",  2019, "EMSE", "Systematic Literature Review", 1.0, "Survey",          0.40, "Systematic Literature Review"),
    ("p4", "A survey", "we conducted a survey",         2022, "JSS",  "Survey",                       1.0, "Empirical Study", 0.50, "Survey"),
    ("p5", "A case",   "an in-depth case study",        2023, "TSE",  "Unknown",                      0.0, "Empirical Study", 0.70, "Case Study"),
    ("p6", "A pos",    "we argue for a viewpoint",      2024, "ICSE", "Position Paper",               1.0, "Position Paper",  0.80, "Position Paper"),
]

Fixtures = namedtuple(
    "Fixtures", "rule_path zs_path gold_all_path gold_subset_path out_path ids gold_subset_ids"
)


@pytest.fixture
def corpus_fixtures(tmp_path) -> Fixtures:
    """Write rule cache, zs cache, and gold CSVs; return their paths."""
    rule = pd.DataFrame(
        [(r[0], r[1], r[2], r[3], r[4], r[5], r[6], "", "") for r in _ROWS],
        columns=["paper_id", "title", "abstract", "year", "venue",
                 "predicted_type", "confidence", "unknown_reason", "matched_rules"],
    )
    zs = pd.DataFrame(
        [(r[0], r[1], r[2], r[3], r[4], r[7], r[8],
          json.dumps({r[7]: r[8], "Tool Paper": 0.05})) for r in _ROWS],
        columns=["paper_id", "title", "abstract", "year", "venue",
                 "predicted_type_zs", "confidence_zs", "zs_scores"],
    )
    gold_all = pd.DataFrame(
        [(r[0], r[1], r[2], r[9]) for r in _ROWS],
        columns=["paper_id", "title", "abstract", "manual_label"],
    )
    # Subset gold (p1, p2 only) so build_dev_set has candidates left to exclude from.
    gold_subset_ids = ["p1", "p2"]
    gold_subset = gold_all[gold_all["paper_id"].isin(gold_subset_ids)]

    rule_path = tmp_path / "corpus_labeled.csv"
    zs_path = tmp_path / "corpus_labeled_zs.csv"
    gold_all_path = tmp_path / "gold_all.csv"
    gold_subset_path = tmp_path / "gold_subset.csv"
    rule.to_csv(rule_path, index=False)
    zs.to_csv(zs_path, index=False)
    gold_all.to_csv(gold_all_path, index=False)
    gold_subset.to_csv(gold_subset_path, index=False)

    return Fixtures(
        rule_path=rule_path, zs_path=zs_path,
        gold_all_path=gold_all_path, gold_subset_path=gold_subset_path,
        out_path=tmp_path / "out.csv",
        ids=[r[0] for r in _ROWS], gold_subset_ids=gold_subset_ids,
    )

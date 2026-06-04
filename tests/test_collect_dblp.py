"""DBLP venue matching + record parsing (src/ingest/collect_dblp.py)."""

import collect_dblp as cd


def test_is_quality_venue_matches_known_acronym():
    core = {"icse"}
    assert cd.is_quality_venue("ICSE 2023", core) is True       # known inside DBLP string
    assert cd.is_quality_venue("Random Workshop", core) is False


def test_is_quality_venue_excludes_preprints():
    core = {"icse", "corr"}   # even if present, preprints are excluded
    assert cd.is_quality_venue("CoRR", core) is False
    assert cd.is_quality_venue("arXiv", core) is False


def test_parse_paper_valid_record():
    out = cd.parse_paper({"info": {"title": "T", "year": "2020", "venue": "ICSE", "doi": "d", "url": "u"}})
    assert out == {"title": "T", "year": "2020", "venue": "ICSE", "doi": "d", "url": "u"}


def test_parse_paper_drops_missing_and_old_and_nonnumeric():
    assert cd.parse_paper({"info": {"title": "T", "year": "2020"}}) is None          # no venue
    assert cd.parse_paper({"info": {"title": "", "venue": "ICSE", "year": "2020"}}) is None  # no title
    assert cd.parse_paper({"info": {"title": "T", "venue": "ICSE", "year": str(cd.MIN_YEAR - 1)}}) is None # too old
    assert cd.parse_paper({"info": {"title": "T", "venue": "ICSE", "year": "n/a"}}) is None # non-numeric year


def test_parse_paper_takes_first_when_list():
    out = cd.parse_paper({"info": {"title": ["T1", "T2"], "year": "2020", "venue": ["V1", "V2"]}})
    assert out["title"] == "T1" and out["venue"] == "V1"

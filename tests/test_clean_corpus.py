"""Venue normalisation helpers (src/ingest/clean_corpus.py)."""

import clean_corpus as cc


def test_normalise_strips_year_ordinal_prefix_and_subtitle():
    assert cc.normalise(
        "42nd IEEE International Conference on Software Engineering (ICSE) 2023"
    ) == "software engineering"


def test_normalise_maps_ampersand_and_drops_subtitle():
    assert cc.normalise("Architecture & Design: an international journal") == "architecture and design"


def test_normalise_handles_none_and_non_string():
    assert cc.normalise(None) == ""
    assert cc.normalise(123) == ""


def test_extract_acronym_from_parenthetical():
    assert cc.extract_acronym("Proceedings of the Conference (ICSE)") == "icse"


def test_extract_acronym_strips_trailing_digits():
    assert cc.extract_acronym("Some Conf (ICSE2023)") == "icse"


def test_extract_acronym_rejects_long_or_spaced_or_absent():
    assert cc.extract_acronym("(was ESEC/FSE, changed in 2024)") == ""   # too long / spaced
    assert cc.extract_acronym("No parenthetical here") == ""

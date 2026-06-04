"""Dev-label normalisation (src/evaluate/evaluate_dev.normalise_gold).

Guarded import: evaluate_dev imports zero_shot, which pulls in torch/transformers.
Skip the module cleanly if those aren't installed.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import evaluate_dev as ed  # noqa: E402


def test_abbreviation_and_typo_map_to_canonical():
    assert ed.normalise_gold("SLR") == "Systematic Literature Review"
    assert ed.normalise_gold("emperical study") == "Empirical Study"


def test_case_and_whitespace_insensitive():
    assert ed.normalise_gold("  Tool Paper  ") == "Tool Paper"
    assert ed.normalise_gold("CASE STUDY") == "Case Study"


def test_unknown_label_passes_through_stripped():
    assert ed.normalise_gold("  Secondary Study ") == "Secondary Study"


def test_non_string_returns_empty():
    assert ed.normalise_gold(float("nan")) == ""
    assert ed.normalise_gold(None) == ""

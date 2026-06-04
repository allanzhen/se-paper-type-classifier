"""The rule-first hybrid arbitration (src/evaluate/evaluate_hybrid.combine)."""

import evaluate_hybrid as eh


def test_rule_wins_when_it_fires():
    assert eh.combine("Tool Paper", 0.8, "Empirical Study", 0.6) == ("Tool Paper", 0.8, "rule")


def test_falls_back_to_zero_shot_when_rule_unknown():
    assert eh.combine("Unknown", 0.0, "Empirical Study", 0.6) == ("Empirical Study", 0.6, "zero-shot")


def test_rule_wins_even_with_lower_confidence():
    # "rule-first" is not confidence-weighted: the rule label is used whenever it fires.
    assert eh.combine("Survey", 0.4, "Empirical Study", 0.95)[2] == "rule"

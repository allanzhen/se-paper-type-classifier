"""Invariants on the rule definitions in src/classify/rules.py."""

import re

import rules

CANONICAL = {
    "Empirical Study", "Controlled Experiment", "Systematic Literature Review",
    "Survey", "Tool Paper", "Experience Report", "Case Study", "Position Paper",
    "Theoretical Contribution",
}


def test_rules_labels_are_canonical():
    assert set(rules.RULES) <= CANONICAL
    # All nine canonical classes should have a rule list.
    assert set(rules.RULES) == CANONICAL


def test_compiled_matches_rules():
    assert set(rules.COMPILED) == set(rules.RULES)
    for label, patterns in rules.RULES.items():
        compiled = rules.COMPILED[label]
        assert len(compiled) == len(patterns)
        assert all(isinstance(p, re.Pattern) for p in compiled)


def test_decision_tree_cue_lists_compiled_and_nonempty():
    for group in (rules.SECONDARY_STUDY, rules.PROTOCOL, rules.PRACTITIONER_SURVEY):
        assert group, "cue list should not be empty"
        assert all(isinstance(p, re.Pattern) for p in group)


def test_no_duplicate_patterns_within_a_label():
    for label, patterns in rules.RULES.items():
        assert len(patterns) == len(set(patterns)), f"duplicate pattern in {label}"

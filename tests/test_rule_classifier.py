"""Behaviour of the rule classifier (src/classify/rule_classifier.py)."""

import rule_classifier as rc


def label(title, abstract):
    return rc.classify(title, abstract)[0]


def test_clear_single_class_phrases():
    assert label("X", "we conduct a controlled experiment with 30 participants") == "Controlled Experiment"
    assert label("FooTool", "we present a tool that supports developers") == "Tool Paper"
    assert label("X", "an in-depth case study of one organization") == "Case Study"
    assert label("X", "we report the lessons learned applying CI in our organization") == "Experience Report"


def test_no_match_returns_unknown():
    lbl, conf, reason, hits = rc.classify("", "")
    assert lbl == "Unknown"
    assert conf == 0.0
    assert reason == "no_match"
    assert hits == []


def test_tie_returns_unknown_with_reason():
    lbl, conf, reason, hits = rc.classify("X", "this controlled experiment and this case study")
    assert lbl == "Unknown"
    assert conf == 0.0
    assert reason.startswith("tie:")
    assert "Controlled Experiment" in reason and "Case Study" in reason


def test_confidence_is_top_over_top_plus_second():
    # 2 Controlled-Experiment hits ("controlled experiment", "randomly assigned")
    # vs 1 Case Study hit -> confidence == 2/(2+1).
    lbl, conf, _, _ = rc.classify("X", "this controlled experiment used randomly assigned groups; also a case study")
    assert lbl == "Controlled Experiment"
    assert conf == 2 / 3


def test_tree_secondary_with_protocol_is_slr():
    assert label("A systematic mapping study",
                 "search string with inclusion criteria and exclusion criteria") == "Systematic Literature Review"


def test_tree_narrative_review_is_survey():
    # Narrative literature review with no protocol cues -> Survey (literature sense).
    # The keyword vote alone wouldn't fire Survey here, so this exercises the tree.
    assert label("An overview of the field",
                 "we survey the literature and review of existing studies") == "Survey"


def test_tree_practitioner_questionnaire_is_survey():
    assert label("X", "we conducted a survey; questionnaire respondents answered") == "Survey"


def test_tool_dominance_overrides_secondary_cue():
    # >=2 Tool hits + a 'literature review' secondary cue: the tree defers to the
    # keyword vote (Tool), rather than mislabelling it a secondary study.
    assert label("FooTool",
                 "we present a tool. our tool supports developers. we also did a literature review") == "Tool Paper"

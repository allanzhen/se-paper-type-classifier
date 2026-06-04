"""Integration: evaluate_hybrid.main joins caches and scores the gold set."""

import pandas as pd
import pytest

import evaluate_hybrid as m


def _patch(monkeypatch, fx, *, rule=None, zs=None, gold=None):
    monkeypatch.setattr(m, "GOLD_PATH", gold or fx.gold_all_path)
    monkeypatch.setattr(m, "RULE_PRED_PATH", rule or fx.rule_path)
    monkeypatch.setattr(m, "ZS_PRED_PATH", zs or fx.zs_path)
    monkeypatch.setattr(m, "OUTPUT_PATH", fx.out_path)


def test_main_writes_rule_first_hybrid(monkeypatch, corpus_fixtures):
    fx = corpus_fixtures
    _patch(monkeypatch, fx)
    m.main()

    out = pd.read_csv(fx.out_path).set_index("paper_id")
    for col in ("gold_label", "rule_predicted", "zs_predicted",
                "hybrid_predicted", "hybrid_method", "hybrid_correct"):
        assert col in out.columns

    # p1: rule fired (Tool Paper) -> hybrid uses the rule label.
    assert out.loc["p1", "hybrid_predicted"] == "Tool Paper"
    assert out.loc["p1", "hybrid_method"] == "rule"
    # p5: rule is Unknown -> hybrid falls back to zero-shot.
    assert out.loc["p5", "hybrid_predicted"] == "Empirical Study"
    assert out.loc["p5", "hybrid_method"] == "zero-shot"


def test_main_raises_when_cache_missing(monkeypatch, corpus_fixtures, tmp_path):
    fx = corpus_fixtures
    _patch(monkeypatch, fx, rule=tmp_path / "nope.csv")
    with pytest.raises(SystemExit):
        m.main()


def test_main_raises_when_gold_paper_absent_from_cache(monkeypatch, corpus_fixtures, tmp_path):
    fx = corpus_fixtures
    # Gold references a paper id that the caches don't contain -> unscored guard.
    extra_gold = pd.read_csv(fx.gold_all_path)
    extra_gold.loc[len(extra_gold)] = ["pX", "t", "a", "Tool Paper"]
    gp = tmp_path / "gold_extra.csv"
    extra_gold.to_csv(gp, index=False)
    _patch(monkeypatch, fx, gold=gp)
    with pytest.raises(SystemExit):
        m.main()

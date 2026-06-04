"""Integration: label_corpus_hybrid.main applies the hybrid to the whole corpus."""

import pandas as pd
import pytest

import label_corpus_hybrid as m


def test_main_labels_every_paper(monkeypatch, corpus_fixtures):
    fx = corpus_fixtures
    monkeypatch.setattr(m, "RULE_PRED_PATH", fx.rule_path)
    monkeypatch.setattr(m, "ZS_PRED_PATH", fx.zs_path)
    monkeypatch.setattr(m, "OUTPUT_PATH", fx.out_path)
    m.main()

    out = pd.read_csv(fx.out_path)
    assert len(out) == len(fx.ids)
    for col in ("paper_id", "hybrid_label", "hybrid_method", "rule_predicted", "zs_predicted"):
        assert col in out.columns
    # Hybrid falls back to zero-shot, which always emits a label -> never Unknown.
    assert (out["hybrid_label"] != "Unknown").all()
    # p5 (rule Unknown) routed to zero-shot; the rule-fired rows routed to rule.
    methods = out.set_index("paper_id")["hybrid_method"]
    assert methods["p5"] == "zero-shot"
    assert methods["p1"] == "rule"


def test_main_raises_when_cache_missing(monkeypatch, corpus_fixtures, tmp_path):
    fx = corpus_fixtures
    monkeypatch.setattr(m, "RULE_PRED_PATH", tmp_path / "nope.csv")
    monkeypatch.setattr(m, "ZS_PRED_PATH", fx.zs_path)
    monkeypatch.setattr(m, "OUTPUT_PATH", fx.out_path)
    with pytest.raises(SystemExit):
        m.main()

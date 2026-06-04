"""Integration: evaluate_zero_shot.main joins the zs cache onto gold.

Guards the schema-drift bug class (the old version merged on a non-existent
'ID' column and re-ran the model).
"""

import pandas as pd
import pytest

import evaluate_zero_shot as m


def test_main_scores_against_gold(monkeypatch, corpus_fixtures):
    fx = corpus_fixtures
    monkeypatch.setattr(m, "GOLD_PATH", fx.gold_all_path)
    monkeypatch.setattr(m, "ZS_PRED_PATH", fx.zs_path)
    monkeypatch.setattr(m, "OUTPUT_PATH", fx.out_path)
    m.main()

    out = pd.read_csv(fx.out_path).set_index("paper_id")
    assert {"gold_label", "predicted_type_zs", "correct"} <= set(out.columns)
    # p2 gold == zs prediction (Empirical Study) -> correct; p1 gold(Tool) != zs(Empirical).
    assert bool(out.loc["p2", "correct"]) is True
    assert bool(out.loc["p1", "correct"]) is False


def test_main_raises_when_cache_missing(monkeypatch, corpus_fixtures, tmp_path):
    fx = corpus_fixtures
    monkeypatch.setattr(m, "GOLD_PATH", fx.gold_all_path)
    monkeypatch.setattr(m, "ZS_PRED_PATH", tmp_path / "nope.csv")
    monkeypatch.setattr(m, "OUTPUT_PATH", fx.out_path)
    with pytest.raises(SystemExit):
        m.main()

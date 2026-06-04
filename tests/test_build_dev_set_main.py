"""Integration: build_dev_set.main excludes gold papers from the candidate pool.

Directly guards the regression where gold exclusion used gold_df["ID"] (the gold
column is paper_id), which raised KeyError and would have leaked gold into the dev set.
"""

import pandas as pd

import build_dev_set as m


def test_main_excludes_gold_and_blanks_classification(monkeypatch, corpus_fixtures):
    fx = corpus_fixtures
    monkeypatch.setattr(m, "RULE_PATH", fx.rule_path)
    monkeypatch.setattr(m, "ZS_PATH", fx.zs_path)
    monkeypatch.setattr(m, "GOLD_PATH", fx.gold_subset_path)   # gold = {p1, p2}
    monkeypatch.setattr(m, "OUTPUT_PATH", fx.out_path)

    m.main()   # must not raise

    out = pd.read_csv(fx.out_path)
    # No gold paper leaked into the dev set.
    assert set(out["paper_id"]) & set(fx.gold_subset_ids) == set()
    # Annotator column is present and empty (NaN after round-trip).
    assert "classification" in out.columns
    assert out["classification"].isna().all()
    # Only non-gold candidates remain.
    assert set(out["paper_id"]) <= (set(fx.ids) - set(fx.gold_subset_ids))

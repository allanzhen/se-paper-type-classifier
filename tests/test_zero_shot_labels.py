"""Structural invariants on the zero-shot label set (src/classify/zero_shot.py).

Guarded import: zero_shot imports torch/transformers at module load. Never
build the pipeline or run the model here.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import zero_shot as zs  # noqa: E402
from labels import CANONICAL_SET as CANONICAL  # noqa: E402


def test_labels_are_the_nine_canonical_with_unique_phrasings():
    assert set(zs.LABELS) == CANONICAL
    assert len(set(zs.LABELS.values())) == len(zs.LABELS)   # phrasings are distinct


def test_expanded_to_short_round_trips():
    assert zs.CANDIDATE_LABELS == list(zs.LABELS.values())
    for short, phrase in zs.LABELS.items():
        assert zs.EXPANDED_TO_SHORT[phrase] == short


def test_no_negation_tokens_in_phrasings():
    # Negation-heavy hypotheses measured ~2% worse on the NLI model; keep phrasings
    # positive. Guards against reintroducing that regression.
    bad = (" not ", "does not", "rather than", "without ", "secondary to")
    offenders = {k: v for k, v in zs.LABELS.items() if any(b in v.lower() for b in bad)}
    assert not offenders, f"negation in label phrasings: {offenders}"


def test_pick_device_cpu_when_mps_unavailable(monkeypatch):
    monkeypatch.setattr(zs.torch.backends.mps, "is_available", lambda: False)
    assert zs._pick_device() == -1

"""Fail-hard policy: a component that was asked for and is not ready is an
ENVIRONMENT fault, never a verdict about the document.

The failure these pin was observed in the field: `--semantic` passed without the
`semantic-grounder` extra produced a well-formed report announcing `0/74 (0.0%)`
and exited 0. Nothing distinguished a broken pipeline from a genuine finding that
no claim in the document was supported, and the lexical layers - which had scored
17 fuzzy + 5 bm25 on the same inputs moments earlier - reported 0 and 0.

Three separate guarantees, one per failure mode:
  * readiness is checked ONCE up front and raises (not once per claim, not a warning)
  * a mid-batch cascade fault never discards the lexical verdict already computed
  * a batch in which EVERY claim errored raises instead of returning a plausible 0%
"""

import pytest

from groundrails import semantic_ov, settings
from groundrails.config import load_document_processing_config
from groundrails.grounding import ground_batch
from groundrails.settings import ComponentNotReadyError

SOURCES = [
    (
        "s.md",
        "The study reports a 3D RMSE of 7.9 cm at a standoff of 70 m using PPK "
        "georeferencing of the UAV block, and notes uneven illumination degrades "
        "crack segmentation on concrete dam surfaces.",
    )
]
IN_SCOPE = "The study reports a 3D RMSE of 7.9 cm at a standoff of 70 m."
OUT_OF_SCOPE = "The purpose of this round was to establish what can honestly be sold."
# Paraphrases: deliberately NOT verbatim in SOURCES, so they cannot short-circuit as
# `exact` at step 2 and must actually enter the cascade. Using a verbatim claim here
# makes the test tautological - it would pass without the code under test.
ESCALATING = [
    "Positioning error of roughly eight centimetres was observed at seventy metres.",
    "Illumination variance harms segmentation quality on concrete structures.",
]


@pytest.fixture(autouse=True)
def _ready():
    settings.mark_ready()
    yield
    settings.reset()


@pytest.fixture
def _wide_band(monkeypatch):
    """Force in-scope claims into the escalation band so they reach the cascade."""
    from groundrails import joint

    real = joint.load_semantic_block

    def _block(path=None):
        b = dict(real(path) or {})
        b["escalation_band"] = [0.0, 1.0]
        return b

    monkeypatch.setattr(joint, "load_semantic_block", _block)


class TestPreflight:
    def test_missing_extra_raises_before_any_claim_is_scored(self, monkeypatch):
        monkeypatch.setattr(semantic_ov, "is_available", lambda: False)

        def _never(*a, **k):
            raise AssertionError("no claim may be scored once the preflight has failed")

        monkeypatch.setattr(semantic_ov.SemanticCascade, "score", _never)
        with pytest.raises(ComponentNotReadyError) as exc:
            ground_batch([IN_SCOPE, "A second claim."], SOURCES, semantic=True)
        assert "semantic cascade" in str(exc.value)
        assert "semantic-grounder" in str(exc.value)  # carries the install hint

    def test_lexical_tier_still_runs_when_semantic_is_declined(self, monkeypatch):
        """Opting out explicitly is not a failure - the cheap tier must still work."""
        monkeypatch.setattr(semantic_ov, "is_available", lambda: False)
        out = ground_batch([IN_SCOPE], SOURCES, semantic=False)
        assert len(out) == 1
        assert out[0].fuzzy_score > 0  # real lexical evidence, not a blank match


class TestNoSilentZero:
    @staticmethod
    def _exploding_cascade(monkeypatch):
        """Cascade that always raises, counting entries so a test cannot pass by
        never reaching it."""
        monkeypatch.setattr(semantic_ov, "is_available", lambda: True)
        calls = []

        def _boom(self, claim, chunks):
            calls.append(claim)
            raise RuntimeError("cascade exploded")

        monkeypatch.setattr(semantic_ov.SemanticCascade, "score", _boom)
        return calls

    def test_cascade_fault_keeps_the_lexical_verdict(self, monkeypatch, _wide_band):
        """DEF-14: the discarded match was already fully scored - never blank it."""
        calls = self._exploding_cascade(monkeypatch)
        claim = ESCALATING[0]

        # OUT_OF_SCOPE never reaches the cascade, so not EVERY claim errors and the
        # all-errored guard does not fire - isolating the fallback behaviour here.
        out = ground_batch([OUT_OF_SCOPE, claim], SOURCES, semantic=True)
        assert calls == [claim], "the claim never reached the cascade - test proves nothing"
        assert len(out) == 2
        lexical_only = ground_batch([claim], SOURCES, semantic=False)[0]
        assert out[1].fuzzy_score == pytest.approx(lexical_only.fuzzy_score)
        assert out[1].fuzzy_score > 0, "lexical evidence was discarded by the cascade fault"

    def test_every_claim_erroring_raises_instead_of_reporting_zero(self, monkeypatch, _wide_band):
        """DEF-16: an all-errored batch is a broken environment, not a 0% finding."""
        calls = self._exploding_cascade(monkeypatch)
        with pytest.raises(ComponentNotReadyError) as exc:
            ground_batch(ESCALATING, SOURCES, semantic=True)
        assert len(calls) == len(ESCALATING), "not every claim reached the cascade"
        assert "all 2 claims errored" in str(exc.value)


def test_config_loads_for_the_fixtures():
    """Guard: these tests are meaningless if the shipped config fails to load."""
    assert load_document_processing_config() is not None

"""Regression tests for grounding robustness: WordNet crash-proofing, per-claim batch
isolation, the language-block bypass, and the exported version. These guard the fixes for
the NLTK zip-reader re-entrancy crash that aborted whole-document grounding under threads."""

import pytest

from groundrails import UnsupportedLanguageError, ground


def test_version_exported():
    """`groundrails.__version__` exists so import-time version checks work."""
    import groundrails

    assert isinstance(groundrails.__version__, str) and groundrails.__version__


def test_wordnet_antonyms_degrades_on_reader_error(monkeypatch):
    """A WordNet reader error (the NLTK zip re-entrancy `assert self.fp is None`) degrades to
    "no antonyms" instead of propagating - the antonym-flip must never crash grounding."""
    from groundrails import lexical as lx

    class _Boom:
        def synsets(self, w):
            raise AssertionError("self.fp is None")  # the NLTK OpenOnDemandZipFile crash

    monkeypatch.setitem(lx._WN, "mod", _Boom())
    monkeypatch.setitem(lx._WN, "cache", {})
    assert lx._wn_antonyms("increase") == set()  # graceful: empty, no exception


def test_ground_batch_isolates_per_claim_crash(monkeypatch):
    """A per-claim bug must not abort the batch - the failed claim returns ungrounded, the rest run."""
    from groundrails import grounding as g

    real = g.ground

    def flaky(claim, *a, **k):
        if claim == "boom":
            raise RuntimeError("simulated WordNet crash")
        return real(claim, *a, **k)

    monkeypatch.setattr(g, "ground", flaky)
    res = g.ground_batch(["boom", "The Eiffel Tower is in Paris."], ["The Eiffel Tower is in Paris."], semantic=False)
    assert len(res) == 2
    assert res[0].match_type == "none"  # the crashing claim is isolated, not fatal
    assert res[1].grounded  # the healthy claim still grounds


def test_ignore_language_bypasses_high_tier_block(monkeypatch):
    """`ignore_language=True` skips the HIGH-tier non-English hard-block; without it the block fires."""
    import groundrails.config as cfg_mod
    from groundrails import lexical as lx
    from groundrails import lexical_mt as mt

    # cross-lingual: claim "xx" (no model), english evidence -> would block without the flag
    monkeypatch.setattr(lx, "detect_lang_confident", lambda text, *a, **k: "xx" if "obcym" in text else "en")
    monkeypatch.setattr(mt, "has_model", lambda iso: False)
    cfg = cfg_mod.load_document_processing_config().overlay(lexical_effort="high")

    with pytest.raises(UnsupportedLanguageError):
        ground("zdanie w obcym jezyku tutaj", ["the english evidence source text"], config=cfg, semantic=False)

    m = ground(
        "zdanie w obcym jezyku tutaj",
        ["the english evidence source text"],
        config=cfg,
        semantic=False,
        ignore_language=True,
    )
    assert m.match_type in ("none", "exact", "fuzzy", "bm25", "semantic")  # scored, not blocked

"""Hard-block guard: HIGH-tier claims in unsupported languages raise, not score."""

import dataclasses

import pytest

from groundrails import UnsupportedLanguageError, ground
from groundrails import lexical as lx
from groundrails import lexical_mt as mt
from groundrails.config import load_document_processing_config


def test_has_model_english_and_undetermined_pass():
    assert mt.has_model("en")
    assert mt.has_model("und")
    assert mt.has_model("")


def test_has_model_missing_model(monkeypatch):
    monkeypatch.setattr(mt, "_find_pkg", lambda code: None)
    assert mt.has_model("xx") is False


def test_has_model_present_model(monkeypatch):
    monkeypatch.setattr(mt, "_find_pkg", lambda code: object())
    assert mt.has_model("it") is True


def test_detect_lang_confident_ignores_low_confidence():
    # English keyword fragment lingua misreads as Latin at low confidence -> und
    assert lx.detect_lang_confident("dolphins mammals intelligent aquatic creatures") == "und"


def test_detect_lang_confident_genuine_nonenglish():
    # a clear Latin sentence is confidently detected
    assert (
        lx.detect_lang_confident(
            "Gallia est omnis divisa in partes tres quarum unam incolunt Belgae"
        )
        == "la"
    )


def test_ground_blocks_unsupported_language(monkeypatch):
    # cross-lingual: claim "la", evidence "en", no MT model -> the bridge is needed but absent -> block
    monkeypatch.setattr(lx, "detect_lang_confident", lambda text, *a, **k: "la" if "Lorem" in text else "en")
    monkeypatch.setattr(mt, "has_model", lambda iso: False)
    with pytest.raises(UnsupportedLanguageError) as exc:
        ground("Lorem ipsum dolor sit amet consectetur adipiscing elit.", ["english source text"])
    assert exc.value.lang == "la"


def test_same_language_evidence_not_blocked(monkeypatch):
    """Claim and evidence in the SAME non-English language need no MT bridge - the lexical
    layers match the raw text directly, so the HIGH-tier guard must not block."""
    monkeypatch.setattr(lx, "detect_lang_confident", lambda *a, **k: "la")  # claim AND evidence "la"
    monkeypatch.setattr(mt, "has_model", lambda iso: False)
    m = ground("Gallia est omnis divisa in partes.", ["Gallia est omnis divisa in partes tres."])
    assert m.match_type is not None  # scored directly, not blocked


def test_ground_passes_supported_language(monkeypatch):
    monkeypatch.setattr(lx, "detect_lang_confident", lambda *a, **k: "de")
    monkeypatch.setattr(mt, "has_model", lambda iso: True)
    m = ground("Der Eiffelturm steht in Paris.", ["The Eiffel Tower is in Paris, France."])
    assert m.match_type is not None


def test_low_tier_does_not_block(monkeypatch):
    # a confidently-unsupported language is NOT blocked at LOW tier (no MT bridge there)
    monkeypatch.setattr(lx, "detect_lang_confident", lambda *a, **k: "la")
    monkeypatch.setattr(mt, "has_model", lambda iso: False)
    cfg_low = dataclasses.replace(load_document_processing_config(), lexical_effort="low")
    m = ground("Lorem ipsum dolor sit amet consectetur.", ["some source text"], config=cfg_low)
    assert m.match_type is not None


def test_ground_english_not_blocked():
    m = ground(
        "The Eiffel Tower is in Paris.",
        ["The Eiffel Tower is located in Paris, France."],
    )
    assert m.match_type is not None

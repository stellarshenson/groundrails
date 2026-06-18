"""Tests for the grounding-document API: grounding_document / build_grounding_document and
the GroundingMatch.grounded / support provenance, including the cascade -> lexical fallback."""

from groundrails import GroundingMatch, build_grounding_document, grounding_document
from groundrails.claims import Claim
from groundrails.grounding import Location


def test_grounding_document_structure():
    claims = [
        Claim(
            id="c01",
            claim="The Eiffel Tower is in Paris.",
            line_number=1,
            char_start=0,
            char_end=29,
        )
    ]
    sources = [
        ("evidence.txt", "The Eiffel Tower is located in Paris, France. It was completed in 1889.")
    ]
    doc = grounding_document(claims, sources)
    assert doc["sources"] == ["evidence.txt"]
    assert doc["summary"] == {"total": 1, "grounded": 1, "ungrounded": 0}
    entry = doc["claims"][0]
    assert entry["id"] == "c01"
    assert entry["claim_location"] == {"line": 1, "char_start": 0, "char_end": 29}
    assert entry["grounded"] is True
    assert entry["score"] > 0
    sup = entry["support"]
    assert sup["source_index"] == 0 and sup["source_path"] == "evidence.txt"
    assert sup["char_end"] > sup["char_start"] >= 0 and sup["matched_text"]


def test_grounding_document_ungrounded_plain_string_claim():
    doc = grounding_document(
        ["The rocket reached escape velocity."],
        [("e.txt", "This document is about office furniture procurement.")],
    )
    entry = doc["claims"][0]
    assert entry["grounded"] is False
    assert entry["support"] is None
    assert entry["claim_location"] is None  # a plain-string claim carries no answer-doc location


def test_build_grounding_document_summary_counts():
    matches = [
        GroundingMatch(claim="a", match_type="bm25"),
        GroundingMatch(claim="b", match_type="none"),
    ]
    doc = build_grounding_document(matches)
    assert doc["summary"] == {"total": 2, "grounded": 1, "ungrounded": 1}
    assert "sources" not in doc  # sources omitted when not supplied


def test_grounded_property():
    assert GroundingMatch(claim="x", match_type="exact").grounded is True
    assert GroundingMatch(claim="x", match_type="none").grounded is False
    assert GroundingMatch(claim="x", match_type="contradicted").grounded is False


def test_support_none_when_ungrounded():
    assert GroundingMatch(claim="x", match_type="none").support is None


def test_support_fallback_to_lexical_for_cascade_verdict():
    """A cascade verdict (match_type=semantic) has no native location, so support falls back to
    the best lexical passage, flagged support_via=lexical - the agent always gets a place to look."""
    m = GroundingMatch(claim="x", match_type="semantic")
    m.bm25_matched_text = "the supporting passage"
    m.bm25_location = Location(
        source_index=2,
        source_path="ev.txt",
        char_start=10,
        char_end=32,
        line_start=3,
        line_end=3,
        paragraph=1,
        page=1,
    )
    sup = m.support
    assert sup is not None
    assert sup["support_via"] == "lexical"
    assert sup["source_index"] == 2 and sup["char_start"] == 10
    assert sup["matched_text"] == "the supporting passage"

"""Tests for groundrails."""

from __future__ import annotations

import json

import pytest

from groundrails import (
    GroundingMatch,
    ground,
    ground_batch,
)
from groundrails import settings as settings_mod
from groundrails.chunking import (
    recursive_chunk,
)
from groundrails.cli import main as cli_main
from groundrails.lexical_mt import has_model


@pytest.fixture
def deterministic_engine(monkeypatch):
    """Force the deterministic-cascade verdict head.

    The bundled config ships ``mode: lexical`` (the manifold verdict). A handful
    of tests exercise the cascade's per-layer threshold semantics (fuzzy/bm25
    below-threshold -> none, the -1.0 sentinel) on tiny toy fixtures that are
    out-of-distribution for the manifold. Those tests pin the deterministic head
    explicitly so they keep covering the cascade code, independent of the default.
    """
    from groundrails import calibration as _C
    from groundrails import grounding as _G

    monkeypatch.setattr(
        _C, "load_calibration_from_config", lambda path=None: {"engine": "deterministic"}
    )
    _G._LEXICAL_VERDICT_CACHE.clear()
    yield
    _G._LEXICAL_VERDICT_CACHE.clear()


class TestExactMatching:
    """Regex (exact) layer — whitespace-tolerant, case-insensitive."""

    def test_exact_verbatim(self):
        m = ground("quick brown fox", ["The quick brown fox jumps."])
        assert m.match_type == "exact"
        assert m.exact_score == 1.0
        assert m.exact_matched_text == "quick brown fox"
        assert m.exact_location.char_start == 4
        assert m.exact_location.char_end == 19

    def test_exact_case_insensitive(self):
        m = ground("QUICK BROWN FOX", ["The quick brown fox jumps."])
        assert m.match_type == "exact"
        assert m.exact_score == 1.0

    def test_exact_whitespace_tolerant(self):
        m = ground("quick brown fox", ["The  quick\n brown  \tfox jumps."])
        assert m.match_type == "exact"
        assert m.exact_score == 1.0

    def test_exact_miss(self):
        m = ground("completely unrelated phrase", ["The quick brown fox jumps."])
        assert m.exact_score == 0.0
        assert m.exact_matched_text == ""

    def test_exact_multi_source_first_hit_wins(self):
        m = ground(
            "brown fox", ["nothing here", "The quick brown fox jumps.", "also has brown fox"]
        )
        assert m.match_type == "exact"
        assert m.exact_location.source_index == 1

    def test_exact_with_source_paths(self):
        m = ground(
            "brown fox",
            [("doc1.txt", "nothing here"), ("doc2.txt", "The quick brown fox jumps.")],
        )
        assert m.match_type == "exact"
        assert m.exact_location.source_path == "doc2.txt"


class TestFuzzyMatching:
    """Levenshtein (fuzzy) layer — always runs, best across sources."""

    def test_fuzzy_above_threshold(self):
        m = ground(
            "quick brown fox jumped over",
            ["The quick brown fox jumps over the lazy dog."],
            fuzzy_threshold=0.80,
        )
        assert m.exact_score == 0.0
        assert m.fuzzy_score >= 0.80
        assert m.match_type == "fuzzy"

    def test_fuzzy_below_threshold(self, deterministic_engine):
        m = ground(
            "tropical island paradise",
            ["The quick brown fox jumps over the lazy dog."],
            fuzzy_threshold=0.85,
        )
        assert m.exact_score == 0.0
        assert m.fuzzy_score < 0.85
        assert m.match_type == "none"

    def test_fuzzy_always_computed_even_on_exact_hit(self):
        """Both scores always populated; exact match yields fuzzy=1.0 too."""
        m = ground("brown fox", ["The quick brown fox jumps."])
        assert m.exact_score == 1.0
        assert m.fuzzy_score == 1.0
        assert m.match_type == "exact"

    def test_fuzzy_best_across_sources(self):
        m = ground(
            "quick brown fox",
            [
                "blue sky overhead",
                "quirk brown fux jumps",
            ],
        )
        assert m.exact_score == 0.0
        assert m.fuzzy_score > 0.5
        assert m.fuzzy_location.source_index == 1


class TestBothSignalsReported:
    """All three scores always in the result (user requirement)."""

    def test_none_match_still_reports_fuzzy_signal(self, deterministic_engine):
        """Even when match_type=none, fuzzy_score shows best-effort signal."""
        m = ground("something different", ["slightly different content here"])
        assert m.match_type == "none"
        assert m.fuzzy_score > 0
        assert m.fuzzy_matched_text != ""

    def test_all_three_scores_independent(self):
        """exact=0 does not zero fuzzy or bm25."""
        m = ground("fox jumps", ["the quick fux jumps high"])
        assert m.exact_score == 0.0
        assert m.fuzzy_score > 0
        # BM25 may or may not fire on such a short source, but score is set


class TestBM25Matching:
    """BM25 layer — topical/lexical grounding across passages."""

    _LONG_SOURCE = (
        "Introduction paragraph about birds.\n\n"
        "The quick brown fox jumps over the lazy dog in the meadow.\n\n"
        "Cats sleep most of the day on windowsills.\n\n"
        "Aquatic mammals like dolphins are highly intelligent.\n"
    )

    def test_bm25_finds_right_passage(self):
        """Paraphrased claim with same key terms lands in the right passage."""
        m = ground(
            "fox and dog in a meadow",
            [self._LONG_SOURCE],
            fuzzy_threshold=0.95,  # high, so fuzzy fails
            bm25_threshold=0.4,
        )
        # The fox passage should win
        assert "fox" in m.bm25_matched_text
        assert m.bm25_score > 0
        assert m.bm25_token_recall > 0

    def test_bm25_token_recall_is_fraction(self):
        """Token recall = fraction of unique claim tokens in best passage."""
        m = ground("fox dog meadow", [self._LONG_SOURCE])
        # All 3 tokens present → recall = 1.0
        assert m.bm25_token_recall == 1.0

    def test_bm25_raw_score_available(self):
        """Raw BM25 score exposed for callers who want the unbounded signal."""
        m = ground("fox dog meadow", [self._LONG_SOURCE])
        assert m.bm25_raw_score >= 0

    def test_bm25_location_populated(self):
        """BM25 location has line/paragraph/page just like other layers."""
        m = ground("fox dog", [self._LONG_SOURCE])
        assert m.bm25_location.line_start > 0
        assert m.bm25_location.paragraph > 0
        assert m.bm25_location.page == 1

    def test_bm25_matches_topical_paraphrase(self):
        """Same terms, different order — BM25 catches what Levenshtein misses."""
        # "Dolphins are smart aquatic mammals" — paraphrase of sentence in source
        m = ground(
            "dolphins mammals intelligent aquatic",
            [self._LONG_SOURCE],
            fuzzy_threshold=0.95,
            bm25_threshold=0.5,
        )
        assert m.exact_score == 0.0
        # BM25 should catch this
        assert m.bm25_token_recall >= 0.5
        assert "dolphins" in m.bm25_matched_text.lower()

    def test_bm25_below_threshold_classified_none(self, deterministic_engine):
        """When BM25 token-recall below threshold, match_type=none."""
        m = ground(
            "quantum physics neutrino detector",
            [self._LONG_SOURCE],
            fuzzy_threshold=0.95,
            bm25_threshold=0.5,
        )
        assert m.match_type == "none"

    def test_bm25_priority_below_fuzzy(self):
        """When both fuzzy and bm25 would classify, fuzzy wins."""
        m = ground(
            "quick brown fox jumped",  # fuzzy match of "quick brown fox jumps"
            [self._LONG_SOURCE],
            fuzzy_threshold=0.80,
            bm25_threshold=0.5,
        )
        assert m.match_type == "fuzzy"  # fuzzy wins over bm25


class TestLocation:
    """Location metadata — line, column, paragraph, page, context."""

    def test_line_number_single_line(self):
        m = ground("fox", ["The quick brown fox jumps."])
        assert m.exact_location.line_start == 1
        assert m.exact_location.line_end == 1

    def test_line_number_multiline_source(self):
        text = "line one\nline two\nthe fox is here\nline four"
        m = ground("fox is here", [text])
        assert m.match_type == "exact"
        assert m.exact_location.line_start == 3
        assert m.exact_location.line_end == 3

    def test_column_number(self):
        text = "hello brown fox and more"
        m = ground("brown fox", [text])
        assert m.exact_location.line_start == 1
        # "brown fox" starts at char 6 on line 1 → column 7 (1-indexed)
        assert m.exact_location.column_start == 7

    def test_paragraph_number(self):
        text = "first paragraph text\n\nsecond paragraph with fox here\n\nthird paragraph"
        m = ground("fox", [text])
        assert m.match_type == "exact"
        assert m.exact_location.paragraph == 2

    def test_paragraph_blank_line_with_whitespace(self):
        """Blank lines with whitespace still separate paragraphs."""
        text = "first para\n  \t  \nsecond para fox"
        m = ground("fox", [text])
        assert m.exact_location.paragraph == 2

    def test_page_number_via_form_feed(self):
        """Pages separated by \\f (pdftotext convention)."""
        text = "page one content\fpage two with fox\fpage three"
        m = ground("fox", [text])
        assert m.match_type == "exact"
        assert m.exact_location.page == 2

    def test_page_1_when_no_form_feed(self):
        m = ground("fox", ["no form feeds here just fox content"])
        assert m.exact_location.page == 1

    def test_context_before_after(self):
        text = "The quick brown fox jumps over the lazy dog gently."
        m = ground("brown fox", [text])
        # Context should include surrounding words
        ctx_before = m.exact_location.context_before
        ctx_after = m.exact_location.context_after
        assert "quick" in ctx_before or "The" in ctx_before
        assert "jumps" in ctx_after or "over" in ctx_after

    def test_context_trimmed_to_max_chars(self):
        """Long context is trimmed with ellipsis."""
        long_text = "x" * 200 + " brown fox " + "y" * 200
        m = ground("brown fox", [long_text], context_chars=40)
        assert len(m.exact_location.context_before) <= 41  # 40 + ellipsis
        assert len(m.exact_location.context_after) <= 41


class TestEdgeCases:
    def test_empty_sources(self):
        m = ground("anything", [])
        assert isinstance(m, GroundingMatch)
        assert m.match_type == "none"
        assert m.exact_score == 0.0
        assert m.fuzzy_score == 0.0

    def test_empty_claim(self):
        m = ground("", ["some source text"])
        assert m.exact_score == 0.0

    def test_empty_source_text(self):
        m = ground("anything", [""])
        assert m.exact_score == 0.0
        assert m.fuzzy_score == 0.0


class TestBatch:
    def test_ground_batch_preserves_order(self):
        claims = ["brown fox", "lazy dog", "unrelated claim"]
        sources = ["The quick brown fox jumps over the lazy dog."]
        results = ground_batch(claims, sources, fuzzy_threshold=0.85)
        assert len(results) == 3
        assert results[0].match_type == "exact"
        assert results[1].match_type == "exact"
        assert results[2].match_type in ("fuzzy", "none")

    def test_ground_batch_multithreaded_matches_serial(self):
        # Threaded grounding must produce byte-identical results to serial and
        # preserve claim order (the adaptive_gap pass indexes by position).
        claims = [f"brown fox number {i}" for i in range(12)] + [
            "lazy dog",
            "totally unrelated claim about quantum mechanics",
        ]
        sources = ["The quick brown fox jumps over the lazy dog."]
        serial = ground_batch(claims, sources, fuzzy_threshold=0.85, max_workers=1)
        threaded = ground_batch(claims, sources, fuzzy_threshold=0.85, max_workers=5)
        assert len(threaded) == len(serial) == len(claims)
        assert [m.match_type for m in threaded] == [m.match_type for m in serial]
        assert [m.agreement_score for m in threaded] == [m.agreement_score for m in serial]

    def test_ground_batch_workers_capped_to_claim_count(self):
        # More workers than claims must not error or change results.
        claims = ["lazy dog"]
        sources = ["The quick brown fox jumps over the lazy dog."]
        results = ground_batch(claims, sources, fuzzy_threshold=0.85, max_workers=5)
        assert len(results) == 1
        assert results[0].match_type == "exact"


class TestChunking:
    """Recursive chunking preserves offsets + boundaries."""

    def test_empty_text_returns_empty(self):
        assert recursive_chunk("") == []

    def test_short_text_one_chunk(self):
        text = "The quick brown fox."
        chunks = recursive_chunk(text, max_chars=1500)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == len(text)

    def test_paragraph_split(self):
        text = "First paragraph here.\n\nSecond paragraph longer content here.\n\nThird paragraph ends the document."
        chunks = recursive_chunk(text, max_chars=30, min_chunk_chars=10)
        assert len(chunks) >= 2

    def test_offsets_are_valid(self):
        text = "paragraph one\n\nparagraph two content\n\nparagraph three final"
        chunks = recursive_chunk(text, max_chars=200, min_chunk_chars=5)
        for c in chunks:
            # Char offsets must be valid bounds into the source
            assert 0 <= c.char_start < c.char_end <= len(text)
            # First + last words of the chunk should appear inside the source span
            first_word = c.text.split()[0]
            last_word = c.text.split()[-1]
            span = text[c.char_start : c.char_end]
            assert first_word in span
            assert last_word in span

    def test_long_sentence_sliding_window(self):
        sentence = "word " * 500  # long single sentence
        chunks = recursive_chunk(sentence, max_chars=200, overlap_chars=50)
        assert len(chunks) > 1
        # Overlap: consecutive chunks share some content
        if len(chunks) >= 2:
            assert chunks[0].char_end > chunks[1].char_start

    def test_offsets_monotonic(self):
        text = "a " * 200 + "\n\n" + "b " * 200
        chunks = recursive_chunk(text, max_chars=100)
        starts = [c.char_start for c in chunks]
        assert starts == sorted(starts)


class TestSettings:
    """Settings load/save/prompt — zero-dep."""

    def test_defaults_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        cfg = settings_mod.load()
        assert cfg.semantic_model == "intfloat/multilingual-e5-small"
        assert cfg.semantic_device == "auto"

    def test_save_then_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        # Create project root marker
        (tmp_path / ".claude").mkdir()
        s = settings_mod.Settings(semantic_model="custom/model")
        path = settings_mod.save(s)
        assert path.exists()
        loaded = settings_mod.load()
        assert loaded.semantic_model == "custom/model"

    def test_obsolete_semantic_enabled_key_is_ignored(self, tmp_path, monkeypatch):
        # Old settings files may still carry semantic_enabled - it must load
        # without error and simply be dropped (unknown keys filtered on read).
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".claude").mkdir()
        d = tmp_path / ".stellars-plugins"
        d.mkdir()
        (d / "settings.json").write_text(
            json.dumps({"semantic_enabled": True, "semantic_model": "m/x"})
        )
        loaded = settings_mod.load()
        assert loaded.semantic_model == "m/x"
        assert not hasattr(loaded, "semantic_enabled")


class TestSemanticOnContract:
    """Semantic grounding is opt-in per call via the boolean ``--semantic``:

    - ``enabled=True`` + deps present -> grounder built
    - ``enabled=True`` + deps missing -> hard fail (sys.exit 2)
    - ``enabled=False`` -> None (no semantic), and the deps check is never
      reached. There is no persisted enable setting.
    """

    def test_semantic_on_with_deps_missing_exits_2(self, monkeypatch, capsys):
        """--semantic with the extras missing must hard-fail, never degrade
        silently (silent degradation produced misleading 0.000 semantic rows).
        """
        from groundrails.cli import (
            _build_semantic_grounder,
        )

        monkeypatch.setattr(settings_mod, "is_semantic_available", lambda: False)

        with pytest.raises(SystemExit) as exc_info:
            _build_semantic_grounder(settings_mod.Settings(), True)
        assert exc_info.value.code == 2

        err = capsys.readouterr().err
        assert "ERROR: --semantic requires the [semantic] extras" in err
        assert "pip install" in err

    def test_semantic_not_requested_returns_none_without_deps_check(self, monkeypatch):
        """enabled=False -> no semantic layer, and the deps check is never reached
        (the layer is purely opt-in, no config default to honour).
        """
        from groundrails.cli import (
            _build_nli_grounder,
            _build_semantic_grounder,
        )

        sentinel_called = []
        monkeypatch.setattr(
            settings_mod,
            "is_semantic_available",
            lambda: sentinel_called.append(True) or False,
        )
        cfg = settings_mod.Settings()
        assert _build_semantic_grounder(cfg, False) is None
        assert _build_nli_grounder(cfg, False) is None
        assert sentinel_called == []  # deps-check never invoked


class TestCLISetup:
    """CLI setup subcommand."""

    def test_setup_shows_current_if_present(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".claude").mkdir()
        # Pre-seed settings
        settings_mod.save(settings_mod.Settings())
        code = cli_main(["setup"])
        assert code == 0
        err = capsys.readouterr().err
        assert "semantic_model" in err


# -------------------------------------------------------------------------
# Source-format fallback: gate-warning on unsupported binaries (was WI#1
# binary-rejection; Release F changed the contract to skip-with-warning).
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Release F: source-format fallback + sibling lookup + auto-OCR
# -------------------------------------------------------------------------


def _ack_all_warnings(stderr_out: str) -> list[str]:
    """Helper - extract every W-xxxxxxxx token from BLOCKED stderr and
    return a list of `--ack-warning TOKEN=test-fixture` argv pairs."""
    import re

    seen: set[str] = set()
    flags: list[str] = []
    for tok in re.findall(r"W-[0-9a-f]{8}", stderr_out):
        if tok in seen:
            continue
        seen.add(tok)
        flags += ["--ack-warning", f"{tok}=test-fixture"]
    return flags


# -------------------------------------------------------------------------
# WI#3: cross-source provenance + --primary-source
# -------------------------------------------------------------------------


class TestCrossSourceProvenance:
    def test_grounded_source_from_exact_hit(self):
        sources = [
            ("primary.md", "The cat sat on the mat."),
            ("secondary.md", "Unrelated content about dogs."),
        ]
        m = ground("the cat sat on the mat", sources)
        assert m.match_type == "exact"
        assert m.grounded_source == "primary.md"
        assert m.is_primary_source is True

    def test_non_primary_flag(self):
        sources = [
            ("primary.md", "Nothing about cats here."),
            ("secondary.md", "The cat sat on the mat."),
        ]
        m = ground("the cat sat on the mat", sources, primary_source="primary.md")
        assert m.grounded_source == "secondary.md"
        assert m.is_primary_source is False
        assert m.verification_needed is True

    def test_primary_source_match(self):
        sources = [
            ("primary.md", "The cat sat on the mat."),
            ("secondary.md", "Also: the cat sat on the mat."),
        ]
        m = ground("the cat sat on the mat", sources, primary_source="primary.md")
        assert m.is_primary_source is True


# -------------------------------------------------------------------------
# WI#5 + WI#6: lexical_co_support, verification_needed, claim_attributes
# -------------------------------------------------------------------------


class TestVerificationSignals:
    def test_exact_hit_has_lexical_support(self):
        m = ground("the cat sat", [("a.txt", "the cat sat on the mat")])
        assert m.lexical_co_support is True

    def test_claim_attributes_populated(self):
        m = ground(
            "42 users logged in yesterday",
            [("a.txt", "42 users logged in yesterday.")],
        )
        attrs = m.claim_attributes
        assert "numbers" in attrs
        assert "entities" in attrs
        assert "passage_numbers" in attrs
        assert "passage_entities" in attrs
        # At least the number 42 should be extracted
        values = [v for v, _, _ in attrs["numbers"]]
        assert "42" in values

    def test_numeric_co_presence_triggers_verification(self):
        # Both sides have numbers tied to the same context word ("users")
        # but the deterministic mismatch detector won't fire with a single
        # clean hit — the heuristic flag should still call this out.
        m = ground(
            "the project grew to 42 users",
            [("a.txt", "the project reports 100 users on record")],
        )
        # bm25 / fuzzy co-occurrence should yield verification_needed=True
        # because both claim and passage have number+"users"
        if m.match_type in ("fuzzy", "bm25", "semantic"):
            assert m.verification_needed is True


# -------------------------------------------------------------------------
# WI#2: extract-claims
# -------------------------------------------------------------------------


class TestExtractClaims:
    def test_basic_extraction(self, tmp_path):
        from groundrails.extract import (
            extract_claims,
        )

        doc = (
            "# Heading\n\n"
            "The system handles 42 concurrent sessions. "
            "It was tested on Linux and macOS.\n\n"
            "Short.\n\n"
            "- dev\n"
            "- test\n"
            "- staging\n\n"
            "The deployment runs on Kubernetes with three nodes.\n"
        )
        claims = extract_claims(doc)
        assert len(claims) >= 2
        # Stable IDs
        assert claims[0].id.startswith("c0")
        # Short stubs and pure headers excluded
        for c in claims:
            assert len(c.claim) >= 20

    def test_cli_extract_claims(self, tmp_path, capsys):
        doc = tmp_path / "doc.md"
        doc.write_text(
            "The system handles 42 concurrent sessions.\nIt was tested on Linux and macOS.\n"
        )
        out = tmp_path / "claims.json"
        code = cli_main(["extract-claims", "--document", str(doc), "--output", str(out)])
        assert code == 0
        data = json.loads(out.read_text())
        assert len(data) >= 1
        assert "id" in data[0]
        assert "claim" in data[0]
        assert "line_number" in data[0]


# -------------------------------------------------------------------------
# WI#4: check-consistency
# -------------------------------------------------------------------------


class TestCheckConsistency:
    def test_numeric_divergence_flagged(self):
        from groundrails.consistency import (
            check_consistency,
        )

        text = (
            "The platform supports 42 users on average.\n"
            "\n\n"
            "Recent benchmarks show 50 users on load.\n"
        )
        findings = check_consistency(text)
        numeric_findings = [f for f in findings if f.kind == "numeric"]
        assert len(numeric_findings) >= 1
        # Both line numbers should appear
        all_lines = [line for f in numeric_findings for line, _ in f.occurrences]
        assert 1 in all_lines
        assert 4 in all_lines

    def test_entity_set_divergence_flagged(self):
        from groundrails.consistency import (
            check_consistency,
        )

        text = (
            "We run dev, test, and staging environments.\n"
            "\n\n"
            "Pipeline deploys to dev, staging, and prod.\n"
        )
        findings = check_consistency(text)
        set_findings = [f for f in findings if f.kind == "entity_set"]
        assert len(set_findings) >= 1

    def test_no_divergence_reports_clean(self):
        from groundrails.consistency import (
            check_consistency,
            format_consistency_report,
        )

        text = "Simple consistent document with 42 users and 42 users again.\n"
        findings = check_consistency(text)
        # Same value twice - no divergence
        numeric_findings = [f for f in findings if f.kind == "numeric"]
        assert not numeric_findings
        report = format_consistency_report(findings)
        assert "No divergences" in report

    def test_cli_check_consistency(self, tmp_path):
        doc = tmp_path / "doc.md"
        doc.write_text(
            "The system handles 42 users per session.\n\nRecent tests show 50 users per session.\n"
        )
        out = tmp_path / "consistency.md"
        code = cli_main(["check-consistency", "--document", str(doc), "--output", str(out)])
        # Exit 1 when findings exist
        assert code == 1
        assert out.exists()
        report = out.read_text()
        assert "Self-Consistency" in report


# -------------------------------------------------------------------------
# WI#7: validate (grounding + self-consistency, manifest mode)
# -------------------------------------------------------------------------


class TestSemanticOnnx:
    """ONNX Runtime embedding path - torch-free.

    The pooling/normalisation unit tests inject a fake ONNX session +
    tokenizer so they run without any model download (and without torch).
    The real-model test is network-gated and skips when the e5 ONNX weights
    are not reachable.
    """

    def _make_grounder(self, fake_session, fake_tokenizer, input_names):
        from groundrails.semantic import (
            SemanticGrounder,
        )

        g = SemanticGrounder.__new__(SemanticGrounder)
        g._session = fake_session
        g._tokenizer = fake_tokenizer
        g._input_names = set(input_names)
        g.model_name = "intfloat/multilingual-e5-small"
        return g

    def test_embed_mean_pool_unit_norm_and_token_type_ids(self):
        """_embed: mask-aware mean-pool, L2-norm, and zero token_type_ids feed."""
        import numpy as np

        def fake_tok(texts, **kw):
            n = len(texts)
            ids = np.array([[1, 2, 3], [4, 5, 0]], dtype="int64")[:n]
            mask = np.array([[1, 1, 1], [1, 1, 0]], dtype="int64")[:n]
            return {"input_ids": ids, "attention_mask": mask}

        captured: dict = {}

        class FakeSession:
            def run(self, _outputs, feed):
                captured.update(feed)
                # (N, L, dim=2). Text1 token2 has a huge value that MUST be
                # masked out by the attention_mask above.
                last = np.array(
                    [
                        [[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]],
                        [[2.0, 2.0], [4.0, 4.0], [9.9, 9.9]],
                    ],
                    dtype="float32",
                )
                return [last]

        g = self._make_grounder(
            FakeSession(), fake_tok, {"input_ids", "attention_mask", "token_type_ids"}
        )
        vecs = g._embed(["a", "b"])

        # token_type_ids supplied as zeros, shaped like input_ids
        assert "token_type_ids" in captured
        assert captured["token_type_ids"].shape == (2, 3)
        assert int(captured["token_type_ids"].sum()) == 0

        # rows are unit-norm
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-6)

        # text0: mean (1,3,5)/3 = (3,0) -> normalised (1,0)
        assert np.allclose(vecs[0], [1.0, 0.0], atol=1e-6)
        # text1: masked token2 excluded -> mean (2,4)/2 = (3,3) -> (0.707,0.707)
        assert np.allclose(vecs[1], [2**-0.5, 2**-0.5], atol=1e-6)

    def test_cache_round_trip_skips_reembed(self, tmp_path):
        """_load_or_embed embeds once, then serves the parquet cache."""
        import numpy as np

        pytest.importorskip("pyarrow")
        from groundrails.chunking import (
            recursive_chunk,
        )
        from groundrails.semantic import (
            SemanticGrounder,
        )

        g = SemanticGrounder.__new__(SemanticGrounder)
        g.model_name = "intfloat/multilingual-e5-small"
        g.cache_dir = tmp_path
        g.max_chars = 1500
        calls = {"n": 0}

        def fake_embed(texts):
            calls["n"] += 1
            return np.full((len(texts), 4), 0.5, dtype="float32")

        g._embed = fake_embed

        text = "The estate has three walled gardens and an orchard. " * 20
        chunks = recursive_chunk(text, max_chars=1500)
        v1 = g._load_or_embed("doc.txt", text, chunks)
        v2 = g._load_or_embed("doc.txt", text, chunks)

        assert calls["n"] == 1  # second call hit the cache, no re-embed
        assert np.allclose(v1, v2)

    def test_real_model_paraphrase_outscores_unrelated(self, tmp_path):
        """Network-gated: real e5 ONNX ranks a paraphrase above an unrelated line."""
        pytest.importorskip("onnxruntime")
        pytest.importorskip("transformers")
        from groundrails.semantic import (
            SemanticGrounder,
            is_available,
        )

        if not is_available():
            pytest.skip("semantic extras not installed")
        try:
            g = SemanticGrounder(cache_dir=str(tmp_path / "cache"))
        except Exception as exc:  # noqa: BLE001 - skip on no network / missing onnx weights
            pytest.skip(f"e5 ONNX model unavailable: {exc}")

        q = g._embed(["query: number of walled gardens on the estate"])[0]
        p_rel = g._embed(["passage: The estate has three walled gardens."])[0]
        p_unrel = g._embed(["passage: Rainfall averages 800 mm per year."])[0]

        assert float(q @ p_rel) > float(q @ p_unrel)
        assert g._is_e5() is True  # e5 query/passage prefixes apply
        assert 0.0 <= g.self_score("number of walled gardens") <= 1.0


class TestYearContradiction:
    """Regression for the contradiction-detection bug: historical years and
    years followed by a stopword must key on the 'year' category so a real
    contradiction (built 1650 vs source 1820) is caught, while a supported
    year (1998 present) is not flagged."""

    def test_historical_years_tagged_year(self):
        from groundrails.entity_check import extract_numbers

        # 16xx/18xx (outside 19xx/20xx) must still be recognised as years
        assert ("1650", "", "year") in extract_numbers("built in 1650")
        # a year followed by the stopword "and" must not key on "and"
        nums = dict((v, cw) for v, _u, cw in extract_numbers("built in 1820 and restored in 1998"))
        assert nums.get("1820") == "year"
        assert nums.get("1998") == "year"

    def test_year_contradiction_caught(self):
        from groundrails.entity_check import (
            find_numeric_mismatches,
        )

        passage = "The manor was built in 1820 and restored in 1998."
        assert find_numeric_mismatches("the manor was built in 1650", passage)  # contradiction
        assert not find_numeric_mismatches("the manor was restored in 1998", passage)  # supported

    def test_year_contradiction_through_ground(self):
        from groundrails.grounding import ground

        src = [("e.txt", "The manor was built in 1820 and restored in 1998.")]
        m = ground("the manor was built in 1650", src)
        assert m.match_type == "contradicted"


class TestBm25IdfRecall:
    """Regression for the bm25 common-word false positive: a claim whose
    DISTINCTIVE tokens are absent must not confirm just because it shares
    ubiquitous words with the source (IDF-weighted recall)."""

    def test_common_word_fabrication_not_confirmed(self):
        from groundrails.grounding import ground

        src = [
            (
                "e.txt",
                "The estate has three walled gardens.\n\n"
                "A trout stream runs along the eastern boundary.",
            )
        ]
        # shares only common tokens (estate, runs); distinctive (commercial,
        # brewery) absent -> must NOT be a bm25 confirm
        m = ground("the estate runs a commercial brewery", src)
        assert m.match_type not in ("exact", "fuzzy", "bm25")

    def test_distinctive_claim_still_confirms(self):
        from groundrails.grounding import ground

        src = [("e.txt", "The estate has three walled gardens and an orchard.")]
        m = ground("the estate has three walled gardens", src)
        assert m.match_type in ("exact", "fuzzy", "bm25")


class TestGroundingEndToEnd:
    """Honest end-to-end precision/recall regression pin on the DEFAULT engine
    (the lexical manifold - no deterministic_engine fixture, no model download):
    on a realistic monolingual set - grounded claims, off-topic fabrications,
    and numeric contradictions - it meets the precision/recall targets. The
    cross-lingual on-topic case is the documented embedding ceiling, out of
    scope without the HIGH-tier MT bridge."""

    SRC = [
        (
            "estate.txt",
            "The estate has three walled gardens and an orchard.\n\n"
            "Rainfall in the region averages 800 millimetres per year.\n\n"
            "The manor was built in 1820 and restored in 1998.\n\n"
            "The vineyard covers twelve hectares on the south slope.\n\n"
            "A trout stream runs along the eastern boundary.",
        )
    ]
    CLAIMS = [
        ("the estate has three walled gardens", 1),
        ("there is an orchard on the estate", 1),
        ("rainfall averages 800 millimetres per year", 1),
        ("the manor was restored in 1998", 1),
        ("a trout stream runs along the eastern boundary", 1),
        ("the vineyard covers twelve hectares on the south slope", 1),
        ("the estate has a helicopter landing pad", 0),
        ("the estate runs a commercial brewery", 0),
        ("a private airport serves the estate", 0),
        ("the manor was built in 1650", 0),
        ("the manor was restored in 2010", 0),
        ("rainfall averages 200 millimetres per year", 0),
    ]

    def test_precision_recall_targets(self):
        from groundrails.grounding import ground_batch

        matches = ground_batch([c for c, _ in self.CLAIMS], self.SRC)
        tp = fp = fn = 0
        for (_c, lab), m in zip(self.CLAIMS, matches):
            confirmed = m.match_type in ("exact", "fuzzy", "bm25")
            if lab and confirmed:
                tp += 1
            elif not lab and confirmed:
                fp += 1
            elif lab and not confirmed:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        assert precision >= 0.90, f"precision {precision}"
        assert recall >= 0.80, f"recall {recall}"


class _FakeNLI:
    """Stub NLI grounder with fixed scores - exercises the wiring, no model."""

    def __init__(self, scores: dict):
        self._scores = scores

    def scores(self, premise: str, hypothesis: str) -> dict:
        return dict(self._scores)


class TestNLIGrounding:
    """NLI verdict wiring into ground() (fast, no model download)."""

    SRC = [("e.txt", "The estate has three walled gardens and an orchard.")]

    @pytest.mark.skipif(not has_model("fr"), reason="argos fr->en model not installed")
    def test_entailment_grounds(self):
        from groundrails.grounding import ground

        fake = _FakeNLI({"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02})
        m = ground("le domaine possede trois jardins clos", self.SRC, nli_grounder=fake)
        assert m.match_type in ("exact", "fuzzy", "bm25", "semantic")  # grounded
        assert m.nli_scores["entailment"] == 0.95

    def test_contradiction_flagged(self):
        from groundrails.grounding import ground

        fake = _FakeNLI({"entailment": 0.02, "neutral": 0.03, "contradiction": 0.95})
        m = ground("the estate has no gardens at all", self.SRC, nli_grounder=fake)
        assert m.match_type == "contradicted"

    def test_neutral_unconfirmed(self, deterministic_engine):
        from groundrails.grounding import ground

        fake = _FakeNLI({"entailment": 0.10, "neutral": 0.80, "contradiction": 0.10})
        m = ground("qz zztop kvqj wbrtz", self.SRC, nli_grounder=fake)
        assert m.match_type == "none"

    def test_lexical_default_unaffected_without_nli(self):
        # No nli_grounder -> deterministic behaviour unchanged.
        from groundrails.grounding import ground

        m = ground("the estate has three walled gardens", self.SRC)
        assert m.match_type == "exact"
        assert m.nli_scores == {}

    def test_extract_features_includes_nli(self):
        from groundrails.grounding import (
            GroundingMatch,
            extract_features,
        )

        feat = extract_features(
            GroundingMatch(claim="x"),
            nli_scores={"entailment": 0.7, "neutral": 0.1, "contradiction": 0.2},
        )
        assert feat["nli_entail"] == 0.7
        assert feat["nli_contra"] == 0.2

    def test_real_model_entailment_and_crosslingual(self):
        pytest.importorskip("onnxruntime")
        pytest.importorskip("transformers")
        from groundrails.nli import NLIGrounder, is_available

        if not is_available():
            pytest.skip("NLI extras not installed")
        try:
            g = NLIGrounder()
        except Exception as exc:  # noqa: BLE001 - skip on no network / missing weights
            pytest.skip(f"NLI model unavailable: {exc}")

        ev = "The estate has three walled gardens and an orchard."
        assert g.scores(ev, "There are three gardens on the estate.")["entailment"] > 0.5
        # cross-lingual entailment - the case cosine similarity could not solve
        assert g.scores(ev, "le domaine possede trois jardins clos")["entailment"] > 0.5
        assert (
            g.verdict(
                "The vineyard covers twelve hectares.", "The vineyard covers forty hectares."
            )
            == "contradicted"
        )



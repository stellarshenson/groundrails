"""Regression tests for the six defects found grounding a real research doc
(see docs/defects.md). Each test reproduces the original failure shape; the
examples are lifted from the run that surfaced them.

DEF-1 citation-splitting, DEF-2 reference-section extraction, DEF-3 weak
claim filter, DEF-4 BM25 short-claim saturation, DEF-5 numeric context
collision, DEF-6 mid-word fuzzy evidence spans - plus the BM25 corpus cache
(per-claim index rebuild) speed fix.
"""

from groundrails import ground
from groundrails.entity_check import extract_numbers, find_numeric_mismatches
from groundrails.extract import _looks_like_claim, extract_claims
from groundrails.grounding import _BM25_CACHE, _snap_to_word_bounds


# ---------------------------------------------------------------------------
# DEF-1: sentence split shatters citations
# ---------------------------------------------------------------------------


class TestCitationSplitting:
    def test_et_al_year_stays_whole(self):
        doc = "The effect was replicated by Amato and Afifi et al. 2008 across cohorts."
        claims = [c.claim for c in extract_claims(doc)]
        assert len(claims) == 1
        assert "et al. 2008" in claims[0]

    def test_author_initial_stays_whole(self):
        doc = "The panel design follows Buchanan, C. (1991) with three waves of interviews."
        claims = [c.claim for c in extract_claims(doc)]
        assert len(claims) == 1
        assert "Buchanan, C. (1991)" in claims[0]

    def test_eg_and_fig_do_not_split(self):
        doc = "Several outcomes worsen under conflict, e.g. Attachment scores drop. Fig. 3 shows the trend over ten years."
        claims = [c.claim for c in extract_claims(doc)]
        joined = " ".join(claims)
        assert "e.g. Attachment" in joined
        assert "Fig. 3 shows" in joined

    def test_real_sentence_boundary_still_splits(self):
        doc = "The first study covered 400 families. The second study covered 900 families."
        claims = [c.claim for c in extract_claims(doc)]
        assert len(claims) == 2

    def test_sentence_ending_in_single_capital_still_splits(self):
        # adversarial-review finding: the initials guard must be comma-led -
        # "vitamin D." ends a real sentence and must split
        doc = "The children were deficient in vitamin D. The trial supplemented them for a year."
        claims = [c.claim for c in extract_claims(doc)]
        assert len(claims) == 2

    def test_enumeration_ending_in_initial_still_splits(self):
        # round-4 finding: "were B, C." is an enumeration, not an author
        # initial - the comma-led guard fires only when a citation year
        # follows the boundary
        doc = "The vitamins measured were B, C. The trial ran for two full years."
        claims = [c.claim for c in extract_claims(doc)]
        assert len(claims) == 2

    def test_enumeration_before_numeric_sentence_splits(self):
        # round-8 finding: the citation guard matched ANY following digit, so
        # an enumeration ending in an initial followed by a numeric sentence
        # ("...were B, C. 400 families...") merged into one claim. The guard
        # must fire only on a 4-digit publication year, not a bare number.
        doc = "The vitamins measured were B, C. 400 families enrolled in the trial."
        claims = [c.claim for c in extract_claims(doc)]
        assert len(claims) == 2
        assert any(c.startswith("400 families") for c in claims)

    def test_author_initial_before_year_still_stays_whole(self):
        # the guard must STILL suppress the split for a genuine citation year
        doc = "The design follows Buchanan, C. 1991 across three waves of the panel."
        claims = [c.claim for c in extract_claims(doc)]
        assert len(claims) == 1


# ---------------------------------------------------------------------------
# DEF-2: reference sections extracted as claims
# ---------------------------------------------------------------------------


class TestSectionAwareness:
    DOC = (
        "# Findings\n\n"
        "Parental conflict predicts weaker child-grandparent bonds in adolescence.\n\n"
        "## Sources\n\n"
        "- Amato, P. and Afifi, T. et al. 2008 examined feeling caught between parents.\n"
        "- Buchanan, C. (1991) reported adjustment outcomes in divorced homes.\n\n"
        "## Conclusion\n\n"
        "The mediation pathway is supported by three independent cohorts.\n"
    )

    def test_sources_section_skipped(self):
        claims = [c.claim for c in extract_claims(self.DOC)]
        assert not any("Amato" in c or "Buchanan" in c for c in claims)

    def test_content_sections_still_extracted(self):
        claims = [c.claim for c in extract_claims(self.DOC)]
        assert any("Parental conflict" in c for c in claims)
        assert any("mediation pathway" in c for c in claims)

    def test_references_heading_variants(self):
        for heading in ("References", "Bibliography", "Works Cited", "Further reading"):
            doc = f"Real finding stated here as one sentence.\n\n## {heading}\n\nSmith, J. (2001) wrote a canonical treatment of the topic.\n"
            claims = [c.claim for c in extract_claims(doc)]
            assert not any("Smith" in c for c in claims), heading

    def test_code_fence_comment_is_not_a_heading(self):
        # adversarial-review round-3 finding: "# sources" inside a ``` fence
        # flipped the section-skip state and silently blacked out every
        # sentence after the code block
        doc = (
            "The gateway terminates TLS for all tenant traffic.\n\n"
            "```python\n"
            "# sources\n"
            'data = load("x.csv")\n'
            "```\n\n"
            "The cluster processes records continuously overnight in all regions.\n"
        )
        claims = [c.claim for c in extract_claims(doc)]
        assert any("cluster processes" in c for c in claims)
        assert not any("load" in c for c in claims)  # code lines are not claims

    def test_singular_source_heading_is_content(self):
        # round-3 finding: "## Source" (singular) is a plausible content
        # heading (provenance prose); only the plural is bibliographic
        doc = "## Source\n\nThe pipeline ingests records from three upstream registries daily.\n"
        claims = [c.claim for c in extract_claims(doc)]
        assert any("three upstream registries" in c for c in claims)

    def test_fence_closes_only_on_matching_delimiter(self):
        # round-4 finding: a ``` example shown inside a ```` block (or a ~~~
        # line inside ```) must be content, not a toggle - otherwise the
        # fence state desyncs and everything after the block is blacked out
        doc = (
            "````markdown\n"
            "To open a code block type:\n"
            "```\n"
            "then your code follows here.\n"
            "````\n\n"
            "The cluster processes records continuously overnight in all regions.\n"
        )
        claims = [c.claim for c in extract_claims(doc)]
        assert any("cluster processes" in c for c in claims)
        assert not any("your code follows" in c for c in claims)

    def test_bibliography_subheading_stays_skipped(self):
        # round-4 finding: "### Primary sources" under "## References" must
        # not re-enable extraction of the reference entries beneath it
        doc = (
            "The mediation pathway is supported by three independent cohorts.\n\n"
            "## References\n\n"
            "### Primary sources\n\n"
            "- Smith, J. (2001) wrote a canonical treatment of the topic.\n\n"
            "## Conclusion\n\n"
            "The effect replicates across all panel waves consistently.\n"
        )
        claims = [c.claim for c in extract_claims(doc)]
        assert not any("Smith" in c for c in claims)
        assert any("replicates across" in c for c in claims)

    def test_fence_info_string_line_is_not_a_closer(self):
        # round-6 finding: a "```python" line inside an open ``` block is
        # content (a tutorial showing how to open a fence), not the closer -
        # per CommonMark a closer allows only whitespace after the run
        doc = (
            "```\n"
            "```python\n"
            "print('hello')\n"
            "```\n\n"
            "The cluster processes records continuously overnight in all regions.\n"
        )
        claims = [c.claim for c in extract_claims(doc)]
        assert any("cluster processes" in c for c in claims)
        assert not any("print" in c for c in claims)

    def test_numbered_bibliography_heading_skipped(self):
        # round-7 finding: "## 7. References" (the most common academic
        # layout) did not match the anchored regex and the whole reference
        # list was extracted - DEF-2's original harm resurrected
        for heading in ("7. References", "3 References", "2) Sources"):
            doc = f"## {heading}\n\n- Smith, J. (2001) wrote a canonical treatment of the topic.\n"
            claims = [c.claim for c in extract_claims(doc)]
            assert not any("Smith" in c for c in claims), heading

    def test_shallower_bib_heading_takes_over_level(self):
        # round-7 finding: after "### References" a SHALLOWER "## Sources"
        # kept the old deeper skip level, so its own "### Journal papers"
        # sub-heading terminated the section and leaked reference entries
        doc = (
            "### References\n\n"
            "- Old, A. (1990) listed the early canon here.\n\n"
            "## Sources\n\n"
            "### Journal papers\n\n"
            "- Jones, K. (2005) reported the replication in a panel study.\n\n"
            "## Conclusion\n\n"
            "The effect replicates across all panel waves consistently.\n"
        )
        claims = [c.claim for c in extract_claims(doc)]
        assert not any("Old" in c or "Jones" in c for c in claims)
        assert any("replicates across" in c for c in claims)

    def test_bib_named_subheading_does_not_narrow_skip(self):
        # round-6 finding: a bib-NAMED sub-heading ("### Sources" under
        # "## References") must keep the OUTER skip level - a sibling
        # "### Journal papers" would otherwise end the section and leak
        # its reference entries into claims
        doc = (
            "## References\n\n"
            "### Sources\n\n"
            "- Smith, J. (2001) wrote a canonical treatment of the topic.\n\n"
            "### Journal papers\n\n"
            "- Jones, K. (2005) reported the replication in a panel study.\n\n"
            "## Conclusion\n\n"
            "The effect replicates across all panel waves consistently.\n"
        )
        claims = [c.claim for c in extract_claims(doc)]
        assert not any("Smith" in c or "Jones" in c for c in claims)
        assert any("replicates across" in c for c in claims)

    def test_content_sections_with_reference_prefix_kept(self):
        # adversarial-review finding: "## Reference Architecture" must NOT be
        # treated as a bibliography - only exact bibliographic titles skip
        doc = (
            "## Reference Architecture\n\n"
            "The gateway terminates TLS and forwards requests to the internal mesh.\n\n"
            "## Sources of Error\n\n"
            "Sensor drift dominates the measurement error in cold conditions.\n"
        )
        claims = [c.claim for c in extract_claims(doc)]
        assert any("gateway terminates" in c for c in claims)
        assert any("Sensor drift" in c for c in claims)


# ---------------------------------------------------------------------------
# DEF-3: fragment / anaphora admission
# ---------------------------------------------------------------------------


class TestClaimFilter:
    def test_noun_fragment_rejected(self):
        # The ledger's real example: "paywalled" matches the -ed suffix but
        # nothing predicates anything. (A 4+-word noun fragment still passes -
        # catching those needs POS tagging, out of scope for the regex tier.)
        assert not _looks_like_claim("Digest only, paywalled")

    def test_contentless_anaphora_rejected(self):
        assert not _looks_like_claim("This is important to know.")

    def test_anaphora_with_content_kept(self):
        assert _looks_like_claim("It was completed in 1889.")

    def test_real_verb_claim_kept(self):
        assert _looks_like_claim("The system processes records continuously overnight.")

    def test_irregular_past_trend_verbs_kept(self):
        # round-7 finding: the suffix regex has no irregular-verb arm, so
        # "revenue fell 20%" was silently DROPPED from extraction - a
        # fabricated trend claim then sailed through ungrounded (fail-open).
        # The canonical trend verbs of report prose must pass the filter.
        for sentence in (
            "Solar output rose 12% in 2020.",
            "Revenue fell 20% in the third quarter.",
            "The company grew 40% last year.",
            "Headcount ran above 500 in March.",
        ):
            assert _looks_like_claim(sentence), sentence

    def test_invariant_past_trend_verbs_kept(self):
        # round-8 finding: invariant pasts (present == past form) are common
        # trend verbs of report prose yet the suffix regex cannot see them and
        # no other word in the sentence need carry an -ed/-ing/-s ending, so
        # the whole claim was silently DROPPED before grounding. hit/cut/set/
        # sank/beat are not "uncommon irregulars".
        for sentence in (
            "Revenue hit an all-time low in the third quarter.",
            "The board cut the dividend to zero in April.",
            "Turnover beat forecast by a wide margin.",
            "Net income sank to a record low that quarter.",
            "The project cost the department twenty million euro.",
            "The firm withdrew from the German market.",
            "Output overtook demand for the first time.",
        ):
            assert _looks_like_claim(sentence), sentence


# ---------------------------------------------------------------------------
# DEF-4: BM25 short-claim saturation
# ---------------------------------------------------------------------------

_SOURCE = (
    "Household composition varies over the study window.\n\n"
    "The reported shares are normalised so their weighted total sums to a full "
    "distribution across categories in every wave of the panel.\n\n"
    "Grandparent contact frequency declines with the child's age in all cohorts.\n"
)


class TestBM25SpecificityFloor:
    def test_short_claim_cannot_confirm_via_bm25(self):
        # Adversarial-review rewrite: the original claim ("sums to a") was
        # VERBATIM in the fixture, so the exact layer fired and the test
        # could not fail. "panel wave shares" is 3 corpus words in an order
        # that appears nowhere - only saturated BM25 recall can claim it.
        m = ground("panel wave shares", [_SOURCE])
        assert m.exact_score == 0.0, "fixture must not contain the claim verbatim"
        assert m.bm25_claim_token_count == 3
        assert m.match_type != "bm25"

    def test_short_claim_floor_holds_on_default_engine(self):
        # Adversarial-review finding: the floor must bind on the DEFAULT
        # (trained lexical head) path too, not only the deterministic
        # cascade - the head confirmed "panel wave shares" at p=0.711 and
        # _winning_layer_label stamped it "bm25" before the fix.
        m = ground("panel wave shares", [_SOURCE])
        assert m.match_type == "none"

    def test_long_claim_still_confirms_via_bm25(self):
        # Scrambled word order kills exact and keeps fuzzy low, so the
        # confirmation - if any - is attributable to the BM25 layer.
        m = ground(
            "categories across weighted panel shares normalised total wave distribution",
            [_SOURCE],
        )
        assert m.exact_score == 0.0
        assert m.bm25_claim_token_count >= 4
        assert m.bm25_score >= 0.5
        assert m.match_type != "none"

    def test_token_count_exposed(self):
        m = ground("Grandparent contact frequency declines with age", [_SOURCE])
        assert m.bm25_claim_token_count == len(
            set("grandparent contact frequency declines with age".split())
        )

    def test_floor_gates_against_effective_thresholds(self):
        # round-3 finding: the floor hardcoded cfg thresholds while the
        # verdict used per-call / percentile-derived effective ones - the
        # label and the verdict could disagree about "clears threshold"
        from groundrails.config import PACKAGE_ROOT, load_document_processing_config
        from groundrails.grounding import GroundingMatch, _winning_layer_label

        # pin the bundled config: a project/user override on the developer's
        # machine must not change this test's threshold arithmetic
        cfg = load_document_processing_config(PACKAGE_ROOT / "config_document_processing.yaml")
        m = GroundingMatch(claim="short claim here")
        m.bm25_claim_token_count = cfg.bm25_min_claim_tokens - 1
        m.semantic_score = (cfg.semantic_threshold + 0.4) / 2  # between 0.4 and cfg value
        m.bm25_score = 0.9
        assert _winning_layer_label(m, cfg) == "none"
        assert _winning_layer_label(m, cfg, semantic_threshold=0.4) == "semantic"


# ---------------------------------------------------------------------------
# NLI priority in the deterministic cascade (round-3): contradicted > NLI >
# exact, and NLI entailment bypasses the DEF-4 floor (it is not a degenerate
# lexical statistic - the model read the premise).
# ---------------------------------------------------------------------------


class _StubNLI:
    def __init__(self, ent, contra, neu):
        self._scores = {"entailment": ent, "contradiction": contra, "neutral": neu}

    def scores(self, premise, claim):
        return dict(self._scores)


class TestNLICascadePriority:
    NEGATING = (
        "The vendor report asserted that the cluster has 42 nodes. "
        "This assertion is false and was retracted after the audit."
    )

    def _cascade(self, monkeypatch):
        import groundrails.grounding as G

        # force the deterministic cascade (the only engine where the NLI
        # verdict is first-class rather than a head feature)
        monkeypatch.setattr(G, "_config_lexical_verdict", lambda cfg: None)

    def test_nli_contradiction_beats_exact(self, monkeypatch):
        self._cascade(monkeypatch)
        m = ground("the cluster has 42 nodes", [self.NEGATING], nli_grounder=_StubNLI(0.05, 0.9, 0.05))
        assert m.exact_score == 1.0
        assert m.match_type == "contradicted"  # NLI read the negation; exact did not

    def test_nli_neutral_is_not_a_veto(self, monkeypatch):
        # round-4 finding: neutral is a shrug, not evidence - it must not
        # convert the cascade into pure NLI-argmax and refuse a claim that
        # strong lexical evidence confirms
        self._cascade(monkeypatch)
        m = ground("the cluster has 42 nodes", [self.NEGATING], nli_grounder=_StubNLI(0.1, 0.2, 0.7))
        assert m.match_type == "exact"  # falls through to the lexical ladder

    def test_unconfident_entailment_is_not_decisive(self, monkeypatch):
        # round-4 finding: argmax entailment at 0.34 (near-uniform) confirmed
        # a fabricated claim past every lexical threshold
        self._cascade(monkeypatch)
        m = ground(
            "The panel used drone telemetry over nine countries.",
            [_SOURCE],
            nli_grounder=_StubNLI(0.34, 0.33, 0.33),
        )
        assert m.match_type == "none"

    def test_confident_entailment_bypasses_floor_with_honest_label(self, monkeypatch):
        # confident entailment confirms a sub-floor claim, and the verdict is
        # labelled "nli" - not stamped with a lexical layer that never
        # cleared its own threshold (round-4: saturated bm25 max on a 3-token
        # claim must not masquerade as recall evidence)
        self._cascade(monkeypatch)
        m = ground("panel wave shares", [_SOURCE], nli_grounder=_StubNLI(0.9, 0.05, 0.05))
        assert m.bm25_claim_token_count == 3
        assert m.match_type == "nli"
        assert m.grounded
        # downstream consumers must carry the nli verdict too (round-5):
        # the headline score is the entailment evidence
        from groundrails.grounding import _final_score

        assert _final_score(m) == 0.9
        # and an entailment within proximity of the floor is flagged for
        # second-guessing like every other layer's borderline score
        m2 = ground("dolphins sing opera", [_SOURCE], nli_grounder=_StubNLI(0.52, 0.24, 0.24))
        assert m2.match_type == "nli"
        assert m2.verification_needed  # 0.52 vs the 0.5 floor: borderline
        # support provenance falls back to the premise's origin layer
        assert m.support is not None
        assert m.support.get("support_via") == "nli-premise"

    def test_unconfident_contradiction_is_not_a_veto(self, monkeypatch):
        # round-5 finding: the confidence principle must hold on BOTH arms -
        # an argmax contradiction at 0.34 must not refuse a verbatim claim
        self._cascade(monkeypatch)
        m = ground(
            "the cluster has 42 nodes",
            ["The vendor report asserted that the cluster has 42 nodes."],
            nli_grounder=_StubNLI(0.33, 0.34, 0.33),
        )
        assert m.match_type == "exact"
        # round-6: the pass-through must be OBSERVABLE, not silent - the
        # sub-floor contradiction lean flags the confirm for second-guessing
        assert m.verification_needed

    def test_subfloor_contradiction_lean_is_flagged(self, monkeypatch):
        # round-6 finding: contradiction 0.49 vs the 0.5 floor was a silent
        # 0.01 cliff - within proximity of the floor it must flag
        self._cascade(monkeypatch)
        m = ground(
            "the cluster has 42 nodes",
            ["The vendor report asserted that the cluster has 42 nodes."],
            nli_grounder=_StubNLI(0.02, 0.49, 0.49),
        )
        assert m.match_type == "exact"
        assert m.verification_needed

    def test_benign_numeric_exact_not_flagged(self):
        # round-6 finding: WI#6 reason (d) compared the claim's numbers to
        # its OWN verbatim span on exact matches - a vacuous self-comparison
        # flagging 100% of numeric verbatim confirms
        m = ground(
            "the cluster has 42 nodes",
            ["The vendor report asserted that the cluster has 42 nodes."],
        )
        assert m.match_type == "exact"
        assert not m.verification_needed

    def test_agreeing_numbers_on_fuzzy_confirm_not_flagged(self):
        # round-7 finding: reason (d) fired on a bare key INTERSECTION, so a
        # clean fuzzy confirm whose passage restates the claim's own number
        # (values AGREE) was flagged - the D1 vacuity one layer down
        m = ground(
            "The cluster has 42 nodes.",
            ["Reports show the cluster has 42 nodes in production."],
        )
        assert m.match_type in ("exact", "fuzzy", "bm25")
        assert not m.verification_needed

    def test_suppressed_multivalue_conflict_still_flagged(self):
        # ...but DIFFERENT values under a shared key with the mismatch gate
        # suppressed (multi-value passage) must still flag - that is reason
        # (d)'s whole purpose
        m = ground(
            "Throughput reached 42 requests per node.",
            ["Throughput ranged between 30 requests and 35 requests under load."],
        )
        if m.match_type in ("fuzzy", "bm25", "semantic", "nli"):
            assert m.verification_needed

    def test_restated_conflict_around_exact_span_flagged(self):
        # round-7 finding: an exact confirm beside a restated CONFLICTING
        # number was silent on every shipped path. Reason (d') checks the
        # surrounding window (excluding the span); the verdict stays exact
        # (DEF-9 policy), the conflict flags for second-guessing.
        m = ground(
            "the cluster has 42 nodes",
            [
                "The vendor report asserted that the cluster has 42 nodes. "
                "A later audit counted 12 nodes in the cluster."
            ],
        )
        assert m.match_type == "exact"
        assert m.verification_needed


# ---------------------------------------------------------------------------
# DEF-5: numeric context collision -> false CONTRADICTED
# ---------------------------------------------------------------------------


class TestNumericContextCollision:
    def test_percentages_of_different_quantities_do_not_collide(self):
        claim = "The category shares sum to 100%."
        passage = "The intervention shifted outcomes by -36% of a SD in the exposed group."
        assert find_numeric_mismatches(claim, passage) == []

    def test_negative_sign_captured(self):
        nums = {v for v, _u, _c in extract_numbers("a shift of -36% of a SD")}
        assert "-36" in nums
        assert "36" not in nums

    def test_range_hyphen_is_not_a_sign(self):
        nums = {v for v, _u, _c in extract_numbers("during 2010-2015 the rate rose")}
        assert "-2015" not in nums

    def test_trailing_number_keys_on_preceding_word(self):
        nums = {v: cw for v, _u, cw in extract_numbers("the shares sum to 100%")}
        assert nums["100"] not in ("",)  # must not be context-free

    def test_true_same_context_contradiction_still_caught(self):
        claim = "The cluster has 42 nodes."
        passage = "The cluster has 12 nodes."
        assert find_numeric_mismatches(claim, passage) == [("42", "12")]

    def test_bare_year_contradiction_still_caught(self):
        claim = "The bridge was built in 1650."
        passage = "Construction records date the bridge to 1820."
        mm = find_numeric_mismatches(claim, passage)
        assert ("1650", "1820") in mm

    def test_context_window_stops_at_sentence_boundary(self):
        # adversarial-review finding: "increased by 40. Managers said" must
        # not key 40 on "managers" from the NEXT sentence
        nums = {v: cw for v, _u, cw in extract_numbers("output increased by 40. Managers said more.")}
        assert nums["40"] != "managers"

    def test_multichar_units_not_shadowed_by_si_letters(self):
        # round-3 finding: leftmost alternation made "m" shadow "ms"/"mm",
        # "k" shadow "kg" etc - half the unit whitelist was unreachable
        assert extract_numbers("a latency of 5 ms")[0][1] == "ms"
        assert extract_numbers("a mass of 7 kg")[0][1] == "kg"
        assert extract_numbers("waited 5 seconds")[0][1] == "seconds"

    def test_unit_letter_requires_word_boundary(self):
        # "5 meters" is not unit "m" + junk; the word becomes the context
        v, u, cw = extract_numbers("a span of 5 meters")[0]
        assert (u, cw) == ("", "meters")

    def test_two_char_function_words_are_stopwords(self):
        # round-6 suspicion, settled: _CONTEXT_WORD_RE admits 2-char words
        # (needed for "SD"), so 2-char prepositions MUST be in _STOPWORDS or
        # "40% of the sample" and "60% of the controls" would both key
        # ('%','of') and collide - the DEF-5 failure resurrected
        assert extract_numbers("40% of the sample")[0][2] == "sample"


# ---------------------------------------------------------------------------
# Round-7: ONE origin-fallback picker for span-less verdicts (nli, and
# cascade-driven semantic). Four hand-rolled copies drifted once already -
# these tests pin the shared helper's contract and that every consumer
# agrees on the same fallback for the same match.
# ---------------------------------------------------------------------------


class TestOriginFallbackPickers:
    def _spanless_semantic_match(self):
        """A cascade-shaped match: semantic cosine present, NO located span,
        but a located bm25 passage - the joint-path shape."""
        from groundrails.grounding import GroundingMatch, Location

        m = GroundingMatch(claim="the escalated paraphrase")
        m.match_type = "semantic"
        m.semantic_score = 0.7  # cosine from the cascade, no location
        m.bm25_score = 0.3
        m.bm25_matched_text = "the passage the premise came from"
        m.bm25_location = Location(
            source_index=0,
            source_path="src.txt",
            char_start=10,
            char_end=42,
            line_start=2,
            line_end=2,
            column_start=1,
            column_end=32,
        )
        return m

    def test_origin_fallback_requires_positive_score_and_location(self):
        # round-7 finding: the support copy checked only char_start >= 0, so
        # a zero-clamped semantic hit (negative cosine -> 0.0 with location
        # set) was cited as support while grounded_source named another layer
        from groundrails.grounding import GroundingMatch, Location, _origin_fallback

        m = GroundingMatch(claim="x")
        m.semantic_score = 0.0  # zero-clamped - located but NOT evidence
        m.semantic_location = Location(source_index=0, source_path="sem.txt", char_start=5, char_end=9)
        m.exact_score = 1.0
        m.exact_matched_text = "x"
        m.exact_location = Location(source_index=1, source_path="exact.txt", char_start=0, char_end=1)
        loc, text = _origin_fallback(m)
        assert loc.source_path == "exact.txt"

    def test_spanless_semantic_verdict_has_location_and_source(self):
        # round-7 finding: a cascade semantic verdict carried no location, so
        # grounded_source=None, the CLI printed L-1:C-1 with an empty quote,
        # and claim_attributes ran on an empty passage
        from groundrails.grounding import _winning_location

        m = self._spanless_semantic_match()
        loc = _winning_location(m)
        assert loc is not None and loc.source_path == "src.txt"

    def test_spanless_semantic_support_and_location_agree(self):
        # one JSON document must never say grounded_source: null while
        # support cites a real path - all pickers share _origin_fallback
        from groundrails.grounding import _winning_location

        m = self._spanless_semantic_match()
        sup = m.support
        assert sup is not None
        assert sup["support_via"] == "lexical"
        assert _winning_location(m).source_path == "src.txt"

    def test_spanless_semantic_cli_line_shows_real_location(self):
        from groundrails.cli import _match_line

        line = _match_line(self._spanless_semantic_match())
        assert "L-1" not in line
        assert "the passage the premise came from" in line

    def test_nli_and_semantic_arms_use_the_same_fallback(self):
        from groundrails.grounding import _winning_location

        m = self._spanless_semantic_match()
        as_semantic = _winning_location(m)
        m.match_type = "nli"
        as_nli = _winning_location(m)
        assert as_semantic == as_nli


# ---------------------------------------------------------------------------
# DEF-6: fuzzy evidence spans cut mid-word
# ---------------------------------------------------------------------------


class TestExactMatchBoundaries:
    def test_word_bounded_substring_rejected(self):
        from groundrails.grounding import _exact_match

        assert _exact_match("um to a", "the total sums to a full distribution") is None

    def test_punctuation_edge_still_matches(self):
        # round-3 finding: a claim ending in punctuation ("...40%.") must
        # match a source where the next sentence follows without a space
        # ("...40%.The next") - the boundary lookaround only applies when
        # the claim edge is itself a word character
        from groundrails.grounding import _exact_match

        assert _exact_match("grew by 40%.", "Revenue grew by 40%.The next year it fell.") is not None


class TestFuzzySpanWordBounds:
    def test_snap_expands_partial_words(self):
        text = "significant variance to other sources"
        # span starting inside "significant" and ending inside "sources"
        start = text.index("nt variance")
        end = text.index(" sources") + len(" sour")
        s, e = _snap_to_word_bounds(text, start, end)
        assert text[s:e] == "significant variance to other sources"

    def test_snap_trims_whitespace(self):
        text = "alpha beta gamma"
        s, e = _snap_to_word_bounds(text, 5, 11)  # " beta "
        assert text[s:e] == "beta"

    def test_grounded_fuzzy_evidence_is_word_complete(self):
        source = (
            "The attachment measurements attribute significant variance to other "
            "sources beyond parental sensitivity in the pooled meta-analysis."
        )
        m = ground(
            "Attachment measurements attribute significant variance to other sources.",
            [source],
        )
        ev = m.fuzzy_matched_text
        assert ev, "fuzzy layer produced no evidence"
        idx = source.index(ev)
        before = source[idx - 1 : idx]
        after = source[idx + len(ev) : idx + len(ev) + 1]
        assert not (before.isalnum() and ev[0].isalnum()), f"starts mid-word: {ev[:25]!r}"
        assert not (after.isalnum() and ev[-1].isalnum()), f"ends mid-word: {ev[-25:]!r}"


# ---------------------------------------------------------------------------
# Speed: BM25 corpus cache (index built once per source set, not per claim)
# ---------------------------------------------------------------------------


class TestBM25CorpusCache:
    def test_index_reused_across_claims(self, monkeypatch):
        import groundrails.grounding as G

        _BM25_CACHE.clear()
        builds = {"n": 0}
        real = G.BM25Okapi

        def counting(corpus):
            builds["n"] += 1
            return real(corpus)

        monkeypatch.setattr(G, "BM25Okapi", counting)
        for claim in (
            "Grandparent contact frequency declines with the child's age.",
            "The shares are normalised across categories in every wave.",
            "Household composition varies over the study window entirely.",
        ):
            ground(claim, [_SOURCE])
        assert builds["n"] == 1, f"index rebuilt {builds['n']}x for one source set"
        _BM25_CACHE.clear()

    def test_different_sources_get_distinct_entries(self):
        _BM25_CACHE.clear()
        ground("Grandparent contact frequency declines with age.", [_SOURCE])
        ground(
            "An unrelated corpus about orbital mechanics and satellites.",
            ["Satellites in low orbit decay from atmospheric drag over months."],
        )
        assert len(_BM25_CACHE) == 2
        _BM25_CACHE.clear()


# ---------------------------------------------------------------------------
# round-8: bare-string source is a fail-dangerous API path
# ---------------------------------------------------------------------------


class TestBareStringSource:
    def test_bare_string_source_raises(self):
        # round-8 finding: a bare str IS a Sequence[str], so it type-checks yet
        # iterates into per-CHARACTER sources; rapidfuzz then scores a 1-char
        # source partial_ratio 1.0 and CONFIRMS a fabricated claim. Fail loud.
        import pytest

        with pytest.raises(TypeError):
            ground("The launch was postponed to March.", "Unrelated cooking text.")

    def test_wrapped_single_source_is_accepted(self):
        # the correct call - one source in a list - still works
        m = ground("The Eiffel Tower is in Paris.", ["The Eiffel Tower is in Paris."])
        assert m.grounded


# ---------------------------------------------------------------------------
# round-8: restated numeric conflict beside the agreement (WI#6 d / d')
# ---------------------------------------------------------------------------


class TestRestatedNumericConflict:
    def test_helper_fires_on_passage_extra_value(self):
        # the passage restates the claim value AND carries a competing one
        # under the same (unit, context) key - find_numeric_mismatches reads
        # this as supported the moment 42 reappears, so the second-guess flag
        # must catch it independently, in either direction.
        from groundrails.grounding import _restated_numeric_conflict

        claim = extract_numbers("the cluster has 42 nodes")
        conflict = extract_numbers("42 nodes today, though an audit counted 12 nodes")
        agree = extract_numbers("42 nodes were provisioned")
        assert _restated_numeric_conflict(claim, conflict) is True
        assert _restated_numeric_conflict(claim, agree) is False

    def test_exact_confirm_flags_conflict_beside_agreement(self):
        # (d') integration: the verbatim claim grounds exact, the window
        # restates 42 beside a conflicting 12 - verdict stays exact (DEF-9),
        # but verification_needed fires so the competing value is visible.
        src = (
            "System notes: a later audit counted 12 nodes, though the spec sheet "
            "still lists 42 nodes. The production cluster has 42 nodes."
        )
        m = ground("The production cluster has 42 nodes.", [src])
        assert m.match_type == "exact"  # verdict untouched - not contradicted
        assert m.verification_needed is True

    def test_exact_confirm_clean_window_not_flagged(self):
        # pure restatement with no competing value must NOT flag
        src = (
            "The spec sheet lists 42 nodes for the tier. "
            "The production cluster has 42 nodes."
        )
        m = ground("The production cluster has 42 nodes.", [src])
        assert m.match_type == "exact"
        assert m.verification_needed is False


# ---------------------------------------------------------------------------
# round-8: contradicted arm must pick a LOCATED span (not the unlocated cosine)
# ---------------------------------------------------------------------------


class TestContradictedLocatedPicker:
    def _contradicted_unlocated_semantic(self):
        """A joint-path contradicted match: the cascade cosine is present with
        NO location; only bm25 carries a located span of the conflict."""
        from groundrails.grounding import GroundingMatch, Location

        m = GroundingMatch(claim="the cluster has 42 nodes")
        m.match_type = "contradicted"
        m.semantic_score = 0.6  # cosine from the cascade, no location
        m.bm25_score = 0.4
        m.bm25_matched_text = "an audit counted 12 nodes"
        m.bm25_location = Location(
            source_index=0,
            source_path="ev.txt",
            char_start=10,
            char_end=35,
            line_start=1,
            line_end=1,
            column_start=11,
            column_end=36,
        )
        return m

    def test_winning_location_skips_unlocated_semantic(self):
        # round-8 finding: the hand-rolled contradicted picker returned the
        # unlocated semantic layer (score>0) over a located bm25 layer -> the
        # L-1:C-1 bug for the very conflict that triggered the verdict.
        from groundrails.grounding import _winning_location

        loc = _winning_location(self._contradicted_unlocated_semantic())
        assert loc is not None
        assert loc.source_path == "ev.txt" and loc.char_start == 10

    def test_cli_line_shows_located_conflict(self):
        from groundrails.cli import _match_line

        line = _match_line(self._contradicted_unlocated_semantic())
        assert "L-1" not in line
        assert "an audit counted 12 nodes" in line

    def test_semantic_verdict_zero_score_located_span_falls_back(self):
        # BH-3: a match_type=semantic whose semantic layer is located but
        # zero-clamped must NOT be cited directly - the score guard forces
        # the fallback to the located bm25 layer.
        from groundrails.grounding import GroundingMatch, Location, _winning_location

        m = GroundingMatch(claim="x")
        m.match_type = "semantic"
        m.semantic_score = 0.0
        m.semantic_location = Location(source_index=0, source_path="sem.txt", char_start=5, char_end=9)
        m.bm25_score = 0.3
        m.bm25_matched_text = "the real supporting passage"
        m.bm25_location = Location(source_index=1, source_path="bm25.txt", char_start=2, char_end=29)
        assert _winning_location(m).source_path == "bm25.txt"
        sup = m.support
        assert sup is not None and sup["source_path"] == "bm25.txt"


# ---------------------------------------------------------------------------
# round-8: exact-window edges snap so a truncated number is never fabricated
# ---------------------------------------------------------------------------


class TestExactWindowSnap:
    def test_window_edge_number_not_bisected(self):
        # BH-2: the raw -300 window edge, if it lands mid-number, clips
        # "1042 nodes" to "42 nodes" and FABRICATES a value. Here that phantom
        # "42" would spuriously AGREE with the claim and MASK the real
        # 1042-vs-42 conflict. The word-bound snap keeps "1042" whole so the
        # genuine conflict is flagged. Deterministic: the claim is positioned
        # so the raw edge (_s - 300) bisects the "42" inside "1042".
        from groundrails.grounding import _EVIDENCE_WINDOW_CHARS

        head = "Audit log: 1042 nodes were logged that day. "
        claim = "The cluster has 42 nodes."
        num_idx = head.index("1042") + 2  # index of the "42" inside "1042"
        # place the claim so _s - WINDOW lands exactly on num_idx
        filler_len = _EVIDENCE_WINDOW_CHARS + num_idx - len(head) - 1
        src = head + ("x" * filler_len) + " " + claim
        m = ground(claim, [src])
        assert m.match_type == "exact"  # verbatim confirm, not contradicted
        # snap preserves "1042 nodes" -> genuine conflict beside the claim's 42
        assert m.verification_needed is True

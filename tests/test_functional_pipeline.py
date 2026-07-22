"""Functional end-to-end tests: document file -> extract -> ground -> report.

Exercise the whole pipeline the way the CLI drives it on a realistic research
document (citations, a Sources section, percentages of different quantities),
verifying the behaviours the defect fixes promise - at the pipeline level,
not the unit level. Also pins batch determinism under threading, which is
what makes ``--workers 5`` trustworthy with the shared caches.
"""

import json

from groundrails import ground, ground_batch
from groundrails.cli import main

DOCUMENT = """# Parental influence on child-relative bonds

Parental conflict predicts weaker child-grandparent bonds in adolescence.
The pooled estimate attributes significant variance to maternal gatekeeping
across three-generation households, e.g. Attachment scores drop under
sustained loyalty conflict.

The category shares sum to 100% in every wave of the panel.

## Sources

- Amato, P. and Afifi, T. et al. 2008 examined feeling caught between parents.
- Buchanan, C. (1991) reported adjustment outcomes in divorced homes.
"""

EVIDENCE = """The cohort review found that parental conflict predicts weaker \
child-grandparent bonds in adolescence. The pooled estimate attributes \
significant variance to maternal gatekeeping across three-generation \
households, e.g. Attachment scores drop under sustained loyalty conflict.

The category shares sum to 100% in every wave of the panel. The intervention \
shifted exposure by -36% of a SD in the treated group.
"""


class TestExtractGroundPipeline:
    def _extract(self, tmp_path, capsys):
        doc = tmp_path / "answer.md"
        doc.write_text(DOCUMENT, encoding="utf-8")
        out = tmp_path / "claims.json"
        rc = main(["extract-claims", "--document", str(doc), "--output", str(out)])
        assert rc == 0
        capsys.readouterr()
        return json.loads(out.read_text(encoding="utf-8"))

    def test_extraction_yields_only_body_claims(self, tmp_path, capsys):
        claims = self._extract(tmp_path, capsys)
        texts = [c["claim"] for c in claims]
        # reference entries must not become claims (DEF-2)
        assert not any("Amato" in t or "Buchanan" in t for t in texts)
        # citations inside body text stay whole (DEF-1)
        assert not any(t.strip().startswith(("2008", "(1991)")) for t in texts)
        # the real assertions are present
        assert any("Parental conflict" in t for t in texts)
        assert any("100%" in t for t in texts)

    def test_extracted_claims_ground_cleanly(self, tmp_path, capsys):
        claims = self._extract(tmp_path, capsys)
        matches = ground_batch([c["claim"] for c in claims], [EVIDENCE])
        by_claim = {m.claim: m for m in matches}
        # the verbatim-supported claim grounds
        supported = next(m for c, m in by_claim.items() if "Parental conflict" in c)
        assert supported.match_type in ("exact", "fuzzy", "bm25", "semantic")
        # the shares claim must NOT be contradicted by the -36% figure (DEF-5)
        shares = next(m for c, m in by_claim.items() if "100%" in c)
        assert shares.match_type != "contradicted"
        assert shares.numeric_mismatches == []

    def test_cli_document_mode_end_to_end(self, tmp_path, capsys):
        doc = tmp_path / "answer.md"
        doc.write_text(DOCUMENT, encoding="utf-8")
        ev = tmp_path / "evidence.txt"
        ev.write_text(EVIDENCE, encoding="utf-8")
        rc = main(["ground", str(doc), str(ev)])
        out = capsys.readouterr().out
        assert rc == 0, f"expected all claims grounded, got:\n{out}"
        assert "CONTRADICTED" not in out

    def test_fabricated_claim_fails_grounding(self, tmp_path, capsys):
        doc = tmp_path / "answer.md"
        doc.write_text(
            "The panel covered 40000 families across nine countries with drone telemetry.\n",
            encoding="utf-8",
        )
        ev = tmp_path / "evidence.txt"
        ev.write_text(EVIDENCE, encoding="utf-8")
        rc = main(["ground", str(doc), str(ev)])
        capsys.readouterr()
        assert rc == 1  # ungrounded claim -> exit 1

    def test_contradicted_claim_fails_grounding(self, tmp_path, capsys):
        # round-7 finding: the exit gate was `match_type != "none"`, so a
        # CONTRADICTED claim - the loudest failure - exited 0. The gate now
        # uses the canonical GroundingMatch.grounded predicate.
        doc = tmp_path / "answer.md"
        doc.write_text("The cluster has 42 nodes in the datacenter.\n", encoding="utf-8")
        ev = tmp_path / "evidence.txt"
        ev.write_text("The cluster has 12 nodes in the datacenter.\n", encoding="utf-8")
        rc = main(["ground", str(doc), str(ev)])
        out = capsys.readouterr().out
        assert "CONTRADICTED" in out
        assert rc == 1

    def test_cli_fuzzy_confirm_without_flags(self, tmp_path, capsys):
        # round-7 settling test: with no threshold flags the None defaults
        # must resolve to config values on the FUZZY path too (the exact-only
        # e2e test could not catch a float >= None TypeError there)
        doc = tmp_path / "answer.md"
        doc.write_text(
            "Attachment scores drop under continued loyalty conflict.\n", encoding="utf-8"
        )
        ev = tmp_path / "evidence.txt"
        ev.write_text(EVIDENCE, encoding="utf-8")
        rc = main(["ground", str(doc), str(ev)])
        out = capsys.readouterr().out
        assert rc == 0, out
        assert "EXACT" not in out.splitlines()[0]  # confirms via a non-exact layer

    def test_cli_threshold_flags_default_to_config(self):
        # round-7 finding: --threshold/--bm25-threshold carried non-None
        # defaults that ALWAYS overrode the loaded yaml, and the two
        # semantic flags were parsed but never read. The defaults must stay
        # None so the config wins when a flag is absent.
        from groundrails.cli import _build_parser

        args = _build_parser().parse_args(["ground", "doc.md", "ev.txt"])
        assert args.threshold is None
        assert args.bm25_threshold is None
        assert args.semantic_threshold is None
        assert args.semantic_threshold_percentile is None
        # round-8 finding: --semantic carried default 0, silently overriding
        # calibration.mode: semantic on every CLI run. Absent -> None so the
        # yaml SSoT (joint.switch_on) decides, same contract as the others.
        assert args.semantic is None


class TestEvidenceQuality:
    def test_all_layer_evidence_is_word_complete(self):
        """Whatever layer wins, cited evidence must not start/end mid-word (DEF-6)."""
        claims = [
            "Attachment scores drop under sustained loyalty conflict.",
            "The pooled estimate attributes significant variance to maternal gatekeeping.",
        ]
        for claim in claims:
            m = ground(claim, [EVIDENCE])
            for ev in (m.fuzzy_matched_text, m.exact_matched_text):
                if not ev:
                    continue
                idx = EVIDENCE.find(ev)
                if idx < 0:
                    continue
                before = EVIDENCE[idx - 1 : idx]
                after = EVIDENCE[idx + len(ev) : idx + len(ev) + 1]
                assert not (before.isalnum() and ev[0].isalnum()), repr(ev[:30])
                assert not (after.isalnum() and ev[-1].isalnum()), repr(ev[-30:])


class TestBatchDeterminism:
    CLAIMS = [
        "Parental conflict predicts weaker child-grandparent bonds in adolescence.",
        "Attachment scores drop under sustained loyalty conflict.",
        "Category shares sum to 100% in every panel wave.",
        "The intervention shifted exposure by -36% of a SD.",
        "The panel used drone telemetry over nine countries.",  # fabricated
    ]

    def test_threaded_batch_matches_serial(self):
        """max_workers>1 must be a speedup, never a behaviour change - the
        shared chunk/corpus/BM25/yaml caches are exercised concurrently here.
        Repeated (races are nondeterministic) and compared on the full
        verdict surface including head probability and features, so a cache
        race that perturbs features without flipping match_type still fails."""
        serial = ground_batch(self.CLAIMS, [EVIDENCE], max_workers=1)
        for _ in range(3):
            threaded = ground_batch(self.CLAIMS, [EVIDENCE], max_workers=5)
            for s, t in zip(serial, threaded):
                assert s.claim == t.claim
                assert s.match_type == t.match_type
                assert abs(s.fuzzy_score - t.fuzzy_score) < 1e-12
                assert abs(s.bm25_score - t.bm25_score) < 1e-12
                assert abs(s.verdict_probability - t.verdict_probability) < 1e-12
                assert s.verdict_features == t.verdict_features
                assert s.numeric_mismatches == t.numeric_mismatches

    def test_repeated_batches_are_stable(self):
        """Cache hits must return identical results to the cold pass."""
        first = ground_batch(self.CLAIMS, [EVIDENCE])
        second = ground_batch(self.CLAIMS, [EVIDENCE])
        for a, b in zip(first, second):
            assert a.match_type == b.match_type
            assert a.bm25_token_recall == b.bm25_token_recall


class TestKnownLimitations:
    """Characterization pins for ACCEPTED failure modes (docs/defects.md).

    These are not aspirations - they document behaviour we knowingly ship.
    If one of these starts failing, the limitation was fixed: update the
    docs and flip the test, do not delete it.
    """

    def test_verbatim_claim_inside_negating_evidence_still_grounds(self):
        # DEF-9 tradeoff: the exact layer is a word-bounded SUBSTRING match;
        # it cannot see that the surrounding sentence negates the quoted
        # sub-phrase. Accepted: exact-wins is the better default; negation-
        # cue detection would need NLI on the surrounding window.
        evidence = (
            "The vendor report asserted that the cluster has 42 nodes. "
            "This assertion is false and was retracted after the audit."
        )
        m = ground("the cluster has 42 nodes", [evidence])
        assert m.match_type == "exact"

    def test_subfloor_paraphrase_cannot_confirm_on_lexical_tier(self):
        # DEF-4 floor tradeoff: a claim under bm25_min_claim_tokens unique
        # tokens that is NOT verbatim in the source (and clears neither the
        # fuzzy nor the semantic threshold) is always refused on the default
        # lexical tier, whatever the head probability says. Intended
        # fail-safe direction; verbatim short claims still ground via exact.
        evidence = (
            "The reported shares are normalised so their weighted total sums "
            "to a full distribution across categories in every wave of the panel."
        )
        m = ground("panel wave shares", [evidence])
        assert m.match_type == "none"
        # the refusal must be the FLOOR's doing: the head itself confirms
        # (if this drops below threshold the pin degrades to a head pin)
        assert m.verdict_probability >= 0.5
        verbatim = ground("every wave of the panel", [evidence])
        assert verbatim.match_type == "exact"

    def test_divergent_numeric_contexts_skip_comparison(self):
        # DEF-5 tradeoff: numbers are compared only within the same
        # (unit, context) key. Paraphrase that shifts the context word
        # ("overall" vs "quarter") gives divergent keys, so a real value
        # conflict across phrasings is NOT flagged - precision over recall,
        # the price of killing the false-CONTRADICTED collisions.
        from groundrails.entity_check import find_numeric_mismatches

        mm = find_numeric_mismatches(
            "The rate rose 40% overall.",
            "The rate rose by 50% in the quarter.",
        )
        assert mm == []

    def test_shared_trailing_context_can_collide_across_subjects(self):
        # DEF-5 tradeoff, inverse direction: numbers about DISJOINT subjects
        # that share a trailing context word key identically and a spurious
        # mismatch fires. Accepted with the divergent-context trade above -
        # subject/entity linking is out of the deterministic tier's scope.
        from groundrails.entity_check import find_numeric_mismatches

        mm = find_numeric_mismatches(
            "Zebra populations declined 25% in the audited quarter.",
            "Revenue grew 40% in the audited quarter.",
        )
        assert mm == [("25", "40")]

    def test_lexical_flags_do_not_gate_head_engine_verdicts(self):
        # round-7 pin: on the default trained-head engine the head owns the
        # verdict - a per-call fuzzy_threshold gates labels/verification
        # only, NOT the confirm (the deterministic cascade honours it as a
        # verdict gate). Documented in the CLI help and DEF-11.
        claim = "Attachment scores drop under continued loyalty conflict."
        m = ground(claim, [EVIDENCE], fuzzy_threshold=0.999)
        assert m.grounded  # head confirms regardless of the label threshold

    def test_signed_percentage_contradiction_is_intended_behaviour(self):
        # DEF-5 sign capture: "-36%" vs "36%" in the same context is now a
        # real contradiction where pre-fix both parsed as "36". Behaviour
        # pin for a population the golden equivalence set does not cover.
        from groundrails.entity_check import find_numeric_mismatches

        mm = find_numeric_mismatches(
            "The exposure shifted by -36% in the treated group.",
            "The exposure shifted by 36% in the treated group.",
        )
        assert mm == [("-36", "36")]

    def test_numeric_neighbor_over_flags_same_context_elaboration(self):
        # round-8/9 tradeoff: the symmetric conflict-beside-agreement flag
        # (WI#6 reason (d)/(d')) fires on ANY differing value under a shared
        # (unit, context-word) key, so it cannot tell a genuine competitor
        # ("42 confirmed, 12 also present") from benign SAME-context
        # elaboration ("400 families ... 250 families"). Distinguishing the
        # two is semantic, out of the deterministic tier's reach; the flag is
        # verification-only (the verdict is never touched) and errs fail-safe.
        # A different context word already avoids the collision.
        from groundrails.entity_check import extract_numbers
        from groundrails.grounding import _restated_numeric_conflict

        claim = extract_numbers("the study enrolled 400 families")
        elaboration = extract_numbers(
            "400 families in the main arm and 250 families in the control"
        )
        assert _restated_numeric_conflict(claim, elaboration) is True  # accepted over-flag
        distinct = extract_numbers("400 families in 2019 and 250 in the pilot cohort")
        assert _restated_numeric_conflict(claim, distinct) is False  # different context saves it

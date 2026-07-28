"""Out-of-scope claim classification: sentences that are not assertions about a
source corpus at all, and therefore must not pay for a semantic escalation.

Root cause these pin: on prose documents a large minority of extracted claims are
hypotheticals, document self-references or directives. Each one runs the lexical
tier, escalates into the OpenVINO cascade (the shipped escalation band is
0.130-0.986, so nearly everything in-band escalates) and is guaranteed to miss.

The dangerous direction is a FALSE POSITIVE - a real claim silently skipped by the
cascade - so the keep-in-scope cases below carry the weight. They are drawn from the
labelled prose set, where the rules score 16/27 recall on human ``not_groundable``
labels at zero false positives against the 24 claims a human verified as groundable
or the lexical tier actually confirmed.
"""

import pytest

from groundrails import joint, settings
from groundrails.config import load_document_processing_config
from groundrails.extract import extract_claims, out_of_scope


@pytest.fixture(autouse=True)
def _ready():
    settings.mark_ready()
    yield
    settings.reset()


class TestOutOfScopeClassification:
    @pytest.mark.parametrize(
        ("claim", "reason"),
        [
            # hypothetical - asserts a branch, not a fact
            (
                "If those outputs were never operationalised, that is a strong angle.",
                "hypothetical",
            ),
            ("Unless the archive was digitised, the prognosis track cannot start.", "hypothetical"),
            (
                "Either the outputs never reached the operations team, or they were "
                "not operationalised.",
                "hypothetical",
            ),
            # self-reference - about this document, not the world
            ("The purpose of this round was to establish what can be sold.", "self-reference"),
            ("The full pre-registration is in `resources/experiments.md`.", "self-reference"),
            (
                "Every hypothesis in the registered set carries a kill criterion.",
                "self-reference",
            ),
            ("No claim in this summary rests on a source that was not read.", "self-reference"),
            # directive - a recommendation, not a statement of fact
            ("Sell the patrol design and the site assessment that de-risks it.", "directive"),
            ("The research says most of it should not be proposed.", "directive"),
            (
                "The correct move is to restate the acceptance criterion as an error budget.",
                "directive",
            ),
        ],
    )
    def test_flags_the_three_classes(self, claim, reason):
        assert out_of_scope(claim) == reason

    @pytest.mark.parametrize(
        "claim",
        [
            # An EMBEDDED "either ... or" is a real claim's internal disjunction.
            "That benchmark was captured at 3 m standoff, so at 50-100 m either a long "
            "focal length is specified or the GSD target moves.",
            # A conditional carrying a digit has a checkable consequent.
            "If hardware sits inside the EUR 50-80k envelope, between zero and EUR 40k "
            "of engineering remains.",
            # "Note that ..." fronts a real assertion - never a directive.
            "Note too that derived crack dimensions carry relative errors from -35% to +120%.",
            # Ordinary source claims must be untouched.
            "The study reports a 3D RMSE of 7.9 cm at a standoff of 70 m.",
            "Crack IoU reached 66.76% while background IoU reached 99.76%.",
            "Uneven illumination degrades crack segmentation on concrete surfaces.",
        ],
    )
    def test_keeps_real_claims_in_scope(self, claim):
        """False positives here silently skip a real claim - the dangerous direction."""
        assert out_of_scope(claim) is None

    def test_extract_claims_records_the_reason(self):
        doc = (
            "The study reports a 3D RMSE of 7.9 cm at a standoff of 70 m.\n"
            "The purpose of this round was to establish what can honestly be sold.\n"
        )
        claims = extract_claims(doc)
        assert len(claims) == 2
        assert claims[0].out_of_scope_reason is None
        assert claims[1].out_of_scope_reason == "self-reference"


class TestCascadeSkippedForOutOfScope:
    """The efficiency win: an out-of-scope claim must never enter the cascade."""

    @staticmethod
    def _wide_band(monkeypatch):
        """Force every in-scope claim into the escalation band, so a claim that does
        NOT reach the cascade proves the skip rather than a lucky lexical verdict."""
        real = joint.load_semantic_block

        def _block(path=None):
            b = dict(real(path) or {})
            b["escalation_band"] = [0.0, 1.0]
            return b

        monkeypatch.setattr(joint, "load_semantic_block", _block)

    def test_out_of_scope_claim_does_not_reach_the_cascade(self, monkeypatch):
        self._wide_band(monkeypatch)
        scored = []

        class SpyCascade:
            def score(self, claim, chunks):
                scored.append(claim)
                raise AssertionError("cascade must not be reached for an out-of-scope claim")

        sources = {"s.md": "The purpose of the round was to price the work honestly."}
        claim = "The purpose of this round was to establish what can honestly be sold."
        m = joint.ground_semantic(
            claim,
            sources,
            cfg=load_document_processing_config(),
            cascade=SpyCascade(),
            joint_verdict=joint.JointVerdict.from_config(joint.load_semantic_block()),
        )
        assert scored == []
        assert m.out_of_scope_reason == "self-reference"

    def test_in_scope_claim_still_escalates(self, monkeypatch):
        """Fail-on-revert guard: the skip must not swallow ordinary claims."""
        self._wide_band(monkeypatch)
        scored = []

        class SpyCascade:
            def score(self, claim, chunks):
                scored.append(claim)
                raise RuntimeError("stop here - reaching the cascade is the assertion")

        sources = {"s.md": "A wholly unrelated passage about kitchen appliances."}
        claim = "The study reports a 3D RMSE of 7.9 cm at a standoff of 70 m."
        with pytest.raises(RuntimeError):
            joint.ground_semantic(
                claim,
                sources,
                cfg=load_document_processing_config(),
                cascade=SpyCascade(),
                joint_verdict=joint.JointVerdict.from_config(joint.load_semantic_block()),
            )
        assert scored == [claim]

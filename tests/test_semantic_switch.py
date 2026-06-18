"""Semantic switch - the OpenVINO cascade composed with the effort tier by escalation.

`semantic` is an orthogonal on/off switch, not an effort tier: with it on, the selected
lexical_effort tier still decides whenever its win is clear, and only the uncertain band
escalates to the cascade. These tests pin the routing (clear -> lexical only; in-band ->
cascade + joint head), the frozen-weight joint verdict, and the CLI hard-fail - all with
a mocked cascade, so no model IRs or transformers are needed.

Degenerate escalation bands make the routing deterministic regardless of the live lexical
probability: ``[0.5, 0.5]`` -> every claim is "clear" (lex_p <= 0.5 OR >= 0.5); ``[0.0,
1.0]`` -> every claim is in-band.
"""

from __future__ import annotations

from groundrails import joint
from groundrails.config import load_document_processing_config
from groundrails.grounding import ground
from groundrails.joint import JOINT_FEATURES, JointVerdict, ground_semantic

SRC = (
    "doc.txt",
    "The Eiffel Tower is 330 metres tall and stands in Paris. "
    "It was completed in 1889 for the World Fair.",
)

BLOCK = {
    "feature_order": JOINT_FEATURES,
    "weights": {
        "Intercept": -2.0,
        "lex_p": 4.0,
        "rr_max": 3.0,
        "nli_ent": 3.0,
        "cos_max": 1.0,
        "nli_contra": -3.0,
        "lex_contra": -2.0,
        "lex_blocked": -1.0,
    },
    "threshold": 0.5,
    "escalation_band": [0.3, 0.9],
    "cosine_gate": [0.493, 0.739],
    "cascade_band": [0.01, 0.66],
    "top_k": 8,
}


class FakeCascade:
    """Records calls and returns controlled max-over-chunks signals (no IR load)."""

    def __init__(self, rr=0.9, ent=0.9, cos=0.8, contra=0.0):
        self.calls = 0
        self._r = (rr, ent, cos, contra)

    def score(self, claim, chunks):
        from groundrails.semantic_ov import CascadeScores

        self.calls += 1
        rr, ent, cos, contra = self._r
        return CascadeScores(
            cos_max=cos, rr_max=rr, nli_ent=ent, nli_contra=contra, ran_rr=True, ran_nli=True
        )


def _cfg():
    return load_document_processing_config()  # effort high


def test_jointverdict_math():
    jv = JointVerdict(weights={"Intercept": -1.0, "lex_p": 4.0}, feature_order=["lex_p"])
    assert jv.predict_proba({"lex_p": 1.0}) > 0.9
    assert jv.predict_proba({"lex_p": 0.0}) < 0.3
    assert jv.confirmed({"lex_p": 1.0})
    assert not jv.confirmed({"lex_p": 0.0})


def test_from_config_roundtrip():
    jv = JointVerdict.from_config(BLOCK)
    assert jv.feature_order == JOINT_FEATURES
    assert jv.threshold == 0.5
    assert JointVerdict.from_config({"weights": {}}) is None
    assert JointVerdict.from_config(None) is None


def test_clear_win_skips_cascade(monkeypatch):
    # band [0.5, 0.5] -> every claim is clear -> the cascade must never run.
    monkeypatch.setattr(
        joint, "load_semantic_block", lambda path=None: dict(BLOCK, escalation_band=[0.5, 0.5])
    )
    fake = FakeCascade()
    m = ground_semantic("The Eiffel Tower stands in Paris", [SRC], cfg=_cfg(), cascade=fake)
    assert fake.calls == 0
    assert m.reranker_score == 0.0


def test_in_band_escalates_and_joint_confirms(monkeypatch):
    # band [0.0, 1.0] -> every claim escalates; strong cascade signals -> joint confirms.
    monkeypatch.setattr(
        joint, "load_semantic_block", lambda path=None: dict(BLOCK, escalation_band=[0.0, 1.0])
    )
    fake = FakeCascade(rr=0.95, ent=0.95, cos=0.85)
    m = ground_semantic("The tower stands in Paris", [SRC], cfg=_cfg(), cascade=fake)
    assert fake.calls == 1
    assert m.reranker_score == 0.95
    assert m.match_type != "none"


def test_in_band_joint_rejects_on_weak_signals(monkeypatch):
    monkeypatch.setattr(
        joint, "load_semantic_block", lambda path=None: dict(BLOCK, escalation_band=[0.0, 1.0])
    )
    fake = FakeCascade(rr=0.02, ent=0.02, cos=0.1)
    m = ground_semantic(
        "Penguins manufacture the tower from cheese", [SRC], cfg=_cfg(), cascade=fake
    )
    assert fake.calls == 1
    assert m.match_type == "none"


def test_switch_off_never_constructs_cascade(monkeypatch):
    # default config mode is lexical -> ground() must not construct the cascade engine.
    constructed = {"n": 0}

    class Boom:
        def __init__(self, *a, **k):
            constructed["n"] += 1

    monkeypatch.setattr("groundrails.semantic_ov.SemanticCascade", Boom)
    m = ground("The Eiffel Tower stands in Paris", [SRC])
    assert constructed["n"] == 0
    assert m.match_type in ("exact", "fuzzy", "bm25", "none")


def test_ground_dispatches_when_semantic_true(monkeypatch):
    monkeypatch.setattr(
        joint, "load_semantic_block", lambda path=None: dict(BLOCK, escalation_band=[0.0, 1.0])
    )
    fake = FakeCascade()
    monkeypatch.setattr("groundrails.semantic_ov.SemanticCascade", lambda **k: fake)
    ground("The tower stands in Paris", [SRC], semantic=True)
    assert fake.calls == 1


def test_cli_semantic_hardfails_without_deps(monkeypatch, capsys, tmp_path):
    from groundrails import cli, semantic_ov

    monkeypatch.setattr(semantic_ov, "is_available", lambda: False)
    src = tmp_path / "s.txt"
    src.write_text("hello world", encoding="utf-8")
    rc = cli.main(["ground", "--claim", "x", "--source", str(src), "--semantic"])
    assert rc == 2
    assert "cascade extras" in capsys.readouterr().err

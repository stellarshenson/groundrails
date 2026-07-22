"""Semantic switch - compose the OpenVINO cascade with the lexical tier by escalation.

`semantic` is an orthogonal on/off switch (config ``calibration.mode: semantic``, or
``--semantic`` on the CLI), NOT an effort tier. It crosses every effort tier: with the
switch on, the selected ``lexical_effort`` (low / medium / high) still runs first and
decides whenever its win is clear; only the uncertain band - and the cross-lingual
claims the lexical tier cannot ground - escalate to the heavy cross-encoder cascade.
The two are then fused by a frozen-weight joint logistic.

The switch's parameters live under ``calibration.semantic`` (escalation band, cosine
gate, cascade band, top-k, and the joint-head weights/threshold) - the same
config-as-source-of-truth pattern as ``calibration.lexical_manifolds.<tier>``. The joint
head is a dot-product + sigmoid, no scikit-learn at inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

JOINT_FEATURES = [
    "lex_p",
    "rr_max",
    "nli_ent",
    "cos_max",
    "nli_contra",
    "lex_contra",
    "lex_blocked",
]


@dataclass
class JointVerdict:
    """Frozen-weight logistic over the joint lexical+semantic feature set.

    sigmoid(Intercept + Σ w·feature); confirmed when P(supported) >= threshold.
    Weights / order / threshold live in ``calibration.semantic`` and transfer verbatim.
    """

    weights: dict[str, float]
    feature_order: list[str] = field(default_factory=lambda: list(JOINT_FEATURES))
    threshold: float = 0.5

    def predict_proba(self, feat: dict[str, float]) -> float:
        z = float(self.weights.get("Intercept", 0.0))
        for name in self.feature_order:
            z += float(self.weights.get(name, 0.0)) * float(feat.get(name, 0.0))
        return 1.0 / (1.0 + math.exp(-z))

    def confirmed(self, feat: dict[str, float]) -> bool:
        return self.predict_proba(feat) >= self.threshold

    @classmethod
    def from_config(cls, block: dict) -> "JointVerdict | None":
        """Build from the ``calibration.semantic`` block; None when weights absent."""
        if not block or not block.get("weights"):
            return None
        return cls(
            weights={k: float(v) for k, v in block["weights"].items()},
            feature_order=list(block.get("feature_order") or JOINT_FEATURES),
            threshold=float(block.get("threshold", 0.5)),
        )


def load_semantic_block(path=None) -> dict | None:
    """Return the ``calibration.semantic`` switch parameters, or None."""
    from groundrails.calibration import load_calibration_from_config

    return (load_calibration_from_config(path) or {}).get("semantic")


def switch_on(path=None) -> bool:
    """True when the config switch is set (``calibration.mode: semantic``)."""
    from groundrails.calibration import load_calibration_from_config

    return (load_calibration_from_config(path) or {}).get("mode") == "semantic"


def _chunk_texts(sources, cfg) -> list[str]:
    """Evidence chunks for the cascade - recursive-chunk every source, flatten."""
    from groundrails.chunking import recursive_chunk

    out: list[str] = []
    for src in sources:
        text = src[1] if isinstance(src, tuple) else src
        chunks = recursive_chunk(text, max_chars=cfg.chunk_max_chars)
        out.extend(c.text for c in chunks) if chunks else out.append(text)
    return out or [s[1] if isinstance(s, tuple) else s for s in sources]


def ground_semantic(
    claim,
    sources,
    *,
    cfg,
    cascade=None,
    joint_verdict=None,
    primary_source=None,
    ignore_language=False,
):
    """Lexical tier first; escalate the uncertain band to the OV cascade; fuse.

    The lexical verdict at ``cfg.lexical_effort`` decides directly when its P(grounded)
    is outside the escalation band (or when the claim grounds/contradicts cleanly). The
    in-band claims and the cross-lingual claims the lexical tier cannot ground are scored
    by the cascade and resolved by the joint head.
    """
    from groundrails.grounding import (
        GroundingMatch,
        UnsupportedLanguageError,
        _populate_match_metadata,
        _tokenize,
        _winning_layer_label,
        ground,
    )

    block = load_semantic_block()
    jv = joint_verdict or (JointVerdict.from_config(block) if block else None)
    # switch on but unconfigured -> just run the lexical tier (no cascade)
    if block is None or jv is None:
        return ground(
            claim,
            sources,
            config=cfg,
            primary_source=primary_source,
            semantic=False,
            ignore_language=ignore_language,
        )

    a, b = float(block["escalation_band"][0]), float(block["escalation_band"][1])

    # 1. lexical tier (semantic=False so we do not re-dispatch into this function)
    blocked = False
    try:
        m = ground(
            claim,
            sources,
            config=cfg,
            primary_source=primary_source,
            semantic=False,
            ignore_language=ignore_language,
        )
        lex_p = m.verdict_probability if m.verdict_probability >= 0 else m.agreement_score
    except UnsupportedLanguageError:
        m = GroundingMatch(claim=claim)
        # DEF-4 floor input: this match never passes through ground(), which
        # otherwise sets the count unconditionally - without it every blocked
        # cross-lingual claim would read as sub-floor (count 0) and the joint
        # head's confirmation would be withheld regardless of claim length.
        m.bm25_claim_token_count = len(set(_tokenize(claim)))
        lex_p = 0.0
        blocked = True
    lex_contra = bool(m.numeric_mismatches or m.entity_mismatches)

    # 2. clear win -> the lexical verdict stands: outside the uncertainty
    # band, or already resolved exact/contradicted (verdicts the joint head
    # is not allowed to change - running the cascade for them is dead work)
    if not blocked and (m.match_type in ("exact", "contradicted") or lex_p <= a or lex_p >= b):
        return m

    # 3. escalate: score the cascade and fuse with the joint head
    if cascade is None:
        from groundrails.semantic_ov import SemanticCascade

        cascade = SemanticCascade(
            top_k=int(block.get("top_k", 8)),
            gate=tuple(block.get("cosine_gate", (0.493, 0.739))),
            band=tuple(block.get("cascade_band", (0.01, 0.66))),
        )
    sem = cascade.score(claim, _chunk_texts(sources, cfg))
    m.reranker_score = sem.rr_max
    # NOTE: semantic_score now carries the cascade's bge-m3 cosine alongside
    # the legacy embedding space's scores - two distributions in one field.
    # cfg.semantic_threshold (tuned on the legacy space) gates both; the
    # cascade's own cosine_gate is the score's native operating range.
    m.semantic_score = max(m.semantic_score, sem.cos_max)
    # full 3-key softmax shape, same contract as nli.NLIGrounder.scores -
    # but ONLY when the cascade's NLI stage actually ran: fabricating
    # {neutral: 1.0} from the zeroed defaults is indistinguishable from
    # "the model read it and said neutral" downstream
    if sem.ran_nli:
        m.nli_scores = {
            "entailment": sem.nli_ent,
            "contradiction": sem.nli_contra,
            "neutral": max(0.0, 1.0 - sem.nli_ent - sem.nli_contra),
        }

    feat = {
        "lex_p": lex_p,
        "rr_max": sem.rr_max,
        "nli_ent": sem.nli_ent,
        "cos_max": sem.cos_max,
        "nli_contra": sem.nli_contra,
        "lex_contra": 1.0 if lex_contra else 0.0,
        "lex_blocked": 1.0 if blocked else 0.0,
    }
    p = jv.predict_proba(feat)
    m.verdict_probability = p
    # Deliberate divergence from _verdict_preamble's ``and has_any_signal``
    # guard: reaching this point means the claim was escalated (uncertain or
    # blocked), and the numeric/entity mismatch that set lex_contra is itself
    # the signal - a full-source fallback mismatch counts here.
    if lex_contra and not blocked:
        m.match_type = "contradicted"
    elif m.exact_score >= 1.0:
        # Policy parity with grounding._verdict_preamble: a verbatim exact
        # match is not an uncertain case - the joint head must not un-ground
        # it. UNREACHABLE on every current path (a non-blocked exact resolves
        # in the inner ground() and early-returns above; the blocked path
        # builds a fresh match with exact_score 0), kept only as defence in
        # depth. If a future change makes it live, thread exact_window_text
        # into the _populate_match_metadata call below so reason (d') is not
        # silently skipped for the escalated exact confirm.
        m.match_type = "exact"
    elif p >= jv.threshold:
        # Same labelling + DEF-4 floor as every other engine. No lexical-
        # evidence special case: a sub-floor claim must clear fuzzy/semantic
        # thresholds whether or not some lexical layer carries a nonzero
        # score - otherwise MORE lexical evidence would face a STRICTER bar.
        # cfg thresholds ARE this tier's operating point (the joint tier has
        # no percentile machinery; see also the two-score-space note above).
        m.match_type = _winning_layer_label(m, cfg)
    else:
        m.match_type = "none"
    # The cascade + head may have flipped match_type relative to the inner
    # ground() call - re-run WI#3/#5/#6 metadata (grounded_source,
    # lexical_co_support, verification_needed) against the final verdict,
    # exactly as the adaptive_gap reclassifier does after its flips.
    _populate_match_metadata(
        m,
        cfg=cfg,
        primary_source=primary_source,
        fuzzy_threshold=cfg.fuzzy_threshold,
        bm25_threshold=cfg.bm25_threshold,
        effective_semantic_threshold=cfg.semantic_threshold,
    )
    # Borderline flag, mirroring the lexical and calibrated engines: a fused
    # verdict within proximity of the joint threshold is cheap to second-guess.
    if m.match_type in ("exact", "fuzzy", "bm25", "semantic"):
        if abs(p - jv.threshold) < cfg.verification_threshold_proximity:
            m.verification_needed = True
    return m

"""Grounding: score a claim against one or more source texts.

Runs THREE independent matching layers and reports ALL scores:

    1. **Exact (regex)** — whitespace-tolerant, case-insensitive regex search.
       Score = 1.0 when a hit is found on any source, 0.0 otherwise.
    2. **Fuzzy (Levenshtein)** — ``rapidfuzz.fuzz.partial_ratio_alignment``
       finds the best-matching substring across all sources. Score in [0,1].
    3. **BM25 (lexical / topical)** — ``rank_bm25.BM25Okapi`` ranks source
       passages (paragraphs or sentences) for term overlap with IDF weighting.
       Score normalised to [0,1] via max-in-corpus. Handles paraphrase with
       same key terms but different word order.

Agent-friendly primary grounding approach: BM25 finds the *right passage*
even when wording differs. Use exact for quoted claims, fuzzy for paraphrases
with close wording, BM25 for topical grounding ("the claim is discussed in
this passage" even when wording differs). A disciplined generative
interpretation is the secondary fallback for semantic claims none of the
three lexical layers capture.

The returned :class:`GroundingMatch` carries all three scores plus a
:class:`Location` for each layer (line, column, paragraph, page, context) so
a grounding agent can cite the hit without rereading the source.

Location semantics:
    - line_start / line_end: 1-indexed, inclusive
    - column_start / column_end: 1-indexed character offset on line
    - paragraph: 1-indexed, paragraphs separated by blank lines
    - page: 1-indexed, pages separated by form-feed ``\\f`` (pdftotext convention)
    - context_before / context_after: up to one adjacent line, trimmed to
      ``context_chars``
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import re
import threading
from typing import Literal, Sequence

import numpy as np
from rank_bm25 import BM25Okapi
from rapidfuzz.fuzz import partial_ratio_alignment

from groundrails.config import (
    GroundingConfig,
)
from groundrails.config import (
    load_document_processing_config as load_config,
)
from groundrails.entity_check import (
    extract_entities,
    extract_numbers,
    find_absent_entities,
    find_mismatches,
    find_numeric_mismatches,
    list_claim_entities,
)

logger = logging.getLogger(__name__)

MatchType = Literal["exact", "fuzzy", "bm25", "semantic", "nli", "contradicted", "none"]

SourceInput = str | tuple[str, str]
"""A source is either raw text or a ``(path, text)`` pair for provenance."""


@dataclass
class SemanticHitSummary:
    """Compact summary of a semantic hit for ``semantic_top_k`` reporting.

    Avoids coupling :class:`GroundingMatch` to the optional
    :class:`semantic.SemanticHit` class (which is only importable with the
    ``[semantic]`` extra). All float scores in [0, 1] for L2-normalised
    embeddings.
    """

    score: float = 0.0
    matched_text: str = ""
    source_index: int = -1
    source_path: str = ""
    char_start: int = -1
    char_end: int = -1


@dataclass
class Location:
    """Location of a match inside source text.

    All line/column/paragraph/page indices 1-based; ``char_start``/``char_end``
    are 0-based character offsets matching Python slicing. Default ``-1``
    indicates "unknown / no match".
    """

    source_index: int = -1
    source_path: str = ""
    char_start: int = -1  # 0-indexed inclusive
    char_end: int = -1  # 0-indexed exclusive
    line_start: int = -1  # 1-indexed inclusive
    line_end: int = -1  # 1-indexed inclusive
    column_start: int = -1  # 1-indexed (column on line_start)
    column_end: int = -1  # 1-indexed (column on line_end)
    paragraph: int = -1  # 1-indexed; paragraphs separated by blank lines
    page: int = -1  # 1-indexed; pages separated by \f form-feed
    context_before: str = ""
    context_after: str = ""


@dataclass
class GroundingMatch:
    """Result of grounding a single claim against a set of sources.

    Three layers always run independently. ``match_type`` is a convenience
    classifier based on thresholds; ``combined_score`` is the max of the
    three normalised scores.

    ``exact_location``, ``fuzzy_location``, and ``bm25_location`` give the
    grounding agent enough metadata (line, paragraph, page, context) to cite
    the hit without rereading the source.
    """

    claim: str
    # Regex / exact layer
    exact_score: float = 0.0  # 1.0 on hit, 0.0 otherwise
    exact_matched_text: str = ""
    exact_location: Location = field(default_factory=Location)
    # Levenshtein / fuzzy layer
    fuzzy_score: float = 0.0  # Levenshtein partial ratio in [0, 1]
    fuzzy_matched_text: str = ""
    fuzzy_location: Location = field(default_factory=Location)
    # BM25 lexical / topical layer
    bm25_score: float = 0.0  # Normalised [0, 1] - best passage vs max in corpus
    bm25_raw_score: float = 0.0  # Raw BM25 Okapi score (unbounded, clamped >= 0)
    bm25_token_recall: float = 0.0  # Fraction of unique claim tokens in best passage
    bm25_claim_token_count: int = 0  # Unique claim tokens (recall denominator size)
    bm25_matched_text: str = ""
    bm25_location: Location = field(default_factory=Location)
    # Semantic (ModernBERT + FAISS) layer — optional, off unless enabled
    semantic_score: float = 0.0  # cosine similarity in [0, 1] with L2-normalised embeddings
    semantic_matched_text: str = ""
    semantic_location: Location = field(default_factory=Location)
    semantic_top_k: list[SemanticHitSummary] = field(default_factory=list)
    """Top-K semantic hits (up to 3 by default) for alternative pointers."""
    semantic_ratio: float = 0.0
    """semantic_score / claim_self_score; calibration anchor per claim. 0 when semantic layer off."""
    # Agreement across layers
    agreement_score: float = 0.0
    """Weighted combination of exact / fuzzy / bm25 / semantic layers with multi-voter bonus."""
    # Contradiction detection (numeric + named-entity mismatch between claim and winning passage)
    numeric_mismatches: list[tuple[str, str]] = field(default_factory=list)
    """List of ``(claim_value, passage_value)`` for numeric disagreements."""
    entity_mismatches: list[tuple[str, str]] = field(default_factory=list)
    """List of ``(claim_entity, passage_entity)`` for tech-entity disagreements."""
    entities_absent: list[str] = field(default_factory=list)
    """Proper-noun entities mentioned in the claim with zero occurrences in ANY source passage.

    Weaker than ``entity_mismatches`` (which requires the source to mention
    the same category with a different value) but catches fabricated-entity
    claims like "RoPE-Mid" or "NVIDIA H100 donated by Meta" when the source
    is a Liu-style paper that never names these. Non-empty ``entities_absent``
    downweights ``agreement_score`` via a graded penalty (does not force
    CONTRADICTED - absence is weaker evidence than disagreement).
    """
    # Borderline expansion flag (H5 - reserved, not set in iter 1)
    expanded: bool = False
    """True when chunk-boundary expansion fired on a borderline semantic hit."""
    # Cross-source provenance (WI#3)
    grounded_source: str | None = None
    """Source path where the winning-layer hit was found, or None if no match."""
    is_primary_source: bool = True
    """False when grounded_source differs from a caller-supplied primary source."""
    # Lexical co-support gate (WI#5) / verification signal (WI#6)
    lexical_co_support: bool = False
    """True when semantic fires AND at least one lexical layer clears its cosupport floor."""
    verification_needed: bool = False
    """True when the match carries one or more second-guess signals (see ground())."""
    claim_attributes: dict = field(default_factory=dict)
    """Side-by-side attribute summary: numbers/entities in claim vs winning passage."""
    # Resolution
    match_type: MatchType = "none"
    combined_score: float = 0.0  # max of all enabled layers
    # Calibrated verdict engine (opt-in via config calibration.engine=calibrated).
    # verdict_probability stays -1.0 when the deterministic classifier was used.
    verdict_probability: float = -1.0
    """Calibrated P(grounded) in [0,1] when the calibrated engine ran; -1.0 otherwise."""
    verdict_uncertainty: float = 0.0
    """Posterior-predictive logit spread for the calibrated verdict (0 for a point-weight config)."""
    verdict_features: dict = field(default_factory=dict)
    """The feature vector fed to the calibrator (audit trail)."""
    nli_scores: dict = field(default_factory=dict)
    """Cross-encoder NLI softmax {entailment, neutral, contradiction} when the NLI layer ran."""
    reranker_score: float = 0.0
    """Max bge-reranker relevance over chunks when the semantic cascade ran (0 otherwise)."""

    @property
    def grounded(self) -> bool:
        """True when a layer supports the claim (``none`` = unsupported, ``contradicted`` = conflict)."""
        return self.match_type not in ("none", "contradicted")

    @property
    def support(self) -> dict | None:
        """Winning-layer support provenance - the quote and its location in the evidence -
        or ``None`` when the claim is not grounded. A cascade verdict (``match_type`` =
        ``semantic``) carries no native location, so it falls back to the best lexical
        passage, flagged ``support_via = "lexical"``; an ``nli`` verdict likewise has no
        span of its own and falls back to the premise's origin layer, flagged
        ``support_via = "nli-premise"``. Rarely (grounded ``nli`` on a near-empty
        source where NO layer carries a located span) the fallback finds nothing and
        a grounded match returns ``None`` - callers must handle it."""
        if not self.grounded:
            return None
        layers = {
            "exact": (self.exact_location, self.exact_matched_text),
            "fuzzy": (self.fuzzy_location, self.fuzzy_matched_text),
            "bm25": (self.bm25_location, self.bm25_matched_text),
            "semantic": (self.semantic_location, self.semantic_matched_text),
        }
        via = None
        loc, text = layers.get(self.match_type, (None, ""))
        if self.match_type == "semantic" and self.semantic_score <= 0:
            # a zero-clamped located semantic hit is not evidence - force the
            # fallback picker (guards on score > 0) rather than cite it directly
            loc = None
        if loc is None or loc.char_start < 0:
            # ONE fallback picker shared with _winning_location / the WI#6
            # passage picker / the CLI - see _origin_fallback
            hit = _origin_fallback(self)
            if hit is not None:
                loc, text = hit
                via = "nli-premise" if self.match_type == "nli" else "lexical"
        if loc is None or loc.char_start < 0:
            return None
        out = {
            "source_index": loc.source_index,
            "source_path": loc.source_path,
            "matched_text": text,
            "char_start": loc.char_start,
            "char_end": loc.char_end,
            "line_start": loc.line_start,
            "line_end": loc.line_end,
            "paragraph": loc.paragraph,
            "page": loc.page,
        }
        if via:
            out["support_via"] = via
        return out


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space, strip ends."""
    return re.sub(r"\s+", " ", text).strip()


def _exact_match(claim: str, source: str) -> tuple[int, int] | None:
    """Find the claim inside ``source`` ignoring whitespace + case differences.

    Word-bounded: the match may not start or end in the middle of a source
    word (``"um to a"`` must not hit inside ``"sums to a"``). It is still a
    substring match across whole words - a claim that is a verbatim
    sub-phrase of a longer (even negating) sentence matches; see
    docs/defects.md known limitations."""
    norm_claim = _normalize_whitespace(claim)
    if not norm_claim:
        return None
    tokens = norm_claim.split(" ")
    # Boundary lookarounds only where the claim edge is itself a word char:
    # a claim ending in punctuation ("...40%.") must still match a source
    # where the next sentence follows without a space ("...40%.The next").
    prefix = r"(?<!\w)" if re.match(r"\w", tokens[0]) else ""
    suffix = r"(?!\w)" if re.search(r"\w$", tokens[-1]) else ""
    pattern = prefix + r"\s+".join(re.escape(t) for t in tokens) + suffix
    m = re.search(pattern, source, flags=re.IGNORECASE)
    if m:
        return (m.start(), m.end())
    return None


def _unpack_sources(sources: Sequence[SourceInput]) -> list[tuple[int, str, str]]:
    """Normalise heterogeneous source inputs to ``(index, path, text)`` tuples."""
    # A bare str is itself a Sequence[str], so it type-checks yet iterates into
    # per-CHARACTER sources - rapidfuzz then treats a 1-char source as a needle
    # and scores partial_ratio 1.0, silently CONFIRMING a fabricated claim
    # against unrelated evidence. Fail loud instead of grounding garbage.
    if isinstance(sources, str):
        raise TypeError(
            "sources must be a sequence of source texts or (path, text) tuples, "
            "not a single str - wrap it as [text] or [(path, text)]"
        )
    out: list[tuple[int, str, str]] = []
    for i, src in enumerate(sources):
        if isinstance(src, tuple):
            path, text = src
            out.append((i, path, text))
        else:
            out.append((i, "", src))
    return out


def _locate(
    text: str,
    start: int,
    end: int,
    *,
    source_index: int,
    source_path: str,
    context_chars: int = 80,
) -> Location:
    """Compute line/column/paragraph/page metadata for a char span."""
    line_start = text.count("\n", 0, start) + 1
    line_end = text.count("\n", 0, end) + 1

    prev_nl_start = text.rfind("\n", 0, start)
    col_start = start - prev_nl_start if prev_nl_start >= 0 else start + 1
    prev_nl_end = text.rfind("\n", 0, end)
    col_end = end - prev_nl_end if prev_nl_end >= 0 else end + 1

    paragraph = 1 + len(re.findall(r"\n\s*\n", text[:start]))
    page = 1 + text.count("\f", 0, start)

    line_begin = text.rfind("\n", 0, start) + 1
    prev_line_begin = text.rfind("\n", 0, line_begin - 1) + 1 if line_begin > 0 else 0
    context_before = text[prev_line_begin:start].replace("\n", " ").strip()
    if len(context_before) > context_chars:
        context_before = "…" + context_before[-context_chars:]

    line_finish = text.find("\n", end)
    if line_finish < 0:
        line_finish = len(text)
    next_line_finish = text.find("\n", line_finish + 1)
    if next_line_finish < 0:
        next_line_finish = len(text)
    context_after = text[end:next_line_finish].replace("\n", " ").strip()
    if len(context_after) > context_chars:
        context_after = context_after[:context_chars] + "…"

    return Location(
        source_index=source_index,
        source_path=source_path,
        char_start=start,
        char_end=end,
        line_start=line_start,
        line_end=line_end,
        column_start=col_start,
        column_end=col_end,
        paragraph=paragraph,
        page=page,
        context_before=context_before,
        context_after=context_after,
    )


# -- BM25 passage ranking -----------------------------------------------


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Half-width of the evidence window taken around an exact span - both the NLI
# premise widening and the WI#6 reason (d') restated-conflict scan read the
# same "surrounding context of the quoted span" concept, so they share one
# radius rather than two literals that could drift apart.
_EVIDENCE_WINDOW_CHARS = 300


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenisation."""
    return _TOKEN_RE.findall(text.lower())


def _snap_to_word_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Expand ``[start, end)`` outward so it never cuts a word in half.

    Also trims leading/trailing whitespace inside the span. Bounds are
    clamped to the text; a degenerate span is returned unchanged."""
    n = len(text)
    start = max(0, min(start, n))
    end = max(start, min(end, n))
    if start == end:
        return start, end
    while start > 0 and text[start - 1].isalnum() and start < n and text[start].isalnum():
        start -= 1
    while end < n and text[end].isalnum() and text[end - 1].isalnum():
        end += 1
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


# NOTE: deliberately NOT abbreviation-guarded like extract._split_sentences -
# this regex only splits BM25 passages on the single-mega-passage fallback,
# where a shattered citation fragment merges back into its neighbour via the
# min-passage-chars pass below; claim extraction is where the guard matters.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_MIN_PASSAGE_CHARS = 40  # fallback when no config is threaded through
_SINGLE_PASSAGE_FALLBACK_LEN = 1500


def _split_passages(
    text: str,
    min_passage_chars: int = _MIN_PASSAGE_CHARS,
    single_passage_fallback_length: int = _SINGLE_PASSAGE_FALLBACK_LEN,
) -> list[tuple[int, int, str]]:
    """Split text into passages by blank-line boundaries.

    Returns ``[(start, end, passage_text), ...]`` with char offsets into the
    original text. Passages with no word characters are dropped.

    Falls back to sentence splitting when blank-line splitting produces a
    single mega-passage on long texts (common with ``pdftotext`` output that
    uses single-newline line breaks). Prevents BM25 from degenerating into a
    1-doc corpus where IDF is meaningless. Both operating points come from
    ``GroundingConfig`` (``min_passage_chars``,
    ``single_passage_fallback_length``) when called via the grounding path.
    """
    if not text:
        return []
    passages: list[tuple[int, int, str]] = []
    # Match runs of non-blank lines as one passage
    for m in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text):
        p = m.group()
        if _TOKEN_RE.search(p):
            passages.append((m.start(), m.end(), p))

    # Fallback: on long single-passage texts, split on sentence boundaries so
    # BM25 has a corpus to rank against.
    if len(passages) == 1 and len(text) > single_passage_fallback_length:
        start_0, _, body = passages[0]
        sentence_passages: list[tuple[int, int, str]] = []
        cursor = 0
        for m in _SENT_SPLIT_RE.finditer(body):
            sent_end = m.start()
            sentence = body[cursor:sent_end].strip()
            if sentence:
                sentence_passages.append((start_0 + cursor, start_0 + sent_end, sentence))
            cursor = m.end()
        tail = body[cursor:].strip()
        if tail:
            sentence_passages.append((start_0 + cursor, start_0 + len(body), tail))

        # Merge too-short sentences with their successor so BM25 IDF stays useful
        merged: list[tuple[int, int, str]] = []
        pending: tuple[int, int, str] | None = None
        for s, e, p in sentence_passages:
            if pending is None:
                pending = (s, e, p)
                continue
            if len(pending[2]) < min_passage_chars:
                pending = (pending[0], e, pending[2] + " " + p)
            else:
                merged.append(pending)
                pending = (s, e, p)
        if pending is not None:
            merged.append(pending)

        if len(merged) >= 2:
            return merged

    return passages


@dataclass
class _BM25Hit:
    source_index: int
    source_path: str
    char_start: int
    char_end: int
    matched_text: str
    raw_score: float
    normalised_score: float
    token_recall: float
    claim_token_count: int = 0


# BM25 corpus cache: passage splitting + tokenisation + the BM25Okapi index
# depend only on the sources, yet were rebuilt per claim - the dominant
# lexical cost in ``ground_batch`` (O(claims x corpus)). The index is
# read-only after construction (``get_scores`` allocates locally), so one
# instance is safely shared across worker threads. Keyed by content hash;
# tiny bound because callers ground many claims against few source sets.
_BM25_CACHE: dict[
    tuple, tuple[list[list[str]], list[tuple[int, str, int, int, str]], BM25Okapi]
] = {}
_BM25_CACHE_LOCK = threading.Lock()
_BM25_CACHE_MAX = 8


_BM25_EMPTY = object()  # cached marker for an empty corpus (still a cache hit)


def _bm25_corpus(
    pairs: list[tuple[int, str, str]],
    min_passage_chars: int = _MIN_PASSAGE_CHARS,
    single_passage_fallback_length: int = _SINGLE_PASSAGE_FALLBACK_LEN,
) -> tuple[list[list[str]], list[tuple[int, str, int, int, str]], BM25Okapi] | None:
    """Build (or fetch cached) tokenised corpus + BM25 index for ``pairs``.

    The split operating points are part of the cache key: a cached corpus is
    only valid for the parameters it was split with, so a config override
    can never be served a cross-config corpus."""
    key = (
        (min_passage_chars, single_passage_fallback_length),
        *((idx, path, len(text), hash(text)) for idx, path, text in pairs),
    )
    with _BM25_CACHE_LOCK:
        hit = _BM25_CACHE.get(key)
        if hit is not None:
            # LRU touch: reinsertion moves the entry to the dict's end so
            # round-robin over >MAX source sets does not evict the entry
            # just before its reuse (FIFO thrash).
            _BM25_CACHE.pop(key, None)
            _BM25_CACHE[key] = hit
    if hit is not None:
        return None if hit is _BM25_EMPTY else hit

    corpus_tokens: list[list[str]] = []
    provenance: list[tuple[int, str, int, int, str]] = []  # (src_idx, path, start, end, text)
    for idx, path, text in pairs:
        for start, end, passage in _split_passages(
            text, min_passage_chars, single_passage_fallback_length
        ):
            tokens = _tokenize(passage)
            if tokens:
                corpus_tokens.append(tokens)
                provenance.append((idx, path, start, end, passage))

    entry = (corpus_tokens, provenance, BM25Okapi(corpus_tokens)) if corpus_tokens else _BM25_EMPTY
    with _BM25_CACHE_LOCK:
        if len(_BM25_CACHE) >= _BM25_CACHE_MAX:
            _BM25_CACHE.pop(next(iter(_BM25_CACHE)))
        _BM25_CACHE[key] = entry
    return None if entry is _BM25_EMPTY else entry


def _bm25_match(
    claim: str,
    pairs: list[tuple[int, str, str]],
    cfg: GroundingConfig | None = None,
) -> _BM25Hit | None:
    """Rank passages across all sources with BM25 Okapi. Return best hit."""
    claim_tokens = _tokenize(claim)
    if not claim_tokens:
        return None

    corpus = (
        _bm25_corpus(pairs, cfg.min_passage_chars, cfg.single_passage_fallback_length)
        if cfg is not None
        else _bm25_corpus(pairs)
    )
    if corpus is None:
        return None
    corpus_tokens, provenance, bm25 = corpus

    scores = bm25.get_scores(claim_tokens)
    scores = np.maximum(scores, 0.0)  # BM25 can go negative on tiny corpora
    max_score = float(scores.max())

    claim_set = set(claim_tokens)
    idf = bm25.idf
    max_idf = max(idf.values()) if idf else 1.0

    def _w(tok: str) -> float:
        return max(0.0, idf.get(tok, max_idf))

    def _recall(passage_idx: int) -> float:
        """IDF-weighted fraction of unique claim tokens present in the passage.

        Weighting by IDF stops corpus-ubiquitous words (e.g. the domain topic
        word that appears in every passage) from inflating recall, and gives
        full weight to distinctive claim tokens that are ABSENT (a claim whose
        specific terms are missing should not count as grounded). Tokens not
        in the corpus are treated as maximally distinctive."""
        passage_set = set(corpus_tokens[passage_idx])
        den = sum(_w(t) for t in claim_set)
        if den > 0:
            return sum(_w(t) for t in claim_set if t in passage_set) / den
        # Degenerate IDF (all claim tokens ubiquitous) -> fall back to raw recall.
        return len(claim_set & passage_set) / len(claim_set) if claim_set else 0.0

    if max_score == 0.0:
        # Tiny corpora (2-3 passages) can push EVERY BM25 score negative -
        # common tokens get negative IDF. The layer is absent there. A
        # recall-argmax fallback was tried and REVERTED: bm25_score feeds the
        # calibrated verdict features, and resurrecting the layer on tiny
        # corpora shifts the pinned golden score distribution (equivalence
        # guard failed). Revisit only together with a recalibration.
        return None

    best_idx = int(scores.argmax())
    raw = float(scores[best_idx])
    normalised = raw / max_score  # relative to top passage = 1.0 for winner
    recall = _recall(best_idx)

    src_idx, path, start, end, text = provenance[best_idx]
    return _BM25Hit(
        source_index=src_idx,
        source_path=path,
        char_start=start,
        char_end=end,
        matched_text=text,
        raw_score=raw,
        normalised_score=normalised,
        token_recall=recall,
        claim_token_count=len(claim_set),
    )


# -- Main API -----------------------------------------------------------


def _compute_agreement_score(
    exact_score: float,
    fuzzy_score: float,
    bm25_token_recall: float,
    semantic_score: float,
    semantic_ratio: float = 0.0,
    cfg: GroundingConfig | None = None,
) -> float:
    """Weighted cross-layer agreement score with multi-voter bonus.

    Per H1 spec: ``any layer firing alone < two layers firing together``.
    Real paraphrases light up multiple layers weakly; fabrications typically
    fire only on semantic (topical similarity with no lexical overlap).

    Each layer has a per-layer "vote" threshold. Layers above threshold are
    counted as voters. The score is a weighted sum of per-layer
    contributions plus a multi-voter bonus that rewards independent signals.

    Semantic contribution (H7, Iter 4): when ``semantic_ratio`` is provided
    (the claim's cosine against itself-as-passage, already computed on the
    semantic path), the ramp uses it as a model-normalised measure instead
    of the raw cosine. ``ratio = cos(claim, match) / cos(claim, claim)`` is
    naturally bounded in [0, 1] and is roughly model-independent for real
    hits (~1.0) versus noise (~0.85 or below). Ramp centre 0.88 so real
    hits contribute strongly regardless of whether the underlying model is
    E5-small (absolute hits 0.85+) or mpnet (absolute hits 0.70+). When
    ``semantic_ratio`` is 0 (semantic layer disabled or no hit), falls back
    to the absolute-cosine ramp for backward compatibility.

    Layer vote thresholds (tuned low so real-but-weak signals still count):
    - exact:    >= 1.0 (exact is binary)
    - fuzzy:    >= 0.55 (partial-ratio of ~half the claim)
    - bm25:     >= 0.15 (some token overlap)
    - semantic: ratio >= 0.90 when available, else raw cosine >= 0.70

    Returns value in [0, 1].
    """
    c = cfg if cfg is not None else load_config()

    def _ramp(x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return max(0.0, min(1.0, (x - lo) / (hi - lo)))

    v_exact = 1.0 if exact_score >= c.voter_exact else 0.0
    v_fuzzy = _ramp(fuzzy_score, c.fuzzy_ramp_low, c.fuzzy_ramp_high)
    v_bm25 = _ramp(bm25_token_recall, c.bm25_ramp_low, c.bm25_ramp_high)
    v_sem_abs = _ramp(semantic_score, c.semantic_abs_ramp_low, c.semantic_abs_ramp_high)
    v_sem_ratio = (
        _ramp(semantic_ratio, c.semantic_ratio_ramp_low, c.semantic_ratio_ramp_high)
        if semantic_ratio > 0.0
        else 0.0
    )
    v_sem = max(v_sem_abs, v_sem_ratio)

    raw = (
        c.agreement_weight_exact * v_exact
        + c.agreement_weight_fuzzy * v_fuzzy
        + c.agreement_weight_bm25 * v_bm25
        + c.agreement_weight_semantic * v_sem
    )

    # Semantic voter combines absolute and ratio per configured mode.
    if c.voter_semantic_mode == "abs_only":
        sem_votes = semantic_score >= c.voter_semantic_abs
    elif c.voter_semantic_mode == "ratio_only":
        sem_votes = semantic_ratio >= c.voter_semantic_ratio
    elif c.voter_semantic_mode == "and":
        sem_votes = (
            semantic_score >= c.voter_semantic_abs and semantic_ratio >= c.voter_semantic_ratio
        )
    else:  # "or"
        sem_votes = (
            semantic_score >= c.voter_semantic_abs or semantic_ratio >= c.voter_semantic_ratio
        )
    voter_flags = (
        exact_score >= c.voter_exact,
        fuzzy_score >= c.voter_fuzzy,
        bm25_token_recall >= c.voter_bm25,
        sem_votes,
    )
    voters = sum(voter_flags)

    if voters >= 3:
        bonus = c.voter_bonus_3_plus
    elif voters == 2:
        bonus = c.voter_bonus_2
    else:
        bonus = 0.0

    return min(1.0, raw + bonus)


def extract_features(
    m: GroundingMatch, cfg: GroundingConfig | None = None, nli_scores: dict | None = None
) -> dict:
    """Feature vector for the Bayesian calibrator (see ``calibration.PREDICTORS``).

    The meaning feature is the *ramped semantic_ratio* - model- and
    language-portable, because a cross-lingual true match has zero lexical
    overlap but a high ratio. Falls back to the absolute-cosine ramp when no
    ratio is available. The other features mirror the layers/voters/penalty
    the deterministic classifier already computes, so the calibrator learns
    the boundary from the same signals an auditor sees.
    """
    c = cfg if cfg is not None else load_config()

    def _ramp(x: float, lo: float, hi: float) -> float:
        return 0.0 if hi <= lo else max(0.0, min(1.0, (x - lo) / (hi - lo)))

    if m.semantic_ratio > 0.0:
        sem = _ramp(m.semantic_ratio, c.semantic_ratio_ramp_low, c.semantic_ratio_ramp_high)
    else:
        sem = _ramp(m.semantic_score, c.semantic_abs_ramp_low, c.semantic_abs_ramp_high)

    if c.voter_semantic_mode == "abs_only":
        sem_votes = m.semantic_score >= c.voter_semantic_abs
    elif c.voter_semantic_mode == "ratio_only":
        sem_votes = m.semantic_ratio >= c.voter_semantic_ratio
    elif c.voter_semantic_mode == "and":
        sem_votes = (
            m.semantic_score >= c.voter_semantic_abs and m.semantic_ratio >= c.voter_semantic_ratio
        )
    else:  # "or"
        sem_votes = (
            m.semantic_score >= c.voter_semantic_abs or m.semantic_ratio >= c.voter_semantic_ratio
        )
    voters = sum(
        (
            m.exact_score >= c.voter_exact,
            m.fuzzy_score >= c.voter_fuzzy,
            m.bm25_token_recall >= c.voter_bm25,
            bool(sem_votes),
        )
    )

    claim_entities = list_claim_entities(m.claim)
    entity_absent = (len(m.entities_absent) / len(claim_entities)) if claim_entities else 0.0

    nli = nli_scores or {}
    return {
        "exact": 1.0 if m.exact_score >= 1.0 else 0.0,
        "fuzzy": float(m.fuzzy_score),
        "bm25_recall": float(m.bm25_token_recall),
        "semantic": float(sem),
        "voters": min(1.0, voters / 4.0),
        "lexical_cosupport": 1.0 if m.lexical_co_support else 0.0,
        "entity_absent": float(entity_absent),
        "nli_entail": float(nli.get("entailment", 0.0)),
        "nli_contra": float(nli.get("contradiction", 0.0)),
    }


_VERDICT_CACHE: dict = {}

# Serialises calibrated-verdict prediction. bambi/PyMC ``model.predict`` is not
# guaranteed thread-safe on a shared model+idata, so when ``ground_batch`` runs
# claims across a thread pool the predict call is the one section taken under a
# lock. The expensive, thread-safe work (ONNX embedding, FAISS search, NLI
# inference - all release the GIL) still runs concurrently.
_VERDICT_LOCK = threading.Lock()


def _config_calibrated_verdict():
    """Return the config-driven calibrated verdict, or None for the lexical engine.

    Active only when ``calibration.engine == "calibrated"`` AND learned
    ``weights`` are present in the config. Cached per (weights, threshold) so a
    batch builds the bambi model once. When inactive, grounding never imports
    the Bayesian stack - the deterministic classifier runs with zero overhead.
    """
    from groundrails import calibration as _cal

    block = _cal.load_calibration_from_config()
    if not block or block.get("engine") != "calibrated" or not block.get("weights"):
        return None
    weights = block["weights"]
    threshold = float(block.get("threshold", 0.5))
    key = (tuple(sorted(weights.items())), threshold)
    v = _VERDICT_CACHE.get(key)
    if v is None:
        v = _cal.CalibratedVerdict.from_weights(weights, threshold=threshold)
        _VERDICT_CACHE[key] = v
    return v


def _verdict_preamble(
    m: GroundingMatch, has_contradiction: bool, has_any_signal: bool
) -> MatchType | None:
    """Engine-independent verdict policy, ONE copy for all three engines.

    ``contradicted`` beats everything; a verbatim exact match beats any
    probabilistic head (a claim found byte-for-byte in the evidence is not
    an uncertain case - adjacent distractor numbers must not push a trained
    head below threshold and un-ground it). Returns ``None`` when the
    engine-specific logic should decide.

    One deliberate carve-out: the deterministic cascade ranks a CONFIDENT
    NLI entailment above the exact returned here (see the cascade branch in
    ``ground``) - on the head engines NLI enters as features instead, so
    exact-beats-head stands there."""
    if has_contradiction and has_any_signal:
        return "contradicted"
    if m.exact_score == 1.0:
        return "exact"
    return None


def _winning_layer_label(
    m: GroundingMatch,
    cfg: GroundingConfig | None = None,
    *,
    fuzzy_threshold: float | None = None,
    semantic_threshold: float | None = None,
) -> MatchType:
    """Provenance label for a calibrated CONFIRMED verdict: the strongest layer.

    Applies the DEF-4 specificity floor when ``cfg`` is given: on a claim
    below ``bm25_min_claim_tokens`` EVERY lexical statistic is computed over
    a degenerate token set (recall saturates, fuzzy alignment is trivially
    partial), so a head confirmation is only honoured when some layer clears
    its own independent threshold - exact (handled above), fuzzy >=
    ``fuzzy_threshold`` or semantic >= ``semantic_threshold``. Otherwise the
    confirmation is withheld ("none") - on ANY engine, not just the
    deterministic cascade. Claims at or above the floor keep the historical
    strongest-positive-layer labelling.

    The keyword thresholds override the cfg values so the floor gates
    against the SAME effective thresholds the verdict used (per-call
    ``fuzzy_threshold`` argument, percentile-derived semantic threshold) -
    otherwise the label and the verdict could disagree about what "clears
    threshold" means."""
    if m.exact_score >= 1.0:
        return "exact"
    cands = {"fuzzy": m.fuzzy_score, "bm25": m.bm25_score, "semantic": m.semantic_score}
    if cfg is not None and m.bm25_claim_token_count < cfg.bm25_min_claim_tokens:
        fuzzy_thr = fuzzy_threshold if fuzzy_threshold is not None else cfg.fuzzy_threshold
        sem_thr = semantic_threshold if semantic_threshold is not None else cfg.semantic_threshold
        if m.fuzzy_score >= fuzzy_thr:
            return "fuzzy"
        if m.semantic_score >= sem_thr:
            return "semantic"
        return "none"
    best = max(cands, key=lambda k: cands[k])
    return best if cands[best] > 0 else "semantic"


_LEXICAL_VERDICT_CACHE: dict = {}


def _config_lexical_verdict(cfg):
    """Resolve lexical mode: (LexicalVerdict, effort, chunk_max_chars, chunk_overlap_ratio) or None.

    Active in lexical mode (``calibration.mode == "lexical"``, the default, which
    load_calibration_from_config resolves to the internal ``engine == "lexical"``)
    AND a ``lexical_manifolds`` block is present. Reads the solution tier from
    ``cfg.lexical_effort``, loads that tier's frozen manifold (cached), and returns
    it with the manifold's chunk operating point. None when manifolds are absent -
    the caller then falls through to the deterministic cascade.
    """
    from groundrails import calibration as _cal
    from groundrails import lexical as _lx

    block = _cal.load_calibration_from_config()
    if not block or block.get("engine") != "lexical" or not block.get("lexical_manifolds"):
        return None
    effort = cfg.lexical_effort
    m = block["lexical_manifolds"].get(effort)
    if not m:
        return None
    key = (
        effort,
        tuple(sorted((m.get("weights") or {}).items())),
        float(m.get("threshold", 0.5)),
        # part of the key: an override editing only the non-EN threshold must
        # not be served a stale verdict for the process lifetime
        m.get("threshold_non_en"),
    )
    v = _LEXICAL_VERDICT_CACHE.get(key)
    if v is None:
        v = _lx.LexicalVerdict.from_config(block, effort)
        if v is None:
            return None
        _LEXICAL_VERDICT_CACHE[key] = v
    return (
        v,
        effort,
        int(m.get("chunk_max_chars", _lx.CHUNK_MAX_CHARS)),
        float(m.get("chunk_overlap_ratio", _lx.CHUNK_OVERLAP_RATIO)),
    )


class UnsupportedLanguageError(ValueError):
    """Raised by :func:`ground` / :func:`ground_batch` when a claim's detected
    language is non-English and has no installed argos ``<lang>->en`` model.

    Cross-lingual grounding cannot run for such a claim, so it is blocked rather
    than scored - unsupported languages must not pollute grounding metrics. Install
    the model (``argospm install translate-<lang>_en``) to enable it. ``lang`` holds
    the offending ISO-639-1 code.
    """

    def __init__(self, lang: str):
        self.lang = lang
        super().__init__(
            f"claim language {lang!r} has no installed argos model; grounding blocked "
            f"(run `argospm install translate-{lang}_en` to enable cross-lingual grounding)"
        )


def ground(
    claim: str,
    sources: Sequence[SourceInput],
    *,
    fuzzy_threshold: float | None = None,
    bm25_threshold: float | None = None,
    semantic_threshold: float | None = None,
    semantic_threshold_percentile: float | None = None,
    agreement_threshold: float | None = None,
    context_chars: int | None = None,
    semantic_grounder=None,
    semantic_top_k: int | None = None,
    config: GroundingConfig | None = None,
    primary_source: str | None = None,
    calibrated_verdict=None,
    nli_grounder=None,
    semantic: bool | None = None,
    ignore_language: bool = False,
) -> GroundingMatch:
    """Ground a single claim against one or more sources.

    Always runs exact, fuzzy, AND BM25 passes independently on every source.
    All three scores are returned so callers see signal from each method and
    can cite line/paragraph/page without rereading.

    Args:
        claim: verbatim claim text to locate
        sources: iterable of raw source text or ``(path, text)`` pairs
        fuzzy_threshold: Levenshtein partial-ratio in [0,1] required to
            classify the best fuzzy alignment as ``"fuzzy"``
        bm25_threshold: token-recall in [0,1] required to classify the best
            BM25 passage as ``"bm25"``. Token-recall = fraction of unique
            claim tokens present in the winning passage. The claim must also
            carry at least ``config.bm25_min_claim_tokens`` unique tokens -
            recall over a tiny token set saturates on a single passage
            (DEF-4), so short claims cannot be confirmed by BM25 alone
        context_chars: max chars of surrounding context per location

    Returns:
        :class:`GroundingMatch` with ``exact_score`` / ``fuzzy_score`` /
        ``bm25_score`` plus locations. ``match_type`` priority: contradicted
        > confident NLI entailment (deterministic cascade only - the head
        engines consume NLI as features/contradiction guard instead) > exact
        > fuzzy (above threshold) > bm25 (token-recall above threshold) >
        semantic > agreement fallback > none. ``combined_score`` =
        ``max(all three)``.

    Raises:
        UnsupportedLanguageError: at the HIGH effort tier, the claim's language is
            confidently detected as non-English with no installed argos
            ``<lang>->en`` model (hard-block so unsupported languages cannot
            pollute scoring).
    """
    from groundrails import settings as _gr_settings

    _gr_settings.require_ready()
    cfg = (config if config is not None else load_config()).overlay(
        fuzzy_threshold=fuzzy_threshold,
        bm25_threshold=bm25_threshold,
        semantic_threshold=semantic_threshold,
        semantic_threshold_percentile=semantic_threshold_percentile,
        agreement_threshold=agreement_threshold,
        context_chars=context_chars,
        semantic_top_k=semantic_top_k,
    )
    # Semantic switch (orthogonal to effort): when on, the OV cascade escalates the
    # uncertain band of the lexical tier. Resolved from config (calibration.mode:
    # semantic) when not passed. The legacy E5 seam (semantic_grounder=) bypasses it;
    # semantic=False short-circuits (the inner lexical call in joint.ground_semantic
    # passes it to avoid re-dispatch).
    if semantic is None and semantic_grounder is None:
        from groundrails.joint import switch_on

        semantic = switch_on()
    if semantic and semantic_grounder is None:
        from groundrails import joint

        return joint.ground_semantic(
            claim, sources, cfg=cfg, primary_source=primary_source, ignore_language=ignore_language
        )
    # Unsupported-language hard-block (HIGH tier only - the tier with the MT
    # bridge): refuse to score a claim whose language is confidently non-English
    # and has no installed argos model. Confidence-gated so lingua misreads of
    # short / keyword English are not blocked; LOW/MEDIUM stay monolingual.
    if cfg.lexical_effort == "high" and not ignore_language:
        from groundrails import lexical as _lx
        from groundrails import lexical_mt as _mt

        _claim_lang = _lx.detect_lang_confident(claim)
        if not _mt.has_model(_claim_lang):
            # The MT bridge only matters cross-lingually: it translates a non-English claim
            # to reach English evidence. When the evidence is in the same language as the
            # claim, the lexical layers match the raw text directly - no bridge needed - so
            # only block when the claim and evidence languages actually differ.
            _ev_lang = _lx.detect_lang_confident(
                " ".join(t for _, _, t in _unpack_sources(sources))[:2000]
            )
            # cross-lingual and no model: try an on-demand argos install before refusing;
            # block only if the model is still unavailable (offline / no such package)
            if _ev_lang != _claim_lang and not _mt.ensure_model(_claim_lang):
                raise UnsupportedLanguageError(_claim_lang)
    # Bind resolved values to local names so the body reads unchanged.
    fuzzy_threshold = cfg.fuzzy_threshold
    bm25_threshold = cfg.bm25_threshold
    semantic_threshold = cfg.semantic_threshold
    semantic_threshold_percentile = cfg.semantic_threshold_percentile
    agreement_threshold = cfg.agreement_threshold
    context_chars = cfg.context_chars
    semantic_top_k = cfg.semantic_top_k

    result = GroundingMatch(claim=claim)
    pairs = _unpack_sources(sources)
    if not pairs:
        return result

    # Exact (regex) pass — first hit wins
    for idx, path, text in pairs:
        span = _exact_match(claim, text)
        if span is not None:
            start, end = span
            result.exact_score = 1.0
            result.exact_matched_text = text[start:end]
            result.exact_location = _locate(
                text, start, end, source_index=idx, source_path=path, context_chars=context_chars
            )
            break

    # Fuzzy (Levenshtein partial-ratio) pass — best across all sources
    for idx, path, text in pairs:
        if not text or not claim:
            continue
        align = partial_ratio_alignment(claim, text)
        ratio = align.score / 100.0
        if ratio > result.fuzzy_score:
            # partial_ratio_alignment optimises edit distance, not readability -
            # its spans routinely start/end mid-word ("nt variance to other").
            # Snap to word boundaries so the cited evidence is usable verbatim.
            f_start, f_end = _snap_to_word_bounds(text, align.dest_start, align.dest_end)
            result.fuzzy_score = ratio
            result.fuzzy_matched_text = text[f_start:f_end]
            result.fuzzy_location = _locate(
                text,
                f_start,
                f_end,
                source_index=idx,
                source_path=path,
                context_chars=context_chars,
            )

    # BM25 pass — rank passages across all sources. The claim token count is
    # a property of the CLAIM, not of the hit - set it unconditionally so the
    # DEF-4 specificity floor never misreads "layer absent" (count 0) as
    # "claim short".
    result.bm25_claim_token_count = len(set(_tokenize(claim)))
    bm25_hit = _bm25_match(claim, pairs, cfg)
    if bm25_hit is not None:
        result.bm25_score = bm25_hit.token_recall  # headline score: agent-interpretable
        result.bm25_raw_score = bm25_hit.raw_score
        result.bm25_token_recall = bm25_hit.token_recall
        result.bm25_matched_text = bm25_hit.matched_text
        # Locate inside the source
        source_text = next(t for i, _, t in pairs if i == bm25_hit.source_index)
        result.bm25_location = _locate(
            source_text,
            bm25_hit.char_start,
            bm25_hit.char_end,
            source_index=bm25_hit.source_index,
            source_path=bm25_hit.source_path,
            context_chars=context_chars,
        )

    # Semantic pass (optional; off unless caller passes a grounder)
    effective_semantic_threshold = semantic_threshold
    if semantic_grounder is not None:
        try:
            # Index only if not already indexed (ground_batch pre-indexes once)
            if getattr(semantic_grounder, "_index", None) is None:
                semantic_grounder.index_sources(pairs)
            # H3: model-agnostic percentile-based threshold override
            if semantic_threshold_percentile is not None:
                pct_thr = semantic_grounder.percentile_threshold(
                    top_pct=semantic_threshold_percentile
                )
                if pct_thr > 0:
                    effective_semantic_threshold = pct_thr
            # Fetch top-3 for semantic_top_k reporting and alternative pointers
            hits = semantic_grounder.search(claim, top_k=semantic_top_k)
            if hits:
                best = hits[0]
                # Raw cosine similarity; for L2-normalised embeddings it lives in
                # [-1, 1] but typical retrieval signal is [0, 1]. Clamp negatives
                # to 0 so the score always reads as an "agreement" level.
                result.semantic_score = max(0.0, min(1.0, best.score))
                result.semantic_matched_text = best.matched_text
                source_text = next(t for i, _, t in pairs if i == best.source_index)
                result.semantic_location = _locate(
                    source_text,
                    best.char_start,
                    best.char_end,
                    source_index=best.source_index,
                    source_path=best.source_path,
                    context_chars=context_chars,
                )
                # Populate top_k summaries
                result.semantic_top_k = [
                    SemanticHitSummary(
                        score=max(0.0, min(1.0, h.score)),
                        matched_text=h.matched_text,
                        source_index=h.source_index,
                        source_path=h.source_path,
                        char_start=h.char_start,
                        char_end=h.char_end,
                    )
                    for h in hits
                ]
                # semantic_ratio (H7): claim vs itself as calibration anchor
                try:
                    self_score = semantic_grounder.self_score(claim)
                    if self_score > 0:
                        result.semantic_ratio = result.semantic_score / self_score
                except Exception as exc:
                    logger.warning("semantic self_score failed (claim=%r): %s", claim[:80], exc)
        except Exception as exc:
            logger.warning(
                "semantic layer failed (claim=%r): %s - lexical-only result returned",
                claim[:80],
                exc,
            )

    # Entity-presence check: flag claim proper nouns absent from ANY source
    # passage (Iter 3 addition). Weaker than contradiction (entity mismatch in
    # the winning passage) but catches fabricated-entity fakes where the
    # specific named entity doesn't appear anywhere in the source.
    full_source_text = "\n".join(t for _, _, t in pairs)
    result.entities_absent = find_absent_entities(claim, full_source_text)

    # Agreement score across layers (always computed; uses 0 for disabled layers).
    # semantic_ratio (H7) is passed so the semantic contribution is
    # model-normalised when available.
    result.agreement_score = _compute_agreement_score(
        exact_score=result.exact_score,
        fuzzy_score=result.fuzzy_score,
        bm25_token_recall=result.bm25_token_recall,
        semantic_score=result.semantic_score,
        semantic_ratio=result.semantic_ratio,
        cfg=cfg,
    )

    # Entity-presence penalty: graded downweight when claim proper-noun
    # entities are absent from the source (Iter 3). Penalty scales with the
    # FRACTION of claim entities missing, capped so agreement_score stays in
    # [0, 1]. Absence is weaker than contradiction so we never escalate to
    # CONTRADICTED here.
    all_claim_entities = list_claim_entities(claim)
    if all_claim_entities and result.entities_absent:
        penalty = cfg.entity_penalty_factor * (
            len(result.entities_absent) / len(all_claim_entities)
        )
        result.agreement_score = max(0.0, result.agreement_score - penalty)

    # combined_score = max of all per-layer scores (legacy, preserved)
    result.combined_score = max(
        result.exact_score,
        result.fuzzy_score,
        result.bm25_score,
        result.semantic_score,
    )

    # Contradiction detection: compare claim against winning passage (H2)
    # Pick the "winning" passage for extraction: priority exact > semantic > bm25 > fuzzy
    # If no layer isolated a clear span, fall back to the single-source text so
    # contradictions are still caught on tiny sources where passage-ranking fails.
    winning_passage = ""
    if result.exact_score == 1.0:
        winning_passage = result.exact_matched_text
    elif result.semantic_score > 0.0 and result.semantic_matched_text:
        winning_passage = result.semantic_matched_text
    elif result.bm25_score > 0.0 and result.bm25_matched_text:
        winning_passage = result.bm25_matched_text
    elif result.fuzzy_score > 0.0 and result.fuzzy_matched_text:
        # Widen the fuzzy window to the full line containing the match so
        # the claim's key value (year, percentage, entity) is not clipped.
        idx = result.fuzzy_location.source_index
        if 0 <= idx < len(pairs):
            src_text = pairs[idx][2]
            start = max(0, result.fuzzy_location.char_start - 100)
            end = min(len(src_text), result.fuzzy_location.char_end + 200)
            winning_passage = src_text[start:end]
        else:
            winning_passage = result.fuzzy_matched_text
    elif len(pairs) == 1:
        # Only one source and no layer anchored: compare claim against the
        # full source text directly. Conservative category-matching in
        # ``find_mismatches`` prevents spurious mismatches here.
        winning_passage = pairs[0][2]

    if winning_passage:
        num_mm, ent_mm = find_mismatches(claim, winning_passage)
        result.numeric_mismatches = num_mm
        result.entity_mismatches = ent_mm

    # Resolve match_type. Contradiction always wins. Then either the calibrated
    # verdict engine (opt-in) or the deterministic cascade decides CONFIRMED.
    has_contradiction = bool(result.numeric_mismatches or result.entity_mismatches)
    has_any_signal = (
        result.exact_score > 0
        or result.fuzzy_score > 0
        or result.bm25_score > 0
        or result.semantic_score > 0
    )

    # NLI / entailment layer (optional): score the best available passage as
    # premise against the claim. Feeds the calibrator as nli_entail/nli_contra -
    # the entailment/truth signal that lexical overlap and cosine similarity miss.
    nli_scores = None
    if nli_grounder is not None:
        premise = result.bm25_matched_text or result.semantic_matched_text
        if not premise and result.exact_matched_text and result.exact_location.char_start >= 0:
            # NEVER feed the claim's own verbatim span back as the premise -
            # premise == claim entails trivially, so NLI would rubber-stamp
            # every exact match. Widen to the surrounding source window so
            # the model can see negation/retraction around the quoted span.
            _, _, src_text = pairs[result.exact_location.source_index]
            s, e = result.exact_location.char_start, result.exact_location.char_end
            lo, _ = _snap_to_word_bounds(src_text, max(0, s - _EVIDENCE_WINDOW_CHARS), s)
            _, hi = _snap_to_word_bounds(src_text, e, e + _EVIDENCE_WINDOW_CHARS)
            premise = src_text[lo:hi]
        if not premise:
            premise = "\n".join(t for _, _, t in pairs)
        try:
            nli_scores = nli_grounder.scores(premise, claim)
            result.nli_scores = nli_scores
        except Exception as exc:
            logger.warning("NLI layer failed (claim=%r): %s", claim[:80], exc)

    # NLI verdict is a first-class grounding signal only when CONFIDENT:
    # argmax contradiction folds into the contradiction guard; entailment
    # confirms only at probability >= cfg.nli_entailment_min (a near-uniform
    # distribution is a shrug, not evidence - argmax at 0.34 must not bypass
    # every lexical threshold). Neutral or unconfident output is NOT a veto:
    # the lexical cascade decides, so NLI enabled can never refuse a claim
    # that strong lexical evidence confirms.
    nli_verdict = None
    if nli_scores is not None:
        top = max(nli_scores, key=nli_scores.get)
        if top == "contradiction" and nli_scores["contradiction"] >= cfg.nli_contradiction_min:
            # same confidence principle as the entailment arm: an argmax
            # contradiction by a nose (0.34) must not refuse a claim that
            # strong lexical evidence confirms
            nli_verdict = "contradicted"
            has_contradiction = True
            has_any_signal = True
        elif top == "entailment" and nli_scores["entailment"] >= cfg.nli_entailment_min:
            nli_verdict = "grounded"
            has_any_signal = True
        # a neutral/unconfident NLI run is a shrug - it is NOT "signal" and
        # must not arm the contradiction preamble on its own

    # Surrounding window of an exact hit, EXCLUDING the claim's own span -
    # WI#6 reason (d') checks it for restated conflicting quantities (the
    # span itself always agrees with itself).
    exact_window_text = ""
    if result.exact_score >= 1.0 and result.exact_location.char_start >= 0:
        _, _, _src_text = pairs[result.exact_location.source_index]
        _s, _e = result.exact_location.char_start, result.exact_location.char_end
        # snap the OUTER edges to word bounds so a truncated window never
        # bisects a number ("counted 1,234 nodes" clipped to "34 nodes"),
        # which would fabricate a value for the reason (d') conflict scan
        _lo, _ = _snap_to_word_bounds(_src_text, max(0, _s - _EVIDENCE_WINDOW_CHARS), _s)
        _, _hi = _snap_to_word_bounds(_src_text, _e, _e + _EVIDENCE_WINDOW_CHARS)
        exact_window_text = _src_text[_lo:_s] + " " + _src_text[_e:_hi]

    # Lexical mode (mode=lexical + shipped manifolds): a per-tier frozen-weight
    # logistic owns the verdict. Tried only when no explicit calibrated_verdict was
    # passed; mutually exclusive with the calibrated head (distinct internal engine).
    lexical_resolved = _config_lexical_verdict(cfg) if calibrated_verdict is None else None
    if lexical_resolved is not None:
        from groundrails import lexical as _lx

        lv, effort, lx_max, lx_ovl = lexical_resolved
        sources_arg = [(path, text) for _, path, text in pairs]
        feat = _lx.extract_lexical_features(
            claim,
            sources_arg,
            effort=effort,
            chunk_max_chars=lx_max,
            chunk_overlap_ratio=lx_ovl,
        )
        result.verdict_features = feat
        result.verdict_probability = lv.predict_proba(feat)
        pre = _verdict_preamble(result, has_contradiction, has_any_signal)
        if pre is not None:
            result.match_type = pre
        elif result.verdict_probability >= lv.threshold_for(feat):
            result.match_type = _winning_layer_label(
                result,
                cfg,
                fuzzy_threshold=fuzzy_threshold,
                semantic_threshold=effective_semantic_threshold,
            )
        else:
            result.match_type = "none"
        _populate_match_metadata(
            result,
            cfg=cfg,
            primary_source=primary_source,
            fuzzy_threshold=fuzzy_threshold,
            bm25_threshold=bm25_threshold,
            effective_semantic_threshold=effective_semantic_threshold,
            exact_window_text=exact_window_text,
        )
        # Borderline flag, mirroring the calibrated engine below: a lexical
        # verdict within the proximity band of its own threshold is cheap to
        # second-guess and should say so.
        if result.match_type in ("exact", "fuzzy", "bm25", "semantic"):
            if (
                abs(result.verdict_probability - lv.threshold_for(feat))
                < cfg.verification_threshold_proximity
            ):
                result.verification_needed = True
        return result

    verdict = (
        calibrated_verdict if calibrated_verdict is not None else _config_calibrated_verdict()
    )

    if verdict is not None:
        # Calibrated engine: P(grounded) from learned weights over the per-layer
        # features. Contradiction still wins; otherwise CONFIRMED iff
        # P >= threshold, labelled by the strongest layer for provenance.
        feat = extract_features(result, cfg, nli_scores)
        with _VERDICT_LOCK:
            p, unc = verdict.predict_with_uncertainty(feat)
        result.verdict_probability = float(p[0] if hasattr(p, "__len__") else p)
        result.verdict_uncertainty = float(unc[0] if hasattr(unc, "__len__") else unc)
        result.verdict_features = feat
        pre = _verdict_preamble(result, has_contradiction, has_any_signal)
        if pre is not None:
            result.match_type = pre
        elif result.verdict_probability >= verdict.threshold:
            result.match_type = _winning_layer_label(
                result,
                cfg,
                fuzzy_threshold=fuzzy_threshold,
                semantic_threshold=effective_semantic_threshold,
            )
        else:
            result.match_type = "none"
    else:
        # Deterministic cascade (default / back-compat):
        # priority: contradicted > confident NLI entailment > exact > fuzzy >
        # bm25 > semantic > agreement > none. Confident entailment confirms
        # past the DEF-4 floor (it is not a degenerate lexical statistic);
        # NLI-read contradiction lands via the contradiction guard above.
        # Anything else NLI says (neutral, unconfident entailment) falls
        # through to the lexical ladder - NLI never vetoes.
        pre = _verdict_preamble(result, has_contradiction, has_any_signal)
        if pre == "contradicted":
            result.match_type = pre
        elif nli_verdict == "grounded":
            # Honest provenance: a layer label only when that layer cleared
            # its OWN threshold; otherwise the verdict is labelled "nli" -
            # a saturated sub-floor bm25 max must not masquerade as lexical
            # recall evidence on an NLI-driven confirm.
            if result.exact_score >= 1.0:
                result.match_type = "exact"
            elif result.fuzzy_score >= fuzzy_threshold:
                result.match_type = "fuzzy"
            elif (
                result.bm25_score >= bm25_threshold
                and result.bm25_claim_token_count >= cfg.bm25_min_claim_tokens
            ):
                result.match_type = "bm25"
            elif result.semantic_score >= effective_semantic_threshold:
                result.match_type = "semantic"
            else:
                result.match_type = "nli"
        elif pre is not None:
            result.match_type = pre
        elif result.fuzzy_score >= fuzzy_threshold:
            result.match_type = "fuzzy"
        elif (
            result.bm25_score >= bm25_threshold
            and result.bm25_claim_token_count >= cfg.bm25_min_claim_tokens
        ):
            # Specificity floor: recall over a tiny claim token set saturates
            # on a single passage, so short claims cannot be confirmed by
            # BM25 alone - they fall through to semantic/agreement.
            result.match_type = "bm25"
        elif result.semantic_score >= effective_semantic_threshold:
            result.match_type = "semantic"
        elif cfg.classifier_mode == "absolute" and result.agreement_score >= agreement_threshold:
            # Multi-layer agreement can confirm even when no single layer passed
            # threshold. adaptive_gap mode applies its per-batch threshold in
            # ground_batch instead.
            if (
                result.semantic_score >= result.bm25_score
                and result.semantic_score >= result.fuzzy_score
            ):
                result.match_type = "semantic"
            elif result.bm25_score >= result.fuzzy_score:
                result.match_type = "bm25"
            elif result.fuzzy_score > 0:
                result.match_type = "fuzzy"
            else:
                result.match_type = "none"
        else:
            result.match_type = "none"

    # WI#3 / WI#5 / WI#6: post-match metadata populated after match_type is set.
    _populate_match_metadata(
        result,
        cfg=cfg,
        primary_source=primary_source,
        fuzzy_threshold=fuzzy_threshold,
        bm25_threshold=bm25_threshold,
        effective_semantic_threshold=effective_semantic_threshold,
        exact_window_text=exact_window_text,
    )

    # Calibrated borderline -> flag for second-guess (P within proximity of tau).
    if verdict is not None and result.match_type in ("exact", "fuzzy", "bm25", "semantic"):
        if (
            abs(result.verdict_probability - verdict.threshold)
            < cfg.verification_threshold_proximity
        ):
            result.verification_needed = True

    return result


# ---------------------------------------------------------------------------
# Post-match metadata (WI#3, WI#5, WI#6)
# ---------------------------------------------------------------------------


def _origin_fallback(m: GroundingMatch) -> tuple[Location, str] | None:
    """Closest LOCATED layer in the premise-origin preference order.

    THE one fallback picker for every consumer that needs a place to look
    when the winning layer carries no span of its own (``nli`` verdicts, and
    cascade-driven ``semantic`` verdicts whose cosine came without a
    location). bm25/semantic passages first; exact points into the window
    the premise widened; fuzzy is a last-resort locator the premise itself
    never used. Guard: positive score AND a located span - four hand-rolled
    copies of this loop drifted apart once already."""
    for score, loc, text in (
        (m.bm25_score, m.bm25_location, m.bm25_matched_text),
        (m.semantic_score, m.semantic_location, m.semantic_matched_text),
        (m.exact_score, m.exact_location, m.exact_matched_text),
        (m.fuzzy_score, m.fuzzy_location, m.fuzzy_matched_text),
    ):
        if score > 0 and loc.char_start >= 0:
            return loc, text
    return None


def _located_support(m: GroundingMatch) -> tuple[Location, str] | None:
    """Located-span picker for a CONTRADICTED verdict - the one shared copy.

    The verbatim exact span first (the strongest locator when the claim is
    quoted then refuted), else the closest located layer via ``_origin_fallback``.
    The guard matters: on the joint path the cascade folds its cosine into
    ``semantic_score`` with NO location before the contradicted verdict is set,
    so a hand-rolled ``semantic_score > 0 -> semantic_location`` arm would cite
    an unlocated span (the ``L-1:C-1`` bug the nli arm was cured of) over a
    located bm25/fuzzy layer that actually carries the conflict."""
    if m.exact_score >= 1.0 and m.exact_location.char_start >= 0:
        return m.exact_location, m.exact_matched_text
    return _origin_fallback(m)


def _restated_numeric_conflict(
    claim_numbers: list[tuple[str, str, str]],
    passage_numbers: list[tuple[str, str, str]],
) -> bool:
    """True when claim and passage share a (unit, context) key but carry
    DIFFERENT value sets under it - a restated conflict beside the agreement
    ("42 nodes" confirmed, "12 nodes" also present under the same key).

    Verification-only: the verdict is untouched. ``find_numeric_mismatches``
    (which drives CONTRADICTED) deliberately reads this shape as a supported
    subset overlap - the moment the claim value is restated anywhere it stops
    looking - so this second-guess flag makes the competing value visible
    without un-grounding the claim. Fires in EITHER direction - a value the
    claim carries that the passage lacks, or one the passage carries beside
    the agreed value - so the exact-window (d') and non-exact (d) reasons
    treat the same shape identically."""
    claim_by_key: dict = {}
    for v, u, cw in claim_numbers:
        if u or cw:
            claim_by_key.setdefault((u, cw), set()).add(v)
    passage_by_key: dict = {}
    for v, u, cw in passage_numbers:
        if u or cw:
            passage_by_key.setdefault((u, cw), set()).add(v)
    for key in claim_by_key.keys() & passage_by_key.keys():
        if claim_by_key[key] != passage_by_key[key]:
            return True
    return False


def _winning_location(m: GroundingMatch) -> Location | None:
    """Return the Location corresponding to the winning match_type."""
    if m.match_type == "exact":
        return m.exact_location
    if m.match_type == "fuzzy":
        return m.fuzzy_location
    if m.match_type == "bm25":
        return m.bm25_location
    if m.match_type == "semantic":
        # a cascade-driven semantic verdict can carry a cosine with no
        # located span (joint path) - fall back like the nli arm below. The
        # score guard mirrors _origin_fallback: a zero-clamped located
        # semantic hit is not evidence and must not be cited directly.
        if m.semantic_score > 0 and m.semantic_location.char_start >= 0:
            return m.semantic_location
        hit = _origin_fallback(m)
        return hit[0] if hit else None
    if m.match_type == "nli":
        hit = _origin_fallback(m)
        return hit[0] if hit else None
    if m.match_type == "contradicted":
        # Navigate to the passage that triggered the contradiction - the one
        # shared located-span picker (exact-first, then _origin_fallback).
        hit = _located_support(m)
        return hit[0] if hit else None
    return None


def _populate_match_metadata(
    m: GroundingMatch,
    *,
    cfg: GroundingConfig,
    primary_source: str | None,
    fuzzy_threshold: float,
    bm25_threshold: float,
    effective_semantic_threshold: float,
    exact_window_text: str = "",
) -> None:
    """Fill WI#3 (cross-source provenance) + WI#5 (lexical co-support) +
    WI#6 (verification_needed, claim_attributes) on an already-scored match.

    Kept as a separate function so ``ground`` stays readable and the
    post-score bookkeeping has a single clear home.
    """
    # --- WI#3: grounded_source + is_primary_source -----------------------
    winning = _winning_location(m)
    if winning is not None and winning.source_path:
        m.grounded_source = winning.source_path
    else:
        m.grounded_source = None

    if primary_source is not None and m.grounded_source is not None:
        m.is_primary_source = m.grounded_source == primary_source
    else:
        m.is_primary_source = True

    # --- WI#5: lexical_co_support --------------------------------------
    # Semantic hits cheaper to second-guess when at least one lexical layer
    # also carries signal. Fuzzy above floor OR bm25 token-recall above
    # floor qualifies. Exact hits are already lexical so they auto-qualify
    # (an exact match is the strongest form of lexical support).
    lexical_ok = (
        m.exact_score >= 1.0
        or m.fuzzy_score >= cfg.lexical_cosupport_fuzzy_min
        or m.bm25_token_recall >= cfg.lexical_cosupport_bm25_min
    )
    m.lexical_co_support = bool(lexical_ok)

    # --- WI#6: claim_attributes (side-by-side numbers + entities) ------
    passage_text = ""
    if winning is not None:
        if m.match_type == "exact":
            passage_text = m.exact_matched_text
        elif m.match_type == "fuzzy":
            passage_text = m.fuzzy_matched_text
        elif m.match_type == "bm25":
            passage_text = m.bm25_matched_text
        elif m.match_type == "semantic":
            passage_text = m.semantic_matched_text
            if not passage_text:
                hit = _origin_fallback(m)
                passage_text = hit[1] if hit else ""
        elif m.match_type == "nli":
            # Mirror the winning-location picker (closest located layer;
            # the exact arm is the bare span, narrower than the +-300
            # window the entailment saw - diagnostic-only difference).
            hit = _origin_fallback(m)
            passage_text = hit[1] if hit else ""
        elif m.match_type == "contradicted":
            # Mirror the winning-location picker above (shared _located_support).
            hit = _located_support(m)
            passage_text = hit[1] if hit else ""

    claim_numbers = extract_numbers(m.claim)
    claim_entities = extract_entities(m.claim)
    passage_numbers = extract_numbers(passage_text) if passage_text else []
    passage_entities = extract_entities(passage_text) if passage_text else []

    m.claim_attributes = {
        "numbers": claim_numbers,
        "entities": claim_entities,
        "passage_numbers": passage_numbers,
        "passage_entities": passage_entities,
    }

    # --- WI#6: verification_needed -------------------------------------
    # A CONFIRMED match carries verification_needed=True when any of these
    # second-guess signals fire. Applies only to the confirmed verdicts
    # (exact / fuzzy / bm25 / semantic / nli); contradicted is already
    # a loud signal and unconfirmed/none needs no further flagging.
    if m.match_type not in ("exact", "fuzzy", "bm25", "semantic", "nli"):
        m.verification_needed = False
        return

    reasons: list[bool] = []

    # (a) semantic-only / NLI-only without lexical support (an nli verdict
    # BY CONSTRUCTION has no lexical layer above threshold)
    if m.match_type in ("semantic", "nli") and not m.lexical_co_support:
        reasons.append(True)

    # (b) cross-source pollution: grounded source != caller's primary
    if not m.is_primary_source:
        reasons.append(True)

    # (c) winning-layer score within proximity of its own threshold
    proximity = cfg.verification_threshold_proximity
    winning_score_floor: tuple[float, float] | None = None
    if m.match_type == "fuzzy":
        winning_score_floor = (m.fuzzy_score, fuzzy_threshold)
    elif m.match_type == "bm25":
        winning_score_floor = (m.bm25_score, bm25_threshold)
    elif m.match_type == "semantic":
        winning_score_floor = (m.semantic_score, effective_semantic_threshold)
    elif m.match_type == "nli" and m.nli_scores:
        winning_score_floor = (m.nli_scores.get("entailment", 0.0), cfg.nli_entailment_min)
    if winning_score_floor is not None:
        score, floor = winning_score_floor
        if score - floor < proximity:
            reasons.append(True)

    # (d) deterministic numeric-mismatch pass was empty, but claim and
    # passage carry DIFFERENT values under a shared (unit, context) key -
    # i.e. the specificity gate in find_numeric_mismatches suppressed a
    # comparison (multi-value ranges, or the claim value restated beside a
    # competing one). Agreeing values under a shared key are the NORMAL shape
    # of a correct numeric confirm and must not flag. Symmetric: a competing
    # value on EITHER side fires (the passage-extra "42 confirmed, 12 also
    # present" case is the same restated conflict (d') catches on exact).
    # Skipped for exact matches: there the picked passage IS the claim's own
    # verbatim span, a vacuous self-comparison; reason (d') below covers
    # exact confirms against their surrounding window instead.
    if m.match_type != "exact" and not m.numeric_mismatches and claim_numbers and passage_numbers:
        if _restated_numeric_conflict(claim_numbers, passage_numbers):
            reasons.append(True)

    # (d') exact confirms: the verbatim span always agrees with itself, but
    # the SURROUNDING window can restate the same quantity with another
    # value ("...has 42 nodes. A later audit counted 12 nodes..."). The
    # verdict stays exact (DEF-9 policy - distractors must not un-ground a
    # verbatim claim); the restated conflict flags for second-guessing. The
    # _restated_numeric_conflict arm also catches the value restated BESIDE
    # the conflict ("...12 nodes, though the spec still lists 42..."), which
    # find_numeric_mismatches reads as supported the moment 42 reappears.
    if m.match_type == "exact" and exact_window_text:
        window_numbers = extract_numbers(exact_window_text)
        if find_numeric_mismatches(m.claim, exact_window_text) or _restated_numeric_conflict(
            claim_numbers, window_numbers
        ):
            reasons.append(True)

    # (e) NLI leaned toward contradiction but stayed below its decisive
    # floor, so the confirm stands on lexical evidence alone. That sub-floor
    # band is exactly what the confidence gate trades away - make it
    # observable instead of silent.
    if m.nli_scores:
        contra = m.nli_scores.get("contradiction", 0.0)
        if contra >= cfg.nli_contradiction_min - proximity or contra >= max(m.nli_scores.values()):
            reasons.append(True)

    m.verification_needed = any(reasons)


def ground_batch(
    claims: Sequence[str],
    sources: Sequence[SourceInput],
    *,
    fuzzy_threshold: float | None = None,
    bm25_threshold: float | None = None,
    semantic_threshold: float | None = None,
    semantic_threshold_percentile: float | None = None,
    agreement_threshold: float | None = None,
    context_chars: int | None = None,
    semantic_grounder=None,
    semantic_top_k: int | None = None,
    config: GroundingConfig | None = None,
    primary_source: str | None = None,
    nli_grounder=None,
    calibrated_verdict=None,
    max_workers: int = 1,
    semantic: bool | None = None,
    ignore_language: bool = False,
) -> list[GroundingMatch]:
    """Batch version of :func:`ground`.

    If ``semantic_grounder`` is provided, the source passages are indexed
    once (chunked + embedded + FAISS) and reused across all claims — major
    speedup over re-indexing per claim.

    ``max_workers`` controls per-claim concurrency. With ``max_workers > 1``
    (and more than one claim) the per-claim :func:`ground` calls run on a
    thread pool. This parallelises the heavy semantic path — ONNX embedding,
    FAISS search and NLI inference all release the GIL — so it is a real
    speedup, not just interleaving. Sources are indexed once up front (before
    the pool), and the calibrated-verdict prediction is serialised internally
    (see ``_VERDICT_LOCK``). Result order matches ``claims``. Default ``1``
    keeps the historical serial behaviour for library callers.

    When ``config.classifier_mode == "adaptive_gap"`` (H11), semantic-zone
    claims (those where no lexical layer cleared its threshold) are
    reclassified using a per-batch threshold computed as the midpoint of
    the largest gap in their ``agreement_score`` distribution. Makes the
    classifier rank-based and therefore portable across embedding models
    with different absolute-score distributions.
    """
    from groundrails import settings as _gr_settings

    _gr_settings.require_ready()
    cfg = (config if config is not None else load_config()).overlay(
        fuzzy_threshold=fuzzy_threshold,
        bm25_threshold=bm25_threshold,
        semantic_threshold=semantic_threshold,
        semantic_threshold_percentile=semantic_threshold_percentile,
        agreement_threshold=agreement_threshold,
        context_chars=context_chars,
        semantic_top_k=semantic_top_k,
    )

    # Semantic switch: build the cascade + joint head once and reuse across claims.
    # Run serially - one OpenVINO compiled model is not safe to share across threads,
    # and the escalation gate sends only a fraction of claims to the cascade anyway.
    if semantic is None and semantic_grounder is None:
        from groundrails.joint import switch_on

        semantic = switch_on()
    if semantic and semantic_grounder is None:
        from groundrails import joint

        block = joint.load_semantic_block()
        jv = joint.JointVerdict.from_config(block) if block else None
        casc = None
        if block and jv is not None:
            from groundrails.semantic_ov import SemanticCascade

            casc = SemanticCascade(
                top_k=int(block.get("top_k", 8)),
                gate=tuple(block.get("cosine_gate", (0.493, 0.739))),
                band=tuple(block.get("cascade_band", (0.01, 0.66))),
            )
        sem_out: list[GroundingMatch] = []
        for c in claims:
            try:
                sem_out.append(
                    joint.ground_semantic(
                        c,
                        sources,
                        cfg=cfg,
                        cascade=casc,
                        joint_verdict=jv,
                        primary_source=primary_source,
                        ignore_language=ignore_language,
                    )
                )
            except Exception as exc:
                # Per-claim isolation: one failed claim must not abort the batch.
                logger.warning(
                    "semantic grounding failed for claim %r: %s - returned ungrounded", c[:80], exc
                )
                sem_out.append(GroundingMatch(claim=c, match_type="none"))
        return sem_out

    # Eagerly index sources once for the semantic layer if supplied
    if semantic_grounder is not None:
        try:
            pairs = _unpack_sources(sources)
            semantic_grounder.index_sources(pairs)
        except Exception as exc:
            logger.error(
                "semantic index_sources failed: %s - disabling semantic layer for this batch",
                exc,
            )
            semantic_grounder = None  # disable on error

    # Resolve the calibrated verdict once (cached); reused for every claim so a
    # batch builds the bambi model at most once. An explicit ``calibrated_verdict``
    # argument takes precedence over the config-driven one.
    verdict = (
        calibrated_verdict if calibrated_verdict is not None else _config_calibrated_verdict()
    )

    def _ground_one(c: str) -> GroundingMatch:
        try:
            return ground(
                c,
                sources,
                semantic_grounder=semantic_grounder,
                config=cfg,
                primary_source=primary_source,
                calibrated_verdict=verdict,
                nli_grounder=nli_grounder,
                ignore_language=ignore_language,
            )
        except UnsupportedLanguageError:
            raise  # deliberate refusal - surface it (use ignore_language to bypass)
        except Exception as exc:
            # Per-claim isolation: a per-claim bug must not abort the batch.
            logger.warning("grounding failed for claim %r: %s - returned ungrounded", c[:80], exc)
            return GroundingMatch(claim=c, match_type="none")

    workers = min(max(1, max_workers), len(claims)) if claims else 1
    if workers > 1:
        # ThreadPoolExecutor.map preserves input order, so matches[i] still
        # corresponds to claims[i] (the adaptive_gap pass below relies on it).
        with ThreadPoolExecutor(max_workers=workers) as pool:
            matches = list(pool.map(_ground_one, claims))
    else:
        matches = [_ground_one(c) for c in claims]

    # H11: adaptive_gap classifier — reclassify semantic-zone claims (those
    # where no lexical layer cleared its threshold) using a per-batch
    # threshold derived from the distribution's gap structure. Model-
    # agnostic: rank ordering of agreement_scores is stable across semantic
    # models even when absolute scales differ wildly. Skipped when the
    # calibrated engine is active (it owns the verdict).
    lexical_active = calibrated_verdict is None and _config_lexical_verdict(cfg) is not None
    if cfg.classifier_mode == "adaptive_gap" and verdict is None and not lexical_active:
        semantic_zone_idxs = [
            i
            for i, m in enumerate(matches)
            if m.exact_score < cfg.voter_exact
            and m.fuzzy_score < cfg.fuzzy_threshold
            # bm25 "cleared" now also requires the DEF-4 specificity floor;
            # a short saturated claim refused by the floor must still be
            # eligible for the semantic-zone rescue, not stranded.
            and (
                m.bm25_score < cfg.bm25_threshold
                or m.bm25_claim_token_count < cfg.bm25_min_claim_tokens
            )
            and not m.numeric_mismatches
            and not m.entity_mismatches
        ]
        if len(semantic_zone_idxs) >= cfg.adaptive_gap_min_claims:
            scored = sorted(
                ((matches[i].agreement_score, i) for i in semantic_zone_idxs),
                key=lambda p: p[0],
            )
            # Find the real-vs-fake boundary: largest gap in the BOTTOM HALF
            # of the sorted distribution. The entity-presence penalty (Iter
            # 3) pushes fakes toward the bottom; the first significant gap
            # above them marks where support begins. Using the global
            # largest gap instead would place the threshold in the top
            # half and misclassify legitimate middling-confidence claims.
            bottom_half_cutoff = max(2, (len(scored) + 1) // 2)
            candidate_gaps = [
                (scored[j + 1][0] - scored[j][0], j)
                for j in range(min(len(scored) - 1, bottom_half_cutoff))
            ]
            largest_gap, gap_idx = max(candidate_gaps)
            adaptive_thr = (scored[gap_idx][0] + scored[gap_idx + 1][0]) / 2
            # In adaptive_gap mode the agreement_threshold config value acts
            # as an absolute FLOOR only when the adaptive gap is meaningless
            # (degenerate distribution). When adaptive_thr is a real
            # separator, use it directly even if below the absolute floor —
            # that's the whole point of making the classifier rank-based.
            effective_thr = adaptive_thr

            for i in semantic_zone_idxs:
                m = matches[i]
                if m.match_type != "none":
                    continue
                if m.agreement_score >= effective_thr:
                    # Label as "semantic" uniformly. The adaptive_gap
                    # classifier is the *semantic-zone* classifier by
                    # definition (we only enter this branch for claims
                    # where no lexical layer cleared threshold). Using the
                    # "highest-contributing layer" label here would
                    # re-introduce model-dependence (different models
                    # weight fuzzy/bm25/semantic differently for the same
                    # claim, breaking portability).
                    m.match_type = "semantic"
                    # Re-run WI#3/#5/#6 metadata - match_type just changed
                    # from "none" to "semantic" so verification_needed and
                    # grounded_source need fresh computation against the
                    # semantic layer's location and threshold. cfg thresholds
                    # here, not the per-claim percentile-effective one (not
                    # retained across the batch); affects only the diagnostic
                    # proximity flag, never the verdict.
                    _populate_match_metadata(
                        m,
                        cfg=cfg,
                        primary_source=primary_source,
                        fuzzy_threshold=cfg.fuzzy_threshold,
                        bm25_threshold=cfg.bm25_threshold,
                        effective_semantic_threshold=cfg.semantic_threshold,
                    )

    return matches


# -- grounding document (business-end provenance report) ----------------------


def _final_score(m: GroundingMatch) -> float:
    """The single headline score: the calibrated verdict probability if it ran, else the
    max-over-layers combined score.

    Exact matches report the layer score, not the head probability: the exact
    short-circuit can confirm BELOW the head's threshold, and a document
    saying ``grounded: true, match_type: exact, score: 0.31`` while the
    deterministic engine reports 1.0 for the same claim is one verdict with
    two score semantics."""
    if m.match_type == "exact":
        return max(m.verdict_probability, m.combined_score)
    if m.match_type == "nli" and m.nli_scores:
        # the deciding evidence IS the entailment probability - the lexical
        # combined_score is sub-threshold by construction on this verdict
        return float(m.nli_scores.get("entailment", 0.0))
    return m.verdict_probability if m.verdict_probability >= 0 else m.combined_score


def _contradiction(m: GroundingMatch) -> dict | None:
    """Numeric / named-entity disagreements with the winning passage, or None."""
    if not (m.numeric_mismatches or m.entity_mismatches):
        return None
    return {
        "numeric": [list(t) for t in m.numeric_mismatches],
        "entity": [list(t) for t in m.entity_mismatches],
    }


def _claim_location(claim_obj) -> dict | None:
    """Where the claim sits in the answer document, from a ``claims.Claim`` (or None)."""
    if claim_obj is None:
        return None
    line = getattr(claim_obj, "line_number", None)
    cs = getattr(claim_obj, "char_start", None)
    ce = getattr(claim_obj, "char_end", None)
    if line is None and cs is None:
        return None
    return {"line": line, "char_start": cs, "char_end": ce}


def build_grounding_document(matches, claims=None, sources=None) -> dict:
    """Assemble the business-end grounding document from already-computed matches: per claim
    its verdict, the single final score, and the support provenance - no per-scorer internals.

    ``claims`` (optional, aligned to ``matches``) supplies each claim's location in the answer
    document (id, line, char span) as written by extract-claims; ``sources`` lists the evidence
    paths. The result is JSON-serialisable - the dict the API and ``--json`` return."""
    entries = []
    for i, m in enumerate(matches):
        c = claims[i] if claims and i < len(claims) else None
        entries.append(
            {
                "id": getattr(c, "id", None),
                "claim": m.claim,
                "claim_location": _claim_location(c),
                "grounded": m.grounded,
                "match_type": m.match_type,
                "score": round(_final_score(m), 4),
                "support": m.support,
                "contradiction": _contradiction(m),
            }
        )
    doc = {
        "summary": {
            "total": len(matches),
            "grounded": sum(1 for m in matches if m.grounded),
            "ungrounded": sum(1 for m in matches if not m.grounded),
        },
        "claims": entries,
    }
    if sources is not None:
        paths = [s[0] for s in sources if isinstance(s, tuple)]
        if paths:
            doc = {"sources": paths, **doc}
    return doc


def grounding_document(claims, sources, **kwargs) -> dict:
    """Ground every claim and return the grounding document (a dict). ``claims`` may be plain
    strings or objects carrying ``id`` / ``line_number`` / ``char_start`` / ``char_end`` (e.g.
    ``claims.Claim`` from extract-claims), which become each claim's answer-document location.
    Extra keyword arguments pass through to :func:`ground_batch`."""
    texts = [c if isinstance(c, str) else c.claim for c in claims]
    objs = [None if isinstance(c, str) else c for c in claims]
    matches = ground_batch(texts, sources, **kwargs)
    return build_grounding_document(matches, claims=objs, sources=sources)

"""Claim extraction: turn a document into a claims.json to feed `ground`.

The grounding tool's heaviest manual step is producing the claims list.
This module provides a deterministic sentence-per-line heuristic extractor
that emits JSON conforming to the ``groundrails.claims.Claim`` schema:

    [
        {"id": "c01", "claim": "We observed 42 concurrent sessions."},
        {"id": "c02", "claim": "The dataset grew to 1.7M rows."},
        ...
    ]

The heuristic is lossy by design - a sales brief with bullet lists and
markdown headers does not parse into clean assertions without human
judgement. The output is a starting point; the caller is expected to
review the extracted claims before grounding.

Filtering rules (drop a sentence when):
    - shorter than 20 chars (headers, stubs)
    - no verb-shaped content (crude: no copula "is/are/was/...", and no
      word ending in -s/-ed/-ing inside a sentence of at least 4 words)
    - a bare anaphoric stub ("This is ..." under 6 words with no digit)
    - under a bibliographic heading (``## Sources``, ``## References`` ...)
      or inside a fenced code block
    - starts with a heading/table prefix (``#``, ``|``); blockquote
      content is kept with the ``>`` marker stripped
    - empty after stripping list markers ``- ``, ``* ``, ``1. ``

Claims keep their order of appearance; IDs are ``c01``..``cNN``
zero-padded to two digits (fits up to 99 claims; beyond that the ID
widens automatically).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from groundrails.entity_check import _CITATION_RE


@dataclass
class ExtractedClaim:
    """A single candidate claim with stable ID, source line, and char span.

    ``char_start`` / ``char_end`` are the claim's 0-based offsets in the answer document
    (``-1`` when the sentence could not be relocated after markdown stripping)."""

    id: str
    claim: str
    line_number: int
    char_start: int = -1
    char_end: int = -1
    # Why this sentence cannot be a claim ABOUT the sources (``None`` when it can).
    # Set by :func:`out_of_scope`; the claim is still extracted and still grounded -
    # this only records that a grounding verdict on it carries no information.
    out_of_scope_reason: str | None = None


# Sentence-end regex. Splits on ``. ! ?`` followed by whitespace and a
# capital-letter or digit start. Tolerates common abbreviations by
# requiring a capital/digit after the whitespace (a sentence rarely
# continues with lowercase). One closing quote or bracket may sit between the
# terminator and the space (``...12%." The``), and one opening quote may sit
# before the capital - without that the boundary was invisible and two
# assertions were grounded as one unit (DEF-CLAIM-28). Citation markers after
# the terminator (``... 2023. [1] Demand``) stay with the sentence they cite
# (DEF-CLAIM-41); ``tail`` is the part kept on the left. An opening bracket is
# not accepted: after an abbreviation the guard does not list (``4 p.m. (EST)
# with ...``) it split the sentence and the part without a verb was dropped.
_SENT_SPLIT_RE = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"'\u201d\u2019)\]]))(?P<tail>(?:[ \t]*"
    + _CITATION_RE.pattern
    + r")*)"
    r"\s+(?=[\"'\u201c\u2018]?[A-Z0-9])"
)

# Abbreviations whose trailing period is NOT a sentence end. Checked against
# the text immediately before a candidate split point so citations like
# "et al. 2008", "Buchanan, C. (1991)", "e.g. Smith" and "Fig. 3" survive as
# one sentence instead of shattering into ungroundable fragments.
_ABBREV_BEFORE_RE = re.compile(
    r"""(?:
        \bet\ al\.                                  # et al. 2008
        | \b(?:e\.g|i\.e|cf|vs|ca|approx)\.         # latinate abbreviations
        | \b(?:Fig|Eq|No|Vol|pp|p|ed|eds|Ch|Sec)\.  # scholarly references
        | \b(?:Dr|Prof|Mr|Mrs|Ms|St|Jr|Sr)\.        # honorifics
    )$""",
    re.VERBOSE,
)

# Comma-led author initial ("Buchanan, C.") - guarded only when what FOLLOWS
# the candidate boundary looks like a citation year ("(1991)" / "1991"), so a
# real sentence ending in an enumeration ("...were B, C. The trial ran...")
# still splits.
_INITIAL_BEFORE_RE = re.compile(r",\s*[A-Z]\.$")
# A 4-digit publication year (optionally parenthesised), NOT any digit: the
# guard suppresses the sentence split only for a genuine "Smith, J. (2020)"
# citation, so an enumeration ending in an initial followed by a numeric
# sentence ("...were B, C. 400 families enrolled...") still splits.
_CITATION_AFTER_RE = re.compile(r"\(?(?:1[5-9]|20)\d{2}(?!\d)")

# Longest guard string is "approx." after a comma-initial; a bounded look-back
# window keeps the per-boundary check O(1) instead of re-scanning the whole
# prefix (O(n^2) on reference-heavy documents).
_ABBREV_WINDOW = 24


def _split_sentences(text: str) -> list[str]:
    """Sentence split with abbreviation protection.

    Uses ``_SENT_SPLIT_RE`` to find candidate boundaries, then rejects any
    boundary whose preceding text ends in a known abbreviation (comma-led
    initials, ``et al.``, ``e.g.``, ``Fig.`` ...). This is the root-cause fix
    for reference-list fragments like ``et al. 2008`` splitting mid-citation.
    Initials guard only after a comma AND only when a citation year follows
    the boundary, so a real sentence ending in a single capital ("... vitamin
    D." / "... were B, C.") still splits; bare ``etc.`` is likewise NOT
    guarded - it legitimately ends sentences far more often than citations.
    """
    parts: list[str] = []
    cursor = 0
    for m in _SENT_SPLIT_RE.finditer(text):
        window = text[max(0, m.start() - _ABBREV_WINDOW) : m.start()]
        if _ABBREV_BEFORE_RE.search(window):
            continue  # abbreviation period, not a sentence end
        if _INITIAL_BEFORE_RE.search(window) and _CITATION_AFTER_RE.match(text, m.end()):
            # author initial followed by a citation year. match(text, pos)
            # avoids slicing the whole remaining document per boundary -
            # comma-initial boundaries cluster exactly on reference-heavy
            # documents, the O(n^2) case _ABBREV_WINDOW exists to prevent.
            continue
        parts.append(text[cursor : m.end("tail")])
        cursor = m.end()
    parts.append(text[cursor:])
    return parts


# Section headings whose content is bibliographic/navigational, not
# assertive. Sentences under these headings are reference entries; grounding
# them against a digest of the same paper is a tautology that inflates the
# score, so the extractor skips the whole section.
_NON_CLAIM_SECTION_RE = re.compile(
    r"^(?:\d+[.)]?\s+)?(?:references?|sources|bibliography|citations|works\s+cited|"
    r"further\s+reading|see\s+also|acknowledg(?:e)?ments?)\s*:?\s*$",
    re.IGNORECASE,
)

# Fenced code block delimiter (``` or ~~~ runs, optionally with a language
# tag). The full run is captured: per CommonMark a fence closes only on the
# SAME character with at least the opener's length, so a ``` example shown
# inside a ```` block (or a ~~~ line inside ```) is content, not a toggle.
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")

# Markdown bullet/number prefixes stripped before length check
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")

# Crude "has verb-like content" check: at least one copula or a word
# ending in -s/-ed/-ing. Used to drop pure noun-phrase headers like
# "Key Features" or "Technical Stack".
_COPULA_RE = re.compile(r"\b(is|are|was|were|has|have|had|will|can|should|must)\b", re.IGNORECASE)
_VERB_SUFFIX_RE = re.compile(r"\b\w+(?:ed|ing|s)\b")

# Irregular pasts the suffix regex cannot see - the canonical trend verbs of
# report prose ("revenue fell", "output rose"). A dropped claim is never
# grounded (fail-open), so recall matters here; the whitelist is closed by
# design (regex tier - see Known limitations in docs/defects.md).
_IRREGULAR_PAST_RE = re.compile(
    r"\b(?:rose|fell|grew|ran|held|led|won|made|took|gave|came|went|got|kept|"
    r"found|left|lost|met|paid|sold|told|built|sent|spent|drew|threw|began|"
    r"brought|bought|thought|sought|taught|caught|became|saw|drove|wrote|"
    r"broke|chose|spoke|stood|understood|shrank|shrunk|slid|rode|did|said|"
    # invariant pasts (present == past form) - common in trend/report prose
    # ("revenue hit a low", "the board cut the dividend", "turnover beat
    # forecast"); the suffix regex cannot see them and no other word in the
    # sentence need carry an -ed/-ing/-s ending, so they were silently dropped
    r"hit|cut|set|let|put|cost|beat|quit|read|spread|split|shut|burst|sank|"
    # prefixed irregulars (the list already carries the bare "understood")
    r"withdrew|overtook|undertook|oversaw|outgrew|rebuilt|withstood)\b",
    re.IGNORECASE,
)

# Minimum chars after trimming. Below this the sentence is almost
# always a heading or stub with no groundable assertion.
_MIN_CLAIM_CHARS = 20


def _strip_markdown_noise(line: str) -> str:
    """Remove leading markdown markup so the sentence starts cleanly."""
    stripped = line.lstrip()
    if stripped.startswith("#"):
        return ""
    if stripped.startswith(">"):
        stripped = stripped.lstrip("> ").strip()
    if stripped.startswith("|"):
        return ""
    stripped = _LIST_PREFIX_RE.sub("", stripped)
    return stripped.strip()


# Bare anaphoric openers: a demonstrative/pronoun subject directly followed by
# a copula has no groundable referent of its own ("This is important.",
# "It was fine."). "This model achieves ..." is NOT matched - there the
# demonstrative modifies a noun, so the claim carries its own subject.
_ANAPHORIC_STUB_RE = re.compile(
    r"^(?:this|that|these|those|it|they)\s+(?:is|are|was|were)\b", re.IGNORECASE
)

# Words counted for the minimum-token requirement on the verb-suffix path.
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


def _looks_like_claim(candidate: str) -> bool:
    """Return True if the candidate sentence has verb-shaped content.

    Predicate requirement: a copula, or a verb-suffixed word inside a
    sentence of at least 4 words. The word floor stops noun-phrase
    fragments like ``Digest only, paywalled`` (where ``paywalled`` matches
    the ``-ed`` suffix but nothing predicates anything) from passing.
    Anaphora rejection: bare ``This is ... / It was ...`` openers are
    dropped - the claim's subject lives outside the sentence, so no
    source passage can ground it on its own.
    """
    if len(candidate) < _MIN_CLAIM_CHARS:
        return False
    if (
        _ANAPHORIC_STUB_RE.match(candidate)
        and len(_WORD_RE.findall(candidate)) < 6
        and not any(ch.isdigit() for ch in candidate)
    ):
        # Contentless anaphoric stub ("This is important."): subject lives
        # outside the sentence and nothing else anchors it. Anaphoric
        # sentences that carry numbers or enough content ("It was completed
        # in 1889.") remain groundable and are kept.
        return False
    if _COPULA_RE.search(candidate):
        return True
    return (
        bool(_VERB_SUFFIX_RE.search(candidate) or _IRREGULAR_PAST_RE.search(candidate))
        and len(_WORD_RE.findall(candidate)) >= 4
    )


# --- out-of-scope classification -----------------------------------------
#
# A sentence can be perfectly well-formed, carry a verb and still be
# ungroundable IN PRINCIPLE, because it is not an assertion about the source
# corpus at all. On prose documents (executive summaries, proposals, reports)
# this class is large - it was 27 of 74 claims on the labelled prose set - and
# every one of them runs the full lexical tier and then escalates to the
# OpenVINO cascade before failing, which is the dominant per-document waste.
#
# The rules below are deliberately narrow. A false positive here is the
# DANGEROUS direction (a real claim skipped by the cascade), so each rule was
# tuned to zero false positives against the claims on that set which a human
# verified as groundable or which the lexical tier confirmed, AND against the
# 63,054 claims of the public VitaminC dev split - every one written about its
# evidence, so any firing there is a false positive. A rule is kept only if it
# cannot describe a sourced fact in principle: superlatives ("the single most"),
# contrastive reframes ("X, not Y"), approximate costs and property lists all
# occur in factual text, so the evaluative, estimate and design shapes of
# authored prose stay in scope (DEF-CLAIM-23) - they need semantics the
# deterministic tier does not have.

# 1. Hypothetical. A leading disjunction of CLAUSES ("either X happened, or Y")
#    asserts a branch, not a fact. Anchored at sentence start: an EMBEDDED
#    "either ... or" is a real claim's internal disjunction ("at 50-100 m either a
#    long focal length is specified or the GSD target moves"), and a disjunction
#    of names ("Either A or B voices the courier") is a sourced fact. A
#    conditional is NOT classified, even with a modal in the consequent: sources
#    state rules and guidance that way ("If you develop an infection, you may need
#    IV antibiotics"), and on public RAG answers the modal-conditional rule fired
#    at the same rate as on their sources, where every firing is wrong
#    (DEF-CLAIM-29, DEF-CLAIM-36).
_EITHER_CLAUSES_RE = re.compile(r"^\s*either\b[^,]*,\s*or\b", re.IGNORECASE)

# 2. Document self-reference. The sentence describes this document's own
#    structure or artefacts rather than the world the sources describe.
_SELF_REF_RE = re.compile(
    r"\bthis\s+(?:summary|document|report|round|note|section|memo|paper|analysis)\b"
    r"|\bthe\s+(?:full\s+)?pre-?registration\b"
    r"|\bregistered\s+(?:set|hypothes[ei]s)\b",
    re.IGNORECASE,
)
# A backticked relative path or filename is this document pointing at its own files.
_SELF_PATH_RE = re.compile(r"`[^`]*(?:/|\.md|\.json|\.ya?ml)[^`]*`")

# 3. Directive. A recommendation about what to do, not a statement of fact.
#    "Note that ..." / "See ..." are deliberately EXCLUDED - they front real
#    assertions ("Note too that derived crack dimensions carry relative errors
#    from -35% to +120%").
_DIRECTIVE_OPENER_RE = re.compile(r"^\s*(?:sell|buy|avoid|prefer|do\s+not|don't)\b", re.IGNORECASE)
_DEONTIC_RE = re.compile(
    r"\bthe\s+(?:correct|right|only\s+defensible|obvious)\s+(?:move|offer|vehicle|answer)\b"
    r"|\bshould\s+(?:not\s+)?be\s+(?:proposed|scoped|sold|offered|written)\b",
    re.IGNORECASE,
)


def out_of_scope(claim: str) -> str | None:
    """Why ``claim`` cannot be an assertion about a source corpus, else ``None``.

    Returns one of ``"hypothetical"``, ``"self-reference"``, ``"directive"``.
    Purely a function of the sentence text, so both the extractor and the
    grounder can call it without threading state between them.

    This never means "false" and never means "drop it" - an out-of-scope claim
    is still extracted and still grounded. It means a verdict on this sentence
    is uninformative, so it should not be counted as a grounding failure and
    should not pay for a semantic-cascade escalation that cannot succeed.
    """
    # A disjunction carrying a DIGIT has a checkable branch - keep it in scope.
    if not any(ch.isdigit() for ch in claim) and _EITHER_CLAUSES_RE.match(claim):
        return "hypothetical"
    if _SELF_REF_RE.search(claim) or _SELF_PATH_RE.search(claim):
        return "self-reference"
    if _DIRECTIVE_OPENER_RE.match(claim) or _DEONTIC_RE.search(claim):
        return "directive"
    return None


# --- language guard --------------------------------------------------------
#
# ``_looks_like_claim`` recognises a claim by English copulas and English verb
# endings, and ``_WORD_RE`` counts ASCII words, so a document in any other language
# loses almost every sentence here: a Polish document kept 5 of 54 sentences, and
# the 5 passed only on English endings inside Polish words (DEF-CLAIM-26). The
# multilingual bridge (``lexical_mt``) translates claims during grounding, which
# runs after extraction, so it cannot recover a sentence dropped here. The guard
# names the problem instead of returning a near-empty claim list in silence.

# Prose sample handed to the detector - the same cap ``ground`` uses for evidence.
_LANG_SAMPLE_CHARS = 2000


def warn_if_not_english(sentences: list[str]) -> str | None:
    """Warn when the document prose is confidently not English.

    Returns the detected ISO 639-1 code when it warned, else ``None``. Detection is
    the confidence-gated lingua read (:func:`groundrails.lexical.detect_lang_confident`),
    so short or ambiguous English reads ``und`` and never warns.
    """
    from groundrails.lexical import detect_lang_confident

    lang = detect_lang_confident(" ".join(sentences)[:_LANG_SAMPLE_CHARS])
    if lang in ("en", "und"):
        return None
    from loguru import logger

    logger.warning(
        f"document language is '{lang}', not English. The claim extractor reads "
        "English only and drops most sentences in other languages; claims from this "
        "document require the multilingual bridge, which the extractor does not use"
    )
    return lang


_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.*)$")

# An HTML comment is an author's note, never an assertion of the document. It is
# removed only outside code: one scan matches fenced blocks, inline code spans and
# comments, and whichever starts first wins. A literal ``<!--``
# inside code no longer opens a comment that runs to a later ``-->`` (DEF-CLAIM-35),
# and a comment whose body holds a backtick is still one comment (DEF-CLAIM-27). The
# fence arm mirrors ``_FENCE_RE`` and the closer rule in ``_split_document``. An
# inline span does not cross a blank line or a fence line, because CommonMark reads
# fenced blocks before inline spans. Both accept CRLF line ends.
_COMMENT_OR_CODE_RE = re.compile(
    r"^[ \t]*(?P<fence>(?P<fc>[`~])(?P=fc){2,}).*?(?:\n[ \t]*(?P=fence)(?P=fc)*[ \t]*\r?$|\Z)"
    r"|(?<!`)(?P<tick>`+)(?!`)(?:(?!\n[ \t]*(?:\r?\n|`{3}|~{3})).)*?(?<!`)(?P=tick)(?!`)"
    r"|(?P<comment><!--.*?-->)",
    re.DOTALL | re.MULTILINE,
)


def _split_document(text: str) -> list[tuple[int, str]]:
    """Split document into ``(source_line_number, sentence)`` pairs.

    Walks line-by-line so we can keep each claim's line number (helps
    the reviewer jump back to the source). Within a paragraph the text
    is joined and split on sentence boundaries.

    Section-aware: markdown headings are tracked, and every line under a
    bibliographic heading (``## Sources``, ``## References`` ...) is
    skipped until the next heading. Reference entries are pointers to
    other documents, not assertions of this one.
    """
    # Comments are removed before anything else (DEF-CLAIM-27). Each is replaced by
    # the newlines it spanned, so every later line keeps its line number.
    text = _COMMENT_OR_CODE_RE.sub(
        lambda m: "\n" * m.group(0).count("\n") if m.group("comment") else m.group(0), text
    )
    out: list[tuple[int, str]] = []
    paragraph_lines: list[tuple[int, str]] = []
    in_non_claim_section = False
    non_claim_level = 0
    fence_open: str | None = None

    def flush() -> None:
        if not paragraph_lines:
            return
        para_text = " ".join(line for _, line in paragraph_lines).strip()
        if not para_text:
            paragraph_lines.clear()
            return
        start_line = paragraph_lines[0][0]
        for sent in _split_sentences(para_text):
            sent = sent.strip()
            if sent:
                out.append((start_line, sent))
        paragraph_lines.clear()

    for idx, raw_line in enumerate(text.splitlines(), start=1):
        # Fenced code blocks: their lines are code, not prose - and a code
        # comment like "# sources" must NOT be parsed as a section heading
        # (that would silently black out extraction of everything after it).
        # A fence closes only on the opener's character at >= its length; a
        # different fence line inside the block is content, not a toggle.
        fence = _FENCE_RE.match(raw_line)
        if fence:
            delim = fence.group(1)
            if fence_open is None:
                flush()
                fence_open = delim
                continue
            # a CLOSER additionally allows nothing but whitespace after the
            # run - a "```python" line inside an open ``` block is content
            # (a tutorial showing how to open a fence), not the closer
            if (
                delim[0] == fence_open[0]
                and len(delim) >= len(fence_open)
                and not raw_line[fence.end() :].strip()
            ):
                fence_open = None
                continue
        if fence_open is not None:
            continue
        heading = _HEADING_RE.match(raw_line)
        if heading:
            flush()
            level = len(heading.group(1))
            if _NON_CLAIM_SECTION_RE.match(heading.group(2).strip()):
                # keep the OUTERMOST bibliographic level: a DEEPER bib-named
                # sub-heading ("### Sources" under "## References") must not
                # narrow the skip so a sibling "### Journal papers" leaks;
                # a SHALLOWER one ("## Sources" after "### References")
                # starts its own wider section and DOES take over the level
                if not in_non_claim_section or level < non_claim_level:
                    non_claim_level = level
                in_non_claim_section = True
            elif in_non_claim_section and level > non_claim_level:
                # a SUB-heading inside the bibliography ("### Primary
                # sources" under "## References") stays part of it - only a
                # heading at the same or higher level ends the section
                pass
            else:
                in_non_claim_section = False
            continue
        if in_non_claim_section:
            continue
        # A list item is its own unit: a bullet with no closing full stop must not
        # run into the next bullet as one compound claim (DEF-CLAIM-24). Indented
        # continuation lines of the same item carry no marker and still join it.
        if _LIST_PREFIX_RE.match(raw_line.lstrip().lstrip("> ")):
            flush()
        stripped = _strip_markdown_noise(raw_line)
        if not stripped:
            flush()
            continue
        paragraph_lines.append((idx, stripped))
    flush()
    return out


def _line_offsets(text: str) -> list[int]:
    """0-based char offset where each 1-indexed line starts (index 0 unused)."""
    offs = [0, 0]
    for ln in text.splitlines(keepends=True):
        offs.append(offs[-1] + len(ln))
    return offs


def _relocate(text: str, sentence: str, search_from: int = 0) -> tuple[int, int]:
    """Best-effort char span of ``sentence`` in ``text``, whitespace-flexible so the
    markdown-stripped, paragraph-joined sentence still matches the original. Returns
    ``(-1, -1)`` when it cannot be located."""
    words = sentence.split()
    if not words:
        return (-1, -1)
    pat = re.compile(r"\s+".join(re.escape(w) for w in words))
    m = pat.search(text, max(search_from - 1, 0)) or pat.search(text)
    return (m.start(), m.end()) if m else (-1, -1)


def extract_claims(document_text: str) -> list[ExtractedClaim]:
    """Extract candidate claims from document text.

    Heuristic: split into sentences, drop fragments that lack verb-shaped
    content, assign stable IDs in order of appearance. Each claim carries its
    char span in the document (relocated whitespace-flexibly; ``-1`` if unfound).
    Warns when the document is not English - see :func:`warn_if_not_english`.
    """
    sentences = _split_document(document_text)
    warn_if_not_english([s for _, s in sentences])
    candidates: list[tuple[int, str]] = [
        (line_no, s) for line_no, s in sentences if _looks_like_claim(s)
    ]

    offs = _line_offsets(document_text)
    pad = max(2, len(str(len(candidates))))
    out: list[ExtractedClaim] = []
    for i, (line_no, claim_text) in enumerate(candidates, start=1):
        anchor = offs[line_no] if 0 < line_no < len(offs) else 0
        cs, ce = _relocate(document_text, claim_text, anchor)
        out.append(
            ExtractedClaim(
                id=f"c{i:0{pad}d}",
                claim=claim_text,
                line_number=line_no,
                char_start=cs,
                char_end=ce,
                out_of_scope_reason=out_of_scope(claim_text),
            )
        )
    return out


def extract_claims_from_file(path: str | Path) -> list[ExtractedClaim]:
    """Read ``path`` (UTF-8 strict) and extract claims.

    Raises ``UnicodeDecodeError`` on non-UTF-8 input so the caller can
    surface a clear error message (the CLI layer handles this).
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="strict")
    return extract_claims(text)

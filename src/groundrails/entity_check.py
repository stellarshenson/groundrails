"""Numeric and named-entity contradiction detection between claim and passage.

Used by :func:`grounding.ground` when some layer has a positive signal. If a
number or named entity appears in the claim with a differing value in the
matched passage (but same category / context word), we record a mismatch.

Design points:
    - Regex-only. No ML. Deterministic.
    - Compares claim tokens against the winning passage only. The full source
      is not scanned because we only flag direct disagreement between claim
      and its own best evidence.
    - Tolerates unit formatting variations ("1,000" == "1000", "10 %" == "10%").
    - Tech-entity whitelist catches common "H100 vs A100" style contradictions.

Public API:
    :func:`extract_numbers` -> list of (value, unit, context_word)
    :func:`extract_entities` -> list of named-entity strings
    :func:`find_mismatches` -> (numeric_mismatches, entity_mismatches)

Each mismatch is ``(claim_value, passage_value)`` suitable for
``GroundingMatch.numeric_mismatches`` / ``entity_mismatches``.
"""

from __future__ import annotations

import re

# ---- numeric extraction --------------------------------------------------

_NUMBER_RE = re.compile(
    r"""
    (?:(?<![\w.,)])(?P<sign>-))?          # signed value (-36%); a hyphen inside
                                          # a range (2010-2015) is preceded by a
                                          # word char and is NOT a sign
    (?P<value>
        \d{1,3}(?:,\d{3})+(?:\.\d+)?      # 1,234 / 1,234,567.89
        |
        \d+(?:\.\d+)?                     # 42 / 3.14
    )
    \s*
    (?P<unit>
        %                                  # percentages (no letter follows)
        |
        (?:
            percent|seconds|hours|         # word units
            sec|min|lbs|USD|EUR|           # 3-letter (before their prefixes)
            GB|MB|KB|TB|                   # storage
            ms|px|em|rem|pt|               # time / typography
            kg|km|cm|mm|                   # mass / distance
            k|K|m|M|b|B|g|s|h              # 1-letter SI/time - LAST, or they
                                           # shadow every longer unit above
        )(?![A-Za-z])                      # word-boundary: "5 meters" is not
                                           # unit "m", "5 sec" is not unit "s"
        )?
    """,
    re.VERBOSE,
)

# Context word: a noun like "nodes", "users", "GPUs", "SD" near the number.
_CONTEXT_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")

# Function words that are never a meaningful numeric context (a number followed
# by one of these has no real "context noun"). Dropping them lets year/category
# detection key the number correctly instead of latching onto e.g. "and".
_STOPWORDS = frozenset(
    {
        "and",
        "or",
        "but",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "were",
        "are",
        "be",
        "been",
        "that",
        "this",
        "these",
        "those",
        "then",
        "than",
        "per",
    }
)

# Dates - cover historical years (1500-2099), not just 19xx/20xx, so a claim
# like "built in 1650" is recognised as a year and can be compared against a
# source year (e.g. 1820). Without this, same-category years get inconsistent
# keys and a real contradiction is missed.
_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")

# Comparative / approximate quantifiers in front of a number. A number qualified
# by one of these is a bound or estimate, not an exact value, so it must NOT be
# treated as an exact contradiction (e.g. claim "over 5000" vs evidence "512" is
# under-determined, not a contradiction). Without this the numeric guard floods
# false contradictions on real comparative/threshold claims. The word quantifiers
# start at a word boundary: "over" inside "cover" is not a quantifier (DEF-NUMBER-47).
_COMPARATIVE_RE = re.compile(
    r"(?:\b(?:more than|greater than|over|above|at least|at most|no more than|no fewer than|"
    r"less than|fewer than|under|below|up to|nearly|almost|about|approximately|around|"
    r"roughly)|>=|<=|>|<|≥|≤|~)\s*"
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# A date after "before" / "after" is a bound as well: "before March 19, 2020" is
# not contradicted by "as of 18 March 2020" (DEF-NUMBER-42). Dates only - "after 5
# years" is a duration and stays exact.
_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b\.?"
_DATE_BOUND_RE = re.compile(
    rf"\b(?:before|after|prior\s+to|earlier\s+than|later\s+than)\s+(?:the\s+)?"
    rf"(?P<date>{_MONTH}\s+\d{{1,2}}(?:\s*,\s*\d{{4}})?|\d{{1,2}}\s+{_MONTH}(?:\s+\d{{4}})?"
    rf"|{_MONTH}\s+\d{{4}}|(?:1[5-9]|20)\d{{2}})\b",
    re.IGNORECASE,
)

# A bracketed citation marker ("[1]", "[2, 3]", "[4-6]") is a reference number,
# not a stated value. Read as one, "rose 12% in 2023 [1]" contradicted a source
# stating 12% - "1" against "12" under the same key (DEF-NUMBER-30).
_CITATION_RE = re.compile(r"\[\d+(?:\s*[,\u2013-]\s*\d+)*\]")


# One value written with a separator inside it: a date ("2024-09-25",
# "06.07.2026"), a range ("5-12 years", "20-30%", "5 to 12", "between 5 and 12")
# or a clock time ("06:00"). ``extract_numbers`` yields every digit run
# separately, so without this a document quoting a range reports its two
# endpoints as a divergence (DEF-SELF-17), and the grounding tier compared a
# range endpoint as an exact value (DEF-NUMBER-22). Dates are tried first so a
# range never takes two components of one. A range end keeps its minus sign under
# the rule ``_NUMBER_RE`` uses, or "-5 to 10" read as -5 plus 5-10 (DEF-NUMBER-48).
_VALUE = r"\d{1,2}:\d{2}(?::\d{2})?|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_END = rf"(?:(?<![\w.,)])-)?(?:{_VALUE})"
_COMPOUND_RE = re.compile(
    rf"""
    (?<![\w.:])(?<!\d-)(?<!\d\u2013)
    (?:
        (?P<d>\d{{4}}-\d{{1,2}}-\d{{1,2}}|\d{{1,2}}[./]\d{{1,2}}[./]\d{{4}})
        |
        between\s+(?P<ba>{_END})\s+and\s+(?P<bb>{_END})
        |
        (?P<a>{_END})\s*(?:-|\u2013|\u2014|\bto\b)\s*(?P<b>{_END})
        |
        (?P<t>\d{{1,2}}:\d{{2}}(?::\d{{2}})?)
    )
    (?![\w:]*\d)(?!\s*[-\u2013]\s*\d)
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _compound_value(m: re.Match) -> str:
    """The rendered value of one ``_COMPOUND_RE`` match: the date, ``a-b`` or the time."""
    if m.group("d") or m.group("t"):
        return m.group("d") or m.group("t")
    a, b = (m.group("ba"), m.group("bb")) if m.group("ba") else (m.group("a"), m.group("b"))
    return f"{a}-{b}"


_DIGITS_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def _compound_part_values(text: str) -> set[str]:
    """Normalised values of every number inside a date, range or clock time.

    Such a number is one end or one field of a larger value, never an exact
    quantity on its own, so it cannot pin a contradiction (DEF-NUMBER-22).
    """
    return {
        _normalise_value(d)
        for m in _COMPOUND_RE.finditer(text)
        for d in _DIGITS_RE.findall(m.group(0))
    }


def _decimals(value: str) -> int:
    return len(value.split(".", 1)[1]) if "." in value else 0


def _agree_at_coarser_precision(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` are the same quantity at the coarser of their two
    precisions: a source that rounds 66.76 to 67 restates the value, it does not
    contradict it (DEF-NUMBER-22)."""
    from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

    try:
        da, db = Decimal(a), Decimal(b)
    except InvalidOperation:
        return a == b
    q = Decimal(1).scaleb(-min(_decimals(a), _decimals(b)))
    try:
        return da.quantize(q, rounding=ROUND_HALF_UP) == db.quantize(q, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        # quantize raises past the 28-digit context precision (a build number or a
        # hash in a source): such a value agrees only when it is equal (DEF-NUMBER-43)
        return da == db


def _comparative_values(text: str) -> set[str]:
    """Normalised values that appear with a comparative/approximate quantifier."""
    return {_normalise_value(m.group(1)) for m in _COMPARATIVE_RE.finditer(text)} | {
        _normalise_value(d)
        for m in _DATE_BOUND_RE.finditer(text)
        for d in _DIGITS_RE.findall(m.group("date"))
    }


def _normalise_value(raw: str) -> str:
    """Strip thousands separators and trailing .0 for canonical comparison."""
    v = raw.replace(",", "")
    if "." in v:
        try:
            f = float(v)
            if f == int(f):
                return str(int(f))
            return f"{f:g}"
        except (ValueError, OverflowError):
            # OverflowError: float() gives infinity above the float range, about 1.8e308 (DEF-NUMBER-49)
            return v
    return v


def _normalise_unit(raw: str | None) -> str:
    if not raw:
        return ""
    u = raw.strip().lower()
    if u == "percent":
        return "%"
    return u


def extract_numbers(text: str) -> list[tuple[str, str, str]]:
    """Return list of ``(value, unit, context_word)`` triples.

    Captures an optional sign and unit, plus a context word: the first
    content word (skipping stopwords) within the same sentence after the
    number, falling back to the nearest preceding content word (e.g.
    "42 nodes" -> ``("42", "", "nodes")``; "sums to 100%" ->
    ``("100", "%", "sums")``). ``context_word`` is lowercased; ``value``
    and ``unit`` preserve normalisation.
    """
    return [(v, u, c) for v, u, c, _ in _numbers_with_spans(text)]


def _numbers_with_spans(text: str) -> list[tuple[str, str, str, int]]:
    """:func:`extract_numbers` plus each number's start offset in ``text``.

    Same numbers, same order; the offset lets a caller tell which numbers sit
    inside a larger span (the consistency check merges a range's endpoints).
    """
    out: list[tuple[str, str, str, int]] = []
    if not text:
        return out
    for m in _NUMBER_RE.finditer(text):
        value = _normalise_value((m.group("sign") or "") + m.group("value"))
        unit = _normalise_unit(m.group("unit"))
        # Context word: first content word (skipping stopwords) within ~40
        # chars AFTER the number, falling back to the nearest content word
        # BEFORE it. The look-behind matters for trailing-position numbers
        # ("sums to 100%") whose only descriptor precedes them - without it
        # they key context-free and collide with every other same-unit number.
        context_word = ""
        # Clip both context windows at sentence punctuation: "increased by 40.
        # Managers said..." must not key 40 on "managers" from the NEXT
        # sentence (divergent keys would silently skip the comparison).
        tail = re.split(r"[.!?;]", text[m.end() : m.end() + 40], maxsplit=1)[0]
        following = _CONTEXT_WORD_RE.findall(tail)[:3]
        # The IMMEDIATE following content word wins ("1820 nodes" keys on
        # "nodes"); a year-shaped bare number whose immediate neighbour is a
        # stopword stays a year ("1820 and restored" must not key on
        # "restored" two words later). Only then scan further/backwards.
        if following and following[0].lower() not in _STOPWORDS:
            context_word = following[0].lower()
        elif not unit and re.fullmatch(r"1[5-9]\d{2}|20\d{2}", value):
            context_word = "year"
        else:
            for w in following[1:]:
                if w.lower() not in _STOPWORDS:
                    context_word = w.lower()
                    break
            if not context_word:
                # Preceding-word fallback: trailing-position numbers
                # ("sums to 100%") whose only descriptor precedes them
                # must not key context-free and collide with every other
                # same-unit number. Clipped at sentence punctuation so the
                # fallback never reaches into the PREVIOUS sentence.
                head = re.split(r"[.!?;]", text[max(0, m.start() - 40) : m.start()])[-1]
                for w in reversed(_CONTEXT_WORD_RE.findall(head)):
                    if w.lower() not in _STOPWORDS:
                        context_word = w.lower()
                        break
        # Filter noise: single-digit years-like tokens without unit or context are uninformative
        if not unit and not context_word and len(value) <= 1:
            continue
        out.append((value, unit, context_word, m.start()))

    # Also pick up standalone 4-digit years for date-style contradictions
    for m in _YEAR_RE.finditer(text):
        year = m.group(0)
        # Skip if already captured as a numeric (would be duplicate) by checking overlap
        already = any(v == year for v, _, _, _ in out)
        if not already:
            out.append((year, "", "year", m.start()))
    return out


# ---- named-entity extraction --------------------------------------------

# Tech-entity whitelist: common hardware / model / framework strings. A
# contradiction between items from the SAME list is a high-signal mismatch.
_TECH_ENTITY_CLASSES: dict[str, list[str]] = {
    "nvidia_gpu": [
        "H100",
        "A100",
        "V100",
        "P100",
        "K80",
        "T4",
        "L4",
        "L40",
        "H200",
        "B100",
        "A10",
    ],
    "amd_gpu": ["MI250", "MI300", "MI300X", "MI100", "MI210"],
    "apple_soc": ["M1", "M2", "M3", "M4", "M1 Pro", "M2 Pro", "M3 Pro"],
    "llm_model": [
        "GPT-3",
        "GPT-3.5",
        "GPT-4",
        "GPT-4o",
        "GPT-5",
        "Claude",
        "Claude 2",
        "Claude 3",
        "Claude 3.5",
        "Llama",
        "Llama 2",
        "Llama 3",
        "PaLM",
        "PaLM 2",
        "Gemini",
        "Mistral",
        "Mixtral",
    ],
    "deep_learning_framework": ["PyTorch", "TensorFlow", "JAX", "MXNet", "Keras"],
    "cloud": ["AWS", "Azure", "GCP", "OCI"],
    "database": ["PostgreSQL", "MySQL", "MongoDB", "Cassandra", "DynamoDB", "SQLite", "Redis"],
}


def _find_tech_entities(text: str) -> dict[str, list[str]]:
    """Return ``{category: [matches]}`` for tech entities found in text.

    Matching is case-sensitive for acronyms but tolerates surrounding
    punctuation.
    """
    out: dict[str, list[str]] = {}
    for category, values in _TECH_ENTITY_CLASSES.items():
        hits = []
        for val in values:
            # Word-boundary-ish match (avoid matching "V1000" when looking for "V100")
            pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(val) + r"(?![A-Za-z0-9])")
            if pattern.search(text):
                hits.append(val)
        if hits:
            out[category] = hits
    return out


_CAPITALISED_PHRASE_RE = re.compile(
    # Single token: starts with uppercase, may contain lowercase, digits, and
    # internal hyphens (e.g. "RoPE", "RoPE-Mid", "GPT-4", "Llama-3.1"). Then
    # optionally up to 3 additional whitespace-separated capitalised tokens
    # for multi-word proper nouns ("New York Times", "Stanford Natural
    # Language Processing Group").
    r"\b[A-Z][a-zA-Z0-9]*(?:-[A-Za-z0-9]+)*(?:\s+[A-Z][a-zA-Z0-9]*(?:-[A-Za-z0-9]+)*){0,3}\b"
)

_STOPWORD_CAPS = {
    "The",
    "This",
    "That",
    "These",
    "Those",
    "A",
    "An",
    "And",
    "Or",
    "But",
    "If",
    "When",
    "Where",
    "While",
    "I",
    "We",
    "You",
    "They",
    "He",
    "She",
    "It",
    "Our",
    "Their",
    "My",
    "Your",
    "His",
    "Her",
}


def extract_entities(text: str) -> list[str]:
    """Return capitalised multi-word named-entity candidates.

    Uses a heuristic: two or more capitalised tokens in a row, excluding the
    first token of the text if it begins a sentence with a stopword like
    "The". Deduplicated while preserving order.
    """
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _CAPITALISED_PHRASE_RE.finditer(text):
        phrase = m.group(0).strip()
        # Drop single-word stopword-only matches
        parts = phrase.split()
        if len(parts) == 1 and parts[0] in _STOPWORD_CAPS:
            continue
        # Drop "plain" single-word matches (risk of false positives on any
        # sentence-initial capitalised word). Keep distinctive single-word
        # proper-noun forms:
        #   - contains a digit           (H100, GPT-4, Llama-3)
        #   - contains a hyphen          (RoPE-Mid, Claude-3)
        #   - camelCase / mixed case     (iPhone, RoPE, MobileBert)
        # A plain word like "Simply", "Where", "Models" does not qualify.
        if len(parts) == 1:
            tok = parts[0]
            if re.search(r"\d", tok) or "-" in tok or any(c.isupper() for c in tok[1:]):
                pass
            else:
                continue
        if phrase in seen:
            continue
        seen.add(phrase)
        out.append(phrase)
    return out


# ---- mismatch detection --------------------------------------------------


def find_numeric_mismatches(claim: str, passage: str) -> list[tuple[str, str]]:
    """Return ``[(claim_num, passage_num)]`` for disagreeing numbers.

    A disagreement requires both sides to share either the same unit
    (``"%"``, ``"GB"``) or the same context word (``"nodes"``, ``"users"``)
    AND have different values.

    Iter 6 specificity gate (mirrors ``find_entity_mismatches``): the claim
    must have EXACTLY ONE number in a given (unit, context) category
    before a mismatch is flagged. Multi-value lists (e.g. "Llama3-8B,
    Llama3-70B, Mistral-7B, Mixtral-8x22B" - four numbers in the
    ``b`` billion-params category) are NOT flagged because the winning
    passage may legitimately cite only a subset; an inventory-style
    claim that overlaps with part of the source is supported, not
    contradicted. The overlap check also skips when any claim value
    already appears among the passage values for the same key.
    """
    claim = _CITATION_RE.sub(" ", claim)
    stated = {(v, u) for v, u, _ in extract_numbers(passage)}
    passage = _CITATION_RE.sub(" ", passage)
    claim_nums = extract_numbers(claim)
    if not claim_nums:
        return []
    passage_nums = extract_numbers(passage)
    if not passage_nums:
        return []

    # Build index of passage numbers by (unit, context_word)
    pass_by_key: dict[tuple[str, str], list[str]] = {}
    for v, u, cw in passage_nums:
        key_full = (u, cw)
        pass_by_key.setdefault(key_full, []).append(v)
        # Partial just-unit key ONLY for numbers with no context of their own:
        # the bare-unit bucket means "underspecified quantity of this unit".
        # Pooling context-bearing numbers into it made every same-unit pair
        # comparable - "sums to 100%" vs "-36% of a SD" collided on ('%','')
        # and tripped a false CONTRADICTED. Distinct measured quantities must
        # not meet through the bare-unit bucket.
        if u and not cw:
            pass_by_key.setdefault((u, ""), []).append(v)
        if cw:
            pass_by_key.setdefault(("", cw), []).append(v)

    # Group claim numbers by the same partial-key lookup used for passage,
    # so we can detect multi-entry claim categories and skip them.
    claim_by_key: dict[tuple[str, str], list[str]] = {}
    for cv, cu, ccw in claim_nums:
        for key in [(cu, ccw), (cu, ""), ("", ccw)]:
            if key == ("", ""):
                continue
            if key in pass_by_key:
                claim_by_key.setdefault(key, []).append(cv)
                break

    # Comparative/approximate values and the parts of a range, date or time on
    # either side are bounds, not exact numbers - they cannot form an exact
    # contradiction.
    claim_comp = _comparative_values(claim) | _compound_part_values(claim)
    pass_comp = _comparative_values(passage) | _compound_part_values(passage)

    # A claim value the passage states with the same unit, or with no unit on
    # either side, is restated, whatever context word each side resolved: in
    # "recall 80.45 percent, precision 80.31 percent" the following-word rule
    # keys 80.45 on "precision", so a keyed comparison alone reported a verbatim
    # claim as contradicted (DEF-NUMBER-22). The stated set reads the passage
    # before citation markers are removed, so a value the source prints inside
    # brackets (a markdown link text, an editorial bracket) still counts as stated.
    claim_units = {v: u for v, u, _ in claim_nums}
    mismatches: list[tuple[str, str]] = []
    for key, claim_values in claim_by_key.items():
        # Specificity gate: multi-value lists aren't contradicted by partial
        # passage coverage.
        if len(claim_values) != 1:
            continue
        passage_values = pass_by_key.get(key, [])
        if not passage_values:
            continue
        cv = claim_values[0]
        if any(u == claim_units[cv] and _agree_at_coarser_precision(cv, v) for v, u in stated):
            continue
        # Comparative claim value (e.g. "over 5000") is a bound, not exact.
        if cv in claim_comp:
            continue
        # Overlap check: any claim value in passage_values means supported, and
        # so does a passage value that states it at a coarser precision.
        if any(_agree_at_coarser_precision(cv, pv) for pv in passage_values):
            continue
        # Contradict only against an EXACT passage value that differs; a
        # comparative passage value ("more than 512") doesn't pin a contradiction.
        pv = next((v for v in passage_values if v not in pass_comp), None)
        if pv is None:
            continue
        mismatches.append((cv, pv))
    return mismatches


def find_entity_mismatches(claim: str, passage: str) -> list[tuple[str, str]]:
    """Return ``[(claim_entity, passage_entity)]`` for tech-category disagreements.

    Only tech entities from the whitelist are checked. A mismatch requires:

    1. Claim and passage share a tech category (e.g. ``gpu_model``).
    2. Claim has EXACTLY ONE entity in that category. Multi-entity lists
       like "we tested GPT-4o, Claude-3.5-Sonnet, and Llama3-70B" are NOT
       flagged because the winning passage might mention only a subset;
       the claim is not "contradicted" by the passage citing a different
       subset of models.
    3. The single claim entity does NOT appear among the passage's
       entities in the same category. Any overlap (even partial) is
       treated as support, not contradiction.

    This catches the "H100 vs A100" / "42 nodes vs 12 nodes" single-value
    fabrication class while allowing list-compatible subsets. The rule was
    tightened in Iter 6 after cross-validation found the old per-item loop
    false-flagging real paraphrases (Ye y07: "ChatGPT, GPT-4o,
    Claude-3.5-Sonnet" vs passage naming Llama3 models in the same
    judge-models category).
    """
    claim_tech = _find_tech_entities(claim)
    if not claim_tech:
        return []
    passage_tech = _find_tech_entities(passage)
    if not passage_tech:
        return []

    mismatches: list[tuple[str, str]] = []
    for category, claim_items in claim_tech.items():
        if category not in passage_tech:
            continue
        passage_items = passage_tech[category]
        # Specificity gate: multi-entity claims are lists, not assertions.
        if len(claim_items) != 1:
            continue
        (ci,) = claim_items
        # Overlap check: if the single claim entity already appears in
        # passage, it's confirmed, not contradicted.
        if ci in passage_items:
            continue
        # One specific claim entity, passage has different entity(ies)
        # in the same category -> real contradiction.
        mismatches.append((ci, passage_items[0]))
    return mismatches


def list_claim_entities(claim: str) -> list[str]:
    """Union of tech-whitelist entities + capitalised proper-noun phrases in ``claim``.

    Deduplicated, preserves insertion order. Used as the denominator for
    entity-presence penalty calculations.
    """
    flat: list[str] = []
    for items in _find_tech_entities(claim).values():
        flat.extend(items)
    for phrase in extract_entities(claim):
        if phrase not in flat:
            flat.append(phrase)
    return flat


def find_absent_entities(claim: str, full_source: str) -> list[str]:
    """Return claim entities that appear nowhere in the full source text.

    This is a weaker signal than :func:`find_entity_mismatches` (which
    requires the source to mention the SAME category with a DIFFERENT
    value). ``find_absent_entities`` catches the "unsupported claim
    entity" pattern: a proper noun named in the claim with zero string
    occurrences in the source. Examples:

        claim "RoPE-Mid fixes middle-of-context degradation", source
        is the Liu 2023 paper that never names "RoPE-Mid"
        → ["RoPE-Mid"]

        claim "experiments on H100 donated by Meta", source mentions
        no "H100" or "Meta"
        → ["H100", "Meta"]

    Comparison is case-insensitive substring. This catches the
    fabricated-specific-entity failure mode where the claim scores high
    lexically+semantically via topical overlap but the distinguishing
    entity is invented.
    """
    flat = list_claim_entities(claim)
    if not flat:
        return []
    source_lower = full_source.lower()
    absent: list[str] = []
    for e in flat:
        if e.lower() not in source_lower:
            absent.append(e)
    return absent


def find_mismatches(
    claim: str, passage: str
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Find all numeric + entity mismatches between claim and passage."""
    return (
        find_numeric_mismatches(claim, passage),
        find_entity_mismatches(claim, passage),
    )


__all__ = [
    "extract_entities",
    "extract_numbers",
    "find_absent_entities",
    "find_entity_mismatches",
    "find_mismatches",
    "find_numeric_mismatches",
    "list_claim_entities",
]

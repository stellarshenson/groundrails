"""R20-H177 NUMERIC-VERIFICATION PORTFOLIO ARM - shared lane machinery, CPU only.

Registered in docs/experiments/semantic-grounding-experiments.md, block "R20-H177
NUMERIC-VERIFICATION PORTFOLIO ARM (compare/direction + operand-role misbind)",
stage 0.  Two lanes sit on top of this module - `R20-H177_lane_B.py`
(`num_compare`, compare/direction verification) and `R20-H177_lane_C.py`
(`num_rolebind`, operand-role / sign / period misbind).

WHAT LIVES HERE
---------------
  * the SOURCE WALL, enforced in code rather than asserted.  The admitted supply
    is exactly two corpora - the banked EDGAR-restricted admitted slice
    (`R18-H150_edgar_admitted.parquet`, 34,014 chunks over 4,297 filings) and
    public TabFact TRAIN tables read through `R17-H144_pairs.py`.  FinQA and
    TAT-QA source corpora are WALLED and never opened; HotpotQA train and HoVer
    are never opened; FEVEROUS is admitted by the registration only if a banked
    supply exists, and `feverous_available()` reports what is on disk
  * the EDGAR PROSE EXTRACTORS both lanes share - role/value bindings, stated
    changes ("an increase of $X"), year-bound values, and from-to trends.  Every
    extractor returns the VERBATIM surface as it is printed in the chunk, so
    "the value is present in the evidence" is true by construction and is
    re-measured in each lane's verify block rather than assumed
  * the ATTESTATION and UNIQUENESS instruments the leak suite rests on.  A
    negative twin is a word-level corruption of a true statement, so the
    corrupted word must itself be readable in the evidence (else the negative is
    detectable by lexical novelty, not by binding), and the asserted value must
    occur exactly once in the chunk (else the corrupted binding might be
    attested somewhere else in the passage and the label would be wrong)
  * the deterministic DOCUMENT SPLIT that keeps the held-out mechanism evals
    doc-disjoint from the training lanes

The verification instruments themselves (converged claim-only probe, within-pair
accuracy, surface parity, pair integrity, AUROC, window census) are IMPORTED from
`R20-H174_lane_common.py` - the banked lane discipline, not a second copy of it.

Nothing here writes an artifact.  Run the per-lane builders.
"""

import hashlib
import importlib.util as _ilu
from pathlib import Path
import re

import polars as pl

HERE = Path(__file__).parent
ROOT = HERE.parent.parent

_spec = _ilu.spec_from_file_location("h174common", HERE / "R20-H174_lane_common.py")
H174 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(H174)

# re-exported banked instruments (one definition, imported not copied)
tokens = H174.tokens
containment = H174.containment
auroc = H174.auroc
claim_only_probe = H174.claim_only_probe
within_pair_accuracy = H174.within_pair_accuracy
surface_parity = H174.surface_parity
pair_integrity = H174.pair_integrity
dedupe = H174.dedupe
char_stats = H174.char_stats
window_census = H174.window_census

EDGAR = HERE / "R18-H150_edgar_admitted.parquet"
FEVEROUS_SRC = ROOT / "tmp" / "R14_H133_feverous.parquet"

SOURCES = {
    "edgar": {
        "dataset": "EDGAR restricted 10-K/10-Q MD&A slice, admitted",
        "artifact": "R18-H150_edgar_admitted.parquet (34,014 chunks, 4,297 filings)",
        "licence": "Apache-2.0 pipeline over public EDGAR filings; "
                   "data/external/datasets/dataset-edgar-restricted.md - raw EDGAR "
                   "outside the restricted slice stays banned",
        "wall": "provenance gate GREEN on admission (R18-H150); this lane re-runs "
                "the R14-H136 8-gram census on the built pairings",
    },
    "tabfact": {
        "dataset": "TabFact TRAIN tables (public)",
        "artifact": "data/external/datasets/dataset-tabfact.zip, read through "
                    "R17-H144_pairs.tabfact_tables()",
        "licence": "CC-BY-4.0 (banked supply, rode in R14-H133 / R17-H144 / "
                   "R17-H146)",
        "wall": "banked GREEN; the lane census re-runs the wall on the new "
                "serialisations",
    },
}

WALLED_NEVER_OPENED = [
    "finqa (RAGBench arena parent) - source corpus WALLED, not read by this lane",
    "tatqa (RAGBench arena parent) - source corpus WALLED, not read by this lane",
    "hotpotqa train - never opened",
    "hover - never opened",
]

# --------------------------------------------------------------------------- #
# document split - held-out mechanism evals are DOC-DISJOINT by construction
# --------------------------------------------------------------------------- #
EVAL_DOC_PERMILLE = 120          # 12% of source documents are eval-only


def is_eval_doc(doc_id):
    """Deterministic, corpus-independent, stable across runs and across lanes."""
    h = hashlib.blake2b(str(doc_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % 1000 < EVAL_DOC_PERMILLE


# --------------------------------------------------------------------------- #
# numerals
# --------------------------------------------------------------------------- #
MAG = r"(?:\s+(?:million|billion|thousand))?"
AMOUNT = rf"\$\s?\d[\d,]*(?:\.\d+)?{MAG}"
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_SCALE = {"million": 1e6, "billion": 1e9, "thousand": 1e3}
_AMOUNT_PARSE = re.compile(
    r"\$\s?(\d[\d,]*(?:\.\d+)?)(?:\s+(million|billion|thousand))?", re.I)


def parse_amount(surface):
    """Scaled numeric value of a printed amount - '$1.6 million' -> 1.6e6.

    Scale words are load-bearing: '$1.6 million' is larger than '$952,000' and a
    raw 1.6-vs-952 comparison would invert the ordering the lane teaches."""
    m = _AMOUNT_PARSE.match(surface.strip())
    if not m:
        return None
    try:
        v = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    return v * _SCALE.get((m.group(2) or "").lower(), 1.0)


def occurrences(chunk, surface):
    """Verbatim occurrences of a printed amount in the evidence.

    The leading '$' anchors the count: '$1.0 million' is not a substring of
    '$21.0 million', so an amount that occurs once really is bound once."""
    return chunk.count(surface)


def word_attested(chunk, word):
    """Is the word readable in the evidence, as a whole word, case-blind?

    The negative twin corrupts ONE word.  If that word appears nowhere in the
    evidence the negative is detectable by lexical novelty rather than by
    binding, which is the leak this check closes."""
    return re.search(rf"\b{re.escape(word.lower())}", chunk.lower()) is not None


def sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


# --------------------------------------------------------------------------- #
# EDGAR prose extraction
#
# A "role" is a labelled quantity as the filing itself names it - "net sales",
# "our income tax expense", "total operating expenses".  The head noun must come
# from the closed vocabulary below and every word of the phrase must be a
# content word, so the extractor cannot walk across a clause boundary and
# manufacture a phrase like "million compared to a loss".
# --------------------------------------------------------------------------- #
ROLE_HEADS = [
    "revenue", "revenues", "sales", "income", "loss", "losses", "earnings",
    "profit", "profits", "margin", "margins", "expense", "expenses", "cost",
    "costs", "assets", "liabilities", "liability", "debt", "borrowings", "cash",
    "equity", "inventory", "inventories", "receivable", "receivables",
    "payable", "payables", "goodwill", "backlog", "dividend", "dividends",
    "interest", "tax", "taxes", "capital", "expenditures", "compensation",
    "impairment", "amortization", "depreciation", "obligations", "payments",
    "reserves", "balance", "proceeds", "investment", "investments",
    "securities", "deposits", "loans", "shares", "contributions", "settlement",
    "penalties", "fee", "fees", "price", "charge", "charges", "benefit",
    "benefits", "provision", "royalties", "premiums", "commissions", "rent",
    "wages", "salaries", "distributions", "repurchases", "writedown",
    "writedowns", "outflows", "inflows", "flow", "flows", "ebitda", "spending",
    "funding", "indebtedness", "notes", "bonds", "leases", "receipts",
]
_HEAD_ALT = "|".join(sorted((re.escape(h) for h in ROLE_HEADS), key=len,
                            reverse=True))

# words that may never appear anywhere inside a role phrase - function words,
# clause hinges and magnitude words all signal the phrase has run off its clause
ROLE_BLOCK = {
    "of", "for", "in", "to", "at", "on", "by", "with", "from", "and", "or",
    "but", "than", "as", "that", "which", "who", "was", "were", "is", "are",
    "be", "been", "being", "has", "have", "had", "compared", "versus", "vs",
    "million", "billion", "thousand", "percent", "approximately", "about",
    "increase", "decrease", "increased", "decreased", "up", "down", "offset",
    "including", "include", "includes", "such", "same", "period", "periods",
    "if", "when", "while", "because", "due", "primarily", "partially", "also",
    "respectively", "above", "below", "over", "under", "per", "a", "an", "the",
    "this", "these", "those", "it", "its", "we", "our", "us", "they", "their",
    "he", "she", "his", "her", "there", "here", "no", "not", "any", "all",
    "each", "both", "either", "neither", "however", "although", "though",
    # comparatives - a role phrase must NAME a quantity, not rank it; "lower
    # insurance revenue" would put a relation word inside the operand label and
    # collide with the relation word lane B flips
    "higher", "lower", "greater", "less", "lesser", "larger", "smaller",
    "highest", "lowest", "largest", "smallest", "more", "fewer", "most",
    "least", "better", "worse", "stronger", "weaker",
}
_LEAD_DROP = {"a", "an", "the", "and", "or", "but", "of", "to", "in", "for",
              "by", "at", "on", "with", "from", "that", "which", "our", "its",
              "their", "this", "these", "those", "his", "her", "no", "not",
              "as", "than", "such", "also", "however", "we", "us", "they", "it"}

ROLE_PAT = re.compile(
    rf"(?P<role>(?:[A-Za-z][A-Za-z\-']*\s+){{0,3}}(?:{_HEAD_ALT}))"
    rf"\s+(?:of|was|were|totaled|totalled|amounted to)\s+(?P<val>{AMOUNT})",
    re.I)

_UP_WORDS = "increase|growth|rise|gain"
_DOWN_WORDS = "decrease|decline|reduction|drop"
CHANGE_PATS = (
    # "an increase of $X" / "a decline of $X" / "a reduction in costs of $X"
    re.compile(rf"\ban?\s+(?P<dir>{_UP_WORDS}|{_DOWN_WORDS})\b"
               rf"(?:\s+in\s+[^.$]{{0,40}}?)?\s+of\s+(?P<val>{AMOUNT})", re.I),
    # "increased by $X" / "declined by $X"
    re.compile(rf"\b(?P<dir>increased|grew|rose|gained|decreased|declined|fell|"
               rf"dropped)\s+by\s+(?P<val>{AMOUNT})", re.I),
    # "a $X increase in ..." - the amount precedes the direction word
    re.compile(rf"\ban?\s+(?P<val>{AMOUNT})\s+(?P<dir>{_UP_WORDS}|{_DOWN_WORDS})\b",
               re.I),
)
_DIR_UP = re.compile(rf"(?i)^(?:{_UP_WORDS}|increased|grew|rose|gained)$")

_INC = r"increased|rose|grew"
_DEC = r"decreased|declined|fell"
TREND_PATS = (
    re.compile(rf"(?P<pre>^.{{2,110}}?)\b(?P<dir>{_INC}|{_DEC})\b"
               rf"(?:\s+by\s+{AMOUNT})?[^.]{{0,40}}?\bfrom\s+(?P<a>{AMOUNT})"
               rf"[^.]{{0,40}}?\bto\s+(?P<b>{AMOUNT})", re.I),
    re.compile(rf"(?P<pre>^.{{2,110}}?)\b(?P<dir>{_INC}|{_DEC})\b[^.]{{0,20}}?"
               rf"\bto\s+(?P<b>{AMOUNT})[^.]{{0,90}}?\bcompared\s+(?:to|with)\s+"
               rf"(?P<a>{AMOUNT})", re.I),
    re.compile(rf"(?P<pre>^.{{2,110}}?)\b(?:was|were)\s+(?P<b>{AMOUNT})"
               rf"[^.]{{0,60}}?,\s+an?\s+(?P<dir>increase|decrease)\s+of\s+"
               rf"{AMOUNT}[^.]{{0,40}}?\bfrom\s+(?P<a>{AMOUNT})", re.I),
)
_IS_UP = re.compile(rf"(?i)^(?:{_INC}|increase)$")


def clean_role(raw):
    """A role phrase, or None if the span is not a clean labelled quantity."""
    words = [w for w in raw.split() if w]
    while words and words[0].lower().strip(",;:") in _LEAD_DROP:
        words = words[1:]
    if not (1 <= len(words) <= 4):
        return None
    low = [w.lower().strip(",;:'\"") for w in words]
    if any(w in ROLE_BLOCK for w in low):
        return None
    if any(any(ch.isdigit() for ch in w) for w in low):
        return None
    if low[-1] not in {h.lower() for h in ROLE_HEADS}:
        return None
    phrase = " ".join(words).strip(" ,;:")
    return phrase if len(phrase) >= 4 else None


def role_bindings(text):
    """[(role, amount_surface)] read out of one passage, in reading order."""
    out = []
    for s in sentences(text):
        for m in ROLE_PAT.finditer(s):
            role = clean_role(m.group("role"))
            if role:
                out.append((role, m.group("val").strip(), s))
    return out


def unique_role_bindings(chunk):
    """Role -> its single amount, for roles bound exactly once in the chunk.

    A role bound to two different amounts inside one passage cannot carry a
    role-swap negative - the "wrong" amount may be its own - so it is dropped."""
    seen = {}
    for role, val, sent in role_bindings(chunk):
        key = role.lower()
        seen.setdefault(key, []).append((role, val, sent))
    out = {}
    for key, hits in seen.items():
        vals = {v for _, v, _ in hits}
        if len(vals) == 1 and len(hits) == 1:
            out[key] = hits[0]
    return out


def _role_before(sentence, upto):
    """The last clean role phrase ending before `upto` in the sentence."""
    best = None
    for m in ROLE_PAT.finditer(sentence):
        if m.start() < upto:
            r = clean_role(m.group("role"))
            if r:
                best = r
    if best:
        return best
    # fall back to a bare head-noun phrase (no connector) before the span
    for m in re.finditer(
            rf"(?P<role>(?:[A-Za-z][A-Za-z\-']*\s+){{0,3}}(?:{_HEAD_ALT}))\b",
            sentence[:upto], re.I):
        r = clean_role(m.group("role"))
        if r:
            best = r
    return best


def change_statements(chunk):
    """[(direction, role, amount_surface, sentence)] - stated changes in words.

    Both the H157 sign exemplars (189 '-$305M' read as '+$305M', 31 and 168
    direction reframes) are of this shape: the passage states the direction of a
    change in WORDS beside its magnitude, so flipping the word is decidable from
    the evidence with no computation."""
    out, seen = [], set()
    for s in sentences(chunk):
        for pat in CHANGE_PATS:
            for m in pat.finditer(s):
                d = "increase" if _DIR_UP.match(m.group("dir")) else "decrease"
                val = m.group("val").strip()
                role = _role_before(s, m.start())
                if not role or (role.lower(), val) in seen:
                    continue
                seen.add((role.lower(), val))
                out.append((d, role, val, s))
    return out


def year_bindings(chunk):
    """[(year, role, amount_surface, sentence)] - values bound to a period."""
    out = []
    for s in sentences(chunk):
        years = [(m.group(0), m.start()) for m in YEAR_RE.finditer(s)]
        if not years:
            continue
        for m in ROLE_PAT.finditer(s):
            role = clean_role(m.group("role"))
            if not role:
                continue
            # the period the value is bound to = the nearest year token
            y = min(years, key=lambda yp: abs(yp[1] - m.start()))[0]
            out.append((y, role, m.group("val").strip(), s))
    return out


def trend_statements(chunk):
    """[(role, direction, old_surface, new_surface, sentence)].

    Only kept when the stated direction AGREES with the ordering of the two
    printed amounts - otherwise the negative twin would not be decidable by
    ordering, which is the whole content of lane B."""
    out = []
    for s in sentences(chunk):
        for pat in TREND_PATS:
            for m in pat.finditer(s):
                a, b = parse_amount(m.group("a")), parse_amount(m.group("b"))
                if a is None or b is None or a == b:
                    continue
                if (b > a) != bool(_IS_UP.match(m.group("dir"))):
                    continue
                role = _role_before(s, m.start("dir")) or clean_role(m.group("pre"))
                if not role:
                    continue
                out.append((role, "increased" if b > a else "decreased",
                            m.group("a").strip(), m.group("b").strip(), s))
    return out


# --------------------------------------------------------------------------- #
# corpora
# --------------------------------------------------------------------------- #
def edgar(split):
    """The admitted EDGAR slice, split into the lane pool and the eval pool."""
    d = pl.read_parquet(EDGAR)
    flag = [is_eval_doc(x) for x in d["doc_id"].to_list()]
    d = d.with_columns(pl.Series("is_eval", flag))
    return d.filter(pl.col("is_eval") if split == "eval" else ~pl.col("is_eval"))


def feverous_available():
    """The registration admits FEVEROUS only if a banked admitted supply exists."""
    return {"path": str(FEVEROUS_SRC.relative_to(ROOT)),
            "present": FEVEROUS_SRC.exists(),
            "admitted": False,
            "reason": "the banked FEVEROUS parquet is an R14-H133 working file, "
                      "not an admitted lane supply with its own provenance "
                      "verdict; lane B is built from TabFact + EDGAR only and "
                      "nothing external is fetched"}


# --------------------------------------------------------------------------- #
# leak-suite instruments specific to the twin construction
# --------------------------------------------------------------------------- #
def claim_only_probe_stratified(claims, labels, docs, strata, rng, n_folds=5):
    """The banked converged claim-only probe with DIRECTION-STRATIFIED folds.

    Document-disjoint folds alone are not enough (R17-H145 finding b): a source
    document usually carries a single relation direction, so unstratified folds
    leave a training complement with no mirrored direction and the probe reads
    BELOW chance - a false clean.  Folds are therefore laid out inside each
    document stratum.  liblinear at tol 1e-7, never default lbfgs (H144 ii)."""
    import collections

    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    doc_stratum = {}
    for d, s in zip(docs, strata):
        doc_stratum.setdefault(d, s)
    buckets = collections.defaultdict(list)
    for d, s in doc_stratum.items():
        buckets[s].append(d)
    fold_of, i = {}, 0
    for s in sorted(buckets):
        ds = sorted(buckets[s])
        rng.shuffle(ds)
        for d in ds:
            fold_of[d] = i % n_folds
            i += 1
    folds = np.array([fold_of[d] for d in docs])
    score = np.zeros(len(claims))
    idx = np.arange(len(claims))
    for f in range(n_folds):
        tr, te = idx[folds != f], idx[folds == f]
        if not te.size or not tr.size:
            continue
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                              max_features=300_000, sublinear_tf=True)
        xtr = vec.fit_transform([claims[j] for j in tr])
        xte = vec.transform([claims[j] for j in te])
        clf = LogisticRegression(solver="liblinear", C=4.0, tol=1e-7, max_iter=3000)
        clf.fit(xtr, [labels[j] for j in tr])
        score[te] = clf.decision_function(xte)
    return float(auroc(labels, score)), score, len(fold_of)


def flip_word_balance(df, word_col="true_word", flip_col="flip_word"):
    """Every relation / direction / role / period word must appear as often on
    the supported leg as on the corrupted one.

    The twin construction gives this exactly when the TRUE word is drawn in
    equal numbers per family - which the builders do with explicit target cells
    - but a construction that is only asserted is not a bar, so it is counted."""
    pos = df.filter(pl.col("label") == 1)
    counts = {}
    for w, f in zip(pos[word_col].to_list(), pos[flip_col].to_list()):
        counts.setdefault(w, [0, 0])[0] += 1        # w used as the true word
        counts.setdefault(f, [0, 0])[1] += 1        # f used as the corrupted word
    worst, rows = 0.0, {}
    for w, (as_true, as_flip) in sorted(counts.items()):
        tot = as_true + as_flip
        share = as_true / tot if tot else 0.5
        rows[w] = {"as_positive": as_true, "as_negative": as_flip,
                   "positive_share": round(share, 4), "pairs": tot}
        worst = max(worst, abs(share - 0.5))
    return {"per_word": rows, "worst_deviation_from_half": round(worst, 4),
            "bar": "every word within 0.02 of a 50/50 positive/negative split",
            "pass": bool(worst <= 0.02)}


def attestation_audit(df, require_attested=()):
    """The corrupted word and the word it replaced must be EQUALLY readable.

    The pair's two claims differ by exactly one word, so the only claim/evidence
    channel that can separate them is whether that word appears in the passage.
    If the corrupted word is novel and the true one is not, the negative is
    detectable by lexical novelty rather than by binding - the leak this closes.
    Symmetry is the bar for every family; the families named in
    `require_attested` carry the stronger bar that BOTH words are present, which
    is what makes a word-level corruption of prose groundable."""
    asym, missing = [], []
    for r in df.filter(pl.col("label") == 0).iter_rows(named=True):
        t = word_attested(r["chunk"], r["true_word"])
        f = word_attested(r["chunk"], r["flip_word"])
        if t != f:
            asym.append({"pair_id": r["pair_id"], "true_word": r["true_word"],
                         "flip_word": r["flip_word"], "true_attested": t,
                         "flip_attested": f})
        if r["neg_family"] in require_attested and not (t and f):
            missing.append({"pair_id": r["pair_id"], "family": r["neg_family"]})
    rates = {}
    for fam, sub in df.filter(pl.col("label") == 0).group_by("neg_family"):
        n = sub.height
        att = sum(word_attested(c, w) for c, w in
                  zip(sub["chunk"].to_list(), sub["flip_word"].to_list()))
        rates[fam[0] if isinstance(fam, tuple) else fam] = {
            "negatives": n, "corrupted_word_attested_rate": round(att / n, 4)}
    return {
        "negatives": int(df.filter(pl.col("label") == 0).height),
        "asymmetric_attestation_rows": len(asym),
        "required_both_attested_families": list(require_attested),
        "rows_missing_required_attestation": len(missing),
        "per_family": rates,
        "bar": "0 asymmetric rows; 0 missing rows on the prose-corruption "
               "families",
        "pass": not (asym or missing),
        "examples": (asym + missing)[:5]}


def value_uniqueness_audit(df, families=()):
    """For the misbind families the asserted amount must be bound ONCE in the
    passage, so the corrupted binding is not attested elsewhere and the label is
    right.  Compare/direction families are exempt: they assert an ORDERING of
    two amounts, not a binding, and repeated printings do not change it."""
    sub = df.filter(pl.col("neg_family").is_in(list(families))) if families else df
    bad = []
    for r in sub.iter_rows(named=True):
        for v in r["asserted_values"].split("|~|"):
            if v and occurrences(r["chunk"], v) != 1:
                bad.append({"pair_id": r["pair_id"], "value": v,
                            "occurrences": occurrences(r["chunk"], v)})
    return {"families": list(families), "rows_checked": int(sub.height),
            "rows_with_non_unique_value": len(bad),
            "bar": "0 - each asserted amount occurs exactly once in its chunk",
            "pass": not bad, "examples": bad[:5]}


def verbatim_value_audit(df, families=()):
    """Every amount the claim asserts is printed in the evidence, verbatim."""
    sub = df.filter(pl.col("neg_family").is_in(list(families))) if families else df
    bad = []
    for r in sub.iter_rows(named=True):
        for v in r["asserted_values"].split("|~|"):
            if v and occurrences(r["chunk"], v) < 1:
                bad.append({"pair_id": r["pair_id"], "value": v})
    return {"rows_checked": int(sub.height), "asserted_values_not_verbatim": len(bad),
            "bar": "0 - every asserted amount is printed in its chunk",
            "pass": not bad, "examples": bad[:5]}

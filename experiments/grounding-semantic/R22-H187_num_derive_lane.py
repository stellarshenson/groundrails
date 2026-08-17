"""R22-H187 NUM_DERIVE LANE - build + verify only, CPU, no GPU, no torch.

Builds `R22-H187_num_derive_lane.parquet`: 30,000 rows = 15,000 minimal pairs
over PUBLIC TabFact-train tables only, seed 2187.

WHY THIS LANE EXISTS.  Both numeric members already in the assembled mix build
their negatives by substituting an OPERAND - `quant_misbind` cites a real cell of
the wrong row or column, `quant_scale_unit` cites the right number under the
wrong unit.  No member's claim ever asserts a result COMPUTED from other stated
numbers, so the model has never been shown a derivation to check.  The R18-H157
finqa autopsy found that 9 of finqa's 20 unsupported responses are exactly that:
every operand correct and present in the evidence, only the computed result
wrong (verbatim: "(\\$30 million / \\$169 million) * 100 = 17.75% ~ 28%").

CONSTRUCTION.  Each pair states the SAME two operand values, both readable
verbatim in the serialized chunk and each bound to a named row and a named
column of the same table:

  positive       the correct difference / percentage / sum / product
  negative twin  the byte-identical claim shape asserting an INCORRECT result

The result is ABSENT from the evidence on BOTH legs by construction, so
attribution is blind to the label by design and the only separator is the
arithmetic.  That is the point of the lane, not a defect in it.

THE CORRUPTION, AND WHY IT IS BUILT THIS WAY.  The binding constraint is C5's
surface-parity band on response-only channels: a separate measurement found a
plain characters-per-sentence counter scores 0.69652 on finqa, above the trained
model itself, so a lane whose wrong results are longer, shorter, rounder or
differently shaped than its right ones would teach that artifact instead of the
arithmetic.  The twin's numeral is therefore obtained by SWAPPING correct results
between two constructions that share a family, a SOURCE TABLE and a digit-surface
signature (decimal places, integer digits, trailing zeros, sign): construction i
asserts j's correct result and construction j asserts i's.  Four consequences,
all intended:

  * an identical signature forces identical rendered length, so every
    response-only channel - characters, words, sentences, numeric tokens,
    numeric density, digit-character fraction, `=` count, derivation markers,
    newlines, characters per sentence - is byte-identical within a pair and
    reads exactly 0.5.  Measured, not assumed
  * the swap makes the multiset of asserted numerals on the negative leg EQUAL
    to the multiset on the positive leg, per signature and pooled.  Any function
    of the asserted value alone is therefore at exactly 0.5, distribution and
    all - not merely in expectation
  * each couple contributes one twin above its true value and one below, so the
    50/50 direction balance is exact rather than quota-enforced
  * a couple sits inside ONE document, so both of a numeral's two appearances -
    once as a positive, once as a negative - always land in the same fold of a
    document-disjoint probe

Two earlier corruption designs were built and measured before this one; both
FAILED and neither is what ships:

  1. twin drawn from the family's pooled result distribution CONDITIONED on the
     wanted direction - claim-only probe 0.5761 against a bar of 0.55.
     Conditioning on a side over-weights the tail of the signature's value range
     relative to the positives, and a character n-gram probe reads distribution
     shape even where the value-magnitude AUROC is exactly 0.5 by symmetry
  2. the swap, but between constructions drawn from ANY two documents -
     claim-only probe 0.1923, a two-sided deviation of 0.308.  Each numeral
     appears exactly twice in the corpus, once per label; when its two documents
     fall in different folds the probe memorises the training occurrence and
     scores the test occurrence with the opposite label.  Confining the couple to
     one document removes the channel rather than hiding it

R17-H144 measured the same family of failure from the other end: a negative drawn
from a tighter band than the positives occupy separates the pair on the numeral
alone (0.600 within-pair against 0.503 for the matched draw).

TABFACT ONLY - never FEVEROUS.  `quant_misbind` is non-conforming on exactly its
FEVEROUS half (C3: the split axis is not measurable for 33.7% of its rows and
their identifier is unstable across rebuilds; C8: no licence, no retrieval date,
no tracked source).  TabFact is archived with a tracked sidecar and its split
axis is MEASURED CLEAN on the archive's own `table_id`.

DISJOINTNESS IS ENFORCED AT BUILD, NOT MEASURED AFTER.  `quant_misbind` reads 69
byte-identical collisions against the evaluation surfaces and that is a recorded
C2 FAIL.  This build rejects a construction outright if its chunk or either
claim matches any of the thirteen evaluation surfaces in ANY of the three string
forms (raw, truncated to 1,500, whitespace-collapsed case-folded), and drops
every TabFact table that an evaluation surface names.

Machinery is IMPORTED, never re-derived: the table loader, the six serializers,
the numeral canonicalisation, the digit-surface helpers and the content-based
eval-set exclusion all come from `R17-H144_pairs.py`; the thirteen evaluation
surfaces come from `contract/quant_misbind_verify.py`.  Neither file is modified.

Run:  CUDA_VISIBLE_DEVICES= uv run python \
        experiments/grounding-semantic/R22-H187_num_derive_lane.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import collections
import hashlib
import importlib.util
import io
import json
import math
import pathlib
import random
import re
import time
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"

OUT = HERE / "R22-H187_num_derive_lane.parquet"
MANIFEST = HERE / "R22-H187_num_derive_lane_manifest.json"

SEED = 2187
N_PAIRS = 15_000                      # 30,000 rows - matches quant_misbind
BODY_ROWS = 6
COUPLE_CAP_LADDER = (1, 2, 3, 4, 6)   # couples (= 2 pairs) taken from one document
DIGIT_LEN_MIN, DIGIT_LEN_MAX = 2, 7
N_FOLDS = 5
CAND_TRIES = 24
PER_TABLE_TRIES = 12                  # constructions attempted per (table, family) visit
MATCH_SCAN = 60                       # partners examined before a candidate is dropped
CHUNK_MAX = 1500

# difference and percentage dominate the finqa failure profile; product and sum
# are secondary.  Declared here, measured in the manifest.
FAMILY_SHARE = {"difference": 0.35, "percentage": 0.35, "sum": 0.15, "product": 0.15}


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _mod("h144pairs", HERE / "R17-H144_pairs.py")
QMV = _mod("qmv", HERE / "contract" / "quant_misbind_verify.py")

NUM_FREE = P.NUM_FREE
SENT = re.compile(r"[.!?]")
MARKER_WORDS = ("so", "therefore", "difference", "product", "sum", "total",
                "percent", "combined", "exceeds", "multiplying", "together",
                "taking", "expressed")


# --------------------------------------------------------------------------- #
# claim templates - three per family; the shape is byte-identical inside a pair
# and the operands are stated with their (row, column) binding on both legs
# --------------------------------------------------------------------------- #
TEMPLATES = {
    "difference": (
        "The {col} of {ka} is {va} and the {col} of {kb} is {vb}, so the {col} "
        "of {ka} exceeds that of {kb} by {r}.",
        "Taking the {col} of {ka} ({va}) and the {col} of {kb} ({vb}), "
        "{va} - {vb} = {r}.",
        "According to the table, the {col} of {ka} is {va} and the {col} of "
        "{kb} is {vb}; the difference between them is {r}.",
    ),
    "percentage": (
        "The {col} of {ka} is {va} and the {col} of {kb} is {vb}, so the {col} "
        "of {ka} is {r} percent of the {col} of {kb}.",
        "Taking the {col} of {ka} ({va}) and the {col} of {kb} ({vb}), "
        "({va} / {vb}) * 100 = {r} percent.",
        "According to the table, the {col} of {ka} is {va} and the {col} of "
        "{kb} is {vb}; expressed as a percentage of {kb}, {ka} is at {r} percent.",
    ),
    "sum": (
        "The {col} of {ka} is {va} and the {col} of {kb} is {vb}, so their "
        "combined {col} is {r}.",
        "Taking the {col} of {ka} ({va}) and the {col} of {kb} ({vb}), "
        "{va} + {vb} = {r}.",
        "According to the table, the {col} of {ka} is {va} and the {col} of "
        "{kb} is {vb}; together they total {r}.",
    ),
    "product": (
        "The {col} of {ka} is {va} and the {col} of {kb} is {vb}, so the "
        "product of the two is {r}.",
        "Taking the {col} of {ka} ({va}) and the {col} of {kb} ({vb}), "
        "{va} * {vb} = {r}.",
        "According to the table, the {col} of {ka} is {va} and the {col} of "
        "{kb} is {vb}; multiplying them gives {r}.",
    ),
}


def render(fam, ti, col, ka, va, kb, vb, r):
    return TEMPLATES[fam][ti].format(col=col, ka=ka, va=va, kb=kb, vb=vb, r=r)


def render_claim(c, r):
    return render(c["neg_family"], c["template_id"], c["column"], c["row_key_a"],
                  c["operand_a_str"], c["row_key_b"], c["operand_b_str"], r)


# --------------------------------------------------------------------------- #
# arithmetic + digit surface
# --------------------------------------------------------------------------- #
def compute(fam, a, b):
    """The lane's arithmetic checker - the instrument C1's diagnostic uses."""
    if fam == "difference":
        return a - b
    if fam == "sum":
        return a + b
    if fam == "product":
        return a * b
    if fam == "percentage":
        return None if abs(b) < 1e-9 else a / b * 100.0
    raise ValueError(fam)


def surface_key(s):
    """(decimal places, integer digits, trailing zeros, negative).

    Equal keys force equal rendered LENGTH, equal digit count, equal magnitude
    decade and equal trailing-zero profile - the whole digit-surface parity
    requirement in one comparison."""
    neg = s.startswith("-")
    t = s[1:] if neg else s
    ip, _, dp = t.partition(".")
    return (len(dp), len(ip), P.trailing_zeros(t), neg)


def alt_results(a, b):
    """Every other plain arithmetic reading of the same operand pair.

    A twin numeral that happens to equal one of these is defensible English
    labelled 0, so it is rejected."""
    out = set()
    cands = [a + b, a - b, b - a, a * b, (a + b) / 2.0, a, b]
    if abs(b) > 1e-9:
        cands += [a / b, a / b * 100.0]
    if abs(a) > 1e-9:
        cands += [b / a, b / a * 100.0]
    for v in cands:
        if v != v or abs(v) > 1e15:
            continue
        out.add(P.fmt(v))
    return out


# --------------------------------------------------------------------------- #
# candidate construction
# --------------------------------------------------------------------------- #
def _dup_labels(tab):
    labs = [r[tab["lab_ci"]].strip() for r in tab["body"]]
    return {k for k, n in collections.Counter(labs).items() if n > 1}


def pick_operands(tab, fam, rng):
    """One (column, row A, row B) draw and its correct result, or None.

    Used by BOTH the value-pool sweep and the build loop, so the pool the twin
    is drawn from is generated by the same process as the positives it must
    match."""
    hdr, body, lab_ci = tab["hdr"], tab["body"], tab["lab_ci"]
    cols = tab["numcols"]
    if not cols:
        return None
    dup = tab["dup"]
    for _ in range(CAND_TRIES):
        ci, vals = cols[rng.randrange(len(cols))]
        col = hdr[ci].strip()
        if len(vals) < 2:
            continue
        (ra, va), (rb, vb) = (vals[i] for i in rng.sample(range(len(vals)), 2))
        if abs(va - vb) < 1e-12:
            continue
        if fam == "difference":
            if va < vb:
                (ra, va), (rb, vb) = (rb, vb), (ra, va)
        elif fam == "percentage":
            if abs(vb) < 1e-9 or va <= 0 or vb <= 0:
                continue
            if va > vb:                      # keep the reading below 100 percent
                (ra, va), (rb, vb) = (rb, vb), (ra, va)
        elif fam == "product":
            if not (2 <= P.sigdigits(P.fmt(va)) <= 3 and 2 <= P.sigdigits(P.fmt(vb)) <= 3):
                continue
        sa, sb = body[ra][ci].strip(), body[rb][ci].strip()
        if "," in sa or "," in sb or sa == sb:
            continue
        ka, kb = body[ra][lab_ci].strip(), body[rb][lab_ci].strip()
        if not ka or not kb or ka == kb or ka in dup or kb in dup:
            continue
        r = compute(fam, va, vb)
        if r is None or r != r or abs(r) > 1e12:
            continue
        sp = P.fmt(r)
        if not (DIGIT_LEN_MIN <= P.digits(sp) <= DIGIT_LEN_MAX):
            continue
        if sp in (sa, sb):
            continue
        return dict(column_index=ci, column=col, row_index_a=ra, row_key_a=ka,
                    operand_a_str=sa, operand_a=va, row_index_b=rb, row_key_b=kb,
                    operand_b_str=sb, operand_b=vb, correct=r, correct_value=sp)
    return None


def fillers(tab, keep, rng):
    others = [r for r in range(len(tab["body"])) if r not in keep]
    rng.shuffle(others)
    return sorted(keep | set(others[: max(0, BODY_ROWS - len(keep))]))


# --------------------------------------------------------------------------- #
# the twin swap
# --------------------------------------------------------------------------- #
def admissible(a, b):
    """May construction `a` assert construction `b`'s correct result?

    The borrowed numeral must be genuinely wrong for a's operands, absent from
    a's evidence, and not a defensible alternative reading of a's own pair."""
    v = b["correct_value"]
    return (v != a["correct_value"]
            and v not in a["alts"]
            and v not in (a["operand_a_str"], a["operand_b_str"])
            and not (P.canon_set(v) & a["present"]))


def match_couples(cands, rng, reject):
    """Pair constructions of one (table, family, digit-surface signature) so each
    takes the other's correct result as its twin.

    Returned as couples, never as singletons: a couple is what makes the
    direction balance exact, the two legs' numeral multisets identical, and both
    appearances of a numeral fold-mates."""
    pool = list(cands)
    rng.shuffle(pool)
    out, dropped = [], 0
    while len(pool) >= 2:
        a = pool.pop()
        hit = None
        for k in range(min(MATCH_SCAN, len(pool))):
            b = pool[-1 - k]
            if not (admissible(a, b) and admissible(b, a)):
                continue
            ca = render_claim(a, b["correct_value"])
            cb = render_claim(b, a["correct_value"])
            if len(ca) != len(a["claim_pos"]) or len(cb) != len(b["claim_pos"]):
                continue
            if blocked(reject, ca) or blocked(reject, cb):
                continue
            hit = (len(pool) - 1 - k, ca, cb)
            break
        if hit is None:
            dropped += 1
            continue
        j, ca, cb = hit
        b = pool.pop(j)
        out.append((a, b, ca, cb))
    return out, dropped + len(pool)


# --------------------------------------------------------------------------- #
# evaluation-surface rejection set
# --------------------------------------------------------------------------- #
def _h(s):
    return hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest()


def surface_reject_set():
    """Hashes of every claim and every evidence unit of all thirteen evaluation
    surfaces, in all three C2 string forms.  Enumerated by the banked
    `quant_misbind_verify.eval_surfaces()` so the build rejects exactly what the
    contract measures."""
    hs = set()
    names = []
    for name, claims, ev, _note in QMV.eval_surfaces():
        names.append(name)
        for s in list(claims) + list(ev):
            if not s or not s.strip():
                continue
            hs.add(_h(s))
            hs.add(_h(s[:CHUNK_MAX]))
            hs.add(_h(QMV.norm_ws(s)))
    return hs, names


def blocked(hs, s):
    return (_h(s) in hs) or (_h(s[:CHUNK_MAX]) in hs) or (_h(QMV.norm_ws(s)) in hs)


def surface_table_ids():
    """TabFact `table_id`s any evaluation surface names, dropped whole."""
    ids = set()
    src = {}
    for f in ("R20-H177_eval_B.parquet", "R20-H177_eval_C.parquet"):
        p = HERE / f
        if not p.exists():
            continue
        d = pl.read_parquet(p, columns=["doc_id"])
        got = {x.split(":", 1)[1] for x in d["doc_id"].to_list() if x.startswith("tabfact:")}
        src[f] = len(got)
        ids |= got
    for f in ("R17-H146_antigaming_set.parquet", "R18-H150_antigaming_set.parquet",
              "R19-H159_antigaming_set.parquet"):
        p = HERE / f
        if not p.exists():
            continue
        got = set(pl.read_parquet(p, columns=["table_id"])["table_id"].to_list())
        src[f] = len(got)
        ids |= got
    return ids, src


def non_train_table_ids():
    """TabFact test + validation `table_id`s - the archive's own split axis."""
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    out, per = set(), {}
    for n in z.namelist():
        if not n.endswith(".parquet") or n.endswith("__train.parquet"):
            continue
        d = pl.read_parquet(io.BytesIO(z.read(n)), columns=["table_id"])
        got = set(d["table_id"].to_list())
        per[n.split("__")[-1].replace(".parquet", "")] = len(got)
        out |= got
    return out, per


# --------------------------------------------------------------------------- #
# response-only surface channels - the binding C5 constraint
# --------------------------------------------------------------------------- #
def response_channels(claims):
    ch = collections.OrderedDict()
    lens = [len(c) for c in claims]
    sents = [max(1, len(SENT.findall(c))) for c in claims]
    nums = [len(NUM_FREE.findall(c)) for c in claims]
    ch["character_length"] = [float(x) for x in lens]
    ch["sentence_count"] = [float(x) for x in sents]
    ch["numeric_token_count"] = [float(x) for x in nums]
    ch["numeric_density_per_100_chars"] = [100.0 * n / max(l, 1) for n, l in zip(nums, lens)]
    ch["derivation_marker_count"] = [
        float(sum(c.count(s) for s in ("=", "+", "-", "*", "/"))
              + sum(1 for w in re.findall(r"[a-z]+", c.lower()) if w in MARKER_WORDS))
        for c in claims]
    ch["equals_count"] = [float(c.count("=")) for c in claims]
    ch["newline_count"] = [float(c.count("\n")) for c in claims]
    ch["word_count"] = [float(len(c.split())) for c in claims]
    ch["digit_char_fraction"] = [
        sum(x.isdigit() for x in c) / max(len(c), 1) for c in claims]
    ch["chars_per_sentence"] = [l / s for l, s in zip(lens, sents)]
    return ch


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def verify(df, rng):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    out = {}
    labels = df["label"].to_list()
    claims = df["claim"].to_list()
    fams = df["neg_family"].to_list()

    # --- claim-only probe: document-disjoint folds, stratified on the
    # document's (family, direction) so no fold's training complement carries
    # only the mirrored direction (R17-H145 finding b).
    doc_key = {d: k for d, k in df.filter(pl.col("label") == 1)
               .group_by("doc_id")
               .agg((pl.col("neg_family") + ":" + pl.col("direction")).first())
               .iter_rows()}
    strata = collections.defaultdict(list)
    for d in sorted(doc_key):
        strata[doc_key[d]].append(d)
    fold_of, i = {}, 0
    for k in sorted(strata):
        ds = strata[k]
        rng.shuffle(ds)
        for d in ds:
            fold_of[d] = i % N_FOLDS
            i += 1
    folds = np.array([fold_of[d] for d in df["doc_id"].to_list()])
    score = np.zeros(len(df))
    idx = np.arange(len(df))
    for f in range(N_FOLDS):
        tr_i, te_i = idx[folds != f], idx[folds == f]
        if not len(te_i):
            continue
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                              max_features=300_000, sublinear_tf=True)
        Xtr = vec.fit_transform([claims[j] for j in tr_i])
        Xte = vec.transform([claims[j] for j in te_i])
        clf = LogisticRegression(solver="liblinear", C=4.0, tol=1e-7, max_iter=3000)
        clf.fit(Xtr, [labels[j] for j in tr_i])
        score[te_i] = clf.decision_function(Xte)
        print(f"    probe fold {f} done", flush=True)
    probe = P.auroc(labels, score)
    out["claim_only_converged_probe"] = {
        "value": round(float(probe), 6), "bar": "< 0.55", "pass": bool(probe < 0.55),
        "scoring": f"{N_FOLDS}-fold document-disjoint, out of fold, "
                   "direction-stratified, liblinear tol 1e-7, char_wb 2-5 TF-IDF",
        "documents": len(fold_of), "rows": len(df),
        "two_sided_deviation_report_only": round(abs(float(probe) - 0.5), 6),
        "why_two_sided_is_reported": "the registered bar is one-sided, and a rejected "
                                     "corruption design of this lane read 0.1923 - a "
                                     "leak the one-sided bar would have passed"}

    scored = df.select(["pair_id", "label", "neg_family"]).with_columns(
        pl.Series("score", score))
    fam_acc, worst = {}, 0.0
    for fam, sub in scored.group_by("neg_family"):
        piv = sub.pivot(on="label", index="pair_id", values="score",
                        aggregate_function="first").drop_nulls()
        if not len(piv):
            continue
        pos, neg = piv["1"].to_numpy(), piv["0"].to_numpy()
        acc = float(((pos > neg) + 0.5 * (pos == neg)).mean())
        fam_acc[fam[0]] = {"acc": round(acc, 6), "pairs": len(piv)}
        worst = max(worst, acc)
    out["within_pair_claim_only"] = {
        "per_family": fam_acc, "worst": round(worst, 6), "bar": "< 0.60",
        "pass": bool(worst < 0.60),
        "worst_two_sided_deviation_report_only": round(
            max((abs(v["acc"] - 0.5) for v in fam_acc.values()), default=0.0), 6)}

    # --- response-only surface channels.  Identical rendered length inside a
    # pair makes every one of these degenerate; measured, never assumed.
    ch_out, worst_ch, worst_name = {}, 0.0, None
    for name, vals in response_channels(claims).items():
        pooled = P.auroc(labels, vals)
        per_fam = {}
        for f in sorted(set(fams)):
            m = [j for j, x in enumerate(fams) if x == f]
            per_fam[f] = round(P.auroc([labels[j] for j in m], [vals[j] for j in m]), 6)
        dev = max([abs(pooled - 0.5)] + [abs(v - 0.5) for v in per_fam.values()
                                         if v == v])
        ch_out[name] = {"pooled": round(pooled, 6), "per_family": per_fam,
                        "worst_deviation_from_0.5": round(dev, 6),
                        "in_band_0.45_0.55": bool(dev <= 0.05)}
        if dev >= worst_ch:
            worst_ch, worst_name = dev, name
    out["surface_parity_channels"] = ch_out
    out["surface_parity_worst"] = {
        "channel": worst_name, "pooled": ch_out[worst_name]["pooled"],
        "worst_deviation_from_0.5": round(worst_ch, 6),
        "bar": "in [0.45, 0.55] on every channel",
        "pass": bool(all(v["in_band_0.45_0.55"] for v in ch_out.values()))}

    # --- digit-surface parity between the two legs of every pair
    piv = (df.select(["pair_id", "label", "asserted_value"])
             .pivot(on="label", index="pair_id", values="asserted_value",
                    aggregate_function="first").drop_nulls())
    pv, nv = piv["1"].to_list(), piv["0"].to_list()
    same_key = [surface_key(a) == surface_key(b) for a, b in zip(pv, nv)]
    same_len = [len(a) == len(b) for a, b in zip(pv, nv)]
    above = [float(b) > float(a) for a, b in zip(pv, nv)]
    asserted = df["asserted_value"].to_list()
    out["digit_surface_parity"] = {
        "pairs": len(piv),
        "identical_signature_rate": round(float(np.mean(same_key)), 6),
        "identical_rendered_length_rate": round(float(np.mean(same_len)), 6),
        "share_twin_above_true": round(float(np.mean(above)), 6),
        "digit_count_auroc": round(P.auroc(labels, [P.digits(v) for v in asserted]), 6),
        "trailing_zero_auroc": round(P.auroc(labels, [P.trailing_zeros(v) for v in asserted]), 6),
        "magnitude_decade_auroc": round(P.auroc(
            labels, [math.floor(math.log10(abs(float(v)))) if abs(float(v)) > 0 else 0.0
                     for v in asserted]), 6),
        "value_magnitude_auroc": round(P.auroc(labels, [float(v) for v in asserted]), 6),
        "bar": "signature identical on every pair, 50/50 above and below, "
               "digit-count / trailing-zero / decade AUROC in [0.45, 0.55]",
        "pass": bool(all(same_key) and abs(float(np.mean(above)) - 0.5) <= 0.01)}

    # --- single-channel probes
    piv_c = (df.select(["pair_id", "label", "chunk"])
               .pivot(on="label", index="pair_id", values="chunk",
                      aggregate_function="first").drop_nulls())
    out["single_channel_probes"] = {
        "evidence_only": {
            "within_pair_identical_chunk_share": round(
                float((piv_c["1"] == piv_c["0"]).mean()), 6),
            "auroc": 0.5,
            "reading": "the legs of a pair share the evidence byte-identically, so an "
                       "evidence-only reader is exactly at chance within pair - "
                       "measured, not assumed"},
        "question_only": {
            "value": None,
            "reading": "NOT COMPUTABLE - the lane carries no question field; the "
                       "construction implies no question channel"}}

    # --- arithmetic re-derivation, full population
    errs = collections.Counter()
    pos_ok = neg_ok = 0
    for r in df.iter_rows(named=True):
        want = compute(r["neg_family"], r["operand_a"], r["operand_b"])
        got = None if want is None else P.fmt(want)
        if r["label"] == 1:
            if got == r["asserted_value"]:
                pos_ok += 1
            else:
                errs["positive asserts a value the arithmetic does not give"] += 1
        else:
            if got is not None and got != r["asserted_value"]:
                neg_ok += 1
            else:
                errs["negative asserts the correct value"] += 1
    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    out["arithmetic_rederivation"] = {
        "method": "every row recomputed from its two stated operands, full population",
        "positive_rows": n_pos, "positives_verified": pos_ok,
        "negative_rows": n_neg, "negatives_refuted": neg_ok,
        "errors": int(sum(errs.values())), "error_kinds": dict(errs),
        "bar": "0 errors", "pass": bool(sum(errs.values()) == 0)}

    # --- balance
    dirs = {f"{a}:{b}": n for a, b, n in
            df.group_by(["neg_family", "direction"]).len().iter_rows()}
    fam_rows = {k: v for k, v in df.group_by("neg_family").len().iter_rows()}
    out["balance"] = {
        "family_rows": fam_rows, "direction_rows": dirs,
        "direction_balance_exact_50_50_per_family": all(
            dirs.get(f"{f}:up") == dirs.get(f"{f}:down") for f in fam_rows)}

    out["all_registered_bars_pass"] = bool(
        out["claim_only_converged_probe"]["pass"]
        and out["within_pair_claim_only"]["pass"]
        and out["surface_parity_worst"]["pass"]
        and out["digit_surface_parity"]["pass"]
        and out["arithmetic_rederivation"]["pass"]
        and out["balance"]["direction_balance_exact_50_50_per_family"])

    # executor-added, reported separately and NOT joined to the registered set
    out["executor_added_probes_reported_separately"] = {
        "note": "NOT part of the registered conjunction; reported so they cannot "
                "silently join it (C5 provenance)",
        "leading_digit_auroc": round(P.auroc(
            labels, [float(P.leading_digit(v) or 0) for v in asserted]), 6),
        "decimal_presence_auroc": round(P.auroc(
            labels, [1.0 if "." in v else 0.0 for v in asserted]), 6),
        "claim_token_count_auroc": round(P.auroc(
            labels, [len(QMV.tok(c)) for c in claims]), 6),
        "containment_auroc": round(P.auroc(
            labels, [QMV.containment(a, b) for a, b in
                     zip(claims, df["chunk"].to_list())]), 6),
    }
    return out


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    print("=== R22-H187 num_derive lane ===", flush=True)
    print("enumerating evaluation surfaces (C2 rejection set)...", flush=True)
    reject, surface_names = surface_reject_set()
    print(f"  {len(surface_names)} surfaces -> {len(reject)} blocked string hashes",
          flush=True)

    surf_ids, surf_src = surface_table_ids()
    non_train, split_sizes = non_train_table_ids()
    print(f"  {len(surf_ids)} tabfact table_ids named by an evaluation surface "
          f"({surf_src}); {len(non_train)} in a non-train split {split_sizes}", flush=True)

    excluded_ids, prints, eval_rows, unmatched = P.evalset_documents()
    print(f"  R17-H143 eval set: {eval_rows} rows -> {len(excluded_ids)} doc_ids, "
          f"{len(prints)} content fingerprints ({unmatched} unmatched)", flush=True)

    print("loading TabFact train (public archive, tracked sidecar)...", flush=True)
    raw = P.tabfact_tables()
    drop_idx = P.excluded_tables(raw, prints)
    print(f"  {len(raw)} candidate tables; {len(drop_idx)} carry eval content", flush=True)

    tables, dropped = [], collections.Counter()
    for ti, t in enumerate(raw):
        tid = t["doc_id"].split(":", 1)[1]
        if ti in drop_idx:
            dropped["eval_content_fingerprint"] += 1
            continue
        if t["doc_id"] in excluded_ids:
            dropped["eval_doc_id"] += 1
            continue
        if tid in surf_ids:
            dropped["named_by_an_eval_surface"] += 1
            continue
        if tid in non_train:
            dropped["non_train_split"] += 1
            continue
        lab = P.label_column(t["hdr"], t["body"])
        if lab is None:
            dropped["no_label_column"] += 1
            continue
        cols = [(ci, v) for ci, v in P.numeric_columns(t["hdr"], t["body"], lab)
                if P.good_header(t["hdr"][ci])]
        if not cols:
            dropped["no_numeric_column"] += 1
            continue
        t["lab_ci"] = lab
        t["numcols"] = cols
        t["dup"] = _dup_labels(t)
        t["table_id"] = tid
        tables.append(t)
    print(f"  dropped {dict(dropped)}; {len(tables)} tables admitted", flush=True)

    forms = list(P.FORM_WEIGHTS)
    w = np.array([P.FORM_WEIGHTS[f] for f in forms], dtype=float)
    w /= w.sum()
    for t, k in zip(tables, np_rng.choice(len(forms), size=len(tables), p=w)):
        t["form"] = forms[int(k)]

    # ---------------- phase 1 + 2: construct, then swap inside the table ---
    pair_target = {f: int(round(N_PAIRS * s)) for f, s in FAMILY_SHARE.items()}
    pair_target["difference"] += N_PAIRS - sum(pair_target.values())
    for f in pair_target:
        assert pair_target[f] % 2 == 0, (f, pair_target[f])
    couple_target = {f: n // 2 for f, n in pair_target.items()}
    print(f"phase 1 - construct and couple; pair target {pair_target}", flush=True)

    rejects = collections.Counter()
    per_doc, seen = collections.Counter(), set()

    def construct(tab, fam):
        c = pick_operands(tab, fam, rng)
        if c is None:
            rejects["no_operand_draw"] += 1
            return None
        keep = {c["row_index_a"], c["row_index_b"]}
        chunk = P.serialize(tab["form"], tab["caption"], tab["hdr"],
                            [tab["body"][r] for r in fillers(tab, keep, rng)],
                            tab["lab_ci"], (c["column_index"],))
        present = P.present_numbers(chunk)
        sa, sb, sp = c["operand_a_str"], c["operand_b_str"], c["correct_value"]
        if sa not in chunk or sb not in chunk:
            rejects["operand_not_verbatim_in_chunk"] += 1
            return None
        if not (P.canon_set(sa) & present) or not (P.canon_set(sb) & present):
            rejects["operand_not_readable_as_numeral"] += 1
            return None
        if c["row_key_a"] not in chunk or c["row_key_b"] not in chunk:
            rejects["row_key_not_in_chunk"] += 1
            return None
        if P.canon_set(sp) & present:
            rejects["correct_result_present_in_evidence"] += 1
            return None
        if blocked(reject, chunk):
            rejects["chunk_collides_with_an_eval_surface"] += 1
            return None
        sig = (chunk, c["column"], c["row_key_a"], c["row_key_b"], sp)
        if sig in seen:
            rejects["duplicate_construction"] += 1
            return None
        ti = rng.randrange(len(TEMPLATES[fam]))
        claim_pos = render(fam, ti, c["column"], c["row_key_a"], sa,
                           c["row_key_b"], sb, sp)
        if blocked(reject, claim_pos):
            rejects["positive_claim_collides_with_an_eval_surface"] += 1
            return None
        seen.add(sig)
        c.update(neg_family=fam, template_id=ti, chunk=chunk, present=present,
                 claim_pos=claim_pos, key=surface_key(sp),
                 alts=alt_results(c["operand_a"], c["operand_b"]) - {sp},
                 doc_id=tab["doc_id"], table_id=tab["table_id"],
                 serial_form=tab["form"])
        return c

    def couple_table(tab, fam):
        """Constructions from ONE table, coupled inside their signature bucket."""
        buckets = collections.defaultdict(list)
        for _ in range(PER_TABLE_TRIES):
            c = construct(tab, fam)
            if c is not None:
                buckets[c["key"]].append(c)
        out, unmatched = [], 0
        for key in sorted(buckets):
            got, drop = match_couples(buckets[key], rng, reject)
            out += got
            unmatched += drop
        rejects["construction_left_uncoupled_in_its_table"] += unmatched
        return out

    couples = {f: [] for f in couple_target}
    for cap in COUPLE_CAP_LADDER:
        if all(len(couples[f]) >= couple_target[f] for f in couple_target):
            break
        order = list(range(len(tables)))
        rng.shuffle(order)
        for oi in order:
            t = tables[oi]
            if per_doc[t["doc_id"]] >= cap:
                continue
            need = [f for f in couple_target if len(couples[f]) < couple_target[f]]
            if not need:
                break
            need.sort(key=lambda f: len(couples[f]) / couple_target[f])
            for fam in need[:2]:
                room = cap - per_doc[t["doc_id"]]
                if room <= 0:
                    break
                got = couple_table(t, fam)[:room]
                couples[fam] += got
                per_doc[t["doc_id"]] += len(got)
        print(f"  couple cap {cap}: "
              f"{ {f: len(v) for f, v in sorted(couples.items())} } "
              f"over {len(per_doc)} documents", flush=True)

    print("phase 2 - emit...", flush=True)
    rows, pid = [], 0
    match_stats = {}
    for fam in sorted(couple_target):
        rng.shuffle(couples[fam])
        want = couple_target[fam]
        match_stats[fam] = {"couples_formed": len(couples[fam]),
                            "couples_wanted": want,
                            "couples_used": min(want, len(couples[fam]))}
        if len(couples[fam]) < want:
            print(f"  WARNING {fam}: {len(couples[fam])} couples formed, {want} wanted",
                  flush=True)
        for a, b, ca, cb in couples[fam][:want]:
            for host, other, neg_claim in ((a, b, ca), (b, a, cb)):
                sn = other["correct_value"]
                base = dict(neg_family=fam,
                            direction="up" if float(sn) > float(host["correct_value"])
                                      else "down",
                            chunk=host["chunk"], doc_id=host["doc_id"],
                            source="tabfact", table_id=host["table_id"],
                            serial_form=host["serial_form"],
                            template_id=host["template_id"], column=host["column"],
                            column_index=host["column_index"],
                            row_key_a=host["row_key_a"], row_index_a=host["row_index_a"],
                            row_key_b=host["row_key_b"], row_index_b=host["row_index_b"],
                            operand_a=host["operand_a"], operand_b=host["operand_b"],
                            operand_a_str=host["operand_a_str"],
                            operand_b_str=host["operand_b_str"],
                            correct_value=host["correct_value"], wrong_value=sn,
                            twin_row_keys=f'{other["row_key_a"]} / {other["row_key_b"]}',
                            surface_signature="|".join(str(x) for x in host["key"]),
                            result_digits=P.digits(host["correct_value"]))
                rows.append(dict(pair_id=pid, label=1, claim=host["claim_pos"],
                                 asserted_value=host["correct_value"], **base))
                rows.append(dict(pair_id=pid, label=0, claim=neg_claim,
                                 asserted_value=sn, **base))
                pid += 1
        print(f"  {fam}: {match_stats[fam]}", flush=True)

    df = pl.DataFrame(rows).sort(["pair_id", "label"], descending=[False, True])
    df.write_parquet(OUT)
    print(f"wrote {OUT}  rows={df.height} pairs={df['pair_id'].n_unique()} "
          f"documents={df['doc_id'].n_unique()}  [{round(time.time() - t0, 1)}s]",
          flush=True)

    print("phase 3 - verify...", flush=True)
    res = verify(df, random.Random(SEED))

    fam_rows = {k: v for k, v in df.group_by("neg_family").len().iter_rows()}
    man = dict(
        member="num_derive",
        experiment="R22-H187 - the missing derivation-checking training member",
        builder="experiments/grounding-semantic/R22-H187_num_derive_lane.py",
        seed=SEED,
        rows=df.height, pairs=int(df["pair_id"].n_unique()),
        unit_note="BOTH units declared: 30,000 rows = 15,000 pairs",
        documents=int(df["doc_id"].n_unique()),
        pairs_per_document=round(df["pair_id"].n_unique() / max(df["doc_id"].n_unique(), 1), 4),
        couple_cap=COUPLE_CAP_LADDER[-1],
        families={k: v for k, v in df.filter(pl.col("label") == 0)
                  .group_by("neg_family").len().iter_rows()},
        family_rows=fam_rows,
        family_shares_declared=FAMILY_SHARE,
        family_shares_measured={k: round(v / df.height, 4) for k, v in fam_rows.items()},
        directions={f"{a}:{b}": n for a, b, n in
                    df.group_by(["neg_family", "direction"]).len().iter_rows()},
        sources={"tabfact": {
            "archive": "data/external/datasets/dataset-tabfact.zip",
            "sidecar": "data/external/datasets/dataset-tabfact.md (tracked)",
            "licence": "CC-BY-4.0 (from the tracked sidecar)",
            "split": "the *__train.parquet member ONLY",
            "selection_predicate":
                "R17-H144_pairs.tabfact_tables: deduplicated on table_text, >= 4 body "
                "rows of uniform width, >= 2 columns; then a label column and at least "
                "one numeric column with a usable header must exist",
            "feverous_used": False,
            "why_no_feverous": "quant_misbind is non-conforming on exactly its FEVEROUS "
                               "half (C3 unmeasurable split axis on 33.7% of rows, C8 no "
                               "licence / retrieval date / tracked source); this lane "
                               "inherits none of it"}},
        exclusions=dict(
            evalset_rows=eval_rows, evalset_doc_ids=len(excluded_ids),
            content_fingerprints=len(prints),
            tables_dropped=dict(dropped),
            tabfact_table_ids_named_by_an_eval_surface=len(surf_ids),
            eval_surface_table_id_sources=surf_src,
            tabfact_non_train_table_ids=len(non_train),
            tabfact_split_sizes=split_sizes,
            blocked_string_hashes=len(reject),
            surfaces_enumerated=surface_names,
            construction_rejections=dict(rejects)),
        twin_construction=dict(
            method="two constructions from the SAME TABLE and the same family, sharing "
                   "a digit-surface signature (decimal places, integer digits, trailing "
                   "zeros, sign), swap correct results: each asserts the other's. The "
                   "negative leg's numeral multiset is therefore identical to the "
                   "positive leg's, each couple contributes one twin above and one "
                   "below its true value, and both appearances of a numeral share a "
                   "document so they never straddle a fold of a document-disjoint probe",
            guards="the borrowed numeral must differ from the host's correct result, "
                   "be absent from the host's evidence, not equal either operand, and "
                   "not equal any other plain arithmetic reading of the host's operand "
                   "pair (sum, difference either way, product, mean, both ratios, both "
                   "percentages)",
            rejected_designs={
                "pooled_draw_conditioned_on_direction": "claim-only probe 0.5761, FAIL",
                "cross_document_swap": "claim-only probe 0.1923 - two-sided deviation "
                                       "0.308 from a numeral memorised across folds"},
            constructions_per_table_visit=PER_TABLE_TRIES, partner_scan=MATCH_SCAN,
            couple_cap_ladder=list(COUPLE_CAP_LADDER),
            pair_target=pair_target, per_family=match_stats),
        diversity=dict(
            serial_forms={k: v for k, v in df.group_by("serial_form").len().iter_rows()},
            templates={f"{a}:{b}": n for a, b, n in
                       df.group_by(["neg_family", "template_id"]).len().iter_rows()},
            distinct_claims=int(df["claim"].n_unique()),
            distinct_chunks=int(df["chunk"].n_unique()),
            distinct_columns=int(df["column"].n_unique()),
            distinct_tables=int(df["table_id"].n_unique()),
            result_digit_lengths={str(k): v for k, v in
                                  sorted(df.group_by("result_digits").len().iter_rows())}),
        verify=res)
    MANIFEST.write_text(json.dumps(man, indent=2))
    print(json.dumps({k: man[k] for k in
                      ("rows", "pairs", "documents", "families", "directions",
                       "family_shares_measured")}, indent=2), flush=True)
    print(json.dumps(res, indent=2), flush=True)
    ok = res["all_registered_bars_pass"]
    print(f"=== R22-H187 num_derive lane {'BUILT' if ok else 'BUILT, A BAR FAILED'} "
          f"[{round(time.time() - t0, 1)}s] ===", flush=True)


if __name__ == "__main__":
    main()

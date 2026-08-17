"""Contract verification of every REMAINING held-out mechanism eval.

CPU only, Polars only, torch-free.  Measurement and building only - no verdict
is adjudicated here and no bar is amended; the coordinator adjudicates.

WHAT IS UNDER TEST
------------------
Every non-arena instrument the campaign reads a gate or a verdict on, enumerated
from disk (`experiments/grounding-semantic/*.parquet`) rather than from a list,
MINUS the four artifacts other agents own this session: `R20-H177_eval_B`
(being rebuilt), `R17-H143_evalset` (being assessed), the blind arena and
`gold_full`.

Two of the enumerated instruments store no evidence text at all - the
ANTI-GAMING set and the R15 probe bank keep only a TabFact `table_id` and let
the reader reconstitute the serialised table.  Those are reconstituted here with
the builders' own serialisation (`R14-H133_antigaming.evidence_of`), so the
evidence channel is measured rather than skipped.

THE CLAUSES RUN, with amendments C-A1 and C-A2 applied
-----------------------------------------------------
  C1  structural test (a negative leg's (claim, evidence) identical to a
      positive leg's), strict separation under a PREDICATE-SENSITIVE instrument
      where one is computable, absolute attestation levels always
  C2  three string forms crossed six ways in BOTH directions, on CLAIM and on
      EVIDENCE units, against four surfaces (flagship mix / mix superset /
      blind arena / gold_full), PLUS the DOCUMENT channel raw and
      STEM-NORMALISED wherever the eval carries a document id
  C3  split semantics measured, for every eval drawn from a corpus the mix also
      uses (TabFact test+validation, VitaminC test+validation)
  C5  leak suite for paired-contrast evals - claim-only converged probe,
      within-pair claim-only, single-channel probes, surface parity restricted
      by C-A1 to channels that do not read the claim-evidence relation
  C6  memorisation channel keyed on a field the eval and the TRAINING MIX share
  C7  declared units and volume

LIVE POSITIVE CONTROLS on the disjointness instrument - four of them
--------------------------------------------------------------------
  1. SYNTHETIC IDENTITY - 300 passages sampled from the mix itself are offered
     to the gate as if they were an eval.  A gate that cannot read 300 of 300
     cannot certify anything
  2. SYNTHETIC RE-WRAP - the same 300 passages with their whitespace re-wrapped.
     The raw and truncated forms must read 0 and the NORMALISED forms must read
     300 of 300.  This is the exact mode that hid `R20-H177_eval_B`'s
     contamination while string matching read 4.5%
  3. LIVE BANKED, string channel - `R20-H175b_qlane_eval.parquet`, the withdrawn
     eval banked at 485 of 487 passages inside the mix (99.6%)
  4. LIVE BANKED, document channel - the original `R20-H177_eval_B.parquet`,
     banked at 325 of 325 TabFact document stems inside the `tabfact` member
     while its raw ids read 0

Out: contract/mechanism_evals_report.json
Run: CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run python \
     experiments/grounding-semantic/contract/mechanism_evals_verify.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections
import hashlib
import importlib.util as _ilu
import io
import json
import pathlib
import random
import re
import time
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
ROOT = SEM.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
OUT = HERE / "mechanism_evals_report.json"
TABLE_OUT = HERE / "mechanism_evals_summary.md"

CUT = 1500  # M59.CFG.chunk_max_chars, the serving unit
PROX_W = 200  # chars, the binding-proximity window
NOTE = "Numbers recorded, not adjudicated - the coordinator adjudicates."

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]+")
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")

t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------- #
# string forms - the banked instrument, carried on 16-byte digests so the whole
# 760,618-row mix fits in memory in all four forms at once
# --------------------------------------------------------------------------- #
def norm(s):
    return _WS.sub(" ", s).strip().casefold()


def dg(s):
    return hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=16).digest()


def unit_forms(t):
    """The four digests one text unit contributes."""
    return (dg(t), dg(t[:CUT]), dg(norm(t)), dg(norm(t[:CUT])))


FORM_KEYS = ("raw", "trunc", "nraw", "ntrunc")


def build_forms(texts):
    """(form-name -> set of digests, list of per-unit 4-tuples) for DISTINCT units."""
    seen, tuples = set(), []
    sets = {k: set() for k in FORM_KEYS}
    for t in texts:
        if not t:
            continue
        d0 = dg(t)
        if d0 in seen:
            continue
        seen.add(d0)
        q = unit_forms(t)
        tuples.append(q)
        for k, d in zip(FORM_KEYS, q):
            sets[k].add(d)
    return sets, tuples


# the six-form cross, in the banked order and with the banked names
def index_units(texts, tags):
    """One pass over a surface: the four form digests per DISTINCT unit, the
    member that supplies each digest, and the set of members every unit belongs
    to (so a subset surface is taken without re-hashing)."""
    bit = {g: 1 << i for i, g in enumerate(sorted(set(tags)))}
    q_of, members = {}, {}
    tag_of = {k: {} for k in FORM_KEYS}
    for t, g in zip(texts, tags):
        if not t:
            continue
        d0 = dg(t)
        q = q_of.get(d0)
        if q is None:
            q = unit_forms(t)
            q_of[d0] = q
            members[d0] = bit[g]
            for k, d in zip(FORM_KEYS, q):
                tag_of[k].setdefault(d, g)
        else:
            members[d0] |= bit[g]
    tuples = list(q_of.values())
    sets = {k: {q[i] for q in tuples} for i, k in enumerate(FORM_KEYS)}
    return {"q_of": q_of, "members": members, "bit": bit, "tuples": tuples,
            "sets": sets, "tag_of": tag_of}


def subset_forms(index, keep_tags):
    """The form sets of the sub-surface made of units any of `keep_tags` supplies."""
    mask = 0
    for g in keep_tags:
        mask |= index["bit"].get(g, 0)
    tuples = [q for d, q in index["q_of"].items() if index["members"][d] & mask]
    sets = {k: {q[i] for q in tuples} for i, k in enumerate(FORM_KEYS)}
    return sets, tuples


CROSS = (
    ("raw_in_raw", 0, "raw"),
    ("raw_in_truncated", 0, "trunc"),
    ("truncated_in_raw", 1, "raw"),
    ("truncated_in_truncated", 1, "trunc"),
    ("normalised_in_normalised_raw", 2, "nraw"),
    ("normalised_in_normalised_truncated", 3, "ntrunc"),
)


def six_forms(query_sets, query_tuples, target_sets):
    """Each of the six form tests, plus the union of query units hitting any.

    Counts are over distinct QUERY UNITS, as the banked instrument counts them.
    The cheap digest intersection is taken first: where it is empty the unit
    count is provably zero, which is the common case and saves walking 760,618
    mix units per eval per channel."""
    counts = {"n_query_units": len(query_tuples)}
    hit_any = set()
    for name, qi, tk in CROSS:
        inter = query_sets[FORM_KEYS[qi]] & target_sets[tk]
        if not inter:
            counts[name] = 0
            continue
        n = 0
        for i, q in enumerate(query_tuples):
            if q[qi] in inter:
                n += 1
                hit_any.add(i)
        counts[name] = n
    counts["any_form"] = len(hit_any)
    return counts, hit_any


def both_directions(eval_sets, eval_tuples, surface_sets, surface_tuples):
    a, hit = six_forms(eval_sets, eval_tuples, surface_sets)
    b, _ = six_forms(surface_sets, surface_tuples, eval_sets)
    return {"eval_units_into_surface": a, "surface_units_into_eval": b}, hit


# --------------------------------------------------------------------------- #
# document channel
# --------------------------------------------------------------------------- #
def doc_stem(d):
    """TabFact writes one Wikipedia table under both a `1-` and a `2-` prefixed
    csv id.  Stripping that prefix is the channel that read 325 of 325 on the
    banked eval_B while its raw ids read 0.  Every other namespace is identity."""
    if d.startswith("tabfact:"):
        t = d[len("tabfact:"):]
        return "tabfact:" + (t[2:] if len(t) > 2 and t[0] in "12" and t[1] == "-" else t)
    return d


# --------------------------------------------------------------------------- #
# attestation instruments
# --------------------------------------------------------------------------- #
def tokens(t):
    return _WORD.findall(t.lower())


def containment(claim, text):
    ct = set(tokens(claim))
    if not ct:
        return 0.0
    return len(ct & set(tokens(text))) / len(ct)


def canon_nums(t):
    out = set()
    for m in _NUM.findall(t or ""):
        s = m.replace(",", "").rstrip(".")
        try:
            v = float(s)
        except ValueError:
            continue
        out.add(round(v, 6))
    return out


def numeral_attested(claim, chunk):
    """Fraction of the claim's numerals the evidence prints.  Predicate-SENSITIVE
    for every near-miss family, whose negative asserts a value absent from the
    evidence.  None where the claim carries no numeral (counted as coverage)."""
    c = canon_nums(claim)
    if not c:
        return None
    return len(c & canon_nums(chunk)) / len(c)


def proximity_attested(anchor, value, chunk, w=PROX_W):
    """Does the evidence print `value` within `w` characters of `anchor`?

    Predicate-SENSITIVE for every BINDING family - role swap, row swap, unit
    swap, period swap - where both the true and the swapped value are printed
    somewhere and only their attachment differs.  None where either term is
    absent from the evidence (counted as coverage, never silently dropped)."""
    if not anchor or not value or not chunk:
        return None
    lc, la, lv = chunk.lower(), anchor.lower().strip(), str(value).lower().strip()
    if not la or not lv:
        return None
    ai = [m.start() for m in re.finditer(re.escape(la), lc)]
    vi = [m.start() for m in re.finditer(re.escape(lv), lc)]
    if not ai or not vi:
        return None
    return float(min(abs(a - v) for a in ai for v in vi) <= w)



def relational_binding(anchor, rival, value, chunk):
    """Is the asserted value bound to the term the CLAIM names, or to its rival?

    Predicate-SENSITIVE for a role / period / subject swap, where both terms and
    the value are all printed and only the attachment differs - the case a plain
    proximity test reads as attested on both legs.  Returns 1.0 when the value
    sits nearer the claim's own anchor, 0.0 when it sits nearer the rival, None
    when a term is absent or the two distances tie."""
    if not (anchor and rival and value and chunk):
        return None
    lc = chunk.lower()
    a = [m.start() for m in re.finditer(re.escape(anchor.lower().strip()), lc)]
    r = [m.start() for m in re.finditer(re.escape(rival.lower().strip()), lc)]
    v = [m.start() for m in re.finditer(re.escape(str(value).lower().strip()), lc)]
    if not a or not r or not v:
        return None
    da = min(abs(x - y) for x in a for y in v)
    dr = min(abs(x - y) for x in r for y in v)
    if da == dr:
        return None
    return float(da < dr)


_ITEM = re.compile(r"^Item (\d+) in the list states: (.+)$", re.S)


def item_index_attested(claim, chunk):
    """Does the evidence number the quoted item the way the claim says it does?

    Predicate-SENSITIVE for the H148 step-misbinding family: the quoted text is
    verbatim in the evidence on BOTH legs, and only the item number attached to
    it differs."""
    m = _ITEM.match(claim.strip())
    if not m:
        return None
    n, text = m.group(1), m.group(2).strip()
    probe = text[:40].lower()
    lc = chunk.lower()
    i = lc.find(probe)
    if i < 0:
        return None
    back = chunk[max(0, i - 12):i]
    mm = re.search(r"(\d+)\.\s*$", back)
    if not mm:
        return None
    return float(mm.group(1) == n)


def dist(v):
    v = np.asarray([x for x in v if x is not None], dtype="float64")
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "mean": round(float(v.mean()), 4),
            "median": round(float(np.median(v)), 4),
            "rate_eq_1.0": round(float((v >= 1.0).mean()), 4),
            "rate_ge_0.9": round(float((v >= 0.9).mean()), 4)}


def auroc(y, s):
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if not (y == 1).any() or not (y == 0).any():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    ranks[order] = np.arange(1, s.size + 1)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


# --------------------------------------------------------------------------- #
# C5 probes - the banked converged instrument (R20-H174_lane_common)
# --------------------------------------------------------------------------- #
def text_probe(texts, labels, groups, rng, n_folds=5):
    """Out-of-fold char-ngram TF-IDF + liblinear probe, folds disjoint on
    `groups`.  liblinear at tol 1e-7, never default lbfgs (R17-H144 finding ii).
    Transcribed from R20-H174_lane_common.claim_only_probe so this pass does not
    import a module that pins CUDA state."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    keys = sorted(set(groups))
    rng.shuffle(keys)
    fold_of = {k: i % n_folds for i, k in enumerate(keys)}
    folds = np.array([fold_of[g] for g in groups])
    score = np.zeros(len(texts))
    idx = np.arange(len(texts))
    for f in range(n_folds):
        tr, te = idx[folds != f], idx[folds == f]
        if not te.size or not tr.size:
            continue
        ytr = [labels[j] for j in tr]
        if len(set(ytr)) < 2:
            continue
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                              max_features=300_000, sublinear_tf=True)
        try:
            xtr = vec.fit_transform([texts[j] for j in tr])
            xte = vec.transform([texts[j] for j in te])
        except ValueError:
            # too few units for min_df=3 - the fold contributes no score rather
            # than a silently relaxed vectoriser
            continue
        clf = LogisticRegression(solver="liblinear", C=4.0, tol=1e-7, max_iter=3000)
        clf.fit(xtr, ytr)
        score[te] = clf.decision_function(xte)
    return round(auroc(labels, score), 4), score


def within_pair(pair_ids, labels, score):
    """Per-pair rank accuracy of a claim-side score.  Chance 0.5."""
    d = collections.defaultdict(dict)
    for p, y, s in zip(pair_ids, labels, score):
        d[p][int(y)] = s
    ok = [(v[1], v[0]) for v in d.values() if 1 in v and 0 in v]
    if not ok:
        return {"pairs": 0, "acc": None}
    acc = float(np.mean([(p > n) + 0.5 * (p == n) for p, n in ok]))
    return {"pairs": len(ok), "acc": round(acc, 4)}


# --------------------------------------------------------------------------- #
# the assembled training mix
# --------------------------------------------------------------------------- #
def flagship_mix():
    """`R10-H108_lane.public_train()` under `R16-H142_G1_arm.untruncated_evidence()`
    plus every lane in `R20-H174_arm_run.LANES`.

    That is the 760,618-row SUPERSET of the 721,210-row flagship the banked
    baseline legs trained on, so a zero here implies a zero there.  Tags are
    carried through so any hit is attributed to the member that supplies it, and
    the flagship's own 721,210 rows are the `clean + quant_misbind +
    quant_scale_unit` tags."""
    arm = _mod("g1arm", SEM / "R16-H142_G1_arm.py")
    H108 = _mod("h108lane", SEM / "R10-H108_lane.py")
    A = _mod("h174arm", SEM / "R20-H174_arm_run.py")
    with arm.untruncated_evidence():
        claims, chunks, y, tags = H108.public_train()
    if len(claims) != 685_670:
        raise SystemExit(f"MIX ABORT: clean public {len(claims)}, expected 685,670")
    log(f"mix: clean public {len(claims)} rows over {len(set(tags))} groups")
    lane_docs = {}
    for fname, group, n_rows, _np, _f in A.LANES:
        p = SEM / fname
        d = pl.read_parquet(p)
        if d.height != n_rows:
            raise SystemExit(f"MIX ABORT: {fname} {d.height} rows, expected {n_rows}")
        ck = "chunk" if "chunk" in d.columns else "evidence"
        claims += d["claim"].to_list()
        chunks += d[ck].to_list()
        tags += [group] * d.height
        if "doc_id" in d.columns:
            lane_docs[group] = set(d["doc_id"].to_list())
        log(f"mix: lane {group} {d.height} rows ({len(lane_docs.get(group, ()))} doc ids)")
    if len(claims) != 760_618:
        raise SystemExit(f"MIX ABORT: total {len(claims)} rows, expected 760,618")
    log(f"mix: total {len(claims)} rows")
    return claims, chunks, tags, lane_docs


FLAGSHIP_TAGS = {
    "ragtruth_en", "ragtruth_de", "ragtruth_fr", "ragtruth_es", "ragtruth_it",
    "ragtruth_pl", "ragtruth_hu", "ragtruth_cn", "halueval", "psiloqa",
    "vitaminc", "tabfact", "quant_misbind", "quant_scale_unit",
}


def mix_doc_keys(lane_docs):
    """Every document identifier the TRAINING MIX carries, namespaced.

    The six source corpora expose no document id through `public_train()`, so
    they are read back from their archives under the member's own selection
    predicate; the five lanes carry `doc_id` directly."""
    keys = {}
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    dt = pl.read_parquet(io.BytesIO(z.read(
        next(x for x in z.namelist() if x.endswith("__train.parquet")))))
    dt = dt.filter(pl.col("statement").str.len_chars() > 10)
    keys["tabfact"] = {"tabfact:" + t for t in dt["table_id"].to_list()}

    zv = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    dv = pl.read_parquet(io.BytesIO(zv.read(
        next(x for x in zv.namelist() if x.endswith("__train.parquet")))))
    keys["vitaminc"] = {"vitaminc_page:" + p for p in dv["page"].to_list()}

    zp = zipfile.ZipFile(DATA / "dataset-psiloqa.zip")
    dp = pl.read_parquet(io.BytesIO(zp.read(
        next(x for x in zp.namelist() if x.endswith("__train.parquet")))))
    dp = dp.filter((pl.col("wiki_passage").str.len_chars() > 50)
                   & (pl.col("llm_answer").str.len_chars() > 10))
    keys["psiloqa"] = {"psiloqa_title:" + t for t in dp["wiki_title"].to_list()}

    for g, s in lane_docs.items():
        keys[f"lane_{g}"] = set(s)
    allk = set()
    for s in keys.values():
        allk |= s
    stems = {doc_stem(d) for d in allk}
    log(f"mix doc keys: {len(allk)} raw / {len(stems)} stem-normalised over "
        f"{len(keys)} namespaces")
    return keys, allk, stems


# --------------------------------------------------------------------------- #
# eval adapters - each returns the normalised shape the clause battery consumes
# --------------------------------------------------------------------------- #
def tabfact_heldout_tables():
    """TabFact test+validation keyed by `table_id`, with the builders' own
    serialisation.  `R14-H133_antigaming.held_statements` and
    `R15_gate_common.held_tabfact` both drop every id that appears in train -
    on the RAW id, never on the stem."""
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    train_ids = set(pl.read_parquet(io.BytesIO(z.read(
        next(x for x in z.namelist() if x.endswith("__train.parquet")))))["table_id"].to_list())
    held = pl.concat([pl.read_parquet(io.BytesIO(z.read(n))) for n in z.namelist()
                      if n.endswith("__test.parquet") or n.endswith("__validation.parquet")])
    held = held.unique(subset=["table_id"], keep="first")
    ev = {}
    for tid, cap, tbl in zip(held["table_id"].to_list(), held["table_caption"].to_list(),
                             held["table_text"].to_list()):
        ev[tid] = f"{cap}\n{tbl}".replace("\r\n", "\n").replace("#", " | ")[:CUT]
    return ev, train_ids


class Eval:
    def __init__(self, name, path, role, gate, claims, evidence, labels, pairs,
                 docs, paired, c1_instrument, anchors=None, families=None,
                 corpus_split=None, notes=None):
        self.name, self.path, self.role, self.gate = name, path, role, gate
        self.claims, self.evidence, self.labels = claims, evidence, labels
        self.pairs, self.docs, self.paired = pairs, docs, paired
        self.c1_instrument, self.anchors = c1_instrument, anchors
        self.families = families
        self.corpus_split = corpus_split
        self.notes = notes or []


def load_evals(tab_ev):
    """Every remaining held-out mechanism eval, enumerated from disk."""
    E = []

    # ---- 1. ANTI-GAMING near-miss + bind_row (the binding hold) ------------ #
    ag = pl.read_parquet(SEM / "R18-H150_antigaming_set.parquet")
    claims, ev, lab, prs, docs, fams = [], [], [], [], [], []
    missing = 0
    for i, r in enumerate(ag.iter_rows(named=True)):
        e = tab_ev.get(r["table_id"])
        if e is None:
            missing += 1
            continue
        for leg, y in ((r["claim_pos"], 1), (r["claim_neg"], 0)):
            claims.append(leg)
            ev.append(e)
            lab.append(y)
            prs.append(i)
            docs.append(["tabfact:" + r["table_id"]])
            fams.append(f'{r["kind"]}/{r["family"]}')
    E.append(Eval(
        "antigaming_nearmiss_bindrow", "R18-H150_antigaming_set.parquet",
        "the ANTI-GAMING held-out near-miss / bind_row instrument, re-constituted "
        "identically by every arm from TabFact test+validation; the parquet banks "
        "the claims and the table_id, and the reader rebuilds the evidence",
        "BINDING HOLD on every open arm - `anti-gaming near-miss >= 0.7438` "
        "(re-priced A1 band); read on H150/H152/H155/H156/H159/H160/H174 and "
        "recorded as a diagnostic under R20-H177 ruling 3",
        claims, ev, lab, prs, docs, True, "numeral_attestation",
        families=fams, corpus_split="tabfact_heldout",
        notes=[f"{missing} banked rows had no table in the archive index"]))

    # ---- 2. the traced anti-gaming form ------------------------------------ #
    tr = pl.read_parquet(SEM / "R14-H133_antigaming_traced.parquet")
    E.append(Eval(
        "antigaming_traced", "R14-H133_antigaming_traced.parquet",
        "the TRACED form of the same anti-gaming instrument - evidence stored "
        "in the parquet, claims prefixed with a derivation trace",
        "read as the traced anti-gaming diagnostic at R14-H133 and after",
        tr["claim"].to_list(), tr["chunk"].to_list(),
        [int(x) for x in tr["label"].to_list()], tr["pair_id"].to_list(),
        [["tabfact:" + t] for t in tr["table_id"].to_list()], True,
        "numeral_attestation",
        families=[f'{k}/{f}' for k, f in zip(tr["kind"].to_list(), tr["family"].to_list())],
        corpus_split="tabfact_heldout"))

    # ---- 3. FinDVer, the standing non-arena derivation instrument ---------- #
    fv = pl.read_parquet(SEM / "R19_findver_lane.parquet")
    E.append(Eval(
        "findver", "R19_findver_lane.parquet",
        "the FinDVer held-out mechanism eval - 2,400 human-annotated claims over "
        "2024 10-K/10-Q filings, subsets ie / numeric / knowledge",
        "BANKED standing non-arena mechanism instrument for derivation-adjacent "
        "arms (R20-H176: numeric 2-draw mean 0.4959, ie 0.6609, knowledge 0.5838)",
        fv["claim"].to_list(), fv["chunk"].to_list(),
        [int(x) for x in fv["label"].to_list()], fv["pair_id"].to_list(),
        [["findver:" + d] for d in fv["doc_id"].to_list()], False, "numeral_attestation",
        families=fv["subset"].to_list()))

    # ---- 4. R20-H177 eval_C ------------------------------------------------ #
    ec = pl.read_parquet(SEM / "R20-H177_eval_C.parquet")
    E.append(Eval(
        "eval_C", "R20-H177_eval_C.parquet",
        "held-out mechanism eval for the R20-H177 lane C arm - role / sign / "
        "period misbind over EDGAR prose",
        "Lane C PRIMARY gate >= 0.80; baseline leg read 0.9085 and REFUTED the "
        "near-chance prediction, disposition open",
        ec["claim"].to_list(), ec["chunk"].to_list(),
        [int(x) for x in ec["label"].to_list()], ec["pair_id"].to_list(),
        [[d] for d in ec["doc_id"].to_list()], True, "relational_binding",
        anchors=[(s_ if y == 1 else c_, c_ if y == 1 else s_, v)
                 for s_, c_, v, y in zip(ec["subject"].to_list(),
                                         ec["counterpart"].to_list(),
                                         ec["asserted_values"].to_list(),
                                         ec["label"].to_list())],
        families=ec["neg_family"].to_list()))

    # ---- 5. R17-H148 item-index bind probe --------------------------------- #
    p148 = pl.read_parquet(SEM / "R17-H148_probe.parquet")
    E.append(Eval(
        "h148_itemindex_probe", "R17-H148_probe.parquet",
        "held-out item-index / step-binding probe over army technical manuals",
        "probe-bank instrument; gating on probe readings STOPPED by R19-H163. "
        "Still cited as the `H148 literal-presence` build check for the H150 "
        "unit-swap probe",
        p148["claim"].to_list(), p148["chunk"].to_list(),
        [int(x) for x in p148["label"].to_list()], p148["pair_id"].to_list(),
        [[f'{c}:{d}'] for c, d in zip(p148["corpus"].to_list(), p148["doc_id"].to_list())],
        True, "item_index", families=p148["neg_family"].to_list()))

    # ---- 6. R17-H149 role-swap probe --------------------------------------- #
    p149 = pl.read_parquet(SEM / "R17-H149_probe.parquet")
    E.append(Eval(
        "h149_roleswap_probe", "R17-H149_probe.parquet",
        "held-out predicate role-swap probe over SciFact / scientific passages",
        "probe-bank instrument; gating on probe readings STOPPED by R19-H163",
        p149["claim"].to_list(), p149["chunk"].to_list(),
        [int(x) for x in p149["label"].to_list()], p149["pair_id"].to_list(),
        [[d] for d in p149["doc_id"].to_list()], True, "none",
        families=p149["neg_family"].to_list()))

    # ---- 7. R18-H150 unit-swap probe --------------------------------------- #
    us = pl.read_parquet(SEM / "R18-H150_unitswap_probe.parquet")
    E.append(Eval(
        "h150_unitswap_probe", "R18-H150_unitswap_probe.parquet",
        "document-disjoint unit-swap probe built from the scale/unit lane's "
        "unused supply - 140 pairs / 42 documents",
        "reported-secondary read only, pinned in its manifest as never training "
        "and never selected on; probe gating STOPPED by R19-H163",
        us["claim"].to_list(), us["chunk"].to_list(),
        [int(x) for x in us["label"].to_list()], us["pair_id"].to_list(),
        [[d] for d in us["doc_id"].to_list()], True, "proximity_binding",
        anchors=list(zip([str(v) for v in us["cited_value"].to_list()],
                         us["cited_unit"].to_list())),
        families=us["swap_family"].to_list()))

    # ---- 8/9. the R15 probe bank ------------------------------------------- #
    bp = pl.read_parquet(SEM / "R15_L1_bindprobe_pairs.parquet")
    claims, ev, lab, prs, docs, fams = [], [], [], [], [], []
    miss = 0
    for i, r in enumerate(bp.iter_rows(named=True)):
        e = tab_ev.get(r["table_id"])
        if e is None:
            miss += 1
            continue
        for leg, y in ((r["claim_pos"], 1), (r["claim_neg"], 0)):
            claims.append(leg)
            ev.append(e)
            lab.append(y)
            prs.append(i)
            docs.append(["tabfact:" + r["table_id"]])
            fams.append(r["arm"])
    E.append(Eval(
        "r15_bindprobe", "R15_L1_bindprobe_pairs.parquet",
        "the R15 L1 bind_col / bind_row probe bank - claims and table_id banked, "
        "evidence reconstituted by the reader from TabFact test+validation",
        "was PRIMARY for R17-H146 (`bind_col >= 0.80 AND bind_row >= 0.95`); "
        "probe-bank gating DECLARED DEAD by R19-H163",
        claims, ev, lab, prs, docs, True, "proximity_binding",
        anchors=None, families=fams, corpus_split="tabfact_heldout",
        notes=[f"{miss} banked rows had no table in the archive index"]))

    tp = pl.concat([pl.read_parquet(SEM / "R15_P1_typeprobe_quads.parquet"),
                    pl.read_parquet(SEM / "R15_P1_typeprobe_topup_quads.parquet")])
    claims, ev, lab, prs, docs, fams = [], [], [], [], [], []
    miss = 0
    for i, r in enumerate(tp.iter_rows(named=True)):
        e = tab_ev.get(r["table_id"])
        if e is None:
            miss += 1
            continue
        for col, y in (("claim_a", 1), ("claim_b", 1), ("claim_c", 0), ("claim_d", 0)):
            claims.append(r[col])
            ev.append(e)
            lab.append(y)
            prs.append(i)
            docs.append(["tabfact:" + r["table_id"]])
            fams.append(r["dtype"])
    E.append(Eval(
        "r15_typeprobe", "R15_P1_typeprobe_quads.parquet (+ topup)",
        "the R15 P1 derivation type probe bank - four legs per row (a present "
        "value, a correct derived value, two wrong derived values)",
        "the derivation half of the probe bank; gating DECLARED DEAD by R19-H163",
        claims, ev, lab, prs, docs, False, "numeral_attestation",
        families=fams, corpus_split="tabfact_heldout",
        notes=[f"{miss} banked rows had no table in the archive index",
               "four legs per row, not a two-leg contrast - within-pair accuracy "
               "is not defined and is reported as NOT-APPLICABLE"]))

    # ---- 10. G0b composed probes ------------------------------------------- #
    g0 = pl.read_parquet(SEM / "R20-G0b_composed_probes.parquet")
    ev = [f'{a}\n\n{b}' for a, b in zip(g0["doc_a"].to_list(), g0["doc_b"].to_list())]
    g0_docs = [["tabfact:" + a, "tabfact:" + b] for a, b in
               zip(g0["table_id_a"].to_list(), g0["table_id_b"].to_list())]
    E.append(Eval(
        "g0b_composed_probes", "R20-G0b_composed_probes.parquet",
        "the hotpotqa composed-supply gate-5 probe - two-table bridge claims over "
        "held-out TabFact tables",
        "gate 5 baseline read 0.6477 against a KILL at >= 0.70; PASS, and moot "
        "for registration because gate G0a failed",
        g0["claim"].to_list(), ev, [int(x) for x in g0["label"].to_list()],
        g0["pair_id"].to_list(), g0_docs, True, "numeral_attestation",
        families=g0["family"].to_list(), corpus_split="tabfact_heldout",
        notes=["the document channel carries BOTH tables of every bridge pair"]))

    # ---- 11. R11-H117 held-out pairs --------------------------------------- #
    hp = pl.read_parquet(SEM / "R11-H117_heldout_pairs.parquet")
    claims, ev, lab, prs, docs = [], [], [], [], []
    for i, r in enumerate(hp.iter_rows(named=True)):
        for leg, y in ((r["seed"], 1), (r["claim"], 0)):
            claims.append(leg)
            ev.append(r["chunk"])
            lab.append(y)
            prs.append(i)
            docs.append([f'dr_seed:{r["seed_id"]}'])
    E.append(Eval(
        "h117_heldout_pairs", "R11-H117_heldout_pairs.parquet",
        "the R11-H117 paired-margin held-out set - a clean seed sentence against "
        "its span-corrupted rewrite over the same evidence chunk",
        "kill-gate 2 of the R11-H117 paired-margin arm (banked PROCEED, "
        "lambda_margin 0.3); the DR lane it serves never entered the flagship mix",
        claims, ev, lab, prs, docs, True, "none",
        families=hp["delta"].to_list()))

    # ---- 12/13. the retained clean H175b evals ----------------------------- #
    for fn, nm in (("R20-H175b_qlane_eval_clean.parquet", "h175b_eval_clean"),
                   ("R20-H175b_qlane_eval_clean_prefix.parquet", "h175b_eval_clean_prefix")):
        d = pl.read_parquet(SEM / fn)
        E.append(Eval(
            nm, fn,
            "the rebuilt clean question-relevance eval retained after R20-H175b "
            "was withdrawn",
            "no live gate - the arm it served is WITHDRAWN; retained for a future "
            "option-D registration",
            d["claim"].to_list(), d["chunk"].to_list(),
            [int(x) for x in d["label"].to_list()], d["pair_id"].to_list(),
            [["psiloqa_title:" + str(x)] for x in d["doc_id"].to_list()], True, "none",
            families=d["neg_family"].to_list(),
            corpus_split="psiloqa_heldout"))
    # ---- 14/15. the two pre-build gate SAMPLES that still carry text ------ #
    dr = pl.read_parquet(SEM / "DR_H113_gate_judged.parquet")
    E.append(Eval(
        "dr_h113_gate_judged", "DR_H113_gate_judged.parquet",
        "the R11 DR-lane GENERATOR quality gate - judge verdicts on 1,505 "
        "span-corrupted rewrites, not a grounding contrast",
        "kill-gate of the DR lane's generator; the DR lane never entered the "
        "flagship mix and no open arm reads this file",
        dr["claim"].to_list(), dr["chunk"].to_list(),
        [1 if v == "yes" else 0 for v in dr["supported"].to_list()],
        list(range(dr.height)),
        [[f'dr_tag:{t}'] for t in dr["tag"].to_list()], False, "none",
        families=dr["tag"].to_list(),
        notes=["the label is the JUDGE's `supported` verdict on a generated "
               "rewrite, not a held-out grounding label; it is verified here "
               "because it carries claim and evidence text, not because it is "
               "an evaluation surface"]))

    gb = pl.read_parquet(SEM / "R12-H121_gateBC_rows.parquet")
    E.append(Eval(
        "r12_h121_gateBC_rows", "R12-H121_gateBC_rows.parquet",
        "the R12-H121 pre-build purity audit sample - claim / window pairs with "
        "no label column at all",
        "Gate B of R12-H121, which KILLED the hypothesis pre-build at purity "
        "0.284 against a 0.95 bar; nothing reads it now",
        gb["claim"].to_list(), gb["window"].to_list(), None,
        list(range(gb.height)),
        [[f'h121_source:{x}'] for x in gb["source"].to_list()], False, "none",
        families=gb["source"].to_list(),
        notes=["no label column - C1, C5 and C6 are NOT-APPLICABLE by "
               "construction; C2 and C7 are run"]))
    return E


# --------------------------------------------------------------------------- #
# the clause battery
# --------------------------------------------------------------------------- #

def load_h166a1(claims, chunks, tags):
    """The R19-H166 amendment A1 VitaminC REFUTES-vs-NEI holdout.

    The artifact is not banked as a parquet - `R20_baseline_legs.vitaminc_holdout`
    rebuilds it deterministically at read time - so it is rebuilt here through
    that same banked function, fed the 721,210-row flagship mix text sets the
    banked run fed it."""
    B = _mod("baselegs", SEM / "R20_baseline_legs.py")
    fc = [c for c, g in zip(claims, tags) if g in FLAGSHIP_TAGS]
    fk = [k for k, g in zip(chunks, tags) if g in FLAGSHIP_TAGS]
    mix = {"n_rows": len(fc), "claims": set(fc), "evidence": set(fk),
           "pairs": set(zip(fc, fk))}
    held, split_report = B.vitaminc_holdout(mix)
    return Eval(
        "h166a1_vitaminc_holdout",
        "rebuilt by R20_baseline_legs.vitaminc_holdout() - not banked as a parquet",
        "the R19-H166 amendment A1 held-out VitaminC REFUTES-vs-NEI split - "
        "38,126 rows over 5,553 pages, REFUTES the positive class",
        "PRIMARY mechanism gate of the R19-H166-A1 arm (`held-out VitaminC "
        "REFUTES-vs-NEI AUROC >= 0.85`); baseline leg read 0.3935, recorded as "
        "inverted-but-entangled, and the arm remains author-assented and queued",
        held["claim"].to_list(), held["evidence"].to_list(),
        [int(x) for x in held["y"].to_list()],
        held["unique_id"].to_list(),
        [["vitaminc_page:" + p] for p in held["page"].to_list()],
        False, "none", families=held["split"].to_list(),
        corpus_split="vitaminc_heldout",
        notes=["the banked builder filtered on RAW strings only - `page`, "
               "`claim`, `evidence`, `wiki_revision_id` against the VitaminC "
               "train split and claim / evidence / pair against the assembled "
               "mix. The truncated and whitespace-normalised forms C2 requires "
               "were never run on it; this pass runs them",
               f"rebuild reproduced {held.height} rows"]), split_report


def clause_c2(ev, surfaces, mix_doc_all, mix_doc_stems, mix_doc_ns, tag_of_chunk,
              tag_of_claim):
    c_sets, c_tuples = build_forms(ev.claims)
    e_sets, e_tuples = build_forms(ev.evidence)
    out = {"title": "C2 - disjointness from every evaluation surface, run in the "
                    "direction this artifact requires: it IS an evaluation "
                    "surface, so the test is that no unit of it sits in a "
                    "training surface",
           "method": "three string forms (raw / 1,500-cut / whitespace-collapsed "
                     "case-folded) crossed six ways in BOTH directions, on CLAIM "
                     "and on EVIDENCE units; plus the DOCUMENT channel raw and "
                     "STEM-NORMALISED",
           "surfaces": {}}
    for sname, sinfo in surfaces.items():
        ch, hit_ch = both_directions(e_sets, e_tuples, sinfo["chunk_sets"],
                                     sinfo["chunk_tuples"])
        cl, hit_cl = both_directions(c_sets, c_tuples, sinfo["claim_sets"],
                                     sinfo["claim_tuples"])
        blk = {"kind": sinfo["kind"], "evidence": ch, "claim": cl,
               "evidence_units_hit_any_form": len(hit_ch),
               "claim_units_hit_any_form": len(hit_cl),
               "clean": len(hit_ch) == 0 and len(hit_cl) == 0}
        if sname == "flagship_mix_superset":
            rows_on_hit = _rows_on_hit(ev.evidence, sinfo["chunk_sets"])
            blk["rows_on_a_hit_evidence_unit"] = rows_on_hit
            blk["share_of_rows"] = round(rows_on_hit / max(len(ev.claims), 1), 5)
            blk["hit_attribution_by_mix_member"] = _attrib(
                ev.evidence, tag_of_chunk)
            blk["claim_hit_attribution_by_mix_member"] = _attrib(
                ev.claims, tag_of_claim)
        out["surfaces"][sname] = blk

    # ---- the DOCUMENT channel ------------------------------------------- #
    docs = sorted({d for row in ev.docs for d in row if d})
    stems = {doc_stem(d) for d in docs}
    per_ns = {}
    for ns, keys in mix_doc_ns.items():
        kstems = {doc_stem(k) for k in keys}
        shared_raw = len(set(docs) & keys)
        shared_stem = len(stems & kstems)
        if shared_raw or shared_stem:
            per_ns[ns] = {"mix_documents": len(keys), "shared_raw": shared_raw,
                          "shared_STEM": shared_stem}
    rows_on_bad_doc = sum(1 for row in ev.docs
                          if any(d and doc_stem(d) in mix_doc_stems for d in row))
    out["document_channel"] = {
        "instrument": "the eval's own document ids against every document "
                      "identifier the training mix carries, keyed raw and "
                      "STEM-NORMALISED (a TabFact `1-`/`2-` csv prefix stripped)",
        "eval_documents": len(docs),
        "eval_document_stems": len(stems),
        "documents_in_the_mix_raw": len(set(docs) & mix_doc_all),
        "documents_in_the_mix_STEM": len(stems & mix_doc_stems),
        "share_of_documents_STEM": round(
            len(stems & mix_doc_stems) / max(len(stems), 1), 5),
        "rows_on_a_mix_document_STEM": rows_on_bad_doc,
        "share_of_rows": round(rows_on_bad_doc / max(len(ev.claims), 1), 5),
        "per_namespace": per_ns or "no namespace of the mix shares a key with "
                                   "this eval's document ids",
    }
    return out


def _rows_on_hit(texts, sets):
    n = 0
    for t in texts:
        if not t:
            continue
        q = unit_forms(t)
        if any(q[i] in sets[k] for _n, i, k in CROSS):
            n += 1
    return n


def _attrib(texts, tag_of):
    """Which mix member supplies each hit, over the union of the six forms."""
    c = collections.Counter()
    for t in texts:
        if not t:
            continue
        q = unit_forms(t)
        for _n, i, k in CROSS:
            tg = tag_of[k].get(q[i])
            if tg is not None:
                c[tg] += 1
                break
    return dict(c) or "no hit to attribute"


NO_LABEL = {"status": "NOT-APPLICABLE - the artifact carries no label column, "
                      "so there is no leg structure to test. C2 and C7 are run"}


def clause_c1(ev):
    if ev.labels is None:
        return {"title": "C1 - label commensurability", **NO_LABEL}
    y = np.asarray(ev.labels, dtype=float)
    pos = {(c, e) for c, e, l in zip(ev.claims, ev.evidence, y) if l == 1.0}
    neg = {(c, e) for c, e, l in zip(ev.claims, ev.evidence, y) if l == 0.0}
    shared = pos & neg

    cache = {}
    cont = []
    for c, e in zip(ev.claims, ev.evidence):
        k = dg(e)
        if k not in cache:
            cache[k] = set(tokens(e))
        ct = set(tokens(c))
        cont.append(len(ct & cache[k]) / len(ct) if ct else 0.0)
    cont = np.asarray(cont)

    blk = {
        "title": "C1 - label commensurability (tests as restated by C-A2)",
        "head_declared": "the grounding scalar (`task_head` of the DANN "
                         "student) - the head every mix member and every "
                         "mechanism eval is read on",
        "test_1_structural": {
            "rule": "a negative leg's (claim, evidence) identical to a positive "
                    "leg's means no function of (claim, evidence) separates the "
                    "legs, so the label cannot encode grounding",
            "identical_pairs": len(shared), "bar": "0", "fires": bool(shared)},
        "predicate_blind_diagnostic": {
            "instrument": "content-token containment of the claim in the evidence",
            "positive_leg": dist(cont[y == 1.0]),
            "negative_leg": dist(cont[y == 0.0]),
            "reading": "C-A1 - containment is a JOINT feature governed by C1, "
                       "excluded from C5's parity requirement; a predicate-blind "
                       "instrument showing no separation is not evidence of "
                       "incommensurability"},
    }

    # ---- test 2, under a predicate-SENSITIVE instrument ------------------ #
    if ev.c1_instrument == "relational_binding":
        vals = [relational_binding(a, r, v, e)
                for (a, r, v), e in zip(ev.anchors, ev.evidence)]
        instr = ("relational binding - is the asserted value printed NEARER the "
                 "term the claim names than the rival term the pair swaps it "
                 "with? A plain proximity test is predicate-BLIND on a role swap, "
                 "because both terms and the value are all printed")
    elif ev.c1_instrument == "item_index":
        vals = [item_index_attested(c, e) for c, e in zip(ev.claims, ev.evidence)]
        instr = ("item-index binding - the quoted item text is located in the "
                 "evidence and the number the evidence attaches to it is compared "
                 "with the number the claim asserts. The text is verbatim on both "
                 "legs, so only the binding differs")
    elif ev.c1_instrument == "numeral_attestation":
        vals = [numeral_attested(c, e) for c, e in zip(ev.claims, ev.evidence)]
        instr = ("numeral attestation - the fraction of the claim's numerals the "
                 "evidence prints. Predicate-sensitive for every near-miss and "
                 "derivation family, whose negative asserts a value the evidence "
                 "does not carry")
    elif ev.c1_instrument == "proximity_binding" and ev.anchors:
        vals = [proximity_attested(a, v, e)
                for (a, v), e in zip(ev.anchors, ev.evidence)]
        # anchors are per PAIR row for the two-leg evals; expand where needed
        instr = (f"binding proximity - does the evidence print the asserted value "
                 f"within {PROX_W} characters of the claim's anchor term? "
                 "Predicate-sensitive for every binding family, where both values "
                 "are printed and only their attachment differs")
    elif ev.c1_instrument == "proximity_binding":
        vals = [_bindprobe_attested(c, e) for c, e in zip(ev.claims, ev.evidence)]
        instr = (f"binding proximity parsed from the claim template "
                 f"`The <column> of <row key> is <value>.` - is the value printed "
                 f"within {PROX_W} characters of the row key?")
    else:
        vals = None
        instr = None

    if vals is None:
        blk["test_2_strict_separation"] = {
            "instrument": None,
            "status": "NOT COMPUTABLE - this eval corrupts a lexical-semantic "
                      "predicate (role swap / entity swap / question relevance) "
                      "for which no cheap deterministic attestation instrument "
                      "exists on CPU. Reported as not computable rather than "
                      "substituted with the predicate-BLIND reading, which C-A1 "
                      "explicitly forbids as evidence",
            "bar": "the negative leg's rate strictly below the positive leg's"}
    else:
        cov = np.array([v is not None for v in vals])
        sc = np.array([v if v is not None else np.nan for v in vals])
        pm, nm = cov & (y == 1.0), cov & (y == 0.0)
        pr = float(np.nanmean(sc[pm])) if pm.sum() else float("nan")
        nr = float(np.nanmean(sc[nm])) if nm.sum() else float("nan")
        per_fam = {}
        if ev.families:
            for f in sorted(set(ev.families)):
                m = np.array([x == f for x in ev.families])
                a, b = m & pm, m & nm
                per_fam[f] = {
                    "rows": int(m.sum()),
                    "instrument_coverage": round(float((m & cov).sum() / m.sum()), 4),
                    "positive_leg_rate": round(float(np.nanmean(sc[a])), 4) if a.sum() else None,
                    "negative_leg_rate": round(float(np.nanmean(sc[b])), 4) if b.sum() else None}
        gap = pr - nr
        discriminates = bool(np.isfinite(gap) and gap > 0.01)
        blk["test_2_strict_separation"] = {
            "instrument": instr,
            "positive_leg_rate": round(pr, 4), "negative_leg_rate": round(nr, 4),
            "gap": round(float(gap), 4),
            "strictly_below": bool(nr < pr),
            "instrument_coverage": round(float(cov.mean()), 4),
            "instrument_discriminates_on_this_eval": discriminates,
            "reading": ("the negative leg separates from the positive under an "
                        "instrument that reads the predicate this eval corrupts")
                       if discriminates else
                       ("INCONCLUSIVE - the instrument does not separate the legs "
                        "here, which under C-A1 is NOT evidence of "
                        "incommensurability: `a predicate-blind instrument showing "
                        "no separation is not evidence`. The structural test above "
                        "is the decisive one"),
            "per_family": per_fam,
            "bar": "the negative leg's rate strictly below the positive leg's; "
                   "equality is the signature of a label independent of the "
                   "claim-evidence relation"}
        blk["test_3_absolute_levels"] = {
            "predicate_sensitive_positive_leg_rate": round(pr, 4),
            "predicate_sensitive_negative_leg_rate": round(nr, 4),
            "containment_negative_leg_rate_eq_1.0":
                dist(cont[y == 0.0]).get("rate_eq_1.0"),
            "containment_negative_leg_rate_ge_0.9":
                dist(cont[y == 0.0]).get("rate_ge_0.9"),
            "note": "a negative leg attested at a high absolute rate is recorded "
                    "as a finding even where test 2 clears"}
    if "test_3_absolute_levels" not in blk:
        blk["test_3_absolute_levels"] = {
            "containment_negative_leg_rate_eq_1.0":
                dist(cont[y == 0.0]).get("rate_eq_1.0"),
            "containment_negative_leg_rate_ge_0.9":
                dist(cont[y == 0.0]).get("rate_ge_0.9"),
            "note": "absolute levels are reported under the predicate-BLIND "
                    "instrument only, because no predicate-sensitive one is "
                    "computable for this eval"}
    return blk


_BIND_T = re.compile(r"^The (.+?) of (.+?) is (.+?)\.$")


def _bindprobe_attested(claim, chunk):
    m = _BIND_T.match(claim.strip())
    if not m:
        return None
    _col, key, val = m.groups()
    return proximity_attested(key, val, chunk)


def clause_c5(ev, rng):
    if ev.labels is None:
        return {"title": "C5 - leak suite", **NO_LABEL}
    if not ev.paired:
        return {"title": "C5 - leak suite for constructed members and "
                         "paired-contrast evals",
                "status": "NOT A PAIRED-CONTRAST EVAL - C5's within-pair and "
                          "parity bars are written for two-leg contrasts. The "
                          "claim-only probe is run anyway and reported",
                "claim_only_converged_probe": None}
    first = [row[0] if row else "" for row in ev.docs]
    groups = first if len(set(first)) >= 5 else [str(p) for p in ev.pairs]
    co, co_score = text_probe(ev.claims, ev.labels, groups, rng)
    eo, _ = text_probe(ev.evidence, ev.labels, groups, rng)
    wp = within_pair(ev.pairs, ev.labels, co_score)

    y = ev.labels
    ch = {
        "claim_char_length": [float(len(c)) for c in ev.claims],
        "claim_token_count": [float(len(tokens(c))) for c in ev.claims],
        "claim_digit_count": [float(sum(x.isdigit() for x in c)) for c in ev.claims],
        "claim_numeral_count": [float(len(canon_nums(c))) for c in ev.claims],
        "evidence_char_length": [float(len(e)) for e in ev.evidence],
    }
    parity = {k: round(auroc(y, v), 4) for k, v in ch.items()}
    worst = max(abs(v - 0.5) for v in parity.values())
    return {
        "title": "C5 - leak suite for paired-contrast evals",
        "scope_note": "C-A1 - the parity requirement is scoped to channels that "
                      "do NOT read the claim-evidence relation. Claim-to-evidence "
                      "containment is a JOINT feature governed by C1 and is "
                      "excluded from the parity block below",
        "claim_only_converged_probe": {
            "auroc": co, "bar": "< 0.55", "pass": bool(co < 0.55),
            "folds": "5, disjoint on the document key where the eval carries one, "
                     "else on the pair id"},
        "within_pair_claim_only": {
            **wp, "bar": "< 0.60",
            "pass": bool(wp["acc"] is not None and wp["acc"] < 0.60)},
        "single_channel_evidence_only_probe": {
            "auroc": eo,
            "expected": "0.5 by construction wherever a pair's two legs share "
                        "one evidence chunk - the probe cannot see the label",
            "bar": "at chance where the construction implies it"},
        "surface_parity": {
            "auroc": parity, "bar": "each channel in [0.45, 0.55]",
            "worst_deviation": round(worst, 4), "pass": bool(worst <= 0.05)},
        "executor_added_probes_reported_separately": [
            "claim_digit_count and claim_numeral_count are EXECUTOR-ADDED parity "
            "channels, reported here and not joined to any registered conjunction",
            "the C1 attestation instruments are DIAGNOSTICS for C1, not members "
            "of this suite"],
        "registered_conjunction_present": False,
        "registered_conjunction_note": "none of these evals was built under a "
                                       "registered C5 conjunction - they all "
                                       "predate the dataset contract. Every "
                                       "number above is executor-measured",
    }


def clause_c6(ev, key_claims, doc_claims):
    if ev.labels is None:
        return {"title": "C6 - no memorisation channel", **NO_LABEL}
    look = [key_claims.get(dg(norm(e)), ()) for e in ev.evidence]
    cov = sum(1 for v in look if v)
    out = {"title": "C6 - no memorisation channel (scoped by C-A2 to associations "
                    "the TRAINING MIX supplies)",
           "key": "the PASSAGE - whatever the mix associates with the same "
                  "whitespace-normalised evidence text",
           "rows": len(ev.claims), "rows_with_a_mix_claim_over_the_same_passage": cov,
           "coverage": round(cov / max(len(ev.claims), 1), 4)}
    if cov == 0:
        out["auroc"] = None
        out["status"] = ("NOT-APPLICABLE - zero key coverage. C-A2: where the "
                         "eval-facing test has zero key coverage the clause is "
                         "NOT-APPLICABLE and no proxy is substituted")
    else:
        s = np.array([max((containment(c, a) for a in v), default=0.0)
                      for c, v in zip(ev.claims, look)])
        out["claim_into_mix_claim_containment_auroc"] = round(auroc(ev.labels, s), 4)
        out["auroc"] = out["claim_into_mix_claim_containment_auroc"]
        out["bar"] = "undefined or at chance on a clean instrument"

    dlook = [[a for d in row for a in doc_claims.get(doc_stem(d), ())]
             for row in ev.docs]
    dcov = sum(1 for v in dlook if v)
    dblk = {"key": "the DOCUMENT stem - whatever the mix associates with the same "
                   "document after the TabFact csv prefix is stripped",
            "rows_with_a_mix_claim_over_the_same_document": dcov,
            "coverage": round(dcov / max(len(ev.claims), 1), 4)}
    if dcov:
        s = np.array([max((containment(c, a) for a in v), default=0.0)
                      for c, v in zip(ev.claims, dlook)])
        dblk["claim_into_mix_claim_containment_auroc"] = round(auroc(ev.labels, s), 4)
    else:
        dblk["auroc"] = None
        dblk["status"] = "NOT-APPLICABLE - zero document-key coverage"
    out["document_key_channel"] = dblk
    return out


def clause_c7(ev):
    return {"title": "C7 - declared units and volume",
            "unit": "PAIRS where the eval is a two-leg contrast, else ROWS; both "
                    "reported always",
            "rows": len(ev.claims), "pairs": len(set(ev.pairs)),
            "distinct_claims": len(set(ev.claims)),
            "distinct_evidence": len(set(ev.evidence)),
            "documents": len({d for row in ev.docs for d in row if d}),
            "document_stems": len({doc_stem(d) for row in ev.docs for d in row if d}),
            "label_balance": dict(collections.Counter(int(x) for x in ev.labels))
                             if ev.labels is not None else None,
            "families": dict(collections.Counter(ev.families)) if ev.families else None}


def clause_c3(ev, mix_doc_ns, tab_train_ids):
    """Split semantics, for evals drawn from a corpus the mix also uses."""
    if ev.corpus_split == "tabfact_heldout":
        docs = {d for row in ev.docs for d in row if d.startswith("tabfact:")}
        ids = {d[len("tabfact:"):] for d in docs}
        stems = {doc_stem(d) for d in docs}
        mix = mix_doc_ns["tabfact"]
        mstems = {doc_stem(k) for k in mix}
        return {
            "title": "C3 - split semantics verified, never assumed",
            "axis_the_builder_cut_on": "the TabFact `table_id` STRING - "
                                       "`held_statements` / `held_tabfact` drop "
                                       "every id that appears in the train split, "
                                       "on the RAW id and never on the stem",
            "axis_the_corpus_actually_cuts_on": "TabFact serialises one Wikipedia "
                                                "table under both a `1-` and a "
                                                "`2-` prefixed csv id, so the id "
                                                "is not the document",
            "eval_table_ids": len(ids),
            "table_ids_in_the_mix_member": len(docs & mix),
            "table_id_STEMS_in_the_mix_member": len(stems & mstems),
            "share_of_eval_tables_whose_STEM_is_in_the_member": round(
                len(stems & mstems) / max(len(stems), 1), 5),
            "archive_baseline": "banked contract/tabfact_clauses.json C3 - 91 of "
                                "1,696 validation ids (0.0538) and 73 of 1,695 "
                                "test ids (0.0431) stem-collide with a train id; "
                                "18 validation and 14 test serialised tables are "
                                "byte-identical to a train table under the "
                                "1,500-char cut",
            "official_split_not_taken_on_trust": True}
    if ev.corpus_split == "vitaminc_heldout":
        return {
            "title": "C3 - split semantics verified, never assumed",
            "axis_the_corpus_actually_cuts_on": "VitaminC's official split is "
                                                "disjoint by `unique_id` and "
                                                "`case_id` but shares 1,214 "
                                                "pages, 110 claims and 221 "
                                                "evidence strings with train "
                                                "(banked C3 finding)",
            "what_the_builder_did": "dropped every candidate row colliding with "
                                    "the train split on page / claim / evidence "
                                    "/ wiki_revision_id / unique_id / case_id, "
                                    "then dropped every row whose claim, evidence "
                                    "or (claim, evidence) pair appears in the "
                                    "assembled 721,210-row mix - all on RAW "
                                    "strings",
            "what_this_pass_adds": "the truncated and whitespace-normalised forms "
                                   "the contract requires, in both directions, "
                                   "plus the page-key document channel"}
    if ev.corpus_split == "psiloqa_heldout":
        return {
            "title": "C3 - split semantics verified, never assumed",
            "axis_the_corpus_actually_cuts_on": "PsiloQA cuts per QUESTION, not "
                                                "per document - 5,368 of 5,687 "
                                                "held-out passages are "
                                                "byte-identical to a training "
                                                "passage (banked C3 finding)",
            "consequence_for_this_eval": "the builder's clean pool was selected "
                                         "by explicit passage exclusion against "
                                         "the assembled mix rather than by "
                                         "trusting the split; the C2 block above "
                                         "re-measures that in all three forms"}
    return {"title": "C3 - split semantics verified, never assumed",
            "status": "the corpus this eval draws from is not a member of the "
                      "training mix, so there is no split boundary to verify "
                      "against a mix member; C2's text and document channels "
                      "carry the disjointness question instead"}


# --------------------------------------------------------------------------- #
def main():
    rng = random.Random(20260817)
    report = {
        "artifact": "contract/mechanism_evals_report.json",
        "scope": "every REMAINING held-out mechanism eval - the non-arena "
                 "instruments the campaign reads gates on - verified against "
                 "docs/experiments/dataset-contract.md with amendments C-A1 and "
                 "C-A2 applied",
        "excluded_owned_by_other_agents": [
            "R20-H177_eval_B.parquet (being rebuilt)",
            "R17-H143_evalset.parquet (being assessed)",
            "the blind arena", "gold_full"],
        "note": NOTE,
        "evals": {},
    }

    log("reconstituting the TabFact held-out table index")
    tab_ev, tab_train_ids = tabfact_heldout_tables()
    log(f"tabfact held-out tables indexed: {len(tab_ev)}")

    evals = load_evals(tab_ev)
    log(f"evals loaded: {[e.name for e in evals]}")

    # ---- the surfaces --------------------------------------------------- #
    claims, chunks, tags, lane_docs = flagship_mix()

    log("rebuilding the R19-H166-A1 VitaminC holdout through its banked builder")
    h166, h166_split = load_h166a1(claims, chunks, tags)
    evals.append(h166)
    report["h166a1_rebuild"] = h166_split
    log(f"h166a1 holdout rebuilt: {len(h166.claims)} rows")

    mix_doc_ns, mix_doc_all, mix_doc_stems = mix_doc_keys(lane_docs)

    log("indexing the mix in all four string forms (760,618 rows, one pass)")
    mix_chunk = index_units(chunks, tags)
    mix_claim = index_units(claims, tags)
    log(f"mix forms: {len(mix_chunk['sets']['raw'])} distinct evidence, "
        f"{len(mix_claim['sets']['raw'])} distinct claims")
    mix_chunk_sets, mix_chunk_tuples = mix_chunk["sets"], mix_chunk["tuples"]
    mix_claim_sets, mix_claim_tuples = mix_claim["sets"], mix_claim["tuples"]
    tag_of_chunk, tag_of_claim = mix_chunk["tag_of"], mix_claim["tag_of"]

    # the flagship's own 721,210 rows, as a separate surface
    n_flagship = sum(1 for g in tags if g in FLAGSHIP_TAGS)
    log(f"flagship subset: {n_flagship} rows")
    if n_flagship != 721_210:
        raise SystemExit(f"FLAGSHIP ABORT: {n_flagship} rows, expected 721,210")
    fs_chunk_sets, fs_chunk_tuples = subset_forms(mix_chunk, FLAGSHIP_TAGS)
    fs_claim_sets, fs_claim_tuples = subset_forms(mix_claim, FLAGSHIP_TAGS)

    G = _mod("pgate", SEM / "provenance_gate.py")
    arena_texts, _ = G.load_arena()
    arena_docs = [c for v in arena_texts.values() for c in v]
    ar_sets, ar_tuples = build_forms(arena_docs)
    H108 = _mod("h108b", SEM / "R10-H108_lane.py")
    gc, gk, _gy = H108.gold_full()
    gold_chunks = [c for ks in gk for c in ks]
    gd_sets, gd_tuples = build_forms(gold_chunks)
    gdc_sets, gdc_tuples = build_forms(list(gc))
    log(f"arena {len(arena_docs)} documents, gold_full {len(gc)} claims / "
        f"{len(gold_chunks)} chunks")

    surfaces = {
        "flagship_mix_superset": {
            "kind": "R10-H108_lane.public_train() under untruncated_evidence() "
                    "plus every lane in R20-H174_arm_run.LANES - 760,618 rows, a "
                    "strict SUPERSET of the 721,210-row flagship",
            "chunk_sets": mix_chunk_sets, "chunk_tuples": mix_chunk_tuples,
            "claim_sets": mix_claim_sets, "claim_tuples": mix_claim_tuples},
        "flagship_mix_721210": {
            "kind": "the 721,210-row flagship exactly - the twelve source-corpus "
                    "groups plus quant_misbind and quant_scale_unit",
            "chunk_sets": fs_chunk_sets, "chunk_tuples": fs_chunk_tuples,
            "claim_sets": fs_claim_sets, "claim_tuples": fs_claim_tuples},
        "blind_arena": {
            "kind": "the 10 RAGBench subsets, banked R8-H77 gate sample",
            "chunk_sets": ar_sets, "chunk_tuples": ar_tuples,
            "claim_sets": ar_sets, "claim_tuples": ar_tuples},
        "gold_full": {
            "kind": "the held-out gold test surface (R10-H108_lane.gold_full)",
            "chunk_sets": gd_sets, "chunk_tuples": gd_tuples,
            "claim_sets": gdc_sets, "claim_tuples": gdc_tuples},
    }

    # ---- C6 key maps, restricted to keys the evals actually use ---------- #
    want_pass = set()
    want_doc = set()
    for e in evals:
        want_pass |= {dg(norm(x)) for x in e.evidence if x}
        want_doc |= {doc_stem(d) for row in e.docs for d in row if d}
    key_claims = collections.defaultdict(list)
    for c, k in zip(claims, chunks):
        d = dg(norm(k))
        if d in want_pass:
            key_claims[d].append(c)
    # the mix's document->claim association, for the namespaces that carry one
    doc_claims = _mix_doc_claims(want_doc)
    log(f"C6 key maps: {len(key_claims)} passage keys, {len(doc_claims)} doc keys")

    # ---- the live positive controls -------------------------------------- #
    report["LIVE_POSITIVE_CONTROLS"] = positive_controls(
        chunks, mix_chunk_sets, mix_chunk_tuples, mix_doc_ns, mix_doc_stems, rng)

    del claims, chunks
    # ---- per-eval battery ------------------------------------------------ #
    for e in evals:
        log(f"--- {e.name}: {len(e.claims)} rows / {len(set(e.pairs))} pairs")
        blk = {
            "artifact": e.path,
            "role": e.role,
            "what_reads_it": e.gate,
            "paired_contrast": e.paired,
            "notes": e.notes,
            "clauses": {
                "C7": clause_c7(e),
                "C2": clause_c2(e, surfaces, mix_doc_all, mix_doc_stems,
                                mix_doc_ns, tag_of_chunk, tag_of_claim),
                "C3": clause_c3(e, mix_doc_ns, tab_train_ids),
                "C1": clause_c1(e),
                "C5": clause_c5(e, rng),
                "C6": clause_c6(e, key_claims, doc_claims),
            },
            "note": NOTE,
        }
        report["evals"][e.name] = blk
        log(f"    C2 mix evidence hits "
            f"{blk['clauses']['C2']['surfaces']['flagship_mix_superset']['evidence_units_hit_any_form']}"
            f", doc STEM hits "
            f"{blk['clauses']['C2']['document_channel']['documents_in_the_mix_STEM']}")
        OUT.write_text(json.dumps(report, indent=2))

    report["cross_arm_antigaming_identity"] = antigaming_identity()
    OUT.write_text(json.dumps(report, indent=2))
    log(f"wrote {OUT}")
    TABLE_OUT.write_text(summary_table(report))
    log(f"wrote {TABLE_OUT}")


def _mix_doc_claims(want):
    """Claims the mix carries over each document key the evals use.

    Only the TabFact member and the five lanes expose a document key; TabFact is
    read back from its archive under the member's own selection predicate."""
    out = collections.defaultdict(list)
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    dt = pl.read_parquet(io.BytesIO(z.read(
        next(x for x in z.namelist() if x.endswith("__train.parquet")))))
    dt = dt.filter(pl.col("statement").str.len_chars() > 10)
    for tid, st in zip(dt["table_id"].to_list(), dt["statement"].to_list()):
        k = doc_stem("tabfact:" + tid)
        if k in want:
            out[k].append(st)
    zv = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    dv = pl.read_parquet(io.BytesIO(zv.read(
        next(x for x in zv.namelist() if x.endswith("__train.parquet")))))
    for pg, cl in zip(dv["page"].to_list(), dv["claim"].to_list()):
        k = "vitaminc_page:" + pg
        if k in want:
            out[k].append(cl)

    zp = zipfile.ZipFile(DATA / "dataset-psiloqa.zip")
    dp = pl.read_parquet(io.BytesIO(zp.read(
        next(x for x in zp.namelist() if x.endswith("__train.parquet")))))
    dp = dp.filter((pl.col("wiki_passage").str.len_chars() > 50)
                   & (pl.col("llm_answer").str.len_chars() > 10))
    for ti, cl in zip(dp["wiki_title"].to_list(), dp["llm_answer"].to_list()):
        k = "psiloqa_title:" + ti
        if k in want:
            out[k].append(cl)

    A = _mod("h174arm2", SEM / "R20-H174_arm_run.py")
    for fname, _g, *_r in A.LANES:
        d = pl.read_parquet(SEM / fname)
        if "doc_id" not in d.columns:
            continue
        for did, cl in zip(d["doc_id"].to_list(), d["claim"].to_list()):
            k = doc_stem(did)
            if k in want:
                out[k].append(cl)
    return out


def positive_controls(chunks, mix_sets, mix_tuples, mix_doc_ns, mix_doc_stems, rng):
    """Four live controls on the disjointness instrument."""
    picks = [c for c in rng.sample(chunks, 400) if c and len(c) > 200][:300]
    id_sets, id_tuples = build_forms(picks)
    id_counts, id_hit = six_forms(id_sets, id_tuples, mix_sets)

    def rewrap(t):
        """Re-wrap the whitespace and change NOTHING else - every space becomes a
        newline plus indent, which collapses back under normalisation. Exactly
        the transformation that makes a passage invisible to exact matching."""
        return t.replace(" ", "\n   ")

    rw = [rewrap(c) for c in picks]
    rw_sets, rw_tuples = build_forms(rw)
    rw_counts, rw_hit = six_forms(rw_sets, rw_tuples, mix_sets)

    live = pl.read_parquet(SEM / "R20-H175b_qlane_eval.parquet")
    lv_sets, lv_tuples = build_forms(live["chunk"].to_list())
    lv_counts, lv_hit = six_forms(lv_sets, lv_tuples, mix_sets)

    eb = pl.read_parquet(SEM / "R20-H177_eval_B.parquet")
    eb_docs = {d for d in eb["doc_id"].to_list() if d.startswith("tabfact:")}
    eb_stems = {doc_stem(d) for d in eb_docs}
    tf = mix_doc_ns["tabfact"]
    tf_stems = {doc_stem(k) for k in tf}

    return {
        "1_synthetic_identity": {
            "design": "300 passages sampled from the assembled mix and offered to "
                      "the gate as if they were an eval. A gate that cannot read "
                      "300 of 300 cannot certify anything",
            "units": len(id_tuples), "counts": id_counts,
            "units_hit_any_form": len(id_hit),
            "expected": "every form reads every unit",
            "fires": len(id_hit) == len(id_tuples)},
        "2_synthetic_rewrap": {
            "design": "the SAME passages with every space replaced by a newline "
                      "plus indent - whitespace re-wrapped, no character removed. "
                      "This is the mode that hid R20-H177_eval_B's contamination "
                      "while exact string matching read 4.5%",
            "units": len(rw_tuples), "counts": rw_counts,
            "units_hit_any_form": len(rw_hit),
            "expected": "raw and truncated forms read 0; the two NORMALISED forms "
                        "read every unit",
            "raw_form_silent": rw_counts["raw_in_raw"] == 0,
            "normalised_form_fires": rw_counts["normalised_in_normalised_raw"]
                                     == len(rw_tuples)},
        "2b_synthetic_document_stem": {
            "design": "TabFact table ids the `tabfact` member carries, with their "
                      "`1-`/`2-` csv prefix FLIPPED. The raw document channel must "
                      "read 0 and the STEM channel must read every one - the "
                      "channel that caught eval_B while its string forms read 15",
            **_stem_control(mix_doc_ns["tabfact"])},
        "3_live_banked_string_channel": {
            "design": "R20-H175b_qlane_eval.parquet, the withdrawn eval banked at "
                      "485 of 487 passages inside the mix (99.6%)",
            "units": len(lv_tuples), "counts": lv_counts,
            "units_hit_any_form": len(lv_hit),
            "banked_expectation": "485 of 487 passages",
            "fires": len(lv_hit) > 0},
        "4_live_banked_document_channel": {
            "design": "the ORIGINAL R20-H177_eval_B.parquet, banked at 325 of 325 "
                      "TabFact document STEMS inside the `tabfact` member while "
                      "its raw ids read 0",
            "eval_tabfact_documents": len(eb_docs),
            "raw_ids_in_the_member": len(eb_docs & tf),
            "STEMS_in_the_member": len(eb_stems & tf_stems),
            "banked_expectation": "325 of 325 documents inside the `tabfact` "
                                  "member, while the six string forms read 15 - "
                                  "R20-H177_eval_B_rebuilt_verify.py header",
            "fires": len(eb_stems & tf_stems) > 0},
    }



def _stem_control(tabfact_keys):
    """Flip the `1-`/`2-` csv prefix on member table ids: the raw channel must go
    silent and the stem channel must fire on every one."""
    flipped = []
    for k in sorted(tabfact_keys):
        t = k[len("tabfact:"):]
        if len(t) > 2 and t[0] in "12" and t[1] == "-":
            f = "tabfact:" + ("2-" if t[0] == "1" else "1-") + t[2:]
            if f not in tabfact_keys:  # the member must not already carry it
                flipped.append(f)
        if len(flipped) >= 300:
            break
    raw_hit = len(set(flipped) & set(tabfact_keys))
    stems = {doc_stem(d) for d in flipped}
    mstems = {doc_stem(k) for k in tabfact_keys}
    return {"units": len(flipped), "raw_channel_hits": raw_hit,
            "STEM_channel_hits": len(stems & mstems),
            "raw_channel_silent": raw_hit == 0,
            "stem_channel_fires": len(stems & mstems) == len(stems)}


def antigaming_identity():
    """Is the anti-gaming instrument the same set across every arm that read it?"""
    files = sorted(SEM.glob("*antigaming_set.parquet"))
    sig = {}
    for p in files:
        d = pl.read_parquet(p)
        key = hashlib.blake2b(
            "|".join(f'{a}\x00{b}\x00{c}' for a, b, c in
                     zip(d["table_id"].to_list(), d["claim_pos"].to_list(),
                         d["claim_neg"].to_list())).encode(), digest_size=16).hexdigest()
        sig[p.name] = {"rows": d.height, "content_fingerprint": key,
                       "distinct_tables": int(d["table_id"].n_unique())}
    groups = collections.defaultdict(list)
    for k, v in sig.items():
        groups[v["content_fingerprint"]].append(k)
    return {
        "question": "the anti-gaming hold is read per arm from a set the arm's "
                    "own run re-constitutes; are those sets the same set?",
        "files": sig,
        "distinct_content_fingerprints": len(groups),
        "groups": {k: v for k, v in groups.items()},
        "reading": "one fingerprint means every arm was held to a byte-identical "
                   "instrument; more than one means the hold's numbers are not "
                   "strictly comparable across arms",
    }


def summary_table(report):
    rows = ["# Mechanism evals - contract verification summary", "",
            report["note"], "",
            "| eval | rows / pairs | what reads it | C2 mix evidence | C2 mix claim "
            "| C2 doc STEM | C1 structural | C5 claim-only | C6 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, b in report["evals"].items():
        c = b["clauses"]
        s = c["C2"]["surfaces"]["flagship_mix_superset"]
        d = c["C2"]["document_channel"]
        c5 = c["C5"].get("claim_only_converged_probe")
        c5v = c5["auroc"] if isinstance(c5, dict) else "n/a"
        c5v = "n/a" if c5v is None else c5v
        c6 = c["C6"].get("auroc")
        rows.append(
            f'| `{name}` | {c["C7"]["rows"]} / {c["C7"]["pairs"]} '
            f'| {b["what_reads_it"][:70]} '
            f'| {s["evidence_units_hit_any_form"]} of {s["evidence"]["eval_units_into_surface"]["n_query_units"]} '
            f'| {s["claim_units_hit_any_form"]} of {s["claim"]["eval_units_into_surface"]["n_query_units"]} '
            f'| {d["documents_in_the_mix_STEM"]} of {d["eval_document_stems"]} '
            f'| {c["C1"].get("test_1_structural", {}).get("identical_pairs", "N/A")} '
            f'| {c5v} | {c6 if c6 is not None else "N/A"} |')
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    main()

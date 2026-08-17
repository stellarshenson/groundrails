"""R20-H175b STAGE 0 - question-relevance contrast lane, CPU only, no GPU.

Registered in docs/experiments/semantic-grounding-experiments.md, block "R20-H175b
QUESTION CONDITIONING (measurement only)", STAGE 0: "build a question-relevance
contrast lane: same evidence, same claim sentence, RIGHT question vs WRONG
question, label flipping on question-claim relevance alone ... wrong questions
drawn from the same corpus and same document register so the negative is not
detectable by topic novelty."

WHY THE LANE EXISTS
-------------------
Only ~97k of 721k mix rows carry a clean question field and NOTHING in the mix
teaches question RELEVANCE - no pair anywhere has a label that turns on whether
the question matches the claim.  A question-conditioned arm trained on that
supply can train the channel to a no-op and return an arena null that is
indistinguishable from "the channel does not help".  This lane is the
precondition that makes the stage-1 measurement attributable, not an enhancement
to it.

CONSTRUCTION - the question is the ONLY thing that moves
--------------------------------------------------------
A pair is two rows over the SAME evidence passage and the SAME claim sentence:

  label 1   question = the claim's own originating question
  label 0   question = a WRONG question - one the claim does not answer

The wrong question is drawn from the SAME Wikipedia passage: it is another
PsiloQA question over that very passage, with its own distinct golden answer.
Nothing about topic, entity, vocabulary, language or register changes between the
two legs - only which of the passage's questions is asked.

BALANCE IS STRUCTURAL, NOT ASSERTED
------------------------------------
A unit is one passage, a subset S of its distinct questions (|S| >= 2), and a
DERANGEMENT sigma on S (a permutation with no fixed point).  Pair i asserts
question q_i over claim a_i and corrupts it to q_sigma(i).  Every question in the
unit therefore appears EXACTLY ONCE as the true question and EXACTLY ONCE as the
wrong one, and every claim, every passage and every question is used the same
number of times on each label.

The consequence is provable and is re-measured in `verify()`: any statistic that
is a function of the claim alone, the evidence alone, or the question alone takes
the identical value in one label-1 row and one label-0 row, so its AUROC against
the label is exactly 0.5.  ONLY a question-x-claim INTERACTION statistic can
separate the classes - which is what the lane is for, and which is why the
selection below matches those interaction channels explicitly.

THE SELECTION - lexical overlap is matched, so only semantics is left
---------------------------------------------------------------------
The true question naturally shares more words with its own answer than a wrong
question does (unmatched gap +0.20 mean on question->claim containment, pooled
AUROC 0.62).  Left alone the lane would teach lexical overlap, not relevance.  So
for every passage the builder enumerates every admissible (subset, derangement)
and keeps the one whose five question-x-claim lexical channels differ LEAST
between the true and the wrong leg; passages are then admitted in ascending order
of that mismatch until the pair target, and the prefix is trimmed if any barred
channel leaves [0.46, 0.54].  Matched wrong questions are hard negatives: the
lane keeps only contrasts a lexical model cannot rank.

Sources REJECTED and why (measured this session, not assumed):
  halueval-qa    9,936 distinct knowledge blocks, only 63 with >= 2 questions -
                 no same-document wrong question exists at scale, and a
                 cross-document one is separable by topic novelty
  ragtruth_en    839 distinct QA contexts, each with exactly ONE query, and the
                 Summary / Data2txt halves carry an instruction, not a question

Run:  uv run python experiments/grounding-semantic/R20-H175b_qlane.py [--force]
"""

import collections
import hashlib
import importlib.util as _ilu
import io
import itertools
import json
import math
from pathlib import Path
import random
import re
import sys
import zipfile

import numpy as np
import polars as pl

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# banked lane instruments - imported, not copied
C = _mod("h174common", HERE / "R20-H174_lane_common.py")

# `--repair` builds the coordinator's repair-pass variant: identical corpus,
# guards, split and seeds, with the admission ALSO trimmed until the composite
# lexical-interaction probe clears 0.55.  Two artifacts, one comparison.
REPAIR = "--repair" in sys.argv
SUFFIX = "_repaired" if REPAIR else ""

OUT = HERE / f"R20-H175b_qlane{SUFFIX}.parquet"
EVAL_OUT = HERE / f"R20-H175b_qlane_eval{SUFFIX}.parquet"
MANIFEST = HERE / f"R20-H175b_qlane{SUFFIX}_manifest.json"

SEED = 1175
SEED_EVAL = 2175
TAG = "qrel_contrast"
FAMILY = "qswap_same_passage"

PAIR_TARGET = 15_000        # ~30k rows - the campaign's core lane scale
EVAL_PAIR_TARGET = 1_000
N_FOLDS = 5

# admission filters on a (passage, question, golden answer) triple
CLAIM_MIN_CHARS = 15        # a claim must be a statement, not a bare token
CLAIM_MIN_TOKENS = 3
GROUNDING_MIN = 0.50        # claim content tokens readable in its own passage
ANSWER_JACCARD_MAX = 0.50   # the two answers of a unit must be different facts
QUESTION_JACCARD_MAX = 0.80  # the two questions must not be paraphrases

# surface bar and the margin the prefix trim keeps below it
SURFACE_BAR = 0.05
SURFACE_MARGIN = 0.04
INTERACTION_BAR = 0.55      # the composite probe, trimmed on only in repair mode
INTERACTION_MARGIN = 0.54

EVAL_DOC_PERMILLE = 120     # 12% of source documents are eval-only

SOURCES = {
    "psiloqa": {
        "dataset": "s-nlp/PsiloQA (train split only)",
        "archive": "data/external/datasets/dataset-psiloqa.zip",
        "licence": "MIT (data/external/datasets/dataset-psiloqa.md)",
        "fields": "wiki_passage -> chunk, question -> question, golden_answer -> "
                  "claim; the validation and test splits are NOT read",
        "wall": "banked clean-mix corpus (DANN group `psiloqa`, in the mix since "
                "R8-H84); it predates the R14-H136 instrument, so this lane's "
                "census is its first 8-gram wall and it is run on the built "
                "artifacts",
    },
}

REJECTED_SOURCES = {
    "halueval_qa": "9,936 distinct knowledge blocks, 63 with >= 2 questions "
                   "(0.6%) - no same-document wrong question at scale; a "
                   "cross-document one is separable by topic novelty",
    "ragtruth_en": "839 distinct QA contexts, each with exactly one query "
                   "(5,034 rows = 839 contexts x model outputs); the Summary "
                   "and Data2txt halves carry an instruction, not a question",
}

_WORD = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def tok(text):
    """Unicode-aware content tokens - PsiloQA spans 14 languages, so the ASCII
    tokenizer the banked lanes use would erase most of the non-EN supply."""
    return _WORD.findall(text.lower())


def jaccard(a, b):
    A, B = set(tok(a)), set(tok(b))
    return len(A & B) / len(A | B) if (A | B) else 0.0


def containment(a, b):
    A = set(tok(a))
    return len(A & set(tok(b))) / len(A) if A else 0.0


def is_eval_doc(doc_id):
    """Deterministic, stable across runs - the banked H177 split rule."""
    h = hashlib.blake2b(str(doc_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % 1000 < EVAL_DOC_PERMILLE


# --------------------------------------------------------------------------- #
# supply
# --------------------------------------------------------------------------- #
def psiloqa_triples():
    """Admitted (passage, question, claim) triples, with the eval flag attached.

    One triple per (passage, question): PsiloQA binds exactly one golden answer
    to each, verified this session (0 of 60,612 carry two)."""
    z = zipfile.ZipFile(DATA / "dataset-psiloqa.zip")
    d = pl.read_parquet(io.BytesIO(z.read("s-nlp__PsiloQA__train.parquet")))
    d = d.select(["wiki_title", "wiki_passage", "question", "golden_answer", "lang"])
    d = d.unique(subset=["wiki_passage", "question"], keep="first", maintain_order=True)
    raw = d.height
    d = d.filter(pl.col("golden_answer").str.len_chars() >= CLAIM_MIN_CHARS)
    keep = [
        len(tok(a)) >= CLAIM_MIN_TOKENS and containment(a, p) >= GROUNDING_MIN
        for a, p in zip(d["golden_answer"].to_list(), d["wiki_passage"].to_list())
    ]
    d = d.filter(pl.Series(keep))
    d = d.with_columns(
        pl.Series("is_eval", [is_eval_doc(t) for t in d["wiki_title"].to_list()])
    )
    print(f"psiloqa triples: {raw} unique (passage,question) -> {d.height} admitted "
          f"({d.filter(~pl.col('is_eval')).height} train / "
          f"{d.filter(pl.col('is_eval')).height} eval)", flush=True)
    return d


def build_idf(triples):
    """Document frequency over the admitted questions and claims, one pool for
    both splits - the IDF is a fixed feature scale, it carries no label."""
    df, n = collections.Counter(), 0
    for q, a in zip(triples["question"].to_list(), triples["golden_answer"].to_list()):
        n += 1
        for w in set(tok(q)) | set(tok(a)):
            df[w] += 1
    return {w: math.log(n / (1 + c)) for w, c in df.items()}, n


# --------------------------------------------------------------------------- #
# the five question-x-claim interaction channels the selection matches
# --------------------------------------------------------------------------- #
def interaction(q, a, idf):
    Q, A = set(tok(q)), set(tok(a))
    I = Q & A
    return np.array([
        len(I) / len(Q) if Q else 0.0,          # question -> claim containment
        len(I) / len(A) if A else 0.0,          # claim -> question containment
        float(len(I)),                          # raw shared-token count
        len(I) / len(Q | A) if (Q | A) else 0.0,  # Jaccard
        sum(idf.get(w, 8.0) for w in I),        # IDF-weighted overlap
    ])


def passage_units(triples, idf, rng):
    """One best (subset, derangement) per passage, with its mismatch score.

    Enumerating every subset AND every derangement - not only the whole-passage
    2-cycle - is what buys the matching its freedom: a 3-question passage offers
    three 2-cycles and two 3-cycles, and the builder keeps whichever pairs the
    questions most tightly."""
    by = collections.defaultdict(list)
    for r in triples.iter_rows(named=True):
        by[r["wiki_passage"]].append(r)

    raw = []
    for passage, rows in by.items():
        k = len(rows)
        if k < 2:
            continue
        qs = [r["question"] for r in rows]
        ans = [r["golden_answer"] for r in rows]
        if len(set(qs)) < k or len(set(ans)) < k:
            continue
        F = {(i, j): interaction(qs[j], ans[i], idf) for i in range(k) for j in range(k)}
        opts = []
        for size in range(2, k + 1):
            for S in itertools.combinations(range(k), size):
                for perm in itertools.permutations(S):
                    mp = dict(zip(S, perm))
                    if any(mp[i] == i for i in S):
                        continue
                    if any(jaccard(ans[i], ans[mp[i]]) > ANSWER_JACCARD_MAX
                           or jaccard(qs[i], qs[mp[i]]) > QUESTION_JACCARD_MAX
                           for i in S):
                        continue
                    gap = np.array([F[(i, i)] - F[(i, mp[i])] for i in S])
                    opts.append((gap, S, mp))
        if opts:
            raw.append((passage, rows, opts))

    if not raw:
        return []
    # channel scale: the pooled sd of the channel over every admitted option, so
    # the max-gap objective compares channels on one footing
    sample = np.array([F for _, _, opts in raw for F in opts[0][0]]).reshape(-1, 5)
    sd = sample.std(axis=0)
    sd[sd == 0] = 1.0

    units = []
    for passage, rows, opts in raw:
        best = min(opts, key=lambda o: (float(np.abs(o[0] / sd).max()), -len(o[1])))
        units.append({
            "passage": passage, "rows": rows, "subset": best[1], "map": best[2],
            "mismatch": float(np.abs(best[0] / sd).max()),
            "jitter": rng.random(),
        })
    units.sort(key=lambda u: (u["mismatch"], u["jitter"]))
    return units


# --------------------------------------------------------------------------- #
# emission
# --------------------------------------------------------------------------- #
def emit_unit(unit, uid, pid):
    rows, S, mp = unit["rows"], unit["subset"], unit["map"]
    out = []
    for i in S:
        r, w = rows[i], rows[mp[i]]
        base = dict(
            chunk=r["wiki_passage"], claim=r["golden_answer"],
            doc_id=r["wiki_title"], lang=r["lang"], source="psiloqa", tag=TAG,
            neg_family=FAMILY, unit_id=uid, unit_size=len(S),
            true_word=r["question"], flip_word=w["question"],
            mismatch=round(unit["mismatch"], 6),
        )
        out.append(dict(pair_id=pid, label=1, question=r["question"], **base))
        out.append(dict(pair_id=pid, label=0, question=w["question"], **base))
        pid += 1
    return out, pid


def dedupe(df):
    """The banked `C.dedupe`, with `question` in the key.

    A lane row is identified by (claim, chunk, question, label) here - the banked
    key would collapse the two legs of every pair, which differ only in the
    question.  Pairs left incomplete by a drop are removed whole, so the
    derangement's usage balance survives de-duplication."""
    d = df.unique(subset=["claim", "chunk", "question", "label"], keep="first",
                  maintain_order=True)
    keep = d.group_by("pair_id").len().filter(pl.col("len") == 2)["pair_id"]
    return d.filter(pl.col("pair_id").is_in(keep)).sort(
        ["pair_id", "label"], descending=[False, True])


def channel_table(df):
    """Every surface channel the lane is barred on, per row.

    The IDF weights are read off the lane itself, so the block is self-contained
    and the eval is scored on its own text rather than on the training lane's."""
    q, c, k = df["question"].to_list(), df["claim"].to_list(), df["chunk"].to_list()
    qt, ct, kt = [set(tok(x)) for x in q], [set(tok(x)) for x in c], \
        [set(tok(x)) for x in k]
    dfq = collections.Counter()
    for a, b in zip(qt, ct):
        for w in a | b:
            dfq[w] += 1
    n = len(q)
    idf = {w: math.log(n / (1 + v)) for w, v in dfq.items()}
    inter = [a & b for a, b in zip(qt, ct)]
    return {
        "claim_char_length": [float(len(x)) for x in c],
        "claim_token_count": [float(len(x)) for x in ct],
        "chunk_char_length": [float(len(x)) for x in k],
        "question_char_length": [float(len(x)) for x in q],
        "question_token_count": [float(len(x)) for x in qt],
        "claim_chunk_containment":
            [len(a & b) / len(a) if a else 0.0 for a, b in zip(ct, kt)],
        "question_chunk_containment":
            [len(a & b) / len(a) if a else 0.0 for a, b in zip(qt, kt)],
        "question_claim_containment":
            [len(i) / len(a) if a else 0.0 for i, a in zip(inter, qt)],
        "claim_question_containment":
            [len(i) / len(a) if a else 0.0 for i, a in zip(inter, ct)],
        "question_claim_overlap_tokens": [float(len(i)) for i in inter],
        "question_claim_jaccard":
            [len(i) / len(a | b) if (a | b) else 0.0
             for i, a, b in zip(inter, qt, ct)],
        "question_claim_idf_overlap":
            [sum(idf.get(w, 8.0) for w in i) for i in inter],
    }


def assemble(units, target, split):
    """Admit whole passages in ascending mismatch order, then trim the prefix
    back to the last point where every barred channel still sits inside the
    surface bar with `SURFACE_MARGIN` to spare.  Volume is what the bar allows,
    never the other way round."""
    rows, uid, pid = [], 0, 0
    for u in units:
        if pid >= target:
            break
        new, pid = emit_unit(u, uid, pid)
        rows.extend(new)
        uid += 1
    df = pl.DataFrame(rows)

    ch = {k: np.asarray(v, dtype=float) for k, v in channel_table(df).items()}
    y = np.asarray(df["label"].to_list())
    unit_ids = np.asarray(df["unit_id"].to_list())

    def ok_upto(cut):
        """(passes, worst channel deviation, composite probe) at this prefix."""
        m = unit_ids < cut
        dev = max(abs(C.auroc(y[m], v[m]) - 0.5) for v in ch.values())
        if not REPAIR:
            return dev <= SURFACE_MARGIN, dev, None
        sub = df.filter(pl.col("unit_id") < cut)
        p, _ = feature_probe(sub, {k: v[m].tolist() for k, v in ch.items()},
                             random.Random(SEED))
        return (dev <= SURFACE_MARGIN and p["value"] <= INTERACTION_MARGIN,
                dev, p["value"])

    passes, dev, probe_at_target = ok_upto(uid)
    print(f"  [{split}] {pid} pairs over {uid} passages, worst channel deviation "
          f"{dev:.4f}, composite probe {probe_at_target}", flush=True)
    base = {"worst_deviation_at_target": round(dev, 4),
            "composite_probe_at_target": probe_at_target,
            "passages_at_target": uid,
            "criterion": (f"every barred channel within {SURFACE_MARGIN} of 0.5"
                          + (f" AND the composite lexical-interaction probe "
                             f"<= {INTERACTION_MARGIN}" if REPAIR else ""))}
    if passes:
        return df, {"trimmed": False, "passages_kept": uid, **base}

    lo, hi = 1, uid            # largest passage prefix that still passes
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ok_upto(mid)[0]:
            lo = mid
        else:
            hi = mid - 1
    _, dev_k, probe_k = ok_upto(lo)
    print(f"  [{split}] trimmed to {lo} passages, deviation {dev_k:.4f}, "
          f"composite probe {probe_k}", flush=True)
    return df.filter(pl.col("unit_id") < lo), {
        "trimmed": True, "passages_kept": int(lo),
        "worst_deviation_kept": round(dev_k, 4),
        "composite_probe_kept": probe_k, **base}


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def text_probe(texts, labels, groups, rng, label):
    auc, score = C.claim_only_probe(texts, labels, groups, rng, n_folds=N_FOLDS)
    return {"value": round(auc, 4), "bar": "< 0.55", "pass": bool(auc < 0.55),
            "scoring": f"{N_FOLDS}-fold document-disjoint, out of fold, char_wb "
                       "2-5 TF-IDF, liblinear tol 1e-7",
            "field": label}, score


def feature_probe(df, ch, rng):
    """Converged linear probe over EVERY barred surface channel at once, plus
    each channel squared.

    A per-channel AUROC only rules out a monotone shortcut; this rules out a
    shortcut built from several channels together or from the shape of one
    channel's distribution, which is the leak a rank statistic cannot see."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.array(list(ch.values()), dtype=float).T
    X = np.hstack([X, X ** 2])
    y = np.array(df["label"].to_list())
    groups = df["doc_id"].to_list()
    keys = sorted(set(groups))
    rng.shuffle(keys)
    fold_of = {k: i % N_FOLDS for i, k in enumerate(keys)}
    folds = np.array([fold_of[g] for g in groups])
    score = np.zeros(len(y))
    for f in range(N_FOLDS):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        if not tr.size or not te.size:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(solver="liblinear", C=4.0, tol=1e-7, max_iter=3000)
        clf.fit(sc.transform(X[tr]), y[tr])
        score[te] = clf.decision_function(sc.transform(X[te]))
    auc = float(C.auroc(y, score))
    return {"value": round(auc, 4), "bar": "< 0.55", "pass": bool(auc < 0.55),
            "features": list(ch) + [f"{k}^2" for k in ch],
            "scoring": f"{N_FOLDS}-fold document-disjoint, out of fold, "
                       "standardized, liblinear tol 1e-7"}, score


def surface_parity(df, ch):
    y = df["label"].to_list()
    vals = {k: round(C.auroc(y, v), 4) for k, v in ch.items()}
    worst = max(abs(v - 0.5) for v in vals.values())
    return {"auroc": vals, "bar": "each channel in [0.45, 0.55]",
            "worst_deviation": round(worst, 4), "pass": bool(worst <= SURFACE_BAR)}


def question_usage_balance(df):
    """Every question must be used exactly as often as the TRUE question as it is
    used as the WRONG one - the derangement gives this, and it is counted rather
    than asserted."""
    counts = collections.Counter()
    for t, f in zip(df.filter(pl.col("label") == 1)["true_word"].to_list(),
                    df.filter(pl.col("label") == 1)["flip_word"].to_list()):
        counts[(t, 0)] += 1
        counts[(f, 1)] += 1
    qs = {q for q, _ in counts}
    worst, offenders = 0.0, []
    hist = collections.Counter()
    for q in qs:
        a, b = counts[(q, 0)], counts[(q, 1)]
        share = a / (a + b)
        hist[(a, b)] += 1
        if abs(share - 0.5) > worst:
            worst = abs(share - 0.5)
        if abs(share - 0.5) > 0.02:
            offenders.append({"as_true": a, "as_wrong": b})
    return {"distinct_questions": len(qs),
            "usage_histogram": {f"true={a},wrong={b}": n for (a, b), n in
                                sorted(hist.items())},
            "questions_off_balance": len(offenders),
            "worst_deviation_from_half": round(worst, 4),
            "bar": "every question within 0.02 of a 50/50 true/wrong split",
            "pass": bool(worst <= 0.02)}


def attestation_symmetry(df):
    """The wrong question must be as answerable FROM THIS PASSAGE as the true one.

    Structurally it is: the wrong question is another PsiloQA question over the
    same passage, so it is attested by source construction.  That is checked
    (0 rows may carry a wrong question from a different passage) and the lexical
    consequence - question/evidence containment parity between the legs - is
    measured, because an asymmetry there is the topic-novelty leak this design
    exists to close."""
    same_passage = collections.defaultdict(set)
    for ch, q in zip(df["chunk"].to_list(), df["question"].to_list()):
        same_passage[ch].add(q)
    foreign = 0
    for r in df.filter(pl.col("label") == 0).iter_rows(named=True):
        if r["question"] not in same_passage[r["chunk"]]:
            foreign += 1
    y = df["label"].to_list()
    cont = [containment(q, c) for q, c in zip(df["question"].to_list(),
                                              df["chunk"].to_list())]
    au = C.auroc(y, cont)
    return {"wrong_questions_from_a_foreign_passage": foreign,
            "question_chunk_containment_auroc": round(au, 4),
            "bar": "0 foreign wrong questions; containment AUROC in [0.45, 0.55]",
            "pass": bool(foreign == 0 and abs(au - 0.5) <= SURFACE_BAR)}


def contrast_is_real(df):
    neg = df.filter(pl.col("label") == 0)
    same = int(sum(1 for t, f in zip(neg["true_word"].to_list(),
                                     neg["flip_word"].to_list()) if t == f))
    identical = df.group_by("pair_id").agg(
        pl.col("question").n_unique().alias("n")).filter(pl.col("n") < 2).height
    bad_ans = int(sum(1 for a, b in zip(neg["claim"].to_list(),
                                        neg["flip_word"].to_list())
                      if jaccard(a, b) > 0.95))
    return {"pairs_with_equal_questions": same,
            "pairs_with_identical_rows": int(identical),
            "negatives_whose_wrong_question_restates_the_claim": bad_ans,
            "bar": "0 on all three", "pass": not (same or identical or bad_ans)}


def grounding_audit(df):
    """The label-1 leg asserts a claim that IS supported by the evidence - the
    lane's whole point is that only the question moves, so a positive that is
    not grounded would make the pair a grounding contrast, not a relevance one."""
    cont = [containment(c, k) for c, k in
            zip(df["claim"].to_list(), df["chunk"].to_list())]
    a = np.array(cont)
    return {"claim_chunk_containment": {
                "mean": round(float(a.mean()), 4),
                "p10": round(float(np.percentile(a, 10)), 4),
                "min": round(float(a.min()), 4)},
            "rows_below_floor": int((a < GROUNDING_MIN).sum()),
            "bar": f"0 rows below the admission floor {GROUNDING_MIN}",
            "pass": bool((a < GROUNDING_MIN).sum() == 0)}


def verify(df, rng):
    out = {"pair_integrity": C.pair_integrity(df)}

    docs = df["doc_id"].to_list()
    labels = df["label"].to_list()

    def sub():
        return random.Random(rng.randrange(1 << 30))

    claim_p, claim_score = text_probe(df["claim"].to_list(), labels, docs,
                                      sub(), "claim")
    out["claim_only_tfidf_auroc"] = claim_p
    out["claim_only_tfidf_auroc"]["structural_note"] = (
        "both rows of a pair carry the SAME claim, so this probe is 0.5 by "
        "construction; it is run because the bar is registered and because a "
        "build defect that broke the pairing would show here")

    wp = C.within_pair_accuracy(df, claim_score)
    worst = max(v["acc"] for v in wp.values())
    out["within_pair_claim_only_accuracy"] = {
        "per_family": wp, "worst": round(worst, 4), "bar": "< 0.60",
        "pass": bool(worst < 0.60)}

    out["question_only_tfidf_auroc"], _ = text_probe(
        df["question"].to_list(), labels, docs, sub(), "question")
    out["evidence_only_tfidf_auroc"], _ = text_probe(
        df["chunk"].to_list(), labels, docs, sub(), "chunk")
    out["question_plus_claim_bag_tfidf_auroc"], _ = text_probe(
        [f"{q} {c}" for q, c in zip(df["question"].to_list(), df["claim"].to_list())],
        labels, docs, sub(), "question + claim")

    ch = channel_table(df)
    out["surface_parity"] = surface_parity(df, ch)
    out["lexical_interaction_probe"], _ = feature_probe(df, ch, sub())
    out["question_usage_balance"] = question_usage_balance(df)
    out["attestation_symmetry"] = attestation_symmetry(df)
    out["contrast_is_real"] = contrast_is_real(df)
    out["positive_leg_grounding"] = grounding_audit(df)

    # `all_bars_pass` covers the bars named in the STAGE 0 registration.  The
    # composite lexical-interaction probe is an instrument this build ADDED, so
    # it is reported separately and never silently promoted into the registered
    # conjunction (a bar added after the data is seen moves the gate exactly as
    # much as one loosened after the data is seen).
    out["all_bars_pass"] = all(
        out[k]["pass"] for k in
        ("pair_integrity", "claim_only_tfidf_auroc",
         "within_pair_claim_only_accuracy", "question_only_tfidf_auroc",
         "evidence_only_tfidf_auroc", "question_plus_claim_bag_tfidf_auroc",
         "surface_parity", "question_usage_balance", "attestation_symmetry",
         "contrast_is_real", "positive_leg_grounding"))
    out["registered_bars"] = [
        "pair_integrity", "claim_only_tfidf_auroc",
        "within_pair_claim_only_accuracy", "question_only_tfidf_auroc",
        "evidence_only_tfidf_auroc", "question_plus_claim_bag_tfidf_auroc",
        "surface_parity", "question_usage_balance", "attestation_symmetry",
        "contrast_is_real", "positive_leg_grounding"]
    out["all_bars_pass_including_composite_probe"] = bool(
        out["all_bars_pass"] and out["lexical_interaction_probe"]["pass"])
    return out


# --------------------------------------------------------------------------- #
def block(df, res, split, seed, target, trim):
    y = df["label"].to_list()
    langs = {k: v for k, v in df.group_by("lang").len().iter_rows()}
    return dict(
        rows=df.height, pairs=int(df["pair_id"].n_unique()),
        passages=int(df["chunk"].n_unique()),
        documents=int(df["doc_id"].n_unique()),
        units=int(df["unit_id"].n_unique()),
        seed=seed, split=split, pair_target=target, selection=trim,
        label_balance={"label_1": int(sum(y)), "label_0": int(len(y) - sum(y))},
        families={FAMILY: df.height},
        family_shares={FAMILY: 1.0},
        languages=dict(sorted(langs.items(), key=lambda kv: -kv[1])),
        language_shares={k: round(v / df.height, 4) for k, v in
                         sorted(langs.items(), key=lambda kv: -kv[1])},
        unit_sizes={str(k): v for k, v in df.group_by("unit_size").len().iter_rows()},
        diversity=dict(distinct_questions=int(df["question"].n_unique()),
                       distinct_claims=int(df["claim"].n_unique()),
                       distinct_chunks=int(df["chunk"].n_unique())),
        char_stats=dict(claim=C.char_stats(df["claim"].to_list()),
                        chunk=C.char_stats(df["chunk"].to_list()),
                        question=C.char_stats(df["question"].to_list())),
        window_census=C.window_census(df["chunk"].to_list()),
        verify=res)


def already_built():
    if "--force" in sys.argv or not (OUT.exists() and MANIFEST.exists()
                                     and EVAL_OUT.exists()):
        return False
    try:
        man = json.loads(MANIFEST.read_text())
        rows = pl.read_parquet(OUT).height
    except Exception:
        return False
    if man.get("verify", {}).get("all_bars_pass") and rows == man.get("rows"):
        print(f"{OUT.name}: {rows} rows already built and passing - skipping "
              f"(pass --force to rebuild)", flush=True)
        return True
    return False


def main():
    if already_built():
        return
    print(f"=== R20-H175b stage 0 - question-relevance contrast lane ({TAG}) "
          f"seed {SEED}", flush=True)
    triples = psiloqa_triples()
    idf, n_idf = build_idf(triples)
    print(f"idf vocabulary {len(idf)} over {n_idf} triples", flush=True)

    tr = triples.filter(~pl.col("is_eval"))
    ev = triples.filter(pl.col("is_eval"))
    units = passage_units(tr, idf, random.Random(SEED))
    units_ev = passage_units(ev, idf, random.Random(SEED_EVAL))
    print(f"candidate passages: {len(units)} train / {len(units_ev)} eval "
          f"(pairs available {sum(len(u['subset']) for u in units)} / "
          f"{sum(len(u['subset']) for u in units_ev)})", flush=True)

    df, trim = assemble(units, PAIR_TARGET, "train")
    df = dedupe(df)
    df.write_parquet(OUT)
    print(f"{df.height} rows / {df['pair_id'].n_unique()} pairs -> {OUT.name}",
          flush=True)

    evdf, trim_ev = assemble(units_ev, EVAL_PAIR_TARGET, "eval")
    evdf = dedupe(evdf)
    evdf.write_parquet(EVAL_OUT)
    print(f"{evdf.height} rows / {evdf['pair_id'].n_unique()} pairs -> "
          f"{EVAL_OUT.name}", flush=True)

    res = verify(df, random.Random(SEED))
    ev_res = verify(evdf, random.Random(SEED_EVAL))
    shared_docs = set(df["doc_id"].to_list()) & set(evdf["doc_id"].to_list())
    shared_chunks = set(df["chunk"].to_list()) & set(evdf["chunk"].to_list())

    man = dict(
        experiment="R20-H175b stage 0 - question-relevance contrast lane "
                   "(qrel_contrast)",
        registration="docs/experiments/semantic-grounding-experiments.md, block "
                     "'R20-H175b QUESTION CONDITIONING (measurement only)', "
                     "STAGE 0",
        tag=TAG, dann_group=TAG,
        variant=("repair pass - admission ALSO trimmed until the composite "
                 "lexical-interaction probe clears 0.55" if REPAIR else
                 "registered build - admission trimmed on the registered "
                 "surface-parity bar only"),
        requires_question_channel=True,
        loader_warning="THIS LANE IS ONLY VALID UNDER A QUESTION-CONDITIONED "
                       "PRESENTATION. Both rows of a pair carry the same claim "
                       "and the same chunk and differ ONLY in `question`; loaded "
                       "into a mix that drops the question field the lane "
                       "becomes label-contradictory duplicate rows, i.e. pure "
                       "label noise. The stage-1 arm must compose the question "
                       "into the model input for every row of this lane.",
        mix_loader="columns claim / chunk / label / pair_id / neg_family as the "
                   "banked lanes, PLUS `question`; chunk is read UNTRUNCATED and "
                   "windowed 1500/750 by R18-H150_arm_run.make_build_mix",
        parquet=OUT.name,
        construction=dict(
            unit="one passage, a subset S of its distinct questions (|S| >= 2) "
                 "and a derangement sigma on S",
            positive="question = the claim's own originating question",
            negative="question = another question over the SAME passage, with "
                     "its own distinct golden answer",
            balance="every question is used exactly once as the true question "
                    "and exactly once as the wrong one, so any claim-only, "
                    "evidence-only or question-only statistic has AUROC exactly "
                    "0.5 and only a question-x-claim interaction can separate "
                    "the classes",
            matching="for each passage the (subset, derangement) minimising the "
                     "largest sd-normalised gap over five question-x-claim "
                     "lexical channels; passages admitted in ascending mismatch "
                     "order until the pair target",
            admission=dict(claim_min_chars=CLAIM_MIN_CHARS,
                           claim_min_tokens=CLAIM_MIN_TOKENS,
                           claim_chunk_containment_min=GROUNDING_MIN,
                           answer_jaccard_max=ANSWER_JACCARD_MAX,
                           question_jaccard_max=QUESTION_JACCARD_MAX),
            seed_role="the generator is deterministic given the supply; the seed "
                      "enters only as the tie-break jitter among equal-mismatch "
                      "passages and as the probe fold assignment. The eval's "
                      "independence comes from the document hash split, not "
                      "from the seed",
        ),
        sources=SOURCES,
        rejected_sources=REJECTED_SOURCES,
        **block(df, res, "train", SEED, PAIR_TARGET, trim))
    man["held_out_eval"] = dict(
        parquet=EVAL_OUT.name,
        purpose="the R20-H175b PRIMARY mechanism gate - held-out "
                "question-relevance AUROC, read on a flagship baseline leg "
                "BEFORE training and again after; no model is read here "
                "(stage 0 is CPU-only)",
        **block(evdf, ev_res, "eval", SEED_EVAL, EVAL_PAIR_TARGET, trim_ev))
    man["document_disjointness"] = dict(
        rule=f"blake2b(wiki_title) % 1000 < {EVAL_DOC_PERMILLE} -> eval only",
        shared_documents=len(shared_docs), shared_chunks=len(shared_chunks),
        bar="0 shared documents", passes=len(shared_docs) == 0)
    MANIFEST.write_text(json.dumps(man, indent=2))

    print(json.dumps({k: man[k] for k in
                      ("rows", "pairs", "passages", "documents", "selection",
                       "language_shares", "char_stats", "verify")}, indent=2),
          flush=True)
    print(json.dumps({"eval_rows": man["held_out_eval"]["rows"],
                      "eval_pairs": man["held_out_eval"]["pairs"],
                      "eval_documents": man["held_out_eval"]["documents"],
                      "eval_selection": man["held_out_eval"]["selection"],
                      "eval_verify": man["held_out_eval"]["verify"],
                      "shared_documents": len(shared_docs),
                      "shared_chunks": len(shared_chunks)}, indent=2), flush=True)
    key = "all_bars_pass_including_composite_probe" if REPAIR else "all_bars_pass"
    ok = res[key] and ev_res[key] and not shared_docs
    print(f"=== R20-H175b QLANE{SUFFIX.upper()} {'BUILT' if ok else 'FAILED BARS'} "
          f"({key}) ===", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

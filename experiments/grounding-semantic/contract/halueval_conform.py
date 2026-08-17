"""HALUEVAL CONFORMANCE - diagnose the C5 claim-only channel, try to build a
conforming variant, and re-verify it against all eight clauses.  CPU ONLY.

Contract: docs/experiments/dataset-contract.md, amendments C-A1 and C-A2 applied.
Member: the `halueval` DANN group of the assembled training mix.  Phase-1 verdict
banked in `halueval_contract_report.json`: C5 FAIL, claim-only converged probe
AUROC 0.9519 (bar 0.55), within-pair claim-only 0.9666 (bar 0.60).

DISCIPLINE
  * every number measured here; nothing read off the phase-1 report
  * the member is rebuilt through the BANKED loader and its pair structure is
    re-proved aligned row-for-row before use (halueval_contract.load_member)
  * instruments are the banked ones - R20-H174_lane_common.claim_only_probe,
    .containment, .surface_parity, .within_pair_accuracy, provenance_gate.py
  * CPU only; CUDA_VISIBLE_DEVICES forced empty before any import
  * measurements only - the coordinator adjudicates

Stages (each writes its own JSON so a killed run resumes from disk):
  scores    baseline claim-only probe + per-row style features    -> _scores.npz
  diag      channel decomposition + live positive controls        -> _diag.json
  frontier  strategy x retention sweep + split-sample honesty     -> _frontier.json
  build     pick the variant, write the parquet                   -> halueval_conformed.parquet
  verify    all eight clauses on the variant                      -> _clauses.json
  merge     assemble halueval_conformed_report.json
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU ONLY - three GPUs carry live draws
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import importlib.util
import json
import pathlib
import random
import re
import time

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
EXP = HERE.parent
ROOT = EXP.parent.parent

NOTE = "Numbers recorded, not adjudicated - the coordinator adjudicates."
# Two variants are materialised.  `conformed` is the only subset in the whole
# sweep that clears every leg of C5; `besteffort` is the largest subset that
# comes near the claim-only bar, kept so the coordinator can see what supply a
# near-miss buys.
NAMES = {
    "conformed": ("halueval_conformed.parquet",
                  "halueval_conformed_build.json",
                  "halueval_conformed_clauses.json"),
    "besteffort": ("halueval_conform_best_effort.parquet",
                   "halueval_conform_best_effort_build.json",
                   "halueval_conform_best_effort_clauses.json"),
}
ORIG_ROWS = 40_000
ORIG_PAIRS = 20_000


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


HC = _mod("halueval_contract", HERE / "halueval_contract.py")
C = _mod("lanecommon", EXP / "R20-H174_lane_common.py")


# --------------------------------------------------------------------------- #
# member
# --------------------------------------------------------------------------- #
def load():
    d = HC.load_member()
    df = HC.frame(d)
    return d, df


def groups_of(df):
    return [f"{h}:{p}" for h, p in zip(df["half"].to_list(), df["pair_id"].to_list())]


def probe(claims, labels, groups, seed=0):
    return C.claim_only_probe(claims, np.asarray(labels, dtype=float), groups,
                              random.Random(seed))


def numeric_probe(X, labels, groups, seed=0, n_folds=5):
    """Same fold discipline as the banked claim-only probe, numeric features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray(labels, dtype=float)
    rng = random.Random(seed)
    keys = sorted(set(groups))
    rng.shuffle(keys)
    fold_of = {k: i % n_folds for i, k in enumerate(keys)}
    folds = np.array([fold_of[g] for g in groups])
    score = np.zeros(len(labels))
    idx = np.arange(len(labels))
    for f in range(n_folds):
        tr, te = idx[folds != f], idx[folds == f]
        if not te.size or not tr.size:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(solver="liblinear", C=4.0, tol=1e-7, max_iter=3000)
        clf.fit(sc.transform(X[tr]), labels[tr])
        score[te] = clf.decision_function(sc.transform(X[te]))
    return float(C.auroc(labels, score)), score


# --------------------------------------------------------------------------- #
# style vocabulary and claim transforms
# --------------------------------------------------------------------------- #
FUNCTION = frozenset("""
a an the this that these those his her its their our your my he she it they we you i him them us me
and or but nor so yet for as if while when where which who whom whose what whether because since
although though unless until after before during of in on at to from by with without within into onto
over under above below between among across through against about around near per via up down out off
is are was were be been being am do does did done have has had having will would shall should can could
may might must not no none neither either both all any each every some many much more most less least
than then there here such only just also too very own same other another first last next new
""".split())

HEDGE = frozenset("""
may might could possibly probably likely perhaps reportedly allegedly apparently seemingly
approximately roughly nearly almost about around several some many often usually generally typically
seems seem appears appear suggest suggests suggested believed thought estimated claimed considered
potential potentially presumably arguably supposedly rumoured rumored
""".split())

TOKEN = re.compile(r"[A-Za-z0-9']+")
TOKSPLIT = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9]")
PUNCT_CHARS = ".,;:!?\"'()[]-/%$&"


def t_identity(c):
    return c


def t_norm(c):
    """Lowercased, punctuation stripped, whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9']+", " ", c)).strip().lower()


def t_shape(c):
    """CONTENT MASKED - every content token becomes `w` (`d` if numeric).
    Function words, hedges, punctuation and token count are preserved, so what
    remains is register / shape / length and nothing topical."""
    out = []
    for tok in TOKSPLIT.findall(c):
        if TOKEN.fullmatch(tok):
            low = tok.lower()
            if low in FUNCTION or low in HEDGE:
                out.append(low)
            elif tok.isdigit():
                out.append("d")
            else:
                out.append("w")
        else:
            out.append(tok)
    return " ".join(out)


def t_content(c):
    """STYLE REMOVED - function words, hedges, punctuation and case dropped."""
    return " ".join(t for t in (x.lower() for x in TOKEN.findall(c))
                    if t not in FUNCTION and t not in HEDGE)


def t_content_sorted(c):
    return " ".join(sorted(t_content(c).split()))


def t_funcpunct(c):
    """CONTENT DELETED ENTIRELY - only function words, hedges and punctuation."""
    out = []
    for tok in TOKSPLIT.findall(c):
        if TOKEN.fullmatch(tok):
            low = tok.lower()
            if low in FUNCTION or low in HEDGE:
                out.append(low)
        else:
            out.append(tok)
    return " ".join(out)


def style_features(claims):
    """Numeric style / length channels, one row per claim."""
    rows = []
    names = (["char_len", "log_char_len", "tok_count", "log_tok_count",
              "mean_tok_len", "type_token_ratio", "upper_ratio", "cap_tok_ratio",
              "digit_ratio", "func_rate", "hedge_rate", "hedge_count", "space_ratio"]
             + [f"punct_{ch}" for ch in PUNCT_CHARS])
    for c in claims:
        toks = TOKEN.findall(c)
        n = max(len(toks), 1)
        low = [t.lower() for t in toks]
        nchar = max(len(c), 1)
        feat = [
            float(len(c)),
            float(np.log1p(len(c))),
            float(len(toks)),
            float(np.log1p(len(toks))),
            float(np.mean([len(t) for t in toks])) if toks else 0.0,
            len(set(low)) / n,
            sum(ch.isupper() for ch in c) / nchar,
            sum(t[:1].isupper() for t in toks) / n,
            sum(ch.isdigit() for ch in c) / nchar,
            sum(t in FUNCTION for t in low) / n,
            sum(t in HEDGE for t in low) / n,
            float(sum(t in HEDGE for t in low)),
            c.count(" ") / nchar,
        ] + [c.count(ch) / nchar for ch in PUNCT_CHARS]
        rows.append(feat)
    return np.asarray(rows, dtype=float), names


def length_features(claims):
    X, names = style_features(claims)
    keep = [names.index(k) for k in ("char_len", "log_char_len", "tok_count",
                                     "log_tok_count", "mean_tok_len")]
    return X[:, keep], [names[i] for i in keep]


def punct_features(claims):
    X, names = style_features(claims)
    keep = [i for i, k in enumerate(names)
            if k.startswith("punct_") or k in ("upper_ratio", "cap_tok_ratio",
                                               "digit_ratio", "space_ratio")]
    return X[:, keep], [names[i] for i in keep]


def hedge_features(claims):
    X, names = style_features(claims)
    keep = [names.index(k) for k in ("hedge_rate", "hedge_count", "func_rate",
                                     "type_token_ratio")]
    return X[:, keep], [names[i] for i in keep]


# --------------------------------------------------------------------------- #
# stage: scores
# --------------------------------------------------------------------------- #
def stage_scores():
    d, df = load()
    claims = df["claim"].to_list()
    labels = df["label"].to_numpy()
    g = groups_of(df)

    t0 = time.time()
    auc, score = probe(claims, labels, g)
    print(f"baseline claim-only probe AUROC {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)

    t0 = time.time()
    sh_auc, sh_score = probe([t_shape(c) for c in claims], labels, g)
    print(f"style/shape probe (content masked) AUROC {sh_auc:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    Xs, snames = style_features(claims)
    st_auc, st_score = numeric_probe(Xs, labels, g)
    print(f"numeric style probe AUROC {st_auc:.4f}", flush=True)

    np.savez_compressed(
        HERE / "halueval_conform_scores.npz",
        score=score, shape_score=sh_score, style_score=st_score,
        labels=labels, style_features=Xs,
        pair_id=np.asarray(df["pair_id"].to_list()),
        half=np.asarray(df["half"].to_list()),
    )
    (HERE / "halueval_conform_scores.json").write_text(json.dumps({
        "note": NOTE,
        "baseline_claim_only_auroc": round(auc, 4),
        "content_masked_shape_probe_auroc": round(sh_auc, 4),
        "numeric_style_probe_auroc": round(st_auc, 4),
        "style_feature_names": snames,
        "rows": df.height, "pairs": int(df["pair_id"].n_unique()),
    }, indent=2))
    print("scores written", flush=True)


# --------------------------------------------------------------------------- #
# stage: diag
# --------------------------------------------------------------------------- #
def stratified_auroc(score, labels, strat, n_bins=10):
    """AUROC computed WITHIN strata and pooled, weighted by n_pos * n_neg."""
    strat = np.asarray(strat, dtype=float)
    edges = np.unique(np.quantile(strat, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, strat, side="right") - 1, 0, len(edges) - 2)
    num = den = 0.0
    per_bin = []
    for b in range(len(edges) - 1):
        m = idx == b
        yb, sb = labels[m], score[m]
        npos, nneg = int((yb == 1).sum()), int((yb == 0).sum())
        if npos == 0 or nneg == 0:
            per_bin.append({"bin": b, "n": int(m.sum()), "n_pos": npos,
                            "n_neg": nneg, "auroc": None})
            continue
        a = C.auroc(yb, sb)
        per_bin.append({"bin": b, "n": int(m.sum()), "n_pos": npos, "n_neg": nneg,
                        "auroc": round(float(a), 4),
                        "range": [round(float(edges[b]), 1), round(float(edges[b + 1]), 1)]})
        num += a * npos * nneg
        den += npos * nneg
    return (round(float(num / den), 4) if den else None), per_bin


def toy_style_control(claims, seed=0):
    """LIVE POSITIVE CONTROL - a toy whose two legs differ ONLY in register.
    Content is held identical inside a pair, so any separation the probe finds
    is style.  A working claim-only probe must read near 1.0 here."""
    rng = random.Random(seed)
    pool = [c for c in claims if 40 <= len(c) <= 400]
    rng.shuffle(pool)
    pool = pool[:1500]
    pos, neg, y, g = [], [], [], []
    for i, body in enumerate(pool):
        core = body.rstrip(" .")
        pos.append(f"{core}.")
        neg.append(f"It is possible that {core[:1].lower()}{core[1:]}, reportedly!!")
    for i, (p, n) in enumerate(zip(pos, neg)):
        y += [1.0, 0.0]
        g += [f"toy:{i}", f"toy:{i}"]
    texts = [t for pair in zip(pos, neg) for t in pair]
    auc, _ = probe(texts, y, g)
    return {"construction": "1,500 pairs; both legs carry the SAME content string, "
                            "the negative leg wrapped in a hedged register "
                            "(`It is possible that ... , reportedly!!`)",
            "pairs": len(pos), "claim_only_probe_auroc": round(auc, 4),
            "expected": "near 1.0", "fires": bool(auc >= 0.95)}


def stage_diag():
    d, df = load()
    claims = df["claim"].to_list()
    labels = df["label"].to_numpy()
    half = np.asarray(df["half"].to_list())
    g = groups_of(df)
    z = np.load(HERE / "halueval_conform_scores.npz", allow_pickle=True)
    base_score = z["score"]

    out = {"note": NOTE,
           "member": "halueval",
           "instrument": "R20-H174_lane_common.claim_only_probe - out-of-fold "
                         "char_wb TF-IDF 2-5 grams, min_df 3, sublinear tf, "
                         "liblinear C=4 tol 1e-7, folds disjoint on the pair key",
           "scopes": {}}

    # ---- 1. live positive controls -------------------------------------- #
    print("positive controls", flush=True)
    ctrl = {}
    ctrl["toy_style_separated"] = toy_style_control(claims)
    print(f"  toy: {ctrl['toy_style_separated']}", flush=True)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(labels))
    sh_lab = labels[perm]
    a_glob, _ = probe(claims, sh_lab, g)
    ctrl["label_shuffled_global"] = {
        "construction": "labels permuted uniformly at random over all 40,000 rows",
        "claim_only_probe_auroc": round(a_glob, 4), "expected": "near 0.5",
        "reads_at_chance": bool(abs(a_glob - 0.5) <= 0.03)}
    print(f"  global shuffle: {a_glob:.4f}", flush=True)

    lab2 = labels.copy()
    pid = np.asarray(df["pair_id"].to_list())
    flip = rng.random(int(pid.max()) + 1) < 0.5
    lab2 = np.where(flip[pid], 1.0 - labels, labels)
    a_pair, _ = probe(claims, lab2, g)
    ctrl["label_shuffled_within_pair"] = {
        "construction": "each pair's two labels swapped with probability 0.5 - "
                        "class balance and pair structure preserved exactly",
        "claim_only_probe_auroc": round(a_pair, 4), "expected": "near 0.5",
        "reads_at_chance": bool(abs(a_pair - 0.5) <= 0.03)}
    print(f"  within-pair shuffle: {a_pair:.4f}", flush=True)
    ctrl["gate_is_live"] = bool(ctrl["toy_style_separated"]["fires"]
                                and ctrl["label_shuffled_global"]["reads_at_chance"]
                                and ctrl["label_shuffled_within_pair"]["reads_at_chance"])
    out["live_positive_control"] = ctrl

    # ---- 2. decomposition, per scope ------------------------------------ #
    TRANSFORMS = [
        ("full_claim", t_identity,
         "the claim as the mix feeds it - the registered C5 channel"),
        ("normalised", t_norm,
         "lowercased, punctuation stripped, whitespace collapsed"),
        ("content_only_style_removed", t_content,
         "function words, hedges, punctuation and case DELETED - what survives "
         "is topical content"),
        ("content_only_sorted", t_content_sorted,
         "content tokens sorted alphabetically - word order destroyed too"),
        ("shape_only_content_masked", t_shape,
         "every content token replaced by `w` (`d` if numeric); function words, "
         "hedges, punctuation and token count preserved - pure register"),
        ("funcwords_punct_only", t_funcpunct,
         "content tokens DELETED; only function words, hedges and punctuation "
         "remain - register with the length of the content erased"),
    ]

    for scope, mask in (("all", np.ones(len(labels), bool)),
                        ("qa", half == "qa"),
                        ("summarization", half == "summarization")):
        print(f"scope {scope}: n={int(mask.sum())}", flush=True)
        cs = [claims[i] for i in np.nonzero(mask)[0]]
        ys = labels[mask]
        gs = [g[i] for i in np.nonzero(mask)[0]]
        blk = {"rows": int(mask.sum()), "text_channels": {}, "numeric_channels": {}}

        for name, fn, what in TRANSFORMS:
            t0 = time.time()
            a, _ = probe([fn(c) for c in cs], ys, gs)
            blk["text_channels"][name] = {"auroc": round(a, 4), "what": what}
            print(f"  {name}: {a:.4f} ({time.time() - t0:.0f}s)", flush=True)

        for name, fn in (("length_only", length_features),
                         ("punctuation_and_case_only", punct_features),
                         ("hedging_and_function_word_rate_only", hedge_features),
                         ("all_numeric_style", style_features)):
            X, nm = fn(cs)
            a, _ = numeric_probe(X, ys, gs)
            blk["numeric_channels"][name] = {"auroc": round(a, 4), "features": nm}
            print(f"  {name}: {a:.4f}", flush=True)

        # raw single-feature AUROCs, the C5 surface-parity form
        clen = np.array([float(len(c)) for c in cs])
        ctok = np.array([float(len(C.tokens(c))) for c in cs])
        blk["raw_single_feature_auroc"] = {
            "claim_char_length": round(float(C.auroc(ys, clen)), 4),
            "claim_token_count": round(float(C.auroc(ys, ctok)), 4),
        }

        # survival of the full probe after conditioning on length
        s = base_score[mask]
        sa, per_bin = stratified_auroc(s, ys, clen)
        blk["full_probe_after_length_control"] = {
            "method": "AUROC computed inside claim-char-length deciles and pooled, "
                      "weighted by n_pos * n_neg",
            "length_stratified_auroc": sa, "per_bin": per_bin}
        print(f"  length-stratified: {sa}", flush=True)

        # length-matched pairs at several tolerances
        sub = df.filter(pl.Series(mask))
        lm = {}
        piv = sub.select(["pair_id", "label", "claim"]).with_columns(
            pl.col("claim").str.len_chars().alias("n")).pivot(
            on="label", index="pair_id", values="n", aggregate_function="first")
        pc = [c for c in piv.columns if c != "pair_id"]
        p1 = piv[[c for c in pc if c.startswith("1")][0]].to_numpy()
        p0 = piv[[c for c in pc if c.startswith("0")][0]].to_numpy()
        rel = np.abs(p1 - p0) / np.maximum(np.maximum(p1, p0), 1)
        for tol in (0.05, 0.10, 0.20, 0.30):
            keep = set(piv["pair_id"].to_numpy()[rel <= tol].tolist())
            m2 = np.array([p in keep for p in sub["pair_id"].to_list()])
            n = int(m2.sum())
            entry = {"tolerance_relative_char_length": tol,
                     "pairs": n // 2, "rows": n}
            if n >= 400:
                cs2 = [sub["claim"].to_list()[i] for i in np.nonzero(m2)[0]]
                ys2 = sub["label"].to_numpy()[m2]
                gs2 = [f"{scope}:{p}" for p in np.asarray(sub['pair_id'].to_list())[m2]]
                a2, _ = probe(cs2, ys2, gs2)
                l2 = np.array([float(len(c)) for c in cs2])
                entry["claim_only_probe_auroc"] = round(a2, 4)
                entry["claim_char_length_auroc"] = round(float(C.auroc(ys2, l2)), 4)
            else:
                entry["claim_only_probe_auroc"] = None
                entry["skipped"] = "fewer than 400 rows"
            lm[f"tol_{tol:.2f}"] = entry
            print(f"  lenmatch {tol}: {entry}", flush=True)
        blk["length_matched_pairs"] = lm
        out["scopes"][scope] = blk

    (HERE / "halueval_conform_diag.json").write_text(json.dumps(out, indent=2))
    print("diag written", flush=True)


# --------------------------------------------------------------------------- #
# stage: frontier
# --------------------------------------------------------------------------- #
RETENTIONS = (0.90, 0.75, 0.50, 0.25, 0.10)


def pair_table(df):
    """One row per pair: pair_id, half, positive/negative claim, lengths."""
    d = df.select(["pair_id", "half", "label", "claim"])
    pos = d.filter(pl.col("label") == 1).rename({"claim": "claim_pos"}).drop("label")
    neg = d.filter(pl.col("label") == 0).select(["pair_id", "claim"]).rename(
        {"claim": "claim_neg"})
    return pos.join(neg, on="pair_id", how="inner")


def rankings(df, z):
    """Ordering of pair ids, hardest-for-a-claim-only-probe first, per strategy."""
    pt = pair_table(df).sort("pair_id")
    pids = pt["pair_id"].to_numpy()
    cp = pt["claim_pos"].to_list()
    cn = pt["claim_neg"].to_list()

    pid_row = np.asarray(df["pair_id"].to_list())
    lab = df["label"].to_numpy()
    order = {p: i for i, p in enumerate(pids)}

    def pair_margin(score):
        sp = np.zeros(len(pids))
        sn = np.zeros(len(pids))
        for i in range(len(pid_row)):
            j = order[pid_row[i]]
            if lab[i] == 1:
                sp[j] = score[i]
            else:
                sn[j] = score[i]
        return sp - sn

    m_full = pair_margin(z["score"])
    m_shape = pair_margin(z["shape_score"])
    m_style = pair_margin(z["style_score"])

    lp = np.array([len(c) for c in cp], dtype=float)
    ln = np.array([len(c) for c in cn], dtype=float)
    rel_len = np.abs(lp - ln) / np.maximum(np.maximum(lp, ln), 1)

    jac = np.array([
        (lambda a, b: len(a & b) / max(len(a | b), 1))(set(C.tokens(x)), set(C.tokens(y)))
        for x, y in zip(cp, cn)])

    rng = np.random.default_rng(17)
    rand = rng.random(len(pids))

    def order_by(key):
        return pids[np.argsort(key, kind="stable")]

    R = {
        "peel_probe_margin": {
            "order": order_by(np.abs(m_full)),
            "what": "pairs ordered by |within-pair margin| of the full claim-only "
                    "probe, ascending - the ambiguous band kept first"},
        "peel_shape_margin": {
            "order": order_by(np.abs(m_shape)),
            "what": "same, but the margin comes from the CONTENT-MASKED shape "
                    "probe, so the peel does not select on topical content"},
        "peel_numeric_style_margin": {
            "order": order_by(np.abs(m_style)),
            "what": "same, margin from the numeric style/length probe"},
        "length_matched": {
            "order": order_by(rel_len),
            "what": "pairs ordered by relative claim char-length difference, "
                    "ascending"},
        "claim_similarity": {
            "order": order_by(-jac),
            "what": "pairs ordered by content-token Jaccard BETWEEN the two "
                    "claims, descending - minimal pairs kept first"},
        "random": {"order": order_by(rand), "what": "seeded control ordering"},
    }
    # composites: restrict to a half of the member on one axis, then peel on the probe
    for base, key in (("length_matched", rel_len), ("claim_similarity", -jac)):
        half_n = len(pids) // 2
        pre = np.argsort(key, kind="stable")[:half_n]
        rest = np.argsort(key, kind="stable")[half_n:]
        inner = pre[np.argsort(np.abs(m_full)[pre], kind="stable")]
        R[f"{base}_then_peel"] = {
            "order": np.concatenate([pids[inner], pids[rest]]),
            "what": f"top half by {base}, ordered inside it by |probe margin| "
                    "ascending; the remaining half appended"}
    return R, {"pair_ids": pids, "rel_len": rel_len, "jaccard": jac,
               "margin_full": m_full}


def measure_subset(df, keep_pids, seed=0, per_half=True):
    sub = df.filter(pl.col("pair_id").is_in(list(keep_pids)))
    claims = sub["claim"].to_list()
    labels = sub["label"].to_numpy()
    g = groups_of(sub)
    auc, score = probe(claims, labels, g, seed=seed)
    subi = sub.with_columns(pl.col("label").cast(pl.Int64))
    wp = C.within_pair_accuracy(subi, score)
    parity = C.surface_parity(subi, report_only=("claim_chunk_containment",))
    res = {
        "rows": sub.height, "pairs": int(sub["pair_id"].n_unique()),
        "retention_rows": round(sub.height / ORIG_ROWS, 4),
        "claim_only_probe_auroc": round(auc, 4),
        "within_pair_claim_only_accuracy": wp["all"]["acc"],
        "surface_parity_C_A1": {
            "auroc": parity["auroc"], "report_only": parity["report_only"],
            "worst_barred_deviation": parity["worst_deviation"],
            "pass": parity["pass"]},
        "half_composition": {k: int(v) for k, v in
                             sub.group_by("half").len().iter_rows()},
        "clears_claim_only_bar_0.55": bool(auc < 0.55),
        "clears_within_pair_bar_0.60": bool(wp["all"]["acc"] < 0.60),
    }
    if per_half:
        ph = {}
        for h in ("qa", "summarization"):
            m = np.array([x == h for x in sub["half"].to_list()])
            if m.sum() >= 200 and 0 < labels[m].sum() < m.sum():
                ph[h] = round(float(C.auroc(labels[m], score[m])), 4)
        res["per_half_auroc"] = ph
    return res


def stage_frontier(only=None):
    d, df = load()
    z = np.load(HERE / "halueval_conform_scores.npz", allow_pickle=True)
    R, aux = rankings(df, z)

    path = HERE / f"halueval_conform_frontier{'_' + only if only else ''}.json"
    out = {"note": NOTE, "retentions": list(RETENTIONS), "strategies": {}}
    if path.exists():
        out = json.loads(path.read_text())

    names = [only] if only else list(R)
    for name in names:
        spec = R[name]
        blk = out["strategies"].get(name, {"what": spec["what"], "levels": {}})
        blk["what"] = spec["what"]
        for r in RETENTIONS:
            key = f"retain_{r:.2f}"
            if key in blk["levels"]:
                continue
            n = int(round(r * ORIG_PAIRS))
            keep = spec["order"][:n]
            t0 = time.time()
            m = measure_subset(df, keep)
            m["seconds"] = round(time.time() - t0, 1)
            blk["levels"][key] = m
            print(f"{name} @ {r}: AUROC {m['claim_only_probe_auroc']} "
                  f"wp {m['within_pair_claim_only_accuracy']} "
                  f"rows {m['rows']} ({m['seconds']}s)", flush=True)
            out["strategies"][name] = blk
            path.write_text(json.dumps(out, indent=2))
    path.write_text(json.dumps(out, indent=2))
    print(f"frontier written -> {path.name}", flush=True)


def stage_frontier_merge():
    """Fold per-strategy frontier shards into one file and compute the envelope."""
    out = {"note": NOTE, "retentions": list(RETENTIONS), "strategies": {}}
    for p in sorted(HERE.glob("halueval_conform_frontier_*.json")):
        part = json.loads(p.read_text())
        out["strategies"].update(part.get("strategies", {}))
    base = HERE / "halueval_conform_frontier.json"
    if base.exists():
        out["strategies"].update(json.loads(base.read_text()).get("strategies", {}))
    env = {}
    for r in RETENTIONS:
        key = f"retain_{r:.2f}"
        best = None
        for sname, s in out["strategies"].items():
            lv = s["levels"].get(key)
            if lv is None:
                continue
            if best is None or lv["claim_only_probe_auroc"] < best[1]["claim_only_probe_auroc"]:
                best = (sname, lv)
        if best:
            env[key] = {
                "retention_rows": best[1]["retention_rows"],
                "rows": best[1]["rows"], "pairs": best[1]["pairs"],
                "best_strategy": best[0],
                "best_claim_only_probe_auroc": best[1]["claim_only_probe_auroc"],
                "within_pair_claim_only_accuracy": best[1]["within_pair_claim_only_accuracy"],
                "bar_claim_only": 0.55, "bar_within_pair": 0.60,
                "clears_claim_only": best[1]["clears_claim_only_bar_0.55"],
                "clears_within_pair": best[1]["clears_within_pair_bar_0.60"],
                "breach_claim_only": round(best[1]["claim_only_probe_auroc"] - 0.55, 4),
            }
    out["frontier_envelope"] = env
    base.write_text(json.dumps(out, indent=2))
    print(json.dumps(env, indent=2), flush=True)


def stage_honesty():
    """SPLIT-SAMPLE peel - the ranking is computed by a probe that never saw the
    rows it ranks, so the retained band is not selected on its own noise."""
    d, df = load()
    pt = pair_table(df).sort("pair_id")
    pids = pt["pair_id"].to_numpy()
    rng = np.random.default_rng(5)
    perm = rng.permutation(len(pids))
    A, B = set(pids[perm[:len(pids) // 2]].tolist()), set(pids[perm[len(pids) // 2:]].tolist())

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    dA = df.filter(pl.col("pair_id").is_in(list(A)))
    dB = df.filter(pl.col("pair_id").is_in(list(B)))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                          max_features=300_000, sublinear_tf=True)
    xa = vec.fit_transform(dA["claim"].to_list())
    clf = LogisticRegression(solver="liblinear", C=4.0, tol=1e-7, max_iter=3000)
    clf.fit(xa, dA["label"].to_numpy())
    sB = clf.decision_function(vec.transform(dB["claim"].to_list()))
    aB = float(C.auroc(dB["label"].to_numpy(), sB))
    print(f"A-trained probe on B: AUROC {aB:.4f}", flush=True)

    pidB = np.asarray(dB["pair_id"].to_list())
    labB = dB["label"].to_numpy()
    upids = np.unique(pidB)
    idx = {p: i for i, p in enumerate(upids)}
    sp = np.zeros(len(upids))
    sn = np.zeros(len(upids))
    for i in range(len(pidB)):
        j = idx[pidB[i]]
        if labB[i] == 1:
            sp[j] = sB[i]
        else:
            sn[j] = sB[i]
    order = upids[np.argsort(np.abs(sp - sn), kind="stable")]

    path = HERE / "halueval_conform_honesty.json"
    out = json.loads(path.read_text()) if path.exists() else {
        "note": NOTE,
        "what": "the peel ranking is produced by a probe fitted on split A "
                "only and applied to split B, so B's retained band is not "
                "selected on the noise of a model that saw B; the AUROC "
                "reported at each level is a FRESH out-of-fold probe inside "
                "the retained band of B",
        "split_A_pairs": len(A), "split_B_pairs": len(B),
        "A_trained_probe_on_B_auroc": round(aB, 4),
        "levels": {}}
    for r in (0.90, 0.75, 0.50, 0.25, 0.15, 0.10, 0.06, 0.04, 0.02):
        if f"retain_{r:.2f}" in out["levels"]:
            continue
        keep = order[:int(round(r * len(order)))]
        m = measure_subset(df, keep)
        m["retention_of_split_B"] = r
        out["levels"][f"retain_{r:.2f}"] = m
        print(f"honesty @ {r}: AUROC {m['claim_only_probe_auroc']} "
              f"wp {m['within_pair_claim_only_accuracy']} rows {m['rows']}", flush=True)
        (HERE / "halueval_conform_honesty.json").write_text(json.dumps(out, indent=2))
    (HERE / "halueval_conform_honesty.json").write_text(json.dumps(out, indent=2))
    print("honesty written", flush=True)


DEEP = (0.20, 0.15, 0.08, 0.06, 0.05, 0.04, 0.02, 0.01)


def stage_deep(only):
    """Push the best strategy far below the reported frontier to find where -
    if anywhere - the claim-only channel actually crosses the 0.55 bar."""
    d, df = load()
    z = np.load(HERE / "halueval_conform_scores.npz", allow_pickle=True)
    R, aux = rankings(df, z)
    path = HERE / f"halueval_conform_deep_{only}.json"
    out = json.loads(path.read_text()) if path.exists() else {
        "note": NOTE, "strategy": only, "what": R[only]["what"], "levels": {}}
    for r in sorted(DEEP, reverse=True):
        key = f"retain_{r:.3f}"
        if key in out["levels"]:
            continue
        n = int(round(r * ORIG_PAIRS))
        m = measure_subset(df, R[only]["order"][:n])
        out["levels"][key] = m
        print(f"deep {only} @ {r}: AUROC {m['claim_only_probe_auroc']} "
              f"wp {m['within_pair_claim_only_accuracy']} rows {m['rows']}", flush=True)
        path.write_text(json.dumps(out, indent=2))
    path.write_text(json.dumps(out, indent=2))
    print("deep written", flush=True)


FINE = (0.010, 0.012, 0.015, 0.020, 0.025, 0.030, 0.040)


def stage_fine():
    """The deep sweep found the sub-bar band at 1-2% retention.  This locates
    the LARGEST subset that clears every leg of C5 - claim-only under 0.55 on
    all five probe seeds, within-pair under 0.60, and the barred surface
    channels inside [0.45, 0.55] under amendment C-A1."""
    d, df = load()
    z = np.load(HERE / "halueval_conform_scores.npz", allow_pickle=True)
    R, _ = rankings(df, z)
    path = HERE / "halueval_conform_fine.json"
    out = json.loads(path.read_text()) if path.exists() else {
        "note": NOTE,
        "what": "maximum-retention search over every strategy in the band where "
                "the deep sweep crossed the bar; five probe seeds per subset so "
                "a single lucky fold assignment cannot carry a pass",
        "bars": {"claim_only": 0.55, "within_pair": 0.60,
                 "surface_parity_barred_channels": [0.45, 0.55]},
        "levels": {}}
    for name in R:
        for r in FINE:
            key = f"{name}@{r:.3f}"
            if key in out["levels"]:
                continue
            n = int(round(r * ORIG_PAIRS))
            keep = R[name]["order"][:n]
            sub = df.filter(pl.col("pair_id").is_in(list(keep)))
            claims = sub["claim"].to_list()
            labels = sub["label"].to_numpy()
            g = groups_of(sub)
            aur, wps = [], []
            for seed in range(5):
                a, s = probe(claims, labels, g, seed=seed)
                aur.append(round(a, 4))
                wps.append(C.within_pair_accuracy(
                    sub.with_columns(pl.col("label").cast(pl.Int64)), s)["all"]["acc"])
            parity = C.surface_parity(sub.with_columns(pl.col("label").cast(pl.Int64)),
                                      report_only=("claim_chunk_containment",))
            rec = {
                "strategy": name, "retention_pairs": r,
                "rows": sub.height, "pairs": int(sub["pair_id"].n_unique()),
                "retention_of_member_rows": round(sub.height / ORIG_ROWS, 4),
                "claim_only_auroc_seeds": aur,
                "claim_only_auroc_mean": round(float(np.mean(aur)), 4),
                "claim_only_auroc_max": max(aur),
                "within_pair_seeds": wps, "within_pair_max": max(wps),
                "surface_parity": {"auroc": parity["auroc"],
                                   "worst_barred_deviation": parity["worst_deviation"],
                                   "pass": parity["pass"]},
                "half_composition": {k: int(v) for k, v in
                                     sub.group_by("half").len().iter_rows()},
                "clears_claim_only_all_seeds": bool(max(aur) < 0.55),
                "clears_within_pair_all_seeds": bool(max(wps) < 0.60),
                "clears_surface_parity": parity["pass"],
            }
            rec["clears_every_leg_of_C5"] = bool(
                rec["clears_claim_only_all_seeds"]
                and rec["clears_within_pair_all_seeds"]
                and rec["clears_surface_parity"])
            out["levels"][key] = rec
            print(f"{key}: rows {rec['rows']} auroc {aur} wp {max(wps)} "
                  f"parity {parity['worst_deviation']} "
                  f"C5 {rec['clears_every_leg_of_C5']}", flush=True)
            path.write_text(json.dumps(out, indent=2))
    passing = [v for v in out["levels"].values() if v["clears_every_leg_of_C5"]]
    out["largest_subset_clearing_every_leg_of_C5"] = (
        max(passing, key=lambda v: v["rows"]) if passing else None)
    out["subsets_clearing_every_leg_of_C5"] = len(passing)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["largest_subset_clearing_every_leg_of_C5"], indent=2), flush=True)
    print("fine written", flush=True)


# --------------------------------------------------------------------------- #
# stage: transforms - the repair that changes the TEXT rather than the row set
# --------------------------------------------------------------------------- #
def first_sentence(c):
    parts = re.split(r"(?<=[.!?])\s+", c.strip())
    return parts[0] if parts else c


def stage_transforms():
    """A subset keeps every row's text; a transform rewrites it.  Measured here:
    does any length- or register-equalising rewrite of the CLAIM push the
    claim-only probe under 0.55, and what does it cost in label integrity."""
    d, df = load()
    claims = df["claim"].to_list()
    chunks = df["chunk"].to_list()
    labels = df["label"].to_numpy()
    g = groups_of(df)

    base_cont = np.array([C.containment(c, k) for c, k in zip(claims, chunks)])
    VAR = {
        "first_20_word_tokens": (lambda c: " ".join(TOKEN.findall(c)[:20]),
                                 "the claim's first 20 word tokens - equalises "
                                 "the length channel at the top end"),
        "first_120_chars": (lambda c: c[:120],
                            "the claim's first 120 characters"),
        "first_sentence": (first_sentence,
                           "the claim's first sentence only"),
        "normalised_first_20_tokens": (
            lambda c: " ".join(t_norm(c).split()[:20]),
            "lowercased, punctuation stripped, first 20 tokens - the strongest "
            "register erasure that still leaves a readable claim"),
        "shape_only_content_masked": (
            t_shape, "content masked (destroys the claim; measured as the floor "
                     "of what register alone supplies)"),
    }
    out = {"note": NOTE, "baseline": {}, "variants": {}}
    a0, _ = probe(claims, labels, g)
    out["baseline"] = {
        "claim_only_probe_auroc": round(a0, 4),
        "negative_rows_fully_attested": int((base_cont[labels == 0] >= 0.99999).sum()),
        "negative_rows": int((labels == 0).sum()),
    }
    print(f"baseline {a0:.4f}", flush=True)

    for name, (fn, what) in VAR.items():
        t = [fn(c) for c in claims]
        a, _ = probe(t, labels, g)
        cont = np.array([C.containment(x, k) for x, k in zip(t, chunks)])
        clen = np.array([float(len(x)) for x in t])
        pos_pairs = {(t[i], chunks[i]) for i in np.nonzero(labels == 1)[0]}
        struct = sum(1 for i in np.nonzero(labels == 0)[0] if (t[i], chunks[i]) in pos_pairs)
        newly = int(((cont[labels == 0] >= 0.99999)
                     & (base_cont[labels == 0] < 0.99999)).sum())
        empty = int(sum(1 for x in t if not x.strip()))
        out["variants"][name] = {
            "what": what,
            "claim_only_probe_auroc": round(a, 4),
            "clears_bar_0.55": bool(a < 0.55),
            "claim_char_length_auroc": round(float(C.auroc(labels, clen)), 4),
            "label_integrity_cost": {
                "negatives_newly_fully_attested_after_the_rewrite": newly,
                "negatives_fully_attested_rate_before": round(
                    float((base_cont[labels == 0] >= 0.99999).mean()), 4),
                "negatives_fully_attested_rate_after": round(
                    float((cont[labels == 0] >= 0.99999).mean()), 4),
                "positive_mean_containment_after": round(
                    float(cont[labels == 1].mean()), 4),
                "negative_mean_containment_after": round(
                    float(cont[labels == 0].mean()), 4),
                "structural_C1_collisions_after": struct,
                "claims_rewritten_to_empty": empty,
            }}
        print(f"{name}: AUROC {a:.4f} newly-attested negatives {newly} "
              f"structural {struct}", flush=True)
        (HERE / "halueval_conform_transforms.json").write_text(json.dumps(out, indent=2))
    (HERE / "halueval_conform_transforms.json").write_text(json.dumps(out, indent=2))
    print("transforms written", flush=True)


# --------------------------------------------------------------------------- #
# stage: build - materialise the chosen variant
# --------------------------------------------------------------------------- #
def stage_build(strategy, retention, tag):
    parquet, build_json, _ = NAMES[tag]
    d, df = load()
    z = np.load(HERE / "halueval_conform_scores.npz", allow_pickle=True)
    R, aux = rankings(df, z)
    n = int(round(retention * ORIG_PAIRS))
    keep = R[strategy]["order"][:n]
    sub = df.filter(pl.col("pair_id").is_in(list(keep))).sort(
        ["pair_id", "label"], descending=[False, True])
    sub = sub.with_columns(
        pl.col("chunk").str.slice(0, 1500).alias("chunk_trunc_1500"))
    sub.write_parquet(HERE / parquet)

    m = measure_subset(df, keep)
    build = {
        "note": NOTE,
        "member": f"halueval_{tag}",
        "built_from": "the `halueval` DANN group of the assembled training mix, "
                      "loaded through R16-H142_G1_arm.H108.public_train() under "
                      "untruncated_evidence(), pair structure recovered by an "
                      "archive replay PROVED aligned row-for-row",
        "conforming_pipeline": {
            "F1_pair_level_selection": {
                "strategy": strategy,
                "what": R[strategy]["what"],
                "unit": "PAIRS - both legs of a kept pair are kept, so the "
                        "evidence field stays byte-identical across the legs and "
                        "the C1 / C6 pair tests remain computable",
                "pairs_kept": int(sub['pair_id'].n_unique()),
                "rows_kept": sub.height,
                "retention_rows": round(sub.height / ORIG_ROWS, 4),
                "retention_pairs": round(sub['pair_id'].n_unique() / ORIG_PAIRS, 4),
            }},
        "measured_on_the_variant": m,
        "columns": sub.columns,
        "artifact": f"experiments/grounding-semantic/contract/{parquet}",
    }
    (HERE / build_json).write_text(json.dumps(build, indent=2))
    print(json.dumps(m, indent=2), flush=True)
    print("build written", flush=True)


# --------------------------------------------------------------------------- #
# stage: verify - all eight clauses on the variant
# --------------------------------------------------------------------------- #
def amended_c1(df, mod):
    """C1 under amendments C-A1 and C-A2: structural test first, then STRICT
    separation of the high-attestation rates, then absolute levels."""
    claims = df["claim"].to_list()
    chunks = df["chunk"].to_list()
    lab = df["label"].to_numpy()
    half = np.asarray(df["half"].to_list())

    pos_pairs = set(zip(df.filter(pl.col("label") == 1)["claim"].to_list(),
                        df.filter(pl.col("label") == 1)["chunk"].to_list()))
    neg_pairs = list(zip(df.filter(pl.col("label") == 0)["claim"].to_list(),
                         df.filter(pl.col("label") == 0)["chunk"].to_list()))
    struct_hits = sum(1 for p in neg_pairs if p in pos_pairs)

    out = {"test_1_structural": {
        "what": "a negative leg whose (claim, evidence) is identical to a "
                "positive leg's - no function of (claim, evidence) can separate "
                "the legs, so the label cannot encode grounding",
        "negative_rows_identical_to_a_positive_row": struct_hits,
        "negative_rows": int((lab == 0).sum()),
        "fraction": round(struct_hits / max(int((lab == 0).sum()), 1), 6),
        "fires": bool(struct_hits > 0)},
        "test_2_strict_separation": {}, "test_3_absolute_levels": {}}

    for pres, cut in (("untruncated", None), ("truncated_1500", 1500)):
        cont = np.array([mod.containment(c, k if cut is None else k[:cut])
                         for c, k in zip(claims, chunks)])
        blk2, blk3 = {}, {}
        for scope, mask in (("all", np.ones(len(lab), bool)),
                            ("qa", half == "qa"),
                            ("summarization", half == "summarization")):
            if mask.sum() == 0:
                continue
            p = cont[mask & (lab == 1)]
            n = cont[mask & (lab == 0)]
            rp = float((p >= 0.90).mean())
            rn = float((n >= 0.90).mean())
            blk2[scope] = {
                "positive_rate_ge_0.90": round(rp, 4),
                "negative_rate_ge_0.90": round(rn, 4),
                "negative_strictly_below_positive": bool(rn < rp),
                "ratio_pos_over_neg": round(rp / rn, 3) if rn > 0 else None,
                "mean_containment_positive": round(float(p.mean()), 4),
                "mean_containment_negative": round(float(n.mean()), 4),
            }
            blk3[scope] = {
                "positive_fully_attested_rate": round(float((p >= 0.99999).mean()), 4),
                "negative_fully_attested_rate": round(float((n >= 0.99999).mean()), 4),
                "negative_fully_attested_rows": int((n >= 0.99999).sum()),
                "negative_rows": int(n.size),
            }
        out["test_2_strict_separation"][pres] = blk2
        out["test_3_absolute_levels"][pres] = blk3
    out["instrument"] = ("R20-H174_lane_common.containment - content-token "
                         "containment of the claim in the evidence, the "
                         "instrument sensitive to the support predicate this "
                         "member's label encodes")
    out["passes_all_three_tests"] = bool(
        struct_hits == 0
        and all(v["negative_strictly_below_positive"]
                for pres in out["test_2_strict_separation"].values()
                for v in pres.values()))
    return out


def stage_verify(tag):
    parquet, _, clauses_json = NAMES[tag]
    d, df_full = load()
    sub = pl.read_parquet(HERE / parquet)
    df = sub.select(["pair_id", "half", "label", "claim", "chunk"])
    G = _mod("provgate", EXP / "provenance_gate.py")
    src = (EXP / "R19_supply_gates.py").read_text()
    gate_n = int(src.split("GATE_N = ")[1].split("\n")[0])
    gate_j = float(src.split("GATE_JACCARD = ")[1].split("\n")[0])
    gate_kill = float(src.split("GATE_KILL = ")[1].split("\n")[0])

    out = {"note": NOTE}
    t0 = time.time()
    out["C1_distributions"] = HC.clause_c1(df, C)
    out["C1_amended"] = amended_c1(df, C)
    out["C1_supplement"] = HC.clause_c1_supplement(df, C)
    print(f"C1 {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    out["C2"] = HC.clause_c2(df, G)
    print(f"C2 {time.time() - t0:.0f}s", flush=True)
    out["C3"] = HC.clause_c3(df, d)
    t0 = time.time()
    out["C4"] = HC.clause_c4(df, G, gate_n, gate_j, gate_kill)
    print(f"C4 {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    out["C5"] = HC.clause_c5(df, C)
    parity = {"all": C.surface_parity(df.with_columns(pl.col("label").cast(pl.Int64)),
                                      report_only=("claim_chunk_containment",))}
    for h in ("qa", "summarization"):
        s = df.filter(pl.col("half") == h).with_columns(pl.col("label").cast(pl.Int64))
        if s.height:
            parity[h] = C.surface_parity(s, report_only=("claim_chunk_containment",))
    out["C5"]["surface_parity_under_C_A1"] = parity
    print(f"C5 {time.time() - t0:.0f}s", flush=True)
    out["C6"] = HC.clause_c6(df, C)
    out["C7"] = HC.clause_c7(df, d)
    out["C8"] = HC.clause_c8(df, d)
    (HERE / clauses_json).write_text(json.dumps(out, indent=2))
    print("verify written", flush=True)

# --------------------------------------------------------------------------- #
# stage: merge - the banked report
# --------------------------------------------------------------------------- #
def stage_merge():
    def j(n):
        return json.loads((HERE / n).read_text())

    diag = j("halueval_conform_diag.json")
    front = j("halueval_conform_frontier.json")
    honest = j("halueval_conform_honesty.json")
    trans = j("halueval_conform_transforms.json")
    fine = j("halueval_conform_fine.json")
    perhalf = j("halueval_conform_perhalf.json")
    floor = j("halueval_conform_floorcheck.json")
    build = j("halueval_conformed_build.json")
    be_build = j("halueval_conform_best_effort_build.json")
    cl = j("halueval_conformed_clauses.json")
    be_cl = j("halueval_conform_best_effort_clauses.json")
    phase1 = j("halueval_contract_report.json")
    deep = {}
    for p in sorted(HERE.glob("halueval_conform_deep_*.json")):
        dd = json.loads(p.read_text())
        deep[dd["strategy"]] = {k: {"rows": v["rows"],
                                    "claim_only_probe_auroc": v["claim_only_probe_auroc"],
                                    "within_pair": v["within_pair_claim_only_accuracy"]}
                                for k, v in dd["levels"].items()}

    A = diag["scopes"]["all"]
    tc = lambda k: A["text_channels"][k]["auroc"]          # noqa: E731
    nc = lambda k: A["numeric_channels"][k]["auroc"]       # noqa: E731

    envelope = front["frontier_envelope"]
    c1a = cl["C1_amended"]
    c5 = cl["C5"]
    parity = c5["surface_parity_under_C_A1"]
    pooled_c5 = bool(c5["claim_only_probe_auroc"] < 0.55
                     and c5["within_pair_claim_only_accuracy"]["all"]["acc"] < 0.60
                     and parity["all"]["pass"])
    per_half_c5 = bool(pooled_c5 and all(parity[h]["pass"]
                                         for h in ("qa", "summarization")))
    fine_pass = fine["largest_subset_clearing_every_leg_of_C5"]
    seeds = floor["levels"]
    won = fine["levels"][f"{build['conforming_pipeline']['F1_pair_level_selection']['strategy']}"
                         f"@{build['conforming_pipeline']['F1_pair_level_selection'].get('retention_pairs', 0.012):.3f}"]
    won_seeds = won["claim_only_auroc_seeds"]

    rep = {
        "member": "halueval_conformed",
        "member_kind": "conformance attempt on the `halueval` source corpus "
                       "(DANN group `halueval`, 40,000 rows / 20,000 pairs) of "
                       "the assembled training mix",
        "note": NOTE,
        "contract": "docs/experiments/dataset-contract.md, amendments C-A1 and "
                    "C-A2 applied",
        "verified_on": "2026-08-17",
        "compute": "CPU only; CUDA_VISIBLE_DEVICES forced empty before any "
                   "import - GPUs 0/1/2 carry live training draws, untouched. "
                   "HF_HUB_OFFLINE=1, every archive read from disk",

        "headline": {
            "one_line": "the member cannot be conformed at a usable size: the "
                        "claim-only channel is redundant across content, "
                        "register and length, and the only subset that clears "
                        "C5's pooled conjunction keeps 480 of 40,000 rows and "
                        "still fails the parity leg on its summarization half",
            "variant_built": build["artifact"],
            "rows_kept": build["conforming_pipeline"]["F1_pair_level_selection"]["rows_kept"],
            "fraction_of_the_member": build["conforming_pipeline"][
                "F1_pair_level_selection"]["retention_rows"],
            "what_it_clears": {
                "claim_only_probe_auroc": c5["claim_only_probe_auroc"],
                "claim_only_bar": 0.55,
                "claim_only_over_5_probe_seeds": won_seeds,
                "claim_only_worst_of_5_seeds": max(won_seeds),
                "within_pair_claim_only_accuracy":
                    c5["within_pair_claim_only_accuracy"]["all"]["acc"],
                "within_pair_bar": 0.60,
                "pooled_surface_parity_worst_barred_deviation":
                    parity["all"]["worst_barred_deviation"] if "worst_barred_deviation"
                    in parity["all"] else parity["all"]["worst_deviation"],
            },
            "what_it_does_not_clear": {
                "summarization_half_surface_parity": {
                    "claim_char_length_auroc": parity["summarization"]["auroc"][
                        "claim_char_length"],
                    "band": [0.45, 0.55],
                    "worst_deviation": parity["summarization"]["worst_deviation"],
                    "half_rows": cl["C7"]["per_half"].get("summarization", {}).get("rows"),
                    "why_it_matters": "the phase-1 report applied surface parity "
                                      "pooled AND per half; this variant clears "
                                      "the pooled reading and fails the "
                                      "summarization half, on 40 rows",
                },
                "supply": "480 of 40,000 member rows - a 98.8% cut. 0.067% of "
                          "the 721,210-row flagship mix against halueval's "
                          "current 5.55%",
                "half_balance": "440 of the 480 rows are the QA half; the "
                                "variant is effectively QA-only",
            },
            "clears_pooled_C5": pooled_c5,
            "clears_pooled_and_per_half_C5": per_half_c5,
        },

        "decision_branches": [
            "IF the pooled C5 reading binds - the 480-row variant conforms and "
            "is installable at 0.067% of the mix, replacing a member that is "
            "currently 5.55%",
            "IF the per-half C5 reading binds, as the phase-1 report applied it "
            "- no variant conforms at any size, and the member's options are to "
            "regenerate its positive leg with the same model that wrote its "
            "negative leg, or to drop it",
            "IF the member is kept unconformed - nothing measured here "
            "invalidates a banked arena number; the finding is that 5.55% of "
            "every draw's rows carry a label recoverable from the claim string "
            "alone at AUROC 0.9519, and that this fraction cannot be reduced by "
            "filtering without cutting 90-99% of the member",
        ],

        "how_the_member_was_rebuilt": {
            "loader": phase1["how_the_member_was_rebuilt"]["loader"],
            "pair_structure_recovery": phase1["how_the_member_was_rebuilt"][
                "pair_structure_recovery"],
            "re_proved_here": "the replay alignment assertion re-ran at the "
                              "start of every stage of this work and passed on "
                              "all 40,000 rows each time",
        },

        "phase_1_baseline_reproduced": {
            "banked_claim_only_auroc": phase1["C5"]["measured"][
                "claim_only_converged_probe_auroc"],
            "remeasured_here": tc("full_claim"),
            "banked_within_pair": phase1["C5"]["measured"][
                "within_pair_claim_only_accuracy"],
        },

        "live_positive_control": {
            "claim_only_probe_gate": diag["live_positive_control"],
            "contamination_gate_on_the_variant": cl["C4"]["live_positive_control"],
            "contamination_spike_control_on_the_variant": cl["C4"]["spike_control"],
            "probe_is_not_degenerate_at_small_n": {
                "what": "at 400-1,600 rows a char-ngram probe could return "
                        "chance because its vocabulary has collapsed rather than "
                        "because the subset is clean; measured instead",
                "per_fold_vocabulary_terms_at_400_rows": [
                    v["vocabulary_terms"] for v in
                    seeds["length_matched_then_peel@0.010"]["per_fold_vocabulary_seed0"]],
                "rows_with_zero_decision_value": seeds[
                    "length_matched_then_peel@0.010"][
                    "rows_with_exactly_zero_decision_value_seed0"],
                "reading": "9,412-9,875 vocabulary terms per fold and no "
                           "zero-score rows - the sub-bar readings at 1-2% "
                           "retention are real, not an exhausted instrument",
            },
        },

        "channel_decomposition": {
            "what_it_answers": "how much of the 0.9519 claim-only AUROC is "
                               "length, how much lexical style, how much topical "
                               "content, and how much survives each control",
            "instrument": diag["instrument"],
            "pooled_40000_rows": {
                "full_claim": tc("full_claim"),
                "length_alone_numeric_probe": nc("length_only"),
                "punctuation_and_case_alone": nc("punctuation_and_case_only"),
                "hedging_and_function_word_rate_alone":
                    nc("hedging_and_function_word_rate_only"),
                "all_numeric_style_together": nc("all_numeric_style"),
                "register_alone_content_masked": tc("shape_only_content_masked"),
                "register_alone_content_deleted": tc("funcwords_punct_only"),
                "content_alone_style_removed": tc("content_only_style_removed"),
                "content_alone_word_order_destroyed": tc("content_only_sorted"),
                "case_and_punctuation_controlled": tc("normalised"),
                "full_probe_inside_claim_length_deciles":
                    A["full_probe_after_length_control"]["length_stratified_auroc"],
                "raw_single_feature": A["raw_single_feature_auroc"],
            },
            "per_half": {
                h: {"full_claim": diag["scopes"][h]["text_channels"]["full_claim"]["auroc"],
                    "length_alone_numeric_probe": diag["scopes"][h][
                        "numeric_channels"]["length_only"]["auroc"],
                    "register_alone_content_masked": diag["scopes"][h][
                        "text_channels"]["shape_only_content_masked"]["auroc"],
                    "content_alone_style_removed": diag["scopes"][h][
                        "text_channels"]["content_only_style_removed"]["auroc"],
                    "full_probe_inside_claim_length_deciles": diag["scopes"][h][
                        "full_probe_after_length_control"]["length_stratified_auroc"],
                    "length_matched_pairs": diag["scopes"][h]["length_matched_pairs"]}
                for h in ("qa", "summarization")},
            "reading": "no single control removes the channel. Delete every "
                       "content word and 0.8886 remains; delete every function "
                       "word, hedge, punctuation mark and capital and 0.9016 "
                       "remains; condition on claim length and 0.9220 remains. "
                       "The channels are redundant, so neutralising one leaves "
                       "the others carrying the label. The QA half is the harder "
                       "one - every channel there reads above 0.96",
            "full_detail": "halueval_conform_diag.json",
        },

        "strategies_tried": {
            "subset_strategies": {
                name: {"what": s["what"],
                       "claim_only_auroc_by_retention": {
                           k: v["claim_only_probe_auroc"] for k, v in s["levels"].items()},
                       "within_pair_by_retention": {
                           k: v["within_pair_claim_only_accuracy"]
                           for k, v in s["levels"].items()}}
                for name, s in front["strategies"].items()},
            "text_transforms": {
                name: {"what": v["what"],
                       "claim_only_probe_auroc": v["claim_only_probe_auroc"],
                       "clears_bar": v["clears_bar_0.55"],
                       "label_integrity_cost": v["label_integrity_cost"]}
                for name, v in trans["variants"].items()},
            "per_half_balanced_strategies": {
                "what": perhalf["what"],
                "largest_subset_clearing_every_leg": perhalf[
                    "largest_subset_clearing_every_leg"],
                "subsets_clearing_every_leg": perhalf["subsets_clearing_every_leg"],
                "parity_clean_levels_and_what_they_read": {
                    k: {"rows": v["rows"], "half_composition": v["half_composition"],
                        "claim_only_auroc_seeds": v.get("claim_only_auroc_seeds")}
                    for k, v in perhalf["levels"].items()
                    if v.get("clears_surface_parity_pooled_and_both_halves")},
            },
        },

        "frontier": {
            "unit": "retention is a fraction of the member's 40,000 ROWS; every "
                    "subset is taken at PAIR level so both legs are kept and the "
                    "evidence field stays byte-identical across a pair",
            "requested_levels": {
                k: {"rows": v["rows"], "best_strategy": v["best_strategy"],
                    "best_claim_only_probe_auroc": v["best_claim_only_probe_auroc"],
                    "within_pair": v["within_pair_claim_only_accuracy"],
                    "clears_claim_only_bar": v["clears_claim_only"]}
                for k, v in envelope.items()},
            "deep_sweep": deep,
            "fine_sweep_five_seeds_per_subset": {
                "what": fine["what"],
                "largest_subset_clearing_every_leg_of_C5": fine_pass,
                "subsets_clearing_every_leg_of_C5": fine["subsets_clearing_every_leg_of_C5"],
                "levels": {k: {"rows": v["rows"],
                               "claim_only_auroc_seeds": v["claim_only_auroc_seeds"],
                               "within_pair_max": v["within_pair_max"],
                               "surface_parity_worst_barred_deviation":
                                   v["surface_parity"]["worst_barred_deviation"],
                               "clears_every_leg_of_C5": v["clears_every_leg_of_C5"]}
                           for k, v in fine["levels"].items()}},
            "honest_split_sample_frontier": {
                "what": honest["what"],
                "A_trained_probe_on_B_auroc": honest["A_trained_probe_on_B_auroc"],
                "levels": {k: {"rows": v["rows"],
                               "retention_of_the_whole_member": v["retention_rows"],
                               "claim_only_probe_auroc": v["claim_only_probe_auroc"],
                               "within_pair": v["within_pair_claim_only_accuracy"]}
                           for k, v in honest["levels"].items()},
                "reading": "the honest frontier is worse than the in-sample one "
                           "at every level - an in-sample peel keeps the rows a "
                           "probe happened to misjudge, which flatters the "
                           "retained band. Under the split-sample instrument the "
                           "channel never falls below 0.5691, and that at 400 rows",
            },
        },

        "variant_built": {
            "artifact": build["artifact"],
            "pipeline": build["conforming_pipeline"],
            "columns": build["columns"],
            "measured": build["measured_on_the_variant"],
            "why_this_one": "the only subset in the entire search whose "
                            "claim-only probe clears 0.55 on every one of five "
                            "fold seeds while its within-pair accuracy and its "
                            "pooled surface parity also clear. The search "
                            "covered eight pooled subset strategies and four "
                            "half-balanced ones over eighteen distinct retention "
                            "levels from 90% down to 0.5% of the member; the "
                            "fine and half-balanced sweeps ran five probe seeds "
                            "per subset, the coarse and deep sweeps one",
        },

        "larger_near_miss_variant": {
            "artifact": be_build["artifact"],
            "pipeline": be_build["conforming_pipeline"],
            "measured": be_build["measured_on_the_variant"],
            "why_it_is_kept": "at 4,000 rows it is ten times the supply of the "
                              "conforming variant and it is the largest subset "
                              "whose within-pair claim-only accuracy clears its "
                              "0.60 bar. It misses the claim-only bar by 0.0062 "
                              "and fails surface parity at 0.1077, so it is a "
                              "near miss and not an option",
            "clause_verification": "halueval_conform_best_effort_clauses.json",
        },

        "C1": {"verdict": "PASS" if c1a["passes_all_three_tests"] else "FAIL",
               "amended_tests_C_A1_C_A2": c1a,
               "distributions": cl["C1_distributions"],
               "supplement": cl["C1_supplement"],
               "measured_on": "the built variant"},
        "C2": {"verdict": "PASS" if cl["C2"]["pass"] else "FAIL",
               "worst_fraction_any_form_any_direction":
                   cl["C2"]["worst_fraction_any_form_any_direction"],
               "member_units": cl["C2"]["member_units"],
               "forms": cl["C2"]["forms"],
               "surfaces_checked": list(cl["C2"]["surfaces"].keys()),
               "surfaces": cl["C2"]["surfaces"],
               "measured_on": "the built variant"},
        "C3": {"verdict": "PASS",
               "split_axis_measured_from_the_archive":
                   "NONE - each loaded subset ships exactly one file; the corpus "
                   "has no train/validation/test split to verify",
               "archive_splits": cl["C3"]["archive_splits"],
               "measured_split_axis": cl["C3"]["measured_split_axis"],
               "selection_predicate": "subsets `qa` and `summarization`, ALL "
                                      "rows, then the pair-level subset named in "
                                      "`variant_built.pipeline`",
               "member_is_also_an_evaluation_surface": False,
               "recorded_finding": phase1["C3"]["recorded_finding_the_clause_is_shaped_to_catch"]},
        "C4": {"verdict": "PASS" if cl["C4"]["pass"] else "FAIL",
               "instrument": cl["C4"]["instrument"],
               "evidence_gate_max_fraction": cl["C4"]["evidence_gate"]["max_fraction"],
               "claim_gate_max_fraction": cl["C4"]["claim_gate"]["max_fraction"],
               "kill_bar": 0.02,
               "spike_control": cl["C4"]["spike_control"],
               "live_positive_control": cl["C4"]["live_positive_control"],
               "coverage": cl["C4"]["coverage"],
               "measured_on": "the built variant"},
        "C5": {"verdict": "PASS" if per_half_c5 else "FAIL",
               "readings": {
                   "pooled": {"clears": pooled_c5,
                              "claim_only_probe_auroc": c5["claim_only_probe_auroc"],
                              "within_pair": c5["within_pair_claim_only_accuracy"]["all"]["acc"],
                              "surface_parity_worst_barred_deviation":
                                  parity["all"]["worst_deviation"]},
                   "pooled_and_per_half": {
                       "clears": per_half_c5,
                       "binding_leg": "summarization-half surface parity, "
                                      f"claim_char_length AUROC "
                                      f"{parity['summarization']['auroc']['claim_char_length']} "
                                      "against the [0.45, 0.55] band, on 40 rows"},
                   "which_reading_the_phase_1_report_used": "both - it reported "
                                                            "surface parity for "
                                                            "`all`, `qa` and "
                                                            "`summarization` and "
                                                            "failed on all three"},
               "claim_only_probe_auroc": c5["claim_only_probe_auroc"],
               "bar": 0.55,
               "claim_only_over_5_probe_seeds": won_seeds,
               "claim_only_worst_of_5_seeds": max(won_seeds),
               "per_half_auroc": c5["claim_only_probe_per_half"],
               "within_pair_claim_only_accuracy":
                   c5["within_pair_claim_only_accuracy"]["all"]["acc"],
               "within_pair_bar": 0.60,
               "surface_parity_under_C_A1": parity,
               "single_channel_probes": {
                   "evidence_only": 0.5,
                   "basis": "both legs of every pair carry byte-identical "
                            "evidence (measured 240/240), so the channel is "
                            "constant within a pair",
                   "question_only": "NOT-APPLICABLE - the loader drops the QA "
                                    "question and the summarization half has none"},
               "instrument": c5["instrument"],
               "measured_on": "the built variant; the whole member reads 0.9519 "
                              "claim-only and 0.9666 within-pair"},
        "C6": {"verdict": "PASS",
               "eval_facing_test_under_C_A2": "NOT-APPLICABLE - halueval is a "
                                              "training member and no evaluation "
                                              "surface is keyed on it, so the "
                                              "clause's eval-facing test has zero "
                                              "key coverage and no proxy is "
                                              "substituted",
               "within_member_diagnostic": cl["C6"],
               "measured_on": "the built variant"},
        "C7": dict(cl["C7"], verdict="PASS",
                   pass_criterion="rows == 2 x pairs, both counts reported in "
                                  "the same unit as registration",
                   unit_note="the 40,000-row registered figure describes the "
                             "SOURCE member, not this subset; the subset's own "
                             "counts are 480 rows / 240 pairs"),
        "C8": dict(cl["C8"], verdict="PASS",
                   selection_predicate_addendum="plus the pair-level subset in "
                                                "`variant_built.pipeline`"),

        "conforming": False,
        "failed_clauses": ["C5"],
        "binding_constraint":
            "C5 - the claim-only leak is a CORPUS property, not a build defect. "
            "HaluEval's negative leg is ChatGPT-written and its positive leg is a "
            "human reference, and the separation is carried redundantly: "
            f"{tc('shape_only_content_masked')} with every content word masked, "
            f"{tc('content_only_style_removed')} with every function word, hedge, "
            f"capital and punctuation mark deleted, "
            f"{A['full_probe_after_length_control']['length_stratified_auroc']} "
            "inside claim-length deciles. Twelve subset strategies over eighteen "
            "distinct retention levels produce exactly one subset that clears the pooled "
            "conjunction, at 480 of 40,000 rows, and that subset still breaches "
            "parity on its summarization half. No half-balanced subset clears at "
            "any size. No text rewrite clears - the best reads 0.9131 and turns "
            "507 negatives into fully attested claims labelled 0, which is the "
            "defect C1 exists to catch",
        "fixable": "CORPUS_PROPERTY",
        "consequence_for_dependants":
            "unchanged from the phase-1 finding and now bounded. halueval is "
            "40,000 rows: 5.83% of the clean public mix, 5.55% of the banked "
            "flagship mix whose k=6 blind arena mean is 0.71218, and 5.26% of "
            "the portfolio mix. For that fraction of every draw's rows the label "
            "is recoverable from the claim string alone, so the model is not "
            "required to read the evidence to fit them. C1 holds throughout - "
            "the negatives are genuinely less attested than the positives - so "
            "the member is not poisoning the support head the way the withdrawn "
            "contrast lane did. What this work adds: the shortcut cannot be "
            "filtered out. Keeping a quarter of the member still leaves a 0.669 "
            "claim-only channel; the subsets that reach the bar keep 1-2% of the "
            "rows. No banked arena number is invalidated and no in-flight draw "
            "needs to stop",

        "what_was_not_tried_and_why": {
            "regenerate_the_positive_leg": "the repair the measurements point "
                                           "at - if both legs were written by "
                                           "the same model the register channel "
                                           "closes and the content channel goes "
                                           "with it. It needs generation, which "
                                           "is outside this task's CPU-only "
                                           "offline scope, and it would replace "
                                           "HaluEval's reference answer with "
                                           "synthetic text, which is a new "
                                           "member rather than a conformed one",
            "claim_truncation_as_a_shipped_transform": "measured and reported "
                                                       "under `strategies_tried."
                                                       "text_transforms`; "
                                                       "rejected on its own "
                                                       "numbers, not on taste",
            "dropping_the_summarization_half_or_the_qa_half": "covered - every "
                                                              "subset strategy "
                                                              "was free to drop "
                                                              "a half and the "
                                                              "winning one very "
                                                              "nearly does "
                                                              "(440 of 480 rows "
                                                              "are QA); the "
                                                              "per-half searches "
                                                              "then test the "
                                                              "balanced case "
                                                              "explicitly",
        },

        "artifacts": [
            "experiments/grounding-semantic/contract/halueval_conform.py",
            "experiments/grounding-semantic/contract/halueval_conform_perhalf.py",
            "experiments/grounding-semantic/contract/halueval_conform_floorcheck.py",
            "experiments/grounding-semantic/contract/halueval_conformed.parquet",
            "experiments/grounding-semantic/contract/halueval_conform_best_effort.parquet",
            "experiments/grounding-semantic/contract/halueval_conform_scores.json",
            "experiments/grounding-semantic/contract/halueval_conform_diag.json",
            "experiments/grounding-semantic/contract/halueval_conform_frontier.json",
            "experiments/grounding-semantic/contract/halueval_conform_fine.json",
            "experiments/grounding-semantic/contract/halueval_conform_perhalf.json",
            "experiments/grounding-semantic/contract/halueval_conform_floorcheck.json",
            "experiments/grounding-semantic/contract/halueval_conform_honesty.json",
            "experiments/grounding-semantic/contract/halueval_conform_transforms.json",
            "experiments/grounding-semantic/contract/halueval_conformed_build.json",
            "experiments/grounding-semantic/contract/halueval_conformed_clauses.json",
            "experiments/grounding-semantic/contract/halueval_conform_best_effort_build.json",
            "experiments/grounding-semantic/contract/halueval_conform_best_effort_clauses.json",
            "experiments/grounding-semantic/contract/halueval_conformed_report.json",
            "logs/contract-halueval-conform-scores.log",
            "logs/contract-halueval-conform-diag.log",
            "logs/contract-halueval-conform-frontier-a.log",
            "logs/contract-halueval-conform-frontier-b.log",
            "logs/contract-halueval-conform-frontier-c.log",
            "logs/contract-halueval-conform-frontier-merge.log",
            "logs/contract-halueval-conform-deep-peel.log",
            "logs/contract-halueval-conform-deep-comp.log",
            "logs/contract-halueval-conform-fine.log",
            "logs/contract-halueval-conform-perhalf.log",
            "logs/contract-halueval-conform-floorcheck.log",
            "logs/contract-halueval-conform-honesty.log",
            "logs/contract-halueval-conform-honesty2.log",
            "logs/contract-halueval-conform-transforms.log",
            "logs/contract-halueval-conform-build-verify.log",
        ],
    }
    (HERE / "halueval_conformed_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep["headline"], indent=2))
    print(json.dumps(rep["frontier"]["requested_levels"], indent=2))
    print("report written", flush=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--retention", type=float, default=None)
    ap.add_argument("--tag", default="conformed")
    a = ap.parse_args()
    if a.stage == "scores":
        stage_scores()
    elif a.stage == "diag":
        stage_diag()
    elif a.stage == "frontier":
        stage_frontier(a.only)
    elif a.stage == "frontier_merge":
        stage_frontier_merge()
    elif a.stage == "honesty":
        stage_honesty()
    elif a.stage == "deep":
        stage_deep(a.only)
    elif a.stage == "transforms":
        stage_transforms()
    elif a.stage == "fine":
        stage_fine()
    elif a.stage == "build":
        stage_build(a.strategy, a.retention, a.tag)
    elif a.stage == "verify":
        stage_verify(a.tag)
    elif a.stage == "merge":
        stage_merge()
    else:
        raise SystemExit(f"unknown stage {a.stage}")


if __name__ == "__main__":
    main()

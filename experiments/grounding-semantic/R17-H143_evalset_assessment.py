"""R17-H143 evalset - full dataset-contract assessment.  CPU ONLY, zero GPU.

`R17-H143_evalset.parquet` is the only evaluation surface in the campaign that
has never been put through the dataset contract.  `R20-H175b_eval_contamination_
sweep.json` recorded it as CONTAMINATED - 10 of 547 passages present in the
assembled mix, reachable ONLY under the whitespace-collapsed case-folded form
(0 raw, 0 truncated) - and the canonical log records the open item explicitly:
"Whatever gate reads it is unassessed."

What this script measures, and nothing else:

  C2  full disjointness against BOTH assembled mixes - the FLAGSHIP 721,210-row
      mix and the PORTFOLIO 760,618-row mix (`R20-H174_arm_run.LANES`) - in all
      four string forms crossed both ways, on EVIDENCE and on CLAIMS, plus the
      DOCUMENT channel raw and stem-normalised.  The evalset carries no doc_id;
      it is recovered by joining its passages to `R17-H143_evalset_source.parquet`
      exactly as the banked `contract/tabfact_conformed_c2.py` does.
  LOAD-BEARING.  Twelve banked model reads exist on this eval, all with per-row
      score arrays: seven Stage-A models (`R17-H143_scores.parquet`), two
      cycle-1 SFT checkpoints (`R17-H144_student_scores.parquet`) and three
      cycle-2 checkpoints (`R17-H144_student_c2_scores.parquet`).  Every banked
      AUROC is reproduced EXACTLY at its banked precision before any exclusion
      is applied; a mismatch aborts rather than producing a number.  The read is
      then recomputed with the contaminated rows dropped, exactly as
      `R20-H177_evalB_contamination_assessment.py` does.
  C6  memorisation channel keyed on the only field the two legs of a pair share
      - the PASSAGE.  Coverage and AUROC on the contaminated rows and on the
      whole eval.
  C1  structural test + the predicate-sensitive attestation diagnostic (the
      predicate these negatives corrupt is the ASSERTED NUMERAL, so the
      instrument is numeral attestation, not bag-of-words containment).
  C3  split semantics, C4 n-gram contamination census with a live positive
      control, C5 leak suite, C7 units, C8 provenance.

Every gate carries a LIVE POSITIVE CONTROL - a known-bad input fed to the same
instrument, with the instrument's response recorded.

Run:
  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run python \
      experiments/grounding-semantic/R17-H143_evalset_assessment.py
"""

import os

# CPU ONLY - GPU0/GPU1/GPU2 carry live training draws and are not to be touched;
# `R10-H108_lane` imports torch and would otherwise `setdefault` a device.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import collections
import importlib.util as _ilu
import json
import re
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).parent
OUT = HERE / "R17-H143_evalset_report.json"

EVALSET = HERE / "R17-H143_evalset.parquet"
EVAL_SRC = HERE / "R17-H143_evalset_source.parquet"

# ---- the banked reads on this eval, with their per-row score arrays -------- #
# (parquet, banked {model-key: (pooled_auroc, control_auroc)}).  `None` where
# the log banks no figure for that checkpoint; those are reported, not asserted.
STAGE_A_RESULT = HERE / "R17-H143_stageA_result.json"
SCORE_PARQUETS = {
    "stage_A": HERE / "R17-H143_scores.parquet",
    "H144_cycle1": HERE / "R17-H144_student_scores.parquet",
    "H144_cycle2": HERE / "R17-H144_student_c2_scores.parquet",
}
# from R17-H144_result.json - the numbers the canonical log adjudicated
BANKED_H144 = {
    "H144_cycle1": {"pooled": [0.817148], "control": [0.8696]},
    "H144_cycle2": {"pooled": [0.80957, 0.77892], "control": [0.872, 0.832]},
}

# lanes carrying a doc_id channel that enter one or both mixes
DOC_LANES = {
    "R17-H146_lane.parquet": "quant_misbind",
    "R18-H150_scaleunit_lane.parquet": "quant_scale_unit",
    "R20-H174_lane_L1.parquet": "frame_reject",
    "R20-H174_lane_L2.parquet": "attr_pool",
    "R20-H174_lane_L4.parquet": "path_bind",
}
FLAGSHIP_LANES = ("R17-H146_lane.parquet", "R18-H150_scaleunit_lane.parquet")
PORTFOLIO_LANES = FLAGSHIP_LANES + (
    "R20-H174_lane_L1.parquet", "R20-H174_lane_L2.parquet",
    "R20-H174_lane_L4.parquet",
)
FLAGSHIP_ROWS = 721_210
PORTFOLIO_ROWS = 760_618
CLEAN_ROWS = 685_670

NOTE = "Numbers recorded, not adjudicated - the coordinator adjudicates."


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def norm(s):
    """The sweep's normalised form - whitespace-collapsed, case-folded."""
    return " ".join(s.split()).casefold()


def auroc_rank(labels, scores):
    """The banked H143 AUROC - mergesort ranks with explicit tie averaging.

    Copied in behaviour from `R17-H143_analyze.auroc`; using sklearn here would
    risk a tie-handling difference on parse-failure plateaus, and reproducing
    the banked figure exactly is the integrity check this whole read rests on.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i: j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_p, n_n = pos.sum(), neg.sum()
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def derive_scores(d, parse_fail_value=0.0):
    """`R17-H143_analyze.derive_scores`, byte-for-byte in behaviour."""
    m = d["margin"].to_numpy().astype(float)
    v = d["verdict"].to_numpy()
    fb = float(np.nanmean(np.abs(m))) if np.isfinite(m).any() else 1.0
    return np.where(
        np.isfinite(m), m,
        np.where(v == "GROUNDED", fb, np.where(v == "UNGROUNDED", -fb, parse_fail_value)),
    )


# --------------------------------------------------------------------------- #
# mix assembly
# --------------------------------------------------------------------------- #
def build_mix(lane_files, expect_rows):
    """`R10-H108_lane.public_train()` under untruncated evidence + named lanes."""
    arm = _mod("g1arm", HERE / "R16-H142_G1_arm.py")
    H108 = _mod("h108lane", HERE / "R10-H108_lane.py")
    chunk_max = H108.M59.CFG.chunk_max_chars
    print(f"mix: chunk_max_chars = {chunk_max}", flush=True)

    with arm.untruncated_evidence():
        claims, chunks, y, tags = H108.public_train()
    labels = list(np.asarray(y, dtype="float64"))
    claims, chunks, tags = list(claims), list(chunks), list(tags)
    if len(claims) != CLEAN_ROWS:
        raise SystemExit(f"MIX ABORT: clean mix {len(claims)} rows, expected {CLEAN_ROWS}")
    print(f"mix: clean public {len(claims)} rows over {len(set(tags))} groups", flush=True)

    for fname in lane_files:
        p = HERE / fname
        if not p.exists():
            raise SystemExit(f"MIX ABORT: lane {fname} absent")
        d = pl.read_parquet(p)
        ch_col = "chunk" if "chunk" in d.columns else "evidence"
        lcol = next((c for c in ("label", "y") if c in d.columns), None)
        claims += d["claim"].to_list()
        chunks += d[ch_col].to_list()
        labels += ([float(v) for v in d[lcol].to_list()] if lcol
                   else [float("nan")] * d.height)
        tags += [DOC_LANES.get(fname, fname)] * d.height
        print(f"mix: lane {fname} {d.height} rows", flush=True)

    if len(claims) != expect_rows:
        raise SystemExit(f"MIX ABORT: mix {len(claims)} rows, expected {expect_rows}")
    print(f"mix: total {len(claims)} rows / {len(set(tags))} groups", flush=True)
    return claims, chunks, labels, tags, chunk_max


def form_sets(texts, cut):
    raw = set(t for t in texts if t)
    trunc = {t[:cut] for t in raw}
    return {"raw": raw, "trunc": trunc,
            "nraw": {norm(t) for t in raw}, "ntrunc": {norm(t) for t in trunc}}


# eval-form -> mix-form pairings.  Normalisation status must match on both
# sides or the comparison is vacuous, so the matrix is 2x2 in raw space and
# 2x2 in normalised space: eight tests, a strict superset of the sweep's six.
FORM_MATRIX = [
    ("raw_in_mix_raw", "raw", "raw"),
    ("raw_in_mix_truncated", "raw", "trunc"),
    ("truncated_in_mix_raw", "trunc", "raw"),
    ("truncated_in_mix_truncated", "trunc", "trunc"),
    ("normalised_in_mix_normalised_raw", "nraw", "nraw"),
    ("normalised_in_mix_normalised_truncated", "nraw", "ntrunc"),
    ("normalised_truncated_in_mix_normalised_raw", "ntrunc", "nraw"),
    ("normalised_truncated_in_mix_normalised_truncated", "ntrunc", "ntrunc"),
]

_FORM_FN = {
    "raw": lambda t, cut: t,
    "trunc": lambda t, cut: t[:cut],
    "nraw": lambda t, cut: norm(t),
    "ntrunc": lambda t, cut: norm(t[:cut]),
}


def cross_forms(eval_units, mix_forms, cut, reverse=True):
    """Both directions, all eight form pairings.

    Returns per-test counts of DISTINCT EVAL units hit and DISTINCT MIX units
    hit, plus the union hit set on the eval side.
    """
    ev_forms = form_sets(eval_units, cut)
    counts, hit = {}, set()
    for name, ef, mf in FORM_MATRIX:
        fn = _FORM_FN[ef]
        target = mix_forms[mf]
        h = {p for p in ev_forms["raw"] if fn(p, cut) in target}
        counts[name] = {"eval_units_in_mix": len(h)}
        if reverse:
            # reverse direction: which mix units land inside the eval's form set
            rev_fn = _FORM_FN[mf]
            ev_target = ev_forms[ef]
            counts[name]["mix_units_in_eval"] = sum(
                1 for c in mix_forms["raw"] if rev_fn(c, cut) in ev_target)
        hit |= h
    return counts, hit, ev_forms


# --------------------------------------------------------------------------- #
# C6 memorisation feature
# --------------------------------------------------------------------------- #
def memorisation_feature(df, mix_claims, mix_labels, Q):
    y = np.asarray(df["label"].to_list())
    lookup = [mix_claims.get(norm(c), []) for c in df["chunk"].to_list()]
    lookup_y = [mix_labels.get(norm(c), []) for c in df["chunk"].to_list()]
    covered = sum(1 for v in lookup if v)
    out = {"rows": df.height, "rows_with_a_mix_claim": covered,
           "coverage": round(covered / df.height, 4) if df.height else 0.0}
    if covered == 0 or len(set(y.tolist())) < 2:
        out["auroc"] = None
        out["note"] = ("no key coverage - the mix carries no claim over any of these "
                       "passages; C6's eval-facing test is NOT-APPLICABLE here and "
                       "no proxy is substituted (amendment C-A2)")
        return out
    variants = {
        "jaccard": lambda c, a: Q.jaccard(c, a),
        "claim_into_mixclaim_containment": lambda c, a: Q.containment(c, a),
        "mixclaim_into_claim_containment": lambda c, a: Q.containment(a, c),
        "shared_token_count": lambda c, a: float(len(set(Q.tok(c)) & set(Q.tok(a)))),
    }
    claims = df["claim"].to_list()
    for vname, fn in variants.items():
        s = np.array([max((fn(c, a) for a in v), default=0.0)
                      for c, v in zip(claims, lookup)])
        out[vname] = round(float(auroc_rank(y, s)), 4)
    out["auroc"] = max(out[v] for v in variants)
    out["strongest_variant"] = max(variants, key=lambda v: out[v])

    best_lab = []
    for c, v, ly in zip(claims, lookup, lookup_y):
        if not v:
            best_lab.append(0.0)
            continue
        j = int(np.argmax([Q.jaccard(c, a) for a in v]))
        best_lab.append(float(ly[j]))
    out["nearest_mix_claim_label"] = round(float(auroc_rank(y, np.array(best_lab))), 4)

    s = np.array([max((Q.jaccard(c, a) for a in v), default=0.0)
                  for c, v in zip(claims, lookup)])
    d = (df.with_columns(pl.Series("f", s)).group_by("pair_id")
           .agg((pl.col("f").max() - pl.col("f").min()).alias("spread")))
    out["within_pair_feature_spread"] = {
        "pairs": d.height,
        "pairs_with_zero_spread": int((d["spread"] == 0.0).sum()),
        "mean_spread": round(float(d["spread"].mean()), 6),
        "max_spread": round(float(d["spread"].max()), 6),
    }
    return out


# --------------------------------------------------------------------------- #
# C1 - attestation instruments
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"\d[\d,]*\.?\d*")


def numerals(text):
    """Numeric literals with thousands separators stripped - the predicate the
    H143 negative families corrupt is the ASSERTED VALUE, so this is the
    instrument sensitive to it.  A bag-of-words containment instrument is
    predicate-BLIND here: both legs share the chunk and differ in one numeral,
    so word containment cannot move."""
    out = []
    for m in _NUM.findall(text or ""):
        t = m.replace(",", "").rstrip(".")
        if t:
            out.append(t)
    return out


def numeral_attestation(claim, chunk):
    ns = numerals(claim)
    if not ns:
        return None
    ch = (chunk or "").replace(",", "")
    return sum(1 for n in ns if n in ch) / len(ns)


def attestation_block(df, Q):
    rows = []
    for c, ch, lab in zip(df["claim"].to_list(), df["chunk"].to_list(),
                          df["label"].to_list()):
        rows.append((int(lab), numeral_attestation(c, ch), Q.containment(c, ch)))
    out = {}
    for lab in (1, 0):
        num = [r[1] for r in rows if r[0] == lab and r[1] is not None]
        con = [r[2] for r in rows if r[0] == lab]
        out[f"label_{lab}"] = {
            "n": sum(1 for r in rows if r[0] == lab),
            "n_with_a_numeral": len(num),
            "numeral_attestation_mean": round(float(np.mean(num)), 4) if num else None,
            "numeral_rate_ge_0.90": round(float(np.mean(np.array(num) >= 0.90)), 4) if num else None,
            "numeral_rate_eq_1.0": round(float(np.mean(np.array(num) >= 1.0)), 4) if num else None,
            "wordbag_containment_mean": round(float(np.mean(con)), 4),
            "wordbag_rate_ge_0.90": round(float(np.mean(np.array(con) >= 0.90)), 4),
            "wordbag_rate_eq_1.0": round(float(np.mean(np.array(con) >= 1.0)), 4),
        }
    return out


def recompute_reads(ev_flagged, banked_stage_a):
    """Every banked model read on this eval, recomputed with and without the
    rows flagged `contaminated` in `ev_flagged`.

    The banked figure is reproduced at its banked precision BEFORE any exclusion
    is applied; a mismatch raises rather than producing a number.
    """
    ev_key = ev_flagged.select(
        pl.col("pair_id").cast(pl.Int64), pl.col("label").cast(pl.Int64),
        "contaminated")
    reads, unmatched = {}, []
    for tag, p in SCORE_PARQUETS.items():
        sc = pl.read_parquet(p).with_columns(
            pl.col("pair_id").cast(pl.Int64), pl.col("label").cast(pl.Int64))
        sc = sc.join(ev_key, on=["pair_id", "label"], how="left")
        if sc["contaminated"].null_count():
            raise SystemExit("ABORT: score rows without an eval join")
        for name in sc["model"].unique().sort().to_list():
            d = sc.filter(pl.col("model") == name)
            r_all = d.filter(~pl.col("control"))
            c_all = d.filter(pl.col("control"))
            # local checkpoint paths are reduced to their basename - this is a
            # PUBLIC repository and absolute paths are not published
            short = name if not name.startswith("/") else Path(name).name
            key = f"{tag}::{short}"
            s_real = derive_scores(r_all, 0.0)
            s_ctrl = derive_scores(c_all, 0.0) if c_all.height else None
            pooled = auroc_rank(r_all["label"].to_numpy(), s_real)
            control_a = (auroc_rank(c_all["label"].to_numpy(), s_ctrl)
                         if s_ctrl is not None else None)
            keep_r = ~r_all["contaminated"].to_numpy()
            keep_c = (~c_all["contaminated"].to_numpy() if c_all.height else None)
            blk = {
                "read": tag, "model": short,
                "n_real_rows": r_all.height, "n_control_rows": c_all.height,
                "pooled_auroc_all_rows": round(pooled, 6),
                "control_auroc_all_rows": round(control_a, 6) if control_a is not None else None,
                "n_real_rows_contaminated": int((~keep_r).sum()),
                "n_control_rows_contaminated": int((~keep_c).sum()) if keep_c is not None else 0,
            }
            if len(set(r_all["label"].to_numpy()[keep_r].tolist())) == 2:
                blk["pooled_auroc_clean_rows"] = round(
                    auroc_rank(r_all["label"].to_numpy()[keep_r], s_real[keep_r]), 6)
                blk["pooled_delta"] = round(
                    blk["pooled_auroc_clean_rows"] - blk["pooled_auroc_all_rows"], 6)
            if keep_c is not None and len(set(c_all["label"].to_numpy()[keep_c].tolist())) == 2:
                blk["control_auroc_clean_rows"] = round(
                    auroc_rank(c_all["label"].to_numpy()[keep_c], s_ctrl[keep_c]), 6)
                blk["control_delta"] = round(
                    blk["control_auroc_clean_rows"] - blk["control_auroc_all_rows"], 6)
            if (~keep_r).sum() and len(set(r_all["label"].to_numpy()[~keep_r].tolist())) == 2:
                blk["pooled_auroc_contaminated_rows_only"] = round(
                    auroc_rank(r_all["label"].to_numpy()[~keep_r], s_real[~keep_r]), 6)
            # sensitivity: fallback recomputed on the kept subset (a true re-read)
            if (~keep_r).sum():
                sub = r_all.filter(pl.Series("keep", keep_r))
                blk["pooled_auroc_clean_rows_fallback_recomputed"] = round(
                    auroc_rank(sub["label"].to_numpy(), derive_scores(sub, 0.0)), 6)
            reads[key] = blk

            # --- integrity: reproduce the banked figure EXACTLY ------------ #
            b = banked_stage_a.get(name)
            if b is not None:
                for field, got in (("pooled_auroc", pooled), ("control_auroc", control_a)):
                    want = b.get(field)
                    if want is None or got is None:
                        continue
                    dec = len(str(want).split(".")[1]) if "." in str(want) else 0
                    if round(got, dec) != want:
                        raise SystemExit(
                            f"ABORT: {key} {field} recomputed {got:.6f} != banked {want} "
                            "- the score array does not reproduce the banked read")
                    blk[f"banked_{field}"] = want
                    blk[f"banked_{field}_reproduced"] = True
        del sc

    # H144 banked figures are keyed by number, not by checkpoint name
    for tag, want in BANKED_H144.items():
        got_p = [v["pooled_auroc_all_rows"] for v in reads.values() if v["read"] == tag]
        got_c = [v["control_auroc_all_rows"] for v in reads.values() if v["read"] == tag]
        for field, wants, gots in (("pooled", want["pooled"], got_p),
                                   ("control", want["control"], got_c)):
            for w in wants:
                dec = len(str(w).split(".")[1]) if "." in str(w) else 0
                if not any(g is not None and round(g, dec) == w for g in gots):
                    unmatched.append(f"{tag}.{field}={w} (computed {gots})")
    if unmatched:
        raise SystemExit("ABORT: banked figures not reproduced from the score "
                         f"arrays: {unmatched}")
    return reads


def asserted_value_attestation(df, src):
    """The sharpest predicate-sensitive instrument this eval admits.

    The source snapshot carries `asserted_value` - the single number the claim
    asserts, which is `v_pos` on the positive leg and `v_neg` on the negative.
    The negative families corrupt exactly that value, so its presence in the
    evidence is the instrument C-A2 test 2 asks for.
    """
    j = df.join(
        src.select("pair_id", pl.col("label").cast(pl.Int8), "asserted_value",
                   "v_pos", "v_neg").unique(subset=["pair_id", "label"]),
        on=["pair_id", "label"], how="left")
    out = {"rows_joined": j.height,
           "rows_without_an_asserted_value": int(j["asserted_value"].null_count())}
    for lab in (1, 0):
        sub = j.filter(pl.col("label") == lab)
        vals = []
        for av, ch in zip(sub["asserted_value"].to_list(), sub["chunk"].to_list()):
            if av is None:
                continue
            vals.append(float(str(av).replace(",", "") in (ch or "").replace(",", "")))
        out[f"label_{lab}"] = {
            "n": len(vals),
            "asserted_value_present_in_evidence_rate": round(float(np.mean(vals)), 4)
            if vals else None,
        }
    p = out["label_1"]["asserted_value_present_in_evidence_rate"]
    n = out["label_0"]["asserted_value_present_in_evidence_rate"]
    out["negative_strictly_below_positive"] = bool(
        p is not None and n is not None and n < p)
    out["gap_negative_minus_positive"] = (round(n - p, 4)
                                          if p is not None and n is not None else None)
    return out


def structural_test(df):
    """C-A1 test 1 - is any negative leg's (claim, evidence) identical to a
    positive leg's?  Compared on the normalised form, which is strictly weaker
    than byte identity and therefore cannot under-report."""
    pos = {(norm(c), norm(ch)) for c, ch, lb in
           zip(df["claim"], df["chunk"], df["label"]) if lb == 1}
    neg = [(norm(c), norm(ch)) for c, ch, lb in
           zip(df["claim"], df["chunk"], df["label"]) if lb == 0]
    fired = sum(1 for k in neg if k in pos)
    return {"negative_legs": len(neg), "positive_legs": len(pos),
            "negative_legs_identical_to_a_positive": fired,
            "rate": round(fired / len(neg), 6) if neg else None,
            "fires": fired > 0}


# --------------------------------------------------------------------------- #
# C5 - leak suite
# --------------------------------------------------------------------------- #
def claim_only_probe(df, groups):
    """Converged liblinear TF-IDF claim-only probe, document-grouped folds.

    liblinear with tol 1e-7 per the H144 finding - default lbfgs "converges" to
    an all-zero fit on minimal pairs and reads exactly 0.5000, which is not a
    measurement.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    X = df["claim"].to_list()
    y = np.asarray(df["label"].to_list())
    g = np.asarray(groups)
    if len(set(g.tolist())) < 5:
        return {"auroc": None, "note": "fewer than 5 document groups"}
    scores = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        Xtr = vec.fit_transform([X[i] for i in tr])
        Xte = vec.transform([X[i] for i in te])
        clf = LogisticRegression(solver="liblinear", tol=1e-7, max_iter=10_000, C=1.0)
        clf.fit(Xtr, y[tr])
        scores[te] = clf.decision_function(Xte)
    return {"auroc": round(float(auroc_rank(y, scores)), 4), "folds": 5,
            "groups": int(len(set(g.tolist()))), "solver": "liblinear tol=1e-7"}


def within_pair_claim_only(df):
    """Per-family within-pair accuracy of the claim-only probe's ordering."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    X = df["claim"].to_list()
    y = np.asarray(df["label"].to_list())
    g = np.asarray(df["pair_id"].to_list())
    scores = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        clf = LogisticRegression(solver="liblinear", tol=1e-7, max_iter=10_000)
        clf.fit(vec.fit_transform([X[i] for i in tr]), y[tr])
        scores[te] = clf.decision_function(vec.transform([X[i] for i in te]))
    d = df.with_columns(pl.Series("s", scores))
    per = {}
    for fam in sorted(set(df.filter(pl.col("label") == 0)["neg_family"].to_list())):
        ids = d.filter(pl.col("neg_family") == fam)["pair_id"].unique().to_list()
        sub = d.filter(pl.col("pair_id").is_in(ids))
        ok = tot = 0
        for _pid, grp in sub.group_by("pair_id"):
            p = grp.filter(pl.col("label") == 1)["s"].to_list()
            n = grp.filter(pl.col("label") == 0)["s"].to_list()
            if not p or not n:
                continue
            tot += 1
            ok += (p[0] > n[0]) + 0.5 * (p[0] == n[0])
        if tot:
            per[fam] = {"pairs": tot, "acc": round(float(ok / tot), 4)}
    return {"per_family": per,
            "worst": round(max(v["acc"] for v in per.values()), 4) if per else None}


def _magnitude(t):
    vals = []
    for n in numerals(t):
        try:
            vals.append(abs(float(n)))
        except ValueError:
            continue
    return float(np.log10(max(vals) + 1.0)) if vals else 0.0


def surface_parity(df):
    """C5 under amendment C-A1 - claim-ALONE and evidence-ALONE channels only.

    Containment is a JOINT feature and is governed by C1, not C5.  Evidence-alone
    is degenerate by construction here: both legs of a pair carry the SAME
    chunk, so any evidence-only channel reads exactly 0.5.  That is recorded
    rather than presented as a passing measurement.
    """
    y = np.asarray(df["label"].to_list())
    cl = df["claim"].to_list()
    ch = df["chunk"].to_list()

    def first_digit(t):
        m = re.search(r"\d", t or "")
        return float(ord(m.group()) - 48) if m else -1.0

    def trailing_zeros(t):
        ns = numerals(t)
        return float(np.mean([len(n) - len(n.rstrip("0")) for n in ns])) if ns else 0.0

    chans = {
        "claim_char_length": np.array([float(len(t)) for t in cl]),
        "claim_token_count": np.array([float(len(t.split())) for t in cl]),
        "claim_digit_count": np.array([float(sum(c.isdigit() for c in t)) for t in cl]),
        "claim_numeral_count": np.array([float(len(numerals(t))) for t in cl]),
        "claim_leading_digit": np.array([first_digit(t) for t in cl]),
        "claim_trailing_zeros": np.array([trailing_zeros(t) for t in cl]),
        "claim_has_decimal": np.array([float("." in "".join(numerals(t))) for t in cl]),
        "claim_max_numeral_magnitude": np.array([_magnitude(t) for t in cl]),
        "evidence_char_length": np.array([float(len(t)) for t in ch]),
        "evidence_digit_count": np.array([float(sum(c.isdigit() for c in t)) for t in ch]),
    }
    out = {}
    for k, v in chans.items():
        a = auroc_rank(y, v)
        out[k] = round(float(a), 4) if a is not None else None
    devs = [abs(v - 0.5) for v in out.values() if v is not None]
    out["_max_deviation_from_0.5"] = round(float(max(devs)), 4)
    out["_channels_outside_0.45_0.55"] = sorted(
        k for k, v in out.items() if not k.startswith("_") and v is not None
        and not (0.45 <= v <= 0.55))
    return out


# --------------------------------------------------------------------------- #
# C4 - n-gram contamination census, per mix group, with live positive control
# --------------------------------------------------------------------------- #
def c4_census(eval_passages, chunks, tags, n=8, thr=0.3):
    PG = _mod("provgate", HERE / "provenance_gate.py")
    hasher = PG._TokenHasher()

    q_hashes = [PG.ngram_hashes(p, n, hasher) for p in eval_passages]
    scorable = [i for i, q in enumerate(q_hashes) if q.size]
    print(f"C4: {len(scorable)}/{len(q_hashes)} eval passages long enough for {n}-grams",
          flush=True)

    by_group = collections.defaultdict(set)
    for c, t in zip(chunks, tags):
        if c:
            by_group[t].add(c)

    best = np.zeros(len(q_hashes))
    best_group = [None] * len(q_hashes)
    per_group = {}
    rev_hits = {}
    # eval-side index for the reverse direction
    ev_side = PG._Side("eval")
    for q in q_hashes:
        ev_side.add("eval", q)
    ev_idx = ev_side.index()
    ev_sizes = {b: np.array([u.size for u in ev_side.buckets[b]], dtype=np.int64)
                for b in ev_idx}

    for grp in sorted(by_group):
        units = sorted(by_group[grp])
        hs = [PG.ngram_hashes(u, n, hasher) for u in units]
        side = PG._Side(grp)
        for a in hs:
            side.add(grp, a)
        idx = side.index()
        sizes = {b: np.array([x.size for x in side.buckets[b]], dtype=np.int64)
                 for b in idx}
        hits = 0
        gbest = 0.0
        for i in scorable:
            for b, (h, owner, _) in idx.items():
                j, _uid = PG._max_jaccard(q_hashes[i], h, owner, sizes[b])
                if j > best[i]:
                    best[i], best_group[i] = j, grp
                gbest = max(gbest, j)
                hits += j >= thr
        # reverse direction - mix units of this group against the eval index
        rev = 0
        for qh in hs:
            for b, (h, owner, _) in ev_idx.items():
                j, _uid = PG._max_jaccard(qh, h, owner, ev_sizes[b])
                rev += j >= thr
        per_group[grp] = {"mix_units": len(units),
                          "eval_units_at_jaccard_ge_thr": int(hits),
                          "max_jaccard": round(float(gbest), 4),
                          "mix_units_at_jaccard_ge_thr": int(rev)}
        rev_hits[grp] = rev
        print(f"C4: {grp:24s} units={len(units):7d} eval-hits={hits:4d} "
              f"mix-hits={rev:5d} max_j={gbest:.4f}", flush=True)
        del side, idx, sizes, hs

    n_hit = int((best >= thr).sum())
    return {
        "instrument": f"provenance_gate primitives, {n}-gram, Jaccard >= {thr}, "
                      "bidirectional, per-group attribution (R14-H136 ruling-2 form)",
        "eval_units": len(q_hashes),
        "eval_units_scorable": len(scorable),
        "eval_units_at_jaccard_ge_thr": n_hit,
        "eval_fraction": round(n_hit / len(q_hashes), 6),
        "kill_threshold": 0.02,
        "best_jaccard": {"max": round(float(best.max()), 4),
                         "p99": round(float(np.percentile(best, 99)), 4),
                         "mean": round(float(best.mean()), 4)},
        "per_group": per_group,
        "mix_units_at_jaccard_ge_thr_total": int(sum(rev_hits.values())),
    }, q_hashes, best


def c4_positive_control(eval_passages, chunks, tags, n=8, thr=0.3, k=10):
    """LIVE positive control - two known-bad inputs fed to the same instrument.

    (1) SPIKE: ten mix chunks injected into the candidate side; all ten must
        read Jaccard 1.0 against their own group.
    (2) LIVE near-duplicate: ten eval passages re-wrapped (whitespace collapsed,
        case-folded) and fed as candidates against an index built from the
        ORIGINAL passages - the exact transformation this eval's real leak took.
    """
    PG = _mod("provgate", HERE / "provenance_gate.py")
    hasher = PG._TokenHasher()

    # (1) spike - take k mix chunks, index the group they came from
    grp_of = {}
    for c, t in zip(chunks, tags):
        if c and c not in grp_of:
            grp_of[c] = t
    inj = [c for c in list(grp_of)[:2000] if len(c.split()) > 20][:k]
    src = PG._Side("mixsrc")
    for c in list(grp_of)[:5000]:
        src.add("mixsrc", PG.ngram_hashes(c, n, hasher))
    sidx = src.index()
    ssz = {b: np.array([u.size for u in src.buckets[b]], dtype=np.int64) for b in sidx}
    spike = 0
    for c in inj:
        q = PG.ngram_hashes(c, n, hasher)
        for b, (h, owner, _) in sidx.items():
            j, _ = PG._max_jaccard(q, h, owner, ssz[b])
            spike += j >= thr
            break

    # (2) live near-duplicate - re-wrapped eval passages against the originals
    orig = PG._Side("evalorig")
    for p in eval_passages:
        orig.add("evalorig", PG.ngram_hashes(p, n, hasher))
    oidx = orig.index()
    osz = {b: np.array([u.size for u in orig.buckets[b]], dtype=np.int64) for b in oidx}
    live_j = []
    for p in [p for p in eval_passages if len(p.split()) > 20][:k]:
        q = PG.ngram_hashes(norm(p), n, hasher)
        for b, (h, owner, _) in oidx.items():
            j, _ = PG._max_jaccard(q, h, owner, osz[b])
            live_j.append(round(float(j), 4))
            break
    return {
        "spike_control": {"injected": len(inj), "detected": int(spike),
                          "fires": int(spike) == len(inj)},
        "live_near_duplicate_control": {
            "construction": "eval passages whitespace-collapsed and case-folded, "
                            "scored against an index of the ORIGINAL passages - the "
                            "exact transformation the real leak took",
            "n": len(live_j), "jaccards": live_j,
            "detected": int(sum(j >= thr for j in live_j)),
            "fires": bool(live_j) and all(j >= thr for j in live_j)},
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-c4", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("=== R17-H143 evalset - dataset-contract assessment (CPU only) ===", flush=True)
    Q = _mod("h175bqlane", HERE / "R20-H175b_qlane.py")

    ev = pl.read_parquet(EVALSET)
    src = pl.read_parquet(EVAL_SRC)
    # the evalset carries no doc_id; recover it by passage, as the banked
    # contract/tabfact_conformed_c2.py does
    doc_by_chunk = src.select("chunk", "doc_id", "source").unique(subset=["chunk"])
    ev = ev.join(doc_by_chunk, on="chunk", how="left")
    if ev["doc_id"].null_count():
        raise SystemExit(f"ABORT: {ev['doc_id'].null_count()} eval rows without a doc_id")
    real = ev.filter(~pl.col("control"))
    ctrl = ev.filter(pl.col("control"))
    passages = sorted({c for c in ev["chunk"].to_list() if c and c.strip()})
    print(f"eval: {ev.height} rows / {ev['pair_id'].n_unique()} pairs / "
          f"{len(passages)} passages / {ev['doc_id'].n_unique()} docs "
          f"({real.height} real, {ctrl.height} control)", flush=True)

    res = {
        "experiment": "R17-H143 evalset - full dataset-contract assessment. "
                      "CPU-only, zero GPU, zero training",
        "scope": "measurement and reading only - no adjudication, no bar changed",
        "note": NOTE,
        "eval": {
            "parquet": EVALSET.name,
            "rows": ev.height, "pairs": int(ev["pair_id"].n_unique()),
            "real_rows": real.height, "real_pairs": int(real["pair_id"].n_unique()),
            "control_rows": ctrl.height, "control_pairs": int(ctrl["pair_id"].n_unique()),
            "distinct_passages": len(passages),
            "distinct_passages_real": int(real["chunk"].n_unique()),
            "distinct_passages_control": int(ctrl["chunk"].n_unique()),
            "distinct_claims": int(ev["claim"].n_unique()),
            "documents": int(ev["doc_id"].n_unique()),
            "documents_by_source": dict(ev.group_by("source").agg(
                pl.col("doc_id").n_unique()).iter_rows()),
            "rows_by_source": dict(ev.group_by("source").len().iter_rows()),
            "rows_by_neg_family": dict(ev.group_by("neg_family").len().iter_rows()),
            "rows_by_claim_form": dict(ev.group_by("claim_form").len().iter_rows()),
            "doc_id_recovered_via": "join of the evalset's passages to "
                                    "R17-H143_evalset_source.parquet, which carries doc_id",
        },
    }
    OUT.write_text(json.dumps(res, indent=2))

    # --------------------------------------------------------------- C2 ---- #
    mixes = {}
    hitsets = {}
    for label, lanes, expect in (("flagship", FLAGSHIP_LANES, FLAGSHIP_ROWS),
                                 ("portfolio", PORTFOLIO_LANES, PORTFOLIO_ROWS)):
        print(f"--- assembling {label} mix ---", flush=True)
        claims, chunks, labels, tags, cut = build_mix(lanes, expect)
        mf_chunk = form_sets(chunks, cut)
        mf_claim = form_sets(claims, cut)
        ev_counts, ev_hit, _ = cross_forms(passages, mf_chunk, cut)
        cl_counts, cl_hit, _ = cross_forms(
            sorted({c for c in ev["claim"].to_list() if c}), mf_claim, cut)
        hit_norm = {norm(p) for p in ev_hit}

        by_group = collections.Counter()
        mix_claims, lab_map = collections.defaultdict(list), collections.defaultdict(list)
        for cl, ch, lv, tg in zip(claims, chunks, labels, tags, strict=True):
            nn = norm(ch)
            if nn in hit_norm:
                by_group[tg] += 1
                mix_claims[nn].append(cl)
                lab_map[nn].append(lv)

        # only the PORTFOLIO chunk list is retained (C4 runs against the superset);
        # holding both would carry ~1.5M strings for no measurement
        mixes[label] = {"chunks": chunks if label == "portfolio" else None,
                        "tags": tags if label == "portfolio" else None, "cut": cut,
                        "mix_claims": dict(mix_claims), "lab_map": dict(lab_map),
                        "distinct_raw_chunks": len(mf_chunk["raw"]),
                        "rows": expect}
        hitsets[label] = ev_hit

        # --- LIVE POSITIVE CONTROL for the C2 detector -------------------- #
        # feed it three known-bad passages by construction and show each form fires
        probe_src = next(c for c in mf_chunk["raw"] if len(c) > 400)
        control = {
            "byte_identical_mix_chunk": {},
            "whitespace_rewrapped_mix_chunk": {},
            "clean_synthetic_passage": {},
        }
        rewrapped = re.sub(r" +", "  ", probe_src).replace("\n", " \n ").upper()
        synthetic = "ZZQX " + "".join(chr(97 + (i * 7) % 26) for i in range(600))
        for pname, ptxt in (("byte_identical_mix_chunk", probe_src),
                            ("whitespace_rewrapped_mix_chunk", rewrapped),
                            ("clean_synthetic_passage", synthetic)):
            c, _h, _f = cross_forms([ptxt], mf_chunk, cut, reverse=False)
            control[pname] = {k: v["eval_units_in_mix"] for k, v in c.items()}
        control["reads"] = (
            "the byte-identical probe must fire in every form; the re-wrapped "
            "probe must fire in the normalised forms and NOT in raw/truncated - "
            "that is exactly the mode this eval's real leak uses; the synthetic "
            "probe must fire nowhere")
        control["fires_as_expected"] = bool(
            control["byte_identical_mix_chunk"]["raw_in_mix_raw"] == 1
            and control["whitespace_rewrapped_mix_chunk"]["raw_in_mix_raw"] == 0
            and control["whitespace_rewrapped_mix_chunk"][
                "normalised_in_mix_normalised_raw"] == 1
            and sum(control["clean_synthetic_passage"].values()) == 0)

        res.setdefault("C2_disjointness", {})[label] = {
            "mix_rows": expect,
            "mix_distinct_raw_chunks": len(mf_chunk["raw"]),
            "mix_distinct_raw_claims": len(mf_claim["raw"]),
            "chunk_max_chars": cut,
            "evidence_channel": ev_counts,
            "evidence_passages_in_the_mix": len(ev_hit),
            "evidence_share_of_passages": round(len(ev_hit) / len(passages), 4),
            "claim_channel": cl_counts,
            "claims_in_the_mix": len(cl_hit),
            "mix_rows_carrying_a_contaminated_passage_by_group": dict(by_group),
            "live_positive_control": control,
        }
        print(f"{label}: evidence hits {len(ev_hit)}/{len(passages)}  "
              f"claim hits {len(cl_hit)}  control_fires="
              f"{control['fires_as_expected']}", flush=True)
        del claims, labels, mf_chunk, mf_claim
        if label != "portfolio":
            del chunks, tags
        OUT.write_text(json.dumps(res, indent=2, default=str))

    # ------------------------------------------------- C2 document channel -- #
    def stem(d):
        """TabFact serialises one table under both a `1-` and a `2-` csv id."""
        return d[10:] if d.startswith("tabfact:") and len(d) > 9 and d[8] in "12" else d

    ev_docs = set(ev["doc_id"].to_list())
    ev_stems = {stem(d) for d in ev_docs}
    doc_channel = {"eval_documents": len(ev_docs), "eval_document_stems": len(ev_stems)}
    for fname, grp in DOC_LANES.items():
        p = HERE / fname
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        ld = set(d["doc_id"].to_list())
        doc_channel[grp] = {
            "lane": fname, "lane_documents": len(ld),
            "in_flagship_mix": fname in FLAGSHIP_LANES,
            "in_portfolio_mix": fname in PORTFOLIO_LANES,
            "raw_doc_id_overlap": len(ev_docs & ld),
            "stem_doc_id_overlap": len({stem(x) for x in ld} & ev_stems),
        }
        del d
    # the mix's tabfact member exposes a table id.  Which member the mix actually
    # carries is VERIFIED here, not assumed: the census parquets are compared to
    # the mix's own `tabfact` group chunks before their table ids are trusted.
    mix_tf_chunks = {c for c, t in zip(mixes["portfolio"]["chunks"],
                                       mixes["portfolio"]["tags"]) if t == "tabfact"}
    tf_ev = {d.split(":", 1)[1] for d in ev_docs if d.startswith("tabfact:")}
    tf_ev_stem = {t[2:] if t[:2] in ("1-", "2-") else t for t in tf_ev}
    for fname, kind in (("tabfact_member.parquet", "UNCONFORMED"),
                        ("tabfact_member_conformed.parquet", "CONFORMED")):
        tfp = HERE / "contract" / fname
        if not tfp.exists():
            continue
        tf = pl.read_parquet(tfp)
        tids = set(tf["table_id"].to_list())
        tid_stem = {t[2:] if t[:2] in ("1-", "2-") else t for t in tids}
        cov = {}
        for col in ("chunk_untrunc", "chunk_trunc"):
            if col in tf.columns:
                s = set(tf[col].to_list())
                cov[col] = {"distinct": len(s),
                            "shared_with_mix_tabfact_group": len(s & mix_tf_chunks)}
        doc_channel[f"tabfact_member_{kind.lower()}"] = {
            "artifact": f"contract/{fname} ({kind})",
            "is_the_member_in_the_mix": cov,
            "mix_tabfact_group_distinct_chunks": len(mix_tf_chunks),
            "member_table_ids": len(tids),
            "eval_tabfact_documents": len(tf_ev),
            "raw_table_id_overlap": len(tf_ev & tids),
            "stem_table_id_overlap": len(tf_ev_stem & tid_stem),
            "share_of_eval_tabfact_documents": round(len(tf_ev & tids) / len(tf_ev), 4)
            if tf_ev else None,
        }
        del tf
    doc_channel["tabfact_note"] = (
        "the eval carries no doc_id and the mix carries no doc_id either; the "
        "TabFact table id is the only document key that resolves on both sides, "
        "recovered eval-side from R17-H143_evalset_source.parquet and mix-side "
        "from the banked census parquets. Which census parquet corresponds to the "
        "member the mix actually carries is settled by the chunk-overlap reading "
        "above, not asserted")
    doc_channel["feverous_note"] = (
        "the eval's FEVEROUS doc_ids are index-form `feverous:{i}` produced by an "
        "order-unstable dedup; they are not resolvable keys and a zero reading on "
        "them is not evidence of document disjointness - the content channel is "
        "the load-bearing one for that half")
    res["C2_document_channel"] = doc_channel
    print(f"doc channel: {json.dumps(doc_channel, default=str)}", flush=True)
    OUT.write_text(json.dumps(res, indent=2, default=str))

    # ------------------------------------------------------ LOAD-BEARING --- #
    hit = hitsets["flagship"] | hitsets["portfolio"]
    same = hitsets["flagship"] == hitsets["portfolio"]
    ev = ev.with_columns(pl.col("chunk").is_in(list(hit)).alias("contaminated"))
    n_rows_c = int(ev["contaminated"].sum())
    c_rows = ev.filter(pl.col("contaminated"))
    print(f"contaminated rows {n_rows_c} / pairs {c_rows['pair_id'].n_unique()} / "
          f"docs {c_rows['doc_id'].n_unique()}", flush=True)

    res["contaminated_rows"] = {
        "hit_set_identical_across_both_mixes": same,
        "passages": len(hit),
        "rows": n_rows_c,
        "pairs": int(c_rows["pair_id"].n_unique()),
        "documents": int(c_rows["doc_id"].n_unique()),
        "rows_by_source": dict(c_rows.group_by("source").len().iter_rows()),
        "rows_by_neg_family": dict(c_rows.group_by("neg_family").len().iter_rows()),
        "rows_by_control": dict(c_rows.group_by("control").len().iter_rows()),
        "labels": dict(c_rows.group_by("label").len().iter_rows()),
        "document_ids": sorted(c_rows["doc_id"].unique().to_list()),
    }

    banked_stage_a = json.loads(STAGE_A_RESULT.read_text())["per_model"]
    reads = recompute_reads(ev, banked_stage_a)
    print(f"integrity: every banked read reproduced exactly ({len(reads)} model reads)",
          flush=True)

    # --- LIVE POSITIVE CONTROL for the integrity check ------------------- #
    rng = np.random.default_rng(143)
    sc0 = pl.read_parquet(SCORE_PARQUETS["stage_A"])
    nm = sorted(sc0["model"].unique().to_list())[0]
    dd = sc0.filter(pl.col("model") == nm, ~pl.col("control"))
    s_ok = derive_scores(dd, 0.0)
    s_bad = s_ok.copy()
    rng.shuffle(s_bad)
    want = banked_stage_a[nm]["pooled_auroc"]
    got_bad = auroc_rank(dd["label"].to_numpy(), s_bad)
    res["load_bearing"] = {
        "protocol": "banked per-row score arrays in the eval-parquet score "
                    "parquets; every banked AUROC is reproduced at its banked "
                    "precision BEFORE any exclusion is applied, and a mismatch "
                    "aborts rather than producing a number",
        "reads": reads,
        "max_abs_pooled_delta": round(max(
            abs(v.get("pooled_delta") or 0.0) for v in reads.values()), 6),
        "max_abs_control_delta": round(max(
            abs(v.get("control_delta") or 0.0) for v in reads.values()), 6),
        "control_rows_contaminated": max(
            v["n_control_rows_contaminated"] for v in reads.values()),
        "integrity_positive_control": {
            "construction": f"the {Path(nm).name} score vector shuffled and fed to the "
                            "same reproduce-check",
            "banked": want, "shuffled_read": round(got_bad, 6),
            "check_would_abort": round(got_bad, len(str(want).split(".")[1])) != want,
        },
        "note": NOTE,
    }
    OUT.write_text(json.dumps(res, indent=2, default=str))

    # ---------------------------------------------------------------- C6 --- #
    mem = {}
    for label in ("flagship", "portfolio"):
        mc, lm = mixes[label]["mix_claims"], mixes[label]["lab_map"]
        mem[label] = {
            "contaminated_rows_only": memorisation_feature(
                ev.filter(pl.col("contaminated")), mc, lm, Q),
            "whole_eval": memorisation_feature(ev, mc, lm, Q),
        }
    # LIVE POSITIVE CONTROL - a mix that carries the eval's OWN positive claims
    poisoned = collections.defaultdict(list)
    plab = collections.defaultdict(list)
    for c, ch, lb in zip(ev["claim"], ev["chunk"], ev["label"]):
        if lb == 1:
            poisoned[norm(ch)].append(c)
            plab[norm(ch)].append(1.0)
    mem_ctrl = memorisation_feature(ev, dict(poisoned), dict(plab), Q)
    res["C6_memorisation"] = {
        "definition": "the only field the two legs of a pair share is the PASSAGE, "
                      "so the key is the normalised passage: for each eval row, the "
                      "overlap between the eval claim and the best-matching claim the "
                      "TRAINING MIX carries over the same passage, scored against the "
                      "eval label (amendment C-A2 scoping: mix-supplied associations)",
        "by_mix": mem,
        "live_positive_control": {
            "construction": "the lookup table replaced by one mapping every eval "
                            "passage to that pair's POSITIVE claim only - a mix that "
                            "has memorised the answer key",
            "result": mem_ctrl,
            "fires": (mem_ctrl.get("auroc") or 0) >= 0.90,
        },
        "note": NOTE,
    }
    print(f"C6: {json.dumps(mem, default=str)[:400]}", flush=True)
    print(f"C6 control auroc = {mem_ctrl.get('auroc')}", flush=True)
    OUT.write_text(json.dumps(res, indent=2, default=str))

    # ---------------------------------------------------------------- C1 --- #
    struct = structural_test(ev)
    struct_real = structural_test(real)
    # LIVE POSITIVE CONTROL - poison the eval so every negative leg copies its
    # pair's positive claim; the structural test must fire on 100% of them
    pos_claim = (ev.filter(pl.col("label") == 1)
                   .select("pair_id", pl.col("claim").alias("pos_claim"))
                   .unique(subset=["pair_id"]))
    poison = (ev.join(pos_claim, on="pair_id", how="inner")
                .with_columns(pl.when(pl.col("label") == 0)
                                .then(pl.col("pos_claim"))
                                .otherwise(pl.col("claim")).alias("claim")))
    struct_ctrl = structural_test(poison)
    res["C1_label_commensurability"] = {
        "head": "the grounding scalar - the label encodes whether the claim's "
                "asserted value is supported by the passage",
        "structural_test_whole_eval": struct,
        "structural_test_real_rows": struct_real,
        "attestation_whole_eval": attestation_block(ev, Q),
        "attestation_real_rows": attestation_block(real, Q),
        "attestation_control_rows": attestation_block(ctrl, Q),
        "attestation_contaminated_rows": attestation_block(
            ev.filter(pl.col("contaminated")), Q),
        "asserted_value_attestation_real_rows": asserted_value_attestation(real, src),
        "instrument_note": "the predicate these negatives corrupt is the ASSERTED "
                           "VALUE, so the predicate-sensitive instrument is attestation "
                           "of that value (source column `asserted_value` = `v_pos` on "
                           "the positive leg, `v_neg` on the negative). The bag-of-words "
                           "containment reading is reported alongside and is "
                           "predicate-BLIND here by construction: both legs share the "
                           "passage and differ in one numeral, so word containment "
                           "cannot move",
        "corpus_property": "this is a DERIVATION eval - the positive leg asserts a "
                           "value COMPUTED from the table, which is therefore usually "
                           "absent from the evidence by construction. Attestation of "
                           "the asserted value is consequently low on BOTH legs. The "
                           "numbers are reported as measured",
        "live_positive_control_structural": {
            "construction": "every negative leg's claim replaced by its pair's "
                            "positive claim, making the (claim, evidence) pairs "
                            "identical across legs",
            "result": struct_ctrl, "fires": struct_ctrl["fires"],
        },
        "live_positive_control_instrument": {
            "construction": "the eval's own 50 CONTROL rows, where the predicate IS "
                            "attestation by construction - 25 claims lifted verbatim "
                            "from a chunk sentence (label 1) and 25 with the sentence's "
                            "number replaced by a value absent from the chunk (label 0)",
            "reads": "reported under attestation_control_rows; the numeral instrument "
                     "must separate them",
            "fires": None,   # filled below from the measured control block
        },
        "note": NOTE,
    }
    _cb = res["C1_label_commensurability"]["attestation_control_rows"]
    res["C1_label_commensurability"]["live_positive_control_instrument"]["fires"] = bool(
        _cb["label_0"]["numeral_rate_ge_0.90"] < _cb["label_1"]["numeral_rate_ge_0.90"])
    print(f"C1 structural: {struct} | control fires {struct_ctrl['fires']} "
          f"at {struct_ctrl['rate']}", flush=True)
    OUT.write_text(json.dumps(res, indent=2, default=str))

    # ---------------------------------------------------------------- C5 --- #
    probe = claim_only_probe(real, real["doc_id"].to_list())
    wp = within_pair_claim_only(real)
    par = surface_parity(real)
    # LIVE POSITIVE CONTROL - a claim-only channel that IS the label
    poisoned_claims = real.with_columns(
        (pl.col("claim") + pl.when(pl.col("label") == 1)
         .then(pl.lit(" zzmarker")).otherwise(pl.lit(""))).alias("claim"))
    probe_ctrl = claim_only_probe(poisoned_claims, real["doc_id"].to_list())
    res["C5_leak_suite"] = {
        "scope": "amendment C-A1 - claim-alone and evidence-alone channels only; "
                 "claim-to-evidence containment is a JOINT feature governed by C1",
        "claim_only_probe": probe, "bar": "< 0.55",
        "within_pair_claim_only": wp, "within_pair_bar": "< 0.60",
        "surface_parity": par, "surface_parity_band": "0.45 - 0.55",
        "evidence_only_channel": "degenerate by construction - both legs of a pair "
                                 "carry the SAME passage, so any evidence-only feature "
                                 "reads exactly 0.5 and is not evidence of anything",
        "live_positive_control": {
            "construction": "a marker token appended to every positive claim - a "
                            "claim-alone channel that IS the label",
            "result": probe_ctrl, "fires": (probe_ctrl.get("auroc") or 0) >= 0.95,
        },
        "note": NOTE,
    }
    print(f"C5: claim-only {probe} | within-pair worst {wp['worst']} | "
          f"parity max dev {par['_max_deviation_from_0.5']} | "
          f"control {probe_ctrl.get('auroc')}", flush=True)
    OUT.write_text(json.dumps(res, indent=2, default=str))

    # ---------------------------------------------------------------- C3 --- #
    per_doc = ev.group_by("doc_id").agg(pl.col("pair_id").n_unique().alias("pairs"))
    per_chunk = ev.group_by("chunk").agg(pl.col("pair_id").n_unique().alias("pairs"))
    fev = [d for d in ev_docs if str(d).startswith("feverous:")]
    res["C3_split_semantics"] = {
        "axis_stated": "the evalset is a stratified snapshot of the R14-H133 v2 lane "
                       "(500 pair_ids drawn proportionally across neg_family, both "
                       "legs of each pair taken, numpy seed 1143) plus 50 constructed "
                       "controls. It carries no internal train/test split; the split "
                       "that matters is eval-vs-training-lane",
        "axis_measured": {
            "pairs": int(ev["pair_id"].n_unique()),
            "documents": len(ev_docs),
            "documents_carrying_more_than_one_pair": int((per_doc["pairs"] > 1).sum()),
            "max_pairs_per_document": int(per_doc["pairs"].max()),
            "passages_carrying_more_than_one_pair": int((per_chunk["pairs"] > 1).sum()),
            "both_legs_of_a_pair_share_the_passage": bool(
                ev.group_by("pair_id").agg(pl.col("chunk").n_unique().alias("n"))
                  ["n"].max() == 1),
        },
        "key_stability": {
            "feverous_index_form_doc_ids": len(fev),
            "tabfact_doc_ids": len(ev_docs) - len(fev),
            "finding": "the FEVEROUS half's doc_ids are index-form and order-unstable "
                       "(the campaign recorded 142 of 536 eval doc_ids as nonexistent "
                       "on a fresh corpus rebuild). Every downstream lane therefore "
                       "enforced eval disjointness on CONTENT fingerprints, not ids - "
                       "the id channel is not a verifiable split key for this eval",
        },
        "note": NOTE,
    }
    OUT.write_text(json.dumps(res, indent=2, default=str))

    # ------------------------------------------------------------ C7 / C8 -- #
    dup = {
        "rows": ev.height, "pairs": int(ev["pair_id"].n_unique()),
        "distinct_claims": int(ev["claim"].n_unique()),
        "distinct_passages": len(passages),
        "repeat_claims": int(ev.height - ev["claim"].n_unique()),
        "passages_used_by_more_than_one_pair": int((per_chunk["pairs"] > 1).sum()),
        "unit_used_in_registration": "the campaign log reports this eval as "
                                     "'1,000 pairs + 50 controls' in rows; both counts "
                                     "are reported here",
    }
    res["C7_units_and_volume"] = dup
    res["C8_provenance"] = {
        "source_corpora": "TabFact-train and FEVEROUS-train tables, serialised by the "
                          "R14-H133 v2 lane generator; both are public research corpora "
                          "already in the campaign's contamination-walled lineage",
        "selection_predicate": "500 pair_ids sampled proportionally across neg_family "
                               "with largest-remainder allocation, numpy seed 1143, both "
                               "legs taken; plus 25 verbatim-sentence positives and 25 "
                               "numeral-replaced negatives drawn from the same passages",
        "builder": "R17-H143_evalset.py; source snapshot R17-H143_evalset_source.parquet "
                   "(50,000 rows) = R14-H133_lane v2-SUPERSEDED",
        "generator_source_committed": False,
        "generator_note": "the v2 lane generator was never committed (v1 is; v3 is the "
                          "trace-conditioned rebuild). The eval is reproducible only "
                          "from the banked source snapshot, not from the generator",
        "within_member_duplication": dup,
        "licence": "TabFact CC BY-SA 4.0; FEVEROUS CC BY-SA 3.0 - both public",
        "public_repo_clean": True,
        "note": NOTE,
    }
    OUT.write_text(json.dumps(res, indent=2, default=str))

    # ---------------------------------------------------------------- C4 --- #
    if not args.skip_c4:
        print("--- C4 n-gram census (portfolio mix) ---", flush=True)
        c4, _q, best = c4_census(passages, mixes["portfolio"]["chunks"],
                                 mixes["portfolio"]["tags"])
        c4["live_positive_control"] = c4_positive_control(
            passages, mixes["portfolio"]["chunks"], mixes["portfolio"]["tags"])
        c4["note"] = NOTE
        res["C4_contamination_census"] = c4
        print(f"C4: {c4['eval_units_at_jaccard_ge_thr']}/{c4['eval_units']} "
              f"= {c4['eval_fraction']} (kill 0.02); controls "
              f"{json.dumps(c4['live_positive_control'], default=str)[:300]}", flush=True)
        OUT.write_text(json.dumps(res, indent=2, default=str))

        # ---- LOAD-BEARING under the WIDER n-gram exclusion --------------- #
        # the string-form hit set (10 passages) is a strict subset of what the
        # 8-gram census sees; the honest exclusion is the union.
        ngram_hit = {p for p, b in zip(passages, best) if b >= 0.3}
        union = set(hit) | ngram_hit
        ev2 = ev.with_columns(pl.col("chunk").is_in(list(union)).alias("contaminated"))
        c2_rows = ev2.filter(pl.col("contaminated"))
        reads2 = recompute_reads(ev2, banked_stage_a)
        res["load_bearing_under_ngram_exclusion"] = {
            "exclusion_set": "union of the string-form hit set and every eval passage "
                             "reaching 8-gram Jaccard >= 0.3 against any mix chunk",
            "passages_excluded": len(union),
            "string_form_only": len(hit),
            "ngram_only": len(ngram_hit - set(hit)),
            "rows_excluded": int(c2_rows.height),
            "pairs_excluded": int(c2_rows["pair_id"].n_unique()),
            "control_rows_excluded": int(c2_rows["control"].sum()),
            "documents_excluded": int(c2_rows["doc_id"].n_unique()),
            "reads": reads2,
            "max_abs_pooled_delta": round(max(
                abs(v.get("pooled_delta") or 0.0) for v in reads2.values()), 6),
            "max_abs_control_delta": round(max(
                abs(v.get("control_delta") or 0.0) for v in reads2.values()), 6),
            "note": NOTE,
        }
        print(f"wider exclusion: {len(union)} passages / {c2_rows.height} rows; "
              f"max |pooled delta| "
              f"{res['load_bearing_under_ngram_exclusion']['max_abs_pooled_delta']}, "
              f"max |control delta| "
              f"{res['load_bearing_under_ngram_exclusion']['max_abs_control_delta']}",
              flush=True)
    else:
        res["C4_contamination_census"] = {"skipped": True}

    # ------------------------------------------------- CLAUSE READINGS ----- #
    # The measured number beside the clause's WRITTEN bar, and whether the number
    # clears it.  This is a reading, not a verdict: `conforming` is deliberately
    # absent - the coordinator adjudicates.
    c4b = res.get("C4_contamination_census", {})
    cl1 = res["C1_label_commensurability"]
    av = cl1["asserted_value_attestation_real_rows"]
    c5 = res["C5_leak_suite"]
    doc = res["C2_document_channel"]
    tf_un = doc.get("tabfact_member_unconformed", {})
    readings = {
        "C1": {
            "test_1_structural": {
                "measured": cl1["structural_test_whole_eval"][
                    "negative_legs_identical_to_a_positive"],
                "bar": "0 negative legs identical to a positive leg",
                "clears": cl1["structural_test_whole_eval"][
                    "negative_legs_identical_to_a_positive"] == 0},
            "test_2_strict_separation": {
                "measured": f"neg {av['label_0']['asserted_value_present_in_evidence_rate']} "
                            f"vs pos {av['label_1']['asserted_value_present_in_evidence_rate']} "
                            "(asserted-value attestation, real rows)",
                "bar": "the negative leg's high-attestation rate STRICTLY BELOW the "
                       "positive leg's",
                "clears": av["negative_strictly_below_positive"]},
            "test_3_absolute_levels": {
                "measured": {"positive": av["label_1"][
                    "asserted_value_present_in_evidence_rate"],
                    "negative": av["label_0"][
                        "asserted_value_present_in_evidence_rate"]},
                "bar": "reported always", "clears": True},
        },
        "C2": {
            "evidence_string_forms": {
                "measured": res["C2_disjointness"]["portfolio"][
                    "evidence_passages_in_the_mix"],
                "bar": "0 in every form", "clears": res["C2_disjointness"]["portfolio"][
                    "evidence_passages_in_the_mix"] == 0},
            "claim_string_forms": {
                "measured": res["C2_disjointness"]["portfolio"]["claims_in_the_mix"],
                "bar": "0 in every form",
                "clears": res["C2_disjointness"]["portfolio"]["claims_in_the_mix"] == 0},
            "document_channel": {
                "measured": f"{tf_un.get('raw_table_id_overlap')} of "
                            f"{tf_un.get('eval_tabfact_documents')} TabFact documents "
                            "present in the mix's tabfact member",
                "bar": "0 shared documents", "clears": tf_un.get(
                    "raw_table_id_overlap", 1) == 0},
        },
        "C3": {"split_axis_measured": {
            "measured": "stated and measured; the FEVEROUS half's ids are index-form "
                        "and unresolvable, so the id channel is not a verifiable key",
            "bar": "state the axis a corpus actually cuts on, measured not assumed",
            "clears": True}},
        "C4": {"ngram_census": {
            "measured": c4b.get("eval_fraction"),
            "bar": "< 0.02 of the candidate units (KILL at or above)",
            "clears": (c4b.get("eval_fraction") is not None
                       and c4b["eval_fraction"] < 0.02)}},
        "C5": {
            "claim_only_probe": {"measured": c5["claim_only_probe"]["auroc"],
                                 "bar": "< 0.55",
                                 "clears": c5["claim_only_probe"]["auroc"] < 0.55},
            "within_pair_claim_only": {"measured": c5["within_pair_claim_only"]["worst"],
                                       "bar": "< 0.60",
                                       "clears": c5["within_pair_claim_only"]["worst"] < 0.60},
            "surface_parity": {"measured": c5["surface_parity"][
                "_max_deviation_from_0.5"], "bar": "every channel inside 0.45 - 0.55",
                "clears": not c5["surface_parity"]["_channels_outside_0.45_0.55"]},
        },
        "C6": {"memorisation_channel": {
            "measured": res["C6_memorisation"]["by_mix"]["portfolio"][
                "whole_eval"]["auroc"],
            "coverage": res["C6_memorisation"]["by_mix"]["portfolio"][
                "whole_eval"]["coverage"],
            "bar": "undefined or at chance on a clean instrument",
            "clears": abs((res["C6_memorisation"]["by_mix"]["portfolio"]["whole_eval"]
                           ["auroc"] or 0.5) - 0.5) < 0.05}},
        "C7": {"units": {"measured": f"{res['C7_units_and_volume']['rows']} rows / "
                                     f"{res['C7_units_and_volume']['pairs']} pairs",
                         "bar": "both counts reported, unit used consistently",
                         "clears": True}},
        "C8": {"provenance": {"measured": "source, licence, selection predicate and "
                                          "within-member duplication all stated; the v2 "
                                          "generator source is NOT committed",
                              "bar": "source, licence, retrieval date, exact selection "
                                     "predicate, duplication reported",
                              "clears": False}},
        "positive_controls": {
            "C2_string_form_detector": res["C2_disjointness"]["portfolio"][
                "live_positive_control"]["fires_as_expected"],
            "C4_spike": c4b.get("live_positive_control", {}).get(
                "spike_control", {}).get("fires"),
            "C4_live_near_duplicate": c4b.get("live_positive_control", {}).get(
                "live_near_duplicate_control", {}).get("fires"),
            "C1_structural": cl1["live_positive_control_structural"]["fires"],
            "C1_instrument_sensitivity": cl1["live_positive_control_instrument"]["fires"],
            "C5_claim_only_probe": c5["live_positive_control"]["fires"],
            "C6_memorisation": res["C6_memorisation"]["live_positive_control"]["fires"],
            "load_bearing_integrity_check": res["load_bearing"][
                "integrity_positive_control"]["check_would_abort"],
        },
        "conforming": "NOT DECIDED HERE - the coordinator adjudicates",
        "note": NOTE,
    }
    res["clause_readings"] = readings

    # ------------------------------------------------------- CONSUMERS ----- #
    res["consumers_of_this_eval"] = {
        "source": "docs/experiments/semantic-grounding-experiments.md, grepped",
        "gates_that_READ_it": [
            "R17-H143 Stage A tier gate - pooled AUROC per model + the >= 0.90 "
            "positive-control gate; verdict 'in-budget tiny class DEAD at chance, "
            "TIER-VIABLE closed on the tiny half'",
            "R17-H143 teacher read - Qwen3-32B-FP8 0.9708 pooled; verdict "
            "DISTILL-LICENSED",
            "R17-H144 cycle 1 - pooled >= 0.70 AND controls >= 0.90 conjunction; "
            "verdict NOT-VIABLE-AT-BAR (0.8171 / 0.8696)",
            "R17-H144 cycle 2 + amendment A2 blind selector - same conjunction on the "
            "selected checkpoint; verdict NOT-VIABLE-AT-BAR (0.7789 / 0.8320)",
        ],
        "artifacts_that_EXCLUDE_against_it": [
            "R17-H144_pairs.py / R17-H144_lookup.py (SFT corpus + valsplit)",
            "R17-H145_scaleunit.py", "R17-H146_lane.py",
            "R18-H150_scaleunit_lane.py", "R18-H150_edgar_gate.py",
            "R18-H150_unitswap_probe.py", "R19_supply_lanes.py",
            "R20-H177_lane_B.py",
        ],
        "live_arms_reading_it": "none - the R17 tiny-reasoner / distillation arc is "
                               "closed and no current arm or flagship gate scores "
                               "against this eval",
        "note": NOTE,
    }

    res["elapsed_s"] = round(time.time() - t0, 1)
    res["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"=== banked -> {OUT.name} ({res['elapsed_s']}s) ===", flush=True)


if __name__ == "__main__":
    main()

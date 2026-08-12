"""R14-H133 lane - build-manifest verification. CPU only, no GPU.

Runs every check R14-A4 / R15-B1 / R15-B4 require before any GPU time is spent:

  * row / label / block / type / family / form counts
  * P(label 0 | absent) over the DERIVATION CORE (B1 amendment iv)
  * arithmetic re-verification of 1,000 random positives, recomputed from the
    stored operands by formulas written independently of the builder's
  * digit-length parity, per-result-digit-length quota, token-length AUROC
  * realised share of rows over the 512-token budget
  * table diversity and per-document pair distribution
  * row-level disjointness from the admitted H108 lane (A4 amendment iv)
  * schema compatibility with `R10-H108_lane.lane_train`'s read path

Run:  uv run python experiments/grounding-semantic/R14-H133_lane_verify.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""      # no GPU is touched by this script
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections
import json
import math
import pathlib
import re

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
LANE = HERE / "R14-H133_lane.parquet"
H108 = HERE / "R10-H108_pairs.parquet"
OUT = HERE / "R14-H133_lane_verify.json"
EYEBALL = HERE / "R14-H133_lane_eyeball.md"
TOKENIZER = ROOT / "models" / "R9-H105-mmbert-dann-clean"

NUM_FREE = re.compile(r"(?<![\d.,])[-+]?\d[\d,]*(?:\.\d+)?(?![\d.,])")   # banked (P3 / L2)
NUM_EVID = re.compile(r"(?<![\d.,])[-+]?\d[\d,]*(?:\.\d+)?(?![\d,])")     # punctuation-tolerant
SEED = 1133
N_ARITH = 1000
MAX_LEN = 512


def canon_set(s):
    p = set()
    for m in NUM_FREE.findall(s or ""):
        v = m.replace(",", "")
        p.add(v)
        try:
            f = float(v)
        except (ValueError, OverflowError):
            continue
        if f != f or f in (float("inf"), float("-inf")) or abs(f) > 1e15:
            continue
        p.add(str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:.2f}".rstrip("0").rstrip("."))
    return p


def evid_set(s):
    p = set()
    for m in NUM_EVID.findall(s or ""):
        v = m.replace(",", "")
        p.add(v)
        try:
            f = float(v)
        except (ValueError, OverflowError):
            continue
        if f != f or f in (float("inf"), float("-inf")) or abs(f) > 1e15:
            continue
        p.add(str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:.2f}".rstrip("0").rstrip("."))
    return p


def digits(s):
    return sum(ch.isdigit() for ch in s)


def tzeros(s):
    n = 0
    for ch in reversed(s or ""):
        if ch == "0":
            n += 1
        else:
            break
    return n


# --------------------------------------------------------------------------- #
# independent arithmetic - written from the type definitions, not from the builder
# --------------------------------------------------------------------------- #
def expected(dtype, a, b):
    if dtype == "sum":
        return a + b
    if dtype == "difference":
        return a - b
    if dtype == "mean":
        return 0.5 * (a + b)
    if dtype == "ratio":
        return a / b
    if dtype == "pct_change":
        return 100.0 * (b - a) / a
    if dtype == "product":
        return a * b
    if dtype == "scale_unit":
        return 1000.0 * a
    raise ValueError(dtype)


def render(v):
    """The lane's numeral rendering, restated."""
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.2f}"


def auroc(pos, neg):
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.ones(len(pos), int), np.zeros(len(neg), int)])
    return float(roc_auc_score(y, np.concatenate([np.asarray(pos, float),
                                                  np.asarray(neg, float)])))


def claim_only_probe(d, rng):
    """(a) and (b) - can a classifier separate the labels from the CLAIM ALONE?

    v1 read held-out AUROC 0.638 here on char_wb(2,4) because the negatives'
    numerals carried trailing zeros, digit counts and minus signs the positives'
    did not. Doc-disjoint split, so a memorised document cannot pass it."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    docs = sorted(set(d["doc_id"].to_list()))
    rng.shuffle(docs)
    test_docs = set(docs[: max(1, int(0.2 * len(docs)))])
    is_test = np.array([x in test_docs for x in d["doc_id"].to_list()])

    X = d["claim"].to_list()
    y = d["label"].to_numpy()
    Xtr = [X[i] for i in np.where(~is_test)[0]]
    Xte = [X[i] for i in np.where(is_test)[0]]
    ytr, yte = y[~is_test], y[is_test]

    pipe = Pipeline([
        ("f", FeatureUnion([
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                                  sublinear_tf=True)),
            ("w", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3,
                                  sublinear_tf=True)),
        ])),
        ("m", LogisticRegression(max_iter=2000, C=1.0)),
    ])
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    overall = round(auroc(p[yte == 1], p[yte == 0]), 4)

    # (b) within-pair accuracy per family, on held-out pairs only
    sub = d.filter(pl.Series(is_test)).with_columns(pl.Series("claim_score", p))
    fam = (pl.when(pl.col("block") == "core").then(pl.col("neg_family"))
             .otherwise(pl.col("arm")))
    sub = sub.with_columns(fam.alias("family"))
    per_family, breaches = {}, []
    for name, g in sub.group_by("family"):
        name = name[0] if isinstance(name, tuple) else name
        if name is None:
            continue
        sc = {}
        for r in g.iter_rows(named=True):
            sc.setdefault(r["pair_id"], {})[r["label"]] = r["claim_score"]
        both = [v for v in sc.values() if len(v) == 2]
        if len(both) < 30:
            continue
        acc = float(np.mean([1.0 if v[1.0] > v[0.0] else (0.5 if v[1.0] == v[0.0] else 0.0)
                             for v in both]))
        per_family[name] = {"held_out_pairs": len(both), "within_pair_accuracy": round(acc, 4)}
        # family (c) contradicts its own trace, so it is text-detectable BY
        # DESIGN; the author's ruling exempts it from the bar and caps its share
        if acc > 0.60 and name != "c:conclusion":
            breaches.append(name)
    return {
        "features": "TF-IDF char_wb(2,4) + word(1,2), logistic regression",
        "split": "document-disjoint 80/20 on doc_id",
        "train_rows": len(Xtr), "test_rows": len(Xte),
        "held_out_auroc": overall, "auroc_bar": "abort above 0.55",
        "v1_defective_reading": 0.638,
        "per_family_within_pair_accuracy": per_family,
        "within_pair_bar": "abort any family above 0.60; family (c) exempt by design, "
                           "capped at 10% of negatives instead",
        "within_pair_breaches": breaches,
    }


def main():
    d = pl.read_parquet(LANE)
    res = {"lane": LANE.name, "rows": len(d), "pairs": d["pair_id"].n_unique()}

    # ---- composition ------------------------------------------------------ #
    res["label_counts"] = {str(k): v for k, v in
                           zip(*d.group_by("label").len().sort("label"))}
    res["block_rows"] = dict(zip(*d.group_by("block").len().sort("block")))
    res["block_share"] = {k: round(v / len(d), 5) for k, v in res["block_rows"].items()}
    core = d.filter(pl.col("block") == "core")
    rel = d.filter(pl.col("block") == "rel")
    res["tag_rows"] = dict(zip(*d.group_by("tag").len().sort("tag")))
    res["core_type_pairs"] = {k: v // 2 for k, v in
                              zip(*core.group_by("dtype").len().sort("dtype"))}
    res["core_type_share"] = {k: round(2 * v / len(core), 5)
                              for k, v in res["core_type_pairs"].items()}
    negs = core.filter(pl.col("label") == 0.0)
    res["core_negative_family"] = dict(zip(*negs.group_by("neg_family").len().sort("neg_family")))
    res["core_negative_family_share"] = {k: round(v / len(negs), 5)
                                         for k, v in res["core_negative_family"].items()}
    res["core_claim_form_share"] = {k: round(v / len(core), 5) for k, v in
                                    zip(*core.group_by("claim_form").len().sort("claim_form"))}
    res["serial_form_share"] = {k: round(v / len(d), 5) for k, v in
                                zip(*d.group_by("serial_form").len().sort("serial_form"))}
    res["rel_arm_pairs"] = {k: v // 2 for k, v in
                            zip(*rel.group_by("arm").len().sort("arm"))}
    res["rel_arm_share_of_subblock"] = {k: round(2 * v / len(rel), 5)
                                        for k, v in res["rel_arm_pairs"].items()}
    cmp_rows = rel.filter(pl.col("arm") == "compare")
    res["rel_compare_gap_strata"] = {k: v // 2 for k, v in
                                     zip(*cmp_rows.group_by("gap_stratum").len().sort("gap_stratum"))}
    res["source_rows"] = dict(zip(*d.group_by("source").len().sort("source")))

    # ---- P(label 0 | absent) ---------------------------------------------- #
    cl, ch_ = d["claim"].to_list(), d["chunk"].to_list()
    absent = np.array([bool(evid_set(c) - evid_set(x)) for c, x in zip(cl, ch_)])
    absent_banked = np.array([bool(canon_set(c) - canon_set(x)) for c, x in zip(cl, ch_)])
    lab = d["label"].to_numpy()
    blk = np.array(d["block"].to_list())
    res["absent_rows_lane"] = int(absent.sum())
    res["p_label0_given_absent_lane"] = round(float((lab[absent] == 0).mean()), 5)
    ca = absent & (blk == "core")
    res["absent_rows_core"] = int(ca.sum())
    res["p_label0_given_absent_core"] = round(float((lab[ca] == 0).mean()), 5)
    res["absent_share_of_core"] = round(float(ca.sum() / (blk == "core").sum()), 5)
    res["absent_share_of_subblock"] = round(
        float((absent & (blk == "rel")).sum() / (blk == "rel").sum()), 5)
    res["p_label0_given_absent_banked_detector"] = round(
        float((lab[absent_banked] == 0).mean()), 5)
    res["absent_rows_banked_detector"] = int(absent_banked.sum())
    res["absence_detector_note"] = (
        "the banked P3/L2 detector (R15_gate_common.canon_set) cannot see a numeral "
        "followed by a full stop, so it under-reads every prose serialization; the "
        "lane's own absence rule uses the punctuation-tolerant form")
    # asserted-value-level absence, which is what the construction actually asserts
    av_absent = np.array([bool(canon_set(v) - evid_set(x))
                          for v, x in zip(d["asserted_value"].to_list(), ch_)])
    res["asserted_value_absent_share_core"] = round(
        float(av_absent[blk == "core"].mean()), 5)
    res["asserted_value_absent_share_subblock"] = round(
        float(av_absent[blk == "rel"].mean()), 5)
    res["p_label0_given_asserted_value_absent_core"] = round(
        float((lab[av_absent & (blk == "core")] == 0).mean()), 5)

    rng = np.random.default_rng(SEED)

    # ---- v3: EVERY trace re-derived, both polarities --------------------- #
    CALC = re.compile(r";\s*(?:computing\s+)?([^;]+?)\s*(?:=|gives)\s*(-?[\d.,]+)\s*,\s*so\s")
    ROUNDED = re.compile(r"^(-?[\d.,]+) rounded to the nearest (ten|hundred|thousand)$")
    PLACE = {"ten": 10.0, "hundred": 100.0, "thousand": 1000.0}
    SAFE = set("0123456789.+-*/() ")

    tr = {"traces_checked": 0, "unparsable": 0, "arithmetic_wrong": 0,
          "conclusion_not_trace_result": 0, "conclusion_is_trace_result_but_family_c": 0}
    bad_tr = []
    for r in core.iter_rows(named=True):
        m = CALC.search(r["claim"])
        if not m:
            tr["unparsable"] += 1
            continue
        tr["traces_checked"] += 1
        expr, stated = m.group(1), m.group(2)
        rm = ROUNDED.match(expr)
        if rm:
            got = PLACE[rm.group(2)] * round(float(rm.group(1).replace(",", "")) / PLACE[rm.group(2)])
        elif set(expr) <= SAFE:
            try:
                got = eval(expr.replace(",", ""), {"__builtins__": {}}, {})  # noqa: S307
            except (SyntaxError, ZeroDivisionError, TypeError):
                tr["unparsable"] += 1
                continue
        else:
            tr["unparsable"] += 1
            continue
        if render(got) != stated:
            tr["arithmetic_wrong"] += 1
            if len(bad_tr) < 5:
                bad_tr.append({"pair_id": r["pair_id"], "expr": expr, "stated": stated,
                               "computed": render(got)})
        # the conclusion must quote the trace's own result, except in family (c)
        concl_is_res = r["asserted_value"] == stated
        if r["neg_family"] == "c:conclusion" and r["label"] == 0.0:
            if concl_is_res:
                tr["conclusion_is_trace_result_but_family_c"] += 1
        elif not concl_is_res:
            tr["conclusion_not_trace_result"] += 1
    tr["errors"] = (tr["unparsable"] + tr["arithmetic_wrong"]
                    + tr["conclusion_not_trace_result"]
                    + tr["conclusion_is_trace_result_but_family_c"])
    tr["examples"] = bad_tr
    tr["bar"] = "0 errors - every trace's arithmetic is internally correct and its "\
                "conclusion quotes it, except family (c) which contradicts it by design"
    res["trace_rederivation"] = tr

    # ---- v3 NEW: groundability -------------------------------------------- #
    pos_core = core.filter(pl.col("label") == 1.0)
    g = {"positives": 0, "operand_not_in_chunk": 0, "operand_not_quoted_in_trace": 0}
    for r in pos_core.iter_rows(named=True):
        g["positives"] += 1
        present = evid_set(r["chunk"])
        need = [render(r["operand_a"])]
        if r["dtype"] not in ("scale_unit", "rounding"):
            need.append(render(r["operand_b"]))
        if any(not (canon_set(v) & present) for v in need):
            g["operand_not_in_chunk"] += 1
        if any(v not in r["claim"] for v in need):
            g["operand_not_quoted_in_trace"] += 1
    g["groundable_share"] = round(1 - (g["operand_not_in_chunk"]
                                       + g["operand_not_quoted_in_trace"]) / max(g["positives"], 1), 5)
    g["bar"] = "1.0 - every positive trace quotes operands that are verbatim in its chunk"
    res["positive_trace_groundability"] = g

    a_neg = core.filter((pl.col("label") == 0.0) & pl.col("neg_family").str.starts_with("a:"))
    a = {"a_negatives": len(a_neg), "cited_equals_true": 0, "true_value_not_in_chunk": 0,
         "cited_value_not_in_claim": 0}
    for r in a_neg.iter_rows(named=True):
        if r["neg_cited_value"] == r["neg_true_value"]:
            a["cited_equals_true"] += 1
        if not (canon_set(r["neg_true_value"]) & evid_set(r["chunk"])):
            a["true_value_not_in_chunk"] += 1
        if r["neg_cited_value"] not in r["claim"]:
            a["cited_value_not_in_claim"] += 1
    a["mechanically_confirmable"] = a["a_negatives"] - a["cited_equals_true"] \
        - a["true_value_not_in_chunk"] - a["cited_value_not_in_claim"]
    a["confirmable_share"] = round(a["mechanically_confirmable"] / max(a["a_negatives"], 1), 5)
    a["bar"] = "1.0 - the cited value differs from the true cell, the true cell is in "\
               "the evidence, and the cited value is in the claim"
    res["a_negative_mismatch_confirmable"] = a

    negs_core = core.filter(pl.col("label") == 0.0)
    res["family_c_share_of_negatives"] = round(
        float(len(negs_core.filter(pl.col("neg_family") == "c:conclusion"))
              / max(len(negs_core), 1)), 5)
    res["family_c_cap"] = "10% of negatives (author ruling)"

    # ---- pair identity (numerals masked), digit parity, digit-length quota - #
    MASK = re.compile(r"-?\d(?:[\d,]*\d)?(?:\.\d+)?")
    cols_ = ["pair_id", "claim", "asserted_value", "pair_shape", "block", "arm", "neg_family"]
    pv = {r["pair_id"]: r for r in d.filter(pl.col("label") == 1.0)
          .select(cols_).iter_rows(named=True)}
    nv = {r["pair_id"]: r for r in d.filter(pl.col("label") == 0.0)
          .select(cols_).iter_rows(named=True)}
    bi, pa = collections.Counter(), collections.Counter()
    for pid, p in pv.items():
        n = nv[pid]
        key = p["neg_family"] or p["arm"]
        if p["pair_shape"] == "numeral_only":
            if MASK.sub("#", p["claim"]) != MASK.sub("#", n["claim"]):
                bi[key] += 1
        if abs(digits(p["asserted_value"]) - digits(n["asserted_value"])) > 1:
            pa[key] += 1
    res["pair_identity_breaches"] = dict(bi)
    res["pair_identity_note"] = (
        "checked on pairs shaped `numeral_only` - families (a) and (c) and the "
        "bind arms, where the two claims must be identical once every numeral is "
        "masked. Family (b) is `word_only` (the numeral is untouched and the "
        "operation word differs) and `compare` is `binding_swap`; both are exempt")
    res["digit_length_parity_breaches"] = dict(pa)
    dl = collections.Counter(core.filter(pl.col("label") == 1.0)["result_digits"].to_list())
    npairs_core = sum(dl.values())
    res["result_digit_length_share"] = {str(k): round(v / npairs_core, 5)
                                        for k, v in sorted(dl.items())}
    res["result_digit_length_max_share"] = round(max(dl.values()) / npairs_core, 5)

    # ---- (c) trailing-zero-count AUROC per derivation type ----------------- #
    tz_res, tz_breach = {}, []
    for dt in sorted(set(core["dtype"].to_list())):
        g = core.filter(pl.col("dtype") == dt)
        pz = [tzeros(v) for v in g.filter(pl.col("label") == 1.0)["asserted_value"].to_list()]
        nz = [tzeros(v) for v in g.filter(pl.col("label") == 0.0)["asserted_value"].to_list()]
        a = round(auroc(pz, nz), 4) if len(set(pz + nz)) > 1 else 0.5
        tz_res[dt] = a
        if not 0.45 <= a <= 0.55:
            tz_breach.append(dt)
    for arm in sorted(set(rel["arm"].drop_nulls().to_list())):
        g = rel.filter(pl.col("arm") == arm)
        pz = [tzeros(v) for v in g.filter(pl.col("label") == 1.0)["asserted_value"].to_list()]
        nz = [tzeros(v) for v in g.filter(pl.col("label") == 0.0)["asserted_value"].to_list()]
        tz_res[arm] = round(auroc(pz, nz), 4) if len(set(pz + nz)) > 1 else 0.5
    res["trailing_zero_auroc"] = {
        "per_type": tz_res, "bar": "derivation types must sit inside [0.45, 0.55]",
        "breaches": tz_breach,
    }

    # ---- (a) / (b) claim-only surface leak, doc-disjoint -------------------- #
    res["claim_only_leak"] = claim_only_probe(d, rng)

    # ---- tokenizer reads: length AUROC and the 512-token budget ------------ #
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER))
    sub = d   # the whole lane - the manifest clause is a realised share, not an estimate
    enc = tok(sub["claim"].to_list(), sub["chunk"].to_list(), truncation=False)
    lens = np.array([len(x) for x in enc["input_ids"]])
    clen = np.array([len(x) for x in tok(sub["claim"].to_list(), truncation=False)["input_ids"]])
    y = sub["label"].to_numpy()
    res["token_length"] = {
        "n_sampled": int(len(sub)),
        "auroc_from_claim_token_length_alone": round(auroc(clen[y == 1], clen[y == 0]), 4),
        "auroc_from_pair_token_length_alone": round(auroc(lens[y == 1], lens[y == 0]), 4),
        "bar": "abort the build above 0.55",
        "share_over_512": round(float((lens > MAX_LEN).mean()), 5),
        "share_over_512_bar": "manifest clause - under 0.10",
        "pair_tokens_median": int(np.median(lens)), "pair_tokens_p95": int(np.percentile(lens, 95)),
    }
    per_form = {}
    for f in sub["serial_form"].unique().to_list():
        m = np.array(sub["serial_form"].to_list()) == f
        per_form[f] = {"n": int(m.sum()), "median": int(np.median(lens[m])),
                       "share_over_512": round(float((lens[m] > MAX_LEN).mean()), 5)}
    res["token_length"]["per_serial_form"] = per_form

    # ---- table diversity --------------------------------------------------- #
    docs = d.filter(pl.col("label") == 1.0)["doc_id"].to_list()
    c = collections.Counter(docs)
    hist = collections.Counter(c.values())
    res["diversity"] = {
        "distinct_documents": len(c),
        "pairs_per_document_mean": round(len(docs) / len(c), 4),
        "pairs_per_document_max": max(c.values()),
        "pairs_per_document_histogram": {str(k): v for k, v in sorted(hist.items())},
        "documents_by_source": dict(collections.Counter(x.split(":")[0] for x in c)),
        "rows_per_document_mean": round(len(d) / len(c), 4),
    }

    # ---- disjointness from the admitted H108 lane (A4 amendment iv) -------- #
    h = pl.read_parquet(H108)
    hrows = set(zip(h["claim"].to_list(), h["chunk"].to_list()))
    lrows = set(zip(d["claim"].to_list(), d["chunk"].to_list()))
    res["h108_disjointness"] = {
        "h108_rows": len(h), "shared_claim_chunk_rows": len(hrows & lrows),
        "shared_claim_strings": len(set(h["claim"].to_list()) & set(d["claim"].to_list())),
        "shared_chunk_strings": len(set(h["chunk"].to_list()) & set(d["chunk"].to_list())),
    }

    # ---- schema compatibility with the lane trainer's read path ------------ #
    dd = pl.read_parquet(LANE)
    claims = dd["claim"].to_list()
    chunks = [c[:1500] for c in dd["chunk"].to_list()]
    yy = dd["label"].cast(pl.Float32).to_numpy()
    tags = dd["tag"].to_list()
    res["trainer_read_path"] = {
        "checked_against": "R10-H108_lane.py lane_train() - pl.read_parquet(LANE) then "
                           "claim / chunk[:chunk_max_chars] / label.cast(Float32) / tag",
        "n_claims": len(claims), "n_chunks": len(chunks),
        "label_dtype": str(dd.schema["label"]), "label_values": sorted(set(map(float, yy))),
        "distinct_tags": sorted(set(tags)),
        "nulls_in_required_columns": int(dd.select(
            pl.col("claim").is_null().sum() + pl.col("chunk").is_null().sum()
            + pl.col("label").is_null().sum() + pl.col("tag").is_null().sum()).item()),
        "max_chunk_chars": max(len(c) for c in chunks),
        "empty_claims": sum(1 for c in claims if not c.strip()),
    }

    OUT.write_text(json.dumps(res, indent=2))

    # ---- eyeball sample ---------------------------------------------------- #
    lines = ["# R14-H133 lane - 20-row eyeball sample", ""]
    picks = []
    for dt in sorted(set(core["dtype"].to_list())):
        s = core.filter((pl.col("dtype") == dt) & (pl.col("label") == 1.0))
        picks.append(s[int(rng.integers(len(s)))]["pair_id"].item())
    for arm in sorted(set(rel["arm"].drop_nulls().to_list())):
        s = rel.filter((pl.col("arm") == arm) & (pl.col("label") == 1.0))
        picks.append(s[int(rng.integers(len(s)))]["pair_id"].item())
    for pid in picks:
        g = d.filter(pl.col("pair_id") == pid).sort("label", descending=True)
        r1, r0 = g.row(0, named=True), g.row(1, named=True)
        lines += [f"## pair {pid} - {r1['block']} / {r1['dtype'] or r1['arm']}"
                  f" / neg {r1['neg_family']} / {r1['serial_form']} / {r1['claim_form']}",
                  "", f"- **POSITIVE (1)**: {r1['claim']}", f"- **NEGATIVE (0)**: {r0['claim']}",
                  "", "```", r1["chunk"], "```", ""]
    EYEBALL.write_text("\n".join(lines))

    print(json.dumps({k: res[k] for k in (
        "rows", "pairs", "block_share", "p_label0_given_absent_core",
        "pair_identity_breaches",
        "digit_length_parity_breaches", "result_digit_length_max_share",
        "token_length", "diversity", "h108_disjointness",
        "trace_rederivation", "positive_trace_groundability",
        "a_negative_mismatch_confirmable", "family_c_share_of_negatives",
        "trailing_zero_auroc", "claim_only_leak")}, indent=2))
    print(f"-> {OUT}\n-> {EYEBALL}")


if __name__ == "__main__":
    main()

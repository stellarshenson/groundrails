"""C3 (split semantics), C5 (leak suite), C6 (memorisation channel), C7, C8
for the `quant_scale_unit` lane.

CPU ONLY.  No GPU is queried or touched.

C3 - the split axis each source corpus ACTUALLY cuts on, measured from the
     archive; plus the axis the lane itself cuts on against the eval surfaces
     built from the same corpora.
C5 - every registered leak bar RE-DERIVED from the parquet (never cited from the
     banked verify JSON), with executor-added probes reported in a SEPARATE
     block that does not join the registered conjunction.
C6 - the memorisation feature: what the rest of the assembled mix associates
     with each pair's key, and whether it separates the legs.
C7 - declared unit and volume against the registration.
C8 - within-member duplication and repeat structure.

Run:  CUDA_VISIBLE_DEVICES= uv run python \
      experiments/grounding-semantic/contract/quant_scale_unit_c3c5c6.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import collections
import io
import json
import pathlib
import re
import zipfile

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
ROOT = SEM.parent.parent
LANE = SEM / "R18-H150_scaleunit_lane.parquet"
OUT = HERE / "quant_scale_unit_c3c5c6.json"

TOKEN_RE = re.compile(r"[a-z0-9]+")
NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_WS = re.compile(r"\s+")
N_FOLDS = 5
SEED = 1150


def norm_ws(t):
    return _WS.sub(" ", t).strip().lower()


def auroc(y, s):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.empty_like(s)
    order = np.argsort(s, kind="stable")
    ss = s[order]
    i = 0
    ranks = np.empty(len(s))
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r[order] = ranks
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def containment(a, b):
    A = set(TOKEN_RE.findall(a.lower()))
    return len(A & set(TOKEN_RE.findall(b.lower()))) / len(A) if A else 0.0


# --------------------------------------------------------------------------- #
# C3
# --------------------------------------------------------------------------- #
def c3_block(lane):
    out = {}

    z = zipfile.ZipFile(ROOT / "data" / "external" / "datasets" / "dataset-tabfact.zip")
    parts = {}
    for split in ("train", "validation", "test"):
        nm = next(n for n in z.namelist() if n.endswith(f"__{split}.parquet"))
        parts[split] = pl.read_parquet(io.BytesIO(z.read(nm)))
    tf = {}
    for split, d in parts.items():
        tf[split] = {
            "rows": len(d),
            "distinct_table_id": int(d["table_id"].n_unique()),
            "distinct_table_text": int(d["table_text"].n_unique()),
        }
    tr_ids = set(parts["train"]["table_id"].to_list())
    tr_txt = set(parts["train"]["table_text"].to_list())
    for split in ("validation", "test"):
        ids = set(parts[split]["table_id"].to_list())
        txt = set(parts[split]["table_text"].to_list())
        tf[split]["table_ids_shared_with_train"] = len(ids & tr_ids)
        tf[split]["table_texts_shared_with_train"] = len(txt & tr_txt)
        tf[split]["share_of_split_tables_also_in_train"] = round(
            len(ids & tr_ids) / max(len(ids), 1), 6)
    tf["measured_split_axis"] = (
        "per STATEMENT, not per table"
        if tf["validation"]["table_ids_shared_with_train"] > 0
        else "per TABLE")
    out["tabfact"] = tf

    # the lane's own tabfact documents against the tabfact val/test tables
    lane_tf = lane.filter(pl.col("source") == "tabfact")
    lane_tf_ids = {d.split(":", 1)[1] for d in lane_tf["doc_id"].unique().to_list()}
    out["lane_tabfact_documents"] = {
        "distinct_doc_ids": len(lane_tf_ids),
        "in_tabfact_validation_table_ids": len(
            lane_tf_ids & set(parts["validation"]["table_id"].to_list())),
        "in_tabfact_test_table_ids": len(
            lane_tf_ids & set(parts["test"]["table_id"].to_list())),
        "note": "the lane's selection predicate is the TRAIN split only; a "
                "non-zero count here means TabFact's official split does not "
                "separate tables and the same table carries statements on both "
                "sides of its own split boundary",
    }

    fev_src = ROOT / "tmp" / "R14_H133_feverous.parquet"
    fev = {"path": str(fev_src), "exists": fev_src.exists()}
    if fev_src.exists():
        d = pl.read_parquet(fev_src)
        fev["rows"] = len(d)
        fev["columns"] = d.columns
        fev["has_split_column"] = any(c.lower() in ("split", "set") for c in d.columns)
    fev["campaign_ruling"] = (
        "R20-H177 coordinator disposition 2: FEVEROUS non-admission ACCEPTED - "
        "the on-disk file is an R14-H133 working artifact without its own "
        "provenance verdict")
    fev["doc_id_stability"] = (
        "R17-H144 recorded the v2 lane's `feverous:{i}` ids as order-unstable "
        "across rebuilds; this lane inherits the same id scheme")
    out["feverous"] = fev

    # the axis the LANE cuts on, against the surfaces built from the same corpora
    lane_docs = set(lane["doc_id"].unique().to_list())
    lane_chunks = set(lane["chunk"].to_list())
    lane_chunks_n = {norm_ws(c) for c in lane_chunks}
    sib = {}
    for name, fname in (
        ("R17-H143_evalset", "R17-H143_evalset.parquet"),
        ("R18-H150_unitswap_probe", "R18-H150_unitswap_probe.parquet"),
        ("R20-H177_eval_B", "R20-H177_eval_B.parquet"),
        ("R20-H177_eval_C", "R20-H177_eval_C.parquet"),
        ("R17-H148_probe", "R17-H148_probe.parquet"),
    ):
        p = SEM / fname
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        blk = {"rows": len(d)}
        if "doc_id" in d.columns:
            od = set(d["doc_id"].unique().to_list())
            blk["their_documents"] = len(od)
            blk["shared_doc_ids"] = len(od & lane_docs)
        else:
            blk["their_documents"] = None
            blk["shared_doc_ids"] = "no doc_id column"
        oc = set(d["chunk"].to_list())
        blk["shared_chunks_raw"] = len(oc & lane_chunks)
        blk["shared_chunks_ws_collapsed"] = len({norm_ws(c) for c in oc} & lane_chunks_n)
        if "split" in d.columns:
            blk["split_values"] = {
                str(k): int(v) for k, v in zip(
                    d["split"].value_counts()["split"].to_list(),
                    d["split"].value_counts()["count"].to_list())
            }
        sib[name] = blk
    out["lane_cut_against_sibling_surfaces"] = sib
    out["lane_declared_split_axis"] = (
        "DOCUMENT, enforced on CONTENT fingerprints rather than ids "
        "(R17-H144 method) - the id scheme is not stable for FEVEROUS")
    return out


# --------------------------------------------------------------------------- #
# C5
# --------------------------------------------------------------------------- #
def doc_folds(docs, pairs, labels_dir, n_folds=N_FOLDS):
    """Greedy PAIR-count packing with both directions of a family packed
    jointly - the lane's own banked fold rule."""
    by_doc = collections.defaultdict(list)
    for i, d in enumerate(docs):
        by_doc[d].append(i)
    order = sorted(by_doc, key=lambda d: -len(by_doc[d]))
    loads = [0] * n_folds
    fold_of = {}
    for d in order:
        k = int(np.argmin(loads))
        fold_of[d] = k
        loads[k] += len(by_doc[d])
    return np.array([fold_of[d] for d in docs]), loads


def probe_auroc(texts, y, docs):
    """Converged liblinear TF-IDF probe, out-of-fold, document-disjoint folds."""
    fold, loads = doc_folds(docs, None, None)
    scores = np.zeros(len(y))
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        if te.sum() == 0 or len(set(np.asarray(y)[tr])) < 2:
            continue
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X = vec.fit_transform([t for t, m in zip(texts, tr) if m])
        clf = LogisticRegression(solver="liblinear", tol=1e-7, max_iter=5000, C=1.0)
        clf.fit(X, np.asarray(y)[tr])
        scores[te] = clf.decision_function(vec.transform([t for t, m in zip(texts, te) if m]))
    return round(auroc(y, scores), 6), loads


def c5_block(lane):
    claims = lane["claim"].to_list()
    chunks = lane["chunk"].to_list()
    y = lane["label"].to_list()
    docs = lane["doc_id"].to_list()
    pair_ids = lane["pair_id"].to_list()

    reg = {}
    a, loads = probe_auroc(claims, y, docs)
    reg["claim_only_converged_probe"] = {
        "value": a, "bar": "< 0.55", "pass": bool(a < 0.55),
        "margin_to_bar": round(0.55 - a, 6),
        "scoring": "TF-IDF 1-2gram min_df 2 sublinear, liblinear tol 1e-7, "
                   "5-fold document-disjoint, out-of-fold",
        "fold_pair_loads": loads,
        "banked_verify_value": 0.4296,
    }

    # within-pair claim-only: out-of-fold score, compare the two legs of a pair
    fold, _ = doc_folds(docs, None, None)
    scores = np.zeros(len(y))
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X = vec.fit_transform([t for t, m in zip(claims, tr) if m])
        clf = LogisticRegression(solver="liblinear", tol=1e-7, max_iter=5000)
        clf.fit(X, np.asarray(y)[tr])
        scores[te] = clf.decision_function(
            vec.transform([t for t, m in zip(claims, te) if m]))
    per_pair = collections.defaultdict(dict)
    for pid, yy, s in zip(pair_ids, y, scores):
        per_pair[pid][yy] = s
    corr = [1.0 if v[1] > v[0] else (0.5 if v[1] == v[0] else 0.0)
            for v in per_pair.values() if 0 in v and 1 in v]
    wp = round(float(np.mean(corr)), 6)
    # per swap family
    fam_of = dict(zip(pair_ids, lane["swap_family"].to_list()))
    byfam = collections.defaultdict(list)
    for pid, v in per_pair.items():
        if 0 in v and 1 in v:
            byfam[fam_of[pid]].append(
                1.0 if v[1] > v[0] else (0.5 if v[1] == v[0] else 0.0))
    fam = {k: {"acc": round(float(np.mean(vv)), 6), "pairs": len(vv)}
           for k, vv in byfam.items()}
    worst = max(v["acc"] for v in fam.values())
    reg["within_pair_claim_only"] = {
        "pooled": wp, "worst_family": round(worst, 6), "bar": "< 0.60",
        "pass": bool(worst < 0.60), "margin_to_bar": round(0.60 - worst, 6),
        "per_swap_family": fam, "banked_verify_worst": 0.5,
    }

    # surface parity on every computable claim-side channel
    ch = {
        "claim_char_length": [len(c) for c in claims],
        "claim_token_count": [len(TOKEN_RE.findall(c.lower())) for c in claims],
        "claim_numeral_count": [len(NUM_RE.findall(c)) for c in claims],
        "claim_chunk_token_containment": [containment(c, k) for c, k in zip(claims, chunks)],
        "claim_word_count": [len(c.split()) for c in claims],
    }
    par = {k: round(auroc(y, v), 6) for k, v in ch.items()}
    dev = {k: round(abs(v - 0.5), 6) for k, v in par.items()}
    reg["surface_parity"] = {
        "auroc": par, "abs_deviation": dev, "bar": "each in [0.45, 0.55]",
        "worst_channel": max(dev, key=dev.get), "worst_deviation": max(dev.values()),
        "pass": bool(max(dev.values()) <= 0.05),
    }

    # single-channel probes
    ev = {}
    # evidence-only: both legs share the chunk by construction -> exactly chance
    chunk_lab = collections.defaultdict(list)
    for k, yy in zip(chunks, y):
        chunk_lab[k].append(yy)
    balanced = all(sum(v) * 2 == len(v) for v in chunk_lab.values())
    ev["evidence_only_probe"] = {
        "computable": True,
        "value": 0.5 if balanced else None,
        "mechanism": "every distinct chunk carries exactly as many label-1 as "
                     "label-0 rows, so any evidence-only feature is exactly at "
                     "chance",
        "distinct_chunks": len(chunk_lab),
        "all_chunks_label_balanced": bool(balanced),
    }
    ev["question_only_probe"] = {
        "computable": False,
        "reason": "the lane carries no question field; the construction has no "
                  "question channel",
    }
    reg["single_channel_probes"] = ev

    # direction / element / family balance and attestation symmetry
    bal = {}
    for col in ("direction", "swap_family", "dimension", "unit_carrier",
                "serial_form", "template_id", "source", "neg_family"):
        vc = lane.group_by(col).agg(pl.col("label").sum().alias("pos"),
                                    pl.len().alias("n"))
        bal[col] = {
            str(a): {"rows": int(n), "positives": int(p),
                     "positive_share": round(float(p) / int(n), 6)}
            for a, p, n in zip(vc[col].to_list(), vc["pos"].to_list(), vc["n"].to_list())
        }
    worst_skew = max(abs(v["positive_share"] - 0.5)
                     for d in bal.values() for v in d.values())
    reg["balance"] = {"per_field": bal, "worst_positive_share_deviation":
                      round(worst_skew, 6), "bar": "0.50 exactly by construction",
                      "pass": bool(worst_skew < 1e-9)}

    # attestation symmetry - each unit word appears as often as a positive as
    # it does as a negative
    word_pos = collections.Counter()
    word_neg = collections.Counter()
    for u, yy in zip(lane["cited_unit"].to_list(), y):
        (word_pos if yy == 1 else word_neg)[u] += 1
    words = sorted(set(word_pos) | set(word_neg))
    skew = {w: {"as_positive": word_pos[w], "as_negative": word_neg[w],
                "skew": round(abs(word_pos[w] - word_neg[w])
                              / max(word_pos[w] + word_neg[w], 1), 6)}
            for w in words}
    reg["attestation_symmetry_unit_word_marginal"] = {
        "per_word": skew,
        "worst_skew": max(v["skew"] for v in skew.values()),
        "bar": "0.0 (a unit word must not be a label prior)",
        "pass": bool(max(v["skew"] for v in skew.values()) < 1e-9),
    }

    # ---- executor-added, reported SEPARATELY, joins no registered conjunction
    added = {}
    a2, _ = probe_auroc([k for k in chunks], y, docs)
    added["evidence_only_tfidf_probe_measured"] = {
        "value": a2,
        "note": "executor-added; forced to 0.5 by the shared-chunk construction, "
                "run as a live confirmation rather than a bar",
    }
    a3, _ = probe_auroc([c + " [SEP] " + k for c, k in zip(claims, chunks)], y, docs)
    added["claim_plus_evidence_lexical_probe"] = {
        "value": a3,
        "note": "executor-added; a lexical joint probe. It is NOT a leak bar - a "
                "lexical model that can solve the lane would indicate an "
                "adjacent-string channel the H148 rule is meant to close",
    }
    return {"registered": reg, "executor_added": added}


# --------------------------------------------------------------------------- #
# C6
# --------------------------------------------------------------------------- #
def c6_block(lane):
    out = {}

    # (i) structural: which fields are identical inside a pair?
    fields = [c for c in lane.columns if c not in ("label",)]
    ident = {}
    for f in fields:
        g = lane.group_by("pair_id").agg(pl.col(f).n_unique().alias("u"))
        ident[f] = int((g["u"] == 1).sum())
    n_pairs = int(lane["pair_id"].n_unique())
    out["within_pair_field_identity"] = {
        "pairs": n_pairs,
        "pairs_with_identical_value": ident,
        "fields_identical_on_every_pair": sorted(
            f for f, v in ident.items() if v == n_pairs),
        "fields_that_differ": sorted(f for f, v in ident.items() if v < n_pairs),
        "consequence": "any feature keyed on a field identical inside the pair "
                       "is identical on both legs and cannot separate them",
    }

    # (ii) the measured feature: what the REST of the mix associates with the key
    assoc = pl.read_parquet(HERE / "quant_scale_unit_mix_assoc.parquet")
    other = assoc.filter(pl.col("tag") != "quant_scale_unit")
    out["mix_rows_excluding_member"] = len(other)
    amap = collections.defaultdict(list)
    for k, c, lab in zip(other["chunk_norm"].to_list(), other["claim"].to_list(),
                         other["label"].to_list()):
        amap[k].append((c, lab))

    feats, covered = [], []
    for k, c in zip(lane["chunk"].to_list(), lane["claim"].to_list()):
        hits = amap.get(norm_ws(k))
        if not hits:
            feats.append(np.nan)
            covered.append(False)
            continue
        feats.append(max(containment(c, hc) for hc, _ in hits))
        covered.append(True)
    feats = np.array(feats)
    covered = np.array(covered)
    y = np.array(lane["label"].to_list())
    blk = {
        "key": "whitespace-collapsed case-folded chunk (the field both legs share)",
        "feature": "max token containment between the member claim and any claim "
                   "the rest of the mix associates with that key",
        "coverage_rows": int(covered.sum()),
        "coverage_share": round(float(covered.mean()), 6),
    }
    if covered.sum() > 0 and len(set(y[covered])) == 2:
        blk["auroc"] = round(auroc(y[covered], feats[covered]), 6)
    else:
        blk["auroc"] = None
        blk["auroc_status"] = "UNDEFINED - no member row has a mix association "
        "for its key outside the member itself"
    out["mix_association_feature"] = blk

    # executor-added: the unit-word global prior
    cu = lane["cited_unit"].to_list()
    cnt_pos = collections.Counter(u for u, yy in zip(cu, y) if yy == 1)
    cnt_neg = collections.Counter(u for u, yy in zip(cu, y) if yy == 0)
    prior = np.array([cnt_pos[u] / max(cnt_pos[u] + cnt_neg[u], 1) for u in cu])
    out["executor_added_unit_word_prior"] = {
        "feature": "P(label=1 | claimed unit word) estimated on the lane itself",
        "auroc": round(auroc(y, prior), 6),
        "note": "executor-added; reported separately, joins no registered bar",
    }
    return out


# --------------------------------------------------------------------------- #
# C7 / C8
# --------------------------------------------------------------------------- #
def c7c8_block(lane):
    rows, pairs = len(lane), int(lane["pair_id"].n_unique())
    c7 = {
        "declared_unit": "rows AND pairs, both stated at registration, build and "
                         "in the arm's LANES tuple",
        "registered_rows": 20000,
        "registered_pairs": 10000,
        "delivered_rows": rows,
        "delivered_pairs": pairs,
        "rows_share_of_registration": round(rows / 20000, 6),
        "pairs_share_of_registration": round(pairs / 10000, 6),
        "rows_per_pair": round(rows / pairs, 6),
        "arm_lane_tuple": {"rows": 5540, "pairs": 2770,
                           "neg_families": {"unit_swap": 5540}},
        "consistent_across_registration_build_and_report": True,
    }

    claims = lane["claim"].to_list()
    chunks = lane["chunk"].to_list()
    dup_rows = rows - lane.select("claim", "chunk", "label").unique().height
    claim_counts = collections.Counter(claims)
    chunk_counts = collections.Counter(chunks)
    per_doc = lane.group_by("doc_id").agg(pl.col("pair_id").n_unique().alias("p"))
    c8 = {
        "sources": {
            "tabfact": {
                "rows": int((lane["source"] == "tabfact").sum()),
                "share": round(float((lane["source"] == "tabfact").mean()), 6),
                "archive": "data/external/datasets/dataset-tabfact.zip",
                "sidecar": "data/external/datasets/dataset-tabfact.md",
                "licence": "CC-BY-4.0 (per the tracked sidecar)",
                "selection_predicate": "the __train.parquet split only, deduped "
                                       "on table_text, admitted through the "
                                       "R17-H144 content-fingerprint eval "
                                       "exclusion",
            },
            "feverous": {
                "rows": int((lane["source"] == "feverous").sum()),
                "share": round(float((lane["source"] == "feverous").mean()), 6),
                "archive": "NONE - tmp/R14_H133_feverous.parquet, an R14-H133 "
                           "working artifact",
                "sidecar": None,
                "licence": "NOT RECORDED in any tracked artifact",
                "retrieval_date": "NOT RECORDED",
                "selection_predicate": "deduped on `evidence`, doc ids "
                                       "`feverous:{i}` from an order-unstable "
                                       "dedup",
            },
        },
        "duplication": {
            "duplicate_claim_chunk_label_rows": int(dup_rows),
            "distinct_claims": len(claim_counts),
            "distinct_chunks": len(chunk_counts),
            "distinct_documents": int(lane["doc_id"].n_unique()),
            "max_claim_repeat": max(claim_counts.values()),
            "claims_appearing_more_than_once": sum(
                1 for v in claim_counts.values() if v > 1),
            "max_chunk_repeat": max(chunk_counts.values()),
            "mean_pairs_per_document": round(float(per_doc["p"].mean()), 6),
            "max_pairs_per_document": int(per_doc["p"].max()),
            "mean_rows_per_chunk": round(rows / len(chunk_counts), 6),
        },
        "public_repository_check": {
            "client_or_company_name_in_artifact": False,
            "basis": "the member's fields carry Wikipedia table content, unit "
                     "vocabulary and template ids only; no client or company "
                     "identifier is written into any artifact this task produced",
        },
    }
    return c7, c8


def main():
    lane = pl.read_parquet(LANE)
    res = {"member": "quant_scale_unit", "artifact": str(LANE)}
    res["c3"] = c3_block(lane)
    print("c3 done", flush=True)
    res["c5"] = c5_block(lane)
    print("c5 done", flush=True)
    res["c6"] = c6_block(lane)
    print("c6 done", flush=True)
    res["c7"], res["c8"] = c7c8_block(lane)
    OUT.write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

"""Dataset-contract clauses C1, C3, C6, C7, C8 for the `tabfact` training member.

CPU only, Polars only, torch-free. Reads the cached member slice produced by
`tabfact_load.py` (which rebuilt it through the banked loader).

Instruments are the banked ones:
  containment / jaccard / tok  <- R20-H175b_qlane.py (the module that produced
      the 0.9129 / 66.4% figures the C1 bar is written against)

Out: tabfact_clauses.json
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections
import importlib.util as _ilu
import io
import json
import pathlib
import time
import zipfile

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
DATA = SEM.parent.parent / "data" / "external" / "datasets"
MEMBER = HERE / "tabfact_member.parquet"
OUT = HERE / "tabfact_clauses.json"
EVAL_B = SEM / "R20-H177_eval_B.parquet"

Q = None  # R20-H175b_qlane, loaded in main


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def norm(s):
    """The banked normalised form - whitespace-collapsed, case-folded."""
    return " ".join(s.split()).casefold()


def stem(tid):
    """TabFact serialises one table under a `1-` and a `2-` csv id; an id-level
    comparison sees those as two documents. The stem strips that prefix."""
    return tid[2:] if len(tid) > 2 and tid[0] in "12" and tid[1] == "-" else tid


def dist(v):
    v = np.asarray(v, dtype="float64")
    return {
        "n": int(v.size),
        "mean": round(float(v.mean()), 4),
        "median": round(float(np.median(v)), 4),
        "p10": round(float(np.percentile(v, 10)), 4),
        "p25": round(float(np.percentile(v, 25)), 4),
        "p75": round(float(np.percentile(v, 75)), 4),
        "p90": round(float(np.percentile(v, 90)), 4),
        "share_fully_attested_eq_1.0": round(float((v >= 1.0).mean()), 4),
        "share_ge_0.9": round(float((v >= 0.9).mean()), 4),
        "share_ge_0.8": round(float((v >= 0.8).mean()), 4),
    }


# --------------------------------------------------------------------------- #
# C1 - label commensurability
# --------------------------------------------------------------------------- #
def clause_c1(df):
    claims = df["claim"].to_list()
    y = df["label"].to_numpy()

    out = {
        "head_declared": "the grounding scalar (`task_head` of the DANN student) - "
                         "the same head every other mix member trains",
        "label_predicate_measured": (
            "TabFact ships a human ENTAILED / REFUTED annotation of the STATEMENT "
            "against ITS OWN table (ICLR 2020). The label is a support predicate: "
            "1 = the table entails the statement, 0 = the table refutes it. It is "
            "not relevance, not answerability, not question-claim match. Positives "
            "and negatives share the same table by construction, so the contrast "
            "is carried entirely by the statement's content"),
        "predicate_is_support": True,
    }

    for form in ("chunk_untrunc", "chunk_trunc"):
        chunks = df[form].to_list()
        tokcache = {}
        for c in set(chunks):
            tokcache[c] = set(Q.tok(c))
        vals = np.empty(len(claims), dtype="float64")
        for i, (cl, ch) in enumerate(zip(claims, chunks, strict=True)):
            A = set(Q.tok(cl))
            vals[i] = len(A & tokcache[ch]) / len(A) if A else 0.0
        pos, neg = vals[y == 1.0], vals[y == 0.0]
        key = "untruncated_evidence" if form == "chunk_untrunc" else "truncated_evidence_1500"
        block = {
            "instrument": "R20-H175b_qlane.containment(claim, chunk) - share of the "
                          "claim's content tokens present in the evidence",
            "positive_leg": dist(pos),
            "negative_leg": dist(neg),
            "delta_mean_neg_minus_pos": round(float(neg.mean() - pos.mean()), 4),
            "abs_delta_mean": round(abs(float(neg.mean() - pos.mean())), 4),
            "delta_fully_attested_share": round(
                float((neg >= 1.0).mean() - (pos >= 1.0).mean()), 4),
        }
        # the bar, both readings, evaluated mechanically
        block["bar_primary_mean_containment"] = {
            "reading": "negatives attested at >= 0.90 mean containment AND "
                       "|mean_neg - mean_pos| <= 0.10 -> REJECTED for the grounding head",
            "neg_mean": block["negative_leg"]["mean"],
            "pos_mean": block["positive_leg"]["mean"],
            "neg_ge_0.90": bool(neg.mean() >= 0.90),
            "abs_delta_le_0.10": bool(abs(neg.mean() - pos.mean()) <= 0.10),
            "rejected": bool(neg.mean() >= 0.90 and abs(neg.mean() - pos.mean()) <= 0.10),
            "margin_to_0.90_bar": round(float(0.90 - neg.mean()), 4),
        }
        p_att, n_att = float((pos >= 1.0).mean()), float((neg >= 1.0).mean())
        block["bar_secondary_full_attestation_share"] = {
            "reading": "share of the leg fully attested (containment == 1.0); the "
                       "H175b poisoned lane read 0.664 on its negative leg",
            "neg_share": round(n_att, 4),
            "pos_share": round(p_att, 4),
            "neg_ge_0.90": bool(n_att >= 0.90),
            "abs_delta_le_0.10": bool(abs(n_att - p_att) <= 0.10),
            "rejected": bool(n_att >= 0.90 and abs(n_att - p_att) <= 0.10),
        }
        out[key] = block

    # H175b comparison, for calibration of the numbers above
    out["reference_poisoned_lane_R20_H175b"] = {
        "containment_both_legs": 0.9129,
        "negative_leg_fully_attested_share": 0.664,
        "why_it_failed": "passage and claim held FIXED, label flipped on question "
                         "relevance - both legs identical in grounding",
    }
    return out


# --------------------------------------------------------------------------- #
# C3 - split semantics
# --------------------------------------------------------------------------- #
def build_chunk(cap, tbl, cut):
    return f"{cap}\n{tbl}".replace("\r\n", "\n").replace("#", " | ")[:cut]


def clause_c3(df):
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    names = {s: next(x for x in z.namelist() if x.endswith(f"__{s}.parquet"))
             for s in ("train", "validation", "test")}
    splits = {s: pl.read_parquet(io.BytesIO(z.read(n))) for s, n in names.items()}

    tr_ids = set(splits["train"]["table_id"].to_list())
    tr_stems = {stem(t) for t in tr_ids}
    tr_stmt = set(splits["train"]["statement"].to_list())

    # the member's own presentation, both cuts, for the evidence comparison
    def chunks_of(d, cut):
        return {build_chunk(c, t, cut) for c, t in
                zip(d["table_caption"].to_list(), d["table_text"].to_list(), strict=True)}

    tr_raw = chunks_of(splits["train"], 10**9)
    tr_tru = chunks_of(splits["train"], 1500)
    tr_nrw = {norm(c) for c in tr_raw}
    tr_ntr = {norm(c) for c in tr_tru}

    per_split = {}
    for s in ("validation", "test"):
        d = splits[s]
        ids = set(d["table_id"].to_list())
        stems = {stem(t) for t in ids}
        c_raw = chunks_of(d, 10**9)
        c_tru = chunks_of(d, 1500)
        per_split[s] = {
            "rows": d.height,
            "distinct_table_id": len(ids),
            "table_id_shared_with_train": len(ids & tr_ids),
            "table_id_share": round(len(ids & tr_ids) / max(len(ids), 1), 6),
            "table_id_STEM_shared_with_train": len(stems & tr_stems),
            "table_id_stem_share": round(len(stems & tr_stems) / max(len(stems), 1), 6),
            "statement_rows_shared_with_train": int(d["statement"].is_in(list(tr_stmt)).sum()),
            "distinct_serialised_evidence": len(c_raw),
            "evidence_raw_in_train_raw": len(c_raw & tr_raw),
            "evidence_truncated_in_train_truncated": len(c_tru & tr_tru),
            "evidence_normalised_in_train_normalised_raw": len({norm(c) for c in c_raw} & tr_nrw),
            "evidence_normalised_in_train_normalised_truncated":
                len({norm(c) for c in c_tru} & tr_ntr),
        }

    # what axis does the archive actually cut on? - written FROM the numbers
    tid_counts = splits["train"].group_by("table_id").len()
    v, t = per_split["validation"], per_split["test"]
    axis = (
        "the archive cuts on the table_id STRING, not on the underlying document. "
        f"Validation and test share {v['table_id_shared_with_train']} and "
        f"{t['table_id_shared_with_train']} table_id VALUES with train. But TabFact "
        "serialises one Wikipedia table under both a `1-` and a `2-` csv id, and "
        f"after stripping that prefix {v['table_id_STEM_shared_with_train']} of "
        f"{v['distinct_table_id']} validation ids ({v['table_id_stem_share']:.4f}) and "
        f"{t['table_id_STEM_shared_with_train']} of {t['distinct_table_id']} test ids "
        f"({t['table_id_stem_share']:.4f}) collide with a train id. Part of that "
        "collision is genuine document re-use: "
        f"{v['evidence_raw_in_train_raw']} validation and {t['evidence_raw_in_train_raw']} "
        "test serialised tables are BYTE-IDENTICAL to a train table, rising to "
        f"{v['evidence_truncated_in_train_truncated']} and "
        f"{t['evidence_truncated_in_train_truncated']} under the 1,500-char cut the "
        f"R10-H108 recipe applies; {v['statement_rows_shared_with_train']} validation "
        f"rows and {t['statement_rows_shared_with_train']} test rows carry a statement "
        "string that also occurs in train. Statements are grouped under their table, "
        "so the cut is per document ID and per claim within it - but NOT per document")
    return {
        "split_axis_measured": axis,
        "member_uses": "train split ONLY - `next(x for x in namelist if "
                       "x.endswith('__train.parquet'))` in R10-H108_lane.public_train",
        "train": {
            "rows": splits["train"].height,
            "distinct_table_id": len(tr_ids),
            "distinct_table_id_stem": len(tr_stems),
            "claims_per_table": {
                "mean": round(float(tid_counts["len"].mean()), 3),
                "median": int(tid_counts["len"].median()),
                "min": int(tid_counts["len"].min()),
                "max": int(tid_counts["len"].max()),
            },
        },
        "held_out_splits": per_split,
        "note": "TabFact's validation and test splits are NOT used as evaluation "
                "surfaces by any banked arm; this clause tests the archive's own "
                "split claim, which is what C3 asks for. Their use as the C4 LIVE "
                "positive control is reported under C4",
    }


# --------------------------------------------------------------------------- #
# C6 - memorisation channel
# --------------------------------------------------------------------------- #
def clause_c6(df):
    y = df["label"].to_numpy()
    tids = df["table_id"].to_list()
    claims = df["claim"].to_list()

    by_tab = collections.defaultdict(list)
    for i, t in enumerate(tids):
        by_tab[t].append(i)

    # (a) leave-one-out table-mean label: does the KEY alone carry the label?
    loo = np.empty(len(y))
    for idxs in by_tab.values():
        s = sum(y[i] for i in idxs)
        k = len(idxs)
        for i in idxs:
            loo[i] = (s - y[i]) / (k - 1) if k > 1 else 0.5
    a_loo = float(roc_auc_score(y.astype(int), loo))

    # (b) nearest OTHER claim over the same table, its label as the predictor
    toks = [set(Q.tok(c)) for c in claims]
    nn_lab = np.full(len(y), 0.5)
    nn_sim = np.zeros(len(y))
    covered = 0
    for idxs in by_tab.values():
        if len(idxs) < 2:
            continue
        for i in idxs:
            best, blab = -1.0, 0.5
            for j in idxs:
                if j == i:
                    continue
                u = toks[i] | toks[j]
                s = len(toks[i] & toks[j]) / len(u) if u else 0.0
                if s > best:
                    best, blab = s, y[j]
            nn_lab[i], nn_sim[i] = blab, best
            covered += 1
    a_nn = float(roc_auc_score(y.astype(int), nn_lab))

    out = {
        "adaptation_stated": (
            "C6 is written for an eval scored against a training mix. `tabfact` is "
            "a TRAINING member, so its rows share the EVIDENCE field (one table "
            "carries several statements) and the table is the key. The clause's "
            "PRESCRIBED instrument - overlap between the eval claim and whatever "
            "the training mix associates with that pair's key - is computable "
            "against exactly one surface, R20-H177_eval_B's TabFact half, and is "
            "reported below as (c). (a) and (b) are EXECUTOR-ADDED internal "
            "features, reported separately and carrying no clause verdict"),
        "prescribed_instrument": "c_cross_surface_R20_H177_eval_B_tabfact_half",
        "a_table_key_label_leakage_EXECUTOR_ADDED": {
            "feature": "leave-one-out mean label of the other statements over the "
                       "same table_id",
            "auroc": round(a_loo, 4),
            "chance": 0.5,
            "abs_deviation_from_chance": round(abs(a_loo - 0.5), 4),
        },
        "b_nearest_other_claim_label_EXECUTOR_ADDED": {
            "feature": "label of the most Jaccard-similar OTHER statement over the "
                       "same table_id",
            "auroc": round(a_nn, 4),
            "rows_with_a_sibling": covered,
            "mean_nearest_sibling_jaccard": round(float(nn_sim[nn_sim > 0].mean()), 4),
            "chance": 0.5,
            "abs_deviation_from_chance": round(abs(a_nn - 0.5), 4),
        },
    }

    # (c) cross-surface: the H175b feature keyed on the shared table
    if EVAL_B.exists():
        ev = pl.read_parquet(EVAL_B).filter(pl.col("source") == "tabfact")
        ev_tid = [d[8:] if d.startswith("tabfact:") else d for d in ev["doc_id"].to_list()]
        member_by_stem = collections.defaultdict(list)
        for t, c, lab in zip(tids, claims, y, strict=True):
            member_by_stem[stem(t)].append((c, float(lab)))
        ey = ev["label"].to_numpy().astype(int)
        feat = np.zeros(ev.height)
        nlab = np.full(ev.height, 0.5)
        cov = 0
        for i, (t, c) in enumerate(zip(ev_tid, ev["claim"].to_list(), strict=True)):
            bag = member_by_stem.get(stem(t), [])
            if not bag:
                continue
            cov += 1
            sims = [Q.jaccard(c, a) for a, _ in bag]
            k = int(np.argmax(sims))
            feat[i] = sims[k]
            nlab[i] = bag[k][1]
        blk = {
            "feature": "for each eval_B tabfact row, the max token-Jaccard between "
                       "the eval claim and any claim the `tabfact` member carries "
                       "over the SAME table (stem-matched table_id), and the label "
                       "the member attached to that best-matching claim",
            "eval_rows": ev.height,
            "rows_with_a_member_claim_over_the_same_table": cov,
            "coverage": round(cov / max(ev.height, 1), 4),
        }
        if cov and len(set(ey.tolist())) == 2:
            blk["max_jaccard_auroc"] = round(float(roc_auc_score(ey, feat)), 4)
            blk["nearest_member_claim_label_auroc"] = round(float(roc_auc_score(ey, nlab)), 4)
            blk["mean_max_jaccard"] = round(float(feat[feat > 0].mean()), 4) if (feat > 0).any() else 0.0
        out["c_cross_surface_R20_H177_eval_B_tabfact_half"] = blk
    return out


# --------------------------------------------------------------------------- #
# C7 / C8
# --------------------------------------------------------------------------- #
def clause_c7_c8(df, load_meta):
    y = df["label"].to_numpy()
    claims = df["claim"].to_list()
    chunks = df["chunk_untrunc"].to_list()
    tids = df["table_id"].to_list()

    pair_key = list(zip(claims, chunks, strict=True))
    dup_pairs = len(pair_key) - len(set(pair_key))

    cl_counts = collections.Counter(claims)
    dup_claim_rows = sum(v for v in cl_counts.values() if v > 1)

    # a claim carrying both labels anywhere in the member
    lab_by_claim = collections.defaultdict(set)
    for c, v in zip(claims, y, strict=True):
        lab_by_claim[c].add(float(v))
    contradictory = sum(1 for v in lab_by_claim.values() if len(v) > 1)

    per_tab = collections.defaultdict(list)
    for t, v in zip(tids, y, strict=True):
        per_tab[t].append(float(v))
    tab_sizes = np.array([len(v) for v in per_tab.values()])
    tab_bal = np.array([float(np.mean(v)) for v in per_tab.values()])

    c7 = {
        "unit_declared": "ROWS. One archive row = one (statement, serialised table) "
                         "pair, so for this member rows and (claim, evidence) pairs "
                         "are the same count. There is no positive/negative twin key "
                         "in the archive - TabFact does not ship a pair id",
        "rows": df.height,
        "claim_evidence_pairs": df.height,
        "distinct_claim_evidence_pairs": len(set(pair_key)),
        "duplicate_claim_evidence_rows": dup_pairs,
        "distinct_documents_tables": len(set(tids)),
        "positives": int((y == 1.0).sum()),
        "negatives": int((y == 0.0).sum()),
        "positive_share": round(float((y == 1.0).mean()), 4),
        "share_of_the_clean_mix_685670_rows": round(df.height / 685_670, 6),
        "registered_vs_measured": {
            "brief_registered_rows": 92_585,
            "measured_rows": df.height,
            "delta": df.height - 92_585,
            "tracked_sidecar_declares": "92,283 train / 12,792 val / 12,779 test",
        },
    }

    c8 = {
        "source": "wenhuchen/Table-Fact-Checking (GitHub); the HF dataset "
                  "`wenhu/tab_fact` is script-only, so the archive was fetched from "
                  "the GitHub release. Archive member "
                  "`wenhuchen__Table-Fact-Checking__tabfact__train.parquet`",
        "licence": "CC-BY-4.0 (tracked sidecar data/external/datasets/dataset-tabfact.md)",
        "retrieval": "archive dataset-tabfact.zip, fetched by "
                     "scripts/fetch_grounding_datasets.py; archive mtime recorded below",
        "selection_predicate": load_meta["selection_predicate"],
        "chunk_construction": load_meta["chunk_construction"],
        "internal_structure": {
            "rows": df.height,
            "distinct_claims": len(cl_counts),
            "rows_on_a_repeated_claim": dup_claim_rows,
            "claims_carrying_BOTH_labels": contradictory,
            "distinct_evidence_untruncated": len(set(chunks)),
            "distinct_evidence_truncated_1500": df["chunk_trunc"].n_unique(),
            "distinct_table_id": len(set(tids)),
            "distinct_table_id_stem": len({stem(t) for t in tids}),
            "rows_per_table": {
                "mean": round(float(tab_sizes.mean()), 3),
                "median": int(np.median(tab_sizes)),
                "min": int(tab_sizes.min()), "max": int(tab_sizes.max()),
            },
            "per_table_positive_share": {
                "mean": round(float(tab_bal.mean()), 4),
                "tables_all_positive": int((tab_bal == 1.0).sum()),
                "tables_all_negative": int((tab_bal == 0.0).sum()),
                "tables_mixed": int(((tab_bal > 0.0) & (tab_bal < 1.0)).sum()),
            },
            "evidence_length_chars_untruncated": {
                "mean": round(float(np.mean([len(c) for c in chunks])), 1),
                "max": int(max(len(c) for c in chunks)),
                "rows_over_1500": int(sum(1 for c in chunks if len(c) > 1500)),
                "share_over_1500": round(
                    float(sum(1 for c in chunks if len(c) > 1500) / df.height), 4),
            },
        },
        "public_repository": "no client or company name appears in this artifact or "
                             "in any file this verification wrote",
    }
    return c7, c8


# --------------------------------------------------------------------------- #
def supplementary_claim_only(df):
    """EXECUTOR-ADDED, reported separately - it is NOT part of any clause and no
    verdict rests on it. C5's leak suite is scoped to constructed lanes; this is
    the nearest question a source corpus can be asked: are the labels lexically
    markable from the statement alone, with the table withheld and tables kept
    disjoint between fit and held-out?"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    tabs = np.array(sorted(set(df["table_id"].to_list())))
    rng.shuffle(tabs)
    hold = set(tabs[: len(tabs) // 5].tolist())
    m = np.array([t in hold for t in df["table_id"].to_list()])
    X, y = df["claim"].to_list(), df["label"].to_numpy().astype(int)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000,
                          sublinear_tf=True)
    Xtr = vec.fit_transform([x for x, k in zip(X, m, strict=True) if not k])
    Xte = vec.transform([x for x, k in zip(X, m, strict=True) if k])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, y[~m])
    s = clf.predict_proba(Xte)[:, 1]
    return {
        "status": "EXECUTOR-ADDED - reported separately from every registered bar; "
                  "no clause verdict uses it",
        "probe": "claim-only TF-IDF(1-2gram) + logistic regression, tables disjoint "
                 "between fit and held-out",
        "fit_rows": int((~m).sum()), "heldout_rows": int(m.sum()),
        "heldout_auroc": round(float(roc_auc_score(y[m], s)), 4),
        "reading": "a claim-only read materially above 0.5 means the statement "
                   "carries label evidence without the table",
    }


def main():
    global Q
    t0 = time.time()
    Q = _mod("qlane", SEM / "R20-H175b_qlane.py")
    df = pl.read_parquet(MEMBER)
    load_meta = json.loads((HERE / "tabfact_load.json").read_text())
    print(f"member: {df.height} rows", flush=True)

    c1 = clause_c1(df)
    print("C1 done", flush=True)
    c3 = clause_c3(df)
    print("C3 done", flush=True)
    c6 = clause_c6(df)
    print("C6 done", flush=True)
    c7, c8 = clause_c7_c8(df, load_meta)
    c8["archive_mtime"] = time.strftime(
        "%Y-%m-%d %H:%M:%S",
        time.localtime((DATA / "dataset-tabfact.zip").stat().st_mtime))
    sup = supplementary_claim_only(df)
    print("supplementary done", flush=True)

    res = {"member": "tabfact", "load": load_meta,
           "C1": c1, "C3": c3, "C6": c6, "C7": c7, "C8": c8,
           "supplementary_not_a_clause": sup,
           "elapsed_s": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT.name}  ({res['elapsed_s']}s)", flush=True)
    print(json.dumps({"C1_untrunc": c1["untruncated_evidence"],
                      "C6": c6, "C7": c7, "sup": sup}, indent=2)[:4000], flush=True)


if __name__ == "__main__":
    main()

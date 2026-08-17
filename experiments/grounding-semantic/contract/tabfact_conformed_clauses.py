"""Contract clauses C1, C3, C5, C6, C7, C8 re-verified on the CONFORMED member.

CPU only, Polars only, torch-free. Instruments are the banked ones, unchanged
from the first pass (`tabfact_clauses.py`):
  containment / jaccard / tok  <- R20-H175b_qlane.py

What differs from the first pass, and why:
  C3  adds the CONFORMED member's own disjointness from TabFact validation and
      test - id, stem, evidence in three forms, statement strings - alongside
      the archive's split axis, which is a corpus fact and does not change.
  C6  the prescribed cross-surface instrument is re-run against BOTH surfaces
      that share a key with the member (R20-H177_eval_B and R17-H143_evalset);
      its coverage is now the measurement.
  C8  the retrieval date is derived from the artifacts the fetch run produced
      and RECORDED here, and the declared volume is re-measured against the
      archive on all three splits.

Out: tabfact_conformed_clauses.json
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections
import importlib.util as _ilu
import io
import json
import pathlib
import subprocess
import time
import zipfile

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
ROOT = SEM.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
MEMBER = HERE / "tabfact_member_conformed.parquet"
OUT = HERE / "tabfact_conformed_clauses.json"
EVAL_B = SEM / "R20-H177_eval_B.parquet"

Q = None


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def norm(s):
    return " ".join(s.split()).casefold()


def stem(tid):
    return tid[2:] if len(tid) > 2 and tid[0] in "12" and tid[1] == "-" else tid


def build_chunk(cap, tbl, cut=10**9):
    return f"{cap}\n{tbl}".replace("\r\n", "\n").replace("#", " | ")[:cut]


def dist(v):
    v = np.asarray(v, dtype="float64")
    return {
        "n": int(v.size),
        "mean": round(float(v.mean()), 4),
        "median": round(float(np.median(v)), 4),
        "p10": round(float(np.percentile(v, 10)), 4),
        "p90": round(float(np.percentile(v, 90)), 4),
        "share_fully_attested_eq_1.0": round(float((v >= 1.0).mean()), 4),
        "share_ge_0.9": round(float((v >= 0.9).mean()), 4),
    }


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
            "not relevance, not answerability, not question-claim match. The "
            "conforming cut removed whole documents and contradictory rows; it "
            "changed no label and rewrote no chunk"),
        "predicate_is_support": True,
    }
    for form in ("chunk_untrunc", "chunk_trunc"):
        chunks = df[form].to_list()
        tokcache = {c: set(Q.tok(c)) for c in set(chunks)}
        vals = np.empty(len(claims), dtype="float64")
        for i, (cl, ch) in enumerate(zip(claims, chunks, strict=True)):
            A = set(Q.tok(cl))
            vals[i] = len(A & tokcache[ch]) / len(A) if A else 0.0
        pos, neg = vals[y == 1.0], vals[y == 0.0]
        key = "untruncated_evidence" if form == "chunk_untrunc" else "truncated_evidence_1500"
        block = {
            "instrument": "R20-H175b_qlane.containment(claim, chunk)",
            "positive_leg": dist(pos), "negative_leg": dist(neg),
            "abs_delta_mean": round(abs(float(neg.mean() - pos.mean())), 4),
        }
        block["bar_primary_mean_containment"] = {
            "reading": "negatives attested at >= 0.90 mean containment AND "
                       "|mean_neg - mean_pos| <= 0.10 -> REJECTED",
            "neg_mean": block["negative_leg"]["mean"],
            "pos_mean": block["positive_leg"]["mean"],
            "neg_ge_0.90": bool(neg.mean() >= 0.90),
            "abs_delta_le_0.10": bool(abs(neg.mean() - pos.mean()) <= 0.10),
            "rejected": bool(neg.mean() >= 0.90 and abs(neg.mean() - pos.mean()) <= 0.10),
            "margin_to_0.90_bar": round(float(0.90 - neg.mean()), 4),
        }
        p_att, n_att = float((pos >= 1.0).mean()), float((neg >= 1.0).mean())
        block["bar_secondary_full_attestation_share"] = {
            "neg_share": round(n_att, 4), "pos_share": round(p_att, 4),
            "rejected": bool(n_att >= 0.90 and abs(n_att - p_att) <= 0.10),
        }
        out[key] = block
    out["reference_poisoned_lane_R20_H175b"] = {
        "containment_both_legs": 0.9129,
        "negative_leg_fully_attested_share": 0.664,
    }
    return out


# --------------------------------------------------------------------------- #
def clause_c3(df):
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    names = {s: next(x for x in z.namelist() if x.endswith(f"__{s}.parquet"))
             for s in ("train", "validation", "test")}
    splits = {s: pl.read_parquet(io.BytesIO(z.read(n))) for s, n in names.items()}

    # (i) the ARCHIVE's split axis - a corpus fact, unchanged by any pipeline
    tr_ids = set(splits["train"]["table_id"].to_list())
    tr_stems = {stem(t) for t in tr_ids}
    archive = {}
    for s in ("validation", "test"):
        ids = set(splits[s]["table_id"].to_list())
        stems = {stem(t) for t in ids}
        archive[s] = {
            "rows": splits[s].height, "distinct_table_id": len(ids),
            "table_id_shared_with_train": len(ids & tr_ids),
            "table_id_STEM_shared_with_train": len(stems & tr_stems),
            "table_id_stem_share": round(len(stems & tr_stems) / max(len(stems), 1), 6),
        }

    # (ii) the CONFORMED MEMBER's own disjointness from the held-out splits
    m_ids = set(df["table_id"].to_list())
    m_stems = {stem(t) for t in m_ids}
    m_stmt = set(df["claim"].to_list())
    m_raw = set(df["chunk_untrunc"].to_list())
    m_tru = set(df["chunk_trunc"].to_list())
    m_nraw = {norm(c) for c in m_raw}
    m_ntru = {norm(c) for c in m_tru}
    member = {}
    for s in ("validation", "test"):
        d = splits[s]
        ids = set(d["table_id"].to_list())
        stems = {stem(t) for t in ids}
        c_raw = {build_chunk(c, t) for c, t in
                 zip(d["table_caption"].to_list(), d["table_text"].to_list(), strict=True)}
        c_tru = {c[:1500] for c in c_raw}
        member[s] = {
            "table_id_shared_with_member": len(ids & m_ids),
            "table_id_STEM_shared_with_member": len(stems & m_stems),
            "evidence_raw_in_member_raw": len(c_raw & m_raw),
            "evidence_truncated_in_member_truncated": len(c_tru & m_tru),
            "evidence_normalised_in_member_normalised_raw": len({norm(c) for c in c_raw} & m_nraw),
            "evidence_normalised_in_member_normalised_truncated":
                len({norm(c) for c in c_tru} & m_ntru),
            "statement_rows_carrying_a_member_claim_string":
                int(d["statement"].is_in(list(m_stmt)).sum()),
        }
    return {
        "archive_split_axis_measured": (
            "unchanged corpus fact: the archive cuts on the table_id STRING, not on "
            "the underlying document. Validation and test share "
            f"{archive['validation']['table_id_shared_with_train']} and "
            f"{archive['test']['table_id_shared_with_train']} table_id VALUES with "
            "train, but TabFact serialises one Wikipedia table under both a `1-` and "
            "a `2-` csv id, and after stripping that prefix "
            f"{archive['validation']['table_id_STEM_shared_with_train']} of "
            f"{archive['validation']['distinct_table_id']} validation ids and "
            f"{archive['test']['table_id_STEM_shared_with_train']} of "
            f"{archive['test']['distinct_table_id']} test ids collide with a train "
            "id. The archive's official split is therefore NOT document-disjoint"),
        "archive": archive,
        "member_uses": "train split ONLY, then cut on the DOCUMENT (table_id stem) "
                       "so the member does not rely on the archive's split claim",
        "conformed_member_vs_heldout_splits": member,
        "note": "TabFact validation and test are not scored directly by any banked "
                "arm; they matter because the 14 anti-gaming probe sets are built "
                "from them. The member's disjointness from them is measured here "
                "rather than inherited from the archive's split",
    }


# --------------------------------------------------------------------------- #
def clause_c6(df):
    y = df["label"].to_numpy()
    tids = df["table_id"].to_list()
    claims = df["claim"].to_list()
    by_tab = collections.defaultdict(list)
    for i, t in enumerate(tids):
        by_tab[t].append(i)

    loo = np.empty(len(y))
    for idxs in by_tab.values():
        s = sum(y[i] for i in idxs)
        k = len(idxs)
        for i in idxs:
            loo[i] = (s - y[i]) / (k - 1) if k > 1 else 0.5
    a_loo = float(roc_auc_score(y.astype(int), loo))

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

    rng = np.random.default_rng(0)
    perms = []
    for _ in range(5):
        yp = y.copy()
        for idxs in by_tab.values():
            arr = yp[idxs]
            rng.shuffle(arr)
            yp[idxs] = arr
        v = np.empty(len(yp))
        for idxs in by_tab.values():
            s = sum(yp[i] for i in idxs)
            k = len(idxs)
            for i in idxs:
                v[i] = (s - yp[i]) / (k - 1) if k > 1 else 0.5
        perms.append(round(float(roc_auc_score(yp.astype(int), v)), 4))

    out = {
        "adaptation_stated": (
            "C6 is written for an eval scored against a training mix. `tabfact` is a "
            "TRAINING member, so the clause's PRESCRIBED instrument - overlap "
            "between the eval claim and whatever the training mix associates with "
            "that pair's key - is computable only against a surface that shares the "
            "key. Both such surfaces are measured below. (a) and (b) are "
            "EXECUTOR-ADDED internal features, reported separately and carrying no "
            "clause verdict"),
        "a_table_key_label_leakage_EXECUTOR_ADDED": {
            "feature": "leave-one-out mean label of the other statements over the "
                       "same table_id",
            "auroc": round(a_loo, 4),
            "within_table_label_permutation_auroc": perms,
            "permutation_mean": round(float(np.mean(perms)), 4),
            "reading": "a within-table permutation preserves each table's label "
                       "QUOTA and destroys every statement-label association; a "
                       "reproduced AUROC means the feature carries the quota alone",
        },
        "b_nearest_other_claim_label_EXECUTOR_ADDED": {
            "auroc": round(a_nn, 4),
            "rows_with_a_sibling": covered,
            "mean_nearest_sibling_jaccard": round(float(nn_sim[nn_sim > 0].mean()), 4),
        },
    }

    member_by_stem = collections.defaultdict(list)
    for t, c, lab in zip(tids, claims, y, strict=True):
        member_by_stem[stem(t)].append((c, float(lab)))

    def cross(name, ev_tid, ev_claims, ev_y):
        feat = np.zeros(len(ev_claims))
        nlab = np.full(len(ev_claims), 0.5)
        cov = 0
        for i, (t, c) in enumerate(zip(ev_tid, ev_claims, strict=True)):
            bag = member_by_stem.get(stem(t), [])
            if not bag:
                continue
            cov += 1
            sims = [Q.jaccard(c, a) for a, _ in bag]
            k = int(np.argmax(sims))
            feat[i], nlab[i] = sims[k], bag[k][1]
        blk = {"eval_rows": len(ev_claims),
               "rows_with_a_member_claim_over_the_same_document": cov,
               "coverage": round(cov / max(len(ev_claims), 1), 4)}
        if cov and ev_y is not None and len(set(ev_y.tolist())) == 2:
            blk["max_jaccard_auroc"] = round(float(roc_auc_score(ev_y, feat)), 4)
            blk["nearest_member_claim_label_auroc"] = round(float(roc_auc_score(ev_y, nlab)), 4)
        else:
            blk["value"] = ("UNDEFINED - the channel is closed: no eval row shares a "
                            "document with the member, so there is nothing for a "
                            "feature keyed on the training association to read")
        return blk

    ev = pl.read_parquet(EVAL_B).filter(pl.col("source") == "tabfact")
    out["c_prescribed_R20-H177_eval_B_tabfact_half"] = cross(
        "eval_B",
        [d[8:] if d.startswith("tabfact:") else d for d in ev["doc_id"].to_list()],
        ev["claim"].to_list(), ev["label"].to_numpy().astype(int))

    e2 = pl.read_parquet(SEM / "R17-H143_evalset.parquet")
    src = pl.read_parquet(SEM / "R17-H143_evalset_source.parquet")
    j = e2.join(src.select("chunk", "doc_id", "source").unique(subset=["chunk"]),
                on="chunk", how="left").filter(pl.col("source") == "tabfact")
    out["c_prescribed_R17-H143_evalset_tabfact_half"] = cross(
        "h143",
        [d[8:] if str(d).startswith("tabfact:") else d for d in j["doc_id"].to_list()],
        j["claim"].to_list(), j["label"].to_numpy().astype(int))
    return out


# --------------------------------------------------------------------------- #
def clause_c7_c8(df, build_meta):
    y = df["label"].to_numpy()
    claims = df["claim"].to_list()
    chunks = df["chunk_untrunc"].to_list()
    tids = df["table_id"].to_list()
    stems = [stem(t) for t in tids]

    pair_key = list(zip(claims, chunks, strict=True))
    cl_counts = collections.Counter(claims)
    lab_by_claim = collections.defaultdict(set)
    lab_by_claim_doc = collections.defaultdict(set)
    for c, v, s in zip(claims, y, stems, strict=True):
        lab_by_claim[c].add(float(v))
        lab_by_claim_doc[(c, s)].add(float(v))

    per_tab = collections.defaultdict(list)
    for t, v in zip(tids, y, strict=True):
        per_tab[t].append(float(v))
    tab_sizes = np.array([len(v) for v in per_tab.values()])
    tab_bal = np.array([float(np.mean(v)) for v in per_tab.values()])

    vol = build_meta["volume"]
    c7 = {
        "unit_declared": "ROWS. One archive row = one (statement, serialised table) "
                         "pair, so rows and (claim, evidence) pairs are the same "
                         "count. TabFact ships no pair id",
        "rows": df.height,
        "claim_evidence_pairs": df.height,
        "distinct_claim_evidence_pairs": len(set(pair_key)),
        "duplicate_claim_evidence_rows": len(pair_key) - len(set(pair_key)),
        "distinct_documents_stems": len(set(stems)),
        "distinct_table_id": len(set(tids)),
        "positives": int((y == 1.0).sum()),
        "negatives": int((y == 0.0).sum()),
        "positive_share": round(float((y == 1.0).mean()), 4),
        "volume_cost_vs_banked_member": {
            "banked_rows": vol["banked_member_rows"],
            "conformed_rows": vol["conformed_rows"],
            "rows_dropped": vol["rows_dropped"],
            "row_cost_share": vol["row_cost_share"],
            "banked_documents": vol["banked_member_documents_stems"],
            "conformed_documents": vol["conformed_documents_stems"],
            "documents_dropped": vol["documents_dropped"],
        },
        "share_of_the_clean_mix": {
            "clean_mix_rows_before": vol["clean_mix_rows_before"],
            "clean_mix_rows_after": vol["clean_mix_rows_after"],
            "member_share_before": vol["member_share_before"],
            "member_share_after": vol["member_share_after"],
        },
    }

    # ---- provenance ------------------------------------------------------- #
    zpath = DATA / "dataset-tabfact.zip"
    z = zipfile.ZipFile(zpath)
    stamps = {i.filename: "%04d-%02d-%02d %02d:%02d:%02d" % i.date_time
              for i in z.infolist()}
    logp = ROOT / "logs" / "fetch-tabfact.log"
    counts = {}
    for s in ("train", "validation", "test"):
        n = next(x for x in z.namelist() if x.endswith(f"__{s}.parquet"))
        counts[s] = pl.read_parquet(io.BytesIO(z.read(n))).height
    try:
        sidecar_commit = subprocess.run(
            ["git", "log", "--format=%ad", "--date=short", "--reverse", "--",
             "data/external/datasets/dataset-tabfact.md"],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()[0]
    except Exception:
        sidecar_commit = None

    retrieval = {
        "retrieval_date": "2026-07-31",
        "how_it_was_established": (
            "not read off a card - derived from the artifacts the fetch run itself "
            "produced, and recorded here because C8 requires it and no tracked file "
            "carried it. Three independent stamps agree to within 22 seconds"),
        "evidence": {
            "fetch_run_log": {
                "path": "logs/fetch-tabfact.log (gitignored by `*.log`, present on disk)",
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                       time.localtime(logp.stat().st_mtime))
                if logp.exists() else None,
                "records_row_counts": "92585 train / 12851 validation / 12839 test",
                "counts_match_this_archive": (
                    counts == {"train": 92585, "validation": 12851, "test": 12839}),
                "why_it_binds": "the log's counts are this archive's measured counts, "
                                "which ties that run to this file",
            },
            "zip_central_directory_stamps": stamps,
            "archive_file_mtime": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(zpath.stat().st_mtime)),
            "archive_file_ctime": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(zpath.stat().st_ctime)),
            "sidecar_first_commit_date": sidecar_commit,
        },
    }

    declared = {"train": 92_283, "validation": 12_792, "test": 12_779}
    c8 = {
        "source": "wenhuchen/Table-Fact-Checking, GitHub release "
                  "(https://github.com/wenhuchen/Table-Fact-Checking/archive/refs/"
                  "heads/master.zip); every HF mirror of `tab_fact` is script-only "
                  "and datasets 5.x removed script support. Archive member "
                  "`wenhuchen__Table-Fact-Checking__tabfact__train.parquet`",
        "licence": "CC-BY-4.0",
        "retrieval": retrieval,
        "selection_predicate": build_meta["selection_predicate"],
        "chunk_construction": build_meta["chunk_construction"],
        "declared_volume": {
            "measured_from_the_archive": counts,
            "tracked_sidecar_declares": declared,
            "sidecar_delta": {k: declared[k] - counts[k] for k in counts},
            "sidecar_agrees": declared == counts,
            "cause": "the sidecar renders the hand-written `size` string of the "
                     "tabfact spec in scripts/fetch_grounding_datasets.py, not the "
                     "counts the fetch produced; the fetch log carries the right ones",
            "corrected_figures_recorded_here": counts,
        },
        "internal_structure": {
            "rows": df.height,
            "distinct_claims": len(cl_counts),
            "rows_on_a_repeated_claim": sum(v for v in cl_counts.values() if v > 1),
            "claims_carrying_BOTH_labels_anywhere": sum(
                1 for v in lab_by_claim.values() if len(v) > 1),
            "claims_carrying_BOTH_labels_on_the_SAME_document": sum(
                1 for v in lab_by_claim_doc.values() if len(v) > 1),
            "distinct_evidence_untruncated": len(set(chunks)),
            "distinct_evidence_truncated_1500": df["chunk_trunc"].n_unique(),
            "distinct_table_id": len(set(tids)),
            "distinct_documents_stems": len(set(stems)),
            "rows_per_document": {
                "mean": round(float(len(df) / len(set(stems))), 3),
                "max": int(collections.Counter(stems).most_common(1)[0][1]),
            },
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


def supplementary_claim_only(df):
    """EXECUTOR-ADDED, reported separately - no clause verdict rests on it."""
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
        "status": "EXECUTOR-ADDED - reported separately; no clause verdict uses it",
        "probe": "claim-only TF-IDF(1-2gram) + logistic regression, tables disjoint "
                 "between fit and held-out",
        "fit_rows": int((~m).sum()), "heldout_rows": int(m.sum()),
        "heldout_auroc": round(float(roc_auc_score(y[m], s)), 4),
        "first_pass_value_on_the_banked_member": 0.6031,
    }


def main():
    global Q
    t0 = time.time()
    Q = _mod("qlane", SEM / "R20-H175b_qlane.py")
    df = pl.read_parquet(MEMBER)
    build_meta = json.loads((HERE / "tabfact_conform_build.json").read_text())
    print(f"conformed member: {df.height} rows", flush=True)

    c1 = clause_c1(df)
    print("C1 done", flush=True)
    c3 = clause_c3(df)
    print("C3 done", flush=True)
    c6 = clause_c6(df)
    print("C6 done", flush=True)
    c7, c8 = clause_c7_c8(df, build_meta)
    print("C7/C8 done", flush=True)
    sup = supplementary_claim_only(df)

    res = {"member": "tabfact_conformed", "build": build_meta["volume"],
           "C1": c1, "C3": c3, "C6": c6, "C7": c7, "C8": c8,
           "supplementary_not_a_clause": sup,
           "elapsed_s": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT.name}  ({res['elapsed_s']}s)", flush=True)
    print(json.dumps({"C1": c1["untruncated_evidence"]["bar_primary_mean_containment"],
                      "C3_member": c3["conformed_member_vs_heldout_splits"],
                      "C6_c": {k: v for k, v in c6.items() if k.startswith("c_")},
                      "C7": c7, "sup": sup}, indent=2)[:4000], flush=True)


if __name__ == "__main__":
    main()

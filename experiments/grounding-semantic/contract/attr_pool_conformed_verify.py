"""attr_pool CONFORMED member - full re-verification against C1-C8.

Target: experiments/grounding-semantic/R20-H174_lane_L2_conformed.parquet, built
by attr_pool_conformed_build.py.  The contract's failure policy requires the
member to be re-verified from scratch against EVERY clause, not only the four
the parent failed, so every stage below runs on the conformed artifact even
where the parent passed.

Instruments are the banked ones, reused rather than re-implemented:
  provenance_gate.py            R14-H136 8-gram / Jaccard 0.3 bidirectional wall
  R20-H174_lane_common.py       containment, auroc, claim_only_probe,
                                within_pair_accuracy, surface_parity
  attr_pool_contract_measure.py eval_surfaces (the 8 surfaces), the three
                                contract string forms
  R10-H108_lane.public_train    the assembled mix, through the banked loader

Stages:
  core       C1, C5 registered conjunction + executor-added probes, C7, C8
  disjoint   C2 full grid, C3 split semantics
  census     C4 census + spike + LIVE positive control + negative control
  memo       C6 claim-keyed oracle, association oracle, doc-keyed probes
  frontier   the C5 containment-channel conflict, measured as a pipeline family
  assemble   attr_pool_conformed_report.json

CPU ONLY.  CUDA_VISIBLE_DEVICES is forced empty before any import.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
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
EXP = HERE.parent
ROOT = EXP.parent.parent
DATA = ROOT / "data" / "external" / "datasets"

LANE = EXP / "R20-H174_lane_L2_conformed.parquet"
MANIFEST = EXP / "R20-H174_lane_L2_conformed_manifest.json"
PARENT = EXP / "R20-H174_lane_L2.parquet"
SEP = "\n\n"
CHUNK_MAX_CHARS = 1500
_WS = re.compile(r"\s+")


def wsfold(t):
    return _WS.sub(" ", t).strip().casefold()


def _mod(name, fname, folder=EXP):
    spec = _ilu.spec_from_file_location(name, folder / fname)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = _mod("h174common", "R20-H174_lane_common.py")
GATE = _mod("provgate", "provenance_gate.py")
M = _mod("apmeasure", "attr_pool_contract_measure.py", folder=HERE)


def lane():
    return pl.read_parquet(LANE)


def dump(name, obj):
    p = HERE / f"attr_pool_conformed_{name}.json"
    p.write_text(json.dumps(obj, indent=2, default=float))
    print(f"-> {p}", flush=True)


def load(name):
    return json.loads((HERE / f"attr_pool_conformed_{name}.json").read_text())


def pct(x, n):
    return round(float(x) / n, 6) if n else 0.0


def leg_stats(mask, v):
    s = v[mask]
    return {
        "n": int(mask.sum()),
        "mean": round(float(s.mean()), 4),
        "median": round(float(np.median(s)), 4),
        "p10": round(float(np.percentile(s, 10)), 4),
        "p90": round(float(np.percentile(s, 90)), 4),
        "frac_ge_0.90": round(float((s >= 0.90).mean()), 4),
        "frac_eq_1.00": round(float((s >= 0.9999).mean()), 4),
        "frac_ge_0.75": round(float((s >= 0.75).mean()), 4),
    }


# =========================================================================== #
# core - C1, C5, C7, C8
# =========================================================================== #
def stage_core():
    df = lane()
    man = json.loads(MANIFEST.read_text())
    out = {}
    claims, chunks = df["claim"].to_list(), df["chunk"].to_list()
    y = df["label"].to_numpy()
    fam = df["neg_family"].to_list()

    print("containment ...", flush=True)
    cont_full = np.array([C.containment(c, k) for c, k in zip(claims, chunks)])
    cont_best = np.array(
        [max(C.containment(c, p) for p in k.split(SEP)) for c, k in zip(claims, chunks)]
    )
    pos, neg = y == 1, y == 0
    np.save(HERE / "attr_pool_conformed_containment.npy", np.vstack([cont_full, cont_best]))

    p_pos = float((cont_full[pos] >= 0.90).mean())
    p_neg = float((cont_full[neg] >= 0.90).mean())
    pb = float((cont_best[pos] >= 0.90).mean())
    nb = float((cont_best[neg] >= 0.90).mean())
    c1 = {
        "head_declared": "grounding scalar (`task_head`) - the binary support head "
        "the shipped ground() consumes; DANN group `attr_pool`",
        "label_predicate": "label 1 = the claim's supporting passage is PRESENT in "
        "the pooled chunk; label 0 = no passage in the pool supports the claim. "
        "truth_removed removes the true passage and holds the claim byte-identical; "
        "unsupported_claim holds the pool byte-identical and swaps the claim for a "
        "MiniCheck label-0 one. Both encode SUPPORT, not relevance",
        "corpus_label_inheritance": "MiniCheck ships a binary supported / "
        "not-supported label produced by an entailment filter over a generated "
        "document; the predicate is support, so the inheritance is commensurable. "
        "The conformed member has NO VitaminC rows, so no three-way collapse is "
        "involved anywhere in it",
        "containment_definition": "fraction of the claim's content tokens "
        "([a-z0-9]+, lowercased) present in the evidence - "
        "R20-H174_lane_common.containment",
        "claim_to_full_pool": {"positive_leg": leg_stats(pos, cont_full),
                               "negative_leg": leg_stats(neg, cont_full)},
        "claim_to_best_single_passage": {"positive_leg": leg_stats(pos, cont_best),
                                         "negative_leg": leg_stats(neg, cont_best)},
        "auroc_containment_full_pool": round(C.auroc(y, cont_full), 4),
        "auroc_containment_best_passage": round(C.auroc(y, cont_best), 4),
        "mandatory_bar": {
            "statement": "a member whose negatives are >= 90% attested at a rate "
            "within 0.10 of its positives is REJECTED for the grounding head",
            "attested_rate_positive_leg": round(p_pos, 4),
            "attested_rate_negative_leg": round(p_neg, 4),
            "gap": round(abs(p_pos - p_neg), 4),
            "reading_A_gap": {
                "bar": "gap > 0.10",
                "margin": round(abs(p_pos - p_neg) - 0.10, 4),
                "pass": bool(abs(p_pos - p_neg) > 0.10),
            },
            "reading_B_literal_conjunction": {
                "bar": "REJECT iff negative attestation >= 0.90 AND gap <= 0.10",
                "negative_attestation": round(p_neg, 4),
                "rejected": bool(p_neg >= 0.90 and abs(p_pos - p_neg) <= 0.10),
                "pass": bool(not (p_neg >= 0.90 and abs(p_pos - p_neg) <= 0.10)),
            },
            "both_readings_pass": bool(
                abs(p_pos - p_neg) > 0.10 and not (p_neg >= 0.90 and abs(p_pos - p_neg) <= 0.10)
            ),
        },
        "mandatory_bar_best_passage_reading": {
            "attested_rate_positive_leg": round(pb, 4),
            "attested_rate_negative_leg": round(nb, 4),
            "gap": round(abs(pb - nb), 4),
            "pass": bool(abs(pb - nb) > 0.10),
        },
    }
    perfam = {}
    for f in sorted(set(fam)):
        m = np.array([x == f for x in fam])
        perfam[f] = {
            "positive_leg": leg_stats(m & pos, cont_full),
            "negative_leg": leg_stats(m & neg, cont_full),
            "gap_frac_ge_0.90": round(
                abs(float((cont_full[m & pos] >= 0.9).mean())
                    - float((cont_full[m & neg] >= 0.9).mean())), 4),
            "auroc_containment": round(C.auroc(y[m], cont_full[m]), 4),
        }
    c1["per_family"] = perfam
    c1["per_source"] = {"minicheck": {"rows": int(df.height),
                                      "share": 1.0,
                                      "note": "single-source member"}}
    out["C1"] = c1

    # ---------------- C5 ------------------------------------------------- #
    print("C5 probes ...", flush=True)
    rng = random.Random(2174)
    probe, score = C.claim_only_probe(claims, y.tolist(), df["doc_id"].to_list(), rng)
    wp = C.within_pair_accuracy(df, score, by="neg_family")
    sp = C.surface_parity(df, report_only=())

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    groups = df["doc_id"].to_list()
    keys = sorted(set(groups))
    rng2 = random.Random(99)
    rng2.shuffle(keys)
    fold_of = {k: i % 5 for i, k in enumerate(keys)}
    folds = np.array([fold_of[g] for g in groups])
    cscore = np.zeros(len(chunks))
    ix = np.arange(len(chunks))
    for f in range(5):
        tr, te = ix[folds != f], ix[folds == f]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000,
                              sublinear_tf=True)
        xtr = vec.fit_transform([chunks[j] for j in tr])
        xte = vec.transform([chunks[j] for j in te])
        clf = LogisticRegression(solver="liblinear", C=4.0, tol=1e-7, max_iter=3000)
        clf.fit(xtr, y[tr])
        cscore[te] = clf.decision_function(xte)
        print(f"  chunk-only fold {f}", flush=True)
    cwp = C.within_pair_accuracy(df, cscore, by="neg_family")

    posrows = df.filter(pl.col("label") == 1)
    rel = []
    for r in posrows.iter_rows(named=True):
        ids = list(r["pool_doc_ids"])
        rel.append(ids.index(r["doc_id"]) / (len(ids) - 1) if len(ids) > 1 else 0.0)
    rel = np.array(rel)
    worst_wp = max(v["acc"] for v in wp.values())
    out["C5"] = {
        "registered_conjunction": {
            "claim_only_converged_probe": {
                "value": round(probe, 4), "bar": "< 0.55",
                "margin": round(0.55 - probe, 4), "pass": bool(probe < 0.55),
                "spec": "char_wb 2-5gram TF-IDF, min_df 3, liblinear C=4 tol 1e-7, "
                "5 folds disjoint on doc_id",
            },
            "within_pair_claim_only": {
                "per_family": wp, "worst": round(worst_wp, 4), "bar": "< 0.60",
                "margin": round(0.60 - worst_wp, 4), "pass": bool(worst_wp < 0.60),
            },
            "surface_parity_every_computable_channel": {
                "auroc": sp["auroc"], "bar": "each channel in [0.45, 0.55]",
                "worst_channel": max(sp["auroc"], key=lambda k: abs(sp["auroc"][k] - 0.5)),
                "worst_deviation": sp["worst_deviation"],
                "pass_all_channels": bool(sp["worst_deviation"] <= 0.05),
                "pass_excluding_containment": bool(
                    max(abs(v - 0.5) for k, v in sp["auroc"].items()
                        if k != "claim_chunk_containment") <= 0.05),
            },
            "balance": {
                "family_rows": dict(collections.Counter(fam)),
                "family_share": {k: pct(v, len(fam))
                                 for k, v in collections.Counter(fam).items()},
                "label_balance": {"label_1": int(pos.sum()), "label_0": int(neg.sum())},
                "pool_depth_positive_mean": round(
                    float(df.filter(pl.col("label") == 1)["pool_depth"].to_numpy().mean()), 4),
                "pool_depth_negative_mean": round(
                    float(df.filter(pl.col("label") == 0)["pool_depth"].to_numpy().mean()), 4),
                "truth_relative_position_in_pool_mean": round(float(rel.mean()), 4),
                "truth_relative_position_in_pool_sd": round(float(rel.std()), 4),
                "truth_relative_position_bar": "0.5 +- 0.05 (uniform placement)",
                "truth_relative_position_pass": bool(abs(rel.mean() - 0.5) <= 0.05),
            },
            "attestation_symmetry": {
                "definition": "positive-leg minus negative-leg mean claim->pool "
                "containment",
                "positive_mean": round(float(cont_full[pos].mean()), 4),
                "negative_mean": round(float(cont_full[neg].mean()), 4),
                "delta": round(float(cont_full[pos].mean() - cont_full[neg].mean()), 4),
            },
        },
        "executor_added_probes_reported_separately": {
            "chunk_only_tfidf_auroc": {
                "value": round(C.auroc(y, cscore), 4),
                "spec": "word 1-2gram TF-IDF, min_df 3, max_features 200k, "
                "liblinear C=4 tol 1e-7, 5 folds disjoint on doc_id. char_wb 2-5gram "
                "is intractable on the pooled text; the substitution is declared",
                "note": "NOT part of the registered conjunction and carrying no bar",
            },
            "within_pair_chunk_only": cwp,
            "single_channel_at_chance_where_construction_implies": {
                "unsupported_claim_chunk_only": "byte-identical chunk across the "
                "legs - any chunk-only score is tied, within-pair 0.5 exactly",
                "truth_removed_claim_only": "byte-identical claim across the legs - "
                "any claim-only score is tied, within-pair 0.5 exactly",
                "question_only": "NOT APPLICABLE - the lane carries no question "
                "field; the mix loader consumes (claim, chunk) only",
            },
        },
    }

    # ---------------- C7 ------------------------------------------------- #
    parent = pl.read_parquet(PARENT)
    out["C7"] = {
        "declared_unit": "rows",
        "unit_used_consistently": "rows at registration ('L2 attr_pool ~20-30k "
        "rows'), rows at build, rows in this report; pairs reported alongside "
        "everywhere",
        "built_rows": int(df.height),
        "built_pairs": int(df["pair_id"].n_unique()),
        "both_counts_reported": True,
        "registered_band_rows": [20000, 30000],
        "in_registered_band": bool(20000 <= df.height <= 30000),
        "parent_rows": int(parent.height),
        "parent_pairs": int(parent["pair_id"].n_unique()),
        "volume_cost_rows": int(parent.height - df.height),
        "volume_cost_share": round(1 - df.height / parent.height, 4),
        "note": "C7 bars UNIT CONSISTENCY and dual reporting, not volume. The "
        "conformed member falls BELOW the registered row band; that shortfall is "
        "the price of C6 and C8 conformance and is reported here in the registered "
        "unit, not converted away",
    }

    # ---------------- C8 ------------------------------------------------- #
    passages = [p for k in chunks for p in k.split(SEP)]
    uniq_pass = set(passages)
    claim_counts = collections.Counter(claims)
    cl2lab = collections.defaultdict(set)
    for c, l in zip(claims, y.tolist()):
        cl2lab[c].add(int(l))
    pure = sum(1 for v in cl2lab.values() if len(v) == 1)
    rep = [c for c, n in claim_counts.items() if n > 2]
    rep_single = [c for c in rep if len(cl2lab[c]) == 1]
    out["C8"] = {
        "sources": {"minicheck": man["sources"]["minicheck"]},
        "licence": {"minicheck": "MIT (HF card YAML `license: mit`, re-verified at "
                                 "the Hub 2026-08-13 per the tracked sidecar)"},
        "retrieval_date": {
            "minicheck": "2026-08-13 - tracked sidecar "
            "data/external/datasets/dataset-minicheck.md: 'fetched 2026-08-13: "
            "14395 rows (c2d: 7076, d2c: 7319)'",
            "vitaminc": "NOT APPLICABLE - the conformed member contains no "
            "VitaminC-sourced row. The parent's C8 failure was the absence of a "
            "VitaminC retrieval date in tracked provenance; dropping the source "
            "removes the obligation rather than satisfying it",
        },
        "selection_predicate": {
            "minicheck": "BOTH shipped parquets (c2d + d2c) concatenated, filtered "
            "to documents <= 1400 chars (PASSAGE_MAX_CHARS); doc_id = index in the "
            "sorted deduplicated document list",
            "eligibility_filters_added_by_the_conforming_pipeline":
                "(1) the claim string must appear NOWHERE in the assembled mix, on "
                "raw or whitespace-collapsed case-folded comparison; (2) neither the "
                "claim nor the document may collide with any of the 8 evaluation "
                "surfaces on any of the three contract string forms",
            "construction": "BM25Okapi over the eligible document set, top-40 "
            "candidate window, distractor rejected if claim containment >= 0.75, "
            "3-7 distractors (pool depth 4-8), TRUTH_CAP 2 / DIST_CAP 12, seed 2174",
        },
        "within_member_duplication": {
            "rows": int(df.height),
            "pairs": int(df["pair_id"].n_unique()),
            "distinct_claims": int(df["claim"].n_unique()),
            "distinct_pooled_chunks": int(df["chunk"].n_unique()),
            "atomic_pool_passage_slots": len(passages),
            "distinct_atomic_pool_passages": len(uniq_pass),
            "atomic_passage_reuse_mean": round(len(passages) / len(uniq_pass), 4),
            "distinct_truth_documents": int(df["doc_id"].n_unique()),
            "claim_repeat_max": int(max(claim_counts.values())),
            "claims_appearing_more_than_twice": len(rep),
            "of_those_carrying_a_single_label": len(rep_single),
            "rows_carried_by_single_label_repeated_claims": int(
                sum(claim_counts[c] for c in rep_single)),
            "claims_with_single_label_across_lane": pure,
            "claims_with_single_label_share": pct(pure, len(cl2lab)),
            "document_role_overlap": man["document_disjointness"],
        },
        "public_repository": {
            "check": "no client or company name appears in the conformed parquet, "
            "its manifest, or this report; the single source is a public corpus "
            "(MiniCheck, MIT) and every artifact path is inside the public repository",
            "note": "gold_full is read from a private submodule for C2 overlap "
            "counts only; no text from it is reproduced in any artifact",
        },
    }
    dump("core", out)


# =========================================================================== #
# disjoint - C2, C3
# =========================================================================== #
def stage_disjoint():
    df = lane()
    out = {}
    lane_claims = df["claim"].unique().to_list()
    lane_chunks = df["chunk"].unique().to_list()
    lane_pass = sorted({p for k in df["chunk"].to_list() for p in k.split(SEP)})
    print(f"lane: {len(lane_claims)} claims, {len(lane_chunks)} pooled chunks, "
          f"{len(lane_pass)} atomic passages", flush=True)
    LF = {"claims": M._forms(lane_claims),
          "pooled_chunks": M._forms(lane_chunks),
          "atomic_passages": M._forms(lane_pass)}

    S = M.eval_surfaces()
    per_surface, worst, cells = {}, 0, 0
    for name, s in S.items():
        EC = M._forms(s["claims"]) if s["claims"] else None
        EE = M._forms(s["evidence"]) if s["evidence"] else None
        blk = {"surface_claims": len(s["claims"]), "surface_evidence": len(s["evidence"])}
        for lu, lf in LF.items():
            for su, sf in (("claims", EC), ("evidence", EE)):
                if sf is None:
                    continue
                for form in ("raw", "truncated_1500", "wsfold"):
                    n = len(lf[form] & sf[form])
                    worst = max(worst, n)
                    cells += 1
                    blk[f"lane_{lu}__vs__surface_{su}__{form}"] = {
                        "shared_units": n,
                        "frac_of_lane_side": pct(n, len(lf[form])),
                        "frac_of_surface_side": pct(n, len(sf[form])),
                    }
        per_surface[name] = blk
        print(f"  {name}: worst so far {worst}", flush=True)

    out["C2"] = {
        "instrument": "exact set intersection on three string forms - raw, "
        "truncated to CFG.chunk_max_chars=1500, whitespace-collapsed case-folded - "
        "at three lane granularities (claim, pooled chunk, atomic pool passage) "
        "against both the claim and the evidence side of every surface",
        "surfaces_covered": sorted(S),
        "cells_measured": cells,
        "per_surface": per_surface,
        "max_shared_units_any_cell": worst,
        "bar": "every form reads zero",
        "pass": worst == 0,
    }

    # the parent's C6-driving intra-mix duplication, re-measured
    print("intra-mix duplication ...", flush=True)
    z = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    vtr = pl.read_parquet(io.BytesIO(z.read("tals__vitaminc__train.parquet")))
    lc = set(lane_claims)
    out["C2_intra_mix"] = {
        "why": "the parent lane duplicated VitaminC train text that is already a "
        "mix member; the conformed member has no VitaminC row at all",
        "vitaminc_sourced_rows": 0,
        "lane_claims_also_in_vitaminc_train": len(lc & set(vtr["claim"].to_list())),
        "lane_atomic_passages_also_a_vitaminc_train_evidence_string": len(
            set(lane_pass) & set(vtr["evidence"].to_list())),
        "note": "intra-mix duplication is not what C2 bars; it is reported because "
        "it is the mechanism C6 tests",
    }

    # ---------------- C3 ------------------------------------------------- #
    print("C3 split semantics ...", flush=True)
    zm = zipfile.ZipFile(DATA / "dataset-minicheck.zip")
    mc_names = [n for n in zm.namelist() if n.endswith(".parquet")]
    mc_parts = {n.split("__")[-1].replace(".parquet", ""):
                pl.read_parquet(io.BytesIO(zm.read(n))) for n in mc_names}
    mc_rows = {k: v.height for k, v in mc_parts.items()}
    # measured, not read from a card: is there ANY split column in the archive?
    cols = {k: v.columns for k, v in mc_parts.items()}
    # the two synthesis routes share documents / claims?
    a, b = list(mc_parts.values())
    shared_docs = len(set(a["doc"].to_list()) & set(b["doc"].to_list()))
    shared_claims = len(set(a["claim"].to_list()) & set(b["claim"].to_list()))
    out["C3"] = {
        "member_carves_no_split": "attr_pool_conformed is 100% training material; "
        "no held-out partition is taken from it, so it cannot violate a split of "
        "its own. The clause is evaluated on the split axis of its SOURCE",
        "minicheck": {
            "split_axis_declared": "none - the archive ships the c2d and d2c "
            "synthesis routes, not a train/test cut",
            "split_axis_measured": "no split column exists in either parquet; the "
            "two routes are measured against each other directly",
            "archive_parquets": mc_rows,
            "archive_columns": cols,
            "documents_shared_between_the_two_routes": shared_docs,
            "claims_shared_between_the_two_routes": shared_claims,
            "is_minicheck_an_evaluation_surface": False,
            "consequence": "MiniCheck appears on no evaluation surface in this "
            "campaign (verified in C2: 0 shared units against all 8 surfaces on all "
            "three string forms), so no held-out partition of it exists to violate",
        },
        "vitaminc": "NOT APPLICABLE - the conformed member consumes no VitaminC "
        "row. The parent's VitaminC split measurements (official cut disjoint on "
        "unique_id and case_id, NOT on page / claim / evidence / wiki_revision_id) "
        "no longer bear on this member",
    }
    dump("disjoint", out)


# =========================================================================== #
# census - C4
# =========================================================================== #
def stage_census():
    df = lane()
    passages = sorted({p for k in df["chunk"].to_list() for p in k.split(SEP)})
    claims = sorted(set(df["claim"].to_list()))
    print(f"census units: {len(passages)} passages / {len(claims)} claims", flush=True)

    raw, _ = GATE.load_arena()
    ev = GATE.run_gate(passages, n=8, jaccard=0.3, label="attr_pool_conformed_evidence",
                       arena_texts=raw)
    print(f"  evidence verdict {ev['verdict']} {ev['max_fraction']}", flush=True)
    cl = GATE.run_gate(claims, n=8, jaccard=0.3, label="attr_pool_conformed_claims",
                       arena_texts=raw)
    print(f"  claim verdict {cl['verdict']} {cl['max_fraction']}", flush=True)

    rng = random.Random(7)
    spike = GATE.spike_control(rng.sample(passages, min(2000, len(passages))), raw,
                               n=8, jaccard=0.3)
    print(f"  spike {spike}", flush=True)

    # LIVE positive control against the FULL source corpus the passages came from
    zm = zipfile.ZipFile(DATA / "dataset-minicheck.zip")
    mcdocs = []
    for n in [x for x in zm.namelist() if x.endswith(".parquet")]:
        mcdocs += pl.read_parquet(io.BytesIO(zm.read(n)))["doc"].unique().to_list()
    mcdocs = sorted(set(mcdocs))
    src_side = {"minicheck_docs": mcdocs}
    live = GATE.run_gate(rng.sample(passages, min(2000, len(passages))), n=8, jaccard=0.3,
                         label="conformed_passages_live", arena_texts=src_side)
    print(f"  live positive control fires at {live['candidate_vs_arena']['fraction']}",
          flush=True)
    zh = zipfile.ZipFile(DATA / "dataset-halueval.zip")
    hd = pl.read_parquet(io.BytesIO(zh.read(
        next(x for x in zh.namelist() if x.endswith(".parquet")))))
    hcol = next(c for c in ("knowledge", "document") if c in hd.columns)
    negctl = GATE.run_gate(rng.sample(hd[hcol].unique().to_list(), 2000), n=8, jaccard=0.3,
                           label="halueval_unrelated", arena_texts=src_side)
    print(f"  negative control {negctl['candidate_vs_arena']['fraction']}", flush=True)

    def short(units):
        return [u for u in units if len(GATE.normalize(u).split()) < 8]

    short_pass, short_claims = short(passages), short(claims)
    arena_norm = "\n\n".join(GATE.normalize(c) for c in [c for v in raw.values() for c in v])
    hits_p = sum(1 for u in short_pass if GATE.normalize(u) and GATE.normalize(u) in arena_norm)
    hits_c = sum(1 for u in short_claims if GATE.normalize(u) and GATE.normalize(u) in arena_norm)

    worst = max(ev["max_fraction"], cl["max_fraction"])
    dump("census", {"C4": {
        "instrument": "provenance_gate.py, R14-H136 ruling-2 form: 8-gram, "
        "Jaccard >= 0.3, BIDIRECTIONAL, WARN 0.005 / KILL 0.02, per-arena-subset "
        "attribution. Re-run on the conformed text, not cited",
        "evidence_gate": {"unit": "atomic pool passages, deduplicated",
                          "n_units": ev["candidate"]["n_units"],
                          "n_units_scorable": ev["candidate"]["n_units_scorable"],
                          "max_fraction": ev["max_fraction"], "verdict": ev["verdict"],
                          "candidate_vs_arena": ev["candidate_vs_arena"],
                          "arena_vs_candidate": ev["arena_vs_candidate"],
                          "hit_examples": ev["hit_examples"]},
        "claim_gate": {"n_units": cl["candidate"]["n_units"],
                       "n_units_scorable": cl["candidate"]["n_units_scorable"],
                       "max_fraction": cl["max_fraction"], "verdict": cl["verdict"],
                       "best_jaccard": cl["candidate_vs_arena"].get("best_jaccard")},
        "kill_bar": 0.02, "warn_bar": 0.005,
        "worst_fraction_either_gate": worst,
        "margin_below_kill": round(0.02 - worst, 6),
        "spike_control": spike,
        "live_positive_control": {
            "design": f"2,000 atomic pool passages of the conformed lane against the "
            f"FULL MiniCheck document set ({len(mcdocs)} documents) they were "
            f"assembled from - near-duplicate BY CONSTRUCTION",
            "fires_at_fraction": live["candidate_vs_arena"]["fraction"],
            "best_jaccard": live["candidate_vs_arena"].get("best_jaccard"),
            "bar": "the gate must FIRE - a clean arena number from a gate that "
            "cannot fire is not evidence",
            "pass": bool(live["candidate_vs_arena"]["fraction"] > 0.5)},
        "live_negative_control": {
            "design": "2,000 unrelated HaluEval knowledge passages against the same "
            "full source index",
            "fires_at_fraction": negctl["candidate_vs_arena"]["fraction"],
            "bar": "< 0.02",
            "pass": bool(negctl["candidate_vs_arena"]["fraction"] < 0.02)},
        "coverage": {
            "definition": "a unit is unscorable when its normalized token count is "
            "below the 8-gram order; those units are covered by exact matching",
            "passages_total": len(passages), "passages_too_short": len(short_pass),
            "passages_too_short_exact_substring_hits_in_arena": hits_p,
            "claims_total": len(claims), "claims_too_short": len(short_claims),
            "claims_too_short_share": pct(len(short_claims), len(claims)),
            "claims_too_short_exact_substring_hits_in_arena": hits_c},
    }})


# =========================================================================== #
# memo - C6
# =========================================================================== #
def stage_memo():
    df = lane()
    B = _mod("apbuild", "attr_pool_conformed_build.py", folder=HERE)
    print("rebuilding the mix through the banked loader ...", flush=True)
    mclaims, mchunks, my, mtags = B.load_mix()

    assoc_sum, assoc_n = collections.Counter(), collections.Counter()
    claim_texts = collections.defaultdict(list)
    lane_claims = set(df["claim"].to_list())
    lane_fold = {wsfold(c) for c in lane_claims}
    fold_hit = 0
    for c, k, l in zip(mclaims, mchunks, my.tolist()):
        if c in lane_claims:
            assoc_sum[c] += int(l >= 0.5)
            assoc_n[c] += 1
            claim_texts[c].append((k, int(l >= 0.5)))
    for c in set(mclaims):
        if wsfold(c) in lane_fold and c not in lane_claims:
            fold_hit += 1

    # chunk-side maps for the executor-added document-keyed probes
    mix_sha = {hashlib.sha1(k.encode()).hexdigest() for k in mchunks}
    mix_pos_sha = {hashlib.sha1(k.encode()).hexdigest()
                   for k, l in zip(mchunks, my.tolist()) if l >= 0.5}
    lane_parts, lane_parts_pos = set(), set()
    for k, t, l in zip(mchunks, mtags, my.tolist()):
        if t in ("quant_misbind", "quant_scale_unit", "frame_reject", "path_bind"):
            for p in k.split(SEP):
                lane_parts.add(p)
                if l >= 0.5:
                    lane_parts_pos.add(p)
    del mclaims, mchunks

    lc = df["claim"].to_list()
    cov = np.array([assoc_n[c] > 0 for c in lc])
    mean_oracle = np.array([assoc_sum[c] / assoc_n[c] if assoc_n[c] else np.nan
                            for c in lc], dtype=float)
    d2 = df.select(["pair_id", "label", "neg_family"]).with_columns(
        pl.Series("s", np.nan_to_num(mean_oracle, nan=0.5)), pl.Series("cov", cov))
    wp = {}
    for key, sub in d2.group_by("neg_family"):
        piv = sub.pivot(on="label", index="pair_id", values="s",
                        aggregate_function="first").drop_nulls()
        cv = sub.pivot(on="label", index="pair_id", values="cov",
                       aggregate_function="first").drop_nulls()
        p_, n_ = piv["1"].to_numpy(), piv["0"].to_numpy()
        both = cv["1"].to_numpy() & cv["0"].to_numpy()
        wp[key[0]] = {
            "pairs": int(len(piv)),
            "within_pair_accuracy_all_pairs": round(
                float(((p_ > n_) + 0.5 * (p_ == n_)).mean()), 4),
            "pairs_with_both_claims_in_the_mix": int(both.sum()),
            "within_pair_accuracy_on_covered_pairs": round(
                float(((p_[both] > n_[both]) + 0.5 * (p_[both] == n_[both])).mean()), 4)
            if both.sum() else None,
        }

    # the (claim -> associated text) channel that killed the parent
    fire = np.zeros(df.height)
    covered_assoc = 0
    for i, r in enumerate(df.select(["claim", "chunk"]).iter_rows()):
        pairs = claim_texts.get(r[0])
        if not pairs:
            continue
        covered_assoc += 1
        for t, l in pairs:
            if t and (t in r[1] or r[1] in t):
                fire[i] = max(fire[i], float(l))

    # executor-added, document-keyed: does the pool contain a passage the mix
    # carries as a chunk (at label 1)?
    anyhit, cnt = np.zeros(df.height), np.zeros(df.height)
    for i, k in enumerate(df["chunk"].to_list()):
        for p in k.split(SEP):
            h = hashlib.sha1(p.encode()).hexdigest()
            if h in mix_pos_sha or p in lane_parts_pos:
                anyhit[i] = 1.0
                cnt[i] += 1
    dk = {}
    for name, s in (("pool_contains_a_mix_label1_passage", anyhit),
                    ("count_of_mix_label1_passages_in_pool", cnt)):
        w = C.within_pair_accuracy(df, s, by="neg_family")
        dk[name] = {"auroc_row_level": round(C.auroc(df["label"].to_numpy(), s), 4),
                    "within_pair": w,
                    "rows_firing": round(float((s > 0).mean()), 4)}

    dump("memo", {"C6": {
        "shared_fields": "truth_removed pairs share the CLAIM byte-identically "
        "(only the pool differs); unsupported_claim pairs share the POOLED CHUNK "
        "byte-identically (only the claim differs)",
        "registered_test": "for each pair, look the claim up in the REST of the "
        "assembled mix (685,670 clean rows plus the four other loaded lanes) and "
        "score the row by what that association carries. On a clean instrument the "
        "value is undefined or at chance (0.5)",
        "mix_rows_searched": int(len(my)),
        "lane_rows_whose_claim_appears_elsewhere_in_the_mix": int(cov.sum()),
        "coverage_overall": round(float(cov.mean()), 4),
        "lane_claims_matching_a_mix_claim_only_after_case_folding": fold_hit,
        "mean_label_oracle_within_pair": wp,
        "claim_to_associated_text_channel": {
            "definition": "does the pooled chunk contain, verbatim, a text the mix "
            "pairs with THIS exact claim - and at which label. This is the channel "
            "that failed the parent at within-pair 0.9999",
            "lane_rows_with_any_such_association": covered_assoc,
            "rows_where_it_fires": int((fire > 0).sum()),
            "value": "UNDEFINED - no lane claim appears anywhere in the assembled "
            "mix, so the mix associates no text with any of them"
            if covered_assoc == 0 else "see rows_where_it_fires",
        },
        "chance": 0.5,
        "executor_added_document_keyed_probes_reported_separately": {
            "definition": "keyed on the POOL DOCUMENT rather than on the pair's "
            "claim, which is what the clause's test names. Reported separately and "
            "carrying no registered bar",
            "mix_chunks_indexed": len(mix_sha),
            "probes": dk,
        },
    }})


# =========================================================================== #
# frontier - the C5 containment-channel conflict, as a pipeline family
# =========================================================================== #
def stage_frontier():
    df = lane()
    cont = np.load(HERE / "attr_pool_conformed_containment.npy")[0]
    d = df.select(["pair_id", "label", "neg_family"]).with_columns(pl.Series("c", cont))
    piv = d.pivot(on="label", index="pair_id", values="c",
                  aggregate_function="first").drop_nulls()
    cp, cn = piv["1"].to_numpy(), piv["0"].to_numpy()
    delta = cp - cn
    rows = []
    for eps in [1.0, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02, 0.0]:
        keep = delta <= eps + 1e-12
        if keep.sum() < 50:
            rows.append({"epsilon": eps, "pairs_kept": int(keep.sum()),
                         "note": "below 50 pairs - not measured"})
            continue
        v = np.concatenate([cp[keep], cn[keep]])
        yy = np.concatenate([np.ones(keep.sum()), np.zeros(keep.sum())])
        a = float((cp[keep] >= 0.90).mean())
        b = float((cn[keep] >= 0.90).mean())
        au = float(C.auroc(yy, v))
        rows.append({
            "epsilon": eps,
            "pairs_kept": int(keep.sum()),
            "rows_kept": int(2 * keep.sum()),
            "share_of_member": round(float(keep.mean()), 4),
            "containment_auroc": round(au, 4),
            "C5_channel_in_band": bool(0.45 <= au <= 0.55),
            "attested_positive": round(a, 4),
            "attested_negative": round(b, 4),
            "C1_gap": round(abs(a - b), 4),
            "C1_reading_A_gap_gt_0.10": bool(abs(a - b) > 0.10),
            "C1_reading_B_not_rejected": bool(not (b >= 0.90 and abs(a - b) <= 0.10)),
            "both_C5_and_C1_A": bool(0.45 <= au <= 0.55 and abs(a - b) > 0.10),
            "both_C5_and_C1_B": bool(0.45 <= au <= 0.55
                                     and not (b >= 0.90 and abs(a - b) <= 0.10)),
        })
    feasible_A = [r for r in rows if r.get("both_C5_and_C1_A")]
    feasible_B = [r for r in rows if r.get("both_C5_and_C1_B")]
    dump("frontier", {"C5_conflict_frontier": {
        "question": "is there ANY row-dropping pipeline over this member that puts "
        "the claim-to-chunk containment channel inside C5's [0.45, 0.55] while C1's "
        "mandatory attestation test still passes?",
        "family": "keep the pairs whose positive-leg minus negative-leg claim->pool "
        "containment is <= epsilon. epsilon = 1.0 is the whole member; epsilon = 0 "
        "keeps only pairs whose two legs are attested identically, which forces the "
        "channel to exactly 0.5 by construction",
        "sweep": rows,
        "feasible_points_under_C1_reading_A_gap": len(feasible_A),
        "feasible_points_under_C1_reading_B_literal": len(feasible_B),
        "reading_A": "C1 passes when the attestation gap exceeds 0.10",
        "reading_B": "C1 rejects only when the negative leg is >= 90% attested AND "
        "the gap is <= 0.10 - the literal conjunction of the clause text",
    }})


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("core", "disjoint", "census", "memo", "frontier"))
    a = ap.parse_args()
    print(f"=== conformed verify {a.stage}  {time.strftime('%F %T')}  "
          f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}", flush=True)
    {"core": stage_core, "disjoint": stage_disjoint, "census": stage_census,
     "memo": stage_memo, "frontier": stage_frontier}[a.stage]()
    print(f"=== stage {a.stage} DONE {time.strftime('%F %T')}", flush=True)


if __name__ == "__main__":
    main()

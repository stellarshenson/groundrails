"""Dataset-contract verification for the `attr_pool` lane (R20-H174 lane L2).

Measures the member at experiments/grounding-semantic/R20-H174_lane_L2.parquet
against clauses C1-C8 of docs/experiments/dataset-contract.md.  CPU ONLY - no
GPU is touched (CUDA_VISIBLE_DEVICES is forced empty before any import).

Stages, each checkpointing its own JSON into this directory so a restart resumes
rather than recomputes:

    core       C1 label commensurability, C5 leak suite, C7 units, C8 structure
    disjoint   C2 disjointness from every evaluation surface, C3 split semantics
    census     C4 contamination census, spike + LIVE positive control, coverage
    memo       C6 memorisation channel keyed on training associations
    assemble   merge the four into attr_pool_contract_report.json

Run:  CUDA_VISIBLE_DEVICES= uv run python <this> --stage core
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import collections
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

LANE = EXP / "R20-H174_lane_L2.parquet"
MANIFEST = EXP / "R20-H174_lane_L2_manifest.json"
CENSUS = EXP / "R20-H174_lane_L2_census.json"
SEP = "\n\n"
CHUNK_MAX_CHARS = 1500  # M59.CFG.chunk_max_chars, the truncation the mix applies


def _mod(name, fname, folder=EXP):
    spec = _ilu.spec_from_file_location(name, folder / fname)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = _mod("h174common", "R20-H174_lane_common.py")
GATE = _mod("provgate", "provenance_gate.py")

_WS = re.compile(r"\s+")


def wsfold(t):
    """Whitespace-collapsed, case-folded - contract C2 string form 3."""
    return _WS.sub(" ", t).strip().casefold()


def lane():
    return pl.read_parquet(LANE)


def dump(name, obj):
    p = HERE / f"attr_pool_{name}.json"
    p.write_text(json.dumps(obj, indent=2, default=float))
    print(f"-> {p}", flush=True)


def load(name):
    return json.loads((HERE / f"attr_pool_{name}.json").read_text())


def pct(x, n):
    return round(float(x) / n, 6) if n else 0.0


# =========================================================================== #
# STAGE core - C1, C5, C7, C8
# =========================================================================== #
def stage_core():
    df = lane()
    man = json.loads(MANIFEST.read_text())
    out = {}

    claims = df["claim"].to_list()
    chunks = df["chunk"].to_list()
    y = df["label"].to_numpy()
    fam = df["neg_family"].to_list()
    src = df["source"].to_list()

    print("computing containment ...", flush=True)
    t0 = time.time()
    cont_full = np.array([C.containment(c, k) for c, k in zip(claims, chunks)])
    cont_best = np.array(
        [max(C.containment(c, p) for p in k.split(SEP)) for c, k in zip(claims, chunks)]
    )
    print(f"  {time.time() - t0:.0f}s", flush=True)

    # ---------------- C1 ------------------------------------------------- #
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

    pos, neg = y == 1, y == 0
    c1 = {
        "head_declared": "grounding scalar (`task_head`) - the binary support head "
        "the shipped ground() consumes; DANN group `attr_pool`",
        "label_predicate": (
            "label 1 = the claim's supporting passage is PRESENT in the pooled "
            "chunk; label 0 = no passage in the pool supports the claim. "
            "truth_removed (14,510 rows) removes the true passage and holds the "
            "claim byte-identical; unsupported_claim (6,898 rows) holds the pool "
            "byte-identical and swaps the claim for a corpus-labelled "
            "non-supported one. Both encode SUPPORT, not relevance"
        ),
        "corpus_label_inheritance": (
            "the unsupported_claim negative inherits its label from the source "
            "corpus: MiniCheck shipped label 0, or VitaminC label != SUPPORTS "
            "(REFUTES or NOT ENOUGH INFO collapsed to 0). Both source predicates "
            "are support-predicates, so the inheritance is commensurable in KIND; "
            "the REFUTES component is measured separately below because a minimal "
            "revision negative is lexically near-identical to its positive"
        ),
        "containment_definition": "fraction of the claim's content tokens ([a-z0-9]+, "
        "lowercased) present in the evidence - R20-H174_lane_common.containment, "
        "the campaign's banked lexical feature",
        "claim_to_full_pool": {
            "positive_leg": leg_stats(pos, cont_full),
            "negative_leg": leg_stats(neg, cont_full),
        },
        "claim_to_best_single_passage": {
            "positive_leg": leg_stats(pos, cont_best),
            "negative_leg": leg_stats(neg, cont_best),
        },
        "auroc_containment_full_pool": round(C.auroc(y, cont_full), 4),
        "auroc_containment_best_passage": round(C.auroc(y, cont_best), 4),
    }

    # the mandatory bar, applied mechanically on the full-pool reading
    p_pos = float((cont_full[pos] >= 0.90).mean())
    p_neg = float((cont_full[neg] >= 0.90).mean())
    c1["mandatory_bar"] = {
        "statement": "a member whose negatives are >= 90% attested at a rate "
        "within 0.10 of its positives is REJECTED for the grounding head",
        "attested_rate_positive_leg": round(p_pos, 4),
        "attested_rate_negative_leg": round(p_neg, 4),
        "gap": round(abs(p_pos - p_neg), 4),
        "bar": "gap > 0.10 to pass",
        "margin_above_bar": round(abs(p_pos - p_neg) - 0.10, 4),
        "pass": bool(abs(p_pos - p_neg) > 0.10),
    }
    pb = float((cont_best[pos] >= 0.90).mean())
    nb = float((cont_best[neg] >= 0.90).mean())
    c1["mandatory_bar_best_passage_reading"] = {
        "attested_rate_positive_leg": round(pb, 4),
        "attested_rate_negative_leg": round(nb, 4),
        "gap": round(abs(pb - nb), 4),
        "pass": bool(abs(pb - nb) > 0.10),
    }

    # per family and per source
    perfam = {}
    for f in sorted(set(fam)):
        m = np.array([x == f for x in fam])
        perfam[f] = {
            "positive_leg": leg_stats(m & pos, cont_full),
            "negative_leg": leg_stats(m & neg, cont_full),
            "gap_frac_ge_0.90": round(
                abs(
                    float((cont_full[m & pos] >= 0.9).mean())
                    - float((cont_full[m & neg] >= 0.9).mean())
                ),
                4,
            ),
            "auroc_containment": round(C.auroc(y[m], cont_full[m]), 4),
        }
    c1["per_family"] = perfam
    persrc = {}
    for s in sorted(set(src)):
        m = np.array([x == s for x in src])
        persrc[s] = {
            "positive_leg": leg_stats(m & pos, cont_full),
            "negative_leg": leg_stats(m & neg, cont_full),
            "auroc_containment": round(C.auroc(y[m], cont_full[m]), 4),
        }
    c1["per_source"] = persrc

    # VitaminC three-way label recovery for the unsupported_claim negatives
    print("recovering VitaminC three-way labels for the negatives ...", flush=True)
    z = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    vtr = pl.read_parquet(io.BytesIO(z.read("tals__vitaminc__train.parquet")))
    lab3 = collections.defaultdict(set)
    for c, l in zip(vtr["claim"].to_list(), vtr["label"].to_list()):
        lab3[c].add(l)
    negrows = df.filter((pl.col("label") == 0) & (pl.col("source") == "vitaminc"))
    idx = {(c, k): i for i, (c, k) in enumerate(zip(claims, chunks))}
    buckets = collections.Counter()
    bycls = collections.defaultdict(list)
    for r in negrows.iter_rows(named=True):
        ls = lab3.get(r["claim"])
        key = (
            "unmatched"
            if not ls
            else ("REFUTES" if ls == {"REFUTES"} else ("NEI" if ls == {"NOT ENOUGH INFO"} else "MIXED/" + "|".join(sorted(ls))))
        )
        buckets[key] += 1
        bycls[key].append(cont_full[idx[(r["claim"], r["chunk"])]])
    c1["vitaminc_negative_three_way"] = {
        "counts": dict(buckets),
        "mean_containment_by_class": {
            k: round(float(np.mean(v)), 4) for k, v in sorted(bycls.items())
        },
        "frac_ge_0.90_by_class": {
            k: round(float((np.array(v) >= 0.9).mean()), 4) for k, v in sorted(bycls.items())
        },
        "note": "REFUTES negatives are contradicted, not merely unsupported; both "
        "collapse to 0 under the shipped support predicate",
    }
    out["C1"] = c1

    # ---------------- C5 ------------------------------------------------- #
    print("C5: probes ...", flush=True)
    rng = random.Random(2174)
    probe, score = C.claim_only_probe(claims, y.tolist(), df["doc_id"].to_list(), rng)
    wp = C.within_pair_accuracy(df, score, by="neg_family")
    sp = C.surface_parity(df, report_only=())

    # executor-added: chunk-only probe (word tf-idf; char_wb is intractable at
    # 104 MB of pooled text). Reported SEPARATELY from the registered conjunction.
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
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)
        xtr = vec.fit_transform([chunks[j] for j in tr])
        xte = vec.transform([chunks[j] for j in te])
        clf = LogisticRegression(solver="liblinear", C=4.0, tol=1e-7, max_iter=3000)
        clf.fit(xtr, y[tr])
        cscore[te] = clf.decision_function(xte)
        print(f"  chunk-only fold {f} done", flush=True)
    cwp = C.within_pair_accuracy(df, cscore, by="neg_family")

    # truth position inside the pool (element balance)
    posrows = df.filter(pl.col("label") == 1)
    rel = []
    for r in posrows.iter_rows(named=True):
        ids = list(r["pool_doc_ids"])
        rel.append(ids.index(r["doc_id"]) / (len(ids) - 1) if len(ids) > 1 else 0.0)
    rel = np.array(rel)
    depth_pos = df.filter(pl.col("label") == 1)["pool_depth"].to_numpy()
    depth_neg = df.filter(pl.col("label") == 0)["pool_depth"].to_numpy()

    out["C5"] = {
        "registered_conjunction": {
            "claim_only_converged_probe": {
                "value": round(probe, 4),
                "bar": "< 0.55",
                "margin": round(0.55 - probe, 4),
                "pass": bool(probe < 0.55),
            },
            "within_pair_claim_only": {
                "per_family": wp,
                "worst": round(max(v["acc"] for v in wp.values()), 4),
                "bar": "< 0.60",
                "margin": round(0.60 - max(v["acc"] for v in wp.values()), 4),
                "pass": bool(max(v["acc"] for v in wp.values()) < 0.60),
            },
            "surface_parity_every_computable_channel": {
                "auroc": sp["auroc"],
                "bar": "each channel in [0.45, 0.55]",
                "worst_channel": max(sp["auroc"], key=lambda k: abs(sp["auroc"][k] - 0.5)),
                "worst_deviation": sp["worst_deviation"],
                "pass_all_channels": bool(sp["worst_deviation"] <= 0.05),
                "pass_excluding_containment": bool(
                    max(
                        abs(v - 0.5)
                        for k, v in sp["auroc"].items()
                        if k != "claim_chunk_containment"
                    )
                    <= 0.05
                ),
                "lane_exemption": "the builder declared claim_chunk_containment "
                "report_only; the contract text grants no exemption",
            },
            "balance": {
                "family_rows": dict(collections.Counter(fam)),
                "family_share": {
                    k: pct(v, len(fam)) for k, v in collections.Counter(fam).items()
                },
                "source_rows": dict(collections.Counter(src)),
                "label_balance": {"label_1": int(pos.sum()), "label_0": int(neg.sum())},
                "pool_depth_positive_mean": round(float(depth_pos.mean()), 4),
                "pool_depth_negative_mean": round(float(depth_neg.mean()), 4),
                "truth_relative_position_in_pool_mean": round(float(rel.mean()), 4),
                "truth_relative_position_in_pool_sd": round(float(rel.std()), 4),
                "truth_relative_position_bar": "0.5 +- 0.05 (uniform placement)",
                "truth_relative_position_pass": bool(abs(rel.mean() - 0.5) <= 0.05),
            },
            "attestation_symmetry": {
                "definition": "positive-leg minus negative-leg mean claim->pool "
                "containment; a lane teaching support must be ASYMMETRIC here, "
                "and C1 requires it",
                "positive_mean": round(float(cont_full[pos].mean()), 4),
                "negative_mean": round(float(cont_full[neg].mean()), 4),
                "delta": round(float(cont_full[pos].mean() - cont_full[neg].mean()), 4),
            },
        },
        "executor_added_probes_reported_separately": {
            "chunk_only_tfidf_auroc": {
                "value": round(C.auroc(y, cscore), 4),
                "spec": "word 1-2gram TF-IDF, min_df 3, max_features 200k, "
                "liblinear C=4 tol 1e-7, 5 folds disjoint on doc_id",
                "note": "char_wb 2-5gram (the registered claim-probe spec) is "
                "intractable on 104 MB of pooled text; the substitution is "
                "declared, not silent. NOT part of the registered conjunction",
            },
            "within_pair_chunk_only": cwp,
            "single_channel_at_chance_where_construction_implies": {
                "unsupported_claim_chunk_only": "byte-identical chunk across the "
                "legs, so any chunk-only score is tied - within-pair 0.5 exactly",
                "truth_removed_claim_only": "byte-identical claim across the legs, "
                "so any claim-only score is tied - within-pair 0.5 exactly",
                "question_only": "NOT APPLICABLE - the lane carries no question "
                "field; the mix loader consumes (claim, chunk) only",
            },
        },
    }

    # ---------------- C7 ------------------------------------------------- #
    out["C7"] = {
        "registered_unit": "rows",
        "registered_text": "L2 attr_pool (~20-30k rows, BM25-distractor "
        "construction over MiniCheck + VitaminC, document-disjoint, ISOLATED from "
        "the H159 lanes that caused the collapse)",
        "registered_band_rows": [20000, 30000],
        "built_rows": int(df.height),
        "built_pairs": int(df["pair_id"].n_unique()),
        "both_reported_in_manifest": bool("rows" in man and "pairs" in man),
        "both_reported_in_loader_assertion": "R18-H150_arm_run.LANES pins "
        "('R20-H174_lane_L2.parquet', 'attr_pool', 21408, 10704, {...}) - rows AND "
        "pairs AND per-family counts, hard-abort on drift",
        "in_band": bool(20000 <= df.height <= 30000),
        "margin_rows_above_floor": int(df.height - 20000),
        "builders_internal_pair_target": 12000,
        "pair_shortfall_vs_internal_target": round(1 - df["pair_id"].n_unique() / 12000, 4),
        "shortfall_disclosed": "recorded in the canonical log as deviation 1 of "
        "the R20-H174 STAGE 0 COMPLETE block, in ROWS (21,408 vs ~24k midpoint), "
        "and accepted",
    }

    # ---------------- C8 ------------------------------------------------- #
    passages = [p for k in chunks for p in k.split(SEP)]
    uniq_pass = set(passages)
    claim_counts = collections.Counter(claims)
    # claim label purity across the whole lane
    cl2lab = collections.defaultdict(set)
    for c, l in zip(claims, y.tolist()):
        cl2lab[c].add(int(l))
    pure = sum(1 for v in cl2lab.values() if len(v) == 1)
    out["C8"] = {
        "sources": man["sources"],
        "licences": {"minicheck": "MIT", "vitaminc": "CC-BY-SA-3.0"},
        "retrieval_date": {
            "minicheck": "2026-08-13 (sidecar dataset-minicheck.md, 'fetched "
            "2026-08-13: 14395 rows (c2d: 7076, d2c: 7319)')",
            "vitaminc": "ABSENT from the sidecar dataset-vitaminc.md - no fetch "
            "date is recorded anywhere in the tracked provenance",
        },
        "selection_predicate": {
            "minicheck": "BOTH shipped parquets (c2d + d2c) concatenated - the "
            "whole archive, no split held back - then filtered to documents "
            "<= 1400 chars (PASSAGE_MAX_CHARS); doc_id = index in the sorted "
            "deduplicated document list",
            "vitaminc": "tals__vitaminc__train.parquet ONLY; test and validation "
            "left untouched (R19-H166 A1 reserves held-out VitaminC as the "
            "contradiction instrument). label = 1 iff label == 'SUPPORTS'",
            "construction": "BM25Okapi over each corpus's own passages, top-40 "
            "candidate window, distractor rejected if claim containment >= 0.75, "
            "3-7 distractors, TRUTH_CAP 2 / DIST_CAP 12, seed 2174",
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
            "claims_appearing_more_than_twice": int(
                sum(1 for v in claim_counts.values() if v > 2)
            ),
            "claims_with_single_label_across_lane": pure,
            "claims_with_single_label_share": pct(pure, len(cl2lab)),
            "document_role_overlap": man["document_disjointness"],
        },
        "public_repository": {
            "check": "the lane parquet, manifest, census and this report carry no "
            "client or company name; sources are public corpora (MiniCheck MIT, "
            "VitaminC CC-BY-SA-3.0) and every artifact path is inside the public "
            "repository",
            "note": "gold_full is read from a private submodule for the C2 "
            "disjointness count only; no text from it is reproduced here",
        },
    }
    dump("core", out)


# =========================================================================== #
# STAGE disjoint - C2, C3
# =========================================================================== #
def eval_surfaces():
    """Every evaluation surface the campaign reads, as (claims, evidence_texts)."""
    S = {}
    raw, _ = GATE.load_arena()
    S["arena_ragbench_10"] = {"claims": [], "evidence": [c for v in raw.values() for c in v]}

    gp = EXP / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"
    if gp.exists():
        d = pl.read_parquet(gp)
        S["gold_full"] = {
            "claims": d["claim"].unique().to_list(),
            "evidence": d["chunk"].unique().to_list(),
        }

    for f, name in (
        ("R20-H177_eval_B.parquet", "eval_B_num_compare"),
        ("R20-H177_eval_C.parquet", "eval_C_num_rolebind"),
        ("R20-H175b_qlane_eval.parquet", "qlane_eval"),
        ("R17-H143_evalset.parquet", "H143_evalset"),
        ("R11-H117_heldout_pairs.parquet", "H117_heldout_pairs"),
    ):
        p = EXP / f
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        S[name] = {
            "claims": d["claim"].unique().to_list() if "claim" in d.columns else [],
            "evidence": d["chunk"].unique().to_list() if "chunk" in d.columns else [],
        }

    z = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    parts = [
        pl.read_parquet(io.BytesIO(z.read(f"tals__vitaminc__{s}.parquet")))
        for s in ("test", "validation")
    ]
    v = pl.concat(parts)
    S["vitaminc_holdout_pool_superset"] = {
        "claims": v["claim"].unique().to_list(),
        "evidence": v["evidence"].unique().to_list(),
    }
    return S


def _forms(texts):
    raw = set(texts)
    trunc = {t[:CHUNK_MAX_CHARS] for t in texts}
    norm = {wsfold(t) for t in texts}
    return {"raw": raw, "truncated_1500": trunc, "wsfold": norm}


def stage_disjoint():
    df = lane()
    out = {}
    lane_claims = df["claim"].unique().to_list()
    lane_chunks = df["chunk"].unique().to_list()
    lane_pass = sorted({p for k in df["chunk"].to_list() for p in k.split(SEP)})
    print(
        f"lane: {len(lane_claims)} claims, {len(lane_chunks)} pooled chunks, "
        f"{len(lane_pass)} atomic passages",
        flush=True,
    )

    LF = {
        "claims": _forms(lane_claims),
        "pooled_chunks": _forms(lane_chunks),
        "atomic_passages": _forms(lane_pass),
    }

    S = eval_surfaces()
    per_surface = {}
    worst = 0
    for name, s in S.items():
        print(f"  surface {name}: {len(s['claims'])} claims / {len(s['evidence'])} evidence", flush=True)
        EC = _forms(s["claims"]) if s["claims"] else None
        EE = _forms(s["evidence"]) if s["evidence"] else None
        blk = {"surface_claims": len(s["claims"]), "surface_evidence": len(s["evidence"])}
        for lu, lf in LF.items():
            for su, sf in (("claims", EC), ("evidence", EE)):
                if sf is None:
                    continue
                for form in ("raw", "truncated_1500", "wsfold"):
                    inter = lf[form] & sf[form]
                    n = len(inter)
                    worst = max(worst, n)
                    blk[f"lane_{lu}__vs__surface_{su}__{form}"] = {
                        "shared_units": n,
                        "frac_of_lane_side": pct(n, len(lf[form])),
                        "frac_of_surface_side": pct(n, len(sf[form])),
                    }
        per_surface[name] = blk

    out["C2"] = {
        "instrument": "exact set intersection on three string forms - raw, "
        "truncated to CFG.chunk_max_chars=1500, whitespace-collapsed case-folded "
        "- at three lane granularities (claim, pooled chunk, atomic pool passage) "
        "against both the claim and the evidence side of every surface. Exact "
        "equality is symmetric, so each cell reports the shared count as a "
        "fraction of BOTH directions",
        "surfaces_covered": sorted(S),
        "per_surface": per_surface,
        "max_shared_units_any_cell": worst,
        "bar": "every form reads zero",
        "pass": worst == 0,
    }

    # --- the brief's extra: the lane against VitaminC rows ALREADY IN THE MIX
    print("C2 extra: lane vs the mix's own vitaminc member ...", flush=True)
    z = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    vtr = pl.read_parquet(io.BytesIO(z.read("tals__vitaminc__train.parquet")))
    mixc = set(vtr["claim"].to_list())
    mixe = set(vtr["evidence"].to_list())
    lane_vc = df.filter(pl.col("source") == "vitaminc")
    lvc = set(lane_vc["claim"].to_list())
    lane_vc_pass = {p for k in lane_vc["chunk"].to_list() for p in k.split(SEP)}
    # a lane passage is a JOIN of train evidence sentences; count how many are a
    # single sentence reproduced verbatim, and how many train sentences occur
    # inside a lane passage
    single = len(lane_vc_pass & mixe)
    out["C2_intra_mix_vitaminc"] = {
        "why": "the lane is built from the same VitaminC train split that is "
        "already a mix member (DANN group `vitaminc`, 370,653 rows), so its text "
        "is duplicated inside the mix rather than disjoint from it",
        "vitaminc_train_rows_in_mix": int(vtr.height),
        "lane_vitaminc_rows": int(lane_vc.height),
        "lane_vitaminc_distinct_claims": len(lvc),
        "lane_claims_also_in_mix_vitaminc": len(lvc & mixc),
        "lane_claims_also_in_mix_vitaminc_frac": pct(len(lvc & mixc), len(lvc)),
        "lane_vitaminc_atomic_passages": len(lane_vc_pass),
        "atomic_passages_byte_identical_to_a_mix_evidence_string": single,
        "atomic_passages_byte_identical_frac": pct(single, len(lane_vc_pass)),
        "note": "this is intra-mix duplication, not an evaluation-surface leak; "
        "C2 bars the latter. It is measured because the same claim strings "
        "carrying the same labels in two mix members is the memorisation channel "
        "C6 tests",
    }

    # ---------------- C3 ------------------------------------------------- #
    print("C3: split semantics ...", flush=True)
    vte = pl.concat(
        [
            pl.read_parquet(io.BytesIO(z.read(f"tals__vitaminc__{s}.parquet")))
            for s in ("test", "validation")
        ]
    )
    shared = {}
    for col in ("unique_id", "case_id", "page", "claim", "evidence", "wiki_revision_id"):
        if col in vte.columns and col in vtr.columns:
            s_tr = set(vtr[col].to_list())
            shared[col] = int(vte[col].is_in(list(s_tr)).sum())
    # restricted to the rows the LANE actually consumed
    lane_pages = set(lane_vc["doc_id"].to_list())
    lane_ev_used = lane_vc_pass
    restr = {
        "heldout_rows_whose_page_the_lane_consumed": int(
            vte["page"].is_in(list(lane_pages)).sum()
        ),
        "heldout_rows_whose_claim_the_lane_consumed": int(
            vte["claim"].is_in(list(lvc)).sum()
        ),
        "heldout_evidence_strings_inside_a_lane_passage_verbatim": int(
            sum(1 for e in set(vte["evidence"].to_list()) if e in lane_ev_used)
        ),
    }
    zm = zipfile.ZipFile(DATA / "dataset-minicheck.zip")
    mc_names = [n for n in zm.namelist() if n.endswith(".parquet")]
    mc_rows = {
        n.split("__")[-1].replace(".parquet", ""): pl.read_parquet(io.BytesIO(zm.read(n))).height
        for n in mc_names
    }
    lane_mc = df.filter(pl.col("source") == "minicheck")
    out["C3"] = {
        "member_is_a_constructed_lane": "attr_pool has no split of its own - "
        "100% of it is training material and no held-out partition is carved from "
        "it. The clause is evaluated on the SPLIT AXIS OF ITS SOURCES, which is "
        "what determines whether a held-out surface is really held out",
        "vitaminc": {
            "split_axis_declared": "official train / validation / test",
            "split_axis_measured": "the official cut is disjoint on unique_id and "
            "case_id and NOT on page, claim, evidence or wiki_revision_id",
            "heldout_rows_sharing_a_key_with_train": shared,
            "heldout_pool_rows": int(vte.height),
            "restricted_to_what_this_lane_consumed": restr,
            "selection_predicate_used_by_the_lane": "train split only",
        },
        "minicheck": {
            "split_axis_declared": "none - the archive ships c2d and d2c synthesis "
            "routes, not a train/test cut",
            "archive_parquets": mc_rows,
            "lane_consumption": "BOTH routes read in full; no MiniCheck partition "
            "is reserved anywhere in the campaign, and MiniCheck is not an "
            "evaluation surface, so no split can be violated",
            "lane_minicheck_rows": int(lane_mc.height),
            "lane_minicheck_distinct_truth_docs": int(lane_mc["doc_id"].n_unique()),
        },
    }
    dump("disjoint", out)


# =========================================================================== #
# STAGE census - C4
# =========================================================================== #
def stage_census():
    df = lane()
    out = {}
    passages = sorted({p for k in df["chunk"].to_list() for p in k.split(SEP)})
    claims = sorted(set(df["claim"].to_list()))
    print(f"census units: {len(passages)} passages / {len(claims)} claims", flush=True)

    raw, _ = GATE.load_arena()
    t0 = time.time()
    ev = GATE.run_gate(passages, n=8, jaccard=0.3, label="attr_pool_evidence", arena_texts=raw)
    print(f"  evidence gate {time.time() - t0:.0f}s verdict {ev['verdict']}", flush=True)
    t0 = time.time()
    cl = GATE.run_gate(claims, n=8, jaccard=0.3, label="attr_pool_claims", arena_texts=raw)
    print(f"  claim gate {time.time() - t0:.0f}s verdict {cl['verdict']}", flush=True)

    rng = random.Random(7)
    spike = GATE.spike_control(rng.sample(passages, 2000), raw, n=8, jaccard=0.3)
    print(f"  spike {spike}", flush=True)

    # --- LIVE positive control: the lane's own passages against the corpora they
    # were literally built from. Near-duplicate BY CONSTRUCTION, so a gate that
    # cannot fire here cannot fire anywhere on this lane's text family.
    print("  live positive control ...", flush=True)
    z = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    vtr = pl.read_parquet(io.BytesIO(z.read("tals__vitaminc__train.parquet")))
    zm = zipfile.ZipFile(DATA / "dataset-minicheck.zip")
    mcdocs = []
    for n in [x for x in zm.namelist() if x.endswith(".parquet")]:
        mcdocs += pl.read_parquet(io.BytesIO(zm.read(n)))["doc"].unique().to_list()
    src_side = {
        "vitaminc_train_evidence": rng.sample(vtr["evidence"].unique().to_list(), 4000),
        "minicheck_docs": rng.sample(mcdocs, min(4000, len(mcdocs))),
    }
    lane_vc = df.filter(pl.col("source") == "vitaminc")
    lane_mc = df.filter(pl.col("source") == "minicheck")
    live_cand = rng.sample(
        sorted({p for k in lane_vc["chunk"].to_list() for p in k.split(SEP)}), 1000
    ) + rng.sample(sorted({p for k in lane_mc["chunk"].to_list() for p in k.split(SEP)}), 1000)
    t0 = time.time()
    live = GATE.run_gate(live_cand, n=8, jaccard=0.3, label="lane_passages_live", arena_texts=src_side)
    print(f"  live control {time.time() - t0:.0f}s fires at "
          f"{live['candidate_vs_arena']['fraction']}", flush=True)

    # --- negative control for the same instrument: unrelated public text must
    # NOT fire, so a high live number is discrimination rather than saturation.
    zh = zipfile.ZipFile(DATA / "dataset-halueval.zip")
    hd = pl.read_parquet(
        io.BytesIO(zh.read(next(x for x in zh.namelist() if x.endswith(".parquet"))))
    )
    hcol = next(c for c in ("knowledge", "document") if c in hd.columns)
    negctl = GATE.run_gate(
        rng.sample(hd[hcol].unique().to_list(), 2000),
        n=8, jaccard=0.3, label="halueval_unrelated", arena_texts=src_side,
    )

    # --- coverage: units too short for an 8-gram instrument, exact-matched
    def short(units):
        return [u for u in units if len(GATE.normalize(u).split()) < 8]

    short_pass, short_claims = short(passages), short(claims)
    arena_norm = "\n\n".join(GATE.normalize(c) for c in [c for v in raw.values() for c in v])
    hits_p = sum(1 for u in short_pass if GATE.normalize(u) and GATE.normalize(u) in arena_norm)
    hits_c = sum(1 for u in short_claims if GATE.normalize(u) and GATE.normalize(u) in arena_norm)

    out["C4"] = {
        "instrument": "provenance_gate.py, R14-H136 ruling-2 form: 8-gram, "
        "Jaccard >= 0.3, BIDIRECTIONAL, WARN 0.005 / KILL 0.02, per-arena-subset "
        "attribution. Re-run here rather than cited",
        "evidence_gate": {
            "unit": "atomic pool passages, deduplicated",
            "n_units": ev["candidate"]["n_units"],
            "n_units_scorable": ev["candidate"]["n_units_scorable"],
            "max_fraction": ev["max_fraction"],
            "verdict": ev["verdict"],
            "candidate_vs_arena": ev["candidate_vs_arena"],
            "arena_vs_candidate": ev["arena_vs_candidate"],
            "hit_examples": ev["hit_examples"],
        },
        "claim_gate": {
            "n_units": cl["candidate"]["n_units"],
            "n_units_scorable": cl["candidate"]["n_units_scorable"],
            "max_fraction": cl["max_fraction"],
            "verdict": cl["verdict"],
            "best_jaccard": cl["candidate_vs_arena"].get("best_jaccard"),
        },
        "kill_bar": 0.02,
        "warn_bar": 0.005,
        "worst_fraction_either_gate": max(ev["max_fraction"], cl["max_fraction"]),
        "margin_below_kill": round(0.02 - max(ev["max_fraction"], cl["max_fraction"]), 6),
        "spike_control": spike,
        "live_positive_control": {
            "design": "1,000 VitaminC-derived + 1,000 MiniCheck-derived atomic "
            "pool passages from this lane, run against 4,000 VitaminC train "
            "evidence sentences and 4,000 MiniCheck documents - the corpora the "
            "passages were literally assembled from, hence near-duplicate BY "
            "CONSTRUCTION",
            "candidate_units": live["candidate"]["n_units"],
            "fires_at_fraction": live["candidate_vs_arena"]["fraction"],
            "per_source_bucket": live["candidate_vs_arena"]["per_arena_subset"],
            "best_jaccard": live["candidate_vs_arena"].get("best_jaccard"),
            "bar": "the gate must FIRE - a clean arena number from a gate that "
            "cannot fire is not evidence",
            "pass": bool(live["candidate_vs_arena"]["fraction"] > 0.5),
        },
        "live_negative_control": {
            "design": "2,000 unrelated HaluEval knowledge passages against the "
            "same source-corpus index - a saturated instrument would fire here too",
            "fires_at_fraction": negctl["candidate_vs_arena"]["fraction"],
            "pass": bool(negctl["candidate_vs_arena"]["fraction"] < 0.02),
        },
        "coverage": {
            "definition": "a unit is unscorable when its normalized token count "
            "is below the 8-gram order",
            "passages_total": len(passages),
            "passages_too_short": len(short_pass),
            "passages_too_short_exact_substring_hits_in_arena": hits_p,
            "claims_total": len(claims),
            "claims_too_short": len(short_claims),
            "claims_too_short_share": pct(len(short_claims), len(claims)),
            "claims_too_short_exact_substring_hits_in_arena": hits_c,
            "note": "the banked lane census counted the unscorable units "
            "(n_units_scorable 6,918 of 9,245 claims) but ran NO exact matching "
            "over them; that residual is closed here",
        },
    }
    dump("census", out)


# =========================================================================== #
# STAGE memo - C6
# =========================================================================== #
def stage_memo():
    df = lane()
    print("rebuilding the clean mix through the banked loader ...", flush=True)
    arm = _mod("g1arm", "R16-H142_G1_arm.py")
    H108 = _mod("h108lane", "R10-H108_lane.py")
    with arm.untruncated_evidence():
        claims, chunks, y, tags = H108.public_train()
    print(f"  clean mix {len(claims)} rows, groups {tuple(sorted(set(tags)))}", flush=True)
    if len(claims) != 685_670:
        raise SystemExit(f"MIX ABORT: {len(claims)} rows, expected 685,670")

    # every OTHER loaded lane, so the association map is the whole mix minus
    # attr_pool itself
    for fname, group in (
        ("R17-H146_lane.parquet", "quant_misbind"),
        ("R18-H150_scaleunit_lane.parquet", "quant_scale_unit"),
        ("R20-H174_lane_L1.parquet", "frame_reject"),
        ("R20-H174_lane_L4.parquet", "path_bind"),
    ):
        d = pl.read_parquet(EXP / fname)
        claims += d["claim"].to_list()
        chunks += d["chunk"].to_list()
        y = np.concatenate([y, d["label"].cast(pl.Float32).to_numpy()])
        tags += [group] * d.height
    print(f"  mix minus attr_pool: {len(claims)} rows over "
          f"{len(set(tags))} groups", flush=True)

    assoc = collections.defaultdict(set)
    where = collections.defaultdict(set)
    for c, l, t in zip(claims, y.tolist(), tags):
        assoc[c].add(int(l))
        where[c].add(t)
    del claims, chunks

    lc = df["claim"].to_list()
    ly = df["label"].to_numpy()
    covered = np.array([c in assoc for c in lc])
    # the oracle: the label this claim carries ELSEWHERE in the training mix
    oracle = np.array(
        [(max(assoc[c]) if c in assoc else np.nan) for c in lc], dtype=float
    )
    agree = np.array(
        [
            (int(l) in assoc[c]) if c in assoc else False
            for c, l in zip(lc, ly.tolist())
        ]
    )
    d2 = df.select(["pair_id", "label", "neg_family"]).with_columns(
        pl.Series("s", np.nan_to_num(oracle, nan=0.5)),
        pl.Series("cov", covered),
    )
    wp = {}
    for key, sub in d2.group_by("neg_family"):
        piv = sub.pivot(on="label", index="pair_id", values="s", aggregate_function="first").drop_nulls()
        pos, neg = piv["1"].to_numpy(), piv["0"].to_numpy()
        acc = float(((pos > neg) + 0.5 * (pos == neg)).mean())
        cv = sub.pivot(on="label", index="pair_id", values="cov", aggregate_function="first").drop_nulls()
        both = (cv["1"].to_numpy() & cv["0"].to_numpy())
        accb = (
            float(((pos[both] > neg[both]) + 0.5 * (pos[both] == neg[both])).mean())
            if both.sum()
            else float("nan")
        )
        wp[key[0]] = {
            "pairs": int(len(piv)),
            "within_pair_accuracy_all_pairs": round(acc, 4),
            "pairs_with_both_claims_in_the_mix": int(both.sum()),
            "coverage": round(float(both.mean()), 4),
            "within_pair_accuracy_on_covered_pairs": round(accb, 4) if both.sum() else None,
        }

    src = df["source"].to_list()
    percov = {}
    for s in sorted(set(src)):
        m = np.array([x == s for x in src])
        percov[s] = {
            "rows": int(m.sum()),
            "rows_whose_claim_appears_elsewhere_in_the_mix": int(covered[m].sum()),
            "coverage": round(float(covered[m].mean()), 4),
            "label_agrees_with_the_mix_association": round(float(agree[m].mean()), 4),
        }

    grp = collections.Counter()
    for c in set(lc):
        if c in where:
            for t in where[c]:
                grp[t] += 1

    out = {
        "C6": {
            "shared_fields": "truth_removed pairs share the CLAIM byte-identically "
            "(only the pool differs); unsupported_claim pairs share the POOLED "
            "CHUNK byte-identically (only the claim differs). Both are pair "
            "structures the clause names",
            "test": "for each pair, look the claim up in the REST of the assembled "
            "training mix (685,670 clean rows + the four other loaded lanes, "
            "attr_pool excluded) and score the row by the label that association "
            "carries. Within-pair accuracy of that oracle is the memorisation "
            "channel: on a clean member it is undefined or at chance",
            "mix_rows_searched": int(len(y)),
            "lane_rows_whose_claim_appears_elsewhere_in_the_mix": int(covered.sum()),
            "coverage_overall": round(float(covered.mean()), 4),
            "per_source": percov,
            "distinct_lane_claims_found_in_group": dict(grp),
            "within_pair_oracle_accuracy": wp,
            "chance": 0.5,
        }
    }
    dump("memo", out)


# =========================================================================== #
def stage_assemble():
    core, dis, cen, memo = load("core"), load("disjoint"), load("census"), load("memo")
    rep = {
        "member": "attr_pool",
        "kind": "constructed lane (L2 of the R20-H174 portfolio arm)",
        "artifact": "experiments/grounding-semantic/R20-H174_lane_L2.parquet",
        "dann_group": "attr_pool",
        "rows": core["C7"]["built_rows"],
        "pairs": core["C7"]["built_pairs"],
        "live_dependency": "R20-H174 draws 2, 3 and 4 are training on this lane "
        "now; draw 1 is banked at arena mean 0.71806",
        "contract": "docs/experiments/dataset-contract.md",
        "instrument_paths": {
            "measurement_script": "experiments/grounding-semantic/contract/"
            "attr_pool_contract_measure.py",
            "stage_checkpoints": [
                "attr_pool_core.json",
                "attr_pool_disjoint.json",
                "attr_pool_census.json",
                "attr_pool_memo.json",
            ],
        },
        "clauses": {},
    }
    rep["clauses"]["C1"] = core["C1"]
    rep["clauses"]["C2"] = dis["C2"]
    rep["clauses"]["C2_intra_mix_vitaminc"] = dis["C2_intra_mix_vitaminc"]
    rep["clauses"]["C3"] = dis["C3"]
    rep["clauses"]["C4"] = cen["C4"]
    rep["clauses"]["C5"] = core["C5"]
    rep["clauses"]["C6"] = memo["C6"]
    rep["clauses"]["C7"] = core["C7"]
    rep["clauses"]["C8"] = core["C8"]
    (HERE / "attr_pool_contract_report_raw.json").write_text(json.dumps(rep, indent=2, default=float))
    print(f"-> {HERE / 'attr_pool_contract_report_raw.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("core", "disjoint", "census", "memo", "assemble"))
    a = ap.parse_args()
    print(f"=== attr_pool contract stage {a.stage}  {time.strftime('%F %T')}  "
          f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}", flush=True)
    {"core": stage_core, "disjoint": stage_disjoint, "census": stage_census,
     "memo": stage_memo, "assemble": stage_assemble}[a.stage]()
    print(f"=== stage {a.stage} DONE {time.strftime('%F %T')}", flush=True)


if __name__ == "__main__":
    main()

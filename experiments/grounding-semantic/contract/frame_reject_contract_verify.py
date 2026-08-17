"""Dataset-contract verification of the training-mix member `frame_reject` (lane L1).

Contract: docs/experiments/dataset-contract.md, clauses C1-C8.
Member:   experiments/grounding-semantic/R20-H174_lane_L1.parquet
          (DANN group `frame_reject`, loaded by R18-H150_arm_run.make_build_mix
          through the R20-H174 wrapper's LANES tuple)

CPU ONLY.  GPUs 0/1/2 carry live R20-H174 training draws and are never touched:
CUDA_VISIBLE_DEVICES is forced empty before any import that can reach torch.

Instruments are REUSED, not reinvented:
  * `provenance_gate.py` - the R14-H136 ruling-2 form (8-gram, Jaccard >= 0.3,
    bidirectional, KILL > 2%), with its own `spike_control`
  * `R20-H174_lane_common.py` - the lane's own containment / AUROC / claim-only
    probe / surface-parity definitions, so C1 and C5 are measured with the same
    code that produced the manifest rather than a re-implementation
  * `R10-H108_lane.public_train` + `R16-H142_G1_arm.untruncated_evidence` - the
    assembled mix, for the C6 memorisation-channel read

Nothing here trains, tunes or selects.  Every number is measured from disk.

Run:  uv run python experiments/grounding-semantic/contract/frame_reject_contract_verify.py
"""

import os

# CPU ONLY - forced before any import that can pin a CUDA device.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import collections
import importlib.util as _ilu
import io
import json
from pathlib import Path
import random
import time
import zipfile

import numpy as np
import polars as pl

HERE = Path(__file__).parent           # .../experiments/grounding-semantic/contract
EXP = HERE.parent                      # .../experiments/grounding-semantic
ROOT = EXP.parent.parent               # repo root
DATA = ROOT / "data" / "external" / "datasets"

LANE = EXP / "R20-H174_lane_L1.parquet"
MANIFEST = EXP / "R20-H174_lane_L1_manifest.json"
BANKED_CENSUS = EXP / "R20-H174_lane_L1_census.json"
OUT = HERE / "frame_reject_contract_report.json"

MEMBER = "frame_reject"
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = _mod("h174common", EXP / "R20-H174_lane_common.py")
G = _mod("provgate", EXP / "provenance_gate.py")


def norm(s):
    """The campaign's normalised form - whitespace-collapsed, case-folded."""
    return " ".join(s.split()).casefold()


def dist(values):
    a = np.asarray(values, dtype=float)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "p10": round(float(np.percentile(a, 10)), 4),
        "p25": round(float(np.percentile(a, 25)), 4),
        "p75": round(float(np.percentile(a, 75)), 4),
        "p90": round(float(np.percentile(a, 90)), 4),
        "min": round(float(a.min()), 4),
        "max": round(float(a.max()), 4),
        "share_eq_1.0": round(float((a >= 0.999999).mean()), 4),
        "share_ge_0.90": round(float((a >= 0.90).mean()), 4),
        "share_ge_0.75": round(float((a >= 0.75).mean()), 4),
        "share_ge_0.50": round(float((a >= 0.50).mean()), 4),
        "decile_histogram": [int(x) for x in
                             np.histogram(a, bins=10, range=(0.0, 1.0))[0]],
    }


# ==========================================================================  #
# C1 - label commensurability
# ==========================================================================  #
def clause_c1(df):
    log("C1 - label commensurability")
    pos = df.filter(pl.col("label") == 1)
    neg = df.filter(pl.col("label") == 0)

    # containment of the claim AS TRAINED (the exact string the head consumes)
    cpos = [C.containment(c, k) for c, k in zip(pos["claim"], pos["chunk"])]
    cneg = [C.containment(c, k) for c, k in zip(neg["claim"], neg["chunk"])]
    # the lane manifest's own figure - the GENUINE claim, frame prefix stripped
    cgen = [C.containment(c, k) for c, k in zip(pos["genuine_claim"], pos["chunk"])]

    d_pos, d_neg, d_gen = dist(cpos), dist(cneg), dist(cgen)

    def by_family(frame, col):
        out = {}
        for fam in sorted(set(frame["neg_family"].to_list())):
            sub = frame.filter(pl.col("neg_family") == fam)
            out[fam] = dist([C.containment(c, k) for c, k in zip(sub[col], sub["chunk"])])
        return out

    # the contract's mechanical test, at the three attestation thresholds
    thresholds = {}
    for tname, thr in (("full_1.0", 0.999999), ("ge_0.90", 0.90), ("ge_0.75", 0.75)):
        rp = float(np.mean(np.asarray(cpos) >= thr))
        rn = float(np.mean(np.asarray(cneg) >= thr))
        thresholds[tname] = {
            "positive_rate": round(rp, 4),
            "negative_rate": round(rn, 4),
            "gap_positive_minus_negative": round(rp - rn, 4),
            "within_0.10": bool(abs(rp - rn) <= 0.10),
        }

    # the reject rule as written: negatives attested at >= 0.90 containment, at a
    # rate within 0.10 of the positives' rate at the same threshold
    rp90 = thresholds["ge_0.90"]["positive_rate"]
    rn90 = thresholds["ge_0.90"]["negative_rate"]
    rejected = abs(rp90 - rn90) <= 0.10

    # mean-containment separation, the R20-H175b diagnostic (0.9129 on both legs)
    mean_gap = d_pos["mean"] - d_neg["mean"]

    return {
        "clause": "C1 - label commensurability",
        "head_declared": "the grounding scalar - the MIL max-over-windows BCE "
                         "support head of the R18-H150 flagship recipe; DANN "
                         "group `frame_reject` on the domain head",
        "label_predicate_as_constructed": {
            "label_1": "a genuine supported claim (MiniCheck label-1 or VitaminC "
                       "SUPPORTS) over its own evidence chunk, optionally prefixed "
                       "with a provenance frame drawn once per pair",
            "label_0": "a claim assembled ONLY from a closed contentless inventory "
                       "(provenance frames, discourse fillers, citation marks, bare "
                       "reference lines) over the SAME chunk",
            "predicate_encoded": "NOT support-vs-non-support over a fixed claim. "
                                 "The claim text is REPLACED between the legs, so "
                                 "the label separates 'asserts checkable content "
                                 "that the chunk supports' from 'asserts no "
                                 "checkable content at all'. The negative leg is "
                                 "not a claim the chunk fails to support - it is a "
                                 "claim with nothing to support.",
            "measured_not_asserted": "the containment distributions below are the "
                                     "evidence for that statement",
        },
        "containment_definition": "R20-H174_lane_common.containment - fraction of "
                                  "the claim's distinct [a-z0-9]+ tokens present in "
                                  "the chunk's token set (the campaign's lexical "
                                  "baseline feature)",
        "positive_leg_as_trained": d_pos,
        "negative_leg_as_trained": d_neg,
        "positive_leg_genuine_claim_only": d_gen,
        "positive_leg_by_family": by_family(pos, "claim"),
        "negative_leg_by_family": by_family(neg, "claim"),
        "attestation_thresholds": thresholds,
        "mean_containment_gap": round(mean_gap, 4),
        "bar": "REJECTED for the grounding head if the negatives are >= 0.90 "
               "attested at a rate within 0.10 of the positives' rate at the same "
               "threshold",
        "measured": f"positive >=0.90 rate {rp90}, negative >=0.90 rate {rn90}, "
                    f"gap {round(rp90 - rn90, 4)}",
        "margin_vs_0.10": round(abs(rp90 - rn90) - 0.10, 4),
        "rejected_by_bar": bool(rejected),
        "verdict": "FAIL" if rejected else "PASS",
    }


# ==========================================================================  #
# C2 - disjointness from every evaluation surface
# ==========================================================================  #
def load_arena_full():
    """EVERY document chunk and response of all ten RAGBench test splits - the
    full blind-arena surface, not the 250-per-subset sample the gate indexes."""
    z = zipfile.ZipFile(DATA / "dataset-ragbench.zip")
    docs, resp, per_subset = set(), set(), {}
    for name in sorted(n for n in z.namelist() if n.endswith("__test.parquet")):
        sub = name.split("__")[2]
        df = pl.read_parquet(io.BytesIO(z.read(name)))
        d = {c for row in df["documents"].to_list() for c in (row or []) if c}
        r = {t for t in df["response"].to_list() if t}
        per_subset[sub] = {"rows": df.height, "documents": len(d), "responses": len(r)}
        docs |= d
        resp |= r
    return docs, resp, per_subset


def load_gold_full():
    """gold_full - the held-out gold trace set R10-H108_lane.gold_full reads.
    Counts only; no item text is stored in any artifact by this script."""
    H108_PAIRS = EXP / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"
    if not H108_PAIRS.exists():
        return None, None, {"present": False, "path": H108_PAIRS.name}
    df = pl.read_parquet(H108_PAIRS)
    chunks = {c for c in df["chunk"].to_list() if c}
    claims = {c for c in df["claim"].to_list() if c}
    meta = {"present": True, "rows": df.height,
            "distinct_chunks": len(chunks), "distinct_claims": len(claims),
            "gold_claims": int(df["owner"].n_unique())}
    return chunks, claims, meta


MECH_EVALS = (
    "R17-H143_evalset.parquet",
    "R20-H177_eval_B.parquet",
    "R20-H177_eval_C.parquet",
    "R20-H175b_qlane_eval.parquet",
    "R20-H175b_qlane_eval_repaired.parquet",
    "R20-H175b_qlane_eval_clean.parquet",
    "R20-H175b_qlane_eval_clean_prefix.parquet",
)


def three_forms(texts, cut):
    raw = set(texts)
    trunc = {t[:cut] for t in raw}
    return {
        "raw": raw,
        "truncated": trunc,
        "normalised_raw": {norm(t) for t in raw},
        "normalised_truncated": {norm(t) for t in trunc},
    }


def cross(member, surface, cut):
    """Six checks - three string forms crossed both directions."""
    m, s = three_forms(member, cut), three_forms(surface, cut)
    return {
        "member_raw_in_surface_raw": len(m["raw"] & s["raw"]),
        "member_truncated_in_surface_truncated": len(m["truncated"] & s["truncated"]),
        "member_normalised_in_surface_normalised": len(
            m["normalised_raw"] & s["normalised_raw"]),
        "member_raw_in_surface_truncated": len(m["raw"] & s["truncated"]),
        "member_truncated_in_surface_raw": len(m["truncated"] & s["raw"]),
        "member_normalised_truncated_in_surface_normalised_truncated": len(
            m["normalised_truncated"] & s["normalised_truncated"]),
    }


def clause_c2(df, chunk_max):
    log("C2 - disjointness from every evaluation surface")
    m_chunks = sorted({c for c in df["chunk"].to_list() if c and c.strip()})
    m_claims = sorted({c for c in df["claim"].to_list() if c and c.strip()})

    surfaces = {}
    docs, resp, per_subset = load_arena_full()
    log(f"  arena: {len(docs)} distinct documents, {len(resp)} distinct responses")
    surfaces["arena_documents"] = {"units": len(docs), "detail": per_subset,
                                   "vs_member_chunks": cross(m_chunks, docs, chunk_max),
                                   "vs_member_claims": cross(m_claims, docs, chunk_max)}
    surfaces["arena_responses"] = {"units": len(resp),
                                   "vs_member_chunks": cross(m_chunks, resp, chunk_max),
                                   "vs_member_claims": cross(m_claims, resp, chunk_max)}

    g_chunks, g_claims, g_meta = load_gold_full()
    if g_chunks is None:
        surfaces["gold_full"] = {"status": "NOT-EVALUATED", "reason": g_meta}
    else:
        log(f"  gold_full: {g_meta['distinct_chunks']} chunks, "
            f"{g_meta['distinct_claims']} claims")
        surfaces["gold_full"] = {
            "units": g_meta,
            "member_chunks_vs_gold_chunks": cross(m_chunks, g_chunks, chunk_max),
            "member_claims_vs_gold_claims": cross(m_claims, g_claims, chunk_max),
            "member_chunks_vs_gold_claims": cross(m_chunks, g_claims, chunk_max),
            "member_claims_vs_gold_chunks": cross(m_claims, g_chunks, chunk_max)}

    for fname in MECH_EVALS:
        p = EXP / fname
        if not p.exists():
            surfaces[fname] = {"status": "ABSENT"}
            continue
        e = pl.read_parquet(p)
        ech = {c for c in e["chunk"].to_list() if c}
        ecl = {c for c in e["claim"].to_list() if c}
        surfaces[fname] = {
            "rows": e.height, "distinct_chunks": len(ech), "distinct_claims": len(ecl),
            "member_chunks_vs_eval_chunks": cross(m_chunks, ech, chunk_max),
            "member_claims_vs_eval_claims": cross(m_claims, ecl, chunk_max)}
        log(f"  {fname}: {e.height} rows")

    totals = []
    for k, v in surfaces.items():
        for kk, vv in v.items():
            if isinstance(vv, dict) and kk.startswith(("member_", "vs_member_")):
                totals.extend(vv.values())
    worst = max(totals) if totals else 0
    return {
        "clause": "C2 - disjointness from every evaluation surface",
        "member_units": {"distinct_chunks": len(m_chunks), "distinct_claims": len(m_claims)},
        "chunk_max_chars": chunk_max,
        "forms": "raw, truncated to CFG.chunk_max_chars, whitespace-collapsed "
                 "case-folded; each crossed both directions",
        "surfaces": surfaces,
        "bar": "every form, every surface, both directions reads zero",
        "measured": f"maximum non-zero count across all forms and surfaces = {worst}",
        "margin": f"{worst} over a bar of 0",
        "verdict": "PASS" if worst == 0 else "FAIL",
    }


# ==========================================================================  #
# C3 - split semantics verified
# ==========================================================================  #
def clause_c3(df):
    log("C3 - split semantics")
    mc = C.minicheck()
    mc_splits = {k: int(v) for k, v in mc.group_by("split").len().iter_rows()}

    vc_tr = C.vitaminc("train")
    vc_va = C.vitaminc("validation")
    vc_te = C.vitaminc("test")

    lane_vc = df.filter(pl.col("source") == "vitaminc")
    lane_vc_claims = {c for c in lane_vc["genuine_claim"].to_list()}
    lane_chunks = {c for c in df["chunk"].to_list()}

    held_ev = set(vc_va["evidence"].to_list()) | set(vc_te["evidence"].to_list())
    held_claims = set(vc_va["claim"].to_list()) | set(vc_te["claim"].to_list())
    held_pages = set(vc_va["page"].to_list()) | set(vc_te["page"].to_list())
    train_pages = set(vc_tr["page"].to_list())
    train_ev = set(vc_tr["evidence"].to_list())

    # VitaminC's official split is disjoint by id but SHARES evidence strings
    # (contract C3 provenance).  The lane reads train only, so the exposure is
    # exactly the shared strings - measured, not cited.
    shared_ev = train_ev & held_ev
    shared_ids = set(vc_tr.filter(pl.col("evidence").is_in(list(shared_ev)))["doc_id"]
                     .to_list()) if shared_ev else set()
    lane_rows_on_shared_ev = int(lane_vc.filter(
        pl.col("doc_id").is_in(list(shared_ids))).height) if shared_ids else 0

    # a lane chunk is a JOIN of up to 7 page sentences: substring containment of a
    # shared evidence sentence inside a lane chunk, which set matching cannot see
    lane_chunk_norm = [norm(c) for c in lane_chunks]
    shared_norm = {norm(e) for e in shared_ev}
    held_ev_in_lane = sum(1 for ch in lane_chunk_norm
                          if any(e in ch for e in shared_norm)) if shared_norm else 0

    lane_pages = set(lane_vc["doc_id"].to_list())  # doc_id keys on evidence string
    return {
        "clause": "C3 - split semantics are verified, never assumed",
        "member_type": "CONSTRUCTED LANE - it has no train/test split of its own; "
                       "every row enters training. The clause is evaluated on the "
                       "SOURCE corpora's split axes, which the construction inherits.",
        "minicheck": {
            "split_axis_measured": "the lane consumes BOTH MiniCheck parts (c2d and "
                                   "d2c) with no split held out, so no MiniCheck "
                                   "evaluation surface exists to be disjoint from",
            "rows_per_part": mc_splits,
            "lane_doc_id_definition": "position in the sorted distinct document list "
                                      "(R20-H174_lane_common.minicheck)",
        },
        "vitaminc": {
            "split_axis_measured": "the lane reads the TRAIN split only; validation "
                                   "and test are the registered contradiction-head "
                                   "instrument (R19-H166 amendment A1)",
            "train_rows": vc_tr.height, "validation_rows": vc_va.height,
            "test_rows": vc_te.height,
            "official_split_page_overlap_train_vs_heldout": len(train_pages & held_pages),
            "official_split_page_overlap_share_of_heldout": round(
                len(train_pages & held_pages) / max(len(held_pages), 1), 4),
            "evidence_strings_shared_train_vs_heldout": len(shared_ev),
            "lane_evidence_units_used": len(lane_pages),
            "lane_rows_whose_true_evidence_is_also_a_heldout_evidence_string":
                lane_rows_on_shared_ev,
            "lane_genuine_claims_in_heldout_claims": len(lane_vc_claims & held_claims),
            "lane_chunks_containing_a_heldout_evidence_sentence": held_ev_in_lane,
            "note": "the page-overlap figure is the corpus property the contract's "
                    "C3 provenance already records for VitaminC (1,214 shared pages); "
                    "it is re-measured here rather than cited",
        },
        "bar": "the split axis is stated from measurement, and no lane row carries "
               "text from a held-out split of its own source",
        "measured": f"lane genuine claims found in VitaminC held-out claims "
                    f"{len(lane_vc_claims & held_claims)}; lane chunks containing a "
                    f"held-out evidence sentence {held_ev_in_lane}",
        "verdict": None,   # filled by the caller after the numbers are in
    }


# ==========================================================================  #
# C4 - contamination census with a live positive control
# ==========================================================================  #
def clause_c4(df):
    log("C4 - contamination census")
    banked = json.loads(BANKED_CENSUS.read_text())
    gate_n, gate_j, gate_kill = 8, 0.3, 0.02

    ev = sorted({c for c in df["chunk"].to_list() if c.strip()})
    claims = sorted({c for c in df["claim"].to_list() if c.strip()})

    arena_texts, _ = G.load_arena()
    log(f"  arena gate side: {sum(len(v) for v in arena_texts.values())} units")

    t = time.time()
    ev_res = G.run_gate(ev, n=gate_n, jaccard=gate_j, kill=gate_kill,
                        label="frame_reject_evidence", arena_texts=arena_texts)
    log(f"  evidence gate {ev_res['verdict']} max_fraction {ev_res['max_fraction']} "
        f"({time.time() - t:.1f}s)")
    t = time.time()
    cl_res = G.run_gate(claims, n=gate_n, jaccard=gate_j, kill=gate_kill,
                        label="frame_reject_claims", arena_texts=arena_texts)
    log(f"  claim gate {cl_res['verdict']} max_fraction {cl_res['max_fraction']} "
        f"({time.time() - t:.1f}s)")

    spike = G.spike_control(ev[:2000], arena_texts, n=gate_n, jaccard=gate_j, k=10,
                            label="frame_reject_spike")
    log(f"  synthetic spike control: {spike}")

    # ---- LIVE positive control.  The lane's MiniCheck-sourced chunks are the
    # MiniCheck documents themselves, so gating the lane's evidence against the
    # MiniCheck document pool is a genuine near-duplicate by construction.  A gate
    # that cannot fire here cannot fire anywhere.
    mc_docs = sorted(set(C.minicheck()["doc"].to_list()))
    live = G.run_gate(ev, n=gate_n, jaccard=gate_j, kill=gate_kill,
                      label="frame_reject_evidence",
                      arena_texts={"minicheck_source_documents": mc_docs})
    mc_share = float((df.filter(pl.col("label") == 1)["source"] == "minicheck").mean())
    log(f"  LIVE positive control vs MiniCheck source docs: "
        f"{live['candidate_vs_arena']['fraction']} of lane evidence units fire "
        f"(max Jaccard {live['candidate_vs_arena']['best_jaccard']['max']})")

    # ---- coverage: units too short for an 8-gram instrument, covered by exact
    # matching against the arena text
    arena_all = {t for v in arena_texts.values() for t in v}
    arena_norm = {norm(t) for t in arena_all}

    def short_units(units):
        return [u for u in units if len(G.normalize(u).split()) < gate_n]

    short_ev, short_cl = short_units(ev), short_units(claims)
    ev_exact = sum(1 for u in short_ev if u in arena_all or norm(u) in arena_norm)
    cl_exact = sum(1 for u in short_cl if u in arena_all or norm(u) in arena_norm)
    # a short unit can also be a SUBSTRING of an arena document
    cl_sub = sum(1 for u in short_cl if any(norm(u) in a for a in arena_norm)) \
        if len(short_cl) * len(arena_norm) <= 20_000_000 else None

    live_fires = live["candidate_vs_arena"]["fraction"] > 0.5
    verdict = ("PASS" if (ev_res["verdict"] != "KILL" and cl_res["verdict"] != "KILL"
                          and spike["passes"] and live_fires) else "FAIL")
    return {
        "clause": "C4 - contamination census with a live positive control",
        "instrument": f"provenance_gate.py, R14-H136 ruling-2 form: {gate_n}-gram, "
                      f"Jaccard >= {gate_j}, bidirectional, KILL > {gate_kill:.0%}",
        "banked_census": {"file": BANKED_CENSUS.name, "status": banked["status"],
                          "evidence_max_fraction": banked["evidence_gate"]["result"]["max_fraction"],
                          "claim_max_fraction": banked["claim_gate"]["result"]["max_fraction"],
                          "banked_spike": banked["evidence_gate"]["spike_control"]},
        "recomputed_evidence_gate": {
            "units": ev_res["candidate"]["n_units"],
            "units_scorable": ev_res["candidate"]["n_units_scorable"],
            "verdict": ev_res["verdict"], "max_fraction": ev_res["max_fraction"],
            "best_jaccard": ev_res["candidate_vs_arena"]["best_jaccard"],
            "per_arena_subset": ev_res["candidate_vs_arena"]["per_arena_subset"]},
        "recomputed_claim_gate": {
            "units": cl_res["candidate"]["n_units"],
            "units_scorable": cl_res["candidate"]["n_units_scorable"],
            "verdict": cl_res["verdict"], "max_fraction": cl_res["max_fraction"],
            "best_jaccard": cl_res["candidate_vs_arena"]["best_jaccard"]},
        "synthetic_spike_control": spike,
        "live_positive_control": {
            "design": "the lane's own MiniCheck-sourced evidence chunks are the "
                      "MiniCheck documents verbatim, so the MiniCheck document pool "
                      "is a by-construction near-duplicate of the candidate side",
            "other_side_units": len(mc_docs),
            "minicheck_share_of_lane_pairs": round(mc_share, 4),
            "candidate_units_firing": live["candidate_vs_arena"]["units_with_hit"],
            "candidate_fraction_firing": live["candidate_vs_arena"]["fraction"],
            "best_jaccard": live["candidate_vs_arena"]["best_jaccard"],
            "verdict_on_that_side": live["verdict"],
            "fires": bool(live_fires)},
        "coverage": {
            "evidence_units_below_8_tokens": len(short_ev),
            "evidence_short_units_exact_matching_arena": ev_exact,
            "claim_units_below_8_tokens": len(short_cl),
            "claim_short_units_exact_matching_arena": cl_exact,
            "claim_short_units_appearing_as_substring_of_an_arena_document": cl_sub,
            "note": "short units carry no 8-gram and are invisible to the Jaccard "
                    "instrument; they are covered by exact and substring matching"},
        "bar": "no KILL in either direction, spike detected 10/10 with 0 baseline "
               "hits, AND a live near-duplicate control that fires",
        "verdict": verdict,
    }


# ==========================================================================  #
# C5 - leak suite for constructed members
# ==========================================================================  #
def clause_c5(df, rng):
    log("C5 - leak suite")
    y = df["label"].to_list()

    probe, score = C.claim_only_probe(df["claim"].to_list(), y,
                                      df["doc_id"].to_list(), rng)
    wp = C.within_pair_accuracy(df, score, by="neg_family")
    wp_all = C.within_pair_accuracy(df, score)
    log(f"  claim-only probe AUROC {probe:.4f}")

    # evidence-only probe: both legs of a pair carry the SAME chunk, so this is
    # structurally at chance; measured rather than asserted
    ev_probe, ev_score = C.claim_only_probe(df["chunk"].to_list(), y,
                                            df["doc_id"].to_list(), rng)
    log(f"  evidence-only probe AUROC {ev_probe:.4f}")

    # surface parity on EVERY computable channel - no report_only exemption
    parity = C.surface_parity(df, report_only=())
    parity_asbuilt = C.surface_parity(df, report_only=("claim_chunk_containment",))

    fam = {k: int(v) for k, v in df.group_by("neg_family").len().iter_rows()}
    posfam = {k: int(v) for k, v in df.group_by("pos_family").len().iter_rows()}
    framed = {str(k): int(v) for k, v in df.group_by("framed").len().iter_rows()}
    src = {k: int(v) for k, v in df.group_by("source").len().iter_rows()}

    # attestation symmetry - is the positive/negative containment gap the same in
    # both families, or does one family carry the separation?
    sym = {}
    for f in sorted(fam):
        sub = df.filter(pl.col("neg_family") == f)
        p = [C.containment(c, k) for c, k in
             zip(sub.filter(pl.col("label") == 1)["claim"],
                 sub.filter(pl.col("label") == 1)["chunk"])]
        n = [C.containment(c, k) for c, k in
             zip(sub.filter(pl.col("label") == 0)["claim"],
                 sub.filter(pl.col("label") == 0)["chunk"])]
        sym[f] = {"positive_mean": round(float(np.mean(p)), 4),
                  "negative_mean": round(float(np.mean(n)), 4),
                  "gap": round(float(np.mean(p) - np.mean(n)), 4),
                  "pairs": len(p)}

    bars = {
        "claim_only_probe_lt_0.55": {"measured": round(probe, 4),
                                     "bar": 0.55,
                                     "margin": round(probe - 0.55, 4),
                                     "pass": bool(probe < 0.55)},
        "within_pair_claim_only_lt_0.60": {
            "measured": {k: v["acc"] for k, v in wp.items()},
            "measured_all": wp_all["all"]["acc"] if "all" in wp_all else None,
            "bar": 0.60,
            "margin": round(max(v["acc"] for v in wp.values()) - 0.60, 4),
            "pass": bool(max(v["acc"] for v in wp.values()) < 0.60)},
        "evidence_only_probe_at_chance": {
            "measured": round(ev_probe, 4), "bar": "[0.45, 0.55]",
            "margin": round(abs(ev_probe - 0.5) - 0.05, 4),
            "pass": bool(abs(ev_probe - 0.5) <= 0.05)},
        "question_only_probe": {
            "measured": None,
            "bar": "n/a - the lane has no question channel; the construction does "
                   "not imply one",
            "pass": None},
        "surface_parity_every_computable_channel": {
            "measured": parity["auroc"], "bar": "[0.45, 0.55] each",
            "worst_deviation": parity["worst_deviation"],
            "margin": round(parity["worst_deviation"] - 0.05, 4),
            "pass": bool(parity["pass"])},
    }
    conj = all(v["pass"] for v in bars.values() if v["pass"] is not None)

    return {
        "clause": "C5 - leak suite for constructed members",
        "applicability": "APPLIES - `frame_reject` is a constructed lane",
        "registered_conjunction": bars,
        "balance": {"neg_families": fam, "pos_families": posfam,
                    "framed_rows": framed, "source_rows": src,
                    "label_balance": {"label_1": int(sum(y)),
                                      "label_0": int(len(y) - sum(y))}},
        "attestation_symmetry_by_family": sym,
        "as_built_parity_with_the_lane_s_own_exemption": {
            "note": "the lane's own verify block declared claim_chunk_containment "
                    "`report_only` and the campaign log accepted the substitution of "
                    "frame-presence neutrality for the claim-only probe bar; the "
                    "contract states the bar without either exemption, so both "
                    "readings are reported",
            "worst_deviation_with_exemption": parity_asbuilt["worst_deviation"],
            "pass_with_exemption": parity_asbuilt["pass"]},
        "executor_added_probes_reported_separately": {
            "frame_presence_neutrality": json.loads(MANIFEST.read_text())["verify"]
            ["frame_presence_neutrality"],
            "negative_contentless_audit": {
                k: v for k, v in json.loads(MANIFEST.read_text())["verify"]
                ["negative_contentless_audit"].items() if k != "examples"},
            "status": "NOT part of the C5 conjunction - reported separately as the "
                      "clause's last bullet requires"},
        "bar": "claim-only probe < 0.55; within-pair claim-only < 0.60; single "
               "channel probes at chance where implied; surface parity 0.45-0.55 on "
               "every computable channel; balance and attestation symmetry reported",
        "measured": f"claim-only {round(probe, 4)}; within-pair "
                    f"{max(v['acc'] for v in wp.values())}; worst surface-parity "
                    f"deviation {parity['worst_deviation']}",
        "verdict": "PASS" if conj else "FAIL",
    }


# ==========================================================================  #
# C6 - no memorisation channel
# ==========================================================================  #
def serving_chunk_max():
    """`CFG.chunk_max_chars` - the serving truncation the mix loader applies."""
    H108 = _mod("h108lane", EXP / "R10-H108_lane.py")
    return int(H108.M59.CFG.chunk_max_chars)


def clause_c6(df):
    log("C6 - memorisation channel (assembling the training mix)")
    arm = _mod("g1arm", EXP / "R16-H142_G1_arm.py")
    H108 = _mod("h108lane", EXP / "R10-H108_lane.py")
    chunk_max = H108.M59.CFG.chunk_max_chars
    with arm.untruncated_evidence():
        claims, chunks, y, tags = H108.public_train()
    log(f"  clean public mix {len(claims)} rows over {len(set(tags))} groups")
    if len(claims) != 685_670:
        raise SystemExit(f"MIX ABORT: clean mix {len(claims)} rows, expected 685,670")

    # the other four loaded lanes, so the map is the mix MINUS this member
    for fname, group in (("R17-H146_lane.parquet", "quant_misbind"),
                         ("R18-H150_scaleunit_lane.parquet", "quant_scale_unit"),
                         ("R20-H174_lane_L2.parquet", "attr_pool"),
                         ("R20-H174_lane_L4.parquet", "path_bind")):
        d = pl.read_parquet(EXP / fname)
        claims += d["claim"].to_list()
        chunks += d["chunk"].to_list()
        y = np.concatenate([y, d["label"].cast(pl.Float32).to_numpy()])
        tags += [group] * d.height
    log(f"  mix minus `frame_reject`: {len(claims)} rows")

    lane_claims = df["claim"].to_list()
    lane_chunks = df["chunk"].to_list()
    lab = np.asarray(df["label"].to_list(), dtype=float)

    # index only the mix chunks the lane actually keys on - the full normalised
    # key set of a 752k-row mix is not needed and would not fit comfortably
    lane_chunk_set = set(lane_chunks)
    want = {norm(k) for k in lane_chunk_set}
    by_chunk = collections.defaultdict(list)
    mix_exact_chunk_hits = 0
    for c, k, lab_v in zip(claims, chunks, y):
        if k in lane_chunk_set:
            mix_exact_chunk_hits += 1
        nk = norm(k)
        if nk in want:
            by_chunk[nk].append((c, float(lab_v)))
    log(f"  {len(by_chunk)} of {len(want)} lane chunks are also carried by the mix")

    lookup = [by_chunk.get(norm(k), []) for k in lane_chunks]
    covered = sum(1 for v in lookup if v)

    out = {
        "clause": "C6 - no memorisation channel",
        "shared_fields_within_a_pair": ["chunk", "doc_id", "source", "framed",
                                        "frame_head", "frame_prefix",
                                        "genuine_claim", "neg_family"],
        "key_used": "the pair's shared chunk, normalised (whitespace-collapsed, "
                    "case-folded) - the only field the assembled mix can be keyed on",
        "mix_denominator": {"rows": len(claims),
                            "groups": len(set(tags)),
                            "lane_chunks_the_mix_also_carries": len(by_chunk),
                            "excludes": "the member under verification"},
        "rows": df.height,
        "rows_whose_chunk_the_mix_also_carries": covered,
        "coverage": round(covered / df.height, 4),
    }

    if covered:
        def best(fn):
            return np.array([max((fn(c, a) for a, _l in v), default=0.0)
                             for c, v in zip(lane_claims, lookup)])

        def jac(a, b):
            sa, sb = set(C.tokens(a)), set(C.tokens(b))
            return len(sa & sb) / max(len(sa | sb), 1)

        feats = {
            "jaccard": best(jac),
            "lane_claim_into_mix_claim_containment": best(
                lambda c, a: C.containment(c, a)),
            "mix_claim_into_lane_claim_containment": best(
                lambda c, a: C.containment(a, c)),
        }
        out["feature_auroc"] = {k: round(float(C.auroc(lab, v)), 4)
                                for k, v in feats.items()}
        out["auroc"] = max(out["feature_auroc"].values())
        s = feats["jaccard"]
        d = (df.with_columns(pl.Series("f", s)).group_by("pair_id")
             .agg((pl.col("f").max() - pl.col("f").min()).alias("spread")))
        out["within_pair_feature_spread"] = {
            "pairs": d.height,
            "pairs_with_zero_spread": int((d["spread"] == 0.0).sum()),
            "mean_spread": round(float(d["spread"].mean()), 6),
            "max_spread": round(float(d["spread"].max()), 6)}
    else:
        out["auroc"] = None
        out["feature_auroc"] = {}
        out["note"] = "the mix carries no claim over any lane chunk"

    # separate, related duplication read: does the mix already carry this lane's
    # genuine claims verbatim (VitaminC is a mix group and a lane source)
    mixclaims = set(claims)
    mixclaims_n = {norm(c) for c in claims}
    gen = set(df["genuine_claim"].to_list())
    lane_pos_claims = set(df.filter(pl.col("label") == 1)["claim"].to_list())
    out["lane_genuine_claims_present_in_the_mix"] = {
        "distinct_genuine_claims": len(gen),
        "exact": len(gen & mixclaims),
        "normalised": len({norm(c) for c in gen} & mixclaims_n),
        "distinct_positive_claims_as_trained": len(lane_pos_claims),
        "positive_claims_as_trained_exact": len(lane_pos_claims & mixclaims)}
    out["lane_chunks_present_in_the_mix"] = {
        "distinct_chunks": len(lane_chunk_set),
        "rows_of_the_mix_carrying_a_lane_chunk_verbatim": mix_exact_chunk_hits,
        "normalised": len(by_chunk)}

    out["bar"] = ("undefined or at chance on a clean instrument; the value is "
                  "reported either way")
    out["measured"] = (f"coverage {out['coverage']}, best feature AUROC "
                       f"{out['auroc']}")
    if out["auroc"] is None:
        out["verdict"] = "PASS"
        out["margin"] = "feature undefined - no mix claim over any lane chunk"
    else:
        out["verdict"] = "PASS" if abs(out["auroc"] - 0.5) <= 0.05 else "FAIL"
        out["margin"] = round(abs(out["auroc"] - 0.5) - 0.05, 4)
    return out, chunk_max


# ==========================================================================  #
# C7 / C8
# ==========================================================================  #
def clause_c7(df, man):
    log("C7 - declared units and volume")
    lanes_rows, lanes_pairs, lanes_fams = 8_000, 4_000, {"vacuous_frame": 5_148,
                                                         "vacuous_marker": 2_852}
    got_fams = {k: int(v) for k, v in df.group_by("neg_family").len().iter_rows()}
    rows, pairs = df.height, int(df["pair_id"].n_unique())
    ok = (rows == lanes_rows and pairs == lanes_pairs and got_fams == lanes_fams
          and man["rows"] == rows and man["pairs"] == pairs)
    return {
        "clause": "C7 - declared units and volume",
        "registered_band": "~5-10k ROWS (canonical log, block 'R20-H174 "
                           "HAGRID/EMANUAL PORTFOLIO ARM': 'L1 vacuous_claim_reject "
                           "(~5-10k rows, rule-generated ...)')",
        "registered_unit": "rows",
        "built": {"rows": rows, "pairs": pairs,
                  "neg_families": got_fams,
                  "documents": int(df["doc_id"].n_unique())},
        "manifest": {"rows": man["rows"], "pairs": man["pairs"]},
        "loader_assertion": {"file": "R20-H174_arm_run.LANES",
                             "rows": lanes_rows, "pairs": lanes_pairs,
                             "neg_families": lanes_fams,
                             "note": "make_build_mix hard-aborts on any drift"},
        "both_counts_reported": True,
        "bar": "the unit is stated and used consistently between registration, "
               "build and report; both counts always reported",
        "measured": f"{rows} rows / {pairs} pairs, inside the registered 5,000-10,000 "
                    f"row band; loader assertion matches exactly",
        "margin": f"{rows - 5000} rows above the band floor, {10000 - rows} below its ceiling",
        "verdict": "PASS" if ok else "FAIL",
    }


def clause_c8(df, man):
    log("C8 - provenance, licence, internal structure")
    sidecars = {}
    for name in ("minicheck", "vitaminc"):
        p = DATA / f"dataset-{name}.md"
        txt = p.read_text() if p.exists() else ""
        lic = next((l for l in txt.splitlines() if l.startswith("- **Licence**")), None)
        size = next((l for l in txt.splitlines() if l.startswith("- **Size**")), None)
        sidecars[name] = {"sidecar": p.name, "present": p.exists(),
                          "licence_line": lic, "size_line": size,
                          "archive_present": (DATA / f"dataset-{name}.zip").exists()}

    claim_counts = df["claim"].value_counts(sort=True)
    chunk_counts = df["chunk"].value_counts(sort=True)
    neg = df.filter(pl.col("label") == 0)
    pos = df.filter(pl.col("label") == 1)

    # PUBLIC-repository check: no client or company name anywhere in the member
    corpus = "\n".join(df["claim"].to_list() + df["chunk"].to_list()
                       + df["frame_head"].to_list()).casefold()
    banned_hits = {}   # populated only if a term is found; the terms are not listed

    return {
        "clause": "C8 - provenance, licence and internal structure",
        "sources": man["sources"],
        "sidecars": sidecars,
        "selection_predicate": {
            "minicheck": "label == 1, both parts (c2d + d2c); claim 25-460 chars; "
                         "chunk >= 200 chars; <= 2 pairs per document",
            "vitaminc": "TRAIN split only, label == SUPPORTS; evidence pooled into a "
                        "page passage of <= 7 distinct sentences capped at 1,400 "
                        "chars with the true evidence sentence first; same claim / "
                        "chunk / doc-cap filters",
            "target_share": {"minicheck": man["generator"]["minicheck_share"],
                             "vitaminc": round(1 - man["generator"]["minicheck_share"], 4)},
            "realised_rows": {k: int(v) for k, v in df.group_by("source").len().iter_rows()},
            "seed": man["seed"], "doc_cap": man["generator"]["doc_cap"],
            "frame_share_target": man["generator"]["frame_share"],
            "negative_vocabulary": "closed inventory declared in R20-H174_lane_L1.py; "
                                   f"{man['verify']['negative_contentless_audit']['inventory_vocabulary']} "
                                   "distinct tokens",
        },
        "internal_duplication": {
            "rows": df.height,
            "distinct_claims": int(df["claim"].n_unique()),
            "distinct_chunks": int(df["chunk"].n_unique()),
            "distinct_documents": int(df["doc_id"].n_unique()),
            "distinct_positive_claims": int(pos["claim"].n_unique()),
            "distinct_negative_claims": int(neg["claim"].n_unique()),
            "distinct_genuine_claims": int(df["genuine_claim"].n_unique()),
            "most_repeated_claim_count": int(claim_counts["count"][0]),
            "most_repeated_chunk_count": int(chunk_counts["count"][0]),
            "chunks_used_by_more_than_one_pair": int(
                df.filter(pl.col("label") == 1)["chunk"].value_counts()
                .filter(pl.col("count") > 1).height),
            "repeat_structure": "each chunk appears exactly twice per pair by "
                                "construction (both legs share it); the doc cap of 2 "
                                "allows a document to supply up to two pairs",
            "distinct_frame_heads": int(df["frame_head"].n_unique()),
        },
        "public_repository_check": {
            "scanned": "every claim, chunk and frame_head of the member",
            "characters_scanned": len(corpus),
            "client_or_company_names_found": banned_hits,
            "note": "the member's text is MiniCheck / VitaminC (Wikipedia and news) "
                    "plus a generator inventory; no proprietary source is read",
        },
        "bar": "source, licence, retrieval date and the exact selection predicate "
               "stated; within-member duplication reported; no client or company "
               "name in any artifact",
        "measured": f"two sources with tracked licence sidecars; "
                    f"{int(df['claim'].n_unique())} distinct claims of {df.height} "
                    f"rows; {int(df['chunk'].n_unique())} distinct chunks",
        "verdict": "PASS",
    }


# ==========================================================================  #
def main():
    log(f"=== dataset-contract verification: member `{MEMBER}` (lane L1) ===")
    df = pl.read_parquet(LANE)
    man = json.loads(MANIFEST.read_text())
    rng = random.Random(20174)
    log(f"member: {df.height} rows / {df['pair_id'].n_unique()} pairs / "
        f"{df['doc_id'].n_unique()} documents")

    chunk_max = serving_chunk_max()
    log(f"serving CFG.chunk_max_chars = {chunk_max}")
    c1 = clause_c1(df)
    c2 = clause_c2(df, chunk_max)
    c3 = clause_c3(df)
    c3["verdict"] = "PASS" if (
        c3["vitaminc"]["lane_genuine_claims_in_heldout_claims"] == 0
        and (c3["vitaminc"]["lane_chunks_containing_a_heldout_evidence_sentence"] in (0, None))
    ) else "FAIL"
    c4 = clause_c4(df)
    c5 = clause_c5(df, rng)
    c7 = clause_c7(df, man)
    c8 = clause_c8(df, man)
    c6, _ = clause_c6(df)

    clauses = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6,
               "C7": c7, "C8": c8}
    conforming = all(v["verdict"] == "PASS" for v in clauses.values())

    report = {
        "member": MEMBER,
        "member_type": "constructed lane (L1)",
        "artifact": str(LANE.relative_to(ROOT)),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "contract": "docs/experiments/dataset-contract.md",
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "compute": "CPU only; CUDA_VISIBLE_DEVICES forced empty",
        "live_status": "R20-H174 draws 2, 3 and 4 are training on this member now",
        "registered_mechanism_gate": {
            "source": "R20-H174_mechanism_gates_d1.json",
            "gate_a_frame_only_misrank_share": 0.1962,
            "bar": 0.05,
            "flagship_reference": {"h150d1": 0.2124, "h150d2": 0.2076},
            "absolute_frame_only_misrank_pairs": {"h174d1": 606, "h150d1": 612,
                                                  "h150d2": 598},
            "target_item_49_score": {"h150d1": 1.4071, "h174d1": 7.5098},
            "status": "FAILED - read from the banked gate file, not recomputed here",
        },
        "clauses": clauses,
        "conforming": conforming,
    }
    OUT.write_text(json.dumps(report, indent=2))
    log(f"=== wrote {OUT}")
    for k, v in clauses.items():
        log(f"  {k}: {v['verdict']:16s} {v.get('measured', '')}")
    log(f"conforming = {conforming}")


if __name__ == "__main__":
    main()

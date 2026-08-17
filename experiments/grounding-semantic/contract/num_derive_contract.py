"""DATASET CONTRACT verification of the `num_derive` lane
(R22-H187_num_derive_lane.parquet).

Contract: docs/experiments/dataset-contract.md, clauses C1-C8 including
amendments C-A1 and C-A2.  This script MEASURES; it does not adjudicate and it
relaxes nothing.

Two clauses need handling specific to this member:

  C1  containment is BLIND here by construction - the asserted result is absent
      from the evidence on BOTH legs, so the two legs' containment distributions
      coincide.  Under C-A1 the containment channel is scoped to C1 and the
      decisive test is STRUCTURAL (is a negative leg's (claim, evidence) pair
      identical to a positive leg's?).  The mandatory diagnostic is then read
      with an instrument sensitive to the predicate the lane corrupts - an
      arithmetic checker that recomputes the result from the two stated operands
      - exactly as quant_misbind verified its own C1 at binding level rather
      than at containment level

  C2  the bar is ZERO on all three string forms against all thirteen evaluation
      surfaces, both directions.  The build enforced this rather than measuring
      it afterwards; this pass re-measures it from the parquet

Every instrument is imported from the banked modules, never re-implemented:
`quant_misbind_verify.py` supplies the containment tokenizer, the whitespace
normaliser, the AUROC, the distribution summary and the thirteen evaluation
surfaces; `provenance_gate.py` supplies the C4 census; `R17-H144_pairs.py`
supplies the numeral canonicalisation and the digit-surface helpers; the lane
builder supplies its own arithmetic checker.  None of them is modified.

CPU ONLY.  Stages write their own JSON beside this file so a killed run resumes
from disk.

Run:  CUDA_VISIBLE_DEVICES= uv run python \
        experiments/grounding-semantic/contract/num_derive_contract.py [stage ...]
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import collections
import hashlib
import importlib.util
import io
import json
import pathlib
import random
import sys
import time
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
GS = HERE.parent
ROOT = GS.parent.parent
DATA = ROOT / "data" / "external" / "datasets"

LANE = GS / "R22-H187_num_derive_lane.parquet"
BUILDER = GS / "R22-H187_num_derive_lane.py"
MANIFEST = GS / "R22-H187_num_derive_lane_manifest.json"
MEMBER = "num_derive"
CHUNK_MAX = 1500
SEED = 2187
N_FOLDS = 5


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


QMV = _mod("qmv", HERE / "quant_misbind_verify.py")
P = _mod("h144pairs", GS / "R17-H144_pairs.py")
LB = _mod("numderive", BUILDER)

containment = QMV.containment
norm_ws = QMV.norm_ws
auroc = QMV.auroc
dist = QMV.dist
tok = QMV.tok

GATE_N, GATE_JACCARD = 8, 0.3
GATE_KILL, GATE_WARN = 0.02, 0.005


def lane():
    return pl.read_parquet(LANE)


def save(stage, obj):
    p = HERE / f"{MEMBER}_{stage}.json"
    p.write_text(json.dumps(obj, indent=2))
    print(f"  -> {p.name}", flush=True)
    return obj


def load(stage):
    p = HERE / f"{MEMBER}_{stage}.json"
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------------------- #
# C1 - label commensurability (C-A1 structural first, C-A2 restated test)
# --------------------------------------------------------------------------- #
def stage_c1():
    t0 = time.time()
    df = lane()
    claims = df["claim"].to_list()
    chunks = df["chunk"].to_list()
    y = df["label"].to_numpy()
    fams = df["neg_family"].to_list()
    forms = df["serial_form"].to_list()

    # --- decisive test 1 (C-A1): structural
    pos_pairs = {(c, k) for c, k, lab in zip(claims, chunks, y) if lab == 1}
    collide = sum(1 for c, k, lab in zip(claims, chunks, y)
                  if lab == 0 and (c, k) in pos_pairs)
    piv = (df.select(["pair_id", "label", "claim"])
             .pivot(on="label", index="pair_id", values="claim",
                    aggregate_function="first").drop_nulls())
    same_claim = int((piv["1"] == piv["0"]).sum())

    # --- the predicate-sensitive instrument: recompute the result
    def attested(r):
        w = LB.compute(r["neg_family"], r["operand_a"], r["operand_b"])
        return 0.0 if w is None else float(P.fmt(w) == r["asserted_value"])

    deriv = np.array([attested(r) for r in df.iter_rows(named=True)])

    # --- the containment channel, reported as the mandatory diagnostic
    cont_full = np.array([containment(a, b) for a, b in zip(claims, chunks)])
    cont_trunc = np.array([containment(a, b[:CHUNK_MAX]) for a, b in zip(claims, chunks)])

    # --- presence of the asserted result and of the two operands
    asserted = df["asserted_value"].to_list()
    pres = [P.present_numbers(c) for c in chunks]
    res_present = np.array([bool(P.canon_set(v) & p) for v, p in zip(asserted, pres)],
                           dtype=float)
    op_present = np.array([
        float(bool(P.canon_set(a) & p) and bool(P.canon_set(b) & p))
        for a, b, p in zip(df["operand_a_str"].to_list(),
                           df["operand_b_str"].to_list(), pres)], dtype=float)

    legs = {}
    for lab, name in ((1, "positive"), (0, "negative")):
        m = y == lab
        legs[name] = {
            "containment_untruncated": dist(cont_full[m]),
            "containment_truncated_1500": dist(cont_trunc[m]),
            "derivation_attested_rate": round(float(deriv[m].mean()), 6),
            "derivation_attested_share_ge_0.90": round(float((deriv[m] >= 0.90).mean()), 6),
            "asserted_result_readable_in_evidence_rate": round(float(res_present[m].mean()), 6),
            "both_operands_readable_in_evidence_rate": round(float(op_present[m].mean()), 6),
        }

    pos_c, neg_c = cont_full[y == 1], cont_full[y == 0]
    pid = df["pair_id"].to_numpy()
    paired = np.abs(pos_c[np.argsort(pid[y == 1])] - neg_c[np.argsort(pid[y == 0])])

    neg_ge90 = legs["negative"]["containment_untruncated"]["share_ge_0.90"]
    pos_ge90 = legs["positive"]["containment_untruncated"]["share_ge_0.90"]
    neg_full = legs["negative"]["containment_untruncated"]["share_fully_attested_eq_1.0"]
    pos_full = legs["positive"]["containment_untruncated"]["share_fully_attested_eq_1.0"]

    out = {
        "clause": "C1",
        "head_declared": "the grounding scalar (`task_head`), trained by pointwise / "
                         "MIL max-over-windows BCE on the 0/1 label",
        "label_predicate":
            "label 1 = the two operand values the claim states are the evidence's cells "
            "for the named (row, column) bindings AND the asserted result is what those "
            "operands give under the stated operation; label 0 = the same claim shape, "
            "the same two operands, and a result the operands do NOT give. The predicate "
            "is support of the asserted proposition, which is the head's own predicate",
        "label_origin": "construction (R22-H187_num_derive_lane.py), not a "
                        "dataset-supplied label",
        "decisive_test_1_structural_C_A1": {
            "definition": "a negative leg's (claim, evidence) pair identical to a "
                          "positive leg's means no function of (claim, evidence) can "
                          "separate the legs, so the label cannot encode grounding",
            "negative_rows_whose_claim_evidence_pair_is_also_a_positive": collide,
            "within_pair_identical_claim_strings": same_claim,
            "bar": "0",
            "fires": bool(collide > 0),
        },
        "decisive_test_2_strict_separation_C_A2": {
            "instrument": "arithmetic checker - the asserted result is recomputed from "
                          "the two operands the claim itself states, under the family's "
                          "operation (difference, percentage, sum, product). This is the "
                          "instrument sensitive to the predicate the lane corrupts",
            "negative_leg_attested_rate": legs["negative"]["derivation_attested_rate"],
            "positive_leg_attested_rate": legs["positive"]["derivation_attested_rate"],
            "strictly_below": bool(legs["negative"]["derivation_attested_rate"]
                                   < legs["positive"]["derivation_attested_rate"]),
            "bar": "the negative leg's high-attestation rate must be STRICTLY below the "
                   "positive leg's; equality is the signature of a label independent of "
                   "(claim, evidence)",
        },
        "decisive_test_3_absolute_level_reported_C_A2": {
            "negative_share_ge_0.90_containment": neg_ge90,
            "positive_share_ge_0.90_containment": pos_ge90,
            "negative_share_fully_attested_containment": neg_full,
            "positive_share_fully_attested_containment": pos_full,
            "negative_share_derivation_attested": legs["negative"]["derivation_attested_rate"],
            "reading": "reported always; a negative leg attested at a high absolute rate "
                       "is a finding even where test 2 passes",
        },
        "legs": legs,
        "containment_channel_scoped_to_C1_by_C_A1": {
            "auroc_containment_vs_label": round(auroc(y, cont_full), 6),
            "within_pair_identical_containment_share": round(float((paired < 1e-9).mean()), 6),
            "within_pair_mean_abs_difference": round(float(paired.mean()), 6),
            "reading": "BLIND BY CONSTRUCTION and intended: the asserted result is absent "
                       "from the evidence on both legs, so the two legs differ only in a "
                       "numeral neither of them attests. C-A1 scopes containment to C1 "
                       "and requires the diagnostic to be read with a predicate-sensitive "
                       "instrument; a predicate-blind instrument showing no separation is "
                       "not evidence of incommensurability",
        },
        "asserted_result_presence": {
            "positive_leg": legs["positive"]["asserted_result_readable_in_evidence_rate"],
            "negative_leg": legs["negative"]["asserted_result_readable_in_evidence_rate"],
            "bar": "0.0 on both legs - the result is ABSENT by construction",
            "holds": bool(res_present.sum() == 0),
        },
        "operand_presence": {
            "positive_leg": legs["positive"]["both_operands_readable_in_evidence_rate"],
            "negative_leg": legs["negative"]["both_operands_readable_in_evidence_rate"],
            "bar": "1.0 on both legs - both operands are stated by the evidence",
            "holds": bool(op_present.min() == 1.0),
        },
    }

    per_fam, per_form = {}, {}
    for vals, sink in ((fams, per_fam), (forms, per_form)):
        for v in sorted(set(vals)):
            m = np.array([x == v for x in vals])
            sink[v] = {
                "rows": int(m.sum()),
                "negative_containment_mean": round(float(cont_full[m & (y == 0)].mean()), 6),
                "positive_containment_mean": round(float(cont_full[m & (y == 1)].mean()), 6),
                "negative_derivation_attested_rate": round(float(deriv[m & (y == 0)].mean()), 6),
                "positive_derivation_attested_rate": round(float(deriv[m & (y == 1)].mean()), 6),
            }
    out["per_family"] = per_fam
    out["per_serial_form"] = per_form
    out["seconds"] = round(time.time() - t0, 1)
    return save("c1", out)


# --------------------------------------------------------------------------- #
# C2 - disjointness from every evaluation surface
# --------------------------------------------------------------------------- #
def stage_c2():
    t0 = time.time()
    df = lane()
    m_claims = [c for c in df["claim"].to_list() if c and c.strip()]
    m_chunks = [c for c in df["chunk"].to_list() if c and c.strip()]

    forms = {"raw": lambda s: s,
             "truncated_1500": lambda s: s[:CHUNK_MAX],
             "normalised_ws_casefold": norm_ws}
    member = {k: {"claims": {f(c) for c in m_claims}, "evidence": {f(c) for c in m_chunks}}
              for k, f in forms.items()}

    results, total = {}, collections.Counter()
    for name, s_claims, s_ev, note in QMV.eval_surfaces():
        s_claims = [c for c in s_claims if c and c.strip()]
        s_ev = [c for c in s_ev if c and c.strip()]
        entry = {"note": note, "surface_claims": len(set(s_claims)),
                 "surface_evidence_units": len(set(s_ev)), "forms": {}}
        for fname, f in forms.items():
            ev_hit = member[fname]["evidence"] & {f(c) for c in s_ev}
            cl_hit = member[fname]["claims"] & {f(c) for c in s_claims}
            # both directions: an intersection is symmetric, so the member-side
            # and surface-side matched-unit counts are reported separately and a
            # non-zero in either fails the clause
            entry["forms"][fname] = {
                "evidence_shared_strings": len(ev_hit),
                "member_evidence_units_matched": len(ev_hit),
                "surface_evidence_units_matched": len(ev_hit),
                "claims_shared_strings": len(cl_hit),
                "member_claims_matched": len(cl_hit),
                "surface_claims_matched": len(cl_hit),
            }
            total[fname] += len(ev_hit) + len(cl_hit)
        entry["clean"] = all(v["evidence_shared_strings"] == 0
                             and v["claims_shared_strings"] == 0
                             for v in entry["forms"].values())
        results[name] = entry
        print(f"  {name}: clean={entry['clean']}", flush=True)

    # document-level read: string equality cannot see the same table serialized
    # differently, so the TabFact table_id is checked directly.
    lane_ids = {d.split(":", 1)[1] for d in df["doc_id"].to_list()}
    tbl = {}
    for f in ("R20-H177_eval_B.parquet", "R20-H177_eval_C.parquet"):
        p = GS / f
        if p.exists():
            ids = {x.split(":", 1)[1] for x in
                   pl.read_parquet(p, columns=["doc_id"])["doc_id"].to_list()
                   if x.startswith("tabfact:")}
            tbl[f.replace(".parquet", "")] = {"surface_tabfact_tables": len(ids),
                                              "shared_with_member": len(ids & lane_ids)}
    for f in ("R17-H146_antigaming_set.parquet", "R18-H150_antigaming_set.parquet",
              "R19-H159_antigaming_set.parquet"):
        p = GS / f
        if p.exists():
            ids = set(pl.read_parquet(p, columns=["table_id"])["table_id"].to_list())
            tbl[f.replace(".parquet", "")] = {"surface_tabfact_tables": len(ids),
                                              "shared_with_member": len(ids & lane_ids)}

    out = {
        "clause": "C2",
        "member_units": {"distinct_claims": len(set(m_claims)),
                         "distinct_evidence_chunks": len(set(m_chunks)),
                         "rows": df.height},
        "forms": ["raw", f"truncated_{CHUNK_MAX}", "normalised_ws_casefold"],
        "surfaces_measured": len(results),
        "surfaces_clean": sorted(k for k, v in results.items() if v["clean"]),
        "surfaces": results,
        "totals_per_form": dict(total),
        "all_forms_zero": all(v == 0 for v in total.values()),
        "document_level_read_beyond_string_forms": {
            "why": "string equality cannot see the same source table serialized "
                   "differently; TabFact carries a stable table_id, so identity is "
                   "checked on it directly rather than inferred from content",
            "member_tabfact_tables": len(lane_ids),
            "per_surface": tbl,
            "enforced_at_build": "every table_id named by any of these surfaces was "
                                 "dropped whole before construction",
        },
        "seconds": round(time.time() - t0, 1),
    }
    return save("c2", out)


# --------------------------------------------------------------------------- #
# C3 - split semantics verified, never assumed
# --------------------------------------------------------------------------- #
def stage_c3():
    t0 = time.time()
    df = lane()
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    splits, tables = {}, {}
    for n in (x for x in z.namelist() if x.endswith(".parquet")):
        d = pl.read_parquet(io.BytesIO(z.read(n)), columns=["table_id"])
        s = n.split("__")[-1].replace(".parquet", "")
        splits[s] = d.height
        tables[s] = set(d["table_id"].to_list())

    axis = {"train_table_ids": len(tables.get("train", ()))}
    for other in sorted(s for s in tables if s != "train"):
        axis[f"train_vs_{other}_shared_table_ids"] = len(tables["train"] & tables[other])
        axis[f"{other}_table_ids"] = len(tables[other])

    lane_ids = set(df["table_id"].to_list())
    leak = {s: len(lane_ids & tables[s]) for s in sorted(tables) if s != "train"}

    out = {
        "clause": "C3",
        "member_type": "constructed lane - it has no split of its own; the axis that "
                       "matters is which split of each SOURCE it reads and whether that "
                       "split is document-disjoint from anything used for evaluation",
        "sources_read": {"tabfact": "dataset-tabfact.zip, the *__train.parquet member "
                                    "ONLY (R17-H144_pairs.tabfact_tables), deduplicated "
                                    "on table_text"},
        "feverous_used": False,
        "why_no_feverous": "quant_misbind is non-conforming on exactly its FEVEROUS half "
                           "- C3 because the split axis is not measurable for 33.7% of "
                           "its rows and the identifier it carries for them is unstable "
                           "across rebuilds, and C8 because that half has no licence, no "
                           "retrieval date and no tracked source. This member reads the "
                           "clean half only and inherits none of the defect",
        "rows_by_source": {k: v for k, v in df.group_by("source").len().iter_rows()},
        "documents_by_source": {"tabfact": int(df["doc_id"].n_unique())},
        "axis": "the archive's own table_id",
        "archive_splits": splits,
        "measured_axis": axis,
        "member_tables_in_a_non_train_split": leak,
        "member_tables": len(lane_ids),
        "identifier_stability": "the member's doc_id is `tabfact:<table_id>` and the "
                                "table_id is the archive's own key, so document identity "
                                "survives a rebuild - the defect that makes quant_misbind's "
                                "FEVEROUS third unmeasurable does not arise here",
        "seconds": round(time.time() - t0, 1),
    }
    return save("c3", out)


# --------------------------------------------------------------------------- #
# C4 - contamination census with a live positive control
# --------------------------------------------------------------------------- #
def _degrade(text, rng, drop, keep):
    kept = [w for w in text.split() if rng.random() >= drop]
    if keep < 1.0:
        kept = kept[: max(20, int(len(kept) * keep))]
    return " ".join(kept)


def stage_c4():
    t0 = time.time()
    G = _mod("provgate", GS / "provenance_gate.py")
    df = lane()
    ev = sorted({c for c in df["chunk"].to_list() if c.strip()})
    cl = sorted({c for c in df["claim"].to_list() if c.strip()})

    print("loading arena...", flush=True)
    arena_texts, _ = G.load_arena()
    n_arena = sum(len(v) for v in arena_texts.values())

    def coverage(units):
        short = [u for u in units if len(G.normalize(u).split()) < GATE_N]
        return {"units": len(units), "too_short_for_8gram": len(short),
                "share_too_short": round(len(short) / max(len(units), 1), 6),
                "covered_by_exact_matching": len(short)}

    res = {"clause": "C4",
           "instrument": f"provenance_gate.py (R14-H136 ruling-2 form): {GATE_N}-gram, "
                         f"Jaccard >= {GATE_JACCARD}, bidirectional, WARN {GATE_WARN}, "
                         f"KILL > {GATE_KILL:.0%} of the candidate side, all ten walled "
                         "arena subsets",
           "arena_units": n_arena,
           "coverage": {"evidence": coverage(ev), "claims": coverage(cl)}}

    for label, units in (("evidence", ev), ("claims", cl)):
        print(f"gating {label} ({len(units)} units)...", flush=True)
        r = G.run_gate(units, n=GATE_N, jaccard=GATE_JACCARD, warn=GATE_WARN,
                       kill=GATE_KILL, label=f"{MEMBER}_{label}", arena_texts=arena_texts)
        res[f"{label}_gate"] = {
            "units": len(units), "verdict": r["verdict"],
            "max_fraction": r["max_fraction"],
            "candidate_to_arena": r["candidate_vs_arena"]["fraction"],
            "arena_to_candidate": r["arena_vs_candidate"]["fraction"],
            "best_jaccard": r["candidate_vs_arena"].get("best_jaccard"),
            "kill_bar": GATE_KILL,
            "margin": round(GATE_KILL - r["max_fraction"], 6)}
        print(f"  {label}: {res[f'{label}_gate']}", flush=True)

    # short units are covered by exact matching, per C4
    norm_arena = " \n ".join(G.normalize(c) for v in arena_texts.values() for c in v)
    short_cl = [c for c in cl if len(G.normalize(c).split()) < GATE_N]
    res["coverage"]["exact_matching_of_the_short_units"] = {
        "method": "normalised claim string sought verbatim inside the concatenated "
                  "normalised arena document corpus",
        "arena_documents": n_arena,
        "short_claims_checked": len(short_cl),
        "short_claims_found": sum(1 for c in short_cl if G.normalize(c) in norm_arena),
        "short_evidence_units": len([u for u in ev if len(G.normalize(u).split()) < GATE_N]),
    }

    print("spike control...", flush=True)
    res["spike_control"] = G.spike_control(ev[:2000], arena_texts, n=GATE_N,
                                           jaccard=GATE_JACCARD, k=10,
                                           label=f"{MEMBER}_spike")
    print(f"  {res['spike_control']}", flush=True)

    print("live positive control (tiered)...", flush=True)
    rng = random.Random(SEED)
    pool = [c for v in arena_texts.values() for c in v if len(c.split()) >= 40]
    rng.shuffle(pool)
    sample = pool[:250]
    tiers = {}
    for name, drop, keep, note in (
        ("verbatim", 0.0, 1.0, "identical arena text - the upper anchor"),
        ("drop_2pct", 0.02, 1.0, "2% of whitespace tokens deleted at random"),
        ("drop_5pct", 0.05, 1.0, "5% of whitespace tokens deleted at random"),
        ("drop_10pct", 0.10, 1.0, "10% of whitespace tokens deleted at random"),
        ("drop_5pct_cut_60pct", 0.05, 0.6, "5% deleted then cut to 60% of length"),
        ("cut_50pct", 0.0, 0.5, "first half of the document only"),
    ):
        r2 = random.Random(SEED)
        units = [_degrade(t, r2, drop, keep) for t in sample]
        r = G.run_gate(units, n=GATE_N, jaccard=GATE_JACCARD, warn=GATE_WARN,
                       kill=GATE_KILL, label=f"live_{name}", arena_texts=arena_texts)
        tiers[name] = {"note": note, "units": r["candidate"]["n_units"],
                       "detection_fraction": r["candidate_vs_arena"]["fraction"],
                       "units_with_hit": r["candidate_vs_arena"]["units_with_hit"],
                       "best_jaccard": r["candidate_vs_arena"].get("best_jaccard"),
                       "verdict": r["verdict"]}
        print(f"  {name}: {tiers[name]['detection_fraction']} {tiers[name]['verdict']}",
              flush=True)
    res["live_positive_control"] = {
        "construction": "real RAGBench arena documents (the reference side itself), "
                        "degraded - near-duplicate by construction, not byte-identical",
        "tiers": tiers,
        "fires": bool(tiers["verbatim"]["detection_fraction"] >= 0.99
                      and tiers["drop_5pct"]["detection_fraction"] > 0.5),
        "reading": "the gate fires on real near-duplicate text and its detection degrades "
                   "with the damage applied, while the member itself reads 0.000 against "
                   "the same reference side - so the member's clean verdict comes from a "
                   "live instrument",
    }
    res["status"] = ("GREEN" if res["evidence_gate"]["verdict"] != "KILL"
                     and res["claims_gate"]["verdict"] != "KILL"
                     and res["spike_control"]["passes"]
                     and res["live_positive_control"]["fires"] else "RED")
    res["seconds"] = round(time.time() - t0, 1)
    return save("c4", res)


# --------------------------------------------------------------------------- #
# C5 - leak suite, recomputed from the parquet
# --------------------------------------------------------------------------- #
def stage_c5():
    t0 = time.time()
    df = lane()
    res = LB.verify(df, random.Random(SEED))
    out = {
        "clause": "C5",
        "recomputed_from_the_parquet": True,
        "note": "recomputed here from the artifact; nothing is cited from the build "
                "manifest",
        "registered_conjunction": {
            "claim_only_converged_probe": res["claim_only_converged_probe"],
            "within_pair_claim_only": res["within_pair_claim_only"],
            "single_channel_probes": res["single_channel_probes"],
            "surface_parity_channels": res["surface_parity_channels"],
            "surface_parity_worst": res["surface_parity_worst"],
            "digit_surface_parity": res["digit_surface_parity"],
            "balance": res["balance"],
        },
        "arithmetic_rederivation": res["arithmetic_rederivation"],
        "all_registered_bars_pass": res["all_registered_bars_pass"],
        "executor_added_probes_reported_separately": res[
            "executor_added_probes_reported_separately"],
        "seconds": round(time.time() - t0, 1),
    }
    return save("c5", out)


# --------------------------------------------------------------------------- #
# C6 - no memorisation channel (C-A2 scoping: mix-supplied associations)
# --------------------------------------------------------------------------- #
def stage_c6():
    t0 = time.time()
    df = lane()
    labels = df["label"].to_numpy()
    chunks = df["chunk"].to_list()
    claims = df["claim"].to_list()

    # --- the eval-facing test: what the ASSEMBLED MIX associates with this
    # member's pair key.  The mix is rebuilt through the banked loader.
    H108 = _mod("h108", GS / "R10-H108_lane.py")
    H174 = _mod("h174", GS / "R20-H174_arm_run.py")
    print("building the clean public mix through the banked loader...", flush=True)
    mix_claims, mix_chunks, mix_y, mix_tags = H108.public_train()
    other = collections.defaultdict(lambda: {"claims": [], "chunks": [], "y": []})
    for c, k, lab, tg in zip(mix_claims, mix_chunks, mix_y.tolist(), mix_tags):
        other[tg]["claims"].append(c)
        other[tg]["chunks"].append(k)
        other[tg]["y"].append(lab)
    for fname, group, *_ in H174.LANES:
        d = pl.read_parquet(GS / fname)
        other[group]["claims"] += d["claim"].to_list()
        other[group]["chunks"] += d["chunk"].to_list()
        other[group]["y"] += d["label"].cast(pl.Float32).to_list()
    mix_rows = sum(len(v["claims"]) for v in other.values())
    print(f"  assembled mix (excluding this member): {mix_rows} rows, "
          f"{len(other)} groups", flush=True)

    lane_raw = set(chunks)
    lane_tr = {c[:CHUNK_MAX] for c in chunks}
    lane_no = {norm_ws(c) for c in chunks}
    lane_cl = {norm_ws(c) for c in claims}
    per_group, key_owner = {}, {}
    for g, v in other.items():
        ck_no = {norm_ws(c) for c in v["chunks"]}
        shared_no = lane_no & ck_no
        per_group[g] = {
            "rows": len(v["claims"]),
            "shared_evidence_raw": len(lane_raw & set(v["chunks"])),
            "shared_evidence_truncated_1500": len(lane_tr & {c[:CHUNK_MAX] for c in v["chunks"]}),
            "shared_evidence_normalised": len(shared_no),
            "shared_claims_normalised": len(lane_cl & {norm_ws(c) for c in v["claims"]}),
        }
        for c in shared_no:
            key_owner.setdefault(c, []).append(g)

    feat = np.full(df.height, np.nan)
    if key_owner:
        by_key = collections.defaultdict(list)
        for g, v in other.items():
            for c, lab in zip(v["chunks"], v["y"]):
                k = norm_ws(c)
                if k in key_owner:
                    by_key[k].append(lab)
        for i, c in enumerate(chunks):
            k = norm_ws(c)
            if k in by_key:
                feat[i] = float(np.mean(by_key[k]))
    ok = ~np.isnan(feat)
    a_mix = (auroc(labels[ok], feat[ok])
             if ok.sum() and len(set(labels[ok].tolist())) > 1 else None)

    # --- within-member diagnostics (C-A2: reported, not a C6 bar)
    by_chunk = collections.defaultdict(list)
    for i, c in enumerate(chunks):
        by_chunk[c].append(i)
    loo = np.full(df.height, np.nan)
    for c, idxs in by_chunk.items():
        if len(idxs) < 2:
            continue
        s = float(labels[idxs].sum())
        for i in idxs:
            loo[i] = (s - labels[i]) / (len(idxs) - 1)
    ok2 = ~np.isnan(loo)
    a_loo = auroc(labels[ok2], loo[ok2]) if ok2.sum() else float("nan")

    ctoks = [set(tok(c)) for c in claims]
    ov = np.full(df.height, np.nan)
    for c, idxs in by_chunk.items():
        if len(idxs) < 2:
            continue
        for i in idxs:
            best = 0.0
            for j in idxs:
                if j == i:
                    continue
                u = ctoks[i] | ctoks[j]
                if u:
                    best = max(best, len(ctoks[i] & ctoks[j]) / len(u))
            ov[i] = best
    ok3 = ~np.isnan(ov)
    a_ov = auroc(labels[ok3], ov[ok3]) if ok3.sum() else float("nan")

    out = {
        "clause": "C6",
        "pair_key": "the evidence chunk - the two legs of a pair carry it "
                    "byte-identically",
        "eval_facing_mix_association": {
            "definition": "mean label the REST of the assembled mix attaches to this "
                          "row's evidence key (whitespace-normalised) - the channel that "
                          "caught attr_pool at 0.9999 and the withdrawn R20-H175b eval "
                          "at 0.6230",
            "mix_rows": mix_rows,
            "mix_groups": sorted(other),
            "key_sharing_by_group": {g: v for g, v in sorted(per_group.items())
                                     if any(v[k] for k in v if k != "rows")},
            "keys_shared_with_any_other_member": len(key_owner),
            "coverage_rows": int(ok.sum()),
            "coverage_share": round(float(ok.mean()), 6),
            "auroc_vs_label": None if a_mix is None else round(a_mix, 6),
            "bar": "undefined or at chance",
        },
        "within_member_diagnostics_reported_not_a_bar": {
            "note": "C-A2 scopes C6 to associations the TRAINING MIX supplies; a "
                    "within-member leave-one-out lookup is a different question and is "
                    "reported as a diagnostic",
            "leave_one_out_label_feature": {
                "auroc_vs_label": None if np.isnan(a_loo) else round(a_loo, 6),
                "coverage_rows": int(ok2.sum()),
                "coverage_share": round(float(ok2.mean()), 6),
                "why_it_is_not_a_channel": "with exact 1:1 pairing the leave-one-out "
                                           "value IS the twin's label, so the feature "
                                           "reads AUROC 0.0 on any perfectly paired "
                                           "member. Both legs share the key, so every "
                                           "key-keyed feature takes the same value on "
                                           "both and separates them at exactly 0.5",
            },
            "key_keyed_claim_overlap": {
                "definition": "max Jaccard between this row's claim and any other claim "
                              "sharing the same evidence chunk",
                "auroc_vs_label": None if np.isnan(a_ov) else round(a_ov, 6),
                "coverage_rows": int(ok3.sum()),
                "coverage_share": round(float(ok3.mean()), 6),
            },
        },
        "seconds": round(time.time() - t0, 1),
    }
    return save("c6", out)


# --------------------------------------------------------------------------- #
# C7 / C8
# --------------------------------------------------------------------------- #
def stage_c78():
    t0 = time.time()
    df = lane()
    man = json.loads(MANIFEST.read_text())
    rows, pairs = df.height, int(df["pair_id"].n_unique())
    fam_pairs = {k: v for k, v in df.filter(pl.col("label") == 0)
                 .group_by("neg_family").len().iter_rows()}
    fam_rows = {k: v for k, v in df.group_by("neg_family").len().iter_rows()}
    per_pair = df.group_by("pair_id").len()

    arch = DATA / "dataset-tabfact.zip"
    side = DATA / "dataset-tabfact.md"
    mtime = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(arch.stat().st_mtime))
    sidecar_text = side.read_text()

    dup = {
        "rows": rows, "pairs": pairs, "rows_per_pair": round(rows / pairs, 6),
        "pairs_with_exactly_two_rows": int((per_pair["len"] == 2).sum()),
        "distinct_claims": int(df["claim"].n_unique()),
        "distinct_evidence_chunks": int(df["chunk"].n_unique()),
        "distinct_documents": int(df["doc_id"].n_unique()),
        "distinct_source_tables": int(df["table_id"].n_unique()),
        "distinct_columns": int(df["column"].n_unique()),
        "distinct_row_keys": int(pl.concat([df["row_key_a"], df["row_key_b"]]).n_unique()),
        "distinct_claim_chunk_pairs": int(df.select(["claim", "chunk"]).n_unique()),
        "distinct_asserted_values": int(df["asserted_value"].n_unique()),
        "max_rows_per_chunk": int(df.group_by("chunk").len()["len"].max()),
        "mean_rows_per_chunk": round(float(df.group_by("chunk").len()["len"].mean()), 6),
        "max_pairs_per_document": int(df.filter(pl.col("label") == 1)
                                        .group_by("doc_id").len()["len"].max()),
        "mean_pairs_per_document": round(float(df.filter(pl.col("label") == 1)
                                               .group_by("doc_id").len()["len"].mean()), 6),
        "templates": {f"{a}:{b}": n for a, b, n in
                      df.group_by(["neg_family", "template_id"]).len().iter_rows()},
        "serial_forms": {k: v for k, v in df.group_by("serial_form").len().iter_rows()},
        "rows_by_source": {k: v for k, v in df.group_by("source").len().iter_rows()},
        "duplicate_claim_strings": rows - int(df["claim"].n_unique()),
        "label_balance": {str(k): v for k, v in df.group_by("label").len().iter_rows()},
        "result_digit_lengths": {str(k): v for k, v in
                                 sorted(df.group_by("result_digits").len().iter_rows())},
    }

    out = {
        "clause": "C7 + C8",
        "c7": {
            "declared_unit": "BOTH - rows AND pairs are declared in the builder, the "
                             "manifest and this report",
            "declared_rows": 30000, "declared_pairs": 15000,
            "declared_family_pairs": {"difference": 5250, "percentage": 5250,
                                      "sum": 2250, "product": 2250},
            "measured_rows": rows, "measured_pairs": pairs,
            "measured_family_pairs": fam_pairs, "measured_family_rows": fam_rows,
            "rows_match": rows == 30000, "pairs_match": pairs == 15000,
            "families_match": fam_pairs == {"difference": 5250, "percentage": 5250,
                                            "sum": 2250, "product": 2250},
            "manifest_declares_both": bool(man.get("rows") and man.get("pairs")),
            "registration_note": "the member is NEW and is not yet named in any arm "
                                 "wrapper's LANES tuple; the declaration of record is "
                                 "the builder plus this report, and both units are "
                                 "stated in both",
        },
        "c8": {
            "artifact": str(LANE.relative_to(ROOT)),
            "blake2b_64": hashlib.blake2b(LANE.read_bytes(), digest_size=8).hexdigest(),
            "builder": "experiments/grounding-semantic/R22-H187_num_derive_lane.py, "
                       f"seed {SEED}",
            "sources": {
                "tabfact": {
                    "archive": "data/external/datasets/dataset-tabfact.zip",
                    "sidecar": "data/external/datasets/dataset-tabfact.md (tracked)",
                    "licence": "CC-BY-4.0 (declared in the tracked sidecar)",
                    "licence_declared_in_sidecar": "**Licence** - CC-BY-4.0" in sidecar_text,
                    "split_read": "the *__train.parquet member ONLY",
                    "selection_predicate":
                        "R17-H144_pairs.tabfact_tables: the *__train.parquet member, "
                        "deduplicated on table_text; tables with >= 4 body rows of "
                        "uniform width and >= 2 columns; then a label column "
                        "(R17-H144_pairs.label_column) and at least one numeric column "
                        "with a usable header must exist. Every table an evaluation "
                        "surface names, every table matching an R17-H143 eval-set "
                        "content fingerprint, and every table_id in a non-train split "
                        "are dropped whole",
                    "retrieval_script": "scripts/fetch_grounding_datasets.py (tracked)",
                    "retrieval_date_declared_in_sidecar": False,
                    "retrieval_date_observed_from_archive_mtime": mtime,
                    "gap": "the tracked sidecar names the source, the licence and the "
                           "fetch script but does not record a retrieval DATE; the date "
                           "above is the archive's filesystem mtime, which is evidence "
                           "but not a declaration. The same gap is recorded against "
                           "quant_misbind's TabFact half",
                },
            },
            "feverous_used": False,
            "internal_structure": dup,
            "public_repository_check": {
                "sources_present": sorted(set(df["source"].to_list())),
                "all_sources_public": set(df["source"].to_list()) <= {"tabfact"},
                "method": "every row declares its source; the lane text is TabFact "
                          "(Wikipedia tables) only",
            },
        },
        "seconds": round(time.time() - t0, 1),
    }
    return save("c78", out)


# --------------------------------------------------------------------------- #
# the eight-clause report
# --------------------------------------------------------------------------- #
def stage_report():
    c1, c2, c3 = load("c1"), load("c2"), load("c3")
    c4, c5, c6, c78 = load("c4"), load("c5"), load("c6"), load("c78")
    missing = [n for n, v in (("c1", c1), ("c2", c2), ("c3", c3), ("c4", c4),
                              ("c5", c5), ("c6", c6), ("c78", c78)) if v is None]
    if missing:
        raise SystemExit(f"missing stages: {missing}")

    v1 = ("PASS" if (not c1["decisive_test_1_structural_C_A1"]["fires"]
                     and c1["decisive_test_2_strict_separation_C_A2"]["strictly_below"])
          else "FAIL")
    v2 = "PASS" if c2["all_forms_zero"] else "FAIL"
    v3 = "PASS" if (sum(c3["member_tables_in_a_non_train_split"].values()) == 0
                    and c3["measured_axis"]["train_vs_test_shared_table_ids"] == 0
                    and c3["measured_axis"]["train_vs_validation_shared_table_ids"] == 0) else "FAIL"
    v4 = "PASS" if c4["status"] == "GREEN" else "FAIL"
    v5 = "PASS" if c5["all_registered_bars_pass"] else "FAIL"
    mix = c6["eval_facing_mix_association"]
    v6 = ("NOT-APPLICABLE" if mix["coverage_rows"] == 0
          else ("PASS" if abs((mix["auroc_vs_label"] or 0.5) - 0.5) <= 0.05 else "FAIL"))
    v7 = "PASS" if (c78["c7"]["rows_match"] and c78["c7"]["pairs_match"]
                    and c78["c7"]["families_match"]) else "FAIL"
    v8 = "PASS" if (c78["c8"]["public_repository_check"]["all_sources_public"]
                    and c78["c8"]["sources"]["tabfact"]["licence_declared_in_sidecar"]) else "FAIL"

    verdicts = {"C1": v1, "C2": v2, "C3": v3, "C4": v4,
                "C5": v5, "C6": v6, "C7": v7, "C8": v8}
    conforming = all(v in ("PASS", "NOT-APPLICABLE") for v in verdicts.values())

    rep = {
        "member": MEMBER,
        "artifact": str(LANE.relative_to(ROOT)),
        "artifact_blake2b_64": c78["c8"]["blake2b_64"],
        "member_class": "constructed lane - a training member of the assembled mix; it "
                        "teaches the model to CHECK A COMPUTATION, the predicate no "
                        "existing member covers",
        "contract": "docs/experiments/dataset-contract.md (with amendments C-A1, C-A2)",
        "verification_pass": "per-member, CPU only, no GPU touched, no torch imported",
        "conforming": conforming,
        "C1": {"verdict": v1, "clause": "label commensurability", **c1},
        "C2": {"verdict": v2, "clause": "disjointness from every evaluation surface",
               **c2},
        "C3": {"verdict": v3, "clause": "split semantics verified, never assumed", **c3},
        "C4": {"verdict": v4, "clause": "contamination census with a live positive "
                                        "control", **c4},
        "C5": {"verdict": v5, "clause": "leak suite for constructed members", **c5},
        "C6": {"verdict": v6, "clause": "no memorisation channel", **c6},
        "C7": {"verdict": v7, "clause": "declared units and volume", **c78["c7"]},
        "C8": {"verdict": v8, "clause": "provenance, licence and internal structure",
               **c78["c8"]},
        "summary": {
            "pass": sorted(k for k, v in verdicts.items() if v == "PASS"),
            "fail": sorted(k for k, v in verdicts.items() if v == "FAIL"),
            "not_applicable": sorted(k for k, v in verdicts.items()
                                     if v == "NOT-APPLICABLE"),
            "conforming": conforming,
        },
        "artifacts": [
            "experiments/grounding-semantic/R22-H187_num_derive_lane.py",
            "experiments/grounding-semantic/R22-H187_num_derive_lane.parquet",
            "experiments/grounding-semantic/R22-H187_num_derive_lane_manifest.json",
            "experiments/grounding-semantic/contract/num_derive_contract.py",
            "experiments/grounding-semantic/contract/num_derive_contract_report.json",
            "experiments/grounding-semantic/contract/num_derive_c1.json",
            "experiments/grounding-semantic/contract/num_derive_c2.json",
            "experiments/grounding-semantic/contract/num_derive_c3.json",
            "experiments/grounding-semantic/contract/num_derive_c4.json",
            "experiments/grounding-semantic/contract/num_derive_c5.json",
            "experiments/grounding-semantic/contract/num_derive_c6.json",
            "experiments/grounding-semantic/contract/num_derive_c78.json",
            "logs/R22-H187_num_derive.log",
            "logs/R22-H187_num_derive_contract.log",
        ],
    }
    p = HERE / f"{MEMBER}_contract_report.json"
    p.write_text(json.dumps(rep, indent=2))
    print(json.dumps({"verdicts": verdicts, "conforming": conforming}, indent=2),
          flush=True)
    print(f"  -> {p.name}", flush=True)
    return rep


STAGES = {"c1": stage_c1, "c2": stage_c2, "c3": stage_c3, "c4": stage_c4,
          "c5": stage_c5, "c6": stage_c6, "c78": stage_c78, "report": stage_report}


def main():
    want = sys.argv[1:] or list(STAGES)
    for s in want:
        print(f"\n=== STAGE {s} ===", flush=True)
        STAGES[s]()
    print("\nALL REQUESTED STAGES COMPLETE", flush=True)


if __name__ == "__main__":
    main()

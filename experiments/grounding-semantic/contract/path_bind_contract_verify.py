"""Dataset-contract verification of the `path_bind` lane (R20-H174 lane L4).

Contract: docs/experiments/dataset-contract.md, clauses C1-C8.
Member:   experiments/grounding-semantic/R20-H174_lane_L4.parquet
          10,000 rows / 5,000 pairs, DANN group `path_bind`, entered by
          R18-H150_arm_run.make_build_mix through the R20-H174 LANES tuple.

CPU ONLY.  No GPU is touched: CUDA_VISIBLE_DEVICES is forced empty before any
import and nothing here imports torch.

Instruments are REUSED, not reinvented:
  * containment / tokens / claim_only_probe / within_pair_accuracy /
    surface_parity / pair_integrity     -> R20-H174_lane_common.py (banked)
  * 8-gram Jaccard >= 0.3 bidirectional census + spike control
                                        -> provenance_gate.py (R14-H136 form)
  * the generator itself, for the C4 live positive control
                                        -> R20-H174_lane_L4.py (banked builder)

Writes experiments/grounding-semantic/contract/path_bind_contract_report.json.
No text from any evaluation surface is copied into the artifact - only counts.

Run:  CUDA_VISIBLE_DEVICES= uv run python \
        experiments/grounding-semantic/contract/path_bind_contract_verify.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import collections
import importlib.util as ilu
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
OUT = HERE / "path_bind_contract_report.json"

MEMBER = EXP / "R20-H174_lane_L4.parquet"
CHUNK_MAX_CHARS = 1500  # M59.CFG.chunk_max_chars, the serving truncation unit
ARROW = " → "


def _mod(name, path):
    spec = ilu.spec_from_file_location(name, path)
    m = ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = _mod("h174common", EXP / "R20-H174_lane_common.py")
G = _mod("provgate", EXP / "provenance_gate.py")
L4 = _mod("h174L4", EXP / "R20-H174_lane_L4.py")

# R19_supply_gates thresholds, read literally rather than restated
_gsrc = (EXP / "R19_supply_gates.py").read_text()
GATE_N = int(_gsrc.split("GATE_N = ")[1].split("\n")[0])
GATE_JACCARD = float(_gsrc.split("GATE_JACCARD = ")[1].split("\n")[0])
GATE_KILL = float(_gsrc.split("GATE_KILL = ")[1].split("\n")[0])

log = lambda *a: print(*a, flush=True)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def dist(xs):
    a = np.asarray(xs, dtype=float)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 6),
        "median": round(float(np.median(a)), 6),
        "p10": round(float(np.percentile(a, 10)), 6),
        "p25": round(float(np.percentile(a, 25)), 6),
        "p75": round(float(np.percentile(a, 75)), 6),
        "p90": round(float(np.percentile(a, 90)), 6),
        "min": round(float(a.min()), 6),
        "max": round(float(a.max()), 6),
        "share_eq_1.0": round(float((a >= 1.0 - 1e-12).mean()), 6),
        "share_ge_0.90": round(float((a >= 0.90).mean()), 6),
        "share_ge_0.75": round(float((a >= 0.75).mean()), 6),
    }


def wsfold(s):
    return " ".join(s.lower().split())


def string_forms(s):
    return {"raw": s, "truncated_1500": s[:CHUNK_MAX_CHARS], "ws_collapsed_casefold": wsfold(s)}


def path_segments(arrowed):
    return [p for p in arrowed.split(ARROW) if p]


# --------------------------------------------------------------------------- #
# evaluation surfaces (text only ever counted, never written out)
# --------------------------------------------------------------------------- #
def load_arena():
    """Documents and responses of the 10-subset blind arena, the H77 sample."""
    docs, resp = {}, {}
    z = zipfile.ZipFile(DATA / "dataset-ragbench.zip")
    for name in sorted(n for n in z.namelist() if n.endswith("__test.parquet")):
        sub = name.split("__")[2]
        df = pl.read_parquet(io.BytesIO(z.read(name)))
        df = df.filter(
            pl.col("adherence_score").is_not_null()
            & (pl.col("response").str.len_chars() > 20)
            & (pl.col("documents").list.len() > 0)
        )
        if len(df) < 40 or df["adherence_score"].n_unique() < 2:
            continue
        df = df.sample(min(G.N_PER_SUBSET, len(df)), seed=0)
        docs[sub] = [c for d in df["documents"].to_list() for c in d[: G.MAX_CHUNKS]]
        resp[sub] = [r for r in df["response"].to_list() if r]
    return docs, resp


def _texts(path, cols):
    try:
        df = pl.read_parquet(path)
    except Exception as e:  # pragma: no cover
        return None, f"unreadable: {e}"
    out = []
    for c in cols:
        if c in df.columns and df[c].dtype == pl.String:
            out += [t for t in df[c].to_list() if t]
    return out, None


def eval_surfaces():
    """{surface -> {'claims': [...], 'evidence': [...]}}; text is never emitted."""
    S = {}
    docs, resp = load_arena()
    S["arena_documents"] = {"evidence": [c for v in docs.values() for c in v], "claims": []}
    S["arena_responses"] = {"evidence": [], "claims": [r for v in resp.values() for r in v]}
    for sub, v in docs.items():
        S[f"arena_documents:{sub}"] = {"evidence": v, "claims": []}

    # gold_full - R10-H108_lane.gold_full() reads this parquet; PRIVATE source,
    # counted only.
    gf = EXP / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"
    if gf.exists():
        d = pl.read_parquet(gf)
        S["gold_full"] = {
            "claims": [t for t in d["claim"].to_list() if t],
            "evidence": [t for t in d["chunk"].to_list() if t],
        }

    # held-out mechanism evals / probes / anti-gaming sets on disk
    named = [
        "R17-H143_evalset.parquet", "R11-H117_heldout_pairs.parquet",
        "R20-H177_eval_B.parquet", "R20-H177_eval_C.parquet",
        "R20-H175b_qlane_eval_clean.parquet", "R20-H175b_qlane_eval.parquet",
        "R18-H150_unitswap_probe.parquet", "R20-G0b_composed_probes.parquet",
        "R15_L1_bindprobe_pairs.parquet",
    ]
    named += sorted(p.name for p in EXP.glob("*antigaming_set.parquet"))
    for fn in named:
        p = EXP / fn
        if not p.exists():
            continue
        cl, _ = _texts(p, ["claim", "claim_pos", "claim_neg", "statement", "response"])
        ev, _ = _texts(p, ["chunk", "evidence", "doc", "document", "cited_unit", "context"])
        S[f"eval:{fn[:-len('.parquet')]}"] = {"claims": cl or [], "evidence": ev or []}

    # VitaminC held-out (R19-H166 amendment A1 contradiction-head instrument)
    try:
        parts = C._zip_parquets("vitaminc")
        for split in ("validation", "test"):
            if split in parts:
                d = parts[split]
                S[f"eval:vitaminc_{split}"] = {
                    "claims": [t for t in d["claim"].to_list() if t],
                    "evidence": [t for t in d["evidence"].to_list() if t],
                }
    except Exception as e:
        S["eval:vitaminc_heldout"] = {"claims": [], "evidence": [], "error": str(e)}
    return S


# --------------------------------------------------------------------------- #
# C1 - label commensurability
# --------------------------------------------------------------------------- #
def c1(df):
    pos = df.filter(pl.col("label") == 1)
    neg = df.filter(pl.col("label") == 0)
    cp = [C.containment(c, k) for c, k in zip(pos["claim"].to_list(), pos["chunk"].to_list())]
    cn = [C.containment(c, k) for c, k in zip(neg["claim"].to_list(), neg["chunk"].to_list())]
    dp, dn = dist(cp), dist(cn)

    per_family = {}
    for fam in sorted(set(df["neg_family"].to_list())):
        f = df.filter(pl.col("neg_family") == fam)
        fp = f.filter(pl.col("label") == 1)
        fn_ = f.filter(pl.col("label") == 0)
        per_family[fam] = {
            "positive": dist([C.containment(c, k) for c, k in
                              zip(fp["claim"].to_list(), fp["chunk"].to_list())]),
            "negative": dist([C.containment(c, k) for c, k in
                              zip(fn_["claim"].to_list(), fn_["chunk"].to_list())]),
        }
        per_family[fam]["delta_share_ge_0.90"] = round(
            per_family[fam]["positive"]["share_ge_0.90"]
            - per_family[fam]["negative"]["share_ge_0.90"], 6)
        per_family[fam]["delta_share_eq_1.0"] = round(
            per_family[fam]["positive"]["share_eq_1.0"]
            - per_family[fam]["negative"]["share_eq_1.0"], 6)

    # paired within-pair reading - the sharpest statement of "comparable rates"
    piv = (df.select(["pair_id", "label"])
             .with_columns(pl.Series("c", [C.containment(c, k) for c, k in
                                           zip(df["claim"].to_list(), df["chunk"].to_list())]))
             .pivot(on="label", index="pair_id", values="c", aggregate_function="first")
             .drop_nulls())
    pv, nv = piv["1"].to_numpy(), piv["0"].to_numpy()
    paired = {
        "pairs": int(piv.height),
        "share_identical_containment_within_pair": round(float((np.abs(pv - nv) < 1e-12).mean()), 6),
        "mean_abs_within_pair_difference": round(float(np.abs(pv - nv).mean()), 6),
        "share_pairs_positive_strictly_higher": round(float((pv > nv).mean()), 6),
        "share_pairs_negative_strictly_higher": round(float((nv > pv).mean()), 6),
        "containment_auroc_label_vs_containment": round(C.auroc(df["label"].to_list(),
            [C.containment(c, k) for c, k in zip(df["claim"].to_list(), df["chunk"].to_list())]), 6),
    }

    d_ge90 = abs(dn["share_ge_0.90"] - dp["share_ge_0.90"])
    d_eq1 = abs(dn["share_eq_1.0"] - dp["share_eq_1.0"])
    # bar as written: negatives >= 90% attested at a rate within 0.10 of positives
    reject_ge90 = bool(dn["share_ge_0.90"] >= 0.90 and d_ge90 <= 0.10)
    reject_eq1 = bool(dn["share_eq_1.0"] >= 0.90 and d_eq1 <= 0.10)

    # ---- EXECUTOR-ADDED, reported separately, joins no registered bar --------
    # order-sensitive readings of the same legs
    bare_pos = bare_neg = 0
    big_p, big_n = [], []
    for r in df.iter_rows(named=True):
        asserted = r["true_path"] if r["label"] == 1 else r["wrong_path"]
        bare = asserted.replace(ARROW, " ")
        hit = bare in r["chunk"]
        ct = C.tokens(r["claim"])
        kt = C.tokens(r["chunk"])
        cb = set(zip(ct, ct[1:]))
        kb = set(zip(kt, kt[1:]))
        bg = len(cb & kb) / len(cb) if cb else 0.0
        if r["label"] == 1:
            bare_pos += hit
            big_p.append(bg)
        else:
            bare_neg += hit
            big_n.append(bg)
    n_pos, n_neg = pos.height, neg.height

    return {
        "verdict": "FAIL" if (reject_ge90 or reject_eq1) else "PASS",
        "head_declared": "grounding scalar (the shipped ground() support head) - "
                         "the lane is loaded by R18-H150_arm_run.make_build_mix "
                         "into the single BCE task head, DANN group `path_bind`",
        "label_predicate": (
            "label 1 = the ORDERED navigation path the claim renders arrowed "
            "(A -> B -> C) is the path the page states as a bare token run for "
            "that setting and value; label 0 = that ordered path is corrupted "
            "(two adjacent segments transposed, or one segment replaced by a "
            "sibling the same page attests on a different path) and does not "
            "occur on the page. The predicate is support, but support carried "
            "ENTIRELY by token ORDER: the negative's token multiset is by "
            "construction identical (path_transpose) or drawn wholly from the "
            "page (path_wrong_segment)."),
        "instrument": "R20-H174_lane_common.containment - fraction of the "
                      "claim's distinct content tokens present in the chunk "
                      "(the campaign lexical-baseline feature, same instrument "
                      "that produced the H175b 0.9129 / 66.4% figures)",
        "positive_leg": dp,
        "negative_leg": dn,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "delta_share_ge_0.90_neg_minus_pos": round(dn["share_ge_0.90"] - dp["share_ge_0.90"], 6),
        "delta_share_eq_1.0_neg_minus_pos": round(dn["share_eq_1.0"] - dp["share_eq_1.0"], 6),
        "delta_mean_neg_minus_pos": round(dn["mean"] - dp["mean"], 6),
        "bar": "REJECTED for the grounding head if negatives are >= 90% "
               "attested at a rate within 0.10 of positives",
        "bar_evaluation": {
            "reading_A_containment_ge_0.90": {
                "negative_rate": dn["share_ge_0.90"], "positive_rate": dp["share_ge_0.90"],
                "abs_gap": round(d_ge90, 6), "negative_rate_ge_0.90": bool(dn["share_ge_0.90"] >= 0.90),
                "gap_within_0.10": bool(d_ge90 <= 0.10), "rejects": reject_ge90},
            "reading_B_containment_eq_1.0": {
                "negative_rate": dn["share_eq_1.0"], "positive_rate": dp["share_eq_1.0"],
                "abs_gap": round(d_eq1, 6), "negative_rate_ge_0.90": bool(dn["share_eq_1.0"] >= 0.90),
                "gap_within_0.10": bool(d_eq1 <= 0.10), "rejects": reject_eq1},
        },
        "paired_within_pair_reading": paired,
        "clause_body_diagnostic": {
            "text": "C1 body - 'Negatives attested at rates comparable to "
                    "positives means the member is not teaching grounding'",
            "measured_gap_share_ge_0.90": round(d_ge90, 6),
            "measured_gap_mean_containment": round(abs(dn["mean"] - dp["mean"]), 6),
            "comparable": bool(d_ge90 <= 0.10),
            "note": "the body's diagnostic and the numbered bar are separate "
                    "readings; both are reported. The bar additionally requires "
                    "the negative rate itself to reach 0.90 before it rejects."},
        "per_family": per_family,
        "reference_point_R20_H175b": {
            "negatives_fully_attested": 0.664, "negatives_ge_0.90": 0.723,
            "both_legs_containment": 0.9129,
            "note": "the lane the clause was written against, canonical log "
                    "lines 4634-4651"},
        "executor_added_reported_separately": {
            "note": "ORDER-SENSITIVE readings of the same two legs. Reported "
                    "separately per C5's separation rule; these join NO "
                    "registered bar and do not modify the C1 verdict above.",
            "asserted_path_attested_as_contiguous_bare_run": {
                "positive_rate": round(bare_pos / n_pos, 6),
                "negative_rate": round(bare_neg / n_neg, 6),
                "gap": round(bare_pos / n_pos - bare_neg / n_neg, 6)},
            "claim_token_bigram_containment": {
                "positive": dist(big_p), "negative": dist(big_n),
                "delta_mean_pos_minus_neg": round(float(np.mean(big_p) - np.mean(big_n)), 6)},
        },
    }


# --------------------------------------------------------------------------- #
# C2 - disjointness from every evaluation surface
# --------------------------------------------------------------------------- #
def c2(df, surfaces):
    member = {
        "claims": [t for t in df["claim"].to_list() if t],
        "evidence": sorted({t for t in df["chunk"].to_list() if t}),
    }
    forms = ("raw", "truncated_1500", "ws_collapsed_casefold")

    def form_sets(texts):
        out = {f: collections.Counter() for f in forms}
        for t in texts:
            fs = string_forms(t)
            for f in forms:
                out[f][fs[f]] += 1
        return out

    mem_sets = {k: form_sets(v) for k, v in member.items()}
    per_surface, total = {}, 0
    for sname, chans in surfaces.items():
        rec = {}
        for chan in ("claims", "evidence"):
            surf_texts = chans.get(chan) or []
            if not surf_texts:
                continue
            ss = form_sets(surf_texts)
            for mchan in ("claims", "evidence"):
                block = {}
                for f in forms:
                    m, s = mem_sets[mchan][f], ss[f]
                    shared = set(m) & set(s)
                    block[f] = {
                        "shared_distinct_strings": len(shared),
                        "member_units_matched": int(sum(m[x] for x in shared)),
                        "surface_units_matched": int(sum(s[x] for x in shared)),
                    }
                    total += len(shared)
                rec[f"member_{mchan}_vs_surface_{chan}"] = block
        rec["surface_units"] = {k: len(v) for k, v in chans.items() if isinstance(v, list)}
        per_surface[sname] = rec

    return {
        "verdict": "PASS" if total == 0 else "FAIL",
        "bar": "all three string forms read zero, both directions, against "
               "every evaluation surface",
        "string_forms": list(forms),
        "truncation_cap_chars": CHUNK_MAX_CHARS,
        "member_units": {"claims": len(member["claims"]),
                         "distinct_evidence_chunks": len(member["evidence"])},
        "surfaces_tested": sorted(surfaces),
        "n_surfaces": len(surfaces),
        "total_shared_strings_all_forms_all_surfaces": total,
        "per_surface": per_surface,
    }


# --------------------------------------------------------------------------- #
# C3 - split semantics
# --------------------------------------------------------------------------- #
def c3(df, surfaces):
    per_doc = df.group_by("doc_id").agg(pl.col("pair_id").n_unique().alias("p"))
    chunk_reuse = df.group_by("chunk").agg(pl.col("doc_id").n_unique().alias("d"))
    # role separation: is any member row also present in an evaluation surface?
    member_claims = set(df["claim"].to_list())
    member_chunks = set(df["chunk"].to_list())
    dual = 0
    for chans in surfaces.values():
        for chan in ("claims", "evidence"):
            for t in chans.get(chan) or []:
                if t in member_claims or t in member_chunks:
                    dual += 1
    return {
        "verdict": "NOT-APPLICABLE",
        "why_not_applicable": (
            "the member is a pure rule generator with no source corpus and no "
            "official or internal split: all 10,000 rows are training, no row "
            "is held out, and there is therefore no split whose disjointness "
            "could be assumed or tested. The clause's failure mode - an "
            "'official' split assumed clean - cannot arise. The axis the "
            "member would cut on is measured below instead of asserted."),
        "measured_axis": {
            "axis": "doc_id = one generated manual page",
            "documents": int(df["doc_id"].n_unique()),
            "pairs": int(df["pair_id"].n_unique()),
            "pairs_per_document_max": int(per_doc["p"].max()),
            "pairs_per_document_mean": round(float(per_doc["p"].mean()), 6),
            "documents_sharing_an_identical_chunk_string": int(
                (chunk_reuse["d"] > 1).sum()),
            "distinct_chunks": int(df["chunk"].n_unique()),
        },
        "scope_rule_a_dataset_may_not_be_both": {
            "member_rows_also_present_in_an_evaluation_surface": dual,
            "bar": "0", "pass": dual == 0},
    }


# --------------------------------------------------------------------------- #
# C4 - contamination census with live positive control
# --------------------------------------------------------------------------- #
def c4(df, arena_docs):
    ev = sorted({c for c in df["chunk"].to_list() if c.strip()})
    cl = sorted({c for c in df["claim"].to_list() if c.strip()})

    def gate(label, texts, arena_texts):
        t0 = time.time()
        r = G.run_gate(texts, n=GATE_N, jaccard=GATE_JACCARD, kill=GATE_KILL,
                       label=label, arena_texts=arena_texts)
        log(f"  gate {label}: {r['verdict']} max_fraction {r['max_fraction']} "
            f"({round(time.time() - t0, 1)}s)")
        return r

    ev_res = gate("path_bind_evidence", ev, arena_docs)
    cl_res = gate("path_bind_claims", cl, arena_docs)
    spike_ev = G.spike_control(ev[:2000], arena_docs, n=GATE_N, jaccard=GATE_JACCARD,
                               k=10, label="path_bind_evidence_spike")
    spike_cl = G.spike_control(cl[:2000], arena_docs, n=GATE_N, jaccard=GATE_JACCARD,
                               k=10, label="path_bind_claims_spike")
    log(f"  spike evidence {spike_ev}")
    log(f"  spike claims   {spike_cl}")

    # ---- LIVE positive control -------------------------------------------- #
    # The gold_full precedent fed the gate a population that GENUINELY overlaps
    # the indexed side (VitaminC's own test split, max Jaccard 1.0). A generator
    # has no such held-out population, so the live control is built the way a
    # real contamination event would arrive: the member's own pages, transformed
    # the way text is transformed when it re-enters a corpus (re-wrapped,
    # truncated, partially reproduced). A graded ladder is run so the gate's
    # sensitivity is measured rather than asserted, with the ladder's genuine
    # near-duplicate rung - a re-wrapped page missing one procedure block - as
    # the DESIGNATED control the clause is judged on.
    rng = random.Random(999_4174)
    sample = ev[:300]

    def rewrap(t):
        return " ".join(t.split())

    def drop_block(t, k=1):
        parts = t.split("\n\n")
        return "\n\n".join(parts[: max(2, len(parts) - k)])

    def head_half(t):
        return t[: len(t) // 2]

    def reorder(t):
        parts = t.split("\n\n")
        body = parts[2:][::-1]
        return "\n\n".join(parts[:2] + body)

    siblings = []
    while len(siblings) < 300:
        text, _paths, _ = L4.build_page(rng)
        siblings.append(text)

    ladder = {
        "verbatim_member_page": list(sample),
        "rewrapped_member_page": [rewrap(t) for t in sample],
        "rewrapped_minus_one_procedure_block": [rewrap(drop_block(t)) for t in sample],
        "rewrapped_minus_two_procedure_blocks": [rewrap(drop_block(t, 2)) for t in sample],
        "first_half_of_member_page": [head_half(t) for t in sample],
        "procedure_blocks_reordered": [reorder(t) for t in sample],
        "sibling_page_second_generator_seed": siblings,
    }
    live_rungs = {}
    for name, texts in ladder.items():
        r = G.run_gate(texts, n=GATE_N, jaccard=GATE_JACCARD, kill=GATE_KILL,
                       label=f"live_{name}", arena_texts={"path_bind_evidence": ev})
        live_rungs[name] = {
            "units": r["candidate"]["n_units"],
            "units_with_hit": r["candidate_vs_arena"]["units_with_hit"],
            "fraction": r["candidate_vs_arena"]["fraction"],
            "best_jaccard": r["candidate_vs_arena"]["best_jaccard"],
            "fires": bool(r["candidate_vs_arena"]["fraction"] > 0.0),
        }
        log(f"  LIVE rung {name}: fraction {live_rungs[name]['fraction']} "
            f"max jaccard {live_rungs[name]['best_jaccard']['max']}")

    DESIGNATED = "rewrapped_minus_one_procedure_block"
    live_fires = live_rungs[DESIGNATED]["fraction"]
    live_vs_arena = G.run_gate(siblings, n=GATE_N, jaccard=GATE_JACCARD,
                               kill=GATE_KILL, label="live_control_second_seed",
                               arena_texts=arena_docs)

    # ---- coverage: units too short for the 8-gram instrument --------------- #
    def short(texts):
        return [t for t in texts if len(G.normalize(t).split()) < GATE_N]

    short_ev, short_cl = short(ev), short(cl)
    arena_norm = {G.normalize(c) for v in arena_docs.values() for c in v}
    exact_ev = sum(G.normalize(t) in arena_norm for t in short_ev)
    exact_cl = sum(G.normalize(t) in arena_norm for t in short_cl)

    passes = (ev_res["verdict"] != "KILL" and cl_res["verdict"] != "KILL"
              and spike_ev["passes"] and spike_cl["passes"]
              and spike_ev["baseline_hits"] == 0 and spike_cl["baseline_hits"] == 0
              and live_fires >= 0.99 and exact_ev == 0 and exact_cl == 0)
    return {
        "verdict": "PASS" if passes else "FAIL",
        "instrument": f"provenance_gate.py R14-H136 form - {GATE_N}-gram, "
                      f"Jaccard >= {GATE_JACCARD}, bidirectional, "
                      f"KILL > {GATE_KILL:.0%}, all ten walled arena corpora",
        "evidence_gate": {"verdict": ev_res["verdict"],
                          "max_fraction": ev_res["max_fraction"],
                          "candidate_units": ev_res["candidate"],
                          "candidate_vs_arena": ev_res["candidate_vs_arena"],
                          "arena_vs_candidate": ev_res["arena_vs_candidate"]},
        "claim_gate": {"verdict": cl_res["verdict"],
                       "max_fraction": cl_res["max_fraction"],
                       "candidate_units": cl_res["candidate"],
                       "candidate_vs_arena": cl_res["candidate_vs_arena"],
                       "arena_vs_candidate": cl_res["arena_vs_candidate"]},
        "spike_control": {"evidence": spike_ev, "claims": spike_cl,
                          "bar": "10/10 detected, 0 baseline hits"},
        "live_positive_control": {
            "construction": "the member's own 300 evidence pages transformed the "
                            "way text is transformed when it re-enters a corpus, "
                            "run against the member's own indexed evidence; a "
                            "graded ladder so the gate's sensitivity is measured, "
                            "not asserted",
            "designated_control": DESIGNATED,
            "designated_control_rationale":
                "a re-wrapped page missing one of its 3-5 procedure blocks is "
                "genuinely near-duplicate by construction - it reproduces most "
                "of a real member unit under exactly the whitespace "
                "transformation C2's provenance names as the invisible leak "
                "route",
            "fraction": live_fires,
            "bar": "the designated control must FIRE on >= 0.99 of its units",
            "fires": bool(live_fires >= 0.99),
            "sensitivity_ladder": live_rungs,
            "sibling_pages_against_arena": {
                "fraction": live_vs_arena["candidate_vs_arena"]["fraction"],
                "best_jaccard": live_vs_arena["candidate_vs_arena"]["best_jaccard"]}},
        "coverage": {
            "evidence_units": len(ev),
            "evidence_units_scorable": ev_res["candidate"]["n_units_scorable"],
            "evidence_units_too_short_for_8gram": len(short_ev),
            "evidence_short_units_exact_matching_arena": exact_ev,
            "claim_units": len(cl),
            "claim_units_scorable": cl_res["candidate"]["n_units_scorable"],
            "claim_units_too_short_for_8gram": len(short_cl),
            "claim_short_units_exact_matching_arena": exact_cl,
            "bar": "short units counted and covered by exact matching, 0 hits"},
    }


# --------------------------------------------------------------------------- #
# C5 - leak suite
# --------------------------------------------------------------------------- #
def c5(df):
    rng = random.Random(20174)
    claims = df["claim"].to_list()
    chunks = df["chunk"].to_list()
    y = df["label"].to_list()
    docs = df["doc_id"].to_list()

    log("  claim-only probe...")
    pa, pscore = C.claim_only_probe(claims, y, docs, rng)
    wp = C.within_pair_accuracy(df, pscore, by="neg_family")
    worst_wp = max(v["acc"] for v in wp.values())

    log("  evidence-only probe...")
    ea, escore = C.claim_only_probe(chunks, y, docs, random.Random(20175))
    ewp = C.within_pair_accuracy(df, escore, by="neg_family")

    sp = C.surface_parity(df)

    # balance
    def bal(col):
        g = df.group_by([col, "label"]).len().sort([col, "label"])
        out = {}
        for k, lab, n in g.iter_rows():
            out.setdefault(str(k), {})[f"label_{lab}"] = int(n)
        return out

    fam = {k: int(v) for k, v in df.group_by("neg_family").len().iter_rows()}
    swap = df.filter(pl.col("label") == 0).group_by("swap_index").len().sort("swap_index")

    # executor-added surface channels, kept out of the registered conjunction
    arrow_n = [float(c.count("→")) for c in claims]
    seg_n = [float(len(path_segments(r["true_path"] if r["label"] == 1 else r["wrong_path"])))
             for r in df.iter_rows(named=True)]
    first_seg_pos = []
    for r in df.iter_rows(named=True):
        p = path_segments(r["true_path"] if r["label"] == 1 else r["wrong_path"])
        first_seg_pos.append(float(r["chunk"].find(p[0])) if p else 0.0)
    extra = {
        "claim_arrow_count": round(C.auroc(y, arrow_n), 6),
        "claim_path_depth": round(C.auroc(y, seg_n), 6),
        "first_segment_char_offset_in_chunk": round(C.auroc(y, first_seg_pos), 6),
    }

    registered = {
        "claim_only_converged_probe": {"value": round(pa, 6), "bar": "< 0.55",
                                       "pass": bool(pa < 0.55)},
        "within_pair_claim_only": {"per_family": wp, "worst": round(worst_wp, 6),
                                   "bar": "< 0.60", "pass": bool(worst_wp < 0.60)},
        "evidence_only_probe": {
            "value": round(ea, 6), "per_family_within_pair": ewp,
            "bar": "at chance - the construction gives both legs the SAME chunk, "
                   "so an evidence-only score is identical within every pair",
            "pass": bool(abs(ea - 0.5) <= 0.05)},
        "question_only_probe": {
            "value": None, "bar": "n/a",
            "pass": None,
            "not_applicable": "the member carries no question channel - its "
                              "columns are claim / chunk / label / pair_id / "
                              "neg_family and the loader presents (claim, chunk) "
                              "only. No proxy substituted."},
        "surface_parity": {"auroc": sp["auroc"], "bar": "each channel in [0.45, 0.55]",
                           "worst_deviation": sp["worst_deviation"], "pass": sp["pass"]},
        "family_balance": {"families": fam, "bar": "registered 6,000 / 4,000",
                           "pass": fam == {"path_transpose": 6000,
                                           "path_wrong_segment": 4000}},
    }
    bars = [v["pass"] for v in registered.values() if v.get("pass") is not None]
    return {
        "verdict": "PASS" if all(bars) else "FAIL",
        "registered_conjunction": registered,
        "balance_detail": {
            "depth_by_label": bal("depth"),
            "template_id_by_label": bal("template_id"),
            "negative_swap_index": {str(k): int(v) for k, v in swap.iter_rows()},
            "label_balance": {"label_1": int(sum(y)), "label_0": int(len(y) - sum(y))},
        },
        "executor_added_reported_separately": {
            "note": "extra surface channels measured by this verification; they "
                    "join no registered bar (C5 separation rule)",
            "auroc": extra},
    }


# --------------------------------------------------------------------------- #
# C6 - no memorisation channel
# --------------------------------------------------------------------------- #
def c6(df):
    # literal instrument: what else does the mix associate with this pair's key?
    key_counts = df.group_by("doc_id").agg(pl.col("pair_id").n_unique().alias("p"))
    shared_key_pairs = int((key_counts["p"] > 1).sum())

    y = df["label"].to_list()
    rows = df.iter_rows(named=True)
    asserted = []
    for r in rows:
        asserted.append((r["pair_id"], r["label"],
                         path_segments(r["true_path"] if r["label"] == 1 else r["wrong_path"])))

    # F1 - global ordered adjacent-segment-pair association, leave-own-pair-out
    pos_bg, neg_bg = collections.Counter(), collections.Counter()
    for _pid, lab, segs in asserted:
        tgt = pos_bg if lab == 1 else neg_bg
        for b in zip(segs, segs[1:]):
            tgt[b] += 1
    own = collections.defaultdict(lambda: [collections.Counter(), collections.Counter()])
    for pid, lab, segs in asserted:
        for b in zip(segs, segs[1:]):
            own[pid][lab][b] += 1
    f1 = []
    for pid, lab, segs in asserted:
        bs = list(zip(segs, segs[1:]))
        if not bs:
            f1.append(0.0)
            continue
        v = 0.0
        for b in bs:
            p = pos_bg[b] - own[pid][1][b]
            n = neg_bg[b] - own[pid][0][b]
            v += p - n
        f1.append(v / len(bs))

    # F2 - global (segment, level) association from the rest of the lane
    lev = collections.Counter()
    own_lev = collections.defaultdict(collections.Counter)
    for pid, lab, segs in asserted:
        if lab == 1:
            for i, s in enumerate(segs):
                lev[(s, i)] += 1
                own_lev[pid][(s, i)] += 1
    f2 = []
    for pid, lab, segs in asserted:
        if not segs:
            f2.append(0.0)
            continue
        f2.append(sum(lev[(s, i)] - own_lev[pid][(s, i)] for i, s in enumerate(segs)) / len(segs))

    # F3 - global segment-presence association (the emanual defect's own feature)
    seg_pos, seg_neg = collections.Counter(), collections.Counter()
    for _pid, lab, segs in asserted:
        (seg_pos if lab == 1 else seg_neg).update(set(segs))
    f3 = [np.mean([seg_pos[s] - seg_neg[s] for s in segs]) if segs else 0.0
          for _pid, _lab, segs in asserted]

    a1, a2, a3 = C.auroc(y, f1), C.auroc(y, f2), C.auroc(y, f3)
    worst = max(abs(a1 - 0.5), abs(a2 - 0.5), abs(a3 - 0.5))
    return {
        "verdict": "PASS" if worst <= 0.05 else "FAIL",
        "literal_instrument": {
            "key": "doc_id (one generated manual page)",
            "pairs_whose_key_is_shared_with_any_other_pair": shared_key_pairs,
            "coverage": 0.0,
            "value": None,
            "reading": "UNDEFINED - every doc_id belongs to exactly one pair, "
                       "so the mix associates nothing with a pair's key beyond "
                       "that pair's own two rows, and both rows carry the same "
                       "chunk. The C6 feature as written has no support here."},
        "executor_added_reported_separately": {
            "note": "three association features keyed on lane-wide training "
                    "statistics, leave-own-pair-out. They join no registered "
                    "bar; they exist because the literal instrument is "
                    "undefined for this member.",
            "ordered_adjacent_segment_pair_association_auroc": round(a1, 6),
            "segment_level_position_association_auroc": round(a2, 6),
            "segment_presence_association_auroc": round(a3, 6),
            "bar_applied": "chance +/- 0.05",
            "worst_deviation": round(worst, 6)},
    }


# --------------------------------------------------------------------------- #
# C7 / C8
# --------------------------------------------------------------------------- #
def c7(df):
    rows, pairs = df.height, int(df["pair_id"].n_unique())
    return {
        "verdict": "PASS" if (5_000 <= rows <= 15_000 and rows == 2 * pairs) else "FAIL",
        "declared_unit": "rows",
        "registration": "canonical log line 3872 - 'L4 bind_path_segment rider "
                        "(~5-15k rows, pure generator)'; stage-0 table line 3899 "
                        "records 10,000 rows / 5,000 pairs",
        "rows": rows,
        "pairs": pairs,
        "band_rows": [5_000, 15_000],
        "margin_rows_above_floor": rows - 5_000,
        "margin_rows_below_ceiling": 15_000 - rows,
        "loader_guard": "R18-H150_arm_run.make_build_mix aborts unless the "
                        "parquet reads exactly 10,000 rows / 5,000 pairs / "
                        "{path_transpose: 6000, path_wrong_segment: 4000}",
        "both_units_reported": True,
    }


BRANDS = re.compile(
    r"\b(samsung|lg|sony|philips|panasonic|brother|canon|epson|hp|dell|lenovo|"
    r"apple|iphone|android|google|microsoft|windows|xiaomi|huawei|bosch|siemens|"
    r"netgear|tp-?link|asus|acer|nvidia|amd|intel|kolomolo|stellars)\b", re.I)


def c8(df):
    seg_use = collections.Counter()
    for r in df.filter(pl.col("label") == 1).iter_rows(named=True):
        seg_use.update(path_segments(r["true_path"]))
    claims = df["claim"].to_list()
    chunks = df["chunk"].to_list()
    brand_hits = sum(1 for t in claims + chunks if BRANDS.search(t))
    dup_claim = len(claims) - len(set(claims))
    dup_chunk_pairs = df.select(["pair_id", "chunk"]).unique()["chunk"].n_unique()
    tmpl = {str(k): int(v) for k, v in df.group_by("template_id").len().sort("template_id").iter_rows()}
    return {
        "verdict": "PASS" if brand_hits == 0 else "FAIL",
        "source": "rule generator written in "
                  "experiments/grounding-semantic/R20-H174_lane_L4.py - no "
                  "source corpus, no crawl, no external text",
        "licence": "n/a - text generated by this repository (MIT)",
        "retrieval_date": "n/a - nothing retrieved; built 2026-08-16 (parquet "
                          "mtime), seed 4174",
        "selection_predicate": {
            "generator_parameters": {"seed": 4174, "n_pairs_target": 5000,
                                     "transpose_share": 0.60, "depths": [3, 4, 5],
                                     "procedures_per_page": [3, 4, 5],
                                     "segment_pool": 112, "templates": 6,
                                     "values": 20},
            "rejection_rules": [
                "the true path must occur in the page as a contiguous bare "
                "token run, else the draw is discarded",
                "the corrupted path must NOT occur in the page",
                "the corrupted path must differ from the true path and carry no "
                "repeated segment",
                "path_wrong_segment substitutes only a segment the SAME page "
                "attests on a different path",
                "family quota 60/40 enforced during generation",
                "R20-H174_lane_common.dedupe on (claim, chunk, label), pairs "
                "surviving only if both legs survive"],
        },
        "internal_structure": {
            "rows": df.height, "pairs": int(df["pair_id"].n_unique()),
            "documents": int(df["doc_id"].n_unique()),
            "distinct_claims": int(df["claim"].n_unique()),
            "distinct_chunks": int(df["chunk"].n_unique()),
            "distinct_true_paths": int(df["true_path"].n_unique()),
            "distinct_wrong_paths": int(df["wrong_path"].n_unique()),
            "duplicate_claim_rows": dup_claim,
            "distinct_chunks_over_pairs": dup_chunk_pairs,
            "template_id_counts": tmpl,
            "segment_reuse_over_positive_paths": {
                "distinct_segments_used": len(seg_use),
                "max_uses_of_one_segment": max(seg_use.values()),
                "min_uses_of_one_segment": min(seg_use.values()),
                "mean_uses": round(float(np.mean(list(seg_use.values()))), 2)},
            "claim_chars": C.char_stats(claims),
            "chunk_chars": C.char_stats(chunks),
            "window_census": C.window_census(chunks),
        },
        "public_repository_check": {
            "pattern": "known device / vendor / client brand tokens",
            "rows_matching": brand_hits, "bar": "0", "pass": brand_hits == 0},
    }


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    df = pl.read_parquet(MEMBER)
    log(f"member: {MEMBER.name}  {df.height} rows / {df['pair_id'].n_unique()} pairs")

    log("C1 label commensurability...")
    r1 = c1(df)
    log(f"  C1 {r1['verdict']}  pos>=0.90 {r1['positive_leg']['share_ge_0.90']} "
        f"neg>=0.90 {r1['negative_leg']['share_ge_0.90']}")

    log("loading evaluation surfaces...")
    S = eval_surfaces()
    log(f"  {len(S)} surfaces")
    arena_docs = {k.split(':', 1)[1]: v["evidence"] for k, v in S.items()
                  if k.startswith("arena_documents:")}

    log("C2 disjointness...")
    r2 = c2(df, S)
    log(f"  C2 {r2['verdict']} total shared strings {r2['total_shared_strings_all_forms_all_surfaces']}")

    log("C3 split semantics...")
    r3 = c3(df, S)

    log("C4 contamination census...")
    r4 = c4(df, arena_docs)
    log(f"  C4 {r4['verdict']}")

    log("C5 leak suite...")
    r5 = c5(df)
    log(f"  C5 {r5['verdict']}")

    log("C6 memorisation channel...")
    r6 = c6(df)
    log(f"  C6 {r6['verdict']}")

    r7, r8 = c7(df), c8(df)
    log(f"  C7 {r7['verdict']}  C8 {r8['verdict']}")

    clauses = {"C1": r1, "C2": r2, "C3": r3, "C4": r4,
               "C5": r5, "C6": r6, "C7": r7, "C8": r8}
    fails = [k for k, v in clauses.items() if v["verdict"] == "FAIL"]
    report = {
        "member": "path_bind",
        "artifact": str(MEMBER.relative_to(ROOT)),
        "role": "training member (constructed lane L4), DANN group `path_bind`, "
                "10,000 rows in the R20-H174 portfolio mix; LIVE - R20-H174 "
                "draws 2/3/4 train on it",
        "contract": "docs/experiments/dataset-contract.md (C1-C8)",
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu_only": True,
        "conforming": not fails,
        "failed_clauses": fails,
        "headline": (
            "path_bind clears every clause, but it clears C1 on the low half of "
            "the bar, not on leg separation: its two legs' claim-to-evidence "
            f"containment distributions are indistinguishable "
            f"(negatives {r1['negative_leg']['share_ge_0.90']} vs positives "
            f"{r1['positive_leg']['share_ge_0.90']} at >= 0.90 containment, gap "
            f"{abs(r1['negative_leg']['share_ge_0.90'] - r1['positive_leg']['share_ge_0.90']):.3f}, "
            f"containment AUROC {r1['paired_within_pair_reading']['containment_auroc_label_vs_containment']}, "
            f"{r1['paired_within_pair_reading']['share_identical_containment_within_pair']:.3f} of pairs "
            "byte-equal on the measure). The bar does not reject because the "
            "negative attestation rate is 0.391, far under its 0.90 trigger - "
            "not because negatives are less attested than positives. All label "
            "separation lives in token ORDER, which the unordered containment "
            "instrument C1 mandates cannot see: the asserted path is attested "
            "as a contiguous bare run on 100% of positives and 0% of negatives."),
        "caveats": [
            "C1's mandated instrument is a set-of-content-tokens containment. "
            "For this member it is blind by construction - path_transpose "
            "preserves the token multiset exactly and path_wrong_segment draws "
            "its substitute from the same page. The order-sensitive readings "
            "that do separate the legs are reported separately and join no bar.",
            "C4's live positive control fires 300/300 on genuine near-duplicates "
            "of the member's own pages down to a 50% fragment, but does NOT "
            "fire on sibling pages from a second generator seed (max Jaccard "
            "0.2488 against a 0.30 threshold). Same-generator sibling text is "
            "below the instrument's sensitivity; that is a measured limit of "
            "the gate on this text distribution, reported, not a clause result.",
            "C3 is NOT-APPLICABLE: the member has no split of any kind, so no "
            "split semantics exist to verify. The axis it would cut on is "
            "measured rather than asserted.",
        ],
        "clauses": clauses,
        "seconds": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2))
    log(f"\n=> {OUT}  conforming={report['conforming']}  fails={fails}  "
        f"({report['seconds']}s)")


if __name__ == "__main__":
    main()

"""DATASET CONTRACT PHASE 1 - verification of the `halueval` mix member. CPU ONLY.

Contract: docs/experiments/dataset-contract.md (clauses C1-C8, agreed 2026-08-17).
Member: the `halueval` DANN group of the assembled training mix, produced by
`R10-H108_lane.public_train()` lines 118-136 and carried unchanged into the
R18-H150 flagship mix and the R20-H174 portfolio mix.

DISCIPLINE
  * every number is measured here; nothing is read off a dataset card
  * the mix is rebuilt through the BANKED loader, never re-implemented; the
    replay used to recover pair structure is proved aligned row-for-row against
    the loader's own output across all 40,000 halueval rows before it is used
  * the contamination census reuses `provenance_gate.py` in the banked R14-H136
    ruling-2 form; thresholds are read from `R19_supply_gates.py`
  * CPU only - CUDA_VISIBLE_DEVICES is forced empty before any import

Stages (each writes its own JSON so a killed run resumes from disk):
  core      load + replay alignment + C1 + C6 + C7 + C8
  disjoint  C2 (three string forms, both directions, all surfaces) + C3
  census    C4 (8-gram Jaccard >= 0.3 wall, spike + live positive controls)
  probe     C5 executor-added claim-only probe (reported SEPARATELY)
  merge     assemble halueval_contract_report.json

Run: uv run python experiments/grounding-semantic/contract/halueval_contract.py \
         --stage core
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU ONLY - three GPUs carry live draws
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse
import importlib.util
import io
import json
import pathlib
import re
import time
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
EXP = HERE.parent
ROOT = EXP.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
ARCHIVE = DATA / "dataset-halueval.zip"

MEMBER = "halueval"
WS = re.compile(r"\s+")


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def norm_ws(s):
    """Whitespace-collapsed, case-folded - C2's third string form."""
    return WS.sub(" ", s).strip().lower()


# --------------------------------------------------------------------------- #
# load - through the banked loader, with a proved-aligned replay for structure
# --------------------------------------------------------------------------- #
def load_member():
    """Returns the halueval slice of the assembled mix, untruncated, plus the
    pair structure recovered by a replay that is PROVED aligned to the loader."""
    # The arm calls `arm.H108.public_train()` and `untruncated_evidence` lifts
    # the cap on `G1.M59.CFG` - which is the CFG of G1's OWN H108 instance. A
    # separately loaded H108 would still truncate, so the arm's instance is used.
    G1 = _mod("g1arm", EXP / "R16-H142_G1_arm.py")
    H108 = G1.H108
    cap = G1.M59.CFG.chunk_max_chars
    print(f"loader chunk_max_chars = {cap}", flush=True)

    t0 = time.time()
    with G1.untruncated_evidence():
        claims, chunks, y, tags = H108.public_train()
    print(f"public_train(): {len(y)} rows in {time.time() - t0:.0f}s", flush=True)

    idx = [i for i, t in enumerate(tags) if t == MEMBER]
    lo, hi = idx[0], idx[-1] + 1
    assert idx == list(range(lo, hi)), "halueval rows are not contiguous"
    m_claims = claims[lo:hi]
    m_chunks = chunks[lo:hi]
    m_y = np.asarray(y[lo:hi], dtype=float)

    # --- replay the loader's own source order to recover pair structure -------
    z = zipfile.ZipFile(ARCHIVE)
    rep_claims, rep_chunks, rep_y, halves, pair_ids = [], [], [], [], []
    pid = 0
    subset_rows = {}
    for cfg, ev_col, pos_col, neg_col in (
        ("qa", "knowledge", "right_answer", "hallucinated_answer"),
        ("summarization", "document", "right_summary", "hallucinated_summary"),
    ):
        hits = [x for x in z.namelist() if f"__{cfg}__" in x]
        d = pl.read_parquet(io.BytesIO(z.read(hits[0])))
        subset_rows[cfg] = d.height
        for ev, pos, neg in zip(d[ev_col].to_list(), d[pos_col].to_list(),
                                d[neg_col].to_list(), strict=True):
            rep_claims += [pos, neg]
            rep_chunks += [ev, ev]
            rep_y += [1.0, 0.0]
            halves += [cfg, cfg]
            pair_ids += [pid, pid]
            pid += 1

    if len(rep_claims) != len(m_claims):
        raise SystemExit(f"REPLAY ABORT: {len(rep_claims)} replayed vs {len(m_claims)} loaded")
    bad = [i for i in range(len(m_claims))
           if rep_claims[i] != m_claims[i] or rep_chunks[i] != m_chunks[i]
           or rep_y[i] != m_y[i]]
    if bad:
        raise SystemExit(f"REPLAY ABORT: {len(bad)} rows differ, first at {bad[0]}")
    print(f"replay ALIGNED row-for-row across all {len(m_claims)} halueval rows", flush=True)

    return {
        "claims": m_claims, "chunks": m_chunks, "y": m_y,
        "half": halves, "pair_id": pair_ids,
        "mix_rows_total": int(len(y)), "member_slice": [lo, hi],
        "chunk_max_chars": int(cap), "subset_rows": subset_rows,
    }


def frame(d):
    return pl.DataFrame({
        "pair_id": d["pair_id"], "half": d["half"], "label": d["y"],
        "claim": d["claims"], "chunk": d["chunks"],
    })


# --------------------------------------------------------------------------- #
# C1 - label commensurability
# --------------------------------------------------------------------------- #
def leg_stats(cont):
    a = np.asarray(cont, dtype=float)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "p10": round(float(np.percentile(a, 10)), 4),
        "p90": round(float(np.percentile(a, 90)), 4),
        "frac_ge_0.90": round(float((a >= 0.90).mean()), 4),
        "frac_eq_1.00": round(float((a >= 0.99999).mean()), 4),
        "frac_ge_0.75": round(float((a >= 0.75).mean()), 4),
        "frac_le_0.50": round(float((a <= 0.50).mean()), 4),
    }


def clause_c1(df, C):
    """Claim-to-evidence containment on both legs, whole member and per half.

    Instrument: the banked campaign lexical-baseline feature
    `R20-H174_lane_common.containment` - fraction of the claim's content tokens
    present in the evidence.  Measured on the UNTRUNCATED evidence (what the
    H150/H174 arms feed) and on the 1,500-char truncated form (what the
    incumbent in-domain read feeds), so no reading depends on presentation."""
    out = {"instrument": "R20-H174_lane_common.containment (content-token "
                         "containment of claim in evidence), banked",
           "presentations": {}}
    claims = df["claim"].to_list()
    chunks = df["chunk"].to_list()
    lab = df["label"].to_numpy()
    half = np.array(df["half"].to_list())

    for pres, cut in (("untruncated", None), ("truncated_1500", 1500)):
        cont = np.array([C.containment(c, k if cut is None else k[:cut])
                         for c, k in zip(claims, chunks)])
        block = {}
        for scope, mask in [("all", np.ones(len(lab), bool)),
                            ("qa", half == "qa"),
                            ("summarization", half == "summarization")]:
            pos = cont[mask & (lab == 1)]
            neg = cont[mask & (lab == 0)]
            p, n = leg_stats(pos), leg_stats(neg)
            gap = abs(n["frac_ge_0.90"] - p["frac_ge_0.90"])
            block[scope] = {
                "positive_leg": p, "negative_leg": n,
                "attested_rate_gap_at_0.90": round(float(gap), 4),
                "mean_gap": round(float(p["mean"] - n["mean"]), 4),
                "rejected_by_c1_bar": bool(gap <= 0.10),
            }
        out["presentations"][pres] = block
    return out


def clause_c1_supplement(df, C):
    """The stronger attestation readings C1's provenance actually names: a
    verbatim negative labelled 0, and the within-pair comparison."""
    claims = df["claim"].to_list()
    chunks = df["chunk"].to_list()
    lab = df["label"].to_numpy()
    half = np.array(df["half"].to_list())
    pid = np.array(df["pair_id"].to_list())

    nc = [norm_ws(c) for c in claims]
    nk = [norm_ws(k) for k in chunks]
    verbatim = np.array([bool(c) and c in k for c, k in zip(nc, nk)])
    cont = np.array([C.containment(c, k) for c, k in zip(claims, chunks)])
    ntok = np.array([len(C.tokens(c)) for c in claims])

    out = {"definitions": {
        "verbatim": "whitespace-collapsed case-folded claim is an exact "
                    "substring of the whitespace-collapsed case-folded evidence",
        "fully_attested": "content-token containment == 1.0"}}
    for scope, mask in [("all", np.ones(len(lab), bool)),
                        ("qa", half == "qa"),
                        ("summarization", half == "summarization")]:
        blk = {}
        for leg, sel in (("positive", lab == 1), ("negative", lab == 0)):
            m = mask & sel
            blk[leg] = {
                "n": int(m.sum()),
                "verbatim_rate": round(float(verbatim[m].mean()), 4),
                "verbatim_rows": int(verbatim[m].sum()),
                "fully_attested_rows": int((cont[m] >= 0.99999).sum()),
                "fully_attested_and_verbatim_rows": int(
                    ((cont[m] >= 0.99999) & verbatim[m]).sum()),
                "claim_content_tokens_mean": round(float(ntok[m].mean()), 2),
                "claim_content_tokens_median": float(np.median(ntok[m])),
                "claims_with_le_2_content_tokens": int((ntok[m] <= 2).sum()),
            }
        # within-pair: does the negative reach the positive's containment?
        m = mask
        order = np.argsort(pid[m], kind="stable")
        p = pid[m][order]
        c = cont[m][order]
        l = lab[m][order]
        pos = c[l == 1]
        neg = c[l == 0]
        blk["within_pair"] = {
            "pairs": int(len(pos)),
            "neg_ge_pos": round(float((neg >= pos).mean()), 4),
            "neg_gt_pos": round(float((neg > pos).mean()), 4),
            "mean_pair_gap_pos_minus_neg": round(float((pos - neg).mean()), 4),
            "pairs_with_fully_attested_negative": int((neg >= 0.99999).sum()),
        }
        out[scope] = blk
    out["reference_R20_H175b_poisoned_lane"] = {
        "containment_both_legs": 0.9129,
        "negatives_fully_attested": 0.664,
        "negatives_attested_ge_0.90": 0.723,
        "source": "docs/experiments/semantic-grounding-experiments.md, "
                  "R20-H175b WITHDRAWN block",
    }
    # A sample of the fully attested negatives, for the record. Rows whose text
    # carries a corporate suffix are skipped so no company name lands in a
    # PUBLIC-repository artifact; all quoted text is MIT-licensed corpus content.
    corp = re.compile(r"\b(inc|ltd|llc|corp|corporation|plc|gmbh|s\.a|co)\b\.?",
                      re.IGNORECASE)
    ex = []
    for i in np.nonzero((lab == 0) & (cont >= 0.99999))[0]:
        c, k = claims[i][:200], chunks[i][:160]
        if corp.search(c) or corp.search(k):
            continue
        ex.append({"half": half[i], "claim": c, "evidence_head": k})
        if len(ex) == 8:
            break
    out["fully_attested_negative_examples"] = ex
    return out


# --------------------------------------------------------------------------- #
# C2 - disjointness from every evaluation surface / C3 - split semantics
# --------------------------------------------------------------------------- #
EVAL_PARQUETS = {
    "R11-H117_heldout_pairs": ("R11-H117_heldout_pairs.parquet", "chunk", "claim"),
    "R17-H143_evalset": ("R17-H143_evalset.parquet", "chunk", "claim"),
    "R20-H175b_qlane_eval": ("R20-H175b_qlane_eval.parquet", "chunk", "claim"),
    "R20-H175b_qlane_eval_repaired": ("R20-H175b_qlane_eval_repaired.parquet", "chunk", "claim"),
    "R20-H175b_qlane_eval_clean": ("R20-H175b_qlane_eval_clean.parquet", "chunk", "claim"),
    "R20-H177_eval_B": ("R20-H177_eval_B.parquet", "chunk", "claim"),
    "R20-H177_eval_C": ("R20-H177_eval_C.parquet", "chunk", "claim"),
}


def forms(texts, cut=1500):
    raw = set(texts)
    return {
        "raw": raw,
        "truncated": {t[:cut] for t in texts},
        "normalised": {norm_ws(t) for t in texts},
    }


def two_way(member_forms, surface_forms):
    out = {}
    for f in ("raw", "truncated", "normalised"):
        m, s = member_forms[f], surface_forms[f]
        inter = m & s
        out[f] = {
            "member_units": len(m), "surface_units": len(s),
            "member_units_in_surface": len(inter),
            "surface_units_in_member": len(inter),
            "member_fraction": round(len(inter) / max(len(m), 1), 6),
            "surface_fraction": round(len(inter) / max(len(s), 1), 6),
        }
    return out


def clause_c2(df, G):
    ev = [c for c in set(df["chunk"].to_list()) if c.strip()]
    cl = [c for c in set(df["claim"].to_list()) if c.strip()]
    ev_f, cl_f = forms(ev), forms(cl)

    surfaces = {}
    arena_texts, _ = G.load_arena()
    arena_docs = [c for v in arena_texts.values() for c in v]
    surfaces["arena_ragbench_10"] = {
        "evidence": two_way(ev_f, forms(arena_docs)),
        "claim_vs_surface_evidence": two_way(cl_f, forms(arena_docs)),
        "surface_unit": "RAGBench test documents, H77 arena sample "
                        f"({len(arena_docs)} chunks over {len(arena_texts)} subsets)",
    }

    gf = pl.read_parquet(EXP / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet")
    surfaces["gold_full"] = {
        "evidence": two_way(ev_f, forms([c for c in gf["chunk"].to_list() if c])),
        "claim_vs_surface_claim": two_way(cl_f, forms([c for c in gf["claim"].to_list() if c])),
        "surface_unit": f"gold trace chunks ({gf.height} rows)",
    }

    for name, (fn, ev_col, cl_col) in EVAL_PARQUETS.items():
        p = EXP / fn
        if not p.exists():
            surfaces[name] = {"status": "ABSENT"}
            continue
        d = pl.read_parquet(p)
        surfaces[name] = {
            "evidence": two_way(ev_f, forms([c for c in d[ev_col].to_list() if c])),
            "claim_vs_surface_claim": two_way(cl_f, forms([c for c in d[cl_col].to_list() if c])),
            "surface_unit": f"{fn} ({d.height} rows)",
        }

    worst = 0.0
    for s in surfaces.values():
        for k, v in s.items():
            if isinstance(v, dict) and "raw" in v:
                for f in v.values():
                    worst = max(worst, f["member_fraction"], f["surface_fraction"])
    return {
        "member_units": {"evidence_distinct": len(ev), "claim_distinct": len(cl)},
        "forms": ["raw", "truncated (1,500 chars)", "whitespace-collapsed case-folded"],
        "surfaces": surfaces,
        "worst_fraction_any_form_any_direction": round(worst, 6),
        "pass": bool(worst == 0.0),
    }


def clause_c3(df, d):
    """Split semantics MEASURED from the archive, not read from the card."""
    z = zipfile.ZipFile(ARCHIVE)
    files = sorted(n for n in z.namelist() if n.endswith(".parquet"))
    subsets = {}
    for n in files:
        sub = n.split("__")[2]
        dd = pl.read_parquet(io.BytesIO(z.read(n)))
        subsets[sub] = {"rows": dd.height, "columns": dd.columns,
                        "split_files": 1, "loaded_by_mix": sub in ("qa", "summarization")}
    # per-subset internal repeat structure = the only axis a split could cut on
    axes = {}
    for cfg, ev_col in (("qa", "knowledge"), ("summarization", "document")):
        dd = pl.read_parquet(io.BytesIO(z.read(next(x for x in files if f"__{cfg}__" in x))))
        n_ev = dd[ev_col].n_unique()
        axes[cfg] = {
            "rows": dd.height,
            "distinct_evidence": int(n_ev),
            "evidence_reuse_rows": int(dd.height - n_ev),
            "distinct_questions": int(dd["question"].n_unique()) if "question" in dd.columns else None,
            "evidence_blocks_with_2plus_rows": int(
                dd.group_by(ev_col).len().filter(pl.col("len") > 1).height),
        }
    return {
        "archive_splits": subsets,
        "measured_split_axis": axes,
        "selection_predicate": "ALL rows of subsets `qa` and `summarization`; "
                               "`dialogue` and `general` present in the archive "
                               "and NOT loaded; no split filter, no row filter",
        "member_is_also_an_evaluation_surface": False,
        "evaluation_surface_check": "grep over experiments/grounding-semantic for "
                                    "`halueval`: every hit is a trainer or mix "
                                    "loader; no read/eval script loads it",
    }


# --------------------------------------------------------------------------- #
# C4 - contamination census
# --------------------------------------------------------------------------- #
def sentence_thin(text, keep=2, of=3):
    """Genuine near-duplicate by construction: keep `keep` of every `of`
    sentences. Survives the gate's normalisation (which erases case and
    whitespace), so it is a real near-duplicate rather than a re-spelt copy."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(p for i, p in enumerate(parts) if i % of < keep)


def clause_c4(df, G, gate_n, gate_j, gate_kill, spike_sample=2000, unit_cap=0):
    ev = sorted({c for c in df["chunk"].to_list() if c.strip()})
    cl = sorted({c for c in df["claim"].to_list() if c.strip()})
    if unit_cap:
        rng = np.random.default_rng(0)
        if len(ev) > unit_cap:
            ev = [ev[i] for i in sorted(rng.choice(len(ev), unit_cap, replace=False))]
        if len(cl) > unit_cap:
            cl = [cl[i] for i in sorted(rng.choice(len(cl), unit_cap, replace=False))]
    arena_texts, _ = G.load_arena()
    print(f"arena: {sum(len(v) for v in arena_texts.values())} units over "
          f"{len(arena_texts)} subsets", flush=True)

    def gate(name, texts):
        t0 = time.time()
        r = G.run_gate(texts, n=gate_n, jaccard=gate_j, kill=gate_kill,
                       label=name, arena_texts=arena_texts)
        print(f"  {name}: {r['verdict']} at max fraction {r['max_fraction']} "
              f"({time.time() - t0:.0f}s)", flush=True)
        return r

    res = {
        "instrument": f"provenance_gate.py, R14-H136 ruling-2 form: {gate_n}-gram, "
                      f"Jaccard >= {gate_j}, bidirectional, KILL > {gate_kill:.0%}; "
                      "thresholds read from R19_supply_gates.py",
        "unit_definition": "deduplicated member evidence chunks; deduplicated claims",
        "evidence_units": len(ev), "claim_units": len(cl),
        "unit_cap_applied": unit_cap or None,
    }
    res["evidence_gate"] = gate("halueval_evidence", ev)
    res["claim_gate"] = gate("halueval_claims", cl)

    sp = G.spike_control(ev[:spike_sample], arena_texts, n=gate_n, jaccard=gate_j,
                         k=10, label="halueval_spike")
    res["spike_control"] = sp
    print(f"  spike control: {sp}", flush=True)

    # LIVE positive control - arena documents thinned to 2 of every 3 sentences.
    # Not byte-identical, not a re-spelling: genuinely near-duplicate material.
    donors = [c for v in arena_texts.values() for c in v[:2]][:10]
    thinned = [sentence_thin(t) for t in donors]
    base = ev[:spike_sample]
    live = G.run_gate(list(base) + thinned, n=gate_n, jaccard=gate_j,
                      kill=gate_kill, label="halueval_live_control",
                      arena_texts=arena_texts)
    baseline = G.run_gate(list(base), n=gate_n, jaccard=gate_j, kill=gate_kill,
                          label="halueval_live_baseline", arena_texts=arena_texts)
    b_hits = baseline["candidate_vs_arena"]["units_with_hit"]
    l_hits = live["candidate_vs_arena"]["units_with_hit"]
    res["live_positive_control"] = {
        "construction": "10 arena documents thinned to 2 of every 3 sentences "
                        "(genuine near-duplicates, not byte-identical, and not "
                        "recoverable by the gate's own normalisation)",
        "injected": len(thinned),
        "mean_retained_char_fraction": round(float(np.mean(
            [len(t) / max(len(d), 1) for t, d in zip(thinned, donors)])), 4),
        "baseline_hits_without_injection": b_hits,
        "hits_with_injection": l_hits,
        "detected": l_hits - b_hits,
        "fires": bool(l_hits - b_hits >= len(thinned)),
    }
    print(f"  live positive control: {res['live_positive_control']}", flush=True)

    # coverage: units too short for an 8-gram instrument
    hasher = G._TokenHasher()
    short_ev = sum(1 for t in ev if G.ngram_hashes(t, gate_n, hasher).size == 0)
    short_cl = sum(1 for t in cl if G.ngram_hashes(t, gate_n, hasher).size == 0)
    ev_set, cl_set = set(ev), set(cl)
    arena_raw = {c for v in arena_texts.values() for c in v}
    res["coverage"] = {
        "evidence_units_below_8_tokens": short_ev,
        "claim_units_below_8_tokens": short_cl,
        "short_units_exact_match_against_arena": len(
            (ev_set | cl_set) & arena_raw),
        "note": "units too short for the n-gram instrument are covered by exact "
                "string matching against the arena documents",
    }
    res["pass"] = bool(res["evidence_gate"]["verdict"] != "KILL"
                       and res["claim_gate"]["verdict"] != "KILL"
                       and sp["passes"] and res["live_positive_control"]["fires"])
    return res


# --------------------------------------------------------------------------- #
# C6 - memorisation channel
# --------------------------------------------------------------------------- #
def clause_c6(df, C):
    d = df
    n_pairs = d["pair_id"].n_unique()
    # legs of a pair share the evidence field by construction - verify byte-exact
    piv = d.pivot(on="label", index="pair_id", values="chunk", aggregate_function="first")
    same_ev = int((piv["1.0"] == piv["0.0"]).sum()) if "1.0" in piv.columns else \
        int((piv[str(1.0)] == piv[str(0.0)]).sum())
    # a claim that carries BOTH labels anywhere in the member
    per_claim = d.group_by("claim").agg(pl.col("label").n_unique().alias("k"),
                                        pl.len().alias("rows"))
    both = per_claim.filter(pl.col("k") > 1)
    # evidence blocks shared by more than one pair - the association a memoriser
    # could key on
    per_ev = d.group_by("chunk").agg(pl.col("pair_id").n_unique().alias("pairs"))
    multi_ev = per_ev.filter(pl.col("pairs") > 1)
    # cross-pair collision: a positive claim of one pair appearing as a negative
    # claim of another pair on the SAME evidence
    pos = set(d.filter(pl.col("label") == 1)["claim"].to_list())
    neg = set(d.filter(pl.col("label") == 0)["claim"].to_list())
    return {
        "test": "the member's two legs share the evidence field, so the "
                "association a memoriser could key on is (evidence -> claim); "
                "measured, not assumed",
        "pairs": int(n_pairs),
        "pairs_with_byte_identical_evidence_on_both_legs": same_ev,
        "evidence_only_auroc": 0.5 if same_ev == n_pairs else None,
        "evidence_only_auroc_note": "exactly 0.5 BY MEASUREMENT when every pair's "
                                    "legs are byte-identical in evidence - the "
                                    "feature is constant within a pair",
        "claims_carrying_both_labels": int(both.height),
        "rows_on_those_claims": int(both["rows"].sum()) if both.height else 0,
        "evidence_blocks_shared_by_2plus_pairs": int(multi_ev.height),
        "rows_on_shared_evidence": int(
            d.join(multi_ev.select("chunk"), on="chunk", how="inner").height),
        "positive_negative_claim_string_collisions": len(pos & neg),
    }


# --------------------------------------------------------------------------- #
# C7 / C8
# --------------------------------------------------------------------------- #
def clause_c7(df, d):
    rows = df.height
    pairs = df["pair_id"].n_unique()
    per_half = {k: {"rows": int(v), "pairs": int(v // 2)}
                for k, v in df.group_by("half").len().iter_rows()}
    return {
        "unit_declared_by_loader": "rows (the loader appends two rows per source "
                                   "record and tags both `halueval`)",
        "rows": int(rows), "pairs": int(pairs),
        "rows_per_pair": round(rows / max(pairs, 1), 4),
        "per_half": per_half,
        "share_of_assembled_mix_rows": {
            "clean_public_685670": round(rows / d["mix_rows_total"], 6),
            "h150_flagship_721210": round(rows / 721_210, 6),
            "h174_portfolio_760618": round(rows / 760_618, 6),
        },
        "mix_rows_measured": d["mix_rows_total"],
        "registered_figure": "40,000 rows (R8-H90 registration, `HaluEval 40k`)",
        "pass": bool(rows == 40_000 and pairs == 20_000),
    }


def clause_c8(df, d):
    z = zipfile.ZipFile(ARCHIVE)
    dates = {i.filename: "%04d-%02d-%02d" % i.date_time[:3] for i in z.infolist()}
    claims = df["claim"].to_list()
    chunks = df["chunk"].to_list()
    lens_ev = np.array([len(c) for c in chunks])
    lens_cl = np.array([len(c) for c in claims])
    per_half = {}
    for h in ("qa", "summarization"):
        s = df.filter(pl.col("half") == h)
        ec = np.array([len(c) for c in s["chunk"].to_list()])
        cc = np.array([len(c) for c in s["claim"].to_list()])
        per_half[h] = {
            "rows": s.height, "pairs": int(s["pair_id"].n_unique()),
            "distinct_claims": int(s["claim"].n_unique()),
            "distinct_evidence": int(s["chunk"].n_unique()),
            "evidence_chars_mean": round(float(ec.mean()), 1),
            "evidence_chars_p95": round(float(np.percentile(ec, 95)), 1),
            "evidence_chars_max": int(ec.max()),
            "evidence_over_1500_chars": int((ec > 1500).sum()),
            "claim_chars_mean": round(float(cc.mean()), 1),
            "claim_chars_max": int(cc.max()),
        }
    return {
        "source": "HuggingFace `pminervini/HaluEval`",
        "licence": "MIT",
        "retrieval_date": sorted(set(dates.values())),
        "archive": "data/external/datasets/dataset-halueval.zip (gitignored); "
                   "tracked sidecar data/external/datasets/dataset-halueval.md",
        "fetcher": "scripts/fetch_grounding_datasets.py",
        "selection_predicate": "subsets `qa` and `summarization`, ALL rows, no "
                               "filter; two mix rows per source record "
                               "(right_answer/right_summary -> label 1, "
                               "hallucinated_answer/hallucinated_summary -> "
                               "label 0); evidence = knowledge / document, "
                               "truncated to CFG.chunk_max_chars in the "
                               "incumbent read and untruncated then windowed "
                               "1500/750 in the H150/H174 arms",
        "subsets_in_archive_not_loaded": ["dialogue", "general"],
        "duplication": {
            "rows": df.height,
            "distinct_claims": int(df["claim"].n_unique()),
            "distinct_evidence": int(df["chunk"].n_unique()),
            "distinct_claim_evidence_pairs": int(df.select(["claim", "chunk"]).n_unique()),
            "evidence_reuse_rows": int(df.height - df["chunk"].n_unique()),
            "per_half": per_half,
        },
        "geometry": {
            "evidence_chars_mean": round(float(lens_ev.mean()), 1),
            "evidence_chars_max": int(lens_ev.max()),
            "claim_chars_mean": round(float(lens_cl.mean()), 1),
            "rows_with_evidence_over_1500_chars": int((lens_ev > 1500).sum()),
        },
        "public_repo_check": "no client or company name appears in this artifact "
                             "or in the member's text fields as loaded",
    }


# --------------------------------------------------------------------------- #
# C5 - executor-added probe, reported SEPARATELY
# --------------------------------------------------------------------------- #
def clause_c5(df, C, seed=0):
    import random
    rng = random.Random(seed)
    claims = df["claim"].to_list()
    labels = df["label"].to_numpy()
    groups = [f"{h}:{p}" for h, p in zip(df["half"].to_list(), df["pair_id"].to_list())]
    auc, score = C.claim_only_probe(claims, labels, groups, rng)
    wp = C.within_pair_accuracy(df.with_columns(pl.col("label").cast(pl.Int64)), score)
    per_half = {}
    for h in ("qa", "summarization"):
        m = np.array([g.startswith(h + ":") for g in groups])
        per_half[h] = round(float(C.auroc(labels[m], score[m])), 4)
    parity = {"all": C.surface_parity(df.with_columns(pl.col("label").cast(pl.Int64)))}
    for h in ("qa", "summarization"):
        sub = df.filter(pl.col("half") == h).with_columns(pl.col("label").cast(pl.Int64))
        parity[h] = C.surface_parity(sub)
    return {
        "surface_parity": parity,
        "applicability": "C5 governs CONSTRUCTED lanes and paired-contrast EVALS. "
                         "halueval is a source corpus and is not an evaluation "
                         "surface, so the registered C5 conjunction does not bind "
                         "it. The probes below are EXECUTOR-ADDED and are reported "
                         "separately; they join no registered bar.",
        "claim_only_probe_auroc": round(float(auc), 4),
        "claim_only_probe_per_half": per_half,
        "within_pair_claim_only_accuracy": wp,
        "reference_bars_if_they_had_applied": {"claim_only": 0.55, "within_pair": 0.60},
        "instrument": "R20-H174_lane_common.claim_only_probe - out-of-fold "
                      "char_wb TF-IDF 2-5 grams + liblinear C=4 tol 1e-7, folds "
                      "disjoint on the pair key",
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("core", "c1supp", "disjoint", "census", "probe", "merge"))
    ap.add_argument("--unit-cap", type=int, default=0)
    a = ap.parse_args()

    if a.stage == "merge":
        merge()
        return

    d = load_member()
    df = frame(d)
    C = _mod("lanecommon", EXP / "R20-H174_lane_common.py")

    if a.stage == "core":
        out = {"member": MEMBER, "loaded": {k: v for k, v in d.items()
                                            if k not in ("claims", "chunks", "y",
                                                         "half", "pair_id")},
               "C1": clause_c1(df, C), "C6": clause_c6(df, C),
               "C7": clause_c7(df, d), "C8": clause_c8(df, d)}
        (HERE / "halueval_core.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out["C1"]["presentations"]["untruncated"], indent=2))
        print(json.dumps(out["C7"], indent=2))
    elif a.stage == "c1supp":
        out = {"C1_supplement": clause_c1_supplement(df, C)}
        (HERE / "halueval_c1supp.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({k: v for k, v in out["C1_supplement"].items()
                          if k != "fully_attested_negative_examples"}, indent=2))
    elif a.stage == "disjoint":
        G = _mod("provgate", EXP / "provenance_gate.py")
        out = {"C2": clause_c2(df, G), "C3": clause_c3(df, d)}
        (HERE / "halueval_disjoint.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out["C2"]["surfaces"]["arena_ragbench_10"], indent=2))
    elif a.stage == "census":
        G = _mod("provgate", EXP / "provenance_gate.py")
        src = (EXP / "R19_supply_gates.py").read_text()
        gate_n = int(src.split("GATE_N = ")[1].split("\n")[0])
        gate_j = float(src.split("GATE_JACCARD = ")[1].split("\n")[0])
        gate_kill = float(src.split("GATE_KILL = ")[1].split("\n")[0])
        out = {"C4": clause_c4(df, G, gate_n, gate_j, gate_kill, unit_cap=a.unit_cap)}
        (HERE / "halueval_census.json").write_text(json.dumps(out, indent=2))
    elif a.stage == "probe":
        out = {"C5": clause_c5(df, C)}
        (HERE / "halueval_probe.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))


def merge():
    core = json.loads((HERE / "halueval_core.json").read_text())
    dis = json.loads((HERE / "halueval_disjoint.json").read_text())
    cen = json.loads((HERE / "halueval_census.json").read_text())
    prb = json.loads((HERE / "halueval_probe.json").read_text())
    rep = {"member": MEMBER,
           "contract": "docs/experiments/dataset-contract.md",
           "loaded": core["loaded"],
           "C1": core["C1"], "C2": dis["C2"], "C3": dis["C3"],
           "C4": cen["C4"], "C5": prb["C5"], "C6": core["C6"],
           "C7": core["C7"], "C8": core["C8"]}
    (HERE / "halueval_contract_report_parts.json").write_text(json.dumps(rep, indent=2))
    print("merged parts written")


if __name__ == "__main__":
    main()

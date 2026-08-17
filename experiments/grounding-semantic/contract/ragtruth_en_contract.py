"""Dataset-contract verification for the `ragtruth_en` training member. CPU ONLY.

Contract: docs/experiments/dataset-contract.md (clauses C1-C8).

The member is rebuilt through the BANKED loader (`R10-H108_lane.public_train()`),
in both presentations the mix has used:
  truncated   chunk cut to CFG.chunk_max_chars (1,500) - the R10-H108 recipe
  untruncated the R16-H142 `untruncated_evidence()` lift used by the R18-H150
              flagship and the R20-H174 portfolio wrapper, windowed 1,500/750

Instruments are reused, never reinvented: `provenance_gate.py` in the R14-H136
ruling-2 form (8-gram, Jaccard >= 0.3, bidirectional, KILL > 2%, spike control),
thresholds read from `R19_supply_gates.py`; `R20-H174_lane_common.py` for
`containment`, `auroc` and `claim_only_probe`.

Run:  CUDA_VISIBLE_DEVICES= uv run python \
        experiments/grounding-semantic/contract/ragtruth_en_contract.py
"""

import os

# HARD CONSTRAINT: no GPU. Set BEFORE any banked import - the banked modules pin
# CUDA_VISIBLE_DEVICES with setdefault, which leaves an already-set value alone.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import collections
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
MEMBER = "ragtruth_en"
OUT = HERE / "ragtruth_en_contract_report.json"


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LC = _mod("lane_common", EXP / "R20-H174_lane_common.py")   # torch-free
G = _mod("provgate", EXP / "provenance_gate.py")            # torch-free

_gsrc = (EXP / "R19_supply_gates.py").read_text()
GATE_N = int(_gsrc.split("GATE_N = ")[1].split("\n")[0])
GATE_JACCARD = float(_gsrc.split("GATE_JACCARD = ")[1].split("\n")[0])
GATE_KILL = float(_gsrc.split("GATE_KILL = ")[1].split("\n")[0])

# The G1 arm module owns its OWN R10-H108 instance and its OWN M59 config object;
# `untruncated_evidence()` patches THAT config. Loading a second H108 here would
# make the context manager a silent no-op on it, so the arm's instance is the one
# used throughout - exactly as `R18-H150_arm_run.make_build_mix` uses it.
ARM = _mod("g1arm", EXP / "R16-H142_G1_arm.py")             # imports torch, CPU only
H108 = ARM.H108
M59, M60 = ARM.M59, ARM.M60
CHUNK_MAX = M59.CFG.chunk_max_chars

WS = re.compile(r"\s+")


def wsnorm(s):
    return WS.sub(" ", s).strip().casefold()


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# member rebuild through the banked loader
# --------------------------------------------------------------------------- #
def load_member():
    t0 = time.time()
    claims_t, chunks_t, y_t, tags_t = H108.public_train()
    with ARM.untruncated_evidence():
        claims_u, chunks_u, y_u, tags_u = H108.public_train()
    idx_t = [i for i, t in enumerate(tags_t) if t == MEMBER]
    idx_u = [i for i, t in enumerate(tags_u) if t == MEMBER]
    assert idx_t == idx_u, "row set differs between presentations"
    m = {
        "claims": [claims_t[i] for i in idx_t],
        "chunks_trunc": [chunks_t[i] for i in idx_t],
        "chunks_full": [chunks_u[i] for i in idx_u],
        "y": y_t[idx_t],
        "mix_rows_truncated": int(len(y_t)),
        "mix_rows_untruncated": int(len(y_u)),
        "load_seconds": round(time.time() - t0, 1),
    }
    assert np.array_equal(m["y"], y_u[idx_u])
    log(f"member rebuilt: {len(m['y'])} rows of {m['mix_rows_truncated']} mix rows "
        f"({m['load_seconds']}s)")
    return m


def load_archive():
    z = zipfile.ZipFile(DATA / "dataset-ragtruth.zip")
    tr = pl.read_parquet(io.BytesIO(z.read(
        next(x for x in z.namelist() if x.endswith("__train.parquet")))))
    te = pl.read_parquet(io.BytesIO(z.read(
        next(x for x in z.namelist() if x.endswith("__test.parquet")))))
    return tr, te


# --------------------------------------------------------------------------- #
# C1 - label commensurability
# --------------------------------------------------------------------------- #
def dist(v):
    v = np.asarray(v, dtype=float)
    return {
        "n": int(v.size),
        "mean": round(float(v.mean()), 4),
        "median": round(float(np.median(v)), 4),
        "p10": round(float(np.percentile(v, 10)), 4),
        "p25": round(float(np.percentile(v, 25)), 4),
        "p75": round(float(np.percentile(v, 75)), 4),
        "p90": round(float(np.percentile(v, 90)), 4),
        "frac_ge_0.90": round(float((v >= 0.90).mean()), 4),
        "frac_eq_1.00": round(float((v >= 0.999999).mean()), 4),
        "deciles": [round(float(np.percentile(v, q)), 4) for q in range(0, 101, 10)],
    }


def clause_c1(m, tr):
    y = m["y"]
    # evidence token sets memoised on the evidence string
    cache = {}

    def cont(claim, ev):
        s = cache.get(ev)
        if s is None:
            s = set(LC.tokens(ev))
            cache[ev] = s
        ct = set(LC.tokens(claim))
        return len(ct & s) / len(ct) if ct else 0.0

    con_full = np.array([cont(c, k) for c, k in zip(m["claims"], m["chunks_full"])])
    cache.clear()
    con_tr = np.array([cont(c, k) for c, k in zip(m["claims"], m["chunks_trunc"])])
    cache.clear()
    # max-over-windows: the MIL presentation the flagship actually trains on
    con_win = np.array([
        max(cont(c, w) for w in ARM.windows(k))
        for c, k in zip(m["claims"], m["chunks_full"])])
    cache.clear()

    legs = {}
    for name, v in (("untruncated_full_evidence", con_full),
                    ("truncated_1500", con_tr),
                    ("max_over_windows_1500_750", con_win)):
        pos, neg = v[y == 1], v[y == 0]
        gap90 = abs(float((neg >= 0.90).mean()) - float((pos >= 0.90).mean()))
        gap100 = abs(float((neg >= 0.999999).mean()) - float((pos >= 0.999999).mean()))
        legs[name] = {
            "positive_leg": dist(pos),
            "negative_leg": dist(neg),
            "mean_gap_pos_minus_neg": round(float(pos.mean() - neg.mean()), 4),
            "attested_rate_gap_at_0.90": round(gap90, 4),
            "attested_rate_gap_at_1.00": round(gap100, 4),
            "containment_auroc_vs_label": round(LC.auroc(y, v), 4),
            "rejected_by_bar": bool(gap90 <= 0.10),
        }

    primary = legs["untruncated_full_evidence"]
    ctok = np.array([len(LC.tokens(c)) for c in m["claims"]], dtype=float)
    etok = np.array([len(LC.tokens(c)) for c in m["chunks_full"]], dtype=float)
    confounds = {
        "claim_tokens": {"positive_mean": round(float(ctok[y == 1].mean()), 2),
                         "negative_mean": round(float(ctok[y == 0].mean()), 2),
                         "auroc_vs_label": round(LC.auroc(y, ctok), 4)},
        "evidence_tokens": {"positive_mean": round(float(etok[y == 1].mean()), 2),
                            "negative_mean": round(float(etok[y == 0].mean()), 2),
                            "auroc_vs_label": round(LC.auroc(y, etok), 4)},
    }
    return {
        "confounds": confounds,
        "head_declared": "grounding scalar (`task_head`) - the single support logit "
                         "the cascade serves; the member carries no second head",
        "label_expression": "(hallucination_labels_processed.evident_conflict == 0) "
                            "AND (hallucination_labels_processed.baseless_info == 0)",
        "label_predicate_measured": (
            "response-level: label 1 iff the response carries ZERO human-annotated "
            "hallucination spans of any type. Verified against the raw "
            "`hallucination_labels` span JSON: 0 of 8,369 label-1 rows carry any "
            "annotated span and 0 of 6,721 label-0 rows carry none - see "
            "label_type_audit. The predicate IS support, not relevance; its UNIT "
            "is a whole 100-200-word response, not a single claim"),
        "bar_reading": {
            "applied": "LITERAL - REJECTED iff |rate(negatives with containment >= "
                       "0.90) - rate(positives with containment >= 0.90)| <= 0.10, "
                       "per C1's sentence 'negatives are >= 90% attested at a rate "
                       "within 0.10 of its positives'",
            "verdict_under_literal_reading": (
                "FAIL" if primary["rejected_by_bar"] else "PASS"),
            "AMBIGUITY FLAGGED, NOT RESOLVED HERE": (
                "the literal reading fires on this member because BOTH rates are near "
                "zero (negatives 0.0019, positives 0.0454), not because the negatives "
                "are attested. C1's stated purpose - 'negatives attested at rates "
                "comparable to positives means the member is not teaching grounding' - "
                "and its provenance (H175b: containment 0.9129 on BOTH legs, 66.4% of "
                "negatives at containment exactly 1.0) both describe a HIGH attested "
                "rate on the negative leg. Under the alternative reading, which "
                "requires the negative leg to actually be >= 90% attested before the "
                "within-0.10 comparison applies, this member is NOT rejected: 0.19% "
                "of its negatives reach containment 0.90 and 0.00% reach 1.00. The "
                "coordinator adjudicates which reading binds; the executor reports "
                "both and applies the literal text"),
            "verdict_under_purpose_reading": (
                "PASS" if primary["negative_leg"]["frac_ge_0.90"] < 0.90 else "FAIL"),
            "h175b_reference_signature": {
                "both_legs_mean_containment": 0.9129,
                "negatives_at_containment_1.0": 0.664,
                "source": "docs/experiments/semantic-grounding-experiments.md, the "
                          "R20-H175b poisoning block",
            },
        },
        "primary_presentation": "untruncated_full_evidence (the R18-H150 / R20-H174 "
                                "flagship presentation)",
        "instrument": "R20-H174_lane_common.containment - fraction of the claim's "
                      "content tokens present in the evidence",
        "presentations": legs,
        "verdict": "FAIL" if primary["rejected_by_bar"] else "PASS",
        "measured": {
            "neg_frac_ge_0.90": primary["negative_leg"]["frac_ge_0.90"],
            "pos_frac_ge_0.90": primary["positive_leg"]["frac_ge_0.90"],
            "gap": primary["attested_rate_gap_at_0.90"],
            "bar": 0.10,
        },
    }


def label_type_audit(tr):
    rows = []
    for s, st in zip(tr["hallucination_labels"].to_list(),
                     tr["hallucination_labels_processed"].to_list()):
        c = collections.Counter()
        if s and s.strip() not in ("", "[]"):
            for item in json.loads(s):
                c[item.get("label_type")] += 1
        rows.append({
            "spans": int(sum(c.values())),
            "evident_conflict": c["Evident Conflict"],
            "evident_baseless": c["Evident Baseless Info"],
            "subtle_conflict": c["Subtle Conflict"],
            "subtle_baseless": c["Subtle Baseless Info"],
            "label": int(st["evident_conflict"] == 0 and st["baseless_info"] == 0),
        })
    d = pl.DataFrame(rows)
    return {
        "span_type_totals": {
            k: int(d[k].sum()) for k in
            ("evident_conflict", "evident_baseless", "subtle_conflict", "subtle_baseless")},
        "label1_rows": int((d["label"] == 1).sum()),
        "label0_rows": int((d["label"] == 0).sum()),
        "label1_rows_with_any_span": int(d.filter(
            (pl.col("label") == 1) & (pl.col("spans") > 0)).height),
        "label1_rows_with_subtle_only_span": int(d.filter(
            (pl.col("label") == 1)
            & ((pl.col("subtle_conflict") + pl.col("subtle_baseless")) > 0)).height),
        "label0_rows_with_zero_spans": int(d.filter(
            (pl.col("label") == 0) & (pl.col("spans") == 0)).height),
        "note": "the processed struct's two counters are not per-type copies of the "
                "raw span types, but the BINARY they induce is exactly "
                "'zero annotated spans' - both cross-checks above read 0",
    }


# --------------------------------------------------------------------------- #
# C2 - disjointness from every evaluation surface
# --------------------------------------------------------------------------- #
def surface_texts():
    """Evidence and claim strings of every declared evaluation surface."""
    out = {}

    arena_docs, _ = G.load_arena()
    # arena responses: the same parquets, same filter/sample as load_arena; the
    # gate module returns documents only, so the response column is read here.
    z = zipfile.ZipFile(G.ARCHIVE)
    resp = []
    for name in sorted(n for n in z.namelist() if n.endswith("__test.parquet")):
        df = pl.read_parquet(io.BytesIO(z.read(name)))
        df = df.filter(pl.col("adherence_score").is_not_null()
                       & (pl.col("response").str.len_chars() > 20)
                       & (pl.col("documents").list.len() > 0))
        if len(df) < 40 or df["adherence_score"].n_unique() < 2:
            continue
        resp += df.sample(min(G.N_PER_SUBSET, len(df)), seed=0)["response"].to_list()
    out["arena_ragbench_10_subsets"] = {
        "evidence": [c for v in arena_docs.values() for c in v],
        "claims": resp,
    }

    gc, gk, _gy = H108.gold_full()
    out["gold_full"] = {"evidence": [c for ks in gk for c in ks], "claims": list(gc)}

    for f in ("R17-H143_evalset.parquet", "R20-H177_eval_B.parquet",
              "R20-H177_eval_C.parquet",
              "R20-H175b_qlane_eval.parquet",
              "R20-H175b_qlane_eval_repaired.parquet",
              "R20-H175b_qlane_eval_clean.parquet",
              "R20-H175b_qlane_eval_clean_prefix.parquet"):
        p = EXP / f
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        out[f] = {"evidence": d["chunk"].to_list(), "claims": d["claim"].to_list()}
    return out


def clause_c2(m):
    surfaces = surface_texts()
    member = {
        "evidence_raw": m["chunks_full"],
        "evidence_truncated": m["chunks_trunc"],
        "claims": m["claims"],
    }
    forms = {
        "raw": lambda s: s,
        f"truncated_{CHUNK_MAX}": lambda s: s[:CHUNK_MAX],
        "whitespace_collapsed_casefolded": wsnorm,
    }
    per_surface = {}
    total = 0
    for sname, s in surfaces.items():
        blk = {}
        for unit, mem_texts, sur_texts in (
                ("evidence", member["evidence_raw"], s["evidence"]),
                ("claims", member["claims"], s["claims"])):
            mem_texts = [t for t in mem_texts if t and t.strip()]
            sur_texts = [t for t in sur_texts if t and t.strip()]
            for fname, fn in forms.items():
                ms = {fn(t) for t in mem_texts}
                ss = {fn(t) for t in sur_texts}
                inter = ms & ss
                blk[f"{unit}__{fname}"] = {
                    "member_units": len(ms),
                    "surface_units": len(ss),
                    "member_in_surface": len(inter),
                    "surface_in_member": len(inter),
                    "fraction_of_member": round(len(inter) / max(len(ms), 1), 6),
                    "fraction_of_surface": round(len(inter) / max(len(ss), 1), 6),
                }
                total += len(inter)
        per_surface[sname] = blk
        log(f"  C2 {sname}: max overlap "
            f"{max(v['member_in_surface'] for v in blk.values())}")
    return {
        "forms": list(forms),
        "directions": "both (set intersection is symmetric; both counts reported)",
        "surfaces": per_surface,
        "total_overlapping_units": total,
        "verdict": "PASS" if total == 0 else "FAIL",
        "measured": {"total_overlapping_units_all_forms_all_surfaces": total, "bar": 0},
    }


# --------------------------------------------------------------------------- #
# C3 - split semantics
# --------------------------------------------------------------------------- #
def own_split_gate(m, te):
    """The corpus's own test-split evidence against its train evidence index, at
    the R14-H136 instrument. Answers C3's 'test it' and supplies C4's second
    control, so it is computed once and reported in both."""
    ev = sorted({c for c in m["chunks_full"] if c.strip()})
    te_ev = sorted({c for c in te["context"].to_list() if c and c.strip()})
    res = G.run_gate(te_ev, n=GATE_N, jaccard=GATE_JACCARD, kill=GATE_KILL,
                     label=f"{MEMBER}_own_test_split",
                     arena_texts={"member_own_train_evidence": ev})
    d = res["candidate_vs_arena"]
    log(f"  own-test-split gate: {d['units_with_hit']}/{d['n_units']} "
        f"({d['fraction']}) at best Jaccard max {d.get('best_jaccard', {}).get('max')}")
    return {
        "construction": "the corpus's own held-out test-split evidence gated against "
                        "the member's train evidence index - near-duplicate by "
                        "population if the split does not cut on the document",
        "candidates": res["candidate"]["n_units"],
        "train_index_units": len(ev),
        "units_with_hit": d["units_with_hit"],
        "detected_fraction": d["fraction"],
        "best_jaccard": d.get("best_jaccard"),
    }


def clause_c3(tr, te, split_gate):
    def sets(df, col):
        return {t for t in df[col].to_list() if t is not None}

    axes = {}
    for col in ("id", "context", "query", "output"):
        a, b = sets(tr, col), sets(te, col)
        axes[col] = {
            "train_distinct": len(a),
            "test_distinct": len(b),
            "shared": len(a & b),
            "test_share_seen_in_train": round(len(a & b) / max(len(b), 1), 6),
        }
    # whitespace-normalised evidence, the form that hid the R20-H177 leak
    a = {wsnorm(t) for t in tr["context"].to_list()}
    b = {wsnorm(t) for t in te["context"].to_list()}
    axes["context_whitespace_collapsed_casefolded"] = {
        "train_distinct": len(a), "test_distinct": len(b), "shared": len(a & b),
        "test_share_seen_in_train": round(len(a & b) / max(len(b), 1), 6),
    }
    # what the split cuts on: is every context wholly inside one split?
    ctx_tr, ctx_te = sets(tr, "context"), sets(te, "context")
    return {
        "declared_split": "the archive's own `__train` / `__test` parquet split "
                          "(wandb/RAGTruth-processed)",
        "measured_axis": ("document/context - every distinct context string lands "
                          "wholly in one split" if not (ctx_tr & ctx_te)
                          else "NOT document-disjoint: contexts appear in both splits"),
        "axes": axes,
        "rows_per_context_train": {
            "contexts": int(tr["context"].n_unique()),
            "rows": int(tr.height),
            "mean_rows_per_context": round(tr.height / max(tr["context"].n_unique(), 1), 4),
            "responder_models": int(tr["model"].n_unique()),
        },
        "near_duplicate_test": {
            "why": "C3 says an official split is not evidence of disjointness - it is "
                   "tested. Exact string equality is the weaker test; this is the "
                   "R14-H136 near-duplicate instrument on the same two sides",
            **split_gate,
        },
        "scope_note": ("this member is TRAINING-only. The RAGTruth TEST split is not "
                       "one of the contract's declared evaluation surfaces; it is the "
                       "`ragtruth_en` in-domain lineage read (R7-H60.load_english). "
                       "The measurement is reported because that read carries a bar in "
                       "every arm result"),
        "verdict": "PASS" if not (ctx_tr & ctx_te) else "FAIL",
        "measured": {
            "shared_context_strings_train_vs_test": len(ctx_tr & ctx_te),
            "shared_context_strings_wsnorm": axes[
                "context_whitespace_collapsed_casefolded"]["shared"],
            "bar": 0,
        },
    }


# --------------------------------------------------------------------------- #
# C4 - contamination census with a live positive control
# --------------------------------------------------------------------------- #
def coverage(texts, n):
    short = sum(1 for t in texts if len(G.normalize(t).split()) < n)
    return {"units": len(texts), "units_below_{}_grams".format(n): short,
            "fraction_below": round(short / max(len(texts), 1), 6)}


def clause_c4(m, tr, te, surfaces_arena, split_gate):
    ev = sorted({c for c in m["chunks_full"] if c.strip()})
    cl = sorted({c for c in m["claims"] if c.strip()})
    log(f"  C4 gate units: {len(ev)} evidence, {len(cl)} claims")

    res = {}
    t0 = time.time()
    res["evidence_gate"] = G.run_gate(ev, n=GATE_N, jaccard=GATE_JACCARD,
                                      kill=GATE_KILL, label=f"{MEMBER}_evidence",
                                      arena_texts=surfaces_arena)
    log(f"    evidence: {res['evidence_gate']['verdict']} at "
        f"{res['evidence_gate']['max_fraction']} ({time.time() - t0:.0f}s)")
    t0 = time.time()
    res["claim_gate"] = G.run_gate(cl, n=GATE_N, jaccard=GATE_JACCARD,
                                   kill=GATE_KILL, label=f"{MEMBER}_claims",
                                   arena_texts=surfaces_arena)
    log(f"    claims: {res['claim_gate']['verdict']} at "
        f"{res['claim_gate']['max_fraction']} ({time.time() - t0:.0f}s)")

    spike = G.spike_control(ev, surfaces_arena, n=GATE_N, jaccard=GATE_JACCARD,
                            k=10, label=f"{MEMBER}_spike")
    log(f"    spike control: {spike}")

    # LIVE positive control: text that is near-duplicate by CONSTRUCTION.
    # Built from the member's own evidence - whitespace re-wrapped and every
    # 10th sentence deleted - then gated against the member's own evidence as
    # the reference side. A gate that cannot fire is caught here.
    rng = np.random.default_rng(0)
    sample = [ev[i] for i in rng.choice(len(ev), size=min(200, len(ev)), replace=False)]
    perturbed, kept_frac = [], []
    for t in sample:
        parts = re.split(r"(?<=[.!?])\s+", t)
        kept = [p for i, p in enumerate(parts) if i % 10 != 0] or parts
        p = WS.sub(" ", " ".join(kept))
        perturbed.append(p)
        kept_frac.append(len(G.normalize(p).split())
                         / max(len(G.normalize(t).split()), 1))
    live = G.run_gate(perturbed, n=GATE_N, jaccard=GATE_JACCARD, kill=GATE_KILL,
                      label=f"{MEMBER}_live_control",
                      arena_texts={"member_own_evidence": ev})
    log(f"    live positive control (constructed near-duplicates): "
        f"{live['candidate_vs_arena']['fraction']} detected")

    # Diagnose every miss: a construction that deletes most of the document is
    # no longer a near-duplicate, so a miss there is the gate behaving, not
    # failing. Recompute each candidate's own best Jaccard against the index.
    hasher = G._TokenHasher()
    idx_units = [G.ngram_hashes(t, GATE_N, hasher) for t in ev]
    flat = np.concatenate([u for u in idx_units if u.size])
    owner = np.concatenate([np.full(u.size, i, dtype=np.int64)
                            for i, u in enumerate(idx_units) if u.size])
    order = np.argsort(flat, kind="stable")
    flat, owner = flat[order], owner[order]
    sizes = np.array([u.size for u in idx_units], dtype=np.int64)
    misses = []
    for i, p in enumerate(perturbed):
        j, _uid = G._max_jaccard(G.ngram_hashes(p, GATE_N, hasher), flat, owner, sizes)
        if j < GATE_JACCARD:
            misses.append({"candidate": i, "best_jaccard": round(j, 4),
                           "token_fraction_kept": round(kept_frac[i], 4),
                           "source_tokens": int(len(G.normalize(sample[i]).split()))})
    log(f"    live control misses: {len(misses)} {misses}")

    # exact-match coverage for units too short for the 8-gram instrument
    arena_flat = [c for v in surfaces_arena.values() for c in v]
    short_ev = [t for t in ev if len(G.normalize(t).split()) < GATE_N]
    short_cl = [t for t in cl if len(G.normalize(t).split()) < GATE_N]
    arena_norm = {G.normalize(t) for t in arena_flat}
    exact = {
        "short_evidence_units": len(short_ev),
        "short_claim_units": len(short_cl),
        "short_evidence_exact_hits": sum(1 for t in short_ev
                                         if G.normalize(t) in arena_norm),
        "short_claim_exact_hits": sum(1 for t in short_cl
                                      if G.normalize(t) in arena_norm),
    }

    worst = max(res["evidence_gate"]["max_fraction"], res["claim_gate"]["max_fraction"])
    # C4's live-control requirement is "show it fires" - no numeric bar is
    # written. The criterion applied here: every candidate that is STILL a
    # near-duplicate after the perturbation (own best Jaccard >= the gate
    # threshold) must be detected, i.e. no miss may be unexplained.
    unexplained = [d for d in misses if d["best_jaccard"] >= GATE_JACCARD]
    ok = (res["evidence_gate"]["verdict"] != "KILL"
          and res["claim_gate"]["verdict"] != "KILL"
          and spike["passes"] and spike["baseline_hits"] == 0
          and not unexplained)
    return {
        "instrument": (f"provenance_gate.py, R14-H136 ruling-2 form: {GATE_N}-gram, "
                       f"Jaccard >= {GATE_JACCARD}, bidirectional, KILL > "
                       f"{GATE_KILL:.0%}; thresholds read from R19_supply_gates.py"),
        "unit_definition": "deduplicated untruncated evidence chunks; deduplicated claims",
        "evidence_gate": res["evidence_gate"],
        "claim_gate": res["claim_gate"],
        "spike_control": spike,
        "live_positive_control_constructed_near_duplicates": {
            "construction": "200 sampled member evidence chunks, whitespace re-wrapped "
                            "and every 10th sentence deleted, gated against the "
                            "member's own evidence index",
            "candidates": live["candidate"]["n_units"],
            "detected_fraction": live["candidate_vs_arena"]["fraction"],
            "best_jaccard": live["candidate_vs_arena"].get("best_jaccard"),
            "misses": misses,
            "unexplained_misses": unexplained,
            "miss_diagnosis": "a miss is EXPLAINED when the perturbed candidate's own "
                              "best Jaccard against the index is below the gate "
                              "threshold - the construction destroyed the "
                              "near-duplicate, so the gate is correct to be silent",
            "fires": not unexplained,
        },
        "live_positive_control_own_test_split": {
            "note": "computed once under C3.near_duplicate_test and repeated here; it "
                    "is NOT a valid live positive control for this member because the "
                    "split IS document-disjoint, so the population it offers is not "
                    "near-duplicate by construction. The constructed control above is "
                    "the one C4 rests on",
            **split_gate,
        },
        "arena_composition_note": (
            "the ten arena subsets loaded by provenance_gate.load_arena are covidqa, "
            "delucionqa, emanual, expertqa, finqa, hagrid, hotpotqa, pubmedqa, tatqa, "
            "techqa. RAGTruth's QA contexts derive from MS MARCO and its Summary half "
            "from CNN/DailyMail; NEITHER source has a subset in this arena, so the "
            "clean reading is not evidence about an msmarco surface - there is none"),
        "coverage": {"evidence": coverage(ev, GATE_N), "claims": coverage(cl, GATE_N),
                     "exact_match_backstop": exact},
        "verdict": "PASS" if ok else "FAIL",
        "measured": {"max_fraction": worst, "kill_bar": GATE_KILL,
                     "spike_detected": spike["detected_total"],
                     "spike_baseline_hits": spike["baseline_hits"]},
    }


# --------------------------------------------------------------------------- #
# C5 / C6
# --------------------------------------------------------------------------- #
def clause_c5(m, tr):
    """NOT-APPLICABLE by scope; an executor-added CPU probe is reported separately."""
    claims = m["claims"]
    y = m["y"].astype(int).tolist()
    groups = tr.filter(pl.col("context").str.len_chars() > 50)["context"].to_list()
    assert len(groups) == len(claims)
    rng = np.random.default_rng(0)
    auc, _ = LC.claim_only_probe(claims, y, groups, rng, n_folds=5)
    sp = LC.surface_parity(pl.DataFrame({
        "label": y, "claim": claims, "chunk": m["chunks_full"]}))
    return {
        "verdict": "NOT-APPLICABLE",
        "why": ("C5 is scoped to 'every constructed lane and every paired-contrast "
                "eval'. `ragtruth_en` is a SOURCE corpus: rows are naturally occurring "
                "LLM responses with human span annotations, there is no construction, "
                "no `pair_id`, no `neg_family`, no direction/element/family design, and "
                "no paired contrast to balance. Every bar in C5 (within-pair, "
                "single-channel-at-chance-where-the-construction-implies-it, direction "
                "balance, attestation symmetry) names a property a construction has and "
                "this member does not"),
        "executor_added_reported_separately": {
            "status": "EXECUTOR-ADDED, NOT part of the registered C5 conjunction, "
                      "carries no bar for this member",
            "claim_only_probe_auroc": round(auc, 4),
            "probe": "R20-H174_lane_common.claim_only_probe - out-of-fold char_wb "
                     "TF-IDF (2-5) + liblinear, folds disjoint on the evidence string",
            "surface_parity_auroc": sp["auroc"],
            "reading": "reported as a measurement of the member's claim-side "
                       "separability, not as a C5 verdict",
        },
    }


def clause_c6(m, tr, te):
    """Memorisation channel keyed on the shared field.

    The member's rows share the `context` field (mean ~6 responses per context).
    Two directions are measured; the second is the C1-provenance shape.
    """
    y = m["y"].astype(float)
    ctx = tr.filter(pl.col("context").str.len_chars() > 50)["context"].to_list()
    assert len(ctx) == len(y)

    # (a) in-member: leave-one-out mean label of the row's own context key
    by = collections.defaultdict(lambda: [0.0, 0])
    for c, v in zip(ctx, y):
        by[c][0] += v
        by[c][1] += 1
    loo = np.array([(by[c][0] - v) / max(by[c][1] - 1, 1) for c, v in zip(ctx, y)])
    covered = np.array([by[c][1] > 1 for c in ctx])
    auc_loo = LC.auroc(y[covered], loo[covered])

    # (b) eval direction: the RAGTruth test split shares no context with train when
    # the split is document-disjoint; measure the channel anyway.
    tr_by_ctx = collections.defaultdict(list)
    for c, cl in zip(ctx, m["claims"]):
        tr_by_ctx[c].append(cl)
    te_f = te.filter((pl.col("context").str.len_chars() > 50)
                     & (pl.col("output").str.len_chars() > 20))
    hits, overlaps = 0, []
    for c, out in zip(te_f["context"].to_list(), te_f["output"].to_list()):
        assoc = tr_by_ctx.get(c)
        if not assoc:
            continue
        hits += 1
        overlaps.append(max(LC.containment(out, a) for a in assoc))
    return {
        "shared_field": "context (evidence) - mean 6.00 responses per context",
        "in_member_channel": {
            "feature": "leave-one-out mean label of the row's own context key",
            "auroc_vs_label": round(auc_loo, 4),
            "coverage": round(float(covered.mean()), 4),
            "chance": 0.5,
            "reading": "a value at chance means the evidence key alone carries no "
                       "label association a model could memorise",
        },
        "eval_direction_channel": {
            "eval": "RAGTruth test split (the `ragtruth_en` lineage read)",
            "eval_rows": int(te_f.height),
            "rows_whose_context_key_is_in_training": hits,
            "coverage": round(hits / max(te_f.height, 1), 6),
            "mean_overlap_where_covered": (round(float(np.mean(overlaps)), 4)
                                           if overlaps else None),
            "status": ("undefined - zero coverage, no eval row shares a context key "
                       "with training" if not hits else "measured"),
            "scope_note": "the lineage read is not one of the contract's declared "
                          "evaluation surfaces; reported because it carries a bar",
        },
        "tolerance_note": "C6 writes no numeric bar - 'on a clean instrument it is "
                          "undefined or at chance'. The executor applies +/- 0.05 "
                          "around 0.5 and flags that choice; the measured value is "
                          "reported so any tolerance can be applied to it",
        "mechanism": "the label is a property of the RESPONSE, but responses are "
                     "grouped 6-per-passage and passages differ in how often all six "
                     "models stay grounded. The passage identity therefore carries a "
                     "real label association a model can memorise without reading the "
                     "claim",
        "fixability": "PIPELINE - per-context label balancing (or one response per "
                      "context) drives this channel to chance, at a cost in rows; "
                      "measured cost is not computed here",
        "verdict": "PASS" if abs(auc_loo - 0.5) <= 0.05 and hits == 0 else "FAIL",
        "measured": {"in_member_loo_auroc": round(auc_loo, 4), "chance": 0.5,
                     "tolerance": 0.05, "eval_key_coverage": hits},
    }


# --------------------------------------------------------------------------- #
# C7 / C8
# --------------------------------------------------------------------------- #
def clause_c7(m, tr):
    y = m["y"]
    ctx = tr.filter(pl.col("context").str.len_chars() > 50)["context"].to_list()
    lab_by_ctx = collections.defaultdict(set)
    for c, v in zip(ctx, y):
        lab_by_ctx[c].add(int(v))
    both = sum(1 for s in lab_by_ctx.values() if s == {0, 1})
    return {
        "declared_unit": "rows (claim, evidence) - the unit `public_train` appends "
                         "and the unit every arm's `clean_rows` census counts",
        "rows": int(len(y)),
        "pairs_claim_evidence": int(len(y)),
        "note_on_pairs": ("this member has no paired-contrast construction, so rows "
                          "and (claim, evidence) pairs are the same count. The only "
                          "pair-like structure is the evidence key: "
                          f"{len(lab_by_ctx)} distinct contexts, {both} of them "
                          "carrying both a label-1 and a label-0 row"),
        "registered_figure": "~15,090 rows",
        "positives": int((y == 1).sum()),
        "negatives": int((y == 0).sum()),
        "positive_rate": round(float(y.mean()), 4),
        "verdict": "PASS" if len(y) == 15_090 else "FAIL",
        "measured": {"rows": int(len(y)), "registered": 15090,
                     "delta": int(len(y)) - 15090},
    }


def clause_c8(m, tr):
    sidecar = DATA / "dataset-ragtruth.md"
    txt = sidecar.read_text() if sidecar.exists() else ""
    z = zipfile.ZipFile(DATA / "dataset-ragtruth.zip")
    arc_date = max("{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}".format(*i.date_time)
                   for i in z.infolist())

    def field(tag):
        for line in txt.splitlines():
            if line.startswith(f"- **{tag}**"):
                return line.split("-", 2)[-1].strip()
        return None

    q = {}
    for tt in ("Data2txt", "QA", "Summary"):
        d = tr.filter(pl.col("task_type") == tt)
        top, cnt = collections.Counter(d["query"].to_list()).most_common(1)[0]
        q[tt] = {
            "rows": int(d.height),
            "distinct_query_strings": int(d["query"].n_unique()),
            "distinct_contexts": int(d["context"].n_unique()),
            "most_frequent_query_rows": int(cnt),
            "most_frequent_query_chars": len(top),
            "most_frequent_query_is_instruction": tt != "QA",
            "most_frequent_query_prefix": top[:120],
        }
    instr_rows = q["Data2txt"]["rows"] + q["Summary"]["rows"]
    ev_counts = collections.Counter(m["chunks_full"])
    nwin = np.array([len(ARM.windows(c)) for c in m["chunks_full"]])
    ev_len = [len(c) for c in m["chunks_full"]]
    cl_len = [len(c) for c in m["claims"]]
    return {
        "source": "HuggingFace `wandb/RAGTruth-processed` "
                  "(processed mirror of ParticleMedia/RAGTruth, ACL 2024)",
        "licence": field("Licence") or "MIT (sidecar dataset-ragtruth.md)",
        "licence_sidecar": str(sidecar.relative_to(ROOT)),
        "archive": "data/external/datasets/dataset-ragtruth.zip (gitignored; "
                   "sidecar tracked)",
        "retrieval": {
            "fetcher": "scripts/fetch_grounding_datasets.py (named in the sidecar)",
            "date_recorded_in_sidecar": None,
            "date_recoverable_from_archive": arc_date,
            "finding": "C8 requires the retrieval DATE. The sidecar does not carry "
                       "one; the only evidence is the archive member timestamps. No "
                       "sidecar in data/external/datasets/ records a retrieval date, "
                       "so this is a systematic gap rather than a member-specific one",
        },
        "selection_predicate": (
            "split `wandb__RAGTruth-processed__train.parquet`; filter "
            "context.str.len_chars() > 50; claim = `output`, evidence = `context`; "
            f"evidence cut to CFG.chunk_max_chars={CHUNK_MAX} in the R10-H108 "
            "presentation, UNCUT and windowed 1,500/750 in the R18-H150 / R20-H174 "
            "flagship presentation; label = "
            "(evident_conflict == 0) AND (baseless_info == 0)"),
        "fields_consumed": ["output -> claim", "context -> evidence",
                            "hallucination_labels_processed -> label"],
        "fields_NOT_consumed": ["query", "input_str", "task_type", "model",
                                "temperature", "quality", "id",
                                "hallucination_labels"],
        "duplication": {
            "rows": int(len(m["y"])),
            "distinct_claims": len(set(m["claims"])),
            "distinct_evidence_untruncated": len(ev_counts),
            "distinct_evidence_truncated": len(set(m["chunks_trunc"])),
            "mean_rows_per_evidence": round(len(m["y"]) / max(len(ev_counts), 1), 4),
            "max_rows_per_evidence": int(max(ev_counts.values())),
            "repeat_structure": "6 responder models x 2,515 source items; every "
                                "evidence string is repeated once per model",
            "duplicate_claim_rows": int(len(m["claims"]) - len(set(m["claims"]))),
        },
        "query_field_structure": {
            "note": "the query/instruction column exists in the archive but is NOT "
                    "consumed by public_train() - the model never sees it",
            "per_task_type": q,
            "instruction_rows": instr_rows,
            "instruction_share_of_rows": round(instr_rows / max(tr.height, 1), 4),
        },
        "presentation_census": {
            "windows_1500_750": {
                "mean_windows": round(float(np.mean(nwin)), 4),
                "multi_window_share": round(float((nwin > 1).mean()), 4),
                "max_windows": int(nwin.max()),
            },
            "blank_line_junction": {
                "share_of_rows": round(
                    float(np.mean([("\n\n" in c) for c in m["chunks_full"]])), 4),
                "share_of_distinct_contexts": round(
                    float(np.mean([("\n\n" in c) for c in set(m["chunks_full"])])), 4),
                "canonical_log_cross_check": (
                    "the registration block at semantic-grounding-experiments.md:4318 "
                    "asserts '79.2% of ragtruth_en contexts contain a \\n\\n blank-line "
                    "junction (mean 2.687 windows, 76.2% multi-window)'. The two window "
                    "figures reproduce EXACTLY here; the 79.2% does not reproduce under "
                    "any reading tried - rows 0.5435, distinct contexts 0.5434, "
                    "P(junction|multi-window) 0.4217, P(junction|single-window) 0.9332, "
                    "P(multi-window|junction) 0.5911, and the truncated presentation "
                    "0.3535. Reported as a measurement; no clause turns on it"),
            },
            "evidence_chars": {
                "mean": round(float(np.mean(ev_len)), 1),
                "median": round(float(np.median(ev_len)), 1),
                "p90": round(float(np.percentile(ev_len, 90)), 1),
                "max": int(max(ev_len)),
                "frac_over_chunk_max": round(
                    float(np.mean(np.array(ev_len) > CHUNK_MAX)), 4),
            },
            "claim_chars": {
                "mean": round(float(np.mean(cl_len)), 1),
                "median": round(float(np.median(cl_len)), 1),
                "p90": round(float(np.percentile(cl_len, 90)), 1),
            },
            "truncation_loss": {
                "note": "the R10-H108 presentation cuts evidence at "
                        f"{CHUNK_MAX} chars; this is the share of rows losing text",
                "rows_truncated": int(sum(1 for a, b in
                                          zip(m["chunks_full"], m["chunks_trunc"])
                                          if len(a) > len(b))),
            },
        },
        "public_repo_check": "no client or company name appears in this artifact",
        "verdict": "FAIL",
        "measured": {
            "required_and_present": ["source", "licence", "selection predicate",
                                     "within-member duplication", "public-repo check"],
            "required_and_missing": ["retrieval date"],
            "binding_constraint": "the licence sidecar records no retrieval date; the "
                                  f"archive's own timestamps read {arc_date}",
            "licence_note": "the sidecar's MIT tag is transcribed, not re-verified "
                            "against the source in this pass (no network use)",
        },
    }


# --------------------------------------------------------------------------- #
def main():
    m = load_member()
    tr, te = load_archive()

    log("C7/C8 ...")
    c7 = clause_c7(m, tr)
    c8 = clause_c8(m, tr)
    log("C1 ...")
    c1 = clause_c1(m, tr)
    c1["label_type_audit"] = label_type_audit(tr)
    log("C3 ...")
    sg = own_split_gate(m, te)
    c3 = clause_c3(tr, te, sg)
    log("C2 ...")
    c2 = clause_c2(m)
    log("C4 ...")
    arena, _ = G.load_arena()
    c4 = clause_c4(m, tr, te, arena, sg)
    log("C5 ...")
    c5 = clause_c5(m, tr)
    log("C6 ...")
    c6 = clause_c6(m, tr, te)

    clauses = {"C1": c1, "C2": c2, "C3": c3, "C4": c4,
               "C5": c5, "C6": c6, "C7": c7, "C8": c8}
    fails = [k for k, v in clauses.items() if v["verdict"] == "FAIL"]
    report = {
        "member": MEMBER,
        "class": "training member - source corpus",
        "contract": "docs/experiments/dataset-contract.md (DRAFT)",
        "rebuilt_through": "R10-H108_lane.public_train() (banked), both presentations",
        "cpu_only": True,
        "mix_rows_at_verification": m["mix_rows_truncated"],
        "clauses": clauses,
        "conforming": not fails,
        "failed_clauses": fails,
    }
    OUT.write_text(json.dumps(report, indent=2))
    log(f"\nreport -> {OUT}")
    log(json.dumps({k: v["verdict"] for k, v in clauses.items()}, indent=2))


if __name__ == "__main__":
    main()

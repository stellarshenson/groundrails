"""Round-2 latency benchmark - H12 gate / H13 early-exit / H14 fused vs adopted cascade.

Measures per-claim warm latency (evidence-side vectors precomputed = cache hit) on the
deployed LATENCY-hint int8 engines at k=8, same 150-claim seed-0 sample as the H11
bench so deltas are attributable. Per claim it times every stage variant once, then
composes the candidate configs arithmetically:

  base          always both cross-encoders (round-0 reference)
  cascade       adopted H11 band on the reranker max (current serving)
  +exit         H13 - rank-ordered progressive reranker, stop at the pass edge
  +gate+exit    H12 - extreme pre-filter cosines skip both cross-encoders
  mixed fused   gate + exit + H14 fused NLI replacing per-chunk NLI in-band
  full fused    gate + H14 fused reranker AND fused NLI (band on the fused score)

H13 verdict-invariance is asserted per claim: early-exit category (flag / in-band /
pass) must equal the same-schedule full scoring; agreement vs the deployed bucketed
batch is reported separately (int8 batch-composition jitter).

Read-only benchmark; prints a report. Run on CPU:
    python scripts/bench_grounder_round2.py 2>&1 | tee logs/grounding-round2-bench.log
"""
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np

from grounding_models import load_gold
from grounding_openvino import (
    CASCADE_BAND,
    compile_ir,
    embed_vectors,
    fused_context_chunks,
    load_ov_hf,
    nli_max,
    pack_sentences,
    rerank_max,
    rerank_max_early,
    split_sentences,
)

K = 8
BAND = CASCADE_BAND            # adopted H11 band [0.01, 0.66]
GATE = (0.493, 0.739)          # H12 gate (a0, b0) - strictly-better point on the OOF
                               # no-worse frontier (reports/grounding_hypotheses.md)
SCHEDULE = (1, 1, 2, 4)        # H13 progressive batches in pre-filter rank order
N_SAMPLE = 150

recs = load_gold()
rng = np.random.default_rng(0)
sample = [recs[i] for i in rng.choice(len(recs), N_SAMPLE, replace=False)]

_, etok, edir = load_ov_hf("bge-m3", compile=False)
_, rtok, rdir = load_ov_hf("bge-reranker-v2-m3", compile=False)
_, ntok, ndir = load_ov_hf("mDeBERTa-v3-nli", compile=False)
emb = compile_ir(edir / "openvino_model.xml", hint="LATENCY")
rr = compile_ir(rdir / "openvino_model.xml", hint="LATENCY")
nli = compile_ir(ndir / "openvino_model.xml", hint="LATENCY")

# ----------------------------------------------------------- reranker batch microbench
# one padded forward per batch size on a median-length pair - informs the H13 schedule
pairs = [(r["claim"], r["chunks"][0]) for r in sample[:24]]
print("=== reranker int8 forward cost vs batch size (median-length pairs) ===")
for b in (1, 2, 4, 8):
    sub = pairs[:b]
    enc = rtok([p[0] for p in sub], [p[1] for p in sub], padding=True, truncation=True,
               max_length=512, return_tensors="np")
    feed = {n: enc[n].astype(np.int64) for n in enc
            if n in {i.get_any_name() for i in rr.inputs}}
    rr(feed)                                       # warm-up this shape
    t0 = time.perf_counter()
    reps = 10
    for _ in range(reps):
        rr(feed)
    dt = (time.perf_counter() - t0) / reps * 1000
    print(f"  batch {b}: {dt:6.1f} ms/forward  ({dt / b:5.1f} ms/pair)", flush=True)

# ----------------------------------------------------------- warm-regime precompute
# evidence-side artifacts are cacheable per answer (same rationale as the chunk-vector
# RAG cache): chunk vectors, sentence splits and sentence vectors are precomputed;
# the claim side (embed, rank, gate, packing, cross-encoders) is timed per claim.
uchunk: dict[str, int] = {}
for r in sample:
    for c in r["chunks"]:
        uchunk.setdefault(c, len(uchunk))
chunk_texts = list(uchunk)
sent_lists = [split_sentences(c) or [c] for c in chunk_texts]
usent: dict[str, int] = {}
for ss in sent_lists:
    for s in ss:
        usent.setdefault(s, len(usent))
sent_texts = list(usent)
print(f"\nprecompute: {len(chunk_texts)} unique chunks, {len(sent_texts)} unique "
      "sentences", flush=True)
chvec_u = embed_vectors(emb, etok, chunk_texts)
svec_u = embed_vectors(emb, etok, sent_texts)
chunk_vecs = [chvec_u[[uchunk[c] for c in r["chunks"]]] for r in sample]

# engine warm-up
for r in sample[:3]:
    embed_vectors(emb, etok, [r["claim"]])
    rerank_max(rr, rtok, r["claim"], r["chunks"][:K])
    nli_max(nli, ntok, r["claim"], r["chunks"][:K])

# ----------------------------------------------------------- per-claim stage timings
cols = ("pre", "rr_ref", "rr_ee", "rr_full", "nli", "rr_fz", "nli_fz", "fz_build")
T = {c: [] for c in cols}
gate_lo, gate_hi = [], []
s_ref_l, s_ee_l, s_full_l, s_fz_l, pairs_ee = [], [], [], [], []
for r, dv in zip(sample, chunk_vecs):
    # pre-filter: claim embed + cosines + rank; the gate signal is the max cosine
    t0 = time.perf_counter()
    cv = embed_vectors(emb, etok, [r["claim"]])[0]
    cos = dv @ cv
    idx = np.argsort(-cos)
    ranked = [r["chunks"][i] for i in idx[:K]]
    cmax = float(cos.max())
    t1 = time.perf_counter()
    # deployed reranker (length-bucketed single batch, original order)
    ch = [r["chunks"][i] for i in sorted(idx[:K])]
    s_ref = rerank_max(rr, rtok, r["claim"], ch)
    t2 = time.perf_counter()
    # H13 early-exit (rank order, progressive schedule)
    s_ee, n_ee = rerank_max_early(rr, rtok, r["claim"], ranked, pass_edge=BAND[1],
                                  schedule=SCHEDULE)
    t3 = time.perf_counter()
    # same schedule without exit - the H13 verdict-equality reference
    s_full, _ = rerank_max_early(rr, rtok, r["claim"], ranked, pass_edge=np.inf,
                                 schedule=SCHEDULE)
    t4 = time.perf_counter()
    # per-chunk NLI (deployed in-band stage)
    nli_max(nli, ntok, r["claim"], ch)
    t5 = time.perf_counter()
    # H14 fused contexts - v2 salience packing built per claim from cached vectors
    ctx1 = fused_context_chunks(ranked)
    sents = [s for i in idx[:K] for s in sent_lists[uchunk[r["chunks"][i]]]]
    sims = svec_u[[usent[s] for s in sents]] @ cv
    ctx2 = pack_sentences(sents, sims) if sents else ctx1
    t6 = time.perf_counter()
    s_fz = rerank_max(rr, rtok, r["claim"], [ctx2])
    t7 = time.perf_counter()
    nli_max(nli, ntok, r["claim"], [ctx2])
    t8 = time.perf_counter()
    for c, v in zip(cols, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t7 - t6,
                           t8 - t7, t6 - t5)):
        T[c].append(v)
    gate_lo.append(cmax <= GATE[0])
    gate_hi.append(cmax >= GATE[1])
    s_ref_l.append(s_ref)
    s_ee_l.append(s_ee)
    s_full_l.append(s_full)
    s_fz_l.append(s_fz)
    pairs_ee.append(n_ee)

T = {c: np.array(v) * 1000 for c, v in T.items()}
gate0 = np.array(gate_lo) | np.array(gate_hi)
s_ref, s_ee = np.array(s_ref_l), np.array(s_ee_l)
s_full, s_fz = np.array(s_full_l), np.array(s_fz_l)
pairs_ee = np.array(pairs_ee)


def cat(s):
    return np.where(s < BAND[0], "flag", np.where(s > BAND[1], "pass", "band"))


# H13 verdict-invariance: early-exit category == same-schedule full category (exact -
# identical batch shapes give identical int8 scores; the exit only fires once the
# verdict is final). Agreement vs the deployed bucketed batch reported (jitter).
mism = (cat(s_ee) != cat(s_full)).sum()
assert mism == 0, f"H13 verdict mismatch vs same-schedule reference: {mism}"
agree = (cat(s_ee) == cat(s_ref)).mean()
print(f"\nH13 verdict equality vs same-schedule full scoring: exact ({N_SAMPLE}/"
      f"{N_SAMPLE})")
print(f"H13 category agreement vs deployed bucketed batch: {agree:.1%} "
      "(int8 batch-composition jitter)")
print(f"H13 pairs scored: mean {pairs_ee.mean():.1f} / {K}  (exit rate "
      f"{(pairs_ee < K).mean():.0%})")
print(f"H12 stage-0 gate rate on the sample: {gate0.mean():.0%} "
      f"(lo {np.array(gate_lo).mean():.0%} / hi {np.array(gate_hi).mean():.0%})")

# ----------------------------------------------------------- composed configurations
in_band = (BAND[0] <= s_ee) & (s_ee <= BAND[1])
in_band_fz = (BAND[0] <= s_fz) & (s_fz <= BAND[1])
g = gate0

base = T["pre"] + T["rr_ref"] + T["nli"]
casc = T["pre"] + T["rr_ref"] + np.where(in_band, T["nli"], 0.0)
ex = T["pre"] + T["rr_ee"] + np.where(in_band, T["nli"], 0.0)
gx = T["pre"] + np.where(g, 0.0, T["rr_ee"] + np.where(in_band, T["nli"], 0.0))
mixed = T["pre"] + np.where(g, 0.0, T["rr_ee"]
                            + np.where(in_band, T["fz_build"] + T["nli_fz"], 0.0))
full = T["pre"] + np.where(g, 0.0, T["fz_build"] + T["rr_fz"]
                           + np.where(in_band_fz, T["nli_fz"], 0.0))

print(f"\n=== round-2 warm latency (k={K}, LATENCY hint, n={N_SAMPLE}, band "
      f"[{BAND[0]}, {BAND[1]}], gate {GATE}, schedule {SCHEDULE}) ===")
print(f"  stage means: pre {T['pre'].mean():.0f} | rr bucketed {T['rr_ref'].mean():.0f} "
      f"| rr early-exit {T['rr_ee'].mean():.0f} | nli k=8 {T['nli'].mean():.0f} | "
      f"fused build {T['fz_build'].mean():.1f} | rr fused {T['rr_fz'].mean():.0f} | "
      f"nli fused {T['nli_fz'].mean():.0f} ms")
for name, t in (("base (always both)", base), ("cascade (adopted)", casc),
                ("cascade + H13 exit", ex), ("H12 gate + exit", gx),
                ("mixed fused (gate+exit+fused NLI)", mixed),
                ("full fused (gate+fused rr+fused NLI)", full)):
    print(f"  {name:38s} mean {t.mean():6.0f} ms  median {np.median(t):6.0f} ms  "
          f"p90 {np.percentile(t, 90):6.0f} ms")
print("DONE", flush=True)

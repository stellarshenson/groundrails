"""H11 cascade latency benchmark - reranker-first vs current always-both serving.

Measures per-claim warm latency (chunk embeddings precomputed = cache hit) on the
deployed LATENCY-hint int8 engines at k=8. The cascade runs the NLI only when the
reranker max falls inside the uncertainty band; out-of-band claims take the
reranker-only verdict. Band from the OOF frontier (reports/grounding_hypotheses.md).

Read-only benchmark; prints a report. Run on CPU:
    python scripts/bench_grounder_cascade.py 2>&1 | tee logs/grounding-cascade-bench.log
"""
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np

from grounding_models import load_gold
from grounding_openvino import (
    compile_ir,
    embed_vectors,
    load_ov_hf,
    nli_max,
    rerank_max,
)

K = 8
BAND = (0.01, 0.66)   # skip 60% at macro-F1 0.797 (zero loss) on the OOF frontier
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

# warm regime: chunk vectors precomputed once (the RAG cache); per-claim work is
# claim embed + rank + cross-encoders
chunk_vecs = [embed_vectors(emb, etok, r["chunks"]) for r in sample]

# engine warm-up
for r in sample[:3]:
    embed_vectors(emb, etok, [r["claim"]])
    rerank_max(rr, rtok, r["claim"], r["chunks"][:K])
    nli_max(nli, ntok, r["claim"], r["chunks"][:K])

t_pre, t_rr, t_nli, rr_scores, in_band = [], [], [], [], []
for r, dv in zip(sample, chunk_vecs):
    t0 = time.perf_counter()
    cv = embed_vectors(emb, etok, [r["claim"]])[0]
    idx = np.argsort(-(dv @ cv))[:K]
    ch = [r["chunks"][i] for i in sorted(idx)]
    t1 = time.perf_counter()
    s = rerank_max(rr, rtok, r["claim"], ch)
    t2 = time.perf_counter()
    nli_max(nli, ntok, r["claim"], ch)
    t3 = time.perf_counter()
    t_pre.append(t1 - t0)
    t_rr.append(t2 - t1)
    t_nli.append(t3 - t2)
    rr_scores.append(s)
    in_band.append(BAND[0] <= s <= BAND[1])

t_pre, t_rr, t_nli = np.array(t_pre), np.array(t_rr), np.array(t_nli)
in_band = np.array(in_band)
base = (t_pre + t_rr + t_nli) * 1000
casc = (t_pre + t_rr + np.where(in_band, t_nli, 0.0)) * 1000
skip = 1.0 - in_band.mean()

print(f"\n=== H11 cascade warm latency (k={K}, LATENCY hint, n={N_SAMPLE}, "
      f"band [{BAND[0]}, {BAND[1]}]) ===")
print(f"  stage means: pre-filter {t_pre.mean()*1000:.0f} ms | reranker {t_rr.mean()*1000:.0f} ms "
      f"| nli {t_nli.mean()*1000:.0f} ms")
print(f"  serving skip rate (NLI avoided): {skip:.0%}")
print(f"  baseline per-claim: mean {base.mean():.0f} ms  median {np.median(base):.0f} ms  "
      f"p90 {np.percentile(base, 90):.0f} ms")
print(f"  cascade  per-claim: mean {casc.mean():.0f} ms  median {np.median(casc):.0f} ms  "
      f"p90 {np.percentile(casc, 90):.0f} ms")
print(f"  mean saving: {1 - casc.mean()/base.mean():.0%}  (gate >= 20%)")
print("DONE", flush=True)

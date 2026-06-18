"""Measure the four mechanical latency levers for the single-engine OpenVINO grounder.

#6 batching  - report current serving shape (per-claim) vs answer-batched potential
#7 max_length - token-length distribution of claims / chunks / (claim,chunk) pairs -> a cap
#8 k=5 / bucket - k=5 vs k=8 stack macro-F1 from the cached pipeline scores; bucket saving estimate
#9 OV hint    - LATENCY vs THROUGHPUT compile hint, single-claim wall time at k=8

Read-only benchmark; prints a report, writes nothing. Run on CPU.
"""
import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import openvino as ov

from grounding_models import load_gold, SCORES_DIR, metrics
from grounding_openvino import (load_ov_hf, HF_REPOS, embed_vectors,
                                                 topk_chunks, rerank_max, nli_max)

K = 8
recs = load_gold()
print(f"records={len(recs)}  chunks/claim median={np.median([len(r['chunks']) for r in recs]):.0f}", flush=True)

# ----------------------------------------------------------------- #7 token lengths
print("\n=== #7 token-length distribution (find the max_length cap) ===", flush=True)
_, rtok, _ = load_ov_hf("bge-reranker-v2-m3", compile=False)
_, ntok, _ = load_ov_hf("mDeBERTa-v3-nli", compile=False)
_, etok, _ = load_ov_hf("bge-m3", compile=False)

claims = [r["claim"] for r in recs]
flat = [c for r in recs for c in r["chunks"]]
# sample pairs: each claim with its first chunk (representative of the cross-encoder input)
pairs = [(r["claim"], r["chunks"][0]) for r in recs if r["chunks"]]


def dist(name, lengths):
    a = np.array(lengths)
    print(f"  {name:24s} p50={np.percentile(a,50):4.0f} p90={np.percentile(a,90):4.0f} "
          f"p95={np.percentile(a,95):4.0f} p99={np.percentile(a,99):4.0f} max={a.max():4.0f} "
          f">512={100*(a>512).mean():4.1f}%", flush=True)


dist("claim alone (e/r/n tok)", [len(etok(c)["input_ids"]) for c in claims])
dist("chunk alone (embedder)", [len(etok(c)["input_ids"]) for c in flat])
dist("reranker pair (c,ch)", [len(rtok(a, b)["input_ids"]) for a, b in pairs])
dist("nli pair (ch,c)", [len(ntok(b, a)["input_ids"]) for a, b in pairs])

# ----------------------------------------------------------------- #8 k=5 vs k=8 quality
print("\n=== #8 k=5 vs k=8 stack macro-F1 (cached pipeline scores) ===", flush=True)
cache = SCORES_DIR / "ov_pipeline_sub800.npz"
if cache.exists():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from grounding_ensemble import best_macro
    d = np.load(cache); rr_k, nli_k, KS = d["rerank_k"], d["nli_k"], list(d["ks"])
    # subset labels align with the cached EVAL_N=800 draw (seed 42)
    np.random.seed(42)
    y_all = np.array([r["label"] for r in recs])
    idx = np.random.choice(len(recs), 800, replace=False)
    y = y_all[idx] if rr_k.shape[0] == 800 else y_all[:rr_k.shape[0]]
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for j, k in enumerate(KS):
        X = np.column_stack([rr_k[:, j], nli_k[:, j]])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        p = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
        mf = best_macro(p, y)[0]
        print(f"  top-k {k:>2}: macro-F1 {mf:.3f}", flush=True)
else:
    print("  (no cache - run score_ov_pipeline first)", flush=True)

# ----------------------------------------------------------------- #9 OV hint + max_len
print("\n=== #9 LATENCY vs THROUGHPUT hint + max_length cap (single-claim wall time @ k=8) ===", flush=True)
core = ov.Core()
print(f"  CPU threads visible: {os.cpu_count()}", flush=True)
np.random.seed(0)
lat_recs = [recs[i] for i in np.random.choice(len(recs), 20, replace=False)]


def compile_hint(name, hint):
    xml = load_ov_hf(name, compile=False)[2] / "openvino_model.xml"
    return core.compile_model(core.read_model(str(xml)), "CPU", {"PERFORMANCE_HINT": hint})


def run(emb, rr, nli, etok, rtok, ntok, records, k, max_len):
    t = time.time()
    for r in records:
        ch = topk_chunks(emb, etok, r["claim"], r["chunks"], k, pool="cls")
        rerank_max(rr, rtok, r["claim"], ch, max_len=max_len)
        nli_max(nli, ntok, r["claim"], ch, max_len=max_len)
    return (time.time() - t) / len(records) * 1000


for hint in ("THROUGHPUT", "LATENCY"):
    emb = compile_hint("bge-m3", hint)
    rr = compile_hint("bge-reranker-v2-m3", hint)
    nli = compile_hint("mDeBERTa-v3-nli", hint)
    run(emb, rr, nli, etok, rtok, ntok, lat_recs[:3], K, 512)            # warm
    for ml in (512, 256):
        ms = run(emb, rr, nli, etok, rtok, ntok, lat_recs, K, ml)
        print(f"  hint={hint:11s} max_len={ml}: {ms:6.0f} ms/claim", flush=True)
print("DONE", flush=True)

"""Full end-to-end run of the adopted grounder over the 2,752-claim gold.

Executes the real serving path per claim on the deployed LATENCY-hint int8 engines -
pre-filter (claim embed + cosine rank) -> stage-0 cosine gate -> early-exit reranker
-> cascade band -> NLI + logistic stack for in-band claims - and records both sides
of the performance characteristics:

- quality: end-to-end macro-F1 / FP / FN of the deployed calibration (logistic fit on
  the round-1 pair cache, thresholds from the OOF protocol) applied to the live
  serving scores - the int8-serving check of the OOF-simulated 0.797
- latency: warm per-claim distribution over the full gold (chunk vectors precomputed,
  the RAG-cache assumption), plus the stage-0 / out-of-band / in-band composition

Run on CPU (detached, ~1 h):
    python scripts/run_grounder_full.py 2>&1 | tee logs/grounding-full-run.log
Optional smoke test: python scripts/run_grounder_full.py 12   (first N claims only)
"""
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from grounding_ensemble import best_macro
import grounding_hypotheses as gh
from grounding_models import load_gold
from grounding_openvino import (
    CASCADE_BAND,
    COSINE_GATE,
    _async_embed,
    compile_ir,
    embed_vectors,
    load_ov_hf,
    nli_max,
    rerank_max_early,
)

K = 8
SCHEDULE = (1, 1, 2, 4)

recs = load_gold()
if len(sys.argv) > 1:
    recs = recs[: int(sys.argv[1])]
y = np.array([r["label"] for r in recs])

# deployed calibration - logistic full-fit on the round-1 pair cache, thresholds from
# the established OOF protocol (T_rr ~ 0.31, T_st ~ 0.58); fit once, applied frozen
feats, y_full, _ = gh.build_features()
X_cal = np.column_stack([feats["rr_max"], feats["nli_ent_max"]])
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
clf.fit(X_cal, y_full)
_, T_rr, _ = best_macro(feats["rr_max"], y_full)
_, T_st, _ = best_macro(gh.oof_p(X_cal, y_full), y_full)
print(f"deployed calibration: T_rr={T_rr:.3f} T_st={T_st:.3f} "
      f"band={CASCADE_BAND} gate={COSINE_GATE}", flush=True)

_, etok, edir = load_ov_hf("bge-m3", compile=False)
_, rtok, rdir = load_ov_hf("bge-reranker-v2-m3", compile=False)
_, ntok, ndir = load_ov_hf("mDeBERTa-v3-nli", compile=False)

# warm-regime precompute: each unique chunk embedded once (offline batch path,
# THROUGHPUT async) - the serving loop below only ever embeds the claim
emb_tp = compile_ir(edir / "openvino_model.xml", hint="THROUGHPUT")
uchunk: dict[str, int] = {}
for r in recs:
    for c in r["chunks"]:
        uchunk.setdefault(c, len(uchunk))
chunk_texts = list(uchunk)
order = np.argsort([len(t) for t in chunk_texts])
v = _async_embed(emb_tp, etok, [chunk_texts[i] for i in order], desc="embed chunks")
chvec_u = np.empty_like(v)
chvec_u[order] = v
del emb_tp

emb = compile_ir(edir / "openvino_model.xml", hint="LATENCY")
rr = compile_ir(rdir / "openvino_model.xml", hint="LATENCY")
nli = compile_ir(ndir / "openvino_model.xml", hint="LATENCY")

for r in recs[:3]:  # engine warm-up
    embed_vectors(emb, etok, [r["claim"]])
    rerank_max_early(rr, rtok, r["claim"], r["chunks"][:2], pass_edge=np.inf,
                     schedule=SCHEDULE)
    nli_max(nli, ntok, r["claim"], r["chunks"][:2])

lat, category, flag = [], [], []
t_report = time.time()
for i, r in enumerate(recs):
    dv = chvec_u[[uchunk[c] for c in r["chunks"]]]
    t0 = time.perf_counter()
    cv = embed_vectors(emb, etok, [r["claim"]])[0]
    cos = dv @ cv
    cmax = float(cos.max())
    if cmax <= COSINE_GATE[0]:
        cat, fl = "gate-flag", True
    elif cmax >= COSINE_GATE[1]:
        cat, fl = "gate-pass", False
    else:
        ranked = [r["chunks"][j] for j in np.argsort(-cos)[:K]]
        s, _ = rerank_max_early(rr, rtok, r["claim"], ranked,
                                pass_edge=CASCADE_BAND[1], schedule=SCHEDULE)
        if s >= CASCADE_BAND[1]:
            cat, fl = "rr-pass", False
        elif s <= CASCADE_BAND[0]:
            cat, fl = "rr-flag", bool(s < T_rr)
        else:
            e = nli_max(nli, ntok, r["claim"], ranked)
            p = clf.predict_proba([[s, e]])[0, 1]
            cat, fl = "in-band", bool(p < T_st)
    lat.append(time.perf_counter() - t0)
    category.append(cat)
    flag.append(fl)
    if time.time() - t_report > 60:
        t_report = time.time()
        print(f"{i + 1}/{len(recs)} claims, mean {np.mean(lat) * 1000:.0f} ms",
              flush=True)

lat = np.array(lat) * 1000
category = np.array(category)
flag = np.array(flag)
macro, fp, fn = gh._macro_counts(flag, y)
hall = (y == 0)
recall = ((flag) & hall).sum() / hall.sum()
ffr = (flag & ~hall).sum() / (~hall).sum()

print(f"\n=== full end-to-end run (n={len(recs)}, k={K}, LATENCY hint, warm) ===")
print("  composition: " + "  ".join(
    f"{c} {np.mean(category == c):.1%}" for c in
    ("gate-flag", "gate-pass", "rr-flag", "rr-pass", "in-band")))
print(f"  quality: macro-F1 {macro:.3f}  FP {fp}  FN {fn}  "
      f"recall {recall:.0%}  false-flag {ffr:.0%}")
print(f"  latency: mean {lat.mean():.0f} ms  median {np.median(lat):.0f} ms  "
      f"p90 {np.percentile(lat, 90):.0f} ms  p99 {np.percentile(lat, 99):.0f} ms")
print("  latency by stage path: " + "  ".join(
    f"{c} {lat[category == c].mean():.0f}ms" for c in
    ("gate-flag", "gate-pass", "rr-flag", "rr-pass", "in-band")
    if (category == c).any()))
print("DONE", flush=True)

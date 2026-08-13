"""R18-H153 stratification baseline - how NON-stratified are the current batches?

The twin/G1 trainer packs batches from a flat permutation (no strata). The
author asks whether per-batch composition is actually representative. This
measures it: rebuild the clean mix, replay the banked draw-1 permutation
(seed 1142), pack batches exactly as the trainer does, and report per-batch
composition spread by DANN group and window-depth bucket.

CPU only. Polars-free (numpy + the banked trainer's own loaders).
Run: uv run python experiments/grounding-semantic/R18-H153_batch_strat_audit.py
"""

import importlib.util
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent

spec = importlib.util.spec_from_file_location("g1arm", HERE / "R16-H142_G1_arm.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

claims, wsets, y, tags = m.build_mix()
sizes = np.array([len(w) for w in wsets], dtype=np.int32)
n = len(tags)
perm = np.random.default_rng(1142).permutation(n)  # twin draw 1's permutation
batches = m.pack_batches(perm, sizes)

tags = np.asarray(tags)
sizes = np.asarray(sizes)
groups = sorted(set(tags.tolist()))
depth_bucket = np.where(sizes == 1, "w1", np.where(sizes <= 3, "w2_3", "w4p"))

per_batch = []
for b in batches:
    bt, bd = tags[b], depth_bucket[b]
    row = {"n": len(b), "mw_share": float((sizes[b] > 1).mean())}
    for g in groups:
        row[g] = float((bt == g).mean())
    for d in ("w1", "w2_3", "w4p"):
        row[d] = float((bd == d).mean())
    per_batch.append(row)

report = {"n_rows": int(n), "n_batches": len(batches), "mix_share": {}, "per_batch": {}}
for g in groups:
    vals = np.array([r[g] for r in per_batch])
    report["mix_share"][g] = round(float((tags == g).mean()), 4)
    report["per_batch"][g] = {
        "mean": round(float(vals.mean()), 4),
        "p05": round(float(np.percentile(vals, 5)), 4),
        "p95": round(float(np.percentile(vals, 95)), 4),
        "zero_share": round(float((vals == 0).mean()), 4),
    }
for d in ("w1", "w2_3", "w4p"):
    vals = np.array([r[d] for r in per_batch])
    report["per_batch"][d] = {
        "mean": round(float(vals.mean()), 4),
        "p05": round(float(np.percentile(vals, 5)), 4),
        "p95": round(float(np.percentile(vals, 95)), 4),
    }
mw = np.array([r["mw_share"] for r in per_batch])
report["multi_window_share"] = {"mean": round(float(mw.mean()), 4),
                                "p05": round(float(np.percentile(mw, 5)), 4),
                                "p95": round(float(np.percentile(mw, 95)), 4),
                                "min": round(float(mw.min()), 4),
                                "max": round(float(mw.max()), 4)}

out = HERE / "R18-H153_batch_strat_audit.json"
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report["multi_window_share"], indent=1))
print(json.dumps({g: report["per_batch"][g] for g in ("vitaminc", "ragtruth_en", "halueval")}, indent=1))
print(f"-> {out}")

"""Generate notebooks/01-kj-grounding-model-comparison.ipynb (thin orchestration).

Heavy scoring logic lives in grounding_models; this builds a
standards-compliant notebook that loads the verified gold, scores it with each
embedding model and cross-encoder on GPU, and compares separation (AUC,
false-flag @ 85% recall).
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Grounding Signal: Embedding & Cross-Encoder Comparison

**Author:** Konrad Jelen<br>
**Approach:** Score the verified golden dataset (real production claims, dual-judge
labelled supported/hallucinated) with several multilingual embedding models and
cross-encoders, directly via transformers + torch on GPU (no grounder plugin).
Goal: find which signal best separates supported claims from hallucinations on
real, paraphrased production answers - measured by ROC-AUC and false-flag rate at 85%
recall. Lexical and stock NLI already failed (85-88% false-flag); semantic e5-small
was the only usable signal (45%). This tests stronger embedders (incl. mmBERT) and
multilingual cross-encoder rerankers.""")

md("## GPU selection\nSelect the RTX 5090 before importing torch. `PCI_BUS_ID` order aligns CUDA indices with `nvidia-smi`.")
co("""import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # nvidia-smi idx 1 = RTX 5090 (32 GB, Blackwell)""")

md("## Imports")
co("""import json
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

import grounding_models as gm

%matplotlib inline
console = Console()
console.print(f"[bold green]GPU:[/] {torch.cuda.get_device_name(0)}  "
              f"({round(torch.cuda.get_device_properties(0).total_memory/1e9)} GB, "
              f"cap {torch.cuda.get_device_capability(0)})")""")

md("## Reproducibility")
co("""import random
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)""")

md("## Configuration")
co("""CHUNK_SIZE = 1100      # evidence sliding-window char size
OVERLAP    = 200
MAX_CHUNKS = 50        # cap chunks per record (bounds compute)
BATCH      = 64
DEVICE     = gm.pick_gpu()

t = Table(title="Models under test", show_lines=False)
t.add_column("kind"); t.add_column("model")
for c in gm.EMBEDDERS: t.add_row("embed", c["name"])
for c in gm.CROSS:     t.add_row(c["kind"], c["name"])
console.print(t)
console.print(f"chunk={CHUNK_SIZE}/{OVERLAP} max_chunks={MAX_CHUNKS} batch={BATCH} device={DEVICE}")""")

md("## Data loading\nVerified golden dataset: claims + their retrieved evidence (chunked), dual-judge labels.")
co("""records = gm.load_gold(CHUNK_SIZE, OVERLAP, MAX_CHUNKS)
labels = np.array([r["label"] for r in records])
from collections import Counter
console.print(f"records: {len(records)}  supported(1): {int(labels.sum())}  "
              f"hallucination(0): {int((labels==0).sum())}")
console.print(f"langs: {dict(Counter(r['lang'] for r in records))}")
console.print(f"avg chunks/record: {np.mean([len(r['chunks']) for r in records]):.1f}")""")

md("""## Score every model on GPU

Each model runs in its **own subprocess** (`python grounding_models.py <model>`),
so a CUDA crash in one model cannot poison the others. Per-claim score = max over its
evidence chunks; scores + metrics are persisted to `data/interim/model_scores/`.""")
co("""import subprocess, sys
results = []
scores_by_model = {}
ALL = [(c["name"], "embed") for c in gm.EMBEDDERS] + [(c["name"], c["kind"]) for c in gm.CROSS]
env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1",
       "HF_HUB_OFFLINE": "1"}  # all models cached; avoids HF Hub network stalls (mmBERT)

for name, kind in ALL:
    safe = name.replace("/", "__")
    console.print(f"[cyan]scoring[/] {name} ...")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "grounding_models", name],
                       env=env, cwd=str(gm.ROOT), capture_output=True, text=True)
    dt = time.time() - t0
    js = gm.SCORES_DIR / f"{safe}.json"
    if p.returncode == 0 and js.exists():
        meta = json.loads(js.read_text()); s = np.load(gm.SCORES_DIR / f"{safe}.npy")
        scores_by_model[name] = s
        b = meta["best_at_recall85"]
        results.append({"model": name.split("/")[-1], "kind": meta["kind"], "auc": meta["auc"],
                        "hall_mean": meta["hall_mean"], "supp_mean": meta["supp_mean"],
                        "ff@recall85": (b["false_flag"] if b else float("nan")),
                        "prec@recall85": (b["precision"] if b else float("nan"))})
        console.print(f"[green]OK[/] {name}  AUC={meta['auc']:.3f}  "
                      f"ff@85%rec={results[-1]['ff@recall85']:.0%}  ({dt:.0f}s)")
    else:
        err = (p.stderr.strip().splitlines() or ["(no output)"])[-1]
        console.print(f"[red]FAIL[/] {name} -> {err[:140]}")""")

md("## Results")
co("""import pandas as pd
df = pd.DataFrame(results).sort_values("auc", ascending=False).reset_index(drop=True)
df_disp = df.copy()
for c in ["auc","hall_mean","supp_mean"]: df_disp[c] = df_disp[c].map(lambda x: f"{x:.3f}")
for c in ["ff@recall85","prec@recall85"]: df_disp[c] = df_disp[c].map(lambda x: f"{x:.0%}")
console.print(df_disp.to_string(index=False))

from pathlib import Path
out = gm.ROOT / "reports" / "grounding_model_comparison.md"
lines = ["# Grounding Model Comparison (verified gold, GPU)", "",
         "Higher AUC = better separation of supported vs hallucination. "
         "ff@recall85 = false-flag rate at >=85% hallucination recall (lower is better).", "",
         df_disp.to_markdown(index=False)]
out.write_text("\\n".join(lines) + "\\n")
df.to_csv(gm.ROOT / "reports" / "grounding_model_comparison.csv", index=False)
console.print(f"[green]wrote[/] {out}")""")

md("## Plots")
co("""ok = df["model"].tolist()
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].barh(df["model"][::-1], df["auc"][::-1], color="#3b7dd8")
ax[0].set_xlabel("ROC-AUC (supported vs hallucination)"); ax[0].set_xlim(0.5, 1.0)
ax[0].set_title("Separation power by model")
ax[1].barh(df["model"][::-1], df["ff@recall85"][::-1], color="#d8643b")
ax[1].set_xlabel("false-flag rate @ 85% recall (lower better)")
ax[1].set_title("False-flag at fixed recall")
plt.tight_layout()
(gm.ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
plt.savefig(gm.ROOT / "reports" / "figures" / "grounding_model_comparison.png", dpi=120, bbox_inches="tight")
plt.show()""")

co("""# score distributions: supported vs hallucination, per model
names = list(scores_by_model)
n = len(names); cols = 3; rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
for i, name in enumerate(names):
    a = axes.flat[i]
    s = scores_by_model[name]
    a.hist(s[labels==1], bins=25, alpha=0.6, label="supported", color="#3b7dd8", density=True)
    a.hist(s[labels==0], bins=25, alpha=0.6, label="hallucination", color="#d8643b", density=True)
    a.set_title(name.split("/")[-1], fontsize=9); a.legend(fontsize=7)
for j in range(n, rows*cols): axes.flat[j].axis("off")
plt.tight_layout()
plt.savefig(gm.ROOT / "reports" / "figures" / "grounding_score_distributions.png", dpi=120, bbox_inches="tight")
plt.show()""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "dbm-ds", "display_name": "dbm-ds", "language": "python"}
path = "notebooks/01-kj-grounding-model-comparison.ipynb"
with open(path, "w") as f:
    nbf.write(nb, f)
print("wrote", path, "cells:", len(cells))

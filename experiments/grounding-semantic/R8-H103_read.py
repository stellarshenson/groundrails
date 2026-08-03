"""R8-H103 - blind reads for the Qwen3-0.6B decoder scorer.

Scores (sentence, window) pairs with the Qwen reranker convention (imported
from the trainer module so train and read tokenize identically) through BOTH
reads: the PRIMARY windowed decomposed-min (H101 windows: 1,500 chars, stride
750, final window flush to the end) and the truncated decomposed-min
(first-window lists, the gate's k[:1500]). No verdict logic - per-subset AUC
tables and means only; adjudication happens externally against the canonical
records.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      uv run python experiments/grounding-semantic/R8-H103_read.py \
        --model models/R8-H103-qwen06b-scorer
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import argparse
import importlib.util
import json
import pathlib

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = pathlib.Path(__file__).parent


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ARENA = _mod("arena", "R8-H77_unseen_arena.py")
H92 = _mod("h92", "R8-H92_decomposed_arena.py")
H103 = _mod("h103", "R8-H103_qwen_scorer.py")

WIN = 1500
STRIDE = 750
BATCH = 16  # GPU2 32GB at MAX_LEN 1,024


def windows(chunk):
    """H101's exact windowing: 1,500-char windows at stride 750, flush to the end."""
    n = len(chunk)
    if n <= WIN:
        return [chunk]
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return [chunk[s : s + WIN] for s in starts]


def read_subset(model, tok, suffix_ids, claims, chunks, y, truncated):
    sent_lists = [H92.sentences(c) for c in claims]
    flat_s, flat_w = [], []
    for sl, ks in zip(sent_lists, chunks, strict=True):
        if truncated:
            wlist = [k[:WIN] for k in ks]
        else:
            wlist = [w for k in ks for w in windows(k)]
        for s in sl:
            for w in wlist:
                flat_s.append(s)
                flat_w.append(w)
    scores = np.zeros(len(flat_s), dtype=np.float32)
    for i in range(0, len(flat_s), BATCH):
        enc = H103.encode_pairs(tok, flat_s[i : i + BATCH], flat_w[i : i + BATCH], suffix_ids)
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits.squeeze(-1)
        scores[i : i + BATCH] = torch.sigmoid(logits.float()).cpu().numpy()
    # max over windows per sentence, min over sentences per response
    resp = np.full(len(y), np.nan)
    pos = 0
    for i, sl in enumerate(sent_lists):
        wcount = len([k[:WIN] for k in chunks[i]]) if truncated else sum(
            len(windows(k)) for k in chunks[i]
        )
        sent_maxes = []
        for _ in sl:
            sent_maxes.append(scores[pos : pos + wcount].max())
            pos += wcount
        resp[i] = min(sent_maxes)
    auc, f1, _ = ARENA.M59.auc_and_f1(y, resp)
    return auc, f1, len(flat_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE.parent.parent / "models" / "R8-H103-qwen06b-scorer"))
    ap.add_argument("--out", default="R8-H103_reads.json")
    ap.add_argument("--truncated-only", action="store_true")
    ap.add_argument("--windowed-only", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    suffix_ids = tok(H103.SUFFIX, add_special_tokens=False)["input_ids"]
    model = (
        AutoModelForSequenceClassification.from_pretrained(args.model, torch_dtype=torch.float32)
        .cuda()
        .eval()
    )

    subs = ARENA.load_subsets()
    modes = []
    if not args.truncated_only:
        modes.append(("windowed", False))
    if not args.windowed_only:
        modes.append(("truncated", True))

    result = {"model": args.model}
    for name, truncated in modes:
        rows = {}
        print(f"\n=== {name} read ===", flush=True)
        for sub, (claims, chunks, y) in subs.items():
            auc, f1, n_pairs = read_subset(model, tok, suffix_ids, claims, chunks, y, truncated)
            rows[sub] = {"n": len(y), "auc": round(auc, 4), "f1": round(f1, 4),
                         "lettuce_auc": H92.LETTUCE[sub], "n_pairs": n_pairs}
            print(f"  {sub:14s} n={len(y):>4} auc {auc:.4f}  lettuce {H92.LETTUCE[sub]:.4f}",
                  flush=True)
        mean = float(np.mean([r["auc"] for r in rows.values()]))
        let = float(np.mean([r["lettuce_auc"] for r in rows.values()]))
        print(f"  {'MEAN':14s} {mean:.4f}  lettuce {let:.4f}", flush=True)
        result[name] = {"per_subset": rows, "mean": mean, "mean_lettuce": let}

    out = HERE / args.out
    out.write_text(json.dumps(result, indent=2))
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()

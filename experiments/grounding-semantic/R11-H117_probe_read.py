"""R11-H117 probe reads - ragtruth_en in-domain AUC + held-out pair accuracy.

Loads a checkpoint dir (trunk/ + dann_student.pt + tokenizer) and reports:
  - ragtruth_en AUC, the in-domain read used in training (`R10-H108_lane.py`
    evaluate(): M60.load_english() scored max-over-top_chunks)
  - held-out pair accuracy on R11-H117_heldout_pairs.parquet, overall and split
    by A7 verbatim vs non-verbatim; pair = p(seed, chunk) > p(claim, chunk),
    single chunk per member (no max-over-chunks - the locus is the chunk)

Usage (GPU1):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python \
      experiments/grounding-semantic/R11-H117_probe_read.py \
      --model models/H117-probe-lam0 --out R11-H117_probe_lam0_read.json
"""

import argparse
import importlib.util
import json
import os
import pathlib

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

HERE = pathlib.Path(__file__).parent
HELDOUT = HERE / "R11-H117_heldout_pairs.parquet"
MAX_LEN = 512


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@torch.inference_mode()
def probs(trunk, head, tok, texts, ctxs):
    out = np.zeros(len(texts), dtype=np.float32)
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], ctxs[i:i + 64], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to("cuda")
        cls = trunk(**enc).last_hidden_state[:, 0]
        out[i:i + 64] = torch.sigmoid(head(cls).float().squeeze(-1)).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="bare filename, written next to this script")
    args = ap.parse_args()
    mdir = pathlib.Path(args.model)

    H108 = _mod("h108", "R10-H108_lane.py")
    M59, M60 = H108.M59, H108.M60

    tok = AutoTokenizer.from_pretrained(str(mdir))
    trunk = AutoModel.from_pretrained(str(mdir / "trunk")).cuda().eval()
    st = torch.load(mdir / "dann_student.pt", map_location="cuda", weights_only=False)
    head = nn.Linear(trunk.config.hidden_size, 1).cuda().eval()
    head.load_state_dict(st["task_head"])

    # 1. ragtruth_en in-domain AUC (identical call to H108.evaluate)
    cl, ctx, y = M60.load_english()
    s = H108.score_student(trunk, head, tok, cl,
                           [M59.top_chunks(c, M59.CFG.semantic_top_k) for c in ctx])
    auc, f1, _ = M59.auc_and_f1(y, s)
    ragtruth = {"auc": round(float(auc), 4), "f1": round(float(f1), 4), "n": len(y)}

    # 2. held-out pair accuracy
    d = pl.read_parquet(HELDOUT)
    chunks = [c[: M59.CFG.chunk_max_chars] for c in d["chunk"].to_list()]
    p_seed = probs(trunk, head, tok, d["seed"].to_list(), chunks)
    p_corr = probs(trunk, head, tok, d["claim"].to_list(), chunks)
    win = p_seed > p_corr
    vb = d["verbatim"].to_numpy()

    def acc(mask):
        return {"pair_acc": round(float(win[mask].mean()), 4), "n": int(mask.sum()),
                "mean_gap": round(float((p_seed - p_corr)[mask].mean()), 4)}

    pair = {"overall": acc(np.ones(len(d), dtype=bool)),
            "verbatim": acc(vb), "non_verbatim": acc(~vb),
            "mean_p_seed": round(float(p_seed.mean()), 4),
            "mean_p_corrupt": round(float(p_corr.mean()), 4)}

    res = {"model": str(mdir), "ragtruth_en": ragtruth, "heldout_pairs": pair}
    (HERE / pathlib.Path(args.out).name).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("=== H117 PROBE READ DONE ===", flush=True)


if __name__ == "__main__":
    main()

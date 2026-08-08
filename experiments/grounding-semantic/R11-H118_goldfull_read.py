"""R11-H118 - gold_full read of a checkpoint dir (deterministic, 2,752 claims).

Usage (GPU0):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run python \
      experiments/grounding-semantic/R11-H118_goldfull_read.py \
      --model models/R11-H118-soup-h108 --out R11-H118_soup_h108_goldfull.json
"""

import argparse
import json
import os
import pathlib

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

HERE = pathlib.Path(__file__).parent
PAIRS = HERE / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"
MAX_LEN, CHUNK_MAX = 512, 4000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="bare filename, written next to this script")
    args = ap.parse_args()
    mdir = pathlib.Path(args.model)

    df = pl.read_parquet(PAIRS)
    claims, chunk_lists, labels = [], [], []
    for _owner, grp in df.group_by("owner"):
        claims.append(grp["claim"][0])
        chunk_lists.append([c[:CHUNK_MAX] for c in grp["chunk"].to_list()])
        labels.append(int(grp["label"][0]))
    y = np.array(labels)

    tok = AutoTokenizer.from_pretrained(str(mdir))
    trunk = AutoModel.from_pretrained(str(mdir / "trunk")).cuda().eval()
    st = torch.load(mdir / "dann_student.pt", map_location="cuda", weights_only=False)
    head = nn.Linear(trunk.config.hidden_size, 1).cuda().eval()
    head.load_state_dict(st["task_head"])

    flat_c, flat_k, owner = [], [], []
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        for k in ks:
            flat_c.append(c)
            flat_k.append(k)
            owner.append(i)
    out = np.zeros(len(flat_c), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, len(flat_c), 64):
            enc = tok(flat_c[i:i + 64], flat_k[i:i + 64], return_tensors="pt",
                      padding=True, truncation=True, max_length=MAX_LEN).to("cuda")
            cls = trunk(**enc).last_hidden_state[:, 0]
            out[i:i + 64] = torch.sigmoid(head(cls).float().squeeze(-1)).cpu().numpy()
    owner = np.array(owner)
    scores = np.array([out[owner == i].max() for i in range(len(claims))])
    auc = float(roc_auc_score(y, scores))
    res = {"model": str(mdir), "gold_full_auc": round(auc, 4), "n": len(y)}
    (HERE / pathlib.Path(args.out).name).write_text(json.dumps(res, indent=2))
    print(json.dumps(res))
    print("=== GOLD FULL READ DONE ===", flush=True)


if __name__ == "__main__":
    main()

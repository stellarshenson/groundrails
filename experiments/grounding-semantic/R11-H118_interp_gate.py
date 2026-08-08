"""R11-H118 kill-gate - linear interpolation barrier on gold_full.

Per the binding amendment: build W(alpha) = (1-alpha)*W1 + alpha*W2 over trunk
AND task_head for alpha in {0, 0.25, 0.5, 0.75, 1.0} on the H105 pair, read
gold_full (2,752 rows, deterministic) at each alpha.

  LICENSE (proceed to the blind read): gold_full(0.5) >= min(parents) - 0.01
  KILL (close the weight-space line):  gold_full(0.5) < 0.75

Intermediate alphas are diagnostic only. Also records the distance footnote
(r and displacement cosine) - evidence, adjudicates nothing.

Run (GPU0):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run python \
      experiments/grounding-semantic/R11-H118_interp_gate.py
"""

import json
import os
import pathlib

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

HERE = pathlib.Path(__file__).parent
A = HERE.parent.parent / "models" / "R9-H105-mmbert-dann-clean"  # draw 1
B = HERE.parent.parent / "models" / "R9-H105-draw2"
PAIRS = HERE / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"
PRETRAINED = "jhu-clsp/mmBERT-base"
MAX_LEN = 512
CHUNK_MAX = 4000
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
OUT = HERE / "R11-H118_interp_gate_result.json"


def gold_full():
    df = pl.read_parquet(PAIRS)
    claims, chunk_lists, labels = [], [], []
    for _owner, grp in df.group_by("owner"):
        claims.append(grp["claim"][0])
        chunk_lists.append([c[:CHUNK_MAX] for c in grp["chunk"].to_list()])
        labels.append(int(grp["label"][0]))
    return claims, chunk_lists, np.array(labels)


@torch.inference_mode()
def gold_auc(trunk, task_head, tok, claims, chunk_lists, y):
    flat_c, flat_k, owner = [], [], []
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        for k in ks:
            flat_c.append(c)
            flat_k.append(k)
            owner.append(i)
    out = np.zeros(len(flat_c), dtype=np.float32)
    for i in range(0, len(flat_c), 64):
        enc = tok(flat_c[i:i + 64], flat_k[i:i + 64], return_tensors="pt",
                  padding=True, truncation=True, max_length=MAX_LEN).to("cuda")
        cls = trunk(**enc).last_hidden_state[:, 0]
        out[i:i + 64] = torch.sigmoid(task_head(cls).float().squeeze(-1)).cpu().numpy()
    owner = np.array(owner)
    scores = np.array([out[owner == i].max() for i in range(len(claims))])
    return float(roc_auc_score(y, scores))


def main():
    ta = load_file(A / "trunk" / "model.safetensors")
    tb = load_file(B / "trunk" / "model.safetensors")
    ha = torch.load(A / "dann_student.pt", map_location="cpu", weights_only=False)
    hb = torch.load(B / "dann_student.pt", map_location="cpu", weights_only=False)

    # distance footnote: r = |W1-W2| / |W1-Wpre| and displacement cosine
    pre = AutoModel.from_pretrained(PRETRAINED)
    tp = {k: v for k, v in pre.state_dict().items()}
    del pre
    common = [k for k in ta if k in tp and ta[k].shape == tp[k].shape]
    d_ab = torch.sqrt(sum(((ta[k].float() - tb[k].float()) ** 2).sum() for k in common))
    d_ap = torch.sqrt(sum(((ta[k].float() - tp[k].float()) ** 2).sum() for k in common))
    dot = sum(((ta[k].float() - tp[k].float()) * (tb[k].float() - tp[k].float())).sum()
              for k in common)
    d_bp = torch.sqrt(sum(((tb[k].float() - tp[k].float()) ** 2).sum() for k in common))
    footnote = {
        "r": float(d_ab / d_ap),
        "cos_displacement": float(dot / (d_ap * d_bp)),
        "n_common_tensors": len(common),
    }
    del tp
    print(f"distance footnote: {footnote}", flush=True)

    tok = AutoTokenizer.from_pretrained(str(A))
    claims, chunk_lists, y = gold_full()
    print(f"gold_full: {len(y)} rows", flush=True)

    trunk = AutoModel.from_pretrained(str(A / "trunk")).cuda().eval()
    d = trunk.config.hidden_size
    head = nn.Linear(d, 1).cuda().eval()

    results = {}
    for al in ALPHAS:
        merged = {k: ((1 - al) * ta[k].float() + al * tb[k].float()).to(ta[k].dtype)
                  for k in ta}
        trunk.load_state_dict(merged, strict=True)
        hs = {k: ((1 - al) * ha["task_head"][k].float()
                  + al * hb["task_head"][k].float())
              for k in ha["task_head"]}
        head.load_state_dict(hs)
        auc = gold_auc(trunk, head, tok, claims, chunk_lists, y)
        results[str(al)] = round(auc, 4)
        print(f"  alpha={al:4.2f}  gold_full AUC {auc:.4f}", flush=True)

    mid = results["0.5"]
    parents = (results["0.0"], results["1.0"])
    license_bar = min(parents) - 0.01
    verdict = ("LICENSE" if mid >= license_bar
               else ("KILL" if mid < 0.75 else "UNLICENSED-MIDDLE (record, adjudicate manually)"))
    out = {"pair": "R9-H105 draws 1+2", "alphas": results,
           "min_parent": min(parents), "license_bar": round(license_bar, 4),
           "kill_bar": 0.75, "verdict": verdict, "footnote": footnote}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print("=== H118 INTERP GATE DONE ===", flush=True)


if __name__ == "__main__":
    main()

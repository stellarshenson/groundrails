"""R14-H131 Stage 1 (register block R14-A2) - frozen-weights windowed read at
`max_length=1024` instead of 512.

The read is R8-H101's windowed read with ONE change: the tokenizer's
`max_length` in `R8-H77_unseen_arena.score_student`. Everything else - the
1,500-char / stride-750 window geometry, the sentence splitter, the
max-over-windows, the min-over-sentences, the checkpoint loading, the batch
size, the AUROC estimator - is the shipped code, imported and called, not
re-implemented. `R8-H101_windowed_read.py` and `R8-H77_unseen_arena.py` are NOT
modified; `score_student` is replaced on the imported module object at runtime.

Stage 2 (training at 1024) is BLOCKED by session ruling 14 and is not run here.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      uv run python experiments/grounding-semantic/R14_H131_read1024.py \
        --model models/R9-H105-mmbert-dann-clean --out R14_H131_h105d1_1024.json
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import argparse
import importlib.util
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent

MAX_LEN = 1024
BATCH = 64


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H101 = _mod("h101", "R8-H101_windowed_read.py")
ARENA = H101.ARENA


def score_student_1024(path, claims, chunk_lists):
    """Byte-for-byte `ARENA.score_student`'s dann_student branch with
    max_length=MAX_LEN and `reference_compile=False` on the trunk config."""
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    assert (pathlib.Path(path) / "dann_student.pt").exists(), path
    tok = AutoTokenizer.from_pretrained(str(path))
    state = torch.load(
        pathlib.Path(path) / "dann_student.pt", map_location="cpu", weights_only=False
    )
    trunk = AutoModel.from_pretrained(str(pathlib.Path(path) / "trunk")).cuda().eval()
    trunk.config.reference_compile = False
    task_head = nn.Linear(trunk.config.hidden_size, 1)
    task_head.load_state_dict(state["task_head"])
    task_head = task_head.cuda().eval()

    flat_c, flat_k, owner = [], [], []
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        for k in ks:
            flat_c.append(c)
            flat_k.append(k[: ARENA.M59.CFG.chunk_max_chars])
            owner.append(i)
    s = np.zeros(len(flat_c), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, len(flat_c), BATCH):
            enc = tok(
                flat_c[i : i + BATCH],
                flat_k[i : i + BATCH],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
            )
            enc = {k: v.cuda() for k, v in enc.items()}
            cls = trunk(**enc).last_hidden_state[:, 0]
            s[i : i + BATCH] = torch.sigmoid(task_head(cls).float().squeeze(-1)).cpu().numpy()
    owner = np.array(owner)
    agg = np.array([s[owner == i].max() for i in range(len(claims))])
    del trunk, task_head
    torch.cuda.empty_cache()
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="bare filename")
    args = ap.parse_args()
    if "/" in args.out:
        raise SystemExit(f"--out must be a bare filename, got {args.out!r}")
    if (HERE / args.out).exists():
        print(f"skip - {args.out} exists", flush=True)
        return

    import torch

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"R14-H131 read at max_length={MAX_LEN}: {args.model}", flush=True)

    ARENA.score_student = score_student_1024
    sys.argv = ["R8-H101_windowed_read.py", "--model", args.model, "--out", args.out]
    H101.main()


if __name__ == "__main__":
    main()

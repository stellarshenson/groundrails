"""R8-H102 - blind reads of the two-head checkpoint, token head and score head.

Pre-registered adjudication (docs/experiments/semantic-grounding-experiments.md,
R8-H102): the paired within-checkpoint comparison - token-head-only PRIMARY
windowed read vs score-head PRIMARY windowed read of the SAME weights. This tool
prints per-subset AUC tables for both heads under both reads (windowed primary,
truncated lineage) with NO baked-in verdict logic; adjudication happens
externally against the registration.

Token-head per-pair score = 1 - max(halluc-token prob over claim tokens),
batched at MAX_LEN 512 identically to the frozen gate's scorer. The score-head
path reuses ARENA.score_student unchanged for exactness. Windowing follows
R8-H101: 1,500-char windows at stride 750 (final window flush to the chunk
end), max over windows over chunks, MIN over sentences.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      uv run python experiments/grounding-semantic/R8-H102_tokenread.py
"""

import argparse
import importlib.util
import json
import pathlib

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

HERE = pathlib.Path(__file__).parent


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ARENA = _mod("arena", "R8-H77_unseen_arena.py")
H92 = _mod("h92", "R8-H92_decomposed_arena.py")

MODEL = str(HERE.parent.parent / "models" / "R8-H102-mmbert-twohead-full")

WIN = 1500
STRIDE = 750
MAX_LEN = 512


def windows(chunk):
    """Sliding 1,500-char windows at stride 750; final window flush to the end."""
    n = len(chunk)
    if n <= WIN:
        return [chunk]
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return [chunk[s : s + WIN] for s in starts]


_TOKEN_CACHE = {}


def _load_token_scorer(path):
    if path not in _TOKEN_CACHE:
        state = torch.load(
            pathlib.Path(path) / "dann_student.pt", map_location="cpu", weights_only=False
        )
        trunk = AutoModel.from_pretrained(str(pathlib.Path(path) / "trunk")).cuda().eval()
        token_head = nn.Linear(trunk.config.hidden_size, 2)
        token_head.load_state_dict(state["token_head"])
        token_head = token_head.cuda().eval()
        tok = AutoTokenizer.from_pretrained(str(path))
        _TOKEN_CACHE[path] = (trunk, token_head, tok)
    return _TOKEN_CACHE[path]


@torch.inference_mode()
def score_token(path, claims, chunk_lists):
    """Per-pair 1 - max(halluc prob over claim tokens), max-over-chunks per claim.

    Mirrors ARENA.score_student's flattening, chunk cut, batch 64, MAX_LEN 512.
    """
    trunk, token_head, tok = _load_token_scorer(path)
    flat_c, flat_k, owner = [], [], []
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        for k in ks:
            flat_c.append(c)
            flat_k.append(k[: ARENA.M59.CFG.chunk_max_chars])
            owner.append(i)
    s = np.zeros(len(flat_c), dtype=np.float32)
    for i in range(0, len(flat_c), 64):
        enc = tok(
            flat_c[i : i + 64],
            flat_k[i : i + 64],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        seq0 = torch.tensor(
            [
                [enc.sequence_ids(r)[j] == 0 for j in range(enc["input_ids"].shape[1])]
                for r in range(enc["input_ids"].shape[0])
            ]
        )
        enc = {k: v.cuda() for k, v in enc.items()}
        h = trunk(**enc).last_hidden_state
        halluc = torch.softmax(token_head(h).float(), dim=-1)[:, :, 1].cpu()
        halluc[~seq0] = 0.0  # only claim tokens vote
        s[i : i + 64] = (1.0 - halluc.max(dim=1).values).numpy()
    owner = np.array(owner)
    return np.array([s[owner == i].max() for i in range(len(claims))])


def score_subset(scorer, model, claims, chunks, y, windowed):
    sent_lists = [H92.sentences(c) for c in claims]
    flat_s, flat_w, owner = [], [], []
    for i, (sl, ks) in enumerate(zip(sent_lists, chunks, strict=True)):
        wlist = [w for k in ks for w in windows(k)] if windowed else [k[:WIN] for k in ks]
        for s in sl:
            flat_s.append(s)
            flat_w.append(wlist)
            owner.append(i)
    owner = np.array(owner)
    scores = scorer(model, flat_s, flat_w)
    resp = np.array([scores[owner == i].min() for i in range(len(y))])
    auc, f1, _ = ARENA.M59.auc_and_f1(y, resp)
    return auc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL, help="two-head checkpoint dir")
    ap.add_argument("--head", choices=["token", "score", "both"], default="both")
    ap.add_argument("--windowed", action="store_true", help="windowed read only")
    ap.add_argument("--truncated", action="store_true", help="truncated read only")
    ap.add_argument("--out", default="R8-H102_reads.json", help="result json (in this dir)")
    args = ap.parse_args()

    heads = ["token", "score"] if args.head == "both" else [args.head]
    if args.windowed and not args.truncated:
        reads = [("windowed", True)]
    elif args.truncated and not args.windowed:
        reads = [("truncated", False)]
    else:
        reads = [("windowed", True), ("truncated", False)]

    scorers = {"token": score_token, "score": ARENA.score_student}
    subs = ARENA.load_subsets()

    result = {"model": args.model, "window": WIN, "stride": STRIDE, "reads": {}}
    for read_name, windowed in reads:
        for head in heads:
            key = f"{head}_{read_name}"
            print(f"\n--- {key} ---", flush=True)
            rows = {}
            for sub, (claims, chunks, y) in subs.items():
                auc, f1 = score_subset(scorers[head], args.model, claims, chunks, y, windowed)
                rows[sub] = {"n": len(y), "auc": round(auc, 4), "f1": round(f1, 4),
                             "lettuce_auc": H92.LETTUCE[sub]}
                print(f"  {sub:14s} n={len(y):>4} auc {auc:.4f}  lettuce {H92.LETTUCE[sub]:.4f}",
                      flush=True)
            mean = float(np.mean([r["auc"] for r in rows.values()]))
            result["reads"][key] = {"per_subset": rows, "mean": round(mean, 4)}
            print(f"  {'MEAN':14s} {mean:.4f}", flush=True)

    print("\n" + "=" * 92)
    print("R8-H102 READS - per-head means (adjudication is external, see registration)")
    print("=" * 92)
    for key, r in result["reads"].items():
        print(f"  {key:20s} mean {r['mean']:.4f}")

    out = HERE / args.out
    out.write_text(json.dumps(result, indent=2))
    print(f"\n  results -> {out}")


if __name__ == "__main__":
    main()

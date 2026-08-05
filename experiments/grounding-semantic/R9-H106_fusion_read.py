"""R9-H106_fusion_read - post-aggregation dual-head fusion read, any two-head checkpoint.

Parameterized successor of R9_PA_response_fusion.py (which was hardwired to the
frozen H102 checkpoint and sanity-gated against its recorded reads - both
H102-specific). Serves the R9-H106 registration: each head aggregated
SEPARATELY through the PRIMARY windowed decomposed-min, fused only at the
RESPONSE level, parameter-free:

  S_sent = min over sentences of max over windows of sigmoid(score-head)
  S_tok  = min over sentences of max over windows of (1 - max halluc-token prob)
  fused  = sigmoid((logit(S_sent) + logit(S_tok)) / 2)        (clamped 1e-6)

One trunk forward per (sentence, window) pair serves both heads. Empty-claim-
token guard per H104: the pair's token component falls back to its score
component. Internal consistency gate (replaces the H102 EXACT-reproduction
gate): all three reads come from the SAME forward passes in one run, and the
fused response probability must lie between its two components on every
response - a hard invariant of the logit-mean; any violation aborts the read.

No verdict logic here - adjudication is external against the R9-H106
registration (bar: fused - score >= +0.003 paired; kill: fused <= score).

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R9-H106_fusion_read.py \
        --model models/R9-H106-twohead-clean --out R9-H106_fusion_result.json
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

WIN = 1500
STRIDE = 750
MAX_LEN = 512
EPS = 1e-6


def windows(chunk):
    """Sliding 1,500-char windows at stride 750; final window flush to the end."""
    n = len(chunk)
    if n <= WIN:
        return [chunk]
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return [chunk[s : s + WIN] for s in starts]


_CACHE = {}


def _load(path):
    if path not in _CACHE:
        state = torch.load(
            pathlib.Path(path) / "dann_student.pt", map_location="cpu", weights_only=False
        )
        trunk = AutoModel.from_pretrained(str(pathlib.Path(path) / "trunk")).cuda().eval()
        task_head = nn.Linear(trunk.config.hidden_size, 1)
        task_head.load_state_dict(state["task_head"])
        task_head = task_head.cuda().eval()
        token_head = nn.Linear(trunk.config.hidden_size, 2)
        token_head.load_state_dict(state["token_head"])
        token_head = token_head.cuda().eval()
        tok = AutoTokenizer.from_pretrained(str(path))
        _CACHE[path] = (trunk, task_head, token_head, tok)
    return _CACHE[path]


@torch.inference_mode()
def score_both(path, claims, chunk_lists):
    """Per-claim (max-over-chunks) scores for BOTH heads from one trunk forward.

    Mirrors R8-H104's flattening, chunk cut, batch 64, MAX_LEN 512 - but keeps
    the two components separate; each head takes its OWN max over chunks.
    """
    trunk, task_head, token_head, tok = _load(path)
    flat_c, flat_k, owner = [], [], []
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        for k in ks:
            flat_c.append(c)
            flat_k.append(k[: ARENA.M59.CFG.chunk_max_chars])
            owner.append(i)
    s_score = np.zeros(len(flat_c), dtype=np.float32)
    s_token = np.zeros(len(flat_c), dtype=np.float32)
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
        score = torch.sigmoid(task_head(h[:, 0]).float().squeeze(-1)).cpu()
        halluc = torch.softmax(token_head(h).float(), dim=-1)[:, :, 1].cpu()
        halluc[~seq0] = 0.0  # only claim tokens vote
        token = 1.0 - halluc.max(dim=1).values
        token = torch.where(seq0.any(dim=1), token, score)  # empty-claim guard
        s_score[i : i + 64] = score.numpy()
        s_token[i : i + 64] = token.numpy()
    owner = np.array(owner)
    per_claim_score = np.array([s_score[owner == i].max() for i in range(len(claims))])
    per_claim_token = np.array([s_token[owner == i].max() for i in range(len(claims))])
    return per_claim_score, per_claim_token


def read_subset(model_path, claims, chunks, y):
    sent_lists = [H92.sentences(c) for c in claims]
    flat_s, flat_w, owner = [], [], []
    for i, (sl, ks) in enumerate(zip(sent_lists, chunks, strict=True)):
        wlist = [w for k in ks for w in windows(k)]
        for s in sl:
            flat_s.append(s)
            flat_w.append(wlist)
            owner.append(i)
    owner = np.array(owner)
    sc, tk = score_both(model_path, flat_s, flat_w)
    s_sent = np.array([sc[owner == i].min() for i in range(len(y))])
    s_tok = np.array([tk[owner == i].min() for i in range(len(y))])
    a = np.clip(s_sent, EPS, 1 - EPS)
    b = np.clip(s_tok, EPS, 1 - EPS)
    fused = 1.0 / (1.0 + np.exp(-((np.log(a / (1 - a)) + np.log(b / (1 - b))) / 2.0)))
    # Internal consistency: the logit-mean must lie between its two components
    # on every response (up to the clamp epsilon).
    lo = np.minimum(a, b) - 1e-6
    hi = np.maximum(a, b) + 1e-6
    if not ((fused >= lo) & (fused <= hi)).all():
        raise SystemExit("CONSISTENCY VIOLATION: fused outside its component envelope")
    out = {}
    for name, resp in [("score", s_sent), ("token", s_tok), ("fused", fused)]:
        auc, _, _ = ARENA.M59.auc_and_f1(y, resp)
        out[name] = round(auc, 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/R9-H106-twohead-clean")
    ap.add_argument("--out", default="R9-H106_fusion_result.json",
                    help="bare filename; written into this script's directory")
    args = ap.parse_args()
    model_path = str((HERE.parent.parent / args.model) if not args.model.startswith("/") else pathlib.Path(args.model))
    out_path = HERE / pathlib.Path(args.out).name  # bare filename, always this dir

    subs = ARENA.load_subsets()
    rows = {}
    for sub, (claims, chunks, y) in subs.items():
        rows[sub] = read_subset(model_path, claims, chunks, y)
        print(
            f"  {sub:14s} n={len(y):>4} score {rows[sub]['score']:.4f}  "
            f"token {rows[sub]['token']:.4f}  fused {rows[sub]['fused']:.4f}",
            flush=True,
        )

    print("\nsanity: fused-within-envelope invariant held on all subsets (same-pass reads)", flush=True)

    means = {k: round(float(np.mean([rows[s][k] for s in rows])), 4) for k in ("score", "token", "fused")}
    print("\n" + "=" * 92)
    print("R9-H106 FUSION READ - post-aggregation dual-head fusion (adjudication external)")
    print("=" * 92)
    print(f"  score {means['score']:.4f}   token {means['token']:.4f}   fused {means['fused']:.4f}")
    print("  registration inputs: bar fused - score >= +0.003 paired; kill fused <= score")

    out_path.write_text(json.dumps({
        "per_subset": rows, "means": means,
        "model": model_path, "window": WIN, "stride": STRIDE,
    }, indent=2))
    print(f"\n  results -> {out_path}")
    print("=== R9-H106 READS DONE ===")


if __name__ == "__main__":
    main()

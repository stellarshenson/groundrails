"""R8-H104 - parameter-free head fusion on the frozen H102 weights (pre-registered).

Because the two heads' subset profiles on one trunk are anti-correlated with
large complementary margins (R8-H102 paired table: token rescues finqa and
delucionqa where score is weakest; score rescues hotpotqa/emanual/pubmedqa
where token is weakest), fusing them at the PAIR level should beat either head
alone. Fusion is H73's exact serving formula, no parameter introduced or tuned:

  p = (sigmoid(score_logit) + 1 - max halluc-token prob over claim tokens) / 2

computed in ONE trunk forward per pair, then aggregated through the PRIMARY
windowed decomposed-min (H101 windows: 1,500 chars stride 750, final window
flush to the end; max over windows per sentence, MIN over sentences) plus the
truncated read for lineage. Empty-claim-token guard: if a pair has zero claim
tokens the token component falls back to the score component (the trainer's
fused in-domain eval hit NaN on non-EN precisely here).

Sanity check (printed before the full run): the score-only path of this
pipeline must reproduce the recorded score_windowed delucionqa AUC from
R8-H102_reads.json exactly, proving aggregation fidelity.

No verdict logic here - adjudication happens externally against the
registration (bar: fused windowed mean >= 0.7272; kill: < 0.7172).

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/R8-H104_fused_read.py
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
def score_fused(path, claims, chunk_lists, score_only=False):
    """Per-pair fused score in one trunk forward, max-over-chunks per claim.

    Mirrors ARENA.score_student's flattening, chunk cut, batch 64, MAX_LEN 512.
    score_only=True ignores the token component (sanity path).
    """
    trunk, task_head, token_head, tok = _load(path)
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
        score = torch.sigmoid(task_head(h[:, 0]).float().squeeze(-1)).cpu()
        if score_only:
            s[i : i + 64] = score.numpy()
            continue
        halluc = torch.softmax(token_head(h).float(), dim=-1)[:, :, 1].cpu()
        halluc[~seq0] = 0.0  # only claim tokens vote
        token = 1.0 - halluc.max(dim=1).values
        token = torch.where(seq0.any(dim=1), token, score)  # empty-claim guard
        s[i : i + 64] = ((score + token) / 2.0).numpy()
    owner = np.array(owner)
    return np.array([s[owner == i].max() for i in range(len(claims))])


def score_subset(claims, chunks, y, windowed, score_only=False):
    sent_lists = [H92.sentences(c) for c in claims]
    flat_s, flat_w, owner = [], [], []
    for i, (sl, ks) in enumerate(zip(sent_lists, chunks, strict=True)):
        wlist = [w for k in ks for w in windows(k)] if windowed else [k[:WIN] for k in ks]
        for s in sl:
            flat_s.append(s)
            flat_w.append(wlist)
            owner.append(i)
    owner = np.array(owner)
    scores = score_fused(MODEL, flat_s, flat_w, score_only=score_only)
    resp = np.array([scores[owner == i].min() for i in range(len(y))])
    auc, f1, _ = ARENA.M59.auc_and_f1(y, resp)
    return auc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="R8-H104_result.json", help="result json (in this dir)")
    args = ap.parse_args()

    subs = ARENA.load_subsets()

    # Sanity: score-only path must reproduce the recorded score_windowed delucionqa AUC.
    recorded = json.loads((HERE / "R8-H102_reads.json").read_text())
    ref = recorded["reads"]["score_windowed"]["per_subset"]["delucionqa"]["auc"]
    claims, chunks, y = subs["delucionqa"]
    auc, _ = score_subset(claims, chunks, y, windowed=True, score_only=True)
    match = round(auc, 4) == ref
    print(f"sanity: score-only windowed delucionqa {auc:.4f} vs recorded {ref:.4f} "
          f"-> {'MATCH' if match else 'MISMATCH'}", flush=True)
    if not match:
        raise SystemExit("sanity check failed - aggregation path is not faithful")

    result = {"model": MODEL, "window": WIN, "stride": STRIDE, "reads": {}}
    for read_name, windowed in [("fused_windowed", True), ("fused_truncated", False)]:
        print(f"\n--- {read_name} ---", flush=True)
        rows = {}
        for sub, (claims, chunks, y) in subs.items():
            auc, f1 = score_subset(claims, chunks, y, windowed)
            rows[sub] = {"n": len(y), "auc": round(auc, 4), "f1": round(f1, 4),
                         "lettuce_auc": H92.LETTUCE[sub]}
            print(f"  {sub:14s} n={len(y):>4} auc {auc:.4f}  lettuce {H92.LETTUCE[sub]:.4f}",
                  flush=True)
        mean = float(np.mean([r["auc"] for r in rows.values()]))
        result["reads"][read_name] = {"per_subset": rows, "mean": round(mean, 4)}
        print(f"  {'MEAN':14s} {mean:.4f}", flush=True)

    print("\n" + "=" * 92)
    print("R8-H104 READS - fused head, both formulas (adjudication is external)")
    print("=" * 92)
    for key, r in result["reads"].items():
        print(f"  {key:20s} mean {r['mean']:.4f}")

    out = HERE / args.out
    out.write_text(json.dumps(result, indent=2))
    print(f"\n  results -> {out}")


if __name__ == "__main__":
    main()

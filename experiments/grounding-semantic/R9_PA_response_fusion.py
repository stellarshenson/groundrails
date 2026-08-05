"""R9_PA_response_fusion - post-aggregation dual-head fusion, frozen H102 weights.

Precursor P-A (round 9, analysis-only). MECHANISM test on a protocol-
disqualified checkpoint (H102 trained on the pre-reset mix): valid because the
head-complementarity mechanism is orthogonal to the private-data protocol; any
deliverable this motivates retrains on the clean mix.

R8-H104 killed PAIR-level fusion (before the min). This is the one fusion shape
it did not test: each head aggregated SEPARATELY through the PRIMARY windowed
decomposed-min, fused only at the RESPONSE level, parameter-free:

  S_sent = min over sentences of max over windows of sigmoid(score-head)
  S_tok  = min over sentences of max over windows of (1 - max halluc-token prob)
  fused  = sigmoid((logit(S_sent) + logit(S_tok)) / 2)        (clamped 1e-6)

One trunk forward per (sentence, window) pair serves both heads. Empty-claim-
token guard per H104: the pair's token component falls back to its score
component. Sanity gate before the fused read counts: the S_sent and S_tok
per-subset AUCs must reproduce R8-H102_reads.json score_windowed AND
token_windowed EXACTLY at 4 decimals.

No verdict logic here - adjudication is external against the registration
(bar: fused mean >= 0.7172 AND fused > both single-head reads; kill: < 0.7172).

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/R9_PA_response_fusion.py
"""

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
OUT = HERE / "R9_PA_result.json"

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


def read_subset(claims, chunks, y):
    sent_lists = [H92.sentences(c) for c in claims]
    flat_s, flat_w, owner = [], [], []
    for i, (sl, ks) in enumerate(zip(sent_lists, chunks, strict=True)):
        wlist = [w for k in ks for w in windows(k)]
        for s in sl:
            flat_s.append(s)
            flat_w.append(wlist)
            owner.append(i)
    owner = np.array(owner)
    sc, tk = score_both(MODEL, flat_s, flat_w)
    s_sent = np.array([sc[owner == i].min() for i in range(len(y))])
    s_tok = np.array([tk[owner == i].min() for i in range(len(y))])
    a = np.clip(s_sent, EPS, 1 - EPS)
    b = np.clip(s_tok, EPS, 1 - EPS)
    fused = 1.0 / (1.0 + np.exp(-((np.log(a / (1 - a)) + np.log(b / (1 - b))) / 2.0)))
    out = {}
    for name, resp in [("score", s_sent), ("token", s_tok), ("fused", fused)]:
        auc, _, _ = ARENA.M59.auc_and_f1(y, resp)
        out[name] = round(auc, 4)
    return out


def main():
    subs = ARENA.load_subsets()
    recorded = json.loads((HERE / "R8-H102_reads.json").read_text())["reads"]
    ref_score = {s: v["auc"] for s, v in recorded["score_windowed"]["per_subset"].items()}
    ref_token = {s: v["auc"] for s, v in recorded["token_windowed"]["per_subset"].items()}

    rows = {}
    for sub, (claims, chunks, y) in subs.items():
        rows[sub] = read_subset(claims, chunks, y)
        print(
            f"  {sub:14s} n={len(y):>4} score {rows[sub]['score']:.4f} (rec {ref_score[sub]:.4f})  "
            f"token {rows[sub]['token']:.4f} (rec {ref_token[sub]:.4f})  fused {rows[sub]['fused']:.4f}",
            flush=True,
        )

    # Sanity gate over ALL subsets, both heads - fused read counts only on MATCH.
    mism = [
        s for s in rows
        if rows[s]["score"] != ref_score[s] or rows[s]["token"] != ref_token[s]
    ]
    if mism:
        print(f"\nSANITY MISMATCH on {mism} - fused read does NOT count")
        raise SystemExit(1)
    print("\nsanity: both single-head windowed reads reproduce R8-H102_reads.json EXACTLY", flush=True)

    means = {k: round(float(np.mean([rows[s][k] for s in rows])), 4) for k in ("score", "token", "fused")}
    print("\n" + "=" * 92)
    print("R9 P-A RESULT - post-aggregation dual-head fusion, frozen H102 (adjudication external)")
    print("=" * 92)
    print(f"  score {means['score']:.4f}   token {means['token']:.4f}   fused {means['fused']:.4f}")
    print(f"  registration inputs: bar fused >= 0.7172 AND fused > both heads; kill fused < 0.7172")

    OUT.write_text(json.dumps({
        "per_subset": rows, "means": means,
        "verdict_inputs": {"bar_mean": 0.7172, "score_mean_recorded": 0.7172,
                            "token_mean_recorded": 0.7051},
        "model": MODEL, "window": WIN, "stride": STRIDE,
    }, indent=2))
    print(f"\n  results -> {OUT}")
    print("=== R9_PA FUSION DONE ===")


if __name__ == "__main__":
    main()

"""R13-H129 - ensemble-output-distillation pre-build kill-gate (stage 1).

Registered in docs/experiments/semantic-grounding-experiments.md (R13-H129):
score both frozen H105 draws on a fixed-seed 20k sample of the public training
mix and measure their output disagreement - the transmissible distillation
signal. KILL if median |p1 - p2| < 0.02 AND frac(|p1 - p2| >= 0.10) < 5%.

The mix has no materialized parquet: the trainer builds it in memory via
`public_train()` (R9-H105_clean_mix.py, byte-identical in R10-H108_lane.py).
The row key is therefore the positional index into that deterministic build
order, carried alongside a blake2b hash of (claim, chunk) so any consumer can
assert alignment.

Run (GPU0):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/R13-H129_gate.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

HERE = pathlib.Path(__file__).parent
MODELS = HERE.parent.parent / "models"
DRAW1 = MODELS / "R9-H105-mmbert-dann-clean"
DRAW2 = MODELS / "R9-H105-draw2"
OUT = HERE / "R13-H129_gate_result.json"
SAMPLE_OUT = HERE / "R13-H129_gate_sample.parquet"

MAX_LEN = 512
BATCH = 64
SEED = 0
N_SAMPLE = 20_000


def build_mix():
    """The clean public-only mix in trainer order (claims, chunks, y, tags)."""
    spec = importlib.util.spec_from_file_location("h105", HERE / "R9-H105_clean_mix.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.public_train()


def row_hash(claim, chunk):
    h = hashlib.blake2b(digest_size=8)
    h.update(claim.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(chunk.encode("utf-8", "replace"))
    return h.hexdigest()


def load_head(mdir, trunk_hidden):
    st = torch.load(mdir / "dann_student.pt", map_location="cuda", weights_only=False)
    head = nn.Linear(trunk_hidden, 1).cuda().eval()
    head.load_state_dict(st["task_head"])
    return head


def score_pairs(claims, chunks, tok, trunks, heads, tag=""):
    """Sigmoid of the linear head on CLS, one output array per (trunk, head)."""
    outs = [np.zeros(len(claims), dtype=np.float32) for _ in trunks]
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(claims), BATCH):
            enc = tok(
                claims[i : i + BATCH],
                chunks[i : i + BATCH],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
            ).to("cuda")
            for j, (trunk, head) in enumerate(zip(trunks, heads, strict=True)):
                cls = trunk(**enc).last_hidden_state[:, 0]
                outs[j][i : i + BATCH] = (
                    torch.sigmoid(head(cls).float().squeeze(-1)).cpu().numpy()
                )
            if i % (BATCH * 50) == 0:
                done = i + len(enc["input_ids"])
                rate = done / max(time.time() - t0, 1e-9)
                print(
                    f"[{tag}] {done}/{len(claims)} rows  {rate:.1f} rows/s",
                    flush=True,
                )
    return outs


def dist(d):
    return {
        "n": int(len(d)),
        "median": round(float(np.median(d)), 6),
        "mean": round(float(np.mean(d)), 6),
        "p90": round(float(np.percentile(d, 90)), 6),
        "p99": round(float(np.percentile(d, 99)), 6),
        "frac_ge_0.10": round(float((d >= 0.10).mean()), 6),
        "frac_ge_0.02": round(float((d >= 0.02).mean()), 6),
    }


def main():
    print("building mix...", flush=True)
    claims, chunks, y, tags = build_mix()
    n = len(claims)
    print(f"mix rows: {n}", flush=True)

    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(n, size=N_SAMPLE, replace=False))
    s_claims = [claims[i] for i in idx]
    s_chunks = [chunks[i] for i in idx]
    s_y = y[idx]
    s_tags = [tags[i] for i in idx]

    tok = AutoTokenizer.from_pretrained(str(DRAW1))
    trunks, heads = [], []
    for mdir in (DRAW1, DRAW2):
        trunk = AutoModel.from_pretrained(str(mdir / "trunk")).cuda().eval()
        trunks.append(trunk)
        heads.append(load_head(mdir, trunk.config.hidden_size))

    p1, p2 = score_pairs(s_claims, s_chunks, tok, trunks, heads, tag="gate")
    d = np.abs(p1 - p2)

    df = pl.DataFrame(
        {
            "row_id": idx.astype(np.int64),
            "key_hash": [row_hash(c, k) for c, k in zip(s_claims, s_chunks, strict=True)],
            "label": s_y.astype(np.float32),
            "tag": s_tags,
            "p_d1": p1,
            "p_d2": p2,
            "abs_diff": d,
        }
    )
    df.write_parquet(SAMPLE_OUT)

    overall = dist(d)
    kill = overall["median"] < 0.02 and overall["frac_ge_0.10"] < 0.05
    res = {
        "hypothesis": "R13-H129",
        "stage": "pre-build kill-gate",
        "mix_rows": n,
        "n_sample": N_SAMPLE,
        "seed": SEED,
        "draws": [str(DRAW1), str(DRAW2)],
        "verdict": "KILL" if kill else "LICENSE",
        "gate": "KILL if median < 0.02 AND frac(>=0.10) < 5%",
        "overall": overall,
        "by_label": {
            "label_1_grounded": dist(d[s_y == 1.0]),
            "label_0_hallucinated": dist(d[s_y == 0.0]),
        },
        "by_group": {
            t: dist(d[np.array(s_tags) == t]) for t in sorted(set(s_tags))
        },
        "mean_p_d1": round(float(p1.mean()), 6),
        "mean_p_d2": round(float(p2.mean()), 6),
        "corr_p1_p2": round(float(np.corrcoef(p1, p2)[0, 1]), 6),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("verdict", "overall")}, indent=2), flush=True)
    print("=== R13-H129 GATE DONE ===", flush=True)


if __name__ == "__main__":
    main()

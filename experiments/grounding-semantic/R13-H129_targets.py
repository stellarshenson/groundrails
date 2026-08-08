"""R13-H129 - teacher targets over the full public mix (stage 2, gate-licensed).

Both frozen H105 draws' sigmoid probabilities plus their mean over every row of
the clean public-only training mix (~686k), written as
`R13-H129_teacher_targets.parquet`. Chunked and resumable: each 20k-row part is
a parquet under `R13-H129_targets_parts/`; a restart re-uses complete parts and
continues from the first missing one.

Row key = positional index into the deterministic `public_train()` build order
(the trainer's own order), with a blake2b hash of (claim, chunk) so the trainer
can assert alignment.

Run (GPU0):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/R13-H129_targets.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import importlib.util  # noqa: E402

HERE = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location("h129gate", HERE / "R13-H129_gate.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

PARTS = HERE / "R13-H129_targets_parts"
OUT = HERE / "R13-H129_teacher_targets.parquet"
VERIFY = HERE / "R13-H129_targets_verify.json"
CHUNK = 20_000


def score_sorted(claims, chunks, tok, trunks, heads, tag):
    """Score a part with length-sorted batches - same math, far less padding."""
    order = np.argsort([len(c) + len(k) for c, k in zip(claims, chunks, strict=True)])
    p1, p2 = G.score_pairs(
        [claims[i] for i in order], [chunks[i] for i in order], tok, trunks, heads, tag=tag
    )
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return p1[inv], p2[inv]


def main():
    PARTS.mkdir(exist_ok=True)
    print("building mix...", flush=True)
    claims, chunks, y, _tags = G.build_mix()
    n = len(claims)
    print(f"mix rows: {n}", flush=True)

    tok = AutoTokenizer.from_pretrained(str(G.DRAW1))
    trunks, heads = None, None

    bounds = [(s, min(s + CHUNK, n)) for s in range(0, n, CHUNK)]
    t0 = time.time()
    for pi, (s, e) in enumerate(bounds):
        pf = PARTS / f"part_{pi:05d}.parquet"
        if pf.exists():
            try:
                if pl.read_parquet(pf, columns=["row_id"]).height == e - s:
                    print(f"[part {pi:03d}] cached", flush=True)
                    continue
            except Exception:
                pass
            pf.unlink()
        if trunks is None:
            from torch import nn  # noqa: F401
            from transformers import AutoModel

            trunks, heads = [], []
            for mdir in (G.DRAW1, G.DRAW2):
                trunk = AutoModel.from_pretrained(str(mdir / "trunk")).cuda().eval()
                trunks.append(trunk)
                heads.append(G.load_head(mdir, trunk.config.hidden_size))
        p1, p2 = score_sorted(
            claims[s:e], chunks[s:e], tok, trunks, heads, tag=f"part {pi:03d}"
        )
        pl.DataFrame(
            {
                "row_id": np.arange(s, e, dtype=np.int64),
                "key_hash": [
                    G.row_hash(c, k)
                    for c, k in zip(claims[s:e], chunks[s:e], strict=True)
                ],
                "label": y[s:e].astype(np.float32),
                "p_d1": p1,
                "p_d2": p2,
                "p_teacher": ((p1 + p2) / 2.0).astype(np.float32),
            }
        ).write_parquet(pf)
        el = time.time() - t0
        print(
            f"[part {pi:03d}] written {e - s} rows  elapsed {el / 60:.1f} min",
            flush=True,
        )

    df = pl.concat([pl.read_parquet(PARTS / f"part_{i:05d}.parquet") for i in range(len(bounds))])
    df.write_parquet(OUT)
    print(f"wrote {OUT.name}: {df.height} rows", flush=True)

    # Verification: exact row count, contiguous keys, and 5 rows rescored.
    assert df.height == n, f"row count {df.height} != mix {n}"
    assert df["row_id"].to_numpy().tolist() == list(range(n)), "row_id not contiguous"
    if trunks is None:
        from transformers import AutoModel

        trunks, heads = [], []
        for mdir in (G.DRAW1, G.DRAW2):
            trunk = AutoModel.from_pretrained(str(mdir / "trunk")).cuda().eval()
            trunks.append(trunk)
            heads.append(G.load_head(mdir, trunk.config.hidden_size))
    rng = np.random.default_rng(13)
    spot = np.sort(rng.choice(n, size=5, replace=False))
    q1, q2 = G.score_pairs(
        [claims[i] for i in spot], [chunks[i] for i in spot], tok, trunks, heads, tag="verify"
    )
    sub = df.filter(pl.col("row_id").is_in(spot.tolist())).sort("row_id")
    d1 = np.abs(sub["p_d1"].to_numpy() - q1)
    d2 = np.abs(sub["p_d2"].to_numpy() - q2)
    hashes_ok = sub["key_hash"].to_list() == [
        G.row_hash(claims[i], chunks[i]) for i in spot
    ]
    res = {
        "rows": int(df.height),
        "mix_rows": n,
        "row_count_match": bool(df.height == n),
        "spot_rows": spot.tolist(),
        "spot_max_abs_diff_d1": round(float(d1.max()), 8),
        "spot_max_abs_diff_d2": round(float(d2.max()), 8),
        "spot_hashes_match": hashes_ok,
        "spot_pass": bool(d1.max() < 1e-4 and d2.max() < 1e-4 and hashes_ok),
        "p_teacher_mean": round(float(df["p_teacher"].mean()), 6),
        "p_d1_mean": round(float(df["p_d1"].mean()), 6),
        "p_d2_mean": round(float(df["p_d2"].mean()), 6),
    }
    VERIFY.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    print("=== R13-H129 TARGETS DONE ===", flush=True)


if __name__ == "__main__":
    main()

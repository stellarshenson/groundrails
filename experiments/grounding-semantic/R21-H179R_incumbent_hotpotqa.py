"""R21-H179R INCUMBENT RE-READ ON THE REDUCED hotpotqa ITEM SET.

WHY THIS EXISTS
---------------
The contamination audit found 16 of the 250 arena `hotpotqa` responses retrieve
documents that are byte-for-byte substrings of OUR training data, and 102 that
are contained in it under the looser containment test. Removing those items
moves OUR arena mean. It also moves the INCUMBENT's, in an unknown direction,
and a margin is only honest when both systems are read on the SAME item set.

Only `hotpotqa` carries `exposed_verbatim` flags, so only `hotpotqa` needs
re-reading; the incumbent's other nine subsets are untouched and their banked
values stand.

CONVENTION
----------
The incumbent's best arena mean (0.67963) is `native_truncated`, produced by
`R19-H171_incumbent_native.py`. That module is imported and its `load_items`
and `score_native` are reused verbatim - nothing is re-derived here, so the
scoring path is the banked one by construction.

FIDELITY CONTROL (hard gate)
----------------------------
The recomputed all-250 AUROC must reproduce the banked 0.6161 to 1e-4 after
rounding to 4 decimals. A mismatch means the scoring path differs from the
banked one; that is the finding and the run ABORTS rather than switching
conventions or tuning anything.

ALIGNMENT
---------
The exposure flags live in `R21-H179_consensus_errors.parquet`, keyed
(subset, item) - `row_id` is the RAGBench SOURCE row and is NOT unique. The
item ordering used by `load_items` is asserted against
`R21-H179_arena_items.parquet` on response text and label before any flag is
applied.

Run: CUDA_VISIBLE_DEVICES=1 uv run python R21-H179R_incumbent_hotpotqa.py
"""

import importlib.util
import json
import os
import pathlib
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: E402

HERE = pathlib.Path(__file__).parent
SUB = "hotpotqa"
BANKED_HOTPOTQA = 0.6161          # R19-H171_incumbent_chunked.json native_truncated_auc
TOL = 1e-4
EXPECT_VERBATIM = 16
EXPECT_CONTAINMENT = 102

ITEMS = HERE / "R21-H179_arena_items.parquet"
FLAGS = HERE / "R21-H179_consensus_errors.parquet"
OUT_NPZ = HERE / "R21-H179R_incumbent_hotpotqa.npz"
OUT_JSON = HERE / "R21-H179R_incumbent_hotpotqa.json"


class Abort(Exception):
    """A gate failed. The failure is the finding; nothing gets adjusted."""


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H171 = _mod("h171", "R19-H171_incumbent_native.py")


def main():
    t0 = time.time()
    print(f"=== R21-H179R incumbent hotpotqa re-read  {time.strftime('%F %T')} ===",
          flush=True)
    print(f"  device {torch.cuda.get_device_name(0)}  "
          f"max_length {H171.MAX_LENGTH}  model {H171.ARENA.LETTUCE}", flush=True)

    # --- items, through the banked loader ------------------------------------------
    items = H171.load_items()
    if SUB not in items:
        raise Abort(f"{SUB} absent from the banked loader's sample")
    qs, ds, rs, y = items[SUB]
    n = len(rs)
    if n != 250:
        raise Abort(f"{SUB} loader returned {n} items, expected 250")

    # --- alignment: loader order vs the arena item table ----------------------------
    tbl = pl.read_parquet(ITEMS).filter(pl.col("subset") == SUB).sort("item")
    if tbl.height != n:
        raise Abort(f"item table has {tbl.height} {SUB} rows, loader has {n}")
    if list(tbl["item"]) != list(range(n)):
        raise Abort("item column is not 0..n-1 after sort")
    if list(tbl["response"]) != list(rs):
        raise Abort("ALIGNMENT: loader response order disagrees with the item table")
    if not np.array_equal(tbl["label"].cast(pl.Int8).to_numpy(), np.asarray(y)):
        raise Abort("ALIGNMENT: loader labels disagree with the item table")
    print(f"  alignment OK: {n} {SUB} items match the arena item table "
          f"on response text and label", flush=True)

    # --- flags, joined on (subset, item) -------------------------------------------
    fl = (pl.read_parquet(FLAGS)
          .filter(pl.col("subset") == SUB)
          .select("item", "row_id", "exposed_verbatim", "exposed_containment"))
    if fl.height != n:
        raise Abort(f"flag table has {fl.height} {SUB} rows, expected {n}")
    joined = tbl.select("item", "row_id", "label").join(fl, on="item", how="inner",
                                                        suffix="_fl")
    if joined.height != n:
        raise Abort(f"join changed row count: {joined.height} != {n}")
    if list(joined["row_id"]) != list(joined["row_id_fl"]):
        raise Abort("JOIN: row_id disagrees between item table and flag table")
    joined = joined.sort("item")
    verb = joined["exposed_verbatim"].to_numpy().astype(bool)
    cont = joined["exposed_containment"].to_numpy().astype(bool)
    if int(verb.sum()) != EXPECT_VERBATIM or int(cont.sum()) != EXPECT_CONTAINMENT:
        raise Abort(f"flag counts verbatim={int(verb.sum())} "
                    f"containment={int(cont.sum())}, expected "
                    f"{EXPECT_VERBATIM}/{EXPECT_CONTAINMENT}")
    print(f"  flags OK: verbatim {int(verb.sum())}  containment {int(cont.sum())}",
          flush=True)

    # --- score, banked path --------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(H171.ARENA.LETTUCE)
    model = AutoModelForTokenClassification.from_pretrained(
        H171.ARENA.LETTUCE, dtype=torch.float16).cuda().eval()
    print("  scoring ...", flush=True)
    scores, over = H171.score_native(tok, model, qs, ds, rs)
    y = np.asarray(y)

    np.savez(OUT_NPZ, score=scores.astype(np.float64), label=y.astype(np.int8),
             item=joined["item"].to_numpy().astype(np.int32),
             row_id=np.asarray([str(v) for v in joined["row_id"]]),
             exposed_verbatim=verb, exposed_containment=cont)
    print(f"  per-item scores -> {OUT_NPZ.name}  ({over}/{n} hit the length cap)",
          flush=True)

    def auc(mask):
        return float(roc_auc_score(y[mask], scores[mask]))

    keep_all = np.ones(n, dtype=bool)
    keep_v = ~verb
    keep_c = ~cont
    a_all, a_v, a_c = auc(keep_all), auc(keep_v), auc(keep_c)

    delta = abs(round(a_all, 4) - BANKED_HOTPOTQA)
    ok = delta <= TOL
    print(f"\n  FIDELITY {'PASS' if ok else 'ABORT'}  recomputed {a_all:.5f} "
          f"(rounded {round(a_all, 4):.4f})  banked {BANKED_HOTPOTQA:.4f}  "
          f"|delta| {delta:.6f}", flush=True)
    print(f"  all 250        AUROC {a_all:.5f}  (n={int(keep_all.sum())}, "
          f"pos={int(y[keep_all].sum())})", flush=True)
    print(f"  minus verbatim AUROC {a_v:.5f}  (n={int(keep_v.sum())}, "
          f"pos={int(y[keep_v].sum())})", flush=True)
    print(f"  minus contain. AUROC {a_c:.5f}  (n={int(keep_c.sum())}, "
          f"pos={int(y[keep_c].sum())})", flush=True)

    res = {
        "arm": "R21-H179R incumbent hotpotqa re-read on the reduced item set",
        "model": H171.ARENA.LETTUCE,
        "convention": "native_truncated (R19-H171_incumbent_native.score_native)",
        "max_length": H171.MAX_LENGTH,
        "subset": SUB,
        "fidelity_control": {
            "banked_hotpotqa": BANKED_HOTPOTQA,
            "recomputed_all250": round(a_all, 5),
            "recomputed_all250_rounded4": round(a_all, 4),
            "abs_delta": round(delta, 6),
            "tolerance": TOL,
            "verdict": "PASS" if ok else "ABORT",
        },
        "auroc": {
            "all_250": round(a_all, 5),
            "minus_exposed_verbatim": round(a_v, 5),
            "minus_exposed_containment": round(a_c, 5),
        },
        "counts": {
            "all": int(keep_all.sum()),
            "minus_verbatim": int(keep_v.sum()),
            "minus_containment": int(keep_c.sum()),
            "exposed_verbatim": int(verb.sum()),
            "exposed_containment": int(cont.sum()),
            "positives_all": int(y.sum()),
            "positives_minus_verbatim": int(y[keep_v].sum()),
            "positives_minus_containment": int(y[keep_c].sum()),
            "items_at_max_length": int(over),
        },
        "deltas_vs_all250": {
            "minus_verbatim": round(a_v - a_all, 5),
            "minus_containment": round(a_c - a_all, 5),
        },
        "artifacts": {
            "npz": str(OUT_NPZ),
            "json": str(OUT_JSON),
            "log": str(HERE.parent.parent / "logs" /
                       "R21-H179R_incumbent_hotpotqa.log"),
        },
        "flags_source": str(FLAGS),
        "items_source": str(ITEMS),
        "join_key": "(subset, item) - row_id is the RAGBench source row and is not unique",
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(res, indent=2))
    print(f"  -> {OUT_JSON.name}", flush=True)
    if not ok:
        raise Abort(f"fidelity control failed: {a_all:.5f} vs {BANKED_HOTPOTQA}")
    print("=== H179R COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()

"""R14-H133 anti-gaming read on the TRACE-CONDITIONED set (A4 clause, MOVED).

The banked anti-gaming set asks bare claims; the v3 lane trains a detector that
verifies a programmatically attached reasoning trace. This reads the same pairs
re-issued in the serving shape (`R14-H133_antigaming_traced.parquet`, built by
`R14-H133_antigaming_traced.py`), where `claim` is already the traced form -
trace + " " + claim_untraced - and `chunk` is the trimmed evidence.

Scoring conventions are inherited byte-for-byte from `R14-H133_antigaming.py`
via `R15_gate_common` (same tokenizer pairing, 512-token truncation, batch 64,
sigmoid head). Both checkpoints read the identical rows: the arm draw and the
BANKED CLEAN DRAW 1 (`models/R9-H105-mmbert-dann-clean`).

Reports; does not adjudicate. Frozen weights, zero arena, zero gold.

Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
     uv run python experiments/grounding-semantic/R14-H133_antigaming_traced_read.py --draw 1
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import importlib.util
import json
import pathlib

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
TRACED = HERE / "R14-H133_antigaming_traced.parquet"
SCORES = HERE / "R14-H133_antigaming_traced_scores.parquet"

CONTROL_CKPT = "R9-H105-mmbert-dann-clean"  # the BANKED clean draw 1
BINDROW_BAR = 0.95
# Kept aligned with the bare-claim read: the family nearest a lane operator.
# `comparative_flip` does not survive into the traced set (a value-lookup trace
# cannot contradict a word-only corruption), so `digit_perturb` is the only
# exclusion that bites here.
EXCLUDED_FAMILIES = ("digit_perturb", "comparative_flip")


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read_ckpt(C, name, claims, evs):
    import torch

    tok, trunk, head = C.load_ckpt(name)
    s = C.score(tok, trunk, head, claims, evs)
    del trunk, head
    torch.cuda.empty_cache()
    return s


def summarise(C, df, s):
    """Pair-aligned AUROC. Rows carry label 1/0 within a pair_id; the two
    kinds have independent pair_id spaces, so pairs are keyed (kind, pair_id)."""
    kind = df["kind"].to_numpy()
    fam = df["family"].to_numpy()
    lab = df["label"].to_numpy()
    out = {}

    def auroc_of(mask):
        p, n = s[mask & (lab == 1)], s[mask & (lab == 0)]
        if min(len(p), len(n)) < 30:
            return {"n_pos": int(len(p)), "n_neg": int(len(n)),
                    "note": "under 30 per side - not adjudicated"}
        return {
            "n_pos": int(len(p)), "n_neg": int(len(n)),
            "auroc_pos_vs_neg": round(C.auroc(p, n), 4),
            "mean_pos": round(float(p.mean()), 5),
            "mean_neg": round(float(n.mean()), 5),
        }

    nm = kind == "nearmiss"
    out["nearmiss_all_families"] = auroc_of(nm)
    out["nearmiss_headline"] = auroc_of(nm & ~np.isin(fam, EXCLUDED_FAMILIES))
    out["bind_row"] = auroc_of(kind == "bind_row")
    out["bind_row"]["target"] = BINDROW_BAR
    out["per_family"] = {}
    for f in sorted(set(fam.tolist())):
        e = auroc_of(fam == f)
        e["in_nearmiss_headline"] = bool(f not in EXCLUDED_FAMILIES
                                         and f in set(fam[nm].tolist()))
        out["per_family"][f] = e
    return out


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", type=int, required=True, choices=(1, 2))
    args = ap.parse_args()
    arm_ckpt = f"R14-H133-arm-draw{args.draw}"
    result = HERE / f"R14-H133_antigaming_traced_draw{args.draw}_result.json"

    C = _mod("c", "R15_gate_common.py")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    df = pl.read_parquet(TRACED)
    # `claim` is the traced concatenation already; verified against
    # trace + " " + claim_untraced before this script was written.
    bad = df.filter(pl.col("claim") != pl.col("trace") + " " + pl.col("claim_untraced")).height
    if bad:
        raise SystemExit(f"{bad} rows where claim != trace + ' ' + claim_untraced")
    claims = df["claim"].to_list()
    evs = df["chunk"].to_list()
    print(f"traced set: {df.height} rows "
          f"({df.filter(pl.col('kind') == 'nearmiss').height} near-miss, "
          f"{df.filter(pl.col('kind') == 'bind_row').height} bind_row)", flush=True)

    per_ckpt, banked = {}, {}
    for name in (arm_ckpt, CONTROL_CKPT):
        s = read_ckpt(C, name, claims, evs)
        banked[name] = s
        per_ckpt[name] = summarise(C, df, s)
        print(f"{name}: near-miss(all) "
              f"{per_ckpt[name]['nearmiss_all_families'].get('auroc_pos_vs_neg')}  "
              f"headline {per_ckpt[name]['nearmiss_headline'].get('auroc_pos_vs_neg')}  "
              f"bind_row {per_ckpt[name]['bind_row'].get('auroc_pos_vs_neg')}", flush=True)

    sc = df.select(["pair_id", "kind", "family", "label", "table_id"])
    for name, s in banked.items():
        sc = sc.with_columns(pl.Series(f"score__{name.replace('-', '_')}", s))
    sc.write_parquet(SCORES)

    def delta(path):
        a, c = per_ckpt[arm_ckpt], per_ckpt[CONTROL_CKPT]
        for k in path:
            a, c = a.get(k, {}), c.get(k, {})
        if "auroc_pos_vs_neg" not in a or "auroc_pos_vs_neg" not in c:
            return None
        return round(a["auroc_pos_vs_neg"] - c["auroc_pos_vs_neg"], 4)

    fams = sorted(per_ckpt[arm_ckpt]["per_family"])
    res = {
        "read": "R14-H133 ANTI-GAMING on the TRACE-CONDITIONED set - the MOVED A4 clause's "
                "own data (serving shape: trace-conditioned claims)",
        "draw": args.draw,
        "arm_checkpoint": arm_ckpt,
        "control_checkpoint": CONTROL_CKPT,
        "control_note": "the BANKED clean draw 1 - unseeded (pre-H126); the comparison is "
                        "arm-vs-banked-control, not init-paired",
        "data": TRACED.name,
        "n_rows": df.height,
        "construction_note": "pairs are the banked anti-gaming pairs re-issued with a "
                             "byte-identical within-pair trace that reports the evidence; "
                             "word-only corruptions (scale_word, pct_pp, comparative_flip) do "
                             "not survive into this set",
        "excluded_from_nearmiss_headline": list(EXCLUDED_FAMILIES),
        "checkpoints": per_ckpt,
        "deltas_arm_minus_control": {
            "nearmiss_all_families": delta(["nearmiss_all_families"]),
            "nearmiss_headline": delta(["nearmiss_headline"]),
            "bind_row": delta(["bind_row"]),
            "per_family": {f: delta(["per_family", f]) for f in fams},
        },
        "adjudication": "NOT ADJUDICATED HERE - measurement only; the coordinator holds the "
                        "verdict on the MOVED clause",
        "scores": SCORES.name,
    }
    result.write_text(json.dumps(res, indent=2))
    print("\n" + "=" * 88)
    for k in ("nearmiss_all_families", "nearmiss_headline", "bind_row"):
        a = per_ckpt[arm_ckpt][k].get("auroc_pos_vs_neg")
        c = per_ckpt[CONTROL_CKPT][k].get("auroc_pos_vs_neg")
        print(f"  {k:24s} arm {a}  control {c}  delta {res['deltas_arm_minus_control'][k]}")
    for f in fams:
        a = per_ckpt[arm_ckpt]["per_family"][f].get("auroc_pos_vs_neg")
        c = per_ckpt[CONTROL_CKPT]["per_family"][f].get("auroc_pos_vs_neg")
        print(f"    {f:22s} arm {a}  control {c}  "
              f"delta {res['deltas_arm_minus_control']['per_family'][f]}")
    print(f"\n  -> {result}\n  -> {SCORES}", flush=True)


if __name__ == "__main__":
    main()

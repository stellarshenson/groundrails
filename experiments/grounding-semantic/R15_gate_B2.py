"""R15-B2 killgate clause 2 (R15-H137) - is the absent-number shortcut EXPRESSED
on the rows the weight would up-weight?

H137 up-weights the 87,933 absent-positive training rows at w* = 1.8038 to drive
pooled P(label 0 | absent) from 0.6433 to 0.5000. That is only worth doing if the
deployed function actually scores those rows below present-number positives.

  Score a fixed-seed 4,000-row sample of absent-positives against a matched
  4,000-row sample of present-positives from the same groups.
  KILL if mean(present-positive) - mean(absent-positive) < 0.05.

Pool = the byte-exact 685,670-row clean mix (tmp/R14_E6_mix.parquet) plus the
admitted H108 lane (R10-H108_pairs.parquet) - the same pool the w* arithmetic in
R15_L2_weight_arith.json is computed over. Absence detector byte-identical to
tmp/R15_L2_weights.py / P3.

Frozen H105 draw 1, in-domain training rows, zero arena, zero gold.
Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 uv run python <this>
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util
import json
import pathlib

import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
RESULT = HERE / "R15_gate_B2_result.json"
SAMPLE = HERE / "R15_gate_B2_sample.parquet"
FLAGS = HERE / "R15_gate_B2_absent_flags.parquet"

CKPT = "R9-H105-mmbert-dann-clean"
SEED = 20260812
N_PER_ARM = 4000
BAR = 0.05


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pool(C):
    """The L2 pool: clean mix + admitted H108 lane, with the absence flag."""
    if FLAGS.exists():
        return pl.read_parquet(FLAGS)
    mix = pl.read_parquet(ROOT / "tmp" / "R14_E6_mix.parquet").select(
        ["claim", "chunk", "label", "tag"])
    h = pl.read_parquet(HERE / "R10-H108_pairs.parquet").select(
        ["claim", "chunk", "label"]).with_columns(pl.lit("h108_lane").alias("tag"))
    df = pl.concat([mix, h.select(mix.columns)])
    print(f"pool rows: {len(df)}", flush=True)
    cl, ch = df["claim"].to_list(), df["chunk"].to_list()
    has_num, absent = [], []
    for c, e in zip(cl, ch):
        s = C.canon_set(c)
        has_num.append(bool(s))
        absent.append(bool(s - C.canon_set(e)) if s else False)
    df = df.with_columns([pl.Series("has_num", has_num), pl.Series("absent", absent)])
    df.write_parquet(FLAGS)
    return df


def main():
    import torch

    C = _mod("c", "R15_gate_common.py")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    df = pool(C).with_row_index("idx")
    ap = df.filter((pl.col("absent")) & (pl.col("label") == 1))
    pp = df.filter((~pl.col("absent")) & (pl.col("has_num")) & (pl.col("label") == 1))
    print(f"absent-positive {len(ap)}   present-positive (numeric claim) {len(pp)}", flush=True)

    ap_s = ap.sample(n=min(N_PER_ARM, len(ap)), shuffle=True, seed=SEED)

    # matched = the SAME per-group counts as the absent-positive draw
    want = dict(ap_s.group_by("tag").len().iter_rows())
    parts, shortfall = [], {}
    for i, tag in enumerate(sorted(want)):
        k = want[tag]
        sub = pp.filter(pl.col("tag") == tag)
        take = min(k, len(sub))
        if take < k:
            shortfall[tag] = {"wanted": int(k), "available": int(len(sub))}
        if take:
            parts.append(sub.sample(n=take, shuffle=True, seed=SEED + i))
    pp_s = pl.concat(parts)

    tok, trunk, head = C.load_ckpt(CKPT)
    s_ap = C.score(tok, trunk, head, ap_s["claim"].to_list(), ap_s["chunk"].to_list())
    s_pp = C.score(tok, trunk, head, pp_s["claim"].to_list(), pp_s["chunk"].to_list())
    del trunk, head
    torch.cuda.empty_cache()

    ap_s = ap_s.with_columns([pl.Series("score", s_ap), pl.lit("absent_positive").alias("arm")])
    pp_s = pp_s.with_columns([pl.Series("score", s_pp), pl.lit("present_positive").alias("arm")])
    pl.concat([ap_s, pp_s]).drop(["claim", "chunk"]).write_parquet(SAMPLE)

    per_group = {}
    for tag in sorted(want):
        a = ap_s.filter(pl.col("tag") == tag)["score"].to_numpy()
        p = pp_s.filter(pl.col("tag") == tag)["score"].to_numpy()
        if len(a) < 20 or len(p) < 20:
            per_group[tag] = {"n_absent_pos": len(a), "n_present_pos": len(p),
                              "note": "under 20 in an arm - not adjudicated"}
            continue
        per_group[tag] = {
            "n_absent_pos": int(len(a)), "n_present_pos": int(len(p)),
            "mean_absent_pos": round(float(a.mean()), 5),
            "mean_present_pos": round(float(p.mean()), 5),
            "delta": round(float(p.mean() - a.mean()), 5),
            "auroc_present_over_absent": round(C.auroc(p, a), 4),
        }

    delta = float(s_pp.mean() - s_ap.mean())
    verdict = "KILL" if delta < BAR else "PASS"

    res = {
        "gate": "R15-B2 (R15-H137) killgate clause 2 - shortcut expression on the up-weighted rows",
        "model": str(C.MODELS / CKPT),
        "data": "byte-exact clean mix (tmp/R14_E6_mix.parquet, 685,670 rows) + admitted H108 lane "
                "(R10-H108_pairs.parquet, 61,184 rows); the pool R15_L2_weight_arith.json is "
                "computed over. Training rows, in-domain, zero arena, zero gold.",
        "implementation_choices": [
            "'matched from the same groups' is implemented as per-DANN-group count matching: the "
            "present-positive draw carries the SAME per-group counts as the absent-positive draw.",
            "present-positive is restricted to rows whose claim CARRIES a numeral (present-number "
            "positives), not to all non-absent positives - the contrast the gate names is "
            "present-number against absent-number, and a claim with no numeral is neither.",
            "score = the shipped sigmoid task score of the training row's own (claim, chunk) pair; "
            "no re-chunking, because the pool rows ARE the training object.",
        ],
        "seed": SEED, "n_per_arm_target": N_PER_ARM,
        "n_absent_positive_pool": int(len(ap)), "n_present_positive_pool": int(len(pp)),
        "n_absent_positive_scored": int(len(ap_s)), "n_present_positive_scored": int(len(pp_s)),
        "group_match_shortfall": shortfall,
        "mean_absent_positive": round(float(s_ap.mean()), 5),
        "mean_present_positive": round(float(s_pp.mean()), 5),
        "delta_present_minus_absent": round(delta, 5),
        "auroc_present_over_absent": round(C.auroc(s_pp, s_ap), 4),
        "per_group": per_group,
        "bar": f"KILL if mean(present-positive) - mean(absent-positive) < {BAR}",
        "verdict": verdict,
        "gates_downstream": "R15-H137's ~12 GPU-h arm; the per-group breakdown is also the arm's "
                            "own pre-arm baseline",
        "sample": SAMPLE.name,
    }
    RESULT.write_text(json.dumps(res, indent=2))
    print("\n" + "=" * 88)
    print(f"  absent-positive {s_ap.mean():.5f}   present-positive {s_pp.mean():.5f}   "
          f"delta {delta:+.5f} (need >= {BAR})")
    print(f"\n  VERDICT: {verdict}\n  -> {RESULT}", flush=True)


if __name__ == "__main__":
    main()

"""R16-H142-T amendment A1 - the anti-gaming hold re-priced to a noise band.

The registered A1 amendment re-prices the anti-gaming hold from the no-tolerance
form (arm headline >= the clean control's 0.7565) to

    bar = clean control - 2 x SE_delta

where SE_delta is the paired-bootstrap standard error of the arm-minus-control
headline AUROC delta, 10,000 resamples over the headline PAIRS.

The bootstrap is paired in two senses, both of which matter: a resample draws
PAIRS (each carrying its positive and its negative claim), and both checkpoints
are re-scored on the SAME resampled pairs, so the shared item difficulty that
dominates near-miss AUROC cancels in the delta exactly as it does in the read.

No GPU work: the anti-gaming stage banks every per-pair score for both
checkpoints in `R16-H142-T_antigaming_set.parquet` at scoring time, so this is a
pure resampling of banked numbers.

Run:  uv run python experiments/grounding-semantic/R16-H142_T_seband.py
"""

import json
import pathlib

import numpy as np
import polars as pl
from scipy.stats import rankdata

HERE = pathlib.Path(__file__).parent

EVALSET = HERE / "R16-H142-T_antigaming_set.parquet"
READ = HERE / "R16-H142-T_antigaming_draw1_result.json"
OUT = HERE / "R16-H142_T_seband_result.json"

ARM_TAG = "R16_H142_T_arm_draw1"          # the banked draw-1 twin
CTL_TAG = "R9_H105_mmbert_dann_clean"     # the clean-recipe control
EXCLUDED_FAMILIES = ("digit_perturb", "comparative_flip")  # R14-H133_antigaming
N_BOOT = 10_000
BOOT_SEED = 20260812


def auroc_rows(pos, neg):
    """AUROC of pos vs neg for each ROW of (B, n) score matrices - the
    Mann-Whitney form with average ranks, identical to
    R15_gate_common.auroc (sklearn roc_auc_score) on a single row."""
    both = np.concatenate([pos, neg], axis=1)
    r = rankdata(both, axis=1)
    n = pos.shape[1]
    r_pos = r[:, :n].sum(axis=1)
    return (r_pos - n * (n + 1) / 2.0) / (n * n)


def main():
    df = pl.read_parquet(EVALSET)
    head = df.filter(
        (pl.col("kind") == "nearmiss") & ~pl.col("family").is_in(list(EXCLUDED_FAMILIES))
    )
    n = head.height
    arm_p = head[f"pos__{ARM_TAG}"].to_numpy().astype(np.float64)
    arm_n = head[f"neg__{ARM_TAG}"].to_numpy().astype(np.float64)
    ctl_p = head[f"pos__{CTL_TAG}"].to_numpy().astype(np.float64)
    ctl_n = head[f"neg__{CTL_TAG}"].to_numpy().astype(np.float64)

    banked = json.loads(READ.read_text())["reads"]
    point_arm = float(auroc_rows(arm_p[None, :], arm_n[None, :])[0])
    point_ctl = float(auroc_rows(ctl_p[None, :], ctl_n[None, :])[0])
    print(f"headline pairs {n}  arm {point_arm:.6f} (banked {banked['nearmiss_headline_arm']})  "
          f"control {point_ctl:.6f} (banked {banked['nearmiss_headline_control']})", flush=True)

    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    d = auroc_rows(arm_p[idx], arm_n[idx]) - auroc_rows(ctl_p[idx], ctl_n[idx])
    se = float(d.std(ddof=1))
    bar = round(point_ctl, 4) - 2 * se
    draw1 = banked["nearmiss_headline_arm"]

    payload = {
        "read": "R16-H142-T amendment A1 - anti-gaming hold re-priced to "
                "clean control - 2 x SE_delta (paired bootstrap)",
        "se_delta": round(se, 5),
        "repriced_bar": round(bar, 5),
        "draw1_read": draw1,
        "pass": bool(draw1 >= bar),
        "clean_control": round(point_ctl, 4),
        "no_tolerance_bar": 0.7565,
        "delta_point": round(point_arm - point_ctl, 5),
        "delta_in_se_units": round((point_arm - point_ctl) / se, 3),
        "margin_vs_repriced_bar": round(draw1 - bar, 5),
        "bootstrap": {
            "n_resamples": N_BOOT,
            "unit": "headline near-miss PAIR (positive + negative claim together); both "
                    "checkpoints re-scored on the same resample, so the delta is paired",
            "n_headline_pairs": n,
            "seed": BOOT_SEED,
            "delta_mean": round(float(d.mean()), 5),
            "delta_p2_5": round(float(np.percentile(d, 2.5)), 5),
            "delta_p97_5": round(float(np.percentile(d, 97.5)), 5),
            "frac_resamples_delta_ge_0": round(float((d >= 0).mean()), 4),
            "excluded_families": list(EXCLUDED_FAMILIES),
        },
        "scores_source": EVALSET.name,
        "gpu_used": "none - the anti-gaming stage banked per-pair scores for both "
                    "checkpoints at scoring time",
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\n  SE_delta {se:.5f}   re-priced bar {bar:.5f}   draw-1 read {draw1}   "
          f"{'PASS' if payload['pass'] else 'FAIL'}", flush=True)
    print(f"  delta {point_arm - point_ctl:+.5f} = {(point_arm - point_ctl) / se:+.2f} SE   "
          f"95% CI [{payload['bootstrap']['delta_p2_5']:+.5f}, "
          f"{payload['bootstrap']['delta_p97_5']:+.5f}]", flush=True)
    print(f"  results -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

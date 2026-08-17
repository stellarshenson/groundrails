"""attr_pool conformed - sampling uncertainty on the two numbers that decide the
verdict.

C1 passes on the attestation gap by 0.0108 (0.1108 against a bar of > 0.10) and
C5's containment channel fails by 0.0478 (0.5978 against a ceiling of 0.55).  A
margin that thin is worth a confidence interval before it is reported as a pass,
and the failing channel is worth one before it is reported as a fail.  Pairs are
the resampling unit because the two legs of a pair are not independent.

CPU only.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import importlib.util as _ilu
import json
import pathlib

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
EXP = HERE.parent
B = 4000
SEED = 2174


def _mod(name, fname, folder=EXP):
    spec = _ilu.spec_from_file_location(name, folder / fname)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = _mod("h174common", "R20-H174_lane_common.py")


def main():
    df = pl.read_parquet(EXP / "R20-H174_lane_L2_conformed.parquet")
    cont = np.load(HERE / "attr_pool_conformed_containment.npy")
    d = df.select(["pair_id", "label"]).with_columns(
        pl.Series("f", cont[0]), pl.Series("b", cont[1]))
    pf = d.pivot(on="label", index="pair_id", values="f",
                 aggregate_function="first").drop_nulls()
    pb = d.pivot(on="label", index="pair_id", values="b",
                 aggregate_function="first").drop_nulls()
    cp, cn = pf["1"].to_numpy(), pf["0"].to_numpy()
    bp, bn = pb["1"].to_numpy(), pb["0"].to_numpy()
    n = len(cp)
    rng = np.random.default_rng(SEED)

    gaps, aurocs, bgaps = [], [], []
    yy = np.concatenate([np.ones(n), np.zeros(n)])
    for _ in range(B):
        i = rng.integers(0, n, n)
        gaps.append(float((cp[i] >= 0.9).mean() - (cn[i] >= 0.9).mean()))
        bgaps.append(float((bp[i] >= 0.9).mean() - (bn[i] >= 0.9).mean()))
        aurocs.append(float(C.auroc(yy, np.concatenate([cp[i], cn[i]]))))
    gaps, aurocs, bgaps = np.array(gaps), np.array(aurocs), np.array(bgaps)

    def ci(a):
        return {"mean": round(float(a.mean()), 4),
                "p2.5": round(float(np.percentile(a, 2.5)), 4),
                "p97.5": round(float(np.percentile(a, 97.5)), 4),
                "sd": round(float(a.std()), 4)}

    out = {
        "resampling": f"{B} bootstrap resamples over the {n} PAIRS (the two legs of "
        "a pair are not independent, so the pair is the unit)",
        "C1_attestation_gap_full_pool": {
            "point": round(float((cp >= 0.9).mean() - (cn >= 0.9).mean()), 4),
            "bar": "> 0.10 under the gap reading",
            **ci(gaps),
            "share_of_resamples_above_the_bar": round(float((gaps > 0.10).mean()), 4),
        },
        "C1_attestation_gap_best_single_passage": {
            "point": round(float((bp >= 0.9).mean() - (bn >= 0.9).mean()), 4),
            **ci(bgaps),
            "share_of_resamples_above_the_bar": round(float((bgaps > 0.10).mean()), 4),
        },
        "C5_containment_channel_auroc": {
            "point": round(float(C.auroc(yy, np.concatenate([cp, cn]))), 4),
            "ceiling": 0.55,
            **ci(aurocs),
            "share_of_resamples_inside_the_C5_band": round(
                float(((aurocs >= 0.45) & (aurocs <= 0.55)).mean()), 4),
        },
        "reading": "a bar cleared by less than its own sampling spread is a pass "
        "that would not survive a redraw; a bar missed by more than its spread is "
        "a fail that would",
    }
    (HERE / "attr_pool_conformed_ci.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()

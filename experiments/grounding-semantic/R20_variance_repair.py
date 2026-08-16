"""R20 VARIANCE PROTOCOL REPAIR - the estimator was biased 25% high and the pair
census was wrong in both directions.

Found by the round-20 re-adjudication review, verified by direct computation
here (this file IS the banked artifact for the amendment; run it, don't trust it).

DEFECT 1 - THE ESTIMATOR. The adopted protocol derived per-pair sd as
gap * sqrt(pi)/2 (unbiased in EXPECTATION for one pair, since for two iid draws
gap = |d2-d1| with E[gap] = 2*sigma/sqrt(pi)), then RMS-pooled those per-pair
sds. RMS-pooling squares the estimate, and E[(gap*sqrt(pi)/2)^2] =
(pi/2) * sigma^2 - the pooled VARIANCE is inflated by pi/2, the pooled sd by
sqrt(pi/2) = 1.2533. The unbiased pooled estimator is

    sigma^2 = sum(gap_i^2) / (2 n)        since E[gap^2] = 2 sigma^2.

DEFECT 2 - THE CENSUS, two errors:
  (a) OMITTED: the R16-H142 twin pair (0.72498 / 0.70073, seeds 1142 / 2142,
      "identical config, new seed" per the registration at log line 3040) -
      gap 0.02425, the very spread the H155 attribution arm was built to
      decompose. No exclusion reason was ever registered.
  (b) MISCLASSIFIED: the R18-H155 pair (0.72439 / 0.72788) SHARES an init
      (fingerprint cd8417f3..., distinct perms only - log line 3227). Its gap
      estimates the ORDER component alone (~14% of variance per the H155
      verdict), not the full per-draw sd; pooling it into the full-draw
      estimate biases the pool low. It is removed from the full-draw pool and
      recorded separately as the order-component estimate.

The repaired pool: TEN full-seed same-recipe pairs (H155 out, H142 in).

Also recomputed here: the corrected floors, the flagship k=4 empirical check,
and the re-adjudication deltas - which recorded sub-floor kills now resolve at
|z| >= 2 against the flagship k=4 mean under the corrected sd.

Run: uv run python experiments/grounding-semantic/R20_variance_repair.py
"""

import json
import math
import pathlib

HERE = pathlib.Path(__file__).parent

# Full-seed same-recipe pairs: (name, draw means from the banked windowed reads)
FULL_SEED_PAIRS = {
    "R18-H150":       (0.71436, 0.71661),
    "R10-H108":       (0.70373, 0.70618),
    "R9-H105":        (0.70151, 0.70471),
    "DR-lane-control": (0.69826, 0.70713),
    "R18-H152":       (0.70955, 0.71862),
    "R10-H107":       (0.65904, 0.67043),
    "R19-H160":       (0.70870, 0.72365),
    "R14-H133":       (0.68474, 0.70374),
    "DR-lane-margin": (0.67693, 0.70680),
    "R16-H142-twin":  (0.72498, 0.70073),   # the omitted pair
}
PERM_ONLY_PAIRS = {
    "R18-H155": (0.72439, 0.72788),          # shared init cd8417f3..., perms only
}

FLAGSHIP_DRAWS = [0.71436, 0.71661, 0.70870, 0.72365]   # H150 d1/d2 + H160 d3/d4

# Single-read and 2-draw arms whose kill margins sat below the OLD floor,
# re-tested here against the flagship k=4 mean under the corrected sd.
BORDERLINE = {  # name: (k, mean)
    "R19-H159":        (1, 0.68941),
    "R18-H156":        (1, 0.69053),
    "R12-H122":        (1, 0.69147),
    "R11-H118-soup":   (1, 0.69218),
    "R16-H142-G1-arm": (1, 0.69268),
    "R17-H145":        (1, 0.69590),
    "R13-H129":        (1, 0.69709),
    "R17-H146":        (1, 0.69847),
    "R9-H105":         (2, 0.70311),
    "R10-H108":        (2, 0.70495),
}


def pooled_sd(pairs):
    gaps = [abs(b - a) for a, b in pairs.values()]
    n = len(gaps)
    var = sum(g * g for g in gaps) / (2 * n)
    return math.sqrt(var), n, gaps


def main():
    sigma, n, gaps = pooled_sd(FULL_SEED_PAIRS)
    # the biased number the protocol adopted, reproduced for the record
    old_gaps = [abs(b - a) for k, (a, b) in FULL_SEED_PAIRS.items()
                if k != "R16-H142-twin"] + [abs(b - a) for a, b
                                            in PERM_ONLY_PAIRS.values()]
    biased = math.sqrt(sum((g * math.sqrt(math.pi) / 2) ** 2
                           for g in old_gaps) / len(old_gaps))
    unbiased_old_census = math.sqrt(sum(g * g for g in old_gaps) / (2 * len(old_gaps)))

    order_gap = next(abs(b - a) for a, b in PERM_ONLY_PAIRS.values())
    order_sd = order_gap / math.sqrt(2)
    order_share = order_sd ** 2 / sigma ** 2

    fl_mean = sum(FLAGSHIP_DRAWS) / 4
    fl_sd = math.sqrt(sum((x - fl_mean) ** 2 for x in FLAGSHIP_DRAWS) / 3)

    floors = {f"k={k}": round(2 * sigma / math.sqrt(k), 5) for k in (1, 2, 3, 4, 6)}
    se6 = sigma / math.sqrt(6)

    rows = {}
    for name, (k, mean) in sorted(BORDERLINE.items(), key=lambda x: x[1][1]):
        se_diff = sigma * math.sqrt(1 / k + 1 / 4)
        z = (mean - fl_mean) / se_diff
        rows[name] = {"k": k, "mean": mean, "z_vs_flagship_k4": round(z, 2),
                      "resolved_below": bool(z <= -2)}

    res = {
        "what": "variance protocol repair - estimator bias + pair census",
        "defect_1_estimator": {
            "biased_pool_reproduced": round(biased, 5),
            "bias_factor": round(math.sqrt(math.pi / 2), 4),
            "unbiased_same_census": round(unbiased_old_census, 5),
            "rule": "sigma^2 = sum(gap^2)/(2n); E[gap^2] = 2 sigma^2",
        },
        "defect_2_census": {
            "added": {"R16-H142-twin": {"draws": FULL_SEED_PAIRS["R16-H142-twin"],
                                        "gap": 0.02425,
                                        "why": "same recipe, seeds 1142/2142; "
                                               "never excluded on the record"}},
            "removed_to_order_component": {
                "R18-H155": {"draws": PERM_ONLY_PAIRS["R18-H155"],
                             "gap": round(order_gap, 5),
                             "why": "shared init fingerprint - the gap estimates "
                                    "the ORDER component only"}},
        },
        "repaired": {
            "pooled_per_draw_sd": round(sigma, 5),
            "n_pairs": n, "df": n,
            "gaps": {k: round(abs(b - a), 5) for k, (a, b) in FULL_SEED_PAIRS.items()},
            "order_component_sd": round(order_sd, 5),
            "order_variance_share": round(order_share, 4),
            "h155_verdict_cross_check": "H155 recorded order ~14% of SPREAD; "
                                        "this pool reads the variance share",
        },
        "floors_2xSE_at_k": floors,
        "flagship": {
            "k4_mean": round(fl_mean, 5), "k4_empirical_sd": round(fl_sd, 5),
            "k6_SE": round(se6, 5),
            "consistency": "empirical k=4 sd vs pooled - chi2 3 df",
        },
        "re_adjudication_vs_flagship_k4": rows,
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    }
    out = HERE / "R20_variance_repair.json"
    out.write_text(json.dumps(res, indent=1))

    print(f"biased pool (reproduced)   {biased:.5f}   bias x{math.sqrt(math.pi/2):.4f}")
    print(f"unbiased, old census       {unbiased_old_census:.5f}")
    print(f"REPAIRED pooled sd         {sigma:.5f}  on {n} full-seed pairs")
    print(f"order-component sd         {order_sd:.5f}  (variance share {order_share:.1%})")
    print(f"floors 2xSE: {floors}")
    print(f"flagship k4 mean {fl_mean:.5f}  empirical sd {fl_sd:.5f}  k6 SE {se6:.5f}")
    for name, r in rows.items():
        print(f"  {name:<18} k={r['k']}  mean {r['mean']:.5f}  z {r['z_vs_flagship_k4']:+.2f}"
              f"  {'RESOLVED-BELOW' if r['resolved_below'] else 'unresolved'}")
    print(f"-> {out.name}")


if __name__ == "__main__":
    main()

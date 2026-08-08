"""R12-H119 adjudication against the pre-registered two-sided bar.

Bar (docs/experiments/semantic-grounding-experiments.md, round 12):
  improve - mean delta >= +0.003 vs the checkpoint's own original windowed mean
            on BOTH pairs (within a direction), AND finqa delta >= +0.010 on at
            least 3 of the 4 draws
  hold    - no non-numeric subset falls more than 0.015; the two draws of a pair
            must not disagree in sign on finqa

Each direction (strip, add) is adjudicated separately. The "verdict" string
printed inside every read JSON is legacy R8-H90 text and is ignored here; the
adjudication reads only the mean and per-subset AUCs against the original result
files.

Run:  uv run python experiments/grounding-semantic/R12-H119_adjudicate.py
"""

import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R12-H119_verdict.json"

# draw -> original (untransformed) windowed result JSON
ORIGINALS = {
    "h105d1": "R9-H105_windowed_result.json",
    "h105d2": "R9-H105_draw2_windowed_result.json",
    "h108d1": "R10-H108_lane_draw1_windowed_result.json",
    "h108d2": "R10-H108_lane_draw2_windowed_result.json",
}
PAIRS = {"h105": ("h105d1", "h105d2"), "h108": ("h108d1", "h108d2")}
# finqa and tatqa are the numeric-register subsets; the hold clause protects the
# rest.
NUMERIC = {"finqa", "tatqa"}
MEAN_BAR = 0.003
FINQA_BAR = 0.010
HOLD_DROP = 0.015


def load(p):
    d = json.loads((HERE / p).read_text())
    return d["mean"], {k: v["auc"] for k, v in d["per_subset"].items()}


def main():
    audit = json.loads((HERE / "R12-H119_audit_result.json").read_text())
    base = {k: load(v) for k, v in ORIGINALS.items()}

    per_read, directions = {}, {}
    for direction in ("strip", "add"):
        md, fd, holds, worst = {}, {}, {}, {}
        for draw in ORIGINALS:
            m, subs = load(f"R12-H119_{draw}_{direction}_windowed_result.json")
            bm, bsubs = base[draw]
            md[draw] = round(m - bm, 5)
            fd[draw] = round(subs["finqa"] - bsubs["finqa"], 4)
            drops = {s: round(subs[s] - bsubs[s], 4) for s in subs if s not in NUMERIC}
            w = min(drops.items(), key=lambda kv: kv[1])
            worst[draw] = {"subset": w[0], "delta": w[1]}
            holds[draw] = w[1] >= -HOLD_DROP
            # Sensitivity: tatqa is treated as a numeric-register subset and so
            # exempt from the hold clause (amendment 2 deleted every tatqa
            # clause from the registration). Recorded both ways because tatqa
            # carries the largest moves in this experiment.
            all_drops = {s: round(subs[s] - bsubs[s], 4) for s in subs if s != "finqa"}
            wa = min(all_drops.items(), key=lambda kv: kv[1])
            worst[draw]["if_tatqa_counted"] = {"subset": wa[0], "delta": wa[1],
                                               "hold": wa[1] >= -HOLD_DROP}
            per_read[f"{draw}_{direction}"] = {
                "mean": round(m, 5), "original_mean": round(bm, 5), "mean_delta": md[draw],
                "finqa": round(subs["finqa"], 4), "original_finqa": round(bsubs["finqa"], 4),
                "finqa_delta": fd[draw], "worst_non_numeric": worst[draw],
                "per_subset_delta": {s: round(subs[s] - bsubs[s], 4) for s in subs},
            }

        pair_mean = {p: round(sum(md[d] for d in ds) / 2, 5) for p, ds in PAIRS.items()}
        mean_ok = all(v >= MEAN_BAR for v in pair_mean.values())
        finqa_n = sum(v >= FINQA_BAR for v in fd.values())
        finqa_ok = finqa_n >= 3
        sign_ok = all(
            (fd[a] > 0) == (fd[b] > 0) or fd[a] == 0 or fd[b] == 0 for a, b in PAIRS.values()
        )
        hold_ok = all(holds.values()) and sign_ok
        improve = mean_ok and finqa_ok
        verdict = "PASS" if (improve and hold_ok) else "REFUTED"

        directions[direction] = {
            "mean_deltas": md, "pair_mean_deltas": pair_mean, "finqa_deltas": fd,
            "worst_non_numeric": worst, "holds": holds, "sign_agreement": sign_ok,
            "mean_bar_met": mean_ok, "finqa_bar_met": finqa_ok,
            "finqa_draws_over_bar": finqa_n, "hold_met": hold_ok, "verdict": verdict,
        }

        print(f"\n=== direction {direction} ===")
        for d in ORIGINALS:
            r = per_read[f"{d}_{direction}"]
            print(
                f"  {d:8s} mean {r['mean']:.5f} (orig {r['original_mean']:.5f}, "
                f"{r['mean_delta']:+.5f})   finqa {r['finqa']:.4f} ({r['finqa_delta']:+.4f})   "
                f"worst non-numeric {r['worst_non_numeric']['subset']} "
                f"{r['worst_non_numeric']['delta']:+.4f}"
            )
        print(
            f"  pair means {pair_mean}  mean_bar={mean_ok}  "
            f"finqa>=+0.010 on {finqa_n}/4 -> {finqa_ok}  hold={hold_ok}  => {verdict}"
        )

    overall = (
        "PASS" if any(v["verdict"] == "PASS" for v in directions.values()) else "REFUTED"
    )
    print(f"\noverall: {overall}")

    OUT.write_text(
        json.dumps(
            {
                "audit": {
                    "baseline_agreement": audit["baseline_agreement"],
                    "baseline_agreement_bare": audit["baseline_agreement_bare"],
                    "per_rule": audit["per_rule"],
                    "full_transform": audit["full_transform"],
                    "verdict": audit["verdict"],
                },
                "shipped_rules": audit["shipped_rules"],
                "bar": {
                    "mean_delta": MEAN_BAR, "finqa_delta": FINQA_BAR,
                    "finqa_draws_required": 3, "hold_max_drop": HOLD_DROP,
                    "numeric_subsets_exempt_from_hold": sorted(NUMERIC),
                },
                "per_read": per_read,
                "strip": directions["strip"],
                "add": directions["add"],
                "overall_verdict": overall,
            },
            indent=2,
        )
    )
    print(f"  verdict -> {OUT}")


if __name__ == "__main__":
    main()

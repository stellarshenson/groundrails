"""R11-H117 kill-gate 2 - mechanical adjudication of the 3-arm probe.

Registered rules (applied verbatim, no post-hoc discretion):
  - a lambda arm is VOID if the A6 magnitude ratio mean(lam*hinge)/mean(BCE) > 0.25
  - a lambda PASSES iff gold_full drop vs the lam=0 probe-control <= 0.01 AND
    held-out pair-accuracy improves over the control
  - KILL if no lambda passes (gold_full damage at both, or no pair-acc gain at both)
  - else PROCEED with the SMALLER passing lambda, unless the larger's pair-acc
    gain exceeds the smaller's by > 0.02 (A5 tie-break)
  - A2: pair-acc up with both global reads flat (|delta| < 0.005 on gold_full AND
    ragtruth_en) is flagged "necessary-not-sufficient, flat-global warning"

Run: uv run python experiments/grounding-semantic/R11-H117_adjudicate.py
"""

import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R11-H117_probe_result.json"
ARMS = ["0", "0.1", "0.3"]
GF_DROP_MAX = 0.01
FLAT_EPS = 0.005
TIEBREAK = 0.02
A6_CAP = 0.25


def load(lam):
    tr = json.loads((HERE / f"R11-H117_probe_lam{lam}_train.json").read_text())
    gf = json.loads((HERE / f"R11-H117_probe_lam{lam}_goldfull.json").read_text())
    rd = json.loads((HERE / f"R11-H117_probe_lam{lam}_read.json").read_text())
    return {
        "lambda_margin": float(lam),
        "gold_full_auc": gf["gold_full_auc"],
        "ragtruth_en_auc": rd["ragtruth_en"]["auc"],
        "pair_acc": rd["heldout_pairs"]["overall"]["pair_acc"],
        "pair_acc_verbatim": rd["heldout_pairs"]["verbatim"]["pair_acc"],
        "pair_acc_non_verbatim": rd["heldout_pairs"]["non_verbatim"]["pair_acc"],
        "mean_gap": rd["heldout_pairs"]["overall"]["mean_gap"],
        "hinge_mean": tr["hinge_mean"],
        "a6_ratio": tr["a6_ratio"],
        "a6_void": tr["a6_ratio"] > A6_CAP,
        "bce_mean": tr["bce_mean"],
        "steps": tr["max_steps"],
        "train_seconds": tr["train_seconds"],
    }


def main():
    arms = {lam: load(lam) for lam in ARMS}
    ctl = arms["0"]

    verdicts = {}
    for lam in ("0.1", "0.3"):
        a = arms[lam]
        gf_delta = round(a["gold_full_auc"] - ctl["gold_full_auc"], 4)
        rt_delta = round(a["ragtruth_en_auc"] - ctl["ragtruth_en_auc"], 4)
        pa_gain = round(a["pair_acc"] - ctl["pair_acc"], 4)
        calib_ok = gf_delta >= -GF_DROP_MAX
        pair_ok = pa_gain > 0
        flat_global = abs(gf_delta) < FLAT_EPS and abs(rt_delta) < FLAT_EPS
        verdicts[lam] = {
            "gold_full_delta": gf_delta, "ragtruth_en_delta": rt_delta,
            "pair_acc_gain": pa_gain,
            "pair_acc_gain_verbatim": round(
                a["pair_acc_verbatim"] - ctl["pair_acc_verbatim"], 4),
            "pair_acc_gain_non_verbatim": round(
                a["pair_acc_non_verbatim"] - ctl["pair_acc_non_verbatim"], 4),
            "calibration_ok": calib_ok, "pair_acc_improves": pair_ok,
            "a6_void": a["a6_void"],
            "passes": bool(calib_ok and pair_ok and not a["a6_void"]),
            "flat_global_warning": bool(flat_global and pair_ok),
        }

    passing = [lam for lam in ("0.1", "0.3") if verdicts[lam]["passes"]]
    if not passing:
        why = []
        if not verdicts["0.1"]["calibration_ok"] and not verdicts["0.3"]["calibration_ok"]:
            why.append("gold_full drops > 0.01 at BOTH lambdas (calibration damage)")
        if not verdicts["0.1"]["pair_acc_improves"] and not verdicts["0.3"]["pair_acc_improves"]:
            why.append("held-out pair-accuracy fails to improve at BOTH lambdas (no mechanism)")
        for lam in ("0.1", "0.3"):
            if verdicts[lam]["a6_void"]:
                why.append(f"lambda {lam} VOID on the A6 cap (ratio > 0.25)")
            elif not verdicts[lam]["passes"]:
                why.append(
                    f"lambda {lam} fails: calibration_ok={verdicts[lam]['calibration_ok']}, "
                    f"pair_acc_improves={verdicts[lam]['pair_acc_improves']}")
        verdict, chosen, reason = "KILL", None, "; ".join(why)
    else:
        chosen = passing[0]
        reason = f"smaller passing lambda ({chosen})"
        if len(passing) == 2:
            gain_s = verdicts["0.1"]["pair_acc_gain"]
            gain_l = verdicts["0.3"]["pair_acc_gain"]
            if gain_l - gain_s > TIEBREAK:
                chosen = "0.3"
                reason = (f"A5 tie-break: larger lambda pair-acc gain {gain_l} exceeds "
                          f"smaller {gain_s} by > {TIEBREAK}")
        verdict = "PROCEED"

    res = {
        "hypothesis": "R11-H117 PAIRED-MARGIN - kill-gate 2 probe",
        "design": {"steps_per_arm": ctl["steps"], "draw_seed": 1117,
                   "shared_perm_prefix": True, "margin_m": 0.25,
                   "heldout_pairs": 2000, "arms": [0.0, 0.1, 0.3]},
        "arms": arms, "per_lambda": verdicts,
        "verdict": verdict, "chosen_lambda": None if chosen is None else float(chosen),
        "reason": reason,
        "a2_flat_global_warning": {
            lam: verdicts[lam]["flat_global_warning"] for lam in ("0.1", "0.3")},
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"=== H117 PROBE VERDICT: {verdict} lambda={chosen} ===")


if __name__ == "__main__":
    main()

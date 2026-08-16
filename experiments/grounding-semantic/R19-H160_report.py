"""R19-H160 BARS REPORT - every registered number against every registered bar.

Registered in docs/experiments/semantic-grounding-experiments.md, block "R19-H160
SEED-DIVERSE WEIGHT AVERAGING AS THE SERVED ARTIFACT". This script ADJUDICATES
NOTHING - it assembles the arm's reads into the bar table the coordinator rules
on, computing each number from the banked result JSONs and nothing else. CPU only.

Bars assembled here:

    PRIMARY      soup B blind windowed decomposed-min mean on the frozen R8-H77
                 gate: GRADUATE >= 0.72049 (flagship 0.71549 + the standing 0.005
                 margin) with all ten subset floors and every hold green;
                 KILL < 0.71549 (the pair mean of its own ingredients) or on any
                 hold breach
    FLOORS       floor_i = flagship_i - max(0.02, swing_i, 2 x SE_i), where
                 flagship_i is the H150 pair's per-subset mean, swing_i the pair's
                 own |draw1 - draw2| on that subset, and SE_i the Hanley-McNeil
                 analytic AUROC standard error from the subset's class counts in
                 the frozen R8-H77 arena (the R14 evidence-E3 convention)
    HOLDS        gold_full >= 0.84, RAGTruth non-EN >= 0.82, anti-gaming held-out
                 near-miss headline AUROC >= 0.7438
    DRAW BAR     draws 3 and 4 each >= 0.7079 individually - the H150 registered
                 draw bar, a recipe-reproduction check independent of the soup
    K-SWEEP      k = 2 (soup B), 3 and 4 uniform soups, predicted monotone
                 non-decreasing in k; gold_full on every soup, predicted up

Run:  uv run python experiments/grounding-semantic/R19-H160_report.py
"""

import json
import math
import pathlib

HERE = pathlib.Path(__file__).parent

FLAGSHIP_MEAN = 0.71549          # the H150 pair mean - the incumbent flagship
GRADUATE_BAR = 0.72049           # flagship + the standing 0.005 promotion margin
KILL_BELOW = 0.71549             # a soup must at least beat its ingredients' mean
DRAW_BAR = 0.7079                # the H150 registered per-draw bar
GOLD_FULL_HOLD = 0.84
NONEN_HOLD = 0.82
ANTIGAMING_HOLD = 0.7438
FLOOR_MIN_TOLERANCE = 0.02

FLAGSHIP_READS = ("R18-H150_arm_draw1_windowed_result.json",
                  "R18-H150_arm_draw2_windowed_result.json")
ARENA = HERE / "R8-H77_arena.json"

DRAWS = {3: ("R19-H160_arm_draw3_windowed_result.json",
             "R19-H160_arm_draw3_result.json"),
         4: ("R19-H160_arm_draw4_windowed_result.json",
             "R19-H160_arm_draw4_result.json")}

SOUP_RESULT = HERE / "R19-H160_soup_result.json"
ANTIGAMING = HERE / "R19-H160-soupB_antigaming_draw1_result.json"
PROBES = HERE / "R19-H160-soupB_probes_draw1_result.json"
SOUP_B_CKPT = "R19-H160-soup-B"
OUT = HERE / "R19-H160_bars_report.json"


def load(p):
    p = pathlib.Path(p)
    return json.loads(p.read_text()) if p.exists() and p.stat().st_size else None


def se_hanley(a, n1, n0):
    """Analytic AUROC standard error, Hanley-McNeil - the R14 evidence-E3 and
    R17-H148/H149 convention, evaluated at max(A, 1 - A) so the estimator stays
    on the well-defined branch."""
    a_ = max(a, 1 - a)
    q1 = a_ / (2 - a_)
    q2 = 2 * a_ ** 2 / (1 + a_)
    return math.sqrt((a_ * (1 - a_) + (n1 - 1) * (q1 - a_ ** 2)
                      + (n0 - 1) * (q2 - a_ ** 2)) / (n0 * n1))


def floors():
    """The ten variance-aware subset floors, priced off the flagship H150 pair."""
    d1, d2 = (load(HERE / f) for f in FLAGSHIP_READS)
    arena = load(ARENA)["per_subset"]
    out = {}
    for sub in d1["per_subset"]:
        a1 = d1["per_subset"][sub]["auc"]
        a2 = d2["per_subset"][sub]["auc"]
        flag = (a1 + a2) / 2
        swing = abs(a1 - a2)
        n = int(arena[sub]["n"])
        n1 = round(n * float(arena[sub]["grounded_rate"]))
        n0 = n - n1
        se = se_hanley(flag, n1, n0)
        tol = max(FLOOR_MIN_TOLERANCE, swing, 2 * se)
        out[sub] = {
            "flagship_d1": a1, "flagship_d2": a2, "flagship_mean": round(flag, 5),
            "swing": round(swing, 5), "n": n, "n_grounded": n1, "n_halluc": n0,
            "se_hanley_mcneil": round(se, 5), "two_se": round(2 * se, 5),
            # reported beside the floor, NOT used in it: the SE of an independent
            # difference of two such AUCs, the E3 `SE_of_diff_indep` column. The
            # registration's formula says 2 x SE_i, so the floor uses the plain SE
            "two_se_of_independent_diff": round(2 * se * math.sqrt(2), 5),
            "tolerance": round(tol, 5),
            "tolerance_driver": ("swing" if tol == swing else
                                 "2xSE" if tol == 2 * se else "0.02 minimum"),
            "floor": round(flag - tol, 5),
        }
    return out


def main():
    fl = floors()
    rep = {
        "arm": "R19-H160 seed-diverse weight averaging as the served artifact",
        "status": "NOT ADJUDICATED HERE - the coordinator holds the verdict",
        "bars": {
            "primary_graduate_at": GRADUATE_BAR, "primary_kill_below": KILL_BELOW,
            "flagship_mean": FLAGSHIP_MEAN, "draw_bar": DRAW_BAR,
            "gold_full_hold": GOLD_FULL_HOLD, "nonen_hold": NONEN_HOLD,
            "antigaming_hold": ANTIGAMING_HOLD,
            "floor_formula": "floor_i = flagship_i - max(0.02, swing_i, 2 x SE_i)",
        },
        "subset_floors": fl,
    }

    # --- the draws (recipe-reproduction check, independent of the soup) ---------
    draws = {}
    for n, (win_f, train_f) in DRAWS.items():
        w, t = load(HERE / win_f), load(HERE / train_f)
        if w is None or t is None:
            draws[n] = {"status": "MISSING", "expected": [win_f, train_f]}
            continue
        draws[n] = {
            "status": "read", "seed": t.get("seed"),
            "checkpoint": t.get("checkpoint"),
            "executor": t.get("executor"),
            "windowed_mean": w["mean"],
            "per_subset": {k: v["auc"] for k, v in w["per_subset"].items()},
            "gold_full": t["gold_full"]["auc"],
            "gold": t["gold"]["auc"],
            "ragtruth_en": t["ragtruth_en"]["auc"],
            "ragtruth_nonen": t["ragtruth_nonen"]["auc"],
            "init_fingerprint": t.get("init_fingerprint"),
            "perm_fingerprint": t.get("perm_fingerprint"),
            "n_steps": t.get("n_steps"),
            "train_seconds": t.get("train_seconds"),
            "draw_bar": DRAW_BAR,
            "draw_bar_pass": bool(w["mean"] >= DRAW_BAR),
            "delta_vs_draw_bar": round(w["mean"] - DRAW_BAR, 5),
        }
    rep["draws"] = draws
    read = [d for d in draws.values() if d.get("status") == "read"]
    if len(read) == 2:
        rep["draw_pair"] = {
            "mean_of_draws": round(sum(d["windowed_mean"] for d in read) / 2, 5),
            "spread": round(abs(read[0]["windowed_mean"] - read[1]["windowed_mean"]), 5),
            "both_inside_draw_bar": all(d["draw_bar_pass"] for d in read),
        }

    # --- the soups --------------------------------------------------------------
    soup = load(SOUP_RESULT)
    ag = load(ANTIGAMING)
    pb = load(PROBES)
    cells = (soup or {}).get("cells", {})
    b = cells.get("soupB")
    if b and b.get("status") == "read":
        mean = b["soup"]["windowed_mean"]
        per = b["soup"]["per_subset"]
        floor_rows = {s: {"soup": per[s], "floor": fl[s]["floor"],
                          "flagship_mean": fl[s]["flagship_mean"],
                          "delta_vs_flagship": round(per[s] - fl[s]["flagship_mean"], 5),
                          "margin_over_floor": round(per[s] - fl[s]["floor"], 5),
                          "pass": bool(per[s] >= fl[s]["floor"])}
                      for s in fl}
        ag_head = None
        if ag:
            ck = ag["checkpoints"].get(f"{SOUP_B_CKPT}") or ag["checkpoints"].get(
                ag.get("arm_checkpoint", ""))
            ag_head = (ck or {}).get("nearmiss_headline", {}).get("auroc_pos_vs_neg")
        holds = {
            "gold_full": {"value": b["soup"]["gold_full"], "bar": GOLD_FULL_HOLD,
                          "pass": bool(b["soup"]["gold_full"] >= GOLD_FULL_HOLD)},
            "ragtruth_nonen": {"value": b["soup"].get("ragtruth_nonen"),
                               "bar": NONEN_HOLD,
                               "pass": bool((b["soup"].get("ragtruth_nonen") or 0)
                                            >= NONEN_HOLD)},
            "antigaming_nearmiss_headline": {
                "value": ag_head, "bar": ANTIGAMING_HOLD,
                "pass": bool((ag_head or 0) >= ANTIGAMING_HOLD)},
        }
        floors_all = all(r["pass"] for r in floor_rows.values())
        holds_all = all(h["pass"] for h in holds.values())
        if mean >= GRADUATE_BAR and floors_all and holds_all:
            verdict = "GRADUATE"
        elif mean < KILL_BELOW or not holds_all:
            verdict = "KILL"
        else:
            verdict = "NEITHER"
        rep["soup_B"] = {
            "checkpoint": b.get("soup_checkpoint"),
            "windowed_mean": mean,
            "per_subset": per,
            "ingredient_windowed_means": b.get("ingredient_windowed_means"),
            "ingredient_mean": b["arena"]["ingredient_mean"],
            "delta_vs_ingredient_mean": b["arena"]["soup_minus_ingredient_mean"],
            "delta_vs_better_ingredient": b["arena"]["soup_minus_better_ingredient"],
            "delta_vs_flagship": round(mean - FLAGSHIP_MEAN, 5),
            "delta_vs_graduate_bar": round(mean - GRADUATE_BAR, 5),
            "delta_vs_kill_bar": round(mean - KILL_BELOW, 5),
            "gold_full": b["soup"]["gold_full"],
            "gold": b["soup"].get("gold"),
            "ragtruth_en": b["soup"].get("ragtruth_en"),
            "ragtruth_nonen": b["soup"].get("ragtruth_nonen"),
            "ragtruth_nonen_per_lang": b["soup"].get("ragtruth_nonen_per_lang"),
            "antigaming_nearmiss_headline": ag_head,
            "probe_bank": (pb or {}).get("headline"),
            "basin_pairwise": b.get("basin_pairwise"),
            "subset_floors": floor_rows,
            "floors_all_green": floors_all,
            "holds": holds,
            "holds_all_green": holds_all,
            "primary_at_or_over_graduate": bool(mean >= GRADUATE_BAR),
            "primary_under_kill": bool(mean < KILL_BELOW),
            "verdict_by_the_registered_bars": verdict,
        }
    else:
        rep["soup_B"] = {"status": "MISSING - run R19-H160_soup.py --stage run"}

    ks = {}
    for c in ("soupB", "k3", "k4"):
        r = cells.get(c)
        if r and r.get("status") == "read":
            ks[c] = {"k": r["k"], "windowed_mean": r["soup"]["windowed_mean"],
                     "gold_full": r["soup"]["gold_full"],
                     "ingredient_mean": r["arena"]["ingredient_mean"],
                     "delta_vs_ingredient_mean":
                         r["arena"]["soup_minus_ingredient_mean"]}
    if {"soupB", "k3", "k4"} <= set(ks):
        seq = [ks["soupB"]["windowed_mean"], ks["k3"]["windowed_mean"],
               ks["k4"]["windowed_mean"]]
        ks["monotone_k2_k3_k4"] = bool(seq[1] >= seq[0] and seq[2] >= seq[1])
        ks["k3_ge_k2"] = bool(seq[1] >= seq[0])
        ks["k4_ge_k3"] = bool(seq[2] >= seq[1])
        ks["note"] = ("k=2 is soup B (draws 3+4); k=3 and k=4 add the banked H150 "
                      "draws, so the k-sweep mixes monolithic-trained (H150 d1, d2) "
                      "and split-trained (H160 d3, d4) ingredients")
    rep["k_sweep"] = ks

    OUT.write_text(json.dumps(rep, indent=2))

    # --- console table ----------------------------------------------------------
    print("=== R19-H160 BARS REPORT ===", flush=True)
    for n, d in draws.items():
        if d.get("status") == "read":
            print(f"  draw {n} (seed {d['seed']}): windowed {d['windowed_mean']:.5f}  "
                  f"bar {DRAW_BAR}  {'PASS' if d['draw_bar_pass'] else 'FAIL'}  "
                  f"gold_full {d['gold_full']:.4f}  non-EN {d['ragtruth_nonen']:.4f}",
                  flush=True)
        else:
            print(f"  draw {n}: {d['status']}", flush=True)
    sb = rep["soup_B"]
    if "windowed_mean" in sb:
        print(f"\n  SOUP B windowed mean {sb['windowed_mean']:.5f}  "
              f"(graduate {GRADUATE_BAR} -> {sb['delta_vs_graduate_bar']:+.5f}; "
              f"kill below {KILL_BELOW} -> {sb['delta_vs_kill_bar']:+.5f})", flush=True)
        print(f"  vs ingredient mean {sb['ingredient_mean']:.5f}: "
              f"{sb['delta_vs_ingredient_mean']:+.5f}   vs better ingredient: "
              f"{sb['delta_vs_better_ingredient']:+.5f}", flush=True)
        print("  subset floors:", flush=True)
        for s, r in sb["subset_floors"].items():
            print(f"    {s:12s} soup {r['soup']:.4f}  floor {r['floor']:.4f}  "
                  f"margin {r['margin_over_floor']:+.4f}  "
                  f"{'PASS' if r['pass'] else 'FAIL'}", flush=True)
        for h, r in sb["holds"].items():
            print(f"  hold {h:30s} {r['value']}  bar {r['bar']}  "
                  f"{'PASS' if r['pass'] else 'FAIL'}", flush=True)
        print(f"  VERDICT BY THE REGISTERED BARS: "
              f"{sb['verdict_by_the_registered_bars']}", flush=True)
    else:
        print(f"\n  SOUP B: {sb.get('status')}", flush=True)
    if ks:
        print("\n  k-sweep:", flush=True)
        for c in ("soupB", "k3", "k4"):
            if c in ks:
                print(f"    {c:6s} k={ks[c]['k']}  mean {ks[c]['windowed_mean']:.5f}  "
                      f"gold_full {ks[c]['gold_full']:.4f}", flush=True)
        if "monotone_k2_k3_k4" in ks:
            print(f"    monotone non-decreasing in k: {ks['monotone_k2_k3_k4']}",
                  flush=True)
    print(f"\nreport -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

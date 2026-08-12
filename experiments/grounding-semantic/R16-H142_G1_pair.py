"""R16-H142 G1 (A1) - the pair delta: arm minus twin.

The two runs share their presentation, their mix, their objective, their seed
and - because the adapter is constructed last and the seed is re-issued after
construction - their init draws and dropout stream. The only difference is the
adapter. So arm minus twin is the adapter's effect and nothing else, which is
what review constraint F3 asked for.

This script reads the six banked result files, tabulates the deltas and sets the
bar FLAGS. It does not adjudicate: no verdict word is written, and a flag is a
measurement of a threshold, not a decision.

Two flag families:

  pair-relative   the adapter question - arm minus twin against the banked
                  control pair's seed sd (pubmedqa 0.0216, hotpotqa 0.0144,
                  tatqa 0.0290)
  absolute        the registered holds, applied to BOTH runs so the coordinator
                  can see whether a breach came from the adapter or from A1's
                  windowed presentation

Run:  uv run python experiments/grounding-semantic/R16-H142_G1_pair.py
"""

import importlib.util
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R16-H142_G1_pair_result.json"


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ARM = _mod("g1arm", "R16-H142_G1_arm.py")
BARS, SEED_SD, CONTROL = ARM.BARS, ARM.SEED_SD, ARM.CONTROL

PRIMARY = "pubmedqa"
GUARDS = ("hotpotqa", "tatqa")


def load(name):
    p = HERE / name
    if not p.exists():
        raise SystemExit(f"missing input: {p}")
    return json.loads(p.read_text())


def main():
    indomain = {r: load(ARM.RUNS[r]["out"]) for r in ("twin", "arm")}
    windowed = {r: load(f"R16-H142_G1_{r}_windowed_result.json") for r in ("twin", "arm")}
    truncated = {r: load(f"R16-H142_G1_{r}_truncated_result.json") for r in ("twin", "arm")}

    subs = sorted(windowed["arm"]["per_subset"])
    per_subset = {}
    for s in subs:
        t = windowed["twin"]["per_subset"][s]["auc"]
        a = windowed["arm"]["per_subset"][s]["auc"]
        per_subset[s] = {
            "twin": t, "arm": a, "delta": round(a - t, 4),
            "banked_control_pair": windowed["arm"]["per_subset"][s].get(
                "banked_control_windowed"),
        }
        if s in SEED_SD:
            per_subset[s]["delta_in_seed_sd"] = round((a - t) / SEED_SD[s], 2)

    means = {r: windowed[r]["mean"] for r in ("twin", "arm")}
    d_primary = per_subset[PRIMARY]["delta"]

    pair_flags = {
        f"{PRIMARY}_delta_ge_2sd": bool(d_primary >= 2 * SEED_SD[PRIMARY]),
        f"{PRIMARY}_delta_ge_1sd": bool(d_primary >= SEED_SD[PRIMARY]),
        f"{PRIMARY}_delta_below_1sd_kill_shaped": bool(d_primary < SEED_SD[PRIMARY]),
    }
    for g in GUARDS:
        pair_flags[f"{g}_delta_within_2sd"] = bool(
            per_subset[g]["delta"] >= -2 * SEED_SD[g])
    pair_flags["arena_mean_delta"] = round(means["arm"] - means["twin"], 5)

    absolute = {}
    for r in ("twin", "arm"):
        w = windowed[r]["per_subset"]
        absolute[r] = {
            "arena_mean": means[r],
            "arena_mean_ge_bar": bool(means[r] >= BARS["arena_mean_min"]),
            "pubmedqa_ge_bar": bool(w["pubmedqa"]["auc"] >= BARS["pubmedqa_primary_min"]),
            "hotpotqa_ge_bar": bool(w["hotpotqa"]["auc"] >= BARS["hotpotqa_min"]),
            "tatqa_ge_bar": bool(w["tatqa"]["auc"] >= BARS["tatqa_min"]),
            "gold_full": indomain[r]["gold_full"]["auc"],
            "gold_full_ge_bar": bool(
                indomain[r]["gold_full"]["auc"] >= BARS["gold_full_min"]),
            "ragtruth_nonen": indomain[r]["ragtruth_nonen"]["auc"],
            "ragtruth_nonen_ge_bar": bool(
                indomain[r]["ragtruth_nonen"]["auc"] >= BARS["ragtruth_nonen_min"]),
        }

    fps = {r: indomain[r]["init_fingerprint"] for r in ("twin", "arm")}
    pairing = {
        "init_fingerprints": fps,
        "init_paired": bool(fps["twin"] == fps["arm"]),
        "perm_fingerprints": {r: indomain[r]["perm_fingerprint"] for r in ("twin", "arm")},
        "perm_matched": bool(
            indomain["twin"]["perm_fingerprint"] == indomain["arm"]["perm_fingerprint"]),
        "twin_adapter_active": indomain["twin"]["adapter_active"],
        "arm_adapter_active": indomain["arm"]["adapter_active"],
    }

    payload = {
        "experiment": "R16-H142 G1 amendment A1 - adapter ablation pair, arm minus twin",
        "difference_under_test": "the zero-init adapter side-head; presentation, mix, "
                                 "objective, seed, init draws and dropout stream shared",
        "pairing": pairing,
        "windowed_primary_read": {"per_subset": per_subset, "means": means},
        "truncated_read": {
            "means": {r: truncated[r]["mean"] for r in ("twin", "arm")},
            "per_subset": {s: {"twin": truncated["twin"]["per_subset"][s]["auc"],
                               "arm": truncated["arm"]["per_subset"][s]["auc"],
                               "delta": round(truncated["arm"]["per_subset"][s]["auc"]
                                              - truncated["twin"]["per_subset"][s]["auc"], 4)}
                          for s in subs},
        },
        "pair_flags": pair_flags,
        "absolute_flags": absolute,
        "bars": BARS, "seed_sd": SEED_SD, "banked_control_pair": CONTROL,
        "note": "Flags are threshold measurements. Numbers recorded, not adjudicated - "
                "the coordinator adjudicates.",
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print("=" * 84)
    print("R16-H142 G1 (A1) PAIR - arm minus twin, windowed PRIMARY read")
    print("=" * 84)
    print(f"  init-paired: {pairing['init_paired']}   perm matched: {pairing['perm_matched']}")
    for s in subs:
        v = per_subset[s]
        sd = f"  ({v['delta_in_seed_sd']:+.2f} sd)" if "delta_in_seed_sd" in v else ""
        print(f"  {s:14s} twin {v['twin']:.4f}  arm {v['arm']:.4f}  "
              f"delta {v['delta']:+.4f}{sd}")
    print(f"  {'MEAN':14s} twin {means['twin']:.5f}  arm {means['arm']:.5f}  "
          f"delta {means['arm'] - means['twin']:+.5f}")
    print(f"\n  pair flags: {pair_flags}")
    print(f"  results -> {OUT}")


if __name__ == "__main__":
    main()

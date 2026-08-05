"""R9_PC_headroom - aggregation-headroom diagnostic on windowed sentence scores.

Precursor P-C step 2 (round 9, analysis-only). Reads R9_PC_windowed_dump.json
(frozen H90, per-sentence windowed scores) and computes per-subset AUC under
fixed parameter-free aggregators:

  hard-min        min(s)                      - the shipped aggregation
  mean            mean(s)
  softmin tau     -tau * log(mean(exp(-s/tau))), tau in {0.5, 1, 2, 4}
  drop-argmin     min of s excluding the single lowest (n_sent > 1)

Sanity gate: the hard-min per-subset AUCs must reproduce R8-H101_result.json
(the recorded H90 windowed read) EXACTLY at 4 decimals, else abort.

Pre-registered threshold: aggregation work stays alive ONLY if some single
fixed aggregator beats hard-min by >= +0.01 on the 10-subset MEAN. The
subset-level best ("ceiling") is a NON-REGISTRABLE diagnostic - selection on
benchmark labels - printed for mechanism reading only.

Run:  uv run python experiments/grounding-semantic/R9_PC_headroom.py   (CPU)
"""

import importlib.util
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ARENA = _mod("arena", "R8-H77_unseen_arena.py")

DUMP = HERE / "R9_PC_windowed_dump.json"
REF = HERE / "R8-H101_result.json"
OUT = HERE / "R9_PC_result.json"

TAUS = (0.5, 1.0, 2.0, 4.0)


def softmin(s, tau):
    s = np.asarray(s, dtype=np.float64)
    return float(-tau * np.log(np.mean(np.exp(-s / tau))))


def drop_argmin(s):
    if len(s) <= 1:
        return float(s[0])
    return float(np.partition(np.asarray(s, dtype=np.float64), 1)[1])


AGGS = {
    "hard_min": lambda s: float(np.min(s)),
    "mean": lambda s: float(np.mean(s)),
    **{f"softmin_tau{t:g}": (lambda s, t=t: softmin(s, t)) for t in TAUS},
    "drop_argmin": drop_argmin,
}


def main():
    dump = json.loads(DUMP.read_text())
    ref = json.loads(REF.read_text())["per_subset"]
    subsets = sorted({r["subset"] for r in dump["records"]})

    by_sub = {s: [r for r in dump["records"] if r["subset"] == s] for s in subsets}

    # Sanity gate: hard-min must reproduce the recorded H90 windowed read exactly.
    for s in subsets:
        y = np.array([r["label"] for r in by_sub[s]])
        resp = np.array([AGGS["hard_min"](r["sent_scores"]) for r in by_sub[s]])
        auc, _, _ = ARENA.M59.auc_and_f1(y, resp)
        if round(auc, 4) != ref[s]["auc"]:
            raise SystemExit(
                f"SANITY MISMATCH {s}: hard-min from dump {auc:.4f} != recorded {ref[s]['auc']:.4f}"
            )
    print("sanity: hard-min reproduces R8-H101_result.json EXACTLY on all 10 subsets", flush=True)

    table = {}
    for s in subsets:
        y = np.array([r["label"] for r in by_sub[s]])
        table[s] = {}
        for name, fn in AGGS.items():
            resp = np.array([fn(r["sent_scores"]) for r in by_sub[s]])
            auc, _, _ = ARENA.M59.auc_and_f1(y, resp)
            table[s][name] = round(auc, 4)

    means = {name: round(float(np.mean([table[s][name] for s in subsets])), 4) for name in AGGS}
    hard = means["hard_min"]
    best_name = max(means, key=means.get)
    headroom = round(means[best_name] - hard, 4)
    alive = best_name != "hard_min" and headroom >= 0.01

    ceiling_rows = {s: max(table[s], key=table[s].get) for s in subsets}
    ceiling = round(float(np.mean([table[s][ceiling_rows[s]] for s in subsets])), 4)

    hdr = ["subset"] + list(AGGS)
    print("\n  " + "  ".join(f"{h:>13s}" for h in hdr))
    for s in subsets:
        print("  " + f"{s:>13s}" + "  " + "  ".join(f"{table[s][n]:13.4f}" for n in AGGS))
    print("  " + f"{'MEAN':>13s}" + "  " + "  ".join(f"{means[n]:13.4f}" for n in AGGS))

    print("\n" + "=" * 92)
    print("R9 P-C RESULT - windowed aggregation headroom, frozen H90 (adjudication external)")
    print("=" * 92)
    print(f"  hard-min mean {hard:.4f}   best fixed aggregator {best_name} {means[best_name]:.4f}   headroom {headroom:+.4f}")
    print(f"  threshold (>= +0.01 mean): {'FIRED - aggregation line alive' if alive else 'NOT FIRED'}")
    print(f"  subset-level-best ceiling {ceiling:.4f} - NON-REGISTRABLE (selection on benchmark), diagnostic only")

    OUT.write_text(json.dumps({
        "per_subset": table, "means": means, "hard_min_mean": hard,
        "best_fixed": best_name, "best_fixed_mean": means[best_name],
        "headroom": headroom, "threshold_fired": bool(alive),
        "ceiling_non_registrable": ceiling, "ceiling_choices": ceiling_rows,
        "model": dump["model"],
    }, indent=2))
    print(f"\n  results -> {OUT}")
    print("=== R9_PC HEADROOM DONE ===")


if __name__ == "__main__":
    main()

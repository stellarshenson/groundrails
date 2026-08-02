"""R8-H88 - complementarity read: unweighted mean of ERM (H84) and DANN (H79 v1).

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 8).

The two students trained on IDENTICAL data with different objectives and their
blind per-subset profiles decorrelate: per-subset max averages 0.6649 against
members 0.6450 / 0.6320. This run measures how much of that oracle gap the
cheapest legal combination recovers - the unweighted mean of per-pair sigmoid
probabilities, max-over-chunks as always, both members frozen as-is. No weight
tuning against the arena: that would un-blind it.

DIAGNOSTIC, not a ship candidate (614M total exceeds the 400M ceiling). If it
clears the incumbent, R8-H89 distills the ensemble into one 307M student.

The incumbent's per-subset numbers are the recorded R8-H77 constants (it is
identical and frozen across every run); only the two students are scored here.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      uv run python experiments/grounding-semantic/R8-H88_ensemble_arena.py
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

H84 = HERE.parent.parent / "models" / "R8-H84-mmbert-vitaminc"
H79 = HERE.parent.parent / "models" / "R8-H79-mmbert-dann"
OUT = HERE / "R8-H88_result.json"

# Recorded incumbent constants from the R8-H77 runs (frozen model, frozen data).
LETTUCE = {
    "covidqa": 0.7355,
    "delucionqa": 0.7929,
    "emanual": 0.5999,
    "expertqa": 0.6503,
    "finqa": 0.7170,
    "hagrid": 0.5992,
    "hotpotqa": 0.5976,
    "pubmedqa": 0.5162,
    "tatqa": 0.6156,
    "techqa": 0.6363,
}


def main():
    subs = ARENA.load_subsets()
    print(f"RAGBench: {len(subs)} subsets, {sum(len(v[2]) for v in subs.values())} responses\n")

    rows = {}
    for sub, (claims, chunks, y) in subs.items():
        s84 = ARENA.score_student(str(H84), claims, chunks)
        s79 = ARENA.score_student(str(H79), claims, chunks)
        s = (s84 + s79) / 2.0
        auc, f1, _ = ARENA.M59.auc_and_f1(y, s)
        a84, _, _ = ARENA.M59.auc_and_f1(y, s84)
        a79, _, _ = ARENA.M59.auc_and_f1(y, s79)
        rows[sub] = {
            "n": len(y),
            "ens_auc": round(auc, 4),
            "ens_f1": round(f1, 4),
            "h84_auc": round(a84, 4),
            "h79_auc": round(a79, 4),
            "lettuce_auc": LETTUCE[sub],
        }
        print(
            f"  {sub:14s} n={len(y):>4} ens {auc:.4f}  h84 {a84:.4f}  h79 {a79:.4f}  "
            f"lettuce {LETTUCE[sub]:.4f}  delta {auc - LETTUCE[sub]:+.4f}",
            flush=True,
        )

    ens = float(np.mean([r["ens_auc"] for r in rows.values()]))
    let = float(np.mean([r["lettuce_auc"] for r in rows.values()]))
    wins = sum(r["ens_auc"] > r["lettuce_auc"] for r in rows.values())

    print("\n" + "=" * 92)
    print("R8-H88 RESULT - unweighted ERM+DANN ensemble on the blind arena")
    print("=" * 92)
    print(f"  ensemble mean AUC {ens:.4f}")
    print(f"  lettucedect-v2    {let:.4f}")
    print(f"  delta {ens - let:+.4f}   subsets won {wins}/{len(rows)}")
    OUT.write_text(
        json.dumps(
            {"per_subset": rows, "mean_ensemble": ens, "mean_lettuce": let, "wins": wins},
            indent=2,
        )
    )
    print(f"\n  results -> {OUT}")


if __name__ == "__main__":
    main()

"""R8-H97 - three-member decomposed-min ensemble (ERM + DANN + two-head).

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 8).

Unweighted mean of the three models' per-sentence scores, min over sentences,
members frozen (H84, H79 v1, H73), identical gate, one shot. Bar: blind mean
>= 0.6920 with >= 7/10 subsets and none below chance; < 0.6893 confirms the
weaker member as dilution and the two-member ensemble stands.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      uv run python experiments/grounding-semantic/R8-H97_ensemble3.py
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
H92 = _mod("h92", "R8-H92_decomposed_arena.py")

ROOT = HERE.parent.parent
MODELS = {
    "h84": ROOT / "models" / "R8-H84-mmbert-vitaminc",
    "h79": ROOT / "models" / "R8-H79-mmbert-dann",
    "h73": ROOT / "models" / "R8-H73-mmbert-twohead",
}
OUT = HERE / "R8-H97_result.json"


def main():
    subs = ARENA.load_subsets()
    rows = {}
    for sub, (claims, chunks, y) in subs.items():
        sent_lists = [H92.sentences(c) for c in claims]
        flat_s, flat_k, owner = [], [], []
        for i, (sl, ks) in enumerate(zip(sent_lists, chunks, strict=True)):
            for s in sl:
                flat_s.append(s)
                flat_k.append(ks)
                owner.append(i)
        owner = np.array(owner)
        per = [ARENA.score_student(str(m), flat_s, flat_k) for m in MODELS.values()]
        ens = np.mean(per, axis=0)
        resp = np.array([ens[owner == i].min() for i in range(len(y))])
        auc, f1, _ = ARENA.M59.auc_and_f1(y, resp)
        rows[sub] = {"n": len(y), "auc": round(auc, 4), "lettuce_auc": H92.LETTUCE[sub]}
        print(
            f"  {sub:14s} ens3 {auc:.4f}  lettuce {H92.LETTUCE[sub]:.4f}  "
            f"delta {auc - H92.LETTUCE[sub]:+.4f}",
            flush=True,
        )

    mean = float(np.mean([r["auc"] for r in rows.values()]))
    let = float(np.mean([r["lettuce_auc"] for r in rows.values()]))
    wins = sum(r["auc"] > r["lettuce_auc"] for r in rows.values())
    print(f"\nR8-H97 RESULT: ens3 decomposed-min mean {mean:.4f}  (bar 0.6920; H92 0.6893)")
    print(f"lettuce {let:.4f}  delta {mean - let:+.4f}  wins {wins}/10")
    OUT.write_text(json.dumps({"per_subset": rows, "mean": mean, "wins": wins}, indent=2))
    print(f"results -> {OUT}")


if __name__ == "__main__":
    main()

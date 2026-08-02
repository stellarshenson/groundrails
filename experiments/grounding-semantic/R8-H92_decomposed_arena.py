"""R8-H92 - decomposed scoring: min over sentences of max over chunks.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 8).

RAGBench `adherence_score` is a response-level AND over the response's claims.
Our scorer feeds the WHOLE response as one claim, so a single hallucinated
sentence hides among supported ones, diluted inside one CLS score - while the
incumbent's token-level 1 - max(halluc) is a fine-grained min-aggregation.

Formula only: frozen checkpoints (H84 ERM, H79 v1 DANN), identical data and
metric; the only change is the aggregation. Deterministic sentence split
(terminal punctuation, min 25 chars, cap 12 sentences, whole-response fallback
below 2). Primary read: MIN over sentences on the H84+H79 ensemble (bar:
blind mean >= 0.667, no subset below chance). Per-member and mean-aggregation
numbers are recorded as diagnostics, not bar-eligible.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      uv run python experiments/grounding-semantic/R8-H92_decomposed_arena.py
"""

import importlib.util
import json
import pathlib
import re

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
OUT = HERE / "R8-H92_result.json"

MIN_SENT_CHARS = 25
MAX_SENTS = 12

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

_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentences(text):
    parts = [s.strip() for s in _SPLIT.split(text)]
    parts = [s for s in parts if len(s) >= MIN_SENT_CHARS][:MAX_SENTS]
    return parts if len(parts) >= 2 else [text]


def main():
    subs = ARENA.load_subsets()
    print(f"RAGBench: {len(subs)} subsets, {sum(len(v[2]) for v in subs.values())} responses\n")

    rows = {}
    for sub, (claims, chunks, y) in subs.items():
        sent_lists = [sentences(c) for c in claims]
        flat_sents, flat_chunks, owner = [], [], []
        for i, (sl, ks) in enumerate(zip(sent_lists, chunks, strict=True)):
            for s in sl:
                flat_sents.append(s)
                flat_chunks.append(ks)
                owner.append(i)
        owner = np.array(owner)
        n_sent = len(flat_sents)

        # Per-sentence max-over-chunks score, per member (the arena scorer's
        # own per-claim aggregation, applied at sentence granularity).
        s84 = ARENA.score_student(str(H84), flat_sents, flat_chunks)
        s79 = ARENA.score_student(str(H79), flat_sents, flat_chunks)
        ens = (s84 + s79) / 2.0

        r = {"n": len(y), "n_sent": n_sent, "lettuce_auc": LETTUCE[sub]}
        for name, ps in (("h84", s84), ("h79", s79), ("ens", ens)):
            for agg, fn in (("min", np.min), ("mean", np.mean)):
                resp = np.array([fn(ps[owner == i]) for i in range(len(y))])
                auc, f1, _ = ARENA.M59.auc_and_f1(y, resp)
                r[f"{name}_{agg}_auc"] = round(auc, 4)
                if name == "ens" and agg == "min":
                    r["ens_min_f1"] = round(f1, 4)
        rows[sub] = r
        print(
            f"  {sub:14s} n={len(y):>4} sents={n_sent:>5}  "
            f"ens_min {r['ens_min_auc']:.4f}  ens_mean {r['ens_mean_auc']:.4f}  "
            f"h84_min {r['h84_min_auc']:.4f}  h79_min {r['h79_min_auc']:.4f}  "
            f"lettuce {LETTUCE[sub]:.4f}  delta {r['ens_min_auc'] - LETTUCE[sub]:+.4f}",
            flush=True,
        )

    means = {
        k: float(np.mean([r[f"{k}_auc"] for r in rows.values()]))
        for k in ("ens_min", "ens_mean", "h84_min", "h84_mean", "h79_min", "h79_mean")
    }
    let = float(np.mean([r["lettuce_auc"] for r in rows.values()]))
    wins = sum(r["ens_min_auc"] > r["lettuce_auc"] for r in rows.values())

    print("\n" + "=" * 92)
    print("R8-H92 RESULT - decomposed scoring on the blind arena")
    print("=" * 92)
    print(f"  PRIMARY ens min-over-sentences  {means['ens_min']:.4f}  (bar 0.6670)")
    print(f"  lettucedect-v2                  {let:.4f}")
    print(f"  delta {means['ens_min'] - let:+.4f}   subsets won {wins}/{len(rows)}")
    print("\n  diagnostics (not bar-eligible):")
    for k in ("ens_mean", "h84_min", "h84_mean", "h79_min", "h79_mean"):
        print(f"    {k:10s} {means[k]:.4f}")
    OUT.write_text(
        json.dumps(
            {"per_subset": rows, "means": means, "mean_lettuce": let, "wins": wins}, indent=2
        )
    )
    print(f"\n  results -> {OUT}")


if __name__ == "__main__":
    main()

"""R8-H94 - soft aggregation between min and mean, tuned on RAGTruth only.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 8).

R8-H92 proved min-over-sentences wins 8 arena subsets while mean wins
delucionqa (+0.0998) and covidqa. A single response-level aggregator with one
shape parameter can interpolate - IF its shape is chosen without touching the
arena. Tuning corpus: RAGTruth test responses (EN + 7 translations), the same
harness that serves the in-domain bars. RAGBench is untouched until the single
frozen-shape arena shot.

Search space, fixed pre-run: quantile q in {0, .05, .1, .15, .2, .25, .35, .5};
soft-min temperature tau in {2, 4, 8, 16, 32}; blend alpha*min + (1-alpha)*mean
with alpha in {.5, .65, .8, .9, 1.0}. Argmax by the mean of the two members'
(H84, H79) mean-over-corpora RAGTruth AUCs.

Stages:
  tune  (default) - cache per-sentence RAGTruth scores (GPU), sweep on the
                    cache (CPU), freeze the winner to R8-H94_winner.json
  arena           - apply the frozen winner once to the blind arena
                    (bar: ens >= 0.6893, delucionqa >= +0.02 over its min read)

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      uv run python experiments/grounding-semantic/R8-H94_soft_aggregation.py [--stage arena]
"""

import argparse
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
M60 = _mod("m60", "R7-H60_multilingual_parallel.py")

H84 = HERE.parent.parent / "models" / "R8-H84-mmbert-vitaminc"
H79 = HERE.parent.parent / "models" / "R8-H79-mmbert-dann"
CACHE = HERE / "R8-H94_ragtruth_cache.npz"
WINNER = HERE / "R8-H94_winner.json"
OUT = HERE / "R8-H94_result.json"

QUANTILES = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5)
TAUS = (2.0, 4.0, 8.0, 16.0, 32.0)
ALPHAS = (0.5, 0.65, 0.8, 0.9, 1.0)


def aggregate(kind, param, scores):
    """One response-level scalar from that response's per-sentence scores."""
    if kind == "quantile":
        return float(np.quantile(scores, param))
    if kind == "softmin":
        return float(-np.log(np.mean(np.exp(-param * scores))) / param)
    if kind == "blend":
        return float(param * scores.min() + (1.0 - param) * scores.mean())
    raise ValueError(kind)


def candidates():
    for q in QUANTILES:
        yield ("quantile", q)
    for t in TAUS:
        yield ("softmin", t)
    for a in ALPHAS:
        yield ("blend", a)


def sentence_scores(model_path, responses, chunk_lists):
    """Per-sentence max-over-chunks scores plus the owner index per sentence."""
    sent_lists = [H92.sentences(r) for r in responses]
    flat_s, flat_k, owner = [], [], []
    for i, (sl, ks) in enumerate(zip(sent_lists, chunk_lists, strict=True)):
        for s in sl:
            flat_s.append(s)
            flat_k.append(ks)
            owner.append(i)
    scores = ARENA.score_student(str(model_path), flat_s, flat_k)
    return scores, np.array(owner)


def stage_tune():
    if not CACHE.exists():
        print("stage 1 - caching per-sentence RAGTruth scores (GPU)...", flush=True)
        blobs = {}
        corpora = [("en", *M60.load_english())]
        for lg in ("de", "fr", "es", "it", "pl", "hu", "cn"):
            corpora.append((lg, *M60.load_translated(lg)))
        for name, cl, ctx, y in corpora:
            chunks = [ARENA.M59.top_chunks(c, ARENA.M59.CFG.semantic_top_k) for c in ctx]
            s84, owner = sentence_scores(H84, cl, chunks)
            s79, _ = sentence_scores(H79, cl, chunks)
            blobs[f"{name}_s84"] = s84
            blobs[f"{name}_s79"] = s79
            blobs[f"{name}_owner"] = owner
            blobs[f"{name}_y"] = np.asarray(y)
            print(f"  cached {name}: {len(y)} responses, {len(owner)} sentences", flush=True)
        np.savez_compressed(CACHE, **blobs)
        print(f"  cache -> {CACHE}\n", flush=True)

    print("stage 2 - aggregator sweep on the cache (CPU)...", flush=True)
    z = np.load(CACHE)
    langs = ("en", "de", "fr", "es", "it", "pl", "hu", "cn")
    table = []
    for kind, param in candidates():
        per_model = []
        for member in ("s84", "s79"):
            aucs = []
            for lg in langs:
                s, owner, y = z[f"{lg}_{member}"], z[f"{lg}_owner"], z[f"{lg}_y"]
                resp = np.array([aggregate(kind, param, s[owner == i]) for i in range(len(y))])
                auc, _, _ = ARENA.M59.auc_and_f1(y, resp)
                aucs.append(auc)
            per_model.append(float(np.mean(aucs)))
        score = float(np.mean(per_model))
        table.append((kind, param, per_model[0], per_model[1], score))
        print(
            f"  {kind:9s} {param:>5}  h84 {per_model[0]:.4f}  h79 {per_model[1]:.4f}  "
            f"mean {score:.4f}",
            flush=True,
        )

    best = max(table, key=lambda r: r[4])
    print(f"\n  WINNER: {best[0]} {best[1]}  ragtruth mean {best[4]:.4f}")
    WINNER.write_text(
        json.dumps(
            {"kind": best[0], "param": best[1], "ragtruth_h84": best[2], "ragtruth_h79": best[3]},
            indent=2,
        )
    )
    print(f"  frozen -> {WINNER}")


def stage_arena():
    w = json.loads(WINNER.read_text())
    kind, param = w["kind"], w["param"]
    print(f"arena shot with frozen aggregator: {kind} {param}\n", flush=True)

    subs = ARENA.load_subsets()
    rows = {}
    for sub, (claims, chunks, y) in subs.items():
        s84, owner = sentence_scores(H84, claims, chunks)
        s79, _ = sentence_scores(H79, claims, chunks)
        ens = (s84 + s79) / 2.0
        resp = np.array([aggregate(kind, param, ens[owner == i]) for i in range(len(y))])
        auc, f1, _ = ARENA.M59.auc_and_f1(y, resp)
        rows[sub] = {
            "n": len(y),
            "ens_soft_auc": round(auc, 4),
            "ens_soft_f1": round(f1, 4),
            "lettuce_auc": H92.LETTUCE[sub],
        }
        print(
            f"  {sub:14s} n={len(y):>4} ens_{kind} {auc:.4f}  lettuce {H92.LETTUCE[sub]:.4f}  "
            f"delta {auc - H92.LETTUCE[sub]:+.4f}",
            flush=True,
        )

    mean = float(np.mean([r["ens_soft_auc"] for r in rows.values()]))
    let = float(np.mean([r["lettuce_auc"] for r in rows.values()]))
    wins = sum(r["ens_soft_auc"] > r["lettuce_auc"] for r in rows.values())
    print("\n" + "=" * 92)
    print(f"R8-H94 RESULT - frozen {kind} {param} on the blind arena")
    print("=" * 92)
    print(f"  ensemble {kind}  mean AUC {mean:.4f}  (bar 0.6893)")
    print(f"  lettucedect-v2  {let:.4f}")
    print(f"  delta {mean - let:+.4f}   subsets won {wins}/{len(rows)}")
    OUT.write_text(
        json.dumps(
            {
                "aggregator": w,
                "per_subset": rows,
                "mean_ens": mean,
                "mean_lettuce": let,
                "wins": wins,
            },
            indent=2,
        )
    )
    print(f"\n  results -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("tune", "arena"), default="tune")
    args = ap.parse_args()
    if args.stage == "tune":
        stage_tune()
    else:
        stage_arena()


if __name__ == "__main__":
    main()

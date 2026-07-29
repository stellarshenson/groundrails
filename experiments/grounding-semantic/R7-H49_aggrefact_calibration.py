"""R7-H49 - external calibration of the shipped cascade on LLM-AggreFact.

Every quality number this project holds was measured on its own private gold,
labelled by its own dual-judge pipeline. macro-F1 0.824 at base rate 0.649
converts to roughly 0.82-0.85 balanced accuracy, which would sit ABOVE the
LLM-AggreFact leaderboard top of 77.4 (Bespoke-MiniCheck-7B). The far more
likely explanation is that our gold is easier or more loosely labelled than the
public benchmark, and that has never been tested.

LLM-AggreFact is the right instrument because it needs no reshaping: each row is
a document plus a claim plus a binary supported label - the task we already ship.
Its published non-LLM encoder ladder is the reference we care about:

    plain NLI          61.4
    AlignScore         70.4
    MiniCheck-DeBERTa  72.6
    FactCG-DeBERTa-L   75.6
    top (7B)           77.4

Metric is the leaderboard's own: per-subset balanced accuracy, then an
UNWEIGHTED mean across subsets, so a large subset cannot dominate.

The cascade runs exactly as deployed - `recursive_chunk` at cfg.chunk_max_chars,
bi-encoder ranking, top cfg.semantic_top_k chunks, reranker and NLI scored
max-over-chunks - with no threshold refitting. Refitting on the benchmark would
answer a different and much less interesting question.

Run:  uv run python experiments/grounding-semantic/R7-H49_aggrefact_calibration.py
"""

import json
import pathlib
import time

from datasets import load_dataset
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from groundrails import semantic_ov, settings
from groundrails.chunking import recursive_chunk
from groundrails.config import load_document_processing_config

settings.mark_ready()
CFG = load_document_processing_config()

HERE = pathlib.Path(__file__).parent
CKPT = HERE / "R7-H49_aggrefact_scores.jsonl"
PER_SUBSET = 250  # stratified; the leaderboard averages subsets unweighted anyway
LADDER = {
    "plain NLI": 61.4,
    "AlignScore": 70.4,
    "MiniCheck-DeBERTa": 72.6,
    "FactCG-DeBERTa-L": 75.6,
    "Bespoke-MiniCheck-7B (top)": 77.4,
}


def main():
    ds = load_dataset("lytang/LLM-AggreFact", split="test")
    print(f"LLM-AggreFact test: {len(ds)} rows, subsets {sorted(set(ds['dataset']))}", flush=True)

    rows = []
    for name in sorted(set(ds["dataset"])):
        sub = ds.filter(lambda r, n=name: r["dataset"] == n)
        take = min(PER_SUBSET, len(sub))
        rows += list(sub.shuffle(seed=0).select(range(take)))
    print(
        f"stratified sample: {len(rows)} rows across {len(set(ds['dataset']))} subsets\n",
        flush=True,
    )

    done = {}
    if CKPT.exists():
        for line in CKPT.read_text().splitlines():
            r = json.loads(line)
            done[r["key"]] = r
        print(f"resuming: {len(done)} rows already scored", flush=True)

    cascade = semantic_ov.SemanticCascade()
    cascade._load()

    fh = CKPT.open("a")
    t0 = time.time()
    for i, r in enumerate(rows):
        key = f"{r['dataset']}|{i}"
        if key in done:
            continue
        chunks = [c.text for c in (recursive_chunk(r["doc"], max_chars=CFG.chunk_max_chars) or [])]
        if not chunks:
            chunks = [r["doc"][: CFG.chunk_max_chars]]
        if len(chunks) > CFG.semantic_top_k:
            qv = cascade._embed([r["claim"]])
            cv = cascade._embed(chunks)
            order = np.argsort(-(cv @ qv.T).ravel())[: CFG.semantic_top_k]
            chunks = [chunks[j] for j in order]
        s = cascade.score(r["claim"], chunks)
        rec = {
            "key": key,
            "dataset": r["dataset"],
            "label": int(r["label"]),
            "rerank": float(s.rerank_max),
            "entail": float(s.entail_max),
        }
        done[key] = rec
        fh.write(json.dumps(rec) + "\n")
        if i % 100 == 0:
            fh.flush()
            el = time.time() - t0
            print(
                f"  {i}/{len(rows)}  {el:.0f}s  ({el / max(i, 1) * 1000:.0f} ms/claim)", flush=True
            )
    fh.close()

    recs = [done[f"{r['dataset']}|{i}"] for i, r in enumerate(rows)]
    by_sub = {}
    for rec in recs:
        by_sub.setdefault(rec["dataset"], []).append(rec)

    print("\n" + "=" * 92)
    print("R7-H49 RESULT - shipped cascade on LLM-AggreFact (no threshold refitting)")
    print("=" * 92)
    print(f"{'subset':28s} {'n':>5} {'base':>6} {'bal-acc(rerank)':>16} {'bal-acc(NLI)':>13}")
    accs_r, accs_n = [], []
    for name in sorted(by_sub):
        v = by_sub[name]
        y = np.array([x["label"] for x in v])
        # Operating point = the subset's own median score, i.e. no fitted
        # threshold transferred from our gold; this measures the RANKING the
        # cascade produces, which is the property the ladder compares.
        for key, store in (("rerank", accs_r), ("entail", accs_n)):
            s = np.array([x[key] for x in v])
            store.append(balanced_accuracy_score(y, (s >= np.median(s)).astype(int)))
        print(
            f"{name:28s} {len(v):>5} {y.mean():>6.3f} {accs_r[-1] * 100:>15.1f}% "
            f"{accs_n[-1] * 100:>12.1f}%"
        )

    mr, mn = float(np.mean(accs_r)) * 100, float(np.mean(accs_n)) * 100
    print(f"\n  {'UNWEIGHTED MEAN - reranker':32s} {mr:.1f}%")
    print(f"  {'UNWEIGHTED MEAN - NLI':32s} {mn:.1f}%")
    print("\n  published ladder for reference:")
    for k, v in LADDER.items():
        best = max(mr, mn)
        mark = "  <- we are here" if abs(best - v) < 2.0 else ""
        print(f"    {k:30s} {v:.1f}%{mark}")
    print(f"\n  our best {max(mr, mn):.1f}%   vs our private-gold balanced accuracy ~82-85%")
    gap = 83.5 - max(mr, mn)
    print(f"  private-vs-public gap: {gap:+.1f} points")
    if gap > 8:
        print("  -> the private gold is materially easier or more loosely labelled;")
        print("     0.824 is NOT a SOTA-comparable number and must not be quoted as one")
    print(f"\n  scores -> {CKPT}")


if __name__ == "__main__":
    main()

"""Round 8 scoring substrate - per-example scores for every model on every corpus.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 8).

Round 7 saved only aggregate AUC and F1, which made R8-H63 (rank normalisation)
and R8-H64 (ensemble) impossible to answer without paying for inference again.
That was a design error: aggregates cannot be re-analysed, per-example scores
can. This writes the scores once so every downstream hypothesis in this round is
arithmetic on a file rather than another GPU pass.

Emits one tidy parquet with columns:

    corpus | lang | idx | label | cascade | lettuce

covering:
  - our private gold, 159 held-out TRACES only (never the training traces)
  - RAGTruth EN test
  - RAGTruth x7 translations test
  - RAGTruth DE human-verified 300, the R8-H61 gate

Both scorers are imported from R7-H59 rather than reimplemented, so a change to
one definition cannot silently diverge from the numbers already recorded.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      uv run python experiments/grounding-semantic/R8_score_substrate.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import importlib.util
import pathlib

from datasets import load_dataset
import numpy as np
import polars as pl
import torch

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent.parent / "data" / "external" / "datasets"
OUT = HERE / "private-rag-forensics" / "R8_scores.parquet"
PAIRS = HERE / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"
GOLD = HERE / "private-rag-forensics" / "gold" / "golden_grounding_evidence_verified.parquet"
N_PER_LANG = 600
MANUAL_DE = "KRLabsOrg/ragtruth-de-translated-manual-300"


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M59 = _mod("m59", "R7-H59_cross_domain_matrix.py")
M60 = _mod("m60", "R7-H60_multilingual_parallel.py")


def our_gold():
    """Held-out TRACES only - the same 159 used throughout round 7."""
    df = pl.read_parquet(PAIRS)
    g = pl.read_parquet(GOLD).with_row_index("owner")
    df = df.join(g.select(["owner", "trace_id"]), on="owner", how="left")
    traces = np.array(sorted(set(df["trace_id"].to_list())))
    rng = np.random.default_rng(0)
    rng.shuffle(traces)
    test = set(traces[: int(len(traces) * 0.25)].tolist())
    t = df.filter(pl.col("trace_id").is_in(list(test)))
    # One row per CLAIM, carrying ALL of its chunks as the evidence list.
    #
    # NOT `[:semantic_top_k]`. The first version sliced the first three chunks in
    # dataframe order, which is arbitrary evidence rather than retrieved
    # evidence, and it read the cascade at 0.6739 on gold against the 0.8619
    # R7-H50 measured by taking max over ALL of a claim's chunks. Truncating
    # here silently changed the task from "is it supported anywhere in the
    # source" to "is it supported in three arbitrary passages".
    claims, chunk_lists, labels, langs, owners = [], [], [], [], []
    for owner, grp in t.group_by("owner"):
        o = owner[0] if isinstance(owner, tuple) else owner
        claims.append(grp["claim"][0])
        chunk_lists.append(grp["chunk"].to_list())
        labels.append(int(grp["label"][0]))
        langs.append(str(grp["lang"][0])[:2])
        owners.append(int(o))
    return claims, chunk_lists, np.array(labels), langs, owners


def manual_de():
    """R8-H61's gate - the only human-VERIFIED translation rows that exist."""
    ds = load_dataset(MANUAL_DE)
    split = next(iter(ds.values())) if len(ds) == 1 else ds.get("test", next(iter(ds.values())))
    df = pl.from_arrow(split.data.table)
    ans = "answer" if "answer" in df.columns else "output"
    ctx = "prompt" if "prompt" in df.columns else "context"
    lab = "labels" if "labels" in df.columns else "hallucination_labels"
    df = df.with_columns((pl.col(lab).list.len() == 0).cast(pl.Int8).alias("label"))
    return df[ans].to_list(), df[ctx].to_list(), df["label"].to_numpy()


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    frames = []

    jobs = [("gold", "mixed", our_gold)]
    jobs.append(("ragtruth", "en", lambda: (*M60.load_english(),)))
    for lg in ("de", "fr", "es", "it", "pl", "hu", "cn"):
        jobs.append((f"ragtruth-{lg}", lg, (lambda lg=lg: M60.load_translated(lg))))
    jobs.append(("ragtruth-de-manual", "de-verified", manual_de))

    for corpus, lang, loader in jobs:
        try:
            loaded = loader()
        except Exception as e:  # noqa: BLE001 - a missing corpus is a result, not a crash
            print(f"  SKIP {corpus}: {type(e).__name__}: {str(e)[:100]}", flush=True)
            continue
        if corpus == "gold":
            claims, chunk_lists, y, langs, _ = loaded
        else:
            claims, contexts, y = loaded
            chunk_lists = [M59.top_chunks(c, M59.CFG.semantic_top_k) for c in contexts]
            langs = [lang] * len(y)
        cas = M59.score_reranker(claims, chunk_lists)
        let = M59.score_lettuce(claims, chunk_lists)
        frames.append(
            pl.DataFrame(
                {
                    "corpus": [corpus] * len(y),
                    "lang": langs,
                    "idx": list(range(len(y))),
                    "label": y.astype("int8"),
                    "cascade": cas.astype("float32"),
                    "lettuce": let.astype("float32"),
                }
            )
        )
        print(f"  {corpus:22s} n={len(y):>5} base {y.mean():.3f}  scored", flush=True)

    out = pl.concat(frames)
    out.write_parquet(OUT)
    print(f"\nper-example scores -> {OUT}  ({len(out)} rows, {out['corpus'].n_unique()} corpora)")
    print("R8-H61 / H63 / H64 are now re-analyses of this file, no further inference needed")


if __name__ == "__main__":
    main()

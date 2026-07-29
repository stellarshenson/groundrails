"""R7-H60 - multilingual degradation on parallel data.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 7).

Our own gold cannot answer this. Its non-English slices are 26-44 claims at base
rates up to 0.973 - roughly one negative in the Norwegian slice - so R7-H57's
per-language AUCs there were noise with a decimal point, and I said so rather
than reporting them as findings.

RAGTruth's seven translations fix that by construction. They are PARALLEL: the
same 2,700 test responses, the same evidence, the same human span annotations,
rendered into 7 languages. Domain, difficulty and class balance are therefore
held constant and the only variable that moves is language. That is the
controlled experiment our own data cannot support at any sample size.

Two models, and they are NOT symmetric - stated plainly because it changes how
the table reads:

  - our cascade's reranker (`bge-reranker-v2-m3`, XLM-R backbone) is ZERO-SHOT
    on every language here, including English
  - `lettucedect-v2-mmbert-base` was TRAINED on the train splits of these very
    translations, so its numbers are in-domain and near home turf

So the head-to-head is unfair to us by design. The number worth reading is our
cascade's DEGRADATION CURVE across languages - English against everything else -
which is a within-model comparison and immune to that asymmetry.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      uv run python experiments/grounding-semantic/R7-H60_multilingual_parallel.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import importlib.util
import io
import json
import pathlib
import zipfile

import numpy as np
import polars as pl
import torch

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent.parent / "data" / "external" / "datasets"
OUT = HERE / "R7-H60_multilingual.json"
N_PER_LANG = 600


def _matrix_module():
    """Reuse R7-H59's scorers verbatim - one definition of each, not two."""
    spec = importlib.util.spec_from_file_location("m59", HERE / "R7-H59_cross_domain_matrix.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _matrix_module()


def load_english():
    """Original RAGTruth: separate `query` / `context` / `output` columns."""
    z = zipfile.ZipFile(DATA / "dataset-ragtruth.zip")
    n = next(x for x in z.namelist() if x.endswith("__test.parquet"))
    df = pl.read_parquet(io.BytesIO(z.read(n)))
    df = df.with_columns(
        (
            (pl.col("hallucination_labels_processed").struct.field("evident_conflict") == 0)
            & (pl.col("hallucination_labels_processed").struct.field("baseless_info") == 0)
        )
        .cast(pl.Int8)
        .alias("label")
    )
    df = df.filter(
        (pl.col("context").str.len_chars() > 50) & (pl.col("output").str.len_chars() > 20)
    )
    df = df.sample(min(N_PER_LANG, len(df)), seed=0)
    return df["output"].to_list(), df["context"].to_list(), df["label"].to_numpy()


def load_translated(lang):
    """Translations merge query+context into `prompt`, and `labels` IS a list here."""
    z = zipfile.ZipFile(DATA / "dataset-ragtruth-translated.zip")
    n = next(x for x in z.namelist() if f"ragtruth-{lang}-" in x and x.endswith("__test.parquet"))
    df = pl.read_parquet(io.BytesIO(z.read(n)))
    df = df.with_columns((pl.col("labels").list.len() == 0).cast(pl.Int8).alias("label"))
    df = df.filter(
        (pl.col("prompt").str.len_chars() > 50) & (pl.col("answer").str.len_chars() > 20)
    )
    df = df.sample(min(N_PER_LANG, len(df)), seed=0)
    return df["answer"].to_list(), df["prompt"].to_list(), df["label"].to_numpy()


LANGS = [("en", load_english)] + [
    (lg, (lambda lg=lg: load_translated(lg))) for lg in ("de", "fr", "es", "it", "pl", "hu", "cn")
]


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print("parallel corpus: same responses, same evidence, same labels, 8 languages\n", flush=True)

    results = {}
    for lang, loader in LANGS:
        claims, contexts, y = loader()
        chunks = [M.top_chunks(c, M.CFG.semantic_top_k) for c in contexts]
        row = {"n": len(y), "grounded_rate": round(float(y.mean()), 4)}
        for name, fn in (("cascade", M.score_reranker), ("lettuce", M.score_lettuce)):
            s = fn(claims, chunks)
            auc, f1, _ = M.auc_and_f1(y, s)
            row[f"{name}_auc"] = round(auc, 4)
            row[f"{name}_f1"] = round(f1, 4)
        results[lang] = row
        print(
            f"  {lang}  n={row['n']:>4} base {row['grounded_rate']:.3f}  "
            f"cascade AUC {row['cascade_auc']:.4f} F1 {row['cascade_f1']:.4f}  |  "
            f"lettuce AUC {row['lettuce_auc']:.4f} F1 {row['lettuce_f1']:.4f}",
            flush=True,
        )

    en = results["en"]
    print("\n" + "=" * 96)
    print("R7-H60 RESULT - multilingual degradation, parallel data")
    print("=" * 96)
    print(
        f"{'lang':6s} {'n':>5} {'base':>6} {'cascade AUC':>13} {'vs EN':>8} "
        f"{'lettuce AUC':>13} {'vs EN':>8}"
    )
    for lang, r in results.items():
        print(
            f"{lang:6s} {r['n']:>5} {r['grounded_rate']:>6.3f} "
            f"{r['cascade_auc']:>13.4f} {r['cascade_auc'] - en['cascade_auc']:>+8.4f} "
            f"{r['lettuce_auc']:>13.4f} {r['lettuce_auc'] - en['lettuce_auc']:>+8.4f}"
        )

    non_en = [r for lg, r in results.items() if lg != "en"]
    c_drop = en["cascade_auc"] - float(np.mean([r["cascade_auc"] for r in non_en]))
    l_drop = en["lettuce_auc"] - float(np.mean([r["lettuce_auc"] for r in non_en]))
    print(f"\n  our cascade (zero-shot everywhere) mean non-EN drop : {c_drop:+.4f}")
    print(f"  lettucedect (TRAINED on these translations)         : {l_drop:+.4f}")
    print("\n  the head-to-head is unfair by design - lettucedect trained on this corpus.")
    print("  read the WITHIN-model degradation columns, not the between-model gap.")
    print("  our cascade holding across languages would mean our multilingual weakness is")
    print("  a gold-set measurement problem, not a model problem.")
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n  results -> {OUT}")


if __name__ == "__main__":
    main()

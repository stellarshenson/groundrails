"""R7-H59 - the cross-domain transfer matrix: our cascade and a public verifier, on both corpora.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 7).

R7-H57 measured one cell: a public-trained verifier on OUR gold, AUC 0.7095
against our cascade's 0.8619. One cell cannot distinguish three explanations -
that public data does not transfer, that our gold is unusually easy, or that our
cascade is overfitted to our own domain. The full matrix separates them:

                        our private gold        RAGTruth (public)
    our cascade         0.8619  (R7-H50)        <- this run
    lettucedect-v2      0.7095  (R7-H57)        <- this run, its HOME turf

  - if our cascade also drops on RAGTruth, both models are domain-bound and the
    deficit is symmetric - nobody transfers, and public data is a different task
  - if our cascade HOLDS on RAGTruth while theirs drops on ours, our cascade is
    the more general model and their training data is the narrower one
  - if lettucedect scores far higher on RAGTruth than on ours, that gap is the
    size of the domain difference, measured rather than argued
  - and if our cascade scores near its own 0.8619 on RAGTruth, our gold is not
    unusually easy after all - which is the R7-H49 question, answered from a
    different direction while that one stays blocked on Hub auth

Both metrics are reported. AUC is threshold-free. Macro-F1 needs an operating
point, so the test set is split in half: the threshold is fitted on half A and
reported on half B, for every model and corpus alike. Fitting the threshold on
the evaluation half is what inflated the first version of R7-H50 and is not
repeated here.

RAGTruth's response-level label is `hallucination_labels` - an empty list means
the response is fully grounded in its context. That is the same binary our task
uses, which is why this corpus needs no reshaping.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      uv run python experiments/grounding-semantic/R7-H59_cross_domain_matrix.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import io
import json
import pathlib
import zipfile

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, roc_auc_score
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from groundrails.chunking import recursive_chunk
from groundrails.config import load_document_processing_config

CFG = load_document_processing_config()
HERE = pathlib.Path(__file__).parent
RAGTRUTH = HERE.parent.parent / "data" / "external" / "datasets" / "dataset-ragtruth.zip"
OUT = HERE / "R7-H59_matrix.json"

RERANKER = "BAAI/bge-reranker-v2-m3"
LETTUCE = "KRLabsOrg/lettucedect-v2-mmbert-base"
N_SAMPLE = 1200  # of RAGTruth's 2,700 test rows
BATCH = 16


def load_ragtruth():
    z = zipfile.ZipFile(RAGTRUTH)
    name = next(n for n in z.namelist() if n.endswith("__test.parquet"))
    df = pl.read_parquet(io.BytesIO(z.read(name)))
    # Response-level binary: no annotated hallucination -> grounded.
    # `hallucination_labels` is a JSON STRING ('[]' or '[{...}]'), not a list, so
    # the struct column is used instead - it carries the two annotated error
    # types directly and needs no parsing.
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
    return df.sample(min(N_SAMPLE, len(df)), seed=0)


def top_chunks(text, k):
    ch = [c.text for c in (recursive_chunk(text, max_chars=CFG.chunk_max_chars) or [])]
    return ch[:k] if ch else [text[: CFG.chunk_max_chars]]


@torch.inference_mode()
def score_reranker(claims, chunk_lists):
    """Our cascade's best single signal, max-over-chunks - the same serving rule."""
    tok = AutoTokenizer.from_pretrained(RERANKER)
    model = (
        AutoModelForSequenceClassification.from_pretrained(RERANKER, dtype=torch.float16)
        .cuda()
        .eval()
    )
    flat_c, flat_k, owner = [], [], []
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        for k in ks:
            flat_c.append(c)
            flat_k.append(k)
            owner.append(i)
    out = np.zeros(len(flat_c), dtype=np.float32)
    for i in range(0, len(flat_c), 64):
        enc = tok(
            flat_c[i : i + 64],
            flat_k[i : i + 64],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda")
        out[i : i + 64] = torch.sigmoid(model(**enc).logits.float().squeeze(-1)).cpu().numpy()
    owner = np.array(owner)
    agg = np.array([out[owner == i].max() for i in range(len(claims))])
    del model
    torch.cuda.empty_cache()
    return agg


@torch.inference_mode()
def score_lettuce(claims, chunk_lists):
    """1 - max P(hallucinated) over answer tokens, max-over-chunks."""
    tok = AutoTokenizer.from_pretrained(LETTUCE)
    model = (
        AutoModelForTokenClassification.from_pretrained(LETTUCE, dtype=torch.float16).cuda().eval()
    )
    sep = tok.sep_token_id
    agg = np.zeros(len(claims), dtype=np.float32)
    flat_c, flat_k, owner = [], [], []
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        for k in ks:
            flat_c.append(c)
            flat_k.append(k)
            owner.append(i)
    scores = np.zeros(len(flat_c), dtype=np.float32)
    for i in range(0, len(flat_c), BATCH):
        enc = tok(
            flat_k[i : i + BATCH],
            flat_c[i : i + BATCH],  # context, answer
            truncation="only_first",
            max_length=4096,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        p = torch.softmax(model(**enc).logits.float(), dim=-1)[..., 1]
        ids = enc["input_ids"]
        for j in range(ids.shape[0]):
            row = ids[j].tolist()
            first = row.index(sep) if sep in row else 0
            m = enc["attention_mask"][j].bool().clone()
            m[: first + 1] = False
            q = p[j][m]
            scores[i + j] = 1.0 - (q.max().item() if q.numel() else 1.0)
    owner = np.array(owner)
    for i in range(len(claims)):
        agg[i] = scores[owner == i].max()
    del model
    torch.cuda.empty_cache()
    return agg


def auc_and_f1(y, s, seed=0):
    """AUC is threshold-free. F1 fits its threshold on half A, reports on half B."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    a, b = idx[: len(idx) // 2], idx[len(idx) // 2 :]
    grid = np.quantile(s[a], np.linspace(0.05, 0.95, 91))
    thr = max(grid, key=lambda t: f1_score(y[a], (s[a] >= t).astype(int), average="macro"))
    return (
        float(roc_auc_score(y, s)),
        float(f1_score(y[b], (s[b] >= thr).astype(int), average="macro")),
        float(thr),
    )


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    df = load_ragtruth()
    y = df["label"].to_numpy()
    claims = df["output"].to_list()
    chunk_lists = [top_chunks(c, CFG.semantic_top_k) for c in df["context"].to_list()]
    print(
        f"RAGTruth test sample: {len(df)} responses, grounded rate {y.mean():.3f}, "
        f"{np.mean([len(k) for k in chunk_lists]):.1f} chunks/response\n",
        flush=True,
    )

    res = {}
    for name, fn in (
        ("our cascade (bge-reranker-v2-m3)", score_reranker),
        ("lettucedect-v2-mmbert-base", score_lettuce),
    ):
        print(f"  scoring {name} ...", flush=True)
        s = fn(claims, chunk_lists)
        auc, f1, thr = auc_and_f1(y, s)
        res[name] = {"auc": round(auc, 4), "macro_f1": round(f1, 4), "threshold": round(thr, 4)}
        print(
            f"    AUC {auc:.4f}  macro-F1 {f1:.4f}  (threshold {thr:.3f} fitted on half A)\n",
            flush=True,
        )

    print("=" * 96)
    print("R7-H59 RESULT - cross-domain transfer matrix")
    print("=" * 96)
    print(f"{'model':34s} {'our private gold':>26s} {'RAGTruth (public)':>26s}")
    ours = res["our cascade (bge-reranker-v2-m3)"]
    theirs = res["lettucedect-v2-mmbert-base"]
    ours_cell = "AUC {:.4f}  F1 {:.4f}".format(ours["auc"], ours["macro_f1"])
    theirs_cell = "AUC {:.4f}  F1 {:.4f}".format(theirs["auc"], theirs["macro_f1"])
    print(f"{'our cascade (reranker)':34s} {'AUC 0.8619':>26s} {ours_cell:>26s}")
    print(
        f"{'lettucedect-v2 (public-trained)':34s} {'AUC 0.7095  F1 0.6313':>26s} "
        f"{theirs_cell:>26s}"
    )
    print()
    print(f"  our cascade, ours -> theirs : {ours['auc'] - 0.8619:+.4f}")
    print(f"  lettucedect, theirs -> ours : {0.7095 - theirs['auc']:+.4f}")
    print("\n  symmetric drops -> both models are domain-bound, public data is a different task")
    print("  our drop smaller -> our cascade is the more general model")
    print(
        "  our cascade holding near 0.86 -> our gold is not unusually easy (the R7-H49 question)"
    )
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\n  results -> {OUT}")


if __name__ == "__main__":
    main()

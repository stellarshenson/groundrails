"""R7-H51 - regenerate the teacher corpus under current chunking, WITH text.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 7).

`model_scores/pairs/full_pairs.npz` holds 111,800 rows of owner / reranker / NLI
/ labels / langs and NO TEXT, and the chunking behind it cannot be reproduced -
the R4 amendment measured cached mean 40.62 chunks/claim against `recursive_chunk`
matching per-claim counts on 39/300 claims at max_chars=1000. So the cached soft
labels cannot be re-attached to the inputs that produced them, and every
distillation hypothesis in this round is blocked until the corpus is rebuilt.

This rebuilds it under CURRENT chunking and keeps the text, emitting for every
(claim, chunk) pair: the claim, the chunk, the reranker score, the three NLI
probabilities, the gold label of the owning claim, and its language.

The acceptance bar is not "it ran" - it is that the regenerated pairs REPRODUCE
the incumbent's own numbers. If max-over-chunks on these pairs does not recover
the reranker's AUC ~0.841 and the stack's macro-F1 ~0.824, the corpus is not a
faithful teacher and nothing may be distilled from it.

Runs the torch checkpoints on GPU rather than the OpenVINO int8 exports: this is
a training-corpus build, so fidelity matters more than serving latency, and the
int8 exports exist to reproduce these models, not to define them.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R7-H51_regenerate_teacher.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import pathlib
import time

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from groundrails import settings
from groundrails.chunking import recursive_chunk
from groundrails.config import load_document_processing_config

settings.mark_ready()
CFG = load_document_processing_config()

HERE = pathlib.Path(__file__).parent
GOLD = HERE / "private-rag-forensics" / "gold" / "golden_grounding_evidence_verified.parquet"
OUT = HERE / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"

RERANKER = "BAAI/bge-reranker-v2-m3"
NLI = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
BATCH = 64
CACHED_AUC = 0.841  # incumbent reranker AUC on the gold - the fidelity bar


def build_pairs():
    g = pl.read_parquet(GOLD)
    claims, chunks, owner = [], [], []
    for i, (c, s) in enumerate(zip(g["claim"].to_list(), g["source_text"].to_list(), strict=True)):
        ch = recursive_chunk(s, max_chars=CFG.chunk_max_chars) or []
        texts = [x.text for x in ch] or [s[: CFG.chunk_max_chars]]
        for t in texts:
            owner.append(i)
            claims.append(c)
            chunks.append(t)
    return g, np.array(owner), claims, chunks


@torch.inference_mode()
def score_reranker(claims, chunks):
    tok = AutoTokenizer.from_pretrained(RERANKER)
    model = (
        AutoModelForSequenceClassification.from_pretrained(RERANKER, dtype=torch.float16)
        .cuda()
        .eval()
    )
    out = np.zeros(len(claims), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(claims), BATCH):
        enc = tok(
            claims[i : i + BATCH],
            chunks[i : i + BATCH],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda")
        out[i : i + BATCH] = torch.sigmoid(model(**enc).logits.float().squeeze(-1)).cpu().numpy()
        if i % (BATCH * 200) == 0:
            print(f"  rerank {i}/{len(claims)} ({time.time() - t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


@torch.inference_mode()
def score_nli(claims, chunks):
    tok = AutoTokenizer.from_pretrained(NLI)
    model = (
        AutoModelForSequenceClassification.from_pretrained(NLI, dtype=torch.float16).cuda().eval()
    )
    order = [model.config.label2id[k] for k in ("entailment", "neutral", "contradiction")]
    out = np.zeros((len(claims), 3), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(claims), BATCH):
        enc = tok(
            chunks[i : i + BATCH],
            claims[i : i + BATCH],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda")
        p = torch.softmax(model(**enc).logits.float(), dim=-1)[:, order]
        out[i : i + BATCH] = p.cpu().numpy()
        if i % (BATCH * 200) == 0:
            print(f"  nli {i}/{len(claims)} ({time.time() - t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def max_over(owner, vals, n):
    out = np.full(n, -1e9, dtype=np.float64)
    np.maximum.at(out, owner, vals)
    return out


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    g, owner, claims, chunks = build_pairs()
    n = len(g)
    print(
        f"gold {n} claims -> {len(claims)} pairs ({len(claims) / n:.1f} chunks/claim); "
        f"cached corpus was 111,800 pairs at 40.62 chunks/claim\n",
        flush=True,
    )

    rr = score_reranker(claims, chunks)
    nli = score_nli(claims, chunks)

    labels = g["label"].to_numpy()
    auc_rr = roc_auc_score(labels, max_over(owner, rr, n))
    auc_ent = roc_auc_score(labels, max_over(owner, nli[:, 0], n))

    pl.DataFrame(
        {
            "owner": owner,
            "claim": claims,
            "chunk": chunks,
            "rerank": rr,
            "entail": nli[:, 0],
            "neutral": nli[:, 1],
            "contradiction": nli[:, 2],
            "label": labels[owner],
            "lang": g["lang"].to_numpy()[owner],
        }
    ).write_parquet(OUT)

    print("\n" + "=" * 88)
    print("R7-H51 RESULT - regenerated teacher corpus")
    print("=" * 88)
    print(f"  pairs written           : {len(claims)}  -> {OUT}")
    print(f"  reranker AUC (max-over) : {auc_rr:.4f}   incumbent reference {CACHED_AUC}")
    print(f"  NLI entail AUC          : {auc_ent:.4f}   incumbent reference 0.806 all-lang")
    ok = abs(auc_rr - CACHED_AUC) <= 0.02
    print(
        f"\n  fidelity bar: reranker AUC within 0.02 of {CACHED_AUC}  ->  "
        f"{'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        print("  -> the regenerated corpus does NOT reproduce the incumbent; it is not a")
        print("     faithful teacher and nothing may be distilled from it until explained")


if __name__ == "__main__":
    main()

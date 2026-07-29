"""R4-H23 - zero-shot knowledge-free reasoner as a grounding judge.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 4).

Scores PleIAs/Monad (56.7M, trained on SYNTH - reasoning traces, not facts) frozen
over (claim, chunk) pairs and compares its separation against the incumbent
mDeBERTa-v3 NLI on the SAME pairs.

Two things this run establishes honestly:

1. Matched comparison. The cached anchors (NLI 0.8087 / reranker 0.8414) were built
   on a chunking this repo can no longer reproduce exactly (cached mean 40.6
   chunks/claim; recursive_chunk at any max_chars misses the per-claim counts). So
   the incumbent is RE-SCORED here on the same freshly-generated pairs rather than
   quoted from cache - otherwise the comparison is confounded by chunking.

2. Lower bound, stated up front. Monad's chat template forces a `<think>` trace
   before every answer. Scoring a direct yes/no continuation therefore violates its
   trained format, so this number is a LOWER BOUND on what the model can do. R4-H27
   measures the trace version; the gap between them is the value of the trace.

Scope: English slice only - the round tests whether the METHOD is load-bearing.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R4-H23_zeroshot_judge.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import pathlib  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from groundrails import settings  # noqa: E402
from groundrails.chunking import recursive_chunk  # noqa: E402
from groundrails.config import load_document_processing_config  # noqa: E402

settings.mark_ready()
CFG = load_document_processing_config()
P = pathlib.Path("experiments/grounding-semantic/private-rag-forensics")
CAND = "PleIAs/Monad"
INCUMBENT = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
OUT = pathlib.Path("experiments/grounding-semantic/R4-H23_scores.npz")


def build_pairs():
    g = pl.read_parquet(P / "gold/golden_grounding_evidence_verified.parquet")
    g = g.with_row_index("cid").filter(pl.col("lang").str.starts_with("en"))
    owner, claims, chunks = [], [], []
    for i, (c, s) in enumerate(zip(g["claim"].to_list(), g["source_text"].to_list(), strict=True)):
        ch = recursive_chunk(s, max_chars=CFG.chunk_max_chars) or []
        texts = [x.text for x in ch] or [s[: CFG.chunk_max_chars]]
        for t in texts:
            owner.append(i)
            claims.append(c)
            chunks.append(t)
    return g["label"].to_numpy(), np.array(owner), claims, chunks


def max_over(owner, vals, n):
    out = np.full(n, -1e9, dtype=np.float64)
    np.maximum.at(out, owner, vals)
    return out


@torch.inference_mode()
def score_monad(claims, chunks, bs=64):
    """logP(' supported') - logP(' unsupported') on the next token. One forward pass
    per pair, no generation - deterministic and comparable to a frozen cross-encoder."""
    tok = AutoTokenizer.from_pretrained(CAND)
    model = AutoModelForCausalLM.from_pretrained(CAND, dtype=torch.bfloat16).cuda().eval()
    tok.pad_token = tok.pad_token or "[PAD]"
    tok.padding_side = "left"  # last position is the prediction slot for every row

    def single_id(word):
        """Verdict words MUST be single tokens. ' unsupported' is 5 pieces
        ([' un','su','pp','ort','ed']) in Monad's 8k vocabulary, so first-token
        scoring silently compares logP(' supported') against logP(' un') - a bare
        prefix shared with thousands of unrelated words. That reads as chance and
        looks like a model null; it is a measurement defect. Fail loud instead."""
        ids = tok.encode(word, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    pos, neg = single_id(" yes"), single_id(" no")
    if pos is None or neg is None:
        raise RuntimeError("verdict words are not single tokens - pick another pair")
    print(f"  verdict token ids: pos={pos} neg={neg}", flush=True)

    prompts = [
        f"<|im_start|>user\nEvidence:\n{ch}\n\nClaim: {cl}\n\nIs the claim supported by the "
        f"evidence? Answer yes or no.<|im_end|>\n<|im_start|>assistant\n"
        for cl, ch in zip(claims, chunks, strict=True)
    ]
    out = np.zeros(len(prompts), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(prompts), bs):
        enc = tok(
            prompts[i : i + bs], return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to("cuda")
        logits = model(**enc).logits[:, -1, :].float()
        lp = torch.log_softmax(logits, dim=-1)
        out[i : i + bs] = (lp[:, pos] - lp[:, neg]).cpu().numpy()
        if i % (bs * 100) == 0:
            print(f"  monad {i}/{len(prompts)} ({time.time() - t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


@torch.inference_mode()
def score_nli(claims, chunks, bs=64):
    """Incumbent entailment probability on the SAME pairs - the matched anchor."""
    tok = AutoTokenizer.from_pretrained(INCUMBENT)
    model = (
        AutoModelForSequenceClassification.from_pretrained(INCUMBENT, dtype=torch.float16)
        .cuda()
        .eval()
    )
    ent = [i for i, v in model.config.id2label.items() if v.lower().startswith("entail")][0]
    print(f"  entailment index = {ent} ({model.config.id2label})", flush=True)
    out = np.zeros(len(claims), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(claims), bs):
        enc = tok(
            chunks[i : i + bs],
            claims[i : i + bs],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda")
        p = torch.softmax(model(**enc).logits.float(), dim=-1)[:, ent]
        out[i : i + bs] = p.cpu().numpy()
        if i % (bs * 100) == 0:
            print(f"  nli {i}/{len(claims)} ({time.time() - t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    labels, owner, claims, chunks = build_pairs()
    n = len(labels)
    print(f"EN slice: {n} claims, {len(claims)} pairs "
          f"({len(claims) / n:.1f} chunks/claim), base rate {labels.mean():.3f}\n", flush=True)

    print("scoring incumbent mDeBERTa-NLI on these pairs...", flush=True)
    nli = score_nli(claims, chunks)
    print("scoring Monad (direct answer, no <think> - LOWER BOUND)...", flush=True)
    mon = score_monad(claims, chunks)

    auc_nli = roc_auc_score(labels, max_over(owner, nli, n))
    auc_mon = roc_auc_score(labels, max_over(owner, mon, n))
    np.savez(OUT, labels=labels, owner=owner, nli=nli, monad=mon)

    print("\n" + "=" * 62)
    print("R4-H23 RESULT (English slice, max-over-chunks, matched pairs)")
    print("=" * 62)
    print(f"  incumbent mDeBERTa-NLI : AUC {auc_nli:.4f}")
    print(f"  Monad 56.7M (no trace) : AUC {auc_mon:.4f}   delta {auc_mon - auc_nli:+.4f}")
    print(f"\n  bar: >= 0.70 unlocks the tiny track; < 0.65 kills zero-shot transfer")
    v = "PASS - proceeds" if auc_mon >= 0.70 else ("KILLED" if auc_mon < 0.65 else "MARGINAL")
    print(f"  R4-H23 VERDICT (lower bound): {v}")
    print(f"\n  scores saved -> {OUT}")


if __name__ == "__main__":
    main()

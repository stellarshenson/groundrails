"""Single-engine OpenVINO int8 semantic grounder - build + scoring helpers.

Everything (bi-encoder pre-filter, reranker, NLI) runs on one runtime: OpenVINO.
DeBERTa-v2 needs SmoothQuant to survive int8 (disentangled-attention outliers); the
standard-attention bge models quantize cleanly with plain NNCF int8. Build once to an IR
directory (cached), then compile + score on CPU.

Pipeline per claim: bi-encoder pre-filters the evidence chunks to top-k, the two
cross-encoders (reranker + NLI) score the k survivors, a logistic over their
max-over-chunks scores classifies supported vs hallucination.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import openvino as ov
from tqdm.auto import tqdm

_core = ov.Core()

# Published OpenVINO int8 IRs on the HuggingFace Hub (built by scripts/build_ov_grounder.py)
HF_REPOS = {
    "bge-m3": "stellars/bge-m3-openvino-int8",
    "bge-reranker-v2-m3": "stellars/bge-reranker-v2-m3-openvino-int8",
    "mDeBERTa-v3-nli": "stellars/mdeberta-v3-base-mnli-xnli-openvino-int8",
}


def load_ov_hf(repo_id_or_name, compile: bool = True,
               tokenizer_kwargs: dict | None = None, model_kwargs: dict | None = None):
    """Download a published OpenVINO int8 IR repo from HF and return (compiled_model, tokenizer, dir).

    Accepts a full repo id (`stellars/bge-m3-openvino-int8`) or a short name key in `HF_REPOS`
    (`bge-m3`). The IR, config and tokenizer ship together in the repo, so no base model is needed.
    Cached after first download; set compile=False to skip compiling (e.g. to inspect files).
    tokenizer_kwargs are forwarded to `AutoTokenizer.from_pretrained`, model_kwargs to
    `compile_ir` (e.g. hint="THROUGHPUT"). fix_mistral_regex=False is the default: the
    transformers mistral-regex warning is a false positive for these sentencepiece
    (Metaspace pre-tokenizer) tokenizers - True crashes in `_patch_mistral_regex` because
    there is no regex pre-tokenizer to patch, and False silences the warning unpatched.
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    repo_id = HF_REPOS.get(repo_id_or_name, repo_id_or_name)
    d = Path(snapshot_download(repo_id))
    tok = AutoTokenizer.from_pretrained(d, **{"fix_mistral_regex": False,
                                              **(tokenizer_kwargs or {})})
    cm = compile_ir(d / "openvino_model.xml", **(model_kwargs or {})) if compile else None
    return cm, tok, d


# --------------------------------------------------------------------------- build

def build_ov_int8(model_id: str, task: str, calib_dataset, out_dir,
                  smooth_alpha: float | None = None, n_calib: int = 128) -> Path:
    """Export + NNCF int8 quantize a model to an OpenVINO IR (cached by out_dir).

    task: "cls" (OVModelForSequenceClassification) or "feat" (OVModelForFeatureExtraction).
    smooth_alpha: enable SmoothQuant (needed for DeBERTa-v2); None = plain int8.
    calib_dataset: a datasets.Dataset of tokenized inputs.
    """
    from optimum.intel import (OVConfig, OVModelForFeatureExtraction,
                               OVModelForSequenceClassification, OVQuantizationConfig,
                               OVQuantizer)
    out_dir = Path(out_dir)
    if (out_dir / "openvino_model.xml").exists():
        return out_dir
    cls = OVModelForSequenceClassification if task == "cls" else OVModelForFeatureExtraction
    model = cls.from_pretrained(model_id, export=True)
    quantizer = OVQuantizer.from_pretrained(model)
    qkw = dict(bits=8, num_samples=n_calib)
    if smooth_alpha is not None:
        qkw["smooth_quant_alpha"] = smooth_alpha
    cfg = OVConfig(quantization_config=OVQuantizationConfig(**qkw))
    quantizer.quantize(calibration_dataset=calib_dataset, save_directory=out_dir, ov_config=cfg)
    return out_dir


# Sequence cap. Measured on the gold (scripts/bench_mechanical_levers.py): chunks alone
# run ~300 tokens median / 418 p95, and (claim, chunk) pairs ~331 median / ~590 p95, so
# 512 already truncates ~6.5% of pairs - there is NO headroom to cap lower without
# clipping the median pair. Kept configurable for deployments with shorter chunks.
MAX_LEN = 512


def compile_ir(xml_path, hint: str = "LATENCY"):
    """Compile an IR for CPU inference.

    hint: "LATENCY" (default) minimises single-request wall time - the right choice for
    the inline per-claim serving path; "THROUGHPUT" spins up multiple async streams and
    wins only when many requests are in flight (the batch/offline scoring path).
    """
    return _core.compile_model(_core.read_model(str(xml_path)), "CPU",
                               {"PERFORMANCE_HINT": hint})


def _feed(cm, enc):
    names = {i.get_any_name() for i in cm.inputs}
    return {n: enc[n].astype(np.int64) for n in enc if n in names}


def _bucket(chunks):
    """Order chunk indices by char length so each padded batch holds similar-length rows.

    The grounder takes the max score over chunks, so reordering is result-invariant; the
    only effect is less padding per batch (10-30% fewer wasted attention cells).
    """
    return sorted(range(len(chunks)), key=lambda i: len(chunks[i]))


# --------------------------------------------------------------------------- scoring

def embed_vectors(cm, tok, texts, bs: int = 32, pool: str = "cls",
                  max_len: int = MAX_LEN) -> np.ndarray:
    """L2-normalised embeddings (cls or mean pool) on an OpenVINO feature IR."""
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_len,
                  return_tensors="np")
        h = cm(_feed(cm, enc))[cm.output(0)]            # last_hidden_state [B,T,H]
        if pool == "cls":
            v = h[:, 0]
        else:
            m = enc["attention_mask"][..., None]
            v = (h * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        out.append(v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9))
    return np.concatenate(out)


def topk_chunks(emb_cm, tok, claim, chunks, k: int, pool: str = "cls",
                qpre: str = "", dpre: str = "", max_len: int = MAX_LEN) -> list[str]:
    """Bi-encoder pre-filter: keep the k chunks most similar to the claim (original order)."""
    if k is None or k >= len(chunks):
        return chunks
    cv = embed_vectors(emb_cm, tok, [qpre + claim], pool=pool, max_len=max_len)[0]
    dv = embed_vectors(emb_cm, tok, [dpre + c for c in chunks], pool=pool, max_len=max_len)
    idx = np.argsort(-(dv @ cv))[:k]
    return [chunks[i] for i in sorted(idx)]


def rerank_max(cm, tok, claim, chunks, bs: int = 16, max_len: int = MAX_LEN) -> float:
    """Max sigmoid relevance (claim, chunk) over chunks - the reranker signal.

    Chunks are length-bucketed before batching so each padded batch wastes less; the
    max is order-invariant so the score is unchanged.
    """
    order = _bucket(chunks)
    best = -1e9
    for i in range(0, len(order), bs):
        sub = [chunks[j] for j in order[i:i + bs]]
        enc = tok([claim] * len(sub), sub, padding=True, truncation=True, max_length=max_len,
                  return_tensors="np")
        lg = cm(_feed(cm, enc))[cm.output(0)].reshape(-1)
        best = max(best, float((1.0 / (1.0 + np.exp(-lg))).max()))
    return best


def nli_max(cm, tok, claim, chunks, ent_idx: int = 0, bs: int = 16,
            max_len: int = MAX_LEN) -> float:
    """Max entailment prob (premise=chunk, hypothesis=claim) over chunks - the NLI signal.

    Length-bucketed like `rerank_max`; the max over chunks is order-invariant.
    """
    order = _bucket(chunks)
    best = -1e9
    for i in range(0, len(order), bs):
        sub = [chunks[j] for j in order[i:i + bs]]
        enc = tok(sub, [claim] * len(sub), padding=True, truncation=True, max_length=max_len,
                  return_tensors="np")
        lg = cm(_feed(cm, enc))[cm.output(0)]
        ex = np.exp(lg - lg.max(1, keepdims=True))
        best = max(best, float((ex / ex.sum(1, keepdims=True))[:, ent_idx].max()))
    return best


# ----------------------------------------------------------------- full-gold async (progress)

def _async_cross(cm, tok, texts_a, texts_b, post, bs: int = 16, desc=None) -> np.ndarray:
    """Async-queue scoring of a flat (a, b) text-pair list; progress bar advances on completion."""
    names = {i.get_any_name() for i in cm.inputs}
    out = np.zeros(len(texts_a))
    q = ov.AsyncInferQueue(cm, cm.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))
    # mininterval rate-limits refreshes so headless (nbconvert) runs don't flood the kernel IOPub
    pbar = tqdm(total=len(texts_a), desc=desc, mininterval=2.0) if desc else None

    def cb(req, start):
        lg = req.get_output_tensor(0).data
        out[start:start + lg.shape[0]] = post(lg)
        if pbar:
            pbar.update(lg.shape[0])
    q.set_callback(cb)
    for j in range(0, len(texts_a), bs):
        b = texts_b[j:j + bs] if texts_b is not None else None
        enc = (tok(texts_a[j:j + bs], b, padding=True, truncation=True, max_length=512,
                   return_tensors="np") if b is not None else
               tok(texts_a[j:j + bs], padding=True, truncation=True, max_length=512,
                   return_tensors="np"))
        q.start_async({n: enc[n].astype(np.int64) for n in enc if n in names}, j)
    q.wait_all()
    if pbar:
        pbar.close()
    return out


def _async_embed(cm, tok, texts, bs: int = 32, pool: str = "cls", desc=None) -> np.ndarray:
    """Async-queue CLS/mean embeddings (L2-normalised); progress bar advances on completion."""
    names = {i.get_any_name() for i in cm.inputs}
    nb = (len(texts) + bs - 1) // bs
    vecs = [None] * nb
    q = ov.AsyncInferQueue(cm, cm.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))
    # mininterval rate-limits refreshes so headless (nbconvert) runs don't flood the kernel IOPub
    pbar = tqdm(total=len(texts), desc=desc, mininterval=2.0) if desc else None
    masks = [None] * nb

    def cb(req, bi):
        h = req.get_output_tensor(0).data
        if pool == "cls":
            v = h[:, 0]
        else:
            m = masks[bi][..., None]
            v = (h * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        vecs[bi] = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        if pbar:
            pbar.update(v.shape[0])
    q.set_callback(cb)
    for bi, j in enumerate(range(0, len(texts), bs)):
        enc = tok(texts[j:j + bs], padding=True, truncation=True, max_length=512, return_tensors="np")
        masks[bi] = enc["attention_mask"]
        q.start_async({n: enc[n].astype(np.int64) for n in enc if n in names}, bi)
    q.wait_all()
    if pbar:
        pbar.close()
    return np.concatenate(vecs)


# H11 reranker-first cascade band, fit OOF on the gold (grounding_hypotheses.py):
# claims whose reranker max falls outside [a, b] take the reranker-only verdict and
# skip the NLI forward (~60% of claims, macro-F1 held at 0.795 vs 0.797 always-both).
CASCADE_BAND = (0.01, 0.66)

# H12 pre-filter cosine gate, fit OOF like the band: claims whose max claim-chunk
# cosine (already computed by the pre-filter ranking) falls outside [a0, b0] skip BOTH
# cross-encoders - cos <= a0 flags hallucination, cos >= b0 passes supported. 22% of
# claims resolve at embed cost with strictly fewer errors than the cascade alone
# (FP 245/FN 216 vs 248/217). Compose with `rerank_max_early` for the adopted round-2
# serving path; reference composition in scripts/bench_grounder_round2.py.
COSINE_GATE = (0.493, 0.739)


def cascade_scores(rr_cm, rr_tok, nli_cm, nli_tok, claim, chunks,
                   band=CASCADE_BAND, bs: int = 16, max_len: int = MAX_LEN):
    """Reranker-first cascade: returns (rr_max, nli_max | None).

    The NLI runs only when the reranker max is inside the uncertainty band; callers
    use the stack verdict when nli is not None, else the reranker-only threshold.
    Warm latency: ~28% mean saving vs always-both (scripts/bench_grounder_cascade.py).
    """
    s = rerank_max(rr_cm, rr_tok, claim, chunks, bs=bs, max_len=max_len)
    if band[0] <= s <= band[1]:
        return s, nli_max(nli_cm, nli_tok, claim, chunks, bs=bs, max_len=max_len)
    return s, None


def rerank_max_early(cm, tok, claim, chunks_ranked, pass_edge: float = CASCADE_BAND[1],
                     schedule=(1, 1, 2, 4), max_len: int = MAX_LEN):
    """H13 - short-circuit reranker max. Chunks must arrive in pre-filter rank order
    (best cosine first); pairs are scored in progressive batches and scoring stops the
    moment the running max crosses `pass_edge` - the cascade then passes the claim and
    skips the NLI, so unscored pairs cannot change the verdict. Claims that never cross
    score every pair, reproducing the exact max. Verdicts are therefore identical to
    `cascade_scores` up to int8 batch-composition jitter.

    Returns (max_score, pairs_scored).
    """
    best, i = -1e9, 0
    sched = list(schedule)
    while i < len(chunks_ranked):
        step = sched.pop(0) if sched else schedule[-1]
        sub = chunks_ranked[i:i + step]
        enc = tok([claim] * len(sub), sub, padding=True, truncation=True,
                  max_length=max_len, return_tensors="np")
        lg = cm(_feed(cm, enc))[cm.output(0)].reshape(-1)
        best = max(best, float((1.0 / (1.0 + np.exp(-lg))).max()))
        i += len(sub)
        if best >= pass_edge:
            break
    return best, i


# ----------------------------------------------------------------- fused evidence (H14)

_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def fused_context_chunks(chunks_ranked, n_chunks: int = 2, chunk_chars: int = 850) -> str:
    """H14 v1 - one evidence context from the top-ranked chunks. Each chunk is
    char-truncated so two ~300-token chunks fit the 512-token cross-encoder window
    (~850 chars ~ 230 tokens)."""
    return "\n\n".join(c[:chunk_chars] for c in chunks_ranked[:n_chunks])


def split_sentences(text: str, min_chars: int = 20) -> list[str]:
    """Sentence segments for salience packing - regex boundary split, short fragments
    dropped. Multilingual-safe enough for evidence packing (mechanism, not parsing)."""
    return [p.strip() for p in _SENT_RE.split(text) if len(p.strip()) >= min_chars]


def pack_sentences(sentences, sims, budget_chars: int = 1800) -> str:
    """H14 v2 - pick sentences by descending claim similarity until the char budget
    (~1,800 chars ~ 480 tokens), then restore original order so the packed premise
    keeps local coherence."""
    order = np.argsort(-np.asarray(sims))
    keep, used = [], 0
    for i in order:
        ln = len(sentences[i])
        if used + ln > budget_chars and keep:
            continue
        keep.append(int(i))
        used += ln
    return " ".join(sentences[i] for i in sorted(keep))


def _async_cross_probs(cm, tok, texts_a, texts_b, n_out: int, bs: int = 16,
                       desc=None) -> np.ndarray:
    """Async-queue scoring keeping the FULL softmax per pair, shape (n_pairs, n_out).

    Same machinery as `_async_cross` but without collapsing to one number - used to
    cache all NLI channels (entailment / neutral / contradiction) per pair.
    """
    names = {i.get_any_name() for i in cm.inputs}
    out = np.zeros((len(texts_a), n_out), dtype=np.float32)
    q = ov.AsyncInferQueue(cm, cm.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))
    pbar = tqdm(total=len(texts_a), desc=desc, mininterval=2.0) if desc else None

    def cb(req, start):
        lg = req.get_output_tensor(0).data
        ex = np.exp(lg - lg.max(1, keepdims=True))
        out[start:start + lg.shape[0]] = ex / ex.sum(1, keepdims=True)
        if pbar:
            pbar.update(lg.shape[0])
    q.set_callback(cb)
    for j in range(0, len(texts_a), bs):
        enc = tok(texts_a[j:j + bs], texts_b[j:j + bs], padding=True, truncation=True,
                  max_length=512, return_tensors="np")
        q.start_async({n: enc[n].astype(np.int64) for n in enc if n in names}, j)
    q.wait_all()
    if pbar:
        pbar.close()
    return out


def pair_scores_full(records, rr_cm, rr_tok, nli_cm, nli_tok, bs: int = 16,
                     progress: bool = True):
    """Per-pair cross-encoder outputs over ALL (claim, chunk) pairs of `records`.

    Returns (owner, rr, nli): owner[p] = record index of pair p, rr[p] = reranker
    sigmoid, nli[p] = full 3-class softmax (model label order, mDeBERTa: entailment /
    neutral / contradiction). Max over a record's pairs reproduces the cached
    max-over-chunks signals; the full distributions feed the aggregation and
    contradiction-channel hypotheses.
    """
    claims = [r["claim"] for r in records]
    owner, flat = [], []
    for i, r in enumerate(records):
        for c in r["chunks"]:
            owner.append(i)
            flat.append(c)
    owner = np.array(owner)
    rr = _async_cross(rr_cm, rr_tok, [claims[o] for o in owner], flat,
                      lambda lg: 1.0 / (1.0 + np.exp(-lg.reshape(-1))), bs=bs,
                      desc="rerank pairs" if progress else None)
    n_out = nli_cm.output(0).get_partial_shape()[1].get_length()
    nli = _async_cross_probs(nli_cm, nli_tok, flat, [claims[o] for o in owner], n_out,
                             bs=bs, desc="nli pairs" if progress else None)
    return owner, rr.astype(np.float32), nli


def score_pipeline_ranked(records, emb_cm, emb_tok, rr_cm, rr_tok, nli_cm, nli_tok, ks,
                          ent_idx: int = 0, pool: str = "cls", progress: bool = True):
    """Full pipeline scores over `records` for every top-k in `ks` (last == all chunks).

    Scores ALL chunks once (embed -> rank, reranker, NLI) via async queues with progress bars,
    then takes the max reranker / NLI over the top-k per record - so the sweep is a free slice.
    Returns (rerank_k, nli_k), each shape [len(records), len(ks)].
    """
    claims = [r["claim"] for r in records]
    owner, flat = [], []
    for i, r in enumerate(records):
        for c in r["chunks"]:
            owner.append(i); flat.append(c)
    owner = np.array(owner)

    cvec = _async_embed(emb_cm, emb_tok, claims, pool=pool, desc="embed claims" if progress else None)
    chvec = _async_embed(emb_cm, emb_tok, flat, pool=pool, desc="embed chunks" if progress else None)
    rr = _async_cross(rr_cm, rr_tok, [claims[o] for o in owner], flat,
                      lambda lg: 1.0 / (1.0 + np.exp(-lg.reshape(-1))),
                      desc="rerank" if progress else None)

    def _entail(lg):
        ex = np.exp(lg - lg.max(1, keepdims=True))
        return (ex / ex.sum(1, keepdims=True))[:, ent_idx]
    nli = _async_cross(nli_cm, nli_tok, flat, [claims[o] for o in owner], _entail,
                       desc="nli" if progress else None)

    rerank_k = np.zeros((len(records), len(ks)))
    nli_k = np.zeros((len(records), len(ks)))
    for i in range(len(records)):
        m = owner == i
        order = np.argsort(-(chvec[m] @ cvec[i]))
        rr_i, nli_i = rr[m][order], nli[m][order]
        for j, k in enumerate(ks):
            kk = min(k, len(rr_i))
            rerank_k[i, j] = rr_i[:kk].max()
            nli_k[i, j] = nli_i[:kk].max()
    return rerank_k, nli_k

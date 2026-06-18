"""Precompute the single-engine OpenVINO grounder's pipeline scores over the full gold.

Heavy CPU work (3 large int8 models x ~110k chunks) - run once in the background with
OpenVINO async throughput, cache to npz so the notebook loads instantly. For each record we
rank its chunks by the bi-encoder, then take the max reranker / NLI score over the top-k for
every k in K_SWEEP (scoring all chunks once makes the top-k sweep a free slice).

Output: data/interim/model_scores/ov_pipeline.npz with rerank_k[n, len(KS)], nli_k[n, len(KS)],
and the KS list (last entry = all chunks).
"""
import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import openvino as ov

from grounding_models import load_gold, SCORES_DIR
from grounding_openvino import load_ov_hf

KS = [5, 8, 12, 50]                      # last (>=max chunks) == all chunks
recs = load_gold()
print(f"records={len(recs)} chunks={sum(len(r['chunks']) for r in recs)}", flush=True)


def _async_flat(cm, tok, texts_a, texts_b, post, bs=16):
    """Score a flat list of (a,b) text pairs (or a-only if texts_b is None) via async queue.

    post(logits)->vector maps each batch's logits to per-row scalars; returns flat array.
    """
    names = {i.get_any_name() for i in cm.inputs}
    n = len(texts_a)
    out = np.zeros(n)
    q = ov.AsyncInferQueue(cm, cm.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))

    def cb(req, start):
        lg = req.get_output_tensor(0).data
        out[start:start + lg.shape[0]] = post(lg)
    q.set_callback(cb)
    for j in range(0, n, bs):
        a = texts_a[j:j + bs]
        b = texts_b[j:j + bs] if texts_b is not None else None
        enc = (tok(a, b, padding=True, truncation=True, max_length=512, return_tensors="np")
               if b is not None else
               tok(a, padding=True, truncation=True, max_length=512, return_tensors="np"))
        feed = {nm: enc[nm].astype(np.int64) for nm in enc if nm in names}
        q.start_async(feed, j)
    q.wait_all()
    return out


def _async_embed(cm, tok, texts, bs=32):
    """CLS-pooled, L2-normalised embeddings via async queue -> [len(texts), H]."""
    names = {i.get_any_name() for i in cm.inputs}
    vecs = [None] * ((len(texts) + bs - 1) // bs)
    q = ov.AsyncInferQueue(cm, cm.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))

    def cb(req, bi):
        h = req.get_output_tensor(0).data            # [B,T,H]
        v = h[:, 0]
        vecs[bi] = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    q.set_callback(cb)
    for bi, j in enumerate(range(0, len(texts), bs)):
        enc = tok(texts[j:j + bs], padding=True, truncation=True, max_length=512, return_tensors="np")
        feed = {nm: enc[nm].astype(np.int64) for nm in enc if nm in names}
        q.start_async(feed, bi)
    q.wait_all()
    return np.concatenate(vecs)


# flatten chunks
owner, flat_chunks, claims = [], [], [r["claim"] for r in recs]
for i, r in enumerate(recs):
    for c in r["chunks"]:
        owner.append(i); flat_chunks.append(c)
owner = np.array(owner)

# 1. embedder ranking (bi-encoder pre-filter)
t = time.time()
ecm, etok, _ = load_ov_hf("bge-m3")                      # OpenVINO int8 IRs pulled from HF
claim_vec = _async_embed(ecm, etok, claims)
chunk_vec = _async_embed(ecm, etok, flat_chunks)
print(f"embedded in {(time.time()-t)/60:.1f} min", flush=True)

# 2. reranker (claim, chunk) sigmoid; 3. NLI (chunk, claim) entailment idx 0
t = time.time()
rcm, rtok, _ = load_ov_hf("bge-reranker-v2-m3")
rr_flat = _async_flat(rcm, rtok, [claims[o] for o in owner], flat_chunks,
                      lambda lg: 1.0 / (1.0 + np.exp(-lg.reshape(-1))))
print(f"reranked in {(time.time()-t)/60:.1f} min", flush=True)

t = time.time()
ncm, ntok, _ = load_ov_hf("mDeBERTa-v3-nli")
def _entail(lg):
    ex = np.exp(lg - lg.max(1, keepdims=True)); return (ex / ex.sum(1, keepdims=True))[:, 0]
nli_flat = _async_flat(ncm, ntok, flat_chunks, [claims[o] for o in owner], _entail)
print(f"nli in {(time.time()-t)/60:.1f} min", flush=True)

# 4. per record: rank chunks by cosine, max rerank/nli over top-k for each k
rerank_k = np.zeros((len(recs), len(KS))); nli_k = np.zeros((len(recs), len(KS)))
for i in range(len(recs)):
    m = owner == i
    cos = chunk_vec[m] @ claim_vec[i]
    order = np.argsort(-cos)
    rr_i, nli_i = rr_flat[m][order], nli_flat[m][order]
    for j, k in enumerate(KS):
        kk = min(k, len(rr_i))
        rerank_k[i, j] = rr_i[:kk].max()
        nli_k[i, j] = nli_i[:kk].max()

np.savez(SCORES_DIR / "ov_pipeline.npz", rerank_k=rerank_k, nli_k=nli_k, ks=np.array(KS))
print(f"CACHED -> {SCORES_DIR / 'ov_pipeline.npz'}  shape {rerank_k.shape}", flush=True)
print("DONE", flush=True)

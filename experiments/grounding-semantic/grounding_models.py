"""Local GPU grounding-signal experiments - bi-encoders and cross-encoders.

Scores the verified golden dataset with several embedding models and multilingual
cross-encoders directly via transformers + torch (no grounder plugin), to compare
how well each separates supported claims from hallucinations on real production traffic.

Each claim is scored against its own answer's retrieved evidence, chunked into
passages; the claim's score is the max over its chunks (best supporting passage).
Higher score = more grounded.

Used by notebooks/grounding-semantic/01-kj-grounding-model-comparison.ipynb.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# torch + transformers are imported lazily inside the GPU reference-scoring functions.
# The gold loader, chunker and metrics below stay torch-free, so the cached-score
# ensemble path runs without a torch install (inference is torch-free by design).

FORENSICS = Path(__file__).resolve().parent / "private-rag-forensics"
GOLD = FORENSICS / "gold" / "golden_grounding_evidence_verified.parquet"

# Bi-encoder embedding models (pool: mean|cls; qpre/dpre: e5-style prefixes)
# gte-multilingual-base dropped: its custom CUDA kernel crashes on torch cu130.
EMBEDDERS = [
    {"name": "intfloat/multilingual-e5-small", "pool": "mean", "qpre": "query: ", "dpre": "passage: "},
    {"name": "intfloat/multilingual-e5-large", "pool": "mean", "qpre": "query: ", "dpre": "passage: "},
    {"name": "BAAI/bge-m3", "pool": "cls", "qpre": "", "dpre": ""},
    {"name": "jhu-clsp/mmBERT-base", "pool": "mean", "qpre": "", "dpre": ""},
]

# Cross-encoders (kind: rerank = 1-logit relevance; nli = entailment prob)
# gte-multilingual-reranker dropped (same custom-kernel crash); replaced with the
# standard-arch multilingual mMiniLM reranker.
CROSS = [
    {"name": "BAAI/bge-reranker-v2-m3", "kind": "rerank"},
    {"name": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", "kind": "nli"},
    # NLI replacement candidates (standard attention -> int8-safe), smallest first;
    # searching for the smallest whose int8 stack holds macro-F1 (mDeBERTa int8 broke).
    {"name": "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli", "kind": "nli"},
    {"name": "MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli", "kind": "nli"},
    {"name": "symanto/xlm-roberta-base-snli-mnli-anli-xnli", "kind": "nli"},
    {"name": "joeddav/xlm-roberta-large-xnli", "kind": "nli", "tok": "FacebookAI/xlm-roberta-large"},
]


def chunk_text(text: str, size: int = 1100, overlap: int = 200, max_chunks: int = 50) -> list[str]:
    """Sliding-window char chunks of the evidence blob."""
    text = text or ""
    if len(text) <= size:
        return [text] if text.strip() else []
    step = max(size - overlap, 1)
    chunks = [text[i:i + size] for i in range(0, len(text), step)]
    chunks = [c for c in chunks if c.strip()]
    return chunks[:max_chunks]


def load_gold(chunk_size: int = 1100, overlap: int = 200, max_chunks: int = 50) -> list[dict]:
    import polars as pl
    recs = pl.read_parquet(GOLD).to_dicts()
    out = []
    for r in recs:
        ch = chunk_text(r["source_text"], chunk_size, overlap, max_chunks)
        if not ch:
            continue
        out.append({"claim": r["claim"], "chunks": ch, "label": int(r["label"]),
                    "lang": r.get("lang", "en")})
    return out


def _mk_config(name: str, trust: bool):
    """Load config and disable custom CUDA fast-paths that crash on Blackwell.

    The gte 'new' architecture's unpadding / memory-efficient-attention kernels
    raise a CUDA device-side assert on torch cu130 / sm_120; forcing the standard
    path keeps it on a portable kernel.
    """
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(name, trust_remote_code=trust)
    # gte custom kernels crash on cu130; ModernBERT (mmBERT) reference_compile hangs
    for k in ("unpad_inputs", "use_memory_efficient_attention", "reference_compile"):
        if hasattr(c, k):
            setattr(c, k, False)
    return c


def _pool(out, mask, how: str):
    h = out.last_hidden_state
    if how == "cls":
        return h[:, 0]
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


def _encode(model, tok, texts, device, pool, bs=64, max_len=512):
    import torch
    import torch.nn.functional as F
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_len,
                    return_tensors="pt").to(device)
            out = model(**b)
            v = _pool(out, b["attention_mask"], pool)
            vecs.append(F.normalize(v, p=2, dim=1).cpu())
    return torch.cat(vecs)


def embed_scores(cfg: dict, records: list[dict], device: str, bs: int = 64) -> np.ndarray:
    """Per-record max cosine(claim, its chunks)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["name"], trust_remote_code=cfg.get("trust", False))
    model = AutoModel.from_pretrained(
        cfg["name"], config=_mk_config(cfg["name"], cfg.get("trust", False)),
        trust_remote_code=cfg.get("trust", False)).to(device).eval()
    claims = [cfg.get("qpre", "") + r["claim"] for r in records]
    cv = _encode(model, tok, claims, device, cfg["pool"], bs)
    scores = np.zeros(len(records), dtype=float)
    for i, r in enumerate(records):
        docs = [cfg.get("dpre", "") + c for c in r["chunks"]]
        dv = _encode(model, tok, docs, device, cfg["pool"], bs)
        scores[i] = float((dv @ cv[i]).max())
    del model
    torch.cuda.empty_cache()
    return scores


def cross_scores(cfg: dict, records: list[dict], device: str, bs: int = 64) -> np.ndarray:
    """Per-record max cross-encoder score(claim, its chunks). rerank=sigmoid logit, nli=entailment."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    # cfg["tok"] overrides the tokenizer repo (e.g. joeddav ships a broken sentencepiece; use the base)
    tok = AutoTokenizer.from_pretrained(cfg.get("tok", cfg["name"]), trust_remote_code=cfg.get("trust", False))
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["name"], config=_mk_config(cfg["name"], cfg.get("trust", False)),
        trust_remote_code=cfg.get("trust", False)).to(device).eval()
    ent_idx = None
    if cfg["kind"] == "nli":
        id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
        ent_idx = next((i for i, v in id2label.items() if "entail" in v), 0)

    scores = np.zeros(len(records), dtype=float)
    with torch.no_grad():
        for i, r in enumerate(records):
            # pair the claim with each chunk; for NLI premise=chunk, hypothesis=claim
            pairs = [(c, r["claim"]) if cfg["kind"] == "nli" else (r["claim"], c) for c in r["chunks"]]
            best = -1e9
            for j in range(0, len(pairs), bs):
                sub = pairs[j:j + bs]
                enc = tok([p[0] for p in sub], [p[1] for p in sub], padding=True, truncation=True,
                          max_length=512, return_tensors="pt").to(device)
                logits = model(**enc).logits
                if cfg["kind"] == "nli":
                    s = F.softmax(logits, dim=1)[:, ent_idx]
                else:
                    s = torch.sigmoid(logits.squeeze(-1)) if logits.shape[-1] == 1 else F.softmax(logits, dim=1)[:, -1]
                best = max(best, float(s.max()))
            scores[i] = best
    del model
    torch.cuda.empty_cache()
    return scores


def metrics(scores: np.ndarray, labels: np.ndarray, thresholds=None) -> dict:
    """Detection of hallucination (label 0). score high = grounded; flag if score < T.

    AUC is computed for separating supported(1) from hallucination(0) by score
    (threshold-free). Sweep reports recall/precision/false-flag/acc per T, and the
    best (lowest) false-flag achievable at recall >= 0.85.
    """
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels)
    auc = float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) > 1 else float("nan")
    if thresholds is None:
        lo, hi = float(np.quantile(scores, 0.02)), float(np.quantile(scores, 0.98))
        thresholds = list(np.linspace(lo, hi, 19))
    sweep, best_ff = [], None
    for T in thresholds:
        flag = scores < T
        hall = labels == 0
        tp = int((flag & hall).sum()); fn = int((~flag & hall).sum())
        fp = int((flag & ~hall).sum()); tn = int((~flag & ~hall).sum())
        rec = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        ff = fp / (fp + tn) if fp + tn else 0.0
        acc = (tp + tn) / len(labels)
        row = {"T": float(T), "recall": rec, "precision": prec, "false_flag": ff, "acc": acc}
        sweep.append(row)
        if rec >= 0.85 and (best_ff is None or ff < best_ff["false_flag"]):
            best_ff = row
    return {"auc": auc, "sweep": sweep, "best_at_recall85": best_ff,
            "hall_mean": float(scores[labels == 0].mean()), "supp_mean": float(scores[labels == 1].mean())}


def pick_gpu() -> str:
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    return "cuda:0"


SCORES_DIR = FORENSICS / "model_scores"


def score_one(name: str, chunk_size=1100, overlap=200, max_chunks=50) -> dict:
    """Score one model on the full gold and persist scores + metrics. Isolated entry
    point so a CUDA crash in one model cannot poison the others (run per subprocess)."""
    recs = load_gold(chunk_size, overlap, max_chunks)
    labels = np.array([r["label"] for r in recs])
    dev = pick_gpu()
    emb = {c["name"]: c for c in EMBEDDERS}
    cro = {c["name"]: c for c in CROSS}
    if name in emb:
        s = embed_scores(emb[name], recs, dev)
        kind = "embed"
    else:
        s = cross_scores(cro[name], recs, dev)
        kind = cro[name]["kind"]
    m = metrics(s, labels)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    safe = name.replace("/", "__")
    np.save(SCORES_DIR / f"{safe}.npy", s)
    meta = {"name": name, "kind": kind, "auc": m["auc"], "hall_mean": m["hall_mean"],
            "supp_mean": m["supp_mean"], "best_at_recall85": m["best_at_recall85"]}
    (SCORES_DIR / f"{safe}.json").write_text(json.dumps(meta))
    return meta


if __name__ == "__main__":
    import sys
    meta = score_one(sys.argv[1])
    print(f"OK {meta['name']} auc={meta['auc']:.3f}")


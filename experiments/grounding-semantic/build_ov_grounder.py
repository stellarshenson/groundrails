"""Build the single-engine OpenVINO int8 grounder: embedder + reranker + NLI.

Writes push-ready IR dirs (IR + config + tokenizer) under models/ov/<name>/ and validates
each against the cached fp32 scores on a stratified sample. bge models use plain NNCF int8
(standard attention); mDeBERTa needs SmoothQuant (alpha 0.7) for its disentangled attention.

Usage: python scripts/build_ov_grounder.py
"""
import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import datasets
from scipy.stats import pearsonr
from transformers import AutoTokenizer

from grounding_models import load_gold, SCORES_DIR, ROOT
from grounding_openvino import (build_ov_int8, compile_ir, embed_vectors,
                                                 rerank_max, nli_max)

OUT = ROOT / "models" / "ov"
OUT.mkdir(parents=True, exist_ok=True)
N_CALIB, N_SAMPLE = 128, 200
recs = load_gold(); y = np.array([r["label"] for r in recs])
np.random.seed(42)
half = N_SAMPLE // 2
sidx = np.sort(np.concatenate([np.random.choice(np.where(y == 0)[0], half, replace=False),
                               np.random.choice(np.where(y == 1)[0], half, replace=False)]))
cidx = np.random.choice(len(recs), N_CALIB, replace=False)


def calib_pairs(model_id, order):
    tok = AutoTokenizer.from_pretrained(model_id)
    a = [recs[i]["claim"] if order == "claim_first" else recs[i]["chunks"][0] for i in cidx]
    b = [recs[i]["chunks"][0] if order == "claim_first" else recs[i]["claim"] for i in cidx]
    enc = tok(a, b, truncation=True, max_length=256, padding="max_length")
    return datasets.Dataset.from_dict({"input_ids": enc["input_ids"],
                                       "attention_mask": enc["attention_mask"]})


def calib_single(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    enc = tok([recs[i]["chunks"][0] for i in cidx], truncation=True, max_length=256,
              padding="max_length")
    return datasets.Dataset.from_dict({"input_ids": enc["input_ids"],
                                       "attention_mask": enc["attention_mask"]})


JOBS = [
    dict(name="bge-reranker-v2-m3", model_id="BAAI/bge-reranker-v2-m3", task="cls",
         smooth=None, calib=lambda mid: calib_pairs(mid, "claim_first"),
         cached="BAAI__bge-reranker-v2-m3.npy", kind="rerank"),
    dict(name="bge-m3", model_id="BAAI/bge-m3", task="feat",
         smooth=None, calib=calib_single,
         cached="BAAI__bge-m3.npy", kind="embed"),
    dict(name="mDeBERTa-v3-nli", model_id="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", task="cls",
         smooth=0.7, calib=lambda mid: calib_pairs(mid, "chunk_first"),
         cached="MoritzLaurer__mDeBERTa-v3-base-mnli-xnli.npy", kind="nli"),
]

for j in JOBS:
    print(f"===== {j['name']} ({j['kind']}) =====", flush=True)
    t = time.time()
    out_dir = OUT / j["name"]
    build_ov_int8(j["model_id"], j["task"], j["calib"](j["model_id"]), out_dir,
                  smooth_alpha=j["smooth"], n_calib=N_CALIB)
    tok = AutoTokenizer.from_pretrained(j["model_id"]); tok.save_pretrained(out_dir)  # push-ready
    size = sum(p.stat().st_size for p in out_dir.glob("*.bin")) / 1e6
    print(f"  built {size:.0f} MB ({time.time()-t:.0f}s)", flush=True)

    cm = compile_ir(out_dir / "openvino_model.xml")
    cached = np.load(SCORES_DIR / j["cached"])
    scores = np.zeros(len(sidx))
    for n, i in enumerate(sidx):
        r = recs[i]
        if j["kind"] == "rerank":
            scores[n] = rerank_max(cm, tok, r["claim"], r["chunks"])
        elif j["kind"] == "nli":
            scores[n] = nli_max(cm, tok, r["claim"], r["chunks"])
        else:
            cv = embed_vectors(cm, tok, [r["claim"]])[0]
            dv = embed_vectors(cm, tok, r["chunks"])
            scores[n] = float((dv @ cv).max())
    p = float(pearsonr(scores, cached[sidx])[0])
    print(f"  PARITY pearson={p:.4f}  std={scores.std():.3f}  "
          f"{'OK' if p >= 0.95 else 'WEAK'}", flush=True)

print("ALL DONE", flush=True)

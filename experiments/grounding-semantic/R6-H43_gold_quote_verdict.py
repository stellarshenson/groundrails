"""R6-H43 / R6-H45 on the verified gold - the model quotes, code decides.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 6).

R6-H42 validated the harness against Pleias' own published example, then showed
the model is a bad JUDGE on trivially separable pairs (11/20, 9 false positives)
and a good QUOTER (18/20, 1 false positive) once deterministic code owns the
verdict. This carries that result from a 20-case control to the real gold.

The verdict rule, unchanged from R6-H45:

  grounded  iff  the model emits an excerpt that is (a) genuinely present in the
                 supplied source AND (b) shares a numeric or entity anchor with
                 the claim

Both halves are string comparisons. The model never states a verdict, so a
fabricated verdict cannot enter - only a fabricated quote, which fails (a).

Retrieval matches the deployed cascade: `recursive_chunk` at cfg.chunk_max_chars
then the shipped bge-m3 bi-encoder's top cfg.semantic_top_k chunks, passed as the
protocol's numbered sources in ONE prompt. The shipped cascade is re-scored on
the SAME claims and the SAME chunks, so the comparison is matched rather than
quoted from a cache built on a different chunking.

Checkpoints every 25 claims so a killed run resumes instead of restarting.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R6-H43_gold_quote_verdict.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import importlib.util
import json
import pathlib
import re

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from groundrails import semantic_ov, settings
from groundrails.chunking import recursive_chunk
from groundrails.config import load_document_processing_config

settings.mark_ready()
CFG = load_document_processing_config()

HERE = pathlib.Path(__file__).parent
GOLD = HERE / "private-rag-forensics" / "gold" / "golden_grounding_evidence_verified.parquet"
CKPT = HERE / "private-rag-forensics" / "R6-H43_gold_generations.jsonl"
MID = "PleIAs/Pleias-RAG-350M"
N_CLAIMS = 600  # stratified subsample; ~1.5 s/claim generative
MAX_NEW = 400
CHANCE_F1 = 0.417  # naive majority baseline, carried from Methodology
EN_INCUMBENT_AUC = 0.7756

CITATION_RE = re.compile(r'<ref name="(?:<\|source_id\|>)?(\d+)">(.*?)</ref>', re.DOTALL)
ANSWER_RE = re.compile(r"<\|answer_start\|>(.*?)<\|answer_end\|>", re.DOTALL)
ANALYSIS_RE = re.compile(r"<\|source_analysis_start\|>(.*?)<\|source_analysis_end\|>", re.DOTALL)
PROMPT_ECHO_RE = re.compile(r"^\s*(?:claim|query|question)\s*:\s*", re.IGNORECASE)
FUZZ_FLOOR = 97.0


def _h45():
    """Reuse R6-H45's checker verbatim - one definition of the rule, not two."""
    spec = importlib.util.spec_from_file_location("h45", HERE / "R6-H45_quote_verdict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.quote_is_real, mod.quote_supports_claim


quote_is_real, quote_supports_claim = _h45()


def build_prompt(query, sources):
    """Verbatim from Pleias-RAG-Library RAGWithCitations.format_prompt."""
    prompt = f"<|query_start|>{query}<|query_end|>\n"
    for idx, text in enumerate(sources, 1):
        prompt += f"<|source_start|><|source_id|>{idx} {text}<|source_end|>\n"
    return prompt + "<|language_start|>\n"


def top_chunks(cascade, claim, source_text, k):
    """The deployed pre-filter: chunk, embed, take the top-k by cosine."""
    chunks = [c.text for c in (recursive_chunk(source_text, max_chars=CFG.chunk_max_chars) or [])]
    if not chunks:
        return [source_text[: CFG.chunk_max_chars]]
    if len(chunks) <= k:
        return chunks
    qv = cascade._embed([claim])
    cv = cascade._embed(chunks)
    sims = (cv @ qv.T).ravel()
    return [chunks[i] for i in np.argsort(-sims)[:k]]


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    g = pl.read_parquet(GOLD).filter(pl.col("lang").str.starts_with("en"))
    pos = g.filter(pl.col("label") == 1).sample(N_CLAIMS // 2, seed=0)
    neg = g.filter(pl.col("label") == 0).sample(N_CLAIMS // 2, seed=0)
    rows = pl.concat([pos, neg]).sample(fraction=1.0, shuffle=True, seed=1)
    print(
        f"EN gold: {len(g)} claims, stratified subsample {len(rows)} "
        f"(base rate {rows['label'].mean():.3f})",
        flush=True,
    )

    done = {}
    if CKPT.exists():
        for line in CKPT.read_text().splitlines():
            r = json.loads(line)
            done[r["claim"]] = r
        print(f"resuming: {len(done)} claims already generated", flush=True)

    cascade = semantic_ov.SemanticCascade()
    cascade._load()  # tokenizers/engines are lazy; _embed needs them resident
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModelForCausalLM.from_pretrained(MID).cuda().eval()

    fh = CKPT.open("a")
    for i, r in enumerate(rows.iter_rows(named=True)):
        if r["claim"] in done:
            continue
        chunks = top_chunks(cascade, r["claim"], r["source_text"], CFG.semantic_top_k)
        q = f"Is this claim supported by the sources? Claim: {r['claim']}"
        enc = tok(build_prompt(q, chunks), return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model.generate(
                enc.input_ids,
                max_new_tokens=MAX_NEW,
                do_sample=False,
                top_p=0.95,
                repetition_penalty=1.0,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][enc.input_ids.shape[1] :], skip_special_tokens=False)
        rec = {"claim": r["claim"], "label": int(r["label"]), "chunks": chunks, "raw": gen}
        done[r["claim"]] = rec
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i % 25 == 0:
            fh.flush()
            print(f"  {i}/{len(rows)} generated", flush=True)
    fh.close()

    y, quote_pred, judge_pred = [], [], []
    for r in rows.iter_rows(named=True):
        rec = done[r["claim"]]
        raw, src = rec["raw"], "\n".join(rec["chunks"])
        quotes = [q for _, q in CITATION_RE.findall(raw)]
        real = [
            q
            for q in quotes
            if quote_is_real(PROMPT_ECHO_RE.sub("", q), src)
            and quote_supports_claim(q, r["claim"])
        ]
        y.append(int(r["label"]))
        quote_pred.append(int(bool(real)))
        am = ANSWER_RE.search(raw)
        ans = am.group(1).lower() if am else ""
        neg_words = (
            "not supported",
            "does not",
            "no information",
            "cannot",
            "unsupported",
            "not mentioned",
            "not contain",
            "unrelated",
            "no mention",
        )
        judge_pred.append(int(not any(w in ans for w in neg_words)) if ans else 1)

    y = np.array(y)
    print("\n" + "=" * 92)
    print("R6-H43 / R6-H45 RESULT - verified gold, English slice")
    print("=" * 92)
    for name, p in (
        ("QUOTE verdict (code decides)", np.array(quote_pred)),
        ("JUDGE verdict (model decides)", np.array(judge_pred)),
    ):
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        print(
            f"  {name:32s} macro-F1 {f1_score(y, p, average='macro'):.4f}   "
            f"P {precision_score(y, p, zero_division=0):.3f} "
            f"R {recall_score(y, p, zero_division=0):.3f}   FP {fp}  FN {fn}"
        )
    print(f"  {'naive majority baseline':32s} macro-F1 {CHANCE_F1:.4f}")
    print(f"  incumbent EN NLI AUC (ranker, not a verdict): {EN_INCUMBENT_AUC}")
    print(f"\n  generations -> {CKPT}")


if __name__ == "__main__":
    main()

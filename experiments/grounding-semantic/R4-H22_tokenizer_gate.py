"""R4-H22 - representation feasibility gate for knowledge-free reasoning models.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 4).

Question: can a model whose 8,192-token BPE was trained exclusively on English
synthetic SYNTH text even REPRESENT our evidence? An 8k vocabulary trained on one
distribution shatters out-of-distribution words into many subword pieces; past a
certain fertility the evidence no longer fits the context window and the model
literally cannot see what it must ground against. That is a representation
failure, not a capability failure, and it is cheap to detect - so it runs before
any weights are downloaded.

PASS bar (fixed before the run): EN fertility <= 2.5x mDeBERTa AND EN context
overflow <= 20%. Failing either kills round 4 before a single forward pass.

Scope: English only. The round is testing whether the METHOD is load-bearing, not
multilingual coverage; the shipped argos MT bridge (lexical_mt.py) is the route
for non-EN if the method survives. Non-EN numbers are reported as information.

Run:  uv run python experiments/grounding-semantic/R4-H22_tokenizer_gate.py
"""

import json
from pathlib import Path
import re

import polars as pl
from transformers import AutoTokenizer

from groundrails import settings
from groundrails.chunking import recursive_chunk
from groundrails.config import load_document_processing_config

GOLD = Path("data/processed/golden_v6/golden_v6.parquet")
CANDIDATE = "PleIAs/Monad"  # 56.7M, ~8k BPE trained only on SYNTH
INCUMBENT = "microsoft/mdeberta-v3-base"  # 250k multilingual vocab, the deployed NLI
WORD_RE = re.compile(r"\w+", re.UNICODE)

settings.mark_ready()
CFG = load_document_processing_config()
TOP_K = CFG.semantic_top_k  # the pre-filter survivor count the cascade scores

FERTILITY_BAR = 2.5  # EN tokens/word ratio vs the incumbent
OVERFLOW_BAR = 0.20  # fraction of EN pairs exceeding the candidate context


def fertility(tok, texts):
    """Mean tokens per whitespace-ish word - the representation-efficiency unit."""
    n_tok = sum(len(tok.encode(t, add_special_tokens=False)) for t in texts)
    n_word = sum(len(WORD_RE.findall(t)) for t in texts)
    return n_tok / max(n_word, 1), n_tok, n_word


def main():
    g = pl.read_parquet(GOLD)
    print(f"gold: {g.height} rows, {g['lang'].n_unique()} languages")

    tok_c = AutoTokenizer.from_pretrained(CANDIDATE)
    tok_i = AutoTokenizer.from_pretrained(INCUMBENT)
    ctx = getattr(tok_c, "model_max_length", None)
    if ctx is None or ctx > 100_000:  # HF sentinel for "unset"
        cfg = json.loads(
            (Path(tok_c.name_or_path) / "config.json").read_text()
            if Path(tok_c.name_or_path).is_dir()
            else "{}"
        )
        ctx = cfg.get("max_position_embeddings", 2048)
    print(f"candidate={CANDIDATE} vocab={tok_c.vocab_size} ctx={ctx}")
    print(f"incumbent={INCUMBENT} vocab={tok_i.vocab_size}")
    print(f"serving unit: claim + one {CFG.chunk_max_chars}-char chunk, top-{TOP_K}\n")

    rows = []
    for lang in ["en"] + [x for x in ("fr", "es", "it", "nb", "de") if x in set(g["lang"])]:
        sub = g.filter(pl.col("lang") == lang)
        if sub.height < 50:
            continue
        # Fertility is a property of the text, so it is measured on the raw evidence.
        raw = (sub["claim"] + " " + sub["source_text"]).to_list()
        f_c, ntok_c, _ = fertility(tok_c, raw)
        f_i, _, _ = fertility(tok_i, raw)
        # Overflow MUST be measured on the unit an engine actually receives. The raw
        # source_text is the whole retrieved blob (median ~37.5k chars); the pipeline
        # recursive-chunks it at cfg.chunk_max_chars and scores the top-k, so a
        # cross-encoder never sees more than one chunk beside the claim. Measuring the
        # blob instead falsely fails EVERY model including the deployed 512-token
        # mDeBERTa, which ships - that mistake killed the first run of this gate.
        pairs = [
            f"{c} {ch.text}"
            for c, s in zip(sub["claim"].to_list(), sub["source_text"].to_list(), strict=True)
            for ch in (recursive_chunk(s, max_chars=CFG.chunk_max_chars) or [])[:TOP_K]
        ]
        lens = [len(tok_c.encode(t, add_special_tokens=True)) for t in pairs]
        overflow = sum(1 for x in lens if x > ctx) / max(len(lens), 1)
        rows.append(
            {
                "lang": lang,
                "n": sub.height,
                "cand_tok_per_word": round(f_c, 3),
                "incumbent_tok_per_word": round(f_i, 3),
                "ratio": round(f_c / f_i, 3),
                "median_len": int(sorted(lens)[len(lens) // 2]),
                "p95_len": int(sorted(lens)[int(len(lens) * 0.95)]),
                "overflow_frac": round(overflow, 4),
            }
        )

    tbl = pl.DataFrame(rows)
    print(tbl)

    en = next(r for r in rows if r["lang"] == "en")
    print(f"\n--- GATE (English, the declared scope) ---")
    print(f"  fertility ratio {en['ratio']:.3f}  bar <= {FERTILITY_BAR}   "
          f"{'PASS' if en['ratio'] <= FERTILITY_BAR else 'FAIL'}")
    print(f"  context overflow {en['overflow_frac']:.1%}  bar <= {OVERFLOW_BAR:.0%}   "
          f"{'PASS' if en['overflow_frac'] <= OVERFLOW_BAR else 'FAIL'}")
    verdict = en["ratio"] <= FERTILITY_BAR and en["overflow_frac"] <= OVERFLOW_BAR
    print(f"\n  R4-H22 VERDICT: {'PASS - round 4 proceeds to R4-H23' if verdict else 'KILLED-AT-GATE - round 4 stops here'}")


if __name__ == "__main__":
    main()

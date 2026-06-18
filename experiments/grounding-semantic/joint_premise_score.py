"""Round 4: joint-premise (SummaC) NLI signals over the gold v3 cascade-fired rows.

The shipped cascade computes `nli_ent` / `nli_contra` as max-over-chunks - each 1100-char
chunk graded against the claim independently, the max taken. The SummaC pattern (Laban et al.
2022), used by the docdistance grounding axis on the identical int8 reranker+NLI stack, instead
joins the top-k aligning *statements* into one premise so a claim that fuses several source
sentences is graded as supported. groundrails chunks are ~319 NLI tokens each, so three joined
chunks (~887) overflow the 512 window - the faithful unit is the sentence, not the chunk.

This recomputes, per cascade-fired row (`ran_nli=True`), three NLI readings at sentence
granularity so the joint-premise mechanism is isolated from the granularity change:
  - sentence-max : max entailment / contradiction over the top-3 cosine-ranked source sentences
  - joint        : entailment / contradiction of those same top-3 sentences joined into one premise
The cached chunk-max stays the shipped baseline. Only `ran_nli=True` rows are touched (3,215 of
7,976); gate/band-skipped rows keep nli=0 in every variant, so they need no recompute.

Reuses the live `SemanticCascade` (`_embed`, `_nli_channels_max`); premise=evidence,
hypothesis=claim, matching the shipped direction. Resumable, checkpointed, torch-free OV int8.
Output: data/processed/golden_v3_joint_premise.parquet (keyed by uid).

Run:  uv run python experiments/grounding-semantic/joint_premise_score.py
"""

from __future__ import annotations

import os
from pathlib import Path
import re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from grounding_models import chunk_text  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from groundrails.semantic_ov import (  # noqa: E402
    MAX_LEN,
    SemanticCascade,
    _feed,
    install_hint,
    is_available,
)

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "data" / "processed" / "golden_v3.parquet"
AUG = ROOT / "data" / "processed" / "golden_v3_synth_aug.parquet"
CAS = ROOT / "data" / "processed" / "golden_v3_cascade_scores.parquet"
OUT = ROOT / "data" / "processed" / "golden_v3_joint_premise.parquet"

KEEP = ["uid", "role", "claim", "source_text"]
COLS = ["uid", "role", "nli_ent_sentmax", "nli_contra_sentmax",
        "nli_ent_joint", "nli_contra_joint", "n_top"]
TOP_K = 3        # statements joined into the premise
CAND_K = 8       # cosine-prefilter sentences the reranker re-ranks (mirrors the cascade top_k)
CHUNK_CAND = 8   # cascade chunk prefilter - only these chunks are segmented (bounds the embed cost)
CHECKPOINT = 200
_SENT = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT.split(str(text)) if s.strip()]
    return parts or [str(text).strip()]


def rerank_scores(eng, claim, chunks):
    """Per-chunk reranker relevance (sigmoid of the logit) - the cascade keeps only the max."""
    cm, tok = eng._rr, eng._rtok
    enc = tok([claim] * len(chunks), list(chunks), padding=True, truncation=True,
              max_length=MAX_LEN, return_tensors="np")
    lg = cm(_feed(cm, enc))[cm.output(0)].reshape(-1)
    return 1.0 / (1.0 + np.exp(-lg))


def candidate_sentences(eng, claim, source):
    """Segment only the cascade's top-CHUNK_CAND cosine chunks into sentences - bounds the embed
    cost on very long sources (some run to 1200+ sentences) to the evidence region the cascade
    itself used."""
    chunks = chunk_text(source)
    if len(chunks) > CHUNK_CAND:
        cv = eng._embed([claim])[0]
        dv = eng._embed(chunks)
        chunks = [chunks[j] for j in np.argsort(-(dv @ cv))[:CHUNK_CAND]]
    out = []
    for c in chunks:
        out.extend(sentences(c))
    return out or [str(source)]


def top_statements(eng, claim, source):
    """Top-K most relevant source sentences: chunk prefilter -> sentence cosine prefilter -> rerank."""
    sents = candidate_sentences(eng, claim, source)
    if len(sents) <= TOP_K:
        return sents
    cv = eng._embed([claim])[0]
    sv = eng._embed(sents)
    cand = [sents[j] for j in np.argsort(-(sv @ cv))[:CAND_K]]
    rr = rerank_scores(eng, claim, cand)
    return [cand[j] for j in np.argsort(-rr)[:TOP_K]]


def main() -> None:
    if not is_available():
        raise SystemExit(install_hint())
    src = pd.concat(
        [pd.read_parquet(GOLDEN)[KEEP], pd.read_parquet(AUG)[KEEP]], ignore_index=True
    )
    cas = pd.read_parquet(CAS)[["uid", "ran_nli"]]
    fired = cas[cas["ran_nli"].astype(bool)]["uid"]
    df = src[src["uid"].isin(set(fired))].reset_index(drop=True)

    rows, done = [], set()
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        rows = prev[COLS].values.tolist()
        done = set(prev["uid"])
        print(f"resume: {len(done)} already scored", flush=True)
    todo = df[~df["uid"].isin(done)]
    n = len(todo)
    print(f"joint-premise NLI over {n} of {len(df)} cascade-fired rows (top-{TOP_K} sentences)", flush=True)

    eng = SemanticCascade()
    eng._load()
    for i, r in enumerate(todo.itertuples(index=False)):
        top = top_statements(eng, r.claim, r.source_text)            # top-3 reranked statements
        ent_sm, con_sm = eng._nli_channels_max(r.claim, top)          # max over those statements
        ent_j, con_j = eng._nli_channels_max(r.claim, [" ".join(top)])  # the same statements, joined
        rows.append([r.uid, r.role, ent_sm, con_sm, ent_j, con_j, len(top)])
        if (i + 1) % CHECKPOINT == 0:
            pd.DataFrame(rows, columns=COLS).to_parquet(OUT, index=False)
            print(f"  scored {i + 1}/{n} (checkpoint)", flush=True)

    out = pd.DataFrame(rows, columns=COLS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(out)} rows)", flush=True)
    d_ent = out["nli_ent_joint"] - out["nli_ent_sentmax"]
    print(f"  joint - sentence-max entailment: mean {d_ent.mean():+.3f}, "
          f">+0.10 on {(d_ent > 0.10).mean():.1%} of fired rows", flush=True)
    print(f"  by role: {out.groupby('role').size().to_dict()}", flush=True)


if __name__ == "__main__":
    main()

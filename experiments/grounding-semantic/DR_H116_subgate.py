"""DR-H116 LONG-HIER splice-only sub-gate (registration: DR-2, semantic-dataset-enhancements.md).

Measures ONLY what the DR-2 registration puts in the sub-gate: splice integrity
and degeneracy. No judge, no NLI.

Build: ~150 long responses (256-2048 mBART tokens) assembled from the H111 public
seed pool, segmented with pysbd; K spans per doc drawn from the RAGTruth
per-response span-count histogram capped at ceil(n_sentences/3); span-bearing
sentence positions drawn from the registered char-offset quantiles
0.18/0.29/0.47/0.71/0.85; within-sentence (position, length) from the shared
targeting module (core loci only); DR-H112 engine on the selected spans; every
other character spliced back verbatim; exact char-offset ledger.

KILL bars:
  (1) splice integrity - text outside the edited spans char-exact vs the
      assembled original; KILL if < 100% of docs
  (2) doc-level degeneracy (share of docs with >= 1 degenerate edited span - the
      compounding measure 1-(1-g)^K) > 2 x the H112 engine bar (> 12.4%)

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/DR_H116_subgate.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import math
from pathlib import Path
import random
import sys
import time

import polars as pl
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import DR_pilot_engines as ENG  # noqa: E402

PAIRS = HERE / "R10-H111_stage1_pairs.parquet"
OUT_JSON = HERE / "DR_H116_subgate_result.json"
OUT_DOCS = HERE / "DR_H116_subgate_docs.parquet"
OUT_SPANS = HERE / "DR_H116_subgate_spans.parquet"

SEED = 0
N_DOCS = 150
DEGEN_KILL = 0.124  # 2 x the measured H112 engine debris bar (6.2%)

log = ENG.log


def build_pools(tok, seed_slice: slice | None = None):
    """register -> [(seed_text, seed_id, n_tok)], seed_id = stable pool index."""
    df = (pl.read_parquet(PAIRS).select("seed", "tag")
          .unique(subset=["seed"], maintain_order=True))
    seeds = df["seed"].to_list()
    tags = df["tag"].to_list()
    lens = ENG.precount_tokens(tok, seeds)
    order = list(range(len(seeds)))
    random.Random(SEED).shuffle(order)
    if seed_slice is not None:
        order = order[seed_slice]
    pools: dict[str, list[tuple]] = {}
    for i in order:
        pools.setdefault(tags[i].rsplit("_", 1)[-1], []).append(
            (seeds[i], i, lens[i]))
    return pools


def edit_docs(tok, model, docs, hist, rng, seg):
    """Apply the H112 engine to K selected sentences per doc; splice char-exact."""
    ks = list(hist)
    kw = [hist[k] for k in ks]
    doc_rows, span_rows, attempts = [], [], 0
    for d in docs:
        text = d["text"]
        sents = [(s.start, s.end) for s in seg.segment(text)
                 if text[s.start:s.end].strip()]
        if len(sents) < 2:
            continue
        k = min(int(rng.choices(ks, weights=kw)[0]),
                max(1, math.ceil(len(sents) / 3)))
        edits = []
        for si in ENG.pick_sentences(sents, len(text), k, rng):
            s0, s1 = sents[si]
            sent = text[s0:s1]
            got = ENG.core_spans(sent, 1, random.Random(rng.getrandbits(31)))
            if not got:
                continue
            c0, c1, stext, ltype, source = got[0]
            attempts += 1
            res = ENG.infill_span(tok, model, sent, c0, c1, stext)
            if res is None:
                continue
            edits.append({
                "abs_start": s0 + c0, "abs_end": s0 + c1,
                "new_span": res["decoded_span"], "orig_span": stext,
                "sent_index": si, "sent_start": s0, "sent_end": s1,
                "sent_c0": c0, "sent_c1": c1, "sent_claim": res["claim"],
                "locus_type": ltype, "source": source,
                "degen": res["degen"], "evasion": res["evasion"],
            })
        if not edits:
            continue
        corrupt, ledger, intact = ENG.splice_doc(text, edits)
        doc_rows.append({
            "doc_id": d["doc_id"], "register": d["register"], "n_tok": d["n_tok"],
            "n_sent": len(sents), "k_drawn": k, "n_spans": len(ledger),
            "splice_intact": intact,
            "doc_degen": any(e["degen"] for e in ledger),
            "doc_clean": text, "doc_corrupt": corrupt,
        })
        for e in ledger:
            span_rows.append({
                "doc_id": d["doc_id"], "register": d["register"],
                "sent_index": e["sent_index"], "sent_start": e["sent_start"],
                "sent_end": e["sent_end"],
                "seed": text[e["sent_start"]:e["sent_end"]],
                "claim": e["sent_claim"], "span_start": e["sent_c0"],
                "span_end": e["sent_c0"] + len(e["new_span"]),
                "orig_span": e["orig_span"], "new_span": e["new_span"],
                "locus_type": e["locus_type"], "source": e["source"],
                "degen": e["degen"], "evasion": e["evasion"],
                "doc_span_start": e["doc_span_start"],
                "doc_span_end": e["doc_span_end"],
                "orig_doc_start": e["abs_start"], "orig_doc_end": e["abs_end"],
                "splice_intact": intact,
            })
    return doc_rows, span_rows, attempts


def main():
    t_start = time.time()
    random.seed(SEED)
    torch.manual_seed(SEED)
    import pysbd

    hist = ENG.span_count_hist()
    log(f"RAGTruth per-response span-count histogram (head): "
        f"{ {k: round(v, 3) for k, v in list(hist.items())[:6]} }")

    tok, model = ENG.S0.load_mbart()
    model.eval()

    rng = random.Random(SEED)
    pools = build_pools(tok)
    log(f"seed pool by register: { {k: len(v) for k, v in pools.items()} }")
    docs = ENG.assemble_docs(pools, rng, N_DOCS)
    log(f"assembled {len(docs)} docs, tokens "
        f"{min(d['n_tok'] for d in docs)}-{max(d['n_tok'] for d in docs)}")

    seg = pysbd.Segmenter(language="en", clean=False, char_span=True)
    t0 = time.time()
    doc_rows, span_rows, attempts = edit_docs(tok, model, docs, hist, rng, seg)
    log(f"{len(doc_rows)} docs edited, {len(span_rows)} spans "
        f"in {time.time() - t0:.0f}s")

    del model
    torch.cuda.empty_cache()

    n_doc = max(len(doc_rows), 1)
    n_span = max(len(span_rows), 1)
    intact_rate = sum(r["splice_intact"] for r in doc_rows) / n_doc
    doc_degen_rate = sum(r["doc_degen"] for r in doc_rows) / n_doc
    span_degen_rate = sum(r["degen"] for r in span_rows) / n_span
    spd = len(span_rows) / n_doc
    # informational second reading: under the registered failed-span policy a
    # degenerate span reverts to its clean sentence, so a doc is only lost when
    # EVERY edited span is degenerate
    by_doc: dict[str, list[bool]] = {}
    for r in span_rows:
        by_doc.setdefault(r["doc_id"], []).append(r["degen"])
    doc_degen_all = sum(1 for v in by_doc.values() if all(v)) / n_doc
    per_locus = {}
    for lt in sorted({r["locus_type"] for r in span_rows}):
        sub = [r for r in span_rows if r["locus_type"] == lt]
        per_locus[lt] = {"n": len(sub),
                         "degen": round(sum(r["degen"] for r in sub) / len(sub), 4)}

    kill_splice = intact_rate < 1.0
    kill_degen = doc_degen_rate > DEGEN_KILL
    result = {
        "n_docs": len(doc_rows), "n_spans": len(span_rows),
        "span_attempts": attempts,
        "tokens_min": min((r["n_tok"] for r in doc_rows), default=0),
        "tokens_max": max((r["n_tok"] for r in doc_rows), default=0),
        "tokens_mean": round(sum(r["n_tok"] for r in doc_rows) / n_doc, 1),
        "sent_mean": round(sum(r["n_sent"] for r in doc_rows) / n_doc, 2),
        "spans_per_doc_mean": round(spd, 3),
        "splice_intact_rate": round(intact_rate, 6),
        "doc_degen_rate": round(doc_degen_rate, 4),
        "span_degen_rate": round(span_degen_rate, 4),
        "doc_degen_expected_from_span_rate": round(1 - (1 - span_degen_rate) ** spd, 4),
        "doc_degen_all_spans": round(doc_degen_all, 4),
        "per_locus_span_degen": per_locus,
        "h112_gate_reference": {"pooled_span_degen": 0.0621,
                                "core_loci_span_degen": 0.0944},
        "bars": {"splice_intact_rate_min": 1.0, "doc_degen_rate_max": DEGEN_KILL},
        "bar_note": ("doc_degen_rate = share of docs with >=1 degenerate edited "
                     "span, the compounding 1-(1-g)^K reading of the H116 skeptic "
                     "amendment; doc_degen_all_spans is the alternative reading "
                     "under the failed-span revert policy - reported, not gated"),
        "kill_splice": kill_splice, "kill_degen": kill_degen,
        "verdict": "KILL" if (kill_splice or kill_degen) else "SURVIVES",
        "runtime_s": round(time.time() - t_start, 1),
    }
    pl.DataFrame(doc_rows).write_parquet(OUT_DOCS)
    pl.DataFrame(span_rows).write_parquet(OUT_SPANS)
    OUT_JSON.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)
    log(f"result -> {OUT_JSON}")


if __name__ == "__main__":
    main()

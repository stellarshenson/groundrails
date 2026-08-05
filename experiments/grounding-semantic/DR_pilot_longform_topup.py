"""DR-2 long-form top-up (main-session adjudication: DR-H116 SURVIVES).

The pilot run (`DR_pilot_gen.py`) shipped its whole 31k H112 budget sentence-level
because the executor read the DR-H116 sub-gate as KILL on the compounding
doc-degeneracy metric. The main session adjudicated SURVIVES under the registered
failed-span revert policy, so this driver mints the missing 20% share of the H112
budget as long-form documents.

Reuses `DR_pilot_gen.gen_h112_longform` unchanged in mechanism: 256-2048-token
pysbd documents assembled from the H112 seed slice, K spans per doc from the
RAGTruth per-response histogram capped at ceil(n_sentences/3), span-bearing
sentence positions from the registered char-offset quantiles, core loci only,
DR-H112 engine per selected sentence, every other character spliced back verbatim
with an exact char-offset ledger.

Guarantees:
- H112's seed slice only - the H113 / H114 slices are never touched
- dedup against every dedup_key already in DR_pilot_raw.parquet
- splice integrity re-verified INDEPENDENTLY from the written rows
- DR_pilot_raw.parquet is never modified

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/DR_pilot_longform_topup.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import json
from pathlib import Path
import random
import sys
import time

import polars as pl
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import DR_pilot_engines as ENG  # noqa: E402
import DR_pilot_gen as P  # noqa: E402

RAW = HERE / "DR_pilot_raw.parquet"
OUT = HERE / "DR_pilot_longform.parquet"
CKPT = HERE / "DR_pilot_longform.parquet.ckpt"
SUMMARY = HERE / "DR_pilot_longform_summary.json"

TARGET = int(P.N_H112 * P.LONGFORM_SHARE)  # 6,200 = 20% of the 31k H112 budget
log = ENG.log


def verify_splice(df: pl.DataFrame) -> dict:
    """Independent re-verification: rebuild each clean doc from the ledger."""
    ok, bad, checked = 0, [], 0
    for doc_id, sub in df.group_by("doc_id", maintain_order=True):
        sub = sub.sort("doc_span_start")
        clean = sub["doc_clean"][0]
        corrupt = sub["doc_corrupt"][0]
        parts, prev = [], 0
        for r in sub.iter_rows(named=True):
            parts.append(corrupt[prev:r["doc_span_start"]])
            parts.append(r["orig_span"])
            prev = r["doc_span_end"]
        parts.append(corrupt[prev:])
        checked += 1
        if "".join(parts) == clean:
            ok += 1
        elif len(bad) < 100:
            bad.append(doc_id[0] if isinstance(doc_id, tuple) else doc_id)
    # every corrupted span must sit exactly where the ledger says it does
    span_ok = sum(1 for r in df.iter_rows(named=True)
                  if r["doc_corrupt"][r["doc_span_start"]:r["doc_span_end"]]
                  == r["new_span"])
    return {"docs_checked": checked, "docs_char_exact": ok,
            "docs_char_exact_rate": round(ok / max(checked, 1), 6),
            "span_offsets_exact": span_ok, "n_spans": df.height,
            "span_offsets_exact_rate": round(span_ok / max(df.height, 1), 6),
            "failing_doc_ids": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    target = 40 if args.smoke else TARGET

    t_start = time.time()
    random.seed(P.SEED)
    torch.manual_seed(P.SEED)

    sub = json.loads((HERE / "DR_H116_subgate_result.json").read_text())
    log(f"DR-H116 verdict {sub['verdict']} (executor {sub.get('verdict_executor')}) "
        f"- {sub.get('adjudication', '')}")
    if sub["verdict"] != "SURVIVES":
        log("sub-gate is not SURVIVES - refusing to spend"); return

    raw = pl.read_parquet(RAW, columns=["dedup_key", "engine", "long_form"])
    skip_keys = set(raw["dedup_key"].to_list())
    log(f"{raw.height} rows already in DR_pilot_raw.parquet "
        f"({raw.filter(pl.col('long_form')).height} long-form); "
        f"{len(skip_keys)} dedup keys loaded")

    seeds, chunks, tags, order = P.load_pool()
    ids112 = order[:int(len(order) * 0.46)]  # H112 slice, exactly as the pilot cut it
    if args.smoke:
        ids112 = ids112[:3000]
    log(f"H112 seed slice: {len(ids112)} seeds; target {target} long-form spans")

    # the Sink checkpoints to the module-level CKPT path - repoint it so the
    # pilot's own checkpoint is never touched
    P.CKPT = CKPT
    sink = P.Sink()

    tok, model = ENG.S0.load_mbart()
    model.eval()
    # ~1.59 shipped spans/doc measured (colliding candidates are resampled, not
    # dropped, so the yield per doc holds); the loop breaks the moment target is met
    made, n_docs, splice_rejects, n_skipped = P.gen_h112_longform(
        tok, model, sink, seeds, chunks, tags, target,
        ids=ids112, skip_keys=skip_keys, spans_per_doc=1.5)
    sink.flush()
    del model
    torch.cuda.empty_cache()

    df = pl.DataFrame(sink.rows, infer_schema_length=None)
    if not df.height:
        log("no rows generated"); return
    df = df.with_columns(
        pl.concat_str([pl.col("doc_id"), pl.col("span_start").cast(pl.Utf8),
                       pl.col("new_span")], separator="|").alias("dedup_key"))
    before = df.height
    df = df.unique(subset=["dedup_key"], keep="first", maintain_order=True)
    log(f"internal dedup: {before} -> {df.height}")
    assert not set(df["dedup_key"].to_list()) & skip_keys, "collision with the pilot lane"

    splice = verify_splice(df)
    log(f"splice re-verification: {splice['docs_char_exact']}/{splice['docs_checked']} "
        f"docs char-exact, {splice['span_offsets_exact']}/{splice['n_spans']} "
        f"span offsets exact")
    if splice["docs_char_exact_rate"] < 1.0 or splice["span_offsets_exact_rate"] < 1.0:
        # a doc whose ledger does not rebuild the clean text carries an
        # unledgered corruption - refuse to publish anything
        fail = HERE / "DR_pilot_longform.FAILED.parquet"
        df.write_parquet(fail)
        log(f"SPLICE RECHECK FAILED - nothing published; rows -> {fail}")
        print(json.dumps({"verdict": "SPLICE_RECHECK_FAILED",
                          "splice_integrity_recheck": splice}, indent=1), flush=True)
        print("=== DR LONGFORM TOPUP DONE ===", flush=True)
        return
    df.write_parquet(CKPT)

    pairs = list(zip(df["seed"].to_list(), df["claim"].to_list()))
    log(f"NLI forward on {len(pairs)} pairs ...")
    _, fwd = ENG.S0.nli_entail(pairs)
    log("NLI backward ...")
    _, bwd = ENG.S0.nli_entail([(b, a) for a, b in pairs])
    df = df.with_columns(pl.Series("nli_fwd", [float(x) for x in fwd]),
                         pl.Series("nli_bwd", [float(x) for x in bwd]))
    df.write_parquet(OUT)
    log(f"wrote {df.height} rows -> {OUT}")

    doc_stats = (df.group_by("doc_id")
                 .agg(pl.len().alias("n_spans"),
                      pl.col("degen").sum().alias("n_degen"),
                      pl.col("doc_clean").first().str.len_chars().alias("chars")))
    n_doc = doc_stats.height
    summary = {
        "target_spans": target,
        "generated_spans": made,
        "rows_after_dedup": df.height,
        "dedup_skips_vs_pilot_lane": n_skipped,
        "docs": n_doc,
        "splice_rejects_during_generation": splice_rejects,
        "splice_integrity_recheck": splice,
        "per_doc": {
            "spans_mean": round(float(doc_stats["n_spans"].mean()), 3),
            "spans_max": int(doc_stats["n_spans"].max()),
            "chars_mean": round(float(doc_stats["chars"].mean()), 1),
            "chars_min": int(doc_stats["chars"].min()),
            "chars_max": int(doc_stats["chars"].max()),
            "doc_degen_any": round(
                doc_stats.filter(pl.col("n_degen") > 0).height / n_doc, 4),
            "doc_degen_all_spans": round(
                doc_stats.filter(pl.col("n_degen") == pl.col("n_spans")).height / n_doc, 4),
        },
        "per_span": {
            "degen": round(float(df["degen"].mean()), 4),
            "degeneracy_pass": round(float(df["degeneracy_pass"].mean()), 4),
            "evasion": round(float(df["evasion"].mean()), 4),
            "usable": round(float(df["usable"].mean()), 4),
            "nli_fwd_ge08": round(
                df.filter(pl.col("nli_fwd") >= 0.8).height / df.height, 4),
            "locus_mix": {k: v for k, v in
                          df["locus_type"].value_counts().sort("locus_type").iter_rows()},
            "register_mix": {k: v for k, v in
                             df["register"].value_counts().sort("register").iter_rows()},
        },
        "bars": {"h112_debris_kill": 0.124,
                 "debris_kill": bool(1 - float(df["degeneracy_pass"].mean()) > 0.124)},
        "runtime_s": round(time.time() - t_start, 1),
        "judge_pass": "pending - GPU1, not run here",
    }
    SUMMARY.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    log(f"summary -> {SUMMARY}")
    print("=== DR LONGFORM TOPUP DONE ===", flush=True)


if __name__ == "__main__":
    main()

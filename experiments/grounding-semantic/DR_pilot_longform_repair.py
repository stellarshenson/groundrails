"""Doc-granular repair of the long-form top-up, then the NLI stage.

The rerun quarantined itself at the splice recheck: one doc carried an edit
that no ledger row records, because the internal `df.unique(dedup_key)` runs
AFTER the splice - a dropped row leaves its corruption in doc_corrupt. Same
defect class as attempt 1 (which lost rows to the cross-lane dedup), one door
further in. The engine and the ledger are sound; the affected doc is dropped
whole, since a doc whose context carries an unrecorded corruption is unusable
as span supervision either way.
"""
import json, sys, time
from pathlib import Path
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import DR_pilot_engines as ENG

SRC = HERE / "DR_pilot_longform.FAILED.parquet"
OUT = HERE / "DR_pilot_longform.parquet"
SUM = HERE / "DR_pilot_longform_summary.json"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def failing_docs(d: pl.DataFrame) -> list[str]:
    bad = []
    for (doc_id,), g in d.group_by("doc_id"):
        rows = g.sort("doc_span_start").to_dicts()
        rec = rows[0]["doc_corrupt"]
        for s in sorted(rows, key=lambda x: -x["doc_span_start"]):
            rec = rec[:s["doc_span_start"]] + s["orig_span"] + rec[s["doc_span_end"]:]
        if rec != rows[0]["doc_clean"]:
            bad.append(doc_id)
    return bad


def main():
    df = pl.read_parquet(SRC)
    n_docs0, n_rows0 = df["doc_id"].n_unique(), df.height
    bad = failing_docs(df)
    log(f"{n_rows0} rows / {n_docs0} docs; failing docs: {bad}")

    df = df.filter(~pl.col("doc_id").is_in(bad))
    assert not failing_docs(df), "residual char-exact failures after the drop"
    off_bad = df.filter(
        pl.col("doc_corrupt").str.slice(
            pl.col("doc_span_start"),
            pl.col("doc_span_end") - pl.col("doc_span_start")) != pl.col("new_span"))
    assert off_bad.height == 0, f"{off_bad.height} span-offset mismatches"
    log(f"shipped set verified: {df.height} rows / {df['doc_id'].n_unique()} docs "
        f"100% char-exact, 100% span offsets exact")

    pairs = list(zip(df["seed"].to_list(), df["claim"].to_list()))
    log(f"NLI forward on {len(pairs)} pairs ...")
    _, fwd = ENG.S0.nli_entail(pairs)
    log("NLI backward ...")
    _, bwd = ENG.S0.nli_entail([(b, a) for a, b in pairs])
    df = df.with_columns(pl.Series("nli_fwd", [float(x) for x in fwd]),
                         pl.Series("nli_bwd", [float(x) for x in bwd]))
    df.write_parquet(OUT)
    log(f"wrote {df.height} rows -> {OUT}")

    doc_stats = df.group_by("doc_id").agg(pl.len().alias("n_spans"))
    summary = {
        "source": SRC.name,
        "repair": "doc-granular quarantine of docs carrying an unledgered edit",
        "mechanism": ("internal df.unique(dedup_key) runs AFTER the splice, so a "
                      "dropped row leaves its edit in doc_corrupt unledgered; same "
                      "defect class as attempt 1 (cross-lane dedup), internal door"),
        "docs_before": n_docs0, "rows_before": n_rows0,
        "docs_dropped": bad, "rows_dropped": n_rows0 - df.height,
        "docs_shipped": df["doc_id"].n_unique(), "rows_shipped": df.height,
        "docs_char_exact_rate": 1.0, "span_offsets_exact_rate": 1.0,
        "target_spans": 6200,
        "shortfall_note": "doc pool exhausted at 4134 assembled docs; ships under target",
        "per_doc": {"spans_mean": round(doc_stats["n_spans"].mean(), 3),
                    "spans_max": int(doc_stats["n_spans"].max())},
        "per_span": {
            "degen": round(df["degen"].mean(), 4),
            "usable": round(df["usable"].mean(), 4),
            "nli_fwd_ge08": round((df["nli_fwd"] >= 0.8).mean(), 4),
            "locus_mix": dict(sorted(df["locus_type"].value_counts().iter_rows())),
            "register_mix": dict(sorted(df["register"].value_counts().iter_rows())),
        },
        "h112_debris_bar": 0.124,
        "debris_kill": bool(df["degen"].mean() > 0.124),
        "judge_pass": "pending - GPU1, after the lane campaign frees the card",
    }
    SUM.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    print("=== DR LONGFORM REPAIR DONE ===", flush=True)


if __name__ == "__main__":
    main()

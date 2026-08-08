"""R13-H124 WINDOW-CONSENSUS-EVIDENCE-READ - offline aggregator amendment.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 13);
full record with binding amendments in R13_synthesis.md section 4 (R1).

Claim: within-chunk max-over-windows lets a single spuriously high window set the
chunk score. At stride 750 in a 1,500-char window genuine support appears in at
least two windows by construction, a spurious maximum does not. Replace the
within-chunk max with the MEAN OF THAT CHUNK'S TOP-2 WINDOWS (single-window
chunks fall back to max); max over chunks and min over sentences unchanged.

Zero GPU: recomputed from the frozen per-(sentence, chunk, window) matrices
dumped by R13_reads_dump.py.

Sanity guard (Gate A precedent, amendment 6): for every checkpoint the STANDARD
aggregation is rebuilt from the matrix first and must reproduce the banked
windowed result to 4 dp on all 10 subsets. A mismatch VOIDS that checkpoint.

Bar (ruling 7, subset-primary): hagrid >= +0.010 on BOTH H108 draws
AND mean delta >= -0.002 (HOLD) AND no subset < -0.02. Both H105 draws are read
as replication and are reported, not bar-bearing.

Run:  uv run python experiments/grounding-semantic/R13-H124_consensus.py
"""

import importlib.util
import json
import pathlib

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
RESULT = HERE / "R13-H124_result.json"


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M59 = _mod("m59", "R7-H59_cross_domain_matrix.py")

BANKED = {
    "h105d1": "R9-H105_windowed_result.json",
    "h105d2": "R9-H105_draw2_windowed_result.json",
    "h108d1": "R10-H108_lane_draw1_windowed_result.json",
    "h108d2": "R10-H108_lane_draw2_windowed_result.json",
}
PRIMARY = ("h108d1", "h108d2")
REPLICATION = ("h105d1", "h105d2")


def response_auc(sent_scores, df):
    """min over sentences -> response score -> AUC against adherence.

    `sent_scores` carries one row per (subset, resp_idx, sent_idx) with column
    `s`; labels come from the dump's own `resp_label` (the arena
    `adherence_score` the banked reads scored).
    """
    out = {}
    for sub in sorted(df["subset"].unique().to_list()):
        r = (
            sent_scores.filter(pl.col("subset") == sub)
            .group_by("resp_idx")
            .agg(pl.col("s").min())
            .sort("resp_idx")
        )
        y = (
            df.filter(pl.col("subset") == sub)
            .group_by("resp_idx")
            .agg(pl.col("resp_label").first())
            .sort("resp_idx")["resp_label"]
            .to_numpy()
        )
        auc, _, _ = M59.auc_and_f1(y, r["s"].to_numpy())
        out[sub] = float(auc)
    return out


def standard(df):
    """max over ALL windows per sentence (the shipped read)."""
    return (
        df.group_by(["subset", "resp_idx", "sent_idx"])
        .agg(pl.col("score").max().alias("s"))
    )


def consensus_top2(df):
    """within-chunk mean of the top-2 windows, then max over chunks."""
    per_chunk = (
        df.sort("score", descending=True)
        .group_by(["subset", "resp_idx", "sent_idx", "doc_idx"])
        .agg(pl.col("score").head(2).mean().alias("c"))
    )
    return (
        per_chunk.group_by(["subset", "resp_idx", "sent_idx"])
        .agg(pl.col("c").max().alias("s"))
    )


def read_checkpoint(tag):
    path = HERE / f"R13_dump_{tag}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pl.read_parquet(path).select(
        "subset", "resp_idx", "sent_idx", "doc_idx", "win_in_doc",
        "n_win_in_doc", "resp_label", "score"
    )
    banked = json.loads((HERE / BANKED[tag]).read_text())["per_subset"]

    std = response_auc(standard(df), df)
    repro = {
        s: {"rebuilt": round(std[s], 4), "banked": banked[s]["auc"],
            "ok": abs(round(std[s], 4) - banked[s]["auc"]) < 5e-5}
        for s in std
    }
    repro_ok = all(v["ok"] for v in repro.values())

    con = response_auc(consensus_top2(df), df)
    deltas = {s: round(con[s] - std[s], 4) for s in std}
    mean_std = float(np.mean(list(std.values())))
    mean_con = float(np.mean(list(con.values())))

    # single-window chunks fall back to max by construction (mean of top-1)
    nchunks = df.select("subset", "resp_idx", "sent_idx", "doc_idx", "n_win_in_doc").unique()
    return {
        "tag": tag,
        "model": json.loads((HERE / BANKED[tag]).read_text())["model"],
        "reproduction_guard": {"per_subset": repro, "ok": bool(repro_ok)},
        "standard_per_subset": {s: round(v, 4) for s, v in std.items()},
        "consensus_per_subset": {s: round(v, 4) for s, v in con.items()},
        "delta_per_subset": deltas,
        "standard_mean": round(mean_std, 5),
        "consensus_mean": round(mean_con, 5),
        "mean_delta": round(mean_con - mean_std, 5),
        "n_chunk_cells": len(nchunks),
        "single_window_chunk_frac": round(
            float((nchunks["n_win_in_doc"] == 1).mean()), 4
        ),
        "void": not repro_ok,
    }


def main():
    res = {}
    for tag in PRIMARY + REPLICATION:
        res[tag] = read_checkpoint(tag)
        r = res[tag]
        print(f"\n=== {tag}  (repro ok={r['reproduction_guard']['ok']})")
        for s in sorted(r["delta_per_subset"]):
            print(f"    {s:12s} std {r['standard_per_subset'][s]:.4f} -> "
                  f"con {r['consensus_per_subset'][s]:.4f}  "
                  f"delta {r['delta_per_subset'][s]:+.4f}")
        print(f"    MEAN         {r['standard_mean']:.5f} -> {r['consensus_mean']:.5f}  "
              f"delta {r['mean_delta']:+.5f}")

    void = any(res[t]["void"] for t in res)
    hagrid = {t: res[t]["delta_per_subset"]["hagrid"] for t in PRIMARY}
    mean_d = {t: res[t]["mean_delta"] for t in PRIMARY}
    worst = {t: min(res[t]["delta_per_subset"].values()) for t in PRIMARY}
    worst_sub = {
        t: min(res[t]["delta_per_subset"], key=res[t]["delta_per_subset"].get)
        for t in PRIMARY
    }

    clauses = {
        "hagrid_ge_0.010_both_h108_draws": all(v >= 0.010 for v in hagrid.values()),
        "mean_HOLD_ge_-0.002_both": all(v >= -0.002 for v in mean_d.values()),
        "no_subset_below_-0.02_both": all(v >= -0.02 for v in worst.values()),
    }
    verdict = "ADMIT" if (not void and all(clauses.values())) else (
        "VOID (reproduction guard failed)" if void else "REFUTE"
    )

    out = {
        "hypothesis": "R13-H124 WINDOW-CONSENSUS-EVIDENCE-READ",
        "aggregator": "within-chunk mean of top-2 windows (single-window chunk = max); "
                      "max over chunks, min over sentences unchanged",
        "bar": "hagrid >= +0.010 on BOTH H108 draws AND mean delta >= -0.002 AND "
               "no subset < -0.02 (deterministic read, ruling 7 subset-primary)",
        "primary_checkpoints": list(PRIMARY),
        "replication_checkpoints": list(REPLICATION),
        "hagrid_delta": hagrid,
        "mean_delta": mean_d,
        "worst_subset_delta": {t: {"subset": worst_sub[t], "delta": worst[t]} for t in PRIMARY},
        "clauses": clauses,
        "verdict": verdict,
        "per_checkpoint": res,
    }
    RESULT.write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 78)
    print(f"  hagrid delta (H108): {hagrid}   bar >= +0.010 both")
    print(f"  mean delta   (H108): {mean_d}   bar >= -0.002 both")
    print(f"  worst subset (H108): {out['worst_subset_delta']}   bar >= -0.02")
    print(f"  clauses: {clauses}")
    print(f"  VERDICT: {verdict}")
    print(f"  -> {RESULT}")


if __name__ == "__main__":
    main()

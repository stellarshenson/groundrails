"""R19-H161 lane L4 - H4 long-window prose register interference, ANALYSIS ONLY.

Reads the three aligned per-pair dumps written by A0 (flagship R18-H150 draw 1,
flagship draw 2, enriched R19-H159 draw 1) and answers five questions on CPU:

  1  positive control - per-subset AUROC rebuilt from `item_score` must match the
     banked windowed reads to 1e-3, else the analysis aborts
  2  HEADLINE - argmax-window drift: for every sentence with >= 2 windows, does the
     enriched checkpoint select a DIFFERENT window than the flagship? The noise floor
     is flagship draw 1 vs draw 2 (same recipe, different seed)
  3  drift consequence - among sentences whose selector moved, did the sentence score
     move toward or away from its item's gold label
  4  window-count stratification - AUROC per pre-stated stratum 1 / 2-3 / 4-7 / 8+,
     items binned by their sinking sentence's window count and, separately, by their
     total window count
  5  positional free rider - does the enriched argmax sit earlier or later in the
     window list

Nothing trains, no threshold is fitted, no GPU is touched. The strata boundaries are
the pre-stated ones and are never re-cut.

Run:  uv run python experiments/grounding-semantic/R19-H161_L4.py
"""

import json
import pathlib
import sys
import time

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R19-H161_L4_result.json"

DUMPS = {
    "h150d1": HERE / "R19-H161_pairs_h150d1.parquet",
    "h150d2": HERE / "R19-H161_pairs_h150d2.parquet",
    "h159d1": HERE / "R19-H161_pairs_h159d1.parquet",
}
BANKED = {
    "h150d1": (HERE / "R18-H150_arm_draw1_windowed_result.json", 0.71436),
    "h150d2": (HERE / "R18-H150_arm_draw2_windowed_result.json", 0.71661),
    "h159d1": (HERE / "R19-H159_arm_draw1_windowed_result.json", 0.68941),
}
CONTROL_TOL = 1e-3

# The three subsets that collapsed under the enriched mix (finqa -0.1429,
# tatqa -0.1328, delucionqa -0.1025).
COLLAPSED = ("finqa", "tatqa", "delucionqa")

# Pre-stated window-count strata. Never re-cut.
STRATA = (("1", 1, 1), ("2-3", 2, 3), ("4-7", 4, 7), ("8+", 8, 10**9))

# Reporting floor for a stratum AUROC - stated up front, not tuned. A cell below
# this is reported as null and named in the caveats.
MIN_ITEMS = 20
MIN_PER_CLASS = 5


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def banked_subsets(path):
    """Per-subset banked AUROC. Some result files nest under `per_subset`."""
    blob = json.loads(path.read_text())
    rows = blob.get("per_subset", blob)
    return {k: v["auc"] for k, v in rows.items() if isinstance(v, dict) and "auc" in v}


def auc_or_none(y, s):
    """AUROC with the pre-stated support floor; returns (auc, n, n_pos, n_neg)."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if len(y) < MIN_ITEMS or n_pos < MIN_PER_CLASS or n_neg < MIN_PER_CLASS:
        return None, len(y), n_pos, n_neg
    return float(roc_auc_score(y, s)), len(y), n_pos, n_neg


def items_of(df):
    """One row per (subset, item_id): label, item_score, sinking-sentence window count,
    the item's per-sentence window count and its total window count."""
    sink = (
        df.filter(pl.col("is_sinking"))
        .group_by(["subset", "item_id"])
        .agg(pl.col("n_win_sent").min().alias("sink_n_win"))
    )
    per_sent = df.unique(subset=["subset", "item_id", "sent_idx"]).select(
        ["subset", "item_id", "sent_idx", "n_win_sent"]
    )
    tot = per_sent.group_by(["subset", "item_id"]).agg(
        pl.col("n_win_sent").sum().alias("item_total_win"),
        pl.col("n_win_sent").n_unique().alias("n_distinct_win_counts"),
        pl.col("n_win_sent").max().alias("item_max_win"),
    )
    base = df.group_by(["subset", "item_id"]).agg(
        pl.col("label").first(), pl.col("item_score").first()
    )
    return base.join(sink, on=["subset", "item_id"]).join(tot, on=["subset", "item_id"])


def argmax_table(df, tag):
    """One row per (subset, item_id, sent_idx): the selected window and its score.
    Ties on `is_argmax` are broken by the lowest win_idx and counted."""
    a = df.filter(pl.col("is_argmax"))
    ties = int(
        a.group_by(["subset", "item_id", "sent_idx"]).len().filter(pl.col("len") > 1).height
    )
    a = (
        a.sort(["subset", "item_id", "sent_idx", "win_idx"])
        .group_by(["subset", "item_id", "sent_idx"], maintain_order=True)
        .first()
    )
    return (
        a.select(
            "subset",
            "item_id",
            "sent_idx",
            "label",
            "n_win_sent",
            pl.col("win_idx").alias(f"arg_{tag}"),
            pl.col("sent_score").alias(f"score_{tag}"),
            pl.col("is_sinking").alias(f"sink_{tag}"),
        ),
        ties,
    )


def stratum_of(col):
    e = pl.when(pl.col(col) <= 1).then(pl.lit("1"))
    for name, lo, hi in STRATA[1:]:
        e = e.when((pl.col(col) >= lo) & (pl.col(col) <= hi)).then(pl.lit(name))
    return e.otherwise(pl.lit("1")).alias("stratum")


def main():
    missing = [str(p) for p in DUMPS.values() if not p.exists()]
    if missing:
        raise SystemExit(f"L4 ABORT: dump not on disk - {missing}")

    log("=== R19-H161 L4 analysis start ===")
    dfs = {}
    for tag, path in DUMPS.items():
        dfs[tag] = pl.read_parquet(path)
        log(f"  loaded {tag}: {dfs[tag].height:,} pair rows from {path.name}")

    res = {
        "lane": "L4",
        "hypothesis": "H4 long-window prose register interference",
        "dumps": {k: str(v) for k, v in DUMPS.items()},
        "strata": [s[0] for s in STRATA],
        "support_floor": {"min_items": MIN_ITEMS, "min_per_class": MIN_PER_CLASS},
    }

    # ---------------- 1  positive control -------------------------------------
    log("-- 1  positive control: rebuilding per-subset AUROC from item_score")
    control, worst = {}, 0.0
    items = {tag: items_of(df) for tag, df in dfs.items()}
    for tag, df in dfs.items():
        bank = banked_subsets(BANKED[tag][0])
        rows, recon = {}, []
        it = items[tag]
        for sub in sorted(bank):
            s = it.filter(pl.col("subset") == sub)
            if s.height == 0:
                rows[sub] = {"banked": bank[sub], "rebuilt": None, "delta": None, "n": 0}
                worst = float("inf")
                continue
            a = float(roc_auc_score(s["label"].to_numpy(), s["item_score"].to_numpy()))
            d = a - bank[sub]
            worst = max(worst, abs(d))
            recon.append(a)
            rows[sub] = {
                "banked": bank[sub],
                "rebuilt": round(a, 5),
                "delta": round(d, 6),
                "n": s.height,
            }
        mean = round(float(np.mean(recon)), 5) if recon else None
        control[tag] = {
            "per_subset": rows,
            "rebuilt_mean": mean,
            "banked_mean": BANKED[tag][1],
            "mean_delta": None if mean is None else round(mean - BANKED[tag][1], 6),
        }
        log(f"  {tag}: rebuilt mean {mean} vs banked {BANKED[tag][1]}")
    res["positive_control"] = {"max_abs_delta": worst, "tol": CONTROL_TOL, "by_checkpoint": control}
    if worst > CONTROL_TOL:
        res["verdict"] = "INDETERMINATE"
        res["abort"] = f"positive control miss {worst:.6f} > {CONTROL_TOL}"
        OUT.write_text(json.dumps(res, indent=2))
        log(f"CONTROL ABORT: max |delta| {worst:.6f} > {CONTROL_TOL}; wrote {OUT}")
        sys.exit(2)
    res["positive_control"]["verdict"] = "PASS"
    log(f"  control PASS - max |delta| {worst:.2e}")

    # ---------------- 2  HEADLINE argmax drift --------------------------------
    log("-- 2  argmax-window drift (sentences with >= 2 windows)")
    tabs, ties = {}, {}
    for tag, df in dfs.items():
        tabs[tag], ties[tag] = argmax_table(df, tag)
    j = tabs["h150d1"].join(
        tabs["h150d2"].drop("label", "n_win_sent"), on=["subset", "item_id", "sent_idx"]
    )
    j = j.join(tabs["h159d1"].drop("label", "n_win_sent"), on=["subset", "item_id", "sent_idx"])
    res["argmax_ties"] = ties
    res["joined_sentences"] = j.height

    multi = j.filter(pl.col("n_win_sent") >= 2).with_columns(
        (pl.col("arg_h150d1") != pl.col("arg_h150d2")).alias("ff"),
        (pl.col("arg_h159d1") != pl.col("arg_h150d1")).alias("ed1"),
        (pl.col("arg_h159d1") != pl.col("arg_h150d2")).alias("ed2"),
    )
    drift = (
        multi.group_by("subset")
        .agg(
            pl.len().alias("n_sent_multi"),
            pl.col("ff").mean().alias("flag_vs_flag"),
            pl.col("ed1").mean().alias("enriched_vs_flag_d1"),
            pl.col("ed2").mean().alias("enriched_vs_flag_d2"),
            pl.col("n_win_sent").mean().alias("mean_n_win"),
        )
        .with_columns(
            (pl.col("enriched_vs_flag_d1") - pl.col("flag_vs_flag")).alias("excess_vs_d1"),
            (pl.col("enriched_vs_flag_d2") - pl.col("flag_vs_flag")).alias("excess_vs_d2"),
        )
        .sort("subset")
    )
    res["argmax_drift"] = {
        r["subset"]: {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}
        for r in drift.to_dicts()
    }
    for r in drift.to_dicts():
        log(
            f"  {r['subset']:12s} n={r['n_sent_multi']:>5} ff={r['flag_vs_flag']:.4f} "
            f"e/d1={r['enriched_vs_flag_d1']:.4f} excess={r['excess_vs_d1']:+.4f}"
        )
    col = drift.filter(pl.col("subset").is_in(COLLAPSED))
    oth = drift.filter(~pl.col("subset").is_in(COLLAPSED))
    res["argmax_drift_summary"] = {
        "collapsed_subsets": list(COLLAPSED),
        "mean_excess_collapsed": round(float(col["excess_vs_d1"].mean()), 5),
        "mean_excess_other": round(float(oth["excess_vs_d1"].mean()), 5),
        "n_subsets_excess_positive": int((drift["excess_vs_d1"] > 0).sum()),
        "collapsed_all_positive": bool((col["excess_vs_d1"] > 0).all()),
        "collapsed_rank_of_excess": [
            {"subset": r["subset"], "rank": i + 1}
            for i, r in enumerate(drift.sort("excess_vs_d1", descending=True).to_dicts())
            if r["subset"] in COLLAPSED
        ],
    }

    # ---------------- 3  drift consequence ------------------------------------
    log("-- 3  drift consequence: score movement toward or away from gold")
    cons = multi.with_columns(
        (pl.col("score_h159d1") - pl.col("score_h150d1")).alias("d_score")
    ).with_columns(
        pl.when(pl.col("label") == 1)
        .then(pl.col("d_score"))
        .otherwise(-pl.col("d_score"))
        .alias("toward_gold")
    )
    cons_rows = {}
    for sub in sorted(cons["subset"].unique().to_list()):
        s = cons.filter(pl.col("subset") == sub)
        blocks = {}
        for moved in (True, False):
            for lab in (1, 0):
                g = s.filter((pl.col("ed1") == moved) & (pl.col("label") == lab))
                key = f"{'moved' if moved else 'stable'}_label{lab}"
                blocks[key] = {
                    "n": g.height,
                    "mean_toward_gold": (
                        None if g.height == 0 else round(float(g["toward_gold"].mean()), 5)
                    ),
                    "median_toward_gold": (
                        None if g.height == 0 else round(float(g["toward_gold"].median()), 5)
                    ),
                    "frac_toward_gold": (
                        None if g.height == 0 else round(float((g["toward_gold"] > 0).mean()), 5)
                    ),
                }
        cons_rows[sub] = blocks
    res["drift_consequence"] = cons_rows
    tot = {}
    for moved in (True, False):
        for lab in (1, 0):
            g = cons.filter((pl.col("ed1") == moved) & (pl.col("label") == lab))
            tot[f"{'moved' if moved else 'stable'}_label{lab}"] = {
                "n": g.height,
                "mean_toward_gold": (
                    None if g.height == 0 else round(float(g["toward_gold"].mean()), 5)
                ),
                "frac_toward_gold": (
                    None if g.height == 0 else round(float((g["toward_gold"] > 0).mean()), 5)
                ),
            }
    res["drift_consequence_pooled"] = tot
    log(f"  pooled: {json.dumps(tot)}")

    # ---------------- 4  window-count stratification --------------------------
    log("-- 4  AUROC by window-count stratum")
    # Is n_win_sent an item property (constant across the item's sentences)?
    const = {
        tag: int(it.filter(pl.col("n_distinct_win_counts") > 1).height) for tag, it in items.items()
    }
    res["items_with_varying_sentence_window_counts"] = const

    strat = {}
    for basis, col_name in (("sinking_sentence", "sink_n_win"), ("item_total", "item_total_win")):
        by_ck = {}
        for tag, it in items.items():
            t = it.with_columns(stratum_of(col_name))
            per_sub = {}
            for sub in sorted(t["subset"].unique().to_list()):
                cells = {}
                for name, _lo, _hi in STRATA:
                    g = t.filter((pl.col("subset") == sub) & (pl.col("stratum") == name))
                    a, n, npos, nneg = auc_or_none(
                        g["label"].to_numpy(), g["item_score"].to_numpy()
                    )
                    cells[name] = {
                        "auc": None if a is None else round(a, 5),
                        "n": n,
                        "n_pos": npos,
                        "n_neg": nneg,
                    }
                per_sub[sub] = cells
            # pooled over all subsets, per stratum
            pooled = {}
            for name, _lo, _hi in STRATA:
                g = t.filter(pl.col("stratum") == name)
                a, n, npos, nneg = auc_or_none(g["label"].to_numpy(), g["item_score"].to_numpy())
                pooled[name] = {
                    "auc": None if a is None else round(a, 5),
                    "n": n,
                    "n_pos": npos,
                    "n_neg": nneg,
                }
            by_ck[tag] = {"per_subset": per_sub, "pooled_all_subsets": pooled}
        # flagship -> enriched delta, per subset per stratum, and its mean over subsets
        delta = {}
        for sub in by_ck["h150d1"]["per_subset"]:
            delta[sub] = {}
            for name, _lo, _hi in STRATA:
                f1 = by_ck["h150d1"]["per_subset"][sub][name]["auc"]
                f2 = by_ck["h150d2"]["per_subset"][sub][name]["auc"]
                e = by_ck["h159d1"]["per_subset"][sub][name]["auc"]
                delta[sub][name] = {
                    "enriched_minus_d1": None if (f1 is None or e is None) else round(e - f1, 5),
                    "d2_minus_d1": None if (f1 is None or f2 is None) else round(f2 - f1, 5),
                    "n": by_ck["h150d1"]["per_subset"][sub][name]["n"],
                }
        mean_delta = {}
        for name, _lo, _hi in STRATA:
            vals = [
                delta[s][name]["enriched_minus_d1"]
                for s in delta
                if delta[s][name]["enriched_minus_d1"] is not None
            ]
            noise = [
                delta[s][name]["d2_minus_d1"]
                for s in delta
                if delta[s][name]["d2_minus_d1"] is not None
            ]
            mean_delta[name] = {
                "mean_enriched_minus_d1": None if not vals else round(float(np.mean(vals)), 5),
                "n_subsets": len(vals),
                "mean_abs_d2_minus_d1_noise": None if not noise else
                round(float(np.mean(np.abs(noise))), 5),
            }
        strat[basis] = {
            "by_checkpoint": by_ck,
            "delta_per_subset": delta,
            "mean_delta_over_subsets": mean_delta,
        }
        log(f"  {basis}: {json.dumps(mean_delta)}")
    res["auroc_by_window_stratum"] = strat

    # ---------------- 5  positional free rider --------------------------------
    log("-- 5  positional shift of the selected window")
    pos = multi.with_columns(
        (pl.col("arg_h150d1") / (pl.col("n_win_sent") - 1)).alias("rel_d1"),
        (pl.col("arg_h150d2") / (pl.col("n_win_sent") - 1)).alias("rel_d2"),
        (pl.col("arg_h159d1") / (pl.col("n_win_sent") - 1)).alias("rel_e"),
    )
    prow = (
        pos.group_by("subset")
        .agg(
            pl.len().alias("n"),
            pl.col("rel_d1").mean().alias("mean_rel_h150d1"),
            pl.col("rel_d2").mean().alias("mean_rel_h150d2"),
            pl.col("rel_e").mean().alias("mean_rel_h159d1"),
        )
        .with_columns(
            (pl.col("mean_rel_h159d1") - pl.col("mean_rel_h150d1")).alias("shift_enriched_vs_d1"),
            (pl.col("mean_rel_h150d2") - pl.col("mean_rel_h150d1")).alias("shift_noise_d2_vs_d1"),
        )
        .sort("subset")
    )
    res["positional_shift"] = {
        r["subset"]: {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}
        for r in prow.to_dicts()
    }
    res["positional_shift_pooled"] = {
        "mean_rel_h150d1": round(float(pos["rel_d1"].mean()), 5),
        "mean_rel_h150d2": round(float(pos["rel_d2"].mean()), 5),
        "mean_rel_h159d1": round(float(pos["rel_e"].mean()), 5),
        "shift_enriched_vs_d1": round(float(pos["rel_e"].mean() - pos["rel_d1"].mean()), 5),
        "shift_noise_d2_vs_d1": round(float(pos["rel_d2"].mean() - pos["rel_d1"].mean()), 5),
    }
    log(f"  pooled: {json.dumps(res['positional_shift_pooled'])}")

    # ---------------- caveats -------------------------------------------------
    thin = []
    for basis, blob in strat.items():
        for tag, ck in blob["by_checkpoint"].items():
            for sub, cells in ck["per_subset"].items():
                for name, c in cells.items():
                    if c["auc"] is None:
                        thin.append(f"{basis}/{tag}/{sub}/stratum {name} (n={c['n']}, "
                                    f"pos={c['n_pos']}, neg={c['n_neg']})")
    res["thin_cells"] = sorted(set(thin))
    res["thin_cell_count"] = len(res["thin_cells"])

    OUT.write_text(json.dumps(res, indent=2))
    log(f"=== R19-H161 L4 analysis done -> {OUT} ===")


if __name__ == "__main__":
    main()

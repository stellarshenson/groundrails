"""ANCHOR-TEACHER ceiling - salvage diagnostic (ANALYSIS ONLY).

Recorded in R13_synthesis.md section 5. The object every consistency /
distillation lane would distil is the OUTPUT-MEAN of the two frozen R9-H105
draws. It has never been measured through the arena read. Average the sigmoid
PROBABILITIES of the two draws per (sentence, window), aggregate through the
standard windowed read (max over windows, min over sentences) and take the blind
windowed mean.

Bar: below the H105 pair mean + 0.005 = 0.70811 closes the whole
consistency / distillation class for 0.5 GPU-h.

Zero extra GPU - both dumps already exist (Task A). Sanity guard: each draw's own
matrix must first reproduce its banked windowed result to 4 dp.

Run:  uv run python experiments/grounding-semantic/R13_anchor_teacher.py
"""

import importlib.util
import json
import pathlib

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
RESULT = HERE / "R13_anchor_teacher_result.json"

BANKED = {
    "h105d1": "R9-H105_windowed_result.json",
    "h105d2": "R9-H105_draw2_windowed_result.json",
}
KEY = ["subset", "resp_idx", "sent_idx", "win_idx"]


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M59 = _mod("m59", "R7-H59_cross_domain_matrix.py")


def response_auc(df, col):
    """max over windows per sentence, min over sentences, AUC vs adherence."""
    out = {}
    sent = df.group_by(["subset", "resp_idx", "sent_idx"]).agg(pl.col(col).max().alias("s"))
    for sub in sorted(df["subset"].unique().to_list()):
        r = (
            sent.filter(pl.col("subset") == sub)
            .group_by("resp_idx").agg(pl.col("s").min()).sort("resp_idx")
        )
        y = (
            df.filter(pl.col("subset") == sub)
            .group_by("resp_idx").agg(pl.col("resp_label").first())
            .sort("resp_idx")["resp_label"].to_numpy()
        )
        auc, _, _ = M59.auc_and_f1(y, r["s"].to_numpy())
        out[sub] = float(auc)
    return out


def main():
    d1 = pl.read_parquet(HERE / "R13_dump_h105d1.parquet").select(
        *KEY, "resp_label", "score"
    )
    d2 = pl.read_parquet(HERE / "R13_dump_h105d2.parquet").select(*KEY, "score")
    assert len(d1) == len(d2), f"matrix shape mismatch {len(d1)} vs {len(d2)}"

    guard = {}
    for tag, df in (("h105d1", d1), ("h105d2", d2.join(d1.select(*KEY, "resp_label"), on=KEY))):
        banked = json.loads((HERE / BANKED[tag]).read_text())["per_subset"]
        au = response_auc(df, "score")
        guard[tag] = {
            "per_subset": {
                s: {"rebuilt": round(au[s], 4), "banked": banked[s]["auc"],
                    "ok": abs(round(au[s], 4) - banked[s]["auc"]) < 5e-5}
                for s in au
            },
            "mean": round(float(np.mean(list(au.values()))), 5),
        }
        guard[tag]["ok"] = all(v["ok"] for v in guard[tag]["per_subset"].values())
    guard_ok = all(guard[t]["ok"] for t in guard)

    j = d1.join(d2.rename({"score": "score2"}), on=KEY, how="inner")
    assert len(j) == len(d1), f"key join lost rows: {len(j)} vs {len(d1)}"
    j = j.with_columns(((pl.col("score") + pl.col("score2")) / 2.0).alias("anchor"))

    au = response_auc(j, "anchor")
    mean = float(np.mean(list(au.values())))
    pair_mean = (guard["h105d1"]["mean"] + guard["h105d2"]["mean"]) / 2.0
    bar = round(pair_mean + 0.005, 5)

    res = {
        "diagnostic": "ANCHOR-TEACHER ceiling (ANALYSIS ONLY)",
        "object": "per-(sentence, window) mean of the two frozen H105 draws' sigmoid "
                  "probabilities, read through the standard windowed arena read",
        "reproduction_guard": {"per_draw": guard, "ok": bool(guard_ok)},
        "h105_pair_mean": round(pair_mean, 5),
        "bar": bar,
        "anchor_per_subset": {s: round(v, 4) for s, v in au.items()},
        "anchor_mean": round(mean, 5),
        "delta_vs_pair_mean": round(mean - pair_mean, 5),
        "verdict": (
            "VOID (reproduction guard failed)" if not guard_ok
            else ("OPEN - anchor teacher clears the bar" if mean >= bar
                  else "CLOSED - consistency/distillation class closed on measurement")
        ),
    }
    RESULT.write_text(json.dumps(res, indent=2))

    print("=" * 78)
    print(f"  reproduction guard ok = {guard_ok}")
    for s in sorted(au):
        print(f"    {s:12s} d1 {guard['h105d1']['per_subset'][s]['rebuilt']:.4f}  "
              f"d2 {guard['h105d2']['per_subset'][s]['rebuilt']:.4f}  "
              f"anchor {au[s]:.4f}")
    print(f"  pair mean {pair_mean:.5f}   bar {bar:.5f}   anchor mean {mean:.5f} "
          f"({mean - pair_mean:+.5f})")
    print(f"  VERDICT: {res['verdict']}")
    print(f"  -> {RESULT}")


if __name__ == "__main__":
    main()

"""DR lane assembly - judged pilot -> the training lane parquet (H117-ready).

Adjudicated assembly rules (recorded in semantic-dataset-enhancements.md):
  - H113 dropped; its 20% share redistributes proportionally over survivors:
    H112 68.75% / H114 31.25% of the 22k negative target
  - H114 supply (4,343) cannot fill 31.25% -> ratio holds, lane SHIPS SMALLER
    (the registration's no-forced-backfill principle): H112 9,555 + H114 4,343
  - long-form <= 20% of the H112 share (A11): 1,911 long-form + 7,644 sentence
  - label-1 reclaim cap 4k: all 2,573 taken
  - H117 amendments A1-A3: every negative ships with its seed as a materialized
    margin-only partner row - bce_mask=True, label=-1 (never in BCE), pair_id
    shared, DANN tag = the corrupt partner's tag. Control and margin arms train
    on the IDENTICAL row set; the control sets lambda_margin=0.

Output: DR_lane.parquet
  role: corrupt (label 0, BCE) | clean_partner (label -1, margin-only)
        | reclaim (label 1, BCE, no partner)

Run: uv run python experiments/grounding-semantic/DR_lane_assemble.py
"""

import json
import pathlib

import polars as pl

HERE = pathlib.Path(__file__).parent
JUDGED = HERE / "DR_judged.parquet"
OUT = HERE / "DR_lane.parquet"
SUMMARY = HERE / "DR_lane_summary.json"

N_H112_SENT = 7644
N_H112_LONG = 1911
SEED = 0


def main():
    df = pl.read_parquet(JUDGED)

    neg = df.filter(pl.col("label") == 0)
    h112_sent = neg.filter((pl.col("engine") == "H112") & ~pl.col("long_form")).sample(
        n=N_H112_SENT, seed=SEED)
    h112_long = neg.filter(pl.col("long_form")).sample(n=N_H112_LONG, seed=SEED)
    h114 = neg.filter(pl.col("engine") == "H114")
    negatives = pl.concat([h112_sent, h112_long, h114])

    reclaim = df.filter(pl.col("label") == 1)

    keep = ["seed_id", "engine", "long_form", "seed", "chunk", "claim",
            "delta", "severity", "nli_fwd", "nli_bwd"]

    def dann_tag(row):
        if row["label"] == 1:
            return "dr_reclaim"
        return "dr_h112_long" if row["long_form"] else f"dr_{row['engine'].lower()}"

    rows = []
    pair_id = 0
    for r in negatives.select(keep + ["label"]).iter_rows(named=True):
        assert r["label"] == 0, "corrupt member of a margin pair must be label 0"
        tag = dann_tag(r)
        base = {k: r[k] for k in keep}
        rows.append({**base, "pair_id": pair_id, "role": "corrupt",
                     "text": r["claim"], "label": 0, "bce_mask": False,
                     "dann_tag": tag})
        rows.append({**base, "pair_id": pair_id, "role": "clean_partner",
                     "text": r["seed"], "label": -1, "bce_mask": True,
                     "dann_tag": tag})
        pair_id += 1
    for r in reclaim.select(keep + ["label"]).iter_rows(named=True):
        rows.append({**{k: r[k] for k in keep}, "pair_id": -1, "role": "reclaim",
                     "text": r["claim"], "label": 1, "bce_mask": False,
                     "dann_tag": "dr_reclaim"})

    lane = pl.DataFrame(rows)
    lane.write_parquet(OUT)

    stats = {
        "pairs": pair_id,
        "rows_total": len(lane),
        "negatives": {"h112_sent": len(h112_sent), "h112_long": len(h112_long),
                      "h114": len(h114)},
        "reclaimed_positives": len(reclaim),
        "bce_rows": len(lane.filter(~pl.col("bce_mask"))),
        "by_dann_tag": {r["dann_tag"]: r["len"] for r in
                        lane.group_by("dann_tag").len().iter_rows(named=True)},
        "h117_pair_floor_8k": pair_id >= 8000,
    }
    SUMMARY.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print("=== DR LANE ASSEMBLED ===")


if __name__ == "__main__":
    main()

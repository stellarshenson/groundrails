"""R11-H117 probe - held-out certified pair set (2,000), excluded from every arm.

Certified negatives from DR_judged.parquet whose (seed, chunk) locus does NOT
appear anywhere in DR_lane.parquet, so no probe arm trained on either member.
A pair is (clean seed, corrupt claim) over the same chunk; pair-accuracy is
p(seed, chunk) > p(claim, chunk).

A7 split recorded here: `verbatim` = the seed is a whitespace-normalized
substring of its chunk (the trivially-easy lexical-copy half).

Run: uv run python experiments/grounding-semantic/R11-H117_heldout_pairs.py
"""

import json
import pathlib
import re

import polars as pl

HERE = pathlib.Path(__file__).parent
JUDGED = HERE / "DR_judged.parquet"
LANE = HERE / "DR_lane.parquet"
OUT = HERE / "R11-H117_heldout_pairs.parquet"
N_HELDOUT = 2000
SEED = 1117


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def main():
    lane = pl.read_parquet(LANE)
    used = set(zip(lane["seed"].to_list(), lane["chunk"].to_list(), strict=True))

    df = pl.read_parquet(JUDGED).filter(pl.col("label") == 0)
    spare_mask = [
        (s, c) not in used
        for s, c in zip(df["seed"].to_list(), df["chunk"].to_list(), strict=True)
    ]
    spare = df.filter(pl.Series(spare_mask))
    # one pair per locus - a locus may carry 2 corruptions
    spare = spare.unique(subset=["seed", "chunk"], keep="first", maintain_order=True)
    pairs = spare.sample(n=N_HELDOUT, seed=SEED)

    verbatim = [
        norm(s) in norm(c)
        for s, c in zip(pairs["seed"].to_list(), pairs["chunk"].to_list(), strict=True)
    ]
    out = pairs.select(["seed_id", "engine", "long_form", "delta", "severity",
                        "seed", "chunk", "claim"]).with_columns(
        pl.Series("verbatim", verbatim))
    out.write_parquet(OUT)

    stats = {
        "certified_negatives_total": len(df),
        "spare_rows_off_lane": len(spare_mask) - sum(1 for m in spare_mask if not m),
        "spare_distinct_loci": len(spare),
        "held_out_pairs": len(out),
        "verbatim_share": round(sum(verbatim) / len(out), 4),
        "by_engine": {r["engine"]: r["len"] for r in
                      out.group_by("engine").len().iter_rows(named=True)},
        "long_form": int(out["long_form"].sum()),
    }
    print(json.dumps(stats, indent=2))
    print(f"=== HELD-OUT PAIRS -> {OUT} ===")


if __name__ == "__main__":
    main()

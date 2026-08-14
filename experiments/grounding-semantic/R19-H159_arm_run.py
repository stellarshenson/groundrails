"""R19-H159 ENRICHED-MIX ARM - the flagship recipe carrying the R19 supply lanes.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R19-H159 ENRICHED-MIX ARM", amendment A1. The arm isolates a DATA change: the
recipe is the flagship (R18-H150 pair) VERBATIM and the only difference is five
new corpora, each admitted as its own DANN group.

The G1 twin trainer and reader (`R16-H142_G1_arm.py`, `R16-H142_G1_reads.py`) are
BANKED. This wrapper rebinds their run-scoped constants and dispatches into their
own `main()`, exactly as the H150 / H152 wrappers do, so the arm trains and reads
through byte-identical code. The ONLY new code is the mix assembly.

    protocol    the flagship recipe VERBATIM - evidence UNTRUNCATED, presented as
                1,500/750 windows, MIL max-over-windows BCE, full trunk at
                lr 1e-5 OneCycleLR 1 epoch 10% warmup, clip 1.0, DANN lambda 0.02
                Ganin ramp, 48 sets / 96 pairs per batch, adapter FROZEN at its
                zero init (TWIN INTEGRITY ABORT guard). NO EMA, NO window
                dropout - those are H152's and composing them is a different arm.
    mix         clean public 685,670 (R10-H108.public_train, untruncated)
                + H146 misbind lane      30,000 -> group `quant_misbind`
                + H150 unit_swap lane     5,540 -> group `quant_scale_unit`
                + R19 FAVA lane          30,073 -> group `fava`
                + R19 AttributionBench   16,426 -> group `attributionbench`
                + R19 MiniCheck          14,356 -> group `minicheck`
                + R19 PubHealth          12,251 -> group `pubhealth`
                + R19 FinDVer             2,400 -> group `findver`
                = 796,716 rows, 19 DANN groups
    seeds       draw 1 -> 1159, draw 2 -> 2159
    checkpoint  models/R19-H159-arm-draw{N}
    results     R19-H159_arm_draw{N}_result.json          (train + in-domain suite)
                R19-H159_arm_draw{N}_windowed_result.json (PRIMARY blind arena read)

AMENDMENT A1 (coordinator ruling, 2026-08-14, after this wrapper's first census
fired the banked BATCH-CAP ABORT):

    1  FActScore is WITHDRAWN from the arm. Its evidence unit is a whole
       Wikipedia biography (mean 30,426 chars over 181 distinct documents,
       40.10 mean windows/row, 1,341 rows over the 96-pair cap). Making it fit
       needs per-fact evidence spans re-derived from the source - a lane BUILD
       with its own registration, not a presentation tweak. It stays banked
       supply.
    2  The other five lanes admit UNTRUNCATED, per the twin protocol. The
       1,500-char lane cap was priced and REJECTED: it reproduces the
       registration's step estimate but breaks the presentation the twin
       protocol exists to keep identical between training and serving. The
       registered mix moves, not the presentation.
    3  The 18 AttributionBench rows over the 96-pair batch cap are DROPPED at
       the trainer's own guard threshold and the count is RECORDED in the
       census JSON and the result JSON - a documented drop, never a silent
       filter. Every other lane is inside the cap already.

Census-before-spend: the `census` stage assembles the mix on CPU, derives the
window census and WRITES `R19-H159_window_census.json`. The `train` stage
re-derives the census and aborts unless it reproduces that banked file, so a
silent change to any lane parquet cannot reach a card.

Stages:
    train      train + the in-domain suite (gold, gold_full, RAGTruth EN + 7)
    windowed   the PRIMARY blind windowed decomposed-min arena read
    census     CPU-only dry run - mix census, window census, init and permutation
               fingerprints, then exit before any GPU work

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
      uv run python experiments/grounding-semantic/R19-H159_arm_run.py \
          --stage train --draw 1
"""

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent

DRAWS = {1: 1159, 2: 2159}
CKPT = "R19-H159-arm-draw{draw}"
TRAIN_OUT = "R19-H159_arm_draw{draw}_result.json"
READ_OUT = "R19-H159_arm_draw{draw}_{mode}_result.json"

EXPECTED_CLEAN_ROWS = 685_670
EXPECTED_MIX_ROWS = 796_716  # amendment A1: factscore withdrawn, 18 over-cap rows dropped
EXPECTED_GROUPS = (
    "attributionbench", "fava", "findver", "halueval", "minicheck",
    "psiloqa", "pubhealth", "quant_misbind", "quant_scale_unit", "ragtruth_cn",
    "ragtruth_de", "ragtruth_en", "ragtruth_es", "ragtruth_fr", "ragtruth_hu",
    "ragtruth_it", "ragtruth_pl", "tabfact", "vitaminc",
)

# The two banked quantitative lanes, at their flagship scale, checked to the
# negative-family level exactly as the H150 wrapper checks them.
QUANT_LANES = (
    ("R17-H146_lane.parquet", "quant_misbind", 30_000, 15_000,
     {"misbound_row": 21_000, "misbound_col": 9_000}),
    ("R18-H150_scaleunit_lane.parquet", "quant_scale_unit", 5_540, 2_770,
     {"unit_swap": 5_540}),
)

# The five admitted R19 supply lanes: (file, group, rows and positives AS BANKED
# IN THE PARQUET, rows KEPT after the over-cap drop). These lanes carry no
# pair/neg-family convention - one row per pair_id - so the file assertion is
# rows + positives + the parquet's own tag column, and a second assertion holds
# the kept count after the drop. FActScore is withdrawn under amendment A1.
R19_LANES = (
    ("R19_fava_lane.parquet", "fava", 30_073, 637, 30_073),
    ("R19_attributionbench_lane.parquet", "attributionbench", 16_444, 10_656, 16_426),
    ("R19_minicheck_lane.parquet", "minicheck", 14_356, 6_638, 14_356),
    ("R19_pubhealth_lane.parquet", "pubhealth", 12_251, 6_306, 12_251),
    ("R19_findver_lane.parquet", "findver", 2_400, 1_201, 2_400),
)

WINDOW_CENSUS = HERE / "R19-H159_window_census.json"
CENSUS_TOL = 0.001
_CENSUS_WRITE = False  # set by the `census` stage; every other stage cross-checks
# Amendment A1 clause 3: the documented over-cap drop, filled by `build_mix` and
# written into BOTH the census JSON and the result JSON. Never a silent filter.
OVER_CAP_DROP = {}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def census_record(sizes):
    """Census-before-spend on the PRESENTATION, not just the rows.

    The `census` stage (re)writes the file; every later stage must reproduce it.
    A missing file on a training run is itself an abort - the census is not
    optional, and an amended lane presentation must be re-censused."""
    share = float((sizes > 1).mean())
    mean_w = float(sizes.mean())
    block = {
        "rows": int(sizes.size),
        "multi_window_share": round(share, 4),
        "mean_windows": round(mean_w, 4),
        "max_windows": int(sizes.max()),
        "total_pairs": int(sizes.sum()),
    }
    if _CENSUS_WRITE:
        WINDOW_CENSUS.write_text(json.dumps(
            {"combined": block, "over_cap_drop": OVER_CAP_DROP}, indent=2))
        print(f"window census WRITTEN -> {WINDOW_CENSUS.name}: {block}", flush=True)
        return
    banked = json.loads(WINDOW_CENSUS.read_text())["combined"]
    print(f"window census cross-check vs {WINDOW_CENSUS.name}: "
          f"multi-window {share:.4f} (banked {banked['multi_window_share']}), "
          f"mean windows {mean_w:.4f} (banked {banked['mean_windows']}), "
          f"rows {sizes.size} (banked {banked['rows']})", flush=True)
    if (sizes.size != banked["rows"]
            or abs(share - banked["multi_window_share"]) > CENSUS_TOL
            or abs(mean_w - banked["mean_windows"]) > CENSUS_TOL):
        raise SystemExit(
            "WINDOW-CENSUS ABORT: the assembled mix does not reproduce the banked "
            "combined census - the presentation changed, do not train")


def make_build_mix(arm):
    """The 8-source mix, read the way the flagship reads it: evidence
    UNTRUNCATED, then windowed 1,500/750."""

    def build_mix():
        with arm.untruncated_evidence():
            claims, chunks, y, tags = arm.H108.public_train()
        if len(y) != EXPECTED_CLEAN_ROWS:
            raise SystemExit(
                f"CENSUS ABORT: clean mix {len(y)} rows "
                f"(want {EXPECTED_CLEAN_ROWS}) - this is NOT the incumbent mix")

        for fname, group, n_rows, n_pairs, fams in QUANT_LANES:
            df = pl.read_parquet(HERE / fname)
            got_fams = {r["neg_family"]: int(r["count"])
                        for r in df["neg_family"].value_counts().to_dicts()}
            got_pairs = df["pair_id"].n_unique()
            if len(df) != n_rows or got_pairs != n_pairs or got_fams != fams:
                raise SystemExit(
                    f"LANE ABORT ({group}): {len(df)} rows / {got_pairs} pairs "
                    f"{got_fams} != {n_rows} rows / {n_pairs} pairs {fams}")
            claims += df["claim"].to_list()
            chunks += df["chunk"].to_list()  # UNTRUNCATED - the flagship protocol
            y = np.concatenate([y, df["label"].cast(pl.Float32).to_numpy()])
            tags += [group] * len(df)
            print(f"lane {group}: {len(df)} rows  {got_pairs} pairs  {got_fams}  "
                  f"(from {fname}, ALL rows, untruncated)", flush=True)

        for fname, group, n_rows, n_pos, n_keep in R19_LANES:
            df = pl.read_parquet(HERE / fname)
            got_pos = int(df["label"].sum())
            got_tags = set(df["tag"].unique().to_list())
            if len(df) != n_rows or got_pos != n_pos or got_tags != {group}:
                raise SystemExit(
                    f"LANE ABORT ({group}): {len(df)} rows / {got_pos} positives / "
                    f"tags {sorted(got_tags)} != {n_rows} rows / {n_pos} positives / "
                    f"['{group}']")

            # Amendment A1 clause 3 - the DOCUMENTED over-cap drop, taken at the
            # banked trainer's own guard threshold (PAIRS_PER_BATCH) with the
            # banked windowing function, so the drop rule is the trainer's rule
            # and not a second opinion. The count is recorded, never silent.
            lane_chunks = df["chunk"].to_list()
            lane_wsets = [arm.windows(k) for k in lane_chunks]
            keep = [i for i, w in enumerate(lane_wsets)
                    if len(w) <= arm.PAIRS_PER_BATCH]
            n_dropped = len(lane_wsets) - len(keep)
            if len(keep) != n_keep:
                raise SystemExit(
                    f"DROP ABORT ({group}): {len(keep)} rows kept after the "
                    f"{arm.PAIRS_PER_BATCH}-pair cap drop, want {n_keep}")
            if n_dropped:
                dropped_w = sorted((len(lane_wsets[i]) for i in range(len(lane_wsets))
                                    if len(lane_wsets[i]) > arm.PAIRS_PER_BATCH),
                                   reverse=True)
                OVER_CAP_DROP[group] = {
                    "rows_in_parquet": n_rows, "rows_dropped": n_dropped,
                    "rows_kept": len(keep), "cap": arm.PAIRS_PER_BATCH,
                    "dropped_pairs": int(sum(dropped_w)),
                    "dropped_window_counts": dropped_w,
                    "rule": "len(windows(chunk)) > PAIRS_PER_BATCH, the banked "
                            "trainer's own BATCH-CAP guard threshold",
                }

            lane_y = df["label"].cast(pl.Float32).to_numpy()
            lane_claims = df["claim"].to_list()
            claims += [lane_claims[i] for i in keep]
            chunks += [lane_chunks[i] for i in keep]  # UNTRUNCATED - the protocol
            y = np.concatenate([y, lane_y[keep]])
            tags += [group] * len(keep)
            kept_pos = int(lane_y[keep].sum())
            print(f"lane {group}: {len(keep)} rows kept ({n_dropped} dropped over "
                  f"the {arm.PAIRS_PER_BATCH}-pair cap)  {kept_pos} positives "
                  f"({kept_pos / len(keep):.3f})  (from {fname}, untruncated)",
                  flush=True)

        names = tuple(sorted(set(tags)))
        if names != EXPECTED_GROUPS:
            raise SystemExit(
                f"GROUP-MAP ABORT: mix groups {names} != registered {EXPECTED_GROUPS}")
        if len(y) != EXPECTED_MIX_ROWS:
            raise SystemExit(
                f"CENSUS ABORT: mix {len(y)} rows (want {EXPECTED_MIX_ROWS})")

        wsets = [arm.windows(k) for k in chunks]
        census_record(np.array([len(w) for w in wsets], dtype=np.int32))
        return claims, wsets, y.astype("float32"), tags

    return build_mix


def rebind(arm, draw):
    """Seed, mix, group map, checkpoint and result path - nothing else."""
    if arm.RUNS["twin"]["use_adapter"]:
        raise SystemExit("TWIN INTEGRITY ABORT: the dispatched run trains the adapter")
    arm.SEED = DRAWS[draw]
    arm.EXPECTED_GROUPS = EXPECTED_GROUPS
    arm.EXPECTED_MIX_ROWS = EXPECTED_MIX_ROWS
    arm.build_mix = make_build_mix(arm)
    arm.RUNS["twin"]["ckpt"] = CKPT.format(draw=draw)
    arm.RUNS["twin"]["out"] = TRAIN_OUT.format(draw=draw)
    return arm


def relabel_result(draw):
    """The banked trainer writes its own mix description into the result file.
    Correct the descriptive fields so the record names the arm it actually ran;
    every measured number is left untouched."""
    p = HERE / TRAIN_OUT.format(draw=draw)
    res = json.loads(p.read_text())
    res["arm"] = f"h159_enriched_mix_draw{draw}"
    res["experiment"] = ("R19-H159 enriched-mix arm draw "
                         f"{draw} (amendment A1) - flagship recipe verbatim, "
                         "five R19 supply lanes admitted as new DANN groups")
    res["mix"] = ("flagship 721,210 (clean public mix + misbind 30,000 + "
                  "unit_swap 5,540) + FAVA 30,073 + AttributionBench 16,426 + "
                  "MiniCheck 14,356 + PubHealth 12,251 + FinDVer 2,400 = "
                  "796,716 rows, 19 DANN groups")
    res["amendment_A1"] = (
        "FActScore WITHDRAWN (whole-biography evidence unit needs a lane build, "
        "not a presentation tweak); the five remaining lanes admit UNTRUNCATED "
        "per the twin protocol (the 1,500-char lane cap was priced and "
        "rejected); the AttributionBench rows over the 96-pair batch cap are "
        "dropped and recorded in `over_cap_drop`")
    res["over_cap_drop"] = OVER_CAP_DROP
    res["clean_rows"] = EXPECTED_CLEAN_ROWS
    res["lane_rows"] = ({g: n for _f, g, n, _p, _fam in QUANT_LANES}
                        | {g: k for _f, g, _n, _pos, k in R19_LANES})
    res["lane_groups"] = ([g for _f, g, _n, _p, _fam in QUANT_LANES]
                          + [g for _f, g, _n, _pos, _k in R19_LANES])
    res["bars_note"] = ("the `bars`/`control` blocks are the banked G1 twin's; "
                        "H159's registered bars are adjudicated by the coordinator")
    p.write_text(json.dumps(res, indent=2))
    print(f"result relabelled -> {p}", flush=True)


def main():
    global _CENSUS_WRITE
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("train", "windowed", "census"))
    ap.add_argument("--draw", type=int, default=1, choices=tuple(DRAWS))
    args = ap.parse_args()

    if args.stage == "windowed":
        reads = _mod("g1reads", "R16-H142_G1_reads.py")
        rebind(reads.ARM, args.draw)
        reads.out_path = lambda run, mode: HERE / READ_OUT.format(draw=args.draw,
                                                                  mode=mode)
        sys.argv = ["reads", "--run", "twin", "--mode", "windowed"]
        reads.main()
        return

    if args.stage == "train" and not WINDOW_CENSUS.exists():
        raise SystemExit(
            f"CENSUS ABORT: {WINDOW_CENSUS.name} is not on disk - run "
            "--stage census before spending a card")

    arm = rebind(_mod("g1arm", "R16-H142_G1_arm.py"), args.draw)
    sys.argv = ["arm", "--run", "twin"]
    if args.stage == "census":
        _CENSUS_WRITE = True
        sys.argv.append("--census-only")
    arm.main()
    if args.stage == "train":
        relabel_result(args.draw)


if __name__ == "__main__":
    main()

"""R18-H150 CONVERGENCE ARM draw 1 - the twin protocol carrying both lanes.

Registered in docs/experiments/semantic-grounding-experiments.md, block "R18-H150
CONVERGENCE ARM", amendment A1 (scale/unit demoted to reported secondary),
amendment A2 (RESOLVED - scale_word unbuildable, A1's demotion stands), and the
author ruling of 2026-08-12 ~12:30 that fixes the final composition and makes the
arm the attribution test for H146's composition-shaped damage.

The G1 twin trainer and reader (`R16-H142_G1_arm.py`, `R16-H142_G1_reads.py`) are
BANKED: they produced the adjudicated twin numbers and are not edited. This
wrapper rebinds their run-scoped constants and dispatches into their own `main()`,
so the arm trains and reads through byte-identical code. The ONLY new code is the
mix assembly - the banked `build_mix` reads the clean mix alone.

    protocol    the twin recipe VERBATIM - evidence UNTRUNCATED, presented as
                1,500/750 windows, MIL max-over-windows BCE, full trunk at
                lr 1e-5 OneCycleLR 1 epoch, DANN lambda 0.02 Ganin ramp,
                adapter FROZEN at its zero init (TWIN INTEGRITY ABORT guard)
    mix         clean public 685,670 (R10-H108.public_train, untruncated)
                + H146 misbind lane 30,000  -> DANN group `quant_misbind`
                + H150 unit_swap lane 5,540 -> DANN group `quant_scale_unit`
                = 721,210 rows, 14 DANN groups
    seed        1150
    checkpoint  models/R18-H150-arm-draw1
    results     R18-H150_arm_draw1_result.json          (train + in-domain suite)
                R18-H150_arm_draw1_windowed_result.json (PRIMARY blind arena read)

Both lanes are max-1,500-char chunks, so every lane row is a single window and
the untruncated read is a no-op on them - they enter the bag as one window each,
which is exactly the interaction the arm is registered to test.

Census-before-spend: the mix assembly re-derives the window census and aborts if
it disagrees with the banked `R18-H150_window_census.json` combined block.

Stages:
    train      train + the in-domain suite (gold, gold_full, RAGTruth EN + 7)
    windowed   the PRIMARY blind windowed decomposed-min arena read
    census     CPU-only dry run - mix census, window census cross-check, init and
               permutation fingerprints, then exit before any GPU work

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
      uv run python experiments/grounding-semantic/R18-H150_arm_run.py --stage train
"""

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent

SEED = 1150
CKPT = "R18-H150-arm-draw1"
TRAIN_OUT = "R18-H150_arm_draw1_result.json"
READ_OUT = "R18-H150_arm_draw1_{mode}_result.json"

EXPECTED_CLEAN_ROWS = 685_670
EXPECTED_MIX_ROWS = 721_210
EXPECTED_GROUPS = (
    "halueval", "psiloqa", "quant_misbind", "quant_scale_unit", "ragtruth_cn",
    "ragtruth_de", "ragtruth_en", "ragtruth_es", "ragtruth_fr", "ragtruth_hu",
    "ragtruth_it", "ragtruth_pl", "tabfact", "vitaminc",
)

# (file, DANN group, rows, pairs, {neg_family: count}) - the registered lanes at
# their banked scale. Any drift aborts before a card is touched.
LANES = (
    ("R17-H146_lane.parquet", "quant_misbind", 30_000, 15_000,
     {"misbound_row": 21_000, "misbound_col": 9_000}),
    ("R18-H150_scaleunit_lane.parquet", "quant_scale_unit", 5_540, 2_770,
     {"unit_swap": 5_540}),
)

# The banked combined census (R18-H150_window_census.json), re-derived here.
WINDOW_CENSUS = HERE / "R18-H150_window_census.json"
CENSUS_TOL = 0.001


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def census_crosscheck(sizes):
    """Census-before-spend on the PRESENTATION, not just the rows: the assembled
    mix must reproduce the banked combined window census."""
    banked = json.loads(WINDOW_CENSUS.read_text())["combined"]
    share = float((sizes > 1).mean())
    mean_w = float(sizes.mean())
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
    """The 3-source mix, read the way the twin trainer reads the clean mix:
    evidence UNTRUNCATED, then windowed 1,500/750. Replaces the banked
    `build_mix`, which loads the clean mix alone."""

    def build_mix():
        with arm.untruncated_evidence():
            claims, chunks, y, tags = arm.H108.public_train()
        if len(y) != EXPECTED_CLEAN_ROWS:
            raise SystemExit(
                f"CENSUS ABORT: clean mix {len(y)} rows "
                f"(want {EXPECTED_CLEAN_ROWS}) - this is NOT the incumbent mix")

        for fname, group, n_rows, n_pairs, fams in LANES:
            df = pl.read_parquet(HERE / fname)
            got_fams = {r["neg_family"]: int(r["count"])
                        for r in df["neg_family"].value_counts().to_dicts()}
            got_pairs = df["pair_id"].n_unique()
            if len(df) != n_rows or got_pairs != n_pairs or got_fams != fams:
                raise SystemExit(
                    f"LANE ABORT ({group}): {len(df)} rows / {got_pairs} pairs "
                    f"{got_fams} != {n_rows} rows / {n_pairs} pairs {fams}")
            claims += df["claim"].to_list()
            chunks += df["chunk"].to_list()  # UNTRUNCATED - the twin protocol
            y = np.concatenate([y, df["label"].cast(pl.Float32).to_numpy()])
            tags += [group] * len(df)
            print(f"lane {group}: {len(df)} rows  {got_pairs} pairs  {got_fams}  "
                  f"(from {fname}, ALL rows, untruncated)", flush=True)

        names = tuple(sorted(set(tags)))
        if names != EXPECTED_GROUPS:
            raise SystemExit(
                f"GROUP-MAP ABORT: mix groups {names} != registered {EXPECTED_GROUPS}")
        if len(y) != EXPECTED_MIX_ROWS:
            raise SystemExit(
                f"CENSUS ABORT: mix {len(y)} rows (want {EXPECTED_MIX_ROWS})")

        wsets = [arm.windows(k) for k in chunks]
        census_crosscheck(np.array([len(w) for w in wsets], dtype=np.int32))
        return claims, wsets, y.astype("float32"), tags

    return build_mix


def rebind(arm):
    """Seed, mix, group map, checkpoint and result path - nothing else."""
    if arm.RUNS["twin"]["use_adapter"]:
        raise SystemExit("TWIN INTEGRITY ABORT: the dispatched run trains the adapter")
    arm.SEED = SEED
    arm.EXPECTED_GROUPS = EXPECTED_GROUPS
    arm.EXPECTED_MIX_ROWS = EXPECTED_MIX_ROWS
    arm.build_mix = make_build_mix(arm)
    arm.RUNS["twin"]["ckpt"] = CKPT
    arm.RUNS["twin"]["out"] = TRAIN_OUT
    return arm


def relabel_result():
    """The banked trainer writes its own mix description into the result file.
    Correct the descriptive fields so the record names the arm it actually ran;
    every measured number is left untouched."""
    p = HERE / TRAIN_OUT
    res = json.loads(p.read_text())
    res["arm"] = "h150_convergence_draw1"
    res["experiment"] = ("R18-H150 convergence arm draw 1 - twin windowed-MIL "
                         "protocol + H146 misbind lane + H150 unit_swap lane")
    res["mix"] = ("clean public mix (R10-H108.public_train) + R17-H146 misbind lane "
                  "30,000 + R18-H150 unit_swap lane 5,540")
    res["clean_rows"] = EXPECTED_CLEAN_ROWS
    res["lane_rows"] = {g: n for _f, g, n, _p, _fam in LANES}
    res["lane_groups"] = [g for _f, g, _n, _p, _fam in LANES]
    res["scale_unit_status"] = ("amendment A1 (A2 resolved): REPORTED SECONDARY, "
                                "no bar - 2,770-pair lane at 28% of registered scale")
    res["bars_note"] = ("the `bars`/`control` blocks are the banked G1 twin's; "
                        "H150's registered bars are adjudicated by the coordinator")
    p.write_text(json.dumps(res, indent=2))
    print(f"result relabelled -> {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("train", "windowed", "census"))
    args = ap.parse_args()

    if args.stage == "windowed":
        reads = _mod("g1reads", "R16-H142_G1_reads.py")
        rebind(reads.ARM)
        reads.out_path = lambda run, mode: HERE / READ_OUT.format(mode=mode)
        sys.argv = ["reads", "--run", "twin", "--mode", "windowed"]
        reads.main()
        return

    arm = rebind(_mod("g1arm", "R16-H142_G1_arm.py"))
    sys.argv = ["arm", "--run", "twin"]
    if args.stage == "census":
        sys.argv.append("--census-only")
    arm.main()
    if args.stage == "train":
        relabel_result()


if __name__ == "__main__":
    main()

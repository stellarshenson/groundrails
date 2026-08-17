"""R20-H174 HAGRID/EMANUAL PORTFOLIO ARM - the training draws.

Registered in docs/experiments/semantic-grounding-experiments.md, blocks
"R20-H174 HAGRID/EMANUAL PORTFOLIO ARM" (2026-08-16 ~19:45) and "R20-H174 STAGE
0 COMPLETE" (2026-08-16 ~21:00). The arm adds three stage-0 lanes to the R18-H150
flagship mix and changes NOTHING else in the recipe.

    recipe    the H150 flagship VERBATIM - evidence UNTRUNCATED presented as
              1,500/750 windows, MIL max-over-windows BCE, per-pair domain CE at
              DANN lambda 0.02 with the Ganin ramp, adapter FROZEN at its zero
              init, MAX_LEN 512, <=48 sets / <=96 pairs per batch, LR 1e-5
              OneCycleLR 10% linear warmup, clip 1.0, 1 epoch, H126 double
              seeding. NO EMA, NO window dropout
    mix       clean public 685,670 (R10-H108.public_train, untruncated)
              + H146 misbind        30,000  -> `quant_misbind`
              + H150 unit_swap       5,540  -> `quant_scale_unit`
              + H174 L1              8,000  -> `frame_reject`
              + H174 L2             21,408  -> `attr_pool`
              + H174 L4             10,000  -> `path_bind`
              = 760,618 rows, 17 DANN groups
    seeds     1174 / 2174 / 3174 / 4174 (draws 1-4). Draws 3-4 are declared by
              the registration at k=4 but are COORDINATOR-COMMITTED after draw 1
              clears its gate - this wrapper only makes them runnable
    ckpt      models/R20-H174-arm-draw<N>
    results   R20-H174_arm_draw<N>_result.json          (train + in-domain suite)
              R20-H174_arm_draw<N>_windowed_result.json (blind arena read)

THE CENSUS REBIND (STAGE 1 AMENDMENT, pre-stated in the registration). The
banked `census_crosscheck` compares the assembled presentation against
`R18-H150_window_census.json` (721,210 rows, mean 1.4821 windows, multi-window
0.1908) and hard-aborts on any drift. L2 `attr_pool` pools 4-8 passages per row,
so the portfolio mix presents 760,618 rows, mean 1.5977 windows, multi-window
0.2094 - the banked crosscheck would abort this arm by design. The control is
NOT weakened or bypassed: `R20-H174_window_census.py` recomputes the combined
census from the actual built mix, asserts it against those three registered
figures, banks it as `R20-H174_window_census.json`, and this wrapper repoints
`census_crosscheck` at that file. A mix that drifts from the portfolio geometry
still aborts before a card is touched.

Everything else is banked code, imported not copied: the H150 mix assembly
(`R18-H150_arm_run.make_build_mix`, rebound to the 5-lane LANES tuple), the H160
draw wrapper (`R19-H160_arm_run.rebind`), the cotangent split executor
(`R19-H160_split_exec`, pass A 32 no-grad / pass B 8 grad windows), and the G1
twin trainer/reader. The injection pattern is R20-H172's.

Stages:
    train     train + the in-domain suite, through the split executor
    windowed  the blind windowed decomposed-min arena read
    census    CPU-only dry run - mix census, the rebound window-census
              cross-check, the draw's init and permutation fingerprints

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
      uv run python experiments/grounding-semantic/R20-H174_arm_run.py \
          --stage train --draw 1
"""

import argparse
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent

EXPECTED_CLEAN_ROWS = 685_670
EXPECTED_MIX_ROWS = 760_618

# sorted() order - `make_build_mix` compares tuple(sorted(set(tags))) to this
EXPECTED_GROUPS = (
    "attr_pool", "frame_reject", "halueval", "path_bind", "psiloqa",
    "quant_misbind", "quant_scale_unit", "ragtruth_cn", "ragtruth_de",
    "ragtruth_en", "ragtruth_es", "ragtruth_fr", "ragtruth_hu", "ragtruth_it",
    "ragtruth_pl", "tabfact", "vitaminc",
)

# (file, DANN group, rows, pairs, {neg_family: count}) - the two flagship lanes
# at their banked scale plus the three stage-0 portfolio lanes at the counts
# recorded in the "R20-H174 STAGE 0 COMPLETE" table. Any drift aborts before a
# card is touched.
LANES = (
    ("R17-H146_lane.parquet", "quant_misbind", 30_000, 15_000,
     {"misbound_row": 21_000, "misbound_col": 9_000}),
    ("R18-H150_scaleunit_lane.parquet", "quant_scale_unit", 5_540, 2_770,
     {"unit_swap": 5_540}),
    ("R20-H174_lane_L1.parquet", "frame_reject", 8_000, 4_000,
     {"vacuous_frame": 5_148, "vacuous_marker": 2_852}),
    ("R20-H174_lane_L2.parquet", "attr_pool", 21_408, 10_704,
     {"truth_removed": 14_510, "unsupported_claim": 6_898}),
    ("R20-H174_lane_L4.parquet", "path_bind", 10_000, 5_000,
     {"path_transpose": 6_000, "path_wrong_segment": 4_000}),
)

# The re-banked combined census this arm's crosscheck reads (STAGE 1 AMENDMENT).
WINDOW_CENSUS = HERE / "R20-H174_window_census.json"

DRAWS = {
    1: {"seed": 1174, "ckpt": "R20-H174-arm-draw1",
        "train_out": "R20-H174_arm_draw1_result.json",
        "read_out": "R20-H174_arm_draw1_{mode}_result.json"},
    2: {"seed": 2174, "ckpt": "R20-H174-arm-draw2",
        "train_out": "R20-H174_arm_draw2_result.json",
        "read_out": "R20-H174_arm_draw2_{mode}_result.json"},
    3: {"seed": 3174, "ckpt": "R20-H174-arm-draw3",
        "train_out": "R20-H174_arm_draw3_result.json",
        "read_out": "R20-H174_arm_draw3_{mode}_result.json"},
    4: {"seed": 4174, "ckpt": "R20-H174-arm-draw4",
        "train_out": "R20-H174_arm_draw4_result.json",
        "read_out": "R20-H174_arm_draw4_{mode}_result.json"},
}

# Banked permutation fingerprints absent from `R19-H160_arm_run.BANKED_PERM_FPS`
# (which covers the draws banked up to H156). The draws-1/2 launch added H160
# d3/d4 and H172 d5/d6; the R20-H175b launch widened the same guard to 21 total
# and that wider list is adopted here so the union is identical across arms.
# THIS ARM'S OWN DRAWS 1 AND 2 ARE IN IT - without them draw 3 could collide
# with its own siblings unseen. Widening only strengthens the guard.
EXTRA_PERM_FPS = {
    "a867296772f8314a", "709afd02843c742e",   # R19-H160 d3 / d4
    "a8e708538a5decd8", "a4244751f7bb646b",   # R20-H172 d5 / d6
    "a75c4b59777d442d",                       # R18-H156 d1
    "ded543769d14f9e3", "a42b9d29e07c9db0",   # R20-H174 d1 / d2
    "51dce43a8ae07065", "ad65e0d529fa257b",   # R14-H133 d1 / d2
    "1227e10c9daa2922", "90f8c77667a667ff",   # R14-H135 d1, R17-H145 d1
    "25bd6d194ff18cc6", "39e7dd9d4a12753f",   # R17-H146 d1, R19-H159 d1
}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_h150():
    """The banked H150 mix assembly with the portfolio lanes, group map, row
    count and census target injected. `make_build_mix`'s closure resolves all
    four through this module's globals at call time, so nothing is copied."""
    if not WINDOW_CENSUS.exists():
        raise SystemExit(
            f"CENSUS REBIND ABORT: {WINDOW_CENSUS.name} is not on disk - run "
            "R20-H174_window_census.py first (it asserts the registered "
            "760,618 / 1.5977 / 0.2094 geometry before writing)")
    h150 = _mod("h150arm", "R18-H150_arm_run.py")
    h150.LANES = LANES
    h150.EXPECTED_GROUPS = EXPECTED_GROUPS
    h150.EXPECTED_MIX_ROWS = EXPECTED_MIX_ROWS
    h150.WINDOW_CENSUS = WINDOW_CENSUS
    return h150


def load_w160():
    """The H160 draw wrapper with the H174 draws installed, the permutation
    guard widened to every banked draw, and its H150 load routed to the injected
    mix assembly."""
    w160 = _mod("h160base", "R19-H160_arm_run.py")
    w160.DRAWS = dict(DRAWS)
    w160.BANKED_PERM_FPS |= EXTRA_PERM_FPS
    h150 = load_h150()
    orig_mod = w160._mod

    def routed(name, fname):
        if fname == "R18-H150_arm_run.py":
            return h150
        return orig_mod(name, fname)

    w160._mod = routed
    return w160


def relabel(draw):
    """The banked executor writes the H150 recipe's own description into its
    result file and its checkpoint fingerprint file. Correct the descriptive
    fields so the record names the arm that actually ran; every measured number
    is left untouched."""
    cfg = DRAWS[draw]
    p = HERE / cfg["train_out"]
    res = json.loads(p.read_text())
    res["arm"] = f"h174_portfolio_draw{draw}"
    res["experiment"] = (
        f"R20-H174 hagrid/emanual portfolio arm draw {draw} - the R18-H150 "
        "flagship recipe verbatim plus the L1/L2/L4 stage-0 lanes")
    res["mix"] = ("clean public mix (R10-H108.public_train) + R17-H146 misbind 30,000 "
                  "+ R18-H150 unit_swap 5,540 + R20-H174 L1 frame_reject 8,000 "
                  "+ L2 attr_pool 21,408 + L4 path_bind 10,000")
    res["clean_rows"] = EXPECTED_CLEAN_ROWS
    res["lane_rows"] = {g: n for _f, g, n, _p, _fam in LANES}
    res["lane_groups"] = [g for _f, g, _n, _p, _fam in LANES]
    res["window_census_source"] = WINDOW_CENSUS.name
    res["bars_note"] = ("the bars/control blocks are the banked G1 twin's; H174's "
                        "registered bars, mechanism gates and table guard are "
                        "adjudicated by the coordinator")
    p.write_text(json.dumps(res, indent=2))
    print(f"result relabelled -> {p}", flush=True)

    fpp = HERE.parent.parent / "models" / cfg["ckpt"] / "init_fingerprint.json"
    if fpp.exists():
        fpj = json.loads(fpp.read_text())
        fpj["arm"] = f"h174_portfolio_draw{draw}"
        fpj["recipe"] = ("R20-H174 portfolio - R18-H150 flagship verbatim "
                         "(clean 685,670 + misbind 30,000 + unit_swap 5,540) "
                         "+ L1 8,000 + L2 21,408 + L4 10,000; MIL max-BCE; "
                         "no EMA, no window dropout")
        fpp.write_text(json.dumps(fpj, indent=2))
        print(f"checkpoint fingerprint relabelled -> {fpp}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("train", "windowed", "census"))
    ap.add_argument("--draw", type=int, required=True, choices=tuple(DRAWS))
    ap.add_argument("--max-steps", type=int, default=0,
                    help="smoke mode for --stage train: stop after N steps")
    args = ap.parse_args()

    w160 = load_w160()
    cfg = w160.DRAWS[args.draw]

    if args.stage == "train":
        # The executor re-loads "R19-H160_arm_run.py" through its own `_mod`;
        # route that load to the injected wrapper (the R20-H172 pattern).
        split = _mod("h160split", "R19-H160_split_exec.py")
        orig_mod = split._mod

        def routed(name, fname):
            if fname == "R19-H160_arm_run.py":
                return w160
            return orig_mod(name, fname)

        split._mod = routed
        split.train(args.draw, max_steps=args.max_steps)
        if not args.max_steps:
            relabel(args.draw)
        return

    if args.stage == "windowed":
        reads = _mod("g1reads", "R16-H142_G1_reads.py")
        w160.rebind(reads.ARM, args.draw)
        reads.out_path = lambda run, mode: HERE / cfg["read_out"].format(mode=mode)
        sys.argv = ["reads", "--run", "twin", "--mode", "windowed"]
        reads.main()
        return

    # census - CPU dry run through the banked arm module
    arm = w160.rebind(_mod("g1arm", "R16-H142_G1_arm.py"), args.draw)
    sys.argv = ["arm", "--run", "twin", "--census-only"]
    arm.main()


if __name__ == "__main__":
    main()

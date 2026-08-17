"""R22-H188 DERIVATION-ENHANCED MIX - the training draws.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R22-H188 DERIVATION-ENHANCED MIX" (2026-08-17 ~17:32). The arm adds the
R22-H187 `num_derive` lane to the R18-H150 flagship mix as ONE additional DANN
group and changes NOTHING else in the recipe.

    recipe    the H150 flagship VERBATIM - evidence UNTRUNCATED presented as
              1,500/750 windows, MIL max-over-windows BCE, per-pair domain CE at
              DANN lambda 0.02 with the Ganin ramp, adapter FROZEN at its zero
              init, MAX_LEN 512, <=48 sets / <=96 pairs per batch, LR 1e-5
              OneCycleLR 10% linear warmup, clip 1.0, 1 epoch, H126 double
              seeding. NO EMA, NO window dropout
    mix       clean public 685,670 (R10-H108.public_train, untruncated)
              + H146 misbind        30,000  -> `quant_misbind`
              + H150 unit_swap       5,540  -> `quant_scale_unit`
              + H187 num_derive     30,000  -> `num_derive`
              = 751,210 rows, 15 DANN groups
    seeds     1188 / 2188 (draws 1 / 2) - k=2 per the registration
    ckpt      models/R22-H188-arm-draw<N>
    results   R22-H188_arm_draw<N>_result.json          (train + in-domain suite)
              R22-H188_arm_draw<N>_windowed_result.json (blind arena read)

CENSUS REBIND, the R20-H174 stage-1 pattern. The banked `census_crosscheck`
compares the assembled presentation against `R18-H150_window_census.json`
(721,210 rows) and hard-aborts on drift; the derivation mix presents 751,210
rows and would abort by design. `R22-H188_window_census.py` recomputes the
combined census from the actual built mix, asserts it against the geometry
derived from the banked clean-mix census (751,210 rows / 1.4628 mean windows /
0.1832 multi-window), banks it, and this wrapper repoints `census_crosscheck` at
that file. A mix that drifts from the derivation geometry still aborts before a
card is touched.

PERMUTATION GUARD. `--stage permguard` is the pre-launch check the registration
requires: it derives every fingerprint visible on disk through
`R20_perm_guard.derive_banked_perm_fps` and refuses the seed if its ordering is
already banked. It is deliberately SEPARATE from the launch, because
`derive_banked_perm_fps` also scrapes `logs/*.log` for the line the trainer
itself prints - once this arm's own draw has run, its fingerprint is on disk and
a naive re-check would read the draw as colliding with itself. The check
therefore reports the provenance of any hit and treats a hit whose every source
is this arm's own artifact as SELF, not COLLISION. The trainer's own in-process
guard keeps the banked static set (H160 base + the H174 union), so a resume
after an interruption is never blocked by this arm's own log line.

Everything else is banked code, imported not copied: the H150 mix assembly
(`R18-H150_arm_run.make_build_mix`, rebound to the 3-lane LANES tuple), the H160
draw wrapper (`R19-H160_arm_run.rebind`), the cotangent split executor
(`R19-H160_split_exec`, pass A 32 no-grad / pass B 8 grad windows), and the G1
twin trainer/reader. The injection pattern is R20-H174's.

Stages:
    permguard the pre-launch permutation-collision check (CPU, no mix build)
    census    CPU-only dry run - mix census, the rebound window-census
              cross-check, the draw's init and permutation fingerprints
    train     train + the in-domain suite, through the split executor
    windowed  the blind windowed decomposed-min arena read

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> HF_HUB_OFFLINE=1 \
      uv run python experiments/grounding-semantic/R22-H188_arm_run.py \
          --stage train --draw 1
"""

import argparse
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent

EXPECTED_CLEAN_ROWS = 685_670
EXPECTED_MIX_ROWS = 751_210

# sorted() order - `make_build_mix` compares tuple(sorted(set(tags))) to this
EXPECTED_GROUPS = (
    "halueval", "num_derive", "psiloqa", "quant_misbind", "quant_scale_unit",
    "ragtruth_cn", "ragtruth_de", "ragtruth_en", "ragtruth_es", "ragtruth_fr",
    "ragtruth_hu", "ragtruth_it", "ragtruth_pl", "tabfact", "vitaminc",
)

# (file, DANN group, rows, pairs, {neg_family: count}) - the two flagship lanes
# at their banked scale plus the H187 derivation lane at the counts recorded in
# R22-H187_num_derive_lane_manifest.json. Any drift aborts before a card.
LANES = (
    ("R17-H146_lane.parquet", "quant_misbind", 30_000, 15_000,
     {"misbound_row": 21_000, "misbound_col": 9_000}),
    ("R18-H150_scaleunit_lane.parquet", "quant_scale_unit", 5_540, 2_770,
     {"unit_swap": 5_540}),
    ("R22-H187_num_derive_lane.parquet", "num_derive", 30_000, 15_000,
     {"difference": 10_500, "percentage": 10_500, "sum": 4_500, "product": 4_500}),
)

# The re-banked combined census this arm's crosscheck reads.
WINDOW_CENSUS = HERE / "R22-H188_window_census.json"

DRAWS = {
    1: {"seed": 1188, "ckpt": "R22-H188-arm-draw1",
        "train_out": "R22-H188_arm_draw1_result.json",
        "read_out": "R22-H188_arm_draw1_{mode}_result.json"},
    2: {"seed": 2188, "ckpt": "R22-H188-arm-draw2",
        "train_out": "R22-H188_arm_draw2_result.json",
        "read_out": "R22-H188_arm_draw2_{mode}_result.json"},
}

# The static union the TRAINER's in-process guard carries: the H160 base set
# plus the widened H174 set. Static by design - see the module docstring.
EXTRA_PERM_FPS = {
    "a867296772f8314a", "709afd02843c742e",   # R19-H160 d3 / d4
    "a8e708538a5decd8", "a4244751f7bb646b",   # R20-H172 d5 / d6
    "a75c4b59777d442d",                       # R18-H156 d1
    "ded543769d14f9e3", "a42b9d29e07c9db0",   # R20-H174 d1 / d2
    "51dce43a8ae07065", "ad65e0d529fa257b",   # R14-H133 d1 / d2
    "1227e10c9daa2922", "90f8c77667a667ff",   # R14-H135 d1, R17-H145 d1
    "25bd6d194ff18cc6", "39e7dd9d4a12753f",   # R17-H146 d1, R19-H159 d1
}

# An on-disk hit whose every provenance entry starts with one of these is this
# arm's own record of itself, not another draw's ordering.
SELF_PREFIXES = ("R22-H188",)


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_h150():
    """The banked H150 mix assembly with the derivation lane, group map, row
    count and census target injected. `make_build_mix`'s closure resolves all
    four through that module's globals at call time, so nothing is copied."""
    if not WINDOW_CENSUS.exists():
        raise SystemExit(
            f"CENSUS REBIND ABORT: {WINDOW_CENSUS.name} is not on disk - run "
            "R22-H188_window_census.py first (it asserts the derived "
            "751,210 / 1.4628 / 0.1832 geometry before writing)")
    h150 = _mod("h150arm", "R18-H150_arm_run.py")
    h150.LANES = LANES
    h150.EXPECTED_GROUPS = EXPECTED_GROUPS
    h150.EXPECTED_MIX_ROWS = EXPECTED_MIX_ROWS
    h150.WINDOW_CENSUS = WINDOW_CENSUS
    return h150


def load_w160():
    """The H160 draw wrapper with the H188 draws installed, the trainer's guard
    widened to the banked static union, and its H150 load routed to the injected
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


def permguard(draw):
    """The pre-launch collision check, against every fingerprint on disk.

    Prints nothing in the trainer's `perm fingerprint <hex>` form, because
    `R20_perm_guard` scrapes exactly that form out of `logs/*.log` and this
    stage's own log must not become a phantom banked draw.
    """
    import numpy as np

    guard = _mod("permguard", "R20_perm_guard.py")
    arm = _mod("g1arm", "R16-H142_G1_arm.py")
    seed = DRAWS[draw]["seed"]
    perm = np.random.default_rng(seed).permutation(EXPECTED_MIX_ROWS)
    fp = arm.perm_fingerprint(perm)
    banked = guard.derive_banked_perm_fps()
    hit = banked.get(fp)
    verdict = "DISTINCT"
    if hit:
        foreign = [h for h in hit if not h.startswith(SELF_PREFIXES)]
        verdict = "SELF" if not foreign else "COLLISION"
    print(f"draw {draw} seed {seed} rows {EXPECTED_MIX_ROWS}", flush=True)
    print(f"  ordering digest [{fp}]  vs {len(banked)} orderings derived from disk",
          flush=True)
    print(f"  verdict: {verdict}" + (f"  sources {hit}" if hit else ""), flush=True)
    if verdict == "COLLISION":
        raise SystemExit(
            f"PERMUTATION COLLISION ABORT: draw {draw} (seed {seed}) draws ordering "
            f"[{fp}], already banked by {hit}. Two draws sharing a permutation are "
            "not independent draws - do not launch.")
    return fp


def relabel(draw):
    """The banked executor writes the H150 recipe's own description into its
    result file and its checkpoint fingerprint file. Correct the descriptive
    fields so the record names the arm that actually ran; every measured number
    is left untouched."""
    cfg = DRAWS[draw]
    p = HERE / cfg["train_out"]
    res = json.loads(p.read_text())
    res["arm"] = f"h188_derivation_draw{draw}"
    res["experiment"] = (
        f"R22-H188 derivation-enhanced mix draw {draw} - the R18-H150 flagship "
        "recipe verbatim plus the R22-H187 num_derive lane")
    res["mix"] = ("clean public mix (R10-H108.public_train) + R17-H146 misbind 30,000 "
                  "+ R18-H150 unit_swap 5,540 + R22-H187 num_derive 30,000")
    res["clean_rows"] = EXPECTED_CLEAN_ROWS
    res["lane_rows"] = {g: n for _f, g, n, _p, _fam in LANES}
    res["lane_groups"] = [g for _f, g, _n, _p, _fam in LANES]
    res["window_census_source"] = WINDOW_CENSUS.name
    res["bars_note"] = ("the bars/control blocks are the banked G1 twin's; H188's "
                        "registered bars - PRIMARY FinDVer-numeric 2-draw mean "
                        ">= 0.55, CONTROL ie/knowledge within 0.02, KILL < 0.52, "
                        "and the arena guards - are adjudicated by the coordinator")
    p.write_text(json.dumps(res, indent=2))
    print(f"result relabelled -> {p}", flush=True)

    fpp = HERE.parent.parent / "models" / cfg["ckpt"] / "init_fingerprint.json"
    if fpp.exists():
        fpj = json.loads(fpp.read_text())
        fpj["arm"] = f"h188_derivation_draw{draw}"
        fpj["recipe"] = ("R22-H188 derivation - R18-H150 flagship verbatim "
                         "(clean 685,670 + misbind 30,000 + unit_swap 5,540) "
                         "+ num_derive 30,000; MIL max-BCE; no EMA, no window "
                         "dropout")
        fpp.write_text(json.dumps(fpj, indent=2))
        print(f"checkpoint fingerprint relabelled -> {fpp}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("permguard", "census", "train", "windowed"))
    ap.add_argument("--draw", type=int, required=True, choices=tuple(DRAWS))
    ap.add_argument("--max-steps", type=int, default=0,
                    help="smoke mode for --stage train: stop after N steps")
    args = ap.parse_args()

    if args.stage == "permguard":
        permguard(args.draw)
        return

    w160 = load_w160()
    cfg = w160.DRAWS[args.draw]

    if args.stage == "train":
        # The executor re-loads "R19-H160_arm_run.py" through its own `_mod`;
        # route that load to the injected wrapper (the R20-H174 pattern).
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

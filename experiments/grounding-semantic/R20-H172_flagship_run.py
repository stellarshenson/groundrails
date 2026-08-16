"""R20-H172 FLAGSHIP VARIANCE DRAWS - draws 5 and 6 of the R18-H150 recipe.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R20-H172 FLAGSHIP VARIANCE DRAWS" (2026-08-16). NOT a hypothesis arm: no new
mechanism, no promotion bar. It executes author ruling 2 of 2026-08-16 - the
flagship recipe extends to k=6 same-recipe draws so its mean carries a real
standard error (pooled per-draw sd 0.01189 -> SE at k=6 = 0.00485).

The R19-H160 wrapper (`R19-H160_arm_run.py`) and the cotangent split executor
(`R19-H160_split_exec.py`) are BANKED - they trained the adjudicated draws 3
and 4 and are not edited. This wrapper:

    1  injects DRAWS[5] / DRAWS[6] into the loaded H160 module
           seed 5150 / 6150   (draws 1-4: 1150, 2150, 3150, 4150)
           checkpoints models/R20-H172-arm-draw{5,6}
           results     R20-H172_arm_draw{5,6}_result.json
                       R20-H172_arm_draw{5,6}_{mode}_result.json
    2  extends BANKED_PERM_FPS with the d3/d4 fingerprints (a867296772f8314a,
       709afd02843c742e) so the split executor's census guard covers all six
       banked draws
    3  patches `R19-H160_split_exec._mod` so the executor's own re-load of
       "R19-H160_arm_run.py" resolves to the INJECTED module - without this the
       executor would see the pristine DRAWS dict and KeyError on draw 5

Everything else - the H150 3-source mix assembly, the G1 twin trainer/reader,
the window-chunked step, the resume payload - is the banked code, imported not
copied.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
      uv run python experiments/grounding-semantic/R20-H172_flagship_run.py \
          --stage train --draw 5
"""

import argparse
import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).parent

NEW_DRAWS = {
    5: {"seed": 5150, "ckpt": "R20-H172-arm-draw5",
        "train_out": "R20-H172_arm_draw5_result.json",
        "read_out": "R20-H172_arm_draw5_{mode}_result.json"},
    6: {"seed": 6150, "ckpt": "R20-H172-arm-draw6",
        "train_out": "R20-H172_arm_draw6_result.json",
        "read_out": "R20-H172_arm_draw6_{mode}_result.json"},
}

# R19-H160 draws 3/4, absent from the banked guard set because they were the
# new fingerprints of that arm; draws 5/6 must stay distinct from them too.
H160_PERM_FPS = {"a867296772f8314a", "709afd02843c742e"}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_w160():
    """The H160 wrapper with draws 5/6 injected and the guard set extended."""
    w160 = _mod("h160base", "R19-H160_arm_run.py")
    w160.DRAWS.update(NEW_DRAWS)
    w160.BANKED_PERM_FPS |= H160_PERM_FPS
    return w160


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("train", "windowed", "census"))
    ap.add_argument("--draw", type=int, required=True, choices=(5, 6))
    ap.add_argument("--max-steps", type=int, default=0,
                    help="smoke mode for --stage train: stop after N steps")
    args = ap.parse_args()

    w160 = load_w160()
    cfg = w160.DRAWS[args.draw]

    if args.stage == "train":
        # The executor re-loads "R19-H160_arm_run.py" through its own _mod;
        # route that load to the injected module.
        split = _mod("h160split", "R19-H160_split_exec.py")
        orig_mod = split._mod

        def routed(name, fname):
            if fname == "R19-H160_arm_run.py":
                return w160
            return orig_mod(name, fname)

        split._mod = routed
        split.train(args.draw, max_steps=args.max_steps)
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

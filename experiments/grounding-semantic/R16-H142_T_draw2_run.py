"""R16-H142-T draw 2 - the twin leg of the G1 campaign at seed 2142.

The draw-1 trainer and reader (`R16-H142_G1_arm.py`, `R16-H142_G1_reads.py`) are
BANKED: they produced the adjudicated draw-1 numbers and are not edited. This
wrapper rebinds their three run-scoped constants - the seed, the checkpoint
directory and the result filenames - and dispatches into their own `main()`, so
draw 2 trains and reads through byte-identical code.

    seed        2142 (draw 1 was 1142) - the ONLY intended difference
    checkpoint  models/R16-H142-T-draw2
    results     R16-H142_T_draw2_result.json          (train + in-domain suite,
                                                       gold_full inside it)
                R16-H142_T_draw2_windowed_result.json (PRIMARY blind arena read)

The adapter stays frozen at its zero init: the wrapper only ever dispatches the
`twin` run, whose `use_adapter` is False, and the trainer's own TWIN INTEGRITY
ABORT fires if the adapter output layer moves off zero.

Stages:
    train      train + the in-domain suite (gold, gold_full, RAGTruth EN + 7)
    windowed   the PRIMARY blind windowed decomposed-min arena read
    census     CPU-only dry run - build the mix, print the census, the init
               fingerprint and the permutation fingerprint, then exit

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
      uv run python experiments/grounding-semantic/R16-H142_T_draw2_run.py --stage train
"""

import argparse
import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).parent

SEED = 2142
CKPT = "R16-H142-T-draw2"
TRAIN_OUT = "R16-H142_T_draw2_result.json"
READ_OUT = "R16-H142_T_draw2_{mode}_result.json"


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rebind(arm):
    arm.SEED = SEED
    arm.RUNS["twin"]["ckpt"] = CKPT
    arm.RUNS["twin"]["out"] = TRAIN_OUT
    return arm


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


if __name__ == "__main__":
    main()

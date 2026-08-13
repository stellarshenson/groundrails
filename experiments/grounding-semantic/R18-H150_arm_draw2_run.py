"""R18-H150 CONVERGENCE ARM draw 2 - amendment A3's confirming draw, seed 2150.

The H150 arm dispatcher (`R18-H150_arm_run.py`) - and through it the banked G1
twin trainer and reader (`R16-H142_G1_arm.py`, `R16-H142_G1_reads.py`) - are
BANKED: they produced the adjudicated draw-1 numbers and are not edited. This
wrapper rebinds the dispatcher's four run-scoped constants - the seed, the
checkpoint directory and the two result filenames - plus the result relabel
strings, and dispatches into its own `main()`, so draw 2 trains and reads
through byte-identical code.

    seed        2150 (draw 1 was 1150) - the ONLY intended difference
    checkpoint  models/R18-H150-arm-draw2
    results     R18-H150_arm_draw2_result.json          (train + in-domain suite,
                                                       gold_full inside it)
                R18-H150_arm_draw2_windowed_result.json (PRIMARY blind arena read)

The mix is UNCHANGED: clean public 685,670 + H146 misbind lane 30,000 + H150
unit_swap lane 5,540 = 721,210 rows, 14 DANN groups - the dispatcher's own mix
assembly aborts (clean census, lane census, group map, window-census
cross-check) fire exactly as in draw 1.

Stages (the dispatcher's own):
    train      train + the in-domain suite (gold, gold_full, RAGTruth EN + 7)
    windowed   the PRIMARY blind windowed decomposed-min arena read
    census     CPU-only dry run - mix census, window census cross-check, init and
               permutation fingerprints, then exit before any GPU work

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
      uv run python experiments/grounding-semantic/R18-H150_arm_draw2_run.py --stage train
"""

import argparse
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent

SEED = 2150
CKPT = "R18-H150-arm-draw2"
TRAIN_OUT = "R18-H150_arm_draw2_result.json"
READ_OUT = "R18-H150_arm_draw2_{mode}_result.json"


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rebind(h150):
    """Seed, checkpoint and result paths on the dispatcher - nothing else about
    the run changes. The result relabel is wrapped so the record names draw 2;
    every other descriptive field is the dispatcher's own, written verbatim."""
    h150.SEED = SEED
    h150.CKPT = CKPT
    h150.TRAIN_OUT = TRAIN_OUT
    h150.READ_OUT = READ_OUT

    orig_relabel = h150.relabel_result

    def relabel_draw2():
        orig_relabel()
        p = HERE / TRAIN_OUT
        res = json.loads(p.read_text())
        res["arm"] = "h150_convergence_draw2"
        res["experiment"] = res["experiment"].replace("draw 1", "draw 2")
        p.write_text(json.dumps(res, indent=2))
        print(f"result relabelled (draw 2) -> {p}", flush=True)

    h150.relabel_result = relabel_draw2
    return h150


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("train", "windowed", "census"))
    args = ap.parse_args()

    h150 = rebind(_mod("h150arm", "R18-H150_arm_run.py"))
    sys.argv = ["h150arm", "--stage", args.stage]
    h150.main()


if __name__ == "__main__":
    main()

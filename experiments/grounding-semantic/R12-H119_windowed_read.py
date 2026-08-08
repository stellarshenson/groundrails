"""R12-H119 - windowed blind read with the numeric-surface canonicalizer applied
symmetrically to claim and evidence immediately before tokenization.

This is R8-H101's read with ONE injection and nothing else. The checkpoint, the
gate data, the sentence splitter, the 1,500/750 window geometry, the
max-over-windows and the min-over-sentences are byte-identical - R8-H101's own
module is imported and its `main()` is called; only `ARENA.score_student` is
wrapped so that every claim sentence and every evidence window passes through
the transform on its way to the tokenizer. Windows are cut from the ORIGINAL
chunk text, so window geometry is unchanged by construction.

Direction (binding amendment 4): `strip` removes thousands separators,
`add` inserts them. Non-thousands shipped rules are direction-free.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      uv run python experiments/grounding-semantic/R12-H119_windowed_read.py \
        --model models/R9-H105-mmbert-dann-clean --direction strip \
        --out R12-H119_h105d1_strip_windowed_result.json

--out takes a BARE filename; the read script prepends its own directory (a
doubled path is on the record as a prior bug).
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import argparse
import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).parent


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CANON = _mod("canon", "R12-H119_canon.py")
H101 = _mod("h101", "R8-H101_windowed_read.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="bare filename; the read script prepends its dir")
    ap.add_argument("--direction", choices=("strip", "add"), required=True)
    args = ap.parse_args()

    if "/" in args.out:
        raise SystemExit(f"--out must be a bare filename, got {args.out!r}")

    tf = CANON.transform(args.direction)
    inner = H101.ARENA.score_student

    def wrapped(path, claims, chunk_lists):
        return inner(path, [tf(c) for c in claims], [[tf(k) for k in ks] for ks in chunk_lists])

    H101.ARENA.score_student = wrapped

    print(f"R12-H119 read: direction={args.direction} rules={list(CANON.SHIPPED)}", flush=True)
    sys.argv = ["R8-H101_windowed_read.py", "--model", args.model, "--out", args.out]
    H101.main()


if __name__ == "__main__":
    main()

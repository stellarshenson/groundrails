"""R14-H130 (R14-A1) - the three fresh per-(sentence, window) dumps.

Checkpoints absent from the spent dump set, per the registered bar: DR-control
draw 1, DR-control draw 2, H117-margin draw 1.

`R13_reads_dump.dump` is imported and called unmodified; only its checkpoint
table and output-path function are rebound, so the dumps are byte-identical in
construction to `R13_dump_h105d1/d2` and `R13_dump_h108d1/d2`.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/R14_H130_dumps.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import importlib.util
import pathlib

HERE = pathlib.Path(__file__).parent


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RD = _mod("rd", "R13_reads_dump.py")

TAGS = {
    "drc1": "DR_lane_draw1_control_windowed_result.json",
    "drc2": "DR_lane_draw2_control_windowed_result.json",
    "mgn1": "DR_lane_draw1_margin_windowed_result.json",
}

RD.CHECKPOINTS = dict(TAGS)
RD.dump_path = lambda tag: HERE / f"R14_H130_dump_{tag}.parquet"


def main():
    for tag in TAGS:
        RD.dump(tag)


if __name__ == "__main__":
    main()

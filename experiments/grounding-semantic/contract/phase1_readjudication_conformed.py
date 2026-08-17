"""Phase-1 re-adjudication - the C-A1 structural C1 test on the CONFORMED artifacts.

CPU ONLY, HF_HUB_OFFLINE. Reads the conformed parquets already on disk; builds
nothing. Two of them (psiloqa, tabfact) were conformed BEFORE amendments C-A1
and C-A2 were adopted and carry no structural reading, so the test is run here.
`vitaminc_conformed` already reports 0 and is re-measured as a cross-check.

Writes phase1_readjudication_conformed.json. Measurement only - no verdicts.

Run:  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run python \
      experiments/grounding-semantic/contract/phase1_readjudication_conformed.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HF_HUB_OFFLINE"] = "1"

import importlib.util
import json
import pathlib

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent

spec = importlib.util.spec_from_file_location(
    "structmod", HERE / "phase1_readjudication_structural.py"
)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

# (label, path, claim column, evidence column, the clause the rebuild was built for)
TARGETS = [
    ("psiloqa_conformed", HERE / "psiloqa_conformed.parquet", "claim", "chunk",
     "C1 (struck band) + C2"),
    ("vitaminc_conformed", HERE / "vitaminc_conformed.parquet", "claim", "chunk",
     "C2 (+ C1 structural, already removed by its own F2)"),
    ("tabfact_conformed", HERE / "tabfact_member_conformed.parquet", "claim",
     "chunk_untrunc", "C2 + C3 + C8"),
    ("quant_misbind_conformed", SEM / "R17-H146_lane_conformed.parquet", "claim",
     "chunk", "C2 + C3 + C8"),
    ("attr_pool_conformed", SEM / "R20-H174_lane_L2_conformed.parquet", "claim",
     "chunk", "C2 + C6 + C8"),
    ("halueval_conformed", HERE / "halueval_conformed.parquet", "claim", "chunk",
     "C5"),
    ("halueval_besteffort", HERE / "halueval_conform_best_effort.parquet", "claim",
     "chunk", "C5"),
]


def main():
    out = {
        "task": "C-A1 structural C1 test and a uniform predicate-blind attestation "
                "reading on every conformed artifact already on disk",
        "contract": "docs/experiments/dataset-contract.md, amendments C-A1 and C-A2",
        "compute": "CPU only, HF_HUB_OFFLINE=1; artifacts READ, nothing rebuilt",
        "artifacts": {},
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    }
    for name, path, cc, ec, built_for in TARGETS:
        if not path.exists():
            out["artifacts"][name] = {"status": "MISSING", "path": str(path)}
            print(f"{name}: MISSING {path}", flush=True)
            continue
        df = pl.read_parquet(path)
        claims = df[cc].to_list()
        chunks = df[ec].to_list()
        y = df["label"].cast(pl.Int8).to_numpy()
        rec = {
            "path": str(path.relative_to(SEM.parent.parent)),
            "rows": int(df.height),
            "built_for_clause": built_for,
            "label_1_rows": int((y == 1).sum()),
            "label_0_rows": int((y == 0).sum()),
            "structural_C1": S.structural_all_forms(claims, chunks, y),
            "uniform_containment_C1": S.containment_legs(claims, chunks, y, name),
        }
        out["artifacts"][name] = rec
        print(f"{name}: rows {rec['rows']}, structural raw "
              f"{rec['structural_C1']['raw']['both_label_pairs']} pairs / "
              f"{rec['structural_C1']['raw']['rows_covered']} rows", flush=True)

    p = HERE / "phase1_readjudication_conformed.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()

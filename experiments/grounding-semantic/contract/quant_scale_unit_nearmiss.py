"""C2/C4 supplement - near-duplicate (not just exact) overlap between the
`quant_scale_unit` lane and every held-out mechanism eval, plus `gold_full`.

CPU ONLY.  C2 tests three EXACT string forms.  The stem-collision finding
(13 TabFact documents shared with `R20-H177_eval_B` at zero shared passages)
makes the near-duplicate question live: two serialisations of one table are not
byte-identical, so exact matching cannot see them.  This runs the banked
R14-H136 instrument (8-gram, Jaccard >= 0.3, bidirectional, KILL above 2%)
against each surface, and reports the shared-document stratum separately.

Run:  CUDA_VISIBLE_DEVICES= uv run python \
      experiments/grounding-semantic/contract/quant_scale_unit_nearmiss.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib.util
import json
import pathlib

import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
LANE = SEM / "R18-H150_scaleunit_lane.parquet"
OUT = HERE / "quant_scale_unit_nearmiss.json"

SURFACES = (
    ("R17-H143_evalset", "R17-H143_evalset.parquet"),
    ("R18-H150_unitswap_probe", "R18-H150_unitswap_probe.parquet"),
    ("R20-H177_eval_B", "R20-H177_eval_B.parquet"),
    ("R20-H177_eval_C", "R20-H177_eval_C.parquet"),
    ("R17-H148_probe", "R17-H148_probe.parquet"),
    ("R17-H149_probe", "R17-H149_probe.parquet"),
)


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, SEM / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    gate = _mod("gate", "provenance_gate.py")
    lane = pl.read_parquet(LANE)
    lane_chunks = lane["chunk"].unique().to_list()
    lane_docs = set(lane["doc_id"].unique().to_list())

    res = {"member": "quant_scale_unit", "instrument":
           "provenance_gate.py, n=8, jaccard 0.3, bidirectional, KILL 0.02",
           "lane_distinct_chunks": len(lane_chunks)}
    per = {}
    for name, fname in SURFACES:
        p = SEM / fname
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        chunks = d["chunk"].unique().to_list()
        r = gate.run_gate(lane_chunks, n=8, jaccard=0.3,
                          arena_texts={name: chunks},
                          label="quant_scale_unit_chunks")
        r.pop("hit_examples", None)
        blk = {
            "surface_distinct_chunks": len(chunks),
            "lane_to_surface_fraction": r["candidate_vs_arena"]["fraction"],
            "surface_to_lane_fraction": r["arena_vs_candidate"]["fraction"],
            "max_fraction": r["max_fraction"],
            "verdict": r["verdict"],
            "lane_best_jaccard": r["candidate_vs_arena"].get("best_jaccard"),
        }
        if "doc_id" in d.columns:
            shared = lane_docs & set(d["doc_id"].unique().to_list())
            blk["shared_doc_ids"] = len(shared)
            if shared:
                sub_lane = lane.filter(pl.col("doc_id").is_in(list(shared)))
                sub_surf = d.filter(pl.col("doc_id").is_in(list(shared)))
                rs = gate.run_gate(
                    sub_lane["chunk"].unique().to_list(), n=8, jaccard=0.3,
                    arena_texts={f"{name}_shared_docs":
                                 sub_surf["chunk"].unique().to_list()},
                    label="quant_scale_unit_shared_doc_chunks")
                rs.pop("hit_examples", None)
                blk["shared_document_stratum"] = {
                    "lane_chunks": rs["candidate"]["n_units"],
                    "surface_chunks": rs["arena"]["n_units"],
                    "lane_to_surface_fraction": rs["candidate_vs_arena"]["fraction"],
                    "surface_to_lane_fraction": rs["arena_vs_candidate"]["fraction"],
                    "max_fraction": rs["max_fraction"],
                    "lane_best_jaccard": rs["candidate_vs_arena"].get("best_jaccard"),
                }
        per[name] = blk
    res["per_surface"] = per
    res["worst_max_fraction_all_surfaces"] = max(
        v["max_fraction"] for v in per.values())
    res["worst_shared_document_stratum_fraction"] = max(
        (v["shared_document_stratum"]["max_fraction"]
         for v in per.values() if "shared_document_stratum" in v), default=0.0)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

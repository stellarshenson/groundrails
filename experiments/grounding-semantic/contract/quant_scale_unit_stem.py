"""C3 supplement - the TabFact document-id stem collision, for `quant_scale_unit`.

CPU ONLY.  R20-H177's eval_B assessment (2026-08-17 ~08:40) found that TabFact's
`1-`/`2-` csv-id prefixes render one serialised table under two document ids, so
an id-keyed "disjoint" split can put the two halves of one table on opposite
sides.  This measures the member's exposure to that mode: stem collisions inside
the lane, and stem collisions between the lane and every sibling evaluation
surface that carries TabFact documents.

Run:  CUDA_VISIBLE_DEVICES= uv run python \
      experiments/grounding-semantic/contract/quant_scale_unit_stem.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import collections
import json
import pathlib
import re

import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
LANE = SEM / "R18-H150_scaleunit_lane.parquet"
OUT = HERE / "quant_scale_unit_stem.json"

STEM = re.compile(r"^(?:tabfact:)?[12]-")


def stem(doc_id):
    """Strip the source prefix and TabFact's `1-`/`2-` csv-id prefix."""
    d = doc_id.split(":", 1)[1] if ":" in doc_id else doc_id
    return re.sub(r"^[12]-", "", d)


SURFACES = (
    ("R17-H143_evalset", "R17-H143_evalset.parquet"),
    ("R18-H150_unitswap_probe", "R18-H150_unitswap_probe.parquet"),
    ("R20-H177_eval_B", "R20-H177_eval_B.parquet"),
    ("R20-H177_eval_C", "R20-H177_eval_C.parquet"),
    ("R17-H148_probe", "R17-H148_probe.parquet"),
)


def main():
    lane = pl.read_parquet(LANE)
    docs = lane["doc_id"].unique().to_list()
    tf = [d for d in docs if d.startswith("tabfact:")]
    stems = collections.Counter(stem(d) for d in tf)
    collide = {s: c for s, c in stems.items() if c > 1}

    res = {
        "member": "quant_scale_unit",
        "lane_documents": len(docs),
        "lane_tabfact_documents": len(tf),
        "lane_feverous_documents": len(docs) - len(tf),
        "within_lane_tabfact_stem_collisions": len(collide),
        "within_lane_collision_examples": dict(list(collide.items())[:5]),
    }

    lane_stems = set(stems)
    per = {}
    for name, fname in SURFACES:
        p = SEM / fname
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        if "doc_id" not in d.columns:
            per[name] = {"status": "no doc_id column"}
            continue
        od = [x for x in d["doc_id"].unique().to_list() if x]
        otf = [x for x in od if str(x).startswith("tabfact:")]
        os_ = {stem(x) for x in otf}
        shared_ids = set(od) & set(docs)
        per[name] = {
            "their_documents": len(od),
            "their_tabfact_documents": len(otf),
            "shared_doc_ids": len(shared_ids),
            "shared_tabfact_stems": len(os_ & lane_stems),
            "stems_shared_but_ids_not": len(os_ & lane_stems) - len(
                {stem(x) for x in shared_ids if str(x).startswith("tabfact:")}),
        }
    res["vs_sibling_surfaces"] = per
    res["worst_stems_shared_but_ids_not"] = max(
        (v.get("stems_shared_but_ids_not", 0) for v in per.values()), default=0)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

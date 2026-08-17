"""C3 supplement for `quant_misbind` - DOCUMENT-level disjointness against the
FEVEROUS-derived held-out eval, re-run on the BUILT lane.

Why this is needed.  `R17-H144_pairs.evalset_documents` records, from its own
measurement, that the FEVEROUS document ids are positional indices over a
`unique()` that is not order-stable - 0 of 95 sampled ids resolved to the table
their claim came from.  So for the 10,110 FEVEROUS rows of this lane the `doc_id`
column is NOT a stable document identity, and no document-disjointness claim can
rest on it.  String comparison (C2) cannot fill the gap either: two different
serializations of the SAME table share no chunk string.

The build's own instrument does fill it - `excluded_tables()` matches a table to
an eval chunk on distinctive long strings plus canonical-numeral overlap.  It was
applied to the CANDIDATE pool before construction; here it is re-applied to the
tables the lane actually shipped, which is the check that was never run.

CPU ONLY.  Run:
  CUDA_VISIBLE_DEVICES= uv run python \
    experiments/grounding-semantic/contract/quant_misbind_doc_disjoint.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import collections
import importlib.util
import json
import pathlib
import time

import polars as pl

HERE = pathlib.Path(__file__).parent
GS = HERE.parent


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    t0 = time.time()
    P = _mod("h144pairs", GS / "R17-H144_pairs.py")
    lane = pl.read_parquet(GS / "R17-H146_lane.parquet")
    lane_docs = set(lane["doc_id"].to_list())

    excluded_ids, prints, eval_rows, unmatched = P.evalset_documents()
    print(f"eval set: {eval_rows} rows -> {len(excluded_ids)} doc_ids, "
          f"{len(prints)} content fingerprints ({unmatched} unmatched)", flush=True)

    raw = P.tabfact_tables() + P.feverous_tables()
    shipped = [t for t in raw if t["doc_id"] in lane_docs]
    print(f"{len(raw)} source tables; {len(shipped)} carry a doc_id the lane shipped",
          flush=True)

    hits = P.excluded_tables(shipped, prints)
    by_src = collections.Counter(shipped[i]["source"] for i in hits)

    # id-level read, reported with its caveat rather than as evidence
    id_hits = lane_docs & set(excluded_ids)
    ns = collections.Counter(d.split(":")[0] for d in lane_docs)

    out = {
        "clause": "C3 (document-level disjointness supplement)",
        "eval_surface": "R17-H143_evalset.parquet - the held-out mechanism eval this "
                        "lane's source pool was content-excluded against; 35% of its "
                        "source rows are FEVEROUS",
        "why_id_matching_is_insufficient": (
            "R17-H144_pairs.evalset_documents records that the FEVEROUS ids are "
            "positional over a non-order-stable unique(); 0 of 95 sampled ids resolved "
            "to their own table. The lane's 10,110 FEVEROUS rows therefore carry an "
            "identifier that is stable only within the build process that wrote it"
        ),
        "lane_documents_by_namespace_distinct": dict(ns),
        "lane_shipped_tables_resolved": len(shipped),
        "content_gate_rerun_on_shipped_tables": {
            "instrument": "R17-H144_pairs.excluded_tables - three distinctive long "
                          "strings readable in an eval chunk AND >= 60% canonical-"
                          "numeral overlap, both signals required",
            "eval_fingerprints": len(prints),
            "shipped_tables_matching_an_eval_document": len(hits),
            "by_source": dict(by_src),
            "bar": "0 - a shipped table matching an eval document would mean the lane "
                   "trains on the eval's own table under a different serialization",
        },
        "id_level_read_reported_with_caveat": {
            "lane_doc_ids_in_the_eval_document_set": len(id_hits),
            "caveat": "meaningful for the tabfact namespace only; the feverous half of "
                      "this number is uninterpretable for the reason above",
        },
        "seconds": round(time.time() - t0, 1),
    }
    p = HERE / "quant_misbind_c3_doc_disjoint.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()

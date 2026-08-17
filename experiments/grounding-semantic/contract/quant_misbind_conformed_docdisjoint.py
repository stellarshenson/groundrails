"""C2 / C3 DOCUMENT-level disjointness for the conformed `quant_misbind` member.

C2's three string forms cannot see the same table serialized differently, which
is why the original pass could only report a similarity heuristic (precision
0.0000) for that direction.  The conformed member is TabFact-only, so every one
of its documents carries the corpus's own stable `table_id` - and every
evaluation surface that is TabFact-derived carries the same identifier.  The
read below is therefore an IDENTITY read on those surfaces, not a heuristic.

Surfaces and the identifier each carries:
  R17-H143_evalset        resolved to source doc_id via the v2 lane join
  R20-H177_eval_B         `doc_id` column, namespaces tabfact / edgar
  R20-H177_eval_C         `doc_id` column, namespace edgar only
  R20-H175b_qlane_eval*   `doc_id` column where present
  anti-gaming probe sets  `table_id` column (bare TabFact table id)
  arena / gold_full / vitaminc   different corpora - namespace-disjoint

CPU ONLY.  Run:
  CUDA_VISIBLE_DEVICES= uv run python \
    experiments/grounding-semantic/contract/quant_misbind_conformed_docdisjoint.py
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
    df = pl.read_parquet(GS / "R17-H146_lane_conformed.parquet")
    member_docs = set(df["doc_id"].to_list())
    member_ids = {d.split(":", 1)[1] for d in member_docs}
    out = {
        "clause": "C2 + C3 (document-level identity read on the conformed member)",
        "member_documents": len(member_docs),
        "member_namespaces": sorted({d.split(":")[0] for d in member_docs}),
        "surfaces": {},
    }

    # --- R17-H143_evalset, resolved through the v2 lane join
    ev = pl.read_parquet(GS / "R17-H143_evalset.parquet")
    v2 = pl.read_parquet(GS / "R14-H133_lane.v2-SUPERSEDED.parquet",
                         columns=["pair_id", "claim", "label", "chunk", "doc_id"])
    graded = ev.filter(~pl.col("control")).join(
        v2.select(["pair_id", "claim", "label", "doc_id"]).with_columns(
            pl.col("label").cast(pl.Int8)), on=["pair_id", "claim", "label"], how="left")
    ctrl = ev.filter(pl.col("control")).join(
        v2.select(["chunk", "doc_id"]).unique(subset=["chunk"]), on="chunk", how="left")
    eval_docs = set()
    for d in (graded, ctrl):
        eval_docs |= {x for x in d["doc_id"].to_list() if x is not None}
    out["surfaces"]["R17-H143_evalset"] = {
        "identifier": "source doc_id resolved through the R14-H133 v2 lane join",
        "documents": len(eval_docs),
        "by_namespace": dict(collections.Counter(d.split(":")[0] for d in eval_docs)),
        "shared_with_member": len(eval_docs & member_docs),
        "note": "the FEVEROUS side of this surface is unstably keyed; the member no "
                "longer carries any FEVEROUS document, so the read has no "
                "uninterpretable side",
    }

    # --- surfaces carrying a doc_id column
    for fname in ("R20-H177_eval_B.parquet", "R20-H177_eval_C.parquet",
                  "R20-H175b_qlane_eval.parquet",
                  "R20-H175b_qlane_eval_repaired.parquet",
                  "R20-H175b_qlane_eval_clean.parquet",
                  "R20-H175b_qlane_eval_clean_prefix.parquet"):
        p = GS / fname
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        name = fname.replace(".parquet", "")
        if "doc_id" not in d.columns:
            out["surfaces"][name] = {"identifier": "NONE - the surface carries no "
                                                   "document identifier",
                                     "shared_with_member": None}
            continue
        docs = {x for x in d["doc_id"].to_list() if x is not None}
        shared = docs & member_docs
        rows_hit = int(d.filter(pl.col("doc_id").is_in(list(shared))).height) if shared else 0
        pairs_hit = (d.filter(pl.col("doc_id").is_in(list(shared)))["pair_id"].n_unique()
                     if shared else 0)
        out["surfaces"][name] = {
            "identifier": "doc_id column",
            "documents": len(docs),
            "by_namespace": dict(collections.Counter(
                x.split(":")[0] if ":" in x else "(no namespace prefix)" for x in docs)),
            "shared_with_member": len(shared),
            "surface_rows_affected": rows_hit,
            "surface_pairs_affected": int(pairs_hit),
            "surface_pairs_total": int(d["pair_id"].n_unique()),
        }

    # --- anti-gaming probe sets, keyed on a bare TabFact table_id
    for fname in ("R17-H146_antigaming_set.parquet", "R18-H150_antigaming_set.parquet",
                  "R19-H159_antigaming_set.parquet"):
        p = GS / fname
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        name = fname.replace(".parquet", "")
        ids = {x for x in d["table_id"].to_list() if x is not None}
        shared = ids & member_ids
        out["surfaces"][name] = {
            "identifier": "table_id column (bare TabFact table id)",
            "documents": len(ids),
            "shared_with_member": len(shared),
            "surface_items_affected": int(d.filter(pl.col("table_id")
                                                   .is_in(list(shared))).height)
            if shared else 0,
            "surface_items_total": d.height,
            "bind_row_items_total": int(d.filter(pl.col("family") == "bind_row").height),
        }

    # --- corpora that share no identifier namespace with the member
    out["surfaces"]["arena_ragbench_10_subsets"] = {
        "identifier": "RAGBench document ids - a different corpus entirely",
        "shared_with_member": 0,
        "note": "namespace-disjoint by construction; the string read is in C2",
    }
    out["surfaces"]["gold_full"] = {
        "identifier": "gold teacher-pair chunks - a different corpus entirely",
        "shared_with_member": 0,
        "note": "namespace-disjoint by construction; the string read is in C2",
    }
    out["surfaces"]["vitaminc_holdout_superset"] = {
        "identifier": "VitaminC unique_id / case_id - a different corpus entirely",
        "shared_with_member": 0,
        "note": "namespace-disjoint by construction; the string read is in C2",
    }

    vals = [v.get("shared_with_member") for v in out["surfaces"].values()]
    out["all_measurable_surfaces_zero"] = all(v == 0 for v in vals if v is not None)
    out["surfaces_without_a_document_identifier"] = [
        k for k, v in out["surfaces"].items() if v.get("shared_with_member") is None]
    out["seconds"] = round(time.time() - t0, 1)
    p = HERE / "quant_misbind_conformed_c2_docdisjoint.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()

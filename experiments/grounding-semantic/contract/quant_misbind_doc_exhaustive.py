"""C2/C3 - EXHAUSTIVE document-level disjointness of `quant_misbind` against the
R17-H143 held-out mechanism eval.

The build excluded eval documents with `R17-H144_pairs.excluded_tables`, whose
DECISION RULE is

    >= 60% of the eval chunk's canonical numerals lie inside the table's own
    AND >= 3 of the table's distinctive long strings are readable in that chunk

but whose CANDIDATE SELECTION only tests tables reachable from an inverted index
on the chunk's three rarest indexed numerals, where rarity is computed over
whatever table list is passed in.  A table that satisfies the decision rule but
never enters the candidate set is not excluded.

This script applies the decision rule EXHAUSTIVELY - every eval fingerprint
against every table, no candidate heuristic - to

  (a) the tables the lane actually shipped, and
  (b) the full source corpus, so the heuristic's miss rate is quantified rather
      than inferred

and then converts any hit into lane rows and eval pairs.

CPU ONLY.  Run:
  CUDA_VISIBLE_DEVICES= uv run python \
    experiments/grounding-semantic/contract/quant_misbind_doc_exhaustive.py
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


def exhaustive(P, tables, prints, tag):
    t0 = time.time()
    table_nums = []
    table_strs = []
    for t in tables:
        table_nums.append(P.present_numbers(
            " ".join([t["caption"]] + t["hdr"] + [c for r in t["body"] for c in r])))
        table_strs.append(P.long_strings(t))
    hits = {}
    for ci, (chunk, p) in enumerate(prints):
        need = 0.6 * len(p)
        for ti in range(len(tables)):
            if len(p & table_nums[ti]) < need:
                continue
            strs = table_strs[ti]
            hit = sum(1 for s in strs if s in chunk)
            if hit >= min(3, len(strs)) and hit > 0:
                hits.setdefault(ti, []).append(ci)
    print(f"  {tag}: {len(hits)} of {len(tables)} tables hit in "
          f"{round(time.time() - t0, 1)}s", flush=True)
    return hits


def main():
    t0 = time.time()
    P = _mod("h144pairs", GS / "R17-H144_pairs.py")
    lane = pl.read_parquet(GS / "R17-H146_lane.parquet")
    lane_docs = set(lane["doc_id"].to_list())

    excluded_ids, prints, eval_rows, unmatched = P.evalset_documents()
    ev = pl.read_parquet(GS / "R17-H143_evalset.parquet")
    print(f"eval: {eval_rows} rows, {len(prints)} fingerprints", flush=True)

    raw = P.tabfact_tables() + P.feverous_tables()
    shipped_idx = [i for i, t in enumerate(raw) if t["doc_id"] in lane_docs]
    shipped = [raw[i] for i in shipped_idx]
    print(f"{len(raw)} source tables, {len(shipped)} shipped by the lane", flush=True)

    print("exhaustive decision rule over the SHIPPED tables...", flush=True)
    hs = exhaustive(P, shipped, prints, "shipped")
    print("exhaustive decision rule over the FULL source corpus...", flush=True)
    hr = exhaustive(P, raw, prints, "full corpus")

    heur = P.excluded_tables(raw, prints)  # what the build actually excluded

    hit_docs = {shipped[ti]["doc_id"] for ti in hs}
    by_src = collections.Counter(shipped[ti]["source"] for ti in hs)
    rows_hit = lane.filter(pl.col("doc_id").is_in(list(hit_docs)))
    pairs_hit = rows_hit["pair_id"].n_unique()

    eval_chunks = [c for c, _ in prints]
    ev_ci = set()
    for cis in hs.values():
        ev_ci |= set(cis)
    hit_eval_chunks = {eval_chunks[i] for i in ev_ci}
    ev_rows_hit = ev.filter(pl.col("chunk").is_in(list(hit_eval_chunks)))

    # string-level control: do any of these shipped chunks equal an eval chunk?
    same_string = len(set(lane["chunk"].to_list()) & set(eval_chunks))

    out = {
        "clause": "C2 + C3 (exhaustive document-level read)",
        "eval_surface": "R17-H143_evalset.parquet (1,050 rows, 547 content fingerprints)",
        "decision_rule": ">= 60% of the eval chunk's canonical numerals inside the "
                         "table's own AND >= 3 of the table's distinctive long strings "
                         "readable in that chunk - R17-H144_pairs.excluded_tables, "
                         "applied with NO candidate heuristic",
        "source_tables": len(raw),
        "lane_shipped_tables": len(shipped),
        "exhaustive_shipped_tables_matching_an_eval_document": len(hs),
        "exhaustive_shipped_share": round(len(hs) / max(len(shipped), 1), 6),
        "by_source": dict(by_src),
        "lane_rows_from_matching_tables": rows_hit.height,
        "lane_rows_share": round(rows_hit.height / lane.height, 6),
        "lane_pairs_from_matching_tables": int(pairs_hit),
        "eval_fingerprints_matched": len(ev_ci),
        "eval_rows_affected": ev_rows_hit.height,
        "eval_rows_share": round(ev_rows_hit.height / ev.height, 6),
        "exhaustive_full_corpus_tables_matching": len(hr),
        "build_heuristic_excluded": len(heur),
        "heuristic_miss": len(set(hr) - set(heur)),
        "heuristic_recall": round(len(set(hr) & set(heur)) / max(len(hr), 1), 6),
        "string_level_control": {
            "lane_chunks_equal_to_an_eval_chunk": same_string,
            "reading": "0 means the overlap is NOT visible to any string comparison - "
                       "the same table is serialized differently on the two sides, "
                       "which is exactly what C2's string forms cannot see",
        },
        "seconds": round(time.time() - t0, 1),
    }
    p = HERE / "quant_misbind_c2c3_doc_exhaustive.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()

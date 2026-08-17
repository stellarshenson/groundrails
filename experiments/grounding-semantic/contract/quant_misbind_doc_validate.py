"""Instrument validation for the exhaustive document-level read.

The exhaustive pass reports 942 shipped lane tables satisfying the build's
content decision rule against an R17-H143 eval fingerprint.  That number is
worthless until the rule's false-positive rate is known, so it is measured here
against ground truth rather than assumed.

GROUND TRUTH.  `evalset_documents()` resolves every eval row to the v2 lane's
`doc_id`.  For the TabFact namespace that id is the corpus's own `table_id` and
is stable across rebuilds, so for a TabFact-sourced eval chunk the identity of
its true table is KNOWN.  The rule's precision is therefore directly measurable:
of the tables it matches to such a chunk, how many are that chunk's own table?

Reported alongside: how many of the rule's long-string tests were decided on
fewer than three strings (the `min(3, len(strs))` relaxation), which is the
rule's known weak edge.

CPU ONLY.  Run:
  CUDA_VISIBLE_DEVICES= uv run python \
    experiments/grounding-semantic/contract/quant_misbind_doc_validate.py
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

    # rebuild the eval -> doc_id join so each fingerprint carries its true table
    ev = pl.read_parquet(GS / "R17-H143_evalset.parquet")
    v2 = pl.read_parquet(GS / "R14-H133_lane.v2-SUPERSEDED.parquet",
                         columns=["pair_id", "claim", "label", "chunk", "doc_id"])
    graded = ev.filter(~pl.col("control")).join(
        v2.select(["pair_id", "claim", "label", "doc_id"]).with_columns(
            pl.col("label").cast(pl.Int8)),
        on=["pair_id", "claim", "label"], how="left")
    ctrl = ev.filter(pl.col("control")).join(
        v2.select(["chunk", "doc_id"]).unique(subset=["chunk"]), on="chunk", how="left")
    chunk_doc = {}
    for d in (graded, ctrl):
        for c, doc in zip(d["chunk"].to_list(), d["doc_id"].to_list()):
            if doc is not None:
                chunk_doc[c] = doc

    prints = [(c.lower(), frozenset(P.present_numbers(c))) for c in chunk_doc]
    prints = [(c, p) for c, p in prints if len(p) >= 4]
    lower_to_doc = {c.lower(): chunk_doc[c] for c in chunk_doc}
    print(f"{len(prints)} fingerprints carry a resolved source doc_id", flush=True)

    raw = P.tabfact_tables() + P.feverous_tables()
    shipped = [t for t in raw if t["doc_id"] in lane_docs]
    table_nums, table_strs = [], []
    for t in shipped:
        table_nums.append(P.present_numbers(
            " ".join([t["caption"]] + t["hdr"] + [c for r in t["body"] for c in r])))
        table_strs.append(P.long_strings(t))

    tp = fp = 0
    thin = 0
    per_chunk = collections.Counter()
    examples = []
    tabfact_prints = [(c, p) for c, p in prints if lower_to_doc[c].startswith("tabfact:")]
    print(f"{len(tabfact_prints)} of them are TabFact-sourced (stable ids - the "
          f"measurable ground truth)", flush=True)

    for chunk, p in tabfact_prints:
        true_doc = lower_to_doc[chunk]
        need = 0.6 * len(p)
        for ti, t in enumerate(shipped):
            if len(p & table_nums[ti]) < need:
                continue
            strs = table_strs[ti]
            hit = sum(1 for s in strs if s in chunk)
            if hit >= min(3, len(strs)) and hit > 0:
                per_chunk[chunk] += 1
                if len(strs) < 3:
                    thin += 1
                if t["doc_id"] == true_doc:
                    tp += 1
                else:
                    fp += 1
                    if len(examples) < 5:
                        examples.append({
                            "eval_true_table": true_doc,
                            "matched_table": t["doc_id"],
                            "long_strings_in_table": len(strs),
                            "long_strings_matched": hit,
                            "numeral_overlap": round(len(p & table_nums[ti]) / len(p), 3),
                            "eval_numerals": len(p),
                        })

    out = {
        "purpose": "false-positive rate of the build's content decision rule, measured "
                   "against known ground truth",
        "ground_truth": "TabFact-sourced eval fingerprints, whose true source table id "
                        "is the corpus's own stable table_id",
        "fingerprints_total": len(prints),
        "fingerprints_with_stable_ground_truth": len(tabfact_prints),
        "matches_on_those_fingerprints": tp + fp,
        "true_positives_same_table": tp,
        "false_positives_different_table": fp,
        "precision": round(tp / max(tp + fp, 1), 6),
        "matches_decided_on_fewer_than_three_long_strings": thin,
        "mean_matches_per_fingerprint": round(
            sum(per_chunk.values()) / max(len(tabfact_prints), 1), 3),
        "max_matches_for_one_fingerprint": max(per_chunk.values()) if per_chunk else 0,
        "false_positive_examples": examples,
        "reading": "a rule that matches many tables per eval chunk and almost never the "
                   "chunk's own table is a similarity heuristic, not a contamination "
                   "detector; its raw hit count cannot be read as document overlap",
        "seconds": round(time.time() - t0, 1),
    }
    p = HERE / "quant_misbind_c2c3_rule_validation.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()

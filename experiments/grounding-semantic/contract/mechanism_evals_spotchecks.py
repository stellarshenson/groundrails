"""Independent re-derivation of the four strongest findings of the mechanism-eval
contract pass.

CPU only.  Measurement only; no verdict is adjudicated here.

Each check is computed from the archives and parquets DIRECTLY, without the
digest machinery `mechanism_evals_verify.py` uses, so a bug in that machinery
cannot produce the same number twice.  Each one also names the mechanism.

  1. The R18-H150 unit-swap probe's documents against the mix, per member -
     registered as a `document-disjoint probe from unused supply`
  2. The R11-H117 held-out pairs' evidence against the `tabfact` member
  3. The R15 type probe's (claim, evidence) pairs carrying both labels
  4. The R19-H166-A1 VitaminC holdout's (claim, evidence) pairs carrying both
     labels, and the anti-gaming set's claims against the lane its builder does
     NOT guard against

Out: contract/mechanism_evals_spotchecks.json
Run: CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run python \
     experiments/grounding-semantic/contract/mechanism_evals_spotchecks.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections
import io
import json
import pathlib
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
DATA = SEM.parent.parent / "data" / "external" / "datasets"
OUT = HERE / "mechanism_evals_spotchecks.json"
NOTE = "Numbers recorded, not adjudicated - the coordinator adjudicates."


def main():
    res = {"artifact": OUT.name,
           "scope": "independent re-derivation of the strongest findings in "
                    "contract/mechanism_evals_report.json",
           "note": NOTE, "checks": {}}

    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    dt = pl.read_parquet(io.BytesIO(z.read(
        next(x for x in z.namelist() if x.endswith("__train.parquet")))))
    dt = dt.filter(pl.col("statement").str.len_chars() > 10)
    tf_ids = {"tabfact:" + t for t in dt["table_id"].to_list()}
    h146_docs = set(pl.read_parquet(SEM / "R17-H146_lane.parquet")["doc_id"].to_list())
    h150_docs = set(pl.read_parquet(SEM / "R18-H150_scaleunit_lane.parquet")["doc_id"].to_list())

    # ---- 1 ---------------------------------------------------------------- #
    us = pl.read_parquet(SEM / "R18-H150_unitswap_probe.parquet")
    pd_ = set(us["doc_id"].to_list())
    union = tf_ids | h146_docs | h150_docs
    rows = sum(1 for d in us["doc_id"].to_list() if d in union)
    res["checks"]["1_unitswap_probe_document_disjointness"] = {
        "registered_as": "document-disjoint probe from unused supply - 140 pairs "
                         "/ 42 documents (canonical log, R18-H150 block)",
        "probe_documents": len(pd_),
        "in_the_tabfact_member": len(pd_ & tf_ids),
        "in_the_quant_misbind_lane": len(pd_ & h146_docs),
        "in_the_quant_scale_unit_lane_it_was_built_beside": len(pd_ & h150_docs),
        "in_the_union": len(pd_ & union),
        "probe_rows_on_such_a_document": rows, "probe_rows": us.height,
        "share_of_rows": round(rows / us.height, 4),
        "mechanism": "the probe reserved documents the SCALE/UNIT lane did not "
                     "use - and is disjoint from that lane at 0. The mix also "
                     "carries the `tabfact` member and the `quant_misbind` lane, "
                     "which is where the 27 documents come from. C2 exists for "
                     "exactly this: disjointness is owed against every surface, "
                     "`not only the lane a member was built beside`",
        "example_overlapping_ids": sorted(pd_ & union)[:5],
    }

    # ---- 2 ---------------------------------------------------------------- #
    hp = pl.read_parquet(SEM / "R11-H117_heldout_pairs.parquet")
    tf_chunks = {f"{c}\n{t}".replace("\r\n", "\n").replace("#", " | ")[:1500]
                 for c, t in zip(dt["table_caption"].to_list(), dt["table_text"].to_list())}
    hits = [c for c in set(hp["chunk"].to_list()) if c in tf_chunks]
    sub = hp.filter(pl.col("chunk").is_in(hits))
    res["checks"]["2_h117_evidence_in_the_tabfact_member"] = {
        "distinct_evidence": int(hp["chunk"].n_unique()),
        "byte_identical_to_a_tabfact_member_passage": len(hits),
        "share": round(len(hits) / hp["chunk"].n_unique(), 4),
        "rows_affected": sub.height, "rows": hp.height,
        "by_engine": dict(collections.Counter(sub["engine"].to_list())),
        "example_head": hits[0][:180] if hits else None,
        "mechanism": "the DR pilot's H112 engine drew seeds from TabFact tables "
                     "serialised the way the mix serialises them, so 306 of the "
                     "held-out set's 1,933 passages are training passages",
    }

    # ---- 3 ---------------------------------------------------------------- #
    tp = pl.concat([pl.read_parquet(SEM / "R15_P1_typeprobe_quads.parquet"),
                    pl.read_parquet(SEM / "R15_P1_typeprobe_topup_quads.parquet")])
    pos, neg = set(), set()
    for r in tp.iter_rows(named=True):
        for col, y in (("claim_a", 1), ("claim_b", 1), ("claim_c", 0), ("claim_d", 0)):
            (pos if y else neg).add((r[col], r["table_id"]))
    sh = pos & neg
    res["checks"]["3_typeprobe_structural_collisions"] = {
        "pairs_carrying_both_labels": len(sh),
        "examples": [{"claim": c, "table_id": t} for c, t in list(sh)[:3]],
        "mechanism": "the derived-value families print an APPROXIMATE value, so a "
                     "rounded correct answer in one quad equals a wrong-value leg "
                     "in another over the same table",
    }

    # ---- 4 ---------------------------------------------------------------- #
    zv = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    tr = pl.read_parquet(io.BytesIO(zv.read("tals__vitaminc__train.parquet")))
    cand = pl.concat([
        pl.read_parquet(io.BytesIO(zv.read(f"tals__vitaminc__{s}.parquet")))
        .with_columns(pl.lit(s).alias("split")) for s in ("test", "validation")])
    keep = np.ones(cand.height, dtype=bool)
    for col in ("page", "claim", "evidence", "wiki_revision_id", "unique_id", "case_id"):
        keep &= ~cand[col].is_in(list(set(tr[col].to_list()))).to_numpy()
    held = cand.filter(keep).filter(pl.col("label").is_in(["REFUTES", "NOT ENOUGH INFO"]))
    p = {(c, e) for c, e, l in zip(held["claim"], held["evidence"], held["label"])
         if l == "REFUTES"}
    n = {(c, e) for c, e, l in zip(held["claim"], held["evidence"], held["label"])
         if l == "NOT ENOUGH INFO"}
    both = p & n
    ex = []
    for c, e in list(both)[:3]:
        s = held.filter((pl.col("claim") == c) & (pl.col("evidence") == e))
        ex.append({"claim": c[:160], "evidence": e[:160],
                   "labels": s["label"].to_list(),
                   "unique_ids": s["unique_id"].to_list()})
    res["checks"]["4a_h166a1_structural_collisions"] = {
        "rows": held.height,
        "pairs_carrying_both_REFUTES_and_NOT_ENOUGH_INFO": len(both),
        "examples": ex,
        "mechanism": "genuine annotation disagreement inside VitaminC's own "
                     "held-out splits - the same claim over the same evidence is "
                     "annotated REFUTES in one case and NOT ENOUGH INFO in "
                     "another. A corpus property, not a builder defect",
    }

    ag = pl.read_parquet(SEM / "R18-H150_antigaming_set.parquet")
    lane146 = set(pl.read_parquet(SEM / "R17-H146_lane.parquet")["claim"].to_list())
    lane133 = set(pl.read_parquet(SEM / "R14-H133_lane.parquet")["claim"].to_list())
    cl = set(ag["claim_pos"].to_list() + ag["claim_neg"].to_list())
    h146 = sorted(c for c in cl if c in lane146)
    kinds = ag.filter(pl.col("claim_pos").is_in(h146)
                      | pl.col("claim_neg").is_in(h146))["kind"].to_list()
    res["checks"]["4b_antigaming_claims_in_a_training_lane"] = {
        "distinct_eval_claims": len(cl),
        "verbatim_in_the_R17_H146_misbind_lane_which_is_in_the_flagship_mix": len(h146),
        "verbatim_in_the_R14_H133_lane_the_only_set_the_builder_guards_against":
            len([c for c in cl if c in lane133]),
        "by_kind": dict(collections.Counter(kinds)),
        "examples": h146[:3],
        "mechanism": "`R14-H133_antigaming.build_bindrow` rejects a candidate "
                     "whose claim is in `R14-H133_lane.parquet` - the lane of the "
                     "arm the instrument was written for. Every arm since trains "
                     "on `R17-H146_lane.parquet` instead, which the guard never "
                     "reads, and which emits the identical "
                     "`The <column> of <row key> is <value>.` template",
    }

    OUT.write_text(json.dumps(res, indent=2))
    for k, v in res["checks"].items():
        print(f"{k}: {json.dumps({a: b for a, b in v.items() if a not in ('mechanism', 'examples', 'example_head')})}",
              flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

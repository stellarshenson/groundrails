"""Supplementary measurements for the `quant_scale_unit` contract report.

CPU ONLY.

  (a) C8 - do any of the 84 repeated claim strings carry BOTH labels (a label
      contradiction inside the member)?
  (b) C3 - do any of the lane's TabFact tables have text byte-identical to a
      TabFact validation/test table under a different table_id (the corpus's own
      content-level split leak, measured from the archive)?
  (c) C1 - the containment gap on the 90th-percentile-and-above stratum, and the
      per-swap-family breakdown of the >= 0.90 attestation rates.

Run:  CUDA_VISIBLE_DEVICES= uv run python \
      experiments/grounding-semantic/contract/quant_scale_unit_supp.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import collections
import io
import json
import pathlib
import re
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
ROOT = SEM.parent.parent
LANE = SEM / "R18-H150_scaleunit_lane.parquet"
OUT = HERE / "quant_scale_unit_supp.json"

_W = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def containment(a, b):
    A = set(w.lower() for w in _W.findall(a))
    return len(A & {w.lower() for w in _W.findall(b)}) / len(A) if A else 0.0


def main():
    lane = pl.read_parquet(LANE)
    res = {"member": "quant_scale_unit"}

    # (a) repeated claim strings
    cc = collections.Counter(lane["claim"].to_list())
    rep = {c for c, n in cc.items() if n > 1}
    sub = lane.filter(pl.col("claim").is_in(list(rep)))
    g = sub.group_by("claim").agg(
        pl.col("label").n_unique().alias("labels"),
        pl.col("chunk").n_unique().alias("chunks"),
        pl.col("doc_id").n_unique().alias("docs"),
        pl.len().alias("n"))
    res["repeated_claim_strings"] = {
        "distinct_repeated_claims": len(rep),
        "rows_involved": len(sub),
        "claims_carrying_both_labels": int((g["labels"] == 2).sum()),
        "claims_on_more_than_one_document": int((g["docs"] > 1).sum()),
        "max_repeat": int(g["n"].max()),
        "interpretation": "a claim string carrying both labels would be a "
                          "label contradiction inside the member",
    }
    if int((g["labels"] == 2).sum()):
        bad = g.filter(pl.col("labels") == 2).head(5)
        res["repeated_claim_strings"]["examples"] = bad.to_dicts()

    # (b) TabFact content-level split leak, and the lane's exposure to it
    z = zipfile.ZipFile(ROOT / "data" / "external" / "datasets" / "dataset-tabfact.zip")
    parts = {}
    for split in ("train", "validation", "test"):
        nm = next(n for n in z.namelist() if n.endswith(f"__{split}.parquet"))
        parts[split] = pl.read_parquet(io.BytesIO(z.read(nm)))
    tr = parts["train"].unique(subset=["table_text"], keep="first")
    txt_of_id = dict(zip(tr["table_id"].to_list(), tr["table_text"].to_list()))
    lane_ids = {d.split(":", 1)[1] for d in lane["doc_id"].unique().to_list()
                if d.startswith("tabfact:")}
    lane_txt = {txt_of_id[i] for i in lane_ids if i in txt_of_id}
    blk = {"lane_tabfact_documents": len(lane_ids),
           "lane_table_texts_resolved": len(lane_txt)}
    for split in ("validation", "test"):
        st = set(parts[split]["table_text"].to_list())
        blk[f"lane_tables_byte_identical_to_a_{split}_table"] = len(lane_txt & st)
    res["tabfact_content_split_leak"] = blk

    # (c) C1 per-swap-family attestation rates
    claims = lane["claim"].to_list()
    chunks = lane["chunk"].to_list()
    y = np.array(lane["label"].to_list())
    fam = lane["swap_family"].to_list()
    cont = np.array([containment(c, k) for c, k in zip(claims, chunks)])
    per = {}
    for f in sorted(set(fam)):
        m = np.array([x == f for x in fam])
        p, n = cont[m & (y == 1)], cont[m & (y == 0)]
        per[f] = {
            "pairs": int(m.sum() // 2),
            "pos_mean": round(float(p.mean()), 6),
            "neg_mean": round(float(n.mean()), 6),
            "pos_rate_ge_0_90": round(float((p >= 0.90).mean()), 6),
            "neg_rate_ge_0_90": round(float((n >= 0.90).mean()), 6),
            "abs_rate_gap": round(abs(float((n >= 0.90).mean() - (p >= 0.90).mean())), 6),
            "pos_rate_eq_1": round(float((p >= 1.0 - 1e-12).mean()), 6),
            "neg_rate_eq_1": round(float((n >= 1.0 - 1e-12).mean()), 6),
        }
    res["c1_per_swap_family"] = per
    res["c1_worst_family_abs_rate_gap"] = max(v["abs_rate_gap"] for v in per.values())
    res["c1_families_with_gap_above_0_10"] = [
        k for k, v in per.items() if v["abs_rate_gap"] > 0.10]

    OUT.write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()

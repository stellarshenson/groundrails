"""attr_pool conformed build - diagnostic for the C5 claim-only bar.

The first conformed build (MiniCheck only) reads within-pair claim-only 0.6310 on
its `unsupported_claim` family against a bar of < 0.60, while the parent lane
reads 0.5594 on the same bar with both sources mixed.  Two candidate causes:

  (a) a corpus property - MiniCheck's label-0 claims are surface-separable from
      its label-1 claims, and the parent's number was carried by the VitaminC
      half, whose negatives are minimal Wikipedia revisions
  (b) a small-sample effect of the conformed member's 889 unsupported_claim pairs

Measured here, on the PARENT lane, split by source, with the banked converged
probe.  Also split by MiniCheck synthesis route (c2d / d2c) in case the leak is
route-specific and a route restriction would conform.

CPU only.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import importlib.util as _ilu
import json
import pathlib
import random

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
EXP = HERE.parent


def _mod(name, fname, folder=EXP):
    spec = _ilu.spec_from_file_location(name, folder / fname)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = _mod("h174common", "R20-H174_lane_common.py")


def within_pair(df, score, keycols):
    d = df.select(["pair_id", "label"] + keycols).with_columns(pl.Series("s", score))
    out = {}
    for key, sub in d.group_by(keycols):
        piv = sub.pivot(on="label", index="pair_id", values="s",
                        aggregate_function="first").drop_nulls()
        if not len(piv):
            continue
        p, n = piv["1"].to_numpy(), piv["0"].to_numpy()
        out["|".join(str(k) for k in key)] = {
            "acc": round(float(((p > n) + 0.5 * (p == n)).mean()), 4),
            "pairs": int(len(piv)),
        }
    return out


def main():
    out = {}
    rng = random.Random(2174)

    # ---- parent lane, per source ----------------------------------------- #
    par = pl.read_parquet(EXP / "R20-H174_lane_L2.parquet")
    probe, score = C.claim_only_probe(par["claim"].to_list(), par["label"].to_list(),
                                      par["doc_id"].to_list(), rng)
    out["parent"] = {
        "rows": int(par.height),
        "claim_only_auroc_all": round(probe, 4),
        "within_pair_by_family": within_pair(par, score, ["neg_family"]),
        "within_pair_by_family_and_source": within_pair(par, score, ["neg_family", "source"]),
        "reading": "the parent's registered within-pair claim-only figure is the "
        "per-FAMILY number; splitting it by source shows what the aggregate hides",
    }
    print(json.dumps(out["parent"], indent=2), flush=True)

    # ---- conformed member, per MiniCheck synthesis route ------------------ #
    con = pl.read_parquet(EXP / "R20-H174_lane_L2_conformed.parquet")
    mc = C.minicheck()
    route = {}
    for c, d, s in mc.select(["claim", "doc_id", "split"]).iter_rows():
        route.setdefault((c, d), set()).add(s)
    rr = []
    for c, d in con.select(["claim", "doc_id"]).iter_rows():
        s = route.get((c, d), set())
        rr.append("|".join(sorted(s)) if s else "unmatched")
    con = con.with_columns(pl.Series("route", rr))
    rng2 = random.Random(2174)
    probe2, score2 = C.claim_only_probe(con["claim"].to_list(), con["label"].to_list(),
                                        con["doc_id"].to_list(), rng2)
    out["conformed_v1"] = {
        "rows": int(con.height),
        "claim_only_auroc_all": round(probe2, 4),
        "within_pair_by_family": within_pair(con, score2, ["neg_family"]),
        "within_pair_by_family_and_route": within_pair(con, score2, ["neg_family", "route"]),
        "route_rows": {k: v for k, v in con.group_by("route").len().iter_rows()},
    }
    print(json.dumps(out["conformed_v1"], indent=2), flush=True)

    # ---- is the unsupported_claim leak fixable by route restriction? ------ #
    # measure the probe RETRAINED on a single route, so the restriction is
    # evaluated as a pipeline rather than as a slice of a mixed-route probe
    for r in sorted({x for x in rr if x != "unmatched"}):
        sub = con.filter((pl.col("neg_family") == "unsupported_claim") & (pl.col("route") == r))
        if sub.height < 200:
            out.setdefault("route_restricted_rebuild_probe", {})[r] = {
                "rows": int(sub.height), "note": "too few rows to retrain the probe"}
            continue
        rng3 = random.Random(2174)
        p3, s3 = C.claim_only_probe(sub["claim"].to_list(), sub["label"].to_list(),
                                    sub["doc_id"].to_list(), rng3)
        out.setdefault("route_restricted_rebuild_probe", {})[r] = {
            "rows": int(sub.height),
            "claim_only_auroc": round(p3, 4),
            "within_pair": within_pair(sub, s3, ["neg_family"]),
        }
    print(json.dumps(out.get("route_restricted_rebuild_probe", {}), indent=2), flush=True)

    # ---- and the same probe retrained on the MiniCheck unsupported_claim
    #      rows of the PARENT, to separate corpus property from sample size --
    sub = par.filter((pl.col("neg_family") == "unsupported_claim")
                     & (pl.col("source") == "minicheck"))
    rng4 = random.Random(2174)
    p4, s4 = C.claim_only_probe(sub["claim"].to_list(), sub["label"].to_list(),
                                sub["doc_id"].to_list(), rng4)
    out["parent_minicheck_unsupported_claim_alone"] = {
        "rows": int(sub.height),
        "claim_only_auroc": round(p4, 4),
        "within_pair": within_pair(sub, s4, ["neg_family"]),
    }
    subv = par.filter((pl.col("neg_family") == "unsupported_claim")
                      & (pl.col("source") == "vitaminc"))
    rng5 = random.Random(2174)
    p5, s5 = C.claim_only_probe(subv["claim"].to_list(), subv["label"].to_list(),
                                subv["doc_id"].to_list(), rng5)
    out["parent_vitaminc_unsupported_claim_alone"] = {
        "rows": int(subv.height),
        "claim_only_auroc": round(p5, 4),
        "within_pair": within_pair(subv, s5, ["neg_family"]),
    }
    print(json.dumps({k: out[k] for k in
                      ("parent_minicheck_unsupported_claim_alone",
                       "parent_vitaminc_unsupported_claim_alone")}, indent=2), flush=True)

    (HERE / "attr_pool_conformed_diag.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"-> {HERE / 'attr_pool_conformed_diag.json'}", flush=True)


if __name__ == "__main__":
    main()

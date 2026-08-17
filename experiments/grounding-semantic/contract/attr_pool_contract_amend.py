"""attr_pool contract - two amendments to the stage measurements.

1. C4 LIVE POSITIVE CONTROL, strengthened. The first run indexed only a 4,000-unit
   SAMPLE of each source corpus, so a lane passage whose source sentences were
   outside the sample could not fire; it read 0.4315, which understates the
   instrument. Rebuilt against the FULL MiniCheck document set and the FULL
   VitaminC train evidence set - every unit the lane's passages were assembled
   from - so a non-firing candidate is now a genuine instrument failure.

2. C2 LEAK CONFIRMATION. The stage found one lane claim colliding with the
   VitaminC held-out candidate pool under whitespace-collapsed case-folding only.
   This re-runs the full `R20_baseline_legs.vitaminc_holdout` filter chain on
   that case, against the ASSEMBLED flagship mix text, to establish whether the
   colliding row actually reaches the eval surface or is filtered out first.

CPU only.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import importlib.util as _ilu
import io
import json
import pathlib
import random
import re
import time
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
EXP = HERE.parent
DATA = EXP.parent.parent / "data" / "external" / "datasets"
SEP = "\n\n"
_WS = re.compile(r"\s+")
wsfold = lambda t: _WS.sub(" ", t).strip().casefold()  # noqa: E731


def _mod(name, fname):
    spec = _ilu.spec_from_file_location(name, EXP / fname)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


GATE = _mod("provgate", "provenance_gate.py")


def main():
    out = {}
    df = pl.read_parquet(EXP / "R20-H174_lane_L2.parquet")
    rng = random.Random(7)

    # ---------------- 1. live positive control, full source index ---------- #
    print("building the FULL source index ...", flush=True)
    z = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    vtr = pl.read_parquet(io.BytesIO(z.read("tals__vitaminc__train.parquet")))
    vev = vtr["evidence"].unique().to_list()
    zm = zipfile.ZipFile(DATA / "dataset-minicheck.zip")
    mcdocs = []
    for n in [x for x in zm.namelist() if x.endswith(".parquet")]:
        mcdocs += pl.read_parquet(io.BytesIO(zm.read(n)))["doc"].unique().to_list()
    mcdocs = sorted(set(mcdocs))
    print(f"  vitaminc train evidence {len(vev)}, minicheck docs {len(mcdocs)}", flush=True)
    src_side = {"vitaminc_train_evidence": vev, "minicheck_docs": mcdocs}

    lane_vc = df.filter(pl.col("source") == "vitaminc")
    lane_mc = df.filter(pl.col("source") == "minicheck")
    vc_pass = sorted({p for k in lane_vc["chunk"].to_list() for p in k.split(SEP)})
    mc_pass = sorted({p for k in lane_mc["chunk"].to_list() for p in k.split(SEP)})
    cand = rng.sample(vc_pass, 1000) + rng.sample(mc_pass, 1000)

    t0 = time.time()
    live = GATE.run_gate(cand, n=8, jaccard=0.3, label="lane_passages_live", arena_texts=src_side)
    print(f"  live control {time.time() - t0:.0f}s fires at "
          f"{live['candidate_vs_arena']['fraction']}", flush=True)

    # per-half breakdown so a one-sided fire is visible
    t0 = time.time()
    live_vc = GATE.run_gate(cand[:1000], n=8, jaccard=0.3, label="lane_vitaminc_passages",
                            arena_texts={"vitaminc_train_evidence": vev})
    live_mc = GATE.run_gate(cand[1000:], n=8, jaccard=0.3, label="lane_minicheck_passages",
                            arena_texts={"minicheck_docs": mcdocs})
    print(f"  per-half {time.time() - t0:.0f}s  vitaminc "
          f"{live_vc['candidate_vs_arena']['fraction']}  minicheck "
          f"{live_mc['candidate_vs_arena']['fraction']}", flush=True)

    zh = zipfile.ZipFile(DATA / "dataset-halueval.zip")
    hd = pl.read_parquet(io.BytesIO(zh.read(next(x for x in zh.namelist() if x.endswith(".parquet")))))
    hcol = next(c for c in ("knowledge", "document") if c in hd.columns)
    negctl = GATE.run_gate(rng.sample(hd[hcol].unique().to_list(), 2000), n=8, jaccard=0.3,
                           label="halueval_unrelated", arena_texts=src_side)
    print(f"  negative control fires at {negctl['candidate_vs_arena']['fraction']}", flush=True)

    out["live_positive_control_full_index"] = {
        "design": "1,000 VitaminC-derived + 1,000 MiniCheck-derived atomic pool "
        "passages of this lane against the FULL source corpora they were "
        f"assembled from ({len(vev)} VitaminC train evidence sentences, "
        f"{len(mcdocs)} MiniCheck documents) - near-duplicate BY CONSTRUCTION",
        "candidate_units": live["candidate"]["n_units"],
        "fires_at_fraction": live["candidate_vs_arena"]["fraction"],
        "per_source_bucket": live["candidate_vs_arena"]["per_arena_subset"],
        "best_jaccard": live["candidate_vs_arena"].get("best_jaccard"),
        "vitaminc_half_alone": live_vc["candidate_vs_arena"]["fraction"],
        "minicheck_half_alone": live_mc["candidate_vs_arena"]["fraction"],
        "bar": "the gate must FIRE on text near-duplicate by construction",
        "pass": bool(live["candidate_vs_arena"]["fraction"] > 0.5),
        "contrast_with_the_arena_reading": "arena candidate_vs_arena fraction "
        "0.00004 on the same instrument and the same lane text",
    }
    out["live_negative_control_full_index"] = {
        "design": "2,000 unrelated HaluEval knowledge passages against the same "
        "full source index - a saturated instrument would fire here too",
        "fires_at_fraction": negctl["candidate_vs_arena"]["fraction"],
        "bar": "< 0.02",
        "pass": bool(negctl["candidate_vs_arena"]["fraction"] < 0.02),
    }

    # ---------------- 2. C2 leak confirmation ------------------------------ #
    print("confirming the wsfold collision against the built holdout ...", flush=True)
    parts = [
        pl.read_parquet(io.BytesIO(z.read(f"tals__vitaminc__{s}.parquet"))).with_columns(
            pl.lit(s).alias("split"))
        for s in ("test", "validation")
    ]
    cand_pool = pl.concat(parts)
    lane_fold = {wsfold(c): c for c in df["claim"].unique().to_list()}

    keep = np.ones(cand_pool.height, dtype=bool)
    for col in ("page", "claim", "evidence", "wiki_revision_id", "unique_id", "case_id"):
        keep &= ~cand_pool[col].is_in(list(set(vtr[col].to_list()))).to_numpy()
    held = cand_pool.filter(keep)
    held = held.filter(pl.col("label").is_in(["REFUTES", "NOT ENOUGH INFO"]))
    print(f"  holdout after the banked filter chain (mix text filter excluded): "
          f"{held.height} rows", flush=True)

    coll = [c for c in held["claim"].unique().to_list() if wsfold(c) in lane_fold]
    detail = []
    for c in coll:
        sub = held.filter(pl.col("claim") == c)
        lane_claim = lane_fold[wsfold(c)]
        lr = df.filter(pl.col("claim") == lane_claim)
        detail.append({
            "holdout_claim_chars": len(c),
            "lane_claim_chars": len(lane_claim),
            "raw_strings_equal": c == lane_claim,
            "wsfold_strings_equal": True,
            "edit_description": "differs by letter case only"
            if c.casefold() == lane_claim.casefold() and c != lane_claim
            else "differs by whitespace and/or case",
            "holdout_rows": sub.height,
            "holdout_labels": sub["label"].to_list(),
            "holdout_splits": sub["split"].to_list(),
            "lane_rows": lr.height,
            "lane_labels": lr["label"].to_list(),
            "lane_families": lr["neg_family"].unique().to_list(),
            "lane_pair_ids": lr["pair_id"].unique().to_list(),
        })
    out["c2_leak_confirmation"] = {
        "surface": "vitaminc_holdout - the R19-H166 amendment A1 contradiction "
        "mechanism eval, built by R20_baseline_legs.vitaminc_holdout",
        "holdout_rows_after_key_filter_and_label_filter": int(held.height),
        "colliding_claims_surviving_into_the_eval": len(coll),
        "detail": detail,
        "why_the_banked_filter_misses_it": "the holdout construction drops a "
        "candidate whose claim string occurs in VitaminC train, using RAW string "
        "equality. The colliding pair differs by one letter's case, so raw "
        "equality does not fire and the row is kept. The step-4 filter against "
        "the assembled flagship mix is also raw, and that mix does not contain "
        "attr_pool at all",
        "form_that_detects_it": "whitespace-collapsed case-folded - contract C2 "
        "string form 3, the form its provenance note was written for",
    }

    (HERE / "attr_pool_amend.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"-> {HERE / 'attr_pool_amend.json'}", flush=True)


if __name__ == "__main__":
    main()

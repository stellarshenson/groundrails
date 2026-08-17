"""C3 / C6 supplement for `frame_reject` - two corrections and one extra read.

1. C3 CORRECTION.  The first pass counted a lane chunk as carrying a held-out
   VitaminC evidence sentence if ANY shared evidence string was a substring of
   it.  Of the 85 evidence strings shared between VitaminC train and its
   validation/test splits, the shortest is one character (`R`), which is a
   substring of every chunk - the 3,519/3,519 reading was that artefact, not a
   measurement.  The check is redone at explicit length floors and at the
   8-token floor the campaign's own contamination instrument uses.

2. C6 / C8 EXTRA READ.  The first pass found 2,909 of the lane's 3,287 distinct
   genuine claims already present in the assembled mix verbatim.  That is
   within-mix duplication, not an evaluation leak, and it is broken out by lane
   source here, together with the question it raises: does the mix carry any of
   those claims with a CONTRADICTORY label over the same evidence?

CPU only.  Writes one JSON; the main report is patched from it separately.

Run:  uv run python experiments/grounding-semantic/contract/frame_reject_c3_c6_supplement.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections
import importlib.util as _ilu
import json
from pathlib import Path
import time

import numpy as np
import polars as pl

HERE = Path(__file__).parent
EXP = HERE.parent
ROOT = EXP.parent.parent
LANE = EXP / "R20-H174_lane_L1.parquet"
OUT = HERE / "frame_reject_c3_c6_supplement.json"
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _mod(name, path):
    s = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(s)
    s.loader.exec_module(m)
    return m


C = _mod("h174common", EXP / "R20-H174_lane_common.py")


def norm(s):
    return " ".join(s.split()).casefold()


def c3(df):
    log("C3 correction")
    tr, va, te = C.vitaminc("train"), C.vitaminc("validation"), C.vitaminc("test")
    held_ev = set(va["evidence"].to_list()) | set(te["evidence"].to_list())
    held_claims = set(va["claim"].to_list()) | set(te["claim"].to_list())
    shared = sorted(set(tr["evidence"].to_list()) & held_ev)

    lane = df.filter(pl.col("source") == "vitaminc")
    lane_chunks = sorted({c for c in lane["chunk"].to_list()})
    lane_chunks_n = [norm(c) for c in lane_chunks]

    by_floor = {}
    for floor in (0, 20, 40, 80):
        pool = {norm(e) for e in shared if len(e) >= floor}
        hits = sum(1 for ch in lane_chunks_n if any(e in ch for e in pool)) if pool else 0
        by_floor[f"min_{floor}_chars"] = {"shared_strings_at_this_floor": len(pool),
                                          "lane_vitaminc_chunks_containing_one": hits}
    # the campaign's own 8-token instrument floor
    pool8 = {norm(e) for e in shared if len(norm(e).split()) >= 8}
    hits8 = sum(1 for ch in lane_chunks_n if any(e in ch for e in pool8)) if pool8 else 0

    lane_gen = {c for c in lane["genuine_claim"].to_list()}
    overlap_claims = sorted(lane_gen & held_claims)
    rows_on_those = int(lane.filter(pl.col("genuine_claim").is_in(overlap_claims)).height) \
        if overlap_claims else 0

    return {
        "correction": "the first pass used a 1-character shared evidence string "
                      "(`R`) as a substring probe, which matches every chunk; the "
                      "3,519/3,519 figure was that artefact",
        "vitaminc_split_rows": {"train": tr.height, "validation": va.height,
                                "test": te.height},
        "evidence_strings_shared_train_vs_heldout": len(shared),
        "shortest_shared_string_chars": min(len(s) for s in shared) if shared else None,
        "lane_vitaminc_chunks": len(lane_chunks),
        "by_length_floor": by_floor,
        "at_the_8_token_instrument_floor": {
            "shared_strings": len(pool8),
            "lane_vitaminc_chunks_containing_one": hits8},
        "lane_genuine_claims_also_in_heldout_claim_set": {
            "count": len(overlap_claims),
            "lane_rows_carrying_one": rows_on_those,
            "share_of_lane_rows": round(rows_on_those / df.height, 6)},
    }


def c6(df):
    log("C6 / C8 extra read - assembling the mix")
    arm = _mod("g1arm", EXP / "R16-H142_G1_arm.py")
    H108 = _mod("h108lane", EXP / "R10-H108_lane.py")
    with arm.untruncated_evidence():
        claims, chunks, y, tags = H108.public_train()
    log(f"  clean mix {len(claims)} rows over {len(set(tags))} groups")
    for fname, group in (("R17-H146_lane.parquet", "quant_misbind"),
                         ("R18-H150_scaleunit_lane.parquet", "quant_scale_unit"),
                         ("R20-H174_lane_L2.parquet", "attr_pool"),
                         ("R20-H174_lane_L4.parquet", "path_bind")):
        d = pl.read_parquet(EXP / fname)
        claims += d["claim"].to_list()
        chunks += d["chunk"].to_list()
        y = np.concatenate([y, d["label"].cast(pl.Float32).to_numpy()])
        tags += [group] * d.height
    log(f"  mix minus the member: {len(claims)} rows")

    # claim -> (group, label) bag, and (claim, chunk) -> label bag
    claim_groups = collections.defaultdict(set)
    claim_labels = collections.defaultdict(set)
    for c, lab, g in zip(claims, y, tags):
        claim_groups[c].add(g)
        claim_labels[c].add(float(lab))
    pair_labels = {}
    for c, k, lab in zip(claims, chunks, y):
        pair_labels.setdefault((c, k), set()).add(float(lab))
    log(f"  {len(claim_groups)} distinct mix claims indexed")

    out = {"mix_rows": len(claims), "mix_groups": sorted(set(tags)),
           "distinct_mix_claims": len(claim_groups)}

    for leg, frame in (("genuine_claim", df), ("positive_claim_as_trained",
                                               df.filter(pl.col("label") == 1)),
                       ("negative_claim_as_trained", df.filter(pl.col("label") == 0))):
        col = "genuine_claim" if leg == "genuine_claim" else "claim"
        vals = frame[col].to_list()
        distinct = set(vals)
        present = {v for v in distinct if v in claim_groups}
        rows_present = sum(1 for v in vals if v in claim_groups)
        grp = collections.Counter(g for v in present for g in claim_groups[v])
        out[leg] = {"distinct": len(distinct),
                    "distinct_present_in_the_mix": len(present),
                    "share_of_distinct": round(len(present) / max(len(distinct), 1), 4),
                    "member_rows_whose_claim_the_mix_also_carries": rows_present,
                    "mix_groups_supplying_them": dict(grp.most_common())}

    # per lane source
    per_src = {}
    for src in sorted(set(df["source"].to_list())):
        sub = df.filter((pl.col("source") == src) & (pl.col("label") == 1))
        gen = set(sub["genuine_claim"].to_list())
        per_src[src] = {"distinct_genuine_claims": len(gen),
                        "present_in_the_mix": len(gen & set(claim_groups)),
                        "share": round(len(gen & set(claim_groups)) / max(len(gen), 1), 4)}
    out["by_lane_source"] = per_src

    # label agreement: does the mix carry any lane genuine claim as a NEGATIVE?
    gen = set(df["genuine_claim"].to_list())
    shared_claims = gen & set(claim_labels)
    lab_bags = collections.Counter()
    for c in shared_claims:
        b = claim_labels[c]
        key = ("only_1" if b == {1.0} else "only_0" if b == {0.0} else "both")
        lab_bags[key] += 1
    out["mix_label_on_shared_genuine_claims"] = dict(lab_bags)

    # exact (claim, chunk) collisions between the member and the mix
    coll = {}
    for lab_v in (1, 0):
        sub = df.filter(pl.col("label") == lab_v)
        pairs = set(zip(sub["claim"].to_list(), sub["chunk"].to_list()))
        hit = {p for p in pairs if p in pair_labels}
        contra = {p for p in hit if pair_labels[p] != {float(lab_v)}}
        coll[f"label_{lab_v}"] = {"member_pairs": len(pairs),
                                  "also_in_the_mix": len(hit),
                                  "with_a_contradictory_mix_label": len(contra)}
    out["claim_chunk_pair_collisions"] = coll
    return out


def main():
    df = pl.read_parquet(LANE)
    res = {"supplement": "C3 correction and C6/C8 mix-duplication read",
           "member": "frame_reject", "rows": df.height,
           "C3": c3(df), "C6_C8": c6(df)}
    OUT.write_text(json.dumps(res, indent=2))
    log(f"wrote {OUT}")
    print(json.dumps(res, indent=2)[:6000], flush=True)


if __name__ == "__main__":
    main()

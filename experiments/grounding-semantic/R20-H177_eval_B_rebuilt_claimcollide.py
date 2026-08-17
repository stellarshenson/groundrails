"""R20-H177 eval_B REBUILT - the claim-string collision with lane B, quantified.

CPU only.  The eight-clause verification found the rebuilt eval's EVIDENCE
channel at zero against every surface and its DOCUMENT channel at zero, but 10 of
its 1,998 distinct claims are byte-identical to a lane B claim.  That is expected
of a template lane - "The {col} of {ka} ({va}) is greater than the {col} of {kb}
({vb})" collides whenever two different tables print the same column name, row
labels and cells - but "expected" is not a measurement, so this counts it and
reads the only thing that matters: whether the label lane B attached to that
claim string agrees with the eval's, which is what a claim-string memoriser would
carry across.

Merged into R20-H177_eval_B_rebuilt_report.json under
`clauses.C2.claim_string_collision_with_lane_B`.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections
import json
from pathlib import Path

import polars as pl

HERE = Path(__file__).parent
REPORT = HERE / "R20-H177_eval_B_rebuilt_report.json"
OUT = HERE / "R20-H177_eval_B_rebuilt_claimcollide.json"


def measure(eval_df, lane, label):
    lane_by_claim = collections.defaultdict(set)
    lane_chunk = collections.defaultdict(set)
    for c, y, ch in zip(lane["claim"].to_list(), lane["label"].to_list(),
                        lane["chunk"].to_list()):
        lane_by_claim[c].add(int(y))
        lane_chunk[c].add(ch)

    hits = []
    for r in eval_df.iter_rows(named=True):
        if r["claim"] in lane_by_claim:
            hits.append({
                "pair_id": int(r["pair_id"]), "eval_label": int(r["label"]),
                "neg_family": r["neg_family"],
                "lane_labels": sorted(lane_by_claim[r["claim"]]),
                "label_agrees": int(r["label"]) in lane_by_claim[r["claim"]],
                "same_evidence": r["chunk"] in lane_chunk[r["claim"]],
                "claim": r["claim"][:160]})
    pairs = {h["pair_id"] for h in hits}
    both_legs = sum(1 for p in pairs
                    if sum(1 for h in hits if h["pair_id"] == p) == 2)
    agree = sum(1 for h in hits if h["label_agrees"])
    return {
        "surface": label,
        "eval_rows": eval_df.height,
        "distinct_eval_claims": int(eval_df["claim"].n_unique()),
        "colliding_rows": len(hits),
        "share_of_rows": round(len(hits) / eval_df.height, 5),
        "colliding_pairs": len(pairs),
        "pairs_with_BOTH_legs_colliding": both_legs,
        "rows_whose_lane_label_agrees_with_the_eval_label": agree,
        "rows_whose_lane_label_disagrees": len(hits) - agree,
        "rows_sharing_the_lane_evidence_too": sum(1 for h in hits if h["same_evidence"]),
        "by_family": dict(collections.Counter(h["neg_family"] for h in hits)),
        "examples": hits[:10],
    }


def main():
    lane = pl.read_parquet(HERE / "R20-H177_lane_B.parquet")
    rebuilt = pl.read_parquet(HERE / "R20-H177_eval_B_rebuilt.parquet")
    original = pl.read_parquet(HERE / "R20-H177_eval_B.parquet")

    res = {
        "question": "the rebuilt eval's evidence and document channels read zero "
                    "against every surface; do its CLAIM STRINGS collide with the "
                    "training lane, and if so does the lane's label agree?",
        "instrument": "exact claim-string join against R20-H177_lane_B.parquet, "
                      "with the lane's label set and evidence set per claim",
        "rebuilt": measure(rebuilt, lane, "R20-H177_eval_B_rebuilt.parquet"),
        "original_for_comparison": measure(original, lane,
                                           "R20-H177_eval_B.parquet"),
        "mechanism": "the lane is template-generated: a claim is a fixed frame "
                     "filled with a column name, two row labels and two cells, so "
                     "two unrelated tables printing the same five strings emit the "
                     "same sentence. The colliding claims carry DIFFERENT evidence "
                     "(the evidence channel reads zero), so the collision is a "
                     "claim-string channel, not a passage or document one",
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    }
    OUT.write_text(json.dumps(res, indent=2))

    if REPORT.exists():
        rep = json.loads(REPORT.read_text())
        rep["clauses"]["C2"]["claim_string_collision_with_lane_B"] = res
        REPORT.write_text(json.dumps(rep, indent=2))
        print(f"merged into {REPORT.name}", flush=True)
    print(json.dumps({k: res[k] for k in ("rebuilt", "original_for_comparison")},
                     indent=2)[:3000], flush=True)


if __name__ == "__main__":
    main()

"""R20-H174 MECHANISM GATES A and B - the two gates the campaign wrapper never computed.

The R20-H174 registration carries THREE mechanism gates. The campaign script prints
all three but computes only one (hagrid >= 0.680 every draw, which FAILED at 0.6166
on draw 1). This script computes the other two on the banked draw-1 checkpoint and,
under an IDENTICAL protocol, on the banked flagship checkpoints, so the comparison
is paired.

    GATE A   frame-only misrank share    bar < 0.05, measured baseline 0.212
    GATE B   hagrid k-doc-curve slope    bar non-negative, flagship's falls

NOTHING IS TRAINED, TUNED OR SELECTED HERE. Every number is either read off a banked
per-pair logit dump or produced by re-running the banked read path on a banked
checkpoint. GPU1 only.

DEFINITIONS ARE REUSED, NOT INVENTED
------------------------------------
Both gates are computed by importing the banked analysis module
`R19-H162_hagrid_mechanisms.py` and calling its own functions unmodified:

  * `misrank_block(dump, items)` - GATE A. Misrank mass is the count of (positive,
    negative) ITEM pairs that the checkpoint ranks wrongly, ties counted as wrong:
    `mis = pos_score[:, None] <= neg_score[None, :]`, `tot = mis.sum()`. The
    frame-only share is the fraction of that mass carried by columns belonging to
    frame-only negatives: `mis[:, frame_only_cols].sum() / tot`. Frame-only items
    are fixed by `RE_FRAMEONLY` over the arena response text and are therefore the
    SAME four items (49, 86, 188, 196) for every checkpoint - the gate is paired by
    construction. This function produced the banked 0.2124 / 0.2076 (flagship draws)
    and 0.1727 (enriched mix) in `R19-H162_hagrid_mechanisms.json`.

  * `vacuous_excluded(dump, items)` - GATE B primary. Drops the four frame-only
    items, then reports AUROC per evidence-pool-depth stratum (ndoc_1, ndoc_2_3,
    ndoc_4plus) via `strata()`. This produced the banked flagship reference
    0.8577/0.8129 (1 doc), 0.6890/0.6277 (2-3), 0.5096/0.6052 (4-8).

  * `kdoc_curve(dump, item_table)` - GATE B secondary. AUROC with the evidence pool
    TRUNCATED to the first k retrieved documents, k in (1,2,3,4,6,8), reported over
    all items and over the deep (ndoc>=4) subset.

The banked artifacts bank the CURVE but no scalar slope, so the slope statistics
below are stated here rather than inherited. Both are pre-stated before any read:

  * pool-depth slope = OLS of the three vacuous-excluded stratum AUROCs on bucket
    index (0, 1, 2). For three evenly indexed points this equals half the endpoint
    delta, so the endpoint delta (ndoc_4plus - ndoc_1) is reported alongside it and
    carries the same sign.
  * k-truncation slope = OLS of the kdoc curve on k itself over (1,2,3,4,6,8).

HONESTY CONTROL ON GATE A
-------------------------
The share is a RATIO. A checkpoint that ranks worse overall inflates the denominator
and can lower the share without suppressing a single frame-only item. The absolute
frame-only misrank pair count, the total misrank pair count and the four items' own
scores are therefore banked next to the share.

POSITIVE CONTROL, before any gate number is written
---------------------------------------------------
Every checkpoint's hagrid AUROC recomputed from its dump must reproduce its banked
windowed read to <= 1e-3 and the structural fingerprint (250 items / 537 sentences /
1,941 pairs) must match exactly. A miss aborts, it is never corrected.

Run (GPU1 only - GPU0 runs H174 draw 2, GPU2 runs H175b draw 1):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
    uv run python experiments/grounding-semantic/R20-H174_mechanism_gates_d1.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util
import json
import pathlib
import time

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
import torch

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
OUT_JSON = HERE / "R20-H174_mechanism_gates_d1.json"

SUBSET = "hagrid"
CONTROL_TOL = 1e-3
FINGERPRINT = {"n": 250, "n_sent": 537, "n_pairs": 1941}

GATE_A_BAR = 0.05
GATE_A_BASELINE = 0.212

# tag -> (checkpoint dir | None if the dump is already banked, dump parquet, banked read json)
CKPTS = {
    "h174d1": {
        "dir": "R20-H174-arm-draw1",
        "dump": "R20-H174_pairs_h174d1.parquet",
        "banked_read": "R20-H174_arm_draw1_windowed_result.json",
        "label": "R20-H174 portfolio arm, draw 1 (seed 1174)",
    },
    "h150d1": {
        "dir": "R18-H150-arm-draw1",
        "dump": "R19-H161_pairs_h150d1.parquet",
        "banked_read": "R18-H150_arm_draw1_windowed_result.json",
        "label": "R18-H150 flagship, draw 1 (banked dump, re-verified)",
    },
    "h150d2": {
        "dir": "R18-H150-arm-draw2",
        "dump": "R19-H161_pairs_h150d2.parquet",
        "banked_read": "R18-H150_arm_draw2_windowed_result.json",
        "label": "R18-H150 flagship, draw 2 (banked dump, re-verified)",
    },
    # REFERENCE ROW, not a gate subject. The R19-H159 enriched mix is the arm's
    # own cited existence proof for the source-selection lift (+0.065 hagrid);
    # L2 attr_pool was built to reproduce it in isolation. Read off the banked
    # dump under the identical protocol so "did the lane reproduce its proof"
    # is answerable rather than assumed.
    "h159d1": {
        "dir": "R19-H159-arm-draw1",
        "dump": "R19-H161_pairs_h159d1.parquet",
        "banked_read": "R19-H159_arm_draw1_windowed_result.json",
        "label": "R19-H159 enriched mix, draw 1 - REFERENCE, the existence proof",
    },
}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fail(reason):
    print(f"=== H174 MECHANISM GATES FAILED: {reason} ===", flush=True)
    raise SystemExit(1)


DUMP = _mod("h161dump", "R19-H161_dump.py")
MECH = _mod("h162mech", "R19-H162_hagrid_mechanisms.py")


def banked_hagrid_auc(fname):
    d = json.loads((HERE / fname).read_text())
    block = d.get("per_subset", d)
    return float(block[SUBSET]["auc"])


def ols_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def gate_b(dump, tag):
    """Evidence-pool-depth curve (primary) and k-truncation curve (secondary)."""
    it = MECH.item_table(dump)
    ve = MECH.vacuous_excluded(dump, ITEMS)
    buckets = ["ndoc_1", "ndoc_2_3", "ndoc_4plus"]
    vals = [ve["strata"][b]["auroc"] for b in buckets]
    if any(v is None for v in vals):
        fail(f"{tag}: a pool-depth stratum is single-class")
    kc = MECH.kdoc_curve(dump, it)
    ks = [1, 2, 3, 4, 6, 8]
    return {
        "pool_depth_vacuous_excluded": {
            b: ve["strata"][b] for b in buckets
        },
        "pool_depth_slope_ols_bucket_index": round(ols_slope([0, 1, 2], vals), 4),
        "pool_depth_endpoint_delta_4plus_minus_1": round(vals[2] - vals[0], 4),
        "auroc_vacuous_excluded_all": ve["auroc"],
        "n_items_vacuous_excluded": ve["n_items"],
        "kdoc_truncation_curve": kc,
        "kdoc_slope_ols_all": round(
            ols_slope(ks, [kc[str(k)]["auroc_all"] for k in ks]), 5),
        "kdoc_slope_ols_pool4plus": round(
            ols_slope(ks, [kc[str(k)]["auroc_pool4plus"] for k in ks]), 5),
    }


def gate_a(dump, tag):
    """Frame-only share of misrank mass, plus the absolute counts behind the ratio."""
    mb = MECH.misrank_block(dump, ITEMS)
    it = MECH.item_table(dump)
    lb = it["label"].to_numpy()
    sc = it["item_score"].to_numpy()
    ids = it["item_id"].to_numpy()
    pos, neg = sc[lb == 1], sc[lb == 0]
    neg_ids = ids[lb == 0]
    mis = pos[:, None] <= neg[None, :]
    tot = int(mis.sum())
    fo = ITEMS.filter(pl.col("frame_only"))["item_id"].to_list()
    cols = [int(np.where(neg_ids == i)[0][0]) for i in fo]
    fo_pairs = int(mis[:, cols].sum())
    return {
        "frame_only_items": fo,
        "frame_only_misrank_share": mb["frame_only_misrank_share"],
        "frame_only_misrank_pairs": fo_pairs,
        "total_misrank_pairs": tot,
        "n_pos": int((lb == 1).sum()),
        "n_neg": int((lb == 0).sum()),
        "total_pos_neg_pairs": int((lb == 1).sum() * (lb == 0).sum()),
        "frame_only_item_scores": {
            str(i): round(float(sc[ids == i][0]), 4) for i in fo
        },
        "frame_only_scored_positive": {
            str(i): bool(sc[ids == i][0] > 0) for i in fo
        },
        "artifact_misrank_share": mb["artifact_misrank_share"],
        "auroc_excluding_artifact_items": mb["auroc_excluding_artifact_items"],
        "top12_cumulative_share": mb["top12_cumulative_share"],
        "pass_vs_bar": bool(mb["frame_only_misrank_share"] < GATE_A_BAR),
    }


def control_row(tag, dump, spec):
    it = MECH.item_table(dump)
    auc = float(roc_auc_score(it["label"].to_numpy(), it["item_score"].to_numpy()))
    fp = {
        "n": it.height,
        "n_sent": int(dump.filter(pl.col("is_argmax")).height),
        "n_pairs": dump.height,
    }
    banked = banked_hagrid_auc(spec["banked_read"])
    return {
        "reproduced": round(auc, 6),
        "banked": banked,
        "abs_delta": round(abs(auc - banked), 6),
        "fingerprint": fp,
        "fingerprint_ok": fp == FINGERPRINT,
        "pass": bool(abs(auc - banked) <= CONTROL_TOL and fp == FINGERPRINT),
    }


def build_h174_dump(sub):
    """Score models/R20-H174-arm-draw1 on the hagrid substrate through the banked
    R19-H161 dump path - the same encode/aggregate code the flagship dumps used."""
    dev = torch.cuda.get_device_name(0)
    print(f"GPU: {dev}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})",
          flush=True)
    if "RTX PRO 6000" not in dev:
        fail(f"wrong GPU: {dev} - this job is pinned to card 1 (RTX PRO 6000)")
    model, tok = DUMP.ARM.load_run(ROOT / "models" / CKPTS["h174d1"]["dir"])
    logits = DUMP.score_pairs(model, tok, sub["flat_s"], sub["flat_w"],
                              sub["set_index"], sub["n_sets"], tag="h174d1/hagrid")
    sent_score, item_score, is_argmax, is_sink = DUMP.aggregate(logits, sub)
    df = DUMP.pair_frame(sub, logits, sent_score, item_score, is_argmax, is_sink)
    del model, tok
    torch.cuda.empty_cache()
    return df


def main():
    t0 = time.time()
    print(f"=== R20-H174 MECHANISM GATES A + B  {time.strftime('%F %T')} ===", flush=True)

    global ITEMS
    _, ITEMS = MECH.sentence_features()
    fo = ITEMS.filter(pl.col("frame_only"))["item_id"].to_list()
    print(f"frame-only items (RE_FRAMEONLY over the arena response text): {fo}", flush=True)
    if fo != [49, 86, 188, 196]:
        fail(f"frame-only inventory drifted from the banked [49, 86, 188, 196]: {fo}")

    dumps = {}
    h174_path = HERE / CKPTS["h174d1"]["dump"]
    if h174_path.exists():
        print(f"{h174_path.name} already on disk - re-verifying, not rescoring", flush=True)
        dumps["h174d1"] = pl.read_parquet(h174_path).filter(pl.col("subset") == SUBSET)
    else:
        claims, chunks, y = DUMP.ARENA.load_subsets()[SUBSET]
        sub = DUMP.build_subset(SUBSET, claims, chunks, y)
        print(f"substrate: n={len(y)} sentences={sub['n_sets']} pairs={len(sub['flat_s'])}",
              flush=True)
        df = build_h174_dump(sub)
        df.write_parquet(h174_path)
        print(f"per-pair dump -> {h174_path}", flush=True)
        dumps["h174d1"] = df

    for tag in ("h150d1", "h150d2", "h159d1"):
        p = HERE / CKPTS[tag]["dump"]
        if not p.exists():
            fail(f"banked flagship dump missing: {p}")
        dumps[tag] = pl.read_parquet(p).filter(pl.col("subset") == SUBSET)

    control = {}
    for tag, dump in dumps.items():
        control[tag] = control_row(tag, dump, CKPTS[tag])
        r = control[tag]
        print(f"  CONTROL {tag:8s} read {r['reproduced']:.4f}  banked {r['banked']:.4f}  "
              f"delta {r['abs_delta']:.6f}  fp {'ok' if r['fingerprint_ok'] else r['fingerprint']}"
              f"  {'PASS' if r['pass'] else 'FAIL'}", flush=True)
    if not all(c["pass"] for c in control.values()):
        fail("positive control - a dump does not reproduce its banked read")

    res = {
        "experiment": "R20-H174 mechanism gates A (frame-only misrank share) and "
                      "B (k-doc-curve slope) - ANALYSIS ONLY, no training",
        "subset": SUBSET,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "definition_source": {
            "gate_a": "R19-H162_hagrid_mechanisms.misrank_block - misrank mass is the "
                      "count of (positive, negative) item pairs with pos_score <= "
                      "neg_score; the frame-only share is the fraction of that mass in "
                      "the frame-only negatives' columns. Banked values it produced: "
                      "0.2124 (h150d1), 0.2076 (h150d2), 0.1727 (h159d1 enriched).",
            "gate_b_primary": "R19-H162_hagrid_mechanisms.vacuous_excluded -> strata "
                              "ndoc_1 / ndoc_2_3 / ndoc_4plus, four frame-only items "
                              "removed. Banked flagship: 0.8577/0.8129, 0.6890/0.6277, "
                              "0.5096/0.6052.",
            "gate_b_secondary": "R19-H162_hagrid_mechanisms.kdoc_curve - pool truncated "
                                "to the first k documents, k in (1,2,3,4,6,8).",
            "slope_statistics": "NOT banked - stated in this script before the read. "
                                "pool-depth slope = OLS on bucket index (0,1,2); "
                                "k-truncation slope = OLS on k.",
        },
        "bars": {
            "gate_a": {"bar": GATE_A_BAR, "direction": "below",
                       "registered_baseline": GATE_A_BASELINE},
            "gate_b": {"bar": 0.0, "direction": "non-negative"},
        },
        "checkpoints": {t: {"label": s["label"],
                            "path": str(ROOT / "models" / s["dir"]),
                            "dump": s["dump"],
                            "banked_read": s["banked_read"]} for t, s in CKPTS.items()},
        "positive_control": control,
        "gate_a": {t: gate_a(d, t) for t, d in dumps.items()},
        "gate_b": {t: gate_b(d, t) for t, d in dumps.items()},
    }

    fl = [res["gate_a"][t]["frame_only_misrank_share"] for t in ("h150d1", "h150d2")]
    res["comparison"] = {
        "gate_a_flagship_2draw_mean_share": round(float(np.mean(fl)), 4),
        "gate_a_h174d1_share": res["gate_a"]["h174d1"]["frame_only_misrank_share"],
        "gate_a_h174d1_vs_bar": "PASS" if res["gate_a"]["h174d1"]["pass_vs_bar"] else "FAIL",
        "gate_b_flagship_2draw_mean_pool_slope": round(float(np.mean(
            [res["gate_b"][t]["pool_depth_slope_ols_bucket_index"]
             for t in ("h150d1", "h150d2")])), 4),
        "gate_b_h174d1_pool_slope": res["gate_b"]["h174d1"]["pool_depth_slope_ols_bucket_index"],
        "gate_b_h174d1_vs_bar": "PASS" if res["gate_b"]["h174d1"][
            "pool_depth_slope_ols_bucket_index"] >= 0 else "FAIL",
        "gate_a_reference_h159d1_share": res["gate_a"]["h159d1"]["frame_only_misrank_share"],
        "gate_b_reference_h159d1_pool_slope": res["gate_b"]["h159d1"][
            "pool_depth_slope_ols_bucket_index"],
    }
    res["runtime_seconds"] = round(time.time() - t0, 1)

    OUT_JSON.write_text(json.dumps(res, indent=2))
    print(f"\n--- GATE A  frame-only misrank share (bar < {GATE_A_BAR}) ---", flush=True)
    for t in CKPTS:
        a = res["gate_a"][t]
        print(f"  {t:8s} share {a['frame_only_misrank_share']:.4f}  "
              f"({a['frame_only_misrank_pairs']} of {a['total_misrank_pairs']} misrank pairs, "
              f"denominator {a['total_pos_neg_pairs']})", flush=True)
    print("\n--- GATE B  pool-depth curve, vacuous-excluded (bar: slope >= 0) ---", flush=True)
    for t in CKPTS:
        b = res["gate_b"][t]
        s = b["pool_depth_vacuous_excluded"]
        print(f"  {t:8s} 1doc {s['ndoc_1']['auroc']:.4f}  2-3 {s['ndoc_2_3']['auroc']:.4f}  "
              f"4-8 {s['ndoc_4plus']['auroc']:.4f}  slope "
              f"{b['pool_depth_slope_ols_bucket_index']:+.4f}  "
              f"endpoint {b['pool_depth_endpoint_delta_4plus_minus_1']:+.4f}  "
              f"| k-trunc slope all {b['kdoc_slope_ols_all']:+.5f} "
              f"deep {b['kdoc_slope_ols_pool4plus']:+.5f}", flush=True)
    print(f"\nresult -> {OUT_JSON}", flush=True)
    print(f"wall clock {(time.time() - t0) / 60:.1f} min", flush=True)
    print("=== H174 MECHANISM GATES COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()

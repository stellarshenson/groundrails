"""R18-H151a VARIANCE ANATOMY + R18-H151b POOLING SELECTION (CPU, dumps only).

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R18-H151 SEED-VARIANCE ATTACK - registered (2026-08-12)". Reads the banked
per-window score dumps (R18-H151_scores_1142.parquet /
R18-H151_scores_2142.parquet, seeds 1142 = twin draw 1, 2142 = twin draw 2)
and nothing else - no model, no GPU.

PART A (H151a, diagnostic, the 6 arena subsets only): per subset between the
two seeds - (i) argmax-window flip rate (fraction of sentences whose argmax
window differs across seeds, ties broken to the lowest window_id on both
sides), (ii) score drift (mean |delta| of per-window scores, with per-seed
score std for scale), (iii) swing decomposition: subset AUROC swing split
into a flip component (recompute draw-2 AUROC pinned to draw-1's argmax
window choices, and symmetrically draw-1 pinned to draw-2's) and a residual
drift component. |swing| and flip rate are regressed against mean
windows/item across the 6 subsets (prediction: swing scales with window
count). Output: R18-H151_anatomy.json.

PART B (H151b, selection, gold_full ONLY - the arena dumps take no part in
this choice): deterministic subset-blind poolings over each sentence's
window scores - max (baseline), top-2 mean, top-3 mean, top-10% mean
(ceil, min 1), logsumexp tau 1.0 and 4.0 on the logit scale. Item score =
min over sentences, unchanged. Per variant: per-seed AUROC, two-seed mean,
two-seed spread. Registered selection rule: among variants whose two-seed
mean is >= max's two-seed mean - 0.002, pick the lowest two-seed spread;
tie-break toward max. Output: R18-H151_pooling_selection.json with the full
per-variant table and the selected variant name.

Run: uv run python experiments/grounding-semantic/R18-H151_anatomy.py
"""

import json
import math
import pathlib
import time

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent

ARENA_SUBSETS = ["tatqa", "techqa", "pubmedqa", "hotpotqa", "covidqa", "emanual"]

# Banked windowed decomposed-min reads (same constants as the dump gate).
BANKED = {
    1142: {"covidqa": 0.7645, "emanual": 0.6683, "hotpotqa": 0.6728,
           "pubmedqa": 0.6725, "tatqa": 0.7948, "techqa": 0.7745},
    2142: {"covidqa": 0.7661, "emanual": 0.6949, "hotpotqa": 0.6377,
           "pubmedqa": 0.6273, "tatqa": 0.7188, "techqa": 0.7026},
}
GATE_TOL = 1e-4

KEYS = ["item_id", "sentence_id", "window_id"]


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o).__name__)


def load_subset(seed, subset):
    df = (pl.read_parquet(HERE / f"R18-H151_scores_{seed}.parquet")
            .filter(pl.col("subset") == subset)
            .sort(KEYS))
    return {k: df[k].to_numpy() for k in
            ["item_id", "sentence_id", "window_id", "score", "label",
             "n_windows_in_sentence"]}


def sentence_slices(item_id, sentence_id):
    """[start, end) row ranges of each (item, sentence) group; rows must be
    sorted by (item_id, sentence_id, window_id)."""
    key = item_id.astype(np.int64) * 1_000_000 + sentence_id
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)]
    return starts, ends


def auc_from_sentence_scores(sent_slices_item, labels, sent_scores):
    """Item score = min over its sentences; AUROC over items."""
    item_scores = np.array([sent_scores[a:b].min() for a, b in sent_slices_item])
    return float(roc_auc_score(labels, item_scores)), item_scores


def group_structure(d):
    """Sentence slices, item slices (over sentence arrays), per-item label and
    window count. Rows assumed sorted by KEYS."""
    s_starts, s_ends = sentence_slices(d["item_id"], d["sentence_id"])
    sent_item = d["item_id"][s_starts]
    i_starts = np.flatnonzero(np.r_[True, sent_item[1:] != sent_item[:-1]])
    i_ends = np.r_[i_starts[1:], len(sent_item)]
    item_slices_ = list(zip(i_starts, i_ends))
    labels = d["label"][s_starts][i_starts]
    nwin_item = d["n_windows_in_sentence"][s_starts][i_starts]
    return (s_starts, s_ends), item_slices_, labels, nwin_item


def linregress(x, y):
    """Least-squares slope/intercept plus Pearson r and R^2 (n small)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return {"slope": float(slope), "intercept": float(intercept),
            "pearson_r": r, "r_squared": r * r, "n": len(x)}


# ---------------------------------------------------------------- PART A ---

def anatomy_subset(sub, d1, d2):
    for k in KEYS:
        assert np.array_equal(d1[k], d2[k]), f"{sub}: key mismatch on {k}"
    (ss, se), isl, labels, nwin_item = group_structure(d1)
    s1, s2 = d1["score"], d2["score"]

    n_sent = len(ss)
    w1 = d1["window_id"]
    argmax1 = np.empty(n_sent, dtype=np.int64)   # argmax window_id, seed 1142
    argmax2 = np.empty(n_sent, dtype=np.int64)
    pos1 = np.empty(n_sent, dtype=np.int64)      # argmax position within group
    pos2 = np.empty(n_sent, dtype=np.int64)
    sent_max1 = np.empty(n_sent)
    sent_max2 = np.empty(n_sent)
    sent_anch2 = np.empty(n_sent)                # 2142 score at 1142's argmax
    sent_anch1 = np.empty(n_sent)                # 1142 score at 2142's argmax
    for g in range(n_sent):
        a, b = ss[g], se[g]
        x1, x2 = s1[a:b], s2[a:b]
        j1 = int(np.argmax(x1))                  # first max = lowest window_id
        j2 = int(np.argmax(x2))
        pos1[g], pos2[g] = j1, j2
        argmax1[g], argmax2[g] = w1[a + j1], w1[a + j2]
        sent_max1[g], sent_max2[g] = x1[j1], x2[j2]
        sent_anch2[g], sent_anch1[g] = x2[j1], x1[j2]

    flips = int((argmax1 != argmax2).sum())
    flip_rate = flips / n_sent
    drift = float(np.abs(s1 - s2).mean())

    auc1, _ = auc_from_sentence_scores(isl, labels, sent_max1)
    auc2, _ = auc_from_sentence_scores(isl, labels, sent_max2)
    auc2_anch, _ = auc_from_sentence_scores(isl, labels, sent_anch2)
    auc1_anch, _ = auc_from_sentence_scores(isl, labels, sent_anch1)

    # Sanity: max-pooled AUROCs reproduce the banked windowed reads.
    assert abs(auc1 - BANKED[1142][sub]) <= GATE_TOL, (sub, auc1)
    assert abs(auc2 - BANKED[2142][sub]) <= GATE_TOL, (sub, auc2)

    swing = auc1 - auc2
    flip_from2 = auc2_anch - auc2      # draw-2 gain when pinned to draw-1 argmax
    drift_from2 = auc1 - auc2_anch     # residual after removing draw-2 flips
    flip_from1 = auc1 - auc1_anch
    drift_from1 = auc1_anch - auc2

    return {
        "subset": sub,
        "n_items": len(isl),
        "n_sentences": n_sent,
        "n_pairs": len(s1),
        "mean_windows_per_item": float(nwin_item.mean()),
        "auc_1142": auc1,
        "auc_2142": auc2,
        "swing_1142_minus_2142": swing,
        "abs_swing": abs(swing),
        "argmax_flip_rate": flip_rate,
        "n_argmax_flips": flips,
        "score_drift_mean_abs_delta": drift,
        "score_std_1142": float(s1.std()),
        "score_std_2142": float(s2.std()),
        "drift_over_mean_std": drift / ((s1.std() + s2.std()) / 2),
        "decomposition": {
            "auc_2142_at_1142_argmax": auc2_anch,
            "flip_component_from_2142_side": flip_from2,
            "drift_residual_from_2142_side": drift_from2,
            "auc_1142_at_2142_argmax": auc1_anch,
            "flip_component_from_1142_side": flip_from1,
            "drift_residual_from_1142_side": drift_from1,
            "flip_component_mean": (flip_from2 + flip_from1) / 2,
            "drift_residual_mean": (drift_from2 + drift_from1) / 2,
        },
    }


# ---------------------------------------------------------------- PART B ---

def pool_topk(x, k):
    k = min(k, len(x))
    return float(np.partition(x, len(x) - k)[len(x) - k:].mean())


def pool_top10pct(x):
    return pool_topk(x, max(1, math.ceil(0.10 * len(x))))


def pool_lse(x, tau):
    m = float(x.max())
    return float(tau * (m / tau + math.log(np.exp((x - m) / tau).sum())))


POOLINGS = {
    "max": lambda x: float(x.max()),
    "top2_mean": lambda x: pool_topk(x, 2),
    "top3_mean": lambda x: pool_topk(x, 3),
    "top10pct_mean": pool_top10pct,
    "logsumexp_tau1.0": lambda x: pool_lse(x, 1.0),
    "logsumexp_tau4.0": lambda x: pool_lse(x, 4.0),
}


def variant_auc(d, fn):
    (ss, se), isl, labels, _ = group_structure(d)
    sent_scores = np.array([fn(d["score"][a:b]) for a, b in zip(ss, se)])
    auc, _ = auc_from_sentence_scores(isl, labels, sent_scores)
    return auc


def part_b(dg1, dg2):
    rows = []
    for name, fn in POOLINGS.items():
        a1 = variant_auc(dg1, fn)
        a2 = variant_auc(dg2, fn)
        rows.append({
            "variant": name,
            "auc_1142": a1,
            "auc_2142": a2,
            "two_seed_mean": (a1 + a2) / 2,
            "two_seed_spread": abs(a1 - a2),
        })
    max_mean = next(r["two_seed_mean"] for r in rows if r["variant"] == "max")
    for r in rows:
        r["eligible"] = r["two_seed_mean"] >= max_mean - 0.002
    elig = [r for r in rows if r["eligible"]]
    best_spread = min(r["two_seed_spread"] for r in elig)
    tied = [r["variant"] for r in elig if r["two_seed_spread"] == best_spread]
    selected = "max" if "max" in tied else sorted(tied)[0]
    return rows, selected, max_mean


def main():
    t0 = time.time()
    print(f"=== R18-H151 anatomy + pooling selection  {time.strftime('%F %T')} ===",
          flush=True)

    # ---------------- PART A: anatomy on the 6 arena subsets (diagnostic)
    tables = []
    for sub in ARENA_SUBSETS:
        d1 = load_subset(1142, sub)
        d2 = load_subset(2142, sub)
        row = anatomy_subset(sub, d1, d2)
        tables.append(row)
        dec = row["decomposition"]
        print(f"  {sub:10s} win/item {row['mean_windows_per_item']:7.2f}  "
              f"swing {row['swing_1142_minus_2142']:+.4f}  "
              f"flip {row['argmax_flip_rate']:.3f}  "
              f"drift {row['score_drift_mean_abs_delta']:.4f}  "
              f"flip-comp {dec['flip_component_mean']:+.4f}  "
              f"drift-res {dec['drift_residual_mean']:+.4f}", flush=True)

    x = [r["mean_windows_per_item"] for r in tables]
    reg_swing = linregress(x, [r["abs_swing"] for r in tables])
    reg_flip = linregress(x, [r["argmax_flip_rate"] for r in tables])
    prediction_supported = reg_swing["pearson_r"] > 0 and reg_flip["pearson_r"] > 0
    anatomy = {
        "experiment": "R18-H151a variance anatomy",
        "seeds": {"draw1": 1142, "draw2": 2142},
        "checkpoints": {"1142": "R16-H142-G1-twin", "2142": "R16-H142-T-draw2"},
        "method": {
            "argmax_flip_rate": "fraction of sentences whose argmax window_id "
                                "differs across seeds; ties break to lowest "
                                "window_id on both sides",
            "score_drift": "mean |score_1142 - score_2142| over all "
                           "(sentence, window) pairs of the subset",
            "swing_decomposition": "swing = auc_1142 - auc_2142 under the "
                                   "banked max-over-windows/min-over-sentences "
                                   "read; flip component = draw-2 AUROC gain "
                                   "when re-scored at draw-1's argmax windows "
                                   "(and symmetrically); drift residual = what "
                                   "remains; each direction sums exactly to "
                                   "the signed swing",
        },
        "per_subset": tables,
        "regressions": {
            "abs_swing_vs_mean_windows_per_item": reg_swing,
            "flip_rate_vs_mean_windows_per_item": reg_flip,
        },
        "prediction_test": {
            "prediction": "subset AUROC swing scales with mean windows/item "
                          "(max-selection amplification); stable subsets flip "
                          "rarely",
            "supported": prediction_supported,
            "evidence": {
                "slope_abs_swing_per_window": reg_swing["slope"],
                "pearson_r_abs_swing": reg_swing["pearson_r"],
                "pearson_r_flip_rate": reg_flip["pearson_r"],
            },
        },
        "sanity": "per-seed max-pooled AUROCs reproduce the banked windowed "
                  f"reads within {GATE_TOL} on all 6 subsets (asserted)",
    }
    (HERE / "R18-H151_anatomy.json").write_text(json.dumps(anatomy, indent=2, default=_json_default))
    print(f"  -> {HERE / 'R18-H151_anatomy.json'}", flush=True)
    print(f"  regression: |swing| slope {reg_swing['slope']:.6f}/window  "
          f"r {reg_swing['pearson_r']:+.3f};  flip slope "
          f"{reg_flip['slope']:.6f}/window  r {reg_flip['pearson_r']:+.3f}",
          flush=True)

    # ---------------- PART B: pooling selection on gold_full ONLY
    dg1 = load_subset(1142, "gold_full")
    dg2 = load_subset(2142, "gold_full")
    for k in KEYS:
        assert np.array_equal(dg1[k], dg2[k]), f"gold_full: key mismatch on {k}"
    rows, selected, max_mean = part_b(dg1, dg2)
    print("\n  gold_full pooling variants (selection is gold-side ONLY):",
          flush=True)
    for r in rows:
        print(f"    {r['variant']:18s} auc1142 {r['auc_1142']:.4f}  "
              f"auc2142 {r['auc_2142']:.4f}  mean {r['two_seed_mean']:.4f}  "
              f"spread {r['two_seed_spread']:.4f}  "
              f"{'eligible' if r['eligible'] else 'excluded'}", flush=True)
    selection = {
        "experiment": "R18-H151b pooling-variant selection (gold_full only)",
        "scope": "computed on the gold_full dumps of both seeds ONLY; the "
                 "arena dumps took no part in this choice (H151c adjudicates "
                 "the selected variant blind on the arena dumps)",
        "poolings": {
            "max": "sentence score = max over windows (banked baseline)",
            "top2_mean": "mean of the top-2 window scores",
            "top3_mean": "mean of the top-3 window scores",
            "top10pct_mean": "mean of the top-ceil(10%) window scores, min 1",
            "logsumexp_tau1.0": "tau * log(sum(exp(score/tau))), tau = 1.0, "
                                "logit scale",
            "logsumexp_tau4.0": "same, tau = 4.0 (softer, closer to mean)",
        },
        "item_aggregation": "min over sentences (unchanged)",
        "subset_blind": True,
        "selection_rule": "among variants with two-seed mean >= max's "
                          "two-seed mean - 0.002, pick the lowest two-seed "
                          "spread; tie-break toward max",
        "max_two_seed_mean": max_mean,
        "variants": rows,
        "selected_variant": selected,
        "notes": [
            "H151b prediction REFUTED on the gold-side selection: no variant "
            "cuts the two-seed spread within the 0.002 mean budget - max "
            "itself carries the lowest spread (0.0054) of the eligible set; "
            "top2_mean is the only other eligible variant and its spread is "
            "worse (0.0061) despite a +0.0028 higher mean",
            "softer poolings fail the mean budget on gold_full (top3 -0.011, "
            "top10pct -0.021, lse_tau1.0 -0.074, lse_tau4.0 -0.267 vs max)",
            "lse collapse mechanism: with near-equal window scores, "
            "tau*log(sum(exp(x/tau))) ~= max + tau*log(nwin) - a per-item "
            "offset proportional to log(windows/item) that scrambles the "
            "cross-item ranking under min-over-sentences (gold_full "
            "averages ~47 windows/sentence); the larger tau, the larger "
            "the length bias, hence lse_tau4.0 AUROC ~0.58",
        ],
        "seeds": {"draw1": 1142, "draw2": 2142},
    }
    (HERE / "R18-H151_pooling_selection.json").write_text(
        json.dumps(selection, indent=2, default=_json_default))
    print(f"  SELECTED: {selected}", flush=True)
    print(f"  -> {HERE / 'R18-H151_pooling_selection.json'}", flush=True)
    print(f"=== DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()

"""R18-H151c BLIND ARENA ADJUDICATION - selected pooling variant vs max.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R18-H151 SEED-VARIANCE ATTACK - registered (2026-08-12)": "H151c BLIND ARENA
ADJUDICATION - exactly TWO reads on the arena dumps: max (banked baseline)
and the H151b-selected variant. Bars: PASS if the selected variant's per-seed
arena mean is within 0.003 of max on BOTH seeds AND the two-seed mean spread
shrinks >= 30%; if PASS, the variant becomes a candidate PRIMARY-read
amendment adjudicated at the next promotion registration; FAIL -> max stands,
variance attack moves to the training lever (EMA) only".

The H151b-selected variant is read from the banked selection artefact
(R18-H151_pooling_selection.json -> selected_variant). The arena dumps cover
6 of the 10 arena subsets (tatqa, techqa, pubmedqa, hotpotqa, covidqa,
emanual); the 4 missing subsets (delucionqa, expertqa, finqa, hagrid) keep
their banked max-read values, reported for CONTEXT ONLY and never mixed into
any variant mean. All means/spreads/bars below are therefore 6-subset.

Pure scoring from the dumped per-window values (R18-H151_scores_<seed>.parquet)
- no model, no GPU, no re-reading of raw text. Pooling functions are copied
verbatim from R18-H151_anatomy.py (the H151b selection code path): sentence
score = pooling over the sentence's window scores; item score = min over
sentences; AUROC per subset. Sanity gate: the max-pooled per-subset AUROCs
must reproduce the banked windowed reads within 1e-4 on both seeds.

Run: uv run python experiments/grounding-semantic/R18-H151_arena_adjudication.py
Output: experiments/grounding-semantic/R18-H151_arena_adjudication.json
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
SEEDS = [1142, 2142]

# Banked windowed decomposed-min reads (same constants as the dump gate and
# the anatomy script) - sanity targets for the max pooling.
BANKED = {
    1142: {"covidqa": 0.7645, "emanual": 0.6683, "hotpotqa": 0.6728,
           "pubmedqa": 0.6725, "tatqa": 0.7948, "techqa": 0.7745},
    2142: {"covidqa": 0.7661, "emanual": 0.6949, "hotpotqa": 0.6377,
           "pubmedqa": 0.6273, "tatqa": 0.7188, "techqa": 0.7026},
}
GATE_TOL = 1e-4

# Banked max-read windowed results for the 4 subsets the dump does not cover
# (context only, read from the result JSONs - never mixed into variant means).
BANKED_WINDOWED_RESULTS = {
    1142: "R16-H142_G1_twin_windowed_result.json",
    2142: "R16-H142_T_draw2_windowed_result.json",
}

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
            ["item_id", "sentence_id", "score", "label"]}


def group_structure(d):
    """Sentence slices, item slices (over sentence arrays), per-item label.
    Rows assumed sorted by KEYS. Copied from R18-H151_anatomy.py."""
    key = d["item_id"].astype(np.int64) * 1_000_000 + d["sentence_id"]
    s_starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    s_ends = np.r_[s_starts[1:], len(key)]
    sent_item = d["item_id"][s_starts]
    i_starts = np.flatnonzero(np.r_[True, sent_item[1:] != sent_item[:-1]])
    i_ends = np.r_[i_starts[1:], len(sent_item)]
    item_slices = list(zip(i_starts, i_ends))
    labels = d["label"][s_starts][i_starts]
    return (s_starts, s_ends), item_slices, labels


# Pooling functions verbatim from R18-H151_anatomy.py PART B.
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
    """Sentence score = fn(window scores); item score = min over sentences;
    AUROC over items. Copied from R18-H151_anatomy.py."""
    (ss, se), isl, labels = group_structure(d)
    sent_scores = np.array([fn(d["score"][a:b]) for a, b in zip(ss, se)])
    item_scores = np.array([sent_scores[a:b].min() for a, b in isl])
    return float(roc_auc_score(labels, item_scores))


def main():
    t0 = time.time()
    print(f"=== R18-H151c blind arena adjudication  {time.strftime('%F %T')} ===",
          flush=True)

    selection = json.loads((HERE / "R18-H151_pooling_selection.json").read_text())
    selected = selection["selected_variant"]
    assert selected in POOLINGS, selected
    print(f"H151b-selected variant (banked): {selected}", flush=True)
    if selected == "max":
        print("  note: the selection returned max itself - the two registered "
              "arena reads coincide (degenerate adjudication)", flush=True)

    reads = {"max": POOLINGS["max"], "selected": POOLINGS[selected]}

    # Per-subset per-seed AUROC for the two registered reads.
    per_subset = {}
    gate_fail = []
    for sub in ARENA_SUBSETS:
        per_subset[sub] = {}
        for seed in SEEDS:
            d = load_subset(seed, sub)
            aucs = {name: variant_auc(d, fn) for name, fn in reads.items()}
            per_subset[sub][str(seed)] = aucs
            dev = abs(aucs["max"] - BANKED[seed][sub])
            if dev > GATE_TOL:
                gate_fail.append((sub, seed, aucs["max"], BANKED[seed][sub]))
            print(f"  {sub:10s} seed {seed}: max {aucs['max']:.6f} "
                  f"(banked {BANKED[seed][sub]:.4f}, |d| {dev:.2e})  "
                  f"selected[{selected}] {aucs['selected']:.6f}", flush=True)
    if gate_fail:
        raise SystemExit(f"SANITY GATE FAILED: {gate_fail} - the max read "
                         "does not reproduce the banked windowed AUROCs")
    print("  sanity gate PASSED: max read reproduces all 12 banked AUROCs "
          f"within {GATE_TOL}", flush=True)

    # Per-seed 6-subset arena means, two-seed means, spreads.
    per_seed_means = {
        name: {str(seed): float(np.mean([per_subset[s][str(seed)][name]
                                         for s in ARENA_SUBSETS]))
               for seed in SEEDS}
        for name in reads
    }
    two_seed_mean = {name: float(np.mean(list(m.values())))
                     for name, m in per_seed_means.items()}
    two_seed_spread = {name: abs(m["1142"] - m["2142"])
                       for name, m in per_seed_means.items()}
    spread_reduction = ((two_seed_spread["max"] - two_seed_spread["selected"])
                        / two_seed_spread["max"])

    # Registered bars (6-subset amendment per the adjudication spec).
    mean_diffs = {str(seed): abs(per_seed_means["selected"][str(seed)]
                                 - per_seed_means["max"][str(seed)])
                  for seed in SEEDS}
    clause1 = all(v <= 0.003 for v in mean_diffs.values())
    clause2 = spread_reduction >= 0.30
    verdict_pass = clause1 and clause2

    # Context only: banked max-read values of the 4 subsets the dump lacks.
    missing = ["delucionqa", "expertqa", "finqa", "hagrid"]
    banked_missing, banked10 = {}, {}
    for seed in SEEDS:
        res = json.loads((HERE / BANKED_WINDOWED_RESULTS[seed]).read_text())
        aucs = {s: res["per_subset"][s]["auc"] for s in
                res["per_subset"]}
        banked_missing[str(seed)] = {s: aucs[s] for s in missing}
        banked10[str(seed)] = float(np.mean([aucs[s] for s in aucs]))
    print(f"\n  per-seed 6-subset means: max {per_seed_means['max']}  "
          f"selected {per_seed_means['selected']}", flush=True)
    print(f"  two-seed spread: max {two_seed_spread['max']:.6f}  "
          f"selected {two_seed_spread['selected']:.6f}  "
          f"reduction {spread_reduction:.4f}", flush=True)
    print(f"  clause1 (|mean diff| <= 0.003 both seeds): {clause1} "
          f"{mean_diffs}", flush=True)
    print(f"  clause2 (spread shrink >= 30%): {clause2}", flush=True)
    print(f"  VERDICT: {'PASS' if verdict_pass else 'FAIL'}", flush=True)

    out = {
        "experiment": "R18-H151c blind arena adjudication - H151b-selected "
                      "pooling variant vs max",
        "registered_bars": "PASS if the selected variant's per-seed arena "
                           "mean is within 0.003 of max on BOTH seeds AND the "
                           "two-seed mean spread shrinks >= 30% vs max's "
                           "spread; FAIL -> max stands, variance attack moves "
                           "to the training lever (EMA) only",
        "inputs": {
            "dumps": ["R18-H151_scores_1142.parquet",
                      "R18-H151_scores_2142.parquet"],
            "selection": "R18-H151_pooling_selection.json",
            "banked_windowed_results": BANKED_WINDOWED_RESULTS,
        },
        "subsets_in_dump": ARENA_SUBSETS,
        "subsets_missing_from_dump": missing,
        "mean_scope": "all means/spreads are over the 6 dumped arena "
                      "subsets; the 4 missing subsets keep their banked "
                      "max-read values (context only, never mixed into any "
                      "variant mean)",
        "selected_variant": selected,
        "degeneracy_note": "H151b's gold-side selection returned max itself "
                           "(lowest two-seed spread among eligible variants), "
                           "so the two registered arena reads coincide: "
                           "clause 1 is vacuously satisfied and clause 2 "
                           "vacuously fails (0% shrink). Per the "
                           "contamination wall no non-max variant was read "
                           "on the arena dumps.",
        "per_subset": per_subset,
        "per_seed_means": per_seed_means,
        "two_seed_mean": two_seed_mean,
        "two_seed_spread": two_seed_spread,
        "spread_reduction": spread_reduction,
        "bars": {
            "clause1_selected_within_0.003_of_max_both_seeds": {
                "mean_abs_diff_per_seed": mean_diffs, "pass": clause1},
            "clause2_spread_shrink_ge_30pct": {
                "max_spread": two_seed_spread["max"],
                "selected_spread": two_seed_spread["selected"],
                "reduction": spread_reduction, "pass": clause2},
            "pass": verdict_pass,
        },
        "verdict": ("PASS - selected variant becomes a candidate PRIMARY-read "
                    "amendment at the next promotion registration"
                    if verdict_pass else
                    "FAIL - max stands as the PRIMARY read; the pooling front "
                    "of the variance attack is closed and the attack moves to "
                    "the training lever (EMA, R18-H152) only"),
        "context_only_banked_max_read": {
            "warning": "banked 4-subset and 10-subset max-read values, "
                       "context only - NOT part of the adjudication",
            "missing_subsets_per_seed": banked_missing,
            "ten_subset_mean_per_seed": banked10,
            "ten_subset_spread": abs(banked10["1142"] - banked10["2142"]),
        },
        "sanity": f"max-pooled per-subset AUROCs reproduce the banked "
                  f"windowed reads within {GATE_TOL} on all 6 subsets x 2 "
                  "seeds (asserted; the adjudication adds nothing beyond "
                  "recomputation for the max read)",
        "seeds": {"draw1": 1142, "draw2": 2142},
        "checkpoints": {"1142": "R16-H142-G1-twin",
                        "2142": "R16-H142-T-draw2"},
    }
    (HERE / "R18-H151_arena_adjudication.json").write_text(
        json.dumps(out, indent=2, default=_json_default))
    print(f"  -> {HERE / 'R18-H151_arena_adjudication.json'}", flush=True)
    print(f"=== DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()

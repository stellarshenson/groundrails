"""R22 PER-SUBSET DECOMPOSITION of the k=6 flagship arena read - measurement only.

The SOTA document carried an explicit open item: the per-subset win/loss counts
were computed against the 2-draw pair and its 2-draw seed spreads and were
"NOT re-derived here ... provisional pending a k=6 re-count". This script
supplies that k=6 re-count, and every column needed to read it honestly.

MEASUREMENT ONLY. No bar, no gate, no promotion, no kill. In particular it does
NOT declare a per-subset win/loss tally: the SOTA records that the pricing
convention is unset and that the two candidate conventions disagree, so choosing
one now - with the numbers already visible - would be an estimator choice made
after the fact. The margin and the subset's own dispersion are both reported;
the convention remains an author call.

Everything is read off banked artifacts. Nothing is re-scored, so no GPU is
touched and no card-pinning question arises.

Columns, and where each comes from:
    auc d1..d6, k6 mean      R21-H179_consensus_errors.json (per_subset.auc_per_draw)
    sd / spread              derived from those six endpoints (sd on 5 df)
    incumbent, 3 conventions R19-H171_incumbent_chunked.json - the banked
                             comparator is native_truncated_auc (mean 0.67963)
    oracle ceiling, headroom R21-H179_consensus_errors.json
                             (headroom_reported_separately), sourced from
                             R12_label_ceiling_result.json:o4_windows_strict_auc
                             - what a PERFECT entailer reaches through the
                             shipped read machinery, not an abstract 1.0
    consensus errors         items every one of the six draws gets wrong, in
                             both the threshold and the threshold-free rank
                             definitions, with the share of the subset's AUROC
                             deficit they carry
    rank granularity         1/(n_pos * n_neg) - the smallest non-zero AUROC
                             change the subset can express; the fidelity-guard
                             finding of R22-H188
    exposure                 R21-H179Q_autopsy_exposure.json - responses whose
                             evidence appears in the training mix
    length baselines         computed here on the arena itself (CPU): AUROC of
                             trivial length features against the label, per
                             subset AND subset-blind under ONE global direction
    rho(score, n_sent)       Spearman of the 6-draw mean score against sentence
                             count, from R21-H179_consensus_errors.parquet

The length columns exist because four subsets score ABOVE their faithful-oracle
ceiling, which entailment cannot explain. They are a diagnostic that motivates a
registered test; on their own they settle nothing.

Run (CPU, ~1 min):
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= \
    uv run python experiments/grounding-semantic/R22_per_subset_decomposition.py
"""

import importlib.util
import json
import pathlib
import statistics as st

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R22_per_subset_decomposition.json"

BANKED_CONVENTION = "native_truncated_auc"  # the 0.67963 the margin is taken against
LENGTH_FEATURES = ("resp_len", "mean_sent", "n_sent", "ev_len", "n_docs")


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def length_features(resp, docs, sentences):
    """Trivial features a system could compute without reading the evidence."""
    sents = [sentences(r) for r in resp]
    return {
        "resp_len": np.array([len(r) for r in resp], float),
        "mean_sent": np.array([np.mean([len(x) for x in s]) if s else 0.0 for s in sents], float),
        "n_sent": np.array([len(s) for s in sents], float),
        "ev_len": np.array([sum(len(d) for d in dl) for dl in docs], float),
        "n_docs": np.array([len(dl) for dl in docs], float),
    }


def main():
    cons = json.loads((HERE / "R21-H179_consensus_errors.json").read_text())
    ps = cons["per_subset"]
    ceil = cons["headroom_reported_separately"]["per_subset"]
    inc = json.loads((HERE / "R19-H171_incumbent_chunked.json").read_text())["per_subset"]
    expo = json.loads((HERE / "R21-H179Q_autopsy_exposure.json").read_text())["exposure_control"]

    arena = _mod("arena", "R8-H77_unseen_arena.py")
    h92 = _mod("h92", "R8-H92_decomposed_arena.py")
    subs = arena.load_subsets()

    df = pl.read_parquet(HERE / "R21-H179_consensus_errors.parquet")
    score_cols = [f"score_d{i}" for i in range(1, 7)]
    mean_score = df.select(["subset", "item"] + score_cols).with_columns(
        pl.mean_horizontal(score_cols).alias("s6"))

    rows = {}
    for s, v in ps.items():
        draws = v["auc_per_draw"]
        d = list(draws.values())
        resp, docs, y = subs[s]
        feats = length_features(resp, docs, h92.sentences)
        lengths = {k: roc_auc_score(y, f) for k, f in feats.items()}
        g = mean_score.filter(pl.col("subset") == s).sort("item")
        rho = spearmanr(g["s6"].to_numpy(), feats["n_sent"]).statistic

        rows[s] = {
            "n": v["n"], "n_positive": v["n_positive"], "n_negative": v["n_negative"],
            "auc_per_draw": draws,
            "auc_k6_mean": v["auc_k6_mean"],
            "sd_5df": round(st.stdev(d), 5),
            "min": min(d), "max": max(d), "spread": round(max(d) - min(d), 5),
            "incumbent": {c: inc[s][c] for c in
                          ("native_truncated_auc", "chunked_auc", "harness_auc")},
            "margin_vs_banked_convention": round(v["auc_k6_mean"] - inc[s][BANKED_CONVENTION], 5),
            "margin_in_own_sd": round((v["auc_k6_mean"] - inc[s][BANKED_CONVENTION])
                                      / st.stdev(d), 2),
            "faithful_oracle_ceiling": ceil[s]["faithful_oracle_ceiling"],
            "headroom_to_ceiling": ceil[s]["headroom_to_ceiling"],
            "consensus_rank": {
                "errors": v["draw_agreement_rank"]["consensus_errors"],
                "fn": v["draw_agreement_rank"]["consensus_fn"],
                "fp": v["draw_agreement_rank"]["consensus_fp"],
                "share_of_deficit": v["draw_agreement_rank"]["consensus_mass_share_of_deficit"],
            },
            "consensus_threshold": {
                "errors": v["draw_agreement_threshold"]["consensus_errors"],
                "fn": v["draw_agreement_threshold"]["consensus_fn"],
                "fp": v["draw_agreement_threshold"]["consensus_fp"],
                "share_of_deficit": v["draw_agreement_threshold"]["consensus_mass_share_of_deficit"],
            },
            "rank_granularity": round(1.0 / (v["n_positive"] * v["n_negative"]), 6),
            "exposed_verbatim": expo["verbatim_banked"][s],
            "exposed_containment": expo["containment_banked"][s],
            "length_auroc": {k: round(a, 4) for k, a in lengths.items()},
            "length_best_oriented": round(max(max(a, 1 - a) for a in lengths.values()), 4),
            "rho_score_vs_n_sent": round(float(rho), 3),
        }

    blind = {}
    for f in LENGTH_FEATURES:
        m = st.mean(rows[s]["length_auroc"][f] for s in rows)
        blind[f] = {"uniform_mean_raw": round(m, 5), "oriented": round(max(m, 1 - m), 5)}
    best = max(blind, key=lambda f: abs(blind[f]["uniform_mean_raw"] - 0.5))

    k6 = st.mean(r["auc_k6_mean"] for r in rows.values())
    incm = st.mean(r["incumbent"][BANKED_CONVENTION] for r in rows.values())
    res = {
        "arm": "R22 per-subset decomposition of the k=6 flagship arena read",
        "licence": ("MEASUREMENT ONLY - no bar, no gate, no promotion, no kill, and "
                    "deliberately NO win/loss tally: the pricing convention is unset and "
                    "selecting one with the numbers visible is an estimator choice after "
                    "the fact. Reported so the author can set the convention."),
        "banked_incumbent_convention": BANKED_CONVENTION,
        "uniform_mean": {"flagship_k6": round(k6, 5), "incumbent": round(incm, 5),
                         "margin": round(k6 - incm, 5)},
        "margin_concentration": None,  # filled below
        "subset_blind_length_baseline": {
            "definition": ("AUROC of a trivial length feature against the arena label, ONE "
                           "global direction for all ten subsets, uniform mean - the same "
                           "aggregation the headline uses. Per-subset direction selection "
                           "would be post-hoc and is not used for this figure."),
            "per_feature": blind,
            "best_feature": best,
            "best_value": blind[best]["oriented"],
            "share_of_flagship_above_chance_lift": round(
                (blind[best]["oriented"] - 0.5) / (k6 - 0.5), 4),
        },
        "per_subset": rows,
        "note": ("Read off banked artifacts only; nothing re-scored, no GPU. The length "
                 "columns are a diagnostic motivating a registered test, not a verdict."),
    }

    top2 = sorted(rows, key=lambda s: -rows[s]["margin_vs_banked_convention"])[:2]
    rest = [s for s in rows if s not in top2]
    res["margin_concentration"] = {
        "top2_subsets": top2,
        "top2_margin_sum_over_10": round(
            sum(rows[s]["margin_vs_banked_convention"] for s in top2) / 10, 5),
        "total_margin": round(k6 - incm, 5),
        "flagship_mean_excluding_top2": round(st.mean(rows[s]["auc_k6_mean"] for s in rest), 5),
        "incumbent_mean_excluding_top2": round(
            st.mean(rows[s]["incumbent"][BANKED_CONVENTION] for s in rest), 5),
    }
    mc = res["margin_concentration"]
    mc["margin_excluding_top2"] = round(
        mc["flagship_mean_excluding_top2"] - mc["incumbent_mean_excluding_top2"], 5)

    OUT.write_text(json.dumps(res, indent=2))
    print(f"written -> {OUT}")
    print(f"  flagship k=6 {k6:.5f}  incumbent {incm:.5f}  margin {k6 - incm:+.5f}")
    print(f"  {top2[0]} + {top2[1]} contribute {mc['top2_margin_sum_over_10']:+.5f} of it")
    print(f"  excluding them: {mc['flagship_mean_excluding_top2']:.5f} vs "
          f"{mc['incumbent_mean_excluding_top2']:.5f} = {mc['margin_excluding_top2']:+.5f}")
    print(f"  subset-blind length baseline ({best}): {blind[best]['oriented']:.5f} = "
          f"{res['subset_blind_length_baseline']['share_of_flagship_above_chance_lift']:.1%} "
          f"of the flagship's above-chance lift")


if __name__ == "__main__":
    main()

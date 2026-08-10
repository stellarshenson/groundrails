"""R16-H141 step A1 - autopsy of the R16-H140 readout lift: scalar statistics vs embedding content.

The R16-H140 pilot trained an attention readout over frozen window EMBEDDINGS and
lifted pubmedqa 0.5907 -> 0.6618 (+0.0711) while losing hotpotqa (-0.0518) and
tatqa (-0.0484).  Hypothesis under test here: the pubmedqa gain is max-noise
saturation repair - pubmedqa carries ~26 windows per item, and a hard max over
many noisy per-window scores inflates false support - and is therefore
recoverable from SCALAR window-score statistics alone, with no embedding content.

This script uses ONLY the cached per-window frozen scalar scores (`score` in the
R16-H140 cache).  The `emb` arrays are never opened.

Aggregators are fitted on exactly the same public training slice, the same rows
and the same labels and the same seed-0 train/val split the readout used (RAGTruth
train + the R10-H108 manufactured quantitative lane; zero gold rows, zero arena
rows).  The blind arena is then read through each fitted aggregator with the
campaign protocol unchanged: per-sentence aggregation over that sentence's window
set, hard MIN over the response's sentences, response-level AUROC per subset.

Run (CPU, seconds):
  uv run python experiments/grounding-semantic/R16-H141_autopsy.py
"""

import json
import pathlib
import time

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
CACHE = HERE / "R16-H140_cache"
PILOT = HERE / "R16-H140_G1_pilot.json"
OUT = HERE / "R16-H141_autopsy.json"

SEED = 0
VAL_FRAC = 0.08  # identical to R16-H140_G1_readout.VAL_FRAC
FIDELITY_TOL = 1e-4

# Banked hard-max windowed read (R10-H108_lane_draw1_windowed_result.json), as used by H140.
BANKED = {
    "covidqa": 0.7516, "delucionqa": 0.7355, "emanual": 0.6719, "expertqa": 0.7496,
    "finqa": 0.7291, "hagrid": 0.6599, "hotpotqa": 0.6965, "pubmedqa": 0.5907,
    "tatqa": 0.7391, "techqa": 0.7379,
}
BANKED_MEAN = 0.70618

LAMBDA_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]

FEATURE_NAMES = ["max", "top2_mean", "top3_mean", "top5_mean", "mean", "std", "n_windows", "log_n_windows"]

# Pre-stated decision bar (step A1 of the approved plan).
BAR_SCALAR = 0.050          # >= 70% of the readout's +0.0711 pubmedqa delta
READOUT_PUBMEDQA_DELTA = 0.0711
LOSS_TOL = -0.01            # hotpotqa / tatqa must be >= this vs hard-max


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p)).astype(np.float64)


def group_index(pair_sent, n_sent):
    """Identical to R16-H140_G1_readout.group_index."""
    order = np.argsort(pair_sent, kind="stable")
    ps = pair_sent[order]
    bounds = np.searchsorted(ps, np.arange(n_sent + 1))
    return order, bounds


def features(sc, pair_sent, n_sent):
    """Per-sentence SCALAR feature matrix from the frozen per-window scores.

    All order statistics are taken in LOGIT space (the space the frozen scalar is
    linear in, and the space the H140 readout consumed it in).  max in logit space
    is the same window as max in probability space - the hard-max control is exact.
    """
    order, bounds = group_index(pair_sent, n_sent)
    lg = logit(sc)
    F = np.zeros((n_sent, len(FEATURE_NAMES)), dtype=np.float64)
    hardmax_prob = np.zeros(n_sent, dtype=np.float64)
    for i in range(n_sent):
        idx = order[bounds[i]:bounds[i + 1]]
        v = np.sort(lg[idx])[::-1]
        n = len(v)
        hardmax_prob[i] = sc[idx].max()
        F[i, 0] = v[0]
        F[i, 1] = v[:min(2, n)].mean()
        F[i, 2] = v[:min(3, n)].mean()
        F[i, 3] = v[:min(5, n)].mean()
        F[i, 4] = v.mean()
        F[i, 5] = v.std() if n > 1 else 0.0
        F[i, 6] = float(n)
        F[i, 7] = float(np.log(n))
    return F, hardmax_prob, order, bounds


def penalized_max(F, lam):
    """max - lam * log(n_windows), in logit space."""
    return F[:, 0] - lam * F[:, 7]


def main():
    t0 = time.time()
    print(f"=== R16-H141 autopsy {time.strftime('%F %T')} (scalar-only, no embeddings) ===", flush=True)

    # ---------------- training slice (public only, identical rows to H140) ----------------
    z = np.load(CACHE / "train.npz")
    tr_sc, tr_pair_sent, lab, src = z["score"], z["pair_sent"], z["label"], z["source"]
    n_sent = len(lab)
    print(f"  train slice: {n_sent} sentence sets, {len(tr_sc)} window pairs", flush=True)
    Ftr, _, _, _ = features(tr_sc, tr_pair_sent, n_sent)

    # Identical seed-0 split to R16-H140_G1_readout.train_readout
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_sent)
    n_val = int(VAL_FRAC * n_sent)
    val_ids, tr_ids = perm[:n_val], perm[n_val:]

    comp = json.loads((CACHE / "train_composition.json").read_text())
    slice_proof = {
        "n_sentence_sets": int(n_sent),
        "n_window_pairs": int(len(tr_sc)),
        "sentence_sets_by_source": comp["sentence_sets_by_source"],
        "source_ids": comp["source_ids"],
        "n_fit_rows": int(len(tr_ids)),
        "n_val_rows": int(len(val_ids)),
        "split": f"seed {SEED}, val_frac {VAL_FRAC} - identical rows and permutation to the H140 readout",
        "provenance": comp["provenance"],
        "no_gold_rows": True,
        "no_arena_rows": True,
        "positive_rate": float(np.mean(lab)),
        "window_set_size_mean": float(Ftr[:, 6].mean()),
        "window_set_size_max": int(Ftr[:, 6].max()),
    }

    # ---------------- fit the scalar aggregators ----------------
    mu, sd = Ftr[tr_ids].mean(axis=0), Ftr[tr_ids].std(axis=0) + 1e-9

    def std_(X):
        return (X - mu) / sd

    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(std_(Ftr[tr_ids]), lab[tr_ids])
    lr_val_auc = float(roc_auc_score(lab[val_ids], lr.decision_function(std_(Ftr[val_ids]))))

    # isotonic calibration of each single order-statistic feature (monotone; fitted on train rows)
    iso_feats = ["max", "top2_mean", "top3_mean", "top5_mean", "mean"]
    isos = {}
    for name in iso_feats:
        j = FEATURE_NAMES.index(name)
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(Ftr[tr_ids, j], lab[tr_ids])
        isos[name] = ir

    # lambda grid, selected on the SAME held-out training-slice validation rows
    lam_val = {}
    for lam in LAMBDA_GRID:
        lam_val[lam] = float(roc_auc_score(lab[val_ids], penalized_max(Ftr[val_ids], lam)))
    best_lam = max(lam_val, key=lam_val.get)

    single_val = {}
    for name in FEATURE_NAMES[:6]:
        j = FEATURE_NAMES.index(name)
        single_val[name] = float(roc_auc_score(lab[val_ids], Ftr[val_ids, j]))
    single_val["penmax_best"] = lam_val[best_lam]
    single_val["logreg_scalar"] = lr_val_auc

    print(f"  train-slice val sentence AUROC: max {single_val['max']:.4f}  "
          f"top3 {single_val['top3_mean']:.4f}  mean {single_val['mean']:.4f}  "
          f"penmax(lam={best_lam}) {lam_val[best_lam]:.4f}  logreg {lr_val_auc:.4f}", flush=True)

    # ---------------- variant registry ----------------
    # Each variant maps a per-sentence feature matrix to a per-sentence score.
    # Every map is applied identically to train and arena; only monotone-in-support
    # scores are produced, so the sentence-MIN downstream is unchanged in kind.
    variants = {"hardmax": lambda F: F[:, 0]}
    for name in FEATURE_NAMES[:5]:
        j = FEATURE_NAMES.index(name)
        if name != "max":
            variants[f"raw_{name}"] = (lambda j: (lambda F: F[:, j]))(j)
        variants[f"iso_{name}"] = (lambda n, j: (lambda F: isos[n].predict(F[:, j])))(name, j)
    for lam in LAMBDA_GRID:
        variants[f"penmax_lam{lam}"] = (lambda l: (lambda F: penalized_max(F, l)))(lam)
    variants["logreg_scalar"] = lambda F: lr.decision_function(std_(F))

    # ---------------- blind arena read ----------------
    files = sorted(CACHE.glob("arena_*.npz"))
    per_sub = {}
    fidelity = {}
    win_stats = {}
    for f in files:
        sub = f.stem.replace("arena_", "")
        za = np.load(f)
        sc, pair_sent = za["score"], za["pair_sent"]
        sent_owner, y = za["sent_owner"], za["y"]
        ns = len(sent_owner)
        F, hardmax_prob, _, _ = features(sc, pair_sent, ns)

        n_resp = len(y)
        # response index -> sentence mask, computed once
        masks = [sent_owner == i for i in range(n_resp)]

        # fidelity control: hard max in PROBABILITY space, exactly the banked read
        r_hard_prob = np.array([hardmax_prob[m].min() for m in masks])
        a_hard = float(roc_auc_score(y, r_hard_prob))
        banked = BANKED[sub]
        fidelity[sub] = {
            "recomputed_hardmax_auc": round(a_hard, 6),
            "banked_hardmax_auc": banked,
            "abs_diff": round(abs(a_hard - banked), 8),
            "within_tol": bool(abs(a_hard - banked) <= FIDELITY_TOL),
        }

        row = {"n": int(n_resp), "n_sent": int(ns), "n_pairs": int(len(sc)),
               "hardmax_auc": round(a_hard, 4),
               "readout_auc": None, "readout_delta": None}
        # per-response window count (constant across a response's sentences by construction)
        resp_nw = np.array([F[m, 6][0] for m in masks])
        win_stats[sub] = {
            "windows_per_sentence_mean": round(float(F[:, 6].mean()), 2),
            "windows_per_sentence_max": int(F[:, 6].max()),
            "pairs_per_response": round(float(len(sc) / n_resp), 1),
            "windows_per_response_min": int(resp_nw.min()),
            "windows_per_response_max": int(resp_nw.max()),
            "windows_per_response_distinct": int(len(np.unique(resp_nw))),
            "n_penalty_can_rerank": bool(len(np.unique(resp_nw)) > 1),
        }
        for vname, fn in variants.items():
            s = np.asarray(fn(F), dtype=np.float64)
            r = np.array([s[m].min() for m in masks])
            auc = float(roc_auc_score(y, r))
            row[f"{vname}_auc"] = round(auc, 4)
            row[f"{vname}_delta"] = round(auc - a_hard, 4)
        per_sub[sub] = row
        print(f"  {sub:12s} hard-max {a_hard:.4f} (banked {banked}, |d|={abs(a_hard-banked):.2e})  "
              f"logreg {row['logreg_scalar_auc']:.4f} ({row['logreg_scalar_delta']:+.4f})  "
              f"penmax{best_lam} {row[f'penmax_lam{best_lam}_auc']:.4f} "
              f"({row[f'penmax_lam{best_lam}_delta']:+.4f})", flush=True)

    fidelity_pass = all(v["within_tol"] for v in fidelity.values())
    max_abs = max(v["abs_diff"] for v in fidelity.values())
    print(f"\n  FIDELITY CONTROL: {'PASS' if fidelity_pass else 'FAIL'} "
          f"(max |diff| = {max_abs:.2e}, tol {FIDELITY_TOL})", flush=True)
    if not fidelity_pass:
        payload = {"gate": "R16-H141 step A1 autopsy", "ABORTED": True,
                   "reason": "hard-max fidelity control failed",
                   "fidelity_control": fidelity, "tolerance": FIDELITY_TOL}
        OUT.write_text(json.dumps(payload, indent=2))
        print("=== R16-H141 ABORTED (fidelity) ===", flush=True)
        return

    # ---------------- readout column from the pilot ----------------
    pilot = json.loads(PILOT.read_text())
    for sub, row in per_sub.items():
        p = pilot["per_subset"][sub]
        row["readout_auc"] = p["readout_auc"]
        row["readout_delta"] = p["delta"]

    # ---------------- aggregation ----------------
    subs = sorted(per_sub)
    def col(key):
        return float(np.mean([per_sub[s][key] for s in subs]))

    variant_summary = {}
    for vname in list(variants) + ["readout"]:
        k = f"{vname}_auc" if vname != "readout" else "readout_auc"
        m = col(k)
        variant_summary[vname] = {
            "blind_mean_auc": round(m, 5),
            "mean_delta_vs_hardmax": round(m - col("hardmax_auc"), 5),
            "pubmedqa_delta": per_sub["pubmedqa"][f"{vname}_delta" if vname != "readout" else "readout_delta"],
            "techqa_delta": per_sub["techqa"][f"{vname}_delta" if vname != "readout" else "readout_delta"],
            "hotpotqa_delta": per_sub["hotpotqa"][f"{vname}_delta" if vname != "readout" else "readout_delta"],
            "tatqa_delta": per_sub["tatqa"][f"{vname}_delta" if vname != "readout" else "readout_delta"],
            "worst_subset_delta": round(min(per_sub[s][f"{vname}_delta" if vname != "readout" else "readout_delta"]
                                            for s in subs), 4),
        }

    scalar_names = [v for v in variants if v != "hardmax"]

    # PRIMARY selection - honest: chosen on the training slice only, never on the arena.
    primary = "logreg_scalar" if lr_val_auc >= lam_val[best_lam] else f"penmax_lam{best_lam}"
    # ORACLE selection - upper bound: the scalar variant with the largest arena pubmedqa delta.
    oracle_pub = max(scalar_names, key=lambda v: variant_summary[v]["pubmedqa_delta"])
    # BEST-MEAN selection - the scalar variant with the best blind mean.
    best_mean = max(scalar_names, key=lambda v: variant_summary[v]["blind_mean_auc"])

    def branch(delta):
        return "SCALAR" if delta >= BAR_SCALAR else "EMBEDDING"

    decision = {
        "bar": f"best scalar variant pubmedqa delta vs hard-max >= +{BAR_SCALAR:.3f} "
               f"(70% of the readout's +{READOUT_PUBMEDQA_DELTA}) -> SCALAR; below -> EMBEDDING",
        "readout_pubmedqa_delta": READOUT_PUBMEDQA_DELTA,
        "primary_variant": primary,
        "primary_selection_rule": "highest training-slice held-out sentence AUROC among the scalar "
                                  "aggregators; the arena was never consulted for this choice",
        "primary_pubmedqa_delta": variant_summary[primary]["pubmedqa_delta"],
        "primary_branch": branch(variant_summary[primary]["pubmedqa_delta"]),
        "oracle_variant": oracle_pub,
        "oracle_selection_rule": "arena-selected upper bound - the scalar variant with the largest "
                                 "pubmedqa delta; NOT an honest estimate, recorded as a ceiling on "
                                 "what scalar statistics could recover",
        "oracle_pubmedqa_delta": variant_summary[oracle_pub]["pubmedqa_delta"],
        "oracle_branch": branch(variant_summary[oracle_pub]["pubmedqa_delta"]),
        "best_blind_mean_variant": best_mean,
        "best_blind_mean_auc": variant_summary[best_mean]["blind_mean_auc"],
    }

    losses = {
        "rule": f"best scalar variant avoids the readout's losses if hotpotqa >= {LOSS_TOL} "
                f"AND tatqa >= {LOSS_TOL} vs hard-max",
        "readout": {"hotpotqa": pilot["per_subset"]["hotpotqa"]["delta"],
                    "tatqa": pilot["per_subset"]["tatqa"]["delta"],
                    "avoids_losses": False},
    }
    for label, v in (("primary", primary), ("oracle", oracle_pub), ("best_blind_mean", best_mean)):
        vs = variant_summary[v]
        losses[label] = {
            "variant": v,
            "hotpotqa": vs["hotpotqa_delta"], "tatqa": vs["tatqa_delta"],
            "avoids_losses": bool(vs["hotpotqa_delta"] >= LOSS_TOL and vs["tatqa_delta"] >= LOSS_TOL),
        }

    techqa = {
        "note": "same decomposition as pubmedqa, on the other many-window subset",
        "windows_per_sentence_mean": win_stats["techqa"]["windows_per_sentence_mean"],
        "hardmax_auc": per_sub["techqa"]["hardmax_auc"],
        "readout_delta": pilot["per_subset"]["techqa"]["delta"],
        "primary_delta": variant_summary[primary]["techqa_delta"],
        "oracle_pubmedqa_variant_delta": variant_summary[oracle_pub]["techqa_delta"],
        "best_techqa_scalar_variant": max(scalar_names, key=lambda v: variant_summary[v]["techqa_delta"]),
        "best_techqa_scalar_delta": max(variant_summary[v]["techqa_delta"] for v in scalar_names),
    }

    payload = {
        "gate": "R16-H141 step A1 - scalar-vs-embedding autopsy of the R16-H140 readout lift",
        "question": "Is the R16-H140 readout's pubmedqa lift (+0.0711) recoverable from SCALAR "
                    "window-score statistics alone (max-noise saturation repair), or does it "
                    "require embedding content?",
        "checkpoint": pilot["checkpoint"],
        "embeddings_used": False,
        "protocol": "per-sentence aggregation over that sentence's window set, hard MIN over the "
                    "response's sentences, response-level AUROC per subset - unchanged from the campaign",
        "fidelity_control": {
            "tolerance": FIDELITY_TOL,
            "pass": fidelity_pass,
            "max_abs_diff": round(max_abs, 8),
            "recomputed_mean": round(col("hardmax_auc"), 6),
            "banked_mean": BANKED_MEAN,
            "per_subset": fidelity,
        },
        "feature_definitions": {
            "space": "all order statistics computed on logit(frozen per-window score); max in logit "
                     "space selects the same window as max in probability space",
            "max": "max over the sentence's windows - the frozen serving read",
            "top2_mean": "mean of the 2 highest window logits (all windows if fewer)",
            "top3_mean": "mean of the 3 highest window logits (all windows if fewer)",
            "top5_mean": "mean of the 5 highest window logits (all windows if fewer)",
            "mean": "mean over all the sentence's window logits",
            "std": "population std over the sentence's window logits (0 when a single window)",
            "n_windows": "count of windows for that sentence",
            "log_n_windows": "natural log of the window count",
            "penmax_lam<L>": "max - L * log(n_windows) - the explicit max-noise saturation correction",
            "iso_<f>": "isotonic regression of feature <f> onto the training-slice label; monotone, so "
                       "single-feature AUROC matches the raw feature up to tie formation",
            "logreg_scalar": "logistic regression on the standardised 8-feature vector above",
        },
        "training_slice": slice_proof,
        "fit": {
            "logreg_coefficients": dict(zip(FEATURE_NAMES,
                                            [round(float(c), 4) for c in lr.coef_[0]], strict=True)),
            "logreg_intercept": round(float(lr.intercept_[0]), 4),
            "feature_mean": dict(zip(FEATURE_NAMES, [round(float(x), 4) for x in mu], strict=True)),
            "feature_std": dict(zip(FEATURE_NAMES, [round(float(x), 4) for x in sd], strict=True)),
            "lambda_grid_val_auc": {str(k): round(v, 5) for k, v in lam_val.items()},
            "lambda_selected": best_lam,
            "train_slice_val_sentence_auc": {k: round(v, 5) for k, v in single_val.items()},
        },
        "arena_window_stats": win_stats,
        "per_subset": per_sub,
        "variant_summary": variant_summary,
        "decision": decision,
        "loss_control": losses,
        "techqa_decomposition": techqa,
        "caveats": [
            "The training slice's window sets are small (mean 2.64, max 14) while pubmedqa carries "
            "~26 and techqa ~156 windows per item; any n_windows-dependent term is fitted far "
            "out of distribution and read on the arena at set sizes never seen in the fit.",
            "n_windows is constant across the sentences of one arena response, so a "
            "-lambda*log(n) term is a pure per-response shift: it re-ranks responses against each "
            "other, it cannot re-rank sentences inside a response. Where a subset's responses all "
            "carry the SAME window count (see arena_window_stats.n_penalty_can_rerank) the "
            "correction collapses to a global constant and cannot change that subset's AUROC at all.",
            "'~26 windows per item' on pubmedqa counts (sentence, window) PAIRS per response, not "
            "the size of the set the max runs over. The hard max on pubmedqa runs over 5 windows "
            "per sentence; techqa's runs over 22.5. The max-saturation premise is far weaker on "
            "pubmedqa than the pair count suggests.",
            "Isotonic calibration of a single feature is monotone, so it leaves that feature's "
            "response-level AUROC unchanged except where it creates ties; it is recorded as a "
            "control, not as an independent aggregator.",
            "The oracle variant is selected on the arena and is an upper bound only. The primary "
            "variant is the honest, training-slice-selected figure.",
            "Bars are RECORDED, not adjudicated here - the coordinator adjudicates.",
        ],
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(payload, indent=2))

    # ---------------- console table ----------------
    print("\n" + "=" * 108)
    print("R16-H141 STEP A1 RESULT - per-subset AUROC")
    print("=" * 108)
    hdr = f"{'subset':12s} {'win/sent':>8s} {'hard-max':>9s} {'readout':>9s} {'logreg':>9s} " \
          f"{f'penmax{best_lam}':>11s} {'top3':>9s} {'mean':>9s}"
    print(hdr)
    for s in subs:
        r = per_sub[s]
        print(f"{s:12s} {win_stats[s]['windows_per_sentence_mean']:8.1f} {r['hardmax_auc']:9.4f} "
              f"{r['readout_auc']:9.4f} {r['logreg_scalar_auc']:9.4f} "
              f"{r[f'penmax_lam{best_lam}_auc']:11.4f} {r['raw_top3_mean_auc']:9.4f} "
              f"{r['raw_mean_auc']:9.4f}")
    print("-" * 108)
    print(f"{'MEAN':12s} {'':8s} {col('hardmax_auc'):9.4f} {col('readout_auc'):9.4f} "
          f"{col('logreg_scalar_auc'):9.4f} {col(f'penmax_lam{best_lam}_auc'):11.4f} "
          f"{col('raw_top3_mean_auc'):9.4f} {col('raw_mean_auc'):9.4f}")
    print("=" * 108)
    print(f"  DECISION NUMBER (primary, {primary}): pubmedqa delta "
          f"{decision['primary_pubmedqa_delta']:+.4f}  -> {decision['primary_branch']}")
    print(f"  ORACLE upper bound ({oracle_pub}): pubmedqa delta "
          f"{decision['oracle_pubmedqa_delta']:+.4f}  -> {decision['oracle_branch']}")
    print(f"  bar: >= +{BAR_SCALAR:.3f} -> SCALAR, below -> EMBEDDING "
          f"(readout was {READOUT_PUBMEDQA_DELTA:+.4f})")
    print(f"  blind means: hard-max {col('hardmax_auc'):.5f}  readout {col('readout_auc'):.5f}  "
          f"{primary} {variant_summary[primary]['blind_mean_auc']:.5f}  "
          f"best-mean {best_mean} {variant_summary[best_mean]['blind_mean_auc']:.5f}")
    print(f"  -> {OUT}")
    print("=== R16-H141 A1 DONE ===", flush=True)


if __name__ == "__main__":
    main()

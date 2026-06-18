"""Meta-classifier over per-model grounding scores (no fine-tuning).

Learns a decision hyperplane that combines the cached per-model scores (plus a
lexical numeric/entity contradiction flag) into one grounded-probability, scored
out-of-fold on the verified gold. Tests whether the combination beats the best
single signal (bge-reranker) on macro-F1 - the driving metric - and on the count
of false positives (supported flagged) and false negatives (hallucination missed),
the two errors the gate must reduce.

Features are the 6 score arrays in data/interim/model_scores/*.npy, index-aligned
to data/processed/golden_grounding_evidence_verified.parquet.

Run:  python grounding_ensemble.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from grounding_models import GOLD as GOLD_JSON
from grounding_models import SCORES_DIR

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "grounding_ensemble.md"

# feature name -> cached score file stem
MODELS = {
    "bge_reranker": "BAAI__bge-reranker-v2-m3",
    "mdeberta_nli": "MoritzLaurer__mDeBERTa-v3-base-mnli-xnli",
    "bge_m3": "BAAI__bge-m3",
    "e5_large": "intfloat__multilingual-e5-large",
    "e5_small": "intfloat__multilingual-e5-small",
    "mmbert": "jhu-clsp__mmBERT-base",
}


FEAT_CACHE = Path(__file__).resolve().parent / "private-rag-forensics" / "ensemble_features.npz"


def load_features():
    """X (n, d), labels (1=supported), feature names, langs - all aligned (cached)."""
    if FEAT_CACHE.exists():
        d = np.load(FEAT_CACHE, allow_pickle=True)
        return d["X"], d["y"], list(d["names"]), list(d["langs"])
    import polars as pl
    recs = pl.read_parquet(GOLD_JSON).to_dicts()
    labels = np.array([int(r["label"]) for r in recs])
    langs = [r.get("lang", "en") for r in recs]
    cols, names = [], []
    for name, stem in MODELS.items():
        cols.append(np.load(SCORES_DIR / f"{stem}.npy"))
        names.append(name)
    # lexical contradiction + fired flags (cheap, catch spec/number hallucinations)
    from groundrails.grounding import ground
    contra = np.zeros(len(recs))
    fired = np.zeros(len(recs))
    for i, r in enumerate(recs):
        m = ground(r["claim"], [("s", r["source_text"])])
        contra[i] = 1.0 if (m.numeric_mismatches or m.entity_mismatches) else 0.0
        fired[i] = 1.0 if m.match_type in ("exact", "fuzzy", "bm25") else 0.0
    cols += [contra, fired]
    names += ["contradiction", "lexical_fired"]
    X = np.column_stack(cols)
    np.savez(FEAT_CACHE, X=X, y=labels, names=np.array(names), langs=np.array(langs))
    return X, labels, names, langs


def _f1c(tp, fp, fn):
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def macro_at(p, y, T):
    """Macro-F1 (mean of hallucination-class and supported-class F1) at threshold T.

    Predict hallucination when grounded-prob p < T. Returns (macro_f1, f1_hall,
    f1_supported, recall_hall, false_flag, acc).
    """
    flag = p < T
    hall = y == 0
    tp = int((flag & hall).sum())   # hallucination caught
    fn = int((~flag & hall).sum())
    fp = int((flag & ~hall).sum())  # supported wrongly flagged
    tn = int((~flag & ~hall).sum())
    f1_h = _f1c(tp, fp, fn)
    f1_s = _f1c(tn, fn, fp)         # supported class: tp=tn, fp=fn, fn=fp
    macro = 0.5 * (f1_h + f1_s)
    rec = tp / (tp + fn) if tp + fn else 0.0
    ff = fp / (fp + tn) if fp + tn else 0.0
    acc = (tp + tn) / len(y)
    return macro, f1_h, f1_s, rec, ff, acc


def best_macro(p, y):
    """Threshold that maximizes macro-F1, scanning the score range."""
    lo, hi = float(np.quantile(p, 0.02)), float(np.quantile(p, 0.98))
    best = None
    for T in np.linspace(lo, hi, 97):
        m = macro_at(p, y, T)
        if best is None or m[0] > best[0]:
            best = (m[0], float(T), m)
    return best  # (macro_f1, T, full tuple)


def counts_at(p, y, T):
    """Error counts at threshold T (flag hallucination when p < T).

    Returns (fp, fn, tp, tn): fp = supported wrongly flagged, fn = hallucination
    missed, tp = hallucination caught, tn = supported kept.
    """
    flag = p < T
    hall = y == 0
    tp = int((flag & hall).sum())
    fn = int((~flag & hall).sum())
    fp = int((flag & ~hall).sum())
    tn = int((~flag & ~hall).sum())
    return fp, fn, tp, tn


def main() -> None:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, names, langs = load_features()
    langs = np.array(langs)
    n_pos = int((y == 0).sum())  # hallucinations
    skf = StratifiedKFold(5, shuffle=True, random_state=42)

    # single-signal AUCs (pretrained, not fit on labels -> honest as-is)
    single = {}
    for j, name in enumerate(names[:6]):
        single[name] = roc_auc_score(y, X[:, j])

    # meta-classifiers, out-of-fold P(supported)
    clfs = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "gbm": GradientBoostingClassifier(random_state=42, n_estimators=200, max_depth=2),
    }
    oof, ens_auc, ens_std = {}, {}, {}
    for name, clf in clfs.items():
        p = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
        oof[name] = p
        fold = []
        for tr, te in skf.split(X, y):
            clf.fit(X[tr], y[tr])
            fold.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        ens_auc[name] = float(np.mean(fold))
        ens_std[name] = float(np.std(fold))

    best_clf = max(ens_auc, key=ens_auc.get)
    p = oof[best_clf]
    # driving metric: macro-F1 (mean of hallucination- and supported-class F1)
    macro_f1, T_macro, _ = best_macro(p, y)
    bge_idx = names.index("bge_reranker")
    single_macro, single_T, _ = best_macro(X[:, bge_idx], y)
    # error counts at each signal's own macro-F1-optimal threshold
    fp_s, fn_s, _, _ = counts_at(X[:, bge_idx], y, single_T)
    fp_m, fn_m, _, _ = counts_at(p, y, T_macro)

    # learned weights (logreg on full data, scaled)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)
    coef = lr.named_steps["logisticregression"].coef_[0]
    weights = sorted(zip(names, coef), key=lambda t: -abs(t[1]))

    # per-language AUC (best classifier OOF) where n is enough
    from collections import Counter
    lang_n = Counter(langs.tolist())
    lang_auc = {}
    for lg, n in lang_n.items():
        idx = langs == lg
        if len(set(y[idx].tolist())) > 1 and n >= 20:
            lang_auc[lg] = (roc_auc_score(y[idx], p[idx]), int(n))

    # error analysis at the macro-F1-optimal (driving-metric) threshold
    Tcut = T_macro
    flagged = p < Tcut
    fp = (flagged & (y == 1))    # supported wrongly flagged
    fn = (~flagged & (y == 0))   # hallucination missed
    import polars as pl
    recs = pl.read_parquet(GOLD_JSON).to_dicts()

    def has_num(s):
        import re
        return bool(re.search(r"\d", s))
    miss_num = sum(1 for i in np.where(fn)[0] if has_num(recs[i]["claim"]))

    L = ["# Grounding Ensemble - meta-classifier over model scores", "",
         f"Verified gold: {len(y)} claims ({n_pos} hallucination / {int((y==1).sum())} supported). "
         "Features = 6 per-model scores + lexical contradiction/fired flags. Out-of-fold 5-fold CV; "
         "single-signal AUCs are pretrained (not fit on labels), so honest as-is.", "",
         "## Single-signal AUC (best to worst)"]
    for name, a in sorted(single.items(), key=lambda t: -t[1]):
        L.append(f"- {name}: {a:.3f}")
    L += ["", "## Meta-classifier (out-of-fold)"]
    for name in clfs:
        L.append(f"- {name}: AUC {ens_auc[name]:.3f} +/- {ens_std[name]:.3f}")
    best_single = max(single.values())
    L += ["",
          f"Best single = bge_reranker {single['bge_reranker']:.3f}; "
          f"best meta = {best_clf} {ens_auc[best_clf]:.3f} +/- {ens_std[best_clf]:.3f} "
          f"({'beats' if ens_auc[best_clf] > best_single else 'does not beat'} the best single signal).",
          "", "## Driving metric - macro-F1 and error counts (out-of-fold)", "",
          "**Macro-F1** (mean of the hallucination-class and supported-class F1, both classes "
          "weighted equally) is the metric. The target is to raise it by cutting the two error "
          "counts: **FP** = supported claims wrongly flagged, **FN** = hallucinations missed. "
          f"Each signal at its own macro-F1-optimal threshold over n={len(y)} "
          f"(hallucination base rate {n_pos}/{len(y)} = {n_pos/len(y):.0%}).", "",
          "| signal | threshold | **macro-F1** | FP | FN | FP+FN | accuracy |",
          "|---|---|---|---|---|---|---|"]
    macc_s = macro_at(X[:, bge_idx], y, single_T)[5]
    macc_m = macro_at(p, y, T_macro)[5]
    L += [f"| bge-reranker (best single) | {single_T:.2f} | **{single_macro:.2f}** | "
          f"{fp_s} | {fn_s} | {fp_s+fn_s} | {macc_s:.0%} |",
          f"| meta-classifier ({best_clf}) | {T_macro:.2f} | **{macro_f1:.2f}** | "
          f"{fp_m} | {fn_m} | {fp_m+fn_m} | {macc_m:.0%} |"]
    d_err = (fp_s + fn_s) - (fp_m + fn_m)
    base_macro = 0.5 * (2 * (len(y)-n_pos)/len(y) / (1 + (len(y)-n_pos)/len(y)))
    L += ["",
          f"The meta-classifier lifts macro-F1 **{single_macro:.3f} -> {macro_f1:.3f}** and cuts "
          f"total errors **{fp_s+fn_s} -> {fp_m+fn_m}** "
          f"(-{d_err}, -{d_err/(fp_s+fn_s):.0%}): FP {fp_s}->{fp_m}, FN {fn_s}->{fn_m}. "
          f"Majority-class baseline macro-F1 = {base_macro:.3f} "
          "(always-predict-supported -> hallucination-class F1 = 0)."]
    L += ["", "## Learned feature weights (logreg, + = predicts supported)"]
    for name, w in weights:
        L.append(f"- {name}: {w:+.2f}")
    if lang_auc:
        L += ["", "## Per-language AUC (n>=20)"]
        for lg, (a, n) in sorted(lang_auc.items(), key=lambda t: -t[1][0]):
            L.append(f"- {lg} (n={n}): {a:.3f}")
    L += ["", "## Error analysis (at the macro-F1-optimal threshold)",
          f"- FN - missed hallucinations: {int(fn.sum())}, of which {miss_num} contain a number/spec",
          f"- FP - wrongly-flagged supported: {int(fp.sum())}",
          "- the missed hallucinations are the overlap region a fine-tuned cross-encoder would target"]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()

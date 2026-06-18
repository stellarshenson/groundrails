"""Cheap resampling probe: does class / language over-under-sampling in the joint-head
training lift gold v3 macro-F1, especially the non-EN slice?

Exploratory follow-up to R1-H4 (a plain OOF retrain gave no lift). The non-EN macro is
depressed by a tiny native hallucination class; resampling the training folds toward that
class (or toward non-English rows) is the cheapest thing to try before harder negatives.
Reuses the cached gold v3 signals - logistic fits are instant - so this is a fast sweep,
not a pre-registered hypothesis.

Run:  python experiments/grounding-semantic/resample_probe.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from joint_xlingual import JF, best_threshold, load, slice_metrics  # noqa: E402


def resample(idx, y, lang, mode, rng):
    """Return a resampled training-index array for the given mode."""
    yy, ll = y[idx], lang[idx]
    neg, pos = idx[yy == 0], idx[yy == 1]
    if mode == "oversample_minority":  # oversample hallucinations to match supported
        n = max(len(neg), len(pos))
        return np.concatenate([rng.choice(neg, n, replace=True), rng.choice(pos, n, replace=True)])
    if mode == "undersample_majority":  # downsample supported to match hallucinations
        n = min(len(neg), len(pos))
        return np.concatenate([rng.choice(neg, n, replace=False), rng.choice(pos, n, replace=False)])
    if mode == "oversample_nonen":  # oversample all non-English rows to the English count
        en, ne = idx[ll == "en"], idx[ll != "en"]
        if len(ne) == 0:
            return idx
        return np.concatenate([en, rng.choice(ne, len(en), replace=True)])
    if mode == "oversample_nonen_neg":  # oversample only the non-English hallucinations
        nn = idx[(ll != "en") & (yy == 0)]
        if len(nn) == 0:
            return idx
        return np.concatenate([idx, rng.choice(nn, len(nn) * 4, replace=True)])
    raise ValueError(mode)


def oof(df, feat, train_roles, mode, k=5, seed=0):
    X, y = df[feat].to_numpy(float), df["label"].to_numpy(int)
    groups, role, lang = df["group_id"].to_numpy(), df["role"].to_numpy(), df["lang_norm"].to_numpy()
    rng = np.random.default_rng(seed)
    p = np.full(len(df), np.nan)
    cw = "balanced" if mode == "balanced" else None
    for tr, te in GroupKFold(k).split(X, y, groups):
        tri = tr[np.isin(role[tr], train_roles)]
        if mode not in ("none", "balanced"):
            tri = resample(tri, y, lang, mode, rng)
        sc = StandardScaler().fit(X[tri])
        lr = LogisticRegression(max_iter=2000, C=1.0, class_weight=cw).fit(sc.transform(X[tri]), y[tri])
        p[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def main() -> None:
    df = load()
    ev = df["role"].eq("eval")
    en = df["lang_norm"].eq("en")
    nonen = ev & ~en
    y = df["label"].to_numpy(int)
    modes = [
        "none",
        "balanced",
        "oversample_minority",
        "undersample_majority",
        "oversample_nonen",
        "oversample_nonen_neg",
    ]
    print(f"{'mode':24} {'macro':>6} {'EN':>6} {'nonEN':>7} {'nEN-TNR':>8} {'nEN-rec':>8}  T")
    for mode in modes:
        p = oof(df, JF, ("eval",), mode)
        m, T = best_threshold(y[ev], p[ev.to_numpy()])
        en_m = slice_metrics(df, p, T, en & ev)["macro"]
        ne = slice_metrics(df, p, T, nonen)
        print(
            f"{mode:24} {m:6.3f} {en_m:6.3f} {ne['macro']:7.3f} "
            f"{ne.get('tnr', float('nan')):8.3f} {ne.get('sup_recall', float('nan')):8.3f}  {T:.2f}"
        )


if __name__ == "__main__":
    main()

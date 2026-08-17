"""FLOOR CHECK - is the sub-bar reading at 1% retention a real result or a
degenerate probe?  CPU ONLY.

`length_matched_then_peel` at retention 0.010 (400 rows / 200 pairs) is the only
point in the entire sweep that reads under the 0.55 claim-only bar (0.4995,
within-pair 0.4900).  Both numbers sit on chance, which is also what a probe with
an empty vocabulary returns.  This script separates the two readings: it reports
the per-fold training vocabulary, the fraction of rows that received a non-zero
decision value, the tie fraction, and the seed-to-seed spread of the AUROC.

Run: uv run python experiments/grounding-semantic/contract/halueval_conform_floorcheck.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import random

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import importlib.util
import pathlib

HERE = pathlib.Path(__file__).parent
EXP = HERE.parent


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


HCF = _mod("halueval_conform", HERE / "halueval_conform.py")
C = HCF.C
NOTE = HCF.NOTE

LEVELS = [("length_matched_then_peel", 0.01), ("length_matched_then_peel", 0.02),
          ("peel_probe_margin", 0.02), ("peel_probe_margin", 0.10),
          ("claim_similarity_then_peel", 0.01)]


def instrumented_probe(claims, labels, groups, seed, n_folds=5):
    rng = random.Random(seed)
    keys = sorted(set(groups))
    rng.shuffle(keys)
    fold_of = {k: i % n_folds for i, k in enumerate(keys)}
    folds = np.array([fold_of[g] for g in groups])
    score = np.zeros(len(claims))
    idx = np.arange(len(claims))
    vocab = []
    for f in range(n_folds):
        tr, te = idx[folds != f], idx[folds == f]
        if not te.size or not tr.size:
            continue
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                              max_features=300_000, sublinear_tf=True)
        xtr = vec.fit_transform([claims[j] for j in tr])
        xte = vec.transform([claims[j] for j in te])
        clf = LogisticRegression(solver="liblinear", C=4.0, tol=1e-7, max_iter=3000)
        clf.fit(xtr, [labels[j] for j in tr])
        score[te] = clf.decision_function(xte)
        vocab.append({"fold": f, "train_rows": int(tr.size),
                      "vocabulary_terms": int(len(vec.vocabulary_)),
                      "test_rows_with_zero_features": int(
                          (xte.getnnz(axis=1) == 0).sum())})
    return float(C.auroc(labels, score)), score, vocab


def main():
    d, df = HCF.load()
    z = np.load(HERE / "halueval_conform_scores.npz", allow_pickle=True)
    R, _ = HCF.rankings(df, z)
    out = {"note": NOTE,
           "what": "is the only sub-bar point in the sweep a real result or an "
                   "instrument that has run out of data",
           "levels": {}}
    for strategy, r in LEVELS:
        n = int(round(r * HCF.ORIG_PAIRS))
        keep = R[strategy]["order"][:n]
        sub = df.filter(pl.col("pair_id").is_in(list(keep)))
        claims = sub["claim"].to_list()
        labels = sub["label"].to_numpy()
        g = HCF.groups_of(sub)
        aurocs, vocabs, ties, nonzero = [], None, None, None
        for seed in range(5):
            a, s, v = instrumented_probe(claims, labels, g, seed)
            aurocs.append(round(a, 4))
            if seed == 0:
                vocabs = v
                ties = round(float((s == 0).mean()), 4)
                nonzero = int((s != 0).sum())
        key = f"{strategy}@{r:.3f}"
        out["levels"][key] = {
            "rows": sub.height, "pairs": int(sub["pair_id"].n_unique()),
            "retention_of_member_rows": round(sub.height / HCF.ORIG_ROWS, 4),
            "auroc_over_5_seeds": aurocs,
            "auroc_mean": round(float(np.mean(aurocs)), 4),
            "auroc_min": min(aurocs), "auroc_max": max(aurocs),
            "auroc_spread": round(max(aurocs) - min(aurocs), 4),
            "seeds_clearing_the_0.55_bar": int(sum(a < 0.55 for a in aurocs)),
            "per_fold_vocabulary_seed0": vocabs,
            "rows_with_exactly_zero_decision_value_seed0": nonzero and sub.height - nonzero,
            "zero_score_fraction_seed0": ties,
        }
        print(f"{key}: rows {sub.height} aurocs {aurocs} "
              f"vocab {[v['vocabulary_terms'] for v in vocabs]}", flush=True)
        (HERE / "halueval_conform_floorcheck.json").write_text(json.dumps(out, indent=2))
    (HERE / "halueval_conform_floorcheck.json").write_text(json.dumps(out, indent=2))
    print("floorcheck written", flush=True)


if __name__ == "__main__":
    main()

"""C5 supplementary probes for `ragtruth_translated`. CPU ONLY.

C5 is NOT-APPLICABLE to this member - it binds constructed lanes and
paired-contrast evals, and this is a source corpus with no construction, no
pairing and no families. These probes are therefore EXECUTOR-ADDED and
NON-GATING, reported separately from the registered clause exactly as C5's own
separation rule requires. They answer the question C5 would ask if it applied:
can either leg be read off one channel?

  claim-only    hashed claim tokens -> label, converged liblinear (never default
                lbfgs, per the H144 finding), 5-fold grouped by source document
                so the ~6 responses of one document never straddle the fold
  evidence-only the same on the evidence text alone
  claim length  AUROC of claim character length against the label - the surface
                parity channel

Run:  uv run python experiments/grounding-semantic/contract/ragtruth_translated_c5_supp.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).parent
DATA = HERE.parent.parent.parent / "data" / "external" / "datasets"
MEMBER = HERE / "ragtruth_translated_member.parquet"
OUT = HERE / "ragtruth_translated_c5_supp.json"

LANGS = ("de", "fr", "es", "it", "pl", "hu", "cn")


def auroc(y, s):
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    rsum = np.zeros(len(uniq))
    np.add.at(rsum, inv, ranks)
    ranks = (rsum / cnt)[inv]
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def probe(texts, y, groups):
    vec = HashingVectorizer(
        n_features=2**18, alternate_sign=False, lowercase=True, analyzer="char_wb",
        ngram_range=(3, 5),
    )
    X = vec.transform(texts)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = LogisticRegression(
            solver="liblinear", C=1.0, tol=1e-7, max_iter=5000
        ).fit(X[tr], y[tr])
        oof[te] = clf.decision_function(X[te])
    return round(auroc(y, oof), 4)


def main():
    mem = pl.read_parquet(MEMBER)
    ze = zipfile.ZipFile(DATA / "dataset-ragtruth.zip")
    en = pl.read_parquet(
        io.BytesIO(ze.read(next(x for x in ze.namelist() if x.endswith("__train.parquet"))))
    )
    ctx = en["context"].to_list()
    ids = {c: i for i, c in enumerate(dict.fromkeys(ctx))}
    doc = np.array([ids[c] for c in ctx])  # row-aligned across every language

    res = {
        "status": "EXECUTOR-ADDED, NON-GATING - C5 itself is NOT-APPLICABLE to a "
        "source corpus with no construction. These probes do not join any "
        "registered conjunction.",
        "folds": "GroupKFold(5) on the source document, so the ~6 responses of one "
        "document never straddle a fold",
        "per_language": {},
    }
    for lg in LANGS:
        sub = mem.filter(pl.col("lang") == lg)
        y = sub["label"].to_numpy().astype(int)
        cl = sub["claim"].to_list()
        ev = sub["chunk"].to_list()
        block = {
            "rows": sub.height,
            "claim_only_auroc": probe(cl, y, doc),
            "evidence_only_auroc": probe([e[:4000] for e in ev], y, doc),
            "claim_length_auroc": round(auroc(y, np.array([len(c) for c in cl])), 4),
            "evidence_length_auroc": round(auroc(y, np.array([len(e) for e in ev])), 4),
        }
        res["per_language"][lg] = block
        print(lg, json.dumps(block), flush=True)
        OUT.write_text(json.dumps(res, indent=2))

    vals = res["per_language"]
    res["pooled_summary"] = {
        "claim_only_auroc_max": max(v["claim_only_auroc"] for v in vals.values()),
        "claim_only_auroc_mean": round(
            float(np.mean([v["claim_only_auroc"] for v in vals.values()])), 4
        ),
        "evidence_only_auroc_max": max(v["evidence_only_auroc"] for v in vals.values()),
        "claim_length_auroc_max_abs_dev_from_0.5": round(
            max(abs(v["claim_length_auroc"] - 0.5) for v in vals.values()), 4
        ),
        "reference_bars_if_C5_applied": {
            "claim_only": "< 0.55",
            "surface_parity": "0.45-0.55",
        },
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["pooled_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

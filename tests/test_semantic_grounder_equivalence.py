"""Equivalence gate for the imported semantic grounder leg.

Proves the ported `experiments/grounding-semantic/grounding_ensemble.py` reproduces
the parent datascience project's headline numbers byte-for-byte. The computation is
deterministic (cached per-model scores in `ensemble_features.npz`, fixed
`random_state=42`), so the meta-classifier's macro-F1 / AUC / error counts must match
the values captured from the parent.

Skips when the gitignored private gold caches are absent (e.g. CI), since the leg's
`private-rag-forensics/` data never leaves the local machine.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

LEG = Path(__file__).resolve().parents[1] / "experiments" / "grounding-semantic"
sys.path.insert(0, str(LEG))

ge = pytest.importorskip("grounding_ensemble")

pytestmark = pytest.mark.skipif(
    not ge.FEAT_CACHE.exists(),
    reason="private-rag-forensics ensemble_features.npz absent (gitignored data, e.g. CI)",
)

# Parent baseline captured from `python -m dbm_improvements.grounding_ensemble`
# on the 2,752-record organic-majority gold (see JOURNAL.md entry 74).
PARENT = {
    "gbm_auc": 0.913,
    "single_bge_reranker_macro": 0.757,
    "meta_macro": 0.824,
    "meta_fp": 248,
    "meta_fn": 160,
}


def _gbm_oof(X, y):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    clf = GradientBoostingClassifier(random_state=42, n_estimators=200, max_depth=2)
    p = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
    fold = []
    for tr, te in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        fold.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return p, float(np.mean(fold))


def test_semantic_grounder_reproduces_parent_headline():
    X, y, names, _ = ge.load_features()
    assert len(y) == 2752, "gold record count drifted from the captured baseline"

    p, gbm_auc = _gbm_oof(X, y)
    assert gbm_auc == pytest.approx(PARENT["gbm_auc"], abs=1e-3)

    meta_macro, t_macro, _ = ge.best_macro(p, y)
    fp, fn, _, _ = ge.counts_at(p, y, t_macro)
    assert meta_macro == pytest.approx(PARENT["meta_macro"], abs=2e-3)
    assert (fp, fn) == (PARENT["meta_fp"], PARENT["meta_fn"])

    bge = X[:, names.index("bge_reranker")]
    single_macro, _, _ = ge.best_macro(bge, y)
    assert single_macro == pytest.approx(PARENT["single_bge_reranker_macro"], abs=2e-3)

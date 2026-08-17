"""Executor-added claim-only diagnostic for the `vitaminc` member - NOT a C5 bar.

C5 scopes to constructed lanes and paired-contrast evals, and `vitaminc` is a
source corpus, so its leak suite is NOT-APPLICABLE.  This measures the one
question C5's bars would otherwise have answered and that C1 makes worth asking:
can the binary label be read off the CLAIM ALONE, with no evidence?  It matters
here because 121,700 of the member's 370,653 rows are SYNTHETIC revisions, and a
synthetic edit can leave a claim-side artifact ("more than" / "less than") that
correlates with the label.

Instrument REUSED: `R20-H174_lane_common.claim_only_probe` - out-of-fold char_wb
TF-IDF (2-5) + liblinear at tol 1e-7, folds disjoint on the grouping key.  The
key is `page`, so no fold's training complement carries its own test page.

Reported SEPARATELY and merged into the report under `C1.executor_added_probes`.
It joins no registered conjunction and moves no verdict.

CPU ONLY.  Run:  uv run python \\
    experiments/grounding-semantic/contract/vitaminc_claim_probe.py \\
    2>&1 | tee logs/vitaminc_contract_claim_probe.log
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import importlib.util
import io
import json
import pathlib
import random
import time
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
DATA = SEM.parent.parent / "data" / "external" / "datasets"
REPORT = HERE / "vitaminc_contract_report.json"

SUBSAMPLE_PAGES = 4000   # document-level subsample, stated not hidden
SEED = 20260817
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


spec = importlib.util.spec_from_file_location("lc", SEM / "R20-H174_lane_common.py")
LC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LC)


def main():
    z = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    d = pl.read_parquet(io.BytesIO(z.read("tals__vitaminc__train.parquet")))
    if d.height != 370_653:
        raise SystemExit(f"MEMBER ABORT: {d.height} rows")
    d = d.with_columns((pl.col("label") == "SUPPORTS").cast(pl.Int64).alias("y"))

    rng = random.Random(SEED)
    pages = sorted(set(d["page"].to_list()))
    rng.shuffle(pages)
    keep = set(pages[:SUBSAMPLE_PAGES])
    s = d.filter(pl.col("page").is_in(list(keep)))
    log(f"subsample: {s.height} rows over {len(keep)} pages "
        f"(of {d.height} rows / {len(pages)} pages), positive rate {s['y'].mean():.4f}")

    claims = s["claim"].to_list()
    y = s["y"].to_numpy()
    groups = s["page"].to_list()
    auc, score = LC.claim_only_probe(claims, y, groups, random.Random(SEED))
    log(f"claim-only AUROC (page-disjoint folds): {auc:.4f}")

    # Within-case reading: only cases whose claims actually differ can carry a
    # claim-side signal at all - where the claim is held fixed the probe is at
    # 0.5 by construction and says nothing.
    s = s.with_columns(pl.Series("probe", score))
    per_case = []
    for (_cid,), grp in s.group_by(["case_id"], maintain_order=True):
        if grp["claim"].n_unique() < 2 or grp["y"].n_unique() < 2:
            continue
        pos = grp.filter(pl.col("y") == 1)["probe"].to_numpy()
        neg = grp.filter(pl.col("y") == 0)["probe"].to_numpy()
        wins = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
        per_case.append(float(wins))
    within = float(np.mean(per_case)) if per_case else None
    log(f"within-case claim-only accuracy over {len(per_case)} claim-varying "
        f"cases: {within}")

    res = {
        "status": ("EXECUTOR-ADDED, reported separately - C5 is NOT-APPLICABLE to "
                   "a source corpus and this is not a C5 bar. It joins no "
                   "registered conjunction and moves no clause verdict"),
        "instrument": ("R20-H174_lane_common.claim_only_probe - out-of-fold "
                       "char_wb TF-IDF (2-5) + liblinear tol 1e-7, folds disjoint "
                       "on `page`"),
        "subsample": {"pages": len(keep), "rows": s.height,
                      "of_member_rows": d.height,
                      "positive_rate": round(float(y.mean()), 4),
                      "why": "the full member is 370,653 rows; the probe is fitted "
                             "five times, so a page-disjoint document subsample is "
                             "used and its size is stated"},
        "claim_only_auroc": round(auc, 4),
        "within_case_claim_only_accuracy": round(within, 4) if within is not None else None,
        "within_case_n": len(per_case),
        "reference_bars_for_context_only": {
            "C5_claim_only_converged_probe": "< 0.55",
            "C5_within_pair_claim_only": "< 0.60",
            "note": "quoted so the numbers can be placed, NOT applied - these bars "
                    "are defined for constructed lanes",
        },
    }
    rep = json.loads(REPORT.read_text())
    rep["C1"]["executor_added_probes"] = {"claim_only": res}
    REPORT.write_text(json.dumps(rep, indent=2) + "\n")
    log(f"merged into {REPORT}")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

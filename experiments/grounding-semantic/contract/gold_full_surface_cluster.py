"""gold_full's internal structure and document clustering (C8 duplication, C3 axis).

The C6 within-surface diagnostic read a document-keyed leave-one-out lookup at
0.9345 against a 0.7118 pair-level majority baseline, which says the LABEL is
strongly clustered by source document. That is a property of the evaluation
surface, and it has one consequence worth measuring rather than asserting: the
2,752 claims are not 2,752 independent draws, so the precision of the in-domain
hold (bar >= 0.84) is not the precision of a 2,752-claim simple random sample.

MEASURED.
  claim-level document-keyed leave-one-out lookup accuracy vs the base rate
  the one-way ANOVA intraclass correlation of the LABEL across documents
  the implied design effect and effective sample size
  the width of a 95% interval on a proportion at n = 2,752 and at n_eff

LIMITATION, stated not buried: the design effect is computed on the LABEL,
because no per-claim model correctness vector is banked for gold_full - every
banked `*_goldfull_result.json` carries aggregates only. Label clustering bounds
correctness clustering only to the extent a model's errors follow the label; the
figure is an indication of the order of magnitude, not the exact inflation on a
particular arm's read.

No surface text is emitted. CPU only. HF_HUB_OFFLINE=1. Polars.

Run: uv run python experiments/grounding-semantic/contract/gold_full_surface_cluster.py \
       2>&1 | tee logs/gold_full_surface_cluster.log
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import math
import pathlib

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
GOLD_SRC = SEM / "private-rag-forensics" / "gold" / "golden_grounding_evidence_verified.parquet"
GOLD_PAIRS = SEM / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"
REPORT = HERE / "gold_full_surface_report.json"


def internal_structure():
    """C8's within-member duplication, measured on the surface's own archive."""
    tp = pl.read_parquet(GOLD_PAIRS)
    src = pl.read_parquet(GOLD_SRC)
    rows = tp.height
    d_owner_chunk = int(tp.select(["owner", "chunk"]).n_unique())
    d_claim_chunk = int(tp.select(["claim", "chunk"]).n_unique())

    oc = tp.select(["owner", "claim"]).unique().group_by("claim").len()
    dup = oc.filter(pl.col("len") > 1)
    dupset = dup["claim"].to_list()
    dup_rows = int(tp.filter(pl.col("claim").is_in(dupset)).height) if dupset else 0
    lab = (tp.filter(pl.col("claim").is_in(dupset)).select(["claim", "label"]).unique()
           .group_by("claim").len()) if dupset else None
    conflicting = int((lab["len"] > 1).sum()) if lab is not None else 0

    pref = src.with_columns(pl.col("source_text").str.slice(0, 1500).alias("p"))
    out = {
        "rows_as_loaded": rows,
        "distinct_owner_chunk_pairs": d_owner_chunk,
        "repeated_rows_within_owner": rows - d_owner_chunk,
        "fraction_rows_repeated_within_owner": round((rows - d_owner_chunk) / rows, 4),
        "distinct_claim_chunk_pairs": d_claim_chunk,
        "distinct_claim_strings": int(tp["claim"].n_unique()),
        "owners": int(tp["owner"].n_unique()),
        "claim_strings_under_more_than_one_owner": int(dup.height),
        "owners_involved_in_a_shared_claim_string": int(dup["len"].sum()) if dup.height else 0,
        "rows_on_shared_claim_strings": dup_rows,
        "shared_claim_strings_carrying_both_labels": conflicting,
        "documents": int(src["source_text"].n_unique()),
        "distinct_1500_char_document_prefixes": int(pref["p"].n_unique()),
        "reading": ("the surface is loaded as 123,579 (claim, chunk) rows but carries only "
                    f"{d_owner_chunk} distinct (owner, chunk) pairs, so {rows - d_owner_chunk} "
                    "rows are exact repeats inside a claim's own chunk list. The max-over-chunks "
                    "read is unchanged in value by a repeated chunk; the redundancy is in the "
                    "row count and the scoring cost, not in the score"),
    }
    print(json.dumps({"internal_structure": out}, indent=2), flush=True)
    return out


def main():
    struct = internal_structure()
    src = pl.read_parquet(GOLD_SRC)
    lab = src["label"].to_numpy().astype(float)
    doc = src["source_text"].to_list()

    groups = {}
    for d, l in zip(doc, lab, strict=True):
        groups.setdefault(d, []).append(l)
    sizes = np.array([len(v) for v in groups.values()], dtype=float)
    n, k = lab.size, len(groups)
    base = float(lab.mean())
    fallback = 1.0 if base >= 0.5 else 0.0

    correct = 0
    pure = 0
    for v in groups.values():
        a = np.asarray(v)
        if a.size == 1:
            correct += int(fallback == a[0])
        else:
            loo = (a.sum() - a) / (a.size - 1)
            correct += int(((loo >= 0.5).astype(float) == a).sum())
        if len(set(v)) == 1:
            pure += a.size
    loo_acc = correct / n

    # one-way ANOVA ICC on the label
    grand = lab.mean()
    means = np.array([np.mean(v) for v in groups.values()])
    ssb = float(np.sum(sizes * (means - grand) ** 2))
    ssw = float(sum(((np.asarray(v) - np.mean(v)) ** 2).sum() for v in groups.values()))
    msb = ssb / (k - 1)
    msw = ssw / (n - k)
    m0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    icc = (msb - msw) / (msb + (m0 - 1) * msw)
    icc = max(0.0, float(icc))
    mbar = n / k
    deff = 1 + (mbar - 1) * icc
    n_eff = n / deff

    def halfwidth(nn, p=0.84):
        return 1.96 * math.sqrt(p * (1 - p) / nn)

    out = {
        "diagnostic": "gold_full document clustering and its effect on the in-domain hold",
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
        "cpu_only": True,
        "claims": int(n),
        "documents": int(k),
        "mean_claims_per_document": round(mbar, 4),
        "label_base_rate": round(base, 4),
        "claims_in_label_pure_documents": int(pure),
        "fraction_claims_in_label_pure_documents": round(pure / n, 4),
        "document_keyed_leave_one_out_accuracy": round(loo_acc, 4),
        "majority_baseline": round(max(base, 1 - base), 4),
        "lift_over_majority": round(loo_acc - max(base, 1 - base), 4),
        "label_icc_across_documents": round(icc, 4),
        "design_effect": round(deff, 4),
        "effective_sample_size": round(n_eff, 1),
        "ci95_halfwidth_at_p_0.84": {
            "nominal_n_2752": round(halfwidth(n), 4),
            "effective_n": round(halfwidth(n_eff), 4),
        },
        "limitation": ("the design effect is computed on the LABEL; no per-claim model "
                       "correctness vector is banked for gold_full, so this bounds the "
                       "inflation on a model read only to the extent errors follow the label"),
    }
    print(json.dumps(out, indent=2), flush=True)

    rep = json.loads(REPORT.read_text())
    rep["internal_structure"] = struct
    rep["document_clustering"] = out
    rep["measurements"]["gold_full_effective_sample_size"] = out["effective_sample_size"]
    rep["measurements"]["gold_full_document_keyed_loo_accuracy_claim_level"] = \
        out["document_keyed_leave_one_out_accuracy"]
    REPORT.write_text(json.dumps(rep, indent=2))
    print(f"merged into {REPORT}", flush=True)
    print("=== CLUSTER DIAGNOSTIC DONE ===", flush=True)


if __name__ == "__main__":
    main()

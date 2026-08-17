"""attr_pool contract - amendment 2. Three measurements the first pass got wrong
or left open.

1. C1 THREE-WAY RECOVERY, REPAIRED. The first pass bucketed every VitaminC
   label-0 row by the label set its CLAIM STRING carries anywhere in train. That
   silently included `truth_removed` negatives, whose claim IS the SUPPORTS
   claim, so every bucket read "MIXED". Redone restricted to the
   `unsupported_claim` family and keyed on (page, claim), which is the row the
   lane actually took its negative from.

2. C6 REFINED. The first oracle scored a claim by max(label) over its mix
   occurrences. A VitaminC claim string appears against several revisions and
   often carries BOTH labels, so max() saturates to 1 on both legs and reads a
   spurious 0.5. Redone with (a) the MEAN mix label, and (b) the channel that
   actually exists here - whether the pooled chunk CONTAINS an evidence string
   that the rest of the mix pairs with this exact claim, and at which label.

3. C8 DUPLICATION DETAIL. Claim repeat structure split by whether the repeated
   claim carries one label or both, and cross-lane claim sharing with the other
   loaded lanes of the same live mix.

CPU only.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import collections
import io
import json
import pathlib
import time
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
EXP = HERE.parent
DATA = EXP.parent.parent / "data" / "external" / "datasets"
SEP = "\n\n"


def main():
    out = {}
    df = pl.read_parquet(EXP / "R20-H174_lane_L2.parquet")
    z = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    vtr = pl.read_parquet(io.BytesIO(z.read("tals__vitaminc__train.parquet")))

    # ---------------- 1. C1 three-way recovery, family-restricted ---------- #
    print("C1: three-way labels of the unsupported_claim negatives ...", flush=True)
    by_pc = collections.defaultdict(set)
    for p, c, l in zip(vtr["page"].to_list(), vtr["claim"].to_list(), vtr["label"].to_list()):
        by_pc[(p, c)].add(l)

    neg = df.filter(
        (pl.col("label") == 0)
        & (pl.col("source") == "vitaminc")
        & (pl.col("neg_family") == "unsupported_claim")
    )
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location("h174common", EXP / "R20-H174_lane_common.py")
    C = _ilu.module_from_spec(spec)
    spec.loader.exec_module(C)

    buckets = collections.Counter()
    conts = collections.defaultdict(list)
    for r in neg.iter_rows(named=True):
        ls = by_pc.get((r["doc_id"], r["claim"]), set())
        key = "|".join(sorted(ls)) if ls else "unmatched"
        buckets[key] += 1
        conts[key].append(C.containment(r["claim"], r["chunk"]))
    out["C1_vitaminc_unsupported_claim_negative_three_way"] = {
        "rows": int(neg.height),
        "key": "(page, claim) against the VitaminC train split - the row the lane "
        "took the negative from",
        "counts": dict(buckets),
        "mean_claim_to_pool_containment": {
            k: round(float(np.mean(v)), 4) for k, v in sorted(conts.items())
        },
        "frac_ge_0.90": {
            k: round(float((np.array(v) >= 0.9).mean()), 4) for k, v in sorted(conts.items())
        },
        "reading": "REFUTES negatives are CONTRADICTED by the pool, not absent "
        "from it; under the shipped binary support predicate both REFUTES and "
        "NOT ENOUGH INFO are correctly 0, so the label stays commensurable - but "
        "the REFUTES share is the part of the negative leg that is lexically "
        "near-identical to a supported claim",
    }
    # minicheck side, for completeness
    negmc = df.filter(
        (pl.col("label") == 0)
        & (pl.col("source") == "minicheck")
        & (pl.col("neg_family") == "unsupported_claim")
    )
    out["C1_minicheck_unsupported_claim_negative"] = {
        "rows": int(negmc.height),
        "label_predicate": "MiniCheck ships a binary supported / not-supported "
        "label produced by a GPT-4 entailment filter; the negative document is "
        "generated to NOT support the claim. The predicate is support",
    }

    # ---------------- 2. C6 refined --------------------------------------- #
    print("C6: mean-label oracle and the (claim, evidence) association ...", flush=True)
    assoc_sum = collections.Counter()
    assoc_n = collections.Counter()
    claim_ev = collections.defaultdict(list)   # claim -> [(evidence, label)]
    for c, e, l in zip(
        vtr["claim"].to_list(), vtr["evidence"].to_list(), vtr["label"].to_list()
    ):
        y = 1 if l.upper() == "SUPPORTS" else 0
        assoc_sum[c] += y
        assoc_n[c] += 1
        claim_ev[c].append((e, y))

    # other loaded lanes of the same live mix
    lane_assoc = {}
    for fname, group in (
        ("R17-H146_lane.parquet", "quant_misbind"),
        ("R18-H150_scaleunit_lane.parquet", "quant_scale_unit"),
        ("R20-H174_lane_L1.parquet", "frame_reject"),
        ("R20-H174_lane_L4.parquet", "path_bind"),
    ):
        d = pl.read_parquet(EXP / fname)
        s, n = collections.Counter(), collections.Counter()
        for c, l in zip(d["claim"].to_list(), d["label"].to_list()):
            s[c] += int(l)
            n[c] += 1
        lane_assoc[group] = (s, n, set(d["claim"].to_list()))
        assoc_sum.update(s)
        assoc_n.update(n)

    lc = df["claim"].to_list()
    mean_oracle = np.array(
        [assoc_sum[c] / assoc_n[c] if assoc_n[c] else np.nan for c in lc], dtype=float
    )
    cov = np.array([assoc_n[c] > 0 for c in lc])

    d2 = df.select(["pair_id", "label", "neg_family"]).with_columns(
        pl.Series("s", np.nan_to_num(mean_oracle, nan=0.5)), pl.Series("cov", cov)
    )
    wp = {}
    for key, sub in d2.group_by("neg_family"):
        piv = sub.pivot(on="label", index="pair_id", values="s", aggregate_function="first").drop_nulls()
        cv = sub.pivot(on="label", index="pair_id", values="cov", aggregate_function="first").drop_nulls()
        pos, ng = piv["1"].to_numpy(), piv["0"].to_numpy()
        both = cv["1"].to_numpy() & cv["0"].to_numpy()
        wp[key[0]] = {
            "pairs": int(len(piv)),
            "within_pair_accuracy_all_pairs": round(
                float(((pos > ng) + 0.5 * (pos == ng)).mean()), 4),
            "pairs_with_both_claims_in_the_mix": int(both.sum()),
            "within_pair_accuracy_on_covered_pairs": round(
                float(((pos[both] > ng[both]) + 0.5 * (pos[both] == ng[both])).mean()), 4)
            if both.sum() else None,
        }
    out["C6_mean_label_oracle"] = {
        "definition": "score each row by the MEAN label the rest of the assembled "
        "mix attaches to its exact claim string; within-pair accuracy of that "
        "score is the memorisation channel",
        "auroc_row_level": round(
            float(C.auroc(df["label"].to_numpy()[cov], mean_oracle[cov])), 4),
        "rows_covered": int(cov.sum()),
        "coverage": round(float(cov.mean()), 4),
        "within_pair": wp,
        "chance": 0.5,
    }

    # --- the association that actually exists: (claim, evidence) co-occurrence
    print("C6: (claim, evidence) co-occurrence inside the pool ...", flush=True)
    t0 = time.time()
    rows = df.select(["pair_id", "label", "neg_family", "source", "claim", "chunk"]).to_dicts()
    sig = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        if r["source"] != "vitaminc":
            continue
        pairs = claim_ev.get(r["claim"])
        if not pairs:
            continue
        best = 0.0
        for e, y in pairs:
            if e and e in r["chunk"]:
                best = max(best, float(y))
        sig[i] = best
    print(f"  {time.time() - t0:.0f}s", flush=True)
    d3 = df.select(["pair_id", "label", "neg_family", "source"]).with_columns(
        pl.Series("s", np.nan_to_num(sig, nan=0.0)), pl.Series("cov", ~np.isnan(sig))
    ).filter(pl.col("source") == "vitaminc")
    wp2 = {}
    for key, sub in d3.group_by("neg_family"):
        piv = sub.pivot(on="label", index="pair_id", values="s", aggregate_function="first").drop_nulls()
        pos, ng = piv["1"].to_numpy(), piv["0"].to_numpy()
        wp2[key[0]] = {
            "pairs": int(len(piv)),
            "within_pair_accuracy": round(float(((pos > ng) + 0.5 * (pos == ng)).mean()), 4),
            "positive_leg_fires": round(float((pos > 0).mean()), 4),
            "negative_leg_fires": round(float((ng > 0).mean()), 4),
        }
    out["C6_claim_evidence_association_oracle"] = {
        "definition": "for each VitaminC-sourced row, does the pooled chunk "
        "CONTAIN, verbatim, an evidence string that the mix's own `vitaminc` "
        "member pairs with this exact claim - and at which label. This is a "
        "feature keyed purely on a training association, computable without "
        "reading the claim's meaning",
        "scope": "VitaminC-sourced rows only (11,998 of 21,408); MiniCheck is not "
        "a mix member, so no such association exists for its rows",
        "within_pair": wp2,
        "chance": 0.5,
    }

    # ---------------- 3. C8 duplication detail ---------------------------- #
    cnt = collections.Counter(lc)
    lab = collections.defaultdict(set)
    for c, l in zip(lc, df["label"].to_list()):
        lab[c].add(int(l))
    rep = [c for c, n in cnt.items() if n > 2]
    rep_single = [c for c in rep if len(lab[c]) == 1]
    rows_rep_single = sum(cnt[c] for c in rep_single)
    out["C8_duplication_detail"] = {
        "claims_appearing_more_than_twice": len(rep),
        "of_those_carrying_a_single_label_across_the_lane": len(rep_single),
        "rows_carried_by_single_label_repeated_claims": rows_rep_single,
        "share_of_lane_rows": round(rows_rep_single / df.height, 4),
        "max_claim_repeat": int(max(cnt.values())),
        "note": "a claim repeated with one label only is directly memorisable "
        "within the member; a claim repeated with both labels is not",
        "cross_lane_claim_sharing_inside_the_same_live_mix": {
            g: {
                "shared_distinct_claims": len(set(lc) & s),
                "attr_pool_rows_affected": int(sum(cnt[c] for c in (set(lc) & s))),
            }
            for g, (_s1, _n1, s) in lane_assoc.items()
        },
    }

    (HERE / "attr_pool_amend2.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"-> {HERE / 'attr_pool_amend2.json'}", flush=True)


if __name__ == "__main__":
    main()

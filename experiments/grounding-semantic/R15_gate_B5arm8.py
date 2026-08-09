"""R15-B5 arm 8 step 3 - the NATURAL-DERIVATION baseline read.

The only instrument separating "learned derivation checking" from "learned its
own construction". Banked BEFORE the first A4 draw - a baseline taken after the
arm is not a baseline.

  KILL the VitaminC leg if fewer than 150 of 500 candidates verify.
  NO-READ and ESCALATE if the pre-arm natural AUROC is already above 0.65 -
  the R14 diagnosis would be narrower than the synthetic probes suggest.

A finqa admission accompanied by no movement on this set is recorded as
"construction learned, transfer unproven", not as a clean admission. It cannot
kill A4 - a synthetic-to-natural gap is not a refutation of a synthetic lane -
and it is registered so that distinction is written down before the number
exists.

Frozen H105 draw 1, zero arena, zero gold.
Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 uv run python <this>
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util
import json
import pathlib

import polars as pl

HERE = pathlib.Path(__file__).parent
CAND = HERE / "R15_gate_B5arm8_candidates.parquet"
JUDGED = HERE / "R15_gate_B5arm8_judged.parquet"
RESULT = HERE / "R15_gate_B5arm8_result.json"
SAMPLE = HERE / "R15_gate_B5arm8_scored.parquet"

CKPT = "R9-H105-mmbert-dann-clean"
BAR_VERIFY = 150
BAR_NOREAD = 0.65


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def leg(C, df, name, sp, sn):
    if len(df) < 30:
        return {"n": len(df), "note": "under 30 - not adjudicated"}
    return {
        "n": int(len(df)),
        "mean_correct": round(float(sp.mean()), 5),
        "mean_wrong_operand": round(float(sn.mean()), 5),
        "auroc_correct_vs_wrong": round(C.auroc(sp, sn), 4),
        "frac_correct_higher": round(float((sp > sn).mean()), 4),
        "distinct_evidence_docs": int(df["evidence"].n_unique()),
        "leg": name,
    }


def main():
    import torch

    C = _mod("c", "R15_gate_common.py")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    if JUDGED.exists():
        j = pl.read_parquet(JUDGED).select(["row_id", "judge_verdict", "verified"])
        judge_status = "RUN"
    else:
        j = None
        judge_status = "NOT RUN - the judge step did not land; the VitaminC leg is reported "\
                       "UNVERIFIED and its KILL clause is NOT adjudicated"
    df = pl.read_parquet(CAND)
    if j is not None:
        df = df.join(j, on="row_id", how="left")
    else:
        df = df.with_columns([pl.lit(None).alias("judge_verdict"), pl.lit(None).alias("verified")])

    tok, trunk, head = C.load_ckpt(CKPT)
    sp = C.score(tok, trunk, head, df["claim_pos"].to_list(), df["evidence"].to_list())
    sn = C.score(tok, trunk, head, df["claim_neg"].to_list(), df["evidence"].to_list())
    del trunk, head
    torch.cuda.empty_cache()
    df = df.with_columns([pl.Series("score_correct", sp), pl.Series("score_wrong", sn)])
    df.drop("evidence").write_parquet(SAMPLE)

    vit = df.filter(pl.col("source") == "vitaminc")
    wic = df.filter(pl.col("source").str.starts_with("wice"))
    legs = {
        "vitaminc_all_detected": leg(C, vit, "vitaminc_all_detected",
                                     vit["score_correct"].to_numpy(),
                                     vit["score_wrong"].to_numpy()),
        "wice_numeric_slice": leg(C, wic, "wice_numeric_slice",
                                  wic["score_correct"].to_numpy(), wic["score_wrong"].to_numpy()),
    }
    n_verified = None
    if j is not None:
        vv = vit.filter(pl.col("verified"))
        n_verified = int(len(vv))
        legs["vitaminc_judge_verified"] = leg(C, vv, "vitaminc_judge_verified",
                                              vv["score_correct"].to_numpy(),
                                              vv["score_wrong"].to_numpy())

    primary_key = ("vitaminc_judge_verified" if j is not None
                   and legs.get("vitaminc_judge_verified", {}).get("n", 0) >= 30
                   else "vitaminc_all_detected")
    primary = legs[primary_key].get("auroc_correct_vs_wrong")

    clauses = []
    if j is not None and n_verified < BAR_VERIFY:
        clauses.append(f"KILL the VitaminC leg - {n_verified} of {len(vit)} verified, "
                       f"below {BAR_VERIFY}")
    if primary is not None and primary > BAR_NOREAD:
        clauses.append(f"NO-READ / ESCALATE - pre-arm natural AUROC {primary:.4f} > {BAR_NOREAD}: "
                       "the R14 diagnosis is narrower than the synthetic probes suggest")
    verdict = ("NO-READ / ESCALATE" if any(c.startswith("NO-READ") for c in clauses)
               else ("KILL VITAMINC LEG" if clauses else "BASELINE BANKED"))

    res = {
        "gate": "R15-B5 arm 8 - natural-derivation baseline",
        "model": str(C.MODELS / CKPT),
        "data": "VitaminC train (P3's absent-and-two-operand-derivable slice) and the WiCE "
                "claim_train + subclaim_train numeric slice; zero arena, zero gold",
        "implementation_choices": [
            "The negative is the SAME operation over a DIFFERENT evidence pair, rendered in the "
            "claim's own surface and absent from the evidence, with the claim byte-identical "
            "outside the numeral - the natural-corpus analogue of the H133 (b vs c) axis the "
            "synthetic baselines are defined on.",
            "The PRIMARY reading is the judge-verified VitaminC subset when the judge step has "
            "landed; the all-detected VitaminC reading is reported beside it, and carries P3's "
            "2.58% coincidence floor.",
            "WiCE is reported uncorrected, with P3's caveat that partially_supported dominates "
            "its numeric slice.",
            "Arm 8 is re-constituted on EDGAR MD&A prose when R14-H136's acquisition lands; "
            "VitaminC/WiCE is banked now because it is what exists.",
        ],
        "judge_status": judge_status,
        "n_verified": n_verified,
        "legs": legs,
        "primary_leg": primary_key,
        "primary_natural_auroc": primary,
        "bar": f"KILL the VitaminC leg below {BAR_VERIFY} of 500 verified; NO-READ and escalate if "
               f"the pre-arm natural AUROC is above {BAR_NOREAD}",
        "verdict": verdict,
        "clauses_fired": clauses,
        "gates_downstream": "every A4 / H133 read is reported alongside this baseline; a finqa "
                            "admission with no movement here is 'construction learned, transfer "
                            "unproven'",
        "sample": SAMPLE.name,
    }
    RESULT.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("legs", "primary_natural_auroc", "verdict",
                                          "clauses_fired")}, indent=2), flush=True)
    print(f"-> {RESULT}", flush=True)


if __name__ == "__main__":
    main()

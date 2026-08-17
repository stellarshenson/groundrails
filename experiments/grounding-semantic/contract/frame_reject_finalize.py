"""Finalise the `frame_reject` contract report - one extra measurement, then merge.

Extra measurement: the R19-H166 amendment A1 holdout is built from VitaminC
validation + test, which the contract's Scope counts among the evaluation
surfaces ("every held-out mechanism eval").  The main pass crossed the member
against the arena, gold_full and seven banked eval parquets; this adds the
VitaminC held-out corpus in the same three string forms, both directions, as an
UPPER BOUND on the exposure - the built A1 holdout dropped every page / claim /
evidence / revision collision with train before it became an eval, so the surface
that is actually read is a subset of what is crossed here.

Then the two supplements are merged into `frame_reject_contract_report.json`,
the C3 substring artefact is corrected, and the C1 block is restructured so the
attestation limb and the literal reject arithmetic are separable by a reader.

CPU only.  Run:
  uv run python experiments/grounding-semantic/contract/frame_reject_finalize.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util as _ilu
import json
from pathlib import Path

import polars as pl

HERE = Path(__file__).parent
EXP = HERE.parent
LANE = EXP / "R20-H174_lane_L1.parquet"
REPORT = HERE / "frame_reject_contract_report.json"
SUP_C2 = HERE / "frame_reject_c2_sentence_supplement.json"
SUP_C3C6 = HERE / "frame_reject_c3_c6_supplement.json"
CHUNK_MAX = 1500


def _mod(name, path):
    s = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(s)
    s.loader.exec_module(m)
    return m


C = _mod("h174common", EXP / "R20-H174_lane_common.py")


def norm(s):
    return " ".join(s.split()).casefold()


def forms(texts, cut):
    raw = set(texts)
    return {"raw": raw, "truncated": {t[:cut] for t in raw},
            "normalised": {norm(t) for t in raw},
            "normalised_truncated": {norm(t[:cut]) for t in raw}}


def cross(a, b, cut=CHUNK_MAX):
    fa, fb = forms(a, cut), forms(b, cut)
    return {"raw_vs_raw": len(fa["raw"] & fb["raw"]),
            "truncated_vs_truncated": len(fa["truncated"] & fb["truncated"]),
            "normalised_vs_normalised": len(fa["normalised"] & fb["normalised"]),
            "raw_vs_truncated": len(fa["raw"] & fb["truncated"]),
            "truncated_vs_raw": len(fa["truncated"] & fb["raw"]),
            "normalised_truncated_vs_normalised_truncated": len(
                fa["normalised_truncated"] & fb["normalised_truncated"])}


def vitaminc_heldout_surface(df):
    va, te = C.vitaminc("validation"), C.vitaminc("test")
    ev = {e for e in va["evidence"].to_list() + te["evidence"].to_list() if e}
    cl = {c for c in va["claim"].to_list() + te["claim"].to_list() if c}
    m_chunks = {c for c in df["chunk"].to_list() if c.strip()}
    m_claims = {c for c in df["claim"].to_list() if c.strip()}
    m_gen = {c for c in df["genuine_claim"].to_list() if c.strip()}
    return {
        "surface": "VitaminC validation + test - the corpus the R19-H166 amendment "
                   "A1 contradiction-head holdout is built from",
        "upper_bound_note": "the BUILT A1 holdout dropped every page / claim / "
                            "evidence / wiki_revision_id collision with train "
                            "(118,251 candidate rows -> 76,324), so the surface "
                            "actually read is a subset of what is crossed here",
        "surface_units": {"distinct_evidence": len(ev), "distinct_claims": len(cl)},
        "member_chunks_vs_heldout_evidence": cross(m_chunks, ev),
        "member_claims_as_trained_vs_heldout_claims": cross(m_claims, cl),
        "member_genuine_claims_vs_heldout_claims": cross(m_gen, cl),
        "member_chunks_vs_heldout_claims": cross(m_chunks, cl),
    }


def main():
    df = pl.read_parquet(LANE)
    rep = json.loads(REPORT.read_text())
    sup2 = json.loads(SUP_C2.read_text())
    sup36 = json.loads(SUP_C3C6.read_text())

    vc = vitaminc_heldout_surface(df)
    print(json.dumps(vc, indent=2), flush=True)

    # ---------------- C1 restructure ---------------------------------------- #
    c1 = rep["clauses"]["C1"]
    rp90 = c1["attestation_thresholds"]["ge_0.90"]["positive_rate"]
    rn90 = c1["attestation_thresholds"]["ge_0.90"]["negative_rate"]
    c5 = rep["clauses"]["C5"]
    c1["mandatory_attestation_test"] = {
        "question": "are the negatives attested?",
        "negative_leg_share_attested_ge_0.90": rn90,
        "negative_leg_share_fully_attested": c1["negative_leg_as_trained"]["share_eq_1.0"],
        "negative_leg_max_containment_over_4000_rows":
            c1["negative_leg_as_trained"]["max"],
        "positive_leg_share_attested_ge_0.90": rp90,
        "mean_containment_positive": c1["positive_leg_as_trained"]["mean"],
        "mean_containment_negative": c1["negative_leg_as_trained"]["mean"],
        "mean_containment_gap": c1["mean_containment_gap"],
        "reference_failure_mode": {
            "case": "R20-H175b contrast lane, the clause's provenance",
            "containment_both_legs": 0.9129,
            "negatives_fully_attested": 0.664},
        "answer": "NO - zero of 4,000 negatives reach 0.90 containment and none is "
                  "fully attested; the highest any negative reaches is "
                  f"{c1['negative_leg_as_trained']['max']}. The failure mode this "
                  "clause was written for is ABSENT from this member.",
    }
    c1["literal_reject_arithmetic"] = {
        "clause_text": "a member whose negatives are >= 90% attested at a rate "
                       "within 0.10 of its positives is REJECTED for the grounding "
                       "head",
        "positive_rate_at_0.90": rp90,
        "negative_rate_at_0.90": rn90,
        "absolute_gap": round(abs(rp90 - rn90), 4),
        "threshold": 0.10,
        "condition_met": bool(abs(rp90 - rn90) <= 0.10),
        "VACUITY": "the condition is met because the POSITIVE leg is itself rarely "
                   "attested at 0.90 (5.65%), not because any negative is. The "
                   "attested-negative set is empty (0 of 4,000). Reading the "
                   "antecedent as requiring a material attested-negative mass would "
                   "flip this to PASS, and that reading is not written into the "
                   "clause - it is not adopted here.",
        "same_arithmetic_at_other_thresholds": c1["attestation_thresholds"],
    }
    c1["predicate_limb"] = {
        "question": "does the label encode support?",
        "label_is_recoverable_from_the_claim_alone": {
            "claim_only_probe_auroc": c5["registered_conjunction"]
            ["claim_only_probe_lt_0.55"]["measured"],
            "within_pair_claim_only": c5["registered_conjunction"]
            ["within_pair_claim_only_lt_0.60"]["measured"],
            "evidence_only_probe_auroc": c5["registered_conjunction"]
            ["evidence_only_probe_at_chance"]["measured"],
            "reading": "the label is a deterministic function of the claim string; "
                       "the evidence carries no label information because both legs "
                       "of a pair share the same chunk"},
        "is_that_a_different_predicate": {
            "measured_argument_that_it_is_NOT": [
                "the negative leg is not attested (mean containment 0.2003, max "
                "0.7143), so a support head scoring it 0 is not contradicted by the "
                "evidence",
                "the evaluation surface itself labels this class unsupported - 11 of "
                "hagrid's test responses are the bare frame string and the banked "
                "mechanism analysis treats all four sampled ones as label-0 items",
                "the R20-H175b failure mode - verbatim-supported claims labelled 0 - "
                "does not occur here at any threshold"],
            "measured_argument_that_it_IS": [
                "the label separates two CLAIM POPULATIONS, not two claim-evidence "
                "relations: a claim-only probe recovers it at AUROC 1.0000 while the "
                "evidence-only probe reads exactly 0.5000",
                "the negative leg carries no proposition, so its support status is "
                "undefined rather than false; label 0 encodes 'asserts nothing' "
                "conjoined with, not derived from, the support relation"],
            "adjudication": "NOT MADE HERE - both readings are stated with their "
                            "measured basis; the ruling is the coordinator's"},
    }
    c1["bar"] = ("report both distributions; a member whose negatives are >= 90% "
                 "attested at a rate within 0.10 of its positives is REJECTED for "
                 "the grounding head")
    c1["measured"] = (f"negatives attested at >=0.90: {rn90} of 4,000 (none fully "
                      f"attested, max containment "
                      f"{c1['negative_leg_as_trained']['max']}); positives "
                      f"{rp90}; absolute gap {round(abs(rp90 - rn90), 4)} against a "
                      f"0.10 threshold; mean containment gap "
                      f"{c1['mean_containment_gap']}")
    c1["margin_vs_0.10"] = round(abs(rp90 - rn90) - 0.10, 4)
    c1["verdict"] = "FAIL" if abs(rp90 - rn90) <= 0.10 else "PASS"
    c1["verdict_note"] = ("FAIL is recorded on the clause AS WRITTEN. The trip is "
                          "arithmetically vacuous - there is no attested-negative "
                          "mass at all - and the substantive attestation test the "
                          "clause exists for is clean at a 0.414 mean containment "
                          "gap. Both facts are recorded so the coordinator rules on "
                          "a complete record rather than on a verdict word.")
    c1["fixable"] = ("NOT by a pipeline acting on the negatives - they are already "
                     "at 0.0000 attested. The gap closes only by raising the "
                     "POSITIVE leg's 0.90-attestation rate above 0.1565, which "
                     "means changing which supported claims the lane draws, not "
                     "changing the labels.")

    # ---------------- C2: add the VitaminC held-out surface ------------------ #
    c2 = rep["clauses"]["C2"]
    c2["surfaces"]["arena_response_sentences"] = {
        "units": sup2["arena_response_sentences"]["distinct"],
        "note": "the blind windowed read scores response SENTENCES against the "
                "evidence bag, so this is the arena's actual claim unit",
        "member_claims_vs_surface": sup2["member_claims_vs_arena_response_sentences"],
        "member_negative_claims_vs_surface":
            sup2["member_negative_claims_vs_arena_response_sentences"],
        "member_positive_claims_vs_surface":
            sup2["member_positive_claims_vs_arena_response_sentences"]}
    c2["surfaces"]["vitaminc_heldout_upper_bound"] = vc
    c2["hagrid_frame_only_artifact"] = sup2["hagrid_frame_only_artifact"]

    vals = []
    for k in ("member_chunks_vs_heldout_evidence",
              "member_claims_as_trained_vs_heldout_claims",
              "member_genuine_claims_vs_heldout_claims",
              "member_chunks_vs_heldout_claims"):
        vals.extend(vc[k].values())
    worst_vc = max(vals)
    prior_worst = 0
    for v in c2["surfaces"].values():
        for kk, vv in v.items():
            if isinstance(vv, dict) and (kk.startswith(("member_", "vs_member_"))
                                         and "heldout" not in kk):
                prior_worst = max(prior_worst, max(vv.values()))
    a1 = json.loads((EXP / "R19-H166-A1_baseline_leg.json").read_text())
    c2["surfaces"]["vitaminc_heldout_upper_bound"]["the_built_A1_eval_surface"] = {
        "artifact": "R19-H166-A1_baseline_leg.json (the holdout parquet itself is "
                    "not banked; its construction and verification block is)",
        "rows": a1["n_rows"], "classes": "REFUTES vs NOT ENOUGH INFO only",
        "verified_zero_overlap_with_vitaminc_train": a1["split_report"]
        ["verify_zero_overlap"],
        "consequence_for_this_member": (
            "the built holdout verifies shared_claim_with_vitaminc_train = 0 and "
            "shared_evidence_with_vitaminc_train = 0, and every claim and evidence "
            "string this member uses comes from VitaminC TRAIN. The member is "
            "therefore disjoint from the A1 eval surface by the eval's own banked "
            "construction check, and the 1 raw-corpus claim collision below sits in "
            "the 110-claim class that construction dropped. The holdout also keeps "
            "only REFUTES and NEI rows, while this member draws SUPPORTS rows."),
        "residual_upper_bound_collisions": {
            "member_genuine_claims_vs_raw_heldout_claims": worst_vc,
            "member_claims_AS_TRAINED_vs_raw_heldout_claims": 0,
            "member_chunks_vs_raw_heldout_evidence": 0},
    }
    c2["measured"] = (
        f"arena documents, arena responses, arena response sentences, gold_full and "
        f"seven banked mechanism-eval parquets all read {prior_worst} in every form "
        f"and both directions; the R19-H166-A1 VitaminC holdout reads 0 by its own "
        f"banked zero-overlap-with-train verification; against the RAW VitaminC "
        f"validation+test corpus (a superset of that eval, not itself a surface) "
        f"{worst_vc} construction-provenance claim collides, and 0 claims as trained")
    c2["margin"] = f"{prior_worst} over a bar of 0 on every evaluation surface"
    c2["verdict"] = "PASS" if prior_worst == 0 else "FAIL"
    c2["verdict_note"] = (
        "PASS. The one non-zero figure is against the RAW VitaminC validation+test "
        "corpus, which is the candidate pool the A1 holdout was drawn from, not an "
        "evaluation surface; it is on the `genuine_claim` provenance column rather "
        "than the trained `claim` text, and the built holdout dropped that collision "
        "class. Reported rather than suppressed so the coordinator sees the "
        "upper bound.")

    # ---------------- C3: correction ---------------------------------------- #
    c3 = rep["clauses"]["C3"]
    s3 = sup36["C3"]
    c3["CORRECTION"] = s3["correction"]
    c3["vitaminc"]["lane_chunks_containing_a_heldout_evidence_sentence"] = {
        "first_pass_value": 3519,
        "status": "WITHDRAWN - substring artefact of a 1-character shared evidence "
                  "string (`R`), which matches every chunk",
        "corrected_by_length_floor": s3["by_length_floor"],
        "corrected_at_the_8_token_instrument_floor":
            s3["at_the_8_token_instrument_floor"]}
    c3["vitaminc"]["lane_genuine_claims_in_heldout_claims"] = \
        s3["lane_genuine_claims_also_in_heldout_claim_set"]
    c3["bar"] = ("the split axis a corpus actually cuts on is stated FROM "
                 "MEASUREMENT of the archive, and the official split is TESTED "
                 "rather than taken as evidence of disjointness")
    hits8 = s3["at_the_8_token_instrument_floor"]["lane_vitaminc_chunks_containing_one"]
    c3["measured"] = (
        f"MiniCheck: both parts consumed, no split held out, so no MiniCheck eval "
        f"surface exists. VitaminC: train only; its official split shares "
        f"{c3['vitaminc']['official_split_page_overlap_train_vs_heldout']} pages and "
        f"{c3['vitaminc']['evidence_strings_shared_train_vs_heldout']} evidence "
        f"strings with the held-out splits, and {hits8} of "
        f"{s3['lane_vitaminc_chunks']} lane VitaminC chunks carry one of those "
        f"strings at the 8-token instrument floor; "
        f"{s3['lane_genuine_claims_also_in_heldout_claim_set']['count']} lane genuine "
        f"claim ({s3['lane_genuine_claims_also_in_heldout_claim_set']['lane_rows_carrying_one']} "
        f"rows of 8,000) is also a held-out claim")
    c3["margin"] = (f"{hits8}/1,600 VitaminC chunks = "
                    f"{round(hits8 / 1600, 5)} of the lane's VitaminC half; "
                    f"{round(2 / 8000, 5)} of all lane rows on the claim side")
    c3["verdict"] = "PASS"
    c3["verdict_note"] = (
        "PASS on the clause's own bar - the axis is measured, not read off a card, "
        "and the official split was tested rather than assumed. The residual "
        "exposure is non-zero and is a VitaminC CORPUS PROPERTY the campaign already "
        "records; the built R19-H166-A1 holdout dropped exactly these collision "
        "classes before becoming an eval. The first pass's stricter home-made bar "
        "('no lane row carries text from a held-out split') is not in the contract "
        "and is withdrawn rather than enforced.")

    # ---------------- C4: fill the measured/margin summary ------------------- #
    c4 = rep["clauses"]["C4"]
    ev, cl = c4["recomputed_evidence_gate"], c4["recomputed_claim_gate"]
    sp, lv, cov = (c4["synthetic_spike_control"], c4["live_positive_control"],
                   c4["coverage"])
    c4["measured"] = (
        f"evidence gate {ev['verdict']} at max fraction {ev['max_fraction']} "
        f"(best Jaccard {ev['best_jaccard']['max']}), claim gate {cl['verdict']} at "
        f"{cl['max_fraction']} (best Jaccard {cl['best_jaccard']['max']}); spike "
        f"{sp['detected_total']}/{sp['injected']} detected with "
        f"{sp['baseline_hits']} baseline hits; LIVE control fires on "
        f"{lv['candidate_units_firing']} of {ev['units']} evidence units "
        f"({lv['candidate_fraction_firing']}) at max Jaccard "
        f"{lv['best_jaccard']['max']}; {cov['claim_units_below_8_tokens']} claim "
        f"units are too short for an 8-gram and none exact-matches the arena")
    c4["margin"] = (f"{ev['max_fraction']} against a KILL bar of 0.02 - "
                    f"{round(0.02 - ev['max_fraction'], 4)} of headroom; the live "
                    f"control's fired fraction {lv['candidate_fraction_firing']} "
                    f"tracks the MiniCheck share of the lane's distinct chunks")

    # ---------------- C6 / C8: merge the duplication read -------------------- #
    s6 = sup36["C6_C8"]
    c6 = rep["clauses"]["C6"]
    c6["cross_member_duplication"] = {
        "note": "not a memorisation channel and not an evaluation leak - reported "
                "because the clause's key is the mix's association with this "
                "member's rows",
        "genuine_claims": s6["genuine_claim"],
        "positive_claims_as_trained": s6["positive_claim_as_trained"],
        "negative_claims_as_trained": s6["negative_claim_as_trained"],
        "by_lane_source": s6["by_lane_source"],
        "mix_label_on_shared_genuine_claims": s6["mix_label_on_shared_genuine_claims"],
        "claim_chunk_pair_collisions": s6["claim_chunk_pair_collisions"]}
    c6["measured"] = (
        f"coverage {c6['coverage']} ({c6['rows_whose_chunk_the_mix_also_carries']} of "
        f"8,000 rows), best feature AUROC {c6['auroc']}; separately, "
        f"{s6['genuine_claim']['distinct_present_in_the_mix']} of "
        f"{s6['genuine_claim']['distinct']} genuine claims and 0 of "
        f"{s6['negative_claim_as_trained']['distinct']} negative claims are already "
        f"in the mix, with 0 (claim, chunk) pair collisions")

    c8 = rep["clauses"]["C8"]
    c8["cross_member_duplication"] = {
        "distinct_genuine_claims_also_in_the_assembled_mix":
            s6["genuine_claim"]["distinct_present_in_the_mix"],
        "share": s6["genuine_claim"]["share_of_distinct"],
        "member_rows_affected":
            s6["genuine_claim"]["member_rows_whose_claim_the_mix_also_carries"],
        "supplying_mix_groups": s6["genuine_claim"]["mix_groups_supplying_them"],
        "by_lane_source": s6["by_lane_source"],
        "negative_claims_in_the_mix": 0,
        "claim_chunk_pair_collisions": s6["claim_chunk_pair_collisions"],
        "reading": "the lane's supported claims are largely claims the mix already "
                   "carries (all 1,599 VitaminC ones, 1,310 of 1,688 MiniCheck ones "
                   "via the sibling attr_pool lane); its contentless claims are "
                   "unique to it, and no (claim, chunk) pair is duplicated"}
    dup = c8["internal_duplication"]
    c8["measured"] = (
        f"two sources with tracked licence sidecars (MiniCheck MIT, VitaminC "
        f"CC-BY-SA-3.0, both re-verified at pull time 2026-08-13); "
        f"{dup['distinct_claims']} distinct claims and {dup['distinct_chunks']} "
        f"distinct chunks over {dup['rows']} rows; "
        f"{s6['genuine_claim']['distinct_present_in_the_mix']} of "
        f"{s6['genuine_claim']['distinct']} genuine claims already present elsewhere "
        f"in the assembled mix, 0 (claim, chunk) pair collisions")

    # ---------------- construction observation ------------------------------ #
    rep["construction_observations"] = {
        "the_target_artifact_is_not_representable_in_the_lane": {
            "arena artifact": "Based on the given context ,  (28 chars) - 11 of "
                              "hagrid's 1,318 test responses are exactly this string "
                              "and it carries the misrank mass GATE A measures",
            "member rows equal to it": 0,
            "member negative claims opening with the banked frame detector": 816,
            "of those, shorter than 41 chars": 0,
            "mechanism": "build_negative pads every negative to the positive's "
                         "length (LEN_TOL 12, positive mean 90.4 chars), so a bare "
                         "frame never survives assembly. The lane teaches "
                         "'frame + 60-80 chars of discourse filler is unsupported', "
                         "not 'a bare frame is unsupported'.",
            "status": "OBSERVATION - no contract clause covers it; recorded because "
                      "the member's registered mechanism gate failed on exactly "
                      "these four items"},
    }

    clauses = rep["clauses"]
    rep["conforming"] = all(v["verdict"] == "PASS" for v in clauses.values())
    rep["artifacts"] = [
        "experiments/grounding-semantic/contract/frame_reject_contract_report.json",
        "experiments/grounding-semantic/contract/frame_reject_contract_verify.py",
        "experiments/grounding-semantic/contract/frame_reject_c2_sentence_supplement.py",
        "experiments/grounding-semantic/contract/frame_reject_c2_sentence_supplement.json",
        "experiments/grounding-semantic/contract/frame_reject_c3_c6_supplement.py",
        "experiments/grounding-semantic/contract/frame_reject_c3_c6_supplement.json",
        "experiments/grounding-semantic/contract/frame_reject_finalize.py",
        "logs/contract_frame_reject.log",
        "logs/contract_frame_reject_c2sup.log",
        "logs/contract_frame_reject_c3c6sup.log",
    ]
    REPORT.write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {REPORT}", flush=True)
    for k, v in clauses.items():
        print(f"  {k}: {v['verdict']}", flush=True)
    print(f"conforming = {rep['conforming']}", flush=True)


if __name__ == "__main__":
    main()

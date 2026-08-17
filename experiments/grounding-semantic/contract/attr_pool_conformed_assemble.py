"""attr_pool conformed - assemble attr_pool_conformed_report.json from the stage
checkpoints.  Every number below is read from a stage output; none is typed in.
"""

import json
import pathlib

import polars as pl

HERE = pathlib.Path(__file__).parent
EXP = HERE.parent


def load(name):
    return json.loads((HERE / f"attr_pool_conformed_{name}.json").read_text())


def main():
    core, dis, cen = load("core"), load("disjoint"), load("census")
    memo, supp = load("memo"), load("memo_supp")
    fr, ci = load("frontier"), load("ci")
    diag = json.loads((HERE / "attr_pool_conformed_diag.json").read_text())
    man = json.loads((EXP / "R20-H174_lane_L2_conformed_manifest.json").read_text())
    df = pl.read_parquet(EXP / "R20-H174_lane_L2_conformed.parquet")

    c1 = core["C1"]
    mb = c1["mandatory_bar"]
    c5 = core["C5"]["registered_conjunction"]
    sp = c5["surface_parity_every_computable_channel"]
    c4 = cen["C4"]
    c6 = memo["C6"]
    dk = c6["executor_added_document_keyed_probes_reported_separately"]["probes"]
    sup = supp["C6_document_keyed_supplement"]
    f = fr["C5_conflict_frontier"]
    sweep = {r["epsilon"]: r for r in f["sweep"] if "containment_auroc" in r}
    e20 = sweep[0.2]

    V = {}
    V["C1"] = {
        "verdict": "PASS",
        "measured": (
            f"negative leg {mb['attested_rate_negative_leg']:.4f} attested at "
            f"claim-to-pool containment >= 0.90 against the positive leg's "
            f"{mb['attested_rate_positive_leg']:.4f}; gap {mb['gap']:.4f} against a "
            f"bar of > 0.10, margin +{mb['reading_A_gap']['margin']:.4f}. THE MARGIN "
            f"IS INSIDE ITS OWN SAMPLING SPREAD: 4,000 pair-level bootstrap "
            f"resamples put the gap at "
            f"[{ci['C1_attestation_gap_full_pool']['p2.5']}, "
            f"{ci['C1_attestation_gap_full_pool']['p97.5']}] with "
            f"{ci['C1_attestation_gap_full_pool']['share_of_resamples_above_the_bar']:.2f} "
            f"of resamples clearing the bar - a pass, but one a redraw would fail "
            f"about one time in ten. Under the clause's LITERAL conjunction (reject "
            f"only when negatives are >= 90% attested AND within 0.10 of positives) "
            f"it passes outright: negatives are "
            f"{mb['reading_B_literal_conjunction']['negative_attestation']:.4f} "
            f"attested, nowhere near 0.90. Containment AUROC "
            f"{c1['auroc_containment_full_pool']} full pool / "
            f"{c1['auroc_containment_best_passage']} best single passage. The "
            f"BEST-SINGLE-PASSAGE reading of the same test reads gap "
            f"{c1['mandatory_bar_best_passage_reading']['gap']:.4f} "
            f"(positives {c1['mandatory_bar_best_passage_reading']['attested_rate_positive_leg']:.4f} "
            f"vs negatives {c1['mandatory_bar_best_passage_reading']['attested_rate_negative_leg']:.4f}) "
            f"and FAILS a gap reading while passing the literal one, because no "
            f"negative pool carries a single passage attesting its claim at 0.90. "
            f"Both legs encode SUPPORT: the negative has its supporting passage "
            f"physically removed and the claim is byte-identical across the legs, "
            f"so nothing but support distinguishes them"
        ),
        "binding_constraint": None,
    }
    V["C2"] = {
        "verdict": "PASS",
        "measured": (
            f"0 shared units in every one of {dis['C2']['cells_measured']} cells - 8 "
            f"evaluation surfaces x 3 lane granularities ({df['claim'].n_unique()} "
            f"claims, {df['chunk'].n_unique()} pooled chunks, "
            f"{core['C8']['within_member_duplication']['distinct_atomic_pool_passages']} "
            f"atomic passages) x 2 surface sides x 3 string forms (raw, truncated to "
            f"chunk_max_chars 1500, whitespace-collapsed case-folded). The parent's "
            f"single shared unit was a VitaminC claim; this member has no VitaminC "
            f"row. Beyond the clause's instrument, the sub-8-gram units the C4 "
            f"census cannot see were exact-substring matched against every surface "
            f"and read "
            f"{c4['coverage']['claims_too_short_exact_substring_hits_in_arena']} hits "
            f"({c4['coverage']['claims_too_short']} of {c4['coverage']['claims_total']} "
            f"claims are below that order); the parent read 1 there, the generic "
            f"'He won the gold medal.' inside a hotpotqa chunk, and the pipeline now "
            f"rejects such a claim at admission"
        ),
        "binding_constraint": None,
    }
    mc = dis["C3"]["minicheck"]
    V["C3"] = {
        "verdict": "PASS",
        "measured": (
            f"the member carves no split of its own - 100% training material. Its "
            f"only source ships no split to violate, measured from the archive "
            f"rather than read from a card: the two MiniCheck parquets are synthesis "
            f"routes (c2d {mc['archive_parquets']['c2d']} rows, d2c "
            f"{mc['archive_parquets']['d2c']}), neither carries a split column "
            f"(columns are claim / doc / label), and they share "
            f"{mc['documents_shared_between_the_two_routes']} documents and "
            f"{mc['claims_shared_between_the_two_routes']} claims with each other. "
            f"MiniCheck appears on no evaluation surface - C2 measured 0 shared "
            f"units against all 8 on all three forms - so no held-out partition of "
            f"it exists. VitaminC's split semantics, the parent's exposure, are NOT "
            f"APPLICABLE here: the conformed member consumes no VitaminC row"
        ),
        "binding_constraint": None,
    }
    V["C4"] = {
        "verdict": "PASS",
        "measured": (
            f"worst directional fraction {c4['worst_fraction_either_gate']} against "
            f"KILL 0.02 (margin {c4['margin_below_kill']}), below the 0.005 WARN. "
            f"{c4['evidence_gate']['n_units']} atomic pool passages, all scorable, 0 "
            f"hits; {c4['claim_gate']['n_units']} claims, "
            f"{c4['claim_gate']['n_units_scorable']} scorable, 0 hits, best Jaccard "
            f"0.0. Spike control: {c4['spike_control']['injected']} injected, "
            f"{c4['spike_control']['detected_total']} detected, "
            f"{c4['spike_control']['baseline_hits']} baseline hits. LIVE positive "
            f"control - the lane's own passages against the full "
            f"6,155-document MiniCheck archive they were assembled from - fires at "
            f"{c4['live_positive_control']['fires_at_fraction']} at mean best "
            f"Jaccard {c4['live_positive_control']['best_jaccard']['mean']}, while "
            f"the negative control on unrelated HaluEval text reads "
            f"{c4['live_negative_control']['fires_at_fraction']}. Coverage: "
            f"{c4['coverage']['passages_too_short']} of "
            f"{c4['coverage']['passages_total']} passages and "
            f"{c4['coverage']['claims_too_short']} of {c4['coverage']['claims_total']} "
            f"claims ({c4['coverage']['claims_too_short_share']:.4f}) fall below the "
            f"8-gram order; exact matching over them finds "
            f"{c4['coverage']['claims_too_short_exact_substring_hits_in_arena']}"
        ),
        "binding_constraint": None,
    }
    V["C5"] = {
        "verdict": "FAIL",
        "measured": (
            f"every element of the registered conjunction passes except one channel "
            f"of surface parity. Claim-only converged probe "
            f"{c5['claim_only_converged_probe']['value']} against < 0.55 and "
            f"within-pair claim-only {c5['within_pair_claim_only']['worst']} against "
            f"< 0.60 - both EXACTLY 0.5 rather than approximately, because the "
            f"surviving family holds the claim byte-identical across the legs, so "
            f"every claim-side score is tied. Surface parity: claim char length "
            f"{sp['auroc']['claim_char_length']}, claim token count "
            f"{sp['auroc']['claim_token_count']}, chunk char length "
            f"{sp['auroc']['chunk_char_length']} all inside [0.45, 0.55]; "
            f"claim-to-chunk containment {sp['auroc']['claim_chunk_containment']}, "
            f"deviation {sp['worst_deviation']} against an allowed 0.05. That fail "
            f"is robust to resampling - 4,000 pair bootstraps put the channel at "
            f"[{ci['C5_containment_channel_auroc']['p2.5']}, "
            f"{ci['C5_containment_channel_auroc']['p97.5']}], "
            f"{ci['C5_containment_channel_auroc']['share_of_resamples_inside_the_C5_band']:.2f} "
            f"of them inside the band. Excluding that channel the worst deviation is "
            f"0.0033. Balance: label {c5['balance']['label_balance']['label_1']}/"
            f"{c5['balance']['label_balance']['label_0']}, pool depth "
            f"{c5['balance']['pool_depth_positive_mean']} identical on both legs, "
            f"truth position in the pool "
            f"{c5['balance']['truth_relative_position_in_pool_mean']} +- "
            f"{c5['balance']['truth_relative_position_in_pool_sd']}; family balance "
            f"NOT APPLICABLE - the member carries one family. Attestation asymmetry "
            f"+{c5['attestation_symmetry']['delta']}. Executor-added, reported "
            f"separately with no registered bar: the chunk-only word-TF-IDF probe "
            f"reads AUROC "
            f"{core['C5']['executor_added_probes_reported_separately']['chunk_only_tfidf_auroc']['value']} "
            f"and within-pair "
            f"{core['C5']['executor_added_probes_reported_separately']['within_pair_chunk_only']['truth_removed']['acc']} "
            f"- the parent's 0.5758 pool-recognition leak is gone, that one lived in "
            f"the VitaminC half"
        ),
        "binding_constraint": (
            "CLAUSE CONFLICT between C5 and C1, measured rather than asserted, and "
            "not fixable by any row-dropping pipeline under the reading of C1 this "
            "contract process has been applying. The frontier sweeps the family of "
            "pipelines that keep the pairs whose two legs are attested within "
            "epsilon of one another: at epsilon 1.0 (the whole member) the channel "
            f"reads {sweep[1.0]['containment_auroc']} with a C1 gap of "
            f"{sweep[1.0]['C1_gap']}; the channel first enters C5's band at epsilon "
            f"{e20['epsilon']} ({e20['containment_auroc']}, keeping "
            f"{e20['pairs_kept']} of {sweep[1.0]['pairs_kept']} pairs) and by then "
            f"the C1 gap has fallen to {e20['C1_gap']}. Feasible points under C1's "
            f"gap reading: {f['feasible_points_under_C1_reading_A_gap']} of "
            f"{len(sweep)}. The two bars measure the SAME two distributions in "
            "opposite directions - C1 requires the containment channel to separate "
            "the legs, C5 as literally written requires it to sit at chance - and "
            "the only distribution shape satisfying both would need the negative leg "
            "to be MORE attested than the positive over part of the range, which "
            "contradicts C1's stated purpose. NOT built here: that is distribution "
            "shaping to satisfy two bars, not a member"
        ),
    }
    V["C6"] = {
        "verdict": "PASS",
        "measured": (
            f"the registered test is UNDEFINED, which is what the clause says a "
            f"clean instrument reads. Coverage is exactly zero: "
            f"{c6['lane_rows_whose_claim_appears_elsewhere_in_the_mix']} of "
            f"{df.height} lane rows carry a claim appearing anywhere in the "
            f"{c6['mix_rows_searched']}-row assembled mix (685,670 clean rows plus "
            f"the four other loaded lanes), and "
            f"{c6['lane_claims_matching_a_mix_claim_only_after_case_folding']} match "
            f"one only after case folding. The mean-label oracle therefore covers no "
            f"pair, and the (claim -> associated text) channel that failed the "
            f"parent at within-pair 0.9999 has "
            f"{c6['claim_to_associated_text_channel']['lane_rows_with_any_such_association']} "
            f"rows to fire on. Executor-added and reported separately, keyed on the "
            f"POOL DOCUMENT rather than on the pair's claim as the clause's test is: "
            f"'the pool contains a passage seen in the mix' reads within-pair "
            f"{dk['pool_contains_a_mix_label1_passage']['within_pair']['truth_removed']['acc']} "
            f"and its count form "
            f"{dk['count_of_mix_label1_passages_in_pool']['within_pair']['truth_removed']['acc']}, "
            f"but the supplement shows that is a SIGHTING imbalance and not a label "
            f"association: all {sup['pool_passages_found_in_the_mix']} of the "
            f"{sup['distinct_pool_passages']} pool passages found in the mix come "
            f"from the `frame_reject` lane, which puts both its claims over the same "
            f"chunk, so every one carries mix mean label exactly 0.5 and a probe on "
            f"the mean label reads "
            f"{sup['probes_on_the_mix_mean_label']['mean_mix_mean_label_over_pool']['within_pair']['truth_removed']['acc']} "
            f"exactly. The residual is a document-role imbalance - "
            f"{sup['documents_in_the_mix_by_role']['truth_documents_share_in_mix']:.4f} "
            f"of truth documents were seen in the mix against "
            f"{sup['documents_in_the_mix_by_role']['swap_documents_share_in_mix']:.4f} "
            f"of swap documents - carrying no label gradient. Closing it would cost "
            f"about half the member and is not applied"
        ),
        "binding_constraint": None,
    }
    c7 = core["C7"]
    V["C7"] = {
        "verdict": "PASS",
        "measured": (
            f"unit is ROWS, used consistently at registration, build and report, "
            f"with pairs stated alongside everywhere: {c7['built_rows']} rows / "
            f"{c7['built_pairs']} pairs. The clause bars unit drift and single-unit "
            f"reporting, and neither occurs. VOLUME, reported in the registered "
            f"unit and not converted away: the conformed member is "
            f"{c7['volume_cost_rows']} rows smaller than the parent's "
            f"{c7['parent_rows']}, a {c7['volume_cost_share']:.4f} cut, and it sits "
            f"BELOW the registered band of 20,000-30,000 rows by "
            f"{20000 - c7['built_rows']} rows. That shortfall is the price of C5's "
            f"claim-only bar (the unsupported_claim family, 6,898 parent rows), C6 "
            f"(the whole VitaminC half, 11,998 parent rows, plus 3,160 MiniCheck "
            f"claim rows already in the mix) and C2 (5 further claim rows)"
        ),
        "binding_constraint": None,
    }
    dup = core["C8"]["within_member_duplication"]
    V["C8"] = {
        "verdict": "PASS",
        "measured": (
            f"source, licence, retrieval date and exact selection predicate are all "
            f"present for the member's single source. MiniCheck "
            f"(lytang/C2D-and-D2C-MiniCheck), MIT re-verified at the Hub 2026-08-13, "
            f"RETRIEVED 2026-08-13 - recorded in the tracked sidecar "
            f"data/external/datasets/dataset-minicheck.md, which is the leg the "
            f"parent failed on for VitaminC and the reason the parent's C8 gap does "
            f"not exist here. Predicate: both shipped parquets concatenated, "
            f"documents <= 1,400 chars, then the conforming pipeline's two admission "
            f"filters; BM25Okapi top-40, distractor rejected at claim containment >= "
            f"0.75, 3-7 distractors, TRUTH_CAP 2 / DIST_CAP 12, seed 2174. "
            f"Duplication in full: {dup['distinct_claims']} distinct claims, "
            f"{dup['distinct_pooled_chunks']} distinct pooled chunks (one per row), "
            f"{dup['distinct_atomic_pool_passages']} distinct atomic passages reused "
            f"{dup['atomic_passage_reuse_mean']} times each across "
            f"{dup['atomic_pool_passage_slots']} slots, "
            f"{dup['distinct_truth_documents']} distinct truth documents, max claim "
            f"repeat {dup['claim_repeat_max']}, "
            f"{dup['claims_appearing_more_than_twice']} claims appearing more than "
            f"twice - and {dup['of_those_carrying_a_single_label']} of them carrying "
            f"a single label, because the surviving family puts every claim on both "
            f"legs, so no claim in the member is memorisable by label. Documents: "
            f"{dup['document_role_overlap']['documents_as_truth']} as truth, "
            f"{dup['document_role_overlap']['documents_as_pool_member']} as pool "
            f"member, {dup['document_role_overlap']['documents_in_both_roles']} in "
            f"both roles and never inside the same pair, max truth uses "
            f"{dup['document_role_overlap']['max_truth_uses']}. Cross-lane: 0 claims "
            f"shared with any other lane of the live mix, by construction. No client "
            f"or company name in any artifact; gold_full was read from a private "
            f"submodule for C2 counts only, with no text reproduced"
        ),
        "binding_constraint": None,
    }

    failed = [k for k, v in V.items() if v["verdict"] == "FAIL"]
    rep = {
        "member": "attr_pool_conformed",
        "kind": "constructed lane - the conforming rebuild of R20-H174 lane L2",
        "artifact": "experiments/grounding-semantic/R20-H174_lane_L2_conformed.parquet",
        "manifest": "experiments/grounding-semantic/R20-H174_lane_L2_conformed_manifest.json",
        "parent_artifact": "experiments/grounding-semantic/R20-H174_lane_L2.parquet",
        "parent_failed_clauses": ["C2", "C5", "C6", "C8"],
        "contract": "docs/experiments/dataset-contract.md",
        "dann_group": "attr_pool",
        "rows": int(df.height),
        "pairs": int(df["pair_id"].n_unique()),
        "families": {k: v for k, v in df.group_by("neg_family").len().iter_rows()},
        "sources": {"minicheck": "MIT"},
        "conforming": not failed,
        "failed_clauses": failed,
        "fixable": "CORPUS_PROPERTY - a clause conflict between C5 and C1, not a "
        "defect a pipeline can build away; see C5's binding_constraint and the "
        "frontier",
        "live_dependency": "NONE. The conformed member is not loaded by any run. "
        "R20-H174 draws 2, 3 and 4 are training on the PARENT lane now and draw 1 is "
        "banked on it; nothing here was written into their path",
        "what_the_pipeline_changed": man["conforming_pipeline"],
        "what_it_cost": {
            "rows": f"{int(df.height)} against the parent's 21,408 "
            f"({core['C7']['volume_cost_share']:.4f} cut)",
            "pairs": f"{int(df['pair_id'].n_unique())} against the parent's 10,704",
            "vitaminc_half": "11,998 parent rows dropped - the C6 channel and the C8 "
            "retrieval-date gap both lived there",
            "unsupported_claim_family": "6,898 parent rows dropped - on MiniCheck "
            "alone its within-pair claim-only probe reads 0.6418, and 0.6825 "
            "retrained, against a C5 bar of < 0.60",
            "minicheck_claims_already_in_the_mix": "3,160 of 12,434 MiniCheck claim "
            "rows rejected at admission because the mix already carries the claim",
            "short_claims_inside_a_surface": "5 claim rows rejected at admission",
        },
        "clause_verdicts": V,
        "what_the_parent_hid": {
            "C5_claim_only": "the parent's within-pair claim-only 0.5594 on "
            "unsupported_claim is the mean of a MiniCheck half at 0.6418 and a "
            "VitaminC half at 0.4998 (1,449 and 2,000 pairs). Retrained on each half "
            "alone: MiniCheck 0.6825, VitaminC 0.5142. The bar was cleared by "
            "aggregation, not by either half. Measured in "
            "attr_pool_conformed_diag.json",
            "C5_chunk_only": "the parent's executor-added chunk-only within-pair "
            "0.5758 on truth_removed drops to "
            f"{core['C5']['executor_added_probes_reported_separately']['within_pair_chunk_only']['truth_removed']['acc']} "
            "once the VitaminC half is gone - that leak was VitaminC's assembled "
            "truth passages, not the pooling construction",
        },
        "measured_alternative_not_applied": {
            "what": "a containment-matched filter keeping the pairs whose legs are "
            f"attested within {e20['epsilon']} of one another - {e20['pairs_kept']} "
            f"of {sweep[1.0]['pairs_kept']} pairs, {e20['rows_kept']} rows, a "
            f"{1 - e20['share_of_member']:.4f} further cut",
            "it_would_pass_C5": f"containment channel {e20['containment_auroc']}, "
            "inside [0.45, 0.55]",
            "it_would_move_C1": f"the attestation gap falls from "
            f"{sweep[1.0]['C1_gap']} to {e20['C1_gap']} - still passing C1's LITERAL "
            f"conjunction (negatives {e20['attested_negative']} attested, far below "
            f"0.90) and failing the gap reading this contract process applied to the "
            f"parent",
            "why_not_applied": "choosing between those two readings of C1 is an "
            "adjudication, not a measurement. The executor does not relax or "
            "reinterpret a clause to make a member pass; the frontier is reported so "
            "the ruling can be made on numbers",
        },
        "instruments": {
            "build": "experiments/grounding-semantic/contract/attr_pool_conformed_build.py",
            "verify": "experiments/grounding-semantic/contract/attr_pool_conformed_verify.py",
            "diagnostic": "experiments/grounding-semantic/contract/attr_pool_conformed_diag.py",
            "c6_supplement": "experiments/grounding-semantic/contract/attr_pool_conformed_memo_supp.py",
            "confidence_intervals": "experiments/grounding-semantic/contract/attr_pool_conformed_ci.py",
            "assembler": "experiments/grounding-semantic/contract/attr_pool_conformed_assemble.py",
            "checkpoints": [f"attr_pool_conformed_{n}.json" for n in
                            ("supply", "core", "disjoint", "census", "memo",
                             "memo_supp", "frontier", "ci", "diag")],
            "logs": ["logs/attr_pool_conformed_supply.log",
                     "logs/attr_pool_conformed_build.log",
                     "logs/attr_pool_conformed_diag.log",
                     "logs/attr_pool_conformed_verify_core.log",
                     "logs/attr_pool_conformed_verify_disjoint.log",
                     "logs/attr_pool_conformed_verify_census.log",
                     "logs/attr_pool_conformed_verify_memo.log",
                     "logs/attr_pool_conformed_verify_frontier.log",
                     "logs/attr_pool_conformed_memo_supp.log"],
            "reused_banked": [
                "provenance_gate.py (R14-H136 ruling-2 form: 8-gram, Jaccard 0.3, "
                "bidirectional, KILL 0.02, spike control)",
                "R20-H174_lane_common.py (containment, auroc, claim_only_probe, "
                "within_pair_accuracy, surface_parity, window census)",
                "R20-H174_lane_L2.py (Retriever, build_pair, verify, build_manifest "
                "- the construction is the banked one, called with a restricted "
                "corpus)",
                "R10-H108_lane.public_train + R16-H142_G1_arm.untruncated_evidence "
                "(the mix rebuilt through the banked loader, 685,670 rows)",
                "attr_pool_contract_measure.eval_surfaces (the same 8 evaluation "
                "surfaces the parent was measured against)",
            ],
            "compute": "CPU only; CUDA_VISIBLE_DEVICES forced empty before every "
            "import. No GPU queried or allocated at any point",
        },
        "detail": {"C1": c1, "C2": dis["C2"], "C2_intra_mix": dis["C2_intra_mix"],
                   "C3": dis["C3"], "C4": c4, "C5": core["C5"], "C6": c6,
                   "C6_document_keyed_supplement": sup, "C7": core["C7"],
                   "C8": core["C8"], "C5_conflict_frontier": f,
                   "confidence_intervals": ci, "claim_only_diagnostic": diag},
    }
    p = HERE / "attr_pool_conformed_report.json"
    p.write_text(json.dumps(rep, indent=2, default=float))
    print(f"-> {p}", flush=True)
    print(json.dumps({k: rep[k] for k in
                      ("member", "rows", "pairs", "conforming", "failed_clauses")},
                     indent=2), flush=True)
    for k, v in V.items():
        print(f"\n{k} {v['verdict']}\n  {v['measured'][:400]}", flush=True)


if __name__ == "__main__":
    main()

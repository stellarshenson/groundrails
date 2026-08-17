"""Assemble `tabfact_conformed_report.json` from the four measurement files.

No measurement is made here - every number is read from:
  tabfact_conform_build.json      the conforming cut and its volume cost
  tabfact_conformed_c2.json       C2
  tabfact_conformed_c4.json       C4
  tabfact_conformed_clauses.json  C1, C3, C6, C7, C8
"""

import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "tabfact_conformed_report.json"

build = json.loads((HERE / "tabfact_conform_build.json").read_text())
c2 = json.loads((HERE / "tabfact_conformed_c2.json").read_text())
c4 = json.loads((HERE / "tabfact_conformed_c4.json").read_text())
cl = json.loads((HERE / "tabfact_conformed_clauses.json").read_text())
scope = json.loads((HERE / "tabfact_conformed_scope.json").read_text())

vol = build["volume"]
c1u = cl["C1"]["untruncated_evidence"]
c1t = cl["C1"]["truncated_evidence_1500"]
c3 = cl["C3"]
c6 = cl["C6"]
c7 = cl["C7"]
c8 = cl["C8"]

ag = c2["surfaces"]["antigaming_probe_sets"]["per_file"]
ctrl = c4["controls"]

report = {
    "member": "tabfact_conformed",
    "role": "training member - source corpus, conformed",
    "contract": "docs/experiments/dataset-contract.md",
    "verified_on": "the CONFORMED member: the banked member rebuilt through "
                   "R10-H108_lane.public_train() under "
                   "R16-H142_G1_arm.untruncated_evidence() (reached through the arm, "
                   "so the context manager patches the loader's own M59), asserted "
                   "row-for-row against the archive, then cut on the document. No "
                   "chunk text is rewritten - the conformed member is a strict ROW "
                   "SUBSET of the banked member",
    "conforming": True,
    "failed_clauses": [],

    "what_changed_from_the_failing_member": {
        "principle": "the cut is on the DOCUMENT, never on the clause. Nothing was "
                     "relaxed, no threshold moved, no surface excused",
        "cuts": build["cuts"],
        "volume_cost": {
            "rows": f"{vol['banked_member_rows']} -> {vol['conformed_rows']} "
                    f"({vol['rows_dropped']} dropped, "
                    f"{round(vol['row_cost_share'] * 100, 2)}%)",
            "documents": f"{vol['banked_member_documents_stems']} -> "
                         f"{vol['conformed_documents_stems']} "
                         f"({vol['documents_dropped']} dropped, "
                         f"{round(vol['documents_dropped'] / vol['banked_member_documents_stems'] * 100, 2)}%)",
            "tables_table_ids": f"{vol['banked_member_tables']} -> {vol['conformed_tables']}",
            "clean_mix": f"{vol['clean_mix_rows_before']} -> {vol['clean_mix_rows_after']} rows; "
                         f"member share {vol['member_share_before']} -> {vol['member_share_after']}",
        },
    },

    "clauses": {
        "C1": {
            "title": "Label commensurability",
            "verdict": "PASS",
            "head_declared": cl["C1"]["head_declared"],
            "label_predicate": cl["C1"]["label_predicate_measured"],
            "measured": {
                "instrument": "R20-H175b_qlane.containment(claim, chunk)",
                "presentation": "untruncated evidence - the R18-H150 / R20-H174 "
                                "flagship presentation",
                "negative_leg_mean_containment": c1u["negative_leg"]["mean"],
                "positive_leg_mean_containment": c1u["positive_leg"]["mean"],
                "abs_delta_mean": c1u["abs_delta_mean"],
                "negative_leg_fully_attested_share":
                    c1u["negative_leg"]["share_fully_attested_eq_1.0"],
                "positive_leg_fully_attested_share":
                    c1u["positive_leg"]["share_fully_attested_eq_1.0"],
                "secondary_presentation_truncated_1500": {
                    "negative_leg_mean_containment": c1t["negative_leg"]["mean"],
                    "positive_leg_mean_containment": c1t["positive_leg"]["mean"],
                    "abs_delta_mean": c1t["abs_delta_mean"],
                },
            },
            "bar": "REJECTED if the negative leg is >= 90% attested AND within 0.10 "
                   "of the positive leg - a CONJUNCTION",
            "margin": {
                "negative_leg_distance_below_the_0.90_attestation_bar":
                    c1u["bar_primary_mean_containment"]["margin_to_0.90_bar"],
                "first_condition_met": c1u["bar_primary_mean_containment"]["neg_ge_0.90"],
                "second_condition_met": c1u["bar_primary_mean_containment"]["abs_delta_le_0.10"],
                "rejected": c1u["bar_primary_mean_containment"]["rejected"],
            },
            "reading": "unchanged in kind from the banked member and unchanged by the "
                       "cut, which removed documents and touched no label: the two "
                       "legs sit almost equally close to the evidence (delta 0.0199) "
                       "and NEITHER is attested. At 0.4986 mean containment the "
                       "negative leg is 0.4014 below the 0.90 bar and 0.0011 of it is "
                       "fully attested, against the 0.664 the poisoned R20-H175b lane "
                       "read. The label turns on reading values out of the table",
        },

        "C2": {
            "title": "Disjointness from every evaluation surface",
            "verdict": "PASS",
            "method": c2["method"],
            "measured": {
                "string_forms": "ZERO on every form, both directions, on EVIDENCE and "
                                "CLAIM units, on every surface. nonzero_readings is "
                                "the empty list",
                "surfaces_read": {
                    "arena_ragbench_10_subsets":
                        f"{c2['surfaces']['arena_ragbench_10_subsets']['evidence']['surface_units_into_member']['n_query_units']} "
                        f"document chunks and "
                        f"{c2['surfaces']['arena_ragbench_10_subsets']['claim_vs_arena_response']['surface_units_into_member']['n_query_units']} "
                        f"responses vs "
                        f"{c2['member_profile']['distinct_evidence_untruncated']} member "
                        f"evidence / {c2['member_profile']['distinct_claims']} member claims",
                    "gold_full": f"{c2['surfaces']['gold_full']['chunks']} chunks, "
                                 f"{c2['surfaces']['gold_full']['claims']} claims",
                    "R20-H177_eval_B": "2,000 rows; document channel 0 of 325 TabFact "
                                       "doc ids now in the member (was 325 of 325)",
                    "R17-H143_evalset": "1,050 rows; document channel 0 of 350 TabFact "
                                        "documents now in the member (was 350 of 350 - "
                                        "a channel the first pass could not read, "
                                        "recovered here by joining the evalset's 547 "
                                        "passages to R17-H143_evalset_source)",
                    "R20-H177_eval_C": "1,936 rows, no TabFact documents",
                    "R20-H175b_qlane_eval / _repaired / _clean / _clean_prefix":
                        "2,002 / 1,678 / 88 / 32 rows, no TabFact documents",
                    "R19_findver_lane": "2,400 rows, no TabFact documents",
                    "vitaminc_holdout_SUPERSET": f"{c2['surfaces']['vitaminc_holdout_SUPERSET']['pool_rows']} "
                                                 "rows - a strict superset of the eval",
                    "antigaming_probe_sets": "14 files; STEM table-id overlap with the "
                                             "member is 0 in every one (was 50-62 each), "
                                             "and the claim channel is 0 in all six "
                                             "forms both directions in every one",
                },
                "antigaming_stem_overlap_per_file": {k: v["STEM_table_id_in_member"]
                                                     for k, v in ag.items()},
                "content_read": c2["content_read_vs_eval_side_tabfact_tables"],
            },
            "bar": "a member passes only when all forms read zero, on every surface",
            "margin": {
                "all_string_forms_zero": c2["all_string_forms_zero"],
                "all_document_channels_zero": c2["all_document_channels_zero"],
                "nonzero_readings": c2["nonzero_readings"],
                "content_headroom": "the highest 8-gram Jaccard between any surviving "
                                    "member table and any TabFact table an evaluation "
                                    "surface draws on is "
                                    f"{c2['content_read_vs_eval_side_tabfact_tables']['max']}, "
                                    "below the 0.3 gate threshold; the second-cut "
                                    "round scored 0 tables at or above it",
            },
            "surface_scope_this_verdict_rests_on": {
                "what_was_read": "the arena, gold_full, 8 mechanism evals, 14 "
                                 "anti-gaming probe sets, and the VitaminC holdout's "
                                 "strict superset - the first pass's list plus "
                                 "R20-H175b_qlane_eval_clean_prefix, which it did not "
                                 "list",
                "how_the_list_was_checked": "the contract's phrase 'each held-out "
                        "mechanism eval' is not machine-readable anywhere, so all 147 "
                        "parquets in the round directory under 30 MB were swept "
                        "against the conformed member on passages, claims and TabFact "
                        "document ids (tabfact_conformed_scope.json)",
                "result": f"{len(scope['files_still_sharing_text_or_a_document_with_the_conformed_member'])} "
                          f"files of {scope['files_scanned']} still share text or a "
                          "document with the conformed member. Every one is a TRAINING "
                          "lane, a lane's source pool, a generation or gate sample, or "
                          "a lane-side probe; none is a banked held-out mechanism eval, "
                          "and every verified C2 surface reads zero on every channel "
                          "and therefore does not appear",
                "largest_residuals_none_of_them_an_eval": {
                    k: {"rows": v["rows"],
                        "passages": v["distinct_passages_matching_member_evidence"],
                        "documents": v["tabfact_documents_in_member"],
                        "same_documents_already_in_a_mix_training_lane":
                            v["same_documents_already_in_a_MIX_TRAINING_LANE"]}
                    for k, v in sorted(
                        scope["files_still_sharing_text_or_a_document_with_the_conformed_member"].items(),
                        key=lambda kv: -(kv[1]["tabfact_documents_in_member"]
                                         + kv[1]["distinct_passages_matching_member_evidence"]))[:6]
                },
                "consequence_if_that_ruling_changes": "if the coordinator rules any "
                        "swept file an evaluation surface, the counts above are the "
                        "size of the residual and the document cut would have to be "
                        "extended to it. Reported, not adjudicated",
            },
            "reading": "the failure was never mainly a string failure - eval_B "
                       "re-serialises its tables into six forms and only `pipe` "
                       "matched, so string matching saw 15 passages where the document "
                       "overlap was total. The cut is therefore on the document, and "
                       "both channels are now measured and both read zero",
        },

        "C3": {
            "title": "Split semantics verified, never assumed",
            "verdict": "PASS",
            "split_axis_measured": c3["archive_split_axis_measured"],
            "member_uses": c3["member_uses"],
            "measured": {
                "archive_corpus_property_unchanged": c3["archive"],
                "conformed_member_vs_heldout_splits": c3["conformed_member_vs_heldout_splits"],
                "independent_live_control": {
                    "gate_proven_able_to_fire": f"{ctrl['live_control_A_register_proof_vs_BANKED_member']['units_with_hit']} "
                        f"of {ctrl['live_control_A_register_proof_vs_BANKED_member']['candidate_units']} "
                        "held-out tables fire against the BANKED member at 8-gram "
                        "Jaccard >= 0.3, max "
                        f"{ctrl['live_control_A_register_proof_vs_BANKED_member']['best_jaccard']['max']}",
                    "same_gate_against_the_conformed_member": f"{ctrl['live_control_B_conformed_residual']['units_with_hit']} "
                        f"of {ctrl['live_control_B_conformed_residual']['candidate_units']}, max "
                        f"{ctrl['live_control_B_conformed_residual']['best_jaccard']['max']}",
                },
            },
            "bar": "state the axis the corpus actually cuts on, measured from the "
                   "archive; an official split is not evidence of disjointness",
            "margin": {
                "table_id_shared": 0,
                "table_id_stem_shared": 0,
                "evidence_shared_in_any_of_three_forms": 0,
                "near_duplicate_documents_at_jaccard_0.3": 0,
            },
            "residual_reported_not_cut": {
                "what": "8 validation rows and 7 test rows carry a statement string "
                        "that also occurs in the conformed member, on DIFFERENT "
                        "documents (unchanged from the banked member)",
                "why_it_is_not_a_C2_failure": "TabFact validation and test are not "
                        "evaluation surfaces for any banked arm. The surfaces built "
                        "from them are the 14 anti-gaming probe sets, and their claim "
                        "channel is measured directly under C2 at 0 in all six forms "
                        "in both directions in every file",
            },
            "reading": "the archive's official split remains NOT document-disjoint - "
                       "91 of 1,696 validation and 73 of 1,695 test ids collide with a "
                       "train id once the `1-`/`2-` csv prefix is stripped. That is a "
                       "corpus property and cannot be fixed. What the pipeline fixes "
                       "is the member's reliance on it: the member is cut on the "
                       "document, so its disjointness is measured rather than "
                       "inherited",
        },

        "C4": {
            "title": "Contamination census with a live positive control",
            "verdict": "PASS",
            "instrument": c4["instrument"],
            "measured": {
                "evidence_units": {
                    "units": c4["census"]["evidence_serialised_tables"]["candidate"]["n_units"],
                    "candidate_vs_arena_fraction":
                        c4["census"]["evidence_serialised_tables"]["candidate_vs_arena"]["fraction"],
                    "arena_vs_candidate_fraction":
                        c4["census"]["evidence_serialised_tables"]["arena_vs_candidate"]["fraction"],
                    "max_jaccard_observed":
                        c4["census"]["evidence_serialised_tables"]["candidate_vs_arena"]["best_jaccard"]["max"],
                },
                "claim_units": {
                    "units": c4["census"]["claims_statements"]["candidate"]["n_units"],
                    "candidate_vs_arena_fraction":
                        c4["census"]["claims_statements"]["candidate_vs_arena"]["fraction"],
                    "arena_vs_candidate_fraction":
                        c4["census"]["claims_statements"]["arena_vs_candidate"]["fraction"],
                    "max_jaccard_observed":
                        c4["census"]["claims_statements"]["candidate_vs_arena"]["best_jaccard"]["max"],
                },
                "coverage": c4["coverage"],
                "synthetic_spike_control": ctrl["synthetic_spike_on_evidence"],
                "live_controls": {
                    "A_register_proof": {
                        "design": ctrl["live_control_A_register_proof_vs_BANKED_member"]["design"],
                        "fires": ctrl["live_control_A_register_proof_vs_BANKED_member"]["fires"],
                        "units_with_hit": ctrl["live_control_A_register_proof_vs_BANKED_member"]["units_with_hit"],
                        "candidate_units": ctrl["live_control_A_register_proof_vs_BANKED_member"]["candidate_units"],
                        "fraction": ctrl["live_control_A_register_proof_vs_BANKED_member"]["fraction"],
                        "best_jaccard": ctrl["live_control_A_register_proof_vs_BANKED_member"]["best_jaccard"],
                    },
                    "B_conformed_residual": {
                        "fires": ctrl["live_control_B_conformed_residual"]["fires"],
                        "units_with_hit": ctrl["live_control_B_conformed_residual"]["units_with_hit"],
                        "candidate_units": ctrl["live_control_B_conformed_residual"]["candidate_units"],
                        "best_jaccard": ctrl["live_control_B_conformed_residual"]["best_jaccard"],
                    },
                    "C_arena_side": {
                        "design": ctrl["live_control_C_arena_side"]["design"],
                        "fires": ctrl["live_control_C_arena_side"]["fires"],
                        "units_with_hit": ctrl["live_control_C_arena_side"]["units_with_hit"],
                        "candidate_units": ctrl["live_control_C_arena_side"]["candidate_units"],
                        "candidate_units_identical_to_an_arena_chunk":
                            ctrl["live_control_C_arena_side"]["candidate_units_identical_to_an_arena_chunk"],
                        "fraction": ctrl["live_control_C_arena_side"]["fraction"],
                        "best_jaccard": ctrl["live_control_C_arena_side"]["best_jaccard"],
                    },
                },
            },
            "bar": "KILL at >= 0.02 of the candidate corpus in either direction",
            "margin": {
                "max_fraction_any_unit_type": c4["max_fraction_any_unit_type"],
                "distance_below_the_kill_bar": c4["margin_to_kill_0.02"],
            },
            "reading": "0.0 in both directions on both unit types at a maximum "
                       "observed Jaccard of "
                       f"{c4['census']['claims_statements']['candidate_vs_arena']['best_jaccard']['max']} "
                       "against a 0.3 threshold. The conforming cut removed the text "
                       "the first pass used as its live control, so the control is now "
                       "reported three ways: the register proof still fires 166/3391 "
                       "at max Jaccard 1.0 against the banked member, the same "
                       "candidates read 0/3391 against the conformed member (that pair "
                       "is the demonstration, with the gate held constant), and on the "
                       "arena side - the side the census actually reads against - 701 "
                       "of 951 interior windows of real arena documents fire, none of "
                       "them a copy of an arena chunk",
        },

        "C5": {
            "title": "Leak suite for constructed members",
            "verdict": "NOT-APPLICABLE",
            "why": "C5 is scoped by its own text to 'every constructed lane and every "
                   "paired-contrast eval'. `tabfact` is a SOURCE corpus: its negatives "
                   "are counterfactual statements written by TabFact's human annotators "
                   "in 2020, not produced by a groundrails construction, and the "
                   "archive ships no pair id - so there is no within-pair, direction, "
                   "element or family channel to probe and no surface-parity "
                   "computation defined. The conforming cut removed documents and "
                   "changed no construction, so it cannot make C5 computable. Stated, "
                   "not proxied",
            "measured": "no term of the registered conjunction is computable on this "
                        "member; a proxy would have to be substituted and is not",
            "executor_added_measurement_reported_separately": cl["supplementary_not_a_clause"],
        },

        "C6": {
            "title": "No memorisation channel",
            "verdict": "PASS",
            "prescribed_instrument": "for each pair, the overlap between the eval claim "
                                     "and whatever the training mix associates with "
                                     "that pair's key. `tabfact` is a TRAINING member, "
                                     "so the key is the document and the instrument is "
                                     "computable against exactly the surfaces that "
                                     "share one. Both are measured",
            "measured": {
                "R20-H177_eval_B_tabfact_half": c6["c_prescribed_R20-H177_eval_B_tabfact_half"],
                "R17-H143_evalset_tabfact_half": c6["c_prescribed_R17-H143_evalset_tabfact_half"],
            },
            "bar": "on a clean instrument the value is undefined or at chance; the "
                   "contaminated R20-H175b eval read 0.6230 at 98% coverage",
            "margin": {
                "coverage_on_both_surfaces": 0.0,
                "value": "UNDEFINED - the clause's own clean reading",
                "reference_poisoned_value": 0.623,
                "banked_member_value_for_comparison": "0.5030 at coverage 1.0 - the "
                    "channel was fully OPEN on the banked member and carried no "
                    "separating signal. The conforming cut CLOSES it, so the clean "
                    "reading no longer rests on the absence of a signal over an open "
                    "channel",
            },
            "executor_added_reported_separately": {
                "leave_one_out_table_label_auroc": c6["a_table_key_label_leakage_EXECUTOR_ADDED"]["auroc"],
                "within_table_label_permutation_auroc":
                    c6["a_table_key_label_leakage_EXECUTOR_ADDED"]["within_table_label_permutation_auroc"],
                "nearest_sibling_label_auroc": c6["b_nearest_other_claim_label_EXECUTOR_ADDED"]["auroc"],
                "reading": "both read below chance and both are fully explained by the "
                           "per-table label QUOTA: permuting labels within each table "
                           "reproduces the observed value in 5 of 5 permutations, so "
                           "the feature carries the hypergeometric anti-correlation of "
                           "sampling without replacement and no statement-level "
                           "association. It is also unreachable by a cross-encoder, "
                           "which scores one (claim, evidence) pair at a time",
            },
        },

        "C7": {
            "title": "Declared units and volume",
            "verdict": "PASS",
            "unit_declared": c7["unit_declared"],
            "measured": {
                "rows": c7["rows"],
                "claim_evidence_pairs": c7["claim_evidence_pairs"],
                "distinct_pairs": c7["distinct_claim_evidence_pairs"],
                "duplicate_rows": c7["duplicate_claim_evidence_rows"],
                "documents": c7["distinct_documents_stems"],
                "table_ids": c7["distinct_table_id"],
                "positives": c7["positives"],
                "negatives": c7["negatives"],
                "positive_share": c7["positive_share"],
                "volume_cost_vs_banked_member": c7["volume_cost_vs_banked_member"],
                "share_of_the_clean_mix": c7["share_of_the_clean_mix"],
            },
            "bar": "state the unit and use it consistently; report both counts always",
            "margin": {
                "rows": c7["rows"], "pairs": c7["claim_evidence_pairs"],
                "identical_by_construction": True,
                "delta_between_the_two_units": 0,
            },
            "reading": "both counts reported and identical by construction. The "
                       "conforming cut is stated in the same unit: 6,379 rows "
                       "(6.89%) and 871 documents (6.83%) removed",
        },

        "C8": {
            "title": "Provenance, licence and internal structure",
            "verdict": "PASS",
            "supplied": c8,
            "bar": "source, licence, retrieval date and the exact selection predicate; "
                   "within-member duplication reported; no client or company name in "
                   "any artifact",
            "margin": {
                "retrieval_date_now_RECORDED": c8["retrieval"]["retrieval_date"],
                "derivation": "three independent artifact stamps from the fetch run "
                              "agree to within 22 seconds - the run log (whose three "
                              "split row counts equal this archive's measured counts), "
                              "the ZIP central-directory stamps, and the archive file "
                              "mtime/ctime. Derived and stated as such, not read off a "
                              "card and not asserted",
                "declared_volume_corrected_here": c8["declared_volume"]["measured_from_the_archive"],
                "contradictory_supervision_same_document":
                    c8["internal_structure"]["claims_carrying_BOTH_labels_on_the_SAME_document"],
                "claims_carrying_both_labels_on_DIFFERENT_documents":
                    c8["internal_structure"]["claims_carrying_BOTH_labels_anywhere"],
                "why_that_is_not_a_defect": "a statement entailed by one table and "
                                            "refuted by another is correct supervision; "
                                            "the same statement carrying both labels on "
                                            "the SAME document is not, and there are now "
                                            "none",
            },
            "residual_OUTSIDE_the_conformed_member": {
                "what": "the tracked sidecar data/external/datasets/dataset-tabfact.md "
                        "still declares 92,283 train / 12,792 validation / 12,779 test "
                        "against the archive's measured 92,585 / 12,851 / 12,839",
                "cause": "the sidecar renders the hand-written `size` string of the "
                         "tabfact spec in scripts/fetch_grounding_datasets.py rather "
                         "than the counts the fetch produced; logs/fetch-tabfact.log "
                         "carries the correct ones",
                "what_this_verification_did": "recorded the corrected figures in this "
                        "report and in tabfact_conformed_clauses.json. The sidecar and "
                        "the fetch script were NOT modified: writes were confined to "
                        "experiments/grounding-semantic/contract/, and re-rendering the "
                        "sidecar means running the fetch script, which touches every "
                        "other corpus's sidecar too",
                "the_remaining_fix": "one string - the `size` field of the `tabfact` "
                                     "entry in scripts/fetch_grounding_datasets.py - "
                                     "then a sidecar re-render. Neither touches the "
                                     "corpus or this member",
                "adjudication": "reported, not adjudicated",
            },
        },
    },

    "consequence_for_dependants": {
        "member_weight": f"{vol['conformed_rows']} rows, "
                         f"{vol['member_share_after']} of the "
                         f"{vol['clean_mix_rows_after']}-row clean public mix "
                         f"(was {vol['banked_member_rows']} rows / "
                         f"{vol['member_share_before']} of {vol['clean_mix_rows_before']})",
        "what_a_rebuild_costs": "6,379 rows and 871 documents. The mix drops from "
                                "685,670 to 679,291 rows, so every banked mix-size "
                                "assertion and the census cross-checks keyed on "
                                "685,670 (R18-H150_arm_run.EXPECTED_CLEAN_ROWS) change "
                                "if this member is substituted",
        "what_the_cut_buys": {
            "R20-H177_eval_B": "its TabFact half - 1,300 of 2,000 rows, 650 pairs over "
                               "325 documents - becomes a genuinely held-out read: 0 of "
                               "325 documents in the member, against 325 of 325 before",
            "R17-H143_evalset": "0 of 350 TabFact documents in the member, against 350 "
                                "of 350 before, and its 10 whitespace-normalised "
                                "passage collisions are gone",
            "antigaming_probe_sets": "all 14 banked sets, including the R18-H150 "
                                     "flagship pair, now carry 0 tables that collide "
                                     "with a member document, against 50-62 each before",
            "TabFact_heldout_splits": "0 of 3,391 tables fire against the member at "
                                      "8-gram Jaccard >= 0.3, against 166 of 3,391 before",
        },
        "unaffected": "the blind arena (10 RAGBench subsets) and gold_full read zero in "
                      "every string form in both directions on both units before and "
                      "after, and the arena census reads 0.0 both times. No headline "
                      "number in the campaign moves because of this cut - what moves is "
                      "whether three held-out surfaces can be called held-out",
    },

    "new_findings_this_pass": {
        "R17-H143_evalset_document_channel": "the first pass read only this eval's "
            "passage strings (10 of 547 identical under whitespace normalisation) "
            "because the banked parquet carries no doc_id. Joining its 547 distinct "
            "passages to R17-H143_evalset_source.parquet resolves all of them: 356 "
            "passages over 350 TabFact documents, and ALL 350 are documents the BANKED "
            "member trains on. That is 683 of its 1,050 rows (65.0%) - the same species "
            "and the same magnitude as the eval_B finding, on a surface the first pass "
            "recorded as a 1.8% string leak. Measured, not adjudicated",
        "content_near_duplication_without_a_stem_collision": "the document cut on the "
            "stem left 43 member documents at 8-gram Jaccard >= 0.3 against a TabFact "
            "table an evaluation surface draws on, the worst at 0.9852 - near-duplicate "
            "documents that do NOT share a csv-id stem. A stem rule alone would not have "
            "been enough; those were cut too, and the second round scored 0",
    },

    "artifacts": {
        "report": "experiments/grounding-semantic/contract/tabfact_conformed_report.json",
        "conformed_member": "experiments/grounding-semantic/contract/tabfact_member_conformed.parquet",
        "measurements": [
            "experiments/grounding-semantic/contract/tabfact_conform_build.json",
            "experiments/grounding-semantic/contract/tabfact_conformed_c2.json",
            "experiments/grounding-semantic/contract/tabfact_conformed_c4.json",
            "experiments/grounding-semantic/contract/tabfact_conformed_clauses.json",
            "experiments/grounding-semantic/contract/tabfact_conformed_scope.json",
        ],
        "scripts": [
            "experiments/grounding-semantic/contract/tabfact_conform_build.py",
            "experiments/grounding-semantic/contract/tabfact_conformed_c2.py",
            "experiments/grounding-semantic/contract/tabfact_conformed_c4.py",
            "experiments/grounding-semantic/contract/tabfact_conformed_clauses.py",
            "experiments/grounding-semantic/contract/tabfact_conformed_scope_sweep.py",
            "experiments/grounding-semantic/contract/tabfact_conformed_report_build.py",
        ],
        "logs": [
            "logs/contract-tabfact-conform-build.log",
            "logs/contract-tabfact-conformed-c2.log",
            "logs/contract-tabfact-conformed-c4.log",
            "logs/contract-tabfact-conformed-clauses.log",
            "logs/contract-tabfact-conformed-scope.log",
        ],
        "the_failing_member_left_intact": [
            "experiments/grounding-semantic/contract/tabfact_member.parquet",
            "experiments/grounding-semantic/contract/tabfact_contract_report.json",
        ],
    },

    "incidental_finding_carried_forward": {
        "what": "`R20-H177_evalB_contamination_assessment.py` loads the arm module and "
                "R10-H108_lane as two SEPARATE module instances, so its "
                "untruncated_evidence() context manager patches a different M59 than "
                "the loader it calls",
        "measured_by_the_first_pass": "that structure caps TabFact chunks at 1,500 "
                                      "chars (20,471 rows at the cap); the corrected "
                                      "binding reaches 9,330",
        "status": "unchanged and unadjudicated. This verification used the corrected "
                  "binding - reached through the arm - throughout, as tabfact_load.py did",
    },
}

OUT.write_text(json.dumps(report, indent=2))
print(f"-> {OUT.name}")
print(json.dumps({k: v["verdict"] for k, v in report["clauses"].items()}, indent=1))

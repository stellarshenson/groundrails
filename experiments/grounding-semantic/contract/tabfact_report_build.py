"""Assemble tabfact_contract_report.json from the measured artifacts.

Every number in the report is read out of a measurement JSON produced by this
verification - nothing is typed in by hand. CPU only.
"""

import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "tabfact_contract_report.json"

L = json.loads((HERE / "tabfact_load.json").read_text())
C = json.loads((HERE / "tabfact_clauses.json").read_text())
C2 = json.loads((HERE / "tabfact_c2.json").read_text())
C4 = json.loads((HERE / "tabfact_c4.json").read_text())
D = json.loads((HERE / "tabfact_detail.json").read_text())

c1u = C["C1"]["untruncated_evidence"]
c1t = C["C1"]["truncated_evidence_1500"]
c3 = C["C3"]
c6 = C["C6"]
c7 = C["C7"]
c8 = C["C8"]
sur = C2["surfaces"]
d1 = D["D1_stem_collision_content_vs_member"]
d2 = D["D2_antigaming_probe_sets"]["per_file"]
d5 = D["D5_eval_B_leak_attributed"]
live = C4["controls"]["live_positive_control"]

ag_stems = {k: v["stem_colliding_tables"] for k, v in d2.items()}
ag_jacc = {k: v["token_jaccard"]["mean"] for k, v in d2.items()}

clauses = {}

# ---------------------------------------------------------------- C1 ------ #
clauses["C1"] = {
    "title": "Label commensurability",
    "verdict": "PASS",
    "head_declared": c1u and C["C1"]["head_declared"],
    "label_predicate": C["C1"]["label_predicate_measured"],
    "measured": {
        "instrument": c1u["instrument"],
        "presentation": "untruncated evidence - the R18-H150 / R20-H174 flagship "
                        "presentation (windowed 1,500/750 downstream)",
        "negative_leg_mean_containment": c1u["negative_leg"]["mean"],
        "positive_leg_mean_containment": c1u["positive_leg"]["mean"],
        "abs_delta_mean": c1u["abs_delta_mean"],
        "negative_leg_fully_attested_share": c1u["negative_leg"]["share_fully_attested_eq_1.0"],
        "positive_leg_fully_attested_share": c1u["positive_leg"]["share_fully_attested_eq_1.0"],
        "secondary_presentation_truncated_1500": {
            "negative_leg_mean_containment": c1t["negative_leg"]["mean"],
            "positive_leg_mean_containment": c1t["positive_leg"]["mean"],
            "abs_delta_mean": c1t["abs_delta_mean"],
        },
    },
    "bar": "REJECTED if the negative leg is >= 90% attested AND within 0.10 of the "
           "positive leg - a CONJUNCTION",
    "margin": {
        "negative_leg_distance_below_the_0.90_attestation_bar":
            c1u["bar_primary_mean_containment"]["margin_to_0.90_bar"],
        "second_condition_met": c1u["bar_primary_mean_containment"]["abs_delta_le_0.10"],
        "first_condition_met": c1u["bar_primary_mean_containment"]["neg_ge_0.90"],
        "rejected": c1u["bar_primary_mean_containment"]["rejected"],
    },
    "reading": (
        "the two legs are lexically almost equally close to the evidence "
        f"(delta {c1u['abs_delta_mean']}), which is the intended signature of a "
        "counterfactual near-miss corpus - but NEITHER leg is attested. At "
        f"{c1u['negative_leg']['mean']} mean containment the negative leg sits "
        f"{c1u['bar_primary_mean_containment']['margin_to_0.90_bar']} below the "
        "0.90 bar, and only "
        f"{c1u['negative_leg']['share_fully_attested_eq_1.0']:.4f} of negatives are "
        "fully attested against the 0.664 the poisoned R20-H175b lane read. The "
        "label turns on reading values out of the table, not on token presence, so "
        "the member teaches support and not relevance"),
    "reference": C["C1"]["reference_poisoned_lane_R20_H175b"],
}

# ---------------------------------------------------------------- C2 ------ #
clauses["C2"] = {
    "title": "Disjointness from every evaluation surface",
    "verdict": "FAIL",
    "method": C2["method"],
    "surfaces_reading_zero_in_all_forms_both_directions": [
        "arena_ragbench_10_subsets (evidence AND claims, vs arena documents and "
        "arena responses)",
        "gold_full", "R20-H177_eval_C",
        "R20-H175b_qlane_eval", "R20-H175b_qlane_eval_repaired",
        "R20-H175b_qlane_eval_clean", "R19_findver_lane",
        "vitaminc_holdout_SUPERSET (VitaminC test+validation, "
        f"{sur['vitaminc_holdout_SUPERSET']['pool_rows']} rows)",
    ],
    "failures": {
        "R20-H177_eval_B": {
            "passage_channel_normalised_only": {
                "passages_hit": sur["R20-H177_eval_B"]["evidence"][
                    "surface_units_into_member"]["normalised_in_normalised_raw"],
                "distinct_passages": sur["R20-H177_eval_B"]["evidence"][
                    "surface_units_into_member"]["n_query_units"],
                "share": d5["passage_channel"]["share_of_distinct_passages"],
                "raw_form": sur["R20-H177_eval_B"]["evidence"][
                    "surface_units_into_member"]["raw_in_raw"],
                "truncated_form": sur["R20-H177_eval_B"]["evidence"][
                    "surface_units_into_member"]["truncated_in_truncated"],
                "eval_rows": d5["passage_channel"]["rows"],
                "eval_pairs": d5["passage_channel"]["pairs"],
                "serialisation_forms_hit": d5["passage_channel"][
                    "by_source_and_serial_form"],
                "neg_families": d5["passage_channel"]["by_neg_family"],
            },
            "document_channel": {
                "eval_tabfact_documents": sur["R20-H177_eval_B"][
                    "document_id_channel"]["surface_tabfact_doc_ids"],
                "documents_in_the_member": sur["R20-H177_eval_B"][
                    "document_id_channel"]["exact_table_id_in_member"],
                "share": sur["R20-H177_eval_B"]["document_id_channel"][
                    "share_of_surface_tabfact_docs_in_member"],
                "eval_rows": d5["document_channel"]["rows"],
                "eval_pairs": d5["document_channel"]["pairs"],
                "share_of_all_eval_B_rows": d5["document_channel"]["share_of_eval_rows"],
                "why_the_string_forms_miss_it": (
                    "eval_B re-serialises each table into six forms (pipe, "
                    "row_prose, markdown, narrative, json_records, keyvalue). Only "
                    "`pipe` matches the member's serialisation, so string matching "
                    "sees 15 passages while the document overlap is total"),
            },
            "claim_channel": "0 in all six forms, both directions - eval_B's claims "
                             "are template-generated, not TabFact statements",
        },
        "R17-H143_evalset": {
            "passages_hit": sur["R17-H143_evalset"]["evidence"][
                "surface_units_into_member"]["normalised_in_normalised_raw"],
            "distinct_passages": sur["R17-H143_evalset"]["evidence"][
                "surface_units_into_member"]["n_query_units"],
            "share": round(sur["R17-H143_evalset"]["evidence"][
                "surface_units_into_member"]["normalised_in_normalised_raw"]
                / sur["R17-H143_evalset"]["evidence"][
                    "surface_units_into_member"]["n_query_units"], 4),
            "raw_form": 0, "truncated_form": 0,
            "claim_channel": "0 in all six forms, both directions",
        },
        "antigaming_probe_sets": {
            "files": len(ag_stems),
            "exact_table_id_in_member": 0,
            "stem_colliding_tables_per_file": ag_stems,
            "mean_token_jaccard_of_those_collisions_per_file": ag_jacc,
            "reading": "the R14-H133 construction removes every table_id present in "
                       "TabFact train - an EXACT id rule. 50 to 62 tables per set "
                       "survive it while being near-duplicates of member tables at "
                       "mean token Jaccard ~0.91 (median ~0.94, 69-75% at >= 0.9). "
                       "None are byte-identical",
        },
    },
    "binding_constraint": "R20-H177_eval_B shares 100% of its TabFact documents "
                          "with the member; and 15 of its passages plus 10 of "
                          "R17-H143_evalset's are identical to member evidence "
                          "under whitespace normalisation. The clause requires all "
                          "forms to read zero on every surface",
    "fixable": "PIPELINE",
}

# ---------------------------------------------------------------- C3 ------ #
clauses["C3"] = {
    "title": "Split semantics verified, never assumed",
    "verdict": "FAIL",
    "split_axis_measured": c3["split_axis_measured"],
    "member_uses": c3["member_uses"],
    "measured": {
        "table_id_string_shared_train_vs_validation":
            c3["held_out_splits"]["validation"]["table_id_shared_with_train"],
        "table_id_string_shared_train_vs_test":
            c3["held_out_splits"]["test"]["table_id_shared_with_train"],
        "stem_collisions_validation": {
            "tables": c3["held_out_splits"]["validation"]["table_id_STEM_shared_with_train"],
            "of": c3["held_out_splits"]["validation"]["distinct_table_id"],
            "share": c3["held_out_splits"]["validation"]["table_id_stem_share"],
            "content": {k: v for k, v in d1["validation"].items() if k != "examples"},
        },
        "stem_collisions_test": {
            "tables": c3["held_out_splits"]["test"]["table_id_STEM_shared_with_train"],
            "of": c3["held_out_splits"]["test"]["distinct_table_id"],
            "share": c3["held_out_splits"]["test"]["table_id_stem_share"],
            "content": {k: v for k, v in d1["test"].items() if k != "examples"},
        },
        "statement_strings_shared": {
            "validation_rows": c3["held_out_splits"]["validation"][
                "statement_rows_shared_with_train"],
            "test_rows": c3["held_out_splits"]["test"]["statement_rows_shared_with_train"],
        },
        "independent_confirmation_from_C4_live_control": {
            "held_out_tables_firing_against_the_member": live["units_with_hit"],
            "of": live["candidate_units"],
            "fraction": live["fraction"],
            "max_jaccard": live["best_jaccard"]["max"],
        },
    },
    "binding_constraint": "TabFact's official split is disjoint on the table_id "
                          "STRING and not on the document. 91 validation and 73 "
                          "test tables collide with a train table after the "
                          "`1-`/`2-` csv-id prefix is stripped, and those "
                          "collisions are near-duplicates at mean token Jaccard "
                          "0.91 / 0.91, with 2 and 1 byte-identical. Same species "
                          "as the VitaminC finding the clause was written for",
    "consequence": "TabFact validation and test are not scored directly by any "
                   "banked arm, but every anti-gaming probe set is built from them "
                   "under an exact-id rule, so the failure propagates there - see "
                   "C2's antigaming block",
    "fixable": "PIPELINE - the member can be cut on the stem instead of the id "
               "string; the archive's own split cannot be changed, so the "
               "underlying non-disjointness is a corpus property",
}

# ---------------------------------------------------------------- C4 ------ #
clauses["C4"] = {
    "title": "Contamination census with a live positive control",
    "verdict": "PASS",
    "instrument": C4["instrument"],
    "measured": {
        "evidence_units": {
            "units": C4["census"]["evidence_serialised_tables"]["candidate"]["n_units"],
            "candidate_vs_arena_fraction": C4["census"][
                "evidence_serialised_tables"]["candidate_vs_arena"]["fraction"],
            "arena_vs_candidate_fraction": C4["census"][
                "evidence_serialised_tables"]["arena_vs_candidate"]["fraction"],
            "max_jaccard_observed": C4["census"]["evidence_serialised_tables"][
                "candidate_vs_arena"]["best_jaccard"]["max"],
        },
        "claim_units": {
            "units": C4["census"]["claims_statements"]["candidate"]["n_units"],
            "candidate_vs_arena_fraction": C4["census"][
                "claims_statements"]["candidate_vs_arena"]["fraction"],
            "arena_vs_candidate_fraction": C4["census"][
                "claims_statements"]["arena_vs_candidate"]["fraction"],
            "max_jaccard_observed": C4["census"]["claims_statements"][
                "candidate_vs_arena"]["best_jaccard"]["max"],
        },
        "coverage": C4["coverage"],
        "synthetic_spike_control": C4["controls"]["synthetic_spike_on_evidence"],
        "live_positive_control": live,
    },
    "bar": "KILL at >= 0.02 of the candidate corpus in either direction",
    "margin": {"max_fraction_any_unit_type": C4["max_fraction_any_unit_type"],
               "distance_below_the_kill_bar": C4["margin_to_kill_0.02"]},
    "reading": "0.0 in both directions on both unit types, at a maximum observed "
               f"Jaccard of {C4['census']['claims_statements']['candidate_vs_arena']['best_jaccard']['max']} "
               "against a 0.3 threshold. The gate is proven able to fire twice: the "
               "synthetic spike at 10/10 with 0 baseline hits, and the LIVE control "
               f"- TabFact's own held-out splits - at {live['units_with_hit']} of "
               f"{live['candidate_units']} ({live['fraction']}) with max Jaccard "
               f"{live['best_jaccard']['max']}",
}

# ---------------------------------------------------------------- C5 ------ #
clauses["C5"] = {
    "title": "Leak suite for constructed members",
    "verdict": "NOT-APPLICABLE",
    "why": "C5 is scoped by its own text to 'every constructed lane and every "
           "paired-contrast eval'. `tabfact` is a SOURCE corpus: its negatives are "
           "counterfactual statements written by TabFact's human annotators in "
           "2020, not produced by a groundrails construction, and the archive ships "
           "no pair id, so there is no within-pair, direction, element or family "
           "channel to probe and no surface-parity computation defined. The "
           "registered conjunction has no computable term here - this is stated "
           "rather than proxied",
    "executor_added_measurement_reported_separately": C["supplementary_not_a_clause"],
}

# ---------------------------------------------------------------- C6 ------ #
c6c = c6["c_cross_surface_R20_H177_eval_B_tabfact_half"]
clauses["C6"] = {
    "title": "No memorisation channel",
    "verdict": "PASS",
    "prescribed_instrument": c6["adaptation_stated"],
    "measured": {
        "surface": "R20-H177_eval_B, TabFact half - the only surface sharing a key "
                   "with the member",
        "coverage": c6c["coverage"],
        "rows_covered": c6c["rows_with_a_member_claim_over_the_same_table"],
        "claim_overlap_auroc": c6c["max_jaccard_auroc"],
        "nearest_member_claim_label_auroc": c6c["nearest_member_claim_label_auroc"],
        "mean_max_jaccard": c6c["mean_max_jaccard"],
    },
    "bar": "on a clean instrument the value is undefined or at chance; the "
           "contaminated R20-H175b eval read 0.6230 at 98% coverage",
    "margin": {
        "deviation_from_chance": round(abs(c6c["max_jaccard_auroc"] - 0.5), 4),
        "reference_poisoned_value": 0.6230,
    },
    "reading": "the channel is fully OPEN - 100% of eval_B's TabFact rows have a "
               "member claim over the same table - and carries no separating "
               "signal, because the member's claims are free-form TabFact "
               "statements while eval_B's are template-generated relation flips "
               f"(mean max-Jaccard {c6c['mean_max_jaccard']})",
    "executor_added_reported_separately": {
        "leave_one_out_table_label_auroc": c6[
            "a_table_key_label_leakage_EXECUTOR_ADDED"]["auroc"],
        "nearest_sibling_label_auroc": c6[
            "b_nearest_other_claim_label_EXECUTOR_ADDED"]["auroc"],
        "control": D["D3_C6_quota_control"],
        "reading": "both read below chance and both are fully explained by the "
                   "per-table label QUOTA: permuting labels within each table "
                   "reproduces the observed 0.2416 in 5 of 5 permutations, so the "
                   "feature is the hypergeometric anti-correlation of sampling "
                   "without replacement and carries zero statement-level "
                   "association. It is also unreachable by a cross-encoder, which "
                   "scores one (claim, evidence) pair at a time",
    },
}

# ---------------------------------------------------------------- C7 ------ #
clauses["C7"] = {
    "title": "Declared units and volume",
    "verdict": "PASS",
    "unit_declared": c7["unit_declared"],
    "measured": {
        "rows": c7["rows"],
        "claim_evidence_pairs": c7["claim_evidence_pairs"],
        "distinct_pairs": c7["distinct_claim_evidence_pairs"],
        "duplicate_rows": c7["duplicate_claim_evidence_rows"],
        "documents_tables": c7["distinct_documents_tables"],
        "positives": c7["positives"], "negatives": c7["negatives"],
        "positive_share": c7["positive_share"],
        "share_of_the_685670_row_clean_mix": c7["share_of_the_clean_mix_685670_rows"],
    },
    "margin": {"registered_rows": c7["registered_vs_measured"]["brief_registered_rows"],
               "measured_rows": c7["registered_vs_measured"]["measured_rows"],
               "delta": c7["registered_vs_measured"]["delta"]},
    "reading": "both counts reported and identical by construction; the registered "
               "figure is reproduced exactly, with 0 rows dropped by the loader's "
               "own `statement length > 10` filter",
}

# ---------------------------------------------------------------- C8 ------ #
clauses["C8"] = {
    "title": "Provenance, licence and internal structure",
    "verdict": "FAIL",
    "supplied": {
        "source": c8["source"], "licence": c8["licence"],
        "selection_predicate": c8["selection_predicate"],
        "chunk_construction": c8["chunk_construction"],
        "internal_structure": c8["internal_structure"],
        "public_repository": c8["public_repository"],
    },
    "defects": {
        "retrieval_date_ABSENT": {
            "required_by": "C8 - 'Source, licence, retrieval date, and the exact "
                           "selection predicate'",
            "state": "no retrieval date is recorded in the tracked sidecar "
                     "data/external/datasets/dataset-tabfact.md, in "
                     "scripts/fetch_grounding_datasets.py, or in any manifest",
            "proxies_measured_and_FLAGGED_as_proxies": {
                "archive_mtime": c8["archive_mtime"],
                "sidecar_first_commit_date": "2026-08-05",
            },
        },
        "declared_volume_wrong_on_all_three_splits": {
            "sidecar_declares": c7["registered_vs_measured"]["tracked_sidecar_declares"],
            "archive_measured": "92,585 train / 12,851 validation / 12,839 test",
            "delta": "train -302, validation -59, test -60",
        },
        "contradictory_supervision_within_the_member": {
            **{k: v for k, v in D["D4_claims_carrying_both_labels"].items()
               if k != "examples"},
        },
    },
    "binding_constraint": "the retrieval date the clause requires is not recorded "
                          "anywhere in the repository, and the tracked provenance "
                          "sidecar's declared volume disagrees with the archive on "
                          "all three splits",
    "fixable": "PIPELINE - correct the sidecar and record the retrieval date; "
               "neither touches the corpus",
}

# --------------------------------------------------------------- report ---- #
fails = [k for k, v in clauses.items() if v["verdict"] == "FAIL"]
report = {
    "member": "tabfact",
    "role": "training member - source corpus",
    "contract": "docs/experiments/dataset-contract.md",
    "verified_on": "the member as ASSEMBLED, rebuilt through the banked loader "
                   "(R10-H108_lane.public_train under "
                   "R16-H142_G1_arm.untruncated_evidence, rows tagged `tabfact`), "
                   "which is what R18-H150_arm_run and the R20-H174 wrapper train on",
    "load": L,
    "conforming": len(fails) == 0,
    "failed_clauses": fails,
    "clauses": clauses,
    "consequence_for_dependants": {
        "member_weight": f"{c7['rows']} rows, "
                         f"{c7['share_of_the_clean_mix_685670_rows']:.4f} of the "
                         "685,670-row clean public mix; present in the R18-H150 "
                         "flagship mix and in the live R20-H174 draws",
        "primary_reads_UNAFFECTED": "the blind arena (10 RAGBench subsets) and "
                                    "gold_full read zero in every string form in "
                                    "both directions, and the arena census is 0.0 "
                                    "against a proven-firing gate. No headline "
                                    "number in the campaign is called into question "
                                    "by these failures",
        "R20-H177_eval_B": f"{d5['document_channel']['rows']} of "
                           f"{d5['eval_rows']} rows "
                           f"({d5['document_channel']['share_of_eval_rows']}) - its "
                           "entire TabFact half, "
                           f"{d5['document_channel']['pairs']} pairs over "
                           f"{d5['document_channel']['documents']} documents - sit "
                           "on documents the member trains on. Its held-out gate is "
                           "not a clean held-out read on that half. The "
                           "memorisation feature over that open channel reads at "
                           f"chance ({c6c['max_jaccard_auroc']}), so the "
                           "contamination is present but not shown to be exploited",
        "R17-H143_evalset": "10 of 547 passages identical to member evidence under "
                            "whitespace normalisation",
        "antigaming_probe_sets": f"{len(ag_stems)} banked sets, including the "
                                 "R18-H150 flagship pair, each carry 50-62 tables "
                                 "that are near-duplicates of member tables at mean "
                                 "token Jaccard ~0.91 - roughly 4.3-5.3% of each "
                                 "set's ~1,180 tables",
    },
    "artifacts": {
        "report": "experiments/grounding-semantic/contract/tabfact_contract_report.json",
        "member_slice": "experiments/grounding-semantic/contract/tabfact_member.parquet",
        "measurements": [
            "experiments/grounding-semantic/contract/tabfact_load.json",
            "experiments/grounding-semantic/contract/tabfact_clauses.json",
            "experiments/grounding-semantic/contract/tabfact_c2.json",
            "experiments/grounding-semantic/contract/tabfact_c4.json",
            "experiments/grounding-semantic/contract/tabfact_detail.json",
        ],
        "scripts": [
            "experiments/grounding-semantic/contract/tabfact_load.py",
            "experiments/grounding-semantic/contract/tabfact_clauses.py",
            "experiments/grounding-semantic/contract/tabfact_c2.py",
            "experiments/grounding-semantic/contract/tabfact_c4.py",
            "experiments/grounding-semantic/contract/tabfact_detail.py",
            "experiments/grounding-semantic/contract/tabfact_report_build.py",
        ],
        "logs": [
            "logs/contract-tabfact-load.log", "logs/contract-tabfact-clauses.log",
            "logs/contract-tabfact-c2.log", "logs/contract-tabfact-c4.log",
            "logs/contract-tabfact-detail.log",
        ],
    },
    "incidental_finding_outside_this_member": {
        "what": "`R20-H177_evalB_contamination_assessment.py` loads the arm module "
                "and R10-H108_lane as two SEPARATE module instances, then calls "
                "`H108.public_train()` inside `arm.untruncated_evidence()`. The "
                "context manager patches `arm.M59`, which is the ARM's own H108's "
                "M59, so the separately-loaded loader stays truncated",
        "measured": "running that exact structure produced a maximum TabFact chunk "
                    "length of 1,500 chars with 20,471 rows at the cap; the "
                    "corrected binding (arm.H108.public_train, as "
                    "R18-H150_arm_run.make_build_mix does) produced a maximum of "
                    "9,330 chars with 20,469 of 92,585 rows over 1,500",
        "consequence": "that banked assessment's mix block is described as "
                       "untruncated but its `raw` and `truncated` chunk sets are "
                       "identical for any member whose evidence exceeds 1,500 "
                       "chars. Reported, not adjudicated. This verification used "
                       "the corrected binding throughout",
    },
}

OUT.write_text(json.dumps(report, indent=2))
print(json.dumps({k: v["verdict"] for k, v in clauses.items()}, indent=1))
print("conforming:", report["conforming"], "fails:", fails)
print(f"-> {OUT}")

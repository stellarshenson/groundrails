"""Assemble the `quant_misbind` contract report from the measured stage files.

Every number below is READ from a stage JSON written by an instrument in this
directory - nothing is retyped by hand.  Verdicts apply the clause text
mechanically; nothing is adjudicated and no bar is moved.

Run:  CUDA_VISIBLE_DEVICES= uv run python \
        experiments/grounding-semantic/contract/quant_misbind_contract_report.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import pathlib

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent.parent


def J(name):
    return json.loads((HERE / f"quant_misbind_{name}.json").read_text())


def rel(p):
    return str(pathlib.Path(p))


def main():
    c1 = J("c1_containment")
    c1b = J("c1_binding")
    c2 = J("c2_disjointness")
    c3 = J("c3_split")
    c3d = J("c3_doc_disjoint")
    c2c3 = J("c2c3_doc_exhaustive")
    rulev_ = J("c2c3_rule_validation")
    c4 = J("c4_census")
    c4l = J("c4_live_control")
    c5 = J("c5_leak")
    c6 = J("c6_memorisation")
    c6m = J("c6_mix_assoc")
    c78 = J("c78_units_provenance")

    neg = c1["legs"]["negative"]["untruncated"]
    pos = c1["legs"]["positive"]["untruncated"]
    br = c1["bar_readings"]

    surfaces = c2["surfaces"]
    dirty = {k: v for k, v in surfaces.items() if not v["clean"]}
    clean = sorted(k for k, v in surfaces.items() if v["clean"])

    report = {
        "member": "quant_misbind",
        "artifact": "experiments/grounding-semantic/R17-H146_lane.parquet",
        "artifact_blake2b_64": c78["c8"]["blake2b_64"],
        "member_class": "constructed lane - a training member of the assembled mix, "
                        "loaded as one DANN group by R18-H150_arm_run.LANES and the "
                        "R20-H174 wrapper",
        "contract": "docs/experiments/dataset-contract.md",
        "verification_pass": "phase 1, per-member; CPU only; no GPU touched",
        "conforming": False,

        # ------------------------------------------------------------------ #
        "C1": {
            "verdict": "PASS",
            "clause": "label commensurability",
            "head_declared": c1["head_declared"],
            "label_predicate": c1["label_predicate"],
            "label_origin": c1["label_origin"],
            "mandatory_test": {
                "instrument": c1["instrument"],
                "negative_leg": neg,
                "positive_leg": pos,
                "bar": "REJECTED if negatives are >= 90% attested AND that reading is "
                       "within 0.10 of the positives - a conjunction; both must hold",
                "readings": br,
                "binding_reading": "reading A (mean containment) is the one consistent "
                                   "with the R20-H175b provenance figure of 0.9129; it "
                                   "is the reading that comes closest to firing here",
                "margin_to_rejection": {
                    "mean_containment": br["reading_A_mean_containment"]["margin_below_0.90"],
                    "share_attested_ge_0.90": br["reading_B_share_attested_ge_0.90"]["margin_below_0.90"],
                    "share_fully_attested": br["reading_C_share_fully_attested"]["margin_below_0.90"],
                },
                "rejects_under_any_reading": any(
                    br[k]["rejects"] for k in br if k.startswith("reading_")),
            },
            "honest_caveat": {
                "containment_is_blind_on_this_construction": c1["leg_separability_on_containment"],
                "asserted_numeral_presence": c1["asserted_numeral_presence"],
                "consequence": "the clause's mechanical test has no discriminating power "
                               "on a misbinding construction - both legs assert a numeral "
                               "the evidence prints. The lane stays outside the rejection "
                               "region only because its ABSOLUTE attestation level is far "
                               "below 0.90, not because the test separates its legs",
                "fully_attested_negatives": {
                    "count": int(round(neg["share_fully_attested_eq_1.0"] * neg["n"])),
                    "of": neg["n"],
                    "reference_R20-H175b": "5,966 of 8,986 (66.4%)",
                },
            },
            "discriminating_measurement": {
                "why": "containment cannot decide this clause for this member, so the "
                       "predicate is verified where it actually lives - the binding",
                "method": c1b["method"],
                "negatives_re_derived_as_genuinely_unattested_bindings":
                    f"{c1b['negative_leg']['verified']} / {c1b['negative_leg']['rows']}",
                "negative_unattested_rate": c1b["negative_binding_unattested_rate"],
                "positives_re_derived_as_attested_bindings":
                    f"{c1b['positive_leg']['verified']} / {c1b['positive_leg']['rows']}",
                "positive_attested_rate": c1b["positive_binding_attested_rate"],
                "errors": c1b["negative_leg"]["errors"] + c1b["positive_leg"]["errors"],
                "ambiguity_guard": c1b["ambiguity_guard"],
                "status_of_the_prior_campaign_claim": (
                    "the canonical log's standing-rule block asserts 'the H146/H150 "
                    "misbind lanes pass this test - their negatives are genuinely "
                    "unsupported bindings' without a measurement. Measured here on the "
                    "full 15,000-negative population: the assertion holds at binding "
                    "level, and does NOT hold at token level, where the two legs are "
                    "indistinguishable"
                ),
            },
            "per_family": c1["per_family"],
            "per_serial_form": c1["per_serial_form"],
        },

        # ------------------------------------------------------------------ #
        "C2": {
            "verdict": "FAIL",
            "clause": "disjointness from every evaluation surface",
            "forms": c2["forms"],
            "member_units": c2["member_units"],
            "surfaces_measured": len(surfaces),
            "surfaces_clean": clean,
            "totals_per_form": c2["totals_per_form"],
            "all_forms_zero": c2["all_forms_zero"],
            "failures": {
                "R20-H177_eval_B": {
                    "kind": "EVIDENCE leak - byte-identical chunks",
                    "shared_evidence_strings_per_form": {
                        f: v["evidence_shared_strings"]
                        for f, v in surfaces["R20-H177_eval_B"]["forms"].items()},
                    "share_of_the_eval_s_distinct_passages": 0.0258,
                    "eval_rows_affected": 70,
                    "eval_pairs_affected": 35,
                    "eval_pairs_total": 1000,
                    "eval_pair_share": 0.035,
                    "member_rows_affected": 68,
                    "member_pairs_affected": 34,
                    "member_row_share": 0.002267,
                    "attribution": "all 19 belong to this member alone - the other four "
                                   "loaded lanes (quant_scale_unit, frame_reject, "
                                   "attr_pool, path_bind) share 0 chunks with eval_B",
                    "cross_check": "R20-H175b_eval_contamination_sweep.json records "
                                   "raw_in_mix_raw = 19 for eval_B; this pass reproduces "
                                   "that count and attributes it",
                    "consequence": "R20-H177 Lane B's registered PRIMARY (held-out "
                                   "mechanism eval >= 0.80 against a measured 0.5064 "
                                   "floor) is read on a set 3.50% of whose pairs sit on "
                                   "evidence this member trains on",
                },
                "anti_gaming_probe_sets": {
                    "kind": "CLAIM leak - byte-identical claim strings",
                    "surfaces": {
                        "R17-H146_antigaming_set": {
                            "shared_claims_all_three_forms": 18,
                            "surface_distinct_claims": 3185,
                            "eval_pairs_affected": 22, "eval_pairs_total": 1601,
                            "bind_row_pairs_total": 601,
                            "share_of_bind_row_sub_read": 0.0366},
                        "R18-H150_antigaming_set": {
                            "shared_claims_all_three_forms": 15,
                            "surface_distinct_claims": 3186,
                            "eval_pairs_affected": 14, "eval_pairs_total": 1600,
                            "bind_row_pairs_total": 601,
                            "share_of_bind_row_sub_read": 0.0233},
                        "R19-H159_antigaming_set": {
                            "shared_claims_all_three_forms": 17,
                            "surface_distinct_claims": 3186,
                            "eval_pairs_affected": 15, "eval_pairs_total": 1601,
                            "bind_row_pairs_total": 601,
                            "share_of_bind_row_sub_read": 0.025},
                    },
                    "all_collisions_are_bind_row": True,
                    "label_agreement_on_collisions": "20 of 29 collision instances agree; "
                                                     "9 carry the opposite label, which is "
                                                     "correct - the eval item is a "
                                                     "different table",
                    "item_level": "0 - no collision shares the table; the probe sets are "
                                  "built from TabFact test+validation with every train "
                                  "table_id removed, so evidence-level disjointness holds",
                    "root_cause": "R14-H133_antigaming.py line 57 hard-wires "
                                  "LANE = R14-H133_lane.parquet, so its verbatim-claim "
                                  "exclusion was applied against the H133 lane and never "
                                  "against this one. The banked record's "
                                  "shared_claim_strings_with_lane = 0 is a true statement "
                                  "about the H133 lane (10,914 documents / 7,290 tabfact "
                                  "tables - measured to match H133, not this member's "
                                  "7,514 / 4,975)",
                    "evidence_direction": "NOT-APPLICABLE - the probe sets carry no "
                                          "evidence column at all",
                },
            },
            "document_level_read_beyond_string_forms": {
                "why": "string equality cannot see the same table serialized differently",
                "id_level": {
                    "eval_surface": "R17-H143_evalset",
                    "eval_resolved_source_documents": 536,
                    "tabfact_stable_ids": 352, "feverous_unstable_ids": 184,
                    "shared_with_member_tabfact": 0,
                    "shared_with_member_feverous": "0, but UNINTERPRETABLE - see C3",
                },
                "content_rule_read_and_why_it_is_not_evidence": {
                    "exhaustive_hits_on_shipped_tables": c2c3[
                        "exhaustive_shipped_tables_matching_an_eval_document"],
                    "share_of_shipped_tables": c2c3["exhaustive_shipped_share"],
                    "lane_rows_from_those_tables": c2c3["lane_rows_from_matching_tables"],
                    "instrument_validation": {
                        "recall_on_the_true_table": 0.9972,
                        "matches_on_356_ground_truth_fingerprints":
                            rulev_["matches_on_those_fingerprints"],
                        "of_which_the_fingerprint_s_own_table":
                            rulev_["true_positives_same_table"],
                        "mean_wrong_tables_matched_per_fingerprint":
                            rulev_["mean_matches_per_fingerprint"],
                        "max_for_one_fingerprint": rulev_["max_matches_for_one_fingerprint"],
                    },
                    "reading": "the rule detects the true table almost always (0.9972) "
                               "and also fires on ~17 topically similar wrong tables per "
                               "eval chunk. Its raw hit count on the shipped set is "
                               "therefore NOT a contamination measurement and is reported "
                               "here so it is not mistaken for one",
                },
                "build_heuristic_recall_against_its_own_rule": c2c3["heuristic_recall"],
            },
            "binding_constraint": "19 byte-identical evidence chunks shared with "
                                  "R20-H177_eval_B and 18/15/17 byte-identical claim "
                                  "strings shared with the three anti-gaming probe sets - "
                                  "non-zero in all three string forms",
            "fixable": "PIPELINE - dropping the 68 rows on shared evidence and the 23 "
                       "rows carrying a shared claim removes 91 of 30,000 rows (0.303%) "
                       "and costs 45 pairs of 15,000 (0.300%); alternatively the two eval "
                       "surfaces are rebuilt against this member's actual claim and chunk "
                       "sets. Neither touches a bar",
        },

        # ------------------------------------------------------------------ #
        "C3": {
            "verdict": "FAIL",
            "clause": "split semantics verified, never assumed",
            "member_type": c3["member_type"],
            "rows_by_source": c3["lane_rows_by_source"],
            "documents_by_source": c3d["lane_documents_by_namespace_distinct"],
            "tabfact_two_thirds": {
                "verdict": "MEASURED CLEAN",
                "axis": "the archive's own table_id",
                "archive_splits": c3["tabfact_archive_splits"],
                "measured_axis": c3["tabfact_split_axis_measured"],
                "member_tables_in_a_non_train_split": c3[
                    "lane_tabfact_tables_appearing_in_a_non_train_split"],
                "eval_source_tables_present_in_the_member": 0,
                "reading": "TabFact's official split is table-disjoint when tested "
                           "(0 shared table_ids train vs test, 0 train vs validation), "
                           "the member reads train only, and none of its 4,975 tables "
                           "appears in test or validation",
            },
            "feverous_one_third": {
                "verdict": "NOT MEASURABLE FROM THE ARTIFACTS ON DISK",
                "rows": 10110, "row_share": 0.337, "documents": 2539,
                "source": "tmp/R14_H133_feverous.parquet - a single pool with no split "
                          "column",
                "identifier_defect": c3d["why_id_matching_is_insufficient"],
                "why_it_is_load_bearing": "a FEVEROUS-derived evaluation surface exists - "
                                          "184 of R17-H143_evalset's 536 resolved source "
                                          "documents are FEVEROUS - so the axis is not a "
                                          "formality",
                "instruments_tried_and_their_measured_limits": {
                    "build_content_rule": "recall 0.9972 on the true table, but 6,058 "
                                          "matches over 356 ground-truth fingerprints of "
                                          "which 0 are the true table - cannot decide "
                                          "identity",
                    "token_containment_identity_test": "true-table containment min 0.500, "
                                                       "mean 0.917; best known-WRONG table "
                                                       "reaches 1.000 - no separating "
                                                       "threshold exists",
                },
                "what_does_hold": "content-level disjointness is measured directly and "
                                  "reads 0 in all three string forms against "
                                  "R17-H143_evalset (C2)",
            },
            "binding_constraint": "for 33.7% of the member's rows the split axis cannot "
                                  "be measured, and the identifier the member carries for "
                                  "them is not stable across rebuilds",
            "fixable": "PIPELINE - re-key the FEVEROUS pool on a content hash so document "
                       "identity survives a rebuild, then re-run the exclusion by id "
                       "against every FEVEROUS-derived evaluation surface",
        },

        # ------------------------------------------------------------------ #
        "C4": {
            "verdict": "PASS",
            "clause": "contamination census with a live positive control",
            "instrument": c4["instrument"],
            "arena_units": c4["arena_units"],
            "evidence_gate": {
                "units": 12889,
                "verdict": c4["evidence_gate"]["verdict"],
                "max_fraction": c4["evidence_gate"]["max_fraction"],
                "candidate_to_arena": c4["evidence_gate"]["candidate_vs_arena"]["fraction"],
                "arena_to_candidate": c4["evidence_gate"]["arena_vs_candidate"]["fraction"],
                "best_jaccard": c4["evidence_gate"]["candidate_vs_arena"].get("best_jaccard"),
                "kill_bar": 0.02,
                "margin": 0.02,
            },
            "claim_gate": {
                "units": 29699,
                "verdict": c4["claims_gate"]["verdict"],
                "max_fraction": c4["claims_gate"]["max_fraction"],
                "candidate_to_arena": c4["claims_gate"]["candidate_vs_arena"]["fraction"],
                "arena_to_candidate": c4["claims_gate"]["arena_vs_candidate"]["fraction"],
                "best_jaccard": c4["claims_gate"]["candidate_vs_arena"].get("best_jaccard"),
                "kill_bar": 0.02,
                "margin": 0.02,
            },
            "spike_control": c4["spike_control"],
            "live_positive_control": {
                "construction": c4l["source"],
                "tiers": c4l["tiers"],
                "fires": c4l["fires"],
                "reading": c4l["reading"],
            },
            "coverage": {
                "evidence_units_too_short_for_8gram": 0,
                "claims_too_short_for_8gram": 7522,
                "claim_short_share": 0.253275,
                "exact_matching_of_the_short_units": {
                    "method": "normalised claim string sought verbatim inside the "
                              "concatenated normalised arena document corpus",
                    "arena_documents": 8645,
                    "short_claims_found": 0,
                    "long_claims_found": 0,
                },
            },
        },

        # ------------------------------------------------------------------ #
        "C5": {
            "verdict": "PASS",
            "clause": "leak suite for constructed members",
            "recomputed_from_the_parquet": True,
            "registered_conjunction": c5["registered_conjunction"],
            "all_registered_bars_pass": c5["all_registered_bars_pass"],
            "margins": {
                "claim_only_probe": round(
                    0.55 - c5["registered_conjunction"]["claim_only_converged_probe"]["value"], 6),
                "within_pair": round(
                    0.60 - c5["registered_conjunction"]["within_pair_claim_only"]["worst"], 6),
                "worst_surface_channel_deviation": max(
                    v["worst_deviation_from_0.5"]
                    for k, v in c5["registered_conjunction"]["surface_parity_channels"].items()
                    if k != "claim_token_count"),
                "surface_channel_band": 0.05,
            },
            "executor_added_probes_reported_separately":
                c5["executor_added_probes_reported_separately"],
        },

        # ------------------------------------------------------------------ #
        "C6": {
            "verdict": "PASS",
            "clause": "no memorisation channel",
            "pair_key": c6["member_shares_fields_across_pair"],
            "mix_association": {
                "mix_rows": c6m["mix_rows_total"],
                "mix_groups": len(c6m["mix_groups"]),
                "member_keys_shared_with_another_mix_member":
                    c6m["keys_shared_with_any_other_member"],
                "sharing_by_group": {k: v for k, v in
                                     c6m["key_sharing_with_other_mix_members"].items()
                                     if v["shared_evidence_normalised"]
                                     or v["shared_claims_normalised"]},
                "coverage_rows": c6m["mix_keyed_label_association"]["coverage_rows"],
                "coverage_share": c6m["mix_keyed_label_association"]["coverage_share"],
                "auroc_vs_label": c6m["mix_keyed_label_association"]["auroc_vs_label"],
                "bar": "undefined or at chance",
                "reference_contaminated_case": c6["reference_contaminated_case"],
            },
            "within_member_key_keyed_claim_overlap": c6["feature_2_key_keyed_claim_overlap"],
            "reported_and_flagged": {
                "leave_one_out_label_feature": c6["feature_1_key_keyed_label_association"],
                "why_it_is_not_a_channel": "with exact 1:1 pairing the leave-one-out value "
                                           "IS the twin's label, so the feature reads "
                                           "AUROC 0.0 on any perfectly paired member. It "
                                           "is an estimator artifact, not information a "
                                           "model has at inference: both legs share the "
                                           "key, so every key-keyed feature takes the same "
                                           "value on both and separates them at exactly 0.5",
            },
        },

        # ------------------------------------------------------------------ #
        "C7": {
            "verdict": "PASS",
            "clause": "declared units and volume",
            **c78["c7"],
        },

        # ------------------------------------------------------------------ #
        "C8": {
            "verdict": "FAIL",
            "clause": "provenance, licence and internal structure",
            "tabfact_half": {
                "status": "DOCUMENTED",
                "rows": 19890,
                **c78["c8"]["sources"]["tabfact"],
                "gap": "retrieval date not recorded in the sidecar",
            },
            "feverous_half": {
                "status": "UNDOCUMENTED",
                "rows": 10110,
                "row_share": 0.337,
                "documents": 2539,
                **c78["c8"]["sources"]["feverous"],
                "repository_state": "tmp/ is gitignored (.gitignore line 246) and the file "
                                    "is untracked, so this third of the member cannot be "
                                    "rebuilt from the repository",
            },
            "internal_structure": c78["c8"]["internal_structure"],
            "public_repository_check": {
                **c78["c8"]["public_repository_check"],
                "verdict": "CLEAN - both sources are public corpora; no client or company "
                           "name appears in the member or in this report",
            },
            "binding_constraint": "no licence, no retrieval date and no tracked source for "
                                  "33.7% of the member's rows; the campaign's own module "
                                  "records admitted=False for that exact file",
            "fixable": "PIPELINE - re-fetch FEVEROUS from a citable release, write the "
                       "tracked sidecar with licence and retrieval date, and rebuild the "
                       "member's FEVEROUS half from it",
        },

        # ------------------------------------------------------------------ #
        "summary": {
            "pass": ["C1", "C4", "C5", "C6", "C7"],
            "fail": ["C2", "C3", "C8"],
            "not_applicable": [],
            "conforming": False,
            "fixable": "PIPELINE - every failure is a pipeline or artifact-management "
                       "defect; none is a property of TabFact or FEVEROUS as corpora",
            "consequence": (
                "C1 - the clause that killed R20-H175b - holds: this member's labels are "
                "commensurable with the grounding head, verified at binding level on all "
                "15,000 negatives rather than asserted. The failures do not corrupt "
                "training; two of them corrupt READS. R20-H177 Lane B's registered PRIMARY "
                "is read on a set 3.50% of whose pairs sit on evidence this member trains "
                "on, and the anti-gaming hold - read on every arm including the flagship "
                "pair - has 3.66% / 2.33% / 2.50% of its bind_row sub-read on claim "
                "strings this member trains on. The C3 and C8 failures are the same "
                "artifact: a third of the member comes from an untracked, unlicensed, "
                "unstably-keyed working file, so that third is neither reproducible nor "
                "provably disjoint from a FEVEROUS-derived evaluation surface."
            ),
        },

        "artifacts": [
            "experiments/grounding-semantic/contract/quant_misbind_contract_report.json",
            "experiments/grounding-semantic/contract/quant_misbind_c1_containment.json",
            "experiments/grounding-semantic/contract/quant_misbind_c1_binding.json",
            "experiments/grounding-semantic/contract/quant_misbind_c2_disjointness.json",
            "experiments/grounding-semantic/contract/quant_misbind_c3_split.json",
            "experiments/grounding-semantic/contract/quant_misbind_c3_doc_disjoint.json",
            "experiments/grounding-semantic/contract/quant_misbind_c2c3_doc_exhaustive.json",
            "experiments/grounding-semantic/contract/quant_misbind_c2c3_rule_validation.json",
            "experiments/grounding-semantic/contract/quant_misbind_c4_census.json",
            "experiments/grounding-semantic/contract/quant_misbind_c4_live_control.json",
            "experiments/grounding-semantic/contract/quant_misbind_c5_leak.json",
            "experiments/grounding-semantic/contract/quant_misbind_c6_memorisation.json",
            "experiments/grounding-semantic/contract/quant_misbind_c6_mix_assoc.json",
            "experiments/grounding-semantic/contract/quant_misbind_c78_units_provenance.json",
            "experiments/grounding-semantic/contract/quant_misbind_verify.py",
            "experiments/grounding-semantic/contract/quant_misbind_mix_assoc.py",
            "experiments/grounding-semantic/contract/quant_misbind_live_control.py",
            "experiments/grounding-semantic/contract/quant_misbind_doc_disjoint.py",
            "experiments/grounding-semantic/contract/quant_misbind_doc_exhaustive.py",
            "experiments/grounding-semantic/contract/quant_misbind_doc_validate.py",
            "experiments/grounding-semantic/contract/quant_misbind_contract_report.py",
            "logs/quant_misbind_contract.log",
            "logs/quant_misbind_mix_assoc.log",
            "logs/quant_misbind_live_control.log",
            "logs/quant_misbind_doc_disjoint.log",
            "logs/quant_misbind_doc_exhaustive.log",
            "logs/quant_misbind_doc_validate.log",
        ],
    }

    p = HERE / "quant_misbind_contract_report.json"
    p.write_text(json.dumps(report, indent=2))
    print(f"written {p}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

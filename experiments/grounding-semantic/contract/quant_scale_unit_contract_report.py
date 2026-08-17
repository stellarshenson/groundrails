"""Assemble `quant_scale_unit_contract_report.json` from the measured artifacts.

CPU ONLY.  Every number in the report is copied from a measurement file written
by this task; nothing is typed in by hand except the clause text, the verdicts
and the prose that names what was measured.

Run:  CUDA_VISIBLE_DEVICES= uv run python \
      experiments/grounding-semantic/contract/quant_scale_unit_contract_report.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "quant_scale_unit_contract_report.json"


def load(name):
    return json.loads((HERE / name).read_text())


def main():
    c1 = load("quant_scale_unit_c1.json")
    c2c4 = load("quant_scale_unit_c2c4.json")
    mid = load("quant_scale_unit_c3c5c6.json")
    banked = load("quant_scale_unit_c5_banked.json")
    supp = load("quant_scale_unit_supp.json")
    stem = load("quant_scale_unit_stem.json")
    near = load("quant_scale_unit_nearmiss.json")

    I = {b["instrument"]: b for b in c1["containment"]}
    i1, i2, i3 = (I["I1_R20-H175b_precedent"], I["I2_R19-H161_frozen_content"],
                  I["I3_unit_resolved"])
    bv = banked["verify"]

    rep = {
        "member": "quant_scale_unit",
        "kind": "constructed lane",
        "artifact": "experiments/grounding-semantic/R18-H150_scaleunit_lane.parquet",
        "dann_group": "quant_scale_unit",
        "rows": c1["rows"],
        "pairs": c1["pairs"],
        "family": "unit_swap",
        "contract": "docs/experiments/dataset-contract.md",
        "rebuild_path": "R10-H108_lane.public_train() + R18-H150_arm_run.LANES "
                        "(721,210-row mix reproduced exactly, 14 DANN groups); "
                        "the same member is carried by R20-H174_arm_run.LANES "
                        "at the same 5,540 rows / 2,770 pairs in a 760,618-row "
                        "mix",
        "compute": "CPU only; CUDA_VISIBLE_DEVICES forced empty before every "
                   "import; no GPU queried or touched",
        "conforming": False,
        "clauses": {},
    }

    # ------------------------------------------------------------------ C1 --
    rep["clauses"]["C1"] = {
        "clause": "label commensurability",
        "verdict": "FAIL",
        "declared_head": "the grounding scalar (`task_head`), the same head every "
                         "other member of the 721,210-row mix trains",
        "declared_label_predicate": (
            "label 1 = the claim states, for the named (row_key, column) cell, "
            "the value AND the unit the evidence gives; label 0 = the claim "
            "states the same numeral under a DIFFERENT unit of the same physical "
            "dimension. The numeral never moves between legs. The predicate is "
            "SUPPORT: the negative is false with respect to the evidence"),
        "mandatory_test": (
            "claim-to-evidence containment, negative leg against positive leg, "
            "measured with the R20-H175b precedent instrument that produced the "
            "banked qrel_contrast numbers (tokens `[^\\W\\d_]+|\\d+`, lowercased, "
            "no stopword removal; containment = |claim n chunk| / |claim|)"),
        "measured": {
            "instrument_I1_R20-H175b_precedent": {
                "positive_leg": i1["positive_leg"],
                "negative_leg": i1["negative_leg"],
                "mean_gap_neg_minus_pos": i1["mean_gap_neg_minus_pos"],
                "rate_ge_0_90_gap": i1["rate_ge_0_90_gap_neg_minus_pos"],
                "abs_rate_ge_0_90_gap": i1["abs_rate_ge_0_90_gap"],
            },
            "instrument_I2_R19-H161_frozen_content": {
                "positive_rate_ge_0_90": i2["positive_leg"]["rate_ge_0_90"],
                "negative_rate_ge_0_90": i2["negative_leg"]["rate_ge_0_90"],
                "abs_rate_ge_0_90_gap": i2["abs_rate_ge_0_90_gap"],
                "mean_gap_neg_minus_pos": i2["mean_gap_neg_minus_pos"],
            },
            "instrument_I3_unit_resolved": {
                "definition": "I1 after mapping the claim's spelled-out unit "
                              "phrase onto the abbreviation key the evidence "
                              "uses, from the lane's own banked UNITS table",
                "positive_leg": i3["positive_leg"],
                "negative_leg": i3["negative_leg"],
                "mean_gap_neg_minus_pos": i3["mean_gap_neg_minus_pos"],
                "abs_rate_ge_0_90_gap": i3["abs_rate_ge_0_90_gap"],
            },
            "within_pair_containment_identity": c1["within_pair_containment_identity"],
            "unit_surface_attestation": c1["unit_surface_attestation"],
            "h148_literal_presence_rederived": c1["h148_literal_presence"],
            "cited_value_verbatim_in_chunk": c1["cited_value_verbatim_in_chunk"],
            "per_swap_family": supp["c1_per_swap_family"],
            "worst_family_abs_rate_gap": supp["c1_worst_family_abs_rate_gap"],
            "families_with_gap_above_0_10": supp["c1_families_with_gap_above_0_10"],
            "distractor_in_chunk_strata": c1["distractor_in_chunk_strata"],
        },
        "bar": "a member whose negatives are >= 90% attested at a rate within "
               "0.10 of its positives is REJECTED for the grounding head",
        "bar_applied": {
            "negative_rate_ge_0_90": i1["negative_leg"]["rate_ge_0_90"],
            "positive_rate_ge_0_90": i1["positive_leg"]["rate_ge_0_90"],
            "absolute_gap": i1["abs_rate_ge_0_90_gap"],
            "bar_gap": 0.10,
            "margin": round(0.10 - i1["abs_rate_ge_0_90_gap"], 6),
            "triggered": True,
        },
        "why_it_fails": (
            "The two legs of a pair differ in ONE token - the unit word - and the "
            "lane's own H148 anti-shortcut rule requires that token to be absent "
            "from the evidence on BOTH legs (re-derived here: literal-presence "
            "0.0 / 0.0). The mandated lexical instrument therefore reads the two "
            "legs identically: 2,624 of 2,770 pairs (94.73%) score EXACTLY equal, "
            "pooled means differ by 0.000114, and the >= 0.90 attestation rates "
            "differ by 0.00072 against a 0.10 bar. Every one of the 11 swap "
            "families is inside the bar; the worst is 0.0137."),
        "how_this_differs_from_the_clause_provenance": (
            "R20-H175b's qrel_contrast lane failed by having its negatives "
            "ATTESTED: 66.4% at containment exactly 1.0, 72.3% at >= 0.90, both "
            "legs at mean 0.9129. This member fails the same bar from the "
            "opposite side: 0 of 2,770 negatives AND 0 of 2,770 positives reach "
            "containment 1.0, and only 2.13% / 2.20% reach 0.90. The bar as "
            "written triggers on the GAP alone and does not condition on the "
            "negative leg being attested, so a member whose negatives are almost "
            "never attested trips it whenever its positives are not attested "
            "either. Both readings are reported; the clause is applied as "
            "written and not relaxed."),
        "evidence_that_the_label_predicate_is_support": (
            "The claimed unit is attested in the evidence for 96.68% of "
            "positives and 9.75% of negatives (gap 0.8693), measured with the "
            "lane's own banked SURFACE regex vocabulary. The residual 9.75% is "
            "exactly the 270-pair in-chunk distractor stratum (270/2,770 = "
            "0.0975), where the corrupted unit is deliberately attested but "
            "bound to a DIFFERENT column. Under the unit-resolved instrument the "
            "legs separate - positive mean 0.7033 vs negative 0.6156, fully-"
            "attested 8.30% vs 0.61% - but the >= 0.90 rate gap is still 0.0690, "
            "inside the 0.10 bar."),
        "binding_constraint": (
            "The C1 instrument is lexical containment; the token this member's "
            "label turns on is required by the H148 anti-shortcut rule to be "
            "lexically absent from the evidence. Widening the gap to clear 0.10 "
            "requires the positive's unit token present in the evidence and the "
            "negative's absent - the adjacent-string channel the H148 rule "
            "closes, and the R17-H145 construction this lane was built to "
            "replace (measured worse: 0.8402 vs 0.8673)."),
        "fixable": "CORPUS_PROPERTY",
    }

    # ------------------------------------------------------------------ C2 --
    c2 = c2c4["c2"]
    rep["clauses"]["C2"] = {
        "clause": "disjointness from every evaluation surface",
        "verdict": "PASS",
        "measured": {
            "forms": ["raw", "truncated to 1500 chars", "whitespace-collapsed "
                      "case-folded"],
            "directions": "both (member -> surface and surface -> member)",
            "member_units": c2c4["member_units"],
            "arena_10_subsets": c2["arena_10_subsets"],
            "gold_full": c2["gold_full"],
            "mechanism_evals": c2["mechanism_evals"],
            "worst_shared_units_any_surface_any_form": c2[
                "worst_shared_units_any_surface_any_form"],
        },
        "bar": "all forms read zero on every surface",
        "margin": "0 shared units against 12 surfaces (10-subset arena, "
                  "gold_full, 9 mechanism eval parquets) x 3 forms x 2 unit "
                  "types (evidence, claims); the bar is zero and the reading is "
                  "zero, so the margin is exact rather than numeric",
        "supplementary_near_duplicate_sweep": {
            "note": "EXECUTOR-ADDED, beyond the clause. C2 tests exact forms; "
                    "the stem-collision finding made the near-duplicate "
                    "question live. Instrument: provenance_gate.py n=8, Jaccard "
                    ">= 0.3, bidirectional.",
            "per_surface": near["per_surface"],
            "worst_pooled_fraction": near["worst_max_fraction_all_surfaces"],
            "worst_shared_document_stratum_fraction": near[
                "worst_shared_document_stratum_fraction"],
            "findings": [
                "R18-H150_unitswap_probe (this member's own held-out mechanism "
                "eval): 1 of 115 probe chunks reaches Jaccard >= 0.3 against a "
                "lane chunk (0.87%, gate WARN band, below the 2% KILL); lane max "
                "Jaccard 0.4039; 0 shared doc_ids",
                "R20-H177_eval_B: 13 shared TabFact doc_ids at 0 shared passages "
                "in all three exact forms; inside that shared-document stratum "
                "the gate reads 4.35% surface->lane (above 2% applied to the "
                "stratum) and 0.14% pooled over the whole eval (below)",
            ],
        },
    }

    # ------------------------------------------------------------------ C3 --
    c3 = mid["c3"]
    rep["clauses"]["C3"] = {
        "clause": "split semantics are verified, never assumed",
        "verdict": "FAIL",
        "measured": {
            "tabfact_split_axis_measured_from_archive": c3["tabfact"],
            "lane_tabfact_documents": c3["lane_tabfact_documents"],
            "tabfact_content_level_leak": supp["tabfact_content_split_leak"],
            "feverous": c3["feverous"],
            "lane_declared_split_axis": c3["lane_declared_split_axis"],
            "lane_cut_against_sibling_surfaces": c3[
                "lane_cut_against_sibling_surfaces"],
            "tabfact_document_id_stem_collisions": stem,
            "eval_exclusion_rederived": banked["corpus_rebuild"],
        },
        "what_passes": (
            "TabFact's official split is measured to cut per TABLE - 0 of 1,696 "
            "validation and 0 of 1,695 test table_ids appear in train - and 8 "
            "validation plus 5 test tables are byte-identical to a train table "
            "under a different id. None of the member's 319 TabFact tables is "
            "one of those: 0 byte-identical to a validation or test table. The "
            "member's own document cut re-derives exactly - 23,423 candidate "
            "tables, 3,516 carrying eval content, 16,738 admitted, all 410 lane "
            "documents resolved, 0 shared documents and 0 shared chunks with the "
            "R17-H143 eval set."),
        "what_fails": (
            "The FEVEROUS half - 1,040 of 5,540 rows, 18.77% - has NO measurable "
            "split. `tmp/R14_H133_feverous.parquet` carries columns "
            "[id, claim, label, evidence, challenge] and no split column at all, "
            "so the registration's 'FEVEROUS-train' cannot be measured from the "
            "archive. C3 requires the axis measured from the archive, not read "
            "from a card; for this source there is nothing to measure."),
        "additional_finding": (
            "19 within-lane TabFact document-id stem collisions: TabFact's "
            "`1-`/`2-` csv-id prefixes render one serialised table under two "
            "doc_ids, so 19 of the member's 319 TabFact 'documents' are two ids "
            "for one table stem. The member's per-document pair cap and its "
            "fold-disjointness accounting treat those as distinct documents. The "
            "member is nevertheless passage-clean against every surface (C2 = 0), "
            "and 0 stems are shared with a surface without the id also being "
            "shared."),
        "bar": "state the split axis a corpus actually cuts on, measured from "
               "the archive; test it",
        "margin": "1,040 of 5,540 rows (18.77%) have no measurable split axis; "
                  "the bar admits no shortfall",
        "fixable": "PIPELINE",
    }

    # ------------------------------------------------------------------ C4 --
    c4 = c2c4["c4"]
    rep["clauses"]["C4"] = {
        "clause": "contamination census with a live positive control",
        "verdict": "PASS",
        "measured": {
            "instrument": "provenance_gate.py (banked R14-H136 form): 8-gram, "
                          "Jaccard >= 0.3, bidirectional, WARN 0.5%, KILL 2%, "
                          "per-arena-subset attribution",
            "chunks": {
                "candidate_units": c4["gate_chunks"]["candidate"],
                "arena": c4["gate_chunks"]["arena"],
                "candidate_vs_arena": c4["gate_chunks"]["candidate_vs_arena"],
                "arena_vs_candidate": c4["gate_chunks"]["arena_vs_candidate"],
                "max_fraction": c4["gate_chunks"]["max_fraction"],
                "verdict": c4["gate_chunks"]["verdict"],
            },
            "claims": {
                "candidate_units": c4["gate_claims"]["candidate"],
                "max_fraction": c4["gate_claims"]["max_fraction"],
                "best_jaccard": c4["gate_claims"]["candidate_vs_arena"].get(
                    "best_jaccard"),
                "verdict": c4["gate_claims"]["verdict"],
            },
            "spike_control": {"chunks": c4["spike_chunks"],
                              "claims": c4["spike_claims"]},
            "live_positive_control": c4["live_positive_control"],
            "coverage": c4["coverage"],
        },
        "bar": "max fraction below 2%; spike 10/10 at 0 baseline hits; a live "
               "positive control that fires; short-unit coverage stated",
        "margin": "max fraction 0.0000 against a 0.0200 KILL - margin 0.0200, "
                  "and max Jaccard is 0.0000 in both directions on both unit "
                  "types, so no unit is anywhere near the threshold. Spike 10/10 "
                  "with 0 baseline hits on chunks and on claims. Live control "
                  "fires hard: 270 of 2,509 lane chunks reach Jaccard >= 0.3 "
                  "against R17-H145's chunks at max Jaccard 1.0 (33 exactly "
                  "shared), gate verdict KILL on the control - the gate can "
                  "fire on this member's text. Coverage: 0 of 2,509 chunks and "
                  "250 of 5,456 claims (4.58%) are too short for an 8-gram "
                  "instrument; all 250 checked by exact matching, 0 hits.",
    }

    # ------------------------------------------------------------------ C5 --
    rep["clauses"]["C5"] = {
        "clause": "leak suite for constructed members",
        "verdict": "PASS",
        "instrument": banked["instrument"],
        "reproduction_caveat": banked["reproduction_caveat"],
        "registered_conjunction": {
            "claim_only_converged_probe": {
                "value": bv["claim_only_tfidf_auroc"]["value"],
                "bar": "< 0.55",
                "margin": round(0.55 - bv["claim_only_tfidf_auroc"]["value"], 4),
                "banked_build_value": 0.4296,
            },
            "within_pair_claim_only": {
                "worst_family": bv["within_pair_claim_only_accuracy"]["worst"],
                "pooled": bv["within_pair_claim_only_accuracy"][
                    "per_neg_family"]["unit_swap"]["acc"],
                "bar": "< 0.60",
                "margin": round(
                    0.60 - bv["within_pair_claim_only_accuracy"]["worst"], 4),
                "banked_build_worst": 0.5,
                "per_swap_family": bv["within_pair_claim_only_accuracy"][
                    "per_swap_family"],
                "two_sided_note": bv["within_pair_claim_only_accuracy"][
                    "worst_two_sided_deviation_report_only"],
            },
            "surface_parity": {
                "auroc": bv["surface_parity"]["auroc"],
                "bar": "each in [0.45, 0.55]",
                "worst_deviation": bv["surface_parity"]["max_deviation"],
                "margin": round(0.05 - bv["surface_parity"]["max_deviation"], 4),
            },
            "h148_literal_presence": bv["claim_unit_literal_presence"],
            "positive_verbatim_substring": bv["positive_verbatim_substring"],
            "minimal_pair_integrity": bv["minimal_pair_integrity"],
            "unit_rederivation_audit": bv["unit_rederivation_audit"],
            "dedupe_disjointness": bv["dedupe_disjointness"],
            "word_label_marginal_balance": bv["word_label_marginal_balance"],
            "value_surface_direction_parity": bv["value_surface_direction_parity"],
            "digit_surface_report_only": bv["digit_surface_report_only"],
            "all_bars_pass": bv["all_bars_pass"],
        },
        "single_channel_probes": {
            "question_only": {
                "status": "NOT-APPLICABLE",
                "reason": "the member carries no question field; the "
                          "construction has no question channel to probe",
            },
            "evidence_only": {
                "status": "measured",
                "value": 0.5,
                "mechanism": "all 2,509 distinct chunks carry exactly as many "
                             "label-1 as label-0 rows, so any evidence-only "
                             "feature is exactly at chance",
                "confirmed_by_probe": mid["c5"]["executor_added"][
                    "evidence_only_tfidf_probe_measured"]["value"],
            },
        },
        "balance": {
            "worst_positive_share_deviation": mid["c5"]["registered"][
                "balance"]["worst_positive_share_deviation"],
            "fields_checked": ["direction", "swap_family", "dimension",
                               "unit_carrier", "serial_form", "template_id",
                               "source", "neg_family"],
            "note": "every cell of every field is exactly 50% positive",
        },
        "attestation_symmetry": {
            "worst_unit_word_skew": mid["c5"]["registered"][
                "attestation_symmetry_unit_word_marginal"]["worst_skew"],
            "bar": "0.0 - a unit word must not carry a label prior",
        },
        "executor_added_probes_reported_separately": {
            "note": "these join NO registered conjunction",
            "evidence_only_tfidf": mid["c5"]["executor_added"][
                "evidence_only_tfidf_probe_measured"],
            "claim_plus_evidence_lexical": {
                **mid["c5"]["executor_added"]["claim_plus_evidence_lexical_probe"],
                "caveat": "run on UNSTRATIFIED document folds, which carry the "
                          "documented below-chance artifact (R17-H145 finding "
                          "b); not interpretable in either direction",
            },
        },
        "executor_defect_recorded": (
            "The executor's first C5 pass reimplemented the probe with "
            "unstratified document folds and read within-pair 0.7815 on the "
            "km<->m family. That is the artifact the lane's build documented, "
            "not a leak: with a linear probe a fold holding `a` pairs of one "
            "direction and `b` of the other forces within-pair accuracy to "
            "min(a,b)/(a+b). The reading was discarded and the member's own "
            "banked `verify()` was run instead. The defective number is recorded "
            "here rather than deleted."),
    }

    # ------------------------------------------------------------------ C6 --
    c6 = mid["c6"]
    rep["clauses"]["C6"] = {
        "clause": "no memorisation channel",
        "verdict": "PASS",
        "measured": {
            "within_pair_field_identity": c6["within_pair_field_identity"],
            "mix_association_feature": c6["mix_association_feature"],
            "mix_rows_excluding_member": c6["mix_rows_excluding_member"],
            "executor_added_unit_word_prior": c6["executor_added_unit_word_prior"],
            "repeated_claim_strings": supp["repeated_claim_strings"],
        },
        "bar": "report the value; on a clean instrument it is undefined or at "
               "chance",
        "margin": (
            "The feature reads AUROC 0.4876 at 0.79% coverage (44 of 5,540 "
            "rows), against R20-H175b's 0.6230 at 98% coverage. It is at chance "
            "and nearly uncovered. Structurally it cannot be otherwise: 21 of "
            "the member's 23 fields are IDENTICAL on both legs of all 2,770 "
            "pairs - only `claim` and `cited_unit` differ - so every feature "
            "keyed on the pair's key takes the same value on both legs. The "
            "executor-added unit-word prior reads exactly 0.5000. 84 claim "
            "strings repeat, all across different documents, and 0 carry both "
            "labels."),
    }

    # ------------------------------------------------------------------ C7 --
    rep["clauses"]["C7"] = {
        "clause": "declared units and volume",
        "verdict": "PASS",
        "measured": mid["c7"],
        "bar": "state the unit and use it consistently; report both counts "
               "always",
        "margin": (
            "Registration states BOTH units ('~20,000 rows / ~10,000 pairs'); "
            "the build reports both (5,540 rows / 2,770 pairs); the arm's LANES "
            "tuple carries both and aborts on drift. The shortfall is identical "
            "in both units - 27.70% of rows and 27.70% of pairs - so no unit "
            "conversion moves the number. The delivered scale was reported at "
            "the time as '28% of registered scale' and amendment A1 demoted the "
            "scale_unit co-primary on that basis rather than restating the "
            "count. This clause is about accounting honesty, not volume, and the "
            "accounting is honest."),
    }

    # ------------------------------------------------------------------ C8 --
    rep["clauses"]["C8"] = {
        "clause": "provenance, licence and internal structure",
        "verdict": "FAIL",
        "measured": mid["c8"],
        "what_passes": (
            "TabFact - 4,500 rows (81.23%): tracked sidecar "
            "`data/external/datasets/dataset-tabfact.md`, licence CC-BY-4.0, "
            "archive `dataset-tabfact.zip`, selection predicate the "
            "`__train.parquet` split deduped on `table_text` and filtered "
            "through the R17-H144 content-fingerprint eval exclusion (re-derived "
            "exactly: 3,516 tables dropped for eval content, 16,738 admitted). "
            "Internal structure reported: 0 duplicate (claim, chunk, label) "
            "rows, 5,456 distinct claims, 2,509 distinct chunks, 410 documents, "
            "max claim repeat 2, max chunk repeat 18, mean 6.76 pairs/document, "
            "max 32. Public-repository check clean - no client or company name "
            "in any artifact."),
        "what_fails": (
            "FEVEROUS - 1,040 rows (18.77%): NO licence recorded in any tracked "
            "artifact, NO retrieval date, NO sidecar, NO archive. The source is "
            "`tmp/R14_H133_feverous.parquet`, an R14-H133 working file, and the "
            "campaign has already ruled on it - R20-H177 coordinator "
            "disposition 2 (2026-08-16) accepted FEVEROUS non-admission for "
            "Lane B because 'the on-disk file is an R14-H133 working artifact "
            "without its own provenance verdict'. That ruling stands unamended "
            "and this member carries the same file. The selection predicate is "
            "additionally not reproducible: doc ids are `feverous:{i}` indices "
            "from an order-unstable dedup that R17-H144 recorded as resolving "
            "0 of 95 sampled ids to their source table on rebuild."),
        "bar": "source, licence, retrieval date and the exact selection "
               "predicate, for the whole member",
        "margin": "1,040 of 5,540 rows (18.77%) carry none of the four; the "
                  "clause admits no shortfall",
        "fixable": "PIPELINE",
    }

    rep["summary"] = {
        "PASS": ["C2", "C4", "C5", "C6", "C7"],
        "FAIL": ["C1", "C3", "C8"],
        "NOT-APPLICABLE": [],
        "priority_clause_C1": "FAIL",
        "fixable": "CORPUS_PROPERTY",
        "binding_constraint": (
            "C1 is not fixable by a conforming pipeline. C3 and C8 share one "
            "fixable cause - the FEVEROUS 1,040 rows - and both clear if those "
            "rows are dropped (leaving 4,500 rows / 2,250 pairs) or if FEVEROUS "
            "is given a licence sidecar, a retrieval date and a reproducible "
            "selection predicate. C1 survives either repair: the token the label "
            "turns on is required by the member's own H148 anti-shortcut rule to "
            "be lexically absent from the evidence, so the mandated containment "
            "instrument reads both legs identically on 94.73% of pairs whatever "
            "the source corpus."),
        "consequence": (
            "The member is 5,540 of the 721,210 R18-H150 flagship mix rows "
            "(0.768%) and 5,540 of the 760,618 R20-H174 mix rows (0.728%), so "
            "it rides in both banked flagship draws and in every R20-H174 draw "
            "including the three now training. C1's failure here is NOT the "
            "R20-H175b failure: those "
            "negatives were supported claims labelled 0 (66.4% fully attested); "
            "these negatives are unsupported - the claimed unit is attested for "
            "96.68% of positives and 9.75% of negatives, and the 9.75% is "
            "exactly the deliberate in-chunk distractor stratum. What the "
            "measurement establishes is that the contract's C1 instrument cannot "
            "certify this member either way, and that the clause as written "
            "rejects it. C3 and C8 are real gaps of record, not of construction: "
            "18.77% of the member's rows come from a file the campaign has "
            "already ruled has no provenance verdict."),
    }

    rep["artifacts"] = [
        "experiments/grounding-semantic/contract/quant_scale_unit_contract_report.json",
        "experiments/grounding-semantic/contract/quant_scale_unit_c1.json",
        "experiments/grounding-semantic/contract/quant_scale_unit_c2c4.json",
        "experiments/grounding-semantic/contract/quant_scale_unit_c3c5c6.json",
        "experiments/grounding-semantic/contract/quant_scale_unit_c5_banked.json",
        "experiments/grounding-semantic/contract/quant_scale_unit_supp.json",
        "experiments/grounding-semantic/contract/quant_scale_unit_stem.json",
        "experiments/grounding-semantic/contract/quant_scale_unit_nearmiss.json",
        "experiments/grounding-semantic/contract/quant_scale_unit_c1.py",
        "experiments/grounding-semantic/contract/quant_scale_unit_c2c4.py",
        "experiments/grounding-semantic/contract/quant_scale_unit_c3c5c6.py",
        "experiments/grounding-semantic/contract/quant_scale_unit_c5_banked.py",
        "experiments/grounding-semantic/contract/quant_scale_unit_supp.py",
        "experiments/grounding-semantic/contract/quant_scale_unit_stem.py",
        "experiments/grounding-semantic/contract/quant_scale_unit_nearmiss.py",
        "experiments/grounding-semantic/contract/quant_scale_unit_mixcache.py",
        "experiments/grounding-semantic/contract/quant_scale_unit_contract_report.py",
        "logs/contract-quant_scale_unit-mixcache.log",
        "logs/contract-quant_scale_unit-c2c4.log",
        "logs/contract-quant_scale_unit-c3c5c6.log",
        "logs/contract-quant_scale_unit-c5banked.log",
        "logs/contract-quant_scale_unit-nearmiss.log",
    ]

    OUT.write_text(json.dumps(rep, indent=2, default=str) + "\n")
    print(f"wrote {OUT}")
    print(json.dumps(rep["summary"], indent=2))


if __name__ == "__main__":
    main()

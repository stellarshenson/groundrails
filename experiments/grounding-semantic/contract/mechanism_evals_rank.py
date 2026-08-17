"""Rank the mechanism-eval contract results by CONSEQUENCE and re-emit the table.

CPU only, no measurement of its own - it reads
`contract/mechanism_evals_report.json` and
`contract/mechanism_evals_antigaming_supp.json`, folds in what the canonical log
says reads each instrument, and writes the ranking back into the report plus a
markdown summary.

Measurement and building only.  The tiers below say what a finding TOUCHES, not
what it means; no verdict is adjudicated here.

Out: contract/mechanism_evals_report.json (ranking section added)
     contract/mechanism_evals_summary.md
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import pathlib

HERE = pathlib.Path(__file__).parent
REPORT = HERE / "mechanism_evals_report.json"
SUPP = HERE / "mechanism_evals_antigaming_supp.json"
SPOT = HERE / "mechanism_evals_spotchecks.json"
TABLE = HERE / "mechanism_evals_summary.md"
NOTE = "Numbers recorded, not adjudicated - the coordinator adjudicates."

# tier, the verdict or gate that reads it, and whether this pass found anything
# that touches that verdict.  Every "what_it_touches" string names a measured
# number from the report and stops there.
RANK = [
    ("antigaming_nearmiss_bindrow", 1,
     "the anti-gaming hold. BINDING on R18-H150 / H152 / H155 / H156 / R19-H159 "
     "/ H160 (`near-miss >= 0.7438`, the flagship's own d2 passing by +0.0049) "
     "and a recorded DIAGNOSTIC since author ruling 3 of 2026-08-16, which "
     "suspended the band. The `bind_row >= 0.95` clause rides the same file",
     "the near-miss half is byte-identical across all 14 banked arm sets, so the "
     "HEADLINE hold is comparable; the bind_row half is a different 600-pair set "
     "on every arm (Jaccard 0.0017-0.0127 against the flagship's). 5.03% of the "
     "eval's TabFact documents (59 of 1,173 stems, 166 of 3,200 rows) are inside "
     "the mix under the stem key while the raw key reads 0; 15 of 3,186 eval "
     "claims are verbatim mix claims, all in the `quant_misbind` lane; the "
     "claim-only probe reads 0.6217 against C5's < 0.55 and within-pair 0.6206 "
     "against < 0.60"),
    ("h166a1_vitaminc_holdout", 1,
     "the PRIMARY mechanism gate of R19-H166-A1 (`held-out VitaminC "
     "REFUTES-vs-NEI AUROC >= 0.85`), an author-assented arm still in the "
     "training queue; its banked baseline leg is 0.3935",
     "6 (claim, evidence) pairs carry BOTH labels - C1's structural test fires. "
     "2 evidence units and 2 claims (3 of 38,126 rows, 0.008%) sit inside the "
     "mix and are visible ONLY under whitespace normalisation, which the banked "
     "builder's raw-string filter could not see. The page document channel is "
     "clean at 0 of 5,553"),
    ("findver", 1,
     "the BANKED standing non-arena mechanism instrument for derivation-adjacent "
     "arms - R20-H176's CONFIRMED verdict (numeric 2-draw mean 0.4959 against ie "
     "0.6609 and knowledge 0.5838) is read on it",
     "nothing. Zero on every string form in both directions against the mix, the "
     "arena and gold_full, and zero on the document channel"),
    ("eval_C", 1,
     "the R20-H177 Lane C PRIMARY gate (>= 0.80); the baseline leg read 0.9085 "
     "and REFUTED the arm's near-chance prediction, and the lane's disposition "
     "is open",
     "nothing. Zero on every string form and every surface, zero documents, C1 "
     "structural 0 with the relational instrument separating 0.967 against "
     "0.033, and every C5 bar clear"),
    ("h150_unitswap_probe", 2,
     "the R18-H150 scale/unit probe, pinned in its manifest as a reported "
     "secondary that never trains and is never selected on; probe gating was "
     "stopped campaign-wide by R19-H163",
     "27 of its 42 documents (196 of 280 rows, 70%) are inside the mix on the "
     "RAW document key - 23 through the `tabfact` member and 17 through the "
     "`quant_misbind` lane - against its registration as a `document-disjoint "
     "probe from unused supply`, which held only against the scale/unit lane it "
     "was built beside. Its text channel is clean. The claim-only probe reads "
     "0.1425, a 0.3575 inverted deviation from chance"),
    ("r15_bindprobe", 2,
     "was the PRIMARY of R17-H146 (`bind_col >= 0.80 AND bind_row >= 0.95`); "
     "probe-bank gating DECLARED DEAD by R19-H163",
     "41 of 804 document stems and 28 of 3,852 claims inside the mix, 3 evidence "
     "tables byte-identical to a mix passage; the claim-only probe reads 0.5508 "
     "against C5's < 0.55"),
    ("r15_typeprobe", 2,
     "the derivation half of the probe bank; gating DECLARED DEAD by R19-H163",
     "C1's structural test fires on 6 pairs; 54 of 1,136 document stems and 21 "
     "of 16,507 claims inside the mix"),
    ("h148_itemindex_probe", 2,
     "the H148 item-index probe, still cited as the `literal-presence` build "
     "check for the H150 unit-swap probe; probe gating dead",
     "nothing on any channel. C1 separates 0.9902 against 0.0 on the 10.4% of "
     "rows the item-index instrument can read"),
    ("h149_roleswap_probe", 2,
     "the H149 role-swap probe; probe gating dead",
     "nothing on any channel; every C5 bar clear. C1's test 2 is NOT COMPUTABLE "
     "on CPU for a lexical role swap and is reported as such"),
    ("antigaming_traced", 3,
     "the traced anti-gaming diagnostic, read at R14-H133 and after; never a "
     "promotion clause",
     "47 of 979 document stems inside the mix; the claim-only probe reads 0.6999 "
     "and within-pair 0.8756, and surface parity fails at 0.2784 on the "
     "EXECUTOR-ADDED `claim_numeral_count` channel - the trace prefix prints the "
     "asserted figure, so the negative leg carries a different numeral count"),
    ("g0b_composed_probes", 3,
     "gate 5 of the hotpotqa composition fanout - baseline 0.6477 against a KILL "
     "at >= 0.70, PASS, and already moot for registration because gate G0a failed",
     "28 of 564 document stems and 5 of 500 evidence units inside the mix; the "
     "claim-only probe reads 0.6867 and within-pair 0.908"),
    ("h117_heldout_pairs", 3,
     "kill-gate 2 of the R11-H117 paired-margin arm, closed at PROCEED with "
     "lambda_margin 0.3; the DR lane it served never entered the flagship mix",
     "306 of 1,933 evidence passages (620 of 4,000 rows, 15.5%) and 286 of 4,000 "
     "claims are inside the mix, every one of them supplied by the `tabfact` "
     "member; the claim-only probe reads 0.6711 and within-pair 0.8195"),
    ("h175b_eval_clean", 3,
     "no live gate - R20-H175b is WITHDRAWN; the eval is retained for a possible "
     "future option-D registration with a parallel relevance head",
     "C1's structural test fires on 44 of 44 pairs (100%) - the same signature "
     "the withdrawn qlane carries, and by the same construction, since both legs "
     "share claim and evidence and only the question differs. C2 is clean on "
     "every channel including the PsiloQA page key"),
    ("h175b_eval_clean_prefix", 3,
     "no live gate - same standing as the eval above",
     "C1's structural test fires on 16 of 16 pairs (100%); C2 clean everywhere"),
    ("dr_h113_gate_judged", 3,
     "the DR lane's generator quality gate, closed in round 11; the DR lane never "
     "entered the flagship mix and no open arm reads the file",
     "159 of its 1,446 evidence chunks are inside the mix and 0 of its 1,505 "
     "claims are. Verified because it carries claim and evidence text, not "
     "because it is an evaluation surface - its label is a judge verdict on a "
     "generated rewrite, and its `chunk` column carries source-code fragments"),
    ("r12_h121_gateBC_rows", 3,
     "Gate B of R12-H121, which KILLED the hypothesis pre-build at purity 0.284 "
     "against a 0.95 bar",
     "2,476 of 2,476 claims and 1,059 of 3,346 windows are inside the mix - "
     "expected by construction rather than a contamination finding, since the "
     "killed hypothesis was to build distractor-window negatives out of the mix's "
     "own documents. No label column, so C1 / C5 / C6 are NOT-APPLICABLE; C2 and "
     "C7 are run so the enumeration leaves nothing unmeasured"),
]

TIER_NAME = {
    1: "carries a LIVE or STANDING gate",
    2: "carried a gate that has since been retired; banked verdicts stand on the record",
    3: "no live gate",
}


def main():
    rep = json.loads(REPORT.read_text())
    supp = json.loads(SUPP.read_text()) if SUPP.exists() else None
    if supp:
        rep["cross_arm_antigaming_identity"]["supplement"] = SUPP.name
        rep["cross_arm_antigaming_identity"]["measured_overlap"] = {
            "reference_set": supp["reference_set"],
            "nearmiss_half": "Jaccard 1.0 against the flagship's set on every one "
                             "of the 14 banked arm files - the HEADLINE near-miss "
                             "read is one instrument",
            "bind_row_half": "Jaccard 0.0017-0.0127 - a different 600-pair set on "
                             "every arm",
            "cause": supp["supply_determinism"]["mechanism"],
            "membership_identical_across_runs":
                supp["supply_determinism"]["membership_identical_across_runs"],
            "order_identical_across_runs":
                supp["supply_determinism"]["ORDER_identical_across_runs"]}

    ranked = []
    for name, tier, reads, touches in RANK:
        if name not in rep["evals"]:
            continue
        c = rep["evals"][name]["clauses"]
        s = c["C2"]["surfaces"]["flagship_mix_superset"]
        d = c["C2"]["document_channel"]
        c5 = c["C5"].get("claim_only_converged_probe")
        ranked.append({
            "eval": name, "tier": tier, "tier_meaning": TIER_NAME[tier],
            "what_reads_it": reads,
            "what_this_pass_found_that_touches_it": touches,
            "measured": {
                "rows": c["C7"]["rows"], "pairs": c["C7"]["pairs"],
                "C2_evidence_units_in_mix": s["evidence_units_hit_any_form"],
                "C2_evidence_units_total":
                    s["evidence"]["eval_units_into_surface"]["n_query_units"],
                "C2_claim_units_in_mix": s["claim_units_hit_any_form"],
                "C2_claim_units_total":
                    s["claim"]["eval_units_into_surface"]["n_query_units"],
                "C2_rows_on_a_hit_evidence_unit":
                    s.get("rows_on_a_hit_evidence_unit"),
                "C2_documents_in_mix_STEM": d["documents_in_the_mix_STEM"],
                "C2_documents_total": d["eval_document_stems"],
                "C2_rows_on_a_mix_document": d["rows_on_a_mix_document_STEM"],
                "C2_arena_clean": c["C2"]["surfaces"]["blind_arena"]["clean"],
                "C2_gold_full_clean": c["C2"]["surfaces"]["gold_full"]["clean"],
                "C1_structural_identical_pairs":
                    c["C1"].get("test_1_structural", {}).get("identical_pairs"),
                "C1_test2_positive_leg":
                    c["C1"].get("test_2_strict_separation", {}).get("positive_leg_rate"),
                "C1_test2_negative_leg":
                    c["C1"].get("test_2_strict_separation", {}).get("negative_leg_rate"),
                "C5_claim_only": c5["auroc"] if isinstance(c5, dict) else None,
                "C5_within_pair": c["C5"].get("within_pair_claim_only", {}).get("acc")
                                  if isinstance(c["C5"].get("within_pair_claim_only"), dict)
                                  else None,
                "C6_auroc": c["C6"].get("auroc"),
                "C6_coverage": c["C6"].get("coverage"),
            },
        })
    ranked.sort(key=lambda r: (r["tier"], -(r["measured"]["C2_documents_in_mix_STEM"] or 0)))
    rep["ranked_by_consequence"] = {
        "method": "each eval was grepped against the canonical log "
                  "(docs/experiments/semantic-grounding-experiments.md) for the "
                  "gate or verdict that reads it; the tier states what the "
                  "instrument carries, and the finding column names only measured "
                  "numbers. Whether a verdict moves is the coordinator's call",
        "tiers": TIER_NAME,
        "order": ranked,
    }
    if SPOT.exists():
        rep["independent_spot_checks"] = json.loads(SPOT.read_text())
    rep["note"] = NOTE
    REPORT.write_text(json.dumps(rep, indent=2))
    TABLE.write_text(table(rep, ranked))
    print(f"wrote {REPORT}\nwrote {TABLE}", flush=True)


def table(rep, ranked):
    L = ["# Held-out mechanism evals - contract verification, ranked by consequence",
         "",
         NOTE,
         "",
         "Sixteen artifacts, verified against `docs/experiments/dataset-contract.md` "
         "with amendments C-A1 and C-A2 applied. `R20-H177_eval_B`, "
         "`R17-H143_evalset`, the blind arena and `gold_full` are excluded - other "
         "agents own them this session.",
         "",
         "## Per-eval measurements",
         "",
         "| # | eval | tier | rows / pairs | C2 evidence in mix | C2 claims in mix "
         "| C2 docs in mix (stem) | C1 structural | C5 claim-only | C5 within-pair | C6 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ranked, 1):
        m = r["measured"]
        L.append(
            f'| {i} | `{r["eval"]}` | {r["tier"]} | {m["rows"]} / {m["pairs"]} '
            f'| {m["C2_evidence_units_in_mix"]} of {m["C2_evidence_units_total"]} '
            f'| {m["C2_claim_units_in_mix"]} of {m["C2_claim_units_total"]} '
            f'| {m["C2_documents_in_mix_STEM"]} of {m["C2_documents_total"]} '
            f'| {fmt(m["C1_structural_identical_pairs"])} '
            f'| {fmt(m["C5_claim_only"])} | {fmt(m["C5_within_pair"])} '
            f'| {fmt(m["C6_auroc"])} |')
    L += ["", "Every eval reads 0 against the blind arena and 0 against `gold_full` "
              "on all six string forms in both directions.", "",
          "## What reads each instrument, and what this pass found", ""]
    for i, r in enumerate(ranked, 1):
        L += [f'### {i}. `{r["eval"]}` - tier {r["tier"]}, {r["tier_meaning"]}', "",
              f'**Reads it**: {r["what_reads_it"]}', "",
              f'**Found**: {r["what_this_pass_found_that_touches_it"]}', ""]
    pc = rep["LIVE_POSITIVE_CONTROLS"]
    L += ["## Live positive controls on the disjointness instrument", "",
          "| control | design | result | fires |", "|---|---|---|---|",
          f'| 1 synthetic identity | {pc["1_synthetic_identity"]["units"]} mix '
          f'passages offered to the gate as an eval | '
          f'{pc["1_synthetic_identity"]["units_hit_any_form"]} of '
          f'{pc["1_synthetic_identity"]["units"]} read | yes |',
          f'| 2 synthetic re-wrap | the same passages with every space replaced by '
          f'a newline plus indent | raw '
          f'{pc["2_synthetic_rewrap"]["counts"]["raw_in_raw"]} of '
          f'{pc["2_synthetic_rewrap"]["units"]}, normalised '
          f'{pc["2_synthetic_rewrap"]["counts"]["normalised_in_normalised_raw"]} of '
          f'{pc["2_synthetic_rewrap"]["units"]} | yes - only the normalised form '
          f'sees it |',
          f'| 2b synthetic document stem | '
          f'{pc["2b_synthetic_document_stem"]["units"]} member table ids with the '
          f'`1-`/`2-` prefix flipped | raw '
          f'{pc["2b_synthetic_document_stem"]["raw_channel_hits"]}, stem '
          f'{pc["2b_synthetic_document_stem"]["STEM_channel_hits"]} | yes - only '
          f'the stem channel sees it |',
          f'| 3 live banked, string | `R20-H175b_qlane_eval.parquet`, banked at 485 '
          f'of 487 passages in the mix | '
          f'{pc["3_live_banked_string_channel"]["units_hit_any_form"]} of '
          f'{pc["3_live_banked_string_channel"]["units"]} | yes - reproduces the '
          f'banked figure exactly, including its 449-raw / 485-truncated split |',
          f'| 4 live banked, document | the original `R20-H177_eval_B.parquet`, '
          f'banked at 325 of 325 TabFact document stems in the member | '
          f'{pc["4_live_banked_document_channel"]["STEMS_in_the_member"]} of '
          f'{pc["4_live_banked_document_channel"]["eval_tabfact_documents"]} | yes |',
          ""]
    ag = rep.get("cross_arm_antigaming_identity", {}).get("measured_overlap")
    if ag:
        L += ["## The anti-gaming instrument is not one instrument", "",
              f'- **Near-miss half** - {ag["nearmiss_half"]}',
              f'- **bind_row half** - {ag["bind_row_half"]}',
              f'- **Cause** - {ag["cause"]}',
              f'- Membership identical across repeated builds: '
              f'{ag["membership_identical_across_runs"]}; order identical: '
              f'{ag["order_identical_across_runs"]}', ""]
    L += ["## Artifacts", "",
          "- `experiments/grounding-semantic/contract/mechanism_evals_report.json`",
          "- `experiments/grounding-semantic/contract/mechanism_evals_summary.md`",
          "- `experiments/grounding-semantic/contract/mechanism_evals_antigaming_supp.json`",
          "- `experiments/grounding-semantic/contract/mechanism_evals_spotchecks.json` - "
          "the four strongest findings re-derived without the digest machinery",
          "- builders `mechanism_evals_verify.py`, `mechanism_evals_antigaming_supp.py`, "
          "`mechanism_evals_spotchecks.py`, `mechanism_evals_rank.py`",
          "- logs `logs/contract-mechanism-evals.log`, "
          "`logs/contract-mechanism-evals-antigaming.log`", ""]
    return "\n".join(L)


def fmt(v):
    return "n/a" if v is None else v


if __name__ == "__main__":
    main()

"""Enumeration record for the mechanism-eval contract pass - what every top-level
parquet in `experiments/grounding-semantic/` is, and why it was or was not
verified.

CPU only.  No measurement of an eval here; this exists so the scope of
`mechanism_evals_report.json` is auditable rather than asserted.  Every file is
classified, and any file the rules cannot place is listed as UNCLASSIFIED rather
than dropped.

Classes:
  VERIFIED                    an instrument in mechanism_evals_report.json
  OWNED_BY_ANOTHER_AGENT      excluded by the task - being rebuilt or assessed
  ARENA                       the blind evaluation arena, excluded by the task
  TRAINING_MEMBER_OR_LANE     phase-1 scope, verified by its own contract agent
  SUPPLY_OR_INTERMEDIATE      a pool a builder draws from, not itself read
  SCORE_DUMP                  model outputs, carries no claim/evidence pair to
                              verify - a reading OF an instrument, not one

Out: contract/mechanism_evals_inventory.json
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import pathlib

import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
OUT = HERE / "mechanism_evals_inventory.json"
REPORT = HERE / "mechanism_evals_report.json"
NOTE = "Numbers recorded, not adjudicated - the coordinator adjudicates."

VERIFIED = {
    "R18-H150_antigaming_set.parquet": "antigaming_nearmiss_bindrow (the reference "
                                       "set; the other 13 per-arm files are "
                                       "compared against it in "
                                       "mechanism_evals_antigaming_supp.json)",
    "R14-H133_antigaming_traced.parquet": "antigaming_traced",
    "R19_findver_lane.parquet": "findver",
    "R20-H177_eval_C.parquet": "eval_C",
    "R17-H148_probe.parquet": "h148_itemindex_probe",
    "R17-H149_probe.parquet": "h149_roleswap_probe",
    "R18-H150_unitswap_probe.parquet": "h150_unitswap_probe",
    "R15_L1_bindprobe_pairs.parquet": "r15_bindprobe",
    "R15_P1_typeprobe_quads.parquet": "r15_typeprobe",
    "R15_P1_typeprobe_topup_quads.parquet": "r15_typeprobe (concatenated)",
    "R20-G0b_composed_probes.parquet": "g0b_composed_probes",
    "R11-H117_heldout_pairs.parquet": "h117_heldout_pairs",
    "R20-H175b_qlane_eval_clean.parquet": "h175b_eval_clean",
    "R20-H175b_qlane_eval_clean_prefix.parquet": "h175b_eval_clean_prefix",
    "DR_H113_gate_judged.parquet": "dr_h113_gate_judged",
    "R12-H121_gateBC_rows.parquet": "r12_h121_gateBC_rows",
}

OWNED = {
    "R20-H177_eval_B.parquet": "being rebuilt by another agent; used here READ-ONLY "
                               "as live positive control 4 for the document channel",
    "R20-H177_eval_B_rebuilt.parquet": "the rebuild in progress, owned by that agent",
    "R17-H143_evalset.parquet": "being assessed by another agent",
    "R20-H175b_qlane_eval.parquet": "the withdrawn contaminated eval; used here "
                                    "READ-ONLY as live positive control 3",
}

ARENA = {"R21-H179_arena_items.parquet": "the blind arena, excluded by the task"}

MEMBER_OR_LANE = {
    "R17-H146_lane.parquet", "R17-H146_lane_conformed.parquet",
    "R18-H150_scaleunit_lane.parquet", "R20-H174_lane_L1.parquet",
    "R20-H174_lane_L2.parquet", "R20-H174_lane_L2_conformed.parquet",
    "R20-H174_lane_L4.parquet", "R14-H133_lane.parquet",
    "R14-H133_lane.v1-DEFECTIVE.parquet", "R14-H133_lane.v2-SUPERSEDED.parquet",
    "R13-H128_lane.parquet", "R10-H108_pairs.parquet", "DR_lane.parquet",
    "R19_attributionbench_lane.parquet", "R19_factscore_lane.parquet",
    "R19_fava_lane.parquet", "R19_minicheck_lane.parquet",
    "R19_pubhealth_lane.parquet", "R20-H177_lane_B.parquet",
    "R20-H177_lane_C.parquet", "R20-H175b_qlane.parquet",
    "R20-H175b_qlane_repaired.parquet", "R20-H175b_qlane_eval_repaired.parquet",
    "R17-H144_pairs.parquet", "R17-H145_scaleunit.parquet",
    "R17-H143_evalset_source.parquet", "R10-H111_pairs_final.parquet",
    "R10-H111_stage1_pairs.parquet", "R10-H111_stage1_judged.parquet",
    "R13-H129_teacher_targets.parquet",
}

SUPPLY = {
    "R17-H148_blocks.parquet", "R17-H149_passages.parquet",
    "R17-H149_audit_sample.parquet", "R18-H150_edgar_admitted.parquet",
    "R18-H150_edgar_chunks.parquet", "R19-H161_L3_sample.parquet",
    "R19-H162_procedural_geometry.parquet", "R19-H162_pubmedqa_sents.parquet",
    "R19-H162_hotpotqa_sentences.parquet", "DR_pilot_raw.parquet",
    "DR_pilot_longform.parquet", "DR_pilot_longform.attempt1.parquet",
    "DR_pilot_longform.FAILED.parquet", "DR_judged.parquet",
    "R10-H111_stage0b_recons.parquet", "R10-H111_judge_validation.parquet",
    "R13-H128_sample_pairs.parquet", "R15_gate_B5arm8_candidates.parquet",
    "R15_gate_B5arm8_judged.parquet", "R12-H121_gateB_eyeball_sample.parquet",
    "R13-H129_gate_sample.parquet", "R15_gate_B2_absent_flags.parquet",
    "R15_gate_B2_sample.parquet", "R14_H133_triples.parquet",
    "R17-H144_lookup.parquet", "R17-H144_ctrlval.parquet",
    "R17-H144_traces.parquet",
    "DR_H113_gate_samples.parquet",
}


def classify(name):
    if name in VERIFIED:
        return "VERIFIED", VERIFIED[name]
    if name in OWNED:
        return "OWNED_BY_ANOTHER_AGENT", OWNED[name]
    if name in ARENA:
        return "ARENA", ARENA[name]
    if name.endswith("antigaming_set.parquet"):
        return "VERIFIED", ("a per-arm build of the anti-gaming instrument; "
                            "content compared against the reference set in "
                            "mechanism_evals_antigaming_supp.json")
    if name in MEMBER_OR_LANE:
        return "TRAINING_MEMBER_OR_LANE", ("phase-1 scope - a member of the "
                                           "training mix or a lane, verified by "
                                           "its own contract agent")
    if name in SUPPLY:
        return "SUPPLY_OR_INTERMEDIATE", ("a pool or intermediate a builder draws "
                                          "from; not itself read as an instrument")
    return None, None


def main():
    rows = []
    unclassified = []
    for p in sorted(SEM.glob("*.parquet")):
        try:
            cols = pl.scan_parquet(p).collect_schema().names()
        except Exception as e:
            cols = [f"UNREADABLE {e}"]
        cls, why = classify(p.name)
        if cls is None:
            # the residual rule: a file with no (claim, evidence, label) triple
            # cannot be verified against C1/C2/C5 - it is a reading, not an
            # instrument.  Anything else is reported UNCLASSIFIED, never dropped
            has_claim = any(c.startswith("claim") or c in
                            ("statement", "sentence", "seed", "response") for c in cols)
            has_ev = any(c in ("chunk", "evidence", "doc_a", "window", "win_text",
                               "passage", "table_text", "documents") for c in cols)
            if not (has_claim and has_ev):
                cls = "SCORE_DUMP"
                why = ("carries no (claim, evidence) pair - a model reading OF an "
                       "instrument rather than an instrument")
            else:
                cls = "UNCLASSIFIED"
                why = "carries claim and evidence columns but matched no rule"
                unclassified.append(p.name)
        rows.append({"file": p.name, "class": cls, "reason": why,
                     "columns": cols[:14]})

    counts = {}
    for r in rows:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    out = {"artifact": OUT.name,
           "scope": "the enumeration behind contract/mechanism_evals_report.json - "
                    "every top-level parquet in experiments/grounding-semantic/, "
                    "classified, so the pass's scope is auditable rather than "
                    "asserted",
           "files_enumerated": len(rows), "class_counts": counts,
           "UNCLASSIFIED": unclassified,
           "inventory": rows, "note": NOTE}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(counts, indent=1), flush=True)
    print(f"UNCLASSIFIED: {unclassified}", flush=True)
    print(f"wrote {OUT}", flush=True)

    if REPORT.exists():
        rep = json.loads(REPORT.read_text())
        rep["enumeration"] = {"artifact": OUT.name,
                              "files_enumerated": len(rows),
                              "class_counts": counts,
                              "UNCLASSIFIED": unclassified}
        REPORT.write_text(json.dumps(rep, indent=2))
        print(f"linked into {REPORT}", flush=True)


if __name__ == "__main__":
    main()

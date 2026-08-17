"""Assemble contract/phase1_readjudication.json.

Restates all 8 clause verdicts for all 11 loaded members under the AMENDED
contract (amendments C-A1 and C-A2 plus the coordinator's 2026-08-17 C5 scoping
ruling), checks the invariant that justified the amendments, and folds in the
two measurement passes this task ran:

  phase1_readjudication_structural.json   C-A1 test 1 on every loaded member,
                                          plus a uniform predicate-blind
                                          attestation reading for C-A2 tests 2/3
  phase1_readjudication_conformed.json    the same two readings on every
                                          conformed artifact on disk

Phase-1 verdicts and every member-specific number are read from the banked
`*_contract_report.json` / `*_conformed_report.json` files - nothing is
re-derived. Measurement only; the coordinator adjudicates.

Run:  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run python \
      experiments/grounding-semantic/contract/phase1_readjudication_build.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import pathlib

HERE = pathlib.Path(__file__).parent

NOTE = "Numbers recorded, not adjudicated - the coordinator adjudicates."

STRUCT = json.loads((HERE / "phase1_readjudication_structural.json").read_text())
CONF = json.loads((HERE / "phase1_readjudication_conformed.json").read_text())


def phase1_verdicts(fname):
    d = json.loads((HERE / fname).read_text())
    out = {}
    for c in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"):
        v = None
        if isinstance(d.get(c), dict):
            for k in ("verdict", "status", "result"):
                if k in d[c]:
                    v = d[c][k]
                    break
        for holder in ("clauses", "verdicts", "clause_verdicts"):
            if v is None and isinstance(d.get(holder), dict) and c in d[holder]:
                x = d[holder][c]
                v = x if isinstance(x, str) else next(
                    (x[k] for k in ("verdict", "status", "result") if k in x), None)
        if v == "GREEN":
            v = "PASS"
        if v == "applicable" or v is None:
            v = "NOT-APPLICABLE"
        out[c] = v
    return out


P1 = {m: phase1_verdicts(f"{m}_contract_report.json") for m in (
    "attr_pool", "frame_reject", "halueval", "path_bind", "psiloqa",
    "quant_misbind", "quant_scale_unit", "ragtruth_en", "ragtruth_translated",
    "tabfact", "vitaminc")}
# vitaminc's report carries C5 as {"applicable": false, ...} - normalise
P1["vitaminc"]["C5"] = "NOT-APPLICABLE"

# --------------------------------------------------------------------------- #
# the restatement, clause by clause
# --------------------------------------------------------------------------- #
# C1 - each member's own predicate-sensitive instrument reading, taken from its
# banked report. The uniform predicate-BLIND reading is attached from the
# measurement pass.
C1_SENSITIVE = {
    "attr_pool": {
        "instrument": "claim-to-pool containment (the pooled chunk IS the object "
                      "the lane corrupts by removing the truth passage)",
        "negative": 0.2293, "positive": 0.5299, "strictly_below": True,
        "source": "attr_pool_contract_report.json C1"},
    "frame_reject": {
        "instrument": "claim-to-chunk containment (the negative claim is drawn "
                      "from a closed contentless inventory)",
        "negative": 0.0, "positive": 0.0565, "strictly_below": True,
        "source": "frame_reject_contract_report.json C1 attestation_thresholds"},
    "halueval": {
        "instrument": "claim-to-evidence containment, untruncated presentation",
        "negative": 0.1032, "positive": 0.6592, "strictly_below": True,
        "source": "halueval_contract_report.json C1"},
    "path_bind": {
        "instrument": "asserted path attested as a CONTIGUOUS bare run - the only "
                      "order-sensitive reading; the lane corrupts token ORDER, so "
                      "every bag-of-tokens instrument is predicate-blind here",
        "negative": 0.0, "positive": 1.0, "strictly_below": True,
        "source": "path_bind_contract_report.json C1 "
                  "executor_added_reported_separately",
        "caveat": "the banked report files this reading as EXECUTOR-ADDED and "
                  "'joining no bar'. Under C-A1 an instrument sensitive to the "
                  "corrupted predicate is what C1 now REQUIRES, so this reading "
                  "is the mandated one rather than an extra. The re-basing is a "
                  "change of justification, not of verdict"},
    "psiloqa": {
        "instrument": "claim-to-passage containment, banked ASCII primary",
        "negative": 0.0292, "positive": 0.1383, "strictly_below": True,
        "source": "psiloqa_contract_report.json C1 containment_banked_ascii"},
    "quant_misbind": {
        "instrument": "binding-level re-derivation against the source table - "
                      "every row re-checked, full population",
        "negative": 0.0, "positive": 1.0, "strictly_below": True,
        "source": "quant_misbind_contract_report.json C1 "
                  "discriminating_measurement",
        "caveat": "the token instrument is blind by construction (the negative's "
                  "numeral is a real cell of the same table) and reads the legs "
                  "EXACTLY EQUAL at fully-attested 0.1086. Only the binding-level "
                  "re-derivation separates them"},
    "quant_scale_unit": {
        "instrument": "I3 unit-resolved containment - I1 after mapping the "
                      "claim's spelled-out unit onto the abbreviation the "
                      "evidence uses, from the lane's own banked UNITS table",
        "negative": 0.03213, "positive": 0.101083, "strictly_below": True,
        "source": "quant_scale_unit_contract_report.json C1 "
                  "instrument_I3_unit_resolved",
        "caveat": "I1/I2 are unit-blind and read the legs equal (0.0213 vs "
                  "0.0220 at >= 0.90, 0.0 vs 0.0 fully attested). C-A1 bullet 4 "
                  "names exactly this case: a predicate-blind instrument showing "
                  "no separation is not evidence of incommensurability"},
    "ragtruth_en": {
        "instrument": "claim-to-evidence containment, untruncated (flagship) "
                      "presentation",
        "negative": 0.0067, "positive": 0.0790, "strictly_below": True,
        "source": "ragtruth_en_contract_report.json C1 presentations"},
    "ragtruth_translated": {
        "instrument": "Unicode \\w+ containment with CJK bigram splitting - mean "
                      "containment, the only leg-paired reading the banked report "
                      "carries",
        "negative": 0.4506, "positive": 0.6147, "strictly_below": True,
        "source": "ragtruth_translated_contract_report.json C1 measured_primary",
        "caveat": "the banked report records the negative leg's >= 0.90 rate "
                  "(0.0015) but NOT the positive leg's, so test 2's rate limb "
                  "cannot be read from it. This task measured both legs under the "
                  "uniform blind instrument instead - 0.0743 vs 0.1153, strictly "
                  "below - and the mean-containment separation above carries the "
                  "banked reading"},
    "tabfact": {
        "instrument": "claim-to-table containment, untruncated presentation",
        "negative": 0.0011, "positive": 0.0014, "strictly_below": True,
        "source": "tabfact_contract_report.json clauses.C1 (fully-attested share)",
        "caveat": "separation is real but thin, and the instrument is close to "
                  "blind for a member whose label turns on reading values out of "
                  "a table rather than on token presence"},
    "vitaminc": {
        "instrument": "claim-to-evidence containment, Unicode primary",
        "negative": 0.0169, "positive": 0.1227, "strictly_below": True,
        "source": "vitaminc_contract_report.json C1"},
}

C6_AMENDED = {
    "attr_pool": ("FAIL",
                  "The mix-supplied (claim -> supporting evidence) lookup the mix's "
                  "own vitaminc member provides separates the legs at within-pair "
                  "0.9999 on 3,999 VitaminC truth_removed pairs. C-A2 preserves "
                  "exactly this channel by name. Unaffected by either amendment"),
    "frame_reject": ("PASS",
                     "Mix-association channel, coverage 0.0003 (2 of 8,000 rows), "
                     "best feature AUROC 0.5001. Coverage is near zero but not "
                     "zero, so the test is computable and at chance"),
    "halueval": ("PASS",
                 "Within-pair evidence-only channel exactly 0.5000 BY MEASUREMENT "
                 "- all 20,000 pairs carry byte-identical evidence on both legs, "
                 "so no feature keyed on the shared field can separate the "
                 "classes. GAP: no cross-member mix-supplied channel was measured "
                 "in phase 1; the amendment does not close that gap"),
    "path_bind": ("NOT-APPLICABLE",
                  "PASS -> NOT-APPLICABLE. The literal instrument keyed on doc_id "
                  "has coverage 0.0 - every doc_id belongs to exactly one pair. "
                  "C-A2: zero key coverage makes C6 NOT-APPLICABLE and no proxy "
                  "is substituted. The three executor-added association features "
                  "(0.4978 to 0.5187) are diagnostics"),
    "psiloqa": ("NOT-APPLICABLE",
                "PASS -> NOT-APPLICABLE. The eval-facing channel has coverage on "
                "the two WITHDRAWN R20-H175b PsiloQA-derived evals only (98%, "
                "value 0.2403) and coverage 0.0 on every live surface. The "
                "within-member key-repeat reading beats its own majority baseline "
                "by -0.0239 and is a diagnostic under C-A2"),
    "quant_misbind": ("PASS",
                      "Mix-association channel keyed on the shared chunk against "
                      "the rest of the mix: AUROC 0.5000 at 3.09% coverage "
                      "(926 of 30,000 rows). Mix-supplied, computable, at chance"),
    "quant_scale_unit": ("PASS",
                         "Mix-association channel keyed on the normalised chunk: "
                         "AUROC 0.4876 at 0.79% coverage (44 of 5,540 rows)"),
    "ragtruth_en": ("NOT-APPLICABLE",
                    "FAIL -> NOT-APPLICABLE. The eval-facing test has ZERO key "
                    "coverage (0 of 2,700 eval rows share a context key with "
                    "training). The 0.6509 that carried the FAIL is the "
                    "within-member leave-one-out lookup, which C-A2 demotes BY "
                    "NAME to a reported diagnostic and records as a corpus "
                    "property (six responses share one source passage)"),
    "ragtruth_translated": ("NOT-APPLICABLE",
                            "PASS -> NOT-APPLICABLE on unchanged evidence. Eval "
                            "side UNDEFINED at coverage 0.0; the report's own "
                            "within-member figure is the SAME 0.6509 that failed "
                            "ragtruth_en. Under C-A2 the two members now read "
                            "alike, which removes a phase-1 contradiction"),
    "tabfact": ("PASS",
                "Eval-facing channel computable at 100% coverage against "
                "R20-H177_eval_B's TabFact half: claim-overlap AUROC 0.503, mean "
                "max-Jaccard 0.1699 against the poisoned reference 0.6230"),
    "vitaminc": ("NOT-APPLICABLE",
                 "PASS -> NOT-APPLICABLE on unchanged evidence. The only "
                 "key-sharing surface (R19-H166-A1 holdout, 38,126 rows) has an "
                 "EMPTY key join - 0 rows share a page or case_id. The member's "
                 "own conformed report states the same move"),
}

C5_AMENDED = {
    "attr_pool": ("PASS",
                  "FAIL -> PASS. The FAIL rested SOLELY on claim-to-chunk "
                  "containment at 0.7030 against [0.45, 0.55]; C-A1 scopes that "
                  "joint channel out of C5 and assigns it to C1. Every remaining "
                  "channel holds: claim-only converged probe 0.5281 (< 0.55), "
                  "within-pair claim-only worst 0.5594 (< 0.60), claim char "
                  "length 0.4765, claim token count 0.4773, chunk char length "
                  "0.5112 - worst remaining deviation 0.0235. FINDING, not "
                  "gating: the executor-added chunk-only probe reads within-pair "
                  "0.5758 on truth_removed, an EVIDENCE-ALONE channel that IS "
                  "inside C-A1's narrowed scope but sits outside the registered "
                  "conjunction, which C5's last bullet keeps separate"),
    "frame_reject": ("FAIL",
                     "STANDS. Claim-only converged probe AUROC 1.0000 against "
                     "< 0.55 and within-pair claim-only 1.0000 in both families "
                     "against < 0.60. Both are CLAIM-ALONE channels, squarely "
                     "inside C-A1's narrowed scope. C-A1's own text names this "
                     "member as one the amendment does not rescue"),
    "halueval": ("FAIL",
                 "STANDS. Claim-only converged probe 0.9519 against < 0.55 and "
                 "within-pair 0.9666 against < 0.60 - claim-alone channels inside "
                 "the narrowed scope. The length-matched summarization subset "
                 "still reads 0.9408 with its length channel at chance (0.4930), "
                 "so no length filter repairs it. The coordinator's 2026-08-17 "
                 "ruling settles applicability in the same direction: the "
                 "negative leg is ChatGPT-written, so C5 binds"),
    "path_bind": ("PASS", "PASS unchanged; narrowing C5's scope can only remove "
                          "channels, never add one"),
    "quant_misbind": ("PASS", "PASS unchanged; narrowing can only remove channels"),
    "quant_scale_unit": ("PASS",
                         "PASS unchanged. C-A1 was adopted after the FIRST pass "
                         "failed C5 on this lane's containment channel; the "
                         "banked report already reads PASS"),
    "psiloqa": ("NOT-APPLICABLE-CONTESTED", None),
    "ragtruth_en": ("NOT-APPLICABLE-CONTESTED", None),
    "ragtruth_translated": ("NOT-APPLICABLE-CONTESTED", None),
    "tabfact": ("NOT-APPLICABLE-CONTESTED", None),
    "vitaminc": ("NOT-APPLICABLE-CONTESTED", None),
}

C5_CONTESTED_NOTE = (
    "Phase 1 returned NOT-APPLICABLE on the narrow reading (C5's body binds "
    "'every constructed lane and every paired-contrast eval'; this is a source "
    "corpus). The coordinator's 2026-08-17 ruling - 'C5 binds any member whose "
    "NEGATIVE leg was GENERATED rather than observed, regardless of who "
    "generated it' - re-opens applicability and requires a generated-vs-observed "
    "classification per corpus that is an adjudication, not a measurement. Both "
    "branches are priced below."
)

C5_CONTESTED = {
    "psiloqa": {
        "negative_leg_origin": "an LLM answer whose unsupported spans a GPT-4o "
                               "annotator FOUND after the fact - the label is "
                               "discovered, not imposed at generation time",
        "claim_only_probe": None,
        "claim_only_note": "NOT MEASURED in phase 1 - the only member of the "
                           "eleven with no claim-only reading on record",
        "rows": 61712},
    "ragtruth_en": {
        "negative_leg_origin": "a naturally occurring LLM response whose "
                               "hallucinated spans a HUMAN annotator found after "
                               "the fact - label discovered, not imposed",
        "claim_only_probe": 0.7965,
        "claim_only_note": "executor-added, reported separately, no bar attached",
        "rows": 15090},
    "ragtruth_translated": {
        "negative_leg_origin": "as ragtruth_en, with the spans re-aligned after "
                               "machine translation",
        "claim_only_probe": 0.7810,
        "claim_only_note": "executor-added; per-language 0.7771 to 0.7821",
        "rows": 105630},
    "tabfact": {
        "negative_leg_origin": "a counterfactual statement WRITTEN BY a human "
                               "annotator to be REFUTED by the table - authored "
                               "to the label, which reads as generated",
        "claim_only_probe": 0.6031,
        "claim_only_note": "executor-added, held-out AUROC on table-disjoint "
                           "folds. PHASE-1 SYNTHESIS TRANSCRIPTION ERROR: its C5 "
                           "line attributes 0.5985 to tabfact; 0.5985 is the "
                           "CONFORMED member's reading, the parent reads 0.6031",
        "rows": 92585},
    "vitaminc": {
        "negative_leg_origin": "a human-authored claim over a Wikipedia revision "
                               "pair, labelled REFUTES or NOT ENOUGH INFO by "
                               "annotation of that pair",
        "claim_only_probe": 0.4998,
        "claim_only_note": "executor-added; at chance, so this member clears the "
                           "claim-only bar under EITHER branch",
        "rows": 370653},
}

# clauses neither amendment nor the ruling touches
UNTOUCHED = {
    "C2": "Disjointness from evaluation surfaces. Neither amendment mentions C2 "
          "and neither changes a string form, a surface or a direction.",
    "C3": "Split semantics. Untouched.",
    "C4": "Contamination census with a live positive control. Untouched.",
    "C7": "Declared units and volume. Untouched.",
    "C8": "Provenance, licence and internal structure. Untouched.",
}


def build():
    members = {}
    for m, p1 in P1.items():
        s = STRUCT["members"][m]
        blind = s["uniform_containment_C1"]
        raw = s["structural_C1"]["raw"]
        trunc = s["structural_C1"]["evidence_truncated_1500"]
        sens = C1_SENSITIVE[m]

        # ---- C1 ----------------------------------------------------------- #
        t1_fires = raw["both_label_pairs"] > 0
        c1 = {
            "phase1": p1["C1"],
            "amended": ("PASS" if not t1_fires else
                        "PASS on tests 2 and 3; TEST 1 FIRES - new measurement"),
            "test_1_structural": {
                "both_label_pairs_raw": raw["both_label_pairs"],
                "rows_covered_raw": raw["rows_covered"],
                "row_share_of_member": round(raw["rows_covered"] / s["rows"], 6),
                "both_label_pairs_evidence_truncated_1500":
                    trunc["both_label_pairs"],
                "rows_covered_evidence_truncated_1500": trunc["rows_covered"],
                "fires": t1_fires,
            },
            "test_2_strict_separation_predicate_sensitive": sens,
            "test_3_absolute_levels_uniform_blind_instrument": {
                "instrument": blind["instrument"],
                "positive_leg": blind["positive_leg"],
                "negative_leg": blind["negative_leg"],
                "neg_strictly_below_pos_at_ge_0.90":
                    blind["test2_neg_strictly_below_pos_at_ge_0.90"],
                "neg_strictly_below_pos_at_fully_attested":
                    blind["test2_neg_strictly_below_pos_at_eq_1.00"],
            },
        }
        if p1["C1"] == "FAIL":
            c1["mechanism_of_the_move"] = (
                "C-A2 STRUCK the 'within 0.10' band this FAIL was taken on. The "
                "restated tests are met: test 1 does not fire and the negative "
                "leg is strictly below the positive under an instrument "
                "sensitive to the predicate the member corrupts")
        elif t1_fires:
            c1["mechanism_of_the_move"] = (
                "No move from the amendments. C-A1 introduced a test that did "
                "not exist in phase 1 and it fires here on a NEW measurement - "
                "this tightens, it does not rescue")
        else:
            c1["mechanism_of_the_move"] = "No move; the amended tests reproduce PASS"

        # ---- C5 ----------------------------------------------------------- #
        v5, why5 = C5_AMENDED[m]
        c5 = {"phase1": p1["C5"], "amended": v5,
              "basis": why5 if why5 else C5_CONTESTED_NOTE}
        if v5 == "NOT-APPLICABLE-CONTESTED":
            c5["branch"] = {
                "if_observed": "NOT-APPLICABLE - phase-1 verdict stands",
                "if_generated": (
                    "C5 binds; the claim-only bar is < 0.55 and the measured "
                    "value decides"),
                **C5_CONTESTED[m],
            }

        # ---- C6 ----------------------------------------------------------- #
        v6, why6 = C6_AMENDED[m]

        members[m] = {
            "rows": s["rows"],
            "dann_groups": s["dann_groups"],
            "phase1_conforming": all(
                v in ("PASS", "NOT-APPLICABLE") for v in p1.values()),
            "clauses": {
                "C1": c1,
                "C2": {"phase1": p1["C2"], "amended": p1["C2"],
                       "basis": UNTOUCHED["C2"]},
                "C3": {"phase1": p1["C3"], "amended": p1["C3"],
                       "basis": UNTOUCHED["C3"]},
                "C4": {"phase1": p1["C4"], "amended": p1["C4"],
                       "basis": UNTOUCHED["C4"]},
                "C5": c5,
                "C6": {"phase1": p1["C6"], "amended": v6, "basis": why6},
                "C7": {"phase1": p1["C7"], "amended": p1["C7"],
                       "basis": UNTOUCHED["C7"]},
                "C8": {"phase1": p1["C8"], "amended": p1["C8"],
                       "basis": UNTOUCHED["C8"]},
            },
        }

    # ---------------------------------------------------------------- #
    # compact table
    # ---------------------------------------------------------------- #
    table = {}
    for m, rec in members.items():
        table[m] = {c: f"{rec['clauses'][c]['phase1']} -> {rec['clauses'][c]['amended']}"
                    for c in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")}

    # ---------------------------------------------------------------- #
    # invariant
    # ---------------------------------------------------------------- #
    moves = []
    for m, rec in members.items():
        for c, x in rec["clauses"].items():
            a = x["amended"].split(" on tests")[0]
            if x["phase1"] == "FAIL" and a != "FAIL":
                moves.append({"member": m, "clause": c,
                              "from": "FAIL", "to": a,
                              "mechanism": x.get("mechanism_of_the_move")
                              or x.get("basis")})
    return members, table, moves


def main():
    members, table, moves = build()

    out = {
        "task": "Restate every phase-1 contract verdict under amendments C-A1 and "
                "C-A2 (and the coordinator's 2026-08-17 C5 scoping ruling), verify "
                "the amendment invariant, re-run C-A1's structural C1 test on every "
                "member, and read the conformed artifacts",
        "contract": "docs/experiments/dataset-contract.md",
        "amendments_applied": [
            "C-A1 - C5's surface-parity requirement scopes only to channels that "
            "do NOT read the claim-evidence relation; containment is a JOINT "
            "feature governed by C1, where separation is required",
            "C-A2 - C1's 'within 0.10' band STRUCK; tests are (1) structural, "
            "(2) strict separation under a predicate-sensitive instrument, "
            "(3) absolute levels always reported. C6 binds mix-supplied "
            "associations; a within-member leave-one-out lookup is a diagnostic; "
            "zero eval key coverage makes C6 NOT-APPLICABLE",
            "Coordinator ruling 2026-08-17 - C5 binds any member whose NEGATIVE "
            "leg was GENERATED rather than observed, regardless of who generated it",
        ],
        "compute": "CPU only, CUDA_VISIBLE_DEVICES empty, HF_HUB_OFFLINE=1. "
                   "GPUs 0/1/2 carry live training draws and were not touched",
        "mix": STRUCT["mix"],
        "live_positive_control": {
            "gate": "C-A1 test 1, the structural C1 test",
            "fed": "the withdrawn poisoned R20-H175b_qlane.parquet (17,972 rows), "
                   "known bad by construction - passage and claim held fixed with "
                   "the label flipped on question relevance",
            "registered_expectation": "8,986 pairs / 17,972 rows",
            "measured": STRUCT["live_positive_control"]["structural_C1"],
            "fires_as_registered":
                STRUCT["live_positive_control"]["control_fires_as_registered"],
            "second_gate": "C-A2 test 2, strict separation",
            "second_gate_measured":
                STRUCT["live_positive_control"]["uniform_containment_C1"],
            "second_gate_reading":
                "both legs identical to six decimals - mean 0.815806, rate "
                ">= 0.90 0.665925, fully attested 0.614511 - so the negative leg "
                "is NOT strictly below the positive on either limb and test 2 "
                "fires. Both gates proven live before any clean number is read",
        },
        "restated_table": table,
        "restated_table_ascii": ascii_table(table),
        "clause_rollup": clause_rollup(members),
        "conforming_rollup": conforming_rollup(members),
        "members": members,
        "invariant_check": invariant_block(moves),
        "structural_C1_rerun": structural_block(),
        "conformed_artifacts": conformed_block(),
        "findings": findings_block(),
        "artifacts": [
            "experiments/grounding-semantic/contract/phase1_readjudication.json",
            "experiments/grounding-semantic/contract/"
            "phase1_readjudication_structural.json",
            "experiments/grounding-semantic/contract/"
            "phase1_readjudication_conformed.json",
            "experiments/grounding-semantic/contract/"
            "phase1_readjudication_structural.py",
            "experiments/grounding-semantic/contract/"
            "phase1_readjudication_conformed.py",
            "experiments/grounding-semantic/contract/phase1_readjudication_build.py",
            "logs/phase1-readjudication-structural.log",
            "logs/phase1-readjudication-conformed.log",
        ],
        "note": NOTE,
    }
    p = HERE / "phase1_readjudication.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}", flush=True)
    for m, row in table.items():
        print(f"{m:22s} " + "  ".join(
            f"{c}:{row[c]}" for c in ("C1", "C5", "C6")), flush=True)


CLAUSES = ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")
MIX_ROWS = 760618


def _short(cell):
    a, b = cell.split(" -> ", 1)
    f = {"PASS": "PASS", "FAIL": "FAIL", "NOT-APPLICABLE": "n/a",
         "NOT-APPLICABLE-CONTESTED": "n/a?"}
    lhs = f.get(a, a[:4])
    rhs = "PASS*" if "TEST 1 FIRES" in b else f.get(b.split(" on tests")[0], b[:4])
    return f"{lhs}>{rhs}"


def ascii_table(table):
    w = 21
    lines = ["member".ljust(w) + "".join(c.ljust(11) for c in CLAUSES),
             "-" * (w + 11 * len(CLAUSES))]
    for m, row in table.items():
        lines.append(m.ljust(w) + "".join(_short(row[c]).ljust(11) for c in CLAUSES))
    lines += [
        "",
        "reads phase1>amended.  n/a = NOT-APPLICABLE.  n/a? = applicability "
        "re-opened by the coordinator's generated-vs-observed C5 ruling and not "
        "settled here.",
        "PASS* = C1 tests 2 and 3 hold, but C-A1's structural test 1 - which did "
        "not exist in phase 1 - fires on a small number of rows.",
    ]
    return "\n".join(lines)


def clause_rollup(members):
    out = {}
    for c in CLAUSES:
        fails = [m for m, r in members.items()
                 if r["clauses"][c]["amended"].startswith("FAIL")]
        contested = [m for m, r in members.items()
                     if "CONTESTED" in r["clauses"][c]["amended"]]
        na = [m for m, r in members.items()
              if r["clauses"][c]["amended"] == "NOT-APPLICABLE"]
        rows = sum(members[m]["rows"] for m in fails)
        p1_fails = [m for m, r in members.items() if r["clauses"][c]["phase1"] == "FAIL"]
        out[c] = {
            "phase1_fail_members": p1_fails,
            "phase1_fail_rows": sum(members[m]["rows"] for m in p1_fails),
            "amended_fail_members": fails,
            "amended_fail_rows": rows,
            "amended_fail_share_of_mix": round(rows / MIX_ROWS, 6),
            "amended_not_applicable": na,
            "applicability_contested": contested,
        }
    out["C1"]["amended_structural_test_1_fires"] = {
        "members": ["vitaminc", "tabfact", "psiloqa"],
        "pairs": 165, "rows": 336, "share_of_mix": round(336 / MIX_ROWS, 6),
        "note": "a NEW test, not a re-reading. Reported, not adjudicated"}
    return out


C1_QUALIFIED = {
    "path_bind": "C1 holds ONLY on the order-sensitive contiguous-run reading "
                 "(negative 0.0, positive 1.0). Under the campaign's own "
                 "bag-of-tokens instrument the negative leg is ABOVE the positive "
                 "at >= 0.90 (0.3910 vs 0.3890) and the fully-attested rates are "
                 "EXACTLY EQUAL at 0.1742 - the signature C-A2 names. The banked "
                 "report files the carrying reading as executor-added",
    "quant_misbind": "C1 holds ONLY on the binding-level re-derivation (negative "
                     "0.0, positive 1.0). Token containment reads the legs "
                     "EXACTLY EQUAL fully attested at 0.1086",
    "quant_scale_unit": "C1 holds ONLY on the unit-resolved instrument (0.0321 vs "
                        "0.1011). The unit-blind instruments read the legs equal",
}


def conforming_rollup(members):
    amended, changed, struct_new = {}, [], []
    for m, r in members.items():
        fails = [c for c in CLAUSES if r["clauses"][c]["amended"].startswith("FAIL")]
        contested = [c for c in CLAUSES
                     if "CONTESTED" in r["clauses"][c]["amended"]]
        struct = r["clauses"]["C1"]["test_1_structural"]["fires"]
        p1_fails = [c for c in CLAUSES if r["clauses"][c]["phase1"] == "FAIL"]
        state = ("CONFORMING" if not fails and not struct else "NON-CONFORMING")
        if state == "CONFORMING" and contested:
            state = "CONFORMING pending the C5 applicability ruling"
        amended[m] = {
            "phase1_failed_clauses": p1_fails,
            "amended_failed_clauses": fails,
            "amended_structural_C1_fires": struct,
            "applicability_contested": contested,
            "state": state,
        }
        if m in C1_QUALIFIED:
            amended[m]["C1_instrument_qualification"] = C1_QUALIFIED[m]
        if sorted(p1_fails) != sorted(fails):
            changed.append(m)
        if struct:
            struct_new.append(m)
    return {
        "per_member": amended,
        "members_whose_failed_clause_set_changed": changed,
        "members_carrying_a_NEW_structural_C1_hit": struct_new,
        "members_that_become_CONFORMING_from_the_amendments": [],
        "headline": "NO member moves from NON-CONFORMING to CONFORMING under "
                    "either amendment. Five members shed at least one FAIL and "
                    "every one of them still carries another. C-A1's own claim - "
                    "'the amendment rescues no member' - extends to C-A2 and is "
                    "reproduced here on measurement",
        "the_one_state_change_in_the_round": {
            "artifact": "attr_pool_conformed (R20-H174_lane_L2_conformed.parquet, "
                        "4,442 rows)",
            "what": "its sole surviving FAIL was the C5 containment channel C-A1 "
                    "removes from C5. Under the amended contract it reads PASS or "
                    "NOT-APPLICABLE on all eight clauses",
            "caveat": "it delivers 4,442 rows against a registered 20,000-30,000 "
                      "band and is loaded by no run",
        },
    }


def invariant_block(moves):
    return {
        "invariant_as_stated_by_the_coordinator":
            "no member moves FAIL to PASS on anything other than the two "
            "mis-specified tests - C1's struck 'within 0.10' band and C5's "
            "containment channel",
        "verdict_on_the_invariant": "HOLDS on direction, with one correction to "
                                    "its wording",
        "correction": {
            "what": "THREE mechanisms are needed to account for every withdrawal, "
                    "not two. C-A2 also rescopes C6, and that is what withdraws "
                    "ragtruth_en's C6 FAIL - a change the coordinator's own "
                    "'withdrawn' list includes but the invariant's wording omits",
            "severity": "wording gap in the invariant, NOT a defect in the "
                        "amendment. Every withdrawal traces to a named clause in "
                        "the adopted text; none traces to an unnamed one",
        },
        "every_FAIL_to_not_FAIL_move": moves,
        "mechanism_census": {
            "C1_struck_band": ["ragtruth_en", "psiloqa", "quant_scale_unit",
                               "frame_reject"],
            "C5_containment_channel": ["attr_pool"],
            "C6_mix_supplied_scoping": ["ragtruth_en"],
            "any_other_mechanism": [],
        },
        "why_no_other_clause_can_move": {
            "C2_C3_C4_C7_C8": "neither amendment nor the ruling names any of "
                              "them, changes a string form, a surface, a split "
                              "axis, a census instrument, a unit or a provenance "
                              "field. Their verdicts are carried through "
                              "unchanged and were re-read from the banked reports "
                              "rather than re-derived",
            "monotonicity_of_C5": "C-A1 narrows C5's scope, which can only REMOVE "
                                  "channels. No C5 PASS can become a FAIL from "
                                  "narrowing, and no FAIL can become a PASS unless "
                                  "the removed channel was the sole binding one - "
                                  "true of attr_pool and of no other member",
        },
        "movement_in_the_other_direction": {
            "statement": "the amendments TIGHTEN in three places. None of these is "
                         "a rescue and none is adjudicated here",
            "items": [
                {"what": "C-A1's structural test did not exist in phase 1 and "
                         "fires on three members no phase-1 report tested",
                 "members": {"vitaminc": "115 pairs / 236 rows",
                             "tabfact": "45 pairs / 90 rows",
                             "psiloqa": "5 pairs / 10 rows"}},
                {"what": "C-A2 test 2 replaces an absolute-gap band with a "
                         "STRICT-inequality test, and two lanes read their legs "
                         "EXACTLY EQUAL under the predicate-blind instrument",
                 "members": {
                     "path_bind": "fully attested 0.1742 on both legs, and at "
                                  ">= 0.90 the negative leg is ABOVE the positive "
                                  "(0.3910 vs 0.3890). Only the order-sensitive "
                                  "contiguous-run reading (0.0 vs 1.0) separates "
                                  "them, and the banked report files that reading "
                                  "as executor-added",
                     "quant_misbind": "fully attested EXACTLY EQUAL at 0.1086; at "
                                      ">= 0.90 the margin is 0.000067. Only the "
                                      "binding-level re-derivation (0.0 vs 1.0) "
                                      "separates them",
                     "quant_scale_unit": "fully attested 0.0 on both legs under "
                                         "the blind instrument; the unit-resolved "
                                         "instrument separates at 0.0321 vs 0.1011"}},
                {"what": "C-A2's C6 scoping turns three further PASS verdicts into "
                         "NOT-APPLICABLE on unchanged evidence",
                 "members": {"path_bind": "key coverage 0.0",
                             "ragtruth_translated": "eval key coverage 0.0",
                             "vitaminc": "eval key join empty on 38,126 rows",
                             "psiloqa": "live-surface eval key coverage 0.0"}},
            ],
        },
    }


def structural_block():
    per = {}
    for m, s in STRUCT["members"].items():
        per[m] = {
            "rows": s["rows"],
            "distinct_pairs_raw": s["distinct_pairs_raw"],
            "raw": s["structural_C1"]["raw"],
            "evidence_truncated_1500": s["structural_C1"]["evidence_truncated_1500"],
            "whitespace_collapsed_casefolded":
                s["structural_C1"]["whitespace_collapsed_casefolded"],
        }
    return {
        "question": "does any member contain a (claim, evidence) pair carrying "
                    "both labels?",
        "answer": "YES on three of eleven, all source corpora, all at a very small "
                  "row share; NO on all five constructed lanes",
        "headline": {
            "vitaminc": "115 pairs / 236 rows (0.0637% of 370,653)",
            "tabfact": "45 pairs / 90 rows (0.0972% of 92,585)",
            "psiloqa": "5 pairs / 10 rows (0.0162% of 61,712)",
            "every_other_member": "0 pairs / 0 rows",
            "whole_mix_cross_member": "165 pairs / 336 rows (0.0442% of 760,618) "
                                      "- no pair crosses a member boundary, the "
                                      "165 is exactly 115 + 45 + 5",
        },
        "reference_the_amendment_asserted":
            "C-A1's pre-adoption control claims 0 pairs in frame_reject, "
            "attr_pool, path_bind, R17-H146_lane and R18-H150_scaleunit_lane. "
            "REPRODUCED EXACTLY here - all five read 0 in all three string forms. "
            "The amendment never ran the test on a source corpus, which is where "
            "the three hits are",
        "corroboration": "vitaminc's own conforming rebuild removed 236 rows under "
                         "a filter it named 'F2_c1_structural_label_conflict'. "
                         "That is the same 236 rows measured here from the parent, "
                         "independently",
        "presentation_conditional_finding": {
            "what": "under the 1,500-char TRUNCATED evidence presentation the "
                    "count rises, because distinct long evidence collapses to "
                    "identical prefixes",
            "attr_pool": "0 raw -> 263 pairs / 558 rows truncated (2.61% of the "
                         "lane) - the truth passage the lane removes often sits "
                         "beyond char 1,500, so the cut hides the only difference "
                         "between the legs",
            "vitaminc": "115 raw -> 128 pairs / 262 rows truncated",
            "whole_mix": "165 raw -> 441 pairs / 920 rows truncated",
            "which_presentation_the_flagship_serves":
                "R16-H142_G1_arm.untruncated_evidence plus 1,500/750 windowing, "
                "so the served model sees the whole chunk across windows and the "
                "RAW reading is the operative one. The 1,500-char cut is the "
                "R10-H108 in-domain presentation",
        },
        "per_member": per,
        "whole_mix_cross_member": STRUCT["whole_mix_cross_member"],
        "instrument_note": "grouping is on 64-bit hashes of each side and every "
                           "flagged group is re-verified on the literal strings, "
                           "so a hash collision cannot manufacture a hit",
    }


def conformed_block():
    a = CONF["artifacts"]
    return {
        "rule_followed": "conformed artifacts were READ, not rebuilt",
        "psiloqa_conformed": {
            "rows": a["psiloqa_conformed"]["rows"],
            "built_for": "C1 (the struck band) and C2",
            "clears_what_it_was_built_for":
                "C2 YES - zero against all nine surfaces in every string form and "
                "direction. C1 - the clause it was built for NO LONGER EXISTS in "
                "the form it targeted",
            "cost_now_unjustified": "9,859 of its 11,238 dropped rows (16.0% of "
                                    "the member) are the 42-content-token claim "
                                    "cap, selected to move the C1 delta across "
                                    "the band C-A2 struck. The unconformed member "
                                    "already passes the restated test (0.0292 < "
                                    "0.1383). Only the 1,379 C2-collision rows "
                                    "are still binding",
            "new_finding": "the conformed member STILL carries C-A1's structural "
                           "hit - 5 pairs / 10 rows, unchanged from the parent. "
                           "Its pipeline was built before C-A1 and has no "
                           "structural filter",
            "structural_C1": a["psiloqa_conformed"]["structural_C1"],
        },
        "vitaminc_conformed": {
            "rows": a["vitaminc_conformed"]["rows"],
            "built_for": "C2, plus C-A1's structural C1 (its own F2 filter)",
            "clears_what_it_was_built_for": "YES on both. C2 PASS; structural C1 "
                                            "0 pairs / 0 rows, re-measured here "
                                            "from the artifact",
            "cost": "260 rows, 0.0701% - 24 rows for the C2 collision and 236 for "
                    "the structural conflict",
            "caveat": "under the 1,500-char truncated presentation it still reads "
                      "13 pairs / 26 rows; the filter was applied on raw strings",
            "structural_C1": a["vitaminc_conformed"]["structural_C1"],
        },
        "tabfact_conformed": {
            "rows": a["tabfact_conformed"]["rows"],
            "built_for": "C2, C3, C8",
            "clears_what_it_was_built_for": "YES - all eight clauses PASS or "
                                            "NOT-APPLICABLE in its own report",
            "bonus": "the document cut incidentally removes all 45 of the parent's "
                     "structural hits: 0 pairs / 0 rows in all three string forms, "
                     "measured here from the artifact",
            "cost": "6,379 rows, 6.89%",
            "structural_C1": a["tabfact_conformed"]["structural_C1"],
        },
        "quant_misbind_conformed": {
            "rows": a["quant_misbind_conformed"]["rows"],
            "built_for": "C2, C3, C8",
            "clears_what_it_was_built_for": "YES - its own report returns all "
                                            "eight PASS",
            "cost": "11,348 rows, 37.83% - the FEVEROUS third dropped whole",
            "structural_C1": a["quant_misbind_conformed"]["structural_C1"],
            "residual": "the predicate-blind legs stay EXACTLY EQUAL at fully "
                        "attested 0.103045; C1 test 2 is carried by the "
                        "binding-level re-derivation, not by containment",
        },
        "attr_pool_conformed": {
            "rows": a["attr_pool_conformed"]["rows"],
            "built_for": "C2, C6, C8",
            "clears_what_it_was_built_for": "YES on all three",
            "surviving_C5_fail_withdrawn_by_C_A1":
                "its only remaining FAIL was claim-to-chunk containment at 0.5978. "
                "C-A1 scopes that channel out of C5. Excluding it the worst "
                "deviation is 0.0033, claim-only is exactly 0.5000, within-pair "
                "exactly 0.5000, and the parent's 0.5758 chunk-only pool-"
                "recognition leak has fallen to within-pair 0.4872",
            "consequence": "under the amended contract the conformed attr_pool "
                           "reads PASS or NOT-APPLICABLE on all eight clauses",
            "cost": "16,966 rows, 79.25% - it delivers 4,442 rows against a "
                    "registered 20,000-30,000 band",
            "structural_C1_raw": a["attr_pool_conformed"]["structural_C1"]["raw"],
            "structural_C1_truncated":
                a["attr_pool_conformed"]["structural_C1"]["evidence_truncated_1500"],
        },
        "halueval_conformed": {
            "rows": a["halueval_conformed"]["rows"],
            "built_for": "C5",
            "clears_what_it_was_built_for":
                "NO. It clears the POOLED conjunction - claim-only 0.5395 against "
                "< 0.55, within-pair 0.5708 against < 0.60, worst barred parity "
                "deviation 0.0277 - and fails the per-half reading phase 1 also "
                "applied: summarization-half claim char length AUROC 0.2913 on 40 "
                "rows. Its own report returns C5 FAIL",
            "cost": "39,520 rows, 98.8%. 440 of the surviving 480 rows are the QA "
                    "half, so the variant is effectively QA-only",
            "structural_C1": a["halueval_conformed"]["structural_C1"],
        },
        "halueval_besteffort": {
            "rows": a["halueval_besteffort"]["rows"],
            "built_for": "C5, at a larger size",
            "clears_what_it_was_built_for":
                "NO. Claim-only 0.5562 against < 0.55 (breach 0.0062) and worst "
                "barred surface-parity deviation 0.1077. It clears only the "
                "within-pair limb",
            "cost": "36,000 rows, 90%",
            "structural_C1": a["halueval_besteffort"]["structural_C1"],
        },
    }


def findings_block():
    return [
        {"id": "F1", "severity": "corrects the coordinator's enumeration",
         "what": "frame_reject's C1 FAIL is ALSO a specification artifact of the "
                 "struck band and is withdrawn, and it appears in neither of the "
                 "coordinator's two lists",
         "evidence": "negatives 0 of 4,000 reach containment 0.90 (highest any "
                     "reaches is 0.7143) against positives 0.0565. C-A2's own "
                     "adoption text names it - 'frame_reject 0.0 < 0.0565'. The "
                     "phase-1 synthesis calls the FAIL 'arithmetically vacuous'",
         "consequence": "four C1 FAILs withdraw, not three. frame_reject remains "
                        "NON-CONFORMING on C5, which is untouched"},
        {"id": "F2", "severity": "new measurement, no phase-1 report tested it",
         "what": "C-A1's structural test fires on three source corpora",
         "evidence": "vitaminc 115 pairs / 236 rows, tabfact 45 / 90, psiloqa "
                     "5 / 10; 0 on all five lanes. Positive control fires at "
                     "8,986 / 17,972 exactly",
         "consequence": "336 rows of 760,618 (0.0442%) carry a label that no "
                        "function of (claim, evidence) can produce. vitaminc's "
                        "conforming rebuild already removes its 236 and tabfact's "
                        "already removes its 90; psiloqa's conformed variant does "
                        "NOT remove its 10"},
        {"id": "F3", "severity": "justification re-based, verdict unchanged",
         "what": "three lanes hold C-A2 test 2 only on a predicate-SENSITIVE "
                 "instrument their banked reports file as executor-added or "
                 "supplementary",
         "evidence": "path_bind - blind instrument reads negative ABOVE positive "
                     "at >= 0.90 (0.3910 vs 0.3890) and EXACTLY EQUAL fully "
                     "attested (0.1742); carried by the contiguous-run reading "
                     "0.0 vs 1.0. quant_misbind - fully attested EXACTLY EQUAL at "
                     "0.1086; carried by binding-level re-derivation 0.0 vs 1.0. "
                     "quant_scale_unit - blind fully attested 0.0 on both legs; "
                     "carried by the unit-resolved instrument 0.0321 vs 0.1011",
         "consequence": "C-A1 bullet 4 makes the predicate-sensitive reading the "
                        "MANDATED one rather than an extra, so the verdicts stand. "
                        "But path_bind is one of only two members phase 1 returned "
                        "as conforming, and its C1 now rests entirely on a probe "
                        "its own report says joins no bar"},
        {"id": "F4", "severity": "consequence of an adopted amendment",
         "what": "psiloqa's conforming rebuild spends 9,859 rows on a bar that no "
                 "longer exists",
         "evidence": "the 42-content-token claim cap was selected as the 'largest "
                     "cap clearing the band by >= 0.01 on both instruments'. That "
                     "band is C1's struck 'within 0.10' test. The unconformed "
                     "member reads 0.0292 against 0.1383 and passes the restated "
                     "test with no cut at all",
         "consequence": "only 1,379 rows (the C2 eval collision) are still "
                        "binding. A rebuild at that cost would return 50,474 -> "
                        "60,333 rows, but it would also have to add the "
                        "structural filter psiloqa_conformed still lacks"},
        {"id": "F5", "severity": "unresolved by either amendment - branch, not "
                                 "a verdict",
         "what": "C5's applicability to the five source corpora turns on the "
                 "coordinator's generated-vs-observed ruling, which requires a "
                 "per-corpus classification that is an adjudication",
         "evidence": "claim-only probe readings on record: halueval 0.9519, "
                     "ragtruth_en 0.7965, ragtruth_translated 0.7810, tabfact "
                     "0.6031, vitaminc 0.4998; psiloqa not measured. Bar is < 0.55",
         "consequence": "if the ruling captures the RAGTruth family and tabfact, "
                        "213,305 further rows (28.04% of the mix) fail C5 on a "
                        "claim-alone channel. If it captures only manufactured "
                        "negatives - halueval's ChatGPT-written leg - nothing "
                        "moves. vitaminc clears either way. psiloqa cannot be "
                        "priced at all: it is the one member of eleven with no "
                        "claim-only reading on record"},
        {"id": "F6", "severity": "consistency gain",
         "what": "C-A2's C6 scoping removes a phase-1 contradiction",
         "evidence": "ragtruth_en FAILED C6 on a within-member leave-one-out "
                     "reading of 0.6509 while ragtruth_translated PASSED on the "
                     "SAME 0.6509. Under C-A2 both are NOT-APPLICABLE - eval key "
                     "coverage 0 on each - and the 0.6509 is a diagnostic on both",
         "consequence": "ragtruth_en's remaining bar is C8 alone (no recorded "
                        "retrieval date), which is where the phase-1 synthesis's "
                        "item A2 already flags a second contradiction the "
                        "amendments do not touch: ragtruth_translated PASSES C8 on "
                        "identical archive-timestamp evidence"},
        {"id": "F7", "severity": "transcription, not measurement",
         "what": "PHASE1_SYNTHESIS.json's C5 clause line attributes claim-only "
                 "0.5985 to tabfact",
         "evidence": "0.5985 is the CONFORMED member's reading "
                     "(tabfact_conformed_clauses.json). The parent's own report "
                     "records held-out AUROC 0.6031 on table-disjoint folds",
         "consequence": "the branch in F5 is priced 0.0046 higher than the "
                        "synthesis states; both readings breach the 0.55 bar, so "
                        "no verdict turns on it"},
        {"id": "F8", "severity": "presentation-conditional, reported not "
                                 "adjudicated",
         "what": "the structural count is much larger under the 1,500-char "
                 "truncated evidence presentation",
         "evidence": "whole mix 165 -> 441 pairs / 336 -> 920 rows. attr_pool "
                     "moves 0 -> 263 pairs / 558 rows because the truth passage "
                     "it removes often sits beyond char 1,500",
         "consequence": "the flagship serves untruncated evidence windowed "
                        "1,500/750, so the model sees the whole chunk and the raw "
                        "reading is the operative one. Any read taken on the "
                        "R10-H108 truncated presentation carries the larger count"},
    ]


if __name__ == "__main__":
    main()

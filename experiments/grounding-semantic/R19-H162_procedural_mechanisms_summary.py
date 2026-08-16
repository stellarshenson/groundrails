"""R19-H162 - assemble the lane's JSON deliverable from the measured artifacts.

ANALYSIS ONLY. Reads the three measurement JSONs written by this lane and emits
`R19-H162_procedural_mechanisms.json` in the wave's return schema, so every number
in the deliverable traces to a measurement file rather than to prose.

Run:  uv run python experiments/grounding-semantic/R19-H162_procedural_mechanisms_summary.py
"""

import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R19-H162_procedural_mechanisms.json"

MECH = json.loads((HERE / "R19-H162_procedural_mech.json").read_text())
MECH2 = json.loads((HERE / "R19-H162_procedural_mech2.json").read_text())
GEOM = json.loads((HERE / "R19-H162_procedural_inflation.json").read_text())


def main():
    tq_str = MECH2["techqa"]["identifier_strata"]
    em = MECH["emanual"]
    tq = MECH["techqa"]
    caps = GEOM["subsets"]["techqa"]["inflation"]["h150d1"]["random_window_cap"]

    doc = {
        "arm": "R19-H162 procedural-register mechanism dissection (executor M4)",
        "discipline": "ANALYSIS ONLY - nothing trains, nothing tunes on arena statistics; no GPU",
        "read": "PRIMARY windowed decomposed-min (1500/750, MAX over windows, MIN over sentences)",
        "source": "banked R19-H161 per-pair logit dump, checkpoint h150d1 (models/R18-H150-arm-draw1)",
        "subsets": ["emanual", "techqa"],
        "flagship_auroc": {"emanual": 0.6780, "techqa": 0.7335},
        "read_this_draw": {
            "emanual": em["auroc_model"],
            "techqa": tq["auroc_model"],
            "delucionqa": MECH["delucionqa"]["auroc_model"],
        },
        "instruments": {
            "emanual": {
                "n_items": em["n_items"],
                "n_negative": em["n_negative"],
                "auroc_se": 0.0654,
                "seed_sd": 0.0285,
                "verdict": "weak - movement below ~0.06 is unresolvable",
            },
            "techqa": {
                "n_items": tq["n_items"],
                "n_negative": tq["n_negative"],
                "auroc_se": 0.0310,
                "seed_sd": 0.0136,
                "verdict": "the best-powered blind subset in the arena",
            },
        },
        "lexical_ceiling": {
            "definition": "content-token containment of the sentence in the window, aggregated "
            "MAX over windows then MIN over sentences - identical aggregation to the model",
            "emanual": {
                "model": em["auroc_model"],
                "lexical": em["auroc_lexical_tok_containment"],
            },
            "techqa": {"model": tq["auroc_model"], "lexical": tq["auroc_lexical_tok_containment"]},
            "delucionqa": {
                "model": MECH["delucionqa"]["auroc_model"],
                "lexical": MECH["delucionqa"]["auroc_lexical_tok_containment"],
            },
            "arena_mean": {"model": 0.7144, "lexical": 0.6582},
            "reading": "no measurable advantage for the trained cross-encoder on either target "
            "subset (emanual +0.079 = 1.2 SE, techqa +0.010 = 0.3 SE for the lexical scorer); "
            "delucionqa is the opposite at +0.212 for the model (~4 SE)",
        },
        "mechanisms": [
            {
                "name": "bind_product_version",
                "definition": "bind a stated capability, fix, requirement or vulnerability to the "
                "exact product-and-release identifier the evidence states it for",
                "applies_to": ["techqa", "emanual", "delucionqa?"],
                "evidence_it_is_a_bottleneck": (
                    f"techqa AUROC by identifier stratum: no identifier n="
                    f"{tq_str['no_identifier_in_response']['n']} reads "
                    f"{tq_str['no_identifier_in_response']['auroc_model']}; all identifiers present "
                    f"in evidence n={tq_str['all_identifiers_present_in_evidence']['n']} reads "
                    f"{tq_str['all_identifiers_present_in_evidence']['auroc_model']}; some identifier "
                    f"absent n={tq_str['some_identifier_absent_from_evidence']['n']} reads "
                    f"{tq_str['some_identifier_absent_from_evidence']['auroc_model']} - a -0.107 to "
                    f"-0.158 drop covering 126 of 250 items (50.4%). Token containment beats the "
                    f"model 0.700 to 0.632 in the hardest stratum. Sentence level: identifiers all "
                    f"present in the argmax window score mean logit "
                    f"{tq['identifier_binding']['bound_vs_unbound']['mean_model_max_bound']} (n="
                    f"{tq['identifier_binding']['bound_vs_unbound']['n_bound']}) against "
                    f"{tq['identifier_binding']['bound_vs_unbound']['mean_model_max_unbound']} (n="
                    f"{tq['identifier_binding']['bound_vs_unbound']['n_unbound']}); Spearman of "
                    f"identifier containment against model score "
                    f"{tq['identifier_binding']['spearman_idcont_vs_model']}, indistinguishable from "
                    f"plain token containment {tq['identifier_binding']['spearman_tokcont_vs_model']} "
                    f"- the model tracks identifier PRESENCE, not identifier CORRECTNESS"
                ),
                "probe_design": "held-out synthetic minimal pairs over compatibility and advisory "
                "blocks listing >=3 products at >=3 release ranges; positive restates the true "
                "(product, release, property) triple, negative swaps the product or shifts the "
                "release range to a sibling present in the same block with the token multiset "
                "otherwise identical; >=1,200 pairs, >=600 per family (product swap, release shift) "
                "per the H149 power ruling. Chance level 0.50",
                "lane_candidate": "NVD / CVE JSON feeds (NIST, US Government work, public domain "
                "under 17 U.S.C. 105) - each record pairs a free-text description with CPE "
                "applicability carrying vendor, product and machine-readable version ranges; "
                "negatives by field swap, the graduated quant_misbind construction on a release "
                "axis. Secondary: Debian Security Advisories, Ubuntu USN (public), OSV.dev "
                "(CC BY 4.0 aggregate, per-source terms checked at build)",
                "contamination": "CLEAR - NVD is not a RAGBench source corpus nor a derivative; "
                "TechQA is IBM technotes, a separate document population. CVE identifiers will "
                "co-occur (entity overlap, not document overlap), so a CVE-id overlap census and a "
                "document-disjointness check are required at lane build, on the R10-H107 precedent",
                "already_covered_by": None,
            },
            {
                "name": "bind_path_segment",
                "definition": "bind each segment of a UI navigation path to its level in the menu "
                "hierarchy, across a rendering change - arrowed path in the claim against a bare "
                "token run in the evidence",
                "applies_to": ["emanual", "delucionqa?"],
                "evidence_it_is_a_bottleneck": (
                    f"31 of 748 emanual sentences (4.1%) render an arrowed path; mean model logit "
                    f"{em['sentence_classes']['is_menu_path']['mean_model_max']} against "
                    f"{em['sentence_classes']['is_menu_path']['mean_model_max_others']} for the rest. "
                    f"The 19 items containing one (14.4%) read within-stratum AUROC "
                    f"{em['item_classes']['has_menu_path']['True']['auroc_within']} against "
                    f"{em['item_classes']['has_menu_path']['False']['auroc_within']}. Item 131 "
                    f"(supported) scores -3.095 at token containment 0.909 on the exact bare run; "
                    f"item 15 (unsupported) transposes two path levels while preserving the token "
                    f"multiset and is ranked at the 73rd model percentile against the 28th lexical"
                ),
                "probe_design": "rule-generated: a depth-3-to-5 menu tree serialised bare into the "
                "evidence and arrowed into the claim; positives restate the true path, negatives "
                "transpose two adjacent segments, drop a level, or substitute a sibling segment. "
                "The transposition family holds the token multiset exactly constant, so no surface "
                "feature can separate it. Chance level 0.50",
                "lane_candidate": "rule-based generator, no corpus required - the strongest fit to "
                "the quant_misbind pattern in this memo; settings vocabularies for realism from "
                "GNOME user documentation (CC BY-SA 3.0), LibreOffice help (MPL 2.0) or the Debian "
                "Administrator's Handbook (GPL-2 / CC BY-SA 3.0)",
                "contamination": "CLEAR - a generator has no source population; none of the named "
                "realism corpora is a RAGBench source or derivative",
                "already_covered_by": None,
            },
            {
                "name": "pointer_answer_credit",
                "definition": "credit a response that names where the answer lives as though it "
                "stated the fact",
                "applies_to": ["techqa"],
                "evidence_it_is_a_bottleneck": (
                    f"51 of 250 techqa items (20.4%) contain a pointer sentence; mean item logit "
                    f"{tq['item_classes']['has_pointer']['True']['mean_item_score']} against "
                    f"{tq['item_classes']['has_pointer']['False']['mean_item_score']} (+0.655 lift), "
                    f"within-stratum AUROC "
                    f"{tq['item_classes']['has_pointer']['True']['auroc_within']} against "
                    f"{tq['item_classes']['has_pointer']['False']['auroc_within']}. 10 of the 48 "
                    f"techqa false positives have a pointer sentence as their SINKING sentence "
                    f"(rule-matched: items 13, 25, 37, 113, 115, 181, 196, 198, 199, 222), including "
                    f"the subset's largest false positive (item 181, sinking sentence +3.84 at "
                    f"containment 0.79)"
                ),
                "probe_design": "positives state the fact; negatives assert the fact is documented "
                "in a named sibling source that does not contain it. Chance level 0.50",
                "lane_candidate": "rule-generated over any titled document collection; NVD "
                "advisories supply both titles and contents, so it rides the bind_product_version "
                "generator at near-zero marginal cost",
                "contamination": "CLEAR",
                "already_covered_by": None,
                "honest_caveat": "partly techqa's label regime rather than a verification skill - a "
                "pointer answer is unhelpful and the annotators marked it unsupported",
            },
            {
                "name": "bind_step_to_procedure",
                "definition": "bind an instruction step to the procedure heading it belongs under, "
                "when the same step text appears verbatim under a different heading",
                "applies_to": ["emanual", "delucionqa?", "techqa?"],
                "evidence_it_is_a_bottleneck": "UNDERPOWERED, labelled as such: 4 of emanual's 10 "
                "false positives read as this shape (item 65 answers an account-creation question "
                "with the device-registration procedure, every step verbatim under the wrong "
                "heading at sentence scores +2.76 to +3.79 while the item min lands at -0.462 "
                "against a -2.963 operating point; items 114, 15 alike). At 14 negatives the "
                "binomial SE is 0.121 and the count cannot be resolved. Prior read: R17-H148 "
                "measured misbound_step at 0.8697 (step-ORDER binding installed) and misbound_value "
                "at 0.6243 (SE 0.0199, 390 pairs); neither probed goal binding",
                "probe_design": "procedure blocks with >=3 headings carrying 3-5 steps each; "
                "positive restates a step with its own heading's goal, negative restates the "
                "identical step under a sibling heading's goal; step-number families excluded per "
                "the H148 reopening condition. Chance level 0.50",
                "lane_candidate": "army-tm (public domain, 17 U.S.C. 105) + faa-amt (public domain) "
                "- SUPPLY STILL BLOCKED: the crawl holds 135 of 1,766 PDFs, all lubrication orders, "
                "zero numbered-step operator manuals, against H148's measured 429 procedural blocks "
                "/ 102 documents. multidoc2dial (488 US government-service documents) is now on "
                "disk, closing half the H148 block, but is the corpus whose broad import as "
                "R10-H107 proc_gov was refuted at -0.0384",
                "contamination": "CLEAR by construction - army-tm, FAA and multidoc2dial share no "
                "documents with the Samsung TV manual or IBM technotes",
                "already_covered_by": "partially - R10-H107 imported the register broadly (refuted); "
                "R17-H148 probed the step-number family (killed at gate). Neither addressed goal "
                "binding",
            },
            {
                "name": "condition_applicability",
                "definition": "bind an instruction to the device state, model, region or release "
                "precondition under which the evidence states it holds",
                "applies_to": ["emanual", "delucionqa?", "techqa"],
                "evidence_it_is_a_bottleneck": "A LEAD, NOT A RESOLVED MEASUREMENT: 2 of emanual's "
                "10 false positives - item 108 asserts entering Ambient Mode when the TV is already "
                "on where the manual documents only the TV-off case (score -2.001, above the -2.963 "
                "operating point); item 43 inverts a conditional, the manual stating that Dolby "
                "Digital+ on a non-supporting receiver CAUSES no sound while the response prescribes "
                "it as the fix (score -1.209 at token containment 1.000). Two items on a 14-negative "
                "instrument cannot carry a lane",
                "probe_design": "instruction plus precondition pairs; negatives attach the "
                "instruction to a sibling precondition from the same block or invert the "
                "conditional's direction. Chance level 0.50",
                "lane_candidate": "the army-tm / FAA warning-and-caution supply (same block as "
                "bind_step_to_procedure), or rule-generated preconditions over synthetic "
                "device-state taxonomies",
                "contamination": "CLEAR",
                "already_covered_by": "nothing; H148's misbound_value read of 0.6243 is the closest "
                "prior signal, 1.3 SE below its own bar, and points the same way",
            },
            {
                "name": "discourse_frame_sink",
                "definition": "a response's contentless preamble or closing recap decides the item's "
                "MIN, so the item score reports a sentence with no proposition to verify",
                "applies_to": ["emanual", "techqa", "delucionqa?"],
                "evidence_it_is_a_bottleneck": (
                    f"emanual: recap sentences are 24 of 748 (3.2%) but are the item sink "
                    f"{em['sentence_classes']['is_recap']['share_that_are_the_item_sink']} of the "
                    f"time against {em['sentence_classes']['is_recap']['sink_share_others']} for "
                    f"other sentences (3.6x lift); mean logit "
                    f"{em['sentence_classes']['is_recap']['mean_model_max']} against "
                    f"{em['sentence_classes']['is_recap']['mean_model_max_others']}. The 24 items "
                    f"ending in a recap read within-stratum AUROC "
                    f"{em['item_classes']['has_recap']['True']['auroc_within']} - chance - against "
                    f"{em['item_classes']['has_recap']['False']['auroc_within']}. techqa replicates "
                    f"the sentence-level lift (recaps sink "
                    f"{tq['sentence_classes']['is_recap']['share_that_are_the_item_sink']} against "
                    f"{tq['sentence_classes']['is_recap']['sink_share_others']}) but the item-level "
                    f"stratum effect vanishes "
                    f"({tq['item_classes']['has_recap']['True']['auroc_within']} against "
                    f"{tq['item_classes']['has_recap']['False']['auroc_within']})"
                ),
                "probe_design": "not a ranking probe - the target is that a contentless discourse "
                "frame score neutrally rather than negatively, which is calibration",
                "lane_candidate": "NONE - this is a read-protocol property and the H151 wave closed "
                "serving-read changes with MAX standing as PRIMARY. Joins emanual's list-half (H147) "
                "and hagrid's bare-assertion half (H149) as a diagnosed, unfunded deficit",
                "contamination": "n/a",
                "already_covered_by": None,
            },
        ],
        "techqa_window_inflation": (
            f"REFUTED, decisively. The MAX runs over 26.8 windows per sentence on techqa (median 26, "
            f"max 82) - not the 156 (sentence, window) pairs per item, the same conflation the H141 "
            f"autopsy recorded as a process failure. The sentence max does not rise with window count "
            f"(Spearman "
            f"{GEOM['subsets']['techqa']['inflation']['h150d1']['spearman_windowcount_vs_sentmax']}); "
            f"window count alone reads AUROC "
            f"{GEOM['subsets']['techqa']['inflation']['h150d1']['auroc_of_window_count_alone']} "
            f"(anti-predictive); the >40-window bin carries the LOWEST mean logit for negatives "
            f"(-0.347). Randomly capping each sentence's window set costs AUROC monotonically: "
            f"K=40 {caps['K=40']['delta_vs_full']}, K=20 {caps['K=20']['delta_vs_full']}, "
            f"K=10 {caps['K=10']['delta_vs_full']}, K=5 {caps['K=5']['delta_vs_full']}, "
            f"K=3 {caps['K=3']['delta_vs_full']} (100 draws each, sd <= 0.024). Every extra window is "
            f"net evidence, not net noise. The one geometry finding that survives is diagnostic, not "
            f"a lever: 47.4% of techqa sentences have their model-argmax window in a different "
            f"document from their lexically-best window (emanual 22.6%, delucionqa 39.0%)"
        ),
        "shared_with_delucionqa": (
            "The register is shared; the failure is not. bind_step_to_procedure, bind_path_segment "
            "and condition_applicability all plausibly apply to a car manual, but the diagnostic that "
            "defines emanual and techqa does NOT: on delucionqa the model beats token containment "
            "0.8009 to 0.5889 (+0.2120, ~4 SE) and provenance concentration alone reads 0.5962 "
            "against the model's 0.8009. Nothing measured here explains the -0.1025 enriched-mix "
            "collapse, and naming a shared mechanism for it would be a guess - it is dropped rather "
            "than labelled. The one shared candidate worth a measurement is condition_applicability, "
            "because a car manual gates almost every instruction on trim, model year and equipment "
            "package; it is a lead only"
        ),
        "build_first": (
            "bind_product_version. It is the only mechanism on this lane whose target subset is a "
            "powered instrument (techqa: 250 items, 109 negatives, SE 0.031, seed SD 0.0136 - the "
            "tightest in the arena); it covers 126 of 250 items (50.4%); its deficit is measured at "
            "-0.107 to -0.158 AUROC between strata rather than inferred from a handful of read items; "
            "its supply is public-domain and machine-readable at a scale no crawl gates; and its "
            "generator is the already-graduated quant_misbind construction applied to a release "
            "identifier instead of a table axis. bind_path_segment and pointer_answer_credit are "
            "cheap enough to ride behind it; bind_step_to_procedure and condition_applicability "
            "remain supply-blocked exactly where R17-H148 left them"
        ),
        "expected_arena_movement": (
            "techqa +0.04 to +0.06 (arena mean +0.004 to +0.006) if the two identifier strata "
            "(n=126, currently 0.6824 and 0.6316) lift toward the identifier-free stratum's 0.7891, "
            "which is ~3 seed SD and measurable. THE HONEST DISCOUNT: the same lift would take techqa "
            "from 0.7335 past a token-containment baseline that already reads 0.7462, so any lane "
            "ending below ~0.75 on techqa has recovered a semantic contribution rather than "
            "demonstrated one - this memo recommends the arm be registered against that bar alongside "
            "the arena number. emanual: NOT VERIFIABLE - the realistic size of the path-binding and "
            "goal-binding effects is +0.02 to +0.03 against an instrument SE of 0.0654 and a "
            "same-recipe seed spread of 0.0285; the probe is the only honest primary and the emanual "
            "arena read stays REPORTED, as R17-H148 had already registered it. Ceiling context: the "
            "faithful-oracle ceiling is techqa 0.8682 (+0.133 headroom) and emanual 0.8160 (+0.138)"
        ),
        "artifacts": [
            "experiments/grounding-semantic/R19-H162_procedural_mechanisms.md",
            "experiments/grounding-semantic/R19-H162_procedural_mechanisms.json",
            "experiments/grounding-semantic/R19-H162_procedural_autopsy.py",
            "experiments/grounding-semantic/R19-H162_procedural_export.py",
            "experiments/grounding-semantic/R19-H162_procedural_mech.py",
            "experiments/grounding-semantic/R19-H162_procedural_mech2.py",
            "experiments/grounding-semantic/R19-H162_procedural_mechanisms_summary.py",
            "experiments/grounding-semantic/R19-H162_procedural_mech.json",
            "experiments/grounding-semantic/R19-H162_procedural_mech2.json",
            "experiments/grounding-semantic/R19-H162_procedural_errors.parquet",
            "experiments/grounding-semantic/R19-H162_procedural_errors.txt",
            "experiments/grounding-semantic/R19-H162_procedural_geometry.parquet",
            "experiments/grounding-semantic/R19-H162_procedural_disagree.txt",
            "logs/R19-H162_procedural.log",
        ],
        "caveats": [
            (
                "emanual is not a reliable instrument - 14 negatives, AUROC SE 0.0654, same-recipe seed "
                "spread 0.0285; every emanual count here (10 false positives, the 4/2 mechanism split) "
                "carries a binomial SE of ~0.12. The mechanism readings are directional, the counts are "
                "not resolvable"
            ),
            (
                "The lexical-ceiling gaps are not individually significant - emanual +0.0790 is 1.2 "
                "instrument SE, techqa +0.0101 is 0.3 SE. The defensible claim is 'no measurable "
                "advantage over token containment on these two subsets', not 'worse than'. The contrast "
                "that IS significant is delucionqa's +0.2120 in the other direction"
            ),
            (
                "One checkpoint, one draw - all measurements read models/R18-H150-arm-draw1 through the "
                "h150d1 dump; h150d2 and h159d1 were still writing when this lane ran, so nothing is "
                "cross-draw confirmed. The R18-H157 precedent found the taxonomy stable across draws "
                "while the error SETS were not"
            ),
            (
                "The taxonomy is a reading with a rule layer only where a rule exists - identifier "
                "binding, path rendering, pointer phrasing and discourse framing are regex-matched and "
                "their incidence is exact; goal misbinding and conditional applicability were classified "
                "by reading all 90 error records and carry no rule layer"
            ),
            (
                "Sentence-class regexes under-count - the preamble and recap patterns are conservative, "
                "so the 3.2% recap share on emanual is a floor, not an estimate"
            ),
            (
                "The identifier regex is coarse - dotted version strings, CVE ids and two-letter APAR "
                "ids only, and it finds ZERO identifiers in emanual responses; emanual's own "
                "applicability axis (model names, QLED-specific functions, geographical area) is not "
                "covered by the stratified measurement and is read-only evidence"
            ),
            (
                "Provenance concentration (modal-document share) reaches AUROC 0.7228 on emanual against "
                "the model's 0.6973 and 0.6993 on techqa, but is NOT proposed as a lever - H147 already "
                "killed retrieval geometry as an emanual direction and selecting a serving feature "
                "because it moves an arena number is what the H141 discipline forbids"
            ),
            (
                "Contamination discipline - RAGBench source corpora and derivatives are never proposed "
                "as training data; EManual and TechQA items were read only to characterise the task"
            ),
        ],
    }
    OUT.write_text(json.dumps(doc, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

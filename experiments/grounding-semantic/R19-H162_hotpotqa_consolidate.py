"""R19-H162 stage 3 - consolidate the hotpotqa dissection into the wave deliverable.

Merges the stage-1 probe measurements (`R19-H162_hotpotqa_probe.json`) and the
stage-2 family taxonomy (`R19-H162_hotpotqa_families.json`) with the mechanism
records, the readout-kill account and the ranking, into
`R19-H162_hotpotqa_mechanisms.json`. Every number in the mechanism records is
copied from those two files, never re-derived here.

Run:  uv run python experiments/grounding-semantic/R19-H162_hotpotqa_consolidate.py
"""

import json
import pathlib

HERE = pathlib.Path(__file__).parent
PROBE = HERE / "R19-H162_hotpotqa_probe.json"
FAMS = HERE / "R19-H162_hotpotqa_families.json"
OUT = HERE / "R19-H162_hotpotqa_mechanisms.json"

MECHANISMS = [
    {
        "name": "bridge_entity",
        "definition": (
            "bind the claim's two named endpoints through an intermediate entity that "
            "the claim omits and that appears in both covering documents"
        ),
        "evidence_it_is_a_bottleneck": (
            "104 of 293 claim sentences (35.49%), the largest family; max-window logit "
            "positive mean -3.680 vs negative mean -3.110, gap -0.570 with the sign "
            "INVERTED, bootstrap 95% CI [-2.567, +1.171]; sentence AUROC 0.5574 against "
            "single_hop 0.7286 and a 0.500 floor; carries 10 of the 23 negative "
            "sentences; the argmax window lands on the highest-containment document "
            "59.6% of the time against single_hop 86.9%"
        ),
        "probe_design": (
            "1,000 held-out synthetic two-relation-join items. Document A states "
            "R1(E) = M, document B states R2(M) = v, plus two distractor documents. "
            "Positive claim elides M ('the R1 of E has R2 v'). Negative takes v from a "
            "distractor entity M' present in a third document, so every surface token "
            "of the claim still appears somewhere in the bag and only the chain is "
            "broken. Entity-disjoint from any training lane. CHANCE 0.500 AUROC; the "
            "flagship's standing on the natural analogue is 0.5574"
        ),
        "lane_candidate": (
            "Rule-based generator over already-banked TabFact tables (CC-BY-4.0, ~16k "
            "Wikipedia tables): pick two tables sharing a key column, emit each row as "
            "its own document, template the claim with the join key elided. The "
            "generator rule is the graduated quant_misbind precedent applied to a JOIN "
            "instead of a cell. Every composed positive is presented as a MULTI-DOCUMENT "
            "bag under the 1500/750 windowed MIL objective"
        ),
        "contamination": (
            "CLEAR - TabFact is banked and licence-verified (CC-BY-4.0) and is not a "
            "RAGBench source corpus nor a derivative of one; the generator never reads "
            "HotpotQA. HotpotQA-derivation checks on the obvious multi-hop alternatives: "
            "HoVer is BLOCKED and already recorded in this project's survey as "
            "'HotpotQA-derived - walled'; FactCG/CG2C multi-hop training data is not "
            "released; 2WikiMultihopQA and MuSiQue are NOT HotpotQA-derived (independent "
            "construction from Wikipedia/Wikidata and from single-hop SQuAD/NQ/T-REx "
            "respectively) but share the Wikipedia-intro document population with walled "
            "HotpotQA, so either would need the R14-H136 8-gram Jaccard gate at 0.02 "
            "before use and their licences verified at source. The generator route "
            "avoids the question entirely"
        ),
        "already_covered_by": None,
        "already_covered_note": (
            "No banked corpus contains claims whose support requires two SEPARATE "
            "evidence documents. MiniCheck (MIT, 14,395 pairs) is the nearest, "
            "multi-fact and multi-sentence by construction, but its support is "
            "assembled within ONE document"
        ),
    },
    {
        "name": "conjoin_attrs",
        "definition": (
            "verify a claim that asserts one attribute about each of two entities, where "
            "each entity's attribute lives in a different document, so support requires "
            "both conjuncts"
        ),
        "evidence_it_is_a_bottleneck": (
            "56 of 293 claim sentences (19.11%); max-window logit positive mean -4.259 "
            "vs negative mean -4.638, gap +0.379 with bootstrap 95% CI [0.127, 0.640] "
            "against single_hop's +2.216, a 6-fold collapse; sentence AUROC 0.6635; the "
            "argmax window lands on the highest-containment document 48.2% of the time, "
            "the WORST of any family and barely double the 25% chance over four "
            "documents; mean minimum clause containment 0.7214 shows each conjunct is "
            "well covered, just by different documents"
        ),
        "probe_design": (
            "1,000 held-out synthetic two-entity conjunction items, each entity's "
            "attribute in its own document plus two distractors. Positives assert both "
            "conjuncts truly. Negatives flip exactly ONE conjunct and leave the other "
            "true, so a coverage meter cannot separate them. A comparative sub-leg "
            "('X has more A than Y') reuses the relational-compare construction across "
            "documents. CHANCE 0.500 AUROC"
        ),
        "lane_candidate": (
            "Same generator, different template: two rows from different banked tables "
            "emitted as separate documents, conjunction and comparative claim templates "
            "over them, negatives by single-conjunct value swap. TabFact (CC-BY-4.0) and "
            "the banked FEVEROUS slice already used by the H108 quant lane family supply "
            "the rows"
        ),
        "contamination": (
            "CLEAR - rule-generated from banked, licence-clean tables with no HotpotQA "
            "lineage at any remove; identical reasoning to bridge_entity"
        ),
        "already_covered_by": "relational compare probe leg (PARTIAL)",
        "already_covered_note": (
            "PARTIALLY. The comparative leg is the relational-compare skill the probe "
            "bank already reads at 0.51, at chance, never installed; this mechanism is "
            "that same missing skill compounded with a cross-document evidence split. "
            "The conjunction leg is covered by nothing banked"
        ),
    },
    {
        "name": "partial_support_credit",
        "definition": (
            "score a window that supplies only its share of a claim as evidence FOR that "
            "claim when the remainder is present elsewhere in the window bag, and "
            "against it when the remainder is absent or contradicted"
        ),
        "evidence_it_is_a_bottleneck": (
            "The cross-cutting mechanism behind both families. On the 209 multi-document "
            "sentences the positive-negative gap in the max-window logit is -0.0006 raw "
            "(positive mean -3.6646, negative mean -3.6640) and the length-and-anchor-"
            "adjusted label coefficient is +0.251 against single-document's +1.766, a "
            "7-fold collapse; a length-matched 70-115 char band reads a gap of -0.170; "
            "158 of 192 multi-document positives (82.3%) score below the MEAN of "
            "single-document NEGATIVES; the max logit correlates +0.5596 with best-"
            "single-document anchor containment and -0.4243 with the coverage a second "
            "document adds, so the scorer measures coverage rather than support"
        ),
        "probe_design": (
            "A paired contrast rather than a new item type: for each held-out composed "
            "item, score the full composed claim AND its two decomposed halves against "
            "the same window bag. A model with the skill scores the composed positive "
            "near its halves; a coverage meter scores it near the floor. Reported as the "
            "composed-minus-decomposed score deficit in logits. CHANCE VALUE 0.0 logits; "
            "flagship expectation strongly negative"
        ),
        "lane_candidate": (
            "Not a separate lane. It is the PRESENTATION CONSTRAINT the two generators "
            "above must satisfy: every composed positive presented as a multi-document "
            "bag under the 1500/750 windowed MIL max objective, so no single window "
            "fully supports and the max objective is forced to reward the best partial "
            "window"
        ),
        "contamination": "CLEAR - generator-only, no corpus involved",
        "already_covered_by": None,
        "already_covered_note": (
            "Null, and the supply census explains why: the R16-H142 executor census "
            "found 100% of the incumbent's 685,670 training rows had a size-1 window "
            "ensemble under 1,500-char truncation; the post-amendment-A1 mix reads mean "
            "1.507 windows per row with 20.1% multi-window, and vitaminc at 54% of the "
            "mix trains at exactly 1.00"
        ),
    },
]

READOUT_KILL = (
    "The R16-H140 learned readout did not fail to compose - it re-ranked registers, and "
    "hotpotqa is the subset with the least to gain and the most variance to absorb. "
    "Three legs. (1) The aggregation axis has the LEAST purchase on hotpotqa of any "
    "subset: recomputing the arena under mean pooling over the same banked per-window "
    "logits moves hotpotqa -0.0023, the smallest magnitude of the ten (techqa -0.1403, "
    "expertqa -0.0991, tatqa -0.0908, finqa +0.0589), and every fixed soft pooling reads "
    "0.6743 to 0.6981 against max 0.6766, all inside the subset's own 0.211-wide CI. "
    "(2) Nothing can be composed out of signal-free inputs: on exactly the multi-document "
    "sentences the readout was built to serve, the per-window logits carry a label gap of "
    "-0.0006, so any reweighting of them moves rank by noise alone. (3) The clean "
    "single-variable test of cross-window conditioning was a NULL on hotpotqa: R16-H142 "
    "G1 was an init-fingerprint-paired ablation (identical init 9d679fcb, matched "
    "permutation) whose only difference was activating a zero-init adapter conditioned on "
    "the mean-pooled window-set context; its arena mean fell -0.0323 and pubmedqa "
    "-0.1113, but hotpotqa moved +0.0028. The channel that could compose was switched on "
    "under the strictest control the campaign has run and the motivating subset was "
    "indifferent. The H140 readout was trained on a public slice (RAGTruth plus "
    "manufactured lanes, mean window-set size 2.64) whose support is single-window; its "
    "learned mapping shifted score distributions between registers, banking pubmedqa "
    "+0.0711 and emanual +0.0400 while pushing hotpotqa -0.0518 and tatqa -0.048. The "
    "H141 capacity-matched control named the same shape - window-count out-of-distribution "
    "extrapolation - for the scalar family, and R18-H156's trained-through logit "
    "aggregator agrees, reading hotpotqa -0.0466 with a serving-read swap that was "
    "neutral at -0.00008."
)

BOTTLENECK = (
    "MISSING_SKILL, with an architectural amplifier that is already closed. FOR missing "
    "skill: 71.33% of claim sentences need two or more documents (209 of 293, agreeing "
    "98.29% sentence-by-sentence with the independent R16-H140 anchor-span census at "
    "71.67%), and on exactly those the model's discrimination collapses to nothing "
    "(sentence AUROC 0.6017 vs 0.7286 single-document; adjusted label coefficient +0.251 "
    "vs +1.766); the score tracks lexical coverage (+0.5596 with best-single-document "
    "containment, -0.4243 with the gain from a second document, partial correlation with "
    "documents-needed -0.5138 after controlling for anchor count), which is a learned "
    "function, not a structural limit. AGAINST architectural: a cross-window channel "
    "already EXISTS in the shipped serving shape - pair_logits(cls, ctx) = "
    "task_head(cls) + adapter([LN(cls); LN(ctx)]) with ctx the mean-pooled CLS over the "
    "whole window set - and in the banked flagship it is inert by training, not absent by "
    "design (models/R18-H150-arm-draw1/adapter.pt carries adapter_active = False with "
    "adapter.2.weight and adapter.2.bias exactly zero). Activating it (H142 G1) moved "
    "hotpotqa +0.0028; a learned readout over embeddings (H140) moved it -0.0518; a "
    "learned aggregator over logits (H156) moved it -0.0466. Three shots, no gain. The "
    "MIL max objective already defines the correct target for a composed positive - some "
    "window must score high - so the skill is learnable inside the shipping architecture. "
    "What is absent is supply: no lane in the mix contains a positive whose support "
    "requires more than one evidence document."
)

BUILD_FIRST = (
    "bridge_entity, using a generator that yields conjoin_attrs from the same machinery. "
    "It is the largest family (35.49% of claim sentences) and the only one whose score "
    "gap has the wrong sign (-0.570, sentence AUROC 0.5574 against a 0.500 floor), and it "
    "carries 10 of the 23 scarce negative sentences. One builder serves two mechanisms: a "
    "two-table join over banked TabFact rows produces bridge items by eliding the join "
    "key and conjunction items by emitting two independent rows, so the marginal cost of "
    "the second mechanism is a template. It needs no new architecture - the lane trains "
    "inside the shipping serving shape under the existing MIL max objective, and the "
    "three architectural routes are closed by H140, H142 G1 and H156. MANDATORY PAIRED "
    "CONSTRUCTION: a lane that teaches 'credit a partially-covering window' WITHOUT "
    "composed negatives will install over-crediting, which is the dominant finqa failure "
    "the R18-H157 autopsy named; every composed positive must ship with a composed "
    "negative whose missing conjunct or broken bridge is absent from the bag."
)

EXPECTED = (
    "hotpotqa +0.02 to +0.06 if the skill installs, which is +0.002 to +0.006 on the "
    "ten-subset arena mean against the 0.71549 flagship. The upper end assumes "
    "multi-document sentences reach the single-document family's 0.7286 discrimination. "
    "CONFIDENCE LOW and the downside is real: three prior arms aimed at this subset read "
    "-0.052, +0.003 and -0.047; the subset's own bootstrap 95% CI is 0.211 wide on 17 "
    "negatives, so hotpotqa alone cannot adjudicate a lane; and a lane that installs "
    "partial-support credit without matched negatives is more likely to cost finqa and "
    "tatqa than to buy hotpotqa. Any build must be a paired-draw arm with the tabular "
    "subsets as pre-registered guardrails and an off-arena probe as the primary read."
)

CAVEATS = [
    (
        "17 negatives in 250 items (base rate 0.932). The draw-1 bootstrap 95% CI is "
        "[0.5683, 0.7796], width 0.211, so every ABSOLUTE hotpotqa number in the campaign "
        "(0.6706 flagship, 0.7039 record, 0.6514 early) sits inside the others' intervals; "
        "only paired same-item comparisons carry usable resolution. Per-family negative "
        "counts are 10/6/4/2/1 and every per-family negative statistic is correspondingly "
        "coarse - the bridge_entity gap CI [-2.567, +1.171] spans zero."
    ),
    (
        "The hop census uses lexical anchor co-occurrence, a PROXY that errs both ways. It "
        "overcounts when one window entails a claim without carrying every surface form, and "
        "undercounts paraphrased support. It bounds where cross-document information could "
        "live, not how much is needed. Mitigation: two independent constructions (H140 "
        "anchor-span, H162 document-level greedy set cover) agree 98.29% sentence-by-sentence."
    ),
    (
        "Negatives are systematically shorter than positives within the multi-document class "
        "(86.7 vs 110.4 chars), so the raw zero label gap is length-flattered. After "
        "adjusting for length and anchor count the collapse is 7-fold rather than total "
        "(+0.251 vs +1.766), and a length-matched band still reads -0.170. The direction of "
        "the finding is robust; its magnitude in raw logits is not."
    ),
    (
        "The claim-family classifier is rule-based (conjunction and comparative markers, "
        "clause-level document assignment, elided-token intersection) and was validated by "
        "reading exemplars, not by exhaustive manual adjudication of all 293 sentences. "
        "multi_doc_other (15.02%) is the unclassified remainder and carries no named "
        "mechanism."
    ),
    (
        "Arena labels are GPT-4o annotations, not human, which caps what the absolute numbers "
        "are worth; this is the standing R8-H77 caveat and it bites hardest on the 17 "
        "negatives."
    ),
    (
        "Model scores are draw 1 only (models/R18-H150-arm-draw1) because the R19-H161 dump "
        "for h150d2 and h159d1 was still running at analysis time. The draw-1 read is "
        "control-verified bit-exact against its banked value; a second draw would harden the "
        "family-level numbers but the multi-document collapse is a 209-sentence effect, not a "
        "seed effect."
    ),
    (
        "Licences for 2WikiMultihopQA and MuSiQue were NOT verified at source in this arm (no "
        "external access was in scope). They are named as alternatives with their derivation "
        "reasoning; the recommended route is the rule-based generator, which needs neither."
    ),
]


def main():
    probe = json.loads(PROBE.read_text())
    fams = json.loads(FAMS.read_text())

    out = {
        "subset": "hotpotqa",
        "arm": "R19-H162 hotpotqa mechanism dissection (executor M3) - ANALYSIS ONLY",
        "flagship_auroc": probe["flagship_2draw_auroc"],
        "draw1_auroc": probe["draw1_auroc"],
        "draw1_auroc_ci95": probe["draw1_auroc_ci95"],
        "n_items": probe["n_items"],
        "n_pos": probe["n_pos"],
        "n_neg": probe["n_neg"],
        "n_sentences": probe["n_sentences"],
        "multi_hop_share_measured": (
            f"{probe['hop_census_pct']['multi_doc']}% of claim sentences "
            f"({probe['hop_census']['multi_doc']} of {probe['n_sentences']}) need two or "
            f"more documents to cover their lexical anchors; "
            f"{probe['hop_census_pct']['single_doc']}% "
            f"({probe['hop_census']['single_doc']}) are single-document. Documents "
            f"needed: {probe['docs_needed_hist']}. Only "
            f"{probe['multi_doc_but_top1_ge_0.8']} of "
            f"{probe['hop_census']['multi_doc']} multi-document sentences (14.8%) have "
            f"one document carrying >= 80% of the anchors, so the residual cannot be "
            f"re-read as mostly single-hop restatements. Independent-method agreement "
            f"with the R16-H140 anchor-span census: "
            f"{fams['h140_cache_crosscheck']['agreement']}"
        ),
        "measurements": {
            "hop_census": probe["hop_census"],
            "hop_census_pct": probe["hop_census_pct"],
            "docs_needed_hist": probe["docs_needed_hist"],
            "saturation_by_hop": probe["saturation_by_hop"],
            "items_any_multi_doc": probe["items_any_multi_doc"],
            "items_all_single_doc": probe["items_all_single_doc"],
            "smax_correlations": probe["smax_correlations"],
            "argmax_provenance": probe["argmax_provenance"],
            "pooling_counterfactual": probe["pooling_counterfactual"],
            "pooling_contrast_all_subsets": probe["pooling_contrast_all_subsets"],
            "h140_cache_crosscheck": fams["h140_cache_crosscheck"],
            "family_census": fams["family_census"],
            "family_census_pct": fams["family_census_pct"],
            "per_family": fams["per_family"],
            "conjunction_detail": fams["conjunction_detail"],
            "bridge_detail": fams["bridge_detail"],
            "length_control": fams["length_control"],
            "smax_by_anchor_bin": fams["smax_by_anchor_bin"],
            "partial_corr_smax_vs_docs_needed_given_n_anchors": fams[
                "partial_corr_smax_vs_docs_needed_given_n_anchors"
            ],
            "adjusted_label_coefficient": {
                "multi_doc": 0.2509,
                "single_doc": 1.7662,
                "model": "smax ~ label + char_len + n_anchors, OLS within hop class",
                "length_matched_band_70_115_gap": -0.1697,
                "multi_doc_pos_below_single_doc_neg_mean": "158 of 192 (82.3%)",
            },
            "flagship_adapter_state": {
                "path": "models/R18-H150-arm-draw1/adapter.pt",
                "adapter_active": False,
                "adapter_out_weight": "all zero",
                "adapter_out_bias": "all zero",
                "consequence": (
                    "pair_logits reduces to task_head(cls); the cross-window ctx channel "
                    "exists in the serving shape but is inert in the banked flagship"
                ),
            },
        },
        "mechanisms": MECHANISMS,
        "readout_kill_explanation": READOUT_KILL,
        "bottleneck_class": BOTTLENECK,
        "build_first": BUILD_FIRST,
        "expected_arena_movement": EXPECTED,
        "ruled_out": [
            (
                "Window geometry - 0 of 3,556 hotpotqa evidence sentences cut by every "
                "window; 0 sentences in the 1,500-6,000 char anchor-span band; dispersion is "
                "purely cross-document (H140 G0 census)"
            ),
            (
                "Aggregation form - hotpotqa is the least aggregation-sensitive subset of the "
                "ten at -0.0023 under mean pooling; every fixed soft pooling sits inside the "
                "subset's CI"
            ),
            (
                "Learned cross-window conditioning - the H142 G1 init-paired ablation moved "
                "hotpotqa +0.0028"
            ),
            (
                "Sentence length as the explanation - the coverage penalty survives partial "
                "correlation on anchor count at -0.5138; the label collapse survives "
                "length-and-anchor adjustment at a 7-fold reduction"
            ),
            (
                "The single-hop restatement hypothesis - only 14.8% of multi-document "
                "sentences have one document carrying >= 80% of their anchors"
            ),
        ],
        "artifacts": [
            "experiments/grounding-semantic/R19-H162_hotpotqa_mechanisms.md",
            "experiments/grounding-semantic/R19-H162_hotpotqa_mechanisms.json",
            "experiments/grounding-semantic/R19-H162_hotpotqa_probe.py",
            "experiments/grounding-semantic/R19-H162_hotpotqa_probe.json",
            "experiments/grounding-semantic/R19-H162_hotpotqa_families.py",
            "experiments/grounding-semantic/R19-H162_hotpotqa_families.json",
            "experiments/grounding-semantic/R19-H162_hotpotqa_consolidate.py",
            "experiments/grounding-semantic/R19-H162_hotpotqa_sentences.parquet",
            "experiments/grounding-semantic/R19-H162_hotpotqa_families.parquet",
            "experiments/grounding-semantic/R19-H162_hotpotqa_eyeball.md",
            "experiments/grounding-semantic/R19-H162_hotpotqa_families_eyeball.md",
            "logs/R19-H162_hotpotqa_probe.log",
            "logs/R19-H162_hotpotqa_families.log",
        ],
        "caveats": CAVEATS,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

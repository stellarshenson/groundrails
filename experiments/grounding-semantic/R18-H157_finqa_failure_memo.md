# R18-H157 Finqa Failure Memo

Finqa is the flagship's sole arena loss to the incumbent (2-draw 0.6825 vs lettucedetect-v2 0.7170). This memo attributes the residual to named mechanisms from the per-item autopsy of the frozen 250-item finqa gate sample under both flagship draws (`models/R18-H150-arm-draw1`, `models/R18-H150-arm-draw2`), read through the PRIMARY windowed decomposed-min protocol with per-sentence sink attribution. Analysis-only arm: nothing here tunes on arena statistics; every lever named below builds from public data and validates off-arena.

- **Read fidelity** - both draws reproduced their banked windowed AUROCs bit-exact (draw1 0.6515 vs banked 0.6515, delta +0.00002; draw2 0.7135 vs 0.7135, delta -0.00002); structural fingerprint matched (250 items, 563 sentences, 2,918 pairs)
- **Artifacts** - scorer+classifier `experiments/grounding-semantic/R18-H157_finqa_autopsy.py`; counts/SEs/agreement `R18-H157_finqa_autopsy.json`; per-item per-sentence scores with argmax-window provenance `R18-H157_finqa_items.parquet` (1,126 rows = 563 sentences x 2 draws; per-draw checkpoints `R18-H157_finqa_items_draw{1,2}.parquet`); log `logs/R18-H157_autopsy.log`
- **Sample** - 250 items, 20 negatives; the negative-side instrument SE is ~0.10, so every false-positive-side count is a coarse reading; classes whose binomial SE exceeds half their share are labelled unresolvable in the JSON, not narrated

## Error split

Operating point per draw is the in-sample macro-F1-optimal threshold (the R17-H147 stated choice; nothing tuned on it - the threshold-free rank-loss view is reported alongside).

- **draw1 (AUROC 0.6515, threshold 0.166)** - 42 errors: 11 false positives (rate 0.55 of 20 negatives, binomial SE 0.111) + 31 false negatives (rate 0.135 of 230, SE 0.023)
- **draw2 (AUROC 0.7135, threshold 0.055)** - 23 errors: 15 false positives (0.75 of 20, SE 0.097) + 8 false negatives (0.035 of 230, SE 0.012)
- **The negative class fails in the FP direction on both draws** - the model over-credits plausible-but-wrong financial computations; 55-75% of unsupported items pass the operating point
- **Draw agreement is moderate** - 15 items erred by both draws, 27 only draw1, 8 only draw2 (error Jaccard 0.30; response-score Spearman 0.703); the 0.062 AUROC seed spread on finqa sits inside the windowed-regime variance the H151 wave measured, and the d1->d2 threshold shift converts false negatives into false positives rather than removing them
- **Argmin localisation is partial** - on 8 of the 26 FP item-draw records the model's sinking sentence IS the annotated unsupported sentence (85, 189, 215 on draw1; 41, 43, 85, 114, 215 on draw2); on the rest the min lands on an innocent sibling while the bad sentence also clears threshold

## Taxonomy

Rule-based pre-classification (the derivation signature: the sinking sentence carries a number absent verbatim from every window) plus manual reading of all 50 error item-draw records; 15 overrides recorded in the script's `MANUAL_OVERRIDES` with one-line reasons. Consensus = erred by both draws (draw-1 class; the two draws' rule classes agreed on 15/15 consensus items pre-override).

- **derivation_arithmetic DOMINATES every view** - consensus 7 of 15 (share 0.467, SE 0.122); draw1 24 of 42 (0.571, SE 0.076); draw2 12 of 23 (0.522, SE 0.104); threshold-free rank-loss mass 0.280 vs binding's 0.114. Both directions: supported items scored low because the claim's number is computed (ratio, difference, percentage change, sum) and appears nowhere verbatim - exemplars 101 (8.2-5.5=2.7), 147 (134-63=71), 213 ((5829-5735)/5735=1.6%), 191 (172+179+147=498) - and unsupported items scored high because a WRONG computed figure goes undetected - exemplars 114 (74.9M repurchase misused as the decrease), 189 (sign error, -305 claimed as +305), 31 (direction error, decrease vs actual +2,751 increase), 157 (sum built from wrong operands). Two compare-subtype cases (100: "291 greater than 180") sit in this family per the probe bank's relational-compare leg
- **table_binding second, small** - consensus 3 of 15 (0.200, SE 0.103 - unresolvable at this base); draw1 8, draw2 4. Two shapes: verbatim lookups under-scored with the right number sitting in the argmax window (217, 221, 155, 222), and corrupted literals over-credited (200: "$ 5 2022 billion" read as \\$5.2 billion; 242: 57,800 vs documented 57,100)
- **entity_confusion** - consensus 2 of 15 (0.067 - unresolvable): both are period errors, correct arithmetic applied to the wrong year (36: 2011 data answering a 2012 question; 229: 2009 values the documents never carry)
- **scale_unit** - consensus 1 of 15 (unresolvable): 71 (per-share \\$78.29 claimed as a total in millions), 160 (billions against a millions table), 43 (thousands-notation overstatement)
- **window_boundary ~zero** - 1 item on draw1 (202: the two operands of a difference land in different windows), 0 on draw2; consistent with finqa's shallow windowing (5.18 windows per sentence, mostly whole-chunk docs) and with the H147/R12 oracle finding that window geometry costs little
- **other_ambiguous** - consensus 2 of 15: false absence-claims ("the passage does not mention X" when it does - 85), speculative projections (116), method-recital sinking sentences (35, 132), a range claim (204); no single mechanism, no lever

## Probe cross-reference: CONFIRM

The residual reads exactly as the flagship's probe bank predicts.

- **Derivation-dominated residual vs compare-at-chance probe** - the largest class in all three views is derivation arithmetic (consensus 7 vs binding 3; rank-loss 0.280 vs 0.114), matching relational compare 0.51 (chance): the model cannot verify a number it must compute, in either direction - it under-credits true derived figures and over-credits wrong ones
- **Binding installed** - bind_col 0.948-0.960 / bind_row 0.988-0.992 at graduation, and binding-class errors are the small second class; the residual does not contradict the installation
- **Scale/unit flat** - 0-2 items per view, consistent with scale_unit 0.859-0.875 vs control 0.866 (flat): the lane gap is real but a minor finqa cost
- **Verdict recorded in the JSON as `confirm`** - a binding-dominated residual would have contradicted the probe story; it does not

## Lever mapping

Levers are named, not built; any build is its own registered arm from public data, validated off-arena. FinQA and TAT-QA are WALLED and are never proposed.

1. **FinDVer numeric lane (addresses derivation_arithmetic, the dominant class)** - the R19-staged `R19_findver_lane.parquet`: 2,400 human-annotated entailed/refuted claims over 2024 10-K/10-Q filings, 850 in the numeric subset, gate GREEN (`R19_findver_gate.json`), document-disjoint from the walled pre-2020 populations by construction. The only staged supply that directly carries refuted financial computations with explanations
2. **EDGAR-restricted synthetic derivation pairs (addresses derivation_arithmetic)** - the R14-H136-ruled EDGAR MD&A restricted slice (non-S&P-500 filers, year >= 2020, provenance gate at lane build): manufacture ratio/difference/percent-change claim pairs over real financial tables, both directions of the observed failure (true derived claims under-credited, wrong derived claims over-credited). Requires derivation pair machinery; the DR corruption engine pattern is the template
3. **Misbind-family extension to financial tables (addresses table_binding, second class)** - the graduated quant_misbind construction applied to EDGAR-restricted table snippets; the lane family already installs the skill, the residual is small, so this is mass-limited. Period-swap negatives (entity_confusion, 2 of 15 consensus) come free from the same machinery - wrong-year variants of correct rows (items 36, 229 are the exemplars)
- **scale_unit** - the A2-ruled scale_word lane from EDGAR MD&A (approved 2026-08-12, >= 3,000 pairs at the registered verify bars) already covers this class; finqa mass is 0-2 items, so no new lever is warranted beyond completing that build
- **window_boundary** - no lever: 1 item in 500 item-draw reads; the H151 wave closed serving-read changes and max stands as PRIMARY

## Instrument limits

- **20 negatives in 250** - the FP rate (0.55-0.75) carries SE ~0.10; per-class FP-side counts are coarse. The consensus derivation-vs-binding gap (7 vs 3 of 15) is directionally consistent across both draws and the threshold-free rank-loss view, but at n=15 the consensus table itself is a small-sample reading
- **Draw1-only errors (27) outnumber draw2-only (8)** - a threshold artefact of the two calibrations (0.166 vs 0.055), not two different models; the taxonomy is stable across draws anyway (derivation 57% / 52%)
- **Rule layer recall is approximate** - the derivation-candidate search uses 1% relative tolerance and misses rounded claims (item 213's true pair (5829,5735) yields 1.639% vs the claimed 1.6%); the formula-constant exclusion (10, 100, ...) hid one true derivation (199). All 50 error records were read manually and 15 overrides are recorded in the script; the taxonomy above is the post-verification one

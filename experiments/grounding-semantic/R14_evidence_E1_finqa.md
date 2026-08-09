# R14 Evidence E1 - what separates finqa success from finqa collapse

**Status**: ANALYSIS ONLY. Every number below is read from banked artifacts on disk or from the canonical log; no training, no GPU, no threshold tuned on arena statistics. Polars throughout.

**Question**: across every configuration this campaign has banked, what property - data mix, objective, read, or seed - separates the ones that succeed on `finqa` from the ones that collapse on it?

**Headline answer, stated first**:

1. `finqa` responds to exactly one data mechanism that has ever replicated: **deterministic near-miss corruption of quantities and spans in training**. It does not respond to register import, to more rows, or to prose-register near-misses.
2. `finqa` collapses under exactly two things: **non-numeric lane displacement** (H107, -0.19 to -0.25, replicated across both draws) and **objective changes that break absolute score comparability across windows** (H117 margin, -0.1020 paired).
3. Roughly half of the measurable `finqa` AUROC is not grounding at all. **Response verbosity alone reads AUROC 0.6958 on `finqa`** against the shipped model's 0.6489 - the label correlates with sentence length (faithful 234.8 chars vs unfaithful 151.1 chars), and the model already partially rides that prior. Any `finqa` "win" below about +0.05 is 1-2 of the subset's 20 negatives moving rank.

---

## 1. The complete banked `finqa` table

Two reads exist. The **windowed decomposed-min read is PRIMARY**; the truncated decomposed-min read is the legacy lineage and is recorded alongside because it is the only read that covers the R8 era.

### 1a. PRIMARY windowed read - every banked `*windowed_result.json` plus derived reads

| rank | config | finqa | arena mean | read | what the config is |
|---|---|---|---|---|---|
| 1 | `R12-H119_h108d1_strip` | **0.7452** | 0.70642 | windowed + serving wrapper | H108 lane draw 1, thousands-separator STRIP wrapper (frozen weights) |
| 2 | `R12-H119_h108d1_add` | 0.7428 | 0.70689 | windowed + serving wrapper | H108 lane draw 1, separator ADD wrapper (frozen weights) |
| 3 | `R13-H124` h108d1 consensus | 0.7422 | 0.70452 | windowed, top-2-window consensus | H108 d1 under the REFUTED consensus read |
| 4 | `R10-H108_lane_draw1` | **0.7291** | 0.70618 | windowed | quantitative-nearmiss lane, 61,184 pairs, 16 DANN groups |
| 5 | `DR_lane_draw2_control` | 0.7098 | 0.70713 | windowed | DR-2 pilot lane, 30,369 rows / 13,898 minimal pairs, BCE-only |
| 6 | `R12-H119_h108d2_strip` | 0.7074 | 0.70153 | windowed + wrapper | H108 lane draw 2, strip |
| 7 | `R10-H108_lane_draw2` | 0.7072 | 0.70373 | windowed | H108 lane draw 2 |
| 8 | `R12-H119_h108d2_add` | 0.7043 | 0.70298 | windowed + wrapper | H108 lane draw 2, add |
| 9 | `DR_lane_draw1_control` | 0.6870 | 0.69826 | windowed | DR-2 pilot lane draw 1, BCE-only |
| 10 | `R11-H118_soup_h105` | 0.6741 | 0.69218 | windowed | weight-space average of the H105 pair (KILLED as a lever) |
| 11 | `R13_anchor_teacher` | 0.6626 | 0.72066 | windowed, output-prob mean | mean of the two H105 draws' sigmoid probabilities |
| 12 | `R9-H105` draw 1 | 0.6489 | 0.70471 | windowed | **clean baseline**, 685,670 rows / 12 DANN groups |
| 13 | `R12-H119_h105d1_add` | 0.6470 | 0.70370 | windowed + wrapper | clean draw 1, add |
| 14 | `R12-H119_h105d2_strip` | 0.6354 | 0.70358 | windowed + wrapper | clean draw 2, strip |
| 15 | `R12-H119_h105d1_strip` | 0.6326 | 0.70755 | windowed + wrapper | clean draw 1, strip |
| 16 | `R12-H119_h105d2_add` | 0.6320 | 0.70175 | windowed + wrapper | clean draw 2, add |
| 17 | `R9-H105` draw 2 | 0.6176 | 0.70151 | windowed | clean baseline draw 2 |
| 18 | `DR_lane_draw1_margin` | **0.5850** | 0.70680 | windowed | DR lane + auxiliary pairwise-margin loss (H117) |
| 19 | `R10-H107_lane_draw1` | 0.4809 | 0.67043 | windowed | procedural-doc-register lane, 83,672 pairs, 14 groups |
| 20 | `R10-H107_lane_draw2` | **0.4261** | 0.65904 | windowed | procedural lane draw 2 - **below chance** |

Serving-wrapper rows (H119) and consensus rows (H124) are frozen-weight re-reads of the same four checkpoints, not independent configurations; they are listed because their spread is itself evidence (see §4, C2). The three remaining H124 consensus reads slot in at `h108d2` 0.7133, `h105d1` 0.6483, `h105d2` 0.6193 - The consensus read's own recorded `finqa` deltas are h108d1 +0.0130, h108d2 +0.0061, h105d2 +0.0017, h105d1 -0.0007 - it nudges `finqa` upward on three of four checkpoints while being REFUTED in sign on its own hagrid target.

### 1b. Truncated decomposed-min ledger - `R8_decomposed_reads.json`, 18 configs

| config | finqa | what changed |
|---|---|---|
| DR-lane-draw2-control | **0.7257** | DR-2 minimal-pair lane |
| R10-H108-lane-draw1 | 0.7248 | quantitative near-miss lane |
| R8-H99 | 0.7135 | full-corpus DANN at lambda 0.1241 (REFUTED on mean) |
| R10-H108-lane-draw2 | 0.7057 | quantitative near-miss lane |
| R8-H95 | 0.7053 | GroupDRO, forced 1/13 group balance (REFUTED on mean) |
| DR-lane-draw1-control | 0.6846 | DR-2 lane |
| R8-H90 | 0.6730 | full-corpus DANN lambda 0.02 - the contaminated-era holder |
| R9-H106 | 0.6693 | clean two-head |
| R9-H105 | 0.6639 | clean baseline draw 1 |
| R8-H100 | 0.6530 | recipe replicate |
| R8-H91 | 0.6439 | full-corpus ERM control (no discriminator) |
| R8-H96 | 0.6417 | GroupDRO→DANN curriculum |
| R9-H105-draw2 | 0.6393 | clean baseline draw 2 |
| R8-H73 | 0.6330 | capped-mix two-head |
| R8-H100-draw3 | 0.5959 | recipe replicate, draw 3 |
| DR-lane-draw1-margin | 0.5907 | margin arm |
| R10-H107-lane-draw2 | 0.5835 | procedural lane |
| R10-H107-lane-draw1 | 0.5702 | procedural lane |

### 1c. Pre-decomposition era, from the canonical log (whole-response and truncated reads)

| config | finqa | mechanism |
|---|---|---|
| R8-H62 | **0.3974** | private gold + RAGTruth only - **below chance, anti-predictive** |
| R8-H83 | 0.5038 | + HaluEval + PsiloQA (generic public diversity) |
| R8-H79 v1 | 0.5439 | DANN lambda 0.1 (feature inversion pathology) |
| R8-H81 | 0.5574 | GroupDRO eta 0.01 (q-collapse onto 2 groups) |
| R8-H88 | 0.5691 | ERM+DANN output ensemble |
| R8-H84 | 0.5797 | + VitaminC prose-register near-miss negatives |
| R8-H92 | 0.6246 | decomposed-min read introduced (ens); H84-min alone 0.6510 |
| R8-H102 score head | 0.6311 | clean-era two-head, score read |
| R8-H102 token head | **0.6913** | same checkpoint, token-span read (+0.0602 paired) |
| **R8-H78** | **0.7433** | **trained on RAGBench-train - CONTAMINATED, not admissible** |

The much-quoted trajectory **0.3974 → 0.5038 → 0.7433** is confirmed exactly (canonical log lines 1402, 1450, 1454). Its third point is the contaminated H78 run, so it is a bound on what perfect coverage buys, not an achievable target under the clean protocol.

---

## 2. Top-5 vs bottom-5: what actually separates them

Restricting to independently trained checkpoints under the PRIMARY read (dropping wrapper re-reads and derived averages):

**Top 5**: H108 d1 (0.7291), DR-control d2 (0.7098), H108 d2 (0.7072), DR-control d1 (0.6870), H105 d1 (0.6489).
**Bottom 5**: H107 d2 (0.4261), H107 d1 (0.4809), DR-margin d1 (0.5850), H105 d2 (0.6176), H105 d1 (0.6489).

### The separating property is the negative-construction method, not the register

- **All four top non-baseline configs add deterministically constructed near-miss negatives over the same evidence.** H108 supplies ~45k unit/period/scale corruption negatives built from TabFact/FEVEROUS/InfoTabS positives plus ~75k human table-cell claims. The DR-2 pilot lane supplies 13,898 minimal pairs whose judged delta types include **7,862 number-change** and 13,314 entity-swap edits (`DR_judge_summary.json`). Both lanes lift `finqa` over the paired clean control: H108 pair mean 0.71815 vs H105 pair mean 0.63325 = **+0.0849**; DR-control pair mean 0.6984 = **+0.0652**.
- **Register import without near-miss construction destroys it.** H107 added 83,672 procedural pairs (code docs + government service documents) as two new DANN groups. `finqa` fell to 0.4809 / 0.4261 - draw 2 below chance - while `delucionqa` gained +0.12, the campaign's largest single-subset gain. In-domain `gold_full` also fell 0.8514 → 0.7360 / 0.7575, so the damage is general displacement, not subset-local.
- **Prose-register near-misses are not enough.** VitaminC (370,653 rows, genuinely near-miss Wikipedia revisions) moved `finqa` only 0.5038 → 0.5797. TabFact in the full mix moved it 0.5797 → 0.6730. The 118k-row H108 lane, a fifth of VitaminC's size but constructed in the target quantity semantics, moved it further than either.

### Objective matters, and it matters in a specific direction

- **Adversarial invariance helps `finqa` at fixed data.** H90 (DANN λ0.02) vs H91 (ERM, byte-identical 762,535-pair mix) reads 0.6730 vs 0.6439 = **+0.0291 for the discriminator**, and the canonical log records this as one of the four subsets where DANN beats its ERM twin by the largest margin.
- **More invariance pressure helps `finqa` further and costs the mean.** H99 at λ0.1241 posts 0.7135 (the best clean-protocol truncated `finqa` of the R8 era) while the arena mean falls -0.0300 vs H90. H95's forced 1/13 group balance posts 0.7053 with TabFact group-val climbing 0.5242 → 0.7815, and loses -0.0095 on the mean. **Both of the top-2 R8-era `finqa` configs are REFUTED configurations.**
- **A pairwise-ranking auxiliary destroys it.** The H117 margin arm shares the DR lane, the seed, and the batch permutation with its control and reads 0.5850 vs 0.6870 = **-0.1020**, while the arena mean is unchanged (0.7068 vs 0.6983) and `emanual` gains +0.156.

### `finqa` is anti-correlated with the biomedical/prose subsets across configurations

Spearman of per-subset AUROC against `finqa` AUROC, computed across configurations:

| paired subset | 18 truncated-ledger configs | 10 distinct windowed checkpoints |
|---|---|---|
| pubmedqa | **-0.616** (p = 0.006) | **-0.794** (p = 0.006) |
| covidqa | -0.369 (p = 0.13) | **-0.661** (p = 0.038) |
| emanual | -0.137 | +0.006 |
| tatqa | -0.088 | +0.067 |
| delucionqa | +0.090 | +0.091 |
| hotpotqa | +0.102 | +0.236 |
| hagrid | +0.158 | +0.358 |
| techqa | +0.176 | +0.467 |
| arena mean | +0.228 | +0.588 |

Two consequences, both load-bearing:

- **`finqa` and `tatqa` do NOT co-move** (rho -0.088 / +0.067). The long-standing "tabular subsets" grouping is not supported by the configuration record. `tatqa` documents are short (median 340 chars, 1.02 windows/doc); `finqa` documents are long (median 2,381 chars, 2.84 windows/doc). Whatever separates finqa configs is not "can the model read a table".
- **The real trade partner is biomedical prose.** Every config that gains `finqa` tends to lose `pubmedqa`, replicated across both read families and both sample sizes at p ≈ 0.006.

---

## 3. Item-level profile of `finqa` in `R12-H121_gateA_scores.parquet`

The dump reproduces the banked clean read exactly (finqa 0.648913 vs banked 0.6489; all 10 subsets match to 4 dp), so every item statistic below describes the shipped read.

### Structure

- **2,918 sentence-x-window rows** over **563 sentences** over **250 responses**.
- **Response base rate 0.92** - 230 faithful, **20 unfaithful**. The entire AUROC rests on 20 items.
- **Sentence labels are nearly one-sided**: 485 label-1, **13 label-0**, 65 unlabelled (-1). Thirteen sentences carry the whole sentence-level discriminative signal.
- **2.252 sentences per response**, the second-shortest in the arena after covidqa and hotpotqa.
- **Sentences are long**: 197.4 chars mean per scored pair, 228.1 chars mean per response - **the longest in the arena** (next: pubmedqa 175.8, expertqa 166.4).

### Evidence length

| statistic | doc_len |
|---|---|
| mean | 2,286.6 |
| 25th pct | 631 |
| median | 2,381 |
| 75th pct | 3,618 |
| max | 5,830 |

62.7% of `finqa` window-pairs come from documents needing more than one 1,500-char window (1,830 of 2,918); mean 2.84 windows per document, 5.18 windows per sentence. Splitting the responses at the median evidence length:

- short half (n = 125, 12 negatives): AUROC **0.6903**
- long half (n = 125, 8 negatives): AUROC **0.5929**

Windowing does not rescue long financial filings - the long half still reads near chance.

### Numeric and table density - `finqa` is the arena extreme

| subset | window digit frac | numeric tokens / window | sentence digit frac | numeric tokens / sentence |
|---|---|---|---|---|
| **finqa** | **0.0702** | **16.996** | **0.0984** | **5.781** |
| tatqa | 0.0656 | 9.592 | 0.0896 | 4.036 |
| techqa | 0.0382 | 21.881 | 0.0147 | 0.920 |
| hotpotqa | 0.0258 | 3.703 | 0.0193 | 0.550 |
| hagrid | 0.0166 | 4.008 | 0.0224 | 0.853 |
| covidqa | 0.0148 | 3.639 | 0.0112 | 0.730 |
| pubmedqa | 0.0126 | 2.902 | 0.0068 | 0.500 |
| expertqa | 0.0114 | 6.274 | 0.0044 | 0.285 |
| delucionqa | 0.0025 | 1.731 | 0.0031 | 0.239 |
| emanual | 0.0021 | 0.846 | 0.0023 | 0.173 |

`finqa` claims are ~10% digits by character; `emanual` claims are 0.23%. All 20 unfaithful `finqa` responses carry a number in their score-carrying (argmin) sentence, as do 225 of 230 faithful ones - the numeric surface is present on both sides of the label, so it is a nuisance variable, not a cue.

### The numeric penalty runs against the label

`finqa` sentence score (max over windows) by numeric-token count:

| numeric tokens in sentence | n sentences | mean score | fraction from faithful responses |
|---|---|---|---|
| 0 | 36 | **0.6765** | 0.944 |
| 1-2 | 162 | 0.5657 | 0.901 |
| 3-5 | 169 | 0.5180 | 0.864 |
| 6+ | 196 | **0.5180** | 0.954 |

More numbers → lower score, while the 6+ bucket is 95.4% faithful. The model penalizes exactly the sentences most likely to be true.

Confirming that this is a mis-calibration rather than a real signal, a deterministic re-read on the same frozen scores (`score' = score + k·sentence_digit_fraction`) moves:

| k | finqa | tatqa | pubmedqa | arena mean |
|---|---|---|---|---|
| 0.0 | 0.6489 | 0.7034 | 0.6201 | 0.70471 |
| 0.5 | 0.6587 (+0.0098) | 0.7116 (+0.0082) | 0.6292 (+0.0091) | 0.7037 (-0.0009) |
| **1.0** | **0.6613 (+0.0124)** | **0.7188 (+0.0154)** | **0.6334 (+0.0133)** | **0.7042 (-0.0004)** |
| 2.0 | 0.6580 (+0.0091) | 0.7085 (+0.0051) | 0.6349 (+0.0148) | 0.7015 (-0.0032) |

The mirror probe (`score' = score − k·window_digit_fraction`) moves `finqa` the wrong way at every k (-0.0296 / -0.0463 / -0.0413), so the model's positive response to number-dense **evidence** windows is genuine signal; only the negative response to number-dense **claims** is a defect.

### The length prior - the single most uncomfortable finding

`finqa` response-level AUROC of pure surface features, no model score involved:

| subset | model | +mean sentence length | +numeric-token count | +digit fraction |
|---|---|---|---|---|
| covidqa | 0.8030 | 0.4225 | 0.3790 | 0.3916 |
| delucionqa | 0.7975 | 0.3089 | 0.6320 | 0.6342 |
| emanual | 0.6883 | 0.7113 | 0.3154 | 0.3005 |
| expertqa | 0.7857 | 0.4886 | 0.3861 | 0.4049 |
| **finqa** | **0.6489** | **0.6958** | **0.6264** | 0.5680 |
| hagrid | 0.6259 | 0.5007 | 0.5834 | 0.6383 |
| hotpotqa | 0.6809 | 0.6354 | 0.4683 | 0.4576 |
| pubmedqa | 0.6201 | 0.5126 | 0.4934 | 0.5173 |
| tatqa | 0.7034 | 0.4961 | 0.5244 | 0.5337 |
| techqa | 0.6934 | 0.5390 | 0.4846 | 0.5212 |

- Mean sentence length: **234.8 chars in faithful responses vs 151.1 in unfaithful ones**.
- Length alone reads **0.6958**, beating the shipped model's 0.6489.
- The model already partly rides it: Spearman(response score, mean sentence length) = **+0.294**; residualizing the score on log-length drops the read to **0.6187** (-0.0302).
- Adding half a z-unit of log-length on top of the model score reads **0.6983**.

This does not invalidate any banked verdict - the read is fixed and applied identically to every configuration - but it does mean **`finqa` is the arena's least trustworthy subset for attributing capability**, and a configuration that "wins finqa" may have shifted its verbosity prior rather than learned quantity semantics.

### Where the 20 negatives actually rank

| resp_idx | score | rank of 250 | sentences | max numeric tokens |
|---|---|---|---|---|
| 81 | 0.0569 | 3 | 1 | 1 |
| 116 | 0.1284 | 11 | 3 | 4 |
| 214 | 0.1408 | 15 | 3 | 5 |
| 198 | 0.1657 | 21 | 5 | 5 |
| 43 | 0.1949 | 27 | 1 | 2 |
| 189 | 0.2349 | 42 | 2 | 10 |
| 41 | 0.2372 | 45 | 1 | 4 |
| 242 | 0.2678 | 59 | 4 | 2 |
| 48 | 0.2739 | 60 | 1 | 2 |
| 168 | 0.2802 | 61 | 4 | 12 |
| 31 | 0.2900 | 64 | 2 | 4 |
| 215 | 0.3346 | 85 | 3 | 12 |
| 85 | 0.3366 | 89 | 4 | 4 |
| 157 | 0.3640 | 106 | 2 | 6 |
| 36 | 0.3815 | 118 | 3 | 10 |
| 114 | 0.4267 | 135 | 2 | 14 |
| 229 | 0.5335 | 185 | 2 | 8 |
| 67 | 0.7087 | 230 | 1 | 2 |
| 71 | 0.7134 | 232 | 1 | 3 |
| 200 | 0.7493 | 237 | 5 | 4 |

Median negative rank **62.5** (ideal ≤ 10.5, chance 125.5). Only **3 of 20** negatives fall in the bottom-20 scores; **3 of 20** rank above 200. One negative traversing the full rank range moves the subset AUROC by up to **0.049**. The margin arm's -0.1020 is therefore about two responses' worth of rank movement.

### Aggregation preference is inverted relative to the arena

Response AUROC under alternative aggregations of the same frozen scores:

| subset | min-of-max (SHIPPED) | mean-of-max | max-of-max | min-of-mean-over-windows |
|---|---|---|---|---|
| covidqa | 0.8030 | 0.8060 | 0.7341 | 0.7245 |
| delucionqa | 0.7975 | **0.8445** | 0.7156 | 0.7461 |
| emanual | 0.6883 | 0.6217 | 0.4973 | 0.6241 |
| expertqa | 0.7857 | 0.7859 | 0.6893 | 0.7410 |
| **finqa** | **0.6489** | 0.5807 | 0.5241 | **0.6730** |
| hagrid | 0.6259 | 0.5603 | 0.4512 | 0.6287 |
| hotpotqa | 0.6809 | 0.6314 | 0.6155 | 0.6847 |
| pubmedqa | 0.6201 | 0.4924 | 0.4659 | 0.6075 |
| tatqa | 0.7034 | 0.6159 | 0.5015 | 0.6362 |
| techqa | 0.6934 | **0.7293** | 0.6324 | 0.5474 |

`finqa` is one of only three subsets that prefer averaging over windows to taking the max (+0.0241), and it is the largest such gain. The max-over-windows operator - which every other high-scoring subset needs - costs `finqa`.

### The sentence splitter is the largest structural loss in the arena, and it lands on `finqa`

From `R12_label_ceiling_result.json` (ANALYSIS; bars stay ceiling-blind):

| subset | oracle (annotation only) | + H92 sentence splitter | splitter cost | + windowing (strict) | windowing cost |
|---|---|---|---|---|---|
| **finqa** | 1.0000 | **0.7500** | **-0.2500** | 0.7348 | -0.0152 |
| tatqa | 1.0000 | 0.8929 | -0.1071 | 0.8823 | -0.0106 |
| hagrid | 1.0000 | 0.9342 | -0.0658 | 0.7833 | -0.1486 |
| emanual | 1.0000 | 0.9643 | -0.0357 | 0.8160 | -0.1483 |
| hotpotqa | 1.0000 | 0.9706 | -0.0294 | 0.5843 | -0.3863 |
| expertqa | 1.0000 | 0.9815 | -0.0185 | 0.6920 | -0.2737 |
| techqa | 1.0000 | 0.9817 | -0.0183 | 0.8682 | -0.1135 |
| pubmedqa | 1.0000 | 0.9870 | -0.0130 | 0.7789 | -0.2081 |
| covidqa | 1.0000 | 1.0000 | 0.0000 | 0.7549 | -0.2451 |
| delucionqa | 1.0000 | 1.0000 | 0.0000 | 0.6657 | -0.3343 |

`finqa` loses **2.3x more ceiling to the sentence splitter than any other subset** and pooled loses only 0.0538. Financial narration is full of decimal points, `$ 383,221.` amounts and abbreviations; the splitter fragments claims so that annotated support keys go uncovered, and the min-over-sentences aggregation then scores fragments that carry no complete claim. `finqa`'s achievable ceiling under the shipped read is **0.7348**, so the H108 lane at 0.7291 sits **0.0057 below its own ceiling** - there is essentially no headroom left in the current read.

### Independent corroboration from Gate A

`R12-H121_gateA_result.json` records `finqa` as the subset whose mid-window premise is **FALSIFIED-SIGN-INVERTED**: mid-window argmax share is 15.38% on unsupported sentences vs 25.98% on supported ones, asymmetry **-10.60 pp**, against techqa's +12.99 pp. Argmax-support-free fraction on the 13 label-0 sentences is 0.7692 with distractor fraction 0.628. The subset's argmax behaviour is opposite in sign to the arena's dominant pattern.

---

## 4. Conclusions - three mechanisms for success, three for collapse

### Success mechanisms

**S1. Deterministic near-miss corruption of quantities and spans over the SAME evidence.** The only mechanism that has replicated across independent lanes and independent draws.
- Evidence: R10-H108 lane, 61,184 pairs including ~45k unit/period/scale corruption negatives - windowed `finqa` 0.7291 / 0.7072, pair mean +0.0849 over the paired clean control (0.63325); truncated 0.7248 / 0.7057, and this is the campaign's only ADMITTED lane.
- Independent replication: the DR-2 pilot lane (13,898 judged minimal pairs, 7,862 number-change deltas per `DR_judge_summary.json`) reads 0.6870 / 0.7098 windowed, pair mean +0.0652 over the same control - a different corpus, a different generation engine, the same mechanism, the same direction.
- Negative control that isolates the mechanism: VitaminC's 370,653 prose-register near-misses bought only 0.5038 → 0.5797 (H84). Six times the rows, one-fifth of the movement. **Construction in the target quantity semantics, not near-missness in general, is what pays.**

**S2. Invariance / register-rebalancing pressure that shifts capacity toward the tabular-financial group.** Real but bought at a mean cost, and every configuration that maximized it was refuted on the mean.
- Evidence: H90 vs H91 - byte-identical 762,535-pair mix, discriminator the only difference - 0.6730 vs 0.6439, **+0.0291 attributable to DANN alone**.
- Evidence: H99 at λ0.1241 reads 0.7135 (best R8-era clean-protocol `finqa`) with the arena mean falling -0.0300; H95's forced 1/13 group balance reads 0.7053 with TabFact group-val 0.5242 → 0.7815 and the mean -0.0095.
- Cross-config signature: Spearman(finqa, pubmedqa) = -0.616 (n=18, p=0.006) and -0.794 (n=10, p=0.006); Spearman(finqa, covidqa) = -0.661 (p=0.038). **`finqa` capacity is taken from biomedical prose, not from other tabular subsets** - `finqa`/`tatqa` correlate at -0.088 / +0.067, i.e. not at all.

**S3. A read that stops averaging away the numeric claim.** Read-side, deterministic, no training.
- Evidence: min-over-mean-over-windows reads `finqa` 0.6730 vs the shipped 0.6489 (+0.0241), the largest such gain in the arena; only hagrid and hotpotqa share the sign, and techqa loses -0.1460 - so this is not a shippable read change, but it identifies the operator that is costing `finqa`.
- Evidence: re-scoring the frozen dump as `score + 1.0·sentence_digit_fraction` reads `finqa` 0.6613 (+0.0124), `tatqa` +0.0154 and `pubmedqa` +0.0133 at an arena-mean cost of -0.0004. The numeric penalty on claims is the removable part; the numeric bonus on evidence windows is genuine (removing it costs `finqa` -0.0463).
- Evidence: the token-span head on the H102 checkpoint read `finqa` 0.6913 against its own score head's 0.6311, **+0.0602 paired on frozen weights** - the largest single-mechanism `finqa` movement ever measured within one checkpoint. (The line is CLOSED as a primary head; it stands here as evidence about where the signal is, not as a proposal.)

### Collapse mechanisms

**C1. Register displacement by a large non-numeric lane.** The most violent and the most replicated failure.
- Evidence: R10-H107, 83,672 procedural pairs added as two DANN groups - `finqa` 0.4809 / 0.4261, deltas -0.1921 / -0.2469, **draw 2 below chance**, while `delucionqa` gained +0.1206 / +0.1080.
- The damage is not subset-local: in-domain `gold_full` fell 0.8514 → 0.7360 / 0.7575 on data the lane never touched.
- Mechanism reading: capacity is traded between registers rather than added. This is the same axis as S2 run backwards, and it is consistent with the pubmedqa/covidqa anti-correlation - `finqa` sits at one end of a single capacity axis and the prose registers sit at the other.

**C2. Objective changes that break absolute score comparability across windows.** The read is max-over-windows then min-over-sentences; both operators compare raw scores across different (sentence, window) pairs.
- Evidence: R11-H117 margin arm vs its paired control (identical mix, identical seed, identical permutation, only the auxiliary `max(0, m − (s_clean − s_corrupt))` term added) - `finqa` 0.6870 → 0.5850, **-0.1020**, `delucionqa` -0.0392, while the arena mean was flat (0.6983 → 0.7068) and `emanual` gained +0.1561.
- Why `finqa` bears it: 5.18 windows per sentence and 2.84 windows per document, the highest window multiplicity among the numeric subsets. A pairwise-ranking auxiliary makes scores comparable within a training pair and nowhere else; the more windows a sentence must be maximized over, the more the loss of absolute calibration costs.
- Corroborating pattern: the H119 serving wrapper - a purely cosmetic, deterministic string transform on frozen weights - swings `finqa` from -0.0163 to +0.0178 across the four checkpoints, and `tatqa` from -0.0227 to +0.0448. `finqa`'s score surface is not stable under any perturbation of the numeric string form.

**C3. Statistical fragility - the subset cannot resolve the effects being attributed to it.**
- Evidence: 20 negatives in 250 responses; 13 label-0 sentences in 563. One negative traversing the full rank range moves the AUROC **0.049**. Median negative rank is 62.5/250; only 3 of 20 sit in the bottom-20, and 3 sit above rank 200.
- Consequence: the campaign-recorded H108 finqa lift (+0.0561 / +0.0342 vs the H90 baseline) corresponds to roughly one to two response rank swaps. Both draws agreeing in sign is the load-bearing evidence, not the magnitude.
- Compounding: the shipped read's `finqa` ceiling is **0.7348**, and the sentence splitter alone costs it **-0.2500** of oracle - 2.3x the next-worst subset. The H108 lane at 0.7291 already sits 0.0057 below that ceiling; further `finqa` gains under this read are arithmetically near-impossible without changing the splitter.
- Confound: response verbosity alone reads 0.6958 - **higher than the shipped model** - and residualizing the model's score on log length drops it to 0.6187. A meaningful fraction of every `finqa` number in this document is a length prior rather than grounding capability.

---

## 5. What this rules out and what it points at

Ruled out by the record above, without needing new experiments:

- **"finqa is a tabular-reading problem shared with tatqa"** - refuted. The two subsets do not co-move across 28 configuration readings (rho -0.088 / +0.067), and their document geometry differs by 7x in length.
- **"finqa needs more numeric data"** - refuted in the volume direction. VitaminC 370k rows bought +0.076; the H108 lane at 118k source rows / 61k pairs bought +0.085. Construction beats volume.
- **"finqa needs longer context"** - refuted twice: H85's coverage gate found Spearman(delta, hidden mass) = -0.128, p = 0.73, and in this dump the windowing loss on `finqa` is only -0.0152 of ceiling while the long-evidence half still reads 0.5929.

Pointed at, each tied to named evidence:

- The **sentence splitter** is the largest single lever on `finqa` that exists (-0.2500 of oracle, `R12_label_ceiling_result.json`), it is read-side, deterministic, and it costs zero GPU to measure. It is also the only lever that would raise the subset's ceiling rather than chase a 0.0057 residual.
- The **claim-side numeric penalty** is removable and measurable on frozen scores (+0.0124 finqa, +0.0154 tatqa, +0.0133 pubmedqa, -0.0004 mean). It is a training-time calibration defect, not a read defect - the evidence-side numeric response is already correct.
- Any future `finqa` claim must be **pre-registered against the length confound**, because the subset's naive verbosity baseline (0.6958) currently exceeds the shipped model (0.6489). A `finqa` bar that a length heuristic clears is not a grounding bar.

---

## Artifacts consulted

- `experiments/grounding-semantic/*_result.json` (72 files; 56 per-subset blocks extracted, 18 windowed-primary reads)
- `experiments/grounding-semantic/R8_decomposed_reads.json` (18-config truncated ledger)
- `experiments/grounding-semantic/R12-H121_gateA_scores.parquet` (77,171 rows; read reproduced to 4 dp on all 10 subsets)
- `experiments/grounding-semantic/R12-H121_gateA_result.json`, `R12_label_ceiling_result.json`, `R13_anchor_teacher_result.json`, `R13-H124_result.json`, `DR_lane_summary.json`, `DR_judge_summary.json`
- `docs/experiments/semantic-grounding-experiments.md` (canonical verdicts; R8-era arena tables at lines 1326-2000, R10-R13 verdicts at lines 2243-2600)

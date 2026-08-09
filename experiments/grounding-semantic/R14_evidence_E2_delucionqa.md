# R14 Evidence E2 - what separates success from failure on delucionqa

Forensic analysis of every banked delucionqa read in `experiments/grounding-semantic/*_result.json`, cross-referenced against `docs/experiments/semantic-grounding-experiments.md` and profiled item-by-item from `R12-H121_gateA_scores.parquet`. Analysis only - no training, no threshold or mix tuning on arena statistics.

## Headline

delucionqa does not separate successful configurations from failed ones. It separates draws.

- **The subset carries 12 negative responses out of 184** (base rate 0.9348) - the entire AUROC is the average rank of twelve items, and a single negative moving across the score range shifts AUROC by up to 0.0833
- **Bootstrap 95% CI on a single banked read is [0.6473, 0.9222]**, width 0.2749 - the widest of any subset except emanual, and wider than the *entire observed spread* across 26 banked configurations (0.7175-0.8842, width 0.1667)
- **Within-lane draw variance exceeds between-lane variance** - pooled within-lane (seed) SD 0.0476 vs between-lane 2-draw-mean SD 0.0261; the observed/expected variance ratio is **0.601**, so the estimated lane-effect variance is negative, i.e. statistically indistinguishable from zero
- **Window geometry is the only reproducible lever** - windowing lifts delucionqa on 10/10 frozen-weight pairs (mean +0.055, range +0.009 to +0.097); no training lane replicates its sign across both its own draws relative to the correct paired baseline
- **No anti-correlation with finqa** - Pearson r = -0.1137 across 26 primary configs, +0.0336 with the two H107 points removed. The "anti-correlation" is two data points from one refuted lane
- **The faithful-oracle ceiling for delucionqa under the shipped read is 0.6657** (`R12_label_ceiling_result.json`), and it is an *upper bound* on itself. Every banked delucionqa read (min 0.7175) sits above it - so every delucionqa "win" this campaign has banked is earned by partial-window firing, not by faithful entailment

## 1. Complete delucionqa score table

### 1.1 PRIMARY family - windowed decomposed-min, score head, whole model

26 comparable configurations, ranked. `mean` is the arena mean (10-subset unweighted) recomputed from each JSON's own per-subset block.

| rank | configuration | source JSON | delucionqa | finqa | arena mean |
|---|---|---|---|---|---|
| 1 | R13-H124 consensus read, H108 draw2 | `R13-H124_result.json` | **0.8842** | 0.7133 | 0.70241 |
| 2 | DR lane draw2, BCE control | `DR_lane_draw2_control_windowed_result.json` | **0.8808** | 0.7098 | 0.70713 |
| 3 | R12-H119 H108 draw2, +canonicalization | `R12-H119_h108d2_add_windowed_result.json` | **0.8624** | 0.7043 | 0.70298 |
| 4 | R10-H108 quant-nearmiss lane, draw2 | `R10-H108_lane_draw2_windowed_result.json` | **0.8614** | 0.7072 | 0.70373 |
| 5 | R12-H119 H108 draw2, -canonicalization | `R12-H119_h108d2_strip_windowed_result.json` | **0.8614** | 0.7074 | 0.70153 |
| 6 | DR lane draw1, BCE control | `DR_lane_draw1_control_windowed_result.json` | 0.8551 | 0.6870 | 0.69826 |
| 7 | R10-H107 procedural lane, draw1 | `R10-H107_lane_draw1_windowed_result.json` | 0.8469 | 0.4809 | 0.67043 |
| 8 | R12-H119 H105 draw2, -canonicalization | `R12-H119_h105d2_strip_windowed_result.json` | 0.8367 | 0.6354 | 0.70358 |
| 9 | R12-H119 H105 draw2, +canonicalization | `R12-H119_h105d2_add_windowed_result.json` | 0.8362 | 0.6320 | 0.70175 |
| 10 | R13 anchor teacher (prob-mean of H105 d1+d2) | `R13_anchor_teacher_result.json` | 0.8362 | 0.6626 | **0.72066** |
| 11 | R9-H105 clean recipe, draw2 | `R9-H105_draw2_windowed_result.json` | 0.8358 | 0.6176 | 0.70151 |
| 12 | R10-H107 procedural lane, draw2 | `R10-H107_lane_draw2_windowed_result.json` | 0.8343 | 0.4261 | 0.65904 |
| 13 | DR lane draw1, margin arm (R11-H117) | `DR_lane_draw1_margin_windowed_result.json` | 0.8159 | 0.5850 | 0.70680 |
| 14 | R13-H124 consensus read, H105 draw2 | `R13-H124_result.json` | 0.8077 | 0.6193 | 0.69499 |
| 15 | R8-H101 windowed on frozen H90 | `R8-H101_result.json` | 0.8072 | 0.6711 | 0.73553 |
| 16 | R9-H105 clean recipe, draw1 | `R9-H105_windowed_result.json` | 0.7975 | 0.6489 | 0.70471 |
| 17 | R12-H119 H105 draw1, +canonicalization | `R12-H119_h105d1_add_windowed_result.json` | 0.7975 | 0.6470 | 0.70370 |
| 18 | R12-H119 H105 draw1, -canonicalization | `R12-H119_h105d1_strip_windowed_result.json` | 0.7975 | 0.6326 | 0.70755 |
| 19 | R11-H118 weight soup (H105 d1+d2) | `R11-H118_soup_h105_windowed_result.json` | 0.7757 | 0.6741 | 0.69218 |
| 20 | R13-H124 consensus read, H105 draw1 | `R13-H124_result.json` | 0.7679 | 0.6483 | 0.69903 |
| 21 | R8-H101 replicate, H100 draw2 | `R8-H101_replicate_result.json` | 0.7578 | 0.6263 | 0.70975 |
| 22 | R12-H119 H108 draw1, -canonicalization | `R12-H119_h108d1_strip_windowed_result.json` | 0.7369 | 0.7452 | 0.70642 |
| 23 | R10-H108 quant-nearmiss lane, draw1 | `R10-H108_lane_draw1_windowed_result.json` | 0.7355 | 0.7291 | 0.70618 |
| 24 | R12-H119 H108 draw1, +canonicalization | `R12-H119_h108d1_add_windowed_result.json` | 0.7355 | 0.7428 | 0.70689 |
| 25 | R8-H101 draw3, H100 draw3 | `R8-H101_draw3_result.json` | 0.7340 | 0.5135 | 0.70655 |
| 26 | R13-H124 consensus read, H108 draw1 | `R13-H124_result.json` | 0.7175 | 0.7422 | 0.70452 |

External incumbent lettucedetect-v2 reads **0.7929** on delucionqa (constant across every JSON). 21 of the 26 primary configurations beat it.

Dispersion across the 26: SD 0.0489, range 0.1667. Compare finqa SD 0.0777 (range 0.3191) and covidqa SD 0.0166 (range 0.0520).

### 1.2 Alternate-read family - other heads, other aggregators, oracles

| configuration | source JSON | delucionqa | note |
|---|---|---|---|
| R9-PA token head, H102 weights, windowed | `R9_PA_result.json` | **0.8663** | campaign-best delucionqa of any read |
| R9-H106 score head, clean two-head, windowed | `R9-H106_fusion_result.json` | 0.8517 | |
| R8-H104 fused two-head, windowed | `R8-H104_result.json` | 0.8547 | |
| R9-PA fused, H102 weights | `R9_PA_result.json` | 0.8484 | lands *between* the heads |
| R9-H106 fused | `R9-H106_fusion_result.json` | 0.8484 | |
| R9-PC softmin tau 0.5, H90 windowed | `R9_PC_result.json` | 0.8314 | |
| R9-PC softmin tau 1 / tau 2 / tau 4 | `R9_PC_result.json` | 0.8295 / 0.8280 / 0.8251 | |
| R9-PC mean-over-sentences, H90 windowed | `R9_PC_result.json` | 0.8256 | |
| R9-PC hard-min (= primary), H90 windowed | `R9_PC_result.json` | 0.8072 | |
| R9-PA score head, H102 weights | `R9_PA_result.json` | 0.7796 | |
| R8-H104 fused two-head, truncated | `R8-H104_result.json` | 0.7573 | |
| R9-PC drop-argmin, H90 windowed | `R9_PC_result.json` | 0.7524 | |
| R9-H106 token head, clean two-head | `R9-H106_fusion_result.json` | 0.7355 | head profile inverted vs H102 |
| **Faithful oracle, shipped windowed read (O4-strict)** | `R12_label_ceiling_result.json` | **0.6657** | upper bound on itself |
| Leaky oracle (fires on partial window support, O4-lenient) | `R12_label_ceiling_result.json` | 1.0000 | |
| Annotation / splitter / chunk-cap oracles (O1-O3) | `R12_label_ceiling_result.json` | 1.0000 | delucionqa loses nothing to the splitter or the 8-document cap |

### 1.3 Truncated-era family - banked JSONs

| configuration | source JSON | delucionqa |
|---|---|---|
| R8-H92 ensemble mean-over-sentences | `R8-H92_result.json` | 0.7485 |
| R8-H92 H79 mean-over-sentences | `R8-H92_result.json` | 0.7422 |
| R8-H94 softmin tau 2.0 (RAGTruth-tuned) | `R8-H94_result.json` | 0.7301 |
| R8-H92 H84 mean-over-sentences | `R8-H92_result.json` | 0.7263 |
| H90 truncated decomposed-min (`baseline_h90`) | embedded in every `*_windowed_result.json` | 0.7263 |
| R9-PB sentence-inclusion (H90 truncated) | `R9_PB_result.json` | 0.7263 |
| R8-H88 H84 whole-response | `R8-H88_result.json` | 0.7195 |
| R8-H88 ensemble whole-response | `R8-H88_result.json` | 0.6962 |
| R9-PB sentence-exclusion | `R9_PB_result.json` | 0.6797 |
| R8-H97 | `R8-H97_result.json` | 0.6797 |
| R8-H88 H79 whole-response | `R8-H88_result.json` | 0.6628 |
| R8-H92 H79 min | `R8-H92_result.json` | 0.6618 |
| R8-H92 ensemble min | `R8-H92_result.json` | 0.6487 |
| R8-H92 H84 min | `R8-H92_result.json` | 0.6231 |

### 1.4 Truncated-era history from the canonical log

Not present in the JSONs (those files bank gold and RAGTruth only); taken from the result tables in `semantic-grounding-experiments.md`.

| checkpoint | whole-response | decomposed-min (primary of its era) | log line |
|---|---|---|---|
| R8-H62 | 0.5325 | - | 1327 |
| R8-H78 | 0.6790 | - | 1410 |
| R8-H83 | 0.7292 | - | 1449 |
| R8-H81 (GroupDRO) | 0.7200 | - | 1504 |
| R8-H84 | 0.7190 | - | 1478 |
| R8-H79 v1 (DANN) | 0.6628 | - | 1551 |
| R8-H90 (ladder holder) | 0.7306 | 0.7263 | 1774 |
| R8-H91 (ERM twin) | 0.7469 | 0.6313 | 1807 |
| R8-H96 | 0.6352 | 0.6662 | 1833 |
| R8-H95 stage 2 | 0.7083 | 0.7088 | 1856 |
| **R8-H99 (lambda 0.1241)** | **0.8367** | **0.7757** | 1889 |

R8-H99 is the only truncated-era configuration to beat lettucedetect-v2 on delucionqa outright, and it is also the truncated-era finqa best (0.7135). It refutes any account in which delucionqa and finqa must trade off.

## 2. Anchor verification

All three named anchors reproduce exactly from the JSONs.

| anchor | claim | computed | source |
|---|---|---|---|
| Windowed read on frozen weights | +0.0809 | 0.8072 - 0.7263 = **+0.0809** | `R8-H101_result.json` (`auc` vs `baseline_h90`) |
| H107 procedural lane | +0.12 | 0.8469 - 0.7263 = **+0.1206** | `R10-H107_lane_draw1_windowed_result.json` (`delucionqa_delta`) |
| H117 margin arm draw1 | -0.0392 | 0.8159 - 0.8551 = **-0.0392** | `DR_lane_draw1_margin_*` vs `DR_lane_draw1_control_*` |

**One anchor is baseline-mismatched.** The H107 (+0.1206) and H108 (+0.0092 / +0.1351) deltas recorded in the log are measured against `baseline_h90` = 0.7263, which is the *truncated-era H90* read - a different recipe scored under a different read. The correct paired comparison is the clean-recipe windowed 2-draw baseline, 0.81665 (draws 0.7975 / 0.8358, `R9-H105_windowed_result.json` and `R9-H105_draw2_windowed_result.json`). Re-based:

| lane | draw1 delta | draw2 delta | 2-draw-mean delta |
|---|---|---|---|
| R10-H107 procedural | +0.0303 | +0.0177 | +0.0240 |
| R10-H108 quant-nearmiss | -0.0812 | +0.0448 | -0.0182 |
| DR lane control | +0.0385 | +0.0642 | +0.0513 |

The headline "H107 lifted delucionqa +0.12" is +0.024 against its own control. The H108 lane, whose log entry records +0.0092/+0.1351, actually *lowers* the delucionqa 2-draw mean by 0.018. The lane that most raises delucionqa is the DR lane, which nobody registered as a delucionqa hypothesis.

## 3. Item profile - delucionqa in `R12-H121_gateA_scores.parquet`

The parquet reproduces the banked H105 draw1 read exactly (recomputed delucionqa AUROC **0.7975**, matching `R9-H105_windowed_result.json` to 4dp).

**Shape**

- 5,093 sentence-x-window rows, 184 responses, 929 sentences, 552 response-x-document pairs (exactly 3 documents per response, zero variance)
- Sentences per response: mean 5.05, median 4, max 12
- Sentence length: mean 123 chars, median 111 - shorter than finqa (197), pubmedqa (170), techqa (142); longer than hotpotqa (101)
- Document length: mean 1,496 chars, median 1,181, max 4,423; **218/552 documents (39.5%) exceed the 1,500-char window**, confirming the log's pre-run sanity figure at line 1959
- Windows per document: 334 single-window, 88 two-window, 60 three, 56 four, 14 five; mean 1.78
- delucionqa is the **second most window-exposed subset** (39.5% multi-window docs) behind techqa (83.3%) and just ahead of finqa (33.4%). covidqa and pubmedqa are 0.0% - which is why they are exact no-ops under windowing on every draw

**Labels**

- Response level: 172 grounded / **12 ungrounded**, base rate 0.9348
- Sentence level: 548 label-1, 14 label-0, 367 label -1 (unlabelled)
- Effective sample for AUROC is the 12 negatives. Each contributes 1/12 = 0.0833 of the metric

**Domain character** - automotive owner-manual QA, confirmed lexically over the window corpus: `vehicle` 4,562, `engine` 1,564, `brake` 1,056, `ignition` 576, `dealer` 360, `tire` 264, `manual` 252, `Uconnect` 240, `shift` 186, `caution` 176, `Jeep` 52, `FCA` 20. Procedural markers: 646 `warning`, 418 `note:`, 176 `caution`. Representative response sentences are conditional-procedural: *"Shift the vehicle out of REVERSE if the Camera Delay is turned on."*, *"If the call is not canceled within 10 seconds, the SOS Call system will initiate a call to an SOS operator."*

**Where the AUROC actually lives** - leave-one-negative-out jackknife on the H105 draw1 read:

| negative response score | AUROC with that item removed |
|---|---|
| 0.0263 | 0.7796 |
| 0.0601 | 0.7796 |
| 0.1261 | 0.7812 |
| 0.2829 | 0.7865 |
| 0.3207 | 0.7875 |
| 0.3274 | 0.7881 |
| 0.3751 | 0.7896 |
| 0.4203 | 0.7918 |
| 0.5308 | 0.7981 |
| 0.6866 | 0.8145 |
| 0.7911 | 0.8266 |
| **0.9152** | **0.8467** |

Deleting one item swings the metric from 0.7796 to 0.8467 - a 0.067 range from a single response. Nine of twelve negatives already rank in the bottom quartile (percentile ranks 0.005 to 0.239); the entire deficit is carried by three items at percentiles 0.413, 0.543, 0.755.

Bootstrap (4,000 resamples, seed 0) on the same read: **[0.6473, 0.9222]**, width 0.2749. Per-subset CI widths under the same procedure: emanual 0.2758, delucionqa 0.2749, finqa 0.2723, tatqa 0.2722, hotpotqa 0.2521, hagrid 0.2379, pubmedqa 0.1470, covidqa 0.1422, techqa 0.1293, expertqa 0.1223.

## 4. Is delucionqa success tied to window geometry rather than model weights?

**Yes for reproducibility; no for magnitude.** Geometry is small and always positive; weights are large and sign-random.

### 4.1 Windowing is 10/10 sign-consistent on delucionqa

Paired windowed-vs-truncated reads on identical frozen weights:

| weights | truncated | windowed | delta | finqa delta on the same pair |
|---|---|---|---|---|
| H90 (R8-H101) | 0.7263 | 0.8072 | **+0.0809** | -0.0019 |
| H100 draw2 (R8-H101 replicate) | - | 0.7578 | +0.0800 (log 1981) | -0.0267 |
| H100 draw3 (R8-H101 draw3) | - | 0.7340 | +0.0916 (log 1998) | - |
| H102 two-head fused (R8-H104) | 0.7573 | 0.8547 | **+0.0974** | -0.0419 |
| H105 draw1 (recomputed from parquet) | 0.7505 | 0.7975 | +0.0470 | -0.0150 |
| H105 draw2 (log 2196) | 0.7573 | 0.8358 | +0.0785 | -0.0217 |
| H107 lane draw1 | 0.7970 | 0.8469 | +0.0499 | -0.0893 |
| H107 lane draw2 | 0.8251 | 0.8343 | +0.0092 | -0.1574 |
| H108 lane draw1 | 0.6957 | 0.7355 | +0.0398 | +0.0043 |
| H108 lane draw2 | 0.8106 | 0.8614 | +0.0508 | +0.0015 |

**10/10 positive**, mean +0.0555. No other lever in the campaign has that record on this subset.

### 4.2 The geometric mechanism is class-asymmetric exposure

From the parquet, decomposing the +0.0470 windowing lift on H105 draw1:

| read | delucionqa AUROC |
|---|---|
| first-window-only (truncation proxy) | 0.7505 |
| full windowed (shipped) | 0.7975 |
| counterfactual: only *positives* windowed | **0.8464** |
| counterfactual: only *negatives* windowed | 0.7064 |

Windowing hands the positives +0.0959 of AUROC and hands back -0.0489 through the negatives. The net +0.0470 is what survives.

Why the asymmetry: **75.6% of grounded responses sit on at least one multi-window document, against 33.3% of ungrounded ones.** Seven of the twelve negatives are on single-window documents only and gain exactly zero. Mean windowing gain is +0.0921 for positives, +0.0463 for negatives. This is a property of how RAGBench retrieved context for delucionqa, not of the model.

Gain by exposure bucket (max windows over a response's three documents):

| max windows | n responses | n negatives | mean gain | fraction gaining |
|---|---|---|---|---|
| 1 | 50 | 8 | 0.0000 | 0.00 |
| 2 | 40 | 1 | 0.0996 | 0.35 |
| 3 | 40 | 2 | 0.1081 | 0.50 |
| 4 | 40 | 1 | 0.1627 | 0.53 |
| 5 | 14 | 0 | 0.1127 | 0.71 |

The negatives concentrate in the bucket that geometry cannot move.

### 4.3 Weights move it further but in random directions

Variance decomposition over the four two-draw lanes trained on the same clean recipe (H105 clean, H107 procedural, H108 quant-nearmiss, DR control):

| quantity | delucionqa | finqa |
|---|---|---|
| pooled within-lane (draw) SD | **0.0476** | 0.0250 |
| observed between-lane 2-draw-mean SD | 0.0261 | 0.1043 |
| expected between-mean SD under pure draw noise | 0.0337 | 0.0177 |
| observed / expected variance ratio | **0.601** | **34.96** |
| lane-effect variance estimate | -0.000453 (negative) | large, positive |

For finqa the lane signal is 35x the draw noise - lanes genuinely separate. For delucionqa the observed between-lane spread is *smaller* than draw noise alone predicts: there is no measurable lane effect at all. The single largest movement in the entire primary table is between two draws of the *same* lane with the *same* data: H108 draw1 0.7355 vs draw2 0.8614, a gap of **0.1259** - larger than any lane-to-lane difference and four times the campaign's quoted ±0.03 single-subset noise band.

### 4.4 Aggregator geometry adds a second free lever

On identical frozen H90 windowed scores (`R9_PC_result.json`): hard-min 0.8072 → mean-over-sentences 0.8256 → softmin tau 0.5 **0.8314**. Another +0.024 from formula alone, with drop-argmin costing -0.0548. The log already recorded this preference twice (line 1658: delucionqa prefers mean over min by +0.0998 truncated; line 2160: delucionqa is one of only three subsets preferring a softer read) and correctly declined to ship it, because the same softening costs the other eight subsets (R8-H94 refuted at 0.6613 vs 0.6893).

**Verdict on the question**: delucionqa's *reliable* component is entirely window geometry - a +0.055 mean shift that reproduces on every checkpoint ever tried, driven by a class-asymmetric document-length property of the subset. Its *large* component is weight-draw lottery with no lane signal. Nothing in the model's training data has been shown to move it.

## 5. Is delucionqa anti-correlated with finqa across configurations?

**No.** Across the 26 primary configurations:

- Pearson r(delucionqa, finqa) = **-0.1137**, Spearman **-0.0263**
- Removing the two R10-H107 points: Pearson **+0.0336**, Spearman **+0.0209**

The negative sign is manufactured by one refuted lane. H107 destroyed finqa (0.4809 / 0.4261, draw2 below chance) for reasons the log attributes to register displacement, and its delucionqa reads happened to be high. Two points out of 26.

Direct counterexamples:

- **H108 draw2** is 4th on delucionqa (0.8614) *and* 5th-highest finqa in the table (0.7072)
- **DR lane draw2** is 2nd on delucionqa (0.8808) with finqa 0.7098
- **H108 draw1** is 23rd on delucionqa (0.7355) with the 2nd-highest finqa (0.7291) - the same lane, opposite pattern, one seed apart
- **R8-H99** (truncated era, log 1889) holds the campaign-best truncated finqa (0.7135) *and* the campaign-best truncated delucionqa (0.7757 min / 0.8367 whole) simultaneously

Where a real anti-correlation exists is at the **head** level, not the configuration level, and the log already recorded it dying: `R9_PA_result.json` on H102 weights shows token 0.8663 vs score 0.7796 on delucionqa and token 0.6913 vs score 0.6311 on finqa - the token head wins both. `R9-H106_fusion_result.json` on a clean draw shows the profile fully inverted (token 0.7355 vs score 0.8517 on delucionqa; token 0.5702 vs score 0.6378 on finqa). The log's own conclusion at line 2223 stands: the complementarity was a property of one checkpoint's draw, not of the architecture.

**Correlation with the arena mean** is the more useful reading: r(finqa, mean) = **+0.6024** while r(delucionqa, mean) = **-0.1773** (-0.0496 excluding H107). finqa tracks overall model quality; delucionqa is close to orthogonal to it. A configuration's delucionqa score carries almost no information about whether the configuration is good.

## 6. Top-5 versus bottom-5

**Top 5** (0.8614-0.8842): R13-H124 consensus H108d2, DR draw2 control, H119 H108d2 +canon, H108 lane draw2, H119 H108d2 -canon.

**Bottom 5** (0.7175-0.7369): R13-H124 consensus H108d1, H101 draw3, H108 lane draw1, H119 H108d1 +canon, H119 H108d1 -canon.

The characterization writes itself: **four of the top five and four of the bottom five are the same lane (R10-H108, quantitative near-miss), separated only by draw.** The three H108-draw2 entries in the top five are the same checkpoint read three ways (base, canonicalization added, canonicalization stripped) and differ by 0.0010 - the read variants are no-ops. The three H108-draw1 entries in the bottom five are likewise the same checkpoint, differing by 0.0014.

Collapsing to checkpoints, the top five reduce to two checkpoints (H108 draw2, DR draw2) and the bottom five to two (H108 draw1, H100 draw3). Both surviving top checkpoints are "draw 2"; the H105 clean pair shows the same ordering (draw2 0.8358 > draw1 0.7975), as does the DR pair (0.8808 > 0.8551) and the H107 pair is the only inversion (0.8343 < 0.8469). Three of four lanes rank draw2 above draw1 - consistent with chance at p = 0.31 under a fair coin, so not evidence of anything, which is itself the point.

The one substantive read: **every configuration that touches the recipe with a training-data lane lands inside the same band as an untouched clean draw.** No lane, loss term, canonicalization wrapper, weight soup or consensus read produces a delucionqa result outside the clean recipe's own seed-to-seed range.

## 7. Three mechanisms for success

**S1 - Window geometry rescues literal procedural sentences whose support sits past char 1,500, and the rescue is class-asymmetric in the subset's favour.** delucionqa carries 218/552 documents (39.5%) longer than one window, second only to techqa; 75.6% of grounded responses touch a multi-window document against 33.3% of ungrounded ones. Windowing is 10/10 sign-positive across every frozen checkpoint ever read (mean +0.0555; `R8-H101_result.json` +0.0809, `R8-H104_result.json` +0.0974, H108 draw2 +0.0508). The counterfactual decomposition on `R12-H121_gateA_scores.parquet` prices the positive-side lift at +0.0959 against a -0.0489 negative-side give-back. Cited: parquet computation section 4.2; log line 1959 (pre-run exposure check), line 1976 (mechanism confirmed at argmin level).

**S2 - Softening the sentence aggregator, because delucionqa responses are long chains of short conditional sentences that min-aggregation over-penalises.** 5.05 sentences per response at 123 chars each - the second-shortest sentences of any subset with the second-highest sentence count. On identical frozen H90 windowed scores, hard-min 0.8072 → mean 0.8256 → softmin tau 0.5 0.8314 (`R9_PC_result.json`). The truncated-era measurement was larger still (min 0.6487 → mean 0.7485, +0.0998; log line 1658). This lever is real and free, and it is not shippable: `R8-H94_result.json` shows a globally tuned softmin costs eight of ten subsets (0.6613 vs pure min 0.6893). Cited: `R9_PC_result.json`, `R8-H94_result.json`, log lines 1658, 1706, 2160.

**S3 - Scoring with token-level span supervision instead of a CLS score, on a checkpoint whose token head happens to be strong.** `R9_PA_result.json` reads token 0.8663 against score 0.7796 on the same H102 trunk - the campaign's highest delucionqa of any read. The mechanism is that delucionqa hallucinations are localized insertions into otherwise verbatim procedural text, which token supervision localizes and a CLS score integrates away. This is listed as a success mechanism with a closure note: the effect is checkpoint-conditional (`R9-H106_fusion_result.json` shows the identical architecture with the profile inverted, token 0.7355 vs score 0.8517), and the token-head-as-primary line is CLOSED (H102/H104/P-A). Cited: `R9_PA_result.json`, `R9-H106_fusion_result.json`, log lines 2033, 2223.

## 8. Three mechanisms for failure

**F1 - Twelve negatives. The metric is a twelve-item ranking problem and its sampling noise swallows every effect anyone has measured on it.** Base rate 0.9348, 12 ungrounded of 184. Leave-one-out jackknife spans 0.7796-0.8467; bootstrap 95% CI is [0.6473, 0.9222], width 0.2749, wider than the full 26-configuration observed range of 0.1667. Pooled within-lane draw SD is 0.0476 against a between-lane 2-draw-mean SD of 0.0261 - an observed/expected variance ratio of 0.601, i.e. zero measurable lane effect, against finqa's 34.96. The H108 lane's own two draws differ by 0.1259 on identical data. Any delucionqa bar below roughly ±0.10 on a 2-draw mean is unenforceable. Cited: parquet sections 3 and 4.3; `R10-H108_lane_draw{1,2}_windowed_result.json`.

**F2 - Windowing feeds the negatives spurious maxima: the same geometry that rescues starved positives lets near-verbatim decoy windows certify fabricated content.** The three worst-ranked negatives on H105 draw1 cost the read 0.0489 of AUROC. The worst of them, response 65, goes from 0.5384 truncated to **0.9152** windowed - a +0.3767 jump - when a third window on document 2 delivers text nearly verbatim to the claim: window *"...the TrailCam image will be displayed continuously until deactivated via the touchscreen X button, the transmiss[ion]..."* against the response sentence *"The TrailCam image can be deactivated by pressing the touchscreen X button, shifting the transmission into PARK, turning the ignition OFF, or activating the windshield washing process."* The response has appended a fabricated fourth trigger to a real list; the scorer reads the lexical overlap and certifies. This is the exact near-miss discrimination H117's margin term was registered to fix, and the margin arm moved delucionqa **-0.0392** (`DR_lane_draw1_margin_windowed_result.json` 0.8159 vs paired control 0.8551). The other two costly negatives (responses 82 and 24, scores 0.7911 and 0.6866) are on single-window documents and gain nothing from windowing - they are plain model errors on hedged manual prose. Cited: parquet section 3; `DR_lane_draw1_margin_windowed_result.json`.

**F3 - The label-and-read combination penalises faithful scoring, so delucionqa "wins" are evidence of leaky entailment, not of grounding capability.** `R12_label_ceiling_result.json` puts the delucionqa faithful-oracle ceiling under the shipped windowed read at **0.6657** (O4-strict), and the file's own caveat marks both O4 numbers as *upper bounds* on their own ceilings (16.5% of supporting keys are resolved optimistically). The leniently-firing oracle - one that fires on partial window support - reads **1.0000**. Every one of the 26 banked primary configurations reads above 0.7175, i.e. above the faithful ceiling. Pooled, 20.9% of supported sentences have no single window carrying all their support and 20.0% have support spanning multiple documents; delucionqa's fixed three-document context makes it a natural home for both. The subset loses nothing to the splitter or the 8-document cap (O1-O3 all 1.0000) - the entire gap is window fragmentation. Optimizing delucionqa under this read therefore selects for a scorer that certifies on partial overlap, which is precisely the behaviour F2 shows costing us the three items that matter. Cited: `R12_label_ceiling_result.json` (per-subset delucionqa block, `ceiling_headline`, `window_reachability`, `caveats`).

## 9. Consequences for Round 14

Stated as facts, then as branches.

**Facts**

- delucionqa's measurement noise on a 2-draw mean is approximately ±0.10. The campaign's ±0.03 single-subset noise figure understates it by roughly 3x
- No training lane in the campaign has moved delucionqa outside the clean recipe's own seed range
- The only reproducible delucionqa lever, window geometry, is already shipped in the primary read
- Every banked delucionqa read exceeds its own faithful-oracle ceiling under that read

**Branches**

- **If** a Round 14 hypothesis proposes delucionqa as a target subset, **then** its bar is unenforceable at any threshold below ±0.10 on a 2-draw mean, and the hypothesis should be re-scoped or declined - verdict: DECLINE ON MEASURABILITY
- **If** a Round 14 hypothesis cites the H107 +0.12 or H108 +0.1351 delucionqa deltas as evidence for a mechanism, **then** the citation is void: both are measured against the truncated-era `baseline_h90` and shrink to +0.024 and -0.018 against the correct paired clean-recipe windowed baseline - verdict: RE-BASE BEFORE ADMITTING
- **If** a Round 14 hypothesis proposes to raise delucionqa by making the scorer fire more readily on partial window support, **then** it is optimizing against the faithful-oracle ceiling and will trade genuine grounding for benchmark AUROC - verdict: REJECT ON MECHANISM
- **If** a Round 14 hypothesis targets near-miss discrimination generally (the response-65 failure class), **then** delucionqa is a legitimate *diagnostic* surface but must not be the bar; register the bar on the arena mean and cite delucionqa as mechanism evidence only - verdict: ADMIT WITH THE BAR ELSEWHERE

## Reproduction

Every number above is recomputed from the artifacts named in each row; no figure is carried from memory. The parquet computations reproduce `R9-H105_windowed_result.json` delucionqa 0.7975 to 4dp before any derived statistic is taken. Bootstrap uses 4,000 resamples at numpy seed 0. Correlations are over the 26 primary-family rows of section 1.1. Polars throughout, no pandas.

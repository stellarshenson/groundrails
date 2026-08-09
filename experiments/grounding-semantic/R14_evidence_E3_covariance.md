# R14 Evidence E3 - subset co-movement across configurations

Forensic analysis of how the ten RAGBench arena subsets move together across every banked configuration. ANALYSIS ONLY - no training, no GPU, no threshold or mix tuned on arena statistics. Reproduction script: `R14_evidence_E3_covariance.py`; matrices banked as `R14_E3_windowed_matrix.parquet` and `R14_E3_truncated_matrix.parquet`.

## Summary

**The finqa/delucionqa problem is not one trade-off structure. It is two separate things, and only one of them is a measurable effect.**

- **finqa and delucionqa do not trade.** Across 14 windowed configurations the correlation of their deltas is r = **+0.013** (p = 0.96). Restricted to the 10 clean-era configurations, r = **-0.186** (p = 0.61), and the jackknife range spans zero, [-0.372, +0.096]. Controlling for techqa the partial correlation turns **positive**, +0.379. There is no trade-off to manage between them
- **finqa gains do not cost hotpotqa either.** r = **+0.169** (p = 0.56) windowed, **+0.015** (p = 0.95) truncated, **-0.156** level-removed. Null in every slice
- **finqa's real partner is pubmedqa, inversely.** r = **-0.837** (p = 0.0002) over all windowed configs, **-0.835** (p = 0.0027) clean-era, **-0.875** level-removed, **-0.665** (p = 0.0026) on the independent truncated matrix, **-0.829** on the 6 draw-matched lane contrasts. Jackknife range [-0.881, -0.710]. This is the single most replicated relationship in the banked evidence, and it is the only one anchored on finqa
- **finqa co-moves with techqa**, r = **+0.818** (p = 0.004) clean-era, **+0.918** (p = 0.010) on lane contrasts, jackknife floor +0.700
- **delucionqa's real partner is techqa, inversely**, r = **-0.666** windowed / **-0.882** level-removed / **-0.759** truncated - but this axis is present at r = **-0.981** inside pure seed-replicate noise, so it is a property of run-to-run training variance, not of any intervention
- **delucionqa cannot currently be measured.** Its seed-replicate sigma is **0.0432** against an analytic AUROC standard error of **0.0485** (only 12 hallucinated responses out of 184). 85% of its config-to-config variance is seed noise; **zero** of 14 windowed configs move it beyond 2 sigma. finqa is the opposite: sigma 0.0421 but 79% of its variance is structure, and 4 of 14 configs clear 2 sigma
- **Three trained checkpoints lift finqa and delucionqa simultaneously in both read types** - DR-lane draw-1 control, DR-lane draw-2 control, and the H108 quantitative lane draw 2 (the H102 token head does it too, as a read variant). None clears 2 sigma on both; six clear 1 sigma on both
- **The anchor teacher lifts both, and lifts eight of ten subsets above BOTH parent draws.** finqa +0.0293 vs the H105 pair mean and +0.0137 above the better parent - a genuine super-parent gain. delucionqa +0.0196 vs pair mean but only **+0.0004** above the better parent - that component is arithmetic, not synergy

**If the goal is a finqa remedy** → the pre-registered guardrail should be **pubmedqa**, not delucionqa or hotpotqa. Every mechanism that has moved finqa has moved pubmedqa the other way.

**If the goal is a delucionqa remedy** → no remedy is adjudicable at the current arena size. With 12 minority-class responses the subset cannot resolve the moves being attempted; any delucionqa bar is a coin flip dressed as a criterion.

## Matrix construction

Two matrices, never mixed, because the truncated and windowed reads disagree on individual subsets by up to 0.08.

- **Windowed (PRIMARY)** - 30 rows: 14 distinct trained checkpoints read through the standard windowed decomposed-min read (1500/750), plus 16 read/ensemble variants on frozen weights. Sources: `R8-H101_*`, `R8-H102_reads.json`, `R8-H104_result.json`, `R9-H105_*windowed*`, `R10-H107/H108_lane_draw*_windowed*`, `DR_lane_draw*_windowed*`, `R11-H118_soup_*`, `R12-H119_*_windowed*`, `R13-H124/H125/anchor_teacher`
- **Truncated (legacy, kept separate)** - 21 rows: the 18 configurations in `R8_decomposed_reads.json` plus the three truncated head reads from H102/H104
- **Baseline** - the clean R9-H105 pair mean per subset, read-type matched. Windowed: covidqa 0.7878, delucionqa 0.8166, emanual 0.6976, expertqa 0.7728, finqa 0.6333, hagrid 0.6340, hotpotqa 0.6667, pubmedqa 0.6063, tatqa 0.7320, techqa 0.6840
- **Correlations are computed over configurations, per subset pair.** Subtracting a per-subset baseline is a per-column constant and does not change Pearson r; the baseline sets sign and win/lose classification only
- **Two views** - RAW delta correlation carries the config's global level (a config that is worse everywhere induces spurious positive r). LEVEL-REMOVED subtracts each config's own mean delta before correlating, isolating pure trade structure. Both are reported

**Baseline-anchor caution.** The campaign record quotes H108 finqa as +0.0561 / +0.0342. Those are against the legacy `baseline_h90` field inside the read JSONs (finqa 0.6730, delucionqa 0.7263). Against the clean H105 pair-mean anchor used here, H108 finqa is **+0.0958 / +0.0739** and delucionqa **-0.0811 / +0.0448**. Same measurements, different zero point. Likewise "H102 token-span head finqa +0.0602" is the token-vs-score head swap on the same checkpoint under the windowed read (0.6913 - 0.6311), not a delta against any baseline.

## (a) Do finqa gains cost delucionqa or hotpotqa?

No. The finqa axis and the delucionqa axis are close to orthogonal, and hotpotqa is on neither.

| pair | windowed raw (n=14) | windowed level-removed | clean-era only (n=10) | truncated raw (n=18) | lane contrasts (n=6) |
|---|---|---|---|---|---|
| finqa ~ delucionqa | +0.013 (p 0.96) | -0.103 (p 0.73) | -0.186 (p 0.61) | +0.020 (p 0.94) | -0.107 (p 0.84) |
| finqa ~ hotpotqa | +0.169 (p 0.56) | -0.156 (p 0.59) | +0.161 (p 0.66) | +0.015 (p 0.95) | +0.262 (p 0.62) |
| finqa ~ pubmedqa | **-0.837 (p 0.0002)** | **-0.834 (p 0.0002)** | **-0.835 (p 0.003)** | **-0.665 (p 0.003)** | **-0.829 (p 0.04)** |
| finqa ~ techqa | +0.417 (p 0.14) | +0.185 (p 0.53) | **+0.818 (p 0.004)** | +0.348 (p 0.16) | **+0.918 (p 0.010)** |
| delucionqa ~ techqa | **-0.666 (p 0.009)** | **-0.882 (p <0.001)** | -0.464 (p 0.18) | **-0.759 (p 0.0003)** | -0.435 (p 0.39) |
| delucionqa ~ tatqa | **+0.549 (p 0.04)** | **+0.683 (p 0.007)** | +0.516 (p 0.13) | +0.452 (p 0.06) | +0.19 (p 0.72) |

Spearman on the windowed matrix agrees with Pearson on the null result: finqa~delucionqa +0.134, finqa~hotpotqa +0.134.

Where the appearance of a finqa/delucionqa trade comes from: **both load on techqa with opposite signs**. finqa~techqa is positive, delucionqa~techqa is negative, so configs that push hard on the techqa axis appear to move finqa and delucionqa in opposition. That is a shared third variable, not a direct trade - conditioning on techqa flips the finqa~delucionqa partial to +0.379.

## Trade clusters

Level-removed clean-era structure resolves into two axes that together carry 77% of the variance.

**PC1 (42.6%) - the register-displacement axis.** Loadings: delucionqa +0.47, tatqa +0.34, pubmedqa +0.27, covidqa +0.18 against techqa -0.54, finqa -0.41, emanual -0.26, expertqa -0.19. Short-context, high-grounded-rate subsets on one side; long technical support documents on the other. This is the axis that H107 (procedural lane) rode into the ground - delucionqa +0.030/+0.018 while techqa -0.127/-0.129 and finqa -0.152/-0.207.

**PC2 (34.5%) - the finqa axis.** finqa +0.773 alone against emanual -0.38, pubmedqa -0.29, expertqa -0.24. finqa is nearly its own principal component. This is why finqa interventions have been separable: they act on a direction that barely touches the register axis.

Clusters that rise together (level-removed, windowed):

- **{delucionqa, tatqa, covidqa, pubmedqa}** - delucionqa~tatqa +0.68, covidqa~pubmedqa +0.73, covidqa~delucionqa +0.40
- **{techqa, expertqa, emanual}** - expertqa~techqa +0.68, emanual~techqa +0.55, emanual~expertqa +0.34
- The two clusters trade against each other: delucionqa~techqa -0.88, delucionqa~expertqa -0.60, delucionqa~emanual -0.58, tatqa~techqa -0.74, covidqa~techqa -0.64
- **finqa** attaches weakly to the techqa cluster and strongly, inversely, to pubmedqa
- **hotpotqa is unattached.** Its largest level-removed correlation with anything is -0.43 (hagrid); against every other subset |r| < 0.30. hotpotqa moves on its own and has never been moved deliberately

The same two clusters reproduce on the independent truncated matrix: delucionqa~techqa -0.81, delucionqa~expertqa -0.72, delucionqa~emanual -0.72, expertqa~techqa +0.66, covidqa~delucionqa +0.64, finqa~pubmedqa -0.62.

## (b) Configurations that lift finqa AND delucionqa vs the clean baseline

Ten of thirty windowed rows lift both. Three are trained checkpoints.

| config | family | finqa | delucionqa | hotpotqa | mean delta |
|---|---|---|---|---|---|
| drd2_control | TRAIN | +0.0765 | +0.0642 | -0.0323 | +0.0040 |
| h108d2_quant | TRAIN | +0.0739 | +0.0448 | +0.0349 | +0.0006 |
| drd1_control | TRAIN | +0.0537 | +0.0384 | -0.0439 | -0.0049 |
| h124_consensus_h108d2 | READVAR | +0.0800 | +0.0676 | +0.0362 | -0.0007 |
| h119_strip_h108d2 | READVAR | +0.0741 | +0.0448 | +0.0351 | -0.0016 |
| h119_add_h108d2 | READVAR | +0.0710 | +0.0458 | +0.0324 | -0.0001 |
| h102_token_head | READVAR | +0.0580 | +0.0496 | -0.0522 | +0.0020 |
| h104_fused_head | READVAR | +0.0374 | +0.0381 | -0.0275 | +0.0125 |
| anchor_teacher_h105pair | READVAR | +0.0293 | +0.0196 | +0.0210 | **+0.0176** |
| h119_strip_h105d2 | READVAR | +0.0021 | +0.0201 | -0.0126 | +0.0005 |

Cross-read-type replication (a config must lift both in the windowed AND truncated matrix):

- **DR-lane draw-1 control** - windowed +0.0537/+0.0384, truncated +0.0330/+0.0237
- **DR-lane draw-2 control** - windowed +0.0765/+0.0642, truncated +0.0741/+0.0581
- **H108 quantitative lane draw 2** - windowed +0.0739/+0.0448, truncated +0.0541/+0.0567
- **H102 token head** - windowed +0.0580/+0.0496, truncated +0.0636/+0.0315

The DR control arm is the strongest joint-lift signal: both draws, both read types, same direction. Note that this is the *control* arm of the margin experiment - the DR lane data itself, without the margin loss. The H108 lane is draw-inconsistent: draw 2 lifts both, draw 1 lifts finqa +0.0958 while dropping delucionqa -0.0811.

Against the 1-sigma seed band per subset (finqa 0.0421, delucionqa 0.0432), six configs clear both: h108d2_quant, drd2_control, h102_token_head, h124_consensus_h108d2, h119_strip_h108d2, h119_add_h108d2. **Against 2 sigma, none does.**

## (c) Is the anchor teacher lifting both or riding one side?

It lifts both, and it lifts almost everything - but the two lifts have different characters.

| subset | h105 d1 | h105 d2 | pair mean | anchor | vs pair mean | vs better draw | draw spread |
|---|---|---|---|---|---|---|---|
| covidqa | 0.8030 | 0.7726 | 0.7878 | 0.7919 | +0.0041 | -0.0111 | 0.0304 |
| delucionqa | 0.7975 | 0.8358 | 0.8166 | 0.8362 | +0.0196 | **+0.0004** | 0.0383 |
| emanual | 0.6883 | 0.7070 | 0.6976 | 0.7318 | +0.0342 | **+0.0248** | 0.0187 |
| expertqa | 0.7857 | 0.7599 | 0.7728 | 0.7720 | -0.0008 | -0.0137 | 0.0258 |
| finqa | 0.6489 | 0.6176 | 0.6333 | 0.6626 | +0.0293 | **+0.0137** | 0.0313 |
| hagrid | 0.6259 | 0.6420 | 0.6340 | 0.6428 | +0.0089 | +0.0008 | 0.0161 |
| hotpotqa | 0.6809 | 0.6526 | 0.6667 | 0.6877 | +0.0210 | +0.0068 | 0.0283 |
| pubmedqa | 0.6201 | 0.5925 | 0.6063 | 0.6220 | +0.0157 | +0.0019 | 0.0276 |
| tatqa | 0.7034 | 0.7606 | 0.7320 | 0.7630 | +0.0310 | +0.0024 | 0.0572 |
| techqa | 0.6934 | 0.6745 | 0.6840 | 0.6966 | +0.0126 | +0.0032 | 0.0189 |

- **Not riding one side.** It is positive on 9 of 10 subsets vs the pair mean, and it never lands below the worse parent on any subset. Mean +0.0176 → 0.72067
- **It escapes the finqa/pubmedqa axis.** finqa +0.0293 and pubmedqa +0.0157 rise together, against a cross-config correlation of r = -0.84. Five windowed rows lift both: h105d1 (+0.0156/+0.0138), h124_consensus_h105d1 (+0.0150/+0.0138), h119_add_h105d1 (+0.0137/+0.0145), h118_soup (+0.0408/+0.0157) and the anchor. The first three are the same H105 draw-1 weights re-read, so they carry no independent information; of the two genuine ensembling objects the soup costs the arena mean (-0.0109) and the anchor gains it (+0.0176). **The anchor is the only banked object that lifts finqa and pubmedqa together while also lifting the mean.** Both ensembling objects escape the axis by not moving along it - averaging removes the seed component that generates it rather than trading against it
- **Its delucionqa credit is mostly bookkeeping.** +0.0196 vs the pair mean but +0.0004 vs the better draw, on a subset whose draw spread is 0.0383. Selecting the better draw would have bought the same delucionqa number. The genuine super-parent gains are emanual +0.0248 and finqa +0.0137
- **Mechanism read, stated weakly** - the plausible story is variance cancellation, largest where the parent draws disagree most (tatqa 0.0572 spread → +0.0310; delucionqa 0.0383 → +0.0196; finqa 0.0313 → +0.0293). Measured across the ten subsets, Pearson(draw spread, anchor gain) = **+0.364, p = 0.30** (Spearman +0.309). Directionally consistent, statistically unestablished at n = 10, and emanual is a clear counterexample - the largest super-parent gain (+0.0248) on the second-smallest spread (0.0187)

## (d) Noise vs structure

The published "+/-0.03 single-subset noise" is an average that hides a 3.5x spread. Seed sigma is estimated from five same-recipe replicate pairs (H105 clean, H107 procedural, H108 quant, DR control, H100 full-era) as sigma = sqrt(sum d^2 / 2n).

| subset | sigma_seed | analytic AUROC SE | sigma/SE | n hallucinated | SD across 14 configs | noise share of variance | structure share |
|---|---|---|---|---|---|---|---|
| covidqa | 0.0123 | 0.0334 | 0.37 | 39 | 0.0144 | 0.74 | 0.27 |
| delucionqa | **0.0432** | 0.0485 | **0.89** | **12** | 0.0469 | **0.85** | **0.15** |
| emanual | 0.0211 | 0.0654 | 0.32 | 14 | 0.0629 | 0.11 | **0.89** |
| expertqa | 0.0212 | 0.0334 | 0.63 | 108 | 0.0464 | 0.21 | 0.79 |
| finqa | **0.0421** | 0.0594 | 0.71 | 20 | **0.0912** | 0.21 | **0.79** |
| hagrid | 0.0193 | 0.0455 | 0.42 | 38 | 0.0208 | 0.86 | 0.14 |
| hotpotqa | 0.0236 | 0.0606 | 0.39 | 17 | 0.0425 | 0.31 | 0.69 |
| pubmedqa | 0.0217 | 0.0375 | 0.58 | 77 | 0.0295 | 0.54 | 0.46 |
| tatqa | **0.0414** | 0.0577 | 0.72 | 14 | 0.0362 | **1.31** | **none detectable** |
| techqa | 0.0265 | 0.0332 | 0.80 | 109 | 0.0726 | 0.13 | **0.87** |

Reading this table:

- **finqa is the one subset where interventions demonstrably work.** SD across configs 0.0912 is more than double its seed sigma; 79% of its config-to-config movement is structure. Four of fourteen trained configs move it beyond 2 sigma = 0.0842: H100 draw 3 (-0.1198), H107 draw 1 (-0.1524), H107 draw 2 (-0.2072), H108 draw 1 (+0.0958)
- **delucionqa, tatqa and hagrid carry no detectable structure.** delucionqa 85% noise, hagrid 86%, tatqa 131% (its total config spread is smaller than its own replicate spread). Zero of fourteen trained configs move delucionqa beyond 2 sigma = 0.0865; its largest trained-config delta is 0.0826 (H100 draw 3, downward)
- **techqa, emanual, expertqa are the cleanest instruments** (11-21% noise) and have never been the target of a hypothesis
- **The small subsets are the noisy ones, and the cause is the minority class.** delucionqa has 12 hallucinated responses, tatqa 14, emanual 14, hotpotqa 17, finqa 20. delucionqa's measured seed sigma is 89% of the analytic Hanley-McNeil standard error - the seed is barely adding anything on top of pure label-sampling variance

### The seed noise is itself structured

Correlating the five replicate difference vectors across subsets (n = 5, sign-arbitrary, so magnitudes only):

- **delucionqa ~ techqa r = -0.981 (p = 0.003)**
- covidqa ~ hagrid r = -0.945 (p = 0.015)
- delucionqa ~ pubmedqa r = -0.874, expertqa ~ tatqa r = -0.872
- covidqa ~ expertqa r = +0.807, emanual ~ techqa r = +0.792, pubmedqa ~ techqa r = +0.771

The strongest "trade-off" in the whole config matrix - delucionqa against techqa - appears at r = -0.98 in **pure seed replication**, where no intervention differs at all. That is the finding that reframes the register-displacement story: **the register axis is a direction the training run wanders along by itself.** Interventions do not create it; they get projected onto it. Any hypothesis whose bar reads a delucionqa/techqa contrast is reading seed placement.

By contrast finqa ~ pubmedqa is -0.62 in the seed vectors versus -0.84 across configs - present but weaker in noise, meaning the finqa axis is partly a real intervention axis rather than only a wandering direction.

### Paired-seed contrast: H117 margin arm (shared seed 1117)

The only true paired-seed comparison in the bank. Margin arm minus control arm, both draw 1, both seed 1117 - the shared component cancels.

| subset | margin | control | delta | sigma_seed | z |
|---|---|---|---|---|---|
| emanual | 0.8117 | 0.6556 | **+0.1561** | 0.0211 | **+7.40** |
| finqa | 0.5850 | 0.6870 | **-0.1020** | 0.0421 | **-2.42** |
| pubmedqa | 0.6311 | 0.5994 | +0.0317 | 0.0217 | +1.46 |
| hagrid | 0.6806 | 0.6583 | +0.0223 | 0.0193 | +1.16 |
| hotpotqa | 0.6468 | 0.6228 | +0.0240 | 0.0236 | +1.02 |
| delucionqa | 0.8159 | 0.8551 | -0.0392 | 0.0432 | **-0.91** |
| expertqa | 0.7384 | 0.7453 | -0.0069 | 0.0212 | -0.33 |
| covidqa | 0.7690 | 0.7649 | +0.0041 | 0.0123 | +0.33 |
| tatqa | 0.7107 | 0.7188 | -0.0081 | 0.0414 | -0.20 |
| techqa | 0.6788 | 0.6754 | +0.0034 | 0.0265 | +0.13 |

Two effects survive the noise floor: emanual +0.156 (z = 7.4) and finqa -0.102 (z = -2.4). The delucionqa -0.0392 recorded in the campaign context is **z = -0.91 - inside one sigma, not an effect**. The margin loss is a large, real emanual intervention and a real finqa cost; its delucionqa reading adjudicates nothing.

Note also that this paired contrast breaks the usual clusters - emanual rises +0.156 while techqa and expertqa, its two cluster partners, do not move at all. The margin loss acts on a direction not represented in the cross-config axes.

### Draw-matched lane contrasts

Six contrasts, each a lane checkpoint minus the clean checkpoint of the same draw index. This removes the level and much of the seed placement.

| contrast | covidqa | delucionqa | emanual | expertqa | finqa | hagrid | hotpotqa | pubmedqa | tatqa | techqa |
|---|---|---|---|---|---|---|---|---|---|---|
| H107 d1 - H105 d1 | -0.0339 | +0.0494 | -0.0787 | -0.0816 | -0.1680 | +0.0139 | -0.0003 | +0.0238 | +0.0690 | -0.1364 |
| H107 d2 - H105 d2 | +0.0078 | -0.0015 | -0.0787 | -0.0370 | -0.1915 | -0.0431 | -0.0262 | +0.0776 | -0.0127 | -0.1194 |
| H108 d1 - H105 d1 | -0.0514 | -0.0620 | -0.0164 | -0.0361 | +0.0802 | +0.0340 | +0.0156 | -0.0294 | +0.0357 | +0.0445 |
| H108 d2 - H105 d2 | -0.0062 | +0.0256 | -0.0938 | +0.0253 | +0.0896 | -0.0066 | +0.0490 | -0.0350 | -0.0124 | -0.0133 |
| DRctl d1 - H105 d1 | -0.0381 | +0.0576 | -0.0327 | -0.0404 | +0.0381 | +0.0324 | -0.0581 | -0.0207 | +0.0154 | -0.0180 |
| DRctl d2 - H105 d2 | -0.0207 | +0.0450 | -0.0393 | -0.0615 | +0.0922 | +0.0149 | -0.0182 | +0.0135 | +0.0569 | -0.0266 |

The H107 and H108 lanes are near mirror images on finqa (-0.17/-0.19 vs +0.08/+0.09) with the same sign on techqa as finqa in both cases, confirming the finqa~techqa +0.92 relationship on this slice. Both DR control draws lift finqa and delucionqa together - the only lane that does so on both draws.

## Caveats

- Correlations over 14 (windowed TRAIN) or 10 (clean-era) configurations have wide intervals; |r| > 0.53 is the p = 0.05 threshold at n = 14, |r| > 0.63 at n = 10. Every claim above is either replicated across independent slicings or explicitly marked non-significant
- The configurations are not an independent sample - lanes are nested (H108 draws share a lane, DR draws share a lane), and correlated configs inflate apparent structure. The level-removed and jackknife views are the guard against this
- The seed-noise covariance uses five replicate pairs and is directional evidence only; the delucionqa~techqa -0.98 should be read as "this axis exists in seed replication", not as a precise coefficient
- Analytic AUROC standard errors use the Hanley-McNeil approximation with independent samples. Two configs read the same responses, so the SE of their *difference* is smaller than the tabulated `SE_of_diff_indep` column - the analytic SE is an upper reference for how well a single subset can be measured at all, not a per-comparison bar
- The full-era configurations (H90/H100/H102) were trained on a mix including private gold and appear only in the pooled correlation views; the clean-era slice excludes them
- Nothing here proposes a bar, a threshold, or a mix. Any hypothesis derived from this evidence must justify its mechanism independently and pre-register blind

## Answers, condensed

**(a)** No. finqa~delucionqa r = +0.013 windowed / -0.186 clean-era, both non-significant, jackknife spanning zero, partial correlation given techqa turning positive at +0.379. finqa~hotpotqa r = +0.169 / +0.015, null. The two subsets fail independently. finqa's systematic cost partner is **pubmedqa at r = -0.84**, replicated in five independent slicings.

**(b)** Yes - three trained checkpoints and seven read variants. Replicating across both read types: **DR-lane draw-1 control**, **DR-lane draw-2 control**, **H108 quantitative lane draw 2**, and the **H102 token head**. Six clear 1 sigma on both subsets; none clears 2 sigma on both.

**(c)** It lifts both, and is positive on nine of ten subsets versus the H105 pair mean. Its finqa gain is genuine super-parent synergy (+0.0137 above the better draw); its delucionqa gain is arithmetic (+0.0004 above the better draw, i.e. equal to picking the good draw). It does not ride one side - it rides the variance, which is why it lifts finqa and pubmedqa together against their r = -0.84 axis, the only banked object to do so while also lifting the arena mean.

**(d)** Subset-specific, and the +/-0.03 rule of thumb misleads on exactly the two subsets under discussion. Seed sigma runs 0.0123 (covidqa) to 0.0432 (delucionqa). Noise accounts for **85% of delucionqa's** config-to-config variance, 131% of tatqa's, 86% of hagrid's - versus **21% of finqa's**, 13% of techqa's, 11% of emanual's. On the one true paired-seed contrast (H117 margin vs control, seed 1117) only emanual (z = +7.4) and finqa (z = -2.4) clear the floor; the delucionqa -0.0392 is z = -0.91. And the seed noise is not isotropic - the delucionqa/techqa trade axis appears at r = -0.98 inside pure seed replication, meaning the register-displacement structure is something training runs generate on their own.

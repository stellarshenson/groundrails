# R20 sweep brief A - re-adjudication of the banked record under the adopted variance protocol

Subagent research brief (read-only, 2026-08-16). Input to the Round 20 adjudication; the coordinator adjudicates. The estimator defect found here is verified and banked in `experiments/grounding-semantic/R20_variance_repair.py` / `.json` - where this brief and that artifact disagree, the artifact wins.

## (a) Recipe table (windowed blind arena reads, grouped by recipe; pooled sd 0.01189 as adopted, SE_diff vs flagship k=4 mean 0.71583)

| arm | k | mean | z vs flagship | classification |
|---|---|---|---|---|
| R18-H150/H160 flagship | 4 | 0.71583 | - | reference |
| R18-H155 twin (shared-init pair) | 2 | 0.72613 | +1.00 | UNRESOLVED (above) |
| R18-H152 | 2 | 0.71409 | -0.17 | UNRESOLVED |
| R16-H142 twin pair | 2 | 0.71286 | -0.29 | UNRESOLVED |
| R18-H158 soups (cross_init / same_init / cross_arm) | 1 each | 0.72306 / 0.71767 / 0.71022 | +0.54 / +0.14 / -0.42 | UNRESOLVED |
| R19-H160 soups (k3 / k4 / soupB) | 1 each | 0.72150 / 0.71066 / 0.69922 | +0.43 / -0.39 / -1.25 | UNRESOLVED |
| R10-H108 lane (prior flagship) | 2 | 0.70495 | -1.06 | UNRESOLVED |
| R12-H119 cells (4) | 2 each | 0.70273-0.70556 | -1.00..-1.27 | UNRESOLVED |
| R9-H105 (clean baseline) | 2 | 0.70311 | -1.24 | UNRESOLVED |
| DR-lane control | 2 | 0.70270 | -1.28 | UNRESOLVED |
| R17-H146 / R13-H129 / R17-H145 / R16-H142-G1-arm / R11-H118 / R12-H122 / R18-H156 / R19-H159 | 1 each | 0.69847 / 0.69709 / 0.69590 / 0.69268 / 0.69218 / 0.69147 / 0.69053 / 0.68941 | -1.31..-1.99 | UNRESOLVED |
| DR-lane margin | 2 | 0.69186 | -2.33 | RESOLVED-BELOW |
| R14-H133 | 2 | 0.69424 | -2.10 | RESOLVED-BELOW |
| R13-H128 / R13-H127 / R14-H135 | 1 each | 0.68320 / 0.68206 / 0.68152 | -2.45..-2.58 | RESOLVED-BELOW |
| R10-H107 | 2 | 0.66473 | -4.96 | RESOLVED-BELOW |
| R19-H168 (EuroBERT) | 1 | 0.54498 | -12.85 | RESOLVED-BELOW |

Headline: only 7 of 29 non-flagship cells resolve; three arms sit above the flagship (H155 +0.0103, H158 cross_init +0.0072, H160 soup-k3 +0.0057) and none is resolvable as read.

## (b) Empirical vs pooled sd

Flagship 4-draw sample sd 0.00618 vs pooled 0.01189 (ratio 0.52); chi-square 3s²/σ² = 0.81 on 3 df, lower-tail p = 0.153 - consistent, no flag. Against the corrected pooled estimate p = 0.265. Partial out-of-sample only: two flagship pair gaps sit inside the pool.

## (c) Formally unresolved past verdicts (margin below the detection floor at their k)

R18-H150 promotion (+0.01053 vs H108 pair), R10-H108 admission (+0.00186), R16-H142-T non-promotion (mean clause), R18-H152 banking (-0.0017), R18-H155 standing (+0.01031 ABOVE), R11-H118 kill (-0.0109), R12-H119 (4 cells), R12-H122 (-0.0116), R13-H129 (-0.0060), DR corruption lane (-0.0023), R17-H145 (-0.0091), R17-H146 (-0.0065, moot - lane carried anyway), R18-H156 (-0.0253 vs floor 0.0266), R18-H158 soups, R19-H159 (-0.0264 vs floor 0.0266), R19-H160 soups, R19-H165 concat kill (-0.0116/-0.0160).

## (d) Surviving kills (resolve at |z| >= 2 below the flagship)

R19-H168 (-0.1709, also chance in-domain), R10-H107 (-0.0511), R14-H135 (-0.0343), R13-H127 (-0.0338), R13-H128 (-0.0326), DR-lane margin (-0.0240), R14-H133 (-0.0216, survives as "below current flagship" not on its original margin). Gate-kills that never spent an arena read and pre-windowed-era verdicts are untouched.

## (e) Protocol defects found

1. **Estimator bias**: the adopted pooling RMS-pools per-pair half-normal sds, squaring in the pi/2 bias - pooled sd inflated by sqrt(pi/2) = 1.2533. Unbiased pooled estimator sigma^2 = sum(gap^2)/(2n). VERIFIED and banked in `R20_variance_repair.py`.
2. **Census omission**: the R16-H142 twin pair (0.72498/0.70073, gap 0.02425, seeds 1142/2142, same recipe) missing from the pool with no registered exclusion. VERIFIED.
3. **Homogeneity assumed, never tested**: per-pair sigma-hat spans 13x; Cochran's C = 0.495 (crit ~0.602) - not rejected, but 1-df estimates have little power.
4. **Winner's curse at the reference**: the flagship is the max over promotion-adjudicated arms across 48 uncorrected reads; every z is against a selection-biased reference. (Resolved in the amendment by the pre-registered headline rule and by the coordinator's H155 reclassification - the shared-init pair is not an independent 2-draw arm.)
5. **Split-executor pooling licensed**: `R19-H160_exec_equivalence.json` PASS establishes exchangeability of draws (loss diffs within 10x CUDNN backend noise floor, no sign bias, weight drift at noise). Caveats: proven at reduced geometry (8/16 vs 48/96), 55 of 94,265 steps, grad-norm sign consistency 0.69 vs 0.8 bar.
6. **Freeze wart**: arms frozen under the biased sd need one dated re-freeze amendment (done in AMENDMENT V1).

## (f) Recommendations (as returned; coordinator dispositions in the log)

1. Repair the pooled estimator before k=6 freezes anything (0 GPU-h) - ADOPTED as AMENDMENT V1.
2. Buy H155 draws 3-4 (~13 GPU-h) to resolve the only k>=2 arm above the flagship - DECLINED AS FRAMED: the H155 pair shares an init; "H155" is not a distinct recipe, and extending it because its arena reads are high is arena selection (H141 discipline). What stands: init variance is large (~95% of pooled variance) and init selection is a real lever iff a non-arena selection surface is exhibited.
3. Close the historical ledger by reclassification, not re-runs; pre-register the k=6 decision rule now - ADOPTED.

Honest k=6 headline (pending draws 5/6): "~0.716 +/- 0.009 (2 SE at k=6)". Beats the incumbent 0.67963 at z ~ 7.5 - resolved, publishable. The 0.74 target is ~5 SE above this recipe - only a new lever reaches it.

# R20 sweep brief D - weight averaging (soups, EMA, SWA) under the variance protocol

Subagent research brief (read-only, 2026-08-16). Input to the Round 20 adjudication; the coordinator adjudicates. Cell JSONs under `experiments/grounding-semantic/`.

## (a) Complete soup evidence table

Six banked soup reads. Delta = soup - mean of its own ingredients. Executors: d1/d2 monolithic, d3/d4 split-cotangent.

| Soup | Ingredients | Ingr. mean | Soup arena | Delta | gold_full delta | Source |
|---|---|---|---|---|---|---|
| H158 same_init | H155 d1 0.72439, d2 0.72788 (shared init) | 0.72613 | 0.71767 | -0.00846 | -0.0091 | R18-H158_soup_cell_same_init.json |
| H158 cross_init | H150 d1, d2 (both monolithic) | 0.71548 | 0.72306 | +0.00758 | +0.01425 | R18-H158_soup_cell_cross_init.json |
| H158 cross_arm | H150 d1 + H152 EMA d1 (different recipes) | 0.71649 | 0.71022 | -0.00627 | +0.01665 | R18-H158_soup_cell_cross_arm.json |
| H160 soupB (k=2, both split) | H160 d3, d4 | 0.71618 | 0.69922 | -0.01696 | +0.00575 | R19-H160_soup_cell_soupB.json |
| H160 k3 | d1, d2 (mono) + d3 (split) | 0.71322 | 0.72150 | +0.00828 | +0.00753 | R19-H160_soup_cell_k3.json |
| H160 k4 | d1, d2 + d3, d4 | 0.71583 | 0.71066 | -0.00517 | -0.0013 | R19-H160_soup_cell_k4.json |

k-sweep non-monotone (0.69922 -> 0.72150 -> 0.71066). H160 KILLED on its registered bars; the log bars k3's graduate-clearing number from promotion (H141 discipline).

## (b) Pattern explanation

**Most supported: there is no reliable soup effect - every banked delta is inside single-draw noise.** Against an independent-draw null (sd 0.0146 at m=2 to 0.0133 at m=4, using the pre-repair sd; repair shrinks these ~8%), the six deltas land at -0.58 to +0.60 null-sd, worst -1.16. The +0.00758 that motivated H160 and the -0.01696 that killed it are both ordinary draws from that null.

Alternatives: (1) split-executor endpoints average-destructive - partially supported, confounded (the only pure-split pair is the worst read; the two positive cells both contain the monolithic H150 pair); (2) ingredient behavioural disagreement predicts damage - best continuous predictor (max per-subset ingredient range vs delta, Pearson -0.61, n=6); mechanistic anchor: soupB's delucionqa reads 0.6943 from ingredients 0.8798/0.7718 - 0.08 BELOW its worse ingredient; (3) "weak draw 3 poisons soups" - REFUTED (sign does not track d3); (4) "more ingredients degrade" - REFUTED (non-monotone).

## (c) Honest null + gold_full audit

A soup is deterministic given its ingredients - delta carries no noise of its own; its distribution is over ingredient re-draws. Pooled empirical soup effect: mean delta -0.00350, sd 0.00978, SE 0.00399, naive 95% CI [-0.0115, +0.0045] - indistinguishable from zero, slightly negative. Caveats: shared ingredients across reads, m varies, one cross-recipe cell.

gold_full does NOT track arena for soups: Pearson +0.36, sign agreement 4/6, and the two disagreements are the decision-relevant reads (soupB gold +0.006 / arena -0.017; cross_arm gold +0.017 / arena -0.006). A gold_full-greedy soup selector would have kept soupB.

## (d) Zero-training-cost registered design at k=6 (adopted as R20-H173)

- PRIMARY: uniform k=6 soup S6 (trunk+task head, equal weights, no selection) vs the k=6 single-draw mean M6, ONE read. Branches pre-declared: >= +0.005 -> averaging re-enters as a registered candidate under a fresh confirmation pair; <= -0.005 -> the H160 kill is recipe-general, class closed; inside -> NULL, class closed as a mean lever. The pooled prior expects null-to-negative; the read buys CLOSURE, and 0.005 is the reporting band, not a significance claim
- SECONDARY (<= 4 mechanism reads, no promotion route): split-only pairs (d5,d6), (d3,d5); mixed pairs (d1,d5), (d2,d6). If split endpoints are average-destructive, split-only pairs read negative (joining soupB) while mixed pairs read at/above the pooled mean -0.0035. A consistent 3-vs-3 sign split is the licensed conclusion
- Free dividend: six same-recipe draws give a direct 5-df per-draw sd estimate from ONE recipe, replacing half-normal range pooling across heterogeneous pairs
- Greedy/learned soup selection: RECOMMENDED AGAINST - the only non-arena surface (gold_full) sign-flips on the largest movers; precondition for any such registration: exhibit a non-arena surface with >= 5/6 sign agreement on the banked soup reads. None exists

## (e) SWA-in-training: NOT worth 6.5 GPU-h

SWA averages points on one trajectory - strictly closer than the same-init cell, the worst same-recipe soup on both surfaces. The in-run class is already priced: H152 proved EMA a VARIANCE lever (spread -63%) with no mean gain; the H120 mechanism note explains why (OneCycle's anneal is itself an implicit average - a trailing average serves a lagged iterate). The same 6.5 GPU-h buys a seventh flagship draw instead.

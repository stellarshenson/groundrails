# R14 hypotheses - LENS: OBJECTIVE / LOSS

**Lens id**: L2 (OBJECTIVE / LOSS). Candidate ids `L2-C1` .. `L2-C4`. Renumber the lens prefix if the campaign assigned a different index; the C-numbers are stable.

**Discipline**: analysis only in producing this document. No training, no GPU, no arena statistic used to set a threshold, a mix size or a lambda. Every anchor number below was re-read from the JSONs on disk, not from the evidence prose. Polars only.

**Scope of the lens**: remediations that change the training objective - auxiliary supervision, nuisance decorrelation, aggregation-shaped losses, calibration and anchoring terms. Weight-space averaging (H118 soup, H120 EMA), head fusion at serving (H104/H106), token-head-as-primary (H102), the DANN lambda sweep (H99), forced group balance (H95), the H119 serving wrapper and H121 distractor-window certification are all closed and none of the four candidates re-proposes them. H129 output-ensemble distillation is queued and orthogonal to all four.

---

## Verified anchors used by every bar

Re-read from `R9-H105_windowed_result.json`, `R9-H105_draw2_windowed_result.json`, `R9-H105_result.json`, `R9-H105_draw2_result.json`, `DR_lane_draw*_*_result.json`.

| anchor | draw 1 | draw 2 | pair mean |
|---|---|---|---|
| arena mean (PRIMARY windowed read) | 0.70471 | 0.70151 | **0.70311** |
| finqa | 0.6489 | 0.6176 | **0.6333** |
| techqa | 0.6934 | 0.6745 | **0.6840** |
| pubmedqa | 0.6201 | 0.5925 | **0.6063** |
| delucionqa | 0.7975 | 0.8358 | 0.8166 |
| gold_full (in-domain hold) | 0.8788 | 0.8240 | **0.8514** |
| DR control arena mean | 0.69826 | 0.70713 | **0.70270** |
| DR control finqa | 0.6870 | 0.7098 | **0.6984** |
| DR control gold_full | 0.8040 | 0.8323 | **0.81815** |

Standing clauses applied to all four bars:

- **Mean HOLD (ruling 9)** - arena pair mean must not fall more than 0.06 below 0.70311, i.e. `>= 0.6431`
- **gold_full HOLD** - in-domain claim-level pair mean `>= 0.8314` (0.8514 - 0.02), except L2-C4 which inherits the H117 continuity form
- **pubmedqa guardrail** - E3 establishes pubmedqa as finqa's only replicated cost partner (r = -0.84 across five independent slicings). Every finqa-targeting bar carries `pubmedqa pair mean >= 0.5763` (0.6063 - 0.03)
- **delucionqa is never a bar** - E2 measures its 2-draw noise at approximately +/-0.10 and E3 puts 85% of its config-to-config variance in seed noise with zero of fourteen configs clearing 2 sigma. It is reported as a diagnostic surface only
- **finqa length-confound clause** - E1 measures response verbosity alone reading finqa AUROC 0.6958 against the shipped model's 0.6489, with the model already riding it at Spearman +0.294. Any finqa-primary claim must additionally clear its threshold after residualizing the response score on log mean-sentence-length. A finqa gain that a length heuristic explains is not admitted
- **Paired draws** - all arms run under the H126 seeded-paired facility (shared init and permutation with their control), so the comparison is paired and the H117 precedent for pairing is preserved
- **Two draws minimum** - unpaired single-draw per-subset blind noise is 0.0295 (E5); one paired draw resolves 0.03 at ~2 sigma, two resolve 0.02

---

## Summary of the four candidates

| id | name | primary target | killgate cost | training cost |
|---|---|---|---|---|
| L2-C1 | Label-conditional numeric-nuisance decorrelation | finqa + pubmedqa together | CPU only, then ~0.3 GPU-h | ~12 GPU-h |
| L2-C2 | Derivation-vs-corruption consistency supervision | finqa | CPU only, then ~0.3 GPU-h | ~14 GPU-h |
| L2-C3 | Window-bag noisy-OR objective (read-shaped aggregation) | techqa + arena mean | CPU census, then ~0.5 GPU-h | ~20 GPU-h |
| L2-C4 | Absolute-anchor repair of the pairwise term | arena mean, with finqa as the mechanism clause | ~0.5 GPU-h | ~12 GPU-h |

The four are mutually independent in mechanism and can be run in any order. L2-C4 is the cheapest and is a post-mortem instrument for a result the campaign already owns; L2-C1 is the cheapest new mechanism; L2-C3 is the largest bet.

---

## L2-C1 - Label-conditional numeric-nuisance decorrelation

### The defect

The shipped clean checkpoint penalises numerically dense claims, and the penalty runs against the label. Recomputed here from `R12-H121_gateA_scores.parquet` (sentence score = max over windows, finqa only):

| numeric tokens in the claim | sentences | mean score |
|---|---|---|
| 0 | 36 | 0.6765 |
| 1-2 | 167 | 0.5695 |
| 3-5 | 166 | 0.5110 |
| 6+ | 194 | 0.5194 |

E1 records the same shape and adds the decisive asymmetry: the 6+ bucket is 95.4% faithful, so the model is suppressing exactly the claims most likely to be true. The mirror probe separates the two sides cleanly - subtracting evidence-window digit fraction moves finqa the wrong way at every k (-0.0296 / -0.0463 / -0.0413), so the model's positive response to number-dense **evidence** is genuine signal and only its negative response to number-dense **claims** is defective.

### Mechanism, justified without the arena

Negatives in this task are overwhelmingly manufactured by editing a numeral. VitaminC revisions and TabFact refutations in the clean 685,670-row mix, the H108 lane's approximately 45k unit/period/scale corruptions, and the DR lane's 7,862 number-change deltas (`DR_judge_summary.json`) all produce a negative that differs from its positive by a digit. The consequence is a spurious label-conditional statistic inside the **training mix**: within a DANN group, claims carrying more numerals are more often labelled 0. A cross-encoder minimising BCE will absorb that as a claim-side prior, because it lowers loss on the training distribution and costs nothing there.

That statement is about the training corpus, not about RAGBench. It is measurable on the mix alone, which is what the killgate does. The arena measurement above is corroboration that the prior survived into the deployed function; it sets no threshold and selects no feature.

### The intervention

Add one term to the existing loss (`BCE + DANN` becomes `BCE + DANN + lambda_dec * R`):

```
R = mean over cells (g, y) of  corr( task_logit , digit_fraction(claim) )^2
```

where cells are (DANN group g, label y) pairs with at least 8 members in the batch, and `digit_fraction` is the fraction of characters in the claim string that are digits. Conditioning on the label is what makes this a nuisance-removal term rather than a capability-removal term: it removes only the component of the score that tracks digit density **within** a class, and leaves any genuine class-conditional information intact.

Fixed a priori and never swept: `lambda_dec = 1.0` (the squared correlation is bounded in [0, 1], so this makes the term's maximum magnitude commensurate with BCE), one nuisance feature only, no schedule. A sweep would reproduce the H99 lambda-sweep pattern, which is closed.

**Claim length is deliberately excluded from the nuisance set.** E1 shows length is arena-predictive on finqa (0.6958 alone), so decorrelating on length would be a change whose sign is knowable only from arena statistics. Digit density is the opposite case: it has no a priori entailment content and its correlation with the label is a documented construction artifact.

### Why it escapes the finqa/pubmedqa axis - the discriminating prediction

Every lever that has moved finqa has moved pubmedqa the other way (E3: r = -0.84, replicated in five slicings, jackknife floor -0.710). That axis is register capacity being traded. Removing a nuisance prior is not a trade - E1's frozen-score proxy `score + 1.0 * sentence_digit_fraction` moves finqa +0.0124, tatqa +0.0154 **and pubmedqa +0.0133 together**, at an arena-mean cost of -0.0004. This hypothesis therefore predicts a co-movement no register-pressure lever has ever produced, and that co-movement is written into the bar as a co-primary clause. If finqa rises while pubmedqa falls, the arm acted as register pressure and is not admitted as this mechanism.

The frozen-score proxy also sets the floor on expected size: a blunt uniform additive shift on frozen scores already buys +0.0124 finqa. Removing the prior at training time, where the trunk can re-allocate rather than be offset, should exceed it.

### Killgate (precondition, before any GPU training arm)

1. **CPU only, Polars, minutes.** Over the clean 685,670-row training mix, compute per DANN group the point-biserial correlation between claim digit fraction and label. **KILL if `|r| < 0.05` in every corruption-bearing group** - the confound is not in the data and there is nothing to decorrelate.
2. **Approximately 0.3 GPU-h, frozen weights, in-domain.** On RAGTruth-dev (legal, no arena), compute the partial correlation between the H105 logit and claim digit fraction controlling for the label. **KILL if `|partial r| < 0.05`** - the prior did not survive into the function in-domain, so the arena reading is a transfer artifact and a training-time term is unlicensed.

Total killgate cost: under half a GPU-hour, and step 1 alone can kill it for zero.

### Bar (pre-registered, blind)

- **CO-PRIMARY A** - finqa pair mean `>= 0.6733` (+0.040 over 0.6333), both draws positive against their paired controls
- **CO-PRIMARY B** - pubmedqa pair mean `>= 0.6063` (must not fall at all; this is the mechanism discriminator, not a guardrail)
- **CONFOUND CLAUSE** - the finqa gain must be `>= +0.020` after residualizing response scores on log mean-sentence-length
- **HOLD** - arena pair mean `>= 0.6431`; gold_full pair mean `>= 0.8314`
- **REPORT ONLY** - delucionqa, tatqa (E3: tatqa carries no detectable structure, 131% noise share)
- **VERDICT MAP** - both co-primaries and the confound clause clear → ADMIT. Finqa clears but pubmedqa falls → REFUTED AS MECHANISM, record as a register-pressure lever. Finqa clears only before residualization → NOT ADMITTED (verbosity shift). Neither clears → KILL.

### Cost

Two paired draws, no extra rows, negligible per-step overhead (one batch-level correlation over scalars). **~12 GPU-h** including reads.

### Legality and closed-line check

No new corpus, no arena data, no RAGBench derivative. Not a lambda sweep (single fixed value). Not DANN (no new adversarial head; a direct penalty on a scalar statistic). Not the H119 canonicalization wrapper (that was a frozen-weight serving transform; this changes weights and ships no wrapper). Single model at serving, 307M unchanged, no head added.

### Known risks

- The term is batch-statistical, so it interacts with batch composition. Mitigation: the cell minimum of 8 and the existing fixed permutation facility.
- Squared correlation is a linear-dependence measure; a non-linear residual prior would survive it. Accepted - the measured relationship is monotone and near-linear across the four buckets above.

---

## L2-C2 - Derivation-vs-corruption consistency supervision

### The defect

E4 diagnoses finqa as an **objective** problem with a data component, and quantifies it: 362 of 563 finqa scored sentences (64.3%) assert at least one number that appears in no retrieved window, and **75.7% of those are gold-supported**. Those sentences score 0.487 against 0.641 for sentences whose numbers are all literally present. RAGBench-finqa calls a correctly derived value supported; the training mix (RAGTruth, HaluEval, PsiloQA, VitaminC, TabFact) defines supported as literally entailed. The model is answering the question it was trained on.

The two canonical items:

- **resp 147** (0.0506): "The net change in Aon's unpaid restructuring liabilities during 2007 was a decrease of \$71 million." The derivation sentence, whose operands are all present, scores 0.8943. `71` appears nowhere in the evidence. Both are gold-supported; the min read hands the response 0.0506
- **resp 200** (0.7493, 13.4% of all finqa discordance): "Ratio = \$6.2 billion / \$38.8 billion = 0.16." Both operands are verbatim in the window and belong to **different line items**. The model sees two exact numeric matches and a fluent relational frame and certifies a category error

These are the same failure seen from both sides: an assertion whose atoms are present and whose composition is not. That is E4's cross-cutting finding and it is the mechanism H108's admitted lane already partially targets.

### The intervention

Two components, both objective-side.

**(a) A consistency term over derivation chains.** For a constructed positive whose conclusion value is arithmetically correct, require

```
p(conclusion)  >=  min_i p(premise_i)  -  delta        (hinge, delta = 0.15)
```

and for its corrupted counterpart, require the reverse. Both members carry their **own BCE label at full weight** - see the H117 post-mortem in L2-C4 for why that anchor is not optional. The hinge adds the relation on top of an absolute anchor; it never replaces one.

**(b) A training-only auxiliary head.** A 2-way linear head on the pooled representation predicting quantity-consistency, weight 0.5, trained only on the constructed rows and **discarded at serving**. Its purpose is attribution: its held-out AUC separates "the trunk learned the relation" from "the score head memorised the templates". This is explicitly not head fusion (H104/H106, closed - those fused two heads' outputs at inference) and not token-head-as-primary (H102, closed). One head ships, unchanged.

### The construction (deterministic, CPU, legal)

Built only from table corpora **already in the clean mix** - TabFact, FEVEROUS, InfoTabS. No RAGBench corpus, no derivative, no LLM judge, no arena statistic anywhere in the pipeline.

For each table, pick two cells and one operation from a fixed set {sum, difference, ratio, percent change}. Emit:

- **positive** - the claim asserts the correct result and binds the correct row labels
- **negative n1** - numeral corrupted, operands and binding unchanged (the near-miss that E1's S1 mechanism has replicated twice)
- **negative n2** - correct arithmetic over the **wrong two row labels** (the resp 200 failure, verbatim operands, wrong binding)
- **negative n3** - operation swapped, operands and binding unchanged

Every label is computed, not judged. The model is never asked to perform arithmetic - E5 is explicit that arithmetic competence is a pretraining-corpus property and no move inside the mmBERT family changes it. It is asked to judge whether an asserted quantity is *compatible with* the evidence quantities and their bindings, which is a relational judgement of the same kind the admitted H108 lane already teaches, extended from present values to absent-but-derived values.

### Why this is not a re-proposal of H108

H108 corrupts units, periods and scales of values that are **present** in the evidence. Every claim in that lane asserts a number the model can find. This lane's positives assert a number the model **cannot** find, and the discrimination it teaches is between two kinds of absence: derivable-and-correct against fabricated-or-misbound. E4 measures that exact population at 64.3% of finqa's scored sentences, and E1 records H108's finqa lift as the campaign's only replicated data lever - so this is the same mechanism family aimed at the part of the population H108 does not reach.

### Killgate

1. **CPU only.** Constructibility census over TabFact / FEVEROUS / InfoTabS: count tables admitting at least one (two-cell, one-operation) tuple whose result is **absent** from the serialized table string and whose n1/n2/n3 counterparts are all constructible. **KILL if fewer than 30,000 positive tuples** - the lane cannot reach a size where the H108 precedent (61,184 pairs) suggests an effect.
2. **~0.3 GPU-h, frozen H105 weights.** Score a 1,000-tuple pilot. **KILL if the frozen model already separates derived-correct from derived-corrupt at AUC `>= 0.70`** - the capability exists and the training term is unlicensed. Expected on the E4 evidence is near-chance, because absent-number sentences score 0.487 regardless of correctness.

### Bar (pre-registered, blind)

- **PRIMARY** - finqa pair mean `>= 0.6933` (+0.060 over 0.6333), both draws same sign. Sized against the mechanism's reach (64.3% of finqa sentences) and the H108 precedent (+0.0849 pair mean over its paired control, E1); it lands 0.0415 below finqa's measured ceiling of 0.7348 under this read, so it is arithmetically attainable
- **CONFOUND CLAUSE** - the gain must be `>= +0.030` after residualizing on log mean-sentence-length
- **ANTI-GAMING CLAUSE** - the in-domain held-out near-miss discrimination AUC (H108-style present-value corruptions) must not fall below the clean-recipe value. A model that learns "absent number implies supported" would clear the primary and fail here. This clause is non-negotiable and is measured in-domain, not on the arena
- **HOLD** - arena pair mean `>= 0.6431`; gold_full pair mean `>= 0.8314`; pubmedqa pair mean `>= 0.5763`
- **REPORT ONLY** - delucionqa, tatqa
- **VERDICT MAP** - primary, confound and anti-gaming all clear → ADMIT. Primary clears, anti-gaming fails → REJECT (the lane taught permissiveness). Primary fails with anti-gaming intact → KILL, and record that finqa's derived-number population is unreachable by constructed supervision

### Cost

Construction is CPU. The lane adds roughly 120k rows to a 685,670-row mix (+17%). Two paired draws at approximately 6.5 GPU-h each plus reads: **~14 GPU-h**.

### Legality and closed-line check

Registers (financial and general tables) are legal; the corpora used are already in the shipped mix. No RAGBench source or derivative. No private gold. No arena quantity enters the lane's size, its operation set or its positive/negative ratio - all are fixed a priori. Serving path unchanged, one head, 307M.

### Known risks

- **Template leakage.** Constructed claims share syntax, so the model may learn the template rather than the relation. Mitigation: the auxiliary head's held-out AUC on a template-disjoint split is reported alongside the bar, and the anti-gaming clause is measured on H108-style rows the lane did not generate
- **The E5 tension, stated openly.** A 307M multilingual encoder cannot verify arithmetic. If the discrimination turns out to require verification rather than compatibility judgement, the arm fails and that failure is informative: it would locate finqa's residual in pretraining, which the 400M cap does not gate

---

## L2-C3 - Window-bag noisy-OR objective (read-shaped aggregation)

### The defect

Training and reading use different evidence geometry. Training scores one (claim, chunk) pair truncated at MAX_LEN 512. The PRIMARY read scores each sentence against every 1,500-character window of the full chunk text and takes the **max over windows**. Nothing in BCE ever trains that max operator, and E5 measures how large the mismatch is: 22.1% of scored pairs exceed 512 tokens, and 46.4% of techqa's score-deciding pairs are truncated.

The operator is doing real selection work. Recomputed here from `R12-H121_gateA_scores.parquet`, over (sentence, document) units with more than one window:

| subset | multi-window units | mean windows | median (max - median) score | fraction with dispersion > 0.05 |
|---|---|---|---|---|
| techqa | 7,301 | 5.16 | 0.0898 | 0.687 |
| delucionqa | 1,180 | 2.95 | 0.0997 | 0.647 |
| expertqa | 662 | 13.81 | 0.1496 | 0.793 |
| finqa | 532 | 3.44 | 0.0425 | 0.442 |
| emanual | 348 | 3.12 | 0.0434 | 0.466 |

The selection misfires in both directions, at item level (E4): finqa resp 217 scores 0.0546 because the supporting value `46.7` is in the response's window pool but not in the window the max selected; delucionqa resp 103 scores 0.2385 because the argmax landed on a window that does not contain the supporting text; delucionqa resp 65 scores 0.9152 because a near-verbatim decoy window certified a fabricated conjunct.

### The intervention

Train the aggregation instead of only the pair. For a training row whose evidence exceeds one 1,500-character window, split it with the read's own geometry (window 1500, stride 750 - the values are fixed pre-run in `R8-H101_windowed_read.py` and are not tuned here), form a bag, and apply the loss to the bag probability under noisy-OR:

```
p_bag = 1 - prod_i (1 - p_i)
loss  = BCE( p_bag , y )
```

Noisy-OR is the smooth surrogate for exactly the read's semantics: a positive bag needs at least one supporting window, a negative bag needs **every** window to be low. That second property is what preserves absolute anchoring - on a negative bag the gradient presses every instance down individually, so no window is left free to drift, which is precisely the failure that cost the H117 margin arm finqa -0.1020 (E1 C2, and see L2-C4).

**No window is ever labelled.** The bag label is the row's existing label; the model chooses which window supports. This is what separates the proposal from H121 (closed), which manufactured *negative* training pairs by pairing a claim with a distractor window certified non-supporting by a lexical grounder. H121 failed on the reliability of that certification; a bag objective needs no certification because it assigns no window-level label.

### Why techqa is the primary target

E3 measures techqa as one of the campaign's cleanest instruments - seed sigma 0.0265, 13% noise share of config-to-config variance, 87% structure - and it has **never been the target of a hypothesis**. It is also by far the most window-exposed subset (83.3% multi-window documents per E2; 7,301 multi-window units and 5.16 windows per unit in the table above). A mechanism about window selection should land there first, and techqa can resolve it where delucionqa cannot.

The one confound to hold fixed: E5 measures 46.4% of techqa's deciding pairs above 512 tokens. **MAX_LEN stays at 512 in both arms** so that truncation is paired out. A MAX_LEN change is a separate, read-lens hypothesis and must not be bundled here.

### Killgate

1. **CPU census over the clean training mix.** Fraction of rows whose evidence text exceeds one 1,500-character window. **KILL if under 20%** - the objective would touch too little of the mix to move a blind read, and the cheaper conclusion would be that the geometry mismatch is a read-side problem only.
2. **~0.5 GPU-h, frozen H105 weights, in-domain, no arena.** On RAGTruth-dev rows with long evidence and annotated hallucination spans, split with the read geometry and measure the **argmax-window hit rate**: does the selected window contain the annotated span? **KILL if the hit rate is `>= 0.90`** - the selection is already correct and there is nothing for a bag objective to sharpen. Also record median (max - median) dispersion in-domain; **KILL if under 0.05**, since a degenerate spread means max and mean already coincide and the aggregation carries no gradient signal

Both gates are arena-free. The arena table above is corroboration only and sets nothing.

### Bar (pre-registered, blind)

- **PRIMARY** - techqa pair mean `>= 0.7240` (+0.040 over 0.6840), both draws positive against their paired controls. That is approximately 1.5 seed sigma unpaired and roughly 3 sigma under the H126 paired-delta SD of 0.014
- **CO-PRIMARY** - arena pair mean `>= 0.70311` (no loss against the clean recipe). A window-selection fix that costs the mean is not a fix
- **HOLD** - arena pair mean `>= 0.6431` (ruling 9, formally); gold_full pair mean `>= 0.8314`; pubmedqa pair mean `>= 0.5763`
- **ABORT SIGNATURE** - finqa pair mean `<= 0.5333` (-0.10) reproduces the H117 comparability collapse and voids the arm regardless of techqa, because it would mean the bag objective destroyed absolute cross-window comparability rather than training it
- **REPORT ONLY** - delucionqa (it is the second most window-exposed subset and is therefore the natural diagnostic surface for this mechanism, and it is barred from being the bar by E2)
- **VERDICT MAP** - primary and co-primary clear → ADMIT. Techqa clears and the mean falls between 0.6431 and 0.70311 → PARTIAL, re-run with the third draw before adjudication. Abort signature fires → REFUTED, and the H117 diagnosis generalises to all aggregation-shaped objectives

### Cost

Bags raise per-step cost in proportion to mean bag size. On the assumption the census returns 20-30% multi-window rows at mean bag size 2.5, the effective step cost is approximately 1.6-1.8x. Two paired draws at approximately 9.5 GPU-h each plus reads: **~20 GPU-h**. This is the most expensive of the four and should run after its census returns.

### Legality and closed-line check

Window geometry values are copied from the shipped read, which was fixed pre-run and never tuned on the arena. No window-level labels, therefore not H121. Not a read change - the serving read is byte-identical to the current one. No new corpus. Single model, one head, 307M.

### Known risks

- **Noisy-OR saturates** on large bags: with 10 windows at p = 0.3 each, p_bag exceeds 0.97 and the gradient vanishes. Mitigation: cap bags at 6 windows chosen contiguously, fixed a priori, and report bag-size distribution alongside the result
- **Positive bags are permissive by construction.** A model can satisfy a positive bag by scoring one window high and the rest arbitrarily. That is the intended semantics of the read, but it means the mechanism's benefit must come from the negative bags, which is where the anchoring pressure lives. If the census returns few long-evidence negatives, the arm is weak by construction and the census should report the positive/negative split as part of gate 1

---

## L2-C4 - Absolute-anchor repair of the pairwise term (H117 post-mortem)

### The finding this rests on

E1 attributes the H117 margin arm's finqa collapse (0.5850 against a paired control's 0.6870, -0.1020 with the arena mean flat) to a loss of absolute score comparability across windows. Reading `DR_lane_trainer.py` and `DR_lane_assemble.py` locates the exact implementation fact behind that diagnosis:

- `DR_lane_assemble.py:16-17, 67` - the clean member of every minimal pair is written with `label = -1, bce_mask = True`, role `clean_partner`, described in the header as "margin-only partner row ... never in BCE"
- `DR_lane_trainer.py:17` - "clean partners carry the corrupt partner's DANN tag ... and DO enter the domain loss; **they never enter BCE**"
- `DR_lane_trainer.py:149-163` - the only gradient reaching a clean partner is `relu(0.25 - (p_clean - p_corrupt))`
- Executed configuration, read from `DR_lane_draw1_margin_result.json`: `arm = margin`, `lambda_margin = 0.3`, `margin_m = 0.25`, `seed = 1117`, `lane_rows = 30369`, `margin_pairs = 13898`

So 13,898 of the lane's 30,369 rows were trained with **no absolute target at all**. Their scores were free to occupy any level satisfying a 0.25 gap. That is a mechanically sufficient explanation for a loss of absolute cross-window comparability, and it predicts the damage should concentrate where the read compares the most raw scores - which is what E1 observes, finqa carrying 5.18 windows per sentence, the highest multiplicity among the numeric subsets.

Note also that the in-domain hold did not see it: margin draw 1 gold_full 0.8042 against control draw 1 0.8040. The damage was invisible to every in-domain instrument the arm carried.

### The intervention

One change, nothing else. Give the clean partner its own BCE label 1 at full weight (`bce_mask = False`), keep `lambda_margin = 0.3`, `m = 0.25`, the identical row set, the identical adjacent-pair packing and the identical seeds. Seeded-paired against the banked DR control draws, so the comparison is the same paired contrast the H117 amendments were built for.

This is not a duplication of H117 and not a lambda change. It is a single, mechanically motivated correction to how the hinge is anchored, and its result adjudicates a general question the campaign needs answered before any other structural loss is proposed: **is pairwise ranking harmful here, or is unanchored pairwise ranking harmful here?** L2-C2's derivation hinge and L2-C3's bag objective both depend on that answer, which is the argument for running this first.

### Killgate

**~0.5 GPU-h, frozen weights, no training.** Score a fixed held-out probe slice of the public mix with both banked checkpoints - `DR_lane_draw1_margin` and its paired `DR_lane_draw1_control`. Measure, on identical inputs:

- the distribution of raw sigmoid scores (mean, SD, and the 10th/90th percentiles)
- within-document across-window score dispersion on long-evidence rows
- the rank correlation between the two checkpoints' scores

**KILL if the margin checkpoint's score surface is calibrated indistinguishably from the control** (score SD within 10% and dispersion within 10%) - the "lost absolute comparability" diagnosis would then be wrong, the finqa collapse would need a different explanation, and restoring the BCE anchor would be unmotivated. **PROCEED if the margin checkpoint shows materially wider or shifted per-window dispersion**, which is the fingerprint of scores that satisfy a gap constraint without an absolute target.

A zero-GPU precursor is available and should be run first: correlate the margin arm's per-subset damage against each subset's windows-per-sentence multiplicity, computed on the banked parquet. E1 asserts that relationship; confirming it costs CPU minutes and strengthens or weakens the gate before any card time is spent.

### Bar (pre-registered, blind)

Inherits the H117 continuity form recorded in `DR_lane_trainer.py:20-21`, plus the mechanism clause that distinguishes the repair from the original arm.

- **PRIMARY** - arena pair mean `>= 0.7127` (DR control pair mean 0.70270 + 0.010)
- **MECHANISM CLAUSE** - finqa pair mean `>= 0.6684` (DR control finqa pair mean 0.6984 - 0.030). The -0.1020 collapse must not recur. This clause is the whole point of the arm: clearing the primary while finqa collapses again would mean the anchor was not the cause
- **HOLD** - gold_full pair mean `>= 0.8132` (DR control 0.81815 - 0.005, the original H117 form); arena pair mean `>= 0.6431` (ruling 9)
- **GUARDRAIL** - pubmedqa pair mean `>= 0.5763`. Note the original margin arm *raised* pubmedqa (+0.0317, z = +1.46 per E3) while destroying finqa, consistent with it having been projected onto the register axis
- **REPORT ONLY** - delucionqa (E3 measures the margin arm's -0.0392 at z = -0.91, which adjudicates nothing), emanual (the original arm's +0.1561 at z = +7.4 is the one large real effect and its survival is the interesting secondary reading)
- **VERDICT MAP** - primary and mechanism clause clear → ADMIT, and the general rule "anchor every member of a structural loss with its own absolute target" is established for L2-C2 and L2-C3. Mechanism clause clears, primary does not → the anchor explains the finqa damage but the margin term buys nothing; record as DIAGNOSTIC RESOLVED, do not ship. Mechanism clause fails → the diagnosis is wrong, pairwise ranking is harmful per se in this read, and both L2-C2's hinge and L2-C3's bag term must be re-argued before running

### Cost

Same row set, same schedule, negligible overhead. Two paired draws: **~12 GPU-h**. Reads reuse the banked DR control draws as the paired baseline, so no control re-run is needed.

### Legality and closed-line check

Uses the existing DR lane, which is already assembled and judged on disk. No arena statistic sets `lambda_margin`, `m`, or the row set - all are carried over unchanged from the executed H117 arm, and the single changed value is a binary mask. Not a duplication of H117 (the instruction permits building on it), not weight averaging, not head fusion. The EMA and step-cosine instruments in the trainer are off by default and stay off.

### Known risks

- **H117 draw 2 may still be in flight.** If the second margin draw lands before this arm is scheduled, its finqa reading either confirms the collapse replicates or refutes the single-draw finding. Either way, this arm should be scheduled only after that verdict, and if draw 2 shows no finqa damage the arm is void
- **Anchoring the clean partner changes the class balance** of the lane (13,898 rows move from unlabelled to label 1). Mitigation: report the lane's post-repair positive rate against the control's, and note that the control arm already carries 2,573 label-1 reclaim rows, so the direction is not novel to the lane

---

## Ordering and shared discipline

**Recommended order**, on cost and on informational dependency:

1. **L2-C4** - cheapest, and its verdict licenses or voids the structural-loss designs in L2-C2 and L2-C3. Killgate is half a GPU-hour with a zero-GPU precursor
2. **L2-C1** - cheapest new mechanism, killable for zero cost on a CPU pass over the training mix, and the only candidate with a discriminating co-movement prediction (finqa and pubmedqa together) that separates it from every register-pressure lever in the record
3. **L2-C2** - largest addressable finqa population (64.3% of scored sentences), construction is CPU, and its hinge design depends on L2-C4's answer
4. **L2-C3** - largest bet and largest cost; run after its CPU census and after L2-C4

**Shared clauses that apply to all four**:

- No arena quantity enters any lane size, mix ratio, lambda, margin, window geometry or threshold. Every such value in this document is either carried over unchanged from an executed arm, copied from the shipped read, or fixed a priori with the justification stated inline
- All bars are blind and ceiling-blind. The label-ceiling numbers (finqa 0.7348, delucionqa 0.6657, pooled faithful oracle 0.7560) appear here only as feasibility analysis, never as a bar
- delucionqa is never a bar in any of the four. E2 and E3 agree it cannot resolve the effects being attributed to it
- Every finqa-primary bar carries the length-residualization clause. E1's measurement that verbosity alone reads 0.6958 on finqa against the model's 0.6489 makes this non-optional
- Every arm reports gold_full and RAGTruth EN/non-EN. The H117 precedent shows in-domain holds can stay flat while the blind read loses 0.10 on a subset, so a passing in-domain hold is necessary and never sufficient
- All four keep a single model, a single serving head, 307M parameters and MAX_LEN 512. No candidate touches the serving read

**What this lens does not propose, and why**:

- **Per-group loss weighting** - E1's S2 establishes that register-rebalancing pressure does lift finqa (H90 vs H91, +0.0291 attributable to DANN alone on byte-identical data), but both configurations that maximised it were refuted on the mean (H99 at lambda 0.1241, -0.0300; H95 forced balance, -0.0095), and forced subset balance plus the lambda sweep are both closed. There is no version of per-group weighting left that is not one of those two
- **A conjunct-decomposition loss** targeting E4's partial-conjunct failures (delucionqa resp 65 at 30.6% of that subset's discordance) - the mechanism is real and cross-cutting, but its measurable upside sits on a subset that cannot carry a bar, and E4 shows two of delucionqa's top three items are relevance rather than entailment failures and are unreachable by any entailment objective. It is recorded here as a mechanism worth revisiting only if the arena gains negatives
- **Calibration losses of the temperature or focal family** - they rescale a score surface monotonically and therefore cannot change an AUROC, which is rank-only. The only calibration that matters under this read is *relative* calibration across windows and sentences, which is what L2-C3 and L2-C4 address directly

---

## Artifacts consulted

Evidence packs `R14_evidence_E1_finqa.md`, `R14_evidence_E2_delucionqa.md`, `R14_evidence_E3_covariance.md`, `R14_evidence_E4_items.md`, `R14_evidence_E5_capacity.md` (all five read in full).

Numbers re-verified directly from `R9-H105_windowed_result.json`, `R9-H105_draw2_windowed_result.json`, `R9-H105_result.json`, `R9-H105_draw2_result.json`, `DR_lane_draw1_control_result.json`, `DR_lane_draw2_control_result.json`, `DR_lane_draw1_margin_result.json`, `DR_lane_draw{1,2}_control_windowed_result.json`, `DR_lane_draw1_margin_windowed_result.json`, `R10-H108_lane_draw{1,2}_windowed_result.json`, and recomputed from `R12-H121_gateA_scores.parquet` (finqa numeric-token score buckets; per-subset multi-window dispersion).

Implementation facts read from `DR_lane_trainer.py` and `DR_lane_assemble.py`.

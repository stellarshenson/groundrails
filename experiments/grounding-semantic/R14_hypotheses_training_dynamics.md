# R14 hypotheses - LENS 4: TRAINING DYNAMICS

**Scope**: remediations reachable through the training *procedure* - schedule, sampler, batch composition, draw economics - at byte-identical data, byte-identical architecture and byte-identical batch size. No data lane is proposed here; no read change is proposed here.

**Discipline**: ANALYSIS ONLY in producing this document. No training, no GPU, Polars throughout. Every number below is either quoted from a named banked artifact or computed here from a schedule/config on disk; the free measurements are listed in section 2 with their reproduction. No bar is set from an arena statistic - all four bars are transfers of the campaign's own noise record and the incumbent pin.

**Incumbent pin** (author ruling 1): clean recipe **0.7031** pair mean, draws {0.70471, 0.70151}. Verified from `R9-H105_windowed_result.json` / `R9-H105_draw2_windowed_result.json`.

---

## 1. What the evidence forces before any hypothesis is written

Four findings from E1-E5 constrain this lens more tightly than any of them constrains a data lane.

**(1) The residual is a variance term, not a bias term.** E5(a): the output-probability mean of the two H105 draws reads **0.72067**, +0.01756 over the 0.70311 pair mean, and it beats an omniscient per-subset draw-picker (0.71774) by +0.00292 - only possible if the two draws make genuinely different errors *inside* each subset. E5(f): in-domain RAGTruth EN spread across the same pair is **0.0021** at score correlation 0.9775 while blind per-subset SD is **0.0318**. The data pins the function in-domain and leaves it free off-distribution. Consequence for this lens: the highest-value training-dynamics move is not to suppress seed variance but to **harvest** it, and the harvest has to happen in output space - E5's decisive contrast is that weight-space averaging of the *same two objects* reads 0.69218 while output-space reads 0.72067, a 0.0285 gap.

**(2) Within-run trajectory diversity is measured near zero under the current schedule.** R12-H120 was **killed at its instrument**: mean consecutive-step update cosine over the final 20% is **0.9378** against an ABORT threshold of 0.5 (log line 2436). The tail of a OneCycle run is a single coherent drift, not exploration around a basin. Any within-run averaging or snapshotting scheme built on the *current* monotone schedule is dead on that measurement - which is precisely why the schedule itself is the variable worth moving.

**(3) The adversarial signal reaching the trunk is small and violently heteroscedastic per batch.** `R12-H122_gradgate_result.json`: GRL domain-gradient norm **0.4439** against task-BCE **12.206**, i.e. the adversary contributes **3.64%** of the trunk gradient norm at lambda 0.02; and the 16-way/9-way norm ratio is **1.1869 as a ratio of means but 25.16 as a mean of per-batch ratios** - the same object measured two ways, three orders apart, because the discriminator's gradient collapses on easy batches and spikes on hard ones. E3's finding that seed noise is *structured* (delucionqa~techqa r = **-0.981** across five pure same-recipe replicate pairs, where no intervention differs at all) is the signature that would be produced by exactly this: a high-variance register-invariance signal that lets each run wander a different distance along one register axis.

**(4) The one replicated finqa mechanism is a contrast, and the trainer currently destroys the contrast.** E1 S1: deterministic near-miss corruption over the *same* evidence is the only finqa lever that has replicated (H108 pair mean finqa 0.71815 vs clean 0.63325 = **+0.0849**; DR-2 pilot lane 0.6984 = **+0.0652** independently). E1 C2: the obvious way to exploit a contrast - a pairwise margin loss - broke absolute cross-window score comparability and cost finqa **-0.1020** on a paired seed. Section 2 measures that under the H108 lane's plain `np.random.permutation` the two members of a near-miss pair land in the same batch with probability **6.29e-5**; the mechanism is being taught 5,186 optimizer steps apart.

**What this lens must therefore NOT propose**: weight averaging in any form (H118 soup, H120 EMA - closed); a pairwise-ranking or margin auxiliary (E1 C2, -0.1020 on finqa); R-Drop or dropout-consistency (measured shut, all task-path dropout channels 0.0, R13 record); GroupDRO->DANN curriculum (closed); a lambda dose sweep (H99 closed); forced subset balance (H95 closed); any batch-size or gradient-accumulation change (**author ruling 7 forbids it explicitly**: "all training serializes on GPU1 at the byte-identical recipe (batch 48 / MAX_LEN 512) ... No batch/accumulation changes authorized"). All four hypotheses below hold batch size at 48 and MAX_LEN at 512.

---

## 2. Free measurements taken for this lens

All computed from configs and lane artifacts on disk, CPU only. These are schedule and sampler properties - they contain no arena quantity and set no bar.

### 2a. The OneCycle schedule concentrates update mass 6.6:1 toward the head of the permutation

`OneCycleLR(max_lr=1e-5, pct_start=0.1, anneal_strategy="linear")`, torch defaults `div_factor=25`, `final_div_factor=1e4`. Over the 15,560 steps of a lane run:

| quintile of the run | share of total LR mass | share of total lambda mass | share of lambda x LR |
|---|---:|---:|---:|
| 1 (steps 0-3.1k) | **29.17%** | 10.07% | 18.29% |
| 2 | 30.99% | 20.69% | **33.55%** |
| 3 | 22.14% | 22.86% | 26.58% |
| 4 | 13.28% | 23.17% | 16.18% |
| 5 (final 20%) | **4.43%** | 23.21% | 5.40% |

An example's contribution is weighted by the LR at the step it is seen. The first quintile of the permutation carries **6.59x** the update weight of the last. Which rows land where is currently a per-draw lottery.

### 2b. The Ganin ramp is anti-phased with the LR schedule

lambda(p) = 0.02 x (2/(1+exp(-10p)) - 1) rises monotonically while LR falls after p=0.1. Their product peaks at **p = 0.269** and decays thereafter; 51.8% of all adversarial work (integral of lambda x LR) is spent before p = 0.4, and 5.4% after p = 0.8. Dose-preserving reshapes (identical integral of lambda dt, phase only) change the adversarial work integral by: constant-lambda **+5.2%**, time-mirrored (front-loaded) **+17.7%**. The phase lever is real but small - this is why L4-C4 below is aimed at *adversary competence*, not at re-timing for its own sake.

### 2c. The i.i.d. sampler leaves whole DANN groups out of most batches

Group sizes from E6 (`R14_evidence_E6_train_composition.md`, mix reconstructed byte-for-byte to 685,670 rows / 12 groups). Binomial per-batch occupancy at BATCH 48:

| group | share | E[rows/batch] | P(absent from a batch) | CV of per-batch count |
|---|---:|---:|---:|---:|
| vitaminc | 54.06% | 25.95 | 0.0000 | 0.133 |
| tabfact | 13.50% | 6.48 | 0.0009 | 0.365 |
| psiloqa | 9.00% | 4.32 | 0.0108 | 0.459 |
| halueval | 5.83% | 2.80 | 0.0558 | 0.580 |
| ragtruth_{en,de,fr,es,it,pl,hu,cn} | 2.20% each | 1.06 | **0.3436** each | **0.962** |

**Expected 2.82 of 12 groups absent from any given batch.** With the H108 lane's four extra groups (16 groups, 746,854 rows) it is **4.99 of 16**, and `quant_scitab` (0.157%) is absent from **92.7%** of batches. The discriminator's cross-entropy - and therefore the GRL gradient handed to the trunk - is computed on a random subset of its own label space in every step.

### 2d. Near-miss pairs in the admitted H108 lane are never co-located

`R10-H108_lane.py:381` draws the epoch order as `np.random.permutation(len(ds))` - a flat i.i.d. shuffle with no pair structure. `R10-H108_pairs.parquet` (61,184 rows, columns `claim/chunk/label/tag`) carries **no `pair_id`**, but the pair structure is recoverable from the shared evidence chunk: 22,248 distinct chunks, of which **6,889 carry both a label-1 and a label-0 row**, covering **29,779 rows = 48.67% of the lane**.

Under the current permutation, for any such pair: P(partner in the same 48-row batch) = 47/746,853 = **6.29e-5**. Expected number of co-located pairs out of 6,889: **0.43**. Mean row separation between partners is N/3 = 248,951 rows = **5,186 optimizer steps**, a third of the whole schedule, across which the LR falls by roughly an order of magnitude.

By contrast `DR_lane_trainer.py:116-147` (`build_perm`) already packs each of its 13,898 pairs adjacent and forbids batch-boundary straddling - and the DR **control** arm, which carries this packing with the margin loss switched off (lambda_margin 0.0, `DR_lane_draw1_control_result.json`), is one of only three trained checkpoints in the bank that lifts finqa and delucionqa together in *both* read types (E3(b)). Packing has never been isolated as a variable.

**Reproduction**: the four blocks above are pure arithmetic over `R10-H108_lane.py` constants (`BATCH=48`, `LR=1e-5`, `WARMUP_FRAC=0.1`, `LAMBDA_MAX=0.02`), the E6 group table, and `R10-H108_pairs.parquet`; no model and no arena row is touched.

---

## 3. The hypotheses

### L4-C1 - CYCLIC-RESTART SNAPSHOT ENSEMBLE (variance harvest at one draw's cost)

**Mechanism.** The campaign's single best banked artifact is an *output-space* average of two functions (anchor teacher 0.72067, E5(a)), and its gain is a variance term: it beats a per-subset oracle draw-picker (+0.00292), it is positive on 9 of 10 subsets against the pair mean, and it never lands below the worse parent (E3(c)). Today that average costs **two full draws (~12 GPU-h)** because the only source of functional diversity the campaign has ever used is a fresh initialization. The measurement that explains why nothing cheaper has worked is H120's: consecutive-step update cosine **0.9378** over the final 20% - a monotone OneCycle anneal produces iterates that are all the same function, so snapshots off the current schedule carry no diversity to average. A cyclic schedule manufactures the missing diversity inside one run: split the single epoch's 15,560 steps into three cosine cycles (peak 1e-5, floor 1e-7, no change to batch, MAX_LEN, data, or the number of times any row is seen), snapshot at each cycle floor, and average the three snapshots' output probabilities. Each restart forces the iterate out of its basin and re-descends into a different one on a different data segment; the cost of three members becomes ~1.1 draws instead of 3.

**Why it is not a closed line.** H118 (cross-draw soup) and H120 (within-run EMA) are both **weight-space** averages, and both are killed. This is an **output-space** average, and E5(a) prices the distinction at +0.0285 on the identical pair of objects. Nothing here changes the loss, the architecture, the head count, or the batch.

**Deliverable status.** A 3-member ensemble is 921M of served parameters and is **not shippable** under ruling 9. Its role is as the **teacher for the already-registered R13-H129 distillation**, whose current teacher is the 2-draw mean and whose prediction is a 307M student pair mean >= 0.7091. A cheaper, more-diverse, higher-scoring teacher raises H129's ceiling without re-paying for the target generation (`R13-H129_teacher_targets.parquet`, 685,670 rows, ~3.3 GPU-h already spent - the target-generation pass is re-run against the new teacher for the same ~3.3 GPU-h).

**Economics against H129** (the comparison the lens brief requires). Buying a third and fourth draw to widen the current ensemble costs ~12 GPU-h for 2 extra members. This buys **3 members for ~6-7 GPU-h**, a 3.4x improvement in members per GPU-hour, and it does not compete with H129 - it feeds it. If the killgate fails, zero training GPU-h are spent.

**Killgate (~0.5 GPU-h, read cards only, no training).** The precondition is that **members sharing an initialization still ensemble**. Cyclic snapshots necessarily share init; every ensemble ever measured in this campaign used different inits. The bank contains exactly one shared-init pair: `models/DR-lane-draw1-control` and `models/DR-lane-draw1-margin` - identical seed 1117, identical row set, bit-identical init and permutation, divergent only through the auxiliary loss. Score their output-probability mean through the frozen windowed read. **LICENSE** if the shared-init output mean lands >= +0.005 above the *better* of the two members' arena means; **KILL** if it lands below the better member. Recorded caveat: the margin member is a damaged function on finqa (0.5850), so this gate is a necessary condition on shared-init ensembling and is deliberately not read as an effect size.

**Bar (pre-registered, blind).** Snapshot ensemble read through the frozen windowed decomposed-min gate:
- **PRIMARY (subset)**: finqa >= **0.6633** (clean pair mean 0.63325 + 0.030, i.e. ~0.7 sigma_seed above the anchor teacher's own finqa 0.6626), AND delucionqa >= 0.7167 (clean pair mean 0.81665 - 0.10, the E2 unenforceability floor - a no-collapse clause only, never a target).
- **SECONDARY (mean, legal here because this is a mix-wide lever per session ruling 7)**: ensemble mean >= **0.7207**, matching the 2-draw anchor at ~55% of its GPU cost. ADMIT-AS-TEACHER at >= 0.7207; PARTIAL at 0.7107-0.7207 (beats the pair mean by >= +0.008, still the cheaper teacher); **KILL** below 0.7107.
- **HOLD**: no subset below the clean pair by more than **0.06** (ruling 9); gold_full pair-equivalent >= **0.8414** (clean pair mean 0.8514 - 0.010).

**Cost**: killgate ~0.5 GPU-h (read card); arm 1 cyclic run **~6-7 GPU-h** + 3 snapshot reads ~1.5 GPU-h = **~8 GPU-h**. Teacher re-targeting for H129 if admitted: +~3.3 GPU-h.

---

### L4-C2 - GROUP-QUOTA BATCH COMPOSITION (stratified sampler, mix-preserving)

**Mechanism.** The DANN discriminator is a 12-way (16-way with a lane) classifier whose loss is computed inside a 48-row batch. Section 2c measures that **2.82 of 12 groups (4.99 of 16 with the H108 lane) are absent from the average batch**, each RAGTruth language missing from **34.4%** of batches at a per-batch count CV of 0.962. `R12-H122_gradgate_result.json` shows what that does downstream: the domain gradient is **3.64%** of the task gradient in norm, and its 16-way/9-way ratio is 1.19 aggregated but **25.16 averaged per batch** - the adversarial signal arrives as rare spikes on batches that happen to be hard, not as a steady invariance pressure. E3's measurement that pure seed replication already produces a **-0.981** delucionqa~techqa correlation - the strongest "trade-off" in the entire configuration matrix, present where *no intervention differs* - is the register axis those spikes walk the model along. The remediation is to remove the sampling variance without touching the sample: build each batch by **largest-remainder quota against the global group shares**, with a deterministic rotation carrying the fractional remainders (so `ragtruth_en` appears in every batch instead of 65.6% of them, and `quant_scitab` appears once every 13 batches by construction instead of once every 13.7 by luck). Identical rows, identical global proportions, identical batch size, identical step count. Only the multinomial noise in per-batch composition is removed.

**Why it is not a closed line.** H95's forced 1/13 group balance (REFUTED) *changed the mix* - it re-weighted groups to equality. This changes **no** group's share by a single row; it is variance reduction in the estimator of a fixed quantity, in the same family as stratified over simple random sampling. It also does not change batch size or accumulation, so it is inside author ruling 7. Interaction to record: if R12-H122 (16 -> 9 group collapse, LICENSED, queued) admits, this hypothesis rebuilds its quota table over the 9-group label space; the mechanism is orthogonal to the label-space size.

**Predicted direction.** A cleaner invariance signal is the H90-vs-H91 lever applied at fixed data, which E1 S2 prices at **+0.0291 on finqa** for the discriminator's presence alone on a byte-identical 762,535-pair mix. Because E1 S2 also records that every configuration which *maximized* invariance pressure (H99 lambda 0.1241, H95 balance) was refuted on the arena mean, and E3 puts finqa's systematic cost partner at pubmedqa **r = -0.84** across five independent slicings, pubmedqa is the pre-registered guardrail - not delucionqa, not hotpotqa.

**Killgate (0 GPU-h, CPU replay).** Two clauses, both computed by replaying the sampler offline with no model in the loop. **(i) Occupancy deficit**: the quota sampler must raise the expected number of DANN groups present per batch by >= **2.0 groups** over the i.i.d. sampler on the arm's actual group table (measured headroom today: 2.82 of 12 / 4.99 of 16 absent). **(ii) Gradient-relevance**: on a frozen `R10-H108-lane-draw1` trunk, the mean cosine between the GRL trunk-gradient direction under quota batches and under i.i.d. batches, over 200 batch pairs, must be <= **0.9** - if the two samplers hand the trunk the same direction, the variance being removed does not reach the weights and the hypothesis is dead. Clause (ii) reuses `R12-H122_gradgate_result.json`'s exact instrument and costs ~0.3 GPU-h on a read card. **KILL** if either clause fails.

**Bar (pre-registered, blind, H126 seeded-paired against a fresh control pair).**
- **PRIMARY (subset)**: finqa pair mean >= **0.6733** (clean 0.63325 + 0.040, ~1 sigma_seed of 0.0421) **with sign agreement on both paired draws**. Sign agreement is load-bearing per E1 C3 - one finqa negative traversing the rank range moves the subset 0.049, so magnitude alone adjudicates nothing.
- **VARIANCE (secondary, the hypothesis's own claim)**: pooled per-subset paired-delta SD <= **0.014**, the H126 registered target, against the H105 pair's measured 0.0293.
- **HOLD**: arena mean >= **0.7031 - 0.006** (mean HOLD, no-loss clause per session ruling 7 - this is a lane-neutral schedule change and is not granted a mean-gain bar); **pubmedqa >= 0.5463** (clean pair 0.6063 - 0.06, ruling 9) as the named guardrail; no other subset below its clean pair by more than 0.06; gold_full pair mean >= 0.8414.
- **KILL** if finqa < +0.020 or the two draws disagree in sign.
- **1-draw pilot gate** (H128 precedent): spend draw 2 only if draw 1 reads mean >= 0.700 AND finqa >= clean pair + 0.020.

**Cost**: killgate ~0.3 GPU-h; arm 2 seeded-paired draws ~12 GPU-h + control pair if H122's seeded control is not reusable (+12 GPU-h; reusable in the likely case) + reads ~1 GPU-h. **~13 GPU-h with the pilot gate expected to halve it on failure.**

---

### L4-C3 - MINIMAL-PAIR CO-LOCATION IN THE ADMITTED LANE

**Mechanism.** The H108 quantitative-near-miss lane is the campaign's only ADMITTED lane (pair mean 0.70496) and the only replicated finqa lever (E1 S1: finqa pair mean 0.71815 vs clean 0.63325). Its supervision is a **contrast**: a corrupted quantity claim against a clean claim over the *same evidence chunk*. Section 2d measures that the trainer never presents that contrast: the two members of a near-miss pair share a batch with probability **6.29e-5**, so **0.43 of 6,889 reconstructible pairs** are co-located in an entire run, and the average partner is **5,186 steps away** - across which the LR falls by an order of magnitude (section 2a) and the parameters have moved. Under BCE with a shared encoder, two nearly-identical inputs with opposite targets *inside one batch* produce a gradient whose leading direction is the difference between them; split across a third of the schedule, the same two rows contribute two weakly-related updates at different learning rates from different parameter points. The remediation is the packing `DR_lane_trainer.py` already implements and which has never been isolated: reconstruct pairs by shared evidence chunk, order each pair adjacent (corrupt, clean), and forbid batch-boundary straddling by the same singleton-swap already in `build_perm`. **The loss is untouched** - this is deliberately not a margin term. E1 C2 is the reason: the H117 pairwise-margin auxiliary on a paired seed cost finqa **-0.1020** by destroying absolute cross-window score comparability, which the read's max-over-windows then min-over-sentences depends on. Co-location changes *when* the gradient sees the contrast, not what the objective rewards, so absolute calibration is preserved by construction.

**Supporting contrast already in the bank.** The DR control arm is BCE-only *with* packing and lifts finqa +0.0652 over the clean control on 30,369 lane rows; the H108 lane is BCE-only *without* packing and lifts +0.0849 on 61,184. Both are positive, so packing is not necessary for the effect - the hypothesis is that it raises the yield per lane row, and the two lanes' rows-per-point of finqa differ by 2.6x in the packed arm's favour (30,369 rows / +0.0652 vs 61,184 / +0.0849). That ratio is a hypothesis-generating observation across two different corpora, not evidence; it is why the arm is worth one pilot draw and not more.

**Why it is not a closed line.** Nothing in the closed list touches batch ordering. It is not the margin loss (H117, refuted, and explicitly excluded above). It is not "forced subset balance" (no group share changes). It does not alter batch size or accumulation (ruling 7 safe), the mix, the lane, the DANN group table, the schedule, or the read.

**Killgate - ALREADY MEASURED AND PASSING (0 GPU-h).** The precondition is that pairs are reconstructible from the shipped lane artifact at scale. Measured here from `R10-H108_pairs.parquet`: **6,889 evidence chunks carry both a label-1 and a label-0 row, covering 29,779 of 61,184 lane rows = 48.67%**, against a pre-registered floor of **>= 30% of lane rows**. Second clause, also free: the reconstructed pairs must be genuine near-misses rather than unrelated claims on a shared chunk - require median character-level edit similarity between paired claims **>= 0.80** on a 500-pair sample, KILL below 0.60. (First clause: PASS at 48.67%. Second clause: to run before build; it is pure string work on the lane parquet, no GPU, no arena.)

**Bar (pre-registered, blind; control is the banked H108 pair, which is the lane this modifies).**
- **PRIMARY (subset)**: finqa pair mean >= **0.7482** (H108 pair mean 0.71815 + 0.030, ~0.7 sigma_seed) **with sign agreement on both draws** against the H108 pair.
- **HOLD**: arena mean >= **0.7031** (the incumbent pin - the modified lane must not fall below the clean recipe it is built on); no subset below the H108 pair by more than 0.06 (ruling 9), with **pubmedqa >= 0.5141** named explicitly (H108 pair 0.5741 - 0.06) since E3 puts it at r = -0.84 with finqa; gold_full pair mean >= 0.8484 (H108 pair 0.8584 - 0.010).
- **KILL** if finqa < +0.010 over the H108 pair, or the two draws disagree in sign, or the arena mean falls below 0.6971.
- **1-draw pilot gate**: spend draw 2 only if draw 1 reads finqa >= 0.7382 (H108 pair + 0.020) AND mean >= 0.700.
- **Recorded confound**: the banked H108 draws are unseeded in init and batch order (`R10-H108_lane.py:18`), so the packed arm can be init-paired to a *fresh* seeded H108 control but not to the banked draws. Either spend the seeded control pair (+12 GPU-h) or accept an UNPAIRED-EXCEPT-INIT comparison at the H121 precedent and widen the primary to +0.040. The pilot design below assumes the unpaired route.
- **Length confound clause (mandatory, E1)**: finqa response verbosity alone reads AUROC **0.6958** on this arena, above the shipped model's 0.6489. Any admitted finqa delta must be re-read after residualizing the response score on log mean-sentence-length; if the delta does not survive residualization at >= 50% of its magnitude, record as LENGTH-PRIOR and do not admit.

**Cost**: killgate **0 GPU-h** (clause 1 done, clause 2 CPU); arm 1 pilot draw **~6 GPU-h**, second draw ~6 GPU-h only on pilot pass, reads ~1 GPU-h. **~7 GPU-h expected, ~13 GPU-h worst case.**

---

### L4-C4 - ADVERSARY-COMPETENCE GATING OF THE GRL (dose-preserving)

**Mechanism.** The DANN discriminator is a 2-layer MLP initialized at random and trained by the same OneCycle schedule as the trunk. The Ganin ramp lambda(p) = 0.02(2 sigmoid(10p) - 1) is already **0.00924 at p = 0.1** and **0.0197 at p = 0.5**, while section 2a shows **60.2% of the run's entire update mass is spent before p = 0.4**. So a substantial fraction of the adversarial gradient the trunk ever receives is delivered at maximum learning rate by a discriminator whose accuracy at that moment has **never been measured in this campaign**. If the early discriminator is near-chance, the GRL is injecting a random direction into the trunk at the point of maximum plasticity - a direct generator of the seed-structured register wander E3 measures (delucionqa~techqa r = -0.981 inside pure replicates). The remediation holds the *dose* exactly: keep lambda_max at 0.02 and keep the integral of lambda dt over the run byte-identical, but gate the ramp on measured adversary competence - train the discriminator on **detached** features (no GRL, no trunk gradient) until its running domain accuracy on the training stream clears a fixed multiple of chance, then release the ramp and compress it so the total dose is unchanged. One switch, one scalar, no new parameters, no loss term, no change to batch, mix, or step count.

**Why it is not a closed line.** H99's lambda 0.1241 (REFUTED) and the H99 lambda-sweep proxy are **dose** changes; the total adversarial dose here is identical to the incumbent by construction. GroupDRO->DANN (H95/H96, closed) is a change of *objective family* across phases; this keeps one objective throughout and only withholds the GRL's backward path. This is not H122 (that changes the discriminator's label space, not its schedule) and composes with it.

**Predicted direction and its guardrail.** E1 S2 prices the discriminator's presence at **+0.0291 on finqa** at byte-identical data (H90 vs H91), and every attempt to buy more of it by raising the dose lost the mean. This buys the same commodity by removing wasted early dose rather than adding late dose, so the mean should hold. Guardrail is pubmedqa at r = -0.84 (E3), plus RAGTruth non-EN, because the discriminator's label space is 8/12 language groups and weakening early invariance pressure could surface as a multilingual regression.

**Killgate (~0.3 GPU-h, read card, no training).** The premise is falsifiable directly: **is the early discriminator incompetent?** `models/H117-probe-lam0` is a checkpoint stopped at **3,125 of 14,918 steps (21% of a run)** under the identical recipe (`R11-H117_probe_lam0_train.json`). Evaluate its domain head's accuracy on a held-out slice of the public mix. **LICENSE** if accuracy at 21% of training is <= **0.60** against a 16-group chance of 0.0625 - i.e. the adversary is still far from the 0.9525/0.9370 it reaches by the end (`R12-H123_layerprobe_result.json`) at the moment 60% of the update mass is being spent. **KILL** if accuracy is already >= 0.80 - a competent early adversary means there is no wasted dose and the hypothesis has no mechanism. Second clause, free: the compressed ramp must preserve the lambda integral to within 1% by construction (arithmetic check on the schedule before launch).

**Bar (pre-registered, blind, H126 seeded-paired).**
- **PRIMARY (subset)**: finqa pair mean >= **0.6733** (clean 0.63325 + 0.040) with sign agreement on both draws.
- **HOLD**: arena mean >= **0.7031 - 0.006**; **pubmedqa >= 0.5463**; **ragtruth_nonen >= 0.82** (the H122 precedent clause, since 8 of 12 groups are language groups); no subset below its clean pair by more than 0.06; gold_full pair mean >= 0.8414.
- **KILL** if finqa < +0.020, or sign disagreement, or ragtruth_nonen < 0.82.
- **1-draw pilot gate**: draw 2 spends only on draw 1 mean >= 0.700 AND finqa >= clean pair + 0.020.
- **Attribution clause**: because the dose is held constant, a mean-level move larger than +0.010 with no finqa movement FALSIFIES the stated mechanism and must be recorded as such rather than admitted on the mean.

**Cost**: killgate ~0.3 GPU-h; arm 1 pilot draw **~6 GPU-h**, +6 on pilot pass, reads ~1 GPU-h. **~7 GPU-h expected, ~13 worst case.** Reuses the H122/H126 seeded control pair if that arm lands first.

---

## 4. Ranking, economics, and what is declined

**Order to spend, cheapest decisive first.**

| rank | id | killgate cost | killgate status | arm cost (expected) | what a KILL closes |
|---|---|---|---|---|---|
| 1 | **L4-C3** | 0 GPU-h | clause 1 **PASSED** (48.67% vs 30% floor) | ~7 GPU-h | batch-ordering as a lever on the only admitted lane |
| 2 | **L4-C1** | ~0.5 GPU-h | not run | ~8 GPU-h | shared-init ensembling; forces H129 to keep paying 2 full draws for its teacher |
| 3 | **L4-C4** | ~0.3 GPU-h | not run | ~7 GPU-h | adversarial-schedule timing; leaves lambda as a pure dose question (already closed) |
| 4 | **L4-C2** | ~0.3 GPU-h | not run | ~13 GPU-h | sampler variance; with C4 killed too, the DANN training procedure is exhausted |

**Economics against H129, stated explicitly because the lens brief requires it.** Only L4-C1 touches draw economics, and it does not compete with H129 - it is upstream of it. H129 as registered distils a 2-draw output mean (teacher 0.72067, targets banked at ~3.3 GPU-h, student prediction >= 0.7091) and E5's amendment already notes that a KILL there cannot distinguish "distillation cannot transmit an OOD advantage" from "307M cannot represent the average". C1 changes the input to that experiment: a 3-member teacher for ~55% of the current teacher's GPU cost. Buying the same third member the conventional way costs a full draw (~6 GPU-h) for one member; C1 buys three for ~6-7. **No hypothesis in this lens proposes buying draws for their own sake**; C2, C3 and C4 all spend draws to test a mechanism and all carry 1-draw pilot gates that abort the second draw on a miss.

**Declined, with reasons.**

- **More epochs / a second pass.** E5(b) measures in-domain RAGTruth EN at 0.838 with a **0.0021** draw spread - the model is not underfitting, so a second epoch buys memorization and, on E5(f)'s reading, more OOD divergence. It also doubles every draw's cost against a mean-level SD of 0.0033.
- **Larger effective batch via gradient accumulation.** The strongest variance-reduction instrument available and **explicitly forbidden by author ruling 7** ("No batch/accumulation changes authorized"). Recorded here as the one lever this lens would rank first if the ruling were reopened; not proposed.
- **Early stopping / checkpoint selection on an OOD proxy.** Its selection signal would have to be in-domain, and E5(b) measures the in-domain draw spread at **0.0021** against a blind per-subset spread of 0.0318 - a selector with 0.0021 of dynamic range cannot rank 0.0318 of blind variance. The gate would kill it for ~1 GPU-h, so it is recorded rather than registered.
- **Reducing seed variance for its own sake.** E5(a) makes this actively counterproductive: the variance is worth +0.01756 when harvested in output space. C2 targets the *sampler's* contribution specifically because that component is structured (E3's replicate-vector r = -0.981) rather than usefully diverse.
- **Any delucionqa-primary bar.** E2's ruling stands: delucionqa's 2-draw noise is ~+/-0.10, zero of fourteen trained configurations have moved it beyond 2 sigma, and every banked read already sits above its own faithful-oracle ceiling of 0.6657. All four hypotheses carry delucionqa only as a no-collapse hold clause, never as a target.

**Cross-cutting clause on every finqa bar in this document.** E1 measures response verbosity alone at finqa AUROC **0.6958**, above the shipped model's 0.6489, and the model already rides it at Spearman +0.294. The length-residualization clause written into L4-C3 applies identically to C2 and C4: a finqa delta that a length heuristic explains is not a grounding result and must not be admitted as one.

---

## Artifacts consulted

Evidence packs: `R14_evidence_E1_finqa.md`, `R14_evidence_E2_delucionqa.md`, `R14_evidence_E3_covariance.md`, `R14_evidence_E4_items.md`, `R14_evidence_E5_capacity.md`, plus `R14_evidence_E6_train_composition.md` for the exact mix group table.

Code and configs read for the schedule/sampler measurements: `R10-H108_lane.py` (constants at lines 58-66, OneCycle at 364-366, permutation at 381, Ganin ramp at 402-404), `DR_lane_trainer.py` (`build_perm` at 116-147, seeds at 68).

Banked results verified for every quoted figure: `R9-H105_windowed_result.json`, `R9-H105_draw2_windowed_result.json`, `R9-H105_result.json`, `R9-H105_draw2_result.json`, `R10-H108_lane_draw{1,2}_windowed_result.json`, `R10-H108_lane_draw{1,2}_result.json`, `DR_lane_draw{1,2}_control_*result.json`, `DR_lane_draw1_margin_windowed_result.json`, `R12-H122_gradgate_result.json`, `R11-H117_probe_lam0_train.json`, `R10-H108_pairs.parquet`, `DR_lane.parquet`.

Canonical log: `docs/experiments/semantic-grounding-experiments.md` - author rulings at 2477-2489 (ruling 7 hardware contract, ruling 9 parameter budget), session rulings at 2517-2530 (ruling 7 subset-primary bars, ruling 9 hold clauses), H120 verdict at 2436, H122 gate at 2572, H129 prediction at 2615.

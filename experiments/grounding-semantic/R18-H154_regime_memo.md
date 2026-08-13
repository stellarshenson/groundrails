# R18-H154 Training-Regime Memo - draw-independence via landscape carving

Research-and-design deliverable for the R18-H154 wave (registered 2026-08-12): which optimization-regime methods can shrink the twin protocol's two-seed endpoint spread, and in what order they should be registered. No GPU was spent on this wave; every surviving lever requires its own registered arm before any GPU spend, per the registration.

## The stack under attack

- **Model** - mmBERT-base cross-encoder, 307M params (trunk + fresh task head + fresh 12-way domain head)
- **Objective** - MIL max-over-window BCE per row + per-pair 12-group domain CE through a GRL, lambda on the Ganin ramp 0 → 0.02
- **Optimizer** - AdamW (bias-corrected), full-trunk LR 1e-5, OneCycleLR with 10% linear warmup, grad clip 1.0
- **Budget** - 1 epoch = 14,300 steps over 685,670 rows, <=48 sets / <=96 window pairs per batch, bf16 autocast on trunk encode, fp32 heads and loss
- **Hardware** - 24GB and 96GB cards; twin draw trains in ~5h on the 96GB card (GPU1)
- **Draw construction** - the two draws differ in BOTH init and data permutation by design (fingerprints banked); the 0.0243 two-seed arena-mean spread therefore conflates init variance and data-order variance

## The failure being attacked

- **Measured spread** - two-seed arena-mean spread 0.0243 (draws 0.72498 / 0.70073, honest 2-draw value 0.71286); per-subset swings to -0.076 (tatqa) and -0.072 (techqa)
- **Amplifier identified** (H142-T reflection) - max-over-window selection turns small weight perturbations into score swings, scaling with windows/item: techqa ~156 and pubmedqa ~26 windows/item carry the two largest swings
- **Diagnostic classification** - per Mosbach et al. (`[paper digest] stability of fine-tuning bert.md`), our optimizer hygiene is already the stabilized recipe (bias-corrected AdamW, warmup, low LR), so the residual spread sits in their unsolved class: generalization variance among runs with equivalent training loss, attacked by regularization and averaging, not by warmup/LR fixes
- **Decomposition unknown** - per Dodge et al. (`[paper digest] finetuning weight initializations data orders early stopping.md`), init and data order contribute comparably to seed variance; the campaign has never measured the split, and the split decides which lever class matters

## Lever assessments

Each lever: mechanism, evidence, expected effect on the two-seed spread, GPU cost class, and compatibility with the exact stack. Cost classes: **A** = free / diagnostic (CPU or rides banked artifacts), **B** = negligible (<5% wall-clock, <=1 extra weight buffer), **C** = moderate (+20-50% wall-clock), **D** = ~2x step cost.

### SAM - sharpness-aware minimization (thread ii)

- **Mechanism** - each step ascends to the worst-case weights in an L2 ball of radius rho, then descends from there, minimizing loss value and basin sharpness together (digest: `[paper digest] sharpness-aware minimization.md`)
- **Evidence** - strong for generalization lift across vision and fine-tuning (WRN 1.6% vs 2.2% CIFAR-10; baselines given 2x epochs still lose); NOT measured as a seed-variance reducer - the flat-basin-to-draw-independence link is our hypothesis, not the paper's measurement
- **Expected effect on spread** - plausibly large on the deep-subset swings: flatter basins shrink the score displacement a given weight perturbation causes, which is exactly the max-window amplification chain; unquantified, no published seed-spread numbers
- **GPU cost** - **D**: 2 backprops per step, 14,300 steps stay but wall-clock doubles: ~10h per draw vs ~5h; a 2-draw pair with reads ~24-28 GPU-h vs ~12-14
- **MIL compatibility** - workable; the ascent pass uses the same max-window loss; the argmax window can flip between ascent and descent - pin the window selection across both passes at implementation, or the perturbation optimizes a different bag member than the descent
- **DANN compatibility** - the one real hazard: the GRL flips the domain gradient sign, so a naive ascent over the combined loss seeks worst-case invariance pressure (ill-posed); register SAM on the TASK loss only, domain term excluded from the ascent and applied at the descent point
- **Budget/hardware** - fits 24GB at our batch shape (+1 perturbation buffer, ~1.2GB); the 2x wall-clock is the binding cost, not memory

### SWA - stochastic weight averaging (thread ii)

- **Mechanism** - average the iterates of a non-annealed (constant or cyclic LR) training tail in weight space; the average sits centered in a wide basin (digest: `[paper digest] swa averaging weights wider optima.md`)
- **Evidence** - strong within-run generalization gains in vision/SGD (+0.8% ImageNet top-1 from ~10 tail epochs); transformer-fine-tuning evidence thin; no direct seed-variance measurement
- **Expected effect on spread** - mechanistically the right shape (averaging cancels per-trajectory noise) but our own H120 instrument already measured the obstruction: under OneCycle anneal-to-zero the terminal trajectory is coherent descent (step-cosine 0.9378), so a trailing average is a lagged under-trained iterate - SWA's own paper says the same: it requires a HIGH-LR tail
- **GPU cost** - **B** mechanically (+1 buffer, no extra backprops), but the real cost is a schedule redesign: constant-LR tail replaces the OneCycle anneal, changing the registered recipe
- **MIL/DANN compatibility** - no interaction with max-over-window or the GRL; LayerNorm architecture means the paper's batch-norm re-estimation step vanishes
- **Conflict** - direct overlap with H152's EMA 0.999 (same mechanism family); registering both on one run is double-counting

### Schedule-free AdamW (thread iii)

- **Mechanism** - replace the LR schedule with evaluation-aware Polyak-Ruppert averaging: gradients at y (interpolation of base iterate z and running average x), serve x; only warmup survives as a schedule element (digest: `[paper digest] the road less scheduled schedule-free.md`)
- **Evidence** - strongest optimizer-suite evidence in the survey: matches or beats tuned cosine schedules across 28 problems including ImageNet and LLM pretraining, won MLCommons AlgoPerf 2024 self-tuning track, 10 seeds per task; no BERT-class fine-tuning with adversarial losses in the suite
- **Expected effect on spread** - removes the endpoint's dependence on where the anneal lands; the served x is a long-window average, so trajectory noise is suppressed by construction; plausible spread cut, unquantified
- **GPU cost** - **B**: no extra backprops; <=1 extra weight buffer (~1.2GB fp32)
- **MIL/DANN compatibility** - clean: it is a drop-in AdamW replacement; the loss shape is irrelevant to the averaging mechanics; warmup retained covers the GRL ramp's early phase unchanged
- **Conflict** - x is already an average, so H152's EMA on top is double-averaging; it also REPLACES OneCycle, which dissolves the H120 instrument's kill premise (that kill reasoned about annealed trajectories) - schedule-free reopens within-run averaging by changing the regime it was measured under
- **Integration note** - checkpoint/eval paths must save and serve the x sequence, not the training iterate y/z

### SMART-class - adversarial smoothness + Bregman proximal trust region (thread iv)

- **Mechanism** - two separable parts: embedding-space adversarial smoothness (minimize worst-case symmetric KL under bounded input perturbation) and Bregman proximal updates (each step anchored to the previous iterate, pricing aggressive moves) (digest: `[paper digest] smart robust efficient fine-tuning.md`)
- **Evidence** - SOTA single-model GLUE/SNLI/ANLI at publication, gains concentrated on lower-resource tasks; the proximal component is the literature's most direct attack on our diagnosed failure ("late-seen hard rows steer the endpoint", H142-T reflection)
- **Expected effect on spread** - the proximal anchor directly damps per-step trajectory noise; adversarial smoothness flattens the function locally; both point at endpoint variance, but neither paper measures seed spread
- **GPU cost** - proximal-only **B** (a parameter-difference penalty, no extra passes); full SMART **D** (ascent passes per step, SAM-class cost)
- **MIL compatibility** - proximal is loss-agnostic; the adversarial term needs the same task-loss-only scoping decision as SAM
- **DANN compatibility** - caution: the proximal anchor resists ALL per-step movement including the GRL's invariance pressure, effectively attenuating lambda; if registered, price the interaction by logging domain-head accuracy against the banked trajectory
- **Regime caveat** - the paper's own task pattern (largest gains on low-resource tasks) matches R3F's scaling warning below

### R3F-class - function-space trust region to the pretrained model (thread iv)

- **Mechanism** - penalize symmetric KL between fine-tuned and pretrained outputs on clean and noise-perturbed inputs, anchoring the fine-tune to the pretrained function (digest: `[paper digest] revisiting few-sample bert fine-tuning.md`)
- **Evidence** - reduces degenerate runs and seed variance in FEW-sample GLUE fine-tuning (20-50 seeds per config); the same paper measures the effect shrinking with data size (on full MNLI even bias correction stops mattering) - our 685,670-row mix is deep in the many-sample regime
- **Expected effect on spread** - small at our scale per the paper's own trend line; the honest prior is weak
- **GPU cost** - **B/C** (extra pretrained-model forward per step; can be precomputed per batch only if inputs are noise-free, they are not, so it is a live forward)
- **MIL/DANN compatibility** - bag-level anchor is well-defined on the max-selected window; no GRL interaction
- **Verdict shaping** - deprioritized; the proximal-only variant of SMART dominates it on mechanism and cost

### Lookahead (thread iii, budget smoothing)

- **Mechanism** - wrapper around AdamW: k=5 fast steps, then slow weights interpolate alpha=0.5 toward the fast weights and the fast weights resync to slow (digest: `[paper digest] lookahead optimizer.md`)
- **Evidence** - variance reduction proven on a noisy quadratic model; gains on CIFAR/ImageNet/LSTM-PTB/Transformer-NMT with default hyperparameters; no BERT-class fine-tuning measured
- **Expected effect on spread** - trajectory smoothing with a hard resync every k steps, which answers the H120 instrument's objection: a trailing EMA under anneal-to-zero lags behind a coherent descent, but Lookahead's slow weights are repeatedly re-anchored to the live trajectory
- **GPU cost** - **B**: +1 weight buffer, one interpolation per 5 steps, no added backprops
- **MIL/DANN compatibility** - optimizer-level wrapper, loss-shape agnostic; composes with OneCycle unchanged (schedule runs on the fast weights)
- **Conflict** - mechanism overlap with H152's EMA (both smooth the trajectory): register as the EMA alternative/successor, not a stack-mate

### Model soups + Re-Basin (thread v, cross-seed merging)

- **Mechanism** - soup: average same-basin fine-tunes, greedy selection on held-out validation (digest: `[paper digest] model soups.md`); Re-Basin: permute one model's units to align with a reference before averaging across different inits (digest: `[paper digest] git re-basin.md`)
- **Evidence** - soups: greedy soup beats the sweep's best model (ViT-G/14 90.94% ImageNet) but the shared-init constraint is hard; Re-Basin: first zero-barrier LMC between independently trained ImageNet ResNets modulo permutation, strongest on wide ConvNets, transformers not its strong suit
- **Expected effect on spread** - a soup of k same-basin draws cancels idiosyncratic endpoint noise directly; but the campaign has already measured the failure mode: H118 cross-draw soup KILLED (0.69218 vs parents 0.70311) - weight-space averaging is closed in its naive form
- **GPU cost** - soup ingredients are existing draws (**A/B** if init-paired draws exist); Re-Basin alignment is a **A**-class diagnostic (seconds of matching + activation statistics) with a training cost only if a merger is actually built
- **MIL/DANN compatibility** - a soup changes served weights, not the read; serving-legal
- **Hard constraint** - our twin draws differ in init by design; the legal variant is an init-PAIRED multi-draw soup via the H126 seeding facility (several same-init draws varying order/dropout only, greedy-souped on gold_full); Re-Basin is gated on a barrier measurement, see the diagnostic below

### Early stopping / in-domain checkpoint selection (thread i)

- **Mechanism** - Dodge et al.: expected validation performance rises with restarts, and per-trial early stopping reallocates compute to more trials; the served weights become a selected checkpoint, not the terminal iterate (digest: `[paper digest] finetuning weight initializations data orders early stopping.md`)
- **Expected effect on spread** - attacks the endpoint directly: the terminal step of a 14,300-step OneCycle run is one noisy point; selecting the best of k banked checkpoints on gold_full (in-domain, contamination-legal) shrinks endpoint variance without touching training
- **GPU cost** - **B**: trainer change to bank k checkpoints + in-domain reads (rides the existing eval suite)
- **MIL/DANN compatibility** - none; selection signal is in-domain only, so no arena contamination
- **Conflict** - composes with H152 (select among EMA checkpoints); the interaction to record is that selection reuses the gold_full hold signal, slightly spending its freshness as a hold - price it in the registration

### Init-vs-order decomposition (thread i, diagnostic arm)

- **Mechanism** - train an init-paired pair (H126 facility: seed re-issued after model construction, bit-identical init asserted) varying only data order, then compare its spread against the twin's full-seed spread; the difference attributes the 0.0243 to init vs order per Dodge's factorial logic
- **Expected effect on spread** - none directly; it is the targeting measurement that decides whether soup-class (init-side) or regime-class (order/trajectory-side) levers deserve the next GPU
- **GPU cost** - **C+**: two training draws (~10-12 GPU-h) for a measurement; sequence AFTER the cheap levers, only if their verdicts leave the attribution decision live

### Mixup (recorded REFUSED)

- Interpolating inputs and labels breaks MIL bag semantics (the max-window label is not interpolable) and breaks DANN group tags; not registered, refusal recorded so the wave's review does not re-raise it

## (a) The DANN head as a variance diagnostic

The adversarial machinery already instruments the trunk every step; the wave can harvest endpoint-stability signals at zero GPU1 cost.

- **Domain-gap trajectory replay** - the trainer logs domain CE and domain accuracy per step; on the two banked twin checkpoints plus mid-run resume points, refit a fresh 12-way probe on frozen CLS features at fixed intervals and track the two draws' separability trajectories; convergence to the same gap = functional alignment even under weight divergence, divergence late = an endpoint-stability alarm that fires BEFORE the arena read
- **Gap-vs-spread correlation** - if the domain-gap trajectory spread across seeds correlates with the per-subset arena spread, the DANN instrumentation becomes the cheap proxy for the expensive two-draw arena measurement, usable inside future arms as an early abort signal
- **Lambda-schedule lever (thin evidence, register last)** - H93 measured invariance pressure monotonically helping transfer, H122 froze the group count at 12 after its kill, and the NON-ADVERSARIAL-INVARIANCE-SWAP premise was measurably false at lambda 0.02 (no degenerate equilibrium); a higher plateau or a late-decay lambda is a LIVE direction but interacts with every trust-region lever above (proximal anchoring attenuates effective lambda) - any lambda arm must log the domain-gap trajectory as its first-class output
- **GRL-in-ascent rule** - for SAM/SMART registrations: the domain term is EXCLUDED from any adversarial ascent pass; the sign-flipped gradient makes a worst-case-invariance direction ill-posed

## (b) Conflicts with the registered H152 (EMA 0.999 + window dropout)

- **EMA vs SWA** - same mechanism family (within-run weight average); never both on one run; SWA additionally requires replacing the OneCycle tail that H120's instrument leaned on
- **EMA vs schedule-free** - schedule-free's served x is already a long-window average; stacking EMA double-counts the averaging; schedule-free is a REGIME SWAP against OneCycle+EMA, not an add-on
- **EMA vs Lookahead** - both smooth the trajectory; Lookahead's resync answers the H120 lag objection, so it is the EMA successor if H152's EMA under-delivers, not a co-lever
- **SAM/SMART-full vs wall-clock** - 2x step cost turns a ~5h draw into ~10h; the H152 pair + H153 queue already commits GPU1; a SAM pair is a ~24-28 GPU-h decision and must clear a proportionally higher evidence bar
- **Window dropout** - no mechanical conflict with any lever; the one implementation rule is SAM's: pin the argmax window selection across ascent and descent passes, since dropout already perturbs selection between steps
- **Attribution discipline** - H152 ships its two regularizers bundled with a recorded attribution caveat; any further lever registered before H152's verdict compounds the bundling problem - the order below sequences around it

## (c) Recommended registration order

1. **Diagnostics first (class A, GPU0/CPU, before any training arm)** - Re-Basin weight/activation-matching barrier between the two banked twin checkpoints (if the modulo-permutation barrier is ~zero, cross-init merging revives; if not, thread v closes permanently on our architecture); DANN domain-gap trajectory replay on banked checkpoints (validates the diagnostic for later arms)
2. **Hold for the H152 pair verdict** - the incumbent variance attack prices the remaining gap; H153 (batch stratification) is already queued behind it
3. **Lookahead pair (class B)** - if H152's EMA under-delivers on spread, register Lookahead as the trajectory-smoothing replacement; if EMA passes, Lookahead is recorded as superseded, same mechanism
4. **In-domain checkpoint selection (class B)** - stacks legally with whatever smoothing survives; register as a trainer-instrumentation amendment (bank k checkpoints, select on gold_full), not a retraining arm
5. **Schedule-free pair (class B, regime swap)** - replaces OneCycle; mutually exclusive with EMA; strongest optimizer-suite evidence in the survey; the registration must rewrite the H120-instrument premise explicitly (anneal removed, averaging made explicit)
6. **SMART proximal-only (class B)** - the trust-region probe; full SMART (class D) only if the proximal probe shows the direction pays
7. **SAM task-loss-only (class D)** - the big-ticket flat-basin arm, registered only if cheaper levers leave spread > 0.010; ~24-28 GPU-h, GRL excluded from ascent, window selection pinned
8. **SWA constant-LR tail (class B + schedule redesign)** - third weight-averaging variant; only if both EMA-class and schedule-free fail, and its registration must confront the H120 0.9378 step-cosine measurement head-on
9. **Init-paired soup / init-vs-order decomposition (class C+)** - register only if the diagnostics in step 1 leave the attribution decision live after the cheap levers read
10. **DANN lambda scheduling (class C)** - last and gated: only if the domain-gap diagnostic shows gap instability tracking endpoint spread; H122 froze group count, not lambda; interacts with every trust-region lever above

## Campaign cross-references

- **H117 paired-margin REFUTED** - auxiliary objective terms are not free; SMART/R3F registrations inherit the burden of proof that their auxiliary terms are regularizers (variance-side), not ranking terms (mean-side), which is what killed H117
- **H118 cross-draw soup KILLED** (0.69218 vs parents 0.70311) - weight-space averaging across draws closed in naive form; Re-Basin diagnostic above is the only revival path
- **H120 within-run EMA KILLED at its instrument** (step-cosine 0.9378 under OneCycle anneal-to-zero) - the binding precedent on trajectory averaging; SWA and schedule-free registrations must state why the changed regime voids it; note H152's EMA 0.999 rides the same schedule H120 was measured under, and the H152 registration already owns that tension
- **H122 DANN group collapse KILLED** - group count frozen at 12; lambda magnitude remains the only live DANN knob
- **H126 seeding facility** - init-pairing infrastructure exists and is verified; makes the init-vs-order decomposition and init-paired soup constructible
- **H142-T reflection** - the target numbers: spread 0.0243, tatqa -0.076, techqa -0.072, amplification scales with windows/item (techqa ~156, pubmedqa ~26); banked truncated-regime seed sd constants (pubmedqa 0.0216, hotpotqa 0.0144, tatqa 0.0290) do NOT bound the windowed protocol
- **H151b/c** - the read-side pooling attack (top-k mean vs max) runs in parallel on banked checkpoints; training-regime levers and read-side pooling are complementary fronts on the same amplifier, and a pooling verdict lands before most of the order above
- **H152 / H153** - incumbent training-side fronts; the order above sequences around their verdicts by construction
- **Doctrine** - two-draw adjudication with init/perm fingerprints and pre-spend census binds every arm this memo proposes; the 0.010 spread bar from H152 onward applies to every survivor's registration

## Sources digested (references/papers/)

- `[paper digest] stability of fine-tuning bert.md` - Mosbach et al. 2020, https://arxiv.org/abs/2006.04884
- `[paper digest] finetuning weight initializations data orders early stopping.md` - Dodge et al. 2020, https://arxiv.org/abs/2002.06305
- `[paper digest] sharpness-aware minimization.md` - Foret et al. 2021, https://arxiv.org/abs/2010.01412
- `[paper digest] swa averaging weights wider optima.md` - Izmailov et al. 2018, https://arxiv.org/abs/1803.05407
- `[paper digest] the road less scheduled schedule-free.md` - Defazio et al. 2024, https://arxiv.org/abs/2405.15682
- `[paper digest] revisiting few-sample bert fine-tuning.md` - Zhang et al. 2021, https://arxiv.org/abs/2006.05987
- `[paper digest] smart robust efficient fine-tuning.md` - Jiang et al. 2020, https://arxiv.org/abs/1911.03437
- `[paper digest] model soups.md` - Wortsman et al. 2022, https://arxiv.org/abs/2203.05482
- `[paper digest] git re-basin.md` - Ainsworth et al. 2022, https://arxiv.org/abs/2209.04836
- `[paper digest] lookahead optimizer.md` - Zhang et al. 2019, https://arxiv.org/abs/1907.08610

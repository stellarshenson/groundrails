I have read the complete field (3 evidence packs, 6 angle packs = 18 candidates, 6 skeptic verdict sets = 18 verdicts). Here is the synthesis.

---

# Round 12 Synthesis - Complete Field

**Judge run**: re-run over the full field after the truncated prior synthesis (2 of 6 angle packs). All hypothesis IDs assigned here are fresh. **The truncated synthesis's numbering is VOID and is not inherited.** Each of the six angle packs independently numbered its candidates H119/H120/H121, so candidate names below are given in full to disambiguate.

**Field**: 18 candidates → 9 KILL (final) → 9 survivors → 2 merges → 8 distinct survivors ranked.

---

## 1. Summary - the decisive facts first

**Fact 1.** No survivor in the field can fund the +0.0369 gap alone. The largest honest post-skeptic expectation among survivors is +0.008 to +0.010 on the mean. The sum of all five top-ranked expectations, assuming perfect additivity and zero displacement, is ≈ +0.031 - and the FM1 record (one admitted lane in two, at +0.0019) says additivity will not hold.

**Fact 2.** The two largest headroom pools in the arena are now unattacked. pubmedqa (0.5741 under the H108 incumbent, +0.18 headroom) and hagrid (0.6477, +0.12) together hold 46% of weak-subset headroom, and both lanes aimed at them were killed on measurement. Nothing in the top 5 targets either mechanism.

**Fact 3.** The field's cost distribution is heavily skewed to the cheap end, and the two cheapest survivors are also the only two that fully solve FM4. R12-H119 (deterministic frozen-weight read) and R12-H120 (within-run paired delta) together cost ≤ 3 GPU-h and carry zero draw-noise term.

**Fact 4.** One mechanism received opposite verdicts from two skeptics. Window-bag / max-over-windows training appears as "SERVE-GEOMETRY-MIL" (angle 4, PROCEED-AMENDED) and "fragment-register training parity" (angle 6, KILL, on measurements that invert its motivating premise). This is only visible over the complete field and is item 6 of the awaiting-author list.

**Branches on the round bar:**
- **If the author holds 0.74 as the Round 12 bar** → verdict is **NOT-REACHABLE-THIS-ROUND**: register the top 5 as a portfolio, expect to land 0.708-0.716, and treat Round 12 as a variance-and-instrument round.
- **If the author accepts an interim bar of 0.72** → verdict is **REACHABLE-CONDITIONAL**: it requires R12-H121 to fire at the top of its band (+0.015 or better) plus one of R12-H119/H120/H122 to hold, with no displacement.
- **If the author reopens the parameter budget (decoder class, Mode 3)** → verdict is **NEW-CLASS-REQUIRED**: the record's own escalation path, blocked only on the author's word, and the only route in the log to an effect larger than +0.02.

---

## 2. Kills dropped (9, final)

| Candidate (angle) | Killing ground |
|---|---|
| BRIDGE-COMPOSE two-hop lane (a1) | Founding premise false by measurement: ≥13.8% of the mix already carries multi-span support vs its own <3% threshold; frozen read never presents two hops in one window; hotpotqa control is already 0.6991, above its own success target |
| CONCLUSION-INFERENCE PMC-abstract lane (a1) | Binding refused-list resubmission (Round 10 hedged-discourse-forge / hedged-verdict-register); arena pubmedqa documents carry zero RESULTS/CONCLUSIONS structure; hedge-marker base rate 37.3% vs its own ≥50% precondition |
| PRETRAIN-RETENTION-LLRD (a2) | Its own precondition (b) fired: H107-vs-H108 lower-layer excess +6.16%/+6.46% against a ≥10% bar, and the shape statistic sign-flips between draws of the same recipe |
| SAM-FLATNESS (a2) | Gate cannot reject (false-pass ≈ 1); mechanism contradicted by H118's own measurement (barrier-free, flat in-domain chord); variance bar demands a 3-6x variance collapse and is wired as a REFUTE on a passing mean |
| ModernBERT-large capacity probe (a3) | Attribution invalid (whole-model swap: corpus, objective, tokenizer, vocab, width, depth, languages); already priced in the record at "+0.00-0.02, below noise"; ~40-45 GPU-h for a declared non-shippable checkpoint |
| DISTRACTOR-DEPTH CALIBRATION (a4) | Measured label corruption: 99.6% of VitaminC rows (54% of the mix) have an entailing sibling revision that the highest-overlap distractor rule selects by construction, backpropagating label 0 onto true-positive pairs |
| crop-invariance (a5) | Its own Gate B executed and failed: best-window content recall median 0.530, only 9.8% of label-1 croppable rows reach 0.80 containment, best window IS offset 0 in 69.2% of rows; motivating finqa anomaly has reversed sign on the H108 recipe (+0.0043/+0.0015) |
| paraphrase-invariance-partner (a5) | Instrument anti-selected: the 2,573 banked reclaims are near-copies (median similarity 0.972, 80.6% ≥0.95) contaminated with judge-passed stutter; Gate 3's 40k floor needs ~785k judged rows; cost wrong by 6-7x |
| fragment-register training parity (a6) | Premise inverted by two free measurements: mid-window share correlates **positively** with blind AUC (+0.233) and with the banked H101 windowing gain (+0.276); anchor already re-attributed to Mode 3 by H85 and H101 |

---

## 3. Merges

**M1 - R12-H121 absorbs two killed candidates.** Same-document distractor-window negatives (angle 1) is exactly the re-registration that angle 4's skeptic prescribed when killing DISTRACTOR-DEPTH ("distractor windows drawn from within the SAME document ... costs nothing to certify"), and per its own amendment A10 it delivers BRIDGE-COMPOSE's cross-composition conflation-FP suppressor on the evidence side at no extra cost. Stronger framing kept: angle 1's, with angle 4's depth-curve read (capped depths 1/2/4/8/all, label-decomposed) folded in as the shared Gate A instrument.

**M2 - window-bag training is one mechanism with two verdicts.** SERVE-GEOMETRY-MIL (angle 4) and fragment-register training parity (angle 6) are the same intervention: emit the read's own 1500/750 windows in training and put BCE on the max-scoring window. Angle 4's skeptic returned PROCEED-AMENDED with 7 binding defects; angle 6's skeptic returned KILL with measurements attacking the shared premise. Stronger framing is angle 4's (it carries the label-1 aggregation fix A5 that angle 6 independently flagged), but the merged entity inherits angle 6's measured objections - the mid-window-share inversion and the ~93%-RAGTruth frequency handle - as binding. **Ranked 7, below the cut, pending the author's ruling (awaiting-author item 6).**

**M3 - residual convergence, not a survivor.** Two independent skeptics (angle 5 on crop-invariance, angle 6 on fragment-parity) killed different candidates and proposed the *same* salvage: train-time evidence-offset augmentation at constant row count - sample each over-length evidence's 1500-char span from a random start rather than always from position 0, with no consistency term, no window rows, no multiple-instance learning. Two adversarial reviewers converging on one construction is signal. It was never proposed as a candidate and needs its own registration and pre-gate. Author's call (item 10).

**Non-merge, sequencing dependency.** R12-H119 (serving-side numeric canonicalization, frozen weights) is a strict prerequisite diagnostic for rank-6 tabular serialization parity's numeric-surface variant: if canonicalization moves nothing deterministically, the training-side surface variant loses its motivation.

---

## 4. Ranking method and the FM4 filter applied

Score = (expected mean gain, post-skeptic-correction × mechanism confidence) / honest GPU-h, gated by adjudicability. Standard error of a 2-draw mean is ~0.0042 (pooled draw SD ~0.006), so an effect below +0.010 is unresolvable at n=2 **unless** the design is genuinely paired. Three designs in the field are genuinely paired: R12-H119 (deterministic, zero variance), R12-H120 (EMA and final arms from the same run - no draw-noise term at all), R12-H122 (row count and step count unchanged, so batch order is shareable). R12-H121 is **unpaired-except-init** because its row count changes - its bar is raised accordingly.

All cost figures use the skeptic-corrected wall-clock: a training draw is 5.3-6.0 GPU-h (not 3.0 - the 10,886 s figure is a resumed partial run), and a full 10-subset windowed arena read is ~0.25 GPU-h (measured 15 min, 77,171 forwards), not 1-2.

---

## 5. Top 5 - registration-ready records

### R12-H119 - NUMERIC-SURFACE CANONICALIZATION (serving wrapper, frozen weights, zero training)

**Causal claim.** Part of finqa's residual is surface-form mismatch, not derivation: claim numbers occur in their own evidence at a different surface form while numerically equal. An idempotent, label-free, subset-blind numeric canonicalizer applied symmetrically to claim and evidence immediately before tokenization raises claim/evidence agreement and therefore the score of genuinely supported numeric sentences, with the weights, read formula, window geometry and gate data byte-identical.

**Falsifiable numeric prediction.** Blind windowed decomposed-min mean rises **+0.002 to +0.006** on both frozen pairs (H105 pair 0.7031 → 0.705-0.709; H108 pair 0.70496 → 0.707-0.711). finqa **+0.010 to +0.030** (H105 pair 0.6489/0.6176; H108 pair 0.7291/0.7072). Every non-numeric subset within ±0.010 - legitimate as a tight clause because the read is deterministic and carries zero training variance.

**Two-sided pre-registered bar.**
- Improve: mean delta ≥ **+0.003** on **both** pairs, and finqa delta ≥ +0.010 on at least three of the four draws.
- Hold: no non-numeric subset falls more than **0.015**; the two draws of a pair must not disagree in sign on finqa.

**Kill-gate (zero GPU, CPU audit committed first).** Commit `R12-H119_surface_audit.py` and adjudicate against the reproducible number, not the proposal's. The reproducible lift is **+8.3 points** of finqa claim-number/evidence agreement from **one** rule (thousands-separator removal); currency spacing, percent spacing, accounting parentheses and whitespace normalization each measure **+0.0**. KILL if the committed per-rule ablation shows no rule with a non-zero measured effect. KILL if the shipped transform alters ≥5% of characters on covidqa/pubmedqa/hotpotqa/hagrid chunks.

**Binding amendments (skeptic 6).** (1) Adjudicate gate 1 against the measured +8.3, and do not restate the threshold downward post hoc - replace it with "only rules with a non-zero measured ablation effect may ship". (2) DELETE every tatqa clause; its precondition measures 0.0 (canonicalization) / 2.8% (numeric equality), not 17.3%. (3) DROP the accounting-parenthesis rule (year/citation corruption 84.7% hotpotqa, 54.2% hagrid, 48.0% expertqa) and DROP whitespace normalization (changes 100% of covidqa and techqa chunk tokenizations). Ship at most: thousands-separator, currency spacing, percent spacing. (4) Run **both** directions as separate deterministic reads - strip-separators (claim/evidence agreement) and add-separators (train-surface parity, RAGTruth carries 2,928 separated numbers) - because the mechanism argues one way and the training surface the other. (5) Prediction band restated to +0.002/+0.006 mean, finqa +0.010/+0.030. (6) The transform must be subset-blind, applied identically to all ten subsets and every future corpus, and shipped in the library serving path; a transform kept because it helped finqa on the arena is arena-fitted preprocessing and out of bounds. (7) Cost line corrected to ~77k forwards per read.

**Cost.** ≤ **2 GPU-h** total (8 frozen reads at ~0.25 each, both directions × both pairs) plus ~30 min CPU. No training. No queue contention.

**Sequencing.** **First, immediately, on GPU2 (RTX 5000 Ada 32GB) or GPU0.** Fully parallel to the DR draws on GPU1. Blocked only on awaiting-author item 2 (legality ruling). Its result is a prerequisite input to rank-6 tabular serialization parity.

---

### R12-H120 - TRAJECTORY-EMA (within-run exponential moving average of the terminal epoch fraction)

**Causal claim.** The terminal iterate of a single training trajectory carries step-level gradient noise that is draw-specific; averaging along **one** trajectory (linearly mode-connected by construction) removes that noise without the functional divergence that killed cross-draw souping. Note explicitly in the registration: R11-H118 closed weight averaging **across** draws (near-orthogonal displacements, cosine 0.1861/0.2262); this is the within-trajectory case and is a distinct mechanism.

**Falsifiable numeric prediction.** EMA arm pair mean ≥ base **+0.005**, and the paired per-draw delta (EMA minus final, same run, zero draw-noise term) positive on **both** draws. Honest prior is lowered: OneCycleLR anneals to ~0, which is itself implicit averaging, so this is the regime where the EMA/SWA literature reports the smallest gains. A null is not surprising and must not be reported as one.

**Two-sided pre-registered bar.**
- Improve: pair mean ≥ base + 0.005 **and** within-run paired delta > 0 on both draws.
- Hold: gold_full ≥ final-weights gold_full − 0.010 (negative filter only - gold_full licenses no positive blind prediction); no blind subset < 0.55.
- REFUTE band: pair mean in [base, base+0.005). KILL: pair mean < base, or paired delta negative on either draw.

**Kill-gate (zero GPU, in-run instrument).** Log the running mean consecutive-**step** update cosine cos(W_t − W_{t−1}, W_{t−1} − W_{t−2}) over the final 20% of the epoch. LICENSE the blind read only if that mean is **below 0.3** (oscillation - averaging has noise to cancel). ABORT if **above 0.5** (coherent descent - the EMA is a lagged, under-trained iterate). Register explicitly that this is the **opposite** direction to the originally proposed c_traj ≥ 0.5 licence.

**Binding amendments (skeptic 1).** A1 - strike the resume.pt CPU gate entirely (wrong regime at 51-67% of epoch, wrong granularity at 1000-step windows, inverted direction); replace with the step-cosine instrument above. A2 - strike the "nonstationary gradient reversal layer (GRL) / lambda still ramping" clause; it is measurably false (lambda moves 1.16e-5 = 0.058% of its value over the terminal 20%). A3 - subset predictions are DIAGNOSTIC ONLY, not bar-eligible; correct the cited evidence - hotpotqa and techqa are **not** the high-variance subsets (hotpotqa ranks 5th of 10 on the H105 pair and tightest of 10 on the H108 pair). A4 - strike the variance falsifier |EMA(d1) − EMA(d2)| < 0.0032; one spread against one spread at n=2 has no power. Record as diagnostic. A5 - gold_full pre-read filter as a negative filter only. A6 - one bar-eligible buffer (decay 0.999 from 80% of epoch); drop decay 0.9999 (effective window exceeds the tail). A7 - ride a **queued** draw, not the running one; EMA and final arms must come from the same run. A8 - all wall-clock figures are ~1.8x optimistic at the measured 1.28 s/step.

**Cost.** ~**1 GPU-h** marginal (2 blind reads at 0.25, 2 gold_full at 0.15). Training marginal cost ~0 - three elementwise operations per step, under 1% of a step, plus one 1.23 GB weight buffer. Standalone fallback if it cannot ride a queued draw: +12 GPU-h.

**Sequencing.** **Folds into the next queued GPU1 draw** (DR draw2 or an R11-H117 arm - author picks, item 11). It produces a second checkpoint from a run already paid for and gives a free paired comparison under FM4. Reads run on GPU0/GPU2.

---

### R12-H121 - DISTRACTOR-WINDOW (same-document non-evidence windows as label-0 negatives)

**Causal claim.** The frozen read takes a max over 3-22.5 windows and chunks per response, of which 29-80% are documents the annotator never utilized, but **no training row has ever paired a true claim with a support-free window of its own document**. The max is therefore taken over an input distribution the model has never seen, and every additional scored window raises the max for near-miss negatives faster than for positives. This is the only untested negative-construction axis in the campaign - every recorded construction (H107 span corruption, H108 quant_corrupt, DR-H112 infill, DR-H114 xattn-blind, R11-H117 margin pairs) varies the **claim**; this varies the **evidence**.

**Falsifiable numeric prediction.** Pair mean **+0.010 to +0.020** over the pinned control. Per-subset, stated as signed deltas against the control draws only: finqa, techqa, emanual and hotpotqa positive; covidqa and pubmedqa **secondary positive targets** (not placebos - measured distractor fractions 0.506 and 0.584 mean the mechanism applies there). Record at registration that this prediction is 5-10x the only admitted data-lane effect in campaign history (H108, +0.0019, itself inside the noise band), and that one of two lanes has been admitted while the other cost −0.0384.

**Two-sided pre-registered bar.**
- Improve: pair mean ≥ control + **0.010** (≈ 2.4 SE at the unpaired-except-init design; +0.005 is 1.2 SE and is not adjudicable).
- Hold: no subset drops more than **0.06** on the 2-draw mean (0.02 clauses fire 2 times in 3 on pure noise - median same-recipe per-subset spread is 0.0260 and 63.3% exceed 0.02); no blind subset < 0.55; per-source distribution logged pre-run with RAGTruth-derived rows capped at ≤50% of the lane.

**Kill-gate.** Gate A (~1-2 GPU-h, frozen weights, no build): extended per-window dump retaining the full score vector and argmax index. KILL if <15% of top-window mass on ungrounded sentences sits on lexically support-free windows. This gate settles the finqa/hagrid windowing-anomaly cause whichever way it lands and is worth its GPU-hour regardless. Gate B+C **jointly on one 2,000-window candidate sample** (zero GPU): require ≥95% label purity on 300 eyeballed **and** lexical-tier separability < 0.95 AUC on a held-out sample simultaneously. These two pull in opposite directions by construction; if no filter setting satisfies both, KILL before any build.

**Binding amendments (skeptic 3).** A1 - strike the covidqa/pubmedqa placebo channel; it is refuted by measurement and would refuse the lane exactly when the mechanism works. A2 - re-price every bar against the actual control: under clean+H108 the per-subset controls are finqa 0.7182 (not 0.62-0.67), techqa 0.6996, emanual 0.6426, hotpotqa 0.6991, pubmedqa 0.5741. The registered "finqa 0.62-0.67 → ≥0.68" is already met before a row is written. A3 - per-subset clauses at 0.02 are inside noise; raise to ≥0.06 or drop them for the mean plus a 0.55 floor. A4 - register as UNPAIRED-EXCEPT-INIT (row count changes → step count and OneCycleLR schedule change → batch order cannot be shared) and raise the bar to control + 0.010. A5 - Gate B and Gate C measured jointly or KILL. A6 - declare and cap the substrate: only >1500-char documents yield a second window, so an uncapped 45k lane is ~85-90% RAGTruth material, which is the H107 displacement setup. A7 - pin the R12 incumbent (see awaiting-author item 1); the in-flight DR lane trains `H108.public_train()` only, so no "clean + H108" control checkpoint exists. A8 - Gate A reads the arena's own sentence-level annotations; licensed precedent, but recorded as ANALYSIS only - no per-subset quantity from Gate A may enter the build, the lane size, the filter thresholds or the per-source mix. A9 - record the prior asymmetry at registration; claim no confirmation on anything landing between control+0.005 and control+0.010. A10 - note the absorbed cross-composition negative family so the killed BRIDGE-COMPOSE is not re-proposed.

**Cost.** Gates ~2 GPU-h (Gate A shared with the rank-7 depth-curve diagnostic). Build 0 GPU-h (~3 CPU-h; certifier is the torch-free lexical tier). 2 arm draws ~12 GPU-h. 2 seeded control draws ~12 GPU-h, **charged once and shared with R12-H122 and R12-H123**. Reads ~1 GPU-h. **Total ~27 GPU-h standalone, ~15 GPU-h marginal with the shared control.**

**Sequencing.** Gates run **now** on GPU0/GPU2, in parallel with the DR draws. Training queues on GPU1 **behind** the DR admission verdict and R11-H117, and must not launch before the incumbent is pinned (item 1). First training arm of the round if it clears both gates.

---

### R12-H122 - DANN-GROUP-COLLAPSE (merge the 8 RAGTruth language groups, 16 → 9)

**Causal claim.** Eight of the twelve public domain-adversarial neural network (DANN) groups are the same RAGTruth corpus in eight languages, so half the adversarial label space encodes **language** rather than register, while the blind arena is English-only. The gradient reversal layer therefore applies a persistent, draw-specific trunk gradient to erase a factor that is not the transfer axis. Merging the eight language tags into one changes **only** the tag→index map: rows, labels, sampling order, natural frequency, lambda, ramp and schedule are identical, and the domain head changes by 1,792 parameters of 307.1M.

**Falsifiable numeric prediction.** Pair mean ≥ control + **0.006**, point band +0.006 to +0.012. **No subset concentration predicted** - this is a trunk-gradient-direction claim, so a result where a single subset carrying |delta| > 0.05 supplies the whole mean move FALSIFIES the attribution and is recorded as unattributed.

**Two-sided pre-registered bar.**
- Improve: pair mean ≥ control + 0.006 **with sign agreement on both genuinely-paired draws**.
- Hold: ragtruth_nonen in-domain ≥ control − 0.02 and gold_full ≥ control − 0.010 (guardrails, never adjudication - FM2 forbids in-domain licensing); no blind subset < 0.55.
- KILL: pair mean < control + 0.002, or the paired draws disagree in sign - group cardinality is then not a lever and the DANN design freezes at 16 for the campaign.

**Kill-gate (~0.5 GPU-h, frozen weights, no training).** Not the confusion matrix - it measures the discriminator, while the hypothesis is about the gradient, and high language-ID accuracy is equally consistent with "the GRL is spending trunk budget" and "the GRL failed and the pressure is inert". Instead: on frozen `models/R10-H108-lane-draw1`, fit a 9-label domain head with the trunk frozen (minutes), then over ~200 held-out batches compute (a) the ratio of ‖∂(domain loss)/∂trunk‖ to ‖∂BCE/∂trunk‖ under the 16-label head versus the 9-label head, and (b) the cosine between the two GRL trunk-gradient directions. LICENSE if the 16-way norm ratio is ≥1.15x the 9-way **and** direction cosine ≤0.9. KILL if norm ratio ≤1.05 **or** cosine ≥0.95.

**Binding amendments (skeptic 4).** A1 - replace the licence logic with the gradient measurement above; keep the 16×16 confusion matrix as a recorded diagnostic that adjudicates nothing. A2 - **seeding trap, one-line fix**: `torch.manual_seed(seed)` is called before model construction, and changing n_groups changes the last Linear's RNG consumption, so every subsequent dropout mask desyncs and the paired design silently degrades to unpaired (where +0.006 is unresolvable against a 0.0295 unpaired spread). Re-issue the seed immediately after model construction in both arms and assert bit-identical trunk and task_head init before step 0. A3 - pin the control as the H108-admitted 16-group recipe at 746,854 rows, frozen at registration; do **not** re-baseline onto the DR lane if it is admitted mid-experiment. A4 - strike the FM5 "splits a 307.1M trunk 16 ways" capacity framing (the head delta is 1,792 params) and carry the counter-prior explicitly: R8-H93 measured LOCO transfer rising monotonically with invariance pressure (0.6278 at lambda 0 → 0.7418 at 0.1241, ERM 19th of 22), so merging groups **reduces** invariance pressure and a null must be read as "the H93 direction stands", not as noise.

**Cost.** Gate 0.5 GPU-h + 2 arm draws ~12 GPU-h + reads ~0.5 GPU-h = **~13 GPU-h marginal** against the shared seeded control pair; ~25 GPU-h standalone.

**Sequencing.** Gate runs **now** on GPU0/GPU2. Training queues on GPU1 after R12-H121's arm, or ahead of it if R12-H121's joint Gate B/C fails. Direction is a genuine coin flip, which is what a 13 GPU-h single-variable experiment is for.

---

### R12-H123 - ADVERSARY-DECOUPLED LAYER-MIX (learned scalar mix over all 23 layer CLS vectors)

**Causal claim.** The task head and the 16-way domain discriminator read the **same** vector (`out = trunk(...).last_hidden_state[:,0]`, then `task_head(out)` and `domain_head(GradReverse.apply(out, lam))`), so every register direction the adversary erases from the final-layer CLS is erased from the task head's only input. Replacing the task head's input with a learned convex scalar mix over layers 0-22 (24 parameters; the discriminator keeps reading CLS_22 alone) gives the task head an un-erased read. Every recorded head experiment changed the head or the fusion; none changed the head's **input layer**.

**Falsifiable numeric prediction.** Pair mean **+0.008 to +0.018**, ADMIT at ≥ control + 0.010. Gains predicted where register information is being erased: finqa and hotpotqa positive; covidqa and delucionqa move <0.01 either way.

**Two-sided pre-registered bar.**
- Improve: pair mean ≥ control + **0.010** (adjudicable under a properly seeded paired design at recent same-recipe paired spreads of 0.0032 and 0.00245).
- Hold: no subset drops more than 0.06 on the 2-draw mean; no blind subset < 0.55; RAGTruth non-EN mean ≥ 0.82 (a real re-read, not a formality - the mix could weight English-favouring layers).
- Null band closes the head-input line.

**Kill-gate (~0.5 GPU-h, frozen weights, no training).** Per-layer probe over a held-out 20k slice of `public_train()` (training distribution only; no arena, no gold): for each layer fit a logistic grounding probe → AUC(l) and a corpus probe → group accuracy(l). LICENSE only if **both**: max over l<22 of AUC(l) ≥ AUC(22) + 0.005, **and** group-accuracy(22) at least 0.05 below its mid-stack maximum. The probe runs on **both** H105 draws and both must satisfy it (the P-A → H106 lesson: a mechanism proven on one checkpoint can be a property of that draw).

**Binding amendments (skeptic 2).** A - correct the registration's evidence: the baseline recipe is **12** DANN groups at chance 0.083 with terminal domain-accuracy ~0.55 (6.6x chance), not 16 groups at 0.0625 with 0.44-0.49. State the consequence honestly - the adversary is **not** winning the invariance fight at lambda 0.02, so "erasure at the top" is a hypothesis the probe must establish, not a premise. B - **nested init is mandatory**: initialise the mix logits with essentially all mass on layer 22 (w22 = +5, rest 0) and gamma = 1, or the arm measures "renormalised head input" as well as "layer mix" and is a two-variable change. C - demote the finqa-spread clause to REPORTED, never gated (fires ~40% of the time under the null at n=2). D - keep the two-checkpoint probe requirement; add that the probe is an in-domain **necessary condition only** and licenses the build without predicting blind transfer. E - build the seeded paired control pair and share it. F - restate cost. G - run before the depth-upscale arm.

**Cost.** Probe 0.5 GPU-h + 2 arm draws ~11-12 GPU-h + reads 0.5 = **~13 GPU-h marginal** against the shared control; ~22 GPU-h standalone.

**Sequencing.** Probe runs **now** on GPU0/GPU2 - it is the only arm in the round whose gate can reject its own premise before any build, so run it early even if the training slot is late. Training queues on GPU1 last among the top 5.

---

### Round-level sequencing and the GPU constraint

**Binding hardware fact.** A training draw holds ~33 GB. GPU1 (96 GB) is the only card that can host one at the registered batch 48 / MAX_LEN 512. **GPU0 (RTX PRO 4000, 24 GB) and GPU2 (RTX 5000 Ada, 32 GB) can host frozen-weight reads, probes and gates only** - reads are ~77k forwards and fit comfortably. Any attempt to train on GPU2 requires a batch or accumulation change, which breaks the byte-identical-recipe contract (awaiting-author item 7).

**Wave 0 (now, GPU0 + GPU2, parallel to the DR draws on GPU1, ~5 GPU-h total):** R12-H119 in full; R12-H121 Gate A + joint Gate B/C; R12-H122 gradient gate; R12-H123 layer probe on both H105 draws; R12-H120's step-cosine instrument wired into the next queued draw.

**Wave 1 (GPU1, after DR admission and R11-H117):** seeded control pair (~12 GPU-h, charged once, serves H121/H122/H123), then arms in gate-survival order.

**Total round budget:** ~5 GPU-h of gates and reads off the critical path, plus ~12 GPU-h shared control, plus ~12-13 GPU-h per surviving arm. Three arms → **~56 GPU-h of GPU1-serialized training.**

---

## 6. Survivors ranked 6 and below

6. **Tabular serialization-register parity** (re-render TabFact and the H108 lane at constant row count) - expected **+0.003 to +0.006** mean; below the cut because H108's own precedent (finqa +0.084/+0.090 bought a mean move of +0.0019, paid by covidqa −0.0514 and expertqa −0.0361) makes its +0.010 bar five times H108's mean movement from a smaller intervention, and its skeptic measured that the target register is bracket-JSON (finqa 34.5%, tatqa 24.2%), not the prose form the renderer would emit; ~25 GPU-h with control.

7. **Window-bag / max-over-windows training** (merged SERVE-GEOMETRY-MIL + fragment-parity, M2) - expected **+0.003 to +0.008**; below the cut because the same mechanism drew a measurement-backed KILL in angle 6 (mid-window share correlates +0.233 with blind AUC and +0.276 with the banked H101 gain, so the OOD-fragment story is inverted), its registered mean bar is ~2x its own per-subset arithmetic, and ~93% of added window rows land in the RAGTruth register - a natural-frequency change of exactly the kind R8-H95 measured a blind cost for; ~29 GPU-h.

8. **Serve-exact nested objective** (sentence-bag over window-bag, span-exact negatives) - expected **+0.004 to +0.008**; below the cut because it is registered as an increment above candidate 7 and is unrunnable until that clears, its pubmedqa clause is struck by the H102 falsifier (−0.0654 on the same subset via the same channel), and per-sentence supervision re-weights the loss toward RAGTruth by ~2.2x; **its variance-reduction secondary is the only recorded route left after weight-space closed and should be carried forward** if candidate 7 is ever revived; ~29 GPU-h.

9. **Depth-upscaled trunk** (22L → 34L, identity-initialised block expansion, 367.11M) - expected **+0.001 to +0.005**; below the cut because its Mode-5 mechanism story is measurably false (layer 0 is already global, local receptive field 1792 tokens ≥ MAX_LEN 512, and Mode 5 is cross-**chunk** and unreachable under max-over-windows serving), the campaign's own in-domain → blind slope predicts ~+0.001-0.005 against a +0.010 admit bar, and the 12 zero-output copies may stay functionally inert over 274M tokens in one epoch; ~33 GPU-h including the paired subsample probe. It remains the **only shippable within-family capacity instrument** in the field and is the correct arm to revive if item 9 (parameter budget) is opened.

**Residual, unranked (M3):** train-time evidence-offset augmentation at constant row count - proposed independently by two skeptics as the executable salvage of two killed candidates. Not a field survivor; needs its own registration and pre-gate.

---

## 7. Failure modes left UNADDRESSED by the top 5

**FM3 - weak-subset floor, on its two largest pools.** pubmedqa (0.5741 under the H108 incumbent; +0.18 headroom; 25% of the recorded residual as Mode 2) has **no** surviving lane. Its hedged-inference register lane was killed as a binding refused-list resubmission, and the record already states that every lever class the campaign owns (windowing exact +0.0000 on all draws, aggregation closed by oracle bound, span supervision −0.0654, discourse exclusion −0.0043) has failed on it. hagrid (0.6477; +0.13 headroom; the only subset where windowing is negative on every one of five draws) has never had a targeted lane and still does not. R12-H121 touches both only as secondary targets. Together these are 46% of the weak-subset headroom.

**FM3/Mode 5 - hotpotqa multi-hop composition.** Both attacks are gone: BRIDGE-COMPOSE killed (the frozen read never presents two hops in one window, and no training lane can teach a relation the read cannot present), depth-upscale below the cut. The structural fact stands unaddressed - max-over-chunks is an OR and a conjunctive claim cannot be confirmed from one pair. The one read-side amendment the literature suggests (union premises over the top-2 windows) was never proposed as a candidate.

**FM5 - capacity competition.** The only true capacity instruments are both out: ModernBERT-large killed on attribution invalidity, depth-upscale ranked 9. R12-H122's skeptic explicitly struck its capacity framing (1,792 parameters is not a capacity claim). The standing tension between R7-H50's "the task IS capacity-limited" and R8's "NOT architecture-limited" is therefore **unresolved at the current operating point**, and the 15% of residual attributed to true model-class capability (Mode 3 numeric derivation) remains blocked on the parameter budget.

**FM2 - OOD functional divergence.** The only direct attack (sharpness-aware minimization) was killed, correctly, on a mechanism contradicted by H118's own barrier-free interpolation. R12-H120 attacks the terminal-noise component as a secondary effect at best. The core finding - that same-recipe draws implement different functions on out-of-domain data while agreeing in-domain - has no live intervention.

**FM4 - small-effect adjudication, at the round level.** The top 5 are each individually adjudicable, but **nothing in the top 5 lowers the adjudication bar for future lanes.** The one candidate whose secondary claim was variance reduction (train under the min, attacking R8-H100's measured 4x noise amplification) is ranked 8 and blocked. Since the record's remaining un-banked levers are all +0.002 to +0.005 and unresolvable at ±0.03 single-subset noise, a round that does not reduce variance leaves the portfolio strategy unavailable.

**Structural, un-owned.** Whether part of the +0.0369 gap is unreachable by construction was never registered by any angle. The evidence pack names it the highest information-per-GPU-hour question in the record: what mean AUC would a perfect per-(sentence, window) entailment oracle achieve under RAGBench's response-level adherence labels? If that ceiling is materially below 0.80, the feasibility of 0.74 and the arithmetic of every remaining lever change. Zero GPU, no arena peeking beyond the licensed annotation read. See item 10.

---

## 8. Awaiting author - decisions that are yours alone

1. **Incumbent pin.** Is the R12 baseline the clean 0.7031 (H105 pair) or clean+H108 0.70496? No "clean + H108" control checkpoint exists - the in-flight DR lane trains `H108.public_train()` only, with 4 DR groups on top of the 12 public and an admission bar of 0.7031. Every per-subset bar in R12-H121, H122 and H123 changes with this ruling, and R12-H121's registered finqa target is already met by one candidate control and not the other. Sub-decision: re-pin after the DR/H117 verdict, or freeze the incumbent at registration and refuse to re-baseline mid-round.

2. **Legality - serving-wrapper text preprocessing.** Does an idempotent, subset-blind text transform applied before tokenization sit **inside** the frozen-read boundary (R8-H101 precedent: serving-side input construction changed on frozen weights, and that lever is banked)? If inside → R12-H119 proceeds as written. If outside → R12-H119 converts to a training-side augmentation and folds into the rank-6 tabular lane.

3. **Legality - fitted parameters in the read.** Does "no per-subset calibration, no test-time peeking" extend to "no fitted parameters in the read at all"? This governs whether the learned-aggregation-over-the-score-distribution line (SummaC-Conv shape, fit on RAGTruth-train and frozen before the arena) is closed on **protocol** or is still open on evidence. It was not proposed by any angle and cannot be adjudicated by a judge.

4. **Protocol - arena text for contamination gates.** May arena **document text** (no labels) be read to build a 13-gram exclusion blocklist for training data? Raised against a killed candidate but recurs for every future external-corpus lane.

5. **Protocol - arena annotations as a diagnostic.** R12-H121's Gate A requires the arena's `sentence_support_information` / `all_utilized_sentence_keys`. Licensed precedent exists (the R8 architecture failure analysis read exactly these). Confirm it is recorded as ANALYSIS only, with no per-subset quantity from Gate A entering the lane's size, filter thresholds or per-source mix.

6. **Cross-angle conflict ruling.** Window-bag / max-over-windows training received PROCEED-AMENDED from one skeptic and KILL from another, on the same mechanism, with the KILL carrying measurements that invert the shared premise. Does the KILL bind the merged entity (closing the line), or does the merged entity survive at rank 7 carrying both amendment sets? I have ranked it 7 and not registered it; the closure decision is yours.

7. **Hardware / recipe contract.** GPU1 is the only card that can host a 33 GB training draw at batch 48 / MAX_LEN 512; GPU0 (24 GB) and GPU2 (32 GB) are read-and-gate cards. Confirm that all Round 12 training serializes on GPU1 behind the DR draws and R11-H117 - or authorize a batch-size / gradient-accumulation change on GPU2, which breaks the byte-identical-recipe contract and voids paired comparison against every prior draw.

8. **Round bar.** Hold 0.74 as the Round 12 bar, or register an interim bar (0.72 was the evidence pack's suggestion)? The honest arithmetic: the top 5 sum to ≈ +0.031 under maximally optimistic additivity and zero displacement, against a required +0.0369, and the FM1 record predicts trading rather than adding.

9. **Parameter budget.** The record's recorded escalation path for Mode 3 (numeric derivation - the 15% true-capability residual) is a decoder-class scorer, parked at 595.8M on the sub-400M deliverable budget, **on budget grounds, not on evidence**. Reopening it is yours alone, and it is the only route in the log to an effect larger than +0.02.

10. **Register the label-ceiling diagnostic?** Zero GPU, no new hypothesis class: estimate what mean AUC a perfect per-(sentence, window) entailment oracle would achieve under RAGBench's response-level adherence labels. If materially below 0.80, the 0.74 target's feasibility and every remaining lever's arithmetic change. Not proposed by any angle; needs your word to occupy a slot.

11. **R12-H120's host draw.** EMA must ride a queued draw with EMA and final arms from the same run. Which - DR draw 2, or an R11-H117 arm? The choice determines the base the +0.005 is measured against.
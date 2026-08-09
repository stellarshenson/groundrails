# R15 LENS 2 - objective-side repair beyond H134

**Role**: objective-side lens for the R15 repair of the measured numeric-derivation defect. Output: four candidates - two amendments to registered blocks, one new hypothesis, one cheap staged arm - each with a pre-arm killgate, a ceiling-blind bar and a cost.

**Discipline**: CPU only, Polars only, zero GPU spent in this session, zero arena quantity in any threshold. Every number marked **[L2]** was measured here from `tmp/R14_E6_mix.parquet` (the byte-exact 685,670-row mix reconstruction) and `R10-H108_pairs.parquet`, banked in `R15_L2_weight_arith.json`, reproducible by `tmp/R15_L2_weights.py` in ~12 minutes. Everything else is quoted from the banked probes and gate JSONs as given.

---

## 0. The finding that organises this lens

**No objective term can invent the missing supervision, and the probes now say so in three independent ways.** P4's Instrument B decodes log-magnitude off the frozen trunk's `[CLS]` at **R² = 0.9987** and solves "is X greater than Y" at **accuracy 1.000 from 200 training examples**, with three controls holding (permuted labels 0.508, 200 rows 0.997, ridge λ raised 10⁴× 0.985), while the shipped task head reads **AUROC 0.5230** on the same comparison over a 30-character table (`R15_probe_P4_numeracy.md` §1). The direction exists in the representation and the readout has zero weight on it. That is a supervision gap, not a loss-function gap: nothing in a re-weighted or re-shaped BCE creates rows whose label depends on that direction.

**What is left for an objective-side lens is therefore exactly three jobs**, and all four candidates below do one of them:

1. **Gradient mass** - make the contradicting supervision that R14-A4 will build carry the weight its mechanism needs, without paying for it in the one resource E6 says is scarce (documents). L2-C1.
2. **Nuisance removal** - stop the competing prior from being re-learned faster than the new supervision installs. L2-C2 and L2-C3.
3. **Delivery cost** - buy the first read of the mechanism for a fraction of the registered spend. L2-C4.

**No input transformation is proposed anywhere in this file**, so the H119 verdict is engaged only to record that it is not at issue: nothing here touches `src/groundrails/semantic_ov.py`, `MAX_LEN`, the windowing, or any pre-tokenization text handling, and no candidate changes the bytes the deployed function receives. Sections 6 and 7 engage H117, H95/H96, the GroupDRO-curriculum line and the forced-subset-balance line explicitly, because three of the four candidates sit near one of them.

**The one thing this lens does not propose, stated up front.** No auxiliary head. R14-A4 amendment (i) declined L2-C2's training-only 2-way head as a second and third variable; head fusion and token-head transfer are closed lines. P4's Instrument B does **not** license reopening them - it says the direction the head is missing is linearly available at the same `[CLS]` the shipped head already reads, so the reachable repair is rows that make the existing head weight it, not a second head that reads it in parallel and is then discarded at serving.

---

## 1. The measurements this lens adds [L2]

All over the pooled training object the gradient actually sees: the clean mix plus the admitted H108 lane, 746,854 rows.

| quantity | value |
|---|---:|
| pooled rows | 746,854 |
| rows asserting a claim numeral absent from their evidence | 246,544 |
| ... of which label 1 (**absent-positive**) | **87,933** |
| ... of which label 0 (absent-negative) | 158,611 |
| pooled P(label 0 \| absent) | **0.6433** |
| importance weight on absent-positives giving pooled P(0 \| absent) = 0.5000 | **w\* = 1.8038** |
| row-equivalents that weight adds | 70,678 (**+9.46%** of the pool) |
| largest resulting DANN group-share shift | **0.923 pp** (vitaminc 0.4963 → 0.5055) |

**Absent-positive inventory by group** - the rows the shortcut is wrong about, which already exist and are already labelled:

| group | rows | absent rows | absent-positive |
|---|---:|---:|---:|
| vitaminc | 370,653 | 117,861 | **53,023** |
| tabfact | 92,585 | 11,312 | 5,218 |
| halueval | 40,000 | 5,015 | 2,285 |
| ragtruth (8 languages, combined) | 120,720 | 55,732 | 24,009 |
| psiloqa | 61,712 | 21,008 | **1,019** |
| H108 lane (admitted) | 61,184 | 35,616 | 2,379 |

**The two nuisance features are not the same feature.** Per (DANN group, label) cell, the correlation between the binary absent-numeral flag and R14-A5's claim digit fraction is **mean r = 0.2837** over 24 cells (median 0.2631, max 0.6948), i.e. **mean shared variance r² = 0.0973** - A5's registered term can reach at most a tenth of the shortcut's carrier.

**And inside the stratum that matters, A5's feature is label-flat**:

| stratum | n | mean claim digit fraction |
|---|---:|---:|
| label 0, number absent | 125,374 | 0.05246 |
| label 1, number absent | 85,554 | 0.05127 |
| label 0, all numbers present | 229,890 | 0.01449 |
| label 1, all numbers present | 244,852 | 0.02299 |

Within absent rows the label-conditional digit-density contrast is **0.00119** (2.3% relative) while the label rate itself swings 0.5944 against 0.4056. Within present rows the contrast is **-0.00850** and points the **other way** - positives are 59% digit-denser than negatives. The whole label-conditional digit-density signal in the mix lives in the present-number stratum with the sign opposite to the shortcut, which is a data-side account of the sign tension already recorded against R14-A5 (in-domain partial r **+0.07307**, per-label r **+0.27716** on label 0 against **-0.0364** on label 1, `R14_gate_H134_partialr.json`).

**Loss weighting is one line in the trainer and does not change the effective learning rate.** `DR_lane_trainer.py:300-302` already computes the task loss as `(lossv * keep).sum() / keep.sum()` with `keep` a 0/1 mask; substituting a float weight vector keeps the normalisation, so the total task-gradient scale is preserved and only the within-batch composition moves. At 11.77% absent-positive prevalence, a 48-row batch carries 5.65 up-weighted rows in expectation - no degenerate batches.

---

## 2. L2-C1 - AMENDMENT to R14-A4: per-row loss weight on the parity lane

**kind**: amendment | **cost**: ~2 GPU-h probe + ~13 GPU-h arm, conditional (0 if A4 clears at weight 1)

**Claim** - Because P3 measures the registered 50,000-row lane moving pooled P(0 | absent) only 0.6433 → 0.6192 (a 3.7% relative reduction, `R15_probe_P3_signal.md` §6) and measures the row count that would reach the dilution branch (200,000 rows) as costing 12.14 rows per document against a mix mean of 2.01 - worse document diversity than any group in the mix, which E6's verdict forbids - while **pooled purity is a function of row-equivalents, not rows** [L2], giving a 50,000-row lane at per-row BCE weight 4.0 a pooled P(0 | absent) of **0.5791**, identical to the 200,000-row branch, at the cap-2 build's **3.03 rows per document** and unchanged GPU time, setting the lane's per-row loss weight to a pre-registered **w = 4.0** will deliver A4's finqa movement at A4's diversity, while the arena mean holds and the anti-gaming clause does not fire.

**Weighted-dilution table** [L2] - pooled P(0 | absent), pool = mix + H108:

| lane rows | w = 1 | w = 2 | w = 3 | **w = 4** | w = 6 | w = 8 |
|---:|---:|---:|---:|---:|---:|---:|
| 25,000 | 0.6301 | 0.6192 | 0.6099 | 0.6020 | 0.5891 | 0.5791 |
| **50,000** | 0.6192 | 0.6020 | 0.5891 | **0.5791** | 0.5647 | 0.5547 |

**Killgate** (pre-arm, cheap):
- **Free, CPU, already run** - the weighted-purity arithmetic above must reach ≤ 0.58 at w ≤ 4 with the build's realised lane size. Measured 0.5791 at w = 4. **KILL** if a smaller realised lane pushes the required weight above 4.0.
- **Free, CPU, build manifest** - realised weighted lane share of total BCE mass must stay ≤ 25%. At 50,000 rows and w = 4 it is 200,000 / 946,854 = **21.1%**. **KILL above 25%** - past that the lane is a co-primary corpus, not a lane, and the arm is measuring a different object.
- **~2 GPU-h, in-domain only, no arena** - H117-precedent subsample probe: 150,000-row subsample carrying the lane at w ∈ {1, 4}, paired seed. **KILL** if `gold_full` at w = 4 falls more than 0.010 below the w = 1 subsample control (calibration damage, the exact clause H117's probe carried).
- **VOID clause** - the arm is void unless A4 draw 1 reads finqa ≥ its paired control + 0.010. A flat null at w = 1 is a lane with no signal to amplify; amplifying it is not a hypothesis.

**Bar** (ceiling-blind, paired) - **PRIMARY**: finqa 2-draw mean ≥ **0.6933** (A4's registered bar) **and** ≥ the A4 w = 1 arm + 0.020, sign agreement on both H126-paired draws. **ANTI-GAMING (binding)**: in-domain held-out H108-style present-value near-miss AUC must not fall below the clean-recipe value; **and** held-out scale/unit AUROC on P1's quad probe ≥ 0.80 (P1 amendment #3 - a lane that has taught "absent implies supported" loses the one numeric competence the model has, measured today at 0.8755, and this instrument is arena-free and costs 0.05 GPU-h). **CONFOUND (binding)**: finqa gain survives log mean-sentence-length residualization with the same sign at ≥ 50% of magnitude. **HOLD**: arena mean ≥ 0.7031; no subset more than 0.06 below its paired control and none < 0.55; pubmedqa ≥ 0.5463; `gold_full` ≥ 0.8414; RAGTruth non-EN ≥ 0.82. **KILL** at finqa below the A4 w = 1 arm on either draw, or arena mean < 0.6971.

**Why this is not "just train longer on the lane"** - it is not duplication in document space. Duplication at w = 4 would need 200,000 rows over the same 16,476 tables at cap 6, which P3 rejects on E6 grounds; the weight buys the gradient mass without buying the repetition. The two are equivalent in expected gradient and inequivalent in every diversity statistic E6 measures.

**Honest risks, declared** - (1) the weight multiplies construction artefacts by the same factor, so P2-B's build KILL (AUROC-from-token-length > 0.55), P2-D's token-budget assertion, P1's byte-identical-template clause and P3's N7 ≤ 10% cap all become load-bearing rather than prudent; the arm must not launch until the build manifest reports all four. (2) A synthetic Wikipedia-table lane at 21.1% of the task gradient is a real register shift; the arena-mean HOLD is the instrument and it is not a formality. (3) w = 4.0 is **frozen in writing before the build** and is never swept - a swept weight is a second variable and would put the arm outside the registered protocol.

---

## 3. L2-C2 - NEW: absent-positive importance weighting (zero-build shortcut neutralisation)

**kind**: new-hypothesis | **cost**: ~0.2 GPU-h killgate + ~12 GPU-h (2 paired draws); zero build, zero new rows, zero new documents

**Claim** - Because the pooled training object makes "the claim asserts a number absent from the evidence" a 0.6433 predictor of label 0 over 246,544 rows while **87,933 rows that already exist and are already labelled contradict it** [L2], and because a per-row BCE weight of **w\* = 1.8038** on exactly those rows drives pooled P(label 0 | absent) to **0.5000** - not 0.6192, which is all the registered 50,000-row lane achieves - for **+9.46% of row-equivalents, no new data, no new documents and a largest DANN group-share shift of 0.923 pp** [L2], up-weighting the absent-positive stratum will lift finqa to a 2-draw mean ≥ **0.6633** (+0.030) with sign agreement, while the near-miss anti-gaming AUC does not fall and the arena mean holds ≥ 0.7031.

**Mechanism, stated precisely, and what it is not** - this is a **calibration repair of the absent-number score region**, not an arithmetic repair. E4 measures finqa's absent-number sentences at score 0.487 against 0.641 for all-numbers-present sentences while RAGBench-finqa gold-supports 75.7% of them; P1 measures the same shape in tatqa at 0.2888 against 0.5441 on deciding sentences, and measures the bare derived assertion - one absent numeral, no operands quoted - as the worst-scoring shape in the register at 0.3059. Because the shipped read takes the min over sentences, any response containing one derived quantity has its response score set by that sentence, so finqa's ranking is substantially a function of how many derived numbers a response contains rather than of whether it is grounded. Moving the region's mean fixes that; it does **not** order correct against wrong derivations - P1 measures that ordering at chance across ten types and C2 predicts it stays at chance. **C2 and A4 are complements, not substitutes**: C2 shifts the region, A4 orders within it. C2 costs 1/13 of A4's build and is available now.

**Killgate**:
- **Free, CPU, already run** [L2] - the stratum must exist at scale and must not be one corpus. Measured: 87,933 rows, 60.3% vitaminc, present in all 12 groups plus the H108 lane. **KILL** below 40,000 rows or above 80% in a single group. Passes at 60.3%.
- **~0.2 GPU-h, frozen H105 draw 1, in-domain, no arena** - the shortcut must be **expressed on the rows being up-weighted**: score a fixed-seed 4,000-row sample of absent-positives against a matched 4,000-row sample of present-positives from the same groups. **KILL if mean(present-positive) - mean(absent-positive) < 0.05** - the function does not penalise these rows and there is nothing to correct. Report the per-group breakdown; it is also the arm's own baseline.
- **Direction is chosen on measurement, not preference** [L2] - the mirror intervention, down-weighting absent-negatives at w = 0.5544, removes 70,678 row-equivalents of which **33,237 are H108 lane negatives**, gutting the campaign's only replicated finqa lever (+0.0849 pair mean). The up-weight direction adds mass instead of removing it and leaves every H108 negative at full weight.

**Bar** - **PRIMARY**: finqa 2-draw mean ≥ **0.6633** (+0.030 over the 0.6333 paired control), sign agreement on both H126-paired draws. **ANTI-GAMING (binding, and this is the load-bearing clause for this candidate)**: C2 pushes directly on "absent implies supported", so (a) in-domain held-out H108-style present-value near-miss AUC must not fall below the clean-recipe value, (b) held-out scale/unit AUROC on P1's quad probe ≥ **0.80**, and (c) on the banked 2,000 H133 triples, **mean score(c) (wrong-operand) must not rise by more than mean score(b) (correct derivation)** - a pure bias shift that lifts both equally is recorded as a calibration move and **not** as a mechanism win. **CONFOUND (binding)**: log-length residualization, ≥ 50% of magnitude, same sign. **HOLD**: arena mean ≥ 0.7031; no subset more than 0.06 below its paired control and none < 0.55; pubmedqa ≥ 0.5463; `gold_full` ≥ 0.8414; RAGTruth non-EN ≥ 0.82; **psiloqa-side hold** - report the in-domain hallucination-detection AUC on the psiloqa group, whose absent-positive rows number only 1,019 against 19,989 absent-negatives and which therefore pays the largest relative re-weighting. **KILL** at finqa negative on either draw, or arena mean < 0.6971, or any anti-gaming clause firing.

**Engaging the closed forced-subset-balance line** - R8-H95's forced balance equalised **DANN group** representation by stratified batching and worst-group weighting, and was refuted blind. C2 weights a **label-conditional claim-evidence property within the existing group mixture**; the incidental group-share consequence is measured at a maximum of **0.923 pp** [L2] (vitaminc +0.923, psiloqa -0.614, tabfact -0.559, H108 -0.474), which is two orders of magnitude short of balance and is reported in the result JSON so the claim is auditable rather than asserted. No `q`, no stratified sampler, no per-group objective.

**Legality, against R14-A5 amendment (i)** - A5's registered legality test is that the nuisance feature's sign must not be knowable only from arena statistics. C2's weight is derived entirely from the training mix, its target (0.5000) is a symmetry condition rather than a direction, and its independent justification stands with the arena deleted: **a grounding library that treats "this number is not literally in the text" as evidence of unsupportedness false-alarms on the commonest shape of numeric RAG answer.** The arena figures in this section (E4's 75.7%, 0.487 vs 0.641) are motivation and appear in no threshold.

**Honest risks** - (1) 60.3% of the stratum is VitaminC Wikipedia-revision claims, so the register carried toward finqa is not financial; this is the same register objection E6 raises against every legal corpus and it is not solved here. (2) Absent-positive rows are absent for many reasons - spelled-out numerals, paraphrase, coincidence - and only a minority are derivations; the lesson taught is the correct one at the level of the feature, but it is not arithmetic supervision, and the bar is set at +0.030 rather than A4's +0.060 for that reason. (3) Label noise in the stratum is not measured here; P3's shuffle control on VitaminC (real derivable rate 5.10% against a 2.58% coincidence floor) is a warning that any *derivation-specific* subselection of these rows would be half accident - which is why C2 weights the whole stratum by its label, never by a derivation detector.

---

## 4. L2-C3 - AMENDMENT to R14-A5 (H134): regress the nuisance the census actually measures

**kind**: amendment | **cost**: ~0.3 GPU-h killgate; **zero marginal GPU** - it rides A5's registered ~12 GPU-h arm as a feature substitution

**Claim** - Because A5 decorrelates the task logit against **claim digit fraction** while the defect this round exists to repair is carried by the **binary absent-numeral indicator**, and because the two features share only **mean r² = 0.0973** of variance across 24 (group, label) cells [L2] and the digit-fraction contrast **inside the absent stratum is 0.00119** against a label-rate swing of 0.5944 vs 0.4056 [L2], regressing the same `lambda_dec = 1.0` term against the binary absent indicator - either replacing digit fraction or entering as a second cell dimension - will decorrelate the feature the shortcut is measured on, while A5's registered co-primary bars and its `lambda_dec` stay exactly as written.

**The sign tension recorded against A5 is explained by this measurement.** A5 was licensed with a noted tension: in-domain partial r is **+0.07307** while the arena-finqa function is negative. [L2] locates it in the data. The mix's label-conditional digit-density contrast is **-0.00850 in the present-number stratum** (positives 59% digit-denser) and **+0.00119 in the absent stratum** - two strata with opposite signs and a 2.25:1 mass ratio (474,742 present rows against 210,928 absent) in favour of the one that has nothing to do with the shortcut. `R14_gate_H134_partialr.json`'s own per-label split says the same thing from the function side: r = **+0.27716** on label 0 (n = 210) against **-0.0364** on label 1 (n = 390). A term aimed at that composite regresses out a mixture; a term aimed at the binary indicator regresses out the thing the census counts.

**Two forms, and the recommendation** - (a) **substitution**: the cell feature becomes the binary absent indicator; (b) **augmentation**: two decorrelation terms, digit fraction and absent indicator, at `lambda_dec = 0.5` each so the total penalty scale is unchanged. **Recommend (a)**, on the no-slop principle and because (b) reintroduces the mixture the amendment exists to separate. Per P2-E, if the digit-density form is retained anywhere it is defined on **token** digit fraction, not character.

**Killgate** - re-run `R14_H134_partialr.py` unmodified except for the feature, on the identical 600-row RAGTruth EN test sample and the identical read (max over `M59.top_chunks`, pre-sigmoid logit of the argmax chunk). **KILL if |partial r| < 0.05** for the binary feature - the shortcut did not survive into the deployed function in-domain and a training-time term against it is unlicensed. Report both features' partial r side by side so the substitution is auditable. **~0.3 GPU-h**, arena-free, gold-free. The free CPU precursor is already satisfied: per-group P(label 0 | absent) ranges 0.3613 (ragtruth_cn) to **0.9515** (psiloqa) against a present-row 0.4842 [L2, P3], so clause 1 of A5's gate passes on the binary feature far more strongly than on digit density.

**Bar** - **A5's bars stand unchanged**: co-primary A finqa ≥ 0.6733; co-primary B pubmedqa ≥ 0.6063 and must not fall at all; confound clause; `gold_full` ≥ 0.8314; arena mean ≥ 0.6431. Two additions, both reporting clauses: **(i)** post-training in-cell correlation against the binary feature must fall by ≥ 50% from its pre-training value, else the term did not act and the run is **VOID** rather than refuted; **(ii)** report the digit-fraction correlation as well, so the arm records which feature moved.

**Disjointness declarations** - P3 §2 records that A4's digit-length parity rule delivers part of A5's mechanism free and **confounds A5 in one arm**. The binary form is confounded **less**, not more: inside the A4 lane every row is absent-bearing by construction, so the binary feature has zero within-lane variance and the term contributes zero gradient there - the substitution makes A5 and A4 orthogonal by construction on lane rows, which the digit-fraction form is not. If A5 and C2 (L2-C2) run together they are **not** disjoint - both act on the same (absent, label) cells by different mechanisms, one by re-weighting and one by decorrelation - so declare them mutually exclusive within an arm and order them C3-inside-A5 first, since it is the zero-marginal-cost one.

---

## 5. L2-C4 - NEW: staged lane fine-tune from the banked clean checkpoints

**kind**: new-hypothesis (delivery form) | **cost**: ~0.3 GPU-h killgate + ~4-5 GPU-h for both draws and reads, against A4's ~13 GPU-h per full arm

**Claim** - Because stage 1 of A4's recipe is **already on disk** (`models/R9-H105-mmbert-dann-clean` and `models/R9-H105-draw2`, the H126-paired clean draws whose banked pair mean is 0.70311 and finqa 0.6333), a short continuation on the A4 lane plus a matched clean-mix replay, under the **byte-identical objective** (BCE + DANN at the terminal `lam` 0.02, no ramp restart, no new head, batch 48, MAX_LEN 512) at a capped constant LR, will move finqa by ≥ +0.030 on both banked draws for roughly a third of one A4 draw's GPU time, while `gold_full` holds ≥ 0.8414 and the arena mean holds ≥ 0.7031.

**Engaging the H95/H96 curriculum refutation, explicitly, because this is the candidate that must** - R8-H96 loaded the R8-H95 GroupDRO-mastered trunk and **replaced the objective**: fresh DANN heads, GRL ramp restarted, lambda moved from the H95 regime to the H93 sweep winner 0.1241. It was refuted (blind min 0.6820 against a 0.7313 bar), and R8-H99 decomposed the deficit as **3/4 lambda (-0.0300, H99 vs H90 at single variable) and 1/4 trunk init (-0.0093, H96 vs H99)**; R8-H100 then measured run-to-run noise at ±0.03 and **demoted the -0.0093 trunk-init term to within-noise**, explicitly annotated in the log's noise amendment. Three consequences bind here and are stated so the entity is not confused with the refuted one:

1. **The refuted entity is an objective phase shift.** C4 changes no objective term, no lambda, no head, no ramp; only which rows the final steps see. The measured cause of H96's failure - lambda - is held byte-identical.
2. **The sequencing component of H96's evidence is null within noise**, not negative. There is no measurement in the record that a data-composition continuation under a fixed objective costs anything; H96 cannot be quoted as one.
3. **This is not the GroupDRO curriculum.** No worst-group weighting, no `q`, no stratified batching, no difficulty ordering inside the lane (P4-5's instruction to mix the strata is carried). The closed line is `GroupDRO curriculum`, and none of its machinery appears.

**What C4 buys that A4's from-scratch arm does not** - the first read of whether this supervision moves finqa **at all**, at ~1/6 of the cost, on the exact checkpoints A4's bar is defined against. It is also a candidate **deliverable form**: one model, under 400M, unchanged serving path, with a 2-GPU-h patch step instead of a 13-GPU-h retrain - which matters if A4 needs iteration on its build (P1, P2 and P3 between them impose eleven binding construction clauses that will not all be right first time).

**Design, pre-registered before any run** - stage 2 = 50,000 lane rows + 50,000 replay rows sampled from the clean mix at the mix's own group proportions (replay is what makes this a continuation rather than a domain shift), flat permutation, batch 48 → 2,084 steps ≈ 1/7 of an epoch; constant LR **2e-6** (0.2× the recipe peak) with 100 warmup steps and no cosine tail; `lam` pinned at 0.02 with the GRL ramp **not** restarted; DANN group tags on lane rows assigned per A4's build. Every one of these constants is frozen in writing before the first run and none is swept.

**Killgate** - **~0.3 GPU-h, in-domain, arena-free**: run 300 steps of **replay only, no lane rows** at the chosen LR from `R9-H105-mmbert-dann-clean` and read `gold_full` and RAGTruth EN. **KILL if `gold_full` falls more than 0.005** - continuing training at this LR is destructive by itself and the arm would measure forgetting, not the lane. This gate also fixes the LR: if it fails, halve the LR once and re-run the gate; a second failure closes the candidate.

**Bar** - **PRIMARY**: finqa ≥ its own banked draw value + **0.030** on **both** banked draws (paired by construction - the same weights, the same seed lineage), sign agreement. **MECHANISM READ (pre-registered before any draw, per P4-5)**: re-read the banked 2,000 H133 triples and P1's ten-type quad set. **Comparison and operand-binding families above 0.65 while ratio, product and pct_change stay below 0.60 confirms the mechanism even if the finqa primary misses**, and is recorded as a mechanism win with a register-transfer failure rather than a kill; **division-type families moving as much as comparison is a construction-artefact warning, not a success**. **ANTI-GAMING (binding)**: present-value near-miss AUC not below clean; scale/unit AUROC ≥ 0.80. **CONFOUND (binding)**: log-length residualization. **HOLD**: arena mean ≥ 0.7031; `gold_full` ≥ 0.8414; no subset more than 0.06 below its own banked value and none < 0.55; pubmedqa ≥ 0.5463. **KILL** at finqa < +0.010 on both draws **as a reading about the staged form only** - it does not kill A4, whose from-scratch arm mixes the lane through the entire schedule at full LR, and the registration must say so in writing before the run so a null is not later spent as an A4 refutation.

**Ruling-7 breach, flagged not smuggled** - author ruling 7 pins the byte-identical recipe (batch 48, MAX_LEN 512, one epoch, OneCycleLR). C4 keeps batch and MAX_LEN and breaks the schedule: it adds a second, shorter phase at constant LR. That is a breach of the same class as A2 Stage 2 and requires an explicit author amendment before launch. It is cheaper than A2 Stage 2 by an order of magnitude and it is the only breach in this file.

---

## 6. Interaction, ordering and what to spend first

| id | kind | first-decision cost | full cost | conditional on |
|---|---|---|---|---|
| **L2-C3** | amendment to A5 | 0.3 GPU-h | 0 marginal (rides A5) | A5 running at all |
| **L2-C4** | new, staged form | 0.3 GPU-h | ~4-5 GPU-h | A4's build existing (CPU) |
| **L2-C2** | new, zero-build | 0.2 GPU-h | ~12 GPU-h | nothing - runnable today |
| **L2-C1** | amendment to A4 | 0 (CPU) + 2 GPU-h probe | ~13 GPU-h | A4 draw 1 finqa ≥ control + 0.010 |

**Spend order, on cost-to-first-decision**: C3 (0.3, and it changes A5's arm for free), then C4's forgetting gate (0.3, and it unlocks a 4-GPU-h read of the entire A4 mechanism), then C2's expression gate (0.2). C1 waits on A4's own draw 1 by construction.

**Mutual exclusivity, declared**:
- **C2 and C3 are not disjoint** - both act on the (absent, label) cells. Run C3-inside-A5 first; if A5 admits, C2's marginal claim must be re-argued against the decorrelated function rather than the shipped one.
- **C1 and C4 both scale the lane's influence**, one by loss weight inside a full run and one by concentration in a short phase. They are alternative deliveries of the same mechanism; running both in one arm is two variables. If C4 admits, C1's question becomes "does the from-scratch arm need the weight" and is answered by A4's own draw 1.
- **C2 and A4 compose** and are the only pair in this file that should be considered for a combined arm - and only after both have been read alone, because C2 alone predicts the H133 AUROC stays at chance and A4 alone predicts it moves; a combined arm cannot attribute either.

---

## 7. Closed lines engaged

- **H117, margin/contrastive pair loss - REFUTED, and nothing here is pairwise.** C1 and C2 are per-row scalar weights inside the existing `BCEWithLogitsLoss` reduction; every row carries its own absolute label and enters BCE. Verdict A's measured cause of the finqa collapse - 13,898 of 30,369 lane rows trained with `label=-1, bce_mask=True`, i.e. **no absolute target at all** - is structurally impossible under any candidate in this file: no candidate masks a row out of BCE, and no candidate introduces a term whose gradient depends on another row's score. The absolute score comparability the windowed decomposed-min read requires is preserved by construction.
- **H95/H96 GroupDRO curriculum - REFUTED**, engaged in full in §5. C4 changes data composition under a fixed objective; H96 changed the objective. The trunk-init component of H96's deficit is -0.0093 and was demoted to within-noise by H100.
- **Forced subset balance - REFUTED**, engaged in §3. C2's incidental group-share shift is measured at 0.923 pp maximum [L2] and is reported in the result JSON.
- **H119 read-time numeric canonicalization - REFUTED, not at issue.** No candidate applies any transformation to any input at any time. `semantic_ov.py` is untouched, `MAX_LEN` is untouched, the windowing is untouched, and the bytes the deployed function receives are byte-identical to today's under every candidate here.
- **Head fusion and token-head transfer - CLOSED**, and no auxiliary head is proposed; §0 gives the reason P4's Instrument B does not license reopening them.
- **Weight averaging - CLOSED**; C4 is a sequential continuation of one checkpoint, not an average of two.

---

## 8. Falsifiers

Each is a claim this lens makes that can be wrong, with its price:

- **C1's equivalence claim** - "pooled purity depends on row-equivalents, not rows" is arithmetic about the shortcut statistic, not about the optimizer. It is falsified if the w = 4 arm's finqa movement is materially smaller than a 200,000-row w = 1 lane's. That comparison is not affordable (P3 prices the 200k build at 12.14 rows/doc and the arm at roughly quadruple GPU), so the honest statement is that **C1's mechanism is untested at the optimizer level** and its bar is a paired contrast against A4 w = 1, which is affordable.
- **C2's calibration claim** - it predicts finqa moves while AUROC(b vs c) on the H133 triples stays within ±0.02 of 0.4924. If AUROC(b vs c) rises materially, re-weighting taught ordering as well as level, which no mechanism in this file predicts, and the result must be re-attributed before it is banked. Free - the re-read is already registered.
- **C3's feature claim** - it predicts the binary feature's in-domain partial r exceeds the digit-fraction feature's 0.07307 on the same 600 rows. If it reads lower, the shortcut is expressed in the function through digit density after all, A5 as registered is the right instrument, and this amendment is withdrawn. **0.3 GPU-h, and it is the killgate.**
- **C4's sequencing claim** - it predicts that a data-composition continuation under a fixed objective does not forget. The 300-step replay-only gate falsifies the LR choice for 0.3 GPU-h before any lane row is seen.
- **The whole lens's premise** - §0 asserts the readout gap is a supervision gap. It is falsified if fitting a linear probe on the frozen trunk's `[CLS]` for the **grounding label** on P1's held-out quads reaches materially above the task head's 0.4924 AUROC(b vs c). P4's Instruments B and C give the harness; it costs ~0.1 GPU-h and would say the direction is reachable by re-fitting the head alone, which would re-price every candidate here. **This is the cheapest unspent measurement on the board and should be taken first.**

---

## 9. What this lens does not claim

- **It does not claim any candidate reaches the arena.** Three of four are finqa-primary with pre-registered ceiling-blind bars; the register's own arithmetic warns that finqa gains are not additive across blocks and must never be summed.
- **It sets no bar from an arena statistic.** Every threshold is either carried verbatim from R14's registered blocks or derived from the in-domain mix measurements in §1. The arena figures quoted (E4's 0.487 vs 0.641 and 75.7%, P1's taxonomy, P3's finqa register profile) motivate constructions and appear in no threshold.
- **It does not measure label reliability** in the absent-positive stratum that C2 up-weights. That is C2's largest unmeasured risk and it is stated in §3 rather than hidden.
- **It does not price C1's optimizer-level equivalence** (see §8) and says so rather than asserting it.
- **It proposes no new head, no new loss term beyond a per-row scalar weight and A5's already-registered decorrelation, and no input transformation.**

---

## 10. Reproduction

```bash
cd /home/lab/workspace/private/ai-assistants/groundrails
uv run python tmp/R15_L2_weights.py     # ~12 min, CPU, Polars, no GPU, no network
```

Inputs: `tmp/R14_E6_mix.parquet` (byte-exact 685,670-row mix reconstruction), `experiments/grounding-semantic/R10-H108_pairs.parquet`. Output: `experiments/grounding-semantic/R15_L2_weight_arith.json` - absent-positive inventory by DANN group, the w\* solution and its group-share consequences, the lane loss-weight dilution table, the per-cell absent-vs-digit-fraction correlations, and the digit-fraction-by-stratum table. No tracked artifact was modified and no model was loaded.

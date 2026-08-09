# R15 LENS 4 - capacity x derivation interaction

**Role**: capacity-interaction lens for the Round 15 repair of the measured numeric-derivation defect. Output: four candidates - three measurements and one amendment - each with a pre-arm killgate, a pre-registered ceiling-blind reading map and a cost. Total GPU ask across the whole lens: **~0.65 GPU-h, all on card 0 or card 2**.

**Discipline**: CPU only in this session, Polars only, **zero GPU spent here**, zero arena quantity in any threshold. One original measurement was taken in this session and every number carrying it is marked **[L4]**; it is banked in `R15_L4_truncation_census.json` and reproduces from `tmp/R15_L4_trunc.py` in ~90 seconds on CPU with no network. Everything else is quoted from the banked gate JSONs, the probe files and the canonical log as given.

**Card discipline, binding on every cost line below**: card 1 is carrying the live training ladder and is not available. Every read proposed here runs as `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0` or `=2`, is frozen-weights, held-out, arena-free and gold-free, and none exceeds 0.3 GPU-h.

---

## 0. The lens had to be re-formed, and it got sharper

Two author-level rulings landed after this lens was specified, and both remove one of its three named surfaces:

- **R14-H132 is PARKED.** The author's 2026-08-09 ruling retires capacity as the binding-constraint explanation - "it never was model capacity; we managed almost to top performance given the dataset" - grounded on the committee's 0.72067 against the faithful-oracle ceiling 0.7560, on E5's ensemble-beats-oracle-draw-picker reading, and on E6's 1,645x / 382x register-prevalence gaps. The mmBERT-small arm returns its ~12-14 GPU-h to the pool and leaves the ladder. **There is no small arm to read a derivation probe on**, so the lens's first named candidate is dead in the form it was written.
- **R13-H129 is REFUTED at draw 1, draw 2 unspent.** Student blind windowed mean **0.69709** against the clean control pair 0.7031, finqa -0.0087, `gold_full` 0.8387. So the teacher-derivation measurement is no longer a gate that could arrive before the student adjudicates - **the student has already adjudicated**.

Neither ruling closes the *question*; both change what answering it costs and what the answer is worth.

**The re-formed thesis.** `R15_probe_P4_numeracy.md` Instrument B locates the derivation-relevant substrate - log-magnitude decoded off the frozen trunk's `[CLS]` at **R² = 0.9987**, pairwise comparison at held-out accuracy **1.000 from 200 training examples**, three controls holding - in the **pretrained trunk**, while the shipped task head reads the same comparison as a grounding claim at **AUROC 0.5230** on a 30-character two-row table. If the missing competence is a *readout* of a substrate that pretraining already supplied, then two things follow and both are cheap to check:

1. **Capacity is answerable without training anything.** The substrate is a frozen-representation property, so the base-versus-small question is a ridge probe on two pretrained checkpoints that are already in the local Hugging Face cache - **no arm, no ladder slot, no revival of H132**. L4-C2.
2. **Every already-trained artifact in the campaign can be read on the same instrument for the price of a forward pass.** The anchor teacher, the refuted student and the admitted H108 lane draws are all on disk, and the H133 triples are banked with their claim text. L4-C1.

**And the third surface - MAX_LEN - is now answered, not proposed.** The lens asked whether the derivation lane needs 1024 to see full tables. The banked triples carry `score_a/score_b/score_c`, so the partition arithmetic runs on CPU for zero GPU, and it was run here [L4]:

**35.65% of the H133 probe's own claim-evidence pairs exceed MAX_LEN 512** - and on the 1,596 triples where **both** operand rows named in the claim survive the truncation, AUROC(correct vs wrong) reads **0.4919** against 0.4924 on the full 2,000. Truncation is **exonerated** as a cause of the derivation defect, measured rather than assumed. What the same partition does measure is that truncation costs the model **0.2179 of verbatim score** and **0.1519 of AUROC(verbatim vs correct-derived)** - it damages the one numeric function the model has, and none of the one it lacks. L4-C3 carries the residual 0.15 GPU-h read and the binding record entry.

| id | kind | one-line | GPU |
|---|---|---|---:|
| **L4-C1** | measurement | Anchor-teacher derivation read: is the campaign's highest measured function (0.72067) derivation-blind, and did distillation transmit anything the arena mean could not see | ~0.2 (card 0 or 2) |
| **L4-C2** | measurement | Numeric-substrate size-invariance: P4 Instrument B on **pretrained** mmBERT-small against base, zero training, H132 stays parked | ~0.3 (card 0 or 2) |
| **L4-C3** | measurement | Read-length x derivation: CPU partition **delivered in this session**; residual 1024 frozen re-read closes the H131 Stage-2 derivation argument | ~0.15 (card 0 or 2) |
| **L4-C4** | amendment | Derivation-competence reporting clause on H132's parked ship candidate and on the H129 serving-side ensemble decision | 0 |

No candidate proposes training. No candidate proposes an input transform, a loss term, a head, or a serving change. `src/groundrails/semantic_ov.py`, `MAX_LEN`, the 1,500-character window and the max-over-windows read are untouched by all four, and the deliverable stays one model under 400M with an unchanged serving contract.

---

## 1. The measurement taken here [L4]

**What it is.** A truncation census over the 2,000 banked `R14_H133_triples.parquet` rows, plus per-partition rescoring of the **already-banked** scores. Evidence is rebuilt from `data/external/datasets/dataset-tabfact.zip` by `table_id` exactly as `R14_H133_probe.build()` constructs it (`f"{caption}\n{table}"`, `\r\n` normalised, `#` mapped to ` | `, truncated at `CHUNK_MAX` 1,500 characters), tokenized with the shipped tokenizer from `models/R9-H105-mmbert-dann-clean`, and partitioned by whether the evidence survives the pair budget. No model weights were loaded and no GPU was touched.

**Token geometry of the probe's own evidence:**

| quantity | value |
|---|---:|
| triples matched to their table | 2,000 / 2,000 |
| evidence characters, mean | 918.87 |
| evidence tokens, mean / median / p95 / max | **437.46** / 343 / 867 / **1,089** |
| characters per token | **2.1005** |
| claim tokens, mean | 20.98 |
| pair tokens (claim + evidence + 3 specials), mean | 461.44 |
| **share of pairs exceeding MAX_LEN 512** | **35.65%** |
| mean retained fraction of the evidence at 512 | 0.9089 |
| share of pairs exceeding 1024 | **0.90%** |
| mean retained fraction at 1024 | 0.9996 |

The 35.65% independently near-replicates `R15_probe_P2_tokenizer.md` §8, which measured **34.93%** of 1,500-character TabFact windows over 512 on a differently drawn sample; the chars/token figure here is 2.1005 against P2's 2.2605, so the probe's own evidence is slightly denser than P2's corpus-wide read. The 0.90% residual at 1024 corroborates R14-A2's "1024 removes 99.99% of the truncation" from a second sample (A2 measured p99.9 = 953, max 1,029; this sample's max is 1,089).

**Operand-row visibility.** The claim template is `"The combined {col} of {ka} and {kb} is {v}."`, so the two rows the derivation is computed from are named in the claim and locatable in the serialized table by first cell. Both rows were located for **1,824 of 2,000 (91.2%)** triples; the 176 misses are key-matching failures (cells carrying a separator or non-canonical whitespace) and are excluded rather than guessed.

| partition | n | mean(a) verbatim | mean(b) correct | mean(c) wrong | **AUROC(b vs c)** | AUROC(a vs b) |
|---|---:|---:|---:|---:|---:|---:|
| all triples (banked anchor) | 2,000 | 0.90507 | 0.24189 | 0.24524 | **0.49239** | 0.96429 |
| evidence fits inside 512 | 1,287 | **0.98275** | 0.18434 | 0.18777 | **0.48836** | **0.99733** |
| evidence truncated at 512 | 713 | **0.76486** | 0.34579 | 0.34897 | **0.49165** | **0.84542** |
| both operand rows visible | 1,596 | 0.97915 | 0.21274 | 0.21511 | **0.49188** | 0.99484 |
| at least one operand row cut away | 228 | 0.63560 | 0.35600 | 0.36460 | **0.47580** | 0.74540 |

**Three readings, and they are the lens's evidential spine.**

- **Truncation is not the derivation defect.** On the cleanest possible subset - 1,596 triples where the model can see both rows it is being asked to add - the discrimination is **0.49188**, five ten-thousandths from the banked 0.49239 and on the wrong side of chance. On the 1,287 whose entire evidence fits, **0.48836**. R14-H133's licensing rests on a clean measurement, and the answer to "does the derivation lane need 1024 to see full tables" is **not for derivation, it does not**
- **Truncation is a real and large defect in the function that works.** Verbatim score falls **0.98275 → 0.76486** (-0.2179) and AUROC(verbatim vs correct-derived) falls **0.99733 → 0.84542** (-0.1519) between the fitting and truncated partitions; on the 228 triples with an operand row cut away, verbatim reads **0.63560** and AUROC(a vs b) **0.74540**. Literal presence - `R15_probe_P1_anatomy.md` §4's single working numeric instrument, the digit-prefix copy detector - degrades exactly in proportion to how much of the table the read discards. This prices `R15_probe_P2_tokenizer.md`'s P2-D amendment in the currency of the model's one competence rather than as build hygiene
- **The correct derivation scores *higher* when the model can see *less*.** mean(b) rises **0.18434 → 0.34579** from the fitting to the truncated partition (+0.1615), and mean(c) rises with it (+0.1612), leaving the ordering untouched. This is the copy-detector account read from a third direction: score is driven by how much confirmed-absent table the model has in front of it, not by whether the asserted value follows from it

**Honest limits of this measurement.** (i) The 512 budget is computed as `512 - 3 specials - claim tokens` and the survival test is index-based against the evidence token offsets; `longest_first` truncation removes from the longer member one token at a time, which for these rows (evidence 437 tokens against a 21-token claim) is the evidence tail, so the approximation is exact for every row where the claim is shorter than the retained evidence - all of them. (ii) 8.8% of triples could not have their operand rows located and are excluded from the visibility partitions; they remain in the truncation partitions. (iii) This is one checkpoint (H105 draw 1) and one derivation family (two-operand sum) - the type-uniformity that makes it general comes from P1's ten types and P4's five families, not from this table.

---

## 2. L4-C1 - ANCHOR-TEACHER DERIVATION READ

**kind**: measurement | **cost**: ~0.2 GPU-h, card 0 or card 2, frozen weights, banked instrument

**Claim** - Because the campaign's highest measured function is the **output-probability mean of two checkpoints of one recipe** (R9-H105 draws 1 and 2, blind windowed 0.72067, `R13_anchor_teacher_result.json`) whose in-domain soft targets agree at correlation **0.97754** with median disagreement **0.01248** and only 14.44% of rows disagreeing by ≥ 0.10 (`R13-H129_gate_result.json`), and because draw 1 reads AUROC(correct vs wrong-operand) = **0.4924** on the banked triples (`R14_gate_H133_probe.json`) with `R15_probe_P1_anatomy.md` §3 replicating chance across ten derivation types on 5,008 held-out quads and `R15_probe_P4_numeracy.md` §0 replicating it across five families at a second seed, **the committee will read AUROC(b vs c) within ±0.03 of 0.50 on the same 2,000 banked triples and the H129 student will read within ±0.03 of its teacher** - establishing that neither output-space ensembling nor distillation can supply derivation competence, and that the 0.72067 anchor, at ~95% of the faithful-oracle ceiling, is derivation-blind.

**What is read**, all frozen, all on the banked `R14_H133_triples.parquet` claim text, zero arena, zero gold, **no re-sampling** (re-drawing the triples would break comparability with the 0.4924 anchor):

| checkpoint | on disk | why it is in the read |
|---|---|---|
| `models/R9-H105-mmbert-dann-clean` | banked in the parquet as `score_a/b/c` | the 0.4924 anchor, **free** |
| `models/R9-H105-draw2` | yes | the committee's second member, and P1 §9's own registered checkpoint-specificity falsifier |
| **committee** = mean of the two members' probabilities | derived | **free CPU arithmetic** once draw 2 is scored - this is the 0.72067 object |
| `models/R13-H129-draw1` | yes | the refuted student: did distillation transmit derivation ordering the arena mean did not show |
| `models/R10-H108-lane-draw1` / `-draw2` | yes | the campaign's **only replicated finqa lever** (pair finqa 0.7182 vs clean 0.6333) - does the lever that moved finqa most carry any derivation ordering |

Four GPU scoring passes of 6,000 claim-evidence pairs each; the committee is arithmetic. Report per checkpoint: mean(a), mean(b), mean(c), AUROC(b vs c), AUROC(a vs b), and P1's scale/unit control AUROC on the banked quad parquet, so the **level** shift is separable from the **ordering** and every future arm inherits a per-checkpoint baseline.

**Killgate** (pre-arm, cheap):
- **Free, CPU, already run in this session** - all five checkpoints exist on disk under `models/` and the triples parquet carries `claim_a/b/c`, `v_correct`, `v_wrong` and the three banked scores. Verified. **NO-READ** if any checkpoint is missing rather than substituting a different one
- **~0.05 GPU-h, card 0 or 2** - score **draw 2 alone first**. **KILL the candidate if draw 2 reads AUROC(b vs c) > 0.60**: the defect is then checkpoint-specific rather than recipe-specific, the committee read is confounded by one competent member, and the finding belongs to P1 §9's falsifier line (which registers exactly this check) rather than to a capacity-interaction claim. This is the whole gate - it costs a quarter of the candidate and it is the one result that would invalidate the rest

**Bar** (pre-registered reading map, frozen in writing before any checkpoint is scored; ceiling-blind; this is a measurement, so the map *is* the bar):

- **committee AUROC(b vs c) ≤ 0.55** → **DERIVATION-BLIND ANCHOR**, recorded as a binding register entry: the 0.72067 advantage is variance cancellation off-distribution and carries no derivation competence, H129's null is *explained* rather than merely observed, and **any future proposal to distil derivation competence from a committee must first exhibit a teacher measured above 0.60 on this instrument**
- **committee ≥ 0.60 while both members ≤ 0.55** → output-space averaging **creates** a discrimination neither member has. That is a genuinely new fact, it re-prices the unspent H129 draw 2, and it re-opens the serving-side two-forward-pass mode on a *capability* argument rather than a mean argument. Recorded, not acted on - reviving H129 would be a fresh registration
- **student > draw 1 + 0.05** → distillation transmitted derivation ordering that the blind arena read could not see; recorded as a mechanism-transmission finding under an otherwise-refuted arm, and it sharpens FM2 rather than reversing the H129 verdict
- **H108 draws > 0.55** → the campaign's only replicated finqa lever carries derivation ordering, and **R14-A4's marginal claim over the H108 lane must be re-argued before A4 builds**. Predicted otherwise: P1 §3 attributes H108's fingerprint to the digit-copy detector (scale/unit 0.8755) and measures rounding - the shape H108's ~45k unit/period/scale negatives most directly taught - as the *most* penalised correct derivation of all ten types at 0.1763
- **HOLD, binding on the read itself**: banked triples only, no re-draw, no threshold set from any arena quantity, and the per-checkpoint mean(a) reported alongside so that a checkpoint scoring everything lower is not read as a discrimination change

**What makes this decisive rather than tidy.** The author's ruling that the binding constraint is data rests on the committee sitting at ~95% of the honestly measurable ceiling. If that committee is at chance on the defect Round 15 exists to repair, then the ceiling argument and the derivation argument are about **different capabilities**, and the record should say so in one measured line before any further ensemble or distillation spend is contemplated. Zero of the campaign's ensemble verdicts (H64, H88, H92, H97, H98, P-A, H104, the anchor teacher) has ever been read on a derivation instrument, because until R14-H133 there was none.

**Honest risks** - (1) The committee is an *output* mean, so a chance reading is close to arithmetically forced when both members are at chance and their scores correlate at 0.97754; the candidate's value is therefore mostly in the **student** and **H108** arms and in closing the ensemble line by measurement rather than by inference. That is stated here rather than discovered at adjudication, and it is why the cost is 0.2 GPU-h and not more. (2) The banked triples are a single derivation family; the generality is inherited from P1 and P4, not established here. (3) A null on every arm is the predicted outcome and is still worth the spend - it is the record entry that stops the next round proposing a distillation route to derivation competence.

---

## 3. L4-C2 - NUMERIC-SUBSTRATE SIZE-INVARIANCE (no training, H132 stays parked)

**kind**: measurement | **cost**: ~0.3 GPU-h standalone, **~0.1 marginal if L3-C4 runs first**; card 0 or card 2

**Claim** - Because the R14-H132 gate measured the two **pretrained, un-fine-tuned** trunks at best-layer grounding AUC **0.7199** (base) against **0.7032** (small), both peaking at layer 10, a gap of **+0.0167** on a fixed-seed 20,000-row public-mix slice (`R14_gate_H132_layerprobe.json`), and because `R15_probe_P4_numeracy.md` Instrument B locates the derivation-relevant substrate in the frozen trunk - log-magnitude interpolation R² **0.9987**, comparison accuracy **1.000**, with permuted-label, 200-row and λ×10⁴ controls at 0.508 / 0.997 / 0.985 - running the **identical ridge protocol on pretrained mmBERT-small** will find comparison accuracy ≥ 0.95 and interpolation R² ≥ 0.95, establishing that the substrate the R15 repair must teach the head to read is **not size-bearing between 42,188,928 and 110,330,112 parameters of compute-bearing stack**, at zero training and with the capacity arm parked.

**Why this is not a revival of H132.** H132's arm was two paired training draws of mmBERT-small on the clean mix, ~12-14 GPU-h, adjudicating CAPACITY LIVE against CAPACITY CLOSED. That arm is parked by the author's word and **nothing here asks for it back**. L4-C2 trains nothing, adjudicates no capacity verdict, and licenses no ladder slot. It reads two checkpoints that are already downloaded, on a probe that already exists, and it answers one narrow question the parking leaves open: **whether the parked 140.9M ship candidate would carry a derivation penalty if the author ever revives it on serving-cost grounds.** That question survives the ruling because the ruling was about capability at the arena mean, not about the numeric substrate.

**Availability, verified in this session** - `~/.cache/huggingface/hub` carries `models--jhu-clsp--mmBERT-base` and `models--jhu-clsp--mmBERT-small`. No network is needed. `R14_H132_layerprobe.py` already loads both by Hub id and `R15_P4_numeracy_probe.py` already implements Instrument B; the read is those two harnesses composed, with the probe input dimension moving 768 → 384 for the small arm.

**What is read**, per model, on the frozen trunk `[CLS]`, held-out, arena-free, gold-free:

1. **Magnitude, interpolation** - ridge on values 1-999, target log₁₀, held-out R²
2. **Comparison** - `"Alpha is X and Beta is Y"`, target `X > Y`, held-out accuracy and AUROC
3. **Magnitude, extrapolation** - train 1-999, test 10,000-99,999, **reported as a direction only** (P4 measured -46.1 and its own uncertainty register calls it a direction, not a coefficient)
4. **P4's three controls, re-run on the small arm** - permuted labels, 200-row training subset, ridge λ raised 10⁴× - because a 384-dimension probe on a deterministic target invites the same overfitting objection P4 pre-empted at 768 and the answer must not depend on which objection was pre-empted where

**Killgate**:
- **~0.1 GPU-h, card 0 or 2** - **protocol reproduction on the base arm first**. The base numbers must reproduce P4's within **0.01 R²** and **0.02 accuracy**. **NO-READ if they do not** - the harness has drifted between sessions and no cross-model comparison is admissible until it is repaired. This gate also supplies L3-C4's un-fine-tuned base baseline, so the two candidates share it
- **Free, CPU, already verified** - both Hub ids resolve from the local cache with no network. **NO-READ** rather than a download attempt if either is absent

**Bar** (pre-registered reading map, frozen before the small arm is run; **no bar on any training arm is set or moved by this candidate**):

- **small comparison ≥ 0.95 AND interpolation R² ≥ 0.95** → **SUBSTRATE SIZE-INVARIANT**: derivation competence is not a capacity question at this scale, the author's capacity retirement stands strengthened in the one place R15 could have contradicted it, and the parked 140.9M ship candidate carries **no a priori derivation penalty**
- **small comparison < 0.80 OR interpolation R² < 0.90** → **SUBSTRATE SIZE-DEPENDENT**: a live counter-fact to the capacity retirement, strictly confined to the numeric substrate and explicitly **not** a claim about the arena mean. Recorded as such, and L4-C4's ship-clause hold becomes load-bearing rather than prudent
- **anything between** → **UNRESOLVED**, reported with both numbers and no consequence claimed. Stated in advance so a middling reading is not narrated into a verdict
- **VOID** if any of P4's three controls fails on the small arm (permuted-label accuracy outside 0.45-0.55, or the 200-row subset reading more than 0.05 below the full-sample probe) - the probe is fitting capacity rather than structure and neither branch is admissible

**Disjointness with L3-C4, declared** - `R15_hypotheses_L3_input_representation.md` §6 owns the trained-checkpoint numeracy-retention HOLD and specifies an un-fine-tuned mmBERT-base baseline alongside H105 draws 1 and 2 and H108 draw 1. L4-C2 adds **exactly one arm** to that baseline - pretrained mmBERT-small - and adopts L3-C4's thresholds verbatim where they overlap (comparison ≥ 0.95 absolute, magnitude R² within 0.05 of baseline). If L3-C4 runs first, L4-C2's marginal cost is ~0.1 GPU-h and its killgate is already satisfied. **They must not be registered as two baselines.**

**Honest risks** - (1) A pretrained-trunk parity result does **not** establish that a fine-tuned small model would retain the substrate; the clean recipe's effect on the substrate is L3-C4's question and is not answered here. Any ship-candidate admission would need the retention read on the *trained* small checkpoint, which is why L4-C4 makes it a reporting clause rather than an inference. (2) Instrument B's synthetic sentences are not the serving register; the probe measures what the trunk encodes, not what it would use. (3) The whole candidate is contingent on the author ever reviving the ship clause; at 0.3 GPU-h - 0.1 marginal - it is priced as insurance and should be declined if the author says the 140.9M line is closed for good.

---

## 4. L4-C3 - READ-LENGTH x DERIVATION (CPU half delivered; 0.15 GPU-h residual)

**kind**: measurement | **cost**: CPU census **DONE in this session, 0 GPU**; residual 1024 re-read ~0.15 GPU-h, card 0 or card 2

**Claim** - Because **35.65%** of the H133 probe's own claim-evidence pairs exceed MAX_LEN 512 while AUROC(b vs c) on the 1,596 triples whose **both** operand rows survive that truncation reads **0.49188** against 0.49239 on the full 2,000, and on the 1,287 whose evidence fits entirely reads **0.48836** [L4] - so truncation is already exonerated as a cause of the derivation defect - but the same partition measures truncation costing **0.2179 of verbatim score** and **0.1519 of AUROC(a vs b)** [L4], **a frozen 1024 re-read of the banked triples will restore AUROC(a vs b) on the truncated partition by ≥ +0.05 while moving AUROC(b vs c) by less than +0.03**, which prices P2-D's build-time token budget in the currency of the model's one working competence and **closes the derivation argument for the blocked H131 Stage-2 amendment**.

**Why the residual read is still worth 0.15 GPU-h when the CPU half already answers the headline.** The CPU partition compares *different rows* - short tables against long ones - so a sceptic can attribute the whole gap to table complexity rather than to truncation. The 1024 re-read compares **the same 713 rows to themselves** under two read lengths, which is the only construction that isolates the read. It is also the exact reading H131 Stage 1 already ran on the arena (session ruling 14 permits the frozen 1024 read on a gate card; it is Stage-2 *training* that breaches ruling 7 and is blocked), so the instrument and its legality are both established.

**What is read** - the banked 2,000 triples, `models/R9-H105-mmbert-dann-clean`, frozen, identical construction, `max_length` 1024 instead of 512, everything else byte-identical to `R14_H133_probe.py`. Reported per partition (fits-512, truncated-at-512, both-operands-visible, operand-cut-away): mean(a), mean(b), mean(c), AUROC(b vs c), AUROC(a vs b), and the paired within-row deltas against the banked 512 scores.

**Killgate**:
- **Free, CPU, already run and reported in §1**. If the truncated partition had been under 15% or over 60% of the sample the comparison would have been degenerate; measured **35.65%**, so it passes. Recorded so the gate is auditable rather than retrospective
- **VOID clause, pre-registered** - if the 1024 re-read moves **AUROC(a vs b) on the truncated partition by less than +0.05**, the longer read is not restoring the verbatim function, the geometry is doing something other than adding coverage, and **no inference about derivation may be drawn from the same run**. Report and stop

**Bar** (pre-registered reading map, frozen before the re-read; ceiling-blind, in-domain only):

- **Δ AUROC(b vs c) at 1024 < +0.03** → **READ LENGTH IS NOT A DERIVATION LEVER**, recorded as a binding register entry with three consequences: (i) R14-H131 Stage 2 **gains no derivation argument** and must not be lobbied for on one when the author takes the ruling-7 amendment decision; (ii) `R15_probe_P2_tokenizer.md`'s P2-D token-budget amendment stays a **build-hygiene KILL** - binding, free, and correct - rather than a mechanism claim; (iii) **R14-A4 may build at a 512 token budget** without waiting on A2/H131, removing P2-D's "A4 must not be built before A2 Stage 1 reports" dependency in the direction that unblocks the lane
- **Δ AUROC(b vs c) ≥ +0.05** → **part of the defect is a READ defect**: P2-D becomes load-bearing rather than prudent, A4 must not build before it is applied, and H131 Stage 2 acquires its first mechanism-level argument. This branch is **not** predicted and would be a surprise worth its own registration
- **between** → report both numbers, claim neither branch
- **REPORT, not bar**: the verbatim restoration on the truncated partition, and the share of the 2,000 still exceeding 1024 (**0.90%**, mean retained evidence fraction **0.9996** [L4])

**Counter-indication carried honestly.** H131 Stage 1 measured the 1024 frozen read moving the arena mean nowhere (+0.0001 to -0.0034) and techqa **negative** on all four checkpoints (-0.0017 / -0.0164 / -0.0129 / -0.0235) - the licensing branch fired precisely because the fine-tuned model does not generalise past its 512 training length. So a 1024 *gain* on this in-domain instrument would contradict the arena reading and must be recorded as an in-domain-only fact about a synthetic table probe, never as evidence that the arena read should change. The two are different objects and the registration must say so before the run.

**What the CPU half already delivers to the lane builder, independent of the residual read** - the A4 lane's evidence budget is a **correctness** constraint, not a tuning choice, and its cost is now quantified on the model's working function rather than asserted: at 512, 35.65% of rows in a probe built exactly like the lane lose evidence, 12.5% of the located rows lose at least one operand row outright, and those rows' verbatim discrimination falls to **0.74540**. P2-D's per-row assertion that both operand rows survive inside the retained prefix is the right fix and it is free.

---

## 5. L4-C4 - DERIVATION-COMPETENCE CLAUSE ON THE PARKED CAPACITY OBJECTS

**kind**: amendment | **cost**: **0 GPU** - a registration clause riding reads proposed above

**Claim** - Because R14-H132's sole surviving clause is the **140.9M ship candidate**, admitted on pair mean ≥ 0.69311, `gold_full` ≥ 0.84, no subset < 0.55 and none more than 0.06 below control - the exact instrument set that `R14_synthesis.md` verdict A records as **blind** to the margin arm's -0.1020 finqa collapse (`gold_full` 0.8042 against control 0.8040, "the damage was invisible to every in-domain instrument the arm carried") and that carried the shipped 307M model through nine rounds without ever seeing a derivation defect P1 now measures uniform across ten types - **any smaller deliverable admitted under that clause would be admitted blind to derivation unless the clause is amended**; binding it to report AUROC(b vs c) on the banked triples, P1's scale/unit control and P4's comparison family, with a single non-regression floor already used elsewhere in the register, closes the blind spot at zero GPU and introduces no new threshold.

**Limb (a) - the ship clause.** Amend R14-H132's parked ship-candidate clause to require, alongside its existing arena holds, three **REPORT** quantities and one **HOLD**, all frozen-weights, in-domain, arena-free and gold-free:

| quantity | instrument | 307M reference | status |
|---|---|---|---|
| AUROC(correct vs wrong-operand) | banked `R14_H133_triples.parquet` | **0.4924** | REPORT |
| comparison family AUROC | P4 Instrument A | **0.5130** | REPORT |
| verbatim discrimination AUROC(a vs b) | banked triples | **0.9643** | REPORT |
| **scale/unit AUROC** | P1 §3 quad probe | **0.8755** | **HOLD ≥ 0.80** |

The scale/unit floor is not a new threshold: `R15_hypotheses_L2_objective.md` carries **≥ 0.80** as a binding anti-gaming clause on L2-C1 and L2-C2, `R15_probe_P1_anatomy.md` amendment #3 registers it as the lane's within-lane control, and `R15_probe_P4_numeracy.md` P4-5 registers it as a non-regression check. This amendment applies the **same** floor to a smaller deliverable, where it is doing the same job: a model that has lost the digit-prefix copy detector has traded away the one numeric competence the product has, and no arena mean will show it.

**Rationale, stated so it is not mistaken for ceremony.** The 140.9M candidate's revival path is explicitly **serving cost, not capability** (author's ruling). A serving-cost trade is exactly the situation in which a capability regression is most likely to be waved through, because the decision is framed as latency against a mean. Three report lines and one floor make the trade visible at the moment it is taken.

**Limb (b) - the serving-side ensemble decision.** The H129 verdict leaves output-space ensembling real but **serving-side only** - two forward passes, 2× cost, on the author's call - while both single-model routes (weight-space averaging, H118 and the H120 within-run EMA; distillation, H129) are closed. Amendment: **that call must be taken with L4-C1's committee derivation number in hand.** If the committee is derivation-blind (predicted), the honest price tag is *2× serving cost buys +0.01756 of arena mean and zero derivation competence*, against a Round 15 register whose entire purpose is derivation competence. If it is not blind, the price tag changes and the decision is different. The clause costs nothing and it prevents the most expensive serving decision on the board being taken on a mean alone.

**Killgate** - none, and none is appropriate: this is a registration clause with no compute. Its only precondition is that the reads it cites exist, and both are proposed above at ≤ 0.3 GPU-h each. If the author declines L4-C1 and L4-C2, limb (a) survives unchanged (the H133 triple read is 0.05 GPU-h on any single checkpoint) and limb (b) lapses.

**Bar** - **not a bar**. It adds three REPORT clauses and one HOLD that already exists elsewhere in the register at the same value. No primary is set, no existing bar is moved, and no arena quantity enters any threshold.

---

## 6. Interaction, ordering and what to spend first

| id | kind | first-decision cost | full cost | card | conditional on |
|---|---|---|---|---|---|
| **L4-C3** | measurement | **0** (CPU census done) | ~0.15 GPU-h | 0 or 2 | nothing - runnable today |
| **L4-C1** | measurement | 0.05 GPU-h (draw-2 gate) | ~0.2 GPU-h | 0 or 2 | nothing - runnable today |
| **L4-C2** | measurement | 0.1 GPU-h (base reproduction) | ~0.3 standalone / ~0.1 after L3-C4 | 0 or 2 | the author keeping the 140.9M line open |
| **L4-C4** | amendment | 0 | 0 | - | L4-C1 for limb (b) |

**Spend order, on decisiveness per GPU-hour**: **L4-C3 first** - the CPU half is delivered and the residual 0.15 GPU-h removes a dependency that currently blocks the A4 lane build (P2-D's "A4 must not be built before A2 Stage 1 reports"). **L4-C1 second** - its 0.05 GPU-h draw-2 gate is simultaneously P1 §9's registered checkpoint-specificity falsifier, so the first twentieth of a GPU-hour buys two registered questions. **L4-C2 third**, and only if the ship line is live. **L4-C4 is free and should be recorded at registration time**, not after.

**Disjointness and overlap, declared**:

- **L4-C2 x L3-C4** - shared baseline, single registration. L3-C4 owns the trained-checkpoint HOLD; L4-C2 adds the pretrained small arm and adopts L3-C4's thresholds. **Must not be registered as two baselines**
- **L4-C1 x P1 §9** - the draw-2 arm **is** P1's registered checkpoint-specificity falsifier ("if H105 draw 2 or an H108-lane checkpoint reads materially above chance on any type, the defect is checkpoint-specific rather than recipe-specific, ~0.2 GPU-h on the banked quad parquet"). Credit to P1; run it once and report it under both
- **L4-C1 x L2-C4 / P4-5** - both register a post-arm re-read of the banked triples on *trained* checkpoints. L4-C1 reads **already-trained** artifacts and establishes the per-checkpoint baseline those post-arm reads compare against. Complementary, not duplicative; the baseline should exist before the first R15 arm lands
- **L4-C3 x P2-D** - C3 measures what P2-D fixes. P2-D remains binding and free **whatever C3 reads**; C3 determines only whether it is hygiene or mechanism, and whether A4's build waits on H131
- **L4-C3 x R14-H131** - C3 reads Stage 1's geometry on an in-domain instrument. It **licenses nothing** about Stage 2 in either direction; the predicted branch explicitly *removes* a derivation argument from Stage 2 rather than supplying one
- **L4-C4 x every anti-gaming clause in the register** - the scale/unit ≥ 0.80 floor is the same floor at the same value; it is applied to a new object, not invented

**What this lens deliberately does not propose.** No training arm, no ladder slot, no revival of R14-H132, no revival of R13-H129, no lane, no loss term, no head, no input transform, no serving change. Four candidates, ~0.65 GPU-h, and the largest single item is 0.3.

---

## 7. Closed lines engaged

- **R14-H132 capacity arm - PARKED by author ruling, and stays parked.** L4-C2 trains nothing and adjudicates no capacity verdict; it reads two pretrained checkpoints already in the local cache. The ruling retired capacity *as the binding-constraint explanation for the arena mean*; the numeric-substrate question it leaves open is priced at 0.1-0.3 GPU-h and is answered without an arm. The ruling's own text names this lens's narrowed scope: "R15's capacity-interaction lens narrows to its distillation-transmission measurement, which is about what distillation CAN transmit, not about size" - L4-C1 is that measurement, and L4-C2 is offered strictly as ship-clause insurance with an explicit decline path
- **R13-H129 - REFUTED at draw 1, draw 2 unspent.** Nothing here re-registers it. L4-C1 reads the artifacts it left on disk and would only *re-price* draw 2 under a branch (committee ≥ 0.60 with both members ≤ 0.55) that is not predicted and that would require a fresh registration to act on
- **Weight-space averaging - CLOSED in both forms** (H118 cross-draw 0.69218, H120 within-run EMA at terminal update cosine 0.9378). L4-C1 reads the **output-space** committee, which is a different object, and it proposes no averaging of any weights
- **R12-H119 read-time numeric canonicalization - REFUTED, and not at issue.** No candidate applies any transformation to any input at any time. `src/groundrails/semantic_ov.py` is untouched, the 1,500-character window is untouched, and the bytes the deployed function receives are byte-identical to today's under all four. L4-C3 changes `max_length` **inside a diagnostic probe on frozen weights** - the same frozen read H131 Stage 1 already ran and session ruling 14 already permits - and explicitly declines to argue from it to any serving change
- **R11-H117 margin / contrastive pair loss - REFUTED, and nothing here is a loss.** No candidate touches the objective. Verdict A's failure mode (13,898 rows trained with `label=-1, bce_mask=True`) is structurally impossible in a set of frozen-weight reads
- **R8-H95/H96 GroupDRO curriculum, forced subset balance, head fusion, token-head transfer, training on RAGBench corpora** - none is touched. No candidate here adds a training row of any kind, from any corpus
- **Author ruling 7 (batch 48 / MAX_LEN 512, byte-identical recipe)** - unbreached by all four. L4-C3's 1024 is a frozen diagnostic read, not a training configuration; H131 Stage 2 remains the register's only ruling-7 breach and this lens **argues against** giving it a derivation justification it does not have

---

## 8. Falsifiers

Each is a claim this lens makes that can be wrong, with its price. Total falsification budget **~0.35 GPU-h**, card 0 or 2.

1. **The truncation-exoneration claim** (L4-C3's delivered half, and the load-bearing new fact). It rests on a between-rows partition of one checkpoint's banked scores. It is falsified by the within-rows 1024 re-read: if AUROC(b vs c) on the previously-truncated 713 rises by ≥ +0.05 against their own banked 512 scores, truncation was hiding derivation signal and §1's reading is wrong. **~0.15 GPU-h** - and it is L4-C3 itself, so the candidate is its own falsifier
2. **The derivation-blind-anchor claim** (L4-C1). Falsified if the committee reads AUROC(b vs c) ≥ 0.60 while both members sit at chance. **~0.1 GPU-h**, and it is the candidate's own read
3. **The recipe-not-checkpoint claim** (shared with P1 §9). Falsified if H105 draw 2 alone reads above 0.60. **~0.05 GPU-h**, and it is L4-C1's killgate - deliberately placed first so the cheapest disconfirmation runs before the rest of the spend
4. **The size-invariant-substrate claim** (L4-C2). Falsified if pretrained mmBERT-small reads comparison accuracy < 0.80 or interpolation R² < 0.90 under P4's protocol with its controls holding. **~0.2 GPU-h**, and it is the candidate's own read
5. **The blind-instrument-set premise** (L4-C4's whole rationale). It asserts that the ship clause's existing holds cannot see a derivation regression. It is falsified if a checkpoint is ever found whose `gold_full` and arena-mean holds move materially *with* its H133 triple AUROC. **Free** - L4-C1's five-checkpoint table is exactly the data that would show it, and if the two do co-move across five artifacts, L4-C4 is unnecessary and should be withdrawn

**Every candidate in this file is its own falsifier.** That is a property of a lens made entirely of measurements, and it is the reason the whole ask is 0.65 GPU-h.

---

## 9. What this lens does not claim

- **It does not claim any candidate moves finqa, or any arena number.** None of the four is a training arm; none carries a finqa primary; none sets or moves a registered bar. The register's finqa arithmetic is explicitly non-additive and nothing here adds to it
- **It sets no threshold from an arena statistic.** Every number above comes from held-out TabFact (`table_id`-disjoint from every training split), the banked gate JSONs, the local tokenizer, or the pretrained checkpoints. The arena figures quoted (the 0.72067 committee read, H131 Stage 1's techqa deltas, the H129 student's subset table) are context and appear in **no** threshold
- **It does not reopen R14-H132 or R13-H129.** Both verdicts stand. L4-C2 reads pretrained weights and licenses no arm; L4-C1 reads a refuted student and re-registers nothing
- **It does not claim the substrate survives fine-tuning at any size.** L4-C2 reads **pretrained** trunks. Whether the clean recipe erodes the substrate is `R15_hypotheses_L3_input_representation.md` L3-C4's question and is not answered here - which is precisely why L4-C4 makes the ship-candidate read a clause rather than an inference
- **The operand-visibility partition excludes 8.8% of the triples** whose operand rows could not be matched by first cell. Their exclusion is reported, not hidden, and they remain inside the truncation partitions
- **The 1024 residual read is in-domain only.** H131 Stage 1 already measured 1024 moving the arena mean nowhere and techqa negative on four checkpoints; nothing here contradicts that, and any in-domain gain must be recorded as a fact about a synthetic table probe rather than as an argument about the serving read

---

## 10. Reproduction

```bash
cd /home/lab/workspace/private/ai-assistants/groundrails
uv run python tmp/R15_L4_trunc.py     # ~90 s, CPU, Polars, no GPU, no network
```

Inputs: `experiments/grounding-semantic/R14_H133_triples.parquet` (2,000 banked triples with `score_a/b/c`), `data/external/datasets/dataset-tabfact.zip` (evidence rebuilt by `table_id`), `models/R9-H105-mmbert-dann-clean/` (tokenizer only - no weights loaded). Output: `experiments/grounding-semantic/R15_L4_truncation_census.json` - token geometry, the truncation and operand-visibility shares, and the four-way partitioned score and AUROC table reproduced in §1. The script lives in `tmp/` (gitignored, analysis-only). No tracked artifact was modified, no model was loaded, no GPU was used, and card 1 was not touched.

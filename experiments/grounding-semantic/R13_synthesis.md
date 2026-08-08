# Round 12/13 Fanout Synthesis — Judge Adjudication

18 candidates across 6 angles. **10 KILLED**, **8 survive amended**. Below: kills dropped, overlaps merged, survivors ranked, top-5 registration-ready, unaddressed failure modes, author queue.

---

## 1. Dropped (10 kills, no appeal)

| Candidate | Angle | Killing fact |
|---|---|---|
| OUTCOME-DIRECTION-FLIP-LANE | pubmedqa | Precondition measured 0.194 vs its own >=0.50 bar; direction-oracle 0.8337 loses to a matched-share random placebo 0.8783-0.9088 and to first-sentence-only 1.0000 — instrument anti-selected |
| BIOMED-TARGET-DOMAIN-ADVERSARIAL-LANE | pubmedqa | Kill gate cannot fire (dom-acc 0.48-0.52 measured through the ramp); R8-H79 v1 counter-prior omitted (pubmedqa -0.0866 under raised invariance); adds a 17th DANN group while H122 collapses 16→9 |
| CITATION-MARKER-STRIP-READ | hagrid | expertqa sign inverted (3 pos / 8 neg → oracle -0.0521); full-mean delta at 2/3/5/10x strip strength = +0.0005/+0.0006/+0.0001/-0.0005; H119 ruling forbids arena-subset-fitted transforms |
| EXHAUSTIVE-PAIR-CONCAT-READ | multihop | Fails its own cost gate: 837,274 pairs vs "<=700k"; 93.9% of spend on techqa+expertqa, neither multi-hop. Load-bearing question absorbed into TOP2 |
| CONJUNCT-SPLIT-MIN-READ | multihop | Fails all four sub-conditions: hotpotqa yield 20.8% (gate 50%), median units 1.0→1.0 (gate >=2), pubmedqa cap breach 22.8% and techqa 21.2% (gate <20%) |
| SHARED-PREFIX-BRANCH-SCREEN | variance | f=0.35 tail = 261,408 rows vs lanes of 61,184/83,672 — requires 3-4x lane repetition at the low-LR tail, a different intervention; fidelity anchor H108 is a null (+0.0019, inside SE) |
| VARIANCE-PROFILED-AGGREGATOR-READ | variance | Constraint set is empty: hard_min 0.7355 vs best alternative trimmed_min 0.7230 (-0.0126) against a 0.005 budget. Gate fired for zero GPU-h |
| UNLABELED-REGISTER-ANCHOR-CONSISTENCY | ood | Provenance gate non-executable (no PMID/title/URL in RAGBench schema); teacher ceiling measured at +0.0051 across 7 output-ensemble records vs a +0.010 bar; H118 midpoint reads 0.69218 blind |
| TRUNCATED-POSITIVE-CONTAINMENT-REWEIGHT | objective-mix | Pool misstated (71,341 not 59,781; TabFact reached silently); both gate clauses pre-satisfied with no full-document control; repaired selector touches 2.29% of the mix — under the 0.0042 SE floor |
| CONFLICT-SEMANTICS-MASS-SHIFT | objective-mix | Mechanism inverted: 49.58% of RAGTruth negatives are baseless-info-only, so RAGTruth is the mix's strongest "unsupported ⇒ negative" teacher — the anti-target the lane doubles. True mismatch pool is 52,981 VitaminC NEI rows; the weight vector cuts 463,238 |

---

## 2. Merges applied

**M1 — Exhaustive pairing folded into TOP2-UNION.** The selection question (is top-2-by-score the fault?) costs 11,928 extra forwards (~0.05 GPU-h) when restricted to hotpotqa/covidqa/tatqa/hagrid. It runs inside the TOP2 read. Stronger framing kept: TOP2 as the registered aggregator, exhaustion as its internal discriminator.

**M2 — Seeding amendments consolidated into one facility.** Three separate survivors independently demanded `torch.manual_seed` after model construction (SEED-LOCKED A4-equivalent, EVIDENCE-TOKEN-MASK A4, NON-ADVERSARIAL-SWAP A4), and R11-H117 amendment A4 already mandates it. One facility, one code change, one registration. Absorbed into record **R3**.

**M3 — One instrumented read pass serves R1 and R2.** Both need the three-level (sentence, chunk, window) score matrix that `R9_PC_windowed_dump.py` does not currently emit — it flattens chunks and windows into one `wlist`. Build the instrument once; R1 recomputes offline at zero GPU, R2 adds composite forwards. Registered separately (different aggregation axes, different targets), executed in one pass.

**M4 — One provenance-gate instrument for all corpus admissions.** SCIFACT, WICE and any future corpus need the same thing, and the RAGBench parquet exposes no PMID/title/URL — only `id` and raw `documents` text. One normalized n-gram-containment tool, run bidirectionally over document text, serves every admission. Author must fix n and threshold (awaiting-author item 4).

**M5 — SCIFACT and WICE are mutually exclusive in a wave.** Two new registers entering concurrently makes either verdict unattributable. Serialize; the loser's provenance gate still runs now on CPU.

---

## 3. Survivor ranking — (expected mean gain × mechanism confidence) / cost

Mean-arithmetic note applied throughout: a single subset moving +0.10 moves the 10-subset mean +0.010. Every subset-targeted lane in this fanout predicted mean +0.010 off a subset gain of +0.03-0.06, which is +0.003-0.006 of arithmetic. Ranking uses derived mean deltas, not claimed ones.

| # | Survivor | Cost (GPU-h) | Derived mean Δ | Confidence | Adjudicable at n=2 | Rank basis |
|---|---|---|---|---|---|---|
| 1 | WINDOW-CONSENSUS-EVIDENCE-READ | 0.50 | +0.001 to +0.002 | High (ceiling measured +0.0230 hagrid, collateral measured) | Yes, zero draw noise | Cheapest information per GPU-hour in the fanout |
| 2 | TOP2-UNION-PREMISE-READ | 0.70 | +0.003 | Low (negative control: hallucinated sentences gain +0.1524 vs grounded +0.1683) | Yes, deterministic paired | Closes the multi-hop read line on measurement for one eighth of a draw |
| 3 | SEED-PAIRED-ARM-ADJUDICATION | 0.00 marginal | 0 direct; unblocks every later lane | High (defect verified in source) | N/A — facility | Infinite ratio; time-critical |
| 4 | RAGTRUTH-PARALLEL-COPY-REBALANCE | 12.5 | claimed +0.010, no arithmetic ceiling (16.2% mass reallocation) | Moderate-high on the premise, moderate on the effect | Yes if paired via R3 | Only training lane whose precondition passed hard, zero new data, zero wall risk |
| 5 | WICE-ATTRIBUTED-SUPPORT-LANE | 11.1-12.5 | +0.004 | Moderate after re-aiming (12 negatives = 68.5% of misranked pairs, +0.1769 ceiling) | hagrid 2-draw SE ~0.010, +0.04 is 4 SE | First targeted hagrid lane; deterministic build, no LLM judge |
| 6 | SCIFACT-ABSTRACT-NEARMISS-LANE | 20.5 | +0.006 | Low-moderate; admission conjunction near self-contradictory | pubmedqa +0.060 = 4.1 SE yes; mean bar 0.7100 above every read ever taken (max 0.7062, n=8) with corr(mean, pubmedqa) = -0.859 | Highest strategic value, worst ratio |
| 7 | EVIDENCE-TOKEN-MASK-CONSISTENCY | 15.5 | unquantified | Low; banks the R-Drop null for free regardless | Only if R3 lands first | FM2 attack, but no measured mechanism |
| 8 | NON-ADVERSARIAL-INVARIANCE-SWAP | 13.0 | unquantified | Lowest; motivating premise measurably false | Confounded with in-flight H122 | Penalty-form test with a hostile counter-prior |

**Cut line after 5.** Items 6-8 get one line each in section 5.

---

## 4. Top 5 — registration-ready records

IDs assigned at registration. Names are descriptive.

---

### R1 — WINDOW-CONSENSUS-EVIDENCE-READ

**Causal claim.** Within-chunk max-over-windows takes a maximum over overlapping windows, so a single window that spuriously scores high sets the chunk score. At stride 750 over a 1500-char window, genuine support appears in at least two windows by construction; a spurious maximum does not. Replacing the within-chunk max with the mean of that chunk's top-2 windows (single-window chunks fall back to max; max over chunks and min over sentences unchanged) suppresses the spurious maxima that carry hagrid's standing windowing cost.

**Grounding.** hagrid windowing delta is negative on **10/10** recorded checkpoints, range -0.0085 to -0.0285 (H90, H100, H100d3, H105, H105d2, H107d1, H107d2, H108d1, H108d2, DR-d1) — stronger than the 8/8 the designer claimed. Exposure: 25/250 hagrid responses, 28/703 chunks (4.0%), 21 exposed positives / 4 exposed negatives. Measured reversion ceiling from the banked windowed dump: reverting exposed hagrid negatives only = **+0.0230**; reverting all exposed = **+0.0209**. Full-reversion collateral: delucionqa -0.0809, emanual -0.0660, expertqa -0.0098, techqa -0.0071, hotpotqa +0.0008.

**Numeric prediction.** On both frozen R10-H108 draws: hagrid >= **+0.010** (pair >= 0.658, ~48% capture of the measured reversion band); blind windowed mean >= **0.7031**; delucionqa, emanual and expertqa each >= **-0.02** against their max-read values.

**Two-sided bar.**
- **ADMIT** — all four clauses hold on both draws. Aggregator ships as the registered read.
- **REFUTE** — hagrid < +0.005, OR mean < 0.7031, OR any guard subset < -0.02. The evidence-side aggregation surface closes on measurement.

**Kill-gate.** The instrumented dump *is* the gate (0.25 GPU-h on draw1). consensus-top2 is recomputed offline from the retained matrix at zero extra GPU. Kill before draw2 on any REFUTE clause.

**Binding amendments.**
1. The instrument must retain **three-level** indices (sentence, chunk, window). `R9_PC_windowed_dump.py` builds `wlist = [w for k in ks for w in windows(k)]`, collapsing chunks and windows into one flat max — "max over chunks unchanged" is not implementable from a two-level extension.
2. Gate mean bar bound to **0.7031**, not "max-read mean -0.005" (= 0.7012). A gate that passes while the prediction fails is a false pass.
3. Strike the ">=3 of the 4 exposed hagrid negatives must have risen" precondition — n=4, statistically void. The offline recomputation is the gate.
4. **consensus-top2 is THE registered aggregator.** The retained matrix permits scanning arbitrary aggregators at zero cost; any other aggregator selected on draw1 is exploratory and cannot be promoted without its own pre-registered draw-2 confirmation. This is the only live false-pass channel.
5. expertqa added to the guard set alongside delucionqa and emanual.
6. Reproduce the recorded hard-min per-subset AUCs **exactly** on frozen H108 draw1 before any aggregator delta counts (P-C sanity protocol).
7. Not the closed line: the R12 window-bag KILL binds training-time window emission. This is frozen-weights, no training.

**Cost.** 0.25 GPU-h instrumented dump + 0.25 GPU-h confirm read = **0.50 GPU-h**. No training.

**Sequencing.** GPU0 or GPU2, **immediately**. Zero contention with the GPU1 serial queue (DR draw2 → H117 → R12 arms). Shares its instrument build with R2.

---

### R2 — TOP2-UNION-PREMISE-READ

**Causal claim.** Max-over-units is a logical OR: a conjunctive claim whose hops sit in different chunks is unconfirmable because no single premise entails it. Adding exactly one composite to the evidence pool — the two highest-scoring units concatenated in document order, each clipped to the frozen 750 stride, giving a 1500-char premise identical to the shipped budget — and scoring `max over {units} ∪ {that pair}` gives conjunctive claims a premise that can carry both hops.

**Grounding and the standing counter-evidence.** hotpotqa carries 4 chunks/response, median chunk 435 chars, 0.6% over 1500 (windowing is a measured no-op there). Oracle precondition passes: 56.3% of grounded-positive hotpotqa sentences gain >=0.15 content-token recall from the best 2-unit union; 98.9% of unions <=1500 chars. **The negative control collapses the discriminative story**: hallucinated hotpotqa sentences gain +0.1524 versus +0.1683 for grounded (56.5% vs 56.3% at the >=0.15 threshold); covidqa negatives gain *more* than its positives (+0.0760 vs +0.0629). The union is largely a text-volume effect. Nearest prior H14 (top-2 chunk concat *replacing* max) read -0.012 to -0.081 macro-F1; union-as-superset is a different construction but H14 is on-point evidence.

**Numeric prediction.** On both frozen R10-H108 draws: hotpotqa >= **+0.030**; blind windowed mean >= **+0.005**; no subset <= **-0.020**. techqa reported diagnostic-only (39,058 units, sd 0.0267); finqa and delucionqa carry no bar (sd 0.0401 / 0.0465).

**Two-sided bar.**
- **ADMIT** — all three clauses on both draws.
- **REFUTE** — hotpotqa < +0.030, OR mean < +0.003, OR any subset < -0.020 on draw1 → draw2 unspent, multi-hop read line closes on measurement.

**Kill-gate.** CPU recall diagnostic already run and passing. Post-read gate on draw1 only, per the REFUTE clauses. Instant kill on any per-subset switch, any tuned k, any tuned length — 1500/750 stay harness constants.

**Binding amendments.**
1. **Delete the monotone-safety claim** from the record. The metric is `roc_auc_score` (rank-based, `R7-H59_cross_domain_matrix.py:174`), so "never lowers a score ⇒ banked windowing gains cannot be undone" is void: raising every score can lower AUC. Per-subset floor tightens to -0.020, not -0.030.
2. The mean >= +0.010 bar is **struck as non-derivable**: a hotpotqa mechanism at +0.030 yields +0.003 of mean. Replaced by mean >= +0.005 with the exhaustive-pair transfer as the only route to more.
3. **Fold M1**: in the same run, also score the exhaustive pair set on hotpotqa/covidqa/tatqa/hagrid only — 11,928 extra forwards, ~0.05 GPU-h. This answers the top-2-selection question here. Never run exhaustive pairing over techqa/expertqa (786,105 of 837,274 forwards, no multi-hop claim, tightest variance band on the board).
4. **Pre-register the union FIRE RATE** (composite is the argmax) split by response label. If the fire rate on hallucinated responses matches grounded, record a diagnostic refutation of the two-hop premise regardless of AUC. This is the discriminator the positives-only precondition cannot see, and it is the clause that makes a null informative.
5. hotpotqa has 17 negatives / 250; Hanley-McNeil SE 0.057. State this at registration — +0.030 is inside one SE of the subset read, so the paired-draw agreement clause is load-bearing.

**Cost.** ~0.30 GPU-h per checkpoint + 0.05 exhaustive fold-in, both draws = **~0.70 GPU-h**. No training, no new data.

**Sequencing.** GPU0 or GPU2, **immediately**, same instrument pass as R1. Run R1's aggregator recomputation and R2's composite forwards off one dump.

---

### R3 — SEED-PAIRED-ARM-ADJUDICATION (facility; merges M2)

**Causal claim.** Neither `R10-H108_lane.py` nor the main-line trainer calls `torch.manual_seed` — `SEED` is a dead constant, and the header states model init and batch permutation are unseeded. Every lane verdict is therefore an unpaired comparison of independent draws against a banked baseline. Seeding model init immediately after construction makes lane-minus-control a paired statistic and cancels the shared-init component of per-subset draw noise, which is the binding noise (like-for-like same-recipe replicate per-subset SD median 0.0198-0.0204; mean-level SD only 0.0023).

**Numeric prediction.** Pooled per-subset paired-delta SD <= **0.014** against the like-for-like unpaired 0.0198-0.0204 (>=30% cut), measured over >=10 subset-seed cells.

**Two-sided bar.**
- **ADMIT** — cut >= 30%; per-subset hold clauses may then be re-priced downward on evidence from full-length paired arms.
- **REFUTE** — cut < 15% or paired SD >= 0.018. Init-pairing is not the noise source; per-subset hold clauses stay at 0.06 and FM4 stands unaddressed.

**Kill-gate.** None required — there is no GPU spend to gate.

**Binding amendments.**
1. **Strike the 4-probe 4.1 GPU-h gate entirely.** 2,000 of ~15,560 steps is 12.9% of one epoch with warmup ending at 1,556; arms would have seen ~7.9k of 61,184 lane rows. Paired SD looks small there for a reason that does not survive to full training, and the one-sided SD>0.05 clause cannot catch it.
2. **Measure at zero marginal GPU** off arms already seeded and queued: R12-H122 (control vs 9-group, seeded paired draws) is the primary vehicle; DR control/margin arms secondary.
3. **Init-pairing only.** Row-order pairing is impossible for data-addition lanes: clean 685,670 vs clean+H108 746,854 rows means `np.random.permutation(len(ds))` diverges at equal seed, and `n_groups` differs — the exact RNG trap H122-A2 records. Row-order pairing is available only under the H117 option-(b) construction (identical row set, effect carried by loss/mask). State this in the registration.
4. **Drop the mean-level claim.** Banked same-recipe mean SD is 0.0023 → 2-draw SE 0.0016, so +0.005 is already 3.1 SE unpaired. Per-subset hold clauses are the only live target.
5. Replace the Spearman >= 0.5 gate (n=10 subsets, p~0.14, cannot adjudicate) with pooled per-subset paired-delta SD across >=10 subset-seed cells.
6. Mask/aug lanes draw from a **dedicated `torch.Generator`** so the global stream is untouched; assert bit-identical trunk + task_head init before step 0 in both arms.
7. Do not relax hold clauses from 0.06 to 0.03 until measured on full-length paired arms.

**Cost.** **0 marginal GPU-h**; ~1 h engineering.

**Sequencing.** **Time-critical and first.** The seed call must land in the trainers *before* H117 and the R12 arms start, or the facility misses its free measurement vehicle and needs dedicated GPU later. Note for the author: adding the seed changes init for all future draws and breaks strict comparability with banked unseeded draws (awaiting-author item 8).

---

### R4 — RAGTRUTH-PARALLEL-COPY-REBALANCE

**Causal claim.** RAGTruth is the only arena-shaped register in the mix (multi-sentence response, whole-response label, document-length evidence), and 87.5% of its mass is translated duplicates of the same 15,090 items in languages the arena never serves. A family-mass-preserving reweight — EN 4.0, each of 7 translations 0.5714, family fixed at 120,717 row-equivalents, all other corpora untouched — spends the same budget on four times the English exposure.

**Grounding (measured, passed hard).** All 8 files 15,090 rows pre- and post-filter. Positional label agreement EN vs de/fr/es/it/pl/hu/cn = 1.0000 / 0.9999 / 0.9999 / 1.0000 / 0.9998 / 0.9999 / 0.9999; task_type agreement 1.0000 on all 7; pos_frac 0.554606-0.554805, spread **0.000199** against a <0.002 gate. Content correspondence independently confirmed by numeric-token Jaccard: aligned median 0.84-0.88 vs 0.13 shuffled control. Marginal information near-zero: EN 0.8434 vs non-EN 0.8407.

**Numeric prediction.** 2-draw pair blind windowed mean >= **0.7150** (+0.010 over the H108 incumbent 0.70496) with sign agreement on both draws. Holds: ragtruth_nonen >= 0.82 on both draws, gold_full >= 0.84, no arena subset < 0.55.

**Two-sided bar.**
- **ADMIT** — pair mean >= 0.7150 with sign agreement and all holds.
- **REFUTE** — pair mean < control (0.70496) OR sign disagreement across draws. (The "<+0.002" form is struck: it sits inside the 0.0042 SE.)

**Kill-gate.** CPU alignment gate **already run and passed** (0.5 CPU-h, 0 GPU-h). No further pre-GPU gate.

**Binding amendments.**
1. Alignment is **positional** — the translated files carry no source-index column (schema: prompt / answer / labels / task_type / dataset / language). Record it as positional-index alignment corroborated by the numeric-Jaccard control; do not claim index-keyed joining.
2. **Strike the second kill-gate** ("non-EN trails EN by >0.03"): measured non-EN *leads* EN on H108 d1 (0.8421 vs 0.8246), d2 (0.8291 vs 0.8140) and H105 d1 (0.8402 vs 0.8382); max observed trail is 0.0024 on H105 d2 — a 12x margin to threshold. Replace with a HOLD only: ragtruth_nonen >= 0.82 on both draws, breach reported as a deliverable finding.
3. **Strike the EN-below-non-EN inversion from the motivation.** It holds on both H108 draws (-0.0175, -0.0151) but reverses on H105 d2, at n=600 where AUC SE is ~0.017 — one SE, not a fact. The lever stands on parallel-copy redundancy alone.
4. The intervention changes per-group DANN mass (EN group 4x). **Do not co-run or cross-adjudicate with R12-H122**; report per-group discriminator accuracy alongside the blind read so the GRL confound is visible.
5. Restate all shares against the served 746,854-row mix (RAGTruth family 16.2%), not the 685,670 public subset.
6. Run under R3's seeding so lane-minus-control is init-paired.

**Cost.** 2 draws at 5.3-6.0 + 2 windowed reads at 0.25 = **~12.5 GPU-h**. Gate already spent.

**Sequencing.** GPU1, after DR draw2 → H117 → R12 arms. First training lane in the post-R12 queue: zero new data, zero contamination-wall exposure, precondition already banked.

---

### R5 — WICE-ATTRIBUTED-SUPPORT-LANE (re-aimed)

**Causal claim (re-aimed — this is the material change).** hagrid's loss is **not** low-scoring positives. On the windowed dump, 68.5% of hagrid's 2,975 misranked pairs involve one of just **12 negatives scoring >0.5**; suppressing those to the negative median is worth **+0.1769**, against **+0.0649** from lifting all 71 low positives. WiCE's partial-support supervision — positive = claim + full minimal evidence set; negative = same claim with one sentence deleted from a >=2-sentence set, or that sentence swapped for the lexically nearest sentence of another article — is a strictness signal on over-claim, aimed directly at the 12.

**Wall status.** WiCE is Wikipedia claims + cited external sources; not a RAGBench source corpus or derivative. Wikipedia substrate is admitted precedent (VitaminC is already a DANN group in `R9-H105_clean_mix.py`). Deterministic build, no LLM judge.

**Numeric prediction.** 2-draw pair: hagrid >= **0.688** (+0.040 vs 0.6477, ~4 SE at the measured 2-draw SE ~0.010); blind windowed mean >= **0.7031**; finqa >= 0.688; techqa >= 0.670 (each >= -0.03 vs H108, the H107 displacement guardrail).

**Two-sided bar.**
- **ADMIT** — all four simultaneously on the 2-draw pair.
- **REFUTE** — any of the four missed. Note at registration that 0.688 exceeds the campaign-max hagrid (0.6600) by +0.028.

**Kill-gate.** Pre-GPU, data only — kill on any of: provenance overlap >=0.5%; buildable pairs <15,000; multi-sentence-evidence fraction <40%; license not permissive. Then a 1-draw pilot: **both** clauses must pass to spend draw2 — mean >= 0.700 **and** hagrid >= +0.02.

**Binding amendments.**
1. **Provenance gate defect.** SHA1 passage-hash overlap cannot fire (any whitespace difference kills a collision). Replace with **normalized 8-gram containment, threshold <0.5%, run in BOTH directions**: WiCE passages vs arena chunks, *and* WiCE **claim** text vs the 703 hagrid + 998 hotpotqa chunks. WiCE claims are Wikipedia sentences and hagrid/hotpotqa chunks are Wikipedia passages — that is the live leak path, and the original gate checks the side that cannot collide.
2. Re-register the mechanism on the **12 high-scoring negatives**, not bottom-quartile positives. Add a pre-GPU diagnostic that the deletion negative is a strictness signal on over-claim; report the 71 low positives as the counter-risk.
3. **Delete the "no attributed-QA register" premise** — VitaminC occupies it. The claim is partial-support *label refinement* over a register already present.
4. **Pilot gate raised** from mean 0.690 to **0.700**: 0.690 is already unrecoverable, since a 0.7031 pair would require draw2 at 0.7162, above the clean-family max 0.7062.
5. hagrid < +0.02 on draw1 is **not** sufficient evidence in the other direction on its own — H108, a lane with zero hagrid content, moved hagrid +0.0137 on the pair. Both pilot clauses required.
6. Retained: evidence-access is not the bottleneck (bottom-quartile grounded argmins have best-single-chunk lexical coverage mean 0.679 / median 0.75, 47.2% >=0.8).
7. Run under R3's seeding.

**Cost.** Build ~0 GPU (deterministic). Draw1 5.3-6.0 + read 0.25 = 5.6-6.3 to the pilot gate; full pair **11.1-12.5 GPU-h**.

**Sequencing.** Provenance and yield gates run on CPU **now**, in parallel with everything. Training on GPU1 after R4. Per M5, mutually exclusive in-wave with SCIFACT.

---

## 5. Survivors below the cut — one line each

- **SCIFACT-ABSTRACT-NEARMISS-LANE** (20.5 GPU-h) — the only live lever on uncovered target #1, but its admission conjunction is near-self-contradictory on the historical record: mean >= 0.7100 is above every windowed read ever taken (max 0.7062, n=8) while corr(mean, pubmedqa) = **-0.859** across those 8; register-absence causation is falsified in sign by both free controls (covidqa 71.8% biomed with per-sentence positive median 0.8930; within pubmedqa, marker-bearing argmins score *higher* at 0.0621 vs 0.0422), leaving only the length-controlled in-register deficit (pubmedqa 0.0743 / 71.4% below 0.1 vs covidqa 0.8224 / 2.8% at n_sent<=2) as the legal restatement — **run its CPU + 0.05 GPU-h pre-build gates now regardless of promotion**, they are nearly free, and put promotion to the author (item 1).
- **EVIDENCE-TOKEN-MASK-CONSISTENCY** (15.5 GPU-h) — banks the R-Drop null for free right now (all four ModernBERT dropout channels read 0.0 in every trunk config; the sole `nn.Dropout(0.1)` sits in `domain_head`, off the task path, so two passes are bit-identical and the KL term is identically zero), but the lane itself needs A1 (pin the double-encode fraction — mechanism says every row, cost says 25% of batches, a 4x ambiguity), A2 (p90 not median support-shift guard), A3 (a *rejecting* precondition: masked inter-draw disagreement must beat a random-token-substitution control of equal token count by >=1.5x, since raw amplification passes by construction), and R3's seeding before its +0.003 REFUTE line is anything but 0.7 SE.
- **NON-ADVERSARIAL-INVARIANCE-SWAP** (13 GPU-h) — motivating premise measurably false (no degenerate equilibrium at the operating point: dom-acc holds 0.48-0.52 against chance 0.077 through the full lambda-0.02 ramp; H79 v1's inversion is recorded as a 60k-pair small-scale artifact), both gate clauses non-rejecting, `domain_head` is 199,948 params not ~600k, and lambda parity is broken because `loss = t_loss + d_loss` applies lambda only inside the GRL — survivable only as a penalty-**form** test with lambda_CORAL calibrated to matched trunk-gradient-norm ratio via the already-built `R12-H122_gradgate.py`, sequenced strictly behind R12-H122 on its seeded control pair.
- **ANCHOR-TEACHER CEILING DIAGNOSTIC** (0.5 GPU-h, salvaged from a killed lane) — score the **output-mean of the two frozen R9-H105 draws** through the R8-H101 windowed arena read; that object has never been measured, it is the teacher every consistency/distillation lane would distil, and if it lands below pair mean +0.005 the whole class is closed for 0.5 GPU-h on GPU0/GPU2. Cheapest class-level kill available; run it alongside R1/R2.

---

## 6. Failure modes still unaddressed after this fanout

**FM3 / pubmedqa — no lever in the top 5.** Target #1 (0.5741, the fattest headroom pool, 46% of weak-subset headroom shared with hagrid) exits this fanout with its only survivor ranked 6th at the worst ratio on the board. Three of the six pubmedqa-angle candidates died on measured preconditions. The register-absence causal story is now **falsified in sign** and only the length-controlled in-register deficit survives — a weaker and less actionable mechanism than the round opened with. **This is the largest open gap.**

**FM2 / OOD functional divergence — every intervention killed or below the cut.** The anchor lane is dead (teacher ceiling +0.0051 measured across 7 output-ensemble records vs its +0.010 bar; H118 midpoint reads 0.69218 blind, i.e. the consensus object is *worse* blind). Mask-consistency and the invariance swap both sit below the cut with unquantified effects. No survivor in the top 5 touches FM2. The R-Drop route is now measured shut (dropout = 0.0 everywhere on the task path).

**FM4 / adjudication variance — half-addressed.** R3 cancels the init component, and only for arms with identical row sets or via H122's free vehicle. **Data-addition lanes remain structurally unpairable on row order** (different row counts, different `n_groups`), which is exactly the class every corpus-admission lane belongs to — R4 and R5 both. The aggregator route is measured shut (hard_min 0.7355 vs best alternative 0.7230; kth2 0.6975, kth3 0.6654, trimmed_min3 0.7144 — nothing inside a 0.005 budget, and the cause is structural since RAGBench `adherence_score` is a response-level AND, making min the label-matched aggregator). The shared-prefix screen is dead on tail arithmetic. R8-H100's confirmed 0.0295 vs 0.0074 min-amplification stands as a real, unsolved problem. Per-subset hold clauses stay at 0.06.

**FM5 / capacity at the operating point — untouched.** No candidate in any of the six angles tested the 307M/16-group tension. R4 brushes it (parameters spent fitting 8x duplicated supervision) but is a mass-allocation lever, not a capacity test.

**Mean-bar arithmetic — a systematic design defect across the fanout.** Nine of eighteen candidates predicted mean >= +0.008 to +0.010 off a single-subset mechanism worth +0.03 to +0.06, which is +0.003 to +0.006 of arithmetic. No candidate supplied a cross-subset transfer argument, and the one piece of evidence on transfer points the wrong way (corr(mean, pubmedqa) = -0.859 over 8 reads). Until the campaign has a measured transfer model, **subset-targeted lanes cannot be adjudicated on mean-gain bars they cannot arithmetically reach** (awaiting-author item 7).

**Label ceiling caps the framing.** `R12_label_ceiling_result.json` reads pubmedqa faithful-oracle **0.7789** and hagrid **0.7833** under the response-level labels. Against 0.5741 and 0.6477 that is +0.205 and +0.136 of *reachable* headroom, materially less than the raw-to-1.0 framing in the campaign brief. The file is licensed ANALYSIS ONLY, so it cannot set bars — but if it may inform priority, hagrid's cost-adjusted attractiveness rises further relative to pubmedqa (item 6).

**Read-amendment concurrency.** Two read amendments (R1 consensus-top2, R2 top2-union) are proposed in one round. Both are subset-blind and pre-registered, but adopting both makes the shipped read a moving target and their interaction is unmeasured — neither pack's guards cover the other's transform (item 5).

**Provenance instrumentation.** The contamination wall currently has **no executable instrument** for corpus admission: RAGBench parquet exposes no PMID, title or URL, only `id` and raw `documents` text. Every provenance gate proposed in this fanout was written against fields that do not exist, and would have reported 0% overlap by construction. Corpus admission is open by ruling but not yet enforceable in practice (item 4 — highest-priority author decision).

---

## 7. Awaiting author

1. **pubmedqa priority vs cost ratio.** SCIFACT-ABSTRACT-NEARMISS ranks 6th by (gain × confidence)/cost and would displace roughly two cheaper training lanes at 20.5 GPU-h. pubmedqa is uncovered target #1 and the admission ruling was opened specifically for its register. **Does strategic priority promote it into the wave, and if so at whose expense (R4 or R5)?** Its CPU + 0.05 GPU-h pre-build gates run now either way.
2. **SciFact admission ruling.** 1,409 expert claims over 5,183 S2ORC Medicine/Biology abstracts, CC BY 4.0 + ODC-By 1.0. Shares PubMed ancestry with pubmedqa. Proposed gate: drop any abstract matching a ragbench pubmedqa document, **extended to covidqa (71.8% biomed, CORD-19 ancestry) and expertqa (28.1%)**, at 8-gram Jaccard >= 0.3, KILL at >2% of the corpus. **Admissible on those terms?**
3. **WiCE admission ruling.** Wikipedia claims + cited external sources; not a RAGBench source corpus; Wikipedia substrate already admitted via VitaminC. Live leak path is the **claim side** (WiCE claims are Wikipedia sentences; hagrid and hotpotqa chunks are Wikipedia passages). **Admissible with a bidirectional claim-side + passage-side containment gate?**
4. **Provenance-gate instrument — highest priority.** RAGBench parquet has no PMID / title / URL field, only `id` (e.g. `pubmedqa_11200`) and raw `documents` text. The only executable dedup is n-gram containment over document text. **Fix the canonical instrument**: proposed normalized **13-gram containment**, run bidirectionally, WARN at 0.5%, KILL at 2% of the candidate corpus. Every future admission depends on this ruling; without it no provenance gate in this fanout can actually fire.
5. **Read-amendment budget.** R1 and R2 are both subset-blind, both pre-registered, both frozen-weights. **May both enter this round, or serialize** (R1 first, R2 re-priced against the amended read)? Their interaction is unmeasured.
6. **Label-ceiling licensing.** `R12_label_ceiling_result.json` (pubmedqa 0.7789, hagrid 0.7833) is ANALYSIS ONLY. **May it inform per-subset bar setting and target prioritization, or must bars remain ceiling-blind?**
7. **Mean-bar arithmetic ruling.** Subset-targeted lanes cannot arithmetically reach mean +0.010 from a +0.03-0.06 subset move. **Adopt subset-primary bars with a mean HOLD (no-loss) clause for subset-targeted lanes**, reserving mean-gain bars for mix-wide levers such as R4? This changes the admission form for R5 and any SCIFACT promotion.
8. **Trainer seeding change.** R3 adds `torch.manual_seed` after model construction. This changes init for all future draws and breaks strict comparability with the banked unseeded draws (H105 pair, H108 pair, DR control). **Confirm the change, and confirm the banked draws remain the comparison baseline despite the init-distribution change.** Time-critical: must land before H117 and the R12 arms start.
9. **Per-subset hold clauses.** Stay at 0.06 until R3 measures the pairing cut on full-length arms. **Confirm** — R5's finqa/techqa guards and R1's delucionqa/emanual/expertqa guards are written tighter (-0.02, -0.03) than the general clause and will fire on noise if the general clause is right.
10. **GPU1 queue order.** Post-R12 queue is DR draw2 → H117 → R12 arms → then R4 → R5 (→ SCIFACT if promoted). That is ~25 GPU-h of new lane work behind the existing queue. **Confirm the ordering, and confirm whether GPU0/GPU2 may take a training draw** to parallelize — currently they are gates-only, and the read work in R1/R2 plus the salvage diagnostic consumes under 2 GPU-h of their capacity.
11. **PUBHEALTH status.** It sits on R10's binding refused list and appeared in two killed candidates. **Confirm it remains refused**, and rule on Evidence Inference 2.0 (4,005 PMC-OA articles) and NLI4CT (1,000 ClinicalTrials.gov CTRs) — both were only ever used by killed lanes and have no standing verdict of their own.
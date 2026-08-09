# R15 LENS 1 - lane-construction amendments to R14-A4 / H133

**Role**: lane-construction lens for the Round 15 repair field. Output: four candidates - two amendments that must land before the registered derivation-parity lane is built, one new sub-block, one standing measurement - each with a pre-arm killgate, a ceiling-blind bar and a cost.

**Discipline**: frozen weights only, held-out TabFact (`table_id`-disjoint from every training split), zero arena, zero gold, Polars throughout. Three original measurements were taken for this lens and are marked **[L1]**; two are CPU-only and one spent **~0.3 GPU-h total on card 0** across two checkpoints. Artifacts: `R15_gate_L1_binding.json`, `R15_gate_L1_binding_R10-H108-lane-draw1.json`, `R15_gate_L1_lcdp.json`, `R15_gate_L1_serialtok.json`, `R15_gate_L1_rowcap.json`, `R15_L1_bindprobe_pairs.parquet`.

**Scope note**: this lens owns the lane's *composition* - operator mix, negative families, value ranges, serialization, document diversity, size, and non-table augmentation. Surface spelling of the asserted value is `R15_hypotheses_L3_input_representation.md`'s territory and objective terms are `R15_hypotheses_L2_objective.md`'s; declared interactions with both are in section 6.

---

## 0. The finding that organises this lens

**The lane as licensed is built on the one binding axis the model has already solved, and is blind to the one it has not.** A present-value probe over 2,400 held-out TabFact pairs [L1] separates two things that `R15_probe_P1_anatomy.md` §4 could only infer jointly:

| arm | claim | AUROC(correct vs wrong) H105 d1 | replication on H108 lane d1 |
|---|---|---:|---:|
| `bind_row` - right column, **wrong row** | "The {col} of {ka} is {v}" | **0.9936** | **0.9943** |
| `bind_col` - right row, **wrong column** | "The {colx} of {ka} is {v}" | **0.5179** | **0.5276** |
| `compare` - ordering, no computation | "The {col} of A is greater than the {col} of B" | **0.5076** | **0.5184** |
| `absent_ctl` - verbatim cell vs fabricated absent value | anchor | 0.9889 | 0.9950 |

Every asserted number in the first three arms is **verbatim present** in the evidence, so no arithmetic and no absence detector is involved. The model resolves a row label at 0.9936 and a column header at 0.5179 on byte-identical templates, and it accepts the wrong-column value at mean score **0.90391** - almost exactly the 0.91559 it gives the right one.

Three consequences, and they redirect the row budget.

1. **`R15_probe_P4_numeracy.md`'s prescription P4-1 spends 35% of the lane on operand binding, which its own §9 flags as "a prior drawn from E4's forensics, not a measurement". It is now measured, and in its row form it is at 0.9936.** Spending 35% there buys nothing. The hole is on the **header** axis.
2. **R14-A4's census enumerates two-operand tuples *within one column* only** (`R14_gate_H133_census.json`: four operations, all two-operand, all within one column; verified in `tmp/R14_gate_H133_census.py`). Every A4 claim therefore names one column, and the lane **cannot teach column binding by construction** - while `R14_evidence_E4_items.md`'s canonical near-miss, response 200's cross-line-item ratio certified at 0.7493 and priced at **13.4% of all finqa discordance**, is precisely a cross-header binding error.
3. **The defect is not the shipped checkpoint's alone.** The replication is on `R10-H108-lane-draw1` - the campaign's only ADMITTED lane and its only replicated finqa lever (pair finqa 0.7182 against clean 0.6333) - whose 33,176 corruption negatives include an explicit `f_comparative_flip` family. **3,743 of the lane's 61,184 claims (6.12%) carry comparative language** [L1], and the checkpoint trained on them reads ordering at 0.5184. Comparative supervision at 6% of a lane does not install ordering.

**The second organising fact is that the model is worse than the zero-parameter rule it is imitating.** Recomputing AUROC from longest-common-digit-prefix alone, with no model involved [L1]: scale/unit **0.9319** against the model's 0.8755, rounding **0.9400** against 0.7049, mean **0.5626** against 0.5257. On all three types where a digit-prefix cue exists, a rule with no parameters out-predicts a 307M cross-encoder. `R15_probe_P1_anatomy.md` §6's allocation for scale (8%, hold-only-and-control) is right; its framing of scale as "the one thing that already works" is generous.

**And the registered probe is clean.** On the 2,000 banked `R14_H133_triples.parquet`, AUROC from digit prefix alone is **0.4871** and from digit length alone **0.4992** [L1], against the model's 0.4924. The 0.4924 is a genuine reading of the model, not a cue artefact - which is what licenses building on it.

| id | kind | one-line | marginal GPU |
|---|---|---|---:|
| **L1-C1** | amendment (R14-A4) | Operator, negative-family and difficulty schedule - replaces the sum-only build, adjudicates P1's deficit ordering against P4's learnability ordering on measurement | ~0.1 |
| **L1-C2** | amendment (R14-A4) | Budget the lane's evidence in **body rows**, not characters - the character budget is form-dependent and P3's recommended serialization mix truncates 46.5% of rows | 0 |
| **L1-C3** | new-hypothesis (A4 sub-block form) | Relational sub-block: cross-header binding and ordering, the two zero-computation families measured dead on two checkpoints | ~0.1 (sub-block) / ~13 (standalone) |
| **L1-C4** | measurement | Natural-derivation transfer instrument - the only read that separates "learned the construction" from "learned derivation checking" | ~0.2 |

---

## 1. The measurements this lens adds [L1]

**(a) Present-value binding and ordering probe.** `R15_L1_bindprobe.py`, frozen weights, held-out TabFact test+validation with train `table_id`s excluded (3,391 tables), 600 pairs per arm, seed 20260809, ~0.15 GPU-h per checkpoint on card 0. Every pair is byte-identical except the swapped item. Results in section 0. Two honest limits: `bind_col` negatives are **not** controlled for digit length or magnitude - "The founded of bridgewater state university is 11201" is a live example - so the negative is *easier* to reject than a controlled one and 0.5179 is a conservative reading, not an inflated one; and `bind_col` draws 600 pairs from 415 distinct tables against 600 for the other arms, because it needs two numeric columns sharing a row.

**(b) Digit-prefix exploitability, per derivation type.** `tmp/R15_L1_lcdp.py`, CPU, over P1's banked 4,139+869 quads and the 2,000 H133 triples. Reported in section 0 and used in section 2's schedule. It also runs `R15_probe_P1_anatomy.md` §9's first falsifier weakly: on `ratio` the *wrong* value carries the longer prefix (0.172 against 0.120, signed advantage -0.052) and the copy account predicts AUROC below 0.50; the measured score AUROC is 0.5121 with Pearson(ΔLCDP, Δscore) = **-0.1731**. The prefix advantage is 0.05 digits, far too small to be a real test - **recorded as untested, not as passed**, and the clean version of the falsifier remains worth its 0.05 GPU-h.

**(c) Serialization token cost and the row-budget re-census.** `tmp/R15_L1_serialtok.py` / `serialtok2.py` (CPU, shipped tokenizer, 1,499 TabFact train tables) and `tmp/R15_L1_rowcap_census.py` (CPU, `R14_gate_H133_census.py`'s predicates reused verbatim, all 13,143 unique TabFact train tables). Both tables are in section 3. The re-census's `full` configuration reproduces the banked census exactly - **8,438 admitting tables, 1,121,632 tuples** - which is the check that the reused predicates are the registered ones.

---

## 2. L1-C1 - AMENDMENT to R14-A4: the operator, negative-family and difficulty schedule

**kind**: amendment | **cost**: ~0.1 GPU-h marginal (gate already spent; 0.1 for the post-arm per-type read). Build is CPU.

**Claim** - Because R14-A4 was licensed on a **sum-only** probe while `R15_probe_P1_anatomy.md` §3 measures the defect as type-uniform across ten types and 5,008 held-out quads (sum 0.5067, difference 0.4994, mean 0.5257, ratio 0.5121, percent-change 0.4861, product 0.5165, count-aggregation 0.4842, date arithmetic 0.5014, none clearing 0.53) with sum itself only **7.1% of finqa's and 6.4% of tatqa's absent numerals**, and because the two competing allocations in the probe pack are now separable on measurement - P1 §6 orders shares by *deficit* while `R15_probe_P4_numeracy.md` P4-1 orders them by *learnability* and spends 35% on an operand-binding family this lens measures at **0.9936** in its row form [L1] - building the lane to the pre-registered schedule below will move held-out per-type AUROC(correct vs wrong-operand) above 0.60 on the four tier-1 combinatorial types while held-out scale/unit AUROC holds at or above 0.80 and A4's registered arena holds are unbroken.

**The schedule.** Shares are stated as fractions of whatever derivation-core row budget survives section 6's arbitration, so the table is scale-free.

| type | share of core | AUROC b vs c (P1) | correct-value score (P1) | why this share |
|---|---:|---:|---:|---|
| difference / period change | **20%** | 0.4994 | 0.3489 | chance, and the only type where wrong-**operator** is an active anti-signal (0.4319 - the model prefers the sum) |
| ratio / share-of-total | **18%** | 0.5121 | 0.3010 | chance, register-dominant in finqa at 19.6% of absent numerals |
| percent-change | **15%** | 0.4861 | 0.4223 | chance, shallowest deficit of the tier - fewer rows buy the same movement |
| sum / total aggregation | **12%** | 0.5067 | 0.2326 | chance, second-deepest deficit, and the only type H133 measured, so it is also the gate's replication anchor |
| rounding / approximation | **12%** | 0.7049 | **0.1763** | deepest deficit of all ten types, and the shape H108's near-miss negatives most directly mistrained. Half the signal is a digit cue already (LCDP-alone 0.9400 [L1]); the lane only has to correct the sign |
| mean / average | **10%** | 0.5257 | 0.2549 | chance, deep deficit, shares operand machinery with the sum rows |
| scale / unit conversion | **8%**, hold-only | 0.8755 | 0.4965 | **the lane's internal control, not a target.** The model reads 0.8755 and a parameterless prefix rule reads 0.9319 [L1] - spend the minimum that prevents regression |
| product | **5%**, 2-3 significant-digit operands only | 0.5165 | 0.2683 | chance, but exact multi-digit multiplication is not a plausible learned function (P4 §4); keep the rows, cap the precision |
| count-aggregation, date arithmetic, depth-2 chains | **0%** | 0.4842 / 0.5014 / - | - | count answers produce no absent number so they cannot participate in the shortcut the lane exists to remove; date arithmetic is 139-of-500 constructible under the absence rule and absent from the register read; chains are 15 of 250 finqa deciding sentences and have no scratchpad (P1 §7) |

**Negative families, per type** (P1 §6.1, `R15_probe_P3_signal.md` §2's taxonomy N1-N7):

- **25% wrong-operator** (N2), concentrated on difference-vs-sum and mean-vs-sum, the two measured inversions (0.4319 and 0.4648)
- **75% wrong-operand** (N1/N3/N4/N6/N7), split **50 / 30 / 20** across order-of-magnitude-wrong, arbitrary wrong-operand, near-miss. P1 §3.1 measures the model at chance even against decade errors (difference 48.8%, percent-change 49.5%, product 47.4% of quads scored correctly), so there is no case for spending the first tranche on 2%-off near-misses
- **N7 (numeral corruption of the correct result) capped at 10% of all negatives.** P3's cap binds: the near-miss slice is 0.75 x 0.20 = 15%, so the cap cuts it. N7 is the operator class that produced the defect - `f_digit_perturb` over roughly 45,000 H108 rows, the artifact measured at P(0 | absent) = 0.9332
- **One negative per positive**, sampled from the family distribution, so P(label 0 | absent) is 0.5 exactly inside the lane. Emitting all four constructible negatives gives 0.75 and inverts the defect

**Three binding construction clauses**, each closing a measured route:

1. **Byte-identical templates within a pair.** P1 §5 measures a free +0.148 of score available to any model that counts quoted operands, with AUROC unchanged at 0.5010 → 0.4973 over 1,600 pairs. The positive and its negative differ in the asserted numeral and in nothing else
2. **Digit-length parity within 1** (P3 §2) and the build KILL from `R15_probe_P2_tokenizer.md` P2-B at AUROC-from-token-length > 0.55. The H133 triples satisfy parity on 88.95% of rows **by luck**; nothing in the construction enforces it
3. **Prefix-balanced rounding negatives.** Rounding is intrinsically digit-preserving, so half its negatives must be rounded from the correct cell in the wrong direction or to the wrong place, preserving prefix length. Otherwise the 12% rounding slice is solvable on the copy detector alone, which reads 0.9400 there [L1]

**Value-range coverage.** `R15_hypotheses_L3_input_representation.md` §2 measures TabFact cells at 68.1% one-to-two digits while H133's derived results are 67.9% four-to-six digits, and P4 §5 makes the covered range the competence range (extrapolation R² **-46.1**). Select source columns to a pre-registered per-result-digit-length quota over 2-7 digits and report the realised distribution in the build manifest; rebalance before training if any single length exceeds 35% of the core. This is a coverage clause and carries no kill of its own - P2-B's length-shortcut KILL is the enforcement.

**Killgate** - **clause 1 is already run and passing** [L1]: P4-1's largest allocation must be a real gap. Measured `bind_row` 0.9936 on H105 draw 1 and 0.9943 on H108 lane draw 1, so the 35% row-binding allocation is **REFUTED by measurement** and is not carried into the schedule; the schedule above is the surviving allocation. **Clause 2, ~0.1 GPU-h, held-out, arena-free, to run before the build freezes**: re-read P1's banked quad parquet on `R10-H108-lane-draw1`. **KILL the type-uniformity premise** - and with it this schedule - if any tier-1 type reads AUROC(b vs c) above 0.60 on that checkpoint, because the defect would then be checkpoint-specific rather than recipe-specific and the shares would be fitted to one draw. This is `R15_probe_P1_anatomy.md` §9's second falsifier, priced there at 0.2 GPU-h and cheaper on the banked quads.

**Bar** - **A4's registered bars stand unchanged and this candidate proposes no movement in them**: PRIMARY finqa 2-draw mean >= **0.6933** (+0.060 over the 0.6333 paired control) with sign agreement on both H126-paired draws; ANTI-GAMING in-domain held-out present-value near-miss AUC not below the clean-recipe value; CONFOUND log mean-sentence-length residualization at >= 50% of magnitude with the same sign; HOLD arena mean >= 0.7031, pubmedqa >= 0.5463, `gold_full` >= 0.8414, RAGTruth non-EN >= 0.82, no subset more than 0.06 below its paired control and none < 0.55; PILOT KILL at draw-1 finqa < control + 0.020. **Declining to move the primary is deliberate**: widening a lane from one operation to eight should not earn a lower bar.

Its own pre-registered, ceiling-blind, arena-free **MECHANISM** reading on both draws, taken on P1's held-out quads:

- **the four tier-1 combinatorial types (difference, ratio, percent-change, sum) above AUROC 0.60**, from a banked baseline of 0.4861-0.5121
- **VOID clause, adopted from P1 amendment 3**: held-out scale/unit AUROC falling below **0.80** voids the run regardless of the finqa number - the lane has taught "absent implies supported" and has cost the model its one working numeric instrument
- **VOID clause, verbatim non-regression**: held-out verbatim mean >= **0.85** against the banked 0.90507, and AUROC(a vs b) >= **0.90** against the banked 0.9643
- **Warning signature, pre-registered per P4-5**: ratio, product and percent-change moving as much as difference and sum is a **construction-artefact warning, not a success** - nothing in the published record or in either probe predicts a 307M encoder learning division from a few thousand examples

**Honest risks** - (1) the schedule is set from a **TabFact** probe, a Wikipedia-table register; it isolates the operation, not the prose, and P1 says so. (2) The register shares in P1 §2.2 are arena-derived and DIAGNOSTIC; every proportion above is set from the held-out severity column, and the arena column is recorded for orientation only. (3) Eight operations at one negative per positive means each type carries a few thousand pairs; if the finqa primary misses, the per-type read is the only instrument that can say whether the cause was dilution across types or a failure of the mechanism, which is why it is pre-registered rather than reported.

---

## 3. L1-C2 - AMENDMENT to R14-A4: budget the lane's evidence in body rows, not characters

**kind**: amendment | **cost**: **zero GPU**, zero new rows. Build-time CPU, already measured.

**Claim** - Because `R15_probe_P2_tokenizer.md` P2-D prices the lane's truncation at **34.93% of 1,500-character windows over `MAX_LEN` 512** measured on the incumbent `pipe` form only, while the serialization mix `R15_probe_P3_signal.md` §4 recommends is dominated by prose forms that carry roughly twice the characters at only ~1.3x better token efficiency - measured here [L1] at 32.62% over 512 for `pipe` against 53.77% for `json_records` and, weighted by P3's own recommended distribution and reserving 40 tokens for the claim, **46.52% of lane rows over budget** - and because a **body-row** budget decouples the two (at six retained body rows the same weighted figure is **7.30%**, and the row-capped re-census loses only **125 of 8,438 admitting TabFact tables, 1.48%**, while still yielding 671,460 tuples against the 25,000 the lane needs [L1]), replacing the character budget with a body-row budget and asserting that both operand rows survive the retained set will remove build-time truncation as a confound with no measurable cost in document supply, while A4's registered bars and the serving path are untouched.

**The measurement** [L1], 1,499 TabFact train tables, shipped tokenizer, median 11 body rows:

| form | 1,500-char window, mean tokens | over 512 | over 472 (claim reserved) | full table, over 512 |
|---|---:|---:|---:|---:|
| `pipe` (incumbent) | 425.83 | 32.62% | 39.23% | 36.02% |
| `markdown` | 445.75 | 36.89% | 42.56% | 40.49% |
| `narrative` | 413.46 | **24.95%** | **35.42%** | 47.10% |
| `row_prose` | 452.96 | 31.42% | 46.30% | 59.11% |
| `keyvalue` | 486.18 | 45.23% | 54.17% | 54.64% |
| `json_records` | 510.09 | **53.77%** | **67.85%** | 63.78% |
| *P3's recommended mixture* | - | *35.20%* | ***46.52%*** | *52.19%* |

Under a fixed body-row budget the ranking survives but the level collapses - share over 472 tokens:

| body rows retained | pipe | markdown | narrative | row_prose | keyvalue | json_records | **P3 mixture** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 1.33% | 1.60% | 3.33% | 9.53% | 7.53% | 17.07% | **7.30%** |
| 8 | 4.93% | 5.67% | 11.07% | 27.60% | 21.53% | 41.47% | **20.48%** |
| 10 | 10.73% | 14.33% | 23.53% | 45.80% | 37.47% | 61.53% | **35.10%** |

And the row budget is nearly free in the resource E6 says is scarce [L1], TabFact train, census predicates reused verbatim:

| body rows retained | admitting tables | constructible tuples | median tuples/table |
|---:|---:|---:|---:|
| 4 | 8,091 (95.9%) | 277,214 | 24 |
| **6** | **8,313 (98.5%)** | **671,460** | 60 |
| 8 | 8,390 (99.4%) | 1,120,338 | 110 |
| full (= banked census) | 8,438 | 1,121,632 | 110 |

**The amendment.**

1. **Serialize a fixed budget of body rows, not a character prefix.** Recommended **6**, which puts every form except `json_records` under 10% over-budget and the mixture at 7.30%. The retained rows are chosen to include both operand rows, and the tuple is dropped if they cannot both be retained - this is the assertion P2-D asks for, made enforceable
2. **Re-weight P3's form distribution against the token cost, not the character length.** `json_records` at 15% contributes 10.2 of the mixture's 46.5 points at full window and 2.6 of its 7.3 at six rows; `narrative` is the cheapest prose form on both metrics. Recommended re-weighting: `row_prose` 30 / `narrative` 30 / `pipe` 15 / `keyvalue` 10 / `markdown` 10 / `json_records` 5, which reads **5.82%** over budget at six rows. P3's independent justification for the mixture - a grounding library receives evidence in whatever form the retriever emits, and the mix contains 92,830 table-marked rows of which 99.7% are pipe-delimited and **zero** are prose-serialized - is carried verbatim and unchanged; only the weights move, and they move on an in-domain token measurement, never on the arena profile
3. **Re-run the admitting census under the chosen budget before the size is fixed.** Free, CPU, already run for TabFact train. **KILL / re-plan if the row-capped pool across TabFact train and FEVEROUS train admits fewer than 12,500 tables**, which is what 25,000 tuples at P3's per-table cap 2 requires. TabFact train alone supplies 8,313 at six rows and FEVEROUS train adds 7,990 at full table, so the clause is expected to pass with margin - it is registered so a build that changes the budget cannot silently starve the pool
4. **Report the realised token distribution per form in the build manifest**, so the 7.30% is audited rather than assumed

**Killgate** - **already run in full** [L1]: clause (a) the character budget must actually be form-dependent - measured 24.95% to 53.77% over 512 across six forms, a 2.2x spread; clause (b) a row budget must fix it - measured 1.33% to 17.07% at six rows; clause (c) the row budget must not starve the pool - measured 98.5% of admitting tables and 27x the required tuple supply retained. **DOWNGRADE to a construction footnote** if a build on the real corpus reads the mixture below 15% over budget at the full character window, in which case P2-D's assertion alone suffices and the row budget is not worth the register shift it introduces.

**Bar** - this amendment **claims no finqa movement of its own** and rides A4's registered arm; A4's PRIMARY and every HOLD stand as written. Its verification is a build-manifest clause, not a read: **realised share of lane rows whose serialized evidence exceeds `MAX_LEN` minus the claim length must be under 10%, and the realised share of tuples dropped for operand-row loss must be reported.** A build reporting either quantity above its clause is rebuilt, not adjudicated.

**Declared cost, stated plainly** - a six-row body budget discards roughly 45% of the rows of the median 11-row table, so the lane's evidence documents are systematically shorter than the tables they come from. That is a real register shift and it is the price of the clause. The alternative is not "full tables": it is `longest_first` truncation removing the **tail** of a third to a half of the lane's evidence uncontrolled, which is `R15_probe_P2_tokenizer.md` §8's build-time correctness bug. Choosing which rows survive is strictly better than discovering which ones did not.

**Engaging H119, which is not at issue here** - R12-H119 was a frozen-weights **serving wrapper** applying numeric string canonicalization to claim and evidence before tokenization, REFUTED in both directions (strip +0.00245 / -0.00098 against a +0.003-on-both bar; finqa clearing +0.010 on 2 of 4 draws against a 3-of-4 bar and sign-disagreeing inside the H105 pair at -0.0163 / +0.0178; add -0.00039 / -0.00002). **No candidate in this lens applies any transformation to any input at serving time.** `src/groundrails/semantic_ov.py` is untouched, `MAX_LEN` is untouched, the 1,500-character window is untouched, the max-over-windows read is untouched, and the bytes the deployed function receives are byte-identical to today's. A build-time decision about how many rows of a training table to serialize is applied to zero serving inputs and cannot be arena-fitted preprocessing. The interaction with **R14-A2** is real and is the one thing to sequence: if A2 Stage 1 adopts the 1024 read, the row budget is re-derived at 1024 and rises; if A2 kills, the six-row budget at 512 is binding. A4 must not build before A2 Stage 1 reports, or it builds at 512 and is re-checked.

---

## 4. L1-C3 - NEW: relational sub-block (cross-header binding and ordering)

**kind**: new-hypothesis, registrable as a declared sub-block of R14-A4 or standalone | **cost**: **~0.1 GPU-h marginal** as a sub-block (gate already spent on two checkpoints; 0.1 for the post-arm read); ~13 GPU-h standalone.

**Claim** - Because the shipped model resolves a **row** label at AUROC 0.9936 and a **column** header at 0.5179 on byte-identical templates over the same evidence with every asserted number verbatim present, and reads ordering at 0.5076 - all three replicating on the admitted H108 lane checkpoint at 0.9943 / 0.5276 / 0.5184 [L1] - and because R14-A4's census enumerates only within-column tuples so its lane cannot present a cross-header contrast at all, while `R14_evidence_E4_items.md` prices a single cross-line-item binding error at **13.4% of all finqa discordance**, adding a sub-block of cross-header binding and ordering pairs at ~15% of lane rows will lift held-out `bind_col` AUROC above **0.70** and `compare` above **0.65** from banked baselines of 0.5179 and 0.5076, while `bind_row` holds at or above **0.95**, held-out scale/unit AUROC holds at or above 0.80, and A4's registered arena holds are unbroken.

**Why this is a new hypothesis and not an amendment.** Both families assert numbers that **are present in the evidence**, so neither participates in the P(label 0 | absent) shortcut R14-A4 exists to remove, and neither dilutes it. This is exactly the argument `R15_probe_P1_anatomy.md` §7 uses to exclude count-aggregation from the lane, and it applies here with equal force. The sub-block therefore needs its own justification and gets one that stands with the arena deleted: **a grounding library that resolves which row a value belongs to but not which column, and that cannot order two quantities both printed in front of it, mis-certifies any tabular evidence with more than one measure per entity.** That is a product defect independent of any benchmark. The arena figure (E4's 13.4%) is corroboration and enters no threshold.

**Construction.** Per table, one named row, two numeric columns:

- **Cross-header binding** - `"The {colx} of {ka} is {v}."` Positive: `v` is `ka`'s cell under `colx`. Negative: `v` is `ka`'s cell under a different column `coly`, byte-identical claim otherwise. **Digit-length parity binding, and this is the correction to the probe**: the probe's negatives are uncontrolled (section 1a), so the sub-block must draw `coly` from columns whose values match `colx`'s digit length, or reject the tuple. Without it the sub-block is solvable on digit length and re-imports the failure class P2-B's KILL exists to catch
- **Ordering** - `"The {col} of A is greater than the {col} of B."` against the reversed claim, both operands printed, no computation. **Magnitude-gap stratification binding**: emit equal thirds at relative gaps under 10%, 10-100%, and over 100%, and report per-stratum AUROC. P4's Instrument B decodes the ordering direction off the frozen trunk's `[CLS]` at **accuracy 1.000 from 200 examples** with three controls holding, so if any stratum fails to move, the failure is not representational and must be recorded as such
- **Non-regression rows** - a declared 20% of the sub-block is `bind_row` form at 0.9936 today, carried so the sub-block cannot install header sensitivity by destroying row sensitivity

**Killgate** - **already run, on two checkpoints, and passing** [L1]. The gate is the probe: **LICENSE** requires `bind_col` <= 0.60 and `compare` <= 0.60 (there is a hole to fill) **and** `absent_ctl` >= 0.90 (the read is intact and the probe is measuring binding, not breakage). Measured 0.5179 / 0.5076 / 0.9889 on H105 draw 1 and 0.5276 / 0.5184 / 0.9950 on H108 lane draw 1. **The clause that would have killed it - `bind_row` at chance - is the opposite of what was measured**, and that reading is what removes P4-1's 35% allocation rather than this candidate. One remaining free clause before build: certify on a 500-tuple CPU sample that digit-length-matched cross-column pairs are constructible at the sub-block's size; **KILL below 60% constructibility**, since an unmatched sub-block is not registrable.

**Bar** - **sub-block form (recommended)**: rides A4's arm at ~15% of rows, claims no separate finqa movement, and A4's PRIMARY (finqa 2-draw mean >= 0.6933, sign agreement) and every HOLD stand as registered (arena mean >= 0.7031, pubmedqa >= 0.5463, `gold_full` >= 0.8414, RAGTruth non-EN >= 0.82, no subset more than 0.06 below its paired control and none < 0.55, log-length residualization at >= 50% of magnitude). Pre-registered ceiling-blind mechanism reading on both draws: **`bind_col` >= 0.70 AND `compare` >= 0.65 AND `bind_row` >= 0.95 AND held-out scale/unit AUROC >= 0.80.** **Standalone form**: its own paired arm at finqa 2-draw mean >= **0.6733** (+0.040 over the 0.6333 paired control, sign agreement on both H126-paired draws), same holds. The +0.040 rather than A4's +0.060 is deliberate and is argued, not assumed: this sub-block does not touch the absent-number shortcut, so its finqa route is narrower than A4's by construction.

**Bar amendment to R14-A4, argued with evidence.** A4's ANTI-GAMING clause is "in-domain held-out H108-style **present-value near-miss** AUC must not fall below the clean-recipe value". This sub-block trains present-value near-miss discrimination directly, so the clause stops being an independent instrument the moment the sub-block is adopted. **Amendment: if L1-C3 rides A4's arm, the anti-gaming held-out set must be re-constituted table-disjoint AND operator-disjoint from the sub-block** - H108-style *value* corruptions (digit perturbation, scale-word swap) over tables the sub-block never used - **and `bind_row` >= 0.95 is added to the clause as a second, independent non-regression read.** Without this amendment A4's binding anti-gaming clause is trained on and the arm cannot be adjudicated. This is the only bar amendment this lens requests.

**Honest risks** - (1) The two families are measured on TabFact only; whether header binding transfers to 10-K prose where the "column" is a line-item label in running text is unmeasured, and L1-C4 is the instrument that would say. (2) Ordering claims are a claim *shape* the mix barely contains - 6.12% of the H108 lane's claims carry comparative language [L1] - so the sub-block shifts the lane's claim-shape distribution as well as its supervision; the arena-mean HOLD is the instrument and it is not a formality. (3) The sub-block competes for the same rows as three other sub-block proposals in the field; see section 6.

---

## 5. L1-C4 - MEASUREMENT: the natural-derivation transfer instrument

**kind**: measurement | **cost**: **~0.2 GPU-h** plus one judge pass over ~500 candidates; no training, no new arm.

**Claim** - Because R14-A4's lane is **entirely synthetic and entirely Wikipedia-tabular** - `R15_probe_P3_signal.md` §7 measures VitaminC's derivable supply at ~2,800 true candidates after a shuffle control puts the coincidence floor at 2.58% against a real rate of 5.10%, WiCE's at ~175, and records that no legal corpus supplies the financial register at all - while every A4 bar is finqa, an arena subset that `R14_synthesis.md` prices at **20 negatives with median rank 62.5**, where two rank swaps are worth 0.10 - the record has no instrument that separates "the lane taught derivation checking" from "the lane taught its own construction", and banking a verified natural-derivation held-out set before the arm runs supplies one for a fraction of one draw's cost, with no effect on any registered bar.

**What is banked**, once, and re-read on every subsequent draw:

1. **VitaminC verified subset** - draw 500 of the 5,837 absent-and-two-operand-derivable rows, verify by judge that the claim's asserted number is genuinely the arithmetic the evidence implies (the shuffle control says roughly half will not be), and bank the survivors with their labels. P3's disqualification stands and is the reason this is a probe and not lane mass: VitaminC's label adjudicates the **revision**, not the arithmetic
2. **WiCE numeric slice** - ~175 real derivations at essentially one row per document, the best diversity per row of any source measured, over real long-form web prose. Record P3's caveat: `partially_supported` dominates the numeric slice (106 of 168 claim rows), so the binary collapse is lossiest exactly where the derivations live
3. **The standing frozen-weights reads already on disk** - P1's ten-type quad set, the 2,000 H133 triples (AUROC b vs c 0.4924, digit-prefix-alone 0.4871, digit-length-alone 0.4992 [L1]), and this lens's four-arm binding probe (0.9936 / 0.5179 / 0.5076 / 0.9889 on H105 d1, replicated on H108 d1)

**Killgate** - **CPU, free**: the verified VitaminC subset must survive its own coincidence floor. **KILL the VitaminC leg if fewer than 150 of 500 candidates verify** - below that the instrument is smaller than its own noise and only the WiCE leg and the synthetic reads remain, which is a weaker but still legal instrument. **~0.2 GPU-h, frozen H105 draw 1, in-domain, arena-free**: read the verified set to establish the pre-arm baseline. **NO-READ** if the baseline AUROC on natural derivations is already above 0.65 - the model discriminates derivations in the wild and the R14 diagnosis is narrower than the synthetic probes suggest, which is a finding that must be escalated before any lane is built rather than discovered after.

**Bar** - this is a measurement and sets no bar on any arm. Its decision rule, pre-registered before any A4 draw: **the natural-derivation AUROC is reported alongside every A4 read, and a finqa admission accompanied by no movement on the natural set is recorded as "construction learned, transfer unproven" rather than as a clean admission.** It cannot kill A4 - a synthetic-to-natural gap is not a refutation of a synthetic lane - and it is registered precisely so that distinction is made in writing before the number exists, not argued after.

**Why this is worth 0.2 GPU-h now** - three of the register's six blocks are finqa-primary, the field's finqa arithmetic is not additive and must not be summed, and the campaign has twice been bitten by a construction-time regularity becoming the learned rule (P(0 | absent) = 0.5944 in the mix, 0.9332 in the admitted lane). An instrument that reads derivation competence **off the training construction entirely** is the cheapest defence against a third instance.

---

## 6. Interaction, ordering, and the oversubscription the author must arbitrate

| id | kind | first-decision cost | full cost | conditional on |
|---|---|---|---|---|
| **L1-C2** | amendment to A4 | 0 (already measured) | 0 marginal | A2 Stage 1's read, for the token budget |
| **L1-C1** | amendment to A4 | 0 (gate run) + 0.1 GPU-h replication | ~0.1 marginal | nothing - it is the build spec |
| **L1-C3** | new, sub-block or standalone | 0 (gate run on two checkpoints) | ~0.1 marginal / ~13 standalone | A4's build existing |
| **L1-C4** | measurement | 0 (CPU) + 0.2 GPU-h | ~0.2 | nothing - runnable today |

**Spend order**: C2 first, because it is free and it is a correctness fix that silently dilutes every other candidate's rows if it is not applied. Then C1, whose 0.1 GPU-h replication is also P1's own falsifier. Then C4's baseline, which must be banked *before* a draw exists. C3's gate is already spent.

**Which amendments are load-bearing, and which are nice-to-have.** The lane is otherwise ready to build, and the coordinator's question is what it must wait for:

- **C2 is load-bearing and blocking.** Unfixed, roughly a third to a half of the lane's positives assert a derived value whose supporting rows are not in the encoder's input. That is not a tuning question, and no bar can be adjudicated over it
- **C1 is load-bearing and blocking.** A sum-only lane under-covers its own diagnosis by nine measured types, and the negative-family schedule is what keeps the repair from re-importing the defect (N7 cap, coarse-first weighting, byte-identical templates)
- **C3 is not blocking.** It is a distinct capability with a distinct justification and can be registered as its own arm at any time; folding it in costs A4 the independence of its anti-gaming clause unless the section-4 bar amendment is adopted
- **C4 is not blocking but must precede the first draw**, because a baseline taken after the arm is not a baseline

**Oversubscription, stated plainly.** The field's sub-block claims against A4's registered 50,000 rows now total more than 100%: L3-C1 asks 15% for the scale-identity factorial, L3-C2 asks ~20% for evidence-side minimal pairs, L3-C3's form-by-label cross halves the tuple count at fixed rows, L1-C3 asks 15% here, and L1-C1's schedule assumes the remainder is a derivation core covering eight operations. **The author must arbitrate before the build, not at adjudication.** This lens's recommendation, offered as one ordering rather than a claim: the derivation core and C2's budget are the lane; C3 and one of L3's sub-blocks are affordable at 15% each; adopting three or more sub-blocks at once produces an arm that cannot attribute any result, and the honest alternative is a larger row budget with `R15_hypotheses_L2_objective.md` L2-C1's weighting arithmetic priced against it.

**Declared interactions**:

- **C1's digit-length parity and C3's digit-length matching both deliver part of R14-A5/H134's decorrelation mechanism for free**, which is P3 §2's declaration and L3-C1's; **A4-with-these-amendments and A5 must not run in one arm**
- **C2 and R14-A2 are sequenced**, not exclusive; the budget is re-derived at whatever `MAX_LEN` A2 Stage 1 leaves standing
- **C1 and L3-C1 both touch the scale and rounding slices.** C1 assigns them 8% (hold-only) and 12%; L3-C1 proposes taking 15% from exactly those two for its factorial. They are compatible only if the factorial's rows count toward the scale and rounding shares rather than adding to them, and the two must not both claim the same 12% of rounding
- **C3's rows are present-value and therefore sit outside the lane's absent stratum entirely**, so they neither dilute nor contradict P(label 0 | absent); the realised in-lane P(0 | absent) must be reported over the derivation core only, with the sub-block's rows accounted separately

---

## 7. Closed lines engaged

- **H119 read-time numeric canonicalization - REFUTED, not at issue.** No candidate here applies any transformation to any input at serving time; section 3 states the four non-transfer points and the binding constraint accepted.
- **H117 margin / contrastive pair loss - REFUTED, and nothing here is pairwise.** Every row in every candidate carries its own absolute label and enters BCE. No candidate masks a row out of BCE and no candidate introduces a term whose gradient depends on another row's score. The absolute score comparability the windowed decomposed-min read requires is preserved by construction.
- **GroupDRO curriculum - REFUTED, and the schedule is not one.** C1's negative-difficulty split (50 / 30 / 20 coarse to near-miss) is a **mixture proportion**, applied uniformly across the run in flat permutation. There is no ordering, no worst-group weighting, no `q`, no stratified sampler. P4-5's instruction to mix the strata rather than order them is carried verbatim.
- **Forced subset balance - REFUTED, not proposed.** No candidate equalises DANN group representation; C1 and C3 change what is inside one lane, and the lane's own group tag is assigned per A4's build.
- **Training on RAGBench corpora and their financial derivatives - FORBIDDEN, and untouched.** Every corpus named in this lens is TabFact, FEVEROUS, InfoTabS (dead at 48 admitting tables), VitaminC, WiCE and RAGTruth train. ConvFinQA, MultiHiertt, FinanceBench, FinTabNet, TAT-HQA and every RAGBench subset appear nowhere as data. FinQA and TAT-QA appear only as arena analysis and as literature context in P4 §3.
- **Head fusion, token-head transfer, weight averaging - CLOSED**, and no candidate proposes a head, a fusion or an average. Every candidate is data-only except C4, which is a read.

---

## 8. Falsifiers

Each is a claim this lens makes that can be wrong, with its price:

- **C1's type-uniformity premise** is measured on one checkpoint family. It is falsified if the banked quads read any tier-1 type above 0.60 on `R10-H108-lane-draw1`. **~0.1 GPU-h, and it is C1's own killgate clause 2** - the cheapest unspent measurement this lens names.
- **C1's coarse-first weighting** rests on P1 §3.1, whose per-type order-of-magnitude subsamples are 16 to 97 quads. A rebuild forcing every negative to a decade error, 500 per type, settles it for **~0.1 GPU-h** and is worth taking before the 50 / 30 / 20 split is frozen.
- **The digit-copy account** predicts that a type constructed so the *wrong* value carries the source cell's digit prefix and the correct value does not reads AUROC **below** 0.50. Section 1b ran the weak version on `ratio` and it did not fire, at a prefix advantage of 0.05 digits - far too small to test. The clean version costs **~0.05 GPU-h** and is still owed.
- **C2's supply claim** is measured on TabFact train only. It is falsified if the row-capped re-census over FEVEROUS train loses materially more than TabFact's 1.48% of admitting tables. **Free, CPU**, and it is C2's registered clause 3.
- **C3's transfer claim** - that cross-header binding learned on Wikipedia tables reaches line-item binding in financial prose - is **unmeasured and is the candidate's largest risk**. C4 is the instrument; nothing cheaper exists.
- **The whole lens's premise** is that the binding hole is on the header axis. It is falsified if a controlled `bind_col` probe - negatives drawn from digit-length-matched columns, which the current probe does not enforce - reads materially above 0.60. **~0.15 GPU-h on the existing harness**, and it should be taken before C3's rows are budgeted, because a controlled probe reading 0.70 would mean the sub-block is chasing a magnitude artefact rather than a header hole.

---

## 9. What this lens does not claim

- **It does not claim any candidate reaches the arena.** C3's standalone form is the only one carrying a finqa primary of its own, and it is set at +0.040 with its narrower mechanism stated as the reason.
- **It sets no bar from an arena statistic.** Every proportion in C1's schedule comes from the held-out severity columns of P1's TabFact probe; the register-share column is arena-derived, labelled DIAGNOSTIC, and enters no threshold. C2's form weights come from an in-domain token measurement. C3's bar thresholds come from banked held-out probe baselines.
- **It requests exactly one bar amendment** - re-constituting A4's anti-gaming held-out set if L1-C3 rides A4's arm - and declines to move A4's finqa primary, its holds, its confound clause or its pilot kill.
- **It does not measure whether the schedule's shares are optimal.** They are a defensible allocation from a severity table, frozen before the build so they are not tuned; C1's per-type mechanism read is what would say afterwards which shares were wrong.
- **It does not price C3's standalone arm against A4's.** Both are ~13 GPU-h and the choice between them is the author's; the sub-block form is recommended because its marginal cost is 0.1.
- **The `bind_col` probe is uncontrolled for digit length and magnitude**, which makes its chance reading conservative but leaves the clean version owed (section 8).

---

## 10. Reproduction

```bash
cd /home/lab/workspace/private/ai-assistants/groundrails

# (a) present-value binding and ordering probe, ~0.15 GPU-h per checkpoint
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  uv run python experiments/grounding-semantic/R15_L1_bindprobe.py
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 R15_CKPT=R10-H108-lane-draw1 \
  uv run python experiments/grounding-semantic/R15_L1_bindprobe.py

# (b) digit-prefix exploitability per type, CPU, seconds
uv run python tmp/R15_L1_lcdp.py

# (c) serialization token cost and the body-row budget, CPU, ~4 min
uv run python tmp/R15_L1_serialtok.py
uv run python tmp/R15_L1_serialtok2.py
uv run python tmp/R15_L1_rowcap_census.py     # ~6 min, reuses R14_gate_H133_census.py verbatim
```

Outputs: `R15_gate_L1_binding.json`, `R15_gate_L1_binding_R10-H108-lane-draw1.json`, `R15_L1_bindprobe_pairs.parquet`, `R15_gate_L1_lcdp.json`, `R15_gate_L1_serialtok.json`, `R15_gate_L1_rowcap.json`, logs in `logs/R15_L1_*.log`. Inputs: `models/R9-H105-mmbert-dann-clean`, `models/R10-H108-lane-draw1`, `data/external/datasets/dataset-tabfact.zip`, `R14_H133_triples.parquet`, `R15_P1_typeprobe_quads.parquet`, `R10-H108_pairs.parquet`. No tracked artifact was modified; the row-cap re-census reproduces the banked `R14_gate_H133_census.json` figures exactly in its `full` configuration (8,438 admitting tables, 1,121,632 tuples), which is the check that its predicates are the registered ones.

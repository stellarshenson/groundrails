# R15 Probe P1 - defect anatomy by derivation type

**Question**: the R14-H133 gate measured the shipped cross-encoder at chance on correct-vs-wrong derivations (AUROC 0.4924). H133 built **sum triples only**. What do "derived numbers" actually look like where the model fails, and does the defect hold across derivation types - or is it a sum artefact?

**Answer, first**: it is not a sum artefact. The defect is **uniform across every genuinely combinatorial operation** - sum, difference, mean, ratio, percent-change, product, count-aggregation and date arithmetic all sit at AUROC 0.484-0.526 against wrong-operand values, and all of them stay at chance even when the wrong value is off by more than an order of magnitude. The two operations where the model does discriminate - unit/scale conversion (0.8755) and rounding (0.7049) - are exactly the two that **preserve the digit string of the source cell**, and their discrimination is a digit-prefix copy detector, measured here at Pearson +0.43 and +0.33 between the digit-prefix advantage and the score advantage. The model has one working numeric instrument, digit copying, and no arithmetic at all.

**Discipline**: arena rows (`R12-H121_gateA_scores.parquet`) are ANALYSIS ONLY - the taxonomy below is diagnostic and **must not set lane proportions**. Every severity number that feeds the lane recommendation was measured on held-out TabFact tables, `table_id`-disjoint from the clean-mix train split, with zero arena and zero gold. Polars throughout. GPU spend for the three new probes: **~0.2 GPU-h on card 0** (main 4,139 quads, top-up 869 quads, shown-work 1,600 pairs x 2 templates).

**Artefacts produced**

| file | content |
|---|---|
| `R15_P1_typeprobe.py` / `.json` / `_quads.parquet` | 8-type quad probe, 4,139 quads, frozen H105 draw 1 |
| `R15_P1_typeprobe_topup.py` / `.json` / `_quads.parquet` | count-aggregation and date-arithmetic, absence rule relaxed |
| `R15_P1_shownwork.py` / `.json` | bare vs operand-quoting claim template, 1,600 pairs |
| `logs/R15_P1_typeprobe.log`, `..._topup.log`, `..._shownwork.log` | run logs |

---

## Summary

- **The defect is type-uniform.** AUROC(correct derivation vs wrong-operand derivation) on frozen H105 draw 1: sum 0.5067, difference 0.4994, mean 0.5257, ratio 0.5121, percent-change 0.4861, product 0.5165, count-aggregation 0.4842, date-arithmetic 0.5014. Ten types, 5,008 quads, and not one of them clears 0.53
- **It is not a precision problem, it is a total absence of magnitude sense.** Restricting to quads where the wrong value differs from the correct one by **at least one order of magnitude**, the correct value is still scored higher only 48.8% of the time (difference), 49.5% (percent-change), 47.4% (product), 55.3% (ratio). A model with any coarse magnitude estimate would beat this
- **Two exceptions, and they are not arithmetic.** Unit/scale conversion reads 0.8755 and rounding 0.7049. In both, the correct value shares a long digit prefix with the named cell (mean longest common digit prefix 3.39 and 3.05) while the wrong value does not (0.60 and 0.61); across all other types both sides sit at 0.07-0.84. Within those two types the per-item score advantage tracks the per-item digit-prefix advantage at Pearson **+0.4300** and **+0.3304**
- **Rounding is the most heavily penalised derivation of all** - correct rounded values score **0.1763**, below sum (0.2326) and mean (0.2549), with only 9.4% clearing 0.5. This is the shape H108's ~45k unit/period/scale corruption negatives most directly taught the model to reject, and it is the commonest benign numeric transformation in financial prose
- **Showing the working is a verbosity effect, not comprehension - and it is a gaming surface.** Quoting both operands in the claim lifts the correct derivation from 0.3338 to 0.4815 (+0.148) and lifts the wrong-operand value by the identical amount, 0.3334 to 0.4838. AUROC moves 0.5010 to 0.4973. **Binding consequence for the R14-A4 lane: the claim template must be held byte-identical between the positive and the negative member of every pair**, or the model clears the lane by counting quoted operands
- **The register's operation mix, read verbatim from 60 arena items and classified value-driven over all 1,031 finqa+tatqa sentences**: difference and percent-change/ratio dominate finqa; unit/scale conversion and count-aggregation dominate tatqa. Sum - the only operation H133 measured - is 7.1% of finqa's and 6.4% of tatqa's absent numerals
- **A quarter of the "absent number" signal is a measurement artefact.** The literal constant **100**, produced by the `x 100` step of a percent formula, is an absent numeral in 46 of 250 finqa deciding sentences (109 numeral instances, 25.5% of all finqa absent numerals) and 19 of 250 tatqa (49 instances, 22.6%). Year ranges parsed as negative numbers add 3 sentences each. The 0.610 / 0.946 shortcut purity that licenses R14-A4 is measured by a detector that counts these

---

## 1. Method

**The unit.** The shipped read is sentence score = max over windows, response score = min over sentences. The **deciding sentence** - the argmin - therefore *is* the response score, so per-type severity measured on deciding sentences is severity in the units the arena AUROC is computed in. Frequencies are reported at two grains: deciding sentences (250 per subset) and all scored sentences (563 finqa, 468 tatqa).

**Absence.** A numeral in a claim is *absent* if no numeral in the union of that response's retrieved windows matches it within 0.2% relative tolerance. The tolerance is deliberate - it prevents a 2-decimal rounding of a present value from being counted as a new assertion.

**Classification is value-driven, not cue-driven.** For each absent numeral the classifier searches for an arithmetic explanation over an operand pool, in a fixed precedence order (scale → percent-scaling → percent-change → ratio → difference → sum → mean → product → percent-of → count). Three passes:

1. **Pass 1** - operand pool = the numerals in the same sentence that *are* present in evidence. This is the strong pass: a large-language-model answer quotes the operands it used
2. **Pass 2 (chain)** - values explained in pass 1 are readmitted as operands and the residual is retried; anything newly explained is a **depth-2 chain**
3. **Pass 3 (wide, deciding sentences only)** - the residual is retried against the whole evidence numeral set (capped at 120 values). This resolves bare assertions that state a result without quoting operands, and it is an **upper bound**: with 120 candidate values, some attributions are arithmetic coincidence

Two artefact classes are separated before anything else: the constant **100** appearing inside a `x 100` percent formula, and four-digit years captured out of a `2017-2019` range.

**The probe.** Held-out TabFact test+validation tables, `table_id`-disjoint from the train split used by the clean mix and by `R10-H108_data.tabfact_positives()` - 3,391 tables. Per table and per type, four claims over the same evidence:

- **(a) verbatim** - the asserted value is a cell of the table
- **(b) correct** - arithmetically correct, absent from the table
- **(c) wrong-operand** - same template and same named operands, value computed from a *different* pair of rows
- **(d) wrong-operator** - same named operands, value computed by a *different* operation

(a) is the ceiling, (b) is what the product must accept, (c) is the H133 axis, (d) is new here. Every asserted value in (b), (c), (d) is verified absent from the evidence string before the quad is admitted.

**Two honest limits of the probe.** First, TabFact is a Wikipedia-table register, not a financial-filing register - it isolates the *operation*, not the prose. Second, the absence requirement makes count-aggregation and date-arithmetic almost unconstructible (0 and 139 quads of a 500 target), because both produce small integers that appear somewhere in any table; the top-up drops the absence rule for those two types only and therefore measures verification ability rather than shortcut presence. Both are flagged in the tables below.

---

## 2. What a "derived number" is in this register

### 2.1 Incidence

| | finqa | tatqa |
|---|---|---|
| responses / scored sentences | 250 / 563 | 250 / 468 |
| deciding sentences carrying >= 1 absent numeral (artefacts excluded) | **154 (61.6%)** | **76 (30.4%)** |
| all sentences carrying >= 1 absent numeral | 273 (48.5%) | 136 (29.1%) |
| absent numerals, deciding sentences (artefacts excluded) | 311 | 162 |
| absent numerals, all sentences (artefacts excluded) | 622 | 296 |
| mean deciding-sentence score, with absent numeral | 0.4159 | 0.2888 |
| mean deciding-sentence score, without | 0.4363 | 0.5441 |
| gold-support rate of the absent-numeral deciding sentences | 0.79 - 1.00 by type | 0.80 - 1.00 by type |

The finqa figure is consistent with E4's 362/563 = 64.3% under a looser presence rule; the difference is the tolerance and the artefact exclusion.

**tatqa carries the cleaner signal.** Its absent-numeral deciding sentences score 0.2888 against 0.5441 for the rest - a 0.255 separation. finqa's equivalent gap is 0.02, because finqa's argmin is already depressed everywhere: 96 of its 250 deciding sentences carry no absent numeral at all and still average only 0.4363.

### 2.2 Type frequency

Numeral-level shares over **all** scored sentences, artefacts excluded. This is the register census; it is DIAGNOSTIC and does not set lane proportions.

| type | finqa n | finqa % | tatqa n | tatqa % | what it looks like |
|---|---:|---:|---:|---:|---|
| ratio / share-of-total | 122 | 19.6 | 31 | 10.5 | "interest rate derivatives made up approximately 44.2% (26,363 / 59,677)" |
| percent-change / growth | 97 | 15.6 | 39 | 13.2 | "Percent Change = ((572.93 - 527.37) / 527.37) x 100 = 8.62%" |
| difference / period change | 84 | 13.5 | 45 | 15.2 | "a decrease of 4,079 million (23,280 in 2016 to 19,201 in 2017)" |
| scale / unit conversion | 69 | 11.1 | 60 | **20.3** | table in thousands, claim in dollars: "5,092,000 + 5,078,000" for cells 5,092 and 5,078 |
| sum / total aggregation | 44 | 7.1 | 19 | 6.4 | "225 + 216 + 217 = 658 million" |
| count-aggregation | 39 | 6.3 | 47 | **15.9** | "revenue exceeded 500,000 thousand in two years: 2017 and 2018" |
| mean / average | 18 | 2.9 | 19 | 6.4 | "(10,806 + 9,022 + 8,737) / 3 = 9,522.33 thousands" |
| percent-scaling (x100 / /100) | 16 | 2.6 | - | - | "0.0862 x 100 = 8.62%" as a separate assertion |
| product | 13 | 2.1 | - | - | "108.11 per share x 427,195,037 shares" |
| unresolved | 118 | 19.0 | 36 | 12.2 | see 2.3 |

Deciding-sentence counts (the units the response AUROC is computed in) rank the same way: finqa difference 64, ratio 36, percent-change 32, sum 19, scale 18, count 11, mean 9, percent-scaling 8, product 4; tatqa difference 34, scale 17, count 15, percent-change 14, ratio 10, sum 8, mean 7.

**Sum - the only operation the H133 gate measured - is 7.1% of finqa's and 6.4% of tatqa's absent numerals.** The gate's verdict happened to generalise, but it was measured on the register's sixth-commonest operation.

### 2.3 The unresolved residual

19.0% of finqa's absent numerals resist explanation under the sentence-local operand pool. Re-run against the whole evidence numeral set (deciding sentences only, 65 finqa numerals and 22 tatqa), the residual resolves as:

| resolves to | finqa numerals | tatqa numerals |
|---|---:|---:|
| difference | 20 | 12 |
| ratio | 12 | - |
| mean | 8 | 3 |
| percent-change | 6 | - |
| sum | 5 | 1 |
| question-borrowed threshold | - | 1 |
| still unresolved | **14** (11 sentences) | **5** (5 sentences) |

The residual is therefore **the same operations stated bare**, without quoting the operands - "The total sales as of December 31, 2014 was 6,957 million", "The revenue from South America decreased by 5,565,000 from 2017 to 2018". The genuinely irreducible remainder, 11 finqa deciding sentences of 250, is dominated by values that live in a **performance graph** rather than in retrieved text - "the five-year cumulative return for Intel Corporation was 114%", "the S&P 500 index return was 105%". Those are not derivation failures at all; no scorer of any size can verify them from the retrieved windows.

### 2.4 The bare-assertion shape is the worst-scoring shape

| absent numerals in the sentence | finqa deciding, n | mean score | finqa all sentences, n | mean score |
|---:|---:|---:|---:|---:|
| 1 | 64 | **0.3059** | 106 | **0.3768** |
| 2 | 53 | 0.4946 | 87 | 0.5338 |
| 3 | 16 | 0.4927 | 34 | 0.4965 |
| 4+ | 21 | 0.4739 | 46 | 0.5093 |

A sentence that states one derived result and nothing else scores far below one that shows its working. Section 5 shows this is a lexical-overlap effect and not comprehension - which makes it a lane-design hazard, not a lane-design opportunity.

### 2.5 Chain depth

Depth-2 chains (a value derived from a value that was itself derived) appear in **15 of 250 finqa** and **13 of 250 tatqa** deciding sentences. Deciding sentences carrying three or more *distinct* derivation types: 12 finqa, 7 tatqa. This is the honest ceiling on the multi-step share - roughly 5-6% of deciding sentences.

---

## 3. Failure severity per type (frozen H105 draw 1, arena-free)

n = 500 per type except where noted. Every value in (b), (c), (d) verified absent from the table.

| type | n | (a) verbatim | (b) correct | (c) wrong-operand | (d) wrong-operator | AUROC b vs c | AUROC b vs d | gap a-b | (b) > 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sum | 500 | 0.9247 | 0.2326 | 0.2312 | 0.2290 | **0.5067** | 0.5072 | 0.6920 | 3.8% |
| difference | 500 | 0.9197 | 0.3489 | 0.3499 | 0.3735 | **0.4994** | 0.4319 | 0.5708 | 9.6% |
| mean | 500 | 0.9235 | 0.2549 | 0.2401 | 0.2719 | **0.5257** | 0.4648 | 0.6686 | 6.6% |
| ratio | 500 | 0.9160 | 0.3010 | 0.2951 | 0.2172 | **0.5121** | 0.6705 | 0.6150 | 8.6% |
| percent-change | 500 | 0.9065 | 0.4223 | 0.4269 | 0.3955 | **0.4861** | 0.5746 | 0.4843 | 22.6% |
| product | 500 | 0.9189 | 0.2683 | 0.2605 | 0.2144 | **0.5165** | 0.6079 | 0.6506 | 7.6% |
| scale / unit | 500 | 0.9084 | **0.4965** | 0.1872 | 0.3383 | **0.8755** | 0.7059 | 0.4118 | 51.0% |
| rounding | 500 | 0.9055 | **0.1763** | 0.0724 | 0.1157 | **0.7049** | 0.5502 | 0.7292 | 9.4% |
| count-aggregation † | 369 | 0.9493 | 0.4128 | 0.4168 | 0.4154 | **0.4842** | 0.4914 | 0.5364 | 12.2% |
| date arithmetic † | 500 | 0.7958 | 0.3915 | 0.3923 | 0.3369 | **0.5014** | 0.6122 | 0.4043 | 23.4% |

† absence rule relaxed - these two types produce small integers that almost always appear somewhere in a table, so they measure verification ability rather than shortcut presence. Under the strict absence rule, count-aggregation was **0 of 500** constructible and date arithmetic **139 of 500** (that stricter date-arithmetic sample reads AUROC b vs c 0.4948, the same verdict).

**Readings.**

- **Verbatim is near-saturated and type-independent** - 0.906 to 0.965, AUROC(a vs b) 0.807 to 0.988. Literal presence remains a near-perfect discriminator. Nothing about the read or the window is broken
- **Every combinatorial type is at chance on the wrong-operand axis.** The R14-H133 number, 0.4924, was not special to sum
- **Wrong-operator is not uniformly better and is sometimes systematically inverted.** For *difference*, the wrong-operator value (the sum of the same two operands) scores **higher** than the correct difference - AUROC 0.4319, a reliable anti-signal. For *mean* it is 0.4648. Where the wrong-operator AUROC does exceed chance (ratio 0.6705, date arithmetic 0.6654, product 0.6079) the wrong value is a *magnitude* outlier relative to the register of the column, not a logically rejected one
- **Percent-change and count-aggregation are the least-penalised correct derivations** (0.4223 and 0.4128, 22.6% and 12.2% above 0.5) and **rounding is the most penalised** (0.1763). The ordering is not the ordering of arithmetic difficulty - it is the ordering of how much the value's surface form resembles a manufactured near-miss negative

### 3.1 Magnitude sense: absent

Restricting to quads where the wrong-operand value is off by at least one order of magnitude:

| type | quads with >= 1 decade error | correct scored higher |
|---|---:|---:|
| difference | 82 | 48.8% |
| percent-change | 97 | 49.5% |
| product | 38 | 47.4% |
| ratio | 47 | 55.3% |
| scale / unit | 29 | 79.3% |
| rounding | 16 | 93.8% |

The four combinatorial families are at chance **even when the wrong answer is ten times too large**. Whatever the encoder is doing with these claims, it is not comparing magnitudes. The two digit-preserving families hold their discrimination, as the next section explains.

---

## 4. The mechanism: a digit-prefix copy detector

Longest common digit prefix (LCDP) between the asserted value and the named entity's own cell value, per type:

| type | LCDP of correct (b) | LCDP of wrong (c) | correct has longer prefix | AUROC b vs c | Pearson(delta LCDP, delta score) |
|---|---:|---:|---:|---:|---:|
| scale / unit | **3.39** | 0.60 | 98.0% | 0.8755 | **+0.4300** |
| rounding | **3.05** | 0.61 | 91.2% | 0.7049 | **+0.3304** |
| mean | 0.84 | 0.62 | 29.2% | 0.5257 | - |
| date arithmetic | 0.24 | 0.28 | 18.0% | 0.4948 | - |
| difference | 0.20 | 0.19 | 14.8% | 0.4994 | -0.0962 |
| percent-change | 0.17 | 0.17 | 12.6% | 0.4861 | - |
| product | 0.16 | 0.13 | 10.4% | 0.5165 | - |
| ratio | 0.12 | 0.17 | 6.6% | 0.5121 | - |
| sum | 0.07 | 0.09 | 4.6% | 0.5067 | -0.2122 |

The correspondence is exact. The only two types where the model discriminates are the only two where the *correct* answer carries the source cell's digits. Within them the per-item score advantage tracks the per-item digit-prefix advantage; outside them the correlation is zero or negative.

**This is the single most useful finding for the lane builder.** The model is not partially arithmetic. It has one numeric competence - "do the digits of this claimed value match the digits of the cell bound to the entity named in the claim" - which is a genuine and sharp key-value binding capability (it distinguishes the named row's cell from another row's cell at 0.8755), and it has no second competence at all. A derivation lane is not repairing a weak arithmetic faculty; it is installing one.

---

## 5. Showing the working does not help, and it is a gaming surface

Same table, same operands, same asserted value; only the claim template varies. Bare: *"The combined attendance of A and B is V."* Shown: *"The attendance of A is 4,671 and the attendance of B is 6,586, so the combined attendance of A and B is V."*

| operation | bare correct | bare wrong | shown correct | shown wrong | AUROC bare | AUROC shown |
|---|---:|---:|---:|---:|---:|---:|
| sum | 0.2458 | 0.2418 | 0.3017 | 0.3086 | 0.5062 | 0.4845 |
| difference | 0.3653 | 0.3685 | 0.5109 | 0.5164 | 0.4933 | 0.4920 |
| ratio | 0.2938 | 0.2924 | 0.5289 | 0.5260 | 0.5007 | 0.5036 |
| percent-change | 0.4305 | 0.4309 | 0.5844 | 0.5842 | 0.4996 | 0.5017 |
| **all, n = 1,600** | **0.3338** | **0.3334** | **0.4815** | **0.4838** | **0.5010** | **0.4973** |

Quoting the operands lifts the absolute score by **+0.148**, and lifts the wrong answer by **+0.150**. Discrimination is unchanged at chance. The arena contrast in section 2.4 (0.377 bare vs 0.534 shown) is therefore a lexical-overlap confound, not evidence that the model follows a derivation.

**Binding lane consequence.** If lane positives quote their operands and lane negatives do not - or if the two members of a pair differ in template at all - the model can clear the lane at +0.15 of score by counting present numerals, learn nothing, and the anti-gaming clause in R14-A4 will not catch it because that clause tests present-value near-miss AUC, a different surface. The pair members must be **byte-identical except for the asserted value**.

---

## 6. The lane construction table

**Legal basis, stated explicitly.** Column *register share* comes from arena rows and is DIAGNOSTIC ONLY - it justifies which operation families exist in the financial-QA register, a domain fact that also stands on R14-A4's independent product justification. Columns *correct-derivation score*, *AUROC b vs c*, *deficit* and the resulting *recommended share* come only from the held-out TabFact probe, which contains no arena and no gold. No proportion below was set from a finqa or tatqa count.

| operation | register share (diagnostic) | correct score (b) | AUROC b vs c | deficit (a - b) | constructible | **recommended lane share** | why |
|---|---:|---:|---:|---:|---|---:|---|
| difference / period change | finqa 13.5%, tatqa 15.2% | 0.3489 | 0.4994 | 0.5708 | unbounded | **20%** | chance discrimination, wrong-operator *anti*-signal at 0.4319 - the only type the model actively prefers to get wrong |
| ratio / share-of-total | finqa 19.6%, tatqa 10.5% | 0.3010 | 0.5121 | 0.6150 | unbounded | **18%** | chance discrimination, register-dominant in finqa |
| percent-change | finqa 15.6%, tatqa 13.2% | 0.4223 | 0.4861 | 0.4843 | unbounded | **15%** | chance discrimination; shallowest deficit of the tier, so fewer rows buy the same movement |
| sum / total aggregation | finqa 7.1%, tatqa 6.4% | 0.2326 | 0.5067 | 0.6920 | unbounded | **12%** | chance discrimination, second-deepest deficit; the only type H133 measured, so it also serves as the gate's own replication anchor |
| rounding / approximation | not separately countable | **0.1763** | 0.7049 | **0.7292** | unbounded | **12%** | deepest deficit of all ten types and the shape H108's near-miss negatives most directly mistrained; the 0.7049 is digit-copy, so half the signal is already there and the lane only has to correct the sign |
| mean / average | finqa 2.9%, tatqa 6.4% | 0.2549 | 0.5257 | 0.6686 | unbounded | **10%** | chance discrimination, deep deficit; cheap because it is a sum plus a divide and shares operand machinery with the sum rows |
| scale / unit conversion | finqa 11.1%, tatqa **20.3%** | **0.4965** | **0.8755** | 0.4118 | unbounded | **8%** | **hold-only, and the lane's internal control**: the model already discriminates here at 0.8755. Spend the minimum needed to prevent regression and use the type as a within-lane sanity read - if scale AUROC falls, the lane is teaching "absent implies supported" |
| product | finqa 2.1% | 0.2683 | 0.5165 | 0.6506 | unbounded | **5%**, 2-3 significant-digit operands only | chance discrimination, but exact multi-digit multiplication is not a plausible learned function (section 7) - keep the rows, cap the precision |
| count-aggregation | finqa 6.3%, tatqa 15.9% | 0.4128 | 0.4842 | 0.5364 | **0 under the absence rule** | **0%** | chance discrimination, but count answers do not produce absent numbers, so this type does not participate in the shortcut R14-A4 exists to remove. Teaching it is a separate capability competing for the same rows |
| date arithmetic | not separately countable | 0.3915 | 0.5014 | 0.4043 | 139/500 strict | **0%** | chance discrimination but essentially absent from the register read (zero genuine instances in 60 verbatim items) and barely constructible under the absence rule |

Tier-1 (difference, ratio, percent-change, sum) takes 65% of rows; tier-2 (rounding, mean) 22%; tier-3 (scale hold, capped product) 13%.

### 6.1 Corruption construction, per axis

- **Start coarse, not near-miss.** Section 3.1 measures the model at chance even against wrong answers off by a decade. There is no reason to spend the first tranche of negatives on 2%-off near-misses. Recommended within-type negative split: **50% order-of-magnitude-wrong, 30% wrong-operand at arbitrary magnitude, 20% near-miss**. The near-miss fraction is the one that risks re-teaching the H108 shortcut and should be the smallest
- **Include the wrong-operator axis, weighted to where it is inverted.** Wrong-operator is a *different* defect from wrong-operand and the probe separates them: difference reads 0.4319 and mean 0.4648, meaning the model prefers the wrong operation. Recommended: **25% of each type's negatives are wrong-operator**, concentrated on difference-vs-sum and mean-vs-sum, which are the two measured inversions
- **Hold the template fixed.** Section 5. Positive and negative differ in the asserted value and in nothing else, byte for byte
- **Include the bare-assertion form.** The register's worst-scoring shape is a lone derived result with no operands quoted (0.3059 on finqa deciding sentences). Recommended: **at least 50% of lane rows in bare-assertion form**, since the shown-work form is where the model already has free lexical overlap to lean on
- **Do not vary digit-prefix overlap between the pair members.** Where the correct and the wrong value differ in how many of the source cell's digits they carry, the model can clear the pair on the copy detector alone - which is exactly how scale and rounding read 0.8755 and 0.7049 today. Where this cannot be avoided (rounding is intrinsically digit-preserving), balance it: half the rounding negatives should be *rounded from the correct cell in the wrong direction or to the wrong place*, preserving prefix length

---

## 7. What a 307M encoder plausibly cannot verify - do not spend rows here

Stated as honest priors, not as measurements, except where a number is given.

- **Depth-2 and deeper chains.** A single forward pass over a 512-token window has no scratchpad; an intermediate result must be held in the residual stream and re-consumed. The register's chain load is small - **15 of 250 finqa and 13 of 250 tatqa deciding sentences** - so the rows are not worth the risk. **Recommend: zero chained rows in the first lane.** If the lane admits, a chained follow-up is a separate hypothesis with its own gate
- **Exact high-precision multiplication and division.** Product values in the probe run to 6-10 digits; ratio and percent-change require division to two decimals. mmBERT tokenizes digit-by-digit (recorded at `semantic-grounding-experiments.md:2106`, `398.0` → `▁,3,9,8,.,0`), so the representation exists, but exact many-digit multiplication read off a single `[CLS]` vector is not a function this architecture is likely to learn from 50k pairs. **The verification task is easier than the generation task** - the model only has to reject values far from the truth - so the practical rule is: keep the operands to 2-3 significant digits so that *coarse* magnitude reasoning suffices, and let precision be out of scope
- **Values that are not in the retrieved text at all.** 11 of 250 finqa deciding sentences assert quantities that live in a performance **graph** ("the five-year cumulative return was 114%"). No lane and no model size fixes these; they are a retrieval and modality limit and they cap what finqa can ever reach
- **Question-borrowed thresholds.** "Revenue exceeded 500,000 thousand in two years" asserts a threshold that was never in the evidence and was never derived. Rare (1 deciding sentence measured) but it will be flagged by any absent-number detector, and the lane should not attempt to teach it
- **Count-aggregation, on this lane.** Not because it is hard - the model reads 0.4128 on correct counts, its second-highest - but because count answers do not create absent numbers, so they cannot participate in fixing P(label 0 | absent). Separate hypothesis if wanted

---

## 8. Amendments this probe recommends to the registered R14-A4 / H133 lane

These strengthen the registered block; none of them re-opens its licensing.

1. **Widen the lane from sum to eight operations, with the shares in section 6.** H133 licensed on a sum-only probe. The defect is now measured type-uniform across ten types and 5,008 quads, so a sum-only lane would under-cover its own diagnosis
2. **Add byte-identical template as a binding construction clause.** Section 5 measures a +0.148 free lift available to any model that counts quoted operands. This is a gaming route the existing anti-gaming clause (present-value near-miss AUC) does not close
3. **Add scale/unit conversion as a within-lane control, not as a target.** The model reads 0.8755 there today. Pre-register: **if held-out scale AUROC falls below 0.80 after the lane, the lane has taught "absent implies supported" and the run is void** regardless of the finqa number. This is a cheaper and more direct anti-gaming instrument than the registered in-domain near-miss clause, and it is arena-free
4. **Weight negatives coarse-first.** 50 / 30 / 20 across order-of-magnitude, arbitrary wrong-operand, near-miss. The near-miss family is the one that manufactured the original defect
5. **Add the wrong-operator axis at 25% of negatives**, concentrated on difference-vs-sum and mean-vs-sum where the model is measured *anti*-correct (0.4319, 0.4648)
6. **Exclude count-aggregation and date arithmetic.** Zero and 139 constructible under the absence rule respectively, and neither participates in the shortcut the lane exists to remove
7. **Record the artefact rate in the shortcut census.** The 0.610 / 0.946 purity figures that motivate the lane are produced by a detector that counts the percent-formula constant `100` as an asserted absent number in 46 of 250 finqa deciding sentences. The purity is probably real but its measurement is inflated; re-measuring it with the artefact filter is free and should be done before the lane's size is fixed

---

## 9. What would falsify the reading in this document

- **The digit-copy account** predicts that a derivation type constructed so that the *wrong* value carries the source cell's digit prefix and the *correct* value does not will read AUROC **below** 0.50. Cheap to build, ~0.05 GPU-h. If it reads at or above 0.50, the scale/rounding discrimination is something better than digit copying and section 6's scale-as-control recommendation is wrong
- **The type-uniformity claim** is a single-checkpoint measurement (H105 draw 1). If H105 draw 2 or an H108-lane checkpoint reads materially above chance on any type, the defect is checkpoint-specific rather than recipe-specific. ~0.2 GPU-h on the banked quad parquet
- **The "coarse first" recommendation** rests on section 3.1, whose per-type order-of-magnitude subsamples are 16 to 97 quads. A targeted rebuild that forces every negative to a decade error, 500 per type, would settle it for ~0.1 GPU-h

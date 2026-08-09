# R15 Probe P4 - What a 307M encoder can and cannot learn about numbers

**Role**: model-knowledge dossier for the R14-A4 lane builder. No web access; every published-work claim comes from model knowledge and is confidence-flagged. Every number attributed to this campaign was re-read from disk in this session, and four original measurements were taken here and are marked **[NEW]**.

**Discipline**: frozen weights only, held-out TabFact (`table_id`-disjoint from every training split), zero arena, zero gold. Total spend ~0.35 GPU-h on card 0. Polars throughout. Artifacts: `R15_gate_P4_numeracy.json`, `R15_gate_P4_controls.json`, `R15_P4_family_scores.parquet`, `R15_P4_numeracy_probe.py`.

---

## 0. The finding, first

**The shipped model's numeric substrate is intact and its numeric task function is empty.** The frozen trunk linearly separates "which of these two numbers is larger" at **99.7% held-out accuracy from 200 training examples**, and linearly decodes log-magnitude at **R² = 0.9987** inside the training range - while the shipped task head, asked the same comparison as a grounding claim over a two-row table, reads **AUROC 0.5230**. The information is in the representation. The head does not use it.

**The defect is broader than arithmetic.** R14-H133 measured AUROC(correct vs wrong) = 0.4924 on one derivation family, the two-operand sum. Extending the identical construction to five families [NEW] finds chance on all four non-verbatim families, including the one that requires **no computation at all**:

| family | asserts a new value? | mean(correct) | mean(wrong) | AUROC(correct vs wrong) |
|---|---|---|---|---|
| verbatim (control) | no - value is a table cell | 0.9037 | 0.0879 | **0.9818** |
| comparison (`A > B`) | **no - both operands verbatim** | 0.3636 | 0.3595 | **0.5130** |
| difference | yes | 0.4115 | 0.4060 | 0.5112 |
| ratio / percent | yes | 0.3333 | 0.3252 | 0.5159 |
| sum | yes | 0.2457 | 0.2474 | 0.4962 |

`n = 1,200` per family, held-out TabFact, single fixed seed 20260810. The sum row replicates H133's 0.4924 at a different seed and a different sample, and sibling probe P1 independently reads 0.5067 on a third - the chance result is not a sampling artifact. §6 reconciles the two taxonomies in full; P1 adds a wrong-**operator** arm that this probe lacks and that turns out to matter.

**Consequence for the register.** "The model cannot do arithmetic" is the wrong diagnosis and would send the lane builder after the wrong rows. The comparison family swaps two entity names in a claim whose every operand is printed in the evidence; getting it right requires ordering two integers the trunk already separates linearly. The deployed function is at chance. What is missing is not numeracy - it is any supervision that makes the head **compose a relation over two table cells**. Arithmetic is the hardest member of that family, not its definition.

---

## 1. What was measured here

Four instruments, all frozen-weights on `R9-H105-mmbert-dann-clean`, all held-out.

**[NEW] Instrument A - derivation-family extension.** Table above. Same construction as `R14_H133_probe.py` (held-out TabFact test+validation, train `table_id`s excluded, correct and wrong values both absent from the evidence string, ≤ 4 tuples per table), extended from one family to five. Per-family scores banked in `R15_P4_family_scores.parquet`.

**[NEW] Instrument B - representation probe** (Wallace-style protocol, see §3). Ridge probe on the frozen trunk's `[CLS]`:

- **Magnitude, interpolation** - train and test on values 1-999, target log₁₀: **R² = 0.9987**
- **Magnitude, extrapolation** - train 1-999, test 10,000-99,999: **R² = -46.1**
- **Comparison** - "Alpha is X and Beta is Y", target `X > Y`, held-out: **accuracy 1.000 / AUROC 1.000**

**[NEW] Instrument C - probe controls** (`R15_gate_P4_controls.json`), because a 768-dimension probe on a deterministic target invites overfitting objections. All three controls hold: permuted labels → **0.508** (chance); 200 training rows instead of 2,800 → **0.997**; ridge λ raised 10⁴× → **0.985**. The comparison signal is linearly present, low-dimensional, and not probe capacity.

**[NEW] Instrument D - task head on a minimal synthetic table.** The comparison result in Instrument A could be blamed on long TabFact evidence or argmax placement. It is not: given a two-row, two-column table containing only `Alpha | X` and `Beta | Y`, and the minimal claim pair "Alpha is greater than Beta" / "Beta is greater than Alpha", the shipped head reads **AUROC 0.5230**, mean 0.4035 correct against 0.3991 wrong. Nothing is truncated, nothing is retrieved, the whole table is 30 characters, and the function is at chance.

**[NEW] Instrument E - tokenizer census** on the shipped `tokenizer.json`. Reported in §2; it is the single most consequential fact in this dossier and it is good news.

---

## 2. The numeric substrate: mmBERT tokenizes digit-by-digit

Measured on the shipped tokenizer [NEW]:

```
'1234'      -> 5 toks  [' ', '1', '2', '3', '4']
'1,234'     -> 6 toks  [' ', '1', ',', '2', '3', '4']
'12.34'     -> 6 toks  [' ', '1', '2', '.', '3', '4']
'1000000'   -> 8 toks  [' ', '1', '0', '0', '0', '0', '0', '0']
'\\$1,234.56' -> 9 toks [' \\$', '1', ',', '2', '3', '4', '.', '5', '6']
```

Every number in every claim and every table cell reaches the trunk as a **left-to-right sequence of single-digit tokens**.

**Deferring to P2 on the vocabulary count, and correcting my own.** Sibling probe `R15_probe_P2_tokenizer.md` audits the BPE merge table directly and is authoritative: **10 pure-digit tokens in vocabulary, 0 multi-digit, and 0 digit-digit merges out of 580,604 total merges** - digit-atomic by construction, not by accident of frequency. My first pass counted "285 pure-digit entries including 31 two-digit"; that was a **Unicode-digit regex artifact** - `\d` in Python matches Devanagari, Bengali, Thai and fullwidth numerals, and 275 of those 285 were non-Latin (e.g. `▁২০১`). Re-run restricted to ASCII: **10 tokens, `0`-`9`, and zero space-prefixed digit tokens.** The corrected fact strengthens the conclusion rather than weakening it. One honest caveat inherited from the artifact: multi-digit tokens *do* exist for some non-Latin scripts, so digit-atomicity is an ASCII-numeral guarantee, which is the register the lane will be built in.

This matters more than any other architectural fact in the file, because the single best-documented obstacle to encoder numeracy is the opposite regime. Original BERT wordpiece splits numbers into arbitrary, inconsistent multi-digit fragments - `1234` might become `12`/`##34` while `1235` becomes `123`/`##5` - so two numerically adjacent values receive unrelated token sequences and no consistent positional meaning. **Wallace, Wang, Li, Singh & Gardner, "Do NLP Models Know Numbers? Probing Numeracy in Embeddings" (EMNLP 2019)** [confidence HIGH] is the canonical measurement: character-level encoders (ELMo, char-CNN) beat subword models on magnitude decoding, list-maximum and addition probes, and the authors attribute the gap to exactly this tokenization inconsistency. **Thawani, Pujara, Ilievski & Szekely, "Representing Numbers in NLP: a Survey and a Vision" (NAACL 2021)** [HIGH] makes the same point as a design recommendation: digit-level or scientific notation, not subword.

The shipped model is already in the recommended regime, and Instrument B confirms the payoff empirically - log-magnitude at R² = 0.9987 and comparison at 1.000 are the numbers a char-level model gets in Wallace's probes, not the numbers a wordpiece BERT gets.

**Amendment to the record.** `R14_evidence_E5_capacity.md:132` concludes "Can a 400M encoder do arithmetic? No, and neither can a 307M one... Arithmetic competence is a pretraining-corpus property, not a parameter-count property... There is no mechanism by which that produces derivation." The capacity half of that verdict stands and is strengthened - no ladder rung changes any of this. The **representation** half is now measured and it is wrong in the pessimistic direction: the magnitude and ordering competence that derivation checking needs is already in the trunk at 0.9987 / 1.000, put there by pretraining, and is being discarded by the task head. E5's inference "pretraining did not supply it, therefore it cannot be reached" does not survive Instrument B. What pretraining did not supply is *exact multi-digit computation* (§4), which is a smaller and more precisely bounded gap than E5 assumes.

---

## 3. What the published record says an encoder can learn

Six lines of work bear on this, ordered by how directly they license or constrain the R14-A4 lane.

**Synthetic numerical data teaches a BERT-scale model numerical reasoning.** **Geva, Gupta & Berant, "Injecting Numerical Reasoning Skills into Language Models" (GenBERT, ACL 2020)** [HIGH on existence and core claim; MEDIUM on exact benchmark deltas, which I do not quote] takes a BERT-base-sized model, pretrains it on two automatically generated corpora - numerical expressions, and templated textual statements over synthetic tables - and reaches performance on numerically-augmented reading comprehension that previously required specialized architectures. Two details of that recipe are load-bearing and both are reproducible here: GenBERT used **digit tokenization** (which the shipped model already has, §2), and it needed **both** the symbolic corpus and the textual-template corpus - the symbolic one alone did not transfer to text. This is the strongest published precedent that R14-A4's ~50k constructed pairs are a mechanism rather than a hope, and it is the strongest published argument that the lane must present derivations **in claim-and-table form**, not as bare arithmetic.

**Encoders succeed by selecting operations and operands, not by computing.** **Andor, He, Lee & Pitler, "Giving BERT a Calculator" (EMNLP 2019)** [HIGH] has BERT predict *which operation over which arguments* and hands execution to an external module; the competence that transfers is the selection. **Herzig et al., TAPAS (ACL 2020)** [HIGH] reaches table QA with a BERT-class encoder plus an explicit aggregation-operator head (COUNT / SUM / AVERAGE) and cell selection under weak supervision - again, the transformer chooses, something else computes. **Liu et al., TAPEX (ICLR 2022)** [HIGH] pretrains a BART-class model to **imitate an SQL executor** over synthetic tables and gets strong table reasoning from purely synthetic executor traces. The consistent message across all three: at this scale, the learnable object is **operand binding and operator identity**, and exact evaluation is what gets outsourced.

This is directly confirmed by our own error forensics. `R14_evidence_E4_items.md:162-165` reads response 200 - `Ratio = \\$6.2 billion / \\$38.8 billion = 0.16` - scored 0.7493 and gold-unsupported, because the two operands are verbatim in the window but belong to **different line items**: repurchase-agreement net transfers and long-term-debt net transfers. E4 calls it "the canonical quantitative near-miss: the tokens are all correct and the binding is not", and prices this single response at 13.4% of all finqa discordance. The published record says binding is the learnable half; our forensics say binding is where the errors are. These agree, and the lane should be built on that agreement.

**Exact arithmetic is length-fragile, and the fragility is about digit length.** **Nogueira, Jiang & Lin, "Investigating the Limitations of Transformers with Simple Arithmetic Tasks" (2021)** [HIGH] trains T5-class models on addition and subtraction and finds high in-distribution accuracy that collapses on digit lengths absent from training, with surface format (explicit position markers, character-level presentation) worth large accuracy swings. **Lee et al., "Teaching Arithmetic to Small Transformers" (2023)** [HIGH on core claims] shows small transformers *do* learn addition from scratch given enough well-formatted data, that presentation format (reversed digit order, scratchpads) dominates, and that multiplication scales far worse than addition. **Spithourakis & Riedel, "Numeracy for Language Models" (ACL 2018)** [MEDIUM-HIGH] reaches the same conclusion from the modelling side: digit-based and hierarchical number models beat a flat softmax over a number vocabulary.

**Magnitude and scale survive pretraining; exactness does not.** Beyond Wallace, **Zhang et al., "Do Language Embeddings Capture Scales?" (EMNLP Findings 2020)** [MEDIUM-HIGH] finds BERT embeddings carry coarse scalar magnitude for objects and quantities. **Kim et al., "Have You Seen That Number? Investigating Extrapolation in Question Answering Models" (EMNLP 2021)** [MEDIUM on attribution, HIGH on the finding] documents the same interpolation-good / extrapolation-bad boundary in QA models. Instrument B reproduces this boundary on the shipped model exactly: R² = 0.9987 inside 1-999, R² = -46.1 on 10,000-99,999.

**Financial numeric reasoning is a program-synthesis benchmark, not an entailment benchmark.** The FinQA and TAT-QA lines [HIGH] both define the task as producing an executable program over a table and executing it externally; encoder-only baselines sit far below program-based ones. This is literature context only - **those corpora and every derivative are FORBIDDEN as training data under the campaign's closed lines, and nothing in this dossier proposes touching them.** The relevant transfer is a warning, not a dataset: the published community found this shape of task unsolvable without an executor, so a lane that bars an executor must aim at the checkable sub-problem, not the computable one.

---

## 4. Verification vs generation: what the asymmetry actually buys

**The claim, stated precisely.** In our framing the candidate value is *supplied by the claim*, so the model never has to emit digits - it has to decide whether a supplied scalar is consistent with the evidence. Checking admits cheap sufficient-to-reject tests that generation cannot use: order of magnitude, digit count, sign, last digit, and monotonicity bounds (a sum of two positive cells exceeds both; a difference is smaller than the larger; a part-of-whole percentage of a smaller by a larger is below 100). Instrument B says every ingredient for the magnitude and ordering tests is linearly available in the trunk.

**The published support is indirect and I flag it as such.** **Cobbe et al., "Training Verifiers to Solve Math Word Problems" (GSM8K, 2021)** [HIGH on the result] is the nearest evidence: a verifier trained to judge candidate solutions substantially outperforms the same-size generator's own accuracy, and scales better with data. But that is a **decoder** verifying **natural-language solution traces**, not an encoder cross-encoder scoring a claim against a serialized table. **I know of no published measurement of the verification/generation asymmetry in our exact setting.** Treat it as a mechanism with a plausible analogue, not an established finding.

**Where the asymmetry does not help, and this is the part the lane builder must internalise.** Cheap checks reject cheap negatives. They do nothing against a negative constructed to survive them - and H133's wrong-operand family is exactly that construction: the wrong value is the same operator applied to a *different pair of cells in the same column*, so it usually has the same digit count, the same order of magnitude, and a plausible relation to the row labels. Our own data shows the cheap checks are not merely unused but anti-correlated [NEW]: binning H133's 2,000 triples by the relative gap between the correct and wrong value, the top quartile - where the wrong value is *most* obviously off - is the one quartile where the model scores the **wrong** value higher (0.2246 against 0.2080). And the per-digit-length breakdown is at chance everywhere, including **two-digit sums at AUROC 0.4819**, which no capacity or multi-digit-carry story explains.

**Honest conclusion.** The asymmetry converts a generation problem into a classification problem over a supplied candidate, which is a real reduction, and it makes the *approximate* half of the taxonomy attainable. It does not make exact multi-digit discrimination attainable, and a lane whose negatives are uniformly exactness-hard will present a gradient the model cannot climb.

---

## 5. Value distribution, interpolation, and digit length

Three constraints follow from §3 and §4, all measurable at build time and all free.

**Interpolation is the operating regime; extrapolation is not available.** Instrument B's -46.1 extrapolation R² is a deliberately severe split (disjoint digit-length ranges, 1-3 digits training against 5 digits testing) and should be read as a direction, not a coefficient. The direction is unambiguous and matches Wallace and Kim: **whatever value range the lane covers is the range where the lane's competence exists.**

**The current construction is digit-length mismatched to the target register.** H133's derived values [NEW] sit at 4 digits (601), 5 digits (801) and 6 digits (84) - **74% of the sample at 4-6 digits** - because they are sums of TabFact sports and demographic cells. `R14_evidence_E6_train_composition.md:128-138` records TabFact at median digit density 13.6 per 100 characters, **3.5× the finqa arena median** and in a different surface form. Financial claims carry few significant digits with scale words and currency (`\\$6.2 billion`, `70.6`, `65.1%`), which under digit-level tokenization is a completely different token sequence from `1234567`. A lane built without controlling this distribution trains the interpolation window in the wrong place.

**Surface format is part of the value distribution, not cosmetics.** Because tokenization is digit-level, `1234`, `1,234`, `1.234 thousand` and `\\$1,234.00` share almost no token structure. The H108 lane's unit / period / scale corruption family is the campaign's only replicated finqa lever precisely because it teaches this axis; `R14_evidence_E1_finqa.md:97` records H108 pair mean 0.71815 against the clean control's 0.63325, **+0.0849**.

---

## 6. Realistic ceilings by derivation type

P1 (`R15_P1_typeprobe.json`) landed while this probe was running and measures ten types on 4,139 held-out quads with a fourth arm this probe lacks - **(d) wrong operator**, same operands. The table below is P1's taxonomy with P1's numbers, plus the two types P1 does not carry (comparison, from Instrument A; operand binding, from E4). Where the two probes overlap they agree: P1 sum 0.5067 against my 0.4962, P1 difference 0.4994 against my 0.5112, P1 ratio 0.5121 against my 0.5159, all at chance across three independent samples and two seeds.

| type | requires | AUROC(b vs c) wrong **operand** | AUROC(b vs d) wrong **operator** | plausible at 307M / ~50k pairs |
|---|---|---|---|---|
| verbatim (control) | string match | - (a vs b 0.92-0.97) | - | **solved** - anchor rows only |
| **comparison / ordering** | binding + ordering, **no computation** | **0.5130** [P4] | - | **HIGH - cheapest large win** |
| **operand binding** (right value, wrong line item) | row-label alignment, **no computation** | not probed | - | **HIGH** |
| scale / unit restatement | single-operand format map | **0.8755** | 0.7059 | **already largely working** |
| rounding | precision map | **0.7049** | 0.5502 | **partially working** |
| pct_change | division + subtraction | 0.4861 | 0.5746 | LOW |
| sum | exact addition | 0.5067 | 0.5072 | MEDIUM, partial |
| difference | exact subtraction | 0.4994 | **0.4319** | MEDIUM, partial |
| mean | addition + division | 0.5257 | 0.4648 | LOW |
| ratio | exact division | 0.5121 | **0.6705** | LOW |
| product | exact multiplication | 0.5165 | 0.6079 | LOW |
| date arithmetic | period binding | 0.4948 | 0.6654 | MEDIUM |
| count / aggregation | small-integer count | n = 0, not adjudicated | - | MEDIUM, unmeasured |
| multi-step chains (≥ 3 ops) | composition + exactness | not probed | - | **OUT OF REACH without an executor** |

**Two readings of P1's numbers that change the lane's shape.**

**First: operator identity is partially readable while operand binding is dead.** Across the arithmetic types, wrong-**operator** negatives separate at 0.57-0.71 (ratio 0.6705, date 0.6654, product 0.6079, pct_change 0.5746) while wrong-**operand** negatives sit at 0.486-0.526 without exception. This is precisely the Andor / TAPAS decomposition - *select the operation, select the arguments* - with one half partially learned from the existing mix and the other half at chance. It is the sharpest available statement of where the ~50k rows should go, and it was not visible from H133's three-arm construction.

**Second: I had `scale_unit` wrong, and P1 corrects me.** My prior rated scale/unit restatement HIGH-and-worth-20%-of-the-lane on the strength of H108's replicated +0.0849. P1 measures it at **AUROC 0.8755 with 51% of correct restatements already scoring above 0.5** - the best non-verbatim number in the taxonomy. It is not a gap; it is the one thing that already works, and it is the H108 lane's fingerprint showing up exactly where H108 taught it. **Spending 20% of a new lane there would buy little.** Prescription P4-1 is rebalanced below. Rounding at 0.7049 is a genuine partial and worth a small budget.

**The `difference` anomaly is worth one line.** Wrong-operator negatives for `difference` separate at **0.4319** - meaningfully *below* chance, meaning the model systematically prefers the wrong operator's value. With `mean` at 0.4648 in the same direction, this is a small live signal that some operator surface forms are actively mis-bound rather than merely ignored. It does not change the prescriptions; it is flagged for whoever reads the lane's per-type verdict.

---

## 7. Engagement with the closed line H119

R14-A4 is a data-only lane and proposes no serving transform, so H119 is not directly at issue. Prescription 3 below does touch input surface form, so the verdict is engaged explicitly.

**What H119 was.** A frozen-weights **serving wrapper**: an idempotent, subset-blind string canonicalization (thousands separators, currency spacing) applied before tokenization to a model whose weights had never seen that distribution. It was **REFUTED in both directions** (log:2577): strip read +0.00245 / -0.00098 against a +0.003-on-both bar, finqa cleared +0.010 on 2 of 4 draws against a 3-of-4 bar and sign-disagreed inside the H105 pair; add read -0.00039 / -0.00002.

**Why that evidence does not transfer to training-time format coverage.** H119's own mechanism finding is the reason: the transform was confirmed **localized but not directional** - "what each checkpoint does with the surface gap is idiosyncratic to its weights", with tatqa swinging +0.0448 / +0.0012 / -0.0142 / -0.0227 across four draws under a deterministic zero-variance read. That is a statement that **the weights, not the input pipeline, are the locus of numeric surface behaviour**. A serving wrapper asks a fixed function to behave better on a shifted input distribution and cannot change what the function learned; training-time format coverage changes which function you get. The two act on different objects, and H119 measured only the first.

**What H119 nevertheless binds.** Its finding licenses a real concern for the lane: format sensitivity is genuine and checkpoint-idiosyncratic, so a lane that renders every derived value in exactly one canonical surface form will bake in a fresh idiosyncrasy of the same kind. Format coverage must therefore be **symmetric across positives and negatives** - otherwise surface form becomes a label cue and the lane trades one shortcut for another, which is the failure R14-A4 exists to repair. No claim is made here that format coverage will move the arena; it is a robustness constraint on the lane's construction, not a lever.

---

## 8. Five design prescriptions for the lane builder

**P4-1. Spend the rows on operand binding, not on arithmetic - and not on what already works.** The two types that need **no computation at all** are the two with the most headroom: operand binding (right value, wrong row or line item) and comparison/ordering, the latter measured here at 0.5130 while the trunk solves the same discrimination linearly from 200 examples. Both are supported by the Andor / TAPAS / TAPEX selection-and-binding precedent and by E4's forensics, which price a single operand-binding error at 13.4% of finqa discordance. Do **not** spend a large budget on scale/unit restatement: P1 measures it at 0.8755 already. Suggested split of ~50k pairs: **operand binding 35%, comparison/ordering 20%, sum+difference 20%, rounding 10%, date arithmetic 5%, count/aggregation 5%, ratio+product+pct_change 5% (measurement only).** Multi-step chains excluded.

**P4-2. Stratify negative difficulty, and hold shortcut-neutrality inside every stratum.** A lane made entirely of H133-style same-column wrong-operand negatives is uniformly exactness-hard and offers no foothold - measured here, the model does not even use the magnitude gap it has, scoring the wrong value *higher* in the top relative-gap quartile (0.2246 vs 0.2080). Build three strata: **gross** (wrong by ≥ 1 order of magnitude or wrong sign), **near** (right magnitude, wrong digits), **binding-hard** (same operator, different operands - the H133 family). R14-A4's binding `P(label 0 | number absent) = 0.5` construction must hold **within each stratum**, not merely in aggregate, or stratum identity becomes the new shortcut.

**P4-3. Control the value and surface distribution to the serving register, symmetrically.** Cover 1-7 digits with explicit per-length quotas rather than accepting what TabFact sums happen to produce (measured: 74% of H133's values land at 4-6 digits). Render values across the surface forms the deliverable will meet - bare integer, thousands-separated, decimal, percent, currency, and scale words - because digit-level tokenization makes these disjoint token sequences (§2). **Apply the format sampler identically to positives and negatives**, per §7. Extrapolation is not available (R² -46.1), so the covered range *is* the competence range.

**P4-4. Diversify tables, not tuples, and hold out at the table level.** The census constructs 2.0M tuples from **16,476 admitting tables at a median 110 tuples per table**, so tuple count is not the scarce resource - table diversity is. Cap tuples per table at ≤ 4 (H133 used 3) so ~50k pairs draw on ≥ 12,500 distinct tables, and make the anti-gaming held-out split **table-disjoint**, never row-disjoint. Declare the row-level operator disjointness against R14-A6 that the A4 registration already requires.

**P4-5. Instrument the verdict per type, and pre-register a partial-credit reading now.** Re-run P1's ten-type probe plus this probe's comparison family on every trained draw, and bank the per-type AUROC alongside the finqa bar. Both probes are frozen-weights, held-out, arena-free and cost ~0.1 GPU-h each, so this is nearly free. The taxonomy predicts a specific signature, and that signature is what distinguishes "the lane taught composition" from "the lane taught a new shortcut". Pre-register the reading before any draw: **comparison and operand binding both above 0.65 while ratio/product/pct_change stay below 0.60 confirms the mechanism even if the finqa bar misses**, and should be recorded as a mechanism win with a register-transfer failure rather than a kill. **Division types moving as much as comparison is a warning sign of a construction artifact, not a success**, because nothing in the published record or in either probe predicts a 307M encoder learning division from ~2,500 examples. Keep `scale_unit` in the read as a **non-regression check** - it is at 0.8755 today and a lane that damages it has traded one competence for another. Do not order the strata as a curriculum - mix them - since ordered-difficulty training is adjacent to the closed GroupDRO-curriculum line and would add a second variable to a data-only lane.

---

## 9. Uncertainty register

Stated plainly, because the prescriptions above are only as good as these.

- **No web access.** Every citation is from model knowledge. Titles, authors and venues carry the confidence flags given inline; where I was unsure of a number I omitted it rather than guess. Nothing here should be quoted onward as a literature fact without a check.
- **The verification/generation asymmetry is a mechanism, not a measurement**, in our setting. The GSM8K verifier result is decoder-based on natural-language traces. §4 says this twice on purpose.
- **mmBERT's pretraining corpus composition is unknown to me** - specifically whether it carries meaningful code, table or math mass. That is the one fact that would most sharpen the T6 ceiling, and I cannot supply it.
- **Instrument A probes one claim template per family.** The chance readings are robust across five families, two seeds and a minimal synthetic table (Instrument D), but template sensitivity is not ruled out. A second template per family is cheap and would be worth building into the lane's evaluation harness.
- **The extrapolation R² of -46.1 is a direction, not a coefficient** - the split was chosen severe on purpose.
- **Operand binding and count/aggregation are still unmeasured.** P1 covers eight types with numbers, this probe adds comparison, and P1's `count_agg` returned n = 0 (not constructible under its caps). Operand binding - the type P4-1 gives the largest single share of the budget - is a **prior drawn from E4's forensics and the TAPAS/Andor precedent, not a measurement**. Both probes' harnesses extend to it for roughly 0.1 GPU-h. Given that P4-1 spends 35% of the lane there, measuring it before committing the row budget is the single highest-value follow-up in this file.
- **One of my own numbers was wrong and is corrected in §2.** The first-pass digit-vocabulary census (285 pure-digit entries) was a Unicode-digit regex artifact; P2's merge-table audit (10 tokens, 0 digit-digit merges) supersedes it. Flagged rather than silently edited, because it is a reminder that `\d` is not ASCII and this campaign's data is multilingual.

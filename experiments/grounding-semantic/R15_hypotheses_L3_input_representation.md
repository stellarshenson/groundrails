# R15 LENS 3 - input / representation at TRAINING time

**Role**: hypothesis author for the input-and-representation lens of the Round 15 repair field. Output: four candidates against the measured R14-H133 defect, each an amendment or a sub-block that leaves the serving contract byte-identical.

**Discipline**: CPU only, Polars only, zero GPU spent in this session, zero arena quantity entering any bar. Four original measurements were taken here and are marked **[P-L3]**; they are banked in `R15_L3_scale_census.json` and `R15_L3_scale_census2.json` and reproduce from `tmp/R15_L3_scale.py` / `tmp/R15_L3_scale2.py` in ~90 seconds.

**Id note**: R14's synthesis used `L3-*` for the DATA/MIXTURE lens. These ids are R15-scoped and are `L3-C1` … `L3-C4` per the task's `L#-C#` convention. Where an R14 `L3-*` id is meant, it is written `R14 L3-C2` in full.

---

## 0. The four candidates, first

The register's one working numeric instrument is a **digit-prefix copy detector** (`R15_probe_P1_anatomy.md` §4: scale/unit AUROC 0.8755 and rounding 0.7049 are exactly the two derivation types whose correct value preserves the source cell's digit string, mean longest-common-digit-prefix 3.39 and 3.05 against 0.60 and 0.61 for the wrong value, with per-item Pearson +0.4300 and +0.3304 between digit-prefix advantage and score advantage). `R15_probe_P2_tokenizer.md` §5 measures the representational consequence: under a digit-atomic tokenizer, `tokens(v)` is a **strict prefix** of `tokens(10v)` in every case tested. The single competence the model has is therefore *blind by construction* to the single commonest surface error in the target register.

That is the lens's whole thesis, and it yields four levers.

| id | kind | one-line | marginal GPU |
|---|---|---|---:|
| **L3-C1** | amendment (R14-A4) | Scale-identity factorial sub-block: surface crossed with label so a value's *spelling* carries zero information and its *magnitude* carries all of it | ~0.4 |
| **L3-C2** | new-hypothesis (A4 sub-block form) | Evidence-side minimal pairs: claim held byte-identical **including the asserted value**, the evidence cell edited, so no claim-side feature whatsoever can predict the label | ~0.4 (sub-block) / ~13 (standalone) |
| **L3-C3** | amendment (R14-A4) | Operand-quoting parity: every tuple emitted in bare **and** shown-work form at both polarities, killing the measured +0.148 free lift at its source | ~0.2 |
| **L3-C4** | measurement | Trunk numeracy-retention probe as a binding HOLD on every training arm - the instrument that would have seen a Pyrrhic win | ~0.3 |

None ships a transform. `src/groundrails/semantic_ov.py` is untouched by all four; `MAX_LEN`, the 1,500-character window, the max-over-windows read and every byte of pre-tokenization handling stay as they are. The deliverable remains one model under 400M with an **unchanged serving contract**.

---

## 1. Why a training-time representation change is a different object from H119 - stated once, binding on all four

**What H119 was and what it measured.** A frozen-weights **serving wrapper**: idempotent, subset-blind numeric string canonicalization (thousands separators, currency spacing) applied to claim and evidence before tokenization, on four checkpoints. REFUTED in both directions - strip read pair means +0.00245 / -0.00098 against a +0.003-on-both bar, finqa cleared +0.010 on 2 of 4 draws against a 3-of-4 bar and **sign-disagreed inside the H105 pair** (-0.0163 draw 1, +0.0178 draw 2); add direction -0.00039 / -0.00002. Its mechanism finding was **localized but not directional**: every non-numeric subset moved under 0.002 while tatqa swung +0.0448 / +0.0012 / -0.0142 / -0.0227 across four draws under a deterministic zero-variance read.

**Why the model can exploit at training what it could not at serving.** H119's own instability *is* the diagnosis. A function whose response to a thousands separator is arbitrary in sign across four checkpoints of the same recipe is a function that holds **no stored equivalence** between the two surfaces - the separator is an out-of-distribution perturbation and the response is a random walk in input space. H119 asked that function to generalize across a gap it had never been shown. It cannot: at serving there is no gradient, so the invariance must already exist to pay. A constructed lane does the opposite operation - it **places labelled examples on both sides of the gap with the label held constant**, which is the only mechanism that installs an equivalence into weights. The two interventions act on different objects: one perturbs the input of a fixed function, the other determines which function you get.

**Four checkable non-transfer points**, carried on every candidate below:

1. **No transform ships.** H119's grant was conditional on the canonicalization running in the library serving path for every corpus and every future input, precisely because a transform kept for one arena subset is arena-fitted preprocessing. A lane-internal surface choice is applied to **zero** inputs at serving time and therefore cannot be arena-fitted preprocessing.
2. **No directional claim about a transform is made.** H119 had to assert that a specific edit helps in a specific direction and failed on that. L3-C1 asserts only that surface must be made **orthogonal to label inside the lane** - a construction property, verifiable at build time from the emitted rows, not a prediction about a checkpoint.
3. **Every mixture weight is frozen from in-domain data.** `R15_probe_P2_tokenizer.md` §6 measures the admissible surfaces: H108 lane claims carry 1,282 separator-bearing numerals in 3,625,494 characters, H108 evidence chunks 18,718 in 24,667,330, TabFact tables **zero in 2,809,937**. The arena figure (finqa responses 11.07% separator-bearing) is recorded as corroboration and enters no weight and no bar.
4. **Each candidate registers a falsifier that could exonerate the null.** H119 had none. Every block below carries one.

**What H119 nevertheless binds, and is honoured.** Format sensitivity is real and checkpoint-idiosyncratic, so a lane that renders derived values in exactly one surface bakes in a fresh idiosyncrasy of the same kind, and a lane whose surface distribution differs between positives and negatives converts surface into a label cue - the exact failure R14-A4 exists to repair. Both are closed by construction in L3-C1's factorial and are auditable in the build manifest.

---

## 2. Original measurements taken here [P-L3]

Held-out TabFact test+validation, `table_id`-disjoint from the train split that built the clean mix and the H108 lane - 3,391 parsed tables, 6,421 numeric columns of ≥3 values, 84,593 numeric cells. Plus the 2,000 banked `R14_H133_triples.parquet` asserted values.

| quantity | measured | why it matters |
|---|---:|---|
| tables carrying a unit word or currency/percent symbol in a header or caption | **1,059 / 3,391 = 31.23%** | header-bound scale restatement is natively constructible on a third of the pool without any synthetic dressing |
| numeric columns whose header carries a unit | **1,092 / 6,421 = 17.01%** | the per-column supply for the header-bound arm of L3-C1 |
| tables where a wrong-factor negative is constructible (both `v×10` and `v/10` absent from the table) | **2,726 / 3,391 = 80.39%** | L3-C1's negative family is not supply-limited |
| tables carrying **any** thousands-separated numeral | **0 / 3,391** | independent confirmation of P2's "0 separator hits in 2,809,937 chars", on the held-out split |
| numeric cells by digit length | 1 digit **36.8%**, 2 digits **31.3%**, 3 digits 7.7%, 4 digits 12.2%, 5 digits 9.2%, 6+ **2.8%** | the corpus's *operands* are 1-2 digits |
| H133 asserted derived values by digit length | 4-6 digits **67.9%**; 5 digits alone 754/2,000 | the lane's *results* are 4-6 digits. Operand and result distributions are disjoint, and P4 Instrument B measures magnitude decoding at R² 0.9987 inside the trained range and **-46.1** outside it |
| H133 `v_correct` vs `v_wrong` equal digit length | **89.65%** | reproduces P2's 88.95% token-length match from the value side; the parity holds by luck, not by construction |
| H133 `v_correct` vs `v_wrong` mean longest common digit prefix | **0.723**, 54.4% share none | the wrong-operand negative is *not* a digit-prefix near-miss, which is why the copy detector reads it at chance |
| numeric columns admitting a **digit-length-matched intra-column swap** (≥2 distinct values of equal digit length) | **6,201 / 6,421 = 96.57%**, over **2,701 / 3,391 = 79.65%** of tables | L3-C2's evidence-side minimal pair is constructible on four tables in five, with digit length controlled |

Two readings that shape the blocks below.

**The lane's value range sits where the corpus's cells do not.** 68.1% of TabFact numeric cells are 1-2 digits; 67.9% of H133's derived values are 4-6. `R15_probe_P4_numeracy.md` §5 makes the covered range the competence range (extrapolation R² -46.1). A lane that accepts what TabFact sums happen to produce trains a 4-6-digit interpolation window over a 1-2-digit operand corpus, in a register (`R14_evidence_E6_train_composition.md`) whose claims carry few significant digits with scale words. L3-C1's surface sampler and per-length quotas are the cheap correction; they are build-time and free.

**The unit-header supply is real but partial.** 31.23% of tables and 17.01% of numeric columns carry a native unit. That is enough for the header-bound arm without touching P3's register-dressing option, which `R15_probe_P3_signal.md` §4 correctly flags as arena-informed and caps at 20% with an ablation. The census must be re-run on the **build** corpus (TabFact train + FEVEROUS train) before the sub-block's size is fixed; the held-out rate above is the estimator, not the count.

---

## 3. L3-C1 - SCALE-IDENTITY FACTORIAL SUB-BLOCK
*kind: amendment to R14-A4 (data-only, no new arm, no new bar on finqa)*

**Claim** - Because the model's only measured numeric competence is a digit-prefix copy detector (`R15_probe_P1_anatomy.md` §4: scale 0.8755 / rounding 0.7049 with LCDP 3.39 / 3.05 against 0.60 / 0.61, Pearson +0.4300 / +0.3304) and because that detector is **structurally blind to magnitude** under digit-atomic tokenization - `tokens(v)` is a strict prefix of `tokens(10v)` in every case tested (`R15_probe_P2_tokenizer.md` §5), while `10.5 million` vs `10,500,000` share 3 tokens and `in millions 10.5` vs `10,500,000` share **0** (P2 §7) - emitting a sub-block in which the asserted value's **surface** (bare / separator-bearing / scale-word / currency-prefixed / decimal-padded) is crossed **factorially with the label** while the label is determined solely by the **scale factor** will move held-out wrong-factor AUROC from its gate baseline to ≥ 0.70 and collapse the surface main effect on score to ≤ 0.05, while P1's scale/unit control AUROC holds ≥ 0.80 and held-out verbatim mean holds ≥ 0.85.

**The construction, precisely.** Per tuple, one table, one numeric column, one named row. Four rows share byte-identical claim text apart from the numeral:

| | surface matched to evidence | surface mismatched |
|---|---|---|
| **correct factor** | label 1 | label 1 |
| **wrong factor** (×10 / ÷10 / ×100 / ÷100) | label 0 | label 0 |

Label depends on factor alone. Surface is drawn from a distribution frozen in writing before build, from the in-domain H108/TabFact measurement (P2 §6) and never from the finqa profile. Two arms:

- **Header-bound** (the 17.01% of numeric columns with a native unit): the header declares the unit, the claim restates in another unit. `header "attendance (thousands)", cell 5,092 → claim "5,092,000"` is label 1; `"50,920,000"` is label 0. This teaches the *binding* between a declared unit and a digit count - the zero-computation family `R15_probe_P4_numeracy.md` §6 ranks HIGH
- **Surface-identity** (all columns): the same *quantity* spelled differently - `1234` / `1,234` / `1.234 thousand` / `$1,234` - all label 1 against wrong-factor spellings at label 0. This is the equivalence the mix demonstrates **zero** times: 0 separators in 2,809,937 characters of TabFact and 0 emitted by A4's `fmt()` (P2 §8)

**Three binding construction clauses**, each closing a shortcut the probes measured:

1. **Balanced factor direction.** ×10 and ÷10 at equal rates, ×100 and ÷100 at equal rates, so the signed digit-length difference between positive and negative has |mean| ≤ 0.05 digits. Without this the negatives are systematically longer and digit count replaces digit presence - the same failure class as P(0|absent). Report AUROC-from-digit-length inside the sub-block; **P2-B's KILL at 0.55 applies to this sub-block separately**. Note that ±1 digit satisfies P3's "digit-length parity within 1" rule, so the two amendments are compatible
2. **Absence parity.** Both asserted values absent from the serialized evidence - constructible on 80.39% of tables [P-L3]
3. **Surface orthogonality is auditable.** The build manifest reports the realised 2×2 cell counts and the point-biserial correlation between surface class and label; **build KILL if |r| > 0.02**

**Kill-gate** (~0.2 GPU-h, frozen H105 draw 1, held-out TabFact, zero arena, zero gold). Matched triples on the same cell: (a) verbatim value; (b) same value re-surfaced (separator / scale word / currency), still the same quantity; (c) same digit string at a wrong factor (×10 or ÷10), absent from the table. Two clauses:

- **LICENSE** if mean |score(a) − score(b)| ≥ **0.10** - the surfaces are *not* equivalent to the model, so there is an equivalence to install - **AND** AUROC(a vs c) ≤ **0.60** - decade errors on an identical digit string are not rejected, so there is an inequivalence to install
- **KILL** if mean |score(a) − score(b)| < 0.10 (surfaces already equivalent; the positive half buys nothing) **or** AUROC(a vs c) > 0.60 (magnitude already discriminated on the copy axis; P1's "scale already works" reading extends to the wrong-factor axis and this candidate is redundant)

This gate is the cheapest way to settle a **live disagreement between two sibling probes**. P1 §6 recommends scale as hold-only-and-a-control at 8% of rows because it reads 0.8755. That 0.8755 is measured against a **wrong-operand** negative - a different row's value, which the copy detector separates trivially. Nobody has measured scale against a **wrong-factor** negative, the token-prefix-nested case P2 §5 predicts is the hardest in the representation. The gate decides which probe is right for 0.2 GPU-h, and if P1 is right this candidate kills itself.

**Bar** - the sub-block rides inside R14-A4's registered arm and **claims no separate finqa movement**; A4's PRIMARY (finqa 2-draw mean ≥ 0.6933, sign agreement) and every HOLD stand as registered (arena mean ≥ 0.70311, pubmedqa ≥ 0.5463, `gold_full` ≥ 0.8414, no subset below control − 0.06, none < 0.55, RAGTruth non-EN ≥ 0.82, log-length residualization at ≥ 50% magnitude). Its own pre-registered, ceiling-blind, arena-free **MECHANISM** reading, taken on both draws:

- **wrong-factor AUROC ≥ 0.70** on held-out TabFact, from the gate's own baseline
- **surface main effect ≤ 0.05** - mean |Δscore| between matched and mismatched surfaces of the *same* quantity, from ≥ 0.10 at the gate
- **VOID clause, adopted from P1 amendment 3**: held-out scale/unit AUROC on **P1's wrong-operand axis** falling below **0.80** voids the run regardless of the finqa number - the lane has taught "absent implies supported"
- **VOID clause, verbatim non-regression**: held-out verbatim mean ≥ **0.85** (banked 0.9051) and AUROC(a vs b) ≥ **0.90** (banked 0.9643). A lane that installs magnitude sense by damaging literal presence has traded the model's best working function
- **ANTI-GAMING (binding)**: the inverse shortcut here is "spelled unlike the evidence ⇒ supported". Held-out set of **wrong** values in mismatched surfaces must not score above the same wrong values in matched surfaces by more than **0.03**. A4's registered present-value near-miss clause does not test this surface and is retained unchanged alongside it

**Falsifier** - if the sub-block's rows are removed and a same-seed, same-rows twin built with a single canonical surface reproduces the wrong-factor AUROC movement to within half its magnitude, **surface is not the mechanism** and the factorial is dropped for the cheaper single-surface build. Costs one extra build and one 0.2 GPU-h read, no extra training draw if run as an ablation slice.

**Cost** - **~0.4 GPU-h marginal** (0.2 gate + 0.2 post-arm read). Build is CPU, ~2 hours of constructor work. Recommended size 15% of A4's 50,000 rows, taken from the shares P1 §6 assigns to scale (8%) and rounding (12%), which are the two digit-preserving types this block subsumes.

**Declared interactions** - (i) making surface orthogonal to label delivers part of R14-A5/H134's decorrelation mechanism for free, so **A4-with-L3-C1 and A5 must not run in one arm**; declare disjointness alongside P3's identical declaration on digit-length parity. (ii) P2-A is subsumed: its 10% register-mismatched positives become one cell of this factorial, and its frozen in-domain mixture weights are adopted verbatim. (iii) P2-D's token budget applies unchanged - 34.93% of 1,500-character windows exceed 512 tokens and both operand rows must survive the retained prefix.

---

## 4. L3-C2 - EVIDENCE-SIDE MINIMAL PAIRS
*kind: new-hypothesis, registrable as a declared sub-block of R14-A4 or as a standalone arm*

**Claim** - Because **every** negative in the entire training corpus is manufactured by editing the claim - `R15_probe_P3_signal.md` §2 verifies that all six H108 operators (`f_scale_word`, `f_digit_perturb`, `f_pct_pp`, `f_year_shift`, `f_comparative_flip`, `f_magnitude_shift`) and all four DR operators are claim-editing, and P3 §6 measures the consequence as P(label 0 | absent) 0.5944 over the mix and 0.9332 over the admitted lane - every claim-side surface statistic is a live label predictor, and R14-A4's own repair leaves the asserted *value* varying across a pair; holding the claim **byte-identical including its numeral** and editing the **evidence cell** instead will produce supervision on which no claim-side feature of any kind carries information, lifting held-out evidence-side AUROC from its gate baseline to ≥ 0.65 while the irrelevant-edit control stays within 0.05 of unedited.

**Why this is not R14-A4 with the sign flipped.** A4's pair differs in the asserted value, so the *value* is still the label-bearing variable: digit length, magnitude, LCDP against the source cell and surface class all differ between the two members even under parity rules, and each is a candidate cue. An evidence-side pair differs in **no** claim-side quantity - digit fraction, claim length, numeral count, absence status, surface register, token count are all exactly equal, bit for bit. The gradient has literally nowhere to go except claim↔evidence comparison. This is the strongest available form of the decorrelation R14-A5 attempts as a soft batch-statistical penalty with a recorded sign tension (in-domain partial r **+0.073** positive, arena-finqa function negative); here it is enforced by construction and costs no loss term.

**Construction.** Take a table, a column, an operator, a named row pair, and the correct derived value V. Emit:

- **Positive** - the true table, claim asserts V, label 1
- **Negative** - the table with **one operand cell replaced** by another value of the *same digit length* drawn from that column's own empirical distribution, claim asserts V byte-identical, label 0

**Origin symmetry, binding.** Half the pairs are built the other way round: start from the *edited* table, compute V from it, and treat the *original* table as the negative. Without this, "which table is the pristine one" is learnable from column statistics and becomes the new shortcut.

**Irrelevant-edit control, binding.** A third arm at ~15% of the sub-block: the same edit magnitude applied to a cell the claim does not reference, label **1**. This is the sub-block's own anti-gaming instrument - it separates "learned to compare the claim against the bound cells" from "learned to detect an edited table".

**Constructibility [P-L3]** - a digit-length-matched intra-column swap exists in **96.57% of numeric columns** and **79.65% of held-out tables**. On the build corpus (TabFact train 13,143 tables + FEVEROUS train 23,716 table-bearing blocks) the supply is far beyond the sub-block's size. **Diversity accounting, stated honestly**: an evidence-side pair consumes one table and emits two evidence strings, so it costs the same document budget as an A4 pair (P3 §5: 50,000 rows at cap 2 = 3.03 rows/document) but produces two *distinct* evidence surfaces per table, which P3's rows-per-document metric does not distinguish. Report both counts in the manifest.

**Kill-gate** (~0.25 GPU-h, frozen H105 draw 1, held-out TabFact, zero arena). Three clauses on matched pairs:

- **Clause 1 (the target)** - AUROC over evidence-side derived pairs (identical claim, correct vs perturbed operand cell). **LICENSE if ≤ 0.60** - the model does not condition on the evidence numerals for derived claims
- **Clause 2 (the discriminator, and it changes the diagnosis if it fires)** - the same edit applied to a cell the claim quotes **verbatim**. The copy detector must catch this: **NO-READ / escalate if AUROC ≤ 0.60 here too**, because then the model is not reading evidence numerals *at all*, the R14 diagnosis ("learned the shortcut, not the arithmetic") is too narrow, and the whole field needs re-scoping before any lane is built. Expected to read high given verbatim AUROC(a vs b) 0.9643
- **Clause 3 (build-side, CPU, free)** - certify that the positive and negative claim strings are byte-identical; any inequality is a build bug

This gate buys a fact no probe in R15 has: **P1, P3 and P4 all measure claim-side edits.** Whether the shipped model conditions on evidence numerals at all outside the verbatim-copy route is unmeasured, and it is the premise of every lane in the register.

**Bar** - **sub-block form (recommended)**: rides A4's arm at ~20% of rows, claims no separate finqa movement, A4's PRIMARY and HOLDs stand as registered. Pre-registered ceiling-blind mechanism reading on both draws: held-out **evidence-side AUROC ≥ 0.65** from a gate baseline ≤ 0.60, **AND** the irrelevant-edit control within **0.05** of unedited (an edit detector fails here), **AND** the verbatim-evidence-edit AUROC not falling below its gate value by more than 0.03. **Standalone form**: its own paired arm at finqa 2-draw mean ≥ **0.6733** (+0.040 over the 0.6333 paired control, sign agreement on both H126-paired draws), arena mean ≥ 0.70311, pubmedqa ≥ 0.5463, `gold_full` ≥ 0.8414, no subset below control − 0.06, none < 0.55, log-length residualization at ≥ 50% magnitude. The +0.040 rather than A4's +0.060 is deliberate: this is a narrower supervision axis with no register-transfer story of its own.

**Falsifier** - if a twin sub-block built with **claim-side** edits at equal rows, equal documents, equal operators and equal seed reproduces the evidence-side AUROC movement to within half its magnitude, the side of the edit is not the mechanism and the sub-block collapses back into A4's ordinary negatives.

**Cost** - **~0.4 GPU-h marginal** as a sub-block (0.25 gate + 0.15 post-arm read), build CPU. **~13 GPU-h** standalone plus the gate.

**Recorded risk** - editing a table cell can make the *table* internally inconsistent (a total row no longer sums). Mitigate by preferring columns with no declared total row, and by drawing the replacement from the column's own empirical values so the edited table remains distributionally ordinary. Report the share of edits landing in tables carrying a total row; if it exceeds 10%, filter.

---

## 5. L3-C3 - OPERAND-QUOTING PARITY
*kind: amendment to R14-A4 (data-only)*

**Claim** - Because quoting both operands inside the claim lifts the **correct** derivation from 0.3338 to 0.4815 and the **wrong-operand** value by the identical +0.150, from 0.3334 to 0.4838, leaving AUROC at 0.5010 → 0.4973 across 1,600 held-out pairs (`R15_probe_P1_anatomy.md` §5, `R15_P1_shownwork.json`), the model reads quoted-operand count as a free +0.148 of score with zero discriminative content; emitting **every** lane tuple in both bare and shown-work form at **both** polarities - a 2×2 of form × label - will collapse the held-out bare-vs-shown score gap to ≤ 0.05 with per-form AUROC differing by ≤ 0.03, while A4's binding log-length residualization clause is satisfied with a larger surviving fraction than an unbalanced lane can deliver.

**Why this is worth rows rather than a construction footnote.** P1 §6.1 already recommends "at least 50% of lane rows in bare-assertion form" and §5 mandates byte-identical templates within a pair. Neither makes quoted-operand count *uninformative*: a lane that is 50% bare and 50% shown, with form drawn independently of label, still leaves form correlated with score, and E1 measures a pure verbosity heuristic at finqa AUROC **0.6958** against the shipped model's 0.6489 - the model is already riding it at Spearman +0.294. **Every finqa-primary bar in the register carries a log-length residualization confound clause because of this.** L3-C3 is the only proposal in the field that attacks that confound at its source rather than measuring around it: crossing form with label inside the lane makes the verbosity route *anti-correlated* with the training signal on the one slice of data where the correct answer is known.

The register also says the bare form is the shape that matters: finqa deciding sentences carrying one absent numeral score **0.3059** against ~0.49 at two or more (P1 §2.4), and the unresolved residual is "the same operations stated bare" (P1 §2.3). The 2×2 keeps the bare form dominant in *effect* while removing form as a cue.

**Construction.** Each tuple emits four rows: {bare, shown} × {correct, wrong}. Shown-work text quotes the operand values verbatim; bare text does not. Within a form, the positive and negative are byte-identical except the asserted numeral (P1 §5's binding clause, honoured). Across forms, the *only* difference is the operand-quoting clause. **Absence bookkeeping**: the shown form makes the operands present in the claim, which changes the row's absent-numeral count - but it changes it identically for both polarities, so P(label 0 | absent) stays 0.5 inside every form cell. Report the realised 2×2 counts.

**Kill-gate** - **clause 1 is already measured and passing**: the +0.148 lift with AUROC unchanged, `R15_P1_shownwork.json`, 1,600 pairs across four operations, every operation showing the same pattern (sum +0.056/+0.067, difference +0.146/+0.148, ratio +0.235/+0.234, pct_change +0.154/+0.153). **Clause 2, fresh, ~0.1 GPU-h**: is the lift a *count* effect or a *presence* effect? Score the same tuples with 0, 1 and 2 operands quoted on held-out TabFact. **LICENSE if monotone increasing with |Δ(2 quoted − 0 quoted)| ≥ 0.10** - the model is counting present numerals and the parity cross is the right repair. **DOWNGRADE to a construction footnote if non-monotone** - the lift is a template artefact, P1 §6.1's 50%-bare rule suffices, and the extra rows are not worth the diversity they cost.

**Bar** - rides A4's arm, claims no separate finqa movement; A4's PRIMARY and HOLDs stand as registered. Pre-registered ceiling-blind mechanism reading on both draws: held-out bare-vs-shown mean score gap ≤ **0.05** (from +0.148) **AND** |AUROC(bare) − AUROC(shown)| ≤ **0.03** **AND** both per-form AUROCs above their gate baselines. **ANTI-GAMING**: the inverse shortcut is "quotes operands ⇒ negative" - if the shown form's mean score drops below the bare form's by more than 0.05, the lane has inverted the verbosity prior rather than removed it, and that is a REFUTED-as-mechanism outcome, not a win. **Reported, not barred**: the surviving fraction of A4's finqa movement after log-length residualization, compared against the 50%-of-magnitude floor.

**Cost** - **~0.2 GPU-h marginal** (0.1 fresh gate clause + 0.1 post-arm read). Build CPU. Row cost: four rows per tuple instead of two, so at fixed 50,000 rows the tuple count halves to 12,500 and the per-table cap drops from 2 to 1 - which P3 §5 prices at **1.52 rows/document, better than the mix mean of 2.01**. The parity cross is therefore *free in diversity* and is paid for in tuples, which P3 measures as the abundant resource (2,009,526 constructible against 25,000 needed).

**Declared tension** - halving the tuple count halves the operator and table coverage per operator. If the author prefers full tuple coverage, apply the parity cross to a declared 50% slice and report the realised form × label cell counts on both slices. That branch is legal and is stated so the choice is made before the build, not discovered at adjudication.

---

## 6. L3-C4 - TRUNK NUMERACY-RETENTION PROBE AS A BINDING HOLD
*kind: measurement*

**Claim** - Because `R15_probe_P4_numeracy.md` Instrument B measures the frozen trunk separating "which of two numbers is larger" at **99.7% held-out accuracy from 200 training examples** and decoding log-magnitude at **R² = 0.9987** inside range, while the shipped task head reads the same discrimination as a grounding claim at **AUROC 0.5230** on a 30-character two-row table (Instrument D) - so the entire register's thesis is "the information is in the representation and the head does not use it" - reading the same ridge probe on every trained draw will certify whether an arm installed the missing head function **or** destroyed the substrate it was supposed to exploit, a failure mode for which no instrument in the campaign currently exists.

**Why it is needed and why nothing else sees it.** Verdict A of `R14_synthesis.md` records that the margin arm's finqa collapse of **-0.1020** was invisible to `gold_full` (margin 0.8042 against control 0.8040) - "the damage was invisible to every in-domain instrument the arm carried". A lane that teaches the head a surface rule while degrading the trunk's magnitude code would look identical at the arena mean and at `gold_full`, and the campaign would ship a model whose numeric substrate is worse than the pretrained checkpoint it started from. The probe costs 0.1 GPU-h per checkpoint and its baseline is already specified by P4's protocol.

**What is read**, per checkpoint, on the frozen trunk `[CLS]`, held-out, arena-free, gold-free:

1. **Magnitude, interpolation** - ridge on values 1-999, target log₁₀, held-out R²
2. **Comparison** - "Alpha is X and Beta is Y", target `X > Y`, held-out accuracy
3. **Magnitude, extrapolation** - train 1-999, test 10,000-99,999, reported as a *direction* only (P4 records -46.1 and explicitly warns it is a direction, not a coefficient)
4. **P4's three controls, re-run** - permuted labels (must read ≈ 0.50; P4 measured 0.508), 200-row training subset (P4 measured 0.997), ridge λ ×10⁴ (P4 measured 0.985)

**Baseline to establish first** (~0.2 GPU-h, no arm required): the same four reads on **un-fine-tuned mmBERT-base**, on **H105 draws 1 and 2**, and on **H108 draw 1**. P4 measured the shipped checkpoint; nobody has measured whether fine-tuning on the clean mix already eroded the substrate relative to the pretrained model, or whether the admitted H108 lane - the campaign's only replicated finqa lever - moved it. If the clean recipe has already cost, say, comparison accuracy from 1.000 to 0.90, that is a first-order fact about every hypothesis in the register and it is currently unknown.

**Pre-registered thresholds, ceiling-blind, arena-free**, binding as a **HOLD on every training arm in the R14/R15 register**, not only on A4:

- **magnitude interpolation R² ≥ (baseline R² − 0.05)**
- **comparison accuracy ≥ 0.95** absolute, and ≥ (baseline − 0.02)
- breaching either is recorded as **SUBSTRATE DAMAGE** and voids a mechanism claim even where the finqa bar clears; the arm may still be reported as a register-pressure lever, exactly as R14-A5's verdict map handles its own mechanism/lever split
- **REPORT, not bar**: extrapolation R², because P4's own uncertainty register calls it a direction

**What would make this measurement worthless** - if all checkpoints read identically to the pretrained base (no arm moves the trunk probe at all), the instrument has no resolving power and should be retired after one round rather than carried indefinitely. State that in the registration so it is dropped on evidence rather than kept on habit.

**Cost** - **~0.3 GPU-h total**: ~0.2 for the four-checkpoint baseline, ~0.1 per subsequent arm read. Reuses `R15_P4_numeracy_probe.py` unmodified.

---

## 7. Interaction matrix and disjointness declarations

| pair | relation | required declaration |
|---|---|---|
| L3-C1 × R14-A5 (H134) | C1's surface-label orthogonality partially delivers A5's decorrelation mechanism | **must not run in one arm**; declare alongside P3's identical declaration on digit-length parity |
| L3-C1 × P2-A | P2-A is **subsumed** - its 10% register-mismatched positives are one cell of C1's factorial; its frozen in-domain mixture weights are adopted verbatim | record P2-A as merged, credit retained |
| L3-C1 × P2-B | C1's ±1-digit wrong-factor negatives satisfy P3's "parity within 1"; P2-B's AUROC-from-length KILL at 0.55 applies to the sub-block **separately** | report per-sub-block length AUROC in the manifest |
| L3-C1 × P1 §6 scale-as-control | different negative axes (wrong-factor vs wrong-operand); P1's control read is retained **unchanged** on its own axis and becomes C1's VOID clause | both readings reported side by side |
| L3-C2 × R14-A5 | C2 enforces by construction what A5 penalizes statistically | **must not run in one arm**; if both are wanted, C2 first - it is free of a loss term and A5 carries a recorded sign tension |
| L3-C3 × A4 log-length confound clause | C3 attacks the confound's source; the clause is **retained**, not relaxed | report surviving fraction against the 50% floor |
| L3-C3 × P3 diversity | four rows per tuple halves tuples and improves rows/document to 1.52 | report both tuple and document counts |
| L3-C4 × everything | read-only HOLD, no interaction | applies to every arm in the register |
| all four × R14-A2 | P2-D's token budget binds; if A2 Stage 1 adopts 1024 the budget is re-derived, else 512 is binding | A4 must not be built before A2 Stage 1 reports |

**Combined sizing, if all three data candidates ride A4's 50,000 rows**: L3-C1 15%, L3-C2 20%, L3-C3 as a cross over the remaining 65% (or over a declared 50% slice of it). The residual 65% carries P1 §6's operation shares renormalized. Each sub-block reports its own held-out mechanism read, so a null on one is separable from a null on another - which is the whole reason to register them as declared sub-blocks rather than as an undifferentiated lane.

---

## 8. Below the cut - recorded so a later round does not re-derive them

- **Training-only digit-position markup** (writing lane values as `1|234` or with explicit place words). It would install place-value alignment directly, which P2 §5 names as the residual weakness. **Rejected**: the markup would appear at training and never at serving, which is a train/serve skew of exactly the kind H119's instability finding warns about, and shipping it would be a serving transform - the closed line. Recorded, not proposed
- **Right-to-left digit grouping and Abacus-style digit-position embeddings**. Already recorded below the cut by P2-F; both need re-pretraining plus a serving change and both break the sub-400M single-model deliverable. Nothing here re-opens them
- **Numeric-token dropout / masking of evidence digits as augmentation**. Considered and rejected on a measured risk: the model's single most valuable working function is literal presence (verbatim mean 0.9051, AUROC(a vs b) 0.9643, `R14_gate_H133_probe.json`), and masking evidence digits removes the very tokens that function reads. The augmentation would train the model to score confidently in the absence of the evidence it needs, which is the opposite of a grounding library's contract. **L3-C2 is the safe form of the same idea** - it *changes* the evidence numeral rather than *removing* it, keeping the verification target well-posed
- **Claim-side numeral dropout**. Same objection from the other side: a claim with its numeral removed asserts a different proposition, so the label is undefined
- **Canonicalizing every lane value to one surface form**. This is P2's measured degeneracy defect (0% separators, 0% currency, 0% scaled forms) written as a policy. Rejected in favour of L3-C1's factorial
- **Register dressing of TabFact tables into financial clothing** (P3 §4). Not adopted by this lens. P3's own framing is correct - it is aimed at an arena register profile, and its rows assert true arithmetic about Wikipedia sports data wearing a suit. If the author wants it, P3's ≤20% cap with a dressing-off ablation is the right shape; this lens adds nothing to it

---

## 9. What would falsify this lens

Four claims, each with a registered cost.

1. **The copy-detector-is-magnitude-blind claim** (L3-C1's premise). Falsified if the gate reads AUROC(verbatim vs wrong-factor) > 0.60 - the model already rejects decade errors on identical digit strings, P1's "scale already works" extends to this axis, and L3-C1 is redundant. **~0.2 GPU-h**
2. **The evidence-conditioning claim** (L3-C2's premise). Falsified if the gate reads evidence-side derived AUROC > 0.60 - the model does compare claim numerals against evidence numerals and only the arithmetic is missing. Inverted in a more serious way if the *verbatim* evidence-edit clause also reads ≤ 0.60, which would mean the model does not read evidence numerals at all and the R14 diagnosis is too narrow. **~0.25 GPU-h**
3. **The quoted-operand-count claim** (L3-C3's premise). Falsified if the 0/1/2-operand ladder is non-monotone - the +0.148 is a template artefact, not a counting effect, and P1's 50%-bare rule suffices. **~0.1 GPU-h**
4. **The substrate-worth-protecting claim** (L3-C4's premise). Falsified if the ridge probe reads identically on the pretrained base, both H105 draws and H108 draw 1 - the trunk probe has no resolving power across this recipe and the HOLD is ceremony. **~0.2 GPU-h**

Total falsification budget for the whole lens: **~0.75 GPU-h**, all on card 0, all frozen-weights, all held-out, all arena-free.

---

## 10. What this lens does not claim

- **It does not claim any of the four will move finqa.** L3-C1, C2 and C3 ride R14-A4's registered arm and its registered bars; none adds a finqa target, and the register's finqa arithmetic is explicitly non-additive (`R14_synthesis.md` register arithmetic: "the field's optimistic finqa arithmetic is not additive and must not be summed")
- **It does not move any registered bar.** A4's +0.060 PRIMARY, its holds, its anti-gaming clause and its log-length confound clause are unchanged. The only new KILLs are build-time and operate on constructed data before GPU is spent
- **It sets no threshold from an arena statistic.** The finqa surface profile in P2 §6 and the register census in P1 §2.2 are cited as motivation and labelled ANALYSIS ONLY; every mixture weight and every threshold above comes from held-out TabFact, the H108 lane, or the banked frozen-weight probes
- **The [P-L3] unit-header rate is an estimator, not a count.** It was measured on the 3,391 held-out tables because those are the probe surface; the build corpus is TabFact train + FEVEROUS train and the census must be re-run there before L3-C1's header-bound arm is sized
- **FEVEROUS surfaces are unverified** in this session (no network), inheriting P2's caveat. All FEVEROUS-side rates above are TabFact-derived estimates

---

## 11. Reproduction

```bash
cd /home/lab/workspace/private/ai-assistants/groundrails
uv run python tmp/R15_L3_scale.py    # unit-header census, digit-length hist, wrong-factor constructibility
uv run python tmp/R15_L3_scale2.py   # H133 value digit lengths, evidence-side pair constructibility
```

CPU only, ~90 seconds, no GPU, no network, no model loaded. Writes `R15_L3_scale_census.json` and `R15_L3_scale_census2.json` alongside this file. Inputs: `data/external/datasets/dataset-tabfact.zip`, `R14_H133_triples.parquet`. No tracked file was modified.

# R15 probe P3 - the constructible training-signal space for derivation parity

**Role**: signal-space cartographer for R14-A4 (derivation-parity lane). Output: an operator
taxonomy, a positive-construction spec, a serialization decision, the document-diversity
arithmetic, and a quantified inventory of non-table legal sources - all as **amendments to the
registered R14-A4 block**, not a competing registration.

**Discipline**: analysis only. No GPU, no training, Polars throughout, no arena label used for
anything except characterizing the arena. Every number below marked **[P3]** was measured in this
session from repo data; everything else is quoted from the banked artifacts as given.

---

## 0. The four findings, first

**1. The lane as registered does not remove the shortcut - it dilutes it by 3.7% relative.**
Measured on the byte-exact 685,670-row mix reconstruction [P3]: 210,928 rows (30.76%) assert a
claim-side number absent from their evidence, and P(label 0 | absent) over those rows is **0.5944**
against 0.4842 when every claim number is present. Pooling the admitted H108 lane (35,616 absent
rows at P(0|absent) = **0.9332**) gives a training-time shortcut of **0.6433** over 246,544 absent
rows. A 50,000-row parity lane moves that to **0.6192**. The lane must be sized against the
shortcut's *mass*, not against a row-count intuition - the sizing table is in section 6.

**2. Document diversity, not tuple count, is the binding constraint - and half the pool is
already spent.** Of the census's 16,476 admitting tables, **8,438 are TabFact train tables already
in the mix**, where the tabfact group carries 92,585 rows over 13,109 distinct chunks = **7.06 rows
per document** [P3], the most-repeated register in the mix. Only **8,038 tables (48.8%) are new
documents** relative to the clean mix, and only ~5,120 (31.1%) are new relative to mix + H108 lane.
The 2,009,526 constructible tuples sit on a document pool 122 times smaller.

**3. Serialization form is measurable and the incumbent form is the wrong one.** Six forms
profiled on 1,500 TabFact tables against E6's register proxies [P3]: the incumbent pipe form scores
table-marker share **1.000** and digit density **14.13** per 100 chars; a row-per-sentence prose
serialization of the *same* tables scores **0.000** and **7.07**. E6 measures finqa's arena evidence
at 0.009 and 3.93. The structural gap closes; the lexical gap (currency 0.057 vs 0.749, finance
vocabulary 0.002 vs 0.503) does not, and no legal corpus closes it.

**4. Naturally-occurring derivations do not exist at lane scale in any legal corpus.** VitaminC's
370,653 rows yield 5,837 rows whose claim asserts an absent number reachable by a two-operand
arithmetic combination of evidence numbers - but a shuffled-evidence control puts the coincidence
floor at **2.58% against a real rate of 5.10%** [P3], so roughly half are accidents and the true
supply is ~2,800 unverified candidates. WiCE contributes ~175. The lane must be constructed, not
harvested. The harvested rows have a different and better use: they are the only *natural-
distribution* held-out check the lane will ever have.

---

## 1. What the H133 census counts, and what it does not

`R14_gate_H133_census.json` returned 2,009,526 constructible positive tuples over 16,476 admitting
tables and did not fire its 30,000 kill clause. Three scope facts matter for the build and are not
visible in the summary JSON - read from `tmp/R14_gate_H133_census.py` in this session:

- **Four operations only** - sum, difference, ratio, pct_change, all two-operand, all within one
  column
- **Every tuple carries its three negatives already** - the census admits a tuple only if n1
  (numeral-corrupted result), n2 (correct arithmetic over a different row pair) and n3 (operation
  swapped on the same operands) are each constructible and absent. The count is therefore a count of
  complete 4-row groups, not of bare positives
- **Hard enumeration caps** - at most 4 numeric columns per table and 8 numeric rows per column, so
  every figure is a strict lower bound

The census also settles two corpus questions permanently. **InfoTabS is dead for derivation**: 1,431
tables scanned, **48** admitting, 1,037 tuples - infoboxes carry one value per key, so there is
rarely a same-column pair to combine. And **FEVEROUS is the larger fresh-document source**: 23,716
table-bearing evidence blocks scanned, 7,990 admitting, 886,857 tuples, against TabFact's 8,438
admitting tables that the mix has already seen seven times each.

---

## 2. Corruption operator taxonomy for derived values

The existing machinery cannot build this lane. Every operator in `R10-H108_data.py`
(`f_scale_word`, `f_digit_perturb`, `f_pct_pp`, `f_year_shift`, `f_comparative_flip`,
`f_magnitude_shift`) and every operator in the DR generator (`number`, `unit`, `comparative`,
`negation`, per `DR_pilot_gen.py:67`) is a **claim-editing** operator: it takes an already-true
claim and perturbs its surface. None can produce a *correct derivation*, because there is no true
claim to start from - the positive has to be computed from the table. The lane needs a
**construction** pipeline, and the corruption operators become value-selection policies inside it.

Seven negative families. Each takes the positive tuple `(table, column, operator, operand set,
correct result)` and returns an alternative asserted value; the claim text is otherwise **byte-
identical to the positive**.

| # | family | definition | constructibility | why it is in the set |
|---|---|---|---|---|
| N1 | **operand misbind** | same operator, same template, result computed from a *different* row pair; both operand labels in the claim text stay those of the positive | census n2, enforced on all 16,476 tables | the H133 probe's (c) arm - the exact discrimination the shipped model reads at AUROC 0.4924 |
| N2 | **operator swap** | same operands, different operation (sum→difference, ratio→pct_change) | census n3, enforced | separates "knows which numbers" from "knows what to do with them" |
| N3 | **off-by-one-row** | one operand replaced by its *vertically adjacent* row | subset of N1, available wherever the neighbour row is numeric | the hard case of N1 and the one that matches real generator error; a random misbind is often distinguishable on magnitude alone |
| N4 | **scale error** | correct result ×10 / ×100 / ÷10, or the accompanying unit word swapped (thousand↔million↔billion) | near-universal | carries `f_scale_word` / `f_magnitude_shift` forward onto the derived value |
| N5 | **sign flip** | a−b asserted as b−a; pct_change negated; "increase" asserted as "decrease" with the same magnitude | defined for difference, pct_change and comparative frames - about half of operator draws | maps onto E1's comparative register, which `f_comparative_flip` already touches on non-derived claims |
| N6 | **plausible-magnitude distractor** | a value drawn uniformly in ±2% to ±15% of the correct result, absent from the evidence, and not equal to any correct derivation over the same column | always available | the control for "the model only checks that the number looks the right size" - without it a lane can be passed by magnitude heuristics |
| N7 | **numeral corruption of the correct result** | last-digit perturbation of the correct value | census n1, enforced | **the shortcut family** - see the cap below |

**Binding cap on N7.** N7 is the same operator class as `f_digit_perturb`, which produced roughly
45,000 of the H108 lane's corruption negatives and 7,862 number-change deltas in the DR lane, and
the H108 lane is the artifact measured at P(0|absent) = 0.9332. Emitting N7 at a third of the
negative budget re-imports the defect inside the repair. **Cap N7 at 10% of negatives** and record
its realised share; the parity accounting (section 3) is computed over all seven families together.

**Binding parity rules** - these are what make the lane a repair rather than another shortcut:

- **Text parity** - the positive and its paired negative differ in the asserted numeral and nothing
  else. Same template instance, same operand labels, same evidence string, same order
- **Absence parity** - both asserted values must be absent from the serialized evidence, so the
  presence feature carries zero information inside the lane by construction
- **Digit-length parity** - `|digits(positive) − digits(negative)| ≤ 1`, enforced by rejection
  sampling. Without it, digit count replaces digit presence as the cue. This also means the lane
  delivers part of R14-A5's decorrelation mechanism for free and **confounds A5 if both run in one
  arm** - declare disjointness alongside A4 amendment (iv)'s A6 clause
- **One negative per positive** - the tuple carries four constructible negatives but emits one,
  sampled from the family distribution, so P(0|absent) = 0.5 exactly. Emitting all four gives 0.75
  and inverts the defect

**Operator availability, measured** [P3]. Extended census over 3,000 sampled TabFact train tables,
1,988 admitting: share of admitting tables supporting each positive operator -

| operator | TabFact train | TabFact test+val | FEVEROUS |
|---|---:|---:|---:|
| difference | 99.7% | 99.4% | 99.8% |
| pct_change | 99.9% | 99.8% | 99.8% |
| share_of_total | 99.8% | 99.7% | 99.8% |
| triple_sum | 97.8% | 98.2% | 99.1% |
| col_total | 97.2% | 97.2% | 97.4% |
| col_mean | 97.0% | 97.2% | 95.5% |
| ratio | 96.6% | 96.7% | 97.8% |
| sum | 96.1% | 96.5% | 97.6% |
| col_max_minus_min | 76.5% | 75.7% | 71.8% |
| col_count_above_mean | 38.3% | 35.4% | 41.7% |

Nine of ten positive operators are available on more than 71% of admitting tables, so a per-table
cap of 4 can be filled with four *operator-disjoint* tuples on the large majority of the pool.
Adding six operators beyond the census's four raises the sampled per-table median from 110 to
**143** and the TabFact-train tuple estimate from 1.12M to ~1.71M - but note these extended counts
do **not** re-enforce the n1/n2/n3 constructibility filter, so they are not directly comparable to
the census total. The multiplier is not the point; **operator disjointness within a low per-table
cap is**.

---

## 3. Positive construction and surface forms

The H133 probe used exactly one claim template - `"The combined {col} of {ka} and {kb} is {v}."`
(`R14_H133_probe.py`). A 25,000-tuple lane built on one template per operator teaches the template.

**Recommendation: one template per emitted pair, drawn from a per-operator bank of ten, with the
positive and its negative sharing the same template instance.** "Surface forms per tuple" is
therefore **one**, used twice. The alternative - emitting the same tuple under several templates -
buys nothing the diversity budget can afford, because templates are cheap and documents are not.

Sizing check: ten templates × ten operators = 100 template slots; a 25,000-tuple lane puts ~250
tuples on each. The admitted H108 lane runs 33,176 corruption rows across 6 families = 5,529 rows
per family, so 250 per template is an order of magnitude finer-grained than the artifact that
already works.

Template bank shape, per operator - the variation that matters is **frame**, not synonymy:

- direct assertion - "The combined X of A and B is V."
- aggregate frame - "Across the listed rows, total X is V."
- comparative frame - "A's X exceeds B's by V."
- relative frame - "A accounts for V percent of total X."
- temporal frame - "X changed by V percent between A and B."
- hedged frame - "X for A and B together comes to about V."
- question-answer frame - claim phrased as the answer sentence a RAG generator would emit
- three-operand frame - "A, B and C together report X of V."
- unit-bearing frame - "Total X is V thousand."
- negated frame - "X for A and B does not exceed V." (paired with a negative that flips the bound,
  not the numeral)

The negated frame is worth its slot: it is the only one where the correct positive asserts a
*relation* rather than a value, and E4's finqa forensics found the model scoring a correct
conclusion at 0.0506 against its own premise sentence at 0.8943.

---

## 4. Serialization - and the H119 verdict, engaged

E6's measurement is unambiguous: finqa's arena evidence carries table markers on **0.9%** of
windows and tatqa on **0.0%** - it is serialized 10-K prose - while TabFact supplies 99.7% of every
table-marked row in the mix and is 100% pipe-delimited by construction. The mix contains **zero**
prose-serialized table rows. A lane built on TabFact and FEVEROUS that keeps the pipe form teaches
"read a pipe table" for a third time.

**Six forms, profiled on the same 1,500 TabFact tables** [P3], against E6's arena targets:

| form | len med | digden med | table-marker share | currency share |
|---|---:|---:|---:|---:|
| `pipe` (incumbent) | 871 | 14.13 | 1.000 | 0.057 |
| `markdown` | 970 | 12.67 | 1.000 | 0.057 |
| `keyvalue` | 1399 | 9.20 | 0.001 | 0.056 |
| `json_records` | 1500 | 7.33 | 0.009 | 0.056 |
| `narrative` | 1198 | 10.96 | 0.000 | 0.050 |
| `row_prose` | 1500 | 7.07 | 0.000 | 0.055 |
| *arena finqa (E6)* | *1500* | *3.93* | *0.009* | *0.749* |
| *arena tatqa (E6)* | *340* | *4.35* | *0.000* | *0.533* |
| *mix tabfact (E6)* | *881* | *13.60* | *1.000* | *0.058* |

`row_prose` matches the finqa target on length exactly and on table-marker share exactly, and halves
the digit-density gap. It cannot touch the lexical gap - Wikipedia sports and census tables have no
currency and no finance vocabulary, and E6 already ruled that no legal corpus supplies that register.

### Legality of the serialization choice - stated plainly

Choosing `row_prose` **because finqa's evidence is prose is tuning on an arena statistic** and is
not admissible as the lane's justification. The registration must carry the independent
justification verbatim, as A4 amendment (ii) requires for the lane itself:

> A grounding library receives retrieved evidence in whatever form the retriever emits - markdown,
> JSON records, key-value blocks, or prose that a document parser has already flattened. A model
> trained on one serialization of tabular content and no others is a shipping defect independent of
> any benchmark. The mix contains 92,830 table-marked rows, 99.7% of them pipe-delimited, and zero
> prose-serialized table rows.

The arena profile above is recorded as **corroboration only** and must not appear in any bar.

### H119 is closed and stays closed - why this is not it

H119 (read-time numeric canonicalization) is REFUTED and is not re-proposed. The H119 evidence does
not transfer to a build-time serialization choice, for three checkable reasons:

1. **H119 was a serving wrapper on frozen weights.** It changed the input distribution of a function
   that could not adapt, so whatever mismatch it introduced was pure covariate shift with no
   compensating fit. This proposal changes the *training* distribution and leaves the serving path
   (`src/groundrails/semantic_ov.py`) byte-identical - the deployed function receives exactly the
   bytes it receives today
2. **H119 transformed arena evidence at inference.** This transforms lane evidence at build time and
   never touches arena text at all
3. **H119 had no control that could have exonerated it.** This one does, and it is registered below

**Registered falsifier**: build a `pipe`-serialized twin of the lane at equal rows, equal documents,
equal operator mix and equal seed. If the twin reproduces the prose lane's finqa movement to within
half its magnitude, **serialization is not the mechanism** and the prose forms are dropped from the
build. This falsifier costs one extra training draw and must be priced into the arm before it runs.

### Form assignment - one form per document

Assign each table **one** serialization form for the whole lane, drawn from a fixed distribution,
carried as a lane sub-tag. Do not emit the same table under multiple forms: the scarce resource is
documents (section 5), and three forms of one table is one document's worth of content occupying
three documents' worth of budget.

Proposed distribution, justified by mix composition rather than arena profile: `row_prose` 30%,
`narrative` 25%, `json_records` 15% (the form E4 found verbatim in arena evidence as
`[["label","value"]]`, and the form of RAGTruth Data2txt at 6.18% of the mix), `keyvalue` 10%,
`markdown` 10%, `pipe` 10% (the incumbent, retained so the lane does not *lose* the surface the
model already reads).

**Exception, declared and separate**: a 1,000-table invariance sub-block emitting three forms of the
same table with identical claims, held disjoint from the main lane and reported separately. This
revives L3-C2's below-the-cut mechanism at 3,000 rows instead of ~16 GPU-h, and gives the
0.2 GPU-h frozen invariance probe a training-side companion.

### Register dressing - an option, flagged as arena-informed

Attaching a unit to each numeric column and a filing-style frame to the serialization moves three of
four proxies into the finqa band [P3]: digit density 7.00 → **4.87** (target 3.93), currency share
0.061 → **0.810** (target 0.749), finance-vocabulary share 0.002 → **0.151** (target 0.503), table
markers 0.000 throughout.

This is **directly aimed at an arena register profile** and must not be load-bearing. If it is
built, build it as a declared sub-block at no more than 20% of the lane, with a dressing-off ablation
in the same arm, and record that the dressed rows assert quantities about Wikipedia sports and census
data wearing financial clothing - the arithmetic is true, the domain is fiction.

---

## 5. The diversity arithmetic

E6's verdict is that document diversity, not row count, is what the registers lack - the entire
procedural register is taught by 27 distinct negative documents and the financial register by 311.
The lane inherits that constraint exactly.

**Document supply, measured** [P3]:

| pool | tables | admitting | status |
|---|---:|---:|---|
| TabFact train | 13,182 distinct ids / 13,143 distinct table texts | 8,438 | **already in the mix** - 92,585 rows over 13,109 chunks = 7.06 rows/doc |
| FEVEROUS train (table-bearing) | 23,716 | 7,990 | fresh to the mix; ~2,918 overlap the H108 challenge-filtered slice already in the admitted lane |
| InfoTabS | 1,431 | 48 | dead |
| **total lane pool** | | **16,476** | 8,038 new to the mix (48.8%); ~5,120 new to mix + H108 (31.1%) |
| TabFact test+val | 3,391 | 2,229 | **reserved** - the H133 probe surface, must not be consumed |

The reservation is binding. `R14_H133_probe.py` drew its 2,000 triples from TabFact test+validation
precisely because those tables are `table_id`-disjoint from the train split that built the mix and
the H108 lane. Building the lane on train + FEVEROUS keeps the 0.4924 AUROC probe a valid held-out
instrument and gives the arm a free post-hoc re-read on the same 2,000 triples. Building the lane on
test/val destroys that instrument for 2,229 documents of gain.

**Per-table cap.** The census cap curve over the pool is 1 → 16,476 tuples, 4 → 65,591, 16 →
255,749, 64 → 900,983. A pair-balanced lane needs one tuple per pair-pair, so an N-row lane consumes
N/2 tuples:

| lane rows | tuples needed | required per-table cap | rows per document | comparison |
|---:|---:|---:|---:|---|
| 25,000 | 12,500 | 1 | 1.52 | better than the mix mean (2.01) |
| 50,000 | 25,000 | 2 | 3.03 | between the mix mean and the tabfact group |
| 100,000 | 50,000 | ~3 | 6.07 | at the tabfact group's 7.06 |
| 150,000 | 75,000 | ~5 | 9.10 | worse than any group in the mix |
| 200,000 | 100,000 | ~6 | 12.14 | worse than ragtruth_en's 6.00 by 2x |

**Recommended build point: 50,000 rows at per-table cap 2, every one of the 16,476 tables used, the
two tuples operator-disjoint and column-disjoint where the table allows it.** That is A4's
registered size, it sits at 3.03 rows per document against the mix's 2.01, and it uses the whole
pool rather than a dense slice of it. Cap 2 is available on essentially the entire pool: the census
cap-4 figure of 65,591 implies ~32,000 tuples at cap 2 against the 25,000 needed.

**Do not chase tuple count.** The 2,009,526 figure is real and irrelevant: past cap 4 the lane is
re-reading the same 16,476 documents, and E6's measured failure mode is documents, not rows.

---

## 6. The dilution amendment - the number A4 does not carry

A4 fixes P(0|absent) at 0.5 **inside the lane**. The gradient sees the pooled distribution.

**Measured on the mix reconstruction** [P3], claim numbers canonicalized and matched against the
evidence chunk:

| group | rows | absent-number rows | P(label 0 \| absent) |
|---|---:|---:|---:|
| vitaminc | 370,653 | 117,861 | 0.5501 |
| psiloqa | 61,712 | 21,008 | **0.9515** |
| ragtruth_cn | 15,090 | 5,677 | 0.6387 |
| ragtruth_pl | 15,090 | 6,936 | 0.5696 |
| ragtruth_hu | 15,090 | 6,864 | 0.5692 |
| ragtruth_it | 15,090 | 7,195 | 0.5607 |
| ragtruth_de | 15,090 | 7,024 | 0.5621 |
| ragtruth_es | 15,090 | 7,349 | 0.5583 |
| ragtruth_en | 15,090 | 7,594 | 0.5561 |
| ragtruth_fr | 15,090 | 7,093 | 0.5542 |
| halueval | 40,000 | 5,015 | 0.5444 |
| tabfact | 92,585 | 11,312 | 0.5387 |
| **clean mix** | **685,670** | **210,928 (30.76%)** | **0.5944** |
| H108 lane (admitted) | 61,184 | 35,616 | **0.9332** |
| **mix + H108 lane** | 746,854 | **246,544** | **0.6433** |

For reference, P(label 0 | every claim number present) over the mix is **0.4842** - the shortcut's
whole magnitude is the 0.110 gap between 0.5944 and 0.4842, and the H108 lane widens the operating
figure to 0.6433. Two carriers hold most of the purity: **psiloqa at 0.9515 over 21,008 rows** and
the **H108 lane at 0.9332 over 35,616 rows**.

**Dilution from an N-row parity lane at exactly 0.5**, all rows absent-bearing by construction:

| lane rows | pooled P(0\|absent), mix only | pooled P(0\|absent), mix + H108 |
|---:|---:|---:|
| 0 | 0.5944 | 0.6433 |
| 25,000 | 0.5844 | 0.6301 |
| 50,000 | 0.5763 | **0.6192** |
| 100,000 | 0.5640 | 0.6020 |
| 150,000 | 0.5552 | 0.5891 |
| 200,000 | 0.5485 | 0.5791 |
| 300,000 | 0.5390 | 0.5647 |

**The registered 50,000-row lane buys a 0.0241 absolute / 3.7% relative reduction in the shortcut's
purity.** That is not a reason to kill it - the lane's mechanism is not only dilution, it is the
first supervision in the entire mix that *contradicts* the shortcut on matched text, and 25,000
contradicting pairs at digit-length parity is a different object from 25,000 diluting rows. But the
registration should stop describing the lane as fixing P(0|absent) and describe what it does:

- **inside the lane** - P(0|absent) = 0.5 exactly, presence carries zero information
- **pooled** - 0.6433 → 0.6192, a 3.7% relative reduction
- **contradiction mass** - 25,000 minimal pairs where the shortcut's prediction is wrong half the
  time on byte-identical claim text, against zero such pairs today

**Three sizing branches, for the author to pick before the build** - all three are legal, and the
choice is a cost decision, not a measurement:

- **If the mechanism is contradiction** → 50,000 rows at cap 2 is right. Diversity 3.03 rows/doc.
  ~13 GPU-h as registered. The dilution number is reported, not targeted
- **If the mechanism is dilution** → the lane must reach ~200,000 rows to bring pooled purity under
  0.58, at cap 6 and 12.14 rows per document, which is worse document diversity than any group in
  the mix and directly contradicts E6's verdict. **Not recommended**, and stated here so it is
  rejected on the arithmetic rather than never considered
- **If both** → 100,000 rows at cap 3 (6.07 rows/doc, at the tabfact group's density, pooled 0.6020)
  is the compromise, at roughly double the registered GPU cost

Raising the lane's positive fraction above 0.5 to accelerate dilution is available and is **not
recommended**: it manufactures the inverse shortcut ("absent implies supported"), which is exactly
what A4's binding anti-gaming clause exists to catch.

---

## 7. Non-table legal sources - inventory and verdicts

Detector used throughout: a claim number is *absent* if its canonical form does not appear in the
evidence; it is *derivable* if it equals `a+b`, `a−b`, `b−a`, `a/b`, `b/a`, or either percent change,
for some pair among the first 40 evidence numbers. Reported with its false-positive floor.

### VitaminC - REJECTED as lane mass, ADMITTED as a natural held-out probe

Measured over all 370,653 train rows [P3]:

| quantity | value |
|---|---:|
| claims containing a number | 163,133 (44.0%) |
| claims asserting a number absent from evidence | 117,729 (31.8% of rows) |
| ... SUPPORTS / NEI / REFUTES | 52,957 / 18,838 / 45,934 |
| ... P(label 0 \| absent) | 0.5501 |
| absent **and** two-operand derivable | 5,837 |
| ... SUPPORTS / NEI / REFUTES | 2,813 / 806 / 2,218 |
| ... distinct evidence documents | 3,548 |

**Shuffle control** (60,000-row sample, claim paired with a random other row's evidence) [P3]: real
derivable rate **5.10%** of absent rows against a shuffled rate of **2.58%**. Roughly **half of every
detected derivation is arithmetic coincidence**, and applying the floor to the full corpus leaves
~2,800 true candidates.

Two further disqualifications. VitaminC's label adjudicates the *revision*, not the arithmetic - a
SUPPORTS row whose claim number happens to equal an evidence sum is not evidence that the annotator
verified the sum. And 2,800 unverified candidates is 5.6% of a 50,000-row lane.

**Verdict**: not a lane source. **Use it as the lane's only natural-distribution held-out check** -
verify a 500-item sample of the 5,837 by hand or by judge, and read the trained lane on the verified
subset. The lane is otherwise entirely synthetic and has no instrument that says its arithmetic
supervision transfers to derivations that occur in the wild.

### WiCE - REJECTED as lane mass, ADMITTED as a second probe

| split | rows | numeric claims | absent | derivable (uncorrected) | distinct evidence docs |
|---|---:|---:|---:|---:|---:|
| `claim_train` | 1,260 | 701 | 276 | 168 | 168 |
| `subclaim_train` | 3,470 | 906 | 313 | 183 | 147 |

Applying the same ~50% coincidence floor leaves ~175 real derivations combined - 0.35% of a 50,000-
row lane. Two properties make them worth keeping anyway: the rows sit at **essentially one row per
document**, the best diversity per row of any source measured here, and WiCE's evidence is real
long-form web prose, which is the register the arena actually has and the mix does not.

One caveat that must be recorded if WiCE is used as a probe: `partially_supported` dominates the
numeric slice (106 of 168 claim rows, 68 of 183 subclaim rows), so the binary collapse is lossiest
exactly where the derivations live.

### RAGTruth train, Data2txt - ADMITTED as a small register-matched sub-block, with a caution

Measured [P3]: 5,298 rows per language, **883 distinct Yelp business records**, every one carrying
`business_stars` and exactly **three** `review_stars`. Derivable per record without inventing
anything: mean, sum and range of the three review scores; `business_stars` minus the review mean;
count at or above 4; count at or below 2; the review-date span; and boolean-attribute counts
(parking options set true, ambience flags set true).

That is ~8 aggregate slots × 883 records ≈ **7,000 tuples over 883 documents**, and it is the only
source measured here that is *natively* key-value serialized and already register-matched to 6.18% of
the mix.

**Caution, and it is the reason for the cap**: those 883 documents already appear at 5,298 rows per
language across eight languages = 42,384 mix rows, the most-repeated document set in the entire mix.
Adding 7,000 rows over the same 883 documents pushes that register further in exactly the direction
E6 says is wrong. **Cap this sub-block at ~2,600 rows (3 pairs per record).**

### SciTab, InfoTabS, and the finance-register hole

SciTab is already in the admitted H108 lane at 1,173 rows over 208 documents - negligible and
already spent. InfoTabS admits 48 derivation tables out of 1,431 and is dead. And the finance
register has no legal source at all: E6's contamination wall removes ConvFinQA, TAT-HQA,
FinanceBench, FinTabNet and EDGAR, and the mix's full finqa signature matches **31 rows out of
685,670**. Synthetic dressing (section 4) is the only route, and it is fiction wearing a suit.

---

## 8. Recommended amendment to R14-A4, in build order

1. **Corpus** - TabFact train (8,438 admitting tables) + FEVEROUS train (7,990). InfoTabS dropped
   at 48. TabFact test+validation (2,229 admitting) **reserved** as the H133 re-read surface
2. **Size and cap** - 50,000 rows = 25,000 pairs at per-table cap 2, operator-disjoint and
   column-disjoint per table, all 16,476 tables used → 3.03 rows/document
3. **Operators** - ten positive operators (four census + col_total, col_mean, col_max_minus_min,
   col_count_above_mean, triple_sum, share_of_total); seven negative families N1-N7 with **N7
   capped at 10%** of negatives
4. **Parity** - text parity, absence parity, digit-length parity within 1, one negative per positive
5. **Templates** - one instance per pair shared by both polarities, from a bank of 10 frames per
   operator (~250 tuples per template slot)
6. **Serialization** - one form per table: `row_prose` 30 / `narrative` 25 / `json_records` 15 /
   `keyvalue` 10 / `markdown` 10 / `pipe` 10; plus a separate 1,000-table three-form invariance
   sub-block (3,000 rows), reported separately
7. **Optional, declared, ablated** - register dressing on ≤20% of rows with a dressing-off arm
8. **Sub-block** - RAGTruth Data2txt aggregates, ≤2,600 rows, natural key-value records
9. **Probes, held out and free** - re-read the 2,000 H133 triples (AUROC(b vs c) 0.4924 baseline);
   read the verified VitaminC natural-derivation sample; read WiCE's numeric slice
10. **Reporting** - realised P(0|absent) inside the lane, pooled with mix + H108, and the realised
    per-family and per-form shares

**Falsifiers to register with the arm** - each is a claim this probe makes that can be wrong:

- **Serialization** - a pipe-serialized twin at equal rows and documents reproduces the finqa
  movement to within half its magnitude → serialization is not the mechanism, drop the prose forms
- **Diversity** - a cap-6 build at 50,000 rows over 8,238 tables (half the pool, twice the depth)
  matches the cap-2 build → documents are not the constraint and E6's diversity verdict does not
  transfer to constructed lanes
- **Contradiction vs dilution** - the 0.6192 pooled figure is reported at adjudication; if the finqa
  movement tracks lane size rather than pair count across the 25,000 / 50,000 sizes, the mechanism
  is dilution and the arm is mis-specified
- **N7** - per-family score breakdown on the held-out triples; if N7 rows are the easiest negatives
  by a wide margin, the lane is still being solved by the presence shortcut on a 10% slice

**What this probe does not claim.** It sets no bar - A4's bars stand as registered. It measures no
arena quantity into any threshold. It does not price the GPU cost of the larger sizing branches
beyond noting they scale roughly with rows.

---

## 9. Reproduction

```bash
cd /home/lab/workspace/private/ai-assistants/groundrails
uv run python tmp/R15_P3_nontable.py     # VitaminC absent/derivable inventory
uv run python tmp/R15_P3_nontable2.py    # shuffle control, WiCE, RAGTruth shape
uv run python tmp/R15_P3_diversity.py    # TabFact split ids, mix document diversity
uv run python tmp/R15_P3_optax.py        # extended 10-operator census, cap curves
uv run python tmp/R15_P3_shortcut.py     # mix-wide P(0|absent) and lane dilution
uv run python tmp/R15_P3_serial.py       # six serialization forms vs E6 proxies
uv run python tmp/R15_P3_dress.py        # register dressing vs the finqa lexical profile
```

Scripts live in `tmp/` (gitignored, analysis-only); JSON outputs alongside them. No tracked file was
modified, no model was loaded, no GPU was used.

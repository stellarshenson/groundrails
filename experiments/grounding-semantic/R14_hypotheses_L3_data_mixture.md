# R14 hypotheses - LENS L3: DATA / MIXTURE

**Lens**: remediation through training-data composition - legal corpora, mixture reweighting,
DANN group design, targeted negatives from the admitted DR/H108 corruption engine.

**Discipline**: ANALYSIS ONLY. Every number below is either read from a banked artifact on disk
or computed here in Polars from `tmp/R14_E6_mix.parquet` (the byte-exact 685,670-row mix
reconstruction), `experiments/grounding-semantic/R12-H121_gateA_scores.parquet`,
`R10-H108_pairs.parquet`, or the banked `*_result.json` reads. No GPU was used. No arena
quantity sets any lane's size, mix or threshold; the arena appears only as the protocol
baseline the bars are stated against, and as the diagnostic surface that named the mechanisms.

**Lens number**: the launching spec named the lens ("DATA / MIXTURE") but not its index. `L3` is
self-assigned so the ids are well-formed; re-label freely at synthesis.

---

## 0. Numbers verified from disk before anything was proposed

Protocol constants (recomputed from `R9-H105_windowed_result.json` and
`R9-H105_draw2_windowed_result.json`, PRIMARY windowed decomposed-min read):

| quantity | draw 1 | draw 2 | pair mean |
|---|---|---|---|
| arena mean | 0.70471 | 0.70151 | **0.70311** |
| finqa | 0.6489 | 0.6176 | 0.6333 |
| tatqa | 0.7034 | 0.7606 | 0.7320 |
| pubmedqa | 0.6201 | 0.5925 | 0.6063 |
| emanual | 0.6883 | 0.7070 | 0.6976 |
| techqa | 0.6934 | 0.6745 | 0.6840 |
| delucionqa | 0.7975 | 0.8358 | 0.8166 |
| gold_full (claim-level, OOD) | 0.8788 | 0.8240 | 0.8514 |
| RAGTruth EN / non-EN | 0.8382 / 0.8402 | 0.8361 / 0.8337 | 0.8372 / 0.8370 |

Admitted H108 lane, same read: finqa 0.7291 / 0.7072, arena mean 0.70618 / 0.70373,
pubmedqa 0.5907 / 0.5575, tatqa 0.7391 / 0.7482, gold_full 0.8589 / 0.8579; lane 61,184 pairs
over 4 tags (`quant_corrupt` 33,176, `quant_infotabs` 16,466, `quant_feverous` 10,369,
`quant_scitab` 1,173); 746,854 mix rows, 16 DANN groups; 21,695 s and 10,886 s of training.

Seed sigma per subset (E3): finqa 0.0421, tatqa 0.0414, delucionqa 0.0432, techqa 0.0265,
emanual 0.0211, pubmedqa 0.0217; arena-mean seed SD 0.0033. H126's paired-delta target is
SD ≤ 0.014, and **every bar below is written on the assumption that H126 pairing is in force**;
unpaired, none of them is enforceable.

### Three measurements this lens took that are not in E1-E6

**M1 - the admitted lane teaches the shortcut that E4 says finqa punishes.**
Over the 326,712 mix rows whose claim contains a multi-digit number, computed here:

| population | P(label = 0 \| claim number ABSENT from evidence) | P(label = 0 \| number PRESENT) |
|---|---|---|
| clean public mix (685,670 rows) | **0.610** (n = 176,380) | 0.427 (n = 150,332) |
| R10-H108 admitted lane (61,184 rows) | **0.946** (n = 35,134) | 0.440 (n = 9,603) |

Against this, E4 measures the arena: 362 of 563 finqa scored sentences (64.3%) assert a number
present in no window, and **75.7% of those are gold-SUPPORTED**. The training distribution and
the target register disagree on the dominant claim shape, and the campaign's one admitted lane
sharpens the disagreement to 94.6% purity. H108 still helps finqa (+0.0849 pair mean) because it
repairs relational binding; M1 says it pays for that with a literal-presence prior, and it is a
direct predictor of E4's compression finding (finqa supported-sentence mean 0.538; absent-number
sentences 0.487 against 0.641 for all-numbers-present).

**M2 - the arena's two numeric subsets are decided by a table serialization the mix contains
zero rows of.** Computed here over `R12-H121_gateA_scores.parquet` and the mix:

| population | rows with `[[`-style bracket/list-of-lists tables | rows with ≥ 3 pipes |
|---|---|---|
| training mix, all 685,670 rows | **0.0000** | 0.1353 (99.7% TabFact, pipe by loader construction) |
| arena tatqa rows | 0.2472 | 0.0000 |
| arena finqa rows | 0.1929 | 0.0093 |
| arena, all other 8 subsets | ≤ 0.0022 | ≤ 0.0108 |

At the **score-deciding** window (argmax window per sentence): **tatqa 66.0%** and **finqa 40.3%**
of sentences are decided by a bracket-serialized table; every other subset is ≤ 0.06%. Mean
sentence score when the deciding window is bracket-serialized versus prose: tatqa 0.505 vs 0.680,
finqa 0.528 vs 0.551. This also corrects E6: its `table` mask needs pipes, tabs, `"key":` runs or
`Row N` headers, so it scores finqa at 0.9% tabular. finqa's tables are there - they are
`[["total scheduled maturities of long term debt","77724"], ...]` (E4 resp 5, verified verbatim
in the parquet) - the mask simply cannot see that form, and neither can the training mix.

**M3 - train and serve disagree on premise length by 4.4x at the median.**

| population | median premise chars | share ≥ 800 chars |
|---|---|---|
| training mix (all rows) | **226** | 28.4% |
| arena score-deciding windows | **1,004** | **59.0%** |
| VitaminC alone - 54.06% of all training rows | 139 | 0.08% |

Per-group training medians: vitaminc 139, psiloqa 281, halueval 817, tabfact 881, ragtruth_cn
907, ragtruth EN + 6 translations 1,500. VitaminC is 54.06% of rows but only **15.34% of premise
characters**; TabFact is 13.50% of rows and 23.14% of characters. E5's independent measurement
sits on top of this: MAX_LEN 512 already truncates 22.1% of scored pairs and 46.4% of techqa's
deciding pairs, so the serving premise is both longer than training and clipped.

---

## 1. What this lens will not propose

Closed lines, not re-proposed: weight averaging (H118 soup, H120 EMA), head fusion (H104/H106),
token-head-as-primary (H102), GroupDRO→DANN curriculum (H95/H96), lambda-sweep proxy (H99),
sentence-exclusion (P-B), the H119 numeric-canonicalization serving wrapper, distractor-window
negatives via lexical certification (H121), layer-mix head input (H123), forced subset balance,
training on RAGBench-train.

Also declined on this lens's own evidence:

- **delucionqa as a target.** E2 measures its 2-draw noise at ≈ ±0.10 and its lane-effect
  variance as negative; E3 finds 85% of its config-to-config variance is seed noise and zero of
  14 trained configs move it past 2 sigma; E6 shows its register is 1,645x under-represented, but
  the register import that would fix that is exactly H107, REFUTED, with finqa at 0.4261 and
  gold_full collapsing 0.8514 → 0.7360. delucionqa appears below only as a **diagnostic**, never
  as a bar (E2 branch 4)
- **A procedural / manual register lane.** H107 already ran this with deterministic span-anchored
  corruptions over `multidoc2dial` and `lettucedetect-code-hallucination` and produced the
  campaign's most violent failure. E3's PC1 (42.6% of variance) is precisely that register axis,
  and the seed-replicate covariance shows delucionqa~techqa at r = -0.981 *with no intervention
  differing at all* - training runs wander along that axis by themselves. Importing more of one
  end of it is not a mechanism, it is a push
- **A new financial corpus.** E6's provenance note is binding: FinQA and TAT-QA descend from
  corporate annual reports and EDGAR filings, so ConvFinQA, FinanceBench, FinTabNet, MultiHiertt
  and raw-EDGAR packagings all sit at material 13-gram containment risk against the arena's own
  documents, and nothing financial is on local disk to gate. L3-C1 and L3-C2 below deliberately
  target the *claim semantics* and the *evidence surface form* rather than the domain, because
  both are reachable inside corpora that are already admitted and already gated

---

## 2. L3-C1 - DERIVATION-PARITY LANE

**Named mechanism**: absent-number-implies-negative shortcut removal by symmetric
derived-value supervision.

### Claim

Because the training distribution makes "the claim asserts a number that is not in the evidence"
a 0.610 predictor of label 0 and the admitted H108 lane sharpens it to 0.946 (M1), while the
arena's numeric register gold-supports 75.7% of exactly those sentences (E4) and scores them
0.487 against 0.641 for all-numbers-present, adding ~50,000 pairs in which a **correctly derived**
value that appears nowhere in the evidence is a POSITIVE and a **wrongly derived or wrongly bound**
value that also appears nowhere is a NEGATIVE will lift finqa to a 2-draw mean ≥ 0.6933 (+0.060
over the 0.6333 paired control) with sign agreement on both draws, while the arena mean holds at
≥ 0.7031, pubmedqa holds within 0.06 of control, and gold_full holds ≥ 0.8414.

### Mechanism detail

The lane is built over evidence already in the mix and already provenance-gated by H108: TabFact
and InfoTabS tables, FEVEROUS tables and SciTab scientific tables. For each table the builder
selects two or three numeric cells and emits four row types over the *same* premise:

- **D+ derived positive** - the claim asserts a value computed from the cells (sum, difference,
  ratio, percent change, share-of-total, min/max over a column) and states the derivation in
  natural language. The asserted value is verified absent from the premise string before emission
- **D- operand-misbinding negative** - the identical arithmetic performed on the wrong cells: the
  right operator over operands drawn from a different row label or a different column header. This
  is E4 resp 200 verbatim as a training row - `$6.2bn / $38.8bn` where both operands are in the
  window and belong to different line items, 13.4% of finqa's entire discordance
- **D- operator-swap negative** - correct operands, wrong operator (difference reported as a sum,
  ratio inverted, percent-change sign flipped). Covers E4 resp 31's direction reversal
- **D- scale/period negative** - the derived value at the wrong scale or the wrong period. This
  overlaps H108's `f_magnitude_shift` / `f_year_shift` by design, to keep the derived and literal
  branches on a common negative surface rather than letting "derived" become a positive cue

Every one of the four types has an absent-from-evidence number, so the shortcut carries **zero**
information inside the lane, and P(label = 0 | absent) inside the lane is fixed at 0.5 by
construction. The lane enters as one or two new DANN groups following the H108 convention
(`tag` → group), which keeps the group design consistent with the admitted precedent.

Two anti-leniency properties matter for admissibility. First, the lane's positives are *harder*
than its negatives lexically - both sides share the operand tokens, so nothing here rewards
partial-overlap firing (E2's REJECT-ON-MECHANISM branch). Second, the negative classes are 3:1
against the positive class within the derived family, so the lane cannot be satisfied by simply
accepting derivations.

### Legality

- **No new corpus, no new admission.** TabFact, InfoTabS, FEVEROUS and SciTab are already in the
  H108 lane on disk (`R10-H108_pairs.parquet`, 4 tags) and already passed that lane's provenance
  gate. Their evidence is Wikipedia tables, Wikipedia infoboxes and scientific-paper tables -
  zero overlap with FinQA / TAT-QA / any RAGBench source corpus
- **No RAGBench data, no private gold, no derivative.** The arithmetic is computed from cells the
  mix already contains
- **The mechanism's justification is independent of the arena.** groundrails ships as a
  claim-grounding library for agentic RAG. A scorer that flags every arithmetically derivable
  quantity as unsupported produces a false hallucination alarm on the single most common shape of
  numeric RAG answer ("revenue grew 12%", "the total was $77,724"). Teaching derived-value
  verification is a product requirement that would stand with the arena deleted; the arena is the
  place it happens to be measurable. This must be recorded explicitly, because the tempting and
  **inadmissible** justification is "RAGBench-finqa labels derived numbers supported"
- If ruling-9's parameter and protocol constraints are unchanged, the lane changes nothing about
  the model, the read, or the serving path

### Killgate (cheap, pre-training)

**Shortcut-presence probe on frozen weights, ~0.3 GPU-h, zero arena.** Build 2,000 triples over
held-out TabFact/InfoTabS tables that are *not* used to build the lane: (a) a claim asserting a
cell value verbatim, (b) a claim asserting a correctly derived value absent from the table,
(c) a claim asserting a wrong-operand derived value, also absent. Score all three with the frozen
R9-H105 draw-1 checkpoint under the shipped read geometry.

- **LICENSE** if mean score(a) − mean score(b) ≥ 0.20 **and** AUROC(b vs c) ≤ 0.60
- **KILL** if AUROC(b vs c) > 0.60 - the model already discriminates correct from incorrect
  derivations and the lane has nothing to teach
- **KILL** if score(a) − score(b) < 0.20 - the literal-presence penalty M1 predicts is not present
  at the operating point, and M1's training-side correlation did not transfer

Both branches are decided on legal data only, and both cost one scoring pass.

### Bar (pre-registered, blind, subset-primary per ruling 7)

- **PRIMARY** - finqa 2-draw mean ≥ **0.6933** (control 0.6333, +0.060) with the same sign on both
  H126-paired draws. Effect size transferred from the two banked lanes that moved this subset:
  H108 +0.0849 and the DR-2 pilot +0.0652; +0.060 is below the weaker of the two replicated
  effects and is ~4 sigma on a paired 2-draw mean at H126's SD ≤ 0.014
- **HOLD (mean)** - arena 2-draw mean ≥ **0.7031** (no loss vs the clean pair mean)
- **HOLD (named guardrail)** - pubmedqa ≥ 0.5463 (control − 0.06). E3 fixes pubmedqa, not
  delucionqa and not hotpotqa, as finqa's systematic cost partner (r = −0.84, replicated in five
  slicings). The H108 precedent lost 0.032 here, so 0.06 is a real but passable guard
- **HOLD (general, ruling 9)** - no subset below its control by more than 0.06, none below 0.55
- **HOLD (in-domain)** - gold_full ≥ 0.8414, RAGTruth non-EN ≥ 0.82
- **RECORD, not bar** - tatqa (co-target, E3 says it does not co-move with finqa so it cannot be
  binding); delucionqa (diagnostic per E2)
- **CONFIRMATORY DIAGNOSTIC, sign only, not a bar** - E1 measures a finqa response-verbosity
  heuristic at AUROC 0.6958, above the shipped model's 0.6489, and shows residualizing the model
  score on log mean-sentence-length costs 0.0302. Re-read the arm's finqa AUROC after the same
  residualization; if the lane's finqa gain does not survive with the same sign, the result is
  recorded as a verbosity shift, not a grounding gain, whatever the primary bar says
- **PILOT KILL** - draw 1 finqa < control + 0.020 → draw 2 unspent, lane closed

### Cost

Build CPU-only (~2 h engineering, no GPU). Killgate ~0.3 GPU-h. Training 2 seeded-paired draws at
~6 GPU-h each on a ~735k-row mix, plus ~1 GPU-h of reads. **~13 GPU-h full, ~7 GPU-h if the pilot
gate kills.**

---

## 3. L3-C2 - TABLE-SERIALIZATION-INVARIANCE LANE

**Named mechanism**: evidence surface-form coverage - the mix teaches one table serialization and
the serving distribution presents another.

### Claim

Because the training mix contains **zero** rows whose evidence is a bracket/list-of-lists
serialized table while 66.0% of tatqa's and 40.3% of finqa's score-deciding windows are exactly
that form (M2), and because the model scores bracket-decided sentences 0.174 (tatqa) and 0.023
(finqa) below prose-decided ones, re-emitting the already-admitted tabular rows in **three**
serializations - pipe (status quo), bracket/list-of-lists, and markdown pipe-with-header - at
equal proportion, with the claim held byte-identical across forms, will lift finqa to a 2-draw
mean ≥ 0.6733 (+0.040) and tatqa to ≥ 0.7720 (+0.040), both with sign agreement, while the arena
mean holds at ≥ 0.7031 and gold_full holds ≥ 0.8414.

### Mechanism detail

TabFact enters the mix through a loader that rewrites `#` separators to ` | `, which is why
100.0% of TabFact rows and 99.7% of all table-marked rows in the mix are pipe-delimited. The lane
does not replace that form - replacing it would be fitting one observed arena surface and would
be indefensible under the H119 legality ruling. It **adds** the other two forms over the same
tables, the same claims and the same labels, so the training signal is "the truth value of a
claim about a table is invariant to how the table was rendered". Concretely, each source table is
emitted three times:

- `A | B | C` pipe rows (current form)
- `[["A","B","C"], ["1","2","3"]]` bracket rows - the default output of
  `DataFrame.values.tolist()` / `json.dumps` in a Python RAG pipeline
- `| A | B | C |` / `| --- |` markdown rows - the default output of `DataFrame.to_markdown()` and
  of most LLM-authored table renderings

Claim text is byte-identical across the three copies, so any score difference the model currently
carries between forms is directly penalized. The three forms enter as one DANN group (not three) -
the adversary should not be handed serialization as a register label, since serialization
invariance is exactly what the task head must learn.

The lane subsumes the H108 corruption operators unchanged: each corrupted claim appears against
all three forms of its own table, so the lane triples the *form* coverage without changing the
*claim* distribution or the class balance.

**Dose.** The mix's tabular population is 92,585 TabFact rows plus the 61,184 H108 lane rows =
153,769. Re-serializing all of them in two extra forms would add 307,538 rows and push the mix
past 1.05M, which is a 1.5x training cost for a surface-form fix. The registered dose is a seeded
60,000-row sample of that population re-emitted in the two additional forms = **+120,000 rows**,
mix ~866k. Class balance and per-source proportions inside the sample are held at the population's
own values, so the sample introduces no reweighting of its own.

**Prior art and why this is not a re-proposal.** "Tabular serialization parity" is recorded in
the R12 register as rank 6 *below the cut*, priced against a mean-gain bar at 5x H108's mean move
and never registered. Two things changed after that pricing. Ruling 7 (adopted in R13) makes
mean-gain the wrong bar for a subset-targeted lane and prescribes subset-primary with a mean
HOLD, which is what is written above. And M2 supplies the measurement the below-the-cut note
lacked: the target form is not merely present in the arena, it *decides* two thirds of tatqa's
sentences, and the mix's share of it is exactly zero.

### Legality

- **No new corpus.** Every row is a re-rendering of tables already in the H108 lane and the
  TabFact group, already gated
- **Not arena-fitted preprocessing.** The lever is training-side form diversity, not a serving
  transform, and it does not privilege the arena's form: pipe is retained at equal weight and a
  third form the arena does not use (markdown) is added. Ruling 2 of R12 voids a transform
  "retained because it helps one arena subset"; a lane that adds three forms symmetrically cannot
  be retained on that basis, and the acceptance criterion below does not select a form
- **Independent justification.** A grounding library is handed whatever the caller's retriever
  emitted. `to_markdown()`, `values.tolist()` and pipe-joined text are the three serializations
  a Python RAG pipeline actually produces; a scorer whose truth judgement moves with the renderer
  is broken as a product regardless of any benchmark

### Killgate (cheap, pre-training)

**Serialization-invariance probe on frozen weights, ~0.2 GPU-h, zero arena.** Take 2,000
(claim, table, label) rows from the held-out TabFact split. Render each table in all three forms.
Score with the frozen R9-H105 draw-1 checkpoint.

- **LICENSE** if mean |Δscore| between pipe and bracket ≥ 0.10 **and** label-AUROC under bracket
  is at least 0.03 below label-AUROC under pipe
- **KILL** if mean |Δscore| < 0.10 - the model is already serialization-invariant and the zero
  training share is not costing anything
- **KILL** if the AUROC gap is < 0.03 - the score shifts but does not damage discrimination, so
  the lane would buy calibration the read's max/min operators discard

### Bar (pre-registered, blind, subset-primary per ruling 7)

- **PRIMARY (both required)** - finqa 2-draw mean ≥ **0.6733** (+0.040) **and** tatqa 2-draw mean
  ≥ **0.7720** (+0.040), each with sign agreement on both H126-paired draws. tatqa carries the
  stronger prediction (66.0% deciding exposure vs finqa's 40.3%) and is included as a co-primary
  precisely because E3 shows it does *not* co-move with finqa under any other lever - a joint
  move is therefore attributable to the shared surface-form mechanism and to nothing else in the
  banked covariance structure. Effect size: a purely cosmetic serving-time string wrapper (H119)
  swung finqa ±0.018 on frozen weights; a trained form-coverage fix that removes an entire
  missing surface should exceed a cosmetic wrapper by at least 2x
- **HOLD (mean)** - arena 2-draw mean ≥ **0.7031**
- **HOLD (guardrail)** - pubmedqa ≥ 0.5463 (control − 0.06)
- **HOLD (general, ruling 9)** - no subset below control − 0.06, none below 0.55
- **HOLD (in-domain)** - gold_full ≥ 0.8414, RAGTruth non-EN ≥ 0.82
- **ATTRIBUTION FALSIFIER** - if finqa and tatqa move but the three-form invariance probe re-run
  on the trained arm still shows mean |Δscore| ≥ 0.10, the gain did not come from the registered
  mechanism; record as unattributed regardless of the bar (H122 precedent)
- **PILOT KILL** - draw 1 missing either co-primary by more than 0.020 → draw 2 unspent

### Cost

Build CPU-only (~3 h engineering; the re-serializers are deterministic). Killgate ~0.2 GPU-h.
At the registered +120,000-row dose the mix is ~866k rows and a draw costs ~7.5 GPU-h.
**~16 GPU-h full, ~8.5 GPU-h if the pilot gate kills.** A 2-form variant (pipe + bracket only,
+60,000 rows) lands at ~13.5 GPU-h and is the fallback if the queue is tight.

---

## 4. L3-C3 - CONJUNCT-COMPOSITION NEGATIVE OPERATOR

**Named mechanism**: composition-level near-miss construction - an assertion whose atoms are all
present in the evidence and whose composition is not.

### Claim

Because E4 identifies one mechanism behind the largest false negatives on both hard subsets - "an
assertion whose atoms are all present and whose composition is not", carrying 30.6% of
delucionqa's discordance in a single item (resp 65, three verbatim conjuncts plus one fabricated
one, scored 0.9152) and 13.4% of finqa's (resp 200) - and because the admitted H108 engine
contains only single-token claim perturbations (`f_scale_word`, `f_digit_perturb`, `f_pct_pp`,
`f_year_shift`, `f_comparative_flip`, `f_magnitude_shift`) and no operator that touches claim
*composition*, adding ~35,000 conjunct-composition negatives built deterministically over
RAGTruth-EN, TabFact and FEVEROUS evidence will lift emanual to a 2-draw mean ≥ 0.7376 (+0.040)
and techqa to ≥ 0.7140 (+0.030), both with sign agreement, while the arena mean holds ≥ 0.7031,
finqa holds within 0.06 of control, and gold_full holds ≥ 0.8414.

### Mechanism detail

Four deterministic operators, all applied to a positive claim that already contains two or more
coordinated elements, all preserving the claim's surface fluency and its vocabulary overlap with
the premise:

- **conjunct injection** - append a fabricated conjunct to a coordinated list, built only from
  content words already present elsewhere in the premise. This is E4 resp 65 as a generator: the
  fabricated element must be lexically indistinguishable from the real ones
- **conjunct deletion under a universal** - drop one element from a list the claim then quantifies
  ("all", "both", "the following three"), so the claim over-generalizes a partially supported set
- **binding permutation** - keep every atom and permute which attribute attaches to which entity
  (row label ↔ value, condition ↔ action, year ↔ figure). This is E4 resp 200 and resp 36 as a
  generator
- **scope widening** - promote a conditional statement to an unconditional one by deleting its
  guard clause ("if the Camera Delay is on" → dropped)

Positives for the lane are the untouched source claims, so every negative shares its premise and
almost all of its tokens with a positive. The lane enters as one new DANN group.

**Why emanual and techqa are the primary targets, not delucionqa.** E3(d) measures noise share of
config-to-config variance at 11% for emanual and 13% for techqa - the two cleanest instruments in
the arena - against 85% for delucionqa. Both have never been the target of any hypothesis.
E6 shows emanual hits the procedural intensity mask on 19% of rows, so the register overlap is
real. And the one measured intervention of this shape in the bank, H117's margin term, moved
emanual +0.1561 at z = +7.4 - the largest and cleanest single-subset effect the campaign has
recorded - which prices a +0.040 bar at roughly a quarter of a measured related effect. E2's
verdict binds: delucionqa is admitted as a diagnostic surface for near-miss discrimination and
never as the bar.

**Why this is not H107.** H107 imported an unseen register (83,672 procedural pairs, two new DANN
groups, 12% of the mix) and displaced everything else - finqa 0.4261, gold_full 0.7360. C3 adds
no register: every premise is a document the mix already contains. Only the claim-side operator
set changes, which is the axis E1 identifies as the only replicated lever (S1) and the axis on
which both admitted lanes sit.

### Legality

- **No new corpus required.** RAGTruth train split is explicitly legal; TabFact and FEVEROUS are
  in the mix; all premises are already gated
- **WiCE is optional and de-conflicted.** WiCE is admitted and PASSED its gate, and its
  human sub-claim partial-support annotations are the natural supervision for conjunct deletion.
  If it is used, C3 must declare row-level disjointness from R13-H128, which is already registered
  against WiCE for hagrid; running both lanes over overlapping rows would confound their
  attribution. Default build excludes WiCE and needs no admission at all
- **No arena quantity enters the lane.** Operator set, dose and class balance are fixed from the
  H108 precedent (roughly half the lane negative), not from any arena statistic

### Killgate (cheap, pre-training)

**Composition-blindness probe on frozen weights, ~0.3 GPU-h, zero arena.** Take 2,000 legal
multi-conjunct positives from RAGTruth-EN and TabFact. Emit each one clean and once per operator.
Score with the frozen R9-H105 draw-1 checkpoint.

- **LICENSE** if AUROC(clean vs composed-negative) ≤ 0.65 **and** the mean score drop from clean
  to composed-negative is ≤ 0.15 - the model is blind to composition and the lane has signal to
  add
- **KILL** if AUROC > 0.65 - the model already catches composition corruption on legal data, and
  the arena failure is then attributable to the read or to relevance-vs-entailment (E4 shows 37.5%
  of delucionqa's discordance is relevance, structurally unreachable), not to this lane
- **PER-OPERATOR ABLATION, binding** - any operator whose individual AUROC exceeds 0.65 is dropped
  from the build before training (the H119 per-rule precedent: only rules with measured effect
  ship)

### Bar (pre-registered, blind, subset-primary per ruling 7)

- **PRIMARY (both required)** - emanual 2-draw mean ≥ **0.7376** (control 0.6976, +0.040) **and**
  techqa 2-draw mean ≥ **0.7140** (control 0.6840, +0.030), each with sign agreement on both
  H126-paired draws. At seed sigma 0.0211 / 0.0265 these are 1.9 / 1.1 sigma per draw and roughly
  2.9 / 2.1 sigma on the H126-paired pair mean
- **HOLD (mean)** - arena 2-draw mean ≥ **0.7031**
- **HOLD (displacement guard)** - finqa ≥ 0.5733 (control − 0.06). This is the explicit H107
  tripwire: the failure that lane produced was a finqa collapse, and any lane touching procedural
  or conditional claim shapes must show it did not repeat
- **HOLD (general, ruling 9)** - no subset below control − 0.06, none below 0.55
- **HOLD (in-domain)** - gold_full ≥ 0.8414, RAGTruth non-EN ≥ 0.82
- **RECORD, not bar** - delucionqa, as the diagnostic surface E2 licenses. Additionally record the
  score the arm assigns to E4 resp 65's argmin sentence; a lane that works on the registered
  mechanism must move that specific item, and a bar-passing arm that leaves it above 0.85 is
  recorded as unattributed
- **PILOT KILL** - draw 1 emanual < control + 0.015 → draw 2 unspent

### Cost

Build CPU-only (~3 h engineering). Killgate ~0.3 GPU-h. Training 2 paired draws at ~6 GPU-h on a
~721k-row mix plus ~1 GPU-h of reads. **~13 GPU-h full, ~7 GPU-h if the pilot gate kills.**

---

## 5. L3-C4 - PREMISE-LENGTH PARITY REBALANCE

**Named mechanism**: train/serve premise-geometry mismatch - the model is trained on 226-character
premises and served 1,004-character windows.

### Claim

Because the training mix's median premise is 226 characters with 28.4% of rows at or above 800,
while the arena's score-deciding window has median 1,004 characters with 59.0% at or above 800
(M3), and because 54.06% of all training rows are VitaminC at a 139-character median (0.08% of
them reaching 800), padding a pre-registered 50% sample of the mix's sub-300-character rows with
legal same-corpus distractor text - to a token target measured with the shipped tokenizer, both
classes padded at equal rate, labels and claims untouched - will lift the arena 2-draw mean to
≥ 0.7091 (+0.006) with sign agreement on both paired draws, with the gain concentrated on the
long-window subsets, while gold_full holds ≥ 0.8414 and RAGTruth EN and non-EN hold ≥ 0.82.

### Mechanism detail

The shipped read scores a sentence against a 1,500-character window; more than half of those
windows exceed 800 characters, and the argmax-over-windows operator then compares raw scores
across windows of very different lengths. The model has almost never seen that regime during
training: over half its rows present a single Wikipedia sentence as the entire premise. Two
consequences follow directly and are both visible in the banked record - the max-over-windows
operator is the operator finqa dislikes (E1: min-over-mean-over-windows reads finqa +0.0241) and
per-subset seed variance is 4x higher under the min aggregator than under whole-response reading
(E5: 0.0295 vs 0.0074). Neither is proof, and C4's killgate below is what decides it.

Construction:

- Pad only rows whose premise is under 300 characters, and only a pre-registered random 50% of
  them (seeded, recorded before the build), so the mix retains an unpadded control population of
  the same rows for post-hoc attribution
- Padding text is drawn from **other documents of the same corpus and the same DANN group**, so no
  new corpus enters and no group's register is diluted by another's
- The supporting sentence remains verbatim in the premise; the label is unchanged. This is the
  structural difference from H121, which relabelled support-free windows as label-0 and died at
  Gate B on 0.284 purity. Here there is no new label to certify - only a leakage risk in the
  other direction, gated below
- Pad to a **token** target measured with the shipped tokenizer, capped so claim + premise stays
  inside MAX_LEN 512. The ruling-7 hardware contract fixes batch 48 / MAX_LEN 512, and a
  char-target pad would silently truncate itself (E5: 22.1% of scored pairs already exceed 512)
- Padding is applied at equal rate to label-0 and label-1 rows within every group, so premise
  length carries no label information

**Token budget and why it prices the dose.** The mix currently holds 378.7M premise characters;
VitaminC contributes 15.34% of them from 54.06% of rows, TabFact 23.14% from 13.50%. Padding half
of the 392,201 sub-300-character rows to ~800 characters adds ~124M characters, a 1.33x premise
token budget, which is where the ~8 GPU-h/draw estimate comes from. The 100% dose is 1.65x and
~10 GPU-h/draw; it is the fallback if the 50% dose lands inside noise with the right sign.

**Why this is not "forced subset balance".** The closed line is H95's GroupDRO forced 1/13 group
balance - an objective-side reweighting of gradient mass toward the worst group. C4 changes no
group's row count, no group's weight and no loss term; it changes the *geometry of the premise*
inside each group. The nearest live precedent is R13-H127 (RAGTruth-parallel-copy rebalance),
which is a registered mixture lever, so the class is open.

### Legality

- **No new corpus, no new admission, no cross-corpus contamination.** Padding is same-corpus and
  same-group by construction
- **Nothing arena-derived enters the build.** The 300-character cut and the 800-character target
  are stated from the *training* distribution's own quartiles and from the read's fixed
  1,500-character window parameter, which was set pre-run in `R8-H101_windowed_read.py` and never
  tuned. M3's arena column is the motivation, not the parameter
- **Interaction with the E5 MAX_LEN line must be declared.** If a MAX_LEN 1024 read or arm is ever
  adopted, C4's token targets must be re-derived; running C4 and a MAX_LEN change in the same arm
  is a two-variable change and is not admissible

### Killgate (two cheap gates, both pre-training)

**G1 - premise-length calibration probe on frozen weights, ~0.3 GPU-h, zero arena.** Take 2,000
legal (claim, supporting-sentence) positives and 2,000 matched negatives from RAGTruth-EN and
VitaminC. Embed each supporting sentence in same-corpus padding at four premise lengths
{150, 400, 800, 1400} characters. Score with the frozen R9-H105 draw-1 checkpoint.

- **LICENSE** if label-AUROC drops by ≥ 0.05 from the 150-character to the 1400-character
  condition, **or** the positive-class mean score drops by ≥ 0.15 across the same range
- **KILL** if both are flat - the model is already length-calibrated, the train/serve geometry gap
  costs nothing, and M3 is a description rather than a defect

**G2 - pad-leakage certifier, CPU, ~1 h.** On a 2,000-row sample of prospective padded
**negatives**, check whether the injected padding accidentally supports the claim, using the
lexical certifier built for H121.

- **LICENSE** at padding-induced label-flip rate ≤ 2%
- **KILL** above 5%; between 2% and 5%, restrict padding sources to documents sharing no
  content-word 5-gram with the claim and re-measure

### Bar (pre-registered, blind, mix-wide lever so a mean-gain bar is legal per ruling 7)

- **PRIMARY** - arena 2-draw mean ≥ **0.7091** (control 0.70311, +0.006) with sign agreement on
  both H126-paired draws. Effect size matched to R12-H122's registered mean bar for a comparable
  mix-wide lever (+0.006); at arena-mean seed SD 0.0033 this is ~1.8 sigma unpaired and
  substantially more paired
- **HOLD (general, ruling 9)** - no subset below its control by more than 0.06, none below 0.55
- **HOLD (in-domain)** - gold_full ≥ 0.8414, RAGTruth EN ≥ 0.82, RAGTruth non-EN ≥ 0.82
- **ATTRIBUTION CLAUSE (binding, H122 precedent)** - the gain must be carried by the long-window
  subsets. Compute the mean delta over {techqa, delucionqa, emanual, finqa, expertqa} (deciding
  window median ≥ 1,000 chars, M3) and over {pubmedqa, hotpotqa, tatqa, covidqa, hagrid}
  (median ≤ 713). If the short-window group's mean delta exceeds the long-window group's, the
  registered mechanism is FALSIFIED and the result is recorded as unattributed even if the primary
  bar passes
- **REFUTE** - mean < 0.70311 or sign disagreement across the two paired draws
- **PILOT KILL** - draw 1 mean < 0.7005 (control − 0.0026, ~0.8 sigma) → draw 2 unspent

### Cost

Build CPU-only (~2 h engineering). Killgates ~0.3 GPU-h + ~1 h CPU. Training 2 paired draws at
~8 GPU-h each under the 1.33x premise token budget, plus ~1 GPU-h of reads. **~17 GPU-h full,
~9 GPU-h if the pilot gate kills.** The 100% dose is ~21 GPU-h and is not registered here.

---

## 6. Ordering, and what each hypothesis is worth if it fails

If the queue takes one, take **L3-C1**. Its killgate is the cheapest of the four and its
mechanism is the only one in this lens supported by a *training-side* measurement (M1) rather
than an arena observation; a KILL at the gate refutes M1's transfer for 0.3 GPU-h and is worth
having on the record either way.

**L3-C2** is second because its killgate result is informative regardless of the lane: a frozen
model that scores the same table differently depending on whether it was rendered with pipes or
brackets is a shipping defect in the library, independent of any benchmark, and the probe
measures that for 0.2 GPU-h.

**L3-C3** is third. It is the only proposal here aimed at subsets E3 identifies as the arena's
cleanest instruments and never targeted, and its per-operator ablation gate means a partial KILL
still yields a smaller, sharper lane rather than nothing.

**L3-C4** is fourth on cost, not on merit. It is the only mix-wide lever in this lens and
therefore the only one entitled to a mean-gain bar, but it is also the most expensive per draw and
the one whose mechanism rests most on inference rather than on a measured model behaviour - which
is exactly why G1 exists.

**Cross-cutting preconditions for all four**: H126 seeded-paired arm adjudication in force (none
of these bars is enforceable unpaired); the byte-identical recipe contract (batch 48, MAX_LEN 512,
GPU1); ruling-4 provenance instrument available and run if any optional corpus (WiCE in C3) is
admitted; and the arena read frozen at the PRIMARY windowed decomposed-min geometry, since three
of the four bars are subset-primary and would not survive a read change mid-round.

---

## Artifacts consulted

Evidence packs: `R14_evidence_E1_finqa.md`, `R14_evidence_E2_delucionqa.md`,
`R14_evidence_E3_covariance.md`, `R14_evidence_E4_items.md`, `R14_evidence_E5_capacity.md`,
`R14_evidence_E6_train_composition.md`.

Recomputed here in Polars: `tmp/R14_E6_mix.parquet` (685,670-row byte-exact mix reconstruction),
`experiments/grounding-semantic/R12-H121_gateA_scores.parquet` (77,171 sentence-x-window rows),
`experiments/grounding-semantic/R10-H108_pairs.parquet` (61,184 lane pairs, 4 tags),
`R9-H105_windowed_result.json`, `R9-H105_draw2_windowed_result.json`, `R9-H105_result.json`,
`R9-H105_draw2_result.json`, `R10-H108_lane_draw{1,2}_windowed_result.json`,
`R10-H108_lane_draw{1,2}_result.json`, `R10-H108_data.py`, `R10-H108_lane.py`.

Canonical log: `docs/experiments/semantic-grounding-experiments.md` - R10 lane registrations
(line 2247), R12 session rulings and pre-registration table (lines 2480-2530), R13 session
rulings 1-11 and pre-registration table (lines 2520-2560), ruling-4 provenance instrument
(line 2636).

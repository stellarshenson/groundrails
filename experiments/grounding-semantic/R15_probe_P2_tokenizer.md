# R15 PROBE 2 - number representation audit on the shipped tokenizer

**Question asked**: does the mmBERT tokenizer's treatment of numerals bound what arithmetic the cross-encoder can learn, and if so what mitigations exist that leave the serving input format untouched.

**Discipline**: CPU only, no model weights loaded, no GPU spent. Polars throughout. Every number below was measured in this session by `R15_P2_tokenizer_audit.py` and banked in `R15_P2_tokenizer_audit.json`. Arena-derived quantities are labelled ANALYSIS ONLY and are used to motivate constructions, never to calibrate a bar.

---

## 0. Verdict, first

**The tokenizer is not the defect.** The shipped tokenizer is digit-atomic by vocabulary construction: exactly ten pure-digit tokens exist (`0`-`9`), zero multi-digit numeral tokens exist anywhere in the 256,000-entry vocabulary, and zero of the 580,604 BPE merges join a digit to a digit. Every numeral therefore fragments into one token per character, deterministically, with no irregular splits and no vocabulary-frequency dependence. This is the favourable regime described in the numeracy literature, not the pathological one. **The 0.4924 AUROC on correct-versus-wrong derivations is not a representation ceiling; it is what the audit predicts a model would score if it had never been trained on the distinction** - which is exactly the R14 diagnosis.

**Three real defects were found, and none of them is the tokenizer.** All three sit in the *construction* of the R14-A4 derivation-parity lane, all three are CPU-time fixes with zero GPU cost, and one of them is a build-time correctness bug rather than a tuning choice:

1. **Evidence truncation inside the lane** - a 1,500-character TabFact window costs **438.99 tokens** on average at **2.2605 chars/token**, and **34.93%** of such windows exceed `MAX_LEN` 512. Built as specified, roughly a third of the lane's positives would be trained with the evidence supporting the derivation truncated away.
2. **Surface-register degeneracy** - TabFact tables contain **zero** thousands-separated numerals in 2,809,937 characters, and the A4 constructor's `fmt()` emits zero as well. The lane cannot teach the equivalence of `1234` and `1,234`, and at the token level those two surfaces diverge after the first digit group (mean shared prefix **2.0** tokens of 5 and 6 at four digits).
3. **Unconstrained negatives** - the H133 triples happen to be 88.95% token-length-matched, giving AUROC(token length alone) = **0.494**, but nothing in the construction enforces it and nothing controls how token-similar the wrong value is to the right one.

**Recommended disposition**: adopt amendments P2-A through P2-D against the already-licensed R14-A4 (data-only form, no new hypothesis, no new GPU), adopt P2-E against R14-A5 (a feature-definition correction), and record P2-F below the cut. **Do not propose any tokenizer change, any input transform, or any embedding change** - section 9 gives the reasoning and section 8 engages the H119 verdict directly.

---

## 1. What was measured

`R15_P2_tokenizer_audit.py` loads `models/R9-H105-mmbert-dann-clean/tokenizer.json` - the tokenizer the shipped cross-encoder was trained with and is served with - and reports eleven blocks: form audit, digit ladder, corpus census, adjacency, magnitude boundaries, separator alignment, place-value misalignment inside table columns, scaled forms, the constructed lane's own value surface, the lane label-length confound, and the token cost of a table evidence window.

Corpora touched, all already on disk:

- **TabFact** held-out (test + validation, train-id-disjoint) - the exact split `R14_H133_probe.build()` draws from; 3,391 tables. Also the train split (4,000 tables), which is where an A4 lane would actually be built
- **`R14_H133_triples.parquet`** - the 2,000 banked triples and their `v_correct` / `v_wrong` asserted values
- **`R10-H108_pairs.parquet`** - the admitted lane's claims and evidence chunks
- **RAGBench-finqa** train + validation + test - ANALYSIS ONLY, used for the register split in section 6 and nowhere else

FEVEROUS could not be re-read (it resolves over `hf://` and this session has no network). TabFact carries 1,121,632 of the census's 2,009,526 constructible tuples and is the corpus the H133 probe itself used, so the tabular findings transfer; the FEVEROUS-specific surface should be re-checked at build time.

---

## 2. The tokenizer is digit-atomic by construction

Declaration read straight out of `tokenizer.json`: BPE model, a `Replace` normalizer mapping space → `▁`, a `Metaspace` pre-tokenizer with `prepend_scheme: always` and `split: true`, vocabulary size 256,000 (the Gemma-2 vocabulary mmBERT inherits).

| property | measured |
|---|---|
| pure-digit tokens in vocabulary | **10** (`0`-`9`) |
| multi-digit numeral tokens in vocabulary | **0** |
| total BPE merges | 580,604 |
| merges joining a digit to a digit | **0** |
| single-token integers in 0-999 | **0** |

Worked fragmentations, verbatim from the form audit:

```
7             2 tok   ['▁','7']
42            3 tok   ['▁','4','2']
10547         6 tok   ['▁','1','0','5','4','7']
10,547        7 tok   ['▁','1','0',',','5','4','7']
10.5          5 tok   ['▁','1','0','.','5']
$10,547       7 tok   ['▁$','1','0',',','5','4','7']
$ 383,221     9 tok   ['▁$','▁','3','8','3',',','2','2','1']
12.5%         6 tok   ['▁','1','2','.','5','%']
(1,234)       7 tok   ['▁(','1',',','2','3','4',')']
10.5 million  6 tok   ['▁','1','0','.','5','▁million']
2019          5 tok   ['▁','2','0','1','9']
FY2019        5 tok   ['▁FY','2','0','1','9']
```

The rule is exact and holds without exception across the ladder: a bare integer costs **digits + 1** tokens (the leading metaspace token is unconditional), and a thousands-separated integer costs **digits + 1 + floor((digits-1)/3)**. Measured over 200 random values at each of 12 digit lengths, every single length produced a single token count - the `bare_unique_counts` field is a singleton set at every rung. There is no irregularity to exploit and none to suffer from.

**This is the good case.** The failure mode the numeracy literature documents - `1234` merging as `12`+`34` while `1235` merges as `123`+`5`, so that arithmetically adjacent values receive unrelated representations - is structurally impossible here. There are no multi-digit tokens to merge into.

---

## 3. Token cost of numerals in the actual corpora

Occurrence-weighted token counts, strict numeral regex (a thousands separator counts only when followed by exactly three digits, so `December 31, 2019` does not read as separator-bearing):

| corpus | numeral occurrences | mean tokens | median | p99 | max | share ≥ 3 tok | share ≥ 5 tok |
|---|---|---|---|---|---|---|---|
| TabFact held-out tables | 234,373 | 3.2767 | 3 | 7 | 16 | 62.51% | 23.27% |
| TabFact train tables | 267,255 | 3.3222 | 3 | 7 | 23 | 63.75% | 24.30% |
| H108 lane claims | 53,812 | 4.0419 | 4 | 8 | 17 | 84.23% | 47.43% |
| H133 `v_correct` (derived) | 2,000 | 5.3255 | 6 | 9 | 13 | 99.70% | 78.55% |
| H133 `v_wrong` (derived) | 2,000 | 5.3230 | 6 | 9 | 13 | 99.60% | 77.90% |
| finqa documents (ANALYSIS ONLY) | 23,418 | 4.6542 | 5 | 8 | 12 | 92.50% | 63.09% |
| finqa responses (ANALYSIS ONLY) | 24,629 | 5.0994 | 5 | 11 | 19 | 95.43% | 69.96% |

No numeral anywhere costs one token - the metaspace prefix guarantees a floor of two.

The load-bearing consequence is for the evidence window, not the claim. A 1,500-character TabFact window encodes to **438.99 tokens** on average, **2.2605 chars/token**, p95 **820.2 tokens**, and **34.93%** of windows exceed 512. **29.27%** of all tokens in such a window are bare digits. This is the mechanism behind E5's chars/token table (tatqa 2.74, techqa 3.23, finqa 4.01 against ~5 for prose): dense numeric text is token-expensive precisely because digits are atomic. R14-A2 already prices the arena-side consequence; section 7 below prices the consequence *inside the A4 lane*, which A2 does not cover.

---

## 4. Numerically adjacent values are token-comparable

Across 2,000 random 4-to-8-digit values scored against `v+1`:

| surface | mean shared token prefix | pairs with zero shared prefix | pairs with equal token length | mean tokens |
|---|---|---|---|---|
| bare | **0.8322** of the sequence | 0.00% | **100%** | 6.974 |
| thousands-separated | **0.8588** | 0.00% | **100%** | 8.3705 |

By digit count (bare): 4 digits 0.7732, 5 digits 0.8128, 6 digits 0.8399, 7 digits 0.8635, 8 digits 0.8750 - the fraction rises with length because the divergence stays a single trailing token.

Worked cases: `10547` vs `10548` share 5 of 6 tokens; `10,547` vs `10,548` share 6 of 7; `$ 383,221` vs `$ 383,222` share 8 of 9; `12.50` vs `12.51` share 5 of 6.

**Conclusion**: two values differing by one unit are presented to the encoder as sequences agreeing on ~85% of their tokens and differing in exactly one position. Discriminating them is a *hard* problem in the sense that the signal is narrow, but it is a *well-posed* one - the differing token is in a fixed, aligned position and the rest of the context is identical. Nothing about the representation makes it incomparable.

The two exceptions are both magnitude crossings, not adjacency: `999` vs `1000` share only the leading `▁` (1 of 5), and `99.9` vs `100.0` share 1 of 6. These are rare and are exactly where the next section's finding bites.

---

## 5. Magnitude is carried by sequence length alone

Digit-atomic tokens carry no place value. The token `5` is the same token whether it is a unit or a hundred-thousand; place value is position-from-the-end, while the encoder indexes position-from-the-start. Two consequences, both measured:

**A value and its ten-fold are token-prefix-nested.** For every case tested, `tokens(v)` is a strict prefix of `tokens(10v)`:

```
9      -> 90       shared prefix 2 of 2 / 3
99     -> 990      shared prefix 3 of 3 / 4
547    -> 5470     shared prefix 4 of 4 / 5
1054   -> 10540    shared prefix 5 of 5 / 6
10547  -> 105470   shared prefix 6 of 6 / 7
```

A scale error - the single commonest failure in financial derivation, where a table is stated "in millions" and the answer is written out in units - is therefore the *least* separable error type in this representation. Every token of the correct answer appears, in order, at the right index, inside the wrong answer. Only the trailing zero and the total length distinguish them.

**Within a table column, place alignment often fails.** Over 4,393 numeric columns of the held-out TabFact tables, 501,528 within-column value pairs: **81.10%** share a digit count and are index-wise place-aligned; **18.90%** do not. For nearly a fifth of within-column comparisons, the model cannot align hundreds against hundreds by relative token offset; it must first infer both lengths.

**This is the real representational cost, and it is a training-signal problem, not a vocabulary problem.** Length-sensitive place alignment is learnable - it is what any digit-atomic model that does arithmetic has learned - but it must be *taught*, and the current mix teaches it nowhere.

---

## 6. The separator register split, and where the surfaces diverge

Adding a thousands separator breaks token alignment early. Measured over 1,500 random values, mean shared prefix between the bare and separated surfaces of the *same* number:

| digits | bare tokens | separated tokens | mean shared prefix |
|---|---|---|---|
| 4 | 5 | 6 | **2.0** |
| 5 | 6 | 7 | 3.0 |
| 6 | 7 | 8 | 4.0 |
| 7 | 8 | 10 | **2.0** |
| 8 | 9 | 11 | 3.0 |
| 9 | 10 | 12 | 4.0 |

The shared prefix is always the leading `▁` plus the digits before the first comma - it resets at every magnitude class. `1,234,567` and `1234567` agree on 2 of 10 and 8 tokens respectively. Two spellings of the same quantity are, past the first group, near-disjoint sequences.

That would be a curiosity if both surfaces appeared on both sides. They do not.

**In-domain (admissible, may calibrate a bar)**:

| text | characters | separator hits |
|---|---|---|
| TabFact tables (evidence the A4 lane is built over) | 2,809,937 | **0** |
| H108 lane claims | 3,625,494 | 1,282 |
| H108 lane evidence chunks | 24,667,330 | 18,718 |

**Arena finqa (ANALYSIS ONLY - motivates a construction, never calibrates a bar)**: across all 16,562 rows and all three splits, evidence documents contain **zero** thousands-separated numerals in **67,333,887** characters, while responses contain **24,484** of them in **7,356,742** characters. The register split is total. Surface-share breakdown on the test split: documents are 81.28% bare integer, 18.72% decimal, 18.55% currency-prefixed, 3.56% percent-suffixed, 0% separator-bearing; responses are 64.17% bare integer, 31.19% currency-prefixed, 24.76% decimal, **11.07% separator-bearing**, 10.09% percent-suffixed.

This is the same asymmetry the R12-H119 audit found from the other end (its currency-spacing rule scored +13.01 points of affix-inclusive agreement because finqa evidence spaces `$` and claims do not), measured here on the separator axis and quantified end to end. Section 8 explains why the H119 refutation does not close the line this opens.

---

## 7. Scaled and scientific forms are effectively disjoint

Token-id Jaccard and shared prefix between equivalent spellings:

| a | b | tokens a | tokens b | Jaccard | shared prefix |
|---|---|---|---|---|---|
| `10.5 million` | `10,500,000` | 6 | 11 | 0.5714 | 3 |
| `10.5 million` | `10500000` | 6 | 9 | 0.6667 | 3 |
| `10,500,000` | `10500000` | 11 | 9 | 0.8000 | 3 |
| `1.05 billion` | `1,050,000,000` | 6 | 14 | 0.5714 | 2 |
| `in millions 10.5` | `10,500,000` | 7 | 11 | 0.5000 | **0** |
| `12.5%` | `0.125` | 6 | 6 | 0.7143 | 1 |
| `(1,234)` | `-1,234` | 7 | 6 | 0.6250 | **0** |

The Jaccard values are inflated by the shared digit alphabet - `0` and `5` recur - so they overstate similarity; the shared prefix is the honest measure, and it is 0 to 3 tokens in every case. The scaled-header case (`in millions 10.5` against the written-out value) shares nothing at all, and the accounting-negative convention `(1,234)` shares nothing with `-1,234`.

**Every scale, unit and sign convention in financial text is a token-level identity the model must learn as a fact, not one it can read off the surface.** None of these equivalences appears in the clean mix's supervision.

---

## 8. The constructed lane's own value surface (R14-A4 / H133)

The A4 constructor, as run in `R14_H133_probe.py`, formats every derived value through:

```python
def fmt(v):
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.2f}"
```

Measured on the 2,000 banked triples: **0.0%** of `v_correct` values carry a thousands separator, 9.35% carry a decimal point, mean 5.3255 tokens. The evidence side matches (TabFact carries no separators at all), so the lane as specified is internally consistent - and, exactly for that reason, **surface-degenerate**. Fifty thousand pairs of it would teach derivation in one register only, the register finqa's claims least often use.

Two further build-side facts:

**The label-length confound is currently absent by luck, not by design.** `v_correct` and `v_wrong` have equal token length on **88.95%** of triples (correct longer 5.35%, correct shorter 5.70%, mean signed difference +0.0025 tokens), and AUROC using token length alone is **0.494** - chance. That is a clean result for the H133 probe: its 0.4924 reading is not a length artefact. But nothing in the construction enforces it, and the campaign has now twice been bitten by a construction-time regularity becoming the learned rule (P(0 | absent) = 0.610 in the clean mix, 0.946 in the H108 lane).

**A third of the lane's evidence would be truncated.** At 2.2605 chars/token, the lane's 1,500-character evidence window costs 438.99 tokens on average and **34.93%** of windows exceed the 512-token `MAX_LEN`. `longest_first` truncation removes the tail of the longer sequence - the table body. A positive whose two operand rows fall past the cut is an unlearnable row: the claim asserts a derived value and the evidence that would justify it is not in the encoder's input. This is not a tuning question; it is a build-time correctness bug, and it is measurable and fixable at build time for zero GPU.

---

## 9. Digit-level learnability, from model knowledge

What is established about subword tokenization and numeracy, and how it applies here:

- **The documented harm is irregular merging, not fragmentation.** The classic negative results (BPE vocabularies where `1234` and `1235` receive structurally unrelated pieces, and where a number's segmentation depends on its pretraining frequency) describe GPT-2-era vocabularies. The demonstrated repair, adopted across PaLM, Gemma, Llama-3 and Mistral, is exactly what this tokenizer already does: forbid multi-digit tokens so segmentation is content-independent and positionally regular. Work on transformer arithmetic consistently reports large gains from moving to character- or digit-level number surfaces. **We are already on the repaired side of that finding.**
- **The residual known weakness of left-to-right digit-atomic schemes is place-value alignment**, which section 5 measures directly. The two published mitigation families are (a) right-to-left digit grouping, so that a fixed offset from the sequence start corresponds to a fixed place, and (b) explicit digit-position embeddings (the Abacus line), which give strong length generalization on arithmetic. **Both require changing the tokenizer or the embedding table and re-pretraining.** Neither is compatible with a single sub-400M model shipped on the current pretrained checkpoint inside this campaign, and both are unnecessary if the actual bottleneck is supervision.
- **Our task is verification, not generation.** The model does not need to *produce* a sum; it needs to decide whether an asserted value is the value the evidence implies. That is a comparison problem over two token sequences that, per section 4, agree on ~85% of their positions when the answer is nearly right. The capability required is nearer to "detect a single-position mismatch in an aligned numeric sequence, conditioned on an operation named in the claim" than to "do arithmetic in the forward pass". Encoders at this scale demonstrably learn tasks of that shape when the training distribution contains them.
- **Therefore the mitigations worth taking are the ones that change what the training distribution contains, not what the input looks like.** Concretely: value canonicalization *inside constructed lanes only*; digit-count-matched negatives; value ranges chosen so that a pair does not straddle a magnitude boundary unless straddling is the thing being taught; and explicit scale-error negatives, which section 5 identifies as the hardest separable case.

### Engaging the H119 verdict

**R12-H119 is REFUTED and stays refuted. Nothing here re-proposes it.** For the record, its measured content: read-time numeric canonicalization applied symmetrically to claim and evidence before tokenization, on frozen weights, over four checkpoints. Strip direction returned pair means +0.00245 / -0.00098 against a +0.003-on-both bar, finqa ≥ +0.010 on 2 of 4 draws against a bar of 3 of 4, and finqa sign-disagreed *within* the H105 pair (-0.0163 on draw 1, +0.0178 on draw 2). Add direction returned -0.00039 / -0.00002. The mechanism finding was that the transform is confirmed localized (every non-numeric subset moved < 0.002) but **not directional** - tatqa swung +0.0448 / +0.0012 / -0.0142 / -0.0227 across four draws under a deterministic zero-variance read. The main-session adjudication closed the serving-wrapper canonicalization line and, on the instability argument, put tabular serialization parity below the cut.

**Why the P2 amendments are not that lever, in four points:**

1. **Different object.** H119 held the weights fixed and changed the input. Its refutation is a statement about one trained function's sensitivity to a surface edit at read time. P2-A holds the read fixed - byte-identical, `src/groundrails/semantic_ov.py:36` untouched, no transform shipped anywhere - and changes which examples exist during training. There is no serving-side transform to be idiosyncratic about.
2. **H119's own mechanism finding is the argument for a training-side repair.** A checkpoint whose response to a separator is arbitrary in sign across draws is a checkpoint that never learned the separator is semantically inert. That is a diagnosis of the training distribution, and it is corroborated here from the data side: the tabular corpus the lane is built over contains zero separators in 2.81M characters, so the equivalence is never demonstrated to the model even once.
3. **No directional claim is made.** H119 had to assert that a specific transform helps in a specific direction, and failed on that. P2-A asserts only that the lane must not be surface-degenerate, and offers the degeneracy as a measurement (0% separators, 0% currency, 0% scaled forms in the constructed values) rather than a prediction.
4. **The legality question does not arise.** H119's grant was conditional on the transform shipping in the library serving path identically for every corpus and every future input, precisely because a transform retained for one arena subset is arena-fitted preprocessing. A lane-internal surface choice ships nothing and is applied to no input at serving time, so it cannot be arena-fitted preprocessing. The arena statistic in section 6 is recorded as ANALYSIS ONLY and the in-domain TabFact/H108 measurement is what carries the amendment.

**Binding constraint accepted**: no P2 amendment may alter the serving read, `MAX_LEN`, the windowing, or any pre-tokenization text handling. Anything that would is out of scope and is not proposed.

---

## 10. Amendments

All five are amendments to blocks already registered in `R14_synthesis.md` section 1. None is a new hypothesis, none requests new GPU time, and none changes any pre-registered bar.

### P2-A - AMENDS R14-A4: surface-register parity inside the lane (BINDING, data-only)

**Defect** - the lane's constructed values are 0% separator-bearing, 0% currency-prefixed and 0% scaled-form, because `fmt()` cannot emit them and TabFact never shows them (0 separator hits in 2,809,937 characters). At the token level the registers diverge after the first digit group (mean shared prefix 2.0 tokens at four digits). The deployed claim register uses all of them.

**Amendment** - for each constructed tuple, draw the asserted value's surface from a fixed, pre-registered distribution over {bare, thousands-separated, currency-prefixed, decimal-padded, scaled-word} rather than always bare, and additionally emit, for a fixed 10% of positives, a **register-mismatched numerically identical** row: evidence writes the operands in the table's own register, the claim writes the correct result in a different one, label 1. The mixture weights are frozen in writing before build and are set from the in-domain H108/TabFact measurement, never from the finqa statistic in section 6.

**Cost** - zero GPU, ~1 hour of constructor work.

**Optional pre-registration reading (~0.1 GPU-h, frozen weights, in-domain, gate card)** - score matched claim pairs differing only in register over identical evidence and report mean |Δscore| on the shipped H105 draw 1. Recorded as a diagnostic baseline for the post-training re-read; **not** a gate, because a directional prediction here is exactly what H119 refuted.

### P2-B - AMENDS R14-A4: token-length parity between positive and negative values (BINDING, data-only)

**Defect** - nothing constrains the digit count of the wrong-operand value. It currently matches on 88.95% of triples by luck, giving AUROC(length alone) = 0.494; a differently seeded build could hand the model a length shortcut, which is the same failure class as P(0 | absent) = 0.610.

**Amendment** - require the wrong-operand value to carry the same digit count as the correct value; where the enumeration cannot satisfy it, drop the tuple. Report the realised AUROC-from-token-length-alone in the build manifest and **KILL the build if it exceeds 0.55**. Free, CPU, and it closes the shortcut before it opens rather than after.

### P2-C - AMENDS R14-A4: stratified token-prefix hardness for negatives (BINDING, data-only)

**Defect** - the audit shows the negative's difficulty is entirely determined by how much token prefix it shares with the correct value, and the current construction does not control it. Adjacent values share 83.22% of prefix; a ten-fold scale error is a strict prefix extension (section 5); an unrelated operand pair may share almost nothing.

**Amendment** - stratify negatives into three pre-registered bands by shared-token-prefix fraction with the correct value - low (< 0.4), mid (0.4-0.8), high (≥ 0.8) - at fixed proportions, and include the scale-error case (×10, ÷10, and the "in millions" header case) as a named sub-band of the high band. Report held-out accuracy per band. Without this the lane can admit on easy negatives while leaving untouched the near-miss regime where finqa's actual errors sit (E4: a cross-line-item ratio certified at 0.7493, 13.4% of all finqa discordance).

### P2-D - AMENDS R14-A4 × R14-A2: budget the lane's evidence in tokens, not characters (BINDING, build-time correctness)

**Defect** - measured: 438.99 tokens mean, 2.2605 chars/token, p95 820.2, **34.93% of 1,500-character TabFact windows exceed `MAX_LEN` 512**. `longest_first` truncation drops the table body. Roughly a third of the lane's positives would assert a derived value whose supporting rows are not in the encoder's input.

**Amendment** - at build time, truncate the lane's evidence to a **token** budget under the read's own `MAX_LEN`, and assert per row that both operand rows survive inside the retained prefix; drop any tuple that fails. Report the realised share of dropped tuples. Note the interaction explicitly in the A4 registration: **if R14-A2 Stage 1 adopts the 1024 read, this budget is re-derived at 1024 and the drop rate falls; if A2 kills, the 512 budget is binding.** A4 must not be built before A2 Stage 1 reports, or it must be built at 512 and re-checked.

**This is the highest-value item in P2.** It costs nothing and, unfixed, it silently dilutes the lane by about a third with rows that cannot teach the target relation.

### P2-E - AMENDS R14-A5 (H134): define the nuisance feature on tokens, not characters

**Defect** - A5 decorrelates the task logit against *claim digit fraction*, defined on characters. The trunk sees tokens. Because a numeral costs digits + 1 tokens and separators add more, character digit fraction and token digit fraction are related but not identical - and the quantity the trunk can actually condition on is the token one. Measured: 29.27% of the tokens in a TabFact evidence window are bare digits.

**Amendment** - define the decorrelation feature as the **share of claim tokens that are bare-digit tokens**, computed from the same encoding the model consumes. Report both definitions in the gate so the substitution is auditable. This does not change A5's mechanism, its `lambda_dec = 1.0`, its co-primary bars, or the recorded sign tension between the in-domain partial r (+0.073) and the arena-finqa function (negative); it makes the regressed variable the one the network can see.

### P2-F - RECORDED BELOW THE CUT, not proposed: representation-level numeracy interventions

Right-to-left digit grouping and digit-position (Abacus-style) embeddings are the two published repairs for the place-value misalignment measured in section 5. Both are rejected for this campaign: each requires re-pretraining and a serving-code change, each breaks the "one model < 400M at serving, on the current checkpoint" deliverable, and the audit shows the current scheme is already in the favourable regime the literature recommends - so the expected return is bounded by the residual alignment issue alone, which P2-C's scale-error sub-band tests for a fraction of the cost. Recorded so a future round does not re-derive it from scratch.

---

## 11. What P2 does not claim

- **It does not claim the lane will work.** It claims three specific construction defects would prevent a fair test of whether it works.
- **It does not re-open H119.** No transform is applied to any input at serving time; see section 9.
- **It does not set or move any bar.** A4's finqa bar (+0.060, 2-draw mean ≥ 0.6933), its anti-gaming clause, its log-length residualization clause and its holds are unchanged. P2-B adds one build-time kill (AUROC-from-length > 0.55) that operates on the constructed data before any GPU is spent.
- **The section 6 finqa register statistic is ANALYSIS ONLY.** The amendment it motivates is carried by the in-domain TabFact/H108 measurement.
- **FEVEROUS surfaces are unverified** (no network this session). The build must re-run the section 6 separator count over FEVEROUS before freezing P2-A's mixture weights.

---

## 12. Reproduction

```
cd /home/lab/workspace/private/ai-assistants/groundrails
uv run python experiments/grounding-semantic/R15_P2_tokenizer_audit.py
```

CPU only, ~3 minutes, no GPU, no network. Writes `R15_P2_tokenizer_audit.json` with all eleven blocks. Inputs: `models/R9-H105-mmbert-dann-clean/tokenizer.json`, `data/external/datasets/dataset-tabfact.zip`, `data/external/datasets/dataset-ragbench.zip`, `R14_H133_triples.parquet`, `R10-H108_pairs.parquet`.

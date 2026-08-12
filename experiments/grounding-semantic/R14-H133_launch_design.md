# R14-H133 derivation lane - build record (v3, Phase 1, CPU)

The lane for R14-A4 (H133), amended by R15-B1 and carrying R15-B4 (H138) as a 15% relational
sub-block, is rebuilt under the author's ruling. No GPU time was spent and no training was
launched.

Version history on disk: `R14-H133_lane.v1-DEFECTIVE.parquet` (two CRITICALs, withdrawn),
`R14-H133_lane.v2-SUPERSEDED.parquet` (leak-clean, superseded by the ruling), and
`R14-H133_lane.parquet` (v3, canonical).

## The author's ruling, verbatim

> "grounding models are not reasoning models - exclude this from the data; but for the same
> claims we can attach a reasoning trace (produced with programmatic tool calling or just
> explained calculation) that has this number, and add this to the corpus."

A 307M encoder cannot compute arithmetic - the P4 probes read derivations at chance - so a bare
derived-value claim asks for reasoning, not grounding, and is excluded. Every derived claim in
v3 instead carries a deterministic reasoning trace emitted from the stored operands:

> The score of somerset is 801 and the score of essex is 686; computing 801 / 686 gives 1.17,
> so the ratio of the score of somerset to that of essex is 1.17.

The grounding task is now **matching**: the trace's operands must ground against the evidence
cells, and the claim's number must match the trace's conclusion. No arithmetic competence is
required to verify a row.

### Serving-shape corollary

In production, a claim carrying a derived number gets its trace attached by **programmatic tool
calling before grounding**. The detector verifies the trace - operand bindings against the
retrieved evidence, conclusion against the calculation - and never computes anything itself.
A deployment that feeds bare derived claims to the detector is outside what this lane trains,
and the lane should not be read as evidence about that case.

### A4 amendment (ii) justification, carried verbatim

> A grounding library that flags every arithmetically derivable quantity as unsupported
> false-alarms on the commonest shape of numeric RAG answer - this is a product requirement
> that stands with the arena deleted.

## Negative families - every defect is groundable

| family | negatives | share | how it is detected |
|---|---:|---:|---|
| **(a) trace-evidence mismatch** | 15,064 | **0.709** | the trace cites a cell value the table does not hold |
| - `a:misbound_row` | 10,602 | 0.499 | value taken from a different row of the same column |
| - `a:misbound_col` | 4,462 | 0.210 | value taken from a different column of the same row |
| **(b) operation-word mismatch** | 4,485 | **0.211** | trace performs one operation, the conclusion names another |
| **(c) conclusion mismatch** | 1,701 | **0.080** | the conclusion contradicts the trace's own result |

Family (a)'s arithmetic is internally consistent - the negative recomputes correctly from the
value it (falsely) cites, so the only tell is against the evidence. Family (c) is capped under
its 10% ceiling.

## Verify table - all bars

| check | bar | v3 result |
|---|---|---|
| **claim+trace-only AUROC**, doc-disjoint 80/20 | abort > 0.55 | **0.5236** |
| **per-family within-pair, family (a)** | abort > 0.60 | **0.5082 / 0.5115** |
| **per-family within-pair, family (b)** | abort > 0.60 | **0.6116 - BREACH, see below** |
| per-family within-pair, family (c) | exempt by design, capped | 0.5203, share **0.0800** |
| per-family within-pair, sub-block | abort > 0.60 | compare 0.4706, bind_row 0.4662, bind_col 0.4543 |
| **trace re-derivation, every core row** | **0 errors** | **0** of 42,500 (0 unparsable, 0 arithmetic wrong, 0 conclusion mismatches) |
| **positive trace operands verbatim-groundable** | **1.0** | **1.0** (0 missing from chunk, 0 unquoted) |
| **(a)-negative mismatch mechanically confirmable** | **1.0** | **1.0** of 15,064 |
| trailing-zero AUROC per derivation type | inside [0.45, 0.55] | 0.5000 on all eight |
| pair identity with numerals masked | 0 breaches | 0 |
| digit-length parity, derivation core | 0 breaches | 0 (`compare` 15, exempt) |
| AUROC from claim token length alone | abort > 0.55 | 0.5001 |
| **rows over the 512-token budget** | **< 0.10** | **0.0772** |
| max share of any single result digit length | < 0.35 | 0.3225 |
| P(label 0 \| absent), derivation core | 0.5000 | 0.50000 (100% of core rows absent) |
| row-level disjointness from the H108 lane | 0 shared | 0 / 0 / 0 |

### The one breach - family (b), and why it needs a ruling

`b:operation_word` reads **0.6116** against a 0.60 bar. Five independent interventions moved it
from 0.659 and it has not gone below 0.60:

1. wrong word drawn uniformly - 0.659
2. counter updates deferred until a pair is accepted (they were being spent on rejected
   candidates, skewing `ratio` to 0.333 of negatives against 0.161 of positives) - 0.603
3. balancer seeded with each word's final target instead of a running count - 0.621
4. `rounding` denied its own word with a wrong place, because the trace already states the
   place and a positive then carried that place word twice against a negative's once - a
   term-frequency tell - 0.623
5. single-arity (b) quotas equalised so the two-member pool can balance at all - 0.612

The marginals are now as flat as they can be made: conclusion-word skew **0.009**, and the
trace-side operator marginals are **identical by construction** (the trace is byte-identical
within a (b) pair). What remains is the *agreement* between the trace's operator and the
conclusion's word - which is the defect itself.

**This is the same property that earned family (c) its exemption.** Both are text-internal
inconsistencies, detectable "by reading, no computation" - the coordinator's own words for (b).
A bag-of-ngrams recovers ~0.61 of it; a real reader would recover all of it. The bar as written
may not be satisfiable for (b) by any construction that keeps the family's meaning.

**Branches for the ruling:**
- **If (b) is exempted like (c)** - the lane ships as built; recommend capping (b) the way (c)
  is capped, at or below its current 21.1%
- **If the 0.60 bar stands for (b)** - the family must be cut and its 4,485 negatives reflowed
  to family (a); the reflow machinery already exists and the rebuild is one CPU cycle
- **If (b) is kept at its current share and the bar is waived without a cap** - record that
  21.1% of the lane's negatives are evidence-free detectable, which weakens the claim that the
  lane teaches grounding rather than text consistency

## Row counts and composition

50,000 rows / 25,000 pairs; labels 25,000 / 25,000. Core 42,500 (85.00%, tag `quant_deriv`);
relational sub-block 7,500 (15.00%, tag `quant_relational`). Sources: TabFact train 33,018
rows, FEVEROUS train 16,982.

Type schedule, unchanged from v1 and exactly on R15-B1's registered shares: difference 3,434 /
ratio 3,434 / pct_change 3,434 / sum 3,436 (0.1616 each), rounding 2,576 (0.1212), mean 2,146
(0.1010), scale_unit 1,717 (0.0808), product 1,073 (0.0505). Zero count-aggregation, date
arithmetic, depth-2 chains.

Sub-block: `bind_col` 1,500, `compare` 1,500, `bind_row` 750 (the registered 20%
non-regression arm); compare gaps 501 / 501 / 498 across under-10% / 10-100% / over-100%.
Every sub-block claim carries its binding statement as its trace; `compare` negatives swap the
two bindings and then reason correctly from the swap, so the defect is groundable rather than
arithmetic.

Serialization row shares: `row_prose` 0.302, `narrative` 0.296, `pipe` 0.149, `keyvalue` 0.102,
`markdown` 0.100, `json_records` 0.051, against targets 30/30/15/10/10/5.

**Table diversity**: 10,914 documents, 2.29 pairs and 4.58 rows each - 773 at 1 pair, 7,531 at
2, 1,518 at 3, 849 at 4, 243 at 5. By source: TabFact 7,290, FEVEROUS 3,624.

## Token-budget impact

Traces lengthen every core row. Median pair 365 tokens (v2: 317), p95 531 (v2: 530), share over
512 **0.0772** against v2's 0.0669 and a 0.10 bar.

The budget is held by trimming **evidence, never traces**, as the ruling requires. `_fit_evidence`
drops body rows one at a time - operand rows and the misbind source row are protected - until a
per-serialization-form character-per-token estimate puts the pair under 500. Dropping a row can
only remove numerals, so the absence rule survives trimming, and the protected rows keep every
trace groundable.

## What could not be preserved from v2

1. **Bare-assertion form is gone.** R15-B1 clause 2 requires at least 50% bare-assertion claims;
   the ruling excludes bare derived claims outright. Every core claim now carries a trace. The
   clause is superseded, not breached silently.
2. **`wrong_scale` operand corruption is dropped.** A x10 operand cannot carry the true
   operand's trailing-zero count or decimal placement, so every surviving instance was separable
   on the claim text alone (measured within-pair 0.651 before removal). Its share went to the
   two misbinding kinds, which is why `a:misbound_row` sits at 0.499 rather than the 0.70x0.60
   the family split implies.
3. **The v2 negative families are gone** - N1 bands, N2 operator swap, N7 numeral corruption
   and the two rounding families do not exist in v3. Their leak fixes carry forward as the
   surface-parity discipline (leading digit, digit count, trailing zeros, decimal presence,
   sign, and the per-(family, type) direction balancer), now applied to **both** the claim's
   conclusion and the trace's cited operands.
4. **Operands must render exactly.** A cell of 2.037 rendered "2.04" made the written arithmetic
   wrong on its face ("2.04 * 1000 = 2037"); 268 such traces existed before the fix. Operands
   whose two-decimal rendering does not round-trip are now rejected, which costs supply on
   columns with three or more decimals.
5. **Row labels carrying a semicolon, a sentence break, or over 60 characters are rejected** -
   they made traces unreadable ("The Year of GSIST opening; Admission of 54 students. is 1995").
   Table supply fell from 13,477 to 13,266.
6. **P1 template match is fully gone.** v2 already broke it for the rounding type; v3 breaks it
   for every type, because the trace prefix changes the claim shape. The pre-registered
   mechanism read on P1's banked quads is now out-of-template for the whole lane and must be
   reported as a shape change, not a like-for-like comparison.

## Carried forward from v2, unchanged

Surface-parity discipline within every pair; P(label 0 \| absent) = 0.5000 over the derivation
core with 100% of core rows asserting an absent value; the 15% relational sub-block; the type
schedule; seed 1133; the six-body-row evidence budget with operand rows protected; the
serialization mixture; per-table diversity caps; and row-level disjointness from the H108 lane
(0 shared rows, claims or chunks).

## Phase 2 - what the coordinator must set

Data-only (A4 binding amendment i): no hinge, no auxiliary head, no objective change.

1. **Lane path** - `R14-H133_lane.parquet`. `R10-H108_lane.lane_train()` reads `claim` /
   `chunk[:chunk_max_chars]` / `label.cast(Float32)` / `tag` unchanged; extra columns ignored.
   Re-read through that path: 50,000 claims and chunks, Float32 labels in {0.0, 1.0}, zero
   nulls, max chunk 1,500 characters, tags `quant_deriv` and `quant_relational`.
2. **DANN groups** discovered from `sorted(set(tags))`; a trainer copied from
   `R14-H135_trainer.py` hardcodes `LANE_GROUPS` / `EXPECTED_GROUPS` and must be updated.
3. **Decide whether the H108 lane rides along** (clean mix + H133 only = 14 groups).
4. Recipe constants frozen: MAX_LEN 512, BATCH 48, LR 1e-5, OneCycleLR, 1 epoch, DANN lambda
   0.02 Ganin ramp.
5. H126 seeded pairing is a precondition; pilot kill at draw-1 finqa below control + 0.020.
6. Anti-gaming set re-constituted per B4 amendment (i), with `bind_row` >= 0.95 added.
7. **Rule on family (b)** before the arm runs - see the three branches above.

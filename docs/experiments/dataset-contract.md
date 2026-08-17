# Dataset Contract - groundrails training mix and evaluation surfaces

**Status: DRAFT, awaiting author agreement.** No dataset is verified against it and no conforming pipeline is built until this is signed off.

Every clause below exists because something in this campaign broke for want of it. The provenance of each is named, so no clause is here on taste.

## Scope

Two classes of artifact, with different obligations.

- **Training members** - anything entering the assembled mix: source corpora (`ragtruth_en`, `ragtruth` translated x7, `halueval`, `psiloqa`, `vitaminc`, `tabfact`) and constructed lanes (`quant_misbind`, `quant_scale_unit`, `frame_reject`, `attr_pool`, `path_bind`, `num_compare`, `num_rolebind`, and any future lane)
- **Evaluation surfaces** - the blind arena (10 RAGBench subsets), `gold_full`, and every held-out mechanism eval

A dataset may not be both. That is checked, not assumed.

## C1 - Label commensurability

The label must encode the predicate of the head that consumes it.

- **Declare the head** each member trains, and the predicate its label encodes
- **A member whose label encodes a different predicate goes to a parallel head or is not built.** The grounding scalar takes support labels only
- **Mandatory test** - claim-to-evidence containment on the NEGATIVE leg against the positive leg. Negatives attested at rates comparable to positives means the member is not teaching grounding
- **Bar** - report both distributions; a member whose negatives are >= 90% attested at a rate within 0.10 of its positives is REJECTED for the grounding head

*Provenance*: `R20-H175b`'s contrast lane held passage and claim fixed and flipped the label on question relevance, so both legs had identical grounding (containment 0.9129 each) and 66.4% of its negatives were fully attested claims labelled 0. It trained a relevance label into the support head, contradicting all 721,210 other rows and the shipped `ground()` semantics. Caught only after the author raised it; the run was killed mid-training.

## C2 - Disjointness from every evaluation surface

- **Three string forms, both directions**: raw, truncated to `CFG.chunk_max_chars`, and whitespace-collapsed case-folded
- **Against all three surfaces**: the arena, `gold_full`, and each held-out mechanism eval - not only the lane a member was built beside
- **Report counts per form.** A member passes only when all forms read zero

*Provenance*: two failures. `R20-H175b`'s eval was certified disjoint from its own lane and was 99.6% inside the training mix. And `R20-H177_eval_B` (4.5%) and `R17-H143_evalset` (1.8%) leak ONLY under whitespace normalisation - those passages entered the mix re-wrapped, invisible to exact matching. Every campaign disjointness claim made with exact matching alone is unproven in that direction.

## C3 - Split semantics are verified, never assumed

- **State the split axis** a corpus actually cuts on - document, question, claim, revision - measured from the archive, not read from the dataset card
- **An "official" split is not evidence of disjointness.** Test it

*Provenance*: PsiloQA cuts per question, not per document, so 5,368 of 5,687 held-out passages are byte-identical to training passages. VitaminC's official split is disjoint by `unique_id` and `case_id` but shares 1,214 pages, 110 claims and 221 evidence strings with train. Both were assumed clean; both were not.

## C4 - Contamination census with a live positive control

- **Instrument**: the banked R14-H136 form - 8-gram, Jaccard >= 0.3, bidirectional, KILL above 2%, per-corpus attribution
- **Synthetic spike control** - injected units must be detected 10/10 with 0 baseline hits
- **A LIVE positive control is required, not only the spike**: feed the gate text that is genuinely near-duplicate by construction and show it fires
- **Coverage stated** - units too short for an 8-gram instrument are counted and covered by exact matching

*Provenance*: the `gold_full` audit's live control (VitaminC's own test split, hitting 58 of 25,689 at max Jaccard 1.0) is what makes its clean verdict trustworthy rather than merely reassuring. A clean number from an unproven gate is not evidence.

## C5 - Leak suite for constructed members

Every constructed lane and every paired-contrast eval:

- claim-only converged probe < 0.55; within-pair claim-only < 0.60
- single-channel probes (question-only, evidence-only) at chance where the construction implies it
- surface parity 0.45-0.55 on every computable channel
- direction / element / family balance; attestation symmetry
- **Any executor-added probe is reported separately from the registered conjunction** and cannot silently join it

*Provenance*: the `R20-H175b` and `R20-H177` builds; the separated bar accounting was adopted after an executor's stricter probe would otherwise have drifted into the registered set.

## C6 - No memorisation channel

For any member or eval whose pairs share fields, verify that no feature keyed on training associations separates the classes.

- **Test** - for each pair, measure overlap between the eval claim and whatever the training mix associates with that pair's key
- **Report the value.** On a clean instrument it is undefined or at chance

*Provenance*: the contaminated `R20-H175b` eval read 0.6230 on exactly such a feature, at 98% coverage, keyed on the question alone. Both registered floors were structurally blind to it, and so was the registered attribution control.

## C7 - Declared units and volume

- **State the unit** - rows or pairs - and use it consistently between registration, build and report
- **Report both** counts always

*Provenance*: `R20-H177` registered "~25-30k pairs" and reported rows, converting a 40-50% shortfall in the registered unit into "in band".

## C8 - Provenance, licence and internal structure

- Source, licence, retrieval date, and the exact selection predicate used (which split, which filter)
- Within-member duplication reported: distinct claims, distinct evidence, repeat structure
- **PUBLIC repository** - no client or company name in any artifact

## Verification output

Each dataset returns a single `contract_report.json`: per-clause PASS / FAIL / NOT-APPLICABLE with the measured number and its margin, the artifact paths, and an explicit `conforming` boolean. A FAIL names the binding constraint and whether it is fixable by a pipeline or is a corpus property.

## What happens on FAIL

- **Fixable** - a conforming pipeline is built, applied, and the member is re-verified from scratch against every clause, not only the failed one
- **Corpus property** - recorded as a finding with its consequence for whatever depends on the member. Supply constraints are accepted; bar relaxations are not
- **No clause is relaxed to make a member pass.** A smaller or absent member is preferable to a conforming-by-amendment one

## Amendment C-A1 (2026-08-17) - the containment channel is scoped to C1

**Status: ADOPTED.** Author delegated the ruling; verified by live positive control before adoption.

C1 and C5 as originally drafted were mutually unsatisfiable. C1 requires the claim-to-evidence containment channel to SEPARATE the legs; C5 requires every computable channel at chance. Containment is computable, so no member could satisfy both - proven empirically when the first pass failed C5 on `quant_scale_unit`, a banked lane that installed at 0.9555 with every hold green.

- **C5's parity requirement scopes to channels that do NOT read the claim-evidence relation** - features of the claim alone, of the evidence alone, and surface statistics of either. Containment is a joint feature and the quantity the grounding head exists to compute
- **Containment is governed by C1**, where separation is required rather than forbidden
- **C1's decisive test is STRUCTURAL**: if a negative leg's `(claim, evidence)` pair is identical to a positive leg's, the label cannot encode grounding, because no function of `(claim, evidence)` separates the legs. No threshold, no instrument choice
- **The distributional containment reading is a mandatory DIAGNOSTIC**, reported on both legs under at least one instrument sensitive to the predicate the lane corrupts. A predicate-blind instrument showing no separation is not evidence of incommensurability

**Live positive control** (run before adoption): the structural test fires on 8,986 of 8,986 pairs (100% of rows) in the withdrawn poisoned `R20-H175b_qlane`, and on 0 pairs in `frame_reject`, `attr_pool`, `path_bind`, `R17-H146_lane` and `R18-H150_scaleunit_lane`.

**The amendment rescues no member.** `frame_reject` still fails (claim-alone leak at AUROC 1.000, inside C5's narrowed scope); `attr_pool` still fails C6 and C2. No verdict moves from FAIL to PASS.

## Amendment C-A2 (2026-08-17) - C1's distributional test was ill-designed; C6 is scoped to mix-supplied associations. Test definitions FROZEN after this

**Status: ADOPTED.** Author delegated the ruling; verified by live positive control before adoption.

### The defect

C1's drafted bar - "negatives >= 90% attested at a rate within 0.10 of its positives is REJECTED" - was read faithfully by the executors as `|rate(neg >= 0.90) - rate(pos >= 0.90)| <= 0.10 -> REJECT`, and that test is wrong. Two small rates are always within 0.10 of each other no matter how well separated they are in ratio. It produced contradictory verdicts on materially identical evidence:

| member | neg rate >= 0.90 | pos rate >= 0.90 | ratio | executor verdict |
|---|---|---|---|---|
| `ragtruth_en` | 0.0067 | 0.0790 | 11.8x | **FAIL** |
| `psiloqa` | 0.0292 | 0.1383 | 4.7x | **FAIL** |
| `vitaminc` | 0.0169 | 0.1227 | 7.3x | **PASS** |

Three members, one clause, and the verdicts differ only by which reading each agent took. The clause's own provenance names the intended signal - `R20-H175b`'s negatives were **fully attested** claims labelled 0 - so the trigger was always the negative leg's ABSOLUTE attestation, never a small absolute gap.

### The test, restated

Decisive tests, in order:

1. **Structural (from C-A1)** - a negative leg's `(claim, evidence)` identical to a positive leg's means the label cannot encode grounding
2. **Strict separation** - under an instrument sensitive to the predicate the lane corrupts, the negative leg's high-attestation rate must be **strictly below** the positive leg's. Equality is the signature of a label independent of `(claim, evidence)`
3. **Absolute level, reported** - both legs' fully-attested and `>= 0.90` rates are reported always. A negative leg attested at a high absolute rate is recorded as a finding even when it clears test 2

The `within 0.10` band is **struck**. It never measured what the clause was written to catch.

### LIVE POSITIVE CONTROL, measured before adoption

Recomputed on the withdrawn poisoned `R20-H175b_qlane` (17,972 rows):

| leg | n | mean containment | rate >= 0.90 | rate = 1.0 |
|---|---|---|---|---|
| label 1 | 8,986 | 0.8158 | 0.6659 | 0.6145 |
| label 0 | 8,986 | 0.8158 | 0.6659 | 0.6145 |

**The two distributions are identical to four decimals** - the signature of a label that does not depend on the claim-evidence relation. Test 2 fires (0.6659 is not strictly below 0.6659) and test 1 fires at 100% of rows. Under test 3, negatives fully attested at 61.5% is a finding on its own.

Under the restated test every other member passes with a wide margin: `ragtruth_en` 0.0067 < 0.0790, `psiloqa` 0.0292 < 0.1383, `vitaminc` 0.0169 < 0.1227, `frame_reject` 0.0 < 0.0565, `quant_scale_unit` 0.0321 < 0.1011 under the unit-resolved instrument.

### C6 scoping

C6's clause text is eval-facing: "measure overlap between the eval claim and whatever the TRAINING MIX associates with that pair's key". It binds features keyed on associations **the training mix supplies**. That is the channel that caught `attr_pool` at 0.9999 within-pair, and it stands.

A **within-member** leave-one-out key lookup is a different question - whether rows inside one corpus predict each other - and is a reported diagnostic, not a C6 bar. `ragtruth_en`'s 0.6509 is recorded as a corpus property (multiple annotations share a source response), not a rejection. Where the eval-facing test has zero key coverage, C6 is NOT-APPLICABLE and no proxy is substituted.

### Freeze

**The contract's test definitions are FROZEN after this amendment.** Two amendments in one day is the limit; a third would be the pattern this campaign has already paid for twice (the table guard rewritten three times, the H175b instrument rebuilt three times before the arm was deleted). If a later finding lands on a contract test, the response is to re-model the clause from its provenance, not to amend it again.

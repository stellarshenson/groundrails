# Grounding Adjudication Rulebook

Labeling policy for adjudicating grounding disagreements where a claim's specificity differs from the evidence. The core rule is entailment: a claim is SUPPORTED only when the evidence entails it - the claim asserts neither more breadth (a range where the evidence gives a point) nor more precision (a point where the evidence gives a range) than the evidence establishes, and never contradicts it.

## Core principle

- **SUPPORTED = the evidence entails the claim** - nothing the claim asserts goes beyond what the evidence states
- **Specificity must match** - a claim may not be broader than the evidence (range asserted over a single stated value) nor narrower (single value asserted over a stated range)
- **Contradiction always wins** - if any claimed value falls outside what the evidence states, the label is UNSUPPORTED regardless of the rules below

## Quantitative rules (confirmed)

These four are the confirmed policy for numeric-range and generalization claims.

- **R1 range covered by range** - claim states a range; evidence states that same range or a wider one that covers it → SUPPORTED
- **R2 range over a single value** - claim states a range or approximation; evidence states one concrete value → UNSUPPORTED, even when the value lies inside the claimed range; the evidence pins a point and the claim asserts a span the evidence never established
- **R3 generalization over many values** - claim generalizes; evidence lists several concrete values and the generalization entails (covers) all of them → SUPPORTED
- **R4 generalization not fully covered** - claim generalizes or ranges; at least one evidence value falls outside the claimed range → UNSUPPORTED; this subsumes the single-value case (R2) and the partial-cover case
- **R5 exact matches exact** - claim states a value; evidence states the same value → SUPPORTED; a different value → UNSUPPORTED (contradiction)
- **R6 exact over a range** - claim states an exact value; evidence states only a range → UNSUPPORTED; a range does not establish any single point inside it (the mirror of R2)

## Decision procedure

1. Classify the claim: exact value, range/bound, or generalization
2. Read what the evidence establishes: one concrete value, a stated range, or several concrete values
3. Apply the table

| Evidence establishes | Claim is an exact value | Claim is a range / generalization |
|---|---|---|
| one concrete value V | SUPPORTED if claim = V, else UNSUPPORTED | UNSUPPORTED (R2 / R4) even when V lies inside the claim |
| a stated range [a, b] | UNSUPPORTED - a point is not established by a range (R6) | SUPPORTED if the claim covers [a, b] (R1), else UNSUPPORTED |
| several values {v₁ … vₙ} | SUPPORTED if claim equals some vᵢ | SUPPORTED if the claim covers all vᵢ (R3), else UNSUPPORTED (R4) |

## Schematic examples

Values are synthetic, for illustration only.

- evidence "circulation 7 min", claim "6 - 10 minutes" → UNSUPPORTED (R2: one value, claim is a range)
- evidence "0.57 % and 0.62 %", claim "~0.5 - 0.7 %" → SUPPORTED (R3: the range entails both values)
- evidence "7 min and 4 min", claim "6 - 10 minutes" → UNSUPPORTED (R4: 4 falls outside the range)
- evidence "end temperature 35 °C", claim "≥ 45 °C" → UNSUPPORTED (contradiction: 35 is not ≥ 45)
- evidence "range 6 - 10 min", claim "6 - 10 minutes" → SUPPORTED (R1)
- evidence "detergent 0.57 %", claim "0.57 %" → SUPPORTED (R5)

## Non-quantitative categories

Rules for the other two flagged categories.

- **Negative or meta claim** (confirmed) - a claim about what the documentation does or does not contain ("the documentation does not specify X"); a global negative cannot be confirmed from a local retrieved snippet, so it is not entailed by the evidence → UNSUPPORTED (unverifiable in scope)
- **Formatting fragment** (default, confirm) - the claim extractor sometimes emits a markdown table piece as a claim; split by what it carries:
  - a header row or divider (column labels, dashes) with no data → NOT_A_CLAIM (excluded from SUPPORTED/UNSUPPORTED)
  - a data row that carries actual values → judged as a normal claim about those values (SUPPORTED if the evidence contains them, else UNSUPPORTED)

## Application

- The rulebook is the adjudication policy for the ensemble-vs-gold disagreements and for the ambiguous set flagged for human judgement
- It is applied as the judge's instruction (the policy text is prepended to the grounding prompt) so a machine pass adjudicates consistently with these rules, and as the reference a human uses when reviewing the flagged cases
- All rules confirmed; the formatting-fragment split (header → NOT_A_CLAIM, data row → claim) is the working default

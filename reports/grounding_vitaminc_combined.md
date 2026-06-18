# Combined grounder on VitaminC (gold v4)

VitaminC dev, 800 rows (SUPPORTS 400 / REFUTES 400), English single-sentence contrastive evidence. The combined grounder fuses the lexical manifold (effort=high) with the OV int8 cascade (bge-m3 -> bge-reranker + mDeBERTa-NLI) via the frozen v1 joint head. This is the regime where the lexical tier collapses: a negative is one edited token in matching text.

## Macro-F1

| configuration | macro-F1 | note |
|---|---|---|
| lexical-only (high, shipped verdict) | 0.701 | token-overlap baseline |
| combined v1-head, gold v3 operating cut (T=0.50) | 0.715 | frozen head + transferred threshold (honest) |
| combined v1-head, VitaminC-optimal cut (T=0.61) | 0.734 | in-sample threshold ceiling |

At the transferred operating point: support-recall 0.807, TNR 0.627.

## Per-signal separation (AUC on SUPPORTS vs REFUTES)

| signal | AUC |
|---|---|
| lex_p (lexical verdict) | 0.766 |
| cos_max (bi-encoder) | 0.594 |
| rr_max (reranker) | 0.482 |
| nli_ent (entailment) | 0.382 |
| nli_contra (contradiction, inverted) | 0.649 |
| v1-head (fused) | 0.784 |

Lift combined over lexical-only: +0.014 (transferred cut), +0.032 (in-sample ceiling).


# Joint lexical + semantic wirings - benchmark

Verified gold: 2752 claims (786 hallucination / 1966 supported, base rate 29%). 42 claims are cross-lingual with no lexical (argos) model - the lexical tier cannot ground them. Cascade signals from the cached int8 pair scores; lexical P from the live effort=high manifold. OOF 5-fold (seed 42), thresholds at the macro-F1 optimum.

## Wirings (out-of-fold)

| wiring | macro-F1 | FP | FN | FP+FN | escalation | d macro vs lexical |
|---|---|---|---|---|---|---|
| lexical-only (high) | 0.759 | 514 | 97 | 611 | - |  |
| W3 reuse-seam {lex,cos,nli} | 0.826 | 155 | 225 | 380 | 100% | +0.067 |
| W2 always-both joint | 0.822 | 243 | 167 | 410 | 100% | +0.063 |
| W1 escalation cascade | 0.822 | 247 | 164 | 411 | 90% | +0.063 |

**Lexical-only (high)** at its shipped manifold threshold scores macro-F1 0.759 (FP 514, FN 97); 42 of the claims are cross-lingual with no lexical (argos) model and are flagged unconfirmed.

All three wirings cluster at macro-F1 0.822-0.826 (within noise at n=2752), each +0.06-0.07 over lexical-only. The shipped high manifold over-flags supported claims (high FP), so escalation favours a wide band.

**Shipped as the `semantic` tier: W1 escalation** (macro-F1 0.822, FP 247, FN 164). It is the requested design - lexical decides, the uncertain band escalates to the cascade - carries the best hallucination recall (lowest FN) and is the only wiring with a cost lever (90% escalation here at band [0.13, 0.99]). A better-calibrated lexical gate would narrow the band and cut the cascade share further at the same quality.


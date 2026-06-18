# Joint lexical + semantic, cross-lingual hypotheses (gold v3)

Golden eval 5857 rows (1965 hallucination), synthetic aug 2119 negatives, 24 eval languages. GroupKFold leave-one-source-out on `group_id`; headline macro-F1 on `role=eval`, synthetic scored as an offline TNR probe. Each hypothesis gates on a precondition first.

## Baselines (role=eval)

| baseline | macro-F1 | EN macro | non-EN macro | non-EN sup-recall |
|---|---|---|---|---|
| lexical-only (high) | 0.763 | 0.802 | 0.559 | 0.65 |
| joint v1-head (frozen) | 0.809 | 0.831 | 0.639 | 0.80 |

Retrained joint heads (OOF, GroupKFold): eval-only macro 0.810 (T=0.54); eval+synthetic macro 0.805 (T=0.44).

## Hypotheses

| id | mechanism | result | kill-gate |
|---|---|---|---|
| R1-H1 | native multilingual cascade (no MT bridge) | non-EN cascade AUC max(cos 0.584, nli_ent 0.520) = 0.584 | gate >= 0.65: KILL; bar >= 0.75: below |
| R1-H2 | synthetic negatives lift cross-lingual TNR | synthetic TNR 0.904 (bar >= 0.80), non-EN sup-recall nan (bar >= 0.70) | gate nli_ent AUC 0.523 >= 0.70: KILL |
| R1-H4 | joint head retrained on enriched gold | non-EN macro 0.639 (v1) -> 0.634  (lift -0.005, bar >= +0.05); EN macro 0.831 -> 0.832 (control +/-0.005) | gate v1 non-EN err 0.211 - EN err 0.160 = +0.051 >= 0.05: PASS |
| R1-H3 | per-language joint calibration | macro global 0.805 -> per-language 0.831 (lift +0.026, bar >= 0.83 abs) | gate max |T_lang - T_global| = 0.380 >= 0.03: PASS |
| R1-H5 | language-aware escalation band | fused macro 0.800 (bar >= 0.825), EN escalation share 85.2% | gate non-EN lex err 0.330 - EN lex err 0.193 = +0.137 >= 0.10: PASS |


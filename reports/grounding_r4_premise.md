# R4-H1: joint-premise NLI (SummaC aggregation) - honest evaluation

Joint-premise scored on 3218 cascade-fired rows; gold v3 eval 5857 rows, 7 calibrated languages. EN/non-EN leave-one-fold-out thresholds on GroupKFold leave-one-source-out OOF probabilities (the Round 3 honest harness).

| head | eval macro-F1 | EN macro | non-EN macro | synthetic TNR |
|---|---|---|---|---|
| R3 baseline (max-over-chunks NLI) | 0.825 | 0.833 | 0.650 | 0.852 |
| R4 joint-premise NLI | 0.823 | 0.828 | 0.669 | 0.899 |
| delta | -0.002 | -0.006 | +0.019 | +0.047 |

Mechanism gate (under-graded supported rows, n=238): joined premise raises entailment >= 0.10 on 21.0% of them, mean rise +0.051.

Bar: macro lift >= 0.014 AND |EN ctrl| <= 0.005 AND synthetic TNR >= 0.88.
Measured: lift -0.002, EN ctrl -0.006, TNR 0.899.

Verdict: **NULL/REFUTED**


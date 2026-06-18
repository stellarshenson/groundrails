# Grounding Ensemble - meta-classifier over model scores

Verified gold: 2752 claims (786 hallucination / 1966 supported). Features = 6 per-model scores + lexical contradiction/fired flags. Out-of-fold 5-fold CV; single-signal AUCs are pretrained (not fit on labels), so honest as-is.

## Single-signal AUC (best to worst)
- bge_reranker: 0.841
- mdeberta_nli: 0.806
- bge_m3: 0.730
- e5_small: 0.635
- e5_large: 0.621
- mmbert: 0.529

## Meta-classifier (out-of-fold)
- logreg: AUC 0.903 +/- 0.016
- gbm: AUC 0.913 +/- 0.012

Best single = bge_reranker 0.841; best meta = gbm 0.913 +/- 0.012 (beats the best single signal).

## Driving metric - macro-F1 and error counts (out-of-fold)

**Macro-F1** (mean of the hallucination-class and supported-class F1, both classes weighted equally) is the metric. The target is to raise it by cutting the two error counts: **FP** = supported claims wrongly flagged, **FN** = hallucinations missed. Each signal at its own macro-F1-optimal threshold over n=2752 (hallucination base rate 786/2752 = 29%).

| signal | threshold | **macro-F1** | FP | FN | FP+FN | accuracy |
|---|---|---|---|---|---|---|
| bge-reranker (best single) | 0.24 | **0.76** | 238 | 295 | 533 | 81% |
| meta-classifier (gbm) | 0.63 | **0.82** | 248 | 160 | 408 | 85% |

The meta-classifier lifts macro-F1 **0.757 -> 0.824** and cuts total errors **533 -> 408** (-125, -23%): FP 238->248, FN 295->160. Majority-class baseline macro-F1 = 0.417 (always-predict-supported -> hallucination-class F1 = 0).

## Learned feature weights (logreg, + = predicts supported)
- bge_m3: +1.34
- bge_reranker: +1.14
- lexical_fired: +0.91
- mdeberta_nli: +0.87
- e5_large: -0.69
- mmbert: -0.46
- e5_small: -0.39
- contradiction: +0.18

## Per-language AUC (n>=20)
- en (n=2117): 0.917
- nl-NL (n=37): 0.833
- es-ES (n=113): 0.802
- fr-FR (n=270): 0.782
- nb-NO (n=149): 0.659
- pt-PT (n=22): 0.635

## Error analysis (at the macro-F1-optimal threshold)
- FN - missed hallucinations: 160, of which 102 contain a number/spec
- FP - wrongly-flagged supported: 248
- the missed hallucinations are the overlap region a fine-tuned cross-encoder would target

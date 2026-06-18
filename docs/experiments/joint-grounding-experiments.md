# Joint lexical + semantic grounding on the cross-lingual gold (gold v3)

Pre-registered hypothesis log for the joint grounder - the lexical manifold fused with the
OpenVINO int8 cascade - evaluated on the enriched cross-lingual gold (gold v3). The lexical
track learned the cross-lingual boundary from synthetic negatives (see
[lexical-grounding-experiments](lexical-grounding-experiments.md), Rounds 9-12); this log
asks whether the semantic / joint grounder does better, on the same enriched data. Branch
artefacts: `experiments/grounding-semantic/{score_enriched,lex_v3,joint_xlingual}.py`; gold
in the gitignored `data/processed/golden_v3*.parquet`; report `reports/grounding_joint_xlingual.md`.

> Rounds 1-3 complete. The five Round-1 hypotheses were pre-registered (mechanism, prediction,
> bar, kill-gate) before the cascade was scored; Round 2 probed resampling; Round 3 found the
> fix. Outcome: a language-split operating point - one threshold for English, one for non-English,
> both chosen out-of-fold - lifts the frozen v1 head from macro-F1 0.808 to **0.829** on gold v3,
> recovering and exceeding the original-gold 0.822/0.826. The Round-1 hypotheses (cascade-as-MT,
> retraining, escalation, resampling) were null; the win is recalibration, not new signals.

## Problem overview

The shipped semantic switch lifts macro-F1 0.759 (lexical-only, high) to 0.822 on the
original 2,752-claim gold, but that gold is 90% English and the cross-lingual story was a
42-claim tail. Gold v3 makes the cross-lingual axis a real benchmark, and the joint grounder
has never been calibrated on it.

- **Golden (eval)** - 5,857 verified rows, 3,892 supported / 1,965 hallucination (33.5%), 619 evidence sources
- **Languages (eval, n >= 40)** - en 4,524, fr 442, nb 269, es 247, it 129, pt 57, nl 56
- **Non-English skew** - supported-heavy (fr 414/28, nb 252/17, es 212/35); native rows measure recall, not false-flag
- **Synthetic augmentation** - 2,119 cross-lingual negatives across 9 languages, derived from 240 golden source sentences (haiku-translated, sonnet-verified); the negative side the native eval lacks
- **Lexical pass** - over gold v3, 81 cross-lingual claims blocked (no argos model), lexical layer fires on 43.3%
- **Core difficulty** - the lexical tier is weakest cross-lingual (R12 non-EN support recall 0.66); a single English-tuned operating point under-serves the non-English tail

## Executive summary

A language-split operating point lifts the joint grounder from macro-F1 **0.808 to 0.829** on
the gold v3 eval (Round 3) - leak-free, and back above the original-gold 0.822/0.826. The fix
is recalibration, not new signals: the shipped single threshold is English-tuned, and one cut
for English plus one for non-English (each chosen out-of-fold) judges each language group at its
own scale. English holds at 0.834; non-EN rises 0.637 -> 0.666. The Round-1 hypotheses (native
cascade, retraining, escalation) and the Round-2 resampling probe were null - the lever was the
operating point all along.

The non-EN macro reads low (0.637 global) mostly as a class-imbalance artifact: the native
non-EN eval is supported-heavy, so macro-F1 is dominated by a tiny hallucination class. On the
proper instrument - the 2,119 synthetic cross-lingual negatives - the joint head already rejects
**90.4%** (OOF synthetic TNR), at non-EN supported recall 0.80.

| # | hypothesis | dataset | key result | verdict |
|---|---|---|---|---|
| R1-H1 | native multilingual cascade vs MT bridge | gold v3 non-EN | cascade non-EN AUC 0.584 (cos), 0.520 (nli_ent) | Killed-at-gate |
| R1-H2 | synthetic negatives lift cross-lingual TNR | gold v3 + synth | OOF synthetic TNR 0.904 (bar met) but nli_ent gate 0.523; eval macro 0.805 <= v1 0.809 | Null (mechanism refuted) |
| R1-H3 | per-language joint calibration | gold v3 (n >= 40) | macro 0.831 via in-sample per-language cuts (not OOF-fair) | Null (optimistic) |
| R1-H4 | joint head retrained on enriched gold | gold v3 | non-EN macro 0.639 -> 0.634 (lift -0.005); EN held | Dropped |
| R1-H5 | language-aware escalation band | gold v3 | fused macro 0.800 < 0.825 bar; v1 was 0.809 | Dropped |

Joint-solution benchmark (gold v3 eval, GroupKFold leave-one-source-out; thresholds chosen
out-of-fold where noted):

| configuration | macro-F1 | EN macro | non-EN macro | non-EN sup-recall |
|---|---|---|---|---|
| lexical-only (high) | 0.763 | 0.802 | 0.559 | 0.65 |
| joint v1-head, global threshold (in-sample) | 0.809 | 0.831 | 0.639 | 0.80 |
| joint v1-head, global threshold (honest OOF) | 0.808 | 0.831 | 0.637 | - |
| joint retrained eval-only (OOF) | 0.810 | - | - | - |
| joint retrained eval+synthetic (OOF) | 0.805 | - | - | - |
| **joint v1-head + EN/non-EN split (honest OOF)** | **0.829** | **0.834** | **0.666** | - |
| joint v1-head + per-language split (honest OOF) | 0.824 | 0.834 | 0.650 | - |

Reference: the original-gold figures were macro-F1 0.822 (W1) / 0.826 (W3) on a 90%-English set.
Gold v3 is harder (a real multilingual tail), so the 0.808 global figure is not a regression -
the English slice 0.831 matches the shipped level. The EN/non-EN split clears the ~0.014 noise
band over the honest global (0.808 -> 0.829) and lands above the original-gold benchmark.

## Methodology

- **Signals** - lexical `lex_p` (effort=high) + blocked/fired/contradiction flags; cascade `cos_max`, `rr_max`, `nli_ent`, `nli_contra` (bge-m3 + bge-reranker + mDeBERTa-NLI int8)
- **Joint head** - frozen logistic over `{lex_p, rr_max, nli_ent, cos_max, nli_contra, lex_contra, lex_blocked}`; no scikit-learn at inference
- **Metric** - macro-F1 on `role=eval`; cross-lingual TNR (offline) on the synthetic negatives; per-language support recall
- **Splits** - GroupKFold leave-one-source-out on `group_id` (a base source and its translations never straddle train/test); synthetic trains in-fold only, scored OOF as a probe
- **Determinism** - frozen weights, fixed grid threshold; no per-call sampling

## Setup

- **Data** - `data/processed/golden_v3.parquet` (eval) + `golden_v3_synth_aug.parquet` (augmentation), joined to the cached signals on `uid`
- **Signals** - `golden_v3_cascade_scores.parquet` (`score_enriched.py`, OV int8 cascade) + `golden_v3_lex.parquet` (`lex_v3.py`, high manifold); both GPU-free
- **Run** - `python experiments/grounding-semantic/joint_xlingual.py` -> `reports/grounding_joint_xlingual.md`
- **Dependencies** - the `semantic-grounder` extra (openvino + transformers); int8 IRs from the HuggingFace Hub

## Round 1: cross-lingual hypotheses

Each hypothesis states one causal claim, a falsifiable prediction, a pre-registered
acceptance bar, and a cheap diagnostic kill-gate measured before the full wiring.

### R1-H1 native multilingual cascade vs the MT bridge

Because the cascade encoders (bge-m3, mDeBERTa-XNLI) are natively multilingual while the
lexical tier depends on a lossy argos bridge and hard-blocks languages with no model,
scoring the cascade directly on non-English claims should separate supported from
hallucination without any MT bridge.

- **Prediction** - non-EN cascade AUC (max of cos_max, nli_ent) >= 0.75; every enriched language covered
- **Bar** - non-EN AUC >= 0.75 AND no blocked-language tail
- **Kill-gate** - 200-row non-EN sample AUC >= 0.65, else the cascade is no better cross-lingual than the bridge
- **Result** - **Killed-at-gate**. Non-EN cascade AUC max(cos_max 0.584, nli_ent 0.520) = 0.584, below the 0.65 gate and far below the 0.75 bar. The cascade alone does not separate supported from hallucination on non-English claims. Caveat: the native non-EN eval is supported-heavy (few hallucinations), so this AUC sits on a thin negative class and is noisy; the cleaner cross-lingual rejection signal is the synthetic TNR in R1-H2, not this AUC. Either way the prediction (>= 0.75) is refuted

### R1-H2 synthetic negatives lift cross-lingual TNR

Because NLI entailment tests support rather than token overlap, calibrating the joint head
with the 2,119 synthetic negatives in-fold should reject translated-but-unsupported claims
more cleanly than lexical recall.

- **Prediction** - cross-lingual TNR >= 0.80 (vs lexical 0.78); non-EN support recall >= 0.70 (vs 0.66)
- **Bar** - TNR >= 0.80 AND non-EN support recall >= 0.70 (two-sided)
- **Kill-gate** - `nli_ent` separates the synthetic negatives from supported non-EN at AUC >= 0.70
- **Result** - **Null, mechanism refuted**. The OOF head trained with the synthetic negatives rejects 90.4% of held-out synthetic cross-lingual negatives (synthetic TNR 0.904) - the prediction (TNR >= 0.80) is met. But the stated mechanism is false: the `nli_ent` kill-gate is 0.523 (near chance, bar 0.70), so the rejection comes from the lexical and reranker channels, not NLI entailment. And it does not ship: adding synthetic to training moves the eval headline 0.810 (eval-only) -> 0.805, below the frozen v1 head (0.809). Synthetic negatives buy cross-lingual TNR at a small eval-macro cost, via features other than the hypothesised one

### R1-H3 per-language joint calibration

Because per-language score distributions differ and the enriched set now has n >= 40 for
about nine languages, fixed per-language thresholds (global cut elsewhere) should beat a
single global cut. Distinct from the refuted batch-adaptive gap detection - these are fixed
cuts, not per-batch.

- **Prediction** - enriched macro-F1 >= 0.83 with per-language cuts
- **Bar** - macro-F1 >= 0.83 AND no slice (n >= 40) regressed vs global
- **Kill-gate** - at least one language (n >= 40) with an OOF-optimal cut >= 0.03 from the global cut
- **Result** - **Null, optimistic**. Per-language cuts lift macro 0.805 -> 0.831, nominally clearing the 0.83 bar, and the gate passes (max |T_lang - T_global| = 0.380). But the per-language thresholds are fit in-sample on the same rows they score - a learner touching its own scoring data - so 0.831 is an upper bound, not an OOF-fair number. The honest comparator is the OOF eval-only retrain (0.810), which shows no lift. Verdict pends a nested-CV per-language calibration; on current evidence the lift is overfitting, not signal

### R1-H4 joint head retrained on the enriched multilingual gold

Because the shipped joint head was fit on a 90%-English gold, its weights under-represent
non-English score scales; retraining OOF on the enriched gold should lift the non-English
slice without touching English.

- **Prediction** - non-EN macro-F1 +>= 0.05 over the v1 head; English within +/-0.005
- **Bar** - non-EN lift >= 0.05 AND English not regressed (control)
- **Kill-gate** - the v1 head's non-EN error exceeds its English error by >= 5pp (evidence of English bias)
- **Result** - **Dropped**. The gate passes - the v1 head does carry an English bias (non-EN error 0.211 vs English 0.160, diff +0.051) - but the fix fails: retraining OOF on the enriched gold moves non-EN macro 0.639 -> 0.634 (lift -0.005, bar +0.05) while English holds (0.831 -> 0.832). The bias is real but reweighting the head does not close it; the non-EN ceiling is set by the signals, not the head's English-tilted weights

### R1-H5 language-aware escalation band

Because the lexical tier is weakest cross-lingual yet the escalation band routes by `lex_p`
alone (language-blind), always escalating non-English claims and band-escalating English
should hold quality while cutting the English cascade share.

- **Prediction** - enriched macro-F1 >= 0.825; English escalation share cut >= 20pp vs the uniform band
- **Bar** - macro-F1 >= 0.825 (no regression) AND English escalation reduced >= 20pp
- **Kill-gate** - non-EN lexical error exceeds English lexical error by >= 10pp
- **Result** - **Dropped**. The gate passes - the lexical tier is much weaker cross-lingual (non-EN lexical error 0.330 vs English 0.193, diff +0.137) - but the routing hurts: always-escalate-non-EN plus band-escalate-English gives fused macro 0.800, below both the 0.825 bar and the plain v1 head (0.809). Routing by language does not help when the escalation target (the cascade) is itself weak on the non-EN slice (R1-H1); escalating into a weak signal cannot beat trusting the head directly

## Round 2: resampling probe (cheap)

R1-H4 left a question - is the non-EN gap class imbalance in training, or the signals? The
cheapest test before harder negatives: resample the OOF training folds toward the minority /
non-English rows and re-measure. `experiments/grounding-semantic/resample_probe.py`, same
cached signals, GroupKFold leave-one-source-out, eval-only training, seed 0.

| training resample | macro | EN | non-EN | non-EN TNR | non-EN recall |
|---|---|---|---|---|---|
| none (R1-H4 retrain) | 0.810 | 0.832 | 0.634 | 0.640 | 0.812 |
| class_weight balanced | 0.811 | 0.833 | 0.638 | 0.662 | 0.809 |
| oversample minority | 0.811 | 0.838 | 0.624 | 0.698 | 0.780 |
| undersample majority | 0.810 | 0.833 | 0.635 | 0.676 | 0.801 |
| oversample non-EN | 0.810 | 0.834 | 0.624 | 0.640 | 0.799 |
| oversample non-EN negatives | 0.810 | 0.836 | 0.630 | 0.698 | 0.787 |

- **Verdict: Null**. No resample lifts macro past the ~0.014 noise band - overall 0.810 -> 0.811, non-EN 0.634 -> 0.638 at best
- **What it does** - resampling toward negatives trades the operating point: non-EN TNR rises 0.640 -> 0.698 but non-EN recall falls 0.812 -> 0.780; macro is flat because the gain and loss cancel. The macro-F1-optimal threshold already absorbs the imbalance, so resampling adds little on top
- **Reading** - the non-EN macro ceiling (~0.63) is set by signal separability on that slice, not by training-set balance; cheap rebalancing cannot move it. A real lift needs harder evidence (stronger negatives) or a new signal, not a reweighting of the existing one

## Round 3: honest language-split calibration (the fix)

R1-H3 lifted macro to 0.831 with per-language cuts, but those were fit in-sample. Round 3 redoes
it leak-free: OOF model probs (GroupKFold leave-one-source-out) AND nested leave-one-fold-out
threshold selection, so no row's verdict saw its own label in either the head or the cut. Three
schemes on the frozen v1 head. `experiments/grounding-semantic/perlang_honest.py`.

| scheme | macro | EN | non-EN |
|---|---|---|---|
| global (single threshold) | 0.808 | 0.831 | 0.637 |
| EN / non-EN (two thresholds) | 0.829 | 0.834 | 0.666 |
| per-language (n >= 40) | 0.824 | 0.834 | 0.650 |

- **Verdict: Ships (candidate)**. The EN/non-EN split lifts macro 0.808 -> 0.829 honestly, +0.021 past the noise band, recovering the original-gold 0.822/0.826. Non-EN rises 0.637 -> 0.666; English holds 0.831 -> 0.834 (no regression)
- **Why it works** - the shipped single threshold is tuned on a 90%-English distribution; non-English claims sit on a different score scale, so one cut misjudges them. A cut per language group restores the operating point. The robust 2-threshold split beats per-language (0.829 > 0.824) - per-language overfits the small slices, EN/non-EN has thousands / hundreds of rows each
- **R1-H3 reconciled** - the mechanism (per-language operating points) was right; the in-sample 0.831 was optimistic, but the honest coarse form (EN vs non-EN) still clears the bar over the global baseline. The earlier Null verdict was for the in-sample per-language cut, not this
- **Cost to ship** - two scalar thresholds plus the language flag the lexical layer already detects; deterministic, no new model, no retraining

## Round 4: joint-premise NLI (the SummaC aggregation)

Rounds 1-3 closed the recalibration lever; Round 4 opens a signal lever - a fix to how the
cascade aggregates NLI. The cascade computes `nli_ent` / `nli_contra` as **max-over-chunks**:
each source chunk is graded against the claim independently and the max is taken. That is the
single-premise failure mode the SummaC multi-premise pattern (Laban et al., TACL 2022) was
built for - a supported claim that fuses several source sentences is entailed by none of them
alone, so the max stays low and the claim reads as unsupported. A sibling internal experiment
on the identical `bge-reranker-v2-m3` + `mdeberta-v3-base-mnli-xnli` int8 stack ported SummaC
(top-3 reranked source statements joined into one premise) and moved its information-loss
residual 0.206 → 0.130 (to gold level) while holding fabrication.

Empirical anchor (gold v4 - the 800-row VitaminC contrastive slice folded onto gold v3,
`role=eval_vitaminc`): the combined grounder lifts only 0.701 → 0.715 over lexical-only (within
the noise band), and the per-signal AUC shows why - reranker `rr_max` 0.482 (random; both
contrastive claims share one evidence sentence) and `nli_ent` 0.382 (inverted), with only
`nli_contra` 0.649 and the lexical verdict `lex_p` 0.766 carrying signal. The cascade's NLI is
mis-firing on evidence it should grade - the case for fixing the aggregation, not the model.
Note VitaminC is single-sentence, so R4-H1 targets the multi-chunk gold v3 / RAG regime, not it.

### R4-H1 top-3 joint-premise NLI

Because a supported claim can be entailed only by combining several source chunks, and the
cascade grades each chunk independently then takes the max, replacing max-over-chunks NLI with
the entailment of the claim against the top-3 reranked chunks joined into one premise will raise
supported-recall on multi-chunk-evidence claims and lift gold v3 macro-F1 without a new model.

- **Prediction** - gold v3 eval macro-F1 +>= 0.014 over the matched-threshold baseline (honest global 0.808), driven by English supported-recall on multi-chunk claims; English not regressed; synthetic cross-lingual TNR held
- **Bar** - macro-F1 lift >= 0.014 (clears the noise band) AND English macro within +/-0.005 AND synthetic TNR >= 0.88 (the joint premise must not over-confirm); the lift must survive stacking with the Round 3 EN/non-EN split
- **Kill-gate** - on a 300-row supported-eval sample, the fraction whose 2nd-ranked chunk carries reranker relevance within 0.10 of the top chunk is >= 15% (enough genuinely multi-chunk support to join), AND on the currently under-graded supported claims (low `nli_ent`, high `lex_p`) the joined-premise `nli_ent` rises >= 0.10 over the max-over-chunks value; if support almost always sits in one chunk, there is nothing to join - kill before the full re-score
- **Method** - add a joint-premise path to the cascade scorer (rank chunks by the reranker, join the top-3 into one premise, one NLI pass → `nli_ent_joint`, `nli_contra_joint`), swap these into `JOINT_FEATURES`, re-evaluate through the Round 3 honest harness (GroupKFold leave-one-source-out + leave-one-fold-out thresholds); re-scores gold v3 eval only (5,857 rows), reuses the shipped int8 models, no new download
- **Caveat to measure** - groundrails chunks at 1100/200 chars, larger than docdistance's sentence statements, so one chunk often already spans several sentences and the multi-chunk gain is smaller than the sentence-level result; the kill-gate quantifies the real multi-chunk support density before committing the re-score
- **Companion throughput lever** - length-bucketing the reranker grid (sort pairs by length, tighten `max_length` to the per-call percentile) cut the same int8 reranker ~43% at bit-identical scores in the sibling experiment; fold it into the re-score to offset the joint-premise NLI cost
- **Verdict** - **Planned** (pre-registered; not yet run)

## Lessons learned

- **Macro-F1 on a supported-heavy slice misleads** - the non-EN macro (0.639) reads like a model failure but is mostly class imbalance: few native hallucinations, so the hallucination-class F1 dominates the average. The synthetic negatives are the right instrument - on them the head rejects 90.4%
- **Measure cross-lingual rejection on negatives, not AUC over a thin negative class** - R1-H1's non-EN AUC 0.584 is noisy because the eval barely has non-EN negatives; the synthetic TNR is the trustworthy readout
- **A met prediction can hide a refuted mechanism** - R1-H2 hit its TNR bar while its `nli_ent` mechanism was near chance (0.523). The kill-gate caught it: the rejection is lexical and reranker work, not NLI entailment
- **In-sample per-language thresholds overfit** - R1-H3's 0.831 came from cuts fit on the rows they score; the OOF-fair comparator (eval-only retrain, 0.810) shows no lift. One free threshold per language buys apparent macro a fair split removes
- **The English operating point is robust** - across the frozen head, the retrain, and the per-language variant, English macro sits at 0.831-0.832; gold v3 did not perturb the shipped English behaviour
- **Routing into a weak signal cannot help** - R1-H5 escalates non-EN into the cascade, but the cascade is itself weak there (R1-H1), so the fused verdict (0.800) trails the head alone (0.809)

## Conclusions

- **The fix is a language-split operating point** - one threshold for English, one for non-English, both OOF-chosen, lift the frozen v1 head from macro-F1 0.808 to 0.829 on gold v3 (Round 3), recovering and exceeding the original-gold 0.822/0.826. Ship candidate: two scalar thresholds plus the language flag the lexical layer already computes
- **The win is recalibration, not new signals or weights** - retraining the head (R1-H4, 0.810) and resampling (Round 2, 0.811) were flat; the gap was the single English-tuned threshold, not the head or the features
- **No regression underneath** - the English slice holds at 0.831-0.834 throughout; the original-to-gold-v3 drop was the harder multilingual benchmark plus one mis-set operating point, both now accounted for
- **Cross-lingual robustness was already present** - OOF synthetic TNR 0.904 and non-EN supported recall 0.80; the recalibration converts that latent robustness into macro
- **SOTA** - the lift is a calibration change to the existing joint design, not a new architecture; fold the EN/non-EN threshold into the semantic-grounding SOTA's operating-point section rather than open a new SOTA doc. Integration into `config_document_processing.yaml` is pending explicit approval (out of this experiment's scope)

## Next steps

- **Integrate the EN/non-EN operating point (Round 3 winner)** - fold the two honest thresholds into the serving calibration (one English cut, one non-English cut, keyed by the lexical layer's language detector) and validate macro 0.829 holds end-to-end; pending explicit approval to touch `config_document_processing.yaml`
- **Strengthen the negatives (next, after the null resampling probe)** - the synthetic negatives are off-topic or clearly-unsupported translations (easy rejects), so the 90.4% TNR may flatter the head. Generate hard near-miss negatives per language: minimal edits to a SUPPORTED translated claim that flip its truth - numeric perturbation (42 -> 43), entity swap, negation, scope or quantifier change - so the claim stays topically and lexically close to its evidence but is false. Re-measure cross-lingual TNR on these; a head holding 0.90 on easy negatives may collapse on near-miss ones, separating real entailment-checking from lexical-overlap luck
- **Balance native negatives per language** - harvest or synthesise hallucinations in the native eval languages so non-EN macro stops being imbalance-dominated and becomes a fair readout rather than a tiny-class average
- **Run Round 4 (joint-premise NLI)** - the pre-registered SummaC aggregation above; kill-gate first (multi-chunk support density on a 300-row supported sample), then re-score gold v3 eval with the top-3 joint premise and evaluate stacked with the Round 3 EN/non-EN split. This is a new signal (R4-H1), addressing the "max-over-chunks mis-grades multi-sentence support" failure - the one form of new feature the next bullet calls for
- **New cross-lingual signals, not re-routing** - R1-H1/H5 show the existing cascade and lexical signals do not separate the non-EN slice; a lift needs a new feature (e.g. a cross-lingual entailment model that actually fires), not a different combination of the current ones
- **Refuted, do not revisit** - native cascade AUC as a cross-lingual readout (R1-H1, thin negative class); `nli_ent` as the cross-lingual rejection mechanism (R1-H2, 0.523); head reweighting to close the non-EN gap (R1-H4, -0.005); class / language over-under-sampling in training (Round 2, macro flat 0.810 -> 0.811); language-aware escalation into the cascade (R1-H5, 0.800 < 0.809)

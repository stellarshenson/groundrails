# Grounding the RAG assistant - model-based experiments

The full grounding investigation for the a production RAG assistant: can a non-LLM grounder catch hallucination in real answers, which signal does it best, and how should it be used. The arc ran in four research phases - an adversarial probe (where deterministic lexical won), building a verified gold from production traffic, a signal comparison on that gold (where lexical flips and the cross-encoder wins), and a score-stacking meta-classifier that beats any single signal without fine-tuning - followed by deployment consolidation (two cross-encoders, single-engine OpenVINO int8) and two hypothesis rounds (H9-H11, one adoption: the reranker-first cascade; H12-H14, two adoptions: the pre-filter cosine gate and the early-exit reranker, taking the warm claim to 662 ms at slightly better quality). This is the model-based counterpart to the deterministic lexical track in `lexical-grounding-experiments.md` / `lexical-grounding-sota.md` (this repo); both run on the same private gold and reach comparable macro-F1 from opposite mechanisms. Final design in `semantic-grounding-sota.md`.

> Gold/meta figures are the 2,752-record run (organic-majority). Probe-phase figures are stable (size-independent).

## Situational overview

The same lexical NOT_FOUND rule that wins on the small adversarial probe collapses on the verified gold - real supported claims restate the documentation in new words and score as "not found". The lever that survives on paraphrased answers is the cross-encoder reranker, which scores claim-vs-evidence relevance directly; stacking the model scores then beats any single one.

- **Probe set (true data)** - 25 adversarial bait questions through the dev assistant, grounded against the exact chunks it retrieved; 33 factual claims, 6 gold hallucinations. The agent refused 16/25 on its own (strong refusal discipline); only q25 carried a confident fabrication
- **Verified gold** - real prod traffic, dual-judge (Haiku + Sonnet) agreed labels; `{claim, source_text, label, lang, user_id, trace_id}`, English-dominant retrieved-doc evidence per claim (~57 KB). Grew 375 → 856 → 1,260 → 1,686 → 2,631 → 2,752 (organic expansion); several conclusions changed with size
- **Few independent contexts** - claims sharing a trace's evidence are correlated, so the effective sample is smaller than the record count
- **The flip** - on the probe set lexical NOT_FOUND is the most sensitive cheap detector (5/6, ~22-26% false-flag); on the gold it inverts (~85% false-flag) because supported paraphrases share no wording, and the cross-encoder becomes the signal

## Executive summary

A cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) is the best single grounding signal on the verified gold; a logistic over the six model scores plus a lexical contradiction flag beats it on macro-F1 and cuts both error types. No model is fine-tuned - only a decision hyperplane is fit on top.

**Research at a glance** - the full sweep across the four phases (detail in the sections below; the original per-experiment reports are archived under `../@archive/`):

| Phase | Experiment / hypothesis | Dataset | Key result | Conclusion |
|---|---|---|---|---|
| A probe | Adversarial probe capture | 25 bait Q, 33 claims | 16/25 refused, 1 confident hallucination (q25), 6 gold hall | strong refusal discipline; assertion-vs-disclaimer is the real signal (raw NOT_FOUND overstates ~15:1) |
| A probe | Lexical grounding vs no-grounding | probe 33 | 0% → 83% recall (5/6), 22% false-flag, ~55 ms/answer | **lexical works** as a cheap NOT_FOUND/CONTRADICTED gate |
| A probe | Lexical-only vs lexical+semantic | probe 33 | lexical 5/6, 26% ff, 28 ms; +semantic 4/6, 30% ff, 647 ms | **semantic dropped** - over-confirms + ~23-40x latency |
| A probe | Bayesian calibrator + NLI layer | probe 33 | calibrator 1-2/6; NLI raw 3/6, verdict 4/6 (44% ff) | **dropped** - 6 negatives cannot calibrate; NLI over-flags |
| A probe | Feedback-loop prototype (1 revise) | probe 3 answers | 6/6 gold hallucinations fixed | **loop, not gate** - revise/retract beats blocking |
| A probe | Synthetic graded benchmark dataset | 6 base → 24 variants | 0/20/40/60% ungrounded levels | scaffold; real prod groundedness is bimodal, not graded |
| B data | Verified gold (the production trace store, dual-judge) | 375 → 2,752 | dual-agreed gold; rate 37% → 29% as organic grows | the labelled set the probe lacked |
| B data | Test-user concentration | gold 2,752 | one QA cohort = 79% of hallucinations | filter by `user_id`; organic rate ~10% |
| C signal | Lexical match-type on the gold | gold 2,752 | ~85% false-flag, AUC ~0.5 on paraphrases | **refuted here** - inverts the probe-set win |
| C signal | Bi-encoder cosine (e5, bge-m3, mmBERT) | gold 2,752 | AUC 0.53-0.73; over-confirms in-domain | **weak** - topical similarity ≠ grounding; bge-m3 best at 0.73 |
| C signal | NLI cross-encoder (mDeBERTa-mnli-xnli) | gold 2,752 | AUC 0.81; raw gate over-flags | **kept as a feature** - strong ranker, bad raw gate |
| C signal | Cross-encoder reranker (bge-reranker-v2-m3) | gold 856→2,752 | AUC 0.82 → 0.84, best single signal | **kept** - the relevance scorer is the lever |
| C signal | gte-multilingual-reranker | gold | custom-kernel CUDA crash | **dropped** - replaced by standard-arch models |
| D meta | Score-stack: logistic vs GBM over 6 scores | gold 2,752 | OOF AUC 0.91, macro-F1 0.82 | **linear ships** - GBM ~ ties within noise |
| D meta | Lexical numeric/entity contradiction flag | gold 2,752 | weight +0.18 (small) on this set | **kept** - cheap, catches spec edits the rerankers miss |
| E deploy | 2-cross-encoder consolidation ablation | gold 2,752 | macro-F1 0.796 with {reranker, NLI} only | **ships** - within ~1 fold-std of the full 0.814 |
| E deploy | NLI replacement search (MiniLM-L6/L12, XLM-R) | gold 2,752 | stacks 0.758-0.765 vs 0.796 | **rejected** - mDeBERTa uniquely strong for grounding |
| E deploy | mDeBERTa int8: ORT vs OpenVINO SmoothQuant | gold 2,752 | ORT parity 0.29-0.75; OV SQ 0.984, stack 0.795 | **SmoothQuant ships** - 318 MB, -0.001 macro-F1 |
| F round 1 | H9 - NLI contradiction/neutral channels | int8 pairs 111,800 | +0.004/+0.005 macro-F1 | **rejected** - inside fold noise; gold hallucinations are omissions, not contradictions |
| F round 1 | H10 - aggregation beyond max-over-chunks | int8 pairs 111,800 | -0.005..+0.005 macro-F1 | **rejected** - max already extracts the distribution's signal |
| F round 1 | H11 - reranker-first confidence cascade | int8 pairs + latency bench | 61% NLI skips at macro-F1 0.795; warm mean -28% | **adopted** - thresholds only, no training |
| G round 2 | H12 - pre-filter cosine gate (stage 0) | int8 pairs + cosine cache | 22% of claims skip both cross-encoders, FP 245/FN 216 vs 248/217 | **adopted** - strictly fewer errors, zero added compute |
| G round 2 | H13 - rank-ordered early-exit reranker | latency bench n=150 | mean 4.8/8 pairs scored; verdicts exactly equal | **adopted** - verdict-invariant by construction |
| G round 2 | H14 - fused-evidence single-forward cross-encoders | fused cache 2,752 x 2 x 2 | macro-F1 0.714-0.784 (-0.012..-0.081) | **rejected** - max-over-chunks is load-bearing |

- **Best single signal** - `bge-reranker-v2-m3`, **AUC 0.841** (out-of-fold), macro-F1 0.757; decisively above every bi-encoder (~0.53-0.73) and the raw lexical rule (~0.5)
- **Best model** - a decision hyperplane over the six per-model scores + a lexical contradiction flag: **out-of-fold AUC 0.913, macro-F1 0.824** (vs 0.757 best single, 0.417 majority baseline), no fine-tuning
- **Reduces both errors** - at the macro-F1-optimal threshold it cuts total errors 533 → 408 (-23%): false-negatives 295 → 160 (false-positives ~flat 238 → 248)
- **GBM ≈ the logistic** - depth-2 boosting reaches a near-tie OOF AUC (0.913 gbm vs 0.903 logreg, within ±0.012 std); the linear hyperplane ships for simplicity and a calibrated probability
- **Driving metric is macro-F1** - the 1,966/786 imbalance makes accuracy misleading (majority predictor: macro-F1 0.417, hallucination-F1 0.000); FP and FN counts are the operational target
- **Residual** - 160 missed hallucinations (102 carry a number/spec) and 248 false-positives sit in the overlap region a fine-tuned cross-encoder would target - the deferred lever

## Results summary

Per-signal separation on the 2,752 gold (out-of-fold AUC, higher = better; the cross-encoder reranker is the lever, the bi-encoders and lexical rule are weak).

| signal | kind | OOF AUC |
|---|---|---|
| bge-reranker-v2-m3 | cross-encoder rerank | **0.841** |
| mDeBERTa-v3-mnli-xnli | NLI cross-encoder | 0.806 |
| bge-m3 | bi-encoder | 0.730 |
| e5-small | bi-encoder | 0.635 |
| e5-large | bi-encoder | 0.621 |
| mmBERT-base | bi-encoder | 0.529 |
| lexical match-type | rule | ~0.50 |

Best single signal vs the score-stack vs the majority baseline (2,752 gold, at each model's macro-F1-optimal threshold). The stack lifts macro-F1 and cuts total errors, almost all of the gain in recall (missed hallucinations).

| metric | best single (bge-reranker) | meta-classifier (stack) | majority baseline |
|---|---|---|---|
| OOF AUC | 0.841 | **0.913** | 0.50 |
| macro-F1 | 0.757 | **0.824** | 0.417 |
| FP - supported flagged | 238 | 248 | 0 |
| FN - hallucination missed | 295 | **160** | 786 |
| total errors (of 2,752) | 533 | **408** | 786 |

## Experiment record - results across data growth

The conclusion held as the gold grew. macro-F1 climbed 0.817 → 0.824 and the score-stack beat the best single signal at every size - the stability across a 3.2x data increase (and a base-rate shift) is the best evidence the result is real, not a snapshot artefact.

| gold | records | hall rate | best-single AUC (bge-reranker) | meta AUC (OOF) | meta macro-F1 | errors cut vs best single |
|---|---|---|---|---|---|---|
| doubled | 856 | 36% | 0.86 | 0.91 | 0.817 | - |
| tripled | 1,260 | 37% | 0.857 | 0.907 | 0.821 | 269 → 210 (-22%) |
| organic-majority | 2,752 | 29% | 0.841 | 0.913 | 0.824 | 533 → 408 (-23%) |

- **macro-F1 stable** - 0.817 → 0.821 → 0.824 across the growth and a 37% → 29% base-rate shift
- **stack > best single at every size** - the combination is not an artefact of one snapshot
- **single-signal AUC drifts down slightly** (0.86 → 0.841) as more diverse organic paraphrases enter - harder data, but the stack compensates and macro-F1 still rises
- **base-rate caveat** - the 2,752 set is organic-majority (the QA-test cohort filtered down in influence), so its 29% rate and raw FP/FN counts are not directly comparable to the earlier test-heavy sets; macro-F1 and AUC are the size- and rate-robust metrics
- **not re-fit at 1,686 / 2,631** - intermediate sizes were data milestones; the model was re-scored and re-fit at 856, 1,260, and 2,752, which bracket the growth

## Phase A - adversarial probe (where lexical won)

Before a labelled gold existed, the grounder was tested end-to-end on real production answers to bait questions, judged per claim by a local `claude -p`.

- **Refusal discipline** - 16/25 hallucination-bait questions correctly refused (non-existent products VMS V400 / T700, fabricated versions, network details, part numbers); only q25 carried a confident fabrication (an invented T300 "100 m" cable limit + a fabricated cable type), caught by NOT_FOUND + CONTRADICTED
- **Lexical lifts capture from 0% to 83%** - the no-grounding baseline ships everything and catches 0/6; lexical NOT_FOUND/CONTRADICTED catches 5/6 at claim level and 3/3 at answer level, ~55 ms/answer
- **Semantic is worse here** - lexical+semantic catches 4/6 at 30% false-flag and ~647 ms/claim (~23-40x the latency); the bi-encoder over-confirms because in-domain topical similarity reads as support at the stock 0.6 threshold, and the Bayesian calibrator (bambi/PyMC) cannot recover the misses with only 6 negatives
- **NLI over-flags** - raw NLI catches 3/6 at 63% false-flag, grounder-with-NLI 4/6 at 44%; entailment is strict sentence-level while the assistant's answers are supported by tables/procedures
- **The 15:1 trap** - raw NOT_FOUND count overstates hallucination ~15:1 because the agent's own refusal sentences score NOT_FOUND; the real signal is assertion-vs-disclaimer, and the pipeline must gate on that
- **Real prod prose is ~94% grounded** - groundedness is bimodal (full answers ~0% ungrounded, short fallbacks ~100%), not a smooth spectrum, so a graded benchmark of 24 variants at 0/20/40/60% ungrounded had to be constructed by controlled injection
- **Verdict** - ship lexical-only as a cheap NOT_FOUND/CONTRADICTED gate with assertion-vs-disclaimer filtering; gate semantic behind calibration; the prerequisite for any learned/embedding layer is a larger balanced labelled set

## Phase B - building the verified gold

The probe's "6 negatives" ceiling forced a real labelled set. A golden dataset was built from production traffic, recipe in `../dataset/2026-06-01-how-to-build-golden-dataset.md`.

- **Pipeline** - scout prod traces → recover evidence from tool/rag spans → extract claims from the raw answer → lexical pre-pass → Haiku primary judge → tightened-prompt cleanup → Sonnet as the stronger second judge → keep only dual-agreed labels
- **Growth** - 375 → 856 → 1,260 → 1,686 → 2,631 → 2,752 records; several conclusions changed with size (a depth-2 GBT that won on 375 overfits later)
- **Test-user concentration** - one QA/test account drives 81% of the hallucinations at a 56% rate; organic users sit at ~10%, so the later batches exclude the test accounts and the gold carries `user_id` / `trace_id` to filter cleanly
- **Storage** - parquet (zstd) at ~2 MB, off Git LFS; the shared evidence blobs deduplicate columnar

## Phase C - signal comparison on the verified gold

With a real labelled set, the probe ranking inverts.

- **Lexical flips to failure** - NOT_FOUND fires on supported restatements (~85% false-flag, AUC ~0.5); the signal that word overlap measured on the probe (fabrications share no wording) is swamped by paraphrase on real answers
- **Bi-encoders over-confirm** - e5-small/large modest (AUC ~0.62-0.64), bge-m3 better (0.73), mmBERT near chance (~0.53); topical similarity to in-domain evidence is not support
- **NLI ranks well, gates badly** - mDeBERTa AUC 0.81 but raw verdict is 100% recall at 88% false-flag (over-flags)
- **Cross-encoder reranker wins** - bge-reranker-v2-m3 AUC 0.82 → 0.84 across the gold's growth, the best single signal, because it scores claim-against-evidence relevance directly
- **Run as a GPU notebook** - `notebooks/01-kj-grounding-model-comparison.ipynb` over six multilingual models, per-model subprocess isolation; gte-multilingual-reranker dropped after a custom-kernel CUDA crash

## Phase D - the score-stacking meta-classifier

Per the no-fine-tuning constraint: learn a verdict over the per-model scores.

- **Features** - 6 model scores (bge-reranker, mDeBERTa-NLI, bge-m3, e5-large, e5-small, mmBERT) + `lexical_fired` + numeric/entity `contradiction` flag
- **Heads raced** - a logistic with `StandardScaler` + L2, a depth-2 GBM, and each raw single signal; 5-fold stratified out-of-fold predictions remove model-selection optimism
- **Result** - OOF AUC 0.913 (gbm) / 0.903 (logreg), macro-F1 0.824 vs 0.757 best single, 0.417 baseline
- **Learned weights** (+ = supported) - bge-m3 +1.34, bge-reranker +1.14, lexical-fired +0.91, NLI +0.87; e5/mmBERT small negative (de-noising), contradiction +0.18
- **GBM ties** - no nonlinear interaction gain, so the linear hyperplane ships

## Methodology

Score every model independently on the gold, then learn a verdict over the scores; no learner sees the fold it scores.

- **Evidence chunking** - recursive 1,100-char chunks, 200 overlap, max 50; claim scored against each chunk, max taken (best-chunk relevance)
- **Bi-encoder** - cosine of claim and chunk embeddings (e5 `query:`/`passage:` prefixes, bge-m3 CLS, mmBERT mean); max over chunks
- **Cross-encoder rerank** - the reranker's single relevance logit per claim×chunk, sigmoid, max over chunks - models claim-vs-evidence directly, not via a shared embedding space
- **NLI** - entailment probability of (chunk → claim) per pair, max over chunks
- **Lexical features** - match-type (`lexical_fired`) and a numeric/entity `contradiction` flag (`find_mismatches`)
- **Verdict head** - logistic with `StandardScaler` + L2; GBM and raw scores raced against it
- **Metric** - macro-F1 headline, FP and FN counts at the operating point, AUC for ranking quality
- **Cross-validation** - 5-fold stratified out-of-fold; single-signal AUCs are pretrained so honest as-is, the meta-classifier re-estimated OOF
- **Per-language AUC** - where n ≥ 20 (en, nb-NO)

## Setup

- **Data** - `data/processed/golden_grounding_evidence_verified.parquet`; per-model scores cached to `data/interim/model_scores/*.npy`, index-aligned
- **Models** - e5-small, e5-large, bge-m3, mmBERT-base (bi-encoders); bge-reranker-v2-m3 (rerank), mDeBERTa-v3-base-mnli-xnli (NLI)
- **Hardware** - RTX 5090 (index 1 under `CUDA_DEVICE_ORDER=PCI_BUS_ID`), torch 2.12 + cu130; one model per subprocess so a CUDA assert cannot poison the others
- **HF quirks** - `HF_HUB_OFFLINE=1` (metadata stalls), `HF_HUB_DISABLE_XET=1` (Xet segfaults on large files), vault token; mmBERT/ModernBERT need `reference_compile=False`; gte dropped (CUDA kernel crash) - all captured in the `my-gpu` skill
- **Commands** - `python -m grounding_models <model>` (score one, isolated), `python -m grounding_ensemble` (fit + report)

## Deployment shape - two-stage verifier in a feedback loop

The grounder is a signal in a self-correction loop, not a hard gate (`../@archive/docs/grounding-feedback-loop.md`); a prototype revise fixed 6/6 gold hallucinations.

- **Stage A - fast pre-filter on every claim** - lexical (sub-ms) or the model stack produces candidate flags, disclaimers excluded
- **Stage B - precision gate on the flagged subset only** - one batched LLM-judge (or the score-stack probability) confirms unsupported vs paraphrase, so the agent is not told to rewrite claims that are fine
- **Loop** - on a confirmed flag the agent re-examines, re-retrieves, and revises or retracts (1-2 iterations); a confident fabrication has no supporting passage so it must be retracted - the correct outcome
- **Why a loop** - lexical/raw model gates false-flag paraphrases; letting the agent prove a flagged claim is supported (by citing the passage) avoids silently dropping genuine claims

## Model class: logistic vs GBM vs raw single signal

The decisive factor is the feature set (cross-encoder scores), not the fitting method.

- **Score-stacking logistic** - OOF AUC 0.903, macro-F1 ~0.82; the reranker + bge-m3 + NLI scores carry the signal
- **Gradient-boosted trees** (depth-2) - OOF AUC 0.913, a near-tie within std; no material nonlinear gain, so the linear model ships
- **Best single signal** - bge-reranker alone, macro-F1 0.757; the +0.07 comes from bge-m3 and NLI adding orthogonal evidence
- **Honesty on small data** - everything re-estimated out-of-fold; the full-set 0.82 carries selection optimism the OOF removes

## What we tried

- **Kept** - the bge-reranker cross-encoder (the lever), the mDeBERTa NLI score (strong ranker), bge-m3 embeddings (best bi-encoder), a logistic decision hyperplane, the lexical contradiction/fired flags; lexical-only as the cheap probe-stage gate; the two-stage verifier loop
- **Dropped / refuted** - raw lexical as a gate on paraphrases (over-flags on the gold), bi-encoder cosine as a verdict (over-confirms), raw NLI threshold (over-flags), the Bayesian calibrator on the probe set (too few negatives), gte-multilingual-reranker (CUDA crash), deep trees (no gain), e5/mmBERT as standalone signals (near-chance)

## Lessons learned

- **The signal flips with the distribution** - lexical NOT_FOUND wins on the probe and on the lexical track's omission-type cross-lingual task, but fails on paraphrased restatements where the cross-encoder wins; never carry a small-sample conclusion across distributions
- **Cross-encoder >> bi-encoder for grounding** - scoring claim-against-evidence beats a shared embedding space; topical similarity is not support
- **Raw model thresholds are mis-calibrated for the domain** - bi-encoders over-confirm, NLI over-flags; both are strong rankers (good AUC) but bad gates until a learned threshold/combination calibrates them
- **Stacking beats the best single signal cheaply** - a hyperplane over the scores adds +0.05 macro-F1 and cuts both error types with no fine-tuning
- **Macro-F1, not accuracy** - the imbalance makes accuracy read ~0.6 while macro-F1 is 0.39 for the majority predictor; the operational target is FP + FN counts
- **Model class barely matters here** - GBM ties the logistic; the gain was the cross-encoder features
- **Assertion-vs-disclaimer first** - raw NOT_FOUND overstates hallucination ~15:1; gate on assertions, not refusal sentences
- **A loop beats a gate** - paraphrase false-flags make a hard gate drop ~1-in-5 genuine claims; a revise loop lets the agent defend or fix
- **The labelled set is the unlock** - 6 probe negatives could calibrate nothing; the verified gold is what let any learned layer work, and growing it (and excluding the test cohort) is what makes the numbers trustworthy
- **GPU hygiene is load-bearing** - per-model subprocess isolation, HF offline, Xet disabled, dropping the gte custom-kernel model were prerequisites to clean scores
- **Complementary to the lexical track** - the engineered translate-then-recall lexical pipeline (sibling `lexical-grounding-sota.md`) reaches comparable macro-F1 from a torch-free CPU mechanism; different operating points, not a contradiction

## Deployment - single-engine OpenVINO int8

The deployable grounder runs on one runtime - OpenVINO int8 for all three models (bge-m3 pre-filter, bge-reranker, mDeBERTa NLI). Full quantization record in `deberta-v3-quantization-experiments.md`; end-to-end pipeline in `notebooks/03-kj-openvino-grounder-pipeline.ipynb`.

- **mDeBERTa int8 was the blocker, solved by SmoothQuant** - stock dynamic int8 broke (parity 0.35); NNCF SmoothQuant (alpha 0.7) reaches full-gold parity 0.9841 and stack macro-F1 0.795 (vs fp32 0.796) at 318 MB
- **ONNX-Runtime cannot quantize the NLI** - even a forked `onnx-neural-compressor` (three crash bugs fixed) stays faithless (parity 0.61-0.62); the single engine is OpenVINO, not ORT
- **All three int8 IRs hold parity** - bge-reranker 0.9976, bge-m3 0.9941, mDeBERTa-NLI 0.9863; sizes 571 / 570 / 318 MB (~1.46 GB total), push-ready under `models/ov/`
- **Top-k pre-filter improves quality** - on an 800-record subset, k=8 macro-F1 0.822 beats all-chunks 0.807; pruning noisy chunks before the cross-encoders helps the verdict, not just latency
- **Cache the source-chunk embeddings - the dominant latency lever** - the typical claim carries ~50 evidence chunks (median 50), and cold the pre-filter re-embeds all of them per claim (CPU ~4.2 s/claim median at k=8 under the LATENCY hint; `notebooks/04-kj-grounder-latency.ipynb`). Caching the chunk vectors (embed each unique chunk once, or reuse the RAG retriever's vectors) so the pre-filter only embeds the claim cuts the typical claim to **~1.2 s (median, 3.6×; p90 1.5 s)**. It also restores top-k as a real lever - warm k=8 is 5.0× faster than all-chunks (cold only 1.4×). Cosine over the ~50 chunks is a brute-force numpy dot-product; no FAISS/ANN needed at that scale
- **OpenVINO `LATENCY` hint - 2.1× free** - `compile_ir` had defaulted to `THROUGHPUT` (multi-stream, right only for the batch/offline path); for inline single-claim serving `LATENCY` is **2.1× faster** (cold 6365 → 3048 ms/claim at k=8 on a 64-thread CPU), measured in `scripts/bench_mechanical_levers.py`. Now the default - no quality cost
- **`max_length` cap does not help** - chunks run ~300 tokens median / 418 p95 and (claim, chunk) pairs ~331 / ~590 p95, so the 512 cap already truncates ~6.5% of pairs; capping to 256 saves only ~17% and clips the median pair. `MAX_LEN` stays 512. Length-bucketing the chunks before batching is the only padding win there (order-invariant, scores unchanged)
- **Whole-answer batching still open** - the serving helpers score per-claim (one padded forward per claim per model); batching an answer's claims × top-k into one forward would amortise overhead - the next mechanical lever to build
- **Portability** - x86-64 Intel/AMD native (AVX2 / AVX-512-VNNI); ARM/Graviton via the OpenVINO ARM plugin, less mature - validate on target

## Hypothesis round 1 - H9/H10/H11

Three mechanism-targeting hypotheses against the deployed int8 stack, all evaluated on the CPU OpenVINO int8 engines (a full per-pair score cache over the 111,800 gold (claim, chunk) pairs, `data/interim/model_scores/pairs/full_pairs.npz`); full ladder and final benchmark in `reports/grounding_hypotheses.md`, driver in `experiments/grounding-semantic/grounding_hypotheses.py`. The int8 pair cache reproduces the baseline (macro-F1 0.797 vs 0.795/0.796 reference), so the round is apples-to-apples.

- **H9 contradiction channel - rejected** - the NLI 3-way softmax's contradiction/neutral channels (free, same forward pass) add only +0.004/+0.005 macro-F1, inside fold noise (±0.014); on this gold the unsupported claims are omissions/fabrications, not contradictions, so the channel has little to bite on (consistent with the bounded-scope finding)
- **H10 aggregation beyond max - rejected** - distributional features of the per-pair scores (top-2 mean, logsumexp, count above threshold, top1−top2 margin) move macro-F1 −0.005 to +0.005; max-over-chunks already extracts what the score distribution knows, evidence redundancy is not a usable signal here
- **H11 reranker-first cascade - adopted** - the reranker always runs first; its max score `s` against the band [0.01, 0.66] decides the rest: `s <= 0.01` → flag as hallucination (reranker is sure nothing supports the claim, NLI skipped), `s >= 0.66` → pass as supported (NLI skipped), in-band → run the NLI and take the stack verdict as before. 61% of claims skip the NLI at macro-F1 0.795 (−0.002, inside noise); on the dense band sweep the adopted band is strictly no-worse than baseline on both error counts (FP 243 vs 244, FN 217 unchanged). Measured warm latency at k=8 (LATENCY hint, n=150): mean **1,184 → 857 ms/claim (−28%)**, median 759 ms (−34%), p90 −14% - hard claims still pay both models, easy claims pay one. Serving helper `cascade_scores` in `grounding_openvino.py`; benchmark `scripts/bench_grounder_cascade.py`
- **The band is empirical, not learned** - no model is fit against it: candidate edges are the quantiles of the reranker score distribution (19 points, 5th-95th percentile); every (a, b) pair is simulated on the out-of-fold scores into a (skip-rate, macro-F1) frontier (`reports/grounding_hypotheses.md`), and [0.01, 0.66] is the point with maximal skip at zero measurable quality loss (one step wider: ~70-80% skip at −0.004 to −0.013). Same calibration class as the operating threshold - chosen on frozen-model scores, in raw reranker-score space, so re-sweep if the reranker, quantization, or evidence distribution changes
- **Two band options from the same frontier** - the sweep also yields an FP-constrained alternative: **quality-neutral [0.01, 0.66]** (adopted default - skip 60%, macro-F1 0.797, FP 243 / FN 217, strictly no-worse than baseline 244/217) and **low-false-flag [0.05, 0.32]** (skip 84%, FP 192 = −52, at FN 279 = +62, macro-F1 0.783) for deployments where a false flag costs more than a miss; the FP cut is bought with recall, not quality elsewhere
- **Principle held** - all three used only softmax channels, aggregation statistics and thresholds over frozen models; no weights touched, nothing fit beyond the decision hyperplane

## Hypothesis round 2 - H12/H13/H14

Three more mechanism-targeting hypotheses against the cascade-adopted grounder (warm mean 869 ms on the round-2 sample), one allowed to restructure the architecture. Same caches, same OOF protocol, all measurement on the deployed CPU int8 engines; ladder and final benchmark in `reports/grounding_hypotheses.md`, driver `experiments/grounding-semantic/grounding_hypotheses.py`, bench `scripts/bench_grounder_round2.py`.

- **H12 pre-filter cosine gate - adopted** - the pre-filter already computes every claim-chunk cosine to rank the top-k; the max was discarded. It becomes a stage-0 gate: `cos <= 0.493` → flag, `cos >= 0.739` → pass, in-between → cascade as before. **22% of claims resolve at embed cost (~39 ms) with strictly fewer errors than the cascade alone** (FP 245 / FN 216 vs 248 / 217, macro-F1 0.797 vs 0.795). The gate works despite the bi-encoder's weak AUC (0.730) because it does not need pure tails - it only needs to agree with the cascade verdict on the claims it absorbs. Zero added compute; two thresholds fit OOF like the band
- **H13 rank-ordered early-exit reranker - adopted** - `rerank_max` scored all k=8 pairs in one padded batch, but the cascade only needs to know whether the max crosses the pass edge (0.66). Scoring pairs best-cosine-first in progressive batches (1, 1, 2, 4) and stopping at the first crossing is **verdict-invariant by construction** (verified exact on the bench sample, 150/150; unscored pairs cannot change a final pass verdict, and never-crossing claims score every pair). Mean pairs scored drop 8 → 4.8 (exit rate 49%); the int8 forward is near-linear in batch rows (122 ms batch-1 vs 95 ms/pair batch-8), so the exits keep most of what they save
- **H14 fused-evidence single forward - rejected** - assembling ONE evidence context per claim (top-2 chunk concat, or salience-packed sentences ranked by the same bi-encoder) and running ONE forward per cross-encoder would have cut 16 forwards to 2 (~211 ms/claim measured). Quality collapses: macro-F1 0.714-0.784 across all six stack/variant configs (-0.012 to -0.081), and the fused NLI correlates only 0.54 with the per-chunk max. **Max-over-chunks is load-bearing** - each chunk in isolation poses one focused entailment/relevance question; packing evidence dilutes it. Together with H10 this brackets the mechanism: nothing beyond the max helps, and the max cannot be approximated in one forward
- **Round-2 net effect (adopted = gate + cascade + exit)** - OOF macro-F1 0.795 → 0.797 with -3 FP / -1 FN; warm mean **869 → 662 ms (-24%)**, median 782 → 593 ms, vs the original always-both pipeline **-45% mean** at equal quality; p90 +4% (never-exit claims pay the progressive-schedule worst case, 876 vs 761 ms of reranker - the spend-where-uncertain shape sharpened). Footprint unchanged; everything added is two thresholds and a batch schedule
- **Principle held** - the gate and exit are thresholds and execution ordering over already-computed frozen-model scores; H14's contexts were input assembly only. No weights touched in either round

## Conclusions

- **Ship the consolidated two-cross-encoder semantic stack** - the full six-model + lexical hyperplane is the research maximum (macro-F1 0.824, AUC 0.913), but an ablation shows a logistic over just the two cross-encoders (bge-reranker + mDeBERTa-NLI) holds macro-F1 0.796 - within one CV fold-std of the full - while dropping four bi-encoders and the lexical layer; that minimal semantic pipeline is the deployable design (`semantic-grounding-sota.md`), served as single-engine OpenVINO int8, no fine-tuning
- **The cross-encoder reranker is the core signal** - bge-reranker-v2-m3 alone is AUC 0.841; bge-m3 and NLI scores are the additive lift
- **Use it as both a soft flag and a hard gate** - report the full curve, pick a low-false-flag point for pre-display blocking and a high-recall point for the feedback-loop re-check
- **Run it in the two-stage loop** - fast pre-filter on every claim, precision gate on the flagged subset, agent revises or retracts
- **The ceiling is the overlap region** - ~100 hallucinations and ~100 supported sit where neither the reranker nor the stack separates them; closing it needs a fine-tuned cross-encoder
- **Bounded scope** - the win is on paraphrased omission/fabrication hallucinations in retrieved-doc evidence; present-but-contradicted negatives need the contradiction signal the sibling lexical track studies

## Next steps

- **Hold confirmed on the 2,752-record organic-majority gold** - re-scored and re-fit: macro-F1 0.824 / AUC 0.913 held as the organic base rate dropped to 29%; per-language AUC en 0.92, es 0.80, fr 0.78, nb 0.66
- **Fine-tune a cross-encoder** - the remaining lever once the stack plateaus; target the overlap residual (the 102 numeric/spec misses)
- **Operating-point calibration** - per-language thresholds where counts allow; the shipped Bayesian calibrator now that a separated signal exists
- **Top-k pre-filter + chunk-embedding cache measured (done)** - the single-engine OpenVINO pipeline holds (k=8 macro-F1 0.822 on the subset); caching the source-chunk embeddings cuts the typical claim from ~4.2 s to ~1.2 s (median, 3.6×, LATENCY hint) and makes k=8 5.0× faster than all-chunks. Next: load the published HF int8 IRs (`stellars/*-openvino-int8`) into the deployment and wire the pre-filter to the RAG retriever's chunk vectors
- **Push the int8 IRs to HuggingFace** - the `models/ov/` IRs are push-ready (IR + config + tokenizer); publish under the org for reuse and CI
- **Confirm top-k macro-F1 on full gold (done)** - the full adopted serving path (gate + cascade + early-exit, deployed calibration frozen) runs end-to-end over the 2,752 gold at **macro-F1 0.789** (within fold noise of the 0.797 OOF simulation; error mix shifts toward recall - FP 328/FN 172 - because serving maxes are over the top-8 pre-filtered chunks, not all chunks), warm mean 585 ms / median 258 ms per claim (`scripts/run_grounder_full.py`). Re-fit thresholds on serving-derived scores before fixing the deployment operating point
- **CPU serving levers measured (done)** - `LATENCY` compile hint is 2.1× over `THROUGHPUT` for inline serving (now the default); `max_length` has no headroom below 512 (refuted); length-bucketing kept. The reranker-first cascade (−28%), the pre-filter cosine gate and the early-exit reranker (round 2, cumulative **−45% warm mean vs always-both, 662 ms/claim**) are measured and adopted. Next mechanical lever: an answer-level batched scorer. The remaining bigger win is GPU fp16 (~0.15-0.4 s/claim)

# Grounding the RAG assistant - model-based experiments

**Canonical Experiments Document**

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
| H round 3 (pre-reg) | R3-H1..H7 - tiny fine-tuned faithfulness checker (persona fanout) | gold 2,752 + 111,800-pair cache | pre-registered predictions/bars; governed by the R3-H1 zero-shot-AUC gate | **pending** - see Hypothesis round 3 |
| I round 4 (pre-reg) | R4-H1..H7 - knowledge-free reasoning models as the grounding head (SAN / SYNTH / Needle transfer) | gold 2,752 + 111,800-pair cache | pre-registered; governed by the R4-H1 tokenizer gate then the R4-H3 mechanism test | **pending** - see Hypothesis round 4 |

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

## Hypothesis round 3 - fine-tuned tiny faithfulness checker (pre-registered)

The deferred lever from Next steps ("fine-tune a cross-encoder ... target the overlap residual") opened as a full phase. Six independent hypothesiser lenses (architecture, distillation, quantization, data, methodology, integration) were run as a fanout; this section is the deduped **pre-registration** - predictions and pass/fail bars fixed BEFORE any run. No verdicts yet.

The thesis, unanimous across the fanout: **replace the mDeBERTa-NLI (and possibly the reranker) with a purpose-built faithfulness checker on a small standard-attention encoder**. Two facts make it a compound win the current architecture structurally cannot get:

- **Head shape** - gold hallucinations are omissions / unsupported-additions, not contradictions (H9), so the incumbent borrows a 3-label MNLI/XNLI entailment logit as a faithfulness proxy (AUC 0.806); a 2-way {supported, unsupported} head trained on faithfulness targets that boundary directly
- **Quantization** - mDeBERTa's disentangled attention gives int8 no speedup (footprint-only, 318 MB); a standard encoder (RoBERTa / MiniLM / ModernBERT) int8-accelerates (1.34x compiled, 1.0 ms bs=1), so moving off DeBERTa cuts params AND unlocks real int8 latency

This is NOT the refuted E-deploy "NLI replacement search" (MiniLM-L6/L12, XLM-R → stacks 0.758-0.765): that swapped in zero-shot 3-label NLI checkpoints. R3 swaps both the head shape (2-way) and the training data (in-domain faithfulness) - the falsifiable difference. That prior result is exactly what caps optimism, so R3-H1 measures whether a faithfulness-TRAINED checker transfers before any fine-tune is spent.

**"One model total" is a fiction.** Max-over-chunks is load-bearing (H10/H14) and warm-cached ranking needs a cheap bi-encoder to cut ~50 chunks to top-8; the honest floor is TWO small models - a bi-encoder pre-filter/gate (kept, 38 ms warm, doubles as the 22%-resolving stage-0 gate) plus ONE small cross-encoder checker replacing the two 889 MB cross-encoders.

### Methodology - the R3 ship-contract and honest-split protocol

"Tiniest that does the job" is made falsifiable by a pre-registered contract; a tiny fine-tuned model can memorise the gold, so the splits are group- and language-disjoint.

- **Ship-contract (conjunction of three)** - macro-F1 ≥ 0.775 (deployed 0.789 − one fold-std 0.014, non-inferiority) AND warm-CPU mean ≤ 200 ms AND int8 footprint ≤ 100 MB; among passers ship the fewest-params; declared relaxation ≤ 150 MB if the smallest passer is base-size (~110M)
- **Honest split** - the gold is **619 sources / 636 traces, not 5,857 rows**; the ship readout is leave-one-source-out (GroupKFold on `group_id`) plus a held-out-language slice, with a pre-registered train − holdout gap ≤ 0.05 (memorisation gate) and held-out-language macro-F1 ≥ 0.637 (no cross-lingual regression vs the frozen cascade)
- **Nested honest re-fit** - a student emits a new score scale, so its verdict head and operating threshold are fit only on folds it never scores (the joint track caught a +0.023 phantom from in-sample threshold selection); harness-validated by reproducing the frozen cascade at 0.789 ± 0.014 before any student number is believed
- **Two-sided error-count parity** - macro-F1 is FP↔FN-invariant, so a passer must also hold FP ≤ cascade + 2σ AND FN ≤ cascade + 2σ at a recall-matched threshold (reference two-CE OOF point FP 266 / FN 203, σ ≈ √n)

### Pre-registration - predictions and bars fixed before the run

| ID | mechanism | prediction | PASS bar (two-sided) | first kill-gate |
|---|---|---|---|---|
| R3-H1 | zero-shot faithfulness-checker upper anchor (MiniCheck-RoBERTa-Large 355M, standard attention, max-over-top-8) | OOF AUC 0.82-0.86 vs incumbent NLI 0.806 | AUC ≥ 0.806 unlocks the ladder; ≥ 0.86 → collapse both | AUC &lt; 0.79 on the cached gold pairs → whole round killed |
| R3-H2 | 2-way faithfulness head replaces the NLI, keep bge-reranker (ModernBERT / RoBERTa-base) | stack macro-F1 0.79-0.80 at ~150 MB, NLI-stage int8 &gt; 1.2x | macro-F1 ≥ 0.789 AND (≤ 200 MB OR NLI-stage ≤ 450 ms) | R3-H1 AUC ≥ 0.806 (faithfulness transfers at all) |
| R3-H3 | collapse both cross-encoders into ONE base checker, keep bge-m3 pre-filter (3→2) | warm ~150-230 ms, ~700 MB, macro-F1 0.78-0.80 | warm mean ≤ 300 ms AND macro-F1 ≥ 0.786 | single-head standalone AUC ≥ 0.86 (else fall back to R3-H2) |
| R3-H4 | cascade self-distillation into a ≤33M student (free per-pair teacher labels; ladder to ~22M) | macro-F1 0.76-0.79 at 22-33M, ≤ 100 MB, warm ≤ 120 ms | macro-F1 ≥ 0.785 AND ≤ 100 MB AND ≤ 120 ms | student-teacher per-pair Spearman ≥ 0.85 on the 111,800-pair cache |
| R3-H5 | addition/omission hard-negative fine-tune (not contradiction) + atomic-claim decomposition | slip-through ≥ 30% headroom, then macro-F1 ≥ 0.789 | macro-F1 ≥ 0.789 AND FN ≤ 203 AND FP ≤ 266 | Haiku-mint 200 addition/omission + 200 contradiction controls; slip-through ≥ 30% AND &gt; controls |
| R3-H6 | cross-lingual near-miss weight fine-tune (multilingual student) | non-EN macro-F1 0.637 → ≥ 0.70, EN held | non-EN ≥ 0.70 AND aggregate ≥ 0.789 AND near-miss TNR ≥ 0.85 | cascade over-accepts near-miss negatives (&lt; 85% rejected) → real headroom |
| R3-H7 | escalation economics - widen the switch_on band / cascade-of-cascades, current stack as hard floor | system macro-F1 +0.005..+0.02 at flat blended latency | macro-F1 ≥ current + 0.005 AND blended latency ≤ current | checker flip-rate on the lexical-clear band ≥ 2% |

### The hypotheses

**R3-H1 zero-shot faithfulness-checker upper anchor** - because MiniCheck-RoBERTa-Large is an encoder-only 2-way head trained on decomposed-doc faithfulness (the exact job the incumbent's entailment-index-0 proxies), scoring it frozen over the cached top-8 pairs and re-fitting the logistic will read OOF AUC 0.82-0.86 vs the incumbent NLI's 0.806 - the gate that governs the whole round

- Mechanism - the incumbent optimises MNLI, not faithfulness; a faithfulness objective should separate omission / fabrication from support on the same frozen chunks. RoBERTa-large is standard attention (int8-accelerable), but at 355M it is the CEILING PROBE, not the deploy target
- Probe / artifacts - one OOF pass, zero training; `data/interim/model_scores/pairs/full_pairs.npz` + `lytang/MiniCheck-RoBERTa-Large`; reuses the single-signal harness
- Verdict space - Killed-at-gate (whole round) if AUC &lt; 0.79; unlocks R3-H2 if 0.806-0.86; unlocks the R3-H3 collapse if ≥ 0.86; Ships as a 355 MB reference only if ≥ 0.876

**R3-H2 2-way faithfulness head replaces the NLI** - because the 3-label softmax wastes two logits this gold never exercises (H9 +0.004) and DeBERTa int8 has no speedup, a 2-way {supported, unsupported} head on a base standard encoder (ModernBERT / RoBERTa-base) fine-tuned on faithfulness data, dropped in for ONLY the NLI stage while keeping bge-reranker, holds stack macro-F1 ≥ 0.789 at ~150 MB with a real int8 latency cut

- Mechanism - all head capacity goes to support-vs-not; ModernBERT-base's 8k context also removes the ~6.5% pair truncation at 512; DeBERTa-v3-base-EN is the same-family fallback if the standard encoder underfits
- Probe / artifacts - one FT run on MiniCheck synthetic, score gold top-8 OOF, re-fit the logistic; ModernBERT `reference_compile=False` (repo HF quirk); NNCF int8 IR via `scripts/build_ov_grounder.py`
- Verdict space - Ships ≥ 0.789 at ≤ 200 MB with a latency cut; Kept 0.783-0.789 (footprint+speed at quality parity); Refuted &lt; 0.78 (confirms mDeBERTa uniquely strong even vs a trained faithfulness head)

**R3-H3 collapse both cross-encoders into one base checker** - because a faithfulness objective subsumes the reranker's relevance (AUC 0.841) and the NLI's entailment (0.806), one base checker over the bge-m3 top-8 could match the two-CE stack's OOF AUC 0.876, collapsing the 889 MB cross-encoder pair to one ~150 MB model at warm ~150-230 ms while the cheap pre-filter/gate stays

- Mechanism - the pre-filter is load-bearing only as the free 22%-resolving stage-0 gate (38 ms warm), not as latency, so it stays; the expensive half is the two cross-encoders. Stretch: shrink bge-m3 to a MiniLM-class embedder → a two-tiny-model ~200-300 MB floor, gated on top-8 recall parity ≥ 98%
- Probe / artifacts - reuse R3-H1's frozen scores as a STANDALONE signal (reranker+NLI dropped), OOF AUC vs 0.876; then a ranking-recall pass for the pre-filter-shrink stretch
- Verdict space - Ships (one checker + pre-filter) if standalone AUC ≥ 0.876; Kept (drop NLI only → R3-H2) if 0.841-0.876; Dropped if &lt; 0.841

**R3-H4 cascade self-distillation into a tiny student** - because the frozen cascade emits a calibrated per-pair grounded-probability for any (claim, chunk) with zero human labels, distilling a ≤33M standard-encoder student to regress it (then max-over-chunks + one threshold) holds macro-F1 0.76-0.79 while collapsing 3 models → 1 and cutting footprint to ≤ 100 MB int8 and warm to ≤ 120 ms

- Mechanism - the teacher's grounded-prob already fuses reranker + NLI (logistic weights +1.14 / +0.87), so one student regressing that per-pair target learns both. Scale the pool from the 111,800-pair cache to a class-balanced trace pool (prod is ~94% grounded / bimodal) to shrink the retention gap; ladder down to ~22M (MiniLM-L6 / DeBERTa-v3-xsmall) to locate the floor, adding TinyBERT attention-transfer or soft-label KD on the κ-0.50 overlap residual if output-KD saturates
- Probe / artifacts - distil on the EXISTING cache (no cascade re-run), measure student-teacher Spearman + OOF macro-F1 first; `full_pairs.npz` + `data/processed/golden_grounding_evidence_verified.parquet`
- Verdict space - Ships ≥ 0.785 at ≤ 100 MB / ≤ 120 ms; Kept 0.757-0.785 (size/speed-favourable point); Killed-at-gate if Spearman &lt; 0.85; Null if OOF &lt; 0.757 (below reranker-alone → distillation added nothing)

**R3-H5 addition/omission hard-negative data program** - because the gold's hallucinations are unsupported additions / silent-evidence omissions (even the 102 numeric misses are evidence-absent specs, not contradicted values) and the incumbent is generic MNLI, minting in-domain negatives by perturbing SUPPORTED gold with evidence-absent additions / dropped-support omissions (single-atom where claims decompose) and fine-tuning the small head holds macro-F1 ≥ 0.789 in the overlap residual a frozen model cannot separate

- Mechanism - the residual (FN ~160-203, FP ~248-266) is where a relevance reranker and generic entailment over-confirm topically-close-but-unsupported claims; a head trained on the "is this specific assertion present?" boundary targets it. Atomic-claim decomposition makes maximally-hard negatives (all-but-one atom still supported) - the claim-side lever, orthogonal to the refuted evidence-side aggregation (H10/H14)
- Probe / artifacts - Haiku-mint 200 addition/omission + 200 contradiction controls, run the deployed `semantic_ov.SemanticCascade.score`, confirm slip-through ≥ 30% AND &gt; controls (the probe IS the kill-gate, minutes, zero training); perturb `golden_v3` supported rows
- Verdict space - Ships if a standalone head ≥ 0.789; Kept if NLI-replacement ≥ 0.789; Killed-at-gate if slip-through &lt; 30% or below the contradiction control (no headroom on this axis)

**R3-H6 cross-lingual near-miss weight fine-tune** - because R1-H2 was refuted for the frozen joint GATE (nli_ent cross-lingual feature near-chance, 0.523) not for a fine-tuned head, fine-tuning a multilingual student's WEIGHTS on bilingual near-miss negatives creates cross-lingual faithfulness features the frozen mDeBERTa lacks, lifting non-EN macro-F1 above the cascade's AUC-0.584 floor

- Mechanism - the cascade fails non-EN because its frozen features do not separate the slice (R1-H1 0.584; R1-H4 head-reweighting −0.005 = a signal gap, not a weighting gap); training weights on in-language near-miss negatives checks support across the language boundary directly - the "new signal, not re-routing" the joint track named as still open
- Probe / artifacts - mint ~150 near-miss cross-lingual negatives from the base sentences, run the cascade; real headroom = the cascade over-accepts them (&lt; 85% rejected vs the easy-TNR 0.904); `golden_v3_synth_aug.parquet`, GroupKFold on `group_id` (translations stay in-fold)
- Verdict space - Ships if non-EN ≥ 0.70 and near-miss TNR ≥ 0.85 with no EN regression; Null if non-EN &lt; 0.66 (matches refuted R1-H4); Killed-at-gate if the cascade already rejects near-miss ≥ 85%

**R3-H7 escalation economics** - because a tiny checker cuts the escalated per-claim cost ~585 → ~150 ms, the lexical→semantic escalation band in `joint.py` can widen to route more of the lexical residual through the checker at flat blended latency (catching paraphrased fabrications the lexical tier scores as grounded), predicting system macro-F1 +0.005..+0.02; alternatively the current stack is kept as a hard floor behind the tiny checker (cascade-of-cascades)

- Mechanism - escalation is gated today only because the cascade is expensive; the band is a cost valve, not a quality-optimal boundary. At ~150 ms/escalation, checking ~2x more claims is still cheaper than today, and the new claims are the omission-type ones the stack's non-entailment signal catches. The cascade-of-cascades variant guarantees macro-F1 ≥ 0.789 as a floor (the stack overrides the checker only on its low-confidence band) at +1 model
- Probe / artifacts - run the R3 checker on the lexical-CLEAR band, count verdict flips vs gold; re-sweep `[a, b]` on the OOF frontier; depends on a shipped R3 checker
- Verdict space - Ships if macro-F1 ≥ +0.01 at flat latency; Kept if +0.005; Null if flip-rate &lt; 2% (band already clean); cascade-of-cascades Kept if the confident band ≥ 85% at warm ≤ 350 ms

### Sequencing

The round runs as a gated ladder, not a parallel sweep. **R3-H1 is the one probe to run this week** - a single frozen OOF pass over the existing 111,800-pair cache, zero training - and it routes everything: AUC &lt; 0.79 stops the program (mDeBERTa stays); 0.806-0.86 → R3-H2 (swap the NLI only); ≥ 0.86 → R3-H3 (collapse both cross-encoders). R3-H4 (distil to the tiniest) and R3-H5 (in-domain data) are the size-and-accuracy pushes that an R3-H1 pass unlocks; R3-H6 (cross-lingual) and R3-H7 (pipeline economics) are the follow-ons. Every model number above is a prediction pending measurement under the R3 ship-contract and honest-split protocol.

## Hypothesis round 4 - knowledge-free reasoning models as the grounding head (pre-registered)

R3 asked what to train. R4 asks a prior question: **what must the model know?** Grounding supplies its own evidence in the prompt, so the head needs no parametric world knowledge - only the ability to route between a claim and a passage. That is exactly the regime where attention-only and knowledge-free models are measured strongest, and exactly the axis (parametric recall) they give up. This section is the **pre-registration** - predictions and pass/fail bars fixed BEFORE any run. No verdicts yet.

The source evidence is a controlled study, not a vendor claim ([digest](../../references/papers/[paper%20digest]%20simple%20attention%20networks%20controlled%20study.md), [arXiv 2607.18363](https://arxiv.org/abs/2607.18363)):

- **FFN is ~2/3 of non-embedding parameters** and deleting it in place costs 0.470 nats - but reallocating that budget into attention depth costs only **0.006 nats (0.27% of loss)** at matched parameters, reproducible to 1e-4 across clean seed pairs
- **The residual gap localises to parametric recall, not to capability** - three independent measurements (token regions, task types, zero-shot benchmarks) agree; by 105B tokens the attention-only arm leads on every answer region and the deficit survives only on low-context query tokens (+0.038)
- **The closest public analogue of grounding favours attention-only, and the margin grows** - Sciq (the answer sits in a provided support passage) goes 0.725 → 0.742 for the SAN across 31B → 105B while the FFN arm regresses 0.702 → 0.661, seed ranges non-overlapping. The knowledge benchmark (Lambada) moves the other way at every budget
- **The applied product exists** - `cactus-compute/needle`, from the same authors: 26M params / 14 MB, attention-only encoder (12 layers, no FFN) + 8-layer decoder, d=512, 8-head/4-KV GQA, RoPE, ZCRMSNorm, tied embeddings, **8,192-token BPE vocabulary**; it outperforms FunctionGemma-270M, Qwen-0.6B, Granite-350M and LFM2.5-350M on single-shot function calling, post-trained in **45 minutes on 2B tokens**, minimum **120 examples per class**. MIT licence
- **Off-the-shelf knowledge-free checkpoints exist**, so the 16 x TPU-v6e / 27-hour pretraining is not on the critical path - `PleIAs/Monad` (56.7M, 64 layers, ~8k vocab, decoder-only, Apache 2.0, 200B tokens of SYNTH) and `PleIAs/Baguettotron` (0.3B). Monad scores MMLU ~30% / GSM8K 8% **by design** - it trades world knowledge for reasoning traces, which is the property under test, not a defect

**Why this is not R3 again.** R3 keeps the discriminative cross-encoder paradigm and changes the head shape and training data on a standard 100-355M encoder. R4 changes the **model class**: pretrained-to-reason-without-knowledge, 26-300M, potentially generative. The two are complements, not rivals - R3-H4 (self-distillation into a standard encoder) is the control R4-H5 must beat to justify a new architecture.

**Three ways this round is wrong, stated in advance.** The evidence is next-token loss on 6-87M decoder LMs, not a fine-tuned pairwise entailment head, so transfer is an assumption to test. Monad is **strictly monolingual English** while the gold is cross-lingual and the non-EN slice is already the cascade's weak point (AUC 0.584). And the paper is explicit that a task's storage-versus-routing identity is **relative to the training distribution**, so the SYNTH-trained split may not survive a move to technical prose.

### Methodology - carried-forward bars and the R4 additions

The R3 ship-contract and honest-split protocol carry forward unchanged; R4 adds one gate and one scope rule.

- **Naive baseline (unchanged)** - majority predictor, macro-F1 **0.417**, AUC **0.50**, hallucination-F1 0.000; every R4 number is reported as a delta against it AND against the deployed cascade (end-to-end macro-F1 **0.789**, two-CE OOF **0.796**, warm mean 585 ms, 318 MB int8 NLI stage)
- **Ship-contract (carried from R3)** - macro-F1 ≥ **0.775** AND warm-CPU mean ≤ **200 ms** AND int8 footprint ≤ **100 MB**, two-sided at FP ≤ 266+2σ and FN ≤ 203+2σ; among passers ship the fewest params
- **Honest split (carried from R3)** - leave-one-source-out GroupKFold on `group_id` (619 sources / 636 traces, not 5,857 rows), held-out-language slice, train − holdout gap ≤ 0.05, nested re-fit of any new score scale on folds it never scores
- **NEW - representation gate** - a knowledge-free model with an 8k SYNTH-trained vocabulary may not be able to *see* the evidence; tokenizer fertility and context overflow are measured before any inference (R4-H1)
- **NEW - scope rule** - Monad-class candidates are pre-declared **English-slice only**; any aggregate number must be reported blended with the incumbent serving the non-EN slice, never as if the head were multilingual

### Pre-registration - predictions and bars fixed before the run

| ID | mechanism | prediction | PASS bar (two-sided) | first kill-gate |
|---|---|---|---|---|
| R4-H1 | representation feasibility - 8k SYNTH BPE over technical prose | EN fertility 1.6-2.2x mDeBERTa, non-EN ≥ 3x; EN context overflow &lt; 10% | EN fertility ≤ 2.5x AND EN overflow ≤ 20% | fertility &gt; 2.5x OR overflow &gt; 20% → round killed pre-inference |
| R4-H2 | zero-shot knowledge-free reasoner as grounding judge (Monad 56.7M, Baguettotron 0.3B, frozen) | Monad OOF AUC 0.62-0.74, Baguettotron 0.70-0.80, both &lt; incumbent 0.806 | Baguettotron ≥ 0.75 unlocks round; Monad ≥ 0.70 unlocks the tiny track | best of the two &lt; 0.65 → no zero-shot transfer; fall through to R4-H4/H5 only |
| R4-H3 | **the mechanism test** - context-grounded vs knowledge-dependent split | AUC deficit vs mDeBERTa ≤ 0.03 on self-contained pairs, ≥ 0.10 on knowledge-dependent | (deficit_knowledge − deficit_selfcontained) ≥ 0.07 | needs R4-H2 scores only - free re-analysis, no new compute |
| R4-H4 | post-train Monad as a 2-way grounding head (Needle recipe, cascade labels) | EN macro-F1 0.74-0.80 at ~57 MB int8, warm ≤ 150 ms | macro-F1 ≥ 0.775 AND ≤ 100 MB AND warm ≤ 200 ms AND FP/FN parity | R4-H2 Monad AUC ≥ 0.70 |
| R4-H5 | attention-only cross-encoder trained from scratch (SAN, QK-norm, domain BPE) | 10-25M params, ≤ 30 MB int8, macro-F1 0.72-0.78 | macro-F1 ≥ 0.757 (reranker-alone) at ≤ 30 MB | must beat R3-H4 distillation at matched footprint, else Null |
| R4-H6 | reasoning trace vs single-token verdict (serving shape) | trace +0.02-0.04 macro-F1 at 8-20x CPU latency | trace ships only if ≥ +0.02 macro-F1 AND ≤ 200 ms warm | trace latency measured on 50 pairs before any quality run |
| R4-H7 | English-only scope gate (EN → knowledge-free head, non-EN → incumbent) | blended macro-F1 ≥ 0.789 at 40-60% lower blended latency | blended macro-F1 ≥ 0.789 AND blended warm ≤ 350 ms | depends on a shipped R4-H4 or R4-H5 head |

### The hypotheses

**R4-H1 representation feasibility gate** - because Monad's 8,192-token BPE was trained exclusively on English synthetic SYNTH text while the gold is multilingual technical prose, measuring tokenizer fertility and context overflow before any inference will show EN fertility 1.6-2.2x mDeBERTa's 250k multilingual vocabulary and EN context overflow below 10%, while non-EN fertility exceeds 3x

- Lever - tokenizer only; model weights untouched, no inference run. Held fixed: the gold pairs and the top-8 pre-filter
- Mechanism - an 8k vocabulary trained on one distribution shatters out-of-distribution words into many subword pieces; past ~2.5x fertility the evidence no longer fits the context and the model literally cannot see what it must ground against. This is a representation failure, not a capability failure, and it is cheap to detect
- Probe / artifacts - encode `data/processed/golden_grounding_evidence_verified.parquet` claims + their top-8 chunks with `PleIAs/Monad` and `mDeBERTa-v3-base` tokenizers; report tokens/word by language and the overflow fraction. Minutes, CPU only, zero GPU
- Verdict space - Killed-at-gate (whole round) if EN fertility &gt; 2.5x or EN overflow &gt; 20%; Kept if EN passes and non-EN fails (activates the R4 scope rule); Ships-forward if both pass

**R4-H2 zero-shot knowledge-free reasoner as a grounding judge** - because SYNTH-pretrained models are trained on reasoning traces rather than fact memorisation, scoring `PleIAs/Monad` (56.7M) and `PleIAs/Baguettotron` (0.3B) frozen over the cached top-8 gold pairs will read OOF AUC 0.62-0.74 and 0.70-0.80 respectively - below the incumbent NLI's 0.806, but far enough above chance to establish that grounding transfers without world knowledge

- Lever - the judge model; the pre-filter, the top-8 chunk set and the max-over-chunks aggregation are held fixed at the deployed configuration
- Mechanism - the verdict is read as the length-normalised log-probability of a `supported` vs `unsupported` continuation given (claim, chunk), so no head is trained and no gradient is taken; this is the same frozen-anchor design as R3-H1 and its number is directly comparable
- Probe / artifacts - one OOF pass over `data/interim/model_scores/pairs/full_pairs.npz` restricted to the EN slice per the scope rule; re-fit only the logistic on folds it never scores
- Verdict space - Killed-at-gate if the better model reads &lt; 0.65 (no zero-shot transfer; only the trained tracks R4-H4/H5 survive); Kept 0.65-0.75; unlocks the tiny track if Monad ≥ 0.70; Ships as a reference anchor only if either ≥ 0.806

**R4-H3 the mechanism test - context-grounded versus knowledge-dependent** - because the source paper localises the entire attention-only deficit to parametric recall and measures attention-only models AHEAD on passage-supplied answers (Sciq 0.742 vs 0.661 at 105B), partitioning the gold into self-contained pairs (adjudicable from the supplied chunk alone) and knowledge-dependent pairs (needing a fact absent from the chunk) will show the knowledge-free models' AUC deficit versus mDeBERTa at ≤ 0.03 on the first and ≥ 0.10 on the second

- Lever - the evaluation partition only; identical scores, identical models, re-analysed. Nothing is trained and no new inference runs
- Mechanism - this is the round's crux and its cheapest experiment. If the deficit is uniform across the partition, the "grounding needs no knowledge" premise is false FOR THIS TASK and R4's rationale collapses to "it is a small model", which R3 already pursues with better-suited architectures. If the deficit splits as predicted, the mechanism transfers from next-token loss to pairwise grounding and the round is justified on evidence rather than analogy
- Probe / artifacts - partition by whether the human rationale in the verified gold cites only the supplied chunk; adjudicate ~200 pairs per side to keep the partition honest, blind to model scores. Reuses R4-H2's scores at zero marginal compute
- Verdict space - Confirmed if (deficit_knowledge − deficit_selfcontained) ≥ 0.07; Refuted if &lt; 0.03 (premise dead - record it and stop the round here rather than spending training budget); Null in between

**R4-H4 post-train Monad as a 2-way grounding head** - because the frozen cascade emits a calibrated per-pair grounded-probability for any (claim, chunk) at zero labelling cost and Needle demonstrates that 45 minutes over 2B tokens specialises a knowledge-free base into a narrow structured task, post-training Monad on the 111,800-pair cache will hold EN macro-F1 0.74-0.80 at ~57 MB int8 and warm ≤ 150 ms

- Lever - the post-training corpus and objective; the base checkpoint, the pre-filter and the top-8 aggregation are fixed
- Mechanism - Needle's result is that a knowledge-free base plus a short task-specific post-train beats general models 5-25x larger on the narrow task, and the minimum-data rule (120 examples per class) is exceeded by three orders of magnitude here. The 8k vocabulary and absent FFN are what make 56.7M affordable at 64 layers
- Probe / artifacts - teacher labels from `full_pairs.npz`; post-train per the needle recipe (`needle finetune`, JSONL with query/tools/answers adapted to claim/evidence/verdict); NNCF int8 IR via `scripts/build_ov_grounder.py`; evaluate under the carried-forward R3 honest-split protocol
- Verdict space - Ships if macro-F1 ≥ 0.775 at ≤ 100 MB and ≤ 200 ms with FP/FN parity; Kept 0.757-0.775 (a size/latency point below the contract); Dropped &lt; 0.757 (below reranker-alone - the knowledge-free base added nothing a standard encoder does not)

**R4-H5 attention-only cross-encoder trained from scratch** - because the FFN holds two thirds of non-embedding parameters and its deletion costs 0.006 nats at matched parameters, an encoder-only SAN cross-encoder (12-20 attention-only layers, d=512, QK-normalization, a domain BPE of 8-16k) trained directly on the cascade-labelled pair cache will reach macro-F1 0.72-0.78 at 10-25M parameters and ≤ 30 MB int8 - a size class below anything R3 can reach

- Lever - the architecture; the training data, labels and split protocol are identical to R4-H4 so the two are directly comparable
- Mechanism - the paper's trainability finding is specific and load-bearing: QK-normalization, not the FFN and not residual gating, is what keeps deep attention-only stacks trainable, so it is a precondition of the build rather than a tuning option. Pairwise grounding never queries weight-stored facts, so the one measured weakness is not exercised
- Probe / artifacts - build on the needle reference implementation (MIT); train a domain BPE on the gold corpus; train on `full_pairs.npz` cascade targets; compare against R3-H4's distilled standard encoder at MATCHED footprint - that comparison, not the absolute number, is what justifies a new architecture
- Verdict space - Ships if macro-F1 ≥ 0.757 at ≤ 30 MB; Kept if it beats R3-H4 at matched footprint; Null if R3-H4 matches or beats it (distillation into a standard encoder is simpler and already in flight); Dropped if training does not converge without the FFN at this data scale

**R4-H6 reasoning trace versus single-token verdict** - because Monad's distinguishing feature is that it emits an intermediary reasoning trace and CPU decode cost scales linearly with emitted tokens, generating a trace before the verdict will add +0.02-0.04 macro-F1 while costing 8-20x the latency of a single-token verdict, failing the 200 ms bar

- Lever - the decode length; model, prompt and data fixed
- Mechanism - a trace is only worth its cost if the grounding decision needs multi-step composition. Most gold hallucinations are omissions (H9), which a single comparison resolves; if the trace helps, it should help disproportionately on the multi-hop subset, which is the diagnostic to record alongside the aggregate
- Probe / artifacts - latency first on 50 pairs (minutes) to establish the multiplier, then quality only if the multiplier leaves headroom under 200 ms
- Verdict space - Ships the trace only if ≥ +0.02 macro-F1 AND ≤ 200 ms warm; Kept as an explainability-only mode (off the serving path) if it adds quality but breaks latency; Dropped if it adds &lt; 0.01

**R4-H7 English-only scope gate** - because Monad is strictly monolingual English while the non-EN slice is already the cascade's weakest (AUC 0.584, non-EN macro-F1 0.637), routing EN claims to a shipped knowledge-free head and leaving non-EN on the incumbent will hold blended macro-F1 ≥ 0.789 at 40-60% lower blended latency

- Lever - the routing rule; both heads are fixed, already-measured artefacts
- Mechanism - the language detector already runs in the lexical tier, so routing is free. This converts Monad's monolingual limitation from a blocker into a scope decision, and it is the only honest way to report an aggregate number for an English-only head
- Probe / artifacts - blend R4-H4/H5 EN verdicts with the deployed cascade's non-EN verdicts over the 2,752 gold; report blended macro-F1, blended warm latency, and the per-language table
- Verdict space - Ships if blended macro-F1 ≥ 0.789 at ≤ 350 ms; Kept if latency wins at macro-F1 ≥ 0.775; Dropped if the routing boundary itself costs more than it saves

### Sequencing

A gated ladder, cheapest-decisive-first. **R4-H1 is minutes on CPU and can kill the round before a single forward pass** - an 8k SYNTH vocabulary that cannot represent the evidence ends it there. R4-H2 is one frozen scoring pass. **R4-H3 then costs nothing** - it re-analyses R4-H2's scores - and it is the decision point: it tests the round's premise directly rather than inferring it from the source paper's next-token results, and a refutation there should stop the round before any training budget is spent. Only on a confirmed mechanism do R4-H4 (post-train the 56.7M base) and R4-H5 (build the attention-only head) run, and R4-H5's verdict is explicitly relative to R3-H4 at matched footprint. R4-H6 and R4-H7 are serving-shape decisions that presuppose a shipped head. Every number above is a prediction pending measurement under the carried-forward R3 ship-contract and honest-split protocol.

# Grounding the RAG assistant - model-based experiments

**Canonical Experiments Document**

The full grounding investigation for the a production RAG assistant: can a non-LLM grounder catch hallucination in real answers, which signal does it best, and how should it be used. The arc ran in four research phases - an adversarial probe (where deterministic lexical won), building a verified gold from production traffic, a signal comparison on that gold (where lexical flips and the cross-encoder wins), and a score-stacking meta-classifier that beats any single signal without fine-tuning - followed by deployment consolidation (two cross-encoders, single-engine OpenVINO int8) and two hypothesis rounds (H9-H11, one adoption: the reranker-first cascade; H12-H14, two adoptions: the pre-filter cosine gate and the early-exit reranker, taking the warm claim to 662 ms at slightly better quality). This is the model-based counterpart to the deterministic lexical track in `lexical-grounding-experiments.md` / `lexical-grounding-sota.md` (this repo); both run on the same private gold and reach comparable macro-F1 from opposite mechanisms. Final design in `semantic-grounding-sota.md`.

> Gold/meta figures are the 2,752-record run (organic-majority). Probe-phase figures are stable (size-independent).

## Intent

**One small fast model that decides whether a claim is supported by its sources, replacing the three-model cascade - and it must beat every comparable public model, decisively, on every corpus we can measure.** That is the target this log exists to reach, and every hypothesis is judged against it.

**The win condition, as of round 8** - a single model **under 400M parameters** that is strictly above `KRLabsOrg/lettucedect-v2-mmbert-base` (307M, MIT) on all three corpora at once, by a margin that is not arguable:

| corpus | bar to beat | decisive margin | status |
|---|---|---|---|
| private gold, 159 held-out traces | 0.7095 | ≥ 0.76 | **held** - our distilled 307M student reads 0.8479 |
| RAGTruth EN, 1,200 responses | 0.7039 | ≥ 0.75 | open - our cascade reads 0.6432 |
| RAGTruth non-EN, 7 languages parallel | 0.6095 | ≥ 0.66 | open - our cascade reads 0.5626 |

A margin of +0.05 is roughly 20x the measured run-to-run noise (0.0023 across three identical trainings), so it is a real separation rather than a tie dressed up as a win. Beating one corpus is not a result: a domain-specialised model already takes the first and a public-trained model already takes the other two.

- **The deliverable** - a single model, **under 400M parameters** (raised from 350M in round 8, which brings mmBERT-base at 307M and the incumbent at 307M both in-band and makes the comparison size-fair), taking (claim, evidence chunk) → supported or not; it replaces the whole shipped stack, not one stage of it
- **Distillation is the backbone**, fine-tuning the optional second stage, anisotropy removal the third family - see round 8
- **Speed is the reason** - the incumbent runs ~662 ms/claim warm on CPU by loading three models (a ~568M bi-encoder over every chunk, a ~568M cross-encoder over the top-8, a ~278M NLI model). One model over the top-3 chunks is ~3 forwards where the cascade does ~60
- **Parity is the accuracy target, not a beat** - the shipped stack reads macro-F1 0.824 and both the training-recipe and architecture research lanes independently put the sub-350M ceiling at **0.80-0.84**. A distilled student cannot exceed its teacher, so the case for building is speed and simplicity at held quality
- **Wide and shallow, not deep and narrow** - measured at matched parameters and FLOPs, narrow-deep is 5.0x slower eager and gives up a third of achieved throughput (R7-H52). Depth's quality edge in the literature is measured on parametric recall, the one axis a grounder never uses because its evidence arrives in the input
- **Inference and deployment are deliberately deferred** - GPU is fine for now; the torch-free OpenVINO int8 CPU path is a later problem and must not filter candidates today
- **Honest measurement is a precondition, not a formality** - macro-F1 0.824 is measured on our own private gold with our own labels and has never been calibrated against anything public. Until R7-H49 lands it may not be quoted as a comparable number

**Standing rules this log enforces**, each earned by a failure recorded below: every hypothesis is pre-registered here before it runs; every new scorer passes a positive control on trivially separable pairs before it scores anything real; every candidate's declared prompt format is audited against its own vocabulary before inference; and a vendor's published number is a claim until someone else reproduces it.

## Situational overview

The same lexical NOT_FOUND rule that wins on the small adversarial probe collapses on the verified gold - real supported claims restate the documentation in new words and score as "not found". The lever that survives on paraphrased answers is the cross-encoder reranker, which scores claim-vs-evidence relevance directly; stacking the model scores then beats any single one.

- **Probe set (true data)** - 25 adversarial bait questions through the dev assistant, grounded against the exact chunks it retrieved; 33 factual claims, 6 gold hallucinations. The agent refused 16/25 on its own (strong refusal discipline); only q25 carried a confident fabrication
- **Verified gold** - real prod traffic, dual-judge (Haiku + Sonnet) agreed labels; `{claim, source_text, label, lang, user_id, trace_id}`, English-dominant retrieved-doc evidence per claim (~57 KB). Grew 375 → 856 → 1,260 → 1,686 → 2,631 → 2,752 (organic expansion); several conclusions changed with size
- **Few independent contexts** - claims sharing a trace's evidence are correlated, so the effective sample is smaller than the record count
- **The flip** - on the probe set lexical NOT_FOUND is the most sensitive cheap detector (5/6, ~22-26% false-flag); on the gold it inverts (~85% false-flag) because supported paraphrases share no wording, and the cross-encoder becomes the signal

## Executive summary

A cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) is the best single grounding signal on the verified gold; a logistic over the six model scores plus a lexical contradiction flag beats it on macro-F1 and cuts both error types. No model is fine-tuned - only a decision hyperplane is fit on top.

**Hypothesis ids** - one global ascending `H<n>`, never reset; the `R<round>-` prefix is a memory slug, the number is the identity. Rounds 3 and 4 were first written with round-local ids (`R3-H1..H7`, `R4-H1..H7`) and were remapped once, on 2026-07-28, to `R3-H15..H21` and `R4-H22..H29` (`R4-H2b` → `R4-H29`, created last). No verdict, result or prediction changed in that remap; experiment scripts, logs and score artefacts were renamed to match. Round 5 continues at `R5-H30`.

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
| H round 3 (pre-reg) | R3-H15..H21 - tiny fine-tuned faithfulness checker (persona fanout) | gold 2,752 + 111,800-pair cache | pre-registered predictions/bars; governed by the R3-H15 zero-shot-AUC gate | **pending** - see Hypothesis round 3 |
| I round 4 | R4-H22 - representation gate (8k SYNTH vocabulary over our evidence) | golden_v6 EN 4,876 | fertility 1.196x incumbent, context overflow 0.3% | **PASS** - a 30x smaller vocabulary costs only 20% more tokens |
| I round 4 (pre-reg) | R4-H23..H28 - knowledge-free reasoning models as the grounding head (SYNTH / Needle / SAN transfer) | gold 2,752 EN slice + golden_v6 | pre-registered; governed by the R4-H24 method test | **pending** - see Hypothesis round 4 |
| I round 4 | R4-H29 - generative-judge positive control (20 trivially separable pairs) | 20-case control | Baguettotron 12/20, Monad 4/20; 0 false negatives, 6 false positives | **closed** - zero-shot generative judging fails for this model class |
| J round 5 | R5-H30 - the prose residue is unmeasured (adjudication gate) | private prose set, 32 unconfirmed in-scope claims | 26/32 correct refusals (81.2%); true recall 28/34 = 82.4% not 46.7% | **Refuted (the framing)** - the metric was wrong, not the grounder |
| J round 5 (pre-reg) | R5-H31..H39 - prompt and elicitation engineering for small-model judgement (persona fanout) | 20-case control + gold EN slice | pre-registered; re-aimed by R5-H30 | **pending** - see Hypothesis round 5 |
| K round 6 | R6-H47/H46 - vocabulary and budget pre-flight | 3 candidates, CPU only | Monad 0/4 template tokens in vocab; no candidate overflows context | **Kept as a standing gate** / H46 **Null** |
| K round 6 | R6-H40 - reproduce Monad's published MMLU | MMLU 2,000 stratified | 0.255 / 0.226 / 0.143 across three protocols vs published 0.30, chance 0.25 | **Killed-at-gate** - Monad cannot be validated, withdrawn |
| K round 6 | R6-H42 - Pleias-RAG-350M as a judge (harness reproduced the vendor example) | 20-case control | 11/20 (55%), 9 false positives | **Refuted** - zero-shot judging closed on a validated instrument |
| K round 6 | R6-H45 - the quote is the verdict (model quotes, code decides) | 20-case control + gold EN 600 | control 18/20, FP 9→1; gold macro-F1 0.511 vs shipped 0.824 | **Refuted for deployment**, kept as a finding - recall 0.267 |
| K round 6 | R6-H43 - the trained refusal path as a verdict | gold EN 600 | macro-F1 0.476, 240 FP on 300 negatives | **Refuted** |
| K round 6 (pre-reg) | R6-H41 - EOS-as-BOS ablation | 20-case control | anomaly confirmed by R6-H47; ablation unrun | **pending** |
| L round 7 | R7-H51 - regenerate the teacher corpus with text | gold 2,752 → 123,579 pairs | reranker AUC 0.8289 vs 0.841 reference (-0.012); NLI drifts -0.039 | **PASS** - ships as the teacher, reranker channel only |
| L round 7 | R7-H52 - encoder shape at matched parameters (ran before registration) | GPU shape bench, B=3 T=512 | narrow-deep 5.0x slower eager; attention-only +18% FLOPs and slower | **Kept as a measurement** - width wins, SAN closed on latency |
| L round 7 | R7-H50 - capacity ablation, 140M-307M distilled from R7-H51 | 159 held-out TRACES, 717 claims | mmBERT-base 0.8479 &gt; mmBERT-small 0.8269; teacher 0.8619; claim-level split leaked 0.050 and inverted the ordering | **Refuted** - the task IS capacity-limited (spread 0.021) |
| L round 7 | R7-H50 depth probe - mmBERT-base truncated 22L/11L/6L/3L | same held-out traces | 22L 0.8502 @ 9.2 ms → 11L 0.8183 @ 5.38 ms | **in progress** - depth costs 0.032 AUC, buys 1.7x |
| L round 7 | R7-H57 - public-trained verifier on our gold | 159 held-out traces | control AUC 1.0000; our gold 0.7095 vs our 0.8619 | **PARTIAL** - public data does not transfer to us |
| L round 7 | R7-H59 - cross-domain matrix, both models on both corpora | gold + RAGTruth 1,200 | ours 0.8619→0.6432 off-domain; theirs 0.7095↔0.7039 flat | **both directions fail** - our number is domain specialisation |
| L round 7 (pre-reg) | R7-H49 - external calibration on LLM-AggreFact | 11 subsets, stratified | predicts 63-72 vs our private-gold ~82-85 | **blocked** - gated dataset, needs Hub auth |
| L round 7 (pre-reg) | R7-H53 - MICE-style split encoder, document side cached | teacher corpus | 0.298 ms vs 2.707 ms measured; AUC deficit unknown | **pending** - needs R7-H50 |

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

This is NOT the refuted E-deploy "NLI replacement search" (MiniLM-L6/L12, XLM-R → stacks 0.758-0.765): that swapped in zero-shot 3-label NLI checkpoints. R3 swaps both the head shape (2-way) and the training data (in-domain faithfulness) - the falsifiable difference. That prior result is exactly what caps optimism, so R3-H15 measures whether a faithfulness-TRAINED checker transfers before any fine-tune is spent.

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
| R3-H15 | zero-shot faithfulness-checker upper anchor (MiniCheck-RoBERTa-Large 355M, standard attention, max-over-top-8) | OOF AUC 0.82-0.86 vs incumbent NLI 0.806 | AUC ≥ 0.806 unlocks the ladder; ≥ 0.86 → collapse both | AUC &lt; 0.79 on the cached gold pairs → whole round killed |
| R3-H16 | 2-way faithfulness head replaces the NLI, keep bge-reranker (ModernBERT / RoBERTa-base) | stack macro-F1 0.79-0.80 at ~150 MB, NLI-stage int8 &gt; 1.2x | macro-F1 ≥ 0.789 AND (≤ 200 MB OR NLI-stage ≤ 450 ms) | R3-H15 AUC ≥ 0.806 (faithfulness transfers at all) |
| R3-H17 | collapse both cross-encoders into ONE base checker, keep bge-m3 pre-filter (3→2) | warm ~150-230 ms, ~700 MB, macro-F1 0.78-0.80 | warm mean ≤ 300 ms AND macro-F1 ≥ 0.786 | single-head standalone AUC ≥ 0.86 (else fall back to R3-H16) |
| R3-H18 | cascade self-distillation into a ≤33M student (free per-pair teacher labels; ladder to ~22M) | macro-F1 0.76-0.79 at 22-33M, ≤ 100 MB, warm ≤ 120 ms | macro-F1 ≥ 0.785 AND ≤ 100 MB AND ≤ 120 ms | student-teacher per-pair Spearman ≥ 0.85 on the 111,800-pair cache |
| R3-H19 | addition/omission hard-negative fine-tune (not contradiction) + atomic-claim decomposition | slip-through ≥ 30% headroom, then macro-F1 ≥ 0.789 | macro-F1 ≥ 0.789 AND FN ≤ 203 AND FP ≤ 266 | Haiku-mint 200 addition/omission + 200 contradiction controls; slip-through ≥ 30% AND &gt; controls |
| R3-H20 | cross-lingual near-miss weight fine-tune (multilingual student) | non-EN macro-F1 0.637 → ≥ 0.70, EN held | non-EN ≥ 0.70 AND aggregate ≥ 0.789 AND near-miss TNR ≥ 0.85 | cascade over-accepts near-miss negatives (&lt; 85% rejected) → real headroom |
| R3-H21 | escalation economics - widen the switch_on band / cascade-of-cascades, current stack as hard floor | system macro-F1 +0.005..+0.02 at flat blended latency | macro-F1 ≥ current + 0.005 AND blended latency ≤ current | checker flip-rate on the lexical-clear band ≥ 2% |

### The hypotheses

**R3-H15 zero-shot faithfulness-checker upper anchor** - because MiniCheck-RoBERTa-Large is an encoder-only 2-way head trained on decomposed-doc faithfulness (the exact job the incumbent's entailment-index-0 proxies), scoring it frozen over the cached top-8 pairs and re-fitting the logistic will read OOF AUC 0.82-0.86 vs the incumbent NLI's 0.806 - the gate that governs the whole round

- Mechanism - the incumbent optimises MNLI, not faithfulness; a faithfulness objective should separate omission / fabrication from support on the same frozen chunks. RoBERTa-large is standard attention (int8-accelerable), but at 355M it is the CEILING PROBE, not the deploy target
- Probe / artifacts - one OOF pass, zero training; `data/interim/model_scores/pairs/full_pairs.npz` + `lytang/MiniCheck-RoBERTa-Large`; reuses the single-signal harness
- Verdict space - Killed-at-gate (whole round) if AUC &lt; 0.79; unlocks R3-H16 if 0.806-0.86; unlocks the R3-H17 collapse if ≥ 0.86; Ships as a 355 MB reference only if ≥ 0.876

**R3-H16 2-way faithfulness head replaces the NLI** - because the 3-label softmax wastes two logits this gold never exercises (H9 +0.004) and DeBERTa int8 has no speedup, a 2-way {supported, unsupported} head on a base standard encoder (ModernBERT / RoBERTa-base) fine-tuned on faithfulness data, dropped in for ONLY the NLI stage while keeping bge-reranker, holds stack macro-F1 ≥ 0.789 at ~150 MB with a real int8 latency cut

- Mechanism - all head capacity goes to support-vs-not; ModernBERT-base's 8k context also removes the ~6.5% pair truncation at 512; DeBERTa-v3-base-EN is the same-family fallback if the standard encoder underfits
- Probe / artifacts - one FT run on MiniCheck synthetic, score gold top-8 OOF, re-fit the logistic; ModernBERT `reference_compile=False` (repo HF quirk); NNCF int8 IR via `scripts/build_ov_grounder.py`
- Verdict space - Ships ≥ 0.789 at ≤ 200 MB with a latency cut; Kept 0.783-0.789 (footprint+speed at quality parity); Refuted &lt; 0.78 (confirms mDeBERTa uniquely strong even vs a trained faithfulness head)

**R3-H17 collapse both cross-encoders into one base checker** - because a faithfulness objective subsumes the reranker's relevance (AUC 0.841) and the NLI's entailment (0.806), one base checker over the bge-m3 top-8 could match the two-CE stack's OOF AUC 0.876, collapsing the 889 MB cross-encoder pair to one ~150 MB model at warm ~150-230 ms while the cheap pre-filter/gate stays

- Mechanism - the pre-filter is load-bearing only as the free 22%-resolving stage-0 gate (38 ms warm), not as latency, so it stays; the expensive half is the two cross-encoders. Stretch: shrink bge-m3 to a MiniLM-class embedder → a two-tiny-model ~200-300 MB floor, gated on top-8 recall parity ≥ 98%
- Probe / artifacts - reuse R3-H15's frozen scores as a STANDALONE signal (reranker+NLI dropped), OOF AUC vs 0.876; then a ranking-recall pass for the pre-filter-shrink stretch
- Verdict space - Ships (one checker + pre-filter) if standalone AUC ≥ 0.876; Kept (drop NLI only → R3-H16) if 0.841-0.876; Dropped if &lt; 0.841

**R3-H18 cascade self-distillation into a tiny student** - because the frozen cascade emits a calibrated per-pair grounded-probability for any (claim, chunk) with zero human labels, distilling a ≤33M standard-encoder student to regress it (then max-over-chunks + one threshold) holds macro-F1 0.76-0.79 while collapsing 3 models → 1 and cutting footprint to ≤ 100 MB int8 and warm to ≤ 120 ms

- Mechanism - the teacher's grounded-prob already fuses reranker + NLI (logistic weights +1.14 / +0.87), so one student regressing that per-pair target learns both. Scale the pool from the 111,800-pair cache to a class-balanced trace pool (prod is ~94% grounded / bimodal) to shrink the retention gap; ladder down to ~22M (MiniLM-L6 / DeBERTa-v3-xsmall) to locate the floor, adding TinyBERT attention-transfer or soft-label KD on the κ-0.50 overlap residual if output-KD saturates
- Probe / artifacts - distil on the EXISTING cache (no cascade re-run), measure student-teacher Spearman + OOF macro-F1 first; `full_pairs.npz` + `data/processed/golden_grounding_evidence_verified.parquet`
- Verdict space - Ships ≥ 0.785 at ≤ 100 MB / ≤ 120 ms; Kept 0.757-0.785 (size/speed-favourable point); Killed-at-gate if Spearman &lt; 0.85; Null if OOF &lt; 0.757 (below reranker-alone → distillation added nothing)

**R3-H19 addition/omission hard-negative data program** - because the gold's hallucinations are unsupported additions / silent-evidence omissions (even the 102 numeric misses are evidence-absent specs, not contradicted values) and the incumbent is generic MNLI, minting in-domain negatives by perturbing SUPPORTED gold with evidence-absent additions / dropped-support omissions (single-atom where claims decompose) and fine-tuning the small head holds macro-F1 ≥ 0.789 in the overlap residual a frozen model cannot separate

- Mechanism - the residual (FN ~160-203, FP ~248-266) is where a relevance reranker and generic entailment over-confirm topically-close-but-unsupported claims; a head trained on the "is this specific assertion present?" boundary targets it. Atomic-claim decomposition makes maximally-hard negatives (all-but-one atom still supported) - the claim-side lever, orthogonal to the refuted evidence-side aggregation (H10/H14)
- Probe / artifacts - Haiku-mint 200 addition/omission + 200 contradiction controls, run the deployed `semantic_ov.SemanticCascade.score`, confirm slip-through ≥ 30% AND &gt; controls (the probe IS the kill-gate, minutes, zero training); perturb `golden_v3` supported rows
- Verdict space - Ships if a standalone head ≥ 0.789; Kept if NLI-replacement ≥ 0.789; Killed-at-gate if slip-through &lt; 30% or below the contradiction control (no headroom on this axis)

**R3-H20 cross-lingual near-miss weight fine-tune** - because R1-H2 was refuted for the frozen joint GATE (nli_ent cross-lingual feature near-chance, 0.523) not for a fine-tuned head, fine-tuning a multilingual student's WEIGHTS on bilingual near-miss negatives creates cross-lingual faithfulness features the frozen mDeBERTa lacks, lifting non-EN macro-F1 above the cascade's AUC-0.584 floor

- Mechanism - the cascade fails non-EN because its frozen features do not separate the slice (R1-H1 0.584; R1-H4 head-reweighting −0.005 = a signal gap, not a weighting gap); training weights on in-language near-miss negatives checks support across the language boundary directly - the "new signal, not re-routing" the joint track named as still open
- Probe / artifacts - mint ~150 near-miss cross-lingual negatives from the base sentences, run the cascade; real headroom = the cascade over-accepts them (&lt; 85% rejected vs the easy-TNR 0.904); `golden_v3_synth_aug.parquet`, GroupKFold on `group_id` (translations stay in-fold)
- Verdict space - Ships if non-EN ≥ 0.70 and near-miss TNR ≥ 0.85 with no EN regression; Null if non-EN &lt; 0.66 (matches refuted R1-H4); Killed-at-gate if the cascade already rejects near-miss ≥ 85%

**R3-H21 escalation economics** - because a tiny checker cuts the escalated per-claim cost ~585 → ~150 ms, the lexical→semantic escalation band in `joint.py` can widen to route more of the lexical residual through the checker at flat blended latency (catching paraphrased fabrications the lexical tier scores as grounded), predicting system macro-F1 +0.005..+0.02; alternatively the current stack is kept as a hard floor behind the tiny checker (cascade-of-cascades)

- Mechanism - escalation is gated today only because the cascade is expensive; the band is a cost valve, not a quality-optimal boundary. At ~150 ms/escalation, checking ~2x more claims is still cheaper than today, and the new claims are the omission-type ones the stack's non-entailment signal catches. The cascade-of-cascades variant guarantees macro-F1 ≥ 0.789 as a floor (the stack overrides the checker only on its low-confidence band) at +1 model
- Probe / artifacts - run the R3 checker on the lexical-CLEAR band, count verdict flips vs gold; re-sweep `[a, b]` on the OOF frontier; depends on a shipped R3 checker
- Verdict space - Ships if macro-F1 ≥ +0.01 at flat latency; Kept if +0.005; Null if flip-rate &lt; 2% (band already clean); cascade-of-cascades Kept if the confident band ≥ 85% at warm ≤ 350 ms

### Sequencing

The round runs as a gated ladder, not a parallel sweep. **R3-H15 is the one probe to run this week** - a single frozen OOF pass over the existing 111,800-pair cache, zero training - and it routes everything: AUC &lt; 0.79 stops the program (mDeBERTa stays); 0.806-0.86 → R3-H16 (swap the NLI only); ≥ 0.86 → R3-H17 (collapse both cross-encoders). R3-H18 (distil to the tiniest) and R3-H19 (in-domain data) are the size-and-accuracy pushes that an R3-H15 pass unlocks; R3-H20 (cross-lingual) and R3-H21 (pipeline economics) are the follow-ons. Every model number above is a prediction pending measurement under the R3 ship-contract and honest-split protocol.

## Hypothesis round 4 - knowledge-free reasoning models as the grounding head (pre-registered)

R3 asked what to train. R4 asks a prior question: **what must the model know?** Grounding supplies its own evidence in the prompt, so the head needs no parametric world knowledge - only the ability to route between a claim and a passage. That is exactly the regime where attention-only and knowledge-free models are measured strongest, and exactly the axis (parametric recall) they give up. This section is the **pre-registration** - predictions and pass/fail bars fixed BEFORE any run. No verdicts yet.

The source evidence is a controlled study, not a vendor claim ([digest](../../references/papers/[paper%20digest]%20simple%20attention%20networks%20controlled%20study.md), [arXiv 2607.18363](https://arxiv.org/abs/2607.18363)):

- **FFN is ~2/3 of non-embedding parameters** and deleting it in place costs 0.470 nats - but reallocating that budget into attention depth costs only **0.006 nats (0.27% of loss)** at matched parameters, reproducible to 1e-4 across clean seed pairs
- **The residual gap localises to parametric recall, not to capability** - three independent measurements (token regions, task types, zero-shot benchmarks) agree; by 105B tokens the attention-only arm leads on every answer region and the deficit survives only on low-context query tokens (+0.038)
- **The closest public analogue of grounding favours attention-only, and the margin grows** - Sciq (the answer sits in a provided support passage) goes 0.725 → 0.742 for the SAN across 31B → 105B while the FFN arm regresses 0.702 → 0.661, seed ranges non-overlapping. The knowledge benchmark (Lambada) moves the other way at every budget
- **The applied product exists** - `cactus-compute/needle`, from the same authors: 26M params / 14 MB, attention-only encoder (12 layers, no FFN) + 8-layer decoder, d=512, 8-head/4-KV GQA, RoPE, ZCRMSNorm, tied embeddings, **8,192-token BPE vocabulary**; it outperforms FunctionGemma-270M, Qwen-0.6B, Granite-350M and LFM2.5-350M on single-shot function calling, post-trained in **45 minutes on 2B tokens**, minimum **120 examples per class**. MIT licence
- **Off-the-shelf knowledge-free checkpoints exist**, so the 16 x TPU-v6e / 27-hour pretraining is not on the critical path - `PleIAs/Monad` (56.7M, 64 layers, ~8k vocab, decoder-only, Apache 2.0, 200B tokens of SYNTH) and `PleIAs/Baguettotron` (0.3B). Monad scores MMLU ~30% / GSM8K 8% **by design** - it trades world knowledge for reasoning traces, which is the property under test, not a defect

**Why this is not R3 again.** R3 keeps the discriminative cross-encoder paradigm and changes the head shape and training data on a standard 100-355M encoder. R4 changes the **model class**: pretrained-to-reason-without-knowledge, 26-300M, potentially generative. The two are complements, not rivals - R3-H18 (self-distillation into a standard encoder) is the control R4-H26 must beat to justify a new architecture.

**Three ways this round is wrong, stated in advance.** The evidence is next-token loss on 6-87M decoder LMs, not a fine-tuned pairwise entailment head, so transfer is an assumption to test. Monad is **strictly monolingual English** while the gold is cross-lingual and the non-EN slice is already the cascade's weak point (AUC 0.584). And the paper is explicit that a task's storage-versus-routing identity is **relative to the training distribution**, so the SYNTH-trained split may not survive a move to technical prose.

### Methodology - carried-forward bars and the R4 additions

The R3 ship-contract and honest-split protocol carry forward unchanged; R4 adds one gate and one scope rule.

- **Naive baseline (unchanged)** - majority predictor, macro-F1 **0.417**, AUC **0.50**, hallucination-F1 0.000; every R4 number is reported as a delta against it AND against the deployed cascade (end-to-end macro-F1 **0.789**, two-CE OOF **0.796**, warm mean 585 ms, 318 MB int8 NLI stage)
- **Ship-contract (carried from R3)** - macro-F1 ≥ **0.775** AND warm-CPU mean ≤ **200 ms** AND int8 footprint ≤ **100 MB**, two-sided at FP ≤ 266+2σ and FN ≤ 203+2σ; among passers ship the fewest params
- **Honest split (carried from R3)** - leave-one-source-out GroupKFold on `group_id` (619 sources / 636 traces, not 5,857 rows), held-out-language slice, train − holdout gap ≤ 0.05, nested re-fit of any new score scale on folds it never scores
- **NEW - representation gate** - a knowledge-free model with an 8k SYNTH-trained vocabulary may not be able to *see* the evidence; tokenizer fertility and context overflow are measured before any inference (R4-H22)
- **NEW - scope rule** - Monad-class candidates are pre-declared **English-slice only**; any aggregate number must be reported blended with the incumbent serving the non-EN slice, never as if the head were multilingual

### Pre-registration - predictions and bars fixed before the run

| ID | mechanism | prediction | PASS bar (two-sided) | first kill-gate |
|---|---|---|---|---|
| R4-H22 | representation feasibility - 8k SYNTH BPE over technical prose | EN fertility 1.6-2.2x mDeBERTa, non-EN ≥ 3x; EN context overflow &lt; 10% | EN fertility ≤ 2.5x AND EN overflow ≤ 20% | fertility &gt; 2.5x OR overflow &gt; 20% → round killed pre-inference |
| R4-H23 | zero-shot knowledge-free reasoner as grounding judge (Monad 56.7M, Baguettotron 0.3B, frozen) | Monad OOF AUC 0.62-0.74, Baguettotron 0.70-0.80, both &lt; incumbent 0.806 | Baguettotron ≥ 0.75 unlocks round; Monad ≥ 0.70 unlocks the tiny track | best of the two &lt; 0.65 → no zero-shot transfer; fall through to R4-H25/H26 only |
| R4-H24 | **the mechanism test** - context-grounded vs knowledge-dependent split | AUC deficit vs mDeBERTa ≤ 0.03 on self-contained pairs, ≥ 0.10 on knowledge-dependent | (deficit_knowledge − deficit_selfcontained) ≥ 0.07 | needs R4-H23 scores only - free re-analysis, no new compute |
| R4-H25 | post-train Monad as a 2-way grounding head (Needle recipe, cascade labels) | EN macro-F1 0.74-0.80 at ~57 MB int8, warm ≤ 150 ms | macro-F1 ≥ 0.775 AND ≤ 100 MB AND warm ≤ 200 ms AND FP/FN parity | R4-H23 Monad AUC ≥ 0.70 |
| R4-H26 | attention-only cross-encoder trained from scratch (SAN, QK-norm, domain BPE) | 10-25M params, ≤ 30 MB int8, macro-F1 0.72-0.78 | macro-F1 ≥ 0.757 (reranker-alone) at ≤ 30 MB | must beat R3-H18 distillation at matched footprint, else Null |
| R4-H27 | reasoning trace vs single-token verdict (serving shape) | trace +0.02-0.04 macro-F1 at 8-20x CPU latency | trace ships only if ≥ +0.02 macro-F1 AND ≤ 200 ms warm | trace latency measured on 50 pairs before any quality run |
| R4-H28 | English-only scope gate (EN → knowledge-free head, non-EN → incumbent) | blended macro-F1 ≥ 0.789 at 40-60% lower blended latency | blended macro-F1 ≥ 0.789 AND blended warm ≤ 350 ms | depends on a shipped R4-H25 or R4-H26 head |

### The hypotheses

**R4-H22 representation feasibility gate** - because Monad's 8,192-token BPE was trained exclusively on English synthetic SYNTH text while the gold is multilingual technical prose, measuring tokenizer fertility and context overflow before any inference will show EN fertility 1.6-2.2x mDeBERTa's 250k multilingual vocabulary and EN context overflow below 10%, while non-EN fertility exceeds 3x

- Lever - tokenizer only; model weights untouched, no inference run. Held fixed: the gold pairs and the top-8 pre-filter
- Mechanism - an 8k vocabulary trained on one distribution shatters out-of-distribution words into many subword pieces; past ~2.5x fertility the evidence no longer fits the context and the model literally cannot see what it must ground against. This is a representation failure, not a capability failure, and it is cheap to detect
- Probe / artifacts - encode `data/processed/golden_grounding_evidence_verified.parquet` claims + their top-8 chunks with `PleIAs/Monad` and `mDeBERTa-v3-base` tokenizers; report tokens/word by language and the overflow fraction. Minutes, CPU only, zero GPU
- Verdict space - Killed-at-gate (whole round) if EN fertility &gt; 2.5x or EN overflow &gt; 20%; Kept if EN passes and non-EN fails (activates the R4 scope rule); Ships-forward if both pass
- Experiment - `experiments/grounding-semantic/R4-H22_tokenizer_gate.py`<br>data: `data/processed/golden_v6/golden_v6.parquet` (8,800 rows, 21 languages; EN slice 4,876)<br>serving unit: claim + one `cfg.chunk_max_chars`=1500 chunk, top-`cfg.semantic_top_k`=3, via `groundrails.chunking.recursive_chunk`<br>run: `uv run python experiments/grounding-semantic/R4-H22_tokenizer_gate.py`<br>execution: CPU only, no weights downloaded, ~4 min
- **Result** - EN fertility **2.342 tok/word vs the incumbent's 1.959, ratio 1.196** (predicted 1.6-2.2 - better than predicted); EN context overflow **0.3%** at median 438 / p95 553 tokens against Monad's 2,048 context, ~4x headroom. Non-EN ratios sit in the same band (fr 1.198, es 1.194, it 1.222, nb 1.202, de 1.193) with overflow ≤ 0.42%, so the tokenizer is NOT the reason non-EN would fail
- **Verdict** - **PASS**, round 4 proceeds to R4-H23. The 30x smaller vocabulary (8,192 vs 250,101) costs only 20% more tokens on English evidence, which is what makes the size class reachable at all: a 250k x 768 embedding table is ~192M parameters, more than three times Monad's entire 56.7M budget
- Log
  - `log: 2026-07-28` first run reported KILLED-AT-GATE on 81.4% overflow. **False kill - a probe defect, not a model property.** It measured claim + the ENTIRE raw `source_text` (median 37,536 chars, ~24 chunks' worth), which no engine ever receives. The tell: the same measurement fails the deployed 512-token mDeBERTa far worse, and that model ships. Corrected to measure the real serving unit (claim + one 1500-char chunk); overflow 81.4% → 0.3%, verdict FAIL → PASS. Fertility was unaffected (1.196 both runs) because it is a property of the text, not the window

**R4-H23 zero-shot knowledge-free reasoner as a grounding judge** - because SYNTH-pretrained models are trained on reasoning traces rather than fact memorisation, scoring `PleIAs/Monad` (56.7M) and `PleIAs/Baguettotron` (0.3B) frozen over the cached top-8 gold pairs will read OOF AUC 0.62-0.74 and 0.70-0.80 respectively - below the incumbent NLI's 0.806, but far enough above chance to establish that grounding transfers without world knowledge

- Lever - the judge model; the pre-filter, the top-8 chunk set and the max-over-chunks aggregation are held fixed at the deployed configuration
- Mechanism - the verdict is read as the length-normalised log-probability of a `supported` vs `unsupported` continuation given (claim, chunk), so no head is trained and no gradient is taken; this is the same frozen-anchor design as R3-H15 and its number is directly comparable
- Probe / artifacts - one OOF pass over `data/interim/model_scores/pairs/full_pairs.npz` restricted to the EN slice per the scope rule; re-fit only the logistic on folds it never scores
- Verdict space - Killed-at-gate if the better model reads &lt; 0.65 (no zero-shot transfer; only the trained tracks R4-H25/H26 survive); Kept 0.65-0.75; unlocks the tiny track if Monad ≥ 0.70; Ships as a reference anchor only if either ≥ 0.806
- Experiment - `experiments/grounding-semantic/R4-H23_zeroshot_judge.py`<br>data: `.../gold/golden_grounding_evidence_verified.parquet`, EN slice 2,117 claims → 94,060 pairs (44.4 chunks/claim), base rate 0.649<br>scoring: length-free `logP(" yes") - logP(" no")` at the first assistant position, max-over-chunks<br>incumbent re-scored on the identical pairs (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, fp16)<br>hardware: RTX PRO 6000, ~6 min/model<br>run: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python experiments/grounding-semantic/R4-H23_zeroshot_judge.py`
- **Result** - **VOID. The measurement is invalid; no verdict on Monad is available from this run.** The recorded numbers were incumbent NLI **0.7333** and Monad **0.4736** (chance), but an instrument check invalidates the Monad figure: at the verdict position Monad assigns **p = 1.0000 to `'<'`** (the opening of its mandatory `<think>` trace), while `" yes"` and `" no"` carry logp -43.0 and -29.3 - **combined probability mass 0.000000**. The scorer was reading numerical noise in the extreme tail. Confirmed independently on trivially separable pairs (claim verbatim in its chunk vs claim against an unrelated recipe): AUC **0.745**, not ~1.0, with every score pinned near -13 regardless of content - the model is not answering the question asked
- **Verdict** - **VOID** for the logprob measurement, then **Refuted for Monad-at-56M** on the generative re-test (see Log). The pre-registered `< 0.65 → kill whole round` rule is NOT triggered: it presumes a valid measurement AND a representative candidate, and Monad is neither. Per the round's own amendment this promotes **R4-H27 from follow-on to decider**: Monad's trained format is non-negotiable, so the only valid zero-shot read allows the `<think>` trace. The pre-registered `< 0.65 → kill` rule is NOT triggered, because it presumes a valid measurement
- **Byproduct that stands** - the incumbent's English-only bar is **0.7756**, not the 0.806 headline. Re-deriving from cache: all-lang 0.8087, EN-only 0.7756, EN reranker 0.8827. English is the HARDER slice; cross-lingual claims separate more easily because NLI fails them uniformly. Every R4 target is therefore 0.7756 on English, and this harness's chunking costs a further 0.042 (0.7756 → 0.7333) which is carried as a caveat on all R4 numbers
- Log
  - `log: 2026-07-28` first run scored `logP(" supported") - logP(" unsupported")`. In Monad's 8k vocabulary `" unsupported"` is five pieces `[' un','su','pp','ort','ed']`, so first-token scoring compared `" supported"` against the bare prefix `" un"`. Fixed with a fail-loud single-token guard and a yes/no pair; AUC moved 0.4764 → 0.4736, i.e. **the token defect was real but was not the cause of the null**
  - `log: 2026-07-28` the instrument check above then invalidated the design itself. Two consecutive false kills in this round (R4-H22's overflow, R4-H23's null) shared one signature: **a result too extreme to be interesting**. Methodology addition below
  - `log: 2026-07-28` **the VOID was correct about the instrument and wrong about the cause - dug deeper on challenge.** The scorer's prompt ended at `<|im_start|>assistant\n`, but `chat_template.json` appends `<|im_start|>assistant\n<think>\n` - **the `<think>` tag is part of the PROMPT, not a model choice**. So `p=1.0000` on `'<'` was the prompt being truncated one token early, not the model refusing to answer. With the correct prefix Monad generates normally
  - `log: 2026-07-28` **generative probe (greedy, correct template, 7 cases incl. the native `<source_1>` RAG shape) - Monad at 56M cannot do the task.** It reproduces the trained FORM faithfully - rubric headers `###`, confidence markers `●◐○`, source-examination structure - and in the source-tagged format it correctly LOCATES the span (`"Source 1 provides direct answer: ..." ● High confidence`). The CONTENT is incoherent: it reframed a grounding question as arithmetic (`1300 - 271 = 911`, actually 1029), wrote `271m = 271,000 ft` (889) and `1300 MW = 1300,000 MW`, computed `66.76 - 99.76 = -1.36%` (actually -33.0), asserted `271m = substantial bread height` against a sourdough recipe, could not answer "what is the capital of France" (looped `Capital of France (capital city)` four times), and could not extract `1937` from the sentence containing it. Not a prompt-format artefact - the failure is identical across plain ChatML, source-tagged RAG, and pure extraction
  - `log: 2026-07-28` **consequence - candidate replaced, thesis untouched.** Monad's own card numbers frame the result: MMLU ~30% against 25% chance for 4-way multiple choice is +5 points, GSM8K 8%, HotPotQA 8%. It is a demonstration of the smallest model emitting well-formed English with reasoning-shaped scaffolding, not a capable reasoner. This REFUTES Monad-at-56M as a zero-shot grounding judge; it does NOT test the round's thesis, which now runs on `PleIAs/Baguettotron` (321M, best-in-class for its size) and `PleIAs/Pleias-RAG-350m` (actually trained for grounding with literal quotes, arXiv 2504.18225). The span-location capability that DID survive is the retrieval half of grounding and is worth carrying forward

**R4-H24 the mechanism test - context-grounded versus knowledge-dependent** - because the source paper localises the entire attention-only deficit to parametric recall and measures attention-only models AHEAD on passage-supplied answers (Sciq 0.742 vs 0.661 at 105B), partitioning the gold into self-contained pairs (adjudicable from the supplied chunk alone) and knowledge-dependent pairs (needing a fact absent from the chunk) will show the knowledge-free models' AUC deficit versus mDeBERTa at ≤ 0.03 on the first and ≥ 0.10 on the second

- Lever - the evaluation partition only; identical scores, identical models, re-analysed. Nothing is trained and no new inference runs
- Mechanism - this is the round's crux and its cheapest experiment. If the deficit is uniform across the partition, the "grounding needs no knowledge" premise is false FOR THIS TASK and R4's rationale collapses to "it is a small model", which R3 already pursues with better-suited architectures. If the deficit splits as predicted, the mechanism transfers from next-token loss to pairwise grounding and the round is justified on evidence rather than analogy
- Probe / artifacts - partition by whether the human rationale in the verified gold cites only the supplied chunk; adjudicate ~200 pairs per side to keep the partition honest, blind to model scores. Reuses R4-H23's scores at zero marginal compute
- Verdict space - Confirmed if (deficit_knowledge − deficit_selfcontained) ≥ 0.07; Refuted if &lt; 0.03 (premise dead - record it and stop the round here rather than spending training budget); Null in between

**R4-H25 post-train Monad as a 2-way grounding head** - because the frozen cascade emits a calibrated per-pair grounded-probability for any (claim, chunk) at zero labelling cost and Needle demonstrates that 45 minutes over 2B tokens specialises a knowledge-free base into a narrow structured task, post-training Monad on the 111,800-pair cache will hold EN macro-F1 0.74-0.80 at ~57 MB int8 and warm ≤ 150 ms

- Lever - the post-training corpus and objective; the base checkpoint, the pre-filter and the top-8 aggregation are fixed
- Mechanism - Needle's result is that a knowledge-free base plus a short task-specific post-train beats general models 5-25x larger on the narrow task, and the minimum-data rule (120 examples per class) is exceeded by three orders of magnitude here. The 8k vocabulary and absent FFN are what make 56.7M affordable at 64 layers
- Probe / artifacts - teacher labels from `full_pairs.npz`; post-train per the needle recipe (`needle finetune`, JSONL with query/tools/answers adapted to claim/evidence/verdict); NNCF int8 IR via `scripts/build_ov_grounder.py`; evaluate under the carried-forward R3 honest-split protocol
- Verdict space - Ships if macro-F1 ≥ 0.775 at ≤ 100 MB and ≤ 200 ms with FP/FN parity; Kept 0.757-0.775 (a size/latency point below the contract); Dropped &lt; 0.757 (below reranker-alone - the knowledge-free base added nothing a standard encoder does not)

**R4-H26 attention-only cross-encoder trained from scratch** - because the FFN holds two thirds of non-embedding parameters and its deletion costs 0.006 nats at matched parameters, an encoder-only SAN cross-encoder (12-20 attention-only layers, d=512, QK-normalization, a domain BPE of 8-16k) trained directly on the cascade-labelled pair cache will reach macro-F1 0.72-0.78 at 10-25M parameters and ≤ 30 MB int8 - a size class below anything R3 can reach

- Lever - the architecture; the training data, labels and split protocol are identical to R4-H25 so the two are directly comparable
- Mechanism - the paper's trainability finding is specific and load-bearing: QK-normalization, not the FFN and not residual gating, is what keeps deep attention-only stacks trainable, so it is a precondition of the build rather than a tuning option. Pairwise grounding never queries weight-stored facts, so the one measured weakness is not exercised
- Probe / artifacts - build on the needle reference implementation (MIT); train a domain BPE on the gold corpus; train on `full_pairs.npz` cascade targets; compare against R3-H18's distilled standard encoder at MATCHED footprint - that comparison, not the absolute number, is what justifies a new architecture
- Verdict space - Ships if macro-F1 ≥ 0.757 at ≤ 30 MB; Kept if it beats R3-H18 at matched footprint; Null if R3-H18 matches or beats it (distillation into a standard encoder is simpler and already in flight); Dropped if training does not converge without the FFN at this data scale

**R4-H27 reasoning trace versus single-token verdict** - because Monad's distinguishing feature is that it emits an intermediary reasoning trace and CPU decode cost scales linearly with emitted tokens, generating a trace before the verdict will add +0.02-0.04 macro-F1 while costing 8-20x the latency of a single-token verdict, failing the 200 ms bar

- Lever - the decode length; model, prompt and data fixed
- Mechanism - a trace is only worth its cost if the grounding decision needs multi-step composition. Most gold hallucinations are omissions (H9), which a single comparison resolves; if the trace helps, it should help disproportionately on the multi-hop subset, which is the diagnostic to record alongside the aggregate
- Probe / artifacts - latency first on 50 pairs (minutes) to establish the multiplier, then quality only if the multiplier leaves headroom under 200 ms
- Verdict space - Ships the trace only if ≥ +0.02 macro-F1 AND ≤ 200 ms warm; Kept as an explainability-only mode (off the serving path) if it adds quality but breaks latency; Dropped if it adds &lt; 0.01

**R4-H28 English-only scope gate** - because Monad is strictly monolingual English while the non-EN slice is already the cascade's weakest (AUC 0.584, non-EN macro-F1 0.637), routing EN claims to a shipped knowledge-free head and leaving non-EN on the incumbent will hold blended macro-F1 ≥ 0.789 at 40-60% lower blended latency

- Lever - the routing rule; both heads are fixed, already-measured artefacts
- Mechanism - the language detector already runs in the lexical tier, so routing is free. This converts Monad's monolingual limitation from a blocker into a scope decision, and it is the only honest way to report an aggregate number for an English-only head
- Probe / artifacts - blend R4-H25/H26 EN verdicts with the deployed cascade's non-EN verdicts over the 2,752 gold; report blended macro-F1, blended warm latency, and the per-language table
- Verdict space - Ships if blended macro-F1 ≥ 0.789 at ≤ 350 ms; Kept if latency wins at macro-F1 ≥ 0.775; Dropped if the routing boundary itself costs more than it saves

### Sequencing

A gated ladder, cheapest-decisive-first. **R4-H22 is minutes on CPU and can kill the round before a single forward pass** - an 8k SYNTH vocabulary that cannot represent the evidence ends it there. R4-H23 is one frozen scoring pass. **R4-H24 then costs nothing** - it re-analyses R4-H23's scores - and it is the decision point: it tests the round's premise directly rather than inferring it from the source paper's next-token results, and a refutation there should stop the round before any training budget is spent. Only on a confirmed mechanism do R4-H25 (post-train the 56.7M base) and R4-H26 (build the attention-only head) run, and R4-H26's verdict is explicitly relative to R3-H18 at matched footprint. R4-H27 and R4-H28 are serving-shape decisions that presuppose a shipped head. Every number above is a prediction pending measurement under the carried-forward R3 ship-contract and honest-split protocol.

### Amendments - 2026-07-28, before R4-H23 ran

Append-only. R4-H22's verdict above is recorded and immutable; these change hypotheses that had NOT yet run, and each states what prompted it.

- **Scope narrowed to English, on the author's decision.** The round is testing whether the METHOD is load-bearing, not multilingual coverage. Every R4 number is an English-slice number unless stated. This is a deliberate narrowing of the question, not a claim that non-EN works
- **R4-H28 rewritten.** The pre-registration routed non-EN to the incumbent and kept two heads. The shipped argos MT bridge (`lexical_mt.py`, torch-free CTranslate2, already load-bearing in the lexical HIGH tier) makes a better design available: translate non-EN claims into English and run ONE head for everything. New prediction - blended macro-F1 ≥ 0.789 with a single head; new risk - MT paraphrase noise on the claim side, which R4-H22 shows is NOT a tokenizer problem (non-EN fertility 1.19-1.22, overflow ≤ 0.42%). The two-head routing variant is retained only as the fallback
- **The round conflated two separable claims; they are now split.** `PleIAs/Monad`'s config is `LlamaForCausalLM`, `hidden_size` 256, 64 layers, `intermediate_size` 768 - it **has MLPs and is not a SAN**. So Monad tests the *knowledge-free training* half of the thesis (SYNTH: reasoning traces, not memorised facts) and says nothing about the *attention-only architecture* half, which only R4-H26 tests. R4-H23/H25 results must be read as evidence about training data, never about deleting FFNs
- **R4-H23 re-scores the incumbent instead of quoting the cache.** The cached anchors were built on a chunking this repo can no longer reproduce - cached mean is 40.62 chunks/claim, and `recursive_chunk` matches the per-claim counts on only 39/300 claims at max_chars=1000 and 21/300 at 1500. Quoting 0.806 against a differently-chunked candidate would confound the comparison, so mDeBERTa-v3-NLI is re-scored on the identical freshly-generated pairs
- **R4-H23's number is a declared LOWER BOUND.** Monad's chat template forces a `<think>` trace before every answer (`<|im_start|>assistant\n<think>\n`), so scoring a direct supported/unsupported continuation violates its trained format. The consequence is a cleaner design than the pre-registration had: R4-H23 (no trace) is the lower bound, R4-H27 (with trace) the upper, and **the gap between them IS the value of the reasoning trace**. A sub-bar R4-H23 therefore promotes R4-H27 from follow-on to decider rather than killing the round outright
- **Harness validated before any candidate number is believed** (the R3 protocol's requirement). Re-deriving the anchors from `full_pairs.npz` with max-over-chunks reproduces the documented figures: NLI entailment **0.8087** (doc 0.806) and reranker **0.8414** (doc 0.841)

### Setup - R4 artefacts and data

Everything R4 runs against, so a reader reproduces it without the transcript.

- **Gold (verdict set)** - `experiments/grounding-semantic/private-rag-forensics/gold/golden_grounding_evidence_verified.parquet`, 2,752 rows `{claim, source_text, label, lang, user_id, trace_id}`, 1,966 supported / 786 hallucination; **EN slice 2,117 claims → 94,060 pairs at 44.4 chunks/claim, base rate 0.649**
- **Gold (tokenizer/fertility set)** - `data/processed/golden_v6/golden_v6.parquet`, 8,800 rows over 21 languages, EN slice 4,876
- **Cached incumbent scores** - `.../model_scores/pairs/full_pairs.npz`: `owner` (111,800), `rr`, `nli` (3-col, `id2label` = entailment/neutral/contradiction), `labels` (2,752), `langs`
- **Chunking** - `groundrails.chunking.recursive_chunk` at `cfg.chunk_max_chars` = 1500, `cfg.semantic_top_k` = 3; note the raw `source_text` is the whole retrieved blob (EN median 37,536 chars), never the serving unit
- **Candidates** - `PleIAs/Monad` (56.7M, 114.4 MB, Apache 2.0, ctx 2048, vocab 8192), `PleIAs/Baguettotron` (0.3B); incumbent `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
- **Hardware** - RTX PRO 6000 Blackwell Max-Q (96 GB, sm_120), `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`
- **Scripts** - `experiments/grounding-semantic/R4-H22_tokenizer_gate.py`, `R4-H23_zeroshot_judge.py`; logs under `logs/R4-*.log`, scores to `R4-H23_scores.npz`

### Methodology addition - validate the instrument before the experiment

Adopted 2026-07-28 after two consecutive false kills in this round. Both were measurement defects that produced verdicts, and both were caught only because the number was implausible - which is luck, not method.

- **Positive-control every new scorer before it scores anything real.** Run it on a handful of trivially separable pairs (claim verbatim inside its chunk vs claim against unrelated text). A scorer that cannot reach AUC ~1.0 there cannot be trusted at 0.75 on real data. This is what exposed R4-H23: trivial-pair AUC 0.745 with every score pinned near -13
- **Check where the probability mass actually is.** For any generative judge, print the top-k tokens at the verdict position before reading a verdict off two hand-picked ids. R4-H23's verdict tokens held 0.000000 of the mass; the model wanted `<think>`
- **Verdict words must be single tokens**, asserted in code, not assumed. Multi-token verdicts silently degrade to prefix comparisons
- **Measure the unit the engine actually receives.** R4-H22 measured claim + the whole 37.5k-char evidence blob rather than the 1,500-char serving chunk and reported 81.4% overflow instead of 0.3%
- **Treat "too extreme to be interesting" as a defect signal, not a finding.** A gate that also fails the shipping incumbent, or an AUC at exactly chance, is evidence about the harness first and the candidate second

### R4-H29 positive-control gate - zero-shot generative judges (result)

Added as a gate, not a hypothesis, under the "validate the instrument before the experiment" rule. It is the cheapest decisive test in the round and it closed a whole path.

**R4-H29 generative judge positive control** - because a judge that cannot separate a claim quoted verbatim from its own chunk from the same claim against an unrelated recipe cannot be trusted on real data, gating each candidate on 20 trivially separable pairs before spending a subsample run will show >= 90% parseable verdicts and >= 90% correct, or stop the candidate

- Lever - the candidate model; prompt, decoding (greedy), token budget (700) and the parser are held identical across candidates
- Mechanism - trivial separation is a necessary condition, not a sufficient one. The pairs share no vocabulary and no topic with the negative evidence, so any judge with a working verdict channel should be near-perfect; failure here means the null on real data would be uninterpretable
- Experiment - `experiments/grounding-semantic/R4-H29_positive_control.py`<br>20 cases: 10 claims verbatim inside their chunk (SUPPORTED) + the same 10 against one sourdough-recipe chunk (UNSUPPORTED)<br>prompt ends with `<|im_start|>assistant\n<think>\n` per `chat_template.json` - the `<think>` tag is part of the PROMPT<br>verdict parsed ONLY from the post-`</think>` segment, preferring an explicit `Answer: yes/no` line<br>run: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python experiments/grounding-semantic/R4-H29_positive_control.py PleIAs/Baguettotron PleIAs/Monad`
- **Result** - **both candidates FAIL, and fail in the same direction**

| candidate | params | parseable | correct | SUPPORTED (ok/wrong/none) | UNSUPPORTED (ok/wrong/none) |
|---|---|---|---|---|---|
| Baguettotron | 321M | 16/20 (80%) | 12/20 (60%) | 7 / **0** / 3 | 5 / **4** / 1 |
| Monad | 56.7M | 6/20 (30%) | 4/20 (20%) | 3 / **0** / 7 | 1 / **2** / 7 |

- **The asymmetry is the finding: 0 false negatives across both models, 6 false positives.** Neither model ever denied a genuinely supported claim; both confirmed claims against a bread recipe. Raw generations show the mechanism - the model **substitutes the claim for the evidence** and then agrees with itself ("Evidence: 'The dam is 271 metres tall'" when the evidence was the recipe)
- Monad additionally fails to commit at all on 70% of cases - it never closes `</think>` within 700 tokens
- **Verdict** - **the zero-shot generative-judge path is CLOSED for this model class.** For a grounder, manufacturing support is strictly worse than missing a hallucination: it launders a fabrication as verified. A candidate that is 40% wrong in that direction on topically-unrelated evidence cannot be trusted on near-miss negatives, which are strictly harder
- **The round's thesis is NOT refuted by this.** What is refuted is *asking an untrained model to perform the discrimination zero-shot*. R4-H25 (post-train on cascade labels) and R4-H26 (attention-only head trained on the pair task) train the discrimination rather than request it, and they remain live. The R4-H22 representation result (fertility 1.196x, overflow 0.3%) also stands and still makes the size class reachable
- Log
  - `log: 2026-07-28` **the Monad row is under review in round 6, not yet void.** A token-level audit found that `PleIAs/Monad`'s shipped `chat_template.json` emits `<|im_start|>`, `<|im_end|>` and `<think>`, none of which exist in its 8,192-token vocabulary - each shatters into 3-5 sub-word pieces, and the model cannot emit `</think>` as a single token, which fits the 70% non-termination observed here. **But the shattering is not by itself a defect**: Monad's `added_tokens` list is exactly `[UNK]`, `<|begin_of_text|>`, `<|end_of_text|>`, `[PAD]` and its BPE fills all 8,192 embedding slots, while its model card states it was trained on "the standard instruction style from Qwen" with that same ChatML block. If PleIAs trained through this tokenizer, the model saw the identical piece sequence during training and the R4-H29 prompt was faithful. R6-H40 discriminates the two readings; until it returns, this row stands as recorded. The Baguettotron row is unaffected - all four control strings are single tokens there (ids 65491-65494)
  - `log: 2026-07-28` this gate was itself repaired twice before it produced a usable number - the first parser took the first `yes|no` anywhere in the output, matching phrases like "No contradiction. No support needed." inside the reasoning trace, and scored a CORRECT Baguettotron answer as False. Rebuilt against raw generations: the verdict lives after `</think>` in an `Answer:` line. Budget also raised 220 → 700 after cases ran out mid-trace

## Hypothesis round 5 - prompt and elicitation engineering for small-model judgement (pre-registered)

Batch **J**, ids `R5-H30..R5-H39`. Round 4 established that the two knowledge-free candidates *can see* the evidence (R4-H22, fertility 1.196x) but cannot *judge* it when asked zero-shot (R4-H29, 0 false negatives / 6 false positives on trivially separable pairs). Round 5 holds the weights fixed and moves the only untested lever left before training: the prompt and the elicitation protocol.

The round is aimed by a second, independent observation - the shipped grounder is weak on prose. On a private proposal-genre prose set (77 claims, 25 source documents) the lexical tier confirms 25/77 (32.5%), the full cascade 29/77 (37.7%), and 28/60 (46.7%) once DEF-20 holds out-of-scope claims out of the denominator. The gold-set macro-F1 of 0.824 was measured on RAG traces, not on authored prose; this genre is materially harder and is the deployment target.

### Methodology - carried-forward bars and the R5 additions

- **Naive baseline** - unchanged: majority predictor macro-F1 **0.417**, hallucination-F1 0.000. Every quality claim below is a delta against it
- **Incumbent bars** - full gold stack macro-F1 0.824 / OOF AUC 0.913; English-only NLI AUC **0.7756** on the matched-chunking re-score (the English slice is the harder one - all-language reads 0.8087). The English bar is what an EN-only head must beat
- **Prose bar** - scope-aware confirm rate **46.7%** (28/60) on the private prose set; raw 37.7% (29/77)
- **NEW - the residue is unmeasured** - only 31 of the 77 prose claims carry a human label. "The grounder is bad on prose" is a hypothesis about the other 46, not an established fact. R5-H30 measures it before any build and can re-aim or kill the round
- **NEW - false positives are the tracked error** - R4-H29 showed the failure is one-directional. For a grounder, manufacturing support is strictly worse than missing a hallucination, so every R5 bar names an FP count, not only an aggregate
- **NEW - negatives must be near-miss** - R4-H29 used topically-unrelated negatives (a bread recipe). Any prompt gain measured there is provisional until it holds on near-miss negatives (R5-H37)
- **Held fixed across the round** - weights (frozen), decoding (greedy), candidate set (`PleIAs/Baguettotron` 321M, `PleIAs/Monad` 56.7M), the 20-case positive control from R4-H29. Only the prompt, the elicitation protocol and the verdict-extraction path vary

### Pre-registration - predictions and bars fixed before the run

| id | persona | lever | prediction | acceptance bar | gate |
|---|---|---|---|---|---|
| R5-H30 | deflationist | adjudicate the unconfirmed in-scope prose claims | ≥ 40% are correct refusals, not misses | ≥ 60% correct refusals → the "bad on prose" framing is refuted and the round is re-aimed at recall of the true residue | runs first; no model inference |
| R5-H31 | deflationist | question polarity - "is the claim UNSUPPORTED?" | verdicts flip wholesale; the model agrees with whatever is asked | flip-rate ≥ 70% → acquiescence bias, a prompt artefact, cheaply fixable | free re-run of the R4-H29 harness |
| R5-H32 | mechanist | decompose locate → compare → judge | locate ≥ 70% correct, judge ~50% | (locate − judge) ≥ 0.20 localises the broken stage | free re-run |
| R5-H33 | contrarian | structured prompt with an explicit absence instruction and a forced quote-before-verdict | Baguettotron 60% → ≥ 90% on the R4-H29 control | ≥ 85% correct AND FP ≤ 1/10 → **supersedes the R4-H29 verdict** | requires R5-H32 to show a working locate stage |
| R5-H34 | transfer | chain-of-verification - draft, generate verification questions, answer from evidence only, revise | false positives −50% | FP ≤ 3/10 at ≤ 2x the single-pass latency | requires R5-H33 ≥ 75% |
| R5-H35 | transfer | cloze reframing - mask a claim token, score recovery from the evidence | the verdict channel is bypassed entirely | AUC ≥ 0.70 on the EN gold slice | independent of R5-H33 |
| R5-H36 | heretical | no verdict is requested - the model emits the supporting span or a null token, deterministic code decides | beats every judged prompt in the round | AUC ≥ 0.75 EN with a code-only verdict | independent of R5-H33 |
| R5-H37 | contrarian | near-miss negatives replace topically-unrelated ones | prompt gains do not transfer | any R5-H33/H34 gain holds within 0.10 on near-miss | runs against whichever prompt wins |
| R5-H38 | hybridizer | self-consistency, k=5 samples, majority vote | FP drops, latency 5x | FP ≤ 2/10 AND warm ≤ 200 ms, else research-only | requires a prompt above 75% |
| R5-H39 | mechanist | claim-first versus evidence-first ordering | position drives the claim-substitution | asymmetry ≥ 0.15 → artefact, not comprehension | free re-run |

### The hypotheses

**R5-H30 the residue is unmeasured** - because only 31 of the 77 prose claims carry a human label and a claim the grounder declines may be a correct refusal rather than a miss, independently adjudicating every unconfirmed in-scope claim against the 25 source documents will show at least 40% are genuinely unsupported by the supplied sources, while the grounder's 46.7% scope-aware rate understates its true precision

- Lever - the measurement, not the system; no code changes, no inference, nothing shipped
- Mechanism - a confirm rate is a ratio whose denominator assumes every in-scope claim IS supported somewhere in the corpus. Authored proposal prose contains forward-looking statements, figures carried from documents outside the source set, and assertions the author never sourced. Each is a correct refusal scored as a failure
- Prediction - 40-70% of unconfirmed in-scope claims are correct refusals; the true miss count is 10-19 of 60
- Acceptance bar - ≥ 60% correct refusals refutes the framing and re-aims the round at the true residual; ≤ 30% confirms the grounder is the problem and the round proceeds as written; 30-60% keeps both live
- Experiment - `experiments/grounding-semantic/R5-H30_adjudicate_residue.py`<br>data: `experiments/grounding-semantic/private-prose-forensics/` (77 claims, 25 sources, gitignored)<br>procedure: for each unconfirmed in-scope claim dump the grounder's top-k retrieved chunks AND an independent exhaustive sentence scan over the raw sources ranked by claim-token overlap, so a refusal is not adjudicated from the grounder's own retrieval<br>adjudication: `miss` (a source states it, retrieval failed) versus `correct_refusal` (no source states it), written back into the gitignored dump<br>run: `uv run python experiments/grounding-semantic/R5-H30_adjudicate_residue.py`<br>execution: CPU, ~2 min; adjudication by reading, every `miss` confirmed by locating the supporting sentence in a named source

- **Result** - **26 of the 32 unconfirmed in-scope claims are correct refusals (81.2%)**, above the top of the predicted 40-70% band. Only 6 are genuine misses

| adjudication | n | share | note |
|---|---|---|---|
| correct refusal - authorial judgement | 10 | 31% | the author's own reading of the material; no source could state it |
| correct refusal - proposal design | 6 | 19% | the solution being proposed, not a claim about the sources |
| correct refusal - author's own estimate | 5 | 16% | costs, hours, market figures carried from experience |
| correct refusal - question or plan | 4 | 13% | interrogatives and next-step statements |
| correct refusal - compound opinion | 1 | 3% | one groundable clause welded to an evaluative one |
| **miss** | **6** | **19%** | a source states it and the grounder did not find it |

- **The confirm rate was a denominator artefact.** True groundable set = 28 confirmed + 6 misses = 34, so recall is **28/34 = 82.4%**, not 46.7%. The other 26 claims are unsourceable by construction and were being counted as failures
- **Adjudication agrees with every human label present** - all 10 residue claims labelled `not_groundable` were independently called correct refusals, and both labelled `supported` were independently called misses (12/12)
- **The misses split by error direction, and half are the dangerous kind** - 3 returned `none`, but **3 returned `contradicted`** on a claim a source states verbatim, including one carrying three percentages that appear unaltered in the source digest. Asserting that a document contradicts a claim it actually supports is strictly worse than declining it
- **Verdict** - **Refuted (the framing)**. The pre-registered `≥ 60% correct refusals → re-aim the round` rule fires at 81.2%. The grounder is not bad on prose; the metric was. What survives as a real defect is precision on the contradiction arm and the recall of the out-of-scope classifier, neither of which is a prompt problem - see the amendment below

**R5-H31 acquiescence bias versus claim substitution** - because R4-H29's six false positives are equally explained by a model that agrees with any proposition put to it and by one that substitutes the claim for the evidence, re-running the identical control with the question polarity inverted ("is the claim UNSUPPORTED by the evidence?") will flip at least 70% of verdicts if the cause is acquiescence, and leave them stable if the cause is substitution

- Lever - the polarity of the question; evidence, claims, decoding and parser identical to R4-H29
- Mechanism - acquiescence is a surface artefact of instruction-tuned agreement and is repairable by prompt design; substitution is a comprehension failure and is not. The two demand different rounds, and R4-H29 cannot distinguish them
- Prediction - flip-rate 70-95% on Baguettotron; Monad's non-committal rate stays near 70% under either polarity
- Acceptance bar - flip-rate ≥ 70% → the round proceeds with prompt engineering as the primary lever; ≤ 30% → substitution is real and R5-H33/H34 are downgraded in favour of R5-H36
- Experiment - `experiments/grounding-semantic/R5-H31_polarity_flip.py`, reusing the R4-H29 20-case control unchanged

**R5-H32 which stage is broken - locate, compare, or judge** - because R4-H29's raw generations show the model correctly quoting the relevant passage before reaching a wrong verdict, splitting the task into three separately scored sub-tasks will show locate accuracy at least 0.20 above judge accuracy, localising the defect to the verdict stage rather than to comprehension

- Lever - the task decomposition; one model call per sub-task instead of one for the whole
- Mechanism - a single prompt conflates retrieval, comparison and commitment. If locate is intact the model is not blind, and the correct design removes the stage that fails rather than trying to strengthen it - which is what R5-H36 does
- Prediction - locate 0.70-0.85, compare 0.55-0.75, judge 0.45-0.60 on Baguettotron
- Acceptance bar - (locate − judge) ≥ 0.20 → the verdict stage is the defect and R5-H36 is promoted; < 0.10 → the model is uniformly weak and no prompt will fix it, closing R5-H33/H34/H38
- Experiment - `experiments/grounding-semantic/R5-H32_stage_decomposition.py`, same 20 cases, three scored prompts per case

**R5-H33 the R4-H29 prompt was the weak link** - because R4-H29's prompt supplied no instruction to check for absence, no explicit refusal option and no output contract, replacing it with a structured prompt that forces a verbatim quote before any verdict and states the absence rule explicitly will lift Baguettotron from 60% to at least 90% on the identical control

- Lever - prompt structure only; model, decoding, cases and parser unchanged from R4-H29
- Mechanism - a model that must first emit a span from the evidence cannot substitute the claim for the evidence without producing a quote that visibly is not in the text. The quote requirement makes the substitution failure detectable rather than silent
- Prediction - correct 0.85-0.95, FP 0-1/10, parseable ≥ 0.95 on Baguettotron; Monad stays below 0.60
- Acceptance bar - ≥ 85% correct AND FP ≤ 1/10 supersedes the R4-H29 verdict with a one-line back-reference; < 0.75 leaves R4-H29 standing
- Experiment - `experiments/grounding-semantic/R5-H33_structured_prompt.py`

**R5-H34 chain-of-verification against self-agreement** - because the observed failure is the model agreeing with its own restatement of the claim, inserting a verification pass - draft a verdict, generate questions about the evidence, answer them from the evidence alone, then revise - will halve the false-positive count at no more than twice the single-pass latency

- Lever - the number of elicitation passes; prompt content held to the R5-H33 winner
- Mechanism - the verification questions are answered against the evidence with the claim out of context, which denies the model the opportunity to restate the claim as its own evidence
- Prediction - FP 6 → 2-3 across both models; latency 1.8-2.4x
- Acceptance bar - FP ≤ 3/10 AND ≤ 2x latency; a quality gain past 2x latency is recorded but does not ship
- Experiment - `experiments/grounding-semantic/R5-H34_chain_of_verification.py`

**R5-H35 cloze reframing bypasses the verdict channel** - because the broken component is commitment to a yes/no verdict rather than reading comprehension (R5-H32), reframing the task as evidence-conditioned completion - mask the claim's key token and score the model's recovery of it - will read AUC ≥ 0.70 on the English gold slice without ever asking for a judgement

- Lever - the output modality: token likelihood instead of a generated verdict
- Mechanism - a supported claim's key token is recoverable from the evidence; an unsupported claim's is not. The signal is a length-normalised logprob, deterministic, single forward pass, and comparable to a cross-encoder score
- Prediction - AUC 0.65-0.78 on Baguettotron, 0.55-0.68 on Monad, both below the 0.7756 English incumbent
- Acceptance bar - ≥ 0.70 keeps the tiny track alive as a cascade stage-0 gate; < 0.60 drops it
- Experiment - `experiments/grounding-semantic/R5-H35_cloze_recovery.py`, EN slice of the verified gold, same matched-chunking pairs as R4-H23

**R5-H36 do not ask for a verdict at all** - because R4-H29 shows these models locate correctly and judge badly, restricting the model to emitting a supporting span or a null token and having deterministic code decide - does the returned span actually contain the claim's numeric, entity and unit anchors - will beat every judged prompt in this round and read AUC ≥ 0.75 on the English slice

- Lever - the division of labour between model and code; the model retrieves, `entity_check` decides
- Mechanism - the failure mode is manufacturing agreement, and a manufactured span is checkable in a way a manufactured verdict is not. A hallucinated quote fails a substring test against the source; a hallucinated "yes" fails nothing. This also reuses the existing deterministic anchor layer rather than adding a new one
- Prediction - AUC 0.72-0.82 EN, FP ≤ 1/10 on the R4-H29 control, span-hallucination rate 5-20% and fully caught by the substring test
- Acceptance bar - AUC ≥ 0.75 EN with a code-only verdict; below 0.65 closes the knowledge-free zero-shot track entirely
- Experiment - `experiments/grounding-semantic/R5-H36_quote_only.py`; verdict from `groundrails.entity_check` anchors over the returned span, never from model text

**R5-H37 near-miss negatives are the real bar** - because R4-H29's negatives shared no vocabulary or topic with the claims, any prompt improvement measured on them is provisional, and re-testing the winning prompt against negatives drawn from the same document with one altered number, entity or qualifier will not transfer

- Lever - the negative class; prompt and model held at the round's best configuration
- Mechanism - topically-unrelated negatives are separable by lexical overlap alone, so a model can score well on them without doing entailment. Near-miss negatives are the operational case - the shipped grounder's false positives are all topically close
- Prediction - accuracy drops 0.15-0.35 from the unrelated-negative figure
- Acceptance bar - any R5-H33/H34 gain must hold within 0.10 on near-miss; a larger drop marks the gain as an artefact of the control set
- Experiment - `experiments/grounding-semantic/R5-H37_near_miss_negatives.py`; 20 near-miss cases built by perturbing one anchor per positive chunk

**R5-H38 self-consistency against the one-directional error** - because a single greedy sample commits to one substitution, sampling five verdicts at temperature and taking the majority will cut false positives without touching the prompt, at five times the latency

- Lever - the number of samples; prompt fixed at the round's best
- Mechanism - substitution is a sampling-path failure rather than a fixed belief, so independent paths should disagree; if they agree, the failure is systematic and no aggregation helps
- Prediction - FP 6 → 2-4, latency 4.5-5.5x
- Acceptance bar - FP ≤ 2/10 AND warm ≤ 200 ms; above 200 ms it is recorded as a research result and does not ship
- Experiment - `experiments/grounding-semantic/R5-H38_self_consistency.py`

**R5-H39 evidence-first versus claim-first ordering** - because claim substitution may be a recency artefact of the claim sitting closest to the generation point, swapping the order so the claim precedes the evidence will change the false-positive count by at least 0.15 if position drives the failure

- Lever - the order of the two blocks in the prompt; everything else identical to R4-H29
- Mechanism - the R4-H29 prompt places the claim immediately before the assistant turn, which is exactly where a restatement would be most available. If ordering moves the result, the failure is positional and cheap to fix; if not, it is comprehension
- Prediction - FP asymmetry 0.10-0.30 in favour of claim-first
- Acceptance bar - asymmetry ≥ 0.15 → positional artefact, fold the winning order into every later prompt; < 0.05 → comprehension, and prompt ordering is closed
- Experiment - `experiments/grounding-semantic/R5-H39_block_ordering.py`, free re-run of the R4-H29 harness

### Sequencing

Cheapest-decisive-first, and the first three can each re-aim the round before any build. **R5-H30 runs first and uses no GPU** - if most of the prose residue is correct refusals, the premise that the grounder is bad on prose is wrong and the round should target recall of a much smaller true residual. **R5-H31, R5-H32 and R5-H39 are free re-runs** of the existing R4-H29 harness with one variable changed each, and together they diagnose whether the failure is acquiescence, a broken verdict stage, or position. Only then do the builds run: R5-H33 is the direct attack on R4-H29's own verdict, R5-H34 and R5-H38 are elaborations that presuppose it, and R5-H35 and R5-H36 are independent alternatives that do not depend on any prompt working. **R5-H37 gates everything that passes** - no prompt gain is believed until it survives near-miss negatives. Every number above is a prediction pending measurement.

### Amendments - 2026-07-28, after R5-H30 ran

Append-only. R5-H30's verdict above is recorded and immutable; these change hypotheses that had NOT yet run, and each states what prompted it.

- **The round loses its prose justification but keeps its gold-set one.** R5-H31..H39 were aimed by two claims: the R4-H29 judgement failure, and "the grounder is bad on prose". The second is refuted - true prose recall is 82.4%, above the gold-set stack's own operating point. The prompt-engineering ladder stays live against R4-H29 and the English gold slice, but it is no longer the highest-value work on this repo and it drops behind the three defects below
- **NEW - false contradiction is the round's real finding.** 3 of the 6 misses were returned as `contradicted`, not `none`. The contradiction arm has a precision problem that no amount of retrieval or prompting touches, and it is the one error class a grounder must never make: it launders a correct document as wrong. Tracked as a defect, not a hypothesis, because the mechanism is deterministic code (`find_numeric_mismatches` and the contradicted-arm span picker), not a model
- **NEW - the out-of-scope classifier under-fires on authored prose.** DEF-20 catches hypothetical, self-referential and directive claims. It does not catch evaluative-authorial statements, proposal design, the author's own cost and effort estimates, or interrogatives - 26 such claims passed the in-scope filter on this set, 10 of them carrying a human `not_groundable` label. This is a recall gap in a classifier whose whole purpose is to keep ungroundable claims out of the denominator
- **NEW - the extractor welds separate bullets into one claim.** Several residue claims concatenate two or three source bullets into a single assertion, one clause of which is groundable and the rest opinion. A compound claim cannot be adjudicated as a unit, so it is unfalsifiable by construction and is scored as a failure
- **Scope-aware scoring must be reported against an adjudicated denominator.** The 46.7% figure and every earlier prose confirm rate assumed each in-scope claim is supported somewhere. Any future prose number states its groundable denominator or it is not comparable

## Hypothesis round 6 - the instrument, not the model (pre-registered)

Batch **K**, ids `R6-H40..R6-H47`. Round 4 concluded that zero-shot generative judging is closed for this model class. A token-level audit of the candidates shows that conclusion rests on a broken instrument for one of the two, and that the model actually trained for this task was never run at all.

**`PleIAs/Monad`'s shipped `chat_template.json` is written in tokens Monad's own tokenizer does not contain.** The template emits `<|im_start|>`, `<|im_end|>` and `<think>`; Monad's 8,192-token SYNTH vocabulary has none of them, so each shatters into 3-5 sub-word pieces. The same template string tokenises to **25 tokens on Monad against 12 on Baguettotron**, where all four are single control tokens (ids 65491-65494). Monad has never seen those markers as tokens, and it cannot emit `</think>` as a unit - which is exactly the observed failure, 70% of R4-H29 cases never closing the trace. The R4-H29 Monad row measured a malformed prompt.

**`PleIAs/Pleias-RAG-350M` ships no chat template because its format is a 19-token structured protocol**, every token present in vocabulary: `<|query_start|>` 65517, `<|query_end|>` 65518, `<|source_start|>` 65519, `<|source_id|>` 65520, `<|source_end|>` 65521, `<|language_start|>` 65522, `<|language_end|>` 65523, `<|query_analysis_start|>` 65524, `<|query_analysis_end|>` 65525, `<|query_report_start|>` 65526, `<|query_report_end|>` 65527, `<|source_analysis_start|>` 65528, `<|source_analysis_end|>` 65529, `<|source_report_start|>` 65530, `<|source_report_end|>` 65531, `<|draft_start|>` 65532, `<|draft_end|>` 65533, `<|answer_start|>` 65534, `<|answer_end|>` 65535. The `source_analysis` segment is the model assessing whether the supplied sources support an answer, and the model carries a trained refusal path for sources that cannot - which is this task in its native format. It is the closest published precedent for a tiny grounding head (`references/papers/[paper digest] pleias-rag small reasoners quote sources.md`, arXiv 2504.18225) and it has never been fed a single pair here.

A third anomaly, cheap to test: `Baguettotron` and `Pleias-RAG-350M` both auto-prepend **id 2 = `<|end_of_text|>`** as the sequence-initial token, because their `bos_token` is null and the post-processor falls back to EOS. Monad prepends the correct id 1 = `<|begin_of_text|>`. Every Baguettotron prompt in round 4 therefore began with an end-of-document marker.

### Methodology - carried forward, plus the pre-flight this round adds

- **Naive baseline** - unchanged: majority macro-F1 **0.417**. English incumbent NLI AUC **0.7756** on matched chunking
- **Control set** - the R4-H29 20-case trivial control, unchanged, so every number in this round is directly comparable to `Baguettotron 12/20` and `Monad 4/20`
- **NEW - look before parsing, formalised** - no scorer is written against an assumed output shape. Every candidate gets a raw-generation dump under each candidate format first, and the parser is built against what was observed. This is the R4 methodology addition applied as a build step rather than a lesson
- **NEW - vocabulary coverage is a pre-flight, not a discovery** - a candidate whose declared prompt format references strings absent from its own vocabulary is mis-specified, and any score taken under that format is void. R6-H47 makes this a gate that runs before a candidate is scored
- **Held fixed** - weights frozen, greedy decoding, the 20 control cases, the post-hoc parser discipline (verdict read only from the trained answer segment)

### Pre-registration - predictions and bars fixed before the run

| id | persona | lever | prediction | acceptance bar | gate |
|---|---|---|---|---|---|
| R6-H47 | scout | vocabulary-coverage pre-flight over every candidate | Monad FAILS its own template, Baguettotron and Pleias-RAG PASS | any FAIL voids that candidate's prior scores | runs first, seconds, no inference |
| R6-H46 | deflationist | token budget under each protocol against the model's real context | Monad ctx 2048 holds one 1,500-char chunk, overflows a multi-source prompt | overflow > 20% on the serving unit kills Monad on capacity | free, CPU |
| R6-H40 | mechanist | rebuild Monad's prompt from tokens it actually has | parseable 30% → ≥ 80%, correct 4/20 → ≥ 10/20 | ≥ 80% parseable **supersedes the R4-H29 Monad row**; still < 50% parseable confirms it as a model limit | needs R6-H47 |
| R6-H42 | follower | Pleias-RAG-350M under its native 19-token protocol | ≥ 90% correct on the trivial control, 0 false positives | < 90% closes the small-model track on a valid instrument | the round's decider |
| R6-H41 | mechanist | BOS ablation - suppress the EOS-as-BOS prepend | accuracy moves ≥ 0.10 on Baguettotron if prompts were off-distribution | ≥ 0.10 → every prior Baguettotron number is re-taken | free re-run |
| R6-H43 | deflationist | the trained refusal path IS the verdict - no prompt engineering | AUC ≥ 0.75 on the EN gold slice | ≥ 0.7756 beats the English incumbent outright | needs R6-H42 ≥ 90% |
| R6-H44 | mechanist | read the verdict from `source_analysis`, not from the answer | ≥ 0.05 AUC over the answer channel | free - same generation, two readouts | needs R6-H42 |
| R6-H45 | contrarian | the model's inline literal quotes + deterministic anchor check | beats every judged prompt in rounds 5 and 6 | AUC ≥ 0.75 with a code-only verdict | R5-H36 with a model trained to quote |

### The hypotheses

**R6-H47 vocabulary-coverage pre-flight** - because a prompt format built from strings a tokenizer does not carry is fed to the model as sub-word noise rather than control tokens, auditing every candidate's declared template against its own vocabulary before any inference will show Monad failing on all four of its template's control strings while Baguettotron and Pleias-RAG-350M pass

- Lever - the audit itself; nothing about the models changes
- Mechanism - a control token is a single embedding the model was trained to condition on. Shattered into pieces it becomes ordinary text in a position where structure was expected, and the model's trained format is never actually invoked
- Prediction - Monad 0/4 template tokens in vocabulary; the shipped template string tokenises to 25 pieces against Baguettotron's 12
- Acceptance bar - any candidate failing its own template has every prior score under that format marked void; the gate is then permanent for later candidates
- Experiment - `experiments/grounding-semantic/R6-H47_vocab_gate.py`<br>for each candidate: resolve the declared template, tokenise it, report per-control-string coverage, and enumerate every control-shaped token actually present in vocabulary<br>run: `uv run python experiments/grounding-semantic/R6-H47_vocab_gate.py`<br>execution: CPU, seconds, no weights loaded

**R6-H46 token budget under the real protocol** - because Monad's context is 2,048 against 4,096 for the other two and its 8k vocabulary costs 1.196x the incumbent's tokens, measuring the serving unit under each model's own protocol will show a single 1,500-char chunk fitting comfortably while a multi-source Pleias-style prompt overflows Monad

- Lever - the measurement unit: the assembled prompt as the engine receives it, per protocol, not the claim plus a blob
- Mechanism - R4-H22 measured fertility on claim + one chunk and passed. The Pleias protocol wraps several sources with per-source ids, which is a different and larger unit
- Prediction - single-chunk overflow < 1% on all three; multi-source overflow 30-70% on Monad, < 5% on the other two
- Acceptance bar - overflow > 20% on the intended serving unit kills that candidate on capacity, independently of quality
- Experiment - folded into `R6-H47_vocab_gate.py`; same run, no extra cost

**R6-H40 Monad under a format it can represent** - because Monad's only in-vocabulary control tokens are `<|begin_of_text|>` (1) and `<|end_of_text|>` (2), rebuilding the prompt as plain text between those two and discovering the trained shape from raw generations will lift parseable verdicts from 30% to at least 80% and correct answers from 4/20 to at least 10/20

- Lever - the prompt encoding; model, cases, decoding and token budget held at R4-H29 values
- Mechanism - the failure signature was not wrong answers but absent ones: 70% never closed the trace. A model asked to terminate a structure it has no token for cannot terminate it. Removing the structure removes the failure
- Prediction - parseable 0.80-0.95; correct 0.50-0.75, still below Baguettotron
- Acceptance bar - ≥ 80% parseable supersedes the R4-H29 Monad row with a back-reference; < 50% parseable under a valid format confirms the original verdict on the model rather than the harness
- Pre-experiment probe - raw-generation dump under three candidate formats before any scoring; the parser is written against the observed shape
- Experiment - `experiments/grounding-semantic/R6-H40_monad_format.py`<br>same 20 cases as R4-H29<br>run: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python experiments/grounding-semantic/R6-H40_monad_format.py`

**R6-H42 the model built for this, in its own protocol** - because Pleias-RAG-350M was mid-trained specifically to answer from supplied sources with inline literal quotes and an explicit refusal path, and its 19 protocol tokens are all present in vocabulary, running the trivial control under that native protocol will reach at least 90% correct with zero false positives

- Lever - the candidate model and its trained protocol together; the control cases are unchanged
- Mechanism - the R4-H29 failure was a model substituting the claim for the evidence. A model trained to emit the supporting excerpt cannot substitute silently: the excerpt either appears in the source or it does not. The `source_analysis` segment exists precisely to assess source adequacy before answering
- Prediction - correct 0.90-1.00, false positives 0-1, parseable ≥ 0.95
- Acceptance bar - ≥ 90% correct and FP ≤ 1 unlocks R6-H43/H44/H45; < 90% on a valid instrument closes the small-model zero-shot track for real
- Pre-experiment probe - raw-generation dump under the assembled protocol before scoring
- Experiment - `experiments/grounding-semantic/R6-H42_pleias_rag_protocol.py`<br>prompt assembled from the in-vocabulary protocol tokens, the claim posed as the query and the chunk as a single source<br>verdict read only from the trained answer segment, never from free text<br>run: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python experiments/grounding-semantic/R6-H42_pleias_rag_protocol.py`

**R6-H41 the EOS-as-BOS prepend** - because Baguettotron and Pleias-RAG-350M declare no `bos_token` and their tokenizers fall back to prepending id 2 `<|end_of_text|>`, every round-4 prompt opened with an end-of-document marker, and suppressing it will move control accuracy by at least 0.10 if the prompts were off-distribution

- Lever - one token at position 0; everything else identical
- Mechanism - an end-of-document marker at the start of a sequence signals a document boundary the model then conditions on. Whether that is off-distribution depends on how the corpus was packed during training, which is not documented, so it must be measured rather than assumed
- Prediction - a move of 0.00-0.15; most likely small, because packed-corpus training often does separate documents with EOS
- Acceptance bar - ≥ 0.10 means every prior Baguettotron number is re-taken under the better setting; < 0.05 closes the question
- Experiment - `experiments/grounding-semantic/R6-H41_bos_ablation.py`, the 20-case control run twice per model

**R6-H43 the refusal path is the verdict** - because Pleias-RAG-350M was trained with "the supplied sources cannot answer this" as a first-class output rather than an absence of output, posing the claim as a query and reading refusal-versus-answer as the grounding verdict will reach AUC ≥ 0.75 on the English gold slice with no prompt engineering at all

- Lever - the readout: a trained behaviour is used directly instead of a judgement being requested
- Mechanism - refusal and "unsupported claim" are the same predicate. The model already computes it; round 5's prompt ladder exists only because the earlier candidates did not
- Prediction - AUC 0.70-0.82 EN
- Acceptance bar - ≥ 0.7756 beats the English incumbent outright and promotes the candidate to a cascade stage; ≥ 0.70 keeps it as a gate
- Experiment - `experiments/grounding-semantic/R6-H43_refusal_verdict.py`, EN gold slice, matched chunking as in R4-H23

**R6-H44 source_analysis versus the answer channel** - because the model emits an explicit source-adequacy assessment before drafting, scoring the verdict from the `source_analysis` segment rather than from the final answer will read at least 0.05 AUC higher on the same generations

- Lever - which segment of one generation is read; no extra inference
- Mechanism - the answer is shaped by fluency pressure and by the draft stage; the analysis segment is the model's own statement about evidence sufficiency, which is closer to the quantity being predicted
- Prediction - +0.03 to +0.10 AUC over the answer channel
- Acceptance bar - ≥ +0.05 makes the analysis segment the shipped readout
- Experiment - re-analysis of the R6-H43 generations at zero marginal compute

**R6-H45 the quote is the verdict** - because Pleias-RAG-350M generates literal source excerpts inline rather than reconstructing them post hoc, and a fabricated excerpt fails a substring test against the source while a fabricated verdict fails nothing, checking the emitted quote deterministically with `entity_check` anchors will beat every judged prompt in rounds 5 and 6

- Lever - the division of labour: the model quotes, deterministic code decides
- Mechanism - this is R5-H36 with a model actually trained to quote. The paper's own argument for generated over post-hoc citation is that retrospective cross-referencing does not hold up in production, which is the same failure mode this hypothesis exploits in reverse
- Prediction - AUC 0.75-0.85 EN, quote-hallucination rate 3-15% and fully caught by the substring test
- Acceptance bar - AUC ≥ 0.75 with a code-only verdict
- Experiment - `experiments/grounding-semantic/R6-H45_quote_verdict.py`, reusing R6-H43 generations plus `groundrails.entity_check`

### Sequencing

Two free CPU steps decide what is worth loading. **R6-H47 and R6-H46 run first in one script** - seconds, no weights - and they say which candidates were ever validly measured and which fit the serving unit at all. Then the two GPU questions run against the unchanged 20-case control: **R6-H40** asks whether Monad was ever given a fair test, and **R6-H42** asks whether the model built for this job works. R6-H42 is the decider - a failure there, on an instrument the pre-flight has certified, closes the small-model zero-shot track on evidence rather than on a malformed prompt. R6-H41 is a free ablation alongside. Only on a passing R6-H42 do the gold-slice runs follow: R6-H43 produces the generations, and R6-H44 and R6-H45 are re-analyses of those same generations at zero marginal compute. Every candidate gets a raw-generation dump before any parser is written.

### Round 6 results

Run 2026-07-28. The pre-flight and the reproduction both landed as predicted; the grounding result did not.

**R6-H47 vocabulary-coverage pre-flight - PASS (prediction confirmed)**

| candidate | declared format | control strings in vocab | rendered template | sequence-initial token |
|---|---|---|---|---|
| Monad 56.7M | ChatML (`chat_template.json`) | **0/4** | **25 tokens** | `<\|begin_of_text\|>` (id 1) |
| Baguettotron 321M | ChatML (attribute) | 4/4 | 12 tokens | `<\|end_of_text\|>` (id 2) |
| Pleias-RAG-350M | structured protocol, no template | 19/19 | n/a | `<\|end_of_text\|>` (id 2) |

- The gate caught an instrument bug in itself first: `tok.chat_template` is empty for Monad because its template lives in a separate `chat_template.json`, so the first version classified Monad as a protocol model and audited it against the wrong 19 tokens - a correct FAIL for an incorrect reason. Fixed before the verdict was recorded
- Monad's `added_tokens` list is exactly `[UNK]`, `<|begin_of_text|>`, `<|end_of_text|>`, `[PAD]`, and its BPE fills all 8,192 embedding slots, so there is no room for the control tokens its own template emits
- The EOS-as-BOS anomaly is confirmed for both 65k-vocab models (R6-H41 remains unrun)
- **Verdict** - **Kept as a standing gate.** Every future candidate is audited before it is scored

**R6-H46 token budget - PASS, prediction wrong in direction**

- Single-source serving unit: Monad 470 tok (23% of its 2,048 ctx), the other two 385 tok (9% of 4,096)
- Three-source unit: Monad 1,348 tok (66%), the other two 1,098 tok (27%)
- **Verdict** - **Null.** Predicted 30-70% multi-source overflow on Monad; measured zero overflow anywhere. Capacity is not a constraint for any candidate and this line of attack is closed

**R6-H40 Monad cannot be validated - Killed-at-gate**

Rather than argue about the format, the published number was targeted directly. Monad's card reports MMLU "close to 30%" against a 25% chance floor; three standard protocols were run over a 2,000-question stratified MMLU sample.

| protocol | accuracy | vs chance |
|---|---|---|
| loglikelihood, 0-shot | 0.255 | +0.005 |
| loglikelihood, 5-shot | 0.226 | -0.024 |
| ChatML generative (the card's own block, 67% parseable) | 0.143 | -0.107 |

- **No protocol reaches the published number, and none clears chance.** The best reading is 0.255 against a 0.250 floor
- This does not establish that the published figure is wrong - PleIAs released no eval code for Monad, so a protocol that reproduces it may exist. It does establish that **no Monad measurement in this repo can be validated**, which is the operative fact
- **Verdict** - **Killed-at-gate.** Monad is withdrawn as a candidate. The R4-H29 Monad row is neither confirmed nor voided; it is simply uninterpretable, and no further Monad work is warranted

**R6-H42 Pleias-RAG-350M as a judge - Refuted**

Stage 1 reproduced the model card's own worked example verbatim - the model returned "Paris is the capital of France" with correctly-formed inline `<ref name="<|source_id|>1">` citations. **The harness is validated**, which is what gives the negative result below its force.

| readout | correct | FP | FN | parseable |
|---|---|---|---|---|
| Pleias-RAG-350M judge | 11/20 (55%) | **9** | 0 | 20/20 |
| R4-H29 Baguettotron judge | 12/20 (60%) | 4 | 0 | 16/20 |

- The purpose-built grounding model, on a validated harness, confirms claims against a sourdough recipe nine times out of ten. Same one-directional failure as round 4, now on an instrument that is beyond dispute
- **Verdict** - **Refuted.** Zero-shot generative judging is closed for this model class on evidence, not on a malformed prompt. This is the round's decider and it settles the question round 4 could not

**R6-H45 the quote is the verdict - Ships on the control, Refuted on the gold**

Same generations, zero extra compute. The model emits literal excerpts, so a fabricated excerpt is caught by two string tests: is the quote genuinely in the source, and does it share a numeric or entity anchor with the claim.

| readout | control 20 cases | FP | gold 600 EN claims (macro-F1) | P | R |
|---|---|---|---|---|---|
| judge (model decides) | 11/20 (55%) | 9 | 0.4762 | 0.521 | 0.870 |
| **quote + code (code decides)** | **18/20 (90%)** | **1** | **0.5107** | 0.615 | **0.267** |
| naive majority baseline | - | - | 0.4170 | - | - |
| shipped cascade (all-lang gold) | - | - | 0.8240 | - | - |

- On the control the mechanism works exactly as designed: false positives 9 → 1, and the model's fabricated citations (`<ref>The dataset spans 964 sensors</ref>` against a bread recipe) are caught by substring alone
- **On the gold it does not transfer.** Macro-F1 0.511 beats the judge channel (+0.035) and the naive baseline (+0.094), and loses to the shipped cascade by **-0.313**
- **Recall is the failure: 0.267.** On the control the supporting sentence WAS the chunk; on the gold the model must locate and quote an exact span inside three 1,500-character sources, and for three quarters of genuinely-supported claims it emits no usable quote at all. Precision 0.615 means even the quotes it does emit are wrong 38% of the time
- **Verdict** - **Refuted for deployment, Kept as a finding.** The control result was a ceiling produced by verbatim-in-chunk positives, not a transferable capability. Recorded because the mechanism is sound and the failure is located precisely: quote *generation* under real retrieval, not the deterministic check
- Correction to the round's own framing - R6-H37 was pre-registered to test whether control gains survive near-miss negatives. They did not even survive the move to real data, so R6-H37 is now moot for this candidate

**R6-H43 the refusal path as a verdict - Refuted**

- The judge readout above IS the refusal path, read from the trained answer segment: macro-F1 **0.4762**, recall 0.870, precision 0.521, 240 false positives on 300 negatives
- Predicted AUC 0.70-0.82 against the 0.7756 English incumbent; measured a verdict that is barely above a coin flip and 0.35 macro-F1 below the shipped stack
- **Verdict** - **Refuted.** The trained refusal behaviour does not discriminate on this data

### Round 6 conclusion

The round did what it was built to do: it replaced two contested measurements with validated ones, and both came back negative.

- **The instrument objection is settled.** Pleias-RAG-350M's harness reproduces its vendor's published example exactly, so its failure is the model's, not ours. Monad's cannot be validated at all, so Monad is out
- **Small generative models cannot judge grounding.** Three models, two vocabularies, a validated harness and a purpose-built grounding checkpoint all produce the same one-directional failure: they manufacture support
- **The quote-and-check mechanism is real but the quoting is not good enough.** Deterministic verification of an emitted excerpt is sound and cheap; a 350M model asked to find that excerpt under realistic retrieval succeeds 27% of the time
- **Nothing here displaces the shipped stack.** The deployed cascade holds macro-F1 0.824; the best round-6 configuration reaches 0.511
- **Where the remaining value is** - not in a smaller model. `DEF-22` (CONTRADICTED returned on claims a source states verbatim) and `DEF-23` (out-of-scope classifier recall) are defects in deterministic code that already ships, and both were surfaced by round 5's adjudication

## Hypothesis round 7 - is the task capacity-limited, and is our number real (pre-registered)

Batch **L**, ids `R7-H49..R7-H53`. Rounds 4-6 closed the generative track. A four-lane research sweep (reports under `reports/research-*.md`) then converged, independently, on two claims that this round tests rather than accepts:

- **architecture is probably not the binding constraint** - the honest ceiling estimated from both the training-recipe and architecture lanes is 0.80-0.84, i.e. parity with the shipped cascade, because a distilled student cannot exceed its teacher and the teacher is what our labels encode
- **our headline number may not be comparable to anything** - macro-F1 0.824 at base rate 0.649 converts to roughly 0.82-0.85 balanced accuracy, which would sit ABOVE the LLM-AggreFact leaderboard top of 77.4. The far likelier reading is that the private gold is easier or more loosely labelled than the public benchmark

**Process note, recorded rather than hidden.** `R7-H52` below was executed BEFORE this pre-registration was written, as GPU shape benchmarks commissioned during the research sweep. It is recorded as a measurement with its numbers intact, and explicitly NOT claimed as a pre-registered hypothesis. `R7-H49`'s script was likewise written before registration. Both are noted here so the log does not imply a discipline that was not followed; the remaining hypotheses are registered before execution as normal.

### Methodology - what this round adds

- **Naive baseline** - unchanged, majority macro-F1 **0.417**. Incumbent full stack **0.824**, English NLI AUC 0.7756, best single signal reranker AUC 0.841
- **NEW - external calibration is now mandatory before any target is set.** No quality target may be quoted against the private gold alone until `R7-H49` establishes what that gold is worth externally
- **NEW - speed is a first-class acceptance criterion**, not a tiebreak. The shipped cascade is ~662 ms/claim warm on CPU across three models
- **NEW - the teacher corpus must be regenerated.** `model_scores/pairs/full_pairs.npz` carries 111,800 rows of `owner`/`rr`/`nli`/`labels`/`langs` and **no text**, and the chunking that produced it cannot be reproduced (R4 amendment: cached mean 40.62 chunks/claim, `recursive_chunk` matches 39/300 at max_chars=1000). Soft labels therefore cannot be re-attached to their inputs, and every distillation hypothesis depends on regenerating the cache under current chunking first

### Pre-registration - predictions and bars fixed before the run

| id | lever | prediction | acceptance bar | gate |
|---|---|---|---|---|
| R7-H49 | score the shipped cascade on LLM-AggreFact, no threshold refitting | 63-72 balanced accuracy against our private-gold ~82-85 | gap &gt; 8 points → 0.824 is not SOTA-comparable and must never be quoted as such | blocked on Hub auth; no GPU |
| R7-H50 | capacity ablation - 4 checkpoints at 140M / 278M / 307M / 278M-minus-6-layers | all four land within 0.02 macro-F1 of each other | spread ≤ 0.02 → the task is NOT capacity-limited and architecture work is closed | needs R7-H51 |
| R7-H51 | regenerate the teacher corpus under current chunking, text retained | ~110k (claim, chunk) pairs with reranker + NLI scores and their text | reproduces the incumbent's 0.824 on the gold from the regenerated pairs | prerequisite for R7-H50/H52/H53 |
| R7-H52 | encoder shape at matched parameters - depth versus width, attention-only, parallel blocks | width wins on latency; attention-only loses | **already measured, see below** | ran before registration |
| R7-H53 | MICE-style split encoder - N independent layers with the document side cached offline, M joint layers | ~9x faster than the joint model at an AUC deficit ≤ 0.01 | deficit ≤ 0.01 → ships; &gt; 0.03 → dropped | needs R7-H50 |

### R7-H52 encoder shape - measured (executed before registration)

Matched at ~140-160M body parameters, bf16, B=3 T=512, RTX PRO 6000. Harness `experiments/grounding-semantic/arch_shape_latency_bench.py` and `arch_shape_seq_bench.py`, logs `logs/shape-bench{,2}.log`.

| shape | L | d | body M | GFLOP | eager ms | graphed ms | TFLOP/s |
|---|---|---|---|---|---|---|---|
| narrow-deep | 53 | 512 | 149.5 | 544.2 | **14.91** | 4.44 | 122.6 |
| | 28 | 768 | 140.5 | 498.9 | 8.64 | 3.33 | 149.8 |
| | 16 | 1024 | 142.7 | 489.6 | 4.69 | 2.81 | 174.5 |
| wide-shallow | 8 | 1536 | 160.5 | 531.5 | **2.98** | 2.81 | **188.9** |
| 16x1024 + parallel attn/FFN block | 16 | 1024 | 142.7 | 489.6 | 4.46 | **2.75** | 177.9 |
| attention-only (SAN shape) | 36 | 1024 | 151.1 | **579.8** | 6.09 | 3.25 | 178.6 |

- **Width beats depth on latency at matched parameters and FLOPs** - narrow-deep is 5.0x slower eager and 1.58x slower graphed, and achieved throughput falls 189 → 123 TFLOP/s. Depth's quality edge in the literature is measured on parametric-recall tasks, the one axis grounding does not use
- **Attention-only is refuted on the latency axis** - at matched parameters it needs 3x the layers and burns **18% more FLOPs** (579.8 vs 489.6), running 1.16-1.30x slower. This closes R4-H26's premise on speed rather than on quality
- **Parallel attention+FFN blocks help modestly and reliably** - 2.806 → 2.752 ms graphed at identical parameters
- **Batching the top-3 chunks as one forward is worth -32%**, and is an execution-order change rather than an architectural one
- MICE-style top-layer cost at T=64 claim tokens only: **0.298 ms** graphed against 2.707 ms for the joint model, which is what motivates R7-H53
- **Verdict** - **Kept as a measurement.** Recommended shape from it: 16 layers x 1024 hidden, GeGLU d_ff 1536, parallel blocks, 128k trimmed vocabulary, 274.9M total (131.1M embeddings + 142.7M body), 2.71 ms/forward graphed. Not a hypothesis verdict - no quality was measured here, only shape and speed

### Sequencing

`R7-H49` is free of GPU and free of the teacher cache, so it runs first the moment Hub auth is available - and it can invalidate every target in this round by showing the private gold is not comparable. `R7-H51` is a prerequisite build, not a hypothesis, and everything downstream waits on it. `R7-H50` is the decisive one: if four checkpoints spanning 140M to 307M all land within 0.02, the task is not capacity-limited, the architecture question is closed, and the remaining work is labels rather than models. `R7-H53` only matters if a joint model ships at all.

### Round 7 results

**R7-H51 regenerate the teacher corpus - PASS**

The cached `full_pairs.npz` carries 111,800 rows of `owner`/`rr`/`nli`/`labels`/`langs` and **no text**, under a chunking this repo can no longer reproduce (R4 amendment: cached mean 40.62 chunks/claim, `recursive_chunk` matching per-claim counts on 39/300 at max_chars=1000). Soft labels could not be re-attached to their inputs, so the corpus was rebuilt under current chunking with the text retained.

| quantity | regenerated | incumbent reference | delta |
|---|---|---|---|
| pairs | 123,579 (44.9 chunks/claim) | 111,800 (40.62) | +10.5% |
| reranker AUC, max-over-chunks | **0.8289** | 0.841 | **-0.012** |
| NLI entailment AUC, max-over-chunks | 0.7674 | 0.806 all-lang | **-0.039** |

- **Verdict** - **PASS on the pre-registered bar** (reranker within 0.02) and the corpus ships as the teacher, at `private-rag-forensics/R7-H51_teacher_pairs.parquet` with `owner`, `claim`, `chunk`, `rerank`, `entail`, `neutral`, `contradiction`, `label`, `lang`
- **Recorded miss, not glossed** - the NLI channel drifts -0.039, outside the tolerance the reranker met. The bar was written against the reranker so the gate passes, but the honest reading is narrower than the gate: this corpus is a faithful teacher **for the reranker signal**, and the NLI channel should not be used as a distillation target without explaining the drift first. Chunk-count shift (44.9 vs 40.62) is the leading suspect
- Harness `experiments/grounding-semantic/R7-H51_regenerate_teacher.py`, log `logs/R7-H51-teacher-corpus.log`, torch fp16 on RTX PRO 6000, ~13 min for both scorers over 123,579 pairs

**R7-H50 capacity ablation - IN PROGRESS, first checkpoint recorded**

Splits are by CLAIM, not by pair, so no student is scored on a claim it trained on: 688 held-out claims (base rate 0.725), 40,000 training pairs subsampled from the remainder, one epoch, teacher signal = the reranker score, aggregation = max-over-chunks.

| checkpoint | params | AUC | macro-F1 | ms/pair |
|---|---|---|---|---|
| mmBERT-small | 140.6M | **0.8772** | 0.7946 | 5.18 |
| mmBERT-base | 307M | pending | pending | pending |
| mDeBERTa-v3-base | 278M | pending | pending | pending |
| mDeBERTa minus 6 layers | ~235M | pending | pending | pending |

- **The 140M student beats its own teacher by +0.048 AUC** (0.8772 against the teacher's 0.8289 on the identical corpus). This contradicts the ceiling assumption both research lanes carried - "a distilled student cannot exceed its teacher" holds for reproducing the teacher's outputs, but the student is scored against HUMAN GOLD, and training on smoothed soft targets averages out per-pair teacher noise. The 0.80-0.84 ceiling estimate in this round's preamble is now in doubt
- **Two caveats attached to the number before anyone quotes it.** The macro-F1 is optimistic: the threshold is chosen to maximise F1 **on the test set**, which is not an honest operating point, so **AUC is the only trustworthy figure here**. And the held-out slice runs base rate 0.725 against the full gold's 0.649, so 0.7946 is not comparable to the incumbent's 0.824 without a matched re-score
- **Verdict** - pending the remaining three checkpoints; the pre-registered bar is a macro-F1 spread ≤ 0.02 across 140M-307M

**R7-H54 fp16 serving latency (pre-registered addition to round 7)**

Added because R7-H50's eval loop produced a latency figure that is not a serving number and must not be quoted as one: the students ran **fp32** (the `from_pretrained` default) while both teachers were timed in **fp16**, inside a DataLoader with variable-length padding. The raw comparison it implied is therefore invalid in the student's disfavour by roughly 2x.

The confound surfaced a real question. Measured at batch 64 during R7-H51 and R7-H50:

| model | params | shape | measured | dtype |
|---|---|---|---|---|
| bge-reranker-v2-m3 | 568M | 24L x 1024 | **3.63 ms/pair** | fp16 |
| mmBERT-small | 140M | 22L x 384 | 5.18 ms/pair | **fp32** |
| mDeBERTa-v3-base | 278M | 12L x 768 | 5.84 ms/pair | fp16 |

- **Parameter count does not predict latency here, and the shape column says why.** mmBERT-small carries 4x fewer parameters than the reranker but nearly the same DEPTH - 22 layers against 24 - and depth is the part that cannot be parallelised. Narrowing 1024 → 384 removes most of the FLOPs and little of the wall-clock, because the GPU goes latency-bound rather than compute-bound at that width, which is the same effect R7-H52 measured as achieved throughput falling 189 → 123 TFLOP/s on narrow shapes
- **mDeBERTa is slower than a model twice its size** - disentangled attention is attention-heavy, so its parameter savings sit outside the bottleneck
- **Consequence for the design** - shrinking parameters does not buy speed; cutting DEPTH does. This makes R7-H50's `mDeBERTa-minus-6L` arm the most informative of the four, since it is the only one that moves depth with width, family and tokenizer held fixed

- Hypothesis - because sequential depth rather than parameter count sets encoder latency, re-timing every candidate at fp16 with `sdpa` and `torch.compile` at fixed shapes will show the sub-350M students clustering with the 568M teacher rather than beating it, and the 6-layer variant separating from all of them
- Prediction - mmBERT-small 1.2-1.6x the teacher's throughput, not the ~4x its parameter count implies; mDeBERTa-minus-6L ≥ 2x
- Acceptance bar - a candidate must be ≥ 2x the teacher at B=3 to justify replacing it on latency grounds alone
- Experiment - `experiments/grounding-semantic/R7-H54_fp16_serving_latency.py`<br>fp16 and bf16, `attn_implementation="sdpa"`, fixed shapes at seq 512, warmup 12, median of 40 reps<br>batches 1 / 3 / 8, where **B=3 is the real serving unit** (top `cfg.semantic_top_k` chunks scored as ONE batch)<br>run: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python experiments/grounding-semantic/R7-H54_fp16_serving_latency.py`

**R7-H55 a stronger teacher raises the student's ceiling (pre-registered)**

R7-H50 measured the student against its teacher on identical held-out traces: `bge-reranker-v2-m3` (568M) reads AUC **0.8619**, the distilled 140M student **0.8269**, a deficit of 0.035. A distilled student is bounded by its teacher, so that 0.8619 is the ceiling every architecture and capacity hypothesis in this round is competing under.

**The teacher carries none of the project's constraints.** It runs once, offline, to label 123,579 pairs, and never serves a request. The sub-350M limit, the latency budget and the CPU export path all apply to the STUDENT. Choosing `bge-reranker-v2-m3` as teacher was inherited from what the cascade happens to ship, which is not a reason - it is an oversight, recorded here as such.

- Hypothesis - because a distilled student's quality is bounded by its teacher's ranking, and the teacher is an offline labeller with no size, latency or export constraint, replacing `bge-reranker-v2-m3` (568M) with the strongest reranker that fits the 96 GB card will raise the teacher's AUC on the held-out traces above 0.8619 and lift the distilled student proportionally
- Lever - the teacher only; student architecture, training recipe, splits and evaluation held exactly at R7-H50 values so the two runs are directly comparable
- Mechanism - soft labels transfer the teacher's ranking, not its parameters. A better ranking is a better target at identical labelling cost, and labelling cost is a one-time offline pass
- Candidates to score before choosing: `BAAI/bge-reranker-v2-gemma` (~2B), `BAAI/bge-reranker-v2-minicpm-layerwise` (~2B), the `Qwen3-Reranker` tier (0.6B / 4B / 8B), `mxbai-rerank-large-v2`, and an ensemble of the strongest two. Licences must be checked - `jina-reranker-v2-base-multilingual` is CC-BY-NC and therefore excluded
- Prediction - the best available reranker reads 0.88-0.92 on the held-out traces against the incumbent's 0.8619; the student closes to within 0.03 of it, landing 0.85-0.89
- Acceptance bar - teacher AUC ≥ 0.88 AND distilled student ≥ 0.85 on the same 159 test traces; a teacher that gains less than 0.02 is not worth a relabelling pass
- Kill condition - if a materially stronger teacher does NOT lift the student, the deficit is the student's capacity or the training budget rather than the label quality, and the round's conclusion changes accordingly
- Pre-experiment probe - score every candidate reranker on the gold FIRST and compare AUC on the held-out traces; only the winner relabels the corpus. Scoring is minutes, relabelling is a full pass
- Experiment - `experiments/grounding-semantic/R7-H55_teacher_search.py`<br>stage 1: rank candidate rerankers on the 159 held-out traces, no relabelling<br>stage 2: winner relabels all 123,579 pairs, replacing the `rerank` column<br>stage 3: re-run R7-H50 unchanged against the new teacher<br>hardware: RTX PRO 6000 96 GB, so up to ~8B dense is in scope for the teacher

**R7-H56 the distillation objective is mismatched to the metric (pre-registered)**

R7-H50 trains with `BCEWithLogitsLoss` against the teacher's sigmoid output and is scored by AUC. Those are different objectives. AUC is a pure ranking statistic - it depends only on whether a supported claim outscores an unsupported one - while BCE is pointwise calibration, penalising a prediction of 0.61 against a target of 0.73 even when every ordering is already correct. The training signal spends capacity on a property the metric ignores.

The saturation problem compounds it. The teacher's most informative pairs are the confident ones, and BCE against probabilities near 0 or 1 carries almost no gradient there, so the clearest teacher judgements transfer the most weakly.

Serving makes it a ranking problem twice over: the verdict is `max-over-chunks`, so what actually matters is whether the BEST chunk of a supported claim outranks the BEST chunk of an unsupported one. That is a claim-level ordering objective, and nothing in the current loss expresses it.

- Hypothesis - because the evaluation metric is a ranking statistic and the serving rule is max-over-chunks, replacing pointwise BCE on the teacher's probability with (a) MSE on the teacher's LOGITS and (b) a claim-level pairwise ranking term will raise the distilled student's AUC by at least 0.02 at identical teacher, data, architecture and budget
- Lever - the loss function only; teacher, corpus, splits, architecture, optimiser and step count all held at R7-H50 values
- Mechanism - logit-space MSE preserves the teacher's full dynamic range where the sigmoid has compressed it, which is the standard reason distillation is done on logits rather than probabilities; the ranking term optimises the quantity the metric actually reports
- Arms to compare, one training run each on the same 385 training traces:
  - **A** BCE on teacher probability - the R7-H50 baseline, AUC 0.8269
  - **B** MSE on teacher logits, temperature 1
  - **C** MSE on teacher logits with temperature T ∈ {2, 4} to soften the target
  - **D** B plus a pairwise margin-ranking term over positive/negative claim pairs sharing a batch
  - **E** D plus the 2,752 hard gold labels as an auxiliary claim-level term, weighted low
- Prediction - B beats A by 0.01-0.02; D beats A by 0.02-0.04; E adds under 0.01 because 2,752 claim-level labels are too few to move a pair-level objective
- Acceptance bar - the best arm must clear A by ≥ 0.02 AUC on the held-out 159 traces to justify the added complexity; below that, BCE stays for simplicity
- Kill condition - if all arms land within 0.01 of A, the objective is not the constraint and attention returns to the teacher (R7-H55) and the data
- Experiment - `experiments/grounding-semantic/R7-H56_distill_objective.py`, mmBERT-small only (the cheapest arm at 4.92 ms/pair and the current best student), five runs, same seed, same splits

**Sequencing note for R7-H55 and R7-H56.** These are independent levers on the same deficit - the teacher sets the ceiling, the objective sets how much of it transfers - and they must be run SEPARATELY before any combination, or a gain cannot be attributed. R7-H56 is the cheaper of the two (no relabelling pass) and runs first.

**R7-H57 does a public-data verifier transfer to our distribution? (pre-registered)**

The dataset survey (`reports/research-grounding-datasets.md`) recommends training on RAGTruth, LettuceDetect-prose and RAGBench, and estimates a 50-100x expansion in distinct source documents against our 619. It also warns that domain transfer will disappoint, because none of those corpora are production assistant traffic over a private corpus.

That warning is testable at zero training cost. `KRLabsOrg/lettucedect-v2-mmbert-base` (MIT, 307M, mmBERT-base backbone) is trained on exactly the recommended stack. Running it on OUR gold measures public → private transfer directly, before a single GPU-hour is committed to training on public data.

- Hypothesis - because the recommended public corpora differ from our traffic in register, document structure and retrieval failure mode, a verifier trained on them will underperform our cascade on the English slice while transferring better to the non-English slices, where our own 639-trace supervision is thinnest
- Lever - the training distribution, held at arm's length: no training, no fine-tuning, frozen public checkpoint against our gold
- Prediction - English AUC 0.65-0.75 against our cascade's 0.8619 on held-out traces; non-English closer, within 0.05 of its own English figure, because its training data covers 14 languages and ours effectively covers one
- Acceptance bar - if it lands within 0.05 of our cascade on ANY language slice, public training data transfers and the RAGTruth track is justified; if it collapses below 0.65 everywhere, public data is a different problem and the budget belongs on annotating private traces instead
- Why this is the right probe - it is the reverse direction of the expensive experiment. Training on public data and testing on ours costs a day; taking a model someone already trained on public data and testing on ours costs an inference pass
- **Format audit first, per the standing rule** - LettuceDetect is a token tagger over (question, context, answer) triples, not a pairwise classifier. The mapping onto our (claim, chunk) task must be read off its model card and verified on a positive control before any number is taken. Round 4 lost a day to exactly this
- Experiment - `experiments/grounding-semantic/R7-H57_public_verifier_transfer.py`, our 2,752-claim gold, per-language breakdown, run on the idle RTX 5000 Ada so it does not contend with the depth probe

**R7-H58 does Search Arena carry its evidence? (pre-registered, 20-minute gate)**

`lmarena-ai/search-arena-24k` is the only public corpus of genuine retrieval-augmented user traffic: 24,069 multi-turn search-assistant conversations, ~11k users, 136 countries, ~90 languages. It is the only candidate that broadens the TRACE distribution rather than the document distribution, which is our actual constraint at 639 traces.

Its usefulness rests entirely on one unverified fact: whether `system_{a,b}_metadata.web_search_trace` contains scraped page TEXT or merely URLs and citation counts. A conversation without its evidence cannot be grounded. The HF dataset viewer fails at 400 MB, so nobody has confirmed it.

- Prediction - the trace carries URLs and citation metadata but not full page text, because storing scraped page bodies for 24k multi-turn conversations is a large payload and the schema mentions a scrape engine without a text field
- Acceptance bar - page text present → Search Arena becomes the primary Class B target and a licence read follows; text absent → the Class B lane collapses to MIRAGE-Bench and NoMIRACL, both of which are Apache-2.0 with passages attached
- Cost - one shard, inspection only, no training and no GPU
- Note - even on a PASS, the licence is unresolved: prompts are CC-BY-4.0 but model outputs are "governed by respective provider terms", which is a legal question rather than an engineering one and must not be answered by this project unilaterally

### Round 7 results - part 2

**R7-H50 capacity ablation - Refuted (the task IS capacity-limited), and the first version was void**

Recorded in full because the correction matters more than the result. Three versions ran:

| version | split | mmBERT-small 140M | mmBERT-base 307M | status |
|---|---|---|---|---|
| v1 | by CLAIM | AUC 0.8772 | AUC 0.8573 | **VOID - leaked** |
| v2 | by TRACE | **AUC 0.8269** | **AUC 0.8479** | recorded |

- **v1 leaked at the document level and INVERTED the conclusion.** The gold's 2,752 claims come from only **639 traces sharing 619 source documents** (4.31 claims/trace, largest trace 35). A claim-level split puts claim A of a trace in train and claim B in test, so the student trains on the very document it is scored on. Under that leak the 140M model appeared to BEAT the 307M model; with trace-level splits the ordering reverses. The leak was worth **0.050 AUC** and would have sent us building the wrong model
- The log's own situational overview had warned about exactly this - "claims sharing a trace's evidence are correlated, so the effective sample is smaller than the record count" - and the first version ignored it. The effective sample is **639, not 2,752**
- **v2 protocol** - split by `trace_id` 60/15/25, threshold fitted on the VALIDATION traces and applied unchanged to test (v1 fitted it on test, which inflated macro-F1), plus warmup, gradient clipping and a divergence guard
- **Reproducibility check** - mmBERT-base-22L was trained three times across runs: **0.8479 / 0.8485 / 0.8502**, spread 0.0023. Run-to-run noise is small, so the 0.021 gap between 140M and 307M is signal
- **Verdict** - **Refuted.** The pre-registered bar was a macro-F1 spread ≤ 0.02 meaning "not capacity-limited"; measured AUC spread is **0.021** and it points UP with size. Capacity matters, mildly. Architecture work is not closed
- **mDeBERTa-v3-base could not be trained** - diverged to NaN at step 200 at lr 2e-5, and again at lr 1e-5 with warmup and clipping. It is excluded rather than fought; the depth arm moved to mmBERT truncation, which trains stably

**Matched-teacher comparison, the number that matters.** All on the identical 159 held-out traces / 717 claims / base rate 0.654:

| model | params | AUC |
|---|---|---|
| `bge-reranker-v2-m3` (teacher, frozen) | 568M | **0.8619** |
| mmBERT-base distilled | 307M | 0.8479 (-0.014) |
| mmBERT-small distilled | 140M | 0.8269 (-0.035) |
| `mDeBERTa-v3` NLI entailment (frozen) | 278M | 0.7601 |

- A 307M student lands **0.014** from a frozen 568M teacher at 54% of the size, on one epoch over 40,000 pairs from 385 traces. The earlier "student beats teacher by +0.048" was **entirely leakage**; clean, the student loses
- Both students already beat the 278M NLI model the cascade ships, by 0.067 and 0.087

**R7-H50 depth probe - depth is purchasable, not idle (2 of 4 arms)**

mmBERT-base truncated within one family, so width, tokenizer, embeddings and recipe are all fixed and only depth moves. This is the arm mDeBERTa was meant to provide.

| depth | params | AUC | macro-F1 | ms/pair |
|---|---|---|---|---|
| 22L | 307.5M | 0.8502 | 0.783 | 9.20 |
| 11L | 252.4M | 0.8183 | 0.751 | **5.38** |
| 6L | pending | | | |
| 3L | pending | | | |

- Halving depth costs **0.032 AUC** and buys **1.7x** speed. Depth is doing real work - it is not idle, and the wide-shallow thesis does not get a free win on quality
- R7-H52 measured that width is FASTER than depth at matched parameters; this measures what depth is WORTH. The two together make the shape a genuine trade rather than a free lunch, and the trade cannot be priced until an accuracy floor is set

**R7-H57 public-trained verifier on our gold - PARTIAL, and my prediction was wrong**

`KRLabsOrg/lettucedect-v2-mmbert-base` (MIT, 307M, trained on RAGTruth + its translations + LettuceDetect-prose).

- **Positive control: AUC 1.0000, accuracy 1.00** on the same 20 trivially separable pairs where Baguettotron scored 60% and Pleias-RAG-350M 55%. The model is competent and the harness is validated, which is what gives the negative result below its force
- **On our gold: AUC 0.7095, macro-F1 0.6313** against our cascade's 0.8619 on the same held-out traces - a deficit of **-0.152**
- **Prediction refuted.** I predicted non-English would transfer BETTER than English; measured EN 0.7038 against a non-EN mean of 0.4030
- **But the non-English slices cannot support that finding and are not reported as one.** They run n=26 to 44 at base rates 0.885-0.973 - the Norwegian slice has roughly ONE negative claim, so its 0.1389 is noise, and Norwegian is not in LettuceDetect's training languages at all. **English is the only slice here with enough negatives to measure**
- **This exposes a defect in our own gold, not just in the candidate** - our non-English evaluation is unusable in either direction, at any sample size. That is a bigger problem than which corpus to train on

**R7-H59 cross-domain transfer matrix - both directions fail, asymmetrically**

1,200 RAGTruth test responses, grounded rate 0.654 against our gold's 0.649, so class balance is controlled. Threshold fitted on half A, reported on half B, for every cell.

| model | our private gold | RAGTruth (public) |
|---|---|---|
| our cascade (reranker) | **0.8619** | **0.6432** · F1 0.5947 |
| `lettucedect-v2` (public-trained) | 0.7095 · F1 0.6313 | 0.7039 · F1 0.6381 |

- **Our cascade collapses off-domain: -0.219.** Predicted 0.75-0.80, measured 0.643 - wrong again, in the same optimistic direction. The fitted threshold came out at **0.983**, i.e. the reranker pushes nearly everything to the top of its range on foreign data: calibration failure, not just accuracy loss
- **LettuceDetect is FLAT: 0.7095 versus 0.7039**, a difference of 0.006. Trained on public data, it performs identically on ours. Domain-general but capped near 0.70
- **On RAGTruth it beats us**, 0.7039 to 0.6432
- **Our 0.8619 is domain specialisation, not general grounding capability.** It does not travel and must not be quoted as if it does
- **This answers R7-H49 from a side door while that stays blocked on Hub auth.** A neutral third-party model finds both corpora equally hard (0.7095 vs 0.7039), so our gold is **not unusually easy** - the benchmark survey's suspicion on that point is refuted. But its *prediction* was right: it forecast 63-72 for us on public benchmarks and our cascade scored **64.3** on RAGTruth, itself an LLM-AggreFact component. Both hold - the gold is fair, and the score does not transfer
- **Strategic consequence** - transfer fails in BOTH directions, so the ~140k RAGTruth examples are unlikely to lift our gold number. Our advantage is domain specialisation, which makes more PRIVATE traces the lever, not more public rows. The dataset survey's warning was right and understated

**Datasets fetched** - `data/external/datasets/`, via `scripts/fetch_grounding_datasets.py`. RAGTruth (15,090 train / 2,700 test, MIT) and its 7 translations (124,530 rows, MIT). Sidecars are tracked in git and travel inside each archive; the archives are gitignored. The survey covering all eight licence-clean corpora, and the four large ones excluded on licence (TrueTeacher, HaluBench, MS MARCO, ANLI) plus MEMERAG which forbids training in terms, is `reports/research-grounding-datasets.md`.

## Hypothesis round 8 - beat LettuceDetect on all three corpora (pre-registered)

Batch **M**, ids `R8-H61..R8-H68`. Round 7 replaced a vague target with a measured one. `KRLabsOrg/lettucedect-v2-mmbert-base` is 307M, MIT, in-band, and it is the incumbent to beat because it is the only public model measured on all three corpora under one harness.

### The objective, stated as three numbers

| corpus | LettuceDetect | our cascade | gap |
|---|---|---|---|
| private gold, 159 held-out traces | 0.7095 (F1 0.6313) | **0.8619** | **+0.152 - we win** |
| RAGTruth EN, 1,200 responses | **0.7039** (F1 0.6381) | 0.6432 (F1 0.5947) | **-0.061 - we lose** |
| RAGTruth non-EN, mean of 7 languages | **0.6095** | 0.5626 | **-0.047 - we lose** |

**Win condition: strictly above all three, simultaneously, with one model.** Beating any one in isolation is easy and uninteresting - a domain-specialised model already wins the first, and a public-trained model already wins the other two. The whole difficulty is holding the +0.152 on our own domain while closing -0.061 and -0.047 elsewhere, and the two objectives pull against each other.

### Methodology - what round 8 adds

- **Naive baseline** unchanged at macro-F1 0.417. Every number above is AUC on the harness in `R7-H59` / `R7-H60`, threshold fitted on half A and reported on half B
- **NEW - three-corpus reporting is mandatory.** No hypothesis in this round may report a single number. A result on one corpus without the other two is not a result, because round 7 established that our headline transfers to neither
- **NEW - the multilingual labels are suspect and gate the third objective.** In `R7-H60` both models degraded by the SAME amount on the translations (-0.091 ours, -0.094 theirs) even though LettuceDetect was TRAINED on those very train splits. Training should have protected it and did not, which points at machine-translated label alignment rather than model capability. `R8-H61` settles this before any multilingual work is funded
- **Held fixed** - the R7-H51 teacher corpus, trace-level splits, max-over-chunks aggregation, and mmBERT-base as the student backbone unless a hypothesis explicitly varies it

### Pre-registration

| id | persona | technique | prediction | acceptance bar |
|---|---|---|---|---|
| R8-H61 | deflationist | verify translated labels on the 300 human-checked German rows | the -0.09 drop shrinks by more than half on verified labels | drop halves → the multilingual objective is mis-specified and its bar is re-derived |
| R8-H62 | follower | multi-corpus distillation - our teacher labels + RAGTruth EN + 7 translations | gold 0.83-0.86, RAGTruth EN 0.70-0.75, non-EN 0.62-0.67 | all three above the bars, gold ≥ 0.84 |
| R8-H63 | mechanist | per-corpus rank normalisation before thresholding | recovers most of the F1 gap at zero training cost | F1 +0.05 on RAGTruth with AUC unchanged |
| R8-H64 | hybridizer | ensemble our cascade with LettuceDetect | if their errors decorrelate, the mean beats both everywhere by construction | beats both on all three; error correlation reported |
| R8-H65 | scout | translate-then-verify via the shipped argos MT bridge | non-EN 0.5626 → within 0.03 of the English figure | non-EN ≥ 0.6095 using English-only inference |
| R8-H66 | contrarian | token-level span supervision instead of pairwise scores | span supervision is what makes LettuceDetect domain-general, not its data | RAGTruth EN ≥ 0.70 while gold holds ≥ 0.84 |
| R8-H67 | heretical | do not train one model - route by domain and language | routing beats any single model on all three trivially | tests whether "one model" is even the right objective |
| R8-H68 | mechanist | near-miss negatives generated from OUR corpus, VitaminC-style | keeps domain, adds hard negatives, no licence exposure | gold ≥ 0.87 and RAGTruth EN ≥ 0.68 |

### Sequencing

**R8-H61 runs first and can re-specify the round** - if the translated labels are unreliable, the -0.047 multilingual bar is measuring translation noise and must be re-derived before anything is trained against it. **R8-H63 and R8-H64 are next because neither requires training**: rank normalisation is arithmetic, and the ensemble is two frozen models already scored on all three corpora, so their error correlation is a free re-analysis of data already on disk. R8-H65 reuses the MT bridge already shipped in the lexical tier. Only then do the training hypotheses run - R8-H62 as the straightforward multi-corpus baseline, R8-H66 and R8-H68 as the two mechanistic bets about WHY LettuceDetect generalises. R8-H67 is deliberately last: if a router beats every trained model, the round's premise was wrong and that is worth knowing after the honest attempts, not before.

### Amendment - round 8 extended with anisotropy removal

Append-only. Nothing recorded above changes; these are added hypotheses, and the round's centre of gravity is restated: **distillation is the backbone** (R8-H62, H66, H68), fine-tuning is the optional second stage, and anisotropy removal is added as a third family because round 7 produced a specific symptom that names it.

**The symptom.** In R7-H59 the reranker's fitted threshold on RAGTruth came out at **0.983** - it pushes nearly every off-domain pair to the top of its range. That is dynamic-range collapse, not merely accuracy loss, and it is the pathology all-but-the-top postprocessing was built for (Mu and Viswanath, 2018: subtract the mean, then project out the top-D principal components). The same track's prior art applies - the docdistance work used exactly this to widen compressed statement cosines.

**Why the cross-lingual version is the most promising of the three.** Multilingual encoders place different languages in different cones of the representation space, so a per-language mean is a nuisance direction rather than signal. If the -0.047 multilingual gap is partly that offset, it is removable by arithmetic with **no training and no labels** - which would be the cheapest result in the round by a wide margin.

| id | persona | technique | prediction | acceptance bar |
|---|---|---|---|---|
| R8-H69 | mechanist | all-but-the-top on the bi-encoder gate - subtract the mean, project out top-D components, D ∈ {1,2,3,5} | cosine dynamic range widens ≥ 1.5x; gate AUC 0.73 → 0.78-0.83 | gate AUC +0.03 on all three corpora at some single D |
| R8-H70 | scout | per-LANGUAGE centroid removal for cross-lingual alignment | languages occupy separate cones; removing the per-language mean aligns them | non-EN AUC 0.5626 → ≥ 0.6095 with NO training, on the parallel corpus |
| R8-H71 | hybridizer | anisotropy removal on the distilled student's pooled representation before the head | the student inherits the teacher's compressed geometry | ≥ +0.02 AUC over the same student untreated, on all three |
| R8-H72 | mechanist | score-space recalibration - per-corpus rank normalisation of the FINAL score | strictly weaker than H69-H71 but nearly free; separates score compression from representation compression | isolates whether the 0.983 threshold is a scoring or a representation artefact |

**Ordering within the round.** R8-H70 runs first of these four: it is training-free, label-free, targets the objective we are furthest from, and its result tells us whether the multilingual gap is geometry or capability. R8-H69 follows because it is also training-free and reuses the same embedding pass. R8-H72 is arithmetic on the R8 substrate and costs nothing. R8-H71 requires a trained student and therefore waits on R8-H62.

**Honest note on what anisotropy removal cannot do.** It is a postprocessing transform on a frozen representation. If the models genuinely cannot read non-English evidence - as opposed to reading it into a displaced subspace - projecting out components will not create the capability, and R8-H70 will return flat. A flat result there is informative: it moves the multilingual objective squarely into training, where R8-H62 and R8-H66 live.

### Amendment - round 8 extended with a multi-head adversarial trunk

Append-only, prompted directly by the R8-H64 result: our cascade and `lettucedect-v2` are **almost perfectly decorrelated** (Spearman -0.046 to +0.083 across all ten corpora), and rank-average fusion therefore beats BOTH on RAGTruth EN (0.7479 vs 0.7030) and non-EN (0.6212 vs 0.6095) with no training at all.

That is the finding these hypotheses generate from. Fusion is not shippable - it is 568M + 307M and fails the sub-400M single-model requirement outright - but it proves the two signals exist, are complementary, and are individually learnable. The task becomes putting both inside ONE trunk.

**The design.** One mmBERT-base trunk (307M, in-band) carrying two supervision heads plus an adversarial third:

- **Head A - relevance/entailment**, regressing our cascade's soft per-pair score. This is the signal the reranker reads and the one our domain advantage lives in
- **Head B - span faithfulness**, token-level classification against RAGTruth's human span annotations. This is the signal LettuceDetect reads, and R8-H64 says it is nearly orthogonal to Head A
- **Head C - adversarial discriminator** predicting language or corpus through a GRADIENT REVERSAL layer. The trunk is trained to make that prediction impossible, which is the standard domain-adversarial construction and the literal GAN-like element: discriminator and trunk optimise opposing objectives
- **Then post-alignment**, the trunk is frozen and a fresh classification head is fitted on the aligned representation

| id | persona | technique | prediction | acceptance bar |
|---|---|---|---|---|
| R8-H73 | hybridizer | two heads on one trunk - soft-score regression + token-span tagging | internalises the fusion; the trunk must represent both decorrelated signals | beats the better single head on ALL three corpora, and lands within 0.02 of the external fusion |
| R8-H74 | heretical | adversarial LANGUAGE discriminator with gradient reversal | a language-invariant trunk transfers English grounding to the other 7 | non-EN 0.5626 → ≥ 0.66 without adding non-English supervision to Head A |
| R8-H75 | mechanist | adversarial CORPUS discriminator (gold vs RAGTruth) with gradient reversal | domain-invariance closes the -0.061 cross-domain gap | RAGTruth EN ≥ 0.75 while gold holds ≥ 0.84 |
| R8-H76 | follower | post-alignment head - freeze the aligned trunk, fit a fresh classifier | the aligned representation is better than the jointly-trained head that produced it | ≥ +0.01 over the joint head on all three; also gives a cheap per-domain re-fit |

**Why the adversarial arm is the interesting bet, and where it can fail.** R8-H61 established that roughly two thirds of the multilingual gap is genuine capability rather than translation-label noise, so it will not be fixed by cleaning labels. Gradient reversal attacks it from the representation side and needs NO additional non-English grounding supervision - only language tags, which are free. The failure mode is equally clear and must be watched: an adversarial objective strong enough to erase language also erases meaning, so the discriminator weight lambda is the load-bearing hyperparameter and a lambda sweep is part of the experiment rather than an afterthought. A trunk that becomes language-invariant AND grounding-blind is the expected failure, not a surprise.

**Ordering.** R8-H73 first - the two-head trunk without adversarial training is the honest baseline, and if it alone captures the fusion gain the adversarial complexity is unnecessary. R8-H74 and R8-H75 add one discriminator each, separately, never together, so a gain can be attributed. R8-H76 runs last on whichever trunk wins.

### Round 8 results - part 1 (training-free)

**R7-H50 depth probe - complete. Depth is load-bearing; the wide-shallow shortcut is closed on quality.**

mmBERT-base truncated within one family - width, tokenizer, embeddings and recipe all fixed, only depth moves.

| depth | params | AUC | macro-F1 | ms/pair |
|---|---|---|---|---|
| 22L | 307.5M | **0.8502** | 0.7830 | 9.20 |
| 11L | 252.4M | 0.8183 | 0.7510 | 5.38 |
| 6L | 227.3M | 0.7332 | 0.6645 | 3.16 |
| 3L | 212.2M | 0.6093 | 0.5996 | 1.75 |

- The curve **accelerates downward**: -0.032 for the first halving, -0.085 for the second, -0.124 for the third. At 3 layers the model is near chance for this task
- Pre-registered bar was a spread ≤ 0.02 meaning "not capacity-limited". Measured macro-F1 spread **0.1834** → **REFUTED**, decisively
- **R7-H52 measured that width is FASTER than depth; this measures what depth is WORTH, and it is worth a lot.** The two together make the shape a real trade, not a free win. 11L is the interesting operating point - 0.8183 at 5.38 ms, still +0.109 over the incumbent on gold at 1.7x the speed
- Verdict - **keep 22 layers.** Recommended spec revised: depth is not the place to economise

**R8-H61 multilingual label gate - the gap is REAL, and my hypothesis was half right**

`KRLabsOrg/ragtruth-de-translated-manual-300` is the only human-VERIFIED translation data that exists.

| corpus | n | base | cascade | lettuce |
|---|---|---|---|---|
| RAGTruth EN | 600 | 0.650 | 0.6537 | 0.7030 |
| RAGTruth DE, machine-translated labels | 600 | 0.650 | 0.5791 | 0.6032 |
| RAGTruth DE, human-VERIFIED labels | 300 | 0.663 | 0.4634 | **0.6351** |

- LettuceDetect's EN→DE drop is **0.0999** on MT labels and **0.0680** on verified labels. The drop shrinks by about a third but does NOT halve
- **Verdict - the pre-registered bar is not met: roughly one third of the multilingual gap is translation-label noise, two thirds is genuine capability.** The -0.047 multilingual objective stands, with that asterisk. Cleaning labels will not fix it and training must
- Caveat recorded: the verified subset is 300 rows against 600, different sample, so the two drops are not perfectly matched

**R8-H64 ensemble headroom - the two models are ORTHOGONAL, and fusion beats both**

Per-example scores from the R8 substrate, rank-normalised before fusing because the two scores live on different scales.

| corpus | cascade | lettuce | Spearman | MEAN-fusion |
|---|---|---|---|---|
| RAGTruth EN | 0.6537 | 0.7030 | +0.073 | **0.7479** |
| RAGTruth de | 0.5791 | 0.6032 | -0.001 | 0.6310 |
| RAGTruth fr | 0.5650 | 0.6118 | -0.010 | 0.6244 |
| RAGTruth es | 0.5986 | 0.5912 | +0.049 | 0.6377 |
| RAGTruth it | 0.5386 | 0.5901 | -0.046 | 0.5888 |
| RAGTruth pl | 0.5503 | 0.6020 | -0.002 | 0.6057 |
| RAGTruth hu | 0.5758 | 0.6032 | -0.046 | 0.6272 |
| RAGTruth cn | 0.5310 | 0.6653 | +0.033 | 0.6336 |
| **non-EN mean** | 0.5626 | 0.6095 | ~0.00 | **0.6212** |

- **Spearman correlation is effectively ZERO on every corpus.** A cross-encoder trained on relevance and a token tagger trained on span faithfulness read genuinely different evidence. This is the round's most useful mechanistic finding
- Fusion clears two of the three bars with **no training**: RAGTruth EN 0.7479 vs 0.7030 (+0.044), non-EN 0.6212 vs 0.6095 (+0.012)
- **It is NOT the deliverable** - 568M + 307M fails the sub-400M single-model requirement outright. What it establishes is that the signal to beat the incumbent exists, is complementary, and is individually learnable. That is what R8-H73 tries to place inside one trunk
- **Defect found and recorded: the `gold` rows of this analysis are VOID.** `R8_score_substrate.our_gold()` sliced `chunk[:semantic_top_k]` - the first three chunks in dataframe order, i.e. arbitrary evidence rather than retrieved evidence - and read the cascade at 0.6739 against the 0.8619 R7-H50 measured by taking max over ALL of a claim's chunks. Gold claims carry a mean of **36.3** chunks, so the cascade was given 8% of its evidence. Fixed to use every chunk; the gold cells above are excluded rather than corrected in place, and the substrate is being re-scored
- This is the same class of error as the R7-H50 claim-level leak: a harness detail that silently changes the task rather than failing loudly

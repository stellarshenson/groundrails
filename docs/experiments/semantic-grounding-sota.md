# Semantic grounder - final design

**Canonical SOTA Document**

The consolidated, deployable semantic grounder: classify each claim as supported or hallucination using **two cross-encoders only** - a relevance reranker and an NLI entailment model - combined by a small logistic over their max-over-chunks scores, with a bi-encoder pre-filtering chunks to top-k in front and three adopted skip mechanisms - a stage-0 cosine gate on the pre-filter's own scores, a reranker-first cascade that skips the NLI when the reranker is decisive, and a rank-ordered early-exit inside the reranker stage - cutting warm latency 45% vs always-both at held quality (full-gold end-to-end: 585 ms mean / 258 ms median per claim, macro-F1 0.789 - see Performance characteristics). No bi-encoder *signals*, no lexical features, no fine-tuning. All models are served as **OpenVINO int8 on a single runtime** (see Serving). This is the pure model-based (semantic) pipeline; the deterministic lexical track lives separately in `lexical-grounding-sota.md` (this repo). Research record and ablation in `semantic-grounding-experiments.md`.

Rounds 5-8 did not change what ships. They produced a **candidate successor** - one distilled cross-encoder replacing all three models - and they corrected what the shipped numbers mean: macro-F1 0.789 is a private-gold figure that does not transfer, and the private gold cannot measure generalisation at any sample size. Evidence for both is in `semantic-grounding-experiments.md` rounds 5-8; the design consequences are in Candidate successor and Measurement validity below.

> Numbers are the 2,752-record organic-majority gold, out-of-fold 5-fold. That gold is one private corpus, and its figures measure deployment fit rather than general capability - the same reranker reads AUC 0.8619 here and 0.6432 on public RAGTruth (see Measurement validity).

## Status

Two designs in this document, one deployed.

- **Ships** - the two-cross-encoder OpenVINO int8 cascade described below, macro-F1 **0.789** end-to-end / **0.796** out-of-fold on the private gold, warm mean 585 ms per claim
- **Candidate, not shipped** - a single distilled `mmBERT-base` cross-encoder (307.5M) that beats the public incumbent 3/3 on corpora it had training exposure to and loses 0.0505 AUC on the one corpus neither model saw
- **The gate it has not passed** - a genuinely blind arena; incarnation R8-H83 is still training and no incarnation has cleared one, so nothing here supersedes the cascade
- **Unchanged by rounds 5-8** - every shipped component, threshold, latency and footprint figure below
- **Number discipline** - the experiments log's rounds 5-6 quote "the shipped cascade, macro-F1 0.824"; 0.824 belongs to the retired 6-model + lexical stack, not to this design, whose figures are 0.789 end-to-end and 0.796 out-of-fold

## Architecture decision

Ablation on the verified gold sets the bar by the full model's CV noise (full 6-model + lexical = macro-F1 0.814, ±0.014 fold std). The two-cross-encoder semantic stack holds within that noise while dropping four models and the lexical layer.

| configuration | models | OOF AUC | macro-F1 | note |
|---|---|---|---|---|
| full (6 models + lexical) | 6 | 0.902 | 0.814 | maximum achieved |
| 2 cross-encoders + lexical | 2 | 0.885 | 0.801 | lexical adds +0.005 - excluded (lexical track) |
| **semantic-only (2 cross-encoders)** | **2** | **0.876** | **0.796** | **shipped** - within ~1 fold-std of full |
| reranker alone | 1 | 0.840 | 0.757 | NLI adds +0.039 - essential |

- **Both cross-encoders are load-bearing** - the reranker alone is macro-F1 0.757; adding NLI lifts it to 0.796 (+0.039, far outside noise). Neither can be dropped
- **The four bi-encoders are pure footprint as signals** - together they add only +0.013 macro-F1 (in-noise); dropped as grounding signals. **bge-m3 is retained in the pipeline as the top-k pre-filter** (ranking chunks before the cross-encoders), not as a signal feeding the logistic
- **Lexical excluded by scope** - the numeric/entity contradiction + match-type flags would add +0.005, but they belong to the lexical track; the semantic pipeline is models-only
- **No-degradation margin is thin** - 0.796 is 0.018 below the full 0.814, just outside one fold-std; the quantized + pre-filtered pipeline measured end-to-end on the full gold holds 0.789 (see Performance characteristics)

## Signal ranking

Out-of-fold AUC on the 2,752 gold. The two cross-encoders carry the signal; everything below is dropped from the deployed pipeline.

| signal | kind | OOF AUC | status |
|---|---|---|---|
| bge-reranker-v2-m3 | cross-encoder rerank | **0.841** | **kept** - the lever |
| mDeBERTa-v3-base-mnli-xnli | NLI cross-encoder | 0.806 | **kept** - additive, essential |
| bge-m3 | bi-encoder | 0.730 | dropped (footprint) |
| e5-small | bi-encoder | 0.635 | dropped |
| e5-large | bi-encoder | 0.621 | dropped |
| mmBERT-base | bi-encoder | 0.529 | dropped |

- **These AUCs were taken under a chunking this repo can no longer reproduce** - the cached pair set averaged 40.62 chunks/claim and `recursive_chunk` no longer reproduces it. Regenerated under current chunking (123,579 pairs, 44.9 chunks/claim) the reranker reads **0.8289** (-0.012 vs 0.841) and NLI entailment **0.7674** (-0.039 vs 0.806). The ranking is unchanged; the absolute figures above are superseded for any new comparison
- **mmBERT-base's 0.529 is a bi-encoder figure, not a verdict on the backbone** - the same backbone trained as a cross-encoder is the candidate successor at 0.8531 AUC on the same gold

## Pipeline

Claim + retrieved evidence in → grounded-probability + verdict out. The gate keys on the **max score over chunks** (only the single best-supporting chunk matters), so the work is to find that chunk cheaply and score it well.

```mermaid
flowchart LR
    IN["Claim +<br/>retrieved chunks (~50)"] --> PRE["bge-m3 pre-filter<br/>claim embed + cosine rank<br/>top k=8, ~39 ms"]
    PRE --> G{"stage-0 cosine gate<br/>max cos vs [0.493, 0.739]"}
    G -->|"cos ≥ 0.739 (19.5%)"| PASS(["passed - supported"])
    G -->|"cos ≤ 0.493 (2.5%)"| FLAG(["flagged - hallucination"])
    G -->|"in between"| RR["bge-reranker-v2-m3<br/>early-exit max over pairs<br/>batches 1, 1, 2, 4"]
    RR --> B{"cascade band<br/>max s vs [0.01, 0.66]"}
    B -->|"s ≥ 0.66 (35.5%)"| PASS
    B -->|"s ≤ 0.01 (5.9%)"| FLAG
    B -->|"in-band (36.5%)"| NLI["mDeBERTa-v3 NLI<br/>entailment max"]
    NLI --> ST["logistic stack<br/>p(rr_max, nli_max) vs T_st"]
    ST -->|"p ≥ T_st"| PASS
    ST -->|"p < T_st"| FLAG

    style IN stroke:#0284c7,stroke-width:2px
    style PRE stroke:#a855f7,stroke-width:2px
    style RR stroke:#a855f7,stroke-width:2px
    style NLI stroke:#a855f7,stroke-width:2px
    style G stroke:#10b981,stroke-width:2px
    style B stroke:#10b981,stroke-width:2px
    style ST stroke:#10b981,stroke-width:2px
    style PASS stroke:#10b981,stroke-width:3px
    style FLAG stroke:#ef4444,stroke-width:3px
```

Stage shares on the edges are the measured composition of the full-gold end-to-end run; each claim takes exactly one exit, and cost rises left to right (gate exits ~24-59 ms, reranker exits ~211-681 ms, in-band ~1,253 ms).

1. **Broad first-stage retrieval** - keep retrieval broad at ~50 chunks per claim; this is the unrecoverable recall ceiling (reranking cannot resurrect a chunk it never sees)
2. **Top-k pre-filter (50 → k=8) + stage-0 cosine gate** - score the 50 chunks with the bi-encoder already warm in the retrieval path and keep the top k=8; the single biggest latency lever (turns 50×2 cross-encoder passes into 8×2). Tighten to k=5 only after validating the best chunk survives. Use the semantic bi-encoder, not BM25 (BM25 misses the paraphrase / negation the gate cares about). The max cosine this ranking already computed doubles as a **stage-0 gate**: `cos <= 0.493` → flagged, `cos >= 0.739` → passed - 22% of claims resolve here at embed cost (~39 ms) with strictly fewer errors than the cascade alone (`COSINE_GATE` in `grounding_openvino.py`)
3. **Two cross-encoders on the k survivors, reranker first (cascade + early exit)** - the reranker scores the k pairs **in pre-filter rank order with progressive batches (1, 1, 2, 4), stopping the moment the running max crosses the pass edge** (verdict-invariant: unscored pairs cannot change a final pass verdict; mean pairs scored 4.8/8, `rerank_max_early`). The max `s` against the band **[0.01, 0.66]** routes the claim: `s <= 0.01` → flagged as hallucination (NLI skipped), `s >= 0.66` → passed as supported (NLI skipped), in-band (~35% of claims after the gate) → the NLI runs and the logistic stack gives the verdict. Easy claims pay a fraction of one cross-encoder, uncertain claims pay both (`cascade_scores` / `rerank_max_early` in `grounding_openvino.py`, frontiers in `reports/grounding_hypotheses.md`). Batch all surviving claim×chunk pairs across the whole answer into one padded forward per model:
   - **bge-reranker-v2-m3** - one relevance logit per pair (XLM-R SentencePiece tokenizer, `tokenizer(pairs,...)`, max_length 512, no token_type_ids; sigmoid only if the cached path used `normalize=True`)
   - **mDeBERTa-v3-mnli-xnli** - 3-class entailment logits (entailment / neutral / contradiction)
   - length-bucket pairs before padding (10-30% saving), run the two models concurrently (separate CUDA streams / processes - independent until the logistic)
4. **Logistic verdict** - the small fitted logistic combines each model's max-over-chunks score into the grounded-probability; weights `bge_reranker +1.19, mdeberta_nli +0.90`
5. **Two-stage loop** - the semantic gate is Stage A; on a flag, one batched LLM judge (Stage B, the few flags only) confirms unsupported vs paraphrase, then the agent revises or retracts (1-2 iterations). Not a hard gate - a hard gate drops ~1-in-5 genuine paraphrased claims

## Operating point

One out-of-fold grounded-probability. macro-F1 is the driving metric; FP (supported flagged) and FN (hallucination missed) are the operational target.

| metric | semantic stack (2 cross-enc) | best single (bge-reranker) |
|---|---|---|
| OOF AUC | **0.876** ± 0.014 | 0.841 |
| macro-F1 | **0.796** | 0.757 |
| FP - supported flagged | 266 | 237 |
| FN - hallucination missed | 203 | 294 |
| recall (hallucinations caught) | 74% | - |
| false-flag rate | 14% | - |

- **The NLI is what cuts misses** - over the reranker alone the stack drops FN 294 → 203 (the reranker over-confirms paraphrased fabrications the NLI catches as non-entailed)
- **Two uses** - a low-false-flag point for pre-display blocking, a higher-recall point for the feedback-loop re-check; re-fit the threshold per deployment
- **Cascade thresholds, two band options** - the cascade adds two fixed band edges plus the reranker-only threshold. **Quality-neutral [0.01, 0.66]** (adopted): 60% NLI skip, strictly no-worse than baseline on both errors (FP 243 vs 244, FN 217 unchanged, int8 stack). **Low-false-flag [0.05, 0.32]**: 84% skip and FP 192 (−52) at FN 279 (+62, macro-F1 0.783) - for deployments where a false flag costs more than a miss. Frontier in `reports/grounding_hypotheses.md`
- **The band is empirical, not learned** - a threshold sweep, no fitted parameters: candidate edges at the reranker-score quantiles (19 points, 5th-95th pct), each (a, b) simulated on the out-of-fold scores into a skip-rate vs macro-F1 frontier; [0.01, 0.66] is maximal skip at zero measurable loss. Same calibration class as the operating threshold (frozen-model scores only); it lives in raw reranker-score space - re-sweep on any reranker, quantization, or evidence-distribution change
- **Stage-0 gate thresholds (round 2, adopted)** - the same sweep mechanism applied to the pre-filter's max cosine: `[0.493, 0.739]` resolves 22% of claims before any cross-encoder at FP 245 / FN 216 vs the cascade's 248 / 217 (strictly fewer errors). The gate needs no pure tails - it only has to agree with the cascade verdict on the claims it absorbs. Same empirical-threshold class; it lives in bge-m3 cosine space - re-sweep on any embedder or evidence change

## Performance characteristics (measured, end-to-end)

The full adopted serving path executed once over the entire 2,752-claim gold on the deployed LATENCY-hint int8 engines, warm regime (chunk vectors precomputed), deployed calibration applied frozen (logistic fit on the pair cache; T_rr=0.313, T_st=0.584, band [0.01, 0.66], gate [0.493, 0.739]) - `scripts/run_grounder_full.py`, log `logs/grounding-full-run.log`.

**How macro-F1 works here**: the verdict is binary (hallucination vs supported) on a 29% / 71% imbalanced gold. macro-F1 is the unweighted mean of the two per-class F1 scores - the hallucination-class F1 (penalising missed hallucinations and false flags) and the supported-class F1 - so neither class can dominate by count. A grounder that flags nothing scores macro ~0.42; one that flags everything ~0.22. Both error types carry operational cost (a missed hallucination misleads the end user, a false flag erodes trust in the assistant), which is why macro-F1 and not accuracy or single-class F1 drives every adoption decision.

**Quality (end-to-end serving, n=2,752)**:

| metric | OOF simulation (cached scores) | end-to-end serving | note |
|---|---|---|---|
| macro-F1 | 0.797 | **0.789** | -0.008, inside fold noise (±0.014) |
| FP (supported flagged) | 245 | 328 | false-flag rate 17% |
| FN (hallucination missed) | 216 | 172 | recall 72% → **78%** |

The error mix shifts toward recall in serving: the live reranker max is taken over the top-8 pre-filtered chunks while the calibration scores were max-over-all (~41) chunks, so serving scores sit slightly lower and more claims fall on the flag side; int8 batch-composition differences (progressive exit batches vs the cache's THROUGHPUT batches) add jitter. Net quality holds within noise - re-fit the thresholds on serving-derived scores if the FP/FN balance matters more than the macro.

**Latency (warm, per claim, full gold)**:

| | mean | median | p90 | p99 |
|---|---|---|---|---|
| per claim | **585 ms** | **258 ms** | 1,342 ms | 1,592 ms |

| stage path | share of claims | mean latency |
|---|---|---|
| gate-flag (cosine alone) | 2.5% | 24 ms |
| gate-pass (cosine alone) | 19.5% | 59 ms |
| rr-pass (reranker, early exit) | 35.5% | 211 ms |
| rr-flag (reranker, all pairs) | 5.9% | 681 ms |
| in-band (reranker + NLI + stack) | 36.5% | 1,253 ms |

The distribution is the design: 55% of claims finish under ~260 ms (gate or reranker pass), and the full ~1.3 s is paid only by the 37% of genuinely uncertain claims. Cold start (no chunk-vector cache) adds the evidence embedding, ~3 s/claim extra at 50 chunks - the warm numbers assume the pre-filter reuses the RAG retriever's vectors.

## Serving - single-engine OpenVINO int8

All three models run as **OpenVINO int8 IRs on one runtime** - embedder pre-filter, reranker, NLI. The choice is OpenVINO because **only NNCF can int8-quantize DeBERTa-v2**: NNCF SmoothQuant (migrate the disentangled-attention activation outliers into the per-channel weights) plus Fast Bias Correction holds the NLI signal, where ONNX-Runtime static int8 collapses it (see `deberta-v3-quantization-experiments.md`). The bge models quantize cleanly with plain NNCF int8. Build with `scripts/build_ov_grounder.py`; IRs land in `models/ov/<name>/` (IR + config + tokenizer, push-ready to HF).

| model | role | int8 method | parity vs fp32 | size |
|---|---|---|---|---|
| bge-reranker-v2-m3 | reranker | NNCF int8 | pearson **0.9976** | 571 MB |
| bge-m3 | bi-encoder pre-filter | NNCF int8 | pearson **0.9941** | 570 MB |
| mDeBERTa-v3-mnli-xnli | NLI | NNCF **SmoothQuant** (alpha 0.7) | pearson **0.9863** (full-gold **0.9841**) | 318 MB |

- **mDeBERTa int8 solved via SmoothQuant** - the stock dynamic-int8 ONNX was broken (pearson 0.35); NNCF SmoothQuant at alpha 0.7 reaches **0.9841 full-gold parity** and the re-fit stack holds **macro-F1 0.795** (vs fp32 0.796) at **318 MB** - no measurable quality loss, 3.6x smaller than fp32
- **ONNX-Runtime is not viable for the NLI** - even via a forked `onnx-neural-compressor` (crash bugs fixed) ORT static int8 of DeBERTa stays faithless (parity 0.61-0.62); hence the single engine is OpenVINO, not ORT
- **No re-fit needed** - all three int8 IRs correlate ~0.99 with the cached scores the logistic was fit on, so the fp32 calibration transfers
- **Load + score via the OpenVINO runtime** - `ov.Core().compile_model(read_model(xml), "CPU", {"PERFORMANCE_HINT":"THROUGHPUT"})`; feed `input_ids` + `attention_mask` only (no `token_type_ids`; mDeBERTa `type_vocab_size=0`). Entailment is index 0 of the 3-class head. Helpers in `experiments/grounding-semantic/grounding_openvino.py`
- **Portability** - x86-64 Intel/AMD native (int8 via AVX2 / AVX-512-VNNI); ARM (aarch64 / Graviton) via the OpenVINO ARM CPU plugin - functional but less mature, validate on the target. ORT is the more uniformly-portable engine but cannot quantize the NLI, so OpenVINO is the quality-driven choice; for x86 Lambda it is a clean fit

## Latency

All numbers are CPU OpenVINO int8 at k=8 under the `LATENCY` compile hint; warm = unique chunk vectors already cached. **Current deployed path** (stage-0 gate + reranker-first cascade + early-exit reranker), measured end-to-end on the full 2,752 gold: **mean 585 ms / median 258 ms / p90 1,342 ms / p99 1,592 ms** at macro-F1 0.789 - stage-path breakdown in Performance characteristics.

### Run history (regression reference)

One row per measured configuration, oldest first; compare any future re-measure against these to catch latency or quality regression. Absolute ms are CPU-load-sensitive (64-thread host) and samples differ per row - the per-row deltas and ratios are the stable signal, the full-gold run is the canonical reference. Re-measure by running `notebooks/07-kj-grounding-sota-benchmark.ipynb` (full gold, full metric set + confusion matrix + latency); benchmarks run as notebooks - create a new numbered notebook for any new bench configuration.

| run | config | sample | warm mean | median | p90 | quality | change |
|---|---|---|---|---|---|---|---|
| baseline k-sweep | always-both, no skips | latency-notebook sample | 1,238 ms | ~1.2 s | ~1.5 s | 0.822 macro (800-rec subset) | first int8 serving measurement - LATENCY hint adopted, k=8 chosen from the k-sweep |
| cascade bench (round 1) | + H11 cascade | 150 seed-0 | 1,184 → 857 ms | 759 ms | 1,280 ms | 0.795 OOF | reranker-first cascade adopted - NLI runs only in the uncertainty band [0.01, 0.66] |
| round-2 bench | always-both / + cascade / + gate + exit | 150 seed-0 | 1,206 / 869 / 662 ms | 1,165 / 782 / 593 ms | 1,515 / 1,329 / 1,384 ms | 0.797 OOF (gate + exit) | stage-0 cosine gate + early-exit reranker adopted (round 2) |
| full-gold end-to-end | deployed path | 2,752 gold | 585 ms | 258 ms | 1,342 ms | 0.789 end-to-end | full-gold validation of the adopted path with the frozen calibration |
| **SOTA benchmark notebook (current)** | deployed path | 2,752 gold | **492 ms** | **238 ms** | 1,103 ms | **0.789 end-to-end** | benchmark moved to the canonical notebook with the full metric set, confusion matrix and plots; verdicts identical to the script run across re-runs - latency deltas are host load |

The round-2 p90 ticks up 4% vs cascade-only - never-exit claims pay the progressive-schedule worst case; every other statistic moves down with each adopted mechanism.

### Cold vs warm and the top-k choice (historical always-both baseline)

Measured before the skip mechanisms on the always-both pipeline (`notebooks/04-kj-grounder-latency.ipynb`; quality column from `03-kj-openvino-grounder-pipeline.ipynb`, 800-record subset). Kept as the cold-regime and top-k evidence - the k=5/8/12/50 trade and the chunk-vector-cache lever have no newer measurement. The typical claim carries the full evidence set - **chunks/claim median 50, mean 40.6**.

Two regimes, distinguished by whether the chunk embeddings already exist when the claim arrives:

- **Cold** - nothing cached; the bi-encoder pre-filter must embed all ~50 evidence chunks *plus* the claim from scratch on every claim. Embedding the chunks dominates the time, so cost scales with chunk count and the top-k cut barely helps (k=8 vs all-chunks only 1.3×). This is the first time a chunk is ever seen
- **Warm** - the source-chunk embeddings are already cached (each unique chunk embedded once, keyed by content, ideally reusing the vectors the RAG retriever already computed during retrieval). The pre-filter then embeds **only the claim** and reuses the cached chunk vectors, so the only model work left is scoring the top-k pairs with the two cross-encoders - independent of chunk count. This is every subsequent claim that reuses already-seen chunks

| top-k | cold ms/claim | warm ms/claim | macro-F1 (subset) |
|---|---|---|---|
| 5 | 3764 | 864 | 0.811 |
| **8 (deployed k)** | **4293** | **1238** | **0.822** |
| 12 | 4947 | 1730 | 0.826 |
| 50 (all chunks) | 6074 | 6144 | 0.807 |

Per-claim distribution at k=8 - cold median 4.2 s / p90 5.4 s; **warm median 1.2 s / p90 1.5 s** (warm is tight because only the top-k pairs are scored, independent of chunk count). The warm column is the no-skip baseline - the deployed path's current numbers are in the run history above.

- **Cache the source-chunk embeddings (the design)** - each unique chunk is embedded once, keyed by content, and reused; ideally the pre-filter reuses the vectors the RAG retriever already computed for that chunk. This is the dominant lever: it cuts the typical claim **~4.2 s → ~1.2 s (3.6×)** and restores top-k as a real lever
- **Top-k is the lever once warm** - cold, k=8 vs all-chunks is only 1.4× (embedding dominates); warm, k=8 is **5.0×** faster than all-chunks (k=5 ~7×). Pre-filter aggressively once embeddings are cached; k=8 also slightly improves quality (drops noisy chunks)
- **Brute-force cosine, no ANN** - the pre-filter ranks the ~50 retrieved chunks with a numpy dot-product; FAISS / an ANN index only pays off for corpus-wide search (the retriever's job), not 50 chunks/claim
- **Warm latency is cross-encoder-bound** - the two large XLM-R / DeBERTa cross-encoders on the top-k pairs dominate once embedding is cached (stage means at k=8: pre-filter 38 ms, reranker 577 ms, NLI 569 ms); the three adopted skip mechanisms attack exactly that. A smaller pre-filter embedder, batching an answer's claims, or GPU fp16 (~0.15-0.4 s/claim) are the further levers. The CPU path fits async/background grading; inline blocking wants GPU or a smaller reranker

### CPU serving levers (measured)

Four mechanical levers tested on the gold (`scripts/bench_mechanical_levers.py`, 64-thread CPU); the cold/warm table above already reflects the `LATENCY` hint.

- **OpenVINO `LATENCY` hint - ~2× (the real win)** - `compile_ir` defaulted to `THROUGHPUT`, which spins up multiple async streams and is correct only for the batch/offline path; for inline single-claim serving it is **~2× slower** than `LATENCY` (the remeasure: cold k=8 dropped from ~8.8 s THROUGHPUT-era to **4.3 s**; isolated bench 6365 → 3048 ms at matched load). `LATENCY` dedicates all cores to the one request. Now the `compile_ir` default - free, no quality cost
- **`max_length` cap - void, do not lower** - chunks run ~300 tokens median / 418 p95 and (claim, chunk) pairs ~331 / ~590 p95, so the 512 cap **already truncates ~6.5% of pairs** - there is no headroom. Capping to 256 saves only ~17% and clips the median pair; `MAX_LEN` stays 512
- **Length-bucketing - kept, modest** - `rerank_max` / `nli_max` order a claim's chunks by length before batching so each padded batch wastes fewer cells; the max-over-chunks is order-invariant, so scores are unchanged. Small saving, no risk
- **k=5 vs k=8** - k=5 trims latency (warm 1.5 s vs 2.3 s) at ~0.011 macro-F1 (0.811 vs 0.822); a real but minor trade, validate the best chunk survives before tightening
- **Reranker-first cascade - adopted (-28% warm mean)** - run the reranker first and the NLI only when the reranker max falls inside the uncertainty band [0.01, 0.66]; 57% of serving claims skip the NLI (61% OOF) at macro-F1 0.795 (-0.002, inside noise). Measured warm at k=8: mean 1,184 → 857 ms, median -34%, p90 -14% (hard claims still pay both models). `cascade_scores` in `grounding_openvino.py`; bench `scripts/bench_grounder_cascade.py`
- **Stage-0 cosine gate - adopted (round 2)** - the pre-filter's own max cosine against [0.493, 0.739] resolves 22% of claims at embed cost before any cross-encoder, with strictly fewer errors than the cascade alone (FP 245/FN 216 vs 248/217). Zero added compute - the signal was being discarded. `COSINE_GATE` in `grounding_openvino.py`
- **Rank-ordered early-exit reranker - adopted (round 2)** - score the k pairs best-cosine-first in progressive batches (1, 1, 2, 4) and stop once the running max crosses the pass edge; verdict-invariant by construction (exact 150/150 on the bench). Mean pairs scored 4.8/8; the int8 forward is near-linear in batch rows (122 ms batch-1 vs 95 ms/pair batch-8) so exits keep what they save. Gate + cascade + exit together: warm mean **662 ms (-45% vs always-both)**. `rerank_max_early`; bench `scripts/bench_grounder_round2.py`
- **Fused-evidence single forward - refuted (round 2)** - packing the top evidence into ONE context and running ONE forward per cross-encoder (~211 ms/claim) collapses quality: macro-F1 0.714-0.784 across all configs. Max-over-chunks is load-bearing; do not approximate it in one forward
- **Joint-premise (SummaC) NLI - evaluated, not adopted** - joining the top-3 reranked chunks into one premise for a single NLI pass (the SummaC multi-premise aggregation) gave no aggregate macro-F1 lift on gold v3 (-0.002); it raised cross-lingual rejection (synthetic TNR +0.047) at a small English cost, so max-over-chunks stays the shipped NLI aggregation. Evidence: `joint-grounding-experiments.md` Round 4
- **Whole-answer batching - not yet done** - the serving helpers score **per-claim** (one padded forward per claim per model); batching all of an answer's claims × top-k into one forward per model would amortise fixed overhead. Needs an answer-level scorer, not a tweak - the next mechanical lever to build

## Footprint

All three deployed as OpenVINO int8 IRs:

- **bge-reranker-v2-m3** (XLM-R-large) - int8 IR **571 MB**
- **bge-m3** (bi-encoder pre-filter) - int8 IR **570 MB**
- **mDeBERTa-v3-base-mnli-xnli** (DeBERTa-v2 base) - SmoothQuant int8 IR **318 MB** (3.6× smaller than the 1.12 GB fp32)
- **Total ~1.46 GB** for the full single-engine grounder (embedder + 2 cross-encoders), all int8 - fits a Lambda-class container. The IRs are gitignored (`models/ov/`) and synced to S3 / pushed to HF rather than committed

## Candidate successor - one distilled cross-encoder

A single fine-tuned cross-encoder that replaces all three shipped models, distilled from the cascade's own reranker and mixed with public human labels. It is a candidate, not a shipped design - it has not passed a blind test.

- **Backbone** - `jhu-clsp/mmBERT-base`, 307.5M params, ModernBERT architecture, 22 layers x 768 hidden, intermediate 1152, 256k vocabulary, 8192 context
- **Supervision** - soft per-pair scores from the shipped `bge-reranker-v2-m3` over 40,000 private pairs, mixed with ~43,000 hard human-labelled public pairs (RAGTruth EN 15k plus ~4k per translation across 7 languages)
- **What it replaces** - the bi-encoder pre-filter, the reranker and the NLI, in one forward per (claim, chunk) pair; the max-over-chunks verdict rule is unchanged
- **Size discipline** - 307.5M is under the 400M ceiling and exactly size-matched to the public incumbent, so the comparison is not a capacity argument
- **Checkpoints** - `models/R8-H62-mmbert-multicorpus` (incarnation 1) and `models/R8-H78-mmbert-tabular` (incarnation 2), ~1.2 GB each fp32
- **Incumbent** - `KRLabsOrg/lettucedect-v2-mmbert-base`, 307M, MIT, the SAME mmBERT-base backbone with a token-classification head; the only public model measured on all three corpora under one harness

### Measured results

All figures AUC, threshold fitted on half A and reported on half B, harness `R7-H59` / `R7-H60` / `R8-H77_unseen_arena.py`. Run-to-run noise is 0.0023 across three identical trainings, so the in-domain margins are ~60x noise.

| corpus | incumbent | R8-H62 student | delta |
|---|---|---|---|
| private gold, 159 held-out traces | 0.7095 | **0.8531** | +0.1436 |
| RAGTruth EN, 1,200 responses | 0.7039 | **0.8434** | +0.1395 |
| RAGTruth non-EN, mean of 7 languages | 0.6095 | **0.8407** | +0.2312 |
| RAGBench blind arena, mean of 10 subsets | **0.6461** | 0.5956 | **-0.0505** |

- **It beats the incumbent 3/3 on corpora it had training exposure to, and loses on the one corpus neither model trained on** - the three wins are decisive against pre-registered bars of 0.76 / 0.75 / 0.66; the blind loss is the reason it does not ship
- **The multilingual gap vanished rather than closed** - EN 0.8434 against a non-EN mean of 0.8407, a spread of 0.003, where the shipped cascade sits at 0.5626 non-English; plain multilingual supervision was sufficient and the registered gradient-reversal machinery was not needed
- **Domain specialisation and generalisation moved together, not against each other** - the private-gold advantage held at +0.1436 while RAGTruth EN went from -0.061 to +0.1395, which the pre-registration did not predict
- **The student is broadly better than its own teacher** - the 568M `bge-reranker-v2-m3` reads 0.8619 on the private gold and collapses to 0.6432 on RAGTruth; the 307M student reads 0.8531 and 0.8434, giving up 0.009 at home for +0.200 abroad at 54% of the size
- **The teacher's off-domain failure is calibration, not only accuracy** - its fitted RAGTruth threshold comes out at **0.983**, i.e. nearly every foreign pair pushed to the top of its range
- **The RAGTruth cells are clean, the gold cell is caveated** - 0 context overlap on RAGTruth EN and the translations; on the private gold the trace split leaves 29 exact (claim, chunk) pairs of 15,313 (0.19%) and 95.8% of test claims carry at least one chunk seen in training
- **Blind generalisation is the failure** - RAGBench mean 0.5956 against 0.6461, with `finqa` at **0.3974**, below chance and therefore anti-predictive rather than merely weak

### The tabular gap was coverage, not capability

Incarnation 2 (`R8-H78`) added RAGBench train across all ten domains, ~30k pairs, to the same mix.

| subset | R8-H62 | R8-H78 | incumbent |
|---|---|---|---|
| finqa | 0.3974 | **0.7433** | 0.7170 |
| tatqa | 0.5118 | **0.7788** | 0.6156 |
| delucionqa | 0.5325 | 0.6790 | **0.7929** |
| blind-arena mean | 0.5956 | **0.7041** | 0.6461 |

- **Direct supervision closed it** - `finqa` moved +0.3459 from below chance to beating the incumbent, `tatqa` +0.2670; plain ERM over thirteen domains produced the whole gain, with no discriminator, no gradient reversal and no lambda
- **Truncation was hypothesised and refuted** - re-scoring at max_length 2048 instead of 512 moved finqa only 0.398 → 0.428, left tatqa and hagrid unchanged, and made `techqa` WORSE (0.703 → 0.641); techqa carries the longest documents in the benchmark at 3,730 chars and is one of the wins, so length is not the variable
- **`tatqa` documents average 399 characters and still scored at chance before supervision** - the model did not know what to do with a TABLE, and the training mix carried no tabular or numeric-reasoning supervision at all
- **The 0.7041 is NOT countable as a win** - training on RAGBench-train makes RAGBench-test no longer blind for us while the incumbent has still never seen it; recorded for the mechanism it proves, not as evidence of generalisation
- **In-domain cost of the diversity was small but real** - private gold 0.8531 → 0.8314 (-0.0217), RAGTruth EN -0.0061, non-EN +0.0008, all three bars held
- **`delucionqa` is the honest residual** - still -0.1139 against the incumbent WITH direct supervision, the only subset where more data did not help, and the one remaining candidate for a genuine capability gap
- **The fair arena has moved to HaluEval** - 24,507 rows, unseen by both models; incarnation R8-H83 (adding HaluEval and PsiloQA) is still training

### Capacity and shape - what the size budget buys

Depth and vocabulary, not parameter count, decide what a sub-400M budget can hold. The depth table is mmBERT-base truncated within one family, so width, tokenizer, embeddings and recipe are fixed and only depth moves; the shape figures below it are a separate matched-parameter GPU bench at bf16, B=3, T=512.

| depth | params | AUC | macro-F1 | ms/pair |
|---|---|---|---|---|
| 22L | 307.5M | **0.8502** | 0.7830 | 9.20 |
| 11L | 252.4M | 0.8183 | 0.7510 | 5.38 |
| 6L | 227.3M | 0.7332 | 0.6645 | 3.16 |
| 3L | 212.2M | 0.6093 | 0.5996 | 1.75 |

- **Depth is load-bearing** - macro-F1 spread **0.1834** across the four arms, and the curve accelerates downward: -0.032 AUC for the first halving, -0.085 for the second, -0.124 for the third. Keep 22 layers
- **Cutting depth, not parameter count, is what buys latency** - 22L 9.20 ms/pair, 11L 5.38 ms, 6L 3.16 ms at GPU bf16; 11L is the interesting trade at -0.032 AUC for 1.7x
- **Width is faster than depth at matched parameters** - narrow-deep 53L x 512 is **5.0x slower eager** than wide-shallow 8L x 1536 (14.91 vs 2.98 ms) and achieved throughput falls 189 → 123 TFLOP/s
- **So the shape is a genuine trade, not a free win** - width wins on latency, depth wins on quality, and the two must be priced against an accuracy floor rather than chosen on one axis
- **Attention-only (SAN) is closed on latency** - at matched parameters it needs 3x the layers and burns **18% more FLOPs** (579.8 vs 489.6 GFLOP), running 1.16-1.30x slower
- **The sub-400M constraint is really a VOCABULARY constraint** - 64-69% of every multilingual candidate's parameters are the embedding table: mmBERT-base 196.6M of 307M (64%), mDeBERTa-v3-base 192.8M of 277.7M (69%), leaving compute-relevant bodies of ~110M and ~85M
- **Trimming that vocabulary is the only real parameter lever, and it is not free** - mmBERT's 256k multilingual vocabulary is what carried the non-EN result to within 0.003 of English
- **`mDeBERTa-v3-base` could not be trained as a student** - diverged to NaN at step 200 at lr 2e-5 and again at lr 1e-5 with warmup and clipping; excluded rather than fought

## Measurement validity - what the private gold can and cannot prove

Rounds 7 and 8 established that the private gold measures deployment fit, not general capability, and that it cannot measure generalisation at any sample size. Every number in this document is read under these three constraints.

- **Transfer fails in BOTH directions** - the shipped reranker reads 0.8619 AUC on the private gold and 0.6432 on RAGTruth EN, a collapse of -0.219; the public incumbent reads 0.7095 and 0.7039, flat to within 0.006
- **Our 0.8619 is domain specialisation, not general grounding capability** - it does not travel, and no number measured on this gold may be quoted as if it does
- **The gold is not unusually easy** - a neutral third-party model finds both corpora equally hard (0.7095 vs 0.7039), which refutes the loose-labels suspicion; what fails is transfer, not label quality
- **The gold cannot measure generalisation** - 2,752 claims come from 639 traces sharing 619 source documents, and the only split unit with zero chunk overlap is a connected component over shared chunks; splitting by claim, trace or document all leak (claim-level leakage was worth 0.050 AUC and INVERTED a capacity ordering)
- **The corpus is one interconnected mass** - a single component holds **2,534 of 2,752 claims (92%)**, so the effective independent sample is **~39 units**, not 2,752 claims or 639 traces; a clean split leaves 218 test claims and no validation set
- **Consequence, stated plainly** - every number this project has produced on its own gold sits under evidence overlap, macro-F1 0.789 and the candidate's 0.8531 included. That IS the deployment condition, since a grounder serves one corpus repeatedly, but it is not a measurement of generalisation to unseen documents
- **More private traces are needed for measurability before accuracy** - no modelling choice fixes an evaluation that cannot separate train from test
- **The non-English gold slice is unusable in either direction** - 26 to 44 claims per language at base rates 0.885-0.973, one slice carrying roughly a single negative claim; non-English capability can only be measured on RAGTruth's parallel translations
- **Public non-English labels carry ~1/3 machine-translation noise** - on the only human-verified translation set (300 German rows) the incumbent's EN→DE drop is 0.0680 against 0.0999 on machine-translated labels, so a third of the multilingual gap is label noise and two thirds is genuine capability
- **External calibration is still open** - `R7-H49` on LLM-AggreFact remains blocked on Hub auth, and RAGTruth test is in-domain-adjacent once its train split is used, so no fully external number exists yet

## Risks and mitigations

- **Single-engine = OpenVINO, with an ARM caveat** - all three int8 on OpenVINO; x86-64 (Intel/AMD) is native, ARM/Graviton via the OpenVINO ARM plugin is functional but less mature - validate on the deploy target before committing to ARM
- **Pre-filter latency only pays off warm** - the 1.3× measured is with the pre-filter re-embedding all chunks; the design's larger speedup needs the retrieval-warm embeddings reused. If retrieval does not expose them, budget the full ~8.8 s/claim (CPU) or use GPU
- **Top-k recall is unrecoverable** - missing the supporting chunk in the 50→k cut cannot be fixed downstream. Keep retrieval broad at 50; k=8 both held and slightly improved quality on the subset, but validate the best chunk survives before tightening to k=5
- **Tokenizer / revision parity** - ship the matching tokenizer with each IR (saved into `models/ov/<name>/`); feed `input_ids` + `attention_mask` only; pin the IR build against the cached-score revision
- **Empirical thresholds are distribution-bound** - the serving path now carries five fitted-on-OOF thresholds (stack threshold, reranker-only threshold, cascade band [0.01, 0.66], cosine gate [0.493, 0.739]) plus the exit schedule; all live in raw frozen-model score space. Any change to a model, quantization, k, or the evidence distribution invalidates them together - re-run the sweeps (`grounding_hypotheses.py`), they are cheap

## Limitations

Three of these are first-class results from rounds 7-8, not caveats: the shipped number does not transfer, the gold cannot measure generalisation, and tabular evidence was never covered.

- **The shipped number does not transfer** - macro-F1 0.789 / reranker AUC 0.8619 are private-gold figures; the same pipeline reads 0.6432 on public RAGTruth EN and 0.5626 on its non-English translations, with a fitted threshold at 0.983 that is calibration failure rather than accuracy loss alone. Quote the number with its corpus attached or not at all
- **The gold cannot measure generalisation** - ~39 independent components hold 2,752 claims, 92% of them in one; every figure this project has measured on its own gold sits under evidence overlap. That matches deployment, where a grounder serves one corpus repeatedly, but it is not evidence of transfer to unseen documents
- **Tabular and numeric evidence was a coverage gap, and only partly closed** - `finqa` scored 0.3974, below chance, until RAGBench supervision took it to 0.7433; truncation was hypothesised and refuted (512 → 2048 barely moved it, and `techqa`, which carries the longest documents, got worse). The shipped cascade has had no tabular supervision at all and has never been measured on tabular evidence
- **Overlap residual - the lever has now been pulled** - the earlier reading, that a fine-tuned cross-encoder was the untried next lever, is superseded: it was built (Candidate successor) and it wins in-domain by +0.14 AUC while losing 0.0505 on blind data. The residual is now a blind-generalisation and data-coverage problem, not an untried architecture
- **Data-bound - the earlier figure understated it** - "~639 source contexts" is superseded by ~39 independent components; the binding constraint is independent evidence, not claim count
- **Scope** - tuned for paraphrased omission/fabrication hallucinations against retrieved-doc evidence; present-but-contradicted negatives are where the lexical track's contradiction signal earns its place (excluded here by design)
- **Non-English is unmeasured on our own data** - 26-44 claims per language at base rates 0.885-0.973; the only usable non-English measurement comes from RAGTruth's parallel translations, whose labels carry ~1/3 machine-translation noise
- **`delucionqa` is an open capability gap** - the one blind subset that direct supervision did not fix, still -0.1139 against the incumbent

## FAQ

- **Why not ship the public incumbent instead of the cascade?** - `lettucedect-v2-mmbert-base` is domain-general but capped near 0.70 (0.7095 on our gold, 0.7039 on RAGTruth), which is 0.15 AUC below the cascade on the corpus we actually serve
- **Why not fuse our model with the incumbent?** - they are genuinely orthogonal (Spearman -0.046 to +0.083 across ten corpora) and untrained rank-average fusion beats both (RAGTruth EN 0.7479, non-EN 0.6212), but at 568M + 307M = 875M it fails the sub-400M single-model requirement outright. It is the proof that the signals are complementary and individually learnable, not a deliverable
- **Why not a small generative model as the judge?** - rounds 4-6 eliminated prompt, tokenizer, budget and harness as explanations one at a time; `Pleias-RAG-350M`, purpose-built for grounded answering, confirmed claims against a sourdough recipe 9 times in 10 on a harness that reproduced its own vendor's published example verbatim. The failure is one-directional - manufactured support, the one error a grounder must never make
- **Why not have the model quote and let deterministic code check the quote?** - sound on the trivial control (false positives 9 → 1, 18/20 correct) and refuted on real data (macro-F1 0.511, recall **0.267**); quote GENERATION under realistic retrieval is the failure, not the deterministic check
- **Why not a shallower, faster student?** - 11L costs 0.032 AUC for 1.7x, 6L costs 0.117 for 2.9x, 3L is near chance at 0.6093. Depth is not where to economise
- **Why not trim the vocabulary to buy body parameters?** - it is the only real lever (64-69% of the budget) but mmBERT's 256k multilingual vocabulary is what took the non-English mean to within 0.003 of English; trimming it is a measured trade, not a free saving
- **Why not more public data to lift the private-gold number?** - transfer fails in both directions, so public rows do not move our domain; public data buys generalisation and blind-arena coverage, private traces buy the domain number and, more urgently, measurability
- **Why not adversarial domain-invariance for the blind gap?** - plain ERM over thirteen domains produced the entire tabular gain with no discriminator and no lambda; the adversarial and GroupDRO arms are re-registered against 0.7041, not the 0.5956 they were written for

## Implementation

- **Shipped serving path** - `experiments/grounding-semantic/grounding_openvino.py` (`COSINE_GATE`, `cascade_scores`, `rerank_max_early`, `rerank_max`, `nli_max`); IR build `build_ov_grounder.py`; full-gold run `run_grounder_full.py`, both in the same directory (the `scripts/` paths quoted elsewhere in this document predate the move)
- **Benchmarks** - `notebooks/grounding-semantic/07-kj-grounding-sota-benchmark.ipynb` (canonical, full metric set), `04-kj-grounder-latency.ipynb` (cold/warm, top-k), plus `bench_mechanical_levers.py`, `bench_grounder_cascade.py`, `bench_grounder_round2.py` under `experiments/grounding-semantic/`
- **Candidate successor** - teacher corpus `R7-H51_regenerate_teacher.py`, split definition `R8_splits.py` (connected component over shared chunks, the single imported definition), blind arena `R8-H77_unseen_arena.py` (takes `--model`, so every incarnation runs the identical gate)
- **Checkpoints** - `models/R8-H62-mmbert-multicorpus`, `models/R8-H78-mmbert-tabular`; gitignored, synced to S3
- **Evidence** - `semantic-grounding-experiments.md` (canonical experiments log, rounds 1-8); deterministic counterpart `lexical-grounding-sota.md`; cross-lingual and joint wirings in `joint-grounding-experiments.md`; quantization record in `deberta-v3-quantization-experiments.md`
- **Research reports** - `reports/research-grounding-architecture.md` (embedding/body splits, shape), `reports/research-grounding-datasets.md` (licence-clean corpora), `reports/research-grounding-benchmarks.md`

## Conclusions

- **The shipped design is the two-cross-encoder OpenVINO int8 cascade** - macro-F1 0.789 end-to-end at warm mean 585 ms / median 258 ms per claim, ~1.46 GB of int8 IRs on one runtime, no fine-tuning and no torch at serving
- **Verdict rule** - max over chunks, stage-0 cosine gate [0.493, 0.739], cascade band [0.01, 0.66], logistic over the two max scores; all five thresholds live in raw frozen-model score space and re-sweep together on any model, quantization or evidence change
- **Use it as a loop, not a hard gate** - Stage A flags, one batched LLM judge confirms, the agent revises or retracts; a hard gate drops roughly one in five genuine paraphrased claims
- **A successor exists and is not ready** - one distilled 307.5M `mmBERT-base` cross-encoder replaces all three models and beats the size-matched public incumbent by +0.14 to +0.23 AUC on three corpora, but loses 0.0505 on the only genuinely blind one; it ships when an incarnation clears a blind arena, not before
- **Read every private-gold number as domain fit** - 0.789 and 0.8531 are measured under evidence overlap on ~39 independent components, and the same pipeline reads 0.6432 on public data. The next lever is more independent private evidence, for measurability first and accuracy second

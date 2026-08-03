# R8 - Failure-mode and architecture analysis of the blind-arena residual

Analysis of `models/R8-H90-mmbert-dann-full` (best draw 0.7213, recipe mean 0.7066 ± ~0.03) on the frozen RAGBench arena, answering three questions: what actually fails (from read examples, not tables), whether the architecture is the limitation, and whether the training + distillation method transfers to other architectures profitably. Inference and analysis only; no training, no arena tuning, RAGBench untouched as a training source.

**Instruments** (all new artifacts, none modify the canonical gate):

- `experiments/grounding-semantic/R8_analysis_h90_dump.py` - re-scores the arena through the frozen decomposed-min path and dumps per-response records (label, min score, per-sentence scores, argmin sentence, splitter diagnostics, comparator whole-response score). Verification: reproduces the recorded R8-H90 read **exactly** on all 10 subsets (mean 0.7213)
- `experiments/grounding-semantic/R8_analysis_h90_dump.json` - 2,264 records
- `experiments/grounding-semantic/R8_analysis_failure_stats.py` / `R8_analysis_failure_stats.json` - splitter stats, min-vs-mean AUCs, distribution shape, label-noise triangulation
- `experiments/grounding-semantic/R8_analysis_cases/` - ranked worst-FP / worst-FN case files for finqa, delucionqa, pubmedqa, hagrid (12 each side, all read)

---

## Q1 - Failure modes, ranked by evidence

### Mode A - evidence-window truncation at 1,500 chars/chunk (harness constant, not the model)

The scorer truncates every evidence chunk to `chunk_max_chars = 1500` (~375 tokens) before pairing. GPT-4 labeled adherence against the FULL documents; the comparator reads 4,096 tokens per document. The worst FNs in both losing subsets are grounded sentences whose support sits past char 1,500.

- **Verified by needle check** - 6 of 7 worst-FN cases checked have the supporting string present in the full chunk and absent from its first 1,500 chars: finqa idx 83 (`398.0`, chunk 4,668 chars), idx 222, idx 13, idx 52; delucionqa idx 14 (`jewelry`, chunk 3,283 chars), idx 142 (`wing bolt`, chunk 3,275 chars)
- **Exposure concentrates exactly in the losing subsets** - share of bottom-quartile grounded responses carrying at least one truncated chunk: **finqa 86%, delucionqa 74%**, techqa 100%, emanual 72%, expertqa 44% - versus pubmedqa 0%, covidqa 0%, hotpotqa 2%, tatqa 3%
- **Consistency with R8-H85's kill** - H85 refuted truncation as a predictor of SUBSET-mean deltas (techqa hides 62% of evidence mass and wins). That verdict stands. The new argmin-level evidence is a different claim: truncation manufactures individual worst-FNs, and min-aggregation is maximally sensitive to exactly those, because one starved sentence sets the response score
- **Not the R8-H78 rescore** - H78 re-scored at max_length 2048 (out-of-distribution for a model trained at 512) and moved little. Windowed inference - split long chunks into 1,500-char windows, max over windows, pairs stay at the trained length - has never been run

Example (delucionqa idx 14, label grounded, min 0.0063, mean 0.6641): eleven jump-start steps score 0.42-0.99, and the two steps whose manual text lies past the truncation boundary score 0.006-0.007 and sink the response:

```
[0.995] - Only use the positive battery post on the main battery to jump start your vehicle.
[0.007] - Avoid the radiator cooling fan when the hood is raised.
[0.006] - Remove any metal jewelry that could make inadvertent electrical contact.
```

**Rough attribution: ~30% of the addressable residual, dominant in the two subsets we lose.**

### Mode B - unsupportable-by-construction sentences (inference, hedge, absence, calculation-step)

Responses contain sentences no evidence chunk can entail because they are discourse, not claims: conclusions ("This suggests..."), hedges ("However, more research is needed"), correct refusals ("There is no information provided about..."), and calculation narration ("Subtract the beginning balance from the ending balance"). RAGBench's GPT-4 annotator counts reasonable inference as supported; per-(sentence, chunk) entailment cannot, and the min executes the response for it.

- **pubmedqa is the extreme case** - 24/43 (56%) of bottom-quartile grounded argmins carry inference markers, +3 absence statements; every verdict-style response ends in a conclusion sentence, so 76% of grounded responses score < 0.1 and ranking happens inside a crowded floor (subset AUC 0.6058, comparator 0.5162 - it suffers the register too)
- **finqa second face** - 11/57 calculation-step argmins ("To calculate the percentage increase: ..."), plus meta-statements in refusal-style answers ("I would need to extrapolate...", label grounded, min 0.027)
- **Global rate** - the argmin is a discourse-type sentence in 21% of all multi-sentence grounded responses (vs 19% of hallucinated ones): the mode does not rank in the wrong direction globally, it collapses per-subset calibration and floors the min wherever verdict-style register dominates
- **Why the comparator is robust here** - its token-level objective trained on span-annotated data learns that discourse tokens are not hallucination spans; absence-of-support is not presence-of-hallucination in its label space

Example (pubmedqa idx 55, label grounded, min 0.0074): three factual sentences score 0.52-0.94, then

```
[0.044] The available data suggests that ... is not strongly associated with Down syndrome.
[0.007] However, further evaluation and monitoring may still be recommended.
```

**Rough attribution: ~25% of the addressable residual; the pubmedqa low-absolute reading is mostly this.**

### Mode C - numeric-derivation blindness, both directions (the genuine model-class gap)

The cross-encoder can neither verify arithmetic nor detect a copied number whose semantics are wrong.

- **FN direction** - 43/57 (75%) of finqa and 40/59 (68%) of tatqa bottom-quartile grounded argmins are numeric/derived sentences: "This represents an increase of 4.9 percentage points" (correct, computable from the table, score 0.012); "$8.2 million - $5.5 million = $2.7 million" (score 0.017)
- **FP direction** - hallucinated responses that copy a true number with wrong unit/year/quantity score as grounded: "$78.29 million" where the table says \$78.29 per share (score 0.69); 2008 column values asserted as 2009 (score 0.65-0.82); "the portion ... is 448.6 million cans" where the question wants a share, not the raw cell (score 0.70). The comparator catches two of these three (0.22, 0.55, 0.35)
- **tatqa wins anyway** (0.7718) because its negatives are also numeric, so ranking survives; finqa's 20 negatives contain exactly these surface-matching copies, and they outrank truncation-starved positives - the -0.0440 loss is Modes A and C compounding

**Rough attribution: ~15% (overlaps A on finqa); the one mode that is plausibly architecture-class-bound - a similarity-shaped encoder without numeracy supervision cannot compute.**

### Mode D - multi-chunk composition blindness

Max-over-chunks is an OR over single chunks: a sentence composed from several chunks can never be confirmed by one pair, and a conflation of two chunks can never be flagged by one pair.

- **hotpotqa calibration collapse** - multi-hop single-sentence answers; 84% splitter fallback-to-whole; median grounded score **0.0118**, 80% of all responses below 0.1. AUC still 0.7253 (ranking inside the floor works) but the subset's scores are incommensurable with the rest - any global threshold fails it
- **hagrid** - citation-style sentences aggregating "[1, 2, 3, 7, 8]" score low; absence statements (correct refusals) score low; 56% fallback rate
- **FP variant (delucionqa)** - two chunks describe two DIFFERENT "Trailer Sway Control" systems (electronic vs telescoping link); the response conflates them into one and scores 0.90+ per sentence because each sentence matches SOME chunk; the comparator flags it (0.066)

**Rough attribution: ~10%; caps hotpotqa/hagrid absolutes and produces the hardest FPs.**

### Mode E - min-aggregation fragility and splitter artifacts

The formula that bought +0.05 blind also manufactures errors and multiplies variance.

- **delucionqa: the formula costs -0.087 on our own scores** - mean-over-sentences of the SAME per-sentence scores reads AUC **0.8130** vs min's 0.7263 (comparator 0.7929); H99's whole-response read there (0.8367) beats the incumbent outright. delucionqa is not lost to capability - it is lost to the aggregation
- **One-benign-sentence executions** - grounded responses with min < 0.2 while mean > 0.6: delucionqa 16, hagrid 13, techqa 11 (mostly Mode A/B argmins doing the sinking)
- **Splitter mechanics measured** - fallback-to-whole fires on 54-84% of covidqa/hagrid/hotpotqa/tatqa (median 1 sentence - the "decomposed" read is largely whole-response there); fragments like "According to the given context ," survive the 25-char floor and become argmins; short sentences (< 25 chars) are dropped entirely, so a short hallucinated sentence is invisible; the cap of 12 truncates 15% of techqa responses
- **Noise amplification** - R8-H100: whole-response run-to-run gap 0.0074, decomposed-min gap 0.0295 - the min concentrates each response's score on its weakest sentence, so per-sentence miscalibration compounds ~4x

**Rough attribution: ~15% (interacts with A and B); the single cheapest lever - delucionqa alone is worth +0.009 on the blind mean under a better aggregation.**

### Mode F - GPT-4 label noise (real, bounded, not binding)

- **External anchor** - RAGBench validated its GPT-4 (gpt-4-0125-preview) adherence labels against human annotation on DelucionQA: 93% example-level accuracy, 0.96 F1; ~2% of examples retained self-conflicts after 3 re-annotations; partially-supported sentences are labeled hallucination (the benchmark itself is an AND over sentences - the decomposed-min formula mirrors its construction)
- **Both-models-confident disagreement is rare** - responses where both our model and the comparator score > 0.7 against a hallucination label: 0-5 per subset (hagrid 5, covidqa 2). The mirror case (both < 0.2/0.3 on grounded) is common but explained by Modes A/B/D, not labels
- **Read estimate** - of 48 worst-FPs read across four subsets, 3-5 look plainly grounded (delucionqa idx 82; finqa idx 114's correct percent computation) → roughly 10% of the extreme-FP tail is label error, consistent with the 7% external figure
- **Where it bites** - the extreme-base-rate subsets have 12-20 negatives (delucionqa 12, tatqa 14, hotpotqa 17, finqa 20); 2-3 mislabels are 10-25% of the negative class. A perfect scorer under q1 = 20-30% mislabeled negatives and ~3% mislabeled positives ceilings at AUC 0.835-0.885 on such a subset
- **Ceiling estimate** - blind-mean ceiling for a perfect model ≈ **0.82-0.88**, not 0.75-0.80. The 0.74 target sits comfortably below the label ceiling; label noise is not the binding constraint at 0.7066, it is a per-subset variance floor

### Quantified summary

| subset | AUC min | AUC mean | comparator | fallback-whole | BQ-pos truncated | dominant modes |
|---|---|---|---|---|---|---|
| covidqa | 0.7755 | 0.7680 | 0.7354 | 61% | 0% | - |
| delucionqa | 0.7263 | **0.8130** | 0.7929 | 19% | 74% | A, E, D(FP) |
| emanual | 0.7058 | 0.6071 | 0.5999 | 9% | 72% | A |
| expertqa | 0.8248 | 0.7933 | 0.6504 | 2% | 44% | - |
| finqa | 0.6732 | 0.6522 | 0.7170 | 35% | 86% | A, C, B |
| hagrid | 0.6517 | 0.5596 | 0.5991 | 56% | 19% | D, B, E |
| hotpotqa | 0.7247 | 0.6551 | 0.5976 | 84% | 2% | D |
| pubmedqa | 0.6058 | 0.4944 | 0.5162 | 2% | 0% | B |
| tatqa | 0.7718 | 0.6622 | 0.6156 | 54% | 3% | C (survived) |
| techqa | 0.7527 | 0.7386 | 0.6365 | 5% | 100% | A (survived) |

Min beats mean on 8/10 subsets - the formula is globally right and locally catastrophic (delucionqa, and pubmedqa's floor crowding). Score mass sits near 0 wherever fallback is low and register is verdict-style: fraction of ALL responses below 0.1 ranges from 7% (covidqa) to 80-81% (hotpotqa, pubmedqa) - per-subset calibration is incommensurable, which any deployment threshold inherits.

---

## Q2 - Is the architecture the limitation? Verdict: no. Recipe-and-formula-limited.

Weighing the registered evidence plus Q1:

- **(a) Same backbone, spread 0.13 across recipes** - every measured configuration on mmBERT-base: bi-encoder 0.53 (in-domain), H62 blind 0.5956 → H90 recipe 0.7066 mean / 0.7213 best; the comparator on the SAME backbone reads 0.6461. Training recipe and scoring formula moved +0.12; no backbone change has ever been measured to move anything comparable. Recipe >> backbone at this operating point
- **(b) The depth ladder is in-domain evidence, and in-domain dissociates** - 22L 0.8502 / 11L 0.8183 / 6L 0.7332 on gold says capacity is load-bearing for the fit; but five recorded in-domain/blind dissociations (H62, H78, H83/84, H81, H91-vs-H90) say the blind residual is not the in-domain quantity. The Q1 modes (truncation, discourse sentences, aggregation) are not capacity phenomena - a 10x model still cannot see chars past 1,500 or entail an inference sentence from a chunk
- **(c) Distance vs noise** - target 0.74 sits 0.033 above the recipe mean, ~1 sigma of the measured run-to-run noise (±0.03). The formula-level levers quantified in Q1 (delucionqa aggregation +0.009 mean; truncation windowing on the 5 exposed subsets; multi-seed selection) plausibly sum to +0.02-0.04 without touching the architecture
- **(d) The 512/4096 context difference is not an architecture difference** - the H90 trunk's own config carries `max_position_embeddings: 8192` (local/global alternating attention). MAX_LEN 512 and the 1,500-char chunk cap are recipe constants. Mode A is real and truncation-shaped, but the same weights, re-trained or re-served with windows or longer pairs, remove it. The comparator's whole-response single-pass mode is likewise available on our backbone
- **(e) Label ceiling ~0.82-0.88** does not bind at 0.71

**Attribution of the current residual (recipe mean 0.7066 → ceiling ~0.85):** formula/harness (Modes A + E) ~45%, training-data coverage of discourse/numeric registers (Modes B + C data-side) ~35%, genuine model-class capability (Mode C arithmetic core, Mode D composition) ~15%, labels ~5%. The one mode with a real architecture-class argument is numeric derivation (Mode C) - an encoder scorer with no numeracy pretraining will not learn arithmetic from BCE on 762k pairs; that is where a decoder-based scorer differs in kind rather than in size.

---

## Q3 - Method transfer to other architectures

The method - multi-corpus mix (soft teacher labels on private pairs + hard public labels), BCE, optional DANN via gradient reversal on pooled features, decomposed-min inference - is architecture-agnostic: DANN attaches to any pooled representation, BCE to any scalar head, the formula to any pair scorer. Transfer cost is therefore dominated by training compute and serving shape, not by method surgery. Cost anchor: mmBERT-base (110M body) trains the 762k mix in ~6.3h/epoch on the RTX PRO 6000 96GB at batch 48/512 tokens.

### Candidates

**ModernBERT-large (answerdotai)** - 395M (body ~344M, 28L x 1024), 8,192 ctx, Apache-2.0, **English-only**.
- Fits the sub-400M ceiling (barely); 3.1x body FLOPs → ~19h/epoch; flash-attention supported
- Addresses: Mode A (long evidence windows), Mode D (whole-response + all evidence single-pass), capacity per (b)
- Conflicts: English-only breaks the multilingual deliverable (mmBERT's 256k vocab carried non-EN to within 0.003 of EN); it would be an arena-only fork, a dual-track maintenance cost
- Honest expected effect: the long-context benefit is available on our own 8,192-ctx backbone without losing multilingual, so this candidate's UNIQUE contribution is +88M parameters; predicted blind effect +0.00-0.02, below the noise bar. Rank: 3rd

**DeBERTa-v3-large / mDeBERTa-v3-base (microsoft)** - 435M (over the 400M budget; 24L x 1024, 131M embeddings) / 278M (86M body), MIT, 512 relative-position ctx, EN / 100-language.
- The MiniCheck and FactCG line chose DeBERTa-v3-large, so it is the NLI-strongest known encoder; disentangled attention has no flash-attention path (slow, ~30-40h/epoch) and known fp16 instability
- The campaign already tried mDeBERTa-v3-base as a student: diverged to NaN at step 200 twice (lr 2e-5 and 1e-5 with warmup + clipping), excluded rather than fought
- Addresses: none of Modes A/D (512 ctx is a hard regression vs our 8,192), possibly sharper entailment for Mode B FPs
- Budget: 435M requires revisiting the sub-400M rule; not worth it for a model that shrinks the context. Rank: 4th

**Qwen3-0.6B as scorer** (Qwen3-Reranker-0.6B init, seq-cls head - a converted checkpoint already exists on the Hub) - 0.6B total (~0.44B body, 28L), 32k ctx, Apache-2.0, 119 languages.
- Addresses **Mode C in kind** - decoder LM pretraining (36T tokens incl. code/tables/math) gives numeric-and-unit competence no encoder in this list has; also Modes A/D via 32k single-pass; multilingual preserved
- Cost: full FT feasible on the 96GB card (~10GB optimizer states + activations at batch ~16/1024 tokens), ~20-25h/epoch at 512-1,024 tokens; DANN/BCE/decomposed-min attach unchanged
- Serving: ~4x mmBERT FLOPs per pair; GPU inference fine, but the torch-free OpenVINO int8 CPU cascade shape suffers (~0.7GB int8, slower per pair); 1.5x over the sub-400M budget - the budget question must be reopened explicitly
- Expected blind effect: finqa +0.04-0.07 subset-level if the numeracy transfers, mean +0.01-0.02 - borderline vs ±0.03 single-draw noise, adjudicable only on multi-seed means. Rank: 2nd, and 1st among things that change what the model CAN do
- Qwen3-1.7B: ~12x body FLOPs (~75h/epoch, LoRA advisable); reserve for a second round only if 0.6B shows the finqa mechanism

**Token-classification head on mmBERT (the comparator's own mode)** - 307M, same backbone, same budget, same multilingual.
- The H73 evidence read correctly: gold 0.8843 (token head alone 0.8896 - the best in-domain number of the campaign) but blind 0.6607 under the FUSED read, which stacks a token-level min inside a sentence-level min - two ANDs over-sharpen. The blind token-head-ONLY read was never taken (parked, unregistered). What H73 proves is that the head learns the localized-number mechanism (best-ever tatqa at its date); what it does NOT prove is that head choice alone confers the comparator's robustness - the comparator's edge on finqa/delucionqa survives our head experiment, so its training distribution (span-annotated benign discourse) is doing the work
- Addresses: Mode B directly (span labels teach that discourse tokens are not hallucinations), Mode C partially (a wrong number is a localized token event), Mode E (token-level aggregation is intrinsically smoother than sentence-min - and the comparator's whole-gap variance is the low one)
- Cost: existing trainer (`R8-H73_twohead.py`), ~7h/epoch, zero integration work. Rank: 1st on cost-adjusted expected value

### Ranked recommendation

1. **Full-mix two-head with DANN, judged on the token-head-only decomposed read** - cheapest, attacks Modes B/C/E, the mechanism is already demonstrated in-domain
2. **Qwen3-0.6B decoder scorer** - the only candidate that changes capability class (numeracy, Mode C), multilingual preserved, budget and serving shape must be consciously re-opened
3. **ModernBERT-large** - only if the English-only fork is acceptable; its unique contribution beyond our own backbone is capacity alone

Before any of these: the formula-level levers (windowed evidence, dispersion-conditioned aggregation, multi-seed selection) are cheaper than every architecture change and plausibly close most of the 0.033 gap on their own.

### Pre-registration sketches

**P1 - windowed evidence + per-response aggregation guard (formula only, frozen weights)**
- Hypothesis: because the worst grounded FNs are truncation-starved sentences (86%/74% bottom-quartile exposure in finqa/delucionqa; needle-verified) and min-aggregation is maximally sensitive to them, scoring each sentence against 1,500-char WINDOWS of every chunk (max over windows over chunks, pairs at trained length) will lift finqa and delucionqa by ≥ +0.03 each with the other eight subsets within ±0.02
- Bar: blind mean ≥ 0.7213 on the H90 checkpoint (single deterministic read, no training)
- Kill: if either losing subset moves < +0.01, windowing is dead as a lever and the residual re-attributes to Modes B/C
- Note: respects the H85 kill (that was a subset-mean-correlation claim; this is an argmin-mechanism claim); no aggregation parameter is tuned anywhere (H94 lesson - RAGTruth is not a valid proxy for aggregation shape; any conditional aggregator must be parameter-free)

**P2 - full-mix two-head, token-head-only blind read**
- Hypothesis: because span supervision teaches that discourse tokens are not hallucination events (Mode B) and a wrong number is a localized token event (Mode C), the H73 recipe scaled to the full 762k mix + DANN lambda 0.02, read blind as 1 - max(halluc-token) per (sentence, chunk) with the standard decomposed-min, will beat the score-head read of the same checkpoint and land ≥ recipe mean + 0.02 (≥ 0.727)
- Kill: token-only read < score-head read of the same checkpoint → head choice is confirmed as not the carrier of the comparator's robustness, closing the head-transfer question
- Cost: ~7-9h GPU1, existing trainer modified for full mix

**P3 - Qwen3-0.6B scorer (the capability-class test)**
- Hypothesis: because Mode C is a pretraining-capability gap, a Qwen3-Reranker-0.6B-initialized seq-cls scorer trained one epoch on the identical 762k mix (BCE, no DANN in round 1, MAX_LEN 1,024) and read through the identical decomposed-min gate will read finqa ≥ 0.72 (from 0.673) while the blind mean lands ≥ 0.70
- Bars stated against recipe means with the ±0.03 noise bar explicit; a single draw cannot confirm, so the claim is two-stage: stage 1 one draw for the finqa mechanism, stage 2 (only if stage 1 fires) n=2 draws for the mean claim
- Kill: training instability, or finqa < 0.70, or gold < 0.80 (in-domain guardrail) → decoder line closed at this size, budget rule stays
- Cost: ~24h GPU1 per draw; requires explicit owner sign-off on breaching the sub-400M budget before launch

---

## Bottom line

The blind residual is manufactured, in order of mass, by the harness's own evidence window, by sentence types that no pair-entailment can support, by numerics the model-class cannot compute, by cross-chunk composition the OR-aggregation cannot see, and by a scoring formula that amplifies all of the above 4x in variance. The architecture is the least-implicated component: the same 307M backbone spans 0.59-0.72 blind across recipes, carries 8,192 context unused, and sits ~0.03 (one noise sigma) from the target under the current recipe. Fix the window and the aggregation first, then buy the token head's robustness on the full mix; spend architecture money only on the one mode that is a capability class - numeracy - and only with the budget rule consciously reopened.

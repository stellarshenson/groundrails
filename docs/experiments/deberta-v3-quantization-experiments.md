# DeBERTa-v3 Quantization for CPU - Experiment Summary

## Situational Overview

The consolidated semantic grounder (`semantic-grounding-sota.md`) is two cross-encoders: `bge-reranker-v2-m3` (rerank) and `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (NLI), combined by a logistic over their max-over-chunks scores (macro-F1 0.796). Deployment is CPU/Lambda, where int8 quantization is needed for throughput. The reranker int8-quantizes perfectly, but the NLI does not - its stock int8 ONNX collapsed to pearson 0.35 vs PyTorch. This document records every approach tried to get a CPU-deployable int8 NLI, and the metrics each was held to.

- **mDeBERTa-v3 is DeBERTa-v2 architecture** (`model_type=deberta-v2`); disentangled attention is the cause
- **The reranker is the control** - XLM-RoBERTa, standard attention, int8 parity 0.9975 (quantizes cleanly)
- **The export is correct** - mDeBERTa fp32 ONNX matches PyTorch at 0.9999; only the int8 quantization fails

## Executive Summary

Two hypotheses were tested: replace the NLI with a quant-friendly model, or salvage mDeBERTa-v3's int8. Both have so far failed to produce a CPU-deployable int8 NLI that holds the grounding signal; the validated fallback is mDeBERTa-v3 fp32 on CPU.

- **H1 (replace the NLI) - refuted** - every standard-attention NLI is far weaker for this grounding task (single AUC 0.58-0.67 vs mDeBERTa 0.806); the stack collapses to ~reranker-alone (0.757-0.765), all below the 0.782 bar
- **H2 (salvage int8) - cracked via SmoothQuant** - naive int8 of DeBERTa-v3 activations destroys it (parity 0.29-0.75) and no ONNX-Runtime method fixes it; but **OpenVINO/NNCF SmoothQuant** (migrating the activation outliers into the weights) reaches **pearson 0.985 at 318 MB** (alpha 0.7, the peak of a 0.5/0.7/0.9 sweep), full-gold parity **0.9841** and a re-fit stack **macro-F1 0.795** (vs fp32 0.796) - 3.6x smaller than fp32, deployable with no measurable quality loss. The OpenVINO int8 is the confirmed deployable NLI
- **Validated fallback** - mDeBERTa-v3 fp32 on CPU + reranker int8: full macro-F1 0.796, ~1.12 GB, the NLI as the CPU latency cost
- **Root cause (web-researched)** - DeBERTa disentangled attention produces extreme per-channel activation outliers; MinMax int8 (dynamic or static) cannot represent them, so activation quantization collapses, while standard attention survives

## Overview

The investigation runs on the 2,752-record verified gold. Each approach is held to four metrics; a method must clear the parity or the stack-quality bar to be deployable.

- **int8 parity (primary gate)** - pearson of the int8 max-over-chunks scores vs the cached PyTorch scores on a 24-32 record stratified sample; **target >= 0.99**. A faithful int8 means the fitted logistic transfers with no re-fit. Baselines: bge reranker int8 0.9975 (control), mDeBERTa fp32 ONNX 0.9999 (export check)
- **stack macro-F1 (quality gate)** - the 2-feature logistic over {reranker, NLI} out-of-fold; **target >= 0.782** (one fold-std below the fp32 stack 0.796; reranker-alone is 0.757). For a parity-passing method this equals 0.796; for a partial method it is measured by re-fitting on the int8 scores
- **footprint** - **target < 1.12 GB** (mDeBERTa fp32), ideally ~339 MB int8 - the CPU/Lambda size win
- **latency** - CPU per-answer within the 20-50s answer budget (the reason int8 is needed); GPU fp16 is ~1-1.4s and not the concern
- **non-degeneracy** - int8 score std > 0 (guards against an always-one-label collapse)

## H1 - Replace the NLI

### Hypothesis

A standard-attention multilingual NLI cross-encoder (BERT/RoBERTa/XLM-R/MiniLM), which int8-quantizes cleanly, can replace mDeBERTa-v3 while holding the stack macro-F1 within noise of 0.796 - removing the DeBERTa quantization problem entirely.

### Business / Technical Problem

mDeBERTa-v3 cannot be int8-quantized for CPU. If an equally-strong but quant-friendly NLI exists, swapping it in is the cleanest fix - the grounder stays two cross-encoders, both int8 on CPU, small and fast, with no exotic quantization.

### Methodology

Score each candidate on the gold with the same path as the incumbent (`grounding_models.cross_scores`, kind=nli, pairs (chunk, claim), softmax entailment, max over chunks), record single-signal AUC, then fit the 2-feature {reranker, candidate} logistic out-of-fold and read the stack macro-F1.

- **Gate** - stack macro-F1 >= 0.782; int8 parity is a formality for standard attention (the reranker proved it at 0.9975)
- **Candidates, smallest first** - MiniLMv2-L6, MiniLMv2-L12, XLM-R-base, XLM-R-large

### Setup

- **Models** - `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`, `-L12-`, `symanto/xlm-roberta-base-snli-mnli-anli-xnli`, `joeddav/xlm-roberta-large-xnli` (the last needed a tokenizer override to `FacebookAI/xlm-roberta-large` - its shipped sentencepiece failed to load)
- **Scoring** - RTX 5090, the existing `score_one` path, cached to `model_scores/*.npy` aligned to the gold
- **Stack** - logistic + StandardScaler + L2, 5-fold OOF, `best_macro`

### Execution

Each candidate scored on all 2,752 records; stack re-fit per candidate against the fixed reranker scores.

### Results

| NLI model | architecture | single AUC | stack macro-F1 | verdict |
|---|---|---|---|---|
| mDeBERTa-v3-base-mnli-xnli (incumbent) | DeBERTa-v2 | 0.806 | 0.796 | int8-hostile |
| multilingual-MiniLMv2-L6-mnli-xnli | XLM-R (MiniLM) | 0.577 | 0.758 | FAIL |
| multilingual-MiniLMv2-L12-mnli-xnli | XLM-R (MiniLM) | 0.674 | 0.764 | FAIL |
| symanto/xlm-roberta-base-snli-mnli-anli-xnli | XLM-R-base | 0.611 | 0.763 | FAIL |
| joeddav/xlm-roberta-large-xnli | XLM-R-large | 0.661 | 0.765 | FAIL |

### Conclusions

- **Refuted** - no standard-attention NLI holds the stack; all sit at 0.758-0.765, barely above reranker-alone (0.757)
- **mDeBERTa-v3 is uniquely strong for this task** - XNLI accuracy does not transfer to grounding paraphrased production claims against retrieved evidence; the incumbent must be kept, so int8 must be solved on DeBERTa itself

## H2 - Salvage mDeBERTa-v3 int8

### Hypothesis

A quantization method exists that makes mDeBERTa-v3's int8 ONNX faithful (parity >= 0.99) at a useful CPU footprint (~339 MB), by handling the disentangled-attention activation outliers that naive int8 cannot.

### Business / Technical Problem

mDeBERTa-v3 fp32 on CPU is ~1.12 GB and is the latency bottleneck (~40-60s/answer). A faithful int8 would cut size ~3x and speed up CPU inference, making the full-quality grounder viable on Lambda.

### Methodology

Build each int8 variant from the fp32 ONNX, score the stratified sample with plain `onnxruntime.InferenceSession`, and compare to the cached PyTorch scores (pearson / spearman / max|delta| / std). Helpers in `grounding_onnx.py`.

- **Quant builders** - dynamic int8; static QDQ (calibrated, per-channel, `OpTypesToExcludeOutputQuantization`); mixed-precision (`nodes_to_exclude` the disentangled-attention MatMuls); weight-only 4-bit (MatMulNBits); fp16; SmoothQuant (onnx-neural-compressor)
- **Calibration** - ~120-150 real claim x chunk pairs sampled from the gold
- **Gate** - parity >= 0.99 (or stack macro-F1 >= 0.782 if re-fit); footprint < 1.12 GB; std > 0

### Setup

- **Base** - the official `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` `onnx/model.onnx` (fp32, opset 12; upgraded to 14 for per-channel QDQ)
- **Tooling** - `onnxruntime.quantization` (dynamic/static/mixed), `matmul_nbits_quantizer` (4-bit), `onnxconverter_common.float16` (fp16), `onnx-neural-compressor` and `optimum-intel` + `nncf` + `openvino` (SmoothQuant); runtime is plain onnxruntime for ONNX variants and the OpenVINO runtime (`ov.Core`) for the working SmoothQuant IR
- **Sample** - 24 records (12 hall + 12 supp), entailment index 0, `type_vocab_size=0` so no token_type_ids

### Execution

Each method built and parity-scored in turn; the static-QDQ first attempt failed to even load (`Unrecognized attribute: axis for DequantizeLinear`) because the ONNX was opset 12 - per-channel QDQ needs opset >= 13, so the model was upgraded to opset 14 before re-running.

### Results

| method | quantizes activations | parity | size | result |
|---|---|---|---|---|
| stock dynamic int8 (`model_quantized.onnx`) | yes (dynamic) | 0.35 | 339 MB | broken (the original finding) |
| static QDQ (opset-14, per-channel, `Add`/`Softmax` outputs + disentangled MatMuls excluded) | yes (static) | 0.29 | 346 MB | worse - outliers survive calibration |
| mixed-precision (FFN MatMuls int8, attention fp32) | yes (FFN) | 0.754 | 346 MB | partial, too lossy |
| weight-only 4-bit (MatMulNBits, activations fp32) | no | 0.90 | 878 MB | best parity, no size/latency win |
| fp16 | n/a | - | - | conversion type-conflicts on DeBERTa control-flow ops; ORT CPU has no fp16 kernels (casts to fp32) - pointless |
| SmoothQuant (onnx-neural-compressor, stock) | yes (smoothed) | - | - | library bug - crashed on the DeBERTa graph before any result |
| SmoothQuant (onnx-neural-compressor, **forked + fixed**) | yes (smoothed) | 0.61 (MinMax) / 0.62 (Percentile) | 339 MB | 3 crash bugs fixed so it runs end-to-end on DeBERTa-v2, but ORT static-quant of the activations stays faithless (std ~0.12) even after smoothing - not viable |
| **SmoothQuant (OpenVINO / NNCF, alpha sweep 0.5 / 0.7 / 0.9)** | yes (smoothed) | **0.980 / 0.985 / 0.978** | **318 MB** | **breakthrough** - peak at alpha 0.7 (pearson 0.985, std 0.334); 3.6x smaller, signal preserved; NNCF migrates the outliers into the weights and handles DeBERTa's 12 `If` subgraphs |

### Conclusions

- **The rule** - quantizing DeBERTa-v3 activations gives parity 0.29-0.75; only weight-only (fp32 activations) preserves the signal (0.90), and that is 878 MB at fp32 speed - no CPU win
- **fp16 is not a CPU option** - ONNX Runtime has no fp16 CPU kernels, so it casts to fp32 (no speed gain), and the float16 conversion of DeBERTa is fragile
- **SmoothQuant is the answer** - migrating the activation outliers into the weights is the one method that works; the onnx-neural-compressor implementation is buggy (`KeyError: 'Shape'`), but the mature **OpenVINO/NNCF** SmoothQuant succeeds at pearson 0.985 (alpha 0.7, the peak of a 0.5/0.7/0.9 sweep on a 50-record balanced sample) and 318 MB (3.6x smaller than fp32). The trade-off: the NLI runtime becomes OpenVINO (the reranker stays ONNX-Runtime int8)
- **FlashDeBERTa - not applicable** - a GPU-only Triton kernel that fuses DeBERTa's disentangled attention (3-5x faster on GPU); no CPU path (open issue #26) and no quantization/footprint change, so it does not help CPU/Lambda - it would only speed a GPU deployment, where mDeBERTa fp16 is already fast
- **Validated fallback** - if the int8 stack does not hold, ship mDeBERTa-v3 fp32 on CPU + reranker int8 (full macro-F1 0.796)

## Next Steps

- **SmoothQuant alpha tuned (done)** - swept 0.5 / 0.7 / 0.9 on a 50-record balanced sample; parity peaks at **alpha 0.7 (pearson 0.985)**, with 0.5 and 0.9 at 0.980 / 0.978 - all near 0.98, just under the 0.99 target, none degenerate (std 0.32-0.34). The deployable int8 IR is the alpha-0.7 SmoothQuant model (318 MB). Reproduced end-to-end in `notebooks/02-kj-deberta-int8-smoothquant.ipynb`
- **End-to-end stack macro-F1 (done - PASS)** - scored the full 2,752 gold with the alpha-0.7 OpenVINO int8 NLI (cached to `data/interim/model_scores/mDeBERTa-v3-int8-sq.npy`); full-gold parity vs fp32 **0.9841**, and the re-fit {reranker, NLI} logistic out-of-fold holds **macro-F1 0.795 (AUC 0.876)** vs fp32 0.796 - within 0.001, clearing the 0.782 gate. The OpenVINO int8 SmoothQuant NLI (318 MB) is deployable with no measurable quality loss
- **Port SmoothQuant to ONNX Runtime (done - quality-limited, not viable)** - to keep a single runtime we forked `onnx-neural-compressor` ([stellarshenson/neural-compressor](https://github.com/stellarshenson/neural-compressor)) and fixed three real bugs that crashed its SmoothQuant on DeBERTa-v2 disentangled attention: (1) `set_initializer` reused stale `dims`, corrupting a scalar constant promoted to a per-channel vector during folding (`cannot reshape array of size N into shape ()`); (2) the absorb functions forced shape-incompatible folds (`operands could not be broadcast (3072,) (768,)`) - now guarded to skip and leave the smooth Mul unfolded; (3) folding into an elementwise `Mul` (DeBERTa pooler GELU) broke the graph (`ShapeInferenceError`) - `Mul` removed from the absorb targets. With these the SmoothQuant **runs end-to-end** on mDeBERTa-v3, but the int8 is faithless: **parity 0.61 (MinMax) / 0.62 (Percentile calib)**, std ~0.12 (near-degenerate) - ORT's static int8 of DeBERTa activations collapses even after smoothing, regardless of folding or calibration method. The gap vs OpenVINO/NNCF (0.984) is the quantizer: NNCF adds fast-bias-correction and better activation granularity that ORT static-quant lacks. The fixes live on the GitHub fork as a record; the local submodule and the ORT build script were removed once OpenVINO was chosen as the single engine
- **Decision - ship OpenVINO int8 for the NLI** - the single-runtime ORT goal is not achievable at quality with this model; the deployable NLI is the OpenVINO alpha-0.7 int8 IR (318 MB, parity 0.984, stack macro-F1 0.795) and the reranker stays ONNX-Runtime int8 (mixed runtime). Update `semantic-grounding-sota.md` accordingly; the fork fixes remain a valid upstream contribution (SmoothQuant no longer crashes on DeBERTa-v2)

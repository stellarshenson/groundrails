# R15 - Qwen Reconnaissance (August 2026)

Research date: 2026-08-09. Compiled by web-research agent; coordinator-persisted. Sources cited inline. Uncertainty flagged explicitly. Highest-confidence sections per the researcher: hardware (§4) and the instruction-following regression (§3.3).

## Executive summary

- **Latest release: Qwen3.8-Max** (3 Aug 2026) - 2.4T MoE, ~95B active, 1M context, multimodal, **API-only**. This is the "beats everything" release (SWE-bench 87.3 vs GPT-5.5 82.6, behind Claude Opus 4.8 89.2; Vals Index #2 open-weight 66.1). **Weights NOT released**; Qwen3.8-Max + Qwen3.8-27B promised on HF week of 10 Aug 2026, license UNANNOUNCED. `Qwen/Qwen3.8-27B` returns 401; third-party "Qwen3.8" repos are self-declared empty placeholders. Do not plan on 3.8 yet; re-check after 10 Aug.
- **Latest usable open weights: Qwen3.6 series** (Apache 2.0, multimodal, 262K native context, hybrid Gated-DeltaNet + Gated-Attention): **Qwen3.6-27B dense** (22 Apr 2026: AIME 94.1, MMLU-Pro 86.2, SWE-bench-V 77.2) and **Qwen3.6-35B-A3B MoE** (16 Apr 2026, 3B active: AIME 92.7, MMLU-Pro 85.2).
- **Best fit for the RTX PRO 6000 96GB (sm_120)**: (1) Qwen3.6-35B-A3B-FP8 (~35GB, 3B active = batching winner) for judge; (2) Qwen3.6-27B-FP8 (~27-30GB) or nvidia/Qwen3.6-27B-NVFP4 (vLLM >= 0.24.0) for generation; (3) Qwen3.5-27B as instruction-following fallback.
- **Use-case verdicts**: Judge - GO on 35B-A3B-FP8 with enable_thinking=false and --language-model-only, BUT A/B against Qwen3.5-27B first (independent testing found Qwen3.6 "significantly worse" than 3.5 on IFBench; 3.5-27B posts IFEval 95.0 / IFBench 76.5 - for a JSON-emitting judge, instruction adherence beats reasoning score). Generator - GO, Qwen3.6-27B-FP8/NVFP4. Derivation verifier - GO with caveat: AIME 94.1 is top-tier but possibly overfit (Kaitchup), GSM8K/MATH unreported; calibrate on our own arithmetic gold set before trusting.

## 1. What exactly is the latest Qwen release

### 1.1 Qwen3.8-Max (API-only, weights pending)

- Previewed 19 Jul 2026, launched 3 Aug 2026; 2.4T total params sparse MoE, ~95B active (~4% activation vs ~10% for Qwen3-235B-A22B); 1M token input / 128k output; native text+image+video; \$2/M in, \$6/M out, \$0.25/M cached
- Weights promised on HF/ModelScope week of 10 Aug 2026 alongside a smaller Qwen3.8-27B; as of 9 Aug neither has appeared; NO license announced for either
- Verified: `huggingface.co/Qwen/Qwen3.8-27B` returns HTTP 401; `huginnfork/Qwen3.8-27B-FP8` card states "Status: placeholder. There are no weights in this repository yet"
- **Nothing about Qwen3.8-27B is confirmed** - not dense-vs-MoE, context, modality, license, or a single benchmark. Blog specs in circulation are speculation

Sources: marktechpost.com (3 Aug + 19 Jul 2026), latent.space/p/ainews-qwen-38-max24t-and-27b-new, datacamp.com/blog/qwen3-8-max, huggingface.co/huginnfork/Qwen3.8-27B-FP8

### 1.2 Latest downloadable general models: Qwen3.6 (April 2026)

Both Apache 2.0, natively multimodal, 262,144 native context (~1M via YaRN), hybrid Gated DeltaNet (linear attention) + Gated Attention:

| Model | Released | Type | Params | Layers | License |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | 16 Apr 2026 | MoE | 35B total / 3B active, 256 experts (8 routed + 1 shared) | 40 | Apache 2.0 |
| Qwen3.6-27B | 22 Apr 2026 | Dense | 27B | 64 | Apache 2.0 |

Qwen3.6-27B: only 16 of 64 layers are full attention; 48 are Gated-DeltaNet linear layers with fixed-size recurrent state (why KV scales well with context - and why the Mamba/GDN state cache is what breaks on WSL2, §4). Both ship official FP8 checkpoints. "Thinking Preservation" across turns (UNVERIFIED, single secondary source).

Sources: github.com/QwenLM/Qwen3.6, huggingface.co/Qwen/Qwen3.6-27B / -35B-A3B / -27B-FP8, simonwillison.net/2026/Apr/22/qwen36-27b/

### 1.3 2026 timeline

| Date | Release | Open weights | License |
|---|---|---|---|
| 16 Feb | Qwen3.5-397B-A17B | Yes | Apache 2.0 |
| 24 Feb | Qwen3.5-122B-A10B, -35B-A3B, -27B | Yes | Apache 2.0 |
| 2 Mar | Qwen3.5-9B/-4B/-2B/-0.8B (201 languages) | Yes | Apache 2.0 |
| 16 Apr | Qwen3.6-35B-A3B | Yes | Apache 2.0 |
| 22 Apr | Qwen3.6-27B | Yes | Apache 2.0 |
| 20 May | Qwen3.7-Max | No | API-only |
| 1 Jun | Qwen3.7-Plus | No | API-only |
| 3 Aug | Qwen3.8-Max (2.4T MoE) | Promised wk of 10 Aug | Unannounced |

Trend: Qwen3.7 shipped API-only; open-weight cadence at flagship tier is no longer guaranteed; Qwen3.8 license silence is consistent with that drift.

## 2. Licenses

**Confirmed Apache 2.0** (LICENSE file present): Qwen3.6-27B(+FP8), Qwen3.6-35B-A3B(+FP8), nvidia/Qwen3.6-27B-NVFP4, all Qwen3.5 open checkpoints.
**Not usable today**: Qwen3.8-Max/-27B (no weights, no license), Qwen3.7-Max/Plus, Qwen3.6-Plus/Max-Preview (API-only).

**Geo-restriction rumour (Qwen3.8): CONTESTED.** AINews reported US/EU/UK/KR restrictions; likely a conflation with MiniMax H3 (4 Aug 2026, documented restriction). Status UNRESOLVED - read the LICENSE at release, do not assume Apache 2.0 by precedent.

## 3. Benchmark standing

### 3.1 "Beats everything" - flagship vs small

Flagship (Qwen3.8-Max, API-only): SWE-bench 87.3 (ahead of GPT-5.5 82.6, behind Claude Opus 4.8 89.2); Vals Index #2 open-weight 66.1; Frontend Code Arena #4 (1,668 Elo); Vision Arena #2. Near-frontier, not clearly ahead of Anthropic - "beats everything" overstates.
Small (Qwen3.6-27B): "flagship-level coding in a 27B dense" - surpasses Qwen's own 397B flagship on coding; Willison: "outstanding for a model of this size."

### 3.2 Vendor-reported, runnable sizes

| Benchmark | Qwen3.6-27B | Qwen3.6-35B-A3B | Qwen3.5-27B |
|---|---|---|---|
| AIME 2026 | 94.1 | 92.7 | - |
| MMLU-Pro | 86.2 | 85.2 | 86.1 |
| GPQA Diamond | 87.8 | 86.0 | 85.5 |
| IFEval | - | - | **95.0** |
| IFBench | - | - | **76.5** |
| SWE-bench Verified | 77.2 | 73.4 | 72.4 |

**GSM8K/MATH not reported for any Qwen3.5/3.6** - the capability our derivation verifier needs has NO published number.

### 3.3 CRITICAL independent finding - Qwen3.6 regressed on instruction following

Kaitchup head-to-head (Qwen3.6-27B vs Qwen3.5-27B vs Gemma 4 31B): IFBench - "Qwen3.6 27B is significantly worse than its predecessor"; GPQA-D - card number did not reproduce; AIME - 3.6 ahead but "suggesting possible specialized fine-tuning" (overfit suspicion). Corroborated independently by HF community reports on 35B-A3B: "more factual hallucinations," "forgets the instructions before conclusion." Two independent signals, same conclusion. **A grounding judge is an instruction-following + schema-compliance task - on the axis that matters most, the newer model is worse.**

### 3.4 Aggregates

Artificial Analysis Intelligence Index: Qwen3.6-27B 38, Qwen3.5-27B 35, Qwen3.6-35B-A3B 32. GPT-5.5 leads at 60; GLM-5.2 is AA's top open model (Qwen is NOT the open-weight leader there). Verbosity warning: 35B-A3B emitted 150M output tokens vs 38M median during AA eval - force non-thinking for judge cost control.

## 4. Fit on the RTX PRO 6000 Blackwell 96GB (sm_120) - READ BEFORE DEPLOYING

### 4.1 What fits

| Checkpoint | Format | Weights | Single 96GB? | vLLM | License |
|---|---|---|---|---|---|
| Qwen3.6-27B | BF16 | ~55.6 GB | Yes, tight | >=0.17 | Apache 2.0 |
| Qwen3.6-27B-FP8 | FP8 | ~27-30 GB | Yes | >=0.19 | Apache 2.0 |
| nvidia/Qwen3.6-27B-NVFP4 | NVFP4 | ~2.5x smaller | Yes | **>=0.24** | Apache 2.0 |
| Qwen3.6-27B Q4_K_M GGUF | GGUF | ~16.8 GB | Yes (llama.cpp) | n/a | Apache 2.0 |
| Qwen3.6-35B-A3B-FP8 | FP8 | ~35 GB | Yes | >=0.19 | Apache 2.0 |
| Qwen3.5-27B (+AWQ ~15GB) | BF16/INT4 | ~54 GB | Yes | main | Apache 2.0 |
| Qwen3.5-122B-A10B | FP8 | ~122 GB | **No** | - | Apache 2.0 |

NVFP4 accuracy recovery excellent (MMLU-Pro 86.3 vs 86.1 FP8).

### 4.2 sm_120 / WSL2 issues

- **(a) WSL2 + Blackwell + hybrid-Mamba state cache OOM (THIS BOX IS WSL2)**: official Qwen3.6-27B-FP8 repo discussion #10 - loads, then OOMs allocating Mamba state cache despite 50GB free (~16GB non-PyTorch overhead invisible to allocator). Workaround that reportedly works on identical hardware: vLLM **0.23.0**, `--gpu-memory-utilization 0.80`, `--max-num-seqs 256`, **bf16 KV cache**, leave PYTORCH_CUDA_ALLOC_CONF unset
- **(b) NVFP4 MoE CUTLASS kernels FAIL on sm_120** (all 80 TMA grouped-GEMM tactics fail; garbage output or Marlin W4A16 fallback only). Dense NVFP4 (27B) reported working. Do not run 35B-A3B at NVFP4
- **(c)** vLLM 0.19.1 has a WSL2 MTP bug (nightly needed); FlashInfer JIT needs `cuda-nvcc-13-0 cuda-cccl-13-0 cuda-cudart-dev-13-0` + ninja; `expandable_segments:True` breaks MTP fork on WSL2; Mamba cache block limit -> set `--max-num-seqs 512` max; bare-metal Linux +27% decode vs WSL2

### 4.3 Measured throughput on this exact card (Millstone AI + community)

Qwen3.6-27B-FP8, 1x RTX PRO 6000: peak 189.3 tok/s @ 5 concurrent 1K ctx; TTFT 170ms @ 1K; 32K ctx 92.5 tok/s. 35B-A3B FP8: 170-200 tok/s single-user configs. **Our judge workload (2-8k ctx) sits at the sweet spot; use BF16 KV (FP8 KV only pays past ~200K ctx and is the OOM-workaround setting anyway).** Peak sustained ~681K tokens/hour on one card at FP8.

### 4.4 llama.cpp fallback

Hybrid GDN needs latest llama.cpp; GGUF quants abundant (unsloth etc.); safe fallback if vLLM Mamba-cache OOM persists, at large batched-throughput cost.

## 5. Fitness for our use cases

### 5.1 Judge (hallucination/grounding certification)

Direct evidence thin: no RewardBench-2/JudgeBench/RAGTruth/FaithBench numbers for 3.5/3.6. Proxies: Qwen3.5-27B tops open-source HallusionBench 0.700 (visual - weak transfer); Qwen3.5-9B 0.859 faithfulness in a RAG study; fine-tuned Qwen3.6-27B judges exist emitting JSON verdicts; strong family precedent (Qwen3-235B, Qwen2.5-32B as judges). Mechanics: `enable_thinking:false` (KNOWN BUG: dify reports 3.6 under vLLM "cannot switch the thinking mode normally" - VERIFY token accounting), `--language-model-only` (skip vision tower), `--reasoning-parser qwen3`.
**Recommendation: serve Qwen3.6-35B-A3B-FP8 primary (3B active = batching), but BAKE-OFF vs Qwen3.5-27B on our gold set first - score verdict agreement AND schema-compliance rate.**

### 5.2 Synthetic generator

Strong fit, fewest caveats: 262K context (whole docs as evidence), excellent multilinguality, Apache 2.0 outputs unencumbered, verbosity is a feature for phrasing variants. **Recommendation: Qwen3.6-27B-FP8.** Caveat: 3.6's reported factual-hallucination increase is fine for corruption generation, risky for faithful claims - consider faithful claims via Qwen3.5-27B, corruptions via 3.6-27B if single-model label noise shows.

### 5.3 Derivation verifier

AIME 94.1 (27B) is frontier-level (GPT-5.2: 96.7, Claude Opus 4.6: 93.3) BUT: suspected AIME overfit (Kaitchup), one card number already failed to reproduce (GPQA-D), and GSM8K/MATH-class step-checking - the actual capability we need - has NO published number. Thinking mode ON needed (conflicts with judge serving profile - separate profile or model). **Recommendation: Qwen3.6-27B thinking-on, calibrated on our own derivation gold set before any lane admission; treat AIME as an upper bound with unknown transfer.**

## 6. The 14B-80B open band

| Model | Type | Active | Batching | Note |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | MoE | 3B | Best | judge pick pending bake-off |
| Qwen3.6-27B | Dense | 27B | Moderate | highest single-card quality |
| Qwen3.5-35B-A3B | MoE | 3B | Best | better IF than 3.6 |
| Qwen3.5-27B | Dense | 27B | Moderate | IF champion (IFEval 95.0) |

No separate "-Instruct" SKU in 3.5/3.6 - unified checkpoints, thinking toggled at serve/request time.

## 7. Open questions register

1. Qwen3.8-27B everything - watch huggingface.co/Qwen week of 10 Aug 2026
2. Qwen3.8 geo-restriction - contested, likely MiniMax H3 conflation; read LICENSE at release
3. WSL2 Mamba-cache OOM workaround at our concurrency - reproduce with vLLM 0.23.0 / bf16 KV / max-num-seqs 256
4. 3.6 vs 3.5 for OUR judge rubric - head-to-head agreement study on our gold set (evidence favours 3.5 on IF)
5. Arithmetic reliability - no published data; build our own derivation gold set
6. Does enable_thinking:false actually take effect on 3.6 under vLLM - verify empirically
7. Dense NVFP4 on our sm_120 card - likely fine, unconfirmed

## 8. Recommended next actions

1. Do not wait for Qwen3.8 (no weights/license/specs); re-check after 10 Aug 2026
2. Stand up Qwen3.6-35B-A3B-FP8 on vLLM 0.23.0+: `--language-model-only --reasoning-parser qwen3`, enable_thinking:false, bf16 KV, `--gpu-memory-utilization 0.80 --max-num-seqs 256`; confirm no Mamba-cache OOM under WSL2
3. Judge bake-off on our gold grounding set: Qwen3.6-35B-A3B vs Qwen3.6-27B vs Qwen3.5-27B - verdict agreement AND schema-compliance rate (the IF regression makes schema compliance first-class)
4. Generate synthetic batches with Qwen3.6-27B-FP8, detached, parquet-checkpointed
5. Build the arithmetic-derivation gold set before admitting the derivation-verifier lane

## Primary sources

github.com/QwenLM/Qwen3.6 · huggingface.co/Qwen/Qwen3.6-27B(-FP8, discussion #10) · huggingface.co/Qwen/Qwen3.6-35B-A3B (discussion #50) · huggingface.co/Qwen/Qwen3.5-27B · huggingface.co/nvidia/Qwen3.6-27B-NVFP4 · recipes.vllm.ai/Qwen/Qwen3.6-27B and -35B-A3B · discuss.vllm.ai/t/sm120-rtx-pro-6000-nvfp4-moe-performance-report-qwen3-5-397b/2536 · github.com/lastloop-ai/vllm-blackwell-guide · millstoneai.com (Qwen3.6-27B-FP8 1x RTX PRO 6000 benchmark) · kaitchup.substack.com/p/qwen36-27b-vs-qwen35-27b-vs-gemma · artificialanalysis.ai · simonwillison.net/2026/Apr/22/qwen36-27b/ · marktechpost.com (3 Aug 2026) · latent.space/p/ainews-qwen-38-max24t-and-27b-new

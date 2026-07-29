# Small Reasoning Bases for a Trained Grounding Head

Survey of sub-350M bases that could be trained into a claim-vs-evidence grounding head, seeded from `cactus-compute/needle` and the Simple Attention Networks paper. Ranked on expected post-training quality and on strength of *independent* evidence, with vendor-only numbers labelled as such throughout.

## Constraints applied

- **Hard ceiling** - under 350M parameters, non-negotiable; several commonly-assumed-small models are over and are called out below
- **Architecture-agnostic** - encoder bases compete on equal terms with decoders; the task is what matters
- **Export is informational only** - GPU training and serving are acceptable, export noted per row but not ranked on
- **Bar to beat** - macro-F1 0.824 on 2,752 verified claims / 21 languages, currently from a 3-stage cascade totalling ~1.41B params; only the 278M NLI stage is itself in-band

## Ranked table

| Model | Params | Ctx | Vocab | Licence | Reasoning-trained? | Independent evidence? | Export path | Verdict |
|---|---|---|---|---|---|---|---|---|
| **jhu-clsp/mmBERT-base** | 307M (110M non-emb) | 8192 | 256k | MIT | No - MLM only | **Yes, strong** - ICML 2026 poster; 2 third-party groups chose it as backbone for this exact task | ONNX via `optimum`, ModernBERT arch, SDPA fallback | **Train this.** Strict upgrade on the incumbent NLI stage |
| **jhu-clsp/mmBERT-small** | 140M (42M non-emb) | 8192 | 256k | MIT | No | Yes - same paper and recipe | Same | **Train this** as the capacity-curve control |
| **google/gemma-3-270m** | ~268M measured (170M is embedding) | 32k | 256k | Gemma (custom, gated) | No - base LM | **Vendor-only** on the 270M tier | `onnx-community/gemma-3-270m-it-ONNX` | Best in-band decoder; tokenizer survives 21 languages |
| microsoft/mdeberta-v3-base | 278M | 512 | 250k | MIT | No | Yes - widely reproduced | ONNX, already shipped | Incumbent. 512 ctx is its ceiling |
| BAAI/bge-reranker-base | 278M | 512 | 250k | MIT | No | Yes - BEIR/MIRACL | ONNX | Lateral move; smaller sibling of what we run |
| EuroBERT/EuroBERT-210m | 210M | 8192 | 128k | Apache-2.0 | No | Partial - paper only | ONNX | Only 15 languages, short of 21 |
| jhu-clsp/ettin-encoder-150m | 150M | 8192 | 50k | MIT | No | **Yes** - the controlled encoder-vs-decoder study | ONNX | English-only. Source of our key evidence, not a candidate |
| answerdotai/ModernBERT-base | 149M | 8192 | 50k | Apache-2.0 | No | Yes | ONNX | English-only |
| BSC-LT/MrBERT | undisclosed | 8192 | adapted | CC-BY-4.0 | No | Paper only, thin | ONNX | Underspecified; revisit if sizes published |
| PleIAs/Baguettotron | 321M (80 layers) | n/s | n/s | Apache-2.0 | **Yes** - SYNTH thinking traces | **No** - vendor-only | HF-standard | Refuted zero-shot here; traces are English-only |
| PleIAs/Pleias-RAG-350M | ~350M, **at ceiling** | n/s | n/s | Apache-2.0 | Yes - grounded-citation traces | **No** - vendor-only | HF-standard | Refuted zero-shot here. Verify exact count before use |
| Cactus-Compute/needle | 30.4M on HF (vendor says 26M) | n/s | **8192** | MIT | No - function-call distil | **Partial, and negative** - one third-party CPU test | `onnx-community/needle-onnx`, fp32 only | **8k vocab kills it** for 21 languages |
| PleIAs/Monad | 56M | n/s | n/s | Apache-2.0 | Yes - SYNTH | **Failed** - could not reproduce own MMLU | HF-standard | Dead |
| HuggingFaceTB/SmolLM2-135M | 135M | 8192 | 49k | Apache-2.0 | No | Yes - open recipe | ONNX | English-centric; too small, wrong tokenizer |
| facebook/MobileLLM-R1-140M | 140M | 4k | 128k | **FAIR NC** | Yes - math/code traces | Paper + OpenReview | ONNX | Non-commercial licence disqualifies |
| EleutherAI/pythia-160m | 160M | 2048 | 50k | Apache-2.0 | No | Yes | ONNX | Obsolete; research control only |

### Over the ceiling - do not propose

- **LiquidAI/LFM2-350M and LFM2.5-350M** - ~357M, measured from `model.safetensors` 714 MB bf16 ÷ 2. The name says 350M, the weights say otherwise. **OUT**
- **ibm-granite/granite-4.0-h-350m** - vendor says "~350M", hybrid SSM; sits exactly on the line, treat as OUT until config-verified
- **HuggingFaceTB/SmolLM2-360M** (362M), **Qwen/Qwen3-0.6B** (596M), **Qwen/Qwen2.5-0.5B** (494M), **Qwen/Qwen3.5-0.8B** (800M), **Qwen3-Embedding/Reranker-0.6B** (596M), **answerdotai/ModernBERT-large** (395M), **TinyLlama-1.1B**, **Falcon-E-1B**, **DeepSeek-R1-Distill-Qwen-1.5B** - all OUT
- **jinaai/jina-reranker-v2-base-multilingual** - 278M and in-band, but CC-BY-NC-4.0. Licence blocker, not a size one

### How each count was determined

- Config dims + vendor card - mmBERT (both), mDeBERTa-v3-base, EuroBERT, Ettin, ModernBERT, SmolLM2, Pythia, MobileLLM-R1
- Measured from safetensors ÷ 2 at bf16 - Gemma-3-270M (536 MB → ~268M), LFM2-350M (714 MB → ~357M)
- HF safetensors metadata - Needle (30.4M reported, vendor blog says 26M; **the vendor number is 15% low**)
- Vendor claim only, unverified - Baguettotron 321M, Pleias-RAG-350M, Monad 56M, granite-4.0-h-350m

## The seed: Needle and Simple Attention Networks

Needle is real, MIT-licensed, 3.3k GitHub stars and actively maintained - it is not vapourware. It is also not what our task needs.

- **What it is** - 26-30M encoder-decoder with the FFN layers deleted entirely; 12 encoder layers, 8 decoder layers, d=512, 8 heads / 4 KV, RoPE, shared embeddings, **SentencePiece BPE vocab 8192**
- **Training** - 200B tokens pretrain on 16× TPU v6e in 27 h, then 2B tokens of function-call data in 45 min, distilled from Gemini 3.1 Flash Lite
- **Recipe specifics** - gated residuals with learnable scalar gates, ZCRMSNorm, Muon optimizer enforcing orthogonality on attention projections, INT4 QAT injected every 100 steps as regularisation noise, token-level loss weighting at 4.0× on argument values, CLIP-style contrastive head for large tool sets
- **Vendor claim** - "beats FunctionGemma-270m, Qwen-0.6B, Granite-350m, LFM2.5-350m on single-shot function call", with no published benchmark table
- **Independent evidence, and it is unflattering** - an autonomous third-party CPU test over 50 queries found prompted Qwen3-0.6B at 84% tool accuracy and 100% parse success against Needle's 72% and 84%. Needle was 4.2× faster. It loses on quality on its own home task
- **Paper status** - arXiv 2607.18363, first-authored by Cactus's own founder, July 2026 preprint. **No independent reproduction found.** No HN or OpenReview critique located

The paper's actual result is more interesting than the marketing. At **matched depth** the standard transformer wins by 0.47 nats and at matched FLOPs by 0.26 nats - deleting FFNs in place is expensive. Only when the freed budget is reallocated into attention depth does the gap close to 0.006 nats at matched parameters. The headline is therefore "not worse", never "better".

The one finding that genuinely bears on grounding is the failure-mode decomposition: SANs are worse at **parametric recall** (Lambada, FFN ahead at every budget) and better at **context-grounded answering** (SciQ with passage provided, SAN 0.742 vs FFN 0.661, margin growing with budget). Everything else - ARC, PIQA, HellaSwag, WinoGrande, MMLU - sits at chance at these scales and cannot discriminate.

## Question 2 - does Needle's recipe transfer to pairwise (claim, evidence) classification?

**Definite answer: no. The architecture insight transfers; the recipe does not.**

What the recipe actually does is train a **generative sequence model** - the decoder's entire job is emitting a JSON function call token by token, and the loss weighting, the 4.0× argument-value weight, the QAT schedule and the contrastive tool-retrieval head all exist to serve that emission. For a binary or 3-way grounded/not-grounded verdict you delete the decoder outright. What remains is a 12-layer, d=512, FFN-free bidirectional encoder with an 8192-token vocabulary - at which point you are not using Needle's recipe, you are using a small, undertrained, monolingual encoder, and better ones already exist in the same band.

Three specific blockers, in order of severity:

- **Vocabulary 8192 is fatal at 21 languages** - the incumbent runs a 250k vocab and mmBERT runs 256k. An 8k English-leaning SentencePiece BPE shreds Polish, Arabic, Korean or Finnish claims into character-level fragments, which destroys both the effective context and the token-level alignment that a claim↔span decision depends on. This alone disqualifies the published checkpoint
- **The shipped checkpoint is function-call post-trained** - 2B tokens of narrow, destructive specialisation. Any grounding work would have to start from the pretrained SAN checkpoints, which are 6M-87M models evaluated only on LM loss, at chance on every MMLU-class benchmark
- **The motivation has evaporated under our constraints** - SAN's entire selling point is memory-bandwidth reduction on edge silicon. With inference explicitly deferred and a 96 GB card available, we would be adopting an architecture whose best controlled result is statistical parity, in exchange for nothing

The part worth keeping is the *hypothesis*, not the artefact: a task that requires zero parametric recall and pure context-grounded alignment is precisely where attention-only capacity allocation should be least penalised, and the SciQ result is direct, if small-scale, evidence for that. That argues for spending parameters on attention depth and context rather than on FFN width and stored knowledge - which is an argument for a deep, long-context, large-vocabulary **encoder**, and mmBERT-base is exactly that model, already trained.

**Blunt summary** - Needle is a well-executed edge tool-calling model with an honest paper attached and a founder-authored evidence base. It is not a grounding base. Treat the paper as a design prior, not as a shortlist entry.

## Question 1 - the three bases to actually train, ranked

### 1. jhu-clsp/mmBERT-base (307M, MIT)

The only candidate in the band with independent, third-party, on-task evidence.

- **Two separate groups independently selected it as the backbone for multilingual grounding verification** - PsiloQA (arXiv 2510.04849) found mmBERT-base fine-tuned leads 12 of 14 languages on both AP and IoU for span-level hallucination detection, and the LettuceDetect span-grounding work (arXiv 2607.00895) shipped `LettuceDetect-mmBERT-base` as its multilingual encoder detector at 0.642 F1
- **Peer-reviewed venue** - ICML 2026 poster, not a vendor preprint. Weights, data and code all released
- **XNLI 77.1 vs XLM-R 74.6**, TyDiQA 74.5 F1 vs 70.5 - it beats the tokenizer family our incumbent NLI head is built on
- **8192 context vs the incumbent's 512** - this is the single biggest structural gain available. Today the NLI stage truncates evidence; it would stop doing so
- **1833 languages** with a curriculum that anneals in low-resource languages during decay, against 21 in our gold - genuine headroom rather than marginal coverage
- Not reasoning-trained, and that is fine: the evidence says this task rewards a strong bidirectional aligner, not a chain-of-thought generator

### 2. jhu-clsp/mmBERT-small (140M, MIT)

Trained in the same run as #1, same tokenizer, same data, 42M non-embedding parameters.

- Its value is **diagnostic, not competitive** - two points on a capacity curve at near-zero marginal cost
- If small lands within ~1 point of base on our soft labels, the tier is **label-bound, not capacity-bound**, and no amount of base-shopping will move the number. That is the most valuable thing to learn before committing budget
- Also the natural fallback if latency ever returns as a constraint

### 3. google/gemma-3-270m (~268M, Gemma licence)

The hedge, in case the decoder framing turns out to matter.

- **The only in-band decoder with a tokenizer that can handle 21 languages** - 256k vocab, 140+ languages in the 6T-token pretraining mix. Every other in-band decoder is either English-centric (SmolLM2-135M), European-only (Baguettotron, Pleias-RAG), 8k-vocab (Needle), or non-commercially licensed (MobileLLM-R1)
- Google positions it explicitly as **a fine-tuning base rather than a usable zero-shot model**, which is the correct posture here and sidesteps the zero-shot refutation directly - we are not asking it to judge, we are asking it to be initialised weights
- **Evidence is vendor-only at this tier.** Google's developer blog and the derived FunctionGemma-270m are the sources; no independent reproduction of the 270M claims was found. Label accordingly
- **Structural weakness to accept going in** - 170M of the 268M is the embedding table, leaving ~100M of transformer. The controlled Ettin study (paired encoders and decoders, identical data and architecture, 17M-1B) found a 400M encoder at **91.3 MNLI vs the matched 400M decoder at 88.2**, and states plainly that "a 400M encoder outperforms a 1B decoder on MNLI". A 268M decoder with 100M of usable transformer is starting behind a 307M encoder with 110M of it

### Deliberately not ranked

- **Baguettotron and Pleias-RAG-350M** - reasoning-trace-trained and in-band, so they are legitimately still open as *training* bases, and Pleias-RAG even emits citations natively. But every published number is vendor-only, the sibling model on the same corpus failed to reproduce its own MMLU, and the reasoning traces are English-only against a 21-language target. If a decoder base is wanted, Gemma-3-270M dominates them on tokenizer and on evidence
- **Anything trained only on English** - Ettin, ModernBERT, SmolLM2. Strong models, wrong problem

## Question 3 - the cheapest decisive experiment

**Distil the entire cascade into one mmBERT-base cross-encoder and score it on the same verified gold. Single-digit GPU-hours, zero annotation, no architecture search.**

- **Data** - the ~111,800 cached (claim, chunk) pairs with calibrated grounded-probabilities, split **by claim not by pair**, holding out every pair whose claim appears in the 2,752 verified gold
- **Objective** - single-logit cross-encoder on `[claim, chunk]`, **soft-label distillation** against the cascade's calibrated probability (KL or MSE), not thresholded hard labels. The calibrated probabilities are the asset; hard labels throw away most of the signal
- **Nothing else** - no reasoning traces, no generation, no prompt engineering, no ensemble
- **Cost** - ~112k pairs × 3 epochs at 8k context is hours on one 96 GB card. Run mmBERT-small identically in the same session for the capacity-curve point

**Decision rule** - does one 307M model, at a threshold tuned on the training split only, reach macro-F1 within noise of 0.824 on the verified gold?

Why this and not something else:

- It tests the **actual claim under test** - that one sub-350M model can replace two 568M cross-encoders plus a 278M NLI head - without confounding it with base selection
- It has a **known ceiling**, which is what makes it decisive. Distillation cannot exceed its teacher. If it lands near 0.82 you have collapsed 1.41B params into 307M and the base question is settled. If it lands at 0.79 the binding constraint is the **cascade's own label quality**, and no choice of base fixes that
- **Kill criterion** - below ~0.75, stop shopping for models. The soft labels are too noisy and the correct next move is label repair, not a bigger base
- The mmBERT-small arm costs almost nothing and answers "is this tier data-bound or capacity-bound", which determines whether a base upgrade is even the right lever

**Caveat that must be stated up front** - distillation caps at the teacher, so this experiment can match 0.824 but cannot beat it. Exceeding the bar requires signal the cascade does not have, which means a **second stage**: supervised fine-tuning on the 2,752 human labels, cross-validated, on top of the distilled checkpoint. That is the only route past the teacher, and it is experiment two, not experiment one. Run it only if experiment one clears ~0.80.

## Sources

Primary sources only; vendor-published material is marked in-line above.

- [A Controlled Study of Attention-Only Transformers, arXiv 2607.18363](https://arxiv.org/abs/2607.18363)
- [cactus-compute/needle](https://github.com/cactus-compute/needle) and [Simple Attention Networks design doc](https://github.com/cactus-compute/needle/blob/main/docs/simple_attention_networks.md)
- [Cactus-Compute/needle on HuggingFace](https://huggingface.co/Cactus-Compute/needle), [onnx-community/needle-onnx](https://huggingface.co/onnx-community/needle-onnx)
- [Independent Needle vs Qwen3-0.6B CPU benchmark](https://heyneo.com/blog/needle-26m-vs-qwen3-0.6b-cpu-function-call-benchmark)
- [mmBERT paper, arXiv 2509.06888](https://arxiv.org/pdf/2509.06888) and [ICML 2026 poster](https://icml.cc/virtual/2026/poster/62254)
- [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base), [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small)
- [PsiloQA: Multilingual Span-Level Hallucination Detection, arXiv 2510.04849](https://arxiv.org/abs/2510.04849)
- [Beyond Document Grounding: Span-Level Hallucination Detection, arXiv 2607.00895](https://arxiv.org/html/2607.00895)
- [Ettin / Seq vs Seq paired encoders and decoders, arXiv 2507.11412](https://arxiv.org/html/2507.11412v1)
- [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m), [Google developer blog, vendor-only](https://developers.googleblog.com/en/introducing-gemma-3-270m/)
- [PleIAs/Baguettotron](https://huggingface.co/PleIAs/Baguettotron), [PleIAs/Pleias-RAG-350M](https://huggingface.co/PleIAs/Pleias-RAG-350M)
- [EuroBERT, arXiv 2503.05500](https://arxiv.org/pdf/2503.05500), [MrBERT, arXiv 2602.21379](https://arxiv.org/pdf/2602.21379)
- [LiquidAI/LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M), [LFM2 Technical Report, arXiv 2511.23404](https://arxiv.org/pdf/2511.23404)
- [MobileLLM-R1, arXiv 2509.24945](https://arxiv.org/abs/2509.24945), [ibm-granite/granite-4.0-h-350m](https://huggingface.co/ibm-granite/granite-4.0-h-350m)

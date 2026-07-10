# Evidence-chunked open dual-judge grounding pipeline

Reproducible recipe for labeling production traces for hallucination with open-model judges over SAT-segmented, semantically-retrieved evidence windows. Replaces the whole-document judge feed (truncated a third of the corpus) and the R1 dual-judge (recall 0.36). Runs entirely on the local RTX PRO 6000 96GB, zero external API tokens. The pipeline is two notebooks - `01-kj-grounding-dataset-pipeline` (retrieve + open ensemble judge → `records_open`) and `02-kj-crosslingual-augmentation` (NLLB MT → `records_curated`/`records_final`) - and produces the frontier-free `golden_v6`, replacing the Claude-judged `golden_v5`.

## Why the previous run failed

Established empirically over the 636 gold-labeled traces (`data/processed/golden_v5`), full per-judge confusion vs Claude gold - not aggregate κ:

- **R1-70B w4a16 is a lenient rationalizer** - hallucination recall 0.360, misses 52.8% of claims that gold and Gemma both independently call hallucination, with full evidence in context; control accuracy on supported claims 0.981 (one-directional error → reasoning bias, not quant damage) → drop R1
- **Gemma-3-27B-it carries the system alone** - recall 0.828, precision 0.941, macro-F1 0.908, calibrated flag rate (34% vs true 38%)
- **Whole-doc feed truncated 208/636 traces (33%)** at 18432-token context - evidence is median ~12k tokens, p99 ~64k, max ~184k; some traces carry entire manuals
- **The dual-agree DISCARD gate dumped 29% of claims** (87% of them hallucinations R1 botched) to buy its 0.920 accuracy - it routed the hard cases to the trash rather than judging them → replace discard with a tiebreaker
- **fp8 KV garbled generation** on the fp8-dynamic checkpoints (E26-H91) → judges run bf16 KV

## Bake-off results and the two failure modes

Five open judges scored over the 636 gold traces / 5857 eval claims: Gemma-3-27B, Llama-3.3-70B, Qwen2.5-72B, and gpt-oss-120b at low and high reasoning effort. The first pass at CHUNK=15 / 512-1024 tokens looked like universal truncation but was two distinct bugs - the correction matters more than the first read.

- **Parser fragility, not token truncation, sank the dense judges** - the lenient first-`[`-to-last-`]` slice failed on a bare 1-item object (Gemma's small chunks), on trailing prose containing `]` (Llama, at every chunk size), and on newline-joined objects; the fix is a regex that extracts every flat `{...}` verdict object regardless of wrapper, key order, or trailing text
- **Reasoning-length truncation is real, and more tokens is the fix, not fewer claims alone** - gpt-oss emits a long `<think>` before the array; at 1024 tokens it truncated 80%, at 3072 tokens with CHUNK=5 it parses 0.997. The reasoning is ~900-1900 output tokens for a 5-claim chunk, independent of chunk size
- **Raw-text capture is mandatory** - the bake-off saved only parsed labels, so a parse failure was undiagnosable without a full re-run; the pipeline now writes `raws_<candidate>.parquet` per chunk
- **Gemma is not the anchor** - it is the lenient judge (hallucination recall 0.771) and returns a bare `\n` for ~18% of chunks (multimodal-model quirk); dropped from the ensemble
- **gpt-oss (reasoning) is the best judge, but honestly ~0.90 recall, not 0.96** - the 0.959 F1 from the truncated bake-off was inflated by a survivorship bias (only the easy short chunks completed); the clean full-corpus number is recall ~0.90, macro-F1 ~0.90

## Corrected ensemble - chat template, real effort, no truncation

The bake-off's "gpt-oss-low + gpt-oss-high ensemble" was later found inert: `llm.generate` on raw completion prompts silently ignores `reasoning_effort`, so low and high were one config run twice, and the 0.963 "agreement" was run-to-run nondeterminism, not judge diversity. The fix makes the ensemble genuinely diverse and lossless.

- **Chat template + explicit effort** - each judge runs through gpt-oss's harmony chat template with `reasoning_effort` set (`apply_chat_template(..., reasoning_effort=...)`); on a labeled sample, chat-template low beats the old raw path (F1 0.913 vs 0.896, κ 0.820 vs 0.787), and low vs high now genuinely differ
- **low + medium, not low + high** - high-effort reasoning runs away past 17k tokens on ~11/1456 prompts (the no-truncation assert caught it) and is no more accurate than low; medium is the stable diverse partner (parse 1.000, no runaway)
- **No-truncation guarantee** - length-bucket the prompts by size, then re-run any generation with `finish_reason=="length"` at a doubled budget, looping until none are capped, and `assert` 0 capped; a truncated verdict is a LOST label (survivorship bias toward easy claims), never accepted. Validated: medium hit 1/1456 at 6144 → re-ran at 12288 → 0 capped
- **Per-judge vs gold (clean full corpus)** - gpt-oss-low recall 0.911 / F1 0.875 / κ 0.808 / parse 1.000; gpt-oss-medium recall 0.909 / F1 0.868 / κ 0.797 / parse 1.000
- **Agreed verdict vs gold** - low vs medium agree 0.956; the agreed verdict is κ 0.848, macro-F1 0.924, hallucination recall 0.932 over 5268 claims - beats the old inert ensemble (κ 0.831)
- **`records_open.parquet`** - 5268 dual-agreed records, corrected to 5242 by the adjudication (88 flukes flipped toward gold, 26 header fragments dropped); `origin = open_gptoss_ensemble`

## Adjudication and golden v6 (the frontier break-off)

The open ensemble disagreed with the Claude gold on 361 of 5268 claims (6.9%). A rulebook-informed gpt-oss high-effort third judge adjudicated all 361 against their evidence, settling whether each is a gold error or an ensemble fluke.

- **Adjudication rulebook** - `docs/auxiliary/grounding-adjudication-rulebook.md`; the entailment principle (a claim is SUPPORTED only if the evidence entails it, specificity matching): a range needs range evidence, a generalization must cover all stated values, a single value cannot license a range, an exact value is not established by a range, negatives are unverifiable in a local snippet
- **~76% of disagreements are gold v5 errors** - independently confirmed at the same 76% rate on both the 303 clear-factual (229 gold-error / 74 fluke) and the 58 ambiguous cases; gold v5 systematically labels unsupported claims as supported (177 of 229 gold errors are gold=supported / ensemble=hallucination)
- **golden v6 is the frontier-free gold** - `data/processed/golden_v6/golden_v6.parquet`, 8800 rows: eval 5242 (gpt-oss low+medium ensemble, rulebook-adjudicated), augmentation 2758 (NLLB MT, bge-m3 fidelity gated), eval_vitaminc 800 (public VitaminC); no Claude in the label path, provenance stamped in `verifier_model` / `translator_model`
- **golden v5 archived** - moved to `data/processed/@archive/golden_v5/` and `s3://general-purpose/groundrails/data/processed/@archive/golden_v5/`; the open ensemble is the more correct labeler, so v6 replaces it

## Data provenance

- **`data/raw/raw_v5/raw_v5.parquet`** - 1434 client traces; `source_text` is the production RAG's assembled retrieval bundle ("Summaries and indices of source documentation for documents matching user input"), NOT the full corpus and NOT verbatim docs; 619 distinct bundles, median 45k chars, max 681k
- **Cross-lingual** - questions/answers in 10 languages (1041 en, 145 fr, 70 nb, 65 es...), evidence always English; translation fields exist only in the processed `golden_v5`, not the raw
- **The bundle is the ground truth** - it is what the original agent reasoned from, so claims are judged against it; the retrieval layer selects the right sub-span, never expands beyond the bundle
- **`data/processed/golden_v6/golden_v6.parquet`** - the current frontier-free gold (oss-driven); `role=="eval"` rows carry `claim`, `source_text` (constant per trace), `label` (1 SUPPORTED / 0 UNSUPPORTED hallucination), `lang`/`lang_norm`, plus `verifier_model` provenance. The Claude-judged `golden_v5` is retained only as reference under `@archive/golden_v5/` and was used as the comparison baseline that surfaced the 76% gold-error finding

## Hardware and environment

- **Card** - RTX PRO 6000 Blackwell Max-Q 96GB, sm_120, `nvidia-smi` index 1 (UUID GPU-a44e514a); align CUDA to nvidia-smi with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, never default to index 0 (24GB PRO 4000, OOMs 70B)
- **Judge venv** - `/home/lab/.venvs/groundrails-judge/bin/python` (vLLM 0.24.0)
- **vLLM env** - `VLLM_WSL2_ENABLE_PIN_MEMORY=1` (NVFP4/MoE UVA), `VLLM_ATTENTION_BACKEND=FLASHINFER`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`; MoE adds `VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0` (DeepGEMM SF bug on sm_120)
- **Isolation** - one judge model per subprocess; a CUDA assert poisons the whole context, and vLLM v1 in-process teardown never frees VRAM → the second model OOMs

## Pipeline stages

Each stage reuses an existing `src/groundrails` component; the orchestration is thin glue.

1. **Segment** - `groundrails.sat.SaTSegmenter().split(text)` → multilingual sentence boundaries (OV int8 `sat-3l-sm`); handles the 10 languages without mid-sentence cuts
2. **Window** - `groundrails.chunking.recursive_chunk(text, max_chars=1500, overlap_ratio=0.25)` → overlapping `Chunk`s with char offsets; feed SAT sentences so windows respect semantic boundaries; ~1500 chars ≈ 400 tokens per window
3. **Retrieve** - bge-m3 embed → cosine top-k windows per claim, then bge-reranker-v2-m3 (relevance) + mDeBERTa-v3-NLI (entailment) over the (claim, window) pairs; **GPU-batched bf16 for data prep** - the cross-encoders are the bottleneck, so the prep sweep runs them on the 96GB card, not the shipped CPU OV cascade
4. **Anchor** - `groundrails.entity_check.find_numeric_mismatches(claim, passage)` and `find_absent_entities(claim, full_source)` → deterministic numeric/entity check for the fabricated-number hallucinations (part numbers, thresholds, alarm codes) that semantic retrieval alone rubber-stamps
5. **Assemble** - union the retrieved windows for a 25-claim chunk, cap the union at ~8k tokens → small context, no truncation
6. **Judge** - vLLM over the assembled context; strict SUPPORTED / UNSUPPORTED / NOT_A_CLAIM prompt through gpt-oss's harmony chat template with `reasoning_effort` (low, then medium); bf16 KV; length-bucketed with the no-truncation re-run loop + 0-capped assert; robust regex parser extracts the final-channel JSON

The cascade in stage 3 runs on CPU (OpenVINO int8, deterministic, already calibrated). GPU option for throughput: torchao int8-dynamic + sdpa + `torch.compile` on bge-m3 (cosine 0.998, ~748k tok/s on this card), batch b32 - swap only if the CPU cascade is the wall.

## Judge candidates and combo protocol

vLLM, fp8-dynamic weights, bf16 KV, short 2-8k context so `max_num_seqs` runs high (batching is the 30× lever):

| Judge | Quant | Throughput | Role |
|---|---|---|---|
| Gemma-3-27B-it | fp8-dynamic | high, fits with room | anchor - proven macro-F1 0.908 |
| Llama-3.3-70B-Instruct | fp8-dynamic | 275 tok/s b16 | R1 replacement, non-reasoning |
| Qwen2.5-72B-Instruct | fp8-dynamic | ~255-275 b16 | structured instruction-follower |
| gpt-oss-120b | native MXFP4 MoE | 8232 tok/s b64 | throughput king; tested at low AND high reasoning effort |

- **Solo first** - each candidate scored vs gold: hallucination recall / precision / F1, per-language κ, NOT_A_CLAIM rate
- **Then pairs** - select the pair maximizing hallucination-F1 at full kept-rate; resolve disagreements with a tiebreaker (third judge or the strictness rule), never a discard gate
- **gpt-oss two-effort probe** - unlike R1's distilled reasoning, gpt-oss was RL-trained to reason; test whether more effort helps or repeats R1's rationalization failure

## Gates (mandatory, in order)

1. **Retrieval recall ≥ 0.95 on gold-SUPPORTED claims** - does the top-k window set contain the support? If not, the retriever is the ceiling, not the judge - fix before trusting any verdict
2. **Parse validity** - normalize the byte-BPE detok markers (Ġ=space, Ċ=newline, ĉ=tab) and strip `<think>` before `json.loads`; a judge with <98% parse rate is discarded
3. **Per-language κ vs gold** - report per `lang_norm`; the Nordic slice (nb) is the cross-lingual stress test

## Metrics

- Per-judge confusion (rows=gold, cols=pred): missed hallucinations (gold=0, pred=1) vs false alarms (gold=1, pred=0)
- Hallucination-positive recall / precision / F1 (the guardrail's job), macro-F1, accuracy
- Combo kept-rate and per-language κ
- Strictness split - verbatim-support bar vs semantic-entailment bar (moves the miss count from ~90 to ~20); a labeling-policy decision, reported both ways

## Reproducible commands

```bash
# 0. environment
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1
VENV=/home/lab/.venvs/groundrails-judge/bin/python

# 1. fetch judge candidates (background, logged)
$VENV scratchpad/dl_judges.py 2>&1 | tee logs/judge-bakeoff-download.log

# 2. build chunk+retrieve cache + retrieval-recall gate (CPU, no judge needed)
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/dataset/02-kj-evidence-chunk-retrieve.ipynb 2>&1 | tee logs/chunk-retrieve.log

# 3. judge bake-off: one candidate per process (LLM_JUDGE_ONLY), resumable caches
LLM_JUDGE_ONLY=gemma   jupyter nbconvert --to notebook --execute --inplace <bakeoff.ipynb>
LLM_JUDGE_ONLY=llama70 jupyter nbconvert --to notebook --execute --inplace <bakeoff.ipynb>
# ... qwen72, gptoss-low, gptoss-high; metrics cells read all caches for the combo table
```

## Known failure modes

Each silently corrupts output - verify, do not assume:

- **R1-style rationalization** - any reasoning judge can talk itself into SUPPORTED; measure recall on consensus hallucinations, not aggregate accuracy
- **fp8 KV garble** - use bf16 KV on fp8-dynamic checkpoints
- **Gemma-3 multimodal tokenizer** - `AutoTokenizer` misbehaves (returns ~1 token); size contexts from chars (≈3.7 chars/token here) or the vLLM tokenizer, never a bare fast tokenizer without sentencepiece
- **Truncation masquerading as hallucination** - 33% of traces exceeded the old context; a "miss" in a truncated trace is unprovable, not wrong
- **PII scrubbing corrupts anchors** - the scrubber redacts article numbers to `<PHONE_NUMBER>`, making some claims unverifiable by construction; entity_check must tolerate redacted anchors
- **Cross-model verdict cache reuse** - key every cache by model id; a shared `p1_{name}.json` silently served a prior model's verdicts

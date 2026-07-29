# Open-weight verifier models under 350M parameters

Survey of open-weight grounding / faithfulness / entailment verifiers, filtered to a hard ceiling of **350M parameters**, ranked on expected grounding quality, on strength of independent evidence, and separately on expected latency. Deployment path (ONNX / OpenVINO / GPU) is informational only and is not a filter; the questions are whether anything in-band beats macro-F1 0.824 and whether one sub-350M forward can replace the three-model cascade that currently costs ~662 ms per claim warm.

## Scope and what the ceiling excludes

The 350M ceiling removes most of the specialised fact-checking literature. Every model below is confirmed over the line and is listed so it is not proposed again.

- `lytang/MiniCheck-Flan-T5-Large` - **~780M** (flan-t5-large), vendor card "size < 1B"
- `lytang/MiniCheck-DeBERTa-v3-Large` - **435M** (304M backbone + 131M embeddings, computed from DeBERTa-v3 card dims: 128K vocab × 1024)
- `lytang/MiniCheck-RoBERTa-Large` - **355M** (roberta-large), 5M over the line and therefore out
- `yaxili96/FactCG-DeBERTa-v3-Large` - **435M**, vendor card states "0.4B params"
- `yzha/AlignScore-large` - **355M** (roberta-large), out by 5M
- `KRLabsOrg/lettucedect-large-modernbert-en-v1` - **396M** (ModernBERT-large)
- `KRLabsOrg/lettucedect-610m-eurobert-*-v1` - **610M** each
- `MoritzLaurer/bge-m3-zeroshot-v2.0` - **568M** (bge-m3-retromae), and `deberta-v3-large-zeroshot-v2.0` - **435M**
- Incumbent retrieval stages `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3` - **568M** each
- Reference points only, far out of scope: `bespokelabs/Bespoke-MiniCheck-7B` 7B CC-BY-NC-4.0 (77.4 avg BAcc on LLM-AggreFact, the field ceiling), `PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct` 8B CC-BY-NC-4.0, `ibm-granite/granite-guardian-3.3-8b` 8B Apache-2.0 (76.5 avg BAcc), `google/t5_11b_trueteacher_and_anli` 11B (61.7 avg BAcc), FaithLens 8B (arXiv 2512.20182, weights unreleased)

**Consequence worth stating plainly**: not a single MiniCheck checkpoint and not a single FactCG checkpoint exists under 350M. The specialised English fact-checking line is entirely out of band, and the in-band field is dominated by span taggers and general NLI encoders.

## Parameter counts and how each was determined

| model | params | method |
|---|---|---|
| `jhu-clsp/mmBERT-base` and derivatives | 307M (110M non-embedding) | vendor model card, explicit total + non-embedding split |
| `jhu-clsp/mmBERT-small` and derivatives | 140M (42M non-embedding) | vendor model card, explicit |
| `answerdotai/ModernBERT-base` derivatives | ~150M | vendor card; HF sidebar rounds to 0.1B |
| `microsoft/mdeberta-v3-base` (incumbent NLI) | ~278M | computed - 86M backbone (vendor card) + 250K vocab × 768 = 192M embeddings |
| `microsoft/deberta-v3-base` | ~184M | computed - 86M backbone (vendor card) + 128K × 768 = 98M embeddings |
| `microsoft/deberta-v3-small` | ~142M | vendor card, explicit "44M backbone + 98M embedding layer" |
| `vectara/hallucination_evaluation_model` | ~110M | derived - `model.safetensors` is 439 MB and the card states < 600 MB RAM at 32-bit → 439e6 / 4 ≈ 110M; corroborated by the 110M figure in Tamber et al. |
| `tals/albert-xlarge-vitaminc-mnli` | 58.7M | HF safetensors metadata on the model page |
| `jhu-clsp/ettin-encoder-{17m,32m,68m}` | 17M / 32M / 68M | vendor cards, with full layer and hidden-size configs |
| `Alibaba-NLP/gte-multilingual-reranker-base` | 306M | vendor card |
| `Alibaba-NLP/gte-multilingual-base` | 305M | vendor card |
| `BAAI/bge-reranker-base` | 278M | vendor card (XLM-RoBERTa-base) |
| `EuroBERT-210m` derivatives | 210M | vendor naming, confirmed in the MultiWikiQHalluA comparison table |
| `yzha/AlignScore-base` | ~125M | computed from roberta-base backbone; distributed as a Lightning `.ckpt`, no HF sidebar figure |

## Quality ranking - in-band candidates

| rank | model | params | arch / form | licence | task | multilingual | best independent evidence | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | `KRLabsOrg/lettucedect-v2-mmbert-base` | 307M | mmBERT-base, 22L × 768, joint token tagger | Apache-2.0 | per-token supported / unsupported over the answer, emits spans | **yes**, 14 PsiloQA languages + 1,833-language backbone | none - **vendor-only** | best in-band shot, and the only one that also returns support location |
| 2 | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | 278M | mDeBERTa-v3-base, 12L × 768, cross-encoder | MIT | 3-way NLI entailment | **yes**, 27 fine-tuning languages, 100 pretrained | XNLI test, EN 0.871 → UR 0.744, vendor-run on a public benchmark | cheapest credible upgrade - strict training superset of the incumbent |
| 3 | `alexandrainst/mmBERT-small-multi-wiki-qa-synthetic-hallucinations-{lang}` | 140M each | mmBERT-small, 22L × 384, token tagger | CC-BY-4.0 | per-token supported / unsupported | **yes**, 30 separate per-language checkpoints | none - **paper-only**, evaluated on en / da / de / is only | best quality-per-millisecond bet, but 30 checkpoints is an operational tax |
| 4 | `KRLabsOrg/lettucedect-210m-eurobert-{de,fr,es,it,pl,cn}-v1` | 210M each | EuroBERT-210m, token tagger | MIT | per-token supported / unsupported | 6 languages, one model each | **vendor blog post only** | superseded by the v2 mmBERT model; keep only as a per-language ceiling check |
| 5 | `tasksource/deberta-base-long-nli` | 184M | deberta-v3-base, 12L × 768, cross-encoder | Apache-2.0 | NLI over long premises, 1,680 ctx | English only | tasksource multi-task suite, self-reported | solid English entailment head with no 512 truncation |
| 6 | `tasksource/deberta-small-long-nli` | 142M | deberta-v3-small, **6L** × 768, cross-encoder | Apache-2.0 | NLI, 1,680 ctx, 600-task training | English only | self-reported: nli_fever 71.7, doc-nli 75.0, anli-a1 57.2 | shallowest credible entailment model - the latency pick |
| 7 | `KRLabsOrg/lettucedect-base-modernbert-en-v1` | ~150M | ModernBERT-base, 22L × 768, token tagger | MIT | span tagging | English only | none - vendor-only, 79.22 example-F1 on RAGTruth | English span baseline; the large sibling is out of band |
| 8 | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` / `DeBERTa-v3-base-mnli-fever-anli` | 184M | deberta-v3-base, 12L × 768, cross-encoder | MIT | entailment / not-entailment | English only | 28-dataset zeroshot mean, vendor-run | reliable English NLI, no grounding-specific training |
| 9 | `vectara/hallucination_evaluation_model` (HHEM-2.1-Open) | ~110M | flan-t5-base **encoder** + custom head, 12L × 768, cross-encoder | Apache-2.0 | 0-1 consistency score for a premise-hypothesis pair | English only (11 languages exist only in the closed HHEM-2.3) | 67.1 avg BAcc / 62.7 F1-macro in Tamber et al., **Vectara-authored** | in-band and cheap, but the weakest credible accuracy in the field |
| 10 | `KRLabsOrg/tinylettuce-ettin-{17m,32m,68m}-en-v1` | 17 / 32 / 68M | Ettin, 7L × 256 / 10L × 384 / 19L × 512, token tagger | MIT | span tagging | English only | **vendor blog only**; RAGTruth F1 68.52 / 72.15 / 74.97 | reject the checkpoints, keep the recipe - see the honesty note below |
| 11 | `tals/albert-xlarge-vitaminc-mnli` | 58.7M | ALBERT-xlarge, **24L × 2048 shared**, cross-encoder | Apache-2.0 | fact verification on contrastive evidence | English only | VitaminC / NAACL 2021, +10% adversarial fact verification | tiny on paper, slow in practice - see latency section |
| 12 | `yzha/AlignScore-base` | ~125M | roberta-base, 12L × 768, 3 custom heads | MIT | alignment score | English only | 60.5 avg BAcc / 52.1 F1-macro in Tamber et al. | superseded by its own descendant; Lightning `.ckpt` needs re-plumbing |
| 13 | `manueldeprada/FactCC` | ~110M | bert-base, 12L × 768 | MIT | binary factual consistency | English only | pre-2021, below AlignScore everywhere | historical baseline only |
| 14 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` | ~107M | MiniLMv2, 6L × 384, cross-encoder | MIT | NLI | yes, XNLI languages | XNLI, materially below mDeBERTa | fast floor, not a quality candidate |

### In-band replacements for the two 568M retrieval stages

The coordinator's framing - can anything under 350M replace two 568M models - has a direct answer at the retrieval layer, independent of the verifier question.

- `Alibaba-NLP/gte-multilingual-reranker-base` - **306M**, cross-encoder, 8,192 ctx, 70+ languages, Apache-2.0, needs `trust_remote_code=True`; a like-for-like swap for bge-reranker-v2-m3 at 54% of the parameters
- `Alibaba-NLP/gte-multilingual-base` - **305M**, bi-encoder, 8,192 ctx, 70+ languages, Apache-2.0, `trust_remote_code=True`; like-for-like swap for the bge-m3 gate
- `BAAI/bge-reranker-base` - **278M** XLM-RoBERTa-base cross-encoder, but only 514 ctx, and BAAI themselves direct multilingual users to the v2-m3 model, so expect a quality drop
- `intfloat/multilingual-e5-base` **278M** / `-small` **118M** - conservative bi-encoder fallbacks, 512 ctx

The more interesting option is not swapping the retrieval stages but **deleting** them. A single 8,192-context token tagger consumes claim plus several chunks in one forward, which removes the need for a bi-encoder gate and a separate reranker entirely.

## Latency ranking - separate from quality, and it disagrees

Sequential depth is the driver, not parameter count. Layers cannot be parallelised away; width can be absorbed by wider matmuls. The proxy below is `layers × hidden²`, **computed by me from published config dims, not measured**, normalised so ModernBERT-base = 1.00. DeBERTa's disentangled attention roughly doubles per-layer attention cost, so DeBERTa rows carry a ×2 adjustment.

| model | layers × hidden | relative compute | ctx | fits claim + 400-token chunk | batches top-3 as one forward | note |
|---|---|---|---|---|---|---|
| `tinylettuce-ettin-17m` | 7 × 256 | **0.04** | 8,192 | yes | yes | 25x cheaper than ModernBERT-base |
| `tinylettuce-ettin-32m` | 10 × 384 | **0.11** | 8,192 | yes | yes | |
| `multilingual-MiniLMv2-L6` | 6 × 384 | **0.07** | 512 | tight | yes | |
| `mmBERT-small` derivatives | 22 × 384 | **0.25** | 8,192 | yes | yes | **deep-and-narrow - flag**: 22 sequential layers means real wall-clock will land nearer 0.40-0.50, not 0.25 |
| `deberta-v3-small` (`deberta-small-long-nli`) | 6 × 768 ×2 | **0.55** | 1,680 | yes | yes | only 6 sequential steps - the depth winner among credible models |
| `tinylettuce-ettin-68m` | 19 × 512 | **0.38** | 8,192 | yes | yes | also deep-and-narrow at 19 layers |
| HHEM-2.1-Open (T5-base encoder) | 12 × 768 | **0.55** | long (chunked internally) | yes | yes | |
| `AlignScore-base` (roberta-base) | 12 × 768 | **0.55** | 512 | yes | yes | |
| `gte-multilingual-reranker-base` | 12 × 768 | **0.55** | 8,192 | yes | yes | |
| `deberta-v3-base` NLI heads | 12 × 768 ×2 | **1.09** | 512 or 1,680 | yes | yes | |
| **incumbent** `mdeberta-v3-base` NLI | 12 × 768 ×2 | **1.09** | 512 | yes, but 512 truncates on non-Latin scripts | yes | baseline for the NLI stage only |
| `ModernBERT-base` / `mmBERT-base` derivatives | 22 × 768 | **1.00** | 8,192 | yes | yes | flash-attention + unpadding + alternating global/local attention claw back much of the 22-layer depth |
| `tals/albert-xlarge-vitaminc-mnli` | **24 × 2048** shared | **7.8** | 512 | yes | yes | **the trap**: 58.7M params but ~8x ModernBERT-base compute. Parameter sharing cuts memory, not FLOPs. Reported in the literature that ALBERT-xxlarge is ~70% of BERT-large's parameters and "about 3 times slower" |

Three practical points on how this would actually be served.

- **All in-band candidates are single-forward architectures** - cross-encoders and token taggers alike take claim and evidence concatenated in one sequence, so top-3 chunks batch as a `[3, seq]` tensor in one forward. No candidate here has a bi-encoder's two-tower constraint or an encoder-decoder's generation loop, which is a real advantage over the out-of-band MiniCheck-Flan-T5-Large
- **The 662 ms is mostly not the NLI stage** - it is a 568M bi-encoder over every chunk plus a 568M cross-encoder over the top-8. Replacing all three with one 307M mmBERT-base forward over 3 batched chunks is a large latency cut even though mmBERT-base is not a small model, because the chunk-count multiplier disappears
- **Published latency figures are thin and all vendor-reported** - Vectara claims HHEM-2.1-Open runs "around 1.5 second for a 2k-token input on a modern x86 CPU" at under 600 MB RSS; LettuceDetect claims 30-60 examples/second on a single unnamed GPU; TinyLettuce claims "real-time on CPU with low latency" with **no number and no hardware named**. None of these has independent confirmation

### Where the two rankings disagree

They disagree, and blending them would hide the real decision.

- **Quality ranking picks `lettucedect-v2-mmbert-base` (307M)** - the most expensive in-band model, at ~1.00 relative compute
- **Latency ranking picks `tinylettuce-ettin-17m` (17M)** at 0.04 relative compute, which is English-only, trained largely on synthetic data, and scores 68.52 F1 on RAGTruth against its own 90.87 on its own synthetic test set
- **Best quality-per-millisecond is `mmBERT-small` (140M)** - roughly a quarter of mmBERT-base's compute, genuinely multilingual, and the alexandrainst release is direct evidence the recipe converges (Danish F1 0.9143 supported / 0.8689 unsupported; German 0.9147 / 0.8627 - **paper-reported, not independently checked**, and measured on synthetic hallucinations they generated themselves). The caveat is architectural: 22 layers at hidden 384 is deep-and-narrow, so it will not deliver the full 4x speedup its FLOP count implies

The honest reading is that quality-per-millisecond favours mmBERT-small, raw quality favours mmBERT-base, and both sit on the same backbone family - so the two can share one training pipeline and be chosen on measured, not predicted, numbers.

## Evidence quality - read before trusting any number here

Independent reproduction in this field is thin, and the fullest cross-model table is written by a vendor whose own model is in it.

- **LLM-AggreFact** (11 human-annotated datasets, average balanced accuracy) is hosted by the MiniCheck authors. Rows where a competitor beats the host carry weight; the host's own rows are vendor. Nothing in-band appears on it at all
- **Tamber et al. 2025** (arXiv 2505.04847) gives the fullest table - HHEM-1.0/2.1, AlignScore base/large, MiniCheck-RoBERTa-L, Bespoke, TrueTeacher - but Ofer Mendelevitch and Forrest Bao are **Vectara**, so it is vendor-authored. It is credible mainly because it is self-unflattering: it places Vectara's own HHEM-2.1-Open at 67.1 avg BAcc, below a 355M RoBERTa at 68.8
- **Godbole & Jia, USC, "Verify with Caution"** (arXiv 2501.14883) is the one genuinely vendor-free audit. Over 11 datasets it found the top metrics mutually inconsistent - for Bespoke-7B (77.4 BAcc) and gpt-4-turbo (76.2 BAcc) the intersection-over-union of their "unattributable" predictions was **below 50% on 5 of 14 datasets**, and gpt-4-turbo misordered 26% of system pairs
- **Seo et al., COLM 2025, "Verifying the Verifiers"** (arXiv 2506.13342) - independent, found "approximately 16% of ambiguous or incorrectly labeled data substantially influences model rankings" across 14 fact-checking benchmarks
- **Vendor-blog-only, flagged**: TinyLettuce (HF blog `adaamko/tinylettuce`), multilingual EuroBERT LettuceDetect (HF blog `adaamko/lettucedetect-multilingual`), Patronus Lynx (patronus.ai announcement)
- **Public dispute on record**: GitHub issue vectara/hallucination-leaderboard#128, "HHEM2.1-Open is poor evaluator, has self-reported F1 only between 45-66%" - 44.83% F1 on RAGTruth-Summ, 60.00% on RAGTruth-QA
- **No independent reproduction exists for any span-level detector.** Every span-F1 and IoU number in this report is vendor-reported

Two biases documented by Godbole & Jia apply to every candidate here and to the incumbent cascade. Metrics "mislabel unattributable examples (have low TNR) on the high ROUGE examples" - heavy paraphrase defeats them - and chunking inputs to 500 tokens raised false "unattributable" predictions by 6%, because "evaluators that chunk their inputs are inherently disadvantaged when verifying attributable claims that reference distant parts of the input document". The 1,500-character chunking sits squarely in that failure mode, which is an argument for the 8,192-context candidates independent of their accuracy.

## Detail on the three that matter

### LettuceDetect v2, mmBERT encoder - the quality pick

`KRLabsOrg/lettucedect-v2-mmbert-base`, arXiv 2607.00895.

- **Params** 307M total, 110M non-embedding (the 256K Gemma2 vocab carries 197M of it, so the compute-relevant body is small for its headline size)
- **Architecture** 22 layers × 768 hidden, ModernBERT family - flash attention, alternating global / local attention, unpadding and sequence packing. Token-classification head, so one joint forward over claim + evidence
- **Task** per-token supported (0) / unsupported (1) over the answer, yielding ungrounded spans; a taxonomy head (`lettucedect-v2-taxonomy-head`) can type them afterwards
- **Training** 145,250 examples - 74,285 new code / tool-output / structured-document cases plus converted RAGTruth and PsiloQA (14 languages)
- **Numbers, all vendor-reported** unified test span-F1 0.642, example-F1 0.869, IoU 0.671; RAGTruth example-F1 81.8; PsiloQA English IoU 0.724, 14-language IoU 0.689
- **Languages** Arabic, Basque, Catalan, Czech, English, Farsi, Finnish, French, German, Hindi, Italian, Spanish, Swedish, Chinese, plus graceful degradation from a 1,833-language backbone
- **Context** 8,192, so claim + several 400-token chunks fit uncut in any script
- **Export note (informational)** ModernBERT-family INT8 is not free - full INT8 quantisation is reported to collapse predictions because attention-layer masked-fill interacts pathologically with INT8 calibration; the working recipe is FP32 ONNX then dynamic INT8 with all attention / QKV nodes excluded, for ~40% size cut and only ~8% CPU latency gain. Irrelevant on GPU

### mmBERT-small hallucination classifiers - the quality-per-millisecond pick

`alexandrainst/mmBERT-small-multi-wiki-qa-synthetic-hallucinations-{bg,bs,ca,cs,da,de,el,en,es,et,fi,fo,fr,hr,hu,is,it,lt,lv,nl,no,pl,ro,sk,sl,sr,sv,uk,...}`, from MultiWikiQHalluA (arXiv 2605.02504).

- **Params** 140M total, 42M non-embedding; 22 layers × 384 hidden, 8,192 context, CC-BY-4.0
- **Task** token-level supported / unsupported, same formulation as LettuceDetect
- **Training** synthetic hallucinations generated over MultiWikiQA using the LettuceDetect framework, 4,000 train / 1,000 test per language; the authors compared Ettin-17m, EuroBERT-210m and mmBERT-small and selected mmBERT-small on F1
- **Numbers, paper-reported only** Danish F1 0.9143 supported / 0.8689 unsupported, German 0.9147 / 0.8627. Evaluated on English, Danish, German and Icelandic only - the other 26 checkpoints are unevaluated
- **Caveat that matters** these are synthetic hallucinations scored against synthetic test sets, the same methodological weakness that makes TinyLettuce's 90.87% meaningless out of domain. Treat the F1 figures as evidence the recipe converges, not as evidence of accuracy on our gold
- **Operational cost** 30 separate per-language checkpoints. The useful move is not to ship 30 models but to fine-tune one mmBERT-small on pooled multilingual data

### mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 - the cheap swap

`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, 278M, MIT.

- Same backbone, same tokenizer, same 512 limit, same 12 × 768 shape as the incumbent `mDeBERTa-v3-base-mnli-xnli` - a drop-in with no pipeline change at all
- Trained on ~3.3M hypothesis-premise pairs across 27 languages (XNLI validation + multilingual-NLI-26lang-2mil7) against the incumbent's MNLI + XNLI only, so it is a strict training superset
- XNLI test accuracy spans 0.871 English → 0.744 Urdu across 15 evaluated languages, vendor-run on a public benchmark
- Zero latency change and near-zero integration risk make this the correct control arm, not the headline candidate

## Gaps

- No open verifier is trained on anything approaching 21 languages with **human** labels. PsiloQA's 14 languages are GPT-4o-annotated; MultiWikiQHalluA's 30 are synthetic
- No in-band model appears on LLM-AggreFact, so there is no cross-comparable accuracy number for any candidate in this report against the specialised English fact-checkers
- LLM-AggreFact, HaluBench and RAGTruth are all English. A multilingual claim can only be validated on our own gold
- Seo et al.'s ~16% label-noise finding means the 2-3 point gaps that separate models on these leaderboards are inside the noise floor; rank ordering among close candidates is not settled
- No measured latency figure exists for any in-band candidate on named hardware. The relative-compute column here is computed from config dims and must be replaced with measurements

## Suggested next step

Run three arms on the existing gold under the Round 3 honest harness (GroupKFold leave-one-source-out OOF thresholds, English and non-English scored separately), timing each on the same hardware: `lettucedect-v2-mmbert-base` at 307M as the quality arm, a `mmBERT-small` fine-tune on pooled PsiloQA + RAGTruth-multi + our gold as the quality-per-millisecond arm, and `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` as the zero-risk control. Run the two mmBERT arms as single-forward replacements for the entire cascade with the top-3 chunks batched, not as drop-ins for the NLI stage alone - that configuration is where the latency win lives, and it is the only way to learn whether one sub-350M forward can carry both retrieval and verification.

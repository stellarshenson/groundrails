# Training Recipes for Small Grounding Verifiers

Survey of how the working small faithfulness classifiers were actually trained, and what that implies for distilling our frozen cascade (macro-F1 0.824, 2,752 gold claims, 21 languages, ~111,800 cached scored pairs) into a single sub-500M encoder. Every recipe that beats an LLM judge at <1B parameters does the same three things - start from an NLI-transferred encoder, add synthetically constructed document-level negatives, mix in hard-mined ANLI.

## Recipe comparison

| Recipe | Base model | Data construction | Volume | Objective | Reported gain | Code released? |
|---|---|---|---|---|---|---|
| **MiniCheck** (EMNLP 2024) | RoBERTa-L 355M / DeBERTa-v3-L 435M / Flan-T5-L 770M | C2D: claim → GPT-3.5 atomic facts → GPT-4 writes sentence *pairs* entailing each fact only jointly → GPT-4 weaves passage; negatives by sentence deletion + GPT-4 entailment recheck. D2C: doc → 3 chunks → GPT-4 chunk summaries → decompose → power-set subclaims → delete-and-relabel + cross-chunk transfer | 14,395 synthetic (7,076 C2D / 7,319 D2C) + 21K hard-mined ANLI = 35K | Binary cross-entropy, neutral+contradiction → unsupported | Flan-T5-L 74.7 BAcc vs GPT-4 75.3, AlignScore 70.4, same backbone on ANLI-only 61.4 (+13.3) | Yes - [MiniCheck](https://github.com/Liyan06/MiniCheck), `synthetic_data_gen/` + HF `lytang/C2D-and-D2C-MiniCheck` |
| **FactCG / CG2C** (NAACL 2025) | RoBERTa-L, DeBERTa-v3-L (0.4B), Flan-T5-L | GPT-4o extracts entity-relation-entity triples → context graph → sample acyclic sub-graphs by hop count/shape → positive claim covers sub-graph; negative = delete one relation's supporting sentence. LLM generates, never labels - labels correct by construction | 8,213 MHQA + 6,433 Doc, mixed with MiniCheck C2D + ANLI, two-stage | Cross-entropy | DeBERTa-L 77.2 avg BAcc > MiniCheck-FT5 75.5 > GPT-4o 75.9; ~18 min training on 4× RTX 8000 | Yes - [FactCG](https://github.com/derenlei/FactCG) |
| **TrueTeacher** (EMNLP 2023) | T5-11B student; mT5-XXL multilingual arm | Train T5-small…11B summarizers → generate summaries over CNN/DM → label every summary zero-shot with FLAN-PaLM 540B teacher | 1.4M (907,899 consistent / 475,563 inconsistent); controlled runs use 100K balanced | Binary classification mixed with ANLI | TRUE ROC-AUC 82.7 → 87.8 (+5.1); ANLI-only 82.0; **student beat its 50× larger teacher**; mFACE 71.6 → 75.3 | Yes - [true_teacher](https://github.com/google-research/google-research/tree/master/true_teacher), full 1.4M set + checkpoint |
| **AlignScore** (ACL 2023) | RoBERTa-base 125M / large 355M | 15 datasets / 7 tasks unified to (context, claim, label); QA → declarative via seq2seq, wrong options → not-aligned; paraphrase + summarization negatives via back-translation and 25% MLM infilling | 4.7M pairs, 500K cap per dataset | Multi-task: 3-way NLI + binary + regression heads, λ=1 each | 355M matches/beats ChatGPT and GPT-4 metrics; SummaC 88.6 AUC, TRUE 87.4; 100 GPU-h base, 532 GPU-h large | Yes - [AlignScore](https://github.com/yuh-zha/AlignScore) |
| **LIM-RA** (NAACL 2024 industry) | DeBERTa | AlignScore corpus denoised, capped at 20K per dataset, plus synthetic robustness samples | 452K (~10% of AlignScore) | Same alignment objective | Beats AlignScore and ChatGPT on 24 of 33 test sets - less data, cleaned, wins | Not stated |
| **PsiloQA** (EMNLP 2025 findings) | ModernBERT-base 136M, mmBERT-base 307M | GPT-4o generates QA from Wikipedia → 24 LLMs answer *without* context → GPT-4o marks hallucinated spans against gold + retrieved context → rule/prompt filters | 63,792 train / 3,355 val / 2,897 test, 14 languages | Token classification (span-level) | mmBERT 84.88 AP / 70.67 IoU EN; multilingual training > per-language > English-only; teacher labels validated at 84.3 AP vs human; USD 535 vs ~USD 3,000 human (~17×) | Yes - [psiloqa](https://github.com/s-nlp/psiloqa), CC-BY-4.0 |
| **VitaminC** (NAACL 2021) | ALBERT-base / xlarge | >100K real Wikipedia revisions mined into near-identical contrastive evidence pairs (one supports, one does not), plus synthetic revisions | 450K+ claim-evidence pairs | 3-way fact verification / NLI | +10% adversarial fact verification, +6% adversarial NLI | Yes - [VitaminC](https://github.com/TalSchuster/VitaminC), HF `tals/albert-base-vitaminc` |
| **TRUE / NLI transfer** (NAACL 2022) | T5-11B | No construction - straight ANLI fine-tune, 25K steps, lr 1e-4, bs 32 | ANLI as-is | 3-way NLI | Established that large-scale NLI transfer is the strongest single signal; now the *floor* every later recipe beats (82.0 ROC-AUC; 61.4 BAcc at Flan-T5-L scale) | Yes - benchmark released |
| **RAGTruth** (ACL 2024) | Llama-2-13B detector | ~18K real RAG responses from 6 LLMs over QA / data-to-text / summarization, **human** word-level annotation with intensity labels | ~18K responses | Response-level + span-level classification | Response F1 78.7 vs GPT-4-turbo prompt 63.4; span F1 52.7 vs 28.3; annotation ~USD 3,000 for the QA subset alone | Yes - dataset public |
| **LettuceDetect** (2025 → 2026) | ModernBERT-base 150M / large 396M; 2026 line adds mmBERT-base 307M and Qwen-2B | 2025: RAGTruth only, context/question tokens masked from loss. 2026: synthetic span-injection - edit correct grounded answers with a large LLM, recover exact char offsets from the applied edit rather than diffing | 18K → 74,285 new + 145,250 total | Token classification (encoder) / JSON span generation (2B) | 79.22 example-F1 > fine-tuned Llama-2-13B 78.7 > GPT-4-turbo 63.4, ~30× smaller, 30-60 ex/s on one GPU | Yes - [LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect), MIT |
| **Lynx** (2024) | Llama-3-8B / 70B-Instruct | GPT-4o writes minimally-different perturbed answers - D' = {(q, c, x̃, 1−y)} - over CovidQA, PubMedQA, DROP, FinanceBench, RAGTruth, HaluEval | 2,400 train (600 × 4 domains) + 800 val | SFT with CoT, emits `{"REASONING": …, "SCORE": "PASS"/"FAIL"}` | 70B: 87.4% HaluBench vs 80.1% base Llama-3-70B (+7.3), GPT-4o 86.5; 3 epochs, lr 5e-7, bs 256, 32× H100 | Yes - [Lynx](https://github.com/patronus-ai/Lynx-hallucination-detection) + HaluBench 15K |
| **Paladin-mini** (2025) | Phi-4-mini-instruct 3.8B | MiniCheck + AggreFact public data plus *proprietary* synthetic sets targeting numeric, temporal and logical errors | 23K | Full SFT, stays generative | LLM-AggreFact 73.08 (below Bespoke-MiniCheck 77.7) but 96.0 vs 46.0 BAcc on their own price/math subset; 70ms vs 7s | Model open, data proprietary, benchmark self-authored |
| **HHEM-2.1** (vendor) | flan-t5-base 0.1B cross-encoder | **Undisclosed** | **Undisclosed** | **Undisclosed** ("contrastive, entailment-based" per blog) | RAGTruth-QA 74.28 BAcc vs GPT-3.5 56.16, GPT-4 74.11; 71.8 on LLM-AggreFact - below MiniCheck-FT5 | Weights only (Apache 2.0), no training code, no paper |
| **Luna** (COLING 2025 industry) | DeBERTa-large 440M | "Meticulously curated" proprietary production RAG data - volume and annotation source undisclosed | Undisclosed | Token-level, long-context chunking | 97% cost / 91% latency cut vs GPT-3.5, +18% accuracy; LettuceDetect reports beating it by 14.8% | No |

## Evidence quality

- **Reproducible and independently rebuilt** - MiniCheck (FactCG retrained on the same mix; Bespoke Labs rescaled the scheme to 7B for 77.4 on LLM-AggreFact), AlignScore (rebuilt by LIM-RA and used as MiniCheck's RoBERTa init), TrueTeacher, VitaminC, PsiloQA, LettuceDetect
- **Vendor blog only, treat as marketing** - HHEM-2.1 discloses nothing beyond "flan-t5-base cross-encoder trained on factual consistency datasets"; its 71.8 on the public leaderboard sits below both open sub-1B recipes, so the "beats GPT-4" framing is benchmark selection
- **Industry paper, undisclosed data** - Luna and Paladin-mini publish a method sketch but not the training corpus; Paladin-mini's headline win is on a benchmark it authored, and it loses by 4.6 BAcc on the public one
- **Current LLM-AggreFact standing** - Bespoke-MiniCheck-7B 77.4, Claude-3.5 Sonnet 77.2, Granite Guardian 3.3 8B 76.5, gpt-4o 75.9; best <1B are FactCG-DeBERTa-L 0.4B at 75.6 and MiniCheck-Flan-T5-L 0.8B at 75.0 - a 0.4B encoder is within 1.8 points of a 7B and beats gpt-4o

## Cross-cutting findings

- **NLI transfer is the floor, not the method** - same Flan-T5-L backbone: 61.4 BAcc on ANLI alone → 74.7 with 14K constructed examples added; the +13.3 comes entirely from document-level synthetic construction
- **Volume is small** - every winning sub-1B recipe used 14K-35K task-specific examples; AlignScore's 4.7M is the outlier and LIM-RA beat it with 10% of it, cleaned
- **Hard-negative mining beats bulk** - MiniCheck's 21K ANLI subset is specifically the examples its own trained entailer got wrong
- **Label provenance matters more than label count** - FactCG's negatives are correct *by construction* (delete the supporting sentence) and never touch an LLM judge; that recipe holds the sub-1B record while being the cheapest to generate on long documents
- **TrueTeacher's ablation is the key number for distillation** - label quality contributed +5.6 and realistic input distribution +5.8, roughly equal and complementary; a teacher labelling *in-distribution* text is worth as much as a better teacher
- **Multilingual training beats per-language and English-only** - PsiloQA across 14 languages, TrueTeacher across 45 (English-only 100K improved 32/45, multilingual 25K × 4 improved 35/45), transferring across scripts and language families
- **Training compute is negligible** - FactCG-DeBERTa: ~18 minutes on 4× RTX 8000; AlignScore-base: 100 GPU-hours; only the generative recipes (Lynx, 32× H100) are expensive

## Answers

### 1. Best-fit recipe for distilling our cascade

**TrueTeacher, with PsiloQA's multilingual arrangement and FactCG's negative construction bolted on.** It is the only published recipe whose shape is identical to ours - an expensive teacher labels large volumes of in-distribution text, a small student trains on those labels mixed with ANLI - and the only one where the student *beat* its teacher. Concretely:

- **Student** - mDeBERTa-v3-base initialised from the XNLI checkpoint already in the cascade, or mmBERT-base 307M if we want ModernBERT-era long context; both quantise to OpenVINO int8 and fit Lambda
- **Labels** - soft grounded-probabilities from the existing 111,800 cached (claim, chunk) pairs; these already satisfy TrueTeacher's "realistic distribution" half at zero cost, which its ablation values at +5.8
- **Mix** - add ~20K hard-mined XNLI/ANLI (MiniCheck's trick - keep the examples the current entailer gets wrong) and a FactCG-style deletion-negative set generated from our own chunks, whose labels are correct by construction and therefore carry information the teacher does not have
- **Gold discipline** - the 2,752 human claims are held out entirely as test; they are far too few to train on (Lynx used 2,400 only to elicit latent ability in a 70B) and too valuable to spend
- **Aggregation** - keep the cascade's max-over-chunks aggregation at inference, as AlignScore does (350-token chunks, mean over claim sentences of max over chunks), otherwise the student inherits the entailment decision without the retrieval view

### 2. Realistic expected outcome

Blunt: **0.78-0.84 macro-F1, i.e. parity give or take, and almost certainly not a clear beat.**

- Distilling a *cascade* is not classic KD. The 0.824 is produced by three stages seeing different views - bi-encoder recall, reranker selection, entailment verdict. A single cross-encoder student sees only (claim, chunk); it can inherit the verdict but only inherits chunk selection if we retain the same aggregation
- Soft labels reproduce the teacher's error surface, they do not fix it. The published cases where a student beat its teacher had *noisy zero-shot LLM* teachers (TrueTeacher's FLAN-PaLM, human-checked at 89% accuracy) whose errors were random and averaged out. Our teacher is already calibrated and hyperplane-tuned, so its errors are systematic and the student will learn them faithfully
- Precedent for parity-or-better at this size is nonetheless strong - LettuceDetect-large 396M beat a fine-tuned Llama-2-13B, FactCG-DeBERTa 0.4B beat gpt-4o
- Upside beyond the teacher requires supervision the teacher does not contain - constructed-label negatives (FactCG) or the gold set used for final threshold calibration only
- **The real payoff is not accuracy** - it is one 300-400M encoder pass replacing three model loads, which is where the few-hundred-ms warm latency budget is actually won

### 3. Cheapest decisive experiment

A learning-curve plus teacher-agreement probe - about half a day on one GPU, zero LLM spend, zero annotation.

1. Train mDeBERTa-v3-base (XNLI-init) on soft teacher probabilities from the cached matrix at three volumes - 10K / 40K / 111K pairs, 1 epoch each → verify: each run completes in 20-40 min (FactCG needed 18 min for DeBERTa-large on 4 GPUs)
2. Score each checkpoint on the held-out 2,752 gold claims (macro-F1) and on a held-out slice of the cached pairs (agreement with the teacher) → verify: two numbers per volume
3. Read the divergence:
   - Teacher-agreement saturating >95% while gold macro-F1 stalls well below 0.824 → the student is faithfully copying errors it cannot see; **distillation is capped, more soft labels will not help**, and budget must go to constructed-label negatives instead
   - Gold macro-F1 rising with volume and the 10K → 111K slope still positive → scale label generation, the ceiling has not been hit
   - Gold macro-F1 already near 0.824 at 10K → the cascade's decision is trivially learnable, ship the student and stop

This also settles the LIM-RA question (does cleaned 10% beat noisy 100%?) on our own data before any training budget is committed.

## Sources

- MiniCheck - [arXiv 2404.10774](https://arxiv.org/abs/2404.10774), [code](https://github.com/Liyan06/MiniCheck), [C2D/D2C data](https://huggingface.co/datasets/lytang/C2D-and-D2C-MiniCheck)
- FactCG - [arXiv 2501.17144](https://arxiv.org/abs/2501.17144), [code](https://github.com/derenlei/FactCG)
- TrueTeacher - [arXiv 2305.11171](https://arxiv.org/abs/2305.11171), [code + 1.4M dataset](https://github.com/google-research/google-research/tree/master/true_teacher)
- AlignScore - [arXiv 2305.16739](https://arxiv.org/abs/2305.16739), [code](https://github.com/yuh-zha/AlignScore)
- LIM-RA - [arXiv 2404.06579](https://arxiv.org/abs/2404.06579)
- PsiloQA - [arXiv 2510.04849](https://arxiv.org/abs/2510.04849), [code](https://github.com/s-nlp/psiloqa)
- VitaminC - [arXiv 2103.08541](https://arxiv.org/abs/2103.08541), [code](https://github.com/TalSchuster/VitaminC)
- TRUE - [arXiv 2204.04991](https://arxiv.org/abs/2204.04991)
- RAGTruth - [arXiv 2401.00396](https://arxiv.org/abs/2401.00396)
- LettuceDetect - [arXiv 2502.17125](https://arxiv.org/abs/2502.17125), [2026 span line arXiv 2607.00895](https://arxiv.org/abs/2607.00895), [code](https://github.com/KRLabsOrg/LettuceDetect)
- Lynx - [arXiv 2407.08488](https://arxiv.org/abs/2407.08488), [code](https://github.com/patronus-ai/Lynx-hallucination-detection)
- Paladin-mini - [arXiv 2506.20384](https://arxiv.org/abs/2506.20384)
- Luna - [arXiv 2406.00975](https://arxiv.org/abs/2406.00975)
- HHEM-2.1 - [model card](https://huggingface.co/vectara/hallucination_evaluation_model), [vendor blog](https://www.vectara.com/blog/hhem-2-1-a-better-hallucination-detection-model)
- LLM-AggreFact leaderboard - [llm-aggrefact.github.io](https://llm-aggrefact.github.io/)

# Grounding and Faithfulness Benchmarks - External Calibration Survey

Survey of the benchmarks the grounded-factuality field actually uses, scoped to our task shape: given a CLAIM and one or more SOURCE DOCUMENTS, emit grounded / not-grounded. Written to answer whether macro-F1 0.824 on a private 2,752-claim gold set is strong, average, or weak by external standards.

Research date 2026-07-28. Every score below is tagged with who reported it and whether any independent party re-ran it.

## Headline

- **Best external calibration → LLM-AggreFact** - it is the only widely-cited benchmark whose unit of evaluation is exactly ours (document + claim → binary supported), with a live public leaderboard and a one-line HuggingFace load
- **Leaderboard top is 77.4 balanced accuracy** (Bespoke-MiniCheck-7B), and the entire published leaderboard spans only 71.8 → 77.4, a 5.6-point band
- **Off-the-shelf NLI on the same benchmark scores 61.4** (FT5-ANLI-L), so an untuned entailment model is ~16 points below the top
- **Nothing English-only calibrates our 21 languages** - the multilingual options (PsiloQA 14, Mu-SHROOM 14, Poly-FEVER 11, X-Fact 25) all change the task shape to span detection or open-web veracity
- Our 0.824 macro-F1 is **not** comparable head-to-head with 77.4 balanced accuracy - see the metric-mapping section

## Comparison Table

| Benchmark | Task shape | Size | Languages | Metric | Current SOTA (reporter) | Independently run? | HF dataset id |
|---|---|---|---|---|---|---|---|
| **LLM-AggreFact** | doc + claim → binary supported | 29,320 test (11 sub-sets) | English | Per-dataset balanced accuracy, unweighted mean | 77.4 Bespoke-MiniCheck-7B (maintainer-posted) | No - academic leaderboard, numbers posted by maintainers, no held-out set | `lytang/LLM-AggreFact` |
| **FACTS Grounding** | prompt + long doc → generated answer, judged | 860 public + private held-out | English | LLM-judge ensemble factuality % | 83.6 Gemini 2.0 Flash (Google/Kaggle) | Vendor-run, but held-out set + org-executed | `google/FACTS-grounding-public` |
| **FACTS Benchmark Suite** | 4 sub-benchmarks incl. Grounding v2 | 3,513 public; Grounding v2 2,104 items | English | FACTS Score, avg over public + private | 68.8 Gemini 3 Pro, Dec 2025 (Google/Kaggle) | Vendor-run, held-out | (suite; public split under same repo) |
| **TRUE** | doc + generated text → binary consistent | 11 datasets, low-thousands each | English | ROC-AUC per dataset, mean | 87.8 avg ROC-AUC, T5-11B ANLI+TrueTeacher, summarization subset (Google, self-reported) | No leaderboard - paper numbers only | none (script in `google-research/true`) |
| **RAGTruth** | RAG response → span + response-level hallucination | 18k responses (15,090 train / 2,700 test) | English | Response-level F1, span F1 | 79.22 F1 LettuceDetect-large; 81.8 example-F1 LettuceDetect-Qwen-2B (authors, self-reported) | No leaderboard | mirrors: `wandb/RAGTruth-processed`, `flowaicom/RAGTruth_test` |
| **HaluBench** (Lynx) | passage + question + answer → binary | 14,900 test | English | Accuracy | Lynx-70B tops it (Patronus AI, self-reported) | No - vendor benchmark for a vendor model | `PatronusAI/HaluBench` |
| **HaluEval** | question/dialogue + answer → hallucinated? | 35k | English | Accuracy | model-dependent, no canonical board | No | `pminervini/HaluEval` and mirrors |
| **HaluEval 2.0** | generator hallucination rate, 5 domains | 8,770 questions | English | Micro/macro hallucination rate | Measures generators, not detectors | No | GitHub `RUCAIBox/HaluEval-2.0` |
| **SummEdits** | doc + edited summary → binary consistent | 6,348 across 10 domains | English | Balanced accuracy | GPT-4 era numbers, paper-reported (Salesforce) | No | `Salesforce/summedits` |
| **FaithBench** | doc + summary → 4-level hallucination label | 660 (66 passages × 10 LLMs) | English | Balanced accuracy, F1-macro | 57.65 BAcc / 53.35 F1-macro best detector (Vectara, self-reported) | No | GitHub `vectara/FaithBench` |
| **Vectara Hallucination Leaderboard / FaithJudge** | generator hallucination rate on summarization | FaithBench + RAGTruth | English | Hallucination rate % | 1.8% top entry, Apr 2026 (Vectara) | Vendor-maintained | GitHub `vectara/hallucination-leaderboard` |
| **ExpertQA** | expert answer claim + cited evidence → supported? | 2,177 questions, 32 fields | English | Balanced accuracy (as LLM-AggreFact slice) | 60.9 Claude-3.5 Sonnet - hardest slice on the board | No | `cmalaviya/expertqa` |
| **QAGS-CNNDM / QAGS-XSum** | summary sentence → consistent? | ~235 / ~239 summaries | English | ROC-AUC (inside TRUE) | 89.4 both, T5-11B (Google, self-reported) | No | via `google-research/true` |
| **FEVER** | claim → retrieve Wikipedia + 3-way verdict | 185,445 claims | English | Label accuracy, FEVER score | shared-task era, retrieval-coupled | Historic shared task, now static | `fever/fever` |
| **FEVEROUS** | claim → tables + text evidence, 3-way | 87k claims | English | FEVEROUS score | shared-task era | Historic shared task | `fever/feverous` |
| **VitaminC** | contrastive Wikipedia revision + claim → 3-way | ~489k pairs (371k/63.1k/55.2k) | English | Accuracy | fine-tune target, not a leaderboard | No | `tals/vitaminc` |
| **ANLI** | premise + hypothesis → 3-way NLI | 162k train / 3.2k dev+test | English | Accuracy per round | R3 is the hard round; general NLI, not grounding | No | `facebook/anli` |
| **PsiloQA** | question + retrieved context + answer → hallucinated spans | 63,792 train / 2,897 test | **14** | Span IoU, span F1 | 0.724 IoU English, LettuceDetect-Qwen-2B (authors) | No | `s-nlp/PsiloQA` |
| **Mu-SHROOM** (SemEval-2025 T3) | LLM output → char-level hallucination probability | ~2,500 val / ~1,900 test | **14** | Char IoU, correlation with annotator probability | 43 teams ranked, per-language | **Yes** - blind shared-task evaluation phase | `Helsinki-NLP/mu-shroom` |
| **Poly-FEVER** | claim → verdict, translated FEVER family | 77,973 claims | **11** | Accuracy | LLM-family analysis, no board | No | GitHub / arXiv 2503.16541 |
| **X-Fact** | real-world claim + web evidence → 7-way veracity | 31,189 claims | **25** | F1 | ~40 F1 best (authors, self-reported) | No | GitHub `utahnlp/x-fact` |
| **MEMERAG** | multilingual RAG answer → claim-level faithfulness | MIRACL questions, answers in 5 langs | **18** questions / 5 answers | Correlation with human judgement | meta-evaluation of metrics | No | GitHub `amazon-science/MEMERAG` |
| **MUCH** | claim-level uncertainty quantification, needs logits | 4,873 samples | **4** (en, fr, es, de) | UQ ranking metrics | LREC 2026, baselines only | No | `orailix/MUCH` |
| **Unified span benchmark** (2026) | span-level hallucination over code, tool output, docs | 74,285 new + RAGTruth + PsiloQA | **14** via PsiloQA | Span F1, example F1, IoU | 0.689 span-F1 LettuceDetect-Qwen-2B (authors) | No | `KRLabsOrg/*` |

## LLM-AggreFact - the primary candidate

The MiniCheck benchmark (Tang, Laban, Durrett; EMNLP 2024). It unifies 11 human-annotated grounded-factuality datasets into one binary schema, which is the exact shape of our verdict layer.

- **Task** - one document, one claim, label 1 if the claim is supported by the document, 0 otherwise; sentence-level factual errors labelled by human annotators in each constituent set
- **Constituents** - AggreFact-CNN, AggreFact-XSum, TofuEval-MediaSum, TofuEval-MeetingBank, WiCE, Reveal, ClaimVerify, FactCheck-GPT, ExpertQA, LFQA, RAGTruth
- **Fields** - `dataset`, `doc`, `claim`, `label`, `contamination_identifier`; splits dev 30,420 / test 29,320
- **Metric** - balanced accuracy = (TPR + TNR) / 2 computed **per constituent dataset**, then an **unweighted mean over the 11**; a small hard dataset counts as much as a large easy one
- **Licence** - CC-BY-ND-4.0; free download, but ND blocks redistributing a modified copy
- **Languages** - English only; documents from Wikipedia, interviews, web text; domains news, dialogue, science, healthcare

### Leaderboard, top to bottom

Balanced accuracy, as published at llm-aggrefact.github.io.

| Model | Size | Avg | Hardest slice (ExpertQA) | Easiest slice (Reveal) |
|---|---|---|---|---|
| Bespoke-MiniCheck-7B | 7B | 77.4 | 59.2 | 88.0 |
| Claude-3.5 Sonnet | - | 77.2 | 60.9 | 89.1 |
| Granite Guardian 3.3 | 8B | 76.5 | 59.6 | 89.6 |
| Mistral-Large 2 | 123B | 76.5 | 60.8 | 87.7 |
| gpt-4o-2024-05-13 | - | 75.9 | 59.6 | 86.5 |
| FactCG-DeBERTa-L | 0.4B | 75.6 | 59.1 | 88.4 |
| Qwen2.5-72B-Instruct | 72B | 75.6 | 60.1 | 88.9 |
| MiniCheck-Flan-T5-L | 0.8B | 75.0 | 59.0 | 86.2 |
| Llama-3.3-70B-Instruct | 70B | 74.5 | 58.3 | 85.5 |
| Llama-3.1-405B-Instruct | 405B | 74.4 | 58.5 | 86.4 |
| QwQ-32B-Preview | 32B | 71.8 | 60.0 | 86.2 |

### Non-LLM baselines - the reference class we belong to

From the MiniCheck paper's own baseline table (10-dataset version of the benchmark, so ~1 point offset from current leaderboard numbers; all self-reported by the MiniCheck authors).

- **T5-NLI-Mixed 61.0** and **FT5-ANLI-L 61.4** - plain off-the-shelf NLI models, no grounding-specific training
- **SummaC-CV 62.1**, **DAE 64.9**, **QAFactEval 66.5**, **SummaC-ZS 67.9** - classic sentence-level NLI aggregation pipelines
- **AlignScore 70.4** - RoBERTa-large trained explicitly for alignment
- **MiniCheck-DeBERTa 72.6**, **MiniCheck-RoBERTa 72.7**, **MiniCheck-FT5 74.7** - encoder/seq2seq trained on synthetic grounding supervision
- **GPT-4 75.3**, **FactCG-DeBERTa-L (0.4B) 75.6** - the ceiling for sub-1B encoders is currently ~75.6

The 61.4 → 75.6 span is the corridor a torch-free encoder cascade lives in. This is the single most useful calibration number set in this document.

### Effort to run our cascade against it

Low - roughly a day of engineering.

- `load_dataset("lytang/LLM-AggreFact")["test"]` → 29,320 rows of `doc` + `claim` + `label`
- Map cascade output → binary; per-dataset `balanced_accuracy_score`, then `.mean()` over the 11 - the official demo notebook does exactly this
- **The real work is document chunking** - several constituent sets have documents far beyond a 512-token cross-encoder/NLI window; the leaderboard's own top systems handle long docs natively, and chunking is a documented failure source (see caveats)
- CPU int8 cost - 29,320 pairs × cascade depth; a bi-encoder gate that rejects most candidates keeps this tractable, but budget for the reranker and NLI stages on the survivors
- **Report both**: the leaderboard-protocol number, and a run with the gate disabled, so gate-induced recall loss is visible separately

## FACTS Grounding - measures the wrong thing for us

Google DeepMind + Kaggle. It evaluates whether a **generator** LLM answers only from a supplied long document, judged by an ensemble of frontier LLM judges.

- **Not our task** - the leaderboard ranks answer-generating models, not verifiers; our cascade would be a candidate *judge*, not a ranked entry
- **Sizes** - 860 public examples plus a private held-out split; the Dec 2025 FACTS Benchmark Suite has 3,513 public examples over Grounding v2 (2,104 items), Multimodal (1,522), Parametric (2,104), Search (1,884)
- **Scores** - Gemini 2.0 Flash 83.6% on the original Grounding leaderboard; Gemini 3 Pro 68.8% FACTS Score on the suite, all 15 evaluated models under 70%
- **Governance** - Kaggle holds the private sets and runs the evals; that makes it more rigorous than a posted-numbers board, but the benchmark author and the top-scoring model share an owner
- **Licence** - CC-BY-4.0, public split freely downloadable, `google/FACTS-grounding-public`
- **Useful to us only** as a source of grounded long-document material to relabel, not as a scoreboard

## TRUE - the AUC reference, and the right home for our reranker number

Honovich et al., NAACL 2022. 11 datasets normalized to binary factual consistency, scored by ROC-AUC so no threshold is required.

- **Composition** - summarization (FRANK, SummEval, MNBM, QAGS-CNNDM, QAGS-XSum), dialogue (BEGIN, Q², DialFact), fact verification (FEVER, VitaminC), paraphrase (PAWS)
- **Metric fit** - ROC-AUC is the natural comparator for our best single signal, the reranker at AUC 0.841; balanced-accuracy boards cannot host that number
- **Top reported** - T5-11B trained on ANLI + TrueTeacher reaches **87.8 mean ROC-AUC on the 5-dataset summarization subset**, up from 82.7 without TrueTeacher data; self-reported by the Google authors, never independently re-run
- **Per-dataset** - MNBM 78.1, QAGS-XSum 89.4, FRANK 93.6, SummEval 88.5, QAGS-CNNDM 89.4
- **Access** - no single HF id; `google-research/true` ships a script that downloads and normalizes all 11, licences vary per constituent
- **Effort** - medium; the download script is the easy part, emitting a calibrated continuous score from a cascade (rather than a hard verdict) is the design question
- **Caveat** - a 11B-parameter model is not our reference class; there is no published small-encoder TRUE number as clean as the LLM-AggreFact baseline table

## RAGTruth and the RAG-native family

- **RAGTruth** - 18k LLM responses over QA, data-to-text, summarization; 15,090 train / 2,700 test; span-level and response-level annotations; MIT licence; official data at `ParticleMedia/RAGTruth` on GitHub, HF mirrors only
- **Reported** - fine-tuned Llama-2-13B 78.7 response-level F1 (RAGTruth authors); LettuceDetect-large 79.22 F1 and LettuceDetect-Qwen-2B 81.8 example-F1 (LettuceDetect authors); all self-reported, no leaderboard
- **Overlap** - RAGTruth is already one of the 11 LLM-AggreFact slices (top there 86.1, Claude-3.5 Sonnet), so running LLM-AggreFact gets us a RAGTruth read for free
- **Task mismatch** - native RAGTruth is response-level and span-level; our claim-level verdicts need aggregation to a response verdict, which introduces a decision rule the benchmark does not specify
- **HaluBench** - `PatronusAI/HaluBench`, 14,900 examples as passage + question + answer + binary label, drawn from FinanceBench, PubMedQA, CovidQA, HaluEval, DROP, RAGTruth; closest to our shape after LLM-AggreFact, but **CC-BY-NC-2.0** blocks commercial use and it exists to showcase one vendor's model
- **RAGBench** - `galileo-ai/ragbench`, 100k examples over 12 open-book QA sets and 5 domains, TRACe metrics (utilization, relevance, adherence, completeness); adherence is the grounding axis; a fine-tuned 400M DeBERTa reportedly beats LLM judges on it, self-reported by the vendor authors

## Adversarial and summary-level

- **SummEdits** - `Salesforce/summedits`, 6,348 samples over 10 domains (News, Podcast, BillSum, SAMSum, Shakespeare, SciTLDR, QMSum, ECTSum, Sales Email, Sales Call), binary consistent/inconsistent, CC-BY-4.0, inter-annotator agreement ~0.91; **summary-level not claim-level**, so it tests a coarser decision than ours
- **FaithBench** - 660 samples (66 passages × 10 LLMs from 8 families), 4-level schema Consistent / Benign / Questionable / Unwanted; deliberately built from cases where existing detectors *disagree*, so it is a near-worst-case probe; **best detector reaches only 57.65 balanced accuracy and 53.35 F1-macro** (Vectara, self-reported); useful as a humility check, useless as a ranking signal at n=660
- **Vectara Hallucination Leaderboard / FaithJudge** - vendor-maintained, ranks generator models by hallucination rate (top entry 1.8%, Apr 2026); HHEM-2.1 as detector scores ~67% balanced accuracy claim-wise on FaithBench, which shows how far a purpose-built commercial detector falls on adversarial data

## Multilingual - the gap for our 21 languages

No benchmark combines our task shape (claim + documents → binary) with broad language coverage. Everything multilingual changes either the unit or the grounding source.

- **PsiloQA** (`s-nlp/PsiloQA`) - **14 languages**, 63,792 train / 2,897 test, question + retrieved Wikipedia context + answer with hallucinated spans marked; built by GPT-4o auto-annotation, so labels are model-derived not human-verified; closest multilingual analogue but **span-level output required**
- **Mu-SHROOM** (`Helsinki-NLP/mu-shroom`) - **14 languages** (ar, eu, ca, zh, cs, en, fa, fi, fr, de, hi, it, es, sv), ~2,500 val / ~1,900 test, character-level hallucination probability, scored by IoU and correlation with annotator probability; **the only genuinely independent evaluation in this survey** - a SemEval-2025 shared task with a blind test phase and 43 participating teams
- **Poly-FEVER** - **11 languages**, 77,973 claims; machine-translated FEVER + Climate-FEVER + SciFact, validated at ~90 GEMBA; translation artefacts make it a weak gold standard
- **X-Fact** - **25 languages**, 31,189 naturally-occurring claims, 7-way veracity, best F1 ~40; real-world political fact-checking with web retrieval, not document-grounded entailment - a different problem
- **MEMERAG** - **18 languages** of MIRACL questions with answers generated in 5, claim-level faithfulness annotated by native experts; small but human-verified and multilingual, and its meta-evaluation framing (do automatic metrics agree with humans?) is the closest published analogue to what our private gold set does
- **MUCH** (`orailix/MUCH`) - 4,873 samples, 4 languages, Apache-2.0; it is a **white-box uncertainty-quantification** benchmark requiring per-token generation logits, which a black-box verifier cascade cannot supply - not applicable
- **Practical read** - report LLM-AggreFact as the headline calibration and Mu-SHROOM or PsiloQA (adapted) as a separate multilingual claim; do not average them

## Foundational NLI and fact-verification sets

These are training substrate, not calibration. Reporting a score on them says little about grounding performance.

- **FEVER** (`fever/fever`) - 185,445 claims, SUPPORTS / REFUTES / NOT ENOUGH INFO, retrieval-coupled; the NEI class has no analogue in our binary schema
- **FEVEROUS** (`fever/feverous`) - 87k claims over tables plus text
- **VitaminC** (`tals/vitaminc`) - ~489k contrastive claim-evidence pairs from 100k+ Wikipedia revisions where the edit flips the verdict; the best available *training* set for making a verifier sensitive to small evidence changes rather than lexical overlap
- **ANLI** (`facebook/anli`) - 162k train, 3 adversarial rounds, CC-BY-NC-4.0; generic NLI, and the NC licence matters
- **QAGS-CNNDM / QAGS-XSum** - ~235 / ~239 annotated summaries, only meaningful inside TRUE
- **AttributedQA / AIS** - attribution of an answer to a cited passage; the framing our citation layer matches, but no maintained leaderboard

## Metric mapping - macro-F1 0.824 vs balanced accuracy

Balanced accuracy is (TPR + TNR)/2 and is invariant to base rate. Macro-F1 averages per-class F1 and is **not** base-rate invariant, because precision moves with prevalence. They are close but not interchangeable.

At our positive base rate of 0.649:

| TPR / TNR | Balanced accuracy | Implied macro-F1 |
|---|---|---|
| 0.830 / 0.830 | 0.830 | 0.819 |
| 0.835 / 0.835 | 0.835 | 0.824 |
| 0.880 / 0.780 | 0.830 | 0.830 |
| 0.780 / 0.880 | 0.830 | 0.808 |

- Macro-F1 **0.824 at a 0.649 positive rate corresponds to balanced accuracy roughly 0.82 - 0.85**, centred near 0.835, depending on how the errors split between TPR and TNR
- Publishing TPR and TNR alongside macro-F1 removes this ambiguity entirely and costs nothing
- Naive-majority macro-F1 0.417 corresponds to balanced accuracy 0.500 - the two floors agree, which is a sanity check that the mapping is behaving

## Question 1 - which single benchmark

**LLM-AggreFact.** Four reasons, in order of weight.

- **Task identity** - it is literally document + claim → binary supported; no aggregation rule to invent, no span offsets to produce, no generator to stand up; every other candidate requires reshaping our output or our task
- **A published reference class** - it is the only benchmark with a documented ladder of *non-LLM encoder* scores (61.4 plain NLI → 70.4 AlignScore → 72.6 MiniCheck-DeBERTa → 75.6 FactCG-DeBERTa-L); that ladder is the actual answer to "is 0.824 good", far more than any single top score
- **Cost of entry** - one `load_dataset` call, one sklearn metric, a published demo notebook defining the protocol exactly; the only engineering is long-document chunking
- **Adversarial breadth** - 11 heterogeneous slices spanning news, meetings, dialogue, science and healthcare, with a known-hard slice (ExpertQA, nobody above 60.9) that will expose whether our numbers come from easy positives

Second choice: **TRUE**, specifically for the reranker AUC 0.841 - it is the only benchmark whose metric accepts a continuous score directly. Third: **Mu-SHROOM**, the only independently-adjudicated evaluation and the only credible multilingual claim, at the cost of adapting to span output.

Explicitly rejected as primary: FACTS Grounding and the Vectara leaderboard rank generators, not verifiers. HaluBench is NC-licensed and vendor-authored. FaithBench at n=660 with a 57.65 ceiling cannot rank anything.

## Question 2 - where would macro-F1 0.824 plausibly sit

**Estimate, not a measurement.** This maps across differing metrics, label schemas and data distributions, and should be treated as a prior to be replaced by an actual run.

- Converting metrics only: macro-F1 0.824 at base rate 0.649 ≈ **balanced accuracy 0.82 - 0.85**, which is numerically **above the LLM-AggreFact leaderboard top of 77.4**
- **That almost certainly does not mean we beat state of the art.** The far likelier reading is that our private set is easier, or more homogeneous, or labelled with a more permissive notion of "grounded" than the 11 human-annotated academic sets
- **Realistic expectation on an actual LLM-AggreFact run: mid-60s to low-70s balanced accuracy.** The reasoning: off-the-shelf NLI sits at 61.4, and grounding-specific supervised training buys ~9 to ~14 points (AlignScore 70.4, MiniCheck-DeBERTa 72.6, FactCG 75.6). A cascade of untrained-for-this-benchmark components, plus a reranker signal that adds genuine discrimination, plausibly lands **63 - 72**
- **Landing above ~73 without benchmark-specific fine-tuning would be a genuinely strong result** for a torch-free int8 CPU system, and would put us within the leaderboard band alongside 70B-class models
- **Landing below ~62** would say the cascade is not adding value over a bare NLI head, and the private-set number reflects the private set rather than the method
- **The single most informative number is not the average.** It is the ExpertQA slice: every system on the board sits at 58.3 - 60.9 there. If we land near 60 we are behaving like the field; well above suggests leakage or an easier reading of the labels, well below points at a specific failure mode

### What would make the comparison invalid

- **Label-definition drift** - if our "grounded" admits reasonable inference from the document while LLM-AggreFact demands explicit support (or vice versa), the two numbers measure different predicates and no conversion rescues them
- **Metric substitution** - reporting our macro-F1 next to their balanced accuracy without publishing TPR/TNR; the published critique of these metrics shows systems with identical balanced accuracy can have inverted TPR/TNR (on ExpertQA: 68/53 for gpt-4-turbo against 53/68 for Bespoke-7B)
- **Scores do not transfer between datasets** - the same critique finds the top two evaluators agree on under 50% of predicted-unattributable examples (intersection-over-union) on 5 of 14 datasets, and mis-rank 20 - 26% of system pairs; a good LLM-AggreFact score is not evidence of a good score on our own distribution, and the converse holds too
- **Chunking artefacts** - if long documents are chunked to fit a 512-token window, claims needing evidence from distant sections fail systematically; that is our engineering choice showing up as a benchmark score, not a property of the method
- **Threshold transfer** - a decision threshold tuned on our private set will not be optimal on 11 foreign distributions; report both the transferred threshold and a per-benchmark-tuned one, and label which is which
- **Base-rate mismatch** - our 0.649 positive rate is ours; the 11 slices each have their own, and macro-F1 moves with prevalence while balanced accuracy does not
- **Contamination** - the dataset ships a `contamination_identifier` field precisely because these documents are in public pretraining corpora; relevant to any LLM component, less so to an encoder cascade, but must be stated
- **Skipped claims** - if the cascade declines to verdict on some inputs, the benchmark has no abstain class; forcing a default silently changes the metric, and the skip rate must be published alongside

## Recommended next step

- Run the cascade over `lytang/LLM-AggreFact` test, following the official per-dataset-then-mean protocol, and publish: the 11 per-slice balanced accuracies, the mean, TPR and TNR per slice, the skip rate, and the chunking strategy
- Publish the reranker's raw AUC per slice as well - it is directly comparable to TRUE's 87.8 ceiling and is our strongest single signal
- Treat the ExpertQA slice as the honesty check on the private-gold number
- Only after that, decide whether a multilingual claim needs Mu-SHROOM or PsiloQA adaptation, and keep it as a separate result rather than folding it into a headline average

## Sources

- [LLM-AggreFact Leaderboard](https://llm-aggrefact.github.io/)
- [lytang/LLM-AggreFact on HuggingFace](https://huggingface.co/datasets/lytang/LLM-AggreFact)
- [MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents (arXiv 2404.10774)](https://arxiv.org/abs/2404.10774)
- [MiniCheck repository and evaluation demo](https://github.com/Liyan06/MiniCheck)
- [Verify with Caution: The Pitfalls of Relying on Imperfect Factuality Metrics (arXiv 2501.14883)](https://arxiv.org/html/2501.14883v1)
- [Paladin-mini (arXiv 2506.20384)](https://arxiv.org/html/2506.20384v1)
- [FACTS Grounding, Google DeepMind](https://deepmind.google/blog/facts-grounding-a-new-benchmark-for-evaluating-the-factuality-of-large-language-models/)
- [FACTS Benchmark Suite, Google DeepMind](https://deepmind.google/blog/facts-benchmark-suite-systematically-evaluating-the-factuality-of-large-language-models/)
- [google/FACTS-grounding-public](https://huggingface.co/datasets/google/FACTS-grounding-public)
- [TRUE: Re-evaluating Factual Consistency Evaluation (arXiv 2204.04991)](https://arxiv.org/pdf/2204.04991)
- [google-research/true](https://github.com/google-research/true)
- [TrueTeacher (arXiv 2305.11171)](https://arxiv.org/pdf/2305.11171)
- [RAGTruth (ACL 2024)](https://aclanthology.org/2024.acl-long.585/)
- [LettuceDetect (arXiv 2502.17125)](https://arxiv.org/pdf/2502.17125)
- [Beyond Document Grounding: Span-Level Hallucination Detection over Code, Tool Output, and Documents (arXiv 2607.00895)](https://arxiv.org/html/2607.00895)
- [PatronusAI/HaluBench](https://huggingface.co/datasets/PatronusAI/HaluBench)
- [Salesforce/summedits](https://huggingface.co/datasets/Salesforce/summedits)
- [FaithBench (NAACL 2025)](https://arxiv.org/html/2410.13210v1)
- [Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards (arXiv 2505.04847)](https://arxiv.org/pdf/2505.04847)
- [vectara/hallucination-leaderboard](https://github.com/vectara/hallucination-leaderboard)
- [RAGBench (arXiv 2407.11005)](https://arxiv.org/abs/2407.11005)
- [PsiloQA (arXiv 2510.04849)](https://arxiv.org/abs/2510.04849)
- [Mu-SHROOM, SemEval-2025 Task 3](https://helsinki-nlp.github.io/shroom/2025.html)
- [Poly-FEVER (arXiv 2503.16541)](https://arxiv.org/abs/2503.16541)
- [X-FACT (arXiv 2106.09248)](https://arxiv.org/abs/2106.09248)
- [MEMERAG (arXiv 2502.17163)](https://arxiv.org/pdf/2502.17163)
- [MUCH (arXiv 2511.17081)](https://arxiv.org/abs/2511.17081)
- [ExpertQA (arXiv 2309.07852)](https://arxiv.org/abs/2309.07852)
- [facebook/anli](https://huggingface.co/datasets/facebook/anli)
- [tals/vitaminc](https://huggingface.co/datasets/tals/vitaminc)
- [fever/fever](https://huggingface.co/datasets/fever/fever)

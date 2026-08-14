# Training Corpora for a Grounding Verifier - Class A / Class B Survey

Survey of public data we could TRAIN on, not evaluate on. Companion to `reports/research-grounding-benchmarks.md`, which covers evaluation; nothing here repeats that file except where a corpus serves both roles.

Split by the distinction that matters for us: **Class A** carries grounding labels already, **Class B** is real conversation data we would label ourselves with the frozen cascade. Research date 2026-07-28.

## Headline

- **The Wikipedia-derived corpora are the wrong register** - VitaminC, FEVER, ANLI, X-Fact and Poly-FEVER are declarative encyclopaedic claims against encyclopaedic text; near-miss negatives do not compensate for a document structure and failure mode our production traffic does not have
- **The best domain match with labels is the RAGTruth family** - real LLM RAG answers, human span annotation, MIT, and now machine-translated into 7 further languages at `KRLabsOrg/ragtruth-*-translated`
- **The best domain match without labels is Search Arena** - 24,069 real multi-turn search-assistant conversations across ~90 languages, with a `web_search_trace` field; this is the only large corpus of genuine retrieval-augmented user traffic in public
- **Three headline corpora are licence-dead for us** - TrueTeacher CC-BY-NC-4.0 (1.38M examples), HaluBench CC-BY-NC-2.0, MS MARCO Microsoft non-commercial-research-only
- **MEMERAG forbids training explicitly** - its card states the data "should not be used to train models", so it stays an eval set regardless of quality
- **Multilingual with source documents attached is rare** - PsiloQA (14), LettuceDetect prose (14), RAGTruth-translated (7 + English), MIRAGE-Bench (18), NoMIRACL (18), Search Arena (~90); everything else is English

## Class A - real RAG or conversation data WITH grounding labels

| Dataset | HF id | Size | Negatives (how made) | Labels (how made) | Languages | Licence | Used for training by |
|---|---|---|---|---|---|---|---|
| **RAGTruth** | `wandb/RAGTruth-processed` (mirror; official GitHub `ParticleMedia/RAGTruth`) | 17,790 responses - 15,090 train / 2,700 test | **Natural LLM hallucinations** - 6 models (Llama2 7/13/70B, Mistral-7B, GPT-3.5, GPT-4) answering real retrieval prompts; no perturbation | Human expert span annotation | English | MIT | LettuceDetect v1 + v2, RAGTruth fine-tunes; also an LLM-AggreFact eval slice |
| **RAGTruth translated x7** | `KRLabsOrg/ragtruth-{de,fr,es,it,pl,hu,cn}-translated` | 17,790 each (15,100 train / 2,700 test) → ~106k train total | Inherited real LLM hallucinations, translated with context | Inherited human spans, re-aligned after MT (Gemma 3 27B, vLLM) | de, fr, es, it, pl, hu, zh | MIT | LettuceDetect v2 |
| **RAGTruth-DE human-checked** | `KRLabsOrg/ragtruth-de-translated-manual-300` | 300 | as above | Human-verified translation subset | de | MIT | LettuceDetect v2 (MT quality control) |
| **LettuceDetect v2 prose** | `KRLabsOrg/lettucedetect-prose-hallucination` | 87,800 - 78,900 train / 3,360 val / 5,600 test | **Near-miss by construction** - LLM proposes localized replacement edits (wrong value, wrong identifier, unsupported addition), applied deterministically to recover exact offsets; card also cites weak-LLM (TinyLlama-1.1B) generation classified by a Qwen3.6-35B judge | LLM-generated + LLM-judged, char spans | 14 (via PsiloQA + RAGTruth) | CC-BY-4.0 | LettuceDetect v2 (`lettucedect-v2-mmbert-base`, `-qwen-2b`) |
| **LettuceDetect v2 code** | `KRLabsOrg/lettucedetect-code-hallucination` | 74,300 | Injected edits - invented APIs/identifiers, wrong values | LLM-generated | English + code | CC-BY-4.0 (unverified on card) | LettuceDetect v2 |
| **RAGBench** | `galileo-ai/ragbench` | 100k total, **78k train** across 12 subsets | **Natural LLM hallucinations** - GPT-3.5-0125 and Claude-3-Haiku prompted permissively (no adherence instruction) so authentic drift appears | **GPT-4-0125-preview with chain-of-thought** - no human annotation | English | CC-BY-4.0 (**but see licence caveat below**) | Galileo Luna (vendor); not used by MiniCheck/AlignScore |
| **PsiloQA** | `s-nlp/PsiloQA` | 63,792 train / 3,355 val / 2,897 test | **Natural LLM hallucinations** - diverse LLMs answer in a no-context setting, then compared against retrieved Wikipedia; wrong entity/date/number is the dominant error | GPT-4o auto-annotation end to end (QA generation + span marking); no human verification | 14 - en, de, fr, es, it, ca, eu, sv, fi, cs, ar, fa, hi, zh | CC-BY-4.0 | LettuceDetect v2, PsiloQA baselines |
| **FaithDial** | `McGill-NLP/FaithDial` | 50,761 turns / 5,649 dialogues (HF plain_text 32.3k rows, 18.4k train) | **Genuine human-written hallucinated utterances** from Wizard of Wikipedia, retained alongside the amended faithful version | Human - Amazon MTurk amendment plus BEGIN labels (Hallucination / Entailment / Generic) | English | MIT | FaithDial hallucination critic (+12.8 F1 on BEGIN vs prior data) |
| **ExpertQA** | `cmalaviya/expertqa` | 2,177 questions, ~2,170 lfqa split; claim-level rows in the low tens of thousands | Real unsupported claims in model answers to real expert questions | **Human - 484 domain experts judged their own questions'** answers for `support`, informativeness, correctness | English | MIT | Eval only (LLM-AggreFact hardest slice, 58 - 61 BAcc) |
| **MTRAG** | GitHub `IBM/mt-rag-benchmark` (no HF id) | 110 conversations, 842 tasks, avg 7.7 turns | Natural LLM answers over 4 corpora (ClapNQ, FiQA, Govt, Cloud) | Human faithfulness judgments (MTRAGEval) | English | CC-BY-4.0 | Eval only |
| **HaluEval** | `pminervini/HaluEval`, `flowaicom/HaluEval` | ~35k - 5k real ChatGPT user queries + 30k task-specific | **LLM-constructed near-miss** - ChatGPT sampling-then-filtering picks the most plausible-yet-wrong candidate | Human annotation on the 5k real-query slice; LLM-constructed on the 30k | English | MIT | Eval mostly; a HaluBench component |
| **MultiDoc2Dial** | `IBM/multidoc2dial` | 4,796 dialogues, 61,078 turns, 488 documents | **None** - grounding spans are all positive | Human grounding-span annotation | English | Apache-2.0 | Eval / retrieval training |
| **HAGRID** | `miracl/hagrid` | ~2.6k queries over MIRACL English | Answers judged non-attributable | Human judgment of informativeness + attributability over GPT-3.5 answers | English (MIRACL subset) | Apache-2.0 (verify) | Eval; a RAGBench component |
| **WiCE** | `jon-tow/wice` | 1,260 / 349 / 358 claims → 3,470 train subclaims | Real unsupported Wikipedia citation claims | Human - supported / partially-supported / not-supported plus unsupported tokens | English | unclear on card - **verify before use** | Eval (LLM-AggreFact slice) |

### Class A - excluded, and why

- **TrueTeacher** `google/trueteacher` - 1,383,462 examples, FLAN-PaLM-540B labels over T5 summaries of CNN/DailyMail; **CC-BY-NC-4.0, research use only** → unusable commercially regardless of size, and it is news summarization not RAG
- **HaluBench** `PatronusAI/HaluBench` - 14,900 examples, closest shape after LLM-AggreFact; **CC-BY-NC-2.0** → excluded
- **MEMERAG** - 5 answer languages, native-expert faithfulness labels, high inter-annotator agreement; the card states **data "should not be used to train models"** → excluded by the authors' own terms
- **Mu-SHROOM** `Helsinki-NLP/mu-shroom` - CC-BY-4.0, 3,350 train_unlabeled / 499 val / 1,900 test, 14 languages, but the schema carries `model_input` and `model_output_text` with **no source-document field** (`wikipedia_url` is empty in visible rows) → not a claim+document pair set without re-retrieving the evidence ourselves
- **FactCG / CG2C** - the synthetic multi-hop training data is **not released**; only the `yaxili96/FactCG-DeBERTa-v3-Large` checkpoint (MIT) → nothing to train on
- **AlignScore's 4.7M** - not a downloadable corpus; it is a recipe that samples ≤500k rows from ~15 existing datasets (SNLI, MNLI, ANLI, PAWS, FEVER, VitaminC, QA and IR sets), several of which are CC-BY-NC → the licence of the mix is the worst licence in it
- **MiniCheck C2D/D2C** `lytang/C2D-and-D2C-MiniCheck` - only 14k (7k C2D + 7k D2C), GPT-4-synthesised documents and claims in Wikipedia/news register; the released MiniCheck recipe is 21k ANLI + this 14k, and **ANLI is CC-BY-NC-4.0** so the published recipe is not reproducible commercially; C2D/D2C card licence not stated - **verify**
- **MS MARCO** - real Bing queries, real passages, human answers, and exactly our shape once labelled, but **"non-commercial research purposes only"** per Microsoft's terms → excluded
- **Wikipedia-derived fact verification** - VitaminC `tals/vitaminc` (~489k contrastive revision pairs, the textbook near-miss construction), FEVER `fever/fever` (185,445), FEVEROUS `fever/feverous` (87k), ANLI `facebook/anli` (162k, **CC-BY-NC-4.0**), X-Fact (25 languages, real-world political claims + web evidence, 7-way), Poly-FEVER (11 languages, machine-translated FEVER). Per the owner's direction these are deprioritised on register, not on quality - VitaminC's negatives are the best-constructed in the field and it remains the fallback if domain-matched data underdelivers

### The RAGBench licence caveat

RAGBench is tagged CC-BY-4.0 at the collection level, but it is assembled from 12 upstream corpora including **MS MARCO**, which Microsoft licenses for non-commercial research only, and **CUAD**. A CC-BY-4.0 tag applied by a vendor over mixed upstream terms does not extinguish the upstream restriction. Treat the MS MARCO and CUAD subsets as licence-suspect and either drop them or seek counsel; the remaining 10 subsets (TechQA, EManual, DelucionQA, ExpertQA, HAGRID, FinQA, TatQA, CovidQA, PubMedQA, HotpotQA) are the safe core.

## Class B - real conversation corpora WITHOUT grounding labels

Candidates for self-labelling with the frozen cascade. The gating question for every row is whether the **source documents ship with the dialogue** - a conversation without its evidence cannot be grounded and is worthless to us.

| Dataset | HF id | Size | Sources included? | Real interaction? | Languages | Licence | Notes |
|---|---|---|---|---|---|---|---|
| **Search Arena 24k** | `lmarena-ai/search-arena-24k` | 24,069 multi-turn conversations, 12,652 preference votes, ~11k users, 136 countries, 13 models | **Probably yes - unverified.** Schema exposes `system_{a,b}_metadata.web_search_trace`, `web_search_config` (`search_engine`, `scrape_engine`, `context_manager`) and `llm_trace`; the presence of a scrape engine implies page text, but the HF viewer fails at 400MB so the trace payload is unconfirmed | **Yes** - in-the-wild traffic, Mar 18 - May 8 2025 | **~90**, 11% multilingual prompts | **Split** - user prompts CC-BY-4.0; model outputs "governed by respective provider terms" | The single best domain match in public. Provider-terms clause is a real complication for training on the answers |
| **MIRAGE-Bench** | GitHub `vectara/mirage-bench` | 39,763 train pairs + 11,195 eval | **Yes** - oracle-judged MIRACL passages | Partly - MIRACL queries are written by native speakers, answers are LLM-generated | **18** | verify (MIRACL upstream is Apache-2.0) | Multilingual query + passage + answer with no per-claim gold → ideal self-labelling target |
| **NoMIRACL** | `miracl/nomiracl` | 18 language subsets, ≤10 annotated passages per query | **Yes** - passage text with docid/title/text | MIRACL queries; human relevance judgments | **18** | Apache-2.0 | Uniquely supplies the *no relevant evidence* case - the true-negative half our gold is thin on |
| **WildChat-1M** | `allenai/WildChat-1M` | 1M conversations | **No** - dialogue only | **Yes** - real ChatGPT traffic with geo metadata | 60+ | ODC-BY | No retrieval, no evidence → not usable for grounding |
| **LMSYS-Chat-1M** | `lmsys/lmsys-chat-1m` | 1M conversations, 210k users, 25 models | **No** - dialogue only | **Yes** - Chatbot Arena and Vicuna demo | 150 | LMSYS custom licence, permits research **and commercial** use; gated (must accept terms) | Same problem - no evidence attached |
| **OASST2 / ShareGPT** | `OpenAssistant/oasst2` | ~135k messages (OASST2) | **No** | Human-written, not production traffic | 35 | Apache-2.0 (OASST); ShareGPT derivatives have unclear provenance | Not usable for grounding |
| **MS MARCO** | `microsoft/ms_marco` | 1M+ real Bing queries, 8.8M passages, human answers | **Yes** | **Yes** - genuine Bing query log | English (mMARCO covers 13) | **Non-commercial research only** | Perfect shape, dead licence |

### Class B - the honest read

- **Only one corpus is both real retrieval-augmented traffic and public** - Search Arena. Everything else is either real chat without evidence (WildChat, LMSYS, OASST) or evidence-bearing but not real user traffic (MIRAGE-Bench, NoMIRACL, MultiDoc2Dial)
- **Search Arena's answers are the licence risk, not its prompts** - prompts are CC-BY-4.0, but "model outputs governed by respective provider terms" means training a verifier on GPT/Gemini/Perplexity output falls under each provider's terms of service, several of which restrict using outputs to build competing models. A verifier is arguably not a competing model; this needs a legal read, not an engineering one
- **The self-labelling ceiling is real** - cascade labels on Class B are soft labels from a teacher we already have. They add **new documents, new registers, new languages and new claim surface** - which is exactly the 639-trace problem - but they cannot correct the teacher where it is wrong. Expect distribution broadening, not accuracy transfer
- **21-language coverage from Class B** - Search Arena ~90 and MIRAGE-Bench/NoMIRACL 18 are the only realistic paths to broad language coverage with evidence attached

## The three I would train on

1. **RAGTruth + its 7 translations** (`wandb/RAGTruth-processed` plus `KRLabsOrg/ragtruth-{de,fr,es,it,pl,hu,cn}-translated`) - ~121k training pairs, MIT throughout, negatives are **naturally occurring LLM RAG hallucinations annotated by human experts**, which is the exact error distribution our production cascade sees. 8 languages. The only corpus in this survey where domain, negative construction and licence are all right at once. Caveat: the 7 translations are machine-translated by Gemma 3 27B, so non-English label alignment is inherited and only 300 German rows are human-checked
2. **LettuceDetect v2 prose** (`KRLabsOrg/lettucedetect-prose-hallucination`) - 78,900 train, CC-BY-4.0, **14 languages**, source context in a dedicated field, and negatives built as localized replacement edits (wrong value, wrong identifier, unsupported addition) - the near-miss construction, at multilingual scale. Its documents are ACL papers, READMEs and Wikipedia markdown rather than business documents, so treat it as language-coverage supervision rather than domain supervision
3. **RAGBench minus MS MARCO and CUAD** (`galileo-ai/ragbench`) - ~78k train, CC-BY-4.0, natural GPT-3.5 and Claude-Haiku hallucinations over **genuine enterprise-shaped documents** (TechQA support tickets, EManual consumer manuals, DelucionQA car manual, financial and legal corpora). English only and **labelled by GPT-4 rather than humans**, so label noise is a known and unmeasured quantity - but it is the only large corpus whose documents look like ours

Runner-up on evidence quality rather than size: **FaithDial** (MIT, real human-human information-seeking dialogue with the Wikipedia snippet attached and human hallucination labels) - 18.4k train, and the only corpus here where both sides of the conversation are human.

## Does public data meaningfully broaden 639 traces?

**Partly, and less than the raw counts suggest. Be sceptical of transfer, but the language axis is real.**

- **Document count is where the win is** - 619 unique source documents is the binding number. RAGTruth alone brings thousands of distinct retrieval contexts, RAGBench tens of thousands across 10 usable domains, PsiloQA and LettuceDetect prose tens of thousands more in 14 languages. On document diversity, public data is a 50 - 100x expansion and that is not a marginal change
- **Domain transfer is the part that will disappoint** - none of these are production assistant traffic over a private corpus. RAGTruth's retrieval contexts are MS MARCO / CNN-DM / Yelp derived, RAGBench's are public manuals and filings, PsiloQA's are Wikipedia. Register, chunk structure and retrieval failure modes differ from ours. Expect a verifier trained on them to be better calibrated on generic English RAG and roughly unchanged on our hardest slice
- **The multilingual case is stronger than the English case** - our English slice is the largest and hardest, and public English data will not fix a hard slice built on private documents. But 21 languages against 639 traces means the non-English slices are thin by construction, and RAGTruth-translated (7) + PsiloQA (14) + LettuceDetect prose (14) is a genuine, licence-clean expansion there. **This is the clearest argument for using public data at all**
- **The blunt version** - public data will not teach the verifier our documents. It can plausibly teach it (a) what an LLM hallucination looks like across many document types, and (b) how to do that in 14+ languages instead of 1. Those are worth having; a jump on our own gold set is not the expected outcome and should not be the success criterion
- **Class B is the more interesting lever and the less certain one** - Search Arena at ~90 languages of real retrieval-augmented traffic is the only corpus that broadens the *trace* distribution rather than the *document* distribution. Its value depends entirely on whether `web_search_trace` contains page text, which is unverified, and on a licence read of provider output terms

## Cheapest way to test transfer before a training run

Three probes, each under a day, in this order. Stop at the first one that fails.

1. **Zero-shot the frozen cascade on each candidate's TEST split, no training** (~2 hours) - run the existing cascade over RAGTruth test (2,700), RAGBench test, PsiloQA test (2,897) and FaithDial test. If the cascade already scores near its private-gold macro-F1 on a corpus, that corpus contains nothing it does not know and training on it is wasted. If it scores far *below*, that is the corpus with new signal - and the size of the gap is the size of the opportunity. **This costs nothing but inference and it is the single highest-information experiment available**
2. **Reverse-direction probe with an existing public verifier** (~3 hours, no training) - run `KRLabsOrg/lettucedect-v2-mmbert-base` (MIT, trained on exactly the Class A stack recommended above) over our 2,752-claim private gold, per language. It is a trained-on-public-data verifier evaluated on our distribution - a direct measurement of public → private transfer with zero training cost. If it lands near our cascade, public training data transfers; if it collapses on our hardest slice, it does not, and no amount of it will
3. **Smallest honest fine-tune, one axis only** (~1 day) - take the ~385 trace-split training traces, fine-tune the reranker head twice: once on private only, once on private + 20k sampled RAGTruth-family rows. Evaluate both on the held-out private traces, **reporting per-language**. Weight the non-English slices - probe 2 will already have told you whether to expect English movement. Only if this shows a per-language gain is the full multi-corpus run justified
4. **Class B verification is a 20-minute job, do it before anything else** - download one shard of `lmarena-ai/search-arena-24k` and inspect `system_a_metadata.web_search_trace` for scraped page text. If it is only URLs and citation counts, Search Arena drops out of consideration entirely and the Class B lane collapses to MIRAGE-Bench and NoMIRACL

**Kill criterion**: if probe 1 shows the cascade already at parity on public test splits AND probe 2 shows a public-trained verifier failing on our gold, then public data is a different problem and the honest conclusion is to spend the budget on annotating more private traces instead.

## Re-survey 2026-08-13 - the 2023-2026 hallucination-detection circuit beyond the register

A web reconnaissance pass over the post-2023 literature (licences verified at source - card YAML, repo LICENSE files - not from marketing pages; contamination-wall derivation checked against each paper's construction section). Everything already in the mix, the register, or the licence-excluded list was out of scope. Net: four admit-candidates and two high-value flagged items; the rest of the circuit is licence-dead, evidence-less, eval-only, or inside the contamination wall.

**ADMIT-CANDIDATES** (clean licence, evidence ships, wall-clean; admission ruling is the author's):

- **FAVA fava-data** (2024, arXiv 2401.06855, CC-BY-4.0) - 30,073 train rows: long-form responses with span-level tags across six error types against embedded reference passages, plus 460 human-annotated gold. The only span-level long-form taxonomy set found; synthetic negatives. Caveat: Wikipedia/web register, not business documents
- **PubHealth** (2020, EMNLP, MIT) - 12,288 real-world public-health fact-checks (Snopes/PolitiFact/Full Fact lineage) with source article text and journalist explanations; human labels throughout, a register nothing in the mix covers. Mapping of the 4-way verdict needs a ruling (recommend unproven/mixture -> 0 or drop); per-row evidence completeness unverified (HF viewer disabled - check on pull)
- **MiniCheck C2D/D2C** (2024, EMNLP, MIT on the card) - 14,395 (claim, doc) pairs, GPT-4-synthesised for multi-fact, multi-sentence checking. Gate before mixing: the seed corpora are named only in the paper's Appendix D - run the R13-style 8-gram provenance instrument against the ten walled corpora on pull; if seeds are HotpotQA-derived it dies at the gate
- **FActScore labeled biographies** (2023, EMNLP, MIT) - order 10k human-labeled atomic facts over LM biographies against Wikipedia. Small; a high-precision slice or a held-out long-form probe, not bulk supervision

**FLAGGED - rulings needed**:

- **FinDVer** (2024, EMNLP, MIT) - 2,400 claims over 2024 SEC 10-K/10-Q filings with paragraph/table evidence indices; the ONLY candidate filling the financial numeric-derivation gap directly (the flagship's sole losing register). Flag: shares the SEC EDGAR population with the walled FinQA/TAT-QA lineage - but its documents are 2024 filings, the walled corpora are pre-2020, so document overlap is structurally impossible. The register's own EDGAR restricted slice was admitted under the same population logic with the 8-gram instrument at 0.0; FinDVer inherits that gate
- **AttributionBench** (2024, ACL Findings, Apache-2.0) - 26.4k attribution pairs BUT its train split contains walled ExpertQA (4,442 rows) and HAGRID (1,088). A carve-out via the `src_dataset` field keeps ~18k clean pairs (Stanford-GenSearch, AttributedQA, LFQA, BEGIN, AttrEval); precedent: the RAGBench fetch itself excluded MS MARCO/CUAD subsets. Gate verifies zero walled rows on pull

Also flagged and parked: FACTS Grounding (licence unverified at source; Class B only), Climate-FEVER / HealthVer / COVID-Fact (licence absent; the latter two also sit on the CORD-19 population near walled CovidQA), DialFact (upstream WoW/TopicalChat licence chain unverified), MNBM XSum annotations (BBC redistribution provenance).

**Dead on arrival** (recorded so they are never re-surveyed): CRAG, HalluLens, FaithBench, FELM, HaluQuestQA, MultiFC, AVeriTeC, HealthFC (all NC-class licences); Hover (HotpotQA-derived - walled), MedHallu (PubMedQA-derived - walled); TofuEval (explicit no-training terms); RAGChecker (gated components); X-Fact / Poly-FEVER / LongFact (no evidence ships); HaluEval-Wild (no grounding labels); ASQA/QAMPARI/ELI5 (no support labels - Class B at best); FRANK/FactCC/SummEval/QAGS/Polytope (no data licence, CNN/DM-XSum provenance); TRUE (aggregate of the above); FAME/FactualBench/HALoGEN/CCHall/MultiHal/REFUTE-v3/DiaHalu (eval-only); WildHallucinations (never released); Kaggle sci-hallucination 2025 (competition licence).

## Sources

- [wandb/RAGTruth-processed](https://huggingface.co/datasets/wandb/RAGTruth-processed) and [ParticleMedia/RAGTruth](https://github.com/ParticleMedia/RAGTruth)
- [KRLabsOrg datasets](https://huggingface.co/KRLabsOrg) - [ragtruth-de-translated](https://huggingface.co/datasets/KRLabsOrg/ragtruth-de-translated), [lettucedetect-prose-hallucination](https://huggingface.co/datasets/KRLabsOrg/lettucedetect-prose-hallucination)
- [Beyond Document Grounding: Span-Level Hallucination Detection (arXiv 2607.00895)](https://arxiv.org/html/2607.00895v1)
- [LettuceDetect (arXiv 2502.17125)](https://arxiv.org/pdf/2502.17125)
- [galileo-ai/ragbench](https://huggingface.co/datasets/galileo-ai/ragbench) and [RAGBench paper (arXiv 2407.11005)](https://arxiv.org/html/2407.11005v1)
- [s-nlp/PsiloQA](https://huggingface.co/datasets/s-nlp/PsiloQA) and [PsiloQA paper (arXiv 2510.04849)](https://arxiv.org/abs/2510.04849)
- [McGill-NLP/FaithDial](https://huggingface.co/datasets/McGill-NLP/FaithDial) and [FaithDial paper (arXiv 2204.10757)](https://arxiv.org/abs/2204.10757)
- [cmalaviya/expertqa](https://huggingface.co/datasets/cmalaviya/expertqa)
- [google/trueteacher](https://huggingface.co/datasets/google/trueteacher) - CC-BY-NC-4.0
- [lytang/C2D-and-D2C-MiniCheck](https://huggingface.co/datasets/lytang/C2D-and-D2C-MiniCheck)
- [FactCG (arXiv 2501.17144)](https://arxiv.org/abs/2501.17144) and [derenlei/FactCG](https://github.com/derenlei/FactCG)
- [AlignScore (ACL 2023)](https://aclanthology.org/2023.acl-long.634.pdf)
- [lmarena-ai/search-arena-24k](https://huggingface.co/datasets/lmarena-ai/search-arena-24k) and [Search Arena paper (arXiv 2506.05334)](https://arxiv.org/abs/2506.05334)
- [miracl/nomiracl](https://huggingface.co/datasets/miracl/nomiracl) and [NoMIRACL paper (arXiv 2312.11361)](https://arxiv.org/abs/2312.11361)
- [MIRAGE-Bench (arXiv 2410.13716)](https://arxiv.org/pdf/2410.13716) and [vectara/mirage-bench](https://github.com/vectara/mirage-bench)
- [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) - ODC-BY
- [lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
- [MS MARCO terms](https://microsoft.github.io/msmarco/) - non-commercial research only
- [MEMERAG (arXiv 2502.17163)](https://arxiv.org/abs/2502.17163) and [amazon-science/MEMERAG](https://github.com/amazon-science/MEMERAG)
- [Helsinki-NLP/mu-shroom](https://huggingface.co/datasets/Helsinki-NLP/mu-shroom)
- [IBM/multidoc2dial](https://huggingface.co/datasets/IBM/multidoc2dial) and [MTRAG (arXiv 2501.03468)](https://arxiv.org/html/2501.03468v1)
- [HAGRID (arXiv 2307.16883)](https://arxiv.org/abs/2307.16883)
- [HaluEval (arXiv 2305.11747)](https://arxiv.org/abs/2305.11747)
- [PatronusAI/HaluBench](https://huggingface.co/datasets/PatronusAI/HaluBench) - CC-BY-NC-2.0
- [WiCE (arXiv 2303.01432)](https://arxiv.org/abs/2303.01432)

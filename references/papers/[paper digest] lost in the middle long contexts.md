**Lost in the Middle: How Language Models Use Long Contexts (2023)**

Liu et al. ask a narrow, controlled question: when relevant information sits somewhere inside a long input context, does its position change what the model can do with it. They construct two tasks where position is the only variable - multi-document question answering built from NaturalQuestions-Open, and a synthetic key-value retrieval task over JSON objects of random UUIDs - and sweep both the number of distractors and the index of the relevant item. Across open models (MPT-30B-Instruct, LongChat-13B-16K) and closed ones (GPT-3.5-Turbo and its 16K variant, Claude-1.3 and its 100K variant), accuracy traces a U-shaped curve: highest when the answer document is first (primacy) or last (recency), lowest in the middle. The degradation is large enough that GPT-3.5-Turbo, with the answer buried mid-context among 20 or 30 documents, scores below its own closed-book accuracy of 56.1%. Extended-context variants are not better at using context than their shorter siblings. The paper also shows retriever recall keeps climbing while reader accuracy saturates, so adding retrieved documents stops paying off well before retrieval stops improving.

**Key mechanism**

- Multi-document QA is instantiated with 2,655 NaturalQuestions-Open queries whose annotated long answer is a paragraph; the gold Wikipedia paragraph is the one answer-bearing document, and k-1 distractors are the top Contriever (MS-MARCO fine-tuned) chunks that contain no annotated answer, presented in decreasing relevance order
- The only manipulations are (i) the index of the answer document among k and (ii) k itself (10, 20, 30 documents, roughly 2K/4K/6K tokens); the desired output never changes, so any accuracy delta is attributable to position or length alone
- The key-value task strips language semantics entirely: a serialized JSON object of k unique 128-bit UUID key-value pairs (75, 140, 300 pairs; 500 examples each) plus one query key, so it measures raw retrieval of matching tokens rather than reasoning
- Scoring is answer-string containment in the generated output, with greedy decoding and a fixed per-model prompt; closed-book (no documents) and oracle (only the answer document) settings bracket the achievable range
- Three explanatory probes: decoder-only versus encoder-decoder (Flan-T5-XXL, Flan-UL2), query-aware contextualization (repeating the query before as well as after the documents), and instruction tuning (MPT-30B-Instruct versus base MPT-30B)

**Main findings**

- The U-shaped curve holds for every evaluated model in multi-document QA; GPT-3.5-Turbo's accuracy swings by more than 20 points as a function of position alone, and in the 20- and 30-document settings the mid-context accuracy falls below its 56.1% closed-book score
- Closed-book versus oracle accuracy: LongChat-13B-16K 35.0% / 83.4%, MPT-30B-Instruct 31.5% / 81.9%, GPT-3.5-Turbo 56.1% / 88.3%, GPT-3.5-Turbo-16K 56.0% / 88.6%, Claude-1.3 48.3% / 76.1%, Claude-1.3-100K 48.2% / 76.4% - the extended-context variant matches its base variant to within 0.3 points in both settings
- On the synthetic key-value task Claude-1.3 and Claude-1.3-100K are near perfect at all lengths, while GPT-3.5-Turbo, GPT-3.5-Turbo-16K and MPT-30B-Instruct degrade sharply in the middle; without query-aware contextualization the worst case is 45.6% accuracy on a pure exact-match retrieval task
- Query-aware contextualization (query placed before and after the data) lifts all models to near-perfect key-value retrieval - GPT-3.5-Turbo-16K reaches 100% at 300 pairs - but barely moves multi-document QA, helping slightly at position 1 and hurting slightly elsewhere
- Encoder-decoder models are position-robust only inside their training-time sequence length: Flan-UL2 varies just 1.9 points absolute between best and worst position within its 2048-token window, and develops the U-shape once inputs exceed it
- The curve is not an artifact of instruction tuning: base MPT-30B shows it too. Instruction tuning narrows the best-worst gap from about 10 points to about 4 points and raises absolute accuracy uniformly
- Model scale matters for the shape: in the Llama-2 appendix, 7B models are purely recency-biased, while 13B and 70B show the full U; supervised fine-tuning and RLHF slightly mitigate the bias at 13B and barely affect 70B
- In the open-domain retriever-reader case study, reader accuracy saturates far earlier than retriever recall - going from 20 to 50 retrieved documents buys roughly 1.5% for GPT-3.5-Turbo and 1% for Claude-1.3
- Methodological caveats the authors state: NaturalQuestions-Open contains some ambiguity so a distractor may occasionally be answerable (a filtered unambiguous subset reproduces the trends); the full GPT-4 sweep was not run because it would have cost upwards of \$6000, and the partial GPT-4 results show the same trends at higher absolute accuracy
- The models studied are the 2023 generation; the finding is a claim about those systems, not a proof about transformers in general

**Key takeaways**

- Long-context capacity is not long-context competence - a model advertising 16K or 100K tokens can be no better at using position 10 of 20 than its 4K sibling. Evaluate a context window by measuring the best-worst spread across positions, which is the protocol this paper proposes
- Prefer many short scored units over one long prompt when the pipeline can choose. Position sensitivity is a property of where an item sits within the window, so a windowing scheme that puts each candidate evidence span near a window boundary sidesteps most of the effect
- Reranking order is load-bearing when documents are concatenated: put the highest-scoring evidence first, and treat the last slot as the second-best position rather than a dumping ground
- More retrieved context is not free. Measure reader accuracy against k directly, and expect the useful k to be well below the point where retriever recall plateaus
- Query-aware contextualization is a cheap intervention with a task-dependent payoff - it essentially solves exact-match retrieval and does close to nothing for reasoning-flavoured QA, so test it rather than adopting it on principle
- A synthetic exact-match probe (the key-value task) is a useful, near-free smoke test for any new long-context model before trusting it with real evidence
- Closed-book accuracy is the honest floor for a retrieval-augmented reader; a configuration that scores below it is actively harmed by its own context

**Relevance**

- Bears directly on the serving read in `docs/experiments/semantic-grounding-sota.md` - the mmBERT cross-encoder scores each (sentence, 1,500-char window) pair separately with stride 750 and takes max over windows, so no evidence ever sits "in the middle" of a long prompt. This paper is the strongest published argument that the windowed formula is the right shape, and it explains why R8-H101 windowing lifted the blind mean (+0.014 to +0.018) even though R8-H85 killed the coarser truncation-coverage hypothesis at gate
- Bears on the labeling pipeline in `docs/experiments/semantic-dataset-enhancements.md`, where a Qwen3-32B-FP8 contrastive judge reads claim plus paired evidence. If judge prompts ever grow to hold many evidence chunks, position becomes a confound in the labels themselves; keeping judge prompts short and evidence-local is the cheap mitigation
- Does not transfer as a numeric baseline. Every measurement here is a 2023 decoder LLM scored by answer-string containment on open-domain QA; groundrails' detector is a 307M encoder cross-encoder scored by AUC on labelled (claim, evidence) pairs, and it has no long-context prompt to be lost in the middle of

**Tags**

- #long-context
- #retrieval
- #evaluation

**Source**

- https://arxiv.org/abs/2307.03172

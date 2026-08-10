**Rephrase and Respond: Let Large Language Models Ask Better Questions for Themselves (2023)**

Deng et al. study a specific failure of zero-shot prompting: questions that are unambiguous to a human can be ambiguous to a language model, and the model then answers a different question than the one asked. Their motivating case is "was X born in an even month", where GPT-4 variously interprets "even month" as a month with an even number of days, declares February "traditionally odd", or denies that parity applies to calendar months at all. The proposed remedy, Rephrase and Respond (RaR), is a single appended instruction - "Rephrase and expand the question, and respond" - that makes the model restate the question in its own terms before answering. A Two-step variant separates the rephrasing model from the responding model, allowing a stronger model to clarify questions for a weaker one. Across ten tasks, RaR raises GPT-4's average zero-shot accuracy from 64.95% to 89.77%, with the largest gains on tasks whose questions are poorly specified for a model. The method is unsupervised, training-free and one line long; the paper is a preprint, and the gains are concentrated in tasks where question quality was the bottleneck.

**Key mechanism**

- One-step RaR appends a fixed instruction to the user question: `"{question}"\nRephrase and expand the question, and respond.` The model emits a rephrasing and an answer in a single completion, so there is one API call and no orchestration
- Two-step RaR splits the loop. A rephrasing model receives the question plus "rephrase and expand it to help you do better answering, maintain all information in the original question"; a responding model then receives both the original and the rephrased question with the instruction "use your answer for the rephrased question to answer the original question". The two models may differ
- Keeping the original question alongside the rephrasing is deliberate - it bounds semantic drift, since the responding model is told to answer the original
- The method is orthogonal to chain-of-thought and composes with it by simply adding "let's think step by step" to the RaR instruction; it is also composable with few-shot CoT exemplars
- Four alternative phrasings of the instruction ("reword and elaborate", "reframe with additional context and detail", "modify for clarity and detail", "restate and elaborate before proceeding") are tested to show the effect is the methodology and not one magic string

**Main findings**

- GPT-4 zero-shot accuracy, original prompt versus RaR, over ten tasks: even day 54.29 → 93.33, even month 58.10 → 83.96, even year 57.14 → 89.52, age comparison 98.44 → 100, CSQA 84.09 → 85.45, date understanding 69.92 → 75.88, sports understanding 79.55 → 84.54, last-letter concatenation with 2 words 52.05 → 99.09, with 4 words 21.36 → 86.82, coin flip 74.55 → 99.09. Average 64.95 → 89.77
- The gain size tracks how badly specified the question is for a model, which the authors state as a diagnostic: a large RaR margin indicates a poorly designed question. CSQA, whose questions are already well posed, gains 1.36 points; last-letter concatenation over four words gains 65.46
- The four prompt variants average 87.17 to 88.97, that is 0.8 to 2.6 points below the main prompt but all well above the 64.95 baseline, so the effect is robust to wording
- Rephrasing capability scales with the model. GPT-4 gains most, GPT-3.5-Turbo and Vicuna-13B-v1.5 gain less, and on the sports task GPT-3.5 and Vicuna slightly regress
- Rephrasing transfers across models. Vicuna-13B answering GPT-4-rephrased questions beats Vicuna answering its own rephrasings on every reported task: even day 58.10 → 61.90, even month 56.19 → 60.95, CSQA 51.36 → 55.00, sports 65.00 → 73.64, dates 32.79 → 37.67, last letter (2) 5.45 → 10.45. The absolute numbers stay low, so clarification helps a weak model but does not rescue it
- Zero-shot CoT is not uniformly safe. On the Chinese-idiom first-character task GPT-4 scores 32.38 originally, 31.43 with zero-shot CoT and 35.24 with RaR - CoT makes it worse, which the authors attribute to hallucination in the intermediate steps compounding
- On StereoSet, RaR raises the Language Modeling Score from 84.09 (unchanged by zero-shot CoT) to 97.73, and the Fair Score from 6.82 to 42.27 with zero-shot CoT at 35.00. Rephrasing largely eliminates selection of the unrelated option
- Few-shot CoT inherits the flaws of its exemplars. With deliberately corrupted exemplars (demonstrating first-letter extraction while asking for last letters), GPT-4 drops to 89.04 on last letter (2) and 78.18 on last letter (4) at one shot, and 87.21 / 52.27 at four shots - more bad exemplars, worse performance. Adding RaR restores 100 / 93.64 at one shot and 100 / 95.45 at four shots
- Caveats that bound the result: preprint, not peer-reviewed; the two knowledge tasks use 105 celebrities and 105 idioms that GPT-4 itself generated because the original data is not open-sourced, which is a self-supplied and therefore favourable test set; several tasks are synthetic symbolic puzzles (coin flip, letter concatenation) where question ambiguity is unusually easy to fix; all GPT-4 access was during a single month in 2023 against a model version that no longer exists
- The comparison baseline is a bare question with no prompt engineering at all, which is a weak baseline for the headline average

**Key takeaways**

- A large accuracy gain from self-rephrasing is a signal about the benchmark, not only about the method. The authors' framing - that human-crafted evaluation tasks should be reviewed by models as well as humans for clarity - is the most transferable idea here, and applies directly to any internally written eval or judge rubric
- Chain-of-thought is not a free win. On a task where intermediate steps can hallucinate, zero-shot CoT measurably lost accuracy; treat CoT as a lever to measure per task, not a default
- Few-shot exemplar quality is load-bearing and silently so. Exemplars whose reasoning is subtly wrong dragged GPT-4 from near-perfect to 52.27, and the model imitated the flawed procedure faithfully. Audit exemplars in any few-shot judge or labeling prompt
- Prompt clarification transfers between models, so a strong model can be used once, offline, to harden a prompt that a cheaper model will then run at scale - a cost structure worth testing wherever a large fleet runs one fixed prompt
- Keep the original text alongside any model rewriting of it, and instruct the consumer to answer the original. That is the cheap guard against a clarification quietly changing the question
- The mechanism is about resolving ambiguity, so expect near-zero gain on tasks whose inputs are already precise - CSQA moved 1.36 points
- The method costs extra output tokens per call and adds a rewriting surface where content can drift; for a constrained classification verdict that surface is a liability rather than an asset

**Relevance**

- Marginal to the current detector campaign. groundrails' detector is a deterministic cross-encoder with no natural-language instruction to rephrase; this paper entered the library from the 2025-04 agentic-RAG assistant reading, before the detector work, and nothing in it bears on the mmBERT recipe or the RAGBench blind gate
- Applies narrowly to prompt hygiene in the labeling pipeline of `docs/experiments/semantic-dataset-enhancements.md`. The few-shot corruption result is the concrete transfer: exemplars in the contrastive judge prompt shape the verdict distribution, and a subtly wrong exemplar will be imitated rather than corrected. The paper's own conclusion - that a benchmark question a model reads differently from its author is a defective question - argues for having a model paraphrase back the judge rubric once during a bake-off, as a cheap ambiguity check on the rubric
- Not a candidate for the serving path. Adding a rephrasing pass would put a generative rewriting step in front of a deterministic verdict, breaking both the determinism guarantee and the latency budget in `docs/experiments/semantic-grounding-sota.md`

**Tags**

- #prompting
- #evaluation
- #llm-agents

**Source**

- https://arxiv.org/abs/2311.04205

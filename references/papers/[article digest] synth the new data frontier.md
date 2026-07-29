**SYNTH: the new data frontier (2025)**

Pleias releases SYNTH, a fully synthetic generalist pretraining dataset built by expanding 50,000 vital Wikipedia articles into problems and resolution paths, and two small reasoning models trained on it - Baguettotron (321M) and Monad (56M). The argument is that standard web-crawl pretraining teaches reasoning only incidentally: the traces exist but are isolated and noisy, which delays acquisition of the skills benchmarks actually measure. SYNTH instead trains directly for reasoning, with every generated answer accompanied by an intermediary reasoning trace. The headline claim is data efficiency rather than raw capability: both models reach state of the art for their size on MMLU, GSM8K and HotPotQA using 10-50x less data than comparable models, on under 200B tokens. This is a vendor blog post, not a peer-reviewed paper, and no arXiv preprint exists for SYNTH, Baguettotron or Monad - the only citable technical document in this line is the earlier Pleias-RAG paper (arXiv 2504.18225). Numbers below are the authors' own and are not independently replicated.

**Key mechanism**

- Seed set is 50,000 vital Wikipedia articles, amplified at minimum 100x into a wide task collection: arithmetic, creative writing, retrieval-augmented generation, information extraction, sourced synthesis
- Every generated answer carries an intermediary reasoning trace in a purpose-built syntax, so the model learns the resolution path rather than only the answer
- Negative queries are deliberately included to reinforce world knowledge boundaries and limit hallucination
- Monad uses a custom BPE tokenizer trained only on the English segment of SYNTH; the authors state this was "a critical measure to contain parameters space, bringing back token embeddings from 20M to less than 2M"
- Baguettotron reuses the standard Pleias multilingual tokenizer, so the two models isolate the tokenizer decision
- Architecture choice was made empirically on top of SYNTH: consistent improvements from stacking more layers, giving Baguettotron 80 layers and Monad 64 layers at hidden size 256

**Main findings**

- Baguettotron (321M) is reported best in class across MMLU, GSM8K and HotPotQA for its size range
- Monad (56M) attains non-random performance on the same suite - MMLU ~30%, GSM8K 8%, HotPotQA 8% per its model card - and is positioned as a contender for the smallest viable language model, under half of GPT-2
- Data efficiency is the central claim: 10-50x less training data than models of similar or lower performance, on fewer than 200B tokens
- Compute is small: final training runs under 1,000 H100 hours; Monad's full pretraining took under six hours on 16 H100s; the entire project including synthetic generation and experiments was 20,000 H100 hours
- Reasoning signal emerges very early - consistent MMLU signal from step 9,000, within the first two hours of Baguettotron's training, versus web-crawl-trained small models that reach non-random results only after trillions of tokens, if at all
- The authors attribute the depth benefit to dense reasoning data: deeper stacks are more often exposed to sequences requiring intensive computation or knowledge interconnection, and depth adds inertia that mitigates surface-level learning
- SYNTH is multilingual (English ~80%, plus French, German, Italian, Spanish, Polish, Dutch, Latin) and not single-turn only, which the authors offer as the reason they avoid the model-collapse failure mode usually associated with synthetic pretraining
- Licence is CC-BY-4.0 (seed texts CC-BY-SA 4.0); models are Apache 2.0
- Explicit limitation: full synthetic training will not build "a GPT-5 at home"; the authors position it as complementary to frontier models

**Key takeaways**

- Vocabulary is where a small model's parameters die - the 20M to under 2M embedding reduction is the single most transferable engineering fact here, and it is what makes a 56M total budget arithmetically possible at all
- A domain-specific tokenizer is a first-class compression lever, not a detail, once the target model is small enough that the embedding table dominates
- Train for the skill, not the corpus: reasoning-dense data produces measurable capability orders of magnitude earlier in training than web text, which changes what an experiment can afford to test
- Depth over width appears to pay specifically when the data is reasoning-dense; do not transfer that shape assumption to a model trained on ordinary web text
- Deliberate negative examples are a cheap hallucination-limiting lever in synthetic generation
- Treat the benchmark claims as vendor-reported and unreplicated; the absence of a peer-reviewed paper for SYNTH means the evidence standard here is materially lower than for the Pleias-RAG paper or the SAN controlled study
- The models trade world knowledge for reasoning transparency by design - MMLU ~30% is not a defect to be fixed but the stated cost of the approach, and it is only acceptable for tasks that supply their own evidence

**Relevance**

- Primary source for the R4 candidate models in `docs/experiments/semantic-grounding-experiments.md` - Monad and Baguettotron are the knowledge-free checkpoints round 4 tests, and this article is where their training provenance, tokenizer decision and compute cost are documented
- Directly supports the R4-H22 result: Monad's SYNTH-only 8k tokenizer measured 1.196x the incumbent's fertility on our English evidence, and this article explains why that tradeoff was made and what it bought
- Tempering caveat: this is a blog post with self-reported benchmarks and no independent replication. It is adequate as provenance for a candidate model and inadequate as evidence for a design claim - the SAN paper and the Pleias-RAG paper carry that weight instead

**Tags**

- #synthetic-data
- #small-models
- #tokenizer

**Source**

- https://pleias.ai/blog/blogsynth-the-new-data-frontier

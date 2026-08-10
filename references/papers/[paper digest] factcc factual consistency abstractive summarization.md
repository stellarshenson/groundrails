**Evaluating the Factual Consistency of Abstractive Text Summarization (2020)**

The paper addresses a gap in summarization evaluation: standard metrics such as ROUGE say nothing about whether a generated summary is factually consistent with its source document. Reported rates of factual inconsistency in abstractive summaries at the time ran up to 30%. The authors propose FactCC, a weakly-supervised BERT-based classifier that reads a source document and a single summary sentence (a "claim") and predicts CONSISTENT or INCONSISTENT. The key move is training-data synthesis: rather than pay for human annotation, they sample sentences from source documents and push them through a set of rule-based transformations, some semantically invariant (back-translation paraphrase, noise injection) producing positive labels, and some semantically variant (entity swap, number swap, pronoun swap, sentence negation) producing negative labels. A second variant, FactCCX, adds span-extraction heads that point at the supporting span in the document and the offending span in the claim. Verification is framed at document-sentence granularity rather than sentence-sentence, so the model can consult the whole source. The result matters because a large synthetic corpus built from cheap rules transfers better to real summarizer errors than strongly supervised natural language inference (NLI) and fact-checking data.

**Key mechanism**

- Data generation loop: for each source document, sample one sentence as a claim, emit it as a positive; apply each invariant transformation to produce more positives; then apply each variant transformation to every positive to produce negatives. Metadata records the claim's original location and the character spans where the transformation landed
- Semantically variant transformations are typed and local: entity swap (named entity replaced by another named entity drawn from the same document, same type group), number swap (dates and numerics swapped within group), pronoun swap (within the same pronoun class to keep syntax valid), sentence negation (auxiliary verb negated or de-negated)
- Semantically invariant transformations: back-translation paraphrase via Google Cloud Translation through French, German, Chinese, Spanish, Russian; plus token-level noise injection (random duplication or deletion) applied to all examples so the classifier is robust to degenerate generation
- Model: uncased BERT-base, source document and claim concatenated as input, two-way classification off the [CLS] token with a single-layer head. FactCCX adds start/end span heads supervised by the recorded transformation and source spans
- Training: 1,003,355 generated examples from CNN/DailyMail (50.2% negative, 49.8% positive), 10 epochs, batch 12, learning rate 2e-5, 8 V100 GPUs
- Evaluation data is human-annotated by the authors, not crowdsourced: 931 development and 503 test (document, summary-sentence) pairs from ten published summarization systems. Crowdsourcing was abandoned because inter-annotator agreement was too low

**Main findings**

- FactCC reaches 74.15 class-balanced accuracy and 0.5106 F1 on the manually annotated test set, against 51.51 / 0.0882 for BERT trained on MNLI and 52.07 / 0.0857 for BERT trained on FEVER. The NLI and fact-checking baselines are barely above chance on this task
- FactCCX, which pays for span explanations, loses only about 1.3 points of accuracy (72.88, F1 0.5005) relative to FactCC
- On the Falke et al. sentence-ranking test, FactCC mis-orders 30.0% of positive/negative claim pairs against a 50.0% random baseline, beating ESIM (32.4%), BERT-MNLI (35.9%), SSE (37.3%), InferSent (41.3%), and DA (42.6%) despite being trained in the document-sentence rather than sentence-sentence setting
- The authors attribute the gap to two causes: domain mismatch between MNLI/FEVER and news text, and the fact that neural summarizer errors are specific enough not to appear in human-authored NLI corpora
- Span highlights help humans: 91.75% of annotators found article highlights at least somewhat helpful, 81.33% for claim highlights. Model spans overlap unbiased human spans at 65.33% accuracy / 0.6207 F1 (article) and 65.66% / 0.6650 F1 (claim), rising to 74.87% and 80.54% when filtered to annotations agreeing with the gold label
- Highlights improved annotation efficiency measurably: mean task time fell from 224.89s to 178.34s (21% faster) and Fleiss' kappa rose from 0.1571 to 0.2526 (38% relative)
- Stated limitations: most residual errors are commonsense mistakes, which no rule set naturally produces; and cross-sentence phenomena inside the summary (temporal inconsistency, coreference across summary sentences) are outside the document-sentence formulation
- Methodological caveat: the test set is small (503 examples) and annotated by the paper's own authors, so the headline numbers carry both sample-size and annotator-independence weaknesses
- The transformation taxonomy is deliberately shallow. The authors decline finer error types on the grounds that a finer taxonomy would assume summarization models reason like humans

**Key takeaways**

- Targeted, typed corruption of grounded text is a viable substitute for human-labelled faithfulness data at million-row scale, and it beat strongly supervised NLI transfer by more than 20 accuracy points on the target distribution
- Corruption type should be chosen from an error analysis of the actual generator being checked, not from what is easy to implement. The four variant transformations here map directly onto the observed failure modes of the summarizers of the day
- Keeping the swap within its own type group (entity for entity, number for number, possessive pronoun for possessive pronoun) is what makes the negative a minimal semantic edit rather than a syntactic anomaly the model can detect for free
- The invariant transformations matter as much as the variant ones: without paraphrase positives, the classifier can learn "surface deviation from the source implies inconsistent", which is exactly the lexical shortcut the task must avoid
- Document-level premise beats sentence-level premise for this task; sentence-sentence NLI models transfer badly even when architecture and pretraining are held constant
- Auxiliary span-extraction heads cost about one point of accuracy and buy a measurable human-workflow gain, so explanation heads are cheap when the consumer is a human reviewer
- Rule-based corruption has a known blind spot at commonsense and cross-sentence errors; a corruption-trained detector should be assumed weak there until measured
- Any evaluation on a few-hundred-row author-annotated set should be treated as directional, and a serious deployment needs a larger and independently annotated held-out set

**Relevance**

- This is the direct ancestor of the groundrails DR (dataset-refinement) track: typed swaps over grounded text (entity, number, negation) generating label-0 minimal pairs against a document premise is the same construction, with the judge-certification step (~0.97 label precision) added on top of what FactCC left as unverified rule output
- Supports the corruption-DATA reading of the campaign rather than the corruption-OBJECTIVE reading. FactCC's entire gain comes from what is in the training set; there is no pairwise or margin objective anywhere in the paper, and the pairs are consumed as independent rows by a plain two-way classifier. That is the configuration R10-H108 admitted (+0.0019 blind, finqa +0.056/+0.034) and not the one R11-H117 tried
- Contains an early signature of register displacement without naming it: the paper's own limitation section notes the model inherits exactly the error registers the transformation set encodes and stays weak on registers it does not (commonsense, cross-sentence). It offers no evidence either way on whether adding corruption registers costs competence on untouched ones, since it never measures a non-corruption register

**Tags**

- #faithfulness
- #synthetic-negatives
- #cross-encoder

**Source**

- https://arxiv.org/abs/1910.12840

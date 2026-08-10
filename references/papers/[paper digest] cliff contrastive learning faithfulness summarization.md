**CLIFF: Contrastive Learning for Improving Faithfulness and Factuality in Abstractive Summarization (2021)**

CLIFF attacks summary hallucination at training time rather than by post-hoc correction or reranking. The observation driving it is that maximum-likelihood training optimizes toward references but never tells the model that a nearly-identical incorrect summary is wrong, so the model has no learned preference between the two. The authors add a supervised contrastive term over summary representations: reference summaries and their back-translations are positives, automatically generated erroneous summaries are negatives, and both live in the same batch so the model learns to separate them in representation space. Four families of negative-sample construction are designed from a new human annotation study of 600 summaries produced by BART and PEGASUS on XSum and CNN/DailyMail: entity swap, mask-and-fill with an unconditioned BART, source-conditioned regeneration, and selecting the model's own low-confidence generations. The framework is end-to-end, adds no architecture, and is evaluated with QuestEval (question-answering-based factuality), FactCC, ROUGE-L, and human judgement. Gains are consistent but small in absolute terms, and the comparison against unlikelihood training on identical negatives is the paper's cleanest result.

**Key mechanism**

- Contrastive loss over decoder-side summary representations: for each article, positives P (reference plus a kept back-translation) and negatives N are placed in the same batch; the loss is the standard supervised-contrastive form with cosine similarity and temperature tau = 1.0. Total objective is L = L_CE + lambda * L_CL with lambda = 1.0
- Summary representation h is the decoder's last-layer output; the best variant averages over all tokens and passes through a one-hidden-layer MLP. Variants over named-entity tokens only, or the last token only, and variants without the MLP, are all worse; without the MLP the models produce more repetitive and degenerate text
- Positives are constructed by back-translation through German using NLPAug, and a translation is kept only if it introduces no new named entity
- Negative construction, four families: SWAPENT (replace a reference entity with a same-type entity from the source, imitating intrinsic errors); MASKENT / MASKREL (mask entity spans, or dependency-relation governor and dependent spans, and fill with an unfine-tuned BART, keeping only fills that introduce an entity or relation absent from both source and reference); REGENENT / REGENREL (use text before the entity or relation as a decoder prompt and let the fine-tuned summarizer continue via nucleus sampling at p = 0.7, so the negative stays grounded in the article); SYSLOWCON (keep the model's own beam outputs whose first proper-noun or number token falls below a confidence threshold tuned to maximize F1 against the error annotations)
- The confidence signal is empirical: first tokens of proper nouns and numbers inside extrinsic-error spans are generated with visibly lower probability than correct tokens, while world knowledge (correct but not in the source) is generated with high probability. Their worked example assigns 0.10 to an erroneous entity and 0.92 to a world-knowledge token
- The comparison objective is unlikelihood training on the identical negatives, which penalizes every token probability in the negative sequence rather than contrasting representations

**Main findings**

- Annotation study on 600 summaries (150 x 2 models x 2 datasets): extrinsic errors dominate, with 58.7% of BART summaries on XSum containing at least one extrinsic error and 44.0% for PEGASUS. Fleiss' kappa is 0.35 (XSum) and 0.45 (CNN/DM), which is modest agreement
- On XSum with BART, CLIFF raises QuestEval from 33.09 (cross-entropy) to 33.35 with MASKREL or SYSLOWCON, and FactCC from 23.92 to 25.73. Absolute gains are roughly 0.2-0.3 QuestEval points and 1-2 FactCC points; significance is reported at p < 0.005 by approximation randomization
- On CNN/DailyMail with PEGASUS, FactCC rises from 44.44 to 53.73 (SYSLOWCON) and QuestEval from 50.21 to 51.17, a larger FactCC movement than on XSum
- CLIFF beats unlikelihood training on the same negatives in 12 of 14 comparisons with BART and 11 with PEGASUS. Unlikelihood occasionally hurts factuality or ROUGE materially, while CLIFF's gains are more consistent
- ROUGE-L is roughly preserved or slightly improved by CLIFF, whereas the sample-filtering baseline SUBSETFT drops ROUGE-L from 37.14 to 30.35 on XSum. Informativeness is not traded away
- Entailment reranking (ENTAILRANK) inflates FactCC dramatically (23.92 to 38.45 on XSum, 44.44 to 61.04 with PEGASUS) while QuestEval stays flat or falls. Human inspection found it selects beams with peculiar high-FactCC wording without real factuality gain, a direct demonstration of metric gaming
- Metric validation: QuestEval correlates with annotated error rate at -0.43 (XSum) and -0.33 (CNN/DM); FactCC at -0.02 and -0.13; ROUGE-L at -0.13 and -0.06. The paper's own primary metric is the only one with a usable correlation, and FactCC is close to uncorrelated on XSum
- Human pairwise evaluation on 100 articles per dataset: CLIFF with SYSLOWCON wins on factuality 31.3% and loses 7.0% against cross-entropy on XSum. Krippendorff's alpha is 0.33-0.34 for informativeness and 0.62-0.89 for factuality, so the informativeness judgements are weakly agreed
- Relation-anchored construction generally beats its entity-anchored counterpart, and combining SYSLOWCON with a second strategy beats any single strategy, indicating error-type coverage rather than negative volume is what carries the effect
- Human error-correction analysis shows contrastive models substitute correct content more often, whereas unlikelihood and entailment reranking more often delete the offending content

**Key takeaways**

- Negative-sample diversity, not negative-sample count, drives the gain. The best single result comes from harvesting the model's own low-confidence generations, and combinations beat singletons
- Model confidence on the first token of a proper noun or number is a usable, nearly free detector of extrinsic error, and it separates hallucination from correct world knowledge. This is a cheap mining signal for anyone building corruption data
- A contrastive term over representations outperformed unlikelihood on identical negatives, so how a negative is consumed matters independently of what the negative is. Unlikelihood pushes down whole sequences and periodically damages fluency and factuality; the contrastive form is more stable
- Do not adjudicate faithfulness work on FactCC alone. Here FactCC correlated -0.02 with annotated error rate on XSum, and a reranking baseline drove it up 14 points without improving factuality
- Generating negatives conditioned on the source (regeneration) keeps them on-topic; unconditioned mask-and-fill drifts topic and produces negatives that are too easy, which the authors flag as weaker training signal
- Representation choice for a contrastive term is load-bearing: pooling over all tokens plus an MLP projection beat entity-only and last-token pooling, and dropping the projection caused degeneration
- Reported gains are small in absolute terms on a metric whose own correlation with human error rate is -0.43 at best, so this is a real but modest effect, not a step change
- The annotation base (600 summaries, kappa 0.35-0.45) is thin, and both the negatives and part of the evaluation depend on the same two model families, which limits how far the specific thresholds transfer

**Relevance**

- Direct precedent for the groundrails DR corruption engines: SWAPENT is the typed-swap family, MASKENT/MASKREL is span infill, and REGENENT/REGENREL is source-conditioned regeneration - the same three construction routes, applied here to build negatives for a generator rather than for a 307M cross-encoder detector
- Partially cautionary for R11-H117. CLIFF is a contrastive objective over minimal pairs and it did work, but at generator scale over decoder representations with the pair term as the whole point, not as an auxiliary hinge alongside BCE feeding a hard-min windowed serving read. The paper offers no evidence for the specific stacking H117 attempted, and its own margin-style comparator (unlikelihood) is the one that periodically damaged unrelated quality
- Closest thing in the set to an observed register-displacement analogue: the paper's SUBSETFT baseline gives up 7 ROUGE-L points to gain factuality, and unlikelihood training "occasionally hurts factuality or ROUGE scores significantly" on the same negatives that helped under a contrastive term. That is displacement caused by objective choice on fixed data, which is the shape of the H117 failure

**Tags**

- #faithfulness
- #contrastive-learning
- #synthetic-negatives

**Source**

- https://arxiv.org/abs/2109.09209

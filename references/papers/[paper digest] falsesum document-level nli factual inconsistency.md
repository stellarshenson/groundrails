**Falsesum: Generating Document-level NLI Examples for Recognizing Factual Inconsistency in Summarization (2022)**

Falsesum builds document-level natural language inference (NLI) training data specifically for detecting factually inconsistent summaries. The starting problem is that off-the-shelf NLI models transfer badly to this task: their premises are single sentences of about 22 words while summarization premises are documents of about 546 words, and the entailment phenomena in curated NLI corpora do not resemble the errors neural summarizers actually make. The prior best synthetic remedy, FactCC, applies fixed rules (entity swap, negation) and is shown by later work to produce a narrow error distribution that models exploit through lexical overlap. Falsesum replaces the rules with a controllable text generator: a T5-base model is trained on a proxy task - reconstruct a gold summary from a shuffled, corrupted bag of predicate and argument spans plus a partially masked summary - and at inference the gold spans are withheld so the model is forced to fill the mask with wrong material. A control code selects intrinsic (misconsolidated source information) or extrinsic (invented information) error. The resulting document/summary pairs are contrastive: the positive and negative differ only in factual consistency, not in style or length. Augmenting MNLI with 100k Falsesum examples sets a new state of the art across four factual-consistency benchmarks.

**Key mechanism**

- Inputs are a source document D and its gold summary S+ from CNN/DailyMail. PredPatt open information extraction pulls (arg0, pred, arg1, ...) tuples from both. Each tuple is deliberately corrupted by dropping one randomly chosen argument, and the dependency-root word of each span is lemmatized so the generator must re-inflect when inserting
- Serialized input template: `Predicates:P; Arguments:A; Code:c; Summary:M`, where P and A are shuffled span lists, c is `intrinsic` or `extrinsic` chosen with probability 0.5, and M is the gold summary with selected predicate and argument spans replaced by `<span_i>` placeholders (i = 0 reserved for the predicate)
- Training objective is reconstruction: given the spans and the masked summary, output the original gold summary. This needs no examples of inconsistent summaries at all - supervision comes entirely from an existing summarization corpus
- The train/test asymmetry is the whole trick. During training under the `intrinsic` code the gold spans are visible in the lists; at generation time (and during `extrinsic` training) they are removed, so the model must substitute other document spans (intrinsic error) or invent new material (extrinsic error)
- Span reduction: adjectives and adverbs are randomly dropped from 10% of gold spans so the generator learns to hallucinate fine-grained modifiers rather than only whole-argument substitutions
- Generator is T5-base fine-tuned on 394,774 formatted training instances; negatives are produced by running it over a 262,692-instance test split. Each (D, S+) yields one entailment and one non-entailment NLI pair. Downstream models are RoBERTa-base fine-tuned on MNLI (binary: neutral and contradiction merged into non-entailment) at 512 max tokens, augmented with 100,000 sampled Falsesum examples

**Main findings**

- Overall score across four benchmarks (FactCC, Ranksum, QAGS, SummEval): MNLI-512 alone 51.43, plus ANLI 53.19, plus DocNLI 55.35, plus FactCC 69.02, plus Falsesum 74.17. Falsesum beats the FactCC augmentation by 5.15 points overall using the same underlying corpus (CNN/DailyMail)
- Per benchmark, Falsesum versus FactCC augmentation: FactCC test 83.52 vs 73.87, Ranksum 72.90 vs 67.29, QAGS 75.05 vs 73.50, SummEval 65.18 vs 60.04. Majority voting scores 50.11 overall
- Length mismatch is not the primary problem. ANLI and DocNLI have long premises and still score 53.19 and 55.35; simply raising MNLI's max token length from 128 to 512 makes performance worse (57.06 to 51.43)
- The sentence-wise decomposition baseline [split-doc] MNLI-128 reaches 66.63 but multiplies inference cost by the number of sentences, so it is not free
- Ablations: removing contrastive pairing costs 1.06 points overall, removing extrinsic negatives costs 2.22, removing intrinsic negatives costs 5.03. Intrinsic errors matter most because they dominate CNN/DailyMail-derived benchmarks
- Lexical-overlap analysis: partitioning the FactCC test set into five overlap bins (overlap = normalized coverage x density), the FactCC-augmented model collapses to near 50% balanced accuracy on the highest-overlap bin (0.9) by predicting almost everything consistent, while Falsesum holds a large margin. The synthetic-data source determines whether the detector learns "high overlap implies consistent"
- Manual verification of 200 generated negatives: the intended non-entailment label is correct for 86% of intrinsic and 81% of extrinsic outputs, the error is inserted at the specified span 94-95% of the time, but the requested error type is honoured only 65% of the time for extrinsic (the generator prefers copying over inventing). So roughly 14-19% label noise is present, and RoBERTa-base is nonetheless robust to it
- Hypothesis-only probe (premise withheld) accuracy: 82.15 on FactCC data, 78.46 on DocNLI, 69.38 on Falsesum, with majority voting at 50.00. Lower is better; Falsesum negatives are the hardest to identify from surface features alone, meaning fewer exploitable artifacts
- Scope caveats: all data derives from a single English news corpus, the generator's quality is bounded by open information extraction quality, and downstream results use one model family (RoBERTa-base) at one size

**Key takeaways**

- A learned, controllable corrupter produces harder and less artifact-laden negatives than a fixed rule set drawing on the same corpus. The 5-point overall gain over FactCC augmentation with everything else held constant is the cleanest available evidence for that claim
- Run a hypothesis-only (claim-only) probe on any corruption dataset before training on it. It is cheap, it directly measures how much of the label is readable off the negative's surface, and it separated the three datasets here by 13 accuracy points
- Contrastive pairing of positive and negative over the same document is worth about 1 point on its own, well below the value of covering both error types. Error-type coverage, not pairing, is the dominant term in this ablation
- Balance error types against the target distribution rather than uniformly. Intrinsic negatives were worth more than twice extrinsic here purely because the evaluation corpora skew intrinsic
- Roughly 15% label noise in generated negatives did not prevent a state-of-the-art result, which sets a useful expectation: aggressive certification of negatives buys precision, but perfect precision is not a precondition for a strong detector
- Long-premise NLI data is not interchangeable with task-matched data. Adding ANLI or DocNLI barely helped and sometimes hurt, so premise length is a red herring relative to error-phenomenon match
- Guard explicitly against the lexical-overlap shortcut; a detector that scores well overall can still be at chance on the highly extractive subset, which in a retrieval-augmented setting is exactly where verbatim-quoted claims live
- Results rest on one corpus, one language, and one encoder size; the pipeline generalizes in principle but the specific thresholds and ratios should be re-measured per domain

**Relevance**

- The closest published match to the groundrails DR generation stack: span infill with typed control codes, corruption placed at a designated masked span over the same evidence chunk, contrastive positive/negative from one source document. Falsesum's control-code approach is the precedent for the DR engines' typed delta taxonomy (entity-swap, number-change, omission, negation, hedge-deletion)
- Reinforces the corruption-DATA reading over the corruption-OBJECTIVE reading. Every gain here is obtained with an ordinary pointwise cross-entropy classifier over shuffled rows; the contrastive pairing ablation is worth only 1.06 points, and no margin or ranking term appears anywhere. That is consistent with R10-H108 (corruption data admitted, +0.0019 blind) and with R11-H117 (pairwise margin refuted at 0.69186 against the 0.7031 bar)
- Directly anticipates a register-displacement risk relevant to the groundrails windowed read: the FactCC-augmented model in Figure 3 is near chance on the highest-overlap bin while performing acceptably overall. The DR audit's finding that 29.6% of seeds are verbatim substrings of their chunk puts groundrails in the same bin structure, and Falsesum's remedy is negative diversity in the data, not a change of objective

**Tags**

- #faithfulness
- #synthetic-negatives
- #nli

**Source**

- https://arxiv.org/abs/2205.06009

**Learning What Makes a Difference from Counterfactual Examples and Gradient Supervision (2020)**

This paper argues that pairing information between counterfactual training examples is a supervisory signal distinct from the labels themselves, and that shuffling examples during stochastic training throws it away. Given a pair of minimally-different, differently-labelled examples, the vector between them in input space indicates the direction that maps a change in input to a change in label. The authors turn that into an auxiliary loss - gradient supervision (GS) - which pushes the network's input gradient at each point to align, in cosine terms, with the vector pointing to its counterfactual partner. Where standard training fits a curve to individual points, GS additionally orients the curve. The method is applied to four tasks known for poor out-of-distribution behaviour: visual question answering on VQA-CP, multi-label classification on COCO, sentiment analysis on IMDb, and natural language inference on SNLI, with the two NLP datasets taken directly from Kaushik et al.'s counterfactually-augmented data. In every case GS improves over simply mixing the counterfactual examples in as extra data, most visibly on out-of-distribution and transfer test sets. The paper is deliberately broad rather than deep: small models, modest data, and an explicit statement that scaling to large pretrained models is future work.

**Key mechanism**

- For a network f with parameters theta, g_i = grad_x f(x_i) is the gradient of the output with respect to the input at training point x_i. The GS loss is a cosine distance, L_GS(g_i, ghat_i) = 1 - (g_i . ghat_i) / (||g_i|| ||ghat_i||)
- The "ground truth" gradient ghat_i for a counterfactual pair {(x_i, y_i), (x_j, y_j)} is simply the difference vector x_j - x_i - the translation in input space that should flip the network output from y_i to y_j
- Total objective is L = L_Main + lambda * L_GS. Optimization requires second-order backpropagation; the overhead is two extra backward passes through the model per mini-batch
- Justification is a first-order Taylor argument: for a genuinely minimal pair, terms beyond first order effectively vanish, so the output distance |f(x_j) - f(x_i)| is maximized exactly when the dot product grad_x f(x_i) . (x_i - x_j) is maximized. GS therefore shapes the decision boundary between classes rather than only fitting points
- Input space means a feature representation (pretrained CNN features or a text encoder output), not raw pixels or tokens. For multiclass outputs the loss is applied only to the gradient of the ground-truth class, and for softmax outputs it is taken on the logits so the derivative depends on one class alone
- Pairs come from two sources: datasets that supply them explicitly (Kaushik et al.'s edited IMDb and SNLI), and datasets where they can be manufactured from existing annotation by masking - VQA-CP images with human-attention regions masked and the answer set emptied, COCO images with objects removed by an inpainter GAN and the label vector edited accordingly

**Main findings**

- Sentiment (IMDb with counterfactuals): baseline without edited training data scores 82.6 on the original test set and 55.3 on the edited test set. Adding edited data as plain augmentation gives 82.0 / 88.7. Adding GS on top gives 83.8 / 91.2, so GS improves both halves over augmentation alone
- Sentiment zero-shot transfer with GS versus augmentation: Amazon 81.6 vs 80.8, Twitter 65.4 vs 63.1, Yelp 88.8 vs 87.4. Gains are consistent but on the order of 1-2 points
- NLI (SNLI with counterfactuals): baseline without edited data 42.0 original / 59.0 edited; with edited data as augmentation 39.1 / 57.8 - augmentation alone made it worse; with GS 44.4 / 61.2. On MultiNLI dev with no fine-tuning, augmentation drops to 42.4 from a 46.0 baseline while GS recovers to 46.8
- The NLI result is the paper's own recorded case of counterfactual data hurting: the authors attribute it to edited sentences being "unnatural" so that easy language cues stop transferring, and GS is what restores and slightly exceeds the baseline
- COCO multi-label classification: 71.8 mAP without edited images, 72.1 with them as augmentation, 72.9 with GS. On edited images 58.1 / 64.0 / 65.2, and on "hard edited" images with class combinations never seen in training 54.8 / 56.0 / 57.7
- Ablation with random pairwise relations instead of true counterfactual relations: on COCO it reverts to baseline (71.8 / 63.9 / 56.1), and on sentiment it collapses to chance (50.8 validation, 49.2 original, 52.0 edited). The signal is in the pairing, not in constraining the gradient per se
- VQA-CP v2: strong baseline plus counterfactual data 46.0 on test and 44.2 on the authors' "focused" test set; adding GS gives 46.8 and 46.2. The focused set (only human-attended image regions retained) shows the effect more clearly than the official test set
- The paper flags that most VQA-CP methods exploit built-in knowledge of the benchmark's construction and that many use the test split for model selection, which it calls an unsanitary practice; its own method does not use that knowledge
- Data-efficiency curve on sentiment shows GS above plain augmentation, which is above original-only, across fractions from 1/9 to 1/1 of the data
- Stated limitations: NLP experiments use simple models and relatively little data, well below the transformer state of the art of the time, and application to the large-data regime is left as future work. Gains are typically 1-2 points and no confidence intervals or repeated seeds are reported

**Key takeaways**

- The pairing between counterfactual examples carries information beyond the two labels, and consuming it via an auxiliary gradient term beat consuming it as extra shuffled rows on all four tasks
- The random-relation ablation is the decisive control and any similar work should replicate it. Without it, a gain from a pairwise auxiliary term cannot be distinguished from generic gradient regularization
- Counterfactual data added as plain augmentation is not guaranteed to help. On SNLI it cost 2.9 points on the original test set and 3.6 on MultiNLI transfer; the objective term is what turned the same data positive
- Counterfactual pairs can be manufactured from annotations already present in a dataset - attention maps, bounding boxes - without new human labelling, which materially changes the cost calculation for pair construction
- Cost is roughly two extra backward passes per batch plus second-order gradient support, so the method is cheap in engineering terms but not free in compute
- The mechanism depends on the pair being genuinely minimal: the Taylor argument that licenses the local linearization only holds when the two points are close in representation space
- Every reported gain is 1-2 points on small models with no seed variance reported, so the evidence supports "the direction is real" rather than a reliable effect size
- The method is untested at the scale where it would matter for a modern encoder; the authors say so explicitly

**Relevance**

- Identity note - the citation in R11-H117 reads "counterfactually-paired training (Kaushik et al., ICLR 2020; Teney et al., 2020)", and this paper is the correct referent. It is the direct follow-on to Kaushik et al., trains on that paper's exact IMDb and SNLI counterfactual sets, and its whole contribution is an auxiliary pairwise objective over minimal pairs. The role matches
- This is the single closest published precedent for the H117 construction - an auxiliary pairwise term over co-batched minimal pairs stacked on top of an unchanged primary loss - and it is a positive result. That makes the H117 refutation (pair 0.69186 against the 0.7031 bar) a scale-and-surface mismatch rather than a contradiction: this paper's evidence base is small models on 1.7k-6.6k pairs with no repeated seeds, and the authors explicitly disclaim the large-model regime that a 307M mmBERT cross-encoder occupies
- Two of its findings bear on register displacement. It records counterfactual data as plain augmentation actively degrading an untouched register (SNLI original -2.9, MultiNLI transfer -3.6), which is the displacement pattern; and its random-relation ablation shows a pairwise auxiliary term with a wrong pairing signal drives sentiment accuracy to chance, meaning the objective family is capable of destroying competence, not merely failing to add any. Neither is a prediction of the specific hard-min windowed-read interaction, which remains unaddressed by all five papers

**Tags**

- #counterfactual-data
- #gradient-supervision
- #out-of-distribution

**Source**

- https://arxiv.org/abs/2004.09034

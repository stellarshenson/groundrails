**Learning the Difference that Makes a Difference with Counterfactually-Augmented Data (2020)**

The paper gives an operational definition of "spurious pattern" by borrowing from causality: an association is spurious when intervening on the feature would not change whether the label applies. Since no automated tool maps text to disentangled causal factors, the authors run the intervention through humans. Crowd workers are given a document and its label and asked to revise the text so that a target counterfactual label applies, while keeping the document coherent and making no gratuitous changes - a least-action principle that keeps the edit minimal. Applied to sentiment analysis (IMDb) and natural language inference (SNLI), this yields two paired datasets where each original example has a minimally-edited counterpart carrying a different label. The headline empirical result is a sharp asymmetry: classifiers trained on original data collapse on revised data and vice versa, while classifiers trained on the union hold up on both at a small cost relative to same-size in-distribution training. Linear-model feature inspection shows why - words like "romantic" or "horror", predictive in either dataset alone, cease to be predictive once the two are combined, because humans never touch them when flipping the label. The datasets are small (1.7k IMDb revisions, 6.6k SNLI revised pairs) and the paper is explicitly exploratory.

**Key mechanism**

- Human-in-the-loop counterfactual revision on Amazon Mechanical Turk, restricted to U.S. workers with 500+ completed HITs and 97%+ approval; 713 workers participated, 518 contributed accepted edits. Total cost \$10,778.14 at \$0.65 per revision and \$0.15 per verification, roughly 5 minutes per review and 4 minutes per NLI sentence
- Three instructions define the intervention: the counterfactual label must apply, the document must stay coherent, and no unnecessary modifications are permitted. The third constraint is what makes the pair minimal and therefore informative about which spans are causally load-bearing
- Sentiment: 2.5k IMDb reviews (longest 20% filtered out, 50:50 class balance), each shown to two workers, split 1707/245/488 train/validation/test. About 2% of revisions were rejected on manual inspection; one revision per original was sampled at random for the final set
- NLI: 1750/250/500 SNLI pairs sampled with balanced classes. Two separate tasks - Revised Hypothesis (RH), edit the hypothesis with the premise fixed, and Revised Premise (RP), edit the premise with the hypothesis fixed - each collecting edits toward both counterfactual classes. A second worker pool verified labels by three-way majority vote; about 9% of data was discarded. Final revised set is 3,332 train pairs per condition, 6,664 combined
- Inter-editor agreement is measured as Jaccard similarity over binary edit-position vectors: 25.5% combined (19.3% replacement, 14.3% insertion), falling with review length from 41.6% on 0-50 token reviews to 16.2% on 301-329 token reviews. Different humans intervene in different places
- Models are ordinary supervised classifiers: SVM and Naive Bayes over TF-IDF, Bi-LSTM (50-dim embeddings, hidden 50, recurrent dropout 0.5, global max-pool), ELMo-LSTM, and fine-tuned uncased BERT-base. No new objective is introduced - the paired data is consumed as ordinary shuffled rows

**Main findings**

- Sentiment, train on 1.7k original: SVM scores 80.0 on original but 51.0 on revised (chance); Bi-LSTM 79.3 to 55.7; ELMo 81.9 to 66.7; BERT 87.4 to 82.2. The reverse direction is equally sharp - trained on 1.7k revised, SVM scores 91.2 revised but 58.3 original, Bi-LSTM 89.1 vs 62.5
- Training on the 3.4k combined set gives 83.7/87.3 (SVM), 81.5/92.0 (Bi-LSTM), 88.5/95.1 (BERT) on original/revised, within roughly 3 points of models trained on the same volume of purely original data when measured in-distribution. The price of breaking the spurious reliance is small
- BERT is markedly more resilient to the drop than the smaller models, which the authors attribute to broader pretraining exposure where the spurious patterns did not hold
- Removing the edited spans from training reviews leaves the remainder still predictive of the true label: SVM 57.8, Naive Bayes 59.1, Bi-LSTM 60.2 - but BERT falls below chance (49.2). The non-causal residue carries real signal for bag-of-words models
- Out-of-domain zero-shot transfer favours the combined set at matched size (3.4k): Amazon reviews 85.1 vs 80.0 (BERT), Twitter/SemEval 82.9 vs 79.3, Yelp 89.4 vs 85.3. Similar direction for every model class
- NLI with BERT: trained on 1.67k original, 72.2 on SNLI but 39.7 on RP. Trained on RP, 66.3 on RP but 50.6 on SNLI. Trained on RH, 67.0 on RH and 71.9 on SNLI but only 47.4 on RP. Combining original with RP and RH (8.3k) gives 73.5 / 64.6 / 69.6 / 67.1, beating the same-size all-original model (77.8 / 44.6 / 66.1 / 55.4) everywhere except in-distribution SNLI
- Even BERT fine-tuned on the full 500k SNLI set scores only 54.3 on RP, so scale on the original distribution does not repair the failure
- Hypothesis-only probe: a Bi-LSTM reading hypotheses alone reaches 69.0 on SNLI test when trained on 500k SNLI, but falls to 15.4 on RP. Trained on the combined original+RP+RH set it reaches only 44.0 on SNLI and 34.5 on the combined test - near the 34.6 majority baseline. The combined design forces the model to read the premise
- The hypersensitivity check: models can partly tell original from revised text (BERT 77.3 on IMDb) but on NLI perform within about 3 points of the 66.7 majority baseline, so domain-splitting behaviour is limited but not absent on sentiment

**Key takeaways**

- Minimal-pair supervision is a data intervention, not an objective change. Every result here comes from mixing counterfactual examples into an ordinary training set; the pairing is never exploited by the loss
- Training on either distribution alone produces a classifier that is at or near chance on the other. This is the clearest available demonstration that a corruption distribution defines a register, and a model trained only inside it does not transfer out
- The combined set costs roughly 3 points in-distribution relative to same-volume original data and buys large gains out-of-distribution, so budget a small in-distribution regression when adding counterfactual data
- Removing the edited spans still leaves a bag-of-words model above chance, meaning the un-edited remainder is not label-neutral. Any evaluation built by editing spans should check what a model can score from the un-edited residue alone
- Run a claim-only or hypothesis-only probe on any paired dataset. It exposed the SNLI artifact at 69% and confirmed that the combined data removed it
- Also run the inverse check - train a classifier to distinguish original from revised examples. If it succeeds, the model may be treating the corruption channel as a domain rather than learning the semantics, which is a distinct failure from the one being fixed
- Larger pretrained models degrade less but do not escape the problem; 500k in-distribution SNLI examples still left BERT at 54.3 on revised premises
- Sample sizes are small (1.7k and 6.6k), one dataset per task, and pre-transformer baselines dominate the model list, so treat effect magnitudes as indicative rather than precise

**Relevance**

- This is the origin of the minimal-pair construction the groundrails DR track uses, with humans doing what the DR engines automate: edit a grounded text minimally so the label flips, leave everything else alone. The DR judge-certification step is the automated stand-in for this paper's three-annotator verification vote, which discarded about 9% of NLI data
- The strongest single prediction of the observed H117 failure mode among the five papers, and it is a data-side warning rather than an objective-side one: train-on-original / test-on-revised falls to chance in both directions, so a corruption register is a distribution the model can occupy at the expense of others. The campaign's own probe result matches the shape - pair-accuracy improved sharply (+0.0950) while ragtruth_en regressed (-0.0137), which is pair-local competence bought against an untouched register
- Also supplies the two diagnostics groundrails could apply to the DR lane directly: the claim-only probe (against the recorded 29.6% verbatim-seed share) and the original-versus-revised discriminability check, which tests whether the lane is being learned as a domain rather than as semantics - and note that the DANN lane tag in the trainer is already an explicit domain-adversarial guard against exactly that

**Tags**

- #counterfactual-data
- #spurious-correlations
- #nli

**Source**

- https://arxiv.org/abs/1909.12434

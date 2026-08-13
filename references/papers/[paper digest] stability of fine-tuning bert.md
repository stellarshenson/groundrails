**On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines (2020)**

Mosbach, Andriushchenko and Klakow study why fine-tuning BERT on small downstream datasets is unstable across random seeds, a failure mode first documented by Devlin et al. for BERT-large. Working on CoLA, MRPC, RTE and QNLI, they run 25-seed sweeps and decompose the instability into two separate phenomena: optimization difficulties early in training, characterized by vanishing gradients and degenerate runs that fail to beat a trivial baseline, and generalization variance late in training, where runs with almost identical training loss end at very different development accuracies. They test and reject the two common explanations - catastrophic forgetting and small dataset size - and trace the optimization failures to the interaction of learning-rate warmup with the omitted bias-correction step in the BERTADAM variant shipped with the original code. Their proposed baseline combines bias-corrected Adam, a smaller learning rate (2e-5), and more epochs, and produces tightly concentrated performance over 25 seeds, clearly outperforming the Mixout regularizer of Lee et al. (2020) on stability. The findings replicate on RoBERTa and ALBERT.

**Key mechanism**

- Failed fine-tuning runs show vanishing gradients early in training and collapse to a trivial training loss plateau, an optimization failure rather than overfitting
- The original BERT fine-tuning recipe uses Adam without bias correction, which interacts with linear warmup to produce an effectively mis-scaled learning rate in the first steps; the paper demonstrates the two corrections are complementary, not interchangeable
- The remedy is bias-corrected Adam with warmup, a smaller learning rate (2e-5 instead of the larger defaults that survived hyperparameter search), and training longer than the default 3 epochs
- Catastrophic forgetting is ruled out by probing language-modeling perplexity of fine-tuned checkpoints: failed and successful runs forget the pretraining distribution to the same degree
- Dataset size is ruled out by downsampling experiments: instability tracks the optimization recipe, not the sample count

**Main findings**

- With the default Devlin recipe (3 epochs, no bias correction), a substantial fraction of seeds on RTE and MRPC degenerate to majority-class performance; with bias correction plus warmup these failures largely disappear
- Bias correction matters most within short 3-epoch budgets; simply training longer is an alternative fix because the early mis-scaled updates are eventually undone
- Even after optimization failures are eliminated, runs with near-identical training loss show a large spread in development accuracy - the residual variance is a generalization phenomenon, not an optimization one
- The stabilized recipe yields very concentrated development performance over 25 seeds on BERT and transfers to RoBERTa and ALBERT
- The paper's recipe beats Mixout (Lee et al. 2020) on stability, and Mixout does not remove the vanishing-gradient failures

**Key takeaways**

- Always run bias-corrected Adam (standard AdamW) with warmup for transformer fine-tuning; the historical BERTADAM omission is a live trap in older codebases
- Prefer the smaller end of the standard learning-rate range (2e-5 for BERT-base) and a longer schedule when seed stability matters
- Expect a residual seed-variance component that no optimizer hygiene removes: it lives in generalization, so it must be attacked with regularization, averaging, or ensembling rather than better optimization
- Multi-seed reporting over at least a handful of seeds is mandatory for any fine-tuning claim on small datasets
- Optimization failure and generalization variance are different problems and need different instruments to diagnose

**Relevance**

- Directly load-bearing for R18-H154 thread (i): our stack (mmBERT 307M, AdamW with bias correction, 10% OneCycle warmup, LR 1e-5, 14,300 steps) already implements the Mosbach baseline, so the observed 0.0243 two-seed arena spread sits in the paper's residual "late generalization variance" class - the class Mosbach explicitly leaves unsolved
- Justifies diagnosing our spread as generalization variance rather than optimization failure, which steers the lever search toward flat-basin and averaging methods rather than warmup/LR fixes

**Tags**

- #fine-tuning-stability
- #optimization
- #bert

**Source**

- https://arxiv.org/abs/2006.04884

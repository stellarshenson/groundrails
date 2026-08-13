**Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping (2020)**

Dodge, Ilharco, Schwartz, Farhadi, Hajishirzi and Smith run the largest controlled study of fine-tuning variance to that date: 2,100 fine-tuning episodes of BERT-large-uncased (340M parameters) on four GLUE tasks (MRPC, RTE, CoLA, SST), varying only the two factors a random seed controls - the weight initialization of the final classification layer (2,048 parameters, sampled from N(0, 0.02)) and the training data order from shuffling. The design is factorial: for each small dataset they run all 25 x 25 combinations of weight-init seeds and data-order seeds (225 episodes for the larger SST), isolating each factor's contribution. Both factors contribute comparably to out-of-sample variance, and some inits or orders are consistently better than others within a dataset, yet an init that is good on one dataset can be bad on another - so seed quality does not transfer across tasks. They advocate reporting expected validation performance as a function of the number of trials, and show a simple early-stopping rule (stop a trial when validation performance plateaus) recovers most of the benefit of a large restart budget at a fraction of the compute.

**Key mechanism**

- Factorial seed design: N distinct weight-init seeds crossed with N distinct data-order seeds (N=25 for MRPC/RTE/CoLA, N=15 for SST), same hyperparameters throughout (3 epochs, batch 16, LR 2e-5, dropout 0.1)
- Variance decomposition attributes each episode's outcome to its weight-init component, its data-order component, and their interaction
- Expected validation performance: the expected best validation score after k trials, computed from the empirical distribution of runs - a principled way to compare models given a restart budget
- Early stopping per trial: halt training when the validation metric stops improving, reallocating compute to fresh restarts

**Main findings**

- Both weight initialization and data order contribute comparably to the variance of out-of-sample performance; neither dominates
- Variance is largest on the smallest datasets (RTE 2.5k, MRPC 3.7k, CoLA 8.6k training samples) and smaller but non-trivial on SST (67k)
- Some weight inits and some data orders are consistently better than others within a dataset, but the ranking does not transfer across datasets
- Expected validation performance keeps increasing with the number of trials up to the largest budget tested (25), so more restarts always buy expected improvement
- The early-stopping algorithm matches the performance of the full-compute protocol while spending substantially less, and allocating saved compute to more trials improves expected performance further
- All 2,100 episodes were released publicly to support follow-up analysis of fine-tuning dynamics

**Key takeaways**

- A seed is two independent levers, not one: init and order contribute separately, so variance studies and variance-reduction methods should say which lever they address
- Init-paired comparisons (same init, different order) isolate the data-order component and sharpen attribution of any observed delta
- Do not chase "lucky seeds" - seed quality does not transfer across datasets, so selection on one evaluation does not generalize to another
- Expected validation performance over a trial budget is a better reporting statistic than the single best run
- Early stopping plus restarts dominates longer single runs at fixed compute when variance is high

**Relevance**

- Directly load-bearing for R18-H154 thread (i): the campaign's twin draws differ in BOTH init and data order (init/perm fingerprints banked per doctrine), so the measured 0.0243 two-seed spread conflates the two components Dodge decomposes - an init-paired diagnostic arm (the H126 seeding facility makes this free to construct) would measure how much of the spread is init vs order
- The early-stopping finding argues for checkpoint selection on the in-domain suite (gold_full) rather than serving the terminal iterate, within the campaign's contamination wall

**Tags**

- #seed-variance
- #fine-tuning
- #experimental-methodology

**Source**

- https://arxiv.org/abs/2002.06305

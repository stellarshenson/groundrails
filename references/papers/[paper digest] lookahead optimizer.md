**Lookahead Optimizer: k steps forward, 1 step back (2019)**

Zhang, Lucas, Ba and Hinton propose Lookahead, an optimizer wrapper that is orthogonal to the choice of inner optimizer. Lookahead maintains two weight sets: fast weights, updated k times by any standard optimizer (SGD, Adam), and slow weights, updated once per cycle by linearly interpolating from the current slow weights toward the final fast weights, after which the fast weights are reset to the slow weights. The slow-weight trajectory is a smoothed, lower-variance path through weight space. The paper proves a variance-reduction result on a noisy quadratic model and a deterministic-quadratic convergence result, and shows empirically that Lookahead improves both convergence speed and final performance for SGD and Adam with their default hyperparameters on CIFAR-10/100, ImageNet, Penn Treebank LSTM language modeling, and WMT-14 English-German Transformer machine translation. The defaults k = 5 and slow step size alpha = 0.5 are used throughout and the method is robust to both. Published at NeurIPS 2019. Overhead is one extra weight copy and one interpolation every k steps - computationally negligible.

**Key mechanism**

- Inner loop: k fast-weight updates with the base optimizer on successive minibatches
- Outer step: slow <- slow + alpha * (fast_final - slow), then fast <- slow (resynchronization)
- Defaults: k = 5 synchronization period, alpha = 0.5 slow step size; performance is robust to both
- On a noisy quadratic model, Lookahead provably reduces iterate variance relative to the inner optimizer; on deterministic quadratics it retains linear convergence
- The slow weights are the deployed model; the fast weights explore and are discarded each cycle

**Main findings**

- Lookahead improves SGD and Adam even with their default hyperparameters on ImageNet and CIFAR-10/100 - faster convergence and better final accuracy
- LSTM language models on Penn Treebank and Transformer NMT on WMT-14 En-De also improve, so the effect is not vision-specific
- The method is robust to the inner optimizer choice, to k, and to alpha across wide ranges - the paper's selling point is reduced hyperparameter sensitivity
- Variance reduction is demonstrated directly on the noisy quadratic model: Lookahead iterates concentrate where SGD iterates diffuse
- Compatible with standard LR schedules (the schedule runs inside the inner loop on the fast weights)
- Cost: one extra full-parameter buffer and a cheap interpolation every k steps; wall-clock overhead negligible

**Key takeaways**

- A near-free variance-reduction wrapper around AdamW: one weight buffer, one interpolation every 5 steps, no gradient computations added
- The slow weights are a trajectory-smoothed endpoint - mechanistically adjacent to EMA but with a hard resync (fast weights restart from slow each cycle), which prevents the lag-under-anneal failure mode of pure trailing averages
- Because the inner schedule (e.g. OneCycle) runs on the fast weights and the slow weights resync every k steps, Lookahead composes with anneal-to-zero schedules where plain EMA lags
- Hyperparameter robustness is the pitch: k = 5, alpha = 0.5 out of the box
- Evidence base covers CNNs, LSTMs and Transformers (NMT), but not BERT-class encoder fine-tuning - transfer is plausible, not measured

**Relevance**

- R18-H154 thread (iii) as the budget smoothing lever: Lookahead is the cheapest trajectory-smoothing method surveyed - no extra backprops (unlike SAM/SMART), no schedule replacement (unlike schedule-free), and the resync mechanic directly addresses the H120 instrument's objection to within-run averaging under OneCycle (0.9378 coherent-descent cosine: a lagged EMA is under-trained, but Lookahead's slow weights are repeatedly re-anchored to the live trajectory)
- Overlaps H152's EMA 0.999 in mechanism (both smooth the trajectory), so the memo ranks it as an EMA alternative or successor, not a stack-mate

**Tags**

- #optimization
- #variance-reduction
- #optimizer-wrapper

**Source**

- https://arxiv.org/abs/1907.08610

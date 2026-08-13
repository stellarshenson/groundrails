**Git Re-Basin: Merging Models modulo Permutation Symmetries (2022)**

Ainsworth, Hayase and Srinivasa attack the obstacle that blocks naive weight averaging of independently trained networks: two networks with different random initializations land in differently-permuted regions of weight space, so their direct average is garbage even when their functions are similar. Because hidden units are symmetric under permutation, any network has factorially many weight-space twins with identical function; Re-Basin finds the permutation of one model's units that best aligns it with a reference model, then merges. They propose three matching algorithms - activation matching (align units by their activation statistics on data), weight matching (align by weight correlation), and a straight-through estimator that directly optimizes for a good merged model - all reducible to linear assignment problems solvable in seconds. The headline result is the first demonstration of zero-barrier linear mode connectivity between two independently trained ResNets on ImageNet modulo permutations: after alignment, the straight line between the two weight vectors never leaves the low-loss region, so merging is safe. The paper also presents evidence that linear mode connectivity is an emergent property of training procedures rather than of model architectures, and documents where the approach still fails.

**Key mechanism**

- Permutation symmetry: permuting the hidden units of a layer (with the matching inverse permutation at the next layer) leaves the function unchanged; alignment searches this symmetry group
- Activation matching: for each layer, match units of model B to model A by correlating their activations on a data sample; solve per-layer linear assignment (Hungarian algorithm)
- Weight matching: same assignment problem but with weight-vector correlation as the cost - no data needed
- Straight-through estimator: optimize the permutation directly for end-task loss of the interpolated/merged model, using a differentiable relaxation
- After alignment, models can be averaged or linearly interpolated without crossing a loss barrier

**Main findings**

- First zero-barrier linear mode connectivity between independently trained ResNet models on ImageNet modulo permutation - the barrier falls to zero for sufficiently wide ResNets
- Alignment quality increases with network width and degrades for narrow or very deep architectures; MLPs and narrow CNNs retain residual barriers
- The fastest matching method finds permutations in seconds on current hardware; data-free weight matching is competitive with activation matching in several settings
- Linear mode connectivity emerges from properties of the training procedure (SGD noise, width) rather than being inherent to the architecture
- Table 1 quantifies the symmetry group sizes: the number of permutation twins dwarfs any feasible search, so greedy per-layer assignment is the practical route
- Limitations are explicit: the linear mode connectivity hypothesis has counterexamples, and alignment is not guaranteed across very different training regimes

**Key takeaways**

- Cross-init merging is possible but is an alignment problem first: never average independently initialized networks directly - H118's kill is the predictable outcome, not bad luck
- Alignment costs are modest in compute (seconds for matching, plus activation statistics over a data sample) but carry real failure risk on narrow/deep stacks; transformers are not the paper's strong suit
- Width is the friend of merging: wider layers align better; a 307M encoder is in a more favorable regime than a narrow CNN but evidence is vision-heavy
- If two models differ in init AND the training mix, alignment answers only the init part - data divergence is not a permutation
- Treat align-then-merge as a research project with a measurable gate (interpolation barrier on the in-domain suite), not an engineering lever

**Relevance**

- R18-H154 thread (v) cross-init caveat, as ordered: our twin draws differ in init, so any seed-soup requires Re-Basin-class alignment; the memo's recommendation is to gate it on an interpolation-barrier measurement between the two banked twin checkpoints (CPU/GPU0, near-free) before any training-side registration
- Also explains WHY the output-space ensemble of the same two draws works (0.72067 committee) while the weight-space soup failed: function-space composition does not need weight alignment

**Tags**

- #model-merging
- #permutation-symmetry
- #mode-connectivity

**Source**

- https://arxiv.org/abs/2209.04836

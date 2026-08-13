**Averaging Weights Leads to Wider Optima and Better Generalization (2018)**

Izmailov, Podoprikhin, Garipov, Vetrov and Wilson introduce Stochastic Weight Averaging (SWA): run SGD with a cyclical or constant learning rate starting from a conventionally trained model, and average the iterates in weight space. The claim is that SGD trajectories at non-trivial learning rates traverse the periphery of a wide, flat region of low loss, and their weight-space average lands near the center of that region - a wider optimum with better generalization. SWA is essentially free computationally (an extra running average buffer and one batch-norm statistics recompute at the end), needs no ensembling at inference, and improved essentially every architecture it touched: +0.8% top-1 for ResNet-50 and DenseNet-161 on ImageNet from just 10 extra epochs, and over 1.3% / 0.4% error reduction on CIFAR-100 / CIFAR-10 across Preactivation ResNet-164, VGG-16 and Wide ResNet-28-10, with larger gains for Shake-Shake and PyramidNet. The paper distinguishes SWA from FGE (Fast Geometric Ensembling), which averages model OUTPUTS along the same trajectories; SWA averages the weights themselves into a single model.

**Key mechanism**

- Start from a pretrained SGD solution; continue training with a cyclical or high constant learning rate so the iterates keep moving instead of settling into a point
- Maintain an equal-weighted running average of the iterates; the average is the final model
- Batch-norm statistics must be re-estimated for the averaged weights with one forward pass over the training data (the running moments belong to the trajectory, not the average)
- Geometry argument: at moderate constant LR the SGD iterates bounce around the boundary of a wide flat basin; their average is interior and flatter
- FGE uses high-frequency LR cycles and ensembles the outputs of the sampled models; SWA collapses the same idea into weight space

**Main findings**

- ImageNet: +0.8% top-1 for ResNet-50 and DenseNet-161, +0.6% for ResNet-152, from only ~10 averaging epochs appended to a converged run
- CIFAR-100: error reductions over 1.3% for Preactivation ResNet-164, VGG-16 and Wide ResNet-28-10; CIFAR-10 over 0.4% on the same architectures
- The SWA solution sits in a visibly flatter region of the loss surface than the SGD iterates it averages (loss-surface visualizations along trajectory planes)
- SWA also improves training-loss-level metrics, not only test - the averaged point is better centered in the low-loss region, not merely better regularized
- Works under both cyclical and constant learning-rate averaging schedules; the key requirement is that the LR stays high enough for the iterates to explore the basin boundary
- Near-zero overhead: one extra weight buffer plus the batch-norm recompute pass

**Key takeaways**

- Weight averaging is a within-run method: one trajectory, averaged over its tail, no cross-seed alignment problem
- The method presupposes a non-annealed tail - an LR schedule that anneals to zero leaves the iterates already converged to a point, with nothing to average; SWA and anneal-to-zero schedules are mutually exclusive designs
- For architectures without batch norm (transformers use LayerNorm), the statistics recompute step disappears and SWA is a pure running average
- A constant-LR tail appended to a decayed run is the standard integration; the averaging phase can be short relative to total training
- SWA generalizes across architectures and tasks, but its evidence base is vision/SGD; transformer fine-tuning with AdamW is an extrapolation

**Relevance**

- Core citation for R18-H154 thread (ii), with a campaign-critical tension: H120's instrument measured mean consecutive-step update cosine 0.9378 under our OneCycle anneal-to-zero and killed within-run averaging as "a lagged under-trained iterate" - SWA's own mechanism statement says the same thing (needs a high-LR tail to average), so adopting SWA means changing the schedule, not appending a buffer
- Direct overlap with H152's EMA 0.999: both are within-run weight averages; registering both on one run is double-counting one mechanism - pick per the registration order in the regime memo

**Tags**

- #weight-averaging
- #flat-minima
- #optimization

**Source**

- https://arxiv.org/abs/1803.05407

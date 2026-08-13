**Revisiting Few-sample BERT Fine-tuning (2021)**

Zhang, Zhao, Saleh and Liu (the R3F paper) systematically dissect instability in few-sample BERT fine-tuning and re-evaluate the remedies proposed for it. Working with eight GLUE tasks downsampled to few-sample budgets and 20-50 seeds per configuration, they first show that much of the instability attributed to exotic causes comes from mundane optimizer details: the original BERTADAM's omitted bias correction produces degenerate runs that fail to beat a random baseline, and simply using debiased Adam removes most failures and sharply reduces variance across seeds. They then re-evaluate recently proposed stabilizers - re-initializing pretrained layers, layer-wise learning-rate decay (LLRD), Mixout, SMART and their own R3F - under matched conditions. Re-initializing the pooler plus the top transformer blocks consistently improves both mean and worst-case performance. R3F, their parametric trust-region method that penalizes symmetric KL divergence between the fine-tuned model's outputs and the pretrained model's outputs on clean and noisy inputs, adds further stability gains in the few-sample regime. Crucially, the gains of all these methods shrink as dataset size grows - on full MNLI even the bias-correction effect disappears.

**Key mechanism**

- R3F anchors the fine-tuned model to the pretrained model in function space: a penalty on symmetric KL between current and pretrained output distributions, computed on both the standard input and a noise-perturbed variant of it
- Re-init: re-initialize the pooler and the top L transformer blocks at fine-tuning start; even pooler-only helps, deeper re-init helps more up to a plateau, then hurts
- Debiased Adam: restore the standard bias-correction terms; fixes the effective early learning rate that warmup alone mis-scales
- Trust-region methods (R3F, SMART) limit how far the fine-tuned function moves from the pretrained function per step, countering catastrophic drift on tiny datasets

**Main findings**

- With BERTADAM (biased), many seeds degenerate to random-baseline performance on RTE, MRPC, STS-B and CoLA (50 seeds per dataset); debiased Adam removes the degenerate tail and tightens the distribution
- The bias-correction benefit vanishes on full-size MNLI - the fix is a small-data phenomenon
- Re-init improves mean performance on all studied datasets and improves worst-case performance more than mean - a stability lever, not just an accuracy lever
- Re-init's effect is not local: re-initializing even one top block measurably perturbs the whole network's behavior
- R3F and SMART add gains on top of the optimizer fixes in few-sample settings; ordering and combination effects are measured under a shared protocol with model selection on validation sets
- Expected validation performance with debiased Adam reaches good results in 5-10 random trials, where the biased variant needs far more

**Key takeaways**

- Audit the optimizer before blaming the data: a missing bias-correction term was responsible for a large fraction of published "instability"
- Stability levers are regime-dependent: what rescues 1k-example fine-tuning may be inert at 100k+ examples, so few-sample evidence transfers weakly to large-mix training
- Re-initializing the top blocks is a near-free stabilizer worth testing even outside the few-sample regime, but the paper's own scaling trend warns of diminishing returns
- Function-space trust regions (R3F) and parametric anchoring are complementary to optimizer hygiene, not substitutes
- Model selection on a validation set across a handful of seeds is assumed throughout - single-seed claims in this space are noise

**Relevance**

- R18-H154 thread (iv): R3F/SMART-class trust regions are candidate levers, but this paper carries the campaign's most important caveat - its effects are largest in few-sample regimes and shrink with data; our 685,670-row mix is the many-sample regime, so expected effect sizes are modest and the papers' own trend line says so
- The re-init finding connects to Dodge et al.'s weight-init variance component: re-init changes which init the fine-tune starts from, a candidate zero-cost probe

**Tags**

- #fine-tuning-stability
- #trust-region
- #few-sample

**Source**

- https://arxiv.org/abs/2006.05987

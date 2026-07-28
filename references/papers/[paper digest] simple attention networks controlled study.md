**A Controlled Study of Attention-Only Transformers (2026)**

Feed-forward networks (FFNs) hold roughly two thirds of a transformer's non-embedding parameters, and a large interpretability literature treats them as the model's parametric memory, yet the converse experiment - delete them and measure what is actually lost - had never been run under controls. Removing the FFN perturbs parameters, compute and depth simultaneously, so the authors (Cactus Compute) pretrain attention-only decoder transformers, which they call Simple Attention Networks (SANs), against standard transformers matched separately on each of the three axes, with per-arm learning-rate sweeps, at 6M to 87M total parameters for up to 105B tokens. The headline is that the FFN is not necessary for general language modelling once its parameter budget is reallocated into attention depth, and that the small residual gap is not diffuse but localises entirely to parametric recall - predicting a fact from weights rather than from context. Attention-only models are measurably better on context-grounded answers and worse where knowledge must come from weights.

**Key mechanism**

- A SAN block is self-attention plus a gated residual, with no FFN; matched-parameter arms spend the freed budget on depth (20 attention-only layers against 4 standard blocks at 24M params)
- Three separate matchings, each fixing one axis: iso-depth (20L vs 20L, 24M vs 87M), iso-FLOP (20L vs 9L, 24M vs 43M), iso-param (20L vs 4L, exact to under 0.04%)
- QK-normalization, not FFNs and not residual gating, is what keeps 48-layer attention-only stacks trainable; an ablation found the gate factor dispensable in practice
- A scalar-bottleneck lemma shows that at sequence length 1 the whole depth-L network collapses to a linear map modulated by exactly L scalars, so all per-position nonlinearity is bottlenecked and cross-token attention is the only escape - pricing that restriction is what the experiments do
- Weight-spectrum tracking explains the result: routing matrices (Q/K) spectrally crystallize in the first quarter of training and then stop moving, while content matrices accumulate rank through the stable phase; deleting the FFN relocates that accumulation to the attention output projection
- Fairness protocol: every arm x optimizer cell got its own learning-rate sweep at 5B tokens (17 cells), all locks are interior minima; batch, schedule, data order, validation set and masking identical across arms

**Main findings**

- Deleting FFNs in place costs 0.470 nats (iso-depth, 87M to 24M, a 72% parameter deletion); at matched training FLOPs the standard transformer leads by 0.263 nats
- At matched parameters the gap is 0.0055 and 0.0054 nats on two clean seed pairs - 0.27% of loss, agreeing to one part in ten thousand; same-seed noise floor is 0.0015 nats
- The gap shrinks with training on separately trained and separately tuned budgets: 0.046 nats at 5B tokens, 0.019 at 30B, 0.0055 at 105B
- Token-region decomposition localises the deficit: at 31B the per-token deficit on query regions is +0.052, five times the sample aggregate, while carrying only 8% of loss; the token-weighted sum reproduces the aggregate to within 2%
- By 105B the localisation is complete - the SAN leads on every answer region including memorization exercises and on reasoning traces, and the deficit survives only on query tokens (+0.038), the positions with the least context to route from
- By task type the SAN is already better at 31B on rag, editing and math answers (retrievable from prompt or trace) and worse on memorization and free-form generation; by 105B it leads on all of them
- Benchmarks split the same way: Lambada (out-of-distribution recall for these models) favours the FFN arm at every budget, while Sciq - whose answer sits in a provided support passage - favours the SAN with a margin that grows in the pre-registered direction, 0.725 to 0.742 from 31B to 105B while the FFN arm regresses 0.702 to 0.661, seed ranges non-overlapping
- A pre-registered out-of-distribution test predicted a 0.02-0.05 nat gap on knowledge-dense web text; a matched pair trained on fineweb-edu measured 0.0398
- The storage-versus-routing identity of a task is relative to the training distribution, not intrinsic: the same fineweb pair reverses on Lambada (SAN 0.203 vs FFN 0.181) because distribution-matched text makes the passage sufficient to infer the final word
- Honest caveats: preprint, vendor-authored (Cactus Compute, which ships the derived Needle model); scale is 6-87M parameters, so extrapolation to billion-parameter models is not tested; the region/exercise decomposition uses a training-exposed sample, valid for localising the gap but not a held-out measurement; one of three iso-param seed pairs had a documented terminal instability and reverses the sign

**Key takeaways**

- Treat the FFN as a parametric-memory component, not a general-purpose computation component - if a task reads its evidence from context, the FFN budget is largely buying capability the task does not use
- Attention-only architectures are a parameter-efficiency lever specifically for context-grounded workloads: entailment, grounding, RAG verification, extraction - the regimes where the paper measures SANs ahead
- The iso-param comparison is the only fair one for a deployment decision; iso-depth deletion (0.47 nats) is what naive FFN removal costs and should not be quoted as the architecture's cost
- Budget depth, not width, when removing FFNs, and adopt QK-normalization before attempting deep attention-only stacks - it is the trainability precondition, not an optimisation nicety
- Expect the residual weakness on low-context tokens; a system that supplies evidence in the prompt largely avoids the failure mode the paper measures
- Validate on a task whose answer is in a supplied passage (Sciq is the paper's own proxy) rather than on a knowledge benchmark like MMLU, which measures the axis these models deliberately give up
- Do not assume the storage/routing split transfers unchanged - it depends on the match between training distribution and task, so it must be re-measured on a new domain
- Scale caveat is real: 87M is the largest arm, so this is evidence for small-model design, not a claim about frontier models

**Relevance**

- Load-bearing citation for hypothesis round 4 in `docs/experiments/semantic-grounding-experiments.md` - grounding supplies its evidence in the prompt, so it sits squarely in the regime where the paper measures attention-only models ahead, and the parametric-recall weakness is the one axis the task never exercises
- Supports moving off the incumbent mDeBERTa-v3-NLI cross-encoder on a mechanism argument rather than a size argument; the Sciq trajectory (margin growing with training while the FFN arm regresses) is the closest public analogue to the grounding task
- Tempering caveat for the same round: the evidence is 6-87M-parameter decoder LMs on next-token loss, not a fine-tuned pairwise entailment head, so transfer to a (claim, evidence) classifier is an assumption this project must test, not inherit

**Tags**

- #architecture
- #small-models
- #grounding

**Source**

- https://arxiv.org/abs/2607.18363

**The Road Less Scheduled (2024)**

Defazio, Yang, Mehta, Mishchenko, Khaled and Cutkosky propose Schedule-Free optimization, which removes the learning-rate schedule entirely while retaining the performance of well-tuned schedules. The motivation: schedules that do not know the stopping step T in advance are greatly outperformed by schedules that do, which forces a training-length commitment before the run starts and makes the endpoint sensitive to that commitment. Schedule-Free methods keep three weight sequences - z where the base optimizer steps, x an online weighted average of z that serves as the evaluation point, and y where gradients are evaluated, an interpolation between x and z controlled by a momentum parameter beta - and replace the schedule's annealing with Polyak-Ruppert-style averaging. Across a 28-problem evaluation suite ranging from logistic regression to ImageNet and LLM pretraining, Schedule-Free AdamW matches or beats heavily tuned cosine schedules at every training length simultaneously, and it won the MLCommons 2024 AlgoPerf Algorithmic Efficiency Challenge self-tuning track. Memory equals the base optimizer (no extra buffer over AdamW), and the only remaining schedule component is a short linear warmup, which stays necessary.

**Key mechanism**

- Three sequences per step: y_t = (1-beta) z_t + beta x_t (gradient location), z_{t+1} = z_t - gamma * grad f(y_t) (base update), x_{t+1} = weighted online average of the z iterates (evaluation point)
- beta = 0.9 interpolates between Polyak-Ruppert averaging (beta = 0) and primal averaging (beta = 1); the gradient at y behaves like momentum with the same immediate effect as EMA momentum but much slower incorporation of the remainder
- x-weighting c_t is proportional to gamma_t^2 and decays as 1/t after warmup, so recent iterates weigh more; weights and evaluation use x, never y or z
- Linear LR warmup over a fixed number of steps is retained; everything after warmup is schedule-free constant LR
- Optimal worst-case convergence for the convex non-smooth setting holds for ANY beta in [0,1], with constants; no extra hyperparameters over base SGD/AdamW

**Main findings**

- On the 28-problem suite (one of the largest optimizer evaluations published), Schedule-Free AdamW matches or outperforms heavily tuned cosine schedules, tracking the loss-vs-time Pareto frontier of a family of cosine schedules in a single run
- Won the MLCommons 2024 AlgoPerf self-tuning track against the NAdamW reference and all entrants, over 8 diverse workloads with 10 seeds each
- Best momentum beta = 0.9 is stable across training durations; weight decay and LR still need tuning per problem (the method removes schedule tuning, not all tuning)
- Warmup remains necessary; the paper uses linear warmup for a fixed step count in all experiments
- Schedule-free variants use the same memory as the base optimizer - no second copy beyond AdamW's moments; SWA or LAWA can be stacked on top for further gains
- ImageNet ResNet-50 result: Schedule-Free 0.9112 vs cosine 0.9110 test loss parity (with SEs overlapping), one of the 28 tracked problems

**Key takeaways**

- Schedule sensitivity is a removable design axis: if endpoint quality depends on where the anneal lands relative to the data, an averaging-based optimizer eliminates the coupling
- A single run becomes valid at every stopping time - useful when a 1-epoch budget is fixed but the "right" amount of training is unknown
- The x-sequence being a running average means schedule-free already contains an implicit EMA-like mechanism; adding an explicit EMA on top double-counts the averaging
- Evaluation must read the x weights, not the training iterate - integration requires the checkpoint/eval path to save and load the right sequence
- Evidence base spans convex problems, vision, and LLM pretraining, but not BERT-class fine-tuning with adversarial multi-component losses - transfer to that regime is unproven

**Relevance**

- Core citation for R18-H154 thread (iii): our 1-epoch / 14,300-step OneCycle budget hard-commits the anneal before training; schedule-free replaces that commitment with anytime-valid averaging - a candidate regime swap, not an add-on
- Mechanistic conflict with H152's EMA 0.999 (x is already an average) and a premise conflict with the H120 instrument kill (H120's "anneal-to-zero is already the implicit average" logic dissolves if the anneal is removed and the average made explicit)

**Tags**

- #optimization
- #learning-rate-schedules
- #weight-averaging

**Source**

- https://arxiv.org/abs/2405.15682

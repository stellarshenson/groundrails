**Self-Refine: Iterative Refinement with Self-Feedback (2023)**

Madaan et al. propose SELF-REFINE, a test-time loop in which one language model generates an output, critiques that output itself, and rewrites it from its own critique - repeated until a stopping condition. The problem addressed is that iterative refinement normally needs a trained refiner, domain-specific supervision or a reward model, all of which are expensive; SELF-REFINE needs none of these, only three few-shot prompts (generate, feedback, refine) and a single base model. The loop is evaluated on seven tasks spanning dialogue response generation, code optimization, code readability, math reasoning, sentiment reversal, and two new tasks the authors introduce (acronym generation and a harder constrained generation with 20-30 keyword constraints). Across GPT-3.5, ChatGPT and GPT-4, refined outputs beat single-pass outputs by roughly 20% absolute on average, with the largest gains in preference-scored generation tasks and essentially none in math reasoning. The most useful part of the paper is its ablation of feedback quality: the mechanism is carried by specific, actionable critique, not by refinement per se, and it fails on weaker base models.

**Key mechanism**

- Three prompts drive one frozen model M: p_gen produces the initial output y_0 = M(p_gen ‖ x); p_fb produces feedback fb_t = M(p_fb ‖ x ‖ y_t); p_refine produces the next draft. No weights are updated and no external critic exists
- The refine step conditions on the entire history, y_{t+1} = M(p_refine ‖ x ‖ y_0 ‖ fb_0 ‖ ... ‖ y_t ‖ fb_t), so past drafts and past critiques stay in context and the model can avoid repeating a rejected fix
- Feedback is deliberately shaped to be actionable (contains a concrete action likely to improve the output) and specific (names concrete phrases to change). The few-shot prompt supplies input-output-feedback triples exemplifying both properties
- The stopping condition is either a fixed iteration count (max 4 in the experiments) or a stop indicator the model itself emits in the feedback, for example a scalar quality score; for multi-aspect tasks the model emits per-aspect numeric scores so a draft that improves one aspect while regressing another can be rejected
- Both feedback and refine are implemented as few-shot prompts even for instruction-following models such as ChatGPT and GPT-4, to keep the setup identical across base models; decoding is greedy with temperature 0.7

**Main findings**

- Main results table, base versus +SELF-REFINE. GPT-4: sentiment reversal 3.8 → 36.2, dialogue response 25.4 → 74.6, code optimization 27.3 → 36.0, code readability 27.4 → 56.2, math reasoning 92.9 → 93.1, acronym generation 30.4 → 56.0, constrained generation 15.0 → 45.0
- ChatGPT: sentiment reversal 11.4 → 43.2, dialogue 40.1 → 59.9, code optimization 23.9 → 27.5, code readability 27.7 → 63.1, math 74.8 → 75.0, acronyms 27.2 → 37.2, constrained generation 44.0 → 67.0. GPT-3.5 shows the same pattern with math exactly flat at 64.1
- Math reasoning is the null result and the authors diagnose it: the model cannot tell that its chain is wrong. ChatGPT's feedback is "everything looks good" on 94% of math instances. With an external oracle signalling incorrectness, gains exceed 5%
- Feedback specificity is the active ingredient. Replacing task-specific feedback with generic feedback and then with no feedback: code optimization 27.5 → 26.0 → 24.8; sentiment reversal 43.2 → 31.2 → 0; acronym generation 56.4 → 54.0 → 48.0. Sentiment reversal collapses entirely without feedback
- Gains front-load across iterations and diminish. Averaged over the three base models: code optimization 22.0 → 27.0 → 27.9 → 28.8 over y_0 to y_3; sentiment reversal 33.9 → 34.9 → 36.1 → 36.8; constrained generation 29.0 → 40.3 → 46.7 → 49.7, where the first iteration alone contributes 11.3 of the 20.7 total
- The gain is not just extra sampling. Compared against ChatGPT generating k = 4 independent samples, human raters still preferred the single SELF-REFINE output over all k initial outputs in the 1-versus-k setting
- The method does not transfer downward. Vicuna-13B could not reliably emit feedback in the required format, and even given oracle or hard-coded feedback it repeated its output or hallucinated a conversation instead of refining. The authors attribute this to conversation-tuning not generalizing to test-time few-shot tasks
- Failure analysis on 70 manually inspected samples from code optimization and math reasoning: when refinement failed, 33% was feedback mislocating the error and 61% was feedback proposing an inappropriate fix, leaving only 6% attributable to the refiner mishandling good feedback. In successful cases 61% came from accurate feedback and 33% succeeded despite partially incorrect feedback
- Evaluation caveats that bound confidence: four of seven tasks have no automatic metric and are scored by blind human A/B preference, with GPT-4 used as a proxy judge correlating with human preference at 82% (sentiment reversal), 71% (dialogue) and 68% (acronyms) - so a third of acronym judgments diverge from humans. All base models are closed (GPT-3.5, ChatGPT, GPT-4, Codex) with undisclosed training data, and all datasets are English
- The preference-scored numbers are win rates in a comparison setting, not accuracies; the 3.8 → 36.2 style jumps reflect how often the refined output is preferred, and should not be read as absolute quality

**Key takeaways**

- Self-critique works only where the model can detect its own error. The math null result is the boundary condition: on tasks with a verifiable but subtle correctness criterion, the model reports "looks good" and the loop is a no-op. Test detection ability before building a refinement loop
- Budget one or two iterations, not four. Most of the gain arrives in the first refine step across all three tasks measured, and later iterations pay steeply diminishing returns for full generation cost each
- Feedback format is the design surface. Prompting for critique that names the offending span and proposes a concrete action is what carries the effect; generic "make this better" recovers only a fraction, and on some tasks nothing
- Emit per-aspect numeric scores when quality is multi-dimensional, so a refinement that trades one aspect for another can be rejected rather than accepted by default
- An external verifier beats self-verification wherever one exists. The 5%+ math gain under an oracle correctness signal is the paper's own evidence that the bottleneck is detection, not rewriting
- Capability threshold is real and undocumented. A 13B conversation-tuned model could not run the loop at all, so do not assume a small local model will substitute for a frontier model in a self-critique architecture without measuring it
- Errors concentrate in the critic, not the rewriter - 94% of failures traced to bad feedback. Invest evaluation effort in feedback quality, and consider a separate stronger critic model
- Treat the reported win rates as preference-relative. Where a task has a hard metric, the improvement is far smaller than where scoring is by preference

**Relevance**

- Not applicable to the current detector campaign. groundrails' deliverable is a single deterministic forward pass through a 307M cross-encoder per (sentence, evidence-window) pair, evaluated blind on RAGBench; a multi-pass generative critique loop is the opposite of that cost and determinism profile, and the paper is here from the 2025-04 agentic-RAG assistant reading, not from the detector work
- One transferable warning for the labeling pipeline in `docs/experiments/semantic-dataset-enhancements.md`: the failure analysis (94% of failures are the critic's, 6% the rewriter's) is direct evidence that in any generate-then-judge cascade, judge error dominates generator error. That justifies spending gate budget on judge agreement measurement rather than on generator tuning, which is what the DR-H112/H113 gates already do
- Second-order relevance to grounding as a research area: the math result - that a model cannot reliably spot an error inside a plausible-looking chain it produced itself - is the same failure mode a dedicated grounding detector exists to cover, and is an argument for an external verifier rather than self-checking

**Tags**

- #self-refinement
- #prompting
- #llm-agents

**Source**

- https://arxiv.org/abs/2303.17651

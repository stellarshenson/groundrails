**Can Tiny Language Models Reason? (2025)**

Technical blog by Bojan Jakimovski (Shekswess) presenting the Tiny Reasoning Language Model (trlm) project: a 135M-parameter decoder (SmolLM2-135M-Instruct) post-trained to emit structured chain-of-thought. The problem addressed is whether multi-step reasoning behavior can be compressed into a model small enough for on-device and low-latency deployment, following DeepSeek-R1's demonstration that reasoning is trainable at scale. The method is a three-stage curriculum: supervised fine-tuning on non-reasoning dialogue to stabilize a conversational prior, a second SFT stage introducing explicit `<think>...</think>` segments as tokenizer-level special tokens, and Direct Preference Optimization (apo_zero variant) over chosen-versus-rejected reasoning traces. The work matters as an existence proof that reasoning FORMAT and style are trainable at 135M with a single-GPU budget, and as a recipe document (full configs and dataset blends published); it is cited by the Falcon-H1-Tiny report. It is a self-published blog by a single researcher, not peer-reviewed, and reports no quantitative benchmark results - confidence in capability claims should be correspondingly low.

**Key mechanism**
- Stage 1 SFT: ~58k non-reasoning dialogues (SmolTalk2 subsets, 57.8% smol_magpie_ultra_no_think), 3 epochs, lr 3e-4, dft loss, max length 4096, NEFTune noise 0.01 - establishes instruction following without hidden thoughts
- Stage 2 SFT: ~78k traces with explicit `<think>` segments (51.5% Llama_Nemotron_reasoning_r1, 25.6% OpenThoughts3, 6.4% Qwen3-generated), 1 epoch; tokenizer extended with `<think>`/`</think>` special tokens and the chat template enforces thoughts inside tags, answers outside
- Stage 3 DPO: 50k chosen/rejected pairs from olmo-3-preference-mix-deltas, beta 0.1, apo_zero loss, lr 1e-5, 1 epoch - preference alignment as a style and correctness filter over reasoning traces
- Each stage initializes from the previous stage's checkpoint; full fine-tuning throughout (no LoRA), single AMD MI300x instance
- Monitoring is loss, token accuracy, entropy, and DPO reward margins - no held-out reasoning benchmark in the loop

**Main findings**
- Training is stable at 135M across all three stages: Stage 2 loss converges without collapse despite longer targets, Stage 3 reward margins widen monotonically - the curriculum is reproducible from published configs
- Delimiters are load-bearing: adding `<think>` as special tokens with an enforced template reduces malformed traces and stabilizes training more than data volume changes
- Curation dominates: the blend ratio among Llama_Nemotron, OpenThoughts, and multi-turn synthetic sources tracks directly with the trace styles the model reproduces
- Preference alignment is a strong lever even at 135M - post-DPO traces are qualitatively more concise and decisive; the authors argue alignment effect outweighs model size for style
- No benchmark numbers are reported anywhere in the post - evidence is training curves plus qualitative trace review; the single worked example (a "write an implausible answer" task) actually shows the model misreading the task instruction while producing well-formed `<think>` structure, illustrating form-over-competence
- Acknowledged limits: hallucinations and brittle steps on hard multi-turn problems, degradation near the context budget, single-epoch Stages 2-3, research-prototype status
- The training datasets are nearly all English (SmolTalk2, Nemotron, OpenThoughts3); a small aya_dataset_Qwen3_32B_think subset (6.4% of Stage 2) is the only multilingual component
- Future-work list places the sweet spot at 250M-300M backbones with multi-epoch curricula and GRPO on top of DPO

**Key takeaways**
- The transferable artifact is the recipe, not the model: staged SFT-then-think-then-DPO with special-token delimiters is a documented, low-cost path to make ANY small backbone emit structured, steerable reasoning
- Reasoning format is cheap to install; reasoning competence is not demonstrated - do not select a tiny reasoner on trace readability, demand task benchmarks
- For teams considering tiny reasoners as judges or verifiers: the class evidence (trlm, Tina, MobileLLM-R1) supports sub-billion reasoning as a serving-cost play, but every capable instance in the citation chain is 350M+ and benchmarked; 135M is below the demonstrated competence floor
- Special-token protocol discipline (thoughts fenced, answers outside) is worth adopting anywhere generative traces must be machine-parseable
- Blend composition should be version-pinned and published with the model - the post's own finding is that output style is a near-deterministic function of the blend
- Single-researcher blog with no eval harness: treat capability statements as hypotheses to test, and the config tables as the reliable content

**Relevance**
- Scale-inappropriate as a backbone for the groundrails campaign: trlm is 135M English-only with no benchmarks, while the campaign's gold is cross-lingual (de/fr/es/it/pl/hu/cn) and the incumbent mmBERT-base already reads 0.70496 blind - `PleIAs/Pleias-RAG-350M` (already in the register as the closest tiny-grounding precedent, arXiv 2504.18225) dominates it as a candidate on every axis
- The classification-head route (frozen or fine-tuned tiny reasoner + sequence-classification head) forfeits the only thing reasoning post-training adds - inference-time multi-step generation; a head reads one forward pass, so it buys only the pretrained representations, and the R15 field measured that representation capacity is not the binding constraint (data supervision is; capacity formally retired by author ruling 2026-08-09)
- The defensible slot is the opposite shape: a small GENERATIVE reasoner as a gated escalation tier for exactly the sentences the encoder reads at chance (numeric derivations, AUROC 0.48-0.53 per R15) - greedy-decoded verdict on the flagged residual only; this would be a new registration, and multilingual in-budget candidates are scarce (Qwen3-0.6B at 595.8M already breached the sub-400M budget in R8-H103)

**Tags**
- #small-language-models
- #reasoning
- #post-training

**Source**
- https://shekswess.github.io/tiny-reasoning-language-model.html

# R14 Evidence E5 - Is 307M mmBERT-base capacity the binding constraint?

**Scope**: analysis only, no training, no GPU. Every number below is either read from a banked artifact in this repo or computed here from a checkpoint / config on local disk. Model facts that come from model knowledge rather than disk are labelled and carry an explicit confidence. Arena-derived quantities are marked ANALYSIS ONLY and set no bars.

---

## Verdict, stated first

**CONFOUNDED, leaning NOT BINDING** - with one correction to the record that changes how the question must be asked.

- Capacity is **not proven** binding. Every capacity-flavoured deficit the campaign has isolated has also moved under a non-capacity lever at identical parameter count: finqa +0.0602 from a head-geometry change (H102), finqa +0.056/+0.034 from data (H108), the whole blind read +0.0142/+0.0180 from the windowed read on frozen weights (H101), and +0.0176 from output-averaging two frozen copies of the same model (anchor teacher)
- Capacity is **not refuted** either. The only direct capacity ablation in the record (R7-H50, 140M vs 307M) measured +0.021 with size, in-domain, against 0.0023 run noise - and it was never repeated on the blind read. FM5 is genuinely open, exactly as `R12_synthesis_full_field.md:203` and `R13_synthesis.md:220` state
- **The record's capacity accounting is wrong by ~3x.** Measured from the shipped checkpoint: 64.1% of "307M" is the embedding table. The compute-bearing transformer stack is **110.3M**, not 307M. Every prior capacity verdict in this campaign was computed against the wrong number

**Cheapest experiment that separates capacity from data/objective**: go DOWN the ladder, not up. Two seeded-paired mmBERT-small draws on the identical clean mix, read blind - ~12-14 GPU-h, versus ~33 GPU-h for the depth-upscale arm the record already priced. Details in section 9.

---

## 0. The measurement the record never took: 307M is a 110M model

Counted directly from `models/R9-H105-mmbert-dann-clean/trunk/model.safetensors` and its `config.json`.

| block | params | share |
|---|---|---|
| embedding table (256,000 x 768, tied) | 196,608,768 | 64.1% |
| 22 transformer layers (5,015,005 each) | 110,330,112 | 35.9% |
| final norm | 768 | - |
| **total** | **306,939,648** | 100% |

The trunk is ModernBERT-shaped: `hidden_size` 768, `intermediate_size` 1152 (GeGLU), 22 layers, `global_attn_every_n_layers` 3, `local_attention` 128, RoPE at theta 160,000, and `max_position_embeddings` **8192**. There are no learned position embeddings (`position_embedding_type: sans_pos`).

Three consequences, all load-bearing for the rest of this document.

- **The 400M cap is not the constraint people think it is.** Headroom above the current model is 93.06M, which at 5.015M/layer is 18.6 more layers. The depth-upscale arm already in the field (22L → 34L) computes to 34 x 5.015M + 196.61M = **367.11M** - which reproduces `R12_synthesis_full_field.md:191`'s stated 367.11M exactly, an independent check that this arithmetic is the same arithmetic the record used. The maximum within-family depth move is **22L → 40L at 397.2M**, never proposed
- **Width is budget-hostile and depth is budget-cheap**, because the vocabulary is 256k. Going 768 → 1024 costs 65.5M in embeddings alone before a single layer is added, and each layer then costs ~8.9M: only 15 layers fit. Depth buys 1.8x the stack under the cap; width buys 0.6x
- **The single largest capacity lever under the cap is the vocabulary, not the architecture.** Trimming to 64k frees ~147M; a rank-128 factorization of the tied table frees ~164M. Either would roughly triple the compute-bearing stack at unchanged total budget. Neither has ever been proposed in this campaign

---

## (a) The two-draw output ensemble: 0.72067 vs 0.70713 best single

`R13_anchor_teacher_result.json`, per-subset, against the two H105 windowed reads. Recomputed here from the three JSONs.

| quantity | value |
|---|---|
| H105 draw 1 windowed mean | 0.70471 |
| H105 draw 2 windowed mean | 0.70151 |
| pair mean | 0.70311 |
| output-probability ensemble | **0.72067** |
| ensemble − pair mean | +0.01756 |
| ensemble − best member | +0.01596 |
| ensemble − **per-subset oracle draw-picker** | **+0.00292** |
| subsets where ensemble beats both members | 8 / 10 |
| best plain checkpoint read ever banked (DR draw2 control) | 0.70713 |

**What a capacity-starved model class would show here: nothing.** Under a bias-dominated regime two draws of the same recipe converge on the same limited function, their errors are correlated, and the mean of their probabilities is the function itself. The measured +0.0176 is the opposite signature - it is a variance term, and it is a *lower bound* on the variance component of the blind gap, since a two-member average removes at most half of an independent error.

The strongest form of the argument is the oracle line. An omniscient picker allowed to choose the better draw *per subset* reaches 0.71774. The ensemble reaches 0.72067 and **beats the oracle picker by +0.0029**. That is only possible if the two draws are making genuinely different errors *within* each subset, cancelling in probability space. Different errors from identical architecture, identical data and identical schedule is a statement about how underdetermined the solution is, not about how much function the architecture can express.

The decisive contrast is already in the record: the **weight**-space average of the same two draws reads 0.69218 (H118 KILL) while the **output**-space average reads 0.72067. Same two objects, +0.0285 apart depending on where the averaging happens. A capacity story cannot produce that gap; a functional-divergence story does.

**Caveat.** Ensemble gains occur at every capacity, so the presence of a gain does not by itself refute a capacity limit - it bounds how much of the gap is *not* capacity. It bounds it at ≥ 0.0176 out of a total faithful-reachable headroom of 0.0529 (section 8), i.e. at least a third of everything available above the current read is variance the model class already spans.

---

## (b) gold_full ~0.85 vs blind ~0.70 - the gap is mostly the read, not the register

| read | H105 draw 1 | H105 draw 2 | draw spread |
|---|---|---|---|
| RAGTruth EN (in-domain, claim-level, n=600) | 0.8382 | 0.8361 | **0.0021** |
| RAGTruth non-EN (in-domain, 7 languages) | 0.8402 | 0.8337 | 0.0065 |
| gold_full (OOD register, claim-level, n=2752) | 0.8788 | 0.8240 | 0.0548 |
| blind arena (OOD, windowed decomposed-min) | 0.70471 | 0.70151 | 0.0032 mean / **0.0293 per subset** |

Two corrections to the "overfit-to-register vs transfer" framing:

- **gold_full is not in-domain for this recipe.** The clean mix removed the private pairs entirely (R9-H105). The 0.85 is transfer, not memorization - which is why H105's own registration predicted a drop to 0.72-0.80 and was refuted in the favourable direction
- **The 0.85 → 0.70 drop confounds register distance with read geometry.** gold_full is a claim-level AUC over 2,752 claims. The blind read is min-over-sentences of max-over-windows at the response level. The label-ceiling diagnostic prices exactly this: a **faithful oracle under the shipped read scores 0.7560 pooled** (`R12_label_ceiling_result.json`, ANALYSIS ONLY). There is no comparable ceiling on the claim-level read

Against a 0.7560 faithful ceiling and a 0.5 chance floor, the model's 0.7036 recipe mean captures **79.5% of the available above-chance signal**. A capacity-limited model does not sit at 79.5% of its task's oracle. What it says about capacity: the trunk fits its training distribution to 0.838 with a **0.0021** draw spread - it is not underfitting, and the thing that degrades with distributional distance is *stability*, not *level*.

---

## (c) H123 layer probe - deeper and mixed representations carry no unused signal

`R12-H123_layerprobe_result.json`, linear probes on all 23 hidden states of both frozen H105 draws, 20k rows of the public mix, no arena.

| draw | AUC(layer 22) | max AUC(l < 22) | margin | verdict clause |
|---|---|---|---|---|
| draw 1 | 0.9644 | 0.9651 (layer 21) | +0.0007 | fails (< +0.005) |
| draw 2 | 0.9654 | 0.9643 (layer 21) | −0.0011 | fails |

The registered bar was max AUC below the top ≥ AUC(22) + 0.005 on both draws; measured margins are +0.0007 and −0.0011. **KILL.** The task head is already reading the best representation in the stack.

Two further readings the probe supports that the verdict line does not state:

- **The top of the stack is saturated for this objective.** Draw 1 rises 0.5 → 0.8763 through layer 14, jumps to 0.9248 at 15, and then the final four layers (19-22) contribute 0.9649 → 0.9644, i.e. **~0.000**. Draw 2 is the same shape: 0.9625 at 18 → 0.9654 at 22, +0.003 across four layers. Adding more layers of the same shape on top has, on the in-domain objective, nothing left to extract - which is precisely why the record's own prior on the 22L → 34L depth-upscale is +0.001 to +0.005
- **There is no capacity competition between the task and the 12 DANN groups.** Domain-group accuracy at layer 22 is **0.9525 / 0.9370** against a chance of 0.0833, while task AUC is 0.964. The trunk carries near-perfect register identity *and* near-ceiling task signal in the same 768 dimensions. The FM5 "307M split n ways" framing has no measurement behind it, consistent with the skeptic amendment that struck it (`R12_synthesis_full_field.md:142`)

**Caveat**: this is an in-domain linear-probe necessary condition, licensed to adjudicate builds and nothing else. It predicts nothing about blind transfer, and its own scope note says so.

---

## (d) Student vs 568M teacher - capacity loses to data by an order of magnitude

R7-H50 / R8-H62, identical 159 held-out traces, 717 claims.

| model | params | gold AUC | RAGTruth EN |
|---|---|---|---|
| `bge-reranker-v2-m3` (frozen teacher) | 568M | **0.8619** | 0.6432 |
| mmBERT-base distilled (R7-H50) | 307M | 0.8479 | - |
| mmBERT-base multi-corpus (R8-H62) | 307M | 0.8531 | **0.8434** |
| mmBERT-small distilled | 140M | 0.8269 | - |
| `mDeBERTa-v3` NLI (frozen) | 278M | 0.7601 | - |

- **On the teacher's home ground, capacity wins - barely.** 568M beats 307M by 0.014 (0.0088 against H62). 1.85x the parameters buys under 0.015 AUC
- **Off it, data wins by 14x that.** The 568M teacher collapses to 0.6432 on RAGTruth where a 307M student trained on the right corpus reads 0.8434 - a **+0.200** swing at 54% of the size
- **Within the family, capacity is real but small and in-domain.** 140M → 307M = +0.021 against 0.0023 run-to-run noise; 11L (252.4M) → 22L (307.5M) = +0.032. The R7-H50 verdict "the task IS capacity-limited" is correctly recorded, but it is a statement about a **claim-level, in-domain, private-gold** read, and it has never been re-taken on the blind arena read that now defines the campaign

The parameter-count axis moves this task by ~0.02. The data and read axes move it by 0.10-0.20. Both facts are in the record; only the first has ever been called "capacity".

---

## (e) finqa numeric derivation - objective and pretraining, not parameter count

The failure is well characterised: derived-arithmetic sentences score ~0.01, and 75% / 68% of finqa / tatqa bottom-quartile argmins are derived-arithmetic sentences. It is the one mode the R8 failure analysis attributed to "true model-class capability" (~15% of residual). Three measurements bear on whether *parameters* are the axis.

- **H101 killed the coverage explanation**: with the full table in view under the windowed read, finqa moved −0.0019, and the penalty is checkpoint-dependent and *grows* (−0.0019 / −0.0267 / −0.0824). More table text in view gives the scorer more numbers to mishandle
- **H102 moved finqa +0.0602 on the identical trunk** by changing the output geometry to a token-span head (its truncated finqa read 0.7152 is the campaign best, 0.002 off the incumbent). Same parameters, same data, different objective. Whatever finqa needs, a 307M mmBERT already contains enough of it to be worth +0.06 when asked differently
- **H108 moved finqa +0.0561 / +0.0342 on the identical trunk** by adding 61,184 deterministic quantity-corruption pairs. Same parameters, different data

**Can a 400M encoder do arithmetic? No, and neither can a 307M one.** Arithmetic competence is a pretraining-corpus property (code, tables, math), not a parameter-count property, and no move inside the mmBERT family changes the pretraining corpus. The record reached the same conclusion in 2026-08-03 when it reopened the budget for a decoder scorer specifically because "decoder LM pretraining (code/tables/math) supplies the numeric-and-unit competence behind Mode C" - and then parked it at 595.8M for breaching the cap. Going 307M → 397M inside mmBERT buys 1.8x the layers over the *same* multilingual web pretraining. There is no mechanism by which that produces derivation.

The honest statement: finqa's residual is **objective-and-pretraining limited**, and it is the one place where "capacity" in the loose sense (what the model class can do at all) is genuinely implicated - but the lever is the pretraining corpus, which the 400M cap does not gate.

---

## (f) Run-to-run variance - underdetermination, and a 4x read amplification

Computed here across the six banked clean-family checkpoints (H105 d1/d2, H108 d1/d2, DR control d1/d2, all PRIMARY windowed reads).

| level | statistic |
|---|---|
| arena mean across 6 checkpoints | 0.70359, **SD 0.0033** |
| pooled per-subset SD across the same 6 | **0.0318** |
| worst subset (delucionqa) | SD 0.0533, range 0.1453 |
| H105 pair, mean per-subset spread | 0.0293 (max 0.0572, tatqa) |
| in-domain RAGTruth EN, H105 pair spread | 0.0021 |
| H105 draw correlation on the public mix (H129 gate) | 0.9775 |

**High seed variance at fixed recipe is not a capacity signature; it is close to its opposite.** An under-parameterized model class has a *narrower* set of achievable functions, so independent draws converge and seed variance falls. What is measured here is: near-identical fit in-domain (spread 0.0021, correlation 0.9775), wide functional divergence off-distribution (per-subset SD 0.032), and a mean that is 10x more stable than its own components because those component errors are independent and cancel. That is an underdetermined-solution regime - the data does not pin the function, not that the parameters cannot express it.

**Two caveats that must travel with the ±0.03 figure:**

- **Roughly 4x of it is the aggregator, not the model.** R8-H100 measured 0.0295 under min-over-sentences against 0.0074 on the whole-response read. The underlying function variance is ~0.007; the shipped read amplifies it
- **The six checkpoints above span three recipes**, so part of the 0.0318 is recipe effect. The H105 same-recipe pair alone gives 0.0293, so most of it is seed

---

## New measurement: MAX_LEN 512 silently truncates the read's own evidence unit

Computed here by tokenizing `R12-H121_gateA_scores.parquet` with the shipped tokenizer (4,000-row sample for the pooled figure; all 2,264 score-determining rows for the deciding figure). **ANALYSIS ONLY - sets no bars.**

| population | median tokens | fraction > 512 |
|---|---|---|
| all scored (sentence, window) pairs | 369 | **22.1%** |
| score-**deciding** pairs (argmin sentence x argmax window) | - | **7.1%** |
| deciding pairs, techqa | 493 | **46.4%** |
| deciding pairs, finqa | 380 | 9.6% |
| deciding pairs, tatqa | 223 | 8.0% |
| deciding pairs, six other subsets | 115-313 | 0.0% |

**The mechanism is independent of the arena and of any label.** The read's evidence unit is a 1,500-character window (`R8-H101_windowed_read.py:46-47`, fixed pre-run, never tuned). The tokenizer encodes 1,500 characters of token-dense text - logs, code, tables, CJK - into more than 512 subwords by construction. So for token-dense registers the scorer's actual evidence unit is smaller than the geometry claims, and the mismatch is a property of the window size and the tokenizer, not of anything measured on RAGBench. The arena figures above quantify exposure; they do not motivate the fix.

**This costs zero parameters.** The trunk carries `max_position_embeddings` 8192 with RoPE at theta 160,000, so MAX_LEN 1024 is deep inside the pretrained positional range - no extrapolation, no new weights. It is a compute move: 2/3 of layers use 128-token local attention, so doubling the sequence roughly doubles their cost and quadruples only the 8 global layers.

This is a **coverage** fix, not a capacity fix - and coverage is the family that produced the campaign's largest deterministic lift (H101 windowing, +0.0142 / +0.0180 on frozen weights).

---

## Capacity moves under the 400M cap

Param counts marked **(measured)** are computed here from local checkpoints or configs. Those marked **(knowledge)** come from model knowledge with the stated confidence, no web access.

| # | move | total | stack | assessment |
|---|---|---|---|---|
| 1 | **MAX_LEN 512 → 1024** (measured: 0 new params) | 306.9M | 110.3M | Not capacity - coverage. Zero parameters, RoPE/8192 already pretrained. Read-only variant is frozen-weights and deterministic. **Cheapest item on this list** |
| 2 | **Depth-upscale 22L → 34L**, identity block expansion (measured: 367.11M, matches the record) | 367.1M | 170.5M (1.55x) | The only shippable within-family capacity instrument. Record's prior +0.001 to +0.005 against a +0.010 bar, ~33 GPU-h. H123 saturation (section c) independently supports the low prior |
| 3 | **Depth-upscale 22L → 40L**, to the cap (measured: 397.2M) | 397.2M | 200.6M (1.82x) | Never proposed. Same mechanism as #2 with maximum dose; identity-initialised copies risk staying inert over one epoch, and that risk grows with 18 of them |
| 4 | **Vocabulary trim to ~64k on the training mix** (measured arithmetic) | ~307M | up to ~257M (2.3x) | Highest leverage per budget dollar by arithmetic - frees ~147M. Costs multilingual coverage, which the deliverable requires (8 RAGTruth languages, PsiloQA 14). Needs repair pretraining. No in-repo precedent |
| 5 | **Rank-128 factorization of the tied embedding** (measured arithmetic: 256k x 128 + 128 x 768 = 32.9M) | ~307M | up to ~274M (2.5x) | Preserves vocabulary coverage; imposes a rank-128 bottleneck on inputs and needs repair training. Speculative - no in-repo precedent, and I would not price its recovery cost confidently |
| 6 | **ModernBERT-large swap** (knowledge, medium confidence: ~395M, 28L, d=1024, ~50k vocab) | ~395M | ~343M (3.1x) | Best capacity-per-budget of any off-the-shelf option, because a 50k vocabulary spends almost nothing on embeddings. **English-only** - forks the multilingual deliverable and voids the 12-group DANN design. Already killed in-record on attribution invalidity (whole-model swap moves corpus, tokenizer, width, depth and languages at once) and priced at "+0.00-0.02, below noise" |
| 7 | **XLM-R-base** (measured from config: 278.0M total, **85.1M stack**) | 278M | 85.1M (0.77x) | **A capacity downgrade.** 250k vocab eats 192.4M; 12 layers of 7.09M. Similar total, 23% less compute than what we run |
| 8 | **mDeBERTa-v3-base** (measured from config: ~279M total, ~86M stack) | ~279M | ~86M (0.78x) | Also a downgrade, and closed in-record twice: diverged to NaN at lr 2e-5 and again at 1e-5 with warmup and clipping (R7-H50), and int8 gives it no speedup |
| 9 | **Larger mmBERT** (knowledge, medium confidence) | - | - | To my knowledge the family ships only small (140M, measured on disk) and base (307M). **No mmBERT-large exists.** Flagging this as the one claim here I would most want re-checked against the Hub |
| 10 | **Deeper / wider task head** | +1-5M | - | Ranked last on direct evidence: H123 shows layer 22 is already the best readout (margins +0.0007 / −0.0011). A larger head cannot add what the trunk does not carry |
| 11 | **Mixture-of-heads** | +small | - | **CLOSED - do not re-propose.** Head fusion (H104/H106) and token-head-as-primary (H102) are both on the closed list. The H102 head anti-correlation (finqa +0.0602, delucionqa +0.0867, hotpotqa −0.1323) is why it looks attractive and is exactly what was tried |
| 12 | **Distillation from the ensemble (H129)** | 307M | 110.3M | Capacity *substitute*, already registered and gate-licensed, teacher targets banked (685,670 rows, ~3.3 GPU-h paid). See the amendment below - as registered it cannot answer the capacity question |

**Qwen3-Reranker-0.6B** is measured in-repo at 595.8M and breaches the cap; it stays parked.

---

## Why H129 as registered is not a capacity test - and a cheap amendment

H129 trains one 307M student on the two-draw output mean. Its outcomes are ambiguous on capacity:

- **ADMIT** (pair mean ≥ 0.7091): the 307M class can represent enough of the averaged function to recover a third of the ensemble lift → capacity is not binding at the operating point. Clean read
- **KILL / REFUTE**: two indistinguishable explanations. Either in-domain distillation cannot transmit an OOD advantage - the pre-registered FM2 risk, and the gate already measured that the transmissible signal is concentrated in ~15% of the mix (RAGTruth median |p1−p2| 0.047-0.061 at 26.5-32.3% of rows ≥ 0.10; HaluEval 0.0017 / 1.8% is dead) - **or** the 307M student lacks the capacity to represent the average of two 307M functions. Nothing in the registered design separates them

**Amendment worth proposing**: run a second student arm on the *identical* banked teacher targets at the depth-upscaled 367.11M trunk. Everything expensive is already paid - teacher, targets, gate, seeding facility, control. The marginal cost is one extra draw (~13-16 GPU-h), and the contrast is a direct capacity measurement at the operating point: if the 367M student captures materially more of the +0.0176 than the 307M student, capacity is binding; if both land together, it is not.

---

## The cheapest experiment that separates capacity from data and objective

**Go down the ladder.** Every capacity probe in the field goes up (34L at ~33 GPU-h, ModernBERT-large at ~40-45 GPU-h for a declared non-shippable checkpoint). Downward is cheaper, and under a monotone-saturating capacity curve it is just as decisive.

**Design.** Train mmBERT-small on the byte-identical clean mix and recipe, under the H126 seeded-paired facility, and read blind through the frozen R8-H77 gate at the PRIMARY windowed read. The 307M arm is already banked (the H105 pair), so only the small arm needs GPU.

- **Measured capacity ratio** (from disk, not from parameter totals): mmBERT-small is 140,897,536 params of which the stack is **42,188,928** (1.92M/layer, d=384, same 22 layers, same 256k vocab). Against mmBERT-base's 110.3M stack that is a **2.61x** cut in compute-bearing capacity at only a 2.18x cut in nominal size - the down-ladder probe is a *sharper* capacity contrast than the headline numbers suggest
- **Cost**: ~6-7 GPU-h per draw at roughly half the base model's step cost, 2 draws for resolvability, plus ~2 GPU-h of reads. **~12-14 GPU-h** versus 33 for the upward arm
- **Why 2 draws**: unpaired single-draw blind noise is 0.0295; H126's target paired-delta SD is ≤ 0.014. One paired draw resolves a 0.03 effect at ~2 sigma, two resolve 0.02
- **Pre-registered two-sided bar** (blind, no arena statistics used to set it - the effect size is transferred from R7-H50's in-domain 0.021 and the campaign's noise record): **CAPACITY LIVE** if the blind pair mean falls ≥ 0.020 below the H105 pair mean of 0.70311, with sign agreement on both draws → the curve is still climbing at 307M, and the 34L / 40L spend is licensed. **CAPACITY CLOSED** if it falls < 0.010 → the curve has already plateaued *below* 307M, and under monotone saturation 367M cannot help; FM5 closes and the residual re-attributes entirely to read, data and objective. Between 0.010 and 0.020, record as unresolved and fall back to the H129 two-arm amendment
- **What it buys beyond the verdict**: a measured in-domain → blind capacity slope, which the campaign has never had. R7-H50 gives the in-domain leg (+0.021); this supplies the blind leg, and the ratio prices every future capacity proposal without running it

**Run first, before any of it** - the zero-training precursor: the **MAX_LEN 1024 read on the two frozen H105 draws**, ~1-2 GPU-h, deterministic, no new weights. It removes the coverage confound (46.4% of techqa's deciding pairs are truncated today) before a single GPU-hour is spent on capacity, and it belongs to the same frozen-weights coverage family that produced the campaign's largest deterministic lift.

---

## Sources

Repo artifacts: `R13_anchor_teacher_result.json`, `R9-H105_result.json`, `R9-H105_draw2_result.json`, `R9-H105_windowed_result.json`, `R9-H105_draw2_windowed_result.json`, `R10-H108_lane_draw{1,2}_windowed_result.json`, `DR_lane_draw{1,2}_control_windowed_result.json`, `R11-H118_soup_h105_windowed_result.json`, `R12-H123_layerprobe_result.json`, `R12-H121_gateA_scores.parquet`, `R12_label_ceiling_result.json` (via canonical log), `R8-H101_windowed_read.py`, `R8-H77_unseen_arena.py`, `models/R9-H105-mmbert-dann-clean/{trunk/config.json,trunk/model.safetensors,tokenizer.json}`, `~/.cache/huggingface/hub/models--jhu-clsp--mmBERT-small/.../model.safetensors`, and configs for `mmBERT-base`, `xlm-roberta-base`, `mdeberta-v3-base`, `Qwen3-Reranker-0.6B-seq-cls`.

Canonical log: `docs/experiments/semantic-grounding-experiments.md` lines 92, 1072-1102, 1289, 1338, 1341, 1531, 1942-1946, 1975-1999, 2033, 2062-2064, 2318-2320, 2452-2467, 2543-2551, 2607-2630. Synthesis: `R12_synthesis_full_field.md` lines 38, 142, 191, 203; `R13_synthesis.md` line 220.

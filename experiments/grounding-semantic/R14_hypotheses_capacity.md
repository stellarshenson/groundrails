# R14 hypotheses - LENS: ARCHITECTURE / CAPACITY

**Scope**: remediation hypotheses that move the deliverable through model capacity and architecture under the 400M parameter cap. Every hypothesis names the capacity MECHANISM it tests, so that a null result rules capacity in or out rather than leaving it poked.

**Discipline**: ANALYSIS ONLY in producing this document - no training, no GPU. Polars throughout. Every number below is either read from a banked artifact, recomputed here from a local checkpoint / parquet, or cited to an evidence pack with its file. No bar is set from an arena AUROC; the only arena-derived quantities entering any bar are (i) per-subset seed sigma, which the campaign's own ruling 1 already uses to price per-subset bars, and (ii) a label-free token-length exposure count. Both are flagged where they appear.

---

## 0. What this lens inherits, verified

### 0.1 The record's capacity accounting is wrong by ~3x - re-verified here

Counted directly from `models/R9-H105-mmbert-dann-clean/trunk/model.safetensors` and `config.json` in this session:

| block | params | share |
|---|---|---|
| tied embedding table (256,000 x 768) + norm | 196,608,768 | **64.05%** |
| 22 transformer layers | 110,330,112 | 35.9% |
| final norm | 768 | - |
| **total** | **306,939,648** | 100% |

Independently re-verified from `~/.cache/huggingface/hub/models--jhu-clsp--mmBERT-small`: total 140,897,536 with the transformer stack at **42,188,928** (per-layer 1,917,678; d=384, 22 layers, same 256k vocabulary). This reproduces E5's figure exactly.

Consequences, all load-bearing below:

- **The compute-bearing model is 110.3M, not 307M.** Every capacity verdict in the campaign was computed against the wrong number
- **Down-ladder contrast is sharper than nominal.** mmBERT-base stack / mmBERT-small stack = 110,330,112 / 42,188,928 = **2.615x**, against a nominal total-size ratio of only 2.18x
- **Depth arithmetic re-verified**: 22L -> 34L = 196,608,768 + 34 x 5,015,040 + 768 = **367.12M** (E5 and `R12_synthesis_full_field.md:191` both state 367.11M - the 0.01M is the per-layer rounding, the arithmetic is the same). 22L -> 40L = **397.21M**, the maximum within-family move under the cap
- **The vocabulary, not the architecture, is the largest single budget item.** A trim to 64k frees 147.46M - more than the entire current transformer stack

### 0.2 Sequence exposure - measured in this session, label-free

Tokenized `R12-H121_gateA_scores.parquet` with the shipped tokenizer (`models/R9-H105-mmbert-dann-clean/tokenizer.json`), no labels involved.

| population | n | median tokens | p99.9 | max | frac > 512 | frac > 1024 |
|---|---|---|---|---|---|---|
| all scored (sentence, window) pairs (20k sample, seed 1) | 20,000 | 372 | 953 | 1,029 | **22.93%** | **0.01%** |
| score-**deciding** pairs (argmin sentence x argmax window), ALL | 2,264 | - | - | - | **7.11%** | 0 |

Per-subset, deciding pairs only (all 2,264, exact):

| subset | n | median tokens | frac > 512 |
|---|---|---|---|
| techqa | 250 | 493 | **46.4%** |
| finqa | 250 | 380 | **9.6%** |
| tatqa | 250 | 223 | **8.0%** |
| expertqa | 203 | 247 | 0.5% |
| delucionqa | 184 | 313 | 0.0% |
| emanual | 132 | 250 | 0.0% |
| covidqa | 245 | 173 | 0.0% |
| hagrid | 250 | 166 | 0.0% |
| pubmedqa | 250 | 115 | 0.0% |
| hotpotqa | 250 | 132 | 0.0% |

This reproduces E5's independent measurement (7.1% pooled deciding, techqa 46.4%, finqa 9.6%) and adds the decisive new number: **MAX_LEN 1024 removes 99.99% of the truncation** - max observed pair length is 1,029 tokens, p99.9 is 953.

### 0.3 The evidence this lens must respect

- **E5**: capacity CONFOUNDED, leaning NOT BINDING. Output-space ensemble of two frozen H105 draws reads 0.72067 against a 0.70311 pair mean (+0.01756), and beats a per-subset oracle draw-picker (0.71774) by +0.00292, on 8/10 subsets. Weight-space average of the same two objects reads 0.69218. H123 layer probe: layer 22 is already the best readout (margins +0.0007 / -0.0011, KILL). 568M teacher beats 307M by 0.014 on home ground and loses by 0.200 on RAGTruth. Verified here from `R13_anchor_teacher_result.json` and `R12-H123_layerprobe_result.json`
- **E3**: finqa carries 79% structure (SD 0.0912 across 14 configs vs seed sigma 0.0421) - the only subset where interventions demonstrably resolve. delucionqa carries 85% noise, seed sigma 0.0432 against an analytic AUROC SE of 0.0485 with 12 hallucinated responses; **zero** of 14 trained configs move it past 2 sigma. techqa and emanual are the cleanest instruments (13% / 11% noise) and have never been targeted. finqa's systematic cost partner is **pubmedqa at r = -0.84**, not delucionqa (r = +0.013, null)
- **E2**: delucionqa's 2-draw noise is ~+/-0.10; any delucionqa bar below that is unenforceable - verdict DECLINE ON MEASURABILITY. Every banked delucionqa read sits above its own faithful-oracle ceiling (0.6657), so raising it further optimizes leaky partial-support firing
- **E1**: finqa's ceiling under the shipped read is 0.7348 and the admitted H108 lane already sits 0.0057 below it; response verbosity alone reads finqa 0.6958 against the model's 0.6489, so a finqa bar a length heuristic clears is not a grounding bar
- **E4**: neither finqa nor delucionqa is a capacity problem at item level - the same model scores 0.9811 and 0.0822 on two lookups into the *same* 266-char table window, differing only by a `$` sign

**Consequence for every bar in this document**: delucionqa is a DIAGNOSTIC surface and is barred from adjudicating any hypothesis here (E2 section 9, E3 (d)). finqa appears as a primary clause only where the mechanism predicts a finqa move, and always with **pubmedqa as its guardrail** per E3, never delucionqa. Where a lever is mix-wide, the mean bar is legal per ruling 7 and is used.

### 0.4 Standing protocol every hypothesis below inherits

- Read: PRIMARY windowed decomposed-min (window 1500, stride 750) through the frozen R8-H77 gate
- Comparison: clean H105 pair mean **0.70311** (draws 0.70471 / 0.70151, verified from `R13_anchor_teacher_result.json`)
- Per-subset clean pair means (E3 baseline row): covidqa 0.7878, delucionqa 0.8166, emanual 0.6976, expertqa 0.7728, finqa 0.6333, hagrid 0.6340, hotpotqa 0.6667, pubmedqa 0.6063, tatqa 0.7320, techqa 0.6840
- Seed sigma per subset (E3 (d)): finqa 0.0421, delucionqa 0.0432, techqa 0.0265, pubmedqa 0.0217, tatqa 0.0414, emanual 0.0211, expertqa 0.0212, hotpotqa 0.0236, hagrid 0.0193, covidqa 0.0123
- Draws: 2 per arm under the H126 seeded-paired facility
- Standing holds: gold_full pair mean >= 0.84 (clean pair mean 0.8514, from E5's 0.8788 / 0.8240); no subset < 0.55; no subset more than **0.06** below the clean control (ruling 9 general hold)
- Training arms serialize on GPU1 at the recipe contract (ruling 7). Any hypothesis that breaches the contract says so explicitly and asks for an amendment

---

## L5-C1 - DOWN-LADDER CAPACITY CALIBRATION

**Named mechanism**: monotone-saturation test of the parameter axis, taken *downward* instead of upward.

### The capacity mechanism being tested

Every capacity probe the campaign has priced goes up the ladder (22L -> 34L at ~33 GPU-h, ModernBERT-large at ~40-45 GPU-h for a declared non-shippable checkpoint), and each one has the property that a null is uninformative - "we added parameters and nothing happened" is equally consistent with "capacity does not bind" and "identity-initialised blocks stayed inert over one epoch". Downward is different. Under any monotone-saturating capacity curve - the only shape consistent with the record's own R7-H50 measurement of +0.021 from 140M to 307M - the slope *below* the operating point upper-bounds the slope above it. If a 2.615x cut in the compute-bearing stack costs less than 0.010 of blind arena mean, the curve is already flat below 307M and no within-family upward move can help; FM5 closes. If it costs 0.020 or more, the curve is still climbing at the operating point and the 34L / 40L spend is licensed on measurement rather than on prior.

The design is cheap for a structural reason: **the 307M arm is already banked** (the H105 pair, both draws, both reads). Only the small arm needs GPU.

Three properties make this the sharpest available capacity instrument:

- **The contrast is 2.615x, not 2.18x.** Both checkpoints carry the same 256k tied embedding table (196.6M at d=768, 98.3M at d=384), so the nominal parameter ratio understates the compute-bearing ratio (verified in section 0.1 from both safetensors files)
- **It is the same family, same pretraining corpus, same tokenizer, same 22 layers, same recipe.** Depth, corpus, tokenizer, language coverage and objective are all held; only width moves (768 -> 384). This is the only capacity contrast available anywhere in this campaign that changes one variable. E5's item 6 correctly kills ModernBERT-large on exactly this ground - a whole-model swap moves corpus, tokenizer, width, depth and languages at once
- **It supplies the leg the record has never had.** R7-H50 measured the in-domain, claim-level, private-gold leg (+0.021 from 140M to 307M). This supplies the *blind* leg. Their ratio prices every future capacity proposal in the campaign without running it

### Why the record's existing capacity evidence does not already settle it

R7-H50's "the task IS capacity-limited" verdict (canonical log line 1081) is recorded against a macro-F1 / AUC spread of 0.021 on 159 held-out traces of private gold at claim level, under a whole-response read, on a data era that no longer exists. It has never been re-taken blind. E5's argument that capacity is not binding rests on the ensemble (+0.01756 over pair mean, beating the oracle draw-picker by +0.00292), which bounds the *variance* component of the gap at >= 0.0176 - it does not bound the bias component. Neither settles the question. This arm does.

### Kill-gate (cheap, before any training)

**Frozen-representation capacity ladder**, ~0.5 GPU-h on a gate card, reusing `R12-H123_layerprobe.py` unmodified:

- Run the H123 linear probe over all hidden states of **pretrained, un-fine-tuned** mmBERT-base and mmBERT-small, on the same fixed-seed 20,000-row slice of the clean public mix H123 used (`R9-H105_clean_mix.public_train()`, seed 20260808, 30% test). No arena, no gold
- **LICENSE** if best-layer probe AUC(base) - best-layer probe AUC(small) >= 0.010: the size axis is visible in frozen features on the training distribution, so the trained arm is measuring capacity and not fine-tuning noise
- **KILL** if the gap is < 0.005: the two sizes are representationally indistinguishable for this objective before any fine-tuning, the arm cannot separate capacity from optimization noise, and the record gains "within-family width is not a representational axis for grounding" for 0.5 GPU-h
- **NO-READ** (fall through to the arm, gate recorded as uninformative) if AUC(base) < 0.60 - a floor effect on frozen features says nothing about either model
- Scope caveat inherited verbatim from H123: this is an in-domain linear-probe necessary condition and predicts nothing about blind transfer

### Pre-registered bar (blind; no arena AUROC used to set it)

Effect size transferred from R7-H50's in-domain 0.021 and the campaign's own noise record; sigma from E3's replicate table.

- **CAPACITY LIVE**: small pair mean <= **0.68311** (0.70311 - 0.020) with both draws below their paired control, **OR** finqa <= **0.5491** (0.6333 - 2 sigma = 0.0842) on both draws -> the curve is still climbing at 307M; the 34L (367.12M) and 40L (397.21M) spend is licensed on measurement
- **CAPACITY CLOSED**: small pair mean >= **0.69311** (drop < 0.010) **AND** finqa >= **0.5912** (within 1 sigma) -> the curve has plateaued below 307M; under monotone saturation 367M cannot help, FM5 closes upward, and the residual re-attributes entirely to read, data and objective
- **UNRESOLVED** between 0.010 and 0.020 -> recorded as such; the question falls through to L5-C3, which measures capacity at the operating point instead of below it
- **SHIPPING clause (secondary, and a real one)**: if small pair mean >= 0.69311 **AND** gold_full pair mean >= 0.84 **AND** no subset < 0.55 **AND** no subset more than 0.06 below the clean control -> mmBERT-small is ADMITTED as a 2.18x-smaller, ~2.6x-cheaper deliverable candidate at 140.9M, and the parameter budget conversation restarts from a different place
- **Guardrail reporting (adjudicates nothing)**: pubmedqa reported alongside finqa per E3's r = -0.84 finding. delucionqa reported and explicitly barred from adjudication per E2 section 9

### Legality

Clean - no contamination surface. mmBERT-small is a public pretrained checkpoint already on local disk; the mix is byte-identical to the clean 685,670-row public mix; no RAGBench corpus or derivative enters. The recipe is unchanged (BCE + DANN lambda 0.02, OneCycleLR 1 epoch, MAX_LEN 512, batch 48), so ruling 7's hardware contract is honoured exactly. The resulting checkpoint is trivially under the 400M cap. The only arena contact is the blind read through the frozen gate.

### Cost

~12-14 GPU-h: 2 training draws at ~5-6 GPU-h each (E5's estimate; my own step-cost arithmetic at 0.382x the base stack gives a floor near 3 GPU-h per draw, so 12-14 is the conservative number), plus ~1 GPU-h of reads, plus the 0.5 GPU-h gate. Against ~33 GPU-h for the upward arm the record already priced.

---

## L5-C2 - SEQUENCE-CAPACITY MATCHING (MAX_LEN 512 -> 1024)

**Named mechanism**: align the model's token budget with the read's own evidence unit. This tests *sequence* capacity - the amount of evidence the scorer can actually condition on - as an axis distinct from parameter capacity, at **zero additional parameters**.

### The capacity mechanism being tested

The shipped read's evidence unit is a 1,500-character window (`R8-H101_windowed_read.py:46-47`, fixed pre-run, never tuned). The scorer's evidence unit is `min(1500 chars, 512 subword tokens)` (`src/groundrails/semantic_ov.py:36`, `MAX_LEN = 512`). For token-dense registers - logs, code, serialized tables, CJK - 1,500 characters encode to more than 512 subwords **by construction**, so the geometry the read advertises and the geometry the model receives are different objects.

The mechanism is a property of the window size and the tokenizer and is independent of the arena and of any label. Section 0.2 quantifies the exposure with a label-free token count: 22.9% of all scored pairs and **7.11% of score-deciding pairs** are clipped today, concentrated at techqa 46.4% and finqa 9.6% of deciding pairs. Raising the cap to 1024 removes 99.99% of it (max observed pair 1,029 tokens).

This costs zero parameters. The trunk carries `max_position_embeddings` 8192 with RoPE at theta 160,000 and `position_embedding_type: sans_pos` (verified from `config.json` in this session), so 1024 is deep inside the pretrained positional range - no extrapolation, no new weights, no cap pressure.

### Why the record's two prior truncation refutations do not cover this

Both must be engaged honestly; neither addresses this intervention.

- **Canonical log line 280** ("`max_length` cap does not help ... `MAX_LEN` stays 512") is a *serving-latency headroom* argument from the round-5 CPU cascade work: it measured that capping *down* to 256 saves only ~17% while clipping the median pair. It is evidence about the cost of shrinking the cap, not evidence that 512 is sufficient
- **Canonical log line 1340** ("Truncation was hypothesised and REFUTED ... re-scoring at 2048 moved finqa only 0.398 -> 0.428 ... made techqa WORSE (0.703 -> 0.641)") is a *whole-chunk* re-score on the R8-H62 checkpoint - the arena-worst configuration ever banked, finqa 0.3974 - before the windowed read existed. At 2048 with whole chunks the model receives *more evidence text*: the input distribution changes. Here the evidence unit is fixed at 1,500 characters and only the clipping is removed: the model receives the *same* text, un-truncated

That said, the techqa 0.703 -> 0.641 result is the single most important counter-signal on this line and it is taken seriously below, because it has a named mechanism: a model fine-tuned entirely at <= 512 tokens has never had a gradient at positions 513-1024, so its behaviour there is unconstrained even though the backbone's RoPE range covers it. That is exactly what the staged design measures rather than assumes.

### Kill-gate

**Stage 0 - precondition, already measured, 0 GPU** (section 0.2): pooled score-deciding truncation >= 5% and at least one subset >= 25%. Measured **7.11%** pooled with techqa at **46.4%**. Precondition PRESENT. Had it read below those thresholds the line would have died here for zero cost.

**Stage 1 - frozen-weights read at MAX_LEN 1024, ~2-3 GPU-h, deterministic (zero draw noise), both H105 draws, no training.** Three pre-registered branches, each decisive:

- **ADOPT (no training needed)**: the three truncation-exposed subsets (techqa, finqa, tatqa - selected by the label-free token count of section 0.2, never by an AUROC) average >= +0.010 on BOTH draws AND arena mean >= -0.002 on both AND no subset <= -0.020. Tighter-than-0.06 guards are legal here because the read is deterministic (ruling 9)
- **LICENSE the training arm**: any exposed subset moves <= -0.010 on either draw -> the fine-tuned model does not generalise past its training length, the position mismatch is confirmed as the mechanism, and the repair is to train at the read's length
- **KILL the whole line** (~2-3 GPU-h spent total): every subset moves within +/-0.010 on both draws -> sequence capacity is not binding, the 512-token bottleneck is ruled out as an explanation for the techqa and finqa deficits, and the record gains a clean null on a named axis

### Pre-registered bar for the Stage-2 training arm (if licensed)

MAX_LEN is a mix-wide lever, so a mean bar is legal per ruling 7.

- **ADMIT**: pair mean >= **0.7091** (0.70311 + 0.006) with sign agreement on both paired draws
- **Subset-primary**: techqa >= **0.7040** (+0.020 over the 0.6840 clean pair mean; techqa carries 46.4% deciding-pair exposure and is E3's cleanest instrument at 13% noise, seed sigma 0.0265, so +0.020 is a resolvable 0.75-sigma move on a subset that has never been targeted) **AND** finqa >= **0.6433** (+0.010 over 0.6333, priced to finqa's measured 9.6% exposure - deliberately not larger, because E1 shows finqa's ceiling under this read is 0.7348 and its verbosity baseline is 0.6958)
- **Holds**: gold_full pair mean >= 0.84; no subset < 0.55; no subset more than 0.06 below control; **pubmedqa >= 0.5846** (1 sigma below its 0.6063 clean pair mean) as E3's mandated finqa guardrail
- **KILL**: pair mean <= 0.70311 or techqa < +0.010 on either draw
- delucionqa is reported and barred from adjudication (E2): its deciding-pair truncation is measured at 0.0%, so the mechanism predicts no delucionqa move at all, and any observed move is seed placement

### Legality - two flags for the author

1. **Ruling 7 breach (Stage 2 only).** The hardware contract fixes the recipe at batch 48 / MAX_LEN 512 with no batch or accumulation changes authorized. A MAX_LEN 1024 training arm changes a recipe constant and **requires an explicit author amendment**. Memory is very likely fine on the 96GB card - 14 of 22 layers use 128-token local attention, and with dynamic padding only the ~6.5% of training pairs above 512 tokens (canonical log line 280: chunk pairs run ~331 median / ~590 p95) actually lengthen - but this must be *measured* before the arm, not assumed. Stage 1 breaches nothing: it is a frozen-weights read
2. **Serving-path consequence.** `MAX_LEN` lives in the shipped library (`src/groundrails/semantic_ov.py:36`). Per ruling 2's precedent the change must ship identically for every corpus and every input - which it does, being a global token cap with no subset conditioning. The cost is CPU serving latency on the 22.9% of pairs that currently clip; the round-5/6 cascade record (662 ms/claim warm mean) is the budget this has to be re-priced against before adoption

Contamination: none. No corpus changes; Stage 1 touches no weights; Stage 2 trains on the byte-identical clean mix.

### Cost

Stage 1 ~2-3 GPU-h (deterministic reads, both draws). Stage 2, if licensed, ~15 GPU-h (2 draws at ~7 GPU-h - the step-cost increase is bounded by the ~6.5% of training pairs that exceed 512 tokens under dynamic padding, not by a 2x sequence factor - plus ~1 GPU-h of reads). **Total 2-3 if it kills, ~17-18 if it runs through.**

---

## L5-C3 - CAPACITY ARM ON THE BANKED ENSEMBLE TEACHER (H129 two-trunk amendment)

**Named mechanism**: measure representational capacity against a target function that is *known to exist and known to be better*, at the operating point, so that a null is informative.

### The capacity mechanism being tested

R13-H129 is registered and licensed: a 307M student trained on the two-draw output-mean soft targets, teacher targets banked (`R13-H129_teacher_targets.parquet`, 685,670 rows, ~3.3 GPU-h already paid, key_hash-aligned and verified). Its gate returned LICENSE (median |p1-p2| 0.01248 with 14.44% of rows >= 0.10 - verified here from `R13-H129_gate_result.json`).

**As registered it cannot answer the capacity question.** Its ADMIT branch is clean (307M can represent enough of the averaged function -> capacity does not bind at the operating point), but its KILL and REFUTE branches are two-way ambiguous:

- either in-domain distillation cannot transmit an OOD advantage - the pre-registered FM2 risk, sharpened by the gate's own finding that the transmissible signal is concentrated in ~15% of the mix (RAGTruth family median 0.047-0.061 at 26.5-32.3% of rows >= 0.10; HaluEval 0.0017 / 1.8% is dead)
- or a 307M student cannot represent the average of two 307M functions

Nothing in the registered design separates them, and FM5 stays open either way.

**The amendment**: run a second student arm on the *identical* banked teacher targets, identical loss (0.5·BCE + 0.5·MSE), identical DANN and schedule, at the depth-upscaled **367.12M** trunk (22L -> 34L, identity block expansion). Everything expensive is already paid - teacher, targets, gate, seeding facility, control. The contrast Δ367 - Δ307 is a direct capacity measurement at the operating point, and both outcomes are informative:

- **Δ367 materially above Δ307** -> the 307M student was capacity-limited on a function proven to exist; capacity binds; 40L (397.21M) is licensed
- **Δ367 ≈ Δ307** -> capacity does not bind at the operating point, and every H129 outcome attributes to transmission (FM2) rather than to parameters. **FM5 closes upward**, and combined with a CAPACITY-CLOSED reading from L5-C1 the capacity question is answered from both directions

This is the property that makes it worth its GPU-h. Standalone, a depth-upscale arm has a bad prior and should not be registered: the record's own prior is +0.001 to +0.005 against a +0.010 bar, H123 measured layer 22 as already the best readout (margins +0.0007 / -0.0011, verified here), and the final four layers of the existing stack contribute ~0.000 in-domain (draw 1: 0.9649 at layer 19 -> 0.9644 at 22). That is why rank 9 sits below the cut in the R12 registration. Its value here is **not** an expected mean gain - it is that a null answers FM5, which nothing else on the board does.

### Kill-gate (cheap, before the training arm)

**Identity-expansion validity gate, ~0.4 GPU-h**, two clauses, both required:

- **(a) Function preservation, numerical.** Build the 34L identity-expanded trunk (each inserted block initialised so its residual contribution is exactly zero at init) and assert `max |p_expanded - p_parent| < 1e-4` over the fixed-seed 20,000-row public-mix sample already used by the H123 and H129 gates. No arena contact. A true identity expansion changes the function not at all; anything above 1e-4 means the expansion recipe is wrong and the arm would be measuring a re-initialisation, not depth
- **(b) Gradient reachability.** Run 200 optimizer steps on the banked teacher targets at the contract recipe and measure mean |Δw| in the 12 inserted blocks against mean |Δw| in the 22 original blocks. **LICENSE** at ratio >= 0.25; **KILL** below - identity-initialised blocks that receive under a quarter of the gradient movement will stay inert over one epoch and the arm measures nothing. This is E5's own stated risk for the 34L move ("identity-initialised copies risk staying inert over one epoch") converted into a 0.4 GPU-h measurement

Absent either precondition the line dies for ~0.4 GPU-h and the H129 lane proceeds unchanged at 307M.

### Pre-registered bar

Both arms read blind through the frozen gate, 2 draws each under H126 seeded pairing. Define Δ307 = (307M student pair mean - 0.70311) and Δ367 = (367M student pair mean - 0.70311).

- **CAPACITY BINDING (primary verdict)**: Δ367 - Δ307 >= **+0.010** with sign agreement on both paired draws -> FM5 answered affirmative at the operating point; the 40L arm is licensed
- **CAPACITY NOT BINDING**: |Δ367 - Δ307| < **0.005** -> FM5 closes upward; every H129 outcome attributes to transmission, and the within-family parameter axis is retired from this campaign
- **UNRESOLVED** between 0.005 and 0.010
- **SHIPPING clause**: the 367M student is ADMITTED if its pair mean >= **0.7091** (H129's own registered bar) with both draws >= control, gold_full pair mean >= 0.84, no subset < 0.55, and no subset more than 0.06 below control. At 367.12M it is inside the 400M cap and is a legitimate deliverable, not a diagnostic-only object
- **Subset clauses**: finqa >= **0.5912** (1 sigma hold below its 0.6333 clean pair mean) AND pubmedqa >= **0.5846** (1 sigma hold) - E3 shows these two are the axis any real change travels along, at r = -0.84. delucionqa reported, barred from adjudication (E2, E3 (d): zero of 14 configs move it past 2 sigma)

### Legality

- **Contamination**: none. The teacher targets are the two frozen H105 draws' probabilities over the clean 685,670-row public mix; no arena data, no private gold, no RAGBench derivative
- **Cap**: 367.12M < 400M, verified by arithmetic from the checkpoint in section 0.1. The 40L follow-on at 397.21M is also inside the cap, with 2.79M of headroom
- **Ruling 7**: the recipe constants (batch 48, MAX_LEN 512, OneCycleLR 1 epoch, lr 1e-5) are unchanged. Depth is a model-shape change, not a batch or accumulation change, so the contract is honoured on its stated terms - but the author should confirm, since a 1.55x stack changes step time and therefore wall-clock queue arithmetic
- **Closed-line check**: this is an amendment to a live registered lane (H129), not a re-proposal. Weight-space averaging (H118 soup, H120 EMA) stays closed - this is output-space distillation, the sole surviving route, which the record explicitly names as such. Head fusion, token-head-as-primary and layer-mix head input are untouched
- **Trainer contract inherited from H129**: the distillation trainer MUST assert `key_hash` alignment before consuming targets (the mix has no materialized parquet; positional order is the key)

### Cost

~18 GPU-h full: 2 draws at the 1.55x stack (~8.5 GPU-h each) + ~1 GPU-h reads + 0.4 GPU-h gate. **A 1-draw pilot at ~9 GPU-h** is the recommended shape - spend draw 2 only if |Δ367 - Δ307| on draw 1 exceeds 0.005, i.e. only if the contrast is live. Teacher, targets and gate are already paid, so this is pure marginal cost on top of H129.

---

## L5-C4 - EMBEDDING-BUDGET REALLOCATION (is the 400M cap constraining compute at all?)

**Named mechanism**: separate the *parameter budget* from the *capacity budget*. 64.05% of the shipped model is a lookup table, not compute. This tests whether that block is load-bearing for the deliverable, and therefore whether the 400M cap constrains capacity at all.

### The capacity mechanism being tested

The cap is 400M parameters. The model spends 196.6M of it on a 256,000 x 768 tied embedding table and 110.3M on the transformer stack (section 0.1). Every capacity proposal in this campaign has been priced against the 400M total, which means every one of them has been competing for the 93M of headroom left over *after* the lookup table takes its 64%.

A vocabulary trim to 64k frees **147.46M** at no architectural cost: 64,000 x 768 = 49.15M against 196.61M. Under the same cap that funds a transformer stack of up to ~350M - roughly **3.2x** the current stack - or, at the comparable 34L dose, a 219.7M total model with the same stack as L5-C3's 367.12M arm at 60% of the budget.

The arm proposed here deliberately does **not** spend the freed budget. It is the isolation experiment: **trim the vocabulary to the gate-determined size, keep 22 layers, change nothing else.** Same stack, same data, same schedule, same seed facility; only the lookup table shrinks. One variable.

- If the trimmed model reads within noise of the clean control, then 147M of the 400M budget is proven dead weight, every future capacity proposal in this campaign re-prices at up to 3.2x stack for the same cap, and the shipped deliverable simultaneously gets 2x smaller at no measured cost
- If it costs 0.020 or more of blind mean, the tail vocabulary is load-bearing, the cap really is 64% lookup, and every upward capacity move stays priced against 93M of headroom

Neither outcome has ever been measured in this campaign, and E5 records the vocabulary as the largest untouched lever under the cap with no in-repo precedent.

**Why the trim is nearly free by construction**: retained rows are *exact copies* of the pretrained embeddings - no repair pretraining is needed for any token that survives. The tokenizer is unchanged; only the ID space is remapped, and tokens outside the kept set map to UNK. So the entire cost of the intervention is the OOV *occurrence* rate, which the kill-gate measures directly and for free. This is what distinguishes the trim from E5's rank-128 factorization variant (item 5), which imposes a rank bottleneck on every input and does need repair training - that variant is deliberately **not** proposed here.

### Kill-gate (ZERO GPU, ~20 minutes CPU)

Tokenize the full 685,670-row clean public mix with the shipped tokenizer and count token-ID **occurrences** (not types), broken out by all 12 DANN groups. Repeat on a held-out multilingual sample covering PsiloQA's 14 languages. Then find the smallest vocabulary V (by descending occurrence frequency, with all special tokens forced in) that covers **>= 99.5% of token occurrences in every group individually** - the worst group binds, so no language can be quietly sacrificed to a pooled average.

- **LICENSE** at V <= 64k -> frees >= 147.46M; run the arm at V
- **RE-PRICE** at 64k < V <= 128k -> frees 98-147M; the arm is still worth running, at the reduced dose, with the freed budget restated
- **KILL** at V > 128k -> the trim frees under half the leverage that motivates it, the multilingual tail is genuinely broad, and the line dies for zero GPU-h

Second free clause: assert that the retained set contains every token appearing in >= 0.001% of occurrences in *any single* RAGTruth language group. This is the multilingual-integrity check; the deliverable requires 8 RAGTruth languages plus PsiloQA's 14, and a trim that quietly guts one script would show up here rather than in a blind read.

### Pre-registered bar

The trim is a mix-wide, register-blind change, so a mean bar is legal per ruling 7.

- **BUDGET VERDICT (primary)**: trimmed-vocab pair mean >= **0.69311** (within 0.010 of the 0.70311 clean pair mean) with sign agreement on both draws -> the 147M embedding block is DEAD WEIGHT under the cap; every future capacity proposal re-prices at up to 3.2x stack, and the 40L-at-64k configuration (249.8M total, 200.6M stack, 1.82x the current stack at 62% of the current budget) is licensed as the next capacity arm
- **KILL**: pair mean <= **0.6831** (-0.020) -> the tail vocabulary is load-bearing; the cap is genuinely 64% lookup and stays that way
- **Holds**: gold_full pair mean >= 0.84; **RAGTruth non-EN in-domain AUC >= 0.82** on re-read (the same multilingual hold H123 carried) - this is the real risk of this hypothesis and it gets an explicit clause; no subset < 0.55; no subset more than 0.06 below control
- **Subset clauses**: finqa >= **0.5912** (1 sigma hold) with pubmedqa >= **0.5846** (1 sigma hold) reported as its E3 guardrail. The mechanism predicts no directional subset effect at all - a large single-subset move under a register-blind lookup-table change would FALSIFY the attribution and is recorded as such, in the pattern of H122's registered concentration clause
- delucionqa reported, barred from adjudication (E2)

### Legality

- **Contamination**: none. The trim is computed from the clean training mix only. No arena text, no RAGBench corpus, no private gold enters the vocabulary decision - and this matters: a vocabulary fitted to arena text would be arena-fitted preprocessing and would void the lever under ruling 2's logic
- **Cap**: the trimmed model is 159.5M total (49.15M embeddings + 110.33M stack), far inside the cap
- **Ruling 7**: batch, MAX_LEN, schedule and optimizer are unchanged. The model shape changes only in the embedding rows
- **Deliverable risk, stated plainly**: multilingual coverage is a product requirement (8 RAGTruth languages in training, PsiloQA's 14, and a multilingual serving promise). The gate's per-group 99.5% clause and the bar's RAGTruth non-EN >= 0.82 hold are the two places that risk is caught. If either fires, the line dies rather than shipping a quietly English-biased model
- **Not on the closed list**: vocabulary budget has never been proposed in this campaign in any form. It is not weight averaging, head fusion, token-head-as-primary, curriculum, lambda sweep, canonicalization, distractor negatives, layer-mix, forced balance, or RAGBench training

### Cost

**Gate: 0 GPU-h** (~20 minutes CPU). Arm: ~12 GPU-h - 2 draws at ~5.5 GPU-h (the stack is unchanged, so step cost is unchanged to first order; the smaller embedding gather is a marginal saving) plus ~1 GPU-h of reads.

---

## Sequencing and what the four together answer

Ordered by cost-to-first-decision, not by expected gain:

| order | hypothesis | first decision costs | what a null closes |
|---|---|---|---|
| 1 | **L5-C4** | **0 GPU-h** (CPU gate) | whether the cap constrains compute at all |
| 2 | **L5-C2** | ~2-3 GPU-h (frozen read) | the 512-token bottleneck as an explanation for techqa / finqa |
| 3 | **L5-C1** | 0.5 GPU-h gate, then ~12-14 | FM5 **downward** - monotone saturation below 307M |
| 4 | **L5-C3** | 0.4 GPU-h gate, then ~9 pilot | FM5 **upward** at the operating point, on a target proven to exist |

**L5-C1 and L5-C3 bracket the capacity curve.** A CAPACITY-CLOSED reading from C1 (a 2.615x stack cut costs under 0.010) together with a CAPACITY-NOT-BINDING reading from C3 (a 1.55x stack increase on a known-better target buys under 0.005) answers FM5 from both directions and retires the within-family parameter axis from this campaign for ~21-23 GPU-h. That is the deliverable the author asked for - the capacity question ANSWERED - and it is cheaper than the single upward arm the record has been carrying below the cut since Round 12.

**L5-C4 is the one that changes the question rather than answering it.** If 147M of the budget is dead weight, every capacity number in this campaign - including C1's and C3's - was measured under a cap that was never really binding on compute, and the next capacity arm should be 40L at 64k vocabulary (249.8M total, 1.82x stack) rather than 34L at 256k vocabulary (367.12M total, 1.55x stack). It costs zero GPU-h to find out.

**What this lens does NOT propose, and why**: ModernBERT-large (English-only, forks the multilingual deliverable, voids the 12-group DANN design, already killed in-record on attribution invalidity); XLM-R-base and mDeBERTa-v3-base (measured capacity *downgrades* at 85-86M stack, and mDeBERTa diverged to NaN twice in R7-H50); Qwen3-Reranker-0.6B (595.8M, breaches the cap, parked); deeper or wider task heads (H123 measured layer 22 as already the best readout; a bigger head cannot add what the trunk does not carry); mixture-of-heads and any head-fusion variant (CLOSED - H102 / H104 / H106); rank-128 embedding factorization (speculative, needs repair pretraining whose cost cannot be priced from anything in this repo - L5-C4 tests the same budget question without that liability); conditional computation / per-group capacity routing (it is a mixture of experts by another name, it multiplies the serving contract, and the H123 probe measured no task-vs-DANN capacity competition to route around: domain-group accuracy 0.9525 / 0.9370 against 0.0833 chance sits alongside task AUC 0.964 in the same 768 dimensions).

---

## Artifacts consulted and re-verified in this session

- `models/R9-H105-mmbert-dann-clean/trunk/model.safetensors` + `config.json` - total 306,939,648; embeddings 196,608,768 (64.05%); 22 layers; d=768, intermediate 1152, `max_position_embeddings` 8192, RoPE theta 160,000, `sans_pos`, vocab 256,000
- `~/.cache/huggingface/hub/models--jhu-clsp--mmBERT-small/.../model.safetensors` + `config.json` - total 140,897,536; transformer stack **42,188,928**; d=384, 22 layers, vocab 256,000
- `experiments/grounding-semantic/R12-H121_gateA_scores.parquet` (77,171 rows) tokenized with the shipped tokenizer - exposure table of section 0.2
- `R13_anchor_teacher_result.json` - pair mean 0.70311, anchor 0.72067, delta +0.01756, bar 0.70811, verdict OPEN
- `R12-H123_layerprobe_result.json` - layer-22 margins +0.0007 / -0.0011, group accuracy 0.9525 / 0.9370 at layer 22 against 0.0833 chance
- `R13-H129_gate_result.json` - LICENSE; median |p1-p2| 0.01248, frac >= 0.10 = 14.44%, draw correlation 0.9775
- `src/groundrails/semantic_ov.py:36` - `MAX_LEN = 512` in the shipped serving path; `R8-H101_windowed_read.py:46-47` - `WIN = 1500`, `STRIDE = 750`
- `docs/experiments/semantic-grounding-experiments.md` - lines 92, 280, 1068-1081, 1340, 2485-2503 (R12 author rulings, below-the-cut ranks), 2519-2531 (R13 session rulings 7 and 9), 2607-2630 (H129 registration, gate result, queue amendment)
- Evidence packs `R14_evidence_E1_finqa.md`, `R14_evidence_E2_delucionqa.md`, `R14_evidence_E3_covariance.md`, `R14_evidence_E4_items.md`, `R14_evidence_E5_capacity.md`

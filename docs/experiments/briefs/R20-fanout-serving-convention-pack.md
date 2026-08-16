# R20 fanout brief - trained-through serving-convention evidence pack + author decision memo

Fable hypothesis-design agent (read-only, 2026-08-16). AUTHOR DECISION ITEM - nothing here is registered; the R20-H175 draft inside is reserved for the author.

All evidence gathered. Assembling the pack.

---

# EVIDENCE PACK: TRAINED-THROUGH SERVING-CONVENTION ARM (question conditioning + pool concatenation)

Read-only research product, 2026-08-16. All paths under `/home/lab/workspace/private/ai-assistants/groundrails/`. Canonical log = `docs/experiments/semantic-grounding-experiments.md` (line numbers cited as `:NNNN`).

## 1. The evidence trail, reconstructed with citations

### 1.1 The H151 ruling - exact text and exact scope

The verdict block (`:3281-3288`), header verbatim:

> **R18-H151 - VERDICT: serving-side pooling route CLOSED; max is load-bearing AND minimal-spread; the variance attack concentrates on training-side levers**

Operative clauses:
- H151c (`:3285`): "vacuous FAIL (selected == max; spread shrink 0% vs the >= 30% bar). Per the registered branch: **max stands as PRIMARY read; no serving-read amendment**"
- Consequence (`:3286`): "**the serving-side lever class is exhausted for variance**; the attack rides entirely on H152... The max read is now doubly evidenced: best mean AND lowest spread among poolings"
- The registered branch it executed (H151c registration `:3174`): "FAIL -> max stands, **variance attack moves to the training lever (EMA) only**"

**Scope analysis - what the ruling actually closes.** By its own words H151 adjudicated exactly one thing: the choice of POOLING OPERATOR over per-window scores (max vs top-2/top-3/top-10% mean vs logsumexp), evaluated as post-hoc reads on frozen checkpoints, framed as a VARIANCE lever (H151b `:3173`). It says nothing about presentation (windowing geometry, concatenation, question conditioning) and nothing about trained-through changes. Three pieces of post-H151 campaign practice confirm the narrow reading:

1. **R18-H156** (`:3290-3292`, registered 2026-08-13, AFTER H151, author-ordered): a learned aggregator replacing max, TRAINED-THROUGH ("loss = serving read = s_agg", twin protocol). Registered, run, killed at draw 1 on its merits (−0.0250, and the failure was the training gradient, not the read - `:3300ff`). If H151 bound trained-through serving-read changes, H156 could never have been registered. It was.
2. **R19-H165** (registered 2026-08-14): a post-hoc PRESENTATION change (concatenation) on a frozen checkpoint, registered and adjudicated "against the supersession pattern used for R8-H101" (`:3530`). So even post-hoc presentation changes were not treated as closed by H151 - they were treated as open pending their own blind read.
3. The H165 correction block (`:3594-3607`) records that a further concatenation-adjacent READ (union-of-windows) "needs an author licence, not a coordinator decision" - i.e. the closure of post-hoc presentation reads dates from the **H165 blind kill**, not from H151.

**Conclusion on scope**: brief B's line ("a door the H151 ruling closed", `docs/experiments/briefs/R20-sweep-B-hagrid-emanual-options.md:35,61`, echoed at `:3862`) is an over-broad gloss. What is actually closed, and by what: (a) post-hoc pooling swaps - closed by H151; (b) post-hoc global concatenation - closed by the H165 blind KILL (`:3553`); (c) content-gated concatenation - closed by the H170 pre-read kill (`:3594`); (d) further concatenation-adjacent post-hoc reads - author-licence-gated per the H165 correction. **A TRAINED-THROUGH presentation change is bound by none of these texts** - its direct precedent is the H142→H150 windowed-MIL protocol, which was itself a trained-through presentation change and is the current flagship. What makes this an author decision anyway: the coordinator already escalated it as one (`:3862`), and question conditioning touches the shipped product API (Section 4.2).

### 1.2 R19-H165 ladder - full results (`:3512-3533`, artifacts `experiments/grounding-semantic/R19-H165_ladder_{L0,C0,L1,L2,L3,L4}_R18-H150-arm-draw1.json`)

| cell | presentation | WIN/MAX_LEN | gold_full | vs L0 | win/item | pairs |
|---|---|---|---|---|---|---|
| L0 | per-chunk (banked) | 1500/512 | 0.8659 | control | 44.91 | 123,579 |
| C0 | pool-concatenated | 1500/512 | **0.9014** | **+0.0355** | 77.98 | 214,615 |
| L1 | concatenated | 3600/1024 | 0.8823 | +0.0164 | 32.20 | 88,606 |
| L2 | concatenated | 7200/2048 | 0.8514 | −0.0145 | 15.88 | 43,711 |
| L3 | concatenated | 14400/4096 | 0.7912 | −0.0747 | 7.75 | 21,337 |
| L4 | concatenated | 28800/8192 | 0.7272 | −0.1387 | 3.69 | 10,140 |

Blind arena verdict (`:3553-3577`): KILLED, draw 1 −0.01163, draw 2 −0.01599; per-subset replication: finqa −0.1332/−0.1318, hotpotqa +0.0939/+0.0972 (both to ≤0.0033 - a mechanism, not noise). The "tables" causal story was WITHDRAWN (`:3594-3607`); the replicated per-subset structure stands unexplained. Standing lessons appended to the record: (i) "gold_full is not a valid selection surface for any presentation change that interacts with document structure" (`:3571`); (ii) gold vs arena disagreed by 0.049 in sign and magnitude (`:3605`). Corrections wave caveat (3) (`:3684`): C0 reads 14.6% more chars per window and 1.74x more pairs than L0 - **spanning and more-text are confounded; no ladder cell separates them**.

### 1.3 The H142 mechanism note (`:3067-3073`)

> "The 0.72498 read has a single mechanism - removing a train-serve mismatch carried since the windowed read shipped (H101)... the model was only ever trained on document-initial text, then asked at serving to judge mid-document windows it had never seen."

Fix = (i) untruncated evidence, (ii) training presentation == serving presentation (1500/750 windows), (iii) MIL max-over-windows BCE. Doctrine consequence recorded there: "the windowed-MIL training protocol supersedes 1,500-char truncation as the presumptive recipe." This became the H150 flagship (+0.022 arena over the clean control era). **This is the campaign's constitutional precedent that aligning training presentation to a serving presentation is the highest-yield lever class it has.**

### 1.4 The H101 supersession pattern (`:1948-2001`)

How a serving-formula change was legally promoted: pre-registered claim with per-subset clauses; the read KILLED on its registered conjunction (finqa clause) but the diagnostic gain (+0.0142) triggered a **conditional supersession recorded BEFORE the deciding read**: "the windowed decomposed-min read becomes the PRIMARY read if and only if it also dominates on the second draw" (`:1979`). It did (+0.0180), supersession CONFIRMED, all subsequent checkpoints read windowed. Template: deterministic reads, replication across independent draws, condition written before the deciding measurement.

### 1.5 The H119 idempotent-transform precedent (`:2482`, verdict `:2579`)

Author ruling 2 (2026-08-08): "an idempotent, subset-blind text transform before tokenization is inside the frozen-read boundary (R8-H101 precedent), **conditional on shipping in the library serving path identically for every corpus and every future input**. A transform retained because it helps one arena subset is arena-fitted preprocessing and voids the lever." (Empirically H119 was then REFUTED in both directions - checkpoint-idiosyncratic sign flips.) Relevance here: the legality frame requires the convention to ship in the library path uniformly - which is exactly where question conditioning is currently in tension (Section 4.2).

## 2. The trained-through arm - design

### 2.1 Candidate presentation (the minimal trained-through change)

- **Training row** = `[question?] [SEP] claim-sentence` vs windows of the **concatenated evidence pool** (concatenation in document order as given; separator `"\n\n"`; windows at the banked 1500/750 over the joined pool, exactly the C0 geometry). MIL max-over-windows BCE unchanged; min-over-sentences response read unchanged; MAX_LEN stays **512** (see 2.3). Everything else the H150 recipe verbatim.
- **Serving/arena read** = identical presentation: question (when present) + sentence vs windows of the concatenated pool, per-sentence max, min over sentences. Subset-blind: the rule is uniform for every input; no subset identity, no content gate, no threshold. This satisfies the serving-legality shape at `:3176` / `:3583`.
- **gold_full and all holds are read under the SAME presentation** (the twin doctrine: loss presentation = serving presentation; H165's kill was precisely a checkpoint trained one way and read another).

### 2.2 Question-field census of the training mix (verified against the raw parquets this session)

| corpus | rows (window census) | question field | note |
|---|---|---|---|
| vitaminc | 370,653 (54%) | **none** | `claim`/`evidence` only |
| tabfact | 92,585 (13.5%) | **none** | statement/table |
| ragtruth_en | 15,090 | **`query`** - yes | verified in `dataset-ragtruth.zip` |
| ragtruth translated ×7 | 105,630 | no clean field | `prompt` (used AS evidence) already embeds the question - implicitly conditioned today |
| psiloqa | 61,712 | **`question`** - yes | verified |
| halueval | 40,000 | qa half yes (`question`), summarization half no | builder currently drops it (`R9-H105_clean_mix.py:119-137`) |
| quant_misbind / scale_unit | 35,540 | none (generators) | templates could synthesize one; that is a lane change, not free |

**~13-14% of rows carry a clean question** (ragtruth_en + psiloqa + halueval-qa ≈ 97k of 721k). The design must therefore be OPTIONAL-field robust: empty-question rows train with a bare `[SEP]` prefix (or omitted segment), exactly matching the shipped no-question serving mode. Honest risk, stated: with 86% empty-question rows the model may learn to ignore the question channel entirely - the mechanism's supervision is thin, and hagrid/emanual (where the question is hypothesized to matter) have no in-mix analogue teaching question RELEVANCE. A question-conditioning contrast lane (same evidence, right vs wrong question) does not exist and is not registered.

### 2.3 Token budget - 512 vs 8192, said precisely

The trunk supports 8192 (`max_position_embeddings`, `:3464`). H165 L1-L4 showed longer windows degrade **on a 512-trained checkpoint** - the registration pre-called that decline as train-serve mismatch and named "retraining at length" as the only follow-on (`:3468`, `:3524`). **The ladder says nothing about an 8192-trained checkpoint.** But retraining at length is a SECOND variable with a quadratic-attention cost multiplier and no existence proof on either side; the minimal arm holds MAX_LEN 512 / WIN 1500 (the geometry with the +0.0355 existence proof) and changes only concatenation + question. Truncation policy: question capped (e.g. 256 chars) and never truncated in favour of the window; window tail-truncated at tokenizer level as today; the question cost at 512 is ~15-40 tokens (~5-8% of budget) - measurable, small, and identical across items.

### 2.4 Serving cost - measured, and it runs AGAINST the arm at 512

From the banked ladder artifacts: at 512, concatenation-then-slide produces **1.74x the pairs** (214,615 vs 123,579 on gold_full; 77.98 vs 44.91 windows/item) because the slide runs continuously across the joined pool (`:3532`). The tasking's assumption that windows per item drops is true only at longer MAX_LEN (L1/1024: 32.2/item = 0.72x; L2/2048: 15.9), which is the two-variable arm this draft does not propose. Ship cost of the minimal arm: **+74% inference per item**, plus a new read script (C0 read + question prefix; `R19-H165_concat_read.py` is 90% of it).

## 3. Pricing

### 3.1 Predicted effect - the honest range and its anchors

- **+0.0355** - in-domain existence proof (C0, our checkpoint, no question). Discounted: gold_full is formally discredited for presentation changes (sign-flip 0.049, `:3605`).
- **+0.155/+0.169** - incumbent cross-convention gain on hagrid/emanual (`R19-H171_incumbent_native.json`); its whole-mean convention effect is **+0.0336** - and NEGATIVE on tatqa (−0.088), delucionqa (−0.091), finqa (−0.103).
- **−0.012 to −0.016** - the counter-anchor: the same presentation, un-trained-through, killed blind (H165).
- **+0.014 to +0.022** - what train-serve alignment historically converts a presentation into (H101 formula effect +0.0134-0.018 across 3 draws; H142 twin +0.022 arena).
- **New computation (this pack)**: per-subset, the incumbent's convention delta and OUR concat delta correlate at **Spearman 0.806, Pearson 0.732, 8/10 sign agreement**. The two disagreements: covidqa (small) and **hagrid (incumbent +0.155, ours −0.003)**. Reading: concatenation reproduces most of the convention effect's SHAPE on our model - including the finqa/tatqa/delucionqa loss side - while **hagrid's gain is NOT delivered by joint pool alone on our checkpoint**; it must live in the question/template component, which is exactly the component with no test on our model and 14% training supply. (This softens brief B's "joint pool answers hagrid's source_select by construction".)

**Honest prediction: +0.005 to +0.020 mean, wide; central +0.012.** Upside case: alignment converts the concat redistribution into net gain (H142 pattern) and the question channel buys part of hagrid/emanual. Downside case: the replicated finqa/tatqa/delucionqa loss persists through training (the redistribution is about information, not mismatch) and the mean lands negative - the H165 kill, paid at 26 GPU-h instead of 2.4.

### 3.2 Bars under the frozen variance protocol (amendment V1, `:3846-3856`)

Pooled per-draw sd **0.01090** (frozen); floors 2×SE_diff vs the k=6 flagship headline: k=2 → 0.0178, k=4 → **0.01407** (the H174 formula, `:3871`). At the central prediction +0.012, k=2 is inadmissible (floor above prediction); **k=4 declared**, resolving only in the top half of the prediction - same honesty note H174 carries.

### 3.3 Registered-arm draft (for the author to accept, amend, or refuse)

- **ID**: R20-H175 (next free), CONVENTION-PARITY TRAINED-THROUGH ARM
- **Claim**: because the incumbent's entire advantage on hagrid/emanual is its serving convention (+0.155/+0.169, R19-H171), because pool concatenation is worth +0.0355 in-domain on our own checkpoint (H165 C0) and its blind kill is attributable to train-serve mismatch (per-doc-trained checkpoint; H142 precedent: alignment converts presentation into gain), training the flagship recipe with the served presentation - optional-question prefix + document-order pool concatenation, 1500/750/512, MIL unchanged - will lift the blind k-draw mean.
- **Single variable**: presentation only. Mix, DANN 14 groups, LR/schedule, seeds protocol - H150 verbatim. Question from each corpus's native field where it exists, empty otherwise.
- **Prediction**: +0.005..+0.020 (central +0.012)
- **PRIMARY**: k=4 mean ≥ k=6 flagship mean + 0.01407 → PROMOTE (and the serving convention ships with the checkpoint - they are one object)
- **TABLE GUARD** (the measured failure mode): finqa/tatqa/delucionqa each within one across-seed spread of the flagship subset mean (0.062/0.025/0.012 - H174's detector, `:3872`), on the 2-draw mean
- **MECHANISM GATES** (report-bearing): hagrid and emanual vs their flagship k-draw means (0.6424/0.678) - if hagrid does not move, the question channel is dead and the H171 escalation is answered negatively regardless of the mean
- **KILL**: draw-1 arena mean < k=6 flagship mean − 0.0218 (the k=1 floor) → remaining draws unspent; any table-guard breach on the 2-draw mean → same
- **HOLDS**: gold_full ≥ 0.84, non-EN ≥ 0.82, anti-gaming ≥ 0.7438, all read under the arm's own presentation
- **Cost**: 4 × ~6.5 GPU-h = **~26 GPU-h** (draw-1 kill caps downside at ~7); read tooling ~0 (adapt `R19-H165_concat_read.py`); serving cost +74% pairs/item at 512 if promoted
- **Decomposition option, stated for the author**: the two components are separable. Concat-only trained-through (no question, no API change) is the cheaper, better-evidenced half; question-only is the untested half that owns the hagrid hypothesis. A 2-component sequence costs more draws but each verdict is attributable. The bundled arm buys the incumbent-parity answer in one purchase and cannot attribute a partial result.

## 4. Steelman AGAINST

**4.1 Why the closures existed.** H151's rationale: max was measured load-bearing AND minimum-spread; serving-formula churn adjudicated on arena reads is the exact shape of fishing that the H141 discipline exists to prevent - and the campaign now carries 48 uncorrected blind reads with the flagship as their maximum (escalated defect, `:3692`). Presentation arms are 0-for-3 since (H165 killed, H170 killed pre-read, H156 killed at draw 1). Architecture/read interventions overall are 0-for-5 (`:3492`); every promotion has come from data or trained-through presentation - this arm leans on the one exception class that has worked, but the base rate is against anything touching the read.

**4.2 The deployment mismatch is real and is a PRODUCT decision.** `grep -rn question src/groundrails/` returns no API surface: `ground()` / `ground_batch()` (`src/groundrails/grounding.py:930,1732`), the semantic tier (`semantic.py`), and the joint head (`joint.py`) verify claim-vs-sources with **no question parameter anywhere**. The H119 legality frame requires the convention to "ship in the library serving path identically for every future input" - a question-conditioned model either forces an API change (optional `question=` field; RAG callers do have the query, so this is plausible, but it is the author's product call) or ships permanently in empty-question mode, in which case the question half of the training was decoration and only concat is live. The architecture report (`reports/research-grounding-architecture.md:229`) also pins "serve at fixed 512" - consistent with the minimal arm, violated by any retrain-at-length variant.

**4.3 The gain may evaporate - or invert - on the 8 subsets we win.** The convention delta table (computed from `R19-H171_incumbent_native.json`, native − harness):

| subset | native | harness | conv. delta | our concat delta (2-draw mean) |
|---|---|---|---|---|
| emanual | 0.7694 | 0.5999 | **+0.1695** | +0.0544 |
| expertqa | 0.8098 | 0.6503 | +0.1595 | +0.0162 |
| hagrid | 0.7542 | 0.5992 | **+0.1550** | **−0.0031** |
| pubmedqa | 0.6070 | 0.5162 | +0.0908 | +0.0255 |
| hotpotqa | 0.6161 | 0.5976 | +0.0185 | +0.0955 |
| techqa | 0.6536 | 0.6363 | +0.0173 | +0.0112 |
| covidqa | 0.7432 | 0.7355 | +0.0077 | −0.0058 |
| tatqa | 0.5275 | 0.6156 | **−0.0881** | −0.0988 |
| delucionqa | 0.7018 | 0.7929 | **−0.0911** | −0.1008 |
| finqa | 0.6137 | 0.7170 | **−0.1033** | −0.1325 |

The convention HURTS the incumbent on finqa/tatqa/delucionqa - the three subsets where our harness-convention wins are largest (tatqa +0.1812, techqa +0.0972, finqa is our only loss but our 0.6825 beats the incumbent's own-convention 0.6137). Our concat deltas track the incumbent's convention deltas at rho 0.806 - strong evidence the redistribution is a property of the CONVENTION, not of the mismatch, in which case training through it trades our fortress subsets for the weak ones and the mean nets near zero. And the one subset the whole escalation is about - hagrid - is the one where the transferable component (concat) measurably does nothing on our model.

**4.4 Thin question supervision.** 86% of training rows have no question; nothing in the mix teaches question RELEVANCE (no wrong-question contrast lane). The mechanism could train to a no-op while still charging its token budget.

**4.5 Opportunity cost.** R20-H174 (portfolio lane arm, registered `:3869-3874`) targets the same two subsets at +0.012..0.020 predicted, k=4, with lanes already built and censused (`R20-H174_lane_{L1,L2,L4}_census.json`) - and it needs no API change and no serving-cost increase. Two arms with overlapping targets and identical bar shapes should be sequenced, not parallel; if H174 delivers, this arm's marginal case shrinks.

## 5. AUTHOR DECISION MEMO (one page)

**Question.** The incumbent's entire advantage on our two losing subsets is its serving convention - question conditioning + joint passage pool (hagrid +0.155, emanual +0.169; R19-H171). Post-hoc adoption of the pool half on our checkpoint gained +0.0355 in-domain and was killed blind (H165: −0.012/−0.016, train-serve mismatch). Do we buy a TRAINED-THROUGH arm that trains the flagship recipe under the served convention?

**What is actually closed.** H151's text closes post-hoc POOLING swaps only (max stands; "the variance attack moves to the training lever"). The H165 blind kill closes post-hoc concatenation; H170's gate kill closes content-gated concatenation. **No ruling binds a trained-through presentation change** - the flagship itself (H142→H150 windowed-MIL, +0.022) is one, and H156 (trained-through read change) was legally registered after H151. Brief B's "door the H151 ruling closed" is an over-broad gloss; the door needs your decision because the coordinator escalated it and because question conditioning changes the shipped API, not because a ruling's text forbids it.

**Evidence FOR.** (1) The campaign's only repeat-winning lever class is train-serve alignment (H101 +0.013..0.018; H142 +0.022). (2) C0's +0.0355 in-domain existence proof on our own weights. (3) The incumbent's +0.0336 whole-mean convention effect. (4) The H165 kill is exactly the signature alignment fixes: a checkpoint trained per-doc read concatenated.

**Evidence AGAINST.** (1) Our concat deltas track the incumbent's convention deltas at Spearman 0.806 (8/10 signs) - the finqa/tatqa/delucionqa losses (−0.10..−0.13, replicated both draws) look like a property of the convention, not the mismatch; we would be trading fortress subsets for weak ones. (2) hagrid - the headline target - is the one subset concat does NOT move on our model (−0.003); its hoped-for gain rests entirely on the question channel, which is untested and supervised by only ~14% of training rows. (3) The shipped `groundrails` API has no question parameter; promotion forces a product decision. (4) Serving cost +74% pairs/item at 512. (5) Presentation arms are 0-for-3 since H150; H174 already targets the same subsets more cheaply.

**Registered-arm draft** (Section 3.3): R20-H175, flagship recipe verbatim with optional-question + document-order pool concatenation at 1500/750/512 trained AND served; prediction +0.005..+0.020 (central +0.012); k=4, PRIMARY ≥ k=6 flagship mean + 0.01407; table guard finqa/tatqa/delucionqa (0.062/0.025/0.012); kill at draw 1 < k=6 mean − 0.0218; holds read under the arm's own presentation; ~26 GPU-h ceiling, ~7 at draw-1 kill.

**Recommendation.** Do not buy the bundled arm now. Sequence it: (1) let R20-H174 (already registered, same targets, no API change) spend first; (2) if the residual after H174 still runs through hagrid/emanual, buy the CONCAT-ONLY trained-through half (no question, no API change, the component with an existence proof) as its own k=4 arm with the same table guard - its verdict is cleanly attributable and it answers whether alignment rescues the replicated table losses; (3) buy question conditioning only with an API decision in hand and a question-relevance contrast lane designed, because without both it is either unshippable or untrained. If you want the incumbent-parity answer in one purchase regardless, the Section 3.3 draft is safe to register as written - its kill gates cap the downside at one draw.

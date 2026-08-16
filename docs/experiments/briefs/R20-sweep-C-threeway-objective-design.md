# R20 sweep brief C - three-way objective arm design (R19-H166 family)

Subagent research brief (read-only, 2026-08-16). Input to the Round 20 adjudication; the coordinator adjudicates.

Standing: the three-way arm is ALREADY REGISTERED (log ~line 3475, author-assented 2026-08-14) as head -> Linear(768,3), supported-logit serving scalar, aux CE on the MIL-argmax window, PRIMARY bar flagship +0.005 at k=2. That registration is internally inadmissible under the adopted variance protocol (bar below the detection floor) and must be amended before launch.

## (a) Attachment points (all through `R16-H142_G1_arm.py`; new arms reuse it via the H150 rebind pattern)

- Head: `R16-H142_G1_arm.py:190` `task_head = nn.Linear(d, 1)`; per-window logit path `pair_logits` (:213-218), adapter term exactly 0 in the flagship
- Loss loop :598-607: trunk encode -> `logits_from_cls` -> MIL `scatter_reduce amax` -> BCE (:604) + DANN CE (:605-606)
- y3 entry: `R19-H166_labels3.build(tags, y)` returns int8 positionally aligned; batches are row-index lists, so aux target = `y3[batch]`, mask `m = y3_batch >= 0`, guard `if m.any()` (a packed batch can land all-masked); loss = t_loss + d_loss + LAMBDA_AUX * aux_loss, LAMBDA_AUX fixed before training
- Serving read: `score_sets` (:328-351) max per sentence-set; `R16-H142_G1_reads.py:103-105` min over sentences; any option keeping `pair_logits`' scalar semantics unchanged needs ZERO reader change
- Decisive geometry fact: the two labelled groups train at mean 1.0 windows/row (vitaminc max 5, quant_misbind exactly 1.0) - window-attribution is moot in TRAINING and live only at SERVING (4-22 windows), the same train-serve OOD shape that sank H140/H141/H156

## (b) Options

| | (A) separate aux 3-way head | (B) replace head with 3-way (registered form) | (C) ordinal cumulative-link | (D) two-head: support + contradiction |
|---|---|---|---|---|
| Wiring | keep Linear(768,1); add head3 = Linear(768,3), CE on labelled rows | task_head -> Linear(768,3); supported logit is the MIL scalar | scalar head + two learned thresholds | task_head untouched; add con_head = Linear(768,1), MIL max-over-windows BCE vs 1[y3==contradicted] |
| MIL semantics | aux has none (per-window CE) | max over support logit only - a contradicting window elsewhere never lowers the score | max HIDES contradiction (bag with contradicted+absent maxes to absent) | CORRECT: contradiction is a bag-level any-window event; max is the right aggregation and handles row-level labels |
| Serving | scalar unchanged | head/checkpoint/reader all change | scalar unchanged | scalar byte-identical; tri-state as a SEPARATE channel |
| Risk to banked binary | low | medium (softmax coupling on the arena's own scalar; 0-for-5 architecture prior) | low-medium (ordinal constraint may fight score geometry) | LOWEST |
| Params | +2,307 | +1,538 | +2 | +769 |

## (c) Evidence audit - is contradiction-vs-absence confusion costing arena AUROC?

Speculative, and the banked evidence leans AGAINST:
- pubmedqa (the one subset with a taxonomy): contradiction is 6.1% of unsupported sentences (n=10) and is the ONE family where the model already beats bag-of-words (+0.1463) - "the only semantics installed is negation/polarity, which is what VitaminC teaches". The failing families (inference_not_stated 37%, aim_vs_finding 16%, paraphrase FNs) are absence-type
- hotpotqa: residual is partial-support saturation/composition; no polarity signal
- historical: H9 (lexical era) rejected the NLI 3-way channel at +0.004/0.005 inside fold noise
- no artifact attributes any arena loss to contradiction/absence conflation
- FOR: gradient-mass argument (199,939 rows carry identical targets on 27.7% of the mix); the author's capability ask (tri-state output) has deployment value independent of the arena mean

## (d) Recommended amendment (option D)

- Claim: a parallel contradiction MIL head on the recovered labels (400,653 rows, masked elsewhere) installs a contradicted-vs-absent distinction the binary objective cannot express, without degrading the binary serving scalar
- Prediction: arena mean delta in [-0.005, +0.008], below every affordable floor - the arm is EXPLORATORY on the mean. Mechanism prediction (the real falsifiable contrast): held-out VitaminC REFUTES-vs-NEI AUROC on the new channel >= 0.85, vs near-chance for the flagship's binary scalar on the same split (baseline leg read off the banked R18-H150 checkpoint BEFORE training - cheap)
- Bars: PRIMARY = the mechanism gate; GUARD = non-inferiority of the 2-draw arena mean at the k=2 floor; HOLDs carried (gold_full >= 0.84, non-EN >= 0.82, anti-gaming diagnostic); superiority contingency only at the corrected floor
- Fallback: arena null EXPECTED - if mechanism passes and guards hold, the contradiction head ships as an auxiliary output on the unchanged flagship scalar (tri-state API) and the arena-gain claim is recorded dead
- lambda_aux = 0.2 on the masked-mean BCE, fixed before launch
- Cost: label build sunk; baseline read minutes; 2 draws ~ 13 GPU-h

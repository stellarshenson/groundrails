# Grounding the RAG assistant - model-based experiments

**Canonical Experiments Document**

The full grounding investigation for the a production RAG assistant: can a non-LLM grounder catch hallucination in real answers, which signal does it best, and how should it be used. The arc ran in four research phases - an adversarial probe (where deterministic lexical won), building a verified gold from production traffic, a signal comparison on that gold (where lexical flips and the cross-encoder wins), and a score-stacking meta-classifier that beats any single signal without fine-tuning - followed by deployment consolidation (two cross-encoders, single-engine OpenVINO int8) and two hypothesis rounds (H9-H11, one adoption: the reranker-first cascade; H12-H14, two adoptions: the pre-filter cosine gate and the early-exit reranker, taking the warm claim to 662 ms at slightly better quality). This is the model-based counterpart to the deterministic lexical track in `lexical-grounding-experiments.md` / `lexical-grounding-sota.md` (this repo); both run on the same private gold and reach comparable macro-F1 from opposite mechanisms. Final design in `semantic-grounding-sota.md`.

> Gold/meta figures are the 2,752-record run (organic-majority). Probe-phase figures are stable (size-independent).

## Intent

**One small fast model that decides whether a claim is supported by its sources, replacing the three-model cascade - and it must beat every comparable public model, decisively, on every corpus we can measure.** That is the target this log exists to reach, and every hypothesis is judged against it.

**The win condition, as of round 8** - a single model **under 400M parameters** that is strictly above `KRLabsOrg/lettucedect-v2-mmbert-base` (307M, MIT) on all three corpora at once, by a margin that is not arguable:

| corpus | bar to beat | decisive margin | status |
|---|---|---|---|
| private gold, 159 held-out traces | 0.7095 | ≥ 0.76 | **held** - our distilled 307M student reads 0.8479 |
| RAGTruth EN, 1,200 responses | 0.7039 | ≥ 0.75 | open - our cascade reads 0.6432 |
| RAGTruth non-EN, 7 languages parallel | 0.6095 | ≥ 0.66 | open - our cascade reads 0.5626 |

A margin of +0.05 is roughly 20x the measured run-to-run noise (0.0023 across three identical trainings), so it is a real separation rather than a tie dressed up as a win. Beating one corpus is not a result: a domain-specialised model already takes the first and a public-trained model already takes the other two.

- **The deliverable** - a single model, **under 400M parameters** (raised from 350M in round 8, which brings mmBERT-base at 307M and the incumbent at 307M both in-band and makes the comparison size-fair), taking (claim, evidence chunk) → supported or not; it replaces the whole shipped stack, not one stage of it
- **Distillation is the backbone**, fine-tuning the optional second stage, anisotropy removal the third family - see round 8
- **Speed is the reason** - the incumbent runs ~662 ms/claim warm on CPU by loading three models (a ~568M bi-encoder over every chunk, a ~568M cross-encoder over the top-8, a ~278M NLI model). One model over the top-3 chunks is ~3 forwards where the cascade does ~60
- **Parity is the accuracy target, not a beat** - the shipped stack reads macro-F1 0.824 and both the training-recipe and architecture research lanes independently put the sub-350M ceiling at **0.80-0.84**. A distilled student cannot exceed its teacher, so the case for building is speed and simplicity at held quality
- **Wide and shallow, not deep and narrow** - measured at matched parameters and FLOPs, narrow-deep is 5.0x slower eager and gives up a third of achieved throughput (R7-H52). Depth's quality edge in the literature is measured on parametric recall, the one axis a grounder never uses because its evidence arrives in the input
- **Inference and deployment are deliberately deferred** - GPU is fine for now; the torch-free OpenVINO int8 CPU path is a later problem and must not filter candidates today
- **Honest measurement is a precondition, not a formality** - macro-F1 0.824 is measured on our own private gold with our own labels and has never been calibrated against anything public. Until R7-H49 lands it may not be quoted as a comparable number

> **ERRATUM - 2026-07-29, the "shipped cascade" macro-F1.** Rounds 5-8 repeatedly write "the shipped cascade, macro-F1 **0.824**". That is wrong and the log's own Phase D and E records say so: **0.824 is the six-model score-stack plus a lexical contradiction flag**, which was the best model measured and was NOT deployed. What ships is the two-cross-encoder consolidation at **macro-F1 0.796 out-of-fold**, and **0.789 end-to-end** once the round-1 and round-2 skip mechanisms are applied. Every comparison written against 0.824 therefore set the bar roughly 0.03 too high, against ourselves. Impact on recorded verdicts: **none reverse.** Rounds 7 and 8 were adjudicated on AUC bars (reranker 0.8619, incumbent 0.7095 / 0.7039 / 0.6095 / 0.6461), which are unaffected. R6-H45's "loses to the shipped cascade by -0.313" should read -0.278 against 0.789 and remains a loss. The individual lines are left as recorded per the append-only rule; this note is the correction, and 0.789 / 0.796 are the figures to use from here.

**Standing rules this log enforces**, each earned by a failure recorded below: every hypothesis is pre-registered here before it runs; every new scorer passes a positive control on trivially separable pairs before it scores anything real; every candidate's declared prompt format is audited against its own vocabulary before inference; and a vendor's published number is a claim until someone else reproduces it.

## Situational overview

The same lexical NOT_FOUND rule that wins on the small adversarial probe collapses on the verified gold - real supported claims restate the documentation in new words and score as "not found". The lever that survives on paraphrased answers is the cross-encoder reranker, which scores claim-vs-evidence relevance directly; stacking the model scores then beats any single one.

- **Probe set (true data)** - 25 adversarial bait questions through the dev assistant, grounded against the exact chunks it retrieved; 33 factual claims, 6 gold hallucinations. The agent refused 16/25 on its own (strong refusal discipline); only q25 carried a confident fabrication
- **Verified gold** - real prod traffic, dual-judge (Haiku + Sonnet) agreed labels; `{claim, source_text, label, lang, user_id, trace_id}`, English-dominant retrieved-doc evidence per claim (~57 KB). Grew 375 → 856 → 1,260 → 1,686 → 2,631 → 2,752 (organic expansion); several conclusions changed with size
- **Few independent contexts** - claims sharing a trace's evidence are correlated, so the effective sample is smaller than the record count
- **The flip** - on the probe set lexical NOT_FOUND is the most sensitive cheap detector (5/6, ~22-26% false-flag); on the gold it inverts (~85% false-flag) because supported paraphrases share no wording, and the cross-encoder becomes the signal

## Executive summary

A cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) is the best single grounding signal on the verified gold; a logistic over the six model scores plus a lexical contradiction flag beats it on macro-F1 and cuts both error types. No model is fine-tuned - only a decision hyperplane is fit on top.

**Hypothesis ids** - one global ascending `H<n>`, never reset; the `R<round>-` prefix is a memory slug, the number is the identity. Rounds 3 and 4 were first written with round-local ids (`R3-H1..H7`, `R4-H1..H7`) and were remapped once, on 2026-07-28, to `R3-H15..H21` and `R4-H22..H29` (`R4-H2b` → `R4-H29`, created last). No verdict, result or prediction changed in that remap; experiment scripts, logs and score artefacts were renamed to match. Round 5 continues at `R5-H30`.

**Research at a glance** - the full sweep across the four phases (detail in the sections below; the original per-experiment reports are archived under `../@archive/`):

| Phase | Experiment / hypothesis | Dataset | Key result | Conclusion |
|---|---|---|---|---|
| A probe | Adversarial probe capture | 25 bait Q, 33 claims | 16/25 refused, 1 confident hallucination (q25), 6 gold hall | strong refusal discipline; assertion-vs-disclaimer is the real signal (raw NOT_FOUND overstates ~15:1) |
| A probe | Lexical grounding vs no-grounding | probe 33 | 0% → 83% recall (5/6), 22% false-flag, ~55 ms/answer | **lexical works** as a cheap NOT_FOUND/CONTRADICTED gate |
| A probe | Lexical-only vs lexical+semantic | probe 33 | lexical 5/6, 26% ff, 28 ms; +semantic 4/6, 30% ff, 647 ms | **semantic dropped** - over-confirms + ~23-40x latency |
| A probe | Bayesian calibrator + NLI layer | probe 33 | calibrator 1-2/6; NLI raw 3/6, verdict 4/6 (44% ff) | **dropped** - 6 negatives cannot calibrate; NLI over-flags |
| A probe | Feedback-loop prototype (1 revise) | probe 3 answers | 6/6 gold hallucinations fixed | **loop, not gate** - revise/retract beats blocking |
| A probe | Synthetic graded benchmark dataset | 6 base → 24 variants | 0/20/40/60% ungrounded levels | scaffold; real prod groundedness is bimodal, not graded |
| B data | Verified gold (the production trace store, dual-judge) | 375 → 2,752 | dual-agreed gold; rate 37% → 29% as organic grows | the labelled set the probe lacked |
| B data | Test-user concentration | gold 2,752 | one QA cohort = 79% of hallucinations | filter by `user_id`; organic rate ~10% |
| C signal | Lexical match-type on the gold | gold 2,752 | ~85% false-flag, AUC ~0.5 on paraphrases | **refuted here** - inverts the probe-set win |
| C signal | Bi-encoder cosine (e5, bge-m3, mmBERT) | gold 2,752 | AUC 0.53-0.73; over-confirms in-domain | **weak** - topical similarity ≠ grounding; bge-m3 best at 0.73 |
| C signal | NLI cross-encoder (mDeBERTa-mnli-xnli) | gold 2,752 | AUC 0.81; raw gate over-flags | **kept as a feature** - strong ranker, bad raw gate |
| C signal | Cross-encoder reranker (bge-reranker-v2-m3) | gold 856→2,752 | AUC 0.82 → 0.84, best single signal | **kept** - the relevance scorer is the lever |
| C signal | gte-multilingual-reranker | gold | custom-kernel CUDA crash | **dropped** - replaced by standard-arch models |
| D meta | Score-stack: logistic vs GBM over 6 scores | gold 2,752 | OOF AUC 0.91, macro-F1 0.82 | **linear ships** - GBM ~ ties within noise |
| D meta | Lexical numeric/entity contradiction flag | gold 2,752 | weight +0.18 (small) on this set | **kept** - cheap, catches spec edits the rerankers miss |
| E deploy | 2-cross-encoder consolidation ablation | gold 2,752 | macro-F1 0.796 with {reranker, NLI} only | **ships** - within ~1 fold-std of the full 0.814 |
| E deploy | NLI replacement search (MiniLM-L6/L12, XLM-R) | gold 2,752 | stacks 0.758-0.765 vs 0.796 | **rejected** - mDeBERTa uniquely strong for grounding |
| E deploy | mDeBERTa int8: ORT vs OpenVINO SmoothQuant | gold 2,752 | ORT parity 0.29-0.75; OV SQ 0.984, stack 0.795 | **SmoothQuant ships** - 318 MB, -0.001 macro-F1 |
| F round 1 | H9 - NLI contradiction/neutral channels | int8 pairs 111,800 | +0.004/+0.005 macro-F1 | **rejected** - inside fold noise; gold hallucinations are omissions, not contradictions |
| F round 1 | H10 - aggregation beyond max-over-chunks | int8 pairs 111,800 | -0.005..+0.005 macro-F1 | **rejected** - max already extracts the distribution's signal |
| F round 1 | H11 - reranker-first confidence cascade | int8 pairs + latency bench | 61% NLI skips at macro-F1 0.795; warm mean -28% | **adopted** - thresholds only, no training |
| G round 2 | H12 - pre-filter cosine gate (stage 0) | int8 pairs + cosine cache | 22% of claims skip both cross-encoders, FP 245/FN 216 vs 248/217 | **adopted** - strictly fewer errors, zero added compute |
| G round 2 | H13 - rank-ordered early-exit reranker | latency bench n=150 | mean 4.8/8 pairs scored; verdicts exactly equal | **adopted** - verdict-invariant by construction |
| G round 2 | H14 - fused-evidence single-forward cross-encoders | fused cache 2,752 x 2 x 2 | macro-F1 0.714-0.784 (-0.012..-0.081) | **rejected** - max-over-chunks is load-bearing |
| H round 3 (pre-reg) | R3-H15..H21 - tiny fine-tuned faithfulness checker (persona fanout) | gold 2,752 + 111,800-pair cache | pre-registered predictions/bars; governed by the R3-H15 zero-shot-AUC gate | **pending** - see Hypothesis round 3 |
| I round 4 | R4-H22 - representation gate (8k SYNTH vocabulary over our evidence) | golden_v6 EN 4,876 | fertility 1.196x incumbent, context overflow 0.3% | **PASS** - a 30x smaller vocabulary costs only 20% more tokens |
| I round 4 (pre-reg) | R4-H23..H28 - knowledge-free reasoning models as the grounding head (SYNTH / Needle / SAN transfer) | gold 2,752 EN slice + golden_v6 | pre-registered; governed by the R4-H24 method test | **pending** - see Hypothesis round 4 |
| I round 4 | R4-H29 - generative-judge positive control (20 trivially separable pairs) | 20-case control | Baguettotron 12/20, Monad 4/20; 0 false negatives, 6 false positives | **closed** - zero-shot generative judging fails for this model class |
| J round 5 | R5-H30 - the prose residue is unmeasured (adjudication gate) | private prose set, 32 unconfirmed in-scope claims | 26/32 correct refusals (81.2%); true recall 28/34 = 82.4% not 46.7% | **Refuted (the framing)** - the metric was wrong, not the grounder |
| J round 5 (pre-reg) | R5-H31..H39 - prompt and elicitation engineering for small-model judgement (persona fanout) | 20-case control + gold EN slice | pre-registered; re-aimed by R5-H30 | **pending** - see Hypothesis round 5 |
| K round 6 | R6-H47/H46 - vocabulary and budget pre-flight | 3 candidates, CPU only | Monad 0/4 template tokens in vocab; no candidate overflows context | **Kept as a standing gate** / H46 **Null** |
| K round 6 | R6-H40 - reproduce Monad's published MMLU | MMLU 2,000 stratified | 0.255 / 0.226 / 0.143 across three protocols vs published 0.30, chance 0.25 | **Killed-at-gate** - Monad cannot be validated, withdrawn |
| K round 6 | R6-H42 - Pleias-RAG-350M as a judge (harness reproduced the vendor example) | 20-case control | 11/20 (55%), 9 false positives | **Refuted** - zero-shot judging closed on a validated instrument |
| K round 6 | R6-H45 - the quote is the verdict (model quotes, code decides) | 20-case control + gold EN 600 | control 18/20, FP 9→1; gold macro-F1 0.511 vs shipped 0.824 | **Refuted for deployment**, kept as a finding - recall 0.267 |
| K round 6 | R6-H43 - the trained refusal path as a verdict | gold EN 600 | macro-F1 0.476, 240 FP on 300 negatives | **Refuted** |
| K round 6 (pre-reg) | R6-H41 - EOS-as-BOS ablation | 20-case control | anomaly confirmed by R6-H47; ablation unrun | **pending** |
| L round 7 | R7-H51 - regenerate the teacher corpus with text | gold 2,752 → 123,579 pairs | reranker AUC 0.8289 vs 0.841 reference (-0.012); NLI drifts -0.039 | **PASS** - ships as the teacher, reranker channel only |
| L round 7 | R7-H52 - encoder shape at matched parameters (ran before registration) | GPU shape bench, B=3 T=512 | narrow-deep 5.0x slower eager; attention-only +18% FLOPs and slower | **Kept as a measurement** - width wins, SAN closed on latency |
| L round 7 | R7-H50 - capacity ablation, 140M-307M distilled from R7-H51 | 159 held-out TRACES, 717 claims | mmBERT-base 0.8479 &gt; mmBERT-small 0.8269; teacher 0.8619; claim-level split leaked 0.050 and inverted the ordering | **Refuted** - the task IS capacity-limited (spread 0.021) |
| L round 7 | R7-H50 depth probe - mmBERT-base truncated 22L/11L/6L/3L | same held-out traces | 22L 0.8502 @ 9.2 ms → 11L 0.8183 @ 5.38 ms | **in progress** - depth costs 0.032 AUC, buys 1.7x |
| L round 7 | R7-H57 - public-trained verifier on our gold | 159 held-out traces | control AUC 1.0000; our gold 0.7095 vs our 0.8619 | **PARTIAL** - public data does not transfer to us |
| L round 7 | R7-H59 - cross-domain matrix, both models on both corpora | gold + RAGTruth 1,200 | ours 0.8619→0.6432 off-domain; theirs 0.7095↔0.7039 flat | **both directions fail** - our number is domain specialisation |
| M round 8 | R8-H61 - are the machine-translated labels reliable? | 300 human-verified DE rows | EN→DE drop 0.0999 (MT labels) vs 0.0680 (verified) | **gap is REAL** - ~1/3 label noise, 2/3 capability |
| M round 8 | R8-H64 - ensemble headroom, ours vs the public verifier | 10 corpora, per-example scores | Spearman -0.046..+0.083 everywhere; fusion 0.7479 EN / 0.6212 non-EN | **orthogonal** - fusion beats both untrained, but is 875M |
| M round 8 | **R8-H62 - multi-corpus distillation, one 307M student** | gold + RAGTruth EN + 7 translations | **gold 0.8531 / EN 0.8434 / non-EN 0.8407 vs 0.7095 / 0.7039 / 0.6095** | **WIN 3/3 DECISIVE** - RAGTruth cells clean, gold caveated |
| M round 8 | R8 leak audit + `R8_splits.py` | 2,752 claims, 4 split units tried | only a connected-component split over shared chunks reaches 0 overlap; 1 component holds 92% of the corpus | **our gold cannot measure generalisation** - ~39 independent units |
| M round 8 | R8-H83 - diversity without the test set (HaluEval + PsiloQA, RAGBench out) | blind arena, 10 subsets | blind mean 0.6161 vs incumbent 0.6461; +0.0205 over H62, still -0.0300 | **REFUTED** - generic diversity buys a fifth of in-domain; failure is negative-class, not domain |
| M round 8 | R8-H84 - near-miss negatives (VitaminC, RAGBench still out) | blind arena, 10 subsets | blind mean 0.6450 vs incumbent 0.6461; +0.0289 over H83, -0.0011 short, 7/10 won | **Kept, not a win** - passes its own bar, ties but does not beat the incumbent; finqa/tatqa residual is coverage |
| M round 8 | R8-H81 - GroupDRO worst-domain loss (H84 mix, loss aggregation only) | blind arena, 10 subsets | blind mean 0.6103, worst subset 0.4425; in-domain 3/3 decisive but gold 0.8177 &lt; 0.84 guardrail | **REFUTED** - q collapsed onto 2 seen groups (97%), starved the rest; worst-seen does not predict unseen |
| M round 8 | R8-H79 v1 - DANN domain-adversarial (H84 mix, lambda 0.1) | blind arena, 10 subsets | blind mean 0.6320, 6/10 won; emanual +0.1150, first tatqa win; discriminator anti-predicts (0.000 vs chance 0.083) | **Refuted at v1 point** - redistributes rather than lifts; complementarity with H84 motivates H88; v2 lambda 0.02 queued |
| M round 8 | R8-H85 - context coverage (windowed inference) | length probe, arena docs | Spearman(deficit, hidden mass) -0.128 p=0.73; techqa longest docs yet won, tatqa shortest yet lost | **Killed-at-gate** - truncation does not explain the residual; content, not coverage |
| M round 8 | R8-H86 - prose-parity (lettucedetect-prose) | provenance check, train parquet | 78,882 rows = 63,792 psiloqa + 15,090 ragtruth, nothing else | **Killed-at-gate** - repackaging of two upstreams already in the mix; H83's exclusion confirmed |
| M round 8 | **R8-H88 - ERM+DANN ensemble, unweighted mean (diagnostic)** | blind arena, 10 subsets | **blind mean 0.6470 vs incumbent 0.6461, 7/10 won, none below chance** | **Confirmed** - FIRST blind result above the incumbent; residual is variance, not capability; feeds H89/H90/H91 |
| M round 8 | **R8-H92 - decomposed scoring: min over sentences of max over chunks** | blind arena, 10 subsets | **ens 0.6893, H84 alone 0.6820, H79 alone 0.6856 vs incumbent 0.6461; 7/10 won** | **CONFIRMED** - largest lift of the campaign, formula only; first legal SINGLE-model blind beats; min becomes the primary read |
| M round 8 | R8-H94 - soft aggregation tuned on RAGTruth (softmin tau 2.0, frozen shot) | blind arena, 10 subsets | 0.6613 vs pure min 0.6893; delucionqa clause confirmed (+0.0814), 8 subsets regressed | **REFUTED** - aggregation preference is corpus-dependent; RAGTruth not a valid proxy for it; min stays primary |
| M round 8 | R8-H73 - two-head trunk (score + token span, fused) | 3 in-domain corpora + blind arena | gold 0.8843 RECORD (token head 0.8896); blind whole 0.6366, decomposed-min 0.6607 (7/10) | **Kept** - in-domain champion, blind mid-pack; double-AND over-sharpens; token head best-ever tatqa 0.7013 |
| M round 8 | R8-H97 - three-member decomposed-min ensemble (+H73) | blind arena, 10 subsets | 0.6871 vs bar 0.6920 and H92's 0.6893; 8/10 subsets won | **REFUTED** by its own branch - third member dilutes the mean; H92 two-member stands |
| M round 8 | R8-H93 - DANN lambda geometry under LOCO(HaluEval), Optuna TPE 22 trials | LOCO sweep, 60k subsamples | ERM 0.6278 ranks 19/22; winner lam 0.1241 LOCO 0.7418 (+0.1140 vs bar +0.02), dom-acc 0.001 | **Confirmed on lift, geometry clause refuted** - anti-predictive band holds the peak at high variance; lambda* feeds H96 |
| M round 8 | **R8-H90 - full-corpus DANN, one 307M student (762k pairs, 13 groups, lam 0.02)** | 3 in-domain corpora + blind arena | **blind decomposed-min 0.7213, 8/10 won (+0.0752); whole 0.6538; gold 0.8418, 3/3 DECISIVE** | **CONFIRMED - new ladder holder; single model beats the H92 ensemble by +0.0320; attribution pending H91** |
| M round 8 | R8-H91 - full-corpus ERM control (identical 762k mix, no discriminator) | 3 in-domain corpora + blind arena | blind decomposed-min 0.6965, 8/10 won; whole 0.6462; gold 0.8576, 3/3 DECISIVE | **Confirmed as control** - attribution resolved: objective +0.0248, data +0.0145; DANN wins 8/10 subsets vs its ERM twin, largest on the residual losses |
| M round 8 | R8-H98 - gated ensemble H90+H91 | gate on H91 solo read | H91 solo 0.6965 < gate 0.7013 (within-0.02 of holder) | **Killed-at-gate** - the H97 dilution boundary applies; no arena shot spent; H90 stands alone |
| M round 8 | R8-H95 - lift-all-groups GroupDRO, curriculum stage 1 (smoothed q, stratified batches) | 13 group-val sets + blind arena | 13/13 groups lifted, group-val mean 0.955, TabFact 0.524→0.782; blind min 0.6870 (6/10) vs ERM twin 0.6965; q pinned uniform | **Mechanism clause confirmed, blind clause refuted** - starvation fixed (gap to ERM 3.7x narrower than H81's) but forced balance still costs blind; trunk delivered to H96 |
| M round 8 | R8-H96 - phase shift: GroupDRO-mastered trunk → DANN lambda* 0.1241 (the curriculum) | 13 group-val sets + blind arena | blind min 0.6820 (8/10) vs bar 0.7313, H90 0.7213, own stage 1 0.6870; dom-acc parked ~0.50 predictive; group-val rose to 0.9610 | **REFUTED on both clauses** - generalisation-from-mastery does not beat single-stage DANN; stage 2 undid stage 1's tabular mastery blind (finqa 0.7053→0.6417); lambda-vs-trunk attribution goes to H99 |
| M round 8 | R8-H99 - full-corpus DANN from scratch at lambda 0.1241 (single-variable vs H90) | 3 in-domain + blind arena | blind min 0.6913 (8/10) vs bar 0.7313; gold 0.8435 3/3 DECISIVE; no anti-prediction at full scale (dom-acc ~0.46); finqa 0.7135 and delucionqa 0.7757 campaign bests | **REFUTED** - lambda 0.1241 costs -0.0300 vs 0.02; curriculum deficit attributed 3/4 lambda + 1/4 trunk; LOCO-HaluEval invalidated as a lambda proxy; high lambda conquers the far registers but pays on the strongholds |
| M round 8 | **R8-H100 - variance probe: verbatim H90 replicate** | 3 in-domain + blind arena | **blind min 0.6918 vs H90's 0.7213 - gap 0.0295 on an identical recipe**; gold 0.8511 3/3 DECISIVE | **Demotion clause FIRES** - full-scale run-to-run noise ~±0.03; all round-8 single-run training deltas ≤ 0.03 demoted to within-noise; campaign moves to multi-seed means (see amendment) |
| M round 8 | **R8-H101 - windowed evidence on frozen H90 weights (un-truncate the chunks)** | blind arena, deterministic | **windowed mean 0.7355** (+0.0142 on identical weights); delucionqa +0.0809 flips to a WIN (0.8072 vs 0.7929); finqa -0.0019; zero-exposure subsets exact no-ops | **KILLED on the finqa clause** - truncation attribution refuted for finqa (numeric derivation is its real mode); highest deterministic blind read of the campaign; supersession conditional on the replicate windowed read |
| L round 7 (pre-reg) | R7-H49 - external calibration on LLM-AggreFact | 11 subsets, stratified | predicts 63-72 vs our private-gold ~82-85 | **blocked** - gated dataset, needs Hub auth |
| L round 7 (pre-reg) | R7-H53 - MICE-style split encoder, document side cached | teacher corpus | 0.298 ms vs 2.707 ms measured; AUC deficit unknown | **pending** - needs R7-H50 |

- **Best single signal** - `bge-reranker-v2-m3`, **AUC 0.841** (out-of-fold), macro-F1 0.757; decisively above every bi-encoder (~0.53-0.73) and the raw lexical rule (~0.5)
- **Best model** - a decision hyperplane over the six per-model scores + a lexical contradiction flag: **out-of-fold AUC 0.913, macro-F1 0.824** (vs 0.757 best single, 0.417 majority baseline), no fine-tuning
- **Reduces both errors** - at the macro-F1-optimal threshold it cuts total errors 533 → 408 (-23%): false-negatives 295 → 160 (false-positives ~flat 238 → 248)
- **GBM ≈ the logistic** - depth-2 boosting reaches a near-tie OOF AUC (0.913 gbm vs 0.903 logreg, within ±0.012 std); the linear hyperplane ships for simplicity and a calibrated probability
- **Driving metric is macro-F1** - the 1,966/786 imbalance makes accuracy misleading (majority predictor: macro-F1 0.417, hallucination-F1 0.000); FP and FN counts are the operational target
- **Residual** - 160 missed hallucinations (102 carry a number/spec) and 248 false-positives sit in the overlap region a fine-tuned cross-encoder would target - the deferred lever

## Results summary

Per-signal separation on the 2,752 gold (out-of-fold AUC, higher = better; the cross-encoder reranker is the lever, the bi-encoders and lexical rule are weak).

| signal | kind | OOF AUC |
|---|---|---|
| bge-reranker-v2-m3 | cross-encoder rerank | **0.841** |
| mDeBERTa-v3-mnli-xnli | NLI cross-encoder | 0.806 |
| bge-m3 | bi-encoder | 0.730 |
| e5-small | bi-encoder | 0.635 |
| e5-large | bi-encoder | 0.621 |
| mmBERT-base | bi-encoder | 0.529 |
| lexical match-type | rule | ~0.50 |

Best single signal vs the score-stack vs the majority baseline (2,752 gold, at each model's macro-F1-optimal threshold). The stack lifts macro-F1 and cuts total errors, almost all of the gain in recall (missed hallucinations).

| metric | best single (bge-reranker) | meta-classifier (stack) | majority baseline |
|---|---|---|---|
| OOF AUC | 0.841 | **0.913** | 0.50 |
| macro-F1 | 0.757 | **0.824** | 0.417 |
| FP - supported flagged | 238 | 248 | 0 |
| FN - hallucination missed | 295 | **160** | 786 |
| total errors (of 2,752) | 533 | **408** | 786 |

## Experiment record - results across data growth

The conclusion held as the gold grew. macro-F1 climbed 0.817 → 0.824 and the score-stack beat the best single signal at every size - the stability across a 3.2x data increase (and a base-rate shift) is the best evidence the result is real, not a snapshot artefact.

| gold | records | hall rate | best-single AUC (bge-reranker) | meta AUC (OOF) | meta macro-F1 | errors cut vs best single |
|---|---|---|---|---|---|---|
| doubled | 856 | 36% | 0.86 | 0.91 | 0.817 | - |
| tripled | 1,260 | 37% | 0.857 | 0.907 | 0.821 | 269 → 210 (-22%) |
| organic-majority | 2,752 | 29% | 0.841 | 0.913 | 0.824 | 533 → 408 (-23%) |

- **macro-F1 stable** - 0.817 → 0.821 → 0.824 across the growth and a 37% → 29% base-rate shift
- **stack > best single at every size** - the combination is not an artefact of one snapshot
- **single-signal AUC drifts down slightly** (0.86 → 0.841) as more diverse organic paraphrases enter - harder data, but the stack compensates and macro-F1 still rises
- **base-rate caveat** - the 2,752 set is organic-majority (the QA-test cohort filtered down in influence), so its 29% rate and raw FP/FN counts are not directly comparable to the earlier test-heavy sets; macro-F1 and AUC are the size- and rate-robust metrics
- **not re-fit at 1,686 / 2,631** - intermediate sizes were data milestones; the model was re-scored and re-fit at 856, 1,260, and 2,752, which bracket the growth

## Phase A - adversarial probe (where lexical won)

Before a labelled gold existed, the grounder was tested end-to-end on real production answers to bait questions, judged per claim by a local `claude -p`.

- **Refusal discipline** - 16/25 hallucination-bait questions correctly refused (non-existent products VMS V400 / T700, fabricated versions, network details, part numbers); only q25 carried a confident fabrication (an invented T300 "100 m" cable limit + a fabricated cable type), caught by NOT_FOUND + CONTRADICTED
- **Lexical lifts capture from 0% to 83%** - the no-grounding baseline ships everything and catches 0/6; lexical NOT_FOUND/CONTRADICTED catches 5/6 at claim level and 3/3 at answer level, ~55 ms/answer
- **Semantic is worse here** - lexical+semantic catches 4/6 at 30% false-flag and ~647 ms/claim (~23-40x the latency); the bi-encoder over-confirms because in-domain topical similarity reads as support at the stock 0.6 threshold, and the Bayesian calibrator (bambi/PyMC) cannot recover the misses with only 6 negatives
- **NLI over-flags** - raw NLI catches 3/6 at 63% false-flag, grounder-with-NLI 4/6 at 44%; entailment is strict sentence-level while the assistant's answers are supported by tables/procedures
- **The 15:1 trap** - raw NOT_FOUND count overstates hallucination ~15:1 because the agent's own refusal sentences score NOT_FOUND; the real signal is assertion-vs-disclaimer, and the pipeline must gate on that
- **Real prod prose is ~94% grounded** - groundedness is bimodal (full answers ~0% ungrounded, short fallbacks ~100%), not a smooth spectrum, so a graded benchmark of 24 variants at 0/20/40/60% ungrounded had to be constructed by controlled injection
- **Verdict** - ship lexical-only as a cheap NOT_FOUND/CONTRADICTED gate with assertion-vs-disclaimer filtering; gate semantic behind calibration; the prerequisite for any learned/embedding layer is a larger balanced labelled set

## Phase B - building the verified gold

The probe's "6 negatives" ceiling forced a real labelled set. A golden dataset was built from production traffic, recipe in `../dataset/2026-06-01-how-to-build-golden-dataset.md`.

- **Pipeline** - scout prod traces → recover evidence from tool/rag spans → extract claims from the raw answer → lexical pre-pass → Haiku primary judge → tightened-prompt cleanup → Sonnet as the stronger second judge → keep only dual-agreed labels
- **Growth** - 375 → 856 → 1,260 → 1,686 → 2,631 → 2,752 records; several conclusions changed with size (a depth-2 GBT that won on 375 overfits later)
- **Test-user concentration** - one QA/test account drives 81% of the hallucinations at a 56% rate; organic users sit at ~10%, so the later batches exclude the test accounts and the gold carries `user_id` / `trace_id` to filter cleanly
- **Storage** - parquet (zstd) at ~2 MB, off Git LFS; the shared evidence blobs deduplicate columnar

## Phase C - signal comparison on the verified gold

With a real labelled set, the probe ranking inverts.

- **Lexical flips to failure** - NOT_FOUND fires on supported restatements (~85% false-flag, AUC ~0.5); the signal that word overlap measured on the probe (fabrications share no wording) is swamped by paraphrase on real answers
- **Bi-encoders over-confirm** - e5-small/large modest (AUC ~0.62-0.64), bge-m3 better (0.73), mmBERT near chance (~0.53); topical similarity to in-domain evidence is not support
- **NLI ranks well, gates badly** - mDeBERTa AUC 0.81 but raw verdict is 100% recall at 88% false-flag (over-flags)
- **Cross-encoder reranker wins** - bge-reranker-v2-m3 AUC 0.82 → 0.84 across the gold's growth, the best single signal, because it scores claim-against-evidence relevance directly
- **Run as a GPU notebook** - `notebooks/01-kj-grounding-model-comparison.ipynb` over six multilingual models, per-model subprocess isolation; gte-multilingual-reranker dropped after a custom-kernel CUDA crash

## Phase D - the score-stacking meta-classifier

Per the no-fine-tuning constraint: learn a verdict over the per-model scores.

- **Features** - 6 model scores (bge-reranker, mDeBERTa-NLI, bge-m3, e5-large, e5-small, mmBERT) + `lexical_fired` + numeric/entity `contradiction` flag
- **Heads raced** - a logistic with `StandardScaler` + L2, a depth-2 GBM, and each raw single signal; 5-fold stratified out-of-fold predictions remove model-selection optimism
- **Result** - OOF AUC 0.913 (gbm) / 0.903 (logreg), macro-F1 0.824 vs 0.757 best single, 0.417 baseline
- **Learned weights** (+ = supported) - bge-m3 +1.34, bge-reranker +1.14, lexical-fired +0.91, NLI +0.87; e5/mmBERT small negative (de-noising), contradiction +0.18
- **GBM ties** - no nonlinear interaction gain, so the linear hyperplane ships

## Methodology

Score every model independently on the gold, then learn a verdict over the scores; no learner sees the fold it scores.

- **Evidence chunking** - recursive 1,100-char chunks, 200 overlap, max 50; claim scored against each chunk, max taken (best-chunk relevance)
- **Bi-encoder** - cosine of claim and chunk embeddings (e5 `query:`/`passage:` prefixes, bge-m3 CLS, mmBERT mean); max over chunks
- **Cross-encoder rerank** - the reranker's single relevance logit per claim×chunk, sigmoid, max over chunks - models claim-vs-evidence directly, not via a shared embedding space
- **NLI** - entailment probability of (chunk → claim) per pair, max over chunks
- **Lexical features** - match-type (`lexical_fired`) and a numeric/entity `contradiction` flag (`find_mismatches`)
- **Verdict head** - logistic with `StandardScaler` + L2; GBM and raw scores raced against it
- **Metric** - macro-F1 headline, FP and FN counts at the operating point, AUC for ranking quality
- **Cross-validation** - 5-fold stratified out-of-fold; single-signal AUCs are pretrained so honest as-is, the meta-classifier re-estimated OOF
- **Per-language AUC** - where n ≥ 20 (en, nb-NO)

## Setup

- **Data** - `data/processed/golden_grounding_evidence_verified.parquet`; per-model scores cached to `data/interim/model_scores/*.npy`, index-aligned
- **Models** - e5-small, e5-large, bge-m3, mmBERT-base (bi-encoders); bge-reranker-v2-m3 (rerank), mDeBERTa-v3-base-mnli-xnli (NLI)
- **Hardware** - RTX 5090 (index 1 under `CUDA_DEVICE_ORDER=PCI_BUS_ID`), torch 2.12 + cu130; one model per subprocess so a CUDA assert cannot poison the others
- **HF quirks** - `HF_HUB_OFFLINE=1` (metadata stalls), `HF_HUB_DISABLE_XET=1` (Xet segfaults on large files), vault token; mmBERT/ModernBERT need `reference_compile=False`; gte dropped (CUDA kernel crash) - all captured in the `my-gpu` skill
- **Commands** - `python -m grounding_models <model>` (score one, isolated), `python -m grounding_ensemble` (fit + report)

## Deployment shape - two-stage verifier in a feedback loop

The grounder is a signal in a self-correction loop, not a hard gate (`../@archive/docs/grounding-feedback-loop.md`); a prototype revise fixed 6/6 gold hallucinations.

- **Stage A - fast pre-filter on every claim** - lexical (sub-ms) or the model stack produces candidate flags, disclaimers excluded
- **Stage B - precision gate on the flagged subset only** - one batched LLM-judge (or the score-stack probability) confirms unsupported vs paraphrase, so the agent is not told to rewrite claims that are fine
- **Loop** - on a confirmed flag the agent re-examines, re-retrieves, and revises or retracts (1-2 iterations); a confident fabrication has no supporting passage so it must be retracted - the correct outcome
- **Why a loop** - lexical/raw model gates false-flag paraphrases; letting the agent prove a flagged claim is supported (by citing the passage) avoids silently dropping genuine claims

## Model class: logistic vs GBM vs raw single signal

The decisive factor is the feature set (cross-encoder scores), not the fitting method.

- **Score-stacking logistic** - OOF AUC 0.903, macro-F1 ~0.82; the reranker + bge-m3 + NLI scores carry the signal
- **Gradient-boosted trees** (depth-2) - OOF AUC 0.913, a near-tie within std; no material nonlinear gain, so the linear model ships
- **Best single signal** - bge-reranker alone, macro-F1 0.757; the +0.07 comes from bge-m3 and NLI adding orthogonal evidence
- **Honesty on small data** - everything re-estimated out-of-fold; the full-set 0.82 carries selection optimism the OOF removes

## What we tried

- **Kept** - the bge-reranker cross-encoder (the lever), the mDeBERTa NLI score (strong ranker), bge-m3 embeddings (best bi-encoder), a logistic decision hyperplane, the lexical contradiction/fired flags; lexical-only as the cheap probe-stage gate; the two-stage verifier loop
- **Dropped / refuted** - raw lexical as a gate on paraphrases (over-flags on the gold), bi-encoder cosine as a verdict (over-confirms), raw NLI threshold (over-flags), the Bayesian calibrator on the probe set (too few negatives), gte-multilingual-reranker (CUDA crash), deep trees (no gain), e5/mmBERT as standalone signals (near-chance)

## Lessons learned

- **The signal flips with the distribution** - lexical NOT_FOUND wins on the probe and on the lexical track's omission-type cross-lingual task, but fails on paraphrased restatements where the cross-encoder wins; never carry a small-sample conclusion across distributions
- **Cross-encoder >> bi-encoder for grounding** - scoring claim-against-evidence beats a shared embedding space; topical similarity is not support
- **Raw model thresholds are mis-calibrated for the domain** - bi-encoders over-confirm, NLI over-flags; both are strong rankers (good AUC) but bad gates until a learned threshold/combination calibrates them
- **Stacking beats the best single signal cheaply** - a hyperplane over the scores adds +0.05 macro-F1 and cuts both error types with no fine-tuning
- **Macro-F1, not accuracy** - the imbalance makes accuracy read ~0.6 while macro-F1 is 0.39 for the majority predictor; the operational target is FP + FN counts
- **Model class barely matters here** - GBM ties the logistic; the gain was the cross-encoder features
- **Assertion-vs-disclaimer first** - raw NOT_FOUND overstates hallucination ~15:1; gate on assertions, not refusal sentences
- **A loop beats a gate** - paraphrase false-flags make a hard gate drop ~1-in-5 genuine claims; a revise loop lets the agent defend or fix
- **The labelled set is the unlock** - 6 probe negatives could calibrate nothing; the verified gold is what let any learned layer work, and growing it (and excluding the test cohort) is what makes the numbers trustworthy
- **GPU hygiene is load-bearing** - per-model subprocess isolation, HF offline, Xet disabled, dropping the gte custom-kernel model were prerequisites to clean scores
- **Complementary to the lexical track** - the engineered translate-then-recall lexical pipeline (sibling `lexical-grounding-sota.md`) reaches comparable macro-F1 from a torch-free CPU mechanism; different operating points, not a contradiction

## Deployment - single-engine OpenVINO int8

The deployable grounder runs on one runtime - OpenVINO int8 for all three models (bge-m3 pre-filter, bge-reranker, mDeBERTa NLI). Full quantization record in `deberta-v3-quantization-experiments.md`; end-to-end pipeline in `notebooks/03-kj-openvino-grounder-pipeline.ipynb`.

- **mDeBERTa int8 was the blocker, solved by SmoothQuant** - stock dynamic int8 broke (parity 0.35); NNCF SmoothQuant (alpha 0.7) reaches full-gold parity 0.9841 and stack macro-F1 0.795 (vs fp32 0.796) at 318 MB
- **ONNX-Runtime cannot quantize the NLI** - even a forked `onnx-neural-compressor` (three crash bugs fixed) stays faithless (parity 0.61-0.62); the single engine is OpenVINO, not ORT
- **All three int8 IRs hold parity** - bge-reranker 0.9976, bge-m3 0.9941, mDeBERTa-NLI 0.9863; sizes 571 / 570 / 318 MB (~1.46 GB total), push-ready under `models/ov/`
- **Top-k pre-filter improves quality** - on an 800-record subset, k=8 macro-F1 0.822 beats all-chunks 0.807; pruning noisy chunks before the cross-encoders helps the verdict, not just latency
- **Cache the source-chunk embeddings - the dominant latency lever** - the typical claim carries ~50 evidence chunks (median 50), and cold the pre-filter re-embeds all of them per claim (CPU ~4.2 s/claim median at k=8 under the LATENCY hint; `notebooks/04-kj-grounder-latency.ipynb`). Caching the chunk vectors (embed each unique chunk once, or reuse the RAG retriever's vectors) so the pre-filter only embeds the claim cuts the typical claim to **~1.2 s (median, 3.6×; p90 1.5 s)**. It also restores top-k as a real lever - warm k=8 is 5.0× faster than all-chunks (cold only 1.4×). Cosine over the ~50 chunks is a brute-force numpy dot-product; no FAISS/ANN needed at that scale
- **OpenVINO `LATENCY` hint - 2.1× free** - `compile_ir` had defaulted to `THROUGHPUT` (multi-stream, right only for the batch/offline path); for inline single-claim serving `LATENCY` is **2.1× faster** (cold 6365 → 3048 ms/claim at k=8 on a 64-thread CPU), measured in `scripts/bench_mechanical_levers.py`. Now the default - no quality cost
- **`max_length` cap does not help** - chunks run ~300 tokens median / 418 p95 and (claim, chunk) pairs ~331 / ~590 p95, so the 512 cap already truncates ~6.5% of pairs; capping to 256 saves only ~17% and clips the median pair. `MAX_LEN` stays 512. Length-bucketing the chunks before batching is the only padding win there (order-invariant, scores unchanged)
- **Whole-answer batching still open** - the serving helpers score per-claim (one padded forward per claim per model); batching an answer's claims × top-k into one forward would amortise overhead - the next mechanical lever to build
- **Portability** - x86-64 Intel/AMD native (AVX2 / AVX-512-VNNI); ARM/Graviton via the OpenVINO ARM plugin, less mature - validate on target

## Hypothesis round 1 - H9/H10/H11

Three mechanism-targeting hypotheses against the deployed int8 stack, all evaluated on the CPU OpenVINO int8 engines (a full per-pair score cache over the 111,800 gold (claim, chunk) pairs, `data/interim/model_scores/pairs/full_pairs.npz`); full ladder and final benchmark in `reports/grounding_hypotheses.md`, driver in `experiments/grounding-semantic/grounding_hypotheses.py`. The int8 pair cache reproduces the baseline (macro-F1 0.797 vs 0.795/0.796 reference), so the round is apples-to-apples.

- **H9 contradiction channel - rejected** - the NLI 3-way softmax's contradiction/neutral channels (free, same forward pass) add only +0.004/+0.005 macro-F1, inside fold noise (±0.014); on this gold the unsupported claims are omissions/fabrications, not contradictions, so the channel has little to bite on (consistent with the bounded-scope finding)
- **H10 aggregation beyond max - rejected** - distributional features of the per-pair scores (top-2 mean, logsumexp, count above threshold, top1−top2 margin) move macro-F1 −0.005 to +0.005; max-over-chunks already extracts what the score distribution knows, evidence redundancy is not a usable signal here
- **H11 reranker-first cascade - adopted** - the reranker always runs first; its max score `s` against the band [0.01, 0.66] decides the rest: `s <= 0.01` → flag as hallucination (reranker is sure nothing supports the claim, NLI skipped), `s >= 0.66` → pass as supported (NLI skipped), in-band → run the NLI and take the stack verdict as before. 61% of claims skip the NLI at macro-F1 0.795 (−0.002, inside noise); on the dense band sweep the adopted band is strictly no-worse than baseline on both error counts (FP 243 vs 244, FN 217 unchanged). Measured warm latency at k=8 (LATENCY hint, n=150): mean **1,184 → 857 ms/claim (−28%)**, median 759 ms (−34%), p90 −14% - hard claims still pay both models, easy claims pay one. Serving helper `cascade_scores` in `grounding_openvino.py`; benchmark `scripts/bench_grounder_cascade.py`
- **The band is empirical, not learned** - no model is fit against it: candidate edges are the quantiles of the reranker score distribution (19 points, 5th-95th percentile); every (a, b) pair is simulated on the out-of-fold scores into a (skip-rate, macro-F1) frontier (`reports/grounding_hypotheses.md`), and [0.01, 0.66] is the point with maximal skip at zero measurable quality loss (one step wider: ~70-80% skip at −0.004 to −0.013). Same calibration class as the operating threshold - chosen on frozen-model scores, in raw reranker-score space, so re-sweep if the reranker, quantization, or evidence distribution changes
- **Two band options from the same frontier** - the sweep also yields an FP-constrained alternative: **quality-neutral [0.01, 0.66]** (adopted default - skip 60%, macro-F1 0.797, FP 243 / FN 217, strictly no-worse than baseline 244/217) and **low-false-flag [0.05, 0.32]** (skip 84%, FP 192 = −52, at FN 279 = +62, macro-F1 0.783) for deployments where a false flag costs more than a miss; the FP cut is bought with recall, not quality elsewhere
- **Principle held** - all three used only softmax channels, aggregation statistics and thresholds over frozen models; no weights touched, nothing fit beyond the decision hyperplane

## Hypothesis round 2 - H12/H13/H14

Three more mechanism-targeting hypotheses against the cascade-adopted grounder (warm mean 869 ms on the round-2 sample), one allowed to restructure the architecture. Same caches, same OOF protocol, all measurement on the deployed CPU int8 engines; ladder and final benchmark in `reports/grounding_hypotheses.md`, driver `experiments/grounding-semantic/grounding_hypotheses.py`, bench `scripts/bench_grounder_round2.py`.

- **H12 pre-filter cosine gate - adopted** - the pre-filter already computes every claim-chunk cosine to rank the top-k; the max was discarded. It becomes a stage-0 gate: `cos <= 0.493` → flag, `cos >= 0.739` → pass, in-between → cascade as before. **22% of claims resolve at embed cost (~39 ms) with strictly fewer errors than the cascade alone** (FP 245 / FN 216 vs 248 / 217, macro-F1 0.797 vs 0.795). The gate works despite the bi-encoder's weak AUC (0.730) because it does not need pure tails - it only needs to agree with the cascade verdict on the claims it absorbs. Zero added compute; two thresholds fit OOF like the band
- **H13 rank-ordered early-exit reranker - adopted** - `rerank_max` scored all k=8 pairs in one padded batch, but the cascade only needs to know whether the max crosses the pass edge (0.66). Scoring pairs best-cosine-first in progressive batches (1, 1, 2, 4) and stopping at the first crossing is **verdict-invariant by construction** (verified exact on the bench sample, 150/150; unscored pairs cannot change a final pass verdict, and never-crossing claims score every pair). Mean pairs scored drop 8 → 4.8 (exit rate 49%); the int8 forward is near-linear in batch rows (122 ms batch-1 vs 95 ms/pair batch-8), so the exits keep most of what they save
- **H14 fused-evidence single forward - rejected** - assembling ONE evidence context per claim (top-2 chunk concat, or salience-packed sentences ranked by the same bi-encoder) and running ONE forward per cross-encoder would have cut 16 forwards to 2 (~211 ms/claim measured). Quality collapses: macro-F1 0.714-0.784 across all six stack/variant configs (-0.012 to -0.081), and the fused NLI correlates only 0.54 with the per-chunk max. **Max-over-chunks is load-bearing** - each chunk in isolation poses one focused entailment/relevance question; packing evidence dilutes it. Together with H10 this brackets the mechanism: nothing beyond the max helps, and the max cannot be approximated in one forward
- **Round-2 net effect (adopted = gate + cascade + exit)** - OOF macro-F1 0.795 → 0.797 with -3 FP / -1 FN; warm mean **869 → 662 ms (-24%)**, median 782 → 593 ms, vs the original always-both pipeline **-45% mean** at equal quality; p90 +4% (never-exit claims pay the progressive-schedule worst case, 876 vs 761 ms of reranker - the spend-where-uncertain shape sharpened). Footprint unchanged; everything added is two thresholds and a batch schedule
- **Principle held** - the gate and exit are thresholds and execution ordering over already-computed frozen-model scores; H14's contexts were input assembly only. No weights touched in either round

## Conclusions

- **Ship the consolidated two-cross-encoder semantic stack** - the full six-model + lexical hyperplane is the research maximum (macro-F1 0.824, AUC 0.913), but an ablation shows a logistic over just the two cross-encoders (bge-reranker + mDeBERTa-NLI) holds macro-F1 0.796 - within one CV fold-std of the full - while dropping four bi-encoders and the lexical layer; that minimal semantic pipeline is the deployable design (`semantic-grounding-sota.md`), served as single-engine OpenVINO int8, no fine-tuning
- **The cross-encoder reranker is the core signal** - bge-reranker-v2-m3 alone is AUC 0.841; bge-m3 and NLI scores are the additive lift
- **Use it as both a soft flag and a hard gate** - report the full curve, pick a low-false-flag point for pre-display blocking and a high-recall point for the feedback-loop re-check
- **Run it in the two-stage loop** - fast pre-filter on every claim, precision gate on the flagged subset, agent revises or retracts
- **The ceiling is the overlap region** - ~100 hallucinations and ~100 supported sit where neither the reranker nor the stack separates them; closing it needs a fine-tuned cross-encoder
- **Bounded scope** - the win is on paraphrased omission/fabrication hallucinations in retrieved-doc evidence; present-but-contradicted negatives need the contradiction signal the sibling lexical track studies

## Next steps

- **Hold confirmed on the 2,752-record organic-majority gold** - re-scored and re-fit: macro-F1 0.824 / AUC 0.913 held as the organic base rate dropped to 29%; per-language AUC en 0.92, es 0.80, fr 0.78, nb 0.66
- **Fine-tune a cross-encoder** - the remaining lever once the stack plateaus; target the overlap residual (the 102 numeric/spec misses)
- **Operating-point calibration** - per-language thresholds where counts allow; the shipped Bayesian calibrator now that a separated signal exists
- **Top-k pre-filter + chunk-embedding cache measured (done)** - the single-engine OpenVINO pipeline holds (k=8 macro-F1 0.822 on the subset); caching the source-chunk embeddings cuts the typical claim from ~4.2 s to ~1.2 s (median, 3.6×, LATENCY hint) and makes k=8 5.0× faster than all-chunks. Next: load the published HF int8 IRs (`stellars/*-openvino-int8`) into the deployment and wire the pre-filter to the RAG retriever's chunk vectors
- **Push the int8 IRs to HuggingFace** - the `models/ov/` IRs are push-ready (IR + config + tokenizer); publish under the org for reuse and CI
- **Confirm top-k macro-F1 on full gold (done)** - the full adopted serving path (gate + cascade + early-exit, deployed calibration frozen) runs end-to-end over the 2,752 gold at **macro-F1 0.789** (within fold noise of the 0.797 OOF simulation; error mix shifts toward recall - FP 328/FN 172 - because serving maxes are over the top-8 pre-filtered chunks, not all chunks), warm mean 585 ms / median 258 ms per claim (`scripts/run_grounder_full.py`). Re-fit thresholds on serving-derived scores before fixing the deployment operating point
- **CPU serving levers measured (done)** - `LATENCY` compile hint is 2.1× over `THROUGHPUT` for inline serving (now the default); `max_length` has no headroom below 512 (refuted); length-bucketing kept. The reranker-first cascade (−28%), the pre-filter cosine gate and the early-exit reranker (round 2, cumulative **−45% warm mean vs always-both, 662 ms/claim**) are measured and adopted. Next mechanical lever: an answer-level batched scorer. The remaining bigger win is GPU fp16 (~0.15-0.4 s/claim)

## Hypothesis round 3 - fine-tuned tiny faithfulness checker (pre-registered)

The deferred lever from Next steps ("fine-tune a cross-encoder ... target the overlap residual") opened as a full phase. Six independent hypothesiser lenses (architecture, distillation, quantization, data, methodology, integration) were run as a fanout; this section is the deduped **pre-registration** - predictions and pass/fail bars fixed BEFORE any run. No verdicts yet.

The thesis, unanimous across the fanout: **replace the mDeBERTa-NLI (and possibly the reranker) with a purpose-built faithfulness checker on a small standard-attention encoder**. Two facts make it a compound win the current architecture structurally cannot get:

- **Head shape** - gold hallucinations are omissions / unsupported-additions, not contradictions (H9), so the incumbent borrows a 3-label MNLI/XNLI entailment logit as a faithfulness proxy (AUC 0.806); a 2-way {supported, unsupported} head trained on faithfulness targets that boundary directly
- **Quantization** - mDeBERTa's disentangled attention gives int8 no speedup (footprint-only, 318 MB); a standard encoder (RoBERTa / MiniLM / ModernBERT) int8-accelerates (1.34x compiled, 1.0 ms bs=1), so moving off DeBERTa cuts params AND unlocks real int8 latency

This is NOT the refuted E-deploy "NLI replacement search" (MiniLM-L6/L12, XLM-R → stacks 0.758-0.765): that swapped in zero-shot 3-label NLI checkpoints. R3 swaps both the head shape (2-way) and the training data (in-domain faithfulness) - the falsifiable difference. That prior result is exactly what caps optimism, so R3-H15 measures whether a faithfulness-TRAINED checker transfers before any fine-tune is spent.

**"One model total" is a fiction.** Max-over-chunks is load-bearing (H10/H14) and warm-cached ranking needs a cheap bi-encoder to cut ~50 chunks to top-8; the honest floor is TWO small models - a bi-encoder pre-filter/gate (kept, 38 ms warm, doubles as the 22%-resolving stage-0 gate) plus ONE small cross-encoder checker replacing the two 889 MB cross-encoders.

### Methodology - the R3 ship-contract and honest-split protocol

"Tiniest that does the job" is made falsifiable by a pre-registered contract; a tiny fine-tuned model can memorise the gold, so the splits are group- and language-disjoint.

- **Ship-contract (conjunction of three)** - macro-F1 ≥ 0.775 (deployed 0.789 − one fold-std 0.014, non-inferiority) AND warm-CPU mean ≤ 200 ms AND int8 footprint ≤ 100 MB; among passers ship the fewest-params; declared relaxation ≤ 150 MB if the smallest passer is base-size (~110M)
- **Honest split** - the gold is **619 sources / 636 traces, not 5,857 rows**; the ship readout is leave-one-source-out (GroupKFold on `group_id`) plus a held-out-language slice, with a pre-registered train − holdout gap ≤ 0.05 (memorisation gate) and held-out-language macro-F1 ≥ 0.637 (no cross-lingual regression vs the frozen cascade)
- **Nested honest re-fit** - a student emits a new score scale, so its verdict head and operating threshold are fit only on folds it never scores (the joint track caught a +0.023 phantom from in-sample threshold selection); harness-validated by reproducing the frozen cascade at 0.789 ± 0.014 before any student number is believed
- **Two-sided error-count parity** - macro-F1 is FP↔FN-invariant, so a passer must also hold FP ≤ cascade + 2σ AND FN ≤ cascade + 2σ at a recall-matched threshold (reference two-CE OOF point FP 266 / FN 203, σ ≈ √n)

### Pre-registration - predictions and bars fixed before the run

| ID | mechanism | prediction | PASS bar (two-sided) | first kill-gate |
|---|---|---|---|---|
| R3-H15 | zero-shot faithfulness-checker upper anchor (MiniCheck-RoBERTa-Large 355M, standard attention, max-over-top-8) | OOF AUC 0.82-0.86 vs incumbent NLI 0.806 | AUC ≥ 0.806 unlocks the ladder; ≥ 0.86 → collapse both | AUC &lt; 0.79 on the cached gold pairs → whole round killed |
| R3-H16 | 2-way faithfulness head replaces the NLI, keep bge-reranker (ModernBERT / RoBERTa-base) | stack macro-F1 0.79-0.80 at ~150 MB, NLI-stage int8 &gt; 1.2x | macro-F1 ≥ 0.789 AND (≤ 200 MB OR NLI-stage ≤ 450 ms) | R3-H15 AUC ≥ 0.806 (faithfulness transfers at all) |
| R3-H17 | collapse both cross-encoders into ONE base checker, keep bge-m3 pre-filter (3→2) | warm ~150-230 ms, ~700 MB, macro-F1 0.78-0.80 | warm mean ≤ 300 ms AND macro-F1 ≥ 0.786 | single-head standalone AUC ≥ 0.86 (else fall back to R3-H16) |
| R3-H18 | cascade self-distillation into a ≤33M student (free per-pair teacher labels; ladder to ~22M) | macro-F1 0.76-0.79 at 22-33M, ≤ 100 MB, warm ≤ 120 ms | macro-F1 ≥ 0.785 AND ≤ 100 MB AND ≤ 120 ms | student-teacher per-pair Spearman ≥ 0.85 on the 111,800-pair cache |
| R3-H19 | addition/omission hard-negative fine-tune (not contradiction) + atomic-claim decomposition | slip-through ≥ 30% headroom, then macro-F1 ≥ 0.789 | macro-F1 ≥ 0.789 AND FN ≤ 203 AND FP ≤ 266 | Haiku-mint 200 addition/omission + 200 contradiction controls; slip-through ≥ 30% AND &gt; controls |
| R3-H20 | cross-lingual near-miss weight fine-tune (multilingual student) | non-EN macro-F1 0.637 → ≥ 0.70, EN held | non-EN ≥ 0.70 AND aggregate ≥ 0.789 AND near-miss TNR ≥ 0.85 | cascade over-accepts near-miss negatives (&lt; 85% rejected) → real headroom |
| R3-H21 | escalation economics - widen the switch_on band / cascade-of-cascades, current stack as hard floor | system macro-F1 +0.005..+0.02 at flat blended latency | macro-F1 ≥ current + 0.005 AND blended latency ≤ current | checker flip-rate on the lexical-clear band ≥ 2% |

### The hypotheses

**R3-H15 zero-shot faithfulness-checker upper anchor** - because MiniCheck-RoBERTa-Large is an encoder-only 2-way head trained on decomposed-doc faithfulness (the exact job the incumbent's entailment-index-0 proxies), scoring it frozen over the cached top-8 pairs and re-fitting the logistic will read OOF AUC 0.82-0.86 vs the incumbent NLI's 0.806 - the gate that governs the whole round

- Mechanism - the incumbent optimises MNLI, not faithfulness; a faithfulness objective should separate omission / fabrication from support on the same frozen chunks. RoBERTa-large is standard attention (int8-accelerable), but at 355M it is the CEILING PROBE, not the deploy target
- Probe / artifacts - one OOF pass, zero training; `data/interim/model_scores/pairs/full_pairs.npz` + `lytang/MiniCheck-RoBERTa-Large`; reuses the single-signal harness
- Verdict space - Killed-at-gate (whole round) if AUC &lt; 0.79; unlocks R3-H16 if 0.806-0.86; unlocks the R3-H17 collapse if ≥ 0.86; Ships as a 355 MB reference only if ≥ 0.876

**R3-H16 2-way faithfulness head replaces the NLI** - because the 3-label softmax wastes two logits this gold never exercises (H9 +0.004) and DeBERTa int8 has no speedup, a 2-way {supported, unsupported} head on a base standard encoder (ModernBERT / RoBERTa-base) fine-tuned on faithfulness data, dropped in for ONLY the NLI stage while keeping bge-reranker, holds stack macro-F1 ≥ 0.789 at ~150 MB with a real int8 latency cut

- Mechanism - all head capacity goes to support-vs-not; ModernBERT-base's 8k context also removes the ~6.5% pair truncation at 512; DeBERTa-v3-base-EN is the same-family fallback if the standard encoder underfits
- Probe / artifacts - one FT run on MiniCheck synthetic, score gold top-8 OOF, re-fit the logistic; ModernBERT `reference_compile=False` (repo HF quirk); NNCF int8 IR via `scripts/build_ov_grounder.py`
- Verdict space - Ships ≥ 0.789 at ≤ 200 MB with a latency cut; Kept 0.783-0.789 (footprint+speed at quality parity); Refuted &lt; 0.78 (confirms mDeBERTa uniquely strong even vs a trained faithfulness head)

**R3-H17 collapse both cross-encoders into one base checker** - because a faithfulness objective subsumes the reranker's relevance (AUC 0.841) and the NLI's entailment (0.806), one base checker over the bge-m3 top-8 could match the two-CE stack's OOF AUC 0.876, collapsing the 889 MB cross-encoder pair to one ~150 MB model at warm ~150-230 ms while the cheap pre-filter/gate stays

- Mechanism - the pre-filter is load-bearing only as the free 22%-resolving stage-0 gate (38 ms warm), not as latency, so it stays; the expensive half is the two cross-encoders. Stretch: shrink bge-m3 to a MiniLM-class embedder → a two-tiny-model ~200-300 MB floor, gated on top-8 recall parity ≥ 98%
- Probe / artifacts - reuse R3-H15's frozen scores as a STANDALONE signal (reranker+NLI dropped), OOF AUC vs 0.876; then a ranking-recall pass for the pre-filter-shrink stretch
- Verdict space - Ships (one checker + pre-filter) if standalone AUC ≥ 0.876; Kept (drop NLI only → R3-H16) if 0.841-0.876; Dropped if &lt; 0.841

**R3-H18 cascade self-distillation into a tiny student** - because the frozen cascade emits a calibrated per-pair grounded-probability for any (claim, chunk) with zero human labels, distilling a ≤33M standard-encoder student to regress it (then max-over-chunks + one threshold) holds macro-F1 0.76-0.79 while collapsing 3 models → 1 and cutting footprint to ≤ 100 MB int8 and warm to ≤ 120 ms

- Mechanism - the teacher's grounded-prob already fuses reranker + NLI (logistic weights +1.14 / +0.87), so one student regressing that per-pair target learns both. Scale the pool from the 111,800-pair cache to a class-balanced trace pool (prod is ~94% grounded / bimodal) to shrink the retention gap; ladder down to ~22M (MiniLM-L6 / DeBERTa-v3-xsmall) to locate the floor, adding TinyBERT attention-transfer or soft-label KD on the κ-0.50 overlap residual if output-KD saturates
- Probe / artifacts - distil on the EXISTING cache (no cascade re-run), measure student-teacher Spearman + OOF macro-F1 first; `full_pairs.npz` + `data/processed/golden_grounding_evidence_verified.parquet`
- Verdict space - Ships ≥ 0.785 at ≤ 100 MB / ≤ 120 ms; Kept 0.757-0.785 (size/speed-favourable point); Killed-at-gate if Spearman &lt; 0.85; Null if OOF &lt; 0.757 (below reranker-alone → distillation added nothing)

**R3-H19 addition/omission hard-negative data program** - because the gold's hallucinations are unsupported additions / silent-evidence omissions (even the 102 numeric misses are evidence-absent specs, not contradicted values) and the incumbent is generic MNLI, minting in-domain negatives by perturbing SUPPORTED gold with evidence-absent additions / dropped-support omissions (single-atom where claims decompose) and fine-tuning the small head holds macro-F1 ≥ 0.789 in the overlap residual a frozen model cannot separate

- Mechanism - the residual (FN ~160-203, FP ~248-266) is where a relevance reranker and generic entailment over-confirm topically-close-but-unsupported claims; a head trained on the "is this specific assertion present?" boundary targets it. Atomic-claim decomposition makes maximally-hard negatives (all-but-one atom still supported) - the claim-side lever, orthogonal to the refuted evidence-side aggregation (H10/H14)
- Probe / artifacts - Haiku-mint 200 addition/omission + 200 contradiction controls, run the deployed `semantic_ov.SemanticCascade.score`, confirm slip-through ≥ 30% AND &gt; controls (the probe IS the kill-gate, minutes, zero training); perturb `golden_v3` supported rows
- Verdict space - Ships if a standalone head ≥ 0.789; Kept if NLI-replacement ≥ 0.789; Killed-at-gate if slip-through &lt; 30% or below the contradiction control (no headroom on this axis)

**R3-H20 cross-lingual near-miss weight fine-tune** - because R1-H2 was refuted for the frozen joint GATE (nli_ent cross-lingual feature near-chance, 0.523) not for a fine-tuned head, fine-tuning a multilingual student's WEIGHTS on bilingual near-miss negatives creates cross-lingual faithfulness features the frozen mDeBERTa lacks, lifting non-EN macro-F1 above the cascade's AUC-0.584 floor

- Mechanism - the cascade fails non-EN because its frozen features do not separate the slice (R1-H1 0.584; R1-H4 head-reweighting −0.005 = a signal gap, not a weighting gap); training weights on in-language near-miss negatives checks support across the language boundary directly - the "new signal, not re-routing" the joint track named as still open
- Probe / artifacts - mint ~150 near-miss cross-lingual negatives from the base sentences, run the cascade; real headroom = the cascade over-accepts them (&lt; 85% rejected vs the easy-TNR 0.904); `golden_v3_synth_aug.parquet`, GroupKFold on `group_id` (translations stay in-fold)
- Verdict space - Ships if non-EN ≥ 0.70 and near-miss TNR ≥ 0.85 with no EN regression; Null if non-EN &lt; 0.66 (matches refuted R1-H4); Killed-at-gate if the cascade already rejects near-miss ≥ 85%

**R3-H21 escalation economics** - because a tiny checker cuts the escalated per-claim cost ~585 → ~150 ms, the lexical→semantic escalation band in `joint.py` can widen to route more of the lexical residual through the checker at flat blended latency (catching paraphrased fabrications the lexical tier scores as grounded), predicting system macro-F1 +0.005..+0.02; alternatively the current stack is kept as a hard floor behind the tiny checker (cascade-of-cascades)

- Mechanism - escalation is gated today only because the cascade is expensive; the band is a cost valve, not a quality-optimal boundary. At ~150 ms/escalation, checking ~2x more claims is still cheaper than today, and the new claims are the omission-type ones the stack's non-entailment signal catches. The cascade-of-cascades variant guarantees macro-F1 ≥ 0.789 as a floor (the stack overrides the checker only on its low-confidence band) at +1 model
- Probe / artifacts - run the R3 checker on the lexical-CLEAR band, count verdict flips vs gold; re-sweep `[a, b]` on the OOF frontier; depends on a shipped R3 checker
- Verdict space - Ships if macro-F1 ≥ +0.01 at flat latency; Kept if +0.005; Null if flip-rate &lt; 2% (band already clean); cascade-of-cascades Kept if the confident band ≥ 85% at warm ≤ 350 ms

### Sequencing

The round runs as a gated ladder, not a parallel sweep. **R3-H15 is the one probe to run this week** - a single frozen OOF pass over the existing 111,800-pair cache, zero training - and it routes everything: AUC &lt; 0.79 stops the program (mDeBERTa stays); 0.806-0.86 → R3-H16 (swap the NLI only); ≥ 0.86 → R3-H17 (collapse both cross-encoders). R3-H18 (distil to the tiniest) and R3-H19 (in-domain data) are the size-and-accuracy pushes that an R3-H15 pass unlocks; R3-H20 (cross-lingual) and R3-H21 (pipeline economics) are the follow-ons. Every model number above is a prediction pending measurement under the R3 ship-contract and honest-split protocol.

## Hypothesis round 4 - knowledge-free reasoning models as the grounding head (pre-registered)

R3 asked what to train. R4 asks a prior question: **what must the model know?** Grounding supplies its own evidence in the prompt, so the head needs no parametric world knowledge - only the ability to route between a claim and a passage. That is exactly the regime where attention-only and knowledge-free models are measured strongest, and exactly the axis (parametric recall) they give up. This section is the **pre-registration** - predictions and pass/fail bars fixed BEFORE any run. No verdicts yet.

The source evidence is a controlled study, not a vendor claim ([digest](../../references/papers/[paper%20digest]%20simple%20attention%20networks%20controlled%20study.md), [arXiv 2607.18363](https://arxiv.org/abs/2607.18363)):

- **FFN is ~2/3 of non-embedding parameters** and deleting it in place costs 0.470 nats - but reallocating that budget into attention depth costs only **0.006 nats (0.27% of loss)** at matched parameters, reproducible to 1e-4 across clean seed pairs
- **The residual gap localises to parametric recall, not to capability** - three independent measurements (token regions, task types, zero-shot benchmarks) agree; by 105B tokens the attention-only arm leads on every answer region and the deficit survives only on low-context query tokens (+0.038)
- **The closest public analogue of grounding favours attention-only, and the margin grows** - Sciq (the answer sits in a provided support passage) goes 0.725 → 0.742 for the SAN across 31B → 105B while the FFN arm regresses 0.702 → 0.661, seed ranges non-overlapping. The knowledge benchmark (Lambada) moves the other way at every budget
- **The applied product exists** - `cactus-compute/needle`, from the same authors: 26M params / 14 MB, attention-only encoder (12 layers, no FFN) + 8-layer decoder, d=512, 8-head/4-KV GQA, RoPE, ZCRMSNorm, tied embeddings, **8,192-token BPE vocabulary**; it outperforms FunctionGemma-270M, Qwen-0.6B, Granite-350M and LFM2.5-350M on single-shot function calling, post-trained in **45 minutes on 2B tokens**, minimum **120 examples per class**. MIT licence
- **Off-the-shelf knowledge-free checkpoints exist**, so the 16 x TPU-v6e / 27-hour pretraining is not on the critical path - `PleIAs/Monad` (56.7M, 64 layers, ~8k vocab, decoder-only, Apache 2.0, 200B tokens of SYNTH) and `PleIAs/Baguettotron` (0.3B). Monad scores MMLU ~30% / GSM8K 8% **by design** - it trades world knowledge for reasoning traces, which is the property under test, not a defect

**Why this is not R3 again.** R3 keeps the discriminative cross-encoder paradigm and changes the head shape and training data on a standard 100-355M encoder. R4 changes the **model class**: pretrained-to-reason-without-knowledge, 26-300M, potentially generative. The two are complements, not rivals - R3-H18 (self-distillation into a standard encoder) is the control R4-H26 must beat to justify a new architecture.

**Three ways this round is wrong, stated in advance.** The evidence is next-token loss on 6-87M decoder LMs, not a fine-tuned pairwise entailment head, so transfer is an assumption to test. Monad is **strictly monolingual English** while the gold is cross-lingual and the non-EN slice is already the cascade's weak point (AUC 0.584). And the paper is explicit that a task's storage-versus-routing identity is **relative to the training distribution**, so the SYNTH-trained split may not survive a move to technical prose.

### Methodology - carried-forward bars and the R4 additions

The R3 ship-contract and honest-split protocol carry forward unchanged; R4 adds one gate and one scope rule.

- **Naive baseline (unchanged)** - majority predictor, macro-F1 **0.417**, AUC **0.50**, hallucination-F1 0.000; every R4 number is reported as a delta against it AND against the deployed cascade (end-to-end macro-F1 **0.789**, two-CE OOF **0.796**, warm mean 585 ms, 318 MB int8 NLI stage)
- **Ship-contract (carried from R3)** - macro-F1 ≥ **0.775** AND warm-CPU mean ≤ **200 ms** AND int8 footprint ≤ **100 MB**, two-sided at FP ≤ 266+2σ and FN ≤ 203+2σ; among passers ship the fewest params
- **Honest split (carried from R3)** - leave-one-source-out GroupKFold on `group_id` (619 sources / 636 traces, not 5,857 rows), held-out-language slice, train − holdout gap ≤ 0.05, nested re-fit of any new score scale on folds it never scores
- **NEW - representation gate** - a knowledge-free model with an 8k SYNTH-trained vocabulary may not be able to *see* the evidence; tokenizer fertility and context overflow are measured before any inference (R4-H22)
- **NEW - scope rule** - Monad-class candidates are pre-declared **English-slice only**; any aggregate number must be reported blended with the incumbent serving the non-EN slice, never as if the head were multilingual

### Pre-registration - predictions and bars fixed before the run

| ID | mechanism | prediction | PASS bar (two-sided) | first kill-gate |
|---|---|---|---|---|
| R4-H22 | representation feasibility - 8k SYNTH BPE over technical prose | EN fertility 1.6-2.2x mDeBERTa, non-EN ≥ 3x; EN context overflow &lt; 10% | EN fertility ≤ 2.5x AND EN overflow ≤ 20% | fertility &gt; 2.5x OR overflow &gt; 20% → round killed pre-inference |
| R4-H23 | zero-shot knowledge-free reasoner as grounding judge (Monad 56.7M, Baguettotron 0.3B, frozen) | Monad OOF AUC 0.62-0.74, Baguettotron 0.70-0.80, both &lt; incumbent 0.806 | Baguettotron ≥ 0.75 unlocks round; Monad ≥ 0.70 unlocks the tiny track | best of the two &lt; 0.65 → no zero-shot transfer; fall through to R4-H25/H26 only |
| R4-H24 | **the mechanism test** - context-grounded vs knowledge-dependent split | AUC deficit vs mDeBERTa ≤ 0.03 on self-contained pairs, ≥ 0.10 on knowledge-dependent | (deficit_knowledge − deficit_selfcontained) ≥ 0.07 | needs R4-H23 scores only - free re-analysis, no new compute |
| R4-H25 | post-train Monad as a 2-way grounding head (Needle recipe, cascade labels) | EN macro-F1 0.74-0.80 at ~57 MB int8, warm ≤ 150 ms | macro-F1 ≥ 0.775 AND ≤ 100 MB AND warm ≤ 200 ms AND FP/FN parity | R4-H23 Monad AUC ≥ 0.70 |
| R4-H26 | attention-only cross-encoder trained from scratch (SAN, QK-norm, domain BPE) | 10-25M params, ≤ 30 MB int8, macro-F1 0.72-0.78 | macro-F1 ≥ 0.757 (reranker-alone) at ≤ 30 MB | must beat R3-H18 distillation at matched footprint, else Null |
| R4-H27 | reasoning trace vs single-token verdict (serving shape) | trace +0.02-0.04 macro-F1 at 8-20x CPU latency | trace ships only if ≥ +0.02 macro-F1 AND ≤ 200 ms warm | trace latency measured on 50 pairs before any quality run |
| R4-H28 | English-only scope gate (EN → knowledge-free head, non-EN → incumbent) | blended macro-F1 ≥ 0.789 at 40-60% lower blended latency | blended macro-F1 ≥ 0.789 AND blended warm ≤ 350 ms | depends on a shipped R4-H25 or R4-H26 head |

### The hypotheses

**R4-H22 representation feasibility gate** - because Monad's 8,192-token BPE was trained exclusively on English synthetic SYNTH text while the gold is multilingual technical prose, measuring tokenizer fertility and context overflow before any inference will show EN fertility 1.6-2.2x mDeBERTa's 250k multilingual vocabulary and EN context overflow below 10%, while non-EN fertility exceeds 3x

- Lever - tokenizer only; model weights untouched, no inference run. Held fixed: the gold pairs and the top-8 pre-filter
- Mechanism - an 8k vocabulary trained on one distribution shatters out-of-distribution words into many subword pieces; past ~2.5x fertility the evidence no longer fits the context and the model literally cannot see what it must ground against. This is a representation failure, not a capability failure, and it is cheap to detect
- Probe / artifacts - encode `data/processed/golden_grounding_evidence_verified.parquet` claims + their top-8 chunks with `PleIAs/Monad` and `mDeBERTa-v3-base` tokenizers; report tokens/word by language and the overflow fraction. Minutes, CPU only, zero GPU
- Verdict space - Killed-at-gate (whole round) if EN fertility &gt; 2.5x or EN overflow &gt; 20%; Kept if EN passes and non-EN fails (activates the R4 scope rule); Ships-forward if both pass
- Experiment - `experiments/grounding-semantic/R4-H22_tokenizer_gate.py`<br>data: `data/processed/golden_v6/golden_v6.parquet` (8,800 rows, 21 languages; EN slice 4,876)<br>serving unit: claim + one `cfg.chunk_max_chars`=1500 chunk, top-`cfg.semantic_top_k`=3, via `groundrails.chunking.recursive_chunk`<br>run: `uv run python experiments/grounding-semantic/R4-H22_tokenizer_gate.py`<br>execution: CPU only, no weights downloaded, ~4 min
- **Result** - EN fertility **2.342 tok/word vs the incumbent's 1.959, ratio 1.196** (predicted 1.6-2.2 - better than predicted); EN context overflow **0.3%** at median 438 / p95 553 tokens against Monad's 2,048 context, ~4x headroom. Non-EN ratios sit in the same band (fr 1.198, es 1.194, it 1.222, nb 1.202, de 1.193) with overflow ≤ 0.42%, so the tokenizer is NOT the reason non-EN would fail
- **Verdict** - **PASS**, round 4 proceeds to R4-H23. The 30x smaller vocabulary (8,192 vs 250,101) costs only 20% more tokens on English evidence, which is what makes the size class reachable at all: a 250k x 768 embedding table is ~192M parameters, more than three times Monad's entire 56.7M budget
- Log
  - `log: 2026-07-28` first run reported KILLED-AT-GATE on 81.4% overflow. **False kill - a probe defect, not a model property.** It measured claim + the ENTIRE raw `source_text` (median 37,536 chars, ~24 chunks' worth), which no engine ever receives. The tell: the same measurement fails the deployed 512-token mDeBERTa far worse, and that model ships. Corrected to measure the real serving unit (claim + one 1500-char chunk); overflow 81.4% → 0.3%, verdict FAIL → PASS. Fertility was unaffected (1.196 both runs) because it is a property of the text, not the window

**R4-H23 zero-shot knowledge-free reasoner as a grounding judge** - because SYNTH-pretrained models are trained on reasoning traces rather than fact memorisation, scoring `PleIAs/Monad` (56.7M) and `PleIAs/Baguettotron` (0.3B) frozen over the cached top-8 gold pairs will read OOF AUC 0.62-0.74 and 0.70-0.80 respectively - below the incumbent NLI's 0.806, but far enough above chance to establish that grounding transfers without world knowledge

- Lever - the judge model; the pre-filter, the top-8 chunk set and the max-over-chunks aggregation are held fixed at the deployed configuration
- Mechanism - the verdict is read as the length-normalised log-probability of a `supported` vs `unsupported` continuation given (claim, chunk), so no head is trained and no gradient is taken; this is the same frozen-anchor design as R3-H15 and its number is directly comparable
- Probe / artifacts - one OOF pass over `data/interim/model_scores/pairs/full_pairs.npz` restricted to the EN slice per the scope rule; re-fit only the logistic on folds it never scores
- Verdict space - Killed-at-gate if the better model reads &lt; 0.65 (no zero-shot transfer; only the trained tracks R4-H25/H26 survive); Kept 0.65-0.75; unlocks the tiny track if Monad ≥ 0.70; Ships as a reference anchor only if either ≥ 0.806
- Experiment - `experiments/grounding-semantic/R4-H23_zeroshot_judge.py`<br>data: `.../gold/golden_grounding_evidence_verified.parquet`, EN slice 2,117 claims → 94,060 pairs (44.4 chunks/claim), base rate 0.649<br>scoring: length-free `logP(" yes") - logP(" no")` at the first assistant position, max-over-chunks<br>incumbent re-scored on the identical pairs (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, fp16)<br>hardware: RTX PRO 6000, ~6 min/model<br>run: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python experiments/grounding-semantic/R4-H23_zeroshot_judge.py`
- **Result** - **VOID. The measurement is invalid; no verdict on Monad is available from this run.** The recorded numbers were incumbent NLI **0.7333** and Monad **0.4736** (chance), but an instrument check invalidates the Monad figure: at the verdict position Monad assigns **p = 1.0000 to `'<'`** (the opening of its mandatory `<think>` trace), while `" yes"` and `" no"` carry logp -43.0 and -29.3 - **combined probability mass 0.000000**. The scorer was reading numerical noise in the extreme tail. Confirmed independently on trivially separable pairs (claim verbatim in its chunk vs claim against an unrelated recipe): AUC **0.745**, not ~1.0, with every score pinned near -13 regardless of content - the model is not answering the question asked
- **Verdict** - **VOID** for the logprob measurement, then **Refuted for Monad-at-56M** on the generative re-test (see Log). The pre-registered `< 0.65 → kill whole round` rule is NOT triggered: it presumes a valid measurement AND a representative candidate, and Monad is neither. Per the round's own amendment this promotes **R4-H27 from follow-on to decider**: Monad's trained format is non-negotiable, so the only valid zero-shot read allows the `<think>` trace. The pre-registered `< 0.65 → kill` rule is NOT triggered, because it presumes a valid measurement
- **Byproduct that stands** - the incumbent's English-only bar is **0.7756**, not the 0.806 headline. Re-deriving from cache: all-lang 0.8087, EN-only 0.7756, EN reranker 0.8827. English is the HARDER slice; cross-lingual claims separate more easily because NLI fails them uniformly. Every R4 target is therefore 0.7756 on English, and this harness's chunking costs a further 0.042 (0.7756 → 0.7333) which is carried as a caveat on all R4 numbers
- Log
  - `log: 2026-07-28` first run scored `logP(" supported") - logP(" unsupported")`. In Monad's 8k vocabulary `" unsupported"` is five pieces `[' un','su','pp','ort','ed']`, so first-token scoring compared `" supported"` against the bare prefix `" un"`. Fixed with a fail-loud single-token guard and a yes/no pair; AUC moved 0.4764 → 0.4736, i.e. **the token defect was real but was not the cause of the null**
  - `log: 2026-07-28` the instrument check above then invalidated the design itself. Two consecutive false kills in this round (R4-H22's overflow, R4-H23's null) shared one signature: **a result too extreme to be interesting**. Methodology addition below
  - `log: 2026-07-28` **the VOID was correct about the instrument and wrong about the cause - dug deeper on challenge.** The scorer's prompt ended at `<|im_start|>assistant\n`, but `chat_template.json` appends `<|im_start|>assistant\n<think>\n` - **the `<think>` tag is part of the PROMPT, not a model choice**. So `p=1.0000` on `'<'` was the prompt being truncated one token early, not the model refusing to answer. With the correct prefix Monad generates normally
  - `log: 2026-07-28` **generative probe (greedy, correct template, 7 cases incl. the native `<source_1>` RAG shape) - Monad at 56M cannot do the task.** It reproduces the trained FORM faithfully - rubric headers `###`, confidence markers `●◐○`, source-examination structure - and in the source-tagged format it correctly LOCATES the span (`"Source 1 provides direct answer: ..." ● High confidence`). The CONTENT is incoherent: it reframed a grounding question as arithmetic (`1300 - 271 = 911`, actually 1029), wrote `271m = 271,000 ft` (889) and `1300 MW = 1300,000 MW`, computed `66.76 - 99.76 = -1.36%` (actually -33.0), asserted `271m = substantial bread height` against a sourdough recipe, could not answer "what is the capital of France" (looped `Capital of France (capital city)` four times), and could not extract `1937` from the sentence containing it. Not a prompt-format artefact - the failure is identical across plain ChatML, source-tagged RAG, and pure extraction
  - `log: 2026-07-28` **consequence - candidate replaced, thesis untouched.** Monad's own card numbers frame the result: MMLU ~30% against 25% chance for 4-way multiple choice is +5 points, GSM8K 8%, HotPotQA 8%. It is a demonstration of the smallest model emitting well-formed English with reasoning-shaped scaffolding, not a capable reasoner. This REFUTES Monad-at-56M as a zero-shot grounding judge; it does NOT test the round's thesis, which now runs on `PleIAs/Baguettotron` (321M, best-in-class for its size) and `PleIAs/Pleias-RAG-350m` (actually trained for grounding with literal quotes, arXiv 2504.18225). The span-location capability that DID survive is the retrieval half of grounding and is worth carrying forward

**R4-H24 the mechanism test - context-grounded versus knowledge-dependent** - because the source paper localises the entire attention-only deficit to parametric recall and measures attention-only models AHEAD on passage-supplied answers (Sciq 0.742 vs 0.661 at 105B), partitioning the gold into self-contained pairs (adjudicable from the supplied chunk alone) and knowledge-dependent pairs (needing a fact absent from the chunk) will show the knowledge-free models' AUC deficit versus mDeBERTa at ≤ 0.03 on the first and ≥ 0.10 on the second

- Lever - the evaluation partition only; identical scores, identical models, re-analysed. Nothing is trained and no new inference runs
- Mechanism - this is the round's crux and its cheapest experiment. If the deficit is uniform across the partition, the "grounding needs no knowledge" premise is false FOR THIS TASK and R4's rationale collapses to "it is a small model", which R3 already pursues with better-suited architectures. If the deficit splits as predicted, the mechanism transfers from next-token loss to pairwise grounding and the round is justified on evidence rather than analogy
- Probe / artifacts - partition by whether the human rationale in the verified gold cites only the supplied chunk; adjudicate ~200 pairs per side to keep the partition honest, blind to model scores. Reuses R4-H23's scores at zero marginal compute
- Verdict space - Confirmed if (deficit_knowledge − deficit_selfcontained) ≥ 0.07; Refuted if &lt; 0.03 (premise dead - record it and stop the round here rather than spending training budget); Null in between

**R4-H25 post-train Monad as a 2-way grounding head** - because the frozen cascade emits a calibrated per-pair grounded-probability for any (claim, chunk) at zero labelling cost and Needle demonstrates that 45 minutes over 2B tokens specialises a knowledge-free base into a narrow structured task, post-training Monad on the 111,800-pair cache will hold EN macro-F1 0.74-0.80 at ~57 MB int8 and warm ≤ 150 ms

- Lever - the post-training corpus and objective; the base checkpoint, the pre-filter and the top-8 aggregation are fixed
- Mechanism - Needle's result is that a knowledge-free base plus a short task-specific post-train beats general models 5-25x larger on the narrow task, and the minimum-data rule (120 examples per class) is exceeded by three orders of magnitude here. The 8k vocabulary and absent FFN are what make 56.7M affordable at 64 layers
- Probe / artifacts - teacher labels from `full_pairs.npz`; post-train per the needle recipe (`needle finetune`, JSONL with query/tools/answers adapted to claim/evidence/verdict); NNCF int8 IR via `scripts/build_ov_grounder.py`; evaluate under the carried-forward R3 honest-split protocol
- Verdict space - Ships if macro-F1 ≥ 0.775 at ≤ 100 MB and ≤ 200 ms with FP/FN parity; Kept 0.757-0.775 (a size/latency point below the contract); Dropped &lt; 0.757 (below reranker-alone - the knowledge-free base added nothing a standard encoder does not)

**R4-H26 attention-only cross-encoder trained from scratch** - because the FFN holds two thirds of non-embedding parameters and its deletion costs 0.006 nats at matched parameters, an encoder-only SAN cross-encoder (12-20 attention-only layers, d=512, QK-normalization, a domain BPE of 8-16k) trained directly on the cascade-labelled pair cache will reach macro-F1 0.72-0.78 at 10-25M parameters and ≤ 30 MB int8 - a size class below anything R3 can reach

- Lever - the architecture; the training data, labels and split protocol are identical to R4-H25 so the two are directly comparable
- Mechanism - the paper's trainability finding is specific and load-bearing: QK-normalization, not the FFN and not residual gating, is what keeps deep attention-only stacks trainable, so it is a precondition of the build rather than a tuning option. Pairwise grounding never queries weight-stored facts, so the one measured weakness is not exercised
- Probe / artifacts - build on the needle reference implementation (MIT); train a domain BPE on the gold corpus; train on `full_pairs.npz` cascade targets; compare against R3-H18's distilled standard encoder at MATCHED footprint - that comparison, not the absolute number, is what justifies a new architecture
- Verdict space - Ships if macro-F1 ≥ 0.757 at ≤ 30 MB; Kept if it beats R3-H18 at matched footprint; Null if R3-H18 matches or beats it (distillation into a standard encoder is simpler and already in flight); Dropped if training does not converge without the FFN at this data scale

**R4-H27 reasoning trace versus single-token verdict** - because Monad's distinguishing feature is that it emits an intermediary reasoning trace and CPU decode cost scales linearly with emitted tokens, generating a trace before the verdict will add +0.02-0.04 macro-F1 while costing 8-20x the latency of a single-token verdict, failing the 200 ms bar

- Lever - the decode length; model, prompt and data fixed
- Mechanism - a trace is only worth its cost if the grounding decision needs multi-step composition. Most gold hallucinations are omissions (H9), which a single comparison resolves; if the trace helps, it should help disproportionately on the multi-hop subset, which is the diagnostic to record alongside the aggregate
- Probe / artifacts - latency first on 50 pairs (minutes) to establish the multiplier, then quality only if the multiplier leaves headroom under 200 ms
- Verdict space - Ships the trace only if ≥ +0.02 macro-F1 AND ≤ 200 ms warm; Kept as an explainability-only mode (off the serving path) if it adds quality but breaks latency; Dropped if it adds &lt; 0.01

**R4-H28 English-only scope gate** - because Monad is strictly monolingual English while the non-EN slice is already the cascade's weakest (AUC 0.584, non-EN macro-F1 0.637), routing EN claims to a shipped knowledge-free head and leaving non-EN on the incumbent will hold blended macro-F1 ≥ 0.789 at 40-60% lower blended latency

- Lever - the routing rule; both heads are fixed, already-measured artefacts
- Mechanism - the language detector already runs in the lexical tier, so routing is free. This converts Monad's monolingual limitation from a blocker into a scope decision, and it is the only honest way to report an aggregate number for an English-only head
- Probe / artifacts - blend R4-H25/H26 EN verdicts with the deployed cascade's non-EN verdicts over the 2,752 gold; report blended macro-F1, blended warm latency, and the per-language table
- Verdict space - Ships if blended macro-F1 ≥ 0.789 at ≤ 350 ms; Kept if latency wins at macro-F1 ≥ 0.775; Dropped if the routing boundary itself costs more than it saves

### Sequencing

A gated ladder, cheapest-decisive-first. **R4-H22 is minutes on CPU and can kill the round before a single forward pass** - an 8k SYNTH vocabulary that cannot represent the evidence ends it there. R4-H23 is one frozen scoring pass. **R4-H24 then costs nothing** - it re-analyses R4-H23's scores - and it is the decision point: it tests the round's premise directly rather than inferring it from the source paper's next-token results, and a refutation there should stop the round before any training budget is spent. Only on a confirmed mechanism do R4-H25 (post-train the 56.7M base) and R4-H26 (build the attention-only head) run, and R4-H26's verdict is explicitly relative to R3-H18 at matched footprint. R4-H27 and R4-H28 are serving-shape decisions that presuppose a shipped head. Every number above is a prediction pending measurement under the carried-forward R3 ship-contract and honest-split protocol.

### Amendments - 2026-07-28, before R4-H23 ran

Append-only. R4-H22's verdict above is recorded and immutable; these change hypotheses that had NOT yet run, and each states what prompted it.

- **Scope narrowed to English, on the author's decision.** The round is testing whether the METHOD is load-bearing, not multilingual coverage. Every R4 number is an English-slice number unless stated. This is a deliberate narrowing of the question, not a claim that non-EN works
- **R4-H28 rewritten.** The pre-registration routed non-EN to the incumbent and kept two heads. The shipped argos MT bridge (`lexical_mt.py`, torch-free CTranslate2, already load-bearing in the lexical HIGH tier) makes a better design available: translate non-EN claims into English and run ONE head for everything. New prediction - blended macro-F1 ≥ 0.789 with a single head; new risk - MT paraphrase noise on the claim side, which R4-H22 shows is NOT a tokenizer problem (non-EN fertility 1.19-1.22, overflow ≤ 0.42%). The two-head routing variant is retained only as the fallback
- **The round conflated two separable claims; they are now split.** `PleIAs/Monad`'s config is `LlamaForCausalLM`, `hidden_size` 256, 64 layers, `intermediate_size` 768 - it **has MLPs and is not a SAN**. So Monad tests the *knowledge-free training* half of the thesis (SYNTH: reasoning traces, not memorised facts) and says nothing about the *attention-only architecture* half, which only R4-H26 tests. R4-H23/H25 results must be read as evidence about training data, never about deleting FFNs
- **R4-H23 re-scores the incumbent instead of quoting the cache.** The cached anchors were built on a chunking this repo can no longer reproduce - cached mean is 40.62 chunks/claim, and `recursive_chunk` matches the per-claim counts on only 39/300 claims at max_chars=1000 and 21/300 at 1500. Quoting 0.806 against a differently-chunked candidate would confound the comparison, so mDeBERTa-v3-NLI is re-scored on the identical freshly-generated pairs
- **R4-H23's number is a declared LOWER BOUND.** Monad's chat template forces a `<think>` trace before every answer (`<|im_start|>assistant\n<think>\n`), so scoring a direct supported/unsupported continuation violates its trained format. The consequence is a cleaner design than the pre-registration had: R4-H23 (no trace) is the lower bound, R4-H27 (with trace) the upper, and **the gap between them IS the value of the reasoning trace**. A sub-bar R4-H23 therefore promotes R4-H27 from follow-on to decider rather than killing the round outright
- **Harness validated before any candidate number is believed** (the R3 protocol's requirement). Re-deriving the anchors from `full_pairs.npz` with max-over-chunks reproduces the documented figures: NLI entailment **0.8087** (doc 0.806) and reranker **0.8414** (doc 0.841)

### Setup - R4 artefacts and data

Everything R4 runs against, so a reader reproduces it without the transcript.

- **Gold (verdict set)** - `experiments/grounding-semantic/private-rag-forensics/gold/golden_grounding_evidence_verified.parquet`, 2,752 rows `{claim, source_text, label, lang, user_id, trace_id}`, 1,966 supported / 786 hallucination; **EN slice 2,117 claims → 94,060 pairs at 44.4 chunks/claim, base rate 0.649**
- **Gold (tokenizer/fertility set)** - `data/processed/golden_v6/golden_v6.parquet`, 8,800 rows over 21 languages, EN slice 4,876
- **Cached incumbent scores** - `.../model_scores/pairs/full_pairs.npz`: `owner` (111,800), `rr`, `nli` (3-col, `id2label` = entailment/neutral/contradiction), `labels` (2,752), `langs`
- **Chunking** - `groundrails.chunking.recursive_chunk` at `cfg.chunk_max_chars` = 1500, `cfg.semantic_top_k` = 3; note the raw `source_text` is the whole retrieved blob (EN median 37,536 chars), never the serving unit
- **Candidates** - `PleIAs/Monad` (56.7M, 114.4 MB, Apache 2.0, ctx 2048, vocab 8192), `PleIAs/Baguettotron` (0.3B); incumbent `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
- **Hardware** - RTX PRO 6000 Blackwell Max-Q (96 GB, sm_120), `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`
- **Scripts** - `experiments/grounding-semantic/R4-H22_tokenizer_gate.py`, `R4-H23_zeroshot_judge.py`; logs under `logs/R4-*.log`, scores to `R4-H23_scores.npz`

### Methodology addition - validate the instrument before the experiment

Adopted 2026-07-28 after two consecutive false kills in this round. Both were measurement defects that produced verdicts, and both were caught only because the number was implausible - which is luck, not method.

- **Positive-control every new scorer before it scores anything real.** Run it on a handful of trivially separable pairs (claim verbatim inside its chunk vs claim against unrelated text). A scorer that cannot reach AUC ~1.0 there cannot be trusted at 0.75 on real data. This is what exposed R4-H23: trivial-pair AUC 0.745 with every score pinned near -13
- **Check where the probability mass actually is.** For any generative judge, print the top-k tokens at the verdict position before reading a verdict off two hand-picked ids. R4-H23's verdict tokens held 0.000000 of the mass; the model wanted `<think>`
- **Verdict words must be single tokens**, asserted in code, not assumed. Multi-token verdicts silently degrade to prefix comparisons
- **Measure the unit the engine actually receives.** R4-H22 measured claim + the whole 37.5k-char evidence blob rather than the 1,500-char serving chunk and reported 81.4% overflow instead of 0.3%
- **Treat "too extreme to be interesting" as a defect signal, not a finding.** A gate that also fails the shipping incumbent, or an AUC at exactly chance, is evidence about the harness first and the candidate second

### R4-H29 positive-control gate - zero-shot generative judges (result)

Added as a gate, not a hypothesis, under the "validate the instrument before the experiment" rule. It is the cheapest decisive test in the round and it closed a whole path.

**R4-H29 generative judge positive control** - because a judge that cannot separate a claim quoted verbatim from its own chunk from the same claim against an unrelated recipe cannot be trusted on real data, gating each candidate on 20 trivially separable pairs before spending a subsample run will show >= 90% parseable verdicts and >= 90% correct, or stop the candidate

- Lever - the candidate model; prompt, decoding (greedy), token budget (700) and the parser are held identical across candidates
- Mechanism - trivial separation is a necessary condition, not a sufficient one. The pairs share no vocabulary and no topic with the negative evidence, so any judge with a working verdict channel should be near-perfect; failure here means the null on real data would be uninterpretable
- Experiment - `experiments/grounding-semantic/R4-H29_positive_control.py`<br>20 cases: 10 claims verbatim inside their chunk (SUPPORTED) + the same 10 against one sourdough-recipe chunk (UNSUPPORTED)<br>prompt ends with `<|im_start|>assistant\n<think>\n` per `chat_template.json` - the `<think>` tag is part of the PROMPT<br>verdict parsed ONLY from the post-`</think>` segment, preferring an explicit `Answer: yes/no` line<br>run: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python experiments/grounding-semantic/R4-H29_positive_control.py PleIAs/Baguettotron PleIAs/Monad`
- **Result** - **both candidates FAIL, and fail in the same direction**

| candidate | params | parseable | correct | SUPPORTED (ok/wrong/none) | UNSUPPORTED (ok/wrong/none) |
|---|---|---|---|---|---|
| Baguettotron | 321M | 16/20 (80%) | 12/20 (60%) | 7 / **0** / 3 | 5 / **4** / 1 |
| Monad | 56.7M | 6/20 (30%) | 4/20 (20%) | 3 / **0** / 7 | 1 / **2** / 7 |

- **The asymmetry is the finding: 0 false negatives across both models, 6 false positives.** Neither model ever denied a genuinely supported claim; both confirmed claims against a bread recipe. Raw generations show the mechanism - the model **substitutes the claim for the evidence** and then agrees with itself ("Evidence: 'The dam is 271 metres tall'" when the evidence was the recipe)
- Monad additionally fails to commit at all on 70% of cases - it never closes `</think>` within 700 tokens
- **Verdict** - **the zero-shot generative-judge path is CLOSED for this model class.** For a grounder, manufacturing support is strictly worse than missing a hallucination: it launders a fabrication as verified. A candidate that is 40% wrong in that direction on topically-unrelated evidence cannot be trusted on near-miss negatives, which are strictly harder
- **The round's thesis is NOT refuted by this.** What is refuted is *asking an untrained model to perform the discrimination zero-shot*. R4-H25 (post-train on cascade labels) and R4-H26 (attention-only head trained on the pair task) train the discrimination rather than request it, and they remain live. The R4-H22 representation result (fertility 1.196x, overflow 0.3%) also stands and still makes the size class reachable
- Log
  - `log: 2026-07-28` **the Monad row is under review in round 6, not yet void.** A token-level audit found that `PleIAs/Monad`'s shipped `chat_template.json` emits `<|im_start|>`, `<|im_end|>` and `<think>`, none of which exist in its 8,192-token vocabulary - each shatters into 3-5 sub-word pieces, and the model cannot emit `</think>` as a single token, which fits the 70% non-termination observed here. **But the shattering is not by itself a defect**: Monad's `added_tokens` list is exactly `[UNK]`, `<|begin_of_text|>`, `<|end_of_text|>`, `[PAD]` and its BPE fills all 8,192 embedding slots, while its model card states it was trained on "the standard instruction style from Qwen" with that same ChatML block. If PleIAs trained through this tokenizer, the model saw the identical piece sequence during training and the R4-H29 prompt was faithful. R6-H40 discriminates the two readings; until it returns, this row stands as recorded. The Baguettotron row is unaffected - all four control strings are single tokens there (ids 65491-65494)
  - `log: 2026-07-28` this gate was itself repaired twice before it produced a usable number - the first parser took the first `yes|no` anywhere in the output, matching phrases like "No contradiction. No support needed." inside the reasoning trace, and scored a CORRECT Baguettotron answer as False. Rebuilt against raw generations: the verdict lives after `</think>` in an `Answer:` line. Budget also raised 220 → 700 after cases ran out mid-trace

## Hypothesis round 5 - prompt and elicitation engineering for small-model judgement (pre-registered)

Batch **J**, ids `R5-H30..R5-H39`. Round 4 established that the two knowledge-free candidates *can see* the evidence (R4-H22, fertility 1.196x) but cannot *judge* it when asked zero-shot (R4-H29, 0 false negatives / 6 false positives on trivially separable pairs). Round 5 holds the weights fixed and moves the only untested lever left before training: the prompt and the elicitation protocol.

The round is aimed by a second, independent observation - the shipped grounder is weak on prose. On a private proposal-genre prose set (77 claims, 25 source documents) the lexical tier confirms 25/77 (32.5%), the full cascade 29/77 (37.7%), and 28/60 (46.7%) once DEF-20 holds out-of-scope claims out of the denominator. The gold-set macro-F1 of 0.824 was measured on RAG traces, not on authored prose; this genre is materially harder and is the deployment target.

### Methodology - carried-forward bars and the R5 additions

- **Naive baseline** - unchanged: majority predictor macro-F1 **0.417**, hallucination-F1 0.000. Every quality claim below is a delta against it
- **Incumbent bars** - full gold stack macro-F1 0.824 / OOF AUC 0.913; English-only NLI AUC **0.7756** on the matched-chunking re-score (the English slice is the harder one - all-language reads 0.8087). The English bar is what an EN-only head must beat
- **Prose bar** - scope-aware confirm rate **46.7%** (28/60) on the private prose set; raw 37.7% (29/77)
- **NEW - the residue is unmeasured** - only 31 of the 77 prose claims carry a human label. "The grounder is bad on prose" is a hypothesis about the other 46, not an established fact. R5-H30 measures it before any build and can re-aim or kill the round
- **NEW - false positives are the tracked error** - R4-H29 showed the failure is one-directional. For a grounder, manufacturing support is strictly worse than missing a hallucination, so every R5 bar names an FP count, not only an aggregate
- **NEW - negatives must be near-miss** - R4-H29 used topically-unrelated negatives (a bread recipe). Any prompt gain measured there is provisional until it holds on near-miss negatives (R5-H37)
- **Held fixed across the round** - weights (frozen), decoding (greedy), candidate set (`PleIAs/Baguettotron` 321M, `PleIAs/Monad` 56.7M), the 20-case positive control from R4-H29. Only the prompt, the elicitation protocol and the verdict-extraction path vary

### Pre-registration - predictions and bars fixed before the run

| id | persona | lever | prediction | acceptance bar | gate |
|---|---|---|---|---|---|
| R5-H30 | deflationist | adjudicate the unconfirmed in-scope prose claims | ≥ 40% are correct refusals, not misses | ≥ 60% correct refusals → the "bad on prose" framing is refuted and the round is re-aimed at recall of the true residue | runs first; no model inference |
| R5-H31 | deflationist | question polarity - "is the claim UNSUPPORTED?" | verdicts flip wholesale; the model agrees with whatever is asked | flip-rate ≥ 70% → acquiescence bias, a prompt artefact, cheaply fixable | free re-run of the R4-H29 harness |
| R5-H32 | mechanist | decompose locate → compare → judge | locate ≥ 70% correct, judge ~50% | (locate − judge) ≥ 0.20 localises the broken stage | free re-run |
| R5-H33 | contrarian | structured prompt with an explicit absence instruction and a forced quote-before-verdict | Baguettotron 60% → ≥ 90% on the R4-H29 control | ≥ 85% correct AND FP ≤ 1/10 → **supersedes the R4-H29 verdict** | requires R5-H32 to show a working locate stage |
| R5-H34 | transfer | chain-of-verification - draft, generate verification questions, answer from evidence only, revise | false positives −50% | FP ≤ 3/10 at ≤ 2x the single-pass latency | requires R5-H33 ≥ 75% |
| R5-H35 | transfer | cloze reframing - mask a claim token, score recovery from the evidence | the verdict channel is bypassed entirely | AUC ≥ 0.70 on the EN gold slice | independent of R5-H33 |
| R5-H36 | heretical | no verdict is requested - the model emits the supporting span or a null token, deterministic code decides | beats every judged prompt in the round | AUC ≥ 0.75 EN with a code-only verdict | independent of R5-H33 |
| R5-H37 | contrarian | near-miss negatives replace topically-unrelated ones | prompt gains do not transfer | any R5-H33/H34 gain holds within 0.10 on near-miss | runs against whichever prompt wins |
| R5-H38 | hybridizer | self-consistency, k=5 samples, majority vote | FP drops, latency 5x | FP ≤ 2/10 AND warm ≤ 200 ms, else research-only | requires a prompt above 75% |
| R5-H39 | mechanist | claim-first versus evidence-first ordering | position drives the claim-substitution | asymmetry ≥ 0.15 → artefact, not comprehension | free re-run |

### The hypotheses

**R5-H30 the residue is unmeasured** - because only 31 of the 77 prose claims carry a human label and a claim the grounder declines may be a correct refusal rather than a miss, independently adjudicating every unconfirmed in-scope claim against the 25 source documents will show at least 40% are genuinely unsupported by the supplied sources, while the grounder's 46.7% scope-aware rate understates its true precision

- Lever - the measurement, not the system; no code changes, no inference, nothing shipped
- Mechanism - a confirm rate is a ratio whose denominator assumes every in-scope claim IS supported somewhere in the corpus. Authored proposal prose contains forward-looking statements, figures carried from documents outside the source set, and assertions the author never sourced. Each is a correct refusal scored as a failure
- Prediction - 40-70% of unconfirmed in-scope claims are correct refusals; the true miss count is 10-19 of 60
- Acceptance bar - ≥ 60% correct refusals refutes the framing and re-aims the round at the true residual; ≤ 30% confirms the grounder is the problem and the round proceeds as written; 30-60% keeps both live
- Experiment - `experiments/grounding-semantic/R5-H30_adjudicate_residue.py`<br>data: `experiments/grounding-semantic/private-prose-forensics/` (77 claims, 25 sources, gitignored)<br>procedure: for each unconfirmed in-scope claim dump the grounder's top-k retrieved chunks AND an independent exhaustive sentence scan over the raw sources ranked by claim-token overlap, so a refusal is not adjudicated from the grounder's own retrieval<br>adjudication: `miss` (a source states it, retrieval failed) versus `correct_refusal` (no source states it), written back into the gitignored dump<br>run: `uv run python experiments/grounding-semantic/R5-H30_adjudicate_residue.py`<br>execution: CPU, ~2 min; adjudication by reading, every `miss` confirmed by locating the supporting sentence in a named source

- **Result** - **26 of the 32 unconfirmed in-scope claims are correct refusals (81.2%)**, above the top of the predicted 40-70% band. Only 6 are genuine misses

| adjudication | n | share | note |
|---|---|---|---|
| correct refusal - authorial judgement | 10 | 31% | the author's own reading of the material; no source could state it |
| correct refusal - proposal design | 6 | 19% | the solution being proposed, not a claim about the sources |
| correct refusal - author's own estimate | 5 | 16% | costs, hours, market figures carried from experience |
| correct refusal - question or plan | 4 | 13% | interrogatives and next-step statements |
| correct refusal - compound opinion | 1 | 3% | one groundable clause welded to an evaluative one |
| **miss** | **6** | **19%** | a source states it and the grounder did not find it |

- **The confirm rate was a denominator artefact.** True groundable set = 28 confirmed + 6 misses = 34, so recall is **28/34 = 82.4%**, not 46.7%. The other 26 claims are unsourceable by construction and were being counted as failures
- **Adjudication agrees with every human label present** - all 10 residue claims labelled `not_groundable` were independently called correct refusals, and both labelled `supported` were independently called misses (12/12)
- **The misses split by error direction, and half are the dangerous kind** - 3 returned `none`, but **3 returned `contradicted`** on a claim a source states verbatim, including one carrying three percentages that appear unaltered in the source digest. Asserting that a document contradicts a claim it actually supports is strictly worse than declining it
- **Verdict** - **Refuted (the framing)**. The pre-registered `≥ 60% correct refusals → re-aim the round` rule fires at 81.2%. The grounder is not bad on prose; the metric was. What survives as a real defect is precision on the contradiction arm and the recall of the out-of-scope classifier, neither of which is a prompt problem - see the amendment below

**R5-H31 acquiescence bias versus claim substitution** - because R4-H29's six false positives are equally explained by a model that agrees with any proposition put to it and by one that substitutes the claim for the evidence, re-running the identical control with the question polarity inverted ("is the claim UNSUPPORTED by the evidence?") will flip at least 70% of verdicts if the cause is acquiescence, and leave them stable if the cause is substitution

- Lever - the polarity of the question; evidence, claims, decoding and parser identical to R4-H29
- Mechanism - acquiescence is a surface artefact of instruction-tuned agreement and is repairable by prompt design; substitution is a comprehension failure and is not. The two demand different rounds, and R4-H29 cannot distinguish them
- Prediction - flip-rate 70-95% on Baguettotron; Monad's non-committal rate stays near 70% under either polarity
- Acceptance bar - flip-rate ≥ 70% → the round proceeds with prompt engineering as the primary lever; ≤ 30% → substitution is real and R5-H33/H34 are downgraded in favour of R5-H36
- Experiment - `experiments/grounding-semantic/R5-H31_polarity_flip.py`, reusing the R4-H29 20-case control unchanged

**R5-H32 which stage is broken - locate, compare, or judge** - because R4-H29's raw generations show the model correctly quoting the relevant passage before reaching a wrong verdict, splitting the task into three separately scored sub-tasks will show locate accuracy at least 0.20 above judge accuracy, localising the defect to the verdict stage rather than to comprehension

- Lever - the task decomposition; one model call per sub-task instead of one for the whole
- Mechanism - a single prompt conflates retrieval, comparison and commitment. If locate is intact the model is not blind, and the correct design removes the stage that fails rather than trying to strengthen it - which is what R5-H36 does
- Prediction - locate 0.70-0.85, compare 0.55-0.75, judge 0.45-0.60 on Baguettotron
- Acceptance bar - (locate − judge) ≥ 0.20 → the verdict stage is the defect and R5-H36 is promoted; < 0.10 → the model is uniformly weak and no prompt will fix it, closing R5-H33/H34/H38
- Experiment - `experiments/grounding-semantic/R5-H32_stage_decomposition.py`, same 20 cases, three scored prompts per case

**R5-H33 the R4-H29 prompt was the weak link** - because R4-H29's prompt supplied no instruction to check for absence, no explicit refusal option and no output contract, replacing it with a structured prompt that forces a verbatim quote before any verdict and states the absence rule explicitly will lift Baguettotron from 60% to at least 90% on the identical control

- Lever - prompt structure only; model, decoding, cases and parser unchanged from R4-H29
- Mechanism - a model that must first emit a span from the evidence cannot substitute the claim for the evidence without producing a quote that visibly is not in the text. The quote requirement makes the substitution failure detectable rather than silent
- Prediction - correct 0.85-0.95, FP 0-1/10, parseable ≥ 0.95 on Baguettotron; Monad stays below 0.60
- Acceptance bar - ≥ 85% correct AND FP ≤ 1/10 supersedes the R4-H29 verdict with a one-line back-reference; < 0.75 leaves R4-H29 standing
- Experiment - `experiments/grounding-semantic/R5-H33_structured_prompt.py`

**R5-H34 chain-of-verification against self-agreement** - because the observed failure is the model agreeing with its own restatement of the claim, inserting a verification pass - draft a verdict, generate questions about the evidence, answer them from the evidence alone, then revise - will halve the false-positive count at no more than twice the single-pass latency

- Lever - the number of elicitation passes; prompt content held to the R5-H33 winner
- Mechanism - the verification questions are answered against the evidence with the claim out of context, which denies the model the opportunity to restate the claim as its own evidence
- Prediction - FP 6 → 2-3 across both models; latency 1.8-2.4x
- Acceptance bar - FP ≤ 3/10 AND ≤ 2x latency; a quality gain past 2x latency is recorded but does not ship
- Experiment - `experiments/grounding-semantic/R5-H34_chain_of_verification.py`

**R5-H35 cloze reframing bypasses the verdict channel** - because the broken component is commitment to a yes/no verdict rather than reading comprehension (R5-H32), reframing the task as evidence-conditioned completion - mask the claim's key token and score the model's recovery of it - will read AUC ≥ 0.70 on the English gold slice without ever asking for a judgement

- Lever - the output modality: token likelihood instead of a generated verdict
- Mechanism - a supported claim's key token is recoverable from the evidence; an unsupported claim's is not. The signal is a length-normalised logprob, deterministic, single forward pass, and comparable to a cross-encoder score
- Prediction - AUC 0.65-0.78 on Baguettotron, 0.55-0.68 on Monad, both below the 0.7756 English incumbent
- Acceptance bar - ≥ 0.70 keeps the tiny track alive as a cascade stage-0 gate; < 0.60 drops it
- Experiment - `experiments/grounding-semantic/R5-H35_cloze_recovery.py`, EN slice of the verified gold, same matched-chunking pairs as R4-H23

**R5-H36 do not ask for a verdict at all** - because R4-H29 shows these models locate correctly and judge badly, restricting the model to emitting a supporting span or a null token and having deterministic code decide - does the returned span actually contain the claim's numeric, entity and unit anchors - will beat every judged prompt in this round and read AUC ≥ 0.75 on the English slice

- Lever - the division of labour between model and code; the model retrieves, `entity_check` decides
- Mechanism - the failure mode is manufacturing agreement, and a manufactured span is checkable in a way a manufactured verdict is not. A hallucinated quote fails a substring test against the source; a hallucinated "yes" fails nothing. This also reuses the existing deterministic anchor layer rather than adding a new one
- Prediction - AUC 0.72-0.82 EN, FP ≤ 1/10 on the R4-H29 control, span-hallucination rate 5-20% and fully caught by the substring test
- Acceptance bar - AUC ≥ 0.75 EN with a code-only verdict; below 0.65 closes the knowledge-free zero-shot track entirely
- Experiment - `experiments/grounding-semantic/R5-H36_quote_only.py`; verdict from `groundrails.entity_check` anchors over the returned span, never from model text

**R5-H37 near-miss negatives are the real bar** - because R4-H29's negatives shared no vocabulary or topic with the claims, any prompt improvement measured on them is provisional, and re-testing the winning prompt against negatives drawn from the same document with one altered number, entity or qualifier will not transfer

- Lever - the negative class; prompt and model held at the round's best configuration
- Mechanism - topically-unrelated negatives are separable by lexical overlap alone, so a model can score well on them without doing entailment. Near-miss negatives are the operational case - the shipped grounder's false positives are all topically close
- Prediction - accuracy drops 0.15-0.35 from the unrelated-negative figure
- Acceptance bar - any R5-H33/H34 gain must hold within 0.10 on near-miss; a larger drop marks the gain as an artefact of the control set
- Experiment - `experiments/grounding-semantic/R5-H37_near_miss_negatives.py`; 20 near-miss cases built by perturbing one anchor per positive chunk

**R5-H38 self-consistency against the one-directional error** - because a single greedy sample commits to one substitution, sampling five verdicts at temperature and taking the majority will cut false positives without touching the prompt, at five times the latency

- Lever - the number of samples; prompt fixed at the round's best
- Mechanism - substitution is a sampling-path failure rather than a fixed belief, so independent paths should disagree; if they agree, the failure is systematic and no aggregation helps
- Prediction - FP 6 → 2-4, latency 4.5-5.5x
- Acceptance bar - FP ≤ 2/10 AND warm ≤ 200 ms; above 200 ms it is recorded as a research result and does not ship
- Experiment - `experiments/grounding-semantic/R5-H38_self_consistency.py`

**R5-H39 evidence-first versus claim-first ordering** - because claim substitution may be a recency artefact of the claim sitting closest to the generation point, swapping the order so the claim precedes the evidence will change the false-positive count by at least 0.15 if position drives the failure

- Lever - the order of the two blocks in the prompt; everything else identical to R4-H29
- Mechanism - the R4-H29 prompt places the claim immediately before the assistant turn, which is exactly where a restatement would be most available. If ordering moves the result, the failure is positional and cheap to fix; if not, it is comprehension
- Prediction - FP asymmetry 0.10-0.30 in favour of claim-first
- Acceptance bar - asymmetry ≥ 0.15 → positional artefact, fold the winning order into every later prompt; < 0.05 → comprehension, and prompt ordering is closed
- Experiment - `experiments/grounding-semantic/R5-H39_block_ordering.py`, free re-run of the R4-H29 harness

### Sequencing

Cheapest-decisive-first, and the first three can each re-aim the round before any build. **R5-H30 runs first and uses no GPU** - if most of the prose residue is correct refusals, the premise that the grounder is bad on prose is wrong and the round should target recall of a much smaller true residual. **R5-H31, R5-H32 and R5-H39 are free re-runs** of the existing R4-H29 harness with one variable changed each, and together they diagnose whether the failure is acquiescence, a broken verdict stage, or position. Only then do the builds run: R5-H33 is the direct attack on R4-H29's own verdict, R5-H34 and R5-H38 are elaborations that presuppose it, and R5-H35 and R5-H36 are independent alternatives that do not depend on any prompt working. **R5-H37 gates everything that passes** - no prompt gain is believed until it survives near-miss negatives. Every number above is a prediction pending measurement.

### Amendments - 2026-07-28, after R5-H30 ran

Append-only. R5-H30's verdict above is recorded and immutable; these change hypotheses that had NOT yet run, and each states what prompted it.

- **The round loses its prose justification but keeps its gold-set one.** R5-H31..H39 were aimed by two claims: the R4-H29 judgement failure, and "the grounder is bad on prose". The second is refuted - true prose recall is 82.4%, above the gold-set stack's own operating point. The prompt-engineering ladder stays live against R4-H29 and the English gold slice, but it is no longer the highest-value work on this repo and it drops behind the three defects below
- **NEW - false contradiction is the round's real finding.** 3 of the 6 misses were returned as `contradicted`, not `none`. The contradiction arm has a precision problem that no amount of retrieval or prompting touches, and it is the one error class a grounder must never make: it launders a correct document as wrong. Tracked as a defect, not a hypothesis, because the mechanism is deterministic code (`find_numeric_mismatches` and the contradicted-arm span picker), not a model
- **NEW - the out-of-scope classifier under-fires on authored prose.** DEF-20 catches hypothetical, self-referential and directive claims. It does not catch evaluative-authorial statements, proposal design, the author's own cost and effort estimates, or interrogatives - 26 such claims passed the in-scope filter on this set, 10 of them carrying a human `not_groundable` label. This is a recall gap in a classifier whose whole purpose is to keep ungroundable claims out of the denominator
- **NEW - the extractor welds separate bullets into one claim.** Several residue claims concatenate two or three source bullets into a single assertion, one clause of which is groundable and the rest opinion. A compound claim cannot be adjudicated as a unit, so it is unfalsifiable by construction and is scored as a failure
- **Scope-aware scoring must be reported against an adjudicated denominator.** The 46.7% figure and every earlier prose confirm rate assumed each in-scope claim is supported somewhere. Any future prose number states its groundable denominator or it is not comparable

## Hypothesis round 6 - the instrument, not the model (pre-registered)

Batch **K**, ids `R6-H40..R6-H47`. Round 4 concluded that zero-shot generative judging is closed for this model class. A token-level audit of the candidates shows that conclusion rests on a broken instrument for one of the two, and that the model actually trained for this task was never run at all.

**`PleIAs/Monad`'s shipped `chat_template.json` is written in tokens Monad's own tokenizer does not contain.** The template emits `<|im_start|>`, `<|im_end|>` and `<think>`; Monad's 8,192-token SYNTH vocabulary has none of them, so each shatters into 3-5 sub-word pieces. The same template string tokenises to **25 tokens on Monad against 12 on Baguettotron**, where all four are single control tokens (ids 65491-65494). Monad has never seen those markers as tokens, and it cannot emit `</think>` as a unit - which is exactly the observed failure, 70% of R4-H29 cases never closing the trace. The R4-H29 Monad row measured a malformed prompt.

**`PleIAs/Pleias-RAG-350M` ships no chat template because its format is a 19-token structured protocol**, every token present in vocabulary: `<|query_start|>` 65517, `<|query_end|>` 65518, `<|source_start|>` 65519, `<|source_id|>` 65520, `<|source_end|>` 65521, `<|language_start|>` 65522, `<|language_end|>` 65523, `<|query_analysis_start|>` 65524, `<|query_analysis_end|>` 65525, `<|query_report_start|>` 65526, `<|query_report_end|>` 65527, `<|source_analysis_start|>` 65528, `<|source_analysis_end|>` 65529, `<|source_report_start|>` 65530, `<|source_report_end|>` 65531, `<|draft_start|>` 65532, `<|draft_end|>` 65533, `<|answer_start|>` 65534, `<|answer_end|>` 65535. The `source_analysis` segment is the model assessing whether the supplied sources support an answer, and the model carries a trained refusal path for sources that cannot - which is this task in its native format. It is the closest published precedent for a tiny grounding head (`references/papers/[paper digest] pleias-rag small reasoners quote sources.md`, arXiv 2504.18225) and it has never been fed a single pair here.

A third anomaly, cheap to test: `Baguettotron` and `Pleias-RAG-350M` both auto-prepend **id 2 = `<|end_of_text|>`** as the sequence-initial token, because their `bos_token` is null and the post-processor falls back to EOS. Monad prepends the correct id 1 = `<|begin_of_text|>`. Every Baguettotron prompt in round 4 therefore began with an end-of-document marker.

### Methodology - carried forward, plus the pre-flight this round adds

- **Naive baseline** - unchanged: majority macro-F1 **0.417**. English incumbent NLI AUC **0.7756** on matched chunking
- **Control set** - the R4-H29 20-case trivial control, unchanged, so every number in this round is directly comparable to `Baguettotron 12/20` and `Monad 4/20`
- **NEW - look before parsing, formalised** - no scorer is written against an assumed output shape. Every candidate gets a raw-generation dump under each candidate format first, and the parser is built against what was observed. This is the R4 methodology addition applied as a build step rather than a lesson
- **NEW - vocabulary coverage is a pre-flight, not a discovery** - a candidate whose declared prompt format references strings absent from its own vocabulary is mis-specified, and any score taken under that format is void. R6-H47 makes this a gate that runs before a candidate is scored
- **Held fixed** - weights frozen, greedy decoding, the 20 control cases, the post-hoc parser discipline (verdict read only from the trained answer segment)

### Pre-registration - predictions and bars fixed before the run

| id | persona | lever | prediction | acceptance bar | gate |
|---|---|---|---|---|---|
| R6-H47 | scout | vocabulary-coverage pre-flight over every candidate | Monad FAILS its own template, Baguettotron and Pleias-RAG PASS | any FAIL voids that candidate's prior scores | runs first, seconds, no inference |
| R6-H46 | deflationist | token budget under each protocol against the model's real context | Monad ctx 2048 holds one 1,500-char chunk, overflows a multi-source prompt | overflow > 20% on the serving unit kills Monad on capacity | free, CPU |
| R6-H40 | mechanist | rebuild Monad's prompt from tokens it actually has | parseable 30% → ≥ 80%, correct 4/20 → ≥ 10/20 | ≥ 80% parseable **supersedes the R4-H29 Monad row**; still < 50% parseable confirms it as a model limit | needs R6-H47 |
| R6-H42 | follower | Pleias-RAG-350M under its native 19-token protocol | ≥ 90% correct on the trivial control, 0 false positives | < 90% closes the small-model track on a valid instrument | the round's decider |
| R6-H41 | mechanist | BOS ablation - suppress the EOS-as-BOS prepend | accuracy moves ≥ 0.10 on Baguettotron if prompts were off-distribution | ≥ 0.10 → every prior Baguettotron number is re-taken | free re-run |
| R6-H43 | deflationist | the trained refusal path IS the verdict - no prompt engineering | AUC ≥ 0.75 on the EN gold slice | ≥ 0.7756 beats the English incumbent outright | needs R6-H42 ≥ 90% |
| R6-H44 | mechanist | read the verdict from `source_analysis`, not from the answer | ≥ 0.05 AUC over the answer channel | free - same generation, two readouts | needs R6-H42 |
| R6-H45 | contrarian | the model's inline literal quotes + deterministic anchor check | beats every judged prompt in rounds 5 and 6 | AUC ≥ 0.75 with a code-only verdict | R5-H36 with a model trained to quote |

### The hypotheses

**R6-H47 vocabulary-coverage pre-flight** - because a prompt format built from strings a tokenizer does not carry is fed to the model as sub-word noise rather than control tokens, auditing every candidate's declared template against its own vocabulary before any inference will show Monad failing on all four of its template's control strings while Baguettotron and Pleias-RAG-350M pass

- Lever - the audit itself; nothing about the models changes
- Mechanism - a control token is a single embedding the model was trained to condition on. Shattered into pieces it becomes ordinary text in a position where structure was expected, and the model's trained format is never actually invoked
- Prediction - Monad 0/4 template tokens in vocabulary; the shipped template string tokenises to 25 pieces against Baguettotron's 12
- Acceptance bar - any candidate failing its own template has every prior score under that format marked void; the gate is then permanent for later candidates
- Experiment - `experiments/grounding-semantic/R6-H47_vocab_gate.py`<br>for each candidate: resolve the declared template, tokenise it, report per-control-string coverage, and enumerate every control-shaped token actually present in vocabulary<br>run: `uv run python experiments/grounding-semantic/R6-H47_vocab_gate.py`<br>execution: CPU, seconds, no weights loaded

**R6-H46 token budget under the real protocol** - because Monad's context is 2,048 against 4,096 for the other two and its 8k vocabulary costs 1.196x the incumbent's tokens, measuring the serving unit under each model's own protocol will show a single 1,500-char chunk fitting comfortably while a multi-source Pleias-style prompt overflows Monad

- Lever - the measurement unit: the assembled prompt as the engine receives it, per protocol, not the claim plus a blob
- Mechanism - R4-H22 measured fertility on claim + one chunk and passed. The Pleias protocol wraps several sources with per-source ids, which is a different and larger unit
- Prediction - single-chunk overflow < 1% on all three; multi-source overflow 30-70% on Monad, < 5% on the other two
- Acceptance bar - overflow > 20% on the intended serving unit kills that candidate on capacity, independently of quality
- Experiment - folded into `R6-H47_vocab_gate.py`; same run, no extra cost

**R6-H40 Monad under a format it can represent** - because Monad's only in-vocabulary control tokens are `<|begin_of_text|>` (1) and `<|end_of_text|>` (2), rebuilding the prompt as plain text between those two and discovering the trained shape from raw generations will lift parseable verdicts from 30% to at least 80% and correct answers from 4/20 to at least 10/20

- Lever - the prompt encoding; model, cases, decoding and token budget held at R4-H29 values
- Mechanism - the failure signature was not wrong answers but absent ones: 70% never closed the trace. A model asked to terminate a structure it has no token for cannot terminate it. Removing the structure removes the failure
- Prediction - parseable 0.80-0.95; correct 0.50-0.75, still below Baguettotron
- Acceptance bar - ≥ 80% parseable supersedes the R4-H29 Monad row with a back-reference; < 50% parseable under a valid format confirms the original verdict on the model rather than the harness
- Pre-experiment probe - raw-generation dump under three candidate formats before any scoring; the parser is written against the observed shape
- Experiment - `experiments/grounding-semantic/R6-H40_monad_format.py`<br>same 20 cases as R4-H29<br>run: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python experiments/grounding-semantic/R6-H40_monad_format.py`

**R6-H42 the model built for this, in its own protocol** - because Pleias-RAG-350M was mid-trained specifically to answer from supplied sources with inline literal quotes and an explicit refusal path, and its 19 protocol tokens are all present in vocabulary, running the trivial control under that native protocol will reach at least 90% correct with zero false positives

- Lever - the candidate model and its trained protocol together; the control cases are unchanged
- Mechanism - the R4-H29 failure was a model substituting the claim for the evidence. A model trained to emit the supporting excerpt cannot substitute silently: the excerpt either appears in the source or it does not. The `source_analysis` segment exists precisely to assess source adequacy before answering
- Prediction - correct 0.90-1.00, false positives 0-1, parseable ≥ 0.95
- Acceptance bar - ≥ 90% correct and FP ≤ 1 unlocks R6-H43/H44/H45; < 90% on a valid instrument closes the small-model zero-shot track for real
- Pre-experiment probe - raw-generation dump under the assembled protocol before scoring
- Experiment - `experiments/grounding-semantic/R6-H42_pleias_rag_protocol.py`<br>prompt assembled from the in-vocabulary protocol tokens, the claim posed as the query and the chunk as a single source<br>verdict read only from the trained answer segment, never from free text<br>run: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python experiments/grounding-semantic/R6-H42_pleias_rag_protocol.py`

**R6-H41 the EOS-as-BOS prepend** - because Baguettotron and Pleias-RAG-350M declare no `bos_token` and their tokenizers fall back to prepending id 2 `<|end_of_text|>`, every round-4 prompt opened with an end-of-document marker, and suppressing it will move control accuracy by at least 0.10 if the prompts were off-distribution

- Lever - one token at position 0; everything else identical
- Mechanism - an end-of-document marker at the start of a sequence signals a document boundary the model then conditions on. Whether that is off-distribution depends on how the corpus was packed during training, which is not documented, so it must be measured rather than assumed
- Prediction - a move of 0.00-0.15; most likely small, because packed-corpus training often does separate documents with EOS
- Acceptance bar - ≥ 0.10 means every prior Baguettotron number is re-taken under the better setting; < 0.05 closes the question
- Experiment - `experiments/grounding-semantic/R6-H41_bos_ablation.py`, the 20-case control run twice per model

**R6-H43 the refusal path is the verdict** - because Pleias-RAG-350M was trained with "the supplied sources cannot answer this" as a first-class output rather than an absence of output, posing the claim as a query and reading refusal-versus-answer as the grounding verdict will reach AUC ≥ 0.75 on the English gold slice with no prompt engineering at all

- Lever - the readout: a trained behaviour is used directly instead of a judgement being requested
- Mechanism - refusal and "unsupported claim" are the same predicate. The model already computes it; round 5's prompt ladder exists only because the earlier candidates did not
- Prediction - AUC 0.70-0.82 EN
- Acceptance bar - ≥ 0.7756 beats the English incumbent outright and promotes the candidate to a cascade stage; ≥ 0.70 keeps it as a gate
- Experiment - `experiments/grounding-semantic/R6-H43_refusal_verdict.py`, EN gold slice, matched chunking as in R4-H23

**R6-H44 source_analysis versus the answer channel** - because the model emits an explicit source-adequacy assessment before drafting, scoring the verdict from the `source_analysis` segment rather than from the final answer will read at least 0.05 AUC higher on the same generations

- Lever - which segment of one generation is read; no extra inference
- Mechanism - the answer is shaped by fluency pressure and by the draft stage; the analysis segment is the model's own statement about evidence sufficiency, which is closer to the quantity being predicted
- Prediction - +0.03 to +0.10 AUC over the answer channel
- Acceptance bar - ≥ +0.05 makes the analysis segment the shipped readout
- Experiment - re-analysis of the R6-H43 generations at zero marginal compute

**R6-H45 the quote is the verdict** - because Pleias-RAG-350M generates literal source excerpts inline rather than reconstructing them post hoc, and a fabricated excerpt fails a substring test against the source while a fabricated verdict fails nothing, checking the emitted quote deterministically with `entity_check` anchors will beat every judged prompt in rounds 5 and 6

- Lever - the division of labour: the model quotes, deterministic code decides
- Mechanism - this is R5-H36 with a model actually trained to quote. The paper's own argument for generated over post-hoc citation is that retrospective cross-referencing does not hold up in production, which is the same failure mode this hypothesis exploits in reverse
- Prediction - AUC 0.75-0.85 EN, quote-hallucination rate 3-15% and fully caught by the substring test
- Acceptance bar - AUC ≥ 0.75 with a code-only verdict
- Experiment - `experiments/grounding-semantic/R6-H45_quote_verdict.py`, reusing R6-H43 generations plus `groundrails.entity_check`

### Sequencing

Two free CPU steps decide what is worth loading. **R6-H47 and R6-H46 run first in one script** - seconds, no weights - and they say which candidates were ever validly measured and which fit the serving unit at all. Then the two GPU questions run against the unchanged 20-case control: **R6-H40** asks whether Monad was ever given a fair test, and **R6-H42** asks whether the model built for this job works. R6-H42 is the decider - a failure there, on an instrument the pre-flight has certified, closes the small-model zero-shot track on evidence rather than on a malformed prompt. R6-H41 is a free ablation alongside. Only on a passing R6-H42 do the gold-slice runs follow: R6-H43 produces the generations, and R6-H44 and R6-H45 are re-analyses of those same generations at zero marginal compute. Every candidate gets a raw-generation dump before any parser is written.

### Round 6 results

Run 2026-07-28. The pre-flight and the reproduction both landed as predicted; the grounding result did not.

**R6-H47 vocabulary-coverage pre-flight - PASS (prediction confirmed)**

| candidate | declared format | control strings in vocab | rendered template | sequence-initial token |
|---|---|---|---|---|
| Monad 56.7M | ChatML (`chat_template.json`) | **0/4** | **25 tokens** | `<\|begin_of_text\|>` (id 1) |
| Baguettotron 321M | ChatML (attribute) | 4/4 | 12 tokens | `<\|end_of_text\|>` (id 2) |
| Pleias-RAG-350M | structured protocol, no template | 19/19 | n/a | `<\|end_of_text\|>` (id 2) |

- The gate caught an instrument bug in itself first: `tok.chat_template` is empty for Monad because its template lives in a separate `chat_template.json`, so the first version classified Monad as a protocol model and audited it against the wrong 19 tokens - a correct FAIL for an incorrect reason. Fixed before the verdict was recorded
- Monad's `added_tokens` list is exactly `[UNK]`, `<|begin_of_text|>`, `<|end_of_text|>`, `[PAD]`, and its BPE fills all 8,192 embedding slots, so there is no room for the control tokens its own template emits
- The EOS-as-BOS anomaly is confirmed for both 65k-vocab models (R6-H41 remains unrun)
- **Verdict** - **Kept as a standing gate.** Every future candidate is audited before it is scored

**R6-H46 token budget - PASS, prediction wrong in direction**

- Single-source serving unit: Monad 470 tok (23% of its 2,048 ctx), the other two 385 tok (9% of 4,096)
- Three-source unit: Monad 1,348 tok (66%), the other two 1,098 tok (27%)
- **Verdict** - **Null.** Predicted 30-70% multi-source overflow on Monad; measured zero overflow anywhere. Capacity is not a constraint for any candidate and this line of attack is closed

**R6-H40 Monad cannot be validated - Killed-at-gate**

Rather than argue about the format, the published number was targeted directly. Monad's card reports MMLU "close to 30%" against a 25% chance floor; three standard protocols were run over a 2,000-question stratified MMLU sample.

| protocol | accuracy | vs chance |
|---|---|---|
| loglikelihood, 0-shot | 0.255 | +0.005 |
| loglikelihood, 5-shot | 0.226 | -0.024 |
| ChatML generative (the card's own block, 67% parseable) | 0.143 | -0.107 |

- **No protocol reaches the published number, and none clears chance.** The best reading is 0.255 against a 0.250 floor
- This does not establish that the published figure is wrong - PleIAs released no eval code for Monad, so a protocol that reproduces it may exist. It does establish that **no Monad measurement in this repo can be validated**, which is the operative fact
- **Verdict** - **Killed-at-gate.** Monad is withdrawn as a candidate. The R4-H29 Monad row is neither confirmed nor voided; it is simply uninterpretable, and no further Monad work is warranted

**R6-H42 Pleias-RAG-350M as a judge - Refuted**

Stage 1 reproduced the model card's own worked example verbatim - the model returned "Paris is the capital of France" with correctly-formed inline `<ref name="<|source_id|>1">` citations. **The harness is validated**, which is what gives the negative result below its force.

| readout | correct | FP | FN | parseable |
|---|---|---|---|---|
| Pleias-RAG-350M judge | 11/20 (55%) | **9** | 0 | 20/20 |
| R4-H29 Baguettotron judge | 12/20 (60%) | 4 | 0 | 16/20 |

- The purpose-built grounding model, on a validated harness, confirms claims against a sourdough recipe nine times out of ten. Same one-directional failure as round 4, now on an instrument that is beyond dispute
- **Verdict** - **Refuted.** Zero-shot generative judging is closed for this model class on evidence, not on a malformed prompt. This is the round's decider and it settles the question round 4 could not

**R6-H45 the quote is the verdict - Ships on the control, Refuted on the gold**

Same generations, zero extra compute. The model emits literal excerpts, so a fabricated excerpt is caught by two string tests: is the quote genuinely in the source, and does it share a numeric or entity anchor with the claim.

| readout | control 20 cases | FP | gold 600 EN claims (macro-F1) | P | R |
|---|---|---|---|---|---|
| judge (model decides) | 11/20 (55%) | 9 | 0.4762 | 0.521 | 0.870 |
| **quote + code (code decides)** | **18/20 (90%)** | **1** | **0.5107** | 0.615 | **0.267** |
| naive majority baseline | - | - | 0.4170 | - | - |
| shipped cascade (all-lang gold) | - | - | 0.8240 | - | - |

- On the control the mechanism works exactly as designed: false positives 9 → 1, and the model's fabricated citations (`<ref>The dataset spans 964 sensors</ref>` against a bread recipe) are caught by substring alone
- **On the gold it does not transfer.** Macro-F1 0.511 beats the judge channel (+0.035) and the naive baseline (+0.094), and loses to the shipped cascade by **-0.313**
- **Recall is the failure: 0.267.** On the control the supporting sentence WAS the chunk; on the gold the model must locate and quote an exact span inside three 1,500-character sources, and for three quarters of genuinely-supported claims it emits no usable quote at all. Precision 0.615 means even the quotes it does emit are wrong 38% of the time
- **Verdict** - **Refuted for deployment, Kept as a finding.** The control result was a ceiling produced by verbatim-in-chunk positives, not a transferable capability. Recorded because the mechanism is sound and the failure is located precisely: quote *generation* under real retrieval, not the deterministic check
- Correction to the round's own framing - R6-H37 was pre-registered to test whether control gains survive near-miss negatives. They did not even survive the move to real data, so R6-H37 is now moot for this candidate

**R6-H43 the refusal path as a verdict - Refuted**

- The judge readout above IS the refusal path, read from the trained answer segment: macro-F1 **0.4762**, recall 0.870, precision 0.521, 240 false positives on 300 negatives
- Predicted AUC 0.70-0.82 against the 0.7756 English incumbent; measured a verdict that is barely above a coin flip and 0.35 macro-F1 below the shipped stack
- **Verdict** - **Refuted.** The trained refusal behaviour does not discriminate on this data

### Round 6 conclusion

The round did what it was built to do: it replaced two contested measurements with validated ones, and both came back negative.

- **The instrument objection is settled.** Pleias-RAG-350M's harness reproduces its vendor's published example exactly, so its failure is the model's, not ours. Monad's cannot be validated at all, so Monad is out
- **Small generative models cannot judge grounding.** Three models, two vocabularies, a validated harness and a purpose-built grounding checkpoint all produce the same one-directional failure: they manufacture support
- **The quote-and-check mechanism is real but the quoting is not good enough.** Deterministic verification of an emitted excerpt is sound and cheap; a 350M model asked to find that excerpt under realistic retrieval succeeds 27% of the time
- **Nothing here displaces the shipped stack.** The deployed cascade holds macro-F1 0.824; the best round-6 configuration reaches 0.511
- **Where the remaining value is** - not in a smaller model. `DEF-22` (CONTRADICTED returned on claims a source states verbatim) and `DEF-23` (out-of-scope classifier recall) are defects in deterministic code that already ships, and both were surfaced by round 5's adjudication

## Hypothesis round 7 - is the task capacity-limited, and is our number real (pre-registered)

Batch **L**, ids `R7-H49..R7-H53`. Rounds 4-6 closed the generative track. A four-lane research sweep (reports under `reports/research-*.md`) then converged, independently, on two claims that this round tests rather than accepts:

- **architecture is probably not the binding constraint** - the honest ceiling estimated from both the training-recipe and architecture lanes is 0.80-0.84, i.e. parity with the shipped cascade, because a distilled student cannot exceed its teacher and the teacher is what our labels encode
- **our headline number may not be comparable to anything** - macro-F1 0.824 at base rate 0.649 converts to roughly 0.82-0.85 balanced accuracy, which would sit ABOVE the LLM-AggreFact leaderboard top of 77.4. The far likelier reading is that the private gold is easier or more loosely labelled than the public benchmark

**Process note, recorded rather than hidden.** `R7-H52` below was executed BEFORE this pre-registration was written, as GPU shape benchmarks commissioned during the research sweep. It is recorded as a measurement with its numbers intact, and explicitly NOT claimed as a pre-registered hypothesis. `R7-H49`'s script was likewise written before registration. Both are noted here so the log does not imply a discipline that was not followed; the remaining hypotheses are registered before execution as normal.

### Methodology - what this round adds

- **Naive baseline** - unchanged, majority macro-F1 **0.417**. Incumbent full stack **0.824**, English NLI AUC 0.7756, best single signal reranker AUC 0.841
- **NEW - external calibration is now mandatory before any target is set.** No quality target may be quoted against the private gold alone until `R7-H49` establishes what that gold is worth externally
- **NEW - speed is a first-class acceptance criterion**, not a tiebreak. The shipped cascade is ~662 ms/claim warm on CPU across three models
- **NEW - the teacher corpus must be regenerated.** `model_scores/pairs/full_pairs.npz` carries 111,800 rows of `owner`/`rr`/`nli`/`labels`/`langs` and **no text**, and the chunking that produced it cannot be reproduced (R4 amendment: cached mean 40.62 chunks/claim, `recursive_chunk` matches 39/300 at max_chars=1000). Soft labels therefore cannot be re-attached to their inputs, and every distillation hypothesis depends on regenerating the cache under current chunking first

### Pre-registration - predictions and bars fixed before the run

| id | lever | prediction | acceptance bar | gate |
|---|---|---|---|---|
| R7-H49 | score the shipped cascade on LLM-AggreFact, no threshold refitting | 63-72 balanced accuracy against our private-gold ~82-85 | gap &gt; 8 points → 0.824 is not SOTA-comparable and must never be quoted as such | blocked on Hub auth; no GPU |
| R7-H50 | capacity ablation - 4 checkpoints at 140M / 278M / 307M / 278M-minus-6-layers | all four land within 0.02 macro-F1 of each other | spread ≤ 0.02 → the task is NOT capacity-limited and architecture work is closed | needs R7-H51 |
| R7-H51 | regenerate the teacher corpus under current chunking, text retained | ~110k (claim, chunk) pairs with reranker + NLI scores and their text | reproduces the incumbent's 0.824 on the gold from the regenerated pairs | prerequisite for R7-H50/H52/H53 |
| R7-H52 | encoder shape at matched parameters - depth versus width, attention-only, parallel blocks | width wins on latency; attention-only loses | **already measured, see below** | ran before registration |
| R7-H53 | MICE-style split encoder - N independent layers with the document side cached offline, M joint layers | ~9x faster than the joint model at an AUC deficit ≤ 0.01 | deficit ≤ 0.01 → ships; &gt; 0.03 → dropped | needs R7-H50 |

### R7-H52 encoder shape - measured (executed before registration)

Matched at ~140-160M body parameters, bf16, B=3 T=512, RTX PRO 6000. Harness `experiments/grounding-semantic/arch_shape_latency_bench.py` and `arch_shape_seq_bench.py`, logs `logs/shape-bench{,2}.log`.

| shape | L | d | body M | GFLOP | eager ms | graphed ms | TFLOP/s |
|---|---|---|---|---|---|---|---|
| narrow-deep | 53 | 512 | 149.5 | 544.2 | **14.91** | 4.44 | 122.6 |
| | 28 | 768 | 140.5 | 498.9 | 8.64 | 3.33 | 149.8 |
| | 16 | 1024 | 142.7 | 489.6 | 4.69 | 2.81 | 174.5 |
| wide-shallow | 8 | 1536 | 160.5 | 531.5 | **2.98** | 2.81 | **188.9** |
| 16x1024 + parallel attn/FFN block | 16 | 1024 | 142.7 | 489.6 | 4.46 | **2.75** | 177.9 |
| attention-only (SAN shape) | 36 | 1024 | 151.1 | **579.8** | 6.09 | 3.25 | 178.6 |

- **Width beats depth on latency at matched parameters and FLOPs** - narrow-deep is 5.0x slower eager and 1.58x slower graphed, and achieved throughput falls 189 → 123 TFLOP/s. Depth's quality edge in the literature is measured on parametric-recall tasks, the one axis grounding does not use
- **Attention-only is refuted on the latency axis** - at matched parameters it needs 3x the layers and burns **18% more FLOPs** (579.8 vs 489.6), running 1.16-1.30x slower. This closes R4-H26's premise on speed rather than on quality
- **Parallel attention+FFN blocks help modestly and reliably** - 2.806 → 2.752 ms graphed at identical parameters
- **Batching the top-3 chunks as one forward is worth -32%**, and is an execution-order change rather than an architectural one
- MICE-style top-layer cost at T=64 claim tokens only: **0.298 ms** graphed against 2.707 ms for the joint model, which is what motivates R7-H53
- **Verdict** - **Kept as a measurement.** Recommended shape from it: 16 layers x 1024 hidden, GeGLU d_ff 1536, parallel blocks, 128k trimmed vocabulary, 274.9M total (131.1M embeddings + 142.7M body), 2.71 ms/forward graphed. Not a hypothesis verdict - no quality was measured here, only shape and speed

### Sequencing

`R7-H49` is free of GPU and free of the teacher cache, so it runs first the moment Hub auth is available - and it can invalidate every target in this round by showing the private gold is not comparable. `R7-H51` is a prerequisite build, not a hypothesis, and everything downstream waits on it. `R7-H50` is the decisive one: if four checkpoints spanning 140M to 307M all land within 0.02, the task is not capacity-limited, the architecture question is closed, and the remaining work is labels rather than models. `R7-H53` only matters if a joint model ships at all.

### Round 7 results

**R7-H51 regenerate the teacher corpus - PASS**

The cached `full_pairs.npz` carries 111,800 rows of `owner`/`rr`/`nli`/`labels`/`langs` and **no text**, under a chunking this repo can no longer reproduce (R4 amendment: cached mean 40.62 chunks/claim, `recursive_chunk` matching per-claim counts on 39/300 at max_chars=1000). Soft labels could not be re-attached to their inputs, so the corpus was rebuilt under current chunking with the text retained.

| quantity | regenerated | incumbent reference | delta |
|---|---|---|---|
| pairs | 123,579 (44.9 chunks/claim) | 111,800 (40.62) | +10.5% |
| reranker AUC, max-over-chunks | **0.8289** | 0.841 | **-0.012** |
| NLI entailment AUC, max-over-chunks | 0.7674 | 0.806 all-lang | **-0.039** |

- **Verdict** - **PASS on the pre-registered bar** (reranker within 0.02) and the corpus ships as the teacher, at `private-rag-forensics/R7-H51_teacher_pairs.parquet` with `owner`, `claim`, `chunk`, `rerank`, `entail`, `neutral`, `contradiction`, `label`, `lang`
- **Recorded miss, not glossed** - the NLI channel drifts -0.039, outside the tolerance the reranker met. The bar was written against the reranker so the gate passes, but the honest reading is narrower than the gate: this corpus is a faithful teacher **for the reranker signal**, and the NLI channel should not be used as a distillation target without explaining the drift first. Chunk-count shift (44.9 vs 40.62) is the leading suspect
- Harness `experiments/grounding-semantic/R7-H51_regenerate_teacher.py`, log `logs/R7-H51-teacher-corpus.log`, torch fp16 on RTX PRO 6000, ~13 min for both scorers over 123,579 pairs

**R7-H50 capacity ablation - IN PROGRESS, first checkpoint recorded**

Splits are by CLAIM, not by pair, so no student is scored on a claim it trained on: 688 held-out claims (base rate 0.725), 40,000 training pairs subsampled from the remainder, one epoch, teacher signal = the reranker score, aggregation = max-over-chunks.

| checkpoint | params | AUC | macro-F1 | ms/pair |
|---|---|---|---|---|
| mmBERT-small | 140.6M | **0.8772** | 0.7946 | 5.18 |
| mmBERT-base | 307M | pending | pending | pending |
| mDeBERTa-v3-base | 278M | pending | pending | pending |
| mDeBERTa minus 6 layers | ~235M | pending | pending | pending |

- **The 140M student beats its own teacher by +0.048 AUC** (0.8772 against the teacher's 0.8289 on the identical corpus). This contradicts the ceiling assumption both research lanes carried - "a distilled student cannot exceed its teacher" holds for reproducing the teacher's outputs, but the student is scored against HUMAN GOLD, and training on smoothed soft targets averages out per-pair teacher noise. The 0.80-0.84 ceiling estimate in this round's preamble is now in doubt
- **Two caveats attached to the number before anyone quotes it.** The macro-F1 is optimistic: the threshold is chosen to maximise F1 **on the test set**, which is not an honest operating point, so **AUC is the only trustworthy figure here**. And the held-out slice runs base rate 0.725 against the full gold's 0.649, so 0.7946 is not comparable to the incumbent's 0.824 without a matched re-score
- **Verdict** - pending the remaining three checkpoints; the pre-registered bar is a macro-F1 spread ≤ 0.02 across 140M-307M

**R7-H54 fp16 serving latency (pre-registered addition to round 7)**

Added because R7-H50's eval loop produced a latency figure that is not a serving number and must not be quoted as one: the students ran **fp32** (the `from_pretrained` default) while both teachers were timed in **fp16**, inside a DataLoader with variable-length padding. The raw comparison it implied is therefore invalid in the student's disfavour by roughly 2x.

The confound surfaced a real question. Measured at batch 64 during R7-H51 and R7-H50:

| model | params | shape | measured | dtype |
|---|---|---|---|---|
| bge-reranker-v2-m3 | 568M | 24L x 1024 | **3.63 ms/pair** | fp16 |
| mmBERT-small | 140M | 22L x 384 | 5.18 ms/pair | **fp32** |
| mDeBERTa-v3-base | 278M | 12L x 768 | 5.84 ms/pair | fp16 |

- **Parameter count does not predict latency here, and the shape column says why.** mmBERT-small carries 4x fewer parameters than the reranker but nearly the same DEPTH - 22 layers against 24 - and depth is the part that cannot be parallelised. Narrowing 1024 → 384 removes most of the FLOPs and little of the wall-clock, because the GPU goes latency-bound rather than compute-bound at that width, which is the same effect R7-H52 measured as achieved throughput falling 189 → 123 TFLOP/s on narrow shapes
- **mDeBERTa is slower than a model twice its size** - disentangled attention is attention-heavy, so its parameter savings sit outside the bottleneck
- **Consequence for the design** - shrinking parameters does not buy speed; cutting DEPTH does. This makes R7-H50's `mDeBERTa-minus-6L` arm the most informative of the four, since it is the only one that moves depth with width, family and tokenizer held fixed

- Hypothesis - because sequential depth rather than parameter count sets encoder latency, re-timing every candidate at fp16 with `sdpa` and `torch.compile` at fixed shapes will show the sub-350M students clustering with the 568M teacher rather than beating it, and the 6-layer variant separating from all of them
- Prediction - mmBERT-small 1.2-1.6x the teacher's throughput, not the ~4x its parameter count implies; mDeBERTa-minus-6L ≥ 2x
- Acceptance bar - a candidate must be ≥ 2x the teacher at B=3 to justify replacing it on latency grounds alone
- Experiment - `experiments/grounding-semantic/R7-H54_fp16_serving_latency.py`<br>fp16 and bf16, `attn_implementation="sdpa"`, fixed shapes at seq 512, warmup 12, median of 40 reps<br>batches 1 / 3 / 8, where **B=3 is the real serving unit** (top `cfg.semantic_top_k` chunks scored as ONE batch)<br>run: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python experiments/grounding-semantic/R7-H54_fp16_serving_latency.py`

**R7-H55 a stronger teacher raises the student's ceiling (pre-registered)**

R7-H50 measured the student against its teacher on identical held-out traces: `bge-reranker-v2-m3` (568M) reads AUC **0.8619**, the distilled 140M student **0.8269**, a deficit of 0.035. A distilled student is bounded by its teacher, so that 0.8619 is the ceiling every architecture and capacity hypothesis in this round is competing under.

**The teacher carries none of the project's constraints.** It runs once, offline, to label 123,579 pairs, and never serves a request. The sub-350M limit, the latency budget and the CPU export path all apply to the STUDENT. Choosing `bge-reranker-v2-m3` as teacher was inherited from what the cascade happens to ship, which is not a reason - it is an oversight, recorded here as such.

- Hypothesis - because a distilled student's quality is bounded by its teacher's ranking, and the teacher is an offline labeller with no size, latency or export constraint, replacing `bge-reranker-v2-m3` (568M) with the strongest reranker that fits the 96 GB card will raise the teacher's AUC on the held-out traces above 0.8619 and lift the distilled student proportionally
- Lever - the teacher only; student architecture, training recipe, splits and evaluation held exactly at R7-H50 values so the two runs are directly comparable
- Mechanism - soft labels transfer the teacher's ranking, not its parameters. A better ranking is a better target at identical labelling cost, and labelling cost is a one-time offline pass
- Candidates to score before choosing: `BAAI/bge-reranker-v2-gemma` (~2B), `BAAI/bge-reranker-v2-minicpm-layerwise` (~2B), the `Qwen3-Reranker` tier (0.6B / 4B / 8B), `mxbai-rerank-large-v2`, and an ensemble of the strongest two. Licences must be checked - `jina-reranker-v2-base-multilingual` is CC-BY-NC and therefore excluded
- Prediction - the best available reranker reads 0.88-0.92 on the held-out traces against the incumbent's 0.8619; the student closes to within 0.03 of it, landing 0.85-0.89
- Acceptance bar - teacher AUC ≥ 0.88 AND distilled student ≥ 0.85 on the same 159 test traces; a teacher that gains less than 0.02 is not worth a relabelling pass
- Kill condition - if a materially stronger teacher does NOT lift the student, the deficit is the student's capacity or the training budget rather than the label quality, and the round's conclusion changes accordingly
- Pre-experiment probe - score every candidate reranker on the gold FIRST and compare AUC on the held-out traces; only the winner relabels the corpus. Scoring is minutes, relabelling is a full pass
- Experiment - `experiments/grounding-semantic/R7-H55_teacher_search.py`<br>stage 1: rank candidate rerankers on the 159 held-out traces, no relabelling<br>stage 2: winner relabels all 123,579 pairs, replacing the `rerank` column<br>stage 3: re-run R7-H50 unchanged against the new teacher<br>hardware: RTX PRO 6000 96 GB, so up to ~8B dense is in scope for the teacher

**R7-H56 the distillation objective is mismatched to the metric (pre-registered)**

R7-H50 trains with `BCEWithLogitsLoss` against the teacher's sigmoid output and is scored by AUC. Those are different objectives. AUC is a pure ranking statistic - it depends only on whether a supported claim outscores an unsupported one - while BCE is pointwise calibration, penalising a prediction of 0.61 against a target of 0.73 even when every ordering is already correct. The training signal spends capacity on a property the metric ignores.

The saturation problem compounds it. The teacher's most informative pairs are the confident ones, and BCE against probabilities near 0 or 1 carries almost no gradient there, so the clearest teacher judgements transfer the most weakly.

Serving makes it a ranking problem twice over: the verdict is `max-over-chunks`, so what actually matters is whether the BEST chunk of a supported claim outranks the BEST chunk of an unsupported one. That is a claim-level ordering objective, and nothing in the current loss expresses it.

- Hypothesis - because the evaluation metric is a ranking statistic and the serving rule is max-over-chunks, replacing pointwise BCE on the teacher's probability with (a) MSE on the teacher's LOGITS and (b) a claim-level pairwise ranking term will raise the distilled student's AUC by at least 0.02 at identical teacher, data, architecture and budget
- Lever - the loss function only; teacher, corpus, splits, architecture, optimiser and step count all held at R7-H50 values
- Mechanism - logit-space MSE preserves the teacher's full dynamic range where the sigmoid has compressed it, which is the standard reason distillation is done on logits rather than probabilities; the ranking term optimises the quantity the metric actually reports
- Arms to compare, one training run each on the same 385 training traces:
  - **A** BCE on teacher probability - the R7-H50 baseline, AUC 0.8269
  - **B** MSE on teacher logits, temperature 1
  - **C** MSE on teacher logits with temperature T ∈ {2, 4} to soften the target
  - **D** B plus a pairwise margin-ranking term over positive/negative claim pairs sharing a batch
  - **E** D plus the 2,752 hard gold labels as an auxiliary claim-level term, weighted low
- Prediction - B beats A by 0.01-0.02; D beats A by 0.02-0.04; E adds under 0.01 because 2,752 claim-level labels are too few to move a pair-level objective
- Acceptance bar - the best arm must clear A by ≥ 0.02 AUC on the held-out 159 traces to justify the added complexity; below that, BCE stays for simplicity
- Kill condition - if all arms land within 0.01 of A, the objective is not the constraint and attention returns to the teacher (R7-H55) and the data
- Experiment - `experiments/grounding-semantic/R7-H56_distill_objective.py`, mmBERT-small only (the cheapest arm at 4.92 ms/pair and the current best student), five runs, same seed, same splits

**Sequencing note for R7-H55 and R7-H56.** These are independent levers on the same deficit - the teacher sets the ceiling, the objective sets how much of it transfers - and they must be run SEPARATELY before any combination, or a gain cannot be attributed. R7-H56 is the cheaper of the two (no relabelling pass) and runs first.

**R7-H57 does a public-data verifier transfer to our distribution? (pre-registered)**

The dataset survey (`reports/research-grounding-datasets.md`) recommends training on RAGTruth, LettuceDetect-prose and RAGBench, and estimates a 50-100x expansion in distinct source documents against our 619. It also warns that domain transfer will disappoint, because none of those corpora are production assistant traffic over a private corpus.

That warning is testable at zero training cost. `KRLabsOrg/lettucedect-v2-mmbert-base` (MIT, 307M, mmBERT-base backbone) is trained on exactly the recommended stack. Running it on OUR gold measures public → private transfer directly, before a single GPU-hour is committed to training on public data.

- Hypothesis - because the recommended public corpora differ from our traffic in register, document structure and retrieval failure mode, a verifier trained on them will underperform our cascade on the English slice while transferring better to the non-English slices, where our own 639-trace supervision is thinnest
- Lever - the training distribution, held at arm's length: no training, no fine-tuning, frozen public checkpoint against our gold
- Prediction - English AUC 0.65-0.75 against our cascade's 0.8619 on held-out traces; non-English closer, within 0.05 of its own English figure, because its training data covers 14 languages and ours effectively covers one
- Acceptance bar - if it lands within 0.05 of our cascade on ANY language slice, public training data transfers and the RAGTruth track is justified; if it collapses below 0.65 everywhere, public data is a different problem and the budget belongs on annotating private traces instead
- Why this is the right probe - it is the reverse direction of the expensive experiment. Training on public data and testing on ours costs a day; taking a model someone already trained on public data and testing on ours costs an inference pass
- **Format audit first, per the standing rule** - LettuceDetect is a token tagger over (question, context, answer) triples, not a pairwise classifier. The mapping onto our (claim, chunk) task must be read off its model card and verified on a positive control before any number is taken. Round 4 lost a day to exactly this
- Experiment - `experiments/grounding-semantic/R7-H57_public_verifier_transfer.py`, our 2,752-claim gold, per-language breakdown, run on the idle RTX 5000 Ada so it does not contend with the depth probe

**R7-H58 does Search Arena carry its evidence? (pre-registered, 20-minute gate)**

`lmarena-ai/search-arena-24k` is the only public corpus of genuine retrieval-augmented user traffic: 24,069 multi-turn search-assistant conversations, ~11k users, 136 countries, ~90 languages. It is the only candidate that broadens the TRACE distribution rather than the document distribution, which is our actual constraint at 639 traces.

Its usefulness rests entirely on one unverified fact: whether `system_{a,b}_metadata.web_search_trace` contains scraped page TEXT or merely URLs and citation counts. A conversation without its evidence cannot be grounded. The HF dataset viewer fails at 400 MB, so nobody has confirmed it.

- Prediction - the trace carries URLs and citation metadata but not full page text, because storing scraped page bodies for 24k multi-turn conversations is a large payload and the schema mentions a scrape engine without a text field
- Acceptance bar - page text present → Search Arena becomes the primary Class B target and a licence read follows; text absent → the Class B lane collapses to MIRAGE-Bench and NoMIRACL, both of which are Apache-2.0 with passages attached
- Cost - one shard, inspection only, no training and no GPU
- Note - even on a PASS, the licence is unresolved: prompts are CC-BY-4.0 but model outputs are "governed by respective provider terms", which is a legal question rather than an engineering one and must not be answered by this project unilaterally

### Round 7 results - part 2

**R7-H50 capacity ablation - Refuted (the task IS capacity-limited), and the first version was void**

Recorded in full because the correction matters more than the result. Three versions ran:

| version | split | mmBERT-small 140M | mmBERT-base 307M | status |
|---|---|---|---|---|
| v1 | by CLAIM | AUC 0.8772 | AUC 0.8573 | **VOID - leaked** |
| v2 | by TRACE | **AUC 0.8269** | **AUC 0.8479** | recorded |

- **v1 leaked at the document level and INVERTED the conclusion.** The gold's 2,752 claims come from only **639 traces sharing 619 source documents** (4.31 claims/trace, largest trace 35). A claim-level split puts claim A of a trace in train and claim B in test, so the student trains on the very document it is scored on. Under that leak the 140M model appeared to BEAT the 307M model; with trace-level splits the ordering reverses. The leak was worth **0.050 AUC** and would have sent us building the wrong model
- The log's own situational overview had warned about exactly this - "claims sharing a trace's evidence are correlated, so the effective sample is smaller than the record count" - and the first version ignored it. The effective sample is **639, not 2,752**
- **v2 protocol** - split by `trace_id` 60/15/25, threshold fitted on the VALIDATION traces and applied unchanged to test (v1 fitted it on test, which inflated macro-F1), plus warmup, gradient clipping and a divergence guard
- **Reproducibility check** - mmBERT-base-22L was trained three times across runs: **0.8479 / 0.8485 / 0.8502**, spread 0.0023. Run-to-run noise is small, so the 0.021 gap between 140M and 307M is signal
- **Verdict** - **Refuted.** The pre-registered bar was a macro-F1 spread ≤ 0.02 meaning "not capacity-limited"; measured AUC spread is **0.021** and it points UP with size. Capacity matters, mildly. Architecture work is not closed
- **mDeBERTa-v3-base could not be trained** - diverged to NaN at step 200 at lr 2e-5, and again at lr 1e-5 with warmup and clipping. It is excluded rather than fought; the depth arm moved to mmBERT truncation, which trains stably

**Matched-teacher comparison, the number that matters.** All on the identical 159 held-out traces / 717 claims / base rate 0.654:

| model | params | AUC |
|---|---|---|
| `bge-reranker-v2-m3` (teacher, frozen) | 568M | **0.8619** |
| mmBERT-base distilled | 307M | 0.8479 (-0.014) |
| mmBERT-small distilled | 140M | 0.8269 (-0.035) |
| `mDeBERTa-v3` NLI entailment (frozen) | 278M | 0.7601 |

- A 307M student lands **0.014** from a frozen 568M teacher at 54% of the size, on one epoch over 40,000 pairs from 385 traces. The earlier "student beats teacher by +0.048" was **entirely leakage**; clean, the student loses
- Both students already beat the 278M NLI model the cascade ships, by 0.067 and 0.087

**R7-H50 depth probe - depth is purchasable, not idle (2 of 4 arms)**

mmBERT-base truncated within one family, so width, tokenizer, embeddings and recipe are all fixed and only depth moves. This is the arm mDeBERTa was meant to provide.

| depth | params | AUC | macro-F1 | ms/pair |
|---|---|---|---|---|
| 22L | 307.5M | 0.8502 | 0.783 | 9.20 |
| 11L | 252.4M | 0.8183 | 0.751 | **5.38** |
| 6L | pending | | | |
| 3L | pending | | | |

- Halving depth costs **0.032 AUC** and buys **1.7x** speed. Depth is doing real work - it is not idle, and the wide-shallow thesis does not get a free win on quality
- R7-H52 measured that width is FASTER than depth at matched parameters; this measures what depth is WORTH. The two together make the shape a genuine trade rather than a free lunch, and the trade cannot be priced until an accuracy floor is set

**R7-H57 public-trained verifier on our gold - PARTIAL, and my prediction was wrong**

`KRLabsOrg/lettucedect-v2-mmbert-base` (MIT, 307M, trained on RAGTruth + its translations + LettuceDetect-prose).

- **Positive control: AUC 1.0000, accuracy 1.00** on the same 20 trivially separable pairs where Baguettotron scored 60% and Pleias-RAG-350M 55%. The model is competent and the harness is validated, which is what gives the negative result below its force
- **On our gold: AUC 0.7095, macro-F1 0.6313** against our cascade's 0.8619 on the same held-out traces - a deficit of **-0.152**
- **Prediction refuted.** I predicted non-English would transfer BETTER than English; measured EN 0.7038 against a non-EN mean of 0.4030
- **But the non-English slices cannot support that finding and are not reported as one.** They run n=26 to 44 at base rates 0.885-0.973 - the Norwegian slice has roughly ONE negative claim, so its 0.1389 is noise, and Norwegian is not in LettuceDetect's training languages at all. **English is the only slice here with enough negatives to measure**
- **This exposes a defect in our own gold, not just in the candidate** - our non-English evaluation is unusable in either direction, at any sample size. That is a bigger problem than which corpus to train on

**R7-H59 cross-domain transfer matrix - both directions fail, asymmetrically**

1,200 RAGTruth test responses, grounded rate 0.654 against our gold's 0.649, so class balance is controlled. Threshold fitted on half A, reported on half B, for every cell.

| model | our private gold | RAGTruth (public) |
|---|---|---|
| our cascade (reranker) | **0.8619** | **0.6432** · F1 0.5947 |
| `lettucedect-v2` (public-trained) | 0.7095 · F1 0.6313 | 0.7039 · F1 0.6381 |

- **Our cascade collapses off-domain: -0.219.** Predicted 0.75-0.80, measured 0.643 - wrong again, in the same optimistic direction. The fitted threshold came out at **0.983**, i.e. the reranker pushes nearly everything to the top of its range on foreign data: calibration failure, not just accuracy loss
- **LettuceDetect is FLAT: 0.7095 versus 0.7039**, a difference of 0.006. Trained on public data, it performs identically on ours. Domain-general but capped near 0.70
- **On RAGTruth it beats us**, 0.7039 to 0.6432
- **Our 0.8619 is domain specialisation, not general grounding capability.** It does not travel and must not be quoted as if it does
- **This answers R7-H49 from a side door while that stays blocked on Hub auth.** A neutral third-party model finds both corpora equally hard (0.7095 vs 0.7039), so our gold is **not unusually easy** - the benchmark survey's suspicion on that point is refuted. But its *prediction* was right: it forecast 63-72 for us on public benchmarks and our cascade scored **64.3** on RAGTruth, itself an LLM-AggreFact component. Both hold - the gold is fair, and the score does not transfer
- **Strategic consequence** - transfer fails in BOTH directions, so the ~140k RAGTruth examples are unlikely to lift our gold number. Our advantage is domain specialisation, which makes more PRIVATE traces the lever, not more public rows. The dataset survey's warning was right and understated

**Datasets fetched** - `data/external/datasets/`, via `scripts/fetch_grounding_datasets.py`. RAGTruth (15,090 train / 2,700 test, MIT) and its 7 translations (124,530 rows, MIT). Sidecars are tracked in git and travel inside each archive; the archives are gitignored. The survey covering all eight licence-clean corpora, and the four large ones excluded on licence (TrueTeacher, HaluBench, MS MARCO, ANLI) plus MEMERAG which forbids training in terms, is `reports/research-grounding-datasets.md`.

## Hypothesis round 8 - beat LettuceDetect on all three corpora (pre-registered)

Batch **M**, ids `R8-H61..R8-H68`. Round 7 replaced a vague target with a measured one. `KRLabsOrg/lettucedect-v2-mmbert-base` is 307M, MIT, in-band, and it is the incumbent to beat because it is the only public model measured on all three corpora under one harness.

### The objective, stated as three numbers

| corpus | LettuceDetect | our cascade | gap |
|---|---|---|---|
| private gold, 159 held-out traces | 0.7095 (F1 0.6313) | **0.8619** | **+0.152 - we win** |
| RAGTruth EN, 1,200 responses | **0.7039** (F1 0.6381) | 0.6432 (F1 0.5947) | **-0.061 - we lose** |
| RAGTruth non-EN, mean of 7 languages | **0.6095** | 0.5626 | **-0.047 - we lose** |

**Win condition: strictly above all three, simultaneously, with one model.** Beating any one in isolation is easy and uninteresting - a domain-specialised model already wins the first, and a public-trained model already wins the other two. The whole difficulty is holding the +0.152 on our own domain while closing -0.061 and -0.047 elsewhere, and the two objectives pull against each other.

### Methodology - what round 8 adds

- **Naive baseline** unchanged at macro-F1 0.417. Every number above is AUC on the harness in `R7-H59` / `R7-H60`, threshold fitted on half A and reported on half B
- **NEW - three-corpus reporting is mandatory.** No hypothesis in this round may report a single number. A result on one corpus without the other two is not a result, because round 7 established that our headline transfers to neither
- **NEW - the multilingual labels are suspect and gate the third objective.** In `R7-H60` both models degraded by the SAME amount on the translations (-0.091 ours, -0.094 theirs) even though LettuceDetect was TRAINED on those very train splits. Training should have protected it and did not, which points at machine-translated label alignment rather than model capability. `R8-H61` settles this before any multilingual work is funded
- **Held fixed** - the R7-H51 teacher corpus, trace-level splits, max-over-chunks aggregation, and mmBERT-base as the student backbone unless a hypothesis explicitly varies it

### Pre-registration

| id | persona | technique | prediction | acceptance bar |
|---|---|---|---|---|
| R8-H61 | deflationist | verify translated labels on the 300 human-checked German rows | the -0.09 drop shrinks by more than half on verified labels | drop halves → the multilingual objective is mis-specified and its bar is re-derived |
| R8-H62 | follower | multi-corpus distillation - our teacher labels + RAGTruth EN + 7 translations | gold 0.83-0.86, RAGTruth EN 0.70-0.75, non-EN 0.62-0.67 | all three above the bars, gold ≥ 0.84 |
| R8-H63 | mechanist | per-corpus rank normalisation before thresholding | recovers most of the F1 gap at zero training cost | F1 +0.05 on RAGTruth with AUC unchanged |
| R8-H64 | hybridizer | ensemble our cascade with LettuceDetect | if their errors decorrelate, the mean beats both everywhere by construction | beats both on all three; error correlation reported |
| R8-H65 | scout | translate-then-verify via the shipped argos MT bridge | non-EN 0.5626 → within 0.03 of the English figure | non-EN ≥ 0.6095 using English-only inference |
| R8-H66 | contrarian | token-level span supervision instead of pairwise scores | span supervision is what makes LettuceDetect domain-general, not its data | RAGTruth EN ≥ 0.70 while gold holds ≥ 0.84 |
| R8-H67 | heretical | do not train one model - route by domain and language | routing beats any single model on all three trivially | tests whether "one model" is even the right objective |
| R8-H68 | mechanist | near-miss negatives generated from OUR corpus, VitaminC-style | keeps domain, adds hard negatives, no licence exposure | gold ≥ 0.87 and RAGTruth EN ≥ 0.68 |

### Sequencing

**R8-H61 runs first and can re-specify the round** - if the translated labels are unreliable, the -0.047 multilingual bar is measuring translation noise and must be re-derived before anything is trained against it. **R8-H63 and R8-H64 are next because neither requires training**: rank normalisation is arithmetic, and the ensemble is two frozen models already scored on all three corpora, so their error correlation is a free re-analysis of data already on disk. R8-H65 reuses the MT bridge already shipped in the lexical tier. Only then do the training hypotheses run - R8-H62 as the straightforward multi-corpus baseline, R8-H66 and R8-H68 as the two mechanistic bets about WHY LettuceDetect generalises. R8-H67 is deliberately last: if a router beats every trained model, the round's premise was wrong and that is worth knowing after the honest attempts, not before.

### Amendment - round 8 extended with anisotropy removal

Append-only. Nothing recorded above changes; these are added hypotheses, and the round's centre of gravity is restated: **distillation is the backbone** (R8-H62, H66, H68), fine-tuning is the optional second stage, and anisotropy removal is added as a third family because round 7 produced a specific symptom that names it.

**The symptom.** In R7-H59 the reranker's fitted threshold on RAGTruth came out at **0.983** - it pushes nearly every off-domain pair to the top of its range. That is dynamic-range collapse, not merely accuracy loss, and it is the pathology all-but-the-top postprocessing was built for (Mu and Viswanath, 2018: subtract the mean, then project out the top-D principal components). The same track's prior art applies - the docdistance work used exactly this to widen compressed statement cosines.

**Why the cross-lingual version is the most promising of the three.** Multilingual encoders place different languages in different cones of the representation space, so a per-language mean is a nuisance direction rather than signal. If the -0.047 multilingual gap is partly that offset, it is removable by arithmetic with **no training and no labels** - which would be the cheapest result in the round by a wide margin.

| id | persona | technique | prediction | acceptance bar |
|---|---|---|---|---|
| R8-H69 | mechanist | all-but-the-top on the bi-encoder gate - subtract the mean, project out top-D components, D ∈ {1,2,3,5} | cosine dynamic range widens ≥ 1.5x; gate AUC 0.73 → 0.78-0.83 | gate AUC +0.03 on all three corpora at some single D |
| R8-H70 | scout | per-LANGUAGE centroid removal for cross-lingual alignment | languages occupy separate cones; removing the per-language mean aligns them | non-EN AUC 0.5626 → ≥ 0.6095 with NO training, on the parallel corpus |
| R8-H71 | hybridizer | anisotropy removal on the distilled student's pooled representation before the head | the student inherits the teacher's compressed geometry | ≥ +0.02 AUC over the same student untreated, on all three |
| R8-H72 | mechanist | score-space recalibration - per-corpus rank normalisation of the FINAL score | strictly weaker than H69-H71 but nearly free; separates score compression from representation compression | isolates whether the 0.983 threshold is a scoring or a representation artefact |

**Ordering within the round.** R8-H70 runs first of these four: it is training-free, label-free, targets the objective we are furthest from, and its result tells us whether the multilingual gap is geometry or capability. R8-H69 follows because it is also training-free and reuses the same embedding pass. R8-H72 is arithmetic on the R8 substrate and costs nothing. R8-H71 requires a trained student and therefore waits on R8-H62.

**Honest note on what anisotropy removal cannot do.** It is a postprocessing transform on a frozen representation. If the models genuinely cannot read non-English evidence - as opposed to reading it into a displaced subspace - projecting out components will not create the capability, and R8-H70 will return flat. A flat result there is informative: it moves the multilingual objective squarely into training, where R8-H62 and R8-H66 live.

### Amendment - round 8 extended with a multi-head adversarial trunk

Append-only, prompted directly by the R8-H64 result: our cascade and `lettucedect-v2` are **almost perfectly decorrelated** (Spearman -0.046 to +0.083 across all ten corpora), and rank-average fusion therefore beats BOTH on RAGTruth EN (0.7479 vs 0.7030) and non-EN (0.6212 vs 0.6095) with no training at all.

That is the finding these hypotheses generate from. Fusion is not shippable - it is 568M + 307M and fails the sub-400M single-model requirement outright - but it proves the two signals exist, are complementary, and are individually learnable. The task becomes putting both inside ONE trunk.

**The design.** One mmBERT-base trunk (307M, in-band) carrying two supervision heads plus an adversarial third:

- **Head A - relevance/entailment**, regressing our cascade's soft per-pair score. This is the signal the reranker reads and the one our domain advantage lives in
- **Head B - span faithfulness**, token-level classification against RAGTruth's human span annotations. This is the signal LettuceDetect reads, and R8-H64 says it is nearly orthogonal to Head A
- **Head C - adversarial discriminator** predicting language or corpus through a GRADIENT REVERSAL layer. The trunk is trained to make that prediction impossible, which is the standard domain-adversarial construction and the literal GAN-like element: discriminator and trunk optimise opposing objectives
- **Then post-alignment**, the trunk is frozen and a fresh classification head is fitted on the aligned representation

| id | persona | technique | prediction | acceptance bar |
|---|---|---|---|---|
| R8-H73 | hybridizer | two heads on one trunk - soft-score regression + token-span tagging | internalises the fusion; the trunk must represent both decorrelated signals | beats the better single head on ALL three corpora, and lands within 0.02 of the external fusion |
| R8-H74 | heretical | adversarial LANGUAGE discriminator with gradient reversal | a language-invariant trunk transfers English grounding to the other 7 | non-EN 0.5626 → ≥ 0.66 without adding non-English supervision to Head A |
| R8-H75 | mechanist | adversarial CORPUS discriminator (gold vs RAGTruth) with gradient reversal | domain-invariance closes the -0.061 cross-domain gap | RAGTruth EN ≥ 0.75 while gold holds ≥ 0.84 |
| R8-H76 | follower | post-alignment head - freeze the aligned trunk, fit a fresh classifier | the aligned representation is better than the jointly-trained head that produced it | ≥ +0.01 over the joint head on all three; also gives a cheap per-domain re-fit |

**Why the adversarial arm is the interesting bet, and where it can fail.** R8-H61 established that roughly two thirds of the multilingual gap is genuine capability rather than translation-label noise, so it will not be fixed by cleaning labels. Gradient reversal attacks it from the representation side and needs NO additional non-English grounding supervision - only language tags, which are free. The failure mode is equally clear and must be watched: an adversarial objective strong enough to erase language also erases meaning, so the discriminator weight lambda is the load-bearing hyperparameter and a lambda sweep is part of the experiment rather than an afterthought. A trunk that becomes language-invariant AND grounding-blind is the expected failure, not a surprise.

**Ordering.** R8-H73 first - the two-head trunk without adversarial training is the honest baseline, and if it alone captures the fusion gain the adversarial complexity is unnecessary. R8-H74 and R8-H75 add one discriminator each, separately, never together, so a gain can be attributed. R8-H76 runs last on whichever trunk wins.

### Round 8 results - part 1 (training-free)

**R7-H50 depth probe - complete. Depth is load-bearing; the wide-shallow shortcut is closed on quality.**

mmBERT-base truncated within one family - width, tokenizer, embeddings and recipe all fixed, only depth moves.

| depth | params | AUC | macro-F1 | ms/pair |
|---|---|---|---|---|
| 22L | 307.5M | **0.8502** | 0.7830 | 9.20 |
| 11L | 252.4M | 0.8183 | 0.7510 | 5.38 |
| 6L | 227.3M | 0.7332 | 0.6645 | 3.16 |
| 3L | 212.2M | 0.6093 | 0.5996 | 1.75 |

- The curve **accelerates downward**: -0.032 for the first halving, -0.085 for the second, -0.124 for the third. At 3 layers the model is near chance for this task
- Pre-registered bar was a spread ≤ 0.02 meaning "not capacity-limited". Measured macro-F1 spread **0.1834** → **REFUTED**, decisively
- **R7-H52 measured that width is FASTER than depth; this measures what depth is WORTH, and it is worth a lot.** The two together make the shape a real trade, not a free win. 11L is the interesting operating point - 0.8183 at 5.38 ms, still +0.109 over the incumbent on gold at 1.7x the speed
- Verdict - **keep 22 layers.** Recommended spec revised: depth is not the place to economise

**R8-H61 multilingual label gate - the gap is REAL, and my hypothesis was half right**

`KRLabsOrg/ragtruth-de-translated-manual-300` is the only human-VERIFIED translation data that exists.

| corpus | n | base | cascade | lettuce |
|---|---|---|---|---|
| RAGTruth EN | 600 | 0.650 | 0.6537 | 0.7030 |
| RAGTruth DE, machine-translated labels | 600 | 0.650 | 0.5791 | 0.6032 |
| RAGTruth DE, human-VERIFIED labels | 300 | 0.663 | 0.4634 | **0.6351** |

- LettuceDetect's EN→DE drop is **0.0999** on MT labels and **0.0680** on verified labels. The drop shrinks by about a third but does NOT halve
- **Verdict - the pre-registered bar is not met: roughly one third of the multilingual gap is translation-label noise, two thirds is genuine capability.** The -0.047 multilingual objective stands, with that asterisk. Cleaning labels will not fix it and training must
- Caveat recorded: the verified subset is 300 rows against 600, different sample, so the two drops are not perfectly matched

**R8-H64 ensemble headroom - the two models are ORTHOGONAL, and fusion beats both**

Per-example scores from the R8 substrate, rank-normalised before fusing because the two scores live on different scales.

| corpus | cascade | lettuce | Spearman | MEAN-fusion |
|---|---|---|---|---|
| RAGTruth EN | 0.6537 | 0.7030 | +0.073 | **0.7479** |
| RAGTruth de | 0.5791 | 0.6032 | -0.001 | 0.6310 |
| RAGTruth fr | 0.5650 | 0.6118 | -0.010 | 0.6244 |
| RAGTruth es | 0.5986 | 0.5912 | +0.049 | 0.6377 |
| RAGTruth it | 0.5386 | 0.5901 | -0.046 | 0.5888 |
| RAGTruth pl | 0.5503 | 0.6020 | -0.002 | 0.6057 |
| RAGTruth hu | 0.5758 | 0.6032 | -0.046 | 0.6272 |
| RAGTruth cn | 0.5310 | 0.6653 | +0.033 | 0.6336 |
| **non-EN mean** | 0.5626 | 0.6095 | ~0.00 | **0.6212** |

- **Spearman correlation is effectively ZERO on every corpus.** A cross-encoder trained on relevance and a token tagger trained on span faithfulness read genuinely different evidence. This is the round's most useful mechanistic finding
- Fusion clears two of the three bars with **no training**: RAGTruth EN 0.7479 vs 0.7030 (+0.044), non-EN 0.6212 vs 0.6095 (+0.012)
- **It is NOT the deliverable** - 568M + 307M fails the sub-400M single-model requirement outright. What it establishes is that the signal to beat the incumbent exists, is complementary, and is individually learnable. That is what R8-H73 tries to place inside one trunk
- **Defect found and recorded: the `gold` rows of this analysis are VOID.** `R8_score_substrate.our_gold()` sliced `chunk[:semantic_top_k]` - the first three chunks in dataframe order, i.e. arbitrary evidence rather than retrieved evidence - and read the cascade at 0.6739 against the 0.8619 R7-H50 measured by taking max over ALL of a claim's chunks. Gold claims carry a mean of **36.3** chunks, so the cascade was given 8% of its evidence. Fixed to use every chunk; the gold cells above are excluded rather than corrected in place, and the substrate is being re-scored
- This is the same class of error as the R7-H50 claim-level leak: a harness detail that silently changes the task rather than failing loudly

### Round 8 results - part 2: the distillation run

**R8-H62 multi-corpus distillation - WIN CONDITION MET on the trace split, with one corpus caveated**

mmBERT-base (307.5M, under the 400M ceiling and size-matched to the incumbent), trained one epoch over 40,000 private pairs carrying SOFT teacher labels mixed with ~43,000 public pairs carrying HARD human labels (RAGTruth EN 15k + 4k per translation), evaluated on all three corpora with the R7-H59 / R7-H60 harness unchanged. Checkpoint at `models/R8-H62-mmbert-multicorpus` - R7-H50 deleted every student it trained, this one persists.

| corpus | ours | lettucedect-v2 | delta | decisive bar | |
|---|---|---|---|---|---|
| private gold | **0.8531** | 0.7095 | **+0.1436** | 0.76 | DECISIVE |
| RAGTruth EN | **0.8434** | 0.7039 | **+0.1395** | 0.75 | DECISIVE |
| RAGTruth non-EN, mean of 7 | **0.8407** | 0.6095 | **+0.2312** | 0.66 | DECISIVE |

Per language: de 0.8325, fr 0.8336, es 0.8419, it 0.8472, pl 0.8402, hu 0.8279, cn 0.8614.

- **The multilingual gap did not close - it vanished.** English 0.8434 against a non-English mean of 0.8407, a spread of 0.003. Our cascade sat at 0.5626 non-English hours earlier, barely above chance. R8-H74's gradient-reversal adversarial trunk was registered specifically to attack that gap and is now very likely unnecessary: **plain multilingual supervision was sufficient**, and the gap was a training-data problem rather than a representation-geometry one. mmBERT's 256k multilingual vocabulary carried the rest
- **Domain specialisation survived generalisation.** The stated risk was that public data would erode the +0.152 advantage on our own gold; it moved to +0.1436 while RAGTruth EN went from -0.061 to +0.1395. The two objectives moved TOGETHER rather than trading off, which the pre-registration did not predict
- **The student is broadly better than its own teacher.** The 568M reranker reads 0.8619 on gold and collapses to 0.6432 on RAGTruth; this 307M student reads 0.8531 and 0.8434. It gives up 0.009 on the teacher's home ground at 54% the size and gains 0.200 everywhere else

**Leak audit, run after the result rather than assumed - and it qualifies one corpus**

| surface | finding | verdict |
|---|---|---|
| RAGTruth EN train vs test | 0 context overlap, 2,514 train / 450 test contexts | **clean** |
| RAGTruth translations | split boundaries identical to English (15,090 / 2,700), 0 prompt overlap | **clean** |
| private gold, trace split | 0 trace overlap, but **29 exact (claim, chunk) pairs** shared of 15,313 test pairs (0.19%), 10 claim texts repeated, and **95.8% of test claims carry at least one chunk seen in training** (42% of their chunks on average) | **caveated** |

- **The two bars we were losing are the clean ones.** RAGTruth EN and non-EN carry no measurable contamination, so +0.1395 and +0.2312 stand without qualification. That is the substantive result
- **The gold figure is not apples-to-apples against LettuceDetect.** Exact-pair leakage at 0.19% is too small to move an AUC of 0.85, but our model has seen these documents and the incumbent never has. In deployment that is the CORRECT condition - a grounder serves one corpus repeatedly - but as a head-to-head it flatters us, and the number is recorded with that attached rather than quoted bare

**The deeper finding: our gold cannot measure generalisation at all**

Chasing the leak to its root produced a more consequential result than the leak itself. Three splits were tried and audited:

| split unit | chunk overlap | pair overlap | note |
|---|---|---|---|
| claim | severe | severe | INVERTED the R7-H50 capacity ordering, worth 0.050 AUC |
| trace (639 units) | 759 | 29 | different traces retrieve the same passage |
| source document (619 units) | 815 | 125 | distinct source texts share identical passages - boilerplate, repeated sections |
| **connected component over shared chunks (39 units)** | **0** | **0** | the only split with no leakage |

- The leak unit is the **chunk**, not the claim, the trace or the document. `R8_splits.py` computes the transitive closure of "shares a chunk" by union-find, which is the smallest unit that can be split cleanly, and it is now the single definition imported everywhere - the trace-split failure was caused by two implementations of "the same" split drifting apart
- **The corpus is one interconnected mass: a single component holds 2,534 of 2,752 claims (92%).** The effective independent sample is ~39 units, not 639 traces or 619 documents. A clean split therefore leaves 218 test claims and NO validation set
- **Consequence, stated plainly: every number this project has produced on its own gold has been measured under evidence overlap**, today's 0.8531 included. That is the deployment condition and not a broken experiment, but it is not a measurement of generalisation to unseen documents and must stop being read as one
- **This is the strongest argument in the log for acquiring more private traces - for measurability, not accuracy.** We currently cannot evaluate generalisation on our own data at any sample size, and no modelling choice fixes that

**Caveats carried forward** - one run at one seed (measured noise 0.0023, so the margins are ~60x it, but a seed sweep is owed); RAGTruth test is in-domain-adjacent since training used its train split, so the honest external number remains R7-H49 on LLM-AggreFact, still blocked on Hub auth; and R8-H61 established that roughly a third of the non-English test labels are machine-translation noise, so 0.8407 is measured against imperfect ground truth.

**R8-H77 the unseen arena - we LOSE on genuinely blind data**

RAGBench, ten subsets of enterprise-shaped documents, in NEITHER model's training data. `lettucedect-v2` trains on RAGTruth + translations + LettuceDetect-prose; our student on our gold + RAGTruth + translations. This is the first blind test for both and the first number in the log that measures generalisation rather than specialisation.

| subset | ours (R8-H62) | lettucedect-v2 | delta | base rate |
|---|---|---|---|---|
| finqa | **0.3974** | 0.7170 | **-0.3196** | 0.920 |
| delucionqa | 0.5325 | 0.7929 | **-0.2604** | 0.935 |
| tatqa | 0.5118 | 0.6156 | -0.1038 | 0.944 |
| hagrid | 0.5416 | 0.5992 | -0.0576 | 0.848 |
| covidqa | 0.6916 | 0.7355 | -0.0439 | 0.841 |
| pubmedqa | 0.5665 | 0.5162 | +0.0503 | 0.692 |
| emanual | 0.6495 | 0.5999 | +0.0496 | 0.894 |
| hotpotqa | 0.6514 | 0.5976 | +0.0538 | 0.932 |
| techqa | 0.6985 | 0.6363 | +0.0622 | 0.564 |
| expertqa | 0.7148 | 0.6503 | +0.0645 | 0.468 |
| **mean** | **0.5956** | **0.6461** | **-0.0505** | 5/10 won |

- **Verdict - REFUTED for generalisation.** The R8-H62 3/3 win stands as recorded but is bounded to corpora we had training exposure to. On blind data LettuceDetect generalises better by 0.0505
- **`finqa` at 0.3974 is BELOW chance** - anti-predictive, not merely weak. That is a capability failure, not a calibration one
- **Truncation was hypothesised and REFUTED.** Our student trains at max_length 512 against LettuceDetect's 4096, which looked like the obvious mechanical cause. Re-scoring at 2048 moved finqa only 0.398 → 0.428, left tatqa and hagrid unchanged, and made **techqa WORSE** (0.703 → 0.641) - and techqa carries the LONGEST documents in the benchmark at 3,730 chars while being one of our wins. Length is not the variable
- **Content type is.** The four worst subsets are financial tables (finqa, tatqa), a car manual and multi-document wiki. **`tatqa` documents average 399 characters and still score at chance**, which rules out context entirely: the model does not know what to do with a TABLE. Our training mix is RAG prose plus news summaries and carries no tabular or numeric-reasoning supervision at all
- **Where we DO win is informative** - expertqa (base 0.468) and techqa (0.564) are the two subsets with balanced classes, and we take both. The five losses all sit at base rates 0.84-0.94, where a handful of negatives decide the AUC
- Caveat - labels are GPT-4o rather than human, which caps the absolute numbers, but it biases both models identically so the comparison holds
- Harness `experiments/grounding-semantic/R8-H77_unseen_arena.py` takes `--model` so every later incarnation runs the identical gate and successive students stay directly comparable

**R8-H78 incarnation 2 (running)** - adds RAGBench TRAIN across all ten domains (~30k pairs) to close the tabular gap. **Note the consequence, recorded before the result exists: once we train on RAGBench-train, RAGBench-test is no longer blind for us while LettuceDetect still has not seen it.** R8-H77 is therefore demoted to an in-domain reading and the fair arena MOVES to HaluEval (24,507 rows, unseen by both). Reporting a RAGBench-test win after training on RAGBench-train would be the same home-field error this round was built to avoid.

### Amendment - the adversarial terms, re-scoped after R8-H77

Append-only. R8-H77 inverted which adversarial term is worth running, so the earlier registrations are re-stated rather than left to be read against obsolete bars.

**The principle that decides it: gradient reversal substitutes for supervision you cannot obtain.**

- **Language (R8-H74) - CLOSED, and correctly so.** We could obtain the labels. RAGTruth's seven translations supplied them, plain supervision took the EN/non-EN spread to 0.003, and no adversarial machinery was needed. Running it now would be running it for completeness
- **Domain (R8-H75) - PROMOTED to the round's primary architectural hypothesis.** We can never obtain labels for the NEXT unseen domain, which is exactly what R8-H77 measured: RAGBench blind, mean 0.5956 against 0.6461, `finqa` below chance at 0.3974. Supervision cannot reach a domain we have not met, and that is the precise condition adversarial invariance exists for

The term also changes shape. Two domains is domain ADAPTATION; we have roughly twelve source domains (our private traces, RAGTruth news/QA/summary, and RAGBench's ten), which makes this multi-source domain GENERALISATION - an N-way discriminator rather than a binary one.

| id | persona | technique | prediction | acceptance bar |
|---|---|---|---|---|
| R8-H79 | heretical | N-way DOMAIN discriminator through a gradient reversal layer, ~12 source domains | domain-invariant features transfer to corpora never seen in training | blind-arena mean ≥ 0.68 (beats 0.6461) with our gold holding ≥ 0.84 |
| R8-H80 | hybridizer | both terms - domain GRL and language GRL, separate lambdas | language is already solved, so this must beat R8-H79 to justify its cost | ≥ +0.01 over R8-H79 alone, or the language branch is dropped |
| R8-H81 | mechanist | GroupDRO instead of adversarial - minimise WORST-domain loss, not average | targets the actual failure (catastrophic collapse on finqa) without a discriminator or a lambda | worst-subset AUC ≥ 0.55 with mean ≥ 0.68 |
| R8-H73 | hybridizer | two heads on one trunk - score regression + token-span tagging | RE-STATED against new bars: must now beat 0.8531 / 0.8434 / 0.8407 AND lift the blind arena | blind mean ≥ 0.68 while all three in-domain bars hold |

**Why GroupDRO is registered alongside, and why it may be the better bet.** The adversarial framing assumes the problem is that features ENCODE domain. GroupDRO makes a different assumption - that the problem is the loss being AVERAGED over domains, so a domain the model handles badly is drowned out by nine it handles well. That is a literal description of R8-H77: mean 0.5956 hides a 0.3974. GroupDRO optimises the worst group directly, needs no discriminator, no gradient reversal and no lambda schedule, and cannot collapse the representation. If it matches R8-H79 it should be preferred on simplicity alone.

**The failure mode to watch on R8-H79/H80, stated before the run.** An adversarial objective strong enough to erase domain also erases meaning: the expected failure is a trunk that is beautifully domain-invariant and grounding-blind, with the domain discriminator at chance AND task AUC collapsed. Lambda is therefore the load-bearing hyperparameter and a sweep is part of the experiment, not a follow-up. Diagnostic: the discriminator should decay toward 1/12 = 0.083 while task AUC holds; a discriminator BELOW chance means the reversal has pushed the trunk into anti-predicting, which is its own pathology.

**Ordering.** R8-H81 (GroupDRO) first - no discriminator, no lambda, and it targets the measured failure most directly. R8-H79 second. R8-H80 only if R8-H79 wins, since it is the only way to attribute the language branch. R8-H73's two heads run last and must now clear the R8-H62 bars, not the pre-H62 ones.

**R8-H82 GroupDRO and DANN composed (pre-registered)**

They compose cleanly because they modify different parts of the objective rather than competing for the same one:

L = Σ_g q_g · L_task,g − λ · L_domain

GroupDRO changes how the task loss is AGGREGATED (worst-group weighting via the exponentiated-gradient weights q_g); DANN adds a SEPARATE reversed term through a gradient reversal layer. Both read the same group labels - corpus of origin - so the combination costs no extra annotation.

- Hypothesis - because the two target different failure modes and R8-H77 exhibits BOTH, composing them will beat either alone: GroupDRO handles seen-but-hard groups (after R8-H78, `finqa` is a group we have trained on and are still bad at), while DANN targets transfer to groups never seen at all
- Prediction - the composition beats the better single method by 0.01-0.03 on the blind arena; below 0.01 it is not worth the hyperparameter surface
- Acceptance bar - must beat max(R8-H79, R8-H81) by ≥ 0.01 on the HaluEval blind arena while our gold holds ≥ 0.84
- **Runs LAST, and the ordering is not negotiable.** Three reasons, each of which has already cost this project a round: attribution is impossible if a composed run wins (the reason R8-H62 was deliberately single-head); the hyperparameter surface is lambda x eta_q x regularisation strength, and GroupDRO REQUIRES strong regularisation while lambda is already DANN's load-bearing knob, so each dimension can silently mask the others; and R8-H78 - plain ERM over thirteen domains - may make both unnecessary, in which case this is a sweep run for its own sake
- **The tension to watch.** GroupDRO up-weights the hardest domain; DANN erases domain identity. One says pay special attention to finqa, the other says forget you are looking at finqa. They can coexist, but at high lambda the DRO reweighting may have nothing left to grip - so the diagnostic is whether the group weights q_g still differentiate once the discriminator has decayed toward chance. Flat q_g plus a chance-level discriminator means the composition has degenerated into plain ERM with extra steps

**R8-H78 incarnation 2 - result. The diagnosis was right, and the result is NOT countable**

Adds RAGBench TRAIN across all ten domains (~30k pairs) to the R8-H62 mix. Checkpoint `models/R8-H78-mmbert-tabular`.

In-domain, against the round's three bars:

| corpus | R8-H62 | R8-H78 | delta | bar |
|---|---|---|---|---|
| private gold | 0.8531 | 0.8314 | -0.0217 | 0.7095 - held |
| RAGTruth EN | 0.8434 | 0.8373 | -0.0061 | 0.7039 - held |
| RAGTruth non-EN | 0.8407 | 0.8415 | +0.0008 | 0.6095 - held |

On the arena, re-run through the identical `--model` gate:

| subset | R8-H62 | R8-H78 | lettucedect-v2 | delta vs incumbent |
|---|---|---|---|---|
| **finqa** | **0.3974** | **0.7433** | 0.7170 | +0.0263 |
| **tatqa** | 0.5118 | **0.7788** | 0.6156 | +0.1632 |
| techqa | 0.6985 | 0.7922 | 0.6363 | +0.1559 |
| pubmedqa | 0.5665 | 0.6346 | 0.5162 | +0.1184 |
| hagrid | 0.5416 | 0.6993 | 0.5992 | +0.1001 |
| expertqa | 0.7148 | 0.7490 | 0.6503 | +0.0987 |
| emanual | 0.6495 | 0.6680 | 0.5999 | +0.0681 |
| hotpotqa | 0.6514 | 0.6459 | 0.5976 | +0.0483 |
| delucionqa | 0.5325 | 0.6790 | 0.7929 | **-0.1139** |
| **mean** | **0.5956** | **0.7041** | 0.6461 | +0.0581, 8/10 |

- **The diagnosis is confirmed: tabular grounding was a COVERAGE gap, not a capability gap.** `finqa` moved +0.3459 from below chance to beating the incumbent, and `tatqa` +0.2670, purely from direct supervision. R8-H77's reading - that the model did not know what to do with a table - was correct
- **DomainBed's prediction held.** Plain ERM over thirteen domains produced the entire gain. No discriminator, no lambda, no gradient reversal. R8-H79/H81/H82 must now beat **0.7041**, not the 0.5956 they were registered against
- **Verdict - NOT COUNTABLE as a win.** Training on RAGBench-train makes RAGBench-test no longer blind for us while the incumbent has still never seen it. The +0.0581 is an in-domain reading against a blind one, which is the exact home-field error R8-H77 was built to expose. Recorded for the mechanism it proves, not as evidence of generalisation
- **`delucionqa` is the honest residual** - still -0.1139 WITH direct supervision, and the only subset where more data did not help. It is the one remaining candidate for a genuine capability gap rather than a coverage gap
- In-domain cost of the diversity was small but real: gold -0.0217, and all three bars held

**R8-H84 incarnation 4 - VitaminC, the near-miss negatives (pre-registered)**

`tals/vitaminc`, 370,653 train rows, labels SUPPORTS 185,714 / REFUTES 131,958 / NOT ENOUGH INFO 52,981 - close to balanced, which nothing else in the mix is.

- Hypothesis - because every other corpus in the mix carries topically DISTANT negatives at base rates of 0.65-0.94, the model can partly learn the prior instead of the boundary; VitaminC's negatives are real Wikipedia revisions where a single factual edit flips the verdict, so a negative differs from its positive by one number, entity or qualifier and nothing else, and training on them will lift the blind arena and specifically the extreme-base-rate subsets where a handful of negatives decide the AUC
- Lever - the negative construction; the rest of the R8-H83 mix and the recipe are unchanged so the contribution is attributable
- Prediction - blind arena +0.02-0.05 over R8-H83, concentrated in delucionqa, tatqa, hotpotqa and covidqa (base rates 0.93-0.94); our gold within 0.02 of R8-H83
- Acceptance bar - blind mean ≥ R8-H83 + 0.02 with all three in-domain bars held
- Risk - Wikipedia register, English only. The dataset survey deprioritised VitaminC on DOMAIN, never on quality, and named it the fallback if domain-matched data underdelivered. It may teach a discrimination that does not transfer to conversational RAG
- **Runs AFTER R8-H83 rather than folded into it**, so VitaminC's contribution is separable from the HaluEval and PsiloQA diversity gain. Folding both into one run would make the round's central question - which data source buys generalisation - unanswerable
- Licence - CC-BY-SA-3.0, Wikipedia-derived, and it must be VERIFIED before any model trained on it ships

**R8-H83 incarnation 3 - diversity WITHOUT the test set. REFUTED, and the negative result is the finding**

Adds HaluEval (~24k balanced claims from identical evidence) and PsiloQA (20k, 14 languages) to the R8-H62 mix. RAGBench excluded entirely so the arena stays blind. Checkpoint `models/R8-H83-mmbert-diverse`.

In-domain, all three bars held: gold 0.8438, RAGTruth EN 0.8097, non-EN 0.8303 (per-language de 0.8298 fr 0.8199 es 0.8291 it 0.8317 pl 0.8206 hu 0.8228 cn 0.8581).

On the blind arena:

| subset | H62 | **H83** | H78 (not countable) | lettucedect-v2 | H83 delta | base |
|---|---|---|---|---|---|---|
| covidqa | 0.6916 | 0.7526 | - | 0.7355 | +0.0171 | 0.841 |
| expertqa | 0.7148 | 0.7212 | 0.7490 | 0.6503 | +0.0709 | **0.468** |
| techqa | 0.6985 | 0.6638 | 0.7922 | 0.6363 | +0.0275 | **0.564** |
| emanual | 0.6495 | 0.5866 | 0.6680 | 0.5999 | -0.0133 | 0.894 |
| hotpotqa | 0.6514 | 0.5790 | 0.6459 | 0.5976 | -0.0186 | 0.932 |
| tatqa | 0.5118 | 0.5863 | 0.7788 | 0.6156 | -0.0293 | 0.944 |
| pubmedqa | 0.5665 | **0.4783** | 0.6346 | 0.5162 | -0.0379 | 0.692 |
| hagrid | 0.5416 | 0.5602 | 0.6993 | 0.5992 | -0.0390 | 0.848 |
| delucionqa | 0.5325 | 0.7292 | 0.6790 | 0.7929 | -0.0637 | 0.935 |
| finqa | 0.3974 | 0.5038 | 0.7433 | 0.7170 | **-0.2132** | 0.920 |
| **mean** | 0.5956 | **0.6161** | 0.7041 | **0.6461** | **-0.0300** | 3/10 won |

- **Verdict - REFUTED.** Blind mean 0.6161 against the incumbent's 0.6461. The bar was "beat 0.6461"; we improved on R8-H62 by +0.0205 and still lose by 0.0300
- **Generic diversity buys roughly a fifth of what in-domain data buys, from MORE rows.** 44k rows of HaluEval + PsiloQA moved the mean +0.0205; 30k rows of the actual test domain moved it +0.1085. `finqa` tells the story cleanly across three incarnations: **0.3974 → 0.5038 → 0.7433**. Public diversity lifted it off anti-predictive; only real financial tables made it competent
- **DomainBed's prediction fails here.** Well-tuned ERM with diverse data was given a fair, well-resourced attempt and did not close the gap. That is what promotes R8-H81 (GroupDRO) and R8-H79 (DANN) from optional to motivated - **and their bar is now 0.6161, not the 0.5956 they were registered against**
- **The failure is a NEGATIVE-CLASS problem, not a domain problem.** We win exactly the balanced subsets - expertqa (base 0.468) +0.0709, techqa (0.564) +0.0275, covidqa (0.841) +0.0171 - and lose every extreme-base-rate one. Where a handful of negatives decide the AUC, we fail
- **`pubmedqa` REGRESSED below chance**, 0.5665 → 0.4783. Adding Wikipedia-register supervision actively harmed biomedical grounding, which is a direct warning against treating "more public data" as monotonically good
- **This diagnosis is what R8-H84 (VitaminC) was registered for** - it is the only corpus in the survey with genuinely near-miss negatives, real Wikipedia revisions where a single edit flips the verdict, and our losses concentrate precisely where the negative class is thin

**R8-H84 incarnation 4 - result. Near-miss negatives buy real transfer and nearly tie the incumbent, but do not beat it**

`tals/vitaminc`, 370,653 train rows collapsed to binary, added to the R8-H83 mix; RAGBench still excluded so the arena stays blind. Checkpoint `models/R8-H84-mmbert-vitaminc`. Scored through the identical `--model` gate.

In-domain, all three bars held: gold 0.8411, RAGTruth EN 0.8233, non-EN 0.8325 (per-language de 0.8297 fr 0.8161 es 0.8414 it 0.8336 pl 0.8208 hu 0.8190 cn 0.8672).

On the blind arena:

| subset | H83 | **H84** | lettucedect-v2 | H84 delta | base |
|---|---|---|---|---|---|
| covidqa | 0.7526 | 0.7417 | 0.7355 | +0.0062 | 0.841 |
| expertqa | 0.7212 | 0.7376 | 0.6503 | **+0.0873** | **0.468** |
| techqa | 0.6638 | 0.6706 | 0.6363 | +0.0343 | **0.564** |
| pubmedqa | 0.4783 | 0.5466 | 0.5162 | +0.0304 | 0.692 |
| hagrid | 0.5602 | 0.6428 | 0.5992 | +0.0436 | 0.848 |
| emanual | 0.5866 | 0.6029 | 0.5999 | +0.0030 | 0.894 |
| hotpotqa | 0.5790 | 0.6099 | 0.5976 | +0.0123 | 0.932 |
| tatqa | 0.5863 | 0.5987 | 0.6156 | -0.0169 | 0.944 |
| delucionqa | 0.7292 | 0.7190 | 0.7929 | -0.0739 | 0.935 |
| finqa | 0.5038 | 0.5797 | 0.7170 | **-0.1373** | 0.920 |
| **mean** | 0.6161 | **0.6450** | **0.6461** | **-0.0011, 7/10 won** | |

- **Verdict - Kept, not a win.** Against its own pre-registered bar it PASSES: blind mean 0.6450 is ≥ 0.6361 and the lift over H83 (+0.0289) lands inside the predicted +0.02-0.05. But the round's objective is beating the incumbent, and 0.6450 vs 0.6461 is -0.0011 - a near-tie, not a beat. The hypothesis is confirmed as a mechanism and refuted as a solution
- **The VitaminC mechanism works exactly where the diagnosis said it would.** The negative-class fix lifted `pubmedqa` back above chance (0.4783 → 0.5466) and pushed `hagrid` +0.0436, `expertqa` +0.0873 - the balanced and mid-base-rate subsets. It did NOT fix the extreme-base-rate tabular subsets: `finqa` is still -0.1373 and `tatqa` flipped negative. Near-miss negatives teach the boundary, but they are Wikipedia-prose near-misses, not financial-table near-misses
- **Consequence for the adversarial arm, stated before it runs.** Data levers have now bought +0.0205 (diversity) and +0.0289 (near-miss) for a combined 0.6450, still short of 0.6461, and the residual is concentrated in `finqa`/`tatqa`/`delucionqa` - three extreme-base-rate domains where coverage, not discrimination, is the gap. This is precisely the seen-but-hard condition R8-H81 (GroupDRO) targets and the never-seen condition R8-H79 (DANN) targets, so both stay motivated. **Their bar is now 0.6450**, the best blind mean achieved without touching the arena
- **`delucionqa` remains the honest residual across every incarnation** - the only subset no data lever has moved, and the strongest candidate for a genuine capability gap rather than coverage

**R8-H81 GroupDRO - result. Worst-group weighting overfits the hardest SEEN boundaries and collapses blind transfer**

Trainer `experiments/grounding-semantic/R8-H81_groupdro.py`, checkpoint `models/R8-H81-mmbert-groupdro`. The recipe, mix and data are byte-identical to R8-H84; the only change is the loss aggregation - exponentiated-gradient group weights (eta 0.01) over 12 corpus-of-origin groups, balanced stratified batches so every group updates q every step, weight decay 0.01 per GroupDRO's regularisation requirement.

In-domain, all three round bars beaten decisively: gold 0.8177 (+0.1082), RAGTruth EN 0.8345 (+0.1306), non-EN 0.8157 (+0.2062) - but gold sits BELOW the arm's own 0.84 guardrail (-0.023 vs H84's 0.8411).

On the blind arena, through the identical `--model` gate:

| subset | H84 | **H81** | lettucedect-v2 | H81 delta | base |
|---|---|---|---|---|---|
| techqa | 0.6706 | 0.7147 | 0.6363 | +0.0784 | 0.564 |
| expertqa | 0.7376 | 0.6919 | 0.6503 | +0.0416 | 0.468 |
| hagrid | 0.6428 | 0.6000 | 0.5992 | +0.0008 | 0.848 |
| covidqa | 0.7417 | 0.7342 | 0.7355 | -0.0013 | 0.841 |
| pubmedqa | 0.5466 | 0.5053 | 0.5162 | -0.0109 | 0.692 |
| hotpotqa | 0.6099 | 0.5809 | 0.5976 | -0.0167 | 0.932 |
| tatqa | 0.5987 | 0.5557 | 0.6156 | -0.0599 | 0.944 |
| delucionqa | 0.7190 | 0.7200 | 0.7929 | -0.0729 | 0.935 |
| finqa | 0.5797 | 0.5574 | 0.7170 | -0.1596 | 0.920 |
| **emanual** | 0.6029 | **0.4425** | 0.5999 | **-0.1574** | 0.894 |
| **mean** | **0.6450** | **0.6103** | **0.6461** | **-0.0358, 3/10 won** | |

- **Verdict - REFUTED, both gates.** Pre-registered: worst-subset AUC ≥ 0.55 with mean ≥ the re-based 0.6450. Measured: worst subset `emanual` 0.4425 (below chance), mean 0.6103 - below H84 (0.6450), below H83 (0.6161), barely above the pre-diversity H62 (0.5956). GroupDRO gave back nearly everything two data rounds had bought
- **The mechanism is visible in the group weights.** By the end q had collapsed onto two groups - `vitaminc` 0.523 + `ragtruth_en` 0.447 = 97% of the total, with `halueval` at 0.0002 and every other group under 0.01. Worst-group minimisation degenerated into training on the two hardest seen boundaries and starving the other ten, which is the OPPOSITE of the diversity that transfers. The assumption the method encodes - the worst seen group predicts the unseen one - is false here: the blind arena's hard subsets (finqa tables, emanual manuals) resemble none of the up-weighted material
- **In-domain vs blind dissociation, again.** All three in-domain bars beaten decisively while blind transfer collapsed - the fourth consecutive demonstration (H62, H78, H83/H84, now H81) that in-domain movement and blind generalisation are near-independent axes in this setup
- **Consequence for the arm.** The seen-but-hard theory of the residual is dead: R8-H82's GroupDRO branch loses its premise, and per the pre-registered ordering (H82 runs only if it can beat max(H79, H81) by ≥ 0.01) the composition now rides entirely on R8-H79's DANN result

### Amendment - 2026-07-31, the adversarial arm becomes an exploration, not a one-shot

Append-only; directed by the author after the R8-H81 refutation. A single point in a method's hyperparameter space refutes that POINT, not the method. DANN and GroupDRO are not discarded on their first read; each measured failure mode has a known remedy, and the remedies are cheap variants of trainers that already exist:

- **R8-H79 variants (DANN)** - the v1 pathology is an anti-predicting discriminator (accuracy 0.001 against a 0.083 chance floor, domain loss RISING 2.49 → 4.14): at lambda 0.1 the trunk learned to invert domain features rather than neutralise them. The lambda sweep was declared part of the experiment at registration; v2 runs lambda 0.02, and a stronger discriminator (so it re-adapts to inverted features faster than the trunk can invert) is the v3 lever if v2 still anti-predicts. True invariance parks the discriminator AT chance, never below
- **R8-H81 variants (GroupDRO)** - the v1 pathology is q-collapse (vitaminc 0.523 + ragtruth_en 0.447 = 97% of the weight, ten groups starved): eta 0.01 let the exponentiated-gradient update fixate. v2 runs eta 0.003 with uniform smoothing on q (q ← (1−α)q + α/n, α = 0.2); v3, if v2 still collapses, replaces the 12 corpus groups with 3 SUPERGROUPS (private / ragtruth / wiki-register) so worst-group cannot fixate on a single corpus
- **Variant discipline** - one lever per variant, recorded as dated `log:` lines under the parent hypothesis; the parent verdict stands until a variant clears the bar, which is recorded as a supersession with a back-reference, never a rewrite
- **R8-H82 (composition)** - decided against the best VARIANTS, not the first points; its rule is unchanged (must beat max(best H79, best H81) by ≥ 0.01)
- **R8-H73 (two-head trunk) joins the executable queue** - registered long before the arm, re-stated here: the R8-H64 orthogonality (Spearman −0.046..+0.083 between score-regression and token-span signals, fusion beating both) is the one supervision-shape lever no incarnation has tried; trainer `R8-H73_twohead.py`, score head on all pairs + token head on the span-carrying corpora (RAGTruth EN JSON spans, translated struct spans, PsiloQA offset pairs, HaluEval whole-answer; private and VitaminC token-masked), fused at inference as (p_score + 1 − max halluc-token)/2, max-over-chunks. The R8-H77 gate gains a twohead-aware scorer branch - data, chunking and metric untouched
- **Queue** - GPU1 serial: H79 v1 (running) → H73 → H79 v2 → H81 v2 → composition decision; arena scorings interleave on GPU2

**R8-H85 - context coverage (windowed inference). KILLED AT GATE**

Because the arena scorer truncates every evidence document to its first 1,500 characters (~375 tokens) while the incumbent reads 4,096 tokens per document, sliding-window max-over-windows scoring will lift finqa ≥ +0.05 and the blind mean past 0.6461 while non-tabular subsets hold within ±0.01.

- **Kill-gate, measured before any build** - per-subset hidden evidence mass (fraction of characters beyond the visible windows, MAX_CHUNKS applied) against the H84 per-subset delta vs the incumbent
- **Result - the precondition is absent.** Spearman(delta, mean hidden mass) = **-0.128, p = 0.73**. The correlation runs the WRONG way on the extremes: `techqa` has the longest documents in the arena (median 3,199 chars, 83% exceed the window, 55% of evidence mass hidden) and is our second-best subset (+0.0343); `tatqa` has the shortest (median 325 chars, 0.4% hidden) and we lose it (-0.0169). `finqa` hides 30% but `expertqa` hides 19% and wins +0.0873
- **Verdict - Killed-at-gate.** Truncation coverage does not explain the residual; no windowed-inference build. The kill sharpens the diagnosis: the finqa/tatqa deficit is CONTENT (numeric-tabular claims our mix never teaches), and the incumbent's edge there plausibly rides its token-classification objective - a hallucinated number is a localized token event, which token supervision detects and a CLS score must integrate. That is R8-H73's mechanism, and it feeds two data levers: R8-H86 (prose-parity: `dataset-lettucedetect-prose.zip`, the incumbent's own auxiliary corpus, absent from our mix - targets `delucionqa`/`emanual` manual-register prose) and R8-H87 (a tabular-numeric near-miss corpus - targets `finqa`/`tatqa`), both pre-registered before any build in their own entries

**R8-H79 DANN v1 - result. Domain-adversarial training redistributes the blind arena instead of lifting it**

Trainer `experiments/grounding-semantic/R8-H79_dann.py`, checkpoint `models/R8-H79-mmbert-dann`, lambda_max 0.1 (Ganin ramp). Data and recipe byte-identical to R8-H84; the only change is the N-way domain discriminator through a gradient-reversal layer. The pre-registered pathology check fired mid-run and held to the end: domain accuracy 0.000-0.001 against a 0.083 chance floor from step 1000 on, domain loss RISING 2.49 → 4.14 - the trunk learned to INVERT domain features, not neutralise them (true invariance parks the discriminator AT chance).

In-domain, 3/3 decisive: gold 0.8328, RAGTruth EN 0.8372, non-EN 0.8411 - the best non-EN any incarnation has posted (beats H84's 0.8325).

On the blind arena, through the identical `--model` gate (DANN-aware loader branch: trunk + task head, same data/chunking/metric):

| subset | H84 | **H79 v1** | lettucedect-v2 | vs lettuce | vs H84 |
|---|---|---|---|---|---|
| emanual | 0.6029 | **0.7149** | 0.5999 | **+0.1150** | +0.1120 |
| techqa | 0.6706 | 0.6878 | 0.6366 | +0.0512 | +0.0172 |
| expertqa | 0.7376 | 0.6769 | 0.6506 | +0.0263 | -0.0607 |
| tatqa | 0.5987 | **0.6413** | 0.6153 | **+0.0260** | +0.0426 |
| covidqa | 0.7417 | 0.7596 | 0.7354 | +0.0242 | +0.0179 |
| hotpotqa | 0.6099 | 0.6198 | 0.5983 | +0.0215 | +0.0099 |
| hagrid | 0.6428 | 0.5530 | 0.5996 | -0.0466 | -0.0898 |
| pubmedqa | 0.5466 | 0.4600 | 0.5163 | -0.0563 | -0.0866 |
| delucionqa | 0.7190 | 0.6628 | 0.7922 | -0.1294 | -0.0562 |
| finqa | 0.5797 | 0.5439 | 0.7170 | -0.1731 | -0.0358 |
| **mean** | 0.6450 | **0.6320** | **0.6461** | **-0.0141, 6/10 won** | -0.0130 |

- **Verdict - Refuted at the v1 point** (lambda 0.1). Pre-registered bar: blind mean ≥ 0.6450. Measured 0.6320. The parent hypothesis stays open per the exploration amendment; v2 (lambda 0.02) tests whether a gentler reversal parks the discriminator at chance instead of inverting
- **The redistribution is structured, not noise.** DANN moved exactly the subsets ERM cannot: `emanual` +0.1120 (the subset GroupDRO destroyed at 0.4425, now our SECOND-BEST beat), the first `tatqa` win of the project, best-ever `covidqa`. It paid with `pubmedqa` (0.4600, below chance) and `hagrid`. Domain-invariance helps register-shifted subsets (manuals, tables-as-text) and hurts subsets that RELY on register features our mix covers well
- **The complementarity is the finding.** Per-subset max(H84, H79) averages **0.6649** - +0.0188 over the incumbent. Same data, two objectives, strongly decorrelated errors: this motivates R8-H88 (score-level ensemble read) below, and R8-H89 (ensemble distillation back into one 307M student) if H88 clears

**R8-H86 - prose-parity data lever. KILLED AT GATE**

Because the incumbent's auxiliary corpus (`KRLabsOrg/lettucedetect-prose-hallucination`) carries manual-register prose our mix lacks, adding it will lift `delucionqa`/`emanual` while the rest holds.

- **Kill-gate, measured before any build** - provenance check on the train parquet's `dataset` column, because R8-H83 recorded a deliberate exclusion of this corpus (PsiloQA-derived, double-counts the upstream) that must be verified, not overruled from a sidecar description
- **Result - the precondition is absent.** The 78,882-row train split is 63,792 rows tagged `dataset: psiloqa` + 15,090 tagged `dataset: ragtruth` and nothing else - a repackaging of two upstreams ALREADY in our mix. The sidecar's "ACL papers, READMEs and Wikipedia markdown" describes document style within those rows, not independent material. There is no new register to add
- **Verdict - Killed-at-gate.** R8-H83's exclusion stands confirmed at the data level. The `delucionqa` residual keeps its status as the honest capability gap; no data lever in the surveyed pool targets it

**R8-H87 - tabular-numeric near-miss negatives (TabFact). Pre-registered**

Because the blind residual concentrates in the tabular subsets (`finqa` -0.1373, `tatqa` -0.0169 at H84) and the R8-H85 probe ruled out context truncation, the gap is REGISTER: no corpus in our mix teaches reading numbers out of structured tables. TabFact (CC-BY-4.0, ~118k statements over 16k Wikipedia tables, human ENTAILED/REFUTED labels, near-balanced) is the near-miss construction VitaminC proved works - counterfactual claims against the SAME table - in exactly the register we lose.

- **Terse claim** - because the finqa/tatqa deficit is tabular-register coverage and TabFact supplies near-miss negatives in that register, adding ~30k serialized TabFact pairs (caption + linearized table → evidence) to the H84 mix will lift finqa ≥ +0.04 and tatqa ≥ +0.02 while the seven non-tabular subsets hold within ±0.02 and the three in-domain bars stay decisive
- **Bar** - blind mean ≥ 0.6461 (beat the incumbent); guardrail: gold ≥ 0.84, no subset below chance
- **Fetch** - every HF mirror is a retired loading script; fetched from the upstream GitHub repo by `scripts/fetch_grounding_datasets.py` (r1+r2 statements joined to `#`-delimited CSV tables, official split ids). Wikipedia tables, not financial filings - register coverage, not domain coverage, stated before the run
- **Caveat recorded pre-run** - TabFact statements are short single-fact claims; if the arena's tabular deficit is table-AGGREGATION reasoning rather than table-value lookup, this lever underdelivers and that miss is itself diagnostic

**R8-H88 - complementarity read: score-level ensemble of ERM and DANN. Pre-registered**

Because H84 (ERM) and H79 v1 (DANN) trained on identical data with different objectives and their per-subset profiles decorrelate (per-subset max 0.6649 vs 0.6450/0.6320 members), averaging their per-pair probabilities will recover part of the oracle gap.

- **Terse claim** - because the two students' errors decorrelate across subsets, the unweighted mean of their per-pair sigmoid scores, max-over-chunks as always, will read blind mean ≥ 0.6461 with ≥ 7/10 subsets won, no subset below chance
- **Status** - DIAGNOSTIC, not a ship candidate: two 307M students exceed the round's 400M single-model ceiling. What it buys is the decision for R8-H89 (distill the ensemble's soft scores into ONE 307M student) - if even the cheap unweighted mean clears the incumbent, the residual is variance, not capability, and distillation is the mechanism that converts an ensemble into a legal single model
- **Rule fixed before the run** - unweighted mean, both members frozen as-is, identical gate; no weight tuning against the arena (that would un-blind it)

**R8-H88 - result. The ensemble is the first blind result above the incumbent**

Scorer `experiments/grounding-semantic/R8-H88_ensemble_arena.py`, members frozen (H84 ERM + H79 v1 DANN), unweighted mean of per-pair sigmoid probabilities, identical gate. (Fixing this run also exposed and repaired a silent arena defect: the twohead edit had dropped `@torch.inference_mode()` from `score_student`, breaking the plain branch - every RECORDED number predates that edit or used a branch with its own inference context, so all stand.)

| subset | ens | H84 | H79 v1 | lettucedect-v2 | delta |
|---|---|---|---|---|---|
| expertqa | 0.7261 | 0.7374 | 0.6769 | 0.6503 | +0.0758 |
| emanual | 0.6725 | 0.6023 | 0.7149 | 0.5999 | +0.0726 |
| techqa | 0.6946 | 0.6708 | 0.6878 | 0.6363 | +0.0583 |
| hotpotqa | 0.6289 | 0.6094 | 0.6198 | 0.5976 | +0.0313 |
| covidqa | 0.7519 | 0.7418 | 0.7596 | 0.7355 | +0.0164 |
| tatqa | 0.6232 | 0.5987 | 0.6413 | 0.6156 | +0.0076 |
| hagrid | 0.5998 | 0.6428 | 0.5530 | 0.5992 | +0.0006 |
| pubmedqa | 0.5074 | 0.5465 | 0.4600 | 0.5162 | -0.0088 |
| delucionqa | 0.6962 | 0.7195 | 0.6628 | 0.7929 | -0.0967 |
| finqa | 0.5691 | 0.5791 | 0.5439 | 0.7170 | -0.1479 |
| **mean** | **0.6470** | 0.6450 | 0.6320 | **0.6461** | **+0.0009, 7/10 won** |

- **Verdict - Confirmed (diagnostic).** Pre-registered bar met on all three clauses: mean 0.6470 ≥ 0.6461, 7/10 subsets won, none below chance. The margin is thin (+0.0009) but this is the FIRST blind mean above the incumbent in the project, from members that individually read 0.6450 and 0.6320
- **What it proves** - the blind residual between our students and the incumbent is substantially VARIANCE between training objectives, not a capability ceiling: two students on identical data with different objectives, averaged with no tuning, recover a third of the max-oracle gap (0.6470 of 0.6450 → 0.6649)
- **What it does not prove** - a ship: 614M total. The conversion mechanisms are R8-H89 (distill the ensemble's soft scores into one 307M student) and the R8-H90/H91 full-corpus pair, whose diverse-objective members would make a stronger ensemble AND a stronger distillation teacher
- **Residual after ensembling** - `finqa` -0.1479 and `delucionqa` -0.0967 survive (both members weak there): consistent with the R8-H85/H86 diagnosis - those two need REGISTER coverage (TabFact, now in the full mix) or a capability the mix cannot teach, not error-averaging

### Amendment - 2026-07-31, the author redirects the arm: full corpora + DANN as the centrepiece

Directed mid-exploration by the author: "lets use all training data but RagBench, and DANN is very important - imho it is the only way to uncover the actual geometry of contradictions and hallucination." Consequences, recorded before the runs:

- **R8-H90 (full-corpus DANN)** - every corpus but RAGBench at FULL size (~760k pairs: private ~74k, RAGTruth EN 15k, translations 106k, HaluEval 40k, PsiloQA 64k, VitaminC 371k, TabFact 93k), 13 domain groups, LAMBDA_MAX 0.02 (the registered v2 anti-prediction fix). Trainer `R8-H90_dann_full.py`. Bar: blind mean ≥ 0.6461; guardrail: three in-domain bars beaten, gold ≥ 0.80 (relaxed from 0.84 pre-run - private falls from 24% to ~10% of the mix; 0.84 still reported). Health check: domain-acc parks in [0.03, 0.15] at full ramp; anti-prediction at 0.02 kills the lambda lever and points at v3 (stronger discriminator)
- **R8-H91 (full-corpus ERM control)** - identical mix, no discriminator. Trainer `R8-H91_erm_full.py`, same bar. H90 − H91 isolates the adversarial objective at full scale; H91 − H84 isolates the data. Without this control the two-lever directive would be unattributable
- **R8-H87 status** - its TabFact lever is ABSORBED into the full-mix pair; the solo H87 run is not scheduled (attribution of finqa movement to TabFact specifically is downgraded from run-isolated to inferred, accepted consciously for GPU budget)
- **Queue** - GPU1 serial: H73 (running) → H90 → H91; arenas on GPU2 as checkpoints land; then the composition/distillation decisions (H82, H89) against the full-scale results

### Amendment - 2026-07-31, the author raises the bar: blind mean ≥ 0.74

The goal moves from "beat the incumbent" (0.6461, cleared by R8-H88's 0.6470) to **blind RAGBench mean ≥ 0.74**. For scale: the only 0.70+ this project has produced is R8-H78's 0.7041, which trained on RAGBench train and is recorded as invalid evidence; the incumbent reads 0.6461; the H84/H79 oracle reads 0.6649. 0.74 therefore requires mechanisms beyond error-averaging over the current members. The registered path: the full-corpus pair (H90/H91), decomposed scoring (H92 below), two-head fusion (H73, training), then ensemble-of-diverse-objectives distilled into one student (H89). Each step keeps RAGBench excluded from training - the bar is only meaningful blind.

**R8-H92 - decomposed scoring: min over sentences of max over chunks. Pre-registered**

Because RAGBench `adherence_score` is a response-level AND over the response's claims, and our scorer feeds the WHOLE response as one claim (a single hallucinated sentence hides among supported ones, diluted in one CLS score), while the incumbent's token-level 1 − max(halluc) is precisely a fine-grained min-aggregation - decomposing the response into sentences and scoring each independently will recover what dilution loses.

- **Terse claim** - because response-level dilution masks local hallucination, scoring per-sentence (max-over-chunks per sentence, MIN over sentences as the response score) will lift the H84+H79 ensemble's blind mean by ≥ +0.02 (to ≥ 0.667) with no subset below chance
- **Formula only** - existing frozen checkpoints (H84, H79 v1), identical data and metric; the only change is the scoring aggregation. Sentence split is deterministic (regex on terminal punctuation, min 25 chars, cap 12 sentences, whole-response fallback below 2 sentences)
- **Primary read** - min-aggregation on the ensemble; per-member and mean-aggregation numbers recorded as diagnostics, not bar-eligible
- **Why it can be large** - the two worst residuals (`finqa` -0.1479, `delucionqa` -0.0967 at H88) are precisely long multi-claim responses where one wrong number or one wrong instruction step is the hallucination; dilution is the failure mode the decomposition targets

**R8-H93 - DANN lambda geometry, searched under a legal objective. Pre-registered**

Because DANN's blind effect is governed by where lambda sits between task-dominance (no invariance, ERM behaviour) and feature inversion (the anti-prediction pathology measured at 0.1), a search over lambda can find the invariance point - but ONLY under an objective that measures generalisation without touching the arena. In-domain validation is disqualified by the four recorded in-domain/blind dissociations (H62, H78, H83/H84, H81); RAGBench is disqualified by blindness. The legal proxy is leave-one-corpus-out: hold ALL of HaluEval out of training and maximise AUC on it.

- **Terse claim** - because the invariance point lies between 0 and the measured inversion at 0.1, Optuna TPE over lambda ∈ [0.003, 0.15] (log) and discriminator hidden ∈ {128, 256, 512}, maximising held-out-HaluEval AUC, will find lambda* whose LOCO AUC beats the in-study ERM baseline (lambda 0) by ≥ +0.02, with the discriminator parked within [0.5x, 1.5x] of chance at lambda*
- **Scale discipline, stated pre-run** - trials are ~60k-pair subsamples of the full mix (HaluEval excluded), bf16 autocast, one epoch, GPU idx0 (RTX PRO 4000); the study compares trial-relative geometry, not absolute quality. lambda* transfers to the NEXT full-scale DANN run (H90 launches at the registered 0.02 regardless; the sweep refines its successor)
- **Baseline inside the study** - trial 0 is forced lambda ≈ 0 (ERM) so the LOCO lift is measured against the same subsample, same budget, same hardware
- **Artifacts** - `R8-H93_lambda_sweep.py`, study db `R8-H93_optuna.db` (gitignored), per-trial log `R8-H93_trials.json`

**R8-H92 - result. Decomposition is the largest single lift of the campaign; a single 307M student now beats the incumbent blind**

Scorer `experiments/grounding-semantic/R8-H92_decomposed_arena.py`, frozen checkpoints, pre-registered formula (per-sentence max-over-chunks, MIN over sentences), single shot.

| subset | ens min | H84 min | H79 min | ens mean | lettucedect-v2 | delta (ens min) |
|---|---|---|---|---|---|---|
| expertqa | 0.8066 | 0.7804 | 0.7995 | 0.7947 | 0.6503 | +0.1563 |
| pubmedqa | 0.6622 | 0.6441 | 0.6560 | 0.5338 | 0.5162 | +0.1460 |
| techqa | 0.7633 | 0.7657 | 0.7604 | 0.7406 | 0.6363 | +0.1270 |
| tatqa | 0.6913 | 0.6643 | 0.6998 | 0.6199 | 0.6156 | +0.0757 |
| hagrid | 0.6734 | 0.6857 | 0.6424 | 0.6036 | 0.5992 | +0.0742 |
| emanual | 0.6689 | 0.6768 | 0.6949 | 0.6126 | 0.5999 | +0.0690 |
| hotpotqa | 0.6226 | 0.6062 | 0.6057 | 0.5567 | 0.5976 | +0.0250 |
| covidqa | 0.7310 | 0.7223 | 0.7344 | 0.7351 | 0.7355 | -0.0045 |
| finqa | 0.6246 | 0.6510 | 0.6015 | 0.6041 | 0.7170 | -0.0924 |
| delucionqa | 0.6487 | 0.6231 | 0.6618 | 0.7485 | 0.7929 | -0.1442 |
| **mean** | **0.6893** | **0.6820** | **0.6856** | 0.6550 | **0.6461** | **+0.0432, 7/10 won** |

- **Verdict - CONFIRMED, decisively.** Bar was ensemble-min ≥ 0.667 with none below chance: measured 0.6893, none below chance. +0.0423 over the H88 whole-response ensemble from a formula change alone on the SAME frozen weights
- **The headline is in the pre-declared diagnostics: single-model beats.** H84 alone under min-scoring reads 0.6820 and H79 v1 alone 0.6856 - both above the incumbent's 0.6461 through the frozen gate with RAGBench untouched. The project's first LEGAL single-model blind beats, at 307M, size-matched. H79 v1's "refutation" was a scoring artifact: whole-response dilution masked the better sentence-level scorer
- **The mechanism read** - dilution was the failure mode exactly where predicted: pubmedqa (long cited abstracts answers) flipped from -0.0088 to +0.1460, expertqa +0.1563, techqa +0.1270. `finqa` improved +0.055 but stays -0.0924 (register gap, TabFact pending in H90/H91). `delucionqa` PREFERS mean-aggregation (0.7485 vs 0.6487) - procedural manual answers where sentence fragments score spuriously low; covidqa marginally likewise
- **Supersession, recorded now** - decomposed-min becomes the PRIMARY arena read for every subsequent incarnation (H90, H91, H73), with whole-response recorded alongside for lineage continuity. The ladder: incumbent 0.6461 → H88 0.6470 → **H92 0.6893**; distance to the 0.74 target: 0.0507
- **The delucionqa/covidqa preference for mean motivates R8-H94** - one soft aggregator between min and mean, its shape tuned LEGALLY (RAGTruth, never RAGBench), frozen, then one arena shot

**R8-H94 - soft aggregation between min and mean, tuned on RAGTruth. Pre-registered**

Because min-aggregation wins 8 subsets while mean wins delucionqa (+0.0998 over min) and covidqa, a single response-level aggregator with one shape parameter - quantile q or soft-min temperature, interpolating min → mean - can keep min's gains where dilution rules and mean's robustness where fragment noise rules, IF its shape is chosen without touching the arena.

- **Terse claim** - because the aggregator shape generalises across corpora of multi-sentence responses, the shape maximising mean AUC across RAGTruth EN + 7 translations (response-level, sentence-decomposed, H84 and H79 scored separately and averaged) will, frozen and applied once to the arena ensemble, read blind mean ≥ 0.6893 (no worse than pure min) with delucionqa improving ≥ +0.02
- **Tuning corpus** - RAGTruth test responses (multi-sentence LLM outputs, human span labels), the harness that already serves the in-domain bars; RAGBench untouched until the single frozen-shape shot
- **Search space, fixed pre-run** - quantile q ∈ {0 (=min), 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5}, soft-min temperature τ ∈ {2, 4, 8, 16, 32}, and blend alpha·min + (1−alpha)·mean, alpha ∈ {0.5, 0.65, 0.8, 0.9, 1.0}; argmax by mean of the two members' RAGTruth AUCs
- **Artifacts** - `R8-H94_soft_aggregation.py`: stage 1 caches per-sentence RAGTruth scores (GPU), stage 2 sweeps aggregators on the cache (CPU), stage 3 `--arena` applies the single frozen winner

### Amendment - 2026-07-31, the author registers the curriculum fanout: understand domains first, then forget them

Directed by the author: "we may perform GroupDRO training such that we lift all groups - and once it plateaus we could enter second stage training with DANN... this way we make sure domain understanding is great - and then perform a phase shift (must be long). Hypothesis is that the model will learn to generalise from well understood domains."

The fanout attacks both recorded single-stage failure modes at once: R8-H79 v1 showed DANN-from-scratch inverts domain features (anti-prediction), R8-H81 showed worst-group GroupDRO starves 10 of 12 groups (q-collapse). The curriculum claim is that each stage supplies what the other lacks - stage 1 builds a trunk that has mastered EVERY domain's boundary, stage 2 then removes domain identity from features that already encode the task.

**R8-H95 - lift-all-groups GroupDRO (stage 1 objective). Pre-registered**

Because H81's q-collapse came from unsmoothed exponentiated-gradient fixation (eta 0.01, two groups took 97% of the weight), GroupDRO with uniform smoothing q ← (1−α)q + α/n (α 0.2) and eta 0.003 on the FULL 762k mix will keep every group's weight above α/n by construction and lift ALL groups rather than two.

- **Terse claim** - because smoothing bounds every group's minimum weight, per-group held-out AUC will improve for ≥ 12 of 13 groups over training (vs H81's starvation), and the blind decomposed-min read will land ≥ the H91 ERM control (same data, same budget) rather than 0.035 below it as H81 did vs H84
- **Per-group validation, defined pre-run** - 2,000 held-out pairs per group sampled from each training corpus before training (train rows only, arena untouched); the all-groups-lifted criterion and the plateau rule both read from these
- **Standalone verdict** - H95 is judged as a single-stage model in its own right AND doubles as stage 1 of H96

**R8-H96 - the phase shift: GroupDRO → DANN curriculum. Pre-registered**

The author's core hypothesis: a model generalises FROM well-understood domains. Stage 1 (H95) trains until plateau; stage 2 loads the plateaued trunk, replaces the objective with DANN (fresh discriminator, GRL ramp restarted, lambda from the H93 sweep winner - 0.02 fallback if H93 is inconclusive), and trains long.

- **Terse claim** - because invariance imposed on features that already encode every domain's task boundary removes domain identity WITHOUT removing task signal (whereas DANN-from-scratch inverted it), the curriculum model's blind decomposed-min read will beat the best single-stage single-model read at its run date by ≥ +0.01, with the stage-2 discriminator parking AT chance (in [0.5x, 1.5x] of 1/13) rather than anti-predicting
- **Plateau rule, fixed pre-run** - evaluate per-group val AUC every 2,000 steps; plateau = no group improves > 0.003 for 3 consecutive evals; stage 1 also hard-caps at 1.5 epochs. Phase shift is a checkpoint save + separate stage-2 launch (detached-compute discipline), so the boundary is inspectable
- **Long, by direction** - stage 2 runs a full epoch over the 762k mix minimum; combined budget ~2.5 epochs, ~12h on GPU1
- **Controls already registered** - H90 (DANN-only, same data) and H91 (ERM-only, same data) at matched scale; H95 supplies the GroupDRO-only arm. Curriculum vs all three isolates the phase-shift contribution
- **Kill condition** - if stage 2 re-enters anti-prediction (domain-acc < 0.02 at half-ramp) on a mastered trunk, the inversion pathology is not a curriculum problem and the DANN lever escalates to v3 (stronger discriminator) as registered
- **Queue** - GPU1 serial after the running pair: H73 → H90 → H91 → H95 (stage 1) → H96 (stage 2); H93's lambda* feeds stage 2

**R8-H94 - result. The RAGTruth-tuned shape does not transfer; pure min stays primary**

Stage 1/2: cache + sweep on RAGTruth EN + 7 translations picked **softmin tau 2.0** (RAGTruth mean 0.7698 vs pure min's 0.7406 - on RAGTruth, softer is decisively better, full 18-candidate table in `logs/R8-H94-soft-aggregation.log`). Stage 3: one frozen shot on the arena.

| read | blind mean | vs H92 min |
|---|---|---|
| ensemble softmin tau 2.0 (frozen from RAGTruth) | 0.6613 | -0.0280 |
| ensemble pure min (H92) | **0.6893** | - |
| incumbent | 0.6461 | - |

- **Verdict - REFUTED on the primary bar** (0.6613 < 0.6893). The delucionqa clause of the prediction was CONFIRMED: +0.0814 (0.6487 → 0.7301), and covidqa +0.0039 - precisely the two subsets diagnosed as mean-preferring. The other eight all regressed (pubmedqa -0.1062, hotpotqa -0.0626, tatqa -0.0563)
- **The finding** - aggregation preference is corpus-dependent and RAGTruth is NOT a valid transfer proxy for it: RAGTruth's global preference (soft) matches delucionqa's failure mode (fragment noise in procedural answers) but not the dilution-dominated majority. A single global shape cannot serve both regimes
- **Consequence** - decomposed pure-MIN remains the primary arena read. A dispersion- or length-conditioned aggregator (choose shape per response from its own sentence-score statistics, no arena input) is the registrable follow-up idea, parked behind the training runs
- **Method note** - this is the loop working as designed: the shape was tuned on legal terrain, frozen, and spent exactly one arena shot to learn a transferable negative

**R8-H73 - result. The two-head trunk is the best in-domain model ever and a mid-pack blind model; fusion does not stack with decomposition**

Trainer `experiments/grounding-semantic/R8-H73_twohead.py`, checkpoint `models/R8-H73-mmbert-twohead`. Score head on all pairs + token head on the span-carrying corpora, fused at inference as (p_score + 1 − max halluc-token)/2.

In-domain, 3/3 DECISIVE and a new gold record: gold **0.8843** (score head 0.8530, token head alone **0.8896**, head Spearman 0.819), RAGTruth EN 0.8170, non-EN 0.8357.

Blind, both reads: whole-response 0.6366 (6/10); decomposed-min PRIMARY **0.6607** (7/10, beats the incumbent's 0.6461 by +0.0146) - below H84-min 0.6820 and H79-min 0.6856.

- **Verdict - Kept, in-domain champion; blind mid-pack.** The R8-H64 orthogonality promise partially held: heads correlate more after joint training (Spearman 0.819 on gold vs the untrained -0.046..+0.083) yet fusion still lifts in-domain. Blind, the fused per-pair score already embeds a token-level min, and sentence-level min on top over-sharpens - two AND-aggregations stack into an over-strict read
- **The token head is real on tables** - tatqa 0.7013 under the primary read is the best tatqa of the project (vs H79-min 0.6998), consistent with the R8-H85 diagnosis that a hallucinated number is a localized token event
- **Follow-up idea, parked unregistered** - a token-head-only blind read exists as a lever but would be read-shopping without a pre-registered mechanism claim; register before running if wanted

**R8-H97 - three-member decomposed-min ensemble (ERM + DANN + two-head). Pre-registered**

Because H73 trains a third objective (token supervision) on the same data and its per-subset profile decorrelates from both members (best-ever tatqa, strong pubmedqa, weak hotpotqa), adding it to the H92 ensemble may recover more of the oracle.

- **Terse claim** - because three objectives decorrelate more than two, the unweighted mean of the three models' per-sentence scores, min over sentences, will read blind mean ≥ 0.6920 (+0.0027 over H92) with ≥ 7/10 subsets and none below chance; if it reads < 0.6893 the two-member ensemble stands and the weaker member is confirmed as dilution
- **Members frozen** - H84, H79 v1, H73; identical gate, identical formula, one shot

**R8-H97 - result. The third member dilutes; the two-member ensemble stands**

One shot, members frozen: blind mean **0.6871** vs the pre-registered bar 0.6920 and the H92 two-member 0.6893. Per the hypothesis's own outcome branch: REFUTED - H73 (0.6607 solo) is confirmed as dilution, not decorrelation-gain. Footnote for the record: ens3 won 8/10 subsets (one more than H92) with pubmedqa +0.1089 and techqa +0.1181, so the third objective DOES redistribute; it just costs more mean than it buys. The H92 configuration remains the ladder holder at 0.6893.

**R8-H93 - result. DANN lifts never-seen-corpus transfer at every lambda; the peak lives in the anti-predictive band, at high variance**

Study `R8-H93_optuna.db`, 22 completed trials over two parallel workers (GPU idx0 + idx2). `R8-H93_trials.json` was rebuilt from the sqlite study post-run: each worker's local write held only its own trials (the anticipated last-writer-wins overwrite), and worker2 exited on a StopIteration in its summary print because its local list lacked trial 0 - after all its trials had committed to the db. No data loss; the db is canonical.

| lambda band | trials | LOCO AUC | final domain-acc (chance 0.083) |
|---|---|---|---|
| 0 (ERM baseline) | 1 | 0.6278 | 0.867 |
| 0.003 | 2 | 0.6249-0.6615 | 0.87-0.91 (no invariance) |
| 0.016-0.028 | 6 | 0.6386-0.6852 | 0.03-0.41 |
| 0.042-0.094 | 8 | 0.6160-0.6937 | 0.001-0.30 |
| 0.124-0.134 | 5 | 0.6226-**0.7418** | 0.000-0.001 (anti-prediction) |

- **Verdict - Confirmed on the lift clause, REFUTED on the geometry clause.** The bar (winner beats in-study ERM by ≥ +0.02) is cleared by 5.7x: lam_max 0.1241, hidden 256, LOCO 0.7418 vs ERM 0.6278, lift +0.1140. But the claim's second clause - discriminator parked within [0.5x, 1.5x] of chance at lambda* - fails: the winner is fully anti-predictive (dom-acc 0.001)
- **The structural finding** - ERM ranks 19th of 22; every lambda ≥ 0.016 beats it in expectation. The high band (0.124-0.134, 5 trials) holds both the peak AND the widest spread (0.6226-0.7418, all anti-predictive) - identical hyperparameters, ~0.12 of AUC apart. The stable regime is lam 0.06-0.07 with dom-acc 0.05-0.14: 0.6937/0.6903, no collapse
- **What this breaks** - the working equivalence "anti-prediction = failure" (from H79 v1's lambda-0.1 collapse) does not survive: under a legal transfer objective the anti-predictive regime can carry the BEST never-seen-corpus AUC. It is a high-variance regime, not a dead one. Scale caveat as registered: trials are 60k-pair subsamples, one epoch; geometry is trial-relative
- **Feeds H96 as registered** - `pick_lambda` reads the rebuilt json and selects lam 0.1241 / hidden 256 (winner beats ERM, so no fallback). The selection mechanism was fixed pre-run and stands; the variance observation is recorded, not acted on

### Amendment - 2026-08-01, the H96 kill condition is re-grounded by H93's evidence

The pre-registered H96 kill ("stage 2 re-enters anti-prediction, domain-acc < 0.02 at half-ramp → kill") would deterministically kill the sweep's own winning lambda: all five high-band trials sit at dom-acc ≤ 0.001, including the LOCO winner. That kill was written when anti-prediction was believed a pure failure mode; H93 refutes the equivalence. Amended BEFORE H96 launches, recorded here:

- **The kill moves task-side** - stage 2 is killed if any group's val AUC falls > 0.05 below its stage-1 plateau value for 2 consecutive evals (2,000-step cadence), or task loss diverges. Domain-acc is demoted from kill criterion to recorded diagnostic
- **The H96 prediction clauses stand as written** - the terse claim still predicts the stage-2 discriminator parks AT chance on a mastered trunk; if it anti-predicts instead, that clause is adjudicated refuted even if the blind read wins
- **Nothing else moves** - lambda* 0.1241 / hidden 256 per `pick_lambda`, bars, plateau rule, and the three controls (H90 DANN-only, H91 ERM-only, H95 GroupDRO-only) unchanged

**R8-H90 - result. Full corpora + DANN in one 307M student: 0.7213 blind, the new ladder holder**

Trainer `R8-H90_dann_full.py`, checkpoint `models/R8-H90-mmbert-dann-full`; 762,535 pairs / 13 groups, LAMBDA_MAX 0.02, one epoch (15,887 steps, ~6.3h GPU1). In-domain guardrail 3/3 DECISIVE: gold **0.8418** (clears even the unrelaxed 0.84 line), RAGTruth EN 0.8201, non-EN 0.8370.

| subset | whole | min (primary) | H92 ens | lettucedect-v2 | delta (min) |
|---|---|---|---|---|---|
| expertqa | 0.6885 | 0.8248 | 0.8066 | 0.6503 | +0.1745 |
| tatqa | 0.6202 | 0.7718 | 0.6913 | 0.6156 | +0.1562 |
| hotpotqa | 0.7238 | 0.7253 | 0.6226 | 0.5976 | +0.1277 |
| techqa | 0.6741 | 0.7529 | 0.7633 | 0.6363 | +0.1166 |
| emanual | 0.6586 | 0.7058 | 0.6689 | 0.5999 | +0.1059 |
| pubmedqa | 0.5011 | 0.6058 | 0.6622 | 0.5162 | +0.0896 |
| hagrid | 0.5852 | 0.6516 | 0.6734 | 0.5992 | +0.0524 |
| covidqa | 0.7882 | 0.7755 | 0.7310 | 0.7355 | +0.0400 |
| finqa | 0.5674 | 0.6730 | 0.6246 | 0.7170 | -0.0440 |
| delucionqa | 0.7306 | 0.7263 | 0.6487 | 0.7929 | -0.0666 |
| **mean** | 0.6538 | **0.7213** | 0.6893 | 0.6461 | **+0.0752, 8/10 won** |

- **Verdict - CONFIRMED, decisively; new ladder holder.** Bar was blind mean ≥ 0.6461: measured **0.7213** under the primary decomposed-min read, 8/10 subsets, none below chance. The ladder: incumbent 0.6461 → H88 0.6470 → H92 0.6893 → **H90 0.7213**. Distance to the author's 0.74 target: 0.0187
- **A single model beats the frozen two-member ensemble by +0.0320** - full-scale data + DANN buys more than error-averaging over the capped-mix members; the H89 distillation premise (ensemble > any member) must be re-checked against H90-class members
- **The registered residuals moved exactly where the levers pointed** - tatqa 0.7718 is the project record (+0.0805 over the ensemble; the TabFact tabular register), finqa 0.6730 narrows its loss from -0.0924 to -0.0440, hotpotqa +0.1027 over the ensemble (multi-hop, plausibly VitaminC+scale), delucionqa narrows -0.1442 → -0.0666. Only finqa and delucionqa still lose
- **Health note, recorded honestly** - the pre-registered health band (domain-acc parks in [0.03, 0.15] at full ramp) was violated on the PREDICTIVE side: dom-acc held ~0.48-0.52 against chance 0.077 through the entire ramp. Lambda 0.02 under-invariances at full scale; no anti-prediction, so no kill. Consistent with H93's independent finding that the transfer optimum sits at higher lambda - the upward lambda lever stays open (H96 stage 2 runs at 0.1241)
- **The decomposition gap replicates on new weights** - whole 0.6538 → min 0.7213 (+0.0675), same order as H92's mechanism predicted; pubmedqa remains the extreme case (whole 0.5011 → min 0.6058)
- **Attribution pending H91** (ERM control, running) - H90 − H91 isolates the adversarial objective at full scale; H91 − H84 isolates the data

**R8-H98 - two-member decomposed-min ensemble at the new holder: H90 + H91. Pre-registered, gated on H91's solo read**

Because H92 showed unweighted error-averaging lifts the mean when members are near-equal (0.6820/0.6856 → 0.6893, +0.0037 over the best member) while H97 located dilution at a member ~0.025 below the pair (0.6871 < 0.6893), the ensembling question at 0.72 has a recorded boundary: it pays only between near-equals. H91 (same 762k mix, ERM objective) is the only candidate that can be both near-equal AND objective-decorrelated from H90; H79/H84 sit 0.036+ below the holder, past the recorded dilution boundary, and are excluded by that evidence.

- **Terse claim** - because H90 and H91 share data but differ in objective (adversarial invariance vs plain ERM), their per-sentence errors partially decorrelate; IF H91's solo decomposed-min lands within 0.02 of H90's 0.7213, the unweighted mean of the two models' per-sentence scores, min over sentences, will read ≥ max(member solo) + 0.002 with ≥ 7/10 subsets; a read below the best member confirms that same-data objective diversity does not decorrelate enough to pay at this level
- **Kill-gate, fixed pre-run** - if H91's solo min-read lands more than 0.02 below H90's (< 0.7013), H98 is killed at gate by the H97 dilution boundary without spending an arena shot
- **Members frozen once trained** - H90, H91; identical gate and formula; one shot; artifacts `R8-H98_ensemble_full.py`, `R8-H98_result.json`

**R8-H91 - result. The ERM control lands at 0.6965; the attribution triangle resolves, and DANN owns the larger lever**

Trainer `R8-H91_erm_full.py`, checkpoint `models/R8-H91-mmbert-erm-full`; identical 762,535-pair mix, plain BCE, one epoch (23,823 steps, ~6h GPU1). In-domain 3/3 DECISIVE: gold **0.8576** (the best gold of the full-mix pair; per-language non-EN 0.8294-0.8775), RAGTruth EN 0.8178, non-EN 0.8430.

| subset | whole | min (primary) | H90 min | lettucedect-v2 | delta (min) |
|---|---|---|---|---|---|
| expertqa | 0.6523 | 0.7962 | 0.8248 | 0.6503 | +0.1459 |
| hotpotqa | 0.7335 | 0.7164 | 0.7253 | 0.5976 | +0.1188 |
| emanual | 0.6096 | 0.7119 | 0.7058 | 0.5999 | +0.1120 |
| tatqa | 0.6061 | 0.7267 | 0.7718 | 0.6156 | +0.1111 |
| techqa | 0.7058 | 0.7441 | 0.7529 | 0.6363 | +0.1078 |
| pubmedqa | 0.5309 | 0.6114 | 0.6058 | 0.5162 | +0.0952 |
| hagrid | 0.6333 | 0.6396 | 0.6516 | 0.5992 | +0.0404 |
| covidqa | 0.7624 | 0.7438 | 0.7755 | 0.7355 | +0.0083 |
| finqa | 0.4810 | 0.6439 | 0.6730 | 0.7170 | -0.0731 |
| delucionqa | 0.7469 | 0.6313 | 0.7263 | 0.7929 | -0.1616 |
| **mean** | 0.6462 | **0.6965** | 0.7213 | 0.6461 | **+0.0504, 8/10 won** |

- **Verdict - Confirmed as the attribution control.** Bar (blind mean ≥ 0.6461) cleared under the primary read at 0.6965, 8/10; guardrail 3/3 DECISIVE. H91 is itself the second-best single model ever measured - and it exists to be subtracted from
- **The attribution triangle, the reason this run was registered** - objective: H90 − H91 = **+0.0248** (adversarial invariance at full scale); data: H91 − H84 = **+0.0145** (full corpora incl. TabFact over the capped mix). Both levers real; DANN's is 1.7x the data's. The author's directive that DANN is the central lever is confirmed by run-isolated evidence
- **Where the adversarial objective acts** - H90 beats its ERM twin on 8/10 subsets, largest exactly on the residual losses: delucionqa +0.0950, tatqa +0.0451, covidqa +0.0317, finqa +0.0291; only emanual (-0.0061) and pubmedqa (-0.0056) marginally prefer ERM. Domain-invariance pays most where the eval register is furthest from the training mix
- **In-domain/blind dissociation, again** - H91 wins gold (0.8576 vs 0.8418) yet loses blind (0.6965 vs 0.7213); the fifth recorded instance. In-domain selection would have picked the wrong model
- **R8-H98 adjudication** - H91 solo 0.6965 < 0.7013: **killed at gate** as pre-registered; the arena shot is not spent; H90 stands alone as the holder. The ensemble path at the holder level is closed until a second model lands within 0.02 of it

**R8-H95 - result. All 13 groups lifted - the starvation fix works - but forced balance still costs blind; the trunk ships to stage 2**

Trainer `R8-H95_groupdro_lift.py`, checkpoint `models/R8-H95-mmbert-groupdro-lift` (trunk/ exported for H96); 762,535 pairs / 13 groups, batch 52 (13×4 stratified), eta_q 0.003, smoothing α 0.2, stopped at the 1.5-epoch cap (21,246 steps, ~9.8h GPU1) - the plateau rule never fired because some group kept improving every eval.

Stage-1 facts: **13/13 groups lifted** vs first eval (bar ≥ 12/13); group-val mean 0.8262 → 0.9551; TabFact, the designated rescue case, 0.5242 → 0.7815 (+0.2573, the biggest climber and still rising at the cap); **q pinned uniform at 1/13 the entire run** (q_max = q_min = 0.077-0.078 at every logged step).

| subset | whole | min (primary) | H91 min | lettucedect-v2 | delta (min) |
|---|---|---|---|---|---|
| expertqa | 0.6764 | 0.7685 | 0.7962 | 0.6503 | +0.1182 |
| pubmedqa | 0.4545 | 0.6191 | 0.6114 | 0.5162 | +0.1029 |
| emanual | 0.5596 | 0.7022 | 0.7119 | 0.5999 | +0.1023 |
| techqa | 0.6384 | 0.7362 | 0.7441 | 0.6363 | +0.0999 |
| hagrid | 0.5254 | 0.6743 | 0.6396 | 0.5992 | +0.0751 |
| tatqa | 0.5870 | 0.6825 | 0.7267 | 0.6156 | +0.0669 |
| covidqa | 0.7320 | 0.7275 | 0.7438 | 0.7355 | -0.0080 |
| hotpotqa | 0.5945 | 0.5879 | 0.7164 | 0.5976 | -0.0097 |
| finqa | 0.5336 | 0.7053 | 0.6439 | 0.7170 | -0.0117 |
| delucionqa | 0.6352 | 0.6662 | 0.6313 | 0.7929 | -0.1267 |
| **mean** | 0.5937 | **0.6870** | 0.6965 | 0.6461 | **+0.0409, 6/10 won** |

- **Verdict - Confirmed on the mechanism clause, REFUTED on the blind clause.** Clause 1 (≥ 12/13 groups lift, vs H81's two-group starvation): 13/13, decisively. Clause 2 (blind decomposed-min ≥ the H91 ERM control): 0.6870 vs 0.6965, short by 0.0095 - though the gap to the ERM twin narrowed 3.7x from H81's -0.035. The smoothing fix repairs the training pathology; it does not make group-balanced training blind-superior
- **Why q never moved** - with smoothing α 0.2 re-flattening every step and stratified batches feeding all groups, no group's loss stayed dominant long enough for the exponentiated-gradient to differentiate; the effective objective was balanced ERM with a floor. The blind cost vs H91 therefore traces to BATCH COMPOSITION: forced 1/13-per-group balance upsamples the small synthetic groups and downsamples the natural-frequency mix that H91 sampled - and natural frequency wins blind
- **The mastery is real and it transfers where it was aimed** - finqa 0.7053 is the best finqa of the campaign (TabFact group-val 0.78 transferring to the tabular-financial register; the residual loss is now -0.0117), delucionqa 0.6662 beats both full-mix twins. The bill lands on hotpotqa (0.5879, -0.1285 vs H91)
- **As curriculum stage 1, the deliverable stands** - the author's hypothesis wanted a trunk that has mastered EVERY domain's boundary before invariance is imposed; 13/13 lifted at group-val 0.955 is that trunk. H96 (launched on this trunk, lambda* 0.1241) adjudicates whether generalisation-from-mastery holds; H95's own blind read is the honest cost accounting of stage 1 alone

**R8-H96 - result. The curriculum is refuted: invariance on a mastered trunk loses to single-stage DANN, and stage 2 undoes stage 1's mastery blind**

Trainer `R8-H96_phase_shift.py`, checkpoint `models/R8-H96-mmbert-phase-shift`; stage-1 trunk loaded, fresh DANN heads, lambda* 0.1241 / hidden 256 via the registered `pick_lambda`, 736,535 pairs (val carve-out held), one epoch (15,345 steps, ~5.7h GPU1), natural-shuffle batches (stage 2 matches H90's sampling; only the trunk init and lambda differ from H90). Task health perfect throughout: per-group val mean rose 0.9551 → 0.9610, TabFact group-val 0.7733 → 0.8092, the amended task-side kill never approached. In-domain: gold 0.8489, RAGTruth EN 0.7818, non-EN 0.8070 - all above decisive, EN ~0.04 below the low-lambda twins (the invariance tax).

| subset | whole | min (primary) | H95 stage1 | H90 min | lettucedect-v2 | delta (min) |
|---|---|---|---|---|---|---|
| pubmedqa | 0.4477 | 0.6446 | 0.6191 | 0.6058 | 0.5162 | +0.1284 |
| tatqa | 0.6289 | 0.6904 | 0.6825 | 0.7718 | 0.6156 | +0.0748 |
| emanual | 0.5569 | 0.6695 | 0.7022 | 0.7058 | 0.5999 | +0.0696 |
| expertqa | 0.6443 | 0.7189 | 0.7685 | 0.8248 | 0.6503 | +0.0686 |
| techqa | 0.6471 | 0.7027 | 0.7362 | 0.7529 | 0.6363 | +0.0664 |
| hagrid | 0.5592 | 0.6562 | 0.6743 | 0.6516 | 0.5992 | +0.0570 |
| covidqa | 0.7654 | 0.7644 | 0.7275 | 0.7755 | 0.7355 | +0.0289 |
| hotpotqa | 0.6175 | 0.6228 | 0.5879 | 0.7253 | 0.5976 | +0.0252 |
| finqa | 0.5157 | 0.6417 | 0.7053 | 0.6730 | 0.7170 | -0.0753 |
| delucionqa | 0.7083 | 0.7088 | 0.6662 | 0.7263 | 0.7929 | -0.0841 |
| **mean** | 0.6091 | **0.6820** | 0.6870 | 0.7213 | 0.6461 | **+0.0359, 8/10 won** |

- **Verdict - REFUTED on both clauses.** Blind clause: 0.6820 vs the bar 0.7313 (best single-stage read + 0.01) - it also sits below its own stage 1 (0.6870) and the ERM twin (0.6965). Geometry clause: the stage-2 discriminator parked at ~0.50 predictive (half-ramp 0.359, final 0.501, chance 0.077), not at chance; notably the ORIGINAL anti-prediction kill would never have fired either - on a mastered trunk, lambda 0.1241 yields a stable predictive equilibrium, a third geometry class beside from-scratch anti-prediction (H93) and low-lambda predictivity (H90)
- **The sharpest finding is the internal dissociation** - stage 2 RAISED every in-domain group-val (TabFact 0.7733 → 0.8092, mean to 0.9610) while LOWERING blind performance on exactly the corresponding registers: finqa fell 0.7053 → 0.6417 (stage 1's best-of-campaign tabular transfer, undone), expertqa 0.7685 → 0.7189, techqa 0.7362 → 0.7027. Invariance imposed on the mastered trunk strips features that transferred, while polishing the in-domain fit. The author's mechanism ("remove domain identity from features that already encode the task") removed more than identity
- **What survives** - pubmedqa 0.6446 is the best single-model pubmedqa of the campaign (the one register where stage-2 invariance helped blind), and the run is still +0.0359 over the incumbent with 8/10: a decent model, not a holder
- **Attribution gap, stated honestly** - H96 differs from H90 in TWO variables (trunk init AND lambda 0.02 → 0.1241); this run cannot say which caused the loss. R8-H99 (below) isolates them
- **Consequence for R8-H82** - the GroupDRO+DANN composition question is adjudicated by the curriculum pair: refuted at this operating point in both orders tested (H81 joint-style worst-group, H95/H96 sequential); no further composition run is scheduled

**R8-H99 - full-corpus DANN from scratch at the H93 winner lambda. Pre-registered**

Because H93's LOCO geometry puts the transfer optimum at high lambda (winner 0.1241 → 0.7418 vs 0.6386 at 0.026, every lambda ≥ 0.016 beating ERM) and H90's discriminator never left the predictive regime at lambda 0.02 (dom-acc ~0.49 at full ramp - under-invariance), raising lambda to the sweep winner on the OTHERWISE UNCHANGED H90 recipe is the single-variable test of whether the LOCO optimum transfers to the blind arena at full scale - and it simultaneously attributes H96's failure: H99 vs H90 isolates lambda, H99 vs H96 isolates the trunk init.

- **Terse claim** - because the LOCO transfer optimum sits at lambda ~0.124 and H90 is under-invariant at 0.02, the identical H90 recipe (full 762k natural-frequency mix, from scratch, one epoch) with LAMBDA_MAX 0.1241 will read blind decomposed-min ≥ 0.7313 (H90 + 0.01) with ≥ 7/10 subsets; guardrail: 3 in-domain bars beaten, gold ≥ 0.80
- **Risk registered pre-run** - the high-lambda band is high-variance in H93 (five same-band trials span 0.6226-0.7418) and went anti-predictive from scratch at sweep scale; a healthy-training read < 0.7213 is adjudicated REFUTED and is itself evidence that the band's variance is real at full scale, with the stable band (lambda ~0.065, LOCO 0.690-0.694, dom-acc 0.05-0.14) as the registered follow-up point
- **Kill condition** - task-loss divergence / non-finite only; discriminator geometry (anti-prediction included) is a recorded diagnostic per the H93 evidence, not a kill
- **Artifacts** - `R8-H99_dann_highlambda.py` (H90 clone, LAMBDA_MAX 0.1241), checkpoint `models/R8-H99-mmbert-dann-highlambda`, `R8-H99_result.json`

**R8-H99 - result. Lambda 0.1241 loses -0.0300 to 0.02 at full scale; the LOCO proxy is invalidated; high lambda conquers the far registers and pays on the strongholds**

Trainer `R8-H99_dann_highlambda.py`, checkpoint `models/R8-H99-mmbert-dann-highlambda`; identical H90 recipe, one epoch, only LAMBDA_MAX moved. Training healthy end to end; in-domain 3/3 DECISIVE (gold 0.8435, EN 0.8167, non-EN 0.8424 - no in-domain tax, unlike H96's EN dip).

| subset | whole | min (primary) | H90 min | H96 min | lettucedect-v2 | delta (min) |
|---|---|---|---|---|---|---|
| expertqa | 0.6591 | 0.7676 | 0.8248 | 0.7189 | 0.6503 | +0.1173 |
| techqa | 0.6662 | 0.7354 | 0.7529 | 0.7027 | 0.6363 | +0.0991 |
| tatqa | 0.6574 | 0.6958 | 0.7718 | 0.6904 | 0.6156 | +0.0802 |
| emanual | 0.7682 | 0.6592 | 0.7058 | 0.6695 | 0.5999 | +0.0593 |
| pubmedqa | 0.4680 | 0.5581 | 0.6058 | 0.6446 | 0.5162 | +0.0419 |
| covidqa | 0.7791 | 0.7696 | 0.7755 | 0.7644 | 0.7355 | +0.0341 |
| hotpotqa | 0.5905 | 0.6241 | 0.7253 | 0.6228 | 0.5976 | +0.0265 |
| hagrid | 0.5648 | 0.6135 | 0.6516 | 0.6562 | 0.5992 | +0.0143 |
| finqa | 0.5491 | 0.7135 | 0.6730 | 0.6417 | 0.7170 | -0.0035 |
| delucionqa | 0.8367 | 0.7757 | 0.7263 | 0.7088 | 0.7929 | -0.0172 |
| **mean** | 0.6539 | **0.6913** | 0.7213 | 0.6820 | 0.6461 | **+0.0452, 8/10 won** |

- **Verdict - REFUTED (branch 3 of the pre-registered table).** 0.6913 vs the bar 0.7313; below H90's 0.7213, above H96's 0.6820. H90 keeps the ladder
- **The attribution the run was built for** - lambda effect at full scale: **-0.0300** (H99 vs H90, single variable); trunk effect at matched lambda: **-0.0093** (H96 vs H99). The curriculum's -0.0393 deficit is fully decomposed: three quarters lambda, one quarter trunk
- **LOCO-HaluEval is invalidated as a lambda-selection proxy** - the sweep said 0.1241 ≫ 0.02-band (0.7418 vs ~0.64-0.69); the blind arena says 0.02 > 0.1241 by 0.0300. Recorded beside H94's RAGTruth-aggregation invalidation: both legal proxies measured so far do NOT transfer their preferences to the arena. Lambda 0.02 stands as validated near-optimal at full scale (full-scale curve: lam 0 → 0.6965, 0.02 → 0.7213, 0.1241 → 0.6913)
- **Geometry** - no anti-prediction at full scale (dom-acc ~0.44-0.49 through the ramp): H93's collapse at this lambda was a small-scale artifact of the 60k/1-epoch sweep trials, not a lambda property
- **The redistribution finding** - strong invariance conquers the two far registers the campaign never held: finqa 0.7135 (campaign best, residual -0.0035 - effectively a tie) and delucionqa 0.7757 (campaign best; whole-response 0.8367 BEATS the incumbent's 0.7929 outright). It pays with the strongholds: hotpotqa -0.1012, tatqa -0.0760, expertqa -0.0572 vs H90. Invariance strength trades near-register sharpness for far-register transfer
- **Ensemble at the holder measured closed, no shot spent** - H90/H99 per-subset Pearson r +0.777 (deltas +0.865): correlated, not complementary; oracle 0.7303 (+0.0090 over H90); member gap 0.0300 past the H97/H98 dilution gate. The complementarity idea died at its own diagnostic

**R8-H100 - full-scale run-to-run variance probe: verbatim H90 replicate. Pre-registered**

Because the H90 recipe seeds only the data split (SEED 0, private trace split - held fixed) while model init and batch order are unseeded, a verbatim rerun samples the run-to-run noise of the exact holder configuration - the quantity every recorded single-run delta is implicitly measured against, and never yet measured at full scale (H93's high-lambda band spread 0.12 at sweep scale is the warning).

- **Terse claim** - because the full-scale regime trains 762k pairs for one epoch (far past the sweep-scale instability), the verbatim H90 rerun will read blind decomposed-min within ±0.010 of 0.7213, confirming the recorded deltas (H90−H91 +0.0248, H90−H99 +0.0300) as signal
- **Pre-registered consequence** - if |read − 0.7213| > 0.02, every single-run delta of that magnitude or less in rounds 8 is demoted to within-noise, the affected verdicts get a noise annotation, and the campaign moves to multi-seed means before any further single-run adjudication
- **Not an improvement hypothesis** - no ladder claim; the artifact paths change so the holder checkpoint is untouched
- **Artifacts** - `R8-H100_dann_replicate.py` (verbatim clone, new CKPT/OUT), checkpoint `models/R8-H100-mmbert-dann-replicate`, `R8-H100_result.json`

**R8-H100 - result. The demotion clause fires: full-scale run-to-run noise is ~±0.03, and the round-8 single-run deltas fall inside it**

Verbatim H90 recipe, identical data and budget, second draw of the unseeded init/order. Training signature replicated (dom-acc ~0.47-0.50, same loss regime). In-domain 3/3 DECISIVE: gold 0.8511, EN 0.8304, non-EN 0.8398 (in-domain noise vs H90 ~±0.01).

| read | H90 (draw 1) | H100 (draw 2) | gap |
|---|---|---|---|
| blind decomposed-min (primary) | 0.7213 | 0.6918 | **0.0295** |
| blind whole-response | 0.6538 | 0.6464 | 0.0074 |
| gold | 0.8418 | 0.8511 | 0.0093 |

- **Verdict - the claim (within ±0.010) is REFUTED and the pre-registered consequence fires**: |0.6918 − 0.7213| = 0.0295 > 0.02. Run-to-run noise of the identical configuration on the primary blind read is ~±0.03. Every round-8 single-run TRAINING delta of that size or less is demoted to within-noise (enumerated in the amendment below)
- **The recipe, honestly stated** - the full-762k DANN lambda-0.02 recipe reads mean 0.7066 over n=2 draws (0.7213, 0.6918). The "holder at 0.7213" was one draw, not the recipe. The multi-seed protocol (registered consequence) extends to n=3 before any further single-run adjudication
- **What survives the noise bar, stated explicitly** - (1) every full-mix model beats the incumbent blind: margins +0.0452 to +0.0752 across four independent draws (H90, H91, H99, H100), all > 0.03, plus the deterministic frozen-weights H92 ensemble at +0.0432 - the incumbent beat is robust; (2) all formula comparisons on identical weights (decomposed-min over whole-response, +0.03 to +0.07; H94's softmin refutation; H97's dilution) are deterministic reads, untouched by training noise; (3) the H96 curriculum refutation vs its bar (margin 0.049) stands; (4) in-domain DECISIVE margins (+0.13 to +0.23) stand
- **Sentence-min amplifies training noise** - whole-response gap 0.0074, decomposed-min gap 0.0295: the min-aggregation concentrates a response's score on its weakest sentence, so small per-sentence miscalibrations compound. The formula that buys +0.05 blind also multiplies variance ~4x. A noise-robust aggregation that keeps min's gains is now a live registrable question

### Amendment - 2026-08-02, the noise bar: which round-8 verdicts demote

R8-H100 measured run-to-run noise ~±0.03 on the primary blind read. Per its pre-registered consequence, the following single-run training-config deltas are DEMOTED to within-noise; recorded verdicts stay in place (append-only) with this amendment as their noise annotation:

- **H90 − H91 = +0.0248 (the adversarial-objective attribution)** - demoted; whether DANN at 0.02 beats ERM at full scale is OPEN pending multi-seed means
- **H99 − H90 = −0.0300 (lambda 0.1241 worse than 0.02)** - at the boundary; demoted to within-noise; the H99 REFUTED verdict vs its own bar (0.6913 vs 0.7313, margin 0.040) stands, but the lambda-attribution reading is open
- **H96 − H99 = −0.0093 (trunk-init attribution)** and the 3/4-lambda-1/4-trunk decomposition - demoted; the curriculum's aggregate refutation vs its bar stands (margin 0.049)
- **H91 − H84 = +0.0145 (the data lever)** - demoted; full-mix-vs-capped attribution open
- **H95 − H91 = −0.0095 and H95's blind clause refutation** - the delta is within noise; the clause verdict (missed "≥ H91") is re-annotated as within-noise; the 13/13-groups-lifted mechanism confirmation (in-domain, different quantity) stands
- **The H98 kill-gate decision** (0.6965 vs threshold 0.7013) - the gate fired on a within-noise difference; the kill stands procedurally (the gate was pre-registered) but the dilution inference is not evidence
- **The benchmark ladder is restated as recipe means** - full-762k DANN lam-0.02: 0.7066 (n=2); full-762k ERM: 0.6965 (n=1); full-762k DANN lam-0.1241: 0.6913 (n=1); frozen H92 ensemble: 0.6893 (deterministic); incumbent: 0.6461. The author's 0.74 target is henceforth read against the recipe MEAN, and the honest distance from the best recipe is ~0.033
- **Protocol from here** - no further single-run cross-config adjudication; new training hypotheses pre-register a multi-seed design (n ≥ 2 draws minimum, mean reported) or an effect-size bar > 0.03

### The failure-mode and architecture analysis - 2026-08-03 (subagent, report in `reports/R8_architecture_failure_analysis.md`)

Per-example error analysis of the holder read (2,264 records reproduced at 0.7213 exactly; 8 ranked worst-case files read; comparator scores triangulated). Findings, recorded as analysis (not hypotheses):

- **Mode 1, ~30% of addressable residual - evidence-window truncation**: the harness truncates every chunk to 1,500 chars; needle-verified that 6/7 worst grounded false-negatives have support present in the full document PAST char 1,500; 86% (finqa) / 74% (delucionqa) of bottom-quartile grounded responses carry a truncated chunk - vs 0-3% in pubmedqa/hotpotqa/tatqa. A harness constant, not the model
- **Mode 2, ~25% - unsupportable-by-construction sentences**: inference/hedge/absence sentences are adherent under RAGBench labels but unentailable per (sentence, chunk); the min executes the response (56% of pubmedqa bottom-quartile argmins; 76% of pubmedqa grounded responses score < 0.1)
- **Mode 3, ~15% - numeric-derivation blindness, both directions**: derived-arithmetic sentences score ~0.01 (75% finqa / 68% tatqa bottom-quartile argmins); mirror FPs copy true numbers with wrong unit/year at 0.65-0.70. The one genuine model-class gap
- **Mode 4, ~15% - min-aggregation fragility + splitter artifacts**: delucionqa mean-over-sentences on H90's own scores reads 0.8130 vs min's 0.7263 (the formula, not capability, loses that subset); fallback-to-whole fires on 54-84% of four subsets; min amplifies training noise 4x (H100)
- **Mode 5, ~10% - multi-chunk composition blindness**: hotpotqa grounded median 0.0118 (ranking survives, calibration collapses); conflation FPs merge entities across chunks at 0.9+. Label-noise ceiling estimated 0.82-0.88 and not binding
- **Architecture verdict: NOT architecture-limited - recipe-and-formula-limited.** Same 307M backbone spans 0.5956-0.7213 blind across recipes (comparator on the identical backbone: 0.6461); the H90 trunk config itself carries 8,192 positions - the 512/1,500 window is a recipe constant. Residual attribution: formula/harness ~45%, training-data register coverage ~35%, true model-class capability ~15%, labels ~5%
- **Transfer ranking**: (1) token-classification head on mmBERT at full scale (H73 recipe, token-head-only blind read never taken; +0.01-0.03 predicted, needs multi-seed); (2) Qwen3-0.6B decoder scorer (only candidate changing capability class, attacks numeracy; over the sub-400M budget, needs explicit reopening); (3) ModernBERT-large (English-only forks the multilingual deliverable; +0.00-0.02, below noise). DeBERTa-v3 family ranks below all three

**R8-H101 - windowed evidence on frozen weights: un-truncate the chunks. Pre-registered**

Because the analysis needle-verified that support for the worst grounded false-negatives exists in the full documents PAST the harness's 1,500-char truncation (Mode 1, the largest addressable mode, concentrated precisely in the two losing subsets), scoring each sentence against sliding WINDOWS over the FULL chunk text recovers evidence the current read never sees - with zero training and no tuned parameters.

- **Terse claim** - because the support exists in text the scorer currently never reads, the frozen H90 checkpoint scored with per-sentence max over 1,500-char windows (stride 750, fixed pre-run) over every full chunk, min over sentences unchanged, will read blind mean ≥ 0.7213 with finqa AND delucionqa each improving ≥ +0.03; deterministic on frozen weights, so the ±0.03 training-noise bar does not apply
- **Kill** - if either finqa or delucionqa moves < +0.01, the truncation attribution is refuted and the residual re-attributes toward register coverage / model class
- **Discipline** - window size = the existing 1,500-char harness constant (not tuned), stride fixed at 50% pre-run, frozen weights, one shot through the identical gate; the frozen gate script is NOT modified - a separate read tool implements the windowing; H85's subset-level coverage kill is a different claim (this is argmin-level with needle verification); H94's no-tuned-aggregation lesson respected
- **Artifacts** - `R8-H101_windowed_read.py`, `R8-H101_result.json`

**R8-H101 - result. KILLED on the finqa clause; the windowed read is nonetheless the highest deterministic blind mean of the campaign**

Sanity pre-run: delucionqa carries 218/552 chunks (39.5%) beyond 1,500 chars (max 4,423); a first-window-only read reproduces the baseline EXACTLY (0.7263 = 0.7263 - path fidelity proven). Implementation note for the record: no reconstruction was needed - `load_subsets()` already returns full documents; the truncation lives in `score_student`'s pair assembly (`k[:1500]`), so windowing passes pre-cut ≤1,500-char window lists through the UNCHANGED scorer. The gate script was not modified; the MAX_CHUNKS=8 cap (gate definition) retained.

| subset | windowed | baseline H90 | delta | lettucedect-v2 |
|---|---|---|---|---|
| covidqa | 0.7755 | 0.7755 | +0.0000 | 0.7355 |
| delucionqa | **0.8072** | 0.7263 | **+0.0809** | 0.7929 |
| emanual | 0.7718 | 0.7058 | +0.0660 | 0.5999 |
| expertqa | 0.8346 | 0.8248 | +0.0098 | 0.6503 |
| finqa | 0.6711 | 0.6730 | -0.0019 | 0.7170 |
| hagrid | 0.6307 | 0.6516 | -0.0209 | 0.5992 |
| hotpotqa | 0.7246 | 0.7253 | -0.0007 | 0.5976 |
| pubmedqa | 0.6058 | 0.6058 | +0.0000 | 0.5162 |
| tatqa | 0.7742 | 0.7718 | +0.0024 | 0.6156 |
| techqa | 0.7598 | 0.7529 | +0.0069 | 0.6363 |
| **mean** | **0.7355** | 0.7213 | **+0.0142** | 0.6461 |

- **Verdict - KILLED on the registered conjunction.** finqa moved -0.0019 (< +0.01): the truncation attribution for finqa is refuted and its residual re-attributes to numeric-derivation blindness (Mode 3) - full table text in view, the scorer still cannot verify derived arithmetic. delucionqa's clause passed decisively (+0.0809): its starved sentences were literal procedural text and were rescued
- **Diagnostics, recorded as in H92** - windowed mean **0.7355** is the highest deterministic blind read of the campaign: +0.0142 over the identical frozen weights' baseline, delucionqa flips to a WIN over the incumbent for the first time under a min-family read (0.8072 vs 0.7929), finqa is the sole remaining loss, and the lift lands exactly on the truncation-exposed subsets while zero-exposure subsets are exact no-ops (covidqa, pubmedqa +0.0000) - the mechanism is confirmed at the argmin level even though the compound bar is not met. hagrid regressed -0.0209 (4% exposure; extra windows only add spurious maxima to its negatives), the only subset harmed
- **Supersession candidate, conditional and recorded BEFORE the deciding read** - the windowed decomposed-min read becomes the PRIMARY read if and only if it also dominates on the second draw of the recipe: windowed(H100 replicate) ≥ its baseline 0.6918. The deciding read is deterministic on frozen weights and is running as this is recorded; a pass records the supersession, a fail keeps the truncated read primary and demotes the windowed lift to draw-specific

**Supersession CONFIRMED - the windowed decomposed-min read is the primary blind read from here on**

The deciding read (`R8-H101_replicate_result.json`, H100 replicate checkpoint, deterministic): windowed mean **0.7097** vs its own baseline 0.6918, **+0.0180 - condition passed**. The mechanism replicates across draws: delucionqa +0.0800, emanual +0.0769, expertqa +0.0378, techqa +0.0309; covidqa/pubmedqa exact no-ops (+0.0000) on both draws; finqa negative on both (-0.0019 / -0.0267 - numeric derivation, not truncation); hagrid mildly negative on both (-0.0209 / -0.0180). Formula effect on the recipe: +0.0142 and +0.0180 on the two draws - consistent sign and size, unlike the training noise it rides on.

- **The ladder under the primary read** - recipe mean **0.7226 (n=2 draws: 0.7355, 0.7097)**; incumbent 0.6461; distance to the author's 0.74-of-mean target: **0.0174**. Best draw 0.7355, 9/10 subsets
- **All subsequent checkpoints (draw 3 onward) get the windowed read as primary**, truncated read recorded alongside for lineage
- **finqa is the campaign's last losing subset under the primary read** and its mode is now isolated twice over: numeric-derivation blindness, requiring a capability change (token-head at full scale or a decoder scorer), not more evidence

**Draw 3 of the holder recipe - the multi-seed protocol's third draw (R8-H100-draw3)**

Verbatim clone of the H90 trainer (only artifact paths changed - `models/R8-H100-mmbert-dann-draw3`), read blind through the frozen gate with the windowed primary read plus the truncated read for lineage.

| read | draw 1 (H90) | draw 2 (H100) | draw 3 | recipe mean | sd |
|---|---|---|---|---|---|
| windowed (PRIMARY) | 0.7355 | 0.7097 | 0.7065 | **0.7172** | 0.0159 |
| truncated (lineage) | 0.7213 | 0.6918 | 0.6987 | 0.7039 | 0.0154 |
| windowing lift | +0.0143 | +0.0180 | +0.0078 | +0.0134 | - |

- **Recipe mean under the primary read: 0.7172 ± 0.0159 (n=3)** - distance to the author's 0.74-of-mean target 0.0228 (~1.4 sd); incumbent 0.6461, beaten by +0.0711 of mean; draw 3 won 8/10 subsets truncated
- **The windowing fingerprint replicates a third time** - delucionqa +0.0916, emanual +0.0672, expertqa +0.0309, techqa +0.0202; covidqa and pubmedqa exact no-ops (+0.0000) on all three draws; hagrid mildly negative on all three (-0.0209/-0.0180/-0.0138). The supersession stands
- **New evidence - the finqa windowing penalty is checkpoint-dependent and grew each draw**: -0.0019 / -0.0267 / -0.0824. Each independently trained checkpoint scores long finqa table-windows differently; draw 3's long-window scores are bad enough to drag its whole-mean lift to +0.0078. Consistent with Mode 3 (numeric-derivation blindness): more table text in view gives the scorer more numbers to mis-handle, with checkpoint-level variance in how badly
- **Adjudication under the multi-seed protocol** - draw 1 (0.7355) is the high draw of the distribution, not the recipe's expectation; the recipe as-is does NOT meet 0.74 in expectation (would need a real effect ≥ +0.023 on top, above the +0.03 effect-size bar only marginally). Reaching 0.74-of-mean requires a registered lever with predicted effect > noise: the transfer ranking's candidates (token-head at full scale; decoder scorer pending budget word) are the live options
- **Artifacts** - `R8-H101_draw3_result.json` (windowed), `R8_decomposed_reads.json` tag `R8-H100-draw3` (truncated), `logs/R8-H100_draw3_train.log`, `logs/R8-H100_draw3_reads.log`. Note: the windowed script's printed verdict line for draw 3 compares against H90's hardcoded per-subset baselines and is not the adjudication; the table above (each draw vs its OWN truncated read) is

**R8-H102 - full-mix two-head with DANN, judged on the token-head-only read. Pre-registered (transfer ranking candidate 1, sketch P2 of the failure analysis)**

Because span supervision teaches that discourse tokens are not hallucination events (Mode B, ~25% of residual) and a wrong number is a localized token event (Mode C, finqa's mode), the H73 two-head recipe - score head + token-span head on one mmBERT trunk - scaled from its capped mix to the holder recipe's full 762k mix with DANN lambda 0.02, will produce a token-head-only blind read that beats the score-head read of the same weights. H73's fused read (0.6607 blind) stacked two ANDs and over-sharpened; the token-head-ONLY read was parked unregistered and has never been taken.

- **Terse claim** - because span supervision addresses Modes B and C directly, one full-762k two-head DANN draw, read blind as 1 − max(halluc-token prob) per (sentence, window) through the PRIMARY windowed decomposed-min, will beat the score-head primary read of the SAME checkpoint by ≥ +0.01, while the score-head read itself stays within the recipe band (≥ 0.7172 − 2×0.0159 = 0.6854)
- **Design note - the paired read is the point**: both heads sit on one trunk from one training run, so the token-vs-score comparison is within-checkpoint and deterministic on frozen weights; the ±0.03 run-to-run noise cancels out of the paired difference. The mean-level claim (does two-head lift the recipe expectation toward 0.74) is NOT adjudicable from one draw and is explicitly deferred: stage 2 (n ≥ 2 draws) runs only if the paired bar fires
- **Kill** - token-head-only primary read < score-head primary read of the same checkpoint → head choice is confirmed as not the carrier of the comparator's robustness; the head-transfer question closes and the remaining levers are the decoder scorer (budget word pending) and register-coverage data work
- **Guardrails** - in-domain gold ≥ 0.80 (score head); training instability (NaN/divergence) = kill; RAGBench untouched; frozen gate; both reads (windowed primary + truncated lineage) recorded for both heads
- **Cost** - ~7-9h GPU1 (existing `R8-H73_twohead.py` architecture, data loading swapped to the holder trainer's full mix), zero integration work at serving time
- **Artifacts** - `R8-H102_twohead_full.py`, `models/R8-H102-mmbert-twohead-full`, `R8-H102_tokenread.py` (token-head windowed read tool), `R8-H102_result.json`, `logs/R8-H102_*.log`

**R8-H102 - result. KILLED on the paired bar; the head-transfer question closes, and the kill carries the round's sharpest diagnostic**

Training clean (15,887 steps, DANN equilibrium held, domain-acc ~0.49); in-domain guardrails passed - score head 3/3 DECISIVE (gold 0.8321 ≥ 0.80, EN 0.8344, non-EN 0.8473), token head gold **0.8929**, the best in-domain number of the campaign. Blind paired reads on the same frozen checkpoint:

| subset | token windowed | score windowed | paired delta | token trunc | score trunc |
|---|---|---|---|---|---|
| covidqa | 0.7715 | 0.7641 | +0.0074 | 0.7715 | 0.7641 |
| delucionqa | **0.8663** | 0.7796 | **+0.0867** | 0.7854 | 0.7054 |
| emanual | 0.6538 | 0.7494 | -0.0956 | 0.5672 | 0.6737 |
| expertqa | 0.7818 | 0.7945 | -0.0127 | 0.7539 | 0.7507 |
| finqa | **0.6913** | 0.6311 | **+0.0602** | **0.7152** | 0.6920 |
| hagrid | 0.6393 | 0.6216 | +0.0177 | 0.6500 | 0.6362 |
| hotpotqa | 0.6145 | 0.7468 | **-0.1323** | 0.6145 | 0.7453 |
| pubmedqa | 0.5703 | 0.6357 | -0.0654 | 0.5703 | 0.6357 |
| tatqa | 0.7258 | 0.6979 | +0.0279 | 0.7222 | 0.6992 |
| techqa | 0.7364 | 0.7517 | -0.0153 | 0.6879 | 0.7222 |
| **mean** | 0.7051 | **0.7172** | **-0.0121** | 0.6838 | 0.7025 |

- **Verdict - KILLED**: token-head-only primary read 0.7051 < score-head 0.7172 of the same weights (bar was ≥ +0.01; kill was any negative). Paired and deterministic, so noise cannot rescue it: head choice alone is NOT the carrier of the comparator's robustness. The head-transfer question is closed
- **The diagnostic the kill delivers**: the heads are subset-ANTI-correlated on one trunk. The token head wins exactly where the score head is weakest - finqa +0.0602 (and its truncated read 0.7152 is the campaign's best finqa, 0.002 from the incumbent), delucionqa +0.0867 (0.8663, the campaign's best delucionqa by far) - and loses exactly where the score head is strong: hotpotqa -0.1323 (multi-chunk composition), emanual -0.0956, pubmedqa -0.0654 (Mode B's blind prediction failed - span supervision did not rescue discourse sentences). Mode C's prediction CONFIRMED at subset level: a wrong number is a localized token event, span supervision attacks it - direct mechanical support for H103's numeracy thesis
- **Score head unharmed**: 0.7172 windowed - exactly on the recipe n=3 mean; the auxiliary token loss cost the score head nothing blind
- **Artifacts** - `R8-H102_result.json` (in-domain; fused non-EN NaN noted, diagnostic only), `R8-H102_reads.json`, `logs/R8-H102_train.log`, `logs/R8-H102_reads.log`

**R8-H104 - parameter-free head fusion on the frozen H102 weights. Pre-registered**

Because the two heads' subset profiles on one trunk are anti-correlated with large complementary margins (H102 table above: token rescues finqa/delucionqa by +0.06/+0.09 where score is weakest; score rescues hotpotqa/emanual/pubmedqa by +0.07-0.13 where token is weakest), fusing them at the pair level should beat either head alone. The fusion is H73's exact serving formula - no parameter is introduced or tuned (H94 lesson respected).

- **Terse claim** - because head errors are subset-anticorrelated on shared features, the parameter-free pair-level fusion p = (sigmoid(score) + 1 − max halluc-token prob) / 2, aggregated through the PRIMARY windowed decomposed-min on the frozen H102 checkpoint, will read blind mean ≥ 0.7272 (score head + 0.01); deterministic on frozen weights, the ±0.03 training-noise bar does not apply
- **Kill** - fused mean < 0.7172 (the score head alone) → pair-level fusion line closed; the H73 fused-read failure generalizes from the capped mix to full scale
- **Caution recorded at registration** - H64/H73 history: rank-average fusion of two MODELS beat both, but H73's fused read stacked two ANDs and lost blind; this test differs in that fusion happens per pair BEFORE the min (one AND, not two). The H99-era lesson (measure correlation before registering an ensemble) is satisfied by the H102 paired table itself - the anti-correlation is measured, not eyeballed
- **Cost** - one deterministic read, ~30 min GPU0; zero training
- **Artifacts** - `R8-H104_fused_read.py`, `R8-H104_result.json`, `logs/R8-H104_read.log`

**R8-H104 - result. KILLED; the fusion line closes at full scale**

Sanity gate passed first (score-only path through the fused pipeline reproduces the recorded delucionqa 0.7796 EXACTLY - aggregation fidelity proven). The fused read on the frozen H102 weights:

| read | fused mean | vs score head alone (0.7172) | bar (≥ 0.7272) |
|---|---|---|---|
| windowed (PRIMARY) | **0.7156** | -0.0016 | missed |
| truncated (lineage) | 0.6953 | -0.0072 vs 0.7025 | - |

- **Verdict - KILLED**: fused windowed 0.7156 < the score head alone (0.7172); the kill threshold fired. Pair-level fusion of anti-correlated heads does NOT harvest the complementarity: per-subset, fusion lands BETWEEN the heads everywhere (delucionqa 0.8547 - below token's 0.8663; finqa 0.6707 - halfway to token's 0.6913; hotpotqa 0.6392 - far below score's 0.7468), and under min-aggregation the halfway points cost more on the strong head's subsets than they earn on the weak ones. H73's fused-read failure generalizes from the capped mix to full scale, now with the mechanism visible
- **What would harvest it and why it is not registrable**: per-SUBSET head selection reads 0.7372 - but choosing the head per subset from arena labels is selection on the benchmark, exactly what the discipline forbids. Recorded as a diagnostic ceiling, not a result
- **Artifacts** - `R8-H104_fused_read.py`, `R8-H104_result.json`, `logs/R8-H104_read.log`

**R8-H103 - Qwen3-0.6B decoder scorer, the capability-class test. Pre-registered (transfer ranking candidate 2, sketch P3; budget reopened)**

2026-08-03: the author reopened the sub-400M parameter budget, making the decoder line live. This is the only candidate that changes what the model CAN do: decoder LM pretraining (code/tables/math) supplies the numeric-and-unit competence behind Mode C - finqa's isolated failure mode, now confirmed three ways (H101 kill on the finqa clause; checkpoint-dependent windowing penalty growing to -0.082; 75%/68% of finqa/tatqa bottom-quartile argmins being derived-arithmetic sentences).

- **Terse claim** - because Mode C is a pretraining-capability gap and not an evidence or formula gap, a Qwen3-Reranker-0.6B-initialized sequence-classification scorer trained one epoch on the identical 762k mix (BCE, no DANN in stage 1, MAX_LEN 1,024) and read through the frozen gate under the PRIMARY windowed read will read finqa ≥ 0.72 while the blind mean lands ≥ 0.70 and in-domain gold holds ≥ 0.80
- **Two-stage design under the multi-seed protocol** - stage 1 (this draw) adjudicates ONLY the finqa mechanism (subset-level effect vs the recipe's primary-read finqa mean 0.6036 across draws - a ≥ +0.12 subset effect, far above subset-level noise); the mean-level claim needs stage-2 n ≥ 2 draws, run only if stage 1 fires
- **Kill** - training instability, or finqa < 0.70, or gold < 0.80 → the decoder line closes at this size and the budget rule re-closes with it
- **Discipline** - RAGBench untouched; frozen gate unmodified (a decoder-scorer read tool implements the pair scoring, windowed + truncated both recorded); serving-shape cost (~4x mmBERT FLOPs per pair, ~0.7GB int8, torch-free CPU cascade degraded) recorded as an accepted consequence of the reopened budget, to be weighed only if the mechanism fires
- **Cost** - ~20-25h GPU1 per draw at batch ~16/1,024 tokens; queued behind the H102 in-domain eval on the same card
- **Artifacts** - `R8-H103_qwen_scorer.py`, `models/R8-H103-qwen06b-scorer`, `R8-H103_read.py`, `R8-H103_result.json`, `logs/R8-H103_*.log`

## Round 9 - the clean-mix protocol and the single-pass fanout

### Amendment - 2026-08-03, the author resets the training protocol: the private gold dataset becomes test-only

Author's order: "redo the pipeline to not train on private gold dataset; we will test on RagBench and on golden only, not train on those."

- **What leaves the mix** - `private_train()`: 76,865 teacher pairs (soft rerank labels) drawn from the TRAIN split of the 639 private gold traces, ~10% of the 762k mix. The clean mix is `public_train()` only: ~686k pairs, 12 domain groups (RAGTruth EN + 7 translations, HaluEval, PsiloQA, VitaminC, TabFact)
- **Test protocol** - two held-out sets, neither ever trained or tuned on: RAGBench (blind, frozen R8-H77 gate, unchanged) and the private gold dataset - now the FULL 2,752 rows, legal as a pure test set since no trace enters training (the old gate used only the 40% held-out trace split)
- **Consequences** - R8-H90 and every full-mix descendant (the H100 draws, H102) trained on private-gold pairs; they remain recorded but are disqualified as deliverables under the new protocol. The recipe baseline must be re-established on the clean mix (R9-H105 below). Old split-gate gold numbers (0.83-0.84 band) stay for lineage but are not comparable to the new full-gold read: gold moves from in-domain to out-of-domain
- **Forward rule** - every future training hypothesis, including R8-H103 if its budget word ever fires, uses the clean mix

**R9-H105 - clean-mix recipe baseline. Pre-registered**

Because the removed private pairs supply domain-specific supervision orthogonal to RAGBench's domains (RAGBench never resembled the private traces; the pairs' value was the in-domain gold gate, which no longer trains), retraining the exact H90 recipe on the clean mix - mmBERT-base 307M, BCE + DANN lambda 0.02, MAX_LEN 512, ~686k pairs, 12 domain groups - will land the blind primary mean within the recipe band (≥ 0.7172 − 2×0.0159 = 0.6854) while full-gold AUC drops below the in-domain-trained 0.84 line (predicted 0.72-0.80, gold now out-of-domain).

- **Bar** - blind primary mean ≥ 0.6461 (beat lettucedect-v2); this draw re-establishes the baseline, its numbers become the reference for every future lever; no lever claim is made
- **Guardrail** - full-gold AUC (2,752 rows) recorded as the first clean reference - no pass line, first measurement; the old split-gate read recorded alongside for lineage
- **Kill** - training instability only
- **Mechanics** - DANN discriminator drops to 12 domain groups (chance 0.083); mid-run health check domain-acc in the equilibrium band as before; multi-seed protocol applies - this draw seeds the clean-recipe ladder, n ≥ 2 before any mean-level lever adjudication against it
- **Cost** - ~5h GPU1
- **Artifacts** - `R9-H105_clean_mix.py`, `models/R9-H105-mmbert-dann-clean`, `R9-H105_result.json`, `logs/R9-H105_*.log`

### The single-pass architecture fanout - 15 hypotheses, 0 survive adversarial review (2026-08-03)

Author's order: fan out hypotheses for internalizing the windowed decomposed-min pipeline into the model architecture or head (target: ≤ 2 forward passes per response, no external orchestration, at ≥ 0.7172 blind). Process: 5 lenses (long-context single-pass, learned aggregation, structure injection, distillation, architecture surgery) generated 15 hypotheses; triage merged near-duplicates and constraint violators to 8; each survivor faced an adversarial skeptic primed with the full campaign record. **All 8 refuted** (6 high-confidence, 2 medium). Full record in the workflow transcript; the refuted list is binding - do not resubmit without addressing the named flaw.

| hypothesis | lens | decisive flaw |
|---|---|---|
| sent-softmin-single-pass | structure-injection | learned global soft-min tau resubmits the H94 kill (RAGTruth-fit aggregation shape, -0.028 blind, does not transfer) |
| gated-noisy-or-head | learned-aggregation | gate has no supervision channel - g=1 degenerate optimum satisfies every loss term; H102 already falsified span-supervision-rescues-discourse |
| sent-abstain-class | structure-injection | mechanism measured and inverted - H102 token read moved pubmedqa -0.0654 under the same only-affirmative-evidence-sinks semantics |
| marker-smoothmin-distill | distillation | headroom double-counted - delucionqa min-vs-mean gap was measured truncated; windowed primary already banks it |
| dual-head-response-fusion | distillation | global fusion must discount the token head to survive hotpotqa, deleting the finqa/delucionqa edge that funds the claim; +0.015-0.030 prediction is 7-15x the closest measured analogue |
| long-context-response-distill | distillation | Mode A fix unsupervised where it lives - response-reconstructible sources carry short evidence; 61% of mix (VitaminC/TabFact) cannot be responses and its boundary supervision would be forgotten |
| colbert-late-interaction-two-tower | architecture-surgery | attacks Mode A which the windowed read already eliminated; MaxSim surfaces wrong-unit/year number copies at max similarity (Mode C FP inflation, no finqa kill) |
| digit-aware-arithmetic-adapter | architecture-surgery | premise falsified against the actual tokenizer - mmBERT already tokenizes digit-by-digit ('398.0' → ▁,3,9,8,.,0); predicted effect below run noise |

- **Recurring kill patterns** (the fanout's real yield): (1) any LEARNED aggregation shape trains on RAGTruth-register rows and the H94 record says that preference does not transfer - conditional aggregators must stay parameter-free; (2) Mode-A gains priced against the truncated baseline are double-counted - the windowed primary already banked them; (3) 61% of the mix is single-claim pairs that teach aggregation nothing - the label-shape gap is structural; (4) H102's pubmedqa -0.0654 stands as the falsifier for every span-supervision-rescues-discourse story
- **Registrable salvages** - three cheap frozen-weights precursors survive the review as pre-conditions, not hypotheses: **P-A** response-level parameter-free dual-head fusion on frozen H102 (fuse AFTER aggregation - min over score-head sentence scores with 1 − max token-head prob at response level, unweighted logit mean; the one fusion shape H104 did not test; ~30 min, deterministic); **P-B** lexicon-excluded min on the existing H90 dump (discourse-marker sentences dropped from the min; zero GPU; only pubmedqa ≥ +0.03 earns the abstain-class head a training run); **P-C** extend the dump to windowed per-sentence scores and measure residual min-vs-oracle headroom post-windowing (proceed with any aggregation work only if ≥ +0.01 blind-mean headroom remains)
- **Caveat** - precursor reads run on protocol-disqualified checkpoints (trained with private pairs); they remain valid MECHANISM tests since private-gold contamination is orthogonal to RAGBench, but any deliverable they motivate trains on the clean mix and adjudicates against the R9-H105 baseline

**R9-H105 - result. Clean baseline established; the clean mix costs nothing blind and RAISES gold**

Training clean: 14,285 steps over 685,670 pairs / 12 domain groups, DANN equilibrium domain-acc ~0.55 (chance 0.083), no instability. Blind reads (each vs this checkpoint's OWN truncated read - the windowed script's printed baselines/verdict are H90's hardcoded dict, not the adjudication):

| subset | windowed (PRIMARY) | truncated | windowing lift | lettuce |
|---|---|---|---|---|
| covidqa | 0.8030 | 0.8030 | +0.0000 | 0.7355 |
| delucionqa | 0.7975 | 0.7505 | +0.0470 | 0.7929 |
| emanual | 0.6883 | 0.6241 | +0.0642 | 0.5999 |
| expertqa | 0.7857 | 0.7684 | +0.0173 | 0.6503 |
| finqa | 0.6489 | 0.6639 | -0.0150 | 0.7170 |
| hagrid | 0.6259 | 0.6424 | -0.0165 | 0.5992 |
| hotpotqa | 0.6809 | 0.6827 | -0.0018 | 0.5976 |
| pubmedqa | 0.6201 | 0.6201 | +0.0000 | 0.5162 |
| tatqa | 0.7034 | 0.7091 | -0.0057 | 0.6156 |
| techqa | 0.6934 | 0.6732 | +0.0202 | 0.6363 |
| **mean** | **0.7047** | 0.6937 | +0.0110 | 0.6461 |

- **Verdict - baseline ESTABLISHED**: windowed PRIMARY 0.7047, bar (≥ 0.6461) passed by +0.0586; truncated 0.6937 with 8/10 subset wins. Band prediction CONFIRMED: 0.7047 ≥ 0.6854, sitting 0.0125 below the contaminated recipe mean - inside 1 sd. The private pairs were NOT load-bearing for blind transfer
- **Gold prediction REFUTED in the favorable direction**: predicted a drop to 0.72-0.80; measured split-gate gold **0.8629** - ABOVE every contaminated model (H90 0.8418, H102 0.8321) - and **gold_full 0.8788** on all 2,752 rows, the first clean measurement. The clean model beats the contaminated ones on the very dataset they trained toward: the private soft teacher labels were plausibly noise, not signal. The protocol reset is costless on both test sets
- **Windowing fingerprint replicates a fourth time on a different mix**: covidqa and pubmedqa exact +0.0000, delucionqa/emanual large positive, hagrid negative, finqa negative (-0.0150, mild draw of the checkpoint-dependent penalty)
- **The clean ladder starts at n=1**: 0.7047 is the reference for future levers until n ≥ 2; the contaminated ladder (0.7172 ± 0.0159) stays recorded for lineage only. Distance to the 0.74 target from the clean point: 0.0353
- **Artifact wrinkle** - the windowed JSON write failed on a doubled path (`--out` was passed with the directory prefix; the script prepends its own dir); all numbers are in `logs/R9-H105_reads.log` and the deterministic read was relaunched with a bare filename to materialize `R9-H105_windowed_result.json` (`logs/R9-H105_windowed_rerun.log`)
- **Artifacts** - `R9-H105_clean_mix.py`, `models/R9-H105-mmbert-dann-clean`, `R9-H105_result.json` (in-domain incl. gold_full), `R8_decomposed_reads.json` tag `R9-H105`, `logs/R9-H105_train.log`, `logs/R9-H105_reads.log`

**Precursor P-B - result. FAILED; the oracle bound closes the sentence-exclusion class, not just the lexicon**

Run on the frozen H90 dump (2,264 responses, per-sentence truncated scores; valid for the mechanism - windowing is an exact no-op on pubmedqa). Provenance caveat recorded honestly: no discourse lexicon was ever committed - the report's marker counts were manual case reading - so the executor reconstructed a 36-term lexicon from the failure report's own category exemplars (11 verbatim, 25 category-standard) AND added an oracle bound that settles the question independently of any lexicon choice.

- **Lexicon read** - sentence match rate 13.6%, all-matched fallback 5.4%; pubmedqa excluded-min delta **-0.0043** against the +0.03 bar; mean -0.0042
- **The oracle bound is the verdict**: dropping each response's single lowest-scoring sentence - an upper bound on ANY sentence-exclusion rule, learned abstain head included - reads pubmedqa **+0.0065** (4.6x short of the bar) and the MEAN at **-0.0359** (hotpotqa -0.0982, tatqa -0.0969). No exclusion rule can beat the oracle, so the bar is unreachable by the whole class
- **Verdict - KILLED, class-level**: the sent-abstain-class salvage line closes permanently. Two mechanism facts recorded: pubmedqa's floor is CROWDED, not sunk by a discourse outlier (removing the argmin barely moves the ranking - consistent with 76% of grounded responses scoring < 0.1); and on most subsets the argmin sentence is load-bearing SIGNAL - deleting it costs -0.036 of mean. Mode B will not fall to sentence exclusion; it is a scoring-quality problem, not an aggregation-membership problem
- **Artifacts** - `R9_PB_lexicon_min.py`, `R9_PB_result.json`, `logs/R9_PB_lexicon_min.log`

**Precursor P-C - result. NOT FIRED; hard-min is already the optimal fixed aggregator on windowed scores - the aggregation line closes**

Run on a new windowed per-sentence dump of the frozen H90 checkpoint (2,264 responses; sanity gate: the hard-min read reproduced the recorded R8-H101 per-subset AUCs EXACTLY on all 10 subsets before anything else counted).

| aggregator | blind mean |
|---|---|
| **hard-min** | **0.7355** |
| softmin tau 0.5 | 0.6975 |
| softmin tau 1 | 0.6912 |
| softmin tau 2 | 0.6872 |
| softmin tau 4 | 0.6853 |
| mean | 0.6840 |
| drop-argmin | 0.6975 |

- **Verdict - threshold NOT FIRED** (needed some fixed aggregator ≥ hard-min + 0.01): headroom is +0.0000 - hard-min wins the mean outright, and per-subset it wins 7/10 (only delucionqa/expertqa/techqa prefer a softer read, by small margins). The subset-level-best ceiling is 0.7400 (+0.0045) and is non-registrable (selection on benchmark)
- **What this closes**: every aggregation-softening lever - learned or fixed, gated or global - on top of per-sentence windowed scores. Together with P-B (exclusion closed) the round's aggregation question is fully answered: the decomposed hard-min IS the right read; the residual lives in per-pair scoring quality (Modes B/C) and in data coverage, not in the formula
- **Artifacts** - `R9_PC_windowed_dump.py`, `R9_PC_windowed_dump.json` (390KB, per-sentence windowed scores), `R9_PC_headroom.py`, `R9_PC_result.json`, `logs/R9_PC_dump.log`, `logs/R9_PC_headroom.log`. Chain note: the executor's watcher died between the dump and the headroom read; the read was recovered manually - numbers unaffected (deterministic from the dump)

**Precursor P-A - result. BAR FIRED; post-aggregation fusion is the round's first surviving mechanism**

Run on the frozen H102 two-head checkpoint, GPU0. Sanity gate EXACT: both single-head windowed reads reproduce `R8-H102_reads.json` on all 10 subsets. Fusion is parameter-free: per response, logit-mean of S_sent (score head through the decomposed windowed min) and S_tok (token head through the same read), fused AFTER each head's aggregation.

| read | blind mean |
|---|---|
| score head (recorded) | 0.7172 |
| token head (recorded) | 0.7051 |
| **fused post-aggregation** | **0.7223** |

- **Verdict - BAR FIRED**: fused 0.7223 ≥ 0.7172 AND above both heads (bar was both conditions; kill was < 0.7172). Deterministic paired comparison on one checkpoint - run noise does not apply. The mechanism H104 could not reach is real: fusing after aggregation lets each head keep its own min structure - fused beats BOTH heads outright on covidqa/expertqa/tatqa/techqa and holds hotpotqa at 0.6829 where pair-level fusion collapsed it to 0.6392
- **Magnitude honesty**: +0.0051 over the score head - a real, banked, serving-cost-free mechanism, not the 0.74 gap-closer on its own. It composes with everything else (two forward passes per pair already produce both heads)
- **Caveat**: measured on the protocol-disqualified H102 checkpoint; deliverable requires the clean-mix two-head retrain - registered next as R9-H106
- **Artifacts** - `R9_PA_response_fusion.py`, `R9_PA_result.json`, `logs/R9_PA_response_fusion.log`

**R9-H106 - clean-mix two-head with post-aggregation fusion serving. Pre-registered**

Because post-aggregation fusion of the subset-anticorrelated heads is proven deterministically on frozen weights (P-A: 0.7223 > 0.7172 > 0.7051, same checkpoint) and the mechanism is architecture-borne rather than data-borne, a clean-mix two-head DANN draw - the H102 recipe minus `private_train()` (~686k pairs, 12 groups, token spans where they exist) - served with the parameter-free post-aggregation logit-mean will read fused ≥ its OWN score-head windowed read + 0.003 (paired, deterministic) while the score head lands within the clean band (≥ 0.7047 − 0.03) and gold_full stays ≥ 0.80.

- **Bar** - paired: fused − score-head ≥ +0.003 on the same checkpoint (P-A measured +0.0051; the bar allows for mechanism attenuation, not reversal). Mean-level claims deferred to the clean ladder at n ≥ 2
- **Kill** - fused ≤ score head paired → the fusion mechanism does not survive retraining on the clean mix; the line closes. Training instability = kill
- **Guardrails** - RAGBench untouched; frozen gate; gold_full ≥ 0.80; both single-head reads recorded alongside the fused read
- **Cost** - ~5h GPU1, queued behind the draw-2 chain
- **Artifacts** - `R9-H106_twohead_clean.py`, `models/R9-H106-twohead-clean`, `R9-H106_result.json`, `R9-H106_fusion_read` outputs, `logs/R9-H106_*.log`

**R9-H105 draw 2 - result. The clean ladder holds at n=2; the gold surprise was a favorable draw, not the expectation**

Same script mechanism as the H100 draws (SEED pins only the data split; init and batch order unseeded sample the run noise). Training clean, all lineage bars DECISIVE. Reads vs the draw's OWN truncated:

| subset | windowed (PRIMARY) | truncated | lift |
|---|---|---|---|
| covidqa | 0.7726 | 0.7726 | +0.0000 |
| delucionqa | 0.8358 | 0.7573 | +0.0785 |
| emanual | 0.7070 | 0.6108 | +0.0962 |
| expertqa | 0.7599 | 0.7303 | +0.0296 |
| finqa | 0.6176 | 0.6393 | -0.0217 |
| hagrid | 0.6420 | 0.6573 | -0.0153 |
| hotpotqa | 0.6526 | 0.6551 | -0.0025 |
| pubmedqa | 0.5925 | 0.5925 | +0.0000 |
| tatqa | 0.7606 | 0.7591 | +0.0015 |
| techqa | 0.6745 | 0.6872 | -0.0127 |
| **mean** | **0.7015** | 0.6862 | +0.0153 |

- **Clean ladder at n=2**: windowed {0.7047, 0.7015} → mean **0.7031**; truncated {0.6937, 0.6862}. Both draws inside the band (≥ 0.6854) - the band prediction is confirmed twice; the clean recipe's draw spread so far (0.0032) is far tighter than the contaminated ladder's, though n=2 cannot establish that. Distance to the 0.74 target from the clean mean: 0.0369
- **Gold honesty update**: draw 2 reads split-gold 0.8177 / gold_full **0.8240** vs draw 1's 0.8629 / 0.8788 - a 0.055 swing. Draw 1's "clean beats every contaminated model on gold" was a favorable draw, not the expectation; the defensible statement at n=2: clean gold_full sits at 0.8514 ± large (both draws ≥ 0.82, still refuting the predicted 0.72-0.80 drop - the protocol reset remains costless, just not miraculous)
- **Fingerprint, fifth replication**: covidqa and pubmedqa exact +0.0000 again; delucionqa/emanual large positive; hagrid/techqa negative; finqa penalty -0.0217 (checkpoint-dependent, both clean draws mild)
- **Artifacts** - `R9-H105_draw2.py`, `models/R9-H105-draw2`, `R9-H105_draw2_result.json`, `R8_decomposed_reads.json` tag `R9-H105-draw2`, `R9-H105_draw2_windowed_result.json`, `logs/R9-H105_draw2_*.log`

**R9-H106 - result. KILLED on the paired kill; the P-A complementarity was a checkpoint property, not an architecture property**

Training clean (14,285 steps, in-domain gate 3/3 DECISIVE on the score head; gold_full score 0.8080 / token 0.8275 / fused 0.8286; fused non-EN NaN again - H102's empty-claim-token artifact, diagnostic only). Blind windowed reads, all three from the same forward passes, envelope invariant held:

| read | blind mean |
|---|---|
| score head | 0.6997 |
| token head | 0.6622 |
| **fused post-aggregation** | **0.6995** |

- **Verdict - KILLED**: fused 0.6995 ≤ score 0.6997 (kill was fused ≤ score; bar was ≥ +0.003). Paired and deterministic on one checkpoint - noise cannot rescue it
- **The diagnostic**: this draw's token head is far weaker than H102's (0.6622 vs 0.7051) and its subset profile INVERTED - it loses finqa (0.5702 vs score's 0.6378) and delucionqa (0.7355 vs 0.8517), exactly where H102's token head won. The anti-correlation P-A harvested was a property of THAT checkpoint's draw, not of the two-head architecture; with no complementarity to harvest, fusion has nothing to add. The post-aggregation fusion line closes - a mechanism that requires a favorable head-draw is not a recipe
- **Score head within the clean band** (0.6997 vs clean ladder {0.7047, 0.7015}) - the auxiliary token loss again cost the score head little blind; truncated lineage read 0.6857 (7/10)
- **Round-9 standing after the kill**: best clean model remains R9-H105 draw 1 (0.7047 windowed); the clean ladder mean 0.7031 is the planning number; remaining levers toward 0.74: register-coverage data work (unregistered), the held decoder line
- **Artifacts** - `R9-H106_twohead_clean.py`, `models/R9-H106-twohead-clean`, `R9-H106_result.json`, `R9-H106_fusion_result.json`, `R8_decomposed_reads.json` tag `R9-H106`, `logs/R9-H106_*.log`

## Round 10 - data-variety levers (2026-08-05)

Author's order: improve the register/domain variety of the clean mix - the supervision is thin and register-narrow exactly where the blind residual lives. Fanout process: 5 personas (synthetic-registers, corpus-miner, register-transformer, numeric-forger, mix-geometer) generated 15 hypotheses; triage kept 7 (dropping the forced-balance family resubmissions and near-duplicates); adversarial skeptics refuted 3 and passed 4. Data-only round: round 9 closed every formula and head lever.

**The contamination wall (binding for the round)**: RAGBench's ten subsets are built from public corpora - CovidQA, DelucionQA, EManual, ExpertQA, FinQA, HAGRID, HotpotQA, PubMedQA, TAT-QA, TechQA. Those corpora and their derivatives are forbidden in training; their REGISTERS are legal. Explicitly rejected at review on this wall: ConvFinQA, TAT-HQA, MultiHiertt, FinanceBench (FinQA/TAT-QA substrate), MTRAG (TechQA-adjacent, excluded on suspicion).

**Refuted at review (binding)**: procedural-manual-forge (mechanism anchored to a Mode A artifact the windowed read already fixed); hedged-discourse-forge (P-B's oracle read is the direct ceiling proxy for the mechanism - near zero); hedged-verdict-register (PUBHEALTH/CLIMATE-FEVER/SciTail hypotheses are declarative claims, not the failing hedged-discourse class). **Note the round's sharpest negative finding: all three pubmedqa/Mode-B hypotheses died independently - the hedged-discourse residual has no live data lever; it joins Mode B's hand-off to scoring-quality work**

**Corrected 2-draw subset baselines for all round-10 bars** (clean ladder windowed, draws 1/2): finqa {0.6489, 0.6176} → **0.6333**; tatqa {0.7034, 0.7606} → **0.7320**; emanual {0.6883, 0.7070} → **0.6977**; techqa {0.6934, 0.6745} → **0.6840**; pubmedqa {0.6201, 0.5925} → 0.6063; mean 0.7031. Subset draw noise is large (tatqa swing 0.057) - every bar below reads against the 2-draw mean with both-draws clauses, as registered.

### Pre-registration at a glance

| id | hypothesis | lens | target | pairs | bar (2-draw) | cost |
|---|---|---|---|---|---|---|
| R10-H107 | procedural-doc-register | corpus-miner | emanual + techqa | ~112k | emanual ≥ 0.735 AND techqa ≥ 0.730 AND mean ≥ 0.7031 | 15 GPU-h |
| R10-H108 | quantitative-nearmiss-register | corpus-miner | finqa (+tatqa guardrail) | ~118k | finqa ≥ 0.700 AND tatqa ≥ 0.690 AND mean ≥ 0.7031 | 15 GPU-h |
| R10-H109 | synthetic-numeric-derivation-forge | synthetic | finqa FN face | ~100k | finqa ≥ baseline +0.030 AND mean ≥ 0.7031 | 24 GPU-h |
| R10-H110 | wiki-table-derivation-pairs | numeric-forger | finqa derivation | ~120k | finqa ≥ 0.6989 AND mean ≥ 0.6981 | 26 GPU-h |

**R10-H107 - procedural-doc-register. Pre-registered.** Because emanual (2-draw 0.6977) and techqa (0.6840) are the manual/procedural residual and NO corpus in the mix contains numbered procedures, conditional eligibility rules, imperative instructions or identifier strings (the H86 "no data lever" verdict rested on a provenance check of one PROSE corpus, never the technical sibling), adding ~112k pairs - `KRLabsOrg/lettucedetect-code-hallucination` ~74k (BEHIND a pre-registered zero-GPU provenance gate: if its source column shows repackaged PsiloQA/RAGTruth upstreams, killed at gate - the H86 outcome repeated) + `IBM/multidoc2dial` ~38k (61k human-grounded turns over 488 US government-service documents; positives = annotated-span-in-window, negatives = deterministic span-anchored corruptions: step numbers, thresholds, identifiers, condition negation) as two new DANN groups - will lift both subsets ≥ +0.04 while the mean holds. Kill: both target subsets < +0.02, or mean < 0.6931, or delucionqa < 0.75 → the procedural residual is Mode A/E mechanics, not coverage. Contamination: government webpages and GitHub docs share zero documents with EManual (Samsung manuals) / TechQA (IBM technotes) / DelucionQA (Jeep manual); register only. Labels: human spans + corruption-by-construction; no LLM judge

**R10-H108 - quantitative-nearmiss-register. Pre-registered.** Because finqa (2-draw 0.6333) is the only subset losing to the incumbent and its Mode C failure is two-faced (75% of bottom-quartile grounded argmins are derived-numeric sentences scoring ~0.01; hallucinated wrong-unit/period copies score 0.65-0.82) while the mix teaches only table LOOKUP, composing the two proven data mechanisms - near-miss negatives (VitaminC, KEPT) and register coverage (TabFact, FIRED) - via ~118k pairs: FEVEROUS table-cell claims ~45k, InfoTabS ~24k, SEM-TAB-FACTS + SciTab ~6k (all human-labeled, Wikipedia/science-paper tables), plus ~45k deterministic unit/period/scale corruption negatives built from TabFact/FEVEROUS/InfoTabS positives only (≥ 6 edit families, distribution logged), as four new DANN groups - will lift finqa to ≥ 0.700 with tatqa ≥ 0.690 and mean held. STRETCH recorded pre-run: finqa ≥ 0.7170 ends the sole-loss status. Kill: finqa < +0.02 or draws disagree in sign → **Mode C is confirmed a capability class; the finqa data lane closes permanently and hands to the held decoder line with a stronger prior**. Licence gate (InfoTabS unstated) before any GPU

**R10-H109 - synthetic-numeric-derivation-forge. Pre-registered, QUEUED behind H108's verdict.** Fully synthetic financial statements (~14k tables, programmatic generator computes every derived quantity; gpt-oss-120b renders filing-style narration; negatives mutate exactly one semantic slot; ~15% supported-edit controls prevent surface shortcuts; labels COMPUTED, drop-not-relabel) → ~100k pairs. Runs only if H108 half-fires or its diagnostic points at derivation-narration rather than quantity semantics - the two hypotheses share the finqa bar and must not be confounded. Its kill escalates formally to the decoder line with the budget question reopened. Contamination: no source corpus exists; SEC/EDGAR/FinTabNet-lineage hard-banned regardless

**R10-H110 - wiki-table-derivation-pairs. Pre-registered, QUEUED as the H109 alternate.** Own extraction of Wikipedia tables (never TabFact's released rows), programmatic derivation engine (sums, deltas, percent change, ratios, ranks), ~40 templates + LLM paraphrase with verbatim-number-survival check on BOTH classes, corruption-by-construction negatives → ~120k pairs. Sequenced last of the numeric lane: same finqa bar, weakest register match (wiki vs financial prose), cheapest falsification of derivation-transfer

**Sequencing** - lane A (H107, procedural) and lane B (H108, quantitative) are subset-disjoint and adjudicate independently; H109/H110 are conditional on H108's outcome. Zero-GPU gates (provenance, licences, corruption generation) run first for both lanes; training draws serialize on GPU1, 2 draws per hypothesis

**R10-H111 - dropout-dial autoencoder corruption (author's mechanism, 2026-08-05). Pre-registered, two-stage**

Because a rich denoising encoder-decoder under inference-time MC dropout produces a monotone corruption spectrum - subtle paraphrase at low p, fluent small hallucinations in a mid band, noise at high p - there exists a dial setting p\* where FLUENT SEMANTIC DRIFT peaks; reconstructions of in-register seed statements at p\*, adjudicated pair-by-pair by an NLI referee (bidirectional entailment → paraphrase; non-entailment + fluent → drift; disfluent → discard), yield genuine in-register hallucination negatives against the seed's own evidence - the register-faithful generator that templates cannot be, applicable to registers where every other data lever died (hedged discourse included).

- **Stage 0 (calibration precursor, ~2-3h GPU0, runs before anything else)** - denoising encoder-decoder capable of near-identity reconstruction (mBART-50-class, or a brief identity-finetuned mT5); sweep dropout p over ~6 values on ~3k mixed-register seed statements (procedural from the H107 parquet, quantitative from H108 positives, hedged-scientific from legal open-access abstracts); referee = mDeBERTa-v3-mnli-xnli + a perplexity fluency gate; measure the composition curve paraphrase/drift/noise vs p, verify by a 50-sample eyeball per band
- **Stage-0 bar** - some p\* reaches fluent-drift yield ≥ 25% with the NLI paraphrase/drift boundary confirmed by eyeball (mislabeled paraphrases < 1 in 10 among admitted drifts). **Kill** - no band reaches 15% fluent drift, or drift is inseparable from disfluency → the dial does not exist at this model scale; line closes at ~zero cost
- **Stage 1 (conditional)** - generate at p\* to ~60-80k adjudicated pairs (drift → label 0 vs the seed's evidence; certified paraphrases → label-1 augmentation), new DANN group(s), one training draw under the standard bars (target-subset + mean-no-regression), n=2 on fire
- **Discipline** - the NLI is the referee, never our own grounder (self-adjudication = training on our own decision boundary, forbidden); referee noise is bounded to one NLI decision per pair and the eyeball sample quantifies it; seeds only from already-gated legal corpora, so contamination is inherited-clean
- **Recorded secondary** - the paraphrase band is a free label-1 augmentation lever (paraphrase-robust positives in-register) even if the drift band fails its yield bar

**R10-H111 stage 0 - result. FIRED with a referee amendment; the dial exists exactly as the author predicted**

Two calibration passes on `facebook/mbart-large-50` (identity fidelity 0.997 at p=0), 2,860 mixed-register seeds, MC dropout at inference, greedy decode. Pass 1's fluency gate (gpt2 NLL) was blind to degenerate repetition (high-probability text) - the main-session eyeball caught it; pass 2 added a calibrated degeneracy gate (distinct-3gram ≥ 0.952, max token-run ≤ 1):

| p | paraphrase | drift (honest) | noise |
|---|---|---|---|
| 0.10 | 0.713 | 0.080 | 0.207 |
| 0.15 | 0.408 | 0.210 | 0.382 |
| 0.20 | 0.076 | **0.302** | 0.622 |

- **Verdict - stage 0 FIRES**: the corruption spectrum is the predicted monotone dial (paraphrase → fluent hallucination → noise), per-register consistent (p=0.2 drift: scientific 0.334, procedural 0.322, quantitative 0.247 - all clear the 15% floor); best_p 0.2 at honest yield 0.302 ≥ the 25% bar; paraphrase-mislabel < 1/10 confirmed on BOTH eyeball files (essentially zero)
- **Amendment binding for stage 1 (from the second eyeball)**: the token-level degeneracy gate still admits symbol-soup and truncation garbage (~40% of admitted drift; LaTeX-heavy scientific seeds worst) - stage-1 referee v3 adds a char-level degeneracy check (char-run / symbol-density) and LaTeX-stripping of scientific seeds; net effective precision of admitted drift ~0.6, so 60-80k clean negatives ≈ 300-400k reconstructions - feasible on GPU0
- **Prized specimens for the record** (the class nothing else generates): hedge-deletion overclaims ("nearly the same" → "the same"), fluent entity corruption ("Danedream" → "Daneam"), fluent fact-substitution ("three awards between 1958 and 1962" → "was in the Golden Globe")
- **Amendment - referee v4, contrastive adjudication (author's design, 2026-08-05)**: the final label is CONTRASTIVE, not absolute - because the seed is label-1 by protocol (a guaranteed-clean reference), an LLM judge (local gpt-oss-120b, batched) sees (baseline, corrupted) side by side and answers only the easy question "did factual content change, and how" - never the hard one "is this grounded". Cascade: deterministic gates → NLI pre-referee → contrastive judge on the admitted band only (~80-120k pairs, not the 400k raw). Judge-confirmed factual deltas → label 0 with a DELTA TYPE (entity-swap / number-change / hedge-deletion / omission / negation); judge-confirmed no-delta → upgraded paraphrase band (label 1). Judge noise is bounded by the contrastive anchoring; a 50-pair main-session eyeball adjudicates judge precision before the parquet is admitted to any mix. Runs as a post-filter on the stage-1 output - no generation is redone
- **Artifacts** - `R10-H111_stage0.py`/`_stage0b.py`, `R10-H111_stage0_result.json`/`_stage0b_result.json`, `R10-H111_stage0b_recons.parquet` (11,440 scored reconstructions), `R10-H111_eyeball.md`/`_eyeball2.md`, `logs/R10-H111_stage0*.log`

**R10-H111 referee v4 - judge validation result. Judge ADMITTED as post-filter (2026-08-05)**

Built and validated on 500 stage-1 checkpoint rows before the full pass. One deviation from registration: the judge is `Qwen/Qwen3-32B-FP8` (vLLM, temp 0, thinking off, ~5-6 judgments/s batched) - the registered gpt-oss-120b is not cached locally (60GB download vs a cached 32GB instruct model); swap accepted, contrastive anchoring does not require the larger model. Author-authorized escalation (2026-08-05): if the full-pass eyeball on the judged parquet shows precision < ~85% or a contaminated paraphrase band, pull gpt-oss-120b and re-judge (post-filter rerun, no regeneration).

- **Validation (n=500, 500/500 parsed)** - delta distribution: degenerate 245 (49%), omission 165, other-factual 35, none 19, entity-swap 18, number-change 12, hedge-deletion 3, negation 3; post-judge 192 clean label-0 kept, accidental-regrounding filter dropped 44, 19 upgraded label-1 paraphrases
- **Judge vs NLI pre-referee** - judge confirms factual delta on only 47.8% of NLI-drift rows and confirms no-delta on only 23.4% of NLI-paraphrase rows - materially stricter in BOTH directions, exactly its cascade role; the NLI paraphrase band is impure, so label-1 augmentation is taken from judged output only
- **Main-session 50-pair eyeball (adjudicated)** - delta-detection precision ~90% (45/50 genuine factual deltas; the misses are duplication disfluency typed as factual change, e.g. "the speed of the speed", "an oil and an oil and"); typing sound on clean cases (3:09:45 → 3:03:45 as number-change, causal-clause drops as omission). Judge ADMITTED as post-filter
- **Caveats recorded** - severity skews obvious (47/50; only 3 subtle), and a residual disfluent fraction survives inside admitted drift (word-level repetition passes the token-level gates) - valid label-0 negatives but the fluent-subtle fraction is smaller than the stage-0 prize specimens suggested; risk of a fluency shortcut noted for training adjudication
- **Yield projection** - effective clean-drift precision ~38% of the NLI-admitted band → a 90-120k admitted band projects to ~35-45k judged negatives, UNDER the 60-80k stage-1 target; extension generation rounds may be needed after the full pass
- **Final parquet admission remains pending** the full judged parquet (`R10-H111_stage1_judged.parquet`) and the stage-1 report eyeball
- **Artifacts** - `R10-H111_judge.py`, `R10-H111_judge_validation.parquet`, `R10-H111_judge_eyeball.md`, `logs/R10-H111_judge_validation.log`; full pass armed on the STAGE1 DONE marker (watcher detached, vLLM starts only at fire time, GPU1 idle until then)

**R10-H111 stage 1 - generation result (2026-08-05). Early-stopped at the drift target; parquet NOT admissible raw - judge pass running**

260,452 reconstructions consumed (112,226 seeds, repeated full-seed rounds at p=0.2), early-stopped at the 80k drift target in ~2h50m on GPU0. Output `R10-H111_stage1_pairs.parquet`: 96,320 rows = 83,714 NLI-admitted drift (procedural 39,832 / scientific 26,333 / quantitative 17,549, label 0) + 12,606 paraphrases (procedural 6,583 / scientific 3,163 / quantitative 2,860, label 1).

- **Main-session 50-pair drift eyeball** - ~15-20/50 are the genuine article: fluent in-register factual corruption ("co-signer is the spouse" → "the person"; Vietnamese host city → "Chinese"; dropped hotline number); the remainder is repetition/truncation junk invisible to the token-level referee-v3 gates - consistent with the judge validation's 49% degenerate finding, and exactly the class the contrastive judge strips
- **Paraphrase-band impurity confirmed on sight** - borderline NLI-paraphrases include outright hallucinations ("high-redshift supernovae" → "high-relationships of supernovae") and degenerate junk; the rule stands - label-1 only from judge-confirmed no-delta rows
- **Verdict** - generation stage did its job (raw ore at scale, register mix healthy); admission rides entirely on the judge pass (launched 06:45, ~5h on GPU1); projected post-judge yield ~32k negatives, under the 60-80k target → extension rounds are a live option after the judged parquet is read
- **Artifacts** - `R10-H111_stage1.py`, `R10-H111_stage1_pairs.parquet` (27.8MB), `R10-H111_stage1_report.md`, `R10-H111_stage1_progress.json`, `logs/R10-H111_stage1_gen.log`

**R10-H111 judge full pass + final admission (2026-08-05). Parquet ADMITTED at 26,142 pairs after a still-entailed filter; no judge escalation**

Full pass: 96,320 pairs judged in 5h22m (Qwen3-32B-FP8, GPU1), 28 parse failures (0.03%). Delta mix: degenerate 46,073 (47.8%, stripped), omission 32,557, entity-swap 6,600, other-factual 6,185, none 2,982, number-change 1,494, negation 275, hedge-deletion 126. Judge kept 38,751 label-0 (regrounding filter dropped 8,486) + 2,982 certified label-1 paraphrases. Judge/NLI agreement 50.1% on drift, 18.6% on paraphrase - full-pass replication of the validation profile.

- **Main-session 50-pair eyeball on the kept set** - strict precision 84% (42/50 genuine negatives); the 5-8 misses are one class: the corrupted claim is STILL TRUE (pure truncation "the annual assessment is \$250", pure stutter "a rock and a rock band from Texas") - a LABEL-MAPPING hole (omission → label 0 is wrong when the residual claim stays entailed), not a judge-capacity failure; escalation to gpt-oss-120b NOT triggered - a larger judge answers the same question the same way
- **Still-entailed filter (deterministic, on existing columns)** - drop kept rows with nli_fwd(seed → claim) ≥ 0.8: catches 6/7 eyeballed mislabels, residual precision ~0.97; the sacrificed rows are truncation junk and typo-swaps NLI reads as near-identical, while fluent semantic corruptions (false dates, flipped comparisons, negations) all sit at low forward entailment and survive - the filter trades junk for purity
- **Final artifact `R10-H111_pairs_final.parquet`** - 26,142 pairs: 23,160 label-0 (procedural 11,518 / quantitative 5,331 / scientific 5,327, plus 984 reclaimed from the NLI-paraphrase band - the contrastive judge exposing hallucinations NLI called paraphrase) + 2,982 judge-certified label-1 paraphrases; negative delta mix omission 13,968 / other-factual 4,112 / entity-swap 3,907 / number-change 893 / negation 264 / hedge-deletion 16
- **Yield vs target** - 26,142 is under the registered 60-80k; end-to-end efficiency 260,452 recons → 23,160 negatives (8.9%). Extension rounds NOT launched pre-training: repeated rounds face dedup pressure on the same 112k seeds, and the lane's mechanism is unproven until a training draw - scale-after-signal, per the R9-H106 lesson
- **Artifacts** - `R10-H111_stage1_judged.parquet` (15.5MB, full judge output), `R10-H111_pairs_final.parquet`, `logs/R10-H111_judge.log`

**Track split (2026-08-05, author's order)**: dataset-GENERATION-method research continues in its own canonical log, `semantic-dataset-enhancements.md`, under task code **DR** (dataset refinement; first entries: the targeted-corruption fanout, DR-H112 through DR-H116). Hypothesis H-numbers remain global across both documents; training verdicts stay here.

**Round 10 training campaign - authorized and launched (2026-08-05)**: the author's order closes all three held decisions. (1) Lane training campaign LAUNCHED - each staged lane trains as an addition to the clean 685,670-pair mix (R9-H105 recipe, BCE + 12-way DANN λ0.02, 2 draws), admission bar: lane mean blind > 0.7031 clean mean under the PRIMARY windowed decomposed-min read through the frozen R8-H77 gate; order H107 (83,672) → H108 (61,184) → DR pilot lane (~26k, REPLACES the H111 lane - the DR engines dominate it on debris 62.2% → 0.2-14.3% and certified yield 8.9% → 30-98%; H111's parquet stays on disk as the fallback lane). (2) DR-2 pilot generation authorized - registration in `semantic-dataset-enhancements.md`. (3) **R8-H103 PARKED** - the 595.8M Qwen3-Reranker breaches the sub-400M deliverable budget and the decoder-head line closed with H102/H104; parked not killed - revisit only if a sub-400M decoder scorer with comparable finqa mechanics appears.

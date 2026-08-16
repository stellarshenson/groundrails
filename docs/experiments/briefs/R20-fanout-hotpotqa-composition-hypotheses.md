# R20 fanout brief - hotpotqa composition hypotheses

Fable hypothesis-design agent (read-only, 2026-08-16). Input to the R20 fanout adjudication; the coordinator adjudicates. Design-pass statistics inside must be re-run and banked before citation.

# hotpotqa composition lane — failure analysis, do-not-revisit list, and ranked hypotheses (design-only, zero GPU spent)

Three free CPU probes over banked artifacts were run as part of this design pass; their numbers are new and quoted below with the exact recipe so an executor can bank them as G0 artifacts.

---

## (a) Failure analysis

**Standing.** hotpotqa (multi-hop QA arena subset) reads 0.6766 / 0.6647 / 0.6751 / 0.6756 across the four banked flagship draws (`R18-H150_arm_draw{1,2}_windowed_result.json`, `R19-H160_arm_draw{3,4}_windowed_result.json`) — 4-draw mean **0.6730**, spread 0.0119, one of the most draw-stable subsets. Arena k=4 mean 0.71583, target 0.74. The subset carries **17 negatives in 250 items** (base rate 0.932); its own 95% CI is 0.211 wide — only paired comparisons resolve anything here (`R19-H162_hotpotqa_mechanisms.md` lines 10-11).

**The mechanism, banked (R19-H162).**
- **71.33% of claim sentences (209/293) need 2+ documents** by greedy anchor set cover; independent method agreement with the H140 G0 census 98.29% (`R19-H162_hotpotqa_probe.json`)
- **Partial-support saturation** — on multi-document sentences the max-window logit label gap is **−0.0006** (pos −3.6646 vs neg −3.6640), sentence AUROC 0.6017; single-document sentences separate at gap +2.2157, AUROC 0.7286. **82.3% of multi-doc positives score below the mean of single-doc negatives.** The score tracks coverage, not truth: corr(smax, best-single-doc containment) +0.5596, corr(smax, second-document coverage gain) −0.4243
- **Aggregation has the least purchase here of any subset** — mean pooling over the same banked logits moves hotpotqa **−0.0023** (vs techqa −0.1403); every fixed pooling sits inside the subset CI (probe JSON, `pooling_contrast_all_subsets`)
- **The skill is untaught, not unreachable** — the census is solid: **no training-mix positive requires more than one evidence document**; vitaminc (54% of the mix) trains at exactly 1.00 windows/row; the post-A1 mix is mean 1.507 windows, 20.1% multi-window. The cross-window channel exists in the shipped shape and is trained-off (`adapter_active = False`, output weights exactly zero)

**New numbers from this design pass (CPU, banked dumps only; recipe: join `R19-H162_hotpotqa_families.parquet` on (item_id, sent_idx) to per-window dumps `R19-H161_pairs_h150d{1,2}.parquet` and `R18-H151_scores_{1142,2142}.parquet`, per-checkpoint z-scored sentence-max, 4-checkpoint pooled, bootstrap over sentences):**

1. **The bridge_entity wrong-sign gap is sign-stable but statistically unrescuable from banked artifacts.** Per-checkpoint gaps: −0.57 / −0.418 / −0.497 / −0.391 (4/4 same sign on same-recipe checkpoints) — it is not a draw-1 fluke — but the pooled CI is **[−0.962, +0.435]**, still spanning zero. Pooling draws removes seed noise; it cannot remove the n_neg=10 sentence-sampling floor. The H164 revival clause **cannot be met by this statistic**. Side observation: h159d1 (enriched mix, different recipe) is the only checkpoint that flips bridge to +0.104 — weak-n but consistent with mix changes moving this family.
2. **The discrimination-collapse contrast IS a revival-grade statistic.** Difference of label gaps (single_doc − multi_doc), pooled over the 4 same-recipe checkpoints: **+0.886, CI95 [+0.043, +1.657], excludes zero**, sign-stable 4/4 (per-checkpoint +0.908/+0.835/+1.081/+0.718), on n 293 sentences / 23 negatives instead of 94/10. Caveat stated: the lower bound is thin (+0.043) and this number was computed in a design pass — it must be re-run and banked as the registered G0 artifact before citation.
3. **The second-best window carries no discarded label signal** (kills direction (b) standalone): on multi_doc sentences, 2nd-window logit AUROC 0.57/0.575 (draws 1/2), top-2-mean ≤ max at sentence level on every family both draws (e.g. conjoin_attrs top2 0.649/0.678 vs smax 0.663/0.678; bridge s2nd gap +0.003/+0.007). There is nothing below the max for a train-time soft pooling to bind **on the current mix**.
4. **Composed-negative masking is serving-structural.** 13 of 17 hotpotqa negative items are multi-doc; 4 of 17 multi-doc negative sentences already sit above the multi-doc positive median. Under max, a conjunctive negative's intact-leg window wins the max (registered limit, canonical log line 3424) — supply can raise composed-**positive** credit but cannot install composed-**negative** detection while the read is max-over-windows. This bounds any supply lane's ceiling: it fixes the pos-vs-single-doc-neg inversions (~19% of the item pair mass), not the multi-neg-vs-multi-pos pairs (~61%, stuck near chance). The banked prediction hotpotqa +0.02..+0.06 (memo line 113) is consistent with this bound.

---

## (b) Do-not-revisit list (with quoted revival conditions)

| Closed line | Verdict and citation | Revival condition, quoted |
|---|---|---|
| **R19-H164 composed-claim lane** | WITHDRAWN unbuilt 2026-08-15 (log :3811) — bridge gap −0.570 has CI [−2.567, +1.171] on n_neg 10 | "What would revive it - a registered variance protocol that makes its bar admissible, plus a motivating statistic with an interval that excludes zero. The hotpotqa families with n_neg < 10 are all unresolved, `conjoin_attrs` included at n_neg 4" (:3819). **Condition 1 is now met** (amendment V1 adopted 2026-08-16, floors k=2 0.01542 / k=3 0.01259 / k=4 0.01090). Condition 2 is met by re-basing to the difference-of-gaps statistic above — not by the bridge gap, which stays unresolved |
| **R16-H140 learned readout** | "NOT LICENSED at bar; kill branch NOT met" (F1 correction :2767); hotpotqa was the WORST mover −0.0518 | "route closure is coordinator adjudication subject to the recorded reopening condition" — no reopening evidence exists; hotpotqa mechanism verdict (signal-free inputs) makes reopening for THIS subset pointless |
| **R16-H142 G1 cross-window conditioning** | Init-paired ablation: hotpotqa **+0.0028, null** under the strictest control run (:58 of the memo) | none recorded — the channel was activated with no composed supply and the subset was indifferent |
| **R18-H156 learned aggregator** | KILLED at draw 1; **resolves-below under V1** (z −2.08, :3853); serving-read swap neutral −0.00008, training-gradient cost −0.0250 | class closure: "learned window aggregation now carries damage evidence on BOTH axes" (:3301). Note the closure covers LEARNED aggregation; deterministic train-time pooling was never run — but probe 3 above removes its precondition on the current mix |
| **R19-H165 pool-concatenation** | KILLED blind on both draws (−0.0116 / −0.0160) despite hotpotqa +0.094 replicated; the tables mechanism story WITHDRAWN; R19-H170 gate killed at zero cost (predicate reads 0.000 on every arena subset) | none for the global read; the C0 confound stands recorded (correction (3): spanning and more-text never separated) |
| **Neighbor-channel / SAT-aligned windows** | Closed by the H140 G0 census — hotpotqa boundary cuts 0.0; dispersion is purely cross-document (memo :117) | new evidence of within-document cuts, which the census excludes |
| **Serving-read pooling (any variant)** | H151: "max stands as PRIMARY read; no serving-read amendment" (:3285); hotpotqa poolings all inside its CI (softmax t4 0.6981 vs max 0.6766) | the read-convention question is ESCALATED TO THE AUTHOR (Brief B disposition :3862) — coordinator may not reopen it |
| **HotpotQA own train split** | WALLED — HotpotQA is one of the ten RAGBench safe-core source corpora (`reports/research-grounding-datasets.md` :49); "Hover (HotpotQA-derived - walled)" (:119) | none — contamination wall is absolute |

---

## (c) Hypotheses, registration shape, ranked by (expected mean gain × plausibility)/cost

Honesty preamble binding all of them: any plausible hotpotqa effect (+0.02..+0.06 on the subset) is **+0.002..+0.006 on the arena mean — below every affordable floor** (k=2 0.01542, k=4 0.01090, even k=6 0.00890). Every arm below is therefore **exploratory on the mean with a mechanism gate as PRIMARY**, per the variance protocol's sub-floor rule. Queue reality: H172 draws 5/6 in flight, R20-H174 (hagrid/emanual portfolio, the only mean-detectable arm on the board) owns the next training slot; these slot behind it.

### HYP-1 — Composed-claim supply, re-based (repairs and revives the H164 lane) — RANK 1

- **Claim**: because the per-window scorer's discrimination collapses on multi-document sentences (difference-of-gaps +0.886, CI [+0.043, +1.657], sign-stable 4/4 same-recipe checkpoints) and no training-mix positive requires more than one evidence document (census, solid), adding a composed-claim lane (~20-30k rows: TabFact two-table-join generator — bridge template with elided join key, conjunction/comparative template — every composed positive presented as a multi-document bag under the existing 1,500/750 MIL max objective, **every positive paired with a composed negative** whose broken bridge or flipped conjunct is absent from the bag) will raise the held-out composed-probe AUROC from near-chance to ≥ 0.70 while the table guard holds
- **Diagnostic kill-gate (free, before build)**: G0a — bank the difference-of-gaps statistic by the exact recipe above; **PASS = CI excludes zero** (expected +0.886 [+0.043, +1.657]). G0b — build the 1,000-item synthetic bridge + conjunction probes (CPU generation) and read the banked flagship checkpoint on them (one is a GPU read ~minutes; if strictly zero-GPU is required, G0b runs as the arm's step 0). **KILL if the baseline leg already reads ≥ 0.70** — nothing to install
- **Supply + contamination**: TabFact banked, CC-BY-4.0, not a RAGBench source at any remove — CLEAR by construction; HotpotQA train and HoVer WALLED (quoted above); generator never touches either
- **Predicted effects**: hotpotqa +0.02..+0.06 (positive-side inversions only — the masking bound caps it; composed-negative pairs stay near chance); mean +0.002..+0.006 — **sub-floor**
- **Bars, k**: declared **k=2**, ~13 GPU-h. PRIMARY = mechanism gate: composed-probe AUROC ≥ 0.70 against a pre-read near-chance baseline leg. Arena EXPLORATORY; GUARD: 2-draw mean ≥ flagship k=6 mean − 0.01542; TABLE GUARD (the H157 over-crediting risk, named in the memo :111): finqa/tatqa/delucionqa each within one across-seed spread of the flagship subset mean (0.062/0.025/0.012); HOLDS gold_full ≥ 0.84, non-EN ≥ 0.82
- **Kill**: G0 fail; baseline probe ≥ 0.70; draw-1 arena mean < 0.695; table-guard breach on the 2-draw mean → draw 2 unspent
- **Cost**: CPU lane + 13 GPU-h

### HYP-2 — Natural-register composed supply by document-splitting MiniCheck — RANK 2 (preferably lane L-B of the SAME arm)

- **Claim**: because TabFact-join items are templated and the campaign's own E6 finding traces failures to register gaps, splitting each MiniCheck multi-fact positive's document into two pseudo-documents at a fact boundary (so no single window covers the claim's anchor set — windowing is per-document, so doc length is irrelevant) manufactures natural-prose composed positives; negatives from MiniCheck's refuted claims plus single-conjunct swaps
- **Diagnostic kill-gate (free, CPU, banked zip)**: greedy set-cover census over the 14,395 banked MiniCheck pairs — **PASS ≥ 8,000 positives admit a 2-split where neither half covers ≥ 80% of claim anchors**; then the R14-H136 8-gram census RE-RUN on the split presentation (MiniCheck holds green today; the Appendix-D seed caveat — C2D seeds are Wikipedia-claim corpora — makes the 8-gram run binding, per its own dataset card)
- **Supply + contamination**: MiniCheck banked, MIT, gate green (`R19_minicheck_gate.json`); census re-run mandatory because splitting creates new pairings
- **Predicted effects / bars / kill**: identical shape to HYP-1; as a second lane in one arm its marginal GPU cost is zero
- **Cost**: CPU only if folded into HYP-1's arm

### HYP-3 — Positive-bag top-2 MIL objective — REFUTED standalone; survives only as an optional cell — RANK 4

- **Verdict from this pass**: the precondition is measured absent — the 2nd-best window carries no incremental label signal on the current model/mix (probe 3: s2nd AUROC 0.53-0.61, top2 ≤ max, both draws), and 79.9% of training rows are single-window so the objective would not engage. H151-legality is noted for the record (training-only change, serving max untouched; H156 closed LEARNED aggregation, not deterministic train-time pooling) — the lever is legal but currently pointless
- **Adapted form**: an optional third-draw cell on HYP-1/2's arm (lane + max vs lane + top-2-on-positives-with-≥2-windows), registered only if the author buys the draw AFTER the mechanism gate passes — composed supply is what would give the second window something to carry
- **Cost**: +6.5 GPU-h, conditional

### HYP-4 — Clause-level decomposition (the parked R19-H167) — the only lever reaching composed NEGATIVES — RANK 3 on value, UNREGISTRABLE without the author

- **Claim**: because the intact-leg window masks a conjunctive negative under max (structural, :3424; 13 of 17 hotpotqa negative items are multi-doc), and supply lanes cannot fix this by construction (probe 4), splitting claim sentences into independently-scored clauses before the min-over-sentences read would expose the broken leg — the only route to the ~61% of pair mass HYP-1 cannot touch
- **Blocker, stated plainly**: this changes the serving formula. The H151 ruling closed serving-read changes and the Brief-B serving-convention finding is already **escalated to the author as a strategic decision item, not actioned** (:3862). This hypothesis rides that escalation; the coordinator may not register it alone
- **Free kill-gate to price the escalation**: rule-based clause splitter over the 56 banked conjoin_attrs sentences (`R19-H162_hotpotqa_families.parquet`; clause containment machinery already exists, `clause_min_cont` 0.7214 banked) — PASS if ≥ 70% split into independently checkable clauses with both clauses' anchors covered by some document
- **Cost if licensed**: CPU splitter + one blind read per banked draw (~45 min GPU each), no training

### HYP-5 — Multi-hop corpus survey: 2WikiMultiHopQA, MuSiQue — RANK 5, contingency only

- Neither appears anywhere in `reports/research-grounding-datasets.md` or `data/external/datasets/` — never surveyed. Neither is HotpotQA-derived by construction (2Wiki: Wikipedia/Wikidata; MuSiQue: composed from single-hop QA sets), unlike HoVer which is walled — but licence verification plus the 8-gram wall census are mandatory before any use, and the fetch requires user authorization (external access). Fires only if HYP-1 and HYP-2 both fail their supply kill-gates. Cost: CPU + one survey pass

**Ranking rationale**: HYP-1 and HYP-2 share one 13 GPU-h arm, have free kill-gates that are already substantially pre-run (the revival statistic exists and excludes zero), clear supply, and satisfy both quoted H164 revival conditions. HYP-4 has the larger theoretical ceiling (it reaches the negatives) but is gated on an author ruling already in flight. HYP-3 is refuted standalone by a measurement made in this pass. All are exploratory on the arena mean — none competes with R20-H174 for the mean-detectable slot, and the coordinator should sequence any of these behind it.

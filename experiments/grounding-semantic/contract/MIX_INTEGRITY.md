# Mix integrity - one statement

Numbers recorded, not adjudicated - the coordinator adjudicates. Nothing here was re-measured; every figure is transcribed from a banked artifact named in the sources section.

## The answer

**The evaluation surfaces that carry the campaign's headline are clean; the training mix is not.** The blind arena reads exactly zero shared strings against the assembled 760,618-row mix on responses and on response sentences, in all six string-form pairings and both directions, and `gold_full` reads exactly zero on every channel, form and direction - so the headline blind-arena mean **0.71218 ± 0.00324 (k=6)** and the **+0.03255** margin over the incumbent are not products of string-level contamination. What the arena is not is spotless: **17 of 6,784 distinct arena documents (0.2506%) sit byte for byte inside a training chunk**, all 17 supplied by the `halueval` member and all 17 landing in the `hotpotqa` subset, touching **16 of 2,264 arena responses (0.7067%)**; the 8-gram census reads **0.00649 against a 0.02 KILL bar and a 0.005 WARN bar** - a WARN, not a kill - and whether dropping those responses moves the arena mean is **not measured**, because it needs a GPU read and all three cards carry live draws. On the training side, **nine of eleven members are non-conforming under the amended contract, covering 644,988 of 760,618 rows (84.80%)**; the failures are concentrated in disjointness from *mechanism evals* (C2, 576,358 rows / 75.78%), provenance (C8, 164,623 rows / 21.64%) and shortcut-learnability (C5, 48,000 rows / 6.31% decided plus 213,305 rows / 28.04% whose applicability is unruled). **No loaded member reproduces the poisoning signature of the withdrawn R20-H175b lane**: C1's structural test fires on 165 pairs / 336 rows of 760,618 (0.0442%) against a positive control that fires on 8,986 pairs / 17,972 rows exactly as registered, and every C1 band failure withdrew under the amendments. The one thing the mix is provably not is *shortcut-free*: **253,305 flagship rows (35.12%) sit in members whose label is recoverable from the claim string alone above 0.60**, led by `halueval` at **0.9519** and `frame_reject` at **1.0000**.

## 1. Evaluation integrity

### 1.1 Blind arena - the surface the headline is read on

10 RAGBench subsets, 2,264 responses, 6,784 distinct documents, the frozen R8-H77 gate. Scanned against the **760,618-row** portfolio mix, of which the **721,210-row** flagship mix is a strict tag-subset - a zero on the superset is a zero on the flagship.

| channel | measured | bar | clears |
|---|---|---|---|
| C2 exact, arena **responses** in mix, 6 form pairings x 2 directions | 0 of 2,264 | 0 | yes |
| C2 exact, arena **response sentences** in mix | 0 of 10,506 | 0 | yes |
| C2 exact, arena **documents** in mix | 1 of 8,645 | 0 | no |
| C2 document-id channel, raw / stem | 3 / 5 key families collide, all bare-numeric on both sides | 0 | contested |
| C4 8-gram census, Jaccard >= 0.3, bidirectional | 0.00649 (44 of 6,721 scorable documents) | KILL 0.02, WARN 0.005 | WARN |
| C4 reverse direction, mix units hit | 55 | reported | n/a |
| verbatim substring, arena document inside a training chunk | 17 of 6,784 (0.2506%) | 0 | no |
| arena responses retrieving a verbatim document | 16 of 2,264 (0.7067%) | 0 | no |

- The single exact document collision is a **one-character punctuation-only string** occurring in three finqa document slots and matching a `vitaminc` claim - a degenerate unit, not a passage
- All three document-id key families that collide are **bare integers on both sides** (`row_id` x `ragtruth_en.id`, x `psiloqa.wiki_title`, x `vitaminc.FEVER_id`) - identifier-space collisions, not shared documents
- Every one of the 44 Jaccard hits and all 17 verbatim documents trace to **`halueval`** (37 mix units at Jaccard >= 0.3, max 0.7903); no other member reaches the threshold against any arena unit
- Exposure by subset: **hotpotqa 42 of 995 documents (4.22%)**, hagrid 2 of 654 (0.31%), **0 in the other eight subsets**
- Response-level exposure is concentrated in one of ten subsets: hotpotqa **16 of 250 responses (6.4%) verbatim**, **102 of 250 (40.8%) at 8-gram containment >= 0.10**; hagrid 15 of 250 (6.0%) at containment >= 0.10; every other subset reads 0 at every threshold
- Whole-arena containment ladder: >= 0.10 touches 143 documents / 117 responses (5.17%); >= 0.25 touches 85 / 74; >= 0.50 touches 35 / 32; = 1.00 touches 17 / 16
- Containment was added because the banked Jaccard instrument **misses re-wrapped text**: the live re-chunk control reads Jaccard 0.2796-0.8841 on passages that are 0.9891-1.0000 contained, so the 44-document Jaccard count is a lower bound
- The **load-bearing read is not measured** - deciding whether the arena mean moves when the 16 or the 117 exposed responses are dropped needs per-row arena scores for the banked checkpoints, which is a GPU read; `R21-H179` is launched and queued behind three live training draws

### 1.2 `gold_full` - the in-domain hold, bar >= 0.84

2,752 claims / 123,579 loaded rows / 619 documents, label base rate 0.7144.

| channel | measured | bar | clears |
|---|---|---|---|
| C2, worst colliding gold units, any channel, any form, both directions | 0 | 0 | yes |
| C2, worst colliding mix rows | 0 | 0 | yes |
| C4 claims census / evidence census | 0.0 / 0.0 | KILL 0.02 | yes |
| document channel, max mix-chunk containment into a gold document | 0.018349 | reported | n/a |
| document channel, max gold-document coverage by the whole mix | 0.000246 | reported | n/a |
| C6 mix-supplied key coverage (passage and document) | 0.0 | zero coverage makes the clause NOT-APPLICABLE under C-A2 | n/a |
| reproduction of the banked `R20_goldfull_split_audit` | 29 of 30 headline numbers reproduce | 30 of 30 | no |

- The single reproduction mismatch is a **control-construction defect, not a gate defect**: the banked claims-channel spike injects each DANN group's lexicographically smallest claim, and `halueval`'s normalises to **zero tokens**, so it carries no 8-gram and cannot hit under any correct gate. Rebuilt with injected units that carry at least one n-gram, the gate detects **10 of 10**
- The surface is **clean but statistically smaller than its row count suggests**: label ICC across documents 0.7821, design effect 3.695, **effective sample size 744.8 against a nominal 2,752**, so the 95% CI half-width at p = 0.84 is **0.0263 rather than 0.0137**
- 80.89% of claims sit in label-pure documents and the **document-keyed leave-one-out accuracy is 0.907** against a 0.7144 majority baseline - a within-surface clustering diagnostic, not a mix-supplied C6 channel, which reads zero coverage
- Internal duplication: 123,579 loaded rows carry only **68,434 distinct (owner, chunk) pairs**, so 55,145 rows (44.62%) are exact repeats inside a claim's own chunk list

### 1.3 Mechanism evals

Sixteen non-arena instruments were enumerated and verified; `R20-H177_eval_B`, `R17-H143_evalset`, the arena and `gold_full` were verified separately. **Every one of the sixteen reads 0 against the blind arena and 0 against `gold_full` in all six string forms, both directions.**

| eval | tier | rows / pairs | evidence in mix | claims in mix | documents in mix (stem) | C1 structural | C5 claim-only |
|---|---|---|---|---|---|---|---|
| `antigaming_nearmiss_bindrow` | 1 live | 3200 / 1600 | 7 of 1177 | 15 of 3186 | **59 of 1173 (5.03%)** | 0 | **0.6217** |
| `h166a1_vitaminc_holdout` | 1 live | 38126 / 38126 | 2 of 30707 | 2 of 37470 | 0 of 5553 | **6** | n/a |
| `findver` | 1 live | 2400 / 2400 | 0 | 0 | 0 of 539 | 0 | n/a |
| `eval_C` | 1 live | 1936 / 968 | 0 | 0 | 0 of 213 | 0 | 0.4757 |
| `R20-H177_eval_B` (rebuilt) | 1 live | 2000 / 1000 | 0 of 731 | 0 of 2000 | 0 of 324 | 0 | 0.4974 |
| `r15_typeprobe` | 2 retired | 20032 / 5008 | 4 of 1142 | 21 of 16507 | 54 of 1136 | **6** | n/a |
| `r15_bindprobe` | 2 retired | 4800 / 2400 | 3 of 808 | 28 of 3852 | 41 of 804 | 0 | **0.5508** |
| `h150_unitswap_probe` | 2 retired | 280 / 140 | 0 | 0 | **27 of 42 (70% of rows)** | 0 | **0.1425** |
| `h148_itemindex_probe` | 2 retired | 1956 / 978 | 0 | 0 | 0 | 0 | 0.5065 |
| `h149_roleswap_probe` | 2 retired | 716 / 358 | 0 | 0 | 0 | 0 | 0.4729 |
| `R17-H143_evalset` | 2 no live arm | 1050 / 550 | 10 of 547 | 0 | **352 of 352 TabFact (100%)** | 0 | 0.5091 |
| `antigaming_traced` | 3 no gate | 2620 / 1310 | 2 of 1076 | 0 | 47 of 979 | 0 | **0.6999** |
| `g0b_composed_probes` | 3 no gate | 1500 / 500 | 5 of 500 | 0 | 28 of 564 | 0 | **0.6867** |
| `h117_heldout_pairs` | 3 no gate | 4000 / 2000 | **306 of 1933 (15.5% of rows)** | 286 of 4000 | 0 | 0 | **0.6711** |
| `h175b_eval_clean` | 3 no gate | 88 / 44 | 0 | 0 | 0 | **44 of 44** | 0.5 |
| `h175b_eval_clean_prefix` | 3 no gate | 32 / 16 | 0 | 0 | 0 | **16 of 16** | 0.5 |
| `dr_h113_gate_judged` | 3 no gate | 1505 / 1505 | 159 of 1446 | 0 | 0 | 0 | n/a |
| `r12_h121_gateBC_rows` | 3 no gate | 6786 / 6786 | 1059 of 3346 | 2476 of 2476 | 0 | n/a | n/a |

Readings that carry live or standing gates:

- **`R20-H177_eval_B` has been rebuilt and is clean.** 2,000 rows / 1,000 pairs / 465 documents / 727 passages, identical size, family composition and cell fills to the contaminated original, reading **zero on every channel** - 0 passages and 0 claims in the mix across all six string forms both directions, 0 of 324 TabFact document stems in the member, 0 against lane B, the arena and `gold_full`. The **original** eval_B reads 325 of 325 TabFact document stems in the member, 1,300 of 2,000 rows (65%)
- **The anti-gaming hold is not one instrument.** Its near-miss half is byte-identical (Jaccard 1.0) across all 14 banked arm files, so the headline `near-miss >= 0.7438` read is comparable across arms; its `bind_row` half is a **different 600-pair set on every arm** (Jaccard 0.0017-0.0127 against the flagship's), because `R14-H133_antigaming.build_bindrow` calls Polars `unique` without `maintain_order=True` and then indexes a seeded permutation of a differently ordered list. The seed is constant, the supply order is not
- **`R19-H166-A1`'s primary eval carries 6 (claim, evidence) pairs holding both labels** and 3 of 38,126 rows (0.008%) inside the mix, visible only under whitespace normalisation, which the banked builder's raw-string filter could not see. The arm is author-assented and still in the training queue
- **`findver` and `eval_C` are clean on every channel** - the R20-H176 CONFIRMED verdict and the R20-H177 Lane C primary rest on uncontaminated instruments
- **`R17-H143_evalset` is contaminated at document level and above the C4 kill bar** - its entire TabFact half (352 of 352 documents) sits on training documents and its 8-gram census reads **0.06947 against a 0.02 KILL**. Its banked reads move by at most **0.00241** under exclusion of the string-form hit set and by at most **0.007379** under the wider n-gram exclusion; every banked AUROC reproduced at banked precision before exclusion. No live arm reads it
- **The RAGTruth in-domain holds are verified.** The EN hold (`ragtruth_en >= 0.7039`) and the non-EN hold (`>= 0.82` / lineage bar 0.6095) are read on the corpora's own `__test` splits: 0 shared context strings train-vs-test in EN, 2 of 450 test contexts at 8-gram Jaccard >= 0.3 (0.44%); 0 shared prompt strings in every one of the seven translations, 1-4 shared answer strings per language (0.037-0.149% of test answers)

## 2. Training integrity

Eleven members. The **flagship** mix (721,210 rows, 14 DANN groups, the R18-H150 recipe carrying 0.71218) is the first eight; the **portfolio** mix (760,618 rows, 17 groups, the in-flight R20-H174 draws) adds the last three. State is under the contract as amended by C-A1 and C-A2.

| member | rows | % flagship | % portfolio | state | binding failures | binding number |
|---|---|---|---|---|---|---|
| `vitaminc` | 370,653 | 51.393 | 48.731 | conformable, rebuild on disk passes | C2 | 2 units against the H166-A1 holdout under whitespace normalisation; rebuild costs 260 rows (0.0701%) |
| `ragtruth_translated` | 105,630 | 14.646 | 13.887 | CONFORMING, pending the C5 applicability ruling | none | claim-only 0.7771-0.7916 across 7 languages if C5 is ruled to bind |
| `tabfact` | 92,585 | 12.837 | 12.172 | conformable, rebuild on disk passes all eight | C2, C3, C8 | rebuild costs 6,379 rows (6.89%) |
| `psiloqa` | 61,712 | 8.557 | 8.113 | conformable, rebuild closes C2 but is incomplete | C2 | 485 + 406 passages shared with the withdrawn R20-H175b evals; rebuild costs 11,238 rows, of which **9,859 buy a struck bar** |
| `halueval` | 40,000 | 5.546 | 5.259 | **NOT conformable at usable size** | C5 | claim-only **0.9519**; best conforming subset 480 rows (1.2%) |
| `quant_misbind` | 30,000 | 4.160 | 3.944 | conformable, rebuild on disk passes all eight | C2, C3, C8 | rebuild costs 11,348 rows (37.83%), the FEVEROUS third dropped whole |
| `ragtruth_en` | 15,090 | 2.092 | 1.984 | one recorded date from conforming | C8 | no recorded retrieval date; C1 and C6 both withdrew under the amendments |
| `quant_scale_unit` | 5,540 | 0.768 | 0.728 | conformable, rebuild not built | C3, C8 | 1,040 rows (18.77%) from an unlicensed, undated, split-less FEVEROUS snapshot |
| `attr_pool` | 21,408 | n/a | 2.815 | rebuild passes all eight under C-A1, at 79.25% cost | C2, C6, C8 | rebuild delivers **4,442 rows against a registered 20,000-30,000 band** |
| `path_bind` | 10,000 | n/a | 1.315 | CONFORMING | none | C1 rests on one executor-added instrument (see 4.4) |
| `frame_reject` | 8,000 | n/a | 1.052 | **NOT conformable - the defect is the construction** | C5 | claim-only **1.0000**, within-pair 1.0000, evidence-only exactly 0.5000 |

Clause rollup over the portfolio mix, amended:

| clause | failing members | failing rows | share |
|---|---|---|---|
| C1 label commensurability | none | 0 | 0.00% |
| C2 disjointness from eval surfaces | attr_pool, psiloqa, quant_misbind, tabfact, vitaminc | 576,358 | 75.78% |
| C3 split semantics | quant_misbind, quant_scale_unit, tabfact | 128,125 | 16.84% |
| C4 contamination census | none | 0 | 0.00% |
| C5 no shortcut channel | frame_reject, halueval | 48,000 | 6.31% |
| C6 no memorisation channel | attr_pool | 21,408 | 2.81% |
| C7 units and volume | none | 0 | 0.00% |
| C8 provenance | attr_pool, quant_misbind, quant_scale_unit, ragtruth_en, tabfact | 164,623 | 21.64% |

- **No C2 failure touches the arena or `gold_full`.** Every one of the five failing members reads exactly 0 against both, in all three string forms and both directions. All five C2 failures are against *mechanism evals*: `R20-H177_eval_B` (tabfact 325 of 325 documents, quant_misbind 19 evidence strings), the anti-gaming probe sets (quant_misbind 15-18 claims per file, tabfact 50-62 stem-colliding tables per file at mean token Jaccard 0.91), the H166-A1 holdout (vitaminc 2 units, attr_pool 1 claim) and the withdrawn R20-H175b evals (psiloqa 485 + 406 passages)
- **C1's structural test - the poisoning signature - fires on 165 pairs / 336 rows of 760,618 (0.0442%)**: vitaminc 115 pairs / 236 rows, tabfact 45 / 90, psiloqa 5 / 10, and **0 on all five constructed lanes**. vitaminc's and tabfact's conforming rebuilds already remove theirs; psiloqa's conformed variant does not
- **Nothing that is loaded is conformed.** The live draws load `R17-H146_lane.parquet`, `R18-H150_scaleunit_lane.parquet` and the three `R20-H174_lane_L*.parquet` parents; `R17-H146_lane_conformed.parquet`, `R20-H174_lane_L2_conformed.parquet`, `psiloqa_conformed.parquet`, `vitaminc_conformed.parquet` and `tabfact_member_conformed.parquet` are loaded by nothing

### 2.1 Which members are shortcut-learnable, and by how much

C5's claim-only converged probe measures how much of the label is recoverable from the claim string with the evidence removed. The bar is **< 0.55**.

| member | claim-only AUROC | rows | % flagship | C5 status |
|---|---|---|---|---|
| `frame_reject` | **1.0000** | 8,000 | n/a (portfolio 1.05%) | FAIL - constructed lane, C5 binds unambiguously |
| `halueval` | **0.9519** | 40,000 | 5.546 | FAIL |
| `ragtruth_en` | **0.7965** | 15,090 | 2.092 | applicability contested |
| `ragtruth_translated` | **0.7771-0.7916** (7 languages) | 105,630 | 14.646 | applicability contested |
| `h117_heldout_pairs` (eval) | 0.6711 | - | - | no live gate |
| `antigaming_nearmiss_bindrow` (eval) | 0.6217 | - | - | carries a suspended hold |
| `tabfact` | **0.6031** (table-disjoint held-out) | 92,585 | 12.837 | applicability contested |
| `r15_bindprobe` (eval) | 0.5508 | - | - | gate retired |
| `attr_pool` | 0.5281 | 21,408 | n/a (portfolio 2.82%) | PASS under C-A1 |
| `path_bind` | 0.5158 | 10,000 | n/a (portfolio 1.32%) | PASS |
| `quant_misbind` | 0.5049 | 30,000 | 4.160 | PASS |
| `vitaminc` | 0.4998 | 370,653 | 51.393 | PASS on the measurement, applicability contested |
| `quant_scale_unit` | 0.4257 | 5,540 | 0.768 | PASS |
| `psiloqa` | **NOT MEASURED** | 61,712 | 8.557 | applicability contested |

- **253,305 flagship rows (35.12%)** sit in members reading above 0.60 on this channel: halueval, ragtruth_en, ragtruth_translated, tabfact. Adding frame_reject gives 261,305 portfolio rows (34.35%)
- `halueval`'s channel is **carried redundantly and cannot be filtered away**: 0.8972 with every content word masked, 0.9016 with every function word, hedge, capital and punctuation deleted, 0.9220 inside claim-length deciles, 0.8255 on length alone. Twelve subset strategies over eighteen retention levels yield exactly one subset clearing C5's pooled conjunction, at **480 of 40,000 rows (1.2%)**; retention frontier 0.9412 / 0.9192 / 0.8469 / 0.6690 at 90 / 75 / 50 / 25%
- `frame_reject`'s label is a deterministic function of the claim string; the evidence contributes exactly nothing (evidence-only 0.5000)
- Whether C5 binds the five source corpora at all is **unruled**. If the coordinator's generated-vs-observed ruling captures the RAGTruth family and tabfact, a further **213,305 rows (28.04% of the portfolio mix)** fail on a claim-alone channel. If it captures only manufactured negatives, nothing moves. `vitaminc` clears either way; `psiloqa` cannot be priced at all

## 3. What remains broken, and what it costs

| # | item | consequence | fix class |
|---|---|---|---|
| 1 | `halueval` claim-only 0.9519, 40,000 rows | 5.55% of every flagship draw and 5.26% of every portfolio draw does not require the model to read the evidence. C1 holds throughout (negative-leg attestation 0.1032 against the positive leg's 0.6592 at the 0.90 containment threshold), so the support head is not poisoned. The only subsets reaching the bar keep 1-2% of the rows | **retirement**, or a regeneration outside this pass's scope: rewrite the positive leg with the same model that wrote the negative leg so both legs share a register |
| 2 | `frame_reject` claim-only 1.0000, 8,000 rows | 1.05% of every portfolio draw updates the support head on a decision the evidence does not enter. Not in the flagship mix | **retirement** - the shortcut is the construction |
| 3 | Arena exposure: 17 verbatim documents, 16 responses, C4 0.00649 WARN | The headline 0.71218 and the +0.03255 margin rest on a surface with sub-1% verbatim document exposure concentrated in one of ten subsets. Effect on the number **unmeasured** | **measurement**, then a decision: `R21-H179` per-item arena scoring, queued behind three live draws |
| 4 | Anti-gaming `bind_row` half differs on every arm | Any cross-arm comparison of the `bind_row >= 0.95` sub-read is not like-for-like; the headline `near-miss >= 0.7438` half is byte-identical and comparable. The band is a recorded diagnostic since the 2026-08-16 author ruling | **pipeline** - pass `maintain_order=True` to the Polars `unique` in `R14-H133_antigaming.build_bindrow` and rebuild |
| 5 | `R19-H166-A1` holdout: 6 both-label pairs, 3 of 38,126 rows in the mix | The arm's primary gate (`>= 0.85`, baseline leg 0.3935) is read on an instrument with a 0.008% overlap and 6 structurally impossible items. Arm still queued | **rebuild** of the holdout with the filter applied on the whitespace-collapsed case-folded form; drops 5 of 38,126 rows |
| 6 | Old `R20-H177_eval_B`: 65% document contamination | The banked baseline-leg reads on the OLD eval (two-draw mean 0.5064) are not comparable to reads on the rebuilt file and need re-reading. The rebuilt eval is on disk, verified clean, same size and shape, gate readable to ±0.0198 at 2 SE | **done** - `R20-H177_eval_B_rebuilt.parquet`; the re-read is outstanding |
| 7 | `R17-H143_evalset`: 100% TabFact document contamination, C4 0.06947 above a 0.02 kill | Banked reads move by at most 0.00241 (string) / 0.007379 (n-gram). No live arm reads it; the R17 distillation arc is closed | **retirement or rebuild**, coordinator's call - nothing live depends on it |
| 8 | C2 against mechanism evals, 576,358 rows | Any ranking of draws that leans on `R17-H143_evalset` or the old `eval_B` is read on surfaces the mix touches at document level. The arena and `gold_full` are untouched | **pipeline** - conforming rebuilds exist for vitaminc / tabfact / quant_misbind / psiloqa and are loaded by nothing |
| 9 | C8 provenance, 164,623 rows | Five members carry no recorded retrieval date, no licence on part of their supply, or a hand-written volume string in the sidecar that is 302 rows off the archive | **pipeline** for four; `quant_scale_unit`'s 1,040 FEVEROUS rows need the cut `quant_misbind`'s rebuild already executed |
| 10 | C1 structural, 336 rows across vitaminc / tabfact / psiloqa | 0.0442% of the mix carries a label no function of (claim, evidence) can produce. Under the 1,500-char truncated presentation the count rises to 920 rows; the flagship serves untruncated windowed evidence, so the raw reading is operative | **pipeline** - already removed by vitaminc's and tabfact's rebuilds, not by psiloqa's |
| 11 | `attr_pool` conforming rebuild delivers 4,442 rows | 79.25% cut against a registered 20,000-30,000 band. Loaded by no run | **rebuild** at a different design point, or accept the size |
| 12 | `psiloqa` rebuild spends 9,859 rows on a struck bar | 16.0% of the member was cut to move a C1 delta across the band C-A2 struck. Only the 1,379-row C2 cut is still binding; a re-cut returns 50,474 → 60,333 rows but must also add the structural filter the conformed variant lacks | **pipeline** |
| 13 | `gold_full` effective n 744.8 against a nominal 2,752 | The `>= 0.84` in-domain hold carries a 95% CI half-width of **0.0263**, not 0.0137. The surface is clean; it is smaller than its row count implies | **none available** - a corpus property (label ICC 0.7821 across 619 documents) |
| 14 | `ragtruth_en` / `ragtruth_translated` C8 contradiction | Identical archive-timestamp evidence yields a C8 FAIL on one member and a PASS on seven others. Either the retrieval-date limb is satisfied by archive timestamps, in which case ragtruth_en's FAIL falls, or it is not, in which case six PASSes fall | **ruling**, then a one-line provenance record |

## 4. What this pass could NOT establish

Stated so a reader can separate what is proven from what is merely unrefuted.

### 4.1 Not measured at all

- **Whether the arena contamination moves the arena number.** The 16 verbatim-exposed responses and the 117 responses at containment >= 0.10 have never been dropped and the mean re-read. That requires per-row arena scores for the banked checkpoints, which is a GPU read; `R21-H179` is launched and waiting on GPU0 behind three live training draws
- **`psiloqa`'s claim-only channel.** It is the only member of eleven with no claim-only reading on record - its C5 was returned NOT-APPLICABLE without a proxy, so its 61,712 rows (8.56% of the flagship mix) cannot be priced on the shortcut question either way
- **`halueval`'s regeneration repair.** Regenerating the positive leg with the model that wrote the negative leg is the only avenue the channel diagnostics point at, and it was not attempted: it needs LLM generation, outside this campaign's CPU-only offline scope, and would produce a new member rather than a conformed one
- **The 8 mechanism-eval reads that were never re-taken.** Where a banked verdict was read on a contaminated instrument (old `eval_B`, `R17-H143_evalset`, the `bind_row` sub-read), the exclusion delta was computed only for `R17-H143_evalset`

### 4.2 Instruments that are lexical only

- **Every disjointness and contamination instrument in this pass is lexical**: exact string equality on six form pairings, 8-gram Jaccard, 8-gram containment, verbatim substring search, TF-IDF character/word probes. **No embedding, paraphrase or cross-lingual near-duplicate search was run anywhere.** A paraphrased or translated copy of an arena document inside the mix would be invisible to every gate reported here
- This is a live risk for the **seven translated RAGTruth members (105,630 rows, 14.65% of the flagship mix)**, whose relationship to any English surface is by construction a translation, not a string match
- The Jaccard instrument is **demonstrably blind to re-wrapping**: the live re-chunk control fires 9 of 10 on Jaccard while reading 0.9891-1.0000 on containment. Containment covers that gap for the arena; the per-member C4 censuses were run on Jaccard alone

### 4.3 Surfaces and channels not covered

- **FEVEROUS document ids are unresolvable.** `R17-H143_evalset`'s FEVEROUS half carries index-form ids (`feverous:{i}`) produced by an order-unstable dedup, so a zero on that id channel is not evidence of document disjointness; only the content channel is load-bearing there
- **One banked helper truncated the mix before comparing.** `R17-H143_evalset_assessment.build_mix` loads two separate module instances, so its `untruncated_evidence()` patch applied to a different config object and its C2 evidence channel was measured against mix text cut at 1,500 characters - any collision living past that cut was outside its reach. The arena and `gold_full` passes do not use that helper and reproduce the banked window census exactly
- **The C2 surface list is closed, not exhaustive.** Members were verified against the 8-18 registered evaluation surfaces each report enumerates. A surface built after those runs is unverified by construction
- **`R20-H177_eval_B`'s claim-string channel against lane B was never measured before this session.** The rebuild found 12 rows / 6 pairs colliding there and added a build-time claim wall; the **original** eval_B still reads 2 rows / 1 pair on that channel, and no banked assessment covers it

### 4.4 Claims resting on a single instrument

- **Three lanes hold C1's strict-separation test only on a predicate-sensitive instrument their own reports file as executor-added.** `path_bind` - the campaign's uniform bag-of-tokens instrument reads the negative leg **above** the positive at >= 0.90 (0.3910 vs 0.3890) and **exactly equal** fully attested (0.1742), the signature C-A2 names; the verdict is carried by a contiguous-run probe the report itself says joins no bar. `quant_misbind` - fully attested exactly equal at 0.1086, carried by a binding-level re-derivation. `quant_scale_unit` - the blind instrument reads 0.0 on both legs, carried by a unit-resolved instrument. C-A1 makes the predicate-sensitive reading the mandated one, so the verdicts stand as amended - but `path_bind` is one of only two members returned as conforming, and its C1 rests entirely on that one probe
- **`gold_full`'s design-effect correction is computed on the LABEL, not on model correctness.** No per-claim correctness vector is banked for `gold_full`, so the effective-n figure bounds the inflation on a model read only to the extent that errors follow the label
- **The C4 census and the C2 exact channel are the only two instruments run against the arena on the member side.** The arena-side pass added containment and verbatim substring search; the eleven per-member passes did not

### 4.5 Reproducibility defects found in banked code

- `R17-H144_pairs.excluded_tables` breaks numeral-ranking ties over a `frozenset`, making it Python-hash-seed dependent
- `R14-H133_antigaming.build_bindrow` selects a different 600-pair set on every run, because Polars `unique` does not preserve order without `maintain_order=True`
- The banked `gold_full` claims-channel spike control is dict-iteration-order dependent and detects 9 or 10 of 10 depending on which bucket is enumerated first

## 5. Live positive controls behind these numbers

This pass ran no gate. Every gate whose number is cited above was proven with a known-bad input:

| gate | fed | result | fires |
|---|---|---|---|
| C1 structural (the poisoning signature) | the withdrawn `R20-H175b_qlane.parquet`, poisoned by construction | 8,986 pairs / 17,972 rows, exactly as registered | yes |
| arena C4 8-gram | 10 real mix chunks injected arena-side | 10 of 10 at Jaccard 1.0 | yes |
| arena C4 8-gram | 10 mix chunks re-wrapped at a different offset | 9 of 10 on Jaccard, **10 of 10 on containment** | partial - Jaccard misses re-wraps |
| arena C4 8-gram | VitaminC's own official test+validation evidence against the VitaminC train the mix carries | 620 of 53,804 detected, max Jaccard 1.0 | yes |
| arena C4 8-gram | arena documents with tokens randomly permuted | 0 of 10 | correctly silent |
| verbatim substring | a real mix chunk's own interior 400-char slice / the same slice with one character changed | found / not found | yes |
| document channel, stem-keyed | the original `R20-H177_eval_B` | 325 of 325 TabFact stems, 1,300 of 2,000 rows | yes |
| string channel, six-form | `R20-H175b_qlane_eval.parquet`, banked at 485 of 487 | 485 of 487, including its 449-raw / 485-truncated split | yes |
| string channel, normalisation form | 251 mix passages re-wrapped with newline plus indent | raw 1 of 251, normalised 251 of 251 | yes - only the normalised form sees it |
| document channel, stem key | 300 member table ids with the `1-`/`2-` prefix flipped | raw 0, stem 300 | yes - only the stem channel sees it |
| C5 claim-only probe | a 1,500-pair toy differing only in register / a label-shuffled halueval / a within-pair-shuffled halueval | 1.0000 / 0.5017 / 0.4951 | yes |
| `gold_full` C6 | a poisoned key-label association | AUROC 0.9981 | yes |
| `gold_full` document channel | the true parent document of a mix chunk | containment 0.973616 | yes |
| `gold_full` C4 claims spike | each group's lexicographically smallest claim | **9 of 10 as shipped**, 10 of 10 with n-gram-bearing units | control defect, not gate defect |
| halueval C4 | 10 arena documents thinned to two of every three sentences | 10 of 10 over a baseline of 2 | yes |

## Sources

Every number above is transcribed from one of:

- `contract/PHASE1_SYNTHESIS.json`, `contract/phase1_readjudication.json`
- `contract/arena_surface_report.json`, `contract/gold_full_surface_report.json`
- `contract/mechanism_evals_report.json`, `contract/mechanism_evals_summary.md`, `contract/mechanism_evals_spotchecks.json`
- `contract/halueval_conformed_report.json`, `contract/halueval_conform_diag.json`, `contract/halueval_conform_frontier.json`
- `contract/{attr_pool,frame_reject,halueval,path_bind,psiloqa,quant_misbind,quant_scale_unit,ragtruth_en,ragtruth_translated,tabfact,vitaminc}_contract_report.json`
- `contract/{attr_pool,psiloqa,quant_misbind,tabfact,vitaminc}_conformed_report.json`
- `experiments/grounding-semantic/R20-H177_eval_B_rebuilt_report.json`, `R17-H143_evalset_report.json`
- `docs/experiments/semantic-grounding-sota.md` and `docs/experiments/semantic-grounding-experiments.md` for the headline 0.71218 ± 0.00324 and the +0.03255 margin

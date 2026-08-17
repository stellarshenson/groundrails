# Held-out mechanism evals - contract verification, ranked by consequence

Numbers recorded, not adjudicated - the coordinator adjudicates.

Sixteen artifacts, verified against `docs/experiments/dataset-contract.md` with amendments C-A1 and C-A2 applied. `R20-H177_eval_B`, `R17-H143_evalset`, the blind arena and `gold_full` are excluded - other agents own them this session.

## Per-eval measurements

| # | eval | tier | rows / pairs | C2 evidence in mix | C2 claims in mix | C2 docs in mix (stem) | C1 structural | C5 claim-only | C5 within-pair | C6 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `antigaming_nearmiss_bindrow` | 1 | 3200 / 1600 | 7 of 1177 | 15 of 3186 | 59 of 1173 | 0 | 0.6217 | 0.6206 | 0.5 |
| 2 | `h166a1_vitaminc_holdout` | 1 | 38126 / 38126 | 2 of 30707 | 2 of 37470 | 0 of 5553 | 6 | n/a | n/a | 0.5001 |
| 3 | `findver` | 1 | 2400 / 2400 | 0 of 2370 | 0 of 2400 | 0 of 539 | 0 | n/a | n/a | n/a |
| 4 | `eval_C` | 1 | 1936 / 968 | 0 of 380 | 0 of 1926 | 0 of 213 | 0 | 0.4757 | 0.4804 | n/a |
| 5 | `r15_typeprobe` | 2 | 20032 / 5008 | 4 of 1142 | 21 of 16507 | 54 of 1136 | 6 | n/a | n/a | 0.5 |
| 6 | `r15_bindprobe` | 2 | 4800 / 2400 | 3 of 808 | 28 of 3852 | 41 of 804 | 0 | 0.5508 | 0.5502 | 0.5 |
| 7 | `h150_unitswap_probe` | 2 | 280 / 140 | 0 of 115 | 0 of 278 | 27 of 42 | 0 | 0.1425 | 0.1714 | n/a |
| 8 | `h148_itemindex_probe` | 2 | 1956 / 978 | 0 of 390 | 0 of 1813 | 0 of 100 | 0 | 0.5065 | 0.5092 | n/a |
| 9 | `h149_roleswap_probe` | 2 | 716 / 358 | 0 of 342 | 0 of 714 | 0 of 240 | 0 | 0.4729 | 0.4735 | n/a |
| 10 | `antigaming_traced` | 3 | 2620 / 1310 | 2 of 1076 | 0 of 2616 | 47 of 979 | 0 | 0.6999 | 0.8756 | 0.5 |
| 11 | `g0b_composed_probes` | 3 | 1500 / 500 | 5 of 500 | 0 of 1499 | 28 of 564 | 0 | 0.6867 | 0.908 | n/a |
| 12 | `h117_heldout_pairs` | 3 | 4000 / 2000 | 306 of 1933 | 286 of 4000 | 0 of 1978 | 0 | 0.6711 | 0.8195 | 0.5097 |
| 13 | `h175b_eval_clean` | 3 | 88 / 44 | 0 of 22 | 0 of 44 | 0 of 22 | 44 | 0.5 | 0.5 | n/a |
| 14 | `h175b_eval_clean_prefix` | 3 | 32 / 16 | 0 of 8 | 0 of 16 | 0 of 8 | 16 | 0.5 | 0.5 | n/a |
| 15 | `dr_h113_gate_judged` | 3 | 1505 / 1505 | 159 of 1446 | 0 of 1505 | 0 of 6 | 0 | n/a | n/a | 0.4585 |
| 16 | `r12_h121_gateBC_rows` | 3 | 6786 / 6786 | 1059 of 3346 | 2476 of 2476 | 0 of 12 | n/a | n/a | n/a | n/a |

Every eval reads 0 against the blind arena and 0 against `gold_full` on all six string forms in both directions.

## What reads each instrument, and what this pass found

### 1. `antigaming_nearmiss_bindrow` - tier 1, carries a LIVE or STANDING gate

**Reads it**: the anti-gaming hold. BINDING on R18-H150 / H152 / H155 / H156 / R19-H159 / H160 (`near-miss >= 0.7438`, the flagship's own d2 passing by +0.0049) and a recorded DIAGNOSTIC since author ruling 3 of 2026-08-16, which suspended the band. The `bind_row >= 0.95` clause rides the same file

**Found**: the near-miss half is byte-identical across all 14 banked arm sets, so the HEADLINE hold is comparable; the bind_row half is a different 600-pair set on every arm (Jaccard 0.0017-0.0127 against the flagship's). 5.03% of the eval's TabFact documents (59 of 1,173 stems, 166 of 3,200 rows) are inside the mix under the stem key while the raw key reads 0; 15 of 3,186 eval claims are verbatim mix claims, all in the `quant_misbind` lane; the claim-only probe reads 0.6217 against C5's < 0.55 and within-pair 0.6206 against < 0.60

### 2. `h166a1_vitaminc_holdout` - tier 1, carries a LIVE or STANDING gate

**Reads it**: the PRIMARY mechanism gate of R19-H166-A1 (`held-out VitaminC REFUTES-vs-NEI AUROC >= 0.85`), an author-assented arm still in the training queue; its banked baseline leg is 0.3935

**Found**: 6 (claim, evidence) pairs carry BOTH labels - C1's structural test fires. 2 evidence units and 2 claims (3 of 38,126 rows, 0.008%) sit inside the mix and are visible ONLY under whitespace normalisation, which the banked builder's raw-string filter could not see. The page document channel is clean at 0 of 5,553

### 3. `findver` - tier 1, carries a LIVE or STANDING gate

**Reads it**: the BANKED standing non-arena mechanism instrument for derivation-adjacent arms - R20-H176's CONFIRMED verdict (numeric 2-draw mean 0.4959 against ie 0.6609 and knowledge 0.5838) is read on it

**Found**: nothing. Zero on every string form in both directions against the mix, the arena and gold_full, and zero on the document channel

### 4. `eval_C` - tier 1, carries a LIVE or STANDING gate

**Reads it**: the R20-H177 Lane C PRIMARY gate (>= 0.80); the baseline leg read 0.9085 and REFUTED the arm's near-chance prediction, and the lane's disposition is open

**Found**: nothing. Zero on every string form and every surface, zero documents, C1 structural 0 with the relational instrument separating 0.967 against 0.033, and every C5 bar clear

### 5. `r15_typeprobe` - tier 2, carried a gate that has since been retired; banked verdicts stand on the record

**Reads it**: the derivation half of the probe bank; gating DECLARED DEAD by R19-H163

**Found**: C1's structural test fires on 6 pairs; 54 of 1,136 document stems and 21 of 16,507 claims inside the mix

### 6. `r15_bindprobe` - tier 2, carried a gate that has since been retired; banked verdicts stand on the record

**Reads it**: was the PRIMARY of R17-H146 (`bind_col >= 0.80 AND bind_row >= 0.95`); probe-bank gating DECLARED DEAD by R19-H163

**Found**: 41 of 804 document stems and 28 of 3,852 claims inside the mix, 3 evidence tables byte-identical to a mix passage; the claim-only probe reads 0.5508 against C5's < 0.55

### 7. `h150_unitswap_probe` - tier 2, carried a gate that has since been retired; banked verdicts stand on the record

**Reads it**: the R18-H150 scale/unit probe, pinned in its manifest as a reported secondary that never trains and is never selected on; probe gating was stopped campaign-wide by R19-H163

**Found**: 27 of its 42 documents (196 of 280 rows, 70%) are inside the mix on the RAW document key - 23 through the `tabfact` member and 17 through the `quant_misbind` lane - against its registration as a `document-disjoint probe from unused supply`, which held only against the scale/unit lane it was built beside. Its text channel is clean. The claim-only probe reads 0.1425, a 0.3575 inverted deviation from chance

### 8. `h148_itemindex_probe` - tier 2, carried a gate that has since been retired; banked verdicts stand on the record

**Reads it**: the H148 item-index probe, still cited as the `literal-presence` build check for the H150 unit-swap probe; probe gating dead

**Found**: nothing on any channel. C1 separates 0.9902 against 0.0 on the 10.4% of rows the item-index instrument can read

### 9. `h149_roleswap_probe` - tier 2, carried a gate that has since been retired; banked verdicts stand on the record

**Reads it**: the H149 role-swap probe; probe gating dead

**Found**: nothing on any channel; every C5 bar clear. C1's test 2 is NOT COMPUTABLE on CPU for a lexical role swap and is reported as such

### 10. `antigaming_traced` - tier 3, no live gate

**Reads it**: the traced anti-gaming diagnostic, read at R14-H133 and after; never a promotion clause

**Found**: 47 of 979 document stems inside the mix; the claim-only probe reads 0.6999 and within-pair 0.8756, and surface parity fails at 0.2784 on the EXECUTOR-ADDED `claim_numeral_count` channel - the trace prefix prints the asserted figure, so the negative leg carries a different numeral count

### 11. `g0b_composed_probes` - tier 3, no live gate

**Reads it**: gate 5 of the hotpotqa composition fanout - baseline 0.6477 against a KILL at >= 0.70, PASS, and already moot for registration because gate G0a failed

**Found**: 28 of 564 document stems and 5 of 500 evidence units inside the mix; the claim-only probe reads 0.6867 and within-pair 0.908

### 12. `h117_heldout_pairs` - tier 3, no live gate

**Reads it**: kill-gate 2 of the R11-H117 paired-margin arm, closed at PROCEED with lambda_margin 0.3; the DR lane it served never entered the flagship mix

**Found**: 306 of 1,933 evidence passages (620 of 4,000 rows, 15.5%) and 286 of 4,000 claims are inside the mix, every one of them supplied by the `tabfact` member; the claim-only probe reads 0.6711 and within-pair 0.8195

### 13. `h175b_eval_clean` - tier 3, no live gate

**Reads it**: no live gate - R20-H175b is WITHDRAWN; the eval is retained for a possible future option-D registration with a parallel relevance head

**Found**: C1's structural test fires on 44 of 44 pairs (100%) - the same signature the withdrawn qlane carries, and by the same construction, since both legs share claim and evidence and only the question differs. C2 is clean on every channel including the PsiloQA page key

### 14. `h175b_eval_clean_prefix` - tier 3, no live gate

**Reads it**: no live gate - same standing as the eval above

**Found**: C1's structural test fires on 16 of 16 pairs (100%); C2 clean everywhere

### 15. `dr_h113_gate_judged` - tier 3, no live gate

**Reads it**: the DR lane's generator quality gate, closed in round 11; the DR lane never entered the flagship mix and no open arm reads the file

**Found**: 159 of its 1,446 evidence chunks are inside the mix and 0 of its 1,505 claims are. Verified because it carries claim and evidence text, not because it is an evaluation surface - its label is a judge verdict on a generated rewrite, and its `chunk` column carries source-code fragments

### 16. `r12_h121_gateBC_rows` - tier 3, no live gate

**Reads it**: Gate B of R12-H121, which KILLED the hypothesis pre-build at purity 0.284 against a 0.95 bar

**Found**: 2,476 of 2,476 claims and 1,059 of 3,346 windows are inside the mix - expected by construction rather than a contamination finding, since the killed hypothesis was to build distractor-window negatives out of the mix's own documents. No label column, so C1 / C5 / C6 are NOT-APPLICABLE; C2 and C7 are run so the enumeration leaves nothing unmeasured

## Live positive controls on the disjointness instrument

| control | design | result | fires |
|---|---|---|---|
| 1 synthetic identity | 251 mix passages offered to the gate as an eval | 251 of 251 read | yes |
| 2 synthetic re-wrap | the same passages with every space replaced by a newline plus indent | raw 1 of 251, normalised 251 of 251 | yes - only the normalised form sees it |
| 2b synthetic document stem | 300 member table ids with the `1-`/`2-` prefix flipped | raw 0, stem 300 | yes - only the stem channel sees it |
| 3 live banked, string | `R20-H175b_qlane_eval.parquet`, banked at 485 of 487 passages in the mix | 485 of 487 | yes - reproduces the banked figure exactly, including its 449-raw / 485-truncated split |
| 4 live banked, document | the original `R20-H177_eval_B.parquet`, banked at 325 of 325 TabFact document stems in the member | 325 of 325 | yes |

## The anti-gaming instrument is not one instrument

- **Near-miss half** - Jaccard 1.0 against the flagship's set on every one of the 14 banked arm files - the HEADLINE near-miss read is one instrument
- **bind_row half** - Jaccard 0.0017-0.0127 - a different 600-pair set on every arm
- **Cause** - the function ends in `unique(subset=['table_id'], keep='first')`. Polars does not preserve row order through `unique` unless `maintain_order=True` is passed, so the seeded `rng.permutation(len(tbls))` in `R14-H133_antigaming.build_bindrow` indexes a differently ordered list on each run and selects different tables. The seed is constant; the SUPPLY ORDER is not
- Membership identical across repeated builds: True; order identical: False

## Artifacts

- `experiments/grounding-semantic/contract/mechanism_evals_report.json`
- `experiments/grounding-semantic/contract/mechanism_evals_summary.md`
- `experiments/grounding-semantic/contract/mechanism_evals_antigaming_supp.json`
- `experiments/grounding-semantic/contract/mechanism_evals_spotchecks.json` - the four strongest findings re-derived without the digest machinery
- builders `mechanism_evals_verify.py`, `mechanism_evals_antigaming_supp.py`, `mechanism_evals_spotchecks.py`, `mechanism_evals_rank.py`
- logs `logs/contract-mechanism-evals.log`, `logs/contract-mechanism-evals-antigaming.log`

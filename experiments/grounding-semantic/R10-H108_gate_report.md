# R10-H108 gate report - licences, identity, contamination (zero-GPU stage)

Pre-registered gates run before any GPU spend. Build artifact: `R10-H108_pairs.parquet`
(61,184 pairs, 4 DANN groups), builder `R10-H108_data.py`, log `logs/R10-H108_data.log`.

## Gate verdicts

| corpus | licence | gate | note |
|---|---|---|---|
| FEVEROUS | CC-BY-SA (Wikipedia substrate; shared-task release) | PASS | official `fever/feverous` HF repo is a loading script requiring the ~30GB wiki DB; used the materialized-evidence mirror `KingTechnician/feverous-label-evidence` whose card documents the label mapping (0=SUPPORTS, 1=REFUTES, 2=NEI) |
| InfoTabS | Apache-2.0 (`github.com/infotabs/infotabs`, LICENSE verified via GitHub API) | PASS | datasheet carries a "read for non-academic usage" note - recorded; data taken from HF mirror `table-benchmark/infotabs` (train split only) |
| SciTab | MIT (`XinyuanLu00/SciTab`, LICENSE verified) | PASS | fetched `dataset/sci_tab.json` (1,224 claims) from the official repo |
| SEM-TAB-FACTS | n/a | **DROPPED at gate** | no programmatic distribution (Google Drive/IBM site only, no HF/GitHub data mirror found); contribution was ~4.5k of the planned mix - SciTab keeps the science-table register present |

## Contamination assertions (the round's wall)

- No ConvFinQA, TAT-HQA, MultiHiertt, FinanceBench, FinTabNet, SEC/EDGAR material anywhere in the build
- Substrates: Wikipedia tables/pages (FEVEROUS - same substrate class as the already-approved TabFact and VitaminC), Wikipedia infoboxes (InfoTabS), open-access science-paper tables (SciTab)
- FinQA (S&P earnings pages / FinTabNet lineage) and TAT-QA (annual reports) share zero documents with these sources; only the quantitative REGISTER is copied, which round 10 rules legal
- Corruption negatives are generated exclusively from TabFact / FEVEROUS / InfoTabS positives already gated above - no new corpus enters through them

## Counts per group/label

| group | label 1 | label 0 | total |
|---|---|---|---|
| quant_feverous | 7,226 | 3,143 | 10,369 |
| quant_infotabs | 5,471 | 10,995 | 16,466 |
| quant_scitab | 431 | 742 | 1,173 |
| quant_corrupt | 0 | 33,176 | 33,176 |
| **total** | 13,128 | 48,056 | **61,184** |

Corruption family histogram (6 families, per-family cap 11,250): digit_perturb 11,250,
magnitude_shift 11,250, year_shift 7,496, comparative_flip 2,457, pct_pp 469, scale_word 254.
Source histogram: tabfact 26,311 (capped at 60%), feverous 5,123, infotabs 1,742.
Value-collision hard filter active on digit_perturb / year_shift / magnitude_shift
(corruptions whose new value appears in the evidence are dropped, not relabelled).

## Deviations vs the registration (recorded, bars unaffected)

- **Total 61,184 vs ~118k planned.** Three causes: (1) FEVEROUS table/numeric selection is
  proxied by the challenge tags `Combining Tables and Text` + `Numerical Reasoning` because
  the materialized mirror flattens evidence (cell-membership is not observable without the
  30GB wiki DB); with the ≤1,500-char localisability rule this keeps 10.4k, not ~45k;
  (2) SEM-TAB-FACTS dropped at gate; (3) corruption generation exhausted eligible numeric
  positives at 33.2k of the 45k target (pct_pp / scale_word patterns are rare in
  Wikipedia-register claims). The bar and kill are outcome-based (finqa/tatqa/mean numbers),
  not size-based - the hypothesis adjudicates unchanged, with the smaller mass noted
- **InfoTabS label skew** - contradict AND neutral both map to 0 per registration, giving
  10,995 negatives vs 5,471 positives in that group
- Training remains HELD - no trainer built or launched at this stage

## QA observation

10 samples per (group, label) printed in `logs/R10-H108_data.log`. Corruptions read as
plausible near-misses (season "1978-79" shifted to "1980-79" is the least natural template
output observed; the value filter and family balance otherwise held). FEVEROUS evidence
retains sentence+cell mixed content with wiki-link markup stripped.

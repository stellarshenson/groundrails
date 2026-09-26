# Fresh test of the round-2 defect designs

Pre-registration, written 2026-09-26 before any row of the test corpora was read. Round 1 (`HELDOUT.md`) reverted four rules; this round redesigns them. Every corpus read so far is now design data: RAGTruth, RAGBench, VitaminC dev, MiniCheck, HaluEval, PsiloQA and lettucedetect-prose (PsiloQA plus RAGTruth). Designs are chosen on design data only; the test corpora below are read once, by the script, after the code is frozen.

## Frozen code

- Base commit `65451f1`; working-tree `git diff HEAD -- src | sha256sum` prefix `8a1b776d8f49a535`, recorded after the designs were settled on design data and before the test run
- Design note, written before the test run: on design data the DEF-NUMBER-46 rule removed 46 HaluEval-summarization contradictions, 32 from hallucinated summaries (0.70 against 0.53), and passed on VitaminC (1,130 removed, 0.41 against 0.47). HaluEval labels mark a whole summary, so a correct claim inside a hallucinated summary counts as labelled; the rule goes to the test because the test corpus labels single claims

## Test corpora

- `osunlp/AttributionBench`, all files deduplicated by `id`, without the `ExpertQA` and `HAGRID` rows (both are in RAGBench); numeric pairs are claims holding a digit against their joined `references`; labelled = `not attributable`; responses are also English answer documents
- `ucinlp/drop` passages, deduplicated: English prose for extraction and the language warning
- `fava-uw/fava-data` `annotations.json` outputs with their markup removed: English answer documents
- `KRLabsOrg/ragtruth-{pl,de,es,fr,it,hu,cn}-translated` train splits, the first 2,000 answers per language: non-English documents (round 1 read 500 test answers per language, never the train splits)

## Rules under test and their switch

| Rule | Defect | Switched off by |
|---|---|---|
| rounding within one quantity stays within 5% | DEF-NUMBER-32 | `_same_quantity_agrees` is the committed `_agree_at_coarser_precision` |
| a year span compares the same in either order | DEF-NUMBER-44 | `_year_span_key` returns its input |
| a currency sign may stand between a comparative and its number | DEF-NUMBER-46 | the committed `_COMPARATIVE_RE` |
| a Latin read with English confidence of 0.2 or more among spoken languages is English | DEF-CLAIM-37 | `_LATIN_ENGLISH_MIN` above 1 |
| rejoin at a dotted, listed or initial abbreviation | DEF-CLAIM-39, DEF-CLAIM-40 | `_ABBREV_SHAPE_RE` never matches |
| an opening parenthesis may start a sentence, except after an abbreviation-shaped token | DEF-CLAIM-41 | judged only |
| a lettered or roman outline item starts its own claim | DEF-CLAIM-45 | judged only |

## Pass criteria

- **Numeric rules** - as in `HELDOUT.md`: on every test corpus with at least 10 added contradictions, their labelled share is at least the committed check's labelled share `P` there; with at least 10 removed, their labelled share is at most `P`; under 10 on every corpus is inconclusive and the rule stays, marked so
- **Latin rule** - no non-English test document that warned with the committed code stops warning; the count of English test documents that stop falsely warning is reported
- **Judged rules** - for each, a seeded sample of 20 changed units on the English test documents, printed in full for audit, holds at least 16 correct ones: a rejoin where the rejoined side belongs to the same sentence; a parenthesis split where both sides are whole sentences; an outline split where the line is an outline item
- A rule that fails is reverted and its defect stays open

The run is `HF_HUB_OFFLINE=1 .venv/bin/python experiments/defects/measure_heldout2.py`, output `measure_heldout2.json`.

## Verdicts

Written after the test run, from `measure_heldout2.json`. Nothing was changed after it.

| Rule | Verdict | Deciding numbers | Action |
|---|---|---|---|
| Latin read re-checked among spoken languages | PASS | 0 of the 13,997 warned non-English answers (of 14,000) stop warning; English false warnings 65 to 47 of 24,974 | kept |
| rejoin at a dotted, listed or initial abbreviation | PASS | judged 20 of 20 (499 changed units) | kept |
| opening parenthesis starts a sentence | PASS | judged 20 of 20 (81 changed units) | kept |
| outline items | INCONCLUSIVE | 0 changed units in the test documents; design data judged 20 of 20 | kept, marked |
| rounding within one quantity | INCONCLUSIVE | under 10 changed pairs on every corpus | kept, marked |
| year span in either order | INCONCLUSIVE | no changed pair; design data removed 41 finance contradictions, 1 labelled | kept, marked |
| currency comparative | INCONCLUSIVE | 4 removed pairs in all | kept, marked |

Two judged parenthesis units hold a bracketed note and the sentence after it: a boundary after `.")` needs two closing characters and the splitter accepts one. That predates this round.

## Reduction after review

Adversarial review round 1 found regressions in the kept rules on inputs the test corpora did not hold ("Ph.D. (Stanford)" cut in two, "E. coli" read as an outline label, "2019-20" against "2019-2020" contradicted, "About 20-30%" against "22-31%" contradicted). The author then ruled that the deterministic tier catches obvious discrepancies only and that a rule without out-of-sample support does not ship. Every rule in the table above was reverted; what remains from both rounds is the date bound, the crash guard, the removal of the modal-conditional rule, and three fixes that follow a stated rule (comments outside code as in CommonMark, nested blockquote bullets, citation markers kept with their sentence). The code this file tested is `heldout2-frozen-src.patch` (applies to `65451f1`; its `sha256sum` prefix is the frozen hash above).

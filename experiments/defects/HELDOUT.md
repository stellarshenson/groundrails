# Held-out check of the open-defect fixes

Pre-registration, written 2026-09-26 before any held-out row was read. Several rules in the fix set were adjusted after reading samples of RAGTruth and RAGBench, so the numbers in `measure_open_defects.json` are in-sample. This check runs the frozen code once on corpora not read while the rules were written, and decides by the criteria below. A rule that fails is reverted.

## Frozen code

- Base commit `65451f1`; working-tree `git diff HEAD -- src | sha256sum` prefix `621a06d8cae68b33`
- `src/` is not edited between this file and the verdicts
- The code at this hash is not kept. `measure_heldout.py` switches symbols that were reverted later, and its `main()` stops when the `src/` hash differs from `621a06d8cae68b33`, so none of these verdicts can be re-run as written on the current tree

## Corpora

- `tals/vitaminc` dev, 63,054 claim-evidence pairs minus the 45 items of `TestVitaminCComposite`; labelled = `REFUTES`
- `lytang/C2D-and-D2C-MiniCheck`, 14,395 claim-document pairs; labelled = `label == 0`
- `pminervini/HaluEval` qa, summarization, dialogue: each right and hallucinated output against its knowledge or document; labelled = the hallucinated output; `general` (4,507 ChatGPT answers) for extraction only
- `s-nlp/PsiloQA` test, 2,897 answers in several languages; labelled = a claim overlapping an annotated span

Numeric pairs are the claims holding a digit, from `extract_claims` on each output, against the whole source text.

## Rules under test and their switch

| Rule | Defect | Switched off by |
|---|---|---|
| a date after before/after is a bound | DEF-NUMBER-42 | `_DATE_BOUND_RE` never matches |
| rounding agrees only within 5% | DEF-NUMBER-32 | committed `_agree_at_coarser_precision`, crash guard kept |
| ranges compared whole | DEF-NUMBER-34 | `_merged_numbers` returns nothing inside `find_numeric_mismatches` |
| year spans left out of the whole-range comparison | DEF-NUMBER-34 | `_YEAR_SPAN_RE` never matches |
| glued citation marker is not a stated value | DEF-NUMBER-33 | `_GLUED_CITATION_RE` never matches |
| verbless side rejoined only at an abbreviation-shaped boundary | DEF-CLAIM-39, DEF-CLAIM-40 | `_ABBREV_SHAPE_RE` matches every boundary (ungated) or none (no rejoin) |
| Latin never read as the language | DEF-CLAIM-37 | committed confident read |
| modal-conditional hypothetical rule removed | DEF-CLAIM-36 | committed `_COND_MODAL_RE` |

Not under test, because each follows the defect's own reported shape: the crash guard (DEF-NUMBER-43), comments stripped only outside code (DEF-CLAIM-35), the nested-blockquote strip (DEF-CLAIM-38), the bracket and citation-marker boundaries (DEF-CLAIM-41), the spaced-dash and sign rules (DEF-NUMBER-34).

## Pass criteria

A numeric rule's effect is the working tree against the working tree with that one rule switched off. `P` is the labelled share of the contradictions the committed code reports on the same corpus.

- **Gains** - contradictions the rule adds: on every corpus where they number at least 10, their labelled share is at least `P`
- **Losses** - contradictions the rule removes: on every corpus where they number at least 10, their labelled share is at most `P`
- A rule failing either clause on any corpus fails; a rule under 10 in both on every corpus is inconclusive and stays, marked so in its defect evidence

The rejoin gate passes when a seeded sample of 20 gated rejoins on held-out answers holds at least 16 where the rejoined side belongs to the same sentence. The sample is printed in full for audit.

The Latin rule passes when no English held-out document that the committed code read as Latin warns after the change, in any language, and no non-English PsiloQA answer changes its warning language.

The modal-conditional removal stands when the committed rule fires on held-out source sentences at no less than half its rate on held-out answer sentences - a rule that cannot tell sources from answers cannot mark authored hypotheticals.

The run is `HF_HUB_OFFLINE=1 .venv/bin/python experiments/defects/measure_heldout.py`, output `measure_heldout.json`.

## Verdicts

Written after the run, from `measure_heldout.json`. `P` is the committed check's labelled share on VitaminC, 0.467 (3,723 contradictions).

| Rule | Verdict | Deciding numbers | Action |
|---|---|---|---|
| a date after before/after is a bound | PASS | VitaminC removes 1,122 contradictions, 342 labelled (0.305 against `P`) | kept |
| ranges compared whole | PASS | VitaminC adds 32, 26 labelled (0.81); HaluEval summarization adds 14, 10 labelled (0.71 against 0.532) | kept |
| modal-conditional rule removed | PASS | fires on 233 of 346,742 source sentences (0.067%) and 33 of 59,721 answer sentences (0.055%) | removal kept |
| glued citation marker | INCONCLUSIVE | under 10 changed pairs on every corpus | kept, marked |
| rounding within 5% | FAIL | VitaminC adds 10, 2 labelled (0.20) | reverted |
| year spans left out | FAIL | removes mostly real conflicts: PsiloQA 35 of 37, VitaminC 15 of 17 | reverted |
| Latin never read | FAIL | all 23 English Latin reads stop warning, but 11 non-English PsiloQA answers (sv, es, it, eu) read as Latin stop warning too | reverted |
| rejoin gate | FAIL | judged sample: 14 of 20 repairs; outline labels "A." to "D." (4), "Yes." (1) and "The U.S." joined to the wrong sentence (1) are not | reverted, with the opening-bracket boundary it protected |

The committed code raised `decimal.InvalidOperation` on no held-out pair in this run; set order decides whether it fires (DEF-NUMBER-43).

The whole-range comparison kept here was reverted later, after review found it contradicting ranges that agree at a coarser precision, and the glued citation marker was reverted with every rule that lacked out-of-sample support; see `HELDOUT2.md`, Reduction after review.

The date bound was measured with a month pattern that matched any word starting with a month prefix (`declining`, `marking`). Review round 9 found that it read `after declining 12%` as a date; the shipped pattern accepts month names only.

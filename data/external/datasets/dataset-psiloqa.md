# PsiloQA

Second-largest multilingual source, and its error profile - wrong number, wrong entity - matches the residual our own numeric and entity checks target.

- **HuggingFace** - `s-nlp/PsiloQA`
- **Licence** - CC-BY-4.0
- **Size** - 63,792 train / 3,355 val / 2,897 test
- **Languages** - 14 - en de fr es it ca eu sv fi cs ar fa hi zh
- **How negatives were made** - Naturally occurring - diverse LLMs answer without context, then the answer is compared against retrieved Wikipedia; wrong entity, date or number dominates
- **How labels were made** - GPT-4o end to end (QA generation and span marking), no human verification
- **Mapping onto our task** - answer -> claim; retrieved Wikipedia passage -> evidence

## Caveats

Entirely LLM-annotated, so label noise is real and unmeasured. Wikipedia register.

## Provenance

Selected in the round 7 dataset survey, `reports/research-grounding-datasets.md`. Every corpus
in this directory passed three filters together: a licence permitting commercial use, source
documents shipping alongside the claims, and a task shape that maps onto (claim, evidence) →
supported. Corpora excluded on licence alone: TrueTeacher (CC-BY-NC, 1.38M), HaluBench
(CC-BY-NC), MS MARCO (non-commercial), ANLI (CC-BY-NC), MEMERAG (card forbids training).

Fetched by `scripts/fetch_grounding_datasets.py`. The archive `dataset-psiloqa.zip` is
gitignored; this sidecar is tracked.

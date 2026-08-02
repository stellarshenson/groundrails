# TabFact

The only permissively-licensed corpus whose claims require reading numbers out of structured tables - registered for R8-H87 after the blind-arena residual concentrated in the tabular subsets (finqa -0.1373, tatqa -0.0169) and the R8-H85 probe ruled out context truncation as the cause.

- **HuggingFace** - `wenhu/tab_fact (script-only; fetched from GitHub instead)`
- **Licence** - CC-BY-4.0
- **Size** - ~118k statements over 16k Wikipedia tables - 92,283 train / 12,792 val / 12,779 test
- **Languages** - English
- **How negatives were made** - Counterfactual statements written by annotators against the SAME table that entails the positives - near-miss by construction, in the numeric-tabular register no other corpus in this directory covers
- **How labels were made** - Human annotation (ICLR 2020), binary ENTAILED / REFUTED
- **Mapping onto our task** - statement -> claim; caption + linearized table -> evidence; ENTAILED -> 1, REFUTED -> 0

## Caveats

Wikipedia tables, not financial filings - register coverage, not domain coverage. Tables arrive linearized (`#`-separated header and rows) and must be serialized into evidence text; statements are short and single-fact.

## Provenance

Selected in the round 7 dataset survey, `reports/research-grounding-datasets.md`. Every corpus
in this directory passed three filters together: a licence permitting commercial use, source
documents shipping alongside the claims, and a task shape that maps onto (claim, evidence) →
supported. Corpora excluded on licence alone: TrueTeacher (CC-BY-NC, 1.38M), HaluBench
(CC-BY-NC), MS MARCO (non-commercial), ANLI (CC-BY-NC), MEMERAG (card forbids training).

Fetched by `scripts/fetch_grounding_datasets.py`. The archive `dataset-tabfact.zip` is
gitignored; this sidecar is tracked.

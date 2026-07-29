# RAGTruth, machine-translated into 7 languages

The cheapest licence-clean route to non-English supervision, and our non-English slices are thin by construction - 21 languages spread over 639 traces.

- **HuggingFace** - `KRLabsOrg/ragtruth-de-translated`, `KRLabsOrg/ragtruth-fr-translated`, `KRLabsOrg/ragtruth-es-translated`, `KRLabsOrg/ragtruth-it-translated`, `KRLabsOrg/ragtruth-pl-translated`, `KRLabsOrg/ragtruth-hu-translated`, `KRLabsOrg/ragtruth-cn-translated`
- **Licence** - MIT
- **Size** - 17,790 each, ~106k train across the seven
- **Languages** - de, fr, es, it, pl, hu, zh
- **How negatives were made** - Inherited from RAGTruth, translated with context
- **How labels were made** - Inherited human spans, re-aligned after translation (Gemma 3 27B via vLLM)
- **Mapping onto our task** - as RAGTruth

## Caveats

Machine-translated. Only 300 German rows are human-verified (`KRLabsOrg/ragtruth-de-translated-manual-300`), so non-English label alignment is inherited rather than checked. Treat as weaker supervision than the English original.

## Provenance

Selected in the round 7 dataset survey, `reports/research-grounding-datasets.md`. Every corpus
in this directory passed three filters together: a licence permitting commercial use, source
documents shipping alongside the claims, and a task shape that maps onto (claim, evidence) →
supported. Corpora excluded on licence alone: TrueTeacher (CC-BY-NC, 1.38M), HaluBench
(CC-BY-NC), MS MARCO (non-commercial), ANLI (CC-BY-NC), MEMERAG (card forbids training).

Fetched by `scripts/fetch_grounding_datasets.py`. The archive `dataset-ragtruth-translated.zip` is
gitignored; this sidecar is tracked.

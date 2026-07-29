# HaluEval

Bulk English supervision with near-miss negatives, MIT-licensed.

- **HuggingFace** - `pminervini/HaluEval`
- **Licence** - MIT
- **Size** - ~35k - 5k real ChatGPT user queries plus 30k task-specific
- **Languages** - English
- **How negatives were made** - LLM-constructed near-miss - ChatGPT sampling-then-filtering selects the most plausible-yet-wrong candidate
- **How labels were made** - Human on the 5k real-query slice; LLM-constructed on the 30k
- **Mapping onto our task** - answer -> claim; knowledge field -> evidence

## Caveats

Only the 5k real-query slice carries human labels. Widely used for EVALUATION, so holding it out of training preserves it as a public comparison point.

## Provenance

Selected in the round 7 dataset survey, `reports/research-grounding-datasets.md`. Every corpus
in this directory passed three filters together: a licence permitting commercial use, source
documents shipping alongside the claims, and a task shape that maps onto (claim, evidence) →
supported. Corpora excluded on licence alone: TrueTeacher (CC-BY-NC, 1.38M), HaluBench
(CC-BY-NC), MS MARCO (non-commercial), ANLI (CC-BY-NC), MEMERAG (card forbids training).

Fetched by `scripts/fetch_grounding_datasets.py`. The archive `dataset-halueval.zip` is
gitignored; this sidecar is tracked.

## Subsets fetched

- `qa`
- `summarization`
- `dialogue`
- `general`

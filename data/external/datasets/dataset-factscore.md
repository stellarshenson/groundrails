# FActScore labeled biographies - human-judged atomic facts

Human-judged atomic facts over long-form biographies - the highest-precision long-form supervision the recon found; a high-precision slice or held-out probe, not bulk supervision.

- **Source** - github.com/shmsw25/FActScore (code; MIT LICENSE in repo) + the authors' Google Drive `data.zip` (`data/labeled/*.jsonl`); Wikipedia evidence pulled per topic from the Wikipedia API, pinned to the last revision on or before 2023-04-01 (the corpus's enwiki-20230401 knowledge source)
- **Licence** - MIT (repo LICENSE file, re-read at fetch 2026-08-13)
- **Size** - 549 labeled biographies - 183 entities x 3 models (InstructGPT, ChatGPT, PerplexityAI); order-10k human atomic-fact judgments; fetched 2026-08-13: 732 rows (InstructGPT: 183, ChatGPT: 183, PerplexityAI: 183, entities: 183)
- **Languages** - English
- **How negatives were made** - Naturally occurring unsupported atomic facts inside LM-generated biographies; no perturbation
- **How labels were made** - Human per-atomic-fact S (supported) / NS (not supported) / IR (off-topic sentence) judgments against Wikipedia
- **Mapping onto our task** - atomic fact → claim; topic's pinned Wikipedia article → evidence; S → 1, NS → 0

## Caveats

Small by design. Evidence is the topic's full Wikipedia article at the pinned 2023-04-01 revision fetched through the public API (the authors' 28GB enwiki-20230401.db is not pulled); HTML flattening to text is ours. IR facts are excluded from the lane (off-topic, support undefined) and counted in the gate JSON.

## Provenance

Selected in the 2026-08-13 recon re-survey (`reports/research-grounding-datasets.md`, "Re-survey
2026-08-13" - licence verified at source there and RE-VERIFIED at pull time in this build; the
licence line above is the tag read from the source pulled, not the recon's say-so). Registered in
`docs/experiments/semantic-dataset-enhancements.md`, section "R19 supply wave" (2026-08-13): SUPPLY ONLY -
nothing enters a training mix without its own registered hypothesis and arm. The contamination
gate (R14-H136 8-gram Jaccard instrument against the ten walled arena corpora, bar 0.02 max
fraction + spike control) runs after this fetch and its verdict is recorded in
`experiments/grounding-semantic/R19_factscore_gate.json`; the pair-formatted lane, manifest and verify
JSON land beside it as `R19_factscore_lane.parquet` / `_manifest.json` / `_verify.json`.

Fetched by `scripts/fetch_grounding_datasets.py factscore`. The downloaded data under
`data/external/datasets/factscore/` is gitignored; this sidecar is tracked.

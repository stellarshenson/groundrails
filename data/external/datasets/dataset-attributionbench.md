# AttributionBench (full_data config) - with the ExpertQA/HAGRID carve-out

Large multi-source attribution supervision in the answer-with-references shape - the closest public analogue to production RAG grounding.

- **HuggingFace** - `osunlp/AttributionBench`
- **Licence** - Apache-2.0 (HF card YAML `license: apache-2.0`, re-verified at the Hub 2026-08-13)
- **Size** - 26,365 rows in the card's `full_data` config (train/dev/test/test_ood) before the carve-out; fetched 2026-08-13: 16524 rows (train: 14200, dev: 728, test: 998, test_ood: 598)
- **Languages** - English
- **How negatives were made** - Not-attributable claims as labeled by the source benchmarks (Stanford-GenSearch, AttributedQA, LFQA, BEGIN, AttrEval) - a mix of human and benchmark-pipeline labels unified by the authors
- **How labels were made** - Source-benchmark attribution labels, unified to attributable / not attributable
- **Mapping onto our task** - claim → claim; references → evidence; attributable → 1, not attributable → 0

## Caveats

**Carve-out, applied at fetch**: every row whose `src_dataset` is ExpertQA or HAGRID (both walled RAGBench source corpora) is dropped BY CONSTRUCTION - precedent: the RAGBench fetch's MS MARCO/CUAD exclusion. The gate re-derives zero walled rows from the kept frame rather than trusting this note. BEGIN and AttrScore-GenSearch exist only in the test_ood split; the kept frame spans all four splits with the split tag retained - a later hypothesis decides split hygiene.

## Provenance

Selected in the 2026-08-13 recon re-survey (`reports/research-grounding-datasets.md`, "Re-survey
2026-08-13" - licence verified at source there and RE-VERIFIED at pull time in this build; the
licence line above is the tag read from the source pulled, not the recon's say-so). Registered in
`docs/experiments/semantic-dataset-enhancements.md`, section "R19 supply wave" (2026-08-13): SUPPLY ONLY -
nothing enters a training mix without its own registered hypothesis and arm. The contamination
gate (R14-H136 8-gram Jaccard instrument against the ten walled arena corpora, bar 0.02 max
fraction + spike control) runs after this fetch and its verdict is recorded in
`experiments/grounding-semantic/R19_attributionbench_gate.json`; the pair-formatted lane, manifest and verify
JSON land beside it as `R19_attributionbench_lane.parquet` / `_manifest.json` / `_verify.json`.

Fetched by `scripts/fetch_grounding_datasets.py`. The archive `dataset-attributionbench.zip` is
gitignored; this sidecar is tracked.

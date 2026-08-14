# FinDVer - claim verification over 2024 SEC filings

The only candidate filling the financial numeric-derivation gap directly - the flagship's sole losing register. Admitted under the EDGAR-population precedent: its documents are 2024 filings while the walled FinQA/TAT-QA corpora are pre-2020, so document overlap is structurally impossible; the 8-gram instrument still runs.

- **Source** - github.com/yilunzhao/FinDVer (MIT LICENSE in repo); the repo ships the claims (`data/test.json`, `data/testmini.json`) and the 2024 10-K/10-Q filing extracts (`financial_reports/*.json`, sec.gov EDGAR urls)
- **Licence** - MIT (repo LICENSE file, re-read at fetch 2026-08-13)
- **Size** - 2,400 claims - test 1,700 + testmini 700 - over 2024 10-K/10-Q filings, subsets ie / numeric / knowledge; fetched 2026-08-13: 2400 rows (test: 1700, testmini: 700)
- **Languages** - English
- **How negatives were made** - Human-annotated refuted claims over the filings with step-by-step explanations (GPT-4o-proofread)
- **How labels were made** - Human entailment judgments (entailed/refuted)
- **Mapping onto our task** - statement → claim; `relevant_context` passages → evidence; entailed → 1, refuted → 0

## Caveats

Claims are benchmark items (test/testmini splits of a public benchmark) banked as supply; a later hypothesis decides any training use. Evidence is the annotated `relevant_context` passages, not the whole filing; subset tags retained.

## Provenance

Selected in the 2026-08-13 recon re-survey (`reports/research-grounding-datasets.md`, "Re-survey
2026-08-13" - licence verified at source there and RE-VERIFIED at pull time in this build; the
licence line above is the tag read from the source pulled, not the recon's say-so). Registered in
`docs/experiments/semantic-dataset-enhancements.md`, section "R19 supply wave" (2026-08-13): SUPPLY ONLY -
nothing enters a training mix without its own registered hypothesis and arm. The contamination
gate (R14-H136 8-gram Jaccard instrument against the ten walled arena corpora, bar 0.02 max
fraction + spike control) runs after this fetch and its verdict is recorded in
`experiments/grounding-semantic/R19_findver_gate.json`; the pair-formatted lane, manifest and verify
JSON land beside it as `R19_findver_lane.parquet` / `_manifest.json` / `_verify.json`.

Fetched by `scripts/fetch_grounding_datasets.py findver`. The downloaded data under
`data/external/datasets/findver/` is gitignored; this sidecar is tracked.

# PubHealth (health_fact) - real-world public-health fact-checks

Real-world fact-check register - journalism over genuine public claims - that nothing else in the directory covers, with human labels end to end.

- **HuggingFace** - `ImperialCollegeLondon/health_fact`
- **Licence** - MIT (HF card YAML `license: mit`, re-verified at the Hub 2026-08-13; the loader script carries an Apache-2.0 code header - data licence per the card)
- **Size** - 12,288 rows (train/dev/test TSVs); fetched 2026-08-13: 12288 rows (train: 9832, dev: 1221, test: 1235)
- **Languages** - English
- **How negatives were made** - Naturally occurring false, unproven and mixed health claims as published and fact-checked by Snopes/PolitiFact/Full Fact lineage sites - no perturbation
- **How labels were made** - Journalist verdicts (4-way true/false/unproven/mixture) with written explanations; human throughout
- **Mapping onto our task** - claim → claim; source article `main_text` → evidence; true → 1, false/unproven/mixture → 0

## Caveats

The HF repo ships only a loading script (datasets 5.x removed script support); the data is pulled from the authors' Google Drive zip the script names. Rows with empty `main_text` or stray labels are excluded from the lane (rates reported in the gate JSON) but retained in the archive. Verdict maps onto SUPPORT, not truth: unproven/mixture read as not-supported (registered coordinator ruling).

## Provenance

Selected in the 2026-08-13 recon re-survey (`reports/research-grounding-datasets.md`, "Re-survey
2026-08-13" - licence verified at source there and RE-VERIFIED at pull time in this build; the
licence line above is the tag read from the source pulled, not the recon's say-so). Registered in
`docs/experiments/semantic-dataset-enhancements.md`, section "R19 supply wave" (2026-08-13): SUPPLY ONLY -
nothing enters a training mix without its own registered hypothesis and arm. The contamination
gate (R14-H136 8-gram Jaccard instrument against the ten walled arena corpora, bar 0.02 max
fraction + spike control) runs after this fetch and its verdict is recorded in
`experiments/grounding-semantic/R19_pubhealth_gate.json`; the pair-formatted lane, manifest and verify
JSON land beside it as `R19_pubhealth_lane.parquet` / `_manifest.json` / `_verify.json`.

Fetched by `scripts/fetch_grounding_datasets.py`. The archive `dataset-pubhealth.zip` is
gitignored; this sidecar is tracked.

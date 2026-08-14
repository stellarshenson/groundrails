# MiniCheck C2D/D2C - multi-fact claim/document checking pairs

Multi-fact, multi-sentence checking at scale - the case where support must be assembled from several sentences, which single-sentence corpora do not teach.

- **HuggingFace** - `lytang/C2D-and-D2C-MiniCheck`
- **Licence** - MIT (HF card YAML `license: mit`, re-verified at the Hub 2026-08-13)
- **Size** - 14,395 (claim, doc, label) pairs - 7,076 C2D + 7,319 D2C; fetched 2026-08-13: 14395 rows (c2d: 7076, d2c: 7319)
- **Languages** - English
- **How negatives were made** - GPT-4-generated non-supporting documents against real claims, multi-fact and multi-sentence by construction; a 53% rejection rate at the entailment filter is reported by the authors
- **How labels were made** - Synthetic pipeline labels with GPT-4 entailment filtering at every construction step
- **Mapping onto our task** - claim → claim; doc → evidence; shipped label (1 supported / 0 not)

## Caveats

Seed corpora are named only in the paper's Appendix D (ACL 2024.emnlp-main.499): C2D seeds are ~400 Wikipedia claims with cited web articles (Kamoi et al. 2023; Petroni et al. 2023), D2C seeds are ~300 Google News articles scraped since November 2023. Neither is a walled RAGBench source corpus; the 8-gram instrument runs regardless and is the binding check.

## Provenance

Selected in the 2026-08-13 recon re-survey (`reports/research-grounding-datasets.md`, "Re-survey
2026-08-13" - licence verified at source there and RE-VERIFIED at pull time in this build; the
licence line above is the tag read from the source pulled, not the recon's say-so). Registered in
`docs/experiments/semantic-dataset-enhancements.md`, section "R19 supply wave" (2026-08-13): SUPPLY ONLY -
nothing enters a training mix without its own registered hypothesis and arm. The contamination
gate (R14-H136 8-gram Jaccard instrument against the ten walled arena corpora, bar 0.02 max
fraction + spike control) runs after this fetch and its verdict is recorded in
`experiments/grounding-semantic/R19_minicheck_gate.json`; the pair-formatted lane, manifest and verify
JSON land beside it as `R19_minicheck_lane.parquet` / `_manifest.json` / `_verify.json`.

Fetched by `scripts/fetch_grounding_datasets.py`. The archive `dataset-minicheck.zip` is
gitignored; this sidecar is tracked.

# FAVA fava-data - span-tagged long-form hallucination corpus

The only span-level long-form taxonomy set the 2026-08-13 recon found; the error taxonomy (six typed spans) is the fine-grained structure no other admitted corpus carries.

- **HuggingFace** - `fava-uw/fava-data`
- **Licence** - CC-BY-4.0 (HF card YAML `license: cc-by-4.0`, re-verified at the Hub 2026-08-13)
- **Size** - ~30k synthetic training rows plus 460 human-annotated gold passages (`annotations.json`, banked alongside, not pair-formatted); fetched 2026-08-13: 30533 rows (train: 30073, gold: 460)
- **Languages** - English
- **How negatives were made** - Synthetic error injection - the FAVA pipeline plants typed hallucination spans (entity, relation, contradictory, subjective, unverifiable, invented) into reference-grounded drafts and ships the draft plus its tagged edit
- **How labels were made** - Silver span tags from the generation pipeline; 460 gold passages carry human annotation
- **Mapping onto our task** - prompt `Text:` passage → claim; `Reference [i]` blocks → evidence; any error span in the tagged completion → 0, clean → 1; span metadata retained for future fine-grained use

## Caveats

Wikipedia/web register, not business documents. Silver labels, not human. The claim is the draft passage embedded in the prompt after `Text:` - the byte-exact pre-edit text the tagged completion edits; the tagged completion is retained as a provenance column.

## Provenance

Selected in the 2026-08-13 recon re-survey (`reports/research-grounding-datasets.md`, "Re-survey
2026-08-13" - licence verified at source there and RE-VERIFIED at pull time in this build; the
licence line above is the tag read from the source pulled, not the recon's say-so). Registered in
`docs/experiments/semantic-dataset-enhancements.md`, section "R19 supply wave" (2026-08-13): SUPPLY ONLY -
nothing enters a training mix without its own registered hypothesis and arm. The contamination
gate (R14-H136 8-gram Jaccard instrument against the ten walled arena corpora, bar 0.02 max
fraction + spike control) runs after this fetch and its verdict is recorded in
`experiments/grounding-semantic/R19_fava_gate.json`; the pair-formatted lane, manifest and verify
JSON land beside it as `R19_fava_lane.parquet` / `_manifest.json` / `_verify.json`.

Fetched by `scripts/fetch_grounding_datasets.py`. The archive `dataset-fava.zip` is
gitignored; this sidecar is tracked.

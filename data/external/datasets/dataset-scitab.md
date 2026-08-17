# SciTab - compositional claim verification over scientific tables

Claims are real result statements from papers - 'the models using BoC outperform models using BoW as well as ASM features' - whose verification needs a comparison across several table cells. Compositional reasoning over numbers is the corpus's stated design point, and it ships a NOT-ENOUGH-INFO leg the mix is thin on.

- **Source** - github.com/XinyuanLu00/SciTab (`dataset/sci_tab.json`); paper Lu et al., EMNLP 2023
- **Licence** - MIT (repo LICENSE, GitHub licence API `spdx_id: MIT`, re-read at fetch 2026-08-17)
- **Size** - 1,224 expert-verified claims - 457 supports / 411 refutes / 356 not enough info - over tables from arXiv papers; fetched 2026-08-17: 1224 rows (all: 1224)
- **Languages** - English
- **How negatives were made** - Counter-claims over the same table, expert-verified; refutation turns on the comparison or the computed quantity, not on a swapped entity
- **How labels were made** - Human, by the papers' own authors' statements plus expert verification
- **Mapping onto our task** - claim → claim; table_caption + linearized table_content_values → evidence; supports → 1, refutes/not-enough-info → 0

## Caveats

Small (1,224). Scientific-paper register. Three-way labels need collapsing to binary and the NEI leg needs a ruling (recommend NEI → 0). Table cells carry `[BOLD]` markup from the SciGen extraction that must be stripped at lane build. Published as an EVAL benchmark - a later hypothesis decides any training use.

## Provenance

Selected in the R22 derivation-supply survey of 2026-08-17, whose findings were returned to the
coordinator for `reports/research-grounding-datasets.md`.  The wave exists because the mix has
never contained a claim stating a value DERIVED from evidence values: every numeric member builds
its negatives by substituting an operand, so no row shows the model correct, present operands
and a false computed result - which is the dominant failure of the finqa arena subset.
Licence read AT SOURCE in the survey and RE-VERIFIED at pull time in this build; the licence
line above is the tag read from the source pulled, not the survey's say-so.  SUPPLY ONLY -
nothing here enters a training mix without its own registered hypothesis and arm.  The contamination gate (R14-H136 8-gram Jaccard instrument against
the ten walled arena corpora, bar 0.02 max fraction plus spike control) runs after this fetch via
`experiments/grounding-semantic/R22_supply_gates.py`, verdict recorded in
`experiments/grounding-semantic/R22_scitab_gate.json`.

Fetched by `scripts/fetch_grounding_datasets.py`. The archive `dataset-scitab.zip` is
gitignored; this sidecar is tracked.

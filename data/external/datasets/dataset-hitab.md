# HiTab - hierarchical statistical tables with aggregation operators and answer formulas

The only corpus found that ships the DERIVATION ITSELF machine-readably: each row carries the aggregation operator (`div`, `diff`, `sum`, `average`, `argmax`, `range`, ...), an Excel-style `answer_formulas` naming the operand cells, and the `sub_sentence` a human analyst actually wrote stating the derived value. That makes a result-only corruption constructible with the operands provably untouched.

- **Source** - github.com/microsoft/HiTab (`data/{train,dev,test}_samples.jsonl` plus `data/tables.zip`, 3,597 tables); paper Cheng et al., ACL 2022
- **Licence** - Computational Use of Data Agreement v1.0 (repo LICENSE, read at fetch 2026-08-17). C-UDA restricts the DATA to Computational Use - clause 5.1, 'activities necessary to enable the use of Data for analysis by a computer' - and clause 2.2 places NO restriction on Results, so a model trained on it is unencumbered. No non-commercial and no research-only clause
- **Size** - 10,672 QA rows over 3,597 hierarchical tables from StatCan, NSF and Wikipedia; fetched 2026-08-17: 17866 rows (train: 7417, dev: 1671, test: 1584, tables: 7194)
- **Languages** - English
- **How negatives were made** - None ship - every row is a true analyst statement. Negatives are constructed at lane build by perturbing the RESULT while leaving the linked cells intact, which the `answer_formulas` and `linked_cells` fields make deterministic
- **How labels were made** - Human annotation of the aggregation operator, the answer formula and the cell-level entity/quantity alignment
- **Mapping onto our task** - `sub_sentence` → claim; linearized table (title + cells) → evidence; shipped rows → 1, result-perturbed rows → 0 (constructed, not shipped)

## Caveats

**Lookup-dominant**: on the dev split 1,195 of 1,671 rows carry `aggregation: ['none']` - 71.5% are pure lookup and add nothing this wave needs; the derived slice is the other ~28%. Statistical-agency register, not business documents. Nested fields (`linked_cells`, `reference_cells_map`, `answer`, `aggregation`, `answer_formulas`) are JSON-encoded into string columns for a stable parquet schema.

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
`experiments/grounding-semantic/R22_hitab_gate.json`.

Fetched by `scripts/fetch_grounding_datasets.py`. The archive `dataset-hitab.zip` is
gitignored; this sidecar is tracked.

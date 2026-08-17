# EQUATE - quantitative-reasoning NLI, incl. AWPNLI arithmetic entailment

The ONLY public corpus found carrying the derivation signal in LABELED form on both legs. `Sam had 9.0 dimes ... his dad gave him 7.0` / `Sam has 16.0 dimes now` → entailment; `... 17.0 dimes now` → contradiction. That is exactly the finqa failure shape - every figure attributable, only the result false - and nothing in the mix teaches it.

- **Source** - github.com/AbhilashaRavichander/EQUATE (`ProcessedDatasets/*.jsonl`, git-LFS); paper Ravichander et al., CoNLL 2019
- **Licence** - MIT (repo LICENSE, GitHub licence API `spdx_id: MIT`, re-read at fetch 2026-08-17)
- **Size** - 9,702 premise/hypothesis pairs - AWPNLI 722, NewsNLI 968, RTE_Quant 166, RedditNLI 250, StressTest 7,596; fetched 2026-08-17: 9702 rows (awpnli: 722, newsnli: 968, rte_quant: 166, redditnli: 250, stresstest: 7596)
- **Languages** - English
- **How negatives were made** - For AWPNLI, the SAME premise paired with a hypothesis stating a wrong arithmetic result - operands correct and present, only the computed value false, 361 such against 361 correct. The other four sets carry natural or template quantity mismatches
- **How labels were made** - Human entailment labels (2-way on AWPNLI and NewsNLI, 3-way elsewhere)
- **Mapping onto our task** - hypothesis → claim; premise → evidence; entailment → 1, contradiction/neutral → 0

## Caveats

Small: the derivation-pure part is AWPNLI's 361 pairs. StressTest (7,596) is template-perturbed quantifier reasoning in a degraded register ('more than 1 years old twin brothers') and is banked SEPARATELY so it can be excluded without a refetch. NumGLUE Type_7 re-serves part of the NewsNLI/RTE_Quant/StressTest lineage - do not double-count the two members. The CoreNLP parse columns (dep / syntax / binary parse, pos, tokens) are DROPPED at fetch; premise, hypothesis and label are kept.

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
`experiments/grounding-semantic/R22_equate_gate.json`.

Fetched by `scripts/fetch_grounding_datasets.py`. The archive `dataset-equate.zip` is
gitignored; this sidecar is tracked.

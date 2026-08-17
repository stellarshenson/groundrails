# NumGLUE - eight-task numerical reasoning suite

Type_7 adds ~9.5k labeled quantitative-entailment rows; Types 1/2/4/8 (~3.5k) add word problems whose answer is a derived value with the operands stated in the text, which is the raw material for a result-perturbation negative the mix has never had.

- **Source** - github.com/allenai/numglue (`data/NumGLUE_{train,dev,test}.json`, JSONL despite the extension); paper Mishra et al., ACL 2022
- **Licence** - ODC-By 1.0 (repo `license.txt`, read at fetch 2026-08-17 - Open Data Commons Attribution, commercial use permitted with attribution)
- **Size** - 92,049 rows - 71,281 train / 10,185 dev / 10,583 test across 8 task types; fetched 2026-08-17: 92049 rows (train: 71281, dev: 10185, test: 10583)
- **Languages** - English
- **How negatives were made** - Only Type_7 (quantitative NLI, 9,452 rows) ships both legs. The word-problem types ship the correct answer alone - negatives must be constructed
- **How labels were made** - Human, inherited from the eight source tasks
- **Mapping onto our task** - Type_7 statement2 → claim, statement1 → evidence, Entailment → 1; word-problem types: answer → claim value, question text → evidence, negatives constructed

## Caveats

**Type_5 and Type_6 (60,857 rows, 66% of the corpus) ARE DROP items** re-served; fetching both this and `drop` double-counts them, and the split axis differs between the two. Type_3 is a multiple-choice commonsense-physics task with no evidence document. Answers for Types 5/6 are DROP answer dicts, stored JSON-encoded. The eight types carry different keys; the frame is the normalized union with nulls.

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
`experiments/grounding-semantic/R22_numglue_gate.json`.

Fetched by `scripts/fetch_grounding_datasets.py`. The archive `dataset-numglue.zip` is
gitignored; this sidecar is tracked.

# DROP - reading comprehension requiring discrete reasoning over paragraphs

The largest licence-clean supply of derived-value claims over natural prose: the benchmark exists precisely because answering needs addition, subtraction, counting or sorting over figures the passage states. 96k questions against ~6.7k passages.

- **HuggingFace** - `ucinlp/drop`
- **Licence** - CC-BY-SA-4.0 (HF card YAML `license: cc-by-sa-4.0`, re-verified at the Hub 2026-08-17)
- **Size** - 86,945 rows - 77,409 train / 9,536 validation; fetched 2026-08-17: 86935 rows (train: 77400, validation: 9535)
- **Languages** - English
- **How negatives were made** - None ship - answers are correct by construction. Negatives are built by perturbing the computed answer while the passage's operands stay untouched
- **How labels were made** - Crowdsourced, adversarially filtered against a QA model
- **Mapping onto our task** - question + answer rewritten to a declarative claim; passage → evidence; shipped answer → 1, perturbed result → 0 (constructed, not shipped)

## Caveats

**QA, not entailment** - forming a claim needs a question+answer → declarative rewrite, and that rewrite is lossy and must be reported as such. Only the `number`-typed answers carry the derivation; span and date answers are lookup. Wikipedia register (NFL recaps, history) and CC-BY-SA share-alike, the same class of term as the banked VitaminC. Overlaps NumGLUE Types 5/6 - do not mix both.

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
`experiments/grounding-semantic/R22_drop_gate.json`.

Fetched by `scripts/fetch_grounding_datasets.py`. The archive `dataset-drop.zip` is
gitignored; this sidecar is tracked.

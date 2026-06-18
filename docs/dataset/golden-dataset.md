# Golden dataset (gold v3)

The single golden dataset for grounding calibration and evaluation, distilled from a private agentic-RAG corpus. One verified ground-truth set plus a derived synthetic augmentation that points back at it - never merged in. Build recipe in [how-to-build-golden-dataset](2026-06-01-how-to-build-golden-dataset.md).

## Two files, one dataset

- **`data/processed/golden_v3.parquet`** - the golden dataset, `role=eval`; human/RAG-verified ground truth; the only set headline metrics are reported on
- **`data/processed/golden_v3_synth_aug.parquet`** - `role=augmentation`; synthetic cross-lingual negatives derived from the golden; training/calibration aid + an offline TNR probe, never counted as ground truth
- **Gitignored** - both carry private text; never committed, synced to the private S3 bucket instead
- **Lineage is explicit** - the augmentation references its parent via `group_id` (shared evidence-source hash) and `parent_sid`; it is a child of the golden, not a sibling

## Golden (verified, role=eval)

The verified ground truth - 5,912 claim/evidence/label rows over 619 distinct evidence sources, re-judged from the prior verified set.

- **Rows** - 5,912; label 3,947 supported / 1,965 hallucination (33% hallucination base rate)
- **Origin** - 3,619 inherited (labels carried from the prior verified set) + 2,293 newly judged
- **Sources** - 619 distinct evidence blobs (`group_id`), the leave-one-source-out unit
- **Schema** - `uid, row_id, role, claim, source_text, label (1=supported, 0=hallucination), lang, lang_norm, origin, trace_id, group_id`
- **Languages (lang_norm, n ≥ 40)** - en 4,569, fr 450, nb 271, es 247, it 129, pt 57, nl 56; a tail of sv 32, de 25, nn 19, da 11 and ~12 lingua mis-detections (n ≤ 15) below 40

The non-English eval is strongly supported-skewed - fr 422/28, nb 254/17, es 212/35, it 122/7, nl 51/5 (supported/hallucination). Native rows measure cross-lingual recall well but carry too few negatives to measure cross-lingual false-flag - the gap the synthetic augmentation exists to fill.

## Synthetic augmentation (derived, role=augmentation)

Cross-lingual hard negatives generated from the golden's own English sources - the "synthetic language translations" enrichment, mirroring the lexical track's R10-R12.

- **Rows** - 2,119, all label 0 (negatives); 240 base source sentences (`parent_sid`) across 37 golden evidence sources, translated into 9 languages
- **Languages (target)** - es 240, fr 239, sv 238, nl 237, nb 237, da 237, de 236, pt 236, it 219
- **Provenance** - `source_lang=en`; translator haiku; verifier sonnet; `verify_method=claude_p_equivalence`; all 2,119 `verified=True`
- **Schema** - golden columns plus `parent_sid, source_corpus, source_lang, target_lang, translator_model, verifier_model, verified, verify_method`
- **Why negatives** - a faithfully translated claim paired cross-lingually against its English source is the hard case a grounder must not over-accept; these supply the cross-lingual negative side the native eval lacks

## Lineage and leakage-safe splitting

- **`group_id`** - a stable 12-hex hash of `source_text`; a base claim and every translation derived from that source share one `group_id`
- **GroupKFold on `group_id`** - keeps a source and all its translations in one fold; a claim can never train on its own translation (the leak a flat concat would have caused)
- **Verified linkage** - 2,119/2,119 augmentation rows resolve to a golden source group; every synthetic source is a golden source
- **Metric discipline** - headline macro-F1 and cross-lingual recall on `role=eval`; the synthetic set enters training folds in-fold and is reported separately as an offline synthetic TNR probe, labelled synthetic

## Version lineage

- **v1** - the original verified set, 2,752 rows, 639 sources (English-dominant)
- **v2** - re-judged enrichment on the same 639 sources, 5,912 rows (inherited + newly judged)
- **v3** - v2 promoted to the canonical golden (`golden_v3.parquet`) plus the synthetic augmentation sidecar, with normalized language, source `group_id`, and explicit lineage; the single source of truth going forward

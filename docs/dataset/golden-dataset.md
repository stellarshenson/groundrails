# Golden dataset (gold v3)

The single golden dataset for grounding calibration and evaluation, distilled from a private agentic-RAG corpus. One verified ground-truth set, plus a derived synthetic augmentation that links back to it rather than merging in.

Build recipe: [how-to-build-golden-dataset](2026-06-01-how-to-build-golden-dataset.md).

## Two files, one dataset

- **`data/processed/golden_v3.parquet`** - `role=eval`; the verified ground truth, the only set headline metrics report on
- **`data/processed/golden_v3_synth_aug.parquet`** - `role=augmentation`; synthetic cross-lingual negatives, a training aid and offline TNR probe, never ground truth
- **Storage** - both gitignored (private text) and synced to the private S3 bucket, never committed
- **Lineage** - the augmentation links to its parent by `group_id` and `parent_sid`, a child of the golden not a sibling

## Golden (verified, role=eval)

Human/RAG-verified ground truth - 5,857 claim/evidence/label rows over 619 evidence sources, re-judged from the prior verified set and deduped.

- **Rows** - 5,857 (deduped on `(claim, source_text, label)`, 55 redundant rows dropped)
- **Label** - 3,892 supported / 1,965 hallucination (33.5% hallucination)
- **Origin** - 3,566 inherited from the prior verified set + 2,291 newly judged
- **Sources** - 619 distinct evidence blobs (the `group_id` leave-one-source-out unit), 636 trace ids
- **Schema** - `uid, row_id, role, claim, source_text, label (1=supported, 0=hallucination), lang, lang_norm, origin, trace_id, group_id`
- **Languages (lang_norm, n ≥ 40)** - en 4,524, fr 442, nb 269, es 247, it 129, pt 57, nl 56
- **Tail (n < 40)** - sv 32, de 25, nn 19, da 11, plus 13 lingua mis-detections (n ≤ 15)
- **Non-English skew** - supported-heavy: fr 414/28, nb 252/17, es 212/35, it 122/7, nl 51/5 (supported/hallucination)
- **Why augmentation** - native non-English rows measure cross-lingual recall but carry too few negatives for false-flag (TNR), which the synthetic set supplies

## Synthetic augmentation (derived, role=augmentation)

Cross-lingual hard negatives generated from the golden's own English sources - the synthetic-translation enrichment, mirroring the lexical track's R10-R12.

- **Rows** - 2,119, all label 0 (negatives)
- **Coverage** - 240 base source sentences (`parent_sid`) across 37 golden sources, each translated into 9 languages
- **Languages (target)** - es 240, fr 239, sv 238, nl 237, nb 237, da 237, de 236, pt 236, it 219
- **Provenance** - `source_lang=en`, translator haiku, verifier sonnet, `verify_method=claude_p_equivalence`, all `verified=True`
- **Schema** - golden columns plus `parent_sid, source_corpus, source_lang, target_lang, translator_model, verifier_model, verified, verify_method`
- **Why negatives** - a faithfully translated claim against its English source is the hard case a grounder must not over-accept

## Lineage and leakage-safe splitting

- **`group_id`** - a stable 12-hex hash of `source_text`, shared by a base claim and every translation from that source
- **GroupKFold on `group_id`** - keeps a source and its translations in one fold, so a claim never trains on its own translation
- **Linkage** - 2,119/2,119 augmentation rows resolve to a golden source group (verified)
- **Metric discipline** - headline macro-F1 and cross-lingual recall on `role=eval`; the synthetic set trains in-fold and is reported as a separate offline TNR probe

## Version lineage

- **v1** - the original verified set, 2,752 rows, 639 sources, English-dominant
- **v2** - re-judged enrichment on the same 639 sources, 5,912 rows
- **v3** - v2 deduped (5,857) as the canonical golden plus the synthetic sidecar, with `lang_norm`, source `group_id`, and explicit lineage - the single source of truth

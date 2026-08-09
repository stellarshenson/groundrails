# R10-H107_pairs.parquet

Procedural-register training pairs for R10-H107 (round 10, pre-registered). Built by `experiments/grounding-semantic/R10-H107_data.py`; gate evidence in `experiments/grounding-semantic/R10-H107_gate_report.md`.

- **proc_code** (55,774 pairs, label mean 0.720) - `KRLabsOrg/lettucedetect-code-hallucination` train split (CC-BY-4.0), sentence-level pairs from token-span annotations; provenance gate PASSED (5 original sources, zero psiloqa/ragtruth rows)
- **proc_gov** (27,898 pairs, label mean 0.855) - IBM MultiDoc2Dial train split (Apache-2.0, `dataset-multidoc2dial.zip` archived here from upstream), human grounding-span positives + deterministic span-anchored corruption negatives
- **Licence** - inherited from the two halves named above: CC-BY-4.0 (`KRLabsOrg/lettucedetect-code-hallucination`) and Apache-2.0 (IBM MultiDoc2Dial). The derived file carries both, so the attribution requirement of CC-BY-4.0 governs it as a whole
- Columns: `claim`, `chunk` (1,500-char window), `label` (float 0/1), `tag` (DANN group)
- 22 rows scrubbed for RAGBench-subset name mentions (conservative; ACL-paper prose citing benchmark names)
- No RAGBench sources, no private data

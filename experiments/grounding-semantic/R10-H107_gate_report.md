# R10-H107 provenance gate report

Zero-GPU gate pre-registered in round 10. Method mirrors R8-H86: inspect the train split's `dataset` column and count rows per source.

## Gate 1 - KRLabsOrg/lettucedetect-code-hallucination: PASSED

- **Train split 66,368 rows, `dataset` column carries 5 values, none an upstream of our mix**: `lettucedetect-wikipedia` 22,577, `lettucedetect-code-agent` 16,319 (SWE-bench substrate), `lettucedetect-readme` 12,475 (GitHub READMEs), `lettucedetect-tool-output` 10,508, `lettucedetect-acl` 4,489 (ACL papers). Zero rows tagged `psiloqa` or `ragtruth` - the R8-H86 repackaging outcome does NOT repeat on the code sibling
- **Labels** - injected edits with exact char offsets (`labels` list of {start, end, label, category}); injector model recorded per row in `metadata`
- **Licence** - `license: cc-by-4.0` confirmed in the card frontmatter
- **Contamination** - no RAGBench source among the five substrates. Conservative scrub applied downstream: 22 built pairs whose claim or evidence mentions a RAGBench subset NAME (ACL-paper prose citing HotpotQA/PubMedQA/ExpertQA as related work) dropped outright
- **Note** - the `lettucedetect-wikipedia` slice shares raw substrate (Wikipedia) with PsiloQA/VitaminC/TabFact; it is original generated material, not repackaged rows, matching the precedent that Wikipedia-as-raw-source is legal

## Gate 2 - IBM MultiDoc2Dial: PASSED (identity + licence)

- Fetched from upstream (`doc2dial.github.io`, the HF repo is a retired loading script); archived at `data/external/datasets/dataset-multidoc2dial.zip`
- Identity verified: 488 documents over 4 US government-service domains (ssa 109, va 138, dmv 149, studentaid 92); train split 3,474 dialogues / 48,002 turns; agent turns carry human grounding-span references. Apache-2.0
- Contamination: government service webpages share no documents with any RAGBench source

## Build output (`R10-H107_pairs.parquet`)

| group | pairs | label mean | notes |
|---|---|---|---|
| proc_code | 55,774 | 0.720 | sentence-level from spans; capped at 2x proc_gov (raw 161,311) |
| proc_gov | 27,898 | 0.855 | 23,856 positives; 4,042 corruption negatives (number_swap 2,926, condition_negation 1,037, identifier_swap 79) |
| **total** | **83,672** | | vs ~112k registered estimate - shortfall is proc_gov corruptible-turn yield |

Deviations from registration: total 83.7k not ~112k (fewer corruptible agent turns than estimated - only turns whose referenced span shares a number/condition/identifier with the utterance are corruptible under the exactness filter); proc_gov negative share 14.5% (natural yield, not forced). Training remains HELD.

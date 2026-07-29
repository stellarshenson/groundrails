# LettuceDetect v2 prose hallucination

Near-miss negatives at multilingual scale. R6-H37 was registered precisely because unrelated negatives flatter a model; these are the hard kind.

- **HuggingFace** - `KRLabsOrg/lettucedetect-prose-hallucination`
- **Licence** - CC-BY-4.0
- **Size** - 87,800 - 78,900 train / 3,360 val / 5,600 test
- **Languages** - 14
- **How negatives were made** - Near-miss by construction - an LLM proposes localized replacement edits (wrong value, wrong identifier, unsupported addition) applied deterministically so exact character offsets survive
- **How labels were made** - LLM-generated and LLM-judged, character spans
- **Mapping onto our task** - answer sentence -> claim; context field -> evidence; tagged span -> unsupported

## Caveats

Documents are ACL papers, READMEs and Wikipedia markdown - treat as LANGUAGE coverage, not domain coverage. Labels are not human-verified.

## Provenance

Selected in the round 7 dataset survey, `reports/research-grounding-datasets.md`. Every corpus
in this directory passed three filters together: a licence permitting commercial use, source
documents shipping alongside the claims, and a task shape that maps onto (claim, evidence) →
supported. Corpora excluded on licence alone: TrueTeacher (CC-BY-NC, 1.38M), HaluBench
(CC-BY-NC), MS MARCO (non-commercial), ANLI (CC-BY-NC), MEMERAG (card forbids training).

Fetched by `scripts/fetch_grounding_datasets.py`. The archive `dataset-lettucedetect-prose.zip` is
gitignored; this sidecar is tracked.

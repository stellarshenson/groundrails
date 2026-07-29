# FaithDial

The only corpus here where BOTH sides of the conversation are human, and the negatives are hallucinations a person actually wrote rather than a model.

- **HuggingFace** - `McGill-NLP/FaithDial`
- **Licence** - MIT
- **Size** - 50,761 turns / 5,649 dialogues; ~18.4k train rows in the HF plain_text config
- **Languages** - English
- **How negatives were made** - Genuine human-written hallucinated utterances from Wizard of Wikipedia, kept alongside the amended faithful version
- **How labels were made** - Human - MTurk amendment plus BEGIN labels (Hallucination / Entailment / Generic)
- **Mapping onto our task** - utterance -> claim; knowledge snippet -> evidence; BEGIN label -> verdict

## Caveats

Small and English-only. Wikipedia-snippet evidence.

## Provenance

Selected in the round 7 dataset survey, `reports/research-grounding-datasets.md`. Every corpus
in this directory passed three filters together: a licence permitting commercial use, source
documents shipping alongside the claims, and a task shape that maps onto (claim, evidence) →
supported. Corpora excluded on licence alone: TrueTeacher (CC-BY-NC, 1.38M), HaluBench
(CC-BY-NC), MS MARCO (non-commercial), ANLI (CC-BY-NC), MEMERAG (card forbids training).

Fetched by `scripts/fetch_grounding_datasets.py`. The archive `dataset-faithdial.zip` is
gitignored; this sidecar is tracked.

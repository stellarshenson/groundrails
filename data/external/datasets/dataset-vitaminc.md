# VitaminC

Every other corpus in this directory teaches the base rate as much as the boundary, because its negatives are topically distant. VitaminC's negatives are minimally different from their positives by construction, which is exactly the discrimination R6-H37 was registered to demand and the failure mode rounds 4-6 kept finding: models that confirm a claim against unrelated text.

- **HuggingFace** - `tals/vitaminc`
- **Licence** - CC-BY-SA-3.0 (Wikipedia-derived) - VERIFY before shipping a model trained on it
- **Size** - ~489k claim-evidence pairs
- **Languages** - English
- **How negatives were made** - **Contrastive Wikipedia revisions** - real edits to real articles where a single factual change flips the verdict, so the negative differs from its positive by one number, entity or qualifier and nothing else. This is the textbook near-miss construction and the best-built negatives in the field
- **How labels were made** - Human annotation over genuine revision pairs (SUPPORTS / REFUTES / NOT ENOUGH INFO)
- **Mapping onto our task** - claim -> claim; evidence -> evidence; SUPPORTS -> 1, otherwise 0

## Caveats

Wikipedia register, not conversational RAG - the dataset survey deprioritised it on DOMAIN, never on quality, and named it the fallback if domain-matched data underdelivers. English only. Three-way labels must be collapsed to binary (SUPPORTS -> grounded, REFUTES and NEI -> not).

## Provenance

Selected in the round 7 dataset survey, `reports/research-grounding-datasets.md`. Every corpus
in this directory passed three filters together: a licence permitting commercial use, source
documents shipping alongside the claims, and a task shape that maps onto (claim, evidence) →
supported. Corpora excluded on licence alone: TrueTeacher (CC-BY-NC, 1.38M), HaluBench
(CC-BY-NC), MS MARCO (non-commercial), ANLI (CC-BY-NC), MEMERAG (card forbids training).

Fetched by `scripts/fetch_grounding_datasets.py`. The archive `dataset-vitaminc.zip` is
gitignored; this sidecar is tracked.

# RAGBench (safe core - MS MARCO and CUAD subsets EXCLUDED)

The only large corpus whose DOCUMENTS resemble ours - support tickets, consumer manuals, a car manual, financial and legal filings.

- **HuggingFace** - `galileo-ai/ragbench`
- **Licence** - CC-BY-4.0 at the collection level - SEE CAVEAT
- **Size** - ~100k total, ~78k train across 12 subsets; ~60k over the 10 safe ones
- **Languages** - English
- **How negatives were made** - Naturally occurring - GPT-3.5-0125 and Claude-3-Haiku prompted permissively with no adherence instruction, so authentic drift appears
- **How labels were made** - GPT-4-0125-preview with chain-of-thought; no human annotation
- **Mapping onto our task** - response sentence -> claim; retrieved documents -> evidence

## Caveats

LICENCE: the CC-BY-4.0 tag sits over 12 upstream corpora including MS MARCO (Microsoft, non-commercial research only) and CUAD. A vendor tag does not extinguish an upstream restriction, so those two subsets are EXCLUDED here and only the 10 listed are fetched. Labels are GPT-4, not human.

## Provenance

Selected in the round 7 dataset survey, `reports/research-grounding-datasets.md`. Every corpus
in this directory passed three filters together: a licence permitting commercial use, source
documents shipping alongside the claims, and a task shape that maps onto (claim, evidence) →
supported. Corpora excluded on licence alone: TrueTeacher (CC-BY-NC, 1.38M), HaluBench
(CC-BY-NC), MS MARCO (non-commercial), ANLI (CC-BY-NC), MEMERAG (card forbids training).

Fetched by `scripts/fetch_grounding_datasets.py`. The archive `dataset-ragbench.zip` is
gitignored; this sidecar is tracked.

## Subsets fetched

- `techqa`
- `emanual`
- `delucionqa`
- `expertqa`
- `hagrid`
- `finqa`
- `tatqa`
- `covidqa`
- `pubmedqa`
- `hotpotqa`

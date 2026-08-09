# SciFact

Expert-written scientific claims paired with the abstracts that support or refute
them - a near-miss construction in a register no other admitted corpus covers, and the R13 gate
already measured it clean against the arena.

- **Source** - the upstream AI2 release,
  `https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz`
- **Licence** - **CC BY 4.0 (claims) + ODC-By (abstracts)**, per the upstream AI2 release. These
  terms are authoritative over the HuggingFace mirror `allenai/scifact`, which tags the dataset
  `cc-by-nc-2.0`; the discrepancy is recorded under Caveats
- **Size** - 1,258 labelled (claim, abstract) rows from train+dev - 508 SUPPORT / 265 CONTRADICT /
  485 NEI; 5,183 abstracts in the corpus file
- **Languages** - English
- **How negatives were made** - expert claim rewriting: annotators negate or alter a claim drawn
  from a citation sentence so the same abstract refutes it
- **How labels were made** - human expert annotation with rationale sentences (SUPPORT /
  CONTRADICT / NOINFO)
- **Mapping onto our task** - claim → claim; cited abstract → evidence; SUPPORT → 1, CONTRADICT
  and NEI → 0

## Caveats

**Licence discrepancy, recorded deliberately**: the HuggingFace mirror
`allenai/scifact` carries a `cc-by-nc-2.0` tag while the upstream AI2 release states CC BY 4.0 for
the claims and ODC-By for the abstracts. Data here is taken from the AI2 S3 release and the
upstream terms are treated as authoritative, per the author's ruling of 2026-08-09. A
non-commercial reading would bar any shipped model trained on it.

The test split is blind (no labels) and unusable for training or evaluation. Biomedical-literature
register, not conversational RAG.

## Provenance

Admitted by the author's ruling of 2026-08-09, clause 2 ("NC-class"); iFixit
was refused in the same clause. The provenance gate is already recorded in the canonical log: 0 of
5,183 abstracts match pubmedqa / covidqa / expertqa arena documents at 8-gram Jaccard >= 0.3 (max
best-Jaccard 0.0163, p99 0.0046, spike controls 9/9) -
`experiments/grounding-semantic/R13-scifact_gates_result.json`.

Fetched by `scripts/fetch_register_corpora.py scifact`. The downloaded data under
`data/external/datasets/scifact/` is gitignored; this sidecar is tracked.

# How to Build the Golden Dataset

A per-claim labelled dataset (supported vs hallucination) built from real a production RAG assistant production traffic, used to calibrate and evaluate grounding signals. Two independent LLM judges label every claim and only their agreements are kept, so each record carries two-judge backing.

**Dataset location**: [`data/processed/golden_grounding_evidence_verified.parquet`](../../data/processed/golden_grounding_evidence_verified.parquet) - dataset card in [`golden_grounding_evidence_verified.md`](../../data/processed/golden_grounding_evidence_verified.md)

## Dataset

- **2,752 records** - one per claim, drawn from 639 real prod conversations; only conversations with at least one dual-agreed checkable fact contribute records (the rest are all-NA or judge-split)
- **786 hallucination / 1,966 supported** - 29% hallucination rate (organic-majority; see Caveats)
- **schema** - `{claim, source_text, label, lang, user_id, trace_id}`; label 1 = supported, 0 = hallucination
- **size** - 2.4 MB parquet (zstd), plain git; columnar zstd collapses the evidence blob (avg ~57 KB `source_text`) that JSON repeated across a trace's claims
- **languages** - en dominant, plus fr-FR, nb-NO, es-ES, nl-NL, it-IT, pt-PT, sv-SE, de-DE
- **path** - `data/processed/golden_grounding_evidence_verified.parquet`
- **built additively** - several batches of ~100 traces; the newest-first window was scouted first, then extended backward to 2025-11 for organic coverage, excluding the QA-test accounts; rebuilding extends the set without re-judging prior batches

## Why this dataset

The the assistant's prose is mostly faithful, so hallucinations are rare and never thumbs-down flagged in volume; a vote is sparse and lagging and most bad answers are never voted. The only reliable label is a per-claim LLM judge applied broadly.

- **Measures** - how well a grounding signal separates supported claims from hallucinations on answers users actually get
- **Not votes** - absence of a thumbs-down is not "supported"; gold comes only from the judges

## Pipeline

Five stages - scout, primary judge, cleanup, verification, build. Every per-trace artefact persists to disk, so the run is resumable and the next batch reuses everything.

### 1. Scout

Reduce each prod trace to a judge unit - the answer's claims plus the documentation it was built from. `experiments/grounding-semantic/golden_dataset.py`.

- **Source** - the production trace store, 986 answered traces in a 90-day window, paged by `scrollId`
- **Evidence** - recovered from the trace's tool / rag spans (the `<documents>` markup the retrieval tools returned), not the answer span; needs the per-id fetch since the search-list omits span bodies
- **Claims** - extracted from the raw markdown answer; an HTML-stripper collapses headers and bullets and yields zero claims on the structured documentation answers we most want
- **Lexical pre-pass** - each claim grounded against its evidence to record a weak prior (NOT_FOUND / CONFIRMED / CONTRADICTED)
- **Cap** - up to 12 claims judged per trace (all lexically flagged plus a sampled cap of confirmed)
- **Two filters removed** - the `guardrail_rejection` flag is set on 84% of traces including real evidence-backed answers; and strip-then-extract drops structured answers - both wrong, both removed

### 2. Primary judge - Haiku

Each trace's claims judged by a fast, cheap model. `data/interim/golden_judge.workflow.mjs` (dynamic workflow, one agent per trace, structured output to `judge_results/`).

- **Model** - Claude Haiku; cheap and high-throughput, the heavier models rate-limit at this fan-out
- **Labels** - SUPPORTED (stated or entailed by the evidence), UNSUPPORTED (a documentation-fact claim the evidence does not support - a hallucination), NA (not a checkable documentation claim)
- **Isolation** - one agent per trace, so a failure or rate-limit on one trace does not lose the others

### 3. Cleanup - tightened prompt

The first Haiku pass over-flagged, so the prompt was tightened and all traces re-judged - the single most important quality step.

- **Problem** - the first pass labelled 295 of 903 claims UNSUPPORTED, mostly not hallucinations: the model's own arithmetic and advice, disclaimers, headings, markdown / image fragments
- **Fix** - the prompt now grades only claims asserted as product-documentation fact and auto-NAs the model's computations, advice, disclaimers, and markup
- **Effect (first batch)** - re-judging moved 365 / 295 / 243 (SUPPORTED / UNSUPPORTED / NA) to 349 / 170 / 384; the inflated UNSUPPORTED bucket nearly halved

### 4. Verification - Sonnet as the second judge

A stronger independent judge re-labels the consequential classes; keep only labels both judges agree on. `experiments/grounding-semantic/golden_verify.py`.

- **Model** - Claude Sonnet, the stronger judge of record, run the same way
- **Scope** - every Haiku-UNSUPPORTED claim plus a matched sample of Haiku-SUPPORTED, against relaxed evidence (all tool / rag spans, no markup requirement, closing a ~5% recovery gap)
- **Agreement** - 69% with Cohen's kappa 0.50; of the Haiku-UNSUPPORTED, Sonnet confirmed under half - the rest were Haiku over-flags (the model's own reasoning)
- **Confidence** - the kept set is by construction the agreed subset, so each label has two-judge backing

### 5. Build

Keep records where Haiku and Sonnet agree on SUPPORTED / UNSUPPORTED; drop disagreements rather than guess. `golden_verify.py --action build` writes the gold parquet (with `user_id` / `trace_id` provenance joined from the source trace).

- **Result** - 2,752 verified records: 1,966 supported, 786 hallucination (29%)
- **Dropped disagreements** - not guessed: UNSUPPORTED-to-NA, SUPPORTED-to-NA, and the cross-flips
- **Tradeoff** - dropping disagreements makes the hallucination set high-precision but conservative; clean labels beat coverage for a calibration gold

## Reproduction

```
python -m golden_dataset --n 400        # scout -> judge_units/ (cumulative target; +100 per batch)
# Haiku primary judge: run golden_judge.workflow.mjs, model:"haiku", over the NEW trace ids
python -m golden_verify --action prep   # UNSUPPORTED + SUPPORTED sample
python -m golden_verify --action prep2  # remaining SUPPORTED
# Sonnet verify: run golden_judge.workflow.mjs, model:"sonnet", over the verify units lacking a result
python -m golden_verify --action agree  # agreement / kappa
python -m golden_verify --action build  # dual-agreed gold -> parquet
```

- **Judging** - a Claude Code dynamic workflow (`Workflow` over `golden_judge.workflow.mjs`): one agent per trace, structured per-claim labels, `model` set via args (haiku primary, sonnet verify); pass only the new/unjudged trace ids so prior batches are not re-judged
- **Additive** - keep the scout window fixed and only raise `--n` so newest-first order is stable and earlier judge-units re-write identically; the next batch judges only the new traces
- **Output** - `build` writes `golden_grounding_evidence_verified.parquet` (zstd, ~2.4 MB) with `user_id` / `trace_id` provenance; filter `user_id` to drop the QA-test cohort (see Caveats)
- **Caching** - all model calls and trace fetches cache, so extending the set re-judges only the new traces

## Caveats

- **Conservative** - dropping judge disagreements under-counts hallucinations in the overlap region
- **Judge ceiling** - labels are an LLM-judge consensus, not human gold; kappa 0.50 means the judges genuinely disagree on a third of the hard cases, which is why only agreements are kept
- **Test-user concentration** - a two-account QA/test cohort (`4bb86d8d...`, `6f392ffc...`) contributes 79% of the hallucinations (623/786) at a 56% rate; organic users sit at ~10%. The later batches exclude these accounts, so the organic majority (1,634 records) is the realistic distribution. Filter on the `user_id` field for the organic subset
- **Watch the two traps** - the `guardrail_rejection` filter and strip-then-extract; both were real and fixed, both recur when extending the pipeline

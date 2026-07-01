# Dataset Methodology Receipt

How the groundrails grounding datasets are produced and judged: a raw-answers corpus and a consolidated per-claim gold, both descended from a dual-LLM-judged production corpus where two independent judges label every claim and only their agreements are kept. This receipt covers what each dataset is, how claims get their verified label, and how the fair extraction re-run re-judges and re-calibrates per extractor.

Lineage root: the upstream golden dataset was built by the protocol in the source assistant project's `how-to-build-golden-dataset.md` (private). The datasets here re-extract and enrich that gold; the judging protocol is unchanged.

Datasets live under `data/` (gitignored, private production data - never committed). This receipt documents the method, not the data.

## Datasets

Two artefacts, one raw and one labelled, plus the per-extractor labelled sets the fair re-run produces.

- **raw_v5** (`data/raw/raw_v5/`) - the raw-answers corpus, one row per production conversation: 1,434 answers, 636 carrying verified gold. Columns: `trace_id`, `lang` / `lang_norm`, `question`, `answer`, `source_text` (the evidence the answer was built from), gold rollups (`n_gold_claims`, `n_gold_supported`, `n_gold_hallucination`), `has_gold`. The answer is the raw document a claim extractor must segment - the individuated gold cannot drive extraction, so the raw answer is kept
- **golden_v5** (`data/processed/golden_v5/`) - the single enriched per-claim gold: 8,776 claims (5,857 real-conversation eval + 2,119 synthetic augmentation + 800 VitaminC contrastive), labels 4,484 hallucination / 4,292 supported. Each claim carries its evidence (`source_text`), verified `label` (1 = supported, 0 = hallucination), language, and both the shipped lexical signals (`lex_p`, `lex_blocked`, `lex_fired`, `lex_contra`) and the semantic-cascade signals (`cos_max`, `rr_max`, `nli_ent`, `nli_contra`)
- **per-extractor sets** (`data/processed/sat_vs_regex_fair/`) - `records_regex.parquet` / `records_sat.parquet`, produced by the fair re-run below: one labelled record per segment each extractor emits, schema `{claim, source_text, label, lang, trace_id, origin}`

## How a claim is judged

The only reliable label is a per-claim LLM judge applied broadly - the production assistant's prose is mostly faithful, so hallucinations are rare and never user-flagged in volume. Two independent judges label every checkable claim and only agreements are kept, so each record carries two-judge backing.

- **Evidence** - the trace's tool / rag span outputs (the retrieved documentation the answer was built from), markup-stripped, deduped, joined; this is the ONLY source of truth the judge sees
- **Primary judge** - Claude Haiku, cheap and high-throughput; labels each claim SUPPORTED (stated or entailed by the evidence), UNSUPPORTED (a documentation-fact claim the evidence does not support - a hallucination), or NOT_A_CLAIM (greeting, advice, the model's own arithmetic, markup fragment)
- **Verify judge** - Claude Sonnet, the stronger judge of record, re-labels the same claims independently
- **Agreement** - keep only labels both judges agree on; ~69% agreement, Cohen's kappa 0.50 on the upstream gold; disagreements are dropped rather than guessed
- **NOT_A_CLAIM** - dual-agreed non-claims are excluded from the gold and reported as the extraction-precision number, not as labels

## Fair extraction re-run

The fair SaT-vs-regex benchmark (`scripts/sat_vs_regex_fair.py`, Round 13 follow-on) tests whether the SaT neural segmenter extracts claims better than the regex `extract_claims`, fixing the two confounds of the original Round 13 - the verdict was reused from SaT-calibrated gold, and regex segments were never re-grounded. It runs the judging protocol above once per extractor, then re-grounds and re-calibrates each extractor's own population.

- **Segment** - each raw answer is segmented by the extractor under test: regex `extract_claims` (sentence split + English verb/copula gate) or SaT (neural segmenter + a language-agnostic 3-token content gate)
- **Inherit or judge** - a segment fuzzy-matching a golden_v5 gold claim (rapidfuzz partial_ratio >= 90) inherits its verified label for free; the rest are judged fresh by the dual-LLM protocol. Inherited / new-to-judge: regex 4,667 / 5,784, SaT 4,047 / 4,387
- **Hygienic judge** - each `claude -p` call runs with `--no-session-persistence --disable-slash-commands --strict-mcp-config --setting-sources ''`: no persisted session, no skills, no MCP, no project settings, so the judge fleet sees only evidence + claims and neither persists state nor bloats context. Claims are chunked at 25 per call; per-trace results cache to disk so the run is resumable
- **Build** - keep dual-agreed SUPPORTED / UNSUPPORTED labels per extractor; drop NOT_A_CLAIM agreements and judge disagreements, reporting the agreed rate as new-segment extraction precision
- **Re-ground + re-calibrate** - each extractor's labelled records go through the shipped `groundrails.calibrate` (dogfood): every segment is re-grounded by the real `ground()` into the manifold features, then the Bayesian calibrator is re-fit on that extractor's own population. Grouped 5-fold by trace (a trace's segments never split across folds) gives an honest macro-F1 at each extractor's own fitted operating point; a full-data fit is exported as the per-extractor calibration JSON via the same library path

## Reproduction

```
uv run python scripts/sat_vs_regex_fair.py units
uv run python scripts/sat_vs_regex_fair.py judge --extractor regex --model haiku
uv run python scripts/sat_vs_regex_fair.py judge --extractor regex --model sonnet
uv run python scripts/sat_vs_regex_fair.py judge --extractor sat   --model haiku
uv run python scripts/sat_vs_regex_fair.py judge --extractor sat   --model sonnet
uv run python scripts/sat_vs_regex_fair.py build
uv run python scripts/sat_vs_regex_fair.py calibrate
```

- **Re-calibration is library-native** - the calibrate step calls `groundrails.calibrate`; the same path is exposed as `groundrails calibration fit --input records.parquet -o calibration.json` and `groundrails calibration eval --input records.parquet --calibration calibration.json`, so any labelled `(claim, source_text, label)` corpus can re-calibrate the manifold and the result drops straight into `groundrails.init`
- **Resumable** - units, per-trace judge results, and records all persist; re-running judges only the unjudged traces
- **Deterministic where it can be** - extraction, grounding, fuzzy inheritance, and the grouped folds are deterministic; only the LLM judge is not, which is why two judges and agreement are required

## Caveats

- **Conservative** - dropping judge disagreements under-counts hallucinations in the overlap region; clean labels beat coverage for a calibration gold
- **Judge ceiling** - labels are an LLM-judge consensus, not human gold; kappa 0.50 means the judges genuinely disagree on a third of the hard cases, which is why only agreements are kept
- **Inheritance anchor** - in the fair re-run, segments that match a golden_v5 claim inherit its label; that gold was itself SaT-extracted, so the shared (inherited) segments favour SaT's granularity. The new-to-judge segments - where the extractors actually diverge - are freshly dual-judged, so the divergent region is unbiased; the residual is the inherited overlap
- **Synthetic and contrastive rows** - golden_v5 mixes real eval claims with synthetic augmentation and VitaminC contrastive pairs; filter on `role == "eval"` for the real-conversation subset

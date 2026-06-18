# groundrails

Grounding guardrails for agentic RAG - deterministic, torch-free claim verification.

groundrails checks whether a claim is supported by source text and flags hallucinations and contradictions, with no LLM in the loop. It runs on CPU, returns a structured verdict per claim, and is the library extracted from the lexical-grounding research line (Rounds 1-12).

## What it does

- **Claim grounding** - locate a claim in source text across three lexical layers (regex exact, Levenshtein fuzzy, BM25 token-recall) and return a verdict with per-layer scores
- **Hallucination and contradiction detection** - numeric mismatch (`512` vs `1000`), named-entity mismatch (`H100` vs `A100`), and unsupported-claim flags
- **Cross-lingual grounding** - a claim in one language against evidence in another, via a torch-free MT bridge (argos / CTranslate2) and a SaT sentence segmenter (OpenVINO INT8)
- **Self-consistency** - intra-document divergence check (same entity or number category, different values)
- **Frozen-weight verdict** - a logistic manifold over 18 features at the `high` tier; deterministic, no per-call sampling
- **Optional semantic layer** - embedding retrieval + NLI entailment behind the `[semantic]` extra; off by default, keeps the core torch-free

## Install

```bash
pip install -e .                 # core grounder (torch-free)
pip install -e ".[semantic]"     # add the optional embedding + NLI layer
make install                     # full uv env with dev extras
```

## CLI

The `groundrails` command verifies claims against source text read as plain UTF-8.

- `groundrails ground --claim "<claim>" --source doc.txt` - ground one claim; exit 0 if grounded, 1 if not
- `groundrails ground --manifest claims.json --source doc.txt [--json]` - batch over many claims
- `groundrails extract-claims --document doc.md` - heuristic sentence-to-claim extractor
- `groundrails check-consistency --document doc.md` - intra-document divergence report
- `groundrails config` - print the resolved config + calibration block
- `groundrails setup` - first-run semantic model/cache config

`--semantic` adds the optional embedding + NLI bundle to `ground`.

## Python API

```python
from groundrails import ground, ground_batch

m = ground(
    "The Eiffel Tower is in Paris.",
    ["The Eiffel Tower is located in Paris, France."],
)
print(m.match_type, m.combined_score, m.verdict_probability)
```

`ground_batch(claims, sources, ...)` runs many claims against the same sources and returns a list of verdicts.

## Language support

Cross-lingual grounding needs an argos `<lang>→en` model for the claim's language. English is native and needs no bridge. Nine non-English languages have models installed.

| Supported (MT bridge) | Not supported (no model) |
|---|---|
| Danish `da`, German `de`, Spanish `es`, French `fr`, Italian `it`, Norwegian Bokmal `nb`, Dutch `nl`, Portuguese `pt`, Swedish `sv` | Nynorsk `nn`, and any language without an installed argos model (`la`, `yo`, `et`, `eo`, `ts`, `tl`, `ca`, `cs`, `hu`, `tn`, ...) |

- **Supported** - full cross-lingual grounding: the claim is translated to English, then recall-matched against the evidence
- **Unsupported** - a non-English claim with no installed model → `ground()` raises `UnsupportedLanguageError`; the claim is hard-blocked, not scored, so unsupported languages cannot pollute metrics (batch callers wrap per claim)
- **Add a language** - `argospm install translate-<code>_en` installs the model; the bridge picks it up automatically
- **Region tags** - the detector strips the region before lookup (`it-IT` → `it`, `nb-NO` → `nb`)

## Project layout

- `src/groundrails/` - the grounder (`grounding`, `lexical`, `lexical_mt`, `entity_check`, `consistency`, `calibration`, `chunking`, `extract`, `sat/`, `config` + the shipped `config_document_processing.yaml`)
- `experiments/grounding/` - research harness (Rounds 1-12)
- `notebooks/` - calibration, SaT / OpenVINO conversion, manifold retraining
- `tests/` - grounder tests + the exact-equivalence golden
- `data/`, `models/`, `references/` - datasets, OpenVINO IR, papers (large/private content gitignored)

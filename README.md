# groundrails

[![CI](https://github.com/stellarshenson/groundrails/actions/workflows/ci.yml/badge.svg)](https://github.com/stellarshenson/groundrails/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/groundrails.svg)](https://pypi.org/project/groundrails/)
[![Total PyPI downloads](https://static.pepy.tech/badge/groundrails)](https://pepy.tech/project/groundrails)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Brought To You By KOLOMOLO](https://img.shields.io/badge/Brought%20To%20You%20By-KOLOMOLO-00ffff?style=flat)](https://kolomolo.com)
[![Donate PayPal](https://img.shields.io/badge/Donate-PayPal-blue?style=flat)](https://www.paypal.com/donate/?hosted_button_id=B4KPBJDLLXTSA)

Grounding guardrails for agentic RAG - deterministic, torch-free claim verification.

groundrails checks whether a claim is supported by source text and flags hallucinations and contradictions, with no LLM in the loop. It runs on CPU, returns a structured verdict per claim, and is the library extracted from the lexical-grounding research line (Rounds 1-12).

## Why

Agentic RAG can assert things its sources never said. The usual fix - a second LLM grading each answer - is non-deterministic, costs a model call per claim, and gives no auditable reason for its verdict. groundrails is the deterministic gate that runs before output reaches the user.

- **No LLM in the loop** - frozen logistic weights over lexical features → same input gives the same verdict on every run
- **Cheap** - CPU-only, torch-free core; milliseconds per claim, no GPU, no API call
- **Auditable** - every verdict carries per-layer scores and the exact numeric or entity mismatch that triggered a flag
- **Cross-lingual offline** - claim-vs-evidence language gap is bridged by an on-device MT bridge, no translation API
- **Research-backed** - distilled from the lexical-grounding experiments (Rounds 1-12); see [Documentation](#documentation)

## What it does

- **Claim grounding** - locate a claim in source text across three lexical layers (regex exact, Levenshtein fuzzy, BM25 token-recall) and return a verdict with per-layer scores
- **Hallucination and contradiction detection** - numeric mismatch (`512` vs `1000`), named-entity mismatch (`H100` vs `A100`), and unsupported-claim flags
- **Cross-lingual grounding** - a claim in one language against evidence in another, via a torch-free MT bridge (argos / CTranslate2) and a SaT sentence segmenter (OpenVINO INT8)
- **Self-consistency** - intra-document divergence check (same entity or number category, different values)
- **Frozen-weight verdict** - a logistic manifold over 18 features at the `high` tier; deterministic, no per-call sampling
- **Optional semantic layer** - embedding retrieval + NLI entailment behind the `[semantic]` extra; off by default, keeps the core torch-free

## Install

```bash
pip install groundrails              # core grounder (torch-free)
pip install "groundrails[semantic]"  # add the optional embedding + NLI layer
```

## CLI

The `groundrails` command verifies claims against source text read as plain UTF-8.

```bash
# put the evidence in a file, then ground a claim against it
echo "The Eiffel Tower is located in Paris, France." > doc.txt
groundrails ground --claim "The Eiffel Tower is in Paris." --source doc.txt
# → exit 0 (grounded); prints the match type, per-layer scores, and matched text
```

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

| Language | Code | Grounding |
|---|:---:|:---:|
| English | `en` | ✓ |
| Danish | `da` | ✓ |
| German | `de` | ✓ |
| Spanish | `es` | ✓ |
| French | `fr` | ✓ |
| Italian | `it` | ✓ |
| Norwegian Bokmål | `nb` | ✓ |
| Dutch | `nl` | ✓ |
| Portuguese | `pt` | ✓ |
| Swedish | `sv` | ✓ |
| Norwegian Nynorsk | `nn` | ✗ |
| Latin | `la` | ✗ |
| Yoruba | `yo` | ✗ |
| Estonian | `et` | ✗ |
| Esperanto | `eo` | ✗ |
| Tsonga | `ts` | ✗ |
| Tagalog | `tl` | ✗ |
| Catalan | `ca` | ✗ |
| Czech | `cs` | ✗ |
| Hungarian | `hu` | ✗ |
| Tswana | `tn` | ✗ |

`✓` grounded - English native, others via the argos MT bridge · `✗` no installed argos model → `UnsupportedLanguageError` (any language not listed defaults to `✗`)

- **Supported** - full cross-lingual grounding: the claim is translated to English, then recall-matched against the evidence
- **Unsupported** - a non-English claim with no installed model → `ground()` raises `UnsupportedLanguageError`; the claim is hard-blocked, not scored, so unsupported languages cannot pollute metrics (batch callers wrap per claim)
- **Add a language** - `argospm install translate-<code>_en` installs the model; the bridge picks it up automatically
- **Region tags** - the detector strips the region before lookup (`it-IT` → `it`, `nb-NO` → `nb`)

## Documentation

The `docs/` tree carries the concept, the calibration method, and the full research history behind the shipped weights.

- **Concept** - [`docs/grounding_concept.md`](docs/grounding_concept.md) - what grounding means here and how a verdict is assembled
- **Calibration** - [`docs/grounding_calibration.md`](docs/grounding_calibration.md) - how the frozen weights and thresholds were fit
- **Experiments log** - [`docs/experiments/lexical-grounding-experiments.md`](docs/experiments/lexical-grounding-experiments.md) - Rounds 1-12, what moved the metrics and what did not
- **State of the art** - [`docs/experiments/lexical-grounding-sota.md`](docs/experiments/lexical-grounding-sota.md) - how the deterministic cascade compares to published grounding methods
- **Positional analysis** - [`docs/lost_in_the_middle_grounding_analysis.md`](docs/lost_in_the_middle_grounding_analysis.md) - lost-in-the-middle behaviour over long evidence

## Project layout

- `src/groundrails/` - the grounder (`grounding`, `lexical`, `lexical_mt`, `entity_check`, `consistency`, `calibration`, `chunking`, `extract`, `sat/`, `config` + the shipped `config_document_processing.yaml`)
- `experiments/grounding/` - research harness (Rounds 1-12)
- `notebooks/` - calibration, SaT / OpenVINO conversion, manifold retraining
- `tests/` - grounder tests + the exact-equivalence golden
- `data/`, `models/`, `references/` - datasets, OpenVINO IR, papers (large/private content gitignored)

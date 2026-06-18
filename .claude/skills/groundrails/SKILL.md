---
name: groundrails
description: >
  Install and use the groundrails claim-grounding library (PyPI) to verify whether claims are
  supported by source text and return the exact support location (quote, line, char offset).
  Use this skill whenever the user wants to ground, verify, or fact-check claims against
  evidence, check an answer or document for hallucinations or contradictions, verify RAG output,
  or set up / install groundrails - even if they do not name the library. Asks lexical-only vs
  semantic (full) and installs the matching tier.
---

# groundrails

Deterministic claim grounder: it checks whether a claim is supported by source text, flags hallucinations and contradictions, and returns the exact support location. No LLM, runs on CPU, two install tiers.

- **lexical** - fast, deterministic, torch-free; English plus 9 languages via an on-device MT bridge; no model download
- **semantic (full)** - adds an OpenVINO int8 cascade (bge-reranker + mDeBERTa NLI) for the claims the lexical tier is unsure about; ~1.4 GB of models, best recall

## Install

Check first, ask the tier, then install with uv (preferred) or pip.

- **Check** - `python3 -c "import groundrails" 2>/dev/null && echo INSTALLED || echo MISSING`
- **Ask the tier** - if MISSING, ask the user lexical vs semantic (full) with AskUserQuestion
- **uv over pip** - prefer `uv pip install` (faster, no lockfile); it targets the active virtualenv (`--system` for the base interpreter)
- **Never `uv add`** - it rewrites `pyproject.toml` and creates a `uv.lock`; `uv pip install` does neither

```bash
if command -v uv >/dev/null 2>&1; then
  uv pip install groundrails                        # lexical
  uv pip install "groundrails[semantic-grounder]"   # semantic (full)
else
  pip install groundrails                           # lexical
  pip install "groundrails[semantic-grounder]"      # semantic (full)
fi
```

- **Semantic only** - prefetch the models once with `groundrails download`
- **Verify** - `groundrails --help`

## Inputs

groundrails grounds plain UTF-8 text only, and takes two kinds of input.

- **claims** - the assertions to verify: a document, a claims file, or inline strings
- **evidence** - the source text the claims must be backed by: one or more text files
- **Text only** - convert a PDF, DOCX, or scanned doc to markdown/text first, then ground the result; a binary file is skipped

## Adapt the claims

Pick the input form by what the user provides.

- **A document or answer** - pass it positionally; groundrails extracts the claims from it
- **A claims list** (JSON or one-per-line) - pass it with `--claims FILE`
- **One or a few claims** - pass each with `--claim "..."` (repeatable)
- **Keep claims atomic** - one verifiable assertion per claim; split a compound claim ("X and Y") into separate `--claim`s; ground facts, not opinion

## Ground

Three forms; every positional after the claim source is evidence.

```bash
groundrails ground answer.md evidence.txt --json                  # claims extracted from a document
groundrails ground --claims claims.json ev1.txt ev2.txt --json    # a claims file, many evidence sources
groundrails ground --claim "The tower is in Paris." ev.txt --json # inline (repeatable)
```

- **`--semantic 1`** - escalate the uncertain claims to the cascade (needs the semantic tier)
- **`--json`** - the grounding document; drop it for a readable line per claim; `--full-output` adds per-scorer detail
- **Exit code** - 0 if every claim is grounded, 1 if any is not

## Read the result

`--json` returns the grounding document - one entry per claim.

- **`grounded`** - true = supported, false = hallucination or contradiction
- **`score`** - confidence in the verdict
- **`support`** - where in the evidence: `matched_text` (the quote), `source_path`, `char_start`/`char_end`, `line_start`
- **`contradiction`** - the conflicting number or entity, when the claim disagrees with the source
- **`claim_location`** - where the claim sits in the answer document
- **Report back** - name the grounded vs ungrounded claims; cite the quote and location for grounded ones; for ungrounded say unsupported, or contradicted with the conflicting value

## Python

```python
from groundrails import grounding_document

doc = grounding_document(
    ["The Eiffel Tower is in Paris."],
    [("evidence.txt", "The Eiffel Tower is located in Paris, France.")],
)
for c in doc["claims"]:
    print(c["claim"], c["grounded"], c["support"])
```

## Reference

- **API, fields, CLI** - `docs/api-reference.md` in the groundrails repo
- **Design and benchmarks** - the two SOTA docs under `docs/experiments/` (`lexical-grounding-sota.md`, `semantic-grounding-sota.md`)

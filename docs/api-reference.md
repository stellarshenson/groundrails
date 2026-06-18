# API reference

The public surface of `groundrails` - the Python functions, the grounding-document shape, and the CLI. The core needs no extras; `semantic=True` / `--semantic 1` needs `groundrails[semantic-grounder]`.

## Python API

Imported from the top-level package: `from groundrails import ground, ground_batch, grounding_document, GroundingMatch, Location`.

| Symbol | Signature | Returns | Purpose |
|---|---|---|---|
| `ground` | `ground(claim, sources, *, semantic=False, ...)` | `GroundingMatch` | check one claim against the sources |
| `ground_batch` | `ground_batch(claims, sources, *, semantic=False, max_workers=5, ...)` | `list[GroundingMatch]` | check many claims |
| `grounding_document` | `grounding_document(claims, sources, **kw)` | `dict` | ground, then return the agent-facing document |
| `build_grounding_document` | `build_grounding_document(matches, claims=None, sources=None)` | `dict` | build the document from matches you already have |
| `GroundingMatch` | dataclass | - | per-claim result: `.grounded`, `.support`, `.match_type`, `.verdict_probability`, per-layer scores and locations |
| `Location` | dataclass | - | a span in a source: `source_index`, `char_start/end`, `line_start/end`, `paragraph`, `page` |
| `UnsupportedLanguageError` | exception | - | raised when a non-English claim has no installed MT model |

- **sources** - a list of strings, or `(path, text)` tuples to carry provenance into the support location
- **claims** (for `grounding_document`) - strings, or objects with `id` / `line_number` / `char_start` / `char_end` (the `Claim` objects extract-claims writes) to carry each claim's location in the answer document
- **determinism** - same input → same verdict; no sampling

## Grounding document

The dict `grounding_document` returns and `--json` prints - the business end, one entry per claim, no per-scorer internals.

| Field | Type | Meaning |
|---|---|---|
| `sources` | list[str] | evidence paths, when supplied |
| `summary` | {total, grounded, ungrounded} | per-run counts |
| `claims[].claim` | str | the claim text |
| `claims[].claim_location` | {line, char_start, char_end} or null | where the claim sits in the answer document |
| `claims[].grounded` | bool | supported (not `none` / `contradicted`) |
| `claims[].match_type` | str | winning layer: exact / fuzzy / bm25 / semantic / contradicted / none |
| `claims[].score` | float | final score - the calibrated verdict, else the combined score |
| `claims[].support` | object or null | where the support sits in the evidence (below) |
| `claims[].contradiction` | {numeric, entity} or null | value conflicts with the winning passage |

`support`:

| Field | Meaning |
|---|---|
| `source_index`, `source_path` | which evidence source |
| `matched_text` | the supporting quote |
| `char_start`, `char_end` | offset of the quote in the evidence |
| `line_start`, `line_end`, `paragraph`, `page` | its position |
| `support_via` | `lexical` when a cascade verdict fell back to the best lexical passage |

## CLI

| Command | What it does |
|---|---|
| `groundrails ground DOCUMENT EVIDENCE...` | extract claims from the one document, ground them against the evidence |
| `groundrails ground --claims FILE EVIDENCE...` | ground a structured claims file |
| `groundrails ground --claim TEXT [--claim TEXT] EVIDENCE...` | ground inline claim(s), repeatable |
| `groundrails extract-claims --document DOC` | pull claims (with their locations) from a document |
| `groundrails check-consistency --document DOC` | intra-document contradiction report |
| `groundrails config` / `download` / `setup` | print config / fetch the cascade models / first-run setup |

- **Flags** - `--json` (grounding document), `--full-output` (per-scorer detail), `--semantic 1` (add the cascade), `--effort {low,medium,high}`
- **Exit code** - 0 if every claim is grounded, 1 if any is not

## Examples

```python
from groundrails import ground, grounding_document

# one claim → a GroundingMatch
m = ground("The tower opened in 1889.", ["The Eiffel Tower opened in 1889."])
print(m.grounded, m.match_type, m.support)

# many claims → the agent-facing document
doc = grounding_document(
    ["The tower is in Paris.", "It is 2000 m tall."],
    [("evidence.txt", "The Eiffel Tower is in Paris, France. It is 330 m tall.")],
)
for c in doc["claims"]:
    print(c["claim"], c["grounded"], c["support"])
```

```bash
# default: extract claims from the answer, check against evidence, emit the document
groundrails ground answer.md evidence.txt --json

# inline claims, repeatable; positionals are evidence
groundrails ground --claim "The tower is in Paris." --claim "It is 330 m tall." evidence.txt
```

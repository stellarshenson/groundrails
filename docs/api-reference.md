# API reference

The Python surface of `groundrails` - the functions, the grounding-document shape, and the result dataclasses. The CLI lives in [`cli-reference.md`](cli-reference.md). The core needs no extras; `semantic=True` needs `groundrails[semantic-grounder]`.

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

The dict `grounding_document` returns - the business end, one entry per claim, no per-scorer internals.

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

## Examples

One claim against one source - a `GroundingMatch`:

```python
from groundrails import ground

m = ground("The tower opened in 1889.", ["The Eiffel Tower opened in 1889."])
print(m.grounded, m.match_type, m.support)
```

Many claims → the agent-facing document, with provenance carried by `(path, text)` sources:

```python
from groundrails import grounding_document

doc = grounding_document(
    ["The tower is in Paris.", "It is 2000 m tall."],
    [("evidence.txt", "The Eiffel Tower is in Paris, France. It is 330 m tall.")],
)
print(doc["summary"])  # {'total': 2, 'grounded': 1, 'ungrounded': 1}
for c in doc["claims"]:
    print(c["claim"], c["grounded"], c["support"])
```

Batch grounding, then build the document from the matches you already have:

```python
from groundrails import ground_batch, build_grounding_document

claims = ["The tower is in Paris.", "It is 330 m tall."]
sources = [("evidence.txt", "The Eiffel Tower is in Paris, France. It is 330 m tall.")]

matches = ground_batch(claims, sources, max_workers=5)
doc = build_grounding_document(matches, claims=claims, sources=sources)
```

Opt-in semantic cascade for a deeper check and cross-lingual claims (needs `groundrails[semantic-grounder]`); a claim in a language with no MT model raises:

```python
from groundrails import ground, UnsupportedLanguageError

m = ground("La tour est à Paris.", ["The Eiffel Tower is in Paris."], semantic=True)

try:
    ground("Tårnet er i Paris.", ["The Eiffel Tower is in Paris."])
except UnsupportedLanguageError as e:
    print("install the MT model first:", e)
```

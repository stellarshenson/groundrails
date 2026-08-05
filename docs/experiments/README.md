# Experiments - index

Canonical hypothesis logs and design documents for every research track in this repo.

A **track** is one research object. A **numbering pool** is one hypothesis id space. They are not the same: `SG` and `DR` are separate tracks that deliberately share one pool.

Read a track's log for the research record, its design doc for what ships.

## Track codes

The code prefixes hypothesis ids so the same number in two tracks cannot be confused. That confusion is live: `H1` names eight different hypotheses inside `LG` alone.

| Code | Track | Research object | Pool | Documents |
|---|---|---|---|---|
| `SG` | Semantic grounder | grounder quality | shared, H1-H116, global-ascending | `semantic-grounding-experiments.md`, `semantic-grounding-sota.md` |
| `DR` | Dataset refinement | training-data generation method | shares `SG`, continues from H112 | `semantic-dataset-enhancements.md` |
| `LG` | Lexical grounder | deterministic grounder quality | round-local, H1-H23 | `lexical-grounding-experiments.md`, `lexical-grounding-sota.md` |
| `JG` | Joint grounder | fusing the lexical manifold with the semantic cascade | round-local, H1-H22 | `joint-grounding-experiments.md` |
| `QZ` | NLI quantization | int8 CPU deployability of the NLI cross-encoder | no rounds, H1-H2 | `deberta-v3-quantization-experiments.md` |

`DR` was spun out of `SG` after `R10-H111` proved surrogate generation viable. It is a distinct research object with its own log and notebook folder, but its ids continue `SG`'s global sequence by design - `semantic-dataset-enhancements.md` states it directly ("Hypothesis IDs continue the main log's global numbering"), and `semantic-grounding-experiments.md` confirms it ("Hypothesis IDs remain global across both documents").

`explainer-ablative-hypothesis-testing.md` carries no track code - it is method documentation. Its `H1`-`H3` are `LG` round-6 hypotheses registered in `experiments/grounding-lexical/BENCHMARK.md`, quoted here as worked examples.

## Index

| Document | Track | Role | Canonical marker |
|---|---|---|---|
| `semantic-grounding-experiments.md` | `SG` | log | yes |
| `semantic-grounding-sota.md` | `SG` | design | yes |
| `semantic-dataset-enhancements.md` | `DR` | log | yes |
| `lexical-grounding-experiments.md` | `LG` | log | missing |
| `lexical-grounding-sota.md` | `LG` | design | missing |
| `joint-grounding-experiments.md` | `JG` | log | missing |
| `deberta-v3-quantization-experiments.md` | `QZ` | log | missing |
| `explainer-ablative-hypothesis-testing.md` | - | method | n/a |

**Role** - a *log* registers hypotheses and records verdicts, append-only. A *design* doc states what ships and cites hypothesis ids as evidence; it registers nothing.

**Registers vs cites** - a hypothesis is registered once, in one log. Every other appearance is a citation, including one log citing another track's hypothesis for provenance.

Notebooks and code artifacts follow the same codes - see `notebooks/README.md`.

## Numbering rules

- **Format** - `R<round>-H<n>`, prefixed by the track code when cited outside its own track: `SG-R10-H111`. A track with no rounds drops the segment (`QZ-H1`)
- **Counter discipline is per track** - `SG` is global-ascending and never resets; its pool is shared with `DR`, which continues it. `LG` and `JG` number per round, so `<n>` alone does not identify a hypothesis there
- **Identity** - in a global-ascending track the number is the identity and survives citation anywhere. In a round-local track the identity is the full `R<round>-H<n>`; a bare `LG-H1` is ambiguous eight ways
- **Registration is single-homed** - exactly one log registers a given id. Ownership follows the pre-registration, not the mention
- **Append-only** - a recorded verdict is immutable; later evidence is a new round that supersedes it with a back-reference
- **Renumbering is a last resort** - it has happened once (commit `3c43c00`, 2026-07-29) to restore global ordinals, and it orphaned a live citation. Prefer a back-reference

## Adding a track

1. Pick a two-letter code not already in the table above
2. Create the track's log, with the `**Canonical Experiments Document**` marker under the H1
3. Start its counter at `H1`, or continue a parent pool if the track is spun out of an existing one
4. Add the row here

## Known gaps

- Four experiment documents lack the `**Canonical Experiments Document**` / `**Canonical SOTA Document**` marker (see Index). The marker identifies the system of record beyond the filename
- The 2026-07-29 remap (commit `3c43c00`) renumbered `LG` rounds 3-4 onto `SG`'s ordinals, creating the `R3-H15` / `R3-H16` / `R3-H17` and `R4-H22` cross-track collisions, and leaving `lexical-grounding-sota.md:76` citing "round-4 H2" - now `R4-H23`, an id that no longer exists
- `LG` registers a second, parallel round series in `experiments/grounding-lexical/BENCHMARK.md` (its own rounds 2-10, ids H1-H3, H12, H13, H17, H18). Its round numbers and its `H17` collide with `lexical-grounding-experiments.md`; the two are not reconciled
- `semantic-grounding-experiments.md:390,392,394` cites `R1-H1`, `R1-H2`, `R1-H4` bare - those are `JG` ids, and `SG` has no round-1 H1-H8
- `joint-grounding-experiments.md:153,:173` record verdicts for rounds 2 and 3 with no hypothesis id, so they are uncitable until registered
- Track-code prefixes are defined here but not yet applied inside the documents; existing bare ids remain valid under the rules above

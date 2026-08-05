# Notebooks - layout and naming

One folder per research track, mirroring the track codes in `docs/experiments/README.md`. Canonical notebooks sit at the track root; per-hypothesis experiment notebooks sit under `experiments/`.

```
notebooks/<track-folder>/                    canonical notebooks - pipelines, demos, benchmarks
notebooks/<track-folder>/experiments/        one notebook per registered hypothesis or round
```

## Track folders

| Code | Track | Folder |
|---|---|---|
| `SG` | Semantic grounder | `grounding-semantic/` |
| `DR` | Dataset refinement | `dataset/` |
| `LG` | Lexical grounder | `grounding-lexical/` |
| `JG` | Joint grounder | `grounding-joint/` |
| `QZ` | NLI quantization | `nli-quantization/` |

## Naming

**Canonical** - `NN-<author>-<slug>.ipynb`, `NN` ascending per folder.

```
grounding-semantic/07-kj-grounding-sota-benchmark.ipynb
```

**Experiment** - `<TRACK>-R<round>-<author>-<slug>.ipynb`, optionally carrying the hypothesis when the notebook serves exactly one.

```
dataset/experiments/DR-R12-kj-gated-dropout-generation.ipynb
dataset/experiments/DR-R12-H114-kj-span-drop-dial.ipynb
```

- **`R<round>`, not `E<batch>`** - the whole project rounds on `R`: 704 `R<n>-H<n>` ids across the logs, and code artifacts in `experiments/` already follow it (`R8-H103_read.py`, `R4-H29_positive_control.py`). There is no `E` axis here
- **Track prefix is mandatory on experiment notebooks** - `LG`, `JG` and `QZ` all number from `H1`, so a bare `R2-H1` is ambiguous across tracks
- **The id must match a registered hypothesis** in that track's canonical log, so a notebook resolves to its pre-registration, prediction and verdict
- **Never renumber** - a notebook keeps its id after the hypothesis is superseded; supersession lives in the log, not the filename

## Rules

- **One track per folder** - a notebook that spans tracks belongs to the track whose hypothesis it tests, not the one whose data it borrows
- **Canonical vs experiment** - if it re-runs a pipeline, demonstrates the shipped design or benchmarks it, it is canonical. If it exists to answer one pre-registered hypothesis, it is an experiment
- **Code artifacts stay in `experiments/<track-folder>/`** at the repo root (`.py`, `.json`, `.parquet`); only notebooks live here

## Known gaps

- Three notebooks sit in the wrong track: `grounding-semantic/08-kj-joint-lexical-semantic-benchmark.ipynb` and `grounding-semantic/09-kj-grounding-round4-joint-premise.ipynb` belong to `JG`; `grounding-semantic/02-kj-deberta-int8-smoothquant.ipynb` belongs to `QZ`
- `grounding-joint/` and `nli-quantization/` do not exist yet
- No notebook carries a track prefix; hypothesis ids appear mid-slug instead (`03-kj-H12-maxgap-batch-experiment.ipynb`). Existing names stay valid - the convention is forward-looking
- Several notebooks are referenced by path from `docs/`, so any migration must update those references in the same change

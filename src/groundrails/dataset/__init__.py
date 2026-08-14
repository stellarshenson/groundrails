"""groundrails.dataset - the corpus preprocessing pipeline for grounding training data.

Six stages, composed and independently runnable:

    fetch        acquire a corpus from its manifest entry
    contaminate  n-gram overlap against walled corpora, with a spike control
    format       normalise into the pair schema (claim, chunk, label, doc_id, ...)
    shape        the evidence-shape gate - windows, document reuse, pair share
    assemble     build a mix from lanes plus a base, with declared assertions
    census       the pre-spend report a trainer wrapper reads before a GPU

Corpora are declared as DATA in ``corpora.yaml`` beside this package and
validated on load, so adding one is adding an entry - Python only when the
source needs bespoke download or parse logic.

`shape` is the reason the module exists. Licence and contamination say a corpus
is legal and unseen; neither says what its evidence costs to train on. A corpus
of whole articles turns 1.7% of a mix's rows into 29.65% of its training pairs,
and the loss is per pair - so the check belongs here, on CPU, before any GPU
spend, not as a trainer abort at the end.

Needs the ``dataset`` extra (``pip install groundrails[dataset]``); the import
error names it at the first call rather than at import.
"""

from __future__ import annotations

from groundrails.dataset.assemble import (
    AssembledMix,
    DropRecord,
    LaneSpec,
    MixAssertionError,
    MixSpec,
    assemble,
)
from groundrails.dataset.census import CensusReport, census
from groundrails.dataset.contaminate import (
    check as contamination_check,
)
from groundrails.dataset.contaminate import (
    gate as contamination_gate,
)
from groundrails.dataset.contaminate import (
    spike_control,
    walled_texts_from_files,
)
from groundrails.dataset.fetch import FetchResult, fetch, register_fetcher
from groundrails.dataset.format import (
    FormatError,
    LaneResult,
    evidence_texts,
    format_corpus,
    register_adapter,
    write_lane,
)
from groundrails.dataset.manifest import (
    PAIR_COLUMNS,
    CorpusEntry,
    Manifest,
    Presentation,
)
from groundrails.dataset.manifest import (
    load as load_manifest,
)
from groundrails.dataset.pipeline import Pipeline
from groundrails.dataset.shape import (
    BLOCK,
    PASS,
    PASS_WITH_DROP,
    MixProjection,
    ShapeReport,
    shape,
    shape_lane,
    window_count,
    windows,
)
from groundrails.dataset.shape import (
    table as shape_table,
)

# Importing these registers the bespoke fetchers and lane adapters the manifest
# names; without them a corpus like FActScore would resolve to nothing.
from groundrails.dataset import adapters as _adapters  # noqa: F401  isort:skip
from groundrails.dataset import fetchers as _fetchers  # noqa: F401  isort:skip

__all__ = [
    "BLOCK",
    "PAIR_COLUMNS",
    "PASS",
    "PASS_WITH_DROP",
    "AssembledMix",
    "CensusReport",
    "CorpusEntry",
    "DropRecord",
    "FetchResult",
    "FormatError",
    "LaneResult",
    "LaneSpec",
    "Manifest",
    "MixAssertionError",
    "MixProjection",
    "MixSpec",
    "Pipeline",
    "Presentation",
    "ShapeReport",
    "assemble",
    "census",
    "contamination_check",
    "contamination_gate",
    "evidence_texts",
    "fetch",
    "format_corpus",
    "load_manifest",
    "register_adapter",
    "register_fetcher",
    "shape",
    "shape_lane",
    "shape_table",
    "spike_control",
    "walled_texts_from_files",
    "window_count",
    "windows",
    "write_lane",
]

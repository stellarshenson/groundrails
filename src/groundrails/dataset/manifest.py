"""Corpus manifest - the pipeline's single source of truth, declared as data.

Every corpus the pipeline knows is one entry in ``corpora.yaml`` beside this
module: where it comes from, the licence tag read at the source, how its
columns map onto the pair schema, what row counts to expect, and the
presentation constraints the shape stage rules on. Adding a corpus is adding an
entry - Python is needed only when the source itself needs bespoke download or
parse logic, and then only for that one function.

The manifest is validated on load. A missing field, an unknown key or an
inconsistent block (an ``adapter`` source with no adapter named, a format block
with no claim column) fails here, loudly, rather than half way through a fetch.

Load order: the packaged ``corpora.yaml`` by default, or any path passed to
:func:`load` - so a project can carry its own manifest without forking the
package.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PACKAGE_MANIFEST = Path(__file__).parent / "corpora.yaml"

#: The canonical pair-schema columns every formatted lane carries, in order.
PAIR_COLUMNS = ("pair_id", "claim", "chunk", "label", "doc_id", "source", "tag")


class _Strict(BaseModel):
    """Reject unknown keys - a typo in the manifest is an error, not a silent default."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Licence(_Strict):
    """The licence as read at the source, not as remembered from a survey."""

    tag: str
    commercial_use: bool
    verified: str


class Source(_Strict):
    """Where the corpus comes from and how it is pulled.

    ``kind`` selects the fetcher:
        hf_dataset  ``load_dataset`` over each id in ``repos`` (and ``subsets``
                    if named) - several ids when one corpus ships per-language
        hf_files    named files inside the single Hub dataset repo ``repo``;
                    each value is a filename or a ``prefix*suffix`` glob
                    resolved against the repo's file list
        http_zip    a zip at ``url``; ``keep_prefixes`` selects members
        adapter     a registered fetcher function named by ``fetcher``
    """

    kind: Literal["hf_dataset", "hf_files", "http_zip", "adapter"]
    repo: str = ""
    repos: tuple[str, ...] = ()
    subsets: tuple[str, ...] = ()
    files: dict[str, str] = Field(default_factory=dict)
    url: str = ""
    keep_prefixes: tuple[str, ...] = ()
    fetcher: str = ""
    #: column -> values dropped AT FETCH (a licence carve-out, applied by
    #: construction so the excluded rows never enter the archive).
    drop_where: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> Source:
        need = {
            "hf_dataset": ("repos",),
            "hf_files": ("repo", "files"),
            "http_zip": ("url",),
            "adapter": ("fetcher",),
        }[self.kind]
        missing = [f for f in need if not getattr(self, f)]
        if missing:
            raise ValueError(f"source kind {self.kind!r} needs {missing}")
        return self


class ChunkSpec(_Strict):
    """The evidence column. ``join`` concatenates a list-of-strings column."""

    column: str
    join: str = ""


class LabelSpec(_Strict):
    """The label column and how its values become 0/1.

    Exactly one of ``map`` (value -> label, unmapped rows dropped), ``truthy``
    (any truthy value -> 1) or neither (the column is already 0/1).
    """

    column: str
    map: dict[str, int] = Field(default_factory=dict)
    truthy: bool = False

    @model_validator(mode="after")
    def _one_mode(self) -> LabelSpec:
        if self.map and self.truthy:
            raise ValueError("label: choose `map` or `truthy`, not both")
        if any(v not in (0, 1) for v in self.map.values()):
            raise ValueError("label: map values must be 0 or 1")
        return self


class DocIdSpec(_Strict):
    """The grouping key. Either a column, or a blake2b hash of another column.

    The document id is what the shape stage counts reuse over and what a
    document-disjoint split groups on, so a corpus whose rows share one
    document must say so here rather than get one id per row.
    """

    column: str = ""
    hash: str = ""
    prefix: str = ""

    @model_validator(mode="after")
    def _one_source(self) -> DocIdSpec:
        if bool(self.column) == bool(self.hash):
            raise ValueError("doc_id: set exactly one of `column` or `hash`")
        return self


class FormatSpec(_Strict):
    """How the fetched frames become the pair schema.

    Declarative for the common case - name the claim, evidence and label
    columns and the pipeline does the rest. ``adapter`` names a registered
    function instead, for corpora whose rows have to be parsed out of tagged
    text or walked across a directory tree.
    """

    adapter: str = ""
    claim: str = ""
    chunk: ChunkSpec | None = None
    label: LabelSpec | None = None
    doc_id: DocIdSpec | None = None
    #: the `source` column: the split/part name the row came from, or a column
    source_from_split: bool = True
    source_column: str = ""
    #: provenance columns carried through unchanged
    retain: tuple[str, ...] = ()
    #: provenance columns carried through under a new name (new -> source column),
    #: which is how a raw verdict survives the column that becomes the 0/1 label
    retain_as: dict[str, str] = Field(default_factory=dict)
    drop_if_empty: tuple[str, ...] = ("claim", "chunk")

    @model_validator(mode="after")
    def _complete(self) -> FormatSpec:
        if self.adapter:
            return self
        missing = [f for f in ("claim", "chunk", "label", "doc_id") if not getattr(self, f)]
        if missing:
            raise ValueError(f"format: declarative block needs {missing} (or name an `adapter`)")
        return self


class Expected(_Strict):
    """Counts a fetch or format must reproduce; ``None`` means unasserted."""

    rows: int | None = None
    positives: int | None = None
    documents: int | None = None


class Presentation(_Strict):
    """How the trainer will present this corpus, and the bars the shape stage rules on.

    The window geometry must match the trainer's own, or the shape stage
    measures a presentation nobody trains on. The bars are what turn a
    measurement into a verdict: rows per document catches a corpus that reuses
    a handful of documents everywhere, pair share catches one that captures the
    gradient out of proportion to its rows, and the over-cap fraction separates
    a droppable tail from a structural misfit.

    Capture is DISPROPORTION, not size. A group holding half a mix's pairs
    because it holds half its rows is not capturing anything, so the pair-share
    bar fires only when the share is both over ``max_pair_share`` and over
    ``max_pair_share_amplification`` times the group's row share. With no row
    context the absolute bar rules alone.
    """

    window_chars: int = 1500
    stride_chars: int = 750
    pairs_per_batch: int = 96
    max_rows_per_document: float = 20.0
    max_pair_share: float = 0.15
    max_pair_share_amplification: float = 3.0
    max_over_cap_fraction: float = 0.02


class CorpusEntry(_Strict):
    """One corpus, end to end."""

    name: str
    title: str
    licence: Licence
    source: Source
    #: `archive` stages parquets then zips them; `tree` lands an extracted
    #: directory (a GitHub checkout, a per-topic evidence pull).
    storage: Literal["archive", "tree"] = "archive"
    #: how the corpus maps onto (claim, evidence) -> supported, in one line
    task_shape: str
    expected: Expected = Field(default_factory=Expected)
    format: FormatSpec | None = None
    presentation: Presentation = Field(default_factory=Presentation)
    notes: str = ""


class Manifest(_Strict):
    """The whole manifest, keyed by corpus name."""

    version: int
    corpora: tuple[CorpusEntry, ...]

    @model_validator(mode="after")
    def _unique_names(self) -> Manifest:
        seen = [c.name for c in self.corpora]
        dupes = sorted({n for n in seen if seen.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate corpus names in the manifest: {dupes}")
        return self

    def get(self, name: str) -> CorpusEntry:
        """The entry registered under ``name``; the error names the known corpora."""
        for c in self.corpora:
            if c.name == name:
                return c
        raise KeyError(f"unknown corpus {name!r}; known: {', '.join(self.names())}")

    def names(self) -> list[str]:
        """Corpus names, in manifest order."""
        return [c.name for c in self.corpora]


def load(path: Path | str | None = None) -> Manifest:
    """Read and validate a manifest; ``None`` loads the packaged ``corpora.yaml``."""
    import yaml

    p = Path(path) if path else PACKAGE_MANIFEST
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{p} is not a yaml mapping")
    return Manifest.model_validate(raw)


@lru_cache(maxsize=1)
def packaged() -> Manifest:
    """The packaged manifest, parsed once per process."""
    return load()


def get(name: str) -> CorpusEntry:
    """One entry from the packaged manifest."""
    return packaged().get(name)


def names() -> list[str]:
    """Corpus names in the packaged manifest."""
    return packaged().names()

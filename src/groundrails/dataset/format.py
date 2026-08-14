"""Format stage - normalise a fetched corpus into the pair schema.

Every lane looks the same downstream: ``pair_id, claim, chunk, label, doc_id,
source, tag`` first, then whatever provenance columns that corpus is worth
keeping. One shape means the shape, assemble and census stages never learn a
corpus's private layout.

Most corpora need no code: the manifest's ``format`` block names the claim,
evidence and label columns and the mapping is applied here. A corpus whose rows
have to be parsed out of tagged text, or walked across a directory tree,
registers one adapter function instead - the exception, not the pattern.

Two invariants the stage enforces rather than assumes: ``doc_id`` is the real
grouping key (rows sharing a document must share an id, or every later
document-disjoint split is a lie), and the manifest's expected counts must be
reproduced, so a silently changed upstream is caught here instead of showing up
as an unexplained metric later.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

from groundrails.dataset import manifest as manifest_mod
from groundrails.dataset._deps import polars
from groundrails.dataset.fetch import read_fetched
from groundrails.dataset.manifest import PAIR_COLUMNS, CorpusEntry

#: Bespoke row builders, keyed by the manifest's ``format.adapter``. An adapter
#: takes ``(entry, data_dir)`` and returns ``(rows, stats)`` - rows being dicts
#: carrying at least claim / chunk / label / doc_id / source.
ADAPTERS: dict[str, Callable[[CorpusEntry, Path], tuple[list[dict], dict]]] = {}


class FormatError(RuntimeError):
    """A corpus cannot be formatted, or did not reproduce its manifest counts."""


def register_adapter(name: str):
    """Register a bespoke row builder under ``name`` (the manifest's ``format.adapter``)."""

    def deco(fn):
        ADAPTERS[name] = fn
        return fn

    return deco


@dataclass
class LaneResult:
    """A formatted lane and the record of how it was built."""

    name: str
    frame: object  # polars.DataFrame
    stats: dict = field(default_factory=dict)
    integrity: dict = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return self.frame.height

    def manifest(self, entry: CorpusEntry) -> dict:
        """The lane manifest written beside the parquet."""
        df = self.frame
        return {
            "corpus": self.name,
            "title": entry.title,
            "licence": entry.licence.tag,
            "task_shape": entry.task_shape,
            "rows": df.height,
            "documents": int(df["doc_id"].n_unique()),
            "label_distribution": {str(k): v for k, v in df.group_by("label").len().iter_rows()},
            "source_distribution": {str(k): v for k, v in df.group_by("source").len().iter_rows()},
            "provenance_columns": [c for c in df.columns if c not in PAIR_COLUMNS],
            "build_stats": self.stats,
            "integrity": self.integrity,
        }


def _blake(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def _declarative_rows(entry: CorpusEntry, parts: dict) -> tuple[list[dict], dict]:
    """Apply a manifest ``format`` block to each fetched split."""
    fmt = entry.format
    rows: list[dict] = []
    dropped_empty = 0
    dropped_unmapped = 0
    rows_in = 0

    for split, df in parts.items():
        rows_in += df.height
        cols = set(df.columns)
        for r in df.iter_rows(named=True):
            claim = (r.get(fmt.claim) or "").strip()
            raw_chunk = r.get(fmt.chunk.column)
            if fmt.chunk.join:
                raw_chunk = fmt.chunk.join.join(x for x in (raw_chunk or []) if x)
            chunk = (raw_chunk or "").strip()

            raw_label = r.get(fmt.label.column)
            if fmt.label.map:
                label = fmt.label.map.get(str(raw_label).strip())
            elif fmt.label.truthy:
                label = 1 if raw_label else 0
            else:
                label = None if raw_label is None else int(raw_label)
            if label is None:
                dropped_unmapped += 1
                continue

            values = {"claim": claim, "chunk": chunk, "label": label}
            if any(not values.get(c, r.get(c)) for c in fmt.drop_if_empty):
                dropped_empty += 1
                continue

            if fmt.doc_id.column:
                doc_id = f"{fmt.doc_id.prefix}{r.get(fmt.doc_id.column)}"
            else:
                doc_id = _blake(values.get(fmt.doc_id.hash, chunk))
            source = split if fmt.source_from_split else str(r.get(fmt.source_column) or "")

            row = {**values, "doc_id": doc_id, "source": source}
            for c in fmt.retain:
                if c in cols:
                    row[c] = r.get(c)
            for new, old in fmt.retain_as.items():
                if old in cols:
                    row[new] = r.get(old)
            rows.append(row)

    return rows, {
        "rows_in": rows_in,
        "dropped_empty": dropped_empty,
        "dropped_unmapped_label": dropped_unmapped,
    }


def finalize(rows: list[dict], tag: str):
    """Rows -> the canonical lane frame: deduplicated, pair-indexed, tagged."""
    pl = polars()
    df = pl.DataFrame(rows, infer_schema_length=None)
    df = df.unique(subset=["claim", "chunk", "label"], keep="first", maintain_order=True)
    df = df.with_row_index("pair_id").with_columns(
        pl.col("pair_id").cast(pl.Int64),
        pl.col("label").cast(pl.Int64),
        pl.lit(tag).alias("tag"),
    )
    rest = [c for c in df.columns if c not in PAIR_COLUMNS]
    return df.select(list(PAIR_COLUMNS) + rest)


def integrity(df) -> dict:
    """The checks a lane must pass by construction, not by inspection."""
    labels = set(df["label"].to_list())
    empty_claims = int(sum(not c for c in df["claim"].to_list()))
    empty_chunks = int(sum(not c for c in df["chunk"].to_list()))
    dup = df.height - df.unique(subset=["claim", "chunk", "label"]).height
    return {
        "labels_in_01": labels <= {0, 1},
        "empty_claims": empty_claims,
        "empty_chunks": empty_chunks,
        "duplicate_rows": int(dup),
        "distinct_claims": int(df["claim"].n_unique()),
        "distinct_chunks": int(df["chunk"].n_unique()),
        "distinct_documents": int(df["doc_id"].n_unique()),
        "pass": bool(labels <= {0, 1} and not empty_claims and not empty_chunks and dup == 0),
    }


def format_corpus(
    name: str | CorpusEntry,
    data_dir: Path | str,
    *,
    strict: bool = True,
) -> LaneResult:
    """Format one fetched corpus into its lane frame.

    ``strict`` holds the manifest's expected row / positive / document counts;
    turn it off only when deliberately working against a changed upstream.
    """
    entry = manifest_mod.get(name) if isinstance(name, str) else name
    if entry.format is None:
        raise FormatError(
            f"corpus {entry.name!r} has no `format` block in the manifest - declare the claim, "
            "chunk, label and doc_id columns, or name an adapter"
        )
    data_dir = Path(data_dir)

    if entry.format.adapter:
        adapter = ADAPTERS.get(entry.format.adapter)
        if adapter is None:
            raise FormatError(
                f"corpus {entry.name!r} names adapter {entry.format.adapter!r}, which is not "
                f"registered; known: {', '.join(sorted(ADAPTERS)) or '(none)'}"
            )
        rows, stats = adapter(entry, data_dir)
    else:
        rows, stats = _declarative_rows(entry, read_fetched(entry, data_dir))

    if not rows:
        raise FormatError(f"corpus {entry.name!r} produced no rows")

    df = finalize(rows, entry.name)
    result = LaneResult(entry.name, df, stats, integrity(df))

    got = {
        "rows": df.height,
        "positives": int(df["label"].sum()),
        "documents": int(df["doc_id"].n_unique()),
    }
    want = {k: v for k, v in entry.expected.model_dump().items() if v is not None}
    mismatch = {k: (v, got[k]) for k, v in want.items() if got[k] != v}
    result.stats["expected_vs_observed"] = {
        k: {"want": v[0], "got": v[1]} for k, v in mismatch.items()
    }
    if mismatch and strict:
        detail = ", ".join(f"{k}: want {w}, got {g}" for k, (w, g) in mismatch.items())
        raise FormatError(f"{entry.name} does not reproduce its manifest counts ({detail})")
    if not result.integrity["pass"] and strict:
        raise FormatError(f"{entry.name} failed lane integrity: {result.integrity}")
    return result


def evidence_texts(name: str | CorpusEntry, data_dir: Path | str) -> list[str]:
    """The corpus's deduplicated evidence texts - the contamination stage's gate units.

    Contamination is a DOCUMENT-overlap property, so the units are the evidence
    side, deduplicated: a corpus that asks 40 claims about one article must not
    count that article 40 times. Read straight from the manifest's chunk column
    where the corpus is declarative, and from the adapter's rows where it is not.
    """
    entry = manifest_mod.get(name) if isinstance(name, str) else name
    if entry.format is None:
        raise FormatError(f"corpus {entry.name!r} has no `format` block to read evidence from")
    if entry.format.adapter:
        rows, _ = ADAPTERS[entry.format.adapter](entry, Path(data_dir))
        return sorted({r["chunk"] for r in rows if r.get("chunk")})
    col = entry.format.chunk
    texts: set[str] = set()
    for df in read_fetched(entry, data_dir).values():
        if col.column not in df.columns:
            continue
        for v in df[col.column].to_list():
            text = col.join.join(x for x in (v or []) if x) if col.join else (v or "")
            if text and text.strip():
                texts.add(text.strip())
    return sorted(texts)


def write_lane(result: LaneResult, entry: CorpusEntry, out_dir: Path | str) -> Path:
    """Write ``<name>_lane.parquet`` plus its manifest JSON; return the parquet path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{result.name}_lane.parquet"
    result.frame.write_parquet(path)
    (out / f"{result.name}_lane_manifest.json").write_text(
        json.dumps(result.manifest(entry), indent=2), encoding="utf-8"
    )
    return path

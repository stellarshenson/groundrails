"""Assemble stage - build a training mix from lanes and a base.

A mix is a set of formatted lanes, optionally on top of a base mix, carrying a
``group`` column that names where each row came from. The group map is not
decoration: a per-group loss, a per-group read, and the census all key on it.

Everything the stage asserts is declared up front - each lane's expected rows
and positives, the mix's expected total, the exact group map. An assembly that
does not reproduce them raises rather than training on a mix nobody described.
The one thing it is allowed to do quietly is nothing: over-cap rows are dropped
only when the mix says so, and every drop is recorded with its window counts.

Lanes keep their provenance columns in their own parquets; the assembled mix
carries the pair schema plus ``group``, because that is what a trainer reads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from groundrails.dataset._deps import polars
from groundrails.dataset.manifest import Presentation
from groundrails.dataset.shape import _window_counts

MIX_COLUMNS = ("pair_id", "claim", "chunk", "label", "doc_id", "source", "tag", "group")


class MixAssertionError(RuntimeError):
    """A mix did not reproduce what its spec declared."""


@dataclass(frozen=True)
class LaneSpec:
    """One lane in a mix, with the counts it must reproduce."""

    path: Path | str
    group: str
    expected_rows: int | None = None
    expected_positives: int | None = None


@dataclass(frozen=True)
class MixSpec:
    """A mix, declared before it is built."""

    name: str
    lanes: tuple[LaneSpec, ...]
    base: Path | str | None = None
    base_group: str = "base"
    expected_rows: int | None = None
    expected_groups: tuple[str, ...] = ()
    presentation: Presentation = field(default_factory=Presentation)
    #: drop rows whose evidence windows exceed the batch cap - the trainer's own
    #: rule, applied here where it can be counted instead of at the first batch
    drop_over_cap: bool = True


@dataclass
class DropRecord:
    """A documented over-cap drop for one group."""

    group: str
    rows_in: int
    rows_dropped: int
    rows_kept: int
    cap: int
    dropped_pairs: int
    dropped_window_counts: list = field(default_factory=list)
    rule: str = "windows(chunk) > pairs_per_batch, the trainer's own batch-cap threshold"


@dataclass
class AssembledMix:
    """The built mix, its group map, and every drop taken to build it."""

    name: str
    frame: object  # polars.DataFrame
    groups: tuple[str, ...]
    drops: dict = field(default_factory=dict)
    presentation: Presentation = field(default_factory=Presentation)

    @property
    def rows(self) -> int:
        return self.frame.height

    def to_dict(self) -> dict:
        return {
            "mix": self.name,
            "rows": self.rows,
            "groups": list(self.groups),
            "group_rows": {str(k): v for k, v in self.frame.group_by("group").len().iter_rows()},
            "drops": {k: asdict(v) for k, v in self.drops.items()},
            "presentation": self.presentation.model_dump(),
        }


def _read_lane(path: Path | str, group: str):
    pl = polars()
    df = pl.read_parquet(path)
    missing = [c for c in ("claim", "chunk", "label", "doc_id") if c not in df.columns]
    if missing:
        raise MixAssertionError(f"lane {path} is not in the pair schema: missing {missing}")
    if "source" not in df.columns:
        df = df.with_columns(pl.lit("").alias("source"))
    if "tag" not in df.columns:
        df = df.with_columns(pl.lit(group).alias("tag"))
    return df.select("claim", "chunk", "label", "doc_id", "source", "tag").with_columns(
        pl.lit(group).alias("group")
    )


def assemble(spec: MixSpec) -> AssembledMix:
    """Build the mix, asserting everything the spec declared."""
    pl = polars()
    frames = []
    drops: dict[str, DropRecord] = {}

    if spec.base is not None:
        frames.append(_read_lane(spec.base, spec.base_group))

    for lane in spec.lanes:
        df = _read_lane(lane.path, lane.group)
        rows, positives = df.height, int(df["label"].sum())
        if lane.expected_rows is not None and rows != lane.expected_rows:
            raise MixAssertionError(
                f"lane {lane.group}: {rows} rows, spec says {lane.expected_rows}"
            )
        if lane.expected_positives is not None and positives != lane.expected_positives:
            raise MixAssertionError(
                f"lane {lane.group}: {positives} positives, spec says {lane.expected_positives}"
            )

        if spec.drop_over_cap:
            lengths = np.asarray(df["chunk"].str.len_chars().to_numpy(), dtype=np.int64)
            counts = _window_counts(lengths, spec.presentation)
            over = counts > spec.presentation.pairs_per_batch
            if over.any():
                drops[lane.group] = DropRecord(
                    group=lane.group,
                    rows_in=rows,
                    rows_dropped=int(over.sum()),
                    rows_kept=rows - int(over.sum()),
                    cap=spec.presentation.pairs_per_batch,
                    dropped_pairs=int(counts[over].sum()),
                    dropped_window_counts=sorted((int(c) for c in counts[over]), reverse=True),
                )
                df = df.filter(pl.Series(~over))
        frames.append(df)

    mix = pl.concat(frames, how="vertical").with_row_index("pair_id")
    mix = mix.with_columns(pl.col("pair_id").cast(pl.Int64)).select(MIX_COLUMNS)
    groups = tuple(sorted(set(mix["group"].to_list())))

    if spec.expected_groups and groups != tuple(sorted(spec.expected_groups)):
        raise MixAssertionError(
            f"group map {groups} != declared {tuple(sorted(spec.expected_groups))}"
        )
    if spec.expected_rows is not None and mix.height != spec.expected_rows:
        raise MixAssertionError(f"mix {mix.height} rows, spec says {spec.expected_rows}")

    return AssembledMix(spec.name, mix, groups, drops, spec.presentation)

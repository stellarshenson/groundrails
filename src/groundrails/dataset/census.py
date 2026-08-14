"""Census stage - the pre-spend report over an assembled mix.

This is what a trainer wrapper calls before it touches a GPU. It answers one
question per group and one for the mix: how much of the training does this group
actually get, and is anything in here shaped so badly that the run should not
start.

Per group it reports rows AND pairs AND pair share - the three diverge whenever
evidence is long, and reading rows alone is exactly how a corpus at 1.7% of the
rows ends up owning 29.65% of the gradient. It also reports mean target (the
positive rate the group teaches), the window census, and each group's shape
verdict.

The mix-level answer is a go/no-go. No-go names the groups responsible; go means
the presentation was measured, not assumed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from groundrails.dataset._deps import polars
from groundrails.dataset.assemble import AssembledMix
from groundrails.dataset.manifest import Presentation
from groundrails.dataset.shape import BLOCK, MixProjection, ShapeReport, _window_counts, shape


@dataclass
class CensusReport:
    """Rows, pairs, share and shape for every group in a mix, plus the verdict."""

    mix: str
    rows: int
    pairs: int
    groups: list = field(default_factory=list)
    mean_target: float = 0.0
    multi_window_share: float = 0.0
    mean_windows: float = 0.0
    max_windows: int = 0
    over_cap_rows: int = 0
    go: bool = True
    blocking: list = field(default_factory=list)
    presentation: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["groups"] = [g if isinstance(g, dict) else g.to_dict() for g in self.groups]
        return d


def census(
    mix: AssembledMix | object,
    presentation: Presentation | None = None,
    *,
    name: str = "mix",
    group_col: str = "group",
) -> CensusReport:
    """Census an assembled mix (or any frame carrying a group column)."""
    pl = polars()
    if isinstance(mix, AssembledMix):
        frame, presentation, name = mix.frame, mix.presentation, mix.name
    else:
        frame = mix
        presentation = presentation or Presentation()

    lengths = np.asarray(frame["chunk"].str.len_chars().to_numpy(), dtype=np.int64)
    counts = _window_counts(lengths, presentation)
    total_pairs = int(counts.sum())
    total_rows = frame.height

    reports: list[ShapeReport] = []
    for group in sorted(set(frame[group_col].to_list())):
        sub = frame.filter(pl.col(group_col) == group)
        sub_pairs = int(
            _window_counts(
                np.asarray(sub["chunk"].str.len_chars().to_numpy(), dtype=np.int64), presentation
            ).sum()
        )
        reports.append(
            shape(
                sub,
                presentation,
                name=group,
                projection=MixProjection(
                    name=name,
                    other_pairs=total_pairs - sub_pairs,
                    other_rows=total_rows - sub.height,
                ),
            )
        )

    blocking = [r.name for r in reports if r.verdict == BLOCK]
    over_cap = int((counts > presentation.pairs_per_batch).sum())
    return CensusReport(
        mix=name,
        rows=total_rows,
        pairs=total_pairs,
        groups=reports,
        mean_target=round(float(frame.select(pl.col("label").mean()).item()), 4),
        multi_window_share=round(float((counts > 1).mean()) if total_rows else 0.0, 4),
        mean_windows=round(float(counts.mean()) if total_rows else 0.0, 4),
        max_windows=int(counts.max()) if total_rows else 0,
        over_cap_rows=over_cap,
        go=not blocking and over_cap == 0,
        blocking=blocking + ([f"{over_cap} rows over the batch cap"] if over_cap else []),
        presentation=presentation.model_dump(),
    )


def census_lane_dir(
    lane_paths: dict[str, Path | str],
    presentation: Presentation | None = None,
    *,
    name: str = "mix",
) -> CensusReport:
    """Census lanes that have not been assembled yet - ``group -> lane parquet``."""
    pl = polars()
    presentation = presentation or Presentation()
    frames = [
        pl.read_parquet(p)
        .select("claim", "chunk", "label", "doc_id")
        .with_columns(pl.lit(group).alias("group"))
        for group, p in lane_paths.items()
    ]
    return census(pl.concat(frames, how="vertical"), presentation, name=name)

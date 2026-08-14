"""Shape stage - the evidence-shape gate, run BEFORE any training spend.

A corpus is not admissible because it passed a licence check and a
contamination gate. Those two say it is legal and unseen; neither says anything
about the SHAPE of its evidence, and shape is what decides how much of a
training run a corpus actually takes.

Two failures this stage exists to catch, both observed on real banked supply:

* a corpus of whole articles - 30k characters of evidence per row, 40 windows
  per row against a 96-pair batch cap - where a handful of documents back every
  row, so a document-disjoint split has almost nothing to split on
* a corpus that captures the gradient out of proportion to its rows: 1.7% of the
  mix's rows contributing 29.65% of its training pairs, because the loss is
  per-PAIR and long evidence multiplies into windows

Both are invisible at row level. Both are obvious here, in seconds, on CPU.

Three verdicts:
    PASS            admissible as presented
    PASS-WITH-DROP  admissible once N named over-cap rows are dropped - a
                    documented drop, never a silent filter
    BLOCK           structural: document reuse or pair-share capture beyond the
                    bar, or an over-cap tail too large to call a tail

The window geometry and the three bars come from the corpus's manifest
``presentation`` block, so the stage measures the presentation the trainer will
actually use rather than a second opinion about it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from groundrails.dataset._deps import polars
from groundrails.dataset.manifest import Presentation

PASS = "PASS"
PASS_WITH_DROP = "PASS-WITH-DROP"
BLOCK = "BLOCK"


@dataclass(frozen=True)
class MixProjection:
    """The mix a lane is being projected into.

    ``other_pairs`` is the training pairs contributed by everything ELSE in the
    mix, so the projected share is honest whether the lane is already a member
    (pass the mix total minus its own) or a candidate for admission (pass the
    whole mix). ``other_rows`` is the same for rows - ``None`` means the row
    context is unknown, which is different from zero (the lane IS the mix).
    """

    name: str
    other_pairs: int
    other_rows: int | None = None


def windows(text: str, presentation: Presentation) -> list[str]:
    """Sliding windows over evidence text; the final window flushes to the end."""
    win, stride = presentation.window_chars, presentation.stride_chars
    n = len(text)
    if n <= win:
        return [text]
    starts = list(range(0, n - win + 1, stride))
    if starts[-1] + win < n:
        starts.append(n - win)
    return [text[s : s + win] for s in starts]


def window_count(n_chars: int, presentation: Presentation) -> int:
    """How many windows a chunk of ``n_chars`` produces.

    The length-only equivalent of ``len(windows(text))`` - identical by
    construction, since the splitter reads nothing but the length, and fast
    enough to census a million rows.
    """
    win, stride = presentation.window_chars, presentation.stride_chars
    if n_chars <= win:
        return 1
    n_starts = (n_chars - win) // stride + 1
    last = (n_starts - 1) * stride
    return n_starts + (1 if last + win < n_chars else 0)


def _window_counts(lengths: np.ndarray, presentation: Presentation) -> np.ndarray:
    """Vectorised :func:`window_count` over an array of chunk lengths."""
    win, stride = presentation.window_chars, presentation.stride_chars
    n_starts = np.where(lengths <= win, 1, (np.maximum(lengths - win, 0)) // stride + 1)
    last = (n_starts - 1) * stride
    extra = (lengths > win) & (last + win < lengths)
    return (n_starts + extra).astype(np.int64)


def _dist(values: np.ndarray) -> dict:
    """mean / median / p90 / max of a distribution, rounded for reading."""
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0}
    return {
        "mean": round(float(values.mean()), 4),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": int(values.max()),
    }


@dataclass
class ShapeReport:
    """What a lane's evidence costs at training time, and the verdict on it."""

    name: str
    verdict: str
    rows: int
    pairs: int
    documents: int
    rows_per_document: float
    windows: dict = field(default_factory=dict)
    multi_window_share: float = 0.0
    over_cap_rows: int = 0
    over_cap_pairs: int = 0
    over_cap_fraction: float = 0.0
    over_cap_window_counts: list = field(default_factory=list)
    rows_after_drop: int = 0
    pairs_after_drop: int = 0
    claim_chars: dict = field(default_factory=dict)
    chunk_chars: dict = field(default_factory=dict)
    positive_fraction: float = 0.0
    projected_pair_share: float | None = None
    projected_row_share: float | None = None
    #: pair share divided by row share - how far a group's gradient runs ahead
    #: of its size; 1.0 is proportionate, and the bar is on this, not on size
    pair_share_amplification: float | None = None
    mix: str = ""
    presentation: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.verdict == BLOCK

    def to_dict(self) -> dict:
        return asdict(self)


def shape(
    frame,
    presentation: Presentation,
    *,
    name: str,
    projection: MixProjection | None = None,
    chunk_col: str = "chunk",
    claim_col: str = "claim",
    label_col: str = "label",
    doc_col: str = "doc_id",
    max_reported_over_cap: int = 32,
) -> ShapeReport:
    """Census one formatted lane's evidence shape and rule on it."""
    pl = polars()
    cap = presentation.pairs_per_batch

    lengths = frame[chunk_col].str.len_chars().to_numpy()
    counts = _window_counts(np.asarray(lengths, dtype=np.int64), presentation)
    over = counts > cap
    documents = int(frame[doc_col].n_unique())
    rows = frame.height
    pairs = int(counts.sum())

    over_counts = sorted((int(c) for c in counts[over]), reverse=True)
    rows_per_doc = round(rows / max(documents, 1), 4)
    over_fraction = round(float(over.mean()) if rows else 0.0, 6)

    share = row_share = None
    if projection is not None:
        share = round(pairs / max(projection.other_pairs + pairs, 1), 6)
        if projection.other_rows is not None:
            row_share = round(rows / max(projection.other_rows + rows, 1), 6)

    reasons: list[str] = []
    if rows_per_doc > presentation.max_rows_per_document:
        reasons.append(
            f"document reuse: {rows_per_doc:.1f} rows per document over the "
            f"{presentation.max_rows_per_document:g} bar ({rows} rows over {documents} documents)"
        )
    if share is not None and share > presentation.max_pair_share:
        # capture is disproportion: a group is only capturing when its share of
        # the pairs runs far ahead of its share of the rows
        amplification = share / row_share if row_share else None
        if amplification is None:
            reasons.append(
                f"pair-share capture: {share:.2%} of {projection.name} over the "
                f"{presentation.max_pair_share:.0%} bar"
            )
        elif amplification > presentation.max_pair_share_amplification:
            reasons.append(
                f"pair-share capture: {share:.2%} of {projection.name}'s pairs from "
                f"{row_share:.2%} of its rows ({amplification:.1f}x, over the "
                f"{presentation.max_pair_share_amplification:g}x bar at "
                f"{presentation.max_pair_share:.0%}+ share)"
            )
    if over_fraction > presentation.max_over_cap_fraction:
        reasons.append(
            f"over-cap tail: {over_fraction:.2%} of rows exceed the {cap}-pair batch cap, over "
            f"the {presentation.max_over_cap_fraction:.0%} droppable bar"
        )

    if reasons:
        verdict = BLOCK
    elif over_counts:
        verdict = PASS_WITH_DROP
        reasons.append(f"drop {len(over_counts)} rows over the {cap}-pair batch cap")
    else:
        verdict = PASS

    return ShapeReport(
        name=name,
        verdict=verdict,
        rows=rows,
        pairs=pairs,
        documents=documents,
        rows_per_document=rows_per_doc,
        windows=_dist(counts),
        multi_window_share=round(float((counts > 1).mean()) if rows else 0.0, 4),
        over_cap_rows=len(over_counts),
        over_cap_pairs=int(counts[over].sum()),
        over_cap_fraction=over_fraction,
        over_cap_window_counts=over_counts[:max_reported_over_cap],
        rows_after_drop=rows - len(over_counts),
        pairs_after_drop=pairs - int(counts[over].sum()),
        claim_chars=_dist(np.asarray(frame[claim_col].str.len_chars().to_numpy(), dtype=np.int64)),
        chunk_chars=_dist(np.asarray(lengths, dtype=np.int64)),
        positive_fraction=round(float(frame.select(pl.col(label_col).mean()).item()), 4),
        projected_pair_share=share,
        projected_row_share=row_share,
        pair_share_amplification=(
            round(share / row_share, 2) if share is not None and row_share else None
        ),
        mix=projection.name if projection else "",
        presentation=presentation.model_dump(),
        reasons=reasons,
    )


def shape_lane(
    path: Path | str,
    presentation: Presentation | None = None,
    *,
    name: str = "",
    projection: MixProjection | None = None,
    **kwargs,
) -> ShapeReport:
    """Census a lane parquet on disk (read-only)."""
    pl = polars()
    p = Path(path)
    return shape(
        pl.read_parquet(p),
        presentation or Presentation(),
        name=name or p.stem,
        projection=projection,
        **kwargs,
    )


HEADERS = (
    "lane",
    "verdict",
    "rows",
    "pairs",
    "win/row",
    "p90",
    "max",
    "over-cap",
    "docs",
    "rows/doc",
    "pos",
    "pair share",
    "amp",
)


def table(reports: list[ShapeReport]) -> str:
    """A fixed-width table over several reports - the pre-spend read at a glance."""
    rows = [
        (
            r.name,
            r.verdict,
            f"{r.rows:,}",
            f"{r.pairs:,}",
            f"{r.windows['mean']:.2f}",
            f"{r.windows['p90']:.0f}",
            f"{r.windows['max']:,}",
            f"{r.over_cap_rows:,}",
            f"{r.documents:,}",
            f"{r.rows_per_document:.2f}",
            f"{r.positive_fraction:.3f}",
            "-" if r.projected_pair_share is None else f"{r.projected_pair_share:.2%}",
            "-" if r.pair_share_amplification is None else f"{r.pair_share_amplification:.1f}x",
        )
        for r in reports
    ]
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(HEADERS)
    ]
    line = "  ".join(h.ljust(w) for h, w in zip(HEADERS, widths, strict=True))
    out = [line, "  ".join("-" * w for w in widths)]
    out += ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in rows]
    return "\n".join(out)

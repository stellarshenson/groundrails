"""The evidence-shape gate: three verdicts, and the window arithmetic under them."""

import polars as pl
import pytest

from groundrails.dataset import (
    BLOCK,
    PASS,
    PASS_WITH_DROP,
    MixProjection,
    Presentation,
    shape,
    shape_lane,
    shape_table,
    window_count,
    windows,
)

PRES = Presentation()  # 1500/750/96, bars 20 rows-per-doc / 0.15 share / 0.02 over-cap


def lane(rows):
    """rows: (doc_id, chunk_chars, label) -> a minimal lane frame."""
    return pl.DataFrame(
        [
            {
                "claim": f"claim {i}",
                "chunk": "x" * n,
                "label": y,
                "doc_id": doc,
                "source": "test",
            }
            for i, (doc, n, y) in enumerate(rows)
        ]
    )


# --- window arithmetic ------------------------------------------------------ #
@pytest.mark.parametrize("n", [0, 1, 100, 1499, 1500, 1501, 2250, 2251, 30_000, 72_751])
def test_window_count_matches_the_splitter(n):
    assert window_count(n, PRES) == len(windows("x" * n, PRES))


def test_windows_flush_to_the_end():
    text = "x" * 3000
    parts = windows(text, PRES)
    assert all(len(p) == PRES.window_chars for p in parts)
    assert parts[-1] == text[-PRES.window_chars :]


# --- the three verdicts ----------------------------------------------------- #
def test_pass_when_every_row_fits():
    df = lane([(f"doc{i}", 800, i % 2) for i in range(20)])
    r = shape(df, PRES, name="clean")
    assert r.verdict == PASS
    assert (r.rows, r.pairs, r.documents) == (20, 20, 20)
    assert r.over_cap_rows == 0
    assert r.reasons == []


def test_pass_with_drop_reports_the_over_cap_rows():
    # 100 short rows plus one 200-window row: droppable at 0.99% of rows
    rows = [(f"doc{i}", 800, 1) for i in range(100)] + [("big", 1500 + 200 * 750, 0)]
    r = shape(lane(rows), PRES, name="tail")
    assert r.verdict == PASS_WITH_DROP
    assert r.over_cap_rows == 1
    assert r.over_cap_window_counts == [201]
    assert r.rows_after_drop == 100
    assert r.pairs_after_drop == r.pairs - 201
    assert "drop 1 rows over the 96-pair batch cap" in r.reasons[0]


def test_block_on_document_reuse():
    df = lane([("shared", 800, i % 2) for i in range(100)])
    r = shape(df, PRES, name="reuse")
    assert r.verdict == BLOCK
    assert r.rows_per_document == 100.0
    assert any("document reuse" in x for x in r.reasons)


def test_block_on_pair_share_capture():
    df = lane([(f"doc{i}", 30_000, 1) for i in range(20)])
    r = shape(df, PRES, name="greedy", projection=MixProjection("mix", other_pairs=1000))
    assert r.verdict == BLOCK
    assert r.projected_pair_share > PRES.max_pair_share
    assert any("pair-share capture" in x for x in r.reasons)


def test_block_on_an_over_cap_tail_too_large_to_drop():
    rows = [(f"doc{i}", 800, 1) for i in range(9)] + [("huge", 1500 + 200 * 750, 0)]
    r = shape(lane(rows), PRES, name="fat-tail")
    assert r.verdict == BLOCK
    assert r.over_cap_fraction == pytest.approx(0.1)
    assert any("over-cap tail" in x for x in r.reasons)


# --- projection and reporting ----------------------------------------------- #
def test_projection_shares_are_against_the_rest_of_the_mix():
    df = lane([(f"doc{i}", 800, 1) for i in range(10)])  # 10 rows, 10 pairs
    r = shape(df, PRES, name="small", projection=MixProjection("mix", 90, other_rows=90))
    assert r.projected_pair_share == pytest.approx(0.1)
    assert r.projected_row_share == pytest.approx(0.1)
    assert r.mix == "mix"


def test_no_projection_leaves_the_share_unset():
    r = shape(lane([("d", 500, 1)]), PRES, name="x")
    assert r.projected_pair_share is None
    assert r.verdict == PASS


def test_distributions_and_positive_fraction():
    df = lane([("a", 500, 1), ("b", 5000, 0), ("c", 500, 1)])
    r = shape(df, PRES, name="dist")
    assert r.positive_fraction == pytest.approx(2 / 3, abs=1e-4)
    assert r.chunk_chars["max"] == 5000
    assert r.claim_chars["max"] >= 7
    assert r.windows["max"] == window_count(5000, PRES)


def test_shape_lane_reads_a_parquet(tmp_path):
    p = tmp_path / "demo_lane.parquet"
    lane([(f"doc{i}", 800, 1) for i in range(5)]).write_parquet(p)
    r = shape_lane(p)
    assert r.name == "demo_lane"
    assert r.rows == 5


def test_table_renders_every_lane():
    reports = [
        shape(lane([("a", 500, 1)]), PRES, name="one"),
        shape(lane([("b", 500, 0)]), PRES, name="two"),
    ]
    out = shape_table(reports)
    assert "verdict" in out and "one" in out and "two" in out

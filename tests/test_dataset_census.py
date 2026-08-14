"""Census arithmetic: rows, pairs and share per group, and the go/no-go."""

import polars as pl
import pytest

from groundrails.dataset import LaneSpec, MixSpec, Presentation, assemble, census
from groundrails.dataset.census import census_lane_dir
from groundrails.dataset.shape import window_count

PRES = Presentation()


def frame(rows):
    """rows: (group, doc_id, chunk_chars, label)."""
    return pl.DataFrame(
        [
            {
                "claim": f"claim {i}",
                "chunk": "x" * n,
                "label": y,
                "doc_id": doc,
                "group": g,
            }
            for i, (g, doc, n, y) in enumerate(rows)
        ]
    )


def test_rows_and_pairs_add_up_per_group():
    df = frame(
        [("a", f"d{i}", 500, 1) for i in range(10)]
        + [("b", f"e{i}", 5000, 0) for i in range(4)]
    )
    rep = census(df, PRES, name="mix")
    assert rep.rows == 14
    assert rep.pairs == 10 * 1 + 4 * window_count(5000, PRES)
    by_group = {g.name: g for g in rep.groups}
    assert by_group["a"].rows == 10
    assert by_group["b"].pairs == 4 * window_count(5000, PRES)
    assert sum(g.rows for g in rep.groups) == rep.rows
    assert sum(g.pairs for g in rep.groups) == rep.pairs


def test_pair_shares_sum_to_one():
    df = frame(
        [("a", f"d{i}", 500, 1) for i in range(10)]
        + [("b", f"e{i}", 5000, 0) for i in range(4)]
    )
    rep = census(df, PRES, name="mix")
    assert sum(g.projected_pair_share for g in rep.groups) == pytest.approx(1.0, abs=1e-4)
    assert sum(g.projected_row_share for g in rep.groups) == pytest.approx(1.0, abs=1e-4)


def test_pair_share_diverges_from_row_share_when_evidence_is_long():
    """The failure this stage exists to make visible: few rows, most of the pairs."""
    df = frame(
        [("bulk", f"d{i}", 500, 1) for i in range(100)]
        + [("long", f"e{i}", 40_000, 1) for i in range(5)]
    )
    by_group = {g.name: g for g in census(df, PRES, name="mix").groups}
    assert by_group["long"].projected_row_share < 0.05
    assert by_group["long"].projected_pair_share > 0.6


def test_mean_target_is_the_positive_rate():
    df = frame([("a", "d", 500, 1), ("a", "d", 500, 0), ("b", "e", 500, 1)])
    rep = census(df, PRES, name="mix")
    assert rep.mean_target == pytest.approx(2 / 3, abs=1e-4)
    assert {g.name: g.positive_fraction for g in rep.groups}["a"] == pytest.approx(0.5)


def test_window_census_matches_the_frame():
    df = frame([("a", f"d{i}", 5000, 1) for i in range(3)] + [("b", "e", 500, 0)])
    rep = census(df, PRES, name="mix")
    assert rep.max_windows == window_count(5000, PRES)
    assert rep.mean_windows == pytest.approx((3 * window_count(5000, PRES) + 1) / 4, abs=1e-4)
    assert rep.multi_window_share == pytest.approx(0.75)


def test_go_when_nothing_blocks():
    rep = census(frame([("a", f"d{i}", 500, i % 2) for i in range(30)]), PRES, name="mix")
    assert rep.go is True
    assert rep.blocking == []


def test_no_go_names_the_blocking_group():
    df = frame(
        [("clean", f"d{i}", 500, 1) for i in range(50)]
        + [("reused", "one-doc", 500, 0) for _ in range(50)]
    )
    rep = census(df, PRES, name="mix")
    assert rep.go is False
    assert "reused" in rep.blocking


def test_no_go_when_over_cap_rows_survive():
    df = frame([("a", f"d{i}", 500, 1) for i in range(50)] + [("a", "big", 1500 + 200 * 750, 0)])
    rep = census(df, PRES, name="mix")
    assert rep.go is False
    assert rep.over_cap_rows == 1


def test_census_of_an_assembled_mix_uses_its_presentation(tmp_path):
    path = tmp_path / "a_lane.parquet"
    pl.DataFrame(
        {
            "claim": ["c1", "c2"],
            "chunk": ["x" * 500, "y" * 5000],
            "label": [1, 0],
            "doc_id": ["d1", "d2"],
            "source": ["s", "s"],
            "tag": ["a", "a"],
        }
    ).write_parquet(path)
    mix = assemble(MixSpec(name="demo", lanes=(LaneSpec(path, "a"),), presentation=PRES))
    rep = census(mix)
    assert rep.mix == "demo"
    assert rep.rows == 2
    assert rep.presentation["window_chars"] == PRES.window_chars


def test_census_lane_dir_reads_lanes_that_are_not_assembled(tmp_path):
    for name in ("a", "b"):
        pl.DataFrame(
            {
                "claim": ["c"],
                "chunk": ["x" * 500],
                "label": [1],
                "doc_id": [name],
            }
        ).write_parquet(tmp_path / f"{name}.parquet")
    rep = census_lane_dir({"a": tmp_path / "a.parquet", "b": tmp_path / "b.parquet"})
    assert rep.rows == 2
    assert [g.name for g in rep.groups] == ["a", "b"]


def test_report_serialises():
    rep = census(frame([("a", "d", 500, 1)]), PRES, name="mix")
    d = rep.to_dict()
    assert d["mix"] == "mix"
    assert d["groups"][0]["verdict"]

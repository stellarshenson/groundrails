"""Mix assembly: the declared assertions, the group map, and the documented drop."""

import polars as pl
import pytest

from groundrails.dataset import LaneSpec, MixAssertionError, MixSpec, Presentation, assemble

PRES = Presentation()


def write_lane(tmp_path, name, rows):
    """rows: (doc_id, chunk_chars, label)."""
    df = pl.DataFrame(
        [
            {
                "pair_id": i,
                "claim": f"{name} claim {i}",
                "chunk": "x" * n,
                "label": y,
                "doc_id": doc,
                "source": "test",
                "tag": name,
                "extra": "provenance",
            }
            for i, (doc, n, y) in enumerate(rows)
        ]
    )
    path = tmp_path / f"{name}_lane.parquet"
    df.write_parquet(path)
    return path


@pytest.fixture
def lanes(tmp_path):
    return {
        "a": write_lane(tmp_path, "a", [(f"d{i}", 500, i % 2) for i in range(10)]),
        "b": write_lane(tmp_path, "b", [(f"e{i}", 800, 1) for i in range(6)]),
    }


def spec(lanes, **kw):
    return MixSpec(
        name="mix",
        lanes=(LaneSpec(lanes["a"], "a"), LaneSpec(lanes["b"], "b")),
        presentation=PRES,
        **kw,
    )


def test_assembles_the_group_map_and_pair_index(lanes):
    mix = assemble(spec(lanes))
    assert mix.rows == 16
    assert mix.groups == ("a", "b")
    assert mix.frame["pair_id"].to_list() == list(range(16))
    assert mix.frame.columns == list(
        ("pair_id", "claim", "chunk", "label", "doc_id", "source", "tag", "group")
    )
    assert mix.to_dict()["group_rows"] == {"a": 10, "b": 6}


def test_provenance_columns_stay_in_the_lane(lanes):
    assert "extra" not in assemble(spec(lanes)).frame.columns


def test_base_mix_is_its_own_group(lanes, tmp_path):
    base = write_lane(tmp_path, "base", [(f"b{i}", 400, 1) for i in range(4)])
    mix = assemble(spec(lanes, base=base, base_group="clean"))
    assert mix.groups == ("a", "b", "clean")
    assert mix.rows == 20


def test_lane_row_assertion_aborts(lanes):
    bad = MixSpec(name="mix", lanes=(LaneSpec(lanes["a"], "a", expected_rows=99),))
    with pytest.raises(MixAssertionError, match="10 rows, spec says 99"):
        assemble(bad)


def test_lane_positive_assertion_aborts(lanes):
    bad = MixSpec(name="mix", lanes=(LaneSpec(lanes["b"], "b", expected_positives=0),))
    with pytest.raises(MixAssertionError, match="positives"):
        assemble(bad)


def test_mix_row_assertion_aborts(lanes):
    with pytest.raises(MixAssertionError, match="16 rows, spec says 15"):
        assemble(spec(lanes, expected_rows=15))


def test_group_map_assertion_aborts(lanes):
    with pytest.raises(MixAssertionError, match="group map"):
        assemble(spec(lanes, expected_groups=("a", "b", "c")))


def test_expected_counts_that_match_pass(lanes):
    mix = assemble(
        MixSpec(
            name="mix",
            lanes=(
                LaneSpec(lanes["a"], "a", expected_rows=10, expected_positives=5),
                LaneSpec(lanes["b"], "b", expected_rows=6, expected_positives=6),
            ),
            expected_rows=16,
            expected_groups=("a", "b"),
        )
    )
    assert mix.rows == 16


def test_over_cap_rows_are_dropped_and_recorded(tmp_path, lanes):
    big = write_lane(
        tmp_path, "big", [("d0", 500, 1), ("d1", 1500 + 200 * 750, 0), ("d2", 500, 1)]
    )
    mix = assemble(
        MixSpec(name="mix", lanes=(LaneSpec(big, "big"),), presentation=PRES)
    )
    assert mix.rows == 2
    drop = mix.drops["big"]
    assert (drop.rows_in, drop.rows_dropped, drop.rows_kept) == (3, 1, 2)
    assert drop.dropped_window_counts == [201]
    assert drop.dropped_pairs == 201
    assert drop.cap == PRES.pairs_per_batch
    assert "batch-cap" in drop.rule


def test_over_cap_drop_can_be_turned_off(tmp_path):
    big = write_lane(tmp_path, "big", [("d0", 500, 1), ("d1", 1500 + 200 * 750, 0)])
    mix = assemble(
        MixSpec(name="mix", lanes=(LaneSpec(big, "big"),), drop_over_cap=False)
    )
    assert mix.rows == 2
    assert mix.drops == {}


def test_a_lane_outside_the_pair_schema_is_refused(tmp_path):
    path = tmp_path / "bad.parquet"
    pl.DataFrame({"text": ["a"], "y": [1]}).write_parquet(path)
    with pytest.raises(MixAssertionError, match="pair schema"):
        assemble(MixSpec(name="mix", lanes=(LaneSpec(path, "bad"),)))

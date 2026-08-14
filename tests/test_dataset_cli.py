"""`groundrails dataset` - the stage subcommands render, gate and exit correctly."""

import json

import polars as pl
import pytest

from groundrails.cli import main


def lane(path, rows):
    """rows: (doc_id, chunk_chars, label)."""
    pl.DataFrame(
        [
            {
                "pair_id": i,
                "claim": f"claim {i}",
                "chunk": "x" * n,
                "label": y,
                "doc_id": doc,
                "source": "test",
                "tag": path.stem,
            }
            for i, (doc, n, y) in enumerate(rows)
        ]
    ).write_parquet(path)
    return path


@pytest.mark.parametrize(
    "argv",
    [
        ["dataset", "--help"],
        ["dataset", "fetch", "--help"],
        ["dataset", "contaminate", "--help"],
        ["dataset", "format", "--help"],
        ["dataset", "shape", "--help"],
        ["dataset", "assemble", "--help"],
        ["dataset", "census", "--help"],
        ["dataset", "run", "--help"],
    ],
)
def test_help_renders(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 0
    assert capsys.readouterr().out


def test_shape_passes_a_clean_lane(tmp_path, capsys):
    p = lane(tmp_path / "clean_lane.parquet", [(f"d{i}", 500, i % 2) for i in range(10)])
    rc = main(["dataset", "shape", str(p)])
    assert rc == 0
    assert "PASS" in capsys.readouterr().out


def test_shape_exits_1_on_a_blocked_lane(tmp_path, capsys):
    p = lane(tmp_path / "reuse_lane.parquet", [("one", 500, i % 2) for i in range(100)])
    rc = main(["dataset", "shape", str(p)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "BLOCK" in out and "document reuse" in out


def test_shape_json_carries_the_full_report(tmp_path, capsys):
    p = lane(tmp_path / "j_lane.parquet", [(f"d{i}", 500, 1) for i in range(5)])
    rc = main(["dataset", "shape", str(p), "--json", "--mix-other-pairs", "95"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["rows"] == 5
    assert payload[0]["projected_pair_share"] == pytest.approx(0.05)


def test_shape_flags_override_the_presentation(tmp_path, capsys):
    rows = [(f"d{i}", 500, 1) for i in range(10)] + [("long", 5000, 0)]
    p = lane(tmp_path / "cap_lane.parquet", rows)
    assert main(["dataset", "shape", str(p)]) == 0  # 6 windows is under the default cap
    capsys.readouterr()
    rc = main(["dataset", "shape", str(p), "--cap", "2", "--max-over-cap-fraction", "0.5"])
    assert rc == 0
    assert "PASS-WITH-DROP" in capsys.readouterr().out


def test_shape_takes_bars_from_a_manifest_corpus(tmp_path, capsys):
    p = lane(tmp_path / "m_lane.parquet", [(f"d{i}", 500, 1) for i in range(5)])
    rc = main(["dataset", "shape", str(p), "--corpus", "minicheck"])
    assert rc == 0
    assert "PASS" in capsys.readouterr().out


def test_assemble_then_census(tmp_path, capsys):
    a = lane(tmp_path / "a_lane.parquet", [(f"d{i}", 500, i % 2) for i in range(10)])
    b = lane(tmp_path / "b_lane.parquet", [(f"e{i}", 800, 1) for i in range(6)])
    spec = tmp_path / "mix.json"
    spec.write_text(
        json.dumps(
            {
                "name": "demo",
                "expected_rows": 16,
                "expected_groups": ["a", "b"],
                "lanes": [
                    {"path": str(a), "group": "a", "expected_rows": 10},
                    {"path": str(b), "group": "b", "expected_positives": 6},
                ],
            }
        )
    )
    out = tmp_path / "mix.parquet"
    assert main(["dataset", "assemble", "--spec", str(spec), "--out", str(out)]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["rows"] == 16
    assert record["group_rows"] == {"a": 10, "b": 6}

    assert main(["dataset", "census", str(out), "--name", "demo"]) == 0
    text = capsys.readouterr().out
    assert "GO: True" in text and "demo" in text


def test_assemble_aborts_on_a_failed_assertion(tmp_path, capsys):
    a = lane(tmp_path / "a_lane.parquet", [(f"d{i}", 500, 1) for i in range(4)])
    spec = tmp_path / "mix.json"
    spec.write_text(json.dumps({"lanes": [{"path": str(a), "group": "a"}], "expected_rows": 99}))
    assert main(["dataset", "assemble", "--spec", str(spec)]) == 1
    assert "MIX ABORT" in capsys.readouterr().err


def test_census_exits_1_on_no_go(tmp_path, capsys):
    a = lane(tmp_path / "a_lane.parquet", [("one", 500, 1) for _ in range(60)])
    b = lane(tmp_path / "b_lane.parquet", [(f"e{i}", 500, 0) for i in range(60)])
    spec = tmp_path / "mix.json"
    spec.write_text(
        json.dumps(
            {"lanes": [{"path": str(a), "group": "a"}, {"path": str(b), "group": "b"}]}
        )
    )
    out = tmp_path / "mix.parquet"
    main(["dataset", "assemble", "--spec", str(spec), "--out", str(out)])
    capsys.readouterr()
    assert main(["dataset", "census", str(out)]) == 1
    assert "GO: False" in capsys.readouterr().out


def test_fetch_dry_run_touches_nothing(tmp_path, capsys):
    rc = main(["dataset", "fetch", "minicheck", "--data-dir", str(tmp_path), "--dry-run"])
    assert rc == 0
    assert "dry-run" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())

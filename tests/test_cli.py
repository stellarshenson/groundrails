"""Smoke tests for the simplified `groundrails` CLI."""

import json

import pytest

from groundrails.cli import main
from groundrails.lexical_mt import has_model


def test_ground_single_grounded(tmp_path, capsys):
    src = tmp_path / "s.txt"
    src.write_text("The Eiffel Tower is located in Paris, France.", encoding="utf-8")
    rc = main(["ground", "--claim", "The Eiffel Tower is in Paris.", "--source", str(src)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "FUZZY" in out or "EXACT" in out


def test_ground_json_contradiction(tmp_path, capsys):
    src = tmp_path / "s.txt"
    src.write_text("The model is built from 1000 transformer layers in total.", encoding="utf-8")
    rc = main(
        [
            "ground",
            "--claim",
            "The model has 512 transformer layers.",
            "--source",
            str(src),
            "--json",
        ]
    )
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["match_type"] == "contradicted"
    assert ["512", "1000"] in [list(x) for x in data["numeric_mismatches"]]


def test_ground_no_match_exits_1(tmp_path, capsys):
    src = tmp_path / "s.txt"
    src.write_text("This document is about office furniture procurement.", encoding="utf-8")
    rc = main(["ground", "--claim", "The rocket reached escape velocity.", "--source", str(src)])
    assert rc == 1


def test_config_runs(capsys):
    rc = main(["config"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "lexical_effort" in out


def test_extract_claims(tmp_path, capsys):
    doc = tmp_path / "d.md"
    doc.write_text(
        "The system processes 1000 records per second.\nIt supports five languages.\n",
        encoding="utf-8",
    )
    rc = main(["extract-claims", "--document", str(doc)])
    out = capsys.readouterr().out
    assert rc == 0
    claims = json.loads(out)
    assert isinstance(claims, list) and len(claims) >= 1


def test_ground_unsupported_language_blocked(tmp_path, capsys, monkeypatch):
    from groundrails import lexical as lx
    from groundrails import lexical_mt as mt

    monkeypatch.setattr(lx, "detect_lang_confident", lambda *a, **k: "la")
    monkeypatch.setattr(mt, "has_model", lambda iso: False)
    src = tmp_path / "s.txt"
    src.write_text("some english source text about geography", encoding="utf-8")
    rc = main(
        ["ground", "--claim", "Lorem ipsum dolor sit amet consectetur.", "--source", str(src)]
    )
    err = capsys.readouterr().err
    assert rc == 3
    assert "argos" in err.lower() or "blocked" in err.lower()


@pytest.mark.skipif(not has_model("de"), reason="argos de->en model not installed")
def test_ground_cross_lingual_supported(tmp_path, capsys):
    src = tmp_path / "s.txt"
    src.write_text("The Eiffel Tower is located in Paris, France.", encoding="utf-8")
    rc = main(["ground", "--claim", "Der Eiffelturm steht in Paris.", "--source", str(src)])
    out = capsys.readouterr().out
    assert rc == 0  # MT bridge grounds the German claim against the English source
    assert any(tag in out for tag in ("EXACT", "FUZZY", "BM25"))

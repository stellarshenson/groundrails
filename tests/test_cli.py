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


def test_ground_positional_claims_source(tmp_path, capsys):
    """Simple form: `ground CLAIMS SOURCE` (claims file first, source second)."""
    src = tmp_path / "s.txt"
    src.write_text("The Eiffel Tower is located in Paris, France.", encoding="utf-8")
    claims = tmp_path / "claims.json"
    claims.write_text(json.dumps(["The Eiffel Tower is in Paris."]), encoding="utf-8")
    rc = main(["ground", str(claims), str(src)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("1. ")  # one numbered report line per claim


def test_ground_claims_flag_objects(tmp_path):
    """`--claims` accepts the {id, claim, ...} objects extract-claims writes."""
    src = tmp_path / "s.txt"
    src.write_text("The Eiffel Tower is located in Paris, France.", encoding="utf-8")
    claims = tmp_path / "claims.json"
    claims.write_text(json.dumps([{"id": "c01", "claim": "The Eiffel Tower is in Paris."}]), "utf-8")
    rc = main(["ground", "--claims", str(claims), "--source", str(src)])
    assert rc == 0


def test_ground_text_claims_one_per_line(tmp_path):
    """A plain-text claims file (one claim per non-empty line) is accepted."""
    src = tmp_path / "s.txt"
    src.write_text("The Eiffel Tower is located in Paris, France.", encoding="utf-8")
    claims = tmp_path / "claims.txt"
    claims.write_text("The Eiffel Tower is in Paris.\n\n", encoding="utf-8")
    rc = main(["ground", str(claims), str(src)])
    assert rc == 0


def test_ground_batch_gate_exits_1_on_ungrounded(tmp_path):
    """`ground` is a gate: a batch with an ungrounded claim exits 1."""
    src = tmp_path / "s.txt"
    src.write_text("This document is about office furniture procurement.", encoding="utf-8")
    claims = tmp_path / "claims.json"
    claims.write_text(json.dumps(["The rocket reached escape velocity."]), encoding="utf-8")
    rc = main(["ground", str(claims), str(src)])
    assert rc == 1


def test_ground_claims_schema_violation(tmp_path, capsys):
    """A claims file that does not conform to the Claim schema fails clearly."""
    src = tmp_path / "s.txt"
    src.write_text("irrelevant", encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"note": "no claim key"}]), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["ground", str(bad), str(src)])
    assert exc.value.code == 1
    assert "schema" in capsys.readouterr().err.lower()


def test_ground_no_input_exits_2(capsys):
    """No claims and no source -> usage error (exit 2)."""
    rc = main(["ground"])
    assert rc == 2
    assert "CLAIMS SOURCE" in capsys.readouterr().err

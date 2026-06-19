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
    entry = data["claims"][0]
    assert entry["match_type"] == "contradicted"
    assert entry["grounded"] is False
    assert ["512", "1000"] in entry["contradiction"]["numeric"]


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


def test_extract_claims_char_span(tmp_path, capsys):
    """extract-claims records each claim's char span; the span slices back to the claim."""
    doc = tmp_path / "d.md"
    text = "The system processes 1000 records per second. It supports five languages."
    doc.write_text(text, encoding="utf-8")
    rc = main(["extract-claims", "--document", str(doc)])
    assert rc == 0
    claims = json.loads(capsys.readouterr().out)
    c0 = claims[0]
    assert c0["char_start"] is not None and c0["char_end"] is not None
    assert text[c0["char_start"] : c0["char_end"]] == c0["claim"]


def test_extract_claims_relocates_across_line_wrap(tmp_path, capsys):
    """_relocate is whitespace-flexible: a sentence wrapped across a newline (joined with a
    single space during extraction) still locates back to the original span, newline and all."""
    doc = tmp_path / "d.md"
    doc.write_text("The system processes\n1000 records per second.", encoding="utf-8")
    rc = main(["extract-claims", "--document", str(doc)])
    assert rc == 0
    c0 = json.loads(capsys.readouterr().out)[0]
    text = doc.read_text(encoding="utf-8")
    located = text[c0["char_start"] : c0["char_end"]]
    assert "\n" in located  # the span bridges the wrapped line, not just one line
    assert " ".join(located.split()) == c0["claim"]  # whitespace-normalises back to the claim


def test_ground_json_grounding_document(tmp_path, capsys):
    """--json emits the business-end document: verdict + final score + support location, no per-scorer internals."""
    src = tmp_path / "evidence.txt"
    src.write_text("The Eiffel Tower is located in Paris, France.", encoding="utf-8")
    rc = main(
        ["ground", "--claim", "The Eiffel Tower is in Paris.", "--source", str(src), "--json"]
    )
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["summary"]["total"] == 1 and doc["summary"]["grounded"] == 1
    entry = doc["claims"][0]
    assert entry["grounded"] is True and entry["score"] > 0
    sup = entry["support"]
    assert sup is not None and sup["source_index"] == 0
    assert sup["char_end"] > sup["char_start"] >= 0 and sup["matched_text"]
    assert "exact_score" not in entry  # business end hides per-scorer detail


def test_ground_full_output_raw(tmp_path, capsys):
    """--full-output keeps the full per-scorer GroundingMatch dump."""
    src = tmp_path / "s.txt"
    src.write_text("The Eiffel Tower is located in Paris, France.", encoding="utf-8")
    rc = main(
        [
            "ground",
            "--claim",
            "The Eiffel Tower is in Paris.",
            "--source",
            str(src),
            "--full-output",
        ]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)[0]  # --full-output is a list of per-claim matches
    assert "exact_score" in data and "fuzzy_score" in data and "bm25_score" in data


def test_ground_batch_json_carries_claim_location(tmp_path, capsys):
    """Batch --json carries each claim's answer-doc location from the claims file into the document."""
    claims = tmp_path / "claims.json"
    claims.write_text(
        json.dumps(
            [
                {
                    "id": "c01",
                    "claim": "The Eiffel Tower is in Paris.",
                    "line_number": 1,
                    "char_start": 0,
                    "char_end": 29,
                }
            ]
        ),
        encoding="utf-8",
    )
    src = tmp_path / "evidence.txt"
    src.write_text("The Eiffel Tower is located in Paris, France.", encoding="utf-8")
    rc = main(["ground", "--claims", str(claims), "--source", str(src), "--json"])
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    entry = doc["claims"][0]
    assert entry["id"] == "c01"
    assert entry["claim_location"] == {"line": 1, "char_start": 0, "char_end": 29}
    assert doc["sources"] == [str(src)]


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


def test_ground_positional_document(tmp_path, capsys):
    """Default form: `ground DOCUMENT EVIDENCE` extracts claims from the document and grounds them."""
    src = tmp_path / "evidence.txt"
    src.write_text(
        "The Eiffel Tower is located in Paris, France. It was completed in 1889.", encoding="utf-8"
    )
    doc = tmp_path / "answer.md"
    doc.write_text("The Eiffel Tower is in Paris. It was completed in 1889.", encoding="utf-8")
    rc = main(["ground", str(doc), str(src)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("1. ")  # two claims extracted -> numbered report lines


def test_ground_multiple_inline_claims(tmp_path, capsys):
    """--claim is repeatable; the positionals are evidence."""
    src = tmp_path / "evidence.txt"
    src.write_text(
        "The Eiffel Tower is located in Paris, France. It was completed in 1889.", encoding="utf-8"
    )
    rc = main(
        [
            "ground",
            "--claim",
            "The Eiffel Tower is in Paris.",
            "--claim",
            "It was completed in 1889.",
            str(src),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("1. ")  # two inline claims -> numbered report lines


def test_ground_document_json_has_claim_location(tmp_path, capsys):
    """The positional-document form populates each claim's answer-doc location in --json output."""
    src = tmp_path / "evidence.txt"
    src.write_text("The Eiffel Tower is located in Paris, France.", encoding="utf-8")
    doc = tmp_path / "answer.md"
    doc.write_text("The Eiffel Tower is in Paris.", encoding="utf-8")
    rc = main(["ground", str(doc), str(src), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    entry = out["claims"][0]
    assert entry["id"] is not None  # extract-claims assigns ids
    assert entry["claim_location"]["char_start"] == 0


def test_ground_claims_flag_objects(tmp_path):
    """`--claims` accepts the {id, claim, ...} objects extract-claims writes."""
    src = tmp_path / "s.txt"
    src.write_text("The Eiffel Tower is located in Paris, France.", encoding="utf-8")
    claims = tmp_path / "claims.json"
    claims.write_text(
        json.dumps([{"id": "c01", "claim": "The Eiffel Tower is in Paris."}]), "utf-8"
    )
    rc = main(["ground", "--claims", str(claims), "--source", str(src)])
    assert rc == 0


def test_ground_text_claims_one_per_line(tmp_path):
    """A plain-text claims file (one claim per non-empty line) via --claims."""
    src = tmp_path / "s.txt"
    src.write_text("The Eiffel Tower is located in Paris, France.", encoding="utf-8")
    claims = tmp_path / "claims.txt"
    claims.write_text("The Eiffel Tower is in Paris.\n\n", encoding="utf-8")
    rc = main(["ground", "--claims", str(claims), "--source", str(src)])
    assert rc == 0


def test_ground_batch_gate_exits_1_on_ungrounded(tmp_path):
    """`ground` is a gate: a batch with an ungrounded claim exits 1."""
    src = tmp_path / "s.txt"
    src.write_text("This document is about office furniture procurement.", encoding="utf-8")
    claims = tmp_path / "claims.json"
    claims.write_text(json.dumps(["The rocket reached escape velocity."]), encoding="utf-8")
    rc = main(["ground", "--claims", str(claims), "--source", str(src)])
    assert rc == 1


def test_ground_claims_schema_violation(tmp_path, capsys):
    """A claims file that does not conform to the Claim schema fails clearly."""
    src = tmp_path / "s.txt"
    src.write_text("irrelevant", encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"note": "no claim key"}]), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["ground", "--claims", str(bad), "--source", str(src)])
    assert exc.value.code == 1
    assert "schema" in capsys.readouterr().err.lower()


def test_ground_no_input_exits_2(capsys):
    """No document, no claims, no evidence -> usage error (exit 2)."""
    rc = main(["ground"])
    assert rc == 2
    assert "DOCUMENT EVIDENCE" in capsys.readouterr().err

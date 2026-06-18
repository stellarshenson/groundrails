"""Schema tests for the Claim pydantic model and the claims-file loader."""

import json

from pydantic import ValidationError
import pytest

from groundrails.claims import Claim, load_claims, parse_claims


def test_parse_claims_accepts_strings_and_objects():
    out = parse_claims(["a claim", {"id": "c01", "claim": "another", "line_number": 3}])
    assert [c.claim for c in out] == ["a claim", "another"]
    assert out[1].id == "c01" and out[1].line_number == 3


def test_claim_requires_non_empty_text():
    with pytest.raises(ValidationError):
        Claim(claim="")
    with pytest.raises(ValidationError):
        parse_claims([{"note": "no claim key"}])


def test_parse_claims_rejects_non_list():
    with pytest.raises(ValueError):
        parse_claims({"claim": "x"})


def test_load_claims_json_mixed(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(["one", {"claim": "two"}]), encoding="utf-8")
    assert [c.claim for c in load_claims(p)] == ["one", "two"]


def test_load_claims_text_one_per_line(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("one\n\n two \n", encoding="utf-8")
    assert [c.claim for c in load_claims(p)] == ["one", "two"]


def test_load_claims_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_claims(tmp_path / "nope.json")


def test_extract_output_conforms_to_schema():
    """The dicts extract-claims writes re-validate against the Claim schema."""
    from groundrails.extract import extract_claims

    extracted = extract_claims("The system processes 1000 records per second.")
    dumps = [Claim(id=c.id, claim=c.claim, line_number=c.line_number).model_dump() for c in extracted]
    assert len(parse_claims(dumps)) == len(dumps)  # no ValidationError

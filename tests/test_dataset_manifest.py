"""Manifest validation - a bad corpus entry must fail at load, not mid-fetch."""

from pathlib import Path

from pydantic import ValidationError
import pytest
import yaml

from groundrails.dataset import load_manifest
from groundrails.dataset import manifest as M

FIXTURE = Path(__file__).parent / "fixtures" / "dataset_corpora.yaml"


def test_packaged_manifest_loads_and_is_consistent():
    m = M.packaged()
    assert m.version == 1
    assert len(m.names()) == len(set(m.names()))
    for entry in m.corpora:
        assert entry.licence.tag and entry.licence.verified
        assert entry.task_shape
        if entry.source.kind == "adapter":
            assert entry.source.fetcher


def test_packaged_manifest_carries_the_r19_lanes():
    m = M.packaged()
    for name in ("fava", "pubhealth", "minicheck", "factscore", "findver", "attributionbench"):
        entry = m.get(name)
        assert entry.format is not None
        assert entry.expected.rows and entry.expected.documents


def test_fixture_manifest_parses_every_mapping_feature():
    m = load_manifest(FIXTURE)
    tiny = m.get("tiny")
    assert tiny.format.chunk.join == "\n\n"
    assert tiny.format.label.map == {"supported": 1, "refuted": 0}
    assert tiny.format.doc_id.hash == "chunk"
    assert tiny.format.retain_as == {"raw_verdict": "verdict"}
    assert m.get("tiny-adapter").presentation.pairs_per_batch == 4


def test_unknown_corpus_names_the_known_ones():
    with pytest.raises(KeyError, match="fava"):
        M.packaged().get("does-not-exist")


def _write(tmp_path, entry):
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump({"version": 1, "corpora": [entry]}), encoding="utf-8")
    return p


BASE = {
    "name": "x",
    "title": "X",
    "licence": {"tag": "MIT", "commercial_use": True, "verified": "test"},
    "source": {"kind": "hf_dataset", "repos": ["a/b"]},
    "task_shape": "claim -> claim",
}


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ValidationError):
        load_manifest(_write(tmp_path, {**BASE, "licence_tag": "MIT"}))


def test_source_kind_requires_its_own_fields(tmp_path):
    with pytest.raises(ValidationError, match="repos"):
        load_manifest(_write(tmp_path, {**BASE, "source": {"kind": "hf_dataset"}}))
    with pytest.raises(ValidationError, match="fetcher"):
        load_manifest(_write(tmp_path, {**BASE, "source": {"kind": "adapter"}}))


def test_incomplete_format_block_is_rejected(tmp_path):
    entry = {**BASE, "format": {"claim": "c"}}
    with pytest.raises(ValidationError, match="chunk"):
        load_manifest(_write(tmp_path, entry))


def test_doc_id_needs_exactly_one_source(tmp_path):
    fmt = {
        "claim": "c",
        "chunk": {"column": "e"},
        "label": {"column": "y"},
        "doc_id": {"column": "d", "hash": "chunk"},
    }
    with pytest.raises(ValidationError, match="doc_id"):
        load_manifest(_write(tmp_path, {**BASE, "format": fmt}))


def test_label_map_values_must_be_binary(tmp_path):
    fmt = {
        "claim": "c",
        "chunk": {"column": "e"},
        "label": {"column": "y", "map": {"yes": 2}},
        "doc_id": {"column": "d"},
    }
    with pytest.raises(ValidationError, match="0 or 1"):
        load_manifest(_write(tmp_path, {**BASE, "format": fmt}))


def test_duplicate_names_are_rejected(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump({"version": 1, "corpora": [BASE, BASE]}), encoding="utf-8")
    with pytest.raises(ValidationError, match="duplicate"):
        load_manifest(p)

"""The format stage: manifest-declared column mapping, adapters, and the counts it holds."""

import json
from pathlib import Path
import zipfile

import polars as pl
import pytest

from groundrails.dataset import (
    FormatError,
    evidence_texts,
    format_corpus,
    load_manifest,
    register_adapter,
    write_lane,
)
from groundrails.dataset.manifest import PAIR_COLUMNS

FIXTURE = Path(__file__).parent / "fixtures" / "dataset_corpora.yaml"

SOURCE_ROWS = [
    {"statement": " first claim ", "refs": ["ref a", "ref b"], "verdict": "supported", "topic": "t1"},
    {"statement": "second claim", "refs": ["ref c"], "verdict": "refuted", "topic": "t2"},
    {"statement": "third claim", "refs": ["ref d"], "verdict": "supported", "topic": "t3"},
    {"statement": "fourth claim", "refs": ["ref e"], "verdict": "unclear", "topic": "t4"},
]


@pytest.fixture
def entry():
    return load_manifest(FIXTURE).get("tiny")


@pytest.fixture
def data_dir(tmp_path):
    """A fetched archive in the layout the fetch stage writes."""
    staged = tmp_path / "tiny__train.parquet"
    pl.DataFrame(SOURCE_ROWS).write_parquet(staged)
    with zipfile.ZipFile(tmp_path / "dataset-tiny.zip", "w") as z:
        z.write(staged, "tiny__train.parquet")
        z.writestr("_counts.json", json.dumps({"counts": {"train": 4}}))
    staged.unlink()
    return tmp_path


def test_declarative_mapping_produces_the_pair_schema(entry, data_dir):
    result = format_corpus(entry, data_dir)
    df = result.frame
    assert tuple(df.columns[: len(PAIR_COLUMNS)]) == PAIR_COLUMNS
    assert df.height == 3  # the unmapped `unclear` verdict is dropped
    assert df["label"].to_list() == [1, 0, 1]
    assert df["claim"][0] == "first claim"  # stripped
    assert df["chunk"][0] == "ref a\n\nref b"  # list joined
    assert df["source"].to_list() == ["train"] * 3
    assert df["tag"].to_list() == ["tiny"] * 3
    assert df["pair_id"].to_list() == [0, 1, 2]
    assert df["topic"].to_list() == ["t1", "t2", "t3"]  # retained
    assert df["raw_verdict"].to_list() == ["supported", "refuted", "supported"]  # renamed
    assert result.stats["dropped_unmapped_label"] == 1


def test_hashed_doc_id_groups_identical_evidence(entry, data_dir):
    df = format_corpus(entry, data_dir).frame
    assert df["doc_id"].n_unique() == 3
    assert all(len(d) == 16 for d in df["doc_id"].to_list())


def test_integrity_block_is_reported(entry, data_dir):
    result = format_corpus(entry, data_dir)
    assert result.integrity["pass"] is True
    assert result.integrity["duplicate_rows"] == 0
    assert result.integrity["distinct_documents"] == 3


def test_expected_counts_are_held(entry, data_dir):
    bad = entry.model_copy(update={"expected": entry.expected.model_copy(update={"rows": 99})})
    with pytest.raises(FormatError, match="want 99, got 3"):
        format_corpus(bad, data_dir)
    # the same mismatch is a report, not a refusal, when strict is off
    result = format_corpus(bad, data_dir, strict=False)
    assert result.stats["expected_vs_observed"]["rows"] == {"want": 99, "got": 3}


def test_duplicate_pairs_are_deduplicated(entry, tmp_path):
    staged = tmp_path / "tiny__train.parquet"
    pl.DataFrame(SOURCE_ROWS[:1] * 3).write_parquet(staged)
    with zipfile.ZipFile(tmp_path / "dataset-tiny.zip", "w") as z:
        z.write(staged, "tiny__train.parquet")
    result = format_corpus(entry, tmp_path, strict=False)
    assert result.rows == 1


def test_evidence_texts_are_deduplicated_evidence(entry, data_dir):
    texts = evidence_texts(entry, data_dir)
    assert texts == sorted({"ref a\n\nref b", "ref c", "ref d", "ref e"})


def test_write_lane_emits_parquet_and_manifest(entry, data_dir, tmp_path):
    out = tmp_path / "lanes"
    path = write_lane(format_corpus(entry, data_dir), entry, out)
    assert path.name == "tiny_lane.parquet"
    manifest = json.loads((out / "tiny_lane_manifest.json").read_text())
    assert manifest["rows"] == 3
    assert manifest["licence"] == "MIT"
    assert manifest["label_distribution"] == {"1": 2, "0": 1}
    assert set(manifest["provenance_columns"]) == {"topic", "raw_verdict"}


def test_adapter_path_is_used_when_named(tmp_path):
    @register_adapter("tiny_adapter")
    def _build(entry, data_dir):
        return [
            {"claim": "a", "chunk": "x" * 300, "label": 1, "doc_id": "d1", "source": "s"},
            {"claim": "b", "chunk": "y" * 300, "label": 0, "doc_id": "d1", "source": "s"},
        ], {"parsed": 2}

    entry = load_manifest(FIXTURE).get("tiny-adapter")
    result = format_corpus(entry, tmp_path)
    assert result.rows == 2
    assert result.stats["parsed"] == 2
    assert result.frame["tag"].to_list() == ["tiny-adapter"] * 2


def test_unknown_adapter_names_the_registered_ones(tmp_path):
    entry = load_manifest(FIXTURE).get("tiny-adapter")
    broken = entry.model_copy(
        update={"format": entry.format.model_copy(update={"adapter": "nope"})}
    )
    with pytest.raises(FormatError, match="not registered"):
        format_corpus(broken, tmp_path)


def test_missing_archive_is_a_clear_error(entry, tmp_path):
    with pytest.raises(FileNotFoundError, match="not fetched"):
        format_corpus(entry, tmp_path)

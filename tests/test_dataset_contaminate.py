"""The contamination gate, and the spike control that proves it can fire."""

import polars as pl
import pytest

from groundrails.dataset import contamination_check, contamination_gate, spike_control
from groundrails.dataset.contaminate import normalize, walled_texts_from_files

WALLED = {
    "arena": [
        "The quarterly report describes a material increase in operating revenue "
        "across the northern division during the second half of the fiscal year.",
        "Patients receiving the intervention showed a measurable reduction in "
        "symptom severity compared with the control arm over twelve weeks.",
    ]
}

CLEAN = [
    "A lighthouse keeper repaired the lamp mechanism before the winter storms arrived.",
    "The recipe calls for softened butter, caster sugar and two large free-range eggs.",
    "Volcanic soil retains moisture differently from the sandy loam further inland.",
]


def test_normalize_strips_punctuation_and_case():
    assert normalize("Hello,   WORLD!!") == "hello world"


def test_clean_corpus_passes():
    res = contamination_gate(CLEAN, WALLED, n=8, jaccard=0.3)
    assert res["verdict"] == "PASS"
    assert res["max_fraction"] == 0.0
    assert res["candidate_vs_walled"]["units_with_hit"] == 0


def test_a_copied_walled_document_is_killed():
    res = contamination_gate(CLEAN + [WALLED["arena"][0]], WALLED, n=8, jaccard=0.3)
    assert res["verdict"] == "KILL"
    assert res["max_fraction"] >= 0.02
    assert res["candidate_vs_walled"]["per_walled_bucket"]["arena"]["units_with_hit"] == 1
    assert res["hit_examples"][0]["jaccard"] == 1.0


def test_containment_mode_fires_on_a_shared_ngram():
    borrowed = "Nothing here except that " + " ".join(WALLED["arena"][1].split()[:12])
    res = contamination_gate(CLEAN + [borrowed], WALLED, n=8, jaccard=None)
    assert res["mode"] == "containment"
    assert res["candidate_vs_walled"]["units_with_hit"] == 1


def test_spike_control_detects_every_injected_unit():
    spike = spike_control(CLEAN, WALLED, n=8, jaccard=0.3, k=2)
    assert spike["injected"] == 2
    assert spike["detected_total"] == 2
    assert spike["baseline_hits"] == 0
    assert spike["baseline_clean"] is True
    assert spike["passes"] is True


def test_spike_control_separates_baseline_hits_from_the_injection():
    spike = spike_control(CLEAN + [WALLED["arena"][0]], WALLED, n=8, jaccard=0.3, k=2)
    assert spike["baseline_hits"] == 1
    assert spike["baseline_clean"] is False
    assert spike["passes"] is True  # the instrument fired; the fraction bar rules on the hit


def test_a_gate_that_cannot_fire_fails_the_control():
    # n larger than any walled document has tokens: no n-gram can ever match
    spike = spike_control(CLEAN, WALLED, n=500, jaccard=0.3, k=2)
    assert spike["detected_total"] == 0
    assert spike["passes"] is False


def test_check_reports_one_status():
    green = contamination_check(CLEAN, WALLED, label="clean")
    assert green["status"] == "GREEN"
    assert green["spike_control"]["passes"]
    red = contamination_check(CLEAN + [WALLED["arena"][0]], WALLED, label="dirty")
    assert red["status"] == "RED"
    assert red["gate"]["verdict"] == "KILL"


def test_walled_texts_from_files_buckets_by_stem(tmp_path):
    p = tmp_path / "arena.parquet"
    pl.DataFrame({"chunk": WALLED["arena"]}).write_parquet(p)
    q = tmp_path / "notes.txt"
    q.write_text("one line\n\nanother line\n", encoding="utf-8")
    loaded = walled_texts_from_files([p, q])
    assert set(loaded) == {"arena", "notes"}
    assert loaded["arena"] == WALLED["arena"]
    assert loaded["notes"] == ["one line", "another line"]


def test_walled_texts_joins_a_list_column(tmp_path):
    p = tmp_path / "docs.parquet"
    pl.DataFrame({"chunk": [["a b", "c d"]]}).write_parquet(p)
    assert walled_texts_from_files([p])["docs"] == ["a b c d"]


def test_walled_texts_rejects_a_missing_column(tmp_path):
    p = tmp_path / "docs.parquet"
    pl.DataFrame({"text": ["a"]}).write_parquet(p)
    with pytest.raises(ValueError, match="chunk"):
        walled_texts_from_files([p])

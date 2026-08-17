"""C2 supplement for `frame_reject` - the arena's CLAIM side at SENTENCE grain.

The main verification (`frame_reject_contract_verify.py`) crosses the member
against arena DOCUMENTS and whole arena RESPONSES.  The blind windowed read
scores a response's SENTENCES against the evidence bag, so the response sentence
is the arena's actual claim unit and is checked here as well.

CPU only.  Reads the RAGBench archive and the lane parquet; writes one JSON.

Run:  uv run python experiments/grounding-semantic/contract/frame_reject_c2_sentence_supplement.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import io
import json
from pathlib import Path
import zipfile

import polars as pl

HERE = Path(__file__).parent
EXP = HERE.parent
ROOT = EXP.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
LANE = EXP / "R20-H174_lane_L1.parquet"
OUT = HERE / "frame_reject_c2_sentence_supplement.json"
CHUNK_MAX = 1500


def norm(s):
    return " ".join(s.split()).casefold()


def arena_sentences():
    z = zipfile.ZipFile(DATA / "dataset-ragbench.zip")
    per, allsent = {}, set()
    for name in sorted(n for n in z.namelist() if n.endswith("__test.parquet")):
        sub = name.split("__")[2]
        df = pl.read_parquet(io.BytesIO(z.read(name)))
        s = set()
        for row in df["response_sentences"].to_list():
            for item in (row or []):
                # each item is [key, sentence]
                if item is not None and len(item) >= 2 and item[1]:
                    s.add(item[1])
        per[sub] = {"rows": df.height, "response_sentences": len(s)}
        allsent |= s
    return allsent, per


def forms(texts, cut):
    raw = set(texts)
    return {"raw": raw, "truncated": {t[:cut] for t in raw},
            "normalised": {norm(t) for t in raw}}


def cross(a, b, cut):
    fa, fb = forms(a, cut), forms(b, cut)
    return {"raw_vs_raw": len(fa["raw"] & fb["raw"]),
            "truncated_vs_truncated": len(fa["truncated"] & fb["truncated"]),
            "normalised_vs_normalised": len(fa["normalised"] & fb["normalised"]),
            "raw_vs_truncated": len(fa["raw"] & fb["truncated"]),
            "truncated_vs_raw": len(fa["truncated"] & fb["raw"])}


def main():
    df = pl.read_parquet(LANE)
    neg = df.filter(pl.col("label") == 0)
    pos = df.filter(pl.col("label") == 1)
    m_claims = sorted({c for c in df["claim"].to_list() if c.strip()})
    m_neg = sorted({c for c in neg["claim"].to_list() if c.strip()})
    m_pos = sorted({c for c in pos["claim"].to_list() if c.strip()})

    sents, per = arena_sentences()
    print(f"arena response sentences: {len(sents)} distinct", flush=True)

    sent_norm = {norm(s) for s in sents}
    hits_neg = sorted({c for c in m_neg if norm(c) in sent_norm})
    hits_pos = sorted({c for c in m_pos if norm(c) in sent_norm})

    # the four hagrid frame-only artifact items the lane was built against
    artifact = "Based on the given context ,"
    z = zipfile.ZipFile(DATA / "dataset-ragbench.zip")
    hag = pl.read_parquet(io.BytesIO(z.read("galileo-ai__ragbench__hagrid__test.parquet")))
    resp = hag["response"].to_list()

    out = {
        "supplement": "C2 - arena CLAIM side at response-sentence grain",
        "member": "frame_reject",
        "arena_response_sentences": {"distinct": len(sents), "per_subset": per},
        "member_units": {"distinct_claims": len(m_claims),
                         "distinct_negative_claims": len(m_neg),
                         "distinct_positive_claims": len(m_pos)},
        "member_claims_vs_arena_response_sentences": cross(m_claims, sents, CHUNK_MAX),
        "member_negative_claims_vs_arena_response_sentences": cross(m_neg, sents, CHUNK_MAX),
        "member_positive_claims_vs_arena_response_sentences": cross(m_pos, sents, CHUNK_MAX),
        "colliding_negative_claims": {
            "count": len(hits_neg),
            "rows_of_the_member_carrying_one": int(
                neg.filter(pl.col("claim").is_in(hits_neg)).height) if hits_neg else 0,
            "examples": hits_neg[:15]},
        "colliding_positive_claims": {
            "count": len(hits_pos),
            "rows_of_the_member_carrying_one": int(
                pos.filter(pl.col("claim").is_in(hits_pos)).height) if hits_pos else 0,
            "examples": hits_pos[:15]},
        "hagrid_frame_only_artifact": {
            "string": artifact,
            "hagrid_test_responses_equal_to_it": sum(1 for t in resp if t == artifact),
            "hagrid_test_responses_opening_with_the_frame": sum(
                1 for t in resp if t and norm(t).startswith("based on the given context")),
            "member_negative_claims_equal_to_it": sum(1 for c in m_neg if c == artifact),
            "member_rows_equal_to_it": int(neg.filter(pl.col("claim") == artifact).height),
            "member_negative_claims_normalising_to_it": sum(
                1 for c in m_neg if norm(c) == norm(artifact)),
            "member_rows_normalising_to_it": int(
                neg.filter(pl.col("claim").map_elements(
                    lambda s: norm(s) == norm(artifact), return_dtype=pl.Boolean)).height),
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("arena_response_sentences",)}, indent=2)[:4000], flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

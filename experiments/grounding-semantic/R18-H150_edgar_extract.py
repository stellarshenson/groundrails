"""R18-H150 amendment A2 - EDGAR MD&A prose extraction + magnitude census, CPU only.

Stage 1 of the scale_word extension.  Reads the R14-H136 restricted EDGAR slice
(non-S&P-500 filers, filing year >= 2020 - the clauses that make it document-
disjoint from FinQA's source population), cuts section_7 (MD&A) into the
project's 1,500-char windows, and censuses the magnitude vocabulary the prose
actually carries.

Nothing here enters a lane: `R18-H150_edgar_gate.py` runs the R14-H136
provenance gate on this output first, and the lane build refuses to start
without a green gate sidecar.

Run:  uv run python experiments/grounding-semantic/R18-H150_edgar_extract.py
"""

import collections
import json
import pathlib
import re

import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
SRC = ROOT / "data" / "external" / "datasets" / "edgar-restricted" / "filings"
SIDECAR = ROOT / "data" / "external" / "datasets" / "dataset-edgar-restricted.md"
OUT = HERE / "R18-H150_edgar_chunks.parquet"
CENSUS = HERE / "R18-H150_edgar_census.json"

CHUNK = 1500
MIN_MDA = 500
DOC_CHUNK_CAP = 20          # windows kept per filing, front-loaded
WS = re.compile(r"\s+")

# magnitude vocabulary, as the prose writes it
MAG = {
    "thousand": re.compile(r"\bthousands?\b", re.IGNORECASE),
    "million":  re.compile(r"\bmillions?\b", re.IGNORECASE),
    "billion":  re.compile(r"\bbillions?\b", re.IGNORECASE),
    "trillion": re.compile(r"\btrillions?\b", re.IGNORECASE),
}
PP = re.compile(r"percentage\s+points?|\bbasis\s+points?\b", re.IGNORECASE)
PCT = re.compile(r"%|\bpercent\b", re.IGNORECASE)
# "$12.4 million" / "12.4 million" / "$1,234" style money quantities
MONEY_MAG = re.compile(
    r"(\$\s?)?(\d[\d,]*(?:\.\d+)?)\s+(thousands?|millions?|billions?|trillions?)\b",
    re.IGNORECASE)
SCALE_CAPTION = re.compile(
    r"\(\s*(?:dollars|amounts|in)\s[^)]{0,40}?(thousands?|millions?)\s*\)|"
    r"\bin\s+(thousands?|millions?)\b(?:\s+of\s+dollars)?", re.IGNORECASE)


def clean(s):
    return WS.sub(" ", s or "").strip()


def windows(text):
    """Non-overlapping 1,500-char windows cut on a whitespace boundary."""
    out, i = [], 0
    while i < len(text):
        j = min(i + CHUNK, len(text))
        if j < len(text):
            k = text.rfind(" ", i + CHUNK // 2, j)
            if k > i:
                j = k
        w = text[i:j].strip()
        if len(w) >= 400:
            out.append(w)
        i = j
    return out


def main():
    assert SIDECAR.exists(), f"licence sidecar missing: {SIDECAR}"
    frames = []
    for shard in ("train", "validate", "test"):
        p = SRC / f"{shard}.parquet"
        if not p.exists():
            continue
        frames.append(pl.read_parquet(p, columns=["filename", "cik", "year", "section_7"])
                      .with_columns(pl.lit(shard).alias("shard")))
    d = pl.concat(frames)
    print(f"filings on disk: {d.height}", flush=True)

    rows, per_doc_kept = [], collections.Counter()
    mag_chunks = collections.Counter()
    multi_mag_chunks = collections.Counter()
    pp_chunks = 0
    for fn, cik, yr, sec, shard in d.iter_rows():
        sec = clean(sec)
        if len(sec) < MIN_MDA:
            continue
        doc_id = f"edgar:{cik}:{fn}"
        for wi, w in enumerate(windows(sec)[:DOC_CHUNK_CAP]):
            present = {u for u, rx in MAG.items() if rx.search(w)}
            for u in present:
                mag_chunks[u] += 1
            if len(present) >= 2:
                for a in present:
                    for b in present:
                        if a < b:
                            multi_mag_chunks[f"{a}|{b}"] += 1
            if PP.search(w):
                pp_chunks += 1
            if not present and not PP.search(w):
                continue
            rows.append({"doc_id": doc_id, "cik": cik, "year": yr, "shard": shard,
                         "window_index": wi, "chunk": w,
                         "n_magnitude_words": len(present),
                         "magnitude_words": ",".join(sorted(present)),
                         "has_pp": bool(PP.search(w)),
                         "has_pct": bool(PCT.search(w)),
                         "has_scale_caption": bool(SCALE_CAPTION.search(w))})
            per_doc_kept[doc_id] += 1

    df = pl.DataFrame(rows).unique(subset=["chunk"], keep="first", maintain_order=True)
    df.write_parquet(OUT)
    print(f"magnitude-bearing chunks: {df.height} over {df['doc_id'].n_unique()} filings",
          flush=True)

    money = collections.Counter()
    for w in df["chunk"].to_list():
        for _, _, unit in MONEY_MAG.findall(w):
            money[unit.lower().rstrip("s")] += 1

    census = {
        "source": "edgar-restricted (R14-H136 slice: non-S&P-500 filers, year >= 2020)",
        "filings_on_disk": d.height,
        "filings_with_mda": int((d["section_7"].str.len_chars() >= MIN_MDA).sum()),
        "chunk_chars": CHUNK, "doc_chunk_cap": DOC_CHUNK_CAP,
        "magnitude_bearing_chunks": df.height,
        "documents": df["doc_id"].n_unique(),
        "chunks_by_magnitude_word": dict(mag_chunks.most_common()),
        "chunks_with_two_or_more_magnitude_words": dict(multi_mag_chunks.most_common()),
        "multi_magnitude_chunk_total": int((df["n_magnitude_words"] >= 2).sum()),
        "chunks_with_percentage_or_basis_points": pp_chunks,
        "chunks_with_scale_caption": int(df["has_scale_caption"].sum()),
        "quantity_mentions_by_magnitude_word": dict(money.most_common()),
    }
    CENSUS.write_text(json.dumps(census, indent=2))
    print(json.dumps(census, indent=2), flush=True)


if __name__ == "__main__":
    main()

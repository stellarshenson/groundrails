"""Contract verification stage A - rebuild `ragtruth_translated` through the BANKED loader.

CPU ONLY. The member is the 7 non-English RAGTruth translation slices as
`R10-H108_lane.public_train()` emits them, read the way the R18-H150 / R20-H174
arms read them (evidence UNTRUNCATED, `untruncated_evidence` inlined exactly as
`R20-H175b_qlane_eval_clean.assemble_mix` does it). Nothing about the member is
re-implemented here; the loader is called and its rows are filtered by DANN tag.

Writes `ragtruth_translated_member.parquet` (claim, chunk, label, tag) plus the
truncated-to-`chunk_max_chars` evidence form both protocols use.

Run:  uv run python experiments/grounding-semantic/contract/ragtruth_translated_extract.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import importlib.util as _ilu
import json
from pathlib import Path

import polars as pl

HERE = Path(__file__).parent
SEM = HERE.parent
OUT = HERE / "ragtruth_translated_member.parquet"
META = HERE / "ragtruth_translated_extract.json"

LANGS = ("de", "fr", "es", "it", "pl", "hu", "cn")
MEMBER_TAGS = tuple(f"ragtruth_{lg}" for lg in LANGS)


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    H108 = _mod("h108lane", SEM / "R10-H108_lane.py")
    M59 = H108.M59
    chunk_max = M59.CFG.chunk_max_chars
    print(f"chunk_max_chars = {chunk_max}", flush=True)

    original = M59.CFG.chunk_max_chars
    M59.CFG.chunk_max_chars = 10**9  # `untruncated_evidence`, inlined
    try:
        claims, chunks, y, tags = H108.public_train()
    finally:
        M59.CFG.chunk_max_chars = original
    print(f"clean public mix: {len(y)} rows over {len(set(tags))} groups", flush=True)

    df = pl.DataFrame(
        {"claim": claims, "chunk": chunks, "label": y, "tag": tags},
        schema={"claim": pl.Utf8, "chunk": pl.Utf8, "label": pl.Float32, "tag": pl.Utf8},
    )
    counts_all = dict(df.group_by("tag").len().sort("tag").iter_rows())
    mem = df.filter(pl.col("tag").is_in(MEMBER_TAGS))
    mem = mem.with_columns(
        pl.col("chunk").str.slice(0, chunk_max).alias("chunk_trunc"),
        pl.col("tag").str.slice(len("ragtruth_")).alias("lang"),
    )
    mem.write_parquet(OUT)

    meta = {
        "loader": "R10-H108_lane.public_train() with the evidence cut lifted",
        "chunk_max_chars": chunk_max,
        "clean_public_rows": int(len(y)),
        "clean_public_rows_per_group": {k: int(v) for k, v in counts_all.items()},
        "member_tags": list(MEMBER_TAGS),
        "member_rows": int(mem.height),
        "member_rows_per_lang": {
            k: int(v) for k, v in mem.group_by("lang").len().sort("lang").iter_rows()
        },
        "member_label_mean": round(float(mem["label"].mean()), 6),
        "parquet": str(OUT),
    }
    META.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()

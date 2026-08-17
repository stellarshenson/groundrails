"""Rebuild the `tabfact` training member THROUGH THE BANKED LOADER and cache it.

CPU only. The member is whatever `R10-H108_lane.public_train()` tags `tabfact`,
read under `R16-H142_G1_arm.untruncated_evidence()` - the presentation the
R18-H150 flagship and the R20-H174 arm actually train on (evidence untruncated,
windowed 1,500/750 downstream). The truncated form (chunk_max_chars = 1,500) is
cached alongside because contract clause C2 requires both string forms.

No re-implementation: the loader is imported and called, and the cached slice is
asserted row-for-row against the archive's own train split.

Out: tabfact_member.parquet  (claim / chunk_untrunc / chunk_trunc / label /
                              table_id / table_caption)
     tabfact_load.json
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # GPUs 0/1/2 are training R20-H174
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util as _ilu
import io
import json
import pathlib
import time
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
DATA = SEM.parent.parent / "data" / "external" / "datasets"
OUT_PARQUET = HERE / "tabfact_member.parquet"
OUT_JSON = HERE / "tabfact_load.json"

EXPECTED_CLEAN_ROWS = 685_670   # R18-H150_arm_run.EXPECTED_CLEAN_ROWS


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    t0 = time.time()
    arm = _mod("g1arm", SEM / "R16-H142_G1_arm.py")
    # `arm.untruncated_evidence()` patches `arm.M59`, which IS `arm.H108.M59`.
    # A separately-loaded H108 instance carries its own M59 and would stay
    # truncated - so the loader is reached through the arm, exactly as
    # R18-H150_arm_run.make_build_mix does.
    H108 = arm.H108
    chunk_max = H108.M59.CFG.chunk_max_chars
    print(f"chunk_max_chars = {chunk_max}", flush=True)

    with arm.untruncated_evidence():
        if H108.M59.CFG.chunk_max_chars != 10**9:
            raise SystemExit("UNTRUNCATED ABORT: the cut was not lifted on the loader's own M59")
        claims, chunks, y, tags = H108.public_train()
    if len(y) != EXPECTED_CLEAN_ROWS:
        raise SystemExit(f"MIX ABORT: clean mix {len(y)} rows, expected {EXPECTED_CLEAN_ROWS}")
    print(f"clean mix {len(y)} rows over {len(set(tags))} groups", flush=True)

    idx = [i for i, t in enumerate(tags) if t == "tabfact"]
    m_claims = [claims[i] for i in idx]
    m_chunks = [chunks[i] for i in idx]
    m_y = np.asarray(y, dtype="float32")[idx]
    del claims, chunks, tags, y

    # --- cross-check against the archive, same selection predicate -----------
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    train_name = next(x for x in z.namelist() if x.endswith("__train.parquet"))
    raw = pl.read_parquet(io.BytesIO(z.read(train_name)))
    sel = raw.filter(pl.col("statement").str.len_chars() > 10)
    if sel.height != len(m_claims):
        raise SystemExit(f"ALIGN ABORT: archive selection {sel.height} != member {len(m_claims)}")
    if sel["statement"].to_list() != m_claims:
        raise SystemExit("ALIGN ABORT: member claims are not the archive statements in order")
    if not np.array_equal(m_y, sel["label"].cast(pl.Float32).to_numpy()):
        raise SystemExit("ALIGN ABORT: member labels are not the archive labels in order")
    print("alignment to the archive train split: exact (claims and labels)", flush=True)

    df = pl.DataFrame({
        "claim": m_claims,
        "chunk_untrunc": m_chunks,
        "chunk_trunc": [c[:chunk_max] for c in m_chunks],
        "label": m_y,
        "table_id": sel["table_id"].to_list(),
        "table_caption": sel["table_caption"].to_list(),
    })
    df.write_parquet(OUT_PARQUET)

    meta = {
        "member": "tabfact",
        "loader": "R10-H108_lane.public_train() under "
                  "R16-H142_G1_arm.untruncated_evidence(), rows tagged `tabfact`",
        "archive": "dataset-tabfact.zip :: " + train_name,
        "selection_predicate": "train split only; filter statement.str.len_chars() > 10",
        "chunk_construction": "f'{table_caption}\\n{table_text}'"
                              ".replace('\\r\\n','\\n').replace('#',' | ')[:chunk_max_chars]",
        "chunk_max_chars": chunk_max,
        "clean_mix_rows": EXPECTED_CLEAN_ROWS,
        "member_rows": df.height,
        "archive_train_rows": raw.height,
        "rows_dropped_by_filter": raw.height - df.height,
        "member_share_of_clean_mix": round(df.height / EXPECTED_CLEAN_ROWS, 6),
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()

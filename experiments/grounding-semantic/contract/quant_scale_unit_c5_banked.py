"""C5 - the registered leak suite RE-DERIVED through the lane's own BANKED
`verify()` instrument, for `quant_scale_unit`.

CPU ONLY.  PYTHONHASHSEED=0 (the lane's recorded reproducibility condition).

The executor's first pass reimplemented the probe with unstratified
document folds and read within-pair 0.7815 on one family.  That is the
artifact the lane's build documented (R17-H145 finding b: with a linear probe a
fold holding `a` pairs of one direction and `b` of the other forces within-pair
accuracy to exactly min(a,b)/(a+b), so unstratified folds read off chance in
BOTH directions however clean the data is).  This script therefore calls
`R18-H150_scaleunit_lane.verify` itself - direction-stratified greedy pair-count
fold packing, char_wb TF-IDF, liblinear C=4.0 tol 1e-7 - over the banked
parquet, rebuilding the corpus tables the re-derivation audit needs.

CAVEAT, stated rather than hidden: `verify()` shuffles documents from the build's
`random.Random(SEED)` at whatever state it had reached by verify time.  That
state cannot be recovered without replaying the whole build, so this run seeds a
FRESH `random.Random(1150)`.  The fold assignment therefore differs from the
build's and the numbers are a re-derivation, not a byte reproduction.

Run:  PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES= uv run python \
      experiments/grounding-semantic/contract/quant_scale_unit_c5_banked.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib.util
import json
import pathlib
import random

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
LANE = SEM / "R18-H150_scaleunit_lane.parquet"
OUT = HERE / "quant_scale_unit_c5_banked.json"


def main():
    spec = importlib.util.spec_from_file_location(
        "h150lane", SEM / "R18-H150_scaleunit_lane.py")
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    P = L.P

    rng = random.Random(L.SEED)
    np_rng = np.random.default_rng(L.SEED)

    excluded_ids, prints, eval_rows, unmatched = P.evalset_documents()
    print(f"eval set: {eval_rows} rows -> {len(excluded_ids)} doc_ids, "
          f"{len(prints)} fingerprints ({unmatched} unmatched)", flush=True)

    raw = P.tabfact_tables() + P.feverous_tables()
    drop_idx = P.excluded_tables(raw, prints)
    tables, dropped_eval = [], 0
    for ti, t in enumerate(raw):
        if ti in drop_idx or t["doc_id"] in excluded_ids:
            dropped_eval += 1
            continue
        lab = P.label_column(t["hdr"], t["body"])
        if lab is None:
            continue
        t["lab_ci"] = lab
        tables.append(t)
    print(f"{len(raw)} candidate tables; {len(drop_idx)} carry eval content; "
          f"{dropped_eval} dropped; {len(tables)} admitted", flush=True)

    forms = list(L.FORM_WEIGHTS)
    w = np.array([L.FORM_WEIGHTS[f] for f in forms], dtype=float)
    w /= w.sum()
    for t, k in zip(tables, np_rng.choice(len(forms), size=len(tables), p=w)):
        t["form"] = forms[int(k)]
    by_doc = {t["doc_id"]: t for t in tables}

    df = pl.read_parquet(LANE)
    resolved = sum(1 for d in df["doc_id"].unique().to_list() if d in by_doc)
    print(f"lane documents resolved against the rebuilt corpus: "
          f"{resolved}/{df['doc_id'].n_unique()}", flush=True)

    res = L.verify(df, rng, by_doc)
    out = {
        "member": "quant_scale_unit",
        "instrument": "R18-H150_scaleunit_lane.verify (BANKED, unedited)",
        "reproduction_caveat": (
            "fresh random.Random(1150); the build's rng state at verify time is "
            "unrecoverable, so fold assignment differs from the banked run"),
        "corpus_rebuild": {
            "candidate_tables": len(raw),
            "tables_carrying_eval_content": len(drop_idx),
            "dropped": dropped_eval,
            "admitted": len(tables),
            "lane_documents_resolved": resolved,
            "lane_documents": int(df["doc_id"].n_unique()),
        },
        "verify": res,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

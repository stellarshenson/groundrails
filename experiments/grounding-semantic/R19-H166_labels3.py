"""R19-H166 THREE-WAY LABEL RECOVERY - the signal the mix already paid for and throws away.

The campaign's lane schema stores `label: pl.Float32`, one scalar per row, and the
task head is `nn.Linear(768, 1)`. So the model has never been asked to tell "the
evidence does not mention this" from "the evidence says otherwise".

That distinction is ALREADY PRESENT in the largest lane. `R10-H108_lane.py:154`
reads VitaminC's native label column and collapses it in a single expression:

    (pl.col(lab_col).cast(pl.Utf8).str.to_uppercase() == "SUPPORTS").cast(pl.Float32)

370,653 rows - 34.69% of the mix - arrive as SUPPORTS 185,714 / REFUTES 131,958 /
NOT ENOUGH INFO 52,981, and 131,958 contradictions become indistinguishable from
52,981 absences at that line.

WHAT THIS MODULE DOES
---------------------
Recovers a three-way label ALIGNED to the mix's own row order, without touching
the banked loader. Class codes:

    2  CONTRADICTED  the evidence asserts something incompatible with the claim
    1  SUPPORTED     the evidence entails the claim
    0  ABSENT        the evidence simply does not carry the claim
   -1  MASKED        no recoverable three-way label; the auxiliary term is skipped

ALIGNMENT, and why it is exact rather than a join
-------------------------------------------------
`R10-H108_lane.py` appends VitaminC contiguously, in the zip's own row order, via
`claims += dv[cl_col].to_list()`. So the i-th row tagged `vitaminc` in the mix is
the i-th row of the zip's `__train.parquet`. This module re-reads that parquet by
the same selection rules and indexes positionally. A content join is deliberately
NOT used: `public_train` truncates chunks to `chunk_max_chars` while the twin
protocol reads them untruncated, so the same row has different text on the two
paths and a text key would silently mis-join.

A count assertion is the guard - if the parquet does not yield exactly as many
rows as the mix tagged `vitaminc`, the alignment premise is broken and this
raises rather than mislabelling 370,653 rows.

SOURCES OF THE THIRD CLASS
--------------------------
- **vitaminc** - native. SUPPORTS -> 1, REFUTES -> 2, NOT ENOUGH INFO -> 0
- **quant_misbind** - contradiction BY CONSTRUCTION. Its negatives bind a value to
  the wrong column or row, so the evidence positively asserts a different value
  for the cell the claim names; that is contradiction, not absence. Positives -> 1,
  negatives -> 2
- **everything else** - MASKED. Guessing a third class for corpora whose negatives
  mix absence and contradiction would inject label noise into 65% of the mix, and
  a guessed label is worse than an absent one.

Run standalone to print the census:
    uv run python experiments/grounding-semantic/R19-H166_labels3.py
"""

import importlib.util
import io
import json
import pathlib
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent

CONTRADICTED, SUPPORTED, ABSENT, MASKED = 2, 1, 0, -1

VITAMINC_MAP = {"SUPPORTS": SUPPORTED, "REFUTES": CONTRADICTED,
                "NOT ENOUGH INFO": ABSENT}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _vitaminc_three_way(lane):
    """VitaminC's native labels, in the zip's row order - the mix's own order."""
    zv = zipfile.ZipFile(lane.DATA / "dataset-vitaminc.zip")
    name = next(x for x in zv.namelist() if x.endswith("__train.parquet"))
    dv = pl.read_parquet(io.BytesIO(zv.read(name)))
    lab_col = next(c for c in ("label", "labels") if c in dv.columns)
    raw = [str(x).upper().strip() for x in dv[lab_col].to_list()]
    unknown = sorted({r for r in raw if r not in VITAMINC_MAP})
    if unknown:
        raise SystemExit(
            f"H166 ABORT: unmapped VitaminC labels {unknown} - the three-way "
            f"recovery must not silently bucket a class it does not recognise")
    return np.array([VITAMINC_MAP[r] for r in raw], dtype=np.int8)


def build(tags, y):
    """Three-way labels aligned to the mix's row order.

    tags / y are exactly what `build_mix` returned, so alignment is positional
    and total. Returns int8 array of the same length.
    """
    lane = _mod("h108lane", "R10-H108_lane.py")
    tags = list(tags)
    y = np.asarray(y)
    y3 = np.full(len(tags), MASKED, dtype=np.int8)

    idx_v = np.flatnonzero(np.array(tags) == "vitaminc")
    vit = _vitaminc_three_way(lane)
    if idx_v.size != vit.size:
        raise SystemExit(
            f"H166 ALIGNMENT ABORT: mix tags {idx_v.size} vitaminc rows but the "
            f"zip yields {vit.size}. Positional alignment is the whole premise of "
            f"this module - refusing to mislabel on a broken assumption.")
    y3[idx_v] = vit

    # Cross-check the recovered labels against the binary the mix already carries:
    # SUPPORTED must coincide exactly with y == 1 on these rows, because the mix's
    # own binary was derived from the same column by the same rule.
    got_bin = (y3[idx_v] == SUPPORTED).astype(np.float32)
    if not np.array_equal(got_bin, y[idx_v].astype(np.float32)):
        n_bad = int((got_bin != y[idx_v]).sum())
        raise SystemExit(
            f"H166 CONSISTENCY ABORT: recovered VitaminC labels disagree with the "
            f"mix's own binary on {n_bad} rows - the row order does NOT match and "
            f"a positional assignment would be wrong.")

    idx_m = np.flatnonzero(np.array(tags) == "quant_misbind")
    y3[idx_m] = np.where(y[idx_m] > 0.5, SUPPORTED, CONTRADICTED)

    return y3


def census(tags, y, y3):
    tags = np.array(tags)
    out = {"n_rows": int(len(y3)),
           "n_labelled": int((y3 != MASKED).sum()),
           "labelled_share": round(float((y3 != MASKED).mean()), 4),
           "by_class": {"contradicted": int((y3 == CONTRADICTED).sum()),
                        "supported": int((y3 == SUPPORTED).sum()),
                        "absent": int((y3 == ABSENT).sum()),
                        "masked": int((y3 == MASKED).sum())},
           "by_group": {}}
    for t in sorted(set(tags.tolist())):
        m = tags == t
        out["by_group"][t] = {
            "rows": int(m.sum()),
            "contradicted": int((y3[m] == CONTRADICTED).sum()),
            "supported": int((y3[m] == SUPPORTED).sum()),
            "absent": int((y3[m] == ABSENT).sum()),
            "masked": int((y3[m] == MASKED).sum()),
        }
    return out


def main():
    arm = _mod("g1arm", "R16-H142_G1_arm.py")
    h150 = _mod("h150run", "R18-H150_arm_run.py")
    h150.rebind(arm)
    claims, wsets, y, tags = arm.build_mix()
    y3 = build(tags, y)
    c = census(tags, y, y3)
    out = HERE / "R19-H166_labels3_census.json"
    out.write_text(json.dumps(c, indent=1))
    print(f"\nrows {c['n_rows']}  labelled {c['n_labelled']} "
          f"({c['labelled_share']:.1%})", flush=True)
    print(f"  contradicted {c['by_class']['contradicted']}  "
          f"supported {c['by_class']['supported']}  "
          f"absent {c['by_class']['absent']}  masked {c['by_class']['masked']}",
          flush=True)
    for t, g in c["by_group"].items():
        if g["masked"] != g["rows"]:
            print(f"  {t:<18} rows {g['rows']:>7}  C {g['contradicted']:>7}  "
                  f"S {g['supported']:>7}  A {g['absent']:>7}", flush=True)
    print(f"  -> {out.name}", flush=True)
    print("=== H166 LABEL CENSUS COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()

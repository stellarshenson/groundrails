"""R17-H143 Stage A - eval-set builder (CPU, Polars).

Snapshots a 1,000-pair stratified sample from the H133 lane v2 parquet
(R14-H133_lane.v2-SUPERSEDED.parquet - the registered eval source; the
unsuffixed R14-H133_lane.parquet is the v3 trace-conditioned rebuild and is NOT
the eval source) plus 50 trivially separable positive-control pairs.

Sampling: 500 pair_ids drawn proportionally across neg_family (the negative
member's family); both members of each pair are taken, giving 500 positives and
500 negatives with matched chunks. numpy seed 1143.

Controls: 25 claims lifted verbatim from a chunk sentence (label 1) and 25 with
the sentence's number replaced by a value absent from the chunk (label 0).
Tagged control=True, pair_id negative.

Run:
  python experiments/grounding-semantic/R17-H143_evalset.py
"""

import pathlib
import re

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
SRC = HERE / "R17-H143_evalset_source.parquet"
OUT = HERE / "R17-H143_evalset.parquet"
SEED = 1143
N_PAIRS = 500
N_CONTROL_EACH = 25

# quant_relational rows carry no neg_family (they are binding/comparison arms,
# not a derivation negative family); name them so per-family reads are total.
RELATIONAL_FAMILY = "r:relational"


def build_sample(df: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("neg_family").fill_null(RELATIONAL_FAMILY).alias("neg_family")
    )
    negs = df.filter(pl.col("label") == 0.0)
    fam_counts = (
        negs.group_by("neg_family").len().sort("neg_family").to_dicts()
    )
    total = sum(r["len"] for r in fam_counts)

    # proportional allocation, largest-remainder so the quota sums exactly
    raw = {r["neg_family"]: N_PAIRS * r["len"] / total for r in fam_counts}
    alloc = {k: int(np.floor(v)) for k, v in raw.items()}
    rem = N_PAIRS - sum(alloc.values())
    for fam in sorted(raw, key=lambda k: raw[k] - alloc[k], reverse=True)[:rem]:
        alloc[fam] += 1

    chosen: list[int] = []
    for fam in sorted(alloc):
        ids = np.sort(negs.filter(pl.col("neg_family") == fam)["pair_id"].unique().to_numpy())
        take = min(alloc[fam], len(ids))
        chosen.extend(rng.choice(ids, size=take, replace=False).tolist())

    sample = df.filter(pl.col("pair_id").is_in(chosen))
    return sample.select(
        "pair_id",
        "claim",
        "chunk",
        pl.col("label").cast(pl.Int8),
        "neg_family",
        "tag",
        pl.col("claim_form").fill_null("relational"),
        "operand_a",
        "operand_b",
        pl.lit(False).alias("control"),
    )


SENT_RE = re.compile(r"(The [^.]*? is ([0-9][0-9,.]*)\.)")


def build_controls(df: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """25 verbatim-present claims (label 1), 25 value-absent claims (label 0)."""
    chunks = df["chunk"].unique().sort().to_list()
    picks = rng.choice(len(chunks), size=min(400, len(chunks)), replace=False)
    rows: list[dict] = []
    pos_done = neg_done = 0
    for idx in picks:
        if pos_done >= N_CONTROL_EACH and neg_done >= N_CONTROL_EACH:
            break
        chunk = chunks[int(idx)]
        matches = SENT_RE.findall(chunk)
        if not matches:
            continue
        sentence, value = matches[len(matches) // 2]
        want_pos = pos_done <= neg_done and pos_done < N_CONTROL_EACH
        if want_pos:
            rows.append(
                dict(
                    pair_id=-(len(rows) + 1),
                    claim=sentence,
                    chunk=chunk,
                    label=1,
                    neg_family="control:verbatim",
                    tag="control",
                    claim_form="control",
                    operand_a=None,
                    operand_b=None,
                    control=True,
                )
            )
            pos_done += 1
        elif neg_done < N_CONTROL_EACH:
            # replace the value with one that appears nowhere in the chunk
            for _ in range(200):
                fake = str(int(rng.integers(10_000, 99_999)))
                if fake not in chunk:
                    break
            else:
                continue
            rows.append(
                dict(
                    pair_id=-(len(rows) + 1),
                    claim=sentence.replace(value, fake),
                    chunk=chunk,
                    label=0,
                    neg_family="control:absent",
                    tag="control",
                    claim_form="control",
                    operand_a=None,
                    operand_b=None,
                    control=True,
                )
            )
            neg_done += 1
    assert pos_done == N_CONTROL_EACH and neg_done == N_CONTROL_EACH, (pos_done, neg_done)
    return pl.DataFrame(
        rows,
        schema={
            "pair_id": pl.Int64,
            "claim": pl.String,
            "chunk": pl.String,
            "label": pl.Int8,
            "neg_family": pl.String,
            "tag": pl.String,
            "claim_form": pl.String,
            "operand_a": pl.Float64,
            "operand_b": pl.Float64,
            "control": pl.Boolean,
        },
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = pl.read_parquet(SRC)
    sample = build_sample(df, rng)
    controls = build_controls(df, rng)
    out = pl.concat([sample, controls], how="vertical").sort("pair_id", "label")
    out.write_parquet(OUT)

    print(f"wrote {OUT}  rows={out.height}")
    print(out.group_by("control").len().sort("control"))
    print(out.filter(~pl.col("control")).group_by(["neg_family", "label"]).len().sort("len", descending=True))
    print(out.filter(~pl.col("control")).group_by("claim_form").len().sort("len", descending=True))
    print(out.filter(pl.col("control")).group_by(["neg_family", "label"]).len())
    print("--- 5 sampled real claims (bare-form check) ---")
    for r in out.filter(~pl.col("control")).sample(5, seed=7).to_dicts():
        print(f"[{r['label']} {r['neg_family']} {r['claim_form']}] {r['claim']}")
    for r in out.filter(pl.col("control")).head(2).to_dicts():
        print("CTRL", r["label"], "|", r["claim"], "| in-chunk:", r["claim"] in r["chunk"])
    for r in out.filter(pl.col("control")).tail(2).to_dicts():
        print("CTRL", r["label"], "|", r["claim"], "| in-chunk:", r["claim"] in r["chunk"])


if __name__ == "__main__":
    main()

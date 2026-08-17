"""Contract verification support - cache the assembled training mix's surface forms.

CPU ONLY. Rebuilds the mix through the BANKED loader (`R10-H108_lane.public_train`
plus the two `R18-H150_arm_run.LANES`), then writes the derived surfaces the
clause tests need:

  quant_scale_unit_mix_chunks.parquet   one row per DISTINCT mix chunk, in all
                                        three contract string forms (raw,
                                        truncated to CFG.chunk_max_chars = 1500,
                                        whitespace-collapsed case-folded)
  quant_scale_unit_mix_assoc.parquet    (normalised chunk, claim, label, tag) for
                                        the C6 memorisation feature - the mix's
                                        claim associations keyed on evidence

Run:  CUDA_VISIBLE_DEVICES= uv run python \
      experiments/grounding-semantic/contract/quant_scale_unit_mixcache.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import importlib.util
import pathlib
import re

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
CHUNK_MAX = 1500  # M59.CFG.chunk_max_chars

_WS = re.compile(r"\s+")


def norm_ws(t):
    """Contract form 3 - whitespace-collapsed, case-folded."""
    return _WS.sub(" ", t).strip().lower()


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, SEM / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    arm = _mod("g1arm", "R16-H142_G1_arm.py")
    H108 = arm.H108
    run = _mod("h150arm", "R18-H150_arm_run.py")

    with arm.untruncated_evidence():
        claims, chunks, y, tags = H108.public_train()
    print(f"clean mix rows {len(y)} (expect {run.EXPECTED_CLEAN_ROWS})", flush=True)
    assert len(y) == run.EXPECTED_CLEAN_ROWS

    for fname, group, n_rows, n_pairs, fams in run.LANES:
        df = pl.read_parquet(SEM / fname)
        assert len(df) == n_rows, (fname, len(df), n_rows)
        assert df["pair_id"].n_unique() == n_pairs
        claims += df["claim"].to_list()
        chunks += df["chunk"].to_list()
        y = np.concatenate([y, df["label"].cast(pl.Float32).to_numpy()])
        tags += [group] * len(df)
        print(f"lane {group}: {len(df)} rows", flush=True)

    assert len(y) == run.EXPECTED_MIX_ROWS, len(y)
    print(f"mix rows {len(y)}", flush=True)

    mix = pl.DataFrame(
        {"chunk": chunks, "claim": claims, "label": y.astype("int8"), "tag": tags}
    ).with_columns(
        pl.col("chunk").str.slice(0, CHUNK_MAX).alias("chunk_trunc"),
    )
    mix = mix.with_columns(
        pl.col("chunk")
        .map_elements(norm_ws, return_dtype=pl.String)
        .alias("chunk_norm")
    )

    distinct = (
        mix.select("chunk", "chunk_trunc", "chunk_norm")
        .unique(subset=["chunk"])
    )
    distinct.write_parquet(HERE / "quant_scale_unit_mix_chunks.parquet")
    print(f"distinct mix chunks (raw) {len(distinct)}", flush=True)

    mix.select("chunk_norm", "claim", "label", "tag").write_parquet(
        HERE / "quant_scale_unit_mix_assoc.parquet"
    )
    print("wrote mix caches", flush=True)


if __name__ == "__main__":
    main()

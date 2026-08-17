"""R20-H174 window census - the 6-source portfolio mix, CPU only.

Executes the STAGE 1 AMENDMENT of the R20-H174 registration (block "R20-H174
STAGE 0 COMPLETE", 2026-08-16 ~21:00): L2 `attr_pool` pools 4-8 passages per row
and so presents a real multi-passage bag (mean 5.98 windows, 99.8%
multi-window). That moves the combined presentation off the banked H150 geometry
(`R18-H150_window_census.json`: 721,210 rows, mean 1.4821, multi-window 0.1908),
which the flagship wrapper's `census_crosscheck` would read as a hard abort.

This script re-banks the combined census for the portfolio mix so the crosscheck
is REPOINTED at the correct expected geometry rather than weakened. The
registered figures are asserted, not assumed: if the recomputed combined block
disagrees with 760,618 rows / 1.5977 mean windows / 0.2094 multi-window, the
script aborts and nothing is written - that would be a lane-composition
discrepancy, not a constant to reconcile.

Windowing is byte-identical to R8-H101 / R16-H142 (1,500 chars, stride 750,
final window flush to the end) and the clean mix is read UNTRUNCATED the way the
twin trainer reads it, exactly as `R18-H150_window_census.py` does.

Run:  uv run python experiments/grounding-semantic/R20-H174_window_census.py
"""

import contextlib
import importlib.util
import json
import pathlib

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R20-H174_window_census.json"
BANKED_H150 = HERE / "R18-H150_window_census.json"

WIN, STRIDE = 1500, 750

# The five lanes of the portfolio mix, in `R20-H174_arm_run.LANES` order.
LANE_FILES = (
    ("misbind_lane_H146", "R17-H146_lane.parquet"),
    ("scaleunit_lane_H150", "R18-H150_scaleunit_lane.parquet"),
    ("lane_L1_frame_reject", "R20-H174_lane_L1.parquet"),
    ("lane_L2_attr_pool", "R20-H174_lane_L2.parquet"),
    ("lane_L4_path_bind", "R20-H174_lane_L4.parquet"),
)

# The registration's pre-stated geometry for this mix - asserted, not assumed.
REGISTERED = {"rows": 760_618, "mean_windows": 1.5977, "multi_window_share": 0.2094}
TOL = 0.001  # the wrapper's CENSUS_TOL


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H108 = _mod("h108", "R10-H108_lane.py")
M59 = H108.M59


def windows(chunk):
    n = len(chunk)
    if n <= WIN:
        return 1
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return len(starts)


@contextlib.contextmanager
def untruncated_evidence():
    original = M59.CFG.chunk_max_chars
    M59.CFG.chunk_max_chars = 10**9
    try:
        yield
    finally:
        M59.CFG.chunk_max_chars = original


def stats(sizes):
    s = np.asarray(sizes, dtype=np.int64)
    return {"rows": int(s.size), "mean_windows": round(float(s.mean()), 4),
            "median_windows": int(np.median(s)), "max_windows": int(s.max()),
            "total_windows": int(s.sum()),
            "multi_window_rows": int((s > 1).sum()),
            "multi_window_share": round(float((s > 1).mean()), 4)}


def main():
    print("loading the clean public mix UNTRUNCATED...", flush=True)
    with untruncated_evidence():
        _claims, chunks, _y, tags = H108.public_train()
    clean = [windows(c) for c in chunks]
    clean_s = stats(clean)
    print(f"  clean mix: {clean_s}", flush=True)

    per_group = {}
    for g in sorted(set(tags)):
        idx = [i for i, t in enumerate(tags) if t == g]
        per_group[g] = stats([clean[i] for i in idx])

    all_sizes = list(clean)
    lanes = {}
    for name, fname in LANE_FILES:
        path = HERE / fname
        if not path.exists():
            raise SystemExit(f"LANE MISSING: {path} - the mix cannot be censused")
        ch = pl.read_parquet(path, columns=["chunk"])["chunk"].to_list()
        sz = [windows(c) for c in ch]
        lanes[name] = stats(sz)
        all_sizes.extend(sz)
        print(f"  {name}: {lanes[name]}", flush=True)

    # The combined block computed from the RAW per-row window counts - no
    # rounded per-part means enter the arithmetic.
    combined_s = stats(all_sizes)
    combined = {"rows": combined_s["rows"],
                "multi_window_rows": combined_s["multi_window_rows"],
                "multi_window_share": combined_s["multi_window_share"],
                "mean_windows": combined_s["mean_windows"],
                "total_windows": combined_s["total_windows"],
                "max_windows": combined_s["max_windows"]}

    ok = (combined["rows"] == REGISTERED["rows"]
          and abs(combined["mean_windows"] - REGISTERED["mean_windows"]) <= TOL
          and abs(combined["multi_window_share"]
                  - REGISTERED["multi_window_share"]) <= TOL)
    print(f"\ncombined: {combined}", flush=True)
    print(f"registered: {REGISTERED}  -> {'MATCH' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        raise SystemExit(
            "CENSUS MISMATCH ABORT: the assembled portfolio mix does not reproduce "
            "the registered geometry (760,618 / 1.5977 / 0.2094). This is a "
            "lane-composition discrepancy - nothing written, do not train.")

    banked = json.loads(BANKED_H150.read_text())["combined"]
    OUT.write_text(json.dumps({
        "windowing": f"{WIN}/{STRIDE}, final window flush to the end "
                     "(byte-identical to R8-H101 / R16-H142)",
        "purpose": "R20-H174 STAGE 1 AMENDMENT - the re-banked combined census the "
                   "portfolio arm's census_crosscheck reads instead of "
                   "R18-H150_window_census.json; the control is repointed, not weakened",
        "clean_mix": clean_s, "clean_mix_per_group": per_group, **lanes,
        "combined": combined,
        "registered": REGISTERED,
        "matches_registered": True,
        "banked_h150_combined": banked,
        "delta_vs_h150": {
            "rows": combined["rows"] - banked["rows"],
            "mean_windows": round(combined["mean_windows"] - banked["mean_windows"], 4),
            "multi_window_share": round(
                combined["multi_window_share"] - banked["multi_window_share"], 4)},
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    }, indent=2))
    print(f"\nre-banked census -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

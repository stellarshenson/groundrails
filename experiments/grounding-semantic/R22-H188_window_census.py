"""R22-H188 window census - the flagship mix plus `num_derive`, CPU only.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R22-H188 DERIVATION-ENHANCED MIX" (2026-08-17 ~17:32). The arm adds the
R22-H187 `num_derive` lane (30,000 rows / 15,000 pairs, `conforming: true`) to
the R18-H150 flagship mix as one additional DANN group and changes nothing else.

The banked `census_crosscheck` in `R18-H150_arm_run.py` compares the assembled
presentation against `R18-H150_window_census.json` (721,210 rows, mean 1.4821
windows, multi-window 0.1908) and hard-aborts on any drift. The derivation mix
presents 30,000 more rows, so that crosscheck would abort this arm by design.
The control is REPOINTED, not weakened - exactly the R20-H174 stage-1 pattern:
this script recomputes the combined census from the actual built mix, asserts it
against the geometry derived below, banks it as `R22-H188_window_census.json`,
and the arm wrapper repoints `census_crosscheck` at that file. A mix that drifts
from this geometry still aborts before a card is touched.

THE EXPECTED GEOMETRY IS DERIVED, NOT OBSERVED. Every `num_derive` chunk is at
most 1,500 characters, so every lane row is exactly one window under the
1,500/750 presentation. From the banked clean-mix census (685,670 rows /
1,033,365 windows / 137,622 multi-window rows) plus the three single-window
lanes:

    rows               685,670 + 30,000 + 5,540 + 30,000 = 751,210
    total windows    1,033,365 + 30,000 + 5,540 + 30,000 = 1,098,905
    mean windows       1,098,905 / 751,210               = 1.4628
    multi-window       137,622 / 751,210                 = 0.1832

Windowing is byte-identical to R8-H101 / R16-H142 (1,500 chars, stride 750,
final window flush to the end) and the clean mix is read UNTRUNCATED the way the
twin trainer reads it, exactly as `R18-H150_window_census.py` does.

Run:  uv run python experiments/grounding-semantic/R22-H188_window_census.py
"""

import contextlib
import importlib.util
import json
import pathlib

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R22-H188_window_census.json"
BANKED_H150 = HERE / "R18-H150_window_census.json"

WIN, STRIDE = 1500, 750

# The three lanes of the derivation mix, in `R22-H188_arm_run.LANES` order.
LANE_FILES = (
    ("misbind_lane_H146", "R17-H146_lane.parquet"),
    ("scaleunit_lane_H150", "R18-H150_scaleunit_lane.parquet"),
    ("num_derive_lane_H187", "R22-H187_num_derive_lane.parquet"),
)

# The derived geometry above - asserted, not assumed.
EXPECTED = {"rows": 751_210, "mean_windows": 1.4628, "multi_window_share": 0.1832}
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

    ok = (combined["rows"] == EXPECTED["rows"]
          and abs(combined["mean_windows"] - EXPECTED["mean_windows"]) <= TOL
          and abs(combined["multi_window_share"]
                  - EXPECTED["multi_window_share"]) <= TOL)
    print(f"\ncombined: {combined}", flush=True)
    print(f"expected: {EXPECTED}  -> {'MATCH' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        raise SystemExit(
            "CENSUS MISMATCH ABORT: the assembled derivation mix does not reproduce "
            "the derived geometry (751,210 / 1.4628 / 0.1832). This is a "
            "lane-composition discrepancy - nothing written, do not train.")

    banked = json.loads(BANKED_H150.read_text())["combined"]
    OUT.write_text(json.dumps({
        "windowing": f"{WIN}/{STRIDE}, final window flush to the end "
                     "(byte-identical to R8-H101 / R16-H142)",
        "purpose": "R22-H188 - the re-banked combined census the derivation arm's "
                   "census_crosscheck reads instead of R18-H150_window_census.json; "
                   "the control is repointed, not weakened",
        "clean_mix": clean_s, "clean_mix_per_group": per_group, **lanes,
        "combined": combined,
        "expected": EXPECTED,
        "expected_derivation": ("clean mix 685,670 rows / 1,033,365 windows / 137,622 "
                                "multi-window (R20-H174_window_census.json) plus three "
                                "all-single-window lanes 30,000 + 5,540 + 30,000"),
        "matches_expected": True,
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

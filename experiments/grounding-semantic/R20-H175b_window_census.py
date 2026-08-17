"""R20-H175b window census - the 4-source question-conditioned mix, CPU only.

Executes the census rebind the R20-H175b stage-1 launch requires, on the
R20-H174 precedent (block "R20-H174 STAGE 1 LAUNCHED", 2026-08-17 01:00:16).
Adding the 17,972-row `qrel_contrast` contrast lane moves the mix off the banked
H150 geometry (`R18-H150_window_census.json`: 721,210 rows, mean 1.4821,
multi-window 0.1908), which `R18-H150_arm_run.census_crosscheck` reads as a hard
abort. This script recomputes the combined census from the actual built mix and
banks it as `R20-H175b_window_census.json`, which the arm's crosscheck then
reads. THE CONTROL IS REPOINTED, NEVER WEAKENED - it still hard-aborts on drift.

The combined figures are NOT known in advance, so they cannot be asserted
directly. Every COMPONENT is asserted instead, which pins the combined block
just as tightly:

    clean mix + H146 misbind + H150 unit_swap  ==  the banked H150 combined block
                                                   (721,210 / 1.4821 / 0.1908)
    R20-H175b_qlane                            ==  the lane manifest's own census
                                                   (17,972 / 1.1108 / 0.0739)
    combined rows                              ==  739,182

A mismatch on any of them aborts and nothing is written - that would be a
composition discrepancy, not a constant to reconcile.

The question channel does NOT enter here: windows are computed from the evidence
side, which the arm leaves untouched. Windowing is byte-identical to R8-H101 /
R16-H142 (1,500 chars, stride 750, final window flush to the end) and the clean
mix is read UNTRUNCATED the way the twin trainer reads it.

Run:  uv run python experiments/grounding-semantic/R20-H175b_window_census.py
"""

import contextlib
import importlib.util
import json
import pathlib

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R20-H175b_window_census.json"
BANKED_H150 = HERE / "R18-H150_window_census.json"
LANE_MANIFEST = HERE / "R20-H175b_qlane_manifest.json"

WIN, STRIDE = 1500, 750
TOL = 0.001  # the wrapper's CENSUS_TOL

# The flagship's two lanes, then the arm's one addition - `R20-H175b_arm_run.LANES` order.
FLAGSHIP_LANES = (
    ("misbind_lane_H146", "R17-H146_lane.parquet"),
    ("scaleunit_lane_H150", "R18-H150_scaleunit_lane.parquet"),
)
QLANE = ("qlane_H175b_qrel_contrast", "R20-H175b_qlane.parquet")

EXPECTED_COMBINED_ROWS = 739_182  # 721,210 flagship + 17,972 contrast lane


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


def block(s):
    return {"rows": s["rows"], "multi_window_rows": s["multi_window_rows"],
            "multi_window_share": s["multi_window_share"],
            "mean_windows": s["mean_windows"], "total_windows": s["total_windows"],
            "max_windows": s["max_windows"]}


def matches(got, want, keys=("rows", "mean_windows", "multi_window_share")):
    for k in keys:
        a, b = got[k], want[k]
        if isinstance(a, int) and isinstance(b, int):
            if a != b:
                return False
        elif abs(float(a) - float(b)) > TOL:
            return False
    return True


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

    flagship_sizes = list(clean)
    lanes = {}
    for name, fname in FLAGSHIP_LANES:
        path = HERE / fname
        if not path.exists():
            raise SystemExit(f"LANE MISSING: {path} - the mix cannot be censused")
        sz = [windows(c) for c in pl.read_parquet(path, columns=["chunk"])["chunk"].to_list()]
        lanes[name] = stats(sz)
        flagship_sizes.extend(sz)
        print(f"  {name}: {lanes[name]}", flush=True)

    # --- assertion 1: the flagship sub-mix reproduces the banked H150 census --- #
    flagship_block = block(stats(flagship_sizes))
    banked = json.loads(BANKED_H150.read_text())["combined"]
    print(f"\nflagship sub-mix: {flagship_block}", flush=True)
    print(f"banked H150     : {banked}", flush=True)
    if not matches(flagship_block, banked):
        raise SystemExit(
            "CENSUS ABORT: the flagship sub-mix (clean + H146 + unit_swap) does not "
            "reproduce the banked R18-H150 combined census. The arm's base is not "
            "the flagship's; nothing written, do not train.")
    print("flagship sub-mix MATCHES the banked H150 combined census", flush=True)

    # --- assertion 2: the contrast lane reproduces its own manifest census ----- #
    name, fname = QLANE
    path = HERE / fname
    if not path.exists():
        raise SystemExit(f"LANE MISSING: {path} - the arm has no intervention lane")
    q_sizes = [windows(c) for c in pl.read_parquet(path, columns=["chunk"])["chunk"].to_list()]
    lanes[name] = stats(q_sizes)
    manifest_cens = json.loads(LANE_MANIFEST.read_text())["window_census"]
    print(f"\n  {name}: {lanes[name]}", flush=True)
    print(f"  lane manifest: {manifest_cens}", flush=True)
    if not matches(lanes[name], manifest_cens):
        raise SystemExit(
            "CENSUS ABORT: R20-H175b_qlane.parquet does not reproduce the window "
            "census banked in its own manifest - the lane on disk is not the "
            "stage-0 artifact; nothing written, do not train.")

    # --- assertion 3: the combined row count ---------------------------------- #
    all_sizes = flagship_sizes + q_sizes
    combined = block(stats(all_sizes))
    print(f"\ncombined: {combined}", flush=True)
    if combined["rows"] != EXPECTED_COMBINED_ROWS:
        raise SystemExit(
            f"CENSUS ABORT: combined mix {combined['rows']} rows, expected "
            f"{EXPECTED_COMBINED_ROWS} (721,210 flagship + 17,972 contrast lane)")

    OUT.write_text(json.dumps({
        "windowing": f"{WIN}/{STRIDE}, final window flush to the end "
                     "(byte-identical to R8-H101 / R16-H142)",
        "purpose": "R20-H175b STAGE 1 census rebind - the re-banked combined census "
                   "the arm's census_crosscheck reads instead of "
                   "R18-H150_window_census.json; the control is repointed, not "
                   "weakened, and every component was asserted before this was written",
        "clean_mix": clean_s, "clean_mix_per_group": per_group, **lanes,
        "flagship_sub_mix": flagship_block,
        "combined": combined,
        "asserted_before_write": {
            "flagship_sub_mix_equals_banked_h150_combined": True,
            "qlane_equals_its_manifest_window_census": True,
            "combined_rows_equals_739182": True,
        },
        "banked_h150_combined": banked,
        "delta_vs_h150": {
            "rows": combined["rows"] - banked["rows"],
            "mean_windows": round(combined["mean_windows"] - banked["mean_windows"], 4),
            "multi_window_share": round(
                combined["multi_window_share"] - banked["multi_window_share"], 4)},
        "note": "The question channel does not enter this census - the arm leaves "
                "the evidence side and the windowing untouched. Numbers recorded, "
                "not adjudicated - the coordinator adjudicates.",
    }, indent=2))
    print(f"\nre-banked census -> {OUT}", flush=True)
    print("=== R20-H175b WINDOW CENSUS BANKED ===", flush=True)


if __name__ == "__main__":
    main()

"""R18-H150 window census - combined-mix serving-window statistics, CPU only.

The registration requires the window census to be re-run at lane build time:
lane rows are short single-window chunks and dilute the mix's multi-window
share.  Windowing is byte-identical to R8-H101 / R16-H142 (1,500 chars, stride
750, final window flush to the end), and the clean mix is read UNTRUNCATED the
way the twin trainer reads it - `R10-H108_lane.public_train()` with
`M59.CFG.chunk_max_chars` lifted - because the banked 20.1% baseline is a
property of the untruncated evidence, not of the 1,500-char cached mix.

Run:  uv run python experiments/grounding-semantic/R18-H150_window_census.py
"""

import contextlib
import importlib.util
import json
import pathlib

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R18-H150_window_census.json"
H146_LANE = HERE / "R17-H146_lane.parquet"
H150_LANE = HERE / "R18-H150_scaleunit_lane.parquet"

WIN, STRIDE = 1500, 750
BASELINE_MULTIWINDOW = 0.201
BASELINE_MEAN_WINDOWS = 1.507


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
    s = np.asarray(sizes)
    return {"rows": int(s.size), "mean_windows": round(float(s.mean()), 4),
            "median_windows": int(np.median(s)), "max_windows": int(s.max()),
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

    lanes = {}
    for name, path in (("misbind_lane_H146", H146_LANE),
                       ("scaleunit_lane_H150", H150_LANE)):
        if not path.exists():
            lanes[name] = {"present": False}
            continue
        ch = pl.read_parquet(path, columns=["chunk"])["chunk"].to_list()
        lanes[name] = stats([windows(c) for c in ch])
        print(f"  {name}: {lanes[name]}", flush=True)

    parts = [clean_s] + [v for v in lanes.values() if v.get("rows")]
    rows = sum(p["rows"] for p in parts)
    multi = sum(p["multi_window_rows"] for p in parts)
    mean_w = sum(p["mean_windows"] * p["rows"] for p in parts) / rows
    out = {
        "windowing": f"{WIN}/{STRIDE}, final window flush to the end "
                     "(byte-identical to R8-H101 / R16-H142)",
        "clean_mix": clean_s, "clean_mix_per_group": per_group, **lanes,
        "combined": {"rows": rows, "multi_window_rows": multi,
                     "multi_window_share": round(multi / rows, 4),
                     "mean_windows": round(mean_w, 4)},
        "baseline": {"multi_window_share": BASELINE_MULTIWINDOW,
                     "mean_windows": BASELINE_MEAN_WINDOWS,
                     "source": "R16-H142 G1 twin arm, clean mix alone"},
        "delta_vs_baseline": {
            "multi_window_share": round(multi / rows - BASELINE_MULTIWINDOW, 4),
            "mean_windows": round(mean_w - BASELINE_MEAN_WINDOWS, 4)},
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("clean_mix", "misbind_lane_H146", "scaleunit_lane_H150",
                       "combined", "baseline", "delta_vs_baseline")}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()

"""Fold a lane draw's two blind-arena reads into its per-draw result JSON.

The reads are produced by the frozen gates themselves - `R8_decomposed_read.py`
(truncated decomposed-min, appended to R8_decomposed_reads.json under a tag) and
`R8-H101_windowed_read.py` (the PRIMARY windowed read, its own JSON). This tool
only copies their per-subset scores and means into
`<lane>_lane_draw<N>_result.json` so one file carries the whole draw, and prints
the draw's PRIMARY mean against the clean 2-draw admission bar.

Run:  uv run python experiments/grounding-semantic/R10_lane_collect.py \
          --lane R10-H107 --draw 1
"""

import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).parent
BOOK = HERE / "R8_decomposed_reads.json"
CLEAN_MEAN = 0.7031  # R9-H105 clean 2-draw mean under the PRIMARY windowed read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True)
    ap.add_argument("--draw", type=int, required=True)
    args = ap.parse_args()

    res_path = HERE / f"{args.lane}_lane_draw{args.draw}_result.json"
    win_path = HERE / f"{args.lane}_lane_draw{args.draw}_windowed_result.json"
    tag = f"{args.lane}-lane-draw{args.draw}"

    res = json.loads(res_path.read_text())
    win = json.loads(win_path.read_text())
    trunc = json.loads(BOOK.read_text())[tag]

    res["reads"] = {
        "windowed_primary": {
            "read": "windowed decomposed-min (1500/750), R8-H101 gate",
            "per_subset": {k: v["auc"] for k, v in win["per_subset"].items()},
            "mean": win["mean"],
            "mean_lettuce": win["mean_lettuce"],
        },
        "truncated": {
            "read": "truncated decomposed-min, R8_decomposed_read gate",
            "per_subset": {k: v["auc"] for k, v in trunc["per_subset"].items()},
            "mean": trunc["mean"],
            "wins": trunc["wins"],
        },
    }
    res["primary_mean"] = win["mean"]
    res["clean_mean_bar"] = CLEAN_MEAN
    res_path.write_text(json.dumps(res, indent=2))

    print(f"{tag}: PRIMARY windowed mean {win['mean']:.4f}  (clean bar {CLEAN_MEAN:.4f}, "
          f"delta {win['mean'] - CLEAN_MEAN:+.4f})   truncated mean {trunc['mean']:.4f}")
    print(f"  collected -> {res_path}")


if __name__ == "__main__":
    main()

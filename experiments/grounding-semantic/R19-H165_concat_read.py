"""R19-H165 CONCATENATION - the ONE blind arena read the ladder licensed.

The ladder selected cell C0 on `gold_full`: joining the evidence pool BEFORE
windowing, at the banked MAX_LEN of 512, read 0.9014 against the banked
presentation's 0.8659 (+0.0355), with the L0 positive control reproducing the
banked number to 1e-05. Length beyond 512 hurt monotonically, so the lever is
concatenation alone.

`gold_full` is the SELECTION surface. The arena is the verdict. This script
spends the single blind read that selection licenses, on BOTH banked flagship
draws, with the bar fixed before it runs.

WHAT CHANGES, AND IT IS ONE FUNCTION
------------------------------------
`R16-H142_G1_reads.evidence_sets` is the whole diff:

    banked      [w for k in chunk_list for w in ARM.windows(k)]   # per chunk
    this read   ARM.windows("\n\n".join(chunk_list))              # pooled first

Windowing stays 1,500/750, MAX_LEN stays 512, the model is untouched, the
decomposed-min response read is untouched, and nothing trains. The reader is the
banked `R16-H142_G1_reads.py` reused unchanged apart from that binding and the
output path - the R18-H150 wrapper pattern.

BAR, fixed before the read, mirroring the R8-H101 supersession
--------------------------------------------------------------
The PRIMARY serving formula is replaced only if, on BOTH banked draws:
  - the blind windowed mean rises by >= 0.005, AND
  - no subset falls by more than 0.01
Anything else leaves the banked per-chunk presentation as PRIMARY. There is no
second read: re-reading a variant after a miss would be tuning on the arena,
which the H141 discipline bars.

Run: CUDA_VISIBLE_DEVICES=<gpu> uv run python R19-H165_concat_read.py --draw 1
"""

import argparse
import importlib.util
import json
import os
import pathlib
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = pathlib.Path(__file__).parent
SEP = "\n\n"

CKPT = {1: "R18-H150-arm-draw1", 2: "R18-H150-arm-draw2"}
# The comparator is the SAME checkpoint read under the banked per-chunk
# presentation, loaded from its own banked result file. It is deliberately NOT
# `reads.CONTROL_WINDOWED`, which belongs to the R16-H142 G1 twin (mean 0.70311)
# - a different checkpoint, and comparing against it would measure the wrong
# difference entirely.
BANKED_READ = {1: "R18-H150_arm_draw1_windowed_result.json",
               2: "R18-H150_arm_draw2_windowed_result.json"}
BARS = {"mean_gain": 0.005, "max_subset_drop": 0.01}


def banked_baseline(draw):
    """The draw's own per-chunk read - mean and per-subset, straight from disk."""
    d = json.loads((HERE / BANKED_READ[draw]).read_text())
    return d["mean"], {s: v["auc"] for s, v in d["per_subset"].items()}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", type=int, required=True, choices=(1, 2))
    args = ap.parse_args()

    out = HERE / f"R19-H165_concat_arena_draw{args.draw}_result.json"
    if out.exists() and out.stat().st_size > 0:
        print(f"SKIP draw {args.draw} (on disk: {out.name})", flush=True)
        print("=== H165 CONCAT READ COMPLETE ===", flush=True)
        return

    reads = _mod("g1reads", "R16-H142_G1_reads.py")

    # --- the single change ----------------------------------------------------
    def concat_evidence_sets(mode, chunk_list):
        if mode != "windowed":
            raise SystemExit("H165 concat read is defined for the windowed mode only")
        return reads.ARM.windows(SEP.join(chunk_list))

    reads.evidence_sets = concat_evidence_sets
    reads.ARM.RUNS["twin"]["ckpt"] = CKPT[args.draw]
    reads.out_path = lambda run, mode: out

    print(f"=== R19-H165 CONCAT blind arena read, draw {args.draw} "
          f"({CKPT[args.draw]})  {time.strftime('%F %T')} ===", flush=True)
    print(f"  presentation: pool concatenated with {SEP!r} then windowed "
          f"{reads.ARM.WIN}/{reads.ARM.STRIDE} at MAX_LEN {reads.ARM.MAX_LEN}",
          flush=True)
    base_mean, base_per = banked_baseline(args.draw)
    print(f"  comparator: banked per-chunk draw {args.draw} = {base_mean:.5f} "
          f"(from {BANKED_READ[args.draw]})", flush=True)

    import sys
    sys.argv = ["reads", "--run", "twin", "--mode", "windowed"]
    reads.main()

    # --- adjudicate against the pre-registered bar ----------------------------
    res = json.loads(out.read_text())
    per = res["per_subset"]
    deltas = {s: round(per[s]["auc"] - base_per[s], 5) for s in per}
    worst = min(deltas.items(), key=lambda kv: kv[1])
    gain = round(res["mean"] - base_mean, 5)
    res["h165_concat"] = {
        "draw": args.draw,
        "presentation": {"pool_concatenated": True, "separator": repr(SEP),
                         "WIN": reads.ARM.WIN, "STRIDE": reads.ARM.STRIDE,
                         "MAX_LEN": reads.ARM.MAX_LEN},
        "banked_per_chunk_mean": base_mean,
        "banked_per_chunk_per_subset": base_per,
        "comparator_source": BANKED_READ[args.draw],
        "mean_gain": gain,
        "per_subset_delta_vs_banked": deltas,
        "worst_subset": {"subset": worst[0], "delta": worst[1]},
        "bars": BARS,
        "mean_leg": bool(gain >= BARS["mean_gain"]),
        "subset_leg": bool(worst[1] >= -BARS["max_subset_drop"]),
        "draw_passes": bool(gain >= BARS["mean_gain"]
                            and worst[1] >= -BARS["max_subset_drop"]),
        "note": ("gold_full selected this presentation; this is the ONE blind read "
                 "that selection licenses. Supersession needs BOTH draws to pass. "
                 "No re-read on a variant - that would be tuning on the arena."),
    }
    out.write_text(json.dumps(res, indent=2))

    print(f"\n  draw {args.draw} concat mean {res['mean']:.5f} vs banked "
          f"{base_mean:.5f}  ({gain:+.5f})", flush=True)
    print(f"  worst subset {worst[0]} {worst[1]:+.5f} "
          f"(bar {-BARS['max_subset_drop']:+.3f})", flush=True)
    print(f"  draw {args.draw} passes: {res['h165_concat']['draw_passes']}", flush=True)
    print("=== H165 CONCAT READ COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()

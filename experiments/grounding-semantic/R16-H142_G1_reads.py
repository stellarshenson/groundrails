"""R16-H142 G1 (A1) - the blind arena reads for the ablation pair.

The campaign's generic read tools (`R8_decomposed_read.py`,
`R8-H101_windowed_read.py`) score a checkpoint through trunk + task head only,
so they cannot read the adapter arm: its logit contribution would be silently
dropped. This script applies the SAME two protocols through the pair's own
forward path, and scores the twin through the identical code - the twin's
adapter is frozen at its zero init, so its contribution is exactly 0.0 and the
twin reads as a plain score head with no branch in the reader.

    truncated  each sentence against the subset's chunk list, chunks cut to
               CFG.chunk_max_chars - the R8-H92 decomposed-min read
    windowed   each sentence against every 1,500-char window (stride 750) of
               every FULL chunk - the R8-H101 PRIMARY read

Aggregation is unchanged in both: max over the set, then MIN over the response's
sentences. The window-ensemble context is mean-pooled over the whole set before
the adapter runs, exactly as in R16-H142 G0.

RAGBench is EVALUATION ONLY and never enters training.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R16-H142_G1_reads.py \
          --run arm --mode windowed
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import argparse
import importlib.util
import json
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).parent


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ARM = _mod("g1arm", "R16-H142_G1_arm.py")
H92 = _mod("h92", "R8-H92_decomposed_arena.py")
ARENA = H92.ARENA
M59 = ARENA.M59

# The banked clean control PAIR under the PRIMARY windowed read - per-subset
# means of R9-H105_windowed_result.json (draw 1) and
# R9-H105_draw2_windowed_result.json. Carried for context only: under A1 the
# comparator for the adapter question is the twin, not this pair.
CONTROL_WINDOWED = {
    "covidqa": 0.7878, "delucionqa": 0.8167, "emanual": 0.6977, "expertqa": 0.7728,
    "finqa": 0.6333, "hagrid": 0.6340, "hotpotqa": 0.6668, "pubmedqa": 0.6063,
    "tatqa": 0.7320, "techqa": 0.6840,
}
CONTROL_WINDOWED_MEAN = 0.70311


def out_path(run, mode):
    return HERE / f"R16-H142_G1_{run}_{mode}_result.json"


def evidence_sets(mode, chunk_list):
    if mode == "truncated":
        return [k[: M59.CFG.chunk_max_chars] for k in chunk_list]
    return [w for k in chunk_list for w in ARM.windows(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=tuple(ARM.RUNS))
    ap.add_argument("--mode", required=True, choices=("truncated", "windowed"))
    args = ap.parse_args()
    ckpt = ARM.ROOT / "models" / ARM.RUNS[args.run]["ckpt"]
    out = out_path(args.run, args.mode)

    print(f"=== R16-H142 G1 {args.run} {args.mode} arena read  "
          f"{time.strftime('%F %T')} ===", flush=True)
    model, tok = ARM.load_run(ckpt)
    subs = ARENA.load_subsets()

    rows = {}
    for sub, (claims, chunks, y) in subs.items():
        flat_s, flat_w, set_index, owner = [], [], [], []
        for i, (c, ks) in enumerate(zip(claims, chunks, strict=True)):
            wlist = evidence_sets(args.mode, ks)
            for s in H92.sentences(c):
                sid = len(owner)
                owner.append(i)
                for w in wlist:
                    flat_s.append(s)
                    flat_w.append(w)
                    set_index.append(sid)
        owner = np.array(owner)
        s_sent = ARM.score_sets(model, tok, flat_s, flat_w, set_index, len(owner),
                                tag=f"{args.run}/{args.mode}/{sub}")
        resp = np.array([s_sent[owner == i].min() for i in range(len(y))])
        auc, f1, _ = M59.auc_and_f1(y, resp)
        rows[sub] = {
            "n": len(y), "n_sent": len(owner), "n_pairs": len(flat_s),
            "auc": round(auc, 4), "f1": round(f1, 4),
            "lettuce_auc": H92.LETTUCE[sub],
        }
        if args.mode == "windowed":
            rows[sub]["banked_control_windowed"] = CONTROL_WINDOWED[sub]
        print(f"  {sub:14s} n={len(y):>4} {args.run} {args.mode} {auc:.4f}", flush=True)

    mean = float(np.mean([r["auc"] for r in rows.values()]))
    payload = {
        "read": f"R16-H142 G1 (A1) {args.run}, {args.mode} decomposed-min read",
        "run": args.run, "checkpoint": str(ckpt), "per_subset": rows,
        "mean": round(mean, 5),
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    }
    if args.mode == "windowed":
        payload.update({
            "banked_control_windowed_mean": CONTROL_WINDOWED_MEAN,
            "mean_delta_vs_banked_control": round(mean - CONTROL_WINDOWED_MEAN, 5),
        })

    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  {args.run} {args.mode} mean {mean:.5f}", flush=True)
    if args.mode == "windowed":
        print(f"  pubmedqa {rows['pubmedqa']['auc']:.4f}  "
              f"hotpotqa {rows['hotpotqa']['auc']:.4f}  "
              f"tatqa {rows['tatqa']['auc']:.4f}", flush=True)
    print(f"  results -> {out}", flush=True)


if __name__ == "__main__":
    main()

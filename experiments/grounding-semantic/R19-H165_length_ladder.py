"""R19-H165 CONTEXT-LENGTH LADDER - one cell per invocation, gold held-out only.

SELECTION SURFACE ONLY. This script never touches the arena. It reads gold_full,
the in-domain held-out set, which is the legal surface for choosing a serving
presentation; the arena gets exactly ONE blind read of whatever this ladder
selects, adjudicated separately.

WHY
---
The trunk is ModernBERT with `max_position_embeddings: 8192` and the campaign
serves it at MAX_LEN 512 over 1,500-char windows - a 16x under-use inherited from
R8-H101 and never revisited after the trunk gained long context. Measured on the
banked arena dump, at 4,096 tokens 82.4% of arena items would fit their ENTIRE
evidence pool in one window, and every hotpotqa item would (largest pool 10,689
chars).

THE TWO VARIABLES, AND WHY BOTH ARE NEEDED
------------------------------------------
`R16-H142_G1_arm.windows()` is applied PER CHUNK, so raising the window width
alone never merges two documents - a 2,000-char document stays its own window at
any width. Making one window carry two documents needs the evidence pool
CONCATENATED before windowing. Length and concatenation are therefore separate
levers and the ladder carries a control that moves concatenation alone:

  cell  presentation        WIN     STRIDE  MAX_LEN   isolates
  L0    per-chunk (banked)  1500    750     512       positive control
  C0    pool-concatenated   1500    750     512       concatenation ALONE
  L1    pool-concatenated   3600    1800    1024      + length
  L2    pool-concatenated   7200    3600    2048      + length
  L3    pool-concatenated   14400   7200    4096      + length
  L4    pool-concatenated   28800   14400   8192      + length

WIN is held at ~3.6 chars per token so the window fills the cap without the
tokenizer truncating it; STRIDE stays at WIN/2, the banked 2:1 ratio.

POSITIVE CONTROL, binding
-------------------------
L0 must reproduce the checkpoint's banked gold_full AUROC to <= 1e-3. It rebuilds
the evidence sets through this script's own code path rather than through
`score_claims`, so a match proves the reimplementation is faithful and every
other cell sits on the same substrate. A miss VOIDS the ladder rather than being
adjusted away.

TRAIN/SERVE MISMATCH is measured, not assumed - these checkpoints were fine-tuned
at MAX_LEN 512. The trunk was pretrained at 8192 with RoPE so positions
extrapolate natively, but the task head has only ever seen 512-token contexts. A
monotone decline across the ladder is that mismatch; the registered follow-on is
retraining at length, not abandoning the lever.

Run:  uv run python experiments/grounding-semantic/R19-H165_length_ladder.py \
        --cell L3 --ckpt R18-H150-arm-draw1
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent

# cell -> (concatenate_pool, WIN, STRIDE, MAX_LEN, encode_batch)
# Batch scales inversely with MAX_LEN off the campaign's fitted memory law
# peak_alloc(GB) ~= 3.69 + 0.307 * n_sequences at 512 tokens.
CELLS = {
    "L0": (False, 1_500, 750, 512, 64),
    "C0": (True, 1_500, 750, 512, 64),
    "L1": (True, 3_600, 1_800, 1_024, 32),
    "L2": (True, 7_200, 3_600, 2_048, 16),
    "L3": (True, 14_400, 7_200, 4_096, 8),
    "L4": (True, 28_800, 14_400, 8_192, 4),
}

# Banked gold_full AUROC per checkpoint - the positive control for cell L0.
BANKED_GOLDFULL = {
    "R18-H150-arm-draw1": 0.8659,
    "R18-H150-arm-draw2": 0.8644,
}
CONTROL_TOL = 1e-3

SEP = "\n\n"


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def evidence_sets(chunk_list, concat, arm):
    """The cell's evidence presentation for one item.

    concat=False - the banked path: every chunk truncated to the serving unit,
    one window each, documents never merged.
    concat=True  - the pool is joined with a blank line and then windowed, so a
    window can span a document boundary when it is wide enough to.
    """
    if not concat:
        return [k[: arm.WIN] for k in chunk_list]
    return arm.windows(SEP.join(chunk_list))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=tuple(CELLS))
    ap.add_argument("--ckpt", default="R18-H150-arm-draw1")
    args = ap.parse_args()

    concat, win, stride, max_len, batch = CELLS[args.cell]
    out = HERE / f"R19-H165_ladder_{args.cell}_{args.ckpt}.json"
    if out.exists() and out.stat().st_size > 0:
        print(f"SKIP {args.cell} (on disk: {out.name})", flush=True)
        print("=== H165 CELL COMPLETE ===", flush=True)
        return

    print(f"=== R19-H165 ladder cell {args.cell} on {args.ckpt}  "
          f"{time.strftime('%F %T')} ===", flush=True)
    print(f"  concat={concat} WIN={win} STRIDE={stride} MAX_LEN={max_len} "
          f"batch={batch}", flush=True)

    arm = _mod("g1arm", "R16-H142_G1_arm.py")
    # The presentation constants are module globals resolved at call time, so
    # patching them here rebinds windows()/score_sets() for this process only.
    arm.WIN, arm.STRIDE, arm.MAX_LEN = win, stride, max_len

    H108 = arm.H108
    M59 = arm.M59

    claims, chunk_lists, y = H108.gold_full()
    y = np.asarray(y)
    print(f"  gold_full: {len(claims)} items, {y.mean():.3f} positive", flush=True)

    flat_s, flat_w, set_index = [], [], []
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        for w in evidence_sets(ks, concat, arm):
            flat_s.append(c)
            flat_w.append(w)
            set_index.append(i)
    n_pairs = len(flat_s)
    wins_per_item = n_pairs / len(claims)
    mean_win_chars = float(np.mean([len(w) for w in flat_w]))
    print(f"  pairs {n_pairs} ({wins_per_item:.2f} windows/item, "
          f"mean window {mean_win_chars:.0f} chars)", flush=True)

    model, tok = arm.load_run(ROOT / "models" / args.ckpt)

    t0 = time.time()
    scores = arm.score_sets(model, tok, flat_s, flat_w, set_index, len(claims),
                            batch=batch, tag=f"{args.cell}/gold_full")
    elapsed = time.time() - t0

    auc, f1, _ = M59.auc_and_f1(y, scores)
    peak = torch.cuda.max_memory_allocated() / 1e9

    control = None
    if args.cell == "L0":
        banked = BANKED_GOLDFULL.get(args.ckpt)
        if banked is not None:
            delta = abs(auc - banked)
            control = {"banked": banked, "reproduced": round(auc, 4),
                       "abs_delta": round(delta, 5), "tol": CONTROL_TOL,
                       "pass": bool(delta <= CONTROL_TOL)}
            print(f"  POSITIVE CONTROL banked {banked:.4f} vs reproduced "
                  f"{auc:.4f} -> {'PASS' if control['pass'] else 'FAIL'}",
                  flush=True)

    res = {
        "arm": "R19-H165 context-length ladder",
        "surface": "gold_full in-domain held-out - SELECTION SURFACE, never the arena",
        "status": "NOT ADJUDICATED HERE - the coordinator holds the verdict",
        "cell": args.cell,
        "checkpoint": args.ckpt,
        "presentation": {"pool_concatenated": concat, "WIN": win,
                         "STRIDE": stride, "MAX_LEN": max_len,
                         "encode_batch": batch, "separator": repr(SEP)},
        "gold_full": {"auc": round(float(auc), 4), "f1": round(float(f1), 4),
                      "n": int(len(claims)), "positive_rate": round(float(y.mean()), 4)},
        "cost": {"n_pairs": n_pairs, "windows_per_item": round(wins_per_item, 3),
                 "mean_window_chars": round(mean_win_chars, 1),
                 "seconds": round(elapsed, 1),
                 "pairs_per_second": round(n_pairs / max(elapsed, 1e-9), 1),
                 "peak_alloc_gb": round(peak, 2)},
        "positive_control": control,
    }
    out.write_text(json.dumps(res, indent=1))
    print(f"  gold_full AUROC {auc:.4f}  f1 {f1:.4f}  "
          f"{elapsed:.0f}s  peak {peak:.1f} GB", flush=True)
    print(f"  -> {out.name}", flush=True)
    print("=== H165 CELL COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()

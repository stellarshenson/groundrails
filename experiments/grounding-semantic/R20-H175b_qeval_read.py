"""R20-H175b PRIMARY mechanism read - question-relevance AUROC on the held-out
contrast eval. Inference only, nothing trains.

Registered in docs/experiments/semantic-grounding-experiments.md: the arm's
PRIMARY is held-out question-relevance AUROC on `R20-H175b_qlane_eval.parquet`
(1,001 pairs / 487 documents, 0 documents shared with the training lane)
**>= 0.80**, read against the HIGHER of two banked floors -

    0.5000   the question-blind flagship, measured exactly on this eval with
             1,001 of 1,001 pairs bit-identical ("BASELINE LEGS BANKED",
             2026-08-17 ~05:45)
    0.5816   the banked surface-probe floor (stage-0 disposition 3)

Neither is recomputed here. The baseline is BANKED; this script reads the
TRAINED checkpoint and reports its number against those two figures.

Read protocol - byte-identical to `R20_baseline_legs.py` LEG 2 and to
`R20-H176_findver_read.py`: evidence UNTRUNCATED, presented as 1,500-char
windows at stride 750, claim scored against every window, MAX over windows.
Every row is claim-level, so the arena read's MIN-over-response-sentences stage
does not apply. Frozen trunk + task head via `R15_gate_common.load_ckpt`/`.score`.

The ONE difference from the baseline leg, and it is the arm's intervention: the
question is composed into the claim side through `R20-H175b_qchannel.compose`.

TWO CONTROLS, both registered here before the read:

    question-blind   the same checkpoint on the same eval with NO question
                     composed. Both legs of a pair are then byte-identical
                     inputs, so this MUST read exactly 0.5000 with 1,001/1,001
                     pairs bit-identical. A departure means the read path leaks
                     something other than the question, and the PRIMARY number
                     is not attributable to the channel
    flagship-recompute  none. The 0.5000 floor is banked and is NOT re-read

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 \
      uv run python experiments/grounding-semantic/R20-H175b_qeval_read.py --draw 1
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import importlib.util
import json
import pathlib
import time

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent

WIN, STRIDE = 1500, 750
EVAL = "R20-H175b_qlane_eval.parquet"

# Banked, carried through - NOT recomputed (see the module docstring).
FLOOR_QUESTION_BLIND = 0.5000
FLOOR_SURFACE_PROBE = 0.5816
PRIMARY_GATE = 0.80

CKPTS = {1: "R20-H175b-arm-draw1"}
OUT = HERE / "R20-H175b_qeval_read.json"


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


Q = _mod("qchan", "R20-H175b_qchannel.py")
C = _mod("c", "R15_gate_common.py")


def windows(chunk):
    """R8-H101 / R16-H142 G0 `windows`, byte-identical."""
    n = len(chunk)
    if n <= WIN:
        return [chunk]
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return [chunk[s: s + WIN] for s in starts]


def flatten(claims, chunks):
    flat_c, flat_w, starts = [], [], []
    for cl, ch in zip(claims, chunks, strict=True):
        starts.append(len(flat_c))
        for w in windows(ch):
            flat_c.append(cl)
            flat_w.append(w)
    return flat_c, flat_w, np.array(starts, dtype=np.int64)


def auroc(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.asarray(y).astype(int), np.asarray(s)))


def pair_stats(s, pid):
    """Within-pair separation - the direct evidence that the channel moved."""
    order = np.argsort(pid, kind="stable")
    so, po = s[order], pid[order]
    a, b = so[0::2], so[1::2]
    if not np.array_equal(po[0::2], po[1::2]):
        raise SystemExit("QEVAL ABORT: rows do not pair up two-per-pair_id")
    d = np.abs(a - b)
    return {"n_pairs": int(len(a)), "pairs_bit_identical": int((a == b).sum()),
            "pairs_not_bit_identical": int((a != b).sum()),
            "max_abs_within_pair_delta": float(d.max()),
            "mean_abs_within_pair_delta": float(d.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", type=int, default=1, choices=tuple(CKPTS))
    args = ap.parse_args()

    import torch

    ckpt = CKPTS[args.draw]
    print(f"=== R20-H175b PRIMARY MECHANISM READ draw {args.draw}  "
          f"{time.strftime('%F %T')} ===", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"(CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})", flush=True)

    ev = pl.read_parquet(HERE / EVAL)
    y = ev["label"].to_numpy()
    pid = ev["pair_id"].to_numpy()
    claims = ev["claim"].to_list()
    chunks = ev["chunk"].to_list()
    quest = ev["question"].to_list()
    if ev.height != 2002 or ev["pair_id"].n_unique() != 1001:
        raise SystemExit(
            f"QEVAL ABORT: {ev.height} rows / {ev['pair_id'].n_unique()} pairs, "
            "expected 2,002 / 1,001 - this is not the banked stage-0 eval")
    if any(not (q or "").strip() for q in quest):
        raise SystemExit("QEVAL ABORT: a row of the eval carries no question")
    print(f"{EVAL}: {ev.height} rows / {ev['pair_id'].n_unique()} pairs / "
          f"{ev['doc_id'].n_unique()} docs", flush=True)

    conditioned = [Q.compose(q, c) for q, c in zip(quest, claims, strict=True)]
    same = sum(1 for a, b in zip(conditioned, claims, strict=True) if a == b)
    if same:
        raise SystemExit(f"QEVAL ABORT: {same} composed inputs equal the bare claim")

    presentations = {"question_conditioned": conditioned, "question_blind": claims}
    tok, trunk, head = C.load_ckpt(ckpt)
    res = {}
    for name, texts in presentations.items():
        fc, fw, st = flatten(texts, chunks)
        t0 = time.time()
        s_pair = C.score(tok, trunk, head, fc, fw)
        s = np.maximum.reduceat(np.asarray(s_pair, dtype=np.float64), st)
        np.save(HERE / f"R20-H175b_qeval_scores_{name}_draw{args.draw}.npy", s)
        res[name] = {"auroc": round(auroc(y, s), 6), **pair_stats(s, pid),
                     "windowed_pairs": len(fc),
                     "mean_windows_per_row": round(len(fc) / len(texts), 3),
                     "seconds": round(time.time() - t0, 1)}
        print(f"  {name}: AUROC {res[name]['auroc']:.6f}  "
              f"bit-identical pairs {res[name]['pairs_bit_identical']}/"
              f"{res[name]['n_pairs']}  max|delta| "
              f"{res[name]['max_abs_within_pair_delta']:.3e}", flush=True)

    blind = res["question_blind"]
    control_ok = (blind["pairs_bit_identical"] == blind["n_pairs"]
                  and abs(blind["auroc"] - 0.5) < 1e-9)
    if not control_ok:
        print("!!! QUESTION-BLIND CONTROL DEPARTED FROM EXACT CHANCE - the read "
              "path separates the two legs by something other than the question; "
              "the PRIMARY number is NOT cleanly attributable", flush=True)

    prim = res["question_conditioned"]["auroc"]
    OUT.write_text(json.dumps({
        "experiment": "R20-H175b PRIMARY mechanism read - held-out question-relevance "
                      "AUROC on the trained checkpoint (inference only)",
        "registration": ("docs/experiments/semantic-grounding-experiments.md, blocks "
                         "'R20-H175b QUESTION CONDITIONING', 'R20-H175b STAGE 0 "
                         "COMPLETE' and 'BASELINE LEGS BANKED'"),
        "draw": args.draw, "checkpoint": ckpt, "eval": EVAL,
        "n_rows": int(ev.height), "n_pairs": int(ev["pair_id"].n_unique()),
        "n_docs": int(ev["doc_id"].n_unique()),
        "protocol": (f"untruncated evidence windowed {WIN}/{STRIDE}, claim scored vs "
                     "every window, MAX over windows; frozen trunk + task head "
                     "(R15_gate_common.load_ckpt/.score); byte-identical to "
                     "R20_baseline_legs.py LEG 2 apart from the composed question"),
        "composition": f"'<question[:{Q.Q_MAX_CHARS}]>{Q.Q_SEP}<claim>'",
        "primary_auroc": prim,
        "primary_gate": PRIMARY_GATE,
        "banked_floors": {"question_blind_flagship": FLOOR_QUESTION_BLIND,
                          "surface_probe": FLOOR_SURFACE_PROBE,
                          "binding_floor": max(FLOOR_QUESTION_BLIND,
                                               FLOOR_SURFACE_PROBE)},
        "delta_vs_binding_floor": round(
            prim - max(FLOOR_QUESTION_BLIND, FLOOR_SURFACE_PROBE), 6),
        "question_blind_control_exact_chance": bool(control_ok),
        "results": res,
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    }, indent=2))
    print(f"\nPRIMARY {prim:.6f}  gate {PRIMARY_GATE}  binding floor "
          f"{max(FLOOR_QUESTION_BLIND, FLOOR_SURFACE_PROBE)}  "
          f"delta {prim - max(FLOOR_QUESTION_BLIND, FLOOR_SURFACE_PROBE):+.6f}",
          flush=True)
    print(f"results -> {OUT}", flush=True)
    print("=== R20-H175b PRIMARY MECHANISM READ COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()

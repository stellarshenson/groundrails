"""R9_PC_windowed_dump - per-response per-sentence WINDOWED scores, frozen H90.

Precursor P-C step 1 (round 9, analysis-only: no training, no tuning). Same
machinery as the R8-H101 windowed read (sentence x 1,500-char windows stride
750 over FULL chunks, max over windows per sentence via score_student's
max-over-chunks) but RETAINS the per-sentence windowed scores, one record per
arena response - the windowed twin of R8_analysis_h90_dump.json, minus the
heavy text fields (sentences/chunks/response omitted; P-B closed the
sentence-text exclusion class, aggregation headroom needs only the scores).

Sanity is downstream: R9_PC_headroom.py must reproduce R8-H101_result.json's
per-subset AUCs EXACTLY from these records before any aggregator is read.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/R9_PC_windowed_dump.py
"""

import importlib.util
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ARENA = _mod("arena", "R8-H77_unseen_arena.py")
H92 = _mod("h92", "R8-H92_decomposed_arena.py")

MODEL = str(HERE.parent.parent / "models" / "R8-H90-mmbert-dann-full")
OUT = HERE / "R9_PC_windowed_dump.json"

WIN = 1500
STRIDE = 750


def windows(chunk):
    """Sliding 1,500-char windows at stride 750; final window flush to the end."""
    n = len(chunk)
    if n <= WIN:
        return [chunk]
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return [chunk[s : s + WIN] for s in starts]


def main():
    subs = ARENA.load_subsets()
    print(f"RAGBench: {len(subs)} subsets, {sum(len(v[2]) for v in subs.values())} responses")

    records = []
    aucs = {}
    for sub, (claims, chunks, y) in subs.items():
        sent_lists = [H92.sentences(c) for c in claims]
        flat_s, flat_w, owner = [], [], []
        for i, (sl, ks) in enumerate(zip(sent_lists, chunks, strict=True)):
            wlist = [w for k in ks for w in windows(k)]
            for s in sl:
                flat_s.append(s)
                flat_w.append(wlist)
                owner.append(i)
        owner = np.array(owner)
        scores = ARENA.score_student(MODEL, flat_s, flat_w)
        resp = np.zeros(len(y))
        for i in range(len(y)):
            ss = scores[owner == i]
            records.append(
                {
                    "subset": sub,
                    "idx": i,
                    "label": int(y[i]),
                    "score": round(float(ss.min()), 6),
                    "mean_score": round(float(ss.mean()), 6),
                    "argmin_idx": int(ss.argmin()),
                    "sent_scores": [round(float(v), 6) for v in ss],
                    "n_sent": int(len(ss)),
                }
            )
            resp[i] = ss.min()
        auc, f1, _ = ARENA.M59.auc_and_f1(y, resp)
        aucs[sub] = round(auc, 4)
        print(f"  {sub:14s} n={len(y):>4}  auc {auc:.4f}", flush=True)

    OUT.write_text(json.dumps(
        {"aucs": aucs, "model": MODEL, "window": WIN, "stride": STRIDE, "records": records}
    ))
    print(f"\nmean AUC {np.mean(list(aucs.values())):.4f}")
    print(f"-> {OUT}  ({len(records)} records)")
    print("=== R9_PC DUMP DONE ===")


if __name__ == "__main__":
    main()

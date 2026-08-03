"""R8_analysis_h90_dump - per-response decomposed-min records for failure analysis.

Analysis-only (no training, no tuning). Scores models/R8-H90-mmbert-dann-full
through the frozen decomposed-min path (H92 sentence split -> per-sentence
max-over-chunks -> min-over-sentences) and dumps one record per arena response:
label, response score, per-sentence scores, the argmin sentence, splitter
diagnostics, plus the comparator's whole-response score for label-noise
triangulation. Verifies per-subset AUC against the recorded R8-H90 read.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/R8_analysis_h90_dump.py
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
OUT = HERE / "R8_analysis_h90_dump.json"


def main():
    subs = ARENA.load_subsets()
    print(f"RAGBench: {len(subs)} subsets, {sum(len(v[2]) for v in subs.values())} responses")

    records = []
    aucs = {}
    for sub, (claims, chunks, y) in subs.items():
        sent_lists = [H92.sentences(c) for c in claims]
        flat_s, flat_k, owner = [], [], []
        for i, (sl, ks) in enumerate(zip(sent_lists, chunks, strict=True)):
            for s in sl:
                flat_s.append(s)
                flat_k.append(ks)
                owner.append(i)
        owner = np.array(owner)
        scores = ARENA.score_student(MODEL, flat_s, flat_k)
        lett = ARENA.score_lettuce(claims, chunks)

        resp = np.zeros(len(y))
        for i in range(len(y)):
            ss = scores[owner == i]
            mn, am = float(ss.min()), int(ss.argmin())
            raw = [p.strip() for p in H92._SPLIT.split(claims[i])]
            surviving = [p for p in raw if len(p) >= H92.MIN_SENT_CHARS]
            fallback = len(surviving[: H92.MAX_SENTS]) < 2
            records.append(
                {
                    "subset": sub,
                    "idx": i,
                    "label": int(y[i]),
                    "score": round(mn, 4),
                    "mean_score": round(float(ss.mean()), 4),
                    "lettuce_score": round(float(lett[i]), 4),
                    "argmin_idx": am,
                    "argmin_sentence": sent_lists[i][am],
                    "sent_scores": [round(float(v), 4) for v in ss],
                    "sentences": sent_lists[i],
                    "n_sent": len(sent_lists[i]),
                    "n_raw_parts": len(raw),
                    "n_short_dropped": len(raw) - len(surviving),
                    "truncated_at_cap": len(surviving) > H92.MAX_SENTS,
                    "fallback_whole": bool(fallback),
                    "resp_chars": len(claims[i]),
                    "n_chunks": len(chunks[i]),
                    "response": claims[i],
                    "chunks": [k[:2000] for k in chunks[i]],
                }
            )
            resp[i] = mn
        auc, f1, _ = ARENA.M59.auc_and_f1(y, resp)
        aucs[sub] = round(auc, 4)
        print(f"  {sub:14s} n={len(y):>4}  auc {auc:.4f}  f1 {f1:.4f}", flush=True)

    OUT.write_text(json.dumps({"aucs": aucs, "model": MODEL, "records": records}))
    print(f"\nmean AUC {np.mean(list(aucs.values())):.4f}")
    print(f"-> {OUT}  ({len(records)} records)")


if __name__ == "__main__":
    main()

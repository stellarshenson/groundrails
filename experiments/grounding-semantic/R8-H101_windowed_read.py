"""R8-H101 - windowed evidence on frozen H90 weights (pre-registered).

Because the worst grounded FNs are truncation-starved sentences (R8 failure
analysis: 86%/74% bottom-quartile exposure in finqa/delucionqa, needle-verified)
and min-aggregation is maximally sensitive to them, scoring each sentence
against 1,500-char windows (stride 750, fixed, not tuned) of every FULL chunk -
max over windows over chunks, MIN over sentences unchanged - will lift finqa
and delucionqa. Bar (recorded, not adjustable here): mean >= 0.7213 AND finqa
AND delucionqa each >= +0.03 vs the H90 baseline read; kill: either < +0.01.

Mechanics: ARENA.load_subsets() already returns FULL chunk texts (the 1,500-char
truncation lives in score_student's pair assembly). Each chunk is expanded to
sliding windows of 1,500 chars at stride 750 (final window flush to the chunk
end, no fully-redundant subset windows); the per-sentence "chunk list" becomes
the window list, so score_student's internal k[:1500] truncation is a no-op and
its max-over-chunks IS max-over-windows. Frozen checkpoint, identical gate data,
identical splitter, identical min.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      uv run python experiments/grounding-semantic/R8-H101_windowed_read.py [--sanity]
"""

import argparse
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
OUT = HERE / "R8-H101_result.json"

WIN = 1500
STRIDE = 750

# H90 baseline decomposed-min read (R8_decomposed_reads.json, tag R8-H90).
BASELINE = {
    "covidqa": 0.7755, "delucionqa": 0.7263, "emanual": 0.7058, "expertqa": 0.8248,
    "finqa": 0.6730, "hagrid": 0.6516, "hotpotqa": 0.7253, "pubmedqa": 0.6058,
    "tatqa": 0.7718, "techqa": 0.7529,
}


def windows(chunk):
    """Sliding 1,500-char windows at stride 750; final window flush to the end."""
    n = len(chunk)
    if n <= WIN:
        return [chunk]
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return [chunk[s : s + WIN] for s in starts]


def score_subset(claims, chunks, y, first_window_only=False):
    sent_lists = [H92.sentences(c) for c in claims]
    flat_s, flat_w, owner = [], [], []
    for i, (sl, ks) in enumerate(zip(sent_lists, chunks, strict=True)):
        if first_window_only:
            wlist = [k[:WIN] for k in ks]
        else:
            wlist = [w for k in ks for w in windows(k)]
        for s in sl:
            flat_s.append(s)
            flat_w.append(wlist)
            owner.append(i)
    owner = np.array(owner)
    scores = ARENA.score_student(MODEL, flat_s, flat_w)
    resp = np.array([scores[owner == i].min() for i in range(len(y))])
    auc, f1, _ = ARENA.M59.auc_and_f1(y, resp)
    n_pairs = sum(len(w) for w in flat_w)
    return auc, f1, n_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true", help="delucionqa only: window stats + first-window-only reproduction")
    ap.add_argument("--model", default=None, help="checkpoint override (default: the H90 holder)")
    ap.add_argument("--out", default=None, help="result json override")
    args = ap.parse_args()
    global MODEL, OUT
    if args.model:
        MODEL = args.model
    if args.out:
        OUT = HERE / args.out

    subs = ARENA.load_subsets()

    if args.sanity:
        claims, chunks, y = subs["delucionqa"]
        lens = [len(k) for ks in chunks for k in ks]
        over = sum(1 for l in lens if l > WIN)
        print(f"delucionqa: {len(lens)} chunks, {over} ({over/len(lens):.1%}) longer than {WIN} chars, "
              f"max {max(lens)}, mean window count {np.mean([len(windows(k)) for ks in chunks for k in ks]):.2f}")
        auc, f1, n_pairs = score_subset(claims, chunks, y, first_window_only=True)
        print(f"first-window-only read: AUC {auc:.4f} (baseline decomposed read {BASELINE['delucionqa']:.4f}) "
              f"[{n_pairs} windows] - should match the baseline")
        return

    rows = {}
    for sub, (claims, chunks, y) in subs.items():
        auc, f1, n_pairs = score_subset(claims, chunks, y)
        base = BASELINE[sub]
        rows[sub] = {
            "n": len(y), "auc": round(auc, 4), "f1": round(f1, 4),
            "baseline_h90": base, "delta": round(auc - base, 4),
            "lettuce_auc": H92.LETTUCE[sub], "n_window_pairs": n_pairs,
        }
        print(f"  {sub:14s} n={len(y):>4} windowed {auc:.4f}  baseline {base:.4f}  "
              f"delta {auc - base:+.4f}  lettuce {H92.LETTUCE[sub]:.4f}", flush=True)

    mean = float(np.mean([r["auc"] for r in rows.values()]))
    base_mean = float(np.mean(list(BASELINE.values())))
    let = float(np.mean([r["lettuce_auc"] for r in rows.values()]))
    d_fin = rows["finqa"]["delta"]
    d_del = rows["delucionqa"]["delta"]

    bar = mean >= 0.7213 and d_fin >= 0.03 and d_del >= 0.03
    kill = d_fin < 0.01 or d_del < 0.01
    verdict = "CONFIRMED" if bar else ("KILLED (a losing-subset delta < +0.01)" if kill else "REFUTED (bar missed, kill not triggered)")

    print("\n" + "=" * 92)
    print("R8-H101 RESULT - windowed evidence, frozen H90 weights, deterministic read")
    print("=" * 92)
    print(f"  windowed mean   {mean:.4f}   baseline {base_mean:.4f}   lettuce {let:.4f}")
    print(f"  finqa delta {d_fin:+.4f}   delucionqa delta {d_del:+.4f}")
    print(f"  bar: mean >= 0.7213 AND finqa/delucionqa each >= +0.03  ->  {verdict}")

    OUT.write_text(json.dumps({
        "per_subset": rows, "mean": mean, "baseline_mean": base_mean,
        "mean_lettuce": let, "finqa_delta": d_fin, "delucionqa_delta": d_del,
        "verdict": verdict, "model": MODEL, "window": WIN, "stride": STRIDE,
    }, indent=2))
    print(f"\n  results -> {OUT}")


if __name__ == "__main__":
    main()

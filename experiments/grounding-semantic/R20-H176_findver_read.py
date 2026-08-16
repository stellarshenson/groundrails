"""R20-H176 FINDVER INSTRUMENT READ - measurement only, zero training.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R20-H176 FINDVER INSTRUMENT READ" (2026-08-16). Scores the banked flagship
draws on the banked FinDVer lane with the SHIPPED windowed read protocol and
reports per-subset AUROC + the FP/FN split at the macro-F1 operating point.

    protocol   evidence UNTRUNCATED, presented as 1,500-char windows at stride
               750 (R8-H101 / R16-H142 G0 `windows`, byte-identical), claim
               scored against every window, MAX over windows
    difference FinDVer rows are CLAIM-level, not response-level, so the arena
               read's MIN-over-response-sentences stage does not apply: one
               claim, one bag, one score
    checkpoints models/R18-H150-arm-draw1, models/R18-H150-arm-draw2 (trunk +
               task head through R15_gate_common.load_ckpt / .score - the same
               frozen-read path every banked probe used)

Branches are the registered ones and are evaluated on the 2-draw mean:
    numeric < 0.65 AND numeric < (ie+knowledge)/2 - 0.05  -> CONFIRMED
    0.65 <= numeric < 0.75                                -> PARTIAL
    numeric >= 0.75 or numeric ~= others                  -> DEPRIORITISE

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
      uv run python experiments/grounding-semantic/R20-H176_findver_read.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util
import json
import pathlib
import time

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
LANE = HERE / "R19_findver_lane.parquet"
OUT = HERE / "R20-H176_findver_read.json"
DRAWS = {"h150d1": "R18-H150-arm-draw1", "h150d2": "R18-H150-arm-draw2"}
WIN, STRIDE = 1500, 750
SUBSETS = ("ie", "numeric", "knowledge")


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def windows(chunk):
    """R8-H101 / R16-H142 G0 `windows`, byte-identical."""
    n = len(chunk)
    if n <= WIN:
        return [chunk]
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return [chunk[s : s + WIN] for s in starts]


def macro_f1_point(y, s):
    """Macro-F1-optimal threshold on the 91-quantile grid (R7-H59 grid), fitted
    and reported on the same set so the FP/FN split covers every row; the
    half-split R7-H59 F1 is reported alongside for continuity."""
    from sklearn.metrics import f1_score, roc_auc_score

    grid = np.quantile(s, np.linspace(0.05, 0.95, 91))
    thr = max(grid, key=lambda t: f1_score(y, (s >= t).astype(int), average="macro"))
    pred = (s >= thr).astype(int)
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    # half-split reference (R7-H59.auc_and_f1)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    a, b = idx[: len(idx) // 2], idx[len(idx) // 2 :]
    ga = np.quantile(s[a], np.linspace(0.05, 0.95, 91))
    ta = max(ga, key=lambda t: f1_score(y[a], (s[a] >= t).astype(int), average="macro"))
    return {
        "auroc": round(float(roc_auc_score(y, s)), 4),
        "thr": round(float(thr), 6),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "fp": fp,
        "fn": fn,
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "fp_rate": round(fp / max(int((y == 0).sum()), 1), 4),
        "fn_rate": round(fn / max(int((y == 1).sum()), 1), 4),
        "fp_share_of_errors": round(fp / max(fp + fn, 1), 4),
        "halfsplit_macro_f1": round(
            float(f1_score(y[b], (s[b] >= ta).astype(int), average="macro")), 4
        ),
    }


def main():
    import torch

    C = _mod("c", "R15_gate_common.py")
    df = pl.read_parquet(LANE)
    print(f"=== R20-H176 findver read  {time.strftime('%F %T')} ===", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"lane {df.height} rows, subsets "
          f"{df.group_by('subset').len().sort('subset').to_dicts()}", flush=True)

    claims = df["claim"].to_list()
    chunks = df["chunk"].to_list()
    y = df["label"].to_numpy().astype(int)
    sub = np.array(df["subset"].to_list())

    flat_c, flat_w, owner = [], [], []
    for i, (cl, ch) in enumerate(zip(claims, chunks, strict=True)):
        for w in windows(ch):
            flat_c.append(cl)
            flat_w.append(w)
            owner.append(i)
    owner = np.array(owner)
    print(f"windowed: {len(flat_c)} (claim, window) pairs, "
          f"mean {len(flat_c) / df.height:.3f} windows/row", flush=True)

    per_draw = {}
    for tag, ckpt in DRAWS.items():
        t0 = time.time()
        tok, trunk, head = C.load_ckpt(ckpt)
        s_pair = C.score(tok, trunk, head, flat_c, flat_w)
        del trunk, head
        torch.cuda.empty_cache()
        s = np.array([s_pair[owner == i].max() for i in range(df.height)])
        rows = {k: macro_f1_point(y[sub == k], s[sub == k]) for k in SUBSETS}
        rows["overall"] = macro_f1_point(y, s)
        per_draw[tag] = {"checkpoint": ckpt, **rows}
        print(f"  {tag}: " + "  ".join(
            f"{k} {rows[k]['auroc']:.4f}" for k in (*SUBSETS, "overall"))
            + f"   ({time.time() - t0:.0f}s)", flush=True)
        np.save(HERE / f"R20-H176_findver_scores_{tag}.npy", s)

    mean = {
        k: {
            "auroc": round(float(np.mean([per_draw[d][k]["auroc"] for d in DRAWS])), 4),
            "fp": int(np.sum([per_draw[d][k]["fp"] for d in DRAWS])),
            "fn": int(np.sum([per_draw[d][k]["fn"] for d in DRAWS])),
            "fp_share_of_errors": round(
                float(np.mean([per_draw[d][k]["fp_share_of_errors"] for d in DRAWS])), 4),
        }
        for k in (*SUBSETS, "overall")
    }
    num = mean["numeric"]["auroc"]
    others = (mean["ie"]["auroc"] + mean["knowledge"]["auroc"]) / 2
    if num < 0.65 and num < others - 0.05:
        branch = "CONFIRMED"
        note = ("numeric < 0.65 and numeric < (ie+knowledge)/2 - 0.05 -> deficit "
                "derivation-specific off-arena; FinDVer-numeric banks as the standing "
                "non-arena mechanism instrument")
    elif num >= 0.75:
        branch = "DEPRIORITISE"
        note = ("numeric >= 0.75 -> the in-model deficit does not transfer to human "
                "financial claims; the derivation lane direction is de-prioritised")
    elif 0.65 <= num < 0.75:
        branch = "PARTIAL"
        note = ("numeric in [0.65, 0.75) -> partial, instrument usable as a delta gate")
    else:
        branch = "NUMERIC_NOT_SEPARATED"
        note = ("numeric < 0.65 but within 0.05 of the non-numeric mean - the low read "
                "is not numeric-specific; coordinator adjudicates")

    payload = {
        "experiment": "R20-H176 FinDVer instrument read (measurement only, zero training)",
        "registration": ("docs/experiments/semantic-grounding-experiments.md, block "
                         "'R20-H176 FINDVER INSTRUMENT READ' (2026-08-16)"),
        "lane": str(LANE.relative_to(HERE.parent.parent)),
        "protocol": (f"untruncated evidence windowed {WIN}/{STRIDE}, claim scored vs every "
                     "window, MAX over windows; claim-level rows so no min-over-sentences "
                     "stage; frozen trunk + task head (R15_gate_common.load_ckpt/.score)"),
        "n_rows": df.height,
        "n_pairs": len(flat_c),
        "mean_windows_per_row": round(len(flat_c) / df.height, 4),
        "per_draw": per_draw,
        "two_draw_mean": mean,
        "numeric_vs_nonnumeric_mean": round(num - others, 4),
        "branch": branch,
        "branch_note": note,
        "operating_point": ("macro-F1-optimal threshold on the 91-quantile grid fitted on "
                            "the scored set itself (so the FP/FN split covers every row); "
                            "the R7-H59 half-split macro-F1 is carried as "
                            "halfsplit_macro_f1 for continuity"),
        "timestamp": time.strftime("%F %T"),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"branch: {branch}  (numeric {num:.4f} vs non-numeric mean {others:.4f})",
          flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

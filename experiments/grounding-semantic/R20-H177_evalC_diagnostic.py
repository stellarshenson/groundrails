"""R20-H177 eval_C ATTRIBUTION DIAGNOSTIC - pre-H146 checkpoints on the two held-out
mechanism evals. Measurement only, zero training, GPU2 only.

Registered by the coordinator (2026-08-17) off the R20-H177 baseline leg, which
found the banked flagship already reads eval_C at 0.9085 against a registration
that predicted near-chance. Two live explanations:

    (a) the flagship's 0.9085 comes from the `R17-H146_lane` 30,000-row
        `quant_misbind` DANN group already in its mix - Lane C would then be
        largely redundant with an installed lane
    (b) any competent grounding checkpoint reads role/period misbind at ~0.9 -
        Lane C's eval would then be measuring general grounding competence
        rather than the targeted channel

The separator is a checkpoint of comparable quality trained WITHOUT the misbind
lane. Under (a) the pre-H146 read drops toward chance; under (b) it stays high.

CHECKPOINT CHOICE, and why it cleanly predates the lane
-------------------------------------------------------
`models/R16-H142-T-arm-draw1` is an untracked SYMLINK to `models/R16-H142-G1-twin`
(recorded in the canonical log, R16-H142 T holds block: "the stage's checkpoint
resolution required an untracked symlink models/R16-H142-T-arm-draw1 ->
R16-H142-G1-twin"). The real directories are used here so the artifact names are
unambiguous.

Both twin draws carry `init_fingerprint.json` with `n_groups: 12` and a group map
of halueval / psiloqa / ragtruth x 8 / tabfact / vitaminc - NO `quant_misbind`
and NO `quant_scale_unit`. The flagship `R18-H150-arm-draw{1,2}` carries 14
groups, the extra two being exactly those lanes (`R18-H150_arm_run.py`,
EXPECTED_GROUPS). So the twin pair predates the misbind lane's entry into the mix
by construction of its own banked fingerprint, not by date alone.

Comparable quality, from the banked blind windowed arena reads:
    R16-H142-G1-twin (seed 1142)  arena mean 0.72498
    R16-H142-T-draw2 (seed 2142)  arena mean 0.70073
    R18-H150-arm-draw1            arena mean 0.71436
    R18-H150-arm-draw2            arena mean 0.71661
The pre-H146 pair brackets the flagship pair, so a drop on eval_C cannot be
dismissed as a weaker checkpoint.

Confound noted, not corrected: neither the twin mix nor the flagship mix contains
any EDGAR text, and eval_C is 100% EDGAR prose. Both sides are equally
EDGAR-naive; the misbind lane (TabFact-derived tables) is the only differing
ingredient bearing on this family.

Second control: the same pre-H146 pair on `R20-H177_eval_B.parquet`. If Lane B's
floor is ~0.5 on old and new checkpoints alike, eval_B measures something
genuinely absent rather than something H146 happened to install.

Read path identical to the baseline leg (`R20_baseline_legs.py`) and to
`R20-H176_findver_read.py`: evidence untruncated, 1,500-char windows at stride
750, claim scored against every window, MAX over windows, frozen trunk + task
head via `R15_gate_common.load_ckpt` / `.score`.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 \
      uv run python experiments/grounding-semantic/R20-H177_evalC_diagnostic.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util
import json
import pathlib
import time

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
MODELS = HERE.parent.parent / "models"
OUT = HERE / "R20-H177_evalC_diagnostic.json"
WIN, STRIDE = 1500, 750

# Pre-H146: the windowed-MIL twin pair, 12 DANN groups, no misbind lane.
PRE_H146 = {"h142twin_d1": "R16-H142-G1-twin", "h142twin_d2": "R16-H142-T-draw2"}
# Banked flagship reads from R20-H177_baseline_leg.json, carried for the table.
FLAGSHIP_BASELINE = HERE / "R20-H177_baseline_leg.json"
# Banked blind windowed arena means, for the comparable-quality statement.
ARENA_MEAN = {
    "R16-H142-G1-twin": 0.72498, "R16-H142-T-draw2": 0.70073,
    "R18-H150-arm-draw1": 0.71436, "R18-H150-arm-draw2": 0.71661,
}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def windows(chunk):
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


def group_map(ckpt):
    """The checkpoint's own banked DANN group map - the misbind-lane proof."""
    fp = json.loads((MODELS / ckpt / "init_fingerprint.json").read_text())
    return sorted(fp["group_counts"].keys()), int(fp["n_groups"]), fp.get("seed")


def main():
    import torch

    C = _mod("c", "R15_gate_common.py")
    t_all = time.time()
    print(f"=== R20-H177 eval_C ATTRIBUTION DIAGNOSTIC  {time.strftime('%F %T')} ===",
          flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"(CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})", flush=True)

    # ---- prove the pre-H146 provenance from the checkpoints' own fingerprints -- #
    prov = {}
    for tag, ckpt in {**PRE_H146,
                      "h150d1": "R18-H150-arm-draw1",
                      "h150d2": "R18-H150-arm-draw2"}.items():
        groups, n, seed = group_map(ckpt)
        has_misbind = "quant_misbind" in groups
        prov[tag] = {"checkpoint": ckpt, "n_dann_groups": n, "seed": seed,
                     "has_quant_misbind_group": has_misbind,
                     "has_quant_scale_unit_group": "quant_scale_unit" in groups,
                     "groups": groups,
                     "banked_arena_mean": ARENA_MEAN.get(ckpt)}
        print(f"  {tag:<12} {ckpt:<20} groups {n:>2}  quant_misbind={has_misbind}  "
              f"arena_mean {ARENA_MEAN.get(ckpt)}", flush=True)
    for tag in PRE_H146:
        if prov[tag]["has_quant_misbind_group"]:
            raise SystemExit(
                f"DIAGNOSTIC ABORT: {prov[tag]['checkpoint']} carries a quant_misbind "
                "group - it does NOT predate the H146 lane and cannot serve as the "
                "pre-H146 control")
    for tag in ("h150d1", "h150d2"):
        if not prov[tag]["has_quant_misbind_group"]:
            raise SystemExit(
                f"DIAGNOSTIC ABORT: flagship {prov[tag]['checkpoint']} does NOT carry a "
                "quant_misbind group - the contrast this diagnostic rests on is not real")
    print("  provenance verified: pre-H146 pair has 12 groups without the misbind "
          "lane; flagship pair has 14 groups with it", flush=True)

    # ---- the two registered evals ---------------------------------------- #
    ev = {"eval_C": pl.read_parquet(HERE / "R20-H177_eval_C.parquet"),
          "eval_B": pl.read_parquet(HERE / "R20-H177_eval_B.parquet")}
    flat = {}
    for k, d in ev.items():
        fc, fw, st = flatten(d["claim"].to_list(), d["chunk"].to_list())
        flat[k] = (fc, fw, st, d.height)
        print(f"windowed {k}: {len(fc)} (claim, window) pairs over {d.height} rows "
              f"({d['pair_id'].n_unique()} pairs, {d['doc_id'].n_unique()} docs)",
              flush=True)

    # ---- score both evals on both pre-H146 draws -------------------------- #
    results = {k: {"n_rows": ev[k].height,
                   "n_pairs": int(ev[k]["pair_id"].n_unique()),
                   "n_docs": int(ev[k]["doc_id"].n_unique()),
                   "pre_h146": {}} for k in ev}
    for tag, ckpt in PRE_H146.items():
        t0 = time.time()
        tok, trunk, head = C.load_ckpt(ckpt)
        for k, (fc, fw, st, n) in flat.items():
            s_pair = C.score(tok, trunk, head, fc, fw)
            s = np.maximum.reduceat(np.asarray(s_pair, dtype=np.float64), st)
            assert len(s) == n
            np.save(HERE / f"R20-H177_evalC_diagnostic_scores_{k}_{tag}.npy", s)
            y = ev[k]["label"].to_numpy()
            famv = np.array(ev[k]["neg_family"].to_list())
            fam = {}
            for f in sorted(set(famv.tolist())):
                m = famv == f
                if len(set(y[m].tolist())) == 2:
                    fam[f] = {"n_rows": int(m.sum()), "auroc": round(auroc(y[m], s[m]), 4)}
            results[k]["pre_h146"][tag] = {
                "checkpoint": ckpt, "banked_arena_mean": ARENA_MEAN[ckpt],
                "auroc": round(auroc(y, s), 4), "by_neg_family": fam}
        del trunk, head
        torch.cuda.empty_cache()
        print(f"  {tag} ({ckpt}) scored both evals in {time.time() - t0:.0f}s", flush=True)

    # ---- carry the banked flagship reads in for the side-by-side ---------- #
    base = json.loads(FLAGSHIP_BASELINE.read_text())["results"]
    for k in ev:
        results[k]["flagship_post_h146"] = base[k]["per_draw"]
        results[k]["flagship_two_draw_mean_auroc"] = base[k]["two_draw_mean_auroc"]
        results[k]["pre_h146_two_draw_mean_auroc"] = round(
            float(np.mean([results[k]["pre_h146"][t]["auroc"] for t in PRE_H146])), 4)
        results[k]["delta_flagship_minus_pre_h146"] = round(
            results[k]["flagship_two_draw_mean_auroc"]
            - results[k]["pre_h146_two_draw_mean_auroc"], 4)
        fams = sorted({f for t in PRE_H146 for f in results[k]["pre_h146"][t]["by_neg_family"]})
        results[k]["per_family_side_by_side"] = {
            f: {"pre_h146_mean": round(float(np.mean(
                    [results[k]["pre_h146"][t]["by_neg_family"][f]["auroc"]
                     for t in PRE_H146])), 4),
                "flagship_mean": round(float(np.mean(
                    [base[k]["per_draw"][t]["by_neg_family"][f]["auroc"]
                     for t in ("h150d1", "h150d2")])), 4)}
            for f in fams}

    payload = {
        "experiment": ("R20-H177 eval_C attribution diagnostic - pre-H146 checkpoints on "
                       "the held-out mechanism evals (measurement only, zero training)"),
        "registered_by": ("coordinator ruling 2026-08-17, off the R20-H177 baseline leg "
                          "(R20-H177_baseline_leg.json)"),
        "question": {
            "a": ("the flagship's eval_C 0.9085 comes from the R17-H146_lane 30,000-row "
                  "quant_misbind group already in its mix"),
            "b": ("any competent grounding checkpoint reads role/period misbind at ~0.9, "
                  "so eval_C measures general grounding competence"),
            "separator": ("a comparable-quality checkpoint trained WITHOUT the misbind "
                          "lane: under (a) the pre-H146 read drops toward chance, under "
                          "(b) it stays high"),
        },
        "checkpoint_provenance": prov,
        "symlink_note": ("models/R16-H142-T-arm-draw1 is an untracked symlink to "
                         "models/R16-H142-G1-twin (canonical log, R16-H142 T holds "
                         "block); the real directories are used here"),
        "confound_noted": ("neither mix contains EDGAR text and eval_C is 100% EDGAR "
                           "prose - both sides are equally EDGAR-naive, so the misbind "
                           "lane is the only differing ingredient bearing on this family"),
        "protocol": (f"untruncated evidence windowed {WIN}/{STRIDE}, claim scored vs every "
                     "window, MAX over windows; frozen trunk + task head "
                     "(R15_gate_common.load_ckpt/.score) - identical to the baseline leg"),
        "results": results,
        "timestamp": time.strftime("%F %T"),
    }
    OUT.write_text(json.dumps(payload, indent=2))

    for k in ("eval_C", "eval_B"):
        r = results[k]
        print(f"{k}: pre-H146 " + "  ".join(
            f"{t} {r['pre_h146'][t]['auroc']:.4f}" for t in PRE_H146)
            + f"  mean {r['pre_h146_two_draw_mean_auroc']:.4f}"
            + f"   |  flagship mean {r['flagship_two_draw_mean_auroc']:.4f}"
            + f"   delta {r['delta_flagship_minus_pre_h146']:+.4f}", flush=True)
        for f, v in r["per_family_side_by_side"].items():
            print(f"    {f:<14} pre-H146 {v['pre_h146_mean']:.4f}   "
                  f"flagship {v['flagship_mean']:.4f}", flush=True)
    print(f"wrote {OUT.name}   total {time.time() - t_all:.0f}s", flush=True)
    print("=== R20-H177 EVALC DIAGNOSTIC COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()

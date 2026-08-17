"""R20-H178 LEXICAL-LATE-FUSION - zero-training paired read on the banked flagship pair.

Registered in docs/experiments/semantic-grounding-experiments.md, blocks
"R20-H178 LEXICAL-LATE-FUSION" and "R20-H178 AMENDMENT A1". Nothing trains here.

The question: the deterministic lexical token-containment scorer and the flagship
cross-encoder decorrelate at subset level on the blind arena (R19-H162: emanual
containment 0.7763 vs model 0.6973; delucionqa model 0.8009 vs containment
0.5889). Does a subset-blind late blend harvest that decorrelation?

    fused(pair) = (1 - w) * sigmoid(model_logit) + w * tok_containment

fused at the PAIR grain, then the SHIPPED aggregation unchanged - arena: max over
the sentence's windows, then MIN over the response's sentences; in-domain: max
over the claim's chunks. One global w, identical for every input; no
subset-conditioned logic anywhere. The pair-grain choice is pinned by gate (b):
at w = 1 the arena read collapses exactly onto R19-H162's banked containment
AUROCs, which were computed as max-over-windows then min-over-sentences.

MODEL SCORES.  The arena leg needs no GPU: `R19-H161_pairs_h150d{1,2}.parquet`
are the banked per-PAIR dumps of exactly this windowed read on exactly these two
checkpoints, positive-controlled at write time against the banked per-subset
AUROCs and structural fingerprints. The registration sanctions their reuse
("recomputed on GPU0 if per-sentence dumps are not banked"). Gate (a) re-verifies
the reproduction at the tighter 1e-4 before anything else runs. The in-domain leg
(gold_full + the seven RAGTruth translations) has no banked per-pair dump, so it
is recomputed on GPU0 through `R19-H161_dump.score_pairs`, which is
`R16-H142_G1_arm.score_sets`' encode path with the per-set max deferred.

STAGES, in order, each idempotent against its on-disk artifact:

    1  gates      CPU. (a) w=0 reproduces the banked windowed arena reads to
                  <= 1e-4 on every subset of both draws; (b) the lexical scorer
                  alone reproduces emanual 0.7763 and delucionqa 0.5889 to
                  <= 1e-3. Either miss -> ABORT, no arena read.
    2  indomain   GPU0. Per-pair (logit, containment) for gold_full and the seven
                  translations, per draw -> R20-H178_indomain_draw{1,2}.parquet
    3  select     CPU. w over {0.05 ... 0.50} on gold_full ONLY. The arena is
                  never consulted. Holds required AT the selected w on BOTH
                  draws: gold_full >= 0.84, non-EN >= 0.82.
    4  arena      CPU. ONE fused read per draw at the selected w, plus the w=0
                  baseline. Ten subsets, uniform mean.

SCOPING (amendment A1): the fused number is a SYSTEM read - cross-encoder plus
lexical tier - banked under `system_mean`. It is never substituted into, averaged
with, or reported as the model's arena mean.

Run (detached, GPU0 only - cards 1 and 2 are running the H172 draws):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  nohup setsid uv run python experiments/grounding-semantic/R20-H178_fusion_read.py \
    2>&1 | tee logs/R20-H178_fusion_read.log &
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import importlib.util
import json
import pathlib
import sys
import time

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
OUT_JSON = HERE / "R20-H178_result.json"

DRAWS = {
    1: {"ckpt": "R18-H150-arm-draw1",
        "pairs": "R19-H161_pairs_h150d1.parquet",
        "banked_arena": "R18-H150_arm_draw1_windowed_result.json",
        "banked_suite": "R18-H150_arm_draw1_result.json",
        "indomain": "R20-H178_indomain_draw1.parquet"},
    2: {"ckpt": "R18-H150-arm-draw2",
        "pairs": "R19-H161_pairs_h150d2.parquet",
        "banked_arena": "R18-H150_arm_draw2_windowed_result.json",
        "banked_suite": "R18-H150_arm_draw2_result.json",
        "indomain": "R20-H178_indomain_draw2.parquet"},
}

# Gate (b) reference - R19-H162_procedural_mech2.json, `auroc_lexical`.
GATE_B_REF = {"emanual": 0.77633, "delucionqa": 0.58891}
GATE_A_TOL = 1e-4
GATE_B_TOL = 1e-3

W_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
HOLD_GOLD_FULL = 0.84
HOLD_NONEN = 0.82
LANGS = ("de", "fr", "es", "it", "pl", "hu", "cn")


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def banked_per_subset(fname):
    data = json.loads((HERE / fname).read_text())
    block = data.get("per_subset", data)
    return {k: v for k, v in block.items() if isinstance(v, dict) and "auc" in v}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


# --- stage 1: gates (CPU) ------------------------------------------------------------


def arena_item_scores(df, w):
    """Fuse at the pair grain, max over the sentence's windows, MIN over the
    response's sentences - the shipped windowed decomposed-min aggregation."""
    fused = (1.0 - w) * sigmoid(df["logit"].to_numpy()) + w * df["tok_containment"].to_numpy()
    return (
        df.with_columns(pl.Series("fused", fused))
        .group_by(["item_id", "sent_idx"])
        .agg(pl.col("fused").max(), pl.col("label").first())
        .group_by("item_id")
        .agg(pl.col("fused").min(), pl.col("label").first())
        .sort("item_id")
    )


def arena_read(frames, w):
    """Per-subset AUROC and the uniform ten-subset mean at one w."""
    rows = {}
    for sub, df in frames.items():
        agg = arena_item_scores(df, w)
        rows[sub] = float(roc_auc_score(agg["label"].to_numpy(), agg["fused"].to_numpy()))
    return rows, float(np.mean(list(rows.values())))


def load_arena_frames(draw):
    df = pl.read_parquet(HERE / DRAWS[draw]["pairs"])
    return {s: df.filter(pl.col("subset") == s) for s in sorted(df["subset"].unique().to_list())}


def run_gates(frames):
    print(f"\n--- stage 1: SANITY GATES  {time.strftime('%F %T')} ---", flush=True)
    gate_a = {"tol": GATE_A_TOL, "per_draw": {}}
    worst_a = 0.0
    for d in DRAWS:
        banked = banked_per_subset(DRAWS[d]["banked_arena"])
        rows, mean = arena_read(frames[d], 0.0)
        per = {}
        for sub, auc in rows.items():
            delta = abs(auc - float(banked[sub]["auc"]))
            worst_a = max(worst_a, delta)
            per[sub] = {"reproduced": round(auc, 6), "banked": float(banked[sub]["auc"]),
                        "abs_delta": round(delta, 6), "pass": bool(delta <= GATE_A_TOL)}
        gate_a["per_draw"][f"draw{d}"] = {
            "per_subset": per,
            "reproduced_mean": round(mean, 5),
            "banked_mean": float(json.loads((HERE / DRAWS[d]["banked_arena"]).read_text())["mean"]),
            "worst_abs_delta": round(max(v["abs_delta"] for v in per.values()), 6),
            "pass": all(v["pass"] for v in per.values()),
        }
        print(f"  gate(a) draw{d}: mean {mean:.5f}  worst |delta| "
              f"{gate_a['per_draw'][f'draw{d}']['worst_abs_delta']:.2e}  "
              f"{'PASS' if gate_a['per_draw'][f'draw{d}']['pass'] else 'FAIL'}", flush=True)
    gate_a["worst_abs_delta_both_draws"] = round(worst_a, 6)
    gate_a["pass"] = all(v["pass"] for v in gate_a["per_draw"].values())

    # (b) the lexical scorer alone == the w=1 limit of the same fusion path.
    gate_b = {"tol": GATE_B_TOL, "per_draw": {}}
    for d in DRAWS:
        per = {}
        for sub, ref in GATE_B_REF.items():
            agg = arena_item_scores(frames[d][sub], 1.0)
            auc = float(roc_auc_score(agg["label"].to_numpy(), agg["fused"].to_numpy()))
            delta = abs(auc - ref)
            per[sub] = {"reproduced": round(auc, 6), "banked": ref,
                        "abs_delta": round(delta, 6), "pass": bool(delta <= GATE_B_TOL)}
        gate_b["per_draw"][f"draw{d}"] = {"per_subset": per,
                                          "pass": all(v["pass"] for v in per.values())}
        print(f"  gate(b) draw{d}: " + "  ".join(
            f"{s} {v['reproduced']:.5f} (banked {v['banked']:.5f}, |d| {v['abs_delta']:.2e})"
            for s, v in per.items()) +
            f"  {'PASS' if gate_b['per_draw'][f'draw{d}']['pass'] else 'FAIL'}", flush=True)
    gate_b["pass"] = all(v["pass"] for v in gate_b["per_draw"].values())
    gate_b["note"] = ("token containment is checkpoint-independent, so both draws read the "
                      "same value; both are checked because both parquets are used")
    return gate_a, gate_b


# --- stage 2: in-domain per-pair cache (GPU0) ------------------------------------------


def containment_of(H161, sent, win):
    """R19-H161's frozen definition: |content(sent) & content(win)| / |content(sent)|,
    content = lowercased [a-z0-9]+ matches minus the fixed stopword list."""
    s = H161.content_set(H161.raw_tokens(sent))
    if not s:
        return 0.0
    return len(s & H161.content_set(H161.raw_tokens(win))) / len(s)


def build_indomain(draw):
    out = HERE / DRAWS[draw]["indomain"]
    if out.exists():
        print(f"  SKIP (on disk): {out.name}", flush=True)
        return
    import torch

    ARM = _mod("g1arm", "R16-H142_G1_arm.py")
    H161 = _mod("h161dump", "R19-H161_dump.py")
    H108 = _mod("h108", "R10-H108_lane.py")
    M59, M60 = H108.M59, H108.M60
    chunk_max = M59.CFG.chunk_max_chars

    print(f"  draw{draw}: loading {DRAWS[draw]['ckpt']}  "
          f"GPU {torch.cuda.get_device_name(0)}", flush=True)
    model, tok = ARM.load_run(ROOT / "models" / DRAWS[draw]["ckpt"])

    evals = []
    cl, ck, y = H108.gold_full()
    evals.append(("gold_full", cl, ck, y))
    for lg in LANGS:
        cl, ctx, y = M60.load_translated(lg)
        evals.append((lg, cl, [M59.top_chunks(c, M59.CFG.semantic_top_k) for c in ctx], y))

    parts = []
    for name, claims, chunk_lists, y in evals:
        flat_s, flat_w, si = [], [], []
        for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
            for k in ks:
                flat_s.append(c)
                flat_w.append(k[:chunk_max])
                si.append(i)
        si = np.asarray(si, dtype=np.int64)
        logits = H161.score_pairs(model, tok, flat_s, flat_w, si, len(claims),
                                  tag=f"draw{draw}/{name}")
        cont = np.fromiter(
            (containment_of(H161, s, w) for s, w in zip(flat_s, flat_w, strict=True)),
            dtype=np.float32, count=len(flat_s))
        parts.append(pl.DataFrame({
            "eval": pl.Series([name] * len(flat_s), dtype=pl.Utf8),
            "item_id": pl.Series(si, dtype=pl.Int32),
            "label": pl.Series(np.asarray(y, dtype=np.int8)[si], dtype=pl.Int8),
            "logit": pl.Series(logits, dtype=pl.Float32),
            "containment": pl.Series(cont, dtype=pl.Float32),
        }))
        print(f"    draw{draw} {name}: {len(flat_s)} pairs over {len(claims)} items", flush=True)

    del model
    torch.cuda.empty_cache()
    pl.concat(parts).write_parquet(out)
    print(f"  wrote {out.name}", flush=True)


# --- stage 3: w selection on gold_full ONLY (CPU) ---------------------------------------


def indomain_auroc(df, w):
    """Fuse at the pair grain, then MAX over the claim's chunks - the in-domain
    serving aggregation every campaign arm is read under."""
    fused = (1.0 - w) * sigmoid(df["logit"].to_numpy()) + w * df["containment"].to_numpy()
    agg = (df.with_columns(pl.Series("fused", fused))
             .group_by("item_id")
             .agg(pl.col("fused").max(), pl.col("label").first())
             .sort("item_id"))
    return float(roc_auc_score(agg["label"].to_numpy(), agg["fused"].to_numpy()))


def select_w(indomain):
    print(f"\n--- stage 3: w selection on gold_full ONLY  {time.strftime('%F %T')} ---",
          flush=True)
    sweep = []
    for w in (0.0,) + W_GRID:
        row = {"w": w, "per_draw": {}}
        for d in DRAWS:
            df = indomain[d]
            gf = indomain_auroc(df.filter(pl.col("eval") == "gold_full"), w)
            per_lang = {lg: indomain_auroc(df.filter(pl.col("eval") == lg), w) for lg in LANGS}
            row["per_draw"][f"draw{d}"] = {
                "gold_full": round(gf, 5),
                "nonen": round(float(np.mean(list(per_lang.values()))), 5),
                "per_lang": {k: round(v, 4) for k, v in per_lang.items()},
            }
        row["gold_full_mean_both_draws"] = round(
            float(np.mean([row["per_draw"][f"draw{d}"]["gold_full"] for d in DRAWS])), 5)
        row["holds_pass"] = bool(all(
            row["per_draw"][f"draw{d}"]["gold_full"] >= HOLD_GOLD_FULL
            and row["per_draw"][f"draw{d}"]["nonen"] >= HOLD_NONEN for d in DRAWS))
        sweep.append(row)
        print(f"  w={w:.2f}  gold_full d1 {row['per_draw']['draw1']['gold_full']:.5f} "
              f"d2 {row['per_draw']['draw2']['gold_full']:.5f} "
              f"(mean {row['gold_full_mean_both_draws']:.5f})  "
              f"nonEN d1 {row['per_draw']['draw1']['nonen']:.5f} "
              f"d2 {row['per_draw']['draw2']['nonen']:.5f}  "
              f"holds {'PASS' if row['holds_pass'] else 'FAIL'}", flush=True)

    grid = [r for r in sweep if r["w"] > 0.0]
    unconstrained = max(grid, key=lambda r: r["gold_full_mean_both_draws"])
    eligible = [r for r in grid if r["holds_pass"]]
    chosen = max(eligible, key=lambda r: r["gold_full_mean_both_draws"]) if eligible else None
    return sweep, unconstrained, chosen


# --- driver ------------------------------------------------------------------------------


def bank(payload, tail=""):
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\n  results -> {OUT_JSON}", flush=True)
    if tail:
        print(tail, flush=True)


def main():
    print(f"=== R20-H178 LEXICAL-LATE-FUSION  {time.strftime('%F %T')} ===", flush=True)
    print("zero-training paired read on models/R18-H150-arm-draw{1,2}; "
          "SYSTEM ledger (amendment A1)", flush=True)

    frames = {d: load_arena_frames(d) for d in DRAWS}
    gate_a, gate_b = run_gates(frames)
    payload = {
        "arm": "R20-H178 LEXICAL-LATE-FUSION",
        "scoping": ("amendment A1 - SYSTEM read (cross-encoder + lexical tier). "
                    "`system_mean` never substitutes for, updates or is averaged into "
                    "the model arena mean."),
        "checkpoints": {f"draw{d}": str(ROOT / "models" / DRAWS[d]["ckpt"]) for d in DRAWS},
        "model_score_source": {
            f"draw{d}": DRAWS[d]["pairs"] for d in DRAWS},
        "fusion": "(1-w)*sigmoid(logit) + w*tok_containment at the (sentence, window) pair grain",
        "aggregation": "arena: max over windows per sentence, then MIN over sentences; "
                       "in-domain: max over the claim's chunks",
        "sanity_gate_a_w0_reproduces_banked_arena": gate_a,
        "sanity_gate_b_lexical_reproduces_banked_containment": gate_b,
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    }
    if not (gate_a["pass"] and gate_b["pass"]):
        payload["outcome"] = "ABORT - sanity gate missed; no arena read taken"
        bank(payload, "=== R20-H178 ABORT: SANITY GATE MISSED ===")
        sys.exit(1)
    print("  BOTH GATES PASS", flush=True)
    print("=== H178 GATES DONE ===", flush=True)

    print(f"\n--- stage 2: in-domain per-pair cache (GPU0)  {time.strftime('%F %T')} ---",
          flush=True)
    for d in DRAWS:
        build_indomain(d)
    print("=== H178 INDOMAIN DONE ===", flush=True)
    indomain = {d: pl.read_parquet(HERE / DRAWS[d]["indomain"]) for d in DRAWS}

    sweep, unconstrained, chosen = select_w(indomain)
    payload["w_grid"] = list(W_GRID)
    payload["holds"] = {"gold_full_min": HOLD_GOLD_FULL, "nonen_min": HOLD_NONEN,
                        "required_on": "both draws at the selected w"}
    payload["gold_full_sweep"] = sweep
    payload["w_unconstrained_argmax"] = unconstrained["w"]
    if chosen is None:
        payload["w_selected"] = None
        payload["outcome"] = ("STOP - no w in the grid holds gold_full >= 0.84 and "
                              "non-EN >= 0.82 on both draws; no arena read taken")
        bank(payload, "=== R20-H178 STOP: NO w HOLDS ===")
        sys.exit(2)

    w = chosen["w"]
    payload["w_selected"] = w
    payload["w_selection_rule"] = (
        "argmax of the two-draw mean gold_full AUROC over the registered grid, restricted "
        "to the w that hold gold_full >= 0.84 and non-EN >= 0.82 on BOTH draws. The arena "
        "was not consulted at any w. Unconstrained argmax recorded alongside.")
    payload["holds_at_w_selected"] = {
        f"draw{d}": {"gold_full": chosen["per_draw"][f"draw{d}"]["gold_full"],
                     "nonen": chosen["per_draw"][f"draw{d}"]["nonen"],
                     "per_lang": chosen["per_draw"][f"draw{d}"]["per_lang"],
                     "pass": bool(chosen["per_draw"][f"draw{d}"]["gold_full"] >= HOLD_GOLD_FULL
                                  and chosen["per_draw"][f"draw{d}"]["nonen"] >= HOLD_NONEN)}
        for d in DRAWS}
    print(f"\n  w_selected = {w:.2f} (unconstrained argmax {unconstrained['w']:.2f})", flush=True)

    print(f"\n--- stage 4: blind arena read at w={w:.2f}  {time.strftime('%F %T')} ---",
          flush=True)
    per_subset, plain, fused_mean, deltas = {}, {}, {}, {}
    for d in DRAWS:
        rows0, mean0 = arena_read(frames[d], 0.0)
        rowsw, meanw = arena_read(frames[d], w)
        per_subset[f"draw{d}"] = {
            s: {"plain_w0": round(rows0[s], 5), "fused": round(rowsw[s], 5),
                "delta": round(rowsw[s] - rows0[s], 5)} for s in rows0}
        plain[f"draw{d}"] = round(mean0, 5)
        fused_mean[f"draw{d}"] = round(meanw, 5)
        deltas[f"draw{d}"] = round(meanw - mean0, 5)
        worst = min(per_subset[f"draw{d}"].items(), key=lambda kv: kv[1]["delta"])
        print(f"  draw{d}: plain {mean0:.5f} -> fused {meanw:.5f}  "
              f"delta {meanw - mean0:+.5f}  worst subset {worst[0]} {worst[1]['delta']:+.5f}",
              flush=True)
        for s in sorted(rows0):
            print(f"    {s:12s} {rows0[s]:.4f} -> {rowsw[s]:.4f}  "
                  f"{rowsw[s] - rows0[s]:+.4f}", flush=True)

    worst_delta = {f"draw{d}": min(per_subset[f"draw{d}"].items(),
                                   key=lambda kv: kv[1]["delta"]) for d in DRAWS}
    payload["per_subset"] = per_subset
    payload["plain_mean"] = plain
    payload["system_mean"] = dict(fused_mean)
    payload["system_mean"]["pair_mean"] = round(
        float(np.mean([fused_mean[f"draw{d}"] for d in DRAWS])), 5)
    payload["plain_pair_mean"] = round(float(np.mean([plain[f"draw{d}"] for d in DRAWS])), 5)
    payload["delta"] = deltas
    payload["delta_pair_mean"] = round(
        payload["system_mean"]["pair_mean"] - payload["plain_pair_mean"], 5)
    payload["worst_subset_delta"] = {k: {"subset": v[0], "delta": v[1]["delta"]}
                                     for k, v in worst_delta.items()}

    d1, d2 = deltas["draw1"], deltas["draw2"]
    wmin = min(v[1]["delta"] for v in worst_delta.values())
    if d1 >= 0.005 and d2 >= 0.005 and wmin >= -0.01:
        verdict = "PASS"
    elif d1 <= 0 and d2 <= 0:
        verdict = "KILL"
    else:
        verdict = "EXPLORATORY"
    payload["verdict"] = verdict
    payload["verdict_basis"] = {
        "delta_draw1": d1, "delta_draw2": d2,
        "worst_subset_delta_either_draw": round(wmin, 5),
        "pass_rule": "delta >= +0.005 on BOTH draws AND no subset below -0.01 on either",
        "kill_rule": "delta <= 0 on both draws",
    }
    bank(payload, f"\n  VERDICT {verdict}   delta d1 {d1:+.5f}  d2 {d2:+.5f}  "
                  f"worst subset {wmin:+.5f}\n=== H178 COMPLETE ===")


if __name__ == "__main__":
    main()

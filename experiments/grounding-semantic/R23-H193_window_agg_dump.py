"""R23-H193 / precursor P-D - dump per-(sentence, window) logits, then measure
window-level aggregation headroom. MEASUREMENT ONLY, no training, no promotion.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R23-P-D WINDOW-LEVEL AGGREGATION HEADROOM", threshold fixed before any read.

R9's P-B and P-C closed aggregation over SENTENCES. Both ran on a dump whose
records carry per-sentence scores in which the max over windows had already been
taken, so the aggregator INSIDE a sentence - across its windows - was never
measured. R16-H142_G1_arm.score_sets fixes it as scatter_reduce(reduce="amax"),
untouched since R8-H101. R12's ceiling ladder says that choice is where all the
ceiling loss lives: strict single-window support 0.7560 against lenient 0.9444.

Stage `dump`   re-runs the windowed arena read on the banked flagship draw 1 and
               writes EVERY per-pair logit plus its sentence and response owner,
               instead of collapsing to the per-sentence max
Stage `read`   applies the registered fixed aggregators to that dump and reports
               the blind arena mean for each, against the pre-registered
               max + 0.010 threshold

FIDELITY CONTROL, hard: the `max` aggregator must reproduce the checkpoint's
banked per-subset AUCs to 1e-4 before any other aggregator's number counts.
R22-H188 measured a cross-card re-score drifting past that tolerance by exactly
one rank flip, so this runs on GPU1 - the card that produced draw 1's banked read.

PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set: expandable_segments kills
.to("cuda") under WSL2 on this box.

Run (detached):
    GPU=1 nohup setsid bash -c 'cd <repo> && \
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
      uv run python experiments/grounding-semantic/R23-H193_window_agg_dump.py --stage dump && \
      uv run python experiments/grounding-semantic/R23-H193_window_agg_dump.py --stage read' \
      > logs/R23-H193_window_agg.log 2>&1 &
"""

import argparse
import importlib.util
import json
import os
import pathlib
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

HERE = pathlib.Path(__file__).parent
DUMP = HERE / "R23-H193_window_agg_dump.npz"
OUT = HERE / "R23-H193_window_agg_result.json"

CKPT = "R18-H150-arm-draw1"
BANKED = HERE / "R18-H150_arm_draw1_windowed_result.json"
FIDELITY_TOL = 1e-4
THRESHOLD = 0.010          # pre-registered, verbatim from P-C
READ_GPU = "1"             # the card that produced draw 1's banked read


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stage_dump():
    import torch

    ARM = _mod("g1arm", "R16-H142_G1_arm.py")
    H92 = _mod("h92", "R8-H92_decomposed_arena.py")
    ARENA = H92.ARENA
    H150 = _mod("h150run", "R18-H150_arm_run.py")
    H150.rebind(ARM)

    ckpt = pathlib.Path(__file__).parents[2] / "models" / CKPT
    print(f"=== P-D dump  {CKPT}  GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}  "
          f"{time.strftime('%F %T')} ===", flush=True)
    model, tok = ARM.load_run(str(ckpt))
    subs = ARENA.load_subsets()

    out = {}
    for sub, (claims, chunks, y) in sorted(subs.items()):
        flat_s, flat_w, set_index, owner = [], [], [], []
        for i, (c, ks) in enumerate(zip(claims, chunks, strict=True)):
            wlist = [w for k in ks for w in ARM.windows(k)]
            for s in H92.sentences(c):
                sid = len(owner)
                owner.append(i)
                for w in wlist:
                    flat_s.append(s)
                    flat_w.append(w)
                    set_index.append(sid)

        n = len(flat_s)
        cls = torch.zeros(n, model.trunk.config.hidden_size, dtype=torch.float32)
        t0 = time.time()
        with torch.inference_mode():
            for i in range(0, n, 64):
                enc = tok(flat_s[i:i + 64], flat_w[i:i + 64], return_tensors="pt",
                          padding=True, truncation=True, max_length=ARM.MAX_LEN)
                enc = {k: v.cuda() for k, v in enc.items()}
                cls[i:i + 64] = model.encode(enc).float().cpu()
                if (i // 64) % 400 == 0 and i:
                    print(f"    {sub} {i}/{n} ({i / max(time.time() - t0, 1e-9):.0f} pairs/s)",
                          flush=True)
            si = torch.as_tensor(set_index, dtype=torch.long).cuda()
            ctx = model.pool_ctx(cls.cuda(), si, len(owner))
            logits = np.empty(n, dtype=np.float32)
            for a in range(0, n, 200_000):
                b = min(a + 200_000, n)
                logits[a:b] = (model.pair_logits(cls[a:b].cuda(), ctx[si[a:b]])
                               .float().cpu().numpy())

        out[f"{sub}__logit"] = logits
        out[f"{sub}__sent"] = np.asarray(set_index, dtype=np.int32)
        out[f"{sub}__owner"] = np.asarray(owner, dtype=np.int32)
        out[f"{sub}__y"] = np.asarray(y, dtype=np.int8)
        print(f"  {sub}: {n} pairs over {len(owner)} sentences, {len(y)} responses",
              flush=True)

    np.savez_compressed(DUMP, **out)
    print(f"dump -> {DUMP} ({DUMP.stat().st_size / 1e6:.1f} MB)  {time.strftime('%F %T')}",
          flush=True)


# ---- window-level aggregators: sentence score from its window logits ------
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def agg_max(v):
    return v.max()


def agg_mean(v):
    return v.mean()


def _topk_mean(v, k):
    if v.size <= k:
        return v.mean()
    return np.partition(v, -k)[-k:].mean()


def agg_top2(v):
    return _topk_mean(v, 2)


def agg_top3(v):
    return _topk_mean(v, 3)


def agg_noisy_or(v):
    """1 - prod(1 - p): fires when support is SPREAD over several windows -
    the lenient-oracle shape, in a fixed subset-blind form."""
    p = np.clip(_sig(v), 1e-6, 1 - 1e-6)
    return float(-np.sum(np.log1p(-p)))


def _lse(v, tau):
    m = v.max()
    return float(m + tau * np.log(np.exp((v - m) / tau).sum()))


def agg_lse05(v):
    return _lse(v, 0.5)


def agg_lse1(v):
    return _lse(v, 1.0)


def agg_lse2(v):
    return _lse(v, 2.0)


def agg_count_pos(v):
    return float((v > 0).sum())


AGGREGATORS = {"max": agg_max, "mean": agg_mean, "top2_mean": agg_top2,
               "top3_mean": agg_top3, "noisy_or": agg_noisy_or,
               "logsumexp_tau0.5": agg_lse05, "logsumexp_tau1": agg_lse1,
               "logsumexp_tau2": agg_lse2, "count_windows_positive": agg_count_pos}


def stage_read():
    M59 = _mod("h92", "R8-H92_decomposed_arena.py").ARENA.M59
    d = np.load(DUMP)
    banked = json.loads(BANKED.read_text())["per_subset"]
    subsets = sorted({k.split("__")[0] for k in d.files})

    per_agg = {}
    for name, fn in AGGREGATORS.items():
        per_sub = {}
        for sub in subsets:
            lg = d[f"{sub}__logit"]
            sent = d[f"{sub}__sent"]
            owner = d[f"{sub}__owner"]
            y = d[f"{sub}__y"]
            # sentence score = aggregator over that sentence's window logits
            order = np.argsort(sent, kind="stable")
            lg_s, sent_s = lg[order], sent[order]
            bounds = np.searchsorted(sent_s, np.arange(len(owner) + 1))
            s_sent = np.array([fn(lg_s[bounds[i]:bounds[i + 1]]) for i in range(len(owner))])
            # response score = MIN over its sentences, unchanged (P-C closed this level)
            resp = np.array([s_sent[owner == i].min() for i in range(len(y))])
            per_sub[sub] = round(float(M59.auc_and_f1(y, resp)[0]), 5)
        per_agg[name] = {"per_subset": per_sub,
                         "mean": round(float(np.mean(list(per_sub.values()))), 5)}

    # FIDELITY: max must reproduce the banked read
    fid = {}
    worst = 0.0
    for sub in subsets:
        b = banked[sub]["auc"] if isinstance(banked[sub], dict) else banked[sub]
        got = per_agg["max"]["per_subset"][sub]
        delta = abs(got - b)
        worst = max(worst, delta)
        fid[sub] = {"banked": b, "recomputed": got, "abs_delta": round(delta, 8),
                    "pass": bool(delta <= FIDELITY_TOL)}
    ok = all(v["pass"] for v in fid.values())

    base = per_agg["max"]["mean"]
    ladder = sorted(((n, v["mean"], v["mean"] - base) for n, v in per_agg.items()),
                    key=lambda r: -r[1])
    best = ladder[0]
    fired = bool(ok and best[0] != "max" and best[2] >= THRESHOLD)

    res = {
        "arm": "R23-H193 / precursor P-D - window-level aggregation headroom",
        "licence": ("MEASUREMENT ONLY. A fired threshold licenses a REGISTERED arm with its own "
                    "bars and a NON-arena selection surface; it promotes nothing and changes no "
                    "serving path. Subset-level-best selection is non-registrable, stated in "
                    "advance, per P-C's own refusal of its +0.0045 subset-best ceiling."),
        "checkpoint": CKPT, "read_gpu": READ_GPU,
        "threshold": THRESHOLD,
        "threshold_provenance": "P-C's bar, reused verbatim so the two precursors compare",
        "fidelity_control": {"tolerance": FIDELITY_TOL, "worst_abs_delta": round(worst, 8),
                             "pass": ok, "per_subset": fid},
        "baseline_max_mean": base,
        "ladder": [{"aggregator": n, "mean": m, "delta_vs_max": round(dl, 5)}
                   for n, m, dl in ladder],
        "per_aggregator": per_agg,
        "best_non_max": next((n, m, dl) for n, m, dl in ladder if n != "max"),
        "verdict": ("FIRED" if fired else "NOT FIRED"),
        "reading": None,
    }
    if not ok:
        res["verdict"] = "VOID"
        res["reading"] = ("fidelity control FAILED - the max aggregator did not reproduce the "
                          "banked read, so no aggregator's number may be quoted")
    else:
        res["reading"] = (
            f"best non-max aggregator {res['best_non_max'][0]} reads {res['best_non_max'][1]:.5f}, "
            f"{res['best_non_max'][2]:+.5f} against max. Threshold {THRESHOLD} "
            f"{'FIRED' if fired else 'NOT FIRED'}.")
    OUT.write_text(json.dumps(res, indent=2))

    print(f"\nfidelity: {'PASS' if ok else 'FAIL'} (worst |d| {worst:.2e}, tol {FIDELITY_TOL})")
    print(f"\n{'aggregator':<24}{'blind mean':>12}{'vs max':>10}")
    for n, m, dl in ladder:
        print(f"{n:<24}{m:>12.5f}{dl:>+10.5f}")
    print(f"\nVERDICT: {res['verdict']}  (threshold max + {THRESHOLD})")
    print(f"written -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("dump", "read"))
    a = ap.parse_args()
    if a.stage == "dump":
        if DUMP.exists() and DUMP.stat().st_size > 0:
            print(f"SKIP (on disk: {DUMP.name})", flush=True)
            return
        stage_dump()
    else:
        stage_read()


if __name__ == "__main__":
    main()

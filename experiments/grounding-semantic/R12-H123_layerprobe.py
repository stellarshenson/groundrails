"""R12-H123 layer probe - per-layer grounding AUC and DANN-group accuracy, both H105 draws.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 12);
full record in R12_synthesis_full_field.md (R12-H123 ADVERSARY-DECOUPLED
LAYER-MIX). The task head and the domain discriminator read the SAME vector
(CLS of the final layer), so every register direction the adversary erases is
erased from the task head's only input. The arm would give the task head a
learned scalar mix over all 23 layer CLS vectors; this probe tests the arm's own
premise before any build.

  LICENSE (both draws)  max over l < 22 of AUC(l) >= AUC(22) + 0.005
                        AND group-accuracy(22) <= mid-stack max group-acc - 0.05
  otherwise             KILL

Binding amendment D: the probe is an IN-DOMAIN necessary condition only. It
licenses the build; it predicts nothing about blind transfer. Amendment A: the
baseline recipe carries 12 DANN groups (chance 0.083), so "erasure at the top"
is what the probe must establish, not a premise.

Layer convention recorded in the result JSON: `layer l` is
`hidden_states[l]`, so l = 0 is the embedding output and l = 1..22 are the 22
transformer layers; l = 22 is the final layer the task head actually reads.

Data: a fixed-seed 20,000-row slice of `public_train()` - training distribution
only. No arena, no gold. The trainer has no eval/holdout split of its own (it
trains one epoch over the full mix), so the slice is a fixed-seed sample and the
same row indices and the same probe train/test split are used for both draws.

Stages are idempotent: cached CLS tensors are reused when present.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/R12-H123_layerprobe.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import importlib.util
import json
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent

DRAWS = {
    "draw1": str(ROOT / json.loads((HERE / "R9-H105_windowed_result.json").read_text())["model"]),
    "draw2": str(ROOT / json.loads((HERE / "R9-H105_draw2_windowed_result.json").read_text())["model"]),
}
RESULT = HERE / "R12-H123_layerprobe_result.json"
CACHE = ROOT / "tmp"
CACHE.mkdir(exist_ok=True)

N_ROWS = 20000
MAX_LEN = 512
BATCH = 32
SEED = 20260808
TEST_FRAC = 0.3
N_LAYERS = 23  # hidden_states: embedding output + 22 transformer layers


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def slice_rows():
    """Fixed-seed 20k slice of the clean public mix, identical for both draws."""
    H105 = _mod("h105", "R9-H105_clean_mix.py")
    claims, chunks, y, tags = H105.public_train()
    tag_names = sorted(set(tags))
    tag_to_idx = {t: i for i, t in enumerate(tag_names)}
    g = np.array([tag_to_idx[t] for t in tags], dtype=np.int64)
    idx = np.random.default_rng(SEED).permutation(len(y))[:N_ROWS]
    return (
        [claims[i] for i in idx],
        [chunks[i] for i in idx],
        y[idx].astype(np.int64),
        g[idx],
        tag_names,
    )


def encode(model_path, claims, chunks, cache):
    """CLS vector of every hidden state for each row: (N_ROWS, 23, hidden)."""
    if cache.exists():
        arr = np.load(cache)["h"]
        print(f"  cached {cache.name} {arr.shape}", flush=True)
        return arr
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    trunk = AutoModel.from_pretrained(str(pathlib.Path(model_path) / "trunk")).cuda().eval()
    trunk.config.reference_compile = False
    d = trunk.config.hidden_size
    out = np.zeros((len(claims), N_LAYERS, d), dtype=np.float16)
    t0 = time.time()
    with torch.inference_mode():
        for j in range(0, len(claims), BATCH):
            enc = tok(claims[j : j + BATCH], chunks[j : j + BATCH], return_tensors="pt",
                      padding=True, truncation=True, max_length=MAX_LEN)
            enc = {k: v.cuda() for k, v in enc.items()}
            hs = trunk(**enc, output_hidden_states=True).hidden_states
            assert len(hs) == N_LAYERS, f"expected {N_LAYERS} hidden states, got {len(hs)}"
            out[j : j + BATCH] = torch.stack([h[:, 0] for h in hs], dim=1).half().cpu().numpy()
            if j % (BATCH * 100) == 0:
                print(f"    {j}/{len(claims)} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(cache, h=out)
    del trunk
    torch.cuda.empty_cache()
    print(f"  encoded {model_path} in {time.time() - t0:.0f}s -> {cache.name}", flush=True)
    return out


def probe(H, y, g, tr, te):
    """Per-layer logistic probes: grounding AUC and DANN-group accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    auc, gacc = [], []
    for l in range(N_LAYERS):
        X = H[:, l, :].astype(np.float32)
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        m = LogisticRegression(max_iter=500, C=1.0, n_jobs=-1).fit(Xtr, y[tr])
        auc.append(round(float(roc_auc_score(y[te], m.predict_proba(Xte)[:, 1])), 4))
        mg = LogisticRegression(max_iter=500, C=1.0, n_jobs=-1).fit(Xtr, g[tr])
        gacc.append(round(float((mg.predict(Xte) == g[te]).mean()), 4))
        print(f"    layer {l:>2}  auc {auc[-1]:.4f}  group-acc {gacc[-1]:.4f}", flush=True)
    return auc, gacc


def main():
    print(f"draws: {DRAWS}", flush=True)
    claims, chunks, y, g, tag_names = slice_rows()
    n_groups = len(tag_names)
    print(f"slice: {len(y)} rows, grounded rate {y.mean():.3f}, {n_groups} groups "
          f"(chance {1 / n_groups:.4f})", flush=True)

    rng = np.random.default_rng(SEED + 1)
    perm = rng.permutation(len(y))
    n_te = int(TEST_FRAC * len(y))
    te, tr = perm[:n_te], perm[n_te:]

    per_draw = {}
    for name, path in DRAWS.items():
        print(f"\n== {name}: {path}", flush=True)
        H = encode(path, claims, chunks, CACHE / f"R12-H123_cls_{name}.npz")
        auc, gacc = probe(H, y, g, tr, te)
        del H

        auc_top = auc[22]
        best_below = max(auc[:22])
        best_below_l = int(np.argmax(auc[:22]))
        # mid-stack = every layer except the final one the task head reads
        mid_gacc_max = max(gacc[:22])
        mid_gacc_max_l = int(np.argmax(gacc[:22]))
        cond_auc = best_below >= auc_top + 0.005
        cond_grp = gacc[22] <= mid_gacc_max - 0.05
        per_draw[name] = {
            "model": path,
            "auc_per_layer": auc,
            "group_acc_per_layer": gacc,
            "auc_final_layer22": auc_top,
            "max_auc_below_22": best_below,
            "argmax_auc_layer_below_22": best_below_l,
            "auc_margin": round(best_below - auc_top, 4),
            "group_acc_layer22": gacc[22],
            "max_group_acc_below_22": mid_gacc_max,
            "argmax_group_acc_layer": mid_gacc_max_l,
            "group_acc_drop_at_22": round(mid_gacc_max - gacc[22], 4),
            "cond_auc_margin_ge_0.005": bool(cond_auc),
            "cond_group_acc_drop_ge_0.05": bool(cond_grp),
            "draw_satisfies": bool(cond_auc and cond_grp),
        }

    both = all(v["draw_satisfies"] for v in per_draw.values())
    verdict = "LICENSE" if both else "KILL"

    res = {
        "gate": "R12-H123 per-layer probe on both H105 draws",
        "layer_convention": "layer l = hidden_states[l]; l=0 embedding output, l=1..22 "
                            "transformer layers, l=22 the final layer the task head reads",
        "n_rows": int(len(y)), "test_frac": TEST_FRAC, "seed": SEED,
        "data": "fixed-seed slice of R9-H105_clean_mix.public_train(); no arena, no gold",
        "dann_groups": tag_names, "group_chance": round(1 / n_groups, 4),
        "per_draw": per_draw,
        "bar": "LICENSE only if BOTH draws satisfy max AUC(l<22) >= AUC(22)+0.005 AND "
               "group-acc(22) <= mid-stack max group-acc - 0.05",
        "verdict": verdict,
        "scope_note": "in-domain necessary condition only (binding amendment D); licenses the "
                      "build, predicts nothing about blind transfer",
    }
    RESULT.write_text(json.dumps(res, indent=2))

    print("\n" + "=" * 92)
    print("R12-H123 LAYER PROBE")
    print("=" * 92)
    for name, v in per_draw.items():
        print(f"  {name}: AUC(22) {v['auc_final_layer22']:.4f}  best AUC below 22 "
              f"{v['max_auc_below_22']:.4f} @ layer {v['argmax_auc_layer_below_22']}  "
              f"margin {v['auc_margin']:+.4f} (need >= +0.005)")
        print(f"         group-acc(22) {v['group_acc_layer22']:.4f}  mid-stack max "
              f"{v['max_group_acc_below_22']:.4f} @ layer {v['argmax_group_acc_layer']}  "
              f"drop {v['group_acc_drop_at_22']:+.4f} (need >= 0.05)")
        print(f"         satisfies: {v['draw_satisfies']}")
    print(f"\n  VERDICT: {verdict}")
    print(f"  -> {RESULT}")


if __name__ == "__main__":
    main()

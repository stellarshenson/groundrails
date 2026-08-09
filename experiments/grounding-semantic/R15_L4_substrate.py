"""R15 LENS-4 - is the DERIVATION-VERIFICATION substrate capacity-dependent?

P4 measured that the shipped trunk linearly separates "which number is larger"
at 99.7% while the task head reads AUROC 0.5230 - the substrate is there, the
head does not use it.  P4 did NOT probe the substrate for the thing the R14-A4
lane actually has to install: "does the asserted value equal the sum of the two
named cells".

This probe reads that target linearly off three frozen trunks:

  base_pretrained   jhu-clsp/mmBERT-base      (307M nominal / 110.3M stack)
  small_pretrained  jhu-clsp/mmBERT-small     (140.9M nominal /  42.2M stack)
  h105_finetuned    R9-H105-mmbert-dann-clean (the shipped trunk)

Targets, all on synthetic two-row tables (no corpus, no arena, no gold):
  T1 magnitude   - log10 of a single stated value            (sanity)
  T2 comparison  - Alpha > Beta                              (P4 instrument B)
  T3 derivation  - the asserted V equals X + Y               (the lane's target)

Controls: permuted labels; a random projection of the larger trunk's CLS down to
the smaller trunk's width, so the two sizes are compared at matched probe
dimensionality rather than at matched architecture.

Frozen weights, no fine-tuning, no arena, no gold.
Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run python <this>
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
RESULT = HERE / "R15_gate_L4_substrate.json"

SEED = 20260811
BATCH = 64
MAX_LEN = 256
N = 6000

H105 = ROOT / "models" / "R9-H105-mmbert-dann-clean"
MODELS = {
    "base_pretrained": ("jhu-clsp/mmBERT-base", "jhu-clsp/mmBERT-base"),
    "small_pretrained": ("jhu-clsp/mmBERT-small", "jhu-clsp/mmBERT-small"),
    "h105_finetuned": (str(H105 / "trunk"), str(H105)),
}


def cls_of(path, claims, evs, tokpath=None):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokpath or path)
    trunk = AutoModel.from_pretrained(path).cuda().eval()
    if hasattr(trunk.config, "reference_compile"):
        trunk.config.reference_compile = False
    h = trunk.config.hidden_size
    C = np.zeros((len(claims), h), dtype=np.float32)
    with torch.inference_mode():
        for j in range(0, len(claims), BATCH):
            enc = tok(claims[j : j + BATCH], evs[j : j + BATCH], return_tensors="pt",
                      padding=True, truncation=True, max_length=MAX_LEN)
            enc = {k: v.cuda() for k, v in enc.items()}
            C[j : j + BATCH] = trunk(**enc).last_hidden_state[:, 0].float().cpu().numpy()
    del trunk
    torch.cuda.empty_cache()
    return C, h


def ridge_fit(X, y, lam=1.0):
    X = np.c_[X, np.ones(len(X))]
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def ridge_r2(Xtr, ytr, Xte, yte, lam=1.0):
    w = ridge_fit(Xtr, ytr, lam)
    p = np.c_[Xte, np.ones(len(Xte))] @ w
    return float(1 - ((yte - p) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())


def auroc(pos, neg):
    from scipy.stats import rankdata

    r = rankdata(np.r_[pos, neg])
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ridge_clf(Xtr, ytr, Xte, yte, lam=1.0):
    w = ridge_fit(Xtr, ytr, lam)
    p = np.c_[Xte, np.ones(len(Xte))] @ w
    acc = float(((p > 0.5).astype(float) == yte).mean())
    return acc, auroc(p[yte == 1], p[yte == 0])


def build(rng):
    """Three probe sets, all over the same synthetic two-row table register."""
    # T1 magnitude
    v1 = rng.integers(1, 1000, N)
    t1 = ([f"The value is {v}." for v in v1],
          ["The table records a single measured quantity." for _ in v1],
          np.log10(v1.astype(float)))

    # T2 comparison - operands printed in the evidence, claim names the order
    a = rng.integers(10, 10000, N)
    b = rng.integers(10, 10000, N)
    fix = a == b
    b[fix] = b[fix] + 1
    ev2 = [f"name | amount\nalpha | {x}\nbeta | {y}" for x, y in zip(a, b)]
    flip = rng.integers(0, 2, N).astype(bool)
    cl2 = [("The amount of alpha is greater than the amount of beta."
            if f else "The amount of beta is greater than the amount of alpha.")
           for f in flip]
    y2 = np.where(flip, (a > b), (b > a)).astype(float)
    t2 = (cl2, ev2, y2)

    # T3 derivation - the lane's own target: asserted V equals X + Y or does not
    x = rng.integers(10, 10000, N)
    y = rng.integers(10, 10000, N)
    ev3 = [f"name | amount\nalpha | {p}\nbeta | {q}" for p, q in zip(x, y)]
    correct = rng.integers(0, 2, N).astype(bool)
    # wrong values: half order-of-magnitude wrong, half arbitrary wrong-operand
    wrongkind = rng.integers(0, 2, N)
    s = x + y
    alt = np.where(wrongkind == 0, s * 10, s + rng.integers(1, 9000, N))
    v3 = np.where(correct, s, alt)
    cl3 = [f"The combined amount of alpha and beta is {v}." for v in v3]
    t3 = (cl3, ev3, correct.astype(float))

    # T4 derivation with DIGIT-LENGTH PARITY (P2-B / P3 parity rule): the wrong
    # value carries the same digit count as the correct one, so the length cue
    # that T3 leaves open is removed by construction.
    x4 = rng.integers(1000, 9000, N)
    y4 = rng.integers(1000, 9000, N)
    s4 = x4 + y4                                   # always 4-5 digits
    ok4 = rng.integers(0, 2, N).astype(bool)
    lo = 10 ** (np.floor(np.log10(s4)).astype(int))
    hi = lo * 10
    # a wrong value uniformly inside the SAME decade as the correct sum
    w4 = rng.integers(lo, hi)
    bad = w4 == s4
    w4[bad] = np.where(w4[bad] + 1 < hi[bad], w4[bad] + 1, w4[bad] - 1)
    v4 = np.where(ok4, s4, w4)
    ev4 = [f"name | amount\nalpha | {p}\nbeta | {q}" for p, q in zip(x4, y4)]
    cl4 = [f"The combined amount of alpha and beta is {v}." for v in v4]
    t4 = (cl4, ev4, ok4.astype(float))

    return {"T1_magnitude": t1, "T2_comparison": t2,
            "T3_derivation": t3, "T4_derivation_lenmatched": t4}


def main():
    rng = np.random.default_rng(SEED)
    sets = build(rng)
    ntr = int(0.7 * N)
    out = {"probe": "R15 LENS-4 derivation-verification substrate ladder",
           "seed": SEED, "n_per_target": N, "train_rows": ntr,
           "data": "synthetic two-row tables, no corpus, no arena, no gold",
           "models": {}}
    cache = {}

    # cue baseline: how far does the asserted value's DIGIT COUNT alone get you?
    import re as _re

    cue = {}
    for tname in ("T3_derivation", "T4_derivation_lenmatched"):
        cl, _, y = sets[tname]
        nd = np.array([len(_re.findall(r"\d", c.split(" is ")[-1])) for c in cl], dtype=float)
        cue[tname] = {"auroc_from_digit_count_alone": round(auroc(nd[y == 1], nd[y == 0]), 4)}
    out["digit_count_cue_baseline"] = cue
    print("cue baseline", json.dumps(cue), flush=True)

    for mname, (path, tokpath) in MODELS.items():
        res = {}
        for tname, (cl, ev, y) in sets.items():
            C, h = cls_of(path, cl, ev, tokpath)
            cache[(mname, tname)] = (C, y)
            res["hidden_size"] = h
            if tname == "T1_magnitude":
                res[tname] = {"r2": round(ridge_r2(C[:ntr], y[:ntr], C[ntr:], y[ntr:]), 4)}
            else:
                acc, au = ridge_clf(C[:ntr], y[:ntr], C[ntr:], y[ntr:])
                yperm = rng.permutation(y)
                acc_p, _ = ridge_clf(C[:ntr], yperm[:ntr], C[ntr:], yperm[ntr:])
                res[tname] = {"acc": round(acc, 4), "auroc": round(au, 4),
                              "acc_permuted_control": round(acc_p, 4),
                              "base_rate": round(float(y[ntr:].mean()), 4)}
            print(mname, tname, res.get(tname, res), flush=True)
        out["models"][mname] = res

    # dimensionality control - project the 768-dim trunks down to the small width
    small_h = out["models"]["small_pretrained"]["hidden_size"]
    ctrl = {}
    for mname in ("base_pretrained", "h105_finetuned"):
        if out["models"][mname]["hidden_size"] <= small_h:
            continue
        R = rng.normal(size=(out["models"][mname]["hidden_size"], small_h)) / np.sqrt(small_h)
        sub = {}
        for tname in ("T2_comparison", "T3_derivation", "T4_derivation_lenmatched"):
            C, y = cache[(mname, tname)]
            Cp = C @ R
            acc, au = ridge_clf(Cp[:ntr], y[:ntr], Cp[ntr:], y[ntr:])
            sub[tname] = {"acc": round(acc, 4), "auroc": round(au, 4)}
        ctrl[mname] = sub
        print("matched-dim control", mname, sub, flush=True)
    out["matched_dimensionality_control"] = {"projected_to": small_h, "results": ctrl}

    RESULT.write_text(json.dumps(out, indent=2))
    print(f"\n-> {RESULT}", flush=True)


if __name__ == "__main__":
    main()

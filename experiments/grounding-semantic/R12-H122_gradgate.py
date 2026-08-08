"""R12-H122 gradient gate - 16-way vs 9-way GRL trunk-gradient norm and direction.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 12);
full record in R12_synthesis_full_field.md (R12-H122 DANN-GROUP-COLLAPSE, binding
amendment A1: the licence is a GRADIENT measurement, not the confusion matrix).

Eight of the twelve public DANN groups are the same RAGTruth corpus in eight
languages. If half the adversarial label space encodes language rather than
register, the gradient reversal layer spends trunk budget erasing a factor the
English-only arena never sees. The gate measures that directly on the frozen
H108 draw-1 checkpoint: fit a 9-label domain head with the trunk frozen, then
over held-out batches compare the trunk gradient the 16-label head sends through
the GRL against the one the 9-label head sends.

  LICENSE   norm ratio >= 1.15x AND direction cosine <= 0.9
  KILL      norm ratio <= 1.05  OR  direction cosine >= 0.95
  otherwise AMBIGUOUS, both numbers recorded

The 16-label head is the one the draw itself trained. A 16-label head REFIT under
the 9-label head's own protocol is recorded as a diagnostic only - it adjudicates
nothing, it exists so a protocol artifact cannot be mistaken for a group-count
effect.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/R12-H122_gradgate.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import importlib.util
import json
import pathlib
import time

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent

MODEL = str(ROOT / json.loads((HERE / "R10-H108_lane_draw1_windowed_result.json").read_text())["model"])
FEATS = ROOT / "tmp" / "R12-H122_gradgate_cls.npz"
RESULT = HERE / "R12-H122_gradgate_result.json"
(ROOT / "tmp").mkdir(exist_ok=True)

LAMBDA = 0.02  # the H108 recipe's terminal adversarial weight
MAX_LEN = 512
FIT_ROWS = 40000  # rows whose CLS is cached for head fitting
HELDOUT_BATCHES = 200
BATCH = 16  # measurement batch on the 24 GB card (not the training recipe's 48)
HEAD_EPOCHS = 6
SEED = 12345

RAGTRUTH_TAGS = (
    "ragtruth_en", "ragtruth_de", "ragtruth_fr", "ragtruth_es",
    "ragtruth_it", "ragtruth_pl", "ragtruth_hu", "ragtruth_cn",
)
MERGED_TAG = "ragtruth_ALL"


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class GradReverse(torch.autograd.Function):
    """Identity forward, negated-scaled backward - the trainer's GRL verbatim."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def domain_head(n_groups, d=768, hidden=256):
    return nn.Sequential(
        nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, n_groups)
    )


def load_mix():
    """The H108 draw-1 training mix and its 16-group tag map, rebuilt verbatim."""
    H108 = _mod("h108", "R10-H108_lane.py")
    claims, chunks, y, tags = H108.public_train()
    lc, lk, ly, lt = H108.lane_train()
    claims += lc
    chunks += lk
    y = np.concatenate([y, ly]).astype("float32")
    tags += lt
    tag_names = sorted(set(tags))
    tag_to_idx = {t: i for i, t in enumerate(tag_names)}
    g16 = np.array([tag_to_idx[t] for t in tags], dtype=np.int64)

    merged = [MERGED_TAG if t in RAGTRUTH_TAGS else t for t in tags]
    m_names = sorted(set(merged))
    m_to_idx = {t: i for i, t in enumerate(m_names)}
    g9 = np.array([m_to_idx[t] for t in merged], dtype=np.int64)
    return claims, chunks, y, g16, g9, tag_names, m_names


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"model: {MODEL}", flush=True)
    t_all = time.time()

    claims, chunks, y, g16, g9, tag_names, m_names = load_mix()
    print(f"mix: {len(y)} rows, {len(tag_names)} groups -> {len(m_names)} merged", flush=True)
    print(f"  16: {tag_names}", flush=True)
    print(f"   9: {m_names}", flush=True)
    assert len(tag_names) == 16 and len(m_names) == 9, "group cardinality does not match the recipe"

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    fit_idx = perm[:FIT_ROWS]
    held_idx = perm[FIT_ROWS : FIT_ROWS + HELDOUT_BATCHES * BATCH]

    tok = AutoTokenizer.from_pretrained(MODEL)
    state = torch.load(
        pathlib.Path(MODEL) / "dann_student.pt", map_location="cpu", weights_only=False
    )
    trunk = AutoModel.from_pretrained(str(pathlib.Path(MODEL) / "trunk")).cuda().eval()
    trunk.config.reference_compile = False
    for p in trunk.parameters():
        p.requires_grad_(True)  # gradients w.r.t. the trunk are the measurement
    d = trunk.config.hidden_size

    task_head = nn.Linear(d, 1)
    task_head.load_state_dict(state["task_head"])
    task_head = task_head.cuda().eval()
    head16 = domain_head(16, d).cuda()
    head16.load_state_dict(state["domain_head"])
    head16.eval()

    # ── step 1: cache CLS for the fit rows, then fit the 9-way head ──────────
    if FEATS.exists():
        z = np.load(FEATS)
        cls_fit = z["cls"]
        print(f"cached CLS features {cls_fit.shape}", flush=True)
    else:
        t0 = time.time()
        cls_fit = np.zeros((len(fit_idx), d), dtype=np.float16)
        with torch.inference_mode():
            for j in range(0, len(fit_idx), 64):
                sl = fit_idx[j : j + 64]
                enc = tok([claims[i] for i in sl], [chunks[i] for i in sl],
                          return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
                enc = {k: v.cuda() for k, v in enc.items()}
                cls_fit[j : j + 64] = trunk(**enc).last_hidden_state[:, 0].half().cpu().numpy()
                if j % 6400 == 0:
                    print(f"  cls {j}/{len(fit_idx)} ({time.time() - t0:.0f}s)", flush=True)
        np.savez(FEATS, cls=cls_fit)
        print(f"  CLS cache {time.time() - t0:.0f}s -> {FEATS}", flush=True)

    def fit_head(n_groups, groups_all, name):
        head = domain_head(n_groups, d).cuda().train()
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        X = torch.from_numpy(cls_fit).float()
        G = torch.from_numpy(groups_all[fit_idx])
        n = len(X)
        for ep in range(HEAD_EPOCHS):
            order = torch.randperm(n)
            tot, corr, seen = 0.0, 0, 0
            for j in range(0, n, 256):
                b = order[j : j + 256]
                xb, gb = X[b].cuda(), G[b].cuda()
                logit = head(xb)
                loss = lossf(logit, gb)
                loss.backward()
                opt.step()
                opt.zero_grad()
                tot += loss.item() * len(b)
                corr += (logit.argmax(-1) == gb).sum().item()
                seen += len(b)
            print(f"  {name} epoch {ep}: loss {tot / seen:.4f}  acc {corr / seen:.4f} "
                  f"(chance {1 / n_groups:.4f})", flush=True)
        head.eval()
        return head, corr / seen

    head9, acc9 = fit_head(9, g9, "head9")
    head16_refit, acc16_refit = fit_head(16, g16, "head16-refit (diagnostic)")

    # ── step 2: trunk-gradient norms and direction cosine over held-out batches ─
    tparams = [p for p in trunk.parameters() if p.requires_grad]
    task_lossf = nn.BCEWithLogitsLoss()
    dom_lossf = nn.CrossEntropyLoss()

    def flat_norm(gs):
        return float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs)).item())

    def flat_dot(a, b):
        return float(sum((x.float() * y.float()).sum() for x, y in zip(a, b, strict=True)).item())

    acc = {k: [] for k in ("n_task", "n16", "n9", "n16r", "cos", "cos_r")}
    t0 = time.time()
    for bi in range(HELDOUT_BATCHES):
        sl = held_idx[bi * BATCH : (bi + 1) * BATCH]
        if len(sl) == 0:
            break
        enc = tok([claims[i] for i in sl], [chunks[i] for i in sl],
                  return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        enc = {k: v.cuda() for k, v in enc.items()}
        yy = torch.from_numpy(y[sl]).cuda()
        gg16 = torch.from_numpy(g16[sl]).cuda()
        gg9 = torch.from_numpy(g9[sl]).cuda()

        cls = trunk(**enc).last_hidden_state[:, 0]

        t_loss = task_lossf(task_head(cls).squeeze(-1), yy)
        gt = torch.autograd.grad(t_loss, tparams, retain_graph=True)
        acc["n_task"].append(flat_norm(gt))
        del gt

        d16 = dom_lossf(head16(GradReverse.apply(cls, LAMBDA)), gg16)
        g_16 = torch.autograd.grad(d16, tparams, retain_graph=True)
        d9 = dom_lossf(head9(GradReverse.apply(cls, LAMBDA)), gg9)
        g_9 = torch.autograd.grad(d9, tparams, retain_graph=True)
        d16r = dom_lossf(head16_refit(GradReverse.apply(cls, LAMBDA)), gg16)
        g_16r = torch.autograd.grad(d16r, tparams)

        n16, n9, n16r = flat_norm(g_16), flat_norm(g_9), flat_norm(g_16r)
        acc["n16"].append(n16)
        acc["n9"].append(n9)
        acc["n16r"].append(n16r)
        acc["cos"].append(flat_dot(g_16, g_9) / max(n16 * n9, 1e-12))
        acc["cos_r"].append(flat_dot(g_16r, g_9) / max(n16r * n9, 1e-12))
        del g_16, g_9, g_16r, cls
        trunk.zero_grad(set_to_none=True)
        if bi % 25 == 0:
            print(f"  batch {bi}/{HELDOUT_BATCHES}  |g16| {n16:.4e}  |g9| {n9:.4e}  "
                  f"cos {acc['cos'][-1]:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    a = {k: np.array(v) for k, v in acc.items()}
    ratio_16_9 = float(a["n16"].mean() / a["n9"].mean())
    ratio_per_batch = float(np.mean(a["n16"] / a["n9"]))
    r16 = float(a["n16"].mean() / a["n_task"].mean())
    r9 = float(a["n9"].mean() / a["n_task"].mean())
    cos = float(a["cos"].mean())

    license_ = ratio_16_9 >= 1.15 and cos <= 0.9
    kill = ratio_16_9 <= 1.05 or cos >= 0.95
    verdict = "LICENSE" if license_ else ("KILL" if kill else "AMBIGUOUS")

    res = {
        "gate": "R12-H122 gradient gate - 16-way vs 9-way GRL trunk gradient",
        "model": MODEL, "lambda": LAMBDA, "batch": BATCH,
        "n_batches": int(len(a["n16"])), "rows_per_batch": BATCH,
        "groups_16": tag_names, "groups_9": m_names,
        "head9_fit": {"rows": int(len(fit_idx)), "epochs": HEAD_EPOCHS,
                      "final_train_acc": round(acc9, 4), "chance": round(1 / 9, 4)},
        "norms": {
            "mean_norm_domain_16": float(a["n16"].mean()),
            "mean_norm_domain_9": float(a["n9"].mean()),
            "mean_norm_task_bce": float(a["n_task"].mean()),
            "ratio_domain16_over_task": round(r16, 4),
            "ratio_domain9_over_task": round(r9, 4),
        },
        "norm_ratio_16_over_9": round(ratio_16_9, 4),
        "norm_ratio_16_over_9_per_batch_mean": round(ratio_per_batch, 4),
        "direction_cosine_16_vs_9": round(cos, 4),
        "direction_cosine_sd": round(float(a["cos"].std()), 4),
        "diagnostic_refit16": {
            "note": "16-way head refit under the 9-way head's protocol; adjudicates nothing",
            "final_train_acc": round(acc16_refit, 4),
            "norm_ratio_refit16_over_9": round(float(a["n16r"].mean() / a["n9"].mean()), 4),
            "direction_cosine_refit16_vs_9": round(float(a["cos_r"].mean()), 4),
        },
        "bar": "LICENSE if ratio >= 1.15 AND cosine <= 0.9; KILL if ratio <= 1.05 OR cosine >= 0.95",
        "verdict": verdict,
    }
    RESULT.write_text(json.dumps(res, indent=2))

    print("\n" + "=" * 92)
    print("R12-H122 GRADIENT GATE")
    print("=" * 92)
    print(f"  |g_dom16| {a['n16'].mean():.4e}   |g_dom9| {a['n9'].mean():.4e}   "
          f"|g_task| {a['n_task'].mean():.4e}")
    print(f"  norm ratio 16/9        {ratio_16_9:.4f}   (bar >= 1.15 license, <= 1.05 kill)")
    print(f"  ratio-of-ratios        dom16/task {r16:.4f}  vs  dom9/task {r9:.4f}")
    print(f"  direction cosine       {cos:.4f} +- {a['cos'].std():.4f}   "
          f"(bar <= 0.9 license, >= 0.95 kill)")
    print(f"  diagnostic refit-16    ratio {a['n16r'].mean() / a['n9'].mean():.4f}  "
          f"cos {a['cos_r'].mean():.4f}")
    print(f"\n  VERDICT: {verdict}   ({time.time() - t_all:.0f}s)")
    print(f"  -> {RESULT}")


if __name__ == "__main__":
    main()

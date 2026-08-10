"""R16-H140 G1 stage 2 - train the window-ensemble readout and read the blind arena.

The readout replaces ONLY the max-over-windows step. Everything else in the
serving read is unchanged: the same frozen checkpoint, the same 1500/750 window
geometry, the same sentence splitter, the same hard MIN over sentences, the same
response-level AUROC per subset.

Readout = single-head attention pooling over the frozen (CLS, frozen-logit) set
of one sentence's windows, followed by a small MLP. <= 2M params, single seed,
trained on cached embeddings of PUBLIC rows only.

Run (CPU is enough, GPU2 used if free):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 uv run python \
    experiments/grounding-semantic/R16-H140_G1_readout.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import argparse
import importlib.util
import json
import pathlib
import random
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn

HERE = pathlib.Path(__file__).parent
CACHE = HERE / "R16-H140_cache"
OUT = HERE / "R16-H140_G1_pilot.json"

SEED = 0
HID = 256
LR = 1e-3
EPOCHS = 8
SETS_PER_BATCH = 64
PAIRS_PER_BATCH = 16_384  # memory guard only
VAL_FRAC = 0.08
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Banked hard-max windowed read of the same checkpoint (R10-H108_lane_draw1_windowed_result.json)
BANKED = {
    "covidqa": 0.7516, "delucionqa": 0.7355, "emanual": 0.6719, "expertqa": 0.7496,
    "finqa": 0.7291, "hagrid": 0.6599, "hotpotqa": 0.6965, "pubmedqa": 0.5907,
    "tatqa": 0.7391, "techqa": 0.7379,
}
BANKED_MEAN = 0.70618


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p)).astype(np.float32)


def group_index(pair_sent, n_sent):
    """List of pair-index arrays, one per sentence (pairs are contiguous by build)."""
    order = np.argsort(pair_sent, kind="stable")
    ps = pair_sent[order]
    bounds = np.searchsorted(ps, np.arange(n_sent + 1))
    return order, bounds


class Readout(nn.Module):
    def __init__(self, d=768, h=HID):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d + 1, h)
        self.att = nn.Linear(h, 1)
        self.val = nn.Linear(h, h)
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, emb, lg, mask):
        # emb [B,K,d] float32, lg [B,K], mask [B,K] bool
        x = torch.tanh(self.proj(torch.cat([self.norm(emb), lg.unsqueeze(-1)], dim=-1)))
        a = self.att(x).squeeze(-1).masked_fill(~mask, float("-inf")).softmax(dim=-1)
        pooled = (a.unsqueeze(-1) * self.val(x)).sum(dim=1)
        return self.out(pooled).squeeze(-1)


def make_batches(sizes, ids, n_sets=SETS_PER_BATCH, cap=PAIRS_PER_BATCH):
    """Group sentence ids into size-sorted batches (bounded sets, padded-pair cap)."""
    order = sorted(ids, key=lambda i: sizes[i])
    out, cur, k = [], [], 0
    for i in order:
        nk = max(k, int(sizes[i]))
        if cur and (len(cur) + 1 > n_sets or (len(cur) + 1) * nk > cap):
            out.append(cur)
            cur, k = [], 0
            nk = int(sizes[i])
        cur.append(i)
        k = nk
    if cur:
        out.append(cur)
    return out


def pack(batch, emb, lg, order, bounds, d):
    K = max(bounds[i + 1] - bounds[i] for i in batch)
    B = len(batch)
    E = np.zeros((B, K, d), dtype=np.float32)
    L = np.zeros((B, K), dtype=np.float32)
    M = np.zeros((B, K), dtype=bool)
    for r, i in enumerate(batch):
        idx = order[bounds[i]:bounds[i + 1]]
        n = len(idx)
        E[r, :n] = emb[idx].astype(np.float32)
        L[r, :n] = lg[idx]
        M[r, :n] = True
    return (torch.from_numpy(E).to(DEV), torch.from_numpy(L).to(DEV),
            torch.from_numpy(M).to(DEV))


def train_readout():
    z = np.load(CACHE / "train.npz")
    emb, sc, pair_sent = z["emb"], z["score"], z["pair_sent"]
    lab, src = z["label"], z["source"]
    n_sent, d = len(lab), emb.shape[1]
    lg = logit(sc)
    order, bounds = group_index(pair_sent, n_sent)
    sizes = bounds[1:] - bounds[:-1]

    rng = np.random.default_rng(SEED)
    pyrng = random.Random(SEED)
    perm = rng.permutation(n_sent)
    n_val = int(VAL_FRAC * n_sent)
    val_ids, tr_ids = list(perm[:n_val]), list(perm[n_val:])

    torch.manual_seed(SEED)
    model = Readout(d=d).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  readout params: {n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    lossf = nn.BCEWithLogitsLoss()

    val_batches = make_batches(sizes, val_ids)
    best, best_state, hist = -1.0, None, []
    for ep in range(EPOCHS):
        model.train()
        batches = make_batches(sizes, tr_ids)
        pyrng.shuffle(batches)
        tot, nb = 0.0, 0
        t0 = time.time()
        for b in batches:
            E, L, M = pack(b, emb, lg, order, bounds, d)
            y = torch.from_numpy(lab[b]).to(DEV)
            out = model(E, L, M)
            loss = lossf(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        model.eval()
        preds, ys = [], []
        with torch.inference_mode():
            for b in val_batches:
                E, L, M = pack(b, emb, lg, order, bounds, d)
                preds.append(model(E, L, M).float().cpu().numpy())
                ys.append(lab[b])
        vp, vy = np.concatenate(preds), np.concatenate(ys)
        vauc = float(roc_auc_score(vy, vp))
        # frozen hard-max on the same validation sentences, for orientation
        hm = np.array([sc[order[bounds[i]:bounds[i + 1]]].max() for i in val_ids])
        hauc = float(roc_auc_score(lab[val_ids], hm))
        hist.append({"epoch": ep, "train_loss": tot / nb, "val_auc_readout": round(vauc, 4),
                     "val_auc_hardmax": round(hauc, 4), "sec": round(time.time() - t0, 1)})
        print(f"  epoch {ep}: loss {tot/nb:.4f}  val readout {vauc:.4f}  "
              f"val hard-max {hauc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if vauc > best:
            best = vauc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, n_params, hist, best, src


def read_arena(model):
    files = sorted(CACHE.glob("arena_*.npz"))
    per_sub, pooled = {}, {"readout": [], "hardmax": [], "y": [], "stratum": []}
    for f in files:
        sub = f.stem.replace("arena_", "")
        z = np.load(f)
        emb, sc, pair_sent = z["emb"], z["score"], z["pair_sent"]
        sent_owner, sent_cross, y = z["sent_owner"], z["sent_cross"], z["y"]
        n_sent, d = len(sent_owner), emb.shape[1]
        lg = logit(sc)
        order, bounds = group_index(pair_sent, n_sent)
        sizes = bounds[1:] - bounds[:-1]

        s_hard = np.array([sc[order[bounds[i]:bounds[i + 1]]].max() for i in range(n_sent)])
        s_read = np.zeros(n_sent, dtype=np.float32)
        with torch.inference_mode():
            for b in make_batches(sizes, list(range(n_sent))):
                E, L, M = pack(b, emb, lg, order, bounds, d)
                s_read[b] = model(E, L, M).float().cpu().numpy()

        n_resp = len(y)
        r_hard = np.array([s_hard[sent_owner == i].min() for i in range(n_resp)])
        r_read = np.array([s_read[sent_owner == i].min() for i in range(n_resp)])
        strat = np.array([bool(sent_cross[sent_owner == i].any()) for i in range(n_resp)])

        a_hard = float(roc_auc_score(y, r_hard))
        a_read = float(roc_auc_score(y, r_read))
        row = {
            "n": int(n_resp), "n_sent": int(n_sent), "n_pairs": len(sc),
            "hardmax_auc": round(a_hard, 4), "readout_auc": round(a_read, 4),
            "delta": round(a_read - a_hard, 4),
            "banked_hardmax_auc": BANKED.get(sub),
            "hardmax_reproduces_banked": bool(abs(a_hard - BANKED.get(sub, -9)) < 1e-3),
            "stratum_n": int(strat.sum()),
        }
        if strat.sum() >= 20 and len(set(y[strat].tolist())) == 2:
            row["stratum_hardmax_auc"] = round(float(roc_auc_score(y[strat], r_hard[strat])), 4)
            row["stratum_readout_auc"] = round(float(roc_auc_score(y[strat], r_read[strat])), 4)
            row["stratum_delta"] = round(row["stratum_readout_auc"] - row["stratum_hardmax_auc"], 4)
        per_sub[sub] = row
        pooled["readout"].append(r_read)
        pooled["hardmax"].append(r_hard)
        pooled["y"].append(y)
        pooled["stratum"].append(strat)
        print(f"  {sub:14s} hard-max {a_hard:.4f} (banked {BANKED.get(sub)})  "
              f"readout {a_read:.4f}  delta {a_read - a_hard:+.4f}", flush=True)

    pr = np.concatenate(pooled["readout"])
    ph = np.concatenate(pooled["hardmax"])
    py = np.concatenate(pooled["y"])
    pst = np.concatenate(pooled["stratum"])
    stratum = {
        "definition": "arena responses holding >= 1 G0-flagged cross-window sentence "
                      "(anchors not co-locatable in any single 1500-char window)",
        "pooled_n": int(pst.sum()), "pooled_n_total": len(py),
        "pooled_hardmax_auc": round(float(roc_auc_score(py[pst], ph[pst])), 4),
        "pooled_readout_auc": round(float(roc_auc_score(py[pst], pr[pst])), 4),
        "complement_hardmax_auc": round(float(roc_auc_score(py[~pst], ph[~pst])), 4),
        "complement_readout_auc": round(float(roc_auc_score(py[~pst], pr[~pst])), 4),
    }
    stratum["pooled_delta"] = round(stratum["pooled_readout_auc"] - stratum["pooled_hardmax_auc"], 4)
    per_sub_strat = [v["stratum_delta"] for v in per_sub.values() if "stratum_delta" in v]
    stratum["mean_per_subset_delta"] = round(float(np.mean(per_sub_strat)), 4) if per_sub_strat else None
    stratum["mean_per_subset_hardmax"] = round(
        float(np.mean([v["stratum_hardmax_auc"] for v in per_sub.values() if "stratum_hardmax_auc" in v])), 4)
    stratum["mean_per_subset_readout"] = round(
        float(np.mean([v["stratum_readout_auc"] for v in per_sub.values() if "stratum_readout_auc" in v])), 4)
    return per_sub, stratum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/R10-H108-lane-draw1")
    ap.parse_args()
    print(f"=== R16-H140 G1 readout {time.strftime('%F %T')} dev={DEV} ===", flush=True)

    model, n_params, hist, best_val, _src = train_readout()
    per_sub, stratum = read_arena(model)

    mean_read = float(np.mean([v["readout_auc"] for v in per_sub.values()]))
    mean_hard = float(np.mean([v["hardmax_auc"] for v in per_sub.values()]))
    worst = min(v["delta"] for v in per_sub.values())
    comp = json.loads((CACHE / "train_composition.json").read_text())

    # Primary stratum aggregation is the per-subset mean, matching how the campaign
    # aggregates the blind read; the pooled-across-subsets figure is recorded beside it.
    sdelta = stratum["mean_per_subset_delta"]
    trend = (mean_read >= mean_hard + 0.005) and (sdelta is not None and sdelta >= 0.03)
    kill = (mean_read <= mean_hard) and (sdelta is not None and sdelta <= 0)

    payload = {
        "gate": "R16-H140 G1 frozen-composability pilot",
        "checkpoint": "models/R10-H108-lane-draw1 (clean+H108 lane draw 1, admitted lane best single ckpt)",
        "readout": "single-head attention pooling over (CLS, frozen-logit) window sets + 2-layer MLP",
        "readout_params": n_params,
        "seed": SEED,
        "readout_mean": round(mean_read, 5),
        "hardmax_mean": round(mean_hard, 5),
        "banked_hardmax_mean": BANKED_MEAN,
        "mean_delta": round(mean_read - mean_hard, 5),
        "worst_subset_delta": round(worst, 4),
        "per_subset": per_sub,
        "cross_window_stratum": stratum,
        "training": {
            "composition": comp,
            "val_frac": VAL_FRAC, "epochs_run": len(hist),
            "best_val_sentence_auc": round(best_val, 4),
            "history": hist,
            "no_gold_rows": True, "no_arena_rows": True,
        },
        "registered_bars": {
            "trend_license": "mean >= hard-max + 0.005 AND cross-window stratum >= +0.03 "
                             "(stratum read as the per-subset mean delta; pooled delta also recorded)",
            "kill": "readout <= hard-max on BOTH mean and stratum",
            "stratum_delta_used": sdelta,
            "stratum_delta_pooled": stratum["pooled_delta"],
            "trend_license_met": bool(trend),
            "kill_met": bool(kill),
        },
        "note": "Bars are RECORDED, not adjudicated here - the coordinator adjudicates.",
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print("\n" + "=" * 88)
    print("R16-H140 G1 RESULT")
    print("=" * 88)
    print(f"  readout mean {mean_read:.5f}   hard-max mean {mean_hard:.5f}   "
          f"delta {mean_read - mean_hard:+.5f}   (banked hard-max {BANKED_MEAN})")
    print(f"  stratum (n={stratum['pooled_n']}): hard-max {stratum['pooled_hardmax_auc']:.4f}  "
          f"readout {stratum['pooled_readout_auc']:.4f}  delta {stratum['pooled_delta']:+.4f}")
    print(f"  stratum per-subset mean: hard-max {stratum['mean_per_subset_hardmax']:.4f}  "
          f"readout {stratum['mean_per_subset_readout']:.4f}  delta {sdelta:+.4f}")
    print(f"  worst subset delta {worst:+.4f}   readout params {n_params:,}")
    print(f"  trend-license met: {trend}   kill met: {kill}")
    print(f"  -> {OUT}")
    print("=== G1 READOUT DONE ===", flush=True)


if __name__ == "__main__":
    main()

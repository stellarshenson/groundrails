"""R16-H141 review-ordered controls A and B for the autopsy of the R16-H140 readout lift.

The autopsy (R16-H141_autopsy.py) concluded EMBEDDING: a logistic regression on 8
per-sentence window-score scalars recovered only +0.0092 of the readout's +0.0711
pubmedqa delta.  An adversarial review found that conclusion under-controlled on
two counts, both answered here on the EXISTING R16-H140 cache (no embedding
recomputation, no arena tuning).

CONTROL A - capacity-matched scalar model.  A logistic regression is a linear
probe; the readout is a 331k-param nonlinear model.  "Scalars cannot do it" was
therefore confounded with "a linear map cannot do it".  Control A trains the SAME
MLP body the readout uses (same depth, same hidden width, same optimiser, same
seed-0 split, same 8-epoch best-val protocol, same batch composition) fed ONLY the
autopsy's 8 scalars, in three capacities: architecture-matched (h=256), parameter-
matched (h=404, ~331k params - the readout's count), and a 6-feature ablation that
drops the two window-count features (fitted at set sizes <= 14, read on the arena
at up to 156).  Control A also adds the smooth forms of the max the autopsy's grid
omitted - log-sum-exp and softmax-weighted mean over the window logits - with the
temperature swept on the TRAINING-slice held-out rows only.

CONTROL B - readout seed noise.  The banked +0.0711 is a single seed.  Control B
retrains the FULL readout unchanged at seeds 1, 2, 3 by importing
R16-H140_G1_readout.py and rebinding its seed, reads each blind, and prices the
lift against the spread of the 4 seeds.  Seed 0 is also re-run once as a
harness-fidelity check against the banked pilot numbers; the banked values are the
ones used in the aggregate.

Discipline: every fit and every selection uses training-slice rows only.  The
arena is read once per model for final numbers and is never consulted for tuning.

Run (~10 min, GPU0 only - GPU1/GPU2 carry other experiments):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run python \
    experiments/grounding-semantic/R16-H141_controls.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import importlib.util
import json
import pathlib
import random
import time

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
import torch
from torch import nn

HERE = pathlib.Path(__file__).parent
CACHE = HERE / "R16-H140_cache"
PILOT = HERE / "R16-H140_G1_pilot.json"
AUTOPSY = HERE / "R16-H141_autopsy.json"
OUT = HERE / "R16-H141_controls.json"

SEED = 0
VAL_FRAC = 0.08
EPOCHS = 8
LR = 1e-3
HID_ARCH = 256   # the readout's hidden width
HID_PARAM = 404  # widened so the scalar MLP matches the readout's 331,266 params
SEEDS_B = [1, 2, 3]
TEMP_GRID = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]

BANKED = {
    "covidqa": 0.7516, "delucionqa": 0.7355, "emanual": 0.6719, "expertqa": 0.7496,
    "finqa": 0.7291, "hagrid": 0.6599, "hotpotqa": 0.6965, "pubmedqa": 0.5907,
    "tatqa": 0.7391, "techqa": 0.7379,
}
BANKED_MEAN = 0.70618
READOUT_PUBMEDQA_DELTA = 0.0711
BAR_SCALAR = 0.050          # same bar the autopsy registered (70% of +0.0711)
CAMPAIGN_SUBSET_SWING = (0.028, 0.033)  # known per-subset swing at fixed recipe


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# Reuse the autopsy's own feature code so Control A's inputs are byte-identical to
# the logreg's, and the readout trainer's own code so Control B is unchanged.
AP = _mod("h141_autopsy", "R16-H141_autopsy.py")
RO = _mod("h140_readout", "R16-H140_G1_readout.py")
FEATURE_NAMES = AP.FEATURE_NAMES
DEV = RO.DEV


# ---------------------------------------------------------------- Control A model


class ScalarNet(nn.Module):
    """The readout's MLP body, fed a per-sentence scalar vector instead of a window set.

    The readout is  proj(LN(emb) . logit) -> tanh -> attention-pool -> val -> LN ->
    Linear -> GELU -> Linear -> 1.  With a single per-sentence vector there is no set
    to pool, so the attention head is dropped and everything else is kept: same
    projection-then-tanh, same value layer, same 2-layer output MLP, same widths.
    Inputs are standardised with training-row mean/std (the readout's LayerNorm over a
    768-d embedding has no meaningful analogue over 8 heterogeneous scalars).
    """

    def __init__(self, d_in, h):
        super().__init__()
        self.proj = nn.Linear(d_in, h)
        self.val = nn.Linear(h, h)
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.out(self.val(torch.tanh(self.proj(x)))).squeeze(-1)


def train_scalar_mlp(X, y, tr_ids, val_ids, sizes, h, seed=SEED, tag=""):
    """Same optimiser / epochs / batch composition / best-val selection as the readout."""
    torch.manual_seed(seed)
    pyrng = random.Random(seed)
    model = ScalarNet(X.shape[1], h).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    lossf = nn.BCEWithLogitsLoss()

    Xt = torch.from_numpy(X.astype(np.float32)).to(DEV)
    yt = torch.from_numpy(y.astype(np.float32)).to(DEV)
    val_batches = RO.make_batches(sizes, list(val_ids))

    best, best_state, hist = -1.0, None, []
    for ep in range(EPOCHS):
        model.train()
        batches = RO.make_batches(sizes, list(tr_ids))
        pyrng.shuffle(batches)
        tot, nb, t0 = 0.0, 0, time.time()
        for b in batches:
            idx = torch.as_tensor(b, device=DEV)
            loss = lossf(model(Xt[idx]), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        model.eval()
        with torch.inference_mode():
            vp = np.concatenate([model(Xt[torch.as_tensor(b, device=DEV)]).float().cpu().numpy()
                                 for b in val_batches])
            vy = np.concatenate([y[b] for b in val_batches])
        vauc = float(roc_auc_score(vy, vp))
        hist.append({"epoch": ep, "train_loss": round(tot / nb, 5),
                     "val_auc": round(vauc, 4), "sec": round(time.time() - t0, 1)})
        print(f"    [{tag}] epoch {ep}: loss {tot/nb:.4f}  val sentence AUROC {vauc:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if vauc > best:
            best, best_state = vauc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, n_params, hist, best


def mlp_scores(model, F):
    with torch.inference_mode():
        x = torch.from_numpy(F.astype(np.float32)).to(DEV)
        return model(x).float().cpu().numpy().astype(np.float64)


# ------------------------------------------------- smooth aggregations over windows


def smooth_aggs(sc, order, bounds, n_sent, temps):
    """log-sum-exp and softmax-weighted mean over a sentence's window LOGITS.

    lse_T  = T * log sum_i exp(v_i / T)   -> max as T -> 0, T*log(n)+mean as T -> inf
    smax_T = sum_i softmax(v/T)_i * v_i   -> max as T -> 0, plain mean as T -> inf
    Both are the smooth forms of the max the autopsy's order-statistic grid omitted.
    """
    lg = AP.logit(sc)
    out = {f"lse_T{t}": np.zeros(n_sent) for t in temps}
    out.update({f"smax_T{t}": np.zeros(n_sent) for t in temps})
    for i in range(n_sent):
        v = lg[order[bounds[i]:bounds[i + 1]]]
        m = v.max()
        for t in temps:
            z = (v - m) / t
            e = np.exp(z)
            s = e.sum()
            out[f"lse_T{t}"][i] = m + t * np.log(s)
            out[f"smax_T{t}"][i] = float((e / s) @ v)
    return out


# ------------------------------------------------------------------ blind arena read


def arena_bundle():
    """Load every arena subset once: features, smooth aggs, response masks, labels."""
    bundle = {}
    for f in sorted(CACHE.glob("arena_*.npz")):
        sub = f.stem.replace("arena_", "")
        z = np.load(f)
        sc, pair_sent, sent_owner, y = z["score"], z["pair_sent"], z["sent_owner"], z["y"]
        ns = len(sent_owner)
        F, hardmax_prob, order, bounds = AP.features(sc, pair_sent, ns)
        masks = [sent_owner == i for i in range(len(y))]
        r_hard = np.array([hardmax_prob[m].min() for m in masks])
        bundle[sub] = {
            "F": F, "masks": masks, "y": y,
            "hardmax_auc": float(roc_auc_score(y, r_hard)),
            "smooth": smooth_aggs(sc, order, bounds, ns, TEMP_GRID),
            "n": int(len(y)),
        }
        d = abs(bundle[sub]["hardmax_auc"] - BANKED[sub])
        assert d <= 1e-4, f"{sub} hard-max fidelity {d:.2e}"
    return bundle


def blind_read(bundle, score_fn):
    """Per-sentence score -> hard MIN over the response's sentences -> per-subset AUROC."""
    per_sub = {}
    for sub, b in bundle.items():
        s = np.asarray(score_fn(sub, b), dtype=np.float64)
        r = np.array([s[m].min() for m in b["masks"]])
        auc = float(roc_auc_score(b["y"], r))
        per_sub[sub] = {"auc": round(auc, 4), "delta": round(auc - b["hardmax_auc"], 4)}
    mean = float(np.mean([v["auc"] for v in per_sub.values()]))
    return {"per_subset": per_sub, "blind_mean_auc": round(mean, 5),
            "mean_delta_vs_hardmax": round(mean - BANKED_MEAN, 5),
            "worst_subset_delta": round(min(v["delta"] for v in per_sub.values()), 4)}


# --------------------------------------------------------------------- Control A


def control_a(bundle):
    print("\n--- CONTROL A: capacity-matched scalar model + smooth max forms ---", flush=True)
    z = np.load(CACHE / "train.npz")
    tr_sc, tr_pair_sent, lab = z["score"], z["pair_sent"], z["label"]
    n_sent = len(lab)
    Ftr, _, order, bounds = AP.features(tr_sc, tr_pair_sent, n_sent)
    sizes = bounds[1:] - bounds[:-1]

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_sent)
    n_val = int(VAL_FRAC * n_sent)
    val_ids, tr_ids = perm[:n_val], perm[n_val:]

    mu, sd = Ftr[tr_ids].mean(axis=0), Ftr[tr_ids].std(axis=0) + 1e-9
    Xtr = (Ftr - mu) / sd

    # --- MLP capacities -------------------------------------------------------
    six = [FEATURE_NAMES.index(n) for n in FEATURE_NAMES if n not in ("n_windows", "log_n_windows")]
    specs = [
        ("mlp_scalar_arch", HID_ARCH, list(range(len(FEATURE_NAMES)))),
        ("mlp_scalar_param", HID_PARAM, list(range(len(FEATURE_NAMES)))),
        ("mlp_scalar_6f", HID_ARCH, six),
    ]
    models = {}
    for name, h, cols in specs:
        model, n_params, hist, best = train_scalar_mlp(
            Xtr[:, cols], lab, tr_ids, val_ids, sizes, h, tag=name)
        models[name] = {"model": model, "cols": cols, "params": n_params,
                        "history": hist, "val_sentence_auc": round(best, 5)}
        print(f"    [{name}] params {n_params:,}  best val sentence AUROC {best:.4f}", flush=True)

    # --- smooth aggregations: temperature swept on TRAINING-slice val rows only ---
    tr_smooth = smooth_aggs(tr_sc, order, bounds, n_sent, TEMP_GRID)
    smooth_val = {k: float(roc_auc_score(lab[val_ids], v[val_ids])) for k, v in tr_smooth.items()}
    best_lse = max((k for k in smooth_val if k.startswith("lse_")), key=lambda k: smooth_val[k])
    best_smax = max((k for k in smooth_val if k.startswith("smax_")), key=lambda k: smooth_val[k])
    print(f"    smooth-agg val selection: {best_lse} ({smooth_val[best_lse]:.4f}), "
          f"{best_smax} ({smooth_val[best_smax]:.4f})", flush=True)

    # --- blind arena read ------------------------------------------------------
    reads = {}
    for name, m in models.items():
        reads[name] = blind_read(bundle, lambda sub, b, mm=m["model"], c=m["cols"]:
                                 mlp_scores(mm, ((b["F"] - mu) / sd)[:, c]))
        reads[name]["val_sentence_auc"] = m["val_sentence_auc"]
        reads[name]["params"] = m["params"]
    for k in tr_smooth:
        reads[k] = blind_read(bundle, lambda sub, b, kk=k: b["smooth"][kk])
        reads[k]["val_sentence_auc"] = round(smooth_val[k], 5)

    ap = json.loads(AUTOPSY.read_text())
    pilot = json.loads(PILOT.read_text())
    reads["logreg_scalar_banked"] = {
        "per_subset": {s: {"auc": ap["per_subset"][s]["logreg_scalar_auc"],
                           "delta": ap["per_subset"][s]["logreg_scalar_delta"]}
                       for s in ap["per_subset"]},
        "blind_mean_auc": ap["variant_summary"]["logreg_scalar"]["blind_mean_auc"],
        "mean_delta_vs_hardmax": ap["variant_summary"]["logreg_scalar"]["mean_delta_vs_hardmax"],
        "worst_subset_delta": ap["variant_summary"]["logreg_scalar"]["worst_subset_delta"],
        "val_sentence_auc": ap["fit"]["train_slice_val_sentence_auc"]["logreg_scalar"],
    }
    reads["readout_banked_seed0"] = {
        "per_subset": {s: {"auc": pilot["per_subset"][s]["readout_auc"],
                           "delta": pilot["per_subset"][s]["delta"]} for s in pilot["per_subset"]},
        "blind_mean_auc": pilot["readout_mean"],
        "mean_delta_vs_hardmax": pilot["mean_delta"],
        "worst_subset_delta": pilot["worst_subset_delta"],
        "val_sentence_auc": pilot["training"]["best_val_sentence_auc"],
    }

    # honest primary: highest training-slice val sentence AUROC among the NEW scalar variants
    new_names = [n for n, _, _ in specs] + list(tr_smooth)
    primary = max(new_names, key=lambda n: reads[n]["val_sentence_auc"])
    best_pub = max(new_names, key=lambda n: reads[n]["per_subset"]["pubmedqa"]["delta"])

    return {
        "question": "Is the readout's pubmedqa lift recoverable by a CAPACITY-MATCHED nonlinear "
                    "function of the same 8 window-score scalars, or by a smooth max?",
        "readout_body_reused": "same depth/widths/optimiser/epochs/best-val protocol/batch "
                               "composition as R16-H140_G1_readout; attention head dropped (no set "
                               "to pool over a single per-sentence scalar vector)",
        "readout_params": pilot["readout_params"],
        "features": FEATURE_NAMES,
        "smooth_temperature_grid": TEMP_GRID,
        "temperature_selected_on": "training-slice held-out rows (val_frac 0.08, seed 0); the arena "
                                   "was never consulted",
        "train_slice_val_sentence_auc": {k: round(v, 5) for k, v in smooth_val.items()},
        "smooth_selected": {"lse": best_lse, "smax": best_smax},
        "models": {n: {"params": models[n]["params"], "hidden": h,
                       "n_features": len(models[n]["cols"]),
                       "val_sentence_auc": models[n]["val_sentence_auc"],
                       "history": models[n]["history"]}
                   for (n, h, _) in specs},
        "reads": reads,
        "primary_variant": primary,
        "primary_selection_rule": "highest training-slice held-out sentence AUROC among the Control A "
                                  "variants; arena never consulted",
        "primary_pubmedqa_delta": reads[primary]["per_subset"]["pubmedqa"]["delta"],
        "oracle_variant": best_pub,
        "oracle_pubmedqa_delta": reads[best_pub]["per_subset"]["pubmedqa"]["delta"],
        "bar": f"pubmedqa delta >= +{BAR_SCALAR:.3f} -> SCALAR (the lift is a nonlinear function of "
               f"window-score scalars); below -> EMBEDDING (the autopsy's conclusion survives)",
        "primary_branch": "SCALAR" if reads[primary]["per_subset"]["pubmedqa"]["delta"] >= BAR_SCALAR
                          else "EMBEDDING",
        "oracle_branch": "SCALAR" if reads[best_pub]["per_subset"]["pubmedqa"]["delta"] >= BAR_SCALAR
                         else "EMBEDDING",
    }


# --------------------------------------------------------------------- Control B


def control_b():
    print("\n--- CONTROL B: readout seed noise (full architecture, unchanged protocol) ---",
          flush=True)
    pilot = json.loads(PILOT.read_text())
    runs = {"0_banked": {"per_subset": {s: {"auc": v["readout_auc"], "delta": v["delta"]}
                                        for s, v in pilot["per_subset"].items()},
                         "blind_mean_auc": pilot["readout_mean"],
                         "mean_delta_vs_hardmax": pilot["mean_delta"],
                         "val_sentence_auc": pilot["training"]["best_val_sentence_auc"],
                         "source": "R16-H140_G1_pilot.json (banked)"}}

    for s in [SEED] + SEEDS_B:
        t0 = time.time()
        RO.SEED = s
        print(f"  seed {s}: training readout ...", flush=True)
        model, n_params, hist, best_val, _ = RO.train_readout()
        per_sub, _strat = RO.read_arena(model)
        mean_read = float(np.mean([v["readout_auc"] for v in per_sub.values()]))
        key = "0_rerun" if s == SEED else str(s)
        runs[key] = {
            "per_subset": {k: {"auc": v["readout_auc"], "delta": v["delta"]}
                           for k, v in per_sub.items()},
            "blind_mean_auc": round(mean_read, 5),
            "mean_delta_vs_hardmax": round(mean_read - BANKED_MEAN, 5),
            "val_sentence_auc": round(best_val, 4),
            "params": n_params,
            "epochs": [h["val_auc_readout"] for h in hist],
            "runtime_sec": round(time.time() - t0, 1),
            "source": "retrained here",
        }
        print(f"  seed {s}: pubmedqa {per_sub['pubmedqa']['delta']:+.4f}  "
              f"mean {mean_read:.5f} ({mean_read - BANKED_MEAN:+.5f})  "
              f"({time.time()-t0:.0f}s)", flush=True)
        del model
        if DEV == "cuda":
            torch.cuda.empty_cache()

    fidelity = {
        "check": "seed 0 retrained here vs the banked pilot - guards against a harness error in the "
                 "seed rebinding",
        "banked_pubmedqa_delta": runs["0_banked"]["per_subset"]["pubmedqa"]["delta"],
        "rerun_pubmedqa_delta": runs["0_rerun"]["per_subset"]["pubmedqa"]["delta"],
        "banked_mean": runs["0_banked"]["blind_mean_auc"],
        "rerun_mean": runs["0_rerun"]["blind_mean_auc"],
        "abs_pubmedqa_diff": round(abs(runs["0_banked"]["per_subset"]["pubmedqa"]["delta"]
                                       - runs["0_rerun"]["per_subset"]["pubmedqa"]["delta"]), 4),
    }

    # aggregate over the 4 registered seeds: banked seed 0 + retrained 1,2,3
    agg_keys = ["0_banked"] + [str(s) for s in SEEDS_B]
    subs = sorted(runs["0_banked"]["per_subset"])
    spread = {}
    for sub in subs:
        d = np.array([runs[k]["per_subset"][sub]["delta"] for k in agg_keys])
        spread[sub] = {"per_seed": [round(float(x), 4) for x in d],
                       "mean": round(float(d.mean()), 4),
                       "std": round(float(d.std(ddof=1)), 4),
                       "min": round(float(d.min()), 4), "max": round(float(d.max()), 4),
                       "range": round(float(d.max() - d.min()), 4),
                       "all_same_sign": bool((d > 0).all() or (d < 0).all())}
    md = np.array([runs[k]["mean_delta_vs_hardmax"] for k in agg_keys])
    return {
        "question": "Is the readout's +0.0711 pubmedqa lift inside or outside seed noise?",
        "seeds_aggregated": agg_keys,
        "protocol": "R16-H140_G1_readout.py imported unchanged; only the module SEED is rebound "
                    "(it drives the train/val permutation, the batch shuffle and torch init)",
        "runs": runs,
        "harness_fidelity": fidelity,
        "per_subset_spread": spread,
        "blind_mean_delta": {"per_seed": [round(float(x), 5) for x in md],
                             "mean": round(float(md.mean()), 5),
                             "std": round(float(md.std(ddof=1)), 5),
                             "min": round(float(md.min()), 5), "max": round(float(md.max()), 5)},
        "campaign_subset_swing": {"low": CAMPAIGN_SUBSET_SWING[0], "high": CAMPAIGN_SUBSET_SWING[1],
                                  "meaning": "known per-subset swing at fixed recipe across the "
                                             "campaign - the reference noise floor"},
    }


# -------------------------------------------------------------------------- main


def table(title, rows, cols):
    df = pl.DataFrame(rows)
    print("\n" + title)
    with pl.Config(tbl_rows=60, tbl_cols=len(cols), tbl_width_chars=200,
                   float_precision=4, tbl_hide_dataframe_shape=True):
        print(df.select(cols))


def main():
    t0 = time.time()
    print(f"=== R16-H141 controls {time.strftime('%F %T')} dev={DEV} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}) ===", flush=True)

    bundle = arena_bundle()
    print(f"  arena loaded, hard-max fidelity vs banked OK (10 subsets, "
          f"{sum(b['n'] for b in bundle.values())} responses)", flush=True)

    A = control_a(bundle)
    B = control_b()

    subs = sorted(bundle)
    a_rows = []
    for name, r in A["reads"].items():
        a_rows.append({
            "variant": name,
            "val_sent_auc": r.get("val_sentence_auc"),
            "params": r.get("params"),
            "blind_mean": r["blind_mean_auc"],
            "mean_delta": r["mean_delta_vs_hardmax"],
            "pubmedqa_d": r["per_subset"]["pubmedqa"]["delta"],
            "techqa_d": r["per_subset"]["techqa"]["delta"],
            "hotpotqa_d": r["per_subset"]["hotpotqa"]["delta"],
            "tatqa_d": r["per_subset"]["tatqa"]["delta"],
            "worst_d": r["worst_subset_delta"],
        })
    table("CONTROL A - scalar-only models and smooth max forms (blind arena)", a_rows,
          ["variant", "val_sent_auc", "params", "blind_mean", "mean_delta", "pubmedqa_d",
           "techqa_d", "hotpotqa_d", "tatqa_d", "worst_d"])
    print(f"  primary (train-selected): {A['primary_variant']}  pubmedqa "
          f"{A['primary_pubmedqa_delta']:+.4f} -> {A['primary_branch']}")
    print(f"  oracle (arena-selected ceiling): {A['oracle_variant']}  pubmedqa "
          f"{A['oracle_pubmedqa_delta']:+.4f} -> {A['oracle_branch']}")

    b_rows = []
    for k in ["0_banked", "0_rerun"] + [str(s) for s in SEEDS_B]:
        r = B["runs"][k]
        b_rows.append({"seed": k, "blind_mean": r["blind_mean_auc"],
                       "mean_delta": r["mean_delta_vs_hardmax"],
                       **{f"{s}_d": r["per_subset"][s]["delta"] for s in subs}})
    table("CONTROL B - full readout, per-seed blind arena deltas vs hard-max", b_rows,
          ["seed", "blind_mean", "mean_delta", "pubmedqa_d", "hotpotqa_d", "tatqa_d",
           "techqa_d", "covidqa_d", "expertqa_d"])
    sp = B["per_subset_spread"]
    s_rows = [{"subset": s, "mean_delta": sp[s]["mean"], "std": sp[s]["std"],
               "min": sp[s]["min"], "max": sp[s]["max"], "range": sp[s]["range"],
               "same_sign": sp[s]["all_same_sign"]} for s in subs]
    table("CONTROL B - spread across the 4 seeds (banked 0 + retrained 1,2,3)", s_rows,
          ["subset", "mean_delta", "std", "min", "max", "range", "same_sign"])

    pub = sp["pubmedqa"]
    summary = {
        "control_a": {
            "primary_variant": A["primary_variant"],
            "primary_pubmedqa_delta": A["primary_pubmedqa_delta"],
            "primary_branch": A["primary_branch"],
            "oracle_pubmedqa_delta": A["oracle_pubmedqa_delta"],
            "logreg_pubmedqa_delta": A["reads"]["logreg_scalar_banked"]["per_subset"]["pubmedqa"]["delta"],
            "readout_pubmedqa_delta": READOUT_PUBMEDQA_DELTA,
            "best_scalar_blind_mean": max(A["reads"][n]["blind_mean_auc"] for n in A["reads"]
                                          if n not in ("readout_banked_seed0",)),
            "readout_blind_mean": A["reads"]["readout_banked_seed0"]["blind_mean_auc"],
            "hardmax_blind_mean": BANKED_MEAN,
            "verdict": ("the autopsy's EMBEDDING conclusion survives the capacity-matched control"
                        if A["primary_branch"] == "EMBEDDING" and A["oracle_branch"] == "EMBEDDING"
                        else "capacity, not embedding content - the autopsy's conclusion is overturned"),
        },
        "control_b": {
            "pubmedqa_delta_per_seed": pub["per_seed"],
            "pubmedqa_delta_mean": pub["mean"],
            "pubmedqa_delta_std": pub["std"],
            "pubmedqa_delta_range": pub["range"],
            "banked_seed0_pubmedqa_delta": READOUT_PUBMEDQA_DELTA,
            "hotpotqa_delta_mean": sp["hotpotqa"]["mean"],
            "hotpotqa_delta_range": sp["hotpotqa"]["range"],
            "tatqa_delta_mean": sp["tatqa"]["mean"],
            "tatqa_delta_range": sp["tatqa"]["range"],
            "blind_mean_delta_mean": B["blind_mean_delta"]["mean"],
            "blind_mean_delta_std": B["blind_mean_delta"]["std"],
            "pubmedqa_range_vs_campaign_swing": (
                f"pubmedqa seed range {pub['range']:.4f} vs campaign per-subset swing "
                f"{CAMPAIGN_SUBSET_SWING[0]}-{CAMPAIGN_SUBSET_SWING[1]}"),
            "lift_survives_all_seeds": bool(min(pub["per_seed"]) > 0),
        },
        "bars_recorded_not_adjudicated": True,
    }

    payload = {
        "gate": "R16-H141 review-ordered controls A (capacity-matched scalar model) and "
                "B (readout seed noise)",
        "context": "controls on the R16-H141 autopsy, which concluded EMBEDDING from a logistic "
                   "regression on 8 window-score scalars recovering +0.0092 of the readout's "
                   "+0.0711 pubmedqa delta",
        "checkpoint": json.loads(PILOT.read_text())["checkpoint"],
        "cache": "R16-H140_cache (existing) - no embedding recomputation",
        "device": DEV, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "protocol": "per-sentence score, hard MIN over the response's sentences, response-level "
                    "AUROC per subset - unchanged from the campaign",
        "discipline": "every fit and every selection (MLP epoch, smooth-agg temperature) uses "
                      "training-slice rows only; the arena is read once per model for final "
                      "numbers and never for tuning",
        "control_a": A,
        "control_b": B,
        "summary": summary,
        "caveats": [
            "Control A drops the readout's attention head - a single per-sentence scalar vector has "
            "no set to pool over. The arch-matched model therefore carries fewer parameters than the "
            "readout (the 769->256 projection is the bulk of the readout's count); the param-matched "
            "h=404 model restores the parameter count and is the capacity control proper.",
            "The two window-count features are fitted on training window sets of at most 14 and read "
            "on arena sets up to 156; a nonlinear model extrapolates there far more aggressively "
            "than a linear one. The 6-feature ablation drops them and bounds that risk.",
            "Control B rebinds only the seed. The seed drives the train/val permutation, the batch "
            "shuffle and the torch initialisation together, so the spread reported is total "
            "training-run noise, not initialisation noise alone.",
            "4 seeds give a coarse spread; the std is reported with ddof=1 and should be read as an "
            "order of magnitude, not a confidence interval.",
            "The seed-0 rerun is a harness check only. Any drift from the banked pilot is GPU "
            "nondeterminism, and the banked value is the one carried into the aggregate.",
            "Bars are RECORDED, not adjudicated here - the coordinator adjudicates.",
        ],
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\n  -> {OUT}   ({payload['runtime_sec']:.0f}s total)")
    print("=== R16-H141 CONTROLS DONE ===", flush=True)


if __name__ == "__main__":
    main()

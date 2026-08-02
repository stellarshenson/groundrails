"""R8-H93 - DANN lambda geometry under a legal LOCO objective (Optuna TPE).

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 8).

Hold ALL of HaluEval out of training; maximise AUC on it. In-domain validation
is disqualified (four recorded in-domain/blind dissociations); RAGBench is
disqualified (blind). Trials are ~60k-pair subsamples of the full mix, bf16
autocast, one epoch each, on GPU idx0 (RTX PRO 4000, 24 GB) - the study
compares trial-relative geometry, not absolute quality. Trial 0 is forced
lambda ~ 0 (ERM) as the in-study baseline; the pre-registered bar is
LOCO(lambda*) - LOCO(ERM) >= +0.02 with domain-acc within [0.5x, 1.5x] chance.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/R8-H93_lambda_sweep.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import importlib.util
import json
import pathlib
import time

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

HERE = pathlib.Path(__file__).parent
DB = HERE / "R8-H93_optuna.db"
TRIALS_LOG = HERE / "R8-H93_trials.json"

STUDENT = "jhu-clsp/mmBERT-base"
MAX_LEN = 512
BATCH = 16
LR = 1e-5
WARMUP_FRAC, CLIP = 0.1, 1.0
SEED = 0
N_TRAIN = 60_000  # per-trial subsample of the non-HaluEval mix
N_VAL = 4_000  # held-out HaluEval pairs scored per trial
N_TRIALS = 11  # trial 0 = forced ERM baseline + 10 TPE trials


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H90 = _mod("h90", "R8-H90_dann_full.py")


def build_data():
    """Full mix once; HaluEval split out as the LOCO validation corpus."""
    pc, pk, py, ptags = H90.private_train()
    uc, uk, uy, utags = H90.public_train()
    claims = pc + uc
    chunks = pk + uk
    y = np.concatenate([py, uy])
    tags = np.array(ptags + utags)

    is_val = tags == "halueval"
    rng = np.random.default_rng(SEED)

    vi = np.flatnonzero(is_val)
    rng.shuffle(vi)
    vi = vi[:N_VAL]
    val = ([claims[i] for i in vi], [chunks[i] for i in vi], y[vi])

    ti = np.flatnonzero(~is_val)
    rng.shuffle(ti)
    ti = ti[:N_TRAIN]
    tr_tags = tags[ti]
    tag_names = sorted(set(tr_tags.tolist()))
    tag_to_idx = {t: i for i, t in enumerate(tag_names)}
    groups = np.array([tag_to_idx[t] for t in tr_tags])
    train = ([claims[i] for i in ti], [chunks[i] for i in ti], y[ti], groups)
    return train, val, len(tag_names)


@torch.inference_mode()
def loco_auc(model, tok, val):
    claims, chunks, y = val
    probs = np.zeros(len(y), dtype=np.float32)
    for i in range(0, len(y), 64):
        enc = tok(
            claims[i : i + 64],
            chunks[i : i + 64],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cls = model.trunk(**enc).last_hidden_state[:, 0]
            probs[i : i + 64] = (
                torch.sigmoid(model.task_head(cls.float()).squeeze(-1)).float().cpu().numpy()
            )
    auc, _, _ = H90.M59.auc_and_f1((y > 0.5).astype(int), probs)
    return auc


def run_trial(lam_max, hidden, train, val, n_groups, tag):
    torch.manual_seed(SEED)
    claims, chunks, y, groups = train
    tok = AutoTokenizer.from_pretrained(STUDENT)
    base = AutoModel.from_pretrained(STUDENT).cuda()
    model = H90.DANNStudent(base, n_groups, hidden=hidden).cuda()

    ds = H90.GroupSet(claims, chunks, y, groups, tok)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=ds.collate, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    n_steps = len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=n_steps, pct_start=WARMUP_FRAC, anneal_strategy="linear"
    )
    task_lossf = torch.nn.BCEWithLogitsLoss()
    domain_lossf = torch.nn.CrossEntropyLoss()
    chance = 1.0 / n_groups

    model.train()
    t0 = time.time()
    dom_correct, dom_total, last_acc = 0, 0, 0.0
    for step, (enc, yy, gg) in enumerate(dl):
        enc = {k: v.cuda() for k, v in enc.items()}
        yy, gg = yy.cuda(), gg.cuda()
        p = step / max(n_steps - 1, 1)
        lam = lam_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            task_logit, domain_logit = model(enc, lam)
            t_loss = task_lossf(task_logit.float(), yy)
            d_loss = domain_lossf(domain_logit.float(), gg)
            loss = t_loss + d_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        sched.step()
        opt.zero_grad()
        dom_correct += (domain_logit.argmax(-1) == gg).sum().item()
        dom_total += len(gg)
        if not torch.isfinite(loss):
            raise RuntimeError(f"diverged at step {step}")
        if step % 500 == 0:
            last_acc = dom_correct / max(dom_total, 1)
            dom_correct, dom_total = 0, 0
            print(
                f"    [{tag}] step {step}/{n_steps} task {t_loss.item():.4f} "
                f"domain {d_loss.item():.4f} lam {lam:.4f} dom-acc {last_acc:.3f} "
                f"(chance {chance:.3f}) ({time.time() - t0:.0f}s)",
                flush=True,
            )
    final_acc = dom_correct / max(dom_total, 1) if dom_total else last_acc

    model.eval()
    auc = loco_auc(model, tok, val)
    del model, base
    torch.cuda.empty_cache()
    return auc, final_acc, chance


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print("assembling data (once)...", flush=True)
    train, val, n_groups = build_data()
    print(
        f"train {len(train[2])} pairs / {n_groups} groups (halueval OUT), "
        f"val {len(val[2])} halueval pairs\n",
        flush=True,
    )

    results = []

    def objective(trial):
        if trial.number == 0:
            lam_max = 0.0  # forced ERM baseline on the identical subsample
            hidden = 256
        else:
            lam_max = trial.suggest_float("lam_max", 0.003, 0.15, log=True)
            hidden = trial.suggest_categorical("hidden", [128, 256, 512])
        tag = f"t{trial.number} lam {lam_max:.4f} h{hidden}"
        print(f"\n=== trial {trial.number}: lam_max {lam_max:.4f} hidden {hidden}", flush=True)
        auc, dom_acc, chance = run_trial(lam_max, hidden, train, val, n_groups, tag)
        rec = {
            "trial": trial.number,
            "lam_max": lam_max,
            "hidden": hidden,
            "loco_auc": round(auc, 4),
            "final_domain_acc": round(dom_acc, 4),
            "chance": round(chance, 4),
        }
        results.append(rec)
        TRIALS_LOG.write_text(json.dumps(results, indent=2))
        print(
            f"=== trial {trial.number} DONE  loco_auc {auc:.4f}  dom-acc {dom_acc:.3f} "
            f"(chance {chance:.3f})",
            flush=True,
        )
        return auc

    study = optuna.create_study(
        study_name="R8-H93",
        storage=f"sqlite:///{DB}",
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS)

    base_auc = next(r["loco_auc"] for r in results if r["trial"] == 0)
    best = study.best_trial
    print("\n" + "=" * 80)
    print("R8-H93 RESULT - lambda geometry under LOCO(HaluEval)")
    print("=" * 80)
    print(f"  ERM baseline (lam 0)  loco_auc {base_auc:.4f}")
    print(f"  best trial {best.number}  params {best.params}  loco_auc {best.value:.4f}")
    print(f"  lift {best.value - base_auc:+.4f}  (bar: >= +0.02)")
    print(f"\n  results -> {TRIALS_LOG}")


if __name__ == "__main__":
    main()

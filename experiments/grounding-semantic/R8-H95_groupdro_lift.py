"""R8-H95 - lift-all-groups GroupDRO: stage 1 of the curriculum fanout.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 8).

H81's q-collapse came from unsmoothed exponentiated-gradient fixation (eta
0.01, two groups took 97% of the weight, ten starved). This variant bounds
every group's weight from below by construction - uniform smoothing
q <- (1-alpha)q + alpha/n with alpha 0.2, eta 0.003 - on the FULL ~762k mix
(TabFact included), so the objective lifts ALL groups rather than the two
hardest seen ones.

Judged standalone AND as stage 1 of R8-H96 (the GroupDRO -> DANN phase shift):
the checkpoint saves the bare trunk alongside the classifier so stage 2 can
mount a fresh DANN discriminator on a domain-mastered backbone.

Pre-registered instrumentation:
  - per-group validation: 2,000 held-out pairs per group, sampled from train
    rows before training (arena untouched); AUC per group every EVAL_EVERY steps
  - plateau rule: no group improves > 0.003 for 3 consecutive evals, hard cap
    1.5 epochs - the phase-shift boundary of H96
  - bar: >= 12 of 13 groups improve over training; blind decomposed-min read
    lands >= the H91 ERM control (H81 landed 0.035 BELOW its ERM twin)

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R8-H95_groupdro_lift.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import importlib.util
import json
import pathlib
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = pathlib.Path(__file__).parent
CKPT_DIR = HERE.parent.parent / "models" / "R8-H95-mmbert-groupdro-lift"
OUT = HERE / "R8-H95_result.json"

STUDENT = "jhu-clsp/mmBERT-base"
MAX_LEN = 512
BATCH = 52  # 13 groups x 4 per group per step
LR = 1e-5
WARMUP_FRAC, CLIP = 0.1, 1.0
WEIGHT_DECAY = 0.01  # GroupDRO's regularisation requirement, as in H81
SEED = 0

ETA_Q = 0.003  # v2 lever: 3.3x gentler than H81's 0.01
SMOOTH_ALPHA = 0.2  # v2 lever: q <- (1-a)q + a/n bounds every group at a/n
N_VAL_PER_GROUP = 2_000
EVAL_EVERY = 2_000
PLATEAU_EPS = 0.003
PLATEAU_PATIENCE = 3
EPOCH_CAP = 1.5


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H90 = _mod("h90", "R8-H90_dann_full.py")  # full-mix loaders, M59/M60
H81 = _mod("h81", "R8-H81_groupdro.py")  # GroupSet, StratifiedGroupSampler


@torch.inference_mode()
def group_val_aucs(model, tok, val_by_group, tag_names):
    model.eval()
    aucs = {}
    for g, (vc, vk, vy) in val_by_group.items():
        probs = np.zeros(len(vy), dtype=np.float32)
        for i in range(0, len(vy), 64):
            enc = tok(
                vc[i : i + 64],
                vk[i : i + 64],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
            )
            enc = {k: v.cuda() for k, v in enc.items()}
            probs[i : i + 64] = (
                torch.sigmoid(model(**enc).logits.float().squeeze(-1)).cpu().numpy()
            )
        yb = (np.asarray(vy) > 0.5).astype(int)
        if yb.min() == yb.max():  # degenerate slice, skip rather than crash
            continue
        auc, _, _ = H90.M59.auc_and_f1(yb, probs)
        aucs[tag_names[g]] = round(auc, 4)
    model.train()
    return aucs


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    pc, pk, py, ptags = H90.private_train()
    uc, uk, uy, utags = H90.public_train()
    claims, chunks = pc + uc, pk + uk
    y = np.concatenate([py, uy])
    tag_names = sorted(set(ptags + utags))
    tag_to_idx = {t: i for i, t in enumerate(tag_names)}
    groups = np.array([tag_to_idx[t] for t in ptags + utags])
    n_groups = len(tag_names)

    # Per-group held-out validation, carved out BEFORE training.
    rng = np.random.default_rng(SEED)
    val_by_group, train_mask = {}, np.ones(len(y), dtype=bool)
    for g in range(n_groups):
        gi = np.flatnonzero(groups == g)
        rng.shuffle(gi)
        vi = gi[:N_VAL_PER_GROUP]
        train_mask[vi] = False
        val_by_group[g] = (
            [claims[i] for i in vi],
            [chunks[i] for i in vi],
            y[vi],
        )
    ti = np.flatnonzero(train_mask)
    t_claims = [claims[i] for i in ti]
    t_chunks = [chunks[i] for i in ti]
    t_y, t_groups = y[ti], groups[ti]
    print(
        f"train {len(t_y)} pairs / {n_groups} groups, val {n_groups}x{N_VAL_PER_GROUP}, "
        f"mean target {t_y.mean():.3f}\n",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(STUDENT)
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT, num_labels=1, ignore_mismatched_sizes=True
    ).cuda()
    n_par = sum(p.numel() for p in model.parameters()) / 1e6

    ds = H81.GroupSet(t_claims, t_chunks, t_y, t_groups, tok)
    sampler = H81.StratifiedGroupSampler(t_groups, BATCH, n_groups, SEED)
    max_steps = int(EPOCH_CAP * len(t_y) / sampler.batch)
    print(f"student {n_par:.1f}M  batch {sampler.batch}  cap {max_steps} steps\n", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=max_steps, pct_start=WARMUP_FRAC, anneal_strategy="linear"
    )
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")
    q = torch.full((n_groups,), 1.0 / n_groups).cuda()

    history, best_auc, stall = [], {}, 0
    model.train()
    t0 = time.time()
    gen = sampler.batches(max_steps)
    stopped_at = max_steps
    for step in range(max_steps):
        idx = next(gen)
        batch = [ds[int(i)] for i in idx]
        enc, yy, gg = ds.collate(batch)
        enc = {k: v.cuda() for k, v in enc.items()}
        yy, gg = yy.cuda(), gg.cuda()
        per_loss = lossf(model(**enc).logits.squeeze(-1), yy)

        group_loss = torch.zeros(n_groups).cuda()
        for g in range(n_groups):
            m = gg == g
            if m.any():
                group_loss[g] = per_loss[m].mean()
        loss = (q.detach() * group_loss).sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        sched.step()
        opt.zero_grad()

        # Exponentiated-gradient update WITH uniform smoothing: every group's
        # weight is bounded below by SMOOTH_ALPHA / n_groups by construction.
        q = q * torch.exp(ETA_Q * group_loss.detach())
        q = q / q.sum()
        q = (1.0 - SMOOTH_ALPHA) * q + SMOOTH_ALPHA / n_groups

        if not torch.isfinite(loss):
            raise RuntimeError(f"diverged at step {step}")
        if step % 200 == 0:
            print(
                f"  step {step}/{max_steps} worst {group_loss.max().item():.4f} "
                f"q_max {q.max().item():.3f} q_min {q.min().item():.3f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

        if step > 0 and step % EVAL_EVERY == 0:
            aucs = group_val_aucs(model, tok, val_by_group, tag_names)
            history.append({"step": step, "aucs": aucs})
            improved = [t for t, a in aucs.items() if a > best_auc.get(t, 0.0) + PLATEAU_EPS]
            for t, a in aucs.items():
                best_auc[t] = max(best_auc.get(t, 0.0), a)
            stall = 0 if improved else stall + 1
            print(
                f"  EVAL step {step}: mean {np.mean(list(aucs.values())):.4f} "
                f"improved {len(improved)}/{len(aucs)} stall {stall}/{PLATEAU_PATIENCE}  "
                f"{aucs}",
                flush=True,
            )
            if stall >= PLATEAU_PATIENCE:
                print(f"\nPLATEAU at step {step} - stopping stage 1", flush=True)
                stopped_at = step
                break

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CKPT_DIR)
    tok.save_pretrained(CKPT_DIR)
    # Bare trunk for R8-H96's phase shift (fresh DANN heads mount on this).
    model.model.save_pretrained(CKPT_DIR / "trunk")
    print(f"\ncheckpoint saved -> {CKPT_DIR} (trunk/ exported for H96)\n", flush=True)

    final_aucs = group_val_aucs(model, tok, val_by_group, tag_names)
    history.append({"step": stopped_at, "aucs": final_aucs, "final": True})
    first = history[0]["aucs"] if history else {}
    lifted = [t for t, a in final_aucs.items() if a >= first.get(t, 0.0)]

    res = {
        "params_M": round(n_par, 1),
        "stopped_at_step": stopped_at,
        "max_steps": max_steps,
        "eta_q": ETA_Q,
        "smooth_alpha": SMOOTH_ALPHA,
        "final_q": {tag_names[g]: round(q[g].item(), 4) for g in range(n_groups)},
        "per_group_val_history": history,
        "groups_lifted_vs_first_eval": lifted,
    }
    OUT.write_text(json.dumps(res, indent=2))

    print("=" * 96)
    print("R8-H95 RESULT - lift-all-groups GroupDRO (stage 1)")
    print("=" * 96)
    print(f"  stopped {stopped_at}/{max_steps}  q_max {max(res['final_q'].values()):.3f}")
    print(f"  groups lifted vs first eval: {len(lifted)}/{n_groups} -> {lifted}")
    print(f"  final per-group val AUC: {final_aucs}")
    print("  blind arena: score via R8-H77 --model (plain branch) + decomposed-min read")
    print(f"\n  results -> {OUT}")


if __name__ == "__main__":
    main()

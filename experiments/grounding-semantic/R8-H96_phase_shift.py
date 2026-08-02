"""R8-H96 - the phase shift: DANN stage 2 on the GroupDRO-mastered trunk.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 8).

The author's curriculum hypothesis: a model generalises FROM well-understood
domains. Stage 1 (R8-H95) trains lift-all-groups GroupDRO to plateau; this
stage loads that trunk, mounts a FRESH task head and a FRESH N-way domain
discriminator, restarts the GRL ramp, and trains one full epoch of DANN over
the identical mix. Invariance imposed on features that already encode every
domain's task boundary should remove domain identity WITHOUT removing task
signal - where DANN-from-scratch (H79 v1) inverted it instead.

Lambda: the R8-H93 sweep winner if its trials log exists, else the registered
0.02 fallback. Kill condition (recorded, not auto-killed): domain-acc < 0.02
at half-ramp on a mastered trunk means inversion is not a curriculum problem
and the DANN lever escalates to v3 (stronger discriminator).

Bar: blind decomposed-min read beats the best single-stage single-model read
at run date by >= +0.01, discriminator parked in [0.5x, 1.5x] of chance.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R8-H96_phase_shift.py
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
from transformers import AutoModel, AutoTokenizer

HERE = pathlib.Path(__file__).parent
STAGE1 = HERE.parent.parent / "models" / "R8-H95-mmbert-groupdro-lift"
CKPT_DIR = HERE.parent.parent / "models" / "R8-H96-mmbert-phase-shift"
H93_TRIALS = HERE / "R8-H93_trials.json"
OUT = HERE / "R8-H96_result.json"

MAX_LEN = 512
BATCH = 48
LR = 1e-5
WARMUP_FRAC, CLIP = 0.1, 1.0
SEED = 0
N_VAL_PER_GROUP = 2_000  # identical carve-out to H95 (same rng, same order)
EVAL_EVERY = 2_000
LAMBDA_FALLBACK = 0.02
DANN_HIDDEN = 256


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H90 = _mod("h90", "R8-H90_dann_full.py")  # loaders, DANNStudent, GroupSet, M59/M60, evaluate
H95 = _mod("h95", "R8-H95_groupdro_lift.py")  # group_val_aucs


def pick_lambda():
    if H93_TRIALS.exists():
        trials = json.loads(H93_TRIALS.read_text())
        swept = [t for t in trials if t["trial"] > 0]
        if swept:
            best = max(swept, key=lambda t: t["loco_auc"])
            base = next((t for t in trials if t["trial"] == 0), None)
            if base and best["loco_auc"] > base["loco_auc"]:
                print(
                    f"lambda from H93 winner: {best['lam_max']:.4f} "
                    f"(loco {best['loco_auc']:.4f} vs ERM {base['loco_auc']:.4f})",
                    flush=True,
                )
                return best["lam_max"], best.get("hidden", DANN_HIDDEN)
    print(f"lambda fallback: {LAMBDA_FALLBACK}", flush=True)
    return LAMBDA_FALLBACK, DANN_HIDDEN


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    lam_max, hidden = pick_lambda()

    pc, pk, py, ptags = H90.private_train()
    uc, uk, uy, utags = H90.public_train()
    claims, chunks = pc + uc, pk + uk
    y = np.concatenate([py, uy])
    tag_names = sorted(set(ptags + utags))
    tag_to_idx = {t: i for i, t in enumerate(tag_names)}
    groups = np.array([tag_to_idx[t] for t in ptags + utags])
    n_groups = len(tag_names)
    chance = 1.0 / n_groups

    # Identical val carve-out to H95: same seed, same per-group order.
    rng = np.random.default_rng(SEED)
    val_by_group, train_mask = {}, np.ones(len(y), dtype=bool)
    for g in range(n_groups):
        gi = np.flatnonzero(groups == g)
        rng.shuffle(gi)
        vi = gi[:N_VAL_PER_GROUP]
        train_mask[vi] = False
        val_by_group[g] = ([claims[i] for i in vi], [chunks[i] for i in vi], y[vi])
    ti = np.flatnonzero(train_mask)
    t_claims = [claims[i] for i in ti]
    t_chunks = [chunks[i] for i in ti]
    t_y, t_groups = y[ti], groups[ti]
    print(
        f"train {len(t_y)} pairs / {n_groups} groups (chance {chance:.3f}), "
        f"lambda_max {lam_max:.4f} hidden {hidden}\n",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(str(STAGE1))
    base = AutoModel.from_pretrained(str(STAGE1 / "trunk")).cuda()
    model = H90.DANNStudent(base, n_groups, hidden=hidden).cuda()
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"stage-1 trunk loaded from {STAGE1 / 'trunk'}  {n_par:.1f}M params\n", flush=True)

    ds = H90.GroupSet(t_claims, t_chunks, t_y, t_groups, tok)
    from torch.utils.data import DataLoader

    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=ds.collate, num_workers=2)
    n_steps = len(dl)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=n_steps, pct_start=WARMUP_FRAC, anneal_strategy="linear"
    )
    task_lossf = torch.nn.BCEWithLogitsLoss()
    domain_lossf = torch.nn.CrossEntropyLoss()

    # group_val_aucs expects a seq-classification-like callable; adapt the
    # DANN student behind a tiny shim with .eval/.train and (**enc).logits.
    class _Shim:
        def __init__(self, m):
            self.m = m

        def __call__(self, **enc):
            cls = self.m.trunk(**enc).last_hidden_state[:, 0]
            out = type("O", (), {})()
            out.logits = self.m.task_head(cls)
            return out

        def eval(self):
            self.m.eval()

        def train(self):
            self.m.train()

    shim = _Shim(model)

    history = []
    model.train()
    t0 = time.time()
    dom_correct, dom_total = 0, 0
    half_ramp_acc = None
    for step, (enc, yy, gg) in enumerate(dl):
        enc = {k: v.cuda() for k, v in enc.items()}
        yy, gg = yy.cuda(), gg.cuda()
        p = step / max(n_steps - 1, 1)
        lam = lam_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

        task_logit, domain_logit = model(enc, lam)
        t_loss = task_lossf(task_logit, yy)
        d_loss = domain_lossf(domain_logit, gg)
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
        if step % 200 == 0:
            acc = dom_correct / max(dom_total, 1)
            dom_correct, dom_total = 0, 0
            if half_ramp_acc is None and p >= 0.5:
                half_ramp_acc = acc
                if acc < 0.02:
                    print(
                        f"  KILL-CONDITION NOTE: domain-acc {acc:.3f} < 0.02 at half-ramp "
                        "- anti-prediction on a mastered trunk (recorded, run continues)",
                        flush=True,
                    )
            print(
                f"  step {step}/{n_steps} task {t_loss.item():.4f} domain {d_loss.item():.4f} "
                f"lam {lam:.4f}  domain-acc {acc:.3f} (chance {chance:.3f})  "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
        if step > 0 and step % EVAL_EVERY == 0:
            aucs = H95.group_val_aucs(shim, tok, val_by_group, tag_names)
            history.append({"step": step, "aucs": aucs})
            print(
                f"  EVAL step {step}: per-group val mean {np.mean(list(aucs.values())):.4f}",
                flush=True,
            )
            model.train()

    final_acc = dom_correct / max(dom_total, 1) if dom_total else 0.0

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trunk": base.state_dict(),
            "task_head": model.task_head.state_dict(),
            "domain_head": model.domain_head.state_dict(),
            "config": base.config,
        },
        CKPT_DIR / "dann_student.pt",
    )
    tok.save_pretrained(CKPT_DIR)
    base.save_pretrained(CKPT_DIR / "trunk")
    print(f"\ncheckpoint saved -> {CKPT_DIR}\n", flush=True)

    model.eval()
    res = H90.evaluate(model, tok)
    res["params_M"] = round(n_par, 1)
    res["lambda_max"] = lam_max
    res["hidden"] = hidden
    res["final_domain_acc"] = round(final_acc, 4)
    res["half_ramp_domain_acc"] = half_ramp_acc
    res["chance"] = round(chance, 4)
    res["per_group_val_history"] = history

    print("=" * 96)
    print("R8-H96 RESULT - phase shift (GroupDRO -> DANN), one sub-400M model")
    print("=" * 96)
    for key, (bar, decisive) in H90.BARS.items():
        a = res[key]["auc"]
        mark = "DECISIVE" if a >= decisive else ("beat" if a > bar else "LOSE")
        print(f"  {key:16s} {a:.4f}  (beat {bar}, decisive {decisive})  {mark}")
    print(
        f"  final domain-acc {final_acc:.3f} (chance {chance:.3f}, healthy band "
        f"[{0.5 * chance:.3f}, {1.5 * chance:.3f}])"
    )
    print("  blind arena: R8-H77 --model (dann branch) + decomposed-min read")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\n  results -> {OUT}")


if __name__ == "__main__":
    main()

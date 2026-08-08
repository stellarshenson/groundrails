"""R13-H129 - ENSEMBLE-OUTPUT-DISTILLATION training draws.

The clean R9-H105 recipe (mmBERT-base cross-encoder, DANN lambda 0.02 Ganin
ramp, MAX_LEN 512, BATCH 48, LR 1e-5, OneCycleLR, 1 epoch) on the PUBLIC MIX
ONLY (685,670 rows, 12 DANN groups - no DR lane, no H108 lane), with the task
loss replaced by the registered distillation form:

    task = 0.5 * BCE_with_logits(logit, hard_label)
         + 0.5 * MSE(sigmoid(logit), p_teacher)

`p_teacher` is the output-probability mean of the two frozen H105 draws, banked
by `R13-H129_targets.py` as `R13-H129_teacher_targets.parquet` (row_id is the
POSITIONAL index into the `public_train()` build order - the mix has no
materialized parquet). Per the registered trainer contract the alignment is
asserted before the targets are consumed: 1,000 random rows (seed 0) must
reproduce `key_hash` = blake2b-64(claim + NUL + chunk); a mismatch aborts.

H126 seeding facility (session ruling 8): `torch.manual_seed(seed)` is issued
BEFORE model construction (so trunk + task_head init is bit-identical across
paired arms) and RE-ISSUED immediately AFTER construction, before any dropout
or forward, so the training RNG stream does not inherit whatever the
construction consumed. The initial trunk+task_head state is fingerprinted into
`init_fingerprint.json` in the checkpoint dir for the paired-arm assertion.

R12-H120 instruments (this trainer may host the H120 read):
  --ema     one EMA buffer (decay 0.999) over trunk + task_head, started at 80%
            of total steps, saved as a SEPARATE checkpoint dir
            models/R13-H129-draw<N>-ema/. Off by default.
  step-cosine  ALWAYS on: running mean cosine of consecutive parameter updates,
            cos(W_t - W_{t-1}, W_{t-1} - W_{t-2}), measured over the final 20%
            of steps on one fixed parameter slice (the largest non-embedding
            parameter tensor, selected deterministically and named in the log
            and in the result json). Logged every 200 steps; the H120 read is
            LICENSEd below 0.3 and ABORTed above 0.5.

Bars (registered): ADMIT pair mean >= 0.7091 with both draws >= control 0.7031;
hold gold_full >= 0.84, no subset < 0.55, none more than 0.06 below control;
KILL pair mean <= 0.7031.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R13-H129_trainer.py --draw 1
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import argparse
import hashlib
import importlib.util
import json
import pathlib
import time

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModel, AutoTokenizer

HERE = pathlib.Path(__file__).parent
TARGETS = HERE / "R13-H129_teacher_targets.parquet"

MAX_LEN, BATCH, LR = 512, 48, 1e-5
WARMUP_FRAC, CLIP = 0.1, 1.0
LAMBDA_MAX = 0.02
RESUME_EVERY = 1000
DISTILL_W = 0.5  # 0.5 * BCE + 0.5 * MSE, registered
EMA_DECAY = 0.999
EMA_START_FRAC = 0.8  # EMA and the step-cosine window both open at 80%
COS_FRAC = 0.8
ALIGN_SAMPLE, ALIGN_SEED = 1000, 0
DRAW_SEEDS = {1: 3129, 2: 4129}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H108 = _mod("h108", "R10-H108_lane.py")


def row_hash(claim, chunk):
    """blake2b-64 of claim + NUL + chunk - byte-identical to R13-H129_gate.row_hash."""
    h = hashlib.blake2b(digest_size=8)
    h.update(claim.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(chunk.encode("utf-8", "replace"))
    return h.hexdigest()


def teacher_targets(claims, chunks):
    """Load p_teacher positionally and assert alignment on a 1,000-row sample."""
    df = pl.read_parquet(TARGETS, columns=["row_id", "key_hash", "p_teacher"]).sort("row_id")
    n = len(claims)
    if df.height != n:
        raise SystemExit(
            f"ALIGNMENT ABORT: {TARGETS.name} has {df.height} rows, mix has {n}. "
            "The teacher targets were built against a different public_train() build."
        )
    if df["row_id"].to_numpy().tolist() != list(range(n)):
        raise SystemExit(f"ALIGNMENT ABORT: {TARGETS.name} row_id is not contiguous 0..{n - 1}.")

    keys = df["key_hash"].to_list()
    rng = np.random.default_rng(ALIGN_SEED)
    sample = rng.choice(n, size=ALIGN_SAMPLE, replace=False)
    bad = [i for i in sample.tolist() if keys[i] != row_hash(claims[i], chunks[i])]
    if bad:
        raise SystemExit(
            f"ALIGNMENT ABORT: {len(bad)}/{ALIGN_SAMPLE} sampled rows have a key_hash "
            f"mismatch (first offending row_id {bad[0]}). The trainer's public_train() "
            "build order differs from the one the teacher targets were scored on - "
            "rebuild the targets before training."
        )
    print(f"teacher targets aligned: {n} rows, {ALIGN_SAMPLE}/{ALIGN_SAMPLE} key_hash match "
          f"(seed {ALIGN_SEED})", flush=True)
    return df["p_teacher"].to_numpy().astype("float32")


class MixSet(Dataset):
    def __init__(self, claims, chunks, y, groups, p_teacher, tok):
        self.c, self.k, self.y, self.g, self.t, self.tok = claims, chunks, y, groups, p_teacher, tok

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.c[i], self.k[i], self.y[i], self.g[i], self.t[i]

    def collate(self, b):
        c, k, y, g, t = zip(*b, strict=True)
        enc = self.tok(list(c), list(k), return_tensors="pt", padding=True,
                       truncation=True, max_length=MAX_LEN)
        return enc, torch.tensor(y), torch.tensor(g), torch.tensor(t)


def ema_named(model):
    """The parameters the EMA buffer and the init fingerprint cover: trunk + task_head."""
    return [(n, p) for n, p in model.named_parameters()
            if n.startswith(("trunk.", "task_head."))]


def init_fingerprint(model):
    """blake2b-128 over trunk + task_head parameter bytes in sorted name order."""
    h = hashlib.blake2b(digest_size=16)
    n_par = 0
    for name, p in sorted(ema_named(model), key=lambda kv: kv[0]):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
        n_par += p.numel()
    return h.hexdigest(), n_par


def cosine_slice(model):
    """Fixed, deterministic parameter slice for the step-cosine instrument: the
    largest non-embedding parameter tensor (ties broken by name). Embeddings are
    excluded - their gradients are token-sparse, so their update direction is
    dominated by which rows happened to appear in the batch."""
    cands = sorted(
        ((p.numel(), n) for n, p in model.named_parameters() if "embed" not in n.lower()),
        key=lambda t: (-t[0], t[1]),
    )
    name = cands[0][1]
    return name, dict(model.named_parameters())[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", type=int, required=True, choices=(1, 2))
    ap.add_argument("--ema", action="store_true",
                    help="maintain the H120 EMA buffer and save models/R13-H129-draw<N>-ema/")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="probe mode: stop after N steps, no checkpoint/eval")
    args = ap.parse_args()
    ckpt_dir = HERE.parent.parent / "models" / f"R13-H129-draw{args.draw}"
    ema_dir = HERE.parent.parent / "models" / f"R13-H129-draw{args.draw}-ema"
    out = HERE / f"R13-H129_draw{args.draw}_result.json"

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    seed = DRAW_SEEDS[args.draw]
    torch.manual_seed(seed)
    np.random.seed(seed)

    claims, chunks, y, tags = H108.public_train()
    p_teacher = teacher_targets(claims, chunks)

    tag_names = sorted(set(tags))
    tag_to_idx = {t: i for i, t in enumerate(tag_names)}
    groups = np.array([tag_to_idx[t] for t in tags])
    n_groups = len(tag_names)
    print(f"train: {len(y)} public rows across {n_groups} domains "
          f"(chance {1.0 / n_groups:.3f})  seed {seed}  "
          f"p_teacher mean {p_teacher.mean():.4f}\n", flush=True)

    tok = AutoTokenizer.from_pretrained(H108.STUDENT)
    base = AutoModel.from_pretrained(H108.STUDENT).cuda()
    base.config.reference_compile = False
    model = H108.DANNStudent(base, n_groups).cuda()
    # H126 / ruling 8: re-issue the seed AFTER construction, before any dropout
    # or forward, so the training stream is independent of construction draws.
    torch.manual_seed(seed)
    fp, fp_numel = init_fingerprint(model)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"student {H108.STUDENT} + DANN heads  {n_par:.1f}M params\n"
          f"init fingerprint (trunk+task_head, {fp_numel} params): {fp}\n", flush=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "init_fingerprint.json").write_text(json.dumps(
        {"draw": args.draw, "seed": seed, "n_groups": n_groups,
         "scope": "trunk+task_head", "n_params": fp_numel, "blake2b_128": fp}, indent=2))

    ds = MixSet(claims, chunks, y, groups, p_teacher, tok)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    n_steps = (len(ds) + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=n_steps, pct_start=WARMUP_FRAC, anneal_strategy="linear")

    cos_name, cos_par = cosine_slice(model)
    cos_start = int(COS_FRAC * n_steps)
    ema_start = int(EMA_START_FRAC * n_steps)
    print(f"step-cosine slice: {cos_name} ({cos_par.numel()} elems), window steps "
          f"{cos_start}..{n_steps}\nEMA: {'on' if args.ema else 'off'}"
          f"{f' (decay {EMA_DECAY}, from step {ema_start})' if args.ema else ''}\n", flush=True)

    resume_path = ckpt_dir / "resume.pt"
    start_step = 0
    ema = None
    cos_sum, cos_n = torch.zeros((), device="cuda"), 0
    if resume_path.exists():
        st = torch.load(resume_path, map_location="cuda", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        perm, start_step = st["perm"], st["step"]
        ema = st.get("ema")
        cos_sum += float(st.get("cos_sum", 0.0))
        cos_n = st.get("cos_n", 0)
        print(f"resumed from {resume_path} at step {start_step}/{n_steps} "
              f"(ema {'present' if ema else 'absent'}, cos_n {cos_n})\n", flush=True)
    else:
        perm = np.random.default_rng(seed).permutation(len(ds)).tolist()

    dl = DataLoader(Subset(ds, perm[start_step * BATCH:]), batch_size=BATCH,
                    shuffle=False, collate_fn=ds.collate, num_workers=2)
    domain_lossf = nn.CrossEntropyLoss()

    def save_resume(step):
        tmp = resume_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "perm": perm, "step": step,
                    "ema": ema, "cos_sum": cos_sum.item(), "cos_n": cos_n}, tmp)
        tmp.replace(resume_path)

    def save_ckpt(d, trunk_state, head_state):
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"trunk": trunk_state, "task_head": head_state,
                    "domain_head": model.domain_head.state_dict(), "config": base.config},
                   d / "dann_student.pt")
        tok.save_pretrained(d)
        base.save_pretrained(d / "trunk")

    model.train()
    t0 = time.time()
    dom_correct, dom_total = 0, 0
    bce_last, mse_last = 0.0, 0.0
    prev_w, prev_delta = None, None
    for i, (enc, yy, gg, tt) in enumerate(dl):
        step = start_step + i
        if args.max_steps and i >= args.max_steps:
            print(f"probe stop at {args.max_steps} steps", flush=True)
            return
        enc = {k: v.cuda() for k, v in enc.items()}
        yy, gg, tt = yy.cuda(), gg.cuda(), tt.cuda()
        p = step / max(n_steps - 1, 1)
        lam = LAMBDA_MAX * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

        task_logit, domain_logit = model(enc, lam)
        bce = F.binary_cross_entropy_with_logits(task_logit, yy)
        mse = F.mse_loss(torch.sigmoid(task_logit), tt)
        t_loss = (1.0 - DISTILL_W) * bce + DISTILL_W * mse
        d_loss = domain_lossf(domain_logit, gg)
        loss = t_loss + d_loss
        bce_last, mse_last = bce.item(), mse.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        sched.step()
        opt.zero_grad()

        with torch.no_grad():
            if step >= cos_start:
                cur = cos_par.detach().flatten().float().clone()
                if prev_w is not None:
                    delta = cur - prev_w
                    if prev_delta is not None:
                        cos_sum += F.cosine_similarity(delta, prev_delta, dim=0)
                        cos_n += 1
                    prev_delta = delta
                prev_w = cur
            if args.ema and step >= ema_start:
                if ema is None:
                    ema = {n: q.detach().clone().float() for n, q in ema_named(model)}
                else:
                    for n, q in ema_named(model):
                        ema[n].mul_(EMA_DECAY).add_(q.detach().float(), alpha=1.0 - EMA_DECAY)

        dom_correct += (domain_logit.argmax(-1) == gg).sum().item()
        dom_total += len(gg)
        if not torch.isfinite(loss):
            raise RuntimeError(f"diverged at step {step}")
        if step % 200 == 0:
            acc = dom_correct / max(dom_total, 1)
            dom_correct, dom_total = 0, 0
            cm = (cos_sum / cos_n).item() if cos_n else float("nan")
            print(f"  step {step}/{n_steps} task {t_loss.item():.4f} "
                  f"(bce {bce_last:.4f} mse {mse_last:.4f}) domain {d_loss.item():.4f} "
                  f"lam {lam:.4f} domain-acc {acc:.3f} step-cos {cm:.4f} (n={cos_n})  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if step and step % RESUME_EVERY == 0:
            save_resume(step + 1)
            print(f"  resume point saved at step {step}", flush=True)

    save_ckpt(ckpt_dir, base.state_dict(), model.task_head.state_dict())
    resume_path.unlink(missing_ok=True)
    print(f"\ncheckpoint saved -> {ckpt_dir}\n", flush=True)

    if ema is not None:
        live = {n: q.detach().clone() for n, q in ema_named(model)}
        with torch.no_grad():
            for n, q in ema_named(model):
                q.copy_(ema[n].to(q.dtype))
        save_ckpt(ema_dir, base.state_dict(), model.task_head.state_dict())
        with torch.no_grad():
            for n, q in ema_named(model):
                q.copy_(live[n])
        print(f"EMA checkpoint saved -> {ema_dir}\n", flush=True)

    cos_mean = (cos_sum / cos_n).item() if cos_n else None
    model.eval()
    res = H108.evaluate(model, tok)
    res.update({
        "params_M": round(n_par, 1), "lambda_max": LAMBDA_MAX,
        "lane": "R13-H129 ensemble-output distillation (public mix only)",
        "draw": args.draw, "seed": seed, "distill_weight": DISTILL_W,
        "teacher_targets": TARGETS.name, "p_teacher_mean": round(float(p_teacher.mean()), 6),
        "mix_rows": len(y), "dann_groups": n_groups,
        "init_fingerprint": fp, "init_fingerprint_scope": "trunk+task_head",
        "step_cosine_mean": None if cos_mean is None else round(cos_mean, 4),
        "step_cosine_slice": cos_name, "step_cosine_n": cos_n,
        "ema": bool(args.ema), "ema_decay": EMA_DECAY if args.ema else None,
        "ema_checkpoint": str(ema_dir) if ema is not None else None,
        "train_seconds": round(time.time() - t0, 1), "checkpoint": str(ckpt_dir),
    })
    gf = res["gold_full"]
    print(f"gold_full {gf['auc']:.4f} (n={gf['n']})  "
          f"gold {res['gold']['auc']:.4f}  ragtruth_en {res['ragtruth_en']['auc']:.4f}")
    print(f"step-cosine mean over final 20%: {cos_mean} "
          f"(H120 read: LICENSE < 0.3, ABORT > 0.5)")
    out.write_text(json.dumps(res, indent=2))
    print(f"results -> {out}")
    print(f"=== H129 DRAW {args.draw} DONE ===", flush=True)


if __name__ == "__main__":
    main()

"""R11-H117 kill-gate-2 PROBE trainer - copy of DR_lane_trainer.py.

Differences from DR_lane_trainer.py (naming/args/reporting ONLY, training logic
byte-identical):
  - checkpoints go to models/H117-probe-lam<LAM>/ so the finished full control
    draws (models/DR-lane-draw{1,2}-control) can never be clobbered
  - --lambda-margin takes 0 for the probe-control arm (no --arm flag)
  - --max-steps saves a checkpoint before returning (the probe reads need it)
  - A6 instrument: running mean of (lambda_margin * hinge) / BCE logged every
    200 steps; > 0.25 voids the arm
  - the hinge is always computed (multiplied by lambda_margin = 0 in the control
    arm, so it contributes exactly zero to loss and gradient) to give the
    control arm a pair-separation diagnostic on the same code path

DR lane training draws - control (BCE-only) and margin (R11-H117) arms.

The exact R9-H105 recipe (mmBERT-base cross-encoder, BCE + DANN lambda 0.02
Ganin ramp, MAX_LEN 512, BATCH 48, LR 1e-5, OneCycleLR, 1 epoch) on
public_train() + the DR lane (DR_lane.parquet, assembled by DR_lane_assemble.py).

H117 binding amendments implemented here:
  A1  both arms train on the IDENTICAL row set - clean seed partners are present
      in the control too, as inert margin partners (BCE-masked, lambda_margin=0)
  A2/A3  paired seeds per draw index: control draw N and margin draw N share
      model init and permutation, so the H117 comparison is paired
  A5-A8  margin pairs are adjacent in the flat resume permutation and never
      straddle a 48-row batch boundary (singleton-swap packing); no second
      dataloader; margin hinge on sigmoid probs, m = 0.25, applied only to
      complete pairs in the batch; corrupt member asserted label 0 at assembly
  - clean partners carry the corrupt partner's DANN tag (set at assembly) and
    DO enter the domain loss; they never enter BCE

DR lane admission bar (control arm): lane mean over 2 draws blind windowed
read > 0.7031. H117 bar (margin arm): blind >= control + 0.01 AND
gold_full >= control - 0.005.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R11-H117_probe_trainer.py \
          --draw 1 --lambda-margin 0 --max-steps 3125
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import argparse
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
LANE = HERE / "DR_lane.parquet"

MAX_LEN, BATCH, LR = 512, 48, 1e-5
WARMUP_FRAC, CLIP = 0.1, 1.0
LAMBDA_MAX = 0.02
RESUME_EVERY = 1000
MARGIN_M = 0.25
DRAW_SEEDS = {1: 1117, 2: 2117}  # paired across arms per A2/A3


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H108 = _mod("h108", "R10-H108_lane.py")
M59 = H108.M59


def lane_rows():
    d = pl.read_parquet(LANE)
    claims = d["text"].to_list()
    chunks = [c[: M59.CFG.chunk_max_chars] for c in d["chunk"].to_list()]
    y = d["label"].cast(pl.Float32).to_numpy()  # -1 on clean partners, masked
    tags = d["dann_tag"].to_list()
    pair_ids = d["pair_id"].to_numpy()
    is_corrupt = (d["role"] == "corrupt").to_numpy()
    bce_mask = d["bce_mask"].to_numpy()
    return claims, chunks, y, tags, pair_ids, is_corrupt, bce_mask


class PairSet(Dataset):
    def __init__(self, claims, chunks, y, groups, pair_ids, is_corrupt, bce_mask, tok):
        self.c, self.k, self.y, self.g = claims, chunks, y, groups
        self.pid, self.cor, self.msk, self.tok = pair_ids, is_corrupt, bce_mask, tok

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.c[i], self.k[i], self.y[i], self.g[i], self.pid[i], self.cor[i], self.msk[i]

    def collate(self, b):
        c, k, y, g, pid, cor, msk = zip(*b, strict=True)
        enc = self.tok(list(c), list(k), return_tensors="pt", padding=True,
                       truncation=True, max_length=MAX_LEN)
        return (enc, torch.tensor(y), torch.tensor(g), torch.tensor(pid),
                torch.tensor(cor), torch.tensor(msk))


def build_perm(n_rows, pair_ids, is_corrupt, rng):
    """Flat permutation where each margin pair is adjacent (corrupt, clean) and
    never straddles a BATCH boundary. Units are shuffled; when a pair would
    land on the last slot of a batch, the next singleton ahead is swapped in."""
    pair_members = {}
    singles = []
    for i in range(n_rows):
        p = pair_ids[i]
        if p >= 0:
            pair_members.setdefault(p, [None, None])[0 if is_corrupt[i] else 1] = i
        else:
            singles.append([i])
    units = [v for v in pair_members.values()] + singles
    rng.shuffle(units)

    flat = []
    idx = 0
    while idx < len(units):
        u = units[idx]
        if len(u) == 2 and (len(flat) % BATCH) == BATCH - 1:
            j = idx + 1
            while j < len(units) and len(units[j]) != 1:
                j += 1
            if j < len(units):
                flat.extend(units.pop(j))
                continue
            # tail is all pairs: accept the straddle; margin loss skips split pairs
        flat.extend(u)
        idx += 1
    assert len(flat) == n_rows
    return flat


def margin_hinge(probs, pid, cor, lam_margin):
    """Hinge max(0, m - (p_clean - p_corrupt)) over COMPLETE pairs in the batch."""
    terms = []
    ids = pid.tolist()
    by_pid = {}
    for j, p in enumerate(ids):
        if p >= 0:
            by_pid.setdefault(p, {})["c" if cor[j] else "s"] = j
    for m in by_pid.values():
        if "c" in m and "s" in m:
            terms.append(F.relu(MARGIN_M - (probs[m["s"]] - probs[m["c"]])))
    if not terms:
        return probs.new_zeros(()), 0
    return torch.stack(terms).mean(), len(terms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", type=int, required=True, choices=(1, 2))
    ap.add_argument("--lambda-margin", type=float, required=True,
                    help="margin weight; 0 is the probe-control arm")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="probe mode: stop after N steps, save checkpoint, no eval")
    args = ap.parse_args()
    lam_margin = args.lambda_margin
    arm = f"lam{lam_margin:g}"
    ckpt_dir = HERE.parent.parent / "models" / f"H117-probe-{arm}"
    out = HERE / f"R11-H117_probe_{arm}_train.json"

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    seed = DRAW_SEEDS[args.draw]
    torch.manual_seed(seed)
    np.random.seed(seed)

    claims, chunks, y, tags = H108.public_train()
    n_public = len(y)
    lc, lk, ly, lt, lpid, lcor, lmsk = lane_rows()
    claims += lc
    chunks += lk
    y = np.concatenate([y, ly]).astype("float32")
    tags += lt
    pair_ids = np.concatenate([np.full(n_public, -1, dtype="int64"), lpid])
    is_corrupt = np.concatenate([np.zeros(n_public, dtype=bool), lcor])
    bce_mask = np.concatenate([np.zeros(n_public, dtype=bool), lmsk])

    tag_names = sorted(set(tags))
    tag_to_idx = {t: i for i, t in enumerate(tag_names)}
    groups = np.array([tag_to_idx[t] for t in tags])
    n_groups = len(tag_names)
    chance = 1.0 / n_groups
    n_pairs = int((pair_ids >= 0).sum() // 2)
    print(f"train: {len(y)} rows ({n_public} clean + {len(ly)} DR lane, "
          f"{n_pairs} margin pairs, {int(bce_mask.sum())} BCE-masked) across "
          f"{n_groups} domains (chance {chance:.3f})\n"
          f"arm {arm}  lambda_margin {lam_margin}  m {MARGIN_M}  seed {seed}\n",
          flush=True)

    tok = AutoTokenizer.from_pretrained(H108.STUDENT)
    base = AutoModel.from_pretrained(H108.STUDENT).cuda()
    base.config.reference_compile = False
    model = H108.DANNStudent(base, n_groups).cuda()
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"student {H108.STUDENT} + DANN heads  {n_par:.1f}M params\n", flush=True)

    ds = PairSet(claims, chunks, y, groups, pair_ids, is_corrupt, bce_mask, tok)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    n_steps = (len(ds) + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=n_steps, pct_start=WARMUP_FRAC, anneal_strategy="linear")

    resume_path = ckpt_dir / "resume.pt"
    start_step = 0
    if resume_path.exists():
        st = torch.load(resume_path, map_location="cuda", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        perm, start_step = st["perm"], st["step"]
        print(f"resumed from {resume_path} at step {start_step}/{n_steps}\n", flush=True)
    else:
        perm = build_perm(len(ds), pair_ids, is_corrupt, np.random.default_rng(seed))

    dl = DataLoader(Subset(ds, perm[start_step * BATCH:]), batch_size=BATCH,
                    shuffle=False, collate_fn=ds.collate, num_workers=2)
    domain_lossf = nn.CrossEntropyLoss()

    def save_resume(step):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tmp = resume_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "perm": perm, "step": step}, tmp)
        tmp.replace(resume_path)

    def save_final():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"trunk": base.state_dict(), "task_head": model.task_head.state_dict(),
                    "domain_head": model.domain_head.state_dict(), "config": base.config},
                   ckpt_dir / "dann_student.pt")
        tok.save_pretrained(ckpt_dir)
        base.save_pretrained(ckpt_dir / "trunk")
        resume_path.unlink(missing_ok=True)
        print(f"\ncheckpoint saved -> {ckpt_dir}\n", flush=True)

    model.train()
    t0 = time.time()
    dom_correct, dom_total = 0, 0
    hinge_sum, hinge_pairs = 0.0, 0
    a6_marg, a6_bce, a6_n = 0.0, 0.0, 0  # A6 running magnitude ratio
    for i, (enc, yy, gg, pid, cor, msk) in enumerate(dl):
        step = start_step + i
        if args.max_steps and i >= args.max_steps:
            ratio = (a6_marg / a6_bce) if a6_bce else 0.0
            print(f"probe stop at {args.max_steps} steps  "
                  f"hinge mean {hinge_sum / max(hinge_pairs, 1):.4f} over {hinge_pairs} pairs  "
                  f"A6 ratio {ratio:.4f}", flush=True)
            save_final()
            out.write_text(json.dumps({
                "arm": arm, "lambda_margin": lam_margin, "margin_m": MARGIN_M,
                "draw": args.draw, "seed": seed, "max_steps": args.max_steps,
                "n_steps_full": n_steps, "mix_rows": int(len(y)),
                "margin_pairs": n_pairs, "hinge_mean": round(hinge_sum / max(hinge_pairs, 1), 5),
                "hinge_pairs_seen": hinge_pairs, "a6_ratio": round(ratio, 5),
                "a6_void": bool(ratio > 0.25), "bce_mean": round(a6_bce / max(a6_n, 1), 5),
                "train_seconds": round(time.time() - t0, 1), "checkpoint": str(ckpt_dir),
            }, indent=2))
            print(f"results -> {out}")
            print(f"=== H117 PROBE {arm} DONE ===", flush=True)
            return
        enc = {k: v.cuda() for k, v in enc.items()}
        yy, gg, msk = yy.cuda(), gg.cuda(), msk.cuda()
        p = step / max(n_steps - 1, 1)
        lam = LAMBDA_MAX * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

        task_logit, domain_logit = model(enc, lam)
        keep = ~msk
        lossv = F.binary_cross_entropy_with_logits(
            task_logit, yy.clamp(min=0.0), reduction="none")
        t_loss = (lossv * keep).sum() / keep.sum().clamp(min=1)
        d_loss = domain_lossf(domain_logit, gg)
        m_loss, n_terms = margin_hinge(torch.sigmoid(task_logit), pid, cor, lam_margin)
        loss = t_loss + d_loss + lam_margin * m_loss
        if n_terms:
            hinge_sum += m_loss.item() * n_terms
            hinge_pairs += n_terms
        a6_marg += lam_margin * m_loss.item()
        a6_bce += t_loss.item()
        a6_n += 1

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
            hm = hinge_sum / max(hinge_pairs, 1)
            ratio = (a6_marg / a6_bce) if a6_bce else 0.0
            print(f"  step {step}/{n_steps} task {t_loss.item():.4f} domain {d_loss.item():.4f} "
                  f"hinge {hm:.4f} lam {lam:.4f} domain-acc {acc:.3f}  "
                  f"A6 {ratio:.4f}{' VOID' if ratio > 0.25 else ''}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if step and step % RESUME_EVERY == 0:
            save_resume(step + 1)
            print(f"  resume point saved at step {step}", flush=True)

    save_final()

    model.eval()
    res = H108.evaluate(model, tok)
    res.update({
        "params_M": round(n_par, 1), "lambda_max": LAMBDA_MAX,
        "lane": "DR (H112+H114 certified + reclaim)", "draw": args.draw,
        "arm": arm, "lambda_margin": lam_margin, "margin_m": MARGIN_M,
        "seed": seed, "mix_rows": int(len(y)), "clean_rows": int(n_public),
        "lane_rows": int(len(ly)), "margin_pairs": n_pairs,
        "hinge_mean_final": round(hinge_sum / max(hinge_pairs, 1), 4),
        "dann_groups": n_groups, "train_seconds": round(time.time() - t0, 1),
        "checkpoint": str(ckpt_dir),
    })
    gf = res["gold_full"]
    print(f"gold_full {gf['auc']:.4f} (n={gf['n']})  "
          f"gold {res['gold']['auc']:.4f}  ragtruth_en {res['ragtruth_en']['auc']:.4f}")
    out.write_text(json.dumps(res, indent=2))
    print(f"results -> {out}")
    print(f"=== H117 PROBE {arm} FULL-RUN DONE ===", flush=True)


if __name__ == "__main__":
    main()

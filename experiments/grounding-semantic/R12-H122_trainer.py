"""R12-H122 - DANN-GROUP-COLLAPSE training draws.

The clean R9-H105 recipe (mmBERT-base cross-encoder, BCE + DANN lambda 0.02
Ganin ramp, MAX_LEN 512, BATCH 48, LR 1e-5, OneCycleLR, 1 epoch) on the PUBLIC
MIX ONLY (685,670 rows), with EXACTLY ONE change against the control recipe:
the domain tag -> index map collapses the eight RAGTruth language tags into a
single `ragtruth` group, 12 DANN groups -> 5.

The registration writes the merge as 16 -> 9 against a clean+H108 mix; session
ruling 1 pinned the R12 incumbent as the CLEAN recipe (0.7031, H105 pair), so
the same merge principle lands at 12 -> 5 here. Resolution, transferred gate
evidence, bars and the pairing caveat are recorded in
`R12-H122_launch_design.md`, written before this trainer was launched.

Rows, labels, sampling order, natural frequency, lambda, ramp and schedule are
identical to the control; only the domain head's output layer differs
(7 x 769 = 5,383 parameters of 307.1M).

H126 seeding (session ruling 8, registration amendment A2 - the trap):
`n_groups` changes the last Linear's RNG consumption at construction, which
would desync every subsequent dropout mask and silently unpair the design. The
seed is issued BEFORE construction (trunk + task_head init is bit-identical
regardless of group count - `task_head` is built first) and RE-ISSUED
immediately AFTER construction, before any dropout or forward, so the training
RNG stream is independent of what construction consumed. The initial
trunk + task_head state is fingerprinted into `init_fingerprint.json`.

Bars (registered, re-priced onto the clean control in the design note):
ADMIT pair mean >= 0.7091 with sign agreement on both draws; hold
ragtruth_nonen >= 0.81695, gold_full >= 0.8414, no blind subset < 0.55;
KILL pair mean < 0.7051 or sign disagreement.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R12-H122_trainer.py --draw 1
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
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModel, AutoTokenizer

HERE = pathlib.Path(__file__).parent

MAX_LEN, BATCH, LR = 512, 48, 1e-5
WARMUP_FRAC, CLIP = 0.1, 1.0
LAMBDA_MAX = 0.02
RESUME_EVERY = 1000
DRAW_SEEDS = {1: 1122, 2: 2122}

# The registered merge: the eight RAGTruth language tags become one group.
RAGTRUTH_TAGS = (
    "ragtruth_en", "ragtruth_de", "ragtruth_fr", "ragtruth_es",
    "ragtruth_it", "ragtruth_pl", "ragtruth_hu", "ragtruth_cn",
)
MERGED_TAG = "ragtruth"
EXPECTED_GROUPS = ("halueval", "psiloqa", MERGED_TAG, "tabfact", "vitaminc")


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H108 = _mod("h108", "R10-H108_lane.py")


def collapse(tags):
    """Apply the registered group map and assert the resolved 5-group set."""
    merged = [MERGED_TAG if t in RAGTRUTH_TAGS else t for t in tags]
    names = tuple(sorted(set(merged)))
    if names != EXPECTED_GROUPS:
        raise SystemExit(
            f"GROUP-MAP ABORT: collapsed groups {names} != registered {EXPECTED_GROUPS}. "
            "The mix build changed; re-resolve the map in R12-H122_launch_design.md."
        )
    return merged, names


class MixSet(Dataset):
    def __init__(self, claims, chunks, y, groups, tok):
        self.c, self.k, self.y, self.g, self.tok = claims, chunks, y, groups, tok

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.c[i], self.k[i], self.y[i], self.g[i]

    def collate(self, b):
        c, k, y, g = zip(*b, strict=True)
        enc = self.tok(list(c), list(k), return_tensors="pt", padding=True,
                       truncation=True, max_length=MAX_LEN)
        return enc, torch.tensor(y), torch.tensor(g)


def fingerprint_named(model):
    """The parameters the init fingerprint covers: trunk + task_head."""
    return [(n, p) for n, p in model.named_parameters()
            if n.startswith(("trunk.", "task_head."))]


def init_fingerprint(model):
    """blake2b-128 over trunk + task_head parameter bytes in sorted name order."""
    h = hashlib.blake2b(digest_size=16)
    n_par = 0
    for name, p in sorted(fingerprint_named(model), key=lambda kv: kv[0]):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
        n_par += p.numel()
    return h.hexdigest(), n_par


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", type=int, required=True, choices=(1, 2))
    ap.add_argument("--max-steps", type=int, default=0,
                    help="probe mode: stop after N steps, no checkpoint/eval")
    args = ap.parse_args()
    ckpt_dir = HERE.parent.parent / "models" / f"R12-H122-draw{args.draw}"
    out = HERE / f"R12-H122_draw{args.draw}_result.json"

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    seed = DRAW_SEEDS[args.draw]
    torch.manual_seed(seed)
    np.random.seed(seed)

    claims, chunks, y, tags = H108.public_train()
    merged, tag_names = collapse(tags)
    tag_to_idx = {t: i for i, t in enumerate(tag_names)}
    groups = np.array([tag_to_idx[t] for t in merged])
    n_groups = len(tag_names)
    counts = {t: int((groups == i).sum()) for t, i in tag_to_idx.items()}
    print(f"train: {len(y)} public rows across {n_groups} COLLAPSED domains "
          f"(chance {1.0 / n_groups:.3f})  seed {seed}\n"
          f"group map (12 -> {n_groups}): {counts}\n", flush=True)

    tok = AutoTokenizer.from_pretrained(H108.STUDENT)
    base = AutoModel.from_pretrained(H108.STUDENT).cuda()
    base.config.reference_compile = False
    model = H108.DANNStudent(base, n_groups).cuda()
    # H126 / ruling 8 / amendment A2: re-issue the seed AFTER construction, before
    # any dropout or forward, so the training stream is independent of the group
    # count's construction draws.
    torch.manual_seed(seed)
    fp, fp_numel = init_fingerprint(model)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"student {H108.STUDENT} + DANN heads  {n_par:.1f}M params\n"
          f"init fingerprint (trunk+task_head, {fp_numel} params): {fp}\n", flush=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "init_fingerprint.json").write_text(json.dumps(
        {"draw": args.draw, "seed": seed, "n_groups": n_groups,
         "group_counts": counts, "scope": "trunk+task_head",
         "n_params": fp_numel, "blake2b_128": fp}, indent=2))

    ds = MixSet(claims, chunks, y, groups, tok)
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
        perm = np.random.default_rng(seed).permutation(len(ds)).tolist()

    dl = DataLoader(Subset(ds, perm[start_step * BATCH:]), batch_size=BATCH,
                    shuffle=False, collate_fn=ds.collate, num_workers=2)
    domain_lossf = nn.CrossEntropyLoss()

    def save_resume(step):
        tmp = resume_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "perm": perm, "step": step}, tmp)
        tmp.replace(resume_path)

    model.train()
    t0 = time.time()
    dom_correct, dom_total = 0, 0
    for i, (enc, yy, gg) in enumerate(dl):
        step = start_step + i
        if args.max_steps and i >= args.max_steps:
            print(f"probe stop at {args.max_steps} steps", flush=True)
            return
        enc = {k: v.cuda() for k, v in enc.items()}
        yy, gg = yy.cuda(), gg.cuda()
        p = step / max(n_steps - 1, 1)
        lam = LAMBDA_MAX * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

        task_logit, domain_logit = model(enc, lam)
        t_loss = F.binary_cross_entropy_with_logits(task_logit, yy)
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
            print(f"  step {step}/{n_steps} task {t_loss.item():.4f} "
                  f"domain {d_loss.item():.4f} lam {lam:.4f} domain-acc {acc:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if step and step % RESUME_EVERY == 0:
            save_resume(step + 1)
            print(f"  resume point saved at step {step}", flush=True)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"trunk": base.state_dict(), "task_head": model.task_head.state_dict(),
                "domain_head": model.domain_head.state_dict(), "config": base.config},
               ckpt_dir / "dann_student.pt")
    tok.save_pretrained(ckpt_dir)
    base.save_pretrained(ckpt_dir / "trunk")
    resume_path.unlink(missing_ok=True)
    print(f"\ncheckpoint saved -> {ckpt_dir}\n", flush=True)

    model.eval()
    res = H108.evaluate(model, tok)
    res.update({
        "params_M": round(n_par, 1), "lambda_max": LAMBDA_MAX,
        "lane": "R12-H122 DANN group collapse (public mix only, 12 -> 5 groups)",
        "draw": args.draw, "seed": seed,
        "mix_rows": len(y), "dann_groups": n_groups, "group_counts": counts,
        "init_fingerprint": fp, "init_fingerprint_scope": "trunk+task_head",
        "train_seconds": round(time.time() - t0, 1), "checkpoint": str(ckpt_dir),
    })
    gf = res["gold_full"]
    print(f"gold_full {gf['auc']:.4f} (n={gf['n']})  "
          f"gold {res['gold']['auc']:.4f}  ragtruth_en {res['ragtruth_en']['auc']:.4f}  "
          f"ragtruth_nonen {res['ragtruth_nonen']['auc']:.4f}")
    out.write_text(json.dumps(res, indent=2))
    print(f"results -> {out}")
    print(f"=== H122 DRAW {args.draw} DONE ===", flush=True)


if __name__ == "__main__":
    main()

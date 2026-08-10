"""R13-H128 - WICE-ATTRIBUTED-SUPPORT-LANE training draws.

The clean R9-H105 recipe (mmBERT-base cross-encoder, BCE + DANN lambda 0.02
Ganin ramp, MAX_LEN 512, BATCH 48, LR 1e-5, OneCycleLR, 1 epoch) on the clean
public mix (685,670 rows, 12 frozen groups) PLUS the WiCE attributed-support
lane, with exactly one change against the control recipe: the lane is added.

    lane   R13-H128_lane.parquet   22,071 rows, one DANN group `wice_attrib`
    mix    public 685,670 + lane   = 707,741 rows

The lane's supervised contrast is presence-vs-absence of WiCE's annotated
minimal evidence set: the full set supports the claim (label 1), the same set
with one sentence deleted or swapped for the lexically nearest sentence of
another article does not (label 0). Built by `R13-H128_build_lane.py` under the
most-conservative construction the pre-GPU gates costed (min-set only, 18,264
pairs against the registered >= 15,000 bar); provenance measured 0.000000
against the full arena on both the evidence and the claim side.

Groups: the 12 clean-mix groups are FROZEN (R12-H122 kill) and the lane adds
one group of its own, the H108/DR convention of one DANN group per lane source.
n_groups is therefore 13, not the control's 12 - see the pairing caveat in
`R13-H128_launch_design.md`.

H126 seeding (session ruling 8): the seed is issued before model construction
and RE-ISSUED immediately after, before any dropout or forward, and the
trunk+task_head init is fingerprinted.

Bars (registered): hagrid >= 0.688 (+0.040), mean HOLD >= 0.7031, finqa/techqa
hold per ruling 9; 1-draw pilot gate mean >= 0.700 AND hagrid >= +0.02, both
required to spend draw 2.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R13-H128_trainer.py --draw 1
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

MAX_LEN, BATCH, LR = 512, 48, 1e-5
WARMUP_FRAC, CLIP = 0.1, 1.0
LAMBDA_MAX = 0.02
RESUME_EVERY = 1000
DRAW_SEEDS = {1: 1128, 2: 2128}

LANE = HERE / "R13-H128_lane.parquet"
LANE_TAG = "wice_attrib"
PUBLIC_GROUPS = (
    "halueval", "psiloqa", "ragtruth_cn", "ragtruth_de", "ragtruth_en",
    "ragtruth_es", "ragtruth_fr", "ragtruth_hu", "ragtruth_it", "ragtruth_pl",
    "tabfact", "vitaminc",
)
EXPECTED_GROUPS = PUBLIC_GROUPS + (LANE_TAG,)


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H108 = _mod("h108", "R10-H108_lane.py")
CHUNK_MAX = H108.M59.CFG.chunk_max_chars


def lane_train():
    """The H128 WiCE lane, already gate-cleared; chunks truncated to the
    serving unit exactly as `R10-H108_lane.lane_train` does."""
    d = pl.read_parquet(LANE)
    claims = d["claim"].to_list()
    chunks = [c[:CHUNK_MAX] for c in d["chunk"].to_list()]
    y = d["label"].cast(pl.Float32).to_numpy()
    tags = d["tag"].to_list()
    return claims, chunks, y, tags


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
    ckpt_dir = HERE.parent.parent / "models" / f"R13-H128-draw{args.draw}"
    out = HERE / f"R13-H128_draw{args.draw}_result.json"

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    seed = DRAW_SEEDS[args.draw]
    torch.manual_seed(seed)
    np.random.seed(seed)

    claims, chunks, y, tags = H108.public_train()
    n_public = len(y)
    lc, lk, ly, lt = lane_train()
    claims += lc
    chunks += lk
    y = np.concatenate([y, ly]).astype("float32")
    tags += lt

    names = tuple(sorted(set(tags)))
    if names != tuple(sorted(EXPECTED_GROUPS)):
        raise SystemExit(
            f"GROUP-MAP ABORT: mix groups {names} != registered {EXPECTED_GROUPS}. "
            "The mix or the lane build changed; re-resolve in R13-H128_launch_design.md."
        )
    tag_names = EXPECTED_GROUPS
    tag_to_idx = {t: i for i, t in enumerate(tag_names)}
    groups = np.array([tag_to_idx[t] for t in tags])
    n_groups = len(tag_names)
    counts = {t: int(sum(1 for x in tags if x == t)) for t in tag_names}
    lane_pos = float(ly.mean())
    print(f"train: {len(y)} rows ({n_public} clean public + {len(ly)} WiCE lane, "
          f"lane positives {lane_pos:.4f}) over {n_groups} domains "
          f"(chance {1.0 / n_groups:.3f})  seed {seed}", flush=True)
    for t in tag_names:
        print(f"  {t:<14} {counts[t]:>7}", flush=True)
    print("", flush=True)

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
         "group_counts": counts, "lane_rows": int(len(ly)),
         "lane_positive_fraction": round(lane_pos, 4), "scope": "trunk+task_head",
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
    # The lane adds a DANN group, so per-group discriminator accuracy over the
    # final 20% of the epoch is reported beside the blind read (H127 amendment 4
    # convention) to keep the GRL confound visible.
    tail_start = int(0.8 * n_steps)
    tail_hit = np.zeros(n_groups, dtype=np.int64)
    tail_n = np.zeros(n_groups, dtype=np.int64)
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

        pred = domain_logit.argmax(-1)
        dom_correct += (pred == gg).sum().item()
        dom_total += len(gg)
        if step >= tail_start:
            g_np, ok = gg.cpu().numpy(), (pred == gg).cpu().numpy()
            np.add.at(tail_n, g_np, 1)
            np.add.at(tail_hit, g_np, ok)
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

    dom_per_group = {t: (round(float(tail_hit[i] / tail_n[i]), 4) if tail_n[i] else None)
                     for t, i in tag_to_idx.items()}
    print(f"domain acc per group (final 20%): {dom_per_group}\n", flush=True)

    model.eval()
    res = H108.evaluate(model, tok)
    res.update({
        "params_M": round(n_par, 1), "lambda_max": LAMBDA_MAX,
        "lane": "R13-H128 WiCE attributed-support lane "
                "(clean public mix + wice_attrib, 13 groups)",
        "draw": args.draw, "seed": seed,
        "mix_rows": int(len(y)), "clean_rows": int(n_public), "lane_rows": int(len(ly)),
        "lane_positive_fraction": round(lane_pos, 4),
        "dann_groups": n_groups, "lane_groups": [LANE_TAG], "group_counts": counts,
        "domain_acc_per_group_final20pct": dom_per_group,
        "init_fingerprint": fp, "init_fingerprint_scope": "trunk+task_head",
        "train_seconds": round(time.time() - t0, 1), "checkpoint": str(ckpt_dir),
    })
    gf = res["gold_full"]
    print(f"gold_full {gf['auc']:.4f} (n={gf['n']})  "
          f"gold {res['gold']['auc']:.4f}  ragtruth_en {res['ragtruth_en']['auc']:.4f}  "
          f"ragtruth_nonen {res['ragtruth_nonen']['auc']:.4f}")
    out.write_text(json.dumps(res, indent=2))
    print(f"results -> {out}")
    print(f"=== H128 DRAW {args.draw} DONE ===", flush=True)


if __name__ == "__main__":
    main()

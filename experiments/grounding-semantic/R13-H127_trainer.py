"""R13-H127 - RAGTRUTH-PARALLEL-COPY-REBALANCE training draws.

The clean R9-H105 recipe (mmBERT-base cross-encoder, BCE + DANN lambda 0.02
Ganin ramp, MAX_LEN 512, BATCH 48, LR 1e-5, OneCycleLR, 1 epoch) on the PUBLIC
MIX ONLY (685,670 rows, 12 DANN groups - the group design is FROZEN at 12 by
the R12-H122 kill), with EXACTLY ONE change against the control recipe: the
RAGTruth family is resampled under a family-mass-preserving reweight.

    ragtruth_en          per-row weight 4.0
    ragtruth_{de,fr,es,it,pl,hu,cn}   per-row weight 4/7 = 0.5714
    family row-equivalents            FIXED at its natural total
    every other group                 untouched (weight 1.0)

87.5% of the only arena-shaped register's mass is positionally-parallel
translated duplicates of the same 15,090 items (CPU alignment gate PASSED:
label agreement >= 0.9998, pos_frac spread 0.000199, numeric-token Jaccard
0.84-0.88 aligned vs 0.13 shuffled). The reweight spends the same budget on
four times the English exposure.

Implementation - SAMPLING WEIGHTS AS ROW MULTIPLICITY (see
`R13-H127_launch_design.md`). The weights are realized as an integer index
multiset over the unchanged mix arrays, drawn once and then permuted by the
draw seed:

  * every `ragtruth_en` row index is emitted 4 times (weight exactly 4.0)
  * each translation contributes a seeded without-replacement subsample sized
    by largest-remainder allocation of the residual family budget
  * the index multiset is permuted by `np.random.default_rng(seed)` and stored
    in `resume.pt`, so a resumed run replays the identical stream

Because 4.0 + 7 x 4/7 = 8.0 exactly, the family mass and therefore the total
mix row count and the step count are unchanged from the control - the arm is
step-for-step paired.

H126 seeding (session ruling 8): `n_groups` is 12 here, identical to the
control, so construction consumes the same RNG either way; the seed is still
issued before construction and RE-ISSUED immediately after, before any dropout
or forward, and the trunk+task_head init is fingerprinted.

Amendment 4 (registered): the intervention changes per-group DANN mass, so the
per-group discriminator accuracy over the final 20% of the epoch is reported
beside the blind read to keep the GRL confound visible.

Bars (registered): ADMIT pair mean >= 0.7150 with sign agreement and holds
(ragtruth_nonen >= 0.82 both draws, gold_full >= 0.84, no arena subset < 0.55);
REFUTE pair mean < 0.70496 or sign disagreement.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R13-H127_trainer.py --draw 1
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
DRAW_SEEDS = {1: 1127, 2: 2127}

EN_TAG = "ragtruth_en"
TRANS_TAGS = ("ragtruth_de", "ragtruth_fr", "ragtruth_es",
              "ragtruth_it", "ragtruth_pl", "ragtruth_hu", "ragtruth_cn")
W_EN = 4.0
W_TRANS = 4.0 / 7.0  # 0.571428..., registered as 0.5714
# The registration records the family at 120,717 row-equivalents; the live mix
# measures 120,720 (8 x 15,090, all files identical pre- and post-filter). The
# +3 is a registration arithmetic slip, not mix drift - the mass constraint is
# enforced against the MEASURED natural total and the slip is asserted small.
REGISTERED_FAMILY_MASS = 120_717
REGISTRATION_SLIP_TOL = 4
MASS_TOL = 1

EXPECTED_GROUPS = (
    "halueval", "psiloqa", "ragtruth_cn", "ragtruth_de", "ragtruth_en",
    "ragtruth_es", "ragtruth_fr", "ragtruth_hu", "ragtruth_it", "ragtruth_pl",
    "tabfact", "vitaminc",
)


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H108 = _mod("h108", "R10-H108_lane.py")


def allocate_translation_targets(counts, budget):
    """Largest-remainder split of `budget` row-equivalents across the seven
    translations in proportion to their natural size. Ties break on the fixed
    TRANS_TAGS order, so the allocation is deterministic."""
    total = sum(counts[t] for t in TRANS_TAGS)
    exact = {t: counts[t] / total * budget for t in TRANS_TAGS}
    target = {t: int(np.floor(exact[t])) for t in TRANS_TAGS}
    short = budget - sum(target.values())
    order = sorted(TRANS_TAGS, key=lambda t: (-(exact[t] - target[t]), TRANS_TAGS.index(t)))
    for t in order[:short]:
        target[t] += 1
    return target


def build_index(tags, seed):
    """The reweighted training index multiset over the unchanged mix arrays.

    Returns (index array, realized per-group row-equivalent dict, weight dict).
    """
    tags = np.asarray(tags)
    names = tuple(sorted(set(tags.tolist())))
    if names != EXPECTED_GROUPS:
        raise SystemExit(
            f"GROUP-MAP ABORT: mix groups {names} != registered {EXPECTED_GROUPS}. "
            "The mix build changed; re-resolve in R13-H127_launch_design.md."
        )
    counts = {t: int((tags == t).sum()) for t in names}
    family = (EN_TAG,) + TRANS_TAGS
    natural_mass = sum(counts[t] for t in family)
    if abs(natural_mass - REGISTERED_FAMILY_MASS) > REGISTRATION_SLIP_TOL:
        raise SystemExit(
            f"FAMILY-MASS ABORT: measured natural family mass {natural_mass} is more than "
            f"{REGISTRATION_SLIP_TOL} from the registered {REGISTERED_FAMILY_MASS}; the "
            "RAGTruth build has drifted, re-register before training."
        )

    en_equiv = int(round(W_EN * counts[EN_TAG]))
    budget = natural_mass - en_equiv
    if budget < 0:
        raise SystemExit("FAMILY-MASS ABORT: EN mass alone exceeds the family budget.")
    target = allocate_translation_targets(counts, budget)

    rng = np.random.default_rng([seed, 127])  # independent of the permutation stream
    parts, realized = [], {}
    for t in names:
        pos = np.flatnonzero(tags == t)
        if t == EN_TAG:
            keep = np.repeat(pos, int(W_EN))
        elif t in TRANS_TAGS:
            if target[t] > len(pos):
                raise SystemExit(f"SAMPLING ABORT: {t} target {target[t]} > {len(pos)} rows.")
            keep = np.sort(rng.choice(pos, size=target[t], replace=False))
        else:
            keep = pos
        parts.append(keep)
        realized[t] = int(len(keep))

    realized_mass = sum(realized[t] for t in family)
    if abs(realized_mass - natural_mass) > MASS_TOL:
        raise SystemExit(
            f"FAMILY-MASS ABORT: realized {realized_mass} vs natural {natural_mass} "
            f"(tolerance {MASS_TOL})."
        )
    for t in names:
        if t not in family and realized[t] != counts[t]:
            raise SystemExit(f"UNTOUCHED-GROUP ABORT: {t} {realized[t]} != {counts[t]}.")
    w_en = realized[EN_TAG] / counts[EN_TAG]
    if abs(w_en - W_EN) > 1e-9:
        raise SystemExit(f"WEIGHT ABORT: realized EN weight {w_en} != {W_EN}.")
    for t in TRANS_TAGS:
        w = realized[t] / counts[t]
        if abs(w - W_TRANS) > 1e-3:
            raise SystemExit(f"WEIGHT ABORT: realized {t} weight {w:.6f} != {W_TRANS:.6f}.")

    weights = {t: round(realized[t] / counts[t], 6) for t in names}
    idx = np.concatenate(parts)

    print("realized per-group row-equivalents (natural -> reweighted, per-row weight):",
          flush=True)
    for t in names:
        mark = "  <-- family" if t in family else ""
        print(f"  {t:<14} {counts[t]:>7} -> {realized[t]:>7}   w={weights[t]:.4f}{mark}",
              flush=True)
    print(f"  {'RAGTruth family':<14} {natural_mass:>7} -> {realized_mass:>7}   "
          f"(registered {REGISTERED_FAMILY_MASS}, measured slip "
          f"{natural_mass - REGISTERED_FAMILY_MASS:+d})", flush=True)
    print(f"  {'MIX TOTAL':<14} {len(tags):>7} -> {len(idx):>7}\n", flush=True)

    return idx, counts, realized, weights, natural_mass, realized_mass


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
    ckpt_dir = HERE.parent.parent / "models" / f"R13-H127-draw{args.draw}"
    out = HERE / f"R13-H127_draw{args.draw}_result.json"

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    seed = DRAW_SEEDS[args.draw]
    torch.manual_seed(seed)
    np.random.seed(seed)

    claims, chunks, y, tags = H108.public_train()
    tag_names = EXPECTED_GROUPS
    tag_to_idx = {t: i for i, t in enumerate(tag_names)}
    groups = np.array([tag_to_idx[t] for t in tags])
    n_groups = len(tag_names)
    idx, counts, realized, weights, nat_mass, re_mass = build_index(tags, seed)
    print(f"train: {len(idx)} row-equivalents over {len(y)} public rows, "
          f"{n_groups} domains (chance {1.0 / n_groups:.3f})  seed {seed}\n", flush=True)

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
         "natural_group_counts": counts, "row_equivalents": realized,
         "per_row_weights": weights, "scope": "trunk+task_head",
         "n_params": fp_numel, "blake2b_128": fp}, indent=2))

    ds = MixSet(claims, chunks, y, groups, tok)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    n_steps = (len(idx) + BATCH - 1) // BATCH
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
        perm = np.random.default_rng(seed).permutation(idx).tolist()

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
    # Amendment 4: per-group discriminator accuracy over the final 20% of steps.
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
        "lane": "R13-H127 RAGTruth parallel-copy rebalance "
                "(public mix only, 12 groups, EN 4.0 / translations 0.5714)",
        "draw": args.draw, "seed": seed,
        "mix_rows": len(y), "train_row_equivalents": len(idx),
        "dann_groups": n_groups,
        "natural_group_counts": counts, "row_equivalents": realized,
        "per_row_weights": weights,
        "family_mass_natural": nat_mass, "family_mass_realized": re_mass,
        "family_mass_registered": REGISTERED_FAMILY_MASS,
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
    print(f"=== H127 DRAW {args.draw} DONE ===", flush=True)


if __name__ == "__main__":
    main()

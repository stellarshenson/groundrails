"""R14-H135 (A6) - MINIMAL-PAIR CO-LOCATION IN THE ADMITTED H108 LANE.

The clean R9-H105 recipe (mmBERT-base cross-encoder, BCE + DANN lambda 0.02
Ganin ramp, MAX_LEN 512, BATCH 48, LR 1e-5, OneCycleLR, 1 epoch) on the exact
H108 mix - clean public 685,670 rows + `R10-H108_pairs.parquet` 61,184 lane rows
= 746,854 rows over 16 DANN groups - with EXACTLY ONE change against the control
recipe: BATCH COMPOSITION.

    arm      colocate   each reconstructed minimal pair occupies two adjacent
                        slots of the epoch stream and never straddles a batch
                        boundary; pair units and singletons are shuffled together
    control  control    the flat `rng.permutation` the H108 lane trainer draws

The loss, the optimizer, the schedule, the mix, the groups, the step count and
the seeds are identical between the two arms. Co-location changes WHEN the
gradient sees a lane contrast, not what the objective rewards, so absolute score
comparability is preserved by construction (R14_synthesis.md, R14-A6).

Pair reconstruction is the registered gate definition, verbatim from
`R14_gate_H135_editsim.json`: "first label-1 and first label-0 row per chunk in
file order" over the lane parquet -> 6,889 pair units covering 13,778 lane rows;
the 6,889 chunks themselves cover 29,779 of 61,184 lane rows (48.67%, clause 1).

Seeding (H126 / session ruling 8): the seed is issued before model construction
and RE-ISSUED immediately after, before any dropout or forward; the
trunk+task_head init is fingerprinted, so arm and control at the same draw share
an identical initialization and differ only in batch order.

Resume: `perm` is persisted with the weights and the step counter, and the
restart slices it at `start_step * BATCH` - a multiple of BATCH - so a resumed
run replays the SAME permutation with the SAME batch boundaries, and the
co-location property is preserved across restarts. The perm fingerprint is
printed at every start for verification.

Bars (registered, R14_synthesis.md A6 + session ruling 13): PRIMARY finqa pair
mean >= H108 pair + 0.030 = 0.7482 with sign agreement on both draws against the
seeded control pair; HOLD arena mean >= 0.7031, pubmedqa >= 0.5141, gold_full >=
0.8484; KILL if finqa < +0.010 over the H108 pair or the draws disagree in sign
or the arena mean < 0.6971. PILOT GATE: spend draw 2 only if draw 1 reads finqa
>= 0.7382 AND mean >= 0.700.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R14-H135_trainer.py \
          --draw 1 --arm colocate
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
DRAW_SEEDS = {1: 1135, 2: 2135}

PUBLIC_GROUPS = (
    "halueval", "psiloqa", "ragtruth_cn", "ragtruth_de", "ragtruth_en",
    "ragtruth_es", "ragtruth_fr", "ragtruth_hu", "ragtruth_it", "ragtruth_pl",
    "tabfact", "vitaminc",
)
LANE_GROUPS = ("quant_corrupt", "quant_feverous", "quant_infotabs", "quant_scitab")
EXPECTED_GROUPS = tuple(sorted(PUBLIC_GROUPS + LANE_GROUPS))
EXPECTED_PAIRS = 6889  # clause-1 measurement, R14_gate_H135_editsim.json


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H108 = _mod("h108", "R10-H108_lane.py")


def lane_pair_units(n_public):
    """Minimal-pair units as GLOBAL mix indices, under the registered gate rule:
    first label-1 and first label-0 row per chunk, in lane-file order. Lane rows
    are appended to the public mix in parquet order, so global index =
    n_public + lane row index."""
    d = pl.read_parquet(H108.LANE)
    first1, first0 = {}, {}
    for i, (chunk, label) in enumerate(zip(d["chunk"].to_list(), d["label"].to_list(), strict=True)):
        tgt = first1 if label == 1.0 else first0
        if chunk not in tgt:
            tgt[chunk] = i
    pairs = [(n_public + first0[c], n_public + first1[c]) for c in first1 if c in first0]
    pairs.sort(key=lambda p: min(p))  # deterministic order before the shuffle
    return pairs


def build_colocated_perm(n_rows, pairs, rng):
    """Flat permutation where each minimal pair is adjacent (corrupted, clean)
    and never straddles a BATCH boundary. Units - a pair or a singleton row - are
    shuffled together; when a pair would land on the last slot of a batch, the
    next singleton ahead is swapped in. Same construction as
    `DR_lane_trainer.build_perm`."""
    paired = {i for p in pairs for i in p}
    units = [list(p) for p in pairs] + [[i] for i in range(n_rows) if i not in paired]
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
            # tail is all pairs: accept the straddle rather than reorder the tail
        flat.extend(u)
        idx += 1
    assert len(flat) == n_rows and len(set(flat)) == n_rows
    return flat


def perm_stats(perm, pairs):
    """Co-location instrumentation: the realised P(partner adjacent) and
    P(partner in the same batch) against the 6.29e-5 flat-shuffle baseline."""
    pos = np.empty(len(perm), dtype=np.int64)
    pos[np.asarray(perm, dtype=np.int64)] = np.arange(len(perm), dtype=np.int64)
    a = pos[np.array([p[0] for p in pairs])]
    b = pos[np.array([p[1] for p in pairs])]
    adjacent = int((np.abs(a - b) == 1).sum())
    same_batch = int(((a // BATCH) == (b // BATCH)).sum())
    return {
        "n_pairs": len(pairs),
        "p_partner_adjacent": round(adjacent / len(pairs), 6),
        "p_partner_same_batch": round(same_batch / len(pairs), 6),
        "n_adjacent": adjacent,
        "n_same_batch": same_batch,
        "flat_shuffle_baseline_p_same_batch": round((BATCH - 1) / (len(perm) - 1), 8),
        "mean_abs_step_distance": round(float(np.abs(a - b).mean()) / BATCH, 2),
    }


def perm_fingerprint(perm):
    return hashlib.blake2b(
        np.asarray(perm, dtype=np.int64).tobytes(), digest_size=8
    ).hexdigest()


def fingerprint_named(model):
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


def build_mix():
    claims, chunks, y, tags = H108.public_train()
    n_public = len(y)
    lc, lk, ly, lt = H108.lane_train()
    claims += lc
    chunks += lk
    y = np.concatenate([y, ly]).astype("float32")
    tags += lt
    names = tuple(sorted(set(tags)))
    if names != EXPECTED_GROUPS:
        raise SystemExit(
            f"GROUP-MAP ABORT: mix groups {names} != registered {EXPECTED_GROUPS}. "
            "The mix or the lane changed; re-resolve in R14-H135_launch_design.md."
        )
    return claims, chunks, y, tags, n_public, len(ly)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", type=int, required=True, choices=(1, 2))
    ap.add_argument("--arm", choices=("colocate", "control"), required=True)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="smoke mode: stop after N steps, no checkpoint/eval, ckpt dir -smoke")
    ap.add_argument("--resume-every", type=int, default=0, help="smoke mode override")
    ap.add_argument("--audit-perm", action="store_true",
                    help="CPU only: build the mix and both permutations, report co-location, exit")
    args = ap.parse_args()

    seed = DRAW_SEEDS[args.draw]
    tag = "arm" if args.arm == "colocate" else "ctrl"
    claims, chunks, y, tags, n_public, n_lane = build_mix()
    pairs = lane_pair_units(n_public)
    n_rows = len(y)

    if args.audit_perm:
        rep = {"n_rows": n_rows, "n_public": n_public, "n_lane": n_lane,
               "n_pairs": len(pairs), "expected_pairs": EXPECTED_PAIRS,
               "pair_rule": "first label-1 and first label-0 row per chunk in file order",
               "n_steps": (n_rows + BATCH - 1) // BATCH, "seed": seed}
        for arm in ("colocate", "control"):
            rng = np.random.default_rng(seed)
            perm = (build_colocated_perm(n_rows, pairs, rng) if arm == "colocate"
                    else rng.permutation(n_rows).tolist())
            rep[arm] = perm_stats(perm, pairs) | {"perm_fingerprint": perm_fingerprint(perm)}
        out = HERE / "R14-H135_perm_audit.json"
        out.write_text(json.dumps(rep, indent=2))
        print(json.dumps(rep, indent=2), flush=True)
        print(f"audit -> {out}", flush=True)
        return

    ckpt_dir = HERE.parent.parent / "models" / (
        "R14-H135-smoke" if args.max_steps else f"R14-H135-{tag}-draw{args.draw}")
    out = HERE / f"R14-H135_{tag}_draw{args.draw}_result.json"
    resume_every = args.resume_every or RESUME_EVERY

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    tag_to_idx = {t: i for i, t in enumerate(EXPECTED_GROUPS)}
    groups = np.array([tag_to_idx[t] for t in tags])
    n_groups = len(EXPECTED_GROUPS)
    counts = {t: int(sum(1 for x in tags if x == t)) for t in EXPECTED_GROUPS}
    if len(pairs) != EXPECTED_PAIRS:
        raise SystemExit(f"PAIR ABORT: {len(pairs)} pair units != registered {EXPECTED_PAIRS}")
    print(f"train: {n_rows} rows ({n_public} clean public + {n_lane} H108 lane) over "
          f"{n_groups} domains (chance {1.0 / n_groups:.3f})  seed {seed}  arm {args.arm}\n"
          f"minimal pairs reconstructed: {len(pairs)} (covering {2 * len(pairs)} lane rows)",
          flush=True)
    for t in EXPECTED_GROUPS:
        print(f"  {t:<15} {counts[t]:>7}", flush=True)
    print("", flush=True)

    tok = AutoTokenizer.from_pretrained(H108.STUDENT)
    base = AutoModel.from_pretrained(H108.STUDENT).cuda()
    base.config.reference_compile = False
    model = H108.DANNStudent(base, n_groups).cuda()
    torch.manual_seed(seed)  # ruling 8: re-issue after construction, before any forward
    fp, fp_numel = init_fingerprint(model)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"student {H108.STUDENT} + DANN heads  {n_par:.1f}M params\n"
          f"init fingerprint (trunk+task_head, {fp_numel} params): {fp}\n", flush=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ds = H108.GroupSet(claims, chunks, y, groups, tok)
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
        rng = np.random.default_rng(seed)
        perm = (build_colocated_perm(len(ds), pairs, rng) if args.arm == "colocate"
                else rng.permutation(len(ds)).tolist())

    stats = perm_stats(perm, pairs)
    print(f"perm fingerprint {perm_fingerprint(perm)}  "
          f"P(partner adjacent) {stats['p_partner_adjacent']:.6f}  "
          f"P(partner same batch) {stats['p_partner_same_batch']:.6f}  "
          f"(flat-shuffle baseline {stats['flat_shuffle_baseline_p_same_batch']:.2e})\n",
          flush=True)
    (ckpt_dir / "init_fingerprint.json").write_text(json.dumps(
        {"draw": args.draw, "arm": args.arm, "seed": seed, "n_groups": n_groups,
         "group_counts": counts, "lane_rows": n_lane, "mix_rows": n_rows,
         "scope": "trunk+task_head", "n_params": fp_numel, "blake2b_128": fp,
         "perm_fingerprint": perm_fingerprint(perm), "colocation": stats}, indent=2))

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
            print(f"smoke stop at {args.max_steps} steps", flush=True)
            return
        if args.max_steps and i < 5:
            # end-to-end check that the emitted batch IS the permutation slice
            exp = y[perm[(start_step + i) * BATCH:(start_step + i + 1) * BATCH]]
            assert np.array_equal(yy.numpy(), exp), f"batch {step} does not match perm slice"
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
        if step % 200 == 0 or (args.max_steps and step % 10 == 0):
            acc = dom_correct / max(dom_total, 1)
            dom_correct, dom_total = 0, 0
            print(f"  step {step}/{n_steps} task {t_loss.item():.4f} "
                  f"domain {d_loss.item():.4f} lam {lam:.4f} domain-acc {acc:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if step and step % resume_every == 0:
            save_resume(step + 1)
            print(f"  resume point saved at step {step}", flush=True)

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
        "lane": "R14-H135 minimal-pair co-location in the admitted H108 lane",
        "arm": args.arm, "draw": args.draw, "seed": seed,
        "mix_rows": n_rows, "clean_rows": n_public, "lane_rows": n_lane,
        "dann_groups": n_groups, "lane_groups": list(LANE_GROUPS), "group_counts": counts,
        "minimal_pairs": len(pairs), "colocation": stats,
        "perm_fingerprint": perm_fingerprint(perm),
        "init_fingerprint": fp, "init_fingerprint_scope": "trunk+task_head",
        "train_seconds": round(time.time() - t0, 1), "checkpoint": str(ckpt_dir),
    })
    gf = res["gold_full"]
    print(f"gold_full {gf['auc']:.4f} (n={gf['n']})  gold {res['gold']['auc']:.4f}  "
          f"ragtruth_en {res['ragtruth_en']['auc']:.4f}  "
          f"ragtruth_nonen {res['ragtruth_nonen']['auc']:.4f}")
    out.write_text(json.dumps(res, indent=2))
    print(f"results -> {out}")
    print(f"=== H135 {tag.upper()} DRAW {args.draw} DONE ===", flush=True)


if __name__ == "__main__":
    main()

"""R17-H145 RELATIONAL-ONLY + SCALE/UNIT LANE - arm draws.

FULL-MIX FRESH RETRAIN of the incumbent recipe (amendment A1 (i)): the clean
R9-H105 recipe (mmBERT-base cross-encoder, BCE + DANN lambda 0.02 Ganin ramp,
MAX_LEN 512, BATCH 48, LR 1e-5, OneCycleLR, 1 epoch, flat shuffle, 1,500-char
chunk truncation) on the complete clean public mix (685,670 rows, 12 frozen
groups) PLUS two added lanes, each its own DANN group. Never continued training
on the slice alone - the mixture ratio and the domain-adversarial head are the
catastrophic-forgetting guard.

    relational  R14-H133_lane.parquet, block == "rel"     7,500 rows
                (arm bind_col 3,000 / compare 3,000 / bind_row 1,500)
                DANN group `quant_relational`
    scaleunit   R17-H145_scaleunit.parquet (build 2)      5,672 rows
                DANN group `quant_scale_unit`
    mix         685,670 + 7,500 + 5,672 = 698,842 rows, 14 DANN groups

NOT in scope: the H133 derivation core (42,500 rows) rides nowhere here - this
is the verification-skill carve-out. The H108 lane does not ride along either.

H126 seeding (session ruling 8): `torch.manual_seed(seed)` is issued before
model construction and RE-ISSUED immediately after, before any dropout or
forward; the trunk+task_head init is fingerprinted (blake2b-128) and written
beside the checkpoint. Draw seeds {1: 1145, 2: 2145}.

Pairing caveat (unchanged from R14-H133, load-bearing for the adjudicator): the
control is the BANKED clean pair (`models/R9-H105-mmbert-dann-clean`,
`models/R9-H105-draw2`), which is UNSEEDED in both init and batch order
(pre-H126). The realized comparison is arm-vs-banked-control, not init-paired,
and n_groups is 14 against the control's 12, so the domain head has a different
shape and construction consumes a different RNG draw.

Resume: `perm` is persisted with the weights and the step counter, and the
restart slices it at `start_step * BATCH`, so a resumed run replays the SAME
permutation and every row is still seen exactly once.

Bars (registered; the coordinator adjudicates, this trainer reports):
CO-PRIMARIES bind_col probe >= 0.80, bind_row >= 0.95, scale/unit >= 0.92;
KILL bind_col < 0.70 or scale/unit < 0.88; HOLDs arena mean >= 0.70311, no
subset below control-pair - 0.06, pubmedqa >= 0.5463, gold_full >= 0.8414,
RAGTruth non-EN >= 0.82, anti-gaming untraced near-miss >= 0.7565.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R17-H145_trainer.py --draw 1
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
from torch.utils.data import DataLoader, Subset
from transformers import AutoModel, AutoTokenizer

HERE = pathlib.Path(__file__).parent

MAX_LEN, BATCH, LR = 512, 48, 1e-5
WARMUP_FRAC, CLIP = 0.1, 1.0
LAMBDA_MAX = 0.02
RESUME_EVERY = 1000
DRAW_SEEDS = {1: 1145, 2: 2145}

RELATIONAL = HERE / "R14-H133_lane.parquet"   # v3, filtered to block == "rel"
SCALEUNIT = HERE / "R17-H145_scaleunit.parquet"  # build 2, FINAL
LANE_GROUPS = ("quant_relational", "quant_scale_unit")
PUBLIC_GROUPS = (
    "halueval", "psiloqa", "ragtruth_cn", "ragtruth_de", "ragtruth_en",
    "ragtruth_es", "ragtruth_fr", "ragtruth_hu", "ragtruth_it", "ragtruth_pl",
    "tabfact", "vitaminc",
)
EXPECTED_GROUPS = tuple(sorted(PUBLIC_GROUPS + LANE_GROUPS))
EXPECTED_REL_ROWS = 7_500
EXPECTED_REL_ARMS = {"bind_col": 3_000, "compare": 3_000, "bind_row": 1_500}
EXPECTED_SU_ROWS = 5_672
EXPECTED_CLEAN_ROWS = 685_670
EXPECTED_MIX_ROWS = 698_842


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H108 = _mod("h108", "R10-H108_lane.py")
CHUNK_MAX = H108.M59.CFG.chunk_max_chars


def _lane_rows(df, group):
    """Read a lane through the exact `R10-H108_lane.lane_train` path (claim /
    chunk[:chunk_max_chars] / label.cast(Float32)); the DANN tag is forced to
    `group` so each lane is one frozen group regardless of its own tag column."""
    claims = df["claim"].to_list()
    chunks = [c[:CHUNK_MAX] for c in df["chunk"].to_list()]
    y = df["label"].cast(pl.Float32).to_numpy()
    return claims, chunks, y, [group] * len(claims)


def lanes():
    rel = pl.read_parquet(RELATIONAL).filter(pl.col("block") == "rel")
    arms = {r["arm"]: int(r["count"]) for r in rel["arm"].value_counts().to_dicts()}
    if len(rel) != EXPECTED_REL_ROWS or arms != EXPECTED_REL_ARMS:
        raise SystemExit(
            f"LANE ABORT (relational): {len(rel)} rows {arms} != "
            f"{EXPECTED_REL_ROWS} rows {EXPECTED_REL_ARMS}")
    su = pl.read_parquet(SCALEUNIT)
    if len(su) != EXPECTED_SU_ROWS:
        raise SystemExit(
            f"LANE ABORT (scale/unit): {len(su)} rows != {EXPECTED_SU_ROWS}")
    print(f"lane relational: {len(rel)} rows  {arms}  "
          f"(from {RELATIONAL.name}, block == 'rel')", flush=True)
    print(f"lane scale/unit: {len(su)} rows  "
          f"{len(su['pair_id'].unique())} pairs  (from {SCALEUNIT.name})", flush=True)
    return (_lane_rows(rel, "quant_relational"),
            _lane_rows(su, "quant_scale_unit"))


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


def perm_fingerprint(perm):
    return hashlib.blake2b(
        np.asarray(perm, dtype=np.int64).tobytes(), digest_size=8
    ).hexdigest()


def build_mix():
    claims, chunks, y, tags = H108.public_train()
    n_public = len(y)
    if n_public != EXPECTED_CLEAN_ROWS:
        raise SystemExit(
            f"CENSUS ABORT: clean mix {n_public} (want {EXPECTED_CLEAN_ROWS})")
    lane_counts = {}
    for lc, lk, ly, lt in lanes():
        claims += lc
        chunks += lk
        y = np.concatenate([y, ly]).astype("float32")
        tags += lt
        lane_counts[lt[0]] = len(ly)
    names = tuple(sorted(set(tags)))
    if names != EXPECTED_GROUPS:
        raise SystemExit(
            f"GROUP-MAP ABORT: mix groups {names} != registered {EXPECTED_GROUPS}.")
    if len(y) != EXPECTED_MIX_ROWS:
        raise SystemExit(f"CENSUS ABORT: mix {len(y)} (want {EXPECTED_MIX_ROWS})")
    return claims, chunks, y, tags, n_public, lane_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", type=int, required=True, choices=(1, 2))
    ap.add_argument("--max-steps", type=int, default=0,
                    help="smoke mode: stop after N steps, no checkpoint/eval, ckpt dir -smoke")
    args = ap.parse_args()

    seed = DRAW_SEEDS[args.draw]
    ckpt_dir = HERE.parent.parent / "models" / (
        "R17-H145-smoke" if args.max_steps else f"R17-H145-arm-draw{args.draw}")
    out = HERE / f"R17-H145_arm_draw{args.draw}_result.json"

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    claims, chunks, y, tags, n_public, lane_counts = build_mix()
    n_rows = len(y)
    tag_to_idx = {t: i for i, t in enumerate(EXPECTED_GROUPS)}
    groups = np.array([tag_to_idx[t] for t in tags])
    n_groups = len(EXPECTED_GROUPS)
    counts = {t: int(sum(1 for x in tags if x == t)) for t in EXPECTED_GROUPS}
    print(f"train: {n_rows} rows ({n_public} clean public + "
          f"{lane_counts['quant_relational']} relational + "
          f"{lane_counts['quant_scale_unit']} scale/unit) over "
          f"{n_groups} domains (chance {1.0 / n_groups:.3f})  seed {seed}  "
          f"mean target {y.mean():.3f}", flush=True)
    for t in EXPECTED_GROUPS:
        print(f"  {t:<18} {counts[t]:>7}", flush=True)
    print("", flush=True)

    if args.max_steps == 0 and (n_rows != EXPECTED_MIX_ROWS or n_groups != 14):
        raise SystemExit("CENSUS ABORT after group map")

    tok = AutoTokenizer.from_pretrained(H108.STUDENT)
    base = AutoModel.from_pretrained(H108.STUDENT).cuda()
    base.config.reference_compile = False  # mmBERT/ModernBERT compile path hangs here
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
        perm = np.random.default_rng(seed).permutation(len(ds)).tolist()

    print(f"perm fingerprint {perm_fingerprint(perm)}  (flat shuffle, "
          f"np.random.default_rng({seed}).permutation)\n", flush=True)
    (ckpt_dir / "init_fingerprint.json").write_text(json.dumps(
        {"draw": args.draw, "arm": "h145_relational_scaleunit", "seed": seed,
         "n_groups": n_groups, "group_counts": counts, "lane_rows": lane_counts,
         "clean_rows": n_public, "mix_rows": n_rows, "n_steps": n_steps,
         "scope": "trunk+task_head", "n_params": fp_numel, "blake2b_128": fp,
         "perm_convention": f"np.random.default_rng({seed}).permutation(n_rows), flat",
         "perm_fingerprint": perm_fingerprint(perm)}, indent=2))

    dl = DataLoader(Subset(ds, perm[start_step * BATCH:]), batch_size=BATCH,
                    shuffle=False, collate_fn=ds.collate, num_workers=2)
    domain_lossf = nn.CrossEntropyLoss()

    def save_resume(step):
        tmp = resume_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "perm": perm, "step": step}, tmp)
        tmp.replace(resume_path)  # atomic: a kill mid-write cannot corrupt the resume

    model.train()
    t0 = time.time()
    dom_correct, dom_total = 0, 0
    for i, (enc, yy, gg) in enumerate(dl):
        step = start_step + i
        if args.max_steps and i >= args.max_steps:
            print(f"smoke stop at {args.max_steps} steps", flush=True)
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
        if step % 200 == 0 or (args.max_steps and step % 10 == 0):
            acc = dom_correct / max(dom_total, 1)
            dom_correct, dom_total = 0, 0
            print(f"  step {step}/{n_steps} task {t_loss.item():.4f} "
                  f"domain {d_loss.item():.4f} lam {lam:.4f} domain-acc {acc:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if step and step % RESUME_EVERY == 0:
            save_resume(step + 1)
            print(f"  resume point saved at step {step}", flush=True)

    torch.save({"trunk": base.state_dict(), "task_head": model.task_head.state_dict(),
                "domain_head": model.domain_head.state_dict(), "config": base.config},
               ckpt_dir / "dann_student.pt")
    tok.save_pretrained(ckpt_dir)
    base.save_pretrained(ckpt_dir / "trunk")
    resume_path.unlink(missing_ok=True)  # the draw is done; a stale resume would rerun it
    print(f"\ncheckpoint saved -> {ckpt_dir}\n", flush=True)

    model.eval()
    res = H108.evaluate(model, tok)
    res.update({
        "params_M": round(n_par, 1), "lambda_max": LAMBDA_MAX,
        "lane": "R17-H145 relational sub-block (7,500) + scale/unit family build 2 (5,672)",
        "arm": "h145_relational_scaleunit", "draw": args.draw, "seed": seed,
        "mix": "clean public-only mix + relational + scale/unit "
               "(H133 derivation core and H108 lane NOT included)",
        "mix_rows": n_rows, "clean_rows": n_public, "lane_rows": lane_counts,
        "dann_groups": n_groups, "lane_groups": list(LANE_GROUPS), "group_counts": counts,
        "n_steps": n_steps,
        "perm_fingerprint": perm_fingerprint(perm),
        "init_fingerprint": fp, "init_fingerprint_scope": "trunk+task_head",
        "control": "BANKED clean pair R9-H105 (unseeded, pre-H126) - see the pairing caveat "
                   "in this trainer's docstring",
        "train_seconds": round(time.time() - t0, 1), "checkpoint": str(ckpt_dir),
    })
    gf = res["gold_full"]
    print(f"gold_full {gf['auc']:.4f} (n={gf['n']})  gold {res['gold']['auc']:.4f}  "
          f"ragtruth_en {res['ragtruth_en']['auc']:.4f}  "
          f"ragtruth_nonen {res['ragtruth_nonen']['auc']:.4f}")
    out.write_text(json.dumps(res, indent=2))
    print(f"results -> {out}")
    print(f"=== H145 ARM DRAW {args.draw} DONE ===", flush=True)


if __name__ == "__main__":
    main()

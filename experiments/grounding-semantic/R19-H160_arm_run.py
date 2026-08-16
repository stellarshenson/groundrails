"""R19-H160 SEED-DIVERSE WEIGHT AVERAGING - the two fresh flagship draws.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R19-H160 SEED-DIVERSE WEIGHT AVERAGING AS THE SERVED ARTIFACT" (2026-08-14
~10:45). The arm tests a PROTOCOL amendment: redefine the served artifact from
"one draw" to the uniform elementwise average of the k draws the official-record
doctrine already requires. Its PRIMARY needs an UNSEEN soup, hence two fresh
draws of the R18-H150 flagship recipe VERBATIM at seeds 3150 and 4150.

The H150 dispatcher (`R18-H150_arm_run.py`) - and through it the banked G1 twin
trainer and reader (`R16-H142_G1_arm.py`, `R16-H142_G1_reads.py`) - are BANKED:
they are the provenance of the current flagship and are NOT edited. This wrapper
rebinds the run-scoped constants (seed, checkpoint, result paths) and reuses the
H150 mix assembly unchanged, the R18-H156 wrapper pattern.

    recipe      the H150 flagship VERBATIM - clean public 685,670
                + H146 misbind lane 30,000 (group `quant_misbind`)
                + H150 unit_swap lane 5,540 (group `quant_scale_unit`)
                = 721,210 rows, 14 DANN groups; evidence UNTRUNCATED, 1,500/750
                windowed presentation, MIL max-over-windows BCE, full trunk at
                lr 1e-5 OneCycleLR 1 epoch, DANN lambda 0.02 Ganin ramp, adapter
                FROZEN at its zero init. NO EMA, NO window dropout (H152-only)
    seed        3150 (draw 3) / 4150 (draw 4) - governs task-head init, data
                order and the dropout stream jointly, which is the divergence
                the soup exploits; the three are NOT decoupled
    checkpoint  models/R19-H160-arm-draw3 / models/R19-H160-arm-draw4 - the
                names the banked readers resolve as models/<arm>-arm-draw<N>,
                so no symlink indirection is needed for either draw

EXECUTOR. The registered batch geometry (<= 48 sets / <= 96 pairs, untruncated
windows) does not fit the free cards monolithically: the banked H152 vram probe
priced the full window stack at 36.96 GB allocated / 56.56 GB reserved, against
GPU0's 24 GB and GPU2's 32 GB (a single 40-window row carries ~15 GB of
activations on its own, so set-level micro-batching cannot bound it). Training
therefore runs through `R19-H160_split_exec.py` - the cotangent window-chunked
executor, the R18-H156 lineage specialised back to the MIL max-BCE task loss -
with a step-level equivalence proof (`R19-H160_exec_equivalence.py`) against the
monolithic reference below. `--stage vram_probe` re-measures the monolithic peak
at REGISTERED geometry on the card actually in use, which is the placement call.

`train_step` is the MONOLITHIC REFERENCE: the banked G1 twin training step
transcribed from `R16-H142_G1_arm.main()`'s loop body with no change of any
kind. It is the equivalence proof's reference arm and is never used to train a
registered draw.

Stages:
    train       train + the in-domain suite (gold, gold_full, RAGTruth EN + 7),
                dispatched into the cotangent split executor
    windowed    the PRIMARY blind windowed decomposed-min arena read, through
                the banked reader unchanged (checkpoint + output path rebound)
    census      CPU-only dry run - mix census, window-census cross-check, the
                draw's init and permutation fingerprints, then exit
    vram_probe  the MONOLITHIC path at registered geometry for a handful of
                steps including the mix's deepest window stacks, reporting
                torch.cuda.max_memory_allocated() - the placement measurement

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
      uv run python experiments/grounding-semantic/R19-H160_arm_run.py \
          --stage train --draw 3
"""

import os

# GPU1 carried R19-H159 exclusively while that arm trained; the reservation was
# LIFTED 2026-08-14 16:38 when H159 was killed at draw 1 and released the card.
# The banked trainer defaults CUDA_VISIBLE_DEVICES to "1" at import, so the
# placement is still asserted BEFORE it is imported and an unset variable stays a
# hard abort rather than a silent default onto whatever card holds index 1.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    raise SystemExit("GPU PLACEMENT ABORT: CUDA_VISIBLE_DEVICES is unset - set it "
                     "explicitly (0, 1 or 2)")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = pathlib.Path(__file__).parent

MASK_FILL = -1e9  # the banked scatter-amax empty-fill

DRAWS = {
    3: {"seed": 3150, "ckpt": "R19-H160-arm-draw3",
        "train_out": "R19-H160_arm_draw3_result.json",
        "read_out": "R19-H160_arm_draw3_{mode}_result.json"},
    4: {"seed": 4150, "ckpt": "R19-H160-arm-draw4",
        "train_out": "R19-H160_arm_draw4_result.json",
        "read_out": "R19-H160_arm_draw4_{mode}_result.json"},
}

# Every banked permutation fingerprint the H160 draws must stay distinct from:
# 1142 (G1 twin), 2142 (H142-T d2), 1150 (H150 d1), 2150 (H150 d2),
# 3151/3152 (H152 d1/d2), 51551/51552 (H155 5155a/5155b), 1156 (H156 d1).
BANKED_PERM_FPS = {"a8b2cf491a236bba", "eebe673dabeef46f", "7d13f9ac86a79574",
                   "8fb06248240a78e1", "5e3de18e48c57632", "70d71966b2f7ebcb",
                   "07fe223aeb6686fb", "76a057088e834027"}

# vram probe: how many head-of-run steps and how many hand-picked deep batches
PROBE_STEPS = 8
PROBE_DEEP = 4


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rebind(arm, draw):
    """Seed, H150 mix, group map, checkpoint and result path - nothing else.
    The twin guard stands: this wrapper only ever dispatches the adapter-frozen
    run, and the H150 mix assembly is the banked one, imported not copied."""
    if arm.RUNS["twin"]["use_adapter"]:
        raise SystemExit("TWIN INTEGRITY ABORT: the dispatched run trains the adapter")
    h150 = _mod("h150arm", "R18-H150_arm_run.py")
    cfg = DRAWS[draw]
    arm.SEED = cfg["seed"]  # save_final records the module-global SEED
    arm.EXPECTED_GROUPS = h150.EXPECTED_GROUPS
    arm.EXPECTED_MIX_ROWS = h150.EXPECTED_MIX_ROWS
    arm.build_mix = h150.make_build_mix(arm)  # the H150 3-source mix, verbatim
    arm.RUNS["twin"]["ckpt"] = cfg["ckpt"]
    arm.RUNS["twin"]["out"] = cfg["train_out"]
    return arm


# --- the monolithic reference step -------------------------------------------------

_DOMAIN_LOSSF = nn.CrossEntropyLoss()


def train_step(arm, model, opt, sched, enc, si, n_sets, yy, gg, lam):
    """The banked G1 twin step, transcribed from `R16-H142_G1_arm.main()`'s loop
    body verbatim: bf16 autocast on the trunk encode, fp32 heads, MIL
    max-over-windows BCE per row, per-pair domain CE off the gradient-reversal
    layer, clip 1.0, one optimizer step. THE EQUIVALENCE REFERENCE - never used
    to train a registered draw."""
    with torch.autocast("cuda", dtype=arm.TRAIN_ENCODE_DTYPE):
        cls = model.encode(enc)
    cls = cls.float()  # heads, adapter and loss in fp32, as in G0
    lg = model.logits_from_cls(cls, si, n_sets)
    agg = torch.full((n_sets,), MASK_FILL, device="cuda").scatter_reduce(
        0, si, lg, reduce="amax")  # MIL: max over the row's windows
    t_loss = F.binary_cross_entropy_with_logits(agg, yy)
    domain_logit = model.domain_head(arm.H108.GradReverse.apply(cls, lam))
    d_loss = _DOMAIN_LOSSF(domain_logit, gg[si])
    loss = t_loss + d_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), arm.CLIP)
    opt.step()
    sched.step()
    opt.zero_grad()
    return t_loss, d_loss, domain_logit


# --- the vram probe (the placement measurement) -------------------------------------


def vram_probe(draw, n_steps=PROBE_STEPS, n_deep=PROBE_DEEP):
    """The MONOLITHIC path at REGISTERED geometry, a handful of steps including
    the mix's deepest window stacks, reporting peak allocated/reserved. Answers
    one question only: does the banked unsplit trainer fit this card."""
    arm = rebind(_mod("g1arm", "R16-H142_G1_arm.py"), draw)
    from transformers import AutoModel, AutoTokenizer

    seed = DRAWS[draw]["seed"]
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"=== R19-H160 VRAM PROBE draw {draw} (seed {seed})  "
          f"{time.strftime('%F %T')} ===", flush=True)
    print(f"card: {torch.cuda.get_device_name(0)}  total {total_gb:.2f} GB  "
          f"MONOLITHIC path, registered geometry "
          f"{arm.SETS_PER_BATCH} sets / {arm.PAIRS_PER_BATCH} pairs", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    claims, wsets, y, tags = arm.build_mix()
    n_rows = len(y)
    tag_to_idx = {t: i for i, t in enumerate(arm.EXPECTED_GROUPS)}
    groups = np.array([tag_to_idx[t] for t in tags])
    n_groups = len(arm.EXPECTED_GROUPS)
    cens, sizes = arm.window_census(wsets, tags)
    print(f"window census: {cens['total_pairs']} pairs, mean set "
          f"{cens['mean_windows_per_row']:.3f}, max {cens['max_windows_per_row']}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(arm.H108.STUDENT)
    base = AutoModel.from_pretrained(arm.H108.STUDENT)
    base.config.reference_compile = False
    base = base.cuda()
    model = arm.DANNAdapterStudent(base, n_groups).cuda()
    torch.manual_seed(seed)
    for n, p in model.named_parameters():
        if n.startswith(arm.ADAPTER_PREFIXES):
            p.requires_grad_(False)
    perm = np.random.default_rng(seed).permutation(n_rows)
    batches = arm.pack_batches(perm, sizes)
    opt = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad],
          "lr": arm.LR}], lr=arm.LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[arm.LR], total_steps=len(batches),
        pct_start=arm.WARMUP_FRAC, anneal_strategy="linear")

    # the deepest registered batches by pair count - where the peak actually is
    deep = sorted(range(len(batches)),
                  key=lambda s: -sum(int(sizes[i]) for i in batches[s]))[:n_deep]
    seq = [(s, "head") for s in range(n_steps)] + [(s, "deep") for s in deep]

    model.train()
    recs = []
    for step, kind in seq:
        batch = batches[step]
        p = step / max(len(batches) - 1, 1)
        lam = arm.LAMBDA_MAX * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        enc, si = arm.encode_batch(tok, claims, wsets, batch)
        yy = torch.as_tensor(y[batch], device="cuda")
        gg = torch.as_tensor(groups[batch], device="cuda")
        t_loss, _d_loss, _ = train_step(arm, model, opt, sched, enc, si,
                                        len(batch), yy, gg, lam)
        rec = {"step": step, "kind": kind, "n_sets": len(batch),
               "n_pairs": int(si.numel()),
               "task_loss": float(t_loss.detach()),
               "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
               "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
               "secs": round(time.time() - t0, 2)}
        recs.append(rec)
        print(f"  [{kind}] step {step} sets {rec['n_sets']:2d} pairs "
              f"{rec['n_pairs']:3d} task {rec['task_loss']:.4f}  peak alloc "
              f"{rec['peak_alloc_gb']:.2f} GB reserved "
              f"{rec['peak_reserved_gb']:.2f} GB  ({rec['secs']:.1f}s)", flush=True)

    peak_a = max(r["peak_alloc_gb"] for r in recs)
    peak_r = max(r["peak_reserved_gb"] for r in recs)
    fits = peak_r <= 0.85 * total_gb
    out = HERE / f"R19-H160_vram_probe_draw{draw}.json"
    out.write_text(json.dumps({
        "arm": "R19-H160", "draw": draw, "seed": seed, "path": "monolithic",
        "card": torch.cuda.get_device_name(0), "card_total_gb": round(total_gb, 2),
        "geometry": {"sets_per_batch": arm.SETS_PER_BATCH,
                     "pairs_per_batch": arm.PAIRS_PER_BATCH},
        "peak_alloc_gb": round(peak_a, 2), "peak_reserved_gb": round(peak_r, 2),
        "bar_reserved_gb": round(0.85 * total_gb, 2),
        "monolithic_fits": bool(fits),
        "median_secs_per_step": round(float(np.median([r["secs"] for r in recs])), 2),
        "records": recs,
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    }, indent=2))
    print(f"\nPEAK allocated {peak_a:.2f} GB  reserved {peak_r:.2f} GB  "
          f"total {total_gb:.2f} GB  bar {0.85 * total_gb:.2f} GB", flush=True)
    print(f"VERDICT: monolithic {'FITS' if fits else 'DOES NOT FIT'} this card "
          f"-> {out}", flush=True)
    return fits


# --- driver ------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("train", "windowed", "census", "vram_probe"))
    ap.add_argument("--draw", type=int, required=True, choices=tuple(DRAWS))
    ap.add_argument("--max-steps", type=int, default=0,
                    help="smoke mode for --stage train: stop after N steps")
    args = ap.parse_args()
    cfg = DRAWS[args.draw]

    if args.stage == "vram_probe":
        vram_probe(args.draw)
        return

    if args.stage == "windowed":
        reads = _mod("g1reads", "R16-H142_G1_reads.py")
        rebind(reads.ARM, args.draw)
        reads.out_path = lambda run, mode: HERE / cfg["read_out"].format(mode=mode)
        sys.argv = ["reads", "--run", "twin", "--mode", "windowed"]
        reads.main()
        return

    if args.stage == "census":
        arm = rebind(_mod("g1arm", "R16-H142_G1_arm.py"), args.draw)
        sys.argv = ["arm", "--run", "twin", "--census-only"]
        arm.main()
        return

    split = _mod("h160split", "R19-H160_split_exec.py")
    split.train(args.draw, max_steps=args.max_steps)


if __name__ == "__main__":
    main()

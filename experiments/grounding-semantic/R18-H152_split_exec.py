"""R18-H152 SPLIT EXECUTOR - the H152 regularized-twin recipe, memory-bounded.

The banked training path (`R18-H152_arm_run.py::train_step` over the banked G1
twin trainer `R16-H142_G1_arm.py` - both READ-ONLY, never modified here)
encodes a whole registered batch - up to 96 (claim, window) pairs - in ONE
monolithic forward+backward. Measured on the H152 draw-1 config that peaks at
36.96 GB allocated / 56.56 GB reserved (logs/R18-H152_gpu0_vram_probe.log),
over the 24 GB card's 21.81 GB bar, and the peak is set by the WINDOW STACK:
one 40-window row carries ~15 GB of activations on its own, so set-level
micro-batching alone cannot bound memory. This executor reproduces the same
recipe with the window stack itself split:

    pass A   every window of the batch scored under no_grad in
             PASS_A_CHUNK-window chunks - the MIL argmax selection and the
             window-dropout mask are computed from these detached logits,
             exactly as the banked code computes them from ITS detached
             logits before any backward runs
    pass B   PASS_B_CHUNK windows per grad-carrying forward; the chunk loss
             is (selected-window BCE terms) + (chunk domain CE), backwarded
             immediately, so peak activation memory is bounded by one chunk
             while the accumulated gradient covers the full registered batch
    step     ONE clip / opt.step / sched.step / ema_update per registered
             batch - the 14,300-step OneCycle schedule and the EMA cadence
             see optimizer-step semantics identical to the banked run

Why the two passes compute the SAME function and the SAME gradient:

  * MIL max-BCE: d(max)/dtheta flows only through the per-set maxima. This
    torch build's scatter-amax backward splits the upstream gradient EQUALLY
    over tying maxima (asserted in kernel_self_check); pass B delivers
    BCE'(logit_w)/(T*n_sets) to each of the T tying kept windows of a set -
    the identical per-window gradient
  * window dropout: the mask is drawn by the wrapper's OWN
    apply_window_dropout on the full [P] detached logit vector - identical
    shape, identical RNG draw, identical mask - and the MIL selection happens
    on the masked logits, mirroring the banked mask-then-amax order
  * DANN domain loss: the banked domain head consumes the per-window [CLS]
    of EVERY pair (targets gg[si]), so pass B runs the same head per chunk
    over all windows. The head's nn.Dropout(0.1) is replaced by the mask
    CAPTURED ONCE at full [P, hidden] shape - bitwise the mask the banked
    forward would draw (asserted in kernel_self_check) - so per-step RNG
    consumption is, in banked order, exactly rand(P) then dropout([P, 256])
    and the streams stay in lock-step across steps. Pass A and pass B
    themselves draw nothing (every mmBERT dropout is 0.0; audited per step
    when audit_rng=True)
  * the adapter side-head is frozen at its zero init in the twin recipe: its
    term is exactly +0.0 in every logit and its zero output layer blocks all
    gradient into the ctx path, so pass B reads the task head directly

What is NOT bit-identical: chunked-vs-monolithic forwards move padding
boundaries and kernel tile decompositions (bf16-scale numeric noise), and the
per-chunk backward accumulates gradients in a different float order than the
single autograd traversal. Both are the same class of variation as a benign
kernel-choice change; R18-H152_exec_equivalence.py measures the split against
exactly such a reference-vs-reference noise floor.

Run (the draw LAUNCH decision is the coordinator's; the equivalence proof
runs through R18-H152_exec_equivalence.py):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
    uv run python experiments/grounding-semantic/R18-H152_split_exec.py --draw 2
"""

import argparse
import importlib.util
import json
import os
import pathlib
import time

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

HERE = pathlib.Path(__file__).parent

PASS_A_CHUNK = 32  # no-grad scoring windows per chunk
PASS_B_CHUNK = 8   # grad-carrying windows per chunk - the memory bound
MASK_FILL = -1e9   # the banked scatter-amax empty-fill (dropped windows)


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def kernel_self_check():
    """Abort unless this torch build shows the two kernel semantics the
    equivalence argument stands on: scatter-amax backward splits the upstream
    gradient EQUALLY over tying maxima, and F.dropout applied to a ones tensor
    yields the exact mask the same RNG state applies to any tensor of that
    shape (so the banked domain-head dropout can be captured whole at full
    [P, hidden] shape and applied per chunk, bitwise)."""
    x = torch.tensor([1.0, 2.0, 2.0, 0.5, 2.0], device="cuda",
                     requires_grad=True)
    idx = torch.tensor([0, 0, 0, 1, 0], device="cuda")
    out = torch.full((2,), MASK_FILL, device="cuda").scatter_reduce(
        0, idx, x, reduce="amax")
    out.sum().backward()
    want = torch.tensor([0.0, 1 / 3, 1 / 3, 1.0, 1 / 3], device="cuda")
    if not torch.allclose(x.grad, want):
        raise SystemExit(f"KERNEL ABORT: scatter-amax tie split is not equal-share "
                         f"(got {x.grad.tolist()})")
    v = torch.randn(96, 256, device="cuda")
    torch.cuda.manual_seed(0)
    a = F.dropout(v, 0.1, True)
    torch.cuda.manual_seed(0)
    b = v * F.dropout(torch.ones_like(v), 0.1, True)
    if not torch.equal(a, b):
        raise SystemExit("KERNEL ABORT: dropout ones-capture is not bitwise")
    print("kernel self-check: scatter-amax splits ties equally; dropout "
          "ones-capture is bitwise", flush=True)


def split_train_step(arm, w152, model, opt, sched, tok, claims, wsets, batch,
                     y_batch, g_batch, lam, ema, audit_rng=False,
                     pass_a_chunk=PASS_A_CHUNK, pass_b_chunk=PASS_B_CHUNK):
    """One registered batch = one optimizer step, computed in window chunks.

    Mirrors w152.train_step (the banked H152 step: window dropout on the MIL
    max input, per-pair domain CE off the GRL, EMA after the optimizer step)
    with the window stack split as described in the module docstring. Returns
    the per-step measurements the equivalence proof records.
    """
    n_sets = len(batch)
    flat_c, flat_w, si_list = [], [], []
    for r, i in enumerate(batch):  # the banked flattening order
        for w in wsets[i]:
            flat_c.append(claims[i])
            flat_w.append(w)
            si_list.append(r)
    P = len(si_list)
    si = torch.tensor(si_list, dtype=torch.long, device="cuda")
    yy = torch.as_tensor(y_batch, device="cuda")
    gg = torch.as_tensor(g_batch, device="cuda")

    def encode_slice(a, b):
        enc = tok(flat_c[a:b], flat_w[a:b], return_tensors="pt", padding=True,
                  truncation=True, max_length=arm.MAX_LEN)
        return {k: v.cuda() for k, v in enc.items()}

    # pass A - no-grad scoring of every window, chunked; must consume no RNG
    rng0 = torch.cuda.get_rng_state() if audit_rng else None
    hidden = model.trunk.config.hidden_size
    cls_all = torch.empty(P, hidden, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        for a in range(0, P, pass_a_chunk):
            b = min(a + pass_a_chunk, P)
            with torch.autocast("cuda", dtype=arm.TRAIN_ENCODE_DTYPE):
                cls_all[a:b] = model.encode(encode_slice(a, b)).float()
        ctx = model.pool_ctx(cls_all, si, n_sets)
        lg = model.pair_logits(cls_all, ctx[si]).detach()
    if audit_rng and not torch.equal(rng0, torch.cuda.get_rng_state()):
        raise RuntimeError("RNG AUDIT: pass A consumed RNG - the mmBERT trunk "
                           "is expected to draw nothing (all dropouts 0.0)")

    # RNG draw 1 of the step: the wrapper's own window dropout at full [P]
    lg_masked, drop_mask, n_dropped, n_eligible = w152.apply_window_dropout(
        lg, si, n_sets, w152.WINDOW_DROPOUT_P)

    # the banked MIL read on the masked logits (detached) - the logged loss
    agg = torch.full((n_sets,), MASK_FILL, device="cuda").scatter_reduce(
        0, si, lg_masked, reduce="amax")
    t_loss_val = float(F.binary_cross_entropy_with_logits(agg, yy))

    # per-set selected windows = tying KEPT maxima; scale carries the amax
    # equal-share tie split and the BCE mean over sets
    sel = (lg_masked == agg[si]) & ~drop_mask
    ties = torch.zeros(n_sets, device="cuda").index_add_(0, si, sel.float())
    sel_scale = sel.float() / ties[si] / n_sets  # per window; 0 if not selected

    # RNG draw 2 of the step: capture the banked domain-dropout mask whole
    dnn_hidden = model.domain_head[0].out_features
    dmask = F.dropout(torch.ones(P, dnn_hidden, device="cuda"),
                      p=model.domain_head[2].p, training=True)

    # pass B - grad-carrying chunks; per-chunk backward accumulates the batch
    # gradient over all windows (domain CE sees every pair, as banked)
    rng1 = torch.cuda.get_rng_state() if audit_rng else None
    d_loss_sum, dom_correct = 0.0, 0
    for a in range(0, P, pass_b_chunk):
        b = min(a + pass_b_chunk, P)
        with torch.autocast("cuda", dtype=arm.TRAIN_ENCODE_DTYPE):
            cls_c = model.encode(encode_slice(a, b))
        cls_c = cls_c.float()  # heads and loss in fp32, as in the banked trainer
        lg_c = model.task_head(cls_c).squeeze(-1)  # adapter term is exactly +0.0
        si_c = si[a:b]
        t_term = (F.binary_cross_entropy_with_logits(
            lg_c, yy[si_c], reduction="none") * sel_scale[a:b]).sum()
        h = model.domain_head[0](arm.H108.GradReverse.apply(cls_c, lam))
        h = model.domain_head[1](h)  # the banked ReLU
        h = h * dmask[a:b]           # the captured banked dropout mask
        dl_c = model.domain_head[3](h)
        d_term = F.cross_entropy(dl_c, gg[si_c], reduction="sum") / P
        (t_term + d_term).backward()
        d_loss_sum += float(d_term.detach())
        dom_correct += int((dl_c.detach().argmax(-1) == gg[si_c]).sum())
    if audit_rng and not torch.equal(rng1, torch.cuda.get_rng_state()):
        raise RuntimeError("RNG AUDIT: pass B consumed RNG - the chunked path "
                           "must draw nothing (dropout mask pre-captured)")

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), arm.CLIP)
    opt.step()
    sched.step()
    opt.zero_grad()
    w152.ema_update(ema, model)
    return {"task_loss": t_loss_val, "domain_loss": d_loss_sum,
            "dom_correct": dom_correct, "dom_total": P, "n_pairs": P,
            "grad_norm": float(grad_norm), "drop_mask": drop_mask,
            "n_dropped": n_dropped, "n_eligible": n_eligible}


# --- full-draw driver (the coordinator decides launches) -----------------------


def train(draw, max_steps=0):
    """The H152 wrapper's train() re-orchestrated over split_train_step. Every
    data/model helper is the banked one; the loop differences are exactly the
    window-chunked step and a resume payload identical in keys and semantics
    to the wrapper's (raw + EMA + RNG state), so a draw can migrate between
    executors at a resume boundary."""
    arm = _mod("g1arm", "R16-H142_G1_arm.py")
    w152 = _mod("h152", "R18-H152_arm_run.py")
    cfg = w152.DRAWS[draw]
    seed = cfg["seed"]
    ckpt_dir = arm.ROOT / "models" / cfg["ckpt"]
    out = HERE / cfg["train_out"]

    print(f"=== R18-H152 SPLIT EXECUTOR draw {draw} (seed {seed})  "
          f"{time.strftime('%F %T')} ===", flush=True)
    print(f"twin protocol verbatim + EMA + window dropout, window-chunked "
          f"(pass A {PASS_A_CHUNK} no-grad, pass B {PASS_B_CHUNK} grad)", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})", flush=True)
    kernel_self_check()
    torch.manual_seed(seed)
    np.random.seed(seed)

    claims, wsets, y, tags = arm.build_mix()
    n_rows = len(y)
    tag_to_idx = {t: i for i, t in enumerate(arm.EXPECTED_GROUPS)}
    groups = np.array([tag_to_idx[t] for t in tags])
    n_groups = len(arm.EXPECTED_GROUPS)
    counts = {t: int(sum(1 for x in tags if x == t)) for t in arm.EXPECTED_GROUPS}
    cens, sizes = arm.window_census(wsets, tags)
    if cens["mean_windows_per_row"] < arm.MIN_MEAN_WINDOWS:
        raise SystemExit("WINDOW-CENSUS ABORT: untruncated evidence did not "
                         "produce multi-window sets")
    if cens["max_windows_per_row"] > arm.PAIRS_PER_BATCH:
        raise SystemExit("BATCH-CAP ABORT: a training set exceeds the pair cap")
    print(f"window census: {cens['total_pairs']} pairs, mean set "
          f"{cens['mean_windows_per_row']:.3f}, max {cens['max_windows_per_row']}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(arm.H108.STUDENT)
    base = AutoModel.from_pretrained(arm.H108.STUDENT)
    base.config.reference_compile = False
    base = base.cuda()
    model = arm.DANNAdapterStudent(base, n_groups)
    model = model.cuda()
    torch.manual_seed(seed)  # H126 ruling 8: re-issue after construction
    fp, fp_numel = arm.init_fingerprint(model)
    if not arm.zero_init_ok(model):
        raise SystemExit("ZERO-INIT ABORT: the adapter output layer is not zero")
    for n, p in model.named_parameters():
        if n.startswith(arm.ADAPTER_PREFIXES):
            p.requires_grad_(False)
    print(f"init fingerprint (trunk+task_head, {fp_numel} params): {fp}", flush=True)

    perm = np.random.default_rng(seed).permutation(n_rows)
    batches = arm.pack_batches(perm, sizes)
    n_steps = len(batches)
    print(f"perm fingerprint {arm.perm_fingerprint(perm)}  {n_steps} steps", flush=True)

    old_params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": old_params, "lr": arm.LR}], lr=arm.LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[arm.LR], total_steps=n_steps, pct_start=arm.WARMUP_FRAC,
        anneal_strategy="linear")
    ema = w152.ema_init(model)
    n_ema_updates = 0
    tot_dropped, tot_eligible = 0, 0

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_path = ckpt_dir / "resume.pt"
    start_step = 0
    if resume_path.exists():
        st = torch.load(resume_path, map_location="cuda", weights_only=False)
        if "ema" not in st:
            raise SystemExit("RESUME ABORT: no EMA state - not an H152 resume point")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_step = st["step"]
        if st["perm_fingerprint"] != arm.perm_fingerprint(perm):
            raise SystemExit("RESUME ABORT: the persisted permutation does not match")
        ema = {k: v.cuda() for k, v in st["ema"].items()}
        n_ema_updates = st.get("ema_updates", start_step)
        tot_dropped, tot_eligible = st.get("drop_stats", (0, 0))
        torch.set_rng_state(st["rng_cpu"].cpu())
        torch.cuda.set_rng_state(st["rng_cuda"].cpu())
        print(f"resumed from {resume_path} at step {start_step}/{n_steps}", flush=True)

    (ckpt_dir / "init_fingerprint.json").write_text(json.dumps(
        {"arm": f"h152_regularized_twin_draw{draw}", "executor": "split",
         "adapter_active": False, "seed": seed, "n_groups": n_groups,
         "group_counts": counts, "mix_rows": n_rows, "n_steps": n_steps,
         "scope": "trunk+task_head", "n_params": fp_numel, "blake2b_128": fp,
         "perm_fingerprint": arm.perm_fingerprint(perm),
         "window_census": cens,
         "split": {"pass_a_chunk": PASS_A_CHUNK, "pass_b_chunk": PASS_B_CHUNK},
         "regularizers": {
             "ema": {"decay": w152.EMA_DECAY, "scope": list(w152.EMA_SCOPE),
                     "served_checkpoint": "EMA copy"},
             "window_dropout": {"p": w152.WINDOW_DROPOUT_P, "training_only": True},
         }}, indent=2))

    def save_resume(step):
        tmp = resume_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step,
                    "perm_fingerprint": arm.perm_fingerprint(perm),
                    "ema": ema, "ema_updates": n_ema_updates,
                    "drop_stats": (tot_dropped, tot_eligible),
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state()}, tmp)
        tmp.replace(resume_path)

    model.train()
    t0 = time.time()
    dom_correct, dom_total = 0, 0
    for step in range(start_step, n_steps):
        if max_steps and step - start_step >= max_steps:
            print(f"smoke stop at {max_steps} steps", flush=True)
            return
        batch = batches[step]
        p = step / max(n_steps - 1, 1)
        lam = arm.LAMBDA_MAX * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
        m = split_train_step(arm, w152, model, opt, sched, tok, claims, wsets,
                             batch, y[batch], groups[batch], lam, ema,
                             audit_rng=True)
        n_ema_updates += 1
        tot_dropped += m["n_dropped"]
        tot_eligible += m["n_eligible"]
        dom_correct += m["dom_correct"]
        dom_total += m["dom_total"]
        if not np.isfinite(m["task_loss"]):
            raise RuntimeError(f"diverged at step {step}")
        if step % 200 == 0:
            print(f"  step {step}/{n_steps} task {m['task_loss']:.4f} "
                  f"domain {m['domain_loss']:.4f} lam {lam:.4f} "
                  f"domain-acc {dom_correct / max(dom_total, 1):.3f} "
                  f"ema-delta {w152.ema_delta(ema, model):.3e} "
                  f"wdrop {tot_dropped}/{tot_eligible}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            dom_correct, dom_total = 0, 0
        if step and step % arm.RESUME_EVERY == 0:
            save_resume(step + 1)
            print(f"  resume point saved at step {step} (raw + EMA)", flush=True)

    final_delta = w152.ema_delta(ema, model)
    w152.ema_load_(model, ema)  # the served checkpoint is the EMA copy
    print(f"training done - EMA loaded (delta {final_delta:.6e}, "
          f"{n_ema_updates} updates; wdrop {tot_dropped}/{tot_eligible})", flush=True)
    arm.save_final(model, base, tok, ckpt_dir, False)
    resume_path.unlink(missing_ok=True)
    print(f"checkpoint saved -> {ckpt_dir} (served weights = EMA copy)", flush=True)

    model.eval()
    res = arm.evaluate(model, tok)
    res.update({
        "run": "twin", "draw": draw, "arm": f"h152_regularized_twin_draw{draw}",
        "executor": "split",
        "split": {"pass_a_chunk": PASS_A_CHUNK, "pass_b_chunk": PASS_B_CHUNK,
                  "equivalence_proof": "R18-H152_exec_equivalence.json"},
        "adapter_active": False, "seed": seed,
        "init_fingerprint": fp, "perm_fingerprint": arm.perm_fingerprint(perm),
        "window_census": cens, "n_steps": n_steps,
        "train_seconds": round(time.time() - t0, 1), "checkpoint": str(ckpt_dir),
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    })
    out.write_text(json.dumps(res, indent=2))
    print(f"results -> {out}", flush=True)
    print(f"=== H152 SPLIT DRAW {draw} TRAIN+INDOMAIN DONE ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", type=int, choices=(1, 2), required=True)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="smoke mode: stop after N steps, no checkpoint/eval")
    args = ap.parse_args()
    train(args.draw, max_steps=args.max_steps)


if __name__ == "__main__":
    main()

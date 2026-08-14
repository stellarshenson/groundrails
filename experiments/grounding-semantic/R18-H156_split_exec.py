"""R18-H156 COTANGENT SPLIT EXECUTOR - the aggregator twin, memory-bounded.

The H156 monolithic reference (`R18-H156_arm_run.py::train_step` over the
banked G1 twin trainer `R16-H142_G1_arm.py` - both dispatched, never modified)
encodes a whole registered batch - up to 96 (claim, window) pairs - in ONE
monolithic forward+backward. The H152 vram probe priced that geometry at
36.96 GB allocated / 56.56 GB reserved (logs/R18-H152_gpu0_vram_probe.log),
over the 32 GB card's budget: the peak is set by the WINDOW STACK, so the
stack itself is split. Where the banked H152 split executor routed the MIL
max gradient through a selection mask, H156's task loss couples EVERY window
of a set through the aggregator (softmax attention + gated max), so the split
is generalised to cotangent form:

    pass A   every window of the batch encoded under no_grad in
             PASS_A_CHUNK-window chunks, giving detached per-window leaves
             e_i ([CLS], fp32) and s_i (task logit). The aggregator forward
             then runs on those leaves with autograd ENABLED: L_task =
             BCE(s_agg, y), mean over sets (the banked reduction). Its
             backward populates the aggregator head's parameter grads and
             yields the exact cotangents cs_i = dL/ds_i and ce_i = dL/de_i
             at the step's forward values
    pass B   PASS_B_CHUNK windows per grad-carrying re-encode; the chunk loss
             is sum_i∈chunk (cs_i * s_i(theta) + ce_i * e_i(theta)) + chunk
             domain CE, backwarded immediately, so peak activation memory is
             bounded by one chunk while the accumulated gradient covers the
             full registered batch
    step     ONE clip / opt.step / sched.step per registered batch over trunk
             + task head + domain head + aggregator head params (single
             AdamW, lr 1e-5, OneCycle as banked) - the OneCycle schedule sees
             optimizer-step semantics identical to the monolithic run

Why the two passes compute the SAME function and the SAME gradient:

  * chain rule: write the task loss as L(theta, phi) = l(s(theta), e(theta),
    phi) with s/e the per-window logit/CLS leaves and phi the head params.
    dL/dtheta = sum_i (cs_i ds_i/dtheta + ce_i de_i/dtheta) with the
    cotangents evaluated at the step's own forward values - pass A evaluates
    them, pass B applies them. Each partial holds the sibling leaf fixed,
    which is exactly what the two-leaf factorization computes; chunk
    boundaries change only the ORDER of float accumulation
  * the max inside s_agg: this torch build's scatter-amax backward splits the
    upstream gradient EQUALLY over tying maxima (asserted in
    kernel_self_check); the same op carries the max in the monolithic
    reference and in pass A, so tie routing is identical by construction
  * the softmax attention: w depends on e only, and its per-set max-shift is
    detached (the shift cancels in the softmax Jacobian), so pass A's ce_i
    carries the full attention path into the trunk
  * DANN domain loss: unchanged from the banked recipe - the domain head
    consumes the per-window [CLS] of EVERY pair (targets gg[si]), Ganin ramp
    to lambda 0.02. The head's nn.Dropout(0.1) is replaced by the mask
    CAPTURED ONCE at full [P, hidden] shape - bitwise the mask the monolithic
    forward would draw (asserted in kernel_self_check) - so per-step RNG
    consumption is exactly ONE dropout([P, 256]) draw in both paths and the
    streams stay in lock-step across steps. Pass A and pass B themselves draw
    nothing (every mmBERT dropout is 0.0; audited per step when
    audit_rng=True)
  * the adapter side-head is frozen at its zero init in the twin recipe: its
    term is exactly +0.0 in every logit and its zero output layer blocks all
    gradient into the ctx path, so pass B reads the task head directly
  * NO H152 regularizers: no EMA, no window dropout - H156 forks the H150
    recipe (3-source mix, 14 DANN groups); the served checkpoint is the RAW
    endpoint and the resume payload carries the banked core keys (model, opt,
    sched, step, perm_fingerprint, rng_cpu, rng_cuda) plus "agg" for the head
    - the H152-only keys (ema, ema_updates, drop_stats) do not exist here

What is NOT bit-identical: chunked-vs-monolithic forwards move padding
boundaries and kernel tile decompositions (bf16-scale numeric noise), the
detached-then-reencoded leaves see pass-A no-grad and pass-B grad kernels,
and the per-chunk backward accumulates gradients in a different float order
than the single autograd traversal. All are the same class of variation as a
benign kernel-choice change; R18-H156_exec_equivalence.py measures the split
against exactly such a reference-vs-reference noise floor.

The aggregator head is saved as a SIDECAR (ckpt_dir/agg_head.pt) after the
banked `save_final` runs, so dann_student.pt / adapter.pt / trunk/ stay
byte-compatible with every banked reader (the hard-max read path never
touches the sidecar).

Run (the draw LAUNCH decision is the coordinator's; the equivalence proof
runs through R18-H156_exec_equivalence.py):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
    uv run python experiments/grounding-semantic/R18-H156_split_exec.py --draw 1
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
MASK_FILL = -1e9   # the banked scatter-amax empty-fill


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


def split_train_step(arm, w156, model, agg, opt, sched, tok, claims, wsets,
                     batch, y_batch, g_batch, lam, audit_rng=False,
                     pass_a_chunk=PASS_A_CHUNK, pass_b_chunk=PASS_B_CHUNK):
    """One registered batch = one optimizer step, computed in window chunks.

    The cotangent form of the banked split step: pass A scores detached and
    backprops the aggregator loss on the leaves, pass B re-encodes with grad
    and applies the cotangents chunk by chunk. Mirrors
    w156.train_step (the monolithic reference) exactly by the chain-rule
    argument in the module docstring. Returns the per-step measurements the
    equivalence proof records.
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

    # pass A - no-grad encoding of every window, chunked; must consume no RNG
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

    # the aggregator forward on the DETACHED leaves, autograd enabled: head
    # grads and the exact cotangents cs = dL/ds, ce = dL/de
    e_leaf = cls_all.requires_grad_(True)
    s_leaf = lg.requires_grad_(True)
    s_agg = w156.agg_forward(agg, e_leaf, s_leaf, si, n_sets)
    t_loss = F.binary_cross_entropy_with_logits(s_agg, yy)  # mean over sets
    t_loss.backward()
    cs = s_leaf.grad.detach()
    ce = e_leaf.grad.detach()
    t_loss_val = float(t_loss.detach())

    # the step's ONE RNG draw: capture the banked domain-dropout mask whole
    dnn_hidden = model.domain_head[0].out_features
    dmask = F.dropout(torch.ones(P, dnn_hidden, device="cuda"),
                      p=model.domain_head[2].p, training=True)

    # pass B - grad-carrying chunks; per-chunk backward applies the cotangents
    # and accumulates the domain gradient over all windows, as banked
    rng1 = torch.cuda.get_rng_state() if audit_rng else None
    d_loss_sum, dom_correct = 0.0, 0
    for a in range(0, P, pass_b_chunk):
        b = min(a + pass_b_chunk, P)
        with torch.autocast("cuda", dtype=arm.TRAIN_ENCODE_DTYPE):
            cls_c = model.encode(encode_slice(a, b))
        cls_c = cls_c.float()  # heads and loss in fp32, as in the banked trainer
        lg_c = model.task_head(cls_c).squeeze(-1)  # adapter term is exactly +0.0
        si_c = si[a:b]
        t_term = (cs[a:b] * lg_c).sum() + (ce[a:b] * cls_c).sum()
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

    grad_norm = torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(agg.parameters()), arm.CLIP)
    opt.step()
    sched.step()
    opt.zero_grad()
    return {"task_loss": t_loss_val, "domain_loss": d_loss_sum,
            "dom_correct": dom_correct, "dom_total": P, "n_pairs": P,
            "grad_norm": float(grad_norm),
            "alpha": float(torch.sigmoid(agg.beta.detach()))}


# --- full-draw driver (the coordinator decides launches) -----------------------


def train(draw, max_steps=0):
    """The H156 wrapper's recipe orchestrated over split_train_step. Every
    data/model helper is the banked one (rebound to the H150 mix by the
    wrapper); the loop differences are exactly the cotangent window-chunked
    step and a resume payload carrying the banked core keys (model, opt,
    sched, step, perm fingerprint, torch RNG states) plus the aggregator
    head under "agg"."""
    arm = _mod("g1arm", "R16-H142_G1_arm.py")
    w156 = _mod("h156", "R18-H156_arm_run.py")
    arm = w156.rebind(arm, draw)
    cfg = w156.DRAWS[draw]
    seed = cfg["seed"]
    ckpt_dir = arm.ROOT / "models" / cfg["ckpt"]
    out = HERE / cfg["train_out"]

    print(f"=== R18-H156 COTANGENT SPLIT EXECUTOR draw {draw} (seed {seed})  "
          f"{time.strftime('%F %T')} ===", flush=True)
    print(f"H150 recipe verbatim (3-source mix, 14 DANN groups; NO EMA, NO "
          f"window dropout) + aggregator s_agg BCE, window-chunked "
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
    base.config.reference_compile = False  # mmBERT/ModernBERT compile path hangs here
    base = base.cuda()
    model = arm.DANNAdapterStudent(base, n_groups)
    model = model.cuda()
    agg = w156.AggHead(base.config.hidden_size)
    agg = agg.cuda()
    torch.manual_seed(seed)  # H126 ruling 8: re-issue after ALL construction
    fp, fp_numel = arm.init_fingerprint(model)
    if not arm.zero_init_ok(model):
        raise SystemExit("ZERO-INIT ABORT: the adapter output layer is not zero")
    if not (bool((agg.scorer[-1].weight == 0).all())
            and bool((agg.scorer[-1].bias == 0).all())
            and float(agg.beta.detach()) == w156.BETA_INIT):
        raise SystemExit("AGG-ZERO-INIT ABORT: scorer output not zero / beta off init")
    for n, p in model.named_parameters():
        if n.startswith(arm.ADAPTER_PREFIXES):
            p.requires_grad_(False)
    n_agg = sum(p.numel() for p in agg.parameters())
    agg_fp = w156.agg_init_fingerprint(agg)
    print(f"init fingerprint (trunk+task_head, {fp_numel} params): {fp}", flush=True)
    print(f"agg head: {n_agg:,} params (hidden {base.config.hidden_size}, "
          f"scorer ->{w156.AGG_HIDDEN}->1 zero-init, beta {w156.BETA_INIT})  "
          f"fingerprint {agg_fp}", flush=True)

    perm = np.random.default_rng(seed).permutation(n_rows)
    batches = arm.pack_batches(perm, sizes)
    n_steps = len(batches)
    print(f"perm fingerprint {arm.perm_fingerprint(perm)}  {n_steps} steps", flush=True)

    train_params = ([p for n, p in model.named_parameters() if p.requires_grad]
                    + list(agg.parameters()))
    opt = torch.optim.AdamW([{"params": train_params, "lr": arm.LR}], lr=arm.LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[arm.LR], total_steps=n_steps, pct_start=arm.WARMUP_FRAC,
        anneal_strategy="linear")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_path = ckpt_dir / "resume.pt"
    start_step = 0
    if resume_path.exists():
        st = torch.load(resume_path, map_location="cuda", weights_only=False)
        if "agg" not in st:
            raise SystemExit("RESUME ABORT: no aggregator state - not an H156 "
                             "resume point")
        model.load_state_dict(st["model"])
        agg.load_state_dict(st["agg"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_step = st["step"]
        if st["perm_fingerprint"] != arm.perm_fingerprint(perm):
            raise SystemExit("RESUME ABORT: the persisted permutation does not match")
        torch.set_rng_state(st["rng_cpu"].cpu())
        torch.cuda.set_rng_state(st["rng_cuda"].cpu())
        print(f"resumed from {resume_path} at step {start_step}/{n_steps}", flush=True)

    (ckpt_dir / "init_fingerprint.json").write_text(json.dumps(
        {"arm": f"h156_aggregator_twin_draw{draw}", "executor": "split-cotangent",
         "adapter_active": False, "seed": seed, "n_groups": n_groups,
         "group_counts": counts, "mix_rows": n_rows, "n_steps": n_steps,
         "scope": "trunk+task_head", "n_params": fp_numel, "blake2b_128": fp,
         "perm_fingerprint": arm.perm_fingerprint(perm),
         "window_census": cens,
         "aggregator": {
             "arch": f"Linear({base.config.hidden_size},{w156.AGG_HIDDEN})->ReLU->"
                     f"Linear({w156.AGG_HIDDEN},1) zero-init output; "
                     "alpha=sigmoid(beta)",
             "params": n_agg, "beta_init": w156.BETA_INIT,
             "blake2b_128": agg_fp, "sidecar": "agg_head.pt"},
         "split": {"pass_a_chunk": PASS_A_CHUNK, "pass_b_chunk": PASS_B_CHUNK},
         "regularizers": None,
         "recipe": "H150 (clean 685,670 + misbind 30,000 + unit_swap 5,540; "
                   "no EMA, no window dropout)"}, indent=2))

    def save_resume(step):
        tmp = resume_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "agg": agg.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step,
                    "perm_fingerprint": arm.perm_fingerprint(perm),
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state()}, tmp)
        tmp.replace(resume_path)  # atomic: a kill mid-write cannot corrupt it

    model.train()
    agg.train()
    t0 = time.time()
    dom_correct, dom_total = 0, 0
    for step in range(start_step, n_steps):
        if max_steps and step - start_step >= max_steps:
            el = time.time() - t0
            print(f"smoke stop at {max_steps} steps ({el:.0f}s, "
                  f"{el / max_steps:.2f} s/step over the loop)", flush=True)
            return
        batch = batches[step]
        p = step / max(n_steps - 1, 1)
        lam = arm.LAMBDA_MAX * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
        m = split_train_step(arm, w156, model, agg, opt, sched, tok, claims,
                             wsets, batch, y[batch], groups[batch], lam,
                             audit_rng=True)
        dom_correct += m["dom_correct"]
        dom_total += m["dom_total"]
        if not np.isfinite(m["task_loss"]):
            raise RuntimeError(f"diverged at step {step}")
        if step % 200 == 0:
            print(f"  step {step}/{n_steps} task {m['task_loss']:.4f} "
                  f"domain {m['domain_loss']:.4f} lam {lam:.4f} "
                  f"domain-acc {dom_correct / max(dom_total, 1):.3f} "
                  f"alpha {m['alpha']:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            dom_correct, dom_total = 0, 0
        if step and step % arm.RESUME_EVERY == 0:
            save_resume(step + 1)
            print(f"  resume point saved at step {step} (model + agg + RNG)",
                  flush=True)

    arm.save_final(model, base, tok, ckpt_dir, False)
    torch.save({"scorer": agg.scorer.state_dict(), "beta": float(agg.beta),
                "hidden": int(base.config.hidden_size),
                "agg_hidden": w156.AGG_HIDDEN, "seed": seed, "params": n_agg,
                "arch": f"Linear({int(base.config.hidden_size)},{w156.AGG_HIDDEN})"
                        f"->ReLU->Linear({w156.AGG_HIDDEN},1) zero-init output; "
                        "alpha=sigmoid(beta)",
                "executor": "split-cotangent"}, ckpt_dir / "agg_head.pt")
    resume_path.unlink(missing_ok=True)  # the run is done; a stale resume would rerun it
    print(f"checkpoint saved -> {ckpt_dir} (raw endpoint; aggregator sidecar "
          f"agg_head.pt, beta {float(agg.beta):+.4f})", flush=True)

    model.eval()
    res = arm.evaluate(model, tok)
    res.update({
        "run": "twin", "draw": draw, "arm": f"h156_aggregator_twin_draw{draw}",
        "executor": "split-cotangent",
        "split": {"pass_a_chunk": PASS_A_CHUNK, "pass_b_chunk": PASS_B_CHUNK,
                  "equivalence_proof": "R18-H156_exec_equivalence.json"},
        "aggregator": {
            "arch": f"Linear({int(base.config.hidden_size)},{w156.AGG_HIDDEN})->ReLU->"
                    f"Linear({w156.AGG_HIDDEN},1), output zero-init; "
                    "alpha=sigmoid(beta)",
            "params": n_agg, "beta_init": w156.BETA_INIT,
            "beta_final": round(float(agg.beta), 6),
            "alpha_final": round(float(torch.sigmoid(agg.beta)), 6),
            "init_fingerprint": agg_fp, "sidecar": str(ckpt_dir / "agg_head.pt")},
        "adapter_active": False, "seed": seed,
        "experiment": "R18-H156 learned window-aggregator twin - H150 recipe/mix "
                      "verbatim, task loss BCE(s_agg, y) in place of MIL max-BCE",
        "regularizers": None,
        "mix": "clean public mix (R10-H108.public_train) + R17-H146 misbind lane "
               "30,000 + R18-H150 unit_swap lane 5,540 (H150 recipe verbatim)",
        "evidence": f"UNTRUNCATED, windowed {arm.WIN}/{arm.STRIDE} (A1)",
        "objective": "aggregator s_agg BCE per row (alpha*max + (1-alpha)*softmax-"
                     "weighted sum over window logits) + per-pair domain CE",
        "precision": "bf16 autocast on the trunk encode, fp32 heads/aggregator/"
                     "loss, fp32 reads",
        "mix_rows": n_rows, "dann_groups": n_groups, "group_counts": counts,
        "window_census": cens, "n_steps": n_steps,
        "sets_per_batch": arm.SETS_PER_BATCH, "pairs_per_batch": arm.PAIRS_PER_BATCH,
        "init_fingerprint": fp, "init_fingerprint_scope": "trunk+task_head",
        "perm_fingerprint": arm.perm_fingerprint(perm),
        "bars": arm.BARS, "seed_sd": arm.SEED_SD, "control": arm.CONTROL,
        "bars_note": "the bars/control blocks are the banked G1 twin's; H156's "
                     "registered bars are adjudicated by the coordinator",
        "train_seconds": round(time.time() - t0, 1), "checkpoint": str(ckpt_dir),
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    })
    out.write_text(json.dumps(res, indent=2))
    gf = res["gold_full"]
    print(f"gold_full {gf['auc']:.4f} (n={gf['n']})  gold {res['gold']['auc']:.4f}  "
          f"ragtruth_en {res['ragtruth_en']['auc']:.4f}  "
          f"ragtruth_nonen {res['ragtruth_nonen']['auc']:.4f}", flush=True)
    print(f"results -> {out}", flush=True)
    print(f"=== H156 SPLIT DRAW {draw} TRAIN+INDOMAIN DONE ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", type=int, choices=(1, 2), required=True)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="smoke mode: stop after N steps, no checkpoint/eval")
    args = ap.parse_args()
    train(args.draw, max_steps=args.max_steps)


if __name__ == "__main__":
    main()

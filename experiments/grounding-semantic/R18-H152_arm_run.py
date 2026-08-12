"""R18-H152 REGULARIZED TWIN PAIR - EMA + window dropout on the twin recipe.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R18-H152 REGULARIZED TWIN PAIR" (2026-08-12 ~17:10): the training-side front
of the seed-variance attack, after the H142-T twin promotion failed on endpoint
VARIANCE (two-seed arena-mean spread 0.0243, per-subset swings to -0.076 - see
the H142-T post-verdict reflection).

Protocol = the twin recipe VERBATIM (banked trainer `R16-H142_G1_arm.py`, run
`twin`): the clean 685,670-row mix via the H108 public_train lineage, evidence
UNTRUNCATED, 1,500/750 windowed presentation, MIL max-over-windows BCE,
12-group DANN, full trunk at lr 1e-5 OneCycleLR 1 epoch, adapter FROZEN at its
zero init (TWIN INTEGRITY ABORT guard) - plus the two registered regularizers:

    EMA             decay 0.999 over trunk + task_head weights (the banked
                    init-fingerprint scope), updated after each optimizer step.
                    The SERVED checkpoint is the EMA copy: the EMA weights are
                    loaded into the model before the banked `save_final` runs,
                    so the in-domain suite, the arena read and the anti-gaming
                    stage all score the smoothed endpoint. The raw weights live
                    in resume.pt while the run is in flight.
    WINDOW DROPOUT  training only: inside each multi-window bag, every
                    NON-argmax window is dropped from the bag with p=0.1 -
                    masked to the -1e9 empty-fill the banked scatter-amax uses,
                    so it leaves that step's MIL max and its BCE gradient. The
                    argmax window is never dropped (ties: every tying window is
                    argmax), so at least one window always survives;
                    single-window rows are unaffected; the per-pair domain CE
                    still sees every window (the regularizer targets the
                    max-selection itself, per the registration). Serving and
                    every eval read all windows - dropout never touches them.

Why a reimplemented loop rather than monkey-patching: the banked trainer's
training step is an INLINE loop in `main()` - there is no step or save
function to subclass or patch, and the banked file must stay byte-identical
(it produced the adjudicated twin numbers). This wrapper therefore reuses
every banked module-level helper (build_mix, window_census, pack_batches,
encode_batch, DANNAdapterStudent, init_fingerprint, perm_fingerprint,
zero_init_ok, save_final, evaluate) and reimplements only the loop-level
orchestration, with the two regularizers factored as `apply_window_dropout` /
`ema_update` and the per-step body factored as `train_step` - the SAME
function the census dry-run drives on a CPU-tiny stub, so the engagement
proof covers the run's own code path.

resume.pt persists BOTH the raw model state and the EMA state (plus optimizer,
scheduler, step, permutation fingerprint, torch CPU+CUDA RNG states and the
drop counters), so a kill mid-run resumes exactly.

Draws (a fresh seed pair - the variance claim needs two draws by construction):

    draw 1  seed 3151  models/R18-H152-ema-draw1  R18-H152_arm_draw1_*.json
    draw 2  seed 3152  models/R18-H152-ema-draw2  R18-H152_arm_draw2_*.json

Stages:
    train      train + the in-domain suite (gold, gold_full, RAGTruth EN + 7)
    windowed   the PRIMARY blind windowed decomposed-min arena read (dispatched
               into the banked reader, byte-identical to the twin's reads)
    census     CPU-only dry run: mix + window census, per-draw permutation and
               init fingerprints and step counts, then the EMA / window-dropout
               engagement proof on a CPU-tiny stub (EMA-vs-raw delta > 0 after
               a few steps, dropped-window counter > 0, the argmax never
               dropped, single-window bags untouched, resume round-trip exact)

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
      uv run python experiments/grounding-semantic/R18-H152_arm_run.py \
          --stage train --draw 1
"""

import argparse
import contextlib
import importlib.util
import json
import os
import pathlib
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

HERE = pathlib.Path(__file__).parent

EMA_DECAY = 0.999
WINDOW_DROPOUT_P = 0.1
EMA_SCOPE = ("trunk.", "task_head.")  # the banked init-fingerprint scope

DRAWS = {
    1: {"seed": 3151, "ckpt": "R18-H152-ema-draw1",
        "train_out": "R18-H152_arm_draw1_result.json",
        "read_out": "R18-H152_arm_draw1_{mode}_result.json"},
    2: {"seed": 3152, "ckpt": "R18-H152-ema-draw2",
        "train_out": "R18-H152_arm_draw2_result.json",
        "read_out": "R18-H152_arm_draw2_{mode}_result.json"},
}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rebind(arm, draw):
    """Seed, checkpoint and result path - nothing else. The twin guard stands:
    this wrapper only ever dispatches the adapter-frozen run."""
    if arm.RUNS["twin"]["use_adapter"]:
        raise SystemExit("TWIN INTEGRITY ABORT: the dispatched run trains the adapter")
    cfg = DRAWS[draw]
    arm.SEED = cfg["seed"]  # save_final records the module-global SEED
    arm.RUNS["twin"]["ckpt"] = cfg["ckpt"]
    arm.RUNS["twin"]["out"] = cfg["train_out"]
    return arm


# --- the two registered regularizers -------------------------------------------


def apply_window_dropout(lg, si, n_sets, p):
    """Drop each NON-argmax window of every multi-window bag with probability p
    (training only; the caller decides when to call this at all).

    `lg` [P] per-window logits, `si` [P] window -> bag index, `n_sets` bags.
    Dropped windows are masked to the -1e9 empty-fill the banked scatter-amax
    uses, so they leave this step's MIL max and its BCE gradient. The argmax
    window of a bag is never dropped, so at least one window always survives;
    single-window bags have no eligible window by construction.

    Returns (masked logits, drop mask, n_dropped, n_eligible)."""
    det = lg.detach()
    set_max = torch.full((n_sets,), -1e9, device=lg.device).scatter_reduce(
        0, si, det, reduce="amax")
    counts = torch.bincount(si, minlength=n_sets)
    is_argmax = det == set_max[si]
    eligible = (counts[si] > 1) & ~is_argmax
    drop = (torch.rand(lg.shape[0], device=lg.device) < p) & eligible
    return lg.masked_fill(drop, -1e9), drop, int(drop.sum()), int(eligible.sum())


def ema_init(model):
    """The EMA copy, one tensor per tracked parameter, starting at the init."""
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if n.startswith(EMA_SCOPE)}


def ema_update(ema, model, decay=EMA_DECAY):
    """In-place EMA step, called after each optimizer step."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in ema:
                ema[n].mul_(decay).add_(p.detach(), alpha=1.0 - decay)


def ema_delta(ema, model):
    """Sum of squared EMA-vs-raw differences over the tracked scope - the
    engagement proof printed by the census and the training log."""
    d = 0.0
    for n, p in model.named_parameters():
        if n in ema:
            d += float((ema[n] - p.detach()).pow(2).sum())
    return d


def ema_load_(model, ema):
    """Copy the EMA weights into the live model: what save_final then writes is
    the smoothed endpoint, and the in-domain suite evaluates it."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in ema:
                p.copy_(ema[n])


# --- the training step (one code path for the GPU loop and the census proof) ----


def train_step(arm, model, opt, sched, enc, si, yy, gg, lam, ema, device):
    """One optimizer step of the banked twin loop with the two regularizers
    inline: window dropout on the MIL-max input, EMA update after opt.step.
    `device` is "cuda" in the run and "cpu" in the census proof; the bf16
    autocast on the trunk encode (the G0 setting the banked trainer carries)
    applies on cuda only."""
    n_sets = yy.shape[0]
    ctx = (torch.autocast("cuda", dtype=arm.TRAIN_ENCODE_DTYPE)
           if device == "cuda" else contextlib.nullcontext())
    with ctx:
        cls = model.encode(enc)
    cls = cls.float()  # heads and loss in fp32, as in the banked trainer
    lg = model.logits_from_cls(cls, si, n_sets)
    lg, drop_mask, n_dropped, n_eligible = apply_window_dropout(
        lg, si, n_sets, WINDOW_DROPOUT_P)
    agg = torch.full((n_sets,), -1e9, device=device).scatter_reduce(
        0, si, lg, reduce="amax")  # MIL: max over the row's kept windows
    t_loss = F.binary_cross_entropy_with_logits(agg, yy)
    domain_logit = model.domain_head(arm.H108.GradReverse.apply(cls, lam))
    d_loss = nn.CrossEntropyLoss()(domain_logit, gg[si])
    loss = t_loss + d_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), arm.CLIP)
    opt.step()
    sched.step()
    opt.zero_grad()
    ema_update(ema, model)
    return t_loss, d_loss, domain_logit, drop_mask, n_dropped, n_eligible


# --- train + in-domain suite -----------------------------------------------------


def train(arm, draw):
    """The banked twin main(), re-orchestrated for the two regularizers. Every
    data/model/read helper is the banked one; the differences are exactly:
    EMA state tracked and served, window dropout in the step, resume.pt
    carrying raw+EMA (and RNG) state."""
    cfg = DRAWS[draw]
    seed = cfg["seed"]
    ckpt_dir = arm.ROOT / "models" / cfg["ckpt"]
    out = HERE / cfg["train_out"]

    print(f"=== R18-H152 REGULARIZED TWIN draw {draw} (seed {seed})  "
          f"{time.strftime('%F %T')} ===", flush=True)
    print(f"twin protocol verbatim + EMA (decay {EMA_DECAY}, scope "
          f"{' + '.join(s.rstrip('.') for s in EMA_SCOPE)}, the served checkpoint "
          f"is the EMA copy) + window dropout (p {WINDOW_DROPOUT_P}, non-argmax "
          "windows of multi-window bags, training only)", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    claims, wsets, y, tags = arm.build_mix()
    n_rows = len(y)
    tag_to_idx = {t: i for i, t in enumerate(arm.EXPECTED_GROUPS)}
    groups = np.array([tag_to_idx[t] for t in tags])
    n_groups = len(arm.EXPECTED_GROUPS)
    counts = {t: int(sum(1 for x in tags if x == t)) for t in arm.EXPECTED_GROUPS}
    cens, sizes = arm.window_census(wsets, tags)

    print(f"train: {n_rows} rows over {n_groups} domains (chance {1.0 / n_groups:.3f})  "
          f"seed {seed}  mean target {y.mean():.3f}", flush=True)
    for t in arm.EXPECTED_GROUPS:
        g = cens["per_group"][t]
        print(f"  {t:<18} {g['rows']:>7}  mean win {g['mean_windows']:.2f}  "
              f"max {g['max_windows']:>2}", flush=True)
    print(f"window census (UNTRUNCATED evidence, {arm.WIN}/{arm.STRIDE}): "
          f"{cens['total_pairs']} pairs, mean set {cens['mean_windows_per_row']:.3f}, "
          f"median {cens['median_windows_per_row']}, p90 {cens['p90_windows_per_row']}, "
          f"max {cens['max_windows_per_row']}, "
          f"{cens['rows_with_multi_window_set']} rows ({cens['multi_window_share']:.1%}) "
          f"multi-window", flush=True)
    print(f"  set-size histogram (clipped at 10): {cens['histogram_clipped_at_10']}",
          flush=True)
    if cens["mean_windows_per_row"] < arm.MIN_MEAN_WINDOWS:
        raise SystemExit(
            f"WINDOW-CENSUS ABORT: mean set {cens['mean_windows_per_row']:.3f} < "
            f"{arm.MIN_MEAN_WINDOWS} - untruncated evidence did NOT produce "
            "multi-window sets")
    if cens["max_windows_per_row"] > arm.PAIRS_PER_BATCH:
        raise SystemExit(
            f"BATCH-CAP ABORT: a training set has {cens['max_windows_per_row']} "
            f"windows, over the {arm.PAIRS_PER_BATCH}-pair batch cap")
    print(flush=True)

    tok = AutoTokenizer.from_pretrained(arm.H108.STUDENT)
    base = AutoModel.from_pretrained(arm.H108.STUDENT)
    base.config.reference_compile = False  # mmBERT/ModernBERT compile path hangs here
    base = base.cuda()
    model = arm.DANNAdapterStudent(base, n_groups)
    model = model.cuda()
    torch.manual_seed(seed)  # H126 ruling 8: re-issue after construction
    fp, fp_numel = arm.init_fingerprint(model)
    n_adapter = sum(p.numel() for n, p in model.named_parameters()
                    if n.startswith(arm.ADAPTER_PREFIXES))
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    if not arm.zero_init_ok(model):
        raise SystemExit("ZERO-INIT ABORT: the adapter output layer is not zero")
    for n, p in model.named_parameters():
        if n.startswith(arm.ADAPTER_PREFIXES):
            p.requires_grad_(False)
    print(f"student {arm.H108.STUDENT} + DANN heads + adapter  {n_par:.1f}M params\n"
          f"adapter (h_norm+ctx_norm+MLP): {n_adapter:,} params (FROZEN at zero)  "
          f"zero-init verified: True\n"
          f"init fingerprint (trunk+task_head, {fp_numel} params): {fp}\n", flush=True)

    perm = np.random.default_rng(seed).permutation(n_rows)
    batches = arm.pack_batches(perm, sizes)
    n_steps = len(batches)
    print(f"perm fingerprint {arm.perm_fingerprint(perm)}  (flat shuffle, "
          f"np.random.default_rng({seed}).permutation)\n"
          f"{n_steps} steps  (<= {arm.SETS_PER_BATCH} sets / <= {arm.PAIRS_PER_BATCH} "
          f"pairs per batch, {cens['total_pairs'] / n_steps:.1f} pairs per step)\n",
          flush=True)

    old_params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": old_params, "lr": arm.LR}], lr=arm.LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[arm.LR], total_steps=n_steps, pct_start=arm.WARMUP_FRAC,
        anneal_strategy="linear")

    ema = ema_init(model)
    n_ema_updates = 0
    tot_dropped, tot_eligible = 0, 0

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_path = ckpt_dir / "resume.pt"
    start_step = 0
    if resume_path.exists():
        st = torch.load(resume_path, map_location="cuda", weights_only=False)
        if "ema" not in st:
            raise SystemExit("RESUME ABORT: resume.pt carries no EMA state - "
                             "this is not an H152 resume point; do not continue")
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
        print(f"resumed from {resume_path} at step {start_step}/{n_steps} "
              f"(raw + EMA + RNG state restored)\n", flush=True)

    (ckpt_dir / "init_fingerprint.json").write_text(json.dumps(
        {"arm": f"h152_regularized_twin_draw{draw}", "adapter_active": False,
         "seed": seed, "n_groups": n_groups, "group_counts": counts,
         "mix_rows": n_rows, "n_steps": n_steps, "scope": "trunk+task_head",
         "n_params": fp_numel, "blake2b_128": fp, "adapter_params": n_adapter,
         "window_census": cens,
         "perm_convention": f"np.random.default_rng({seed}).permutation(n_rows), flat",
         "perm_fingerprint": arm.perm_fingerprint(perm),
         "regularizers": {
             "ema": {"decay": EMA_DECAY, "scope": list(EMA_SCOPE),
                     "served_checkpoint": "EMA copy"},
             "window_dropout": {"p": WINDOW_DROPOUT_P, "training_only": True,
                                "scope": "non-argmax windows of multi-window bags"},
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
        tmp.replace(resume_path)  # atomic: a kill mid-write cannot corrupt it

    model.train()
    t0 = time.time()
    dom_correct, dom_total = 0, 0
    for step in range(start_step, n_steps):
        batch = batches[step]
        enc, si = arm.encode_batch(tok, claims, wsets, batch)
        yy = torch.as_tensor(y[batch], device="cuda")
        gg = torch.as_tensor(groups[batch], device="cuda")
        p = step / max(n_steps - 1, 1)
        lam = arm.LAMBDA_MAX * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

        t_loss, d_loss, domain_logit, _mask, n_drop, n_elig = train_step(
            arm, model, opt, sched, enc, si, yy, gg, lam, ema, "cuda")
        n_ema_updates += 1
        tot_dropped += n_drop
        tot_eligible += n_elig

        dom_correct += (domain_logit.argmax(-1) == gg[si]).sum().item()
        dom_total += len(si)
        if not torch.isfinite(t_loss):
            raise RuntimeError(f"diverged at step {step}")
        if step % 200 == 0:
            acc = dom_correct / max(dom_total, 1)
            dom_correct, dom_total = 0, 0
            print(f"  step {step}/{n_steps} task {t_loss.item():.4f} "
                  f"domain {d_loss.item():.4f} lam {lam:.4f} domain-acc {acc:.3f} "
                  f"ema-delta {ema_delta(ema, model):.3e} "
                  f"wdrop {tot_dropped}/{tot_eligible}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if step and step % arm.RESUME_EVERY == 0:
            save_resume(step + 1)
            print(f"  resume point saved at step {step} (raw + EMA)", flush=True)

    final_delta = ema_delta(ema, model)
    ema_load_(model, ema)  # the served checkpoint is the EMA copy
    print(f"\ntraining done - EMA weights loaded for serving "
          f"(final EMA-vs-raw delta {final_delta:.6e}, {n_ema_updates} updates; "
          f"window dropout dropped {tot_dropped}/{tot_eligible} eligible windows)",
          flush=True)
    arm.save_final(model, base, tok, ckpt_dir, False)
    resume_path.unlink(missing_ok=True)  # the run is done; a stale resume would rerun it
    print(f"checkpoint saved -> {ckpt_dir} (served weights = EMA copy)\n", flush=True)

    model.eval()
    res = arm.evaluate(model, tok)
    res.update({
        "run": "twin", "draw": draw, "arm": f"h152_regularized_twin_draw{draw}",
        "adapter_active": False,
        "experiment": "R18-H152 regularized twin pair - twin windowed-MIL "
                      "protocol + EMA (decay 0.999, served) + window dropout "
                      "(p 0.1, non-argmax, training only)",
        "seed": seed, "params_M": round(n_par, 1), "adapter_params": n_adapter,
        "adapter_hidden": arm.ADAPTER_HIDDEN, "lr": arm.LR, "lr_adapter": None,
        "lambda_max": arm.LAMBDA_MAX,
        "regularizers": {
            "ema": {"decay": EMA_DECAY, "scope": list(EMA_SCOPE),
                    "updates": n_ema_updates,
                    "final_ema_vs_raw_delta": final_delta,
                    "served_checkpoint": "EMA copy",
                    "raw_weights": "resume.pt while in flight; deleted at completion"},
            "window_dropout": {"p": WINDOW_DROPOUT_P, "training_only": True,
                               "scope": "non-argmax windows of multi-window bags",
                               "dropped": tot_dropped, "eligible": tot_eligible},
        },
        "mix": "incumbent clean public-only mix (R10-H108.public_train), no lane",
        "evidence": f"UNTRUNCATED, windowed {arm.WIN}/{arm.STRIDE} (A1)",
        "objective": "MIL max-over-window BCE per row (window dropout on the max "
                     "input, training only) + per-pair domain CE",
        "precision": "bf16 autocast on the trunk encode, fp32 heads/adapter/loss, "
                     "fp32 for both reads (read protocol byte-identical to the "
                     "banked windowed read)",
        "mix_rows": n_rows, "dann_groups": n_groups, "group_counts": counts,
        "window_census": cens, "n_steps": n_steps,
        "sets_per_batch": arm.SETS_PER_BATCH, "pairs_per_batch": arm.PAIRS_PER_BATCH,
        "init_fingerprint": fp, "init_fingerprint_scope": "trunk+task_head",
        "perm_fingerprint": arm.perm_fingerprint(perm),
        "bars": arm.BARS, "seed_sd": arm.SEED_SD, "control": arm.CONTROL,
        "bars_note": "the bars/control blocks are the banked G1 twin's; H152's "
                     "registered bars are adjudicated by the coordinator",
        "train_seconds": round(time.time() - t0, 1), "checkpoint": str(ckpt_dir),
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    })
    gf = res["gold_full"]
    print(f"gold_full {gf['auc']:.4f} (n={gf['n']})  gold {res['gold']['auc']:.4f}  "
          f"ragtruth_en {res['ragtruth_en']['auc']:.4f}  "
          f"ragtruth_nonen {res['ragtruth_nonen']['auc']:.4f}", flush=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"results -> {out}")
    print(f"=== H152 DRAW {draw} TRAIN+INDOMAIN DONE ===", flush=True)


# --- windowed arena read (banked reader dispatch) --------------------------------


def windowed(draw):
    cfg = DRAWS[draw]
    reads = _mod("g1reads", "R16-H142_G1_reads.py")
    rebind(reads.ARM, draw)  # the served checkpoint is the EMA copy on disk
    reads.out_path = lambda run, mode: HERE / cfg["read_out"].format(mode=mode)
    sys.argv = ["reads", "--run", "twin", "--mode", "windowed"]
    reads.main()


# --- census + engagement proof (CPU only) -----------------------------------------


def census(arm):
    print(f"=== R18-H152 CPU census (dry run, no GPU)  {time.strftime('%F %T')} ===",
          flush=True)
    seed1 = DRAWS[1]["seed"]
    torch.manual_seed(seed1)
    np.random.seed(seed1)
    rng_before = torch.get_rng_state()
    _claims, wsets, y, tags = arm.build_mix()
    rng_after = torch.get_rng_state()
    n_rows = len(y)
    cens, sizes = arm.window_census(wsets, tags)
    print(f"build_mix consumed no torch RNG: {torch.equal(rng_before, rng_after)}",
          flush=True)

    print(f"train: {n_rows} rows over {len(arm.EXPECTED_GROUPS)} domains  "
          f"mean target {y.mean():.3f}", flush=True)
    for t in arm.EXPECTED_GROUPS:
        g = cens["per_group"][t]
        print(f"  {t:<18} {g['rows']:>7}  mean win {g['mean_windows']:.2f}  "
              f"max {g['max_windows']:>2}", flush=True)
    print(f"window census (UNTRUNCATED evidence, {arm.WIN}/{arm.STRIDE}): "
          f"{cens['total_pairs']} pairs, mean set {cens['mean_windows_per_row']:.3f}, "
          f"median {cens['median_windows_per_row']}, p90 {cens['p90_windows_per_row']}, "
          f"max {cens['max_windows_per_row']}, "
          f"{cens['rows_with_multi_window_set']} rows ({cens['multi_window_share']:.1%}) "
          f"multi-window", flush=True)
    print(f"  set-size histogram (clipped at 10): {cens['histogram_clipped_at_10']}",
          flush=True)
    if cens["mean_windows_per_row"] < arm.MIN_MEAN_WINDOWS:
        raise SystemExit("WINDOW-CENSUS ABORT: untruncated evidence did not "
                         "produce multi-window sets")
    if cens["max_windows_per_row"] > arm.PAIRS_PER_BATCH:
        raise SystemExit("BATCH-CAP ABORT: a training set exceeds the pair cap")
    print(flush=True)

    tok = AutoTokenizer.from_pretrained(arm.H108.STUDENT)
    base = AutoModel.from_pretrained(arm.H108.STUDENT)
    base.config.reference_compile = False
    n_groups = len(arm.EXPECTED_GROUPS)
    known = {1142: "a8b2cf491a236bba", 2142: "eebe673dabeef46f",
             1150: "7d13f9ac86a79574"}  # banked: G1 twin, H142-T d2, H150 d1
    for draw in (1, 2):
        seed = DRAWS[draw]["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = arm.DANNAdapterStudent(base, n_groups)
        torch.manual_seed(seed)  # H126 ruling 8 re-issue, as in the run
        fp, fp_numel = arm.init_fingerprint(model)
        if not arm.zero_init_ok(model):
            raise SystemExit("ZERO-INIT ABORT: the adapter output layer is not zero")
        perm = np.random.default_rng(seed).permutation(n_rows)
        pfp = arm.perm_fingerprint(perm)
        n_steps = len(arm.pack_batches(perm, sizes))
        print(f"draw {draw}: seed {seed}  init fingerprint {fp} ({fp_numel} params)\n"
              f"  perm fingerprint {pfp}  {n_steps} steps  "
              f"(distinct from banked seeds {known}: "
              f"{pfp not in known.values() and seed not in known})", flush=True)
        del model
    del base, tok
    print(flush=True)

    ema_dropout_proof(arm)
    print("=== CENSUS ONLY - no training ===", flush=True)


def ema_dropout_proof(arm):
    """CPU-tiny engagement proof for the two regularizers. Part 1 drives the
    SAME train_step the GPU loop calls, over a stub trunk standing in for
    mmBERT; part 2 unit-checks the dropout invariants directly; part 3
    round-trips a resume payload."""
    print("=== EMA / window-dropout engagement proof (CPU-tiny stub) ===", flush=True)
    d, vocab, T = 32, 97, 8
    n_groups = len(arm.EXPECTED_GROUPS)
    torch.manual_seed(DRAWS[2]["seed"])

    class TinyTrunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, d)
            self.config = SimpleNamespace(hidden_size=d)

        def forward(self, input_ids=None, **kw):
            return SimpleNamespace(last_hidden_state=self.emb(input_ids))

    model = arm.DANNAdapterStudent(TinyTrunk(), n_groups)
    for n, prm in model.named_parameters():
        if n.startswith(arm.ADAPTER_PREFIXES):
            prm.requires_grad_(False)
    opt = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad]}], lr=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[1e-2], total_steps=6, pct_start=0.3, anneal_strategy="linear")
    ema = ema_init(model)

    bag_sizes = [1, 6, 4, 1, 9, 5, 3, 7]  # 8 bags; two single-window
    tot_drop = tot_elig = 0
    for step in range(6):
        si = torch.cat([torch.full((k,), r, dtype=torch.long)
                        for r, k in enumerate(bag_sizes)])
        enc = {"input_ids": torch.randint(0, vocab, (len(si), T))}
        yy = torch.randint(0, 2, (len(bag_sizes),)).float()
        gg = torch.randint(0, n_groups, (len(bag_sizes),))
        t_loss, d_loss, _dl, _mask, n_drop, n_elig = train_step(
            arm, model, opt, sched, enc, si, yy, gg, 0.02, ema, "cpu")
        tot_drop += n_drop
        tot_elig += n_elig
        print(f"  proof step {step}: task {t_loss.item():.4f} "
              f"domain {d_loss.item():.4f}  dropped {n_drop}/{n_elig} eligible  "
              f"ema-delta {ema_delta(ema, model):.6e}", flush=True)
    final_delta = ema_delta(ema, model)
    print(f"  PROOF ema: {tot_drop} windows dropped over 6 steps through "
          f"train_step; EMA-vs-raw delta {final_delta:.6e} > 0 after 6 steps",
          flush=True)

    # Part 2: the dropout invariants, checked directly over 200 trials.
    si = torch.cat([torch.full((k,), r, dtype=torch.long)
                    for r, k in enumerate(bag_sizes)])
    n_sets = len(bag_sizes)
    n_drop_total = 0
    for _trial in range(200):
        lg = torch.randn(len(si), requires_grad=True)
        det = lg.detach()
        set_max = torch.full((n_sets,), -1e9).scatter_reduce(0, si, det, reduce="amax")
        is_argmax = det == set_max[si]
        masked, drop, n_d, _ne = apply_window_dropout(lg, si, n_sets, WINDOW_DROPOUT_P)
        n_drop_total += n_d
        assert not bool((drop & is_argmax).any()), "argmax window dropped"
        kept = ~drop
        kept_per_set = torch.bincount(si[kept], minlength=n_sets)
        assert int(kept_per_set.min()) >= 1, "a bag lost every window"
        single = torch.as_tensor([k == 1 for k in bag_sizes])[si]
        assert not bool((drop & single).any()), "single-window bag touched"
        grad = torch.autograd.grad(masked.sum(), lg, retain_graph=False)[0]
        assert not bool((grad[drop] != 0).any()), "a dropped window kept gradient"
    lg0 = torch.randn(len(si), requires_grad=True)
    _m0, drop0, n0, _e0 = apply_window_dropout(lg0, si, n_sets, 0.0)
    assert n0 == 0 and not bool(drop0.any()), "p=0 still dropped windows"
    print(f"  PROOF window dropout: 200 trials, {n_drop_total} drops total, "
          f"0 argmax drops, every bag kept >= 1 window, single-window bags "
          f"untouched, dropped windows carry no gradient, p=0 drops nothing",
          flush=True)

    # Part 3: the resume payload round-trips raw + EMA exactly.
    tmp = HERE / "_h152_ema_proof_resume.pt"
    torch.save({"model": model.state_dict(), "ema": ema}, tmp)
    st = torch.load(tmp, map_location="cpu", weights_only=False)
    raw_ok = all(torch.equal(a, b) for (na, a), (nb, b) in
                 zip(model.state_dict().items(), st["model"].items(), strict=True)
                 if na == nb)
    ema_ok = all(torch.equal(ema[k], st["ema"][k]) for k in ema)
    tmp.unlink()
    assert raw_ok and ema_ok, "resume round-trip lost state"
    print(f"  PROOF resume: raw + EMA state save/load exactly ({len(ema)} tracked "
          f"tensors, scope {EMA_SCOPE})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("train", "windowed", "census"))
    ap.add_argument("--draw", type=int, choices=(1, 2), default=None)
    args = ap.parse_args()

    if args.stage == "census":
        census(rebind(_mod("g1arm", "R16-H142_G1_arm.py"), 1))
        return
    if args.draw is None:
        ap.error("--draw is required for the train and windowed stages")
    if args.stage == "windowed":
        windowed(args.draw)
        return
    train(rebind(_mod("g1arm", "R16-H142_G1_arm.py"), args.draw), args.draw)


if __name__ == "__main__":
    main()

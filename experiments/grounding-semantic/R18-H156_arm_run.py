"""R18-H156 LEARNED WINDOW-AGGREGATOR TWIN - the H150 recipe with a learned read.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R18-H156 LEARNED WINDOW-AGGREGATOR TWIN" (2026-08-13 ~21:52): the window max
is noisy on many-window items (H151a: argmax-flip rate r 0.9125 vs window
count), so a small aggregator over per-window LOGITS gated to start AT max is
trained end-to-end with the trunk -

    s_agg = alpha * max_i(s_i) + (1 - alpha) * sum_i w_i s_i
    w       = softmax over the item's windows of scorer(e_i)
    scorer  = Linear(hidden, 64) -> ReLU -> Linear(64, 1), the final Linear
              ZERO-INITIALISED (weight and bias), so the initial attention is
              uniform over each item's windows
    alpha   = sigmoid(beta), beta a single scalar parameter init +3.0
              (alpha ~= 0.9526 - the read starts ~95% at the hard max)

e_i is the window's [CLS] feature at trunk hidden size (mmBERT-base: 768), s_i
the per-window scalar logit. Exact head param count at hidden 768: 49,282
(768*64+64 + 64+1 + 1 for beta) - the registration's "~17k / 256->64->1"
phrasing priced the scorer at a 256-wide input; the binding mechanical spec is
Linear(trunk hidden, 64), and 768 is the trunk hidden. The exact count is
recorded in the checkpoint fingerprint, the result JSON and the equivalence
proof.

Protocol = the H150 recipe/mix VERBATIM (this wrapper imports the banked
H150 dispatcher's mix assembly unchanged): clean public 685,670 + H146 misbind
lane 30,000 + H150 unit_swap lane 5,540 = 721,210 rows, 14 DANN groups,
evidence UNTRUNCATED, 1,500/750 windowed presentation, full trunk at lr 1e-5
OneCycleLR 1 epoch, DANN lambda 0.02 Ganin ramp with domain CE over ALL
windows of the batch, adapter FROZEN at its zero init (TWIN INTEGRITY ABORT
guard). NO H152 regularizers - no EMA, no window dropout; those are H152-only.
The ONE difference from the H150 arm: the task loss is BCE(s_agg, y) in place
of the MIL max-over-windows BCE, mean over sets in the batch (identical
reduction semantics to the banked loss). The min-over-sentences axis of every
read is untouched.

The registered batch geometry (<= 48 sets / <= 96 pairs) does not fit the 32 GB
card monolithically (the H152 vram probe priced a single 40-window row at ~15
GB of activations), so training runs through the cotangent split executor
`R18-H156_split_exec.py` - this wrapper's train stage dispatches into it. The
monolithic reference step lives here as `train_step`: it is the equivalence
proof's reference arm and the census engagement proof's code path, and it is
never used to train a registered draw.

The aggregator head is saved as a SIDECAR (`ckpt_dir/agg_head.pt`) by the
executor; the banked `save_final` outputs (dann_student.pt, adapter.pt,
trunk/, tokenizer) stay byte-compatible, so the banked readers, the
anti-gaming stage and the probe bank all read the checkpoint through the hard
max WITHOUT touching the sidecar.

Draws (H150 recipe, fresh seed pair):

    draw 1  seed 1156  models/R18-H156-arm-draw1  R18-H156_arm_draw1_*.json
    draw 2  seed 2156  models/R18-H156-arm-draw2  R18-H156_arm_draw2_*.json

Stages:
    train         train + the in-domain suite (gold, gold_full, RAGTruth EN +
                  7), dispatched into the cotangent split executor
    windowed_agg  the PRIMARY blind windowed decomposed-min arena read, sets
                  aggregated by the learned aggregator (loads the sidecar)
    windowed      the registered SECONDARY: hard-max windowed arena read on the
                  same checkpoint, dispatched into the banked reader unchanged
    census        CPU-only dry run: mix + window census cross-check, per-draw
                  init/permutation fingerprints, then the aggregator/cotangent
                  engagement proof on a CPU-tiny stub

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
      uv run python experiments/grounding-semantic/R18-H156_arm_run.py \
          --stage train --draw 1
"""

import argparse
import contextlib
import importlib.util
import json
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

AGG_HIDDEN = 64
BETA_INIT = 3.0
MASK_FILL = -1e9  # the banked scatter-amax empty-fill

DRAWS = {
    1: {"seed": 1156, "ckpt": "R18-H156-arm-draw1",
        "train_out": "R18-H156_arm_draw1_result.json",
        "read_out": "R18-H156_arm_draw1_{mode}_result.json"},
    2: {"seed": 2156, "ckpt": "R18-H156-arm-draw2",
        "train_out": "R18-H156_arm_draw2_result.json",
        "read_out": "R18-H156_arm_draw2_{mode}_result.json"},
}

# Every banked permutation fingerprint the H156 draws must stay distinct from:
# 1142 (G1 twin), 2142 (H142-T d2), 1150 (H150 d1), 2150 (H150 d2),
# 3151/3152 (H152 d1/d2), 51551/51552 (H155 5155a/5155b).
BANKED_PERM_FPS = {"a8b2cf491a236bba", "eebe673dabeef46f", "7d13f9ac86a79574",
                   "8fb06248240a78e1", "5e3de18e48c57632", "70d71966b2f7ebcb",
                   "07fe223aeb6686fb", "76a057088e834027"}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rebind(arm, draw):
    """Seed, H150 mix, group map, checkpoint and result path - nothing else.
    The twin guard stands: this wrapper only ever dispatches the
    adapter-frozen run."""
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


# --- the aggregator head ---------------------------------------------------------


class AggHead(nn.Module):
    """The learned window aggregator: a per-window attention scorer over the
    [CLS] features, gated to start at the hard max.

    scorer  Linear(hidden, AGG_HIDDEN) -> ReLU -> Linear(AGG_HIDDEN, 1); the
            final Linear is ZERO-INITIALISED, so at init every window of an
            item scores 0 and the softmax attention is uniform
    beta    single scalar, init +3.0 -> alpha = sigmoid(beta) ~= 0.9526, so at
            init s_agg ~= 0.95 * max + 0.05 * uniform-mean of the window logits
    """

    def __init__(self, hidden, agg_hidden=AGG_HIDDEN):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden, agg_hidden), nn.ReLU(), nn.Linear(agg_hidden, 1))
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)
        self.beta = nn.Parameter(torch.tensor(float(BETA_INIT)))


def combine(agg, att, lg, si, n_sets):
    """s_agg per set from per-window attention logits `att` and task logits
    `lg`: alpha * max(lg) + (1 - alpha) * softmax(att)-weighted sum of lg,
    sets indexed by `si` [P] -> [n_sets]. The softmax max-shift is DETACHED -
    the shift cancels in the softmax Jacobian, so the gradient is exact. The
    max is the banked scatter-amax (equal-share tie split, asserted in the
    executor's kernel_self_check) - the same op the monolithic reference and
    the cotangent pass A both run, so tie routing is identical by construction.
    """
    am = torch.full((n_sets,), MASK_FILL, device=att.device, dtype=att.dtype)
    am = am.scatter_reduce(0, si, att.detach(), reduce="amax")
    ex = torch.exp(att - am[si])
    den = torch.zeros(n_sets, device=att.device, dtype=att.dtype)
    den = den.index_add(0, si, ex)
    w = ex / den[si]
    mx = torch.full((n_sets,), MASK_FILL, device=lg.device, dtype=lg.dtype)
    mx = mx.scatter_reduce(0, si, lg, reduce="amax")
    ws = torch.zeros(n_sets, device=lg.device, dtype=lg.dtype)
    ws = ws.index_add(0, si, w * lg)
    alpha = torch.sigmoid(agg.beta)
    return alpha * mx + (1.0 - alpha) * ws


def agg_forward(agg, cls, lg, si, n_sets):
    """The training-time aggregator forward: attention logits from the [CLS]
    features through the scorer, then the gated combination."""
    return combine(agg, agg.scorer(cls).squeeze(-1), lg, si, n_sets)


def agg_init_fingerprint(agg):
    """blake2b-128 over the aggregator head's parameter bytes in sorted name
    order - the head-side analogue of the banked trunk+task_head fingerprint."""
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    for name, p in sorted(agg.named_parameters(), key=lambda kv: kv[0]):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# --- the monolithic reference step (proof arm + census proof; never a draw) -------


def train_step(arm, model, agg, opt, sched, enc, si, yy, gg, lam, device):
    """One optimizer step of the H156 monolithic reference: the banked twin
    step with the MIL max-BCE replaced by BCE(s_agg, y) - mean over sets, the
    banked reduction. `device` is "cuda" in the proof and "cpu" in the census
    engagement proof; the bf16 autocast on the trunk encode (the G0 setting
    the banked trainer carries) applies on cuda only. Heads and loss fp32."""
    n_sets = yy.shape[0]
    ctx = (torch.autocast("cuda", dtype=arm.TRAIN_ENCODE_DTYPE)
           if device == "cuda" else contextlib.nullcontext())
    with ctx:
        cls = model.encode(enc)
    cls = cls.float()  # heads, aggregator and loss in fp32, as in the banked trainer
    lg = model.logits_from_cls(cls, si, n_sets)  # adapter term exactly +0.0
    s_agg = agg_forward(agg, cls, lg, si, n_sets)
    t_loss = F.binary_cross_entropy_with_logits(s_agg, yy)
    domain_logit = model.domain_head(arm.H108.GradReverse.apply(cls, lam))
    d_loss = nn.CrossEntropyLoss()(domain_logit, gg[si])
    loss = t_loss + d_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(agg.parameters()), arm.CLIP)
    opt.step()
    sched.step()
    opt.zero_grad()
    return t_loss, d_loss, domain_logit


# --- windowed arena reads ----------------------------------------------------------


def windowed(draw):
    """The registered SECONDARY: the hard-max windowed read, dispatched into
    the banked reader byte-identical (the sidecar is never touched)."""
    cfg = DRAWS[draw]
    reads = _mod("g1reads", "R16-H142_G1_reads.py")
    rebind(reads.ARM, draw)
    reads.out_path = lambda run, mode: HERE / cfg["read_out"].format(mode=mode)
    sys.argv = ["reads", "--run", "twin", "--mode", "windowed"]
    reads.main()


@torch.inference_mode()
def score_sets_agg(model, agg, tok, flat_s, flat_w, set_index, n_sets, batch=64,
                   tag=""):
    """The banked `score_sets` encode loop (fp32 CPU staging, 200k slices) with
    the per-set aggregation replaced by the learned aggregator: per-window
    logits through the same pair_logits path the max read uses, attention
    logits through the sidecar scorer, then `combine`."""
    n = len(flat_s)
    cls_all = torch.zeros(n, model.trunk.config.hidden_size, dtype=torch.float32)
    t0 = time.time()
    for i in range(0, n, batch):
        enc = tok(flat_s[i : i + batch], flat_w[i : i + batch], return_tensors="pt",
                  padding=True, truncation=True, max_length=512)
        enc = {k: v.cuda() for k, v in enc.items()}
        cls_all[i : i + batch] = model.encode(enc).float().cpu()
        if (i // batch) % 400 == 0 and i:
            print(f"    {tag} {i}/{n} ({i / max(time.time() - t0, 1e-9):.0f} pairs/s)",
                  flush=True)
    si = torch.as_tensor(set_index, dtype=torch.long).cuda()
    ctx = model.pool_ctx(cls_all.cuda(), si, n_sets)
    att_all = torch.zeros(n, dtype=torch.float32)
    lg_all = torch.zeros(n, dtype=torch.float32)
    step = 200_000
    for a in range(0, n, step):
        b = min(a + step, n)
        cls_b = cls_all[a:b].cuda()
        lg_all[a:b] = model.pair_logits(cls_b, ctx[si[a:b]]).cpu()
        att_all[a:b] = agg.scorer(cls_b).squeeze(-1).float().cpu()
    s_agg = combine(agg, att_all.cuda(), lg_all.cuda(), si, n_sets)
    return s_agg.float().cpu().numpy()


def load_agg_head(ckpt):
    """Rebuild the aggregator head from the sidecar. Aborts on any mismatch -
    a missing or shape-drifted sidecar must never read as a max twin."""
    ckpt = pathlib.Path(ckpt)
    side = torch.load(ckpt / "agg_head.pt", map_location="cpu", weights_only=False)
    agg = AggHead(int(side["hidden"]), int(side["agg_hidden"]))
    agg.scorer.load_state_dict(side["scorer"])
    with torch.no_grad():
        agg.beta.copy_(torch.tensor(float(side["beta"])))
    return agg.cuda().eval(), side


def windowed_agg(draw):
    """The PRIMARY read: the banked windowed decomposed-min arena protocol
    with the per-sentence set aggregation done by the learned aggregator
    (the min-over-sentences axis untouched). Mirrors the banked reader's
    main() subset loop line for line."""
    cfg = DRAWS[draw]
    reads = _mod("g1reads", "R16-H142_G1_reads.py")
    arm = rebind(reads.ARM, draw)
    ckpt = arm.ROOT / "models" / cfg["ckpt"]
    out = HERE / cfg["read_out"].format(mode="windowed_agg")

    print(f"=== R18-H156 AGGREGATOR windowed arena read draw {draw}  "
          f"{time.strftime('%F %T')} ===", flush=True)
    model, tok = arm.load_run(ckpt)  # banked loader; the twin guard fires inside
    agg, side = load_agg_head(ckpt)
    alpha = float(torch.sigmoid(agg.beta))
    print(f"sidecar {ckpt / 'agg_head.pt'}: beta {float(agg.beta):+.4f}  "
          f"alpha {alpha:.4f}  trained seed {side.get('seed')}", flush=True)
    subs = reads.ARENA.load_subsets()

    rows = {}
    for sub, (claims, chunks, y) in subs.items():
        flat_s, flat_w, set_index, owner = [], [], [], []
        for i, (c, ks) in enumerate(zip(claims, chunks, strict=True)):
            wlist = reads.evidence_sets("windowed", ks)
            for s in reads.H92.sentences(c):
                sid = len(owner)
                owner.append(i)
                for w in wlist:
                    flat_s.append(s)
                    flat_w.append(w)
                    set_index.append(sid)
        owner = np.array(owner)
        s_sent = score_sets_agg(model, agg, tok, flat_s, flat_w, set_index,
                                len(owner), tag=f"agg/windowed/{sub}")
        resp = np.array([s_sent[owner == i].min() for i in range(len(y))])
        auc, f1, _ = reads.M59.auc_and_f1(y, resp)
        rows[sub] = {
            "n": len(y), "n_sent": len(owner), "n_pairs": len(flat_s),
            "auc": round(auc, 4), "f1": round(f1, 4),
            "lettuce_auc": reads.H92.LETTUCE[sub],
            "banked_control_windowed": reads.CONTROL_WINDOWED[sub],
        }
        print(f"  {sub:14s} n={len(y):>4} agg windowed {auc:.4f}", flush=True)

    mean = float(np.mean([r["auc"] for r in rows.values()]))
    payload = {
        "read": f"R18-H156 learned-aggregator twin draw {draw}, windowed "
                "decomposed-min read, sets aggregated by s_agg (PRIMARY)",
        "run": "twin", "draw": draw, "read_mode": "windowed_agg",
        "checkpoint": str(ckpt), "per_subset": rows, "mean": round(mean, 5),
        "banked_control_windowed_mean": reads.CONTROL_WINDOWED_MEAN,
        "mean_delta_vs_banked_control": round(mean - reads.CONTROL_WINDOWED_MEAN, 5),
        "aggregator": {"sidecar": str(ckpt / "agg_head.pt"),
                       "beta": round(float(agg.beta), 6), "alpha": round(alpha, 6),
                       "arch": side.get("arch"),
                       "params": side.get("params")},
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  agg windowed mean {mean:.5f}", flush=True)
    print(f"  pubmedqa {rows['pubmedqa']['auc']:.4f}  "
          f"hotpotqa {rows['hotpotqa']['auc']:.4f}  "
          f"tatqa {rows['tatqa']['auc']:.4f}", flush=True)
    print(f"  results -> {out}", flush=True)


# --- census + engagement proof (CPU only) -------------------------------------------


def census():
    print(f"=== R18-H156 CPU census (dry run, no GPU)  {time.strftime('%F %T')} ===",
          flush=True)
    arm = rebind(_mod("g1arm", "R16-H142_G1_arm.py"), 1)
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
    if cens["mean_windows_per_row"] < arm.MIN_MEAN_WINDOWS:
        raise SystemExit("WINDOW-CENSUS ABORT: untruncated evidence did not "
                         "produce multi-window sets")
    print(flush=True)

    tok = AutoTokenizer.from_pretrained(arm.H108.STUDENT)
    base = AutoModel.from_pretrained(arm.H108.STUDENT)
    base.config.reference_compile = False
    n_groups = len(arm.EXPECTED_GROUPS)
    for draw in (1, 2):
        seed = DRAWS[draw]["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = arm.DANNAdapterStudent(base, n_groups)
        agg = AggHead(base.config.hidden_size)
        torch.manual_seed(seed)  # H126 ruling 8 re-issue, as in the run
        fp, fp_numel = arm.init_fingerprint(model)
        if not arm.zero_init_ok(model):
            raise SystemExit("ZERO-INIT ABORT: the adapter output layer is not zero")
        perm = np.random.default_rng(seed).permutation(n_rows)
        pfp = arm.perm_fingerprint(perm)
        n_steps = len(arm.pack_batches(perm, sizes))
        n_agg = sum(p.numel() for p in agg.parameters())
        print(f"draw {draw}: seed {seed}  init fingerprint {fp} ({fp_numel} params)\n"
              f"  agg head fingerprint {agg_init_fingerprint(agg)} "
              f"({n_agg:,} params at hidden {base.config.hidden_size})\n"
              f"  perm fingerprint {pfp}  {n_steps} steps  "
              f"(distinct from every banked perm: {pfp not in BANKED_PERM_FPS})",
              flush=True)
        del model, agg
    del base, tok
    print(flush=True)

    agg_engagement_proof(arm)
    print("=== CENSUS ONLY - no training ===", flush=True)


def agg_engagement_proof(arm):
    """CPU-tiny engagement proof for the aggregator head and the cotangent
    identity. Part 1 drives the SAME monolithic train_step the GPU proof's
    reference arm runs, over a stub trunk standing in for mmBERT; part 2
    proves the cotangent factorization - pass-A detached backward cotangents
    applied to a grad-carrying re-encode reproduce the monolithic gradient on
    every parameter; part 3 round-trips the resume payload."""
    print("=== aggregator / cotangent engagement proof (CPU-tiny stub) ===", flush=True)
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
    agg = AggHead(d)
    n_agg = sum(p.numel() for p in agg.parameters())
    if not (bool((agg.scorer[-1].weight == 0).all())
            and bool((agg.scorer[-1].bias == 0).all())
            and float(agg.beta.detach()) == BETA_INIT):
        raise SystemExit("AGG-ZERO-INIT ABORT: scorer output layer not zero or "
                         "beta not at its registered init")
    alpha0 = float(torch.sigmoid(agg.beta))
    print(f"  agg head at hidden {d}: {n_agg:,} params; scorer output zero-init "
          f"verified; alpha = sigmoid({BETA_INIT}) = {alpha0:.4f} (gated AT max)",
          flush=True)

    bag_sizes = [1, 6, 4, 1, 9, 5, 3, 7]  # 8 bags; two single-window
    si = torch.cat([torch.full((k,), r, dtype=torch.long)
                    for r, k in enumerate(bag_sizes)])
    enc = {"input_ids": torch.randint(0, vocab, (len(si), T))}
    yy = torch.randint(0, 2, (len(bag_sizes),)).float()
    gg = torch.randint(0, n_groups, (len(bag_sizes),))

    # Part 1: engagement through the run's own monolithic train_step.
    opt = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad]
          + list(agg.parameters())}], lr=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[1e-2], total_steps=6, pct_start=0.3, anneal_strategy="linear")
    before = {n: p.detach().clone() for n, p in agg.named_parameters()}
    for step in range(3):
        torch.manual_seed(1000 + step)  # deterministic domain-dropout masks
        t_loss, d_loss, _dl = train_step(
            arm, model, agg, opt, sched, enc, si, yy, gg, 0.02, "cpu")
        print(f"  proof step {step}: task {t_loss.item():.4f} "
              f"domain {d_loss.item():.4f}  beta {float(agg.beta):+.5f} "
              f"alpha {float(torch.sigmoid(agg.beta)):.4f}", flush=True)
    moved = {n for n, p in agg.named_parameters()
             if not torch.equal(before[n], p.detach())}
    if not {"beta", "scorer.2.weight", "scorer.2.bias"} <= moved:
        raise SystemExit("AGG-ENGAGEMENT ABORT: beta/scorer output did not move")
    print(f"  PROOF engagement: 3 steps through train_step moved {sorted(moved)} "
          f"(scorer.0 stays frozen at step 0 by the zero output layer, the "
          f"banked adapter's zero-init dynamic)", flush=True)

    # Part 2: the cotangent identity. Domain dropout switched to identity
    # (p=0) so the comparison is RNG-free; the RNG/mask lock-step is the GPU
    # kernel_self_check's and the proof's final-RNG-fingerprint's job.
    torch.manual_seed(777)
    m2 = arm.DANNAdapterStudent(TinyTrunk(), n_groups)
    for n, prm in m2.named_parameters():
        if n.startswith(arm.ADAPTER_PREFIXES):
            prm.requires_grad_(False)
    a2 = AggHead(d)
    m2.domain_head[2].p = 0.0
    n_sets = len(bag_sizes)
    cls_m = m2.encode(enc).float()
    lg_m = m2.logits_from_cls(cls_m, si, n_sets)
    t_m = F.binary_cross_entropy_with_logits(agg_forward(a2, cls_m, lg_m, si, n_sets), yy)
    d_m = F.cross_entropy(m2.domain_head(arm.H108.GradReverse.apply(cls_m, 0.02)), gg[si])
    (t_m + d_m).backward()
    g_mono = {n: p.grad.detach().clone()
              for mod in (m2, a2) for n, p in mod.named_parameters()
              if p.grad is not None}
    m2.zero_grad()
    a2.zero_grad()

    with torch.no_grad():
        cls_d = m2.encode(enc).float()
        ctx_d = m2.pool_ctx(cls_d, si, n_sets)
        lg_d = m2.pair_logits(cls_d, ctx_d[si]).detach()
    e_leaf = cls_d.requires_grad_(True)
    s_leaf = lg_d.requires_grad_(True)
    t_c = F.binary_cross_entropy_with_logits(
        agg_forward(a2, e_leaf, s_leaf, si, n_sets), yy)
    t_c.backward()  # head-param grads + cotangents
    cs, ce = s_leaf.grad.detach(), e_leaf.grad.detach()
    cls_g = m2.encode(enc).float()
    lg_g = m2.task_head(cls_g).squeeze(-1)  # adapter term exactly +0.0, as banked
    t_term = (cs * lg_g).sum() + (ce * cls_g).sum()
    d_term = F.cross_entropy(
        m2.domain_head(arm.H108.GradReverse.apply(cls_g, 0.02)), gg[si])
    (t_term + d_term).backward()
    worst = 0.0
    for mod in (m2, a2):
        for n, p in mod.named_parameters():
            g = p.grad
            if g is None:
                continue
            worst = max(worst, float((g - g_mono[n]).abs().max()))
    if not worst < 1e-5:
        raise SystemExit(f"COTANGENT ABORT: split-vs-monolithic grad max diff {worst:.3e}")
    print(f"  PROOF cotangent identity: split accumulation reproduces the "
          f"monolithic gradient on every parameter, max abs diff {worst:.3e} "
          f"(fp32 CPU stub, single chunk)", flush=True)

    # Part 3: the resume payload round-trips model + aggregator exactly.
    tmp = HERE / "_h156_agg_proof_resume.pt"
    torch.save({"model": m2.state_dict(), "agg": a2.state_dict()}, tmp)
    st = torch.load(tmp, map_location="cpu", weights_only=False)
    model_ok = all(torch.equal(a, b) for (na, a), (nb, b) in
                   zip(m2.state_dict().items(), st["model"].items(), strict=True)
                   if na == nb)
    agg_ok = all(torch.equal(a, b) for (na, a), (nb, b) in
                 zip(a2.state_dict().items(), st["agg"].items(), strict=True)
                 if na == nb)
    tmp.unlink()
    if not (model_ok and agg_ok):
        raise SystemExit("RESUME ABORT: model + aggregator state did not round-trip")
    print("  PROOF resume: model + aggregator state save/load exactly", flush=True)


# --- driver ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("train", "windowed", "windowed_agg", "census"))
    ap.add_argument("--draw", type=int, choices=(1, 2), default=None)
    args = ap.parse_args()

    if args.stage == "census":
        census()
        return
    if args.draw is None:
        ap.error("--draw is required for the train and read stages")
    if args.stage == "train":
        # the registered training path is the cotangent split executor - the
        # monolithic geometry does not fit the 32 GB card
        split_exec = _mod("h156split", "R18-H156_split_exec.py")
        split_exec.train(args.draw)
        return
    if args.stage == "windowed_agg":
        windowed_agg(args.draw)
        return
    windowed(args.draw)


if __name__ == "__main__":
    main()

"""R16-H142 G1 (amendment A1) - the adapter ABLATION PAIR, twin and arm.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R16-H142 G1 - registered (2026-08-11)", amended A1 after the pre-launch census
showed the registered single-variable form was unrunnable: the incumbent
truncates every training chunk to CFG.chunk_max_chars (1,500) and the serve-time
window is also 1,500 chars, so every training window set had size 1 and the
adapter's ensemble input was `ctx == cls` for all 685,670 rows - the ensemble
channel would have received no gradient at all.

Review constraint F3 allows either form: isolate the adapter, OR carry its own
ablation. A1 takes the ablation. Two runs, one presentation, one difference:

    TWIN  windowed presentation, MIL max-BCE, 12-group DANN, no adapter
    ARM   the same, plus the zero-init adapter side-head

Because the presentation is shared, the arm-minus-twin delta attributes to the
adapter and to nothing else. The twin runs FIRST: it is the control, and a
catastrophic twin is learned before the arm spends a card.

SHARED PRESENTATION (both runs, byte-identical):

    mix        `R10-H108_lane.public_train()` - 685,670 rows, 12 DANN groups,
               the incumbent row set exactly, but read with the evidence
               UNTRUNCATED (see `untruncated_evidence`) and then windowed at
               1,500 chars / stride 750, the serve-time protocol
    objective  MIL max-BCE - max over the row's window logits, BCE against the
               row label (G0's objective; forced by the window-set shape, and
               identical in both runs so it cancels in the delta)
               plus per-pair domain CE off the gradient-reversal layer,
               lambda 0.02 with the Ganin ramp (the incumbent's invariance term)
    schedule   MAX_LEN 512, <=48 sets and <=96 pairs per batch, LR 1e-5,
               OneCycleLR 10% linear warmup, grad clip 1.0, 1 epoch,
               bf16 autocast on the trunk encode (G0's setting) with the heads
               and both reads in fp32, flat shuffled row permutation greedily
               packed into batches
    seeding    H126 ruling 8 - manual_seed before construction, RE-ISSUED after,
               trunk+task_head init fingerprinted (blake2b-128), perm persisted

THE ONE DIFFERENCE (arm only):

    ctx        = mean over the row's window set of the per-window [CLS]
    logit_k    = task_head(cls_k) + adapter([LN(cls_k) ; LN(ctx)])
    adapter    Linear(2d -> 512) + GELU + Linear(512 -> 1), output layer
               ZERO-INITIALISED, so at step 0 the arm IS the twin

The adapter modules are CONSTRUCTED IN BOTH runs and constructed last, after
trunk, task_head and domain_head. Construction order plus the H126 re-issue of
the seed after construction means the two runs share their trunk+task_head init
draws AND their dropout stream exactly - the pair is init-paired, which the
banked clean control pair never was. The twin's adapter is frozen at its zero
init and excluded from the optimiser, so its output is identically 0.0 and the
read path needs no branch: the same adapter-aware reader scores both runs, and
for the twin the adapter term contributes exactly zero.

Resume: `perm` and the batch index are persisted with the weights, and the
restart replays the SAME permutation packed the same way, so every row is still
seen exactly once.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R16-H142_G1_arm.py --run twin
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import argparse
import contextlib
import hashlib
import importlib.util
import json
import pathlib
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent

MAX_LEN, LR = 512, 1e-5
LR_ADAPTER = 1e-3  # G0's fresh-parameter rate, for the only new module (arm only)
SETS_PER_BATCH, PAIRS_PER_BATCH = 48, 96  # 48 = the incumbent row batch
# G0's setting, applied to the trunk encode in BOTH runs of the pair (coordinator
# ruling on D4). The heads, the adapter and both reads stay fp32, so the read
# protocol remains byte-identical to the banked windowed read.
TRAIN_ENCODE_DTYPE = torch.bfloat16
WARMUP_FRAC, CLIP = 0.1, 1.0
LAMBDA_MAX = 0.02
RESUME_EVERY = 1000
SEED = 1142
ADAPTER_HIDDEN = 512
WIN, STRIDE = 1500, 750  # the serve-time window protocol (R8-H101)

EXPECTED_GROUPS = (
    "halueval", "psiloqa", "ragtruth_cn", "ragtruth_de", "ragtruth_en",
    "ragtruth_es", "ragtruth_fr", "ragtruth_hu", "ragtruth_it", "ragtruth_pl",
    "tabfact", "vitaminc",
)
EXPECTED_MIX_ROWS = 685_670
# A1's premise: untruncated evidence yields real multi-window sets. Measured at
# 1.507 mean before this script was written; the run aborts if it is not there.
MIN_MEAN_WINDOWS = 1.05

# Registered bars, priced off the banked clean control pair. Under A1 the twin
# is the comparator for the adapter question, so these are carried as ABSOLUTE
# holds; the pair deltas are computed by R16-H142_G1_pair.py.
BARS = {
    "pubmedqa_primary_min": 0.6495,
    "pubmedqa_kill_below": 0.6279,
    "hotpotqa_min": 0.6380,
    "tatqa_min": 0.6740,
    "arena_mean_min": 0.70311,
    "gold_full_min": 0.8414,
    "ragtruth_nonen_min": 0.82,
}
# Seed sd of the banked control pair, per the G1 registration.
SEED_SD = {"pubmedqa": 0.0216, "hotpotqa": 0.0144, "tatqa": 0.0290}
CONTROL = {"pubmedqa": 0.6063, "hotpotqa": 0.6668, "tatqa": 0.7320, "arena_mean": 0.70311}

RUNS = {
    "twin": {"ckpt": "R16-H142-G1-twin", "out": "R16-H142_G1_twin_result.json",
             "use_adapter": False},
    "arm": {"ckpt": "R16-H142-G1-arm", "out": "R16-H142_G1_arm_result.json",
            "use_adapter": True},
}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H108 = _mod("h108", "R10-H108_lane.py")
M59 = H108.M59
M60 = H108.M60


def windows(chunk):
    """Sliding 1,500-char windows at stride 750; final window flush to the end.
    Byte-identical to R8-H101 / R16-H142 G0."""
    n = len(chunk)
    if n <= WIN:
        return [chunk]
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return [chunk[s : s + WIN] for s in starts]


@contextlib.contextmanager
def untruncated_evidence():
    """`public_train` cuts every chunk to `M59.CFG.chunk_max_chars`. A1 needs the
    FULL evidence so it can be windowed, and needs the row set to stay the
    incumbent's - so the loader is reused unchanged with the cut lifted rather
    than reimplemented, which would risk a different row set. Every filter in
    that loader runs on untruncated text already, so the rows are identical.
    The cut is restored on the way out; the reads depend on it."""
    original = M59.CFG.chunk_max_chars
    M59.CFG.chunk_max_chars = 10**9
    try:
        yield
    finally:
        M59.CFG.chunk_max_chars = original


# --- model ---------------------------------------------------------------------


class DANNAdapterStudent(nn.Module):
    """The incumbent DANN student, plus the zero-init adapter side-head.

    trunk / task_head / domain_head are constructed in `DANNStudent`'s order and
    BEFORE the adapter, so both runs of the pair consume the same RNG draws for
    them and their init fingerprints must match.
    """

    def __init__(self, base, n_groups, hidden=H108.DANN_HIDDEN, adapter_hidden=ADAPTER_HIDDEN):
        super().__init__()
        self.trunk = base
        d = base.config.hidden_size
        self.pool = nn.Identity()
        self.task_head = nn.Linear(d, 1)
        self.domain_head = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, n_groups)
        )
        self.h_norm = nn.LayerNorm(d)
        self.ctx_norm = nn.LayerNorm(d)
        self.adapter = nn.Sequential(
            nn.Linear(2 * d, adapter_hidden), nn.GELU(), nn.Linear(adapter_hidden, 1)
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def encode(self, enc):
        return self.trunk(**enc).last_hidden_state[:, 0]

    def pool_ctx(self, cls, set_index, n_sets):
        """Mean-pool the window-ensemble context: cls [P,d] -> ctx [n_sets,d]."""
        summed = torch.zeros(n_sets, cls.shape[1], dtype=cls.dtype, device=cls.device)
        summed.index_add_(0, set_index, cls)
        counts = torch.zeros(n_sets, dtype=cls.dtype, device=cls.device)
        counts.index_add_(0, set_index, torch.ones_like(set_index, dtype=cls.dtype))
        return summed / counts.clamp(min=1).unsqueeze(-1)

    def pair_logits(self, cls, ctx_rows):
        """The adapter term is identically zero while its output layer is zero,
        which is the twin's state for the whole of its life - so this one path
        serves both runs and the twin reads as a plain score head."""
        z = torch.cat([self.h_norm(cls), self.ctx_norm(ctx_rows)], dim=-1)
        return self.task_head(cls).squeeze(-1) + self.adapter(z).squeeze(-1)

    def logits_from_cls(self, cls, set_index, n_sets):
        ctx = self.pool_ctx(cls, set_index, n_sets)
        return self.pair_logits(cls, ctx[set_index])


ADAPTER_PREFIXES = ("h_norm.", "ctx_norm.", "adapter.")


def zero_init_ok(model):
    w, b = model.adapter[-1].weight, model.adapter[-1].bias
    return bool(torch.all(w == 0).item() and torch.all(b == 0).item())


def init_fingerprint(model):
    """blake2b-128 over trunk + task_head parameter bytes in sorted name order -
    the campaign's fingerprint scope. Must be IDENTICAL across the pair."""
    h = hashlib.blake2b(digest_size=16)
    n_par = 0
    named = [(n, p) for n, p in model.named_parameters()
             if n.startswith(("trunk.", "task_head."))]
    for name, p in sorted(named, key=lambda kv: kv[0]):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
        n_par += p.numel()
    return h.hexdigest(), n_par


def perm_fingerprint(perm):
    return hashlib.blake2b(
        np.asarray(perm, dtype=np.int64).tobytes(), digest_size=8
    ).hexdigest()


# --- data ------------------------------------------------------------------------


def build_mix():
    """The incumbent 685,670-row clean mix with UNTRUNCATED evidence."""
    with untruncated_evidence():
        claims, chunks, y, tags = H108.public_train()
    names = tuple(sorted(set(tags)))
    if names != EXPECTED_GROUPS:
        raise SystemExit(
            f"GROUP-MAP ABORT: mix groups {names} != registered {EXPECTED_GROUPS}"
        )
    if len(y) != EXPECTED_MIX_ROWS:
        raise SystemExit(
            f"CENSUS ABORT: mix {len(y)} rows (want {EXPECTED_MIX_ROWS}) - "
            "this is NOT the incumbent clean mix; do not train"
        )
    wsets = [windows(k) for k in chunks]
    return claims, wsets, y.astype("float32"), tags


def window_census(wsets, tags):
    sizes = np.array([len(w) for w in wsets], dtype=np.int32)
    hist = {int(k): int(v) for k, v in zip(*np.unique(np.clip(sizes, 1, 10),
                                                     return_counts=True), strict=True)}
    per_group = {}
    for g in EXPECTED_GROUPS:
        m = np.fromiter((t == g for t in tags), dtype=bool, count=len(tags))
        per_group[g] = {"rows": int(m.sum()), "mean_windows": round(float(sizes[m].mean()), 3),
                        "max_windows": int(sizes[m].max())}
    return {
        "mean_windows_per_row": round(float(sizes.mean()), 4),
        "median_windows_per_row": int(np.median(sizes)),
        "p90_windows_per_row": int(np.percentile(sizes, 90)),
        "max_windows_per_row": int(sizes.max()),
        "rows_with_multi_window_set": int((sizes > 1).sum()),
        "multi_window_share": round(float((sizes > 1).mean()), 4),
        "total_pairs": int(sizes.sum()),
        "histogram_clipped_at_10": hist,
        "per_group": per_group,
    }, sizes


def pack_batches(perm, sizes):
    """Greedy packing in permutation order under the set and pair caps. Purely a
    function of (perm, sizes), so a resumed run rebuilds the same batches."""
    out, cur, tot = [], [], 0
    for i in perm:
        k = int(sizes[i])
        if cur and (len(cur) + 1 > SETS_PER_BATCH or tot + k > PAIRS_PER_BATCH):
            out.append(cur)
            cur, tot = [], 0
        cur.append(int(i))
        tot += k
    if cur:
        out.append(cur)
    return out


def encode_batch(tok, claims, wsets, batch):
    flat_c, flat_w, set_index = [], [], []
    for r, i in enumerate(batch):
        for w in wsets[i]:
            flat_c.append(claims[i])
            flat_w.append(w)
            set_index.append(r)
    enc = tok(flat_c, flat_w, return_tensors="pt", padding=True,
              truncation=True, max_length=MAX_LEN)
    enc = {k: v.cuda() for k, v in enc.items()}
    return enc, torch.tensor(set_index, dtype=torch.long, device="cuda")


# --- adapter-aware reads -----------------------------------------------------------


@torch.inference_mode()
def score_sets(model, tok, flat_s, flat_w, set_index, n_sets, batch=64, tag=""):
    """Encode every (sentence, window) pair, pool the window-ensemble context per
    set, then max over the set - the G0 read path, fp32."""
    n = len(flat_s)
    cls_all = torch.zeros(n, model.trunk.config.hidden_size, dtype=torch.float32)
    t0 = time.time()
    for i in range(0, n, batch):
        enc = tok(flat_s[i : i + batch], flat_w[i : i + batch], return_tensors="pt",
                  padding=True, truncation=True, max_length=MAX_LEN)
        enc = {k: v.cuda() for k, v in enc.items()}
        cls_all[i : i + batch] = model.encode(enc).float().cpu()
        if (i // batch) % 400 == 0 and i:
            print(f"    {tag} {i}/{n} ({i / max(time.time() - t0, 1e-9):.0f} pairs/s)",
                  flush=True)
    si = torch.as_tensor(set_index, dtype=torch.long).cuda()
    ctx = model.pool_ctx(cls_all.cuda(), si, n_sets)
    agg = torch.full((n_sets,), -1e9, device="cuda")
    step = 200_000
    for a in range(0, n, step):
        b = min(a + step, n)
        lg = model.pair_logits(cls_all[a:b].cuda(), ctx[si[a:b]])
        agg = agg.scatter_reduce(0, si[a:b], lg, reduce="amax")
    return agg.float().cpu().numpy()


def score_claims(model, tok, claims, chunk_lists, tag=""):
    """Max over the claim's chunk list, chunks cut to the serving unit - the
    in-domain protocol every campaign arm is read under, unchanged."""
    flat_s, flat_w, si = [], [], []
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        for k in ks:
            flat_s.append(c)
            flat_w.append(k[: M59.CFG.chunk_max_chars])
            si.append(i)
    return score_sets(model, tok, flat_s, flat_w, si, len(claims), tag=tag)


def evaluate(model, tok):
    """The in-domain suite the campaign runs on every arm - gold, gold_full,
    RAGTruth EN and the seven translations."""
    res = {}
    claims, chunk_lists, y, _, _ = _mod("sub", "R8_score_substrate.py").our_gold()
    s = score_claims(model, tok, claims, chunk_lists, tag="gold")
    auc, f1, _ = M59.auc_and_f1(y, s)
    res["gold"] = {"auc": round(auc, 4), "f1": round(f1, 4), "n": len(y)}

    cl_f, ck_f, y_f = H108.gold_full()
    s = score_claims(model, tok, cl_f, ck_f, tag="gold_full")
    auc, f1, _ = M59.auc_and_f1(y_f, s)
    res["gold_full"] = {"auc": round(auc, 4), "f1": round(f1, 4), "n": len(y_f)}

    cl, ctx, y = M60.load_english()
    s = score_claims(model, tok, cl,
                     [M59.top_chunks(c, M59.CFG.semantic_top_k) for c in ctx],
                     tag="ragtruth_en")
    auc, f1, _ = M59.auc_and_f1(y, s)
    res["ragtruth_en"] = {"auc": round(auc, 4), "f1": round(f1, 4), "n": len(y)}

    per_lang = {}
    for lg in ("de", "fr", "es", "it", "pl", "hu", "cn"):
        cl, ctx, y = M60.load_translated(lg)
        s = score_claims(model, tok, cl,
                         [M59.top_chunks(c, M59.CFG.semantic_top_k) for c in ctx],
                         tag=f"ragtruth_{lg}")
        auc, _, _ = M59.auc_and_f1(y, s)
        per_lang[lg] = round(auc, 4)
    res["ragtruth_nonen"] = {
        "auc": round(float(np.mean(list(per_lang.values()))), 4), "per_lang": per_lang,
    }
    return res


# --- checkpoint --------------------------------------------------------------------


def save_final(model, base, tok, ckpt_dir, use_adapter):
    if not use_adapter and not zero_init_ok(model):
        raise SystemExit("TWIN INTEGRITY ABORT: the twin's adapter output layer moved "
                         "off zero - it was trained, and the pair is no longer an ablation")
    torch.save({"trunk": base.state_dict(), "task_head": model.task_head.state_dict(),
                "domain_head": model.domain_head.state_dict(), "config": base.config},
               ckpt_dir / "dann_student.pt")
    torch.save({"h_norm": model.h_norm.state_dict(), "ctx_norm": model.ctx_norm.state_dict(),
                "adapter": model.adapter.state_dict(), "hidden": ADAPTER_HIDDEN,
                "seed": SEED, "adapter_active": use_adapter}, ckpt_dir / "adapter.pt")
    tok.save_pretrained(ckpt_dir)
    base.save_pretrained(ckpt_dir / "trunk")


def load_run(ckpt_dir):
    """Rebuild a trained run for a read stage. The twin's adapter is its frozen
    zero init, so the same reader serves both."""
    ckpt_dir = pathlib.Path(ckpt_dir)
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir))
    base = AutoModel.from_pretrained(str(ckpt_dir / "trunk"), attn_implementation="sdpa")
    base.config.reference_compile = False
    st = torch.load(ckpt_dir / "dann_student.pt", map_location="cpu", weights_only=False)
    ad = torch.load(ckpt_dir / "adapter.pt", map_location="cpu", weights_only=False)
    n_groups = st["domain_head"]["3.weight"].shape[0]
    model = DANNAdapterStudent(base, n_groups)
    model.task_head.load_state_dict(st["task_head"])
    model.domain_head.load_state_dict(st["domain_head"])
    model.h_norm.load_state_dict(ad["h_norm"])
    model.ctx_norm.load_state_dict(ad["ctx_norm"])
    model.adapter.load_state_dict(ad["adapter"])
    if not ad.get("adapter_active", True) and not zero_init_ok(model):
        raise SystemExit(f"{ckpt_dir} is marked twin but its adapter is not zero")
    return model.cuda().eval(), tok


# --- driver ------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=tuple(RUNS))
    ap.add_argument("--max-steps", type=int, default=0,
                    help="smoke mode: stop after N steps, no checkpoint/eval, ckpt dir -smoke")
    ap.add_argument("--census-only", action="store_true",
                    help="build the mix, print the census and the model report, then exit (CPU)")
    args = ap.parse_args()

    cfg = RUNS[args.run]
    use_adapter = cfg["use_adapter"]
    ckpt_dir = ROOT / "models" / (f"R16-H142-G1-{args.run}-smoke" if args.max_steps
                                  else cfg["ckpt"])
    out = HERE / cfg["out"]

    print(f"=== R16-H142 G1 {args.run.upper()} (A1 ablation pair)  "
          f"{time.strftime('%F %T')} ===", flush=True)
    print(f"adapter: {'ACTIVE (trained)' if use_adapter else 'FROZEN AT ZERO INIT (control)'}\n"
          f"precision: {TRAIN_ENCODE_DTYPE} autocast on the trunk encode, "
          "fp32 heads and fp32 reads", flush=True)
    if not args.census_only:
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    claims, wsets, y, tags = build_mix()
    n_rows = len(y)
    tag_to_idx = {t: i for i, t in enumerate(EXPECTED_GROUPS)}
    groups = np.array([tag_to_idx[t] for t in tags])
    n_groups = len(EXPECTED_GROUPS)
    counts = {t: int(sum(1 for x in tags if x == t)) for t in EXPECTED_GROUPS}
    cens, sizes = window_census(wsets, tags)

    print(f"train: {n_rows} rows over {n_groups} domains (chance {1.0 / n_groups:.3f})  "
          f"seed {SEED}  mean target {y.mean():.3f}", flush=True)
    for t in EXPECTED_GROUPS:
        g = cens["per_group"][t]
        print(f"  {t:<18} {g['rows']:>7}  mean win {g['mean_windows']:.2f}  "
              f"max {g['max_windows']:>2}", flush=True)
    print(f"window census (UNTRUNCATED evidence, {WIN}/{STRIDE}): "
          f"{cens['total_pairs']} pairs, mean set {cens['mean_windows_per_row']:.3f}, "
          f"median {cens['median_windows_per_row']}, p90 {cens['p90_windows_per_row']}, "
          f"max {cens['max_windows_per_row']}, "
          f"{cens['rows_with_multi_window_set']} rows ({cens['multi_window_share']:.1%}) "
          f"multi-window", flush=True)
    print(f"  set-size histogram (clipped at 10): {cens['histogram_clipped_at_10']}", flush=True)
    if cens["mean_windows_per_row"] < MIN_MEAN_WINDOWS:
        raise SystemExit(
            f"WINDOW-CENSUS ABORT: mean set {cens['mean_windows_per_row']:.3f} < "
            f"{MIN_MEAN_WINDOWS} - untruncated evidence did NOT produce multi-window "
            "sets, so A1's premise fails and the adapter would train on ctx == cls"
        )
    if cens["max_windows_per_row"] > PAIRS_PER_BATCH:
        raise SystemExit(
            f"BATCH-CAP ABORT: a training set has {cens['max_windows_per_row']} windows, "
            f"over the {PAIRS_PER_BATCH}-pair batch cap"
        )
    print(flush=True)

    tok = AutoTokenizer.from_pretrained(H108.STUDENT)
    base = AutoModel.from_pretrained(H108.STUDENT)
    base.config.reference_compile = False  # mmBERT/ModernBERT compile path hangs here
    if not args.census_only:
        base = base.cuda()
    model = DANNAdapterStudent(base, n_groups)
    if not args.census_only:
        model = model.cuda()
    torch.manual_seed(SEED)  # H126 ruling 8: re-issue after construction, before any forward
    fp, fp_numel = init_fingerprint(model)
    n_adapter = sum(p.numel() for n, p in model.named_parameters()
                    if n.startswith(ADAPTER_PREFIXES))
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    if not zero_init_ok(model):
        raise SystemExit("ZERO-INIT ABORT: the adapter output layer is not zero")
    if not use_adapter:
        for n, p in model.named_parameters():
            if n.startswith(ADAPTER_PREFIXES):
                p.requires_grad_(False)
    adapter_state = f"trained at lr {LR_ADAPTER:.0e}" if use_adapter else "FROZEN at zero"
    print(f"student {H108.STUDENT} + DANN heads + adapter  {n_par:.1f}M params\n"
          f"adapter (h_norm+ctx_norm+MLP): {n_adapter:,} params ({adapter_state})  "
          f"zero-init verified: True\n"
          f"init fingerprint (trunk+task_head, {fp_numel} params): {fp}\n", flush=True)

    perm = np.random.default_rng(SEED).permutation(n_rows)
    batches = pack_batches(perm, sizes)
    n_steps = len(batches)
    print(f"perm fingerprint {perm_fingerprint(perm)}  (flat shuffle, "
          f"np.random.default_rng({SEED}).permutation)\n"
          f"{n_steps} steps  (<= {SETS_PER_BATCH} sets / <= {PAIRS_PER_BATCH} pairs per batch, "
          f"{cens['total_pairs'] / n_steps:.1f} pairs per step)\n", flush=True)

    if args.census_only:
        print("=== CENSUS ONLY - no training ===", flush=True)
        return

    train_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    new_params = [p for n, p in train_params if n.startswith(ADAPTER_PREFIXES)]
    old_params = [p for n, p in train_params if not n.startswith(ADAPTER_PREFIXES)]
    param_groups = [{"params": old_params, "lr": LR}]
    max_lrs = [LR]
    if new_params:
        param_groups.append({"params": new_params, "lr": LR_ADAPTER})
        max_lrs.append(LR_ADAPTER)
    opt = torch.optim.AdamW(param_groups, lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lrs, total_steps=n_steps, pct_start=WARMUP_FRAC,
        anneal_strategy="linear")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_path = ckpt_dir / "resume.pt"
    start_step = 0
    if resume_path.exists():
        st = torch.load(resume_path, map_location="cuda", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_step = st["step"]
        if st["perm_fingerprint"] != perm_fingerprint(perm):
            raise SystemExit("RESUME ABORT: the persisted permutation does not match")
        print(f"resumed from {resume_path} at step {start_step}/{n_steps}\n", flush=True)

    (ckpt_dir / "init_fingerprint.json").write_text(json.dumps(
        {"arm": f"h142_g1_{args.run}", "adapter_active": use_adapter, "seed": SEED,
         "n_groups": n_groups, "group_counts": counts, "mix_rows": n_rows,
         "n_steps": n_steps, "scope": "trunk+task_head", "n_params": fp_numel,
         "blake2b_128": fp, "adapter_params": n_adapter,
         "window_census": cens,
         "perm_convention": f"np.random.default_rng({SEED}).permutation(n_rows), flat",
         "perm_fingerprint": perm_fingerprint(perm)}, indent=2))

    domain_lossf = nn.CrossEntropyLoss()

    def save_resume(step):
        tmp = resume_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step,
                    "perm_fingerprint": perm_fingerprint(perm)}, tmp)
        tmp.replace(resume_path)  # atomic: a kill mid-write cannot corrupt the resume

    model.train()
    t0 = time.time()
    dom_correct, dom_total = 0, 0
    for step in range(start_step, n_steps):
        if args.max_steps and step - start_step >= args.max_steps:
            print(f"smoke stop at {args.max_steps} steps", flush=True)
            return
        batch = batches[step]
        enc, si = encode_batch(tok, claims, wsets, batch)
        n_sets = len(batch)
        yy = torch.as_tensor(y[batch], device="cuda")
        gg = torch.as_tensor(groups[batch], device="cuda")
        p = step / max(n_steps - 1, 1)
        lam = LAMBDA_MAX * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

        with torch.autocast("cuda", dtype=TRAIN_ENCODE_DTYPE):
            cls = model.encode(enc)
        cls = cls.float()  # heads, adapter and loss in fp32, as in G0
        lg = model.logits_from_cls(cls, si, n_sets)
        agg = torch.full((n_sets,), -1e9, device="cuda").scatter_reduce(
            0, si, lg, reduce="amax")  # MIL: max over the row's windows
        t_loss = F.binary_cross_entropy_with_logits(agg, yy)
        domain_logit = model.domain_head(H108.GradReverse.apply(cls, lam))
        d_loss = domain_lossf(domain_logit, gg[si])
        loss = t_loss + d_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        sched.step()
        opt.zero_grad()

        dom_correct += (domain_logit.argmax(-1) == gg[si]).sum().item()
        dom_total += len(si)
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

    save_final(model, base, tok, ckpt_dir, use_adapter)
    resume_path.unlink(missing_ok=True)  # the run is done; a stale resume would rerun it
    print(f"\ncheckpoint saved -> {ckpt_dir}\n", flush=True)

    model.eval()
    res = evaluate(model, tok)
    res.update({
        "run": args.run, "adapter_active": use_adapter,
        "experiment": "R16-H142 G1 amendment A1 - adapter ablation pair "
                      "(twin = no adapter, arm = adapter; presentation shared)",
        "seed": SEED, "params_M": round(n_par, 1), "adapter_params": n_adapter,
        "adapter_hidden": ADAPTER_HIDDEN, "lr": LR,
        "lr_adapter": LR_ADAPTER if use_adapter else None,
        "lambda_max": LAMBDA_MAX,
        "mix": "incumbent clean public-only mix (R10-H108.public_train), no lane",
        "evidence": f"UNTRUNCATED, windowed {WIN}/{STRIDE} (A1)",
        "objective": "MIL max-over-window BCE per row + per-pair domain CE",
        "precision": "bf16 autocast on the trunk encode, fp32 heads/adapter/loss, "
                     "fp32 for both reads (read protocol byte-identical to the "
                     "banked windowed read); identical in both runs of the pair",
        "mix_rows": n_rows, "dann_groups": n_groups, "group_counts": counts,
        "window_census": cens, "n_steps": n_steps,
        "sets_per_batch": SETS_PER_BATCH, "pairs_per_batch": PAIRS_PER_BATCH,
        "init_fingerprint": fp, "init_fingerprint_scope": "trunk+task_head",
        "perm_fingerprint": perm_fingerprint(perm),
        "bars": BARS, "seed_sd": SEED_SD, "control": CONTROL,
        "train_seconds": round(time.time() - t0, 1), "checkpoint": str(ckpt_dir),
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    })
    gf = res["gold_full"]
    print(f"gold_full {gf['auc']:.4f} (n={gf['n']})  gold {res['gold']['auc']:.4f}  "
          f"ragtruth_en {res['ragtruth_en']['auc']:.4f}  "
          f"ragtruth_nonen {res['ragtruth_nonen']['auc']:.4f}", flush=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"results -> {out}")
    print(f"=== H142 G1 {args.run.upper()} TRAIN+INDOMAIN DONE ===", flush=True)


if __name__ == "__main__":
    main()

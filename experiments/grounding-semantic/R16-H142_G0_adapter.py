"""R16-H142 G0 - adapter-side-head gate: condition the per-window sentence score
on the window-ensemble embedding, keep the hard max/min aggregation untouched.

R16-H140 showed a learned attention readout over the window-ensemble embeddings
lifts pubmedqa +0.0711; R16-H141 proved that lift is EMBEDDING CONTENT (the best
scalar window statistic recovers only 13% of it). H142 captures that payload
legally: an adapter branch parallel to the residual stream that feeds the
ensemble context into the PER-WINDOW score, so the serving aggregation stays
exactly max-over-windows then min-over-sentences.

  cls_k          = trunk([sentence ; window_k])[CLS]        (per (sentence, window) pair)
  ctx            = mean_k cls_k                              (window-ensemble context)
  logit_k        = score_head(cls_k) + adapter([LN(cls_k) ; LN(ctx)])
  sentence score = max_k sigmoid(logit_k)                    UNCHANGED
  response score = min over sentences                        UNCHANGED

The adapter's output layer is zero-initialised, so at step 0 the model reproduces
the frozen R10-H108 draw-1 read exactly and can only depart from it by training.

Trainable: adapter + score head + the last 2 trunk layers. Everything else frozen.
Training data: the R16-H140 G1 public-only slice, imported from that script so it
is the same rows by construction - RAGTruth train + the R10-H108 manufactured
lane. No private gold, no RAGBench/arena rows, ever.

The RAGBench arena and the private gold set are EVALUATION ONLY.

Run (detached, GPU2):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 \
  nohup setsid uv run python experiments/grounding-semantic/R16-H142_G0_adapter.py \
    >> logs/R16-H142_G0_gate.log 2>&1 &
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util
import json
import pathlib
import random
import time

import numpy as np
import polars as pl
import torch
from torch import nn

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
BASE = ROOT / "models" / "R10-H108-lane-draw1"
CKPT_DIR = ROOT / "models" / "R16-H142-G0"
OUT = HERE / "R16-H142_G0_result.json"
GOLD_PAIRS = HERE / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"

SEED = 1142
MAX_LEN = 512
WIN, STRIDE = 1500, 750

EPOCHS = 2
PAIRS_PER_BATCH = 64  # 3.6 GB peak at 512 tokens on the RTX 5000 Ada
SETS_PER_BATCH = 48
LR_TRUNK = 1e-5  # the campaign's fine-tuning rate (R9-H105 recipe)
LR_HEAD = 1e-3  # the R16-H140 readout rate for the fresh parameters
WEIGHT_DECAY = 0.01
WARMUP_FRAC, CLIP = 0.1, 1.0
ADAPTER_HIDDEN = 512
UNFROZEN_LAYERS = 2
VAL_FRAC = 0.08
VAL_EVERY_FRAC = 0.5  # validation passes per epoch
RESUME_EVERY = 500
EVAL_BATCH = 64

# Same-checkpoint hard-max baseline: the banked windowed decomposed read of
# models/R10-H108-lane-draw1 (R10-H108_lane_draw1_windowed_result.json). R16-H140
# and R16-H141 each reproduced it from an independent cache to within 4.4e-5.
BASELINE = {
    "covidqa": 0.7516, "delucionqa": 0.7355, "emanual": 0.6719, "expertqa": 0.7496,
    "finqa": 0.7291, "hagrid": 0.6599, "hotpotqa": 0.6965, "pubmedqa": 0.5907,
    "tatqa": 0.7391, "techqa": 0.7379,
}
BASELINE_MEAN = 0.70618          # this checkpoint (draw 1)
BASELINE_LANE_2DRAW_MEAN = 0.70496  # the registered H108 lane figure (draws 1+2)
BASELINE_GOLD_FULL = {"auc": 0.8589, "f1": 0.7333, "n": 2752}

KILL_BLIND_MEAN = 0.65
KILL_GOLD_FULL = 0.75


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CACHE_MOD = _mod("h140cache", "R16-H140_G1_cache.py")  # the exact G1 training slice
H92 = _mod("h92", "R8-H92_decomposed_arena.py")        # sentence splitter
M59 = H92.ARENA.M59                                    # auc_and_f1, chunk_max_chars


def windows(chunk):
    """1,500-char sliding windows at stride 750, final window flush to the end."""
    n = len(chunk)
    if n <= WIN:
        return [chunk]
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return [chunk[s : s + WIN] for s in starts]


# --- model ---------------------------------------------------------------------


class EnsembleAdapterScorer(nn.Module):
    """Trunk + score head + an adapter branch parallel to the residual stream.

    The adapter consumes [per-window sentence representation ; window-ensemble
    context] and its output is added to the score head's logit. Its final layer
    starts at zero, so the module is initially the frozen incumbent scorer.
    """

    def __init__(self, trunk, task_head_state, hidden=ADAPTER_HIDDEN):
        super().__init__()
        self.trunk = trunk
        d = trunk.config.hidden_size
        self.score_head = nn.Linear(d, 1)
        self.score_head.load_state_dict(task_head_state)
        self.h_norm = nn.LayerNorm(d)
        self.ctx_norm = nn.LayerNorm(d)
        self.adapter = nn.Sequential(
            nn.Linear(2 * d, hidden), nn.GELU(), nn.Linear(hidden, 1)
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
        """cls [P,d], ctx_rows [P,d] (this pair's set context) -> logit [P]."""
        z = torch.cat([self.h_norm(cls), self.ctx_norm(ctx_rows)], dim=-1)
        return self.score_head(cls).squeeze(-1) + self.adapter(z).squeeze(-1)

    def logits_from_cls(self, cls, set_index, n_sets):
        ctx = self.pool_ctx(cls, set_index, n_sets)
        return self.pair_logits(cls, ctx[set_index])


def build_model():
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(BASE))
    trunk = AutoModel.from_pretrained(str(BASE / "trunk"), attn_implementation="sdpa")
    trunk.config.reference_compile = False  # mmBERT/ModernBERT compile path hangs here
    st = torch.load(BASE / "dann_student.pt", map_location="cpu", weights_only=False)
    model = EnsembleAdapterScorer(trunk, st["task_head"]).cuda()

    for p in model.trunk.parameters():
        p.requires_grad_(False)
    for layer in model.trunk.layers[-UNFROZEN_LAYERS:]:
        for p in layer.parameters():
            p.requires_grad_(True)
    return model, tok


def trainable_state(model):
    return {k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
            if not k.startswith("trunk.") or _in_unfrozen(model, k)}


def _in_unfrozen(model, key):
    n = len(model.trunk.layers)
    return any(key.startswith(f"trunk.layers.{i}.") for i in range(n - UNFROZEN_LAYERS, n))


# --- training data --------------------------------------------------------------


def load_slice():
    """The R16-H140 G1 public-only slice, produced by that script's own functions."""
    rows = CACHE_MOD.ragtruth_rows() + CACHE_MOD.lane_rows()
    sents = [r[0] for r in rows]
    wsets = [r[1] for r in rows]
    labels = np.array([r[2] for r in rows], dtype=np.float32)
    sources = [r[3] for r in rows]
    sizes = np.array([len(w) for w in wsets], dtype=np.int32)
    comp = {
        "n_sentence_sets": len(rows),
        "n_pairs": int(sizes.sum()),
        "positives": float(labels.mean()),
        "sentence_sets_by_source": {s: int(sum(1 for x in sources if x == s))
                                    for s in sorted(set(sources))},
        "window_set_size": {"mean": float(sizes.mean()), "max": int(sizes.max())},
        "provenance": "PUBLIC ONLY - RAGTruth train + R10-H108 manufactured lane, "
                      "imported from R16-H140_G1_cache.py. No private gold rows, "
                      "no RAGBench/arena rows.",
    }
    return sents, wsets, labels, sizes, comp


def pack_batches(ids, sizes, rng):
    """Random-order greedy packing under the pair cap; no size sorting, so batch
    composition carries no window-count bias."""
    ids = list(ids)
    rng.shuffle(ids)
    out, cur, tot = [], [], 0
    for i in ids:
        k = int(sizes[i])
        if cur and (len(cur) + 1 > SETS_PER_BATCH or tot + k > PAIRS_PER_BATCH):
            out.append(cur)
            cur, tot = [], 0
        cur.append(i)
        tot += k
    if cur:
        out.append(cur)
    return out


def encode_batch(tok, sents, wsets, batch):
    flat_s, flat_w, set_index = [], [], []
    for r, i in enumerate(batch):
        for w in wsets[i]:
            flat_s.append(sents[i])
            flat_w.append(w)
            set_index.append(r)
    enc = tok(flat_s, flat_w, return_tensors="pt", padding=True,
              truncation=True, max_length=MAX_LEN)
    enc = {k: v.cuda() for k, v in enc.items()}
    return enc, torch.tensor(set_index, dtype=torch.long, device="cuda")


@torch.inference_mode()
def val_pass(model, tok, sents, wsets, labels, batches):
    from sklearn.metrics import roc_auc_score

    model.eval()
    preds, ys = [], []
    for b in batches:
        enc, si = encode_batch(tok, sents, wsets, b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cls = model.encode(enc)
        lg = model.logits_from_cls(cls.float(), si, len(b))
        agg = torch.full((len(b),), -1e9, device="cuda")
        agg = agg.scatter_reduce(0, si, lg, reduce="amax")
        preds.append(agg.float().cpu().numpy())
        ys.append(labels[b])
    model.train()
    return float(roc_auc_score(np.concatenate(ys), np.concatenate(preds)))


def train(model, tok, sents, wsets, labels, sizes):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(labels))
    n_val = int(VAL_FRAC * len(labels))
    val_ids, tr_ids = perm[:n_val], perm[n_val:]
    val_batches = pack_batches(val_ids, sizes, random.Random(SEED))

    epoch_batches = [pack_batches(tr_ids, sizes, random.Random(SEED + e))
                     for e in range(EPOCHS)]
    total_steps = sum(len(b) for b in epoch_batches)
    val_every = max(1, int(len(epoch_batches[0]) * VAL_EVERY_FRAC))

    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("trunk.")]
    trunk_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith("trunk.")]
    n_head = sum(p.numel() for p in head_params)
    n_trunk = sum(p.numel() for p in trunk_params)
    print(f"trainable: adapter+score-head {n_head:,}  last-{UNFROZEN_LAYERS} trunk layers "
          f"{n_trunk:,}  total {n_head + n_trunk:,}", flush=True)
    print(f"schedule: {EPOCHS} epochs, {total_steps} steps, {len(val_batches)} val batches, "
          f"validation every {val_every} steps", flush=True)

    opt = torch.optim.AdamW(
        [{"params": trunk_params, "lr": LR_TRUNK}, {"params": head_params, "lr": LR_HEAD}],
        lr=LR_TRUNK, weight_decay=WEIGHT_DECAY,
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[LR_TRUNK, LR_HEAD], total_steps=total_steps,
        pct_start=WARMUP_FRAC, anneal_strategy="linear",
    )
    lossf = nn.BCEWithLogitsLoss()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    resume_path = CKPT_DIR / "resume.pt"
    gstep, best_val, best_state, hist = 0, -1.0, None, []
    if resume_path.exists():
        st = torch.load(resume_path, map_location="cuda", weights_only=False)
        model.load_state_dict(st["model"], strict=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        gstep, best_val, best_state, hist = st["gstep"], st["best_val"], st["best_state"], st["hist"]
        print(f"resumed from {resume_path} at step {gstep}/{total_steps}", flush=True)

    def save_resume():
        tmp = resume_path.with_suffix(".tmp")
        torch.save({"model": trainable_state(model), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "gstep": gstep, "best_val": best_val,
                    "best_state": best_state, "hist": hist}, tmp)
        tmp.replace(resume_path)  # atomic: a kill mid-write cannot corrupt the resume

    model.train()
    t0 = time.time()
    run, nrun = 0.0, 0
    seen = 0
    for ep in range(EPOCHS):
        for b in epoch_batches[ep]:
            if seen < gstep:  # fast-forward over already-consumed batches on resume
                seen += 1
                continue
            enc, si = encode_batch(tok, sents, wsets, b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cls = model.encode(enc)
            lg = model.logits_from_cls(cls.float(), si, len(b))
            agg = torch.full((len(b),), -1e9, device="cuda").scatter_reduce(
                0, si, lg, reduce="amax")
            y = torch.from_numpy(labels[b]).cuda()
            loss = lossf(agg, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at step {gstep}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], CLIP)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            run += float(loss)
            nrun += 1
            gstep += 1
            seen += 1
            if gstep % 100 == 0:
                print(f"  step {gstep}/{total_steps} ep {ep} loss {run / max(nrun, 1):.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e}/{sched.get_last_lr()[1]:.2e} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                run, nrun = 0.0, 0
            if gstep % val_every == 0 or gstep == total_steps:
                vauc = val_pass(model, tok, sents, wsets, labels, val_batches)
                hist.append({"step": gstep, "epoch": ep, "val_sentence_auc": round(vauc, 4),
                             "sec": round(time.time() - t0, 1)})
                print(f"  [val] step {gstep} sentence AUROC {vauc:.4f} "
                      f"(best {max(best_val, vauc):.4f})", flush=True)
                if vauc > best_val:
                    best_val = vauc
                    best_state = trainable_state(model)
                save_resume()
            elif gstep % RESUME_EVERY == 0:
                save_resume()

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    model.eval()
    torch.save({"state": best_state, "base": str(BASE), "best_val_sentence_auc": best_val,
                "hist": hist, "seed": SEED}, CKPT_DIR / "adapter.pt")
    resume_path.unlink(missing_ok=True)  # the run is done; a stale resume would rerun it
    print(f"checkpoint saved -> {CKPT_DIR / 'adapter.pt'}  best val {best_val:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return {"best_val_sentence_auc": round(best_val, 4), "history": hist,
            "total_steps": total_steps, "train_seconds": round(time.time() - t0, 1)}


# --- evaluation reads ------------------------------------------------------------


@torch.inference_mode()
def score_sets(model, tok, flat_s, flat_w, set_index, n_sets, tag=""):
    """Two-phase read: encode every (sentence, window) pair, then pool the
    window-ensemble context per set and take the max over the set."""
    n = len(flat_s)
    cls_all = torch.zeros(n, model.trunk.config.hidden_size, dtype=torch.float32)
    t0 = time.time()
    for i in range(0, n, EVAL_BATCH):
        enc = tok(flat_s[i : i + EVAL_BATCH], flat_w[i : i + EVAL_BATCH],
                  return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        enc = {k: v.cuda() for k, v in enc.items()}
        cls_all[i : i + EVAL_BATCH] = model.encode(enc).float().cpu()
        if (i // EVAL_BATCH) % 200 == 0 and i:
            print(f"    {tag} {i}/{n} ({i / max(time.time() - t0, 1e-9):.0f} pairs/s)", flush=True)
    # The ensemble context is pooled over the WHOLE set first, then the adapter runs
    # in pair-chunks - so a set spanning a chunk boundary still sees its full context.
    si = torch.as_tensor(set_index, dtype=torch.long).cuda()
    ctx = model.pool_ctx(cls_all.cuda(), si, n_sets)
    agg = torch.full((n_sets,), -1e9, device="cuda")
    step = 200_000
    for a in range(0, n, step):
        b = min(a + step, n)
        lg = model.pair_logits(cls_all[a:b].cuda(), ctx[si[a:b]])
        agg = agg.scatter_reduce(0, si[a:b], lg, reduce="amax")
    return agg.float().cpu().numpy()


def read_arena(model, tok):
    ARENA = H92.ARENA
    subs = ARENA.load_subsets()
    rows = {}
    for sub, (claims, chunks, y) in subs.items():
        flat_s, flat_w, set_index, owner = [], [], [], []
        for i, (c, ks) in enumerate(zip(claims, chunks, strict=True)):
            wlist = [w for k in ks for w in windows(k)]
            for s in H92.sentences(c):
                sid = len(owner)
                owner.append(i)
                for w in wlist:
                    flat_s.append(s)
                    flat_w.append(w)
                    set_index.append(sid)
        owner = np.array(owner)
        s_sent = score_sets(model, tok, flat_s, flat_w, set_index, len(owner), tag=f"arena/{sub}")
        resp = np.array([s_sent[owner == i].min() for i in range(len(y))])
        auc, f1, _ = M59.auc_and_f1(y, resp)
        rows[sub] = {
            "n": len(y), "n_sent": len(owner), "n_window_pairs": len(flat_s),
            "adapter_auc": round(auc, 4), "f1": round(f1, 4),
            "baseline_hardmax_auc": BASELINE[sub],
            "delta": round(auc - BASELINE[sub], 4),
        }
        print(f"  {sub:14s} n={len(y):>4} adapter {auc:.4f}  baseline {BASELINE[sub]:.4f}  "
              f"delta {auc - BASELINE[sub]:+.4f}", flush=True)
    return rows


def read_gold_full(model, tok):
    """ALL 2,752 private gold claims - a pure held-out test set. Same architecture,
    the claim's chunk list playing the role of the window set."""
    df = pl.read_parquet(GOLD_PAIRS)
    claims, chunk_lists, labels = [], [], []
    for _owner, grp in df.group_by("owner"):
        claims.append(grp["claim"][0])
        chunk_lists.append(grp["chunk"].to_list())
        labels.append(int(grp["label"][0]))
    y = np.array(labels)
    flat_s, flat_w, set_index = [], [], []
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        for k in ks:
            flat_s.append(c)
            flat_w.append(k[: M59.CFG.chunk_max_chars])
            set_index.append(i)
    s = score_sets(model, tok, flat_s, flat_w, set_index, len(claims), tag="gold_full")
    auc, f1, _ = M59.auc_and_f1(y, s)
    return {"auc": round(auc, 4), "f1": round(f1, 4), "n": len(y),
            "n_pairs": len(flat_s)}


# --- driver ------------------------------------------------------------------------


def write_result(payload):
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\n  result -> {OUT}", flush=True)


def main():
    if OUT.exists():
        print(f"{OUT} already exists - nothing to do:\n{OUT.read_text()}", flush=True)
        print("=== H142 G0 GATE COMPLETE ===", flush=True)
        return

    print(f"=== R16-H142 G0 adapter gate {time.strftime('%F %T')} ===", flush=True)
    dev = torch.cuda.get_device_name(0)
    print(f"GPU: {dev}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})", flush=True)
    if "RTX 5000 Ada" not in dev:
        raise SystemExit(f"wrong GPU: {dev} - this gate is pinned to the RTX 5000 Ada (card 2)")

    model, tok = build_model()
    n_head = sum(p.numel() for n, p in model.named_parameters()
                 if p.requires_grad and not n.startswith("trunk."))

    done = CKPT_DIR / "adapter.pt"
    if done.exists():
        st = torch.load(done, map_location="cuda", weights_only=False)
        model.load_state_dict(st["state"], strict=False)
        model.eval()
        tinfo = {"best_val_sentence_auc": st["best_val_sentence_auc"], "history": st["hist"],
                 "resumed_from_checkpoint": True}
        comp = {"note": "training already complete; slice not re-materialised"}
        print(f"training already complete -> {done} (best val {st['best_val_sentence_auc']:.4f})",
              flush=True)
    else:
        print("loading the R16-H140 G1 public-only training slice ...", flush=True)
        sents, wsets, labels, sizes, comp = load_slice()
        print(f"slice: {comp['n_sentence_sets']} sentence-sets, {comp['n_pairs']} pairs, "
              f"positives {comp['positives']:.4f}, window sets mean "
              f"{comp['window_set_size']['mean']:.2f} max {comp['window_set_size']['max']}",
              flush=True)
        if comp["window_set_size"]["max"] > PAIRS_PER_BATCH:
            raise SystemExit("a training window set exceeds the per-batch pair cap")
        try:
            tinfo = train(model, tok, sents, wsets, labels, sizes)
        except FloatingPointError as e:
            write_result({
                "gate": "R16-H142 G0 adapter side-head", "seed": SEED,
                "kill": {"non_finite_loss": True, "reason": str(e)},
                "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
            })
            print("=== H142 G0 GATE COMPLETE ===", flush=True)
            return
        del sents, wsets, labels, sizes

    print("\nblind windowed decomposed read on the RAGBench arena "
          "(EVALUATION ONLY) ...", flush=True)
    per_sub = read_arena(model, tok)
    blind_mean = float(np.mean([v["adapter_auc"] for v in per_sub.values()]))

    print("\ngold_full held-out read ...", flush=True)
    gold = read_gold_full(model, tok)
    print(f"  gold_full auc {gold['auc']:.4f} f1 {gold['f1']:.4f} (baseline "
          f"{BASELINE_GOLD_FULL['auc']})", flush=True)

    kill = {
        "non_finite_loss": False,
        "blind_mean_below_0.65": bool(blind_mean < KILL_BLIND_MEAN),
        "gold_full_below_0.75": bool(gold["auc"] < KILL_GOLD_FULL),
    }
    kill["any"] = bool(any(kill.values()))

    payload = {
        "gate": "R16-H142 G0 adapter side-head (window-ensemble conditioning, "
                "hard max/min aggregation unchanged)",
        "base_checkpoint": str(BASE),
        "architecture": {
            "ensemble_pooling": "mean-pool over the sentence's window-set [CLS] vectors",
            "adapter": f"Linear(2*768 -> {ADAPTER_HIDDEN}) + GELU + Linear({ADAPTER_HIDDEN} -> 1), "
                       "zero-initialised output layer, added to the score-head logit "
                       "(parallel branch on the residual stream)",
            "aggregation": "max over windows, then min over sentences - UNCHANGED",
            "adapter_plus_head_params": int(n_head),
            "trunk_trainable_params": int(sum(
                p.numel() for n, p in model.named_parameters()
                if p.requires_grad and n.startswith("trunk."))),
            "frozen": f"all trunk parameters except the last {UNFROZEN_LAYERS} layers",
        },
        "seed": SEED,
        "precision": "bf16 autocast for training, fp32 for both reads (read protocol "
                     "byte-identical to the banked windowed read)",
        "blind_mean": round(blind_mean, 5),
        "baseline_blind_mean": BASELINE_MEAN,
        "baseline_lane_2draw_mean": BASELINE_LANE_2DRAW_MEAN,
        "blind_mean_delta": round(blind_mean - BASELINE_MEAN, 5),
        "per_subset": per_sub,
        "gold_full": gold,
        "baseline_gold_full": BASELINE_GOLD_FULL,
        "training": {**tinfo, "composition": comp, "epochs": EPOCHS,
                     "pairs_per_batch": PAIRS_PER_BATCH, "lr_trunk": LR_TRUNK,
                     "lr_head": LR_HEAD, "val_frac": VAL_FRAC,
                     "no_gold_rows": True, "no_arena_rows": True},
        "kill": {**kill,
                 "definition": "non-finite training loss OR blind mean < 0.65 OR gold_full < 0.75"},
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    }
    write_result(payload)

    print("\n" + "=" * 88)
    print("R16-H142 G0 RESULT")
    print("=" * 88)
    print(f"  blind mean {blind_mean:.5f}   baseline (same checkpoint) {BASELINE_MEAN}   "
          f"delta {blind_mean - BASELINE_MEAN:+.5f}")
    print(f"  pubmedqa {per_sub['pubmedqa']['adapter_auc']:.4f} "
          f"(baseline {BASELINE['pubmedqa']}, delta {per_sub['pubmedqa']['delta']:+.4f})")
    print(f"  gold_full {gold['auc']:.4f} (baseline {BASELINE_GOLD_FULL['auc']})")
    print(f"  kill triggered: {kill['any']}  {kill}")
    print("=== H142 G0 GATE COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()

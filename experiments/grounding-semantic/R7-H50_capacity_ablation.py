"""R7-H50 - is the grounding task capacity-limited at all?

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 7).

Before any architecture is designed, one question decides whether the design
matters: does model capacity move the number? Four checkpoints spanning 140M to
307M are distilled identically from the R7-H51 teacher corpus and scored on the
same held-out gold claims. If they all land within 0.02 macro-F1 of each other,
capacity is not the binding constraint, the architecture work is closed, and the
remaining lever is labels rather than models.

  mmBERT-small        140M   22L x 384   8192 ctx   multilingual
  mmBERT-base         307M   22L x 768   8192 ctx   multilingual
  mDeBERTa-v3-base    278M   12L x 768    512 ctx   the incumbent NLI backbone
  mDeBERTa minus 6L   ~235M   6L x 768    512 ctx   same width, half the depth

The last one is the direct depth probe: same family, same width, same tokenizer,
six fewer layers. It isolates depth from every other variable, which no
cross-family comparison can.

Teacher signal is the reranker score - the best single signal on this gold at
AUC 0.841, and the only one that is genuinely per-pair. Students are trained on
it as a soft target, then aggregated max-over-chunks exactly as the cascade
does, and scored against the human gold labels.

Splits are by CLAIM, not by pair: a claim's chunks all fall on one side, so no
student is scored on a claim it trained on.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R7-H50_capacity_ablation.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import json
import pathlib
import time

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, roc_auc_score
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = pathlib.Path(__file__).parent
PAIRS = HERE / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"
GOLD = HERE / "private-rag-forensics" / "gold" / "golden_grounding_evidence_verified.parquet"
OUT = HERE / "R7-H50_ablation.json"

MAX_LEN = 512
BATCH = 32
LR = 1e-5  # 2e-5 sent mDeBERTa-v3 to NaN by step 200 with a fresh head
WARMUP_FRAC = 0.1
CLIP = 1.0
TRAIN_PAIRS = 40_000  # subsample for a decisive read, not a final model
TEST_FRAC = 0.25
VAL_FRAC = 0.15  # threshold is fitted here, never on test
SEED = 0

CHECKPOINTS = [
    # Depth probe inside ONE family that trains stably. mDeBERTa-v3 diverged at
    # both 2e-5 and 1e-5 with warmup and clipping, so it cannot serve as the
    # depth arm; truncating mmBERT-base isolates depth with width, tokenizer,
    # embeddings and training recipe all held fixed.
    ("mmBERT-base-22L", "jhu-clsp/mmBERT-base", None),
    ("mmBERT-base-11L", "jhu-clsp/mmBERT-base", 11),
    ("mmBERT-base-6L", "jhu-clsp/mmBERT-base", 6),
    ("mmBERT-base-3L", "jhu-clsp/mmBERT-base", 3),
]


class PairSet(Dataset):
    def __init__(self, df, tok):
        self.claim = df["claim"].to_list()
        self.chunk = df["chunk"].to_list()
        self.y = df["rerank"].to_numpy().astype("float32")
        self.tok = tok

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.claim[i], self.chunk[i], self.y[i]

    def collate(self, batch):
        c, k, y = zip(*batch, strict=True)
        enc = self.tok(
            list(c),
            list(k),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        return enc, torch.tensor(y)


def truncate_layers(model, keep):
    """Drop encoder layers, keeping the first `keep`. Same width, same tokenizer,
    same embeddings - the only variable that moves is depth."""
    for attr in ("deberta", "bert", "roberta", "model"):
        base = getattr(model, attr, None)
        if base is None:
            continue
        # ModernBERT (mmBERT) keeps its stack at `.layers`; BERT/DeBERTa put it
        # at `.encoder.layer`. Verified against the live module tree rather than
        # assumed - the first version of this function guessed and silently
        # matched neither.
        if hasattr(base, "layers"):
            base.layers = torch.nn.ModuleList(list(base.layers[:keep]))
            model.config.num_hidden_layers = keep
            return model
        if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
            base.encoder.layer = torch.nn.ModuleList(list(base.encoder.layer[:keep]))
            model.config.num_hidden_layers = keep
            return model
    raise RuntimeError(
        f"could not locate the layer stack; top-level children were "
        f"{[n for n, _ in model.named_children()]}"
    )


def run_one(name, mid, keep_layers, train_df, val_df, test_df, gold):
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForSequenceClassification.from_pretrained(
        mid, num_labels=1, ignore_mismatched_sizes=True
    )
    if keep_layers:
        model = truncate_layers(model, keep_layers)
    model = model.cuda().train()
    n_par = sum(p.numel() for p in model.parameters())

    tr = PairSet(train_df, tok)
    dl = DataLoader(tr, batch_size=BATCH, shuffle=True, collate_fn=tr.collate, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    # DeBERTa-v3 diverges on a freshly-initialised regression head without both
    # of these; applied to every checkpoint so the comparison stays matched.
    total = len(train_df) // BATCH + 1
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=total, pct_start=WARMUP_FRAC, anneal_strategy="linear"
    )
    lossf = torch.nn.BCEWithLogitsLoss()

    t0 = time.time()
    for step, (enc, y) in enumerate(dl):
        enc = {k: v.cuda() for k, v in enc.items()}
        loss = lossf(model(**enc).logits.squeeze(-1), y.cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        sched.step()
        opt.zero_grad()
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name} diverged at step {step} - loss is not finite")
        if step % 200 == 0:
            print(
                f"    {name} step {step}/{len(dl)} loss {loss.item():.4f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    train_s = time.time() - t0

    model.eval()

    def predict(frame, m=model):  # bind explicitly - `del model` below hides it from the closure
        ds = PairSet(frame, tok)
        dl_x = DataLoader(ds, batch_size=64, collate_fn=ds.collate, num_workers=2)
        out, t1 = [], time.time()
        with torch.inference_mode():
            for enc, _ in dl_x:
                enc = {k: v.cuda() for k, v in enc.items()}
                out.append(torch.sigmoid(m(**enc).logits.squeeze(-1)).float().cpu().numpy())
        return np.concatenate(out), (time.time() - t1) / len(ds) * 1000

    def agg_by_claim(frame, preds):
        owners = frame["owner"].to_numpy()
        uniq = np.unique(owners)
        return (
            np.array([preds[owners == o].max() for o in uniq]),
            np.array([gold[o] for o in uniq]),
        )

    # Threshold is fitted on VALIDATION traces and then applied unchanged to
    # test. Selecting it on test - as the first version did - is test-set
    # fitting and inflates macro-F1.
    vp, _ = predict(val_df)
    va, vy = agg_by_claim(val_df, vp)
    grid = np.quantile(va, np.linspace(0.05, 0.95, 91))
    thr = max(grid, key=lambda t: f1_score(vy, (va >= t).astype(int), average="macro"))

    tp, infer_ms = predict(test_df)
    ta, ty = agg_by_claim(test_df, tp)
    auc = roc_auc_score(ty, ta)
    f1 = f1_score(ty, (ta >= thr).astype(int), average="macro")

    del model
    torch.cuda.empty_cache()
    return {
        "name": name,
        "params_M": round(n_par / 1e6, 1),
        "auc": round(float(auc), 4),
        "macro_f1": round(float(f1), 4),
        "train_s": round(train_s),
        "threshold": round(float(thr), 4),
        "infer_ms_per_pair": round(infer_ms, 2),
    }


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    df = pl.read_parquet(PAIRS)

    # Split by TRACE, not by claim. 2,752 claims come from only 639 traces
    # sharing 619 source documents (4.31 claims/trace, largest 35), so a
    # claim-level split trains and tests on the SAME document and leaks. The
    # log's own situational overview flags this: claims sharing a trace's
    # evidence are correlated, so the effective sample is ~639, not 2,752.
    gold_df = pl.read_parquet(GOLD).with_row_index("owner")
    df = df.join(gold_df.select(["owner", "trace_id"]), on="owner", how="left")
    gold = {o: int(v) for o, v in zip(df["owner"].to_list(), df["label"].to_list(), strict=True)}

    traces = np.array(sorted(set(df["trace_id"].to_list())))
    rng = np.random.default_rng(SEED)
    rng.shuffle(traces)
    n_test = int(len(traces) * TEST_FRAC)
    n_val = int(len(traces) * VAL_FRAC)
    test_tr = set(traces[:n_test].tolist())
    val_tr = set(traces[n_test : n_test + n_val].tolist())

    test_df = df.filter(pl.col("trace_id").is_in(list(test_tr)))
    val_df = df.filter(pl.col("trace_id").is_in(list(val_tr)))
    train_df = df.filter(~pl.col("trace_id").is_in(list(test_tr | val_tr)))
    if len(train_df) > TRAIN_PAIRS:
        train_df = train_df.sample(TRAIN_PAIRS, seed=SEED)
    print(
        f"traces {len(traces)} -> train {len(traces) - n_test - n_val} / val {n_val} / test {n_test}\n"
        f"pairs  train {len(train_df)} / val {len(val_df)} / test {len(test_df)}\n"
        f"test claims {test_df['owner'].n_unique()}, base rate "
        f"{np.mean([gold[o] for o in test_df['owner'].unique().to_list()]):.3f}\n",
        flush=True,
    )

    results = []
    for name, mid, keep in CHECKPOINTS:
        print(f"  === {name} ({mid}{f', first {keep} layers' if keep else ''})", flush=True)
        r = run_one(name, mid, keep, train_df, val_df, test_df, gold)
        results.append(r)
        print(
            f"    -> AUC {r['auc']}  macro-F1 {r['macro_f1']}  "
            f"{r['params_M']}M  {r['infer_ms_per_pair']} ms/pair\n",
            flush=True,
        )

    OUT.write_text(json.dumps(results, indent=2))
    f1s = [r["macro_f1"] for r in results]
    spread = max(f1s) - min(f1s)

    print("=" * 92)
    print("R7-H50 RESULT - capacity ablation, distilled from the R7-H51 teacher")
    print("=" * 92)
    print(
        f"{'checkpoint':22s} {'params':>8} {'AUC':>7} {'macro-F1':>9} {'ms/pair':>9} {'train s':>8}"
    )
    for r in results:
        print(
            f"{r['name']:22s} {r['params_M']:>7.1f}M {r['auc']:>7.4f} {r['macro_f1']:>9.4f} "
            f"{r['infer_ms_per_pair']:>9.2f} {r['train_s']:>8}"
        )
    print("\n  incumbent cascade macro-F1 0.824 (all-lang gold), naive baseline 0.417")
    print("  teacher reranker AUC on this corpus 0.8289")
    print(f"\n  macro-F1 spread across 140M-307M: {spread:.4f}")
    print(
        f"  bar: spread <= 0.02 -> the task is NOT capacity-limited  ->  "
        f"{'CONFIRMED' if spread <= 0.02 else 'REFUTED'}"
    )
    if spread <= 0.02:
        print("  -> architecture work is closed; the binding constraint is the labels")
    print(f"\n  results -> {OUT}")


if __name__ == "__main__":
    main()

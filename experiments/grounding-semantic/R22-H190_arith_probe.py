"""R22-H190 ARITHMETIC-CAPABILITY PROBE SUITE - can this trunk verify a computation?

Registered in docs/experiments/semantic-grounding-experiments.md, block "R22-H190
ARITHMETIC-CAPABILITY PROBE SUITE", author-ordered 2026-08-18 ~02:45.

R22-H188 killed a contract-conforming derivation lane that sat at 4.0% of a
721,210-row mix. That result cannot distinguish "the architecture cannot hold the
predicate" from "the lane was drowned". Training the same lane ALONE separates
those two readings, and four variant conditions locate the constraint.

    P1  isolation   (claim, evidence) pairs, full fine-tune, Linear(d,1) head
    P2  frozen      trunk frozen, head only - is the predicate already there
    P3  direction   same operands, claim asserts a RELATION not a computed value
    P4  deephead    2-layer MLP head, otherwise P1
    P5  clean       claim ONLY - operands and asserted result, no evidence

P5 is NOT a leak. The lane's claims restate both operands by construction, so a
claim-only read is a legitimate arithmetic probe; the contract's C5 control reads
0.5000 here because TF-IDF-plus-logistic cannot do arithmetic, not because the
information is absent. P1 against P5 measures whether evidence helps or distracts.

ENTIRELY OFF-ARENA. No arena read, no promotion path, no serving change.

The recipe mirrors the flagship's so the answer transfers: mmBERT-base trunk,
CLS readout, BCE, LR 1e-5, OneCycleLR with 10% warmup, grad clip 1.0, MAX_LEN 512.
Difference from the flagship: 3 epochs rather than 1, because the lane alone is
2% of the flagship's row count and one pass would undertrain it - the probe must
fail for capability reasons, never for step count.

PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set: expandable_segments kills
.to("cuda") under WSL2 on this box.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
      uv run python experiments/grounding-semantic/R22-H190_arith_probe.py --condition P1
"""

import argparse
import json
import pathlib

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

HERE = pathlib.Path(__file__).parent
LANE = HERE / "R22-H187_num_derive_lane.parquet"
TRUNK = "jhu-clsp/mmBERT-base"

SEED = 2190
MAX_LEN, LR, EPOCHS, BATCH = 512, 1e-5, 3, 32
HOLDOUT_FRAC = 0.20
BARS = {"learnable_at_or_above": 0.70, "not_learnable_below": 0.55}

CONDITIONS = ("P1", "P2", "P3", "P4", "P5")


def auroc(y, s):
    """Rank AUROC with proper tie handling - the campaign's own convention."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def build_direction(df):
    """P3 - the same operands, asserting a RELATION instead of a computed value.

    One true-relation and one false-relation claim per source pair, over identical
    operands, so the legs differ only in the relation word and every surface
    statistic besides that word is held. Chance stays exactly 0.5."""
    rows = []
    for r in df.iter_rows(named=True):
        a, b = r["operand_a"], r["operand_b"]
        if a == b:
            continue
        col, ka, kb = r["column"], r["row_key_a"], r["row_key_b"]
        stem = (f"The {col} of {ka} is {r['operand_a_str']} and the {col} of {kb} "
                f"is {r['operand_b_str']}, so the {col} of {ka} ")
        true_rel, false_rel = ("exceeds", "is below") if a > b else ("is below", "exceeds")
        for lab, rel in ((1, true_rel), (0, false_rel)):
            rows.append({**{k: r[k] for k in ("pair_id", "doc_id", "chunk",
                                              "neg_family", "result_digits")},
                         "label": lab, "claim": f"{stem}{rel} that of {kb}."})
    return pl.DataFrame(rows)


class Probe(nn.Module):
    def __init__(self, base, deep_head=False):
        super().__init__()
        self.trunk = base
        d = base.config.hidden_size
        self.task_head = (nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
                          if deep_head else nn.Linear(d, 1))

    def forward(self, enc):
        cls = self.trunk(**enc).last_hidden_state[:, 0]
        return self.task_head(cls).squeeze(-1)


def encode(tok, claims, evidence):
    if evidence is None:                      # P5 - claim only
        return tok(claims, return_tensors="pt", padding=True,
                   truncation=True, max_length=MAX_LEN)
    return tok(claims, evidence, return_tensors="pt", padding=True,
               truncation=True, max_length=MAX_LEN)


def run(cond):
    out = HERE / f"R22-H190_{cond}_result.json"
    if out.exists() and out.stat().st_size > 0:
        print(f"SKIP (on disk: {out.name})", flush=True)
        return

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = pl.read_parquet(LANE)
    if cond == "P3":
        df = build_direction(df)

    # Document-disjoint split, IDENTICAL across conditions: the holdout document
    # set is chosen from the source lane's doc list under a fixed seed, so every
    # condition scores the same tables and the five numbers are comparable.
    docs = sorted(pl.read_parquet(LANE)["doc_id"].unique().to_list())
    rng = np.random.default_rng(SEED)
    held = set(np.array(docs)[rng.permutation(len(docs))[:int(round(HOLDOUT_FRAC * len(docs)))]])
    tr = df.filter(~pl.col("doc_id").is_in(list(held)))
    te = df.filter(pl.col("doc_id").is_in(list(held)))
    print(f"[{cond}] train {tr.height} rows / {tr['doc_id'].n_unique()} docs   "
          f"holdout {te.height} rows / {te['doc_id'].n_unique()} docs   "
          f"holdout positive rate {te['label'].mean():.4f}", flush=True)

    tok = AutoTokenizer.from_pretrained(TRUNK)
    base = AutoModel.from_pretrained(TRUNK, attn_implementation="sdpa")
    base.config.reference_compile = False     # mmBERT/ModernBERT compile path hangs
    model = Probe(base, deep_head=(cond == "P4")).cuda()

    if cond == "P2":                          # frozen trunk, head only
        for p in model.trunk.parameters():
            p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[{cond}] trainable params {sum(p.numel() for p in trainable)/1e6:.1f}M "
          f"of {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    use_evidence = cond != "P5"
    tr_c, tr_e = tr["claim"].to_list(), (tr["chunk"].to_list() if use_evidence else None)
    tr_y = np.asarray(tr["label"].to_list(), dtype=np.float32)
    n_steps = EPOCHS * int(np.ceil(len(tr_c) / BATCH))
    opt = torch.optim.AdamW(trainable, lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=n_steps,
                                                pct_start=0.10)
    lossf = nn.BCEWithLogitsLoss()

    step = 0
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.default_rng(SEED + ep).permutation(len(tr_c))
        for s in range(0, len(perm), BATCH):
            idx = perm[s:s + BATCH]
            enc = encode(tok, [tr_c[i] for i in idx],
                         [tr_e[i] for i in idx] if use_evidence else None)
            enc = {k: v.cuda() for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(enc)
            loss = lossf(logit.float(), torch.tensor(tr_y[idx], device="cuda"))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                print(f"  [{cond}] step {step}/{n_steps} ep {ep} loss {loss.item():.4f}",
                      flush=True)

    model.eval()
    te_c, te_e = te["claim"].to_list(), (te["chunk"].to_list() if use_evidence else None)
    scores = []
    with torch.no_grad():
        for s in range(0, len(te_c), 64):
            enc = encode(tok, te_c[s:s + 64], te_e[s:s + 64] if use_evidence else None)
            enc = {k: v.cuda() for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores.append(model(enc).float().cpu().numpy())
    sc = np.concatenate(scores)
    y = np.asarray(te["label"].to_list(), dtype=np.int64)
    a = auroc(y, sc)
    verdict = ("LEARNABLE" if a >= BARS["learnable_at_or_above"]
               else "NOT LEARNABLE" if a < BARS["not_learnable_below"] else "PARTIAL")

    strat = {}
    for key in ("result_digits", "neg_family"):
        if key not in te.columns:
            continue
        vals = te[key].to_list()
        strat[key] = {str(v): {"n": int(np.sum(np.asarray(vals) == v)),
                               "auroc": round(auroc(y[np.asarray(vals) == v],
                                                    sc[np.asarray(vals) == v]), 5)}
                      for v in sorted(set(vals))}

    res = {"experiment": "R22-H190 arithmetic-capability probe suite",
           "condition": cond, "trunk": TRUNK, "seed": SEED,
           "off_arena": "diagnostic only - no arena read, no promotion path",
           "n_train": tr.height, "n_holdout": te.height,
           "n_train_docs": tr["doc_id"].n_unique(), "n_holdout_docs": te["doc_id"].n_unique(),
           "holdout_positive_rate": round(float(te["label"].mean()), 5),
           "epochs": EPOCHS, "batch": BATCH, "lr": LR, "n_steps": n_steps,
           "holdout_auroc": round(a, 5), "bars": BARS, "verdict": verdict,
           "stratified": strat,
           "note": "Numbers recorded, not adjudicated - the coordinator adjudicates."}
    out.write_text(json.dumps(res, indent=2))
    print(f"\n=== {cond}  holdout AUROC {a:.5f}  -> {verdict} ===", flush=True)
    for k, v in strat.items():
        print(f"  by {k}: " + "  ".join(f"{kk}={vv['auroc']}(n={vv['n']})"
                                        for kk, vv in v.items()), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    run(ap.parse_args().condition)

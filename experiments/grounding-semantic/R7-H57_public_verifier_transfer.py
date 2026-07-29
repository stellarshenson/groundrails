"""R7-H57 - does a verifier trained on public data transfer to our distribution?

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 7).

The dataset survey recommends training on RAGTruth, LettuceDetect-prose and
RAGBench, and warns that domain transfer will disappoint because none of them is
production assistant traffic over a private corpus. That warning is testable for
the price of an inference pass, in the REVERSE direction of the expensive
experiment: instead of training on public data and testing on ours, take a model
somebody already trained on exactly that stack and test it on ours.

`KRLabsOrg/lettucedect-v2-mmbert-base` is MIT, 307M, mmBERT-base backbone, and
trained on RAGTruth + its translations + LettuceDetect-prose - the top two
recommendations. If it holds up on our gold, public training data transfers. If
it collapses, no quantity of that data will help and the budget belongs on
annotating private traces.

FORMAT, read off the model card rather than guessed (the standing rule this
project earned in round 4):
  - it is a TOKEN classifier, not a pairwise classifier
  - `tok(context, answer, truncation="only_first")`
  - label 0 = supported, label 1 = hallucinated; only answer tokens are scored
  - our mapping: evidence chunk -> context, claim -> answer

A continuous score is needed for AUC, so the claim's grounding score is
`1 - max P(hallucinated)` over its answer tokens, and max-over-chunks aggregates
exactly as the cascade serves.

STAGE 1 IS A POSITIVE CONTROL and it gates everything: the same 20 trivially
separable pairs from R4-H29. A verifier that cannot separate a claim quoted
verbatim from its own chunk against the same claim versus a bread recipe cannot
be trusted at 0.75 on real data, and its null would be uninterpretable.

Runs on the idle RTX 5000 Ada so it does not contend with the depth probe.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      uv run python experiments/grounding-semantic/R7-H57_public_verifier_transfer.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import json
import pathlib

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, roc_auc_score
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

HERE = pathlib.Path(__file__).parent
PAIRS = HERE / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"
GOLD = HERE / "private-rag-forensics" / "gold" / "golden_grounding_evidence_verified.parquet"
OUT = HERE / "R7-H57_transfer.json"

MID = "KRLabsOrg/lettucedect-v2-mmbert-base"
MAX_LEN = 4096
BATCH = 16
OUR_CASCADE_AUC = 0.8619  # reranker on the same held-out traces, R7-H50


def load_control():
    """The unchanged R4-H29 control, rebuilt from the R6-H42 generator."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("h42", HERE / "R6-H42_pleias_rag_protocol.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pos = [(c, ev, 1) for c, ev in mod.CONTROL]
    neg = [(c, mod.RECIPE, 0) for c, _ in mod.CONTROL]
    return pos + neg


@torch.inference_mode()
def score(model, tok, pairs, batch=BATCH):
    """P(grounded) per (claim, chunk) = 1 - max P(hallucinated) over answer tokens."""
    out = np.zeros(len(pairs), dtype=np.float32)
    for i in range(0, len(pairs), batch):
        chunk_batch = pairs[i : i + batch]
        enc = tok(
            [c for _, c in chunk_batch],  # context = evidence
            [q for q, _ in chunk_batch],  # answer  = claim
            truncation="only_first",
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        probs = torch.softmax(model(**enc).logits.float(), dim=-1)[..., 1]  # P(hallucinated)
        # Score ONLY the answer segment; ModernBERT has no token_type_ids, so the
        # answer is everything after the final separator of the context segment.
        ids = enc["input_ids"]
        sep = tok.sep_token_id
        for j in range(ids.shape[0]):
            row = ids[j].tolist()
            first_sep = row.index(sep) if sep in row else 0
            mask = enc["attention_mask"][j].bool().clone()
            mask[: first_sep + 1] = False
            p = probs[j][mask]
            out[i + j] = 1.0 - (p.max().item() if p.numel() else 1.0)
    return out


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModelForTokenClassification.from_pretrained(MID, dtype=torch.float16)
    model = model.cuda().eval()
    print(
        f"{MID}  labels {model.config.id2label}  "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n",
        flush=True,
    )

    # ---- stage 1: positive control ----
    ctrl = load_control()
    s = score(model, tok, [(c, ev) for c, ev, _ in ctrl])
    y = np.array([lab for _, _, lab in ctrl])
    ctrl_auc = roc_auc_score(y, s)
    thr = float(np.median(s))
    ctrl_acc = float(((s >= thr).astype(int) == y).mean())
    print("=" * 88)
    print("STAGE 1 - positive control, 20 trivially separable pairs")
    print("=" * 88)
    for (c, _, lab), sc in list(zip(ctrl, s, strict=True))[:6]:
        print(f"  {'SUP  ' if lab else 'UNSUP'} {sc:.3f}  {c[:62]}")
    print(f"\n  control AUC {ctrl_auc:.4f}, accuracy at median {ctrl_acc:.2f}")
    if ctrl_auc < 0.90:
        print("  FAIL - cannot separate trivial pairs; any number on real data would be")
        print("         uninterpretable. Stopping before the gold run.")
        OUT.write_text(json.dumps({"control_auc": ctrl_auc, "stopped": True}, indent=2))
        return
    print("  PASS - proceeding to the gold\n")

    # ---- stage 2: our gold, held-out traces only, per language ----
    df = pl.read_parquet(PAIRS)
    g = pl.read_parquet(GOLD).with_row_index("owner")
    df = df.join(g.select(["owner", "trace_id"]), on="owner", how="left")
    traces = np.array(sorted(set(df["trace_id"].to_list())))
    rng = np.random.default_rng(0)
    rng.shuffle(traces)
    test = set(traces[: int(len(traces) * 0.25)].tolist())
    t = df.filter(pl.col("trace_id").is_in(list(test)))
    print(
        f"held-out traces {len(test)}, pairs {len(t)}, claims {t['owner'].n_unique()}", flush=True
    )

    preds = score(model, tok, list(zip(t["claim"].to_list(), t["chunk"].to_list(), strict=True)))
    own = t["owner"].to_numpy()
    lang = dict(zip(t["owner"].to_list(), t["lang"].to_list(), strict=True))
    lab = dict(zip(t["owner"].to_list(), t["label"].to_list(), strict=True))
    uniq = np.unique(own)
    agg = np.array([preds[own == o].max() for o in uniq])
    y = np.array([int(lab[o]) for o in uniq])
    langs = np.array([lang[o][:2] for o in uniq])

    auc = roc_auc_score(y, agg)
    f1 = f1_score(y, (agg >= np.median(agg)).astype(int), average="macro")

    print("\n" + "=" * 88)
    print("R7-H57 RESULT - public-trained verifier on our gold")
    print("=" * 88)
    print(f"  overall            AUC {auc:.4f}   macro-F1 {f1:.4f}   n={len(uniq)}")
    print(f"  our cascade        AUC {OUR_CASCADE_AUC:.4f}   (same held-out traces)")
    print(f"  deficit            {auc - OUR_CASCADE_AUC:+.4f}\n")
    rows = {}
    for lg in sorted(set(langs)):
        m = langs == lg
        if m.sum() < 20 or len(set(y[m])) < 2:
            continue
        a = roc_auc_score(y[m], agg[m])
        rows[lg] = round(float(a), 4)
        print(f"  {lg:6s} n={int(m.sum()):>4}  base {y[m].mean():.3f}  AUC {a:.4f}")

    en = rows.get("en")
    non_en = [v for k, v in rows.items() if k != "en"]
    print("\n  bar: within 0.05 of our cascade on ANY slice -> public data transfers")
    best = max(rows.values()) if rows else auc
    verdict = (
        "TRANSFERS"
        if best >= OUR_CASCADE_AUC - 0.05
        else ("DOES NOT TRANSFER" if best < 0.65 else "PARTIAL")
    )
    print(f"  best slice {best:.4f}  ->  {verdict}")
    if en and non_en:
        print(
            f"  predicted asymmetry: non-EN should transfer better than EN "
            f"(EN {en:.4f} vs non-EN mean {np.mean(non_en):.4f})"
        )
    OUT.write_text(
        json.dumps(
            {
                "control_auc": ctrl_auc,
                "auc": auc,
                "macro_f1": f1,
                "by_lang": rows,
                "cascade_auc": OUR_CASCADE_AUC,
                "verdict": verdict,
            },
            indent=2,
        )
    )
    print(f"\n  results -> {OUT}")


if __name__ == "__main__":
    main()

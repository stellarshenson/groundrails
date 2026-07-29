"""R8-H77 - the unseen arena: RAGBench, which NEITHER model has trained on.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 8).

Every comparison so far has had a home-field problem in one direction or the
other. Our gold favours us - we have seen those documents and the incumbent has
not. RAGTruth is fair but in-domain for BOTH, since each model trained on its
train split. Neither settles which model generalises.

RAGBench settles it. Ten subsets of enterprise-shaped documents - support
tickets, consumer manuals, a car manual, financial filings, biomedical
abstracts, multi-hop wiki - and it appears in NEITHER model's training data:

  - `lettucedect-v2-mmbert-base` trains on RAGTruth, its translations and
    LettuceDetect-prose
  - our student trains on our private gold plus RAGTruth and its translations

So this is the first genuinely blind test for both, and the first number in the
project that measures generalisation rather than specialisation.

Schema maps without reshaping: `response` is the claim, `documents` is already
the evidence list so max-over-chunks applies directly, and `adherence_score` is
the response-level binary the annotators assigned. Labels are GPT-4o, not human -
recorded here because it caps how much the absolute numbers are worth, though it
biases both models identically so the COMPARISON stays sound.

Built to be re-run: `--model <path>` scores any later incarnation through the
identical gate, so successive students are directly comparable rather than each
carrying its own harness.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R8-H77_unseen_arena.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import argparse
import importlib.util
import io
import json
import pathlib
import zipfile

import numpy as np
import polars as pl
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

HERE = pathlib.Path(__file__).parent
ARCHIVE = HERE.parent.parent / "data" / "external" / "datasets" / "dataset-ragbench.zip"
STUDENT = HERE.parent.parent / "models" / "R8-H62-mmbert-multicorpus"
LETTUCE = "KRLabsOrg/lettucedect-v2-mmbert-base"
OUT = HERE / "R8-H77_arena.json"
MAX_CHUNKS = 8
N_PER_SUBSET = 250


def _m59():
    spec = importlib.util.spec_from_file_location("m59", HERE / "R7-H59_cross_domain_matrix.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M59 = _m59()


def load_subsets():
    z = zipfile.ZipFile(ARCHIVE)
    out = {}
    for name in sorted(n for n in z.namelist() if n.endswith("__test.parquet")):
        sub = name.split("__")[2]
        df = pl.read_parquet(io.BytesIO(z.read(name)))
        df = df.filter(
            pl.col("adherence_score").is_not_null()
            & (pl.col("response").str.len_chars() > 20)
            & (pl.col("documents").list.len() > 0)
        )
        if len(df) < 40 or df["adherence_score"].n_unique() < 2:
            continue
        df = df.sample(min(N_PER_SUBSET, len(df)), seed=0)
        out[sub] = (
            df["response"].to_list(),
            [d[:MAX_CHUNKS] for d in df["documents"].to_list()],
            df["adherence_score"].cast(pl.Int8).to_numpy(),
        )
    return out


@torch.inference_mode()
def score_student(path, claims, chunk_lists):
    tok = AutoTokenizer.from_pretrained(str(path))
    model = (
        AutoModelForSequenceClassification.from_pretrained(str(path), dtype=torch.float16)
        .cuda()
        .eval()
    )
    flat_c, flat_k, owner = [], [], []
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        for k in ks:
            flat_c.append(c)
            flat_k.append(k[: M59.CFG.chunk_max_chars])
            owner.append(i)
    s = np.zeros(len(flat_c), dtype=np.float32)
    for i in range(0, len(flat_c), 64):
        enc = tok(
            flat_c[i : i + 64],
            flat_k[i : i + 64],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda")
        s[i : i + 64] = torch.sigmoid(model(**enc).logits.float().squeeze(-1)).cpu().numpy()
    owner = np.array(owner)
    agg = np.array([s[owner == i].max() for i in range(len(claims))])
    del model
    torch.cuda.empty_cache()
    return agg


@torch.inference_mode()
def score_lettuce(claims, chunk_lists):
    tok = AutoTokenizer.from_pretrained(LETTUCE)
    model = (
        AutoModelForTokenClassification.from_pretrained(LETTUCE, dtype=torch.float16).cuda().eval()
    )
    sep = tok.sep_token_id
    agg = np.zeros(len(claims), dtype=np.float32)
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        best = 0.0
        for j in range(0, len(ks), 8):
            batch = [k[: M59.CFG.chunk_max_chars] for k in ks[j : j + 8]]
            enc = tok(
                batch,
                [c] * len(batch),
                truncation="only_first",
                max_length=4096,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            p = torch.softmax(model(**enc).logits.float(), dim=-1)[..., 1]
            ids = enc["input_ids"]
            for r in range(ids.shape[0]):
                row = ids[r].tolist()
                first = row.index(sep) if sep in row else 0
                m = enc["attention_mask"][r].bool().clone()
                m[: first + 1] = False
                q = p[r][m]
                best = max(best, 1.0 - (q.max().item() if q.numel() else 1.0))
        agg[i] = best
    del model
    torch.cuda.empty_cache()
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(STUDENT), help="student checkpoint to score")
    ap.add_argument("--tag", default="R8-H62", help="name for this incarnation in the table")
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    subs = load_subsets()
    print(
        f"RAGBench: {len(subs)} subsets, {sum(len(v[2]) for v in subs.values())} responses\n",
        flush=True,
    )

    rows = {}
    for sub, (claims, chunks, y) in subs.items():
        r = {"n": len(y), "grounded_rate": round(float(y.mean()), 3)}
        for name, fn in (
            (args.tag, lambda c, k: score_student(args.model, c, k)),
            ("lettuce", score_lettuce),
        ):
            s = fn(claims, chunks)
            auc, f1, _ = M59.auc_and_f1(y, s)
            r[f"{name}_auc"], r[f"{name}_f1"] = round(auc, 4), round(f1, 4)
        rows[sub] = r
        print(
            f"  {sub:14s} n={r['n']:>4} base {r['grounded_rate']:.3f}  "
            f"{args.tag} {r[f'{args.tag}_auc']:.4f}  lettuce {r['lettuce_auc']:.4f}  "
            f"delta {r[f'{args.tag}_auc'] - r['lettuce_auc']:+.4f}",
            flush=True,
        )

    ours = float(np.mean([r[f"{args.tag}_auc"] for r in rows.values()]))
    theirs = float(np.mean([r["lettuce_auc"] for r in rows.values()]))
    wins = sum(r[f"{args.tag}_auc"] > r["lettuce_auc"] for r in rows.values())

    print("\n" + "=" * 92)
    print("R8-H77 RESULT - RAGBench, unseen by BOTH models")
    print("=" * 92)
    print(f"  {args.tag:22s} mean AUC {ours:.4f}")
    print(f"  {'lettucedect-v2':22s} mean AUC {theirs:.4f}")
    print(f"  delta {ours - theirs:+.4f}   subsets won {wins}/{len(rows)}")
    print("\n  this is the first blind test for both - neither trained on RAGBench.")
    print("  labels are GPT-4o rather than human, which caps the absolute numbers but")
    print("  biases both models identically, so the COMPARISON stands.")
    OUT.write_text(
        json.dumps(
            {
                "per_subset": rows,
                "mean_ours": ours,
                "mean_lettuce": theirs,
                "wins": wins,
                "model": args.model,
                "tag": args.tag,
            },
            indent=2,
        )
    )
    print(f"\n  results -> {OUT}")


if __name__ == "__main__":
    main()

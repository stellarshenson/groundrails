"""The decomposed-min primary read, generic over checkpoints.

R8-H92 established min-over-sentences of max-over-chunks as the primary blind
read (its pre-registered confirmation, +0.0432 over the incumbent on frozen
weights), and R8-H94 refuted the RAGTruth-tuned soft alternative. This tool
applies that frozen formula to ANY student checkpoint - plain
sequence-classification, DANN (dann_student.pt) or two-head (twohead.pt), via
the R8-H77 scorer's own branches - so successive incarnations get their primary
read through one code path.

Appends the result under `--tag` to R8_decomposed_reads.json.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      uv run python experiments/grounding-semantic/R8_decomposed_read.py \
          --model models/<ckpt> [--tag <TAG>]
"""

import argparse
import importlib.util
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ARENA = _mod("arena", "R8-H77_unseen_arena.py")
H92 = _mod("h92", "R8-H92_decomposed_arena.py")

OUT = HERE / "R8_decomposed_reads.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or pathlib.Path(args.model).name

    subs = ARENA.load_subsets()
    print(f"decomposed-min read: {tag}\n", flush=True)

    rows = {}
    for sub, (claims, chunks, y) in subs.items():
        sent_lists = [H92.sentences(c) for c in claims]
        flat_s, flat_k, owner = [], [], []
        for i, (sl, ks) in enumerate(zip(sent_lists, chunks, strict=True)):
            for s in sl:
                flat_s.append(s)
                flat_k.append(ks)
                owner.append(i)
        owner = np.array(owner)
        scores = ARENA.score_student(args.model, flat_s, flat_k)
        resp = np.array([scores[owner == i].min() for i in range(len(y))])
        auc, f1, _ = ARENA.M59.auc_and_f1(y, resp)
        rows[sub] = {
            "n": len(y),
            "auc": round(auc, 4),
            "f1": round(f1, 4),
            "lettuce_auc": H92.LETTUCE[sub],
        }
        print(
            f"  {sub:14s} n={len(y):>4} {tag} {auc:.4f}  lettuce {H92.LETTUCE[sub]:.4f}  "
            f"delta {auc - H92.LETTUCE[sub]:+.4f}",
            flush=True,
        )

    mean = float(np.mean([r["auc"] for r in rows.values()]))
    let = float(np.mean([r["lettuce_auc"] for r in rows.values()]))
    wins = sum(r["auc"] > r["lettuce_auc"] for r in rows.values())
    print("\n" + "=" * 92)
    print(f"DECOMPOSED-MIN READ - {tag}")
    print("=" * 92)
    print(f"  {tag:22s} mean AUC {mean:.4f}")
    print(f"  {'lettucedect-v2':22s} mean AUC {let:.4f}")
    print(f"  delta {mean - let:+.4f}   subsets won {wins}/{len(rows)}")

    book = json.loads(OUT.read_text()) if OUT.exists() else {}
    book[tag] = {"per_subset": rows, "mean": mean, "wins": wins, "model": args.model}
    OUT.write_text(json.dumps(book, indent=2))
    print(f"\n  results -> {OUT} [{tag}]")


if __name__ == "__main__":
    main()

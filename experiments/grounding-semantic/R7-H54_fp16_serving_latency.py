"""R7-H54 - fair fp16 serving latency for the sub-350M candidates.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 7).

The 5.18 ms/pair reported by R7-H50's eval loop is not a serving number and must
not be quoted as one: it ran fp32 (the `from_pretrained` default) against
teachers timed in fp16, inside a DataLoader, with variable-length padding. This
measures what actually ships instead.

Per the encoder recipe: fp16/bf16 + `attn_implementation="sdpa"` + `torch.compile`
with FIXED input shapes - dynamic shapes recompile and erase the win. Warmup
before timing, median over repeats.

The unit that matters is B=3: the top cfg.semantic_top_k chunks for one claim,
scored as ONE batch. Per-claim latency is what the cascade's 662 ms compares
against, so both are reported.

The bar is the teacher: bge-reranker-v2-m3 at 568M is the model a sub-350M
student has to be faster than, and R7-H50's fp32 numbers suggested it was not.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R7-H54_fp16_serving_latency.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import json
import pathlib
import statistics
import time

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

OUT = pathlib.Path(__file__).parent / "R7-H54_latency.json"
SEQ = 512
BATCHES = (1, 3, 8)  # single pair, the top-3 serving unit, the old top-8 rerank batch
WARMUP, REPS = 12, 40

CANDIDATES = [
    ("bge-reranker-v2-m3 (teacher)", "BAAI/bge-reranker-v2-m3", None),
    ("mmBERT-small", "jhu-clsp/mmBERT-small", None),
    ("mmBERT-base", "jhu-clsp/mmBERT-base", None),
    ("mDeBERTa-v3-base", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", None),
    ("mDeBERTa-minus-6L", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", 6),
]


def truncate_layers(model, keep):
    for attr in ("deberta", "bert", "roberta", "model"):
        base = getattr(model, attr, None)
        if base is not None and hasattr(base, "encoder"):
            base.encoder.layer = torch.nn.ModuleList(list(base.encoder.layer[:keep]))
            model.config.num_hidden_layers = keep
            return model
    raise RuntimeError("could not locate the encoder layer stack")


def depth_width(mid, keep):
    cfg = AutoConfig.from_pretrained(mid)
    return (keep or cfg.num_hidden_layers), cfg.hidden_size


@torch.inference_mode()
def bench(model, batch, dtype, compiled):
    ids = torch.randint(1000, 2000, (batch, SEQ), device="cuda")
    mask = torch.ones_like(ids)
    fn = torch.compile(model, mode="max-autotune") if compiled else model
    for _ in range(WARMUP):
        fn(input_ids=ids, attention_mask=mask)
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn(input_ids=ids, attention_mask=mask)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)
    print(f"seq {SEQ}, warmup {WARMUP}, reps {REPS}, median reported\n", flush=True)
    rows = []
    for name, mid, keep in CANDIDATES:
        L, H = depth_width(mid, keep)
        rec = {"name": name, "layers": L, "hidden": H}
        for dtype, tag in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
            model = AutoModelForSequenceClassification.from_pretrained(
                mid,
                num_labels=1,
                ignore_mismatched_sizes=True,
                dtype=dtype,
                attn_implementation="sdpa",
            )
            if keep:
                model = truncate_layers(model, keep)
            model = model.cuda().eval()
            rec["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
            for b in BATCHES:
                rec[f"{tag}_eager_b{b}"] = round(bench(model, b, dtype, False), 3)
            try:
                rec[f"{tag}_compiled_b3"] = round(bench(model, 3, dtype, True), 3)
            except Exception as e:  # noqa: BLE001 - compile failure is a result, not a crash
                rec[f"{tag}_compiled_b3"] = None
                print(f"    {name} {tag} compile FAILED: {type(e).__name__}", flush=True)
            del model
            torch.cuda.empty_cache()
        rows.append(rec)
        print(
            f"  {name:30s} {rec['params_M']:>6.1f}M  {L:>2}L x {H:<5} "
            f"fp16 b3 {rec['fp16_eager_b3']:>7.3f} ms  "
            f"compiled {rec.get('fp16_compiled_b3')} ms",
            flush=True,
        )

    OUT.write_text(json.dumps(rows, indent=2))
    ref = next(r for r in rows if r["name"].startswith("bge-reranker"))

    print("\n" + "=" * 104)
    print("R7-H54 RESULT - fp16 serving latency, B=3 is the real unit (top-3 chunks, one batch)")
    print("=" * 104)
    print(
        f"{'model':30s} {'params':>8} {'shape':>12} {'fp16 b1':>9} {'fp16 b3':>9} "
        f"{'compiled':>9} {'bf16 b3':>9} {'vs teacher':>11}"
    )
    for r in rows:
        c = r.get("fp16_compiled_b3")
        best = min(x for x in (r["fp16_eager_b3"], c) if x is not None)
        ref_best = min(
            x for x in (ref["fp16_eager_b3"], ref.get("fp16_compiled_b3")) if x is not None
        )
        shape = "{}Lx{}".format(r["layers"], r["hidden"])
        comp = f"{c:.3f}" if c else "FAILED"
        print(
            f"{r['name']:30s} {r['params_M']:>7.1f}M {shape:>12} "
            f"{r['fp16_eager_b1']:>9.3f} {r['fp16_eager_b3']:>9.3f} "
            f"{comp:>9} {r['bf16_eager_b3']:>9.3f} "
            f"{ref_best / best:>10.2f}x"
        )
    print("\n  'vs teacher' > 1 means faster than bge-reranker-v2-m3, the model it must replace")
    print("  depth, not parameter count, sets the critical path - compare the shape column")
    print(f"\n  results -> {OUT}")


if __name__ == "__main__":
    main()

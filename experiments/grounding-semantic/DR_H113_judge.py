"""DR-H113 two-question judge (binding skeptic amendment).

Question (a) contrastive vs seed: did factual content change, and how (delta).
Question (b) vs the PAIRED EVIDENCE: is the REWRITE supported by the evidence?
A sample is judge-agreed only if (a) confirms a factual delta AND (b) says no.

Run (vllm env, GPU1):
  CC=/home/lab/.conda/envs/cudabuild/bin/gcc CXX=/home/lab/.conda/envs/cudabuild/bin/g++ \
  CPATH=/home/lab/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12 \
  VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  VLLM_WSL2_ENABLE_PIN_MEMORY=1 /home/lab/venvs/vllm/bin/python \
  experiments/grounding-semantic/DR_H113_judge.py
"""

import json
import os
import pathlib
import re

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")

import polars as pl  # noqa: E402

HERE = pathlib.Path(__file__).parent
SAMPLES = HERE / "DR_H113_gate_samples.parquet"
OUT = HERE / "DR_H113_gate_judged.parquet"
JUDGE_MODEL = "Qwen/Qwen3-32B-FP8"
DELTAS = ["none", "entity-swap", "number-change", "hedge-deletion", "omission",
          "negation", "other-factual", "degenerate"]

SYSTEM = (
    "You analyse a rewritten sentence against its baseline and against an "
    "evidence passage. Answer with a single JSON object and nothing else."
)
USER_TMPL = (
    "BASELINE: {seed}\n"
    "REWRITE: {rec}\n"
    "EVIDENCE: {chunk}\n\n"
    "Two questions.\n"
    "1. What changed in the REWRITE relative to the BASELINE?\n"
    "2. Is the REWRITE's factual content supported by the EVIDENCE?\n"
    'Reply with ONLY this JSON: {{"delta": D, "severity": S, "changed": C, '
    '"supported": P}}\n'
    'D one of: "none" (same factual content), "entity-swap", "number-change", '
    '"hedge-deletion" (a qualifier was removed/weakened so the claim got '
    'stronger), "omission", "negation", "other-factual", "degenerate" (broken/'
    'garbled).\n'
    'S one of: "none", "subtle", "obvious".\n'
    'C: the exact words IN THE REWRITE carrying the factual change, or "".\n'
    'P: "yes" if the EVIDENCE supports the REWRITE as written, "no" if the '
    "REWRITE contradicts the EVIDENCE or asserts something the EVIDENCE does "
    "not support."
)


def parse_verdict(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if d.get("delta") not in DELTAS or d.get("supported") not in ("yes", "no"):
        return None
    return {"delta": d["delta"], "severity": d.get("severity", "none"),
            "changed": str(d.get("changed", ""))[:400],
            "supported": d["supported"]}


def main():
    df = pl.read_parquet(SAMPLES)
    print(f"judge input: {len(df)} rows", flush=True)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    llm = LLM(model=JUDGE_MODEL, max_model_len=4096, gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0.0, max_tokens=180)

    def prompts_for(rows):
        out = []
        for seed, rec, chunk in rows:
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER_TMPL.format(
                        seed=seed[:900], rec=rec[:900], chunk=chunk[:1600])}]
            out.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False))
        return out

    rows = list(zip(df["seed"].to_list(), df["claim"].to_list(),
                    df["chunk"].to_list(), strict=True))
    verdicts = [None] * len(rows)
    CHUNK = 2000
    for lo in range(0, len(rows), CHUNK):
        hi = min(lo + CHUNK, len(rows))
        outs = llm.generate(prompts_for(rows[lo:hi]), sp)
        for i, o in enumerate(outs):
            verdicts[lo + i] = parse_verdict(o.outputs[0].text)
        bad = [i for i in range(lo, hi) if verdicts[i] is None]
        if bad:
            outs2 = llm.generate(
                [p + "\nJSON ONLY. No prose." for p in
                 prompts_for([rows[i] for i in bad])], sp)
            for j, o in enumerate(outs2):
                verdicts[bad[j]] = parse_verdict(o.outputs[0].text)
        print(f"  judged {hi}/{len(rows)}  parsed "
              f"{sum(v is not None for v in verdicts[:hi])}", flush=True)

    df = df.with_columns(
        pl.Series("delta", [(v or {}).get("delta", "parse_fail") for v in verdicts]),
        pl.Series("severity", [(v or {}).get("severity", "none") for v in verdicts]),
        pl.Series("changed", [(v or {}).get("changed", "") for v in verdicts]),
        pl.Series("supported", [(v or {}).get("supported", "parse_fail")
                                for v in verdicts]),
    )
    df.write_parquet(OUT)
    print(df.group_by("op", "delta").len().sort("op"), flush=True)
    print("=== DR-H113 JUDGED ===", flush=True)


if __name__ == "__main__":
    main()

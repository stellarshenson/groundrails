"""R15-B5 arm 8 step 2 - judge pass over the VitaminC natural-derivation candidates.

P3 measures the arithmetic detector's false-positive floor at 2.58% against a
5.10% real rate: roughly half of every detected derivation is coincidence. The
judge decides, per candidate, whether the claim genuinely reports a quantity
DERIVED from the evidence numbers, or whether the arithmetic equality is an
accident.

  KILL the VitaminC leg if fewer than 150 of 500 candidates verify.

Judge substitution, recorded: the campaign's registered judge is Qwen3-32B-FP8
(DR_judge.py / R10-H111), whose 32 GB of weights do not fit on card 0 (24 GB) or
card 2 (32 GB with a KV cache), and card 1 is carrying H127 training. The judge
here is the largest locally cached instruct model that fits card 2.

Resumable: verdicts checkpoint every chunk; a restart re-judges only the
remainder.

Run (detached, card 2):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
  /home/lab/venvs/vllm/bin/python experiments/grounding-semantic/R15_gate_B5arm8_judge.py
"""

import json
import os
import pathlib

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
# no nvcc on this host, so flashinfer's sampling kernels cannot JIT-build
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import polars as pl  # noqa: E402

HERE = pathlib.Path(__file__).parent
CAND = HERE / "R15_gate_B5arm8_candidates.parquet"
OUT = HERE / "R15_gate_B5arm8_judged.parquet"
CKPT = HERE / "R15_gate_B5arm8_judged.parquet.ckpt"
SUMMARY = HERE / "R15_gate_B5arm8_judge.json"

JUDGE_MODEL = os.environ.get("R15_JUDGE_MODEL", "speakleash/Bielik-11B-v2.3-Instruct")
CHUNK = 100
OP_NAMES = {
    "sum": "the sum of the two evidence numbers",
    "diff_ab": "the difference of the two evidence numbers",
    "diff_ba": "the difference of the two evidence numbers",
    "ratio_ab": "the ratio of the two evidence numbers",
    "ratio_ba": "the ratio of the two evidence numbers",
    "pct_ab": "the percentage change between the two evidence numbers",
    "pct_ba": "the percentage change between the two evidence numbers",
}

SYSTEM = (
    "You decide whether a number stated in a CLAIM is a quantity computed from numbers in the "
    "EVIDENCE, or whether it is about something else and the arithmetic match is a coincidence. "
    "You never judge whether the claim is true. Answer with one word: DERIVED or COINCIDENCE."
)
USER = (
    "EVIDENCE:\n{evidence}\n\n"
    "CLAIM:\n{claim}\n\n"
    "The claim states the number {value}, which does not appear in the evidence. "
    "Arithmetic finds that it equals {op} {a} and {b}.\n"
    "Does the claim actually report that computed quantity - that is, does the number {value} "
    "mean {op} {a} and {b} as the claim uses it? "
    "Answer DERIVED if yes, COINCIDENCE if the number means something else.\n"
    "Answer with one word."
)


def main():
    from vllm import LLM, SamplingParams

    df = pl.read_parquet(CAND).filter(pl.col("source") == "vitaminc").with_row_index("i")
    done = {}
    if CKPT.exists():
        done = {int(k): v for k, v in json.loads(CKPT.read_text()).items()}
    todo = [r for r in df.iter_rows(named=True) if r["i"] not in done]
    print(f"vitaminc candidates {len(df)}   already judged {len(done)}   to judge {len(todo)}",
          flush=True)

    if todo:
        llm = LLM(model=JUDGE_MODEL, dtype="bfloat16", gpu_memory_utilization=0.90,
                  max_model_len=4096, enforce_eager=True)
        sp = SamplingParams(temperature=0.0, max_tokens=6)
        tok = llm.get_tokenizer()

        def prompt(r):
            user = USER.format(evidence=r["evidence"][:2500], claim=r["claim_pos"],
                               value=r["v_correct"], op=OP_NAMES.get(r["op"], "an operation over"),
                               a=r["a"], b=r["b"])
            try:
                return tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                    tokenize=False, add_generation_prompt=True)
            except Exception:  # templates that reject a system turn
                return tok.apply_chat_template(
                    [{"role": "user", "content": SYSTEM + "\n\n" + user}],
                    tokenize=False, add_generation_prompt=True)

        for s in range(0, len(todo), CHUNK):
            batch = todo[s:s + CHUNK]
            prompts = [prompt(r) for r in batch]
            outs = llm.generate(prompts, sp)
            for r, o in zip(batch, outs):
                txt = o.outputs[0].text.strip().upper()
                done[r["i"]] = ("derived" if "DERIV" in txt
                                else ("coincidence" if "COINC" in txt else "parse_fail"))
            CKPT.write_text(json.dumps({str(k): v for k, v in done.items()}))
            print(f"judged {len(done)}/{len(df)}", flush=True)

    verdicts = [done.get(i, "parse_fail") for i in df["i"].to_list()]
    df = df.with_columns([
        pl.Series("judge_verdict", verdicts),
        pl.Series("verified", [v == "derived" for v in verdicts]),
    ])
    df.write_parquet(OUT)

    counts = {v: verdicts.count(v) for v in set(verdicts)}
    summary = {
        "step": "R15-B5 arm 8 step 2 - VitaminC derivation judge pass",
        "judge_model": JUDGE_MODEL,
        "judge_substitution_recorded": "the campaign's registered judge Qwen3-32B-FP8 needs 32 GB "
                                       "of weights and does not fit card 0 (24 GB) or card 2 "
                                       "(32 GB with a KV cache); card 1 is carrying H127 "
                                       "training and is untouched",
        "n_candidates": len(df),
        "counts": counts,
        "n_verified": int(sum(v == "derived" for v in verdicts)),
        "bar": "KILL the VitaminC leg if fewer than 150 of 500 candidates verify",
        "out": OUT.name,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

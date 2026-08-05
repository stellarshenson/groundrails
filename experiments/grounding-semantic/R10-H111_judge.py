"""R10-H111 referee v4 - the CONTRASTIVE judge post-filter (Round 10 amendment).

The final label is contrastive, not absolute: the seed is label-1 by protocol
(guaranteed-clean reference), so the judge sees (baseline, corrupted) side by
side and answers only "did factual content change, and how" - never "is this
grounded". Cascade position: runs on the stage-1 referee-v3 admitted band only.

Judge: Qwen/Qwen3-32B-FP8 via vLLM (in-process, GPU1). gpt-oss-120b was
registered but is not cached locally (60GB download); the 32B-FP8 instruct
checkpoint is cached and the fp8 path is proven on sm_120 - swap recorded.

Post-judge label logic:
  delta none          -> paraphrase band, label 1 (drift rows are re-tagged)
  delta degenerate    -> dropped
  factual delta       -> label 0, THEN the accidental-regrounding filter:
                         if the judge's changed-span appears (normalized) in the
                         seed's evidence chunk, the pair is DROPPED (mislabel risk)

Run (validation):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
  /home/lab/venvs/vllm/bin/python experiments/grounding-semantic/R10-H111_judge.py \
      --n 500 --eyeball R10-H111_judge_eyeball.md
Full pass (armed by the watcher on the STAGE1 DONE marker): same, no --n.
"""

import argparse
import json
import os
import pathlib
import re
import sys

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")

import polars as pl  # noqa: E402

HERE = pathlib.Path(__file__).parent
JUDGE_MODEL = "Qwen/Qwen3-32B-FP8"
DELTAS = [
    "none", "entity-swap", "number-change", "hedge-deletion",
    "omission", "negation", "other-factual", "degenerate",
]

SYSTEM = (
    "You compare two versions of a sentence: a BASELINE and a REWRITE. "
    "Report only what changed between them. You never judge truth or grounding - "
    "only the difference. Answer with a single JSON object and nothing else."
)

USER_TMPL = (
    "BASELINE: {seed}\n"
    "REWRITE: {rec}\n\n"
    "What changed in the REWRITE relative to the BASELINE?\n"
    'Reply with ONLY this JSON: {{"delta": D, "severity": S, "changed": C}}\n'
    'D one of: "none" (same factual content, wording may differ), "entity-swap" '
    '(a name/entity was altered), "number-change" (a number, date, unit or '
    'quantity was altered), "hedge-deletion" (a qualifier/hedge was removed or '
    'weakened so the claim got stronger), "omission" (factual content present in '
    'the BASELINE is missing, changing what is claimed), "negation" (the meaning '
    'was negated or reversed), "other-factual" (factual content changed in '
    'another way), "degenerate" (the REWRITE is broken, garbled, truncated '
    'mid-thought or repetitive junk).\n'
    'S one of: "none", "subtle", "obvious".\n'
    "C: the exact words IN THE REWRITE that carry the factual change, or \"\" "
    "if delta is none or degenerate."
)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.casefold()).replace("  ", " ").strip()


def parse_verdict(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if d.get("delta") not in DELTAS:
        return None
    return {
        "delta": d["delta"],
        "severity": d.get("severity", "none"),
        "changed": str(d.get("changed", ""))[:400],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="R10-H111_stage1_pairs.parquet")
    ap.add_argument("--out", default="R10-H111_stage1_judged.parquet")
    ap.add_argument("--n", type=int, default=0, help="limit rows (0 = all)")
    ap.add_argument("--eyeball", default="", help="write a 50-pair eyeball md")
    ap.add_argument("--checkpoint-every", type=int, default=10000)
    args = ap.parse_args()

    df = pl.read_parquet(HERE / pathlib.Path(args.input).name)
    if args.n:
        df = df.sample(n=min(args.n, len(df)), seed=0)
    print(f"judge input: {len(df)} rows from {args.input}", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    llm = LLM(model=JUDGE_MODEL, max_model_len=4096, gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0.0, max_tokens=160)

    def prompts_for(rows):
        out = []
        for seed, rec in rows:
            msgs = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TMPL.format(
                    seed=seed[:1200], rec=rec[:1200])},
            ]
            out.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False))
        return out

    rows = list(zip(df["seed"].to_list(), df["claim"].to_list(), strict=True))
    verdicts: list = [None] * len(rows)
    CHUNK = 4000
    ckpt_path = HERE / (pathlib.Path(args.out).name + ".ckpt")
    for lo in range(0, len(rows), CHUNK):
        hi = min(lo + CHUNK, len(rows))
        outs = llm.generate(prompts_for(rows[lo:hi]), sp)
        for i, o in enumerate(outs):
            verdicts[lo + i] = parse_verdict(o.outputs[0].text)
        # one strict retry for the chunk's failures
        bad = [i for i in range(lo, hi) if verdicts[i] is None]
        if bad:
            outs2 = llm.generate(
                [p + "\nJSON ONLY. No prose." for p in
                 prompts_for([rows[i] for i in bad])], sp)
            for j, o in enumerate(outs2):
                verdicts[bad[j]] = parse_verdict(o.outputs[0].text)
        done = sum(v is not None for v in verdicts[:hi])
        print(f"  judged {hi}/{len(rows)}  parsed {done}", flush=True)
        if hi % args.checkpoint_every < CHUNK:
            pl.DataFrame({"idx": list(range(hi)),
                          "verdict": [json.dumps(v) for v in verdicts[:hi]]}
                         ).write_parquet(ckpt_path)

    df = df.with_columns(
        pl.Series("delta", [(v or {}).get("delta", "parse_fail") for v in verdicts]),
        pl.Series("severity", [(v or {}).get("severity", "none") for v in verdicts]),
        pl.Series("changed", [(v or {}).get("changed", "") for v in verdicts]),
    )

    # post-judge label logic per the registered amendment
    factual = [d for d in DELTAS if d not in ("none", "degenerate")]
    para = df.filter(pl.col("delta") == "none").with_columns(
        pl.lit(1).alias("label"),
        pl.col("tag").str.replace("ae_drift_", "ae_para_").alias("tag"),
    )
    drift = df.filter(pl.col("delta").is_in(factual))
    # accidental-regrounding filter: changed-span present in evidence -> drop
    keep = []
    for chunk_txt, changed in zip(drift["chunk"].to_list(),
                                  drift["changed"].to_list(), strict=True):
        nc = norm(changed)
        keep.append(not (nc and len(nc) > 3 and nc in norm(chunk_txt)))
    drift = drift.filter(pl.Series(keep)).with_columns(pl.lit(0).alias("label"))
    dropped_reground = keep.count(False)
    out = pl.concat([drift, para])
    out.write_parquet(HERE / pathlib.Path(args.out).name)

    print("\n=== judge stats ===", flush=True)
    print(df.group_by("delta").len().sort("len", descending=True))
    print(f"kept drift (label 0): {len(drift)}  (regrounding-dropped {dropped_reground})")
    print(f"paraphrase band (label 1): {len(para)}")
    print(f"dropped degenerate/parse_fail: "
          f"{len(df.filter(pl.col('delta').is_in(['degenerate', 'parse_fail'])))}")
    # NLI-agreement view: input label 0 rows were NLI-drift, 1 were NLI-para
    for lab in (0, 1):
        sub = df.filter(pl.col("label") == lab)
        if len(sub):
            agree = (sub.filter(pl.col("delta").is_in(factual if lab == 0
                                                      else ["none"])))
            print(f"NLI label {lab}: judge agrees {len(agree)}/{len(sub)} "
                  f"({len(agree)/len(sub):.3f})")

    if args.eyeball:
        import random
        random.seed(0)
        dr = drift.sample(n=min(50, len(drift)), seed=0)
        lines = ["# R10-H111 contrastive-judge eyeball (validation)\n",
                 f"Judge {JUDGE_MODEL}, n={len(df)} judged.\n"]
        for i, r in enumerate(dr.iter_rows(named=True), 1):
            lines += [f"**J{i}** [{r['tag']}] delta={r['delta']} sev={r['severity']}",
                      f"- seed: {r['seed'][:400]}",
                      f"- rec : {r['claim'][:400]}",
                      f"- changed: {r['changed'][:200]}\n"]
        (HERE / pathlib.Path(args.eyeball).name).write_text("\n".join(lines))
        print(f"eyeball -> {args.eyeball}", flush=True)

    print("=== R10-H111 JUDGED ===", flush=True)


if __name__ == "__main__":
    main()

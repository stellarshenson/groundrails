"""DR-2 pilot judge pass - contrastive judge over the DR lane candidates.

Registered cascade (DR-2 registration, semantic-dataset-enhancements.md):
  input = usable rows of DR_pilot_raw (H112+H114 only; H113 DROPPED at its
  pilot bar) + usable rows of DR_pilot_longform
  -> contrastive judge (Qwen3-32B-FP8, temp 0; BASELINE=seed, REWRITE=claim,
     delta-typed - the R10-H111 referee-v4 protocol)
  -> post-judge logic:
       factual delta -> label 0, then accidental-regrounding drop (changed
                        span present normalized in the evidence chunk), then
                        still-entailed veto (nli_fwd >= 0.8 -> drop)
       delta none    -> label-1 paraphrase reclaim iff bidirectional NLI >= 0.8
                        (the H111 reclaim rule); otherwise dropped
       degenerate / parse_fail -> dropped
  -> pooled 50-pair stratified eyeball (engine x long_form strata) for the
     main-session precision read (bar >= 85%, else gpt-oss-120b escalation)

Lane assembly is a SEPARATE step after the eyeball adjudication - this script
ships the judged parquet, the eyeball md, and a summary json only.

Resumable: verdicts checkpoint to <out>.ckpt every chunk; a restart reloads
parsed verdicts and re-judges only the remainder (container restarts are live).

Run (detached, GPU1):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
  /home/lab/venvs/vllm/bin/python experiments/grounding-semantic/DR_judge.py
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
JUDGE_MODEL = "Qwen/Qwen3-32B-FP8"
OUT = HERE / "DR_judged.parquet"
CKPT = HERE / "DR_judged.parquet.ckpt"
EYEBALL = HERE / "DR_judge_eyeball.md"
SUMMARY = HERE / "DR_judge_summary.json"
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
    raw = pl.read_parquet(HERE / "DR_pilot_raw.parquet").filter(
        (pl.col("engine") != "H113") & pl.col("usable"))
    lf = pl.read_parquet(HERE / "DR_pilot_longform.parquet").filter(pl.col("usable"))
    df = pl.concat([raw, lf], how="vertical_relaxed").with_row_index("jidx")
    print(f"judge input: {len(df)} rows "
          f"(sentence H112 {len(raw.filter(pl.col('engine') == 'H112'))}, "
          f"H114 {len(raw.filter(pl.col('engine') == 'H114'))}, "
          f"long-form {len(lf)})", flush=True)

    verdicts: list = [None] * len(df)
    if CKPT.exists():
        ck = pl.read_parquet(CKPT)
        for i, v in zip(ck["idx"].to_list(), ck["verdict"].to_list(), strict=True):
            if v and i < len(verdicts):
                verdicts[i] = json.loads(v) if isinstance(v, str) and v != "null" else None
        done0 = sum(v is not None for v in verdicts)
        print(f"resumed from checkpoint: {done0}/{len(df)} already judged", flush=True)

    todo = [i for i, v in enumerate(verdicts) if v is None]
    if todo:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tok = AutoTokenizer.from_pretrained(JUDGE_MODEL)
        llm = LLM(model=JUDGE_MODEL, max_model_len=4096, gpu_memory_utilization=0.85)
        sp = SamplingParams(temperature=0.0, max_tokens=160)

        seeds = df["seed"].to_list()
        claims = df["claim"].to_list()

        def prompts_for(idxs, suffix=""):
            out = []
            for i in idxs:
                msgs = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER_TMPL.format(
                        seed=seeds[i][:1200], rec=claims[i][:1200]) + suffix},
                ]
                out.append(tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False))
            return out

        CHUNK = 4000
        for lo in range(0, len(todo), CHUNK):
            batch = todo[lo:lo + CHUNK]
            outs = llm.generate(prompts_for(batch), sp)
            for i, o in zip(batch, outs, strict=True):
                verdicts[i] = parse_verdict(o.outputs[0].text)
            bad = [i for i in batch if verdicts[i] is None]
            if bad:
                outs2 = llm.generate(prompts_for(bad, "\nJSON ONLY. No prose."), sp)
                for i, o in zip(bad, outs2, strict=True):
                    verdicts[i] = parse_verdict(o.outputs[0].text)
            done = sum(v is not None for v in verdicts)
            print(f"  judged {min(lo + CHUNK, len(todo))}/{len(todo)} new  "
                  f"total parsed {done}/{len(df)}", flush=True)
            tmp = CKPT.with_suffix(".tmp")
            pl.DataFrame({
                "idx": list(range(len(df))),
                "verdict": [json.dumps(v) if v else None for v in verdicts],
            }).write_parquet(tmp)
            tmp.replace(CKPT)

    df = df.with_columns(
        pl.Series("delta", [(v or {}).get("delta", "parse_fail") for v in verdicts]),
        pl.Series("severity", [(v or {}).get("severity", "none") for v in verdicts]),
        pl.Series("changed", [(v or {}).get("changed", "") for v in verdicts]),
    )

    factual = [d for d in DELTAS if d not in ("none", "degenerate")]
    n_in = len(df)

    # negatives: factual delta -> regrounding drop -> still-entailed veto
    drift = df.filter(pl.col("delta").is_in(factual))
    keep = []
    for chunk_txt, changed in zip(drift["chunk"].to_list(),
                                  drift["changed"].to_list(), strict=True):
        nc = norm(changed)
        keep.append(not (nc and len(nc) > 3 and nc in norm(chunk_txt)))
    n_reground = keep.count(False)
    drift = drift.filter(pl.Series(keep))
    n_veto = len(drift.filter(pl.col("nli_fwd") >= 0.8))
    negatives = drift.filter(pl.col("nli_fwd") < 0.8).with_columns(
        pl.lit(0).alias("label"))

    # positives: judge no-delta AND bidirectional NLI >= 0.8 (H111 reclaim rule)
    para = df.filter(pl.col("delta") == "none")
    reclaim = para.filter(
        (pl.col("nli_fwd") >= 0.8) & (pl.col("nli_bwd") >= 0.8)
    ).with_columns(pl.lit(1).alias("label"))

    out = pl.concat([negatives, reclaim])
    out.write_parquet(OUT)

    stats = {
        "input_rows": n_in,
        "delta_counts": {r["delta"]: r["len"] for r in
                         df.group_by("delta").len().iter_rows(named=True)},
        "regrounding_dropped": n_reground,
        "still_entailed_vetoed": n_veto,
        "certified_negatives": len(negatives),
        "reclaimed_positives": len(reclaim),
        "para_not_reclaimed": len(para) - len(reclaim),
        "by_engine_negatives": {r["engine"]: r["len"] for r in
                                negatives.group_by("engine").len().iter_rows(named=True)},
        "longform_negatives": len(negatives.filter(pl.col("long_form"))),
    }
    SUMMARY.write_text(json.dumps(stats, indent=2))
    print("\n=== judge stats ===", flush=True)
    print(json.dumps(stats, indent=2), flush=True)

    # pooled 50-pair stratified eyeball over certified negatives
    strata = [
        ("H112-sent", negatives.filter((pl.col("engine") == "H112") & ~pl.col("long_form"))),
        ("H112-long", negatives.filter(pl.col("long_form"))),
        ("H114-sent", negatives.filter(pl.col("engine") == "H114")),
    ]
    total = sum(len(s) for _, s in strata) or 1
    lines = ["# DR-2 pilot judge eyeball - 50 certified negatives, stratified\n",
             f"Judge {JUDGE_MODEL}, {n_in} judged, {len(negatives)} certified negatives.",
             "Grade: a pair PASSES if the claim is genuinely unsupported vs the seed "
             "(the corruption changed factual content). Bar: >= 85% pass.\n"]
    k = 1
    for name, s in strata:
        n_take = max(1, round(50 * len(s) / total)) if len(s) else 0
        if not n_take:
            continue
        samp = s.sample(n=min(n_take, len(s)), seed=0)
        for r in samp.iter_rows(named=True):
            lines += [f"**E{k}** [{name}] delta={r['delta']} sev={r['severity']} "
                      f"nli_fwd={r['nli_fwd']:.3f}",
                      f"- seed : {r['seed'][:400]}",
                      f"- claim: {r['claim'][:400]}",
                      f"- changed: {r['changed'][:200]}\n"]
            k += 1
    EYEBALL.write_text("\n".join(lines))
    print(f"eyeball ({k - 1} pairs) -> {EYEBALL.name}", flush=True)
    print("=== DR JUDGED ===", flush=True)


if __name__ == "__main__":
    main()

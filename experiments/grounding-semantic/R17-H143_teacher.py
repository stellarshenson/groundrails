"""R17-H143 Stage A - the CACHED teacher read (Qwen/Qwen3-32B-FP8, GPU1).

The registered teacher (Mistral-Small-24B) has no cached weights, so Stage A
closed no teacher-keyed branch. The ordered completion is a cached Qwen3-32B-FP8
read on GPU1; this is that read.

Every scoring convention is inherited from `R17-H143_stageA.py`, imported as a
module rather than copied: the same INSTRUCTION, the same `chat_think` prompt
the Qwen3-0.6B ran through (its own chat template, enable_thinking=True), the
same capped 512-token think budget, the same forced-choice margin at a FIXED
answer position - the free trace is cut at the end of reasoning, the think block
is CLOSED, the answer is elicited, and the margin is
logsumexp(GROUNDED ids) - logsumexp(UNGROUNDED ids) on the first answer token.
The free-trace greedy parse is recorded as the compliance diagnostic
(parse_fail_rate); scoring never reads off it.

Checkpoints incrementally into the shared `R17-H143_scores.parquet` on the row
identity (model, pair_id, label) - keying on (model, pair_id) drops one label
per pair and leaves an all-positive set (the Stage A lesson).

Positive-control gate: the 50 control pairs are scored first and must read
>= 0.90 AUROC before the real 1,000 are spent.

Run (detached):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
  nohup setsid /home/lab/venvs/vllm/bin/python \
      experiments/grounding-semantic/R17-H143_teacher.py \
      >> logs/R17-H143_teacher.log 2>&1 &
"""

import importlib.util
import json
import os
import pathlib
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")   # NVFP4/FP8 on WSL2
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ["HF_HUB_OFFLINE"] = "1"                          # weights are cached
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

HERE = pathlib.Path(__file__).parent
MODEL = "Qwen/Qwen3-32B-FP8"
STYLE = "chat_think"     # the path the Qwen3-0.6B ran
STEP = 200               # checkpoint granularity on the real set


def _stage_a():
    spec = importlib.util.spec_from_file_location("sa", HERE / "R17-H143_stageA.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SA = _stage_a()


def build_engine(tok):
    from vllm import LLM

    g_ids, u_ids = SA.verdict_token_ids(tok)
    print(f"[audit] teacher tokenizer={type(tok).__name__} "
          f"GROUNDED={g_ids} UNGROUNDED={u_ids}", flush=True)
    SA.audit_prompt(STYLE, tok, MODEL)
    llm = LLM(model=MODEL, max_model_len=4096, gpu_memory_utilization=0.92,
              enforce_eager=False)
    return llm, g_ids, u_ids


def run(llm, tok, g_ids, u_ids, rows: pl.DataFrame, samples: dict) -> pl.DataFrame:
    from vllm import SamplingParams

    recs = rows.to_dicts()
    prompts = [SA.build_prompt(STYLE, tok, r["chunk"], r["claim"]) for r in recs]
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(temperature=0.0,
                                                max_tokens=SA.MAX_NEW_TOKENS))
    traces = [o.outputs[0] for o in outs]

    # forced answer position: cut the free trace at the end of reasoning, CLOSE
    # the think block, elicit the answer - the chat_think discipline verbatim
    close = SA.THINK_CLOSE[STYLE]
    fprompts = []
    for p, comp in zip(prompts, traces, strict=True):
        tail = comp.text.split(close)[0]
        for t in SA.STRIP_TAIL:
            tail = tail.replace(t, "")
        fprompts.append(p + tail + SA.FORCE_SUFFIX[STYLE])
    fouts = llm.generate(fprompts, SamplingParams(temperature=0.0, max_tokens=8,
                                                  logprobs=20))
    per = (time.time() - t0) / len(prompts)

    out_rows = []
    for r, comp, fo in zip(recs, traces, fouts, strict=True):
        fc = fo.outputs[0]
        pieces = tok.batch_decode([[t] for t in fc.token_ids]) if fc.token_ids else []
        v, _ = SA.parse_verdict("".join(pieces), first=True)
        margin, floored = None, False
        if fc.logprobs:
            d = fc.logprobs[0]
            gv = [x.logprob for t, x in d.items() if t in g_ids]
            uv = [x.logprob for t, x in d.items() if t in u_ids]
            floor = min(x.logprob for x in d.values())
            if not gv:
                gv, floored = [floor], True
            if not uv:
                uv, floored = [floor], True
            margin = float(np.logaddexp.reduce(gv) - np.logaddexp.reduce(uv))
        fv, _ = SA.parse_verdict(comp.text)
        out_rows.append(
            dict(model=MODEL, pair_id=r["pair_id"], label=r["label"], control=r["control"],
                 neg_family=r["neg_family"], claim_form=r["claim_form"], margin=margin,
                 margin_floored=floored, verdict=v, free_verdict=fv, parse_fail=fv is None,
                 forced_answer=fv is None, answer_text=fc.text, gen_text=comp.text,
                 n_gen_tokens=len(comp.token_ids), latency_s=per)
        )
    samples.setdefault(MODEL, [])
    if len(samples[MODEL]) < 4:
        samples[MODEL].append(dict(prompt=prompts[0][:900], gen=out_rows[0]["gen_text"][:900]))
    return pl.DataFrame(out_rows)


def main() -> None:
    from transformers import AutoTokenizer

    ev = pl.read_parquet(SA.EVALSET)
    controls = ev.filter(pl.col("control"))
    real = ev.filter(~pl.col("control"))
    samples = SA.load_samples()
    gates = json.loads(SA.GATE_LOG.read_text()) if SA.GATE_LOG.exists() else {}

    tok = AutoTokenizer.from_pretrained(MODEL)
    assert getattr(tok, "is_fast", False), f"tokenizer is {type(tok).__name__}, not fast"
    llm, g_ids, u_ids = build_engine(tok)

    key = pl.col("pair_id") * 2 + pl.col("label")
    done = SA.already(MODEL)
    print(f"\n===== {MODEL} (already scored: {len(done)}) =====", flush=True)

    # --- positive-control gate ------------------------------------------- #
    todo_ctrl = controls.filter(~key.is_in(list(done))) if done else controls
    if todo_ctrl.height:
        t0 = time.time()
        SA.checkpoint(run(llm, tok, g_ids, u_ids, todo_ctrl, samples))
        print(f"[teacher] {todo_ctrl.height} controls in {time.time() - t0:.0f}s", flush=True)

    cd = pl.read_parquet(SA.SCORES).filter(pl.col("model") == MODEL, pl.col("control"))
    m = cd["margin"].to_numpy().astype(float)
    fallback = np.nanmean(np.abs(m)) if np.isfinite(m).any() else 1.0
    sc = np.where(np.isfinite(m), m,
                  np.where(cd["verdict"].to_numpy() == "GROUNDED", fallback,
                           np.where(cd["verdict"].to_numpy() == "UNGROUNDED", -fallback, 0.0)))
    ctrl_auroc = SA.auroc(cd["label"].to_numpy(), sc)
    passed = bool((ctrl_auroc or 0) >= SA.CONTROL_GATE)
    gates.setdefault(MODEL, {}).update(
        control_auroc=ctrl_auroc, gate_exempt=False, passed=passed,
        parse_fail_rate=float(cd["parse_fail"].mean()), quant_path="fp8-native",
    )
    SA.GATE_LOG.write_text(json.dumps(gates, indent=2))
    SA.SAMPLES.write_text(json.dumps(samples, indent=2))
    print(f"[teacher] CONTROL AUROC = {ctrl_auroc} (gate {SA.CONTROL_GATE})", flush=True)
    if not passed:
        raise SystemExit("CONTROL GATE FAILED - fix the harness before trusting the read")

    # --- full eval set ---------------------------------------------------- #
    done = SA.already(MODEL)
    todo = real.filter(~key.is_in(list(done))) if done else real
    print(f"[teacher] scoring {todo.height} real pairs", flush=True)
    for s in range(0, todo.height, STEP):
        part = todo.slice(s, STEP)
        SA.checkpoint(run(llm, tok, g_ids, u_ids, part, samples))
        SA.SAMPLES.write_text(json.dumps(samples, indent=2))
        print(f"[teacher] {min(s + STEP, todo.height)}/{todo.height} checkpointed", flush=True)
    print("[teacher] DONE", flush=True)


if __name__ == "__main__":
    main()

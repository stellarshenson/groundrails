"""R17-H144 Stage B / Job 1 - teacher verdict-trace generation (Qwen3-32B-FP8, GPU1).

Generates a capped think trace plus a forced GROUNDED / UNGROUNDED verdict for
every row of `R17-H144_pairs.parquet`, through the EXACT prompt semantics the
H143 teacher read used: `R17-H143_stageA.py` is imported as a module, so the
INSTRUCTION, the `chat_think` template, the 512-token think budget, the
truncate-at-`</think>` / close / elicit force position and the verdict parser are
the banked ones, not copies.

Rejection filter: a row is ACCEPTED only when the teacher's forced verdict
agrees with the constructed label. The acceptance rate overall and per negative
family is the distillation quality gate (the teacher reads 0.9708 pooled AUROC on
the banked eval, so ~0.95 is expected).

Checkpoints every STEP rows to `R17-H144_traces.parquet` on the row identity
(pair_id, label); a re-run skips what is already there, so a relaunch after a
container restart is the same command.

Run (detached):
  source experiments/grounding-semantic/env_cc.sh
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
  nohup setsid /home/lab/venvs/vllm/bin/python \
      experiments/grounding-semantic/R17-H144_tracegen.py \
      >> logs/R17-H144_tracegen.log 2>&1 &
"""

import importlib.util
import json
import os
import pathlib
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")   # FP8 on WSL2
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ["HF_HUB_OFFLINE"] = "1"                          # weights are cached
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import polars as pl  # noqa: E402

HERE = pathlib.Path(__file__).parent
PAIRS = HERE / "R17-H144_pairs.parquet"
TRACES = HERE / "R17-H144_traces.parquet"
SAMPLES = HERE / "R17-H144_trace_samples.json"

MODEL = "Qwen/Qwen3-32B-FP8"
STYLE = "chat_think"
STEP = 1000              # checkpoint granularity AND vLLM batch size


def _stage_a():
    spec = importlib.util.spec_from_file_location("sa", HERE / "R17-H143_stageA.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SA = _stage_a()


def already() -> set[int]:
    if not TRACES.exists():
        return set()
    d = pl.read_parquet(TRACES)
    return set((d["pair_id"] * 2 + d["label"].cast(pl.Int64)).to_list())


def checkpoint(df: pl.DataFrame) -> None:
    if TRACES.exists():
        df = pl.concat([pl.read_parquet(TRACES), df], how="diagonal_relaxed")
    df.unique(subset=["pair_id", "label"], keep="last").write_parquet(TRACES)


def run(llm, tok, rows: pl.DataFrame) -> pl.DataFrame:
    from vllm import SamplingParams

    recs = rows.to_dicts()
    prompts = [SA.build_prompt(STYLE, tok, r["chunk"], r["claim"]) for r in recs]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0,
                                                max_tokens=SA.MAX_NEW_TOKENS))
    traces = [o.outputs[0] for o in outs]

    close = SA.THINK_CLOSE[STYLE]
    fprompts, thinks, closed = [], [], []
    for p, comp in zip(prompts, traces, strict=True):
        tail = comp.text.split(close)[0]
        for t in SA.STRIP_TAIL:
            tail = tail.replace(t, "")
        closed.append(close in comp.text)
        thinks.append(tail)
        fprompts.append(p + tail + SA.FORCE_SUFFIX[STYLE])
    fouts = llm.generate(fprompts, SamplingParams(temperature=0.0, max_tokens=8))

    out_rows = []
    for r, comp, fo, think, cl in zip(recs, traces, fouts, thinks, closed, strict=True):
        v, _ = SA.parse_verdict(fo.outputs[0].text, first=True)
        fv, _ = SA.parse_verdict(comp.text)
        want = "GROUNDED" if int(r["label"]) == 1 else "UNGROUNDED"
        out_rows.append(dict(
            pair_id=r["pair_id"], label=int(r["label"]), neg_family=r["neg_family"],
            doc_id=r["doc_id"], dtype=r["dtype"], think=think, think_closed=cl,
            n_think_tokens=len(tok.encode(think, add_special_tokens=False)),
            n_gen_tokens=len(comp.token_ids), verdict=v, free_verdict=fv,
            parse_fail=fv is None, accepted=v == want,
        ))
    return pl.DataFrame(out_rows)


def main() -> None:
    from transformers import AutoTokenizer
    from vllm import LLM

    pairs = pl.read_parquet(PAIRS)
    tok = AutoTokenizer.from_pretrained(MODEL)
    assert getattr(tok, "is_fast", False), f"tokenizer is {type(tok).__name__}, not fast"
    SA.audit_prompt(STYLE, tok, MODEL)
    llm = LLM(model=MODEL, max_model_len=4096, gpu_memory_utilization=0.92,
              enforce_eager=False)

    done = already()
    key = pl.col("pair_id") * 2 + pl.col("label").cast(pl.Int64)
    todo = pairs.filter(~key.is_in(list(done))) if done else pairs
    print(f"[tracegen] {len(done)} rows already on disk; {todo.height} to generate",
          flush=True)

    t_start = time.time()
    for s in range(0, todo.height, STEP):
        part = todo.slice(s, STEP)
        t0 = time.time()
        df = run(llm, tok, part)
        checkpoint(df)
        done_n = min(s + STEP, todo.height)
        rate = df["accepted"].mean()
        el = time.time() - t_start
        eta = el / done_n * (todo.height - done_n)
        print(f"[tracegen] {done_n}/{todo.height} checkpointed "
              f"({time.time() - t0:.0f}s, batch accept {rate:.4f}, eta {eta / 60:.0f}m)",
              flush=True)
        if not SAMPLES.exists():
            SAMPLES.write_text(json.dumps(dict(
                prompt=SA.build_prompt(STYLE, tok, part["chunk"][0], part["claim"][0])[:1200],
                think=df["think"][0][:1500], verdict=df["verdict"][0],
                label=int(df["label"][0]),
            ), indent=2))

    d = pl.read_parquet(TRACES)
    print(f"[tracegen] DONE n={d.height} accept={d['accepted'].mean():.4f}", flush=True)
    print(d.group_by("neg_family").agg(
        pl.len().alias("n"), pl.col("accepted").mean().alias("accept")
    ).sort("neg_family"), flush=True)


if __name__ == "__main__":
    main()

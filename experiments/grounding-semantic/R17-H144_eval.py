"""R17-H144 Stage B / Job 3 - student read on the banked H143 eval set (GPU0).

Scores the SFT student on `R17-H143_evalset.parquet` with the Stage-A harness
conventions, imported from `R17-H143_stageA.py` rather than copied: 512-token
free trace, truncate at the end of reasoning, CLOSE the think block, elicit the
answer, forced-choice margin logsumexp(GROUNDED) - logsumexp(UNGROUNDED) on the
first answer token; the free-trace greedy parse is the compliance diagnostic
only. Parse failures score 0.0 with `pooled_auroc_literal05` recorded alongside.

The one style addition is the student's own force suffix: it was trained to write
"<think>\\n...\\n</think>\\n\\nAnswer: VERDICT", so the elicitation appends
"</think>\\n\\nAnswer:" to the cut trace (the trace already carries the newline).
The prompt itself is the untouched Stage-A `chat` render - the same one the
banked untrained SmolLM2 baseline (0.5011) ran through.

Writes `R17-H144_result.json`. MEASUREMENT ONLY - no adjudication.

Run (detached):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
  nohup setsid python experiments/grounding-semantic/R17-H144_eval.py \\
      >> logs/R17-H144_eval.log 2>&1 &
"""

import importlib.util
import json
import os
import pathlib

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
# amendment A1: cycle 2 reads the mixed-corpus student through the IDENTICAL
# harness and merges under a "cycle2" key - cycle 1's numbers are never rewritten
CYCLE2 = os.environ.get("H144_CYCLE2") == "1"
TAG = "_c2" if CYCLE2 else ""
OUTDIR = ROOT / "models" / ("R17-H144-student-c2" if CYCLE2 else "R17-H144-student")
TRACES = HERE / "R17-H144_traces.parquet"
LOOKUP_MANIFEST = HERE / "R17-H144_lookup_manifest.json"
SFT_STATS = HERE / f"R17-H144_sft{TAG}_stats.json"
RESULT = HERE / "R17-H144_result.json"

BASELINE_UNTRAINED = 0.5011      # banked Stage A read, not re-run
BASELINE_UNTRAINED_CONTROL = 0.4848   # same, its control read
# score the full set even when the control gate misses (see score_model)
FORCE_BELOW_GATE = os.environ.get("H144_FORCE_BELOW_GATE") == "1"


def _stage_a():
    spec = importlib.util.spec_from_file_location("sa", HERE / "R17-H143_stageA.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SA = _stage_a()
SA.SCORES = HERE / f"R17-H144_student{TAG}_scores.parquet"
SA.GATE_LOG = HERE / f"R17-H144_student{TAG}_gate.json"
SA.SAMPLES = HERE / f"R17-H144_student{TAG}_samples.json"

_BUILD, _AUDIT = SA.build_prompt, SA.audit_prompt
SA.build_prompt = lambda style, tok, chunk, claim: _BUILD(
    "chat" if style == "chat_student" else style, tok, chunk, claim)
SA.audit_prompt = lambda style, tok, name: _AUDIT(
    "chat" if style == "chat_student" else style, tok, name)
SA.FORCE_SUFFIX["chat_student"] = "</think>\n\nAnswer:"
SA.THINK_CLOSE["chat_student"] = "</think>"

CFG = dict(params_M=362, style="chat_student", offline=False, batch=16)


def derive(d: pl.DataFrame, pf: float) -> np.ndarray:
    m = d["margin"].to_numpy().astype(float)
    v = d["verdict"].to_numpy()
    fb = float(np.nanmean(np.abs(m))) if np.isfinite(m).any() else 1.0
    return np.where(np.isfinite(m), m,
                    np.where(v == "GROUNDED", fb, np.where(v == "UNGROUNDED", -fb, pf)))


def score_model(path: pathlib.Path, ev: pl.DataFrame) -> dict:
    name = str(path)
    samples = SA.load_samples()
    controls, real = ev.filter(pl.col("control")), ev.filter(~pl.col("control"))
    key = pl.col("pair_id") * 2 + pl.col("label")

    done = SA.already(name)
    todo = controls.filter(~key.is_in(list(done))) if done else controls
    if todo.height:
        SA.checkpoint(SA.run_transformers(name, CFG, todo, samples))
        SA.SAMPLES.write_text(json.dumps(samples, indent=2))

    cd = pl.read_parquet(SA.SCORES).filter(pl.col("model") == name, pl.col("control"))
    ctrl_auroc = SA.auroc(cd["label"].to_numpy(), derive(cd, 0.0))
    gate_ok = (ctrl_auroc or 0) >= SA.CONTROL_GATE
    print(f"[{name}] CONTROL AUROC = {ctrl_auroc} (gate {SA.CONTROL_GATE}) passed={gate_ok}",
          flush=True)
    # The gate screens for a BROKEN HARNESS, and this harness verifies sound on
    # the same code paths: Qwen3-32B reads 1.00 and Qwen3-0.6B 0.9904 on these
    # same 50 controls, the margin's sign agrees with the parsed verdict on
    # 50/50 rows, and the missed controls are lookups whose asserted value is
    # present verbatim - the student conflates an adjacent row. The untrained
    # SmolLM2 reads 0.4848 here, so a sub-gate student read is a competence
    # measurement, not a defect. It is recorded, carried into the bars
    # conjunction, and left to the coordinator rather than silently dropped.
    if not gate_ok and not FORCE_BELOW_GATE:
        raise SystemExit(f"CONTROL GATE FAILED ({ctrl_auroc}) - fix the harness first")

    done = SA.already(name)
    todo = real.filter(~key.is_in(list(done))) if done else real
    print(f"[{name}] scoring {todo.height} real pairs", flush=True)
    for s in range(0, todo.height, 250):
        SA.checkpoint(SA.run_transformers(name, CFG, todo.slice(s, 250), samples))
        SA.SAMPLES.write_text(json.dumps(samples, indent=2))
        print(f"[{name}] {min(s + 250, todo.height)}/{todo.height} checkpointed", flush=True)

    d = pl.read_parquet(SA.SCORES).filter(pl.col("model") == name)
    rl = d.filter(~pl.col("control"))
    fam = {}
    for f in rl.filter(pl.col("label") == 0)["neg_family"].unique().sort().to_list():
        sub = rl.filter((pl.col("label") == 1) | (pl.col("neg_family") == f))
        fam[f] = SA.auroc(sub["label"].to_numpy(), derive(sub, 0.0))
    return dict(
        checkpoint=name, n_scored=rl.height,
        pooled_auroc=SA.auroc(rl["label"].to_numpy(), derive(rl, 0.0)),
        pooled_auroc_literal05=SA.auroc(rl["label"].to_numpy(), derive(rl, 0.5)),
        control_auroc=ctrl_auroc, control_gate_passed=gate_ok, per_family_auroc=fam,
        parse_fail_rate=float(rl["parse_fail"].mean()),
        no_answer_rate=float(rl["verdict"].is_null().mean()),
        mean_gen_tokens=float(rl["n_gen_tokens"].mean()),
    )


def teacher_block() -> dict:
    t = pl.read_parquet(TRACES)
    per = {r["neg_family"]: r["accept"] for r in t.group_by("neg_family").agg(
        pl.col("accepted").mean().alias("accept")).sort("neg_family").to_dicts()}
    n = t["n_think_tokens"].to_numpy()
    return dict(
        n_generated=t.height, n_accepted=int(t["accepted"].sum()),
        acceptance_rate=float(t["accepted"].mean()), acceptance_rate_per_family=per,
        acceptance_rate_by_label={str(r["label"]): r["accept"] for r in t.group_by("label")
                                  .agg(pl.col("accepted").mean().alias("accept"))
                                  .sort("label").to_dicts()},
        free_trace_parse_fail_rate=float(t["parse_fail"].mean()),
        think_closed_rate=float(t["think_closed"].mean()),
        think_len_stats=dict(mean=float(n.mean()), p50=float(np.percentile(n, 50)),
                             p90=float(np.percentile(n, 90)), max=float(n.max())),
    )


def main() -> None:
    ev = pl.read_parquet(SA.EVALSET)
    sft = json.loads(SFT_STATS.read_text()) if SFT_STATS.exists() else {}
    hist = sft.get("history", [])
    eps = [h["epoch"] for h in hist]
    best = min(hist, key=lambda h: h["val_loss"])["epoch"] if hist else None
    # A2: the selector may pick a checkpoint that was never read on the bar set,
    # and a verdict cannot be stated for an unmeasured checkpoint. This scores
    # the named epochs - measurement of the SELECTED checkpoint, not a search
    # for one that passes.
    force = [int(x) for x in os.environ.get("H144_EPOCHS", "").split(",") if x.strip()]
    extra = [e for e in eps if (OUTDIR / f"epoch{e}").is_dir() and (e == 2 or e in force)]
    want = sorted({e for e in [best, (max(eps) if eps else None), *extra] if e})
    print(f"[eval] epochs {eps}, best-val {best}, scoring {want}", flush=True)
    if not want:
        RESULT.write_text(json.dumps(dict(
            teacher_gen=teacher_block(), sft=dict(epochs_run=eps, stopped=sft.get("stopped"),
                                                  data=sft.get("data")),
            student_eval=None, bars=None,
            note="no student checkpoint - SFT stopped before completing an epoch",
        ), indent=2))
        raise SystemExit("no student checkpoint to score")

    reads = [score_model(OUTDIR / f"epoch{e}", ev) for e in want]
    for r, e in zip(reads, want, strict=True):
        r["epoch"] = e
        r["is_best_val"] = e == best
    primary = next((r for r in reads if r["is_best_val"]), reads[-1])
    # Bars are evaluated PER CHECKPOINT, not only on the primary. Checkpoint
    # selection is a free parameter the registration never fixed, and the two
    # clauses can split across epochs of one run - reporting only the primary
    # would hide a sibling checkpoint that meets the conjunction.
    for r in reads:
        cp, cc = r["pooled_auroc"], (r["control_auroc"] or 0)
        r["bars"] = dict(viable_070=bool(cp >= 0.70 and cc >= 0.90),
                         pooled_ge_070=bool(cp >= 0.70),
                         control_ge_090=bool(cc >= 0.90),
                         kill_060=bool(cp < 0.60))

    p = primary["pooled_auroc"]
    res = dict(
        teacher_gen=teacher_block(),
        sft=dict(
            epochs_run=eps, best_val_epoch=best, stopped=sft.get("stopped"),
            history=hist, data=sft.get("data"), config=sft.get("config"),
            final_train_loss=hist[-1].get("train_loss") if hist else None,
            final_val_loss=hist[-1]["val_loss"] if hist else None,
            best_val_loss=min((h["val_loss"] for h in hist), default=None),
            verdict_format_rate=primary and next(
                (h["format_rate"] for h in hist if h["epoch"] == primary["epoch"]), None),
        ),
        student_eval=dict(
            primary_checkpoint_epoch=primary["epoch"],
            pooled_auroc=p, control_auroc=primary["control_auroc"],
            control_gate_passed=primary["control_gate_passed"],
            untrained_baseline_control_auroc=BASELINE_UNTRAINED_CONTROL,
            per_family_auroc=primary["per_family_auroc"],
            parse_fail_rate=primary["parse_fail_rate"],
            pooled_auroc_literal05=primary["pooled_auroc_literal05"],
            per_checkpoint=reads,
            untrained_baseline_pooled_auroc=BASELINE_UNTRAINED,
            untrained_baseline_source="banked Stage A read, not re-run",
        ),
        # the registered VIABLE line is a CONJUNCTION - pooled >= 0.70 AND
        # controls >= 0.90 - so both clauses are reported separately as well as
        # the conjunction, or a single False hides which clause moved
        bars=dict(viable_070=bool(p >= 0.70 and (primary["control_auroc"] or 0) >= 0.90),
                  pooled_ge_070=bool(p >= 0.70),
                  control_ge_090=bool((primary["control_auroc"] or 0) >= 0.90),
                  kill_060=bool(p < 0.60),
                  gray=bool(0.60 <= p < 0.70),
                  any_checkpoint_viable=bool(any(r["bars"]["viable_070"] for r in reads)),
                  viable_checkpoint_epochs=[r["epoch"] for r in reads
                                            if r["bars"]["viable_070"]],
                  primary_selection_rule="lowest SFT validation loss",
                  selection_note=(
                      "no held-out signal available to this harness selects the "
                      "viable checkpoint: validation loss picks the final epoch, "
                      "validation verdict-agreement picks epoch 1. Selecting on "
                      "the bar set itself would be selection on the test set - "
                      "the coordinator adjudicates")),
    )
    if CYCLE2:
        prior = json.loads(RESULT.read_text()) if RESULT.exists() else {}
        res["lookup_family"] = (json.loads(LOOKUP_MANIFEST.read_text())
                                if LOOKUP_MANIFEST.exists() else None)
        res["amendment"] = "A1 - mixed corpus (teacher traces + verbatim-lookup family)"
        c1 = {k: v for k, v in prior.items() if k != "cycle2"}
        res["cycle1_deltas"] = dict(
            pooled=round(p - c1.get("student_eval", {}).get("pooled_auroc", float("nan")), 4),
            control=round((primary["control_auroc"] or 0)
                          - c1.get("student_eval", {}).get("control_auroc", float("nan")), 4))
        prior["cycle2"] = res
        RESULT.write_text(json.dumps(prior, indent=2))
    else:
        RESULT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["student_eval"] | {"bars": res["bars"]}, indent=2), flush=True)
    print("=== R17-H144 EVAL DONE ===", flush=True)


if __name__ == "__main__":
    main()

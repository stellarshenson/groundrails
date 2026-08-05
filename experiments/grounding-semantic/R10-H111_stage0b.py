"""R10-H111 stage 0b - referee upgrade: degeneracy gate over the dropout dial.

The stage-0 fluency referee (gpt2 NLL) is blind to degenerate repetition
("the status of the status of", "drive drive drive") because repetition is
high-probability under a causal LM. The main-session eyeball adjudication found
~60-70% of the p=0.2 admitted drift is repetition junk while paraphrase-mislabel
is ~zero. This re-run adds a DEGENERACY GATE (distinct-3gram ratio + max
same-token run, thresholds calibrated on the p=0.05 reconstruction distribution,
same protocol as the NLL p95 threshold) and re-measures the honest fluent-drift
band on p in {0.10, 0.15, 0.20} over the SAME 2,868 seeds. p=0.05 is generated
only to calibrate the referee thresholds. All reconstructions are saved this
time. Adjudication is external.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/R10-H111_stage0b.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import importlib.util
import json
import pathlib
import random

import numpy as np
import polars as pl
import torch

HERE = pathlib.Path(__file__).parent
OUT_JSON = HERE / "R10-H111_stage0b_result.json"
OUT_EYE = HERE / "R10-H111_eyeball2.md"
OUT_PARQ = HERE / "R10-H111_stage0b_recons.parquet"

CAL_P = 0.05  # calibration-only reconstructions (thresholds), not a candidate band
P_TARGET = [0.1, 0.15, 0.2]
SEED = 0
DEGEN_D3_PCTL = 5  # fail if distinct-3gram ratio < p5 of the p=0.05 distribution
DEGEN_RUN_PCTL = 95  # fail if max same-token run > p95 of the p=0.05 distribution
FLUENCY_PCTL = 95  # unchanged from stage 0

spec = importlib.util.spec_from_file_location("s0", HERE / "R10-H111_stage0.py")
S0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S0)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

log = S0.log


def degeneracy(text):
    """(distinct-3gram ratio, max same-token run) on lowercased whitespace tokens."""
    toks = text.lower().split()
    if len(toks) < 3:
        return 1.0, 1
    grams = [tuple(toks[i : i + 3]) for i in range(len(toks) - 2)]
    d3 = len(set(grams)) / len(grams)
    run, best = 1, 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return d3, best


def main():
    seeds = S0.load_seeds()
    registers = [r for r, _ in seeds]
    texts = [t for _, t in seeds]

    tok, model = S0.load_mbart()

    recs = {}
    for p in [CAL_P] + P_TARGET:
        log(f"sweep p={p} ...")
        torch.manual_seed(SEED + int(p * 100))  # same per-p seeds as stage 0
        recs[p] = S0.reconstruct(tok, model, texts, p=p, train_mode=True)
    del model
    torch.cuda.empty_cache()

    # referee thresholds, all calibrated on the p=0.05 reconstruction distribution
    log("fluency referee (gpt2) ...")
    all_nll = {p: S0.gpt2_nll(recs[p]) for p in [CAL_P] + P_TARGET}
    nll_thresh = float(np.percentile(all_nll[CAL_P], FLUENCY_PCTL))
    cal_deg = [degeneracy(t) for t in recs[CAL_P]]
    d3_thresh = float(np.percentile([d for d, _ in cal_deg], DEGEN_D3_PCTL))
    run_thresh = float(np.percentile([r for _, r in cal_deg], DEGEN_RUN_PCTL))
    log(
        f"thresholds: nll<= {nll_thresh:.3f} (p{FLUENCY_PCTL}), "
        f"distinct3 >= {d3_thresh:.3f} (p{DEGEN_D3_PCTL}), "
        f"maxrun <= {run_thresh:.1f} (p{DEGEN_RUN_PCTL}) - all from p={CAL_P} recons"
    )

    comp, per_reg, detail, rows = {}, {}, {}, []
    for p in P_TARGET:
        log(f"nli referee p={p} ...")
        fwd_am, fwd_pe = S0.nli_entail(list(zip(texts, recs[p])))
        bwd_am, bwd_pe = S0.nli_entail(list(zip(recs[p], texts)))
        cls, minent = [], []
        n_degen = 0
        for i in range(len(texts)):
            d3, mrun = degeneracy(recs[p][i])
            degen = d3 < d3_thresh or mrun > run_thresh
            fluent = all_nll[p][i] <= nll_thresh
            exact = recs[p][i].lower() == texts[i].lower()
            para = (fwd_am[i] and bwd_am[i]) or exact
            if degen:
                cls.append("noise")  # degeneracy -> NOISE regardless of NLL
                n_degen += 1
            elif para and fluent:
                cls.append("paraphrase")
            elif fluent:
                cls.append("drift")
            else:
                cls.append("noise")
            minent.append(min(fwd_pe[i], bwd_pe[i]))
            rows.append(
                {
                    "seed": texts[i],
                    "register": registers[i],
                    "p": p,
                    "reconstruction": recs[p][i],
                    "nll": all_nll[p][i],
                    "distinct3": d3,
                    "maxrun": mrun,
                    "nli_fwd": fwd_pe[i],
                    "nli_bwd": bwd_pe[i],
                    "verdict": cls[-1],
                }
            )
        n = len(cls)
        comp[p] = {
            "paraphrase": round(cls.count("paraphrase") / n, 4),
            "drift": round(cls.count("drift") / n, 4),
            "noise": round(cls.count("noise") / n, 4),
            "degenerate_share": round(n_degen / n, 4),
            "n": n,
        }
        for reg in ("procedural", "quantitative", "scientific"):
            idx = [i for i in range(n) if registers[i] == reg]
            per_reg.setdefault(reg, {})[p] = {
                "paraphrase": round(sum(cls[i] == "paraphrase" for i in idx) / len(idx), 4),
                "drift": round(sum(cls[i] == "drift" for i in idx) / len(idx), 4),
                "noise": round(sum(cls[i] == "noise" for i in idx) / len(idx), 4),
            }
        detail[p] = {"cls": cls, "minent": minent}
        log(f"  p={p}: {comp[p]}")

    # calibration rows go into the parquet too (verdict left empty - no NLI run)
    for i in range(len(texts)):
        d3, mrun = degeneracy(recs[CAL_P][i])
        rows.append(
            {
                "seed": texts[i],
                "register": registers[i],
                "p": CAL_P,
                "reconstruction": recs[CAL_P][i],
                "nll": all_nll[CAL_P][i],
                "distinct3": d3,
                "maxrun": mrun,
                "nli_fwd": None,
                "nli_bwd": None,
                "verdict": "calibration",
            }
        )
    pl.DataFrame(rows).write_parquet(OUT_PARQ)

    best_p = max(P_TARGET, key=lambda p: comp[p]["drift"])
    cls, minent = detail[best_p]["cls"], detail[best_p]["minent"]
    drift_idx = [i for i in range(len(texts)) if cls[i] == "drift"]
    para_idx = sorted(
        [i for i in range(len(texts)) if cls[i] == "paraphrase"], key=lambda i: minent[i]
    )
    rng = random.Random(SEED)
    pick_drift = rng.sample(drift_idx, min(50, len(drift_idx)))
    pick_border = para_idx[:25]
    lines = [
        "# R10-H111 stage 0b - eyeball sample (degeneracy-gated referee)",
        "",
        f"Model facebook/mbart-large-50, best_p {best_p}. Thresholds: nll <= {nll_thresh:.3f}, "
        f"distinct3 >= {d3_thresh:.3f}, maxrun <= {run_thresh:.1f} (all from p={CAL_P} recons).",
        "Adjudication bar: < 1 in 10 of the ADMITTED DRIFT pairs below is a meaning-preserving",
        "paraphrase (and degenerate repetition should now be absent - flag any that slipped).",
        "",
        f"## Admitted drift at p={best_p} ({len(pick_drift)} random)",
        "",
    ]
    for k, i in enumerate(pick_drift):
        lines += [
            f"**D{k + 1}** [{registers[i]}] min-entail {minent[i]:.2f}",
            f"- seed: {texts[i]}",
            f"- rec : {recs[best_p][i]}",
            "",
        ]
    lines += ["## Borderline paraphrase (25 lowest min-entailment among admitted paraphrases)", ""]
    for k, i in enumerate(pick_border):
        lines += [
            f"**B{k + 1}** [{registers[i]}] min-entail {minent[i]:.2f}",
            f"- seed: {texts[i]}",
            f"- rec : {recs[best_p][i]}",
            "",
        ]
    OUT_EYE.write_text("\n".join(lines))

    result = {
        "model_used": "facebook/mbart-large-50",
        "dropout_mechanism": "unchanged from stage 0 (model.train() + nn.Dropout modules + "
        "float dropout attrs, greedy decode); same per-p torch seeds as stage 0",
        "fluency_referee": f"gpt2 length-normalized NLL <= {nll_thresh:.3f} "
        f"(p{FLUENCY_PCTL} of p={CAL_P})",
        "degeneracy_referee": {
            "distinct3_min": d3_thresh,
            "maxrun_max": run_thresh,
            "calibration": f"p{DEGEN_D3_PCTL} / p{DEGEN_RUN_PCTL} of the p={CAL_P} "
            "reconstruction distribution; failing pair -> noise regardless of NLL",
        },
        "nli_referee": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli, both directions, argmax entailment",
        "per_p": comp,
        "per_register": per_reg,
        "best_p": best_p,
        "drift_yield_at_best_p": comp[best_p]["drift"],
        "examples_path": str(OUT_EYE),
        "recons_path": str(OUT_PARQ),
    }
    OUT_JSON.write_text(json.dumps(result, indent=1))
    log(json.dumps({"per_p": comp, "best_p": best_p}, indent=1))
    log(f"results -> {OUT_JSON}")
    log("=== R10-H111 STAGE0B DONE ===")


if __name__ == "__main__":
    main()
